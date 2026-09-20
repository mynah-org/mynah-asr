# CPU serving — the operator's guide

**Production is Linux x86-64 and ARM64.** Every decision in this guide is taken
there and every number that promotes comes from there. macOS builds and runs,
and its numbers are development signals: useful for catching a regression while
writing code, never a capacity claim and never an input to a configuration.

This page is the one to read before serving this engine on a box. It is not an
architecture document (`docs/serving.md` is the vocabulary and the method,
`docs/server.md` is the wire protocol and the admission ladder); it is what to
run, in what order, and what to set.

---

## 1. The shape of the problem, in one line

A streaming ASR server is not a throughput engine. **The client paces the
audio**, so the deadline is a wall clock: each stream delivers one chunk every

```
P = (lookahead + 1) x encoder_frame_ms
```

— 320 ms for Nemotron at its default lookahead 3 — and a worker must finish one
batched step for all of its streams inside `P`, with headroom. The step cost is
affine in the number of ready streams:

```
T_step(B) = a + b*B                    B_max = (rho*P - a) / b
```

| | what it is | why it dominates the topology |
|---|---|---|
| `a` | the cost of **walking the model once**: every weight read, every layer visited — paid whether the worker serves 1 stream or 50 | **every worker re-pays it.** W workers spend `W*a` of each period on fixed cost before serving anyone |
| `b` | what one more stream costs **while those weights are already being walked** | small, because the batched step stacks the ready streams' rows into one pass |

Two consequences drive everything below, and both are falsifiable:

- **Lookahead is capacity.** Raising it buys streams with latency the speaker
  feels instead of with hardware.
- **Start WIDE.** Capacity is `W * (rho*P - a) / b`, maximised as `W` shrinks —
  until one worker can no longer keep its threads busy. If a `W` sweep finds
  capacity flat, the model is wrong and the advice changes.

---

## 2. Three tools, three verbs

Run them in this order. They are separate because they answer different
questions and have different rights to be believed.

```sh
make box-doctor                                   # 1. DESCRIBE
make box-advise MODEL_DIR=models/<pack>           # 2. PREDICT
tools/bench/box_qualify.sh -m models/<pack> ...   # 3. MEASURE
```

### 1. `box-doctor` — describe the machine, and refuse when it is not fit

Reads CPU model, SMT siblings, the core-major order, caches, the ISA features
**this** runtime uses, what `--dispatch-map` says the binary will actually do,
memory, the noise sources that invalidate a measurement, and the limits that
bite at high fan-in. It proposes candidate `W x T` topologies and decides
nothing.

It **refuses** (exit 3) when `loadavg >= 2.0`, using the same threshold as the
qualifier so a box the doctor calls idle is a box the qualifier accepts. Every
fact names its source; anything it cannot read prints `UNAVAILABLE <fact> —
<why>` rather than a guess.

### 2. `box-advise` — predict capacity, recommend a configuration

**It calibrates on the machine in front of it.** It runs the real step table
(`tests/test_stream_batch` with `MYNAH_ASR_STEP_TIME=1` — the serving path, not
a proxy) at B = 1..8 and fits `a` and `b`, discarding B=1 because the first step
pays the first touch of the weights. Then it prints capacity per lookahead and a
pasteable command line.

Every figure carries its provenance:

| label | meaning |
|---|---|
| `[MEASURED]` | read on this machine, in this run |
| `[CALIBRATED]` | fitted from this machine's own step table |
| `[EXTRAPOLATED]` | the fit evaluated outside the range that produced it (past B=8) |
| `[PREDICTED]` | config plus a constant carried in from another machine |
| `[UNKNOWN]` | nothing supports a number, and it says so |

It refuses rather than guessing in three places: on an `UNKNOWN` dispatch row (a
capacity attributed to a kernel nobody resolved is fiction), on a loaded box,
and when the only cost model available came from a machine of a different size —
both `a` and `b` are functions of how many cores walk the weights, so carrying
a 32-core pair onto an 8-core box describes neither.

### 3. `box_qualify.sh` — measure

A **WAVE** screens and may disqualify; **only a SOAK promotes**: fixed
concurrency, ≥ 10 minutes, mixed-length bank, drift gate. The harness refuses to
print cadence percentiles when the client could not pace at 1x.

---

## 3. What to set, and why

### Quantisation — `int8`, and it is the serving default

```sh
./mynah-asr-server -m <pack> --quant int8      # the default; --quant f32 to opt out
```

int8 is **2.24x faster** than f32 (one stream, explicit language) for a
documented mean CER of 0.133 → 0.145 over 34 locales, identical on ~90 % of them
(`docs/quantization.md`). On a CPU box that trade is not close. The CLI keeps
f32 because there it is the reference the other modes are checked against.

int4 exists and is **not** recommended for multilingual serving: mean CER 0.227,
half the languages lose more than 0.05.

### Lookahead — the largest single lever

`--lookahead L` selects one of the pack's presets. It changes `P` and therefore
capacity, linearly, while `a` is subtracted first:

| lookahead | chunk period | what the speaker feels |
|---|---|---|
| 0 | 80 ms | the tightest; capacity collapses because `a` eats most of the period |
| 3 | 320 ms | the pack default |
| 6 | 560 ms | ~2x the capacity of lookahead 3 |
| 13 | 1120 ms | for batch-like or transcription-after-the-fact workloads |

Decide it from a latency budget, not from a benchmark:
`make box-advise MODEL_DIR=<pack> ADVISOR_ARGS="--latency-budget-ms 400"`.

### Topology — `--prefork W --threads T`, wide first

```sh
--prefork 2 --threads 16      # on a 32-core box: start here, not at 8x4
```

Workers are pinned core-major from sysfs, so SMT siblings stay together and two
workers never share a physical core. `--prefork-plan` prints the plan without
serving. The reason to start wide is `a`: eight workers on a 32-core box spend
`8a` of every period walking the model before serving anyone.

### Admission — `--cap`, and the rungs beneath it

`--cap C` is the per-worker slot count and rung 1 of the ladder. Set it to about
90 % of the advisor's per-worker prediction so the **ladder** refuses before the
**cadence** does: a 503 with a `Retry-After` is a working server, a stream that
misses its chunk deadline is not.

The rungs, in order: slots → bounded parent queue → queue deadline checked at
pop → per-request service cap. The kernel backlog in front of them is derived
from `W x (slots + queue)` and printed on `[SERVER-CONFIG]` beside the kernel's
`somaxconn`; `RLIMIT_NOFILE` is raised at start-up and reported as it *became*.

### BLAS — none, and that is the default on Linux

`make` on Linux builds with `BLAS=none`: `src/sgemm.c` computes every f32 GEMM
on the same pool as everything else. OpenBLAS would bring a second thread pool
into a worker already pinned to T cpus (measured on an Axion: one worker pinned
to 8 cpus held 63 threads with OpenBLAS linked, 32 without). `make BLAS=openblas`
remains as the comparison arm.

---

## 4. Environment flags that matter for serving

The authoritative list — every flag, its scope, its default, and the reason it
would be **ignored in this build on this host** — comes from the binary:

```sh
./mynah-asr --flags --all      # the whole registry: what a run CAN be given
./mynah-asr --flags            # what THIS run was given
./mynah-asr --dispatch-map     # what the binary will actually execute
```

A document that restated that table would go stale the first time someone added
a flag, so this section names only the handful an operator touches:

| flag | default | when to touch it |
|---|---|---|
| `MYNAH_ASR_THREADS` | online cores | the pool width; the server sets it per worker from `--threads` |
| `MYNAH_ASR_POOL_SPIN_US` | 50 | microseconds a pool thread spins before parking. `0` is the pure condvar pool and the A/B arm — a stream step issues dozens of small GEMMs back to back, and the spin is what stops each one paying a kernel round trip |
| `MYNAH_ASR_CAPS` | auto | forces the int8 kernel level (`scalar\|sdot\|smmla`, `scalar\|avx2\|vnni`). For proving a fallback is real, not for production |
| `MYNAH_ASR_BATCH_F32` | auto | stacks the f32 step's rows. On by default where the GEMM is proven row-stable, off on OpenBLAS where it is not |
| `MYNAH_ASR_GEMM_PROFILE` | unset | dumps the f32 GEMM shapes the model actually issued, per shape, with call counts. The input to `tests/bench_gemm_shapes` |
| `MYNAH_ASR_LISTEN_BACKLOG` | derived | overrides the derived kernel backlog |
| `OPENBLAS_NUM_THREADS` | — | **only meaningful in a `BLAS=openblas` build**; reported IGNORED on `[EFFECTIVE-CONFIG]` otherwise |

---

## 5. What "it works" means — the envelope

A configuration is acceptable when, at the operating concurrency:

| line | gate |
|---|---|
| emission lag p95 | < one chunk period |
| TTFP p95 | < 3 chunk periods + 200 ms (it carries the model's own emission delay) |
| finalization lag p95 | < 500 ms |
| backlog max | < 2 chunks |
| rejects at the operating point | zero |
| a new arrival | does not raise an established stream's emission lag p95 above the gate |

and the transcripts did not move. The harness gives one of:

`GOOD` · `MARGINAL` · **`DEGRADED`** (cadence held, transcripts got worse) ·
`NOT STREAMABLE` · `INVALID` (identity broken, nothing completed, or the client
could not pace at 1x).

`DEGRADED` exists so that "faster" and "better" are not the same word. Pass
`--transcripts <bank manifest>` to give every utterance a CER against a human
reference and let a run reach that verdict.

---

## 6. The commands, end to end

```sh
# 0. build (Linux default: BLAS=none, int8 serving default)
make -j"$(nproc)" && make check && make test

# 1. prove what the binary runs before measuring anything
./mynah-asr --dispatch-map
./tests/test_qmat            # on ARM with i8mm: expect `neon-smmla exact OK`

# 2. describe the machine, and let it refuse if the box is busy
make box-doctor

# 3. predict and get a pasteable command line
make box-advise MODEL_DIR=models/nemotron-3.5-asr-streaming-0.6b

# 4. serve
./mynah-asr-server -m models/nemotron-3.5-asr-streaming-0.6b -p 8090 \
    --quant int8 --lookahead 3 --prefork 2 --threads 16 --cap <advised> \
    --metrics-port 9109

# 5. screen, then qualify
python3 tools/bench/stream_load.py --mode wave --streams <c> --repeat 2 \
    --clips samples/*/*.wav --port 8090 --transcripts samples/manifest.json
python3 tools/bench/stream_load.py --mode soak --streams <c> --duration 600 \
    --warmup 30 --window 60 --bank short,medium,long --seed 42 \
    --clips samples/*/*.wav --port 8090 --transcripts samples/manifest.json \
    --json soak.json
```

Run the load generator **on another host**, or pin it to a reserved slice
excluded from the server's mask, and record which. A generator sharing the
server's cores does not produce a capacity number: on a 32-core Axion an
unpinned one cost worker 0 about 8 % of its first-audio latency in the sibling
TTS project.

---

## 7. Observability

```sh
curl -s localhost:8090/v1/health          # facts: slots, cancels by reason, lag p50/p95
curl -s localhost:9109/metrics            # Prometheus, its own port, token bucket 5/s
kill -USR1 <pid>                          # one-shot [DUMP] per worker, bracketed
```

`/metrics` is **off** unless `--metrics-port` is given, binds 127.0.0.1 by
default, and sets no `SO_REUSEPORT` — a second bind on a live metrics port must
fail loudly rather than hand a scraper one process's counters labelled as the
fleet's.

Known gap, tracked as S6: in `--prefork` the router exports routing facts only
(`worker_up`, `worker_inflight`, `worker_slots`, assigned/completed), while the
scheduler counters live inside each worker and are reachable only by giving each
worker its own `--metrics-port`. There is no single fleet scrape target yet, and
`/v1/health` is answered by whichever worker the router picked.

---

## 8. What is known today, with its provenance

**Machine**: Neoverse-V2, 32 cores, 62 GB, 80 MiB L3, `i8mm` + `bf16` + `sve2`.

| fact | value | label |
|---|---|---|
| resolved int8 batched kernel | `neon-smmla`, exact over 82 shape/T cases, 466 kernel blocks | MEASURED |
| f32 GEMM provider | `own` (no OpenBLAS installed) | MEASURED |
| batched step identity, B=1..8, int8 and f32 | byte-identical, with 208,896 SMMLA blocks executed during the check | MEASURED |
| step table, int8, one 32-thread process, **idle box** (loadavg 0.01) | `a = 31.2 ms`, `b = 4.54 ms/stream`, worst residual 4.3 ms (8 %) | CALIBRATED |
| the same, earlier the same day on a **busy** box | `a = 31.7`, `b = 4.58` — under 2 % apart | the constants REPRODUCE |
| capacity, lookahead 3, ρ=0.8 | ~49 streams in one 32-thread worker | EXTRAPOLATED — the step table ends at B=8 |
| SMT | absent (`control=notsupported`), 32 physical cores | MEASURED |
| `somaxconn` | 4096 — the derived backlog will not be clamped | MEASURED |
| `ulimit -n` | soft **1024**, hard 524288 — the server raises it at start-up and says what it became | MEASURED |
| memory | 63 GB available; at ~14 MB/stream memory is not the constraint | MEASURED |

**What the advisor recommends there** (`make box-advise`):

```
topology  1x32     lookahead 3     quant int8     --cap 44     fleet ~49  [EXTRAPOLATED]
```

with the caveat it prints itself: the cost model optimises **capacity, not
availability**. It picks `W=1` because one worker pays `a` once — but one worker
is one crash away from losing every stream and cannot be restarted under load.
Run the `W` sweep and take `2xN` if it costs little.

### Two levers the doctor found on that machine

- **`bf16`, `sve` and `sve2` are present and never issued** (`IDLE HARDWARE` in
  the dispatch footer). Neoverse-V2 has BFMMLA; nothing in this runtime emits
  it. That is an unexplored path for both `a` and `b`.
- **Transparent huge pages are in `madvise` mode and nothing calls `madvise`**,
  with 0 explicit hugepages reserved. The weights are a 0.79 GB mmap, so at
  4 KiB pages that is ~200 k TLB entries against a few hundred at 2 MiB. A
  one-line `MADV_HUGEPAGE` is the cheapest experiment against `a` on the list.

**Not known yet, and not to be quoted until it is**: no WAVE, no SOAK and no
server load test has been run on any machine. The gap between what the compute
can do and what the *server* delivers — ingest threads, WebSocket framing, the
writer, admission — is unmeasured, and it only goes one way. Treat ~49 as an
upper bound on a number nobody has seen.

### Parakeet, and why lighter is not faster here

| | Nemotron 0.6b | Parakeet-110m |
|---|---|---|
| encoder MACs / frame | 553.6 M | **98.0 M** (5.65x less) |
| joint head MACs / token | 8.38 M | 0.66 M |
| cache-aware streaming | **yes** | **no** |

Parakeet is not cache-aware, so streaming it means re-encoding a window every
chunk. At NeMo's own buffering defaults (left 10 s, chunk 2 s, right 2 s = 7x
real time) the lighter model is a **worse** streaming server than Nemotron,
0.81x. Aggressive buffering claws it back to about 1.6x at an accuracy cost
nobody has measured.

**Parakeet is the throughput product**: offline and batch, where a file is fed
faster than real time and there is no re-encoding penalty, and where the 5.65x
is real. The fleet already serves it that way — `--model name=dir:...` puts it
in its own group, and a WebSocket to it is a `400 model_not_streaming`.
