# Serving operations — how a new box is approached

For the engineer who has just been handed a Linux machine and has to say how many
real-time streams it serves. The order below is not a style preference: every step
exists because skipping it once produced a number that turned out to be false.
Normative rules: [`ENGINEERING.md`](../ENGINEERING.md) §8, §10, §12, §14. Design:
[`.work/serving-v2-design.md`](../.work/serving-v2-design.md). Harness detail:
[`.work/bench-harness-streaming.md`](../.work/bench-harness-streaming.md). Method:
[`.work/engineering-method.md`](../.work/engineering-method.md).

## Vocabulary — use these four words and no others

| word | what it is | what it can do |
|---|---|---|
| **WAVE** | C streams at once, R utterances each, then stop | **screens**: it may DISQUALIFY an operating point, it never promotes one |
| **SOAK** | closed loop at fixed concurrency for minutes, mixed bank, drift gate across windows | the **only** thing that promotes |
| **POISSON** | open arrivals past capacity | characterises overload and refusal behaviour |
| **DIAGNOSTIC** | an instrumented or unpaced run | never quotes a headline number |

Never a bare "RTF" and never an unqualified "concurrency N". `STREAM_RTF` is a
**capacity** number (compute seconds per audio second); it is not a promise that any
stream stayed real-time, and it never replaces the cadence metrics.

## 1. Inspect the machine before building anything

```sh
nproc; lscpu | head -20                  # cores, sockets, ISA
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list   # SMT pairs
cat /sys/fs/cgroup/cpu.max                                       # a quota the box will not mention
uptime                                   # refuse to measure while something else runs (§10)
```

All of it, with every fact's source named, is one read-only command:

```sh
tools/bench/box_doctor.sh                # describes the box; measures nothing
```

It adds the SMT sibling pairs and the core-major order the server would use,
what `--dispatch-map` resolves to here, the ~14 MB per stream quoted from
`.work/fleet-observability.md` §4, the limits that bite at high fan-in
(`nofile`, `somaxconn`, the ephemeral range), and the affinity mask that makes
`nproc` a lie. It refuses with exit 3 above loadavg 2.0 as `box_qualify.sh` does,
and prints UNAVAILABLE for what a platform cannot tell it. Its `W x T` table
lists CANDIDATES for the sweep in §3: the sweep decides, it never recommends.

## 2. Build for this ISA, then make the binary prove what it runs

```sh
make clean && make -j"$(nproc)"          # a gate that decides something gets a clean build
./mynah-asr --flags                      # [FLAGS] / [EFFECTIVE-CONFIG]: build, blas, simd
./mynah-asr --dispatch-map               # which kernel each operation resolved to, and why
```

`--dispatch-map` ends with `N row(s) UNKNOWN` and an `IDLE HARDWARE` footer. **A
benchmark is invalid until dispatch is proven** (§5): if the row you intend to measure
resolved to something else, or to UNKNOWN, stop and fix the build. An explicitly
requested path that silently fell back is a failed run, not a slower one.

## 3. Choose the operating point, do not guess it

```sh
./mynah-asr-server --prefork-plan        # topology, the W x T grid, the admission ladder
```

It prints the **procedure**, not a number: find `T*` (the smallest pool width within
~10% of the best single-stream RTF — the stream step is bandwidth bound, so `T*` is
usually small), then `W = cpus / T*`. Start the candidate:

```sh
./mynah-asr-server -m models/nemotron-3.5-asr-streaming-0.6b --quant int8 -p 8090 \
    --prefork W --prefork-threads T --threads T --cap C
curl -s localhost:8090/v1/health         # facts, not configuration: inflight, slots, lag_*
```

`--cap C` is streams per worker; fleet capacity is `W x C`. Beyond it the admission
ladder refuses with `503 server_at_capacity` **before** the WebSocket upgrade, which is
the behaviour you want and which the harness counts as a REJECTION, not an error.

## 4. WAVE — screening

```sh
python3 tools/bench/stream_load.py --mode wave --streams 4 --repeat 2 \
    --clips samples/*/fleurs_15*.wav tests/audio/test_*.wav --port 8090 --json wave4.json
```

Read it as a gate, not as a result. A WAVE that comes out NOT STREAMABLE removes that
operating point from the candidate list. A WAVE that comes out GOOD has proved nothing
about the tenth minute: it is a permission to soak, and that is all.

## 5. SOAK — the only thing that promotes

```sh
python3 tools/bench/stream_load.py --mode soak --streams 8 --duration 600 \
    --warmup 30 --window 60 --bank short,medium,long --seed 42 --class-bounds 7,11 \
    --clips samples/*/fleurs_*.wav tests/audio/test_*.wav --port 8090 --json soak8.json
```

- **closed loop**: each of the `--streams` client processes starts its next utterance as
  soon as the previous one finished, so concurrency is held, not aimed at.
- **stratified bank**: `--bank` names duration classes built from the clips by
  `--class-bounds`; the schedule is round-robin over the classes with a seeded shuffle
  inside each, so `--seed` reproduces the exact audio order. A bank of one length
  measures one length. Every class must be shorter than the soak.
- **warm-up**: the first `--warmup` seconds are excluded from the KPIs and said so in the
  output (`excluded by warm-up`), never silently dropped.
- **windows**: `--window` seconds each, with per-window p50/p95 printed, and a drift gate
  (`--max-drift-pct`, default 20) on the distance between a window p95 and the pooled
  p95. A server that is fine for two minutes and degrades in the tenth fails here and
  nowhere else — which is the reason a SOAK exists.
- **manifest**: every run writes utc, client host and python, tree revision and dirty
  flag, server URL, concurrency, duration, the bank with each clip's sha256, the seed,
  frame size, pace, the thresholds in force, and `/v1/health` before and after.

Minimum for a promotion: **≥ 10 minutes**, a mixed bank, a clean committed tree (§12;
a dirty-tree binary is NON-QUALIFYING and serves WAVE and DIAGNOSTIC only).

Both invocations have a Makefile shortcut against an already running server:
`make bench-stream-wave STREAM_N=4 STREAM_PORT=8090` and
`make bench-stream-soak STREAM_N=8 STREAM_DURATION=900`.

## 6. The envelope

Provisional (`.work/serving-v2-design.md` §6), to be recalibrated by S0. One chunk period
= `encoder_frame_ms x (lookahead + 1)` = **320 ms** at the Nemotron default lookahead 3.
Every line is a CLI flag; the defaults are these.

| line | limit | flag |
|---|---|---|
| TTFP p95 | 3 chunks + 200 ms (1160 ms): TTFP includes the model's own emission delay, ~2 chunks after the first on Nemotron; the server's cost is the emission-lag line | `--ttfp-p95-ms` |
| emission lag p95, pooled over deltas | 1 chunk (320 ms) | `--lag-p95-ms` |
| finalization lag p95 | 500 ms | `--fin-p95-ms` |
| backlog max | 2 chunks (0.64 s) | `--backlog-max-s` |
| drift of any p95 across windows | 20% | `--max-drift-pct` |
| rejects at the operating point | 0 | — |

Verdict, with the failing line always named:

- **GOOD** — every line inside, no rejects, no errors.
- **MARGINAL** — a line over its limit but within `--marginal-factor` (1.5x), or any
  reject or error. Not a promotion.
- **NOT STREAMABLE** — a line beyond that factor.
- **INVALID** — the run cannot be judged at all: two streams produced different text for
  the same clip, nothing completed, or the client could not hold 1x pacing. Timing
  numbers from an INVALID run are never quoted.

## 7. Reading the metrics

All of them are defined once, in `tools/bench/streaming_metrics.py`; `--self-test` there
is a known-answer gate that `make test` runs without a model. Never re-derive one in a
spreadsheet.

- **TTFP** — first delta minus first audio byte sent. The number a user feels first.
- **emission lag (client-observed)** — a delta's arrival minus the send time of the frame
  that carried the last sample the server said it had consumed. An **upper bound** on the
  server's own lateness: kernel and Python buffering sit inside it. The server's
  `lag_ms` is printed beside it, never averaged into it. If the two diverge, the gap is
  the client and the network, and it is an engineering question, not a rounding.
- **finalization lag** — `done` minus the last audio byte. Where a tail flush hides.
- **backlog (proxy)** — audio sent minus audio the server reports consumed, at each
  delta. It is what the client can see, not the server's ring: it is named a proxy.
- **pacing lateness** — how late the client sent its own frames. Above half a frame the
  run did not hold 1x, every cadence percentile is labelled DIAGNOSTIC, and the envelope
  says INVALID. This refusal is the point (§4 of the method note): a number the data
  cannot support is not printed as a measurement.
- Percentiles are **nearest rank** over a named basis, and the basis is printed:
  *pooled over deltas* and *per utterance* answer different questions. Never a ratio of
  percentiles.
- **text identity** — the same clip must produce byte-identical text in every stream. A
  mismatch invalidates the run whatever the timing says (§9).

## 8. What a report must contain (§14)

WHAT CHANGED · WHAT PATH ACTUALLY RAN · WHAT WAS MEASURED · WHAT REMAINS UNKNOWN ·
VERDICT: PROMOTE / KEEP / INCONCLUSIVE / REJECT.

Concretely, for a serving claim: the mode (WAVE/SOAK) and duration, concurrency, the
`--prefork`/`--cap`/`--threads`/`--quant` actually used, the dispatch map rows that
matter, the bank and its seed, the manifest path, every envelope line with its value,
the drift, rejects/errors/timeouts, and the survivor count after teardown. Unknowns are
stated, not omitted; a model-gated test that was skipped is reported with the exact
command and the reason, never replaced by a claim.

## 9. Run lifecycle (§10)

A persistent server is never inside a bare `wait`. Start it, save its PID, confirm
`/v1/health` answers and its counters advance before trusting anything, wait only on the
client PIDs with bounded timeouts, then terminate the server explicitly and report
`survivors=0`. Refuse to start a measurement while `uptime` shows another load.
