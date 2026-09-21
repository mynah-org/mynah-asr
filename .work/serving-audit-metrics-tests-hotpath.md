# Serving audit: metrics, test tiers, quality gates, and the hot path

Status: IN PROGRESS. The audit was written 2026-09-20 with the box off, from
code, scripts and the frozen ledger only. On 2026-09-21 four of its items
landed on the development host with their gates green: S10-1 (the rel-pos
table, `5fc9d36`), S10-2 (the joint head, `ebd91de`), Q-1 and T-1 and T-2
(the harness, `a822d31` and after). The box has still measured nothing: the
capacity claim is open and F28 says so. The runbook at the end of this note is
what the box day runs.

Task: S10-1 .. S10-9, Q-1 .. Q-4, T-1 .. T-3 (the board section "S10 — After
the audit")
Question: after two box days that produced F18–F27 and one confirmed operating
point (C=32 on 3x8), is the measurement apparatus professional enough to trust,
where does the code waste per-row time beyond the F27 projection, and what
would make a box day cost ninety minutes instead of a day?

## Known facts

- F22/F23/F27 (`docs/serving-findings.md`): the 3x8 server is compute-bound at
  the knee, `T_step(B) = 14.52 + 19.00·B ms`, and at C=32 the rel-pos
  projection is 11.75 of 24.19 ms per row (48.6 % of the encoder step).
- The 600 s soak at C=32 (F25) lost nothing and scored **zero** transcripts:
  `quality: 0 utterance(s) scored against a reference`. Nothing so far says
  the text at C=32 is the text at C=1.
- The BEFORE ladder was 12 fixed rungs, a fresh server per rung, ~27 min. The
  finalize share it printed was wrong by 3x and nine rungs could not be
  recomputed because a later run reused the output directory (F21).

## Unknowns

- Whether `expf` is vectorised in the Linux binary (branchy sigmoid, two
  `expf` calls; gcc must if-convert before libmvec applies). If scalar it is
  ~25 % of `b`; if vectorised it is nothing. One command on the box:
  `objdump -d mynah-asr-server | grep -c _ZGVnN4v_expf` (0 = scalar).
- The park/spin ratio of the pool at C=32 under the ~300–400 barriers per
  step. The meter exists (`mynah_asr_pool_stats`); it has never been read
  under load.

## Files/functions inspected

`src/encoder.c` (`mynah_asr_pos_emb`, `stream_attention_core`, the batched
step and its S1-7 share block), `src/decoder.c` (joint head call),
`src/qmat.c` (`QMAT_SMALL_T`, `QMAT_ACT_PAR_ELEMS`), `src/backend.c` (SiLU
Linux branch), `src/qmat.h` (`mynah_asr_sigmoid`), `src/threads.c`,
`src/sgemm.h`, `server/sched.c`, `server/slot.c`, `server/stream_out.c`,
`Makefile`, `tests/*.sh`, `tests/test_stream_batch.c`,
`tools/bench/stream_load.py`, `tools/bench/streaming_metrics.py`,
`tools/bench/rest_load.py`, `tools/bench/box_session.sh`,
`tools/bench/box_qualify.sh`, `tools/eval/test_samples.py`,
`tools/eval/test_langs.py`, `tools/perf_profile.py`, `samples/manifest.json`,
and the sibling TTS runtime's serving docs, findings and harness.

## Evidence

### A. Hot path — verified by reading the source (three items)

**A1. Half of every rel-pos projection is never read.** In
`stream_attention_core` the rel_shift reads `brow[base + j]` with
`base = K-1-valid-t`; since `valid = K-Q`, `base` is in `[0, Q-1]` and `j < K`,
so only rows `[0, K+Q-1)` of the `P = 2K-1` projected rows are used. At
K=74, Q=4 that is 77 of 147 rows: 47 % of `relpos_priv` and `relpos` is dead
before any memoisation.

**A2. The projection is a window of one table per layer.** `pe(K)[p]` depends
only on `pos = K-1-p`, so `rk(K) = pe(K) @ relk_w^T` is the contiguous row
window `[Kmax-K, Kmax-K + K+Q-1)` of `RK = pe(Kmax) @ relk_w^T`, computed
once per layer at load (~14 MB per model at 24 layers, 147 rows, d=1024).
`matmul_wt` is the DOT family of `src/sgemm.c`: each output row is one dot
over k and nothing depends on M, so the window is bit-identical to the
per-stream call and `tests/test_stream_batch` proves it. This makes the S1-7
K-group sharing machinery (`k_sh`, `k_sh_n`, `P_sh`, `rk_sh`) redundant.

**A3. `mynah_asr_pos_emb` has `pow` in the inner loop and runs on every
cache-fill step.** `pow(10000.0, -2.0*j/d)` depends only on `j` but sits
inside the `p × j` loop: 75k `pow` + 75k `sin` + 75k `cos` in double per call.
It is called whenever `pe_K != sa_pe_K`, i.e. on each of the ~18 steps while
`cache_valid < left`, and again after every `mynah_asr_enc_stream_reset`
(every utterance in the server). With A2 it is not called at all.

**A4. The RNNT joint head runs through the small-T serial dot path.**
`mynah_asr_qmat_mul(&dec->head, jin, logits, Bc)` with `Bc <= 4` takes the
`T <= QMAT_SMALL_T` branch: `for t { quantize; for i { dot } }`, weights
(8.4 MB int8) re-read per frame, no pool. `mynah_asr_qmat_mul_rows` is
weight-stationary and bit-identical by the S1-4 contract. This cost is
outside the F26/F27 encoder profile and was never seen.

### B. Hot path — reported by the audit, not independently verified

| Item | Where | Estimate per row at B≈3 | Fix |
|---|---|---|---|
| Elementwise stages serial on the scheduler thread while 7 cores idle: 5 layer norms, SiLU per stream, activation quantisation (`QMAT_ACT_PAR_ELEMS = 2^21` never reached at R·k ≤ 512k), depthwise conv, the whole attention + cache loop | `src/encoder.c` batched step, `src/qmat.c` | 3–6 ms of b=19 | `parallel_for` over streams; SiLU fused in the qgemm epilogue; act-quant by row when T ≥ 8 |
| Layer norm in double, three passes, 5× per layer | `layer_norm_f` | 0.3–0.5 ms | f32 two-pass, oracle gate |
| Mel/FFT in double, scalar, per stream, on the scheduler thread | `src/features.c` | 0.3–0.5 ms | twiddle table, real FFT, parallel over streams, features golden gate |
| ~300–400 pool barriers per step with a 50 µs spin: serial gaps exceed the spin, workers park, the next dispatch pays futex wakes | `src/threads.c` | part of a=14.5 | read the meter, then A/B `MYNAH_ASR_POOL_SPIN_US=1000` in SOAK, not WAVE |
| K/V cache copied into `sa_keys` every layer (600 KB) plus a shift memmove (540 KB): ~27 MB moved per row-step | `stream_attention_core`, `update_kv_cache` | 0.3–0.6 ms | `[left+Qmax, d]` layout written in place, compaction every left/Q steps |
| Subsampling per stream: weights re-read per stream, 4–5 barriers each, 17.8 MB f32 linear | `mynah_asr_enc_stream_step` prologue | 0.94 (measured, F26) | stack across B, quantise the linear |
| `xn` quantised three times for q/k/v | batched step | small | once |
| int8 micro-kernel 1×4 tile, load-bound | `dots_q8_sdot_x4`, SMMLA 2×2 | kernel tuning | 4×4 register tile, bit-exact by integer accumulation |

Clean, checked: `BLAS=none` on Linux, one pool, no second thread team;
zero per-step allocations in encoder/subsampling/decoder/detok/sgemm/pool for
what the gates cover; output writer truly asynchronous; pre-quantised int8
zero-copy from the mmap; no double literals in f32 loops.

Two claims narrower than they read: (1) "0 allocations per chunk" is gated on
the CLI single path (`tests/test_stream_allocs.sh`) and the batched path at
B=4 with no callback (`tests/test_stream_batch.c`); the server's cJSON delta
framing allocates ~30 times per delta inline on the scheduler thread and no
gate covers it. (2) Offline REST jobs execute on the streaming scheduler
thread (phase 6): one REST request stalls every stream of that worker for its
duration (S9-5, open).

### C. Measurement apparatus

Present and sound: TTFB/TTFP, emission lag pooled and per-utterance,
finalization lag, backlog proxy, pacing with refusal, drift and trend per
window with ramp/drain excluded, cross-stream identity, CER vs reference,
scheduler wall split, ready→selected→model-start histograms, per-position
step cost, component profile, exact over-threshold counters on `/metrics`,
provenance manifest with per-clip sha256.

Missing, raw data already in the run JSON (reporting change only): p99 and
p99.9, audio-seconds per wall-second in the WS report, end-of-utterance
latency (the `eou` event is counted, not timed), per-stream fairness (worst
stream vs median), per-stream stall (max inter-delta gap), WER beside CER,
bootstrap confidence intervals, a cross-run comparator that refuses to
compare runs with different model/quant/lookahead/C/bank hash/commit.

Missing, needs instrumentation: resource usage. No CPU %, RSS, context
switches, page faults or frequency anywhere, server or client. Server side is
`getrusage(RUSAGE_SELF)` + `/proc/self/status` in `[DUMP]` and `/metrics`;
client side is a `/proc/<pid>` sampler at window cadence written into the JSON.

Definition drift found:
- Two CER normalisers and two Levenshteins (`tools/eval/test_langs.py` with
  NFKC and all Unicode punctuation vs `tools/bench/streaming_metrics.py`
  without NFKC and a fixed list missing `«»„¿`): es/de references score
  differently depending on which tool ran.
- Two percentile definitions (`rest_load.pct` linear rank vs
  `streaming_metrics.pct` nearest rank): the same word "p95", two numbers.
- Five WebSocket framing implementations; two box orchestrators
  (`box_session.sh`, `box_qualify.sh`) with different banks and topology
  models, neither in `make`.
- Bug: `stream_load.load_transcripts` falls back to the basename, so
  `samples/<lang>/fleurs_1521.wav` collides across 11 languages and a
  multilingual CER run scores against the wrong reference.

Dead or never run: `tests/test_kquant.sh` always exits 77 (`tests/test_gguf`
has no source and no Makefile rule); `tests/test_ingot_parity` is a tracked
binary with no source (rule 6); `make fetch-stress-bank` never run
(`samples/stress-en` holds a README); `make bench-stream-wave` includes the
94.6 s clip by default so it cannot finish under two minutes;
`tools/perf_profile.py validate` is not in `make check`.

### D. Quality as a regression system

Exists: human references for 11 languages (`samples/manifest.json`), 102
local clips over 34 locales, CER, `stream_load --transcripts` (self-test
only, never run on a real bank), byte-identity REST/WS vs CLI.

Missing: a committed CER baseline per model/quant/lookahead; a gate that
fails on regression beyond a margin (today's thresholds are absolute 0.20 /
0.30 / 0.25 and cannot see 0.13 → 0.19); the streaming-vs-offline CER delta
per lookahead over a corpus (int8 streaming ≠ offline on `test_es`, and
nothing measures by how much); CER on REST (none anywhere). Partial revision
rate is structurally zero on Nemotron (every delta is `final:true`) and
becomes relevant only with an unstable-tail decoder (M-4).

### E. Parakeet

Not in the streaming suite for a config fact: no Parakeet pack declares
`att_context_presets`, and `mynah_asr_stream_unsupported` refuses on six
weight facts regardless. Covered today: oracle parity, e2e, offline CER, REST
concurrency in CI (110m), `rest_load` ladder. The only box ladder was
withdrawn (it measured the Nemotron server still running). A light/heavy
two-model suite at near-zero cost: Parakeet on REST with a per-request latency
envelope, same corpus and references, `rest_load` gaining `--reference` and
`--transcripts` and using the one `pct`. Buffered streaming (M-4) is 1–2 weeks
and ~7x RT per stream; not on the critical path.

### F. From the sibling TTS runtime, not yet borrowed

- **A screen is defined by sample count, not minutes**, and validated by
  injecting a known regression: its 2-minute mini-soak could not separate the
  arms, 1500 samples could. Its knee screen (30 s warm-up + 2×90 s per rung)
  agreed with the 30-minute soak within 0.02 in three pairs. Walk the ladder
  at 3–4 min per rung; run the long soak once at the highest clean rung.
- **One persistent parallel region per step with spin barriers between
  phases** instead of one dispatch per projection: dispatches per frame
  506→202, csw 12k→7k/s, bit-identical. Directly the 300–400-barrier problem.
- **ARM topology preflight**: `roof_matvec_int8` run as 1x8/2x8/4x8
  simultaneously on disjoint masks; on one Arm host 4x8 aggregate was 0.90x
  of an isolated 1x8. Before fixing W×T on any box.
- Pool spin budget swept in SOAK, not WAVE (neutral in wave, +11 % p95 in
  soak on the same host).
- A parity gate must carry a marker that the treatment executed; a prefork
  identity test never exercises the batched path (two requests → two
  workers at B=1).
- Per-worker exact threshold counters for alerting, `terminated_{ok,
  client_gone,timeout}`, gap-behind-realtime as the stall proxy,
  `resources.csv` from `/proc` every 15 s with FAIL on anon/pss growth >10 %.
- Negatives not to repeat: more and smaller tasks per region lose;
  runtime-indexed accumulator arrays spill to stack (cohort kernel 1.2–1.6x
  slower); vector bf16 on SPR is half the f32 FMA rate and pays only on
  Zen4/5, Neoverse-V2 BFMMLA and AMX; glibc allocator tuning 0 %;
  cross-worker batching 2.7 % coincidence.

## Conclusion

The latency measurement is professional; the quality and resource
measurement is absent, so "C=32 holds" is a latency claim and not yet a
transcript claim. The ladder is four times slower than it needs to be and the
soak has never scored text. On the code, F27 was the right target and it is
larger than recorded: half its rows are dead and the whole projection plus
`pos_emb` collapses to one table per layer at load. The next largest costs
(serial elementwise stages, the joint head on the small-T path, the barrier
count) are all mechanical, bit-identical changes.

## Next action

**Done on the development host, 2026-09-21, each with its gate:**

- S10-1, the rel-pos table. 0 of 266,240 floats differ at B=8 in both quants,
  with the table built between the two passes so the comparison is
  table-against-per-step. Recorded as F28.
- S10-2, the joint head on the weight-stationary path. Transcripts byte-identical
  on five languages; the qmat counters account for the migration exactly
  (`dot` -276, `dot_rows` +276 at B=8).
- Q-1, one CER and one percentile. It exposed a real bug: the percentile rank
  was computed as `(p/100)*n`, so the p99.9 of a thousand samples was silently
  the maximum. Fixed and pinned.
- T-1, the reporting the data already supported: p99/p99.9 (withheld below the
  n that can name a rank under the maximum), audio per wall second,
  end-of-utterance latency, per-stream fairness, longest inter-delta gap, WER.
- T-2, the three tiers as one command each, with `knee.py` (bisection on a warm
  server), `cer_offline.py` (baseline and margin) and `compare_runs.py` (which
  refuses to compare runs that differ in an undeclared way).
- The first quality baseline, `configs/quality/`: this pack at int8 scores
  CER mean 0.045, weighted 0.060, worst clip pt/fleurs_1521 at 0.272. Before
  today no committed number said what "still correct" means for this model.

**Deliberately NOT done, and why.** S10-3 (the elementwise stages), S10-4 (the
pool spin sweep) and S10-5 (the K/V cache) are the next largest items and all
three are left alone until the box has measured the two changes already
stacked. Landing a third would make the ladder unattributable, which is the
guardrail this campaign has been run under from the start.

**Next: the box.** See the runbook at the end of this note. The success
criterion is unchanged and is not a mechanism metric: the capacity curve moves,
or the change did not work.

---

## Box runbook, written 2026-09-21 before the box is opened

Budget: 90 minutes. Everything below runs from a clean committed tree, in tmux,
on the Axion. The order is fixed: nothing that costs time runs before the thing
that could invalidate it.

**Before anything (5 min) — does this build do what the Mac build did.**

```
git pull && make clean && make
./mynah-asr-server --dispatch-map | tee dispatch.txt      # int8_rows must resolve
./tests/test_stream_batch models/<pack>                    # EXACT OK / IDENTICAL OK
MYNAH_ASR_RELPOS_TABLE=0 ./tests/test_stream_batch models/<pack>
objdump -d ./mynah-asr-server | grep -c _ZGVnN4v_expf      # S10-7: 0 means scalar
```

The gate is the same one that ran on the dev host, so a difference here is the
provider, not the change. The `objdump` line settles S10-7 in one command: it
is the only open question in the audit that a Linux binary can answer for free.

**The measurement that decides everything (25 min) — the same ladder, twice.**

Same corpus, same seed, same affinity, same warm-up, same run length as the
`a48d443` BEFORE ladder of F22. Run the AFTER arm first, then the BEFORE arm by
setting `MYNAH_ASR_RELPOS_TABLE=0` on the same binary, so the two differ in one
environment variable and nothing else — not a rebuild, not a checkout.

Report, per rung, exactly the F22 columns: audio/wall, lag p50/p95, backlog,
step, finalize, runnable idle, no-work, model, mean B, ready-to-model-start
p50/p95/p99, verdict. The success criterion is F28's, and it is not a
percentage: **C_safe and C_knee move, or the change did not work.** The
predicted range is C = 42..53 against a measured 36. Outside it, the cadence law
or the reasoning behind F28 is wrong, and that is the finding.

`tools/bench/knee.py` can find the knee in about five probes on one warm server,
but the BEFORE/AFTER comparison must be the SAME ladder F22 ran, rung for rung.
Use the bisection for the exploring, the full ladder for the claim.

**Only if the knee moved (35 min) — qualify the new point.**

```
make fetch-stress-bank                 # once; every soak before this scored nothing
make cer-baseline MODEL_DIR=models/<pack>
sh tools/bench/tier.sh 2 -m models/<pack> -C <the new knee>
```

Tier 2 runs two identical 600 s soaks with references and refuses to start on a
dirty tree or without a manifest. Two runs, because the widest move between two
identical runs IS the noise floor of this harness at that point, and no smaller
difference may be called a result afterwards.

**What NOT to do on this box day.**

- Do not also land S10-3 (the elementwise stages), S10-4 (the pool spin sweep)
  or S10-5 (the K/V cache) first. Two changes are already stacked, S10-1 and
  S10-2; a third makes the ladder unattributable. The table has an A/B arm
  (`MYNAH_ASR_RELPOS_TABLE`) precisely so that one of the two can be isolated.
- Do not tune a threshold to make a rung pass.
- Do not promote C from a wave. Only tier 2 promotes.
- Do not read the Darwin 1.32x as a prediction for this host. It is a direction.

**What the box owes back to the repo**, whatever the answer: the ladder table in
`docs/serving-findings.md` as F29, the profile in `configs/perf/` updated with
`measured_short_run_safe` and, only if tier 2 passed twice, `long_soak_qualified`.
A negative result is written up in the same detail as a positive one.
