# Serving audit: metrics, test tiers, quality gates, and the hot path

Status: OPEN (written 2026-09-20 with the box off; everything below is read
from code, scripts and the frozen ledger, nothing was measured)

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

Before the box (Mac, every step with its bit-exactness gate): S10-1 (RK
table, K+Q-1 rows), S10-2 (joint head on `qmat_mul_rows`, stacked across B),
S10-3 (parallel_for over streams), Q-1 (one CER, one pct, `load_transcripts`
keyed by `lang/file`), T-1 reporting-only additions, `make fetch-stress-bank`.

On the box, 90-minute budget: dispatch map, identity gates int8 and f32, the
`objdump` check for `expf`, pool meter at C=32, then the same BEFORE ladder
walked by bisection with validity-length rungs on the new binary, and only if
the knee moved, one 600 s soak at the new C with `--transcripts`. Success is
the curve moving (Phase 7 criterion), not any mechanism metric.
