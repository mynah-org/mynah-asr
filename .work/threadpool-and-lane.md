# S1 — the thread pool, the lane, and BLAS ownership

Status: S1-6 CLOSED for correctness and ownership · ALL TIMING DEFERRED to S1-6a on the Linux box · S1-5 OPEN

Task: S1-5, S1-6, S1-6a
Question: replace `src/threads.c` (a parallel_for that serialises under
concurrency, plus a process-global weak-symbol BLAS knob) with the mynah-tts
pool, and decide who owns the threads in a worker.

Known facts
- mynah-tts `src/threads.{c,h}` is self-contained (`<stddef.h>` only): spin-
  then-park with a calibrated budget (ns target, not iterations; 65536 on
  aarch64 Linux measured as the knee, 4096 would cost 16.7% at 32 threads),
  precheck ON, narrow/fastexit OFF with the measurements recorded, litmus
  self-test, `after_fork`, a pool meter compiled in always (dispatches per
  frame, region µs, barrier %, width entered vs useful), and the one-line lane
  redirect `pool = g_on_lane ? &g_lane : &g_engine` keyed on a thread-local.
- mynah-tts `src/sgemm.{c,h}` is a drop-in `cblas_sgemm` with five families,
  deterministic across thread counts (reduction never split; row blocks rounded
  to the micro-kernel), scalar reference always compiled.
- Measured on the Axion: one worker pinned to 8 cpus held 63 threads with
  OpenBLAS linked and 32 without. qwen-tts carries "OPENBLAS_NUM_THREADS must
  be absent" as a forbidden-env rule because a stray export replaced a
  qualified topology once.
- Here: every f32 linear, attention GEMMs, subsampling, LSTM sgemv and the
  joint head went through `cblas_*` directly, in seven translation units.
  Accelerate on macOS, OpenBLAS on Linux, hard `$(error)` without `cblas.h`.

Decision (standing, as in mynah-tts): **OpenBLAS leaves the Linux worker**,
for ownership, not speed. Own f32 GEMM plus the int8 kernels; `BLAS=openblas`
kept as a comparison build forever; Accelerate stays available on macOS as a
development convenience behind the same seam, never as a production claim.

Hazard recorded 2026-09-18 (S1-6a): inside a prefork worker pinned to T cpus,
`mynah_asr_num_threads()` = T builds a pool of T threads AND OpenBLAS builds its
own team of T (the server sets the budget once, to one inference in flight), so
a worker runs 2T threads on T cpus and the two pools take turns owning the
cores. `parallel_for` already forces OpenBLAS to 1 thread inside a region; the
f32 GEMMs outside regions do not. The int8 stream step barely touches BLAS
(attention scores, subsampling, LSTM), so the interim Linux default candidate is
OpenBLAS = 1 thread in a worker; it is a measurement on the box, not a guess,
and it disappears with S1-6.

Unknowns (state at the close of S1-6)
- **Every timing measurement is DEFERRED TO THE LINUX BOX (S1-6a).** The
  decision on 2026-09-18 was to take none of them here: this Mac was loaded
  (another agent's full `make test`, loadavg 4.7-7.6 on 8 cores) and low on
  RAM, so an RTF or step-time table taken on it would have been a number with
  no envelope. Concretely NOT measured and owed by S1-6a: `mynah-asr bench`
  RTF, the `--steptime` table at B = 1,2,4,8, `openblas` vs `none` in any
  form, the thread count inside a pinned prefork worker, and the OpenBLAS f32
  row-stability gate. The only timing that IS recorded below is the
  server-stream envelope, and only because it was a correctness GATE that
  failed and had to be explained — it is labelled with its load and it
  promotes nothing.
- **Everything Linux.** No number exists for our own sgemm on the Axion or on
  x86. That is the whole of S1-6a and it is the half of this decision that
  actually matters, because production is Linux. What landed below is an
  ownership change plus correctness gates.
- macOS: `parallel_for` P-core default is a macOS concept; on Linux the pool
  width is the worker's mask size.
- S1-5 (the mynah-tts pool itself) has NOT been lifted. `src/sgemm.c` runs on
  this repo's existing pool, which takes ONE dispatch at a time and runs inline
  when busy. That is correct and deterministic, but it is not the spin-then-park
  pool with the meter, and the sgemm cost model
  (`SG_PARALLEL_MIN_WORK`, `SG_PANEL_NV`) was written against the sibling's
  dispatch cost, not against this one measured here.

---

## Acceptance gate for S1-6 (stated before the work)

1. Transcripts byte-identical between `BLAS=accelerate` and `BLAS=none` on this
   host: `make test` green in both, `tests/test_streaming` and
   `tests/test_stream_batch` identical. A difference stops the work and goes in
   this note against the oracle's per-stage tolerances.
2. `mynah_asr_sgemm_self_test()` in `make test`: every compiled family against
   the reference over the real shapes and the edge cases, a thread-count sweep
   proving the result is identical at 1, 2, 4 and 8 threads, and a REFUSAL if a
   family never ran.
3. `BLAS=none` links no cblas at all; `BLAS=openblas` keeps building;
   `--dispatch-map` and `[EFFECTIVE-CONFIG]` report the provider actually linked.
4. A/B on this host, labelled as a macOS dev signal.
5. `make check` green, one UBSan run of `tests/test_streaming`.
6. CI exercises `BLAS=none` on Linux x86 and ARM.

## Evidence — S1-6, 2026-09-18

Host: Apple M1 (4P+4E, 8 logical), macOS 25.5, Apple clang, `-O3 -march=native
-ffast-math`. Model `nemotron-3.5-asr-streaming-0.6b`, int8 checkpoint,
preset [56,3]. **The box was NOT quiet**: `loadavg` 2.1–2.8 throughout
(other agents on the same machine). Every A/B below is therefore a WAVE-grade
dev signal and nothing here promotes anything.

### What landed

- `src/sgemm.{c,h}` — lifted from mynah-tts with the `mynah_asr_` prefix and
  this repo's pool. Five families (REFERENCE, DOT, MATVEC, NARROW, PANEL), the
  reduction over k never split, the row block always rounded up to `SG_MR`, so
  **the result does not depend on the thread count**; `narrow_max` derived from
  the register file (NEON 16, AVX2 16, scalar 4); the never-vectorised
  `mynah_asr_sgemm_f32_reference` always compiled, used both as the oracle and
  as the scalar fallback. Not lifted: the conv-tap fusion (this runtime has no
  conv1d tap GEMM — the subsampling convolutions go through im2col and one
  GEMM), and `kernels.c` (the DOT family's dot is written in the same `SG_*`
  ISA macros as the micro-kernels, so this file has one ISA abstraction rather
  than two).
- `src/backend.c` — `mynah_asr_gemm_f32` and `mynah_asr_gemv_f32` are the ONE
  seam, plus `mynah_asr_gemm_provider()` returning `accelerate` / `openblas` /
  `own`. Every direct `cblas_*` call site now goes through it: 22 GEMMs
  (encoder 9, subsampling 5, decoder 3, decoder_ctc 2, vad 2, and the f32
  fallback in qmat 1) and 7 GEMVs (decoder 3, decoder_aed 2, vad 2). **No file outside
  `backend.c` includes `cblas.h` or `Accelerate/Accelerate.h` for arithmetic.**
- `Makefile` — `BLAS=none|openblas|accelerate`, `none` the Linux default,
  `accelerate` the macOS default. The `$(error)` on a missing `cblas.h` now
  lives inside the `openblas` branch only.
- **`MYNAH_ASR_ACCELERATE` is split from `MYNAH_ASR_BLAS_ACCELERATE`.** The
  Accelerate FRAMEWORK is linked in every macOS build (Metal needs it, and so
  does the vForce `vvexpf` in `mynah_asr_silu`); the BLAS provider is a
  different question. Without that split, `BLAS=none` on macOS would silently
  have changed the SiLU's arithmetic as well as the GEMM's, and an
  accelerate-vs-none transcript difference would have had two possible causes.
- `src/threads.c` — the budget mechanism is kept and made honest: with provider
  `own` there is exactly one pool, `mynah_asr_blas_budget()` IS its width, and
  `mynah_asr_blas_set_concurrency()` records the declaration without inventing a
  smaller number for a team that does not exist. `OPENBLAS_NUM_THREADS` is
  reported IGNORED by the flag registry in that build. The OpenBLAS path is
  untouched.
- `tests/test_sgemm.c` in `make test` and in CI before the model download.

### Gate 2 — the self-test

`tests/test_sgemm` (provider `own`, isa `neon`, narrow_max 16), all OK:

- every compiled family against the reference over 13 real shapes
  (m = 1..176, n/k in {1024, 4096, 13088}, the attention window, the
  subsampling flatten) x {trans_b, no trans} x {beta 0, beta 1} and 13 edge
  cases x {both transpose flags} x {alpha 1 / -0.75 / 0.5, beta 0 / 2.5 /
  -1.25}, plus the coverage refusal;
- worst relative deviation **6.55e-05** against a 1e-04 bound. That number was
  chased rather than accepted: against a DOUBLE-PRECISION oracle on the same
  shape (m=64 n=1024 k=4096, trans_b) the **reference** is 6.81e-05 off and the
  **kernel** 3.85e-05 — the kernel is nearly twice as close to the exact answer
  as the scalar reference it is compared against, and the gap between them is
  dominated by the reference's own sequential summation over 4096 terms. On the
  same shape with trans_b == 0 (NARROW/PANEL) the kernel is **bit-identical**
  to the reference. The bound is therefore a bound on a known quantity;
  tightening it would first trip over the reference, not over a bug.
- task-count sweep (1, 2, 3, 5, 64 blocks) byte-identical, in both transposes;
- **real** thread widths 1, 2, 4, 8, each a re-exec'd child with its own pool
  (`mynah_asr_num_threads()` caches on first call, so a plain fork would have
  reported a sweep it never ran): the whole self-test passes at each width and
  the 64x1024x1024 trans_b output is byte-identical across all four.

### Gate 1 — transcripts: IDENTICAL, with one named exception

`make test` with `BLAS=none`: everything green except the load-sensitive server
phase below. Per-stage oracle parity (Nemotron, `test_it.wav`), `BLAS=none`:

| stage        | max abs diff | tolerance | verdict |
|--------------|--------------|-----------|---------|
| mel          | 0.000e+00    | 5e-04     | OK      |
| subsampling  | 5.320e-04    | 1.2e-02   | OK      |
| layer_0      | 8.144e-05    | 3.5e-02   | OK      |
| layer_12     | 5.188e-05    | 1.5e-02   | OK      |
| layer_23     | 2.668e-07    | 9.0e-05   | OK      |
| enc_proj     | 1.927e-06    | 1.2e-04   | OK      |

Every stage is 1.5 to 2.5 orders of magnitude inside its tolerance. The e2e
transcripts (it/en/de/fr/es, int8 quant, timestamps, segment, metal) are all OK
and word-for-word what the accelerate build produces.

Direct diff of the two builds, same host, same binaries' outputs:

- `tests/test_streaming` — **byte-identical** (`accelerate` vs `none`), and also
  identical to the pre-change `accelerate` baseline, which is the regression
  check that says the seam itself changed nothing for Accelerate.
- `tests/test_stream_batch` — all identity gates pass in BOTH builds:
  `[f32] encoder bit-exact B=2/4/8`: 74,240 / 133,120 / 266,240 floats compared,
  **0 differ**, 0 caches differ, in both. `[int8]` the same. `[f32] B=1..8
  IDENTICAL OK` in both, and on `none` the f32 path really did stack (104 / 152 /
  196 / 416 stacked rows at B=2/3/4/8, `rows_stacked > 0`), which is the
  by-construction row-stability claim actually exercised rather than asserted.
- **The one difference**, and it is named rather than smoothed over: the
  **es-ES clip's single-stream int8 decode**.

      offline (both builds):  "...empieza a las vuelve en la sala grande."
      single, accelerate:     "...empieza a las nueve en la sala grande."
      single, none:           "...empieza a la submuer en la sala grande."

  This is the clip `tests/test_stream_batch` already prints as
  `single stream != offline (pre-existing, not a batching effect)`: BOTH builds
  disagree with their own offline transcript on it, before and after this work.
  What moved is which way it falls. The mechanism is not mysterious — an int8
  model still runs f32 attention scores, f32 subsampling and an f32 LSTM
  pred-net, all of which now go through a different GEMM, and a greedy decode
  sitting on a knife edge flips. It is a NUMERICAL difference, so per
  ENGINEERING.md §9 it is not promoted on anything: **the int8 weight path,
  which is exact by construction, is the production answer**, and a CER gate on
  `samples/` on the Linux box is what would settle whether `own` is better or
  worse here. One clip on a dev host is not that gate.

### Gate 4 — the server-stream phase, and what it really measured

`tests/test_server_stream.sh`, 4 real-time paced streams, run twice on `none`
and once on `accelerate`, on a box carrying another agent's full `make test`:

| build      | TTFP p50/p95 | emission lag p50/p95 | envelope | 4-stream identity |
|------------|--------------|----------------------|----------|-------------------|
| none       | 1359/2481 ms | 759/2943 ms          | INVALID  | **FAIL**          |
| none (2nd) | 1292/2383 ms | 612/1787 ms          | INVALID  | **FAIL**          |
| accelerate |  965/1029 ms | 159/ 242 ms          | GOOD     | OK                |

The identity failure is the French clip losing its leading word
(`"Bonjour, la réunion…"` -> `"La réunion…"`). Three facts place it:

1. the harness itself refuses the run — `envelope INVALID`, backlog max 1.74 s
   against a 0.64 s limit, and it prints `NOT BELIEVABLE` next to the diff;
2. on the SAME `none` build the `prefork-identity` and the `batched-identity`
   phases (8 streams, 5 clips, `--cap 8`) both passed byte-identical, and the
   CLI's own e2e French transcript is correct;
3. so the stream fell behind real time and was finalized before its first chunk
   had been served — a pacing failure, not an arithmetic one.

**But the cause of the pacing failure is ours**: our sgemm is materially slower
than Accelerate on this M1 — enough that four real-time streams stop being
streamable where Accelerate stays GOOD. That is the expected macOS result
(Accelerate reaches the AMX block; a portable NEON micro-kernel does not) and it
is stated plainly here rather than buried: **on macOS, Accelerate wins, and it
is not close.** It changes nothing about the decision, because the decision is
ownership on Linux against OpenBLAS, and no Linux number exists yet. It does
mean `BLAS=none` must not be made the macOS default, and it is not.

No RTF and no step-time A/B is reported at all: benchmarking on this host was
stopped by decision (loaded, low on RAM) and every timing is deferred to the
Linux box as S1-6a. The table above is a GATE that failed, kept because a
failing gate must be explained, not a measurement anyone may quote.

### Gate 2 — the self-test: see above. Gate 3 — the builds

`BLAS=none` links no cblas symbol (`nm -u` is empty of them) and the
`$(error)` on a missing `cblas.h` does not fire. `--dispatch-map` prints
`blas=own` / `gemm.f32 own` on that build and `blas=accelerate` on the other,
with two extra rows (`gemm.f32_kernel`, `gemm.f32_families`) that call the
predicate and read the counters rather than restating them. `BLAS=openblas`
cannot be built on this host (no OpenBLAS) — CI covers it.

### Gate 5 — `make check` and UBSan

`make check`: check_plan PASS, check_repo_integrity PASS, flag registry ok
(21 flags). `tests/test_streaming` built with `-fsanitize=undefined` (no
`-ffast-math`) and run on the Nemotron model: **no sanitizer diagnostic**,
transcripts IDENTICAL.

### Gate 6 — CI

`.github/workflows/ci.yml`: the Linux x86 and ARM matrix entries now build and
run the whole functional suite with `BLAS=none` (macOS with `accelerate`), each
job proving the provider from `--dispatch-map` before it trusts the build; a
new model-free `tests/test_sgemm` step runs before the model download; and a
separate `build-openblas` job keeps the comparison arm compiling and passing the
model-free self-tests forever.

## Conclusion

S1-6 is done as an OWNERSHIP change and is honest about being nothing else yet.
The f32 GEMM is ours, it is deterministic across thread counts by construction
and proven so with memcmp at 1/2/4/8 real threads, the seam is the only place in
the tree that knows what a BLAS is, `BLAS=none` is the Linux default and links
none, and `BLAS=openblas` stays as the arm every Linux A/B will be measured
against. Transcripts are unchanged except one already-unstable clip, named above.

**On speed this closes nothing.** Accelerate beats our sgemm on macOS by enough
to break a 4-stream real-time envelope, and the Linux question — the only one
that matters — has not been asked at all.

## Next action

S1-6a on the Linux box, with nothing else running on it:

1. `BLAS=openblas` vs `BLAS=none`, x86 and Axion: `mynah-asr bench` RTF and the
   `tests/test_stream_batch --steptime` table at B = 1,2,4,8, dispatch proven
   per run.
2. Thread count inside a prefork worker pinned to T cpus: confirm T, not 2T.
3. `tests/test_server_stream.sh` in both builds on a quiet box — the envelope
   comparison this dev host could not give.
4. The OpenBLAS f32 row-stability gate (`MYNAH_ASR_BATCH_F32=1`), which decides
   that default on the openblas arm (S1-8).
5. A CER gate on `samples/` for `own` vs `accelerate`/`openblas`, to settle the
   es-ES clip rather than leave it as an anecdote.
6. If the Linux numbers are bad, the cost-model constants written blind against
   the sibling's dispatch cost (`SG_PARALLEL_MIN_WORK`, `SG_PANEL_NV`,
   `SG_PANEL_FLOATS`) are the first things to sweep, and S1-5 (the spin-then-park
   pool with the meter) is the second.

---

## 2026-09-19 — why `own` was slow, and the default taken

The line above — "Accelerate beats our sgemm by enough to break a 4-stream
real-time envelope" — was true and was treated as a fact about AMX. It was a
fact about a defect.

### What was measured

A new instrument, because "which provider ships" is not a question about GEMM
in general but about the twenty-odd shapes this model issues:

- `MYNAH_ASR_GEMM_PROFILE=1` (src/backend.c) records every call through the f32
  seam per DISTINCT shape — counts and time — and dumps the table at exit.
  Model-, family- and quantisation-specific by construction.
- `tests/bench_gemm_shapes` replays such a table, or a representative one with
  `--demo`, against BOTH arms **interleaved in one process**: ours is compiled
  into every build whatever `BLAS=` linked, so `mynah_asr_sgemm_f32` and
  `mynah_asr_gemm_f32` are both callable, on the same operands, under the same
  thermal state and page cache. It checks the two agree numerically before it
  times them, and weights each shape by its call count when a profile is given.

M1, one thread, 77 representative shapes. The DOT family — which is every
`x @ W^T` in this runtime, so every f32 linear, the joint head and the
attention scores — ran at **8.7 GF/s on a core that does about 100**, flat
across every shape, which is the signature of a latency bound rather than a
bandwidth or a blocking problem.

### The cause

`sg_dot` accumulated into ONE vector accumulator and `sg_tile_dot` called it
once per output element. One FMA dependency chain: 4 lanes x 2 flops per ~4
cycle FMA latency at 3.2 GHz is 6.4 GF/s, against 8.7 measured. The kernel was
doing exactly what its structure allowed.

### The fix, and why it is not a numerics change

A register tile: `SG_DOT_MR x SG_DOT_NC_TILE` dot products at once (NEON 4x4,
AVX2 4x2), plus a `1 x SG_DOT_NC_WIDE` strip for the rows under MR — which is
the whole of a gemv, i.e. the RNNT prediction network, once per emitted token.
16 independent chains instead of one, and each loaded vector feeds several FMAs.

Each element still accumulates over the whole of k into one accumulator, in
SG_LANES steps, in the same order, folds with the same `sg_hadd` and finishes
with the same scalar tail. So the tiled answer is the SAME BYTES as the untiled
one, and `tiling_identity` in `tests/test_sgemm.c` proves it for DOT, PANEL and
NARROW by running the same GEMM one column at a time with the family forced on
both sides. That gate matters beyond tidiness: if the tile width could change an
element, a stream's transcript would depend on how many columns sat beside it,
which is rule 4 failing at the bottom of the stack.

`SG_PANEL_NV` also stopped being a choice ("the conservative end", 2 everywhere)
and became `SG_NV_MAX`, derived from the accumulator budget like the narrow
boundary. AVX2 does not move; NEON goes 2 -> 4.

### Result (M1 dev host, one thread, development signals)

| family / shape | before | after |
|---|---|---|
| DOT (`x @ W^T`, m = 4..256) | 8.7 GF/s | **75-80 GF/s** |
| PANEL (attention context, 26 shapes) | 48.9 GF/s mean | **74.8** |
| MATVEC-through-DOT (LSTM pred-net gemv) | 8.2 GF/s | **30.2** |

At 4 threads the DOT family reaches 285 GF/s. And the number that decides the
serving question: **at 4 to 16 stacked rows — exactly what a streaming step
issues — `own` is now FASTER than Accelerate** (ratios 0.40-0.93). Accelerate's
AMX only pays from m >= 16, which is the offline path.

The gemv stays 2.1x behind Accelerate and that is understood rather than open:
one row makes the kernel load-bound (9 loads per 8 FMAs), and the measured 30
GF/s is within a few percent of what that ratio allows on this core. A sweep of
the strip width (4/8/12/16) picked 8, which is what `SG_ACC_VECS / 2` already
gave.

### A second defect the same instrument found: the parallel threshold

`sg_plan_rows` went parallel when the GEMM had at least `SG_PARALLEL_MIN_WORK`
(131072) MACs **in total**, whatever the pool width. But a dispatch is not a
fixed cost: `mynah_asr_parallel_for` wakes `min(tasks, width)` workers and waits
for all of them, so the overhead grows with the width while the arithmetic per
worker shrinks. A flat threshold gets worse the more cores you have, which is
the opposite of its purpose.

Measured on the M1, the 49 representative shapes under 2M MACs — which is
exactly where a streaming step's attention GEMMs live (`4 x 128 x 320` is
163,840 MACs, just over the old threshold):

| MYNAH_ASR_THREADS | before | after |
|---|---|---|
| 1 | 318 us | 318 us |
| 2 | 339 | 324 |
| 4 | 385 | 286 |
| 8 | **630** | **330** |

Twice as slow for having eight cores, and a prefork worker pinned to 8 cpus is
precisely the configuration this server ships. The rule is now
`work >= threads * SG_PARALLEL_MIN_WORK`; the constant itself did not move, it
was being compared against the wrong side of the multiplication. Large shapes
are unaffected (all shapes: 178 ms at T=1, 55 at 4, 44 at 8 — still a 4x).

This one is worth re-deriving on the box rather than assuming: the dispatch cost
is per host, and 32 Neoverse cores is a much wider pool than 8 Firestorms.

### The decision

**`BLAS=none` is the Linux default again** (Makefile, 2026-09-19). It is a
decision, not a measurement, and it is stated as one — but it now rests on
something: the reason to doubt `own` was a defect that is fixed and gated, the
ownership argument was always the stronger one (63 threads in a worker pinned to
8 cpus with OpenBLAS linked, 32 without, measured on the Axion), and the docs
have said `BLAS=none` since S1-6 while only the Makefile disagreed — a
contradiction in the tree is worse than either answer.

S1-6a is now a VERIFICATION and costs two commands instead of a box-day:

```
make BLAS=openblas
MYNAH_ASR_GEMM_PROFILE=1 ./mynah-asr transcribe -m <model> tests/audio/test_it.wav 2> shapes.txt
tests/bench_gemm_shapes shapes.txt          # weighted total, both arms, one process
```

If the weighted total says OpenBLAS wins by enough to cost capacity, the
Makefile line goes back and this note records the number. That is what
reversible-by-a-number means.

---

## 2026-09-19 — S1-5: the pool spins before it parks

The third finding of the same instrument, and the other half of the parallel
threshold above. `sg_plan_rows` now refuses to give the pool work it cannot pay
for; this is about the work just ABOVE that line, which a stream step issues
dozens of times in a row.

A dispatch was a condvar broadcast to N sleeping threads plus a completion wake
back to the caller — two kernel round trips per GEMM, at a few microseconds
each, on GEMMs that take tens of microseconds. The workers now watch an atomic
generation counter for `MYNAH_ASR_POOL_SPIN_US` microseconds (default 50) before
parking, so a dispatch that follows another by a microsecond finds them hot.
`0` restores the pure condvar pool and is the arm this is measured against.

### Result (M1 dev host, `tests/bench_gemm_shapes --demo`)

The 49 representative shapes under 2M MACs — a streaming step's attention:

| threads | spin off | spin 50 us |
|---|---|---|
| 4 | 318 us | **219 us** (-31%) |
| 8 | 355 us | **285 us** (-20%) |

The large shapes do not move (57.6 vs 56.4 ms at T=4, 47.7 vs 49.8 at T=8, i.e.
inside the noise of a loaded dev host): as expected, a 4 ms GEMM does not care
about a 3 us wake-up.

Idle cost is bounded by construction: a worker burns at most the budget of its
own core per wait, once, then parks. Between two streaming chunks (320 ms apart
at lookahead 3) every worker is parked.

### The meter

`worker_spin_pct` and `caller_spin_pct` (via `mynah_asr_pool_stats_get`, printed
in the server's SIGUSR1 dump). A pool that silently stopped spinning and one
that spins and catches nothing look identical from the outside; near-zero under
load means the budget is too small for the host, near-100 on an idle server
means it is too large.

### What the A/B arm found on its first run

**The `SPIN_US=0` arm deadlocked on the first dispatch, every time.** Every
worker parked on the job condvar, the caller parked on the completion condvar,
the process at 0% cpu.

The cause was one line of the rewrite: the worker initialised its generation
from the live counter. `pool_init` runs under `pthread_once` INSIDE the first
dispatch, so a worker can reach its first read after that dispatch has already
published its job and counted the thread in `g_pending` — it then sees
`gen == seen`, parks, and waits for a job published before it looked, while the
caller waits for a count that will never reach zero. Generations only increase
and the first job is generation 1, so the correct start is 0: a thread the
dispatcher is already counting on must find work at its first check.

Two things worth keeping from that:

1. **The default hid it.** With a 50 us spin the worker catches the bump inside
   its spin window and the race never fires. The bug was reachable only through
   the flag that exists to make the comparison possible. A knob whose other
   position is never exercised is a knob that hides bugs — which is the argument
   for running every A/B in both directions, not only the one expected to win.
2. **A deadlock is not a failing assertion, it is a test that never returns.**
   `tests/test_threads.c` gates the first dispatch at three spin budgets in
   PRISTINE children (the flags are read once and cached, so a fresh process is
   the only honest way to set them) with the parent holding a deadline.

Writing that gate turned up a second hang, this one in the test rather than the
pool: a child forked from a process with a warm pool inherits `g_workers` but
not the worker threads, so it waits forever for threads that no longer exist.
That is exactly the contract `mynah_asr_threadpool_after_fork()` exists for and
exactly what the prefork server does — now gated too, with its failure mode on
record: a hang, not a crash.

---

## 2026-09-19 (later) — what flipping the default did to CI, two layers away

The ASan job had run in 6m47s for weeks. From the commit that made `BLAS=none`
the Linux default it timed out at 30 minutes, three times, and the timeout read
as `cancelled` rather than as a failure.

The chain:

1. `make asan` / `ubsan` / `debug` replace CFLAGS wholesale, and dropped
   `-march` with it.
2. `src/qmat.c` did not care: its kernels carry `__attribute__((target(...)))`
   and are chosen by a runtime probe, so they are compiled into every build.
   `src/sgemm.c` is gated on `__ARM_NEON` / `__AVX2__` instead, so with no
   `-march` on x86 it compiled its SCALAR fallback — about 2.5 GF/s against 80.
3. While OpenBLAS was the Linux default that cost nothing: OpenBLAS is a
   prebuilt library, vectorised, threaded, and NOT instrumented by ASan. The
   moment `own` became the default, every f32 GEMM in the sanitizer job went
   through a scalar kernel at `-O1` with ASan checking each access.

Two things are worth keeping from this beyond the fix.

**A default is not a local change.** The flip was argued on Linux production
grounds and was right there; its first real consequence landed in a CI job
nobody was looking at, through a Makefile target that had been silently wrong
since before `sgemm.c` existed.

**The symptom was self-camouflaging.** "Slower under a sanitizer" is exactly
what one expects, so a sanitizer job that got slower is the last place anyone
looks — and a 30-minute ceiling on a 7-minute job gave it room to look normal
for three runs.

Fixed in three places, each answering a different half:

- `SAN_MARCH` (default `-march=native`) on the three diagnostic targets, so the
  sanitizer checks the code production runs instead of a path production never
  executes. That is a COVERAGE fix; the speed is a side effect.
- `--dispatch-map` now reports a scalar sgemm on a vector host as
  `scalar DOWNGRADE` with the reason, instead of printing it as a resolved
  value. One line, and the next occurrence is visible from the binary.
- `timeout-minutes` 30 -> 12 on the ASan job. A healthy run is under 7; a run
  that doubles should fail, not queue.

### And the parallel threshold, corrected again

`work >= threads * SG_PARALLEL_MIN_WORK` fixed the pathology (a flat threshold
gets worse the wider the pool) but replaced it with a cliff: on a 32-core
Neoverse it would leave every GEMM under 4.2M MACs single-threaded, which is a
lot of the streaming step. The rule is now proportional — as many tasks as the
work can fill at `SG_PARALLEL_MIN_WORK` each, capped at `2 * threads`:

```
by_work = work / SG_PARALLEL_MIN_WORK      (capped at 2*threads)
want    = by_work > 1 ? by_work : 1
```

A GEMM worth three tasks gets three, on a four-core box and on a thirty-two-core
one alike. On the M1 the difference against the cliff version is inside the
noise of a loaded dev host (tiny shapes 254/295 us at T=4/8 against 219/285,
large shapes unchanged); it is chosen for the many-core case, which is the
target, and the constant itself still has to be re-derived on the box.

---

## 2026-09-19 (later still) — the gate that asserted more than the code promises

With `-march` restored, the ASan job went from timing out at 30 minutes to
FAILING in 3m21s, which is the point of a tight ceiling. The failure was real
and it was mine:

```
sgemm: provider=own isa=avx2 narrow_max=16
sgemm ok:   DOT: the tiled block is byte-identical to one column at a time
sgemm FAIL: PANEL: ... (213 of 333 elements differ, 9x37x77)
sgemm FAIL: NARROW: ... (53 of 117 elements differ, 9x13x77)
```

Reproduced on ARM in one command, which settled it without a CI round trip:
build the same test with `-ffp-contract=off` and it fails identically (240 of
333). So it is not AVX2 — it is FP CONTRACTION.

The mechanism: the vector micro-kernels accumulate with `sg_fma`, an
EXPLICITLY fused intrinsic. `sg_micro_tail`, which serves the last
`n % SG_LANES` columns, accumulates with `acc += av * b`, which is fused only
if the compiler contracts it — and that depends on the ISA (AVX2 does not imply
FMA) and on the flags. Production builds carry `-ffast-math`, so the two agree
there and the divergence never showed; the sanitizer builds do not, which is
exactly why they are worth running.

**The code is fine. The gate was wrong.** Both results are valid roundings of
the same sum, the choice is deterministic for a given shape and build, and
nothing in this runtime varies `n` for a call site: `n` is the output width of a
weight matrix. What the runtime DOES vary is `m` — a batched stream step stacks
B streams' rows into one GEMM — and the pool width.

So the gate now asks each family what the contract actually says:

- `tiling_identity` stays, for DOT ONLY, because that is the schedule change
  that had to be proven equal to the untiled dot — and it holds on every ISA
  under any contraction setting, since `sg_dot` and `sg_dot_tile` spell BOTH
  halves of the reduction identically (same `sg_fma` over the vector part, same
  scalar `sum += a[i]*b[i]` over the k remainder).
- `row_stability` is new and covers all three families: rows 0..3 must be the
  same bytes at `m = 4` and at `m = 9`. ISA- and contraction-independent by
  construction — the same rows go through the same kernel at both widths — and
  it is the thing rule 4 is about. If row blocking ever stopped rounding up to
  `SG_MR`, this is what catches it.

A comment in `sg_micro_tail` claimed its results were "consistent with the rest
of the row". The ORDER is the same; the ROUNDING is not guaranteed to be.
Corrected in place, because a comment that overstates a guarantee is how the
wrong gate gets written in the first place.

---

## 2026-09-19 (end of day) — what owning the GEMM costs the sanitizer

With `-march` restored and the gate corrected, the ASan job still hit its
ceiling, at the same place both times: `tests/test_batch`, after
`tests/test_encoder`, with no output for nine minutes.

**It is not a hang.** The pool was the obvious suspect, because `test_batch` is
the only test that nests — `src/encoder.c:1281` runs a `parallel_for` over the
batch segments and every sgemm inside each segment dispatches again. A
model-free stress harness (outer `parallel_for` over 4 segments, each doing
thousands of inner ones) ran 1.4M nested dispatches at widths 2/4/8 with the
spin both on and off: no hang, correct counts, every time. The pool is clear.

**It is the ownership decision, arriving.** The timestamps give the multiplier
directly: `tests/test_encoder` on the 110m takes about 0.2 s in the normal CI
and 5.6 s here — about 28x. `test_batch` does eight full forwards of a 17-layer
model, so twenty-odd seconds becomes nine minutes. The reason is not ASan being
slow in general: it is that **until this morning every f32 GEMM in that job went
into OpenBLAS, which is a prebuilt library that ASan does not instrument**. Now
they go into our code, which it does. Owning the GEMM means sanitizing the GEMM.

The honest options were: raise the ceiling (hides it), cut coverage (pays for
speed with the thing the job is for), or make the instrumented code faster. The
third one turned out to be available for free, because the ASan target was
built at `-O1` while the UBSan target beside it has always been `-O2`. Measured
model-free, our DOT family under ASan:

| build | own DOT mean |
|---|---|
| ASan `-O1` | 21.9 GF/s |
| ASan `-O2` | **60.4 GF/s** |
| production `-O3 -ffast-math`, no sanitizer | 172.1 GF/s |

2.76x, no coverage lost: `-fno-omit-frame-pointer` is what keeps the stack
traces readable and it was already there — `-O1` was buying nothing else. That
should put `test_batch` near three minutes and the job near six, which is where
it was before.

Worth writing down as a method point: **a default flip lands wherever the old
default was doing work for you, and a sanitizer job is exactly such a place.**
The first symptom was a 30-minute timeout that read as `cancelled`; the second,
after fixing `-march`, was the same timeout at 12 minutes. Neither said "the
GEMM moved". Only the ratio between two jobs running the same tests did.
