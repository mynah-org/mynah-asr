# S1 — the thread pool, the lane, and BLAS ownership

Status: OPEN

Task: S1-5, S1-6
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
  joint head go through `cblas_*` (src/backend.c, src/encoder.c, src/decoder.c,
  src/subsampling.c). Accelerate on macOS, OpenBLAS on Linux, hard `$(error)`
  without `cblas.h`.

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

Unknowns
- Whether the own sgemm at the stream-step shapes (m = B·Q ≤ 64, n = 1024/4096,
  k = 1024/4096) is within the OpenBLAS single-thread number; it needs the
  A/B before it becomes default. The int8 batched path (S5-1) is the real
  production kernel, so f32 only has to be *not embarrassing*.
- macOS: `parallel_for` P-core default is a macOS concept; on Linux the pool
  width is the worker's mask size.

Plan
- S1-5 copy `threads.{c,h}` with the `mynah_asr_` prefix; replace
  `mynah_asr_parallel_for`; delete `g_blas_budget`, the weak
  `openblas_set_num_threads` and `blas_set_concurrency`; register
  `after_fork`; run the litmus in `make test`. Gate: `make test` green,
  `make leaks`, `make ubsan`, meter report from a 60 s stream showing
  dispatches per chunk and barrier %.
- S1-6 copy `sgemm.{c,h}` and `kernels.{c,h}` (dot, layernorm, softmax, silu
  vectors); add `BLAS=none|openblas|accelerate` to the Makefile with `none`
  the Linux default; per-call-site switch behind `mynah_asr_gemm_wt`. Gate:
  self-test of every family against the reference on the stream-step shapes;
  transcripts byte-identical; A/B table f32 stream step OpenBLAS vs own at
  T=1,2,4 on the Axion.

Evidence / Conclusion / Next action: pending S0.
