# CUDA serving loop: stage-ahead, feed team, graph buckets, warm-up (S14-13, S14-14)

Status: IN PROGRESS — implemented behind flags (default off), identity gates green
on an L4 2026-10-08; A/B screens at the knee not run yet.

## Task

S14-13: hide the engine thread's host work (ring copy + streaming mel) under the
encoder pass (`--stage-ahead 1`), and spread the host mel of a staged batch over
a small feed team on hosts with spare CPUs (`--host-threads N|auto`).
S14-14: replay the encoder pass from CUDA graphs captured at start-up per lane
bucket x row bucket (`--graphs buckets`), and warm the server with traffic-shaped
steps before it listens (`--warmup 1`).

## Question

Can the host work and the ~700 launches per encoder pass leave the critical path
without changing one byte of any transcript, and what do start-up time and VRAM
cost?

## Known facts (before)

- L4, F32 v1, C=128: engine thread 86 % in step, host work ~14 % of wall between
  device passes, no overlap (`cuda-batched-streaming-server.md`, S14-6b).
- Every per-lane kernel reads its geometry from a `gpu_row` descriptor and loops
  over the lane's own `q` / `to[]` / `n_mel`; every per-row kernel and GEMM is
  independent per row; the GEMMs are row-stable in M by construction (contract 4,
  the `test_cuda_kernels` byte gate).
- Nothing is allocated after open (no pooled buffer that a short warm-up could
  under-size, unlike the sibling TTS server).

## Design

- **Split step.** `asr_engine_step_submit` decides the lanes, takes every chunk
  out of its slot (mel consumed, `first` cleared, the delta's `t1` and the finish
  flag fixed), stages each pass into its own pinned slab (`h_rows`/`h_mel` x
  passes) and enqueues H2D + encoder of the first pass; `_finish` runs the label
  loops, waits and detokenises. `step == submit + finish`.
- **Stage-ahead scan** (server): between submit and finish, every active slot
  whose ring holds the whole next chunk has it taken and fed, in one batch.
  It takes what the regular scan would take (the engine's own need, one chunk)
  and skips slots with a cancel, reset or finalize pending, so each slot's feed
  sequence is unchanged. The mel stream is partition-invariant anyway.
- **Feed team.** Persistent helpers, atomic claim per slot, condition-variable
  park, no CUDA call in the body; inline below 4 slots. `auto`: affinity mask
  capped by the cgroup quota; 1 up to 8 CPUs, 2 up to 16, 4 above.
- **Graphs.** One capture per (lane bucket in `--graph-buckets`, default
  8,16,32,64,128 plus Bmax) x (row bucket, step 64 = the GEMM's M tile, up to
  min(Bk x qmax, Rmax)). Padding lanes: q = 0 on one extra scratch slot (index
  cap), so the per-lane kernels do nothing for them. Padding rows/positions: no
  lane owns them; P1 <= (2R + B) x Fo1 and P2 = R x Fo2 bound the subsampling
  GEMMs. A pass not covered falls back to eager (counted). The label loop stays
  eager (host-synchronised per iteration; a conditional-node loop is S14-8).
  Refused with `--gemm cublas` and `--profile-stages`.
- **Warm-up.** First, steady and tail chunks of the default preset at every
  lane bucket through the normal step (graphs included) on the idle slots, then
  every slot reset (host and device, scratch slot included) and the counters
  zeroed.

## Acceptance gate (stated before the runs)

1. `tests/test_cuda_stream` gates A and B unchanged, and gate C: each option
   alone and all together reproduce, per clip, the default engine's per-step
   output (text, t0/t1, token count, finish), alone and in a mixed cohort; a
   graph arm must replay graphs.
2. Server: a C=64 WAVE over 64 bank clips per arm; every delta (text and
   audio_s) of every clip identical to the flags-off arm; no errors.
3. No new data race: TSAN on the test with the feed team (host side).

## Evidence (L4, sm_89, CUDA 12.8, own GEMM v1 unless stated; dirty-tree WAVE = DIAGNOSTIC)

- Gate C: 5 clips x 5 languages, all five arms OK (85 passes each; the graph arms
  85/85 from graphs, buckets 1,2,4,6 so padded lanes and padded rows both occur).
  The same with `--gemm splitk` (graphs capture the split-K partial + reduce
  launches and their workspace): all five arms OK, gates A and B green too.
- Server identity, C=64 WAVE, 64 clips of `samples/stress-en` (every 7th):
  stage-ahead, stage-ahead + team of 4, graphs + warm-up, all together: 64/64
  clips identical in every delta's text and audio_s; 0 errors.
- Start-up (open to listening), cap 192: 0.65 s flags off; 1.93 s with graphs +
  warm-up (55 graph execs captured and uploaded in 0.19 s, warm-up 1.06 s).
- VRAM of the process at ready: 5694 MiB off, 5938 MiB with graphs (+244 MiB:
  12 MiB scratch slot, the rest graph executables).
- C=64 sanity (light load, GPU ~33 % busy, 40-49 W, step wall 58-64 ms, lag p95
  128-136 ms in every arm): graphs take the encoder enqueue off the host (submit
  2.5 -> 0.1 ms per step); at this load few chunks arrive inside the window
  (stage-ahead staged 174 chunks in 159 windows), so no latency effect is
  expected or seen. The A/B that can credit or reject these is at the knee.

- TSAN: not run. The box's kernel uses 32 random mmap bits and the container may
  not change its personality (`setarch -R` refused), so gcc 13's TSAN aborts at
  start-up. Race-freedom rests on the design: each slot's host state is touched
  by exactly one team thread per batch (atomic claim, distinct slots), the job
  is published and the results collected under the team mutex, the helpers make
  no CUDA call, and the engine thread is the only caller of every other op.
  Owed: TSAN on a host that allows it (clang >= 18 or ASLR off).

## Unknowns

- The effect at C=128..160 (back-to-back cohorts), with and without split-K /
  bf16: the screen belongs to the A/B campaign, ABBA, profile off.
- The first-request cost without a warm-up (lazy module loading) was not
  measured in isolation.
- Graph VRAM per exec and capture time grow with the bucket count; a coarser
  row step (128) halves both.

## Next action

ABBA WAVE screens at the knee (C=128/144/160) for `--stage-ahead 1`,
`--host-threads auto`, `--graphs buckets --warmup 1`, then the decision per flag.
