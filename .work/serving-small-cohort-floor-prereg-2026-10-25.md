# Small-cohort floor (~14 ms/pass) on GB10: decomposition and one pre-registered lever (2026-10-25)

A desk study only. No GPU was used, nothing was built or committed. dgx2 files were read, and the existing nsys SQLite
was queried on the CPU (`nice -n 19`). Code: main **d947275** (a detached worktree in the scratchpad). The box run was
`~/srv-pr10` = dd3e7ec, with the same engine code.

## 0. Corrections to the brief (read first)

- **The model is Nemotron 3.5 streaming 0.6B, not EOU-120M.** Every CUDA number here comes from
  `nemotron-3.5-asr-streaming-0.6b` (24 layers, d = 1024, ffn = 4096, V = 13088, q = 4 rows/lane): the benches in
  `~/ft/dgx2_newmain_prof.sh` and the server A/B `[SERVER-CONFIG] model=...0.6b`. At main the CUDA engine cannot run
  EOU-120M (tests/gemm_shapes_bench.cu:41-43: "the CUDA engine at main refuses a pack without [a prompt projector]";
  enqueue_encoder always runs prompt_cat + pl1/pl2). For EOU-120M (~0.2 GB bf16, 17 layers) the floor would be
  launch-dominated. That is a different study.
- **The server already replays the encoder from CUDA graphs.** It is the default for own GEMMs without the profile
  (gpu/server/main.c:1515). The A/B logs show `graphs=55 ... label loop eager` and `graph_passes=797 eager_passes=0`.
  The 15.3 ms at C=8 comes from `test_cuda_stream --bench`, which runs **eager with the event stage profile**
  (banner "graphs off"). That bench has no graph mode (closure audit F2).
- **A 7-lane server cohort runs a 64-row graph, not a 32-row one.** RG = 64 (engine.cu:178): 28 real rows are
  padded to the (8 lanes, 64 rows) bucket. The bench point that matches it best is C=16 (64 rows), not C=8.

## 1. Decomposition

Sources. Stage ms: means of 3 rotated reps from `dgx2:~/ft/prof-newmain-20261024/bench-c{8,16,128}-r*.txt` (OBSERVED).
Launch counts: code at d947275, which matches the C=128 trace exactly (566 = 564 encoder + dec_begin + first compact;
21 per label-loop iteration). Runtime API, gaps and syncs: `dgx2:~/srv-pr10/build/gpu/gemm-prof/20261024T103722Z/
pass-c128.sqlite` (C=128 only; there is **no C=8 trace**; the kernel table holds one cold pass, while the runtime
table holds all 12 steps). Weight roofline: bf16 weight bytes / 230.1 GB/s (copy bandwidth measured by gemm-bench.txt).

### 1a. Launches and stage time per pass (bench, eager, profile on)

| stage | launches/pass C=8 (M=32, split-K two-kernel) | C=128 (M=512, serial reduce) | stage ms C=8 | C=16 | C=128 | bf16 weight roofline (any C<=32) |
|---|---|---|---|---|---|---|
| subsample (ss0, dw1, pw1, dw2, pw2, flatten, lin+red) | 8 | 7 | 0.59 | 0.64 | 2.92 | 0.04 |
| ffn1 (LN, w1+red, w2+red, res) x24 | 144 | 96 | 3.06 | 3.19 | 8.05 | 1.75 |
| att_proj (LN, k/v/q + 3 red) x24 | 168 | 96 | 1.62 | 1.70 | 4.15 | 0.66 |
| att_core (attention, kv_commit) x24 | 48 | 48 | 1.30 | 1.49 | 8.14 | 0 (f32 ring reads) |
| att_out (o+red, res) x24 | 72 | 48 | 0.53 | 0.59 | 1.28 | 0.22 |
| conv (LN, pw1+red, glu_dw, LN, pw2+red, res) x24 | 192 | 144 | 2.11 | 2.33 | 6.86 | 0.66 |
| ffn2 (LN, w1+red, w2+red, res, LN out) x24 | 168 | 120 | 3.44 | 3.84 | 9.08 | 1.75 |
| post (kv_advance, prompt_cat, pl1/pl2/encproj + red) | 8 | 5 | 0.09 | 0.10 | 0.22 | 0.04 |
| **encoder** | **808** | **564** | 12.75 | 13.88 | 40.70 | 5.12 |
| label loop: 21 launches x 9.56 iters + begin/compact | ~203 | ~203 | 2.57 (joint 0.99, pred 1.35, sync 0.24) | 2.65 | 3.53 | 1.28 (30.7 MB x 9.56) |
| **pass** | **~1011** | **~767** | **15.32** | **16.54** | **44.29** | **6.40** |

### 1b. Where the C=8 pass goes (15.3 ms)

| component | ms | share | status |
|---|---|---|---|
| GEMMs at the bf16-weight roofline (1.18 GB encoder + 0.29 GB label loop per pass) | 6.4 | 42 % | arithmetic from shapes |
| GEMMs below roofline (about 66 % of DRAM bw at M=32; gemm-bench shows 55-74 % at M=128) | ~3.3 | ~21 % | HYPOTHESIS (remainder model) |
| GPU idle gaps between kernels: ~1011 x 2.24 us | ~2.3 | ~15 % | gap OBSERVED (C=128 encoder: median 2.24 us, p90 3.3 us, idle 1.67 of 34.3 ms) x code counts = HYPOTHESIS at C=8 |
| non-GEMM kernel execution: attention+kv ~1.2, subsampling ~0.5, ~490 tiny glue kernels (254 split reduces, 96 residuals, ~140 decoder glue) ~0.9, LN x144 ~0.6, glu_dw ~0.2 | ~3.4 | ~22 % | HYPOTHESIS (stage residuals) |
| label-loop host-sync bubbles (10.6 syncs/pass) | 0.24 | 1.5 % | OBSERVED (dec_sync stage) |
| host launch API | hidden | 0 % | OBSERVED (below) |
| H2D + D2H | 0.01 | 0.1 % | OBSERVED |
| per-pass allocations | none | 0 | OBSERVED |

Host side (OBSERVED, C=128, nsys on). One pass enqueues 566 encoder launches in 1.9-2.0 ms (cudaLaunchKernel at
about 2.6 us each, plus 148 cudaEventRecord) and then waits 30-41 ms in the first cudaStreamSynchronize. Each
label-loop iteration costs 21 launches, 1 cudaMemcpyAsync D2H (4 B n_active) and 1 cudaStreamSynchronize: about
0.08 ms of host time and a 0.20-0.33 ms wait, with ~0.35 ms between iterations. Each pass has 1 + iters + 1 stream
syncs (11-12 at C=128) and no cudaDeviceSynchronize. The steady state has no cudaMalloc, cudaMemcpy or host alloc: the
931 cudaMalloc and 633 cudaMemcpy all happen at load. The bench's ~21 ms of host mel between passes lies outside
device_ms. In the server, graphs bring submit down to 0.07 ms/step (DUMP `submit_ms_mean`).

At C=8 the host launch cost is still hidden (HYPOTHESIS): 808 x 2.6 us = 2.1 ms of enqueue against ~12.8 ms of GPU
encoder time.

**Conclusion.** The floor is GPU execution. Per pass, ~1.5 GB of bf16 weights are streamed by GEMMs running at
about two thirds of DRAM bandwidth, alongside ~1000 small kernels. Launch API time and host syncs are not the floor.
Cross-check (OBSERVED numbers, HYPOTHESIS reading): the graph-on server step at 6.9 lanes is 14.9-15.0 ms
(`pvfused-server/server-C064-*.log`, 64-row graphs, 6.35 decode iters per cohort), and 15.8-16.3 ms at 10.3 lanes.
The eager bench gives 16.5 ms at C=16 with 9.56 iters. Corrected for decoder iterations at 0.278 ms each, the encoder
graph saves at most ~0.85 ms per pass.

## 2. CUDA graph feasibility

- **Static and already captured:** the whole encoder (ss through projector) per (lane bucket {8,16,32,64,128,Bmax}
  x 64-row step). That is 55 execs, 143 ms of capture and a 395 ms warm-up. The capture is safe because nothing is
  allocated after open, padding lanes go to the scratch slot, padding rows are row-stable, and cuBLAS and the event
  profile are excluded.
- **What breaks capture of the rest:** the label loop. Its trip count depends on the data (`n_active` goes D2H, then
  the host syncs and branches on `n <= 0`). Its GEMM M = n changes every iteration, which changes the grids and the
  reduce mode. The fix would be a conditional WHILE node (dec_compact sets the handle on the device), running every
  iteration at a fixed M = Bk with masked lanes. Row-stability keeps the bytes the same.
- **Remaining upper bound in the server:** label-loop gaps ~135 kernels x ~1.2 us saved, plus 7.35 sync bubbles x
  ~22 us, gives about 0.3 ms per step at 6.35 iterations (2 %). In the bench at C=8: 203 x 1.2 us + 0.24 ms = about
  0.48 ms (3 %). The encoder graph's gain (<= ~1 ms: 808 x (2.24 - ~1) us) is **already banked** in the server.
  Fusing the 254 split reduces, 96 residuals and the LN pairs (A12/C1) would save ~3-4 us per removed launch, about
  0.3-1 ms in total.

## 3. Pre-registered lever: small-M own-tc GEMM with a cp.async W pipeline (one variable)

**Why this one.** It targets the largest pool that can be compressed while staying byte-identical: the ~3.3 ms of
GEMM time below roofline at M = 32-64, which is about 21 % of the pass and the very M the server graphs run. cuBLAS
bf16 shows that the headroom can be reached on these exact shapes (C=32: 9.66 vs 12.34 ms GEMM per pass, -22 %,
gemm-bench.txt). The runners-up have smaller upper bounds: E2 blocked label loop ~1.0 ms (already GPU-READY as G-E2),
RG = 32 row buckets for cohorts of 8 lanes or fewer ~0.5-0.8 ms, label-loop graph ~0.3-0.5 ms, A12 ~0.3 ms.

**Upper bound.** At C=8, GEMM busy time drops from ~9.7 ms to 6.4 ms at roofline, a saving of 3.3 ms (-21 %).
Realistic target, at cuBLAS-level 85 % of roofline: 6.4 / 0.85 = 7.5 ms, saving ~2.2 ms (-14 % of 15.3 ms). At C=16,
on the 64-row server graph, it should be about the same in ms.

**Hypothesis.** At M <= 64 the own-tc kernel (2-4 warps per block, ~2.7 blocks/SM, one register prefetch, two
__syncthreads per BK=32 slab) cannot keep enough W bytes in flight. With a 3-4-stage shared-memory ring for W
slabs, filled by cp.async, the same GEMMs should run at >= 80 % of DRAM bandwidth.

**Exact change (not implemented).** In gpu/cuda/gemm.cu, a small-M variant of gemm_tc_kernel that tc_choose selects
only for M <= 64 (today's TC_32x64 / TC_64x64 paths):
- W slabs are copied global to shared with `cp.async.cg` 16 B, NSTAGE = 3 (4 as a forced test config), with
  `cp.async.wait_group(NSTAGE-2)`;
- the A path stays as it is (register fetch, then tc_bf16_bits in stash);
- S = k_gemm_tc_splits(N, K), KC, BK = 32, the slab order, the wmma load/mma_sync sequence, the partial write and
  gemm_reduce_kernel stay **unchanged**, so every element's chain is identical (byte-identical by construction);
- a `k_gemm_tc_force_config` entry exposes the variant for tests.

M > 64, the engine call sites, graphs and the label loop are untouched.

**Baseline / patch.** A = main d947275. B = d947275 + this change only. Both are built on dgx2 in separate trees
(`make -C gpu test-stream all`, tests/gemm_shapes_bench, tests/test_cuda_kernels).

**Method and gates (in order; stop at the first FAIL):**
0. **Kill probe, before any code (GPU ~2 min).** On A:
   `tests/gemm_shapes_bench --model nemotron --lanes 8,16 --q 4 --dec-iters 10 --reps 20`.
   PROCEED only if, at lanes 8 and at lanes 16, the summed own-tc minus cuBLAS time is >= 1.0 ms per pass, and
   own-tc is <= 75 % of roofline. Otherwise the pool cannot be reached: KILL the lever and fall back to G-E2.
1. **Parity (GPU ~5 min).** test_cuda_kernels compares the variant byte for byte (memcmp) against the current path:
   9 shapes x M in {1, 7, 8, 28, 32, 33, 63, 64} x act {0,1,2} x accumulate {0,1} x vec on/off, for NSTAGE 3 and 4.
   Then test_cuda_stream gates A/B/C (`--precision bf16 --gemm own-tc`). Any mismatch is a REJECT.
2. **Kernel A/B (GPU ~2 min).** gemm_shapes_bench on A vs B at lanes 8, 16, 32. B must cut summed GEMM ms/pass by
   >= 10 % at lanes 8 and lanes 16, and every shape must be within +3 %.
3. **Device ABBA (GPU ~5 min, 12 runs at ~10 s plus cool-down).**
   `PACK=$M LEVELS="8 16 128" PRECISION=bf16 GEMM=own-tc STEPS=40 A_LABEL=main B_LABEL=cpasync gpu/tools/ab/ab_device.sh <A>/tests/test_cuda_stream <B>/tests/test_cuda_stream`.
   Primary: device_ms_per_pass at C=8 and C=16.
   - **ADOPT**: C=8 <= -6 % and C=16 <= -5 %, with both mirror pairs agreeing in sign and each <= -4 %, and C=128
     within +1 %.
   - **NULL**: C=8 or C=16 within +/-3 %.
   - **HARM**: any level > +2 %.
4. **Server ABBA (GPU ~41 min)** for an ADOPT only: `ab_server.sh` with LEVELS "16 64", COMMON_ARGS "--gemm own-tc",
   graphs on (default). Gates:
   - transcript identity PASS on 23 clips;
   - integrity PASS;
   - `step_wall_ms_mean` (DUMP) <= -5 % at both levels;
   - lag p50/p95 no worse than +5 ms.

Total GPU time is ~55 min: ~2 for the probe, then ~12 at device level and ~41 for the server. The probe alone
decides whether the implementation (estimated at 1-2 days) is worth starting.

**What it can show.** It can show whether the small-M GEMM path is latency-bound, and how much a deeper W pipeline
recovers, in bench and server step time, while keeping the bytes identical.

**What it cannot show.**
- Why the path is slow, if it is: ncu is unavailable without root, and the probe only bounds the headroom.
- Server lag p95 gains: lag p95 is ~64-70 ms, of which ~40 ms is the cohort window, so a 1.5-2 ms step gain sits
  below the harness's 5 ms threshold. Expect NULL on lag, and judge on step_wall plus identity.
- A graph-on bench: the device bench is eager with the profile on. The server run is the graph-on transfer test.
- Anything about EOU-120M.

**Blockers.**
- Both DGX GPUs are training. Every step above waits for ~/gpu.lock.
- No ncu counters.
- The lever needs implementing before steps 1-4.

## Implementation gate (2026-10-26)

Gate 0 ran on 2026-10-25 on dgx2: PROCEED (own-tc minus cuBLAS 1.37 ms/pass at lanes 8, 1.58 ms at lanes 16; own-tc
at 70 / 69 % of roofline). Implementation starts 2026-10-26 ~11:00 CET on branch `feat/smallm-cpasync` from main
d947275 (local only, not pushed). Gates restated with no threshold change:

1. **Parity.** test_cuda_kernels memcmp of the cp.async variant against the current path: 9 shapes x M in
   {1, 7, 8, 28, 32, 33, 63, 64} x act {0,1,2} x accumulate {0,1} x vec on/off, NSTAGE 3 and 4; then test_cuda_stream
   gates A/B/C (`--precision bf16 --gemm own-tc`). Any mismatch = REJECT.
2. **Kernel A/B.** `tests/gemm_shapes_bench --model nemotron --lanes 8,16,32 --q 4 --dec-iters 10 --reps 20`, A vs B,
   ABBA. PASS iff B cuts summed own-tc GEMM ms/pass by >= 10 % at lanes 8 AND lanes 16, and every shape is within
   +3 %. Otherwise NULL/HARM and STOP; the only variation allowed is NSTAGE 3 vs 4.

Gate 3 (device ABBA) only after a gate-2 PASS; gate 4 (server) is out of this session's scope.
**Time-box:** if the prototype has not reached gate 2 by ~14:00 CET (3 h wall), stop and report where it is.

**Outcome (2026-10-26, dgx2).** Branch feat/smallm-cpasync c71cdf0 + 24ac5fc (local, not pushed). Gate 1 PASS
(4320 memcmp calls 0 mismatches; stream gates A/B/C PASS, also with --pass-lanes 2). Gate 2 FAIL = **HARM**:
summed own-tc GEMM ms/pass A -> B, NSTAGE 3: lanes 8 9.80 -> 11.06 (+12.9 %), lanes 16 10.44 -> 11.75 (+12.5 %),
lanes 32 +4.1 %; NSTAGE 4: +12.7 % / +15.2 % / +3.1 %. Wide shapes (ff_w1/ff_w2, head, ss_lin) +15..33 %; narrow
1024x1024 shapes within noise. Lever closed; gates 3-4 not run. Raw: HF results/dgx-2026-10-26/serving-cpasync.
