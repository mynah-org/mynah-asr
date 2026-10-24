# Serving speedup closure audit against current main (2026-10-24)

Scope: every inference/serving speedup idea collected so far for the CUDA streaming engine (Nemotron 3.5 streaming
0.6B, bf16 own-tc, GB10 sm_121). Each idea gets one status, checked against current main, with the evidence cited.
A desk audit: no GPU was used and nothing was committed. Box files were only read (cat/ls/grep). The two nsys
SQLite exports were copied to the Mac and queried there.

## 0. Baseline and sources

- **origin/main = d947275**: #5 int8 leaf accounting, #6 attention tiles from pack.qmax, #7 register-blocked score/P.V,
  #8 fused own-tc split-K reduce, #9 A/B harness, and **#10 gemm_shapes_bench + gemm_profile.sh** (merged after this
  audit was requested; the brief still listed #10 as open). The engine code is unchanged since c00b27f.
- Today's profile: `dgx2:~/ft/prof-newmain-20261024/`. The box worktree `~/srv-pr10` is at dd3e7ec ("Add an own-tc
  GEMM benchmark...": the #10 content under a different SHA than the local d057b74; the engine is c00b27f per
  `dgx2_newmain_prof.sh`). Phases P1-P4 ran 12:35-12:37 CEST and the log ended with PROF-DONE at 12:37:55.
- Local notes: serving-gb10-2026-10-23.md, restart-2026-10-24.md, cuda-batched-streaming-server.md (S14),
  cuda-stage-ahead-graphs.md, cuda-quant.md, l40s-2026-10-08-asr-cuda.md, nvidia-asr-streaming-landscape-20260923.md
  (S13-1c/1e), cache-ring-1.md, serving-v2-design.md, serving-audit-metrics-tests-hotpath.md, multi-model-serving.md,
  sibling-wins-and-flags.md.
- Branches: feat/serving-labelloop 12e3bfa, feat/serving-c-release 0711ddb, feat/serving-a-dwconv-grid 5792343,
  tools/gemm-real-shapes (now on main).
- Web research, done by a sub-agent from WebFetch summaries. Numbers marked UNVERIFIED there stay unverified here.

## 1. Current-main profile (bf16 own-tc, graphs OFF in the bench, 3 rotated reps, ms per pass, share of the device pass)

| stage | C=32 | C=64 | C=128 | C=192 (2 passes x 96 lanes) |
|---|---|---|---|---|
| subsample | 0.92 (4.4 %) | 1.45 (4.9 %) | 2.92 (6.6 %) | 2.25 (6.2 %) |
| ffn1 | 3.96 (18.9 %) | 5.60 (19.1 %) | 8.05 (18.2 %) | 6.72 (18.4 %) |
| att_proj | 1.91 (9.1 %) | 2.89 (9.9 %) | 4.15 (9.4 %) | 3.48 (9.5 %) |
| att_core | 2.85 (13.6 %) | 4.73 (16.1 %) | 8.14 (18.4 %) | 6.36 (17.4 %) |
| att_out | 0.75 (3.6 %) | 0.90 (3.1 %) | 1.28 (2.9 %) | 1.06 (2.9 %) |
| conv | 2.91 (13.9 %) | 3.92 (13.4 %) | 6.86 (15.5 %) | 5.43 (14.8 %) |
| ffn2 | 4.69 (22.4 %) | 6.66 (22.7 %) | 9.08 (20.5 %) | 7.81 (21.3 %) |
| post | 0.11 (0.5 %) | 0.16 (0.5 %) | 0.22 (0.5 %) | 0.18 (0.5 %) |
| dec_joint + dec_pred + dec_sync | 2.81 (13.4 %) | 3.00 (10.2 %) | 3.53 (7.9 %) | 3.26 (8.9 %) |
| h2d + d2h | 0.02 (0.1 %) | 0.02 (0.1 %) | 0.06 (0.1 %) | 0.05 (0.1 %) |
| **device pass** | **20.93** [20.73-21.24] | **29.34** [29.13-29.48] | **44.29** [44.09-44.60] | **36.60** (step wall 73.39) |

The reps agree to about 1.5 % or better. decode_iters_per_pass is 9.56 at every C.

**Kernel-level view (nsys, C=128).** The run is valid only as a partial capture. The SQLite export holds 904 kernel
records against 28,238 CUDA events. Those records are the load-time `k_f32_to_bf16_kernel` weight conversion
(252 launches, 18.2 ms, engine.cu:589, **not per pass**) followed by exactly one cold first pass. The C=192 export has
no kernel table at all. The second `nsys stats` call printed a usage error. Per-kernel costs that depend only on shape
are still usable, and each one is confirmed by the steady-state stage residual (stage ms minus GEMM ms):

| kernel per pass (C=128, 512 rows) | launches | us each | ms / pass | share | steady-state cross-check |
|---|---|---|---|---|---|
| layernorm_kernel | 144 | 44.6 | 6.42 | 14.5 % | (ffn2 - ffn1 residual) / 24 = 44 us; att_proj residual / 24 = 41 us |
| residual_kernel | 96 | 10.2 | 0.98 | 2.2 % | -- |
| glu_dwconv_kernel | 24 | 53.6 | 1.29 | 2.9 % | conv residual 3.9 ms = 2 LN + glu + residual |
| ss_dw_kernel (stage 1, stage 2) | 2 | 1039 / 339 | 1.38 | 3.1 % | subsample 2.92 = 0.54 stage0 + 1.38 dw + 0.53 GEMM + ... |
| ss_stage0_kernel | 1 | 542 | 0.54 | 1.2 % | -- |
| attention_kernel | 24 | 88 | (2.1) | -- | **not usable**: the cold pass has an empty ring (slots_reset just before); steady att_core is 8.14 ms |
| gemm_tc_kernel (all) | 270 | -- | 20.2 | -- | gemm_shapes_bench 23.4 ms (weights cold) |

**GEMM vs ceilings (gemm_shapes_bench, weights cold).** Totals are own-tc / cuBLAS bf16 / roofline (230 GB/s,
62 TF ASSUMED). C=32: 12.34 / 9.66 / 8.24 ms (own-tc at 67 % of roofline). C=128: 23.40 / 14.76 / 14.40 ms (62 %).
C=192 (one 192-lane pass): 30.06 / 18.85 / 18.50 ms. The GEMMs are about 53 % of the C=128 pass and 59 % of the
C=32 pass. The gap to cuBLAS is 8.6 ms (19.5 %) at C=128 and 2.7 ms (12.8 %) at C=32.

**ncu: no data.** Every gemm_tc_kernel and cuBLAS profile failed with `ERR_NVGPUCTRPERM` (no access to the
performance counters as a normal user). The reduce-kernel runs reported "No kernels were profiled", which **confirms
that the serial in-register reduce of #8 is taken at M = 512**. Every ncu-dependent item stays blocked until counter
access is granted. That needs root: run `sudo ncu`, or set the `NVreg_RestrictProfilingToAdminUsers=0` module option
and reload the driver.

**The serving regime is far below the bench's C.** In yesterday's server A/B (`dgx2:~/ft/ab-res/pvfused-server`,
[DUMP] lines), cohorts averaged **lanes_mean 6.9-7.2 at C=64 and 10.1-10.4 at C=192**, with step_wall_ms_mean
14.9-16.3, cohort wait 40.6-40.9 ms and about 6.5 decode iterations per cohort. A real pass is therefore about 28-42
rows, below today's smallest bench point (C=32 = 128 rows). It sits in the weight-walk and launch-latency regime:
the bf16 weights are about 1.2 GB, which needs about 5.2 ms at 230 GB/s, against a 15-16 ms step. About 40 ms of the
roughly 70 ms lag p95 is the cohort window, not the pass. **The ranking below therefore weighs C=32 as heavily as
C=128, and the first GPU action is a C=8/16 profile (G0).**

## 2. Closure matrix

Status vocabulary: MERGED / MEASURED-POSITIVE-NOT-MERGED / MEASURED-NEGATIVE / PARITY-RISK-REJECTED / ALREADY-PRESENT /
NOT-TRANSFERABLE / UNTESTED / SUPERSEDED. The coordinator's in-flight item is marked IN-PROGRESS. "Opt-in" means the
code is on main with the flag off by default.

### A. GEMM (own-tc, layouts, epilogues, split-K)

| id | idea | status | evidence |
|---|---|---|---|
| A1 | Fixed-order bf16 tensor-core GEMM (own-tc), default on sm_80+ | MERGED | 999a985, #4 3449ac5; bank 482/498 identical, WER 0.14483 -> 0.14469 (cuda-batched S14-8c); L40S knee moved from <C416 to >C512 |
| A2 | Split-K f32 with S = f(N, K), fixed-order reduce | SUPERSEDED | S14-8b, gates green on L4; replaced as the default by own-tc (A1) |
| A3 | GEMM v2 (re-tiled v1 chain, double buffering) | MEASURED-NEGATIVE | S14-8a: best tile +10-15 % on a few shapes, loses on most; kept as the `--gemm own-v2` arm |
| A4 | cuBLAS as the default GEMM | PARITY-RISK-REJECTED | S14-6b: measured not row-stable on all 12 shapes from M=1 (contract 4) |
| A5 | Pinned cuBLASLt algorithm with measured row stability | UNTESTED | S14-6b names it as an option; never built. Numerical change vs own-tc |
| A6 | Fused split reduce, serial in-register chains at M >= 256 | MERGED | #8 c00b27f: C=128 -17.6 %, C=64 -5.2 %, C=32 neutral; today's ncu shows no reduce kernel at M=512 |
| A7 | Fused reduce by last-block arrival (TC_RED_LAST) | MEASURED-NEGATIVE | serving-gb10-2026-10-23: slower than two-kernel at every M (C=32 +9 %); reachable by force only |
| A8 | Forced serial reduce at small M | MEASURED-NEGATIVE | serving-gb10-2026-10-23: C=32 +5.6 % -> rule M >= 256 (gemm.cu TC_SERIAL_MIN_M) |
| A9 | own-tc2: multi-stage cp.async pipeline, ldmatrix + mma.sync m16n8k16, BK=64, tile-major prepacked W (same per-element chains) | UNTESTED | restart sheet: "needs an ncu profile first"; ncu blocked today (ERR_NVGPUCTRPERM). Current kernel: wmma, single register prefetch, BK=32, 2 syncthreads per slab (gemm.cu:517-640) |
| A10 | Stream-K / persistent tile scheduler with M-independent K chunks (fixes the small-tile-count tail, e.g. ff_w2 N=1024: 64-128 CTAs on 48 SMs) | UNTESTED | numerics unchanged if the chunk boundaries and combine order stay f(N, K); never built |
| A11 | Merge q/k/v into one N=3072 GEMM with S pinned to the per-matrix S=4 | UNTESTED | engine.cu:986-988 issues three GEMMs on the same xn. Plain concatenation changes S (k_gemm_tc_splits(3072,1024)=3 vs 4), which breaks byte identity, so S must be pinned |
| A12 | Fold the residual add into the GEMM epilogue (o, ff_w2 x2, pw2) | UNTESTED | gemm_epilogue already computes `v + crow[n]`; alpha 0.5 is an exact scaling, so this is byte-identical by construction apart from subnormals. Removes 96 residual launches |
| A13 | bf16 activation hand-off (producer epilogue writes bf16 with the same RNE `tc_bf16_bits`; consumer GEMM reads bf16 A) for ff_w1 -> ff_w2 and LN -> GEMM | UNTESTED | own-tc rounds f32 A to bf16 at staging (gemm.cu `stash`), so a bf16 input produces identical values. Halves the 512x4096 f32 FFN hidden round trip |
| A14 | int8 weights decoded to bf16 in the own-tc tile load (w8a16, row scale in the epilogue) | UNTESTED | cuda-quant.md "durable form"; `--weights int8` + own-tc is refused today |
| A15 | int8 weights on the f32 v1 path (`--weights int8`) | MERGED (opt-in) | ed0c000; L4 -8.5..-10.7 % vs f32 own, WER -0.0015, 167/227 identical; never compared against bf16 own-tc |
| A16 | LSTM wih/whh pair as one dual-chain kernel (predictor-state grouping) | UNTESTED | engine.cu:1142-1143, two GEMMs + accumulate per layer per iteration; byte-identical only if `(Px0+Px1)+b` then `+(Ph0+Ph1)` is kept. A single K=2H chain would be a numerical change |
| A17 | Tile chooser per call (64x128 / 64x64 / 32x64), bit-neutral | ALREADY-PRESENT | gemm.cu tc_choose |
| A18 | Drop the resident f32 weight copy (+~1.2 GB) | UNTESTED | VRAM only, no pass gain; irrelevant on GB10 unified memory |

### B. Attention, K/V and state cache

| id | idea | status | evidence |
|---|---|---|---|
| B1 | Attention tiles sized from pack.qmax | MERGED | #6 b842432: att_core 16.9 -> 11.0 ms C=128; server fin p99 -10/-20 ms |
| B2 | Register-blocked score and P.V loops (order-preserving) | MERGED | #7 0aa18fd: att_core 11.0 -> 8.8 ms; C=192 lag p95 88.5 -> 78.4 ms |
| B3 | Cache projected K/V (not the pre-projection input) | ALREADY-PRESENT | engine design; NeMo-Speech.cpp independently does the same (S13-1c); parakeet.cpp caches the input |
| B4 | Ring head instead of a memmove of the saturated cache | ALREADY-PRESENT | GPU: `kv_advance_kernel`, "rel-pos table, ring window" banner. (CPU CACHE-RING-1: ring NEUTRAL, slide small win) |
| B5 | K/V ring int8 as a SPEED lever on the L4 / old kernel | MEASURED-NEGATIVE | cuda-quant.md: -1.1 % (C=64), 0.0 % (C=128); attention was latency-bound then. VRAM -69 % per slot (that part is positive) |
| B6 | K/V ring bf16 / int8 on GB10 current main (after #6/#7) | UNTESTED | flag on main (`--kv-dtype`). Gates A/B/C PASS with kv bf16 (L40S note). L4 bank: bf16 227/227 identical, int8 224/227. Model: the ring is 11.0 MB/lane/pass; at C=128 that is 1.41 GB in 8.14 ms = 173 GB/s, 75 % of copy bandwidth, i.e. now plausibly bandwidth-bound |
| B7 | Attention on tensor cores (bf16 QK^T / PV) | UNTESTED | numerical change in an f32 kernel whose loops are order-preserving by design; high parity risk; no plan |
| B8 | Conv cache / attention cache per slot on device, heterogeneous cohorts (cache-aware FastConformer execution) | ALREADY-PRESENT | slot arena + gpu_row geometry; gate C covers first/steady/tail chunks in mixed cohorts |
| B9 | NeMo #16310-type batch dependence (short final chunk) | ALREADY-PRESENT (as a guarantee) | gate A requires alone == cohort per clip; the issue is a NeMo bug, not a lever |

### C. LayerNorm, elementwise, fusion, launch count

| id | idea | status | evidence |
|---|---|---|---|
| C1 | Byte-identical LayerNorm restructure (same double DAG, warp-level tree; optional ln_out + next ln_ff1 pairing) | **IN-PROGRESS** (coordinator, feat/serving-layernorm) | 144 x 44.6 us = 6.42 ms = 14.5 % at C=128 (kernels.cu:54-83: one 256-thread block per row, 2 x 8-level double shared-memory trees, 16 syncthreads). Not prepared here, as instructed |
| C2 | Residual + LN in one kernel | UNTESTED (overlaps C1 and A12) | residual (x + alpha*y) feeds the next LN at engine.cu:982/985, 994/997, 1002/1005, 1008/1009; pick either A12 or C2 per site |
| C3 | Launch-count reduction overall (~800 encoder kernels per pass: 24 x (10 GEMM + 10 reduce at M<256 + 6 LN + 4 residual + 3 others)) | UNTESTED (as a programme) | at server cohorts (~30-40 rows) the per-kernel floor dominates; A11, A12, C1 and C2 each remove launches |
| C4 | Per-layer megakernel / persistent encoder layer | UNTESTED | high effort; not justified before G0 shows the per-kernel floor |

### D. Subsampling and convolution

| id | idea | status | evidence |
|---|---|---|---|
| D1 | Depthwise conv grid split by channel blocks (glu_dwconv) | MEASURED-NEGATIVE | 5792343, serving-gb10-2026-10-23 "Closed": conv stage 13.5 -> 13.8 ms C=128 |
| D2 | Subsampling dw stages parallel over positions (ss_dw: one 256-thread block per lane, each thread walks To x Fo positions serially) and ss_stage0 likewise | UNTESTED | kernels.cu:468-557; one ss_dw call = 1039 us (stage 1), 13-19x one glu_dwconv call. Each output keeps its 9-tap fmaf chain, so it is byte-identical by construction. Not the same case as D1, which had only q=4 positions per lane |
| D3 | Fuse ss_stage0 -> ss_dw stage 1 (ReLU'd f32 kept in shared memory, NeMo #16114-like "fused depthwise-striding") | UNTESTED | removes the s0 round trip (per lane ~17 x 64 x 256 f32 by the geometry estimate); byte-identical (the values are unchanged) |
| D4 | Pointwise conv as a plain GEMM (NeMo #16052) | ALREADY-PRESENT | pw1/pw2 and ss_pw via gemm() (engine.cu:967-1001) |

### E. Decoder (RNNT label loop)

| id | idea | status | evidence |
|---|---|---|---|
| E1 | Batched greedy RNNT on device, first-index argmax, compaction | MERGED | S14-4 |
| E2 | Blocked label loop (`--label-loop blocked`: about emissions+1 iterations, predictor only for emitters), `--label-sync-every K` | UNTESTED (GPU-READY) | branch 12e3bfa (base 3449ac5): gate D, kernel tests; **does not merge cleanly onto main: conflict in tests/test_cuda_stream.c**; branch predicts -3 to -6 % |
| E3 | Decoder loop inside a CUDA graph with conditional (while) nodes, no host sync per iteration (NeMo "speed of light", arXiv 2406.03791) | UNTESTED | S14-13 note: the label loop stays eager; dec_sync is 0.24 ms/pass (0.5-1.1 %) plus the launch gaps hidden in dec_joint/dec_pred |
| E4 | Label-looping (NeMo, arXiv 2406.06220) | SUPERSEDED by E2 | same mechanism (outer loop over labels, frame scan per lane); E2 is the byte-identical implementation |
| E5 | Single-step LSTM cell instead of a cuDNN LSTM (NeMo PR #16231) | ALREADY-PRESENT | own dec_lstm_gates kernel + 2 GEMMs per layer, no cuDNN |
| E6 | Label-loop host syncs as a lever on the L4 f32 path | MEASURED-NEGATIVE (at that time) | S14-6b: host syncs 0.4 %, decoder 10 % of a 54.5 ms f32 pass; the bf16 pass has since shrunk, so the share is 7.9-13.4 % now (see E2) |

### F. CUDA graphs, sync, memcpy, buffers, host side

| id | idea | status | evidence |
|---|---|---|---|
| F1 | Encoder pass replayed from CUDA graphs per lane x row bucket | MERGED (server default) | 75a360b + 4378448; L40S: submit 1.8 -> 0.1 ms/step, 4216/4216 graph passes; latency indistinguishable at C384 |
| F2 | Graph-on device time on GB10 (the bench is graphs OFF, the server ON) | UNTESTED (measurement gap) | bench banner "graphs off -- launched kernel by kernel"; test_cuda_stream has no `--graphs` for --bench. Matters most at small cohorts |
| F3 | Stage-ahead, feed team, warm-up | MERGED (server default) | #4; gate C and the C=64 wave 64/64 identical |
| F4 | Persistent buffers / workspaces, nothing allocated after open, capture-safe GEMM launcher | ALREADY-PRESENT | cuda-stage-ahead-graphs.md "Known facts", L40S note (own-tc under graphs) |
| F5 | H2D/D2H and layout conversions in the pass | ALREADY-PRESENT (no headroom) | h2d+d2h 0.02-0.06 ms (0.1 %); ss_flatten 0.07 ms; prompt_cat 0.015 ms; the f32->bf16 conversion is load-time only |
| F6 | Mel on the device (cuFFT) | UNTESTED | S14-8 lever; host mel is outside the device pass (stage-ahead + team); a CPU-budget lever only |
| F7 | Weight conversion at open (k_f32_to_bf16, 56 ms GPU) | ALREADY-PRESENT | one-time, engine.cu:589 |

### G. Scheduling, batching, admission

| id | idea | status | evidence |
|---|---|---|---|
| G1 | Cohort batching window (`--cohort-ms`, default 40) | ALREADY-PRESENT | S14 section 5; == NeMo-Speech.cpp "bounded dynamic microbatching" (max_queue_delay / max_batch_size) |
| G2 | Cohort-ms sweep {40, 0, 20} on GB10 | UNTESTED (GPU-READY, packages.sh cohort) | registered ladder never run; measured cohorts are 7-10 lanes with a 40.6-40.9 ms wait, so the window is ~55 % of lag p95 |
| G3 | Work-conserving release (all-active / full-pass triggers) | UNTESTED (GPU-READY) | 0711ddb merges cleanly onto main; with 7-10 lanes per cohort out of 64-192 live, the all-active trigger rarely fires: expect NULL at C>=64 |
| G4 | Early batch release on the CPU server | NOT-TRANSFERABLE (verdict) | S13-1e REJECTED: the CPU fleet runs window 0; the GPU runs 40 ms, so that verdict does not carry over (G3 carries the GPU question) |
| G5 | `--pass-lanes cap` (one 192-lane pass instead of 2 x 96) | UNTESTED | bench C=192 step 73.4 ms vs a linear C=64..128 model at ~59 ms (HYPOTHESIS); only binds when a cohort exceeds 128 lanes, which the measured 7-10 lane cohorts never do |
| G6 | Cross-worker batching | NOT-TRANSFERABLE | CPU F3 measured 1.6 % coincidence; the GPU has one engine thread per device by design |
| G7 | Longest-first sorting / merged last+non-last chunk step (NeMo PR #16231) | ALREADY-PRESENT | ragged per-lane geometry in one pass, no padding except graph buckets |
| G8 | Admission ladder, state arena of indexed device rows | ALREADY-PRESENT | slot arena + 503 before upgrade (S14 section 6); NeMo-Speech.cpp `state_arena_slots` is the same idea |

### H. External implementations as oracles

| id | idea | status | evidence |
|---|---|---|---|
| H1 | NeMo-Speech.cpp (Apache-2.0, ggml): CUDA graphs + patched FastConformer kernels, compute mutex | ALREADY-PRESENT | F1, one engine thread |
| H2 | NeMo-Speech.cpp "planar Q8" layout for high concurrency | UNTESTED (maps to A14) | their docs/asr/models.md; needs their patched runtime; layout idea only |
| H3 | parakeet.cpp (mudler, MIT code; weights under their own licences): single stream, graph rebuilt per step, f32 caches on the host fed per step, pre-projection cache | NOT-TRANSFERABLE | we are ahead on each (B3, B8, F1, multi-stream batching). Its offline batched decode (merged decode steps, ~10-12x at batch 16 on GB10) = our lane-batched decoder (E1). Its README "label-loop" claim is UNVERIFIED in the source |
| H4 | parakeet.cpp q8_0 weights (1.89x on CPU) | UNTESTED (maps to A14) | supports the int8-weight direction; not a GPU measurement |
| H5 | NeMo `use_cuda_graph_decoder` + `loop_labels=True` | SUPERSEDED by E2/E3 | NeMo docs (sub-agent); our equivalents are E2 (built) and E3 (not built) |
| H6 | NeMo PR #16231 `torch.compile` with static chunk length | ALREADY-PRESENT | graph buckets (F1) |
| H7 | Riva/NIM sequence-batcher settings | UNTESTED (no source) | UNVERIFIED, nothing actionable found |

### I. ISA (sm_121)

| id | idea | status | evidence |
|---|---|---|---|
| I1 | wgmma | NOT-TRANSFERABLE | sm_90a only; ptxas "wgmma.fence not supported on .target sm_120" (backend.ai blog) |
| I2 | tcgen05 / TMEM | NOT-TRANSFERABLE | NVIDIA in CUTLASS #2947: "tcgen05 instructions do not exist for SM121" |
| I3 | FP4/FP6 mma.sync block-scaled | UNTESTED | exists on sm_12x per the same thread (exact PTX variants UNVERIFIED); large numerical change, no plan |
| I4 | mma.sync + ldmatrix + cp.async (Ampere family) | UNTESTED (= A9) | the only tensor-core path left to exploit on sm_121 |

### J. CPU-era and sibling-engine levers

| id | idea | status | evidence |
|---|---|---|---|
| J1 | Pool spin, AMX/VNNI/SDOT kernels, prefork admission bound, decoder lane, allocation glue | NOT-TRANSFERABLE (to the CUDA pass) | sibling-wins-and-flags.md sections 1.1-1.4; CPU-host mechanisms |
| J2 | "Parallelise inside one call" (sibling P2) | applied as D2 | principle, not a separate item |
| J3 | Fused residual (sibling TTS win) | UNTESTED as A12/C2 | sibling-wins 1.2 |

## 3. Detail of the UNTESTED items worth a GPU slot

Shares are from today's profile. A bound is the most the idea could save, as a % of the total device pass.

| id | stage | share C=32 / C=128 | upper bound C=32 / C=128 | effort | parity risk | smallest A/B (5-15 min) | GPU |
|---|---|---|---|---|---|---|---|
| E2 | dec_* | 13.4 % / 7.9 % | about -4.2 % / -2.5 % (iterations 9.56 -> ~6.5 at constant cost per iteration); branch predicts -3..-6 % | S (rebase + 1 conflict) | none: byte-identical, gate D | G-E2 device ABBA + gates A-D | yes |
| B6 | att_core | 13.6 % / 18.4 % | bf16 ring <= -3.7 % / -7.0 %; int8 <= -5.5 % / -10 % (ring-read model) | none (flag) | numerical: bits change vs the f32 ring; gates A/B/C pass; bank transcript check needed | G-B6 device ABBA | yes |
| A12 | residual | ~1.4 % / 2.2 % (+96 launches/pass) | about -1.5 % / -1.8 % net; more at server cohorts (launch floor) | S | none by construction (alpha 0.5 exact); byte gate decides | test_cuda_kernels byte check + G-A12 device ABBA | yes |
| D2+D3 | subsample | 3.5 % / 5.4 % non-GEMM | about -2.4 % / -3.4 % | S-M | none by construction | test_cuda_kernels byte check + device ABBA, LEVELS "16 32 128" | yes |
| A9 | all GEMMs | 59 % / 53 % | to cuBLAS: -12.8 % / -19.5 %; to roofline -19.6 % / -20.3 % | L (days) | none if the chains are kept (memcmp gate exists) | gemm_shapes_bench --only ff_w2,qkv (M=128, 512) A/B of a new tile; ncu needs counter access first | yes |
| A11 | att_proj | 9.1 % / 9.4 % | -3.5 % / -2.7 % | M | none if S is pinned | gemm_shapes_bench variant + device ABBA | yes |
| A13 | ffn1->ffn2 hand-off | small (L2-resident at 128 rows) / ~7 % of bytes | about -0.5 % / -3.8 % | M | none by construction (same RNE) | device ABBA | yes |
| A14 | all GEMMs | 59 % / 53 % | about -20..-26 % / -10..-20 % (weight bytes halved; small M gains most) | L | numerical: bank WER gate (w8 f32 arm changed 60/227 clips, WER -0.0015) | gemm_shapes_bench w8a16 variant at lanes 8, 32 | yes |
| F2 | whole pass | -- | unknown; launch gaps at small M | S (bench flag) | none | bench --graphs on/off at C=8/16 | yes |
| G2 | server lag (not the pass) | cohort wait ~40 of ~70 ms lag p95 | up to ~-20 ms lag p95 at 20 ms if GPU duty allows (C=192 duty ~23 %) | none | none (scheduling only, gate A) | packages.sh cohort, ~62 min | yes |
| G3 | server lag | -- | expected NULL at C>=64 (see G3) | none | none | packages.sh release, ~41 min | yes |
| G5 | step at >128-lane cohorts | 0 at measured loads | -19 % step only in bursts > 128 lanes (HYPOTHESIS) | none (flag) | none (gate A/C with pass-lanes) | device bench C=192 --pass-lanes 192 vs default | yes |
| E3 | dec_sync + gaps | 1.1 % / 0.5 % (+ gaps) | about -1 % after E2 | M-L | none if the loop body is unchanged | after E2 only | yes |
| A16 | dec_pred | part of 7.0 % / 3.9 % | <= -1 % | M | byte-identical only with the 2-chain order | gemm_shapes_bench dec shapes | yes |

## 4. TOP 5 (what remains; C1 LayerNorm excluded, IN-PROGRESS)

Score = (expected pass gain, averaged over C=32 and C=128) x confidence / engineering risk. Gains are bounds or
predictions. None of them is measured.

| rank | id | idea | expected gain (C=32 / C=128) | confidence | eng. risk | why this rank |
|---|---|---|---|---|---|---|
| 1 | E2 | Blocked label loop (GPU-READY branch) | -3..-6 % / -2.5..-4 % | 0.7 | low (rebase, 1 conflict) | code and gate D exist; byte-identical; the decoder share grew from 5.6 % to 7.9-13.4 % after #6-#8, and it is largest at the small cohorts the server actually runs |
| 2 | B6 | K/V ring bf16 (then int8) on GB10 | -2..-3.7 % / -4..-7 % | 0.55 | lowest code risk (flag exists), medium parity process | zero code, 5-min device ABBA; the ring-read model says att_core is now ~75 % bandwidth-bound; needs the bank transcript gate before any default change |
| 3 | A12 | Residual folded into the GEMM epilogue | -1.5 % / -1.8 % (+ launch floor at server cohorts) | 0.85 | low | byte-identical by construction; touches the same call sites as C1, so do it in or right after the LayerNorm branch |
| 4 | D2+D3 | Subsampling kernels parallel over positions, stage0 -> dw1 fusion | -2.4 % / -3.4 % | 0.7 | low-medium | ss_dw stage 1 is one 1.04 ms launch with a serial 288-position chain per thread; byte-identical; the D1 null result does not apply (q=4 there) |
| 5 | A9 | own-tc2 pipelined GEMM (cp.async multistage, ldmatrix/mma.sync, BK=64, prepacked W) | -6..-12.8 % / -10..-19.5 % | 0.35 | high (days; ncu blocked) | the largest pool by far (cuBLAS shows 1.6x headroom on these exact shapes), byte-identical if the chains are kept, but the costliest and blind until counter access is granted |

Next after the TOP 5: A14 int8-weight own-tc (numerical, potentially the biggest small-cohort lever), A11 QKV merge,
A13 bf16 hand-off. **Outside the pass metric**, the largest server-latency lever is G2 (cohort-ms sweep): the 40 ms
window is about 55 % of today's lag p95.

## 5. GPU-READY commands (prepared, NOT run)

Common prefix on dgx2. ab_device.sh takes ~/gpu.lock itself.

```bash
cd ~/srv-pr10 && M=$HOME/mynah-asr-ft/models/nemotron-3.5-asr-streaming-0.6b O=$HOME/ft/prof-newmain-20261024
```

**G0. Small-cohort profile (~2 min; do this first, it can reorder the TOP 5).**

```bash
( flock 9; for r in 1 2 3; do for C in 8 16; do timeout 300 ./tests/test_cuda_stream $M --gemm own-tc --precision bf16 \
    --bench $C --steps 40 > $O/bench-c$C-r$r.txt 2>&1; grep -m1 '^BENCH C=' $O/bench-c$C-r$r.txt; done; done ) 9>$HOME/gpu.lock
```

**G0b. Repair the nsys kernel share (~2 min GPU; stats are CPU only).** Check first that the option exists, using
`nsys profile --help | grep -c cuda-flush-interval`. Accept the capture only if attention_kernel shows about 24 x
(passes + warm-up) instances, not 24. If it is still truncated, the bench needs a `cudaDeviceSynchronize();
cudaProfilerStop();` before exit (a code change).

```bash
( flock 9; timeout 900 nsys profile -t cuda --cuda-flush-interval=100 --force-overwrite true -o $O/pass-c32 \
    ./tests/test_cuda_stream $M --precision bf16 --gemm own-tc --bench 32 --steps 12 > $O/nsys2-c32.txt 2>&1 ) 9>$HOME/gpu.lock
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --output $O/pass-c32 $O/pass-c32.nsys-rep
```

**G-B6. K/V ring bf16 and int8, device ABBA (~5 min each) + gates (~3 min).** Pre-registered LEVER: device_ms
<= -3 % at C=128 with both mirror pairs agreeing and no regression at C=16. NULL otherwise. A LEVER still needs the
bank transcript gate (`tests/cuda_bank_transcribe`, kv bf16 vs f32) before any default change.

```bash
PACK=$M LEVELS="16 32 128" PRECISION=bf16 GEMM=own-tc A_LABEL=kv-f32 B_LABEL=kv-bf16 B_ARGS="--kv-dtype bf16" \
  TAG=kvbf16-$(date -u +%H%M) gpu/tools/ab/ab_device.sh tests/test_cuda_stream tests/test_cuda_stream
PACK=$M LEVELS="16 32 128" PRECISION=bf16 GEMM=own-tc A_LABEL=kv-f32 B_LABEL=kv-int8 B_ARGS="--kv-dtype int8" \
  TAG=kvint8-$(date -u +%H%M) gpu/tools/ab/ab_device.sh tests/test_cuda_stream tests/test_cuda_stream
( flock 9; timeout 900 ./tests/test_cuda_stream $M --precision bf16 --gemm own-tc --kv-dtype bf16 ) 9>$HOME/gpu.lock   # gates A/B/C
```

**G-E2. Blocked label loop.** Prerequisite (CPU only, no GPU): rebase feat/serving-labelloop onto origin/main and
resolve the tests/test_cuda_stream.c conflict. Then build it on dgx2 in its own tree, e.g. ~/srv-ll
(`make -C gpu test-stream all`). Device ABBA ~5 min, gates ~5 min, server ~41 min.

```bash
LL=$HOME/srv-ll
PACK=$M LEVELS="16 32 128" PRECISION=bf16 GEMM=own-tc A_LABEL=frame B_LABEL=blocked B_ARGS="--label-loop blocked" \
  TAG=ll-dev-$(date -u +%H%M) gpu/tools/ab/ab_device.sh $LL/tests/test_cuda_stream $LL/tests/test_cuda_stream
( flock 9; timeout 900 $LL/tests/test_cuda_stream $M --precision bf16 --gemm own-tc --label-loop blocked ) 9>$HOME/gpu.lock  # gates A-D
gpu/tools/ab/packages.sh labelloop $LL/mynah-asr-server-cuda      # server ABBA, strict identity, ~41 min
```

**G-A12 / D2 / A9 / A11 / A13 / A14** need code first. Their A/B is the same shape: `test_cuda_kernels`
(byte-for-byte against the current path), then `ab_device.sh BASE PATCH` with `LEVELS="16 32 128"`,
`PRECISION=bf16 GEMM=own-tc`. For the GEMM items, run first
`tests/gemm_shapes_bench --model nemotron --lanes 8,32,128 --only <shape> --reps 20` on base vs patch.

**G-F2 (tooling).** Add `--graphs buckets` to `test_cuda_stream --bench`, then run an ABBA graphs off vs on at
C=8/16/32.

**G2 / G3 (server, scheduling only).**

```bash
gpu/tools/ab/packages.sh cohort ~/srv-pr10/mynah-asr-server-cuda          # {40,0,20}, ~62 min
gpu/tools/ab/packages.sh release <0711ddb build>/mynah-asr-server-cuda 20 # after the sweep, ~41 min per cohort value
```

**G5.**

```bash
PACK=$M LEVELS="192" PRECISION=bf16 GEMM=own-tc A_LABEL=pl128 B_LABEL=pl192 B_ARGS="--pass-lanes 192" \
  TAG=pl-$(date -u +%H%M) gpu/tools/ab/ab_device.sh tests/test_cuda_stream tests/test_cuda_stream
```

Compare step_wall_ms, not device_ms_per_pass: the pass count differs between the arms.

## 6. CPU-only prerequisites (no GPU needed)

1. Rebase feat/serving-labelloop onto main (conflict: tests/test_cuda_stream.c). feat/serving-c-release merges cleanly.
2. gemm_profile.sh: add `--force-export=true` to `nsys stats`, a flush option to `nsys profile`, and fix the second
   `nsys stats` call (usage error).
3. Ask the owner for counter access on dgx2: `sudo ncu`, or `NVreg_RestrictProfilingToAdminUsers=0` and a driver
   reload. This unblocks A9.
4. Coordinate A12/C2 with the LayerNorm branch, since they touch the same call sites (engine.cu:979-1009).

## Status

**NEXT-AGENT / GPU-READY.** The matrix is closed for every idea that has evidence. Run order: G0 (small-cohort
profile), then G-B6 and G-E2 (both cheap), then the code items A12 and D2 next to the LayerNorm branch. A9 waits for
counter access.
