# Mynah — Compute backends (CPU / Metal / CUDA)

Runtime selection: `--backend cpu|metal|cuda` (CLI and server). qwen-tts pattern:
request → `resolve()` → note on stderr → **graceful fallback to CPU** if the backend
is unavailable or not compiled in. GEMMs under 24 rows always stay on CPU
(the GPU round-trip does not pay off on streaming chunks).

## CPU (default)

BLAS: Accelerate (macOS) / OpenBLAS (Linux) for GEMMs; own SDOT/VNNI/AVX2 kernels
for quantized dot products (see [quantization.md](quantization.md)). On Apple Silicon
it is the fastest backend today (AMX): offline RTF 0.10 (int8).

**Runtime ISA dispatch (x86)**: the VNNI and AVX2 kernels are always compiled with
target-attribute (no `-march` required: a single multi-target release binary)
and selected via cpuid+xgetbv on first call. Override with `--caps
auto|scalar|avx2|vnni` (CLI and server) or the `MYNAH_ASR_CAPS` env var — the `--caps`
pattern from qwen-tts; a level above what the CPU supports is downgraded with a note.
On ARM, NEON/SDOT remain compile-time (Apple Silicon always has dotprod).
⚠️ As with CUDA: the AVX2/VNNI paths are validated by `tests/test_qmat` in Linux x86
CI, not on this Mac (Rosetta here does not expose AVX2 → scalar level, tested).

## Metal (macOS, compiled in by default)

`src/metal_mps.m`: MPSMatrixMultiplication with **resident weights** (MTLBuffer cached
per pointer — qwen-tts weight-cache pattern) and reusable I/O buffers.

**v2 (fp16 + fusions)**: resident fp16 weights; fused FFN (GEMM→SiLU shader→GEMM,
one sync) and q/k/v together → −15% vs CPU.

**v3 (whole blocks on GPU)**: full attention in ONE command buffer — qkv,
bias_u/v (shader), relk, per-head GEMM on **strided MPS views** (zero permute),
softmax+rel_shift+chunked window in a shader (float accumulation), ctx, o_proj — and
the entire conv module (pw1→GLU→dwconv9→LayerNorm→SiLU→pw2, all own shaders).
4 syncs per layer (were ~10 in v1).

**v4 (entire encoder on GPU, one sync per forward)**: the residual stream also stays
on GPU — **f32 for numerical fidelity** (LN between blocks and residual add accumulate
in f32 like the CPU; the blocks stay f16 as in v3). f32↔f16 conversions in the
shaders; the CPU only does two memcpys per forward. One command buffer per layer,
commit without wait (the GPU executes layer i while the CPU encodes i+1), **wait only
on the last one**. The layernorms are threadgroup-per-row with `simd_sum` reduction
(coalesced reads: one thread per row with stride d cost ~15 ms/layer and
was the bottleneck of the first v4). Weight conversion at load via vImage.
The **server batch** goes through Metal: each segment runs the entire encoder on GPU
(resident weights = weight-stationary anyway), parity 4/4 vs B=1.

Measured **in-process warm** (server scenario, 63 s, best-of-N within the same
minute — the per-process measurement is dominated by page-in and weight conversion):
encoder 0.70 s vs 2.1 s for v3; total **RTF 0.051 vs 0.068 CPU (−25%)**
(v3: 0.072). Identical text IT/EN/DE/FR/ES, 0 leaks. Under 24 rows (streaming
chunks) it stays on CPU. `MYNAH_ASR_METAL_PROF=1` prints encode/wait/GPU time.

With the optimized CPU pipeline (blocked greedy, im2col subsampling, sparse
mel — see TODO M5 2026-07-17), warm totals on the 63 s drop to
**Metal RTF 0.042, CPU 0.060**. Note: the decode ALWAYS stays on CPU BLAS
even with the Metal backend (determinism across backends: same logits, same text).

## CUDA (Linux, `make cuda`)

`src/cuda_gemm.cu`: cuBLAS sgemm with per-pointer resident weight-cache, reusable
device buffers, own stream+handle (qwen_tts_cuda.c pattern).

> ⚠️ **Cross-compiled, NOT yet validated on hardware** (same approach used in
> qwen-tts for VNNI): on a Linux+CUDA machine run `make cuda && make test`
> before trusting it. Automatic CPU fallback on any CUDA error.

Planned evolutions (TODO, qwen-tts pattern): resident bf16 weights + `cublasGemmEx`,
fused resident decode, CUDA Graphs on the streaming loop.
