# Mynah — Compute backends (CPU / Metal / CUDA)

Runtime selection: `--backend cpu|metal|cuda` (CLI and server). qwen-tts pattern:
request → `resolve()` → note on stderr → **graceful fallback to CPU** if the backend
is unavailable or not compiled in. GEMMs under 24 rows always stay on CPU
(the GPU round-trip does not pay off on streaming chunks).

## CPU (default)

f32 GEMM provider: `make BLAS=none|openblas|accelerate`, reported as
`gemm.f32` by `--dispatch-map` and on the `[EFFECTIVE-CONFIG]` line. `none` is
the **Linux default** and links no cblas at all — `src/sgemm.c` computes every
f32 GEMM and GEMV on the same pool as everything else, which is the point: a
worker pinned to T cpus runs T threads and not 2T. `accelerate` is the macOS
default and `openblas` the comparison arm. Own SDOT/VNNI/AVX2 kernels
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

## Proving which path ran: `--dispatch-map`, `--flags`, the ISA guard

ENGINEERING.md §5 forbids inferring the active kernel from a flag, a build target
or a filename, so the binary answers for itself. All three surfaces are
model-free and cost nothing to run before a measurement.

**`mynah-asr --dispatch-map [--json]`** prints one row per logical feature —
int8 dot, int4 dot, the opt-in int8 GEMM, the f32 GEMM provider, Metal, CUDA,
the pool width, the BLAS budget — with six columns: `feature · compiled ·
supported · env · resolved · reason`. The rule the report is built on is that
`resolved` is **never** derived as `compiled && supported`: it comes from a
predicate exported by the file that owns the decision (`src/qmat.c`
`mynah_asr_qmat_int8_kernel()` and friends), from a real runtime call
(`mynah_asr_metal_available()`, `mynah_asr_num_threads()`), or from a pure
compile-time gate — and `reason` says which, tagged `[predicate]`, `[runtime]`
or `[gate]`. Anything else prints `UNKNOWN`, and the footer counts those rows:
an UNKNOWN is a predicate somebody still has to export, not a failure. The
`IDLE HARDWARE:` footer lists the CPU features this host has that this binary
never issues an instruction for (ARM: dotprod, i8mm, bf16, sve, sve2 — from
sysctl on macOS, `getauxval` on Linux; x86: avx2, avx512f, avx512vnni, avxvnni,
amx — from CPUID), and each idle line names the kernel that would have to be
**written** to use it. Feature probes are tri-state: a feature this process
cannot interrogate reads `unknown`, never `absent`.

**`mynah-asr --flags`** prints exactly two machine-readable lines, the same two
that `MYNAH_ASR_VERBOSE=1` puts on stderr before a `transcribe`, `stream` or
`bench` run:

```
[FLAGS] v=1 MYNAH_ASR_CAPS=vnni MYNAH_ASR_THREADS=999 OPENBLAS_NUM_THREADS=3
[EFFECTIVE-CONFIG] v=1 build=<git rev> blas=accelerate simd=neon+dotprod \
  MYNAH_ASR_CAPS=vnni->none(IGNORED:_not_an_x86_build:_...) \
  MYNAH_ASR_THREADS=999->64(clamped) \
  OPENBLAS_NUM_THREADS=3->none(IGNORED:_this_build_links_Accelerate;_...)
```

Every environment variable the runtime reads is one row of the single registry
in `src/flags.c` — name, scope (`runtime|kernel|server|debug`), default,
description, and an optional predicate that returns the reason the flag is
**inert** in this build on this host. A flag that cannot act is reported
`IGNORED:` with that reason instead of being echoed back as if it had been
applied, which is the whole point: a quoted result must not rest on a variable
that did nothing. `make check` runs `tools/check_flag_registry.py`, which fails
when a name read through `getenv()` anywhere in `src/ cli/ server/ tests/` is
missing from the table, or when the table names a flag nothing reads.

**The ISA guard** is the first statement of `main()`. If the binary was compiled
with an instruction set this CPU definitely lacks — `__ARM_FEATURE_DOTPROD` on a
CPU whose auxv/sysctl says no dotprod, `__AVX2__` where CPUID says no AVX2 — it
prints one line naming the ISA, the missing feature and the fix, and exits **78**
(`EX_CONFIG`) instead of dying with a bare `SIGILL` three frames inside a kernel.
It fires only on a **definite** absence: the probes are tri-state, and refusing
to start over our own ignorance would be a worse bug than the one it prevents.
