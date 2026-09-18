# S5 — CPU kernels for the batched stream step, ARM and x86 together

Status: OPEN

Task: S5-1, S5-2, S5-3, S5-4, S5-5
Question: once the server batches B streams into one step, the hot shape is
`[B·Q ≤ 64, k] × [n, k]ᵀ` with int8 weights. Today that shape leaves the
native int8 path (`T > 16` dequantises the whole matrix and mallocs per call).

Known facts (src/qmat.c at HEAD)
- Native int8×int8 dot per row for `T ≤ 16`, `k ≤ 8192`: NEON SDOT
  (`dot_q8_sdot`, `dot_q4_sdot`), AVX-512 VNNI (`dot_q8_vnni`), AVX2
  (`dot_q8_avx2`), runtime CPUID dispatch on x86, compile-time on ARM.
- `T > 16`: threaded int8 GEMM is opt-in and off (`MYNAH_ASR_QGEMM=1`, lost to
  AMX on Apple); default is dequant + sgemm with `malloc(n*k*4)` per call.
- No SMMLA (i8mm), no SVE, no BF16 kernels; depthwise conv is a scalar triple
  loop; layer_norm SIMD was archived as not worth it.
- Siblings: mynah-tts has `matvec_q8_pair_i8mm` (two activations per SMMLA,
  one weight load serving two rows) proven **bit-identical to SDOT** for the
  same (row, activation), because otherwise a request's output would depend on
  who it was batched with; exported int8 primitives (`act_quantize` with one
  encoding definition: signed for SDOT, unsigned x+128 for VPDPBUSD,
  `pack_q8` with per-row scale + rowsum, `dots_i8`, `epilogue`). qwen-tts has
  the full VNNI/AMX matmat family, KleidiAI (`qsi8cxp` i8mm GEMM, requires
  i8mm at build), and the three-predicate dispatch (`compiled / supported /
  use(gate, B, rows, cols)`) with a `force()` pin for in-process paired A/B.
- Measured elsewhere: the activation-quantisation pass was the serial
  fraction of an int8 conv call (flat 3.8 ms while the GEMM went 16.7 → 7.7
  ms); put it on the pool. The smaller the model, the larger the share of
  preparation over arithmetic.
- Axion has i8mm, bf16, sve2 idle; the binary uses SDOT only.

Plan (each gated by the S0/S4 profile that blames it, in this order)
- S5-1 weight-stationary int8 GEMM for `T ≤ 64`: ARM SMMLA with the SDOT
  fallback, x86 VNNI (`vpdpbusd`) with the AVX2 `maddubs` fallback; activation
  quantisation on the pool; **bit-exact against T single-row calls** (integer
  accumulation, asserted with `==`); removes the dequant+malloc path from the
  server's shapes. Gate: `test_qmat` extended, transcripts byte-identical,
  step time at B=8 before/after on Axion and on an x86 box.
- S5-2 depthwise conv: weights transposed to `[k][d]` at load, NEON/AVX2 FMA
  over `d`. Gate: parity vs scalar, per-layer µs before/after.
- S5-3 KleidiAI behind `KLEIDI=1` as an A/B arm for the same shapes; promoted
  only if it beats S5-1 on the soak envelope, not on a microbenchmark.
- S5-4 build: `SIMD=auto|portable|neon|avx2|avx512` with the double test
  (cpuinfo and a compiler probe), build-flag stamp so objects cannot mix,
  link-only CI job per ISA ("guards the guard"), release flags per target.
- S5-5 pool meter run on the batched step to pick W×T from the 200 µs
  break-even rule, then re-run the S4 ladder.

Numerical rule: int8 for the encoder already exists and is qualified by the
CER samples; any new arithmetic (bf16, different accumulation) is a separate
quality-gated item.

Evidence / Conclusion / Next action: after S4's first soak.
