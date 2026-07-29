# Mynah — INT8 / INT4 quantization

## TL;DR

```sh
# once: save the pre-quantized checkpoints (built-in tool, zero dependencies)
./mynah-asr quantize -m models/nemotron-3.5-asr-streaming-0.6b --quant int8   # 0.79 GB
./mynah-asr quantize -m models/nemotron-3.5-asr-streaming-0.6b --quant int4   # 0.57 GB

# then: INSTANT load (zero-copy mmap, the 2.55 GB f32 is no longer needed at runtime)
./mynah-asr transcribe -m <dir> -i audio.wav --quant int8
./mynah-asr-server -m <dir> --quant int8
```

## Numbers (Apple Silicon, 5.2 s fixture, warm cache)

| | f32 | int8 prequant | int4 prequant |
|---|---|---|---|
| weight file | 2.55 GB | **0.79 GB** | **0.57 GB** |
| load | ~0.04 s (mmap) | **0.01 s** | **0.01 s** |
| offline RTF | 0.17 | **0.10** | 0.15 |
| streaming (5.2 s, wall) | 9.2 s | **~2.1 s** | **~1.9 s** |
| fixture text | reference | **identical** | nearly identical (one capital letter, one "it"/"this") |

The small-T dots (streaming/decode) use **native int8×int8**: per-row activation
quantization (qwen-tts recipe) + **SDOT** on ARMv8.2 (`vdotq_s32`, q8 and q4 —
nibbles deinterleaved with `vld2q_s8`), **VNNI** (`dpbusd`) or **AVX2** (llama.cpp
sign/abs trick) on x86 for q8; NEON-f32/scalar fallback elsewhere. Kernels validated by
`tests/test_qmat` (model-free self-test, also runs in x86 CI).
Streaming int4: RTF ~0.37 (~0.11 s/chunk on a 0.32 budget) — the lightest format is
also the fastest, as it should be.

## What gets quantized

Only the large linears consumed by the GEMMs (~92% of the parameters, prior-art recipe):
FFN linear1/2, attention q/k/v/o, pointwise conv1/2, joint head. Kept in f32:
`relative_k_proj` (large T on every chunk), conv2d subsampling, depthwise conv, LSTM,
embedding, norm, bias.

Schemes: **INT8** per-row symmetric (scales [n]); **INT4** per-group of 32 with packed
nibbles (scales [n, k/32], Q4_0 style).

## On-disk format

`model.int8.safetensors` / `model.int4.safetensors`, **self-contained**: for each
quantized tensor `<name>.q8` (I8 [n,k]) or `<name>.q4` (U8 [n,k/2]) + `<name>.scales`
(F32); all other tensors copied f32 verbatim. The runtime maps them zero-copy.

Fallback: without a pre-quantized checkpoint, `--quant int8|int4` quantizes at load time
from the f32 (~9 s, and on macOS the touched f32 pages stay counted in RSS — on Linux
they are released with madvise).

## Kernels (src/qmat.c)

- `T <= 16` (streaming chunk, decode): direct quantized dot — reads 4×/8× fewer bytes
  where the matvec is bandwidth-bound.
- `T > 16` (offline/batch): per-call dequant into scratch + BLAS GEMM.

## Tests

`make test` includes e2e with `--quant int8` and `--quant int4` (skipped if the
checkpoints haven't been generated). Leak check: 0 bytes on the quantized path too.

## Multilingual CER regression (2026-07-17)

Full `test-nemo-langs` suite (34 locales + 4 variants, 102 Tatoeba/FLEURS samples,
explicit language, CER threshold 0.3) with `uv run python -m eval.test_langs --quant int8|int4`:

|            | f32   | int8  | int4  |
|------------|-------|-------|-------|
| mean CER   | 0.133 | 0.145 | 0.227 |
| languages OK | 37/37 | 36/37 | 35/37 |
| ΔCER>0.05  | —     | 2/38  | 19/38 |

- **INT8 ≈ transparent**: CER identical to f32 on ~90% of the languages. Only
  ro-RO regresses (+0.15, already borderline in f32: 0.47) and ja-JP (+0.33 on 3 samples: one
  degenerate sample — statistical noise, to be verified with more data).
- **INT4 hurts on the real suite** (the `say` fixtures didn't show it): half the
  languages lose >0.05 CER, es-ES and ro-RO fall below threshold. It remains valid for
  streaming where bandwidth dominates, but for multilingual quality use int8.
