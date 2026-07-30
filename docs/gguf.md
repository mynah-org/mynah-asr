# Mynah — GGUF weight container

## TL;DR

```sh
# export a converted model to GGUF (weights only — mynah.json/tokens stay as-is)
cd tools && uv run python export_gguf.py ../models/parakeet-tdt_ctc-110m --dtype q8_0

# the runtime picks model.gguf up automatically when model.safetensors is absent
./mynah-asr transcribe -m models/parakeet-tdt_ctc-110m -i audio.wav
```

**safetensors stays the default.** GGUF is an *alternative container* for the
same weights: at load everything is normalized into the same internal tensor
table (F32 zero-copy from mmap; F16/BF16/Q8_0/Q4_0 and the K-quants Q4_K/Q5_K/Q6_K
dequantized to f32), so the
encoder/decoder/Metal/CUDA code paths are identical for both containers — one
code path, per repo rule.

## What is supported

| | |
|---|---|
| GGUF versions | v2 and v3 (v1 has a different, u32-based layout: rejected) |
| ggml types (read) | F32, F16, BF16, Q8_0, Q4_0, **Q4_K, Q5_K, Q6_K** — anything else (Q2_K, Q3_K, the IQ family) fails with a clear error |
| ggml types (write) | `export_gguf.py`: F32, F16, Q8_0, Q4_0, Q4_K — the 5/6-bit K-quants are read-only for now (see "Not (yet) supported") |
| model config | **always from `mynah.json`** (config-driven rule): the GGUF KV metadata is not trusted for hyperparameters |
| lookup order | `model.int8/int4.safetensors` (with `--quant`) → `model.safetensors` → `model.gguf` |

`export_gguf.py --dtype q8_0|q4_0|q4_k` quantizes only ≥2-D matrices whose last
dim is a multiple of 32 (256 for q4_k, else per-tensor fallback to q8_0); 1-D
tensors (bias, norms, BatchNorm running stats) stay f32 — quantizing
`running_var` corrupts `1/sqrt(var)` (found the hard way: the model output
degenerated to garbage until 1-D tensors were exempted). The q4_k writer uses
a simple non-iterative quantizer (ggml's search is finer): it exists to
validate the C dequant layout per the oracle rule, not to produce the best
possible int4 files.

File sizes for parakeet-tdt_ctc-110m: f32 459 MB · q8_0 164 MB · q4_k 117 · q4_0 114 MB.
Note: GGUF quantized types are dequantized to f32 **at load** (RAM = f32 size);
for lowest RAM and the native SDOT/VNNI kernels use mynah's own
`mynah-asr quantize` int8/int4 checkpoints ([quantization.md](quantization.md)).

## Validation

- `tests/test_gguf` (model-free, runs in CI): synthetic GGUF files — dims
  reversal (ggml `ne[0]` = fastest), F32 bit-exact zero-copy, per-type dequant
  tolerances, and a malformed-file battery (bad magic, v1, truncation,
  out-of-file tensor offsets, absurd string lengths, non-power-of-2 alignment,
  unsupported types) that must all fail cleanly.
- `tests/test_gguf.sh` (part of `make test`, skips without model): exports the
  110m to f32 and q8_0 GGUF and checks the f32 transcription is **identical**
  to the safetensors load, q8_0 matches the expected text.
- `tests/test_kquant.sh` (part of `make test`, model-free, needs uv): the
  K-quant dequant against the **reference implementation**. Our C dequant and
  the synthetic fixtures above were written from the same reading of the spec,
  so a shared misunderstanding would pass unnoticed;
  `tools/gen_kquant_ref.py` therefore builds the file with llama.cpp's own
  `gguf` writer and dumps `gguf.quants.dequantize()` per tensor, and the C side
  must reproduce it. Blocks are filled with deterministic pseudo-random bytes so
  every scale, min, nibble and high-bit position is exercised (quantized smooth
  data leaves most bit patterns untouched). Result: **bit-exact**, err_rel 0 for
  Q4_K/Q5_K/Q6_K.

## Third-party GGUFs (parakeet.cpp ecosystem)

Community GGUFs use their own tensor naming and carry the config in GGUF
metadata. `tools/import_gguf.py` bridges them **without torch**: from the
single .gguf it generates the whole model dir (mynah.json translated from the
`stt.*` metadata, tokens.json from the embedded tokenizer, computed mel
filters, model.gguf with tensors renamed to the runtime's HF-verbatim naming
— quantized payloads copied verbatim).

```sh
curl -LO https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q4_K_M.gguf
cd tools && uv run python import_gguf.py ../parakeet-tdt_ctc-110m-Q4_K_M.gguf --out ../models/110m-q4k
./mynah-asr transcribe -m models/110m-q4k -i audio.wav
```

Verified real file: **handy-computer/parakeet-tdt_ctc-110m-gguf Q4_K_M**
(90 MB): 354 shared f32 tensors bit-exact vs our own `.nemo` conversion,
e2e EN/timestamps/segmentation/Metal green, RTF 0.021. Known limitation of
that file: the community quantizer drops the auxiliary CTC head →
`--decoder ctc` is cleanly rejected. Per the incremental-support policy,
other repos/architectures get enabled as they are tested on real files.

## Not (yet) supported

**Writing** Q5_K/Q6_K: reading them is done and validated (see the parity test
above), but the exporter cannot produce them — a good K-quant writer needs the
per-sub-block scale search that llama.cpp does in C (its own numpy `gguf`
package raises `NotImplementedError` for these types). Not a blocker: the point
of the 5/6-bit K-quants is consuming third-party files, and for our own weights
q4_k/q8_0 already cover the size/quality range.

Also missing: Q2_K/Q3_K and the IQ family (no verified real file needs them; the
loader rejects them explicitly rather than guessing), and non-parakeet
third-party architectures. Q4_K dequant is ported from the keyra project and
validated both on the real community file above and against llama.cpp.
The parser is hardened against hostile input (overflow-checked arithmetic,
bounded strings/arrays/recursion, mmap range validation; origin: the keyra
project's parser, reviewed and extended).
