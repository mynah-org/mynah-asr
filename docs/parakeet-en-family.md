# Parakeet EN family — usage, features, limits

The English-only Parakeets supported by the runtime. Architecture: FastConformer
encoder (80 mel, non-causal) + RNNT/CTC/TDT decoders — details shared with
[parakeet-tdt-arch.md](parakeet-tdt-arch.md); catalog & licenses in
[models.md](models.md). All CC-BY-4.0, all `.nemo`-distributed (converted with
`tools/convert_nemo.py`; rnnt/ctc 0.6b also have HF-native safetensors).

| model | decoder | layers | output style | RTF 65s (CPU/Metal) | RAM f32 |
|---|---|---|---|---|---|
| parakeet-tdt_ctc-110m | hybrid TDT + CTC | 17L·512 | PnC (capitals + punctuation) | 0.015 / 0.010 | 0.44 GB |
| parakeet-rnnt-0.6b | RNNT | 24L·1024 | lowercase, no punctuation | 0.050 / 0.028 | ~2.4 GB |
| parakeet-ctc-0.6b | CTC | 24L·1024 | lowercase, no punctuation | 0.042 / 0.022 | ~2.4 GB |
| parakeet-rnnt-1.1b | RNNT | 42L·1024 | lowercase, no punctuation | 0.068 / 0.041 | 4.0 GB |
| parakeet-ctc-1.1b | CTC | 42L·1024 | lowercase, no punctuation | 0.062 / 0.033 | 4.0 GB |

## Features in mynah

- **Language**: English only — `--lang` is ignored (no language prompt in the
  model); `lang=` in the CLI footer reports nothing meaningful for these.
- **110m hybrid**: `--decoder ctc` switches to the auxiliary CTC head
  (faster: RTF 0.023 vs 0.027 TDT on the fixture; slightly different text
  normalization). The 0.6b/1.1b models are single-decoder.
- **Timestamps** (`--timestamps`): accurate on the 110m TDT head (frame =
  cumsum of TDT durations); RNNT/CTC use the emission frame (~80 ms grain).
- **Quantization**: `mynah-asr quantize --quant int8` ≈ transparent on all;
  int8/int4 checkpoints load zero-copy (RAM ~⅓).
- **Metal**: whole family runs on GPU (BatchNorm folded, 'same' depthwise,
  full attention) — see the RTF column; falls back to CPU for quantized
  weights.
- Long files: silence-based segmentation, default limit 300 s per segment.

## Limits / gotchas

- The 0.6b/1.1b write **lowercase without punctuation** by design (model
  cards): don't compare their output against PnC references.
- No streaming: `mynah-asr stream` rejects offline-only models — for live use
  pick nemotron (cache-aware) instead.
- 1.1b models: converted from `.nemo` with zero code changes (42L is
  config-driven), but the `.nemo` archives are ~4.4 GB — deleted locally
  after conversion, re-download to reconvert.
- The 110m is the CI model: smallest, fastest, exercises the hybrid path.
