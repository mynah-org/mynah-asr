# Canary (AED) — usage, languages, translation matrix

User-facing guide for the Canary models. Architecture (encoder/decoder,
tokenizer, prompt format) in [canary-arch.md](canary-arch.md); catalog in
[models.md](models.md). All CC-BY-4.0, offline-only (no cache-aware streaming).

| model | ASR languages | translation | word timestamps | RTF 65s CPU/Metal (int8) |
|---|---|---|---|---|
| canary-180m-flash | en, de, es, fr | en ↔ de/es/fr | yes (experimental) | 0.060 / 0.054 (0.030) |
| canary-1b-flash | en, de, es, fr | en ↔ de/es/fr | yes | 0.143 / 0.081 (~0.07) |
| canary-1b-v2 | **25 EU languages** | **en ↔ 24** | no (see below) | to be measured |

## Speech translation

Output language ≠ source language = translation:

```sh
./mynah-asr transcribe -m models/canary-1b-v2 -i italian.wav --lang it --target-lang en
# API: mynah_asr_set_target_lang(m, "en")  or  lang "it>en" per-call (thread-safe)
# server: POST /v1/audio/translations  (target_language, default en)
```

- flash models: EN↔{DE,ES,FR} only. v2: EN↔24 (both directions), it included.
- The FLEURS samples in `samples/` are parallel across languages: the English
  clip doubles as reference for scoring translations (`make test-samples`).
- fr→en on the 180m emits EOS immediately (model limitation, verified against
  the oracle too) — use the 1b models for that pair.

## Behavior notes

- **ITN**: v2 applies inverse text normalization by default (numbers as
  digits: "um 9 Uhr"); the flash models spell them out ("um neun Uhr").
  The e2e expectations in `tests/test_e2e.sh` encode exactly this difference.
- **Timestamps**: flash models bracket words with generative `<|N|>` tokens
  (80 ms frames) — `--timestamps` / verbose_json work; in ts-mode punctuation
  can differ slightly (model behavior). **v2 does not support them** (its
  `.nemo` bundles a separate ASR aligner model — extraction is a TODO);
  `timestamp_tokens: false` in mynah.json makes the e2e skip them.
- **Decoder int8**: the AED decode loop is autoregressive and dot-bound —
  int8 gives 2.8–3.2× on the decoder (SDOT T=1 beats sgemv f32).
- 1b models on 16 GB machines: first Metal runs pay swap (3.4 GB f32 +
  1.7 GB f16 resident) — warm up before benchmarking (see benchmarks.md).
