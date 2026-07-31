# Canary (AED) — usage, languages, translation matrix

User-facing guide for the Canary models. Architecture (encoder/decoder,
tokenizer, prompt format) in [canary-arch.md](canary-arch.md); catalog in
[models.md](models.md). All CC-BY-4.0, offline-only (no cache-aware streaming).

| model | ASR languages | translation | word timestamps | RTF 65s CPU/Metal (int8) |
|---|---|---|---|---|
| canary-180m-flash | en, de, es, fr | en ↔ de/es/fr | yes (experimental) | 0.060 / 0.054 (0.030) |
| canary-1b-flash | en, de, es, fr | en ↔ de/es/fr | yes | 0.143 / 0.081 (~0.07) |
| canary-1b-v2 | **25 EU languages** | **en ↔ 24** | yes, via the bundled aligner | to be measured |

## Speech translation

Output language ≠ source language = translation:

```sh
./mynah-asr transcribe -m models/canary-1b-v2 -i italian.wav --lang it --target-lang en
# API: mynah_asr_set_target_lang(m, "en")  or  lang "it>en" per-call (thread-safe)
# server: POST /v1/audio/translations  (target_language, default en)
```

- flash models: EN↔{DE,ES,FR} only. v2: EN↔24 (both directions), it included.
- **The source language is an input, not a guess**: `--lang auto` on a Canary does
  NOT detect anything, it falls back to the model's default (`en`). For audio of
  unknown language, pair it with a detector (below).
- The FLEURS samples in `samples/` are parallel across languages: the English
  clip doubles as reference for scoring translations (`make test-samples`).
- fr→en on the 180m emits EOS immediately (model limitation, verified against
  the oracle too) — use the 1b models for that pair.

## Unknown source language: `--lid-model`

Nemotron is the only model here that says which language it heard. Point Canary at
one and the source language stops being something the caller has to know:

```sh
./mynah-asr transcribe -m models/canary-1b-v2 -i unknown.wav --lang auto \
    --lid-model models/nemotron-3.5-asr-streaming-0.6b --target-lang en --quant int8
# [lid 0.53s | detected it-IT -> --lang it]
```

- The detector runs on a **short prefix** (8 s from the first speech, 24 s on a retry)
  and only its language tag is kept, so the cost does not grow with the file: **~0.5 s
  on CPU with int8** — run `mynah-asr quantize -m <lid_dir> --quant int8` first, f32
  costs several times that. On the CLI the detector is freed before the real
  transcription, so only one large model is ever resident.
- The locale is adapted to what the target accepts (`it-IT` → `it`), which is also
  where a mismatch surfaces: Nemotron knows 40 locales, canary-1b-v2 takes 25.
- **Never fails the request.** Nothing detected, or a language this Canary does not
  have → a warning on stderr and the model's default (`en`) is used. The warning is
  the signal that the text may be wrong; there is no silent version of it.
- Server: `mynah-asr-server -m <canary> --lid-model <nemotron>` does the same for
  every request with `language=auto`, and reports the detected language in
  `verbose_json`. See [server.md](server.md).
- Not needed on Nemotron itself (`--lang auto` already detects); passing it there is
  ignored with a note.

## Behavior notes

- **ITN**: v2 applies inverse text normalization by default (numbers as
  digits: "um 9 Uhr"); the flash models spell them out ("um neun Uhr").
  The e2e expectations in `tests/test_e2e.sh` encode exactly this difference.
- **Timestamps**: flash models bracket words with generative `<|N|>` tokens
  (80 ms frames) — `--timestamps` / verbose_json work; in ts-mode punctuation
  can differ slightly (model behavior).
  **v2 works differently**: the generative tokens do nothing on it (non-monotonic
  attention — it emits EOS immediately), so the times come from the **CTC aligner
  bundled in its `.nemo`**. The converter extracts it to `<model_dir>/aligner/`
  (see [canary-arch.md](canary-arch.md)) and `--timestamps` then works normally:
  the text still comes from the AED, only the times come from the aligner, by
  Viterbi-aligning that text against the aligner's per-frame CTC scores.
  Without the aligner directory `timestamp_tokens: false` still means no times.
  Cross-check on `test_en.wav`: the aligned times are **identical** to the ones the
  aligner produces decoding on its own, word for word — which is the expected
  outcome when both look at the same posteriors.
- **Decoder int8**: the AED decode loop is autoregressive and dot-bound —
  int8 gives 2.8–3.2× on the decoder (SDOT T=1 beats sgemv f32).
- 1b models on 16 GB machines: first Metal runs pay swap (3.4 GB f32 +
  1.7 GB f16 resident) — warm up before benchmarking (see benchmarks.md).
