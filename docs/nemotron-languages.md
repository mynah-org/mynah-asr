# Nemotron 3.5 ASR Streaming — supported languages

> Source: HF model card (tiers) + the actual `processor_config.json` (`prompt_dictionary`,
> in `reference/nemotron-3.5-asr-streaming-0.6b/`). Updated 2026-07-16.

The model supports **40 official locales** split into 3 quality tiers, plus automatic
language detection (`target_lang=auto`, prompt id 101). The `prompt_dictionary` actually
contains **105 slots** (ids 0–104): the slots beyond the 40 official ones are meant for
fine-tuning on new languages (see the NVIDIA fine-tuning blog).

## Tier 1 — Transcription-ready (19 locales)

Production-ready quality. FLEURS WER @1.12s in parentheses where published.

| Locale | Language | Prompt id | Notes |
|---|---|---|---|
| en-US | English (US) | 0 | WER 7.91% |
| en-GB | English (UK) | 1 | |
| es-ES | Spanish (Spain) | 2 | |
| es-US | Spanish (US) | 3 | WER 4.11% |
| fr-FR | French | 8 | |
| fr-CA | French (Canada) | 100 | |
| **it-IT** | **Italian** | **15** | **WER 4.25%** |
| pt-BR | Portuguese (Brazil) | 12 | WER 5.48% |
| pt-PT | Portuguese (Portugal) | 13 | |
| nl-NL | Dutch | 16 | |
| de-DE | German | 9 | |
| tr-TR | Turkish | 18 | |
| ru-RU | Russian | 11 | |
| ar-AR | Arabic | 7 | |
| hi-IN | Hindi | 6 | |
| ja-JP | Japanese | 10 | |
| ko-KR | Korean | 14 | |
| vi-VN | Vietnamese | 33 | |
| uk-UA | Ukrainian | 19 | |

## Tier 2 — Broad-coverage (13 locales)

Broad coverage, lower quality (tier average: FLEURS WER 22.13%).

pl-PL (17) · sv-SE (24) · cs-CZ (22) · nb-NO (103) · da-DK (25) · bg-BG (30) · fi-FI (26) ·
hr-HR (29) · zh-CN (4) · hu-HU (23) · ro-RO (20) · sk-SK (28) · et-EE (60)

## Tier 3 — Adaptation-ready (8 locales)

Supported but intended for fine-tuning before production use.

el-GR (21) · lt-LT (31) · lv-LV (61) · mt-MT (102) · sl-SI (62) · he-IL (64) · th-TH (32) ·
nn-NO (104)

> Empirical check (`make test-nemo-langs`, 2026-07-16): el/lt/lv/sl/he still work
> well on our samples (CER 0.10–0.24); **th-TH, however, produces near-empty output
> even on clean audio** — it really does need the fine-tuning the card recommends.
> mt-MT and nn-NO untested (no free audio samples found).

## Suite coverage (`make test-nemo-langs`)

**38/40 locales exercised, 37 OK** (2026-07-16): 33 base languages + the 4 regional
variants (en-GB, es-US, fr-CA, pt-PT — own prompt ids, tested on the same samples as
the base language) all OK; th-TH WEAK (expected); mt-MT and nn-NO with no audio.
102 real samples: Tatoeba (CC, 21 languages) + FLEURS validation (CC-BY-4.0, 13 languages).

## Language detection and output tag

- With `auto` (id 101) the model detects the language and **emits the locale tag** (e.g. `<it-IT>`)
  as a token in the output, after the terminal punctuation — the runtime strips it from the text
  and exposes it as `mynah_asr_result.lang`.
- The 39 language tags are special vocabulary tokens (e.g. `<it-IT>` = id 1279); the
  complete list is in the `added_tokens` of `tokenizer.json`.
- Aliases accepted in the dictionary: short forms (`it`, `de`, `fr`, …) map to the same id
  as the main locale.

## Extras (slots present in the dictionary but NOT among the 40 official ones)

The `prompt_dictionary` also includes ids for: zh-TW, id-ID, ms-MY, fa-IR, ur-PK, bn-IN,
ta-IN, te-IN, kn-IN, ml-IN, gu-IN, mr-IN, ne-NP, si-LK, km-KH, sw-KE, am-ET, ha-NG, yo-NG,
ig-NG, zu-ZA, af-ZA, and others (~65 slots total beyond the 40). These are training/fine-tuning
slots: undeclared quality, not to be exposed as "supported" without testing.

## Punctuation and capitalization

Native in the output for all languages (no post-processing).
