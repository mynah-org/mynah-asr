# Arabic specialist: a cross-script stress test of the frozen factory recipe (plan, 2026-10-24)

Status: PLAN ONLY. No data downloaded, no GPU used, nothing committed. Research by web + reading the repo
(finetune/README.md, finetune/eou/plain_asr.py, finetune/eou/prepare_mix.py, .work/release-final-2026-10-22.md s9-13).
Tags: **OBSERVED** = measured by us; **SOURCE** = stated by a cited page; **UNVERIFIED** = plausible, not checked;
**HYPOTHESIS** = our expectation.

## 0. TL;DR

- The "factory" is the EOU-120M plain recipe (ES-1: stock init + **stock English tokenizer, de-accented `[a-z' ]`
  targets**, whole encoder 4e-4, dec/joint 3e-4, WSD, bs 16, horizon = 4.60 passes; ES macro greedy 18.09 / CER 7.43,
  12,670 steps, 3 h 42 min on GB10; OBSERVED). Arabic cannot be expressed in `[a-z' ]`, so for the 120M the recipe
  **stops being a pure fine-tune**: a new SentencePiece vocabulary + new decoder embedding + new joint output layer.
- **Nemotron 3.5 streaming 0.6B already supports Arabic** (`ar-AR`, transcription-ready tier, FLEURS WER 12.0-13.2
  depending on chunk; 13k-token multilingual tokenizer, NVIDIA says no tokenizer change for supported languages;
  SOURCE: model card + discussion #4). For the 0.6B, Arabic is a normal in-vocabulary fine-tune, not a cross-script
  test. It is still essential here: **teacher** for the YouTube-pool filter and the **zero-shot reference**.
- Data: enough clean-licence Arabic exists (MASC CC-BY 4.0 ~1,000 h, CV17-ar CC0, FLEURS ar_eg CC-BY 4.0, YODAS2-ar
  CC-BY 3.0 ~290 h manual subs). **Granary has no Arabic.** The big broadcast corpora (MGB-2, QASR, SADA, Casablanca)
  are non-commercial and/or gated: eval-only or excluded.
- Recommendation: run it as **two screen arms** (T = new Arabic tokenizer with weight-preserving decoder/joint
  surgery; R = literal frozen recipe via a reversible `[a-z']` romanisation), then one full-horizon run of the winner.
  ~6-7 GPU-h on one GB10 end to end. Fallback cross-script language: **Russian** (Cyrillic; Granary-ru + CV + FLEURS,
  all permissive; Nemotron `ru-RU` teacher).

## 1. Datasets

| corpus | HF id (or source) | hours (approx.) | variety | licence | gated? | role here |
|---|---|---|---|---|---|---|
| Common Voice 17 ar | `fixie-ai/common_voice_17_0` config `ar` (the mirror `prepare_mix.py` already uses); official `mozilla-foundation/common_voice_17_0` now redirects to Mozilla Data Collective (account + ToS since Oct 2025) | ~158 h recorded / ~93 h validated in recent versions; train is much smaller after speaker split (UNVERIFIED for v17: ~28k train rows) | MSA read, many sentences partly diacritised | CC0 | mirror: no; official: MDC ToS | train (40 h cap or all, <= 20 clips/speaker) + **eval** (test 200, speaker-disjoint) |
| FLEURS ar_eg | `google/fleurs` config `ar_eg` | ~8-10 h total (train ~7 h) | MSA read by Egyptian speakers | CC-BY 4.0 | no | train (0.15 weight, as in ES-1) + **eval** (test) |
| MASC | `pain/MASC` (CC-BY 4.0, 87.7 GB, train/dev/test with clean `c` / noisy `n` type); `abdusah/masc` copy says NC | ~1,000 h YouTube, 700+ channels; MSA 569 h, Syrian 197, Egyptian 120, Jordanian 39, Saudi 31, Lebanese 23 | MSA + Levantine/Egyptian/Gulf | CC-BY 4.0 (pain/MASC card; licence of the original release to re-confirm at audit) | no | **main train pool** (MLS role, 40 h cut over many channels) + **eval** (test clean 200, test noisy 200) |
| YODAS2 ar | `espnet/yodas2` `ar000` (manual subs) / `ar100` (auto) | ar000 ~290 h (SOURCE: YODAS v1 table via search; UNVERIFIED for v2) | mixed YouTube | CC-BY 3.0 | no | train, **Nemotron-teacher filtered** (VP role in ES-1, t50) |
| Granary | `nvidia/Granary` | - | - | CC-BY 4.0 | - | **no Arabic** (25 European languages incl. ru/uk) |
| `espnet/yodas-granary` | - | - | - | CC-BY 3.0 | - | no Arabic (Granary languages only; UNVERIFIED but follows from Granary) |
| ClArTTS | MBZUAI ClArTTS (HF mirrors exist) | ~12 h, 1 speaker | Classical Arabic, diacritised | CC-BY 4.0 | no | exclude (single speaker, Classical) |
| Arabic Speech Corpus | `halabi2016/arabic_speech_corpus` | ~4 h, 1 speaker | MSA, Damascene accent | CC-BY 4.0 | no | exclude (TTS-style) |
| EveryAyah (Tarteel) | various | ~390 h (as used by NVIDIA) | Quranic recitation | permissive (UNVERIFIED per mirror) | no | exclude (domain) |
| MGB-2 | `QCRI/mgb2` | ~1,200 h transcribed (of ~2,000 h audio) | Al Jazeera broadcast, ~70 % MSA + dialects | QCRI-ALT research licence, **no commercial use** | **yes** (accept agreement) | eval-only if the agreement is signed; not needed |
| MGB-3 | arabicspeech.org (HF copy `MightyStudent/Egyptian-ASR-MGB-3`, provenance unclear) | ~16 h | Egyptian YouTube | restricted | yes (agreement) | skip |
| MGB-5 | arabicspeech.org | ~13-14 h | Moroccan YouTube | restricted | yes | skip |
| QASR | `QCRI/QASR` / arabicspeech.org | ~2,000 h | Al Jazeera, multi-dialect + code-switch | CC BY-NC-SA 4.0 (**NC**) | yes; download issues reported | skip |
| SADA | `m6011/sada2022` (also `MohamedRashad/SADA22`) | ~667 h (train ~647, test/dev ~20) | Saudi: Najdi, Hijazi, Khaliji, ... | CC BY-NC-SA 4.0 per the paper (the HF card's CC-BY 4.0 tag contradicts it: treat as **NC**) | no | **OOD eval only** (report, no band) |
| Casablanca | `UBC-NLP/Casablanca` | ~48 h, **only validation + test released** | 8 dialects: Algerian, Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, Yemeni; code-switch annotated | CC BY-NC-ND 4.0 (**NC, ND**) | no | **OOD eval only** per dialect (report, no band; scoring is not a derivative) |

Sources: [CV mirror](https://huggingface.co/datasets/fixie-ai/common_voice_17_0), [CV move to MDC](https://discourse.mozilla.org/t/datasets-a-public-api-and-more-coming-in-2025/145859),
[Arab Voices survey arXiv 2601.13319](https://arxiv.org/html/2601.13319v2), [FLEURS](https://huggingface.co/datasets/google/fleurs),
[pain/MASC](https://huggingface.co/datasets/pain/MASC), [abdusah/masc](https://huggingface.co/datasets/abdusah/masc),
[YODAS2](https://huggingface.co/datasets/espnet/yodas2), [YODAS paper](https://arxiv.org/html/2406.00899v1),
[Granary](https://huggingface.co/datasets/nvidia/Granary) / [arXiv 2505.13404](https://arxiv.org/pdf/2505.13404),
[ClArTTS](https://www.isca-archive.org/interspeech_2023/kulkarni23_interspeech.pdf), [ASC](https://huggingface.co/datasets/halabi2016/arabic_speech_corpus),
[QCRI/mgb2](https://huggingface.co/datasets/QCRI/mgb2), [MGB-2](https://arxiv.org/abs/1609.05625), [MGB-3](https://arxiv.org/pdf/1709.07276),
[MGB-5](https://www.researchgate.net/publication/335795406_The_MGB-5_Challenge_Recognition_and_Dialect_Identification_of_Dialectal_Arabic_Speech),
[QASR](https://huggingface.co/datasets/QCRI/QASR), [SADA](https://huggingface.co/datasets/m6011/sada2022) / [paper](https://ieeexplore.ieee.org/abstract/document/10446243),
[Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca) / [arXiv 2410.04527](https://arxiv.org/abs/2410.04527).

**Proposed frozen-definition mix (A2 analogue, weights unchanged):** MASC train clean 40 h (<= N clips / channel) 0.35
(MLS role), CV17-ar train <= 40 h (<= 20 / speaker) 0.30, YODAS2-ar000 teacher t50 40 h 0.20 (VP role),
FLEURS ar_eg train 0.15. ~120-127 physical h, mostly MSA with Levantine/Egyptian colour from MASC/YODAS.
Free from HF without an agreement: MASC (pain), CV mirror, FLEURS, YODAS2, Casablanca, SADA. Gated: MGB-2, QASR.

**harness-ar (sealed before training, model-free, SHA256SUMS):** CV17-ar test 200, FLEURS ar_eg test (all),
MASC test-clean 200, MASC test-noisy 200 -> 4-set macro, same shape as harness-es. OOD report-only: Casablanca test
per dialect (200 each), SADA test 200. Audit must check speaker/channel/video disjointness (MASC and YODAS share
YouTube; drop YODAS videos whose id appears in MASC) and CV sentence overlap between train and test.

## 2. Tokenizer

### 2.1 What the stock vocabularies cover
| model | stock tokenizer | Arabic? | consequence |
|---|---|---|---|
| EOU-120M (`parakeet_realtime_eou_120m-v1`) | SentencePiece 1024 + `<EOU>`/`<EOB>` = 1026, blank 1026 (OBSERVED in `plain_asr.py`); the factory feeds it `[a-z' ]` only | **no** (English pieces) | new vocabulary needed, or a romanisation front-end |
| Nemotron 3.5 streaming 0.6B | multilingual ~13k pieces, 40 locales, 128-dim language one-hot fused before the RNNT decoder | **yes**, `ar-AR` transcription-ready (FLEURS 12.03-13.17 WER) | fine-tune with `target_lang=ar-AR`, no tokenizer change; still verify on CPU that the pool encodes with zero `<unk>` |

Sources: [Nemotron card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
[tokenizer discussion #4](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/discussions/4).

### 2.2 The NeMo path for the 120M (arm T)
1. Train SentencePiece (unigram, 1024, same trainer spec as the stock model read via `sp_spec`, as
   `canary/tokenizer_it.py` does) on the **training-pool transcripts only**; append `<EOU>`/`<EOB>` with
   `eou/tokenizer_eou.py` -> vocab 1026, blank 1026. Keeping 1026 keeps every exported tensor shape identical for the
   Mynah pack.
2. `model.change_vocabulary(new_tokenizer_dir, "bpe")`. In NeMo `EncDecRNNTBPEModel.change_vocabulary` this
   **deletes and re-instantiates the whole decoder (prediction net) and the whole joint, and rebuilds the RNNT loss,
   WER and decoding** - no weights are carried over (SOURCE:
   [rnnt_bpe_models.py](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/rnnt_bpe_models.py)).
3. Surgery (proposed, mirrors the Canary `it` copy-back): only two tensors depend on V - the decoder embedding
   `(V+1) x H_pred` and the joint output `Linear(H_joint, V+1)`. Copy back from the stock model the prediction LSTM,
   the joint encoder/pred projections, and the **blank row** (embedding + output weight + bias); new token rows
   random (or mean of stock rows). Verify: copied tensors bit-identical; fresh `restore_from` works; Arabic round trip
   text -> ids -> text exact on the pool. HYPOTHESIS: keeping the blank prior and the LSTM avoids the cold
   decoder collapse seen in the 2026-10-09 experimental new-vocabulary EOU runs.

**Consequence:** with T the factory is no longer "same model, new data". Decoder embeddings and output rows are
learned from zero on ~120 h, so (a) the `--abort-wer 1000:85` guard was calibrated on stock-vocab runs and may kill a
healthy T run (declare a CER-based guard for T: abort if CER > 70 at step 1000), (b) the micro-overfit (32 utt,
1,000 steps) becomes a required pre-flight again, (c) the Mynah export path will for the first time emit multi-byte
UTF-8 pieces from this factory (the ES/IT/FR packs are `[a-z' ]`) - a detokeniser check is part of qualification.

### 2.3 Arm R: the literal frozen recipe via romanisation
A reversible, prefix-free lowercase transliteration into `[a-z']` (apostrophe as the only modifier, e.g. emphatics
and interdentals as `letter'`), round-trip tested on 100 % of the pool, keeps the stock tokenizer, decoder and joint
untouched: the recipe is literally frozen. Expected cost (HYPOTHESIS): English pieces fit the romanised strings badly
-> longer U per second (more joint T x U, more VRAM), worse WER than T. Its value is diagnostic: it measures how much
of the factory's Latin-script success comes from the pretrained decoder vs the acoustics. Scoring is always on
**de-romanised Arabic**, with the same normaliser as T.

### 2.4 Normalisation (training targets vs scoring)
| issue | codepoints | training targets (both arms) | scoring (primary = leaderboard-style) | strict-lenient (secondary) |
|---|---|---|---|---|
| tashkeel / diacritics | U+064B-U+0652, U+0670, U+0653-U+0655 | strip | strip | strip |
| tatweel | U+0640 | strip | strip | strip |
| alef / hamza forms | أ إ آ ٱ -> ا | fold (YouTube orthography is inconsistent) | fold hamza/madda forms | fold |
| hamza on carriers | ؤ ئ ء | keep | fold to bare per leaderboard (UNVERIFIED exact rule; reimplement and freeze ours) | fold |
| alef maqsura / ya | ى vs ي; Persian ی U+06CC -> ي; ک -> ك | map Persian forms; keep ى | keep | ى -> ي |
| ta marbuta | ة vs ه | keep | keep | ة -> ه |
| digits | Arabic-Indic ٠-٩ U+0660-9, Extended ۰-۹ U+06F0-9, Western | drop clips with any digit (ES-1 rule) | Eastern -> Western | same |
| punctuation | ، ؛ ؟ and Latin | strip | strip | strip |
| bidi / joiners | U+200E/F, U+061C, U+200C/D | strip | strip | strip |
| code-switching (Latin words, Fr/En) | `[A-Za-z]` | drop the clip (alphabet restriction, as in the factory); report the drop rate per source | keep refs as-is; report a Latin-word subset WER on Casablanca | - |

The leaderboard protocol: remove punctuation and diacritics, normalise characters with hamza and madda, convert
Eastern to Western numerals (SOURCE: [arXiv 2412.13788](https://arxiv.org/html/2412.13788)). Arabic WER is inflated
by clitics (و، ب، ل، ال attach to words): WER/CER ~3 is normal (leaderboard Conformer CV: 8.80 / 2.77). Report CER
always; it is the primary acoustic signal.

## 3. Expected failure modes and what each tells us

| # | symptom (measurable) | likely cause | what it says about the factory |
|---|---|---|---|
| F1 | empty / blank-only outputs, non-blank frame % near 0, empty rate > 5 % (T) | re-initialised decoder/joint; blank dominates | the recipe relies on a pretrained prediction net; new scripts need decoder warm-up (surgery or a frozen-encoder stage). Generality: conditional |
| F2 | fluent Arabic unrelated to audio (low audio dependence, repeated n-grams) | decoder learns the text prior faster than the encoder adapts | dec/joint LR ratio is Latin-tuned; not general |
| F3 | WER high but CER fine (WER/CER > 4) | clitic segmentation, orthography (ة/ه, ى/ي, hamza) | evaluation/normalisation problem, not a recipe problem |
| F4 | CV/FLEURS fine, MASC much worse (> 2x) | YouTube domain, dialect, subtitle label noise | data problem; the recipe transfers, the mix does not |
| F5 | substitutions dominated by emphatic/pharyngeal pairs (ص/س، ط/ت، ض/د، ح/ه، ع/ء، ق/ك) | English-pretrained encoder lacks the contrasts | acoustic transfer limit of an English 120M; argues for a longer horizon or a multilingual init, not for recipe tuning |
| F6 | R ~ T (within 2 WER) | stock decoder not the bottleneck | strongest generality result: a romanisation front-end is enough |
| F7 | R >> T | English pieces cannot carry Arabic | tokenizer surgery is a mandatory factory step for non-Latin scripts |
| F8 | teacher filter keeps < 30 % of YODAS-ar | Nemotron-ar weaker on YouTube/dialect, or bad subtitles | the t50 filter does not transfer as-is; re-derive its threshold per language |
| F9 | peak VRAM / ms per step >> ES-1 | longer U (R arm) or different clip durations | the horizon rule is in steps, not cost; record xRT |
| F10 | NeMo vs Mynah transcripts differ on non-ASCII | detokeniser/UTF-8 handling in the C engine | serving side of the factory is not script-general |

## 4. Reference numbers and pre-registered bands

### 4.1 Literature (WER / CER, %)
| model | params | CV ar | FLEURS ar | MASC clean | MASC noisy | MGB-2 | SADA | Casablanca | normalisation | source |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA Conformer-CTC-large-ar (+LM) | ~120M (UNVERIFIED) | 8.80 / 2.77 (CV18) | - | 23.74 / 5.63 | 34.29 / 11.07 | 17.20 / 6.87 | 44.52 / 23.76 | - | leaderboard | [2412.13788](https://arxiv.org/html/2412.13788) |
| Whisper large-v3 | 1.55B | 17.83 / 5.74 | ~12-16 (UNVERIFIED, normalisation-dependent) | 24.66 / 7.24 | 34.63 / 12.89 | 16.26 / 7.74 | 55.96 / 34.62 | 71.81 / - | leaderboard | [2412.13788](https://arxiv.org/html/2412.13788), [blog](https://cohere.com/blog/transcribe-arabic) |
| Whisper large-v2 / medium / small (CER on CV) | - | 7.44 / 10.38 / 15.75 | - | - | - | - | - | - | leaderboard | [2412.13788](https://arxiv.org/html/2412.13788) |
| Seamless M4T v2 large | 2.3B | 21.70 / 6.24 | - | 25.04 / 7.19 | 33.24 / 11.92 | 20.23 / 9.37 | 62.52 / 37.61 | - | leaderboard | same |
| MMS-1b-all | 1B | 26.52 / 7.21 | - | 38.82 / 10.36 | 57.33 / 19.76 | 39.16 / 13.48 | 77.48 / 37.50 | - | leaderboard | same |
| NVIDIA `stt_ar_fastconformer_hybrid_large_pcd_v1.0` (RNNT+CTC, 1,100 h, CC-BY 4.0) | 115M | 8.20 (own MCV17 test, no diacritics); 25.60 with diacritics+punct | 5.01; 12.94 with diacritics+punct | 8.50 (own MASC test, no diacritics) | - | - | - | - | own (diacritics-stripped / -kept) | [card](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0), [arXiv 2507.13977](https://arxiv.org/html/2507.13977v1) |
| Nemotron 3.5 streaming 0.6B (`ar-AR`) | 0.6B | - | 12.03 (1.12 s) - 13.17 (80 ms) | - | - | - | - | - | "aligned" normalisation, unspecified | [card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |
| Cohere Transcribe Arabic (2026-07) | - | 5.82 | - | 15.54 | 27.07 | 15.54 | 37.47 | 49.71 | leaderboard | [blog](https://cohere.com/blog/transcribe-arabic) |

Note: the NVIDIA 115M's MASC 8.50 (own test subset) vs the leaderboard's 23.74 for an NVIDIA Conformer on MASC clean
shows how much the test subset and normaliser move numbers: **only our own sealed harness + our own frozen normaliser
decide**, and we decode Nemotron-ar zero-shot on it as the in-house reference. The 115M FastConformer is the closest
like-for-like (same size class, ~9x our hours, full training): it is a ceiling, not a target.

### 4.2 Bands (pre-register before any Arabic statistic is seen)
Primary: harness-ar macro (CV, FLEURS, MASC-clean, MASC-noisy), greedy, 0.5 s flush, leaderboard-style normaliser.
A band needs **both** WER and CER inside it; the worse of the two decides.

| band | macro WER | macro CER | reading |
|---|---|---|---|
| STRONG | <= 30 | <= 9 | the factory generalises across script (ES-1 18.09 / 7.43 scaled by the Arabic WER/CER ratio) |
| OK | 30-40 | 9-13 | transfers; usable specialist after one more lever |
| WEAK | 40-55 | 13-20 | acoustics transfer, something structural is off (F2-F5) |
| FAIL | > 55, or CER > 20, or abort | | a Latin-script assumption broke (F1/F7) |

HYPOTHESIS for T at full horizon: CV 18-28, FLEURS 12-20, MASC-clean 30-40, MASC-noisy 40-50 -> macro ~28-35 (OK, STRONG
edge). Report alongside: Nemotron-ar zero-shot and stock EOU-120M zero-shot (expected ~100 WER, script mismatch) on
the same harness; OOD Casablanca / SADA per dialect, no band.

## 5. Staged plan (one GB10; times from ES-1 / CF-E1, OBSERVED rates: ~1.0 s/step, 1,500 steps ~26 min)

| stage | what | wall | GPU | gate to the next stage |
|---|---|---|---|---|
| **a. Audit (CPU on the box)** | download (MASC clean shards only, CV mirror ar, FLEURS ar_eg, YODAS2 ar000 shards, Casablanca/SADA test); licence record per source; hours / clips / speakers / channels; character inventory; normalisation drop rates (digits, Latin, unmapped chars); train/test disjointness (speaker, video id, CV sentence text); SentencePiece 1024 train + round trip; romanisation round trip; Nemotron tokenizer coverage of the pool; seal harness-ar | 3-5 h (download-bound; MASC is 87.7 GB in full) | ~0.7 h: Nemotron 300-clip CLI gate + YODAS t50 teacher filter; ~10 min micro-overfit for T and R | **G-a:** >= 100 physical h across >= 3 permissive sources; drop rate <= 15 % per source; round trips 100 %; 0 `<unk>`; harness sealed; MASC licence confirmed CC-BY (else MASC -> eval only, and if the pool falls below 100 h -> **fallback language**); both micro-overfits memorise with audio-dependent output |
| **b. Screen** | 2 arms x 1,500 steps, ES-1 recipe otherwise unchanged: **T** (new vocab + surgery), **R** (romanised, stock vocab); one eval at the last step on harness-ar val (200/set) | ~1.2 h | ~1.1 h | **G-b:** pick the arm with lower macro CER; GO if it has macro WER <= 60 and CER <= 22 at 1,500 steps (ES-1's 1,500-step R was 30.82 vs 18.09 final, ratio 0.59), empty rate <= 5 %, audio dependence OK. If T fails and R passes, the cause is the decoder re-init (F1); add **T0** (change_vocabulary without surgery) only if needed to attribute it. Both fail -> stop, write up (that is the result) |
| **c. Full horizon** | winner arm, horizon rule (steps = round(4.60 x H_phys / h_step), cap 23,762; ~12-13k steps for ~125 h); final = last step; probes greedy / beam-8 / oracle / CER, raw EOF + 0.5 s flush; Nemotron-ar and stock zero-shot on the harness; OOD dialect report; Mynah export + NeMo/Mynah parity on all harness clips | ~4.5-5 h | ~3.7-4 h | band from 4.2 is the result; stage 2 (`<EOU>`) only if macro <= 40 and >= 45 min of box time remain, v4 recipe unchanged |

Total ~6-7 GPU-h, ~1.5 working days. Every remote job time-boxed; archive run + harness seal to the Hub before
reading results. Bucketing / joint-subbatch speedups stay OFF (not part of the frozen recipe).

### Fallback cross-script language: Russian
If G-a fails (licence or < 100 h): **Russian**. Cyrillic is a new script for the stock 120M tokenizer (same
tokenizer-surgery question as Arabic) but alphabetic and left-to-right with no diacritics/clitic issues, so failures
are easier to attribute; data is permissive and large: **Granary-ru** (CC-BY 4.0, plus `espnet/yodas-granary` ru
pre-segmented clips the pipeline already reads), CV17-ru (CC0), FLEURS ru_ru (CC-BY 4.0); Nemotron supports `ru-RU`
(teacher + reference). No MLS/VoxPopuli-ru transcripts (UNVERIFIED for VoxPopuli), so YODAS-Granary takes the large-pool
role. Ukrainian is the same case with less data. Hindi (Devanagari, Nemotron `hi-IN`) would be a harder second
cross-script test but its open data is thinner.

## 6. Biggest risks
1. **Decoder re-init (T)** may need a recipe change (warm-up / surgery) - then the claim is "factory + a tokenizer step",
   not "frozen factory". The R arm exists to make that conclusion measurable rather than argued.
2. **Licences**: MASC licence is inconsistent between HF copies; most dialect-rich data is NC/gated, so the specialist
   will be mostly MSA; dialect generalisation can only be reported (Casablanca/SADA), not trained.
3. **Label noise**: MASC/YODAS are YouTube subtitles; the teacher filter threshold is Latin-calibrated (F8).
4. **Metric drift**: Arabic WER moves by 10+ points with the normaliser; only our frozen normaliser and our harness count.
5. **Serving**: first non-ASCII vocabulary through the Mynah export/detokeniser (F10).
