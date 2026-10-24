# EOU-120M beyond English / Latin script: prior-art survey (2026-10-24)

Scope: has anyone adapted `nvidia/parakeet_realtime_eou_120m-v1` (cache-aware FastConformer-RNNT,
inline `<EOU>`), or a comparable RNNT with an inline end-of-utterance token, to non-English and
especially non-Latin languages (AR, JA, KO, RU, HI, HE, TH, ZH)? Web research only, no compute.

Classes: **A** = same NVIDIA EOU-120M checkpoint; **B** = same FastConformer/RNNT family, different
checkpoint; **C** = other RNNT/ASR with inline EOU token; **D** = separate EOU/turn classifier
(prior art, NOT equivalent).

Wording rule: "no public evidence found" means a search found nothing, not proof it was never done.
UNVERIFIED = source not opened or claim comes from a third party only.

---

## 1. Evidence table

| # | Item | Class | Languages / script | What it shows | URL |
|---|------|-------|--------------------|---------------|-----|
| 1 | `nvidia/parakeet_realtime_eou_120m-v1` card | A | EN only | 120M, 17 layers, att ctx [70,1], RNNT, lowercase no punctuation, `<EOU>` emitted inline; EOU latency p50 160 ms / p90 280 ms on TTS DialogStudio + 3 s silence; NVIDIA Open Model License; training data incl. AMI, DialogStudio, Granary, Speech Commands, LibriTTS, NeMo ASR Set 3.0 (10k h). EOU label method not documented. | https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 |
| 2 | HF discussion "Multilingual support?" | A | - | NVIDIA staff (Nov 13 2025): "we plan to extend to multilingual, and we will also share the scripts". Community asked for FR, RU, DE, ES, IT, JA, AR, PT. Closed Dec 1 2025, no release linked. | https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/discussions/2 |
| 3 | HF search `realtime_eou`, `eou_120m` (23 repos) + "finetunes" tree of base model (7 repos) | A | EN | Every listed derivative is a format conversion or mirror (CoreML, ONNX, INT8, GGUF, sherpa-onnx, fp16) or a duplicate (CollectionStudio = "Duplicated from nvidia/..."). **No public evidence found of any language fine-tune of EOU-120M**, Latin or non-Latin. | https://huggingface.co/models?search=eou_120m , https://huggingface.co/models?other=base_model:finetune:nvidia/parakeet_realtime_eou_120m-v1 , https://huggingface.co/CollectionStudio/parakeet_realtime_eou_120m-v1 |
| 4 | FluidInference CoreML card | A | EN | States joint = 1027 classes "1024 tokens + EOU + SOS + blank", EOU id 1024. Third-party reading; the third special is probably `<EOB>` (cf. #5). UNVERIFIED against the .nemo. | https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml |
| 5 | NeMo `examples/asr/asr_eou` (README, `speech_to_text_rnnt_eou_train.py`, eval) | B (recipe) | EN example (base = nemotron-speech-streaming-en-0.6b) | Official public EOU recipe: `EncDecRNNTBPEEOUModel`; appends `<EOU>`,`<EOB>` to SentencePiece (1024 -> 1026, ids 1024/1025) via `add_special_tokens_to_sentencepiece.py`; new-row init `token_init_method` in {min,max,mean,constant}, `token_init_bias_value` default -1000; manifests need `sou_time`/`eou_time`, optional `is_backchannel`; silence padding (post-pad >= 3 s, max 6 s), white noise p=0.9, SNR 0-20 dB; metrics: WER, EOU latency, early-cutoff rate, miss rate. No non-English example. | https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/asr/asr_eou , https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/main/examples/asr/asr_eou/README.md |
| 6 | `echodict/asr_rnnt_eou_from_scratch` (HF dataset) | B (notes) | docs in Chinese | Chinese-language walkthrough of the NeMo ASR-EOU script (EOU/EOB, vocab expansion, weight transfer). 2 samples; no evidence of an actual Chinese EOU model being trained. Licence not stated. | https://huggingface.co/datasets/echodict/asr_rnnt_eou_from_scratch |
| 7 | Nemotron 3.5 ASR Streaming 0.6B | B | 40 locales incl. AR, HI, JA, KO, RU (tier 1), ZH (tier 2), HE, TH (tier 3 "adaptation-ready") | Cache-aware FastConformer-RNNT, 24 layers, 128-dim one-hot language prompt, emits language tag after terminal punctuation; reported ar 12.03 % WER @1.12 s, ja 11.48 % CER, ko 7.12 % CER, ru 9.17 %, hi 6.81 %, zh 19.28 % CER. **No EOU token** documented. OpenMDW-1.1. | https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b |
| 8 | Third-party Nemotron 3.5 guide | B | same | Tokenizer reported as single BPE "13 087 tokens + blank"; no EOU; suggests terminal punctuation or VAD as boundary. UNVERIFIED vs .nemo. | https://soniqo.audio/guides/nemotron |
| 9 | `oddadmix/nemotron-3.5-asr-arabic-dialectal` | B | AR dialects | Arabic FT of Nemotron 3.5 (~40k clips, diacritics removed); WER 0.592 -> 0.422 on 932 clips. No EOU. "Private/internal" data. | https://huggingface.co/oddadmix/nemotron-3.5-asr-arabic-dialectal |
| 10 | Nemotron 3.5 Kenyan adaptation (arXiv 2607.18912) | B | Kikuyu, Dholuo, Kalenjin (Latin) | Full FT keeping tokenizer, prompt conditioning, streaming decoder; Unicode normalisation stressed. No tokenizer surgery, no EOU. | https://arxiv.org/abs/2607.18912 |
| 11 | NVIDIA HF blog: fine-tuning Nemotron 3.5 | B | EL, BG examples | ~290 h -> 31-32 % rel. WER gain; no tokenizer or EOU guidance. | https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr |
| 12 | Parakeet-RNNT-1.1B multilingual (Riva/NIM, "universal tokenizer") | B | 25 locales incl. ar-AR, ja-JP, ko-KR, ru-RU, hi-IN, th-TH, he-IL (+ zh in NIM matrix) | FastConformer-RNNT ~1.1B, >90k h; "Individual tokenizers are trained for each language and then merged into a single tokenizer"; output includes case + punctuation; data normalised to "upper-cased, lower-cased, punctuated, and spoken forms" (CTC concat card). Vocab size **not disclosed**. NVIDIA Community Model License; NGC/Riva distribution only (no HF weights found). NGC container marked "no longer supported". | https://catalog.ngc.nvidia.com/orgs/nvidia/riva/models/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal/- , https://catalog.ngc.nvidia.com/orgs/nvidia/riva/models/parakeet-ctc-riva-1-1b-unified-ml-cs-concat/- , https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/parakeet-1-1b-rnnt-multilingual |
| 13 | build.nvidia.com model card for #12 | B | - | JS-rendered; content could not be extracted. **UNVERIFIED** (claims above come from NGC pages instead). | https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr/modelcard |
| 14 | NIM ASR support matrix | B / D | as #12 | Profiles: default (auto LID), prompt (client language), indic (en+hi). Endpointing in NIM is **Silero VAD**, not an inline token. | https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html |
| 15 | NVIDIA forum: Arabic in Parakeet-RNNT multilingual | B | AR | Arabic audio transcribed as Bengali/Tamil = wrong (indic) profile; fix: "multi"/"prompt" profile. | https://forums.developer.nvidia.com/t/arabic-language-support-in-parakeet-rnnt-multilingual-asr-transcription-issue-tts-inquiry/363161 |
| 16 | NeMo issue #14921 aggregate vs single tokenizer | B | - | Community answer: Canary-1B-v2 and Parakeet-TDT-0.6B-v3 use a **single** tokenizer; aggregate (~1500/lang merged) only worth it for few, low-overlap languages; "except for Chinese, Japanese, and Korean, a vocabulary size of 1500 is unnecessary". Not an NVIDIA-official statement. | https://github.com/NVIDIA-NeMo/NeMo/issues/14921 |
| 17 | `nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0` | B | AR (MSA) | FastConformer Large hybrid RNNT+CTC, ~115M, **SentencePiece 1024**, **no diacritics**, period / Arabic comma / Arabic question mark; ~760 h (MASC 690, CV17 65, FLEURS 5); RNNT WER MASC 11.46, CV 10.20, FLEURS 8.18; CC-BY-4.0; offline (not cache-aware). | https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0 |
| 18 | `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` | B | AR (with diacritics) | Same arch, SP 1024, **with diacritics**; ~1100 h (adds EveryAyah 390 h Quranic); WER MASC 16.67, CV 25.60, FLEURS 12.94, EveryAyah 6.55; CC-BY-4.0. Third-party arena strips diacritics before scoring. | https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 , https://github.com/kadirnar/arabic-arena/issues/52 |
| 19 | `scb10x/typhoon-asr-realtime` | B | TH | FastConformer-Transducer 114M fine-tuned from `stt_en_fastconformer_transducer_large` on 10k h Thai (English base -> Thai script, i.e. tokenizer replaced). No EOU. CC-BY-4.0. Tokenizer size not stated. | https://huggingface.co/scb10x/typhoon-asr-realtime |
| 20 | jeremy110/Finetune_Nemo_ASR | B | EN+HI (private data) | Recipe: merge EN SentencePiece with new-language pieces, `change_vocabulary`, copy old decoder/joint rows back; or replace vocabulary outright. No released checkpoint. | https://github.com/jeremy110/Finetune_Nemo_ASR |
| 21 | Japanese FastConformer report (NeMo discussion #8473) | B | JA | RNNT with SentencePiece unigram 3000 for Japanese. | https://github.com/NVIDIA-NeMo/Speech/discussions/8473 |
| 22 | arXiv 2606.24169 "Data Scale, Not Latency, Shapes Cross-Lingual Encoder Transfer in Streaming ASR" | B | 8 European | 0.6B cache-aware FastConformer transducer: EN-only vs multilingual init gap +4.21 pp at 100 h -> +0.20 pp at 2500 h (FLEURS, 160 ms); halves per doubling of data. No EOU, Latin only. | https://arxiv.org/abs/2606.24169 |
| 23 | Li et al., "A Language Agnostic Multilingual Streaming On-Device ASR System" (Google, Interspeech 2022, arXiv 2208.13916) | **C** | 9 locales incl. **zh-TW, ja-JP** (non-Latin) + en/fr/de/es/it | RNN-T with a single 16K WPM output; EOU token appended to every training WPM sequence (RNN-T EP / decoder endpointing) + FastEmit, no extra late penalty needed. Adding EOU **degraded WER**; fix = two-stage training: train ASR, freeze, init a separate **EOU Joint Layer** from the original joint, fine-tune only it; decode with original joint, read EOU prob from the new one. Also notes EOU-as-final-token teaches the model to stop emitting -> early cutoffs. | https://arxiv.org/abs/2208.13916 , https://www.isca-archive.org/interspeech_2022/li22n_interspeech.pdf |
| 24 | Chang et al., "Joint Endpointing and Decoding with End-to-End Models" (Google, ICASSP 2019) | C | EN voice search | Original RNN-T `</s>` end-of-query token; beam ends when emitted; early/late penalties. PDF not opened (OpenReview captcha) -> details UNVERIFIED. | https://www.researchgate.net/publication/332791270_Joint_Endpointing_and_Decoding_with_End-to-end_Models |
| 25 | Lu et al., "Endpoint Detection for Streaming E2E Multi-talker ASR" (arXiv 2201.09979) | C | EN | EOS token as output unit in SURT transducer + latency penalty. | https://arxiv.org/abs/2201.09979 |
| 26 | "Improving endpoint detection in E2E streaming ASR for conversational speech" (arXiv 2505.17070) | C | EN (Switchboard) | Transducer with end-of-word token + delay penalty + auxiliary frame-level VAD to fight early cut-off and late EOS. | https://arxiv.org/abs/2505.17070 |
| 27 | "The Trade-off Was in the Labels: Causal Supervision for Turn-Aware Streaming ASR" (arXiv 2609.04225) | C (not RNNT) | EN only | LoRA on Qwen3-ASR-0.6B, in-transcript `<END_SPEECH>` tokens; labels must be causally computable (fire iff semantically complete AND >= 0.3 s silence). Reports stock EOU-120M "rarely fires" on their benchmark. Non-English = future work. | https://arxiv.org/html/2609.04225 |
| 28 | Deepgram Flux Multilingual (proprietary) | C? | 10 incl. HI, RU, JA | "Integrated" end-of-turn in the STT model; mechanism (token vs head) not disclosed. Press release 403 -> UNVERIFIED beyond secondary press. | https://siliconangle.com/2026/04/29/deepgram-expands-flux-10-languages-mid-call-switching-voice-agents/ , https://www.businesswire.com/news/home/20260429043418/en/Deepgram-Launches-Flux-Multilingual-The-Worlds-First-Multilingual-Conversational-Speech-Recognition-Model |
| 29 | Pipecat Smart Turn v3.x | D | 23 langs incl. **AR, KO, JA, ZH, HI, RU** (no HE, TH) | Audio-only Whisper-Tiny + linear head (~8M), BSD-2; reported (v2 blog) AR acc 88.6 % vs JA 96.8 %. | https://github.com/pipecat-ai/smart-turn , https://www.daily.co/blog/smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/ |
| 30 | LiveKit turn-detector (multilingual) | D | 14 incl. AR, HI, JA, KO, ZH | Text (transcript) model, per-language thresholds. | https://docs.livekit.io/agents/logic/turns/turn-detector/ , https://huggingface.co/livekit/turn-detector |
| 31 | `Moustafa3092/livekit-turn-detector-arabic` | D | AR (MSA, Gulf) | Qwen2-0.5B LoRA, 57,475 text samples (20,194 complete / 37,281 incomplete), Apache-2.0; dataset not linked. | https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic |
| 32 | `Waqf-AI/egyptian-arabic-eot-qwen35-0.8b` | D | AR (Egyptian) | Text-only, two decision tokens `<|turn_done|>`/`<|turn_continue|>`; 21,280 train ex.; ROC-AUC 0.955; Apache-2.0; dataset **not public** ("None public yet"). | https://huggingface.co/Waqf-AI/egyptian-arabic-eot-qwen35-0.8b , https://huggingface.co/Waqf-AI |
| 33 | `Estonel/turnlet-bert-multilingual-eou` | D | EN, HI, ES | DistilBERT text EOU, distilled from Qwen teacher; licence blank. | https://huggingface.co/Estonel/turnlet-bert-multilingual-eou |
| 34 | AraBERT / SaudiBERT EOU detectors | D | AR | Searched; no public EOU detector based on AraBERT/SaudiBERT found (only other NLP tasks). Arabic EOU detectors found are Qwen-based (#31, #32). | https://arxiv.org/html/2405.06239v1 |
| 35 | NeMo-Speech.cpp, parakeet.cpp (mudler), parakeet-rs, sherpa-onnx/icefall | runtime | - | Runtimes support EOU-120M as English model (parakeet.cpp: "end-of-utterance events (`--stream`)"); NeMo-Speech.cpp endpointing is silence-rule based; sherpa/icefall endpointing is rule-based (silence thresholds). No multilingual inline-EOU checkpoint found. | https://github.com/mudler/parakeet.cpp , https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/asr/configuration.md , https://github.com/k2-fsa/sherpa-onnx |
| 36 | FluidAudio issue #1007 | A (runtime) | EN | EOU-120M fires mid-sentence on pauses; runtime stops decoding after `<EOU>`; asks for post-EOU contract. | https://github.com/FluidInference/FluidAudio/issues/1007 |

## 2. Closest prior art

1. **For "same model, new language": none public.** No HF derivative, GitHub repo, paper or forum
   post shows EOU-120M fine-tuned to any non-English language (#2, #3). NVIDIA has stated intent
   (Nov 2025) but no multilingual EOU checkpoint or EOU-specific multilingual recipe was found as of
   2026-10-24.
2. **For "inline EOU in a multilingual non-Latin RNN-T": Google 2022 (#23)** is the closest: one RNN-T,
   shared 16K wordpieces over 9 locales including Chinese (zh-TW) and Japanese, EOU token appended to
   every target. Differences: trained from scratch on 214k h proprietary data, multilingual from day
   one, no tokenizer transplant, no Arabic, and they ultimately moved EOU to a separate joint layer.
3. **For "English FastConformer -> non-Latin script via tokenizer replacement": Typhoon Thai (#19)**
   and NVIDIA's own Arabic FastConformer Large models (#17, #18; base not stated). Both have no EOU.
4. **For "Arabic streaming RNN-T": Nemotron 3.5 (#7, #9)** - Arabic is tier-1, but there is no EOU
   token; boundary = punctuation/VAD.
5. **Recipe-level: NeMo `asr_eou` (#5)** - the public way to add `<EOU>`/`<EOB>` to any NeMo RNN-T,
   only demonstrated on English.

## 3. What is genuinely new in our experiment

- First public-evidence attempt (as far as this search reaches) to carry the *EOU-120M checkpoint*
  (inline `<EOU>`, 120M, [70,1] context) to a language whose script the stock tokenizer cannot
  represent.
- A controlled A/B of **romanisation into the frozen tokenizer (R)** vs **same-size native
  SentencePiece with only vocabulary-dependent rows re-initialised (T)** on an EOU-emitting
  transducer. Prior romanisation-vs-native comparisons exist only for CTC/offline models whose
  pretraining already saw Arabic (Quranic ASR, XLSR-53: native undiacritised WER 0.08 vs Buckwalter
  0.41 vs English transliteration 0.38, https://arxiv.org/html/2606.19747); no public result for an
  English-only encoder, and none measuring EOU latency / early cutoff under either arm.
- Measuring whether the `<EOU>` behaviour (latency, early-cutoff) survives re-initialising the
  decoder embedding and joint output rows - prior art (#23) only reports that adding EOU costs WER,
  not what happens when the vocabulary around a pre-trained EOU is replaced.
- Small-model regime: 120M English encoder with presumably modest Arabic data, where #22 predicts the
  English-only-init penalty is largest.

## 4. Tokenizer lessons for arms R and T

1. **Vocab layout to preserve (both arms).** EOU-120M is reported as 1024 SentencePiece + specials,
   `<EOU>` id 1024 (#4); the NeMo recipe appends `<EOU>`,`<EOB>` at 1024/1025 (#5). For T, train
   the Arabic SP at 1024 and re-append the two specials at the same ids so the EOU rows (decoder
   embedding row, joint output row + bias) can be **copied, not re-initialised** - only rows 0..1023
   are vocabulary-dependent. Verify the actual special-id map from the .nemo before surgery.
2. **1024 is a validated Arabic size.** NVIDIA's own Arabic FastConformer Large models use SP-1024
   (#17, #18), same order of size as EOU-120M; community guidance says >1500 is only needed for
   CJK (#16), Japanese RNNT used 3000 (#21). So T at 1024 is not under-sized for Arabic; it would be
   for JA/ZH.
3. **Normalisation: strip diacritics.** NVIDIA ships both a no-diacritics (pc) and a diacritics (pcd)
   Arabic model; pcd has much higher WER on general sets (CV 25.6 vs 10.2) and its gains come from
   Quranic data; arenas strip diacritics before scoring (#18). Our stock target is lowercase/no
   punctuation, so for T normalise to: no tashkeel, no tatweel, unify alef variants/ya/ta-marbuta
   policy explicitly, drop punctuation (EOU-120M emits none). For R the same normalisation must run
   *before* romanisation or the mapping is not reversible on real text.
4. **Ready-made Arabic SP-1024 + text-side weights exist (CC-BY-4.0).** `stt_ar_fastconformer_hybrid_large_pc_v1.0`
   provides an Arabic SentencePiece-1024 plus an RNNT prediction network and joint trained on it.
   If its decoder dims (pred_hidden, joint_hidden) match EOU-120M (UNVERIFIED - check both
   `model_config.yaml`), a third T variant can initialise the Arabic rows (decoder embedding,
   joint output rows) from that model instead of random/mean - a vocab-aligned init. Its encoder is
   offline (not cache-aware) so only text-side parts are transplantable; `joint.enc` must stay
   EOU-120M's.
5. **Parakeet-RNNT-1.1B multilingual cannot practically inform surgery.** Its tokenizer is "per-language
   tokenizers merged into a universal one" (#12), but vocab size is undisclosed, weights are on
   NGC/Riva only under the NVIDIA Community Model License, the container is deprecated, and the
   model is cased+punctuated. Nemotron 3.5 (single ~13k BPE, cased, punctuated, prompt-conditioned;
   #7, #8) is a better source of an Arabic-capable tokenizer, but its pieces are not 1024-sized and
   would require resizing every vocab-dependent row - closer to arm T at a larger vocab than to a
   row-aligned init. Lesson taken from it: a single shared tokenizer is what NVIDIA now uses (#16).
6. **R risks (from evidence).** Romanised Arabic Buckwalter-style labels underperform native script
   on a multilingual-pretrained encoder (WER 0.41 vs 0.08) (https://arxiv.org/html/2606.19747);
   one-to-many romanisation also lengthens target sequences in the stock [a-z' ] pieces, which
   increases RNNT emissions per frame and may shift `<EOU>` timing. Measure tokens/second of audio
   for R vs T and report it alongside EOU latency.
7. **Init of new rows.** NeMo exposes `token_init_method` (min/max/mean/constant, bias -1000 for
   constant) only for the special tokens (#5); for T's 1024 re-initialised rows the community recipe
   relies on `change_vocabulary` defaults (#20). Mean-init of embeddings + bias set so blank stays
   dominant is the conservative choice; record it as a pre-registered variable.

## 5. Usable Arabic EOU / continuation datasets (licences as stated by source)

| Dataset | Content | Use for us | Licence | URL |
|---------|---------|-----------|---------|-----|
| pipecat-ai smart-turn-data v3.1/v3.2 (Arabic rows) | Audio turns labelled `endpoint_bool` (complete vs cut-off), `midfiller`, `endfiller`, `synthetic` flag; many non-English rows TTS; no transcripts | EOU / early-cutoff test set (pause-inside-turn vs end) | CC-BY-4.0 | https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train |
| kadirnar/smart-turn-arabic-asr-benchmark | 7,396 Arabic clips, 17.6 h, from smart-turn v3.1, ROVER pseudo-references, mostly Egyptian | Ready-made Arabic subset with pseudo transcripts | CC-BY-4.0 | https://huggingface.co/datasets/kadirnar/smart-turn-arabic-asr-benchmark |
| MASC | ~1000 h YouTube, multi-dialect | ASR training; derive `sou_time`/`eou_time` from segments + silence padding | CC-BY-4.0 on IEEE DataPort; HF mirror lists NC - check | https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus , https://huggingface.co/datasets/abdusah/masc |
| Common Voice Arabic | Read sentences | Complete-utterance positives with clean ends | CC0 (Mozilla CV standard; not re-checked here) | https://commonvoice.mozilla.org |
| FLEURS ar_eg | Read sentences | Eval (also used by NVIDIA AR models) | CC-BY-4.0 (not re-checked here) | https://huggingface.co/datasets/google/fleurs |
| SADA (Saudi) | Conversational TV audio with segments | Dialogue-style EOU, research only | CC BY-NC-SA 4.0 (per search snippet, UNVERIFIED) | https://huggingface.co/datasets?search=sada |
| Casablanca | 8 dialects, transcribed | Dialect eval | CC-BY-NC-ND-4.0 | https://huggingface.co/datasets/UBC-NLP/Casablanca |
| MGB-2 / QASR | Al Jazeera broadcast, conversations/interviews | Turn-rich audio | research / non-commercial | https://huggingface.co/datasets/Arabic-STT/mgb2 , https://aclanthology.org/2021.acl-long.177.pdf |

Text-only continuation (complete vs incomplete) data: the Arabic LiveKit (57k) and Waqf (21k)
sets are **not published** (#31, #32). No public Arabic text EOU dataset with a permissive licence
was found; a synthetic one (truncate complete sentences at word boundaries) is the fallback.

## 6. Three new experiments suggested by the evidence

1. **T-transplant (vocab-aligned init).** Arm T3: Arabic SP-1024 *taken from*
   `stt_ar_fastconformer_hybrid_large_pc_v1.0`; initialise decoder embedding + joint output rows
   0..1023 (and, if dims match, the prediction LSTM) from that model; copy EOU-120M rows for
   `<EOU>`/`<EOB>`; keep EOU-120M encoder + `joint.enc`. Gate: dev WER and EOU p50/p90 at a fixed
   step budget vs T (random/mean init). Hypothesis: faster convergence, same asymptote (#22 logic).
   Cheap pre-check: diff the two `model_config.yaml` (decoder/joint sizes) before any GPU time.
2. **Separate EOU joint head (Google 2-stage, #23) as a guard arm.** After the winning R/T ASR
   fine-tune, freeze everything, clone the joint into an "EOU joint", fine-tune only it on
   EOU-labelled Arabic; decode text with the original joint. Measures whether the WER cost of
   `<EOU>` in Arabic (and early cutoffs) is recoverable without touching ASR weights. One variable:
   shared vs separate EOU joint.
3. **Arabic early-cutoff probe on smart-turn audio.** Use the CC-BY-4.0 smart-turn Arabic rows
   (`endpoint_bool`, `midfiller`) as an EOU-specific test: rate of `<EOU>` on cut-off/mid-filler
   clips (false EOU) vs on complete ones (+ latency with 3 s silence padding as in #1/#5). Run it on
   stock EOU-120M (expected near-useless), R, T and the Nemotron 3.5 punctuation-as-boundary
   baseline. This gives an EOU metric that is not just WER and is directly comparable with the
   Smart Turn audio classifier (#29) as a D-class reference.

---

Status: **CLOSED** (survey complete; open verification items listed as UNVERIFIED: EOU-120M special
id map and decoder/joint dims vs Arabic FastConformer Large, Parakeet-1.1B vocab size, SADA licence).
