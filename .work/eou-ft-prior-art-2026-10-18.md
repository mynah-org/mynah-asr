# EOU fine-tuning prior art (2026-10-18)

Scope: prior-art sweep (web plus local reading of code) for fine-tuning `nvidia/parakeet_realtime_eou_120m-v1`
(EOU120M) to Italian, with the focus on today's finding: stage 2 emits premature `<EOU>` at internal pauses.
No training was run and no remote box was touched.

Labels: **OBSERVED** = read in the cited source (code, card, paper, thread). **INFERENCE** = my reasoning, not
stated by any source. **NOT FOUND** = searched, no public evidence.

## 0. Versions read

| source | version / date |
|---|---|
| our repo (`lw-finetune-tooling`) | `626bcba` (2026-10-18 16:25 +0200) |
| NVIDIA-NeMo/Speech `main` | `a3afbd901807bfa5b46ae386573af31dced77834` (2026-10-16 20:34 UTC). Last commit touching `examples/asr/asr_eou`: `1f73a910` (2026-09-03, license headers). The scripts were added in `71fe0c68` (PR #14740, 2026-03-31). |
| mudler/parakeet.cpp | `5c9b0fbbd8e611d88e47d7e1ebae9bca9b447e8f` (2026-10-07) |
| FluidInference/FluidAudio | `2d6c42ca5f3accc0faf9fee9e0fff5b9b2cf6042` (2026-10-16) |
| NVIDIA/NeMo-Speech.cpp | `caa3f47c7c75fa2e9e6340272ebad57c1647dd44` (2026-10-16) |
| HF model card EOU120M | page read 2026-10-18 (the card gives no date; HF createdAt 2025-10-10) |

NVIDIA files read (all under https://github.com/NVIDIA-NeMo/Speech, at the commit above):
`examples/asr/asr_eou/{README.md, speech_to_text_rnnt_eou_train.py, speech_to_text_eou_eval.py}`,
`examples/asr/conf/asr_eou/fastconformer_transducer_bpe_streaming_large.yaml` (17 x 512 = the 120M geometry),
`nemo/collections/asr/data/audio_to_eou_label_lhotse.py`, `nemo/collections/asr/models/asr_eou_models.py`,
`nemo/collections/asr/parts/utils/eou_utils.py`, `scripts/asr_eou/{conf/data.yaml, generate_noisy_eval_data.py,
eval_eou_metrics.py, clean_manifest.py, add_eob_labels.py}`, `tools/nemo_forced_aligner/align_eou.py`,
`nemo/agents/voice_agent/pipecat/services/nemo/{streaming_asr.py, stt.py, turn_taking.py}`,
`examples/voice_agent/server/server_configs/{default.yaml, stt_configs/nemo_cache_aware_streaming.yaml}`,
`examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`.

Important caveat (OBSERVED): the tutorial fine-tunes **nemotron-speech-streaming-en-0.6b into an EOU model**. It does
not document how EOU120M itself was trained, and it does not cover fine-tuning EOU120M to a new language. The card
gives only the data list (section 2).

## 1. NVIDIA's EOU tutorial vs our recipe

### 1.1 How NVIDIA builds EOU examples (OBSERVED, `audio_to_eou_label_lhotse.py`)

- **Target text** = `text + "<EOU>"` per utterance. There is no space before it unless `add_sep_before_eou`
  (default False). `add_eou_to_text` defaults to True. Text is assembled in `_get_text_tokens` (l. 372-412).
- **Multi-utterance support exists in code**: if a cut has `sou_time`/`eou_time`/`utterances` lists, the text
  becomes `A<EOU> B<EOB> ...` and the frame labels mark each end (l. 334-370, docstring example l. 118-126).
  **The README forbids it in practice**: "we need to make sure the dataset used for finetuning meets the criteria
  that each sample should contain a single utterance, otherwise the model's EOU prediction accuracy will be
  degraded" (README section 2). `align_eou.py` writes one `sou_time` (first word start) and one `eou_time`
  (last word end) per clip (l. 559-565). So **the documented recipe never shows text after `<EOU>`**, the same
  as ours.
- **"Plain ASR" data still gets `<EOU>`**: a cut without `sou_time`/`eou_time` falls back to "single speech
  segment". It gets `<EOU>` appended to its text and an EOU frame label at its last frame (l. 322-332, 380-382).
  The tutorial blend gives this "ASR" group (LibriSpeech train-other-500) weight 0.1 and the "EOU" group (LibriTTS
  train-clean-360 with alignments) weight 0.9 (README 2.5). There is no task-dependent switch in the dataset class.
  So in NVIDIA's recipe **every** training target ends in `<EOU>`.
- **EOB**: `<EOB>` replaces `<EOU>` when `is_backchannel` is set and `ignore_eob_label` is False. All three shipped
  configs and the README set `ignore_eob_label: true` ("treat them the same as EOU"). The README also proposes the
  alternative of treating EOB as EOU and matching against a phrase list. `add_eob_labels.py` uses a fixed English
  phrase list.
- **The frame labels (0 non-speech / 1 speech / 2 EOU / 3 EOB) are used only for metrics**: `training_step` uses
  the text tokens and the RNNT loss only (`asr_eou_models.py` l. 430-526). No frame-level EOU loss.
- **Padding** (`_random_pad_audio`, l. 414-499): with probability `prob` it draws a TOTAL pad duration
  U(min_pre+min_post, max_total - dur) (uniform), then post = U(min_post, total - min_pre), pre = the rest. Each
  side is capped at `max_pad_duration`. The pad is **digital zeros**. Augmentation is applied **after** padding, so
  white noise and real noise cover both the pads and the speech (l. 233-237).
- **Augmentations not allowed with EOU**: `random_segment`, `speed`, `time_stretch` are skipped with a warning
  (l. 42-43, 184-187), because they would move the SOU/EOU timestamps.
- **Tokenizer**: `add_special_tokens_to_sentencepiece.py` appends `<EOU>`, `<EOB>` as the last two pieces
  (1024 -> 1026). The dataset checks that they are the last two ids (l. 194-207). The new joint rows are initialised by
  `token_init_method` (default `constant`, bias -1000, weight unchanged) (train script `init_from_pretrained_nemo`).
- **Text cleaning**: README recommends dropping punctuation "for lowest possible EOU latency". `clean_manifest.py`
  uses an English normaliser (whisper_normalizer, num2words).

### 1.2 Side-by-side table

| item | NVIDIA tutorial (OBSERVED unless noted) | ours (OBSERVED in `two_stage.py`, `s2_mix.sh`, `p1_additive.sh`, `mix_arm.sh`, roadmap) |
|---|---|---|
| init | nemotron-speech-streaming-en-0.6b + 2 new tokens. For a 120M: `..._large.yaml` (17 x 512, att [70,1]) | stock EOU120M (already has `<EOU>`/`<EOB>`) -> plain-IT stage 1 -> stage 2 |
| stages | **one** stage: ASR and EOU learnt jointly from the start | two stages: stage 1 plain IT (no EOU in targets), stage 2 adds `<EOU>` |
| EOU example | `text<EOU>`, one utterance per clip, single SOU/EOU from forced alignment | `text + [<EOU>]` (`--eou-append 1`), one source clip per sample |
| `<EOU>` position | end of every target (EOU group and the "ASR" group) | end of every target |
| `<EOB>` | supported, but `ignore_eob_label: true` in every shipped config -> never trained | never in targets (stock rows kept) |
| text after `<EOU>` | possible in code (multi-utterance lists); **not used** in the documented recipe | never (`--concat-prob` exists, A + gap + B with only ONE final EOU, but no documented stage-2 run uses it) |
| trailing silence | `random_padding.prob` 0.9 (README) / **0.99** (large/xlarge yaml); **min post-pad 3.0 s**, max 6.0 s per side, total <= 40 s; digital zeros, then noise on top | `--pad-prob 0.5`, U(1, 3) s, Gaussian noise floor -90..-60 dB; the other **50 % get no added trailing quiet** |
| leading silence | pre-pad = total - post (min 0, max 6 s) | off in v4 (`--pre-pad-prob` tested on E3: worse, not adopted) |
| internal pauses | nothing explicit: no silence insertion, no pause labels, no concatenation | nothing explicit (`--concat-prob` unused) |
| source segmentation | EOU group = LibriTTS ("speech is split at sentence breaks", https://www.openslr.org/60/), plus 10 % LibriSpeech | stage-2 v4 mix: MLS 0.196, CV 0.168, VP-t50 0.112, FLEURS 0.084, it-yg 0.189, it-yg6 0.250 (Granary pseudo-labels). MLS segments are cut at the longest silence in a 10-20 s window (sec. 1.4) |
| tokenizer | base SPE + `<EOU>`,`<EOB>` appended | stock English SPE (Italian de-accented), or the warm-started Italian SPE (`--tokenizer`) |
| trainable parts | whole model by default (`freeze_encoder` optional, default False); adapters optional | stage 1: whole model; stage 2: top-4 encoder blocks only |
| LRs | one LR group: AdamW, NoamAnnealing `lr 5.0`, d_model 512, warmup 10k, min 1e-6 (peak ~2.2e-3 at step 10k, INFERENCE from the Noam formula), wd 1e-3, clip 1.0, fp32 | stage 1: enc 4e-4 / dec+joint 3e-4, WSD. Stage 2: enc top-4 3e-5, dec+joint 2e-5, cosine |
| loss | RNNT + **FastEmit 3e-2** ("a good start", README; yaml comment says 1e-4..1e-2 is typical). No aux loss in the recipe (InterCTC hooks empty) | RNNT, FastEmit 0 |
| EOU vs plain ratio | EOU group 0.9 / ASR group 0.1, but both end in `<EOU>` (1.1) | 100 % of stage-2 targets end in `<EOU>`; stage 1 0 % |
| max duration | `max_duration: 30` (train), padded up to `max_total_duration: 40` | `--max-dur 20` |
| augmentation | white noise p 0.9 (-90..-46 dB); gain p 0.2 (+-10 dB); **real noise p 0.9 at SNR 0-20 dB**; SpecAugment 2 x F27, 10 x T0.05; speed/stretch disabled | stage 2: none besides the quiet pad (`--aug-*` off); SpecAugment = the model's own config |
| steps / epochs | example: `max_steps` 1,000,000, `limit_train_batches` 1000, val every 1000 (yaml max_steps 100k); stop rule not given | stage 2: 1500 steps, evals every 250 |
| batch | 16 (single GPU in the example) | 16 |
| checkpoint selection | `exp_manager` monitor `val_wer`, min, top-3; **no EOU metric in selection** | min macro WER among evals with EOU rate >= 80 % and <= 10 % empty |
| EOU metrics | frame-level SegLST matching (`eou_utils.py`): TP if \|pred_end - ref_end\| <= collar (0 by default); **early cutoff** = predicted EOU inside the reference utterance before its end (FP, distance logged); late EOU = FN; missing; latency percentiles p50/p90/p95; `early_cutoff_avg_num`, `miss_rate` | EOU rate (clips with any EOU), empties; Mynah m1: speech-end -> EOU p50/p95, missed @2 s, premature %, A+1s+B EOU-in-gap |
| eval data | `generate_noisy_eval_data.py`: constant pad 0.2 s pre / 3.0 s post, real noise p 1.0 at SNR 0-20 dB | 2 s trailing quiet at -75 dB in in-run evals; Mynah bank FLEURS-it 200 |
| eval mode | offline: full encoder forward (chunked-limited attention), `greedy_batch`, single-utterance clips, no reset | offline greedy in-run; streaming in Mynah; since today: offline with decoder reset after `<EOU>` (`s2_reset.py`) |
| state after `<EOU>` | training: never sees text after EOU. Runtime (voice agent): **full reset** (sec. 3B) | Mynah: decoder-only reset, encoder cache kept |

### 1.3 Published numbers for EOU120M (OBSERVED, https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)

- Data: "~10,000 hours from human-transcribed NeMo ASR Set 3.0" plus AMI, DialogStudio (subset), Granary, Google
  Speech Commands, LibriTTS. Human + synthetic. English only, no PnC.
- EOU latency on TTS DialogStudio audio with 3 s of silence appended: p50 160 ms, p90 280 ms, p95 320 ms.
- **No early-cutoff / false-EOU / miss-rate number is published** (NOT FOUND), although the eval code computes them.
- WER (HF OpenASR set, 160 ms streaming): avg 9.30 %, 3.61 (LS clean) .. 15.76 (Earnings22).
- Training GPUs / steps / init / padding actually used for this checkpoint: NOT FOUND.

### 1.4 Differences that could explain false EOUs at internal pauses

1. **Fraction of EOU targets with long trailing silence** (OBSERVED difference; causal role INFERENCE).
   NVIDIA pads 90-99 % of samples with at least 3 s after the speech. We pad 50 % with 1-3 s, and the other 50 %
   carry `<EOU>` at whatever natural trailing silence the clip has (often a few hundred ms or less). RNNT
   puts no constraint on WHERE the EOU goes. But a target whose audio ends 0.1-0.3 s after the last word can only place
   `<EOU>` inside that short tail. Half of our stage-2 data therefore rewards "EOU after a ~0.3 s pause". Our
   premature EOUs sit at internal pauses with median 0.32 s. This is the strongest recipe-level candidate.
   It is cheap to measure (sec. 5).
2. **Source segmentation** (OBSERVED for MLS and LibriTTS; INFERENCE for the effect). MLS: "we select the longest
   silence duration within 10 sec to 20 sec range from the start of an audio and split the audio at the mid point
   of the silence chunk ... If no silence frames are found ... we split the audio at 20 sec mark" (Pratap et al.
   2020, sec. 3.2, https://arxiv.org/abs/2012.03411). So MLS clip ends are pause positions, not sentence or turn
   ends, and some are hard cuts mid-speech. NVIDIA's EOU group is LibriTTS, "split at sentence breaks"
   (https://www.openslr.org/60/), and the README requires single-utterance samples. Our stage-2 v4 mix is ~20 %
   MLS and ~44 % Granary pseudo-label clips (segmentation of those: NOT VERIFIED here). A large share of our
   `<EOU>` labels therefore marks "a pause where a segmenter cut", which teaches pause => EOU. FLEURS (complete
   read sentences with slow, long internal pauses) shows the highest premature rate (17 %). That fits a model
   that learnt a pause-length rule rather than completeness.
3. **No noise in stage 2** (OBSERVED difference). NVIDIA adds real noise at SNR 0-20 dB to 90 % of samples
   *after* padding, so "quiet" is rarely clean. Our pads are near-digital (-90..-60 dB) and no noise is added. A
   model trained this way may read any low-energy stretch (an internal pause in clean read speech) as end-of-turn
   evidence. INFERENCE.
4. **Text after `<EOU>`: no difference.** NVIDIA's documented recipe also never puts text after `<EOU>`, and the
   README warns against multi-utterance samples. The absorbing blank state after `<EOU>` is therefore expected of
   **any** model trained this way. It is not a defect of our stage 2. Supporting evidence: the stock English EOU120M
   "stops transcribing after the first [EOU]" in parakeet.cpp until a reset on EOU was added (issue #13, sec. 3B).
   NVIDIA's runtime resets on every EOU, so it never exercises that state.
5. **EOB**: no difference in practice (NVIDIA ships `ignore_eob_label: true`).
6. **FastEmit 3e-2 vs 0**: a difference, but its sign for false EOUs is not documented. FastEmit pushes emissions
   earlier, which helps latency and could raise early cutoffs. NOT FOUND: any NVIDIA statement on its effect on
   early cutoff.
7. **One-stage joint training vs our two-stage**: NVIDIA learns text and EOU together from the start. We learn EOU in
   1500 steps on top of a plain model with only the top-4 encoder blocks open. Whether this changes pause
   handling: NOT FOUND (no public ablation).
8. **Selection rule**: NVIDIA selects on val WER only. Ours also requires EOU rate >= 80 %, which rewards firing an
   EOU at all and never penalises early firing. INFERENCE: our rule is blind to EOU precision. NVIDIA's code does
   compute `early_cutoff`, but on single-utterance padded eval clips.

## 2. Web sweep 2025-2026 (fine-tuning EOU120M, non-English, internal pauses, reset)

### 2.1 Fine-tuning EOU120M itself
- **NOT FOUND**: any public fine-tune of EOU120M (any language). The HF "finetune" tree of the base
  (https://huggingface.co/models?other=base_model:finetune:nvidia/parakeet_realtime_eou_120m-v1) lists 7 entries.
  All are CoreML / INT8 conversions (FluidInference, aufklarer, playstonex, aoiandroid, kamilobad, busyfarm), not
  weight fine-tunes. The HF search for `parakeet_realtime`, `realtime-eou`, `parakeet-eou` (2026-10-18) returned
  only conversions (ONNX/CoreML/fp16/INT8/sherpa). One entry `odunola/parakeet-EOU` (2025-12-31) has an empty card
  (license only), so nothing can be said about it.
- HF Discussions on the model (https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/discussions):
  #9 "Multilingual support?" (2026-01-15, plus "please we need spanish support" 2026-05-03), **no NVIDIA reply**.
  #7 inconsistent EOU with 1 s chunks: NVIDIA (kunaldhawan, 2025-12-09) says the model "processes audio in 80msec
  chunks by design" and points to the cache-aware streaming script. #6 "Poor performance" (2025-11-28): high
  WER and "very high false positives on the EOU", self-resolved: the cause was independent 1 s chunks discarding
  context.
- GitHub issue search (`parakeet_realtime_eou`, `asr_eou`, `EncDecRNNTBPEEOUModel`, "EOU token", 2026-10-18):
  only bug reports on NVIDIA's EOU code (NeMo #16085: `eou_pred` filter never applied, fixed in current code;
  #16150: hybrid EOU model `validation_pass` UnboundLocalError), integration requests (sherpa-onnx #2805,
  TypeWhisper #573/#155) and runtime issues (below). **No fine-tuning report, no hours/GPU/WER/EOU-F1 for an
  EOU120M fine-tune.** NOT FOUND.
- Soniqo docs (https://soniqo.audio/guides/dictate) claim Parakeet-EOU-120M covers "25 European languages — same
  vocabulary family as upstream NeMo Parakeet TDT". This **contradicts the primary source** ("supports only
  English", NVIDIA card). Treat it as an unsupported secondary claim.

### 2.2 Internal pauses / false endpoints (community evidence)
- joeblau/voice PR #111 (merged 2026-10-07, https://github.com/joeblau/voice/pull/111), stock EOU120M via FluidAudio
  at 320 ms chunks, on macOS-TTS fixtures. OBSERVED: "Parakeet's EOU token arrives 0.8–1.4 s after the speech ends,
  and for 7 of the 12 fixture utterances it doesn't arrive before the next one". They commit on the first of:
  model EOU (two-chunk debounce "against stray EOU tokens"), VAD end of speech + 0.9 s, 30 s cap, stream end.
  They **reset after every commit**. Their result: finals 0.89-1.06 s after end of speech.
- FluidAudio issue #827 (2026-07-31): a user wanted ~4 s pauses ("2 s felt like it cut people off mid-thought"),
  found the debounce semantics unreliable, and ended turns on gaps in decoded words instead, "we no longer use the
  EOU head at all".
- arXiv 2609.04225 (Li & Shi, 2026-07-12, "The Trade-off Was in the Labels"): "within-turn pauses routinely exceed
  between-turn gaps". They state that open recognizers with turn signals "either rarely fire (Parakeet-EOU) or fire
  spuriously (Kyutai)" (no Parakeet numbers). Their recipe (LoRA on Qwen3-ASR-0.6B, English, ~20k synthesized
  examples from LibriSpeech + AMI, "hours on one GPU"):
  - an end marker iff the words so far are semantically complete AND >= 0.3 s of silence has been heard;
  - "pause pairs": two utterances separated by a pause, with the marker at the pause only when the words before it are
    complete;
  - minimal pairs of the same utterance with a silence tail (fire) and without one (hold);
  - truncated-mid-utterance samples.
  Their principle: labels must be computable from the input up to the decision point. Forced-alignment cuts of
  offline corpora are "clairvoyant". Result: 0.97 boundary recall at 0.39 s median latency, 0.3 false fires per
  speech-minute. This is the only public recipe found that targets internal pauses directly (different model
  family).
- NVIDIA/NeMo-Speech.cpp issue #40 / PR #41 (2026-09-10, open): mid-stream endpointing fires on ordinary pauses. This
  is **token-silence endpointing** (no EOU token) on nemotron-3.5-asr-streaming. The fix keeps encoder and predictor
  state across the mid-stream EOU (a "soft reporting checkpoint"). Relevant to runtime semantics only.

### 2.3 Prediction-network reset after EOU
See section 3B.

## 3. Our false-EOU problem, split by layer

### A. Checkpoint behaviour
- NVIDIA (OBSERVED): one-stage joint training, 90-99 % of samples with >= 3 s trailing pad, real noise at SNR 0-20
  dB, single-utterance sentence-level EOU data (LibriTTS), FastEmit 3e-2, no internal-pause supervision, no text
  after EOU. Published EOU metric: latency only. Early cutoff is computed by their tooling but not published.
- Community: nobody else has published a re-trained EOU120M checkpoint. The only pause-aware labelling scheme found
  is arXiv 2609.04225 (different model).
- Ours: premature EOU at 0.08-2 s internal pauses in 8.4 % of clips (17 % FLEURS). Candidate recipe factors are in
  1.4 (1-3): pad fraction, source segmentation, noise. All INFERENCE until measured.

### B. Decoder / runtime behaviour
- **NVIDIA voice agent (OBSERVED, `streaming_asr.py` l. 287-318)**: when `<EOU>` or `<EOB>` appears in a chunk's
  text, `reset_state()` resets **the audio/feature buffer, the encoder cache (`get_initial_cache_state`) and the
  decoder hypothesis** (blank hypothesis, `dec_state=None`). So NVIDIA does a **full reset**, not decoder-only. In
  `stt.py` l. 302-312, a VAD stop also forces `reset_state()` and logs `[EOU missing]` if the model had not ended the
  utterance. Docs (https://docs.nvidia.com/nemo/labs-voice-agent/about/core-concepts/speech-pipeline/speech-recognition,
  Nightly): "resets the encoder cache". `ignore_eou_eob: true` strips the tokens and disables EOU turn taking.
- **NeMo generic cache-aware streaming script** (`speech_to_text_cache_aware_streaming_infer.py`, OBSERVED): no
  EOU handling and no reset on EOU. Running it on multi-utterance audio would hit the absorbing state (INFERENCE,
  consistent with parakeet.cpp #13).
- **parakeet.cpp (OBSERVED)**: issue #13 (2026-06-06), "it stops transcribing after the first [EOU]" with the stock
  model. The fix (`src/streaming.cpp` l. 162-179) resets **only the decoder** (LSTM h/c = 0, last_token = SOS) and
  keeps the encoder cache. The comment says NeMo's full reset "was verified to be a no-op for the decoded tokens
  (decoder-only reset == NeMo's full reset_state byte-for-byte on multi-utterance clips)". This is tested by
  `tests/test_streaming_eou_reset.cpp` against `scripts/gen_stream_reset_baseline.py` (speech + silence + speech).
  That equivalence was shown on clips with a real silence gap. Whether it holds when the EOU is premature inside
  continuous speech: NOT FOUND (INFERENCE: probably not, since a fresh encoder cache loses left context).
- **FluidAudio (OBSERVED, `RnntDecoder.swift` l. 105-125)**: on `<EOU>` it sets a flag and `break outerLoop`. The EOU
  is **never committed to the decoder state** (`lastToken`, h, c stay at the last text token), and the remaining
  frames of that chunk are not decoded. The next chunk decodes from the pre-EOU state, so text after a false EOU
  can continue without any reset. EOU is "confirmed" only after `eouDebounceMs` (default 1280 ms) with no new
  words (`StreamingEouAsrManager.swift` l. 209-325, after issue #827). Encoder caches are reset only on an explicit
  `reset()`.
- **Research** (arXiv 2608.22101, Chattopadhyay, Erdmann, Ostendorf, 2026-08-22): the full "fresh start" reset at
  each turn hurts turn onsets. State carry-over / rollback gives 15-21 % relative WER reduction at onsets.
  Switchboard first-word WER for fastconformer-114m: 51.44 -> 34.78. The paper evaluates fastconformer-114m and
  nemotron-streaming-0.6b, **not** EOU120M, and has no EOU false-positive numbers.
- Ours (Mynah): decoder-only reset, the same as parakeet.cpp. Offline reset test: recovers ~74 % of the +1.85 WER.
  FluidAudio's "never feed EOU into the predictor" is a third option that we have not tested.

### C. Turn-taking layer
- **NVIDIA voice agent (OBSERVED, `turn_taking.py`, `default.yaml`)**:
  - While VAD says the user is speaking, a buffer ending in `<EOU>` ends the turn **immediately**: no debounce, no
    probability threshold. `eou_prob` is computed and only logged (`stt.py` l. 208-212).
  - `<EOB>` while the bot speaks is treated as a backchannel. When the bot is not speaking, `<EOB>` is stripped and
    the turn continues.
  - EOU text matching the backchannel list (78 phrases) is suppressed only while the bot speaks.
  - VAD (Silero, confidence 0.6, start 0.1 s, **stop_secs 1.2**) is the fallback: on VAD stop the buffer is pushed
    as final. Docs: "the end of turn usually arrives from <EOU> first, and VAD stop acts as the fallback"
    (https://docs.nvidia.com/nemo/labs-voice-agent/about/core-concepts/speech-pipeline/turn-taking-backchannels).
  - So NVIDIA itself has **no defence against a premature EOU** at the turn layer.
- FluidAudio: debounce `eouDebounceMs` 1280 ms of decoded-audio silence after the first EOU signal. New words cancel
  it.
- joeblau/voice: two-chunk debounce, VAD end + 0.9 s fallback, 30 s cap, reset per commit.
- NeMo-Speech.cpp: no EOU token. Endpointing uses token-silence or VAD with `stop_history_eou_ms` 800 ms default
  (docs/asr/configuration.md).
- arXiv 2609.04225: argues that silence timeouts cannot separate within-turn pauses from turn ends, because
  completeness must come from the recognizer.

## 4. Economics: language adaptation of ~100-200M (and nearby) ASR models

| model / work | size, mode | target data | compute | result | source |
|---|---|---|---|---|---|
| NVIDIA stt_it_fastconformer_hybrid_large_pc | ~115M, hybrid RNNT/CTC, **offline** | 487 h (CV12 220, MLS 214, VP 53) | "several hundred epochs"; GPUs NOT FOUND | WER CV 5.64 / MLS 11.39 / VP 16.22 (no PnC) | https://huggingface.co/nvidia/stt_it_fastconformer_hybrid_large_pc |
| NVIDIA stt_fr_fastconformer_hybrid_large_pc | ~115M, hybrid, offline | 1,800 h (CV 710, MLS 925, VP 165) | NOT FOUND | CV 7.92 / MLS 5.21 / VP 6.49 | https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc |
| Banfic 2026, arXiv 2606.24169 | 0.6B cache-aware FastConformer RNNT, **streaming** (+ offline) | 100-2500 h/lang, 8 langs incl. FR (no IT) | H100s; GPU-hours NOT reported. AdamW 1e-4 cosine, bs 48, FastEmit 0.005, 25/30 epochs | FR FLEURS 160 ms: EN-init 25.38 (100 h) -> 16.81 (500 h) -> 13.81 (2500 h); ML-init 19.51 / 15.55 / 13.75. No EOU | https://arxiv.org/html/2606.24169v1 |
| Frisian on nemotron-3.5-asr-streaming-0.6b | 0.6B streaming | 40.4 h CV22 | 1 x A100, 8000 steps, peak LR ~2e-4 (Noam) | 82.87 -> 20.36 WER @80 ms | https://huggingface.co/blog/jellewas/frisian-asr (2026-06-11) |
| NVIDIA Saudi dialects on Nemotron 3.5 ASR | 0.6B streaming | 133.7 h + 10 % FLEURS replay | 2 x RTX PRO 6000 Blackwell, 12k steps, ~4.5 h, LR 2e-5 | 55.05 -> 29.96 WER | https://developer.nvidia.com/blog/fine-tuning-nvidia-nemotron-for-saudi-arabic-dialects-with-a-path-to-other-languages/ (2026-09-30) |
| NVIDIA Nemotron 3.5 fine-tune blog (Greek, Bulgarian) | 0.6B streaming | ~2000 h, then ~2300 h | "single GPU for a quick pass"; no wall-clock | EL 35 -> 24, BG 22 -> 15 | https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr |
| ours (stage 1) | 120M cache-aware RNNT, streaming [70,1] | ~357 h IT mix | 1 GPU; e.g. W4 8k steps = 62 min on L40S (401.6 processed audio h), 16k = 127 min on RTX 6000 Ada (roadmap) | best plain C250-36k harness greedy 22.27 | roadmap 2026-10-15/18 |

- IT/FR streaming small model **with EOU**: NOT FOUND. No comparable public EOU120M (or other ~100-200M streaming
  ASR with an in-model end-of-utterance token) specialist for Italian or French was found in this sweep. This is
  an absence of found prior art, not demonstrated novelty. The search was limited to HF, GitHub issues/code search,
  NVIDIA docs/blogs and general web search on 2026-10-18.
- Nearby but different: nemotron-3.5-asr-streaming-0.6b covers it-IT and fr-FR (FLEURS IT 5.23 @80 ms -> 4.25 @1.12 s,
  FR 10.97 -> 9.03), 0.6B, no EOU/EOB tokens (https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b,
  card dated 2026-06-04). LiveKit's text turn detector covers Italian (text model, not ASR). Many small text EOU
  classifiers exist on HF (Arabic, Hindi, Spanish), all text-only.
- Cost of an EOU120M fine-tune (GPU-hours, $): NOT FOUND anywhere public.

## 5. Actionable reading (cheap first)

1. **Measure the trailing-silence distribution of stage-2 targets** per source (MLS, CV, VP, FLEURS, yg, yg6) on
   the unpadded clips, with a VAD or the aligner: the fraction with < 0.5 s after the last word. If large, the 50 %
   unpadded half teaches EOU after short pauses (1.4.1). CPU only.
2. **Stage-2 v5 data arm (one variable)**: NVIDIA-like padding (p 0.9-0.99, post-pad >= 3 s) with the rest
   unchanged. A second, separate arm: EOU only on sentence-complete sources (CV/FLEURS-like) or on clips whose end
   is a sentence end, with MLS/yg kept in the mix *without* a trailing `<EOU>`. Note that NVIDIA's code appends
   EOU to every sample, so that would be a deviation from NVIDIA. A third option from arXiv 2609.04225 / our
   unused `--concat-prob`: "pause pairs" A + internal pause + B with EOU only at the end (negatives at internal
   pauses). Pre-register gates on **false-endpoint rate at internal pauses** plus true-endpoint latency/recall, not
   only WER and EOU rate.
3. **Runtime**: compare three semantics on the same frozen checkpoint and multi-pause clips (CPU):
   - (a) decoder reset (Mynah, parakeet.cpp);
   - (b) full reset (NVIDIA voice agent);
   - (c) FluidAudio-style, where `<EOU>` is never fed to the predictor and decoding continues from the pre-EOU state.
   Pair each with a turn-layer debounce (e.g. EOU confirmed only after N ms with no new words). (c) plus a debounce
   should turn a premature EOU into a non-event for ASR continuity, at a latency cost equal to the debounce.
4. Add early-cutoff / FP-per-minute metrics (NVIDIA's `evaluate_eou` definitions, or arXiv 2609.04225's false
   fires per speech-minute) to the selection rule, which today rewards EOU rate only.

## 6. Unknowns

- How EOU120M itself was trained (init, stages, padding actually used, FastEmit, steps, GPUs): NOT FOUND.
- NVIDIA's own early-cutoff / false-EOU rate for EOU120M: NOT FOUND (only latency published).
- Whether NVIDIA's full reset and parakeet.cpp's decoder-only reset are equivalent after a **premature** EOU inside
  continuous speech: NOT FOUND (parakeet.cpp's equivalence test uses real silence gaps).
- Effect of FastEmit, one-stage vs two-stage, and pad fraction on early cutoff: no public ablation (NOT FOUND).
- Segmentation of Granary/YODAS and VoxPopuli clip ends (sentence vs pause vs subtitle): NOT VERIFIED here.
- Any public non-English EOU120M fine-tune, its data, cost or EOU F1: NOT FOUND.
