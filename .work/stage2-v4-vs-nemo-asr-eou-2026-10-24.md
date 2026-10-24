# Stage 2 v4 vs NVIDIA NeMo `asr_eou`: difference table (2026-10-24)

**Status: CLOSED.** This is a comparison only. It ran on CPU and the web: no GPU, no remote box, no training, no
code change. It proposes nothing for today's experiments. The "what to test later" column records ideas and is
not a plan.

## 0. Sources and versions

- **Ours.** Repo `lw-finetune-tooling` @ `5829edf` (2026-10-24 13:22 +0200). Files read:
  `finetune/eou/two_stage.py`, `finetune/jobs/s2_mix.sh`, `finetune/eou/diag/s2_reset.py`,
  `finetune/eou/diag/eou_v2_score.py`, `finetune/eou/prepare_eou_v2.py`, `finetune/eou/tokenizer_eou.py` and
  `finetune/eou/README.md`. Notes read: `.work/release-it-fr-2026-10-19.md` (the v4 definition, the RC-IT v4 choice
  and the FR-S2 command), `.work/release-final-2026-10-22.md` (FR-E0, sec. 12), `.work/fr-v2-eou-autopsy-2026-10-21.md`,
  `.work/eou-ft-prior-art-2026-10-18.md`, `.work/eou-multilingual-prior-art-2026-10-24.md` and
  `.work/restart-2026-10-24.md`. The `-wt-trainB` worktree (feat/trainer-throughput) carries the same `two_stage.py`
  plus throughput flags (bucket pool, subbatching). These flags are off in v4. v4 itself was produced on
  `lw-finetune-tooling`.
- **NeMo.** `github.com/NVIDIA-NeMo/NeMo` `main` = `b19b1a79d86ec30c1de078682b4b3ed0f59098bb`, fetched 2026-10-24.
  Short form below: `NeMo@b19b1a7:<path>:<lines>`, i.e. `https://github.com/NVIDIA-NeMo/NeMo/blob/b19b1a79d86ec30c1de078682b4b3ed0f59098bb/<path>`.
  Files read today:
  - `examples/asr/asr_eou/README.md`
  - `examples/asr/asr_eou/speech_to_text_rnnt_eou_train.py`
  - `examples/asr/asr_eou/speech_to_text_eou_eval.py`
  - `nemo/collections/asr/data/audio_to_eou_label_lhotse.py`
  - `nemo/collections/asr/models/asr_eou_models.py`
  - `nemo/collections/asr/parts/utils/eou_utils.py`
  - `examples/asr/conf/asr_eou/fastconformer_transducer_bpe_streaming_large.yaml`
  - `scripts/asr_eou/tokenizers/add_special_tokens_to_sentencepiece.py`
  - `scripts/asr_eou/conf/data.yaml`
- **Model card.** `https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1`, read 2026-10-24.
- **Not re-read today.** The voice-agent runtime (`nemo/agents/voice_agent/...`) and the NeMo fine-tuning docs page.
  Facts about them come from `.work/eou-ft-prior-art-2026-10-18.md` (NVIDIA-NeMo/Speech @ a3afbd9) and are marked
  **(2026-10-18 read)**.
- **Global caveat (OBSERVED, README l.1-7).** The NeMo tutorial fine-tunes *nemotron-speech-streaming-en-0.6b into*
  an EOU model. How EOU-120M itself was trained is not published anywhere: **UNVERIFIED** (the card does not say).

## 1. What Stage 2 v4 is (reconstructed from code + notes)

- **Parent.** A plain (no-`<EOU>`) stage-1 checkpoint fine-tuned from stock EOU-120M with the **stock** SPE
  (1024 + `<EOU>`=1024, `<EOB>`=1025, blank 1026; de-accented `[a-z' ]` targets). Parents are C250-36k for IT and
  FR-G1E / FR-H2 for FR.
- **Command.** Read back from the archived v4 `metrics.json`, release-it-fr-2026-10-19.md l.1063-1067:

  `two_stage.py --init <parent> --freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5 --lr 2e-5 --bs 16 --steps 1500
  --lr-shape cosine --eou-append 1 --pad-prob 0.5 --pad-min 1 --pad-max 3 --pad-noise-db -90,-60 --fastemit 0
  --max-dur 20 --eval-every 250 --val-n 200 --seed 0`

  For FR, `--lang fr` is added.
- **Data.** The parent's own stage-1 mix ("the v4 principle").
  - IT: MLS 0.196, CV 0.168, VP-t50 0.112, FLEURS 0.084, it-yg 0.189, it-yg6 0.250. The last two are Granary
    clips with Nemotron pseudo-labels.
  - FR: MLS-fr 0.2356, CV-fr 0.2020, VP-fr-t50 0.1347, FLEURS-fr 0.1010, Granary-fr agree<=20 0.3267.
- **Labels.** Target = `text_ids + [<EOU>]` on **100 %** of draws (`two_stage.py:233-237`; `--eou-min-tail 0`).
  - No timing information is used: there are no `sou_time` / `eou_time` fields and no aligner.
  - `<EOB>` is never in the targets.
  - No text ever follows `<EOU>` (`--concat-prob 0`, `--pause-stretch-prob 0`).
- **Silence.** With p = 0.5 per sample, U(1, 3) s of Gaussian noise at -90..-60 dB is appended after the clip
  (`two_stage.py:483-487`, `quiet()` l.387-388). The other half keeps its natural trailing silence. FR medians are
  0.00-1.60 s depending on the source (fr-v2 autopsy sec. 3). No leading pad (`--pre-pad-prob 0`).
- **Augmentation.** No noise, gain or speed (`--aug-*` 0). SpecAugment comes from the model's own config, because
  `m.training` stays True. The encoder runs in eval mode (dropout off) because it is partly frozen
  (`two_stage.py:566-575`).
- **Loss.** Plain RNNT (`warprnnt_numba`, `mean_batch`), FastEmit 0 (`two_stage.py:211-213`). No frame-level EOU
  loss and no auxiliary loss.
- **Optimiser and trainable parts.**
  - AdamW (0.9, 0.98), wd 1e-3, grad clip 1.0 (`two_stage.py:351, 746`).
  - Linear warmup over steps/20 = 75 steps, then cosine to 1 % (`lr_factor`, l.352-362).
  - Two parameter groups: decoder + joint at 2e-5, top-4 encoder blocks at 3e-5. The other 13 blocks, pre-encode
    and the preprocessor are frozen.
- **Selection (in-run).** An eval every 250 steps on the 4-set harness (200 clips per set, +2 s of -75 dB quiet).
  `final.nemo` = the lowest macro WER among evals with EOU rate >= 80 % and empty <= 10 % (`two_stage.py:790-799`).
  The selected steps were 250 for IT v4, 250 for FR-S2 and 1250 for FR-H2.
- **EOU metrics** (`s2_reset.py:195-212`). Offline frame-synchronous greedy, max 10 symbols per frame, decoder
  reset after `<EOU>`, +2.0 s of -75 dB quiet. The speech end is an energy endpoint: the last 80 ms frame within
  35 dB of the clip peak, measured on the original clip.
  - **FE** = % of clips with an `<EOU>` while >= 0.5 s of speech remains.
  - **FE_any** = an `<EOU>` followed by >= 0.16 s of speech.
  - **Recall** = first `<EOU>` at or after end - 0.5 s.
  - **Latency** = that first `<EOU>` minus the speech end, reported as p50 / p90.
  - DEV-EOU-FR-v2 adds two classes. *Internal* clips have a natural internal pause of 0.8 s or more, split into
    strata. *Endpoint* clips are clean single utterances (`prepare_eou_v2.py`, `eou_v2_score.py`).
- **Gates.**
  - **RC-FR candidate:** DEV delta <= +1.0, no source > +2.0, OOD1 delta <= +1.0, recall >= 94 %, FE <= 10 %,
    p50 <= 1.0 s.
  - **FAIL:** recall < 90 %, FE > 15 % or p50 > 1.5 s (release-it-fr l.1089-1095).
  - IT: v4 was chosen on DEV730 by DEV + product criteria and frozen before FINAL-IT.
- **Observed v4 numbers.**

  | model | WER | FE | recall | p50 / p90 |
  |---|---|---|---|---|
  | IT DEV | reset-WER 22.54 | 8.36 % | 94.38 % | 0.72 / 1.92 s |
  | IT FINAL-IT | 20.49 | 3.50 % | - | p50 0.88 s |
  | FR-S2 DEV | - | 7.25 % | 97.5 % | p50 0.72 s |

  - DEV-EOU-FR-v2 internal class: FE 55.2 % for v4 and 68.4 % for v1.1.
  - Premature FR cuts sit at clause punctuation: 16.0 % mid-phrase vs a 56.4 % control (FR-E0).

## 2. Difference table

| aspect | our Stage2-v4 | NeMo `asr_eou` (source) | consequence / what to test later |
|---|---|---|---|
| **Where `<EOU>` goes in the text** | `ids + [<EOU>]` on every draw, one utterance per clip, nothing after it (`two_stage.py:236`) | `text + "<EOU>"` per utterance (no space unless `add_sep_before_eou`), `add_eou_to_text` default True. Multi-utterance `A<EOU> B<EOU>` is possible in code via `utterances` / `sou_time` / `eou_time` lists. (`NeMo@b19b1a7:nemo/collections/asr/data/audio_to_eou_label_lhotse.py:172-173, 380-410`) | Same text semantics. In both recipes the "absorbing" post-EOU state goes untrained. Later: whether multi-utterance targets (text after `<EOU>`) help, which NeMo supports and we do not use. |
| **Timing source** | None. No aligner. The EOU position is left entirely to RNNT alignment. The energy endpoint is used only at eval | `sou_time` / `eou_time` from forced alignment (`tools/nemo_forced_aligner/align_eou.py`, CTC model) is recommended (README sec. 2.2). Without them the whole clip is one utterance and the last frame is EOU (`audio_to_eou_label_lhotse.py:322-331`) | **The training loss ignores timing in NeMo too**: `training_step` uses `batch.text_tokens` only (`NeMo@b19b1a7:nemo/collections/asr/models/asr_eou_models.py:435-463`). Frame labels feed validation metrics only (l.188-231). So timing changes NeMo's *metrics* and its *padding placement*, not the loss. Later: aligned `eou_time` matters for us only once we pad relative to the true speech end (see silence row). |
| **Frame labels** | None in training. Eval uses the energy endpoint of the original clip | 0 non-speech / 1 speech / 2 EOU / 3 EOB. EOU sits on the frame at `eou_time` (l.35-38, 355-368). Optional `pad_eou_label_secs` repeats the label into the following silence, default 0.0 (l.174, 279-310) | Not comparable 1:1 with our FE (see metrics rows). |
| **`<EOB>` semantics** | Never trained. The stock `<EOB>` rows are kept untouched. Decoding treats only `<EOU>` | "End of backchannel", optional `is_backchannel` from an English phrase list (`scripts/asr_eou/add_eob_labels.py`, README sec. 2.3). README, yaml and val all set `ignore_eob_label: true` = "treat them the same as EOU" (README sec. 2.7; `.../streaming_large.yaml:59, 107, 127`) | No practical difference: NVIDIA's shipped configs do not train EOB either. Later: an IT/FR backchannel list exists in neither recipe. |
| **Tokenizer expansion** | None in v4. The stock SPE already contains `<EOU>` / `<EOB>` (1024 / 1025). Our `tokenizer_eou.py` (new SPE + specials appended) is the experimental path that collapsed on 2026-10-09 and is not used | `add_special_tokens_to_sentencepiece.py` appends `<EOU>`, `<EOB>` as the last two pieces, score 0, type USER_DEFINED by default (`NeMo@b19b1a7:scripts/asr_eou/tokenizers/add_special_tokens_to_sentencepiece.py:120-140, 184-187`), 1024 -> 1026. The dataset refuses a tokenizer whose last two ids are not those strings (`audio_to_eou_label_lhotse.py:194-207`) | We inherit a **pretrained** `<EOU>` row (trained on English). NeMo starts it from scratch. Later: relevant only for a new-script language (AR line, kept separate), not for IT/FR. |
| **Initialisation of the new rows** | N/A: the stock `<EOU>` joint and embedding rows are reused as trained by NVIDIA, then further trained in stage 1 (where they get no positive targets) and stage 2 | `init_from_pretrained_nemo`: strict load if shapes match. Otherwise it copies encoder, pred-net LSTM, joint enc/pred and old vocabulary rows and **requires new vocab = old + 2**. Joint output rows for EOU/EOB use `token_init_method` (default `constant`: weight value None = left at the fresh random init, bias `token_init_bias_value` -1000). The decoder embedding rows for EOU/EOB are not copied (fresh init) (`NeMo@b19b1a7:examples/asr/asr_eou/speech_to_text_rnnt_eou_train.py:192-318`; README sec. 1.3) | A -1000 bias makes EOU start silent, so NeMo learns it from zero. Our row starts with English EOU behaviour. Later: none for IT/FR. |
| **Data sources** | IT: MLS, CV, VP-t50, FLEURS, two Granary pseudo-label pools. FR: MLS-fr, CV, VP-t50, FLEURS, Granary-fr. Clip ends = corpus segmentation (MLS cut at the longest silence in a 10-20 s window; Granary/VP segmentation not verified). 100 % of draws carry `<EOU>` | Example blend: 0.9 "EOU" group = LibriTTS train-clean-360 with aligned `sou`/`eou` (sentence-split corpus) and 0.1 "ASR" group = LibriSpeech other-500 (README sec. 2.5). The ASR group still gets `<EOU>` (no timestamps -> fallback, l.322-331, 380-382). README: "each sample should contain a single utterance, otherwise the model's EOU prediction accuracy will be degraded"; drop punctuation "for lowest possible EOU latency" (README sec. 2). Card: EOU-120M data = AMI, DialogStudio subset, Granary, Speech Commands, LibriTTS, 10k h NeMo ASR Set 3.0 (HF card); how its labels were made is **UNVERIFIED** | Both put `<EOU>` on every target. Our EOU positives include many clip ends that are pause cuts rather than sentence ends (MLS, Granary). NVIDIA's EOU group is sentence-split. Later: EOU only on sentence-complete ends (a deviation from NeMo, which labels every sample). |
| **Silence padding** | p 0.5, **trailing only**, U(1, 3) s, Gaussian -90..-60 dB, appended to the raw clip end (any natural tail stays in front of it). The other 50 %: no pad, `<EOU>` at the natural tail (often < 0.4 s) | `random_padding` **prob 0.99** (large yaml l.61-71) / 0.9 (README sec. 2.7). Total U(min_pre+min_post, max_total - dur), post = U(**min_post 3.0 s**, total - min_pre), pre = the rest, each side capped at **6.0 s**, `max_total_duration` 40. **Digital zeros**, before and after (`audio_to_eou_label_lhotse.py:414-499`). Data must be trimmed of leading/trailing silence (README sec. 2) | Largest recipe gap. NVIDIA shows >= 3 s of post-silence on ~all samples. We show 1-3 s on half and short tails on the other half. Our autopsy found the model learnt "about 0.8-1 s of silence => `<EOU>`" (FR hazard 29 % at 1.04-1.6 s). Later: NVIDIA-like post-pad (p >= 0.9, >= 3 s), and leading pad (our earlier `--pre-pad-prob` E3 arm was worse). |
| **Noise / other augmentation** | None in stage 2 besides the faint noise inside the pad. Speech and natural pauses are clean. SpecAugment = model cfg | After padding, over pad + speech: white noise p 0.9 at -90..-46 dB, gain p 0.2 at ±10 dB, **real noise p 0.9 at SNR 0-20 dB** (large yaml l.73-87; README sec. 2.7; applied after the pad, `audio_to_eou_label_lhotse.py:233-237, 501-526`). `random_segment`, `speed`, `time_stretch` are refused for EOU (l.43). SpecAugment 2 F-masks, 10 T-masks (yaml l.150-151) | With noise everywhere, "low energy" is not a clean end-of-turn cue for NVIDIA. For us any quiet stretch looks like our pads. Later: noise at SNR 0-20 dB over the whole stage-2 sample, which needs a noise corpus. |
| **Loss weighting** | Plain RNNT, `mean_batch`, **FastEmit 0**. No EOU weight, no frame loss | Plain RNNT on `text_tokens` (asr_eou_models.py:435-463), **FastEmit 3e-2** ("a good start", README sec. 1.3; yaml l.271-276, whose comment says 1e-4..1e-2). No EOU-specific weight, no frame-level EOU loss | No EOU weighting in either. FastEmit pushes emissions earlier: it may cut latency and may raise early cutoffs. Its sign for early cutoff is undocumented. Later: FastEmit as one variable (our `jobs/s2b.sh` ablation exists from 2026-10-09). |
| **Training schedule / freezing** | Two stages: stage 1 plain language ASR (no `<EOU>`), stage 2 = 1500 steps (selected 250-1250), bs 16, top-4 encoder 3e-5 + decoder/joint 2e-5, cosine, 75-step warmup, encoder in eval mode | **One** stage: ASR + EOU together from the start. AdamW, Noam `lr 5.0`, d_model scaling, warmup 10k, min 1e-6 (peak ~2.2e-3: INFERENCE from the Noam formula), wd 1e-3, fp32 (yaml l.280-308). Example `max_steps` 1,000,000 with `limit_train_batches` 1000, val every 1000 (README sec. 3); yaml `max_steps` 100k. `freeze_encoder` optional, default False (train script l.336-338) | Our stage 2 is a short, low-LR, mostly frozen fine-tune on top of a language model that never saw `<EOU>`. NeMo's is a long full-model run. No public ablation of one- vs two-stage. Later: a longer / whole-encoder stage 2 (`S2POL`) as one variable. |
| **Checkpoint selection** | Lowest macro WER among evals with EOU rate >= 80 % and empty <= 10 % (blind to early EOU) | `exp_manager` monitors `val_wer` (min) (yaml l.326). No EOU metric in selection | Neither selects on early cutoff. Later: add FE / FE-internal to the in-run rule. |
| **EOU decoding at inference** | Greedy RNNT, `<EOU>` read from the 1-best token stream. After `<EOU>`: decoder state reset (LSTM zero, SOS), encoder cache kept (Mynah and `s2_reset.py`). No threshold, debounce or probability | Eval: greedy hypotheses. EOU = `<EOU>` token present in the frame-level token sequence. `eou_prob` = the softmax prob of the EOU id, logged but unused because the threshold is 0.0 (asr_eou_models.py:116-170, 219-231). Voice agent **(2026-10-18 read)**: on `<EOU>`/`<EOB>` it resets audio buffer, **encoder cache** and decoder; VAD stop (Silero, 1.2 s) is the fallback; no debounce | Runtime differs (decoder-only reset vs full reset). Equivalence after a *premature* EOU inside speech is unknown. Later: compare the reset semantics on internal-pause clips (CPU). |
| **Metric: early cutoff** | **FE** = % of clips with an `<EOU>` while >= 0.5 s of speech remains (energy endpoint). FE_any = >= 0.16 s of speech after it. Per class / stratum on DEV-EOU-FR-v2 | `evaluate_eou`: the reference segment runs from the previous EOU (0 for the first) to the EOU frame. A prediction with `r_start <= p_end < r_end - collar` is an **early cutoff** (FP, distance logged). Collar **0.0** in validation. `early_cutoff_rate` = number of early cutoffs / number of reference utterances (average count, not a clip share) (`NeMo@b19b1a7:nemo/collections/asr/parts/utils/eou_utils.py:75-190, 194-213, 266-295`) | NeMo counts **any** EOU before the labelled end (no 0.5 s slack, counted per event). It also counts one emitted in leading silence, because the segment starts at 0. Our FE is per clip with a 0.5 s slack. NeMo's number is stricter and not directly comparable. Later: report NeMo's definition next to FE on the same dump. |
| **Metric: miss** | Recall = first `<EOU>` at or after end - 0.5 s. No `<EOU>` = miss. No time cap other than the +2 s quiet | `miss_rate` = references with no matched or late prediction / utterances. An early-cut reference that never gets a later EOU is removed from `missing` (counted only as early cutoff) (eou_utils.py:170-181, 295) | Similar concept. NeMo's "missing" excludes early-cut references, ours does not. |
| **Metric: latency** | First true `<EOU>` time minus the energy endpoint of the original clip, p50 / p90. Includes the natural tail. Typical v4 0.72-0.88 s p50 | `p_end - r_end` for exact (collar 0) and late matches, percentiles p50 / p90 / p95 (eou_utils.py:143-164, 266-292). `r_end` = aligned `eou_time` (last word end). Card: p50 160 / p90 280 / p95 320 ms on TTS DialogStudio + 3 s silence (HF card) | Different reference points (aligner word end vs energy endpoint) and different audio (TTS + 3 s vs natural + 2 s at -75 dB). Our 0.72 s and NVIDIA's 0.16 s are not comparable as is. Later: run the NeMo protocol (TTS-like, aligned end) on our checkpoint to get a like-for-like number. |
| **Evaluation sets** | IT: DEV730 4-set harness, FINAL-IT 6 banks (1,826 clips, sealed, one-shot), Mynah FLEURS-it 200 streaming. FR: DEV-fr 800, FR-OOD1 800 (retention only), FINAL-FR / FINAL-FR-v2 (sealed, spent), DEV-EOU-FR-v2 (mTEDx + MLS, 274 clips, internal vs endpoint classes, natural audio) | `generate_noisy_eval_data.py` + `scripts/asr_eou/conf/data.yaml`: constant pad **0.2 s pre / 3.0 s post**, real noise **p 1.0 at SNR 0-20 dB**, white noise / gain off. Single-utterance clips. No internal-pause class. English only | NVIDIA's eval has no "internal pause" class, so the failure we see (pause-length rule) would be invisible in it. Later: keep our internal-class banks as the primary EOU test. |
| **Streaming geometry** | Unchanged from the stock model ([70,1], chunked_limited, causal) and asserted | `att_context_size [70,1]`, `chunked_limited` (yaml l.186-187; README sec. 3: lookahead 80 ms, chunk 160 ms) | Same. |

## 3. Summary of material differences (OBSERVED differences; causal role untested)

1. **Padding.** NeMo pads almost every sample with >= 3 s of post-silence, plus a leading pad. We pad half of the
   samples, with 1-3 s and trailing only.
2. **Noise.** NeMo puts real noise at SNR 0-20 dB over the whole sample. We add none.
3. **FastEmit.** NeMo uses 3e-2, we use 0.
4. **Stages.** NeMo trains everything in one long stage. Ours is a short, mostly frozen stage 2 on top of a plain
   parent.
5. **Metrics.** NeMo's early cutoff is per event with no slack and a collar of 0. Our FE is per clip with a 0.5 s
   slack. The latency anchors differ too (aligner word end vs energy endpoint).

What is the same: plain RNNT on `text<EOU>`, no frame loss, `<EOU>` on every target, EOB unused, single-utterance
targets with nothing after `<EOU>`, and the [70,1] geometry.

## 4. FUTURE Stage2-v5 candidates (record only; not scheduled)

Scope: **the French early-cutoff failure mode only.** In FR-E0, RC-FR v1.1 fired a premature `<EOU>` at internal
pauses in 68.4 % of the DEV-EOU-FR-v2 internal-class clips (v4: 55.2 %). The cuts sit at clause punctuation (16.0 %
mid-phrase vs a 56.4 % control). This candidate is **unrelated to the Arabic AR-A0 / AR-A1 line** (tokenizer
romanisation vs native SPE, `.work/arabic-specialist-plan-2026-10-24.md`), and must not be merged with it, run with
it or compared against it.

**Candidate F-EOUJ: a separate EOU joint, second stage, rest frozen.** Source: Li et al., "A Language Agnostic
Multilingual Streaming On-Device ASR System", Interspeech 2022, arXiv 2208.13916 (https://arxiv.org/abs/2208.13916,
PDF read 2026-10-24).

- What the paper says (OBSERVED, sec. 2.3 and 4.2):
  - RNN-T EP appends an EOU token to every training wordpiece sequence.
  - "Since the EOU token only appears at the sentence end during training, E2E model learns to stop emitting
    tokens after it. This can cause early cutoffs when inaccurate EOU predictions are made."
  - Adding EOU "further degrades the recognition quality" (Table 2, S4 vs S3).
  - The fix is two-stage training. Stage one optimises encoder, prediction network and the original joint for
    recognition. Stage two freezes all of them, "initialize[s] a new EOU Joint Layer from the original joint layer
    and fine-tune[s] only this EOU Joint Layer".
  - At inference the original joint does the (beam) decoding and the EOU joint only supplies the EOU probability.
  - FastEmit is 5e-3, with no extra late penalty for EOU.
  - The system also has a separate acoustic Encoder Endpointer. 9 locales, including fr and it.
- What the paper does not give (UNVERIFIED / NOT FOUND):
  - an early-cutoff rate before vs after the change (the cutoff risk is stated as a motivation, not measured in the
    text read);
  - how the EOU probability is combined with the acoustic endpointer at decision time;
  - the EOU-joint training data and label policy.
- Why it fits the FR failure (HYPOTHESIS):
  - Our early cuts come from the same tokens that drive recognition.
  - The v1.1 result (a longer plain stage, H2, gave a higher internal FE) shows that ASR training and EOU behaviour
    are coupled in one joint.
  - A separate EOU joint decouples the two. The ASR path stays bit-identical to the parent, so the reset-WER cost of
    stage 2 (+1.87 offline in IT) disappears by construction. EOU becomes a probability that can be thresholded.
  - Threshold that probability, not the greedy token. That turns today's binary "token fired" event into an
    operating curve (FE vs latency) to compare against the confirmation-window frontier already measured in FR-E0
    (v1.1: W 0.32 s -> internal FE 43.1 % at p50 1.20 s).
- Mapping to our model (INFERENCE; nothing implemented):
  - Parent = the FR plain checkpoint, fully frozen (encoder, prediction net, joint).
  - Add a copy of `joint` (enc/pred projections + output layer) as `joint_eou`, initialised from the parent joint.
  - Train only `joint_eou` with the stage-2 targets (`text<EOU>`).
  - Decode text with the original joint and read P(`<EOU>`) per frame from `joint_eou`.
  - This needs a runtime change (a second joint evaluation per frame) in Mynah and in the export. It is not a
    drop-in `.nemo`: NeMo's EOU recipe has no two-joint model class, so this is **UNVERIFIED** against NeMo.
- What a later pre-registration would have to fix (not now):
  - the parent (FR-H2 or its successor);
  - the `joint_eou` data (v4 mix + v4 label policy, one variable at a time);
  - the decision rule (threshold on P(`<EOU>`) chosen on DEV-EOU-FR-v2 only, FINAL-EOU-FR-v2 sealed);
  - the comparison targets: v1.1 native and v1.1 + window W at equal endpoint-class latency;
  - gates on internal-class FE at matched p50, recall, and reset-WER = parent (must hold exactly);
  - the runtime cost of the second joint (CUDA / CPU per-frame).
- Other v5 candidates already recorded elsewhere (not repeated here): NVIDIA-like padding / noise arms
  (`.work/eou-ft-prior-art-2026-10-18.md` sec. 5); the closed truncated-prefix arm (FR-E0 rule FAIL); FR-v2
  pause-stretch (FAIL, closed).
