# EOU 120M -> new language (Italian first): EXPERIMENTAL tooling

**Status: no working Italian EOU model exists.** The combined language + EOU
recipe of 2026-10-09 collapses toward blank (all four arms). Everything here is
research tooling for the ablation ladder in `../README.md`; nothing is a
validated recipe. Base model: `nvidia/parakeet_realtime_eou_120m-v1` (cache-aware
FastConformer 17L, att context [70,1], RNNT, `<EOU>`=1024, `<EOB>`=1025,
blank 1026). Data: the Canary kit's manifests (`../canary/prepare_it.py`).

| file | stage | what |
|---|---|---|
| `plain_asr.py` | 1 | plain ASR on the STOCK model + stock tokenizer (de-accented targets), minimal torch loop, FastEmit 0, no padding/gain; `--micro-overfit` debug preset; per-eval greedy+beam hyps, WER/CER, empty/EOU rate, blank margin, non-blank %, audio dependence, silence probe; `--resume` |
| `train_eou.py` | 2 (machinery) | NeMo `EncDecRNNTBPEEOUModel` + lhotse EOU dataset (zero padding, white noise), energy trim, gain aug, cold/warm decoder+joint, encoder freeze, FastEmit from the stock cfg, NeMo cache-aware streaming check, saved as plain `EncDecRNNTBPEModel`. `--lr` and `--eou-weight` have NO default (the 2026-10-09 values are the known-bad config) |
| `tokenizer_eou.py` | 2 | new SPE copying the stock spec + `<EOU>`/`<EOB>` appended as user-defined pieces; `check_special_ids` = the id contract |
| `rnnt_diag.py` | - | blank margin at the SOS state, non-blank frame share, beam/greedy switch, audio dependence |
| `diag/blank_margin.py` | - | greedy vs beam vs per-frame blank margin on a checkpoint (CPU) |
| `diag/batch_probe.py` | - | what the EOU dataloader really feeds (levels, padding, target tail with `<EOU>`) |
| `diag/encoder_diff.py` | - | base vs fine-tuned encoder: weight change, output cosine, delta norm, temporal std, near-constant dims, cross-utterance cosine, block-shuffle test, audio dependence |
| `diag/tok_coverage.py` | - | stock tokenizer coverage of a language's transcripts (`<unk>` chars, round trip) |
| `export_to_mynah.sh` | - | `.nemo` -> Mynah pack (`tools/convert_nemo.py`), id/preset checks, `mynah-asr stream --deltas` sanity, then prints the `tools/eval/lang_gate.py` / `tools/eval/eou_metrics.py` commands for endpoint latency / premature / miss on a language bank |

Endpoint evaluation is not re-implemented here: once a pack exists, use
`tools/eval/eou_metrics.py` (`--mode composite` measures premature EOU inside
pauses, which this training data does not teach).

## NeMo v3.0.0 recipe facts relied on

Line numbers are NeMo tag v3.0.0 (`https://github.com/NVIDIA/NeMo/blob/v3.0.0/<path>`).

| # | fact | source (file:line) |
|---|---|---|
| 1 | `<EOU>`/`<EOB>` are appended to an EXISTING SentencePiece model as the last two pieces, type 4 (USER_DEFINED), score 0; vocab 1024 -> 1026, so `<EOU>`=1024, `<EOB>`=1025; NeMo adds blank after them (1026) | `scripts/asr_eou/tokenizers/add_special_tokens_to_sentencepiece.py:124-138`; README `examples/asr/asr_eou/README.md:23-33` |
| 2 | the dataset REFUSES a tokenizer whose last two ids are not `<EOU>`/`<EOB>` | `nemo/collections/asr/data/audio_to_eou_label_lhotse.py:192-205` |
| 3 | model class of the recipe: `EncDecRNNTBPEEOUModel` (= `EncDecRNNTBPEModel` + `LhotseSpeechToTextBpeEOUDataset` + EOU metrics in validation); the released `.nemo` is saved as plain `EncDecRNNTBPEModel` | `nemo/collections/asr/models/asr_eou_models.py:370-411`; `examples/asr/asr_eou/speech_to_text_rnnt_eou_train.py:329` |
| 4 | **the training loss is plain RNNT on the text with `<EOU>` appended**; the frame labels (`eou_targets`) are NOT used by `training_step`, only by validation metrics | `asr_eou_models.py:423-519` (uses `text_tokens` only); text: `audio_to_eou_label_lhotse.py:370-410` (`add_eou_to_text` default True, l.171) |
| 5 | manifest without `sou_time`/`eou_time` is legal: the whole clip is one utterance, last frame = EOU (label 2) | `audio_to_eou_label_lhotse.py:320-330` |
| 6 | EOU-positive construction = random padding with **digital zeros** before/after the speech: `prob 0.99` (large yaml) / 0.9 (README), `min_post_pad_duration 3.0`, `max_pad_duration 6.0`, `max_total_duration 40`, uniform | `audio_to_eou_label_lhotse.py:412-497`; `examples/asr/conf/asr_eou/fastconformer_transducer_bpe_streaming_large.yaml:61-71`; README:184-192 |
| 7 | augmentations after padding: white_noise p0.9 -90..-46 dB, gain p0.2 -10..+10 dB, noise file p0.9 SNR 0-20 dB; `speed`/`time_stretch`/`random_segment` refused (they would move the timestamps) | `audio_to_eou_label_lhotse.py:40-41,178-190`; yaml:73-87 |
| 8 | data must be single utterances WITHOUT leading/trailing silence ("otherwise EOU accuracy degraded"); timestamps from `tools/nemo_forced_aligner/align_eou.py` with a CTC model | README:77, 97-107 |
| 9 | EOB (backchannel) labels: optional, English phrase list; `ignore_eob_label: true` treats EOB as EOU | README:110-122, 182; yaml:59 |
| 10 | blend 0.1 plain-ASR group + 0.9 EOU group via lhotse `input_cfg` | README:128-152 |
| 11 | recipe init (`init_from_pretrained_nemo`): strict load if same shapes; else encoder + pred-net LSTM + joint enc/pred projections + old vocab rows, and **requires new vocab = old + 2** (raises otherwise); new rows `token_init_method constant`, bias -1000 | `speech_to_text_rnnt_eou_train.py:191-318`; yaml:26-28 |
| 12 | NEW vocabulary (new language): docs say set `model.tokenizer.dir` to the new tokenizer and initialise with `init_from_nemo_model_exclude: [decoder, joint]` (encoder [+ preprocessor] only) | https://docs.nvidia.com/nemo/speech/nightly/asr/fine_tuning.html ("Tokenizer Changes") |
| 13 | `change_vocabulary()` on RNNT-BPE rebuilds joint + decoder AND rebuilds the loss as `RNNTLoss(num_classes=...)` **without the cfg's loss kwargs** -> FastEmit 0.03 silently dropped if one trains right after it | `nemo/collections/asr/models/rnnt_bpe_models.py:270-345` (loss at ~l.340) |
| 14 | streaming settings: `att_context_size [70,1]`, `att_context_style chunked_limited`, `conv_context_size causal`, `causal_downsampling true`, preprocessor `normalize: "NA"`; `fastemit_lambda 3e-2`; `fused_batch_size 4`; greedy_batch max_symbols 10; README sets `att_context_size=[70,1]` on the command line | yaml:138,167,186-187,200,239,247-251,276; README:248,263 |
| 15 | optimiser of the recipe: AdamW, Noam lr 5.0 (d_model 512, warmup 10k -> peak ~2.2e-3), `precision: 32` (RNNT loss runs fp32 anyway) | yaml:283-292, 308 |
| 16 | recipe eval: `speech_to_text_eou_eval.py` (WER + EOU latency / early cutoff / miss); `generate_noisy_eval_data.py` pads/noises | README:155-169, 266-298 |

Consequences used by the kit: no forced aligner is needed for training (facts
4-5; an energy trim replaces `align_eou.py`, fact 8); a new vocabulary goes
through fact 12 (fresh decoder/joint, encoder copied), not `change_vocabulary`
(fact 13 would drop FastEmit) nor the +2 loader (fact 11). Every streaming key
of fact 14 is asserted unchanged before training and after save.

Real vs synthetic in the EOU labels: transcripts, utterance boundaries and the
voiced span are real; every pause (digital zeros + white noise), the gain and
the absence of incomplete-utterance negatives / backchannels are synthetic or
missing, as in the public recipe without `align_eou`.

## The working path (2026-10-09 evening): two stages, `two_stage.py`

Validated on one L40S, MLS-it 40 h, accent-insensitive scores:
stage 1 plain Italian ASR from the STOCK EOU 120M with its STOCK tokenizer
(de-accented targets), stock decoder/joint, top-4 encoder blocks unfrozen at
3e-5, decoder/joint 3e-4, FastEmit 0, no padding, 4000 steps -> MLS-it val
WER 41.7 / CER 12.3; stage 2 from that best with text + <EOU> targets, 50 % of
the samples with 1-3 s trailing quiet, lr 1e-4, 1500 steps -> 40.4 / 12.5,
0 empty, <EOU> on 98 % of the val clips. Inside Mynah (CPU stream + CUDA
server): CPU/CUDA parity incl. EOU events, 1.9 GiB VRAM, 0 % premature EOU;
the single-clip endpoint is slower than the English stock (p50 2.3 s on
FLEURS-it). Jobs: `../jobs/chain_it.sh` (40 h + stage 2), `../jobs/m1.sh`
(Mynah evaluation), `../jobs/s2b.sh` (FastEmit 0.03 ablation),
`../jobs/close.sh` (end-of-session archival). `plain_asr.py` is the earlier
port of the same loop; `two_stage.py` is the exact file that produced the
numbers above (stage 2 options, EOU-aware best selection, per-utterance greedy
eval because NeMo's batched greedy crashed mid-run).
