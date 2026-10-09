# Italian-specialist fine-tune of parakeet_realtime_eou_120m-v1 (Phase 0 notes)

Goal: an Italian ASR + integrated `<EOU>` model on the stock 120M cache-aware
encoder, still streaming ([70,1], causal), still a plain RNNT pack Mynah can run.
English forgetting is accepted (specialist). Background on the model, licence and
the earlier plan: `.work/lightweight-asr-upstream.md` §1, `.work/lightweight-asr-ft.md`
§2.2 / Track E. Kit: `.work/lightweight-asr/eou-ft/`.

All line numbers below are NeMo tag **v3.0.0** (`https://github.com/NVIDIA/NeMo/blob/v3.0.0/<path>`).
The NVIDIA-NeMo/Speech `main` copy of `examples/asr/asr_eou/README.md` differs from
v3.0.0 only in two links (NFA and voice-agent now point at NVIDIA-NeMo/Speech).

## Recipe facts relied on

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

## Decisions for the Italian kit (and why)

- **Tokenizer**: new Italian SPE, same `model_type` and normaliser spec as the
  stock `tokenizer.model` (read from the `.nemo` at run time, not assumed), vocab
  1024 incl. `<unk>`, trained on the MLS-it training-pool text normalised the way
  the stock model's targets are (lower case, no punctuation, apostrophe kept);
  then fact 1 reproduced in-process (USER_DEFINED pieces appended) and checked by
  reload: `<EOU>`=1024, `<EOB>`=1025, model blank = 1026.
- **Init**: fact 12 (the documented new-vocabulary path), not `change_vocabulary`
  (fact 13) and not the recipe's +2 loader (fact 11 cannot apply): the stock
  config with `tokenizer.dir` swapped builds a fresh `EncDecRNNTBPEEOUModel`
  (decoder + joint random, loss built from the cfg so FastEmit 0.03 stays), then
  `encoder.*` and `preprocessor.*` are copied from the stock state dict (strict
  on those prefixes). Every streaming key of fact 14 comes from the stock cfg and
  is asserted unchanged before training and after save.
- **Labels**: fact 4 + 5 mean no forced aligner is needed for TRAINING: each MLS
  utterance -> text + `<EOU>`, zero padding per fact 6 (large-yaml values). The
  aligner of fact 8 is replaced by an energy trim (rows get `offset`/`duration`
  of the voiced span, -35 dB below the clip's loudest 25 ms frame, 0.10 s
  margins). Blend per fact 10: 0.9 EOU group (padded) + 0.1 plain group (same
  audio, no padding, still with `<EOU>`: the dataset always appends it).
- **Gain**: random gain -30..+6 dB (p 0.8), applied to the padded batch on the
  device (Canary kit `ftlib.make_gain_aug`), on top of the recipe's white noise;
  the recipe's own gain (p0.2, +-10 dB) is left out so the range is ours. No
  noise-file augmentation (no noise corpus on the box).
- **Trainable**: whole model; encoder LR x0.3 (`--enc-lr-scale`), decoder/joint
  at `--lr` 1e-3 cosine (fresh pred-net/joint need a high LR; the recipe trains
  everything, fact 15). `--freeze-encoder-layers N` freezes subsampling + the
  bottom N layers.
- **Precision**: bf16-mixed (RNNT loss fp32 internally).

## Real vs synthetic in the EOU labels

- real: transcripts (MLS-it, book text auto-aligned by MLS), utterance
  boundaries (MLS segmentation), the voiced span (energy trim of real audio);
- synthetic: every pause before/after the utterance (digital zeros + recipe white
  noise at -90..-46 dB), the gain; there are NO incomplete-utterance negatives,
  no backchannels (EOB unused), no multi-utterance clips with internal pauses —
  the same as the public recipe without `align_eou` (which also only marks one
  utterance per clip). Premature-EOU behaviour inside pauses is therefore
  untrained beyond what the stock encoder brings, and must be measured
  (`tools/eval/eou_metrics.py --mode composite`).

## Not read / open

- `tools/nemo_forced_aligner/align_eou.py` and `scripts/asr_eou/add_eob_labels.py`
  were not re-read for this kit (not used). Whether `align_eou.py` accepts the
  Italian hybrid `stt_it_fastconformer_hybrid_large_pc` stays UNKNOWN.
- The exact label pipeline/data mix of the released checkpoint is unpublished.
