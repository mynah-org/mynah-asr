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

## Measurements, 2026-10-09 (L40S, NeMo 3.0.0, MLS-it 5 h, eou kit)

Fixes needed before the first step ran: tokenizer paths in the model cfg
(register_artifact on a None model_path), and the numba RNNT loss
(numba 0.68 needs the `numba-cuda` package; numba-cuda 0.30.4 needs numpy < 2.4).

| run | encoder | decoder/joint | lr (dec/joint, enc) | steps | train loss | FLEURS-it / MLS-it WER | EOU emitted |
|---|---|---|---|---|---|---|---|
| eou-it-5h-e52 (cold) | trainable | reinitialised | 1e-3, 3e-4 | 2987 | 404 -> ~10 | 100 / 100 (all empty) | 0 |
| eou-it-5h-e52-warm | trainable | stock | 1e-3, 3e-4 | stopped at ~1600 | | in-loop VAL 100 / 100 at 1000 and 1500 | 0 |
| eou-it-5h-e26-frz (cold) | FROZEN | reinitialised | 1e-3, - | 1494 | ~119 at 1200 | 99.57 / 99.69 (VAL 98 at 500 -> 99 at 1000) | 0 |
| eou-it-5h-e26-frz-warm | FROZEN | stock | 1e-3, - | 1494 | 441 (50) -> 179 (500) -> 187 (1000) -> 154 (1450) | 100 / 100: no text token at 500 already; EOU rate 0.72/0.83 at 500 -> 0.03/0.12 at the end | decays to ~0 |

Diagnostics on the cold checkpoint (`blankdiag.py`, `batchprobe.py`, CPU):
- greedy empty on TRAIN and eval clips; max(non-blank) - blank < 0 on every
  frame of 10 clips (max -0.26 .. -0.65);
- beam-4 returns the SAME sentence ("che ero un uomo di cartapesta senza
  sangue nelle vene ...") for every input: the RNNT became an audio-independent
  text prior;
- RNNT loss on a train clip: text+<EOU> 44.9, text only 192.9, <EOU> only
  38.6: the objective converges toward the targets, but NOT to a decodable,
  audio-conditioned distribution;
- encoder weights barely moved (median 0.9 %, max 9.3 %), encoder output cosine
  to stock 0.911, but per-channel temporal std 0.169 -> 0.040; the FROZEN arm
  collapses as well, so encoder drift is not the primary cause;
- a real training batch is sane: levels (rms 0.03-0.10, peak 0.3-0.7),
  durations 12-27 s after padding, Italian text with <EOU> (1024) last;
- ~1 % of the cuts raise a suppressed lhotse AudioLoadingError (offset
  ignored, declared vs loaded samples differ): to fix, not the cause.
- at step 500 there is still some output; continued training converges to
  blank: the objective/optimisation REWARDS the degenerate solution.
Suspects, in order: decoder/joint lr 1e-3; 90 % of the samples padded with
3-6 s of silence (most frames are blank targets); FastEmit 0.03 with that
padding distribution vs the NVIDIA recipe's actual settings.

## Next session: ablation ladder before any 5 h run

1. Micro-overfit: 16-32 Italian utterances, no artificial padding, no gain,
   FastEmit 0, encoder frozen, stock decoder/joint, low lr (1e-4). A healthy
   RNNT must memorise them and greedy must return sensible text.
2. Then add ONE variable at a time: plain -> +<EOU> -> +padding ->
   +FastEmit -> +gain -> unfreeze encoder top blocks (discriminative lr,
   encoder 1e-5..3e-5).
3. Checkpoints/metrics at steps 0, 100, 250, 500, 750, 1000: train loss,
   WER, non-blank frame %, blank margin, EOU rate, and an audio-dependence
   check (different audio -> different hypotheses).
4. Only then 5 h -> 20 h, and EOU-aware second stage (incomplete-utterance
   negatives, backchannels) per the NVIDIA recipe.

Reading of the last arm (frozen stock encoder + stock decoder/joint, i.e.
starting from a model that decodes): the training loop first switches off
EVERY text token (at step 500 the model emits only <EOU>), then <EOU> too,
until only blank remains. With the encoder untouched and a working start,
this is near-causal evidence that the TRAINING RECIPE drives the RNNT to the
blank solution (suspects above), not the language transfer or the
initialisation. Stopped here for the session, per the stop/go rule.

## The two-stage path WORKS (2026-10-09 evening, `eou-ft/plain_it.py`, `jobs/chain_it.sh`, `jobs/m1.sh`)

Correction of approach: NVIDIA's EOU recipe starts from an ASR that already
knows the language, keeps its vocabulary and appends two tokens. So: stage 1 =
plain Italian ASR from the STOCK model with the STOCK tokenizer (Italian
de-accented: the English SPE maps accented vowels to <unk>, 1.8 % of tokens;
otherwise 2.8 tokens/word, round-trips), stock decoder/joint, FastEmit 0, no
padding, no gain; stage 2 = text + <EOU> targets from the stage-1 best.
Minimal PyTorch loop (no Lightning/lhotse). Scores below are accent-insensitive.

| run | data | encoder | steps | MLS-it val WER / CER | empty | EOU | wall |
|---|---|---|---|---|---|---|---|
| A0 micro-overfit | 32 utts | frozen | 400 | train set: 26.5 (100) -> 1.1 (200) -> 0.0 (300) | 0 | n/a | 65 s |
| B1 | 5 h | frozen | 1500 (died in eval at ~1750) | 69.9 (250) -> 59.1 (1500) / 26.6 | 11/200 | 0 | ~6 min |
| P40 | 40 h | top 4 unfrozen (lr 3e-5), dec/joint 3e-4 | 4000 | 54.7 (500) -> 48.2 (1000) -> 42.3 (2500) -> **41.7 / 12.3** | 0 | 0 | 813 s |
| S2 (stage 2) | 40 h, text+<EOU>, 50 % with 1-3 s trailing quiet | same | 1500, lr 1e-4, FastEmit 0 | **40.4 / 12.5** | 0 | **98 %** | 436 s |

B1/B2 died in NeMo's batched greedy decoding (CUDA graphs: "illegal memory
access" in batched_hyps_to_hypotheses) during an eval, not in training; the
loop now evaluates with the per-utterance greedy decoder.

Inside Mynah (m1; pack converted with tools/convert_nemo.py, CLI stream f32):
- FLEURS-it test 200 clips (out of domain: training is MLS audiobooks):
  WER 55.12 / CER 18.97 pooled, 0 empty, model <EOU> on 175/200 clips.
- eou_metrics (60 clips, Silero speech end, gap 1 s): speech end -> EOU p50
  2346 / p90 3770 / p95 3866 ms; missed 1 s 95 %, 2 s 60 %, never 10 %;
  premature 0 % at every pause threshold; A + 1 s + B: EOU in the gap 58.3 %,
  latency p50 330 / p90 522 ms.
- CUDA: tests/test_cuda_stream gates A/B/C PASS on 4 Italian clips, CPU f32 ==
  GPU incl. the model eous (e.g. 1 @ 19.32 s); the CUDA server starts in
  500 ms, bf16 own-tc, 9 graphs, 1940 MiB VRAM — the stock EOU's profile.
Artifacts: HF private repo runs/plain-p40-b2pol, runs/plain-s2-eou-p40-b2pol
(final.nemo + last.ckpt torch state), size-verified.
Open: endpoint decision slower than the English stock on single clips
(s2b tests FastEmit 0.03 as the one changed variable); FLEURS gap (domain
and/or level: no gain augmentation was used tonight); accents (add the
accented vowels as appended tokens, as NVIDIA appends <EOU>/<EOB>); replicate
on FR or DE to show it is a method, not a lucky language.
