# Canary 180M -> Italian: restartable fine-tuning kit (VALIDATED)

The Canary track of `finetune/README.md` (C0 zero-shot probe, C1 tokenizer
surgery, C2/C3 fine-tunes, replay) as scripts for one rented Linux GPU box.
Measured results and the validated recipe: `finetune/README.md`.
Nothing here runs on the dev laptop except syntax checks and the synthetic
plan (`python3 prepare_it.py --synthetic`).

Target box: Ubuntu 24.04, NVIDIA L40S 48 GB now (L4 24 GB is the target the
defaults are sized for), driver 575 (CUDA <= 12.9), ~20 GB free disk,
30-CPU cgroup quota, NeMo 3.0.0 in `/root/nemo-venv` (`uv venv -p 3.12` +
`uv pip install nemo_toolkit[asr]`). No HF token needed: every dataset and the
checkpoint are public.

## Files

| file | step | what it does |
|---|---|---|
| `env.sh` | - | paths (`FT_ROOT`, default `/root/ft`; `VENV`, default `/root/nemo-venv`), `step`/`ok` marker helpers, disk guard |
| `setup.sh` | 1 | replaces torch in the venv with the same version built for **cu128** (fallback cu126), adds `num2words regex soundfile pyarrow sentencepiece protobuf`, fetches the Open ASR Leaderboard normaliser (pinned commit) and checks our copy against it, downloads `canary-180m-flash.nemo`, verifies CUDA + bf16 + `import nemo` |
| `prepare_it.sh` / `prepare_it.py` | 2 | MLS it + FLEURS it/en -> 16 kHz mono PCM16 wav + NeMo/lhotse manifests, frozen eval, nested 5/20/40 h cuts, exact hours |
| `replay.py` | 2b | optional replay pool against forgetting: FLEURS train en_us/de_de/es_419/fr_fr capped at `REPLAY_H` (2) h each -> `manifests/replay_train.json` (source_lang = target_lang = the row's language, pnc=yes), plus 100-clip FLEURS test probes `eval_fleurs_{de,es,fr}.json`; ~1 GB of wav |
| `tokenizer_it.sh` / `tokenizer_it.py` | 3 | `it` SentencePiece on the training-pool text, appended after `fr`, row copy-back, verification |
| `probe_it.sh` / `probe_it.py` | 4 | C0: untouched model with `<|it|>` (and `es`/`fr` floors) on the frozen IT eval + EN sanity |
| `train_it.sh` / `train_it.py` | 5 | smoke fine-tunes, arm A then B, with per-run accounting JSON |
| `eval_it.py` | 6 | WER/CER with S/D/I on the frozen IT eval for any `.nemo` |
| `report.py` | - | markdown summary of everything measured |
| `../common/ftlib.py` | - | manifests, normaliser, WER/CER/S/D/I, the shared decode loop, replay mix, gain aug |
| `../common/subsets.py` | - | speaker-balanced nested subsets |
| `../common/runmeta.py` | - | `run.json` provenance + checkpoint records |
| `run_all.sh` | 1-5 | all of the above in order |

## Command order on the box

```bash
# from the dev machine: copy the kit (scripts only, ~60 KB)
scp -r finetune root@BOX:/root/finetune     # the whole dir: canary/ needs ../common/

# on the box (FT_ROOT is required by every step; SUBSET by the training step)
export FT_ROOT=/root/ft
mkdir -p /root/ft/logs
bash /root/finetune/canary/prepare_it.sh --dry-run          # optional: metadata + plan only, prints hours (needs setup-deps first)
SUBSET=5h RATE_USD_H=<price per GPU-hour> \
  tmux new -d -s ft 'bash /root/finetune/canary/run_all.sh 2>&1 | tee -a /root/ft/logs/run_all.log'
# or one step at a time, same order:
bash /root/finetune/canary/setup.sh
bash /root/finetune/canary/prepare_it.sh                     # add --with-fleurs-train to put FLEURS train in the pool
bash /root/finetune/canary/tokenizer_it.sh
bash /root/finetune/canary/probe.sh
SUBSET=5h RATE_USD_H=... bash /root/finetune/canary/train_it.sh          # A-5h, then B4-5h
SUBSET=5h RATE_USD_H=... ARMS=B TOP_N=8 bash /root/finetune/canary/train_it.sh     # another rung, e.g. B8
SUBSET=5h RATE_USD_H=... GAIN_AUG=1 ARMS=A bash /root/finetune/canary/train_it.sh   # level A/B: A-5h-gain vs A-5h
RATE_USD_H=... SUBSET=20h ARMS="A B" bash /root/finetune/canary/train_it.sh
/root/nemo-venv/bin/python /root/finetune/canary/eval_it.py --nemo /root/ft/runs/A-5h/final.nemo --en
/root/nemo-venv/bin/python /root/finetune/canary/report.py
```

Every step is guarded by `timeout` and leaves a marker in `/root/ft/done/`;
re-running any script skips finished steps (delete a marker to redo one).
`prepare_it.py` has its own stage markers (`data-meta`, `data-plan`, `data-audio`,
`data-manifests`); the audio stage only fetches files that are missing, so a
killed download resumes. A killed training run restarts from step 0 (runs are
short; no mid-run checkpoint; `run.json` keeps status "started"), a finished run is skipped because its
`runs/<tag>/metrics.json` exists. To change the plan (seed, hours, FLEURS
train), delete `done/data*`, `plan.json` and `manifests/`.

Replay (`REPLAY_RATIO=0.2`, needs `replay.py` first): the IT manifest is
mixed with replay rows so ~20 % of the training audio hours are en/de/es/fr
(rows sampled from the pool with the run seed, whole file shuffled), steps grow
so the IT data is still seen `EPOCHS` times, and `metrics.json` gets a
`forget` dict (en/de/es/fr WER/CER before/after, per-language normaliser).
`REPLAY_RATIO=0` (default) trains exactly as before.

Checkpoints: `SAVE_CKPT=1` also writes `runs/<tag>/last.ckpt` (full Lightning
state: optimizer, scheduler, AMP scaler, global step; size in
`metrics.json:checkpoint`). A NEW stage only needs `--init runs/<tag>/final.nemo`;
`last.ckpt` is for resuming the SAME run (`trainer.fit(model, ckpt_path=...)`
with the same schedule).

## Expected wall time (L40S estimates; L4 roughly 2-3x for the GPU parts)

| step | wall | disk after |
|---|---|---|
| setup (torch cu128 wheel ~1 GB, deps, 0.74 GB checkpoint) | 5-15 min | +3 GB in the venv/cache, +0.74 GB |
| data: meta (parquet column projection, tsv) | 2-5 min | ~20 MB |
| data: audio (8 MLS shards of ~0.49 GB each, one at a time per worker, deleted after use; FLEURS it test tar 0.66 GB streamed; first 100 FLEURS en test clips) | 15-40 min | ~5.6 GB wav (40 h + 8.8 h eval at ~115 MB/h); transient peak +2 GB (4 shards) |
| data: manifests + hours | 1-2 min | - |
| tokenizer surgery + verification | 3-10 min | +0.75 GB (`canary-180m-flash-it.nemo`) |
| probe (3 prompts x 8.8 h eval + 100 EN clips) | 5-20 min | ~10 MB |
| train A-5h (10 equivalent epochs = ~50 audio-h, 2 in-loop validations, final eval) | 15-40 min | +0.75 GB |
| train B4-5h | 20-50 min | +0.75 GB |

The level sweeps add roughly 3x the decode time of the sets they cover
(`eval_it.py` full eval: ~4x a single pass; training runs: 600 IT + 100 EN
clips x 3 levels, before and after).

Total disk use stays around 9-10 GB above the venv (budget: peak < 15 GB).
Throughput numbers are NOT assumed anywhere: each run measures them.

## Data (step 2)

- **Frozen eval** = ALL of FLEURS it_it test (865 utts, ~3.5 h; raw cased text
  as reference) + ALL of MLS it test (1,262 utts, ~5.3 h). With
  `CV_TARBALL` set, also a fixed 2,000-clip Common Voice it test subset.
- **Training pool** = MLS it train (65 speakers, ~247 h), test speakers excluded
  (checked; MLS is speaker-disjoint by design). `--with-fleurs-train` adds
  FLEURS it train at a 15 % duration share, max 2 readings per sentence, test
  sentence ids excluded. Common Voice is optional (`CV_TARBALL=<MDC tarball>`
  or `MDC_API_KEY` + `CV_DATASET`; <= 20 clips per speaker, share `CV_SHARE`).
- **Nested cuts**: one deterministic, speaker-balanced sequence (seed
  20261009; each step takes the next shuffled utterance of the speaker with the
  least audio so far; cap 60 min per MLS speaker; 1-20 s utterances); 5 h,
  20 h and 40 h are prefixes of it, so 5 h c 20 h c 40 h by construction (and
  re-checked on the written manifests). `hours.json` records the exact seconds
  summed from the written wav files.
- **EN forgetting probe**: the first 100 FLEURS en_us test clips (sorted by
  sentence id).
- Manifest rows carry the canary2 fields NeMo 3.0.0 reads:
  `audio_filepath, duration, sampling_rate, text, source_lang=it,
  target_lang=it, pnc, taskname=asr` (+ `corpus, utt_id, speaker`). MLS text is
  lower-case and unpunctuated, so MLS rows use `pnc=no`; FLEURS/CV rows use
  `pnc=yes`. The other canary2 slots are left to NeMo's defaults.

## Tokenizer surgery (step 3)

`it` SentencePiece (default 1024, `IT_VOCAB=512` to try smaller) trained on the
whole training pool's transcripts (never eval text) with the `es`
sub-tokenizer's trainer settings. The aggregate tokenizer is rebuilt as
`spl_tokens, en, de, es, fr` in their original order, then `it` LAST: ids
0..5247 keep their meaning. `change_vocabulary` keeps the 4 decoder layers but
re-initialises the embedding (weight-tied head) and the head bias; the script
copies the 5,248 old rows back and seeds each `it` row from the `es` row of the
same piece (or the mean of its `es` decomposition). It then saves, restores the
`.nemo` fresh, and refuses to write `canary-180m-flash-it.nemo` unless the old
rows are bit-identical, the EN transcripts of 20 clips are identical to the
untouched model's, and Italian text tokenises into the `it` id range
(`tokenizers/surgery_report.json`).

## What each arm answers

| run | trainable | question |
|---|---|---|
| C0 probe `it` | none | How bad is the untouched decoder with `<|it|>` and only en/de/es/fr pieces? (the floor) |
| C0 probe `es`/`fr` | none | Is a "nearest language" prompt a better floor than `<|it|>`? |
| A-5h | decoder (4 layers), enc->dec projection, embedding/head incl. new `it` rows; encoder frozen | Is the pretrained encoder already good enough for Italian, i.e. is the gap only a text-side (tokenizer + LM) problem? Cheapest arm. |
| B4-5h | A + top 4 encoder layers (LR x0.3) | Does adapting the top of the acoustic model buy WER beyond A at the same data and steps, and at what extra cost per epoch / VRAM? |
| A/B at 20 h, 40 h | same | Data-scaling slope on nested data (only if 5 h shows a usable trend). |
| EN before/after (every run) | - | Forgetting: EN WER on 100 FLEURS en clips before and after; no replay data is used in these smoke runs. |
| A-5h-gain vs A-5h | same as A | Does random-gain augmentation make Italian WER level-independent (-3 / -20 / -40 dBFS peak) without costing WER at the native level? |

Per run, `runs/<tag>/metrics.json` holds: audio-hours per GPU-hour (whole run
and steady state), samples/s, peak allocated + reserved VRAM, equivalent-epoch
wall time, equivalent epochs done, total and training GPU-hours, cost at
`RATE_USD_H` (total, training, per epoch), loss curve, validation curve
(300-utt strided subsets of each eval set, every 200 steps), final WER/CER with
S/D/I on the full frozen eval, EN before/after. `runs/summary.jsonl` gets one
line per run. Defaults: AdamW lr 2e-4, cosine, 10 % warmup, bf16-mixed,
lhotse bucketing with 300 s of audio per batch, 10 equivalent epochs
(`EPOCHS`), grad clip 1.0. The validation subsets come from the eval sets
(smoke test; no checkpoint selection: the last step is reported).

## Level robustness (input gain)

Measured finding that motivates this (EOU 120M, whose preprocessor has
`normalize: NA`, i.e. no per-feature normalisation): it returns EMPTY
transcripts on low-level audio. On FLEURS EN test the empty clips have a median
peak of -45 dBFS against -35 dBFS for the transcribed ones, and there are 0
empties in the loudest quintile. Canary 180M must be checked the same way:

- every model load records `preprocessor.normalize` (and dither / log guard)
  in the JSON (`probe/zeroshot.json`, `runs/<tag>/metrics.json`,
  `evals/<tag>/eval.json`). `per_feature` normalisation cancels most of a pure
  gain change; what remains is the dither / log-floor regime at very low level
  and PCM16 quantisation. `NA` cancels nothing.
- `hours.json` and every manifest row carry the clip peak (`peak_dbfs`;
  p10/p50/p90 per set), so the eval level distribution is on record.
- **Level sweep**: `eval_it.py` (default `--levels -3,-20,-40`), the probe (EN
  clips on the untouched model) and every training run (IT val subsets + EN
  clips, before AND after training) decode with every clip re-levelled to a
  fixed peak and re-quantised to the PCM16 grid. WER per level shows whether
  quiet input breaks the model and whether fine-tuning changes that.
- **Gain augmentation** (`GAIN_AUG=1`, `--gain-aug 1`): random per-utterance
  gain uniform in -30..+6 dB with p = 0.8, applied to each training batch on
  the GPU (padding stays zero), clipped to [-1, 1] and re-quantised to PCM16.
  NeMo 3.0.0's lhotse loader has no plain gain/volume option (it offers noise
  mixing, speed perturbation, RIR, low-pass, codec compression and a
  clipping-with-gain transform), so this is a Lightning callback
  (`common/ftlib.make_gain_aug`) and its applied-dB statistics are logged in
  `metrics.json`. For Canary it is OFF by default and run as a recorded A/B
  (`A-5h` vs `A-5h-gain`, same seed and steps). It is meant to be ON by default
  for the EOU 120M arm (`normalize: NA`) when that trainer is added; the
  callback is shared for that purpose.

## Scoring

Open ASR Leaderboard normalisation: `MultilingualNormalizer(remove_diacritics=
False)` with `lang="it"` (Whisper basic multilingual normaliser + num2words)
for Italian, `EnglishTextNormalizer` for English, both from
huggingface/open_asr_leaderboard at a pinned commit. WER and CER are corpus
level (sum of S+D+I over sum of reference words/characters), per set and
pooled. Decoding: the checkpoint's own decoding config (beam size 1), prompt
`source_lang=target_lang`, `pnc=no`, fp32, clips shorter than 1 s padded to 1 s.

## Known uncertainties (not runnable on the laptop)

- NeMo 3.0.0 calls used and read from source but not executed: `save_tokenizers`,
  `change_vocabulary(DictConfig, "agg")` with a NEW language key (the script
  writes the new tokenizer config into `model.cfg` first, because
  `_setup_aggregate_tokenizer` writes back per-language entries),
  `decoding.decode_predictions_tensor`, `prompt.encode_dialog`,
  `setup_training_data` with a plain dict config, `optim_param_groups`.
- The decode loop bypasses `transcribe()` because NeMo's lhotse transcribe
  path tokenises the (empty) answer with the target language's sub-tokenizer
  and raises for a language the model has no sub-tokenizer for (the C0 probe).
- MLS parquet audio bytes are decoded with soundfile (ffmpeg fallback); the
  codec inside the HF parquet was not inspected.
- The Common Voice path (MDC SDK call, tarball member names) is untested.
- Canary 180M's `preprocessor.normalize` value was not read here (the `.nemo`
  is not on the laptop); the kit records it on the box. The default
  `fast-conformer_aed.yaml` uses `per_feature`.
- The gain callback assumes Lightning 2.x calls `on_train_batch_start` with
  the batch already on the device and passes the same object to
  `training_step` (true for the 2.2-2.4 range NeMo 3.0.0 pins); the logged
  applied-dB statistics show whether it ran.
