# finetune/: language adaptation of the small NeMo models Mynah serves

Research tooling to adapt the two lightweight models to a new language
(Italian first) on one rented Linux GPU, with every number reproducible from a
script and a `run.json`. `tools/` keeps conversion, benchmarks and evaluation of
packs; `finetune/` is where checkpoints are produced.

**This directory is the maintained copy.** The scripts were developed in
`.work/lightweight-asr/{ft,eou-ft,jobs}/` on 2026-10-09; those copies are frozen
(they are what that day's box ran) and are not edited any more. Old -> new
names are listed at the end.

| track | status | entry points |
|---|---|---|
| Canary 180M Flash -> Italian specialist | **validated** (2026-10-09, L40S) | `canary/` |
| Canary 180M Flash -> EN/DE/ES/FR + IT (20 % replay) | **validated** (2026-10-09, L40S) | `canary/` + `canary/replay.py` |
| parakeet_realtime_eou_120m-v1 -> Italian (+ `<EOU>`) | **EXPERIMENTAL, not working**: the combined language + EOU recipe collapses toward blank | `eou/` |

```
finetune/
  README.md             this file
  common/               ftlib.py      manifests, leaderboard normalisers, WER/CER with S/D/I, decode loop,
                                      level re-quantisation + level sweep, replay mixing, gain-aug callback
                        subsets.py    speaker-balanced, deterministic, NESTED subsets (5 h c 20 h c 40 h)
                        trim.py       energy trim to the voiced span (aligner stand-in)
                        runmeta.py    run.json writer, sha256 of files/dirs, versions/GPU probe, checkpoint records
                        hf_archive.py size-verified, resumable upload of runs to a private HF repo
  canary/               README.md (detailed kit doc), env.sh, setup.sh, prepare_it.{sh,py}, replay.py,
                        tokenizer_it.{sh,py}, probe_it.{sh,py}, train_it.{sh,py}, eval_it.py, report.py, run_all.sh
  eou/                  README.md (NeMo recipe facts), plain_asr.py (stage 1), train_eou.py (stage 2 machinery),
                        tokenizer_eou.py, rnnt_diag.py, export_to_mynah.sh,
                        diag/{blank_margin,batch_probe,encoder_diff,tok_coverage}.py
  jobs/                 gpu_lock.sh (lock + env pattern), ft2.sh, ft3.sh, eou1.sh, micro_overfit.sh
  tests/test_finetune.py  CPU self-test (part of `make check`)
```

## Validated: Canary 180M Flash -> Italian

Recipe (defaults of `canary/train_it.py`, unchanged from the runs below):
Italian SentencePiece (1024) appended LAST to the aggregate tokenizer (ids
0..5247 keep their meaning, old rows copied back and verified bit-identical,
new rows seeded from `es`), arm **B4** = decoder + projection + embedding/head
+ top 4 encoder layers, AdamW lr **2e-4**, encoder lr **x0.3**, cosine with 10 %
warmup, **bf16-mixed**, lhotse buckets of **300 s** of audio per batch, grad clip
1.0, seed 1234. Data: MLS-it train, nested speaker-balanced subsets (seed
20261009, 65 speakers, 60 min/speaker cap); frozen eval = all of FLEURS-it test
(865) + MLS-it test (1262), Open ASR Leaderboard multilingual normaliser.

| run (tag) | data | steps | FLEURS-it WER / CER | MLS-it WER / CER | original languages (FLEURS WER) | GPU-h incl. evals |
|---|---|---|---|---|---|---|
| zero-shot `es` prompt (floor) | - | - | 87.67 / 30.00 | 82.01 / 28.13 | EN 6.86 | - |
| B4-5h-e52 | 5 h x 52 | 3120 | 48.77 / 15.28 | 34.40 / 8.66 | EN 27.75 | 0.333 |
| B4-20h-e13 | 20 h x 13 | 3120 | 43.58 / 13.23 | 30.34 / 7.25 | EN 52.33 | 0.326 |
| **B4-40h-e7** (IT specialist) | 40 h x 7 | 3360 | **42.06 / 12.75** | **29.42 / 7.04** | EN 6.86 -> 54.53 | 0.339 |
| **B4-20h-e13-rp20** (multilingual) | 20 h x 13 + 20 % replay | 3900 | **44.17 / 13.44** | **30.27 / 7.17** | EN 7.96, DE 9.80, ES 8.33, FR 10.41 (base 6.86 / 8.86 / 5.63 / 10.25) | 0.393 |

Training throughput ~2,700-2,900 audio-h per GPU-h, peak 8.33 GB allocated
(fits a 24 GB L4; the L4 rerun is still owed). Reading: the acoustic side
transfers (in-domain CER 7 %), WER is limited by the decoder still learning
Italian; 5 -> 20 h buys ~5 WER points at equal compute, 20 -> 40 h ~1.5 (7
epochs at 40 h is probably undertrained). 20 % replay of FLEURS en/de/es/fr
(2 h per language) keeps Italian and holds the four base languages within
+0.2 .. +2.7 WER: Canary 180M can be EXTENDED, not only specialised.

Artifacts: `gabrione/mynah-asr-canary-it-experiments` on the Hugging Face Hub,
a **private archive** (not a release), revision
`f6d4a3076a7884fef62465ecc10c9ba22aea0f95`: `runs/B4-40h-e7/final.nemo`,
`runs/B4-20h-e13-rp20/{final.nemo,last.ckpt}`, metrics/evals/logs of every run,
the tokenizers and the recipe as it ran. Uploads were size-verified.

### Reproduce (box: Ubuntu, NVIDIA driver 575 / CUDA <= 12.9, NeMo 3.0.0 venv)

```bash
scp -r finetune root@BOX:/root/finetune          # the whole directory: scripts import ../common
export FT_ROOT=/root/ft VENV=/root/nemo-venv       # FT_ROOT is REQUIRED by every step (data, models, runs, logs)
bash finetune/canary/setup.sh                     # torch cu128 wheel, deps, normaliser, checkpoint, bf16 check
bash finetune/canary/prepare_it.sh --dry-run      # plan + hours only
bash finetune/canary/prepare_it.sh                # MLS-it + FLEURS-it/en -> wav + manifests (resumable)
bash finetune/canary/tokenizer_it.sh              # it SPE + surgery + verification -> canary-180m-flash-it.nemo
bash finetune/canary/probe_it.sh                  # C0 zero-shot floors
# IT specialist (B4-40h-e7):
SUBSET=40h EPOCHS=7 ARMS=B TAG_SUFFIX=-e7 bash finetune/canary/train_it.sh
# multilingual (B4-20h-e13-rp20): replay pool first (network/CPU), then train
$VENV/bin/python finetune/canary/replay.py --hours-per-lang 2
SUBSET=20h EPOCHS=13 ARMS=B REPLAY_RATIO=0.2 TAG_SUFFIX=-e13-rp20 SAVE_CKPT=1 bash finetune/canary/train_it.sh
# or the original jobs (GPU lock + timeouts): jobs/ft2.sh (ARM_LONG=B), jobs/ft3.sh
$VENV/bin/python finetune/canary/eval_it.py --nemo $FT_ROOT/runs/B4-40h-e7/final.nemo --en
$VENV/bin/python finetune/canary/report.py        # markdown summary of everything measured
```

`python3 finetune/canary/train_it.py --subset 5h --dry-run` (with `FT_ROOT`
pointing at a prepared tree) prints the resolved configuration, steps, LR per
group and replay mix without importing NeMo. Details of every step, data
construction, tokenizer surgery, level robustness and scoring:
`canary/README.md`.

## Experimental: EOU 120M -> Italian (NOT working)

Goal: an Italian streaming RNNT with integrated `<EOU>` on the stock
cache-aware encoder ([70,1], causal), still a plain RNNT pack Mynah runs.
**No working model exists.** On 2026-10-09 (MLS-it 5 h, new Italian SPE 1024 +
`<EOU>`/`<EOB>`, decoder/joint lr 1e-3, 90 % of samples padded with 3-6 s of
zeros + 10 % plain, FastEmit 0.03, gain aug) all four arms ended at WER ~100
with empty greedy output:

| arm | encoder | decoder/joint | result |
|---|---|---|---|
| eou-it-5h-e52 | trainable (lr x0.3) | fresh | 100 / 100, 0 `<EOU>` |
| eou-it-5h-e52-warm | trainable | stock | in-loop 100 / 100 at 1000 and 1500 (stopped) |
| eou-it-5h-e26-frz | frozen | fresh | 99.57 / 99.69 |
| eou-it-5h-e26-frz-warm | frozen | stock | 100 / 100; text tokens gone by step 500, `<EOU>` decays to ~0 |

Diagnostics (`eou/diag/`): blank beats every token on every frame; beam-4
returns the same sentence for every input (an audio-independent text prior);
the loss converges (text+`<EOU>` 44.9 vs text only 192.9) toward a
non-decodable solution; a real batch is sane; the encoder barely moved (median
0.9 % weight change, output cosine 0.911, but temporal std 0.169 -> 0.040) and
the frozen-encoder arms collapse too. Starting from the working stock model
with a frozen encoder, the loop removes text first and `<EOU>` next: the
**training recipe** drives the collapse, not the language transfer.

This configuration is therefore never a default: `eou/train_eou.py` and
`jobs/eou1.sh` require `--lr`/`LR` and `--eou-weight`/`EOU_WEIGHT` explicitly
and print an EXPERIMENTAL banner. Exact reproduction of the cold arm:
`LR=1e-3 EOU_WEIGHT=0.9 SUBSET=5h EPOCHS=52 bash finetune/jobs/eou1.sh`
(`WARM_DEC_JOINT=1` / `FREEZE=<layers>` for the other arms).

Plan (two stages, one variable at a time):
1. **Stage 1, plain Italian ASR with the stock tokenizer** (`eou/plain_asr.py`):
   stock model and decoder/joint, de-accented targets, FastEmit 0, no padding,
   no gain, frozen encoder. First the micro-overfit
   (`--micro-overfit` = 32 utterances, 1000 steps, evals at 100/250/500/750/1000,
   lr 1e-4; `jobs/micro_overfit.sh`): it must memorise them with
   audio-dependent greedy output before anything else runs. Then 5 h with
   `--val`, then unfreezing the top encoder blocks at 1e-5..3e-5.
2. **Stage 2, gentle EOU curriculum** from the stage-1 checkpoint: add ONE
   variable at a time (`<EOU>` targets -> padding at a low share -> FastEmit
   -> gain), measuring at every step: train loss, WER/CER, empty rate,
   non-blank frame %, blank margin, `<EOU>` rate, audio dependence. Only then
   incomplete-utterance negatives and backchannels. `train_eou.py` holds the
   NeMo EOU dataset machinery for this stage; starting it from a stage-1
   checkpoint with the stock tokenizer is NOT implemented yet.

Endpoint quality of any resulting pack: `eou/export_to_mynah.sh` then
`tools/eval/eou_metrics.py` (latency, premature, miss; `--mode composite`).
Per-eval outputs of `plain_asr.py`: `<run>/metrics.json` (log per eval step),
`<run>/evals/step_<n>.jsonl` (greedy + beam hypothesis and blank margin per clip).

## Outputs of a run

| file | what |
|---|---|
| `runs/<tag>/run.json` | provenance, written before step 1 (`status: started`) and completed at the end (`done`): base model id/revision/path/sha256, git SHA + dirty flag, tokenizer path + sha256, dataset/subset/manifest/hours/utterances, seed, optimizer, LR per param group, scheduler, FastEmit, augmentation, EOU/plain mix, freeze policy, max steps/epochs, GPU/driver, Python/torch/CUDA/NeMo/Lightning/numpy/numba versions, the resolved CLI, outputs |
| `runs/<tag>/metrics.json` | measured results: economics (audio-h per GPU-h, samples/s, peak VRAM, GPU-h, cost at `RATE_USD_H`), loss and validation curves, final WER/CER with S/D/I, forgetting before/after, level sweep |
| `runs/<tag>/final_eval/hyp_*.jsonl` | per-utterance reference / hypothesis / normalised forms / errors |
| `runs/summary.jsonl` | one line per finished run |
| `evals/<tag>/eval.json` | `canary/eval_it.py` on any `.nemo` |

## Checkpoints and resume

- Canary: `SAVE_CKPT=1` writes `runs/<tag>/last.ckpt` (full Lightning state:
  optimizer, scheduler, AMP scaler, global step; 1.43 GB) at the END of the
  run. Continue it with `RESUME_CKPT=<last.ckpt>` / `--resume-ckpt`, which
  calls `trainer.fit(model, ckpt_path=last.ckpt)`; the run only proceeds if
  `--max-steps` is above the saved step (the cosine schedule is already at
  its floor). A NEW stage needs only `--init <final.nemo>`. There is no
  periodic mid-run checkpoint: a killed run restarts from step 0 (~10 min).
- EOU stage 1 (`plain_asr.py`): `last.ckpt` = torch state {model, opt,
  sched, step, args} at every eval step, `best.ckpt` + `final.nemo` on the
  best val WER; `--resume <ckpt> --max-steps <larger>` continues.
- EOU stage 2 (`train_eou.py --save-ckpt 1`): Lightning `last.ckpt` of the
  `EncDecRNNTBPEEOUModel`; `final.nemo` is saved as plain `EncDecRNNTBPEModel`.

## Environment notes (measured 2026-10-09)

- NeMo 3.0.0 in a `uv venv -p 3.12` + `uv pip install nemo_toolkit[asr]`.
- Driver 575 (CUDA <= 12.9): the default PyPI torch wheel targets a newer CUDA
  runtime than the driver supports; `canary/setup.sh` reinstalls the SAME
  torch version from the **cu128** index (fallback cu126) and checks bf16.
- NeMo's RNNT loss (EOU kit) runs on numba CUDA: numba 0.68 needs the separate
  **numba-cuda** package and numba-cuda 0.30.4 needs **numpy < 2.4**
  (`WITH_RNNT=1 bash finetune/canary/setup.sh`).
- EOU tokenizer config: the stock cfg points `model_path`/`vocab_path` at
  `nemo:` artifacts of the OLD tokenizer; `train_eou.py` rewrites all three
  paths (otherwise `register_artifact` fails on a None `model_path`).
- One GPU job at a time: `jobs/gpu_lock.sh` (flock on `/root/gpu.lock`, the
  same lock benchmarks take); every GPU step carries its own `timeout`.
- No bare command starts a long run: `FT_ROOT` must be exported, `SUBSET` /
  `--subset` and EOU `--lr`/`--eou-weight` are required; `--dry-run` exists
  on `train_it.py`, `train_eou.py` and `plain_asr.py`.

## Tests

`python3 finetune/tests/test_finetune.py` (also run by `make check`): CPU only,
no downloads: manifest rows and round trip, voiced-span trim, subset nesting
and determinism, replay mixing ratio, run.json serialisation, the
`<EOU>`/`<EOB>`/blank id contract, a tiny SentencePiece round trip with
`<EOU>`/`<EOB>` appended (skipped with a message without `sentencepiece`),
and the plain-stage CLI presets. Also cheap: `python3
finetune/canary/prepare_it.py --synthetic` and `python3
finetune/eou/tokenizer_eou.py --dry-run --out <tmp>`.

## Old -> new names

| `.work/lightweight-asr/` (frozen) | `finetune/` |
|---|---|
| `ft/ftlib.py` | `common/ftlib.py` (+ `common/subsets.py` for the subset functions) |
| `ft/common.sh` | `canary/env.sh` (FT_ROOT now required) |
| `ft/data_it.{py,sh}`, `ft/data_replay.py` | `canary/prepare_it.{py,sh}`, `canary/replay.py` |
| `ft/train_canary_it.{py,sh}`, `ft/probe_zeroshot.py`, `ft/probe.sh` | `canary/train_it.{py,sh}`, `canary/probe_it.py`, `canary/probe_it.sh` |
| `ft/{setup.sh,tokenizer_it.*,eval_it.py,report.py,run_all.sh}` | `canary/` (same names) |
| `eou-ft/plain_it.py` | `eou/plain_asr.py` (+ micro-overfit, beam, diagnostics, resume) |
| `eou-ft/train_eou_it.py`, `eou-ft/tokenizer_eou_it.py` | `eou/train_eou.py`, `eou/tokenizer_eou.py` |
| `eou-ft/{blankdiag,batchprobe,tokcov}.py` | `eou/diag/{blank_margin,batch_probe,tok_coverage}.py` |
| (ad-hoc encoder comparison, not in git) | `eou/diag/encoder_diff.py` |
| `jobs/{ft2,ft3,eou1}.sh`, `jobs/arch.py` | `jobs/`, `common/hf_archive.py` |
