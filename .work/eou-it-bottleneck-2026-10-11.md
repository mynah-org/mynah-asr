# EOU-IT: what limits the 120M Italian specialist (2026-10-11)

Question of the day: the EOU 120M learns Italian very fast at first (Friday: ~2 h to a usable
model) and then sits at a 45-50 macro WER regime (B2 49.08; Nemotron 0.6B on the same clips
15.75). Which of these limits it: optimisation/exposure, English output representation,
predictor/joint adaptation, encoder acoustics, our minimal loop vs the NVIDIA recipe, data,
capacity? Labels: OBSERVED / NVIDIA-EVIDENCE / HYPOTHESIS / PROPOSED.

Box: L40S (Japan, 30-CPU quota), resumed from empty with `finetune/jobs/prov.sh` +
`finetune/jobs/resume_it.sh` (archived teacher scores, no teacher re-run).

## S15-14 error analysis (no GPU) — done

Tool: `finetune/eou/diag/error_analysis.py` on yesterday's paired decodes (P40, A0, A2, Nemotron;
same 730 val clips), lexicon = every training transcript (106k words). Output on the box:
`/root/ft/s14/err.json`.

| model/set | WER | CER | WER/CER | S | D | I | S near (1 ch) % | S far (>=3) % | non-words % of hyp |
|---|---|---|---|---|---|---|---|---|---|
| A2/mls | 45.76 | 14.33 | 3.19 | 32.6 | 9.3 | 3.8 | 21.1 | 57.9 | 15.7 |
| Nemotron/mls | 18.77 | 4.94 | 3.80 | 13.2 | 3.6 | 2.0 | 37.3 | 41.1 | 7.0 |
| A2/fleurs | 45.39 | 15.02 | 3.02 | 32.2 | 6.2 | 7.1 | 20.8 | 60.7 | 15.3 |
| Nemotron/fleurs | 6.28 | 3.52 | 1.79 | 3.3 | 0.8 | 2.2 | 36.8 | 36.8 | 3.1 |
| A2/cv | 56.93 | 22.42 | 2.54 | 39.8 | 12.9 | 4.3 | 18.6 | 64.3 | 19.3 |
| Nemotron/cv | 9.40 | 2.61 | 3.60 | 7.1 | 1.6 | 0.6 | 39.7 | 29.4 | 5.8 |
| A2/vp | 59.11 | 34.84 | 1.70 | 30.5 | 24.5 | 4.1 | 14.4 | 70.1 | 13.6 |
| Nemotron/vp | 28.53 | 22.83 | 1.25 | 7.7 | 16.7 | 4.1 | 17.2 | 60.1 | 2.6 |

- OBSERVED: the WER/CER ratio is not diagnostic (Nemotron/mls 3.80 > student 3.19; yesterday's
  "1.8" was Nemotron on FLEURS). Correction of the 2026-10-10 reading.
- OBSERVED: student substitutions are mostly far (58-70 % >= 3 chars) vs Nemotron 29-41 %;
  near-misses 14-21 % vs 37-40 %. Segmentation pairs 2-2.8 / 100 words; CER without spaces is
  only ~1 point lower: spacing / word boundaries explain little.
- OBSERVED: 14-19 % of the student's hypothesis words exist in no training transcript (Nemotron
  3-7 %): it spells sounds. Top confusions are context homophones (ha/a, hanno/anno, nei/ne,
  la/alla). Suggestive of a weak internal LM / predictor; NOT proof.
- OBSERVED (`diag/tok_coverage.py`): the stock SPE spends 2.95 tokens per Italian word on
  FLEURS+CV eval text ("il" -> `▁i`+`l`), accented vowels -> `<unk>` (hence de-accented targets).

## Today's GPU queue (launched 10:59, `/root/res/queue.sh`)

1. S15-15b `finetune/jobs/tok_it.sh`: Italian SPE (A2 transcripts + Granary teacher text, accented,
   same size, <EOU>/<EOB> last) warm-started into B2 (copy shared rows, mean of the stock split
   for new pieces, blank/EOU/EOB unchanged); step-0 eval stock vs warm-started; then B2 with
   `--tokenizer` + evals at 100/250 + collapse gate (macro > 80 at 1000 -> abort).
   Caveat (review): this changes vocabulary AND accented targets together; `DEACCENT=1
   tok_it.sh` (`it-tokna-b2`) is the vocabulary-only arm, prepared, not queued.
2. S15-15a C1 at 12k steps (`mix_arm.sh it-c1-12k`, STEPS=12000). NOTE: a fresh run with the
   cosine over 12k, NOT a continuation of C1@4k: its 4k/8k points sit at higher LR than C1@4k;
   the comparable numbers are the end points (C1@4k 49.30 vs C1-12k@12k) and the slope.

## Workstreams (parallel agents, no GPU; outputs on the box under /root/ft/analysis/{A..F})

A NVIDIA recipe audit, B training-dynamics forensics, C tokenizer/output representation,
D RNNT predictor/joint adaptation, E EOU recipe archaeology, F data/effective exposure.
Synthesis below when they return.

## S15-15b result: Italian SPE on the B2 recipe (it-tok-b2) — negative at this budget

Fertility (FLEURS eval): stock 2.94 tokens/word (2 % <unk>) -> Italian SPE 2.10 (0 <unk>); U/T p50
-25..-32 % on every set (agent C, `/root/ft/analysis/C/tokenizer_audit.md`).
Step 0 (same weights, before training): stock 92.38 / 57.62 macro, warm-started 95.84 / 70.28
(264 pieces copied, 762 mean-initialised); the surgery is correct (identity control bit-identical,
agent C): vocabulary damage only.

| step | tok-B2 macro WER / CER | B2 | gap |
|---|---|---|---|
| 100 / 250 | 95.4 / 81.8 · 87.7 / 57.6 | - | - |
| 500 | 79.43 / 43.36 | 64.97 / 29.53 | +14.5 |
| 1000 | 70.16 / 34.06 | 56.90 / 22.93 | +13.3 |
| 2000 | 60.52 / 26.85 | 52.43 / 21.70 | +8.1 |
| 3000 | 56.32 / 22.77 | 49.81 / 19.03 | +6.5 |
| 4000 | 56.06 / 22.59 (mls 49.70, fleurs 53.07, cv 61.03, vp 60.43; 10 empty) | 49.08 / 18.22 | +7.0 |

OBSERVED: under B2's budget the Italian SPE costs 7 macro; the gap closes until ~3000 and freezes
where the cosine runs out of LR, like every arm. NOT shown: that the tokenizer is a bad lever in a
recipe that can re-learn 762 pieces. Re-test only on a recipe that wins E1/E2 below. Two init/
normaliser defects found by agent C and fixed after the run (4be5367): word-internal pieces were
mean-initialised from word-initial stock rows; norm_text glued "po' confuso".

## Synthesis of workstreams A-F (2026-10-11 midday)

Key OBSERVED facts:
- B2 weight change vs stock (A, D): encoder blocks 0-8 0.01-0.04 %, 9-12 0.12-0.17 %, 13-16
  0.36-0.56 %; decoder LSTM 5-6 %, joint.pred 37 %. The lower encoder never adapted: blocks open
  at 50 % of the run with an lr that peaks at ~2.5e-6 (integrated LR ~100x below NVIDIA's
  fine-tune default, B).
- Underfitting, not overfitting: train8 WER ~ val WER in every arm (A, B, F); the flat tail starts
  at ~step 2500 in every run (LR multiplier < 0.3; 91 % of the integrated LR spent by 2500, B).
- Re-warms keep improving (every stage-2 continuation lowered plain WER 1.3-2.5, B).
- Same B2 encoder output (D): greedy 49.25, beam-8 no LM 44.84, 8-best oracle 41.32: a free
  4.4 from beam search (the Mynah engine decodes greedily); for most error words the beam has
  no correct hypothesis (acoustic evidence missing). Token n-gram fusion from the training text:
  no gain.
- The predictor learned Italian (internal-LM perplexity 3.9-8.8k -> 34-89 per token, D); a
  linear CTC head on frozen encoder frames: CER stock 34.8, B2 28.9 (D).
- C1/C2 replaced exposure (Granary took 50 % of the draws; MLS passes 2.33 -> 1.16); at equal MLS
  hours C1 = B2 on MLS (F). Granary at equal own-domain exposure: ~+3 FLEURS/CV, ~+7 VP.
- Unfreezing the whole encoder at step 2000 is a level shift (-2.3..-2.8 macro vs A2, same LR and
  batches), not a new slope (B). Eval noise floor ~0.5 macro / 0.8 per domain (A2 vs B2 before
  their recipes diverge).
- Stage 2 VP regression lives in the stage-2 decoder/joint, not the top-4 encoder (E: S2 encoder
  + B2 dec/joint VP 54.96 / 7 empty; B2 encoder + S2 dec/joint 57.99 / 16 empty).

| # | hypothesis | evidence for | evidence against | discriminating experiment | GPU | info gain |
|---|---|---|---|---|---|---|
| 1 | encoder acoustic adaptation starved by our recipe (LR/freeze) | weight deltas; integrated LR; unfreeze level shift; CTC head still improvable; oracle 41 = acoustic | the encoder did move enough to gain 6 CER in the CTC probe | E1a vs E1b from B2; E2 whole encoder 1e-4 from stock | 13 + 22 min | highest |
| 2 | undertraining / schedule end (optimization exposure) | flat tail = LR end in every run; underfitting; re-warms improve | diminishing returns per step | C1-12k (running); E1b (re-warm at constant-ish LR) | running | high |
| 3 | our loop vs NVIDIA recipe | lr split (dec 3x, enc 3-20x below NVIDIA's single 1e-4), 40x fewer updates, global batch 16 vs 512-2048 | SpecAugment, AdamW, wd, clip, dropout match | covered by #1; batch/accum later | - | covered |
| 4 | English output representation | 2.94 tokens/word, morphology unrepresented | tok-B2 7 worse at fixed budget; predictor learned Italian | re-test SPE only on the winning recipe | later | medium |
| 5 | predictor/joint capacity or LM | homophones, non-words | ILM learned Italian; fusion no gain; oracle close to beam | E1a (dec+joint only) | in E1 | medium |
| 6 | data quality/diversity | VP deletions; Granary helps FLEURS/VP | train8 ~ val: data not binding at 4k | none now | - | low now |
| 7 | model capacity (120M) | Nemotron 0.6B 15.75 | cannot be judged before #1/#2 are fixed | none now | - | low now |

Next GPU (queued 12:1x after C1-12k, `finetune/jobs/enc_budget.sh`): E1a it-e1a-decjoint, E1b
it-e1b-all1e4 (from B2, 1000 steps each), E2 it-e2-full1e4 (stock, whole encoder 1e-4, 4000
steps, batch-paired with B2), each with a CPU greedy/beam-8/oracle probe.
Deployment note (not training): beam-8 alone is worth ~4.4 macro on B2 — a Mynah beam decoder is
a candidate engine item.

## S15-15a result: C1 with the cosine over 12k steps — CENSORED at 9k (CUDA OOM)

`it-c1-12k` (C1 mix and policy, STEPS=12000) died of CUDA OOM after step 9000 (5.3 GiB RNNT
gradient for a batch of long Granary clips, 11.4 GiB reserved but unallocated: fragmentation).
No 12k endpoint is claimed. Fix for every later job: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
(gpu_lock.sh); two_stage.py now logs peak reserved memory and the longest clip.

| step | macro WER / CER | mls | fleurs | cv | vp | train8 |
|---|---|---|---|---|---|---|
| 2000 | 52.73 / 20.33 | 49.48 | 46.55 | 58.82 | 56.08 | 54.4 |
| 4000 | 46.60 / 17.73 | 43.20 | 40.63 | 52.10 | 50.44 | 50.9 |
| 6000 | 43.40 / 16.42 | 41.15 | 36.14 | 48.74 | 47.58 | 48.8 |
| 8000 | 41.64 / 15.59 | 39.74 | 34.48 | 46.69 | 45.67 | 45.6 |
| 9000 | **41.15 / 15.31** | 39.12 | 33.75 | 45.90 | 45.83 | 45.3 |

OBSERVED: C1-12k@4k 46.60 vs C1 (cosine over 4k)@4k 49.30 on the same first 4000 batches: the
shorter schedule throttled learning (LR available, not just more steps); 4k -> 9k adds another
-5.45. Every domain falls; train8 ~ val (still underfitting). This is strong evidence of an
undertraining / LR-budget bottleneck; it does not by itself say which module needs the budget
(E1a/E1b decide that).

## E1a vs E1b (from B2, A2 mix, 1000 steps each, `finetune/jobs/enc_budget.sh`)

| step | E1a: encoder frozen, dec+joint 1e-4 | E1b: whole model 1e-4 |
|---|---|---|
| 0 (= B2) | 49.08 / 18.22 | 49.08 / 18.22 |
| 250 | 49.39 / 19.76 | 48.07 / 18.54 |
| 500 | 49.55 / 19.74 | 45.95 / 17.57 |
| 750 | 49.41 / 19.64 | 44.99 / 16.82 |
| 1000 | 49.27 / 19.56 (mls 43.36, fleurs 42.85, cv 54.52, vp 56.38; 20 empty) | 44.97 / 16.87 (mls 40.27, fleurs 38.57, cv 49.26, vp 51.76; 12 empty) |

CPU probe (`diag/lm_fusion_probe.py`, 200 clips/set, alpha 0): greedy / beam-8 / 8-best oracle
— B2 49.25 / 44.84 / 41.32; E1a 49.33 / 44.71 / 40.29; E1b 44.89 / 40.25 / 35.95.

OBSERVED: with the same LR, the same batches and the same start, training only the output side
is flat within noise (CER slightly worse); unfreezing the encoder gives -4.1 WER / -1.4 CER in
1000 steps on every domain and moves the N-best oracle down 5.4 (new acoustic evidence, not
search). The encoder is causally implicated. Caveat: E1b is a re-warm of B2 (5000 steps total);
E2 (stock -> whole encoder 1e-4 from step 0) is the recipe test.

## E2: stock -> whole encoder 1e-4 from step 0 (decoder+joint 3e-4), A2 mix, 4000 steps — NEW BASELINE

Batch-paired with B2 (same seed, same sampler draws, same val sets); 25.1 min, peak 22.4 GB
allocated / 24.3 reserved, longest clip 20 s.

| step | E2 macro WER / CER | mls | fleurs | cv | vp | empty | B2 macro |
|---|---|---|---|---|---|---|---|
| 100 | 76.24 / 30.79 | 71.93 | 72.79 | 77.42 | 82.80 | 24 | - |
| 250 | 62.79 / 24.08 | 58.20 | 56.85 | 67.54 | 68.57 | 18 | - |
| 500 | 54.19 / 18.72 | 50.16 | 49.32 | 56.25 | 61.02 | 8 | 64.97 |
| 1000 | 46.80 / 15.69 | 43.27 | 42.14 | 49.89 | 51.89 | 3 | 56.90 |
| 1500 | 45.52 / 17.63 | 40.33 | 37.53 | 49.84 | 54.37 | 20 | 54.48 |
| 2000 | 42.12 / 15.54 | 38.05 | 34.63 | 45.22 | 50.58 | 6 | 52.43 |
| 3000 | 40.10 / 14.93 | 35.92 | 33.71 | 42.23 | 48.53 | 9 | 49.81 |
| 4000 | **39.65 / 14.53** | 36.28 | 33.28 | 42.49 | 46.56 | 8 | 49.08 |

CPU probe at 4000: greedy 39.66 / beam-8 **35.09** / 8-best oracle 31.02
(beam per set: mls 32.52, fleurs 29.10, cv 37.03, vp 41.71).

OBSERVED: same data, same batches, same 4000 steps as B2: -9.43 macro WER, -3.69 CER, every
domain (mls -7.6, fleurs -10.6, cv -11.3, vp -8.3). Better than C1-12k@9k (41.15) with 2.25x
fewer steps and no Granary. E2@1000 (46.80) already beats B2@4000. The freeze curriculum was the
bottleneck; adapting the whole 120M from step 0 is the recipe. Step 1500 shows a transient (VP
54.4, 20 empty) that recovers by 2000 — watch stability at higher LR.

## Encoder-LR sweep on E2's schedule and batches (`finetune/jobs/lr_sweep.sh`, stop at 1000)

| enc lr | @100 | @250 | @500 | @1000 | vp @500 / @1000 | empty @500 / @1000 | beam-8 / oracle @1000 |
|---|---|---|---|---|---|---|---|
| 3e-5 | 79.88 / 36.50 | 65.87 / 25.74 | 58.52 / 20.95 | 51.89 / 17.77 | 64.35 / 57.33 | 9 / 4 | 47.49 / 43.53 |
| 1e-4 (E2) | 76.24 / 30.79 | 62.79 / 24.08 | 54.19 / 18.72 | 46.80 / 15.69 | 61.02 / 51.89 | 8 / 3 | - |
| 2e-4 | 73.82 / 30.13 | 61.13 / 22.73 | 52.92 / 19.50 | 43.90 / 15.16 | 63.36 / 49.62 | 15 / 4 | 40.18 / 36.40 |

OBSERVED: more encoder LR = faster curve at every eval; 2e-4 leads E2 by 2.9 WER at 1000 on
every domain, with a wobble at 500 (CER +0.8, VP +2.3, 15 empty) gone by 1000. Its WER, CER, VP
and empty curves are monotone 100 -> 1000. Decision: ONE full 4000-step run of 2e-4
(`it-e2-enc2e-4-s4000`, launched 14:05) with extra evals at 750/1250/1750/2250 to see LR-peak
instabilities; no other arms.

## E3 = E2 recipe with encoder LR 2e-4, full 4000 steps (`it-e2-enc2e-4-s4000`) — NEW BEST

`FT_ROOT=/root/ft STOP=4000 LRS=2e-4 KEEP_CKPT=1 EVALS=100,250,750,1250,1750,2250 bash finetune/jobs/lr_sweep.sh`
(first launch died of ENOSPC at start; disk gate + optimizer-state cleanup added). 30 min GPU.

| step | macro WER / CER | mls | fleurs | cv | vp | empty | E2 |
|---|---|---|---|---|---|---|---|
| 1000 | 43.90 / 15.16 | 40.64 | 38.24 | 47.11 | 49.62 | 4 | 46.80 |
| 1250 | 43.14 / 15.78 | 39.28 | 36.27 | 46.06 | 50.97 | 14 | - |
| 1500 | 41.99 / 15.90 | 37.44 | 34.07 | 45.48 | 50.97 | 12 | 45.52 |
| 1750 | 39.81 / 14.15 | 36.00 | 33.94 | 43.33 | 45.96 | 5 | - |
| 2000 | 39.15 / 13.79 | 35.02 | 33.19 | 42.12 | 46.26 | 7 | 42.12 |
| 3000 | 36.80 / 13.47 | 32.38 | 30.69 | 38.76 | 45.37 | 8 | 40.10 |
| 4000 | **36.52 / 13.53** | 32.32 | 29.64 | 39.34 | 44.78 | 8 | 39.65 |

CPU probe at 4000: greedy 36.52 / beam-8 **33.09** (mls 30.14, fleurs 27.25, cv 35.08, vp 39.87) /
8-best oracle 28.52. Same data and batches as B2 (49.08): -12.56 greedy. The LR-peak wobble
(1250-1500: VP 51, 12-14 empty) recovers by 1750; the tail flattens with the cosine (-0.28 from
3000 to 4000), as every run.

## Mynah RNNT beam search (branch `feat/rnnt-beam`, e217c2d, local, not pushed)

Port of NeMo 3.0 malsd (no fusion, score_norm), prefix trie with shared LSTM states, streaming
with a commit lag (default 640 ms), width 1 = the unchanged greedy code. Gates (re-run by the main
session: `make test` rc 0, test_beam OK; parity file on the box `/root/ft/analysis/beam/`):
greedy identity 0/60 differences; beam-1 == greedy 240/240; beam-8 tokens == NeMo 240/240 and
8-best sets equal 240/240 (B2 and E1b, same encoder outputs). Real Mynah streaming beam-8 (640 ms
lag): E1b 45.23 -> 38.07, B2 49.84 -> 42.91 (streaming with lag beats offline beam, mostly on VP:
interpretation, not proven). Cost: 4.3 -> ~21 ms CPU per audio second (~2 % of a core per
real-time stream), ~355 KiB per stream, +640 ms text latency without partials. Not wired: CUDA
engine, server option, partial results; <EOU> path only checked on the stock English model.

## E3G: E3 recipe on the C1 mix (A2 at half weights + Granary agree<=15, max-dur 30), 4000 steps

Command: `/root/res/e3g.sh` = `MIX="<A2 x0.5>,yg_agree<=15.json:0.5" POL="--freeze-enc 0 --enc-lr 2e-4 --lr 3e-4 --bs 16"
SMOKE=1 mix_arm.sh it-e3g-yg15 --max-dur 30 --eval-steps 100,250,750,1250,1750,2250`. 36.7 min, peak 36.5 GB
allocated / 41.9 reserved (E3: 21.6 / 22.7), longest clip 30 s.

| @4000 | E3 | E3G | E3G - E3 |
|---|---|---|---|
| mls | 32.32 / 7.9 | 34.95 / 8.9 | +2.63 |
| fleurs | 29.64 / 8.9 | 29.08 / 8.8 | -0.56 |
| cv | 39.34 / 11.6 | 38.76 / 11.1 | -0.58 |
| vp | 44.78 / 25.7 | 43.39 / 23.7 | -1.39 |
| empty | 8 | 3 | -5 |
| macro | 36.52 / 13.53 | 36.55 / 13.38 | +0.03 |

Probe (best checkpoint; E3G best = step 3500 36.49): greedy / beam-8 / oracle — E3 36.52 / 33.09 /
28.52; E3G 36.49 / 33.05 / 28.84.

OBSERVED: with the encoder adapted, Granary (replacing half of A2's draws) does not move the macro;
it trades MLS (-50 % exposure, +2.6) for VP/CV/FLEURS (-0.6..-1.4, the CV/FLEURS part within the
~0.8 per-domain noise) and fewer empties — the C1 pattern again. Decision: E3 is the parent of
EOU-v3 (same accuracy, 30 vs 37 min, 22.7 vs 41.9 GB). Granary, if used later, should ADD
exposure (more steps / extra weight), not replace A2.

## Stage 2 on E3 (EOU-v3 candidates; `/root/res/s2e3.sh`, `/root/res/s2djlr.sh` -> `finetune/jobs/s2_mix.sh`)

All: v2 stage-2 recipe (text+<EOU>, 50 % 1-3 s trailing quiet, top-4 encoder 3e-5, 1500 steps, FastEmit
0), init `plain-it-e2-enc2e-4-s4000/final.nemo` (E3 plain: 36.52, vp 44.78 / 8 empty).

| arm | change | best macro (step) | @1500 macro / EOU / vp WER / vp empty / all empty |
|---|---|---|---|
| it-s2-e3 (control) | none | 37.10 (1250) | 37.41 / 96.5 % / 48.34 / 16 / 18 |
| it-s2-e3-pre | + leading quiet p 0.9, U(0,4) s | - | 37.98 / 95.3 % / 49.72 / 23 / 26 |
| it-s2-e3-djlr | decoder+joint lr 1e-4 -> 2e-5 | **36.57 (250)**: EOU 99.3 %, vp 43.0 / 3 empty, all empty 4 | 36.99 / 97.5 % / 46.23 / 11 / 14 |

Mynah (m1.sh; FLEURS-it bank 200 clips, greedy): control it-s2-e3 WER **30.11** / CER 10.38 (v2: 42.00 /
15.07), EOU on 200/200 clips (v2 188), speech end -> EOU p50 714 / p95 2218 ms (v2 1546 / 3370), never
0 % (v2 5 %), premature 3.3 % (all in 0.5-1.2 s pauses, none in speech; v2 0 %), A+1s+B: EOU in gap 95 %
(v2 75 %), p50 330 ms; CPU/CUDA parity PASS.

OBSERVED: the VP stage-2 blanking reproduces on E3 (vp empty 8 -> 16). This particular leading-quiet
intervention (p 0.9, U(0,4) s on top of the current recipe) does not fix it (23 empty, EOU 95.3 %);
other doses/distributions are not tested and not worth GPU now.

djlr per eval (macro / EOU / vp WER / vp empty; control in brackets): 250 36.57 / 99.3 % / 43.0 / 3
(37.54 / 98.7 % / 46.52 / 7); 500 37.64 / 99.0 % / 46.59 / 5 (38.95 / 96.5 % / 50.38 / 17); 750 36.94 /
98.3 % / 45.21 / 7 (37.92 / 96.9 % / 48.17 / 15); 1000 37.08 / 97.6 % / 46.39 / 10 (37.45 / 96.4 % / 48.83 / 16).
Wins 4/4 evals; the vp blanking still drifts up (3 -> 10) but more slowly. Strong causal evidence that
excessive stage-2 decoder/joint updating is a major contributor to the VP regression (component swap ->
damage localised in decoder/joint -> targeted LR cut largely removes it); `--lr` moves decoder and joint
together, so neither the component nor the mechanism is isolated. The selected checkpoint (step 250)
comes from the fixed 2026-10-09 selection rule (min WER among evals with EOU >= 80 % and <= 10 % empty).

| Mynah (m1.sh, FLEURS-it bank 200, greedy) | control | pre | djlr@250 |
|---|---|---|---|
| WER / CER | 30.11 / 10.38 | 30.53 / 10.46 | 30.91 / 10.27 |
| clips with model EOU | 200 | 200 | 200 |
| speech end -> EOU p50 / p95 | 714 / 2218 ms | 618 / 1290 ms | 810 / 2570 ms |
| missed @2 s / never | 11.7 % / 0 % | 0 % / 0 % | 13.3 % / 0 % |
| premature (any) | 3.3 % | 1.7 % | 1.7 % |
| A+1s+B EOU in gap, p50 | 95.0 %, 330 ms | 98.3 %, 490 ms | 95.0 %, 394 ms |
| CPU/CUDA parity | PASS | PASS | PASS |

**Decision: EOU-v3 = `plain-it-s2-e3-djlr` (step 250)**: passes every gate (macro <= 37.5, vp <= 46, vp
empty ~3, EOU >= 95 %); Mynah endpointing equivalent to the control except +96 ms p50; FLEURS-bank WER
differences are within noise (the gain is on VP, absent from that bank). Notes for later (not run):
leading quiet gave the best Mynah endpointing but the worst NeMo VP -> `djlr + leading quiet` is the
natural next stage-2 arm; a "light EOU adaptation" (~250 steps at low decoder/joint LR) is a hypothesis
for a cheap per-language stage 2, untested.

## E3-R1: short LR restart from E3 weights (opportunistic end-of-day probe, S15-21 P0)

NOT "E3 trained for 5k steps". Parent `plain-it-e2-enc2e-4-s4000/final.nemo` (E3 ended at 1 % of peak:
encoder 2e-6, decoder/joint 3e-6). `--init` loads weights only: fresh AdamW state, fresh schedule (warmup
50, cosine over 1000 to 1 %). Whole model trainable, encoder/decoder/joint 1e-4 (the only measured restart:
E1b, B2 -> all 1e-4, -4.1 in 1000 steps). A2 mix identical to E3, plain ASR, evals 250/500/750/1000, then
the CPU beam/oracle probe. Script `/root/res/e3r1.sh`; tag `it-e3r1-lr1e4`; launched 17:27 box time.

E3-R1 result (8.4 GPU-min, 46.9 h processed, peak 20.4 GB allocated / 22.2 reserved):

| step | macro WER / CER | mls | fleurs | cv | vp WER / CER / empty | all empty |
|---|---|---|---|---|---|---|
| 250 | 37.09 / 14.37 | 33.17 | 30.46 | 39.44 | 45.30 / 27.68 / 9 | 11 |
| 500 | 36.51 / 13.45 | 32.72 | 29.53 | 39.65 | 44.12 / 25.67 / 8 | 9 |
| 750 | 36.06 / 12.98 | 32.32 | 29.94 | 38.97 | 43.00 / 23.81 / 7 | 8 |
| 1000 | **35.92 / 13.03** | 31.90 | 29.51 | 38.76 | 43.53 / 24.38 / 7 | 8 |
| E3 (same harness) | 36.52 / 13.53 | 32.32 | 29.64 | 39.34 | 44.78 / 25.73 / 8 | 8 |

CPU probe @1000: greedy 35.91 / beam-8 **32.42** / oracle 27.70 (E3: 36.52 / 33.09 / 28.52).
OBSERVED: -0.60 macro (just above the ~0.5 noise floor), small gains on every domain. Reading per the
pre-registered bands: "~36-37, flat-ish" -> a short local restart from E3 buys little; tomorrow's P0 weight
moves from "continue E3" to the fresh long cosine (C1-12k was still falling at 9k) and to P1/P2
(additive data, distillation). R1 is the best plain checkpoint so far and is kept in the archive.
