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
