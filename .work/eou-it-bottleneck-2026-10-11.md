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
