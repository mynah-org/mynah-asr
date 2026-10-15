# IT 120M — method lab on the DGX (GB10), 2026-10-15

Continuation of `it-120m-sub20-roadmap-2026-10-12.md`. Schedule scaling C250 20k -> 28k -> 36k is in diminishing
returns (clean greedy 26.48 -> 24.70 -> 23.73, 5 L40S-hours for the last -1.0). This session asks a different
question: which training-free or cheap change buys what more steps buy, and which recipe change gives a steeper
curve at 1-4k steps. Training-free first, micro-FT second; the tokenizer stays as is (762/1026 new rows, confounded).

## DGX qualification (OBSERVED)

- Box: DGX Spark GB10 (sm_121, aarch64, 121 GB unified memory, CUDA 13.0 driver), user-space venv
  `~/nemo-venv`: NeMo 3.0.0 + torch 2.11.0+cu128 (the L40S stack). numba-cuda 0.30.4 needs the CUDA 12.9 toolchain
  for sm_121 (nvcc/nvrtc/runtime/nvjitlink-cu12 12.9.x; with 12.8 NVVM rejects compute_121a, mixing cu13 NVVM with
  torch's cu12 nvJitLink fails the link). RNNT loss gate: `warprnnt_numba` forward/backward OK.
- Throughput, C250 replica (real durations/texts/weights of the 36k mix, synthetic audio, same recipe and LR
  schedule, 600 steps): **121x RT, 41.7 steps/min, peak 35.6 GB, GPU util ~95 %** vs L40S C250-36k 360x RT,
  ~120 steps/min: the GB10 is **3.0x slower**; compute-bound, not loader-bound. A 36k run = ~14-15 h here.
- Sanity FT: stock -> 800 steps on FLEURS-it train (2,175 clips), 200 held-out clips: WER 56.7 -> 46.3 -> 43.3
  -> 43.0 (steps 200/400/600/800), CER 17.6 -> 12.7, no empties/NaN; 187x RT on these shorter clips.
- Verdict: method lab (decoding, analysis, soups, 1-4k-step screens), not the long-run machine.

## Pre-registration (2026-10-15, before any number on the eval banks)

Checkpoint `plain-it-c250-36k` (clean 23.73 / 22.00 / 15.11, harness 22.27 / 20.53 / 16.25). Dev = the harness
(mls, fleurs, cv, vp; 200 per set); report = clean-it (tedx 909, italic 475), read ONCE at the dev-chosen setting.

1. **Greedy blank penalty** (`diag/greedy_blank_sweep.py`; serving decoder; parity gate PASSED: at P = 0 it
   reproduces NeMo greedy_batch exactly, 10.72 / 2.98 / len 0.981 on 200 FLEURS clips). Grid
   P in {-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2}, chosen by harness macro greedy WER.
   adopt: clean greedy -0.5 or better AND no clean set worse by > 0.3; marginal: -0.2..-0.5; null: < 0.2.
2. **Model soup** (`diag/soup.py`, uniform weights; all from the same stock init): 28k+36k and 20k+28k+36k.
   Selected on harness greedy; clean read once for the selected soup. keep: harness AND clean greedy -0.3 or
   better vs 36k; null otherwise. (No intermediate checkpoints of the 36k run were archived: only finals.)
3. **Error decomposition** of the 36k (`diag/nbest_gap.py` over the greedy dump + the beam-8 N-best dump):
   descriptive, no gate. Splits the greedy -> beam -> oracle@8 mass into search / ranking / no-good-hypothesis,
   and errors into near (ending, spelling, elision) vs far substitutions, deletions, insertions, segmentation,
   by set and clip duration. It decides which micro-FT arms are worth running; the E3-8k autopsy
   (88 % model errors, LM rescoring not transferable to TEDx) is the prior.

## Result: lab 1 on C250-36k (OBSERVED, GB10, 2026-10-15)

**Cross-box parity**: GB10 harness 22.30 / 20.53 / 16.25, clean 23.77 / 22.02 / 15.12 vs L40S 22.27 / 20.53 / 16.25,
23.73 / 22.00 / 15.11 (greedy / beam-8 / oracle-8). Same model, same numbers to a few hundredths.

**1. Greedy blank penalty — MARGINAL** (pre-registered band -0.2..-0.5). Dev-chosen P = 0.25 (harness 22.30 -> 22.26);
clean read once: 23.76 -> 23.52 (TEDx 23.08 -> 22.75, ITALIC 24.44 -> 24.28). It trades deletions for substitutions
(TEDx D 9.02 -> 8.32, S 11.71 -> 11.91): the missing words are mostly not recoverable by a blank prior.

**2. Model soups — NULL / negative.** Harness greedy: 28k+36k 23.20, 20k+28k+36k 23.86 vs 36k 22.30. Clean (28k+36k,
selected on harness): 24.45 vs 23.77. Finals of different schedules are too far apart; future runs should keep
close late checkpoints (or an EMA) if averaging is to be tested.

**3. Error decomposition** (`nbest_gap.py`, greedy / beam-8 / oracle-8, per 100 reference words):

| set | greedy WER | S_near | S_far | D | I | oracle WER | no-good-hyp share of oracle errors |
|---|---|---|---|---|---|---|---|
| mls | 21.46 | 8.94 | 5.81 | 4.30 | 2.41 | 17.12 | 28 % |
| fleurs | 15.21 | 5.36 | 4.13 | 2.34 | 3.38 | 10.84 | 14 % |
| cv | 24.37 | 10.24 | 7.88 | 3.47 | 2.78 | 14.39 | 57 % |
| vp | 28.14 | 5.27 | 5.83 | 13.57 | 3.46 | 22.67 | 59 % |
| tedx | 23.08 | 5.67 | 6.04 | 9.02 | 2.35 | 15.82 | 30 % |
| italic | 24.44 | 11.63 | 6.57 | 4.54 | 1.71 | 14.43 | 58 % |

Two failure modes: read speech (MLS, CV, ITALIC) is dominated by NEAR substitutions (ending / spelling / elision);
TEDx and VP by DELETIONS, concentrated at the clip boundaries (TEDx: 54 % of clips start with deleted words = 47 %
of its D; VP 39 % = 57 %). VP examples show the hypothesis running past the reference end (audio/reference offset,
consistent with the earlier VP audit); TEDx examples lose the first words and truncate the last one.

**4. Boundary diagnostic: digital silence padding at inference** (`greedy_blank_sweep.py --pre-pad-s/--post-pad-s`;
NOT pre-registered: run after the decomposition; the clean bank was read for every cell, so clean numbers below are
diagnostic, not a selection):

| pre / post silence | P | harness (dev) | tedx | italic | clean macro |
|---|---|---|---|---|---|
| 0 / 0 | 0 | 22.30 | 23.08 | 24.44 | 23.76 |
| 0 / 0.25 s | 0 | 22.22 | 21.36 | 23.03 | 22.20 |
| **0 / 0.5 s** | 0 | **22.21** | **20.71** | **22.80** | **21.76** |
| 0 / 1.0 s | 0 | 22.23 | 20.47 | 22.80 | 21.63 |
| 0.5 s / 0 | 0 | 22.41 | 21.80 | 25.06 | 23.43 |
| 0.5 s / 0.5 s | 0 | 22.37 | 19.00 | 23.41 | 21.20 |
| 0.5 s / 1.0 s | 0.25 | 22.40 | 18.71 | 23.16 | 20.94 |

- **Trailing flush is causal and free**: 0.25-1 s of trailing silence never hurts a harness set and removes most
  end-of-clip losses (TEDx trailing-deletion clips 10 % -> 2 %, last-2-word share of D 16 % -> 5 %; truncated last
  words come back). The dev-best cell (0 / 0.5 s, harness 22.21) gives clean **21.76 (-2.0)**. The streaming encoder
  needs right context to emit the last tokens; our offline evals never fed it, so every clean/harness number so far
  understates the models at clip ends (the harness read sets barely move: their audio already ends in silence).
  Serving: the final transcript must flush the encoder with silence (check the CUDA server's finalisation path).
- **Leading silence is mixed**: TEDx -1.3..-1.7 on top (leading-deletion clips 54 % -> 23 %: a real cold-start /
  onset effect), but MLS +0.75, CV +0.4, ITALIC +0.6. Not an inference fix; it motivates an onset-robustness
  training arm (random pre-roll, `two_stage.py --pre-pad-prob/--pre-pad-max`, already implemented for stage 2).
- The best cells reach clean ~20.9-21.2 (TEDx 18.7-19.0) but were picked with the clean bank in view: NOT claimable.
- Next (re-baseline): every model is re-scored with the dev-chosen trailing flush (0.5 s) before comparing; the
  micro-FT tournament (R0 control / R1 WSD schedule / R2-R3 CTC 0.1-0.25, then pre-roll) reports S_near / S_far / D / I
  per set, not only WER.

## Method tournament: objective and R0 vs R2 pre-registration (2026-10-15, before results)

Objective (reframed): a cheap, reproducible adaptation recipe English EOU120M -> language specialist that reaches
most of a long run's accuracy in a fraction of the steps / processed audio / GPU time. The 36k is a reference,
not the recipe. Arms are judged against the CONTROL AT THE SAME BUDGET (learning curve), one variable per arm.
Every arm reports WER / CER / S_near / S_far / D / I / empty, audio hours, wall time and encoder grad norm
(`diag/arm_report.py`), raw EOF and 0.5 s trailing flush. Dev = harness (in-run val + post-hoc); clean-it is
post hoc only and now CONTAMINATED as a research-development bank (read many times today): a fresh untouched
holdout is needed before any final claim.

Common: E3 recipe (stock init, A2 mix MLS 40h .35 / CV .30 / VP-t50 .20 / FLEURS .15 — identical manifests to the
L40S, VP t50 rebuilt from the archived scores and byte-checked), whole encoder 2e-4, decoder+joint 3e-4, bs 16,
max-dur 20, seed 0, `--steps 8000` cosine, `--stop-at 1000`, evals at 250/500/1000. Cross-box anchor: L40S E3-8k
in-run val at 1k = 44.51 (R0 should land within ~1 of it).

- R0 control: no extra flag. Gate passed before launch: with `--ctc-weight 0` the patched trainer reproduces the
  original losses bit for bit (317.981 / 236.349 at steps 25 / 50).
- R2: `--ctc-weight 0.1` only. Hypothesis: a direct monotonic alignment signal on the encoder speeds Italian
  acoustic adaptation. Mechanism checks (smoke, OBSERVED): CTC uses the encoder lengths and the RNNT targets /
  lengths, blank = V (1026, same as RNNT), sum per utterance / batch (= RNNT `mean_batch`), training-only head
  (the saved .nemo has the stock's 467 keys, none CTC); losses finite; per-utterance CTC ~2.5-3.5x RNNT early,
  so w 0.1 is ~20-25 % of the total. Caveat: 80 ms frames vs ~9 Italian pieces/s: fast speech can make CTC
  infeasible for a clip (zero_infinity drops it).
- Bands at 1k (harness macro greedy, in-run val and post-hoc flushed; R2 - R0): **lever** <= -2.0 with D not
  converted into S_far (S_far must not rise by more than half the D drop); **promising** -1.0..-2.0;
  **null** within +-1.0; **harmful** > +1.0. Null -> no 8k nursing: move to the schedule axis (R1, WSD).

## Result: R0 vs R2 @1k (OBSERVED, GB10) — CTC 0.1 NULL

Parity gate PASSED: R0 in-run harness val 65.48 / 55.33 / 45.34 at 250 / 500 / 1000 vs L40S E3-8k 64.96 / 53.76 /
44.51 (+0.8 at 1k: hardware nondeterminism, same recipe/data/seed).

| | R0 control | R2 CTC 0.1 | R2 - R0 |
|---|---|---|---|
| harness macro greedy, raw EOF | 45.34 | 46.12 | **+0.78 (NULL band)** |
| harness macro greedy, 0.5 s flush | 45.41 | 46.20 | +0.79 |
| harness CER | 16.05 | 15.52 | -0.53 |
| harness empties (in-run val) | 8 | 3 | |
| wall s / audio h (1k steps) | 1143 / 46.9 | 1150 / 46.9 | CTC is free |
| enc grad norm, steps 500-1000 | 160 | 176 | +10 % |

Mechanism: CTC cuts deletions (VP D 19.6 -> 14.5, TEDx 15.4 -> 12.8 per 100 words; fewer empties) but turns them
into far substitutions and insertions (VP S_far 17.0 -> 19.3, I 5.1 -> 6.7; CV +2.4 overall). The curve crosses
(R2 -0.8 at 500, +0.8 at 1k): within noise. Per the pre-registration: fixed-weight CTC is NULL; not nursed to 8k.
Annealed CTC (strong early, -> 0) stays an untested hypothesis, low priority.

Full report: `arm_report.py` output (DGX `~/ft/analysis/arms/r0_r2_1k.md`), copied below in short form only.
Artifacts on the private Hub repo: `runs/dgx-r0-ctl-1k`, `runs/dgx-r2-ctc01-1k` (final.nemo + metrics.json),
`results/dgx-2026-10-15/{analysis,logs}` (lab 1 dumps, N-best dumps, arm dumps and reports, all logs).

## Why R1 (WSD at an 8k horizon) was NOT run, and the next experiment

With `--steps 8000`, the cosine is still at 96 % of peak at step 1000 (warmup ends at 400), so a flat-LR arm differs
from R0 by < 4 % LR up to 1k: an R1 at 500-1000 steps cannot discriminate anything. Schedule hypotheses must be
tested at budgets where the decay actually happens. Next GPU session (DGX ~25 min per 1k steps on A2):

- Budget B = 2000, every arm's schedule COMPLETES at B (`--steps 2000`, evals 500/1000/1500/2000), A2 mix, seed 0:
  - S0 `--steps 2000` (cosine to B: the control at that budget);
  - S1 `--steps 2000 --lr-shape wsd --decay-frac 0.2` (flat then short decay);
  - S2 S0 with the ENCODER peak LR x2 only (`--enc-lr 4e-4`, decoder+joint stay 3e-4): the E3 lesson was encoder
    plasticity; doubling both would confound it.
  Every run now writes `lr_auc` (sum of LR over optimizer steps, per parameter group) in metrics.json: the
  integrated-LR-starvation hypothesis is read from it, not from nominal peak LRs.
  Compare against each other at 2k AND against R0's 8k-horizon curve at the same steps; report integrated LR per
  group. Reference point: E3-8k reached 39.77 at 2k and 35.53 at 3k on the L40S.
- Pre-registered bands (vs S0 at 2k, harness macro greedy, flushed): lever <= -2.0; promising -1..-2; null +-1.
- If schedule is null too: curriculum / bounded hard-example sampling (teacher-good / student-bad band).
- Untouched holdout needed before any final claim (clean-it is now a research-development bank).

## Result: minimum sufficient trailing flush on C250-36k (OBSERVED; harness = dev, clean = contaminated bank)

| trailing silence | +encoder frames (80 ms) | harness | clean | tedx (D) | italic |
|---|---|---|---|---|---|
| 0 | 0 | 22.30 | 23.76 | 23.08 (9.02) | 24.44 |
| 40 ms | 0.5 | 22.20 | 23.25 | 22.82 (8.82) | 23.67 |
| 80 ms | 1 | **22.16** | 22.82 | 22.36 (8.67) | 23.29 |
| 120 ms | 1.5 | 22.20 | 22.84 | 22.13 (8.38) | 23.54 |
| 160 ms | 2 | 22.22 | 22.66 | 21.88 (8.31) | 23.45 |
| 240 ms | 3 | 22.21 | 22.16 | 21.32 (8.03) | 23.00 |
| 320 ms | 4 | 22.22 | 22.02 | 21.04 (7.87) | 23.00 |
| 500 ms | 6.2 | 22.21 | 21.76 | 20.71 (7.75) | 22.80 |
| 1000 ms | 12.5 | 22.23 | 21.63 | 20.47 | 22.80 |

- The harness saturates after one encoder frame (its read clips already end in silence) and CANNOT select this knob;
  clean-it keeps improving to ~0.5-1 s (TEDx segments end on speech, often mid-word). Most of the gain is in the
  first 1-4 frames (clean -0.9 by 80 ms, -1.7 by 320 ms).
- Reading: the end-of-input loss is real and grows with how abruptly the clip ends; a production finaliser should
  feed >= 0.3-0.5 s of silence (or the equivalent zero-input encoder steps) before the final transcript. A fresh
  holdout with natural end-of-utterance boundaries is needed to fix the number without the contaminated bank.
- All future evals report raw EOF AND 0.5 s flush; historical numbers are kept as raw EOF.
