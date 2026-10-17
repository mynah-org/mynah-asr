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

## Result: schedule tournament @2k (OBSERVED, L40S Vast, same box/seed/data/budget)

Box: fresh L40S (8 cores, 84 GB disk); A2 data rebuilt and gated (all audio present, VP t50 from the archived scores).
First launch crashed at step 1 (numba NVVM 12.9 vs torch nvJitLink 12.8): fixed by the 12.9 toolchain, now pinned
and gated in prov.sh. The arm logs lost their step lines (grep treated NeMo output as binary; now `grep -a`), so
no RNNT-loss / encoder-grad-norm windows for these three arms; metrics.json, lr_auc and the dumps are complete.

In-run harness macro greedy WER (raw EOF), schedules COMPLETING at 2k (warmup 100):

| arm | difference | 500 | 1000 | 1500 | 2000 | lr_auc dec+joint / encoder | wall s |
|---|---|---|---|---|---|---|---|
| S0 | cosine 2k (control) | 51.10 | 44.17 | 42.19 | **41.96** | 0.289 / 0.192 | 806 |
| S1 | WSD, decay last 20 % | 51.39 | 42.36 | 40.14 | **38.13 (-3.83)** | 0.526 / 0.351 | 784 |
| S2 | encoder peak LR x2 (4e-4) | 48.17 | 40.88 | 39.55 | **38.85 (-3.11)** | 0.289 / 0.385 | 785 |

0.5 s flush, harness: S0 41.98, S1 38.17, S2 38.84. Clean-it (contaminated, post hoc, flushed): S0 44.50, S1 40.94,
S2 41.66. For scale: L40S E3-8k (cosine 8k) was 39.77 at 2k and 35.53 at 3k.

- Both are LEVERS (pre-registered <= -2.0 vs S0). Curve shapes differ: S2's advantage appears from 500 steps and
  stays ~3 (-2.9 / -3.3 / -2.6 / -3.1): faster encoder adaptation from the start. S1 equals S0 at 500 (identical
  LR there), then grows (-1.8 / -2.1 / -3.8): the gain comes from holding the peak and decaying late.
- Both raise the ENCODER integrated LR to ~0.35-0.39 (S0 0.19): consistent with the integrated-LR-starvation
  hypothesis (S1 also doubles the decoder+joint AUC, so S1 alone cannot attribute it to the encoder; S2 can).
- Error composition (harness, raw): the gains are mostly FAR substitutions (MLS S_far 14.3 -> 12.4 / 12.1, VP
  16.5 -> 13.9 / 13.6, CV 17.2 -> 15.8 / 15.6) with near substitutions and deletions down a little: real lexical /
  acoustic learning, not a deletion -> substitution trade. Empties rise slightly on VP (0.8 -> 3.1 %).

### S3 pre-registration (launched 11:40 UTC, before its numbers): WSD + encoder 4e-4 @2k

`--steps 2000 --lr-shape wsd --decay-frac 0.2 --enc-lr 4e-4`, everything else as S0. Question: are the two levers
additive? Bands vs the best single arm S1 (38.13, harness macro greedy raw at 2k): **additive** <= -1.5;
**partly additive** -0.5..-1.5; **redundant** within +-0.5; **overshoot** > +0.5 (then: longer warmup or enc 3e-4).
Next if additive: promote the combined recipe to an 8k-horizon run and compare against E3-8k / C250 trajectories
(target: E3-8k's 8k quality in <= 4k steps).

## Result: S3 = WSD + encoder 4e-4 @2k — ADDITIVE (OBSERVED)

| arm | 500 | 1000 | 1500 | 2000 | lr_auc dec+joint / encoder |
|---|---|---|---|---|---|
| S0 cosine 2k | 51.10 | 44.17 | 42.19 | 41.96 | 0.289 / 0.192 |
| S1 WSD | 51.39 | 42.36 | 40.14 | 38.13 | 0.526 / 0.351 |
| S2 enc x2 | 48.17 | 40.88 | 39.55 | 38.85 | 0.289 / 0.385 |
| **S3 WSD + enc x2** | 49.79 | 42.02 | 39.54 | **35.64** | 0.526 / 0.701 |

- vs S1 -2.49 (pre-registered additive band <= -1.5); vs S0 -6.32. Flushed harness 35.75; clean-it (contaminated)
  flushed 38.20 (S0 44.50). E3-8k (cosine 8k) needed ~3k steps for 35.53: S3 gets there in 2k.
- Gains again mostly far substitutions (MLS S_far 14.3 -> 11.6, VP 16.5 -> 11.6, CV 17.2 -> 14.3) plus near ones.
- Shape: S3 trails S2 at 500 (49.8 vs 48.2: flat peak vs cosine) and gains most in the final decay (-3.9 from 1500
  to 2000). Encoder grad norm (logs now intact) 206 -> 126 over the run: no instability at 4e-4 flat.
- The encoder's integrated LR keeps paying (0.19 -> 0.35/0.38 -> 0.70, WER 42.0 -> 38.1/38.9 -> 35.6): the
  frontier is not found yet.

### Next two arms, pre-registered (2026-10-15 ~12:00 UTC, before results)

- **S4 — encoder LR frontier @2k**: S3 with `--enc-lr 8e-4` (encoder AUC ~1.4). Bands vs S3 (35.64): frontier still
  rising <= -1.0; flat within +-1.0 (S3's 4e-4 is near the optimum); overshoot > +1.0 (or any collapse / empties
  > 5 %) -> the optimum is below 8e-4.
- **S3-4k — horizon**: S3's recipe with `--steps 4000` (WSD decay over the last 800). Question: does the gain hold
  with more budget? Reference: E3-8k harness greedy 31.36 at 8k. **Strong**: <= 31.4 at 4k (E3-8k quality at half
  the steps); **good**: 31.4-33.0; **weak**: > 33.0 (then the 2k advantage is mostly a fast start).

## Result: S4 (encoder 8e-4 @2k) and S3-4k (OBSERVED)

| arm | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | lr_auc enc | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| S3 WSD + enc 4e-4, 2k | 49.79 | 42.02 | 39.54 | 35.64 | | | | | 0.70 | 784 |
| **S4** WSD + enc 8e-4, 2k | 47.51 | 41.15 | 38.35 | **34.46** | | | | | 1.40 | 793 |
| **S3-4k** WSD + enc 4e-4, 4k | 49.26 | 41.20 | 38.96 | 36.42 | 36.36 | 33.40 | 31.02 | **29.96** | 1.40 | 1583 |

- **S4: frontier still rising** (-1.18 vs S3; band <= -1.0, at the edge). No collapse: empties 4-6 / 730, encoder grad
  norm lower than S3 (96 vs 126 late). Gains on MLS (S_far 11.6 -> 9.9); VP / TEDx about flat. The optimum
  encoder LR is >= 8e-4 at this budget; the next step up (1.2e-3 or 1.6e-3) must show where it turns.
- **S3-4k: STRONG** (pre-registered <= 31.4): harness greedy **29.96** at 4k (flushed 29.97) vs E3-8k's 31.36 at 8k:
  better than the old 8k recipe with half the steps (4000 x 16 clips = ~187 audio h, 26 min of L40S). Clean-it
  (contaminated) flushed 33.55. A transient dip at 2500 (36.36, 18 empties) during the flat phase recovers by 3000:
  watch empties in longer flat phases.
- Same encoder AUC (1.40) at 2k (S4 34.46) and at 4k-halfway (S3-4k 36.42 at 2k): AUC is not the whole story; the
  terminal decay matters (most of each run's gain lands in its last 20 %).
- Data note: these are A2-only (MLS/CV/VP/FLEURS ~120 h). The C250 long runs add ~228 h of Granary; the recipe
  should be promoted onto the C250 mix and compared with C250-20k (25.90 harness greedy) at <= 8k steps.

### Proposed next (not launched)

1. S5 = S3-4k with encoder 8e-4 (both levers at their current best) at 4k — one arm, 26 min.
2. Encoder LR frontier at 2k: 1.6e-3 — one arm, 13 min.
3. Promotion: best recipe on the C250 mix (needs the two Granary pools; ~55 GB, fits this 84 GB box) at 8k,
   vs C250-20k / C250-36k. Pre-register before launch.

### S5 and S6 pre-registration (2026-10-15 ~12:55 UTC, before results)

- **S5** = S3-4k with `--enc-lr 8e-4` (WSD, decay last 20 %, 4k). Bands vs S3-4k (29.96 at 4k, harness greedy raw):
  **win** <= -1.0 -> freeze S5 as the recipe and promote it to the C250 mix; **flat** +-1.0 -> keep 4e-4 (lower risk);
  **worse** > +1.0. Also report the dips/empties in the flat phase (S3-4k had 18 empties at 2500).
- **S6 boundary probe** = S4 with `--enc-lr 1.6e-3` at 2k. NOT a promotion candidate: it locates the LR boundary.
  vs S4 (34.46): still rising <= -1.0; flat +-1.0; overshoot > +1.0 or empties > 5 % or a NaN/collapse.
After these two: no more A2 micro-tuning; the next run is the promotion on the C250 mix (pre-registered separately).

## Roadmap toward ~10 % (agreed 2026-10-15; phases gate on ORACLE@8, not only greedy)

Error budget at C250-36k (harness greedy / beam-8 / oracle-8 = 22.3 / 20.5 / 16.25; clean raw 23.7 / 22.0 / 15.1, flushed
greedy 21.8): candidate generation sets an oracle floor ~15; ranking/search costs ~5; finalisation ~2 (now fixed in
eval). Rule: oracle does not move -> work on encoder / data / KD; oracle moves but beam does not -> decoder / ranking.
Wall target: oracle@8 < 10 first, then turn it into 1-best. 1-5 % only as a separate, domain-restricted experiment.

- A. Optimizer (today): S5 / S6 -> freeze the winning recipe. No further A2 micro-tuning.
- B. Exposure: winner recipe on the C250 mix at 8k (then 16k if still steep) vs C250-20k / 36k (raw + flushed,
  greedy / beam / oracle). Pre-register before launch.
- C. Curriculum: bucketed dynamic sampling (easy / near-S / far-S / deletion-heavy / empty / teacher-disagreement),
  up-weighting "teacher right, student wrong, label credible"; never up-weight alignment garbage (VP offsets).
- D. it-yg7 500 h: teacher- and student-guided selection (teacher quality, student-teacher disagreement, deletion
  ratio, duration, source diversity, near-duplicates), not uniform addition (OLDREP: more videos != better).
- E. True KD: encoder representation distillation from Nemotron (both FastConformer, 80 ms frames: alignment is a
  projection 512 -> teacher dim + length match; cosine/MSE; lambda annealed to 0), teacher cost reported apart.
  Then N-best / sequence KD with teacher scores.
- F. Ranking: only once oracle@8 is low enough; discriminative N-best KD / reranking.
- G. EOU stage 2 on the best plain checkpoint (EOU-v3 low decoder/joint LR recipe).
- Holdout: build a FINAL-IT untouched set NOW (sources disjoint from training and from today's analyses), freeze its
  manifest hash in this file and do not score it until a final claim. clean-it is a development bank from today on.

## Result: S5 and S6 (OBSERVED) — optimizer recipe frozen

| arm | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | lr_auc enc | empties (last) |
|---|---|---|---|---|---|---|---|---|---|---|
| S3-4k enc 4e-4 | 49.26 | 41.20 | 38.96 | 36.42 | 36.36 | 33.40 | 31.02 | 29.96 | 1.40 | 5 |
| **S5** enc 8e-4, 4k | 48.39 | 41.01 | 37.11 | 36.17 | 35.39 | 32.64 | 31.03 | **29.35** | 2.80 | 6 |
| S4 enc 8e-4, 2k | 47.51 | 41.15 | 38.35 | 34.46 | | | | | 1.40 | 6 |
| **S6** enc 1.6e-3, 2k | 50.26 | 44.90 | 41.37 | **35.91** | | | | | 2.81 | 9 |

- **S5: FLAT** (-0.61 vs S3-4k; band +-1.0). Better on MLS (26.06 vs 27.56, S_far 8.6 -> 7.8), VP deletions / empties
  slightly up (empties 4.6 % on VP). No dip at 2500 this time (35.39). Per the pre-registration: keep 4e-4.
- **S6: OVERSHOOT** (+1.45 vs S4): worse from 500 steps on (50.3 vs 47.5), more empties, VP deletions 18.5. The
  encoder LR boundary at this batch / horizon lies between 8e-4 and 1.6e-3; at 4k, 4e-4 and 8e-4 are equivalent.
- **Frozen recipe "W4"**: stock init, whole encoder 4e-4 + decoder/joint 3e-4, AdamW (0.9, 0.98) wd 1e-3, bs 16,
  warmup 5 % then flat peak, cosine decay over the last 20 % (`--lr-shape wsd --decay-frac 0.2 --enc-lr 4e-4`).
  A2-only, 4k steps: harness greedy 29.96 (E3-8k at 8k: 31.36); 26 min on an L40S.
- Next = phase B: W4 on the C250 mix at 8k vs C250-20k (harness greedy 25.90, clean 26.48 raw) and C250-36k
  (22.27 / 23.73 raw), greedy / beam-8 / oracle-8, raw and 0.5 s flush. Pre-register before launch.

## Phase B pre-registration (2026-10-15 ~13:50 UTC, before launch): W4 on the C250 mix at 8k

Run `it-w4-c250-8k`: `p1_additive.sh` with the C250 external pools (it-yg + it-yg6 agree<=15, PASSES 1.5 -> W 0.4388,
the C250-20k/36k mix unchanged), `FIX_STEPS=8000`, `P1_EXTRA="--lr-shape wsd --decay-frac 0.2 --enc-lr 4e-4"`
(W4, frozen: NOT changed on intermediate results), seed 0, same box family (L40S). Probes: greedy / beam-8 / oracle-8
on the harness and clean-it, raw EOF AND 0.5 s trailing flush; C250-36k is re-probed with the flush on the same box
so both protocols have a same-box reference. Clean-it is a contaminated development bank: secondary.

Baselines (raw EOF, greedy / beam-8 / oracle-8): C250-20k harness 25.90 / 22.96 / 18.98, clean 26.48 / 23.76 / 17.11;
C250-36k harness 22.27 / 20.53 / 16.25, clean 23.73 / 22.00 / 15.11.

Primary = harness oracle@8 and greedy (raw); the reading:
- **compute compression**: W4-8k harness greedy <= 25.90 (C250-20k quality in 2.5x fewer steps);
- **strong compression**: harness greedy <= 22.27 (C250-36k quality in 4.5x fewer steps);
- **representation shift**: harness oracle@8 <= 15.25 (1.0 below C250-36k) — W4 changes candidate generation, not
  only speed; oracle@8 <= 12 on harness AND clean -> switch priorities to teacher/student + ranking (roadmap E/F);
- **null**: harness greedy > 25.90 (the A2 gain does not survive the larger mix).
Stop after phase B and report before any further training.

## Result: phase B — W4 on the C250 mix at 8k (OBSERVED, L40S Japan, 2026-10-15)

Same mix as C250-20k/36k (W 0.4388, identical pools 28,573 + 38,140 clips and weights), W4 recipe, 8k steps, 3714 s
(62 min), 401.6 processed audio hours, lr_auc dec+joint 2.10 / encoder 2.80. Probes (greedy / beam-8 / oracle-8):

| run | steps | harness raw | harness 0.5 s flush | clean raw | clean 0.5 s flush |
|---|---|---|---|---|---|
| **W4-C250-8k** | 8k | **26.12 / 23.67 / 19.43** | 26.17 / 23.72 / 19.51 | 27.29 / 25.05 / 17.45 | 25.96 / 23.85 / 16.45 |
| C250-20k | 20k | 25.90 / 22.96 / 18.98 | — | 26.48 / 23.76 / 17.11 | — |
| C250-36k (re-probed here) | 36k | 22.27 / 20.54 / 16.26 | 22.21 / 20.56 / 16.25 | 23.73 / 22.00 / 15.12 | **21.73 / 20.23 / 13.71** |

In-run harness val at equal steps (raw): W4 44.16 / 38.02 / 32.17 / 29.51 / 26.11 at 1k / 2k / 4k / 6k / 8k vs the
C250-36k trajectory 53.89 / 43.08 / 34.41 / 31.54 / 29.07 (then 27.84 @12k, 25.43 @16k, 23.90 @20k, 22.37 @36k).

- **Pre-registered reading: NULL** (harness greedy 26.12 > 25.90, by 0.22): W4 at 8k does NOT beat C250-20k, and
  oracle@8 (19.43) is worse than C250-36k (16.26): no representation shift.
- What it is instead: C250-20k quality within 0.2-0.8 in 2.5x fewer steps (62 vs ~150 L40S-min), and ~3 points ahead of
  the long cosine at the same step; W4@8k sits where the 36k-cosine run was at ~14k steps.
- The A2 lesson (4k beats the old 8k) does NOT transfer as "8k beats 20k" on a 3x larger pool: at 8k the model sees the
  357 h pool ~1.1 times; the steep part of the C250 curve is still exposure-limited. W4 compresses the schedule, it does
  not remove the need for exposure on the bigger mix.
- Flush (same-box reference): C250-36k clean flushed **21.73 / 20.23 / 13.71** — the current best model with the
  corrected finalisation; the harness is flat under flush (its clips end in silence).
- Next candidates (NOT launched; pre-register first): W4 at 16k on C250 (does the faster schedule beat C250-36k at
  less than half its steps?), or keep 8k and move to curriculum / guided data selection (phase C/D). Clean-it stays a
  development bank; a FINAL-IT holdout is still to be built.

Artifacts: Hub `runs/l40s-it-w4-c250-8k` (final.nemo, metrics, probes), `results/l40s-jp-2026-10-15/{analysis,logs}`
(C250-36k raw/flushed reference probes in analysis/phaseB).

## Phase B2 pre-registration (2026-10-16, before launch): W4 on the C250 mix at 16k

Run `it-w4-c250-16k`: identical to phase B (same pools, W 0.4388, W4 frozen, seed 0) except `FIX_STEPS=16000`
(~2.2 passes over the 357 h pool). Box: RTX 6000 Ada 48 GB (Vast, Estonia; same AD102 family as the L40S); C250-36k
is re-probed on the same box (raw + 0.5 s flush) as the reference. Question: does the faster schedule reach C250-36k
quality at less than half its steps once exposure is no longer the limit?

Primary = harness greedy and oracle@8 (raw EOF); reading:
- **strong compression**: harness greedy <= 22.27 (C250-36k quality in 2.25x fewer steps) -> W4 becomes the default
  schedule for every next phase (C/D/E run at 16k instead of 36k);
- **partial**: 22.27 < greedy <= 23.90 (ahead of the 36k-cosine trajectory at 20k steps) -> W4 is a cheaper screening
  schedule, final runs keep the long horizon;
- **null**: greedy > 23.90 (the W4 advantage at 8k is a phase effect, not a better endpoint);
- **representation shift**: harness oracle@8 <= 15.25 (1.0 below C250-36k), independent of the greedy reading.
Secondary: clean-it (contaminated development bank) greedy/beam/oracle raw and flushed vs C250-36k flushed
21.73 / 20.23 / 13.71. W4 is NOT changed on intermediate in-run numbers. Stop and report after B2.

## Result: phase B2 — W4 on the C250 mix at 16k (OBSERVED, RTX 6000 Ada, Vast EE, 2026-10-16)

Same mix/pools/recipe as phase B, 16k steps, 7621 s (127 min, 803.6 processed audio hours, lr_auc dec+joint 4.21 /
encoder 5.61); final.nemo = best in-run harness val (step 15500, as for every run). Probes (greedy / beam-8 / oracle-8),
C250-36k re-probed on the same box:

| run | steps | harness raw | harness 0.5 s flush | clean raw | clean 0.5 s flush |
|---|---|---|---|---|---|
| **W4-C250-16k** | 16k | **22.76 / 21.02 / 16.71** | 22.83 / 21.23 / 16.87 | 23.64 / 21.48 / 15.01 | 22.64 / 20.55 / 14.29 |
| W4-C250-8k | 8k | 26.12 / 23.67 / 19.43 | 26.17 / 23.72 / 19.51 | 27.29 / 25.05 / 17.45 | 25.96 / 23.85 / 16.45 |
| C250-36k (same box) | 36k | 22.27 / 20.53 / 16.25 | 22.21 / 20.57 / 16.27 | 23.73 / 22.00 / 15.11 | **21.74 / 20.23 / 13.71** |

In-run harness val: 47.87 / 38.16 / 33.91 / 29.86 / 27.70 / 26.41 / 27.04 (12k, end of the flat peak) / 24.38 / 22.76
at 1k / 2k / 4k / 6k / 8k / 10k / 12k / 14k / 15.5k; the 20 % decay (12.8k-16k) is worth ~4 points, as at 8k.

- **Pre-registered reading: PARTIAL** (22.27 < 22.76 <= 23.90): W4 reaches C250-36k within 0.5 greedy in 2.25x fewer
  steps (127 vs ~290 GPU-min), but does not match it. **No representation shift**: harness oracle@8 16.71 vs 16.25
  (worse by 0.46); clean raw is a tie (23.64 / 21.48 / 15.01 vs 23.73 / 22.00 / 15.11), clean flushed slightly worse.
- Per the pre-registration: W4 is the screening schedule (cheap probes at 8k-16k); a final run keeps the long horizon.
- Reading of the schedule axis: the faster schedule converges to the SAME ceiling (oracle ~16 harness, ~14-15 clean)
  that 36k steps of plain fine-tuning on this mix reach. More steps / a better LR shape no longer move candidate
  generation: the schedule axis is closed. Next = data and signal (phase C curriculum, D guided selection, E encoder
  KD), screened with W4 at 8k against W4-C250-8k (26.12 / 23.67 / 19.43), promoted to 16k only on an oracle gain.
- Throughput: 6000 Ada ~126 steps/min over the whole run including evals, the same as the L40S (129).

Artifacts: Hub `runs/ada-it-w4-c250-16k` (final.nemo, metrics, probes), `results/ada-ee-2026-10-16/{analysis,logs}`;
data snapshot `data/snapshots/it-c250` (22 shards, 22.2 GiB; `resume_it.sh SNAPSHOT=it-c250`).

## Throughput profile (OBSERVED, RTX 6000 Ada, 2026-10-16; `two_stage.py --profile-steps 300`, B2 mix/policy, 2 seeds)

| step | host batch build | GPU fwd+loss | GPU bwd | GPU clip+opt | GPU idle between steps | steps/min |
|---|---|---|---|---|---|---|
| 273-276 ms | 25.2-25.6 ms (9.2 %) | 112-114 ms | 120-121 ms | 15 ms | 25.3-25.7 ms (9.2-9.4 %) | 218-220 |

- The synchronous batch build costs ~9 % of a step (GPU idle ~= host time): BELOW the 10-15 % gate, so no prefetch
  thread for now (at most ~10 % on the training part).
- The real overhead is the in-run evaluation: pure training runs at ~219 steps/min, the B2 run averaged ~126 over
  16k steps (7621 s vs ~4400 s of training): ~50 min (~40 %) went to 34 evals (4 harness sets x 200 clips, decoded,
  plus a final.nemo + 1.4 GB last.ckpt save on every new best). For screening runs, an eval every 2000 steps in the
  flat phase and every 500 in the decay (where WSD's best always lands) would cut a 16k run from ~127 to ~80 min;
  pre-register it before use (it changes which checkpoint is final.nemo only if a flat-phase step were the best).

## Phase C1 pre-registration (2026-10-16 ~15:05 UTC, before launch): within-source difficulty curriculum, W4 8k

Scorer (frozen): W4-C250-16k greedy on all 116,526 C250 training clips (`student_score.py`, parity with
lm_fusion_probe greedy exact on the harness), sha256 of the scores 8d4c9271..., of the mix 0df9a4c1....
Buckets (`curriculum_mix.py`, credibility estimated without the student: Granary two-teacher agreement, VP teacher
vs label joined from train_vp.teacher.json, MLS/CV/FLEURS human labels + chars/s within p1..p99): hard 10,794 clips
18.4 h, mid 80,238 / 270.0 h, solved 22,522 / 51.4 h, lowcred 2,831 / 7.5 h, suspect 141 / 0.4 h.
C1 = hard x2, solved x0.5, everything else x1, normalised INSIDE each source (C250 source marginals unchanged):
hard 8.5 % -> 16.5 % of draws, solved 19.5 % -> 9.9 %, mid 68.9 % -> 70.3 %. W4 recipe, 8k steps, seed 0, screening
harness policy (`--eval-sparse 2000,500 --final-eof 1`: final.nemo = step 8000; W4-C250-8k's final was step 8000 too).

Single gate: harness oracle@8 (raw) <= 18.4 (W4-C250-8k: 19.43) -> promote to 16k; > 18.4 -> C1 has not added enough
information; next is targeted selection / encoder KD. Greedy / beam / CER / empties are recorded, not decisive.
Declared confounds (not corrected, to avoid designing on the distribution): `hard` is 60 % CV clips (5 s median: a
WER >= 30 threshold is easier to cross on short clips; CV hard goes 24 % -> 44 % of CV draws); VP hard (232 clips) is
deletion-heavy (hyp/ref 0.70, 8.6 % empty, teacher 5.3). If C1 passes, the first read is per-set delta oracle
(MLS / FLEURS / CV / VP) and VP deletions / empties, not only the macro.

## Result: phase C1 — within-source difficulty curriculum, W4 8k (OBSERVED, RTX 6000 Ada, 2026-10-16) — NULL

final.nemo = step 8000 (EOF), 3164 s (53 min with the sparse evals). Probes (greedy / beam-8 / oracle-8):

| run | harness raw | harness 0.5 s flush | clean raw | clean 0.5 s flush |
|---|---|---|---|---|
| **C1** (hard x2, solved x0.5) | **26.27 / 23.64 / 19.45** | 26.29 / 23.68 / 19.50 | 27.04 / 24.47 / 17.53 | 26.08 / 23.56 / 16.78 |
| W4-C250-8k (reference) | 26.12 / 23.67 / 19.43 | 26.17 / 23.72 / 19.51 | 27.29 / 25.05 / 17.45 | 25.96 / 23.85 / 16.45 |

C1 per set, harness raw greedy (oracle-8): MLS 24.97 (19.90), FLEURS 19.42 (14.13), CV 27.89 (18.12), VP 32.82 (25.63).
In-run harness val: 43.42 / 38.52 / 32.58 / 29.76 / 27.40 / 26.47 / 26.28 at 1k / 2k / 4k / 6k / 7k / 7.5k / 8k.

- **Pre-registered reading: NULL** (oracle@8 19.45 vs gate 18.4; reference 19.43). Every number is within +-0.6 of
  the reference on both banks; doubling the draws of credible clips a strong model still gets wrong and halving the
  solved ones, at fixed source marginals and budget, does not change candidate generation at 8k.
- Reading (scoped): THIS static difficulty function (hard x2 / solved x0.5, defined by the W4-16k scorer and the
  credibility rules above) does not change candidate generation at 8k. Not excluded: dynamic / loss-aware sampling,
  competence schedules, domain-specific curricula; not pursued now (low ROI vs D / E). Together with B / B2 (exposure
  helped up to 28-36k, W4 compresses it to the same ceiling) and the earlier untargeted Granary additions: the next
  large gain is unlikely to come from the schedule or from statically re-sampling C250; it needs a different learning
  signal (E) and/or new, selected information (D). Phase C is closed at this granularity.
- Reusable: the per-clip student scores of the whole C250 pool (W4-C250-16k, 116,526 clips) and the bucket tables.

Artifacts (no Hub token on the box at close; deliberately not re-added for a null): local copy of probes, metrics, logs, bucket stats and the student
scores (`student_w4_16k.jsonl.gz`, 11.8 MB); to be uploaded to the Hub under results/ada-ee-2026-10-16/c1.

## Next session plan (2026-10-17; each arm pre-registered before launch, judged first on harness oracle@8)

- **D: D0 uniform-new vs D1 selected-new.** Score it-yg7 with `student_score.py` (W4-C250-16k) + the teacher, look
  at the DATA distribution only (teacher-good / student-bad / structurally credible: hours, durations, sources), then
  pre-register the amount. D1 = selected clips, D0 = random clips from the same it-yg7, duration-matched to D1, same
  physical hours, same external weight, same W4-8k / draws; so a D1 win is the selection, not yg7 or short clips.
- **E: encoder KD, audit first.** Same WAVs through teacher (Nemotron 0.6B) and student: T_teacher vs T_student,
  effective frame rate / stride and their ratio, hidden sizes, available teacher layers (early / mid / late
  statistics), temporal offset from subsampling / look-ahead; a similarity / alignment check per layer BEFORE choosing
  layer, projection (512 -> D_teacher) and loss.
- Target stays candidate generation: clean flushed oracle@8 13.71 -> <10-12 first, ranking (21.74 -> oracle) after.
- Box: restore with `resume_it.sh SNAPSHOT=it-c250`; upload today's C1 artifacts with the new (rotated) token.

## 2026-10-17 — Japan L40S (Vast instance 56378381; TLS fingerprints matched the dev machine, no proxy)

### Phase D design (pre-registered 2026-10-17 before any yg7 score is seen; the amount H is fixed after the bucket table)

Question: at the same new-audio hours, source, duration distribution, external draw share, steps and recipe, do
clips where a credible teacher succeeds and the 120M fails (D1) move the candidate set more than random new clips
(D0)?
- Pool: it-yg7 rebuilt from the same shard spread (it000/it100/it101 300/300/300 minus the it-yg / it-yg6 shards,
  500 h, 0 shard / video overlap with C250), FLAC; teacher pass (Nemotron 0.6B, forced it); `pseudo_subsets.py`
  C250 defaults -> `yg_all.json` (label = teacher text, digits dropped).
- Frozen scorer: W4-C250-16k (`runs/ada-it-w4-c250-16k/final.nemo`), `student_score.py`, greedy as the probe.
- Buckets (`d_select.py`, thresholds = curriculum_mix.py's, not tuned): E label flags or empty student; D teacher_wer
  >= 30 and student agrees with Granary-Whisper (<= 10); C teacher_wer > 15; B student <= 5; A teacher_wer <= 5 and
  student >= 30; M the rest. Credibility never uses the student.
- D1 = H h random from A. D0 = H h random from the student-free gate (teacher_wer <= 5, no label flags), matched to
  D1 per duration bin (1-2-4-8-15-30 s); overlap reported. Seed 20261017.
- Mix: A2 weights unchanged (mls 0.1964, cv 0.1684, vp_t50 0.1122, fleurs 0.0842); external W 0.4388 split over
  yg / yg6 / D set in proportion to hours (D set counted as H in both arms: identical weights).
- Training: W4 8k (enc 4e-4, dec/joint 3e-4, WSD decay 0.2, bs 16, seed 0, max-dur 30), stock init, sparse evals
  (100, 250, 1000; every 2000 flat / 500 decay), final.nemo = EOF. Order: D0 then D1; D1 is not changed after D0.
- Primary: harness oracle@8 D1 vs D0 (reference W4-C250-8k 26.12 / 23.67 / 19.43). Reading:
  D1 <= D0 - 1.0 and <= 18.4 -> selection helps, promote D1 to 16k; D1 < D0 - 1.0 but > 18.4 -> selection signal,
  per-domain check before promotion; |D1 - D0| < 1.0 -> selection heuristic null (no threshold tuning); both <= 18.4
  vs the reference while D0 ~ D1 -> new yg7 exposure matters, selection less; neither beats the reference by > 0.5
  -> simple extra-data route weak, compute to E. Differences < 0.5 are not read. Clean-it diagnostic only.

### Phase E0 result: teacher / student encoder audit (`kd_audit.py`, OBSERVED; no training)

Teacher Nemotron 3.5 streaming 0.6B (`EncDecRNNTBPEModelWithPrompt`, 638M) vs student EOU 120M stock (115M):

| | EOU 120M | Nemotron 0.6B |
|---|---:|---:|
| preprocessor | 16 kHz, 128 mel, 25 / 10 ms, n_fft 512, no norm | identical |
| encoder layers / d_model | 17 / 512 | 24 / 1024 |
| subsampling | dw_striding x8, causal | identical |
| frame | 80 ms | 80 ms |
| T at 1 / 5 / 10 / 20 s | 14 / 64 / 126 / 251 | 14 / 64 / 126 / 251 |
| att_context | [70, 1] | [56, 3] default (0 / 3 / 6 / 13 selectable) |
| encoder fwd 16 x 10 s bf16 (uncontended) | 19.6 ms | 37.5 ms |

- T_teacher == T_student on every clip of every audit (up to 300 real clips + synthetic 0.4-20 s; contexts [56,3] and [56,0]): no interpolation /
  DTW; frame-to-frame 1:1. The language prompt enters AFTER the encoder (`prompt_kernel` on the encoder output):
  the teacher encoder states are prompt-free.
- Linear alignability (standardised ridge, fitted on half the clips of each domain, scored held out; 248 training
  clips MLS / CV / VP / YG, 60 each; diagnostic only, not "information content"): held-out R^2 at the best shift

| student \ teacher | L5 | L11 | L17 | L23 |
|---|---|---|---|---|
| mel stack (acoustic baseline) | .070 (+1) | .070 (+1) | .053 (+2) | .018 |
| L4 | .509 (0) | .347 (0) | .333 (0) | .123 |
| **L7** | .288 (0) | .285 (0) | **.434 (0)** | .158 |
| L12 | .194 (-1) | .270 (-1) | .349 (-1) | .159 |
| L16 (final) | .087 (-4) | .153 (-4) | .208 (-4) | .157 (0) |

  S7 -> T17 per domain (held out): MLS .449, CV .462, VP .426, YG .395 (consistent across domains); R^2 at shift
  -1 / 0 / +1: .381 / .434 / .339 (sharp peak at 0). The teacher's final layer is weakly predictable from anything
  (task / tokenizer specific); the student's final layer maps poorly and lags (best shift -4).
- **Frozen for E1: student L7 (0-based) -> teacher L17 (0-based), shift 0, teacher default context [56, 3]**
  (intermediate -> intermediate; S4 -> T5 is higher but early-to-early, near-acoustic). The ridge probe is NOT reused:
  E1's projection is a fresh random-init Linear(512, 1024).
- Precompute estimate: 8.6 GB per 100 h for one teacher layer in bf16; on-the-fly teacher costs ~+40 ms / batch.
- Fallback (E2, a distinct hypothesis, only if E1 S7 -> T17 is stable but null): early-layer distillation S4 -> T5
  (highest R^2, .509, but early-to-early and near-acoustic).
- Teacher context check (S7 -> T17, same 248 clips): [56,0] .445, [56,3] .434, [56,6] .422, [56,13] .423, best
  shift 0 in all four, T equal in all four -> the context does not change the pair or the alignment; the frozen
  default [56,3] stays. E0 closed.

### Path to EOU-IT v4 (plan, 2026-10-17; after D / E1 read)

Success = FINAL-IT (new, untouched) 1-best < 20 WER, EOU success >= 98-99 %, plain -> EOU delta 0..+1 WER, Mynah
parity, serving advantage kept (120M, ~2 GB, true streaming); 15-18 WER = top band; ~10-12 = stretch.
1. Promote the one real D / KD win (if a probe justifies it) to a sufficient budget.
2. KD + selected data together only if the effects are clearly additive.
3. If oracle@8 reaches ~10-12 while 1-best stays 17-20: switch axis to ranking / sequence (N-best) distillation,
   not more encoder steps.
4. Freeze the best plain; 5. Stage 2 EOU with the conservative v3 recipe (decoder / joint LR / 5), gate on the
   plain -> EOU WER delta, not the EOU rate alone; 6. Mynah export + parity; 7. FINAL-IT built before the last
   choice and used once: greedy / beam WER + CER, empties, EOU precision / recall, endpoint latency, streaming lag,
   failures by duration / domain; L4 / L40S serving benchmark.

### Phase D data (OBSERVED, before any D training) and the frozen D0 / D1 (pre-registration, 2026-10-17 ~13:20 UTC)

it-yg7 rebuilt exactly (156,247 clips / 500.7 h / 9,441 videos / 201 shards, 0 video / shard overlap with it-yg and
it-yg6, max 30 s); teacher pass complete; `yg_all` (digits dropped) 142,080 clips / 445.5 h. Duplicates removed
before bucketing (`pool_sanity.py --xhash`, decoded samples): 81 within-pool groups (re-uploaded videos, first kept)
+ 12 / 23 clips equal to it-yg / it-yg6 agree<=15 clips -> 107 dropped. Frozen scorer W4-C250-16k.

| bucket | clips | h | dur p10/p50/p90 | student WER | teacher vs Granary | empty % | hyp/ref | videos | top-10 % videos' h |
|---|---|---|---|---|---|---|---|---|---|
| A learnable | 4,535 | 7.86 | 1.7 / 4.2 / 13.7 | 43.4 | 0.8 | 0.0 | 0.94 | 2,550 | 33 % |
| B solved | 10,987 | 27.4 | 2.4 / 7.1 / 18.9 | 1.2 | 2.9 | 0.0 | 1.00 | 4,160 | 40 % |
| M mid | 66,823 | 247.0 | 3.9 / 11.5 / 26.0 | 19.8 | 6.7 | 0.0 | 0.97 | 8,351 | 45 % |
| C ambiguous | 53,538 | 147.0 | 2.0 / 7.0 / 23.2 | 43.4 | 33.3 | 0.0 | 0.96 | 7,146 | 49 % |
| D suspicious | 19 | 0.03 | - | 45.8 | 44.1 | - | - | 19 | - |
| E excluded | 6,071 | 15.8 | 1.5 / 6.4 / 23.2 | 88.3 | 44.0 | 11.2 | 1.00 | 2,779 | 39 % |

Student-free gate (teacher_wer <= 5, no label flags): 116 h. Only 7.9 of the 116 credible hours are hard for the
16k scorer (short-clip heavy: A median 4.2 s vs pool 8.9 s); the scorer already handles most unseen-video yg7.

Design amendment (registered now, before any D result; A NOT widened, thresholds unchanged): H = 7.86 h = all of A.
With the hours-proportional weight the D set would get ~1.5 % of the draws (< 0.5 pass): no power. Instead the D set
gets a FIXED draw share 0.10, taken from the external weight (yg 0.1455, yg6 0.1933; A2 unchanged).
Question tested: does a fixed 10 % training budget allocated to a small new-data subset selected by teacher-supported
student difficulty beat the identical budget on a random, duration-matched subset of the same deduplicated,
credibility-gated it-yg7 universe?

| | clips | h | dur p10/p50/p90 | videos | top-10 % videos' h | draw share | expected draws / clip | student WER | teacher vs Granary |
|---|---|---|---|---|---|---|---|---|---|
| D0 (random, gate, per-bin matched) | 4,395 | 7.86 | 1.7 / 4.3 / 14.0 | 2,561 | 33.8 % | 0.10 | 2.91 | 18.1 | 0.7 |
| D1 (all of A) | 4,534 | 7.86 | 1.7 / 4.2 / 13.7 | 2,549 | 32.9 % | 0.10 | 2.82 | 43.4 | 0.8 |

Overlap D0 / D1: 849 clips (18.7 % of D1; mostly short bins where A is ~1/3 of the gate). sha256: D0.json f3c9a54f,
D1.json e6172110, mix_D0 262567a0, mix_D1 a073d0b8, yg_all 9ed5bb09, student scores 6bd9e14b, scorer b13e4e7e,
dup groups 4281b425. Runs: `c1_curriculum.sh` with MIXFILE = mix_D0 / mix_D1, W4 8k (enc 4e-4, dec/joint 3e-4, WSD
0.2, bs 16, seed 0, max-dur 30), stock init, sparse evals, final.nemo = EOF; D0 first, D1 not changed after D0.
Primary = harness oracle@8 (reference W4-C250-8k 19.43):
- D1 <= D0 - 1.0 -> selection win; additionally <= 18.4 -> strong selection win (16k candidate);
- |D1 - D0| < 1.0 -> selection null (no threshold tuning afterwards);
- D0 and D1 both clearly below 19.43 (> 0.5) -> new yg7 exposure useful, selection not needed;
- D1 better on greedy / beam but not oracle -> not a candidate-generation win;
- deltas < 0.5 are not read; a -0.3 is never promoted to 16k. Clean-it diagnostic only.

### Phase E1-short pre-registration (2026-10-17 ~14:25 UTC, before any E1 training)

KD gradient calibration (`two_stage.py --kd-calib 16`, same 16 C250 batches / seed, fresh random projection, no
optimizer step, no eval, no WER):

| student state | median ‖g_RNNT‖ enc | median ‖g_KD‖ enc (lambda 1) | median ratio | ratio of summed grads | grad cosine (mean / summed) | RNNT / KD loss |
|---|---|---|---|---|---|---|
| stock (step 0) | ~5,300 | 0.150 | 4e-5 | 4e-5 | 0.008 / -0.001 | 508 / 0.998 |
| W4-C250-16k | 92.3 | 0.1271 | 0.00138 | 0.0049 | -0.009 / -0.007 | 54 / 0.997 |

The stock step 0 is a transient (in-run encoder grad norm ~1,460 at step 25, ~100-185 from step 100 on, C1 logs):
not used. Rule (unchanged): lambda0 = 0.375 x median ‖g_RNNT‖ / median ‖g_KD‖ on the adapted student = 272 ->
**lambda0 = 270**, lambda(s) = 270 x (1 - s / 2000). The KD and RNNT gradients are ~orthogonal (complementary).
The summed-gradient ratio (0.0049 -> lambda ~76) is a diagnostic only. No lambda sweep.

Arms (`e1_kd.sh`, run in this order after D1): ctl = W4-C250 (it-yg + it-yg6 agree<=15, W 0.4388), FIX_STEPS 2000,
W4 recipe, seed 0, sparse evals (500 / 250), final.nemo = EOF; kd = the same + `--kd-teacher` Nemotron 0.6B, student
L7 -> fresh Linear(512, 1024) -> teacher L17 (default context [56, 3]), 1 - cos over valid frames, teacher on the
student's pre-SpecAugment features (eval, no grad, bf16), lambda as above. Only variable: the KD term.
Stop rules: KD smoke peak reserved within ~2 GB of the 44.4 GB card -> stop (no batch / duration change in one arm);
NaN / divergence -> stop.
Reading (mechanistic probe, not a WER claim; harness oracle@8 primary, EOF):
- healthy = KD loss falls and the cosine rises clearly (> +0.1 over its step-25 value), RNNT train loss at EOF
  not worse than ctl by > 3 % (mean of the last 200 logged steps), encoder grad norm in ctl's range, step-time
  overhead reported;
- promote to a pre-registered E1-8k if healthy AND harness oracle@8 <= ctl + 0.5 (not materially worse);
  oracle <= ctl - 1.0 at 2k = strong early signal; greedy / beam / CER / empties reported alongside;
- stop (KD null at this layer pair) if unhealthy, or oracle > ctl + 0.5; fallback E2 = S4 -> T5 (separate
  pre-registration).

### Phase D result: D1 (selected) vs D0 (random) (OBSERVED, one seed, same box; EOF; greedy / beam-8 / oracle@8)

| | harness raw | harness 0.5 s flush | clean raw | clean 0.5 s flush |
|---|---|---|---|---|
| D0 random | 26.56 / 24.29 / **19.87** | 26.55 / 24.30 / 19.79 | 27.28 / 25.23 / 18.27 | 25.86 / 23.80 / 17.27 |
| D1 selected | 26.87 / 24.74 / **20.41** | 26.97 / 24.87 / 20.46 | 27.23 / 24.94 / 18.05 | 26.27 / 24.08 / 17.45 |
| ref W4-C250-8k | 26.12 / 23.67 / 19.43 | | | |

Harness per set, greedy (oracle@8): D0 MLS 25.16 (20.06), FLEURS 19.57 (14.78), CV 29.36 (18.64), VP 32.16 (26.00);
D1 MLS 24.71 (20.16), FLEURS 20.24 (15.27), CV 28.20 (19.17), VP 34.33 (27.05). Harness CER 10.22 vs 10.51.
In-run harness val at 8k: 26.56 vs 26.87.

- **Pre-registered reading: selection NULL** (D1 - D0 oracle@8 = +0.54, inside +-1.0; the sign is against D1).
  D0 itself is null vs the reference (+0.44): a fixed 10 % of draws on 7.9 h of credible random yg7 does not help.
- Reading (scoped): spending a fixed 10 % budget on the clips where the W4-16k scorer fails but the two-teacher
  agreement is high (short-clip heavy, 2.8 passes) does not improve candidate generation at 8k vs the same budget
  on random credible clips; if anything it costs a little (VP greedy +2.2). With C1 (static difficulty re-weighting
  inside C250, null) this closes static student-difficulty selection / re-weighting at this scale. Not tested:
  larger selected sets from a bigger pool, dynamic / loss-aware selection.
- Compute moves to E (encoder KD).
