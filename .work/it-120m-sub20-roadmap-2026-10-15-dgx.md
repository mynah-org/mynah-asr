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

### Phase E1-short result: encoder KD S7 -> T17 vs its exact control, 2k (OBSERVED, one seed, same box; EOF)

| greedy / beam-8 / oracle@8 | harness raw | harness 0.5 s flush | clean raw | clean 0.5 s flush |
|---|---|---|---|---|
| ctl 2k | 35.59 / 32.79 / 28.47 | 35.61 / 32.79 / 28.46 | 36.73 / 33.77 / 25.55 | 35.57 / 32.59 / 24.59 |
| kd 2k (lambda0 270) | 35.70 / 32.62 / **28.32** | 35.79 / 32.69 / 28.45 | 36.80 / 33.75 / 25.87 | 35.82 / 32.76 / 25.12 |

Harness CER greedy 13.07 -> 12.61 (-0.46), beam 11.28 -> 11.21. Per set greedy (oracle@8): ctl MLS 33.20 (27.74),
FLEURS 29.08 (23.83), CV 38.39 (28.15), VP 41.68 (34.17); kd MLS 33.49 (28.03), FLEURS 30.24 (24.41), CV 38.97
(28.05), VP 40.10 (32.78). In-run harness greedy at 2k 35.60 vs 35.72.

Mechanism (training logs): cosine S7 -> T17 0.05 (step 25) -> 0.77 (225) -> 0.85 (1k) -> 0.86 (EOF, plateau from
~1.2k); KD loss 0.95 -> 0.14; projection grad norm 102 -> 0.5 as lambda -> 0; RNNT loss mean of the last 200 steps
71.6 vs ctl 73.6 (not obstructed); encoder grad norm median 129 vs 133; peak 36.1 / 39.9 GB vs 34.8 / 38.6;
wall +19 % (1385 vs 1166 s incl. evals; teacher on the fly).

- **Pre-registered reading: healthy; oracle@8 -0.15 vs ctl (<= ctl + 0.5) -> promote to a pre-registered E1-8k;
  NOT a strong early signal (needs <= ctl - 1.0).** Every task delta is under 0.5 (not read) except CER -0.46.
- Reading (scoped): at 2k the student L7 can be driven close to the teacher L17 (cos 0.86) without obstructing
  RNNT, at +19 % wall; that alignment has not (yet) moved the 2k candidate set. Open: whether a full 8k schedule
  (lambda over 8k) transfers task-level information, or representation matching and task transfer are decoupled
  for this pair / loss.

## Next session plan (2026-10-18)

- **E1-8k (pre-registered now, before launch):** `e1_kd.sh ARM=kd STEPS=8000 KD_LAMBDA=270 KD_END=1.0`
  (lambda(s) = 270 x (1 - s / 8000); everything else = E1-short), vs the existing W4-C250-8k control 26.12 / 23.67 /
  19.43 -- BUT that control used the old save-best path; for an exact A/B also run `ARM=ctl STEPS=8000` (same
  code path, sparse evals, EOF), ~65 min. Primary harness oracle@8 kd vs ctl-8k (<= ctl - 1.0 = KD moves the
  candidate set; <= 18.4 = strong; |delta| < 0.5 = null); greedy / beam / CER / per domain alongside. E1-8k is a
  BRANCH POINT, decided by which of the three moves:
  - oracle AND beam / greedy improve together -> plain win: validate / promote (16k), then stage 2 EOU;
  - oracle improves (>= 1.0) but beam / greedy stay ~flat -> the representation transfer works and the bottleneck is
    ranking: stop encoder-KD variants, move to ranking / N-best / sequence distillation;
  - oracle null -> the candidate set did not change, ranking is NOT the answer yet: change the representation
    objective / layer (E2 = S4 -> T5, the highest-R^2 pair in E0), not the decoder.
- Teacher cost: on the fly +19 % wall at 2k; precompute only if 8k+ runs multiply (8.6 GB / 100 h one layer bf16).
- FINAL-IT: inclusion criteria, speaker / source disjointness from every pool and bank, manifest sha256 committed
  first; then sealed (no model scored on it until the final candidate).
- Restore: `SNAPSHOT=it-c250 resume_it.sh`; the it-yg7 pool + teacher / student scores are in results/l40s-jp-2026-10-17
  (audio rebuild: `d_yg7.sh` download only, the teacher pass is skipped when train_yg.teacher.json is restored).
- Rotate the HF token (exposed in this session's transcript).

## 2026-10-18 — Japan L40S (Vast, same host 61.206.39.5, new instance; empty disk; TLS issuers public)

Mission: converge to a production EOU-IT v4 (IT first, FR transfer after the IT recipe freezes). Order: E1-8k
A/B (frozen below) on the GPU, FINAL-IT built + sealed on CPU in parallel, then conservative stage 2 on the best
plain parent, FINAL-IT once, Mynah export + serving.

### FINAL-IT pre-registration (2026-10-18 ~12:10 UTC, before any clip is selected; `eou/prepare_final_it.py`)

Purpose: the one-shot final test of the frozen candidates (plain parent, EOU-v4, Nemotron as context). The old
clean-it bank stays a DEV bank (it has been read many times).
- **Sources = six sub-banks, each from a split no run of ours trained on, selected on or was scored on:**
  MLS-it dev; Common Voice 17 it validation; VoxPopuli it validation; FLEURS it_it dev; Multilingual TEDx it
  TRAIN shards 0-1 (talks absent from mTEDx test / valid = the clean dev talks, and absent from EVERY Granary-it
  original_audio_id, not only our pools); ITALIC hard_speaker validation.
- **Disjointness:** MLS dev speakers asserted absent from MLS train + test; CV clips whose client_id occurs in CV
  train or test dropped; VP clips whose speaker_id occurs in VP train or test dropped; FLEURS dev sentence ids
  asserted absent from train / test (FLEURS has no speaker id: one reading per sentence); ITALIC validation
  speakers asserted absent from hard_speaker test (the clean dev speakers); normalised text absent from every
  training pool and dev bank manifest on the box; decoded-sample duplicates vs C250 / yg7 pools checked with
  `pool_sanity.py --xhash` (any hit dropped, count reported).
- **Clip rules (all banks):** 1-20 s; no digit and none of []()&<>; >= 2 words (tedx >= 3); natural ending =
  whole utterances (read / dictated corpora) and tedx segments ending in . ? ! or ...; text unique in FINAL-IT.
- **Selection:** candidates in sha1("<bank>:<id>") order, round-robin over speakers / talks, <= 400 clips per bank
  (tedx <= 25 per talk). Model-free: no ASR / teacher / student output, WER or manual listening is used; the only
  audio measurement is an energy endpoint (20 ms frames, last frame within 35 dB of the clip peak) giving
  `trail_s`, recorded, never filtered on.
- **EOF variant:** `final_trim_<bank>` = the same clips cut at the energy endpoint + 50 ms, so the finalisation /
  flush policy is measured on clips WITHOUT trailing silence instead of being inferred from clips that already
  carry it.
- **Seal:** sha256 of every manifest and wav (SHA256SUMS), the SHA256SUMS hash committed here, set uploaded to the
  private Hub repo (`data/final-it/`; mTEDx is CC BY-NC-ND: private evaluation only). No model is run on it until
  the Phase-4 candidates (model + decoding + flush policy) are frozen in writing.
- **Primary metric (Phase 4):** FINAL-IT WER = unweighted macro over the six banks, 1-best greedy, the frozen
  production finalisation policy, on the natural clips; beam-8, CER, empty rate, per bank, the trim variant, EOU
  detection and endpoint latency reported alongside. Bands: < 20 = v4 success, 15-18 strong, ~10-12 stretch.
  Caveat for the Nemotron reference: MLS / CV / VP / FLEURS (train splits at least) are in its training data.

### E1-8k (the 2026-10-17 pre-registration, unchanged; decision rule made explicit before launch)

Arms in order: `e1_kd.sh ARM=ctl STEPS=8000`, then `ARM=kd STEPS=8000 KD_LAMBDA=270 KD_END=1.0` (lambda(s) = 270 x
(1 - s / 8000); S7 -> fresh Linear(512, 1024) -> T17, Nemotron 0.6B default context, 1 - cos); same code path, data
(C250 mix), seed 0, sparse evals, final.nemo = EOF; probes harness + clean, greedy / beam-8 / oracle@8, raw + 0.5 s
flush. lambda is not touched after any partial number; stop only on NaN / divergence / OOM. Deltas kd - ctl on harness
raw EOF:
- **plain win:** oracle@8 <= -1.0 AND beam-8 <= -0.5 AND greedy <= -0.5 -> promote KD (no other KD layer), staged
  16k next (vs W4-C250-16k 22.76 / 21.02 / 16.71), then stage 2;
- **oracle-only win:** oracle@8 <= -1.0 with beam-8 and greedy within +-0.5 -> stop encoder-KD variants; one bounded
  ranking / N-best experiment;
- **null:** oracle@8 > -1.0 -> close E1; ONE bounded E2 = S4 -> T5 (same discipline, own control = ctl-8k); if E2 is
  null too, cosine feature KD is closed (no layer sweep).
In every branch the v4 production path continues in parallel: plain parent = C250-36k unless KD produces a
clearly superior plain model.

### Stage 2 v4 pre-registration (2026-10-18 ~12:35 UTC, before launch): conservative EOU-v3 recipe on C250-36k

Production path, independent of the E1 outcome (if KD later yields a clearly better plain parent, the SAME
recipe is re-applied to it). Run `s2_mix.sh` TAG=it-s2v4-c250-36k: init `plain-it-c250-36k/final.nemo`; the
EOU-v3 recipe unchanged (text + <EOU>, 50 % of samples with 1-3 s trailing quiet, top-4 encoder at 3e-5, decoder +
joint 2e-5 = the djlr fix of the v3 VP blanking, bs 16, 1500 steps, FastEmit 0, max-dur 20, evals every 250 on the
four harness sets, final.nemo = min WER among evals with EOU >= 80 % and <= 10 % empty); the only adaptation is
the principle s2_mix.sh already follows: stage 2 on the parent's OWN mix = the C250 mix (MLS 0.1964, CV 0.1684,
VP-t50 0.1122, FLEURS 0.0842, it-yg 0.1885, it-yg6 0.2503). GPU order today: this first (~1 h incl. probes), then
E1 ctl-8k, kd-8k. Probes (lm_fusion_probe, <EOU>/<EOB> dropped from the text) of the stage-2 model AND the parent
on the same box: harness + clean, greedy / beam-8 / oracle-8, raw and 0.5 s flush.
Gates (delta = stage 2 - parent, harness raw EOF, same box):
- **v4 candidate** if greedy <= +1.0 AND beam-8 <= +1.0 AND no harness domain greedy > +2.0 AND in-run EOU rate
  at the selected step >= 98 % AND empty clips <= parent + 1 % abs; clean (dev) and flushed numbers reported;
- **diagnose before promoting** (per domain, empties, EOU placement) if +1.0 < greedy <= +2.0;
- **not v4** if greedy > +2.0 or EOU < 95 %: the stage-2 update pays back the plain gains; a next arm is
  pre-registered separately (not tuned on these numbers).
Then the frozen candidate goes to Mynah (m1.sh: export, FLEURS-it 200 stream WER / EOU / endpoint latency, CPU /
CUDA parity) and only then to FINAL-IT.

FINAL-IT amendment (2026-10-18 ~12:47 UTC; from selection COUNTS only, no model has decoded any clip): dropping every
VP validation clip whose speaker occurs anywhere in the VP train / test splits left 110 clips from 3 speakers (VP's
splits are not speaker-disjoint: 1,124 of 1,257 clips dropped) -- too narrow for a 1/6 macro weight. Rule now:
VP speakers absent from OUR VP pools and bank (`train_vp.json` = every VP clip ever drawn for training, superset of
t50, and the harness `eval_vp.json`), i.e. disjoint from what our models saw, as for the other sources. The set is
rebuilt from scratch with this rule (deterministic). FLEURS stays at its natural size (116 unique dev sentences
<= 20 s without digits).

### Result: stage 2 v4 on C250-36k (OBSERVED, Singapore L40S, same box re-probe of the parent)

Run 875 s; fixed rule picked step 250 (in-run harness 23.89, EOU 99.5 %, 0 empty; later evals worse: 26.23 /
25.08 / 27.47 / 25.64 / 25.51 at 500-1500). Parent re-probe reproduces 2026-10-15 exactly (22.27 / 20.53 / 16.25).

| greedy / beam-8 / oracle@8 | parent C250-36k | stage 2 v4 @250 | delta |
|---|---|---|---|
| harness raw | 22.27 / 20.53 / 16.25 | 24.14 / 22.64 / 18.01 | +1.87 / +2.11 / +1.76 |
| harness 0.5 s flush | 22.21 / 20.57 / 16.27 | 23.89 / 22.49 / 17.89 | +1.68 / +1.93 / +1.63 |
| clean raw (dev) | 23.73 / 22.00 / 15.11 | 25.35 / 23.48 / 16.08 | +1.62 / +1.48 / +0.97 |
| clean 0.5 s flush (dev) | 21.74 / 20.23 / 13.71 | 23.18 / 21.37 / 14.54 | +1.44 / +1.14 / +0.82 |

Greedy per set: MLS +2.85, FLEURS +2.71, CV -0.10, VP +2.04. **Pre-registered reading: NOT a v4 candidate**
(greedy +1.87 is in the diagnose band, beam +2.11 > +1.0, three domains > +2.0; EOU rate passes).

Diagnosis (CPU, offline greedy, 200 clips / set; `s2_diag.py`, `s2_frames.py`):
- the loss is end-of-clip deletions (MLS D_end 0.20 -> 2.46 % of ref words, FLEURS 0.02 -> 0.90, VP 2.21 -> 3.26);
  CV unchanged; flushes of 1.0 / 2.0 s change nothing (not an EOF-flush problem); no MLS clip > 20 s (not max-dur);
- component swap: parent encoder + s2 decoder/joint 21.77 / 15.32 / 27.55 (MLS / FLEURS / VP), s2 encoder +
  parent decoder/joint 22.14 / 15.85 / 28.53, neither truncates: the failure needs the s2 encoder (pause read as
  an end) AND the s2 decoder/joint;
- all 15 pathological clips (s2 drops >= 3 final words, parent none): an <EOU> emitted at an INTERNAL pause (median
  0.32 s, 0.08-2.1 s) with 3-18 s of speech left; after it s2 sits at P(blank) 0.988, best text-token prob 0.000 on
  every remaining frame (EOU max 0.97), while the parent keeps emitting (P(text) ~0.37, 5-93 tokens on the same
  frames). The "premature EOU" counter of s2_diag missed it (it required text AFTER the EOU).
- **Candidate cause: premature <EOU> at internal pauses + an absorbing blank state after <EOU>** (stage-2 targets
  never contain text after <EOU>). Offline NeMo decoding has no reset; Mynah resets the decoder after every model
  <EOU>, so part of this WER may be an offline-evaluation artefact while the real product cost is premature
  endpoints. Next (CPU, not run): offline decode with a decoder reset after <EOU> to split the two. No stage-2 v5
  launched today (E1-8k keeps the GPU).

Reset test (`s2_reset.py`, CPU, offline encoder, frame-synchronous greedy; decoder reset after every <EOU> as Mynah
does, encoder untouched): harness greedy macro parent 22.29 / s2 no reset 24.14 / **s2 with reset 22.77** (MLS 21.45 /
24.29 / 22.27, FLEURS 15.21 / 17.92 / 15.96, CV 24.37 / 24.16 / 24.16, VP 28.14 / 30.18 / 28.70); words recovered by
the reset 131 / 91 / 0 / 45; end deletions back to the parent's level. Premature <EOU> (>= 0.5 s of speech left):
MLS 12 / 200 clips, FLEURS 34 / 200, CV 7 / 200, VP 8 / 130 = 61 / 730 (8.4 %).
- ~74 % of the offline +1.85 is a decoding-protocol mismatch (no reset): ASR continues after the false EOU; the
  remaining ASR cost of this stage 2 is ~+0.5 greedy (MLS +0.8, FLEURS +0.75, VP +0.56, CV 0).
- The product problem is EOU PRECISION: ~8 % of clips (17 % FLEURS) get a false endpoint at an internal pause
  (0.08-2 s). Two metrics from now on: ASR continuity under the serving protocol, and the false-endpoint rate.
  A stage-2 v5 must cut false EOUs at internal pauses without losing true-endpoint recall; not launched today.
- The DEV probes of stage-2 models must use the serving protocol (reset after <EOU>) from now on.

### Result: E1 ctl-8k (OBSERVED, Singapore L40S, same code path as kd-8k; EOF)

Harness raw 26.32 / 24.27 / **19.84** (CER 10.22; MLS 24.52 (19.87), FLEURS 18.65 (14.24), CV 29.67 (19.49), VP 32.45
(25.77)); 0.5 s flush 26.36 / 24.39 / 19.84; clean raw 27.06 / 24.73 / 17.43, flushed 26.12 / 23.77 / 16.82. Wall
~77 min incl. evals (13:58-14:15 UTC train end), peak 36.5 / 40.2 GB. kd-8k launched 14:22 UTC.

### Result: E1-8k KD vs ctl-8k (OBSERVED, one seed, same box / code path; EOF) -- E1 NULL

| greedy / beam-8 / oracle@8 | harness raw | harness 0.5 s flush | clean raw (dev) | clean flushed (dev) |
|---|---|---|---|---|
| ctl-8k | 26.32 / 24.27 / 19.84 | 26.36 / 24.39 / 19.84 | 27.06 / 24.73 / 17.43 | 26.12 / 23.77 / 16.82 |
| kd-8k (lambda0 270 -> 0) | 26.43 / 23.88 / **19.85** | 26.48 / 23.96 / 19.91 | 27.06 / 25.01 / 17.85 | 26.23 / 24.17 / 17.30 |
| kd - ctl | +0.11 / -0.39 / +0.01 | +0.12 / -0.43 / +0.07 | 0.00 / +0.28 / +0.42 | +0.11 / +0.40 / +0.48 |

Per set greedy (oracle) kd: MLS 25.56 (20.26), FLEURS 19.38 (14.43), CV 28.31 (19.28), VP 32.49 (25.44); ctl: 24.52
(19.87), 18.65 (14.24), 29.67 (19.49), 32.45 (25.77). Harness CER greedy 10.13 vs 10.22, beam 8.85 vs 8.97.
Mechanism: cos S7 -> T17 0.84 at 1k, ~0.87 plateau, 0.88 at EOF; RNNT not obstructed; teacher 50-85 ms / step; wall
5832 s vs ~4620 s (+26 %), peak 37.9 / 42.2 GB. In-run greedy kd was 0.5-1.2 behind ctl at 2-3k (not acted on).
- **Pre-registered reading: NULL** (oracle@8 +0.01 > -1.0; greedy / beam inside +-0.5). E1 (cosine feature KD
  S7 -> T17, Nemotron 0.6B) closed: the student layer matches the teacher closely (cos ~0.87) without moving the
  candidate set or the 1-best at 8k. Per the tree the next KD step would be ONE bounded E2 (S4 -> T5); NOT launched:
  the next GPU experiment is chosen by information gain against stage-2 v5-B (see endpoint audit below).

### Stage-2 endpoint audit (model-free, `s2_endpoints.py`, 600 clips / source, energy mask on 80 ms frames; aggregate only)

Code facts (two_stage.py): every stage-2 target = text + <EOU>; pad p 0.5, U(1, 3) s near-digital quiet (-90..-60 dB);
--concat-prob 0 (A + pause + B exists, unused); no augmentation in stage 2.

| source | natural tail p10/p50/p90 s | ends on speech (< 0.1 s) | tail < 0.5 s | trained EOU tails < 0.5 s | pauses 0.08-2.1 s -> EOU vs -> speech |
|---|---|---|---|---|---|
| MLS | 0.16 / 0.40 / 0.80 | 3.8 % | 69 % | 34 % | 292 vs 5,737 |
| CV | 0.16 / 0.56 / 1.12 | 9.7 % | 49 % | 25 % | 281 vs 1,132 |
| VP | 0.00 / 0.16 / 0.48 | 44 % | 93 % | 45 % | 216 vs 10,180 |
| FLEURS | 0.32 / 0.72 / 2.48 | 6.2 % | 38 % | 20 % | 234 vs 2,921 |
| Granary yg | 0.00 / 0.24 / 1.60 | 42 % | 72 % | 37 % | 184 vs 7,272 |
| Granary yg6 | 0.00 / 0.08 / 0.88 | 58 % | 81 % | 40 % | 157 vs 6,509 |

Premature-EOU pauses (15 pathological clips) 0.08-2.08 s, median 0.32; 51 % of natural training tails are <= 0.32 s.
Reading: hypothesis "stage 2 teaches pause / cut = EOU" PARTIALLY SUPPORTED: ~20-45 % of trained <EOU> labels have
no real trailing silence (VP / Granary clips end ON speech in 42-58 % of unpadded cases) -> acoustically ambiguous
positives; NOT a lack of negatives (pauses followed by speech outnumber pause -> EOU 4-50x in the same range).
Best-supported candidate cause (not demonstrated): ambiguous positive EOU labels. v5-B (EOU only on credible
endpoints, model-free rule, same exposure) would test it; v5-C (A + pause + B) and NVIDIA-style padding (A) later.

### Stage-2 v5-B pre-registration (2026-10-18 ~16:20 UTC, before launch; E2 parked)

Question: does removing acoustically ambiguous POSITIVE <EOU> supervision reduce premature internal-pause endpoints
without sacrificing true-endpoint recall / latency or ASR?
- **Acoustic endpoint credibility (model-free, fixed before any v5 output):** an unpadded training sample is a
  credible acoustic endpoint iff its natural trailing non-speech `trail_s` >= 0.5 s (`annotate_tail.py`: energy
  mask on 80 ms frames, speech = within 35 dB of the clip peak and >= 10 dB above its 10th-percentile frame).
  0.5 s is set a priori: the shortest pause we would accept as an end of turn in production (published recipes:
  NVIDIA asr_eou never trains <EOU> with < 3 s after it; arXiv 2609.04225 uses >= 0.3 s PLUS lexical completeness,
  which we cannot check, so a stricter acoustic margin). It is acoustic credibility, NOT semantic completion. Not
  chosen with WER, EOU predictions, the pathological set or FINAL-IT.
- **Treatment (one variable):** `two_stage.py --eou-min-tail 0.5`: an UNPADDED sample with trail_s < 0.5 gets the
  text target WITHOUT <EOU>; padded samples (p 0.5, U(1, 3) s quiet, unchanged) and credible ends keep <EOU>.
  Same parent (C250-36k), mix rows and weights (annotated copies of the same manifests, same order), seed, LR
  (top-4 enc 3e-5, dec/joint 2e-5), bs, 1500 steps, evals, selection rule, max-dur 20; the random-call order of the
  padding loop is unchanged, so the audio batches are those of v4 and only labels differ.
- Pool after annotation (rows <= 20 s; accepted = trail_s >= 0.5): MLS 9,683 / 29.7 %, CV 26,800 / 51.8 %, VP
  10,955 / 5.6 %, FLEURS 2,375 / 60.9 %, yg 22,875 / 27.7 %, yg6 29,914 / 19.1 %; all 102,602 clips 251.0 h, 30.1 %
  (67.2 h) accepted; accepted / rejected duration distributions similar except FLEURS / CV (accepted slightly longer).
  Exposure: identical draws; share of samples whose target ends in <EOU> 100 % (v4) -> ~65 % (0.5 padded + 0.5 x
  0.303 draw-weighted accepted); VP keeps <EOU> only on its padded half.
- **Evaluation protocol (both v4 and v5-B, DEV harness only, FINAL-IT sealed):** `s2_reset.py --post-pad-s 2.0`:
  730 harness clips (MLS / FLEURS / CV 200, VP 130), offline encoder, frame-synchronous greedy with the Mynah decoder
  reset after <EOU>, 2 s of -75 dB quiet appended; FE = % clips with a premature <EOU> (>= 0.5 s of speech left,
  energy endpoint of the original clip), R = % clips with an <EOU> at >= speech end - 0.5 s, L = latency of that
  first true <EOU> (p50 / p90), W = greedy WER macro with reset; CER, empties alongside; no-reset WER and the usual
  probes informational.
- **Gates (v5-B vs v4, same protocol):**
  - PASS: FE <= 0.5 x FE(v4) AND R >= R(v4) - 2.0 pts AND L p50 <= L(v4) + 0.20 s AND W <= W(v4) + 0.5;
  - PARTIAL: 0.5 x FE(v4) < FE <= 0.75 x FE(v4) with the safety conditions met;
  - NULL: FE > 0.75 x FE(v4) with safety met;
  - FAIL (safety): R < R(v4) - 5.0 pts or R < 90 %, or L p50 > L(v4) + 0.40 s, or W > W(v4) + 1.0.
  A PASS makes v5-B the EOU recipe candidate (then plain->EOU delta, Mynah m1, FINAL-IT once); otherwise no recipe
  change is promoted on this run.
- Diagnostic added before launch (NOT a gate): FE_any = % clips with an <EOU> followed by >= 0.16 s (2 x 80 ms
  frames) of speech in the original clip, so errors cannot hide by moving under the 0.5 s FE definition. Gates and
  b202980 unchanged. After the run: table v4 vs v5-B (FE | FE_any | R | L p50/p90 | reset-WER) + verdict; only a
  PASS continues to the follow-up probes; PARTIAL / NULL / FAIL stop (no automatic v5-C / A or E2).

### Result: stage-2 v5-B vs v4 (OBSERVED, same box / parent / batches; EOU protocol of b202980) -- pre-registered FAIL (latency)

Selected step (fixed rule) 1000: in-run val 22.07, EOU 93.2 %, 3 empty (v4: step 250, 23.89, 99.5 %, 0).

| harness, reset protocol, +2 s quiet | FE (gate) | FE_any (diag) | true-EOU recall | latency p50 / p90 s | reset greedy WER | CER | empty |
|---|---|---|---|---|---|---|---|
| v4 | 8.36 % | 8.90 % | 94.38 % | 0.72 / 1.92 | 22.54 | 8.95 | 0 |
| v5-B | **0.82 %** | **1.64 %** | 92.47 % | **1.84 / 3.04** | 22.08 | 9.26 | 3 (VP) |

Per set FE v4 -> v5-B: MLS 6.0 -> 0.5, FLEURS 17.0 -> 2.0, CV 3.5 -> 0.5, VP 6.2 -> 0.0; recall 98.0 -> 96.5, 86.0 -> 92.5,
96.0 -> 90.0, 99.2 -> 90.0; latency p50 0.56 -> 1.04, 0.80 -> 2.32, 0.72 -> 2.32, 0.72 -> 2.00 s.
Gates: FE 0.82 <= 4.18 PASS; recall 92.47 >= 92.38 pass (by 0.09); WER 22.08 <= 23.04 pass; **latency p50 1.84 s >
L(v4) + 0.40 = 1.12 s -> FAIL (safety)**. Per the pre-registration: no recipe change is promoted; stop.
Informational (standard probes, offline no reset): v5-B harness raw 22.09 / 20.33 / 16.28 vs parent C250-36k 22.27 /
20.53 / 16.25 (v4: 24.14 / 22.64 / 18.01) -- the plain -> EOU WER regression is GONE; clean flushed 22.25 / 20.13 / 13.68
vs parent 21.74 / 20.23 / 13.71.
Reading (scoped): the positive-label policy causally controls premature endpoints (-90 % FE, FLEURS 17 -> 2 %), and
removing the ambiguous positives also removes the offline ASR damage; the cost is a later decision boundary (EOU
~1.1 s later at p50: the remaining positives are mostly the 1-3 s padded tails). Best-supported cause of the v4
false endpoints: ambiguous positive EOU labels (now supported by an intervention, one seed). Open, NOT run: the
precision / latency trade-off (e.g. padding range / credible-tail threshold) -- any next arm needs its own
pre-registration; FINAL-IT still sealed.

**Frozen conclusion (2026-10-18):** v5-B = pre-registered FAIL (latency) but a strong mechanistic positive result:
positive EOU supervision at acoustically ambiguous clip boundaries is a MAJOR causal driver of premature EOU (this
seed / protocol; not shown to be the only cause). Removing it largely solves false endpoints (8.36 -> 0.82 %) and
removes the stage-2 ASR regression, but over-corrects the EOU decision boundary toward waiting (p50 0.72 -> 1.84 s).
v5-B is NOT promoted to production.

### PROPOSED (not pre-registered, not run): Stage2-v5-CAL

Same frozen definition credible = trail_s >= 0.5 s; <EOU> always on credible and on padded samples; on AMBIGUOUS
unpadded samples <EOU> reintroduced stochastically with a fraction q fixed a priori (one midpoint arm, e.g. q 0.5,
chosen in the pre-registration, not on these numbers), to test the precision / latency dose-response between v4
(q 1: FE 8.36 %, p50 0.72 s) and v5-B (q 0: FE 0.82 %, p50 1.84 s). The decision threshold stays in the checkpoint;
a runtime debounce / commit policy is a separate, later layer. No q chosen today.

### State map (end of 2026-10-18)

| role | checkpoint | key numbers |
|---|---|---|
| best plain IT | `plain-it-c250-36k` (Hub runs/plain-it-c250-36k) | harness 22.27 / 20.53 / 16.25; clean flushed 21.74 / 20.23 / 13.71 |
| production EOU | EOU-v3 `plain-it-s2-e3-djlr` (unchanged) | Mynah FLEURS-it 30.91, EOU 200 / 200, p50 810 ms |
| EOU candidates tested | stage 2 v4 (FAIL: +1.87 offline, false endpoints 8.4 %), v5-B (FAIL: latency 1.84 s; FE 0.82 %, no ASR regression) | see above |
| closed today | E1-8k encoder KD S7 -> T17 (NULL, oracle +0.01); static selection (D, 10-17) | |
| FINAL-IT | 5 / 6 banks built (MLS 400, CV 400, VP 110 / 3 speakers, FLEURS 116, TEDx 400 / 30 talks); ITALIC pending (Zenodo throttled); NOT sealed, never decoded | |

Why v4 fails: premature <EOU> at internal pauses (ambiguous positive labels) + an absorbing blank state after
<EOU>; with the Mynah reset ~74 % of its offline WER loss disappears, the false-endpoint rate (8.4 %) remains.
Why v5-B fails: the label fix over-shoots the endpoint timing (latency gate). Next decision: pre-register v5-CAL (q),
or keep v3 in production; plain 1-best < 20 remains a separate axis (oracle 16.25 vs 1-best 22.27: ranking /
candidate conversion, not encoder KD). E2 (S4 -> T5) parked.

## 2026-10-19 -- Japan L40S (new host 61.206.39.5:26451)

Qualification, restore / parity (identical to Singapore), FINAL-IT completed + sealed (1,826 clips, 6 banks, SEAL
8d023737; never decoded), the snapshot-hang fix, the release checklist / licensing audit and the FR gap analysis:
`.work/release-it-fr-2026-10-19.md`.

### Stage2-v5-CAL -- PRE-REGISTERED (2026-10-19 ~08:45 UTC, before launch)
Full text (definition, q = 0.5, batch audit, anchors v4 / v5-B / v3 re-measured on this box, gates, diagnostic):
`.work/release-it-fr-2026-10-19.md` section E. Summary: ambiguous unpadded ends keep <EOU> with probability 0.5
(dedicated RNG); positive share 0.8213 measured; audio batches hash-identical to v4 / v5-B; PASS = FE <= 4.18 % AND
FE <= 5.89 % AND L p50 <= 1.04 s AND R >= 92.66 % AND reset-WER <= 22.58 AND empty <= 5. After this arm the stage-2 IT
axis closes unless PASS.

### Result: Stage2-v5-CAL (q = 0.5) -- pre-registered class NULL (OBSERVED, Japan L40S, same box / parent / batches)
Selected step (fixed rule, EOU >= 80 %, <= 10 % empty): 500 (in-run val 22.40, EOU 92.8 %; step 250 had EOU 78.6 %);
v5-B selected 1000, v4 250. final.nemo sha256 7c8236c6. Not uploaded (NULL arm; results/jp-2026-10-19 has metrics).

| harness, reset, +2 s quiet | FE | FE_any | R | L p50 / p90 s | reset-WER | CER | empty |
|---|---|---|---|---|---|---|---|
| v4 (q = 1, 100 % positives) | 8.36 % | 8.90 % | 94.38 % | 0.72 / 1.92 | 22.54 | 8.95 | 0 |
| **v5-CAL (q = 0.5, 82.1 %)** | **1.51 %** | **2.19 %** | **91.92 %** | **1.92 / 3.04** | **22.14** | 9.02 | **0** |
| v5-B (q = 0, 65.0 %) | 0.82 % | 1.64 % | 92.47 % | 1.84 / 3.04 | 22.08 | 9.26 | 3 |
| production v3 | 5.89 % | 6.58 % | 94.66 % | 0.64 / 1.92 | 36.07 | 12.82 | 4 |

Per set (FE / R / L p50): MLS 1.5 % / 95.0 / 1.44 s; FLEURS 3.5 % / 91.0 / 2.48 s; CV 0.5 % / 90.0 / 2.20 s;
VP 0.0 % / 91.5 / 1.44 s (v4 FLEURS 17 %, v5-B 2 %, v3 14.5 %).
Gates: PASS no (L 1.92 > 1.04; R 91.92 < 92.66); PARTIAL no (L > 1.28); FAIL no (R >= 90, W 22.14 <= 23.08, empty 0,
FE <= 5.89); NULL yes -- within 1.0 pt FE (0.69) and 0.20 s L p50 (0.08) of the v5-B anchor. **Class: NULL.**
Diagnostic (not a gate): L_chord(1.51 %) = 1.74 s; CAL 1.92 s = 0.18 s ABOVE the v4 -> v5-B chord (worse than
interpolation): reintroducing half of the ambiguous positives buys back almost none of v4's latency (and some FE).
Informational offline probes (no reset): s2 harness 22.67 / 20.98 / 16.71 vs parent 22.27 / 20.54 / 16.26 (+0.40
greedy; v5-B -0.18); clean flushed 22.64 / 20.81 / 14.01 vs parent 21.73 / 20.23 / 13.71.
Reading (HYPOTHESIS, scoped, one seed): the latency response to the positive dose is strongly non-linear -- with
greedy argmax, an <EOU> trained as a coin flip on short tails stays below blank there, so the model still fires only
on the unambiguous (padded, long-tail) evidence; the fast v4 boundary needs (nearly) all short-tail positives,
which also carry its false endpoints. The selected step differs (500 vs 1000; the fixed rule), a known confound of
the latency comparison, not acted on. Per the pre-registration: **the stage-2 label-dose axis is CLOSED** (no
v5-C, no q grid, no threshold change on this DEV). A decision-layer knob (EOU posterior bias / threshold or a
runtime commit policy on v5-B) would be a NEW, separately pre-registered axis -- not opened today.

### Result: EOU logit bias b = 110.3 on v5-B -- pre-registered class NOT-SEPARABLE (OBSERVED, one decode)

| DEV730, reset, +2 s | FE | FE_any | R | L p50 / p90 s | reset-WER | empty | <EOU> / clip |
|---|---|---|---|---|---|---|---|
| v5-B (b = 0) | 0.82 % | 1.64 % | 92.47 % | 1.84 / 3.04 | 22.08 | 3 | ~1.0 |
| **v5-B, b = 110.3** | **14.79 %** | **15.34 %** | **94.79 %** | **0.64 / 1.20** | **22.02** | **3** | MLS 2.40, FLEURS 1.65, CV 1.31, VP 1.75 |
| v4 | 8.36 % | 8.90 % | 94.38 % | 0.72 / 1.92 | 22.54 | 0 | |
| v3 | 5.89 % | 6.58 % | 94.66 % | 0.64 / 1.92 | 36.07 | 4 | |

Per set (FE / R / L p50; clips split into >= 2 segments): MLS 13.0 % / 100 / 0.56 s (164 / 200 split); FLEURS 18.5 % /
85.5 / 0.64 s (64); CV 6.0 % / 97.0 / 0.72 s (60); VP 25.4 % / 97.7 / 0.72 s (70 / 130). No-reset WER 29.67 (50 empty).
Gates: FAIL no (R 94.79, W 22.02, empty 3); PASS no (FE 14.79 > 2.95 though L 0.64 <= 1.04); PARTIAL no;
**NOT-SEPARABLE yes**. Reading: the rule hit its latency target exactly (p50 0.64 s = v3), but a global <EOU> offset
large enough for that fires at internal pauses in 15 % of clips (2.5 x v3, 1.8 x v4) and splits most MLS / VP clips:
on v5-B the <EOU>-vs-blank score does not separate true ends from pauses at fast operating points; it is dominated
by v4 and v3 there. Reset-WER is preserved only because the reset makes split segments transcribe correctly; for
turn-taking the splits are the failure. Per the pre-registration: **IT EOU research STOPS** (no sweep, no second b).

### RC-IT v4 -- FROZEN (2026-10-19 ~11:45 UTC). IT model development CLOSED.
- **Artifact reproduction** (NOT a second FINAL evaluation):
  - the frozen decode re-run only to save per-clip outputs (`--dump`);
  - aggregates byte-identical to the authoritative run: natural sha256 91bec6df..., trim 11b31d34...;
  - per-clip files `final-it-v4/repro/v4_{final,final_trim}_clips.jsonl`;
  - the first run stays authoritative.
- **Mynah parity** (m1.sh, DEV FLEURS-it test 200, never FINAL-IT):
  - CLI stream f32: WER 17.17, CER 7.19, 0 empty, model <EOU> on 196 / 200 (EOU-v3: 30.91, 200 / 200);
  - eou_metrics (60 clips): speech end -> EOU p50 778 / p90 2794 ms (v3 p50 810), premature 3.3 %;
  - A + 1 s + B: EOU in gap 83.3 %, p50 426 ms;
  - CUDA test_cuda_stream gates A / B / C PASS (cpu f32 == gpu);
  - CUDA server v0.9.1-607-g52e5271 bf16, cap 128, 1.93 GB VRAM used, start-up 520 ms.
- **Frozen RC:**
  - checkpoint sha256 0a4d1b10...3ff;
  - pack (Hub `rc/it-v4/mynah-pack/`): model.safetensors 2aa4783d, mynah.json ff6d7c5a, tokens.json 70a5a46e,
    mel_filters 2dde1d9b;
  - runtime mynah-asr 52e5271 (v0.9.1-607): att_context [70, 1], lookahead 1, 80 ms frames, max 10 symbols / step;
  - decoder-only reset after <EOU>;
  - finalisation as the FINAL protocol (2 s quiet offline; Mynah stream flushes at EOF);
  - normalisation NFKD [a-z' ]; CPU f32 / CUDA bf16;
  - commands: finalv4.sh / rcv4.sh in results/jp-2026-10-19.
- Model-card draft: `finetune/release/it/README.md` (not published; LICENSE / NOTICE files and usage examples still
  to write).
- FR-0 coverage complete:
  - every FR source has 0 % <unk> and 100 % round trip with the stock SPE;
  - curly apostrophes occur in 8.9 % of CV-fr and 20 % of FLEURS-fr sentences, so the fr pre-map is required.

### Result: FR-P1 (W4 8k on A2-fr) -- pre-registered class LEVER (OBSERVED, 2026-10-19, Japan L40S, one seed)

- Run: 8000 steps in 2,367 s, final.nemo = step 8000 (sha256 1ec3fec4); Hub `runs/jp-fr-p1-w4-8k`.
- VP-fr teacher filter (<= 50, frozen): kept 15,997 / 16,191 utts, 39.66 / 40.02 h, 236 / 236 speakers.
  VP-it kept 80.7 %: VP-fr labels are far cleaner.
- Harness-fr frozen (sha256 recorded) before the baselines.

| harness-fr macro (MLS / FLEURS / CV / VP test, 200 each), 0.5 s flush | greedy | beam-8 | oracle-8 | CER (greedy) |
|---|---|---|---|---|
| stock parakeet_realtime_eou_120m-v1 (English) | 94.90 | 95.61 | 94.45 | 73.81 |
| old plain-fr-p40 (2026-10-09 recipe, MLS-only) | 65.85 | 64.88 | 61.65 | 40.78 |
| old plain-fr-s2-eou | 65.02 | 63.68 | 60.29 | 40.52 |
| **FR-P1** | **27.66** | **25.91** | **21.76** | **11.34** |

Per source, FR-P1 vs old p40 (greedy, flushed):
- MLS 25.13 vs 50.93;
- FLEURS 33.51 vs 76.30;
- CV 29.53 vs 67.92;
- VP 22.46 vs 68.24.

FR-P1 per source, beam-8 / oracle-8:
- MLS 23.49 / 21.20;
- FLEURS 31.08 / 27.84;
- CV 27.65 / 20.50;
- VP 21.41 / 17.50.

Raw EOF: 27.84 / 26.06 / 21.83 (flush ~ -0.2).
Empties: in-run final eval 3 / 730 (MLS 1, FLEURS 1, VP 1); lm_fusion_probe does not report empties.
Sanity: old p40 MLS 50.97 raw = its 2026-10-09 number 50.99.

**Class:**
- delta macro greedy flushed = 27.66 - 65.85 = **-38.19 -> LEVER** (<= -10).
- Recipe + diversity + encoder adaptation explains most of the old FR gap: every source drops 26-46 points,
  out-of-domain FLEURS from 76.3 to 33.5.

**Diagnostic gaps:**
- greedy -> beam 1.75, beam -> oracle 4.15.
- IT W4-8k on C250 was 26.12 / 23.67 / 19.43, so this FR result without Granary is close to IT with Granary at
  the same horizon.
- The remaining gap is mostly not ranking (beam gains little): exposure / data scale is the next lever (HYPOTHESIS).

Per the decision rule: LEVER does NOT authorise stage 2. Next = ONE pre-registered decision on how to scale plain FR
(Granary-fr pools, MLS-fr hours, schedule). Stopped here.

### Result: FR-G1 (stock -> A2-fr + Granary-fr agree <= 20, W4 8k) -- pre-registered class HARM (OBSERVED, one seed)

- final.nemo sha256 de457d0b; Hub `runs/jp-fr-g1-w4-8k` + `results/jp-2026-10-19/fr-g1` (7 files OK).
- Peak GPU memory 39.5 GB (30 s Granary clips).

| harness-fr macro, 0.5 s flush | greedy | beam-8 | oracle-8 | CER |
|---|---|---|---|---|
| FR-P1 (A2-fr) | 27.66 | 25.91 | 21.76 | 11.34 |
| **FR-G1 (A2-fr + Granary 89 h)** | **29.41** | **26.54** | **22.74** | **12.22** |
| delta | **+1.75** | +0.63 | +0.98 | +0.88 |

Per source, greedy / beam-8 / oracle-8, FR-G1 vs FR-P1:

| source | FR-G1 | FR-P1 |
|---|---|---|
| MLS | 27.41 / 24.82 / 22.38 | 25.13 / 23.49 / 21.20 |
| FLEURS | 33.73 / 31.16 / 28.14 | 33.51 / 31.08 / 27.84 |
| CV | 32.97 / 28.64 / 22.22 | 29.53 / 27.65 / 20.50 |
| VP | 23.54 / 21.55 / 18.23 | 22.46 / 21.41 / 17.50 |

- Raw EOF: 29.51 / 26.67 / 22.86.
- Empties: in-run final eval 4 / 730 (FR-P1 3).

**Class:** +1.75 > +0.5, so **HARM**. Per the pre-registration: stop; no G2 (16k), no larger pool, no compensation.

**Diagnosis (HYPOTHESES, from the observed numbers only):**
1. Exposure was reallocated, not added. At fixed 8k steps the A2 draws fall 33 % (MLS 4.7 -> 3.1 passes, CV
   1.4 -> 0.9) and the total processed audio falls 8 % (335 vs 364 h). The harness is the four A2 domains; the
   largest losses are CV (+3.4) and MLS (+2.3), the domains whose exposure fell most. FLEURS (+0.2) and VP (+1.1)
   move least.
2. Label quality: agree <= 20 admits clips whose teacher text disagrees with the Granary reference on up to 1 in
   5 words (pool median 18.2). Greedy -> beam widens from 1.75 to 2.87, which reads as less confident or
   less calibrated 1-best decisions, consistent with noisier targets.
3. Domain: Granary is YouTube speech; no harness source is.
   (1) and (2) are not separable with this run. The IT analogue (W4-C250-8k) also traded A2 draws for external
   audio, but its external pool was cleaner (median agree 8.7) and twice as large.

Not run (would need their own pre-registration): an exposure-preserving additive design (A2 held at FR-P1 draws,
Granary on top, i.e. longer steps), a stricter label threshold with a larger pool, or a clean-fr / out-of-domain
bank where Granary diversity could show (harness-fr is in-domain for A2).

## State map (end of 2026-10-19)

| item | state | key numbers |
|---|---|---|
| RC-IT | **Stage2-v4 on C250-36k, FROZEN** (ckpt 0a4d1b10, Mynah pack Hub rc/it-v4) | FINAL-IT one-shot macro greedy reset-WER 20.49 (band < 20 missed), FE 3.50 %, recall 96.2 %, p50 0.88 s; FLEURS FE 16.4 %; Mynah FLEURS-it 17.17, EOU p50 778 ms |
| IT EOU research | CLOSED | v5-B FE 0.82 % / p50 1.84 s (precision-first variant); v5-CAL NULL; logit bias NOT-SEPARABLE |
| FINAL-IT | spent (one decode + artifact reproduction) | |
| FINAL-FR | SEALED 8b0ebba0, never decoded | 6 banks, 2,113 clips |
| best plain FR | **FR-P1** (stock -> A2-fr, W4 8k; Hub runs/jp-fr-p1-w4-8k, 1ec3fec4) | harness-fr 27.66 / 25.91 / 21.76 vs old p40 65.85 (LEVER) |
| FR-G1 | HARM | +89 h Granary-fr agree <= 20 at fixed 8k: 29.41 / 26.54 / 22.74 |
| release | model-card draft finetune/release/it/README.md; licence NVOML; token to rotate | |

Next (each its own pre-registration):
1. FR: an exposure-preserving additive Granary design, or a stricter label threshold on a larger pool, or a clean-fr
   bank first.
2. Then FR stage 2 with the frozen IT recipe.
3. Then RC-FR and FINAL-FR once.
4. IT release: LICENSE / NOTICE files and usage examples.
