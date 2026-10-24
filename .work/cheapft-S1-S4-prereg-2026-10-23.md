# S1-S4: sample-efficiency / GPU-time pre-registrations (2026-10-23, before any of these runs)

Status: **GPU-READY** (`finetune/jobs/cf_s_full.sh ARM=S1|S2|S3|S4`). Each is one run, one seed, the ES-1 recipe
plus ONLY the arm's change, evaluated by the ES-1 protocol and gated by `cf_compare.py s`.

## Common to all four

**Reference: the ES-1 run itself**, the frozen pre-registration of section 9 / 11 of the release notes:
- MLS-es 40 h 0.35 / CV17-es 40 h 0.30 / VP-es t50 0.20 / FLEURS es_419 0.15, **126.5 physical unique hours**;
- stock EOU-120M init and tokenizer; whole encoder 4e-4, decoder / joint 3e-4;
- WSD with 5 % warm-up and the last 20 % decay; bs 16; max-dur 30; seed 0;
- `--final-eof 1`, `--abort-wer 1000:85`;
- 12,670 steps = **584.5 effective audio-hours = 4.62 passes**;
- about 3.75 h on GB10: 215 min of steps at 1,019 ms + ~10 min of in-run evals.

ES-1 is read, never re-run or modified. If its final probe is not available, the S runs wait.

**Exposure accounting.** 0.04613 audio-hours per step: CF-E1 saw 69.2 h in 1,500 steps, and bucketing does not
change it (same draws).

**Primary metric:**
- **macro greedy WER, 0.5 s flush**, on the harness-es (MLS-es test, FLEURS es_419, CV17-es 200, VP-es 200) with
  the `fr_p1.sh` probe, exactly as ES-1 is scored (`probe_harness_post0.5.json`, record `greedy a=0.0`);
- secondary: macro CER; per-set WER reported.

**WER / CER gate:**
- delta macro WER <= +1.0 vs ES-1
- **and** delta macro CER <= +0.5
- A CER breach marks the arm "not promotable", whatever its class.

**Economic success criterion.** GPU time = the run's `metrics.json` `wall_s`: the training steps plus that run's own
in-run eval schedule, with probes excluded, against ES-1's `wall_s`.

| class | condition |
|---|---|
| PASS | >= 25 % GPU-time reduction at <= +1.0 WER |
| STRONG | >= 40 % at <= +1.0 |
| EXCEPTIONAL | >= 50 % at <= +0.5 |
| FAIL | > +1.0 WER, whatever the speed |

**EOU gate (stage 2).** It is not applicable to these stage-1 screens. It applies when a passing arm is promoted into
the factory recipe:
- the frozen v4 stage-2 recipe, run on the candidate parent and on the ES-1 parent, same seed;
- the trainer's own rule: `val_eou_rate` >= 0.8 and `val_empty` <= 10 % of the val clips;
- where a DEV-EOU bank exists for the language: false-endpoint rate within +2.0 pp and endpoint p50 within
  +80 ms (`eou_v2_score.py`);
- for ES today there is no DEV-EOU bank, so the gate is declared **NOT-EVALUATED** beyond the in-run rule.

**Abort rules:**
- collapse: `--abort-wer 1000:85` (rc 5);
- **speed**: ms / step over steps 1025-1975 (a window with no in-run eval) > 1.15 x the predicted ms / step. The
  run is killed (rc 6) and no quality reading is taken, because the economic premise failed;
- a mid-run WER is **never** an early stop: mid-run WSD evals are not what a shorter run reaches, the decay bonus
  being 9-14 % relative;
- a Traceback or NaN loss means rerun only after a fix, with the same pre-registration.

**Declared limits.** One seed. A STRONG or EXCEPTIONAL class licenses one replicate seed before adoption, not
adoption. GB10 only. The Ada (L40S) minutes below are GB10 / 3 (OBSERVED 2026-10-15 for this trainer). The relative
savings are assumed to transfer; on a faster GPU the fixed host / launch milliseconds weigh more, so the savings may
be smaller [H].

## S1: shorter full WSD horizon, decay rescaled

**One variable:** `--steps 7602` (0.6 x 12,670). two_stage.py derives the warm-up (steps / 20 = 380) and the decay
(the last 20 %, from step 6,082) from `--steps`, so the whole schedule is rescaled, not truncated.

| physical h | effective audio-h | passes | optimizer steps | schedule | GPU-min GB10 | GPU-min Ada (L40S) |
|---|---|---|---|---|---|---|
| 126.5 | 350.7 | 2.77 | 7,602 | WSD 380 warm-up, flat to 6,082, decay to 7,602 | 129 steps + 7 evals = **136 (-39.5 %)** | ~45 |

- **Prediction [H]:** delta +0.3 to +1.5 WER.
  - CF-C1: ~95 % of the gain arrives at half the horizon, but its mid-run evals sat +3.4-3.5 WER above the final.
  - A completed decay recovers 9-14 % relative, about 3 WER at this level.
- **Reading:**
  - PASS (-39.5 %, just under STRONG) if delta <= +1.0.
  - If delta <= +0.5, a 0.5x arm (6,335 steps, ~-49 %) is licensed as S1b. It is the only route to EXCEPTIONAL here.
  - If delta > +1.0, the horizon stays at 4.6 passes.
- Speed abort: PRED_MS 1,019.

## S2: full-horizon bucket replicate with logging

**One variable:** `--bucket-pool 32` (sorted). `--log-batch-stats 1` is on, which is training-transparent (tested).
This replicates CF-E1's B (-42.7 %, +0.99 at 1,500 steps) at the production horizon.

Deviation from CF-E1's licensed next step (pool 8), declared:
- T3's simulation shows pool 8 does **not** restore domain mixing (TV 0.433 vs pool 32's 0.439, random 0.172);
- it only gives up speed (-38 % vs -42 %);
- the replicate therefore tests the measured arm.

| physical h | effective audio-h | passes | optimizer steps | schedule | GPU-min GB10 | GPU-min Ada |
|---|---|---|---|---|---|---|
| 126.5 | 584.5 (same draws) | 4.62 | 12,670 | ES-1's | 123 steps + 10 evals = **134 (-40.8 %)** | ~45 |

- **Prediction [H]:** delta +0.3 to +1.2. A longer horizon averages the batch-composition noise, but a systematic
  re-weighting would persist.
- **Logged diagnostics** (from batch_stats.jsonl), to test the clip-reweighting hypothesis:
  - corr(pre-clip grad norm, batch mean duration);
  - the gn p10-p90 spread vs ES-1;
  - the batch-audio CV;
  - per-domain WER deltas.
- **Reading:**
  - delta <= +1.0: STRONG (-40.8 %).
  - delta <= +0.5 would make it the factory default, pending one replicate.
  - delta > +1.0: FAIL, and bucketing is limited to the grouping-free route (`--enc-subbatch`, T2).
- Speed abort: PRED_MS 584.

## S3: the best constrained-bucketing variant (T3)

**One variable:** the grouping, `--bucket-pool 32 --bucket-mode blend --bucket-alpha 0.65`. It is predicted to keep
the random sampler's per-batch domain mixing: TV 0.139 vs 0.172, entropy 1.768 vs 1.777 bits, 3.80 vs 3.89 sources
per batch. The batch-audio CV is 0.378 vs 0.115, so the steps remain length-graded.

**With S4** (`JS4=1`) only if T1 passed parity and compute. S4 is a numerics-only change (same batches), so the
quality variable stays the grouping alone.

| physical h | effective audio-h | passes | optimizer steps | schedule | GPU-min GB10 | GPU-min Ada |
|---|---|---|---|---|---|---|
| 126.5 | 584.5 | 4.62 | 12,670 | ES-1's | 153 + 10 = **163 (-27.5 %)**; with S4 142 + 10 = **152 (-32.5 %)** | ~54 / ~51 |

**Licensed only if** T2's 1,500-step BLS4 quality is not FAIL (delta <= +1.0 vs R).

**Reading (pre-registered, the mechanism split with S2):**

| S3 | S2 | conclusion |
|---|---|---|
| ≈ ES-1 (delta <= +0.5) | worse | per-batch domain mixing was the harm: blend is the bucketing to use (PASS at -27.5 / -32.5 %) |
| ≈ S2, both worse | | length-graded clipped steps are the harm: no bucketing at bs 16; `--enc-subbatch` is the speed route |

Speed abort: PRED_MS 725 (672 with S4).

## S4: `--joint-subbatch 4` full-recipe equivalence

**One variable:** `--joint-subbatch 4`. Same draws, same batches, same per-utterance weights. It is exact in maths,
up to the summation order and bf16 GEMM tiling.

| physical h | effective audio-h | passes | optimizer steps | schedule | GPU-min GB10 | GPU-min Ada |
|---|---|---|---|---|---|---|
| 126.5 | 584.5 | 4.62 | 12,670 | ES-1's | 178 + 10 = **188 (-16.6 %)**, or T1's measured ms | ~63 |

- **Gate: equivalence.** |delta macro| <= 0.5 and |delta CER| <= 0.3.
  - Above +0.5 with T1's parity passed would point to a long-horizon drift of the bf16 path: report it, do not adopt.
- **Economic criterion:** -16.6 % is **below** the 25 % PASS line. Alone, S4 is at best "NULL (speed < 25 %)" by
  construction. It is pre-registered as a **component certification**: it lets S4 ride along in S2 / S3 / MB recipes
  without its own quality bet.
- **Licensed only if** T1 passed P, C and Q **and** a combination recipe that adopts S4 is on the table. Otherwise
  the 3.3 h are better spent on S1 / S2.
- Speed abort: PRED_MS = T1's measured S4 ms / step (`PRED_MS_S4=`), default 842.

## Order and expected value

S1 and S2 come first: each is about 2.3 h of GB10 for a possible STRONG class. S3 runs only after T2-BLS4 is
not FAIL. S4 runs only when a combination needs certifying. A grouping-free MB4+S4 full-horizon arm (T2's predicted
-37 %) should get its own pre-registration once T2 measures its compute and parity: **NEXT-AGENT**.

## Commands

```bash
# dgx1; ES1_RUN = the ES-1 run dir holding probe_harness_post0.5.json + metrics.json
FT_ROOT=$HOME/ft ARM=S1 ES1_RUN=$HOME/ft/runs/plain-<es-1 tag> \
  tmux new -d -s cfs1 'FT_ROOT=$HOME/ft ARM=S1 ES1_RUN=... bash finetune/jobs/cf_s_full.sh 2>&1 | tee -a $HOME/ft/logs/cfs1.log'
# ARM=S2 | ARM=S3 [JS4=1] | ARM=S4 [PRED_MS_S4=<T1 ms>]; result: $FT_ROOT/runs/plain-cfs-<arm>-es/s_gate.{txt,json}
```

---
**Launch record (2026-10-23, before either run):** S1 on dgx1 and S2 on dgx2, code = branch feat/trainer-throughput
e613e87 (flags off = bit-identical to the ES-1 trainer, Phase 0), `finetune/jobs/cf_s_full.sh ARM=S1|S2`, reference
= ES-1 (runs/dgx-es1-w4-12670: macro greedy 0.5 s 18.09, CER 7.43, wall_s 13,325.6). Context known before launch:
CF-E3 measured a 1,500-step seed spread of 0.46 WER; S1 / S2 are single-seed, so a delta within ~0.5 of a gate edge
is reported as such.

## S2 result (dgx2, bucket-pool 32, full horizon 12,670 steps, seed 0; ended 17:09 CEST)
- Pre-registered S gate (`cf_compare.py s`, post 0.5 probe): macro 18.68 vs ES-1 18.09 (+0.59), CER +0.14,
  GPU time -41.0 % -> **STRONG** (within +1 WER, saving >= 30 %). Speed guard: 569 ms / step (limit 671).
- Greedy probe (post 0), per domain, ES-1 -> S2 (WER / CER / empty):
  MLS 15.12 / 5.22 / 0 -> 15.16 / 5.12 / 0; FLEURS 17.10 / 6.90 / 0 -> 18.02 / 7.20 / 0;
  CV 21.13 / 7.70 / 1 -> 22.98 / 8.05 / 1; VP 19.48 / 10.30 / 1 -> 19.32 / 10.54 / 0; macro 18.21 -> 18.87 (+0.66).
  Beam-8: MLS 13.31, FLEURS 16.82 (S2).
- OBSERVED: the loss is concentrated on CV (+1.85) and FLEURS (+0.92); MLS and VP are flat. Within ES-1's
  seed spread? Unknown for ES (the 0.46 seed spread was measured on CF-E3, a different setting).
- Batch stats (bstats_cmp.py, whole run): domain shares 35.0 / 29.9 / 20.0 / 15.1 (mix 35 / 30 / 20 / 15), audio
  165.4 s / step, encoder efficiency 0.968, joint 0.695, in-batch duration spread 0.55 s; pre-clip grad norm
  p10 / 50 / 90 = 72 / 101 / 150, clip factor median 0.0099 (every step clipped); corr(log gn, duration) 0.68
  (stable) / 0.70 (decay). S1 (unbucketed, partial): efficiency 0.56 / 0.35, corr 0.24.
- HYPOTHESIS (not tested): with every step clipped, bucketing changes the per-sample weighting by duration; it
  could explain a domain-specific loss (CV clips are short). The blend mode and a clip reweighting are candidate
  arms to pre-register, not established remedies.

## S1 result (dgx1, 0.6x horizon 7,602 steps, default sampler, seed 0; ended 17:22 CEST)
- Pre-registered S gate (post 0.5 probe): macro 20.77 vs 18.09 (+2.68), CER +0.41, GPU time -39.3 % -> **FAIL**.
- Per domain (post 0.5), ES-1 -> S1: MLS 14.86 -> 16.58, FLEURS 17.10 -> 20.06, CV 21.18 -> 25.32, VP 19.20 -> 21.14.
  Every domain loses 1.7-4.1 WER: a shorter horizon is not a free saving on this recipe.
- Joint reading with S2: at the same ~40 % GPU saving, bucketing (S2, +0.59) costs about a fifth of what cutting
  the horizon costs (S1, +2.68). The horizon stays at ES-1's 12,670 steps; bucketing is the lever.

## S2 replicate (seed 1), launched 17:26 CEST on dgx1
Licensed by the S2 STRONG class (this file: "one replicate seed before adoption"). Code f3161d7 (`SEED=1`, tag
cfs-s2-s1-es; only --seed changes, which also reseeds the bucket RNG). Same S gate. Reading: if seed 1 is also
within +1.0, S2 is adopted for the ES recipe; the S2 - ES-1 gap (two S2 seeds vs one ES-1 seed) is reported with
the S2 seed spread, not as a significance claim.

## S2 replicate (seed 1) result (dgx1, ended 19:38 CEST)
- S gate (post 0.5): macro 18.13 vs ES-1 18.09 (+0.04), CER -0.28, GPU time -41.1 % -> **STRONG** (speed 568 ms/step).
- Greedy post 0, ES-1 -> S2 s0 -> S2 s1 (WER): MLS 15.12 -> 15.16 -> 14.20; FLEURS 17.10 -> 18.02 -> 17.68;
  CV 21.13 -> 22.98 -> 22.18; VP 19.48 -> 19.32 -> 19.16; macro 18.21 -> 18.87 -> 18.30; empty 2 -> 1 -> 0.
- Two S2 seeds: post 0.5 macro 18.68 / 18.13, mean 18.41 (+0.32 vs one ES-1 seed), seed spread 0.55 (CF-E3: 0.46).
  The S2 - ES-1 gap is inside the seed spread: no significance claim, in either direction.
- OBSERVED in both seeds: CV (+1.85 / +1.05) and FLEURS (+0.92 / +0.58) worse than ES-1, MLS / VP flat or better.
  The clip-reweighting reading (short CV clips) stays a HYPOTHESIS; a test would be one pre-registered arm (blend or
  a clip reweighting) against two S2 seeds, CV as the primary domain.
- Batch stats seed 1: same as seed 0 (domains 34.9 / 30.0 / 20.0 / 15.0, encoder 0.968, joint 0.696, corr 0.49).
- **Decision: bucket-pool 32 is ADOPTED** for the ES plain recipe (the pre-registered replicate passed): ~41 %
  less GPU time at the full 12,670-step exposure. Hub: runs/dgx-cfs-s2-es, runs/dgx-cfs-s2-s1-es (final.nemo +
  sha256 + evidence); S1 evidence only in results/dgx-2026-10-23/cfs-s1.

## S5 result (2026-10-24, two seeds; plans/S5-prereg.md on feat/trainer-throughput b4efe59; Hub runs/dgx-cfs-s5-es, runs/dgx-cfs-s5-s1-es)
S5 = S2 (bucket-pool 32) with an unbucketed decay phase (`--bucket-decay-mode none`, from step 10,144). Probe post-0.5,
greedy:

| | S5 s0 | S5 s1 | S5 mean | ES-1 | S2 mean (spread) |
|---|---|---|---|---|---|
| macro WER / CER | 18.08 / 7.15 | 18.07 / 7.09 | 18.08 / 7.12 | 18.09 / 7.43 | 18.41 (0.55) |
| CV / FLEURS | 21.98 / 17.04 | 23.03 / 17.24 | 22.51 / 17.14 | 21.18 / 17.10 | 22.41 / 17.85 |
| MLS / VP | 14.54 / 18.77 | 13.56 / 18.44 | 14.05 / 18.61 | 14.86 / 19.20 | 14.38 / 19.00 |
| GPU vs ES-1 | -32.4 % | -33.2 % | -32.8 % | 0 | -41 % |
| pre-registered class | RECOVERS | NULL | | | |

**Conclusion (frozen):** S5 is the preferred compute-efficient factory recipe: across two seeds it matches the
full-compute ES-1 macro WER (18.08 mean vs 18.09; S5 seed spread 0.01) while reducing training GPU time by ~33 %.
This is NOT a statistical equivalence claim (ES-1 has one seed). Domain-level recovery is not uniform: FLEURS recovery
replicates and MLS/VP beat ES-1 on both seeds, but the CV decay-recovery hypothesis did NOT replicate (seed 1 CV 23.03).
S2 stays as the aggressive/economy profile: ~41 % GPU saving, mean macro ~18.41, larger seed spread.
