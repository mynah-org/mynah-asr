# Factory v1 CANDIDATE: the EOU-120M language-specialist recipe we would run today (2026-10-26)

Status: **CANDIDATE, not closed.** This is a CPU-only spec written from the archived reports, pre-registrations, job
scripts and run `metrics.json`. No training, no decode, no GPU. Sealed banks (FINAL-IT, FINAL-FR, FINAL-FR-v2,
FINAL-EOU-FR-v2) were not touched. Their numbers are quoted from the one-shot results that were already recorded.

Scope: a new **Latin-script Romance or European language** that the stock SPE can write after de-accenting into
`[a-z' ]`. Non-Latin scripts (AR) are out of scope (section 1, AR row).

Tags: **OBSERVED** (an archived artifact is named), **HYPOTHESIS** (an inference, not tested), **UNKNOWN** (no
artifact answers it), **COMPUTED** (arithmetic on OBSERVED numbers, done here). Data-contract items also carry
**REQUIRED / RECOMMENDED / UNKNOWN**.

Source keys are those of `eou-specialist-evolution.md` §0:
- RIF = release-it-fr-2026-10-19.md
- RF22 = release-final-2026-10-22.md
- S14 = cheapft-S1-S4-prereg-2026-10-23.md
- RM12 / RM15 = the IT roadmaps
- AUT = fr-v2-eou-autopsy-2026-10-21.md
- P:<x> = trainB `finetune/eou/plans/<x>.md`
- J:<x> = trainB `finetune/jobs/<x>.sh`

Also used: EVO = eou-specialist-evolution.md, MAT = factory-validation-matrix.md, SUB15 = sub15-error-budget-plan.md,
and **GATE5** = the S5-FR gate output (dgx1 `~/ft/s5fr/s5fr_gate.txt`, Hub `results/dgx-2026-10-25/s5fr-gate`).

Code reference: `mynah-asr-wt-trainB`, branch feat/trainer-throughput, head ae3bcd8 (local-only branch).

## 1. Historical evolution (summary; the full genealogy is in `eou-specialist-evolution.md` §1-2)

IT, FR, ES and AR are a research **sequence**, not four instances of one recipe (EVO "Read this first"). No language
so far was trained on the exact Factory-v1 spec below:
- IT's release parent is off-recipe (cosine, encoder 2e-4).
- FR and ES used the random sampler for their reference runs.
- ES has no Stage 2.
- AR used S2, max-dur 20 and romanisation.

| | IT (laboratory) | FR (refinement / replication) | ES (frozen-recipe replication) | AR (cross-script stress test) |
|---|---|---|---|---|
| parent | stock EOU-120M | stock | stock | stock, through reversible romanisation |
| tokenizer | stock SPE 1024 + `<EOU>`/`<EOB>`, de-accented | stock + `textnorm fr` pre-map | stock + `textnorm es` | stock, via romanised `[a-z']` |
| data (unique physical h) | C250: A2 (MLS / CV / VP t50 / FLEURS) + two Granary agree<=15 pools, 347.7 h COMPUTED (~357 quoted) | G1E / H2: A2-fr + Granary-fr agree<=20, 216.3 h | A2-type 4 sources, 126.48 h, no web source | CV17-ar + MASC teacher CER<=15, 177.1 h |
| steps | 36,000 (+ Stage 2 1,500) | G1E 11,881; H2 23,762 (+ Stage 2 1,500) | 12,670 (4.60 passes rule) | 12,670 |
| schedule | **cosine, encoder 2e-4**, dec/joint 3e-4 | W4 (WSD 5 % warm-up / 20 % decay, enc 4e-4, dec/joint 3e-4) | W4 | W4 |
| bucketing | none (random) | none in the release runs; S5-FR A/B 2026-10-25 | ES-1 none; S2 and S5 two seeds each | S2 (bucket 32 sorted, no decay switch), max-dur 20 |
| Stage 2 EOU | **v4** (origin) | **v4 unchanged**: v1 on G1E, v1.1 on H2 | **not run** | not run |
| best greedy (own DEV bank, 0.5 s) | DEV730 22.27 plain; v4 reset-WER 22.54 | DEV-fr: H2 22.06, v1.1 22.08; OOD1 24.97 | harness-es 18.09 (ES-1); S5 18.08 / 18.07 | FLEURS ta-fold 33.28 / 34.19 (A2 two seeds) |
| sealed FINAL | **FINAL-IT 20.49** (CER 6.95), FE 3.50 %, R 96.17 % | **FINAL-FR 27.76** (v1, 6 banks), FE 7.57 %. **FINAL-FR-v2 23.00** (v1.1, 4 banks), FE 12.70 %. The two are not comparable | none (harness reused for compute selection) | none |
| evidence | EVO 1.A-1.D | EVO; RIF §I-O; RF22 | RF22 §9-11; S14 | P:AR-SEED1 |

All values OBSERVED as cited in EVO §1. Cross-language WER is never comparable (EVO §4).

## 2. Factory v1 candidate spec

### 2.1 MODEL

| item | value | tag / evidence |
|---|---|---|
| parent | stock `parakeet_realtime_eou_120m-v1.nemo` (two_stage.py `--stock` default), sha256 prefix **6603a22a** (= the Hub LFS sha) | OBSERVED, RF22 §11 l.306. The full hash is recorded on the box (`sha256sum` at prov); not reproduced here |
| init policy | **stock English init** for every new language. No warm start from another specialist | OBSERVED: CF-E1, warm start from FR-H2 for ES was HARM +1.87 at 1,500 steps (RF22 l.466-477) |
| tokenizer | **stock SPE 1024** (tokens.json 70a5a46e) + `<EOU>`/`<EOB>`. Targets via `common/textnorm.py --lang <lg>`: lower case, per-language pre-map (ligatures, typographic apostrophes), NFKD de-accent, `[a-z' ]`; digit clips dropped | OBSERVED: IT/FR/ES 0 % `<unk>`, 100 % round trip (RIF l.527-528); textnorm sha 19ebfd07 at RC-FR (RIF l.1243). The Italian SPE was negative, but under a confounded budget (BN l.58-80). AR: the raw script maps to `<unk>`, and a new text side fails (R24 l.87-95) |
| new language precondition | REQUIRED: a `textnorm` pre-map entry, plus a CPU audit of 0 % `<unk>` and 100 % round trip on the train text before any GPU | OBSERVED practice RIF l.527; AR shows what happens without it |
| trainable, Stage 1 | **everything**: whole encoder from step 0 (`--freeze-enc 0`), decoder and joint | OBSERVED: IT-E3 freeze curriculum was the bottleneck, -12.56 (BN l.161-218) |
| trainable, Stage 2 | top-4 encoder blocks + decoder + joint | OBSERVED: v4 (S2N §1) |

**Stage 1 (plain ASR), exact flags** (J:fr_p1 + J:mix_arm + the S5 EXTRA of J:s5fr, `LANG_=<lg>`):

```
two_stage.py --mix "<frozen mix>" --freeze-enc 0 --enc-lr 4e-4 --lr 3e-4 --bs 16 --steps <N> --eval-every 500 \
  --vals <DEV harness 4 sets> --lang <lg> --max-dur 30 --eval-steps 100,250,1000 --abort-wer 1000:85 \
  --lr-shape wsd --decay-frac 0.2 --eval-sparse 2000,500 --final-eof 1 --seed 0 \
  --bucket-pool 32 --bucket-decay-mode none
```

What the trainer does with these flags (two_stage.py at ae3bcd8, OBSERVED in the code):
- **Optimiser:** AdamW betas (0.9, 0.98), weight decay 1e-3 (l.384); total grad-norm clip 1.0 (l.876).
- **Warm-up:** max(10, steps // 20) steps.
- **WSD:** flat peak LR, then a cosine decay over the last 20 %; floor 1 % of peak (`lr_factor`, l.385-394).
- **Checkpoint:** final.nemo = the LAST step (`--final-eof 1`). The in-run DEV evals are diagnostic and never select.
- **Smoke:** 40-step smoke first (SMOKE=1 in fr_p1.sh). Disk gate: >= 4 GB free.

**Stage 2 (EOU) = v4, unchanged** (RIF §N l.1062-1067; J:s2_mix with `S2EXTRA="--lr 2e-5"`):

```
two_stage.py --init <Stage-1 final.nemo> --mix "<the parent's own mix>" --freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5 \
  --lr 2e-5 --bs 16 --steps 1500 --lr-shape cosine --eou-append 1 --pad-prob 0.5 --pad-min 1 --pad-max 3 \
  --pad-noise-db -90,-60 --fastemit 0 --max-dur 20 --eval-every 250 --val-n 200 --seed 0 --lang <lg>
```

- **Checkpoint rule:** final.nemo = the min in-run DEV WER among evals with EOU >= 80 % and empty <= 10 %. That gave
  step 250 for IT, step 250 for FR-G1E and step 1250 for FR-H2.
- **Sampler:** random. S5 is a Stage-1 flag only.
- **Evaluation protocol:** `s2_reset.py --lang <lg> --post-pad-s 2.0` (frame-synchronous greedy, decoder reset after
  `<EOU>`, 2 s of quiet). The product check is the Mynah CUDA bf16 server.

### 2.2 DATA CONTRACT

Every item carries a level (REQUIRED / RECOMMENDED / UNKNOWN), its basis and its evidence pointer. No threshold below
is stronger than its evidence.

| # | item | level | value | evidence |
|---|---|---|---|---|
| D1 | human-labelled physical hours | REQUIRED | **~120-130 h** from 4 A2-type sources | OBSERVED: IT 119.4, FR 127.37, ES 126.48 (EVO 1.A); all three gave the language's best plain line at its horizon |
| D2 | total physical hours with a web source | RECOMMENDED | **~200-220 h** (FR G1E/H2: 216.3) | OBSERVED: FR-OOD1 -4.55 (RIF l.1042). A benefit above ~220 h is UNKNOWN: IT's 130 h of new same-source videos added nothing at a fixed schedule (RM12 l.420-426); the 347 h C250 never ran at a controlled horizon |
| D3 | minimum hours below which the recipe fails | UNKNOWN | - | No Romance run used < 119 h, apart from the single-source 40 h runs (FR-0 65.85, a different recipe) |
| D4 | number and type of sources | REQUIRED | 4: read audiobook (MLS 40 h cut over **all** speakers), crowd-read (CV17, 40 h), parliament (VoxPopuli, 40 h), FLEURS train (all) | OBSERVED: the A2 design (MD l.19-31; RF22 §9); every good Romance run had them (EVO §5 obs. 6) |
| D5 | 5th, teacher-labelled web source (Granary-type) | RECOMMENDED when it exists with a teacher | additive, never substitutive | OBSERVED: IT E3G (substitution) NULL vs C-add25 (addition) win (RM12 l.126-157, 309-333); FR G1 HARM at 8k vs NULL in-domain / OOD -4.55 at 11,881 (RIF §J-M). For ES its value is UNKNOWN (never run) |
| D6 | speaker caps | REQUIRED | CV <= 20 clips / speaker; VP <= 60 / speaker; MLS cut spread over all speakers; budget spread over every train shard | OBSERVED: `prepare_mix.py` / `data_mix.sh` defaults (J:data_mix l.7); RF22 §9 |
| D7 | minimum speaker count | UNKNOWN | report per source only | OBSERVED counts: MLS 86-142, CV 1,358-4,795, VP 136-236 (EVO 1.A). AR CV had 23 speakers, a known weakness with no threshold derived (MAT AR) |
| D8 | accent / geographic diversity | UNKNOWN | record the FLEURS variant (e.g. es_419) and the country metadata where available | no accent-controlled experiment exists |
| D9 | style composition | REQUIRED (as built) | read (MLS, CV, FLEURS) + semi-spontaneous parliament (VP) + web (Granary) | OBSERVED A2 / G1E. Conversational, telephony and far-field/noisy data: none in any run, so the effect is UNKNOWN. Gain augmentation +-10 dB was NULL (IT, MD l.47-51) |
| D10 | clip duration | REQUIRED | source filter 1-20 s for CV / VP / FLEURS / web pieces (`prepare_mix.py`; web split at pauses, `--yg-split-long 20`); MLS as cut (8-30 s); trainer `--max-dur 30` Stage 1, 20 Stage 2 | OBSERVED: IT C250, FR, ES. AR used 20 for Stage 1, and 20 vs 30 was not compared (UNKNOWN) |
| D11 | mix weights | REQUIRED | A2: MLS 0.35 / CV 0.30 / VP 0.20 / FLEURS 0.15. With a web source: A2 x (1 - W), web = W, with W from the IT `p1_additive` rule (1.5 passes over the external hours). FR: W = 0.3267, giving 0.2356 / 0.2020 / 0.1347 / 0.1010 / 0.3267 | OBSERVED (RIF §J l.664-675; J:s5fr MIX) |
| D12 | source dominance cap | UNKNOWN | none enforced. ES audio share MLS 50.8 % (S5-prereg §1) | no cap was ever tested |
| D13 | teacher filter, parliament | REQUIRED | Nemotron 0.6B teacher WER vs reference **<= 50** (`a2_vpfilter.sh FILTER_ONLY=1`), plus the 300-clip teacher CLI gate | OBSERVED: halves VP blanking on IT (MD l.29-31); keeps 80.7 % IT, 98.8 % FR (RIF l.605-606); ES gate 298/300 |
| D14 | teacher filter, web source | REQUIRED if D5 | two-teacher agreement (Nemotron vs the source's own text). T = the strictest of {5, 10, 15, 20} keeping >= 90 % of the videos AND >= 50 % of every dir's hours, model-free; label = teacher text; drop empty teacher texts and duplicates | OBSERVED: FR rule gave T = 20 (RIF §J l.656-662); IT used 15 (pre-rule). Teacher contamination by its own training data (MLS / CV / FLEURS) is UNKNOWN (SUB15 Rank 3) |
| D15 | teacher relabelling of human sources | REQUIRED (do NOT) | keep human labels | OBSERVED: P2-labels NULL (RM12 l.159-186) |
| D16 | DEV leakage | REQUIRED | DEV = the official test splits (MLS test, FLEURS eval, CV test 200, VP test 200); `two_stage.subsample` rule; SHA256SUMS recorded **before training** | OBSERVED: harness-es seal 8e9c226f (RF22 §9) |
| D17 | FINAL / OOD leakage | REQUIRED | built model-free and sealed before any score. Text exclusion = every train manifest + DEV + OOD; VP speakers of our pools excluded; word 8-gram dedup of the web pool vs DEV / FINAL; speaker exclusion wherever the speaker lists are readable, otherwise a declared limitation | OBSERVED: FINAL-FR (RIF l.565-593), FINAL-FR-v2 (RF22 l.220-224, l.339-340), Granary leak dedup (RIF l.655) |
| D18 | licence | REQUIRED | train on CC0 / CC-BY sources only; NC / ND sets only as private eval | OBSERVED: RIF l.600 |

### 2.3 EXPOSURE CONTRACT

| item | value | tag / evidence |
|---|---|---|
| batch | 16 clips / step | OBSERVED, all runs |
| step count N | `round(4.60 x H_phys / h_step)`, with h_step = 16 x the weight-averaged mean clip duration (h); **cap 23,762** | OBSERVED: the ES-1 pre-registered rule (RF22 §9 l.261-264) |
| N for the D1-only mix (~126 h) | ~12,670 | OBSERVED, ES-1 |
| N for the D1 + D5 mix (~216 h) | 23,762 = FR-H2 | COMPUTED: 4.60 x 216.3 / 0.0419 ≈ 23,750 |
| audio-hours seen | ES 582.2 h (h_step 0.04592); H2 995.9 h (0.0419) | OBSERVED, HUB metrics |
| effective passes | 4.60 overall by construction. Per source it is uneven: ES MLS 7.4, CV 2.2, VP 2.6, FLEURS 14.0 | OBSERVED, S5-prereg §1 |
| LR | encoder 4e-4, decoder / joint 3e-4 (AdamW, wd 1e-3, clip 1.0) | OBSERVED: IT S3 tournament; 8e-4 flat, 1.6e-3 overshoots (RM15 l.222-323) |
| schedule | WSD: warm-up steps/20, flat, cosine decay over the last 20 % to 1 % of peak; final = last step | OBSERVED, code l.385-394 |
| S5 bucketing | `--bucket-pool 32` (pools of 32 x 16 drawn rows, sorted by duration, served in shuffled order) through the stable phase. At the **first pool refill at or after ceil(0.8 N)** the batches become exactly the unbucketed sampler's batches (`--bucket-decay-mode none`) | OBSERVED: code (train_speed.py l.207-212); replay-exact on ES (S5-prereg §3) |
| switch step | N = 11,881: 9,505 → **9,536**. N = 12,670: 10,136 → **10,144**. N = 23,762: 19,010 → 19,040 (COMPUTED, 595 x 32) | OBSERVED: the first two (P:S5-FR amend. 1; S5-prereg §3) |
| switch logging | `switched_at` is stored but **not printed** | OBSERVED (restart-2026-10-25). Open item O1 |

**Passes vs steps: which one is causal?** The evidence is weighed here and not resolved:

- **For the steps / horizon reading (stronger).**
  - At fixed steps, passes did not predict WER: FR-E1 4.25 vs G1E 2.30 passes gave 25.41 vs 25.54; IT C250 2.06 vs
    oldrep 3.30 gave 27.66 vs 27.53 (EVO §5 obs. 2).
  - The cost-curve agent: "no passes-based rule holds" (RF22 l.432-435).
  - At fixed data, more steps helped every time: ES S1 (0.6x horizon) +2.68 FAIL (S14); FR G1E → H2 (2x steps)
    DEV -3.48 / OOD1 -4.00.
  - IT: W4 8k → 16k -3.36, but W4-16k (22.76) ≈ cosine-36k (22.27). IT saturates around a ~16-36k horizon on 348 h
    (RM15 l.401-412).
- **For passes.** The 4.60 number comes from one run (FR-H2). ES reproduced it by construction, so it never tested
  it.
- **Reading (HYPOTHESIS).** What tracks in-domain WER is the number of updates under a full WSD horizon. "4.60
  passes" is a convenient sizing rule that coincides with good step counts (12.7k-23.8k) for pools of 126-216 h.
- **For Factory v1.** Keep the rule as written (it was validated prospectively once, on ES), and keep the 23,762
  cap. For a pool above ~216 h the rule and the cap disagree; which one predicts WER is UNKNOWN (EVO §5 obs. 4). The
  untested FR-H3 (> 23,762) is the deciding experiment.
- **Never** shorten N to save compute: S1 shows a 40 % cut costs +2.68, while S5 saves 33-40 % at equal quality.

### 2.4 EOU

- **Directly validated:** Stage 2 v4 on **IT** (C250-36k parent: FINAL-IT FE 3.50 %, R 96.17 %, p50 0.88 s) and
  **FR** (G1E parent: FINAL-FR FE 7.57 %, R 96.73 %, p50 0.80 s; H2 parent: FINAL-FR-v2 FE 12.70 %). Three parents,
  two languages, all trained with the random sampler. OBSERVED, EVO 1.C.
- **ES is NOT EOU-qualified:** Stage 2 was not run and no DEV-EOU-ES bank exists (S14 l.40-48). No S5 parent has
  ever had Stage 2 (UNKNOWN). Plan: `es-stage2-qualification-plan.md`.
- **Known failure mode:** false endpoints at internal pauses of about 1 s or more. The model's rule is close to "~1 s
  of silence" (AUT §2). The internal-class FE is 55.2 % for v1 and 68.4 % for v1.1. Pause-stretch negatives FAIL
  (AUT §6); the truncated-prefix arm is closed. This is a declared limitation, not a gate.

**Definition: a Factory model is EOU-qualified when all of the following hold.** The gates are the RC-FR
pre-registered ones, i.e. the RC-IT acceptance level (RIF §N l.1086-1096).

| # | gate (Stage 2 vs its own Stage-1 parent, `s2_reset` protocol, same tool, same box) | threshold |
|---|---|---|
| E1 | DEV macro reset-WER delta | <= +1.0, and no DEV source > +2.0 |
| E2 | DEV empty | <= parent + 1 % absolute |
| E3 | OOD macro delta (decoded ONCE, on the single checkpoint the DEV rule selected) | <= +1.0 |
| E4 | EOU recall (DEV, pooled) | >= 94 % |
| E5 | FE rate (`<EOU>` with >= 0.5 s of speech left; DEV pooled) | <= 10 % |
| E6 | endpoint latency p50 (DEV) | <= 1.0 s |
| E7 | FAIL, any of | delta > +2.0 (DEV or OOD), R < 90 %, FE > 15 %, p50 > 1.5 s |
| E8 | product parity | Mynah CUDA bf16 within ~0.1 of NeMo on DEV (FR 25.64 vs 25.68; ES plain 18.09 vs 18.09); CPU == CUDA per clip |
| E9 | reported, not gated (until the bank exists) | internal-pause-class FE on a DEV-EOU-<lg> bank (built model-free, the DEV-EOU-FR-v2 design) |

### 2.5 COST on GB10 (DGX Spark; GB10 ≈ 3x slower than an L40S for this trainer, RM15 l.15-17)

| run | steps | baseline (random) wall | S5 wall | saving | tag |
|---|---|---|---|---|---|
| ES-1 mix, 126.5 h | 12,670 | 13,326 s (1.052 s/step) | 9,011 / 8,906 s (0.711 / 0.703 s/step) | -32.4 / -33.2 % | OBSERVED (S14; HUB) |
| FR G1E mix, 216.3 h | 11,881 | 14,737 s (1.240 s/step) | 8,828 s (0.743 s/step) | -40.1 % (box factor 0.997) | OBSERVED (GATE5) |
| FR phase rates | - | stable 1.215, decay 1.278 s/step | stable 0.600, decay 1.276 s/step | 0.49x / 1.00x | OBSERVED (GATE5) |
| peak reserved VRAM, FR | - | 42.25 GB | 42.99 GB | +0.7 GB | OBSERVED (metrics.json on dgx1 / dgx2) |
| **Factory v1, D1 only (~126 h)** | ~12,670 | ~3.7 h | **~2.5 h** | ~-33 % | OBSERVED (= ES) |
| **Factory v1, D1 + D5 (~216 h)** | 23,762 | ~8.2 h (23,762 x 1.240) | **~4.8-4.9 h** (19,040 x 0.600 + 4,722 x 1.276 ≈ 17.4 ks, plus warm-up and evals; or 23,762 x 0.743 = 17.7 ks) | ~-40 % | COMPUTED / HYPOTHESIS (FR rates, applied linearly to 2x the steps) |
| Stage 2 v4 | 1,500 | Ada 816-859 s; GB10 UNKNOWN | (random sampler) | - | HYPOTHESIS ~0.7 h on GB10 (Ada ≈ L40S, GB10 ~3x) |
| teacher passes (VP t50; web agreement) | - | UNKNOWN on GB10 | - | - | UNKNOWN (the AR F7 pass wall is also UNKNOWN) |

- **Expected saving band from S5: 32-40 %** (OBSERVED in two languages).
- HYPOTHESIS: the saving grows with the spread of clip durations (padding share). The FR mix (long MLS clips plus
  short web pieces) saved more than ES.
- **Expected end-to-end GPU time** for a new language with D1 + D5: ~5-6 h (Stage 1 S5 + Stage 2 + probes),
  excluding the teacher passes (HYPOTHESIS).
- Data download and cutting are CPU and network work: hours, bounded by the Hub line (RF22 §10).

### 2.6 QUALITY

**(a) Demonstrated historical results** (OBSERVED; each on its own bank, never cross-compared):

| line | value | caveat |
|---|---|---|
| ES-1 plain, harness-es, streaming greedy (Mynah bf16) | **18.09** (beam 16.55, oracle-8 12.98); S5 18.08 / 18.07 | DEV harness, later reused to select the compute recipe. No FINAL-ES, no OOD-ES, no EOU |
| FR, S5-FR, GB10 DEV / OOD1 | 25.91 / 30.54 (B 25.83 / 30.48) | G1E horizon (11,881); one paired seed |
| FR-H2 DEV / OOD1; v1.1 | 22.06 / 25.66; v1.1 22.08 / 24.97 | random sampler |
| FINAL-FR (v1, G1E + v4) | **27.76**, FE 7.57 % | one shot, spent |
| FINAL-FR-v2 (v1.1, H2 + v4) | **23.00**, FE 12.70 % | one shot, spent, different banks from FINAL-FR |
| FINAL-IT (RC-IT v4) | **20.49** (CER 6.95), FE 3.50 % | off-recipe parent (cosine, enc 2e-4, 36k); missed the pre-registered < 20 by 0.49 |
| AR | FLEURS ta-fold 33.28 / 34.19 | different metric, script and normaliser: no Romance claim |

**(b) Expected Factory-v1 band for a new Romance language** (HYPOTHESIS):
- Own 4-set DEV harness, streaming greedy: **~18-26**. The low end is the ES-like case (D1 only). The high end is the
  FR-like case at the G1E horizon; FR at H2 reached ~22.
- Sealed FINAL: plausibly +0 to +2 over DEV. FINAL-IT 20.49 was 1.8 *below* DEV730 22.27; FINAL-FR 27.76 was 2.1
  above DEV 25.64. The sign is language- and bank-dependent, so treat this as UNKNOWN in practice.
- The ES pre-registered reading bands apply: <= 25 strong transfer, 25-30 transfers, 30-40 weak, > 40 or abort means
  an assumption failed (RF22 §9).

**(c) SLAs** (defined on the language's **sealed FINAL**, streaming greedy, Mynah bf16 product path; a DEV-only
number does not satisfy an SLA):

| SLA | definition | confidence for a new language | evidence |
|---|---|---|---|
| **SLA-A** | macro WER **< 20** greedy streaming | **MEDIUM-LOW.** Met on DEV for ES only (18.09). Missed on FINAL-IT by 0.49 (off-recipe parent). Not met on FR (FINAL-FR 27.76, FINAL-FR-v2 23.00). No sealed FINAL below 20 exists | ES 18.09 is the only < 20 point, and it is on a reused DEV |
| **SLA-B** | macro WER **< 15** greedy streaming | **UNPROVEN.** No line has reached it | ES greedy 18.09 → beam 16.55 → oracle-8 12.98: the whole search gap (1.54) is less than the 3.09 needed, so the candidate set itself must improve (SUB15 §3). First experiment: ES-D2, predicted -1.3 (HYPOTHESIS) |

## 3. What we can claim today / what we cannot

**Can claim (OBSERVED):**
- A frozen, pre-registered plain recipe gave strong transfer on the first attempt, on a language not used to design
  it: ES 18.09 on a 4-set DEV, Mynah CPU == CUDA 764/764.
- Stage 1 recipe components with direct ablation evidence:
  - stock init beats a Romance warm start;
  - whole-encoder adaptation;
  - W4 LRs and WSD;
  - the VP teacher filter;
  - additive, not substitutive, web data;
  - the full WSD horizon is needed (S1 FAIL).
- S5 cuts Stage-1 GPU time by 32-40 % at unchanged macro quality in two languages:
  - ES: two seeds, 18.08 / 18.07 vs 18.09;
  - FR: one paired seed on the same GPU, DEV +0.08 / OOD1 +0.06, STRONG.
  - The mechanism is causal on/off: 0.49x while bucketed, 1.00x after the switch.
- Stage 2 v4 ports unchanged across two languages and three parents, and passes the RC gates each time.
- A web source buys out-of-domain robustness (FR OOD1 -4.55) at a long enough horizon, at no in-domain cost.

**Cannot claim:**
- That IT, FR, ES or AR used "the Factory v1 recipe". None did exactly (section 1).
- Statistical equivalence of S5 and the baseline: ES-1 has one seed and S5-FR one paired seed.
- Any EOU property of an S5 parent, or of any ES model.
- Any SLA-A result on a sealed FINAL bank for a Factory-recipe model. SLA-B is unproven everywhere.
- That passes (rather than steps) are the causal horizon variable, or the right N for pools above 216 h.
- A minimum hours, speaker, accent or source-dominance threshold.
- Robustness to conversational, telephony or noisy audio: none of these was in any training or DEV bank.
- Transfer to non-Latin scripts. AR needs romanisation, a normaliser decision and orthography audits.
- Cross-language WER comparisons.

**Reasons against S5 as the default.** None is blocking; S5 stays the DEFAULT CANDIDATE. Three watch items:
1. **ES CV.** S5 CV is 21.98 / 23.03 vs ES-1 21.18, worse on both seeds, and the CV decay-recovery hypothesis did not
   replicate (S14). FR CV is flat (-0.05).
   - Treat it as a per-short-clip-domain watch item: report CV and the short-clip buckets per run.
2. **No Stage 2 on an S5 parent yet.** Stage 2 itself uses the random sampler, and S5's decay batches are the
   baseline's. HYPOTHESIS: low risk. The first Factory model must still pass section 2.4 itself.
3. **The switch is not logged** (O1). Until it is, the switch is verified only by the replay and the phase rates.

## 4. Open items (one line each; owner doc in brackets)

- **O1. `switched_at` logging.** Print the step at which PoolBatcher switches to unbucketed. Gate every S5 run on it
  matching ceil(0.8 N) rounded up to a pool boundary [this doc; restart-2026-10-25.md Lessons].
- **O2. ES Stage-2 qualification.** v4 on an ES S5 parent, a DEV-EOU-ES bank with an internal-pause class, FINAL-ES
  and OOD-ES sealed first, gates E1-E9 [`es-stage2-qualification-plan.md`].
- **O3. Prospective new-language validation.** Factory v1 as frozen here, on a language never used in design, with
  every contract row filled and banks sealed before data stats [`factory-v1-prospective-language.md`].
- **O4. SUB15 line.** M1 / M2 measurements (~15 GPU-min, pre-registered in trainB plans/SUB15-M1M2-prereg.md), whose frozen decision rules choose the first SUB15 FT (ES-D2 is no longer the automatic choice)
  [`sub15-error-budget-plan.md`].
- **O5. Horizon variable.** Passes vs steps, and N above 23,762 (FR-H3 deferred) [EVO §5, §7; MAT exposure contract].
- **O6. Web-source value without FR.** An ES G1E analogue: Granary-es availability UNVERIFIED [MAT ES row; SUB15
  Rank 3].
- **O7. Seed spread at the full horizon for FR / IT.** Only ES has two-seed data [EVO §6].

**Blocking "Factory v1 CLOSED":**
- O2 or O3 producing an EOU-qualified S5-parent model;
- a sealed FINAL for a Factory-recipe model (O3);
- O1 logged.

The other items are refinements.

## 5. Update 2026-10-26 evening: the horizon result (OBSERVED; trainB plans/SUB15-ESD2-prereg.md)
At the same 126.5 h ES-1 mix with the S5 recipe, raising the steps from 12,670 to 23,762 gives:

| run | steps | greedy | Δ vs S5 s0 (18.08) |
|---|---|---|---|
| ES-H3 seed 0 | 23,762 | 15.35 | -2.73 |
| ES-H3 seed 1 | 23,762 | 15.66 | -2.42 |
| two-seed mean | | **15.51** | (REPLICATED) |
| ES-D2 seed 0, 246 h | 23,762 | 15.07 | D2 - H3 = -0.28 |

- The attribution is HORIZON LEVER: doubling the hours adds about 10 % of the gain.
- GB10 wall is about 4.6 h with S5, versus 2.5 h at 12,670 steps.
- The NO-GOOD-HYP mass fell by about 1.8 points: the candidate set improves, not only greedy.

**Exposure contract, proposed v1.1 (supersedes section 2.3's rule for ~126 h pools):**
- N = 23,762 steps for a ~126 h four-source pool (9.4 passes).
- The "4.6 passes" rule undersized the horizon at this pool size and is retired as a sizing rule. The horizon above
  23,762 is still UNKNOWN.

**SLA-A** (< 20 greedy): HIGH confidence on DEV for ES (two seeds at about 15.5). It is still DEV only.

**SLA-B** (< 15): not reached. D2 seed 0 is 15.07; D2 seed 1 is pending. No sealed FINAL-ES exists.

**EOU:** Stage 2 v4 on H3 is MARGINAL (reset-WER +1.15, recall 93.2 %, FE 11.7 %). No EOU-qualified
Factory-v1.1 model exists yet. Stage-2 preservation is a separate line, pre-registered before any GPU use.
