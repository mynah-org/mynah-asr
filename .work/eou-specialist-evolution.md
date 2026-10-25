# EOU-120M language specialists: how the recipe evolved IT -> FR -> ES -> AR (reconstruction, 2026-10-25)

Status: research reconstruction from the archived reports, pre-registrations, job scripts and Hub `metrics.json`
files. No training, no decode, no GPU. Sealed banks were not decoded: FINAL-IT, FINAL-FR, FINAL-FR-v2 and
FINAL-EOU-FR-v2 numbers below are quoted from the archived one-shot results.

**Read this first.** IT, FR, ES and AR are a research **sequence**, not four applications of one recipe. Each
language started from what the earlier ones had taught:
- **IT** was the laboratory where the recipe was discovered.
- **FR** refined and replicated the method.
- **ES** is the cleanest replication of a frozen recipe.
- **AR** is a cross-script stress test that works through romanisation. It is NOT direct evidence about the Romance
  factory.

The WER numbers of different languages sit on different eval banks with different normalisers. They are never
comparable across languages (section 4).

Tags: **OBSERVED** = measured and archived. **HYPOTHESIS** = an interpretation that was not tested. **UNKNOWN** = no
evidence was found. **COMPUTED** = arithmetic from archived numbers, done in this document.

## 0. Source key (every number carries one)

| key | file |
|---|---|
| FT | `.work/eou-it-ft.md` |
| MD | `.work/eou-it-multidomain-2026-10-10.md` |
| BN | `.work/eou-it-bottleneck-2026-10-11.md` |
| RM12 | `.work/it-120m-sub20-roadmap-2026-10-12.md` |
| RM15 | `.work/it-120m-sub20-roadmap-2026-10-15-dgx.md` |
| RIF | `.work/release-it-fr-2026-10-19.md` |
| AUT | `.work/fr-v2-eou-autopsy-2026-10-21.md` |
| RF22 | `.work/release-final-2026-10-22.md` |
| CF23 | `.work/release-final-2026-10-23-cheapft.md` |
| S14 | `.work/cheapft-S1-S4-prereg-2026-10-23.md` |
| R24 | `.work/restart-2026-10-24.md` |
| S2N | `.work/stage2-v4-vs-nemo-asr-eou-2026-10-24.md` |
| P:<name> | `mynah-asr-wt-trainB/finetune/eou/plans/<name>.md` |
| J:<name> | `mynah-asr-wt-trainB/finetune/jobs/<name>.sh` |
| HUB:<run> | Hub `gabrione/mynah-asr-canary-it-experiments`, `runs/<run>/metrics.json` (`args`, `wall_s`, `audio_h_seen`, `audio_h_per_gpu_h`, `peak_reserved_gb`). Fetched 2026-10-25 on dgx2 into `~/ft/research-evo/hub/` |

All `.work` paths are relative to `mynah-asr-wt-finetune/`. "Passes" = audio-hours seen / unique physical hours.
That is COMPUTED, except where a report states the value.

## 1. Milestone runs

The requested columns are split over four tables (A identity and data, B optimisation and exposure, C evaluation,
D cost and conclusions). Each run keeps the same row id in every table.

Recipe shorthands used below:
- **E3 recipe:** stock init, whole encoder 2e-4 from step 0, decoder / joint 3e-4, cosine, bs 16
  (BN l.200-218; HUB args `enc_lr 0.0002`, no `lr_shape` key, i.e. the pre-WSD cosine code).
- **W4 recipe:** stock init, whole encoder 4e-4, decoder / joint 3e-4, AdamW (0.9, 0.98), wd 1e-3, WSD with a 5 %
  warm-up (steps / 20) and the last 20 % as decay, bs 16 (RM15 l.317-320).
- **Stage 2 v4:** parent's own mix; top-4 encoder blocks 3e-5, decoder / joint 2e-5; cosine over 1500 steps;
  text + `<EOU>` on every draw; pad p 0.5 with U(1, 3) s of -90..-60 dB; FastEmit 0; max-dur 20.
  final.nemo = the lowest in-run WER among evals with EOU >= 80 % and empty <= 10 % (RIF l.1063-1067; S2N §1 l.37-97).
- **S5 sampler:** `--bucket-pool 32` (duration-sorted pools of 32 batches) plus `--bucket-decay-mode none`, which
  serves unbucketed batches from the first pool boundary of the decay phase (P:S5-prereg §3; J:cf_s_full l.31).

### 1.A Identity and data

| id | language / role | parent | tokenizer | script / normaliser | physical datasets (weight) | UNIQUE physical h | speakers / groups | teacher filtering |
|---|---|---|---|---|---|---|---|---|
| IT-0 P40 (+S2 = EOU-IT v1), 2026-10-09 | IT, discovery | stock EOU-120M | stock SPE 1024 + `<EOU>`/`<EOB>` | Latin, de-accented `[a-z' ]` (the stock SPE maps accented vowels to `<unk>`) (FT l.141-148) | MLS-it 40 h cut (1.0) | 40.0 (MD l.23) | 65 MLS speakers | no |
| IT-B2 (+S2 = EOU-IT v2), 2026-10-10 | IT, discovery | stock | stock | as IT-0 | A2 = MLS 40 h (0.35), CV17 <= 20 / speaker (0.30), VP t50 (0.20), FLEURS train (0.15) | 119.4 COMPUTED (40 + 40 + 32.6 + 6.8, MD l.19-31) | MLS 65, CV 1,358, VP 141 (before filtering), FLEURS UNKNOWN | VP: Nemotron 0.6B teacher WER <= 50 (MD l.29-31); VP-it keeps 80.7 % (RIF l.606) |
| IT-E3 (+S2 djlr = EOU-v3, production until RC-IT), 2026-10-11 | IT, discovery | stock | stock | as IT-0 | A2 (same weights) | 119.4 | as B2 | VP t50 |
| IT-E3-8k, 2026-10-12 | IT, discovery | stock | stock | as IT-0 | A2 | 119.4 | as B2 | VP t50 |
| IT-C-add25, 2026-10-12 | IT, discovery | stock | stock | as IT-0 | A2 x0.75 + it-yg agree <= 15 (0.25) (HUB:plain-it-p1-add25 mix) | 217.5 COMPUTED (119.4 + 98.1) | + 3,704 Granary videos (RM12 l.312) | VP t50 + Granary two-teacher agreement (Nemotron vs Granary-Whisper) <= 15; label = teacher text (MD l.105-112) |
| IT-C250, 2026-10-13 | IT, discovery | stock | stock | as IT-0 | MLS 0.1964, CV 0.1684, VP t50 0.1122, FLEURS 0.0842, it-yg 0.1885, it-yg6 0.2503 (HUB:plain-it-c250) | 347.7 COMPUTED (+130.2 h it-yg6); the reports say "~357 h" (RIF l.79, l.230; RF22 l.159). The ~9 h gap is UNRESOLVED | + 3,752 new it-yg6 pool videos, 0 overlap with it-yg (RM12 l.352-372) | as C-add25 (agree <= 15 on both pools) |
| IT-C250-oldrep, 2026-10-13 | IT, discovery (diversity vs exposure test) | stock | stock | as IT-0 | A2 + it-yg only, at W 0.4388 (HUB:plain-it-c250-oldrep) | 217.5 COMPUTED | 3,704 videos | as C-add25 |
| **IT-C250-36k (RC-IT parent) + Stage 2 v4 = RC-IT v4 (FROZEN)**, 2026-10-14 / 18 / 19 | IT, discovery; the IT release | stock -> C250-36k (7e919281) -> v4 (0a4d1b10) | stock | as IT-0 | C250 mix (above); stage 2 on the same mix | 347.7 (or ~357, see C250) | as C250 | as C250 |
| IT-S3-4k (origin of W4), 2026-10-15 | IT, recipe tournament | stock | stock | as IT-0 | A2 | 119.4 | as B2 | VP t50 |
| IT-W4-C250-8k / -16k, 2026-10-15 / 16 | IT, W4 on the big mix (screening) | stock | stock | as IT-0 | C250 mix | 347.7 | as C250 | as C250 |
| FR-0 old p40 (+s2-eou), 2026-10-09 | FR, first naive replica | stock | stock | de-accented `[a-z' ]`, no FR pre-map yet | MLS-fr, 6 of 34 shards (1.0) | 38.58 (FT l.204-206) | 37 speakers | no |
| FR-P1, 2026-10-19 | FR, methodology replication | stock | stock | `textnorm fr`: oe / ae ligatures, curly apostrophes, then the IT rule (RIF l.443-447) | A2-fr: MLS-fr 40 h cut over all 142 speakers (0.35), CV17-fr (0.30), VP-fr t50 (0.20), FLEURS fr_fr (0.15) (RIF l.449-458) | 127.37 (RIF l.779): MLS 40.0, CV 40.03, VP 39.66, FLEURS 7.69 (HUB data/a2-fr/hours.json) | MLS 142, CV 4,795, VP 236, FLEURS 1,187 ids | VP-fr t50 keeps 15,997 / 16,191 (RIF l.605) |
| FR-G1, 2026-10-19 | FR, data-diversity test at 8k | stock | stock | as FR-P1 | A2-fr x0.6733 + Granary-fr agree <= 20 (0.3267) (RIF l.664-675) | 216.3 (RIF l.720) | + 1,408 Granary videos (RIF l.660) | VP t50 + Granary agree <= 20 (T chosen by a model-free rule; label = teacher text) (RIF l.658-662) |
| FR-E1, 2026-10-20 | FR, horizon test | stock | stock | as FR-P1 | A2-fr | 127.37 | as FR-P1 | VP t50 |
| **FR-G1E (+ FR-S2 = RC-FR v1, FROZEN)**, 2026-10-20 / 21 | FR, mix x horizon; FR release v1 | stock -> G1E (1271deff) -> FR-S2 (7814141d) | stock | as FR-P1 | G1 mix | 216.3 | as G1 | as G1 |
| **FR-H2 (+ s2v4-h2 = RC-FR v1.1 candidate)**, 2026-10-22 | FR, exposure; the source of the "4.60 passes" rule | stock -> H2 (d2062aca) -> 3130534d | stock | as FR-P1 | G1 mix | 216.3 | as G1 | as G1 |
| FR-v2 stage 2 (pause-stretch), 2026-10-21 | FR, EOU intervention | G1E | stock | as FR-P1 | G1 mix + stretched internal pauses, p = 0.2 (AUT §6) | 216.3 | as G1 | as G1 |
| S5-FR (in progress), 2026-10-25 | FR, compute-recipe replication | stock | stock | as FR-P1 | G1 mix (P:S5-FR-prereg) | 216.3 | as G1 | as G1 |
| **ES-1**, 2026-10-23 | ES, cleanest frozen-recipe replication | stock | stock | `textnorm es` (NFKD, typographic apostrophes, digit clips dropped) (RF22 l.249-252) | MLS-es 40 h (0.35), CV17-es (0.30), VP-es t50 (0.20), FLEURS es_419 (0.15). No Granary-es (declared limitation, RF22 l.254-255) | 126.48: 40.00 / 40.02 / 39.85 / 6.62 (HUB runs/dgx-es1-w4-12670/data_stats.txt) | MLS 86, CV 3,566, VP 136, FLEURS 1,150 ids | VP-es t50; Nemotron CLI gate 298 / 300 (RF22 l.480-482) |
| ES-S2 (seeds 0, 1), 2026-10-23 | ES, compute | stock | stock | as ES-1 | ES-1 mix | 126.48 | as ES-1 | as ES-1 |
| **ES-S5 (seeds 0, 1) = preferred compute recipe**, 2026-10-24 | ES, compute | stock | stock | as ES-1 | ES-1 mix | 126.48 | as ES-1 | as ES-1 |
| AR-A0, 2026-10-24 | AR, cross-script plumbing | stock (T3: NVIDIA Arabic text side) | R: stock via romanisation; T: new SP-1024; T3: NVIDIA Arabic SP-1024 + `<EOU>`/`<EOB>` (P:AR-A0 arms) | `norm_ar` (strips tashkeel / tatweel, folds the alef-hamza forms), then a reversible romanisation into `[a-z']` for R | 32 CV17-ar clips | ~0.04 h (32 clips of 2-8 s) | 23 speakers (P:AR-A0 Data) | no |
| AR-A1-R, 2026-10-24 | AR, learning curve | stock | stock via romanisation | `norm_ar` + romanise; scored in Arabic script | CV17-ar train (0.70), FLEURS ar_eg train (0.30) | 36.96: CV 32.34 + FLEURS 4.62 (HUB runs/dgx-ar-a1-r-s4000/data.json) | **CV 23 speakers**, FLEURS 1,022 ids | no |
| **AR-A2-R (seeds 0, 1)**, 2026-10-24 / 25 | AR, data diversity + horizon | stock | as A1 | as A1 | CV17-ar (0.25) + MASC train shards 01-03 after F1-F7 (0.75). FLEURS evaluation-only (P:AR-A2 §0, §3) | 177.1 COMPUTED: CV 32.34 + MASC 144.8 (P:AR-A2 amendment 2) | CV 23; MASC candidates span 11,355 videos (kept-pool videos UNKNOWN) | MASC: Nemotron `?lang=ar`, teacher CER <= 15 in ta-fold space; keeps 134,192 / 277,154 rows (clean 74.1 %, noisy 29.6 %) (HUB teacher_stats.json) |
| AR-D2-R (seeds 0, 1), 2026-10-24 / 25 | AR, target orthography | stock | as A1 | as A1, plus ta marbuta restored in MASC targets from a frozen lexicon of 1,644 forms (P:AR-D2) | as AR-A2 | 177.1 | as AR-A2 | as AR-A2 |

### 1.B Optimisation and exposure

| id | steps | batch | audio-h seen (HUB) | passes = seen / unique | encoder | LR / decay shape | sampler | max-dur | Stage 2 EOU |
|---|---|---|---|---|---|---|---|---|---|
| IT-0 P40 | 4,000 | 16 | 264.0 | 6.6 | top-4 at 3e-5, decoder / joint 3e-4 | cosine (warm-up not in args) | single manifest | 20 | S2 (v1): decoder / joint 1e-4, top-4 3e-5, pad 0.5 U(1, 3), 1500 steps (HUB:plain-s2-eou-p40-b2pol) |
| IT-B2 | 4,000 | 16 | 187.2 | 1.57 | gradual unfreeze 0:4, 0.2:8, 0.5:all at 3e-5 | cosine | random mixture | 20 | v2 = S2 on B2, top-4 (MD l.53-70) |
| IT-E3 | 4,000 | 16 | 187.2 | 1.57 | **whole encoder 2e-4 from step 0**, decoder / joint 3e-4 | cosine | random mixture | 20 | v3 = `s2-e3-djlr`: decoder / joint **2e-5** (from 1e-4), step 250 (BN l.256-300) |
| IT-E3-8k | 8,000 | 16 | 374.1 | 3.13 | E3 | cosine | random | 20 | none |
| IT-C-add25 | 10,667 | 16 | 520.3 | 2.39 | E3 | cosine | random; W = 0.25 keeps E3-8k's A2 draws (RM12 l.309-316) | 30 | none |
| IT-C250 | 14,254 | 16 | 716.1 | 2.06 | E3 | cosine | random; W 0.4388 | 30 | none |
| IT-C250-oldrep | 14,254 | 16 | 717.3 | 3.30 | E3 | cosine | random; W 0.4388 on the old pool only | 30 | none |
| IT-C250-36k + v4 | 36,000; stage 2 1500 (step 250 selected) | 16 | 1,808.0; stage 2 72.3 | 5.20 on 347.7 h (5.06 on 357 h); the reports say "~5" (RF22 l.57) | E3; stage 2 top-4 3e-5 | cosine | random | 30; stage 2 20 | **v4** (RM15 l.759-786). Later IT arms: v5-B FAIL (latency), v5-CAL NULL, logit bias NOT-SEPARABLE (RIF §E-G) |
| IT-S3-4k | 4,000 | 16 | 187.2 | 1.57 | **whole encoder 4e-4** | **WSD, 5 % warm-up, last 20 % decay** | random | 20 | none |
| IT-W4-C250-8k / 16k | 8,000 / 16,000 | 16 | 401.6 / 803.6 | 1.15 / 2.31 | W4 | WSD | random | 30 | none |
| FR-0 p40 | 4,000 (+ s2 1500) | 16 | 267.0 | 6.9 | top-4 3e-5, decoder / joint 3e-4 | cosine | single manifest | 20 | s2-eou: decoder / joint 1e-4 (before djlr), pad 0.5 (RIF l.82) |
| FR-P1 | 8,000 | 16 | 364.7 | 2.86 | W4 | WSD | random | 30 | none (by rule) |
| FR-G1 | 8,000 | 16 | 335.4 | 1.55 | W4 | WSD | random | 30 | none |
| FR-E1 | 11,881 | 16 | 541.4 | 4.25 | W4 | WSD (warm-up 594, decay from ~9,505) | random | 30 | none |
| FR-G1E + FR-S2 | 11,881; stage 2 1500 (step 250) | 16 | 498.5; stage 2 66.0 | 2.30 | W4; stage 2 top-4 | WSD; stage 2 cosine | random | 30; stage 2 20 | v4 unchanged + `--lang fr` (RIF l.1058-1075) |
| FR-H2 + s2v4-h2 | 23,762; stage 2 1500 (step 1250) | 16 | 995.9; stage 2 66.0 | 4.60 (RF22 l.261) | W4 | WSD (warm-up 1,188, decay from ~19,010) | random | 30; 20 | v4 unchanged (RF22 l.132-152) |
| FR-v2 stage 2 | 1500 (step 250) | 16 | 66.9 | - | top-4 | cosine | v4 sampler + pause stretch | 20 | v4 + `--pause-stretch-prob 0.2` (AUT §6) |
| S5-FR | 11,881 per arm | 16 | 498.5 (S5 arm) | 2.30 | W4 | WSD | B = random; S5 = bucket 32 + unbucketed from step 9,536 | 30 | none |
| ES-1 | 12,670 = round(4.60 x 126.48 / 0.04592) (data_stats.txt) | 16 | 582.2 | 4.60 | W4 | WSD | random | 30 (longest clip 20 s) | **NOT RUN** (R24 queue item 7) |
| ES-S2 s0 / s1 | 12,670 | 16 | 582.2 / 581.4 | 4.60 | W4 | WSD | bucket 32 sorted | 30 | not run |
| ES-S5 s0 / s1 | 12,670 | 16 | 582.2 / 581.5 | 4.60 | W4 | WSD | S5 (unbucketed from step 10,144) | 30 | not run |
| AR-A0 | 1,000 | 16 | UNKNOWN | memorisation (32 clips) | **frozen** (the T-unfrozen diagnostic: encoder 0.1 x lr) | lr 3e-4 | 32 clips | - | none |
| AR-A1-R | 4,000 | 16 | 103.5 | 2.80 | W4 | WSD | **bucket 32 sorted (S2, not S5)** (J:ar_a1 l.21) | **20** | none |
| AR-A2-R s0 / s1 | 12,670 | 16 | 221.6 / 221.7 | 1.25 (CV ~1.8, MASC ~1.1 per clip, P:AR-A2 §3) | W4 | WSD | bucket 32 sorted (S2) | 20 | none |
| AR-D2-R s0 / s1 | 12,670 | 16 | 221.6 / 221.7 | 1.25 | W4 | WSD | bucket 32 sorted | 20 | none |

OBSERVED in the AR `metrics.json` args: `lang=it` on every AR run. The AR jobs pass no `--lang` (J:ar_a1, J:ar_a2),
so the romanised `[a-z']` targets go through the default (IT) text path. The impact was not audited. HYPOTHESIS:
none, because romanised text is already in the IT alphabet. AR runs also carry no `--abort-wer` (P:AR-A1 l.21-25).

### 1.C Evaluation

Primary decoding is offline greedy RNNT in every row. The plain rows use `lm_fusion_probe` greedy / malsd beam-8 /
oracle-8 with a 0.5 s flush. The EOU rows use `s2_reset.py` (frame-synchronous greedy, decoder reset after `<EOU>`,
2 s of quiet).

| id | DEV / FINAL bank | greedy WER (CER) | beam-8 | oracle-8 | EOU metrics |
|---|---|---|---|---|---|
| IT-0 P40 | MLS-it val 200; later the IT 4-set in-run val (730) | MLS 41.72 (12.25); 4-set macro 57.27 / 22.66 (MD l.37) | - | - | S2: in-run EOU 98 %; Mynah FLEURS-it 55.1 WER, speech end -> EOU p50 2,346 ms, never 10 %, premature 0 % (FT l.150-170) |
| IT-B2 | IT 4-set in-run val (MLS / FLEURS / CV / VP 200 / 200 / 200 / 130) | 49.08 (18.22) (MD l.45) | 44.84 (BN l.153) | 41.32 | v2: Mynah FLEURS-it 42.0; EOU p50 1,546 ms; never 5 % (MD l.57-66) |
| IT-E3 | same 4 sets (= DEV730, the later "harness") | 36.52 (13.53) | 33.09 | 28.52 (BN l.212-214) | v3: Mynah FLEURS-it 30.91, p50 810 ms. On DEV730 with the reset protocol: FE 5.89 %, R 94.66 %, L 0.64 / 1.92 s, reset-WER 36.07 (RIF l.37-40) |
| IT-E3-8k | DEV730 | 31.36 (11.75) | 28.27 | 23.60 | - |
| IT-C-add25 | DEV730; clean-it (TEDx 909 + ITALIC 475, development) | 29.27; clean 31.87 | 26.72; 29.08 | 22.22; 21.28 (RM12 l.320-325) | - |
| IT-C250 | DEV730; clean-it | 27.66; clean 29.23 | 24.80; 26.16 | 20.62; 18.80 | - |
| IT-C250-oldrep | DEV730; clean-it | 27.53; clean 28.62 | 24.92; 25.87 | 20.42; 18.88 (RM12 l.412-418) | - |
| IT-C250-36k + v4 | DEV730; clean-it; FINAL-IT (6 banks, 1,826 clips, one shot, spent) | plain 22.27 (9.15 in-run), clean 23.73 (flushed 21.73); **v4 DEV730 reset-WER 22.54; FINAL-IT 20.49 (CER 6.95)** (RIF l.414-432) | 20.53 | 16.25 | v4 DEV730: FE 8.36 %, R 94.38 %, L 0.72 / 1.92 s. **FINAL-IT: FE 3.50 %, FE_any 3.78 %, R 96.17 %, L 0.88 / 2.24 s; FINAL FLEURS FE 16.4 %.** Mynah FLEURS-it 17.17, EOU p50 778 ms (RIF l.509-513) |
| IT-S3-4k | DEV730 (in-run raw) | 29.96 (11.37) | - | - | - |
| IT-W4-C250-8k / 16k | DEV730; clean-it | 26.12 / 22.76 | 23.67 / 21.02 | 19.43 / 16.71 (RM15 l.348-352, l.395-399) | - |
| FR-0 p40 | MLS-fr val; later harness-fr (MLS / FLEURS / CV / VP test, 4 x 200) | MLS-fr 50.99 (26.91); **harness-fr 65.85** (40.78) (RIF l.612) | 64.88 | 61.65 | s2-eou: EOU 97 %; Mynah FLEURS-fr 78.70 (47 empty), p50 2,154 ms, never 27.5 % (FT l.209-221) |
| FR-P1 | harness-fr | 27.66 (11.34) | 25.91 | 21.76 (RIF l.614) | - |
| FR-G1 | harness-fr | 29.41 (12.22) | 26.54 | 22.74 (RIF l.737) | - |
| FR-E1 | harness-fr; FR-OOD1 (tedxdev + smtrain, 800 clips) | 25.41 (10.39); OOD1 34.21 | 23.63; 30.14 | 19.84; 23.71 (RIF l.831, l.1028) | - |
| FR-G1E + FR-S2 | harness-fr = DEV-fr; FR-OOD1; FINAL-FR (6 banks, 2,113 clips, spent) | G1E 25.54 / OOD1 29.66; **FR-S2 DEV 25.64, OOD1 29.53; FINAL-FR 27.76 (11.62)**, Mynah 27.90 (RIF l.1120-1123, l.1275) | G1E 23.56 | G1E 19.84 | FR-S2 DEV: FE 7.25 %, R 97.5 %, L 0.72 / 1.20 s. **FINAL-FR: FE 7.57 %, R 96.73 %, L 0.80 / 1.60 s; FINAL MLS FE 25.2 %.** DEV-EOU-FR-v2 internal-class FE 55.2 % (AUT l.137) |
| FR-H2 + s2v4-h2 | DEV-fr; FR-OOD1; mTEDx dev (secondary); FINAL-FR-v2 (4 banks, 1,307 clips, spent) | H2 22.06 / OOD1 25.66 / mTEDx 23.24; **v1.1 DEV 22.08, OOD1 24.97; FINAL-FR-v2 23.00 (10.95)**, Mynah 23.02 (RF22 l.96-99, l.144-145, l.370) | H2 20.51 | H2 16.81 | v1.1 DEV: FE 7.75 %, R 97.12 %, L 0.72 / 1.12 s. **FINAL-FR-v2: FE 12.70 %, R 97.25 %, L 0.80 / 1.36 s (MLS FE 20.8 %, mTEDx 14.0 %).** DEV-EOU-FR-v2 internal FE **68.4 %** vs v4 55.2 % (RF22 l.409-412) |
| FR-v2 stage 2 | DEV-EOU-FR-v2 (mTEDx, 274 clips); DEV-fr; OOD1 | DEV-fr reset 25.48; OOD1 29.62 | - | - | internal FE 59.77 % vs v4 55.17 % -> **FAIL** (AUT l.137-151) |
| S5-FR | DEV-fr; OOD1 (GB10) | S5 arm (archived, UNPAIRED): DEV 25.91, OOD1 30.54 (HUB runs/dgx-s5fr-s5-g1e-11881/probe_harness_post0.5.json, ood1_post0.5.json). **Not classifiable**: baseline arm B still running | S5 arm DEV beam-8 per set only | - | - |
| ES-1 | harness-es (MLS-es test, FLEURS es_419, CV17-es 200, VP-es 164 = 764 clips), sealed 8e9c226f before training. **No FINAL-ES, no OOD-ES** | **18.09 (7.43)**; stock zero-shot 89.27; Mynah bf16 18.09 (RF22 l.485-511) | 16.55 | 12.98 (HUB results/dgx-2026-10-23/es1/summary.txt) | **none: stage 2 not run; no DEV-EOU-ES bank** (S14 l.40-48) |
| ES-S2 s0 / s1 | harness-es | 18.68 / 18.13 (CER +0.14 / -0.28 vs ES-1) (S14 l.172, l.202) | - | - | none |
| ES-S5 s0 / s1 | harness-es | **18.08 / 18.07 (7.15 / 7.09)** (S14 l.217-224) | - | - | none |
| AR-A0 | the 32 training clips (memorisation) | R: CER 32.9 -> 0 from step 250 (MEMORISE); T: CER 616 (FAIL); T3: 100 / 40.7 / 2.1 / 0 / 0 (learnable, slower); T-unfrozen: all empty (FAIL) (R24 l.87-95) | - | - | - |
| AR-A1-R | FLEURS ar_eg dev 250 (primary); CV17-ar test 300 (51 sentences in train) | FLEURS dev **35.86 (10.12)**, ta-fold 35.82 (10.09); CV 37.81 (12.38); stock 107.6 (82.5) (R24 l.107-112) | - | - | - |
| AR-A2-R s0 / s1 | FLEURS dev (OOD for A2) ta-fold primary; cv_decon; masc_c; masc_n; FLEURS test once | ta-fold **33.28 (9.64) / 34.19**; strict 41.61 / 42.38; cv_decon 29.45 / 31.49; masc_c 28.76 / 29.37; masc_n 52.02 / 51.32 (P:AR-SEED1 l.97-105) | - | - | - |
| AR-D2-R s0 / s1 | as AR-A2 | strict 36.44 / 36.35; ta-fold 33.02 / 32.63; cv_decon 31.55 / 33.21 (paired +2.10 / +1.72) -> **HARM REPLICATED** (P:AR-SEED1 l.97-116) | - | - | - |

### 1.D Cost, and what each run proved and did not prove

| id | GPU, training wall (HUB `wall_s`) | audio-h / GPU-h | PROVED (OBSERVED) | did NOT prove |
|---|---|---|---|---|
| IT-0 P40 | L40S, 813 s (+ S2 436 s) | 1,168 | After the 2026-10-09 collapse of the single-stage cold runs (all-blank decoders, FT l.84-117), the two-stage path (plain language stage, then `<EOU>` stage) works with the stock tokenizer | quality off domain (Mynah FLEURS-it 55.1) |
| IT-B2 | L40S, 1,259 s | 535 | Diversity closes the MLS -> FLEURS gap at equal compute. The VP teacher filter halves VP blanking. ±10 dB gain is null (MD l.47-51) | that the encoder budget matters (shown next) |
| IT-E3 | L40S, 1,801 s | 374 | The freeze curriculum was the bottleneck: the whole encoder from step 0 gives -12.56 macro on the same batches as B2 (BN l.161-218). The djlr stage-2 LR cut removes most of the VP blanking (BN l.282-291) | that 4k was enough (E3-8k: no) |
| IT-E3-8k | L40S, 2,970 s | 454 | The horizon is a lever (-5.2 vs E3). On A2 alone it is flat from 7k to 8k (RM12 l.110-124) | the cause of the saturation |
| IT-C-add25 | L40S, 4,903 s | 382 | Additive external audio with A2 exposure held fixed is a strong win on clean-it (oracle -4.18), whereas replacement (E3G-8k) was null (RM12 l.309-333) | a pure data effect: steps and the cosine changed too |
| IT-C250 / oldrep | L40S, 7,311 / 7,169 s | 353 / 360 | Recycling the old 98 h pool at the C250 schedule recovers 97 % of C250's gain over C. **Exposure / schedule dominates; the 130 h of new videos add nothing measurable at that scale** (RM12 l.420-426) | W vs steps (moved together); domain diversity (both pools are YODAS / Granary) |
| IT-C250-36k + v4 | L40S, 18,037 s (5.0 h) + stage 2 875 s | 361 | Schedule scaling 14k -> 20k -> 28k -> 36k keeps paying, then diminishes. v4 gives a usable `<EOU>` at reset-WER +0.5 over the parent under the serving protocol (RM15 l.818-828). FINAL-IT 20.49 misses the pre-registered < 20 band by 0.49 | that 36k is the optimum; that the release parent follows W4 (it does not); seed variance |
| IT-S3-4k | L40S, 1,583 s | 426 | WSD and encoder 4e-4 are additive. 4k W4 beats the 8k cosine (29.96 vs 31.36); 8e-4 is flat and 1.6e-3 overshoots (RM15 l.222-323) | transfer to a larger mix (next row) |
| IT-W4-C250-8k / 16k | L40S 3,714 s / Ada 7,621 s | 389 / 380 | W4 compresses the schedule (16k within 0.49 of 36k in 2.25x fewer steps). It reaches the **same ceiling**, with no representation shift (RM15 l.401-412). Note: the 16k final.nemo = best in-run step 15,500, not EOF | that W4 at a matched horizon beats cosine on C250 |
| FR-0 p40 | L40S, 836 s (+ 447 s) | 1,149 | The two-stage method runs on a second language in ~21 GPU-min | quality: harness-fr 65.85; FLEURS-fr 73-79 |
| FR-P1 | L40S (Japan), 2,366 s | 555 | **LEVER -38.19** vs old p40 on the same harness: recipe + diversity + encoder adaptation explain most of the old gap (RIF l.632-641) | which of the three did it (changed together) |
| FR-G1 | L40S, 2,669 s | 452 | Swapping a third of the A2 draws for Granary-fr at a fixed 8k is HARM (+1.75) (RIF l.752) | whether the cause is dilution or labels (not separable, RIF l.754-764) |
| FR-E1 | RTX 6000 Ada, 3,527 s | 553 | FR-P1 was horizon / exposure limited (-2.25) (RIF l.846-848) | the cause of G1's harm |
| FR-G1E + FR-S2 | Ada, 3,928 s + 859 s | 457 | Mix x horizon interaction (-1.62): at 11,881 steps Granary costs nothing in domain (+0.13) and buys **-4.55 out of domain** (RIF l.940-950, l.1042-1053). v4 ported unchanged: RC-FR candidate, DEV +0.10, OOD1 -0.10 | anything beyond one seed per cell |
| FR-H2 + s2v4-h2 | Ada, 7,773 s + 817 s | 461 | Doubling the horizon: DEV -3.48, OOD1 -4.00, mTEDx -3.91. v4 still passes its gates on the new parent (RF22 l.101-105, l.147-150) | saturation (H3 deferred, RF22 l.389); EOU robustness: internal FE got worse (68.4 vs 55.2 %, a +13 pp HYPOTHESIS link to H2, RF22 l.411-412) |
| FR-v2 stage 2 | L40S, 887 s | 272 | Room-tone pause stretching at p = 0.2 does not reduce internal false endpoints (FAIL) | that hard negatives cannot work (possibly under-exposed at step 250, AUT l.155-157) |
| S5-FR | GB10: S5 arm 8,828 s; B arm running on dgx1 | 203 (S5) | - | (pending) |
| ES-1 | **GB10, 13,326 s (3 h 42 min)** | 157 | A frozen, pre-registered plain recipe (W4, stock init / tokenizer, A2-type 4-source mix, VP teacher filter, the 4.60-pass horizon rule) gives **strong transfer on the first attempt** (18.09, band <= 25), plus a Mynah pack with CPU == CUDA parity 764 / 764 (RF22 l.485-513) | EOU behaviour; a held-out FINAL number; OOD robustness; the value of a Granary-like source in ES; seed variance of ES-1 itself |
| ES-S2 | GB10, 7,867 / 7,849 s (-41 %) | 266 | Bucketing at full horizon: STRONG on both seeds, mean +0.32, seed spread 0.55; the loss sits on CV / FLEURS (S14 l.201-213) | that the CV loss is real (inside the seed spread) |
| ES-S5 | GB10, 9,011 / 8,906 s (-32.4 / -33.2 %) | 233 / 235 | Two seeds give macro 18.08 / 18.07 = ES-1 18.09 at -33 % GPU. **Preferred factory compute recipe** (S14 l.225-231) | statistical equivalence (ES-1 has one seed); the CV decay-recovery hypothesis (did not replicate); EOU on an S5 parent; transfer to other languages (S5-FR pending) |
| AR-A0 | GB10, wall UNKNOWN (minutes) | - | The stock tokenizer cannot write raw Arabic (every word -> `<unk>`). Romanisation keeps the literal recipe and memorises. A fresh text side fails; the NVIDIA Arabic text side is learnable but slower (R24 l.87-95, l.101-105) | generalisation |
| AR-A1-R | GB10, 1,483 s | 251 | Romanisation + stock tokenizer generalise to unseen Arabic sentences (GO-FULL) (R24 l.107-112) | a final Arabic recipe; read speech only, 23 CV speakers |
| AR-A2-R | GB10, 3,545 / 3,341 s (+ the F7 teacher pass, wall UNKNOWN) | 225 / 239 | Large in-domain gains (cv_decon 40.60 -> 29.45, masc_c 49.54 -> 28.76, P:AR-D2 l.326-332). The model learns MASC's ه-for-ة convention (strict - ta-fold gap 8.3). Primary band NULL | that diversity alone helps: data, horizon and the FLEURS-in-training flag changed together (P:AR-A2 §0) |
| AR-D2-R | GB10, 3,463 / 3,350 s | 230 / 238 | Restoring ta marbuta fixes the strict orthography (-5.6 two-seed mean), but costs cv_decon +1.91 two-seed mean (HARM REPLICATED) (P:AR-SEED1 l.109-130) | the mechanism behind the CV cost |

## 2. Genealogy: experiment -> lesson -> recipe change -> next experiment (chronological)

1. **2026-10-09, IT single-stage `<EOU>` from 5 h (cold and warm, frozen and unfrozen)** -> every run collapses to
   blank or to an audio-independent text prior (FT l.84-117) -> two stages: plain language ASR with the stock
   tokenizer first, `<EOU>` second -> **IT-0 P40 + S2**, which works (FT l.141-178).
2. **IT-0** -> good on MLS, poor on FLEURS (55.1) -> **FR-0** replicates the method in ~21 GPU-min, with the same
   failure (MLS 49 vs FLEURS 73-79) (FT l.198-226) -> multi-domain data -> **IT-A0..A2** (MD l.33-51).
3. **IT-A2 / B2** -> diversity closes the domain gap, the VP teacher filter helps, gain augmentation is null, more
   encoder helps (B2 > B1 > A2) -> unfreeze more encoder -> **E1a / E1b, E2, the LR sweep, E3** (BN l.142-218).
4. **E3** -> the freeze curriculum was the bottleneck -> whole encoder 2e-4 from step 0 becomes the recipe. In
   parallel, the stage-2 VP blanking is traced to decoder / joint over-updating -> djlr 2e-5 -> **EOU-v3** becomes
   production (BN l.256-300).
5. **E3G-4k / -8k (Granary replaces half of A2)** -> null at both horizons -> external data must be **added, not
   substituted** -> **C-add25** (RM12 l.126-157, l.309-333).
6. **P2-labels (teacher relabelling of A2)** -> null -> human labels stay, teacher text is used only for new pools
   (RM12 l.159-186).
7. **C-add25** -> strong on clean-it -> more external hours -> **C250** -> gain -> but is it the new videos or the
   exposure? -> **C250-oldrep** -> exposure / schedule dominates -> scale steps, not downloads -> **C250-20k ->
   28k -> 36k** (RM12 l.374-533).
8. **C250-36k** diminishing -> the DGX lab looks for cheaper levers (soups null, blank penalty marginal, CTC 0.1
   null) -> the schedule tournament on A2 -> **WSD + encoder 4e-4 = W4** (RM15 l.187-323).
9. **W4 on C250 at 8k / 16k** -> same ceiling, just faster; the schedule axis is closed -> the data / signal axes ->
   C1 curriculum NULL, D selection NULL, E1 encoder KD NULL (RM15 l.391-851).
10. **Stage 2 v4 on C250-36k** -> offline WER +1.87 -> found to be mostly a no-reset protocol artefact plus premature
    `<EOU>` at internal pauses -> DEV is scored with the serving protocol from then on (RM15 l.787-828) -> v5-B
    (ambiguous positives removed: FE 0.82 % but p50 1.84 s, FAIL), v5-CAL NULL, logit bias NOT-SEPARABLE -> **IT EOU
    research stops; RC-IT = v4** -> FINAL-IT 20.49 (RIF §E-H).
11. **FR gap analysis**: FR is "~5 recipe generations behind IT" (RIF l.86-87) -> **FR-P1** = the IT A2 data design +
    W4 + an FR text pre-map, with FINAL-FR sealed before any FR score -> LEVER -38 (RIF §I).
12. **FR-G1** (Granary at a fixed 8k) HARM -> **FR-E1** (horizon only) CLEAR -> **FR-G1E** (mix at the long
    horizon) NULL in domain -> FR-OOD1 built and sealed -> G1E **OOD BENEFIT -4.55** -> G1E becomes the parent.
    Lesson: the mix effect depends on the horizon, and an in-domain harness cannot see diversity (RIF §J-M).
13. **FR-S2** = v4 unchanged on G1E -> RC-FR candidate -> Mynah parity -> **RC-FR v1 frozen**, FINAL-FR 27.76
    (RIF §N-O).
14. **FR false-endpoint autopsy** -> the model's EOU rule is "about 1 s of silence"; false endpoints are long
    internal pauses (AUT §2) -> **FR-v2** pause-stretch negatives -> FAIL -> stop.
15. **FR-H2** (2x horizon, the only high-prior WER axis left) -> LEVER -> **v1.1** (v4 on H2) passes its gates ->
    FINAL-FR-v2 23.00 -> FR-E0: internal FE 68 %, the cuts sit at punctuation -> truncated-prefix arm closed (RF22
    §4, §11-12).
16. **The H2 exposure lesson becomes a rule**: 995.9 seen / 216.3 physical = 4.60 passes -> ES steps = 4.60 x
    physical h / h_step, capped at 23,762 (RF22 l.261-264) -> **ES-1**, pre-registered before any ES statistic,
    no Granary -> 18.09 strong transfer (RF22 §9, §11).
17. **Cheap-factory question** (fewer GPU-min per language) -> GB10 profile: only 55 % of encoder frames are real
    (padding) (RF22 l.427-431) -> **CF-E1** (1,500 steps): bucketing -42.7 % wall at +0.99; **warm start from FR-H2
    HARM** (+1.87) -> stock init stays (RF22 l.460-477) -> CF-E2 / E3: the seed floor is 0.46 at 1,500 steps (CF23).
18. **S1** (0.6x horizon) FAIL +2.68 -> the horizon stays. **S2** (bucketing at full horizon) STRONG on two seeds
    -> adopted. The residual on CV / FLEURS opens in the decay -> **S5** (unbucketed decay) -> two seeds = ES-1 at
    -33 % GPU -> the preferred compute recipe (S14 l.171-231).
19. **AR plan**: Arabic is unrepresentable in `[a-z' ]` -> **AR-A0** arms R / T / T3 -> R (romanisation) is the
    only arm that keeps the frozen recipe and memorises -> **AR-A1-R** GO-FULL -> **AR-A2** (MASC + teacher CER
    filter, full horizon) NULL on FLEURS but large in-domain gains, and an orthography trigger -> **AR-D2** (ta
    marbuta restored) HARM on cv_decon -> **seed-1 pair**: the strict gain and the CV cost both replicate.
    Note: AR used the S2 sampler, because S5 was only adopted later the same day (R24 l.113-118 vs P:AR-A1 l.18).
20. **S5-FR** (2026-10-25): S5 vs a same-GPU G1E baseline on GB10. The S5 arm is archived; baseline B is running on
    dgx1. Result pending.

## 3. What each language contributed to the recipe (one line each)

- **IT:** two stages; the stock tokenizer with de-accented targets; whole-encoder adaptation; multi-source A2 data
  with a VP teacher filter; additive (not substitutive) external data; horizon scaling; W4; stage 2 v4 with the
  low-LR decoder / joint, the parent's own mix, the selection rule and the serving-protocol DEV metrics. Closed
  levers: KD, CTC, curriculum, static selection, soups, tokenizer swap (confounded), teacher relabelling.
- **FR:** sealed FINAL before any score; per-language text pre-map; the mix x horizon interaction; an OOD dev bank
  as a mandatory second axis; the 4.60-pass horizon; v4 ports unchanged; the internal-pause false-endpoint failure
  characterised (autopsy, DEV-EOU-FR-v2).
- **ES:** the frozen plain recipe transfers when it is pre-registered before the data statistics; then the compute
  recipe (bucketing, S5) is measured with two seeds against that reference.
- **AR:** the factory meets a script it cannot write. Romanisation keeps the literal recipe. Two new constraints
  appear that the Romance factory never met: the normaliser floor (the hamza fold costs 15.6 WER against a minimal
  normaliser, P:AR-A2 §2.3) and target-orthography conventions between sources (ta marbuta).

## 4. Comparability: which numbers live on which bank (do not cross-compare)

| bank | language | clips | used for | comparable with |
|---|---|---|---|---|
| IT in-run 4-set val = DEV730 "harness" (MLS / FLEURS / CV / VP 200 / 200 / 200 / 130) | IT | 730 | all IT plain and stage-2 selection | IT rows only. Pre-10-15 rows are in-run greedy (raw EOF); later rows are `lm_fusion_probe` raw or 0.5 s flush. They differ by ~0.1, e.g. C250-36k in-run 22.37 (HUB) vs probe 22.27; the flush is flat on this bank (RM15 l.165-186, l.365) |
| clean-it (TEDx 909 + ITALIC 475) | IT | 1,384 | development (it was in C-add25's selection mean, RM12 l.320-321) | IT rows from C-add25 on |
| IT reset protocol on DEV730 (FE / R / L / reset-WER) | IT | 730 | EOU selection | IT stage-2 rows only |
| FINAL-IT (6 banks) | IT | 1,826 | one shot of RC-IT v4, spent | nothing else |
| harness-fr = DEV-fr (4 x 200) | FR | 800 | FR selection | FR rows only (old p40 re-scored on it) |
| FR-OOD1 (tedxdev + smtrain) | FR | 800 | the E1 vs G1E decision, then retention only | FR rows from E1 on |
| mTEDx dev (secondary), DEV-EOU-FR-v2 (274) | FR | - | diagnostics; DEV-EOU-FR-v2 used twice for selection | FR EOU rows only |
| FINAL-FR (6 banks) | FR | 2,113 | one shot of RC-FR v1, spent | **not** FINAL-FR-v2 (different banks and model) |
| FINAL-FR-v2 (4 banks; VP only 107 clips) | FR | 1,307 | one shot of RC-FR v1.1, spent | nothing (RF22 l.375-377) |
| harness-es (MLS / FLEURS / CV / VP 200 / 200 / 200 / 164) | ES | 764 | ES-1 primary, **then reused as development for CF-E1-E3, S1, S2 and S5** (RF22 l.457; S14) | ES rows only. CF-E1-E3 numbers are in-run VAL at 1,500 steps, not the probe |
| FLEURS ar_eg dev 250 (ta-fold primary), cv_decon 300, masc_c / masc_n 300, FLEURS test once | AR | 250-354 each | AR decisions | AR rows only. FLEURS dev is in domain for AR-A1 (FLEURS train in its mix) and **out of domain for A2 / D2** |

Do not compare:
- IT 20.49 (FINAL-IT) vs FR 27.76 / 23.00 (FINAL-FR / -v2) vs ES 18.09 (DEV harness, reused) vs AR 33.28 (FLEURS
  ta-fold). They are different banks, normalisers and difficulty. RIF l.1288 records "FR is +7.3 WER behind IT" on
  the FINALs: that is a descriptive gap, not a recipe comparison.
- The EOU numbers of NeMo `asr_eou` (per-event early cutoff, aligned word end) vs our FE (per clip, 0.5 s slack,
  energy endpoint) (S2N §2 l.115-118).
- The wall times of L40S, RTX 6000 Ada and GB10. GB10 is ~3.0x slower than an L40S for this trainer (RM15 l.15-17).
  Ada ≈ L40S per step (RM15 l.416-417).

## 5. Hours vs exposure: every data point we have

Each block is one bank. Rows are comparable only inside a block. "Steps" are optimizer steps of 16 clips. All rows
are one seed unless stated.

**IT, DEV730 greedy (and clean-it oracle-8 where measured)** (HUB, RM12, RM15):

| run | unique h | sources | steps | audio-h seen | passes | greedy | clean oracle-8 |
|---|---|---|---|---|---|---|---|
| E3 (cosine, enc 2e-4) | 119.4 | 4 | 4,000 | 187.2 | 1.57 | 36.52 | - |
| S3-4k (W4) | 119.4 | 4 | 4,000 | 187.2 | 1.57 | 29.96 | - |
| E3-8k | 119.4 | 4 | 8,000 | 374.1 | 3.13 | 31.36 | 25.46 |
| E3G-8k (Granary replaces half of A2) | 217.5 | 5 | 8,000 | 406.3 | 1.87 | 31.71 | - |
| C-add25 | 217.5 | 5 | 10,667 | 520.3 | 2.39 | 29.27 | 21.28 |
| C250-oldrep | 217.5 | 5 | 14,254 | 717.3 | 3.30 | 27.53 | 18.88 |
| C250 | 347.7 | 6 | 14,254 | 716.1 | 2.06 | 27.66 | 18.80 |
| W4-C250-8k | 347.7 | 6 | 8,000 | 401.6 | 1.15 | 26.12 | 17.45 (clean raw) |
| C250-20k | 347.7 | 6 | 20,000 | 1,004.3 | 2.89 | 25.90 | 17.11 |
| W4-C250-16k | 347.7 | 6 | 16,000 | 803.6 | 2.31 | 22.76 | 15.01 |
| C250-28k | 347.7 | 6 | 28,000 | 1,406.6 | 4.05 | 23.60 | 15.72 |
| C250-36k | 347.7 | 6 | 36,000 | 1,808.0 | 5.20 | 22.27 | 15.11 |

**FR, harness-fr greedy 0.5 s / FR-OOD1 greedy 0.5 s** (RIF, RF22, HUB):

| run | unique h | sources | steps | audio-h seen | passes | DEV | OOD1 |
|---|---|---|---|---|---|---|---|
| old p40 (MLS-fr only, top-4) | 38.58 | 1 | 4,000 | 267.0 | 6.9 | 65.85 | - |
| FR-P1 | 127.37 | 4 | 8,000 | 364.7 | 2.86 | 27.66 | - |
| FR-G1 | 216.3 | 5 | 8,000 | 335.4 | 1.55 | 29.41 | - |
| FR-E1 | 127.37 | 4 | 11,881 | 541.4 | 4.25 | 25.41 | 34.21 |
| FR-G1E | 216.3 | 5 | 11,881 | 498.5 | 2.30 | 25.54 | 29.66 |
| FR-H2 | 216.3 | 5 | 23,762 | 995.9 | 4.60 | 22.06 | 25.66 |

**ES, harness-es greedy 0.5 s probe** (S14, HUB):

| run | unique h | steps | audio-h seen | passes | greedy |
|---|---|---|---|---|---|
| S1 (0.6x WSD horizon, rescaled) | 126.48 | 7,602 | 350.7 | 2.77 | 20.77 |
| ES-1 | 126.48 | 12,670 | 582.2 | 4.60 | 18.09 |
| S5 (two seeds) | 126.48 | 12,670 | 582.2 / 581.5 | 4.60 | 18.08 / 18.07 |
| (CF-E1 R, 1,500 steps; in-run VAL, a different metric path) | 126.48 | 1,500 | 69.2 | 0.55 | 30.82 / seed 1 31.28 |

**AR, FLEURS ar_eg dev ta-fold greedy** (R24, P:AR-SEED1):

| run | unique h | FLEURS in training? | steps | audio-h seen | passes | FLEURS dev | cv_decon |
|---|---|---|---|---|---|---|---|
| AR-A1-R | 36.96 | yes (0.30) | 4,000 | 103.5 | 2.80 | 35.82 | 40.60 |
| AR-A2-R seed 0 / 1 | 177.1 | no | 12,670 | 221.6 | 1.25 | 33.28 / 34.19 | 29.45 / 31.49 |

**Observations (OBSERVED unless marked; reported, not concluded):**
1. **More steps on a fixed mix improved every language up to the longest horizon tried.** IT 14k -> 36k: -5.4 on
   DEV730, diminishing at 28k -> 36k. FR 8k -> 11,881 -> 23,762. ES 7,602 -> 12,670: S1 is +2.68 worse than ES-1 at
   the same unique hours. The decay bonus means a mid-run eval does not predict what a shorter run reaches
   (S14 l.42-45).
2. **At a fixed number of steps, the pass count did not predict in-domain WER.**
   - FR-E1 vs FR-G1E: 4.25 vs 2.30 passes, 25.41 vs 25.54.
   - IT C250 vs oldrep: 2.06 vs 3.30 passes, 27.66 vs 27.53.
   - The cost-curve agent says the same: "no passes-based rule holds (same steps, different passes -> same WER)"
     (RF22 l.432-435).
   HYPOTHESIS: in domain, the variable that tracks WER is the number of updates (draws) under the full WSD /
   cosine horizon, not the number of passes.
3. **Physical diversity paid off only on a bank that could see it, and only when it added a new domain.**
   - FR: in domain, Granary-fr was HARM at 8k and NULL at 11,881. Out of domain it was -4.55 (FR-OOD1).
   - IT: 130 h of new videos from the same source (Granary) added nothing on clean-it at a fixed schedule (oldrep
     f = 0.97). Additive external data at fixed A2 exposure was a big win (C-add25: clean oracle -4.18), but its
     steps changed too.
   - AR: A2's 140 extra MASC hours gave large in-domain gains (cv_decon -11, masc_c -21 vs AR-A1). On FLEURS (OOD
     for A2) the gain was small (-2.5 / -1.6 over two seeds), and FLEURS had left the training mix.
4. **The "4.60 passes" horizon rule** (RF22 l.261-264) comes from one FR run (H2). In ES it reproduced the same
   pass count by construction. Observations 1-2 say the steps (12,670) and not the passes may be what mattered.
   COMPUTED: at 4.60 passes a 347.7 h pool would need ~31.9k steps at IT's 0.0502 h / step, i.e. above the
   23,762-step cap. The cap and the pass rule disagree for pools above ~216 h. **UNKNOWN** which one predicts WER.
5. **Unique hours alone did not order the results.** FR-G1 (216 h) < FR-P1 (127 h) at 8k; FR-G1E ≈ FR-E1 at 11,881.
   ES-1 (126.5 h, 4 sources, no web source) reached the lowest absolute WER of any language, on its own bank: that
   is not comparable (section 4).
6. **Sources:** every Romance run with a good outcome had at least 4 human-labelled sources (read audiobooks,
   crowd-read, parliament, FLEURS). FR and IT then added a teacher-labelled web source. ES had none. The value of
   that fifth source in ES is UNKNOWN.

## 6. Biggest confounds (carried into the factory matrix)

- **IT's release parent is not on the factory recipe.** It used cosine with encoder 2e-4 for 36k steps. W4 was
  only tested on C250 at 8k / 16k (16k PARTIAL, 22.76 vs 22.27).
- **One seed** for almost every run. Known seed spreads: ES 1,500-step 0.46 (CF23 l.35-40); ES full-horizon S2
  0.55, S5 0.01 (S14); AR cv_decon ~2.0 in both arms (P:AR-SEED1 l.120-122).
- **Mixed hardware:** L40S, RTX 6000 Ada and GB10 (GB10 ~3x slower). FR's same-GPU S5 test is still running.
- **Granary threshold:** IT agree <= 15 vs FR agree <= 20. The FR threshold was set by a model-free coverage rule
  (RIF l.658-662).
- **ES harness reuse:** harness-es was used to select the compute recipe (S1-S5). There is no FINAL-ES or OOD-ES.
- **AR moves several variables at once** (A1 -> A2: data, horizon and FLEURS-in-training). It also uses the S2 sampler
  and max-dur 20, and its normaliser floor and orthography conventions do not exist in Romance.
- **EOU only exists for IT and FR**, and only with v4 on cosine / WSD parents trained with random sampling. No
  stage 2 ran on an S5 parent or on ES / AR.

## 7. UNKNOWNs that matter

- ES stage 2 v4: EOU quality, latency and reset-WER cost. There is no DEV-EOU-ES bank.
- A held-out FINAL-ES and an OOD-ES bank.
- ES with a Granary-like fifth source: the G1E analogue was never run.
- S5 on FR (pending) and on IT (never run). Stage 2 on an S5 parent.
- Whether the horizon rule should be expressed in passes or in steps for pools > 216 h. FR H3 was deferred.
- Seed spread at full horizon for IT and FR.
- C250 physical hours (347.7 computed vs ~357 quoted). IT FLEURS speakers. Kept-pool video counts for the Granary
  and MASC subsets.
- The GB10 cost of an IT-scale run (only an estimate: ~14-15 h for 36k, RM15 l.15-17). The AR F7 teacher-pass wall.
- Whether the NeMo-style EOU differences (padding p >= 0.9 with >= 3 s, noise, FastEmit) would change the
  internal-pause false endpoints (S2N §3). Untested.
