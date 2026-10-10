# EOU 120M Italian specialist, day 2 (2026-10-10): multi-domain data, encoder plasticity, teacher ceiling

Box: one Vast L40S (Japan), provisioned from scratch with `finetune/jobs/prov.sh` in ~4 min.
Tooling: `finetune/` on branch `lw-finetune-tooling` (`eou/two_stage.py`, `eou/prepare_mix.py`,
`eou/diag/*`, `jobs/mix_a.sh`, `jobs/mix_arm.sh`, `jobs/a2_vpfilter.sh`, `jobs/s2_mix.sh`,
`jobs/it_ceiling.sh`, `jobs/vp_mynah.sh`, `jobs/yodas_audit.sh`). Archive: private HF repo
`gabrione/mynah-asr-canary-it-experiments`, `runs/plain-it-*`, `data/it/`.

## Protocol

Four val sets, scored identically for every candidate (accent-insensitive a-z normaliser, pooled
WER/CER, per-utterance greedy, the `rs[::len//n][:n]` subsample): MLS-it test (200, = the
2026-10-09 val), FLEURS-it test (200), Common Voice 17 it test (200), VoxPopuli it test (130).
Selection on the macro mean; every domain reported. One variable per arm. Gates before GPU runs:
data stats (`mix_stats.py`), regression A/B of the refactored `two_stage.py` against the
2026-10-09 file (identical loss trace and trainable params), yesterday's checkpoints re-scored
(`--eval-only`: P40 reproduces MLS 41.72), 100-step smoke.

## Data (stage 1 mix, ~125 h)

| source | utts | h | speakers | weight |
|---|---|---|---|---|
| MLS-it 40 h cut | 9,683 | 40.0 | 65 | 0.35 |
| Common Voice 17 it (fixie-ai mirror, <=20/speaker) | 26,800 | 40.0 | 1,358 | 0.30 |
| VoxPopuli it (<=60/speaker) | 13,575 (A2+: 10,955) | 38.3 (32.6) | 141 | 0.20 |
| FLEURS it train | 2,375 | 6.8 | - | 0.15 |

VoxPopuli label noise: an independent teacher (Nemotron 0.6B) disagrees (>50 % WER) with 17 % of
VP train vs 0.7 % of CV train (300 random clips each); examples include English audio under an
Italian reference. A2+ keep VP clips with teacher WER <= 50 % (per-utterance scores kept:
`data/it/train_vp.teacher.json`).

## Stage 1 (plain ASR), WER / CER / empty

| run | variable | MLS | FLEURS | CV | VP | macro |
|---|---|---|---|---|---|---|
| P40 (2026-10-09) | MLS 40 h only, top-4 | 41.72 / 12.25 / 0 | 55.82 / 18.18 / 0 | 64.92 / 27.17 / 9 | 66.62 / 33.03 / 1 | 57.27 / 22.66 |
| A0 @3000 | multi-domain mix | 45.94 / 14.65 / 2 | 46.25 / 15.14 / 0 | 57.88 / 23.30 / 10 | 61.35 / 39.64 / 17 | 52.85 / 23.18 |
| A1 @3500 | + gain +-10 dB | 45.97 / 14.41 / 2 | 46.16 / 15.03 / 0 | 57.30 / 23.14 / 9 | 62.54 / 41.15 / 19 | 52.99 / 23.43 |
| A2 @4000 | VP teacher-filtered | 45.76 / 14.33 / 2 | 45.39 / 15.02 / 0 | 56.93 / 22.42 / 9 | 59.11 / 34.84 / 9 | 51.80 / 21.65 |
| B1 @3500 | A2 + top-8 | 44.94 / 12.80 / 0 | 44.22 / 14.56 / 0 | 55.78 / 20.85 / 8 | 59.34 / 35.91 / 18 | 51.07 / 21.03 |
| **B2 @4000** | A2 + gradual unfreeze to the whole encoder | **43.84 / 11.93 / 0** | **43.86 / 14.10 / 0** | **53.78 / 18.69 / 6** | **54.86 / 28.14 / 3** | **49.08 / 18.22** |
| Nemotron 3.5 0.6B (teacher) | - | 18.77 / 4.94 / 0 | 6.28 / 3.52 / 0 | 9.40 / 2.61 / 0 | 28.53 / 22.83 / 7 | 15.75 / 8.47 |

Readings: data diversity closes the MLS->FLEURS gap (14 -> 0.3 points) at equal compute; gain
+-10 dB is null (the quiet CV failures sit 15-35 dB below typical speech); teacher-filtering VP
halves the VP blanking; the whole encoder (B2) moves every domain and the CER most. Paired per
utterance, Nemotron beats A2 on 684/730 clips, uniformly over domain, duration and level: the
Italian gap is a training deficit (the English stock gap on our runtime is ~1 WER), not the eval.
A1 audit: CV failures = level (also with P40); VP failures = A0-only regressions on audio the
teacher reads (11/22 at <= 30 % teacher WER).

## Stage 2 (<EOU>) and Mynah

| | S2 (2026-10-09) | S2 on B2, top-4 recipe unchanged (= **EOU-IT v2**) |
|---|---|---|
| NeMo 4-set macro WER / CER | 56.51 / 23.34 | 49.07 / 21.10 |
| Mynah FLEURS-it WER / CER / empty | 55.1 / 19.0 / 0 | **42.0 / 15.1 / 0** |
| clips with model EOU | 175/200 | 188/200 |
| speech end -> EOU p50 / p95 | 2346 ms / - | 1546 ms / 3370 ms |
| EOU never / premature | 10 % / 0 % | 5 % / 0 % |
| A + 1 s + B: EOU in gap, lat p50 | 58 %, 330 ms | 75 %, 394 ms |
| CPU f32 == CUDA (incl. EOU events) | PASS | PASS (gates A/B/C) |
| CUDA server VRAM / start-up | 1.9 GB / 500 ms | 1.94 GB / 593 ms |

Open: in the NeMo offline eval stage 2 regresses VP (54.86/28.14/3 -> 59.11/38.62/22). Not the
eval tail (B2 with the same 2 s tail is unchanged). Candidates: the top-4 stage-2 policy (a
whole-encoder stage 2 is the running A/B) and premature <EOU> + no decoder reset in the offline
eval (NVIDIA's voice agent resets after every EOU; Mynah does; `jobs/vp_mynah.sh` measures VP
in Mynah streaming).

## Research pass (two read-only agents, 2026-10-10)

NVIDIA recipe vs ours (NVIDIA-NeMo/Speech main e4a34fd, v3.0.0 fd6a877, the released model cfg):
- fine-tune LR: NVIDIA default 1e-4 cosine on the whole model; the released EOU 120M peaked at
  Noam 1e-4 (warmup 2500, bs 32, FastEmit 0.03). Ours: encoder 3e-5..5e-6 (A/B candidate, not a
  prescription: check which regime the "50-100 epochs" guidance refers to).
- EOU data: NVIDIA pads silence BEFORE (~4 s mean) and after (>= 3 s) with p 0.99, white noise
  -90..-46 dB, real noise SNR 0-20 over the whole clip; ours: 50 % trailing 1-3 s only, no noise.
  Multi-utterance targets (A<EOU> B<EOU>) exist in their dataset class; we never train text after
  an <EOU>. EOU only on sentence-complete clips (Granary chunks belong in the plain-ASR group).
- our `norm()` deletes digits (targets and references); NVIDIA verbalises numbers.
- checkpoint averaging (keep only if it beats the best single) is in NVIDIA's fine-tune guide.
- README facts to fix: 3 (evidence = released cfg target/nemo_version), 2 (check_tokenizer
  flag), 7 (invalid augmentations are skipped, not refused), 10 (0.1/0.9 is source weighting,
  both groups get <EOU> and padding), 15 (release trained at Noam peak 1e-4), 6/16 (effective
  padding distribution, eval-set defaults 0.2 s pre + 3 s post + noise).

Community / data:
- No public non-English fine-tune of parakeet_realtime_eou_120m-v1 and no non-English
  integrated-EOU streaming ASR (NVIDIA promised multilingual in HF discussion #2; not released).
- `espnet/yodas-granary` Italian: ~6.1k h of YODAS already segmented with Whisper-v3 labels
  (Granary filters: LID, hallucination n-grams, char rate; no confidence / agreement filter).
  Shards `data/it000|it100|it101/ast/*.parquet` (~530 MB, ~5 h each). Two-teacher agreement
  (Nemotron vs Granary-Whisper) is our filter candidate (uDistil: agreement is the best
  label-free filter; StreamHear: with a strong teacher, no filter can win -> test both).
- Scale references: 120M FastConformer on 2.1k h Granary-hr -> FLEURS-hr ~17-22; community rule
  ~1k h for ~20 % WER. 120 h is an order of magnitude short.
- Tokenizer: every public cross-lingual derivative retrains/extends it; warm-start recipe
  (copy shared rows, move blank) in NVIDIA-NeMo/Speech#15793. Not dead for us: the 2026-10-09
  failure confounded it with lr 1e-3, 90 % padding and FastEmit.

## Granary-it pseudo-labels (C1 / C2)

Pool: `espnet/yodas-granary` it000 `ast`, 70 shards spread over the language, clips <= 30 s:
46,280 clips, 152.2 h, 4,068 videos; Nemotron on every clip (2035 s). Two-teacher agreement
(Nemotron vs Granary-Whisper) WER p50 9.1, p90 37.5; hours at <= 10/15/30 %: 88/111/139.
99.8 % Italian by the teacher's LID; 3,970 clips (8.6 %) dropped because the teacher writes
digits (to verbalise next time). Short clips agree less (p50 12.5 below 6.6 s vs ~8 above).
Subsets (`eou/pseudo_subsets.py`, nested, label = teacher text): agree<=15 98.1 h / 3,704
videos; random hours-matched control 98.1 h / 3,832 videos. 50/50 with A2 adds 5-9 % audio/step.

Both on B2's policy, 4000 steps, 50 % A2 (internal proportions kept) / 50 % Granary, --max-dur 30
(peak 36.6 GB). C1 = hygiene + agreement <= 15 %; C2 = hygiene only, random, hours-matched.

| | MLS | FLEURS | CV | VP | macro |
|---|---|---|---|---|---|
| B2 | 43.84 / 11.93 / 0 | 43.86 / 14.10 / 0 | 53.78 / 18.69 / 6 | 54.86 / 28.14 / 3 | 49.08 / 18.22 |
| C1 | 46.78 / 12.81 / 0 | 42.76 / 13.55 / 0 | 53.83 / 19.10 / 6 | 53.84 / 26.69 / 0 | 49.30 / 18.04 |
| C2 | 47.26 / 13.21 / 0 | 43.60 / 13.81 / 0 | 55.15 / 19.60 / 7 | 53.28 / 27.09 / 1 | 49.82 / 18.43 |

Readings (one seed): agreement filtering gives a small, consistent edge (C1 ahead of C2 at every
eval and on 3/4 domains, macro -0.5 WER / -0.4 CER); at fixed compute neither moves the macro
frontier vs B2: Granary shifts the model toward FLEURS/VP and away from MLS (the MLS gap is
~2.8 from step 1000 on: reweighting, not progressive forgetting). The student's WER/CER ratio
(~3.1 on FLEURS vs Nemotron's ~1.8) points at near-miss words (morphology, elision,
segmentation): an INDICATION toward the output representation (English SPE, de-accented
targets) or the predictor, not proof -- the error analysis decides.

## Archive (end of day)

Private HF repo pruned + squashed (`common/hf_prune.py`): 28.6 -> 13.8 GB, then the day's runs:
16.1 GB, 359 files, 15 `.nemo`, 7 `last.ckpt` (resume points: B2, EOU-IT v2 (`plain-it-s2-b2`),
C1, C2, EOU-FR). Survival bundle `results/survival-2026-10-10.tgz` (40.6 MB, verified; also on
the dev machine): logs, manifests (no audio), metrics, audits, ceiling per-utterance hypotheses,
Mynah m1 outputs, teacher scores (`data/it/train_vp.teacher.json`, `granary_yg_pool.teacher.json`).

Resume tomorrow: `finetune/jobs/prov.sh` (KEEP_TOKEN=1 inside the tmux command), data via
`jobs/data_mix.sh it` + `jobs/a2_vpfilter.sh` (scores on HF: no teacher re-run needed),
Granary via `jobs/granary_build.sh` (or the archived pool scores + `prepare_mix --yg-spread`),
then `jobs/mix_arm.sh` / `jobs/granary_ab.sh` / `jobs/s2_mix.sh`; checkpoints from HF `runs/*`.

## Next (ranked)

1. Error analysis, no GPU (bundle: ceiling per-utterance hyps of P40/A0/A2/Nemotron): S/D/I,
   top confusions, share of words within 1-2 characters of the reference, English-SPE tokens
   per Italian word. Then the discriminating pair: C1 longer (4k -> 8k -> 12k steps) vs an
   Italian tokenizer with warm-start (NVIDIA-NeMo/Speech#15793: shared rows copied, blank
   moved) in a clean plain stage 1. Only if the long curve keeps falling: scale Granary.
2. Stage 2 v3 ablations, one at a time: pre+post padding at p 0.99, noise, multi-utterance
   targets, whole-encoder policy (running), FastEmit 0.005/0.03.
3. Whole-model 1e-4 A/B; longer runs once data is larger; checkpoint averaging.
4. Number verbalisation (Italian TN) in targets and references.
5. Italian tokenizer with warm-start, after data scaling.
