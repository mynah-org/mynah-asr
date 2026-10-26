# ES Stage 2 qualification: ES Factory-v1 Stage 1 -> frozen Stage 2 v4 -> quality + EOU preservation (plan, 2026-10-26)

Status: **PLAN only, CPU-only, nothing launched, nothing committed.** Today the boxes were touched read-only (nice 19, no
GPU, `~/gpu.lock` untouched, no token read): a sha256 of the ES checkpoints, their `metrics.json` args, the manifests,
and the public Hub file listings. No sealed set (FINAL-IT, FINAL-FR, FINAL-FR-v2, FINAL-EOU-FR-v2) was read or decoded.

Tags: **OBSERVED** (artifact named), **HYPOTHESIS**, **UNKNOWN**. Source keys follow `.work/eou-specialist-evolution.md`
section 0 (RIF = release-it-fr-2026-10-19.md, RF22 = release-final-2026-10-22.md, AUT = fr-v2-eou-autopsy-2026-10-21.md,
S2N = stage2-v4-vs-nemo-asr-eou-2026-10-24.md, S14 = cheapft-S1-S4-prereg-2026-10-23.md, SUB = sub15-error-budget-plan.md,
FVM = factory-validation-matrix.md).

## 0. Question and scope

Does `ES Factory-v1 Stage 1 (S5) -> Stage 2 v4 exactly as frozen for IT/FR` keep ES quality and give a usable `<EOU>`?
- This is a **qualification of an existing recipe on a new cell**: a new language (ES) and a new parent type (S5 sampler).
  Both are untested today (FVM: "Stage 2 v4 on the S5 parent is untested"; ES EOU evidence "none").
- It is **not** an EOU research arm: no v5, no padding / noise / FastEmit change, no threshold, no window.
- It is **not** a release: there is no FINAL-ES, so a PASS cannot become an RC-ES. It earns "ES EOU qualified on DEV".
- It must not contaminate SUB15 (the ES greedy < 15 line through Stage 1 data / recipe changes, run on its own lineage).

## 1. Parent: ES-S5 seed 0

| candidate | sha256 (OBSERVED) | location | harness-es greedy 0.5 s | recipe |
|---|---|---|---|---|
| **ES-S5 s0 (chosen)** | **583d6954a285d85194029e7fd93c9957b92b9a8b9058ca81b47e59b7a74683c2** (dgx1, today) | dgx1 `~/ft/runs/plain-cfs-s5-es/final.nemo` (460,072,960 B); Hub `runs/dgx-cfs-s5-es` | 18.08 (MLS 14.54 / FLEURS 17.04 / CV 21.98 / VP 18.77) (SUB §0) | W4, WSD, 12,670 steps, bucket-pool 32 + decay-mode none, seed 0, lang es, max-dur 30, the ES-1 4-source mix (OBSERVED, metrics.json args) |
| ES-S5 s1 | df7ae15e4193321449defd149695b20bfe474601b727a82f20f84500535d8919 (dgx1, today) | dgx1 `~/ft/runs/plain-cfs-s5-s1-es/` | 18.07 | as s0, seed 1 |
| ES-1 | 172b1539 (SUB §0) | dgx2 `~/ft/runs/plain-es1-w4-12670/` | 18.09 | random sampler; the SUB15 reference |

Why S5 s0:
1. **Factory v1 uses the S5 sampler** (FVM reference table). The S5 evidence: ES two seeds -33 % GPU at equal macro (S14),
   and the S5-FR result, STRONG at -40 % (restart-2026-10-25). So an S5 parent is the natural Factory-v1 Stage 1. Stage 2
   on an S5 parent is exactly the cell the EOU contract does not cover yet. Qualifying ES-1 instead would cover only the
   language, and leave the parent-type gap open.
2. **Seed 0 is chosen by rule, not by score.** It matches every frozen stage-2 run (all seed 0). s0 and s1 differ by 0.01
   macro, which is far inside the noise (S5 per-set seed spread up to 1.05, SUB §2.1).
3. **ES-1 stays untouched.** All of SUB15's per-utterance artifacts and its 18.09 reference live on ES-1 (SUB §1). Not
   branching it keeps the SUB15 reference lineage free of any stage-2 descendant.

## 2. Stage 2 recipe: v4, frozen (no ES tuning)

Command, read back from the archived v4 `metrics.json` (RIF l.1063-1067; FR-S2 pre-registration RIF §N). Launch through
`s2_mix.sh`, as FR-S2 did:

```
LANG_=es INIT=$HOME/ft/runs/plain-cfs-s5-es/final.nemo TAG=es-s2v4-s5s0 \
  S2MIX="$MF/train_40h_es.json:0.35,$MX/train_cv.json:0.30,$MX/train_vp_t50.json:0.20,$MX/train_fleurs.json:0.15" \
  S2EXTRA="--lr 2e-5" PROBE=1 M1=0 bash finetune/jobs/s2_mix.sh
```

This resolves to:

`two_stage.py --freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5 --lr 2e-5 --bs 16 --steps 1500 --lr-shape cosine
--eou-append 1 --pad-prob 0.5 --pad-min 1 --pad-max 3 --pad-noise-db -90,-60 --fastemit 0 --max-dur 20
--eval-every 250 --val-n 200 --seed 0 --lang es`

Checked against the code (OBSERVED: `s2_mix.sh` and `two_stage.py` @ trainB ae3bcd8):
- `S2EXTRA` overrides the script's `--lr 1e-4` (the last flag wins). That is how IT v4 and FR got the djlr 2e-5.
- The cosine shape, `--pad-noise-db`, `--max-dur 20`, `--val-n 200` and `--seed 0` are the `two_stage.py` defaults.
- No `--bucket-pool`. Stage 2 keeps v4's random sampler: S5 is a Stage-1 compute recipe and stays out of Stage 2.
- The throughput flags (`--bucket-pool`, `--enc-subbatch`, `--joint-subbatch`, `--prefetch`) are 0 by default.

Recipe details:
- **Data: the parent's own mix** (the v4 principle). These are exactly the S5 s0 manifests and weights (OBSERVED,
  metrics.json args). Every ES clip is <= 20 s (FVM: "data <= 20 s"), so `--max-dur 20` drops nothing (HYPOTHESIS; the
  preflight counts it).
- **EOU targets:** `textnorm es` ids + `[<EOU>]` on 100 % of draws (stock SPE id 1024). Silence: p 0.5 trailing
  U(1, 3) s at -90..-60 dB. No aligner, no timing labels, nothing after `<EOU>`, `<EOB>` never used (S2N §1).
  **Nothing ES-specific has to be built for training.** The manifests exist on both DGX (OBSERVED: dgx1 and dgx2
  `~/ft/manifests/*_es.json`, `~/ft/mix/es/`). `--lang es` already runs through `two_stage.py`, `s2_reset.py` and
  `lm_fusion_probe` (textnorm `PRE` has `es`; ES-1 used it).
- **Selection: the v4 rule.** final.nemo = the lowest in-run harness-es macro WER among the evals with EOU >= 80 % and
  empty <= 10 %, at steps 250 / 500 / ... / 1500. IT and FR v1 picked step 250, FR v1.1 picked step 1250.
- **Code:** trainB `ae3bcd8` or a later commit whose diff in `two_stage.py` / `s2_mix.sh` / `s2_reset.py` is only this
  plan's ops amendment. The tree must be clean.

Ops amendments (no recipe change), part of the pre-registration commit:
- **A1.** `s2_mix.sh` hard-codes `timeout 3600` on the training. The GB10 estimate (section 4) is 25-50 min, which leaves
  little margin. Add `TIMEOUT_S2=${TIMEOUT_S2:-3600}`, keeping the default, and launch with 7200.
- **A2.** Print the selected step and the per-eval EOU / empty values in the log. They are already in metrics.json, so
  this is for reading only.

## 3. Evaluation (all deltas = stage 2 vs the exact parent, same tool, same box)

**E-A. Primary bank: harness-es** (764 clips, seal 8e9c226f, the ES-1 primary; OBSERVED SUB §0). There is no other ES DEV
bank. Run `s2_reset.py --lang es --post-pad-s 2.0 --per-set 200 --parent <S5 s0> --s2 <selected> --dump` with all three
protocols. This is the frozen RC-IT / RC-FR serving protocol: frame-synchronous greedy, max 10 symbols, decoder reset
after `<EOU>`, 2 s of -75 dB quiet, energy endpoint of the original clip.
- Reported: reset-WER / CER per set and macro, S / D / I, empties, FE, FE_any, recall, latency p50 / p90, end deletions.
- Diagnostic: `PROBE=1` adds `lm_fusion_probe` greedy / beam-8 / oracle-8 at raw and 0.5 s, on stage 2 (`<EOU>` dropped,
  no reset) and on the parent.
- **Validity check:** the parent probe must reproduce greedy 0.5 s **18.08 ± 0.02** on the executing box. If not: STOP
  (environment / build). This is the "suspect the environment first" rule.

**E-B. Secondary bank: DEV-EOU-ES** (new, model-free, built and hashed before any decode). It replaces FR's OOD1 retention
gate and adds the internal-pause characterisation. ES has no OOD bank and no long-form natural-ending bank today.
- **Source.** mTEDx-es on the Hub mirror the FR banks used (`deepdml/mtedx/es`): 54 train shards of ~0.36-0.52 GB, plus
  `test` (245 MB) and `valid` (223 MB) (OBSERVED, Hub API listing today from dgx2). mTEDx is in no ES training pool and no
  ES bank.
- **MLS-es cannot be used.** The 40 h cut covers all 86 MLS-es train speakers (OBSERVED: `train_40h_es.json` has 86
  speakers), so no speaker-disjoint MLS train audio exists. This is FR's situation (AUT seal record: 142 / 142). Under
  FR's rule the bank is mTEDx-only, and the single-domain limitation is declared. MLS-es `dev` (157 MB, OBSERVED listing)
  is speaker-disjoint and unused. It is **reserved for a future FINAL-ES**, so it is not used here.
- **Split rule (mirrors FR, frozen now):**
  - mTEDx-es `test` + `valid` talks -> reserved for a future OOD-ES.
  - Train shards 0-1 -> reserved for a future FINAL-ES.
  - Train shards 2-9 -> the EOU banks, split by `sha1(talk_id)` parity: DEV-EOU-ES (decoded here) and FINAL-EOU-ES (built
    in the same pass, sealed SEAL.sha256, tarred, local audio removed, never decoded in this plan).
  - Exclusions: the texts of every ES train and eval manifest, digits / markup, fewer than 3 words.
  - Classes: the frozen energy rule of `prepare_eou_v2.py`. Internal strata 0.8-1.04 / 1.04-1.6 / 1.6-2.5 / >2.5 s, with
    up to 50 clips each. Endpoint: up to 100 clips, natural ending `. ? !` or `...`. At most 4 clips per talk and class.
  - Target size: ~270 clips (FR DEV 274).
- **Status: DEV, not sealed for selection.** It is decoded once, on the parent and on the single selected checkpoint,
  after selection. It is never used to choose between checkpoints. It is talk-disjoint, which is not speaker-disjoint
  (FR's declared caveat).
- **Decode and score.** `s2_reset.py --lang es --post-pad-s 2.0 --models parent,s2_reset`, scored with `eou_v2_score.py`:
  - all-clip reset-WER delta (gate R1);
  - internal-class FE per stratum and endpoint recall / p50 / p90 (reported only, see R2).
- **Port needed (CPU, code).** `prepare_eou_v2.py` hard-codes `a.lang = "fr"`, `french`, `fr/` and a Granary-ids
  exclusion. Generalise it with `--lang`, an `ENDING["es"]` regex, `--granary-ids` optional (ES has no web source) and
  `SRC["es"]` in `prepare_final_it.py`. FR output must stay byte-identical (rebuild the FR DEV manifest and compare it to
  sha 851084f2 on a CPU box that holds the FR cache, or diff the code path only).

**Not in scope:**
- Mynah export / streaming parity. That comes after a PASS, as its own step: CPU CLI on harness-es; IT took 845 s for 730
  clips (RF22).
- Any FINAL.
- A stage-2 seed 1. It is allowed only as the pre-registered follow-up of a MARGINAL result.

## 4. Cost (GB10)

| item | where | estimate | basis |
|---|---|---|---|
| Stage 2 training, 1500 steps, 6 in-run evals | GPU, the non-SUB15 DGX | **25-50 min** | HYPOTHESIS. Same command 816-887 s on Ada / L40S (RIF, RF22, AUT; OBSERVED), and GB10 is ~3x slower for this trainer (RM15 l.15-17), giving ~41-44 min. 13 of 17 encoder blocks are frozen, so it could be lower. No GB10 stage-2 wall exists (UNKNOWN) |
| PROBE=1 (2 models x 2 flushes, greedy + beam-8) | GPU, same box | ~5 min | OBSERVED: harness-es greedy 7.0 s and beam-8 13.6 s on GB10 (SUB §3), plus model loads |
| **Total GPU** | | **~0.5-1.0 GB10-h** | |
| s2_reset on harness-es, 3 protocols x 764 | CPU, same box after training | 45-70 min | HYPOTHESIS. FR DEV (800 clips) took 52 min CPU (RIF §N result) |
| port of `prepare_eou_v2.py` + review | Mac / agent | 1-2 h of work | HYPOTHESIS |
| mTEDx-es download (shards 0-9 talk ids, shards 2-9 audio, test / valid ids) | dgx2 CPU / network | ~4.5-5 GB, ~10-30 min | COMPUTED from today's listing |
| bank build + seal (20 workers) | CPU | 30-60 min | HYPOTHESIS |
| DEV-EOU-ES decode, 2 protocols x ~270 | CPU | 15-25 min | HYPOTHESIS, scaled from the FR 274-clip bank |
| **Total CPU** | | **~2.5-4 h wall; it overlaps the GPU job** | |

## 5. Pre-registered gates (one seed; deltas = s2_reset(stage 2) - s2_reset-tool parent, macro over the 4 sets)

The thresholds are the RC-FR candidate / FAIL bars, unchanged (RIF §N, which equals the RC-IT acceptance level). FR's
DIAGNOSE becomes MARGINAL. Two lines are new and marked: Q3, the SLA line, and R1, the OOD1 replacement.

| id | metric (bank) | PASS | MARGINAL | FAIL |
|---|---|---|---|---|
| V | validity: parent probe 18.08 ± 0.02; harness seal 8e9c226f; v4 rule satisfied at the selected step; clean tree | all hold | - | any fails -> **INVALID** (no class, fix the environment, rerun) |
| Q1 | reset-WER macro delta (harness-es) | <= +1.0 | +1.0 < d <= +2.0 | > +2.0 |
| Q2 | worst single-source delta (harness-es) | <= +2.0 | > +2.0 | - |
| Q3 (new) | SLA-A: stage 2 reset-WER macro (harness-es) | < 20.0 | - | >= 20.0 |
| Q4 | empties (harness-es) | <= parent + 1 % abs (<= parent + 7 clips) | above that | - |
| E1 | EOU recall, pooled (harness-es) | >= 94 % | 90-94 % | < 90 % |
| E2 | FE (`<EOU>` with >= 0.5 s of speech left), pooled | <= 10 % | 10-15 % | > 15 % |
| E3 | endpoint latency p50, pooled (p90 reported) | <= 1.0 s | 1.0-1.5 s | > 1.5 s |
| R1 (new) | all-clip reset-WER delta (DEV-EOU-ES) | <= +1.0 | +1.0 < d <= +2.0 | > +2.0 |
| R2 | internal-class FE per stratum; endpoint recall / p50 (DEV-EOU-ES) | report only | - | - |

**Class:**
- **PASS** = every row is PASS.
- **FAIL** = any row is FAIL.
- **MARGINAL** = otherwise.

**Decisions:**
- **PASS:** "Stage 2 v4 qualified on ES and on an S5 parent (DEV, one seed)". The FVM EOU contract extends to 3 languages
  and 4 parents. The next steps are the Mynah ES parity, and FINAL-ES + OOD-ES built and sealed before any RC-ES.
- **MARGINAL:** no promotion. The only allowed follow-up is a stage-2 seed 1 on the same parent, pre-registered separately.
  No flag change.
- **FAIL:** STOP. The EOU contract is not "language- and parent-agnostic". Root-cause it on CPU from the dumps before
  touching the recipe.

**Not a gate, not comparable:** ES FE / recall vs IT or FR numbers (different banks); R2 vs FR's 55.2 % internal FE.

**Priors (HYPOTHESIS):**
- FR deltas were +0.10 and +0.01; IT was +0.27 reset vs plain. So Q1 PASS is likely.
- The per-source FE risk sits on MLS-es: FR MLS FE was 18.0 % on both parents, IT FLEURS 17 %. A pooled E2 between 7 and
  10 % is the expected range.

## 6. Contamination guard vs SUB15

1. **Separate lineage and directories.**
   - TAG `es-s2v4-s5s0`, run dir `runs/plain-es-s2v4-s5s0`.
   - Hub `runs/dgx-es-s2v4-s5s0` and `results/dgx-<date>/es-s2v4`.
   - SUB15 tags must never start with `es-s2`.
2. **Never a SUB15 input.** The stage-2 model, its dumps and DEV-EOU-ES are never a SUB15 init, teacher, soup ingredient,
   distillation target or comparator. SUB15's reference stays ES-1 / S5 plain greedy 0.5 s.
3. **SUB15 lineage stays plain.** If a SUB15 winner later needs EOU, it gets its **own** stage 2 v4 run. It is compared to
   this run's numbers only as an EOU reference, never as a WER reference.
4. **No harness tuning.**
   - harness-es seal and textnorm `es` stay frozen; the s2_reset run checks the seal.
   - SUB15's normalisation lenses (digits, homophones, SUB §2.7) stay report-only and are never applied here.
   - No new ES eval bank is created from harness-es material.
5. **Bank hygiene.**
   - DEV-EOU-ES and the reserved mTEDx-es talks (test / valid, shards 0-1) are written into an exclusion list (talk ids +
     normalised texts).
   - Any SUB15 data arm that adds web / TED-like audio (for example Granary-es) must exclude them.
   - SUB15 never selects on DEV-EOU-ES.
6. **Box separation.**
   - The stage 2 takes the GPU lock on the other DGX.
   - The CPU decodes never run on the box that trains SUB15: `cf_s_full.sh`'s speed guard kills a run at 1.15x the
     predicted ms / step, so CPU contention could cause a false SPEED ABORT (OBSERVED: guard code; the abort under
     contention is a HYPOTHESIS).
7. **Decisions stay decoupled.** No SUB15 data or recipe choice is changed because of this run's EOU findings until
   SUB15's own pre-registration closes.

## 7. Priority and scheduling (no auto-launch)

**Order:**
1. **The SUB15 M1 / M2 measurements (~15 GPU-min, SUB) and the first SUB15 full FT keep priority on the box they choose.**
   The full FT is ~2.5 h at S5 speed (OBSERVED 9,011 s for 12,670 steps).
2. **ES Stage 2 runs on the other DGX in the same window.** It costs ~0.5-1.0 GB10-h, uses its own lock and its own CPU,
   and shares no inputs or outputs with SUB15.
   - S5 s0 sits on dgx1. If SUB15 takes dgx1, copy final.nemo to dgx2 and verify sha256 583d6954 there. ES manifests are
     already present on dgx2 (OBSERVED).
3. **If SUB15 needs both DGX at once** (two seeds or two arms in parallel), ES Stage 2 **waits** for the first free slot.
   It must **not** be queued behind SUB15 on the same lock, because `s2_mix.sh` would then auto-start when the lock is
   released.

**Why run it now and not after SUB15:**
- It is the only missing cell of the Factory-v1 EOU contract: ES and an S5 parent.
- It gives SUB15 the ES EOU reference that SUB15's own goal ("EOU / empty / latency preserved") needs.
- It is ~5 % of one SUB15 full FT in GPU terms.

**CPU work that can start immediately, without approval for GPU:** the `prepare_eou_v2.py` port, the bank build and seal,
the v4 identity preflight, and the pre-registration commit. Launching the stage 2 needs the user's go.

**Preflight before launch (CPU, minutes):**
- (a) The v4 code identity: run `two_stage.py --batch-audit 300` on the FR G1E mix with the v4 flags on the launch commit.
  It must equal X 14abd104 / Y 5f8b73b6 (AUT §6). That needs the FR manifests, present on dgx1 `~/ftfr`.
- (b) `--batch-audit` on the ES mix: zero clips dropped by `--max-dur 20`.
- (c) The sha256 of the parent on the executing box.
- (d) Read the env from `/proc` after the tmux launch, before declaring RUNNING.
- (e) One capped waiter (2 h timeout), no polling.

## 8. Missing inputs (to create) and UNKNOWNs

**To create:**
- `prepare_eou_v2.py` / `prepare_final_it.py` ES generalisation, with a test that FR is unchanged.
- DEV-EOU-ES (decoded) and FINAL-EOU-ES (sealed) from mTEDx-es shards 2-9.
- The exclusion list of reserved mTEDx-es talks.
- The `TIMEOUT_S2` ops amendment.
- The pre-registration commit (this plan, condensed into `finetune/eou/plans/ES-S2-prereg.md`).

**UNKNOWN:**
- The GB10 wall of stage 2.
- The share of mTEDx-es clips that pass the class rules (the bank may come out under 270; the rule still caps per stratum).
- Whether internal-class FE in ES matches FR. R2 is report only.
- The ES stage-2 seed spread.
- Whether the v4 selection rule behaves the same on an S5 parent (which step gets selected).

## CPU prep status (2026-10-26)

CPU only on dgx2 (nice 19 / ionice idle, <= 4 workers, no GPU, no model decode, `~/gpu.lock` untouched, anonymous Hub
reads). Pre-registration: trainB `finetune/eou/plans/ES-S2Q-prereg.md` (356eec0). **Stage 2 NOT launched.**
- Commits (trainB, local): 9d2ccad `prepare_eou_v2.py --lang es` (+ SRC / ENDING es; fr unchanged, unit test);
  6fadfeb `s2_mix.sh` A1 `TIMEOUT_S2` (default 3600) + A2 selection summary; 8c5733a `two_stage.py --device cpu`
  (audits only, ops amendment A3, needed to audit on a busy-GPU box); 356eec0 pre-registration.
- v4 identity (OBSERVED, CPU, git 8c5733a): FR G1E mix X 14abd1047c66d1df / Y 5f8b73b686b48e15 = frozen v4,
  IDENTICAL. ES mix X 716e67088eca5382 / Y 500bc3621129cf16 (4,800 samples, EOU share 1.0); 0 ES clips > 20 s.
- Banks (mTEDx-es train shards 2-9, 3.51 GB downloaded; test / valid 28 talks and shards 0-1 34 talks reserved):
  DEV-EOU-ES 258 clips / 57 talks / 0.56 h (internal 50 / 50 / 50 / 8, endpoint 100), SHA256SUMS sha256
  19713f2fddc1f16cf5036106ba7d8c5c82a83152d84291d3027663dd9dbd6495 (dgx2 `~/ft/eou_es/bank/dev`).
  **FINAL-EOU-ES SEALED**: 265 clips / 83 talks / 0.61 h (50 / 50 / 50 / 15, endpoint 100),
  **SEAL.sha256 b8a15e44d92d13d75b1a034c4a944325f4a6f921fc13eea84c80871fef1e51e7**; tgz c148f09b (55.7 MB) in
  `~/ft/eou_es/sealed/`; local FINAL audio, manifests and raw shards removed. Never decoded.
- Leakage: 0 text / 0 talk-id / 0 speaker-field overlap of both halves with train_*_es, mix/es train_*, the ES-D2 mix
  (dgx1, read-only copy) and harness-es; DEV vs FINAL 0; banks vs reserved talks 0. Exclusion list
  `~/ft/eou_es/bank/exclusions_es.json` (6da7d342).
- Disk: ~/ft/eou_es 277 MB after cleanup (peak ~3.8 GB); dgx2 1.6 TB free.
- Pending (need a model): launch-box parent sha256 + `/proc` env; Stage 2 GPU 25-50 min; PROBE ~5 min; s2_reset
  harness-es CPU 45-70 min; DEV-EOU-ES decode CPU 15-25 min.
