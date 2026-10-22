# EOU-120M IT + FR: release finalisation (2026-10-22, RTX 6000 Ada, Vast, egress London)

Goal of the day: close RC-IT v4 and RC-FR v1 as reproducible, packaged, serving-qualified release candidates.
No new research axis. Frozen and NOT reopened: RC-IT v4 (0a4d1b10), RC-FR = FR-S2 (7814141d), FR-v2 (FAIL, closed).
FINAL-IT and FINAL-FR are spent: their frozen results are quoted, never re-decoded. FINAL-EOU-FR-v2 stays sealed.

## 0. Box and inventory (OBSERVED 10:17-10:45 UTC)
- RTX 6000 Ada 49 GB, driver 580.126.09 / CUDA 13.0; EPYC 7542, **cgroup cpu.max 3071999/100000 = 30.7 CPUs**
  (128 visible); 503 GB RAM; root overlay 100 GB; /dev/shm 62 GB (< 120: audio dirs on the root disk, prov.sh).
- Network: no proxy; TLS issuers public (Amazon / GlobalSign / Sectigo); HF 7.4 MB/s single stream, 31.9 MB/s 8-way.
  (A first box on the same host, egress port 19230, gave 1.7-2.9 / 7.2 MB/s and was rejected.)
- Repo `lw-finetune-tooling` 2c03ceb (boxwatch heartbeat added today). HF token: a new one (stdin -> /root/.hf_token,
  mode 600). It was pasted in chat, so it counts as exposed: private staging only; rotate before any public release.
- Hub inventory (`hf_tree.json`, 1,497 files, 62 GB): RC checkpoints, parents, Mynah packs (rc/it-v4, rc/fr-s2v4),
  data snapshots it-c250 (120,367 files, 35.0 GB as WAV) and fr-g1e (101,177 files, 22.2 GB), frozen results of
  2026-10-18..21. Restored and verified: 132 hub files, 0 bad (size + LFS sha256).

## 1. Watchdog
`boxwatch.sh hb` every 300 s in tmux `watchdog`: per tmux job a progress token (log size + a per-job progress
command), GPU util / VRAM / power / pstate, CPU vs the cgroup quota, free disk, last log line; CHECK after 2 unchanged
heartbeats, STALLED after 3, GPU-IDLE for a registered GPU job; alerts in `/root/ft/logs/alerts.log`, read by the dev
machine every 5 min.

## 2. Reproduction training -- DECIDED NO (2026-10-22 10:45 UTC, before any launch)

- **IT: REPRO TRAIN REQUIRED = NO.** The release artifact is the frozen checkpoint 0a4d1b10 (restored from the Hub,
  LFS sha256 verified). Its parent (C250-36k 7e919281), the exact flags (metrics.json args), the data snapshot
  it-c250 (index + per-shard sha256) and the selection rule are archived. Fresh-box evaluation reproduction already
  exists: Japan 2026-10-19 reproduced the Singapore numbers exactly (parent greedy 22.27, v4 s2_reset FE / R / L /
  WER identical). A retrain would only show seed / kernel nondeterminism; no release proof depends on it.
- **FR: REPRO TRAIN REQUIRED = NO.** Same reasoning: 7814141d + parent 1271deff + flags + snapshot fr-g1e archived;
  fresh-box reproduction 2026-10-21 exact (per-set error counts identical), Mynah parity done.
- Declared limitation instead: the training runs were never re-executed (one seed each); bit-exact training
  reproducibility is not claimed. The training question is closed for today; GPU idleness during qualification is
  expected (the GPU work today is the Mynah CUDA server qualification).
- Consequence: the fr-g1e snapshot restore was stopped (only a retrain would need it); the it-c250 snapshot is
  restored only for the DEV730 audio of the IT parity matrix.

## 3. Release qualification (frozen checkpoints only)
- Re-export both Mynah packs with today's export_to_mynah.sh and compare with the archived packs (byte identity).
- IT: the parity matrix that FR got on 2026-10-21 (DEV730 harness clips, 2 s quiet as PCM16; NeMo s2_reset dump vs
  Mynah CPU CLI f32 vs CUDA server f32 / bf16; PRE_EOU / POST_RESET classes), the CUDA engine gate, VRAM, realtime
  serving smoke at 16 / 64 / 128 streams. FR: 2026-10-21 results are quoted, not re-run.
- Package load test: each staged package downloaded fresh from the Hub into an empty directory, SHA256SUMS checked,
  model.nemo loaded in NeMo (example script) and the Mynah pack in the CLI (CPU) and the CUDA server, on one public
  FLEURS clip; outputs compared with the package's expected output.

## 4. FR-H2 -- PRE-REGISTERED (2026-10-22 ~11:05 UTC, before launch; GO given by the user)

WER opportunity review (2026-10-22 ~10:55 UTC, from frozen results only): IT stays frozen at FINAL-IT 20.49 (closed
null axes: encoder KD, CTC, soups, schedule = same ceiling, static selection, tokenizer; LM rescoring does not
transfer; ranking / N-best KD = a multi-day engineering project). FR: the only high-prior axis is exposure.

**Hypothesis:** the plain FR model is exposure-limited; doubling the horizon on the G1E mix lowers DEV WER without
losing the OOD robustness that Granary-fr bought.
Evidence: FR-P1 (8k) -> FR-E1 (11,881) = -2.25 on harness-fr; IT W4 8k -> 16k = -3.36 (16k -> 36k only -0.5); FR at
11,881 is ~2 passes over its 216 h vs IT C250-36k ~5 passes. Against: IT saturation past 16k; MLS-fr (weight
0.2356, 9,608 clips) is seen ~9x at 23,762 steps.

**Run FR-H2 = FR-G1E with `STEPS=23762`.** Only experimental axis = training horizon. The W4 schedule is defined
fractionally (two_stage.py: warm-up = steps / 20, WSD decay over the last `--decay-frac 0.2`), so it stretches
proportionally with the horizon: warm-up 1,188 steps (G1E 594), decay from step ~19,010 (G1E ~9,505); peak LRs, the
fractional shape, data and seed are identical; the LR schedule in absolute steps is NOT identical.
Everything else = G1E (metrics.json args read back): stock init, enc 4e-4 / dec + joint 3e-4, bs 16, max-dur 30,
seed 0, `--final-eof 1` (final.nemo = last step), `--abort-wer 1000:85`, the G1E mix (MLS-fr 40 h 0.2356, CV-fr
0.2020, VP-fr t50 0.1347, FLEURS-fr 0.1010, Granary-fr agree <= 20 0.3267; 97,675 utts, manifest hashes as the
2026-10-21 restore report). Launch: `MIX=<G1E mix> STEPS=23762 TAG=fr-h2-g1e-w4-23762 bash finetune/jobs/fr_p1.sh`.
In-run evals are diagnostic only.

**Primary:** harness-fr macro greedy, 0.5 s flush (lm_fusion_probe, fr_p1.sh's own probe) vs G1E 25.54 (archived,
same tool). **Retention gate:** FR-OOD1 macro greedy 0.5 s (tedxdev + smtrain, all 800 clips, ood1eval.sh's call)
vs G1E 29.66 (archived). **Secondary (no gate):** beam-8 / oracle-8 / CER / per source; the mTEDx DEV-EOU-FR-v2
bank (274 clips, never used for WER) with G1E and H2 decoded by the same probe on this box.

| class | rule (H2 - G1E, macro greedy 0.5 s) |
|---|---|
| LEVER | DEV <= -1.0 AND OOD1 <= 0.0 |
| BENEFIT | DEV in (-1.0, -0.5] AND OOD1 <= +0.5 |
| NULL | DEV within +-0.5 (and not HARM) |
| HARM | DEV > +0.5 OR OOD1 > +1.0 |
(DEV <= -0.5 with OOD1 in (+0.5, +1.0], or DEV <= -1.0 with OOD1 in (0, +0.5]: DIAGNOSE, no promotion.)

**After:** only LEVER / BENEFIT -> stage 2 v4 unchanged on H2 (FR-S2's flags) with FR-S2's gates (DEV s2_reset, OOD1
once, EOU bars); a passing stage 2 is RC-FR v1.1, characterised on DEV / OOD1 / mTEDx dev only (FINAL-FR is spent;
FINAL-EOU-FR-v2 is not used as an ASR FINAL). No second horizon (18k / 30k / 36k), no other variant, no FINAL.
Budget: one plain run, <= 2 h 30 GPU (G1E: 3,929 s on the same GPU type).
