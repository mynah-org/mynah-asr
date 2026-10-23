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

### Result: FR-H2 -- pre-registered class LEVER (OBSERVED, 2026-10-22, one seed, git 180a05d)
- Train 7,774 s (23,762 steps, ~200 steps/min, 480-490x RT, peak 39.5 GB); final.nemo = step 23,762, sha256
  **d2062aca**28d8db0abff823fe4be8051665bdb4c7d3d15cc8ff929f32cf8bfda6; Hub `runs/ada-fr-h2-g1e-w4-23762` +
  `results/ada-2026-10-22/fr-h2` (archive verified by size).

| macro, 0.5 s flush | greedy | beam-8 | oracle-8 |
|---|---|---|---|
| DEV harness-fr G1E | 25.54 | 23.56 | 19.84 |
| **DEV harness-fr H2** | **22.06** (-3.48) | 20.51 | 16.81 |
| FR-OOD1 G1E | 29.66 | | |
| **FR-OOD1 H2** | **25.66** (-4.00) | | |
| mTEDx dev (secondary) G1E / H2 | 27.15 / **23.24** (-3.91) | | |

Per DEV source (greedy): MLS -3.62, FLEURS -4.42, CV -2.76, VP -3.10; OOD1 tedxdev 25.17 -> 21.03, smtrain
34.15 -> 30.29. **Class LEVER** (DEV <= -1.0 and OOD1 <= 0.0). Reading: OBSERVED, the 11,881-step FR model was
exposure-limited; doubling the horizon helps every in-domain source and the two out-of-domain sets by about the same
amount (no sign that the longer horizon traded OOD robustness for in-domain fit). One seed; a third horizon is NOT
run (pre-registered). Per the pre-registration: stage 2 v4 unchanged on H2 (`fr-s2v4-h2`), FR-S2's gates.

### Result: IT release qualification of RC-IT v4 (OBSERVED, 2026-10-22; DEV730, 2 s quiet as PCM16)
- Re-export of 0a4d1b10 with today's export_to_mynah.sh: model.safetensors / tokens.json / mel_filters byte-identical
  to the frozen pack (rc/it-v4); mynah.json differs only in its `name` field (taken from the output directory).
- NeMo s2_reset on the PCM16 bytes: macro 22.54, FE 8.36 %, R 94.38 %, L 0.72 / 1.92 s = the frozen DEV730 values.
- Matrix (`mynah_parity.py compare`):

| comparison | identical text | <EOU> frames equal | PRE_EOU (1st-segment text identical) | POST_RESET | macro WER |
|---|---|---|---|---|---|
| NeMo f32 vs Mynah CPU f32 | 712 / 730 | 676 / 730 | 27 (27) | 28 | 22.56 vs 22.54 |
| NeMo f32 vs Mynah CUDA f32 | 712 / 730 | 676 / 730 | 27 (27) | 28 | 22.56 |
| Mynah CPU f32 vs CUDA f32 | **730 / 730** (text + <EOU> t) | | | | |
| NeMo f32 vs CUDA bf16 (serving default) | 675 / 730 | 658 / 730 | 83 (46) | 24 | 22.60 |

  As for FR: no text divergence before the first <EOU> in f32 (the 27 PRE_EOU clips differ only in the placement of
  the end-of-stream <EOU>); POST_RESET = the runtime's encoder + decoder reset after a premature <EOU> (24 of the
  divergent clips follow a premature NeMo <EOU>); bf16 = precision divergence, +0.04 macro WER. EOU metrics of the
  runtime equal NeMo's (FE 8.36 %, R 94.66 %, p50 0.72 s).
- CUDA engine gate (tests/test_cuda_stream, 4 IT clips): batch identity, repeat, slot independence all OK.
- Serving (bf16 default, own-tc-bf16-rowstable, cap 128; weights 635 MB): VRAM 1,966 MiB with the model loaded, flat
  at 16 / 64 / 128 realtime streams; GPU util p50 7-10 %; delta lag p99 45-46 ms; 0 errors; realtime == fast feed
  730 / 730, 256 / 256, 256 / 256. Run while the FR-H2 job only waited on the GPU lock (no training running), host
  load average up to 8.6 (functional smoke, not a benchmark).
- Mynah CPU CLI: 730 clips in 845 s (16 x 1 thread), 0 errors, final == concatenated deltas on 730 / 730.
- NeMo release example (`examples/transcribe_nemo.py`) vs the frozen s2_reset dump: text 40 / 40, <EOU> frames 40 / 40.

### Result: stage 2 v4 on FR-H2 (`fr-s2v4-h2`) -- FR-S2 gates: RC-FR CANDIDATE (OBSERVED, 2026-10-22, one seed)
- Unchanged FR-S2 command with INIT = H2 (d2062aca); 816 s; the v4 rule selected **step 1250** (in-run 24.27,
  EOU 99.5 %, 1 empty; FR-S2 had selected 250). final.nemo sha256 **3130534d**577e340df129fea9074a63a070aaa05292bcf3aadd96b4aa39fd969c;
  Hub `runs/ada-fr-s2v4-h2` + `results/ada-2026-10-22/fr-s2v4-h2` (archive verified).
- s2_reset protocol (`--lang fr --post-pad-s 2.0`, decoder reset, 2 s quiet), same tool as FR-S2:

| set | parent H2 | s2_reset | delta | FE | FE_any | recall | latency p50 / p90 s | empty |
|---|---|---|---|---|---|---|---|---|
| DEV MLS | 19.90 | 20.11 | +0.21 | **18.0 %** | | 96.5 % | 0.64 / 1.20 | 1 |
| DEV FLEURS | 26.30 | 26.22 | -0.08 | 4.5 % | | 98.0 % | 0.80 / 1.12 | 0 |
| DEV CV | 24.05 | 23.94 | -0.11 | 5.5 % | | 95.0 % | 0.80 / 1.12 | 0 |
| DEV VP | 18.04 | 18.06 | +0.02 | 3.0 % | | 99.0 % | 0.56 / 1.28 | 0 |
| **DEV macro / pooled** | 22.07 | **22.08** | **+0.01** | **7.75 %** | 8.12 % | **97.12 %** | **0.72 / 1.12** | 1 / 1 |
| **FR-OOD1** (800, once) | 25.64 | **24.97** | -0.67 | 8.5 % | 10.0 % | 96.75 % | 0.96 / 1.68 | 7 / 13 |

- All seven FR-S2 gates PASS -> **RC-FR CANDIDATE**.
- vs RC-FR v1 (same tool, archived): DEV 25.64 -> **22.08 (-3.56)**, OOD1 29.53 -> **24.97 (-4.56)**; EOU class
  unchanged (FE 7.25 -> 7.75 %, R 97.5 -> 97.12 %, p50 0.72 = 0.72 s). The MLS long-read false endpoints are unchanged
  (DEV MLS FE 18.0 %, as v1): the declared limitation carries over.
- Not available: a FINAL number (FINAL-FR is spent; FINAL-EOU-FR-v2 is not an ASR FINAL). The candidate is
  characterised on DEV / OOD1 / mTEDx dev only. Mynah qualification of 3130534d: `day22frq.sh` (parity-fr2).

## 5. Release comparison (frozen numbers only; DEV numbers never in the FINAL column)

| | IT: RC-IT v4 | FR: RC-FR v1 | FR: RC-FR v1.1 (candidate) |
|---|---|---|---|
| base / params | Parakeet Realtime EOU 120M, FastConformer cache-aware [70, 1] + RNNT, 120M | same | same |
| plain parent | C250-36k (7e919281), 36k steps, ~357 h | FR-G1E (1271deff), 11,881 steps, 216 h | FR-H2 (d2062aca), 23,762 steps, 216 h |
| stage 2 | v4, step 250 | v4, step 250 | v4 (unchanged), step 1250 |
| checkpoint sha256 | 0a4d1b10 | 7814141d | 3130534d |
| **research selection metric** (DEV, s2_reset, macro WER) | DEV730 22.54 | DEV-fr 25.64 | DEV-fr 22.08 |
| **spent FINAL metric** macro WER / CER | **20.49 / 6.95** (FINAL-IT) | **27.76 / 11.62** (FINAL-FR) | none (FINAL-FR spent) |
| FINAL false endpoint / recall / p50 / p90 | 3.50 % / 96.17 % / 0.88 / 2.24 s | 7.57 % / 96.73 % / 0.80 / 1.60 s | n/a (DEV: 7.75 % / 97.12 % / 0.72 / 1.12 s) |
| FINAL empty | 9 / 1,826 | 10 / 2,113 | n/a (DEV 1 / 800) |
| retention / OOD | | FR-OOD1 29.53 | FR-OOD1 24.97 |
| **Mynah runtime metric** | DEV730 CPU / CUDA f32 22.56, bf16 22.60 | FINAL-FR 27.90 (bf16); DEV 25.68 (bf16 realtime) | DEV 22.20 (f32), 22.24 (bf16) |
| Mynah parity | CPU f32 == CUDA f32 730 / 730; no pre-<EOU> text divergence | 800 / 800; idem | 800 / 800; idem |
| **serving characterisation** | 1,966 MiB VRAM, 16 / 64 / 128 realtime streams, 0 errors | 1,941 MiB, 16 / 64 / 128, 0 errors | 1,966 MiB, 16 / 64 / 128, 0 errors |
| known limitations | FLEURS-style internal pauses (FINAL FLEURS FE 16.4 %) | long-read internal pauses (FINAL MLS FE 25.2 %) | same as v1 (DEV MLS FE 18.0 %); no FINAL number |
| package (Hub, private) | release/mynah-eou-it-120m | release/mynah-eou-fr-120m | release/mynah-eou-fr-120m-v1.1 |

## 6. Journal (UTC; CEST = UTC + 2)

| time | stage | GPU | CPU | result / problem |
|---|---|---|---|---|
| 10:17-10:30 | box checks: 19230 rejected (HF 7 MB/s), 19325 accepted (32 MB/s) | idle | - | TLS public, no proxy |
| 10:33-11:06 | prov.sh (venv 33 min: torch + CUDA wheels at ~6-8 MB/s) | idle | low | PROV-DONE 11:06 |
| 10:35 | watchdog heartbeat every 300 s | - | ~0 | alerts.log read by the dev machine every 5 min |
| 10:38-10:58 | Hub restore (132 files) + snapshots it-c250 / fr-g1e | idle | extract | fr-g1e stopped 10:45 on a premature no-train decision, restarted 10:50 (user correction) |
| 10:55 | WER opportunity review -> FR-H2 (GO), IT frozen | | | prereg 180a05d |
| 11:03 | H2 launcher stopped on its own seal-path bug (no training) | | | fixed, relaunched 11:03 |
| 11:06-13:17 | **FR-H2 training** | 61-100 %, ~40 GB, ~250 W | IT NeMo dumps + CLI at nice | 23,762 steps in 7,774 s |
| 13:17-13:40 | H2 probes; IT CUDA parity + serving smoke in the lock gap | low | probes | **H2 LEVER** (DEV -3.48, OOD1 -4.00) |
| 13:42-14:00 | **stage 2 on H2** | ~80 % | - | step 1250 |
| 14:00-14:14 | s2_reset DEV + OOD1 (CPU); FR v1.1 Mynah GPU passes alongside | low (server) | 28 threads | **RC-FR CANDIDATE** |
| 14:14-14:49 | v1.1 NeMo dumps, CLI, serving smoke | low | 22-28 | parity as v1 |
| 14:19-15:00 | packages IT / FR v1 / FR v1.1: build, upload, verify, fresh-download load test | - | low | all PASS |

## 7. Destructive-instance readiness audit (2026-10-22 15:05 UTC)
- repo commits pushed: PASS (lw-finetune-tooling clean, = origin).
- IT canonical model archived + hash: PASS (release/mynah-eou-it-120m/model.nemo 0a4d1b10, LFS sha256 = SHA256SUMS).
- FR canonical model archived + hash: PASS (release/mynah-eou-fr-120m 7814141d; candidate release/mynah-eou-fr-120m-v1.1
  3130534d; runs/ada-fr-h2-g1e-w4-23762 d2062aca, runs/ada-fr-s2v4-h2).
- Mynah packages archived: PASS (mynah/ in each package; IT = the frozen rc/it-v4 bytes, FR v1 = rc/fr-s2v4 bytes).
- configs / tokenizers archived: PASS (nemo/model_config.yaml + tokenizer files, inference.json, provenance.json).
- evaluation results archived: PASS (results/ada-2026-10-22/{fr-h2, fr-s2v4-h2, qual-it, qual-fr-v1.1}).
- serving results archived: PASS (qual-it, qual-fr-v1.1, results/serving.json in each package).
- logs / journal archived: PASS (results/ada-2026-10-22/day: logs incl. heartbeat + alerts, scripts, inventory).
- checksum verification against HF: PASS (close-out 258 files, 0 bad; each package verified file by file and
  re-downloaded into an empty directory: SHA256SUMS OK, NeMo + Mynah 3 / 3 identical, CUDA gate 18 OK).
- no unique required artifact only on Vast: PASS (left on the box: Hub copies, regenerable padded wavs, restored
  snapshots).
- token removed from box: PASS. GPU jobs: NONE. CPU jobs: NONE.

**SAFE TO DESTROY VAST INSTANCE.** The HF token used today was pasted in chat: revoke it before any public release.
Open decision (user): publish RC-FR v1 (FINAL 27.76) and / or the v1.1 candidate (DEV 22.08, OOD1 24.97, no FINAL;
a fresh FINAL-FR-v2 bank would be needed for a held-out number).

## 8. FINAL-FR-v2 -- PRE-REGISTERED (2026-10-22 ~15:20 UTC, before the bank is built; model-free)

Purpose: a fresh held-out number for RC-FR v1.1 (3130534d). FINAL-FR (2026-10-19) is spent; FINAL-EOU-FR-v2 is an
EOU bank and stays sealed and untouched.

**Builder:** `prepare_final_it.py --lang fr` (the FINAL-FR builder, rules unchanged: model-free selection in sha1
order, natural endings, >= 3 words for tedx, text dedup across banks, per-speaker / per-talk caps) with:
- `--banks mls,cv,vp,tedx --n 400 --tedx-per-talk 25`;
- `--seed-final <FINAL-FR final_*.json>`: every FINAL-FR text seeds the dedup, so no FINAL-FR clip / sentence is
  re-selected (the selection order is deterministic: without it the same clips would come back);
- `--exclude-texts`: every FR training manifest (G1E mix), the DEV-fr harness, FR-OOD1 (tedxdev + smtrain) and the
  DEV-EOU-FR-v2 manifest;
- `--vp-exclude` = our VP pools + the FINAL-FR VP bank (as FINAL-FR: speakers of our pools);
- tedx: mTEDx fr TRAIN shards **10-11** (new flag `--tedx-shards`), dropping every talk with a row in train shards
  **0-9** (`--tedx-exclude-shards`: FINAL-FR used 0-1, both EOU-v2 banks 2-9; this excludes them by construction
  without opening FINAL-EOU-FR-v2), plus the mTEDx test / valid talks and every Granary-fr video id (as FINAL-FR).
- Sources and splits: MLS-fr dev, CV17-fr validation, VoxPopuli-fr validation (the FINAL-FR splits, other clips;
  speakers may overlap FINAL-FR, never training), mTEDx-fr train shards 10-11.
- Not included (declared): FLEURS-fr (its dev sentences are FINAL-FR's), Speech-MASSIVE (validation spent, train in
  FR-OOD1, test gated). Four banks instead of six: the macro is NOT comparable to FINAL-FR's six-bank macro.

**Seal:** SHA256SUMS (manifests + every wav) and SEAL.sha256 = sha256 of SHA256SUMS, recorded in a commit BEFORE any
decode; sealed tgz archived on the Hub.

**One decode, after the seal commit, of RC-FR v1.1 only** (= the FINAL-FR protocol, section O):
1. primary: `s2_reset.py --lang fr --post-pad-s 2.0 --models s2_reset --per-set 100000` on the natural banks:
   per bank WER / CER / empty / FE / FE_any / R / L p50 / p90 / split; macro over the 4 banks, EOU pooled; trim
   variant secondary;
2. product: Mynah CUDA server bf16 on the same banks (2 s quiet, PCM16), `mynah_parity.py score`.
Reporting wording (fixed now): FINAL-FR-v2 is a *fresh evaluation bank relative to our process* (disjoint from our
training / DEV / OOD / earlier FINAL banks as far as verifiable), not a universally uncontaminated benchmark.
No band, no comparison used for any decision; v1 is NOT decoded on it (no selection between v1 and v1.1 on FINAL-FR-v2).
Nothing changes in response.

## 9. ES-1 third-language replication -- PRE-REGISTERED (2026-10-22 ~15:20 UTC, before any ES data stat is seen)

Question: does the frozen IT / FR plain recipe give a useful Spanish 120M model on the first attempt? One run, one
number, no tuning.

- **Data (A2-style, frozen definitions):** `data_mix.sh es` = MLS-es 40 h cut over all speakers (prepare_it.py, the
  FR rules), CV17-es 40 h (<= 20 clips / speaker), VoxPopuli-es 40 h (<= 60 / speaker), FLEURS es_419 train; text rule
  textnorm es (NFKD, typographic apostrophes; digit clips dropped).
  - VP teacher filter (t50, Nemotron 0.6B, `a2_vpfilter.sh LANG_=es FILTER_ONLY=1`) ONLY if the teacher and its
    300-clip CLI gate are ready by **15:45 UTC**; otherwise VoxPopuli-es unfiltered (`train_vp.json`). Either way the
    choice is recorded before training.
  - No Granary-es (pool + teacher scoring would be the critical path): declared limitation; the recipe analogue is
    FR-E1 (A2 only), not G1E / H2.
- **Weights:** the A2 weights (MLS 0.35, CV 0.30, VP 0.20, FLEURS 0.15), fr_p1.sh's default.
- **Recipe:** fr_p1.sh with `LANG_=es` = the H2 / G1E recipe: stock init + stock tokenizer, whole encoder 4e-4,
  decoder / joint 3e-4, W4, WSD fractional (5 % warm-up, last 20 % decay), bs 16, max-dur 30, seed 0, final = last
  step, `--abort-wer 1000:85`; no CTC / KD / tokenizer / stage 2 / LM.
- **Horizon rule (the H2 exposure lesson):** H2 consumed 995.9 audio-hours over 216.3 physical hours = 4.60 passes.
  ES steps = round(4.60 x H_phys / h_step), h_step = 16 x the weight-averaged mean clip duration of the frozen mix
  (hours per step), capped at 23,762. Reported: physical hours / clips / speakers per source, weights, steps, audio
  hours consumed (metrics.json), passes, wall, GPU time, peak VRAM, xRT.
- **Evaluation (sealed before training, model-free):** the harness-es = MLS-es test eval (prepare_it), FLEURS es_419
  eval, CV17-es test 200, VP-es test 200 (prepare_mix); SHA256SUMS recorded before the launch. fr_p1.sh probes:
  greedy / beam-8 / oracle-8 / CER, raw EOF and 0.5 s flush, per source + macro; primary = macro greedy 0.5 s.
  Stock EOU-120M (English) probed on the same harness as the zero-shot reference.
- **Reading bands (descriptive, set now):** <= 25 strong transfer; 25-30 transfers; 30-40 weak; > 40 or abort =
  failure of an IT / FR assumption. No ES v2 today; stage 2 ES only if the plain run ends with >= 45 min of box time
  left AND macro greedy <= 30, with the v4 recipe unchanged.

## 10. Late session (15:10-16:40 UTC): FINAL-FR-v2 and ES-1 ABORTED, no result seen

- **FINAL-FR-v2: ABORTED BEFORE SEAL.** The build never left the metadata phase (speaker / text exclusions over the MLS /
  CV / VP train + test splits); no bank, no SHA256SUMS, nothing decoded. It can be rebuilt on a new box from the
  frozen pre-registration (a3b4563 + fca3074) without contamination. RC-FR v1.1 therefore still has no held-out
  number.
- **ES-1: ABORTED BEFORE TRAINING.** Only data download had started (MLS-es metadata, CV / VP shards); no Spanish
  statistic was computed, no model trained or decoded. It restarts on a new box from its pre-registration (a3b4563),
  unchanged. The es code support (textnorm, MLS / FLEURS configs, fr_p1.sh LANG_) is committed (00dcee2).
- **Cause 1, ours, measured -- metadata read amplification.** `Hub.column` (prepare_final_it.py) with the library's
  default read-ahead pulls MB-sized ranges of audio parquets to read a projected column. `--hub-block-kb 256` is
  I/O-only, validated on two deterministic probes (`blockeq.py`, the builder's own reader):

| parquet | rows (sha) | default read-ahead | 256 KiB |
|---|---|---|---|
| MLS-fr dev | 2,416 (2da6d609), identical | 302.2 MB, 184.2 s | **52.5 MB, 45.8 s** |
| VP-fr test | 1,742 (2d3aa85f), identical | 40.5 MB, 31.1 s | **6.4 MB, 6.5 s** |

  i.e. ~6x fewer bytes and 4-5x less wall time for the same rows. The first FINAL-FR-v2 attempt (default read-ahead)
  had read ~14 GB in 38 min without finishing the metadata; logs kept (f2build-defaultreadahead.log).
- **Cause 2, the host -- network degradation** (`network-2026-10-22.txt`): HF 7.4 MB/s single / 31.9 MB/s 8-way at
  10:30 UTC; at 15:45 UTC every destination ~2-3 MB/s single stream (HF anonymous 1.9, authenticated 2.3, pytorch 3.0,
  PyPI 2.6, GitHub 3.0); 0.3-0.9 MB/s aggregate during the restarted build. Box load < 1: not our CPU.
- **Before the next FINAL-FR-v2 / ES-1 attempt:** apply the small-block metadata reads to every Hub parquet reader
  (prepare_mix.py, canary/prepare_it.py, prepare_final_it.py default), re-validate row identity, network smoke test
  at box start, then run both pre-registrations unchanged.

## 11. 2026-10-23: FINAL-FR-v2 and ES-1 restarted on the DGX (pre-registrations unchanged)

**Box (OBSERVED 07:20 UTC):** `ssh dgx` = DGX Spark GB10 (sm_121, aarch64, 20 cores, 121 GB unified, no cgroup CPU
quota), 1.2 TB free, user account (no root). Network: HF / cdn-lfs TLS issuer Amazon RSA 2048 M01, PyPI GlobalSign
(no MITM); a 100 MB LFS range read in 2.0 s (52 MB/s). Repo checkout synced to 81fc376 (the 2026-10-15 local edits
stashed, not lost). Write token placed by the user in a 0600 file, read only from that file, never printed.
NeMo 3.0.0 / torch 2.11.0+cu128; stock EOU-120M sha256 6603a22a = the Hub LFS sha. Mynah CLI + CUDA server built
from 7eccdac for sm_121 (CUDA 13.3). GB10 is ~3x slower than an L40S for training (2026-10-15).

**Code before the runs (I/O and paths only):**
- 7eccdac: token file `HF_TOKEN_FILE` (default `~/.hf_token`), FT / --stock / --out / --tmp defaults home-relative
  (identical on the root boxes); `prepare_final_it.py --hub-block-kb` default 256; `canary/prepare_it.py` MLS metadata
  read-ahead 8 MiB -> 256 KiB. 2ac8abd: boxwatch home-relative. `test_finetune.py` 35 passed on the DGX.
- MLS reader re-validated (`mlsblockeq.py`, prepare_it.mls_meta_file, 8 MiB vs 256 KiB):

| parquet | rows (sha) | 8 MiB | 256 KiB |
|---|---|---|---|
| MLS-es test | 2,385 (f2f38418), identical | 102.0 MB, 8.6 s | 7.1 MB, 10.9 s |
| MLS-es train 0 / 30 | 7,357 (0f31d733), identical | 318.4 MB, 20.0 s | 21.9 MB, 30.9 s |
| MLS-fr dev | 2,416 (f716031b), identical | 104.2 MB, 6.8 s | 7.2 MB, 10.4 s |

  On a fast link the small block costs ~1.5x wall time for ~14x fewer bytes; on a degraded link (2026-10-22) it wins.

**FINAL-FR-v2 inputs:** the same builder flags as the 2026-10-22 attempt; the exclusion / seed manifests are the
archived files (fr-g1e snapshot meta, eou-fr-v2-dev, ood-fr1, FINAL-FR sealed tgz, granary ids; FINAL-FR SHA256SUMS
sha 8b0ebba0 checked) extracted under `~/vroot/<original absolute path>`; each input's sha256 is in f2build.log.
Build started 07:30 UTC.

**ES-1, one clock substitution (declared in 8dc9562, before any ES data statistic was read):** the pre-registered VP rule
"teacher filter only if the teacher and its 300-clip CLI gate are ready by 15:45 UTC" was a box-time deadline of
2026-10-22. On the DGX it reads: **the filter runs only if the teacher and its CLI gate are ready before the ES data
mix completes (DATA-MIX-JOB-DONE); otherwise VoxPopuli-es unfiltered.** Either way the choice is recorded before
training. Everything else in section 9 unchanged (stage 2 rule: ">= 45 min of box time left" is moot on the DGX;
the macro greedy <= 30 condition stands).

### FINAL-FR-v2 SEALED (OBSERVED 2026-10-23 07:45 UTC, DGX, git 7eccdac; nothing decoded)
- **SEAL.sha256 = bd2219e03bb045e2d5b6f952d32a2be3d637ae1710b989505e36dd22ae9931ac** (sha256 of SHA256SUMS: 2,623
  files = manifests + 2,614 wavs, verified file by file after the build). Hub `data/final-fr-v2/`
  (final-fr-v2-sealed-2026-10-23.tgz 566,650,963 B, SHA256SUMS, SEAL.sha256, stats.md, f2build.log).
- Exclusions: 100,727 texts from 13 manifests; dedup seeded with the 2,113 FINAL-FR texts; 290 VP speakers of our
  pools; 86,238 Granary-fr ids; 234 mTEDx talks of train shards 0-9 excluded.

| bank | clips | h | groups | dur p50 s | trail < 0.2 s |
|---|---|---|---|---|---|
| cv (CV17-fr validation) | 400 | 0.65 | 400 | 5.7 | 11 % |
| mls (MLS-fr dev) | 400 | 1.68 | 18 | 15.2 | 18 % |
| tedx (mTEDx-fr train shards 10-11) | 400 | 0.62 | 46 | 4.6 | 95 % |
| vp (VoxPopuli-fr validation) | **107** | 0.26 | 12 | 7.9 | 98 % |

- **VP has 107 clips, not 400** (model-free, the frozen rules): of 507 candidates after the speaker exclusion, the 400
  FINAL-FR texts were the dedup seeds; 1,150 rows dropped for speakers in our train / test pools. Declared before
  the decode: VP is a small bank (wide CI); the 4-bank macro weights it equally as pre-registered.
- Next, unchanged from section 8: ONE decode of RC-FR v1.1 (3130534d): s2_reset natural (primary) + trim
  (secondary), Mynah CUDA bf16 server on the natural banks. v1 is not decoded.

### Result: FINAL-FR-v2, the one-shot of RC-FR v1.1 (OBSERVED 2026-10-23 07:47-08:00 UTC, DGX GB10) -- now SPENT
Seal verified file by file (2,623) before the decode; checkpoint 3130534d / parent d2062aca and the v1.1 Mynah pack
(its package SHA256SUMS) verified. Decoded once, nothing changed in response. Hub
`results/dgx-2026-10-23/final-fr-v2-results` (16 files). **FINAL-FR-v2 is SPENT from this point: no selection,
tuning or recipe decision may use it.** It is a fresh evaluation bank relative to our process (disjoint from our
training / DEV / OOD / earlier FINAL banks as far as verifiable), not a universally uncontaminated benchmark.

Primary = NeMo s2_reset frozen protocol, natural endings (greedy, decoder reset after `<EOU>`, 2 s quiet):

| bank | clips | WER | CER | empty | FE | FE_any | recall | latency p50 / p90 | Mynah CUDA bf16 WER |
|---|---|---|---|---|---|---|---|---|---|
| cv | 400 | 22.37 | 9.13 | 0 | 6.0 % | 6.5 % | 94.8 % | 0.80 / 1.12 s | 22.42 |
| mls | 400 | 23.73 | 10.83 | 2 | **20.8 %** | 21.0 % | 97.8 % | 0.64 / 1.36 s | 23.73 |
| tedx | 400 | 26.23 | 13.89 | 4 | 14.0 % | 14.0 % | 99.2 % | 0.88 / 1.68 s | 26.44 |
| vp | 107 | 19.68 | 9.96 | 0 | 2.8 % | 2.8 % | 97.2 % | 0.64 / 1.28 s | 19.49 |
| **macro (4 banks)** | 1,307 | **23.00** | **10.95** | 6 | **12.70 %** | 12.93 % | **97.25 %** | 0.80 / 1.36 s | **23.02** |

- Trim variant (secondary): macro 23.04, FE 12.47 %, recall 97.4 %.
- Mynah runtime (the CUDA server, bf16 serving default, encoder + decoder reset): macro 23.02 / CER 11.01, FE 12.7 %,
  recall 96.63 %, 0 runtime errors; +0.02 vs the NeMo protocol.
- Descriptive only (different banks, never a selection): DEV-fr 22.08, FR-OOD1 24.97; FINAL-FR v1 27.76 is a 6-bank
  macro of other clips and the v1 model, NOT comparable. The known limitation is confirmed on fresh clips: MLS
  (long read speech) false endpoints 20.8 % and mTEDx 14.0 %, CV / VP <= 6 %.
- The Mynah pack's `mynah.json` name field reads `mynah-eou-fr-120m-h2` (cosmetic; the pack files are the v1.1
  package's, sha-verified).

## 12. FR next R&D: decision gate (2026-10-23, after FINAL-FR-v2; the R&D agents never saw FINAL-FR-v2)

Four analysis-only agents (read-only, a pre-FINAL-FR-v2 snapshot of the reports + the archived DEV / OOD metrics):
C1 EOU / false endpoints, C2 plain-parent WER, C3 prior art, D adversarial review (with a veto round on C1 / C3).

| candidate | hypothesis | prior evidence | contradicting evidence | impl. | GPU (GB10) | expected effect | falsification gate | contamination | verdict |
|---|---|---|---|---|---|---|---|---|---|
| H2-R: LR restart from H2, 6k steps (C2) | H2 limited by schedule / exposure | H2 flat-phase slope -0.3 / 1k; E3-R1 restart -0.60 | H2's tail flat; the restart recovered ~12 % of a doubling; prediction -0.4 is inside the ~0.5 noise | 15 min | ~100 min | DEV -0.4 (0 to -0.8) | DEV >= -0.5 -> NULL | low | NOT RUN (EV inside the noise; C2 and D) |
| H3 fresh ~47.5k (C2, D) | per-doubling return not yet falling for FR | G1E -> H2 -3.48 DEV / -4.0 OOD1 | none strong | 0 | ~13 h | -1 to -3 | pre-register vs H2 | low | DEFERRED (an Ada / L40S box; keep late checkpoints) |
| soup (C2, D) | averaging helps | none | IT soups worse (23.20 / 23.86 vs 22.30) | - | - | - | - | - | REJECTED |
| concat A + gap + B, no `<EOU>` (C1 falsifier) | continuation negatives teach "wait" | prior-art list | synthetic gap (= v2 artefact), skips MLS (> 20 s), contradicts the "A + 1 s + B: EOU in gap" acceptance test | 0.2 h | ~55 min | NULL (C1) | - | low | VETOED (D) |
| truncated-prefix "hold" negatives (C3) | same terminal quiet, incomplete prefix, no `<EOU>`: completeness becomes the cue | Chang 2022, Smart Turn, Li & Shi 2609.04225 | the premature cuts may be at clause ends (C1: 1 / 33 MLS, 1 / 115 mTEDx after a function word); v5-B / CAL label conflict; AUC 0.60 | 1.5-2 h | ~55 min + on-box control | internal FE -25 to -45 % rel. | >= 40 % rel. internal-FE reduction | low | CONDITIONAL on FR-E0 |
| confirmation window W (C1) | the false events are pauses where speech resumes within W | v4 + W 0.32: internal FE 55 -> 33 % at p50 1.12 s | 1:1 latency cost | 0.5 h | 0 | product trade-off | DEV-only characterisation | none | FR-E0 (diagnostic) |

### FR-E0 -- PRE-REGISTERED (this commit, before any v1.1 decode on DEV-EOU-FR-v2)
- One CPU decode of RC-FR v1.1 (3130534d, parent d2062aca) on DEV-EOU-FR-v2 (SHA256SUMS sha 851084f2...; 274 clips),
  the s2_reset command of the 2026-10-21 v4 / v2 decodes; scored by `eou_v2_score.py` against the archived v4 dump
  (per class / stratum FE, recall, latency, and the window frontier W in {0.16, 0.32, 0.48, 0.64, 0.8, 1.04} --
  diagnostic, DEV-only, never a selection; FINAL-EOU-FR-v2 stays sealed).
- `finetune/eou/diag/cut_class.py` (this commit; validated on the archived v4 dump: 115 cuts, 13.0 % mid-phrase vs a
  56.1 % word-boundary control) classifies v1.1's premature cuts against the punctuated mTEDx references.
- **Rule (D's veto round):** the truncated-prefix arm is admitted only if v1.1's premature cuts on DEV-EOU-FR-v2 are
  >= 40 % mid-phrase AND >= +10 pp above the control (its 40 % FE-reduction gate is capped by the mid-phrase share).
  Otherwise it is closed without training, and no FR training arm runs today ("parallelise thinking, not arms").
- This is the second selection use of DEV-EOU-FR-v2.

### Result: FR-E0 (OBSERVED 2026-10-23 08:14 UTC, dgx1 CPU, git 7376f53) -- truncated-prefix arm CLOSED by its rule
- v1.1 on DEV-EOU-FR-v2 (274 clips): reset-WER 23.90; internal class (174 clips) WER 22.22 (v4 25.02), **FE 68.4 %
  (v4 55.2 %, +24 % relative)**, recall 99.4 %; endpoint class (100) WER 28.26, FE 4.0 %, recall 99.0 %, p50 / p90
  0.88 / 1.76 s (v4 0.80 / 1.68). Per internal stratum (v11 / v4 FE): 0.8-1.04 s 32 / 24 %, 1.04-1.6 s 72 / 52 %,
  1.6-2.5 s 90 / 70 %, >= 2.5 s 92 / 96 %. HYPOTHESIS: the longer plain stage (H2) made `<EOU>` at internal pauses of
  >= 1 s more likely on mTEDx (+13 pp, ~2.5 SE at 174 clips); DEV-fr FE moved only 7.25 -> 7.75 % (inside noise).
- **cut_class: 163 premature cuts, 16.0 % mid-phrase vs a 56.4 % control (excess -40.4 pp), 4.9 % at a
  sentence-final mark.** The pre-registered rule (>= 40 % AND >= +10 pp) FAILS: v1.1's premature cuts sit at clause
  punctuation far more often than chance, so a "complete vs incomplete prefix" cue has little to act on. The
  truncated-prefix treatment is closed without training; no FR EOU training arm runs today.
- Window frontier (diagnostic, DEV-only, never a selection), v1.1 internal FE / endpoint p50: W 0.16 52.9 % / 1.04 s,
  0.32 43.1 % / 1.20 s, 0.48 37.4 % / 1.36 s, 0.64 32.8 % / 1.52 s, 0.8 21.8 % / 1.68 s, 1.04 14.9 % / 1.92 s (recall
  99 % throughout). A product-side commit policy remains the lever, at an explicit latency price.
