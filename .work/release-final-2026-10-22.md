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
