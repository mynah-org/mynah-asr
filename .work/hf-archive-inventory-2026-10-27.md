# Private HF archive: inventory and retention plan (2026-10-27)

Status: READ-ONLY analysis. Nothing on the Hub was deleted, moved, squashed or re-uploaded. Every deletion below is a
proposal that waits for Gabriele's explicit approval.

Method: `huggingface_hub` 2.2.0 read APIs on dgx2 (`nice -n 19`, CPU / network only): `list_models` /
`list_datasets` / `list_spaces(author=…)`, `repo_info` (`used_storage`), `list_repo_tree(recursive=True, expand=True)`
(per-file size and LFS sha256, no download), `list_repo_commits`. Small metadata files (`metrics.json`,
`provenance.json`, `*.sha256`, `git_head.txt`, `launch.txt`) were downloaded on the box to read the parent (`init`)
relations. The two FINAL-IT tarballs were downloaded on the box once to compare their member lists (cache deleted
afterwards). Scripts and the raw listing: dgx2 `~/ft/hfinv/{inv.py,meta.py,wip.py,inv.json,meta.json}`. The token was
read only inside Python from `~/.hf_token.rw` and never printed. Status sources: `.work/*.md` (restart sheets, RIF,
RF22, EVO, FVM, FV1, RM12, RM15, BN, AUT, S14) and `mynah-asr-wt-trainB/finetune/eou/plans/*.md`; the key of the
short names is EVO section 0. Sizes are decimal GB (1e9 bytes).

## 1. Repositories of the account

| repo | type | visibility | files | tree size | `used_storage` | commits | project |
|---|---|---|---|---|---|---|---|
| `gabrione/mynah-asr-canary-it-experiments` | **model** (not a dataset: `hf_up.py` / `hf_archive.py` use the default repo type) | private | 2,430 | 74.83 GB | **72.08 GB** | 77 (first = the 2026-10-20 squash) | mynah-asr (this archive) |
| `gabrione/qwen3-tts-italian-expr` | model | public | 38 | 1.67 GB | 1.53 GB | 14 | not mynah-asr (TTS) |
| `gabrione/qwen3-tts-voices` | model | public | 6 | 0.10 GB | 0.10 GB | 6 | not mynah-asr (TTS) |

No dataset or Space exists. The mynah-asr project uses one repo only; everything below is about it. The two public
TTS repos are outside this project and, being public, should not count toward the private-storage quota
(HYPOTHESIS: the ~72 GB the user sees matches the private repo's `used_storage` exactly).

## 2. Totals and what the Hub actually bills

| quantity | GB |
|---|---|
| current tree (sum of all file sizes, every path) | 74.83 |
| of which LFS files | 74.51 |
| of which non-LFS small files (2,203 files: json / logs / txt / md) | 0.32 |
| **unique LFS objects in the current tree** | **72.08** |
| `used_storage` reported by the Hub | **72.08** (equal to the byte) |
| pure path duplication (same sha256 at several paths) | 2.43 tree GB, **0.00 billed GB** |

OBSERVED: `used_storage` equals the unique-LFS size of the current tree to the byte (72,082,424,058). Two
consequences:
1. **The Hub deduplicates LFS by sha256.** A byte-identical file at two paths is billed once, so deleting a duplicate
   path reclaims nothing.
2. **The history holds no orphaned weight.** The 2026-10-20 `--squash` (commit 6829b2c079) removed every older LFS
   object, and since then the 76 commits only added files. But **from now on a delete commit alone reclaims nothing**:
   the deleted object stays referenced by the older commits and stays billed until the history is squashed again
   (section 7).

### By file type (current tree)

| type | GB | files | note |
|---|---|---|---|
| `.tar` data shards | 38.57 | 77 | `data/snapshots/it-c250` (22 shards) and `fr-g1e` (14 shards): the release training data |
| `.nemo` | 27.43 | 59 | 56 unique objects (3 release copies are deduplicated) |
| `.tgz` | 3.69 | 26 | sealed eval banks (FINAL-IT / FR / FR-v2, OOD1, EOU-FR-v2), WIP bank, survival bundles, M1 dumps |
| `.safetensors` (Mynah packs) | 2.76 | 12 | 2 pairs deduplicated |
| `.ckpt` (optimizer / trainer state) | 1.38 | 1 | only `runs/plain-it-e2-enc2e-4-s4000/last.ckpt` |
| json / jsonl | 0.74 | 956 | metrics, hypotheses, manifests |
| git bundles | 0.18 | 17 | local-only branch bundles |
| logs / txt / other | 0.08 | 1,282 | |

### By top-level area

| area | tree GB | billed GB (each LFS object counted once, `runs/` first) | share of billed |
|---|---|---|---|
| `data/`: snapshots 38.70, sealed banks 2.45, FINAL-IT WIP bank 0.70, manifests 0.08 | 41.92 | 41.92 | 58 % |
| `runs/` (114 run dirs, 56 with weights) | 27.97 | 27.97 | 39 % |
| `rc/` (2 Mynah packs) | 0.92 | 0.92 | 1.3 % |
| `release/` (3 packages) | 2.76 | 0.46 (only the v1.1 Mynah safetensors is unique; the 3 `.nemo` are the `runs/` objects, the 2 other safetensors are the `rc/` objects) | 0.6 % |
| `results/` (dated result dirs, survival tgz, bundles) | 1.20 | 1.07 | 1.5 % |
| `recipe/`, `tokenizers/`, `logs/`, root | 0.06 | 0.06 | <0.1 % |

## 3. Byte duplicates (same LFS sha256 at several paths)

| sha256 (16) | GB | paths | canonical | billed reclaim if one path is deleted |
|---|---|---|---|---|
| 3130534d577e340d | 0.460 | `release/mynah-eou-fr-120m-v1.1/model.nemo`, `runs/ada-fr-s2v4-h2/final.nemo` | both (the release package must be self-contained; the run dir is the provenance target) | 0 |
| 7814141d515ca347 | 0.460 | `release/mynah-eou-fr-120m/model.nemo`, `runs/jp-fr-s2v4-g1e/final.nemo` | both | 0 |
| 0a4d1b103297e169 | 0.460 | `release/mynah-eou-it-120m/model.nemo`, `runs/sg-it-s2v4-c250-36k/final.nemo` | both | 0 |
| 9e0e6b50a20f3e4f | 0.459 | `rc/fr-s2v4/model.safetensors`, `release/mynah-eou-fr-120m/mynah/model.safetensors` | both (provenance cites `rc/fr-s2v4`) | 0 |
| 2aa4783d566880a6 | 0.459 | `rc/it-v4/mynah-pack/model.safetensors`, `release/mynah-eou-it-120m/mynah/model.safetensors` | both (provenance cites `rc/it-v4`) | 0 |
| ba7b91e7287ce93b | 0.069 | `results/jp-2026-10-19/fr-g1/train_yg.teacher.json`, `…/sweep/fr-yg/train_yg.teacher.json` | both | 0 |
| 89d0b06cb546b867 | 0.043 | `results/jp-2026-10-19/fr-g1/train_yg.json`, `…/sweep/fr-yg/train_yg.json` | both | 0 |
| 158cb9cdd6ad022d | 0.018 | `results/jp-2026-10-19/fr-g1/yg_agree20.json`, `…/sweep/fr-yg/yg_agree20.json` | both | 0 |

Recommendation: **delete none of them.** They cost 0 billed bytes, and every path is cited by a provenance file,
a README or a report. Deleting would only break citations.

A content-level (not byte-level) duplicate exists: `data/final-it-wip/final-it-wip-2026-10-18.tgz` (0.699 GB) is a
strict subset of `data/final-it/final-it-sealed-2026-10-19.tgz`. OBSERVED on dgx2: 2,874 members, 0 absent from the
sealed tarball, 0 size mismatches on common members, the 10 manifests byte-identical. The sealed tarball adds the
ITALIC bank, `final_all.json`, `stats.md` and `SHA256SUMS`. The WIP tgz is a different gzip stream (different sha), so
it IS billed. -> tier 1.

## 4. Retention rules applied

- **KEEP-FULL:** releases IT / FR v1 / FR v1.1 (package + run dir + rc pack); every release parent (C250-36k, FR-G1E,
  FR-H2); ES-1 (172b1539) incl. its Mynah pack; ES-S5 s0 / s1; ES-H3 s0 / s1; ES-D2 s0 / s1; ES Stage-2 H3; the
  S5-FR pair (current factory compute evidence, FV1 / FVM); AR-A2 s0 (best-current Arabic model); EOU-v3 (former
  production, RIF anchor 3) and its parent IT-E3 plain; all sealed banks (FINAL-IT / FR / FR-v2, FINAL-EOU-FR-v2,
  OOD-FR1, DEV-EOU-FR-v2); both data snapshots (the only copy of the release training data:
  `provenance.json` -> `data_snapshot.py restore --name it-c250 / fr-g1e`); recipe / tokenizers / logs / results /
  bundles; every `metrics.json`, prereg, log, eval and hypothesis file of every run.
- **KEEP-EVIDENCE:** every other CLOSED run. Only its `.nemo` can become a candidate, and only when it is not a parent
  of a kept model, not a release, not FINAL-relevant, not best-current and not needed by an open comparison.
- **DELETE-CANDIDATE:** byte duplicates with a canonical copy (none worth deleting, see section 3), optimizer /
  trainer states of CLOSED runs, smoke / aborted / temporary artifacts, superseded intermediates.
- FINAL-ES (`plans/FINAL-ES-prereg.md`, written 2026-10-27 03:10, l.129-132) decodes exactly H3 s0 1c28256c, H3 s1
  5b24cfe8, D2 s0 fb886e09 and D2 s1 5e89541a: all four are KEEP-FULL. No other Hub checkpoint is FINAL-ES-relevant.
- Smoke runs: the 20 smoke / regression dirs hold only a `metrics.json` (a few kB each). They are the record of the
  smoke gates and cost nothing: keep.

## 5. Inventory: runs with weights (56 dirs; sorted by tier)

Columns: GB per type; `.nemo` sha256 = the first 12 hex of the LFS sha256 (full hashes in Appendix A); parent from
`metrics.json` `args.init` (empty = stock `parakeet_realtime_eou_120m-v1`, sha256 prefix 6603a22a per FV1) or
`provenance.json`; "cited by" = docs that mention the run tag or the sha prefix (key in EVO section 0; P: = trainB
plans). Reproducibility: every run dir holds `metrics.json` with the full `args` (mix, steps, LRs, seed), and the
recipe is in git (`recipe/` on the Hub, branches lw-finetune-tooling / feat/trainer-throughput); training was never
re-executed (one seed per run except where stated), so a deleted `.nemo` is reproducible only by a re-run (GPU
hours: from `wall_s` in `metrics.json`, 0.5-5 h each) with seed-level, not bit-level, fidelity.

| path | total GB | .nemo GB | ckpt/safetensors GB | other GB | .nemo sha256 (12) | experiment | status | class | tier | parent | successor | cited by | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| runs/dgx-ar-d2-r-s12670 | 0.464 | 0.460 | 0.000 | 0.004 | 069c1ee05b19 | AR-D2-R seed 0 | HARM (cv_decon +2.10) | KEEP-EVIDENCE | 2a | stock | AR-A2-R | P:AR-D2-prereg,P:AR-SEED1-prereg,R24 |  |
| runs/dgx-ar-d2-r-s12670-seed1 | 0.464 | 0.460 | 0.000 | 0.004 | c53209a177ca | AR-D2-R seed 1 | HARM REPLICATED | KEEP-EVIDENCE | 2a | stock | AR-A2-R | P:AR-SEED1-prereg |  |
| runs/dgx-r0-ctl-1k | 0.460 | 0.460 | 0.000 | 0.000 | a5f5311003a6 | IT DGX lab R0 control | NULL (lab control) | KEEP-EVIDENCE | 2a | stock | - | RM15 |  |
| runs/dgx-r2-ctc01-1k | 0.460 | 0.460 | 0.000 | 0.000 | ec2dfc054103 | IT DGX lab R2 CTC 0.1 | NULL | KEEP-EVIDENCE | 2a | stock | - | RM15 |  |
| runs/jp-fr-g1-w4-8k | 0.460 | 0.460 | 0.000 | 0.000 | de457d0b6501 | FR-G1 | HARM (+1.75) | KEEP-EVIDENCE | 2a | stock | FR-G1E | RIF,RM15 |  |
| runs/l40s-fr-s2v2-ps | 0.460 | 0.460 | 0.000 | 0.000 | 0817c2f1da6a | FR-v2 stage 2 (pause stretch) | FAILED | KEEP-EVIDENCE | 2a | jp-fr-g1e-w4-11881 | - | AUT |  |
| runs/l40s-s0-cos2k | 0.460 | 0.460 | 0.000 | 0.000 | 696c3ac5c746 | IT tournament S0 (cosine 2k control) | SUPERSEDED (screen) | KEEP-EVIDENCE | 2a | stock | W4 | - |  |
| runs/l40s-s1-wsd2k | 0.460 | 0.460 | 0.000 | 0.000 | 415cd21d0066 | IT tournament S1 (WSD 2k) | SUPERSEDED (screen) | KEEP-EVIDENCE | 2a | stock | W4 | - |  |
| runs/l40s-s2-enc2x2k | 0.460 | 0.460 | 0.000 | 0.000 | 668c959085ad | IT tournament S2 (enc x2, 2k) | SUPERSEDED (screen) | KEEP-EVIDENCE | 2a | stock | W4 | - |  |
| runs/l40s-s3-wsd-enc2x2k | 0.460 | 0.460 | 0.000 | 0.000 | e6c7569fd55b | IT tournament S3 (2k) | SUPERSEDED (screen) | KEEP-EVIDENCE | 2a | stock | IT-S3-4k | - |  |
| runs/l40s-s4-wsd-enc8e4-2k | 0.460 | 0.460 | 0.000 | 0.000 | b92b5c80e0dc | IT tournament S4 (enc 8e-4, 2k) | SUPERSEDED (screen) | KEEP-EVIDENCE | 2a | stock | W4 | - |  |
| runs/l40s-s5-wsd-enc8e4-4k | 0.460 | 0.460 | 0.000 | 0.000 | d14ce0120d1d | IT tournament S5 (encoder 8e-4, 4k) | NULL (flat) | KEEP-EVIDENCE | 2a | stock | W4 | - |  |
| runs/l40s-s6-wsd-enc16e4-2k | 0.460 | 0.460 | 0.000 | 0.000 | 8dd43942eac7 | IT tournament S6 (enc 1.6e-3 boundary probe) | FAILED (overshoot probe) | KEEP-EVIDENCE | 2a | stock | W4 | - |  |
| runs/plain-it-e2-full1e4 | 0.460 | 0.460 | 0.000 | 0.000 | f5ab05fbbcd2 | IT-E2 LR sweep (whole encoder 1e-4) | SUPERSEDED (sweep arm) | KEEP-EVIDENCE | 2a | stock | IT-E3 | - |  |
| runs/plain-it-tok-b2 | 0.460 | 0.460 | 0.000 | 0.000 | cd4ea4767b19 | IT Italian SPE on B2 | NULL/HARM (confounded negative) | KEEP-EVIDENCE | 2a | stock | - | - |  |
| runs/sg-it-s2v5b-c250-36k | 0.460 | 0.460 | 0.000 | 0.000 | ab84cd3be6a0 | IT stage-2 v5-B | FAILED (pre-registered latency FAIL) | KEEP-EVIDENCE | 2a | plain-it-c250-36k | RC-IT v4 | RIF |  |
| runs/B4-40h-e7 | 0.745 | 0.741 | 0.000 | 0.004 | 3196d6706349 | Canary-180M IT B4 40 h (lightweight track 2026-10-09) | SUPERSEDED (other model family, closed track) | KEEP-EVIDENCE | 2b | canary-180m-flash | - | LWR | the only Canary-180M .nemo on the Hub |
| runs/ada-it-w4-c250-16k | 0.460 | 0.460 | 0.000 | 0.000 | b13e4e7e8ad0 | IT-W4-C250-16k | SUPERSEDED (PARTIAL vs 36k); was the frozen scorer of the closed D-selection axis | KEEP-EVIDENCE | 2b | stock | C250-36k | RM15 | keep if the D-selection scorer must stay re-runnable |
| runs/dgx-ar-a1-r-s4000 | 0.462 | 0.460 | 0.000 | 0.002 | b981e33c69d9 | AR-A1-R | SUPERSEDED (GO-FULL, replaced by A2) | KEEP-EVIDENCE | 2b | stock | AR-A2-R | EVO,P:AR-A2-prereg,P:AR-D2-prereg,R24 |  |
| runs/dgx-ar-a2-r-s12670-seed1 | 0.464 | 0.460 | 0.000 | 0.004 | 0024c4ed9169 | AR-A2-R seed 1 | NULL (seed replicate) | KEEP-EVIDENCE | 2b | stock | AR-A2-R s0 | P:AR-SEED1-prereg | seed evidence lives in metrics / ar_eval |
| runs/dgx-cfs-s2-es | 0.463 | 0.460 | 0.000 | 0.003 | 3f50100b6ab2 | ES-S2 seed 0 (bucketing) | SUPERSEDED (STRONG, replaced by S5) | KEEP-EVIDENCE | 2b | stock | ES-S5 | S14 |  |
| runs/dgx-cfs-s2-s1-es | 0.463 | 0.460 | 0.000 | 0.003 | 5e51e0fdcf22 | ES-S2 seed 1 | SUPERSEDED (STRONG, replaced by S5) | KEEP-EVIDENCE | 2b | stock | ES-S5 | P:S5-prereg,S14 |  |
| runs/jp-fr-e1-w4-11881 | 0.460 | 0.460 | 0.000 | 0.000 | 3a2ebe311e92 | FR-E1 | SUPERSEDED (CLEAR, G1E chosen) | KEEP-EVIDENCE | 2b | stock | FR-G1E | RIF |  |
| runs/jp-fr-p1-w4-8k | 0.460 | 0.460 | 0.000 | 0.000 | 1ec3fec4e178 | FR-P1 | SUPERSEDED (LEVER -38, then E1/G1E) | KEEP-EVIDENCE | 2b | stock | FR-E1 / FR-G1E | RIF,RM15 | no metrics.json in the Hub dir (only final.nemo): keep the nemo or upload the metrics first |
| runs/l40s-it-w4-c250-8k | 0.460 | 0.460 | 0.000 | 0.000 | 7635d9cfc9be | IT-W4-C250-8k | SUPERSEDED (screen) | KEEP-EVIDENCE | 2b | stock | W4-C250-16k | RIF,RM15 |  |
| runs/l40s-s3-wsd-enc2x-4k | 0.460 | 0.460 | 0.000 | 0.000 | 5c10f0982768 | IT-S3-4k (origin of W4) | SUPERSEDED (milestone) | KEEP-EVIDENCE | 2b | stock | factory W4 | - |  |
| runs/plain-fr-p40 | 0.460 | 0.460 | 0.000 | 0.000 | e23b11e2459b | FR-0 old p40 | SUPERSEDED (harness-fr 65.85) | KEEP-EVIDENCE | 2b | stock | FR-P1 | FT,RIF,RM15 | parent of plain-fr-s2-eou: delete as a pair or not at all |
| runs/plain-fr-s2-eou | 0.460 | 0.460 | 0.000 | 0.000 | 5266c90f6302 | FR-0 stage 2 (s2-eou) | SUPERSEDED | KEEP-EVIDENCE | 2b | plain-fr-p40 | RC-FR v1 | FT,RIF,RM15 |  |
| runs/plain-it-b2-gradual | 0.460 | 0.460 | 0.000 | 0.000 | 995270c888d9 | IT-B2 plain | SUPERSEDED | KEEP-EVIDENCE | 2b | stock | IT-E3 | - | parent of plain-it-s2-b2 (EOU-IT v2): delete as a pair or not at all |
| runs/plain-it-c250 | 0.460 | 0.460 | 0.000 | 0.000 | 28cbfc954acf | IT-C250 14k | SUPERSEDED (by C250-36k) | KEEP-EVIDENCE | 2b | stock | C250-36k | EVO,RIF,RM12,RM15 |  |
| runs/plain-it-c250-20k | 0.460 | 0.460 | 0.000 | 0.000 | 031cfdaf11df | IT-C250-20k | SUPERSEDED (horizon ladder) | KEEP-EVIDENCE | 2b | stock | C250-36k | RM12 |  |
| runs/plain-it-c250-28k | 0.460 | 0.460 | 0.000 | 0.000 | 9129842ff85f | IT-C250-28k | SUPERSEDED (horizon ladder) | KEEP-EVIDENCE | 2b | stock | C250-36k | RM12 |  |
| runs/plain-it-c250-oldrep | 0.460 | 0.460 | 0.000 | 0.000 | b17362c12753 | IT-C250-oldrep | SUPERSEDED (exposure test, closed) | KEEP-EVIDENCE | 2b | stock | C250-36k | EVO,RM12 |  |
| runs/plain-it-e3-s8000 | 0.460 | 0.460 | 0.000 | 0.000 | 0090f15bf8d0 | IT-E3-8k | SUPERSEDED (horizon lever) | KEEP-EVIDENCE | 2b | stock | C-add25 / C250 | RM12 |  |
| runs/plain-it-e3r1-lr1e4 | 0.460 | 0.460 | 0.000 | 0.000 | 3e6009551ab9 | IT-E3 R1 re-warm | SUPERSEDED (best plain on 10-12) | KEEP-EVIDENCE | 2b | plain-it-e2-enc2e-4-s4000 | C250 line | RM12 |  |
| runs/plain-it-p1-add25 | 0.460 | 0.460 | 0.000 | 0.000 | 7ee0b8f21012 | IT-C-add25 | SUPERSEDED (by C250) | KEEP-EVIDENCE | 2b | stock | C250 | EVO,RM12 |  |
| runs/plain-it-s2-b2 | 0.460 | 0.460 | 0.000 | 0.000 | 63e5305aa127 | EOU-IT v2 (stage 2 on B2) | SUPERSEDED (by EOU-v3) | KEEP-EVIDENCE | 2b | plain-it-b2-gradual | EOU-v3 | MD |  |
| runs/ada-fr-h2-g1e-w4-23762 | 0.460 | 0.460 | 0.000 | 0.000 | d2062aca28d8 | FR-H2 plain parent | IMPORTANT (release parent) | KEEP-FULL | - | stock | - | EVO,P:S5-FR-prereg,RF22 | parent of v1.1; S5-FR baseline horizon reference |
| runs/ada-fr-s2v4-h2 | 0.460 | 0.460 | 0.000 | 0.000 | 3130534d577e | RC-FR v1.1 (FINAL-FR-v2 23.00) | RELEASE | KEEP-FULL | - | ada-fr-h2-g1e-w4-23762 | - | EVO,RF22 | same bytes as release/mynah-eou-fr-120m-v1.1/model.nemo |
| runs/dgx-ar-a2-r-s12670 | 0.464 | 0.460 | 0.000 | 0.004 | 11b8790e6db5 | AR-A2-R seed 0 | IMPORTANT (best AR model; primary band NULL) | KEEP-FULL | - | stock | - | P:AR-D2-prereg,P:AR-SEED1-prereg,R24 | best-current for Arabic |
| runs/dgx-cfs-d2-s1-es | 0.466 | 0.460 | 0.000 | 0.006 | 5e89541aa38c | ES-D2 seed 1 | CURRENT (FINAL-ES candidate; best DEV 14.81) | KEEP-FULL | - | stock | - | P:FINAL-ES-prereg,P:SUB15-ESD2-prereg,R26 |  |
| runs/dgx-cfs-h3-s1-es | 0.466 | 0.460 | 0.000 | 0.006 | 5b24cfe80aa4 | ES-H3 seed 1 | CURRENT (FINAL-ES candidate) | KEEP-FULL | - | stock | - | P:FINAL-ES-prereg,P:SUB15-ESD2-prereg,R26 |  |
| runs/dgx-cfs-s5-es | 0.463 | 0.460 | 0.000 | 0.003 | 583d6954a285 | ES-S5 seed 0 | CURRENT (preferred compute recipe; M1/M2 model) | KEEP-FULL | - | stock | ES-H3 / ES-D2 | ESQplan,P:ES-S2Q-prereg,P:SUB15-ESD2-prereg,P:SUB15-M1M2-prereg,R24,R26,S14,SUB15plan |  |
| runs/dgx-cfs-s5-s1-es | 0.463 | 0.460 | 0.000 | 0.003 | df7ae15e4193 | ES-S5 seed 1 | CURRENT (preferred compute recipe; M1/M2 model) | KEEP-FULL | - | stock | ES-H3 / ES-D2 | ESQplan,P:SUB15-M1M2-prereg,R26,S14,SUB15plan |  |
| runs/dgx-es-d2-23762 | 0.466 | 0.460 | 0.000 | 0.006 | fb886e09eb7a | ES-D2 seed 0, 246 h | CURRENT (FINAL-ES candidate; best DEV) | KEEP-FULL | - | stock | - | P:ES-S2Q-prereg,P:FINAL-ES-prereg,P:SUB15-ESD2-prereg,R26 |  |
| runs/dgx-es-h3-23762 | 0.466 | 0.460 | 0.000 | 0.006 | 1c28256c929d | ES-H3 seed 0 (SUB15) | CURRENT (FINAL-ES candidate; Stage-2 parent) | KEEP-FULL | - | stock | - | P:ES-S2Q-prereg,P:FINAL-ES-prereg,P:SUB15-ESD2-prereg,R26 | parent of dgx-es-s2v4-h3 |
| runs/dgx-es-s2v4-h3 | 0.460 | 0.460 | 0.000 | 0.000 | beeef283a31a | ES Stage-2 v4 on H3 (ES-S2Q) | CURRENT (MARGINAL; open preservation-line RCA) | KEEP-FULL | - | dgx-es-h3-23762 | - | P:ES-S2Q-prereg,R26 | needed by the Stage-2 RCA |
| runs/dgx-es1-w4-12670 | 0.920 | 0.460 | 0.460 | 0.001 | 172b1539f592 | ES-1 (172b1539) | IMPORTANT (factory reference) | KEEP-FULL | - | stock | ES-S5 / ES-H3 / ES-D2 | ESQplan,EVO,P:S5-prereg,P:SUB15-M1M2-prereg,R24,R26,RF22,S14,SUB15plan | + ES-1 Mynah pack (dd54fb20) |
| runs/dgx-s5fr-b-g1e-11881 | 0.460 | 0.460 | 0.000 | 0.000 | ef840fd4e074 | S5-FR arm B (same-GPU G1E baseline) | IMPORTANT (factory compute evidence) | KEEP-FULL | - | stock | - | P:S5-FR-prereg,R25 | paired with arm S5 |
| runs/dgx-s5fr-s5-g1e-11881 | 0.460 | 0.460 | 0.000 | 0.000 | 51db5578541b | S5-FR arm S5 | IMPORTANT (factory compute evidence, STRONG) | KEEP-FULL | - | stock | - | EVO,P:S5-FR-prereg,R25 | paired with arm B; FR S5 parent for a future Stage 2 |
| runs/jp-fr-g1e-w4-11881 | 0.460 | 0.460 | 0.000 | 0.000 | 1271deff6a83 | FR-G1E plain parent | IMPORTANT (release parent) | KEEP-FULL | - | stock | FR-H2 | AUT,EVO,P:S5-FR-prereg,RF22,RIF | parent of RC-FR v1 and FR-v2 stage 2 |
| runs/jp-fr-s2v4-g1e | 0.460 | 0.460 | 0.000 | 0.000 | 7814141d515c | RC-FR v1 = FR-S2 (FR release, FINAL-FR 27.76) | RELEASE | KEEP-FULL | - | jp-fr-g1e-w4-11881 | RC-FR v1.1 candidate | AUT,EVO,RF22,RIF | same bytes as release/mynah-eou-fr-120m/model.nemo |
| runs/plain-it-c250-36k | 0.460 | 0.460 | 0.000 | 0.000 | 7e9192811937 | IT-C250-36k plain parent | IMPORTANT (release parent) | KEEP-FULL | - | stock | - | EVO,RF22,RIF,RM12,RM15 | parent of RC-IT v4 and v5-B |
| runs/plain-it-e2-enc2e-4-s4000 | 1.839 | 0.460 | 1.379 | 0.000 | a74468820a14 | IT-E3 plain (whole encoder 2e-4) | IMPORTANT (parent of former production EOU-v3) | KEEP-FULL | - | stock | IT-E3-8k / C-line | BN,RM12 | last.ckpt = optimizer/trainer state of a CLOSED run |
| runs/plain-it-s2-e3-djlr | 0.460 | 0.460 | 0.000 | 0.000 | 57d04dc42443 | EOU-v3 (production until RC-IT) | IMPORTANT (former production anchor, RIF anchor 3) | KEEP-FULL | - | plain-it-e2-enc2e-4-s4000 | RC-IT v4 | BN,RIF,RM15 |  |
| runs/sg-it-s2v4-c250-36k | 0.460 | 0.460 | 0.000 | 0.000 | 0a4d1b103297 | RC-IT v4 (IT release, FINAL-IT 20.49) | RELEASE | KEEP-FULL | - | plain-it-c250-36k | - | EVO,RF22,RIF,RM15 | same bytes as release/mynah-eou-it-120m/model.nemo |

Notes on single rows:
- `runs/plain-it-e2-enc2e-4-s4000`: the `.nemo` is KEEP-FULL (parent of EOU-v3 and of R1 / the s2-e3 arms). Its
  `last.ckpt` (1.379 GB) is tier 1.
- `runs/jp-fr-p1-w4-8k` has **no `metrics.json` on the Hub** (only `final.nemo`). Its numbers live in RIF only.
  Before its `.nemo` can go, the metrics must be uploaded from wherever they survive; otherwise keep it.
- `runs/ada-it-w4-c250-16k` was the frozen scorer (`student_score.py`) of the closed D-selection axis (RM15 l.500).
  Deleting it means that axis cannot be re-scored bit-identically.
- Pairs: `plain-it-b2-gradual` + `plain-it-s2-b2` and `plain-fr-p40` + `plain-fr-s2-eou` are parent + child; delete
  both or neither.

## 6. Inventory: runs without weights (58 dirs, metrics / evals only)

All KEEP (the record of the run; together 0.06 GB). Smoke and regression runs are labelled.

| path | MB | experiment | status | parent (from metrics.json) |
|---|---|---|---|---|
| runs/A-5h | 4.0 | A-5h | CLOSED (metrics/evals only; no weights on Hub) | /root/ft/models/canary-180m-flash-it.nemo |
| runs/A-5h-e52 | 4.0 | A-5h-e52 | CLOSED (metrics/evals only; no weights on Hub) | /root/ft/models/canary-180m-flash-it.nemo |
| runs/B4-20h-e13 | 4.0 | B4-20h-e13 | CLOSED (metrics/evals only; no weights on Hub) | /root/ft/models/canary-180m-flash-it.nemo |
| runs/B4-20h-e13-rp20 | 7.0 | B4-20h-e13-rp20 | CLOSED (metrics/evals only; no weights on Hub) | /root/ft/models/canary-180m-flash-it.nemo |
| runs/B4-5h | 4.0 | B4-5h | CLOSED (metrics/evals only; no weights on Hub) | /root/ft/models/canary-180m-flash-it.nemo |
| runs/B4-5h-e52 | 4.0 | B4-5h-e52 | CLOSED (metrics/evals only; no weights on Hub) | /root/ft/models/canary-180m-flash-it.nemo |
| runs/eou-it-5h-e26-frz | 2.0 | eou-it-5h-e26-frz | CLOSED (metrics/evals only; no weights on Hub) | see metrics.json |
| runs/eou-it-5h-e26-frz-warm | 2.0 | eou-it-5h-e26-frz-warm | CLOSED (metrics/evals only; no weights on Hub) | see metrics.json |
| runs/eou-it-5h-e52 | 2.0 | eou-it-5h-e52 | CLOSED (metrics/evals only; no weights on Hub) | see metrics.json |
| runs/plain-a0-n32-frz | 0.0 | plain-a0-n32-frz | CLOSED (metrics/evals only; no weights on Hub) | see metrics.json |
| runs/plain-b1-5h-frz | 0.0 | plain-b1-5h-frz | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-b2-5h-top4 | 0.0 | plain-b2-5h-top4 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-base-p40 | 0.0 | plain-base-p40 | CLOSED (metrics/evals only; no weights on Hub) | plain-p40-b2pol |
| runs/plain-base-s2 | 0.0 | plain-base-s2 | CLOSED (metrics/evals only; no weights on Hub) | plain-s2-eou-p40-b2pol |
| runs/plain-fr-e1-w4-11881 | 0.0 | plain-fr-e1-w4-11881 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-fr-g1e-w4-11881 | 0.0 | plain-fr-g1e-w4-11881 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-fr-s2v4-g1e | 0.0 | plain-fr-s2v4-g1e | CLOSED (metrics/evals only; no weights on Hub) | plain-fr-g1e-w4-11881 |
| runs/plain-it-a0-mix | 0.0 | plain-it-a0-mix | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-a1-gain10 | 0.0 | plain-it-a1-gain10 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-a2-vpt50 | 0.0 | plain-it-a2-vpt50 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-b1-top8 | 0.0 | plain-it-b1-top8 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-c1-12k | 0.0 | plain-it-c1-12k | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-c1-yg15 | 0.0 | plain-it-c1-yg15 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-c2-ygrand | 0.0 | plain-it-c2-ygrand | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-e1a-decjoint | 0.0 | plain-it-e1a-decjoint | CLOSED (metrics/evals only; no weights on Hub) | plain-it-b2-gradual |
| runs/plain-it-e1b-all1e4 | 0.0 | plain-it-e1b-all1e4 | CLOSED (metrics/evals only; no weights on Hub) | plain-it-b2-gradual |
| runs/plain-it-e2-enc2e-4-s1000 | 0.0 | plain-it-e2-enc2e-4-s1000 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-e2-enc3e-5-s1000 | 0.0 | plain-it-e2-enc3e-5-s1000 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-e3g-s8000 | 0.0 | plain-it-e3g-s8000 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-e3g-yg15 | 0.0 | plain-it-e3g-yg15 | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-p2-tl | 0.0 | plain-it-p2-tl | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-it-s2-b2-full | 0.0 | plain-it-s2-b2-full | CLOSED (metrics/evals only; no weights on Hub) | plain-it-b2-gradual |
| runs/plain-it-s2-e3 | 0.0 | plain-it-s2-e3 | CLOSED (metrics/evals only; no weights on Hub) | plain-it-e2-enc2e-4-s4000 |
| runs/plain-it-s2-e3-pre | 0.0 | plain-it-s2-e3-pre | CLOSED (metrics/evals only; no weights on Hub) | plain-it-e2-enc2e-4-s4000 |
| runs/plain-it-tok-b2-step0 | 0.0 | plain-it-tok-b2-step0 | SMOKE (metrics only) | stock |
| runs/plain-p40-b2pol | 0.0 | plain-p40-b2pol | CLOSED (metrics/evals only; no weights on Hub) | stock |
| runs/plain-reg-new | 0.0 | plain-reg-new | SMOKE (metrics only) | stock |
| runs/plain-reg-ref | 0.0 | plain-reg-ref | SMOKE (metrics only) | stock |
| runs/plain-s2-eou-p40-b2pol | 0.0 | plain-s2-eou-p40-b2pol | CLOSED (metrics/evals only; no weights on Hub) | plain-p40-b2pol |
| runs/plain-s2b-eou-fe03 | 0.0 | plain-s2b-eou-fe03 | CLOSED (metrics/evals only; no weights on Hub) | plain-p40-b2pol |
| runs/plain-smoke-fr-e1-w4-11881 | 0.0 | plain-smoke-fr-e1-w4-11881 | SMOKE (metrics only) | stock |
| runs/plain-smoke-fr-g1e-w4-11881 | 0.0 | plain-smoke-fr-g1e-w4-11881 | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-b2-gradual | 0.0 | plain-smoke-it-b2-gradual | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-c1-yg15 | 0.0 | plain-smoke-it-c1-yg15 | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-c250 | 0.0 | plain-smoke-it-c250 | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-c250-20k | 0.0 | plain-smoke-it-c250-20k | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-c250-28k | 0.0 | plain-smoke-it-c250-28k | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-c250-36k | 0.0 | plain-smoke-it-c250-36k | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-c250-oldrep | 0.0 | plain-smoke-it-c250-oldrep | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-e2-full1e4 | 0.0 | plain-smoke-it-e2-full1e4 | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-e3g-s8000 | 0.0 | plain-smoke-it-e3g-s8000 | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-e3g-yg15 | 0.0 | plain-smoke-it-e3g-yg15 | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-p1-add25 | 0.0 | plain-smoke-it-p1-add25 | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-p2-tl | 0.0 | plain-smoke-it-p2-tl | SMOKE (metrics only) | stock |
| runs/plain-smoke-it-tok-b2 | 0.0 | plain-smoke-it-tok-b2 | SMOKE (metrics only) | stock |
| runs/plain-smoke-mix | 0.0 | plain-smoke-mix | SMOKE (metrics only) | stock |
| runs/plain-stock-step0 | 0.0 | plain-stock-step0 | SMOKE (metrics only) | stock |
| runs/plain-vpchk-b2-tail | 0.0 | plain-vpchk-b2-tail | CLOSED (metrics/evals only; no weights on Hub) | plain-it-b2-gradual |

## 7. Inventory: everything outside `runs/`

| path | files | total GB | .nemo | ckpt | safetensors | tar/tgz | bundles | small | class |
|---|---|---|---|---|---|---|---|---|---|
| data/snapshots | 83 | 38.699 | - | - | - | 38.699 | - | 0.000 | KEEP-FULL (release training data, only copy) |
| release/mynah-eou-fr-120m | 23 | 0.920 | 0.460 | - | 0.460 | - | - | 0.000 | KEEP-FULL (RELEASE) |
| release/mynah-eou-fr-120m-v1.1 | 26 | 0.920 | 0.460 | - | 0.460 | - | - | 0.000 | KEEP-FULL (RELEASE candidate) |
| release/mynah-eou-it-120m | 21 | 0.920 | 0.460 | - | 0.460 | - | - | 0.000 | KEEP-FULL (RELEASE) |
| data/final-fr | 7 | 0.831 | - | - | - | 0.829 | - | 0.002 | KEEP-FULL (sealed bank) |
| data/final-it | 5 | 0.762 | - | - | - | 0.762 | - | 0.000 | KEEP-FULL (sealed bank) |
| data/final-it-wip | 1 | 0.699 | - | - | - | 0.699 | - | - | DELETE-CANDIDATE tier 1 (subset of the sealed bank) |
| data/final-fr-v2 | 5 | 0.567 | - | - | - | 0.567 | - | 0.000 | KEEP-FULL (sealed bank) |
| rc/it-v4 | 4 | 0.460 | - | - | 0.460 | - | - | 0.000 | KEEP-FULL (RC-IT Mynah pack, cited by provenance) |
| rc/fr-s2v4 | 4 | 0.460 | - | - | 0.460 | - | - | 0.000 | KEEP-FULL (RC-FR Mynah pack, cited by provenance) |
| results | 10 | 0.416 | - | - | - | 0.416 | - | - | KEEP-FULL (record) |
| results/jp-2026-10-19 | 212 | 0.285 | - | - | - | - | - | 0.285 | KEEP-FULL (record) |
| data/ood-fr1 | 10 | 0.182 | - | - | - | 0.180 | - | 0.001 | KEEP-FULL (sealed bank) |
| results/l40s-jp-2026-10-17 | 87 | 0.155 | - | - | - | - | - | 0.155 | KEEP-FULL (record) |
| results/dgx-2026-10-24 | 48 | 0.104 | - | - | - | - | 0.102 | 0.001 | KEEP-FULL (record) |
| results/dgx-2026-10-23 | 345 | 0.068 | - | - | - | - | 0.002 | 0.066 | KEEP-FULL (record) |
| data/eou-fr-v2-dev | 5 | 0.056 | - | - | - | 0.056 | - | 0.000 | KEEP-FULL (sealed bank) |
| results/dgx-2026-10-26 | 36 | 0.053 | - | - | - | 0.001 | 0.052 | 0.000 | KEEP-FULL (record) |
| data/eou-fr-v2-final | 4 | 0.051 | - | - | - | 0.051 | - | 0.000 | KEEP-FULL (sealed bank) |
| tokenizers | 36 | 0.041 | - | - | - | - | - | 0.041 | KEEP-FULL (record) |
| data/it | 4 | 0.041 | - | - | - | - | - | 0.041 | KEEP-FULL (record) |
| data/a2-fr | 13 | 0.034 | - | - | - | - | - | 0.034 | KEEP-FULL (record) |
| results/dgx-2026-10-25 | 10 | 0.026 | - | - | - | - | 0.026 | 0.000 | KEEP-FULL (record) |
| results/sg-2026-10-18 | 82 | 0.024 | - | - | - | - | - | 0.024 | KEEP-FULL (record) |
| results/dgx-2026-10-15 | 52 | 0.020 | - | - | - | - | - | 0.020 | KEEP-FULL (record) |
| results/l40s-2026-10-21 | 168 | 0.017 | - | - | - | - | - | 0.017 | KEEP-FULL (record) |
| logs | 174 | 0.015 | - | - | - | - | - | 0.015 | KEEP-FULL (record) |
| results/ada-2026-10-22 | 197 | 0.014 | - | - | - | - | - | 0.014 | KEEP-FULL (record) |
| results/l40s-2026-10-15 | 44 | 0.011 | - | - | - | - | - | 0.011 | KEEP-FULL (record) |
| results/probe | 8 | 0.004 | - | - | - | - | - | 0.004 | KEEP-FULL (record) |
| results/ada-ee-2026-10-16 | 14 | 0.002 | - | - | - | - | - | 0.002 | KEEP-FULL (record) |
| recipe | 140 | 0.001 | - | - | - | - | - | 0.001 | KEEP-FULL (record) |
| results/jp-2026-10-20 | 38 | 0.000 | - | - | - | - | - | 0.000 | KEEP-FULL (record) |
| results/l40s-jp-2026-10-15 | 13 | 0.000 | - | - | - | - | - | 0.000 | KEEP-FULL (record) |
| (root files) | 2 | 0.000 | - | - | - | - | - | 0.000 | KEEP-FULL (record) |
| data | 1 | 0.000 | - | - | - | - | - | 0.000 | KEEP-FULL (record) |

`results` (root files) = the survival tarballs `results/survival-*.tgz` (10 files, 0.42 GB): box-result bundles of
2026-10-10..20, the only copy of some early logs. KEEP.

## 8. Tiered reclaim plan (billed GB; every item verified to have no other path with the same sha256)

Baseline: **72.08 GB** billed. Each tier reclaims space **only after a history squash** (section 9).

### Tier 1: unequivocally safe (2 files, 2.078 GB) -> projected 70.00 GB

| path | GB | sha256 (16) | justification |
|---|---|---|---|
| `runs/plain-it-e2-enc2e-4-s4000/last.ckpt` | 1.379 | 9c41098daecd5f7c | optimizer / trainer state of a run CLOSED on 2026-10-11 (IT-E3). The `final.nemo` (kept) is the model; nothing resumes from this state (every later IT run starts from stock or from a `final.nemo`); the only `.ckpt` left on the Hub |
| `data/final-it-wip/final-it-wip-2026-10-18.tgz` | 0.699 | a9fc209fec8551b6 | temporary transfer bundle of the FINAL-IT bank before sealing. Verified on dgx2 to be a strict subset of `data/final-it/final-it-sealed-2026-10-19.tgz` (0 missing members, 0 size mismatches, manifests byte-identical). RIF l.41 already deleted the same WIP bundle on the box as a duplicate |

Not proposed although they look like duplicates: the 8 byte-duplicate paths of section 3 (0 billed GB; cited by
provenance / READMEs), the 20 smoke `metrics.json` dirs (a few kB; the gate record).

Tier 1 alone lands at **70.0 GB**: inside the 60-70 GB target only at its upper edge.

### Tier 2a: needs user judgement, low value (16 `.nemo`, 7.361 GB) -> projected 62.64 GB

CLOSED dead ends and screens: FAILED / NULL / HARM or 2k-step tournament arms whose result is fully recorded in
`metrics.json` + reports, that are no parent of a kept model and that no open comparison needs. Their dirs keep every
small file (KEEP-EVIDENCE).

| path | GB | sha256 (12) | status | why it can go |
|---|---|---|---|---|
| `runs/l40s-s0-cos2k/final.nemo` | 0.460 | 696c3ac5c746 | IT tournament control, 2k | screen; recipe frozen as W4 (RM15) |
| `runs/l40s-s1-wsd2k/final.nemo` | 0.460 | 415cd21d0066 | IT tournament, 2k | screen |
| `runs/l40s-s2-enc2x2k/final.nemo` | 0.460 | 668c959085ad | IT tournament, 2k | screen |
| `runs/l40s-s3-wsd-enc2x2k/final.nemo` | 0.460 | e6c7569fd55b | IT tournament, 2k | screen; its 4k version is kept in tier 2b |
| `runs/l40s-s4-wsd-enc8e4-2k/final.nemo` | 0.460 | b92b5c80e0dc | IT tournament, 2k | screen |
| `runs/l40s-s5-wsd-enc8e4-4k/final.nemo` | 0.460 | d14ce0120d1d | IT encoder 8e-4, NULL (flat) | not adopted |
| `runs/l40s-s6-wsd-enc16e4-2k/final.nemo` | 0.460 | 8dd43942eac7 | IT LR boundary probe, overshoot | pre-registered as "not a promotion candidate" |
| `runs/dgx-r0-ctl-1k/final.nemo` | 0.460 | a5f5311003a6 | DGX lab control | lab axis closed (RM15 l.143) |
| `runs/dgx-r2-ctc01-1k/final.nemo` | 0.460 | ec2dfc054103 | CTC 0.1, NULL | closed lever |
| `runs/plain-it-e2-full1e4/final.nemo` | 0.460 | f5ab05fbbcd2 | IT LR-sweep arm | superseded by E3 (BN) |
| `runs/plain-it-tok-b2/final.nemo` | 0.460 | cd4ea4767b19 | Italian SPE, negative (confounded) | closed; a re-test would be on the factory recipe, not on this model |
| `runs/sg-it-s2v5b-c250-36k/final.nemo` | 0.460 | ab84cd3be6a0 | stage-2 v5-B, pre-registered FAIL (latency) | IT EOU axis closed (RIF); sha verified in RIF for reproduction only |
| `runs/jp-fr-g1-w4-8k/final.nemo` | 0.460 | de457d0b6501 | FR-G1, HARM +1.75 | closed; G1E chosen |
| `runs/l40s-fr-s2v2-ps/final.nemo` | 0.460 | 0817c2f1da6a | FR-v2 stage 2 pause stretch, FAIL | closed (AUT); the parent G1E is kept |
| `runs/dgx-ar-d2-r-s12670/final.nemo` | 0.460 | 069c1ee05b19 | AR-D2 s0, HARM | HARM REPLICATED; A2 s0 is the kept AR model |
| `runs/dgx-ar-d2-r-s12670-seed1/final.nemo` | 0.460 | c53209a177ca | AR-D2 s1, HARM REPLICATED | as above |

Tier 1 + 2a: **62.64 GB**, inside the target.

### Tier 2b: needs user judgement, superseded milestones (21 `.nemo`, 9.943 GB) -> projected 52.70 GB

Superseded but historically meaningful lineage checkpoints. Recommended only if more headroom is wanted; KEEP by
default.

| path | GB | sha256 (12) | status | caveat |
|---|---|---|---|---|
| `runs/B4-40h-e7/final.nemo` | 0.741 | 3196d6706349 | Canary-180M IT (closed 2026-10-09 track) | the only Canary-180M weights on the Hub |
| `runs/plain-it-b2-gradual/final.nemo` | 0.460 | 995270c888d9 | IT-B2 | delete together with its child below |
| `runs/plain-it-s2-b2/final.nemo` | 0.460 | 63e5305aa127 | EOU-IT v2 | superseded by EOU-v3 (kept) |
| `runs/plain-it-e3-s8000/final.nemo` | 0.460 | 0090f15bf8d0 | IT-E3-8k | |
| `runs/plain-it-e3r1-lr1e4/final.nemo` | 0.460 | 3e6009551ab9 | IT-E3 R1 (best plain on 2026-10-12) | |
| `runs/plain-it-p1-add25/final.nemo` | 0.460 | 7ee0b8f21012 | IT-C-add25 | |
| `runs/plain-it-c250/final.nemo` | 0.460 | 28cbfc954acf | IT-C250 14k | |
| `runs/plain-it-c250-oldrep/final.nemo` | 0.460 | b17362c12753 | IT-C250-oldrep | |
| `runs/plain-it-c250-20k/final.nemo` | 0.460 | 031cfdaf11df | horizon ladder | |
| `runs/plain-it-c250-28k/final.nemo` | 0.460 | 9129842ff85f | horizon ladder | |
| `runs/l40s-it-w4-c250-8k/final.nemo` | 0.460 | 7635d9cfc9be | IT-W4-C250-8k | |
| `runs/ada-it-w4-c250-16k/final.nemo` | 0.460 | b13e4e7e8ad0 | IT-W4-C250-16k, PARTIAL | frozen scorer of the closed D-selection axis |
| `runs/l40s-s3-wsd-enc2x-4k/final.nemo` | 0.460 | 5c10f0982768 | IT-S3-4k = origin of W4 | factory milestone |
| `runs/plain-fr-p40/final.nemo` | 0.460 | e23b11e2459b | FR-0 p40 | delete together with its child below |
| `runs/plain-fr-s2-eou/final.nemo` | 0.460 | 5266c90f6302 | FR-0 stage 2 | |
| `runs/jp-fr-p1-w4-8k/final.nemo` | 0.460 | 1ec3fec4e178 | FR-P1 | **no metrics.json on the Hub**: upload the metrics first, or keep |
| `runs/jp-fr-e1-w4-11881/final.nemo` | 0.460 | 3a2ebe311e92 | FR-E1 (CLEAR, then G1E chosen) | the no-web-source FR comparator |
| `runs/dgx-cfs-s2-es/final.nemo` | 0.460 | 3f50100b6ab2 | ES-S2 s0 (STRONG, replaced by S5) | ES compute evidence (S14) |
| `runs/dgx-cfs-s2-s1-es/final.nemo` | 0.460 | 5e51e0fdcf22 | ES-S2 s1 | as above |
| `runs/dgx-ar-a1-r-s4000/final.nemo` | 0.460 | b981e33c69d9 | AR-A1-R (GO-FULL, replaced by A2) | |
| `runs/dgx-ar-a2-r-s12670-seed1/final.nemo` | 0.460 | 0024c4ed9169 | AR-A2 s1 (seed replicate) | A2 s0 kept |

### Never candidates (KEEP-FULL weights, 19 unique `.nemo` + 4 unique Mynah packs + data)

RC-IT v4 0a4d1b10, C250-36k 7e919281, EOU-v3 57d04dc4 and its parent a7446882, RC-FR v1 7814141d, FR-G1E 1271deff,
RC-FR v1.1 3130534d, FR-H2 d2062aca, ES-1 172b1539 (+ pack dd54fb20), ES-S5 583d6954 / df7ae15e, ES-H3
1c28256c / 5b24cfe8, ES-D2 fb886e09 / 5e89541a, ES Stage-2 H3 beeef283, S5-FR ef840fd4 / 51db5578, AR-A2 s0
11b8790e; the Mynah packs 2aa4783d, 9e0e6b50, 28e3f0f5; the snapshots (38.70 GB) and the sealed banks (2.45 GB).
Floor if every tier is applied: 52.70 GB billed (COMPUTED); that floor is all KEEP-FULL plus the small files.

### Summary

| step | reclaim GB | projected billed GB |
|---|---|---|
| now | - | 72.08 |
| tier 1 + squash | 2.08 | **70.00** |
| + tier 2a + squash | 7.36 | **62.64** |
| + tier 2b + squash | 9.94 | 52.70 |

The data snapshots (38.70 GB, 54 % of the bill) are the single biggest item and stay KEEP-FULL: they are the only
copy of the IT and FR release training data, and `provenance.json` of every release points at them. If the quota ever
forces it, the honest option is to rebuild them from the public sources with `data_snapshot.py` manifests (UNKNOWN
effort; Granary teacher outputs are in `meta.tgz`), not to delete them silently.

## 9. History squash: needed to actually reclaim space

- OBSERVED: `used_storage` == the unique LFS bytes of the current tree. The previous squash (2026-10-20, commit
  6829b2c079 "Squash history: keep only the current files") left no orphaned objects, and no file was deleted since.
- Therefore **any delete commit from now on reclaims 0 bytes until `super_squash_history` runs again**: the deleted
  objects stay referenced by the 77 existing commits.
- How it was done before: `~/mynah-asr-ft/finetune/common/hf_prune.py --repo <repo> --delete '<glob>' … [--squash]
  [--dry-run] [--token-file …]`. It lists the matched paths (fnmatch globs) with sizes, makes ONE
  `CommitOperationDelete` commit, then, with `--squash`, calls `api.super_squash_history(repo)`; the repo keeps only
  its current files. Irreversible for the history; the current files stay.
- Caveats for the next squash (report only):
  1. Every commit sha changes. Anything that pins a Hub **revision** (none found in `provenance.json`; they cite
     paths) would break. Paths stay valid.
  2. `hf_prune.py` uses `--delete` globs: pass the exact paths of section 8, never a broad glob like `runs/*/*.nemo`.
     Run `--dry-run` first and diff its list against this document.
  3. Squash while no upload is running (a box archiving at the same time would race the squash).
  4. The bill may take a while to refresh after the squash (UNKNOWN delay); verify with `repo_info().used_storage`.
  5. `hf_prune.py` defaults to `--token-file ~/.hf_token`; on the DGX boxes the write token is `~/.hf_token.rw`.

## 10. Checkpoint hygiene going forward

What to upload per run (by state):

| state | upload | never upload |
|---|---|---|
| RUNNING | nothing heavy; optionally `metrics.json` snapshots | `last.ckpt` unless a resume across boxes is planned (then delete it at close) |
| CLOSED, model not needed | `metrics.json`, `run.json` / `launch.txt` (git head, prereg commit, cmdline), logs, evals / hyps, gate outputs, `final.nemo.sha256`, data stats / SHA256SUMS. The `.nemo` only if it is a parent, a release / FINAL candidate or best-current | optimizer / trainer state (`last.ckpt`, `*.distcp`), intermediate step checkpoints, temporary exports, WIP banks once sealed |
| CLOSED, model needed (parent / RC / FINAL / best-current) | the above + `final.nemo` + its `.sha256` | a second copy under another path unless it is byte-identical (dedup is free, but say which path is canonical) |
| RELEASE | the package (`release/<name>/`: model.nemo, Mynah pack, provenance, SHA256SUMS, results) | re-encoded tarballs of data already in `data/snapshots` |
| smoke | `metrics.json` only | weights |

Rules:
- One canonical path per checkpoint: `runs/<box>-<tag>/final.nemo`. Releases may hold a byte-identical copy (free).
- Before uploading a WIP / transfer bundle, ask whether its sealed version will supersede it; if yes, keep it on the
  box only.
- Record the sha256 at close (`final.nemo.sha256`), and add a line to `ARCHIVE_INDEX.md` (draft:
  `.work/hf-archive-index-draft-2026-10-27.md`) with: experiment, status, checkpoint sha256, canonical path, size,
  retention class, dependencies (parent) / children, cited by, successor.
- At each pruning: one `hf_prune.py --dry-run`, the user's approval of the exact list, then the delete commit plus
  `--squash`, then a fresh `used_storage` check and an updated index.

## Appendix A: full sha256 of every heavy weight file

| path | bytes | LFS sha256 |
|---|---|---|
| rc/fr-s2v4/model.safetensors | 459487524 | `9e0e6b50a20f3e4f655c5916092bf523ce5730317da89fdae7b3f5bba73dbfa8` |
| rc/it-v4/mynah-pack/model.safetensors | 459487524 | `2aa4783d566880a67e8694cd39ba4eeb84b8dc724ea047f2f82d5c78da3df79f` |
| release/mynah-eou-fr-120m-v1.1/model.nemo | 460072960 | `3130534d577e340df129fea9074a63a070aaa05292bcf3aadd96b4aa39fd969c` |
| release/mynah-eou-fr-120m-v1.1/mynah/model.safetensors | 459487524 | `28e3f0f54a9d1c2a48278b08a5dea7d7b4a63abb03b42d6f808c0bce8b8daf05` |
| release/mynah-eou-fr-120m/model.nemo | 460072960 | `7814141d515ca347004a862435fc5029d553d885252c8ac5e5b23ce793d48105` |
| release/mynah-eou-fr-120m/mynah/model.safetensors | 459487524 | `9e0e6b50a20f3e4f655c5916092bf523ce5730317da89fdae7b3f5bba73dbfa8` |
| release/mynah-eou-it-120m/model.nemo | 460072960 | `0a4d1b103297e169af74b6673ff97864747bb2c54b70a7ec91253d9583c003ff` |
| release/mynah-eou-it-120m/mynah/model.safetensors | 459487524 | `2aa4783d566880a67e8694cd39ba4eeb84b8dc724ea047f2f82d5c78da3df79f` |
| runs/B4-40h-e7/final.nemo | 741140480 | `3196d670634911ebe5c1f71235d9c80eb63591a7419e6e9e77ec1a2e5e744b85` |
| runs/ada-fr-h2-g1e-w4-23762/final.nemo | 460072960 | `d2062aca28d8db0abff823fe4be8051665bdb4c7d3d15cc8ff929f32cf8bfda6` |
| runs/ada-fr-s2v4-h2/final.nemo | 460072960 | `3130534d577e340df129fea9074a63a070aaa05292bcf3aadd96b4aa39fd969c` |
| runs/ada-it-w4-c250-16k/final.nemo | 460072960 | `b13e4e7e8ad0e44accf8eff6dfb41e5afa9a5fd491b246e27047356a3f7b2073` |
| runs/dgx-ar-a1-r-s4000/final.nemo | 460072960 | `b981e33c69d94c95c08fb9150f9f1b023ff50610f79ff48f6d305c6a186d8c33` |
| runs/dgx-ar-a2-r-s12670-seed1/final.nemo | 460072960 | `0024c4ed9169f238642bbc3ff00bf4511321037896a324e41bdd1641d7002232` |
| runs/dgx-ar-a2-r-s12670/final.nemo | 460072960 | `11b8790e6db5a3cd18e2e22a1bc4f4d0efaa960ddf55465d0751edd3f86ee290` |
| runs/dgx-ar-d2-r-s12670-seed1/final.nemo | 460072960 | `c53209a177ca22346c84158d491aabbb85272e4b7a41973553c1920dfbcef1c3` |
| runs/dgx-ar-d2-r-s12670/final.nemo | 460072960 | `069c1ee05b1918bb028296e2c25c1501c59f250d0716bc6645c1da380ca8a281` |
| runs/dgx-cfs-d2-s1-es/final.nemo | 460072960 | `5e89541aa38c55fb9f326137460ef09f29ac983f24f7caeecc420a819f4b61d4` |
| runs/dgx-cfs-h3-s1-es/final.nemo | 460072960 | `5b24cfe80aa46f322da552f6c097ad484ca96cadc0765ff62098209127a227f7` |
| runs/dgx-cfs-s2-es/final.nemo | 460072960 | `3f50100b6ab23e17ccdc9d28d770e8b0a31aec6e8cd870bfb59bebe4929de97c` |
| runs/dgx-cfs-s2-s1-es/final.nemo | 460072960 | `5e51e0fdcf22caee62abbc6e83f131216dc710997f97f928c71e9d601f684d80` |
| runs/dgx-cfs-s5-es/final.nemo | 460072960 | `583d6954a285d85194029e7fd93c9957b92b9a8b9058ca81b47e59b7a74683c2` |
| runs/dgx-cfs-s5-s1-es/final.nemo | 460072960 | `df7ae15e4193321449defd149695b20bfe474601b727a82f20f84500535d8919` |
| runs/dgx-es-d2-23762/final.nemo | 460072960 | `fb886e09eb7a34182b95457c07f0b33486658c885eff807419e2ee8d946a7218` |
| runs/dgx-es-h3-23762/final.nemo | 460072960 | `1c28256c929d6b5fc5685ba854488ad69ea801b61b95d7ebb1414720c18ea900` |
| runs/dgx-es-s2v4-h3/final.nemo | 460072960 | `beeef283a31a5d68b4d7b44c068b88b5f7e4f8022ef925a91fe3da392619da77` |
| runs/dgx-es1-w4-12670/final.nemo | 460072960 | `172b1539f5927f7f1b08cebb3f5c02a3011f89ed368a4cb5b5525dca58444638` |
| runs/dgx-es1-w4-12670/mynah-pack/model.safetensors | 459487524 | `dd54fb207d927745c163873f4f771bbe10b4ea974511ff7656ccb7543836e5bf` |
| runs/dgx-r0-ctl-1k/final.nemo | 460072960 | `a5f5311003a6f914ee8a38c6bfffd3197aac8eed2cfc048ff65ec0ef11b2f8bc` |
| runs/dgx-r2-ctc01-1k/final.nemo | 460072960 | `ec2dfc0541036171c42d80fab27f4062d7a72ce27b885686d7bdd7b7cd4d78ce` |
| runs/dgx-s5fr-b-g1e-11881/final.nemo | 460072960 | `ef840fd4e074731e6bf9e0824e6e74e9ab0b66c96a8be78d49ba70f9f5426743` |
| runs/dgx-s5fr-s5-g1e-11881/final.nemo | 460072960 | `51db5578541b8e2f681dec5399ee27d5b29e2238501d16d55e5a0da01f21e467` |
| runs/jp-fr-e1-w4-11881/final.nemo | 460072960 | `3a2ebe311e9293cff4dca5df3a8debb55728323919d9eec3ec95ce3a1bec3858` |
| runs/jp-fr-g1-w4-8k/final.nemo | 460072960 | `de457d0b6501e7f4bf2332cf91cf3fc0683891e482a18d2885b1af3d23a71982` |
| runs/jp-fr-g1e-w4-11881/final.nemo | 460072960 | `1271deff6a833805b5dc9fb021fba43c058782e686842a49718f4e362b030973` |
| runs/jp-fr-p1-w4-8k/final.nemo | 460072960 | `1ec3fec4e1784e668638d0526fdd51141ff4678ddc2abd56a5ed6dcd1513880c` |
| runs/jp-fr-s2v4-g1e/final.nemo | 460072960 | `7814141d515ca347004a862435fc5029d553d885252c8ac5e5b23ce793d48105` |
| runs/l40s-fr-s2v2-ps/final.nemo | 460072960 | `0817c2f1da6a4aec7260aab23024a77ccb2cb5ece3670afe70dbc7835c793085` |
| runs/l40s-it-w4-c250-8k/final.nemo | 460072960 | `7635d9cfc9bee3543e25bcce35dcf71592e2f3b4c8af47108cd1d8c8b4233ec1` |
| runs/l40s-s0-cos2k/final.nemo | 460072960 | `696c3ac5c7463a9c6459ac060677f216b88b37e2a905e5f258c8ddbde681eb37` |
| runs/l40s-s1-wsd2k/final.nemo | 460072960 | `415cd21d00662f69994d68be83aac004013e6a8c6fa67f83e0509d89ede34e54` |
| runs/l40s-s2-enc2x2k/final.nemo | 460072960 | `668c959085ad453729656c77af9c4d3b6273f18f6e718c3510082292b3b7de60` |
| runs/l40s-s3-wsd-enc2x-4k/final.nemo | 460072960 | `5c10f0982768d2246af4b9260d856c11f06ebb68e94b74dd9e3ba6933f45c6a2` |
| runs/l40s-s3-wsd-enc2x2k/final.nemo | 460072960 | `e6c7569fd55bc68cb4562966f53fbe50f071715d8ff88291d7c6b5045b4dc2ae` |
| runs/l40s-s4-wsd-enc8e4-2k/final.nemo | 460072960 | `b92b5c80e0dc0f285edc0addead9f455a9e522aa4ad4f270e94449ae78f46006` |
| runs/l40s-s5-wsd-enc8e4-4k/final.nemo | 460072960 | `d14ce0120d1d3d12d2382dc761c3121731e30d9f737a538cb52a1f80cec4fca2` |
| runs/l40s-s6-wsd-enc16e4-2k/final.nemo | 460072960 | `8dd43942eac7647051b0a6a82253e513422a72fe726ae37479e5bd20e0029069` |
| runs/plain-fr-p40/final.nemo | 460072960 | `e23b11e2459bc042b228bf0222fb7b2b9c73ec8b8db71ab067afa802a591a6de` |
| runs/plain-fr-s2-eou/final.nemo | 460072960 | `5266c90f6302922f9ec04ed454f93ad9405ecfd7bab4e4bd3ef8797ac67ad054` |
| runs/plain-it-b2-gradual/final.nemo | 460072960 | `995270c888d91e49d043af1a50201eeea1728477f496099ebdf96cd25ba0229d` |
| runs/plain-it-c250-20k/final.nemo | 460072960 | `031cfdaf11df9ba50534e45e4f05eae55935ae9d65b06490ac022d4a0fb0d2a4` |
| runs/plain-it-c250-28k/final.nemo | 460072960 | `9129842ff85f513e340fad770c364ae7d7d7a7a9b93029cb2067038cdc9eed73` |
| runs/plain-it-c250-36k/final.nemo | 460072960 | `7e919281193735210f46a37985263e858b31a2d92621ac4378efd5cfb243a798` |
| runs/plain-it-c250-oldrep/final.nemo | 460072960 | `b17362c12753473f0b4d9a12daea69f0f1a560521e37ae773ac20dcda480253d` |
| runs/plain-it-c250/final.nemo | 460072960 | `28cbfc954acfc44db03495b2ef1a72395b08f410d9be055423d83db085c9873f` |
| runs/plain-it-e2-enc2e-4-s4000/final.nemo | 460072960 | `a74468820a14b0744c7df379bbaea0264c66f6ef34b2e212809db992f9a1a134` |
| runs/plain-it-e2-enc2e-4-s4000/last.ckpt | 1378928761 | `9c41098daecd5f7cbd37c534689e2a1e2235c45c14ad11ff090ecfe52f9713a3` |
| runs/plain-it-e2-full1e4/final.nemo | 460072960 | `f5ab05fbbcd2852fbf57a764b9231495acfe634c3fe67e543e2fd527341caf00` |
| runs/plain-it-e3-s8000/final.nemo | 460072960 | `0090f15bf8d0af49bc330bc1b577d64b15f8cecce5fdd29f9c82d2fe37d875f7` |
| runs/plain-it-e3r1-lr1e4/final.nemo | 460072960 | `3e6009551ab9715d76814acf056b5bfce17f47036bd7a715e4deafaad7336ba0` |
| runs/plain-it-p1-add25/final.nemo | 460072960 | `7ee0b8f2101264d03e203e445e981827ca2f5876141646f6db60ebb1477fe643` |
| runs/plain-it-s2-b2/final.nemo | 460072960 | `63e5305aa1272fe9fc6cbdd99a7137796f0a329aca3a65ed1c415bddd39dfeb0` |
| runs/plain-it-s2-e3-djlr/final.nemo | 460072960 | `57d04dc4244322a57b4260c04354133dc80862c7f8c8652df0ebf4c9d27c9407` |
| runs/plain-it-tok-b2/final.nemo | 460083200 | `cd4ea4767b191ba034a41792fe9422013ac10ceb88d01768930e8a121cea3dea` |
| runs/sg-it-s2v4-c250-36k/final.nemo | 460072960 | `0a4d1b103297e169af74b6673ff97864747bb2c54b70a7ec91253d9583c003ff` |
| runs/sg-it-s2v5b-c250-36k/final.nemo | 460072960 | `ab84cd3be6a0abf32f31bbb50c7675fe623cdd6fcd4587383e5c04fdf276e38e` |
