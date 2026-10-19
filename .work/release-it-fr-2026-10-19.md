# EOU-120M IT + FR: path to a public release (2026-10-19, Japan L40S)

Goal shift: from open-ended FT to two frozen, reproducible artifacts (IT, FR), each going
research candidate -> release candidate -> public model. Nothing is published from this document.

## A. Japan box qualification (Vast, 61.206.39.5:26451, new host; OBSERVED 07:40-07:55 UTC)

| item | value |
|---|---|
| GPU | 1x L40S 46068 MiB, driver 580.65.06, CUDA 13.0 toolkit in image; idle P8 34 C 34 W, 0 MiB, no foreign GPU process |
| CPU / RAM | 2x Xeon Gold 6430, 128 threads, 2 NUMA; 503 GB RAM (~400 GB available); load ~3 (shared host), steal 0 %, idle 98 % |
| disk | overlay root 100 GB (98 GB free); /dev/shm 62 GB (< 120 GB -> prov.sh puts the audio dirs on the root disk); dd 8 GiB direct write 4.7 GB/s, read 5.6 GB/s |
| network | eth0 docker bridge 172.17.0.4; DNS 1.1.1.1 / 8.8.8.8; no *_PROXY variables; UTC clock correct |
| latency (TCP connect / TLS / TTFB, avg of 5) | HF 8 / 31 / 37 ms; GitHub 8 / 32 / 38; PyPI 9 / 32 / 35; pythonhosted 8 / 30 / 33; PyTorch CDN 16 / 39 / 322; Zenodo 253 / 508 / 864 (EU origin) |
| sustained download | HF 2.51 GB file 2x: 37.9 / 38.4 MB/s (~305 Mbit/s, stable); PyTorch CDN 89 MB/s; GitHub release 9.7 MB/s (5 MB file, too short to saturate) |
| TLS (issuer + leaf sha256, Mac vs box) | identical on huggingface.co, cas-bridge.xethub.hf.co, github.com, objects.githubusercontent.com, pypi.org, files.pythonhosted.org, download.pytorch.org, zenodo.org, api.ngc.nvidia.com (all public CAs: Amazon, Sectigo, Let's Encrypt, GlobalSign) |
| trust store / config | only extra CA = `jvastai.crt` (Vast's own, used for the instance portal cert; not in any chain we use); no git config / credential helper; caches: conda, pip only |
| processes | Vast base image services only (jupyter, tensorboard, syncthing, caddy, cloudflared tunnels, portal); no unexpected mounts |

Verdict: PASS (sanity check only, not proof of host safety). Token written over ssh stdin to /root/.hf_token (0600),
never in argv / logs; prov.sh re-runs its own TLS gate before using it. ICMP not permitted in the container (ping n/a).

## B. Restore / parity (OBSERVED, 2026-10-19)

- Code: prov.sh clone of `lw-finetune-tooling` (52e5271 at restore, then 347ff4b / 59ad40c / 3490e2b for today's tooling);
  venv torch 2.11.0+cu128 (prov.sh pin), prov 07:43-07:51 UTC.
- Data: `resume_it.sh SNAPSHOT=it-c250` (22.1 GiB, 22 shards) -- data_snapshot.py hung again in the ProcessPool shutdown
  after extracting every file (12 defunct + 3 sleeping workers, as in Singapore). Not accepted on file count:
  `snapcheck.py` read every manifest row of the snapshot: **120,367 / 120,367 present, duration within 0.05 s, 0
  unreadable**; then the hung process group was killed and the chain resumed after the snapshot step.
- Checkpoints sha256: C250-36k 7e919281, EOU-v3 57d04dc4, v4 0a4d1b10, v5-B ab84cd3b -- **all match**. trail_s
  manifests (s2tail): 6 sources, 116,526 rows, 0 missing audio.
- Parity 1, GPU harness probe of C250-36k (lm_fusion_probe, 730 clips): **22.27 / 20.54 / 16.26** vs Singapore
  22.27 / 20.53 / 16.25 (greedy identical; beam / oracle +0.01 = numerical noise).
- Parity 2, CPU EOU protocol (s2_reset --post-pad-s 2.0, reset): **v4 FE 8.36 % / R 94.38 % / L 0.72 / 1.92 s / reset-WER
  22.54; v5-B 0.82 % / 92.47 % / 1.84 / 3.04 s / 22.08** -- identical to Singapore on every number.
- Anchor 3 (new, measured before any v5-CAL output): production EOU-v3 under the same protocol:
  FE 5.89 %, FE_any 6.58 %, R 94.66 %, L p50 / p90 0.64 / 1.92 s, reset-WER 36.07, CER 12.82, empty 4;
  per set FE MLS 1.5 / FLEURS 14.5 / CV 2.5 / VP 4.6 %, WER 32.27 / 30.57 / 39.23 / 42.21 (FLEURS 30.57 vs Mynah 30.91:
  consistent). v3 already has a ~6 % false-endpoint rate (FLEURS 14.5 %), and its ASR is 14 points worse than C250-36k.
- Disk: 100 GB root; after the sealed FINAL-IT was archived and verified, the duplicates (ITALIC zip, wip bundle,
  reference copies, qual dir) were deleted: **43 GB free** (gate >= 20 GB before training: PASS).

Verdict: environment parity PASS; the Japan box is the box of the day.

### data_snapshot.py restore hang -- diagnosed and fixed (axis closed)
- Symptom (2026-10-18 SG, 2026-10-19 JP): every shard extracted and verified, then the `ProcessPoolExecutor` never
  shut down (defunct + sleeping workers).
- Reproduction (`tests/snapshot_pool_smoke.sh`): a tiny 41-shard snapshot (200 CV clips) restored with the real code:
  **fork: 5 / 5 runs hung** (180 s timeout; 4 in the A/B, 1 more to keep a stack log). **spawn: 6 / 6 completed** in
  9-14 s with the full restore gate (per-shard size + sha256, file count, every referenced path present).
- Cause (OBSERVED, faulthandler stacks of the hung workers via SIGUSR1):
  1. `huggingface_hub/utils/_http.py:458` registers `os.register_at_fork(after_in_child=close_session)`.
  2. In each forked worker, close_session -> `httpx Client.close` -> `httpcore` connection pool close -> `_synchronization.py:285 RLock.acquire()`.
  3. That lock was held by a download thread (hf_hub_download, 16 threads in flight) at fork time; the thread does
     not exist in the child, so the worker blocks forever before taking any task.
  4. The other workers do all the extraction; the executor shutdown then waits forever for the blocked ones.
- Fix: extraction pool started with `spawn` (fresh interpreter, no inherited locks / at-fork hooks); `--mp-context`
  keeps fork available only to reproduce; SIGUSR1 dumps all thread stacks of a hung restore (faulthandler).
- Evidence log of one fork hang: `ft_logs/snaptest-fork-evidence/fork-1.log` in today's archive. The first 4 hang
  logs were deleted by a cleanup glob (operator error); their summary lines remain in `snaptest.log`.

## C. State map

### Italian (frozen facts, 2026-10-18)
| role | checkpoint (Hub) | numbers |
|---|---|---|
| best plain | `runs/plain-it-c250-36k` (sha 7e919281) | harness greedy / beam-8 / oracle-8 22.27 / 20.53 / 16.25; clean + 0.5 s flush 21.74 / 20.23 / 13.71 |
| production EOU | EOU-v3 `runs/plain-it-s2-e3-djlr` (57d04dc4; parent = E3, not C250) | Mynah FLEURS-it 30.91, EOU 200 / 200, p50 810 ms |
| stage 2 v4 | `runs/sg-it-s2v4-c250-36k` | FE 8.36 %, R 94.38 %, L p50 / p90 0.72 / 1.92 s, reset-WER 22.54; offline +1.87 vs parent |
| stage 2 v5-B | `runs/sg-it-s2v5b-c250-36k` | FE 0.82 %, R 92.47 %, L 1.84 / 3.04 s (pre-registered FAIL, latency), reset-WER 22.08; offline = parent |
| closed | KD E1-8k (NULL), static selection D, schedule axis (W4 = same ceiling) | do not reopen |

Next EOU axis: v5-CAL (section E). Plain 1-best < 20 is a separate ranking axis (oracle 16.25 vs greedy 22.27).

### French (reconstructed; nothing run since 2026-10-09, commits ff9d583 / f12f298 / a1153f1)
| item | value |
|---|---|
| data | MLS-fr only, 6 / 34 shards, 38.58 h, 37 speakers (IT A2-era had 65 speakers, now ~357 h / 6 sources) |
| stock EOU-120M on FR | NOT MEASURED (first eval at step 1500) |
| plain `runs/plain-fr-p40` | top-4 encoder 3e-5, dec/joint 3e-4, 4000 steps: MLS-fr val 50.99 / CER 26.91 |
| stage 2 `runs/plain-fr-s2-eou` | dec/joint 1e-4 (pre-djlr), pad p 0.5 U(1,3), every target + <EOU>: 49.18 / 25.87, EOU 97 % |
| Mynah FLEURS-fr 200 | 78.70 WER as recorded (47 empty); 73.39 peak-normalised; EOU p50 2154 ms, never 27.5 % |
| references | Nemotron 0.6B FLEURS-fr 13.2 (card 10.97 vs IT 5.23: FR ~2x harder for the teacher); Canary 180M 8.86 |

FR is ~5 recipe generations behind IT: data diversity, whole-encoder LR, steps / exposure, djlr, own-mix stage 2,
label policy, harness + flush + beam/oracle evaluation are all missing (section F).

## D. Release-readiness checklist and licensing audit

### Licensing (web-verified 2026-10-19; not legal advice)
- Base `nvidia/parakeet_realtime_eou_120m-v1`: NVIDIA Open Model License (NVOML, text of 2025-10-24). Derivative
  Models may be created and distributed, commercially; we own our Derivative Model "subject to NVIDIA's underlying
  ownership rights". Distribution: ship a copy of the Agreement + a NOTICE file with "Licensed by NVIDIA Corporation
  under the NVIDIA Open Model License". We may add terms for the derivative as a whole but may not drop NVOML's
  conditions -> the weights are released as `license: other / nvidia-open-model-license`, NOT Apache / MIT / CC.
  Guardrail clause (none documented in the base, so none removed), Trustworthy-AI terms apply, licence revocable /
  updatable by NVIDIA. Trademarks: "Parakeet" / "NVIDIA" only to describe origin -> name e.g.
  `mynah-eou-it-120m` / `mynah-eou-fr-120m`, "fine-tuned from NVIDIA Parakeet Realtime EOU 120M" in the card.
- Teacher `nvidia/nemotron-3.5-asr-streaming-0.6b`: OpenMDW-1.1. NOTE: the teacher IS in the final IT data path:
  the Granary pools of C250 (~228 h) carry Nemotron transcripts (agree <= 15). OpenMDW imposes no obligation on
  outputs; acknowledge it in the card anyway. NeMo toolkit: Apache-2.0 (only if we ship its code).
- Training data (all permit permissive release of weights; attribution in the card):
  MLS CC BY 4.0; Common Voice 17 CC0 (+ no speaker identification); VoxPopuli CC0 (European Parliament source);
  FLEURS CC BY 4.0; YODAS-Granary CC BY 3.0 (YouTube CC-BY uploads, self-declared -> mention takedown channel).
- Eval only: mTEDx CC BY-NC-ND 4.0 -> report numbers only, NEVER redistribute audio / segments / manifests
  (this covers clean-it tedx and the FINAL-IT tedx bank); ITALIC CC BY 4.0 + no identification / voice cloning.
  The FINAL-IT / clean-it bundles stay in the private Hub repo; publish selection scripts + ids + SHA256 instead.
- Uncertain: weights-as-adaptation of CC-BY data (attribution covers it); NC eval if a commercial purpose is
  attached (mTEDx); "Parakeet" trademark registry not checked.
- No blocking issue found.

### Release checklist (per language; all must be filled from frozen artifacts, none from memory)
- [ ] LICENSE (full NVOML text) + NOTICE (NVIDIA sentence, base-model link, "Modifications (c) 2026 Gabriele
      Mastrapasqua", dataset attributions, Nemotron pseudo-label acknowledgement)
- [ ] model card front-matter: license other / nvidia-open-model-license + link, base_model, language, pipeline tag,
      library nemo, datasets
- [ ] architecture: FastConformer cache-aware streaming encoder ([70,1] context), RNNT, stock SPE 1024 + <EOU> / <EOB>
      (blank 1026), 120M params, 80 ms frames; what changed vs base (all weights fine-tuned; vocabulary unchanged)
- [ ] training data table: source, hours, speakers / videos, licence, sampling weight; label provenance (human vs
      Nemotron pseudo-label, filters: VP teacher-WER <= 50, Granary agree <= 15, digits dropped)
- [ ] recipe: stage 1 (init, LR per group, schedule, steps, bs, max-dur, seed), stage 2 (EOU label policy, padding,
      LR, steps, selection rule); exact commands + git SHA
- [ ] compute: GPU type, wall-clock per stage (C250-36k ~5 h on one L40S at ~120 steps/min; stage 2 ~0.5 h)
- [ ] checkpoint sha256 of the released .nemo (and of the Mynah pack)
- [ ] accuracy: WER and CER, greedy / beam-8 / oracle-8 reported SEPARATELY; normalisation spelled out
      (lowercase, accents stripped, [a-z' ]); flush protocol (raw EOF vs 0.5 s); per test set; FINAL holdout one-shot
      result + the dev results, clearly labelled. "sub-20" only if 1-best greedy (or the stated decode) is < 20
- [ ] streaming parameters: chunk / lookahead / cache config used by Mynah and NeMo; decoder reset after <EOU>
- [ ] EOU metrics: false-endpoint rate (FE, >= 0.5 s speech left) and FE_any, true-EOU recall, latency p50 / p90
      from speech end, under the stated protocol (+2 s quiet, energy endpoint); Mynah streaming numbers
- [ ] known limitations: accent-free lowercase output, no punctuation / casing, digits not verbalised (numbers
      read as words only), domain gaps (VP / spontaneous), EOU is acoustic (pause) not semantic, latency vs FE trade-off
- [ ] inference examples: NeMo (`ASRModel.restore_from` + cache-aware streaming) and Mynah (pack export,
      CLI / server); export instructions (`export_to_mynah.sh`), Mynah version tested, CPU / CUDA parity
- [ ] reproducibility: public selection scripts for eval sets (no ND audio), SHA256 of eval manifests
- [ ] private archive keeps everything else (raw outputs, logs, pools)

## E. Stage2-v5-CAL -- PRE-REGISTERED (2026-10-19 ~08:45 UTC, GO given; frozen before launch)

Question: is the precision / latency trade-off controlled by the DOSE of positive <EOU> supervision on acoustically
ambiguous ends, with a usable operating point between v4 (q = 1) and v5-B (q = 0)?

**Exact definition.** Every stage-2 training draw i has the v4 target t_i = text_i + <EOU>. Unchanged from v4 / v5-B:
the global Python RNG (`random.seed(seed)`) draws, in this order, the sources, rows, and padded_i ~ Bernoulli(0.5)
with U(1, 3) s of quiet at U(-90, -60) dB. credible_i = trail_s_i >= 0.5 s (annotate_tail.py, frozen in b202980).
Then:
- padded_i or credible_i: keep <EOU> (as v4 and v5-B);
- otherwise (ambiguous): keep <EOU> iff u_i < q, with u_i ~ U(0, 1) from a DEDICATED generator
  `CAL_RNG = random.Random(seed + 7919)`, consumed only for ambiguous unpadded draws, in draw order; else the target
  is text_i only (as v5-B).
q = 0 is v5-B exactly (no CAL_RNG call), q = 1 is v4 exactly. `two_stage.py --eou-amb-keep q` (347ff4b).

**q = 0.5.** The a-priori midpoint of the only two anchors (q = 0, q = 1): no value fitted to, or interpolated from,
DEV numbers; one arm, no sweep.

**Expected EOU-positive share.** E[share] = p + (1 - p)(c + (1 - c) q), with p = 0.5 the pad probability and c the
draw-weighted credible share (~0.30): q = 0 -> ~0.65, q = 0.5 -> ~0.83, q = 1 -> 1. MEASURED by
`--batch-audit 300` (4,800 draws, the exact launch arguments): **v4 1.0000, v5-B 0.6498, v5-CAL 0.8213**.

**Determinism and invariance (OBSERVED, batch audit, 59ad40c).**
- 300 batches each: the audio hash (X, lens) is **identical** for v4, v5-B and v5-CAL
  (7a24a83e...aa799); only the target hash (Y) differs.
- A second v5-CAL run gives the identical Y hash (9b216053...533e6): the label draws are deterministic given the seed.
- Caveat (pre-existing, applies equally to the anchors): `np.random` is never seeded in training, so the sample-level
  realisation of the -90..-60 dB pad noise floor differs run to run (v4 and v5-B differed in it too). The audit seeds
  np only to make X hashable. Clip choice, order, pad decision, length and level are seeded. v5-CAL keeps the
  anchors' code path unchanged (no np seed), so the comparison is like-for-like. Hence "one variable" means the
  INTENDED TREATMENT VARIABLE (q); the stochastic padding-noise realisation remains uncontrolled, as in both anchors.

**Everything else unchanged.** v5b.sh plus one flag:
`S2MIX=<s2tail mls/cv/vp/fleurs/yg/yg6 at 0.1964/0.1684/0.1122/0.0842/0.1885/0.2503> INIT=plain-it-c250-36k
TAG=it-s2v5cal-c250-36k S2EXTRA="--lr 2e-5 --eou-min-tail 0.5 --eou-amb-keep 0.5" PROBE=1 M1=0 s2_mix.sh`
- seed 0, top-4 encoder 3e-5, dec / joint 2e-5, bs 16, 1500 steps, eval every 250;
- pad p 0.5 U(1, 3), max-dur 20, FastEmit 0;
- selection rule: min WER among evals with EOU >= 80 % and <= 10 % empty.

**Evaluation (identical to b202980; DEV harness only, FINAL-IT sealed).**
- `s2_reset.py --post-pad-s 2.0` on 730 clips, with the reset.
- Metrics: FE, FE_any (diag), R, L p50 / p90, reset-WER, CER, empty.
- Anchors re-measured on this box (section B):

| anchor | FE | R | L p50 / p90 | reset-WER | empty |
|---|---|---|---|---|---|
| v4 (q = 1) | 8.36 % | 94.38 % | 0.72 / 1.92 | 22.54 | 0 |
| v5-B (q = 0) | 0.82 % | 92.47 % | 1.84 / 3.04 | 22.08 | 3 |
| production v3 | 5.89 % | 94.66 % | 0.64 / 1.92 | 36.07 | 4 |

**Expectation (stated, NOT a gate).** If FE and L were linear in the positive share (0.65 -> 1.0), q = 0.5 (0.82) would
sit near FE ~4.5 %, L ~1.3 s. That is above the latency bar, so LATENCY-BOUND is a plausible outcome. A non-linear
response (most false endpoints from a minority of short tails) is exactly what this arm measures.

**Gates (fixed now; L(v3)/R(v3) enter because v3 is what a candidate must replace):**
- **PASS** (EOU release candidate): FE <= 4.18 % (0.5 x v4) AND FE <= FE(v3) 5.89 % AND L p50 <= 1.04 s
  (L(v3) + 0.40; tighter than L(v4) + 0.40 = 1.12) AND R >= 92.66 % (R(v3) - 2.0) AND reset-WER <= 22.58
  (W(v5-B) + 0.5) AND empty <= 5.
- **PARTIAL**: safety holds and exactly one primary misses by a bounded amount. Either FE <= 4.18 % with
  1.04 < L p50 <= 1.28 s (the v4 / v5-B midpoint) -- latency-leaning; or L p50 <= 1.04 s with 4.18 < FE <= 5.89 %
  -- precision-leaning. Recorded as one point of the dose-response; no promotion.
- **NULL** (no dose effect at the midpoint): within 1.0 pt FE AND 0.20 s L p50 of either anchor.
- **FAIL**: R < 90 %, or reset-WER > 23.08 (W(v5-B) + 1.0), or empty > 10, or FE > 5.89 % AND L p50 > 1.28 s
  (worse than v3 on precision and slower than the midpoint).
- Diagnostic, NOT a gate (Pareto reading): the chord v4 -> v5-B in the (FE, L p50) plane,
  L_chord(FE) = 0.72 + (8.36 - FE) / 7.54 x 1.12 s. CAL at (FE_c, L_c): L_c < L_chord(FE_c) - 0.10 s = better than
  interpolation (the dose buys latency back cheaper than linearly: improves the frontier); within +-0.10 s =
  interpolation (the positive share alone sets the trade-off); above = worse than the chord. Same for p90 and FE_any.
- Precedence FAIL > PASS > PARTIAL > NULL; any other outcome = BOUND (between the anchors but outside the bands):
  reported, not promoted.
- Always reported: the q = 0 / 0.5 / 1 table (FE, FE_any, R, L, W) and per-set FE (FLEURS is the hard set: v4 17 %,
  v5-B 2 %, v3 14.5 %).
- No second q, no threshold change, no runtime-policy tuning in response.
- Only a PASS continues: the standard probes (plain -> EOU delta), then Mynah m1 (FLEURS-it EOU p50, CPU / CUDA
  parity), then a recipe-freeze record, then FINAL-IT once.
- PARTIAL / NULL / FAIL stop the label-dose axis; the next decision (runtime debounce / commit policy as a separate
  layer, or ship v5-B / v4 behind a runtime policy) gets its own pre-registration.

Cost: ~30-40 min GPU (stage 2 + probes) + ~25 min CPU EOU protocol.
During the run: health only (NaN / OOM / stall); no intermediate metric changes recipe, q, gates or the selection
rule. After: exactly the probes above (s2_mix.sh PROBE=1 harness + clean, raw + 0.5 s, s2 + parent) and the
s2_reset EOU protocol; report q = 0 / 0.5 / 1 + v3, per set, the chord position, the class. PASS -> plain -> EOU
delta + Mynah m1 parity, stop before FINAL-IT. PARTIAL / NULL / FAIL / BOUND -> close the stage-2 IT axis (no v5-C,
no grid).
Launch record: see "v5-CAL launch record" below (commit SHA, exact command, UTC time, written before any output).

## F. FR on the frozen IT pipeline -- operational gap analysis (2026-10-19; no FR GPU until the IT EOU recipe freezes)

### Old FR recipe (2026-10-09, frozen kit `.work/lightweight-asr/`) vs the IT recipe being frozen
| axis | old FR | IT now (frozen or freezing) | directly replicable? |
|---|---|---|---|
| data | MLS-fr 6 / 34 shards, 38.6 h, 37 speakers, 1 domain | C250: MLS 40 h + CV17 + VP t50 + FLEURS train + Granary agree<=15 (~357 h, 6 sources) | yes for MLS / CV / VP / FLEURS (`prepare_mix.py --lang fr` accepted); Granary-fr needs the teacher pass (Nemotron fr validated) + dir-layout check |
| labels | human only | human + Nemotron pseudo-labels (VP filter t50, Granary agree<=15, digit clips dropped) | yes (teacher_ws.py --lang fr), GPU teacher pass |
| stage 1 | top-4 encoder 3e-5, dec/joint 3e-4, 4k steps cosine | whole encoder 2e-4 (E3) / 4e-4 (W4 WSD), dec/joint 3e-4, 8k-36k steps, max-dur 30 | yes (flags) |
| exposure | ~4k x 16 on 38.6 h (several passes of one domain) | ~50 processed h / 1k steps over 357 h | yes |
| stage 2 | dec/joint 1e-4 (pre-djlr), stage-1 manifest, every target + <EOU> | dec/joint 2e-5, parent's own mix, label policy = the frozen IT one (v4 / v5-B / v5-CAL) | yes (annotate_tail.py language-agnostic) |
| eval | MLS-fr val + Mynah FLEURS-fr, raw EOF | 4-set harness + clean, raw + 0.5 s flush, greedy / beam-8 / oracle-8, s2_reset EOU protocol, sealed FINAL | needs FR manifests + FINAL-FR |
| text | de-accented [a-z' ], stock English SPE | same | NOT as is: oe / ae ligatures (NFKD keeps them -> "c ur") and the curly apostrophe in training targets (frequent French elisions) must be normalised; FR-only change so IT numbers do not move; stock-SPE coverage audit on FR text first |

Missing diversity for FR = everything except MLS read speech: crowd-sourced read speech (CV), parliamentary
spontaneous-ish speech (VP), FLEURS-style read Wikipedia, and web / YouTube speech (Granary-fr). On IT the move from
MLS-only (P40, FLEURS-it Mynah 55.1) to multi-domain (v3 30.9) was the single biggest gain; the old FR shows the same
symptom (MLS-fr 49 vs Mynah FLEURS-fr 73-79).

### Order of work (CPU first, while the IT GPU closes)
1. FR-0 code: `LANG` through the job scripts; MLS-fr all shards in prepare_it.py; shared FR text normaliser
   (oe / ae, curly apostrophe, hyphen policy); stock-SPE coverage audit; Granary-fr dir layout. Unit tests.
2. FR-1 eval before any FR training number: FINAL-FR built and sealed (MLS-fr dev, CV17-fr validation, VP-fr
   validation speakers outside our pool, FLEURS-fr dev, mTEDx-fr train talks outside Granary-fr; an SLU-style
   bank only if licence-clean); harness-fr (MLS / FLEURS / CV / VP test, 200 each) + clean-fr (mTEDx-fr test/valid).
   Baselines on harness-fr, same protocol as IT: stock EOU-120M (never measured), old plain-fr-p40, old s2-eou.
3. Data A2-fr: MLS-fr ~40 h cut (all speakers), CV17-fr ~40 h (<= 20 per speaker), VP-fr teacher-t50, FLEURS-fr
   train (the teacher pass on VP-fr is the only GPU step before the run).

### The single high-information FR plain run (PROPOSED, not pre-registered)
FR-P1 = A2-fr with the frozen IT stage-1 recipe at the screening horizon: W4 (whole encoder 4e-4, dec/joint 3e-4,
WSD 5 % warm-up, decay last 20 %), 8k steps, bs 16, max-dur 30, seed 0 (~70 min L40S). Judged on harness-fr greedy /
beam-8 / oracle-8 (+ flush) against stock and old plain-fr-p40 on the SAME harness. It answers the question of the
week with one run: how much of the old FR gap is recipe + diversity (no Granary, no new labels beyond VP filtering).
IT reference for reading it: W4-8k on C250 = 26.12 / 23.67 / 19.43; on the A2-era IT mix the 8k E3 family was ~31-32 macro greedy (E3G-8k 31.71).
Then, by a pre-registered rule: Granary-fr agree<=15 to C250 scale and the long schedule (E3 36k or W4 16k), then
stage 2 with the frozen IT EOU recipe, Mynah export, FINAL-FR once. No FR < 20 promise (Nemotron itself: FR ~2x IT).

### v5-CAL launch record (written before any output)
- git 6df2d476703402280cc471159e84fec3f882dab1 (pre-registration commit)
- utc 2026-10-19T09:01:20Z, box Japan L40S, tmux `v5cal`, GPU lock
- cmd `S2MIX="<s2tail>/mls.json:0.1964,<s2tail>/cv.json:0.1684,<s2tail>/vp.json:0.1122,<s2tail>/fleurs.json:0.0842,<s2tail>/yg.json:0.1885,<s2tail>/yg6.json:0.2503"
  FT_ROOT=/root/ft INIT=/root/ft/runs/plain-it-c250-36k/final.nemo TAG=it-s2v5cal-c250-36k
  S2EXTRA="--lr 2e-5 --eou-min-tail 0.5 --eou-amb-keep 0.5" PROBE=1 M1=0 timeout 7200 bash finetune/jobs/s2_mix.sh`
  (<s2tail> = /root/ft/mix/s2tail), then `s2_reset.py --post-pad-s 2.0 --models s2_reset,s2` on the 730 harness clips.

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

## G. EOU decision-layer calibration on v5-B -- PRE-REGISTERED rule + gates (2026-10-19 ~09:50 UTC; b not yet computed, no biased decode run)

Last IT EOU axis, inference only: after it the IT EOU research stops whatever the result. No training, v5-B weights
frozen (sha ab84cd3b), FINAL-IT sealed and unused. CAL is NOT used to choose anything.

**Intervention.** In the frame-synchronous greedy RNNT decode of the protocol (`s2_reset.py`, Mynah reset), at every
symbol step the joint output z (log-probabilities on CPU; argmax-equivalent to logits since the normaliser is
common) is replaced by z' with z'_EOU = z_EOU + b and every other entry unchanged; k = argmax z'. Blank, text, <EOB>,
decoder state, encoder, reset after <EOU>, the +2 s quiet, the energy endpoint and every metric are unchanged.
The parent is never biased. `s2_reset.py --eou-bias b`.

**Rule for the single b (fixed now, before any margin is seen).**
1. Run the UNBIASED v5-B on DEV730 with the protocol (reset, +2 s).
2. Per frame t, take M_t = max over the frame's symbol steps of z_EOU - max(other outputs).
3. For each clip, need_i = -max M_t over frames in [speech end, speech end + W) with W = 0.64 s. This is the
   smallest bias that makes <EOU> win somewhere within W of the energy speech end.
4. b = max(0, median_i need_i), rounded UP to 0.05.

Reading: the bias at which the median clip would fire within production v3's p50 latency (0.64 s, an existing
anchor), on the unbiased path. Only the true-end window enters; internal pauses / FE never enter the rule.
`s2_reset.py --margin-rule 0.64` prints ONLY need p10/p50/p90 and b (no FE / latency / WER).

Disclosure: the unbiased trace would make the first-fire FE / latency computable for ANY b (the path is identical
up to the first <EOU>). That curve is deliberately NOT computed: one b, one decode.

**Baseline (frozen, this box):**
- v5-B: FE 0.82 %, FE_any 1.64 %, R 92.47 %, L p50 / p90 1.84 / 3.04 s, reset-WER 22.08, empty 3;
- v3: FE 5.89 %, L p50 0.64 s;
- v4: FE 8.36 %, L 0.72 s.

**Gates (biased v5-B vs these baselines):**
- **PASS:** L p50 <= 1.04 s (L(v3) + 0.40) AND FE <= 2.95 % (0.5 x FE(v3)) AND R >= 92.47 % AND
  reset-WER <= 22.58 (W(v5-B) + 0.5) AND empty <= 5.
- **PARTIAL:** FE <= 2.95 %, safety holds, 1.04 < L p50 <= 1.44 s (>= 0.40 s recovered). No RC.
- **NOT-SEPARABLE:** FE > 2.95 % (safety holds). Calibration buys latency only by buying back false endpoints: the
  EOU-vs-blank ranking does not separate true ends from pauses at that operating point.
- **NULL:** L p50 > 1.64 s (within 0.20 s of v5-B: b ineffective).
- **FAIL (safety):** R < 90 % or reset-WER > 23.08 or empty > 10.
- Precedence: FAIL > PASS > PARTIAL > NOT-SEPARABLE > NULL; anything else = BOUND.
- Diagnostic: the point vs the v4 -> v5-B chord and vs v3; FE_any; per set (FLEURS).

**After the result.**
- PASS: Mynah parity of the bias (runtime implementation, CPU / CUDA), freeze v5-B weights + runtime policy, then
  FINAL-IT once, then RC.
- Otherwise: STOP IT EOU research (no sweep, no second b, no threshold search) and choose v5-B / v3 on product grounds.

**Rule output (OBSERVED, 2026-10-19 ~10:05 UTC, before any biased decode):**
`s2_reset.py --margin-rule 0.64` on v5-B, DEV730: need p10 / p50 / p90 = 6.8 / 110.3 / 946.9, so **b = 110.3**.
Units are joint log-probability differences.
- At the energy speech end + 0.64 s, the median clip's <EOU> trails the winning output (blank) by ~110 nats.
- Fewer than 10 % of clips are within 7 nats.
The scale is degenerate for a "small calibration": v5-B is not marginally under-confident at the true end; its
<EOU> evidence is absent there and appears ~1-2 s later. Biased decode NOT run; stopped for a decision (the rule is
pre-registered, its output is reported as is).

**Biased decode launch record (before any output):** git dc03a16, v5-B sha256 ab84cd3b, b = 110.3,
utc 2026-10-19T10:09:50Z, `s2_reset.py --post-pad-s 2.0 --models s2_reset,s2 --eou-bias 110.3` on DEV730 (same vals,
parent C250-36k unbiased). No clipping / debounce / threshold / conditional bias; no second b.

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

## H. RC-IT freeze (2026-10-19 ~10:30 UTC, BEFORE any FINAL-IT decode)

**Choice (DEV + product criteria only, made before FINAL-IT is opened): RC-IT = Stage2-v4 on C250-36k.**
- Hub `runs/sg-it-s2v4-c250-36k/final.nemo`, sha256
  0a4d1b103297e169af74b6673ff97864747bb2c54b70a7ec91253d9583c003ff.
- Why, on DEV730 with the reset protocol:
  - v4 vs v3: -13.5 reset-WER (22.54 vs 36.07) for +0.08 s p50 (0.72 vs 0.64) and +2.47 pt FE (8.36 vs 5.89);
  - v4 vs v5-B: v5-B is only 0.46 WER better but +1.12 s p50, which is unacceptable for a realtime EOU model.
- Known limitations, to be declared: FE 8.36 % on DEV, FLEURS 17 %; false endpoints are mostly at internal pauses
  (ambiguous positive labels, b202980 / fbcdfbb).
- v5-B (ab84cd3b) = documented precision-first research checkpoint; NOT decoded on FINAL-IT, NOT a selection
  alternative after FINAL. v5-CAL and the bias arm are closed results.

**Frozen runtime / protocol (= the DEV protocol on which v4 was chosen):**
- Decode: frame-synchronous greedy RNNT, max 10 symbols per frame, no beam, no LM.
- Mynah reset after <EOU>: decoder state zero and last token SOS; encoder untouched; the transcript is the
  concatenation of segments.
- EOF / finalisation: 2.0 s of -75 dB quiet appended (`--post-pad-s 2.0`; it includes the DEV-chosen >= 0.5 s
  trailing flush). No debounce, threshold or bias.
- Implementation: `finetune/eou/diag/s2_reset.py` at the commit recorded in the launch line, `--models s2_reset`.
  The parent is loaded but not reported.
- Normalisation: s2_reset.norm (NFKD, lowercase, [^a-z' ] -> space).
- Endpoint: energy (80 ms frames, last within 35 dB of the peak) of the ORIGINAL clip.
- Not covered here: Mynah streaming (cache-aware chunked); that is the separate parity step after FINAL, on the DEV
  bank. This offline protocol is the frozen characterisation.

**FINAL-IT:**
- Seal SHA256SUMS sha256 8d02373775b548ec32bc1d8632665e0835b01adc87ecf0a0ecfe7a53f4ce42f2, re-verified file by file
  (`sha256sum -c`) right before the decode; 6 banks, 1,826 clips, all clips (--per-set 100000).

**Reported (and only this):**
- Natural clips, primary: macro over the 6 banks of greedy reset-WER, CER, empty, FE, FE_any, R, L p50 / p90;
  per bank the same.
- Trim variant (`final_trim_*`, cut at the energy endpoint + 50 ms), secondary: the same table.
- Bands from the 2026-10-18 pre-registration: < 20 macro greedy = success; 15-18 strong.
- Caveat: Nemotron labels / training data overlap MLS / CV / VP / FLEURS train, not these splits.

One run. No change to checkpoint, runtime, normalisation or reporting in response to FINAL-IT. FINAL-IT is
characterisation, not selection; after this decode it is spent for IT development.

### Result: FINAL-IT one-shot of RC-IT v4 (OBSERVED, 2026-10-19 10:42-10:52 UTC, git 9f0bc8d, seal verified 3,665 files)

Natural clips (primary), reset protocol, +2 s quiet, greedy:

| bank | clips | WER | CER | FE | FE_any | R | L p50 / p90 s | empty | split |
|---|---|---|---|---|---|---|---|---|---|
| mls | 400 | 23.44 | 5.90 | 2.5 % | 2.8 % | 97.5 % | 0.56 / 2.24 | 0 | 10 |
| cv | 400 | 19.44 | 5.51 | 2.5 % | 3.5 % | 97.2 % | 0.80 / 2.32 | 0 | 10 |
| vp | 110 | 20.35 | 10.55 | 7.3 % | 7.3 % | 98.2 % | 0.32 / 1.04 | 1 | 8 |
| fleurs | 116 | 12.56 | 3.02 | 16.4 % | 16.4 % | 84.5 % | 0.96 / 2.63 | 0 | 2 |
| tedx | 400 | 21.45 | 9.79 | 4.2 % | 4.2 % | 95.5 % | 0.96 / 2.08 | 8 | 23 |
| italic | 400 | 25.68 | 6.91 | 0.0 % | 0.0 % | 97.2 % | 1.12 / 2.40 | 0 | 0 |
| **macro / pooled** | 1,826 | **20.49** | **6.95** | **3.50 %** | **3.78 %** | **96.17 %** | **0.88 / 2.24** | **9** | |

(WER / CER = unweighted macro over the 6 banks; FE / FE_any / R / L / empty pooled over clips.)
Trim variant (secondary): macro WER 20.52, CER 6.95, FE 3.40 %, FE_any 3.67 %, R 96.44 %, L 0.80 / 2.00 s, empty 9;
per bank WER within 0.2 of natural (FLEURS FE 14.7 %).
Band (2026-10-18 pre-registration): < 20 = success -> **20.49 misses the success band by 0.49**; not "strong".
FINAL-IT is now spent for IT development. Nothing changes in response.

## I. FR-P1 -- PRE-REGISTERED (2026-10-19 ~12:00 UTC, frozen BEFORE any FR baseline or model score)

Question: how much of the old FR gap is recipe + data diversity + encoder adaptation? One plain run replicating
the IT A2+ methodology; NOT an attempt to maximise FR. Tokenizer axis closed for FR-P1: stock SPE, 0 % <unk> and
100 % round trip on every FR source.

**Order:** FR wiring -> FINAL-FR built + sealed (never decoded until a FR candidate is frozen) -> DEV harness-fr
frozen -> baselines -> VP-fr teacher pass -> FR-P1. Stop after FR-P1 before opening any new axis; no stage 2.

**Text policy (FR):**
- `textnorm.norm(t, "fr")`: oe / ae ligatures -> oe / ae, curly / modifier apostrophes -> ', then the IT rule
  (NFKD, drop marks, [^a-z' ] -> space). Hyphens are word breaks.
- `textnorm.text_ok(t, "fr")` replaces prepare_mix's text_ok for fr.
- IT output byte-identical (proven on 389,564 texts + every code point), with unit tests before any manifest.

**A2-fr (replica of IT A2+, weights unchanged):**

| source | cut | weight |
|---|---|---|
| MLS-fr train | 40 h cut spread over all 142 speakers (equal-share cap, speakers in sha1 order, clips 1-20 s), from all 34 shards | 0.35 |
| CV17-fr train | 40 h, <= 20 clips / speaker (prepare_mix defaults) | 0.30 |
| VP-fr train | <= 60 / speaker, 40 h pool (prepare_mix defaults), then the teacher filter below | 0.20 |
| FLEURS fr_fr train | all | 0.15 |

Common to all sources: digit / markup rows dropped (text_ok) and durations 1-20 s, as IT.

**VP-fr teacher filter (frozen now, before the teacher pass):**
- Teacher: nvidia/nemotron-3.5-asr-streaming-0.6b in mynah-asr-server-cuda, `teacher_ws.py --lang fr`, after the
  same 300-clip server-vs-CLI agreement gate as IT.
- teacher_wer = 100 x word Levenshtein(norm_fr(ref), norm_fr(teacher)) / len(ref words).
- An empty teacher transcript gives 100 (dropped); a missing / empty reference gives None (dropped).
- keep iff teacher_wer <= 50 (inclusive); durations as the pool (1-20 s); no extra dedup beyond prepare_mix's.
- The threshold is NOT changed after seeing the yield. If < 10 h survive: the run still goes ahead with the
  same weight, and the yield is reported as a finding.

**DEV harness-fr (frozen once built):**
- MLS-fr test, FLEURS fr_fr test, CV17-fr test and VP-fr test, 200 clips each, deterministic selection as IT;
- clean-fr = mTEDx-fr test + valid (post hoc).

**Baselines on the same harness, same protocol:**
1. stock parakeet_realtime_eou_120m-v1;
2. old `runs/plain-fr-p40`;
3. old `runs/plain-fr-s2-eou`.

**Arm FR-P1 = the IT W4 recipe, unchanged:**
- stock init, whole encoder 4e-4, dec / joint 3e-4, WSD (5 % warm-up, decay last 20 %);
- 8k steps, bs 16, max-dur 30, seed 0, `--abort-wer 1000:85`;
- no CTC / KD / tokenizer change / stage 2.

**Reporting (mandatory, macro + per source):** greedy WER, beam-8 WER, oracle-8 WER, CER, empty rate, raw EOF and
0.5 s flush (`lm_fusion_probe.py --lang fr`). Beam / oracle are diagnostic: they do not change the class.

**A. Causal lever class** (FR-P1 vs old plain-fr-p40, same harness, macro greedy, 0.5 s flush):

| class | delta WER |
|---|---|
| LEVER | <= -10 |
| CLEAR | -5 .. -10 |
| WEAK | -2 .. -5 |
| NULL | within +-2 |
| HARM | > +2 |

**B. Decision rule:**
- LEVER / CLEAR shows that recipe + diversity is a lever; it does NOT authorise stage 2. Next, ONE decision on how
  to scale plain FR (Granary-fr pools, MLS-fr hours, schedule), taken from the absolute WER and the greedy ->
  beam -> oracle gaps, with its own pre-registration.
- WEAK / NULL: audit labels / text / data before any more training.
- HARM: stop and audit.

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

**FR-P1 AMENDMENT A1 (2026-10-19 ~13:20 UTC; pre-score: no FR model, teacher or baseline decode has run).**
Labelled amendment, not clarification: "equal-share cap" in the frozen text was ambiguous, and it was resolved after
seeing FR pool METADATA (durations per speaker, no scores).

1. **MLS-fr 40 h cut rule.**
   - The IT procedure (`subsets.balanced_order`): each step adds the next clip of the speaker with the least
     accumulated duration.
   - Speakers and their clips are taken in sha1("mls:<id>") order, with 1-20 s clips; cut at 40 h.
   - Pool: 258,146 clips, 142 speakers, 1,076.2 h, very uneven (median speaker 13 min in the whole pool).
   - The resulting cut: 9,608 clips, 39.996 h, all 142 speakers.
     - The highest speaker level is 35.4 min; 91 speakers are exhausted before the cut.
     - Clips per speaker: min 1 / median 52 / max 149.
   - The per-speaker cap is NOT a free parameter: re-planning from the same metadata with a cap of 36, 60 and
     120 min gives the identical 40 h cut (sha256 e8d62ba3 in all three).
   - A hard 17-min cap (40 h / 142) cannot reach 40 h; it is rejected because it would contradict the
     pre-registered 40 h.
2. **Checkpoint selection.**
   - final.nemo = the last step (`--final-eof 1`), exactly as every IT W4 run: W4-C250-8k final = step 8000, C1,
     D0 / D1, E1.
   - This flag is the checkpoint decision point only. It is NOT an audio EOF / flush policy: the probes stay raw
     EOF + 0.5 s flush as pre-registered.
   - The frozen text did not state the selection, so it is fixed here.
   - In-run harness evals at 100 / 250 / 1000, then every 2000 flat / 500 in decay, are diagnostic only.
3. **clean-fr** (post hoc, mTEDx-fr test / valid) is not built for FR-P1; harness-fr and FINAL-FR are unaffected.

IT equivalence of the wiring (c6cfe56):
- IT MLS plan.json byte-identical (sha256 1036420f);
- 0 norm / text_ok mismatches on 608,252 IT text fields + every code point;
- the IT VP t50 manifest reproduced byte-identically.

FR-P1 is FINAL with A1; no further change.

**FINAL-FR builder choices (model-free, fixed before the build; no FR audio decoded by any model):**
- `prepare_final_it.py --lang fr`. The IT path is byte-identical to HEAD on an offline fake-Hub replay of all 6
  banks: manifests, trims, final_all, stats, SHA256SUMS, every wav, request log and stdout.
- VP: the pre-registered rule, speakers outside OUR VP-fr pool (`--vp-exclude mix/fr/train_vp.json,eval_vp.json`).
- Speech-MASSIVE fr-FR validation:
  - group = speaker; speakers of train / train_115 dropped; `is_validated == False` dropped;
  - test-split speakers NOT excluded (gated repo, 403 for this token): stated limitation;
  - CC BY-NC-SA 4.0, private.
- mTEDx-fr: TRAIN shards 0-1 of 41; talks in fr test / valid and in every Granary-fr original_audio_id dropped.
  - FR ending rule also accepts a spaced closing guillemet (`. »`).
  - Granary-fr ids (5 dirs, 4,432 shards) were listed from the dev machine because the box IP was rate-limited.
    Same logic as `granary_ids`; the cache is copied to `/root/final-fr-dl/granary_fr_ids.json`.

### FINAL-FR -- SEALED (2026-10-19 14:21 UTC, builder 847af09; BEFORE any FR model / teacher / baseline score)

- SHA256SUMS sha256 **8b0ebba088b5e6de1aa31a70341ed4b20c4058a4dd2fb190eff76a08766fd41c**;
  2,113 clips, 4,226 wavs (natural + trim).
- Granary-fr id cache sha256 ae4dea92 (86,238 videos).
- Archived: Hub `data/final-fr/` (bundle, SHA256SUMS, stats, SEAL, ids, logs) + `data/a2-fr/` (manifests), 19 files
  size-verified.
- Never decoded.

| bank | source | clips | h | groups | disjointness / exclusions |
|---|---|---|---|---|---|
| mls | MLS-fr dev | 400 | 1.68 | 18 speakers | dev speakers asserted absent from train + test; 2 texts in train / dev dropped |
| cv | CV17-fr validation | 400 | 0.65 | 400 speakers | client_ids of train / test dropped (none occurred); 18 texts in train / dev dropped |
| vp | VoxPopuli-fr validation | 400 | 1.06 | 17 speakers | 290 speakers of our VP-fr pool excluded (pre-registered rule) |
| fleurs | FLEURS fr_fr dev | 113 | 0.31 | 113 sentences | sentence ids asserted absent from train / test; 131 dup texts inside FINAL |
| tedx | mTEDx-fr TRAIN shards 0-1 (of 41) | 400 | 0.56 | 44 talks | talks of fr test / valid and all 86,238 Granary-fr videos dropped; <= 25 / talk |
| smassive | Speech-MASSIVE fr-FR validation | 400 | 0.43 | 55 speakers | 103 train / train_115 speakers excluded; **test-split speakers NOT excluded (gated, 403): speaker-disjointness from the test split is NOT shown**; 2 not validated dropped |

Text exclusions: 58,240 texts from 11 FR train / dev manifests.

**Granary-fr id source.** The espnet/yodas-granary column scan was too slow from the box (~1 s per shard: range reads on
600 MB parquets). The ids come from the original_source_id of nvidia/Granary fr/yodas/{fr_asr, fr_ast-en}.jsonl:
558,848 + 4,766,239 rows, 86,238 videos. Equivalence check: 150 random espnet shards (of 4,432) -> 4,752 videos, all
present in the NVIDIA manifests (0 missing). The cache is the union.

Licences: mTEDx CC BY-NC-ND, Speech-MASSIVE CC BY-NC-SA -> private evaluation only.

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

## J. FR-G1 -- PRE-REGISTERED (2026-10-19 ~19:00 CEST, before launch)

Question: with init, recipe and 8k steps fixed, does adding physically diverse Granary-fr speech to A2-fr lower the
FR WER? The comparison is FR-P1 (stock -> A2-fr, W4 8k) vs FR-G1 (stock -> A2-fr + Granary-fr, W4 8k).

**Granary-fr audit pool (model-free w.r.t. any student):**
- 8 spread ast shards per dir of fr000 / fr100-fr103: 78,246 clips, 156.5 h, 1,470 videos;
- every clip scored by Nemotron `--lang fr` (agree = teacher vs Granary reference, fr normalisation);
- agree p10 / p25 / p50 / p75 / p90 = 0 / 7.1 / 18.2 / 33.3 / 57.1 (IT pool: 0 / 2.5 / 8.7 / 18.8 / 33.3);
- excluded: 3,119 empty teacher texts, 201 duplicate texts, 1 dev / FINAL-FR leak (word 8-gram vs harness-fr + sealed
  FINAL-FR manifests: a model-free dedup, no FINAL decode); 0 digit clips; 48 clips overlap the A2-fr text (kept).

**Eligibility rule (fixed from this data table only, no model output):**
- the strictest T in {5, 10, 15, 20} keeping >= 90 % of the pool's videos AND >= 50 % of every dir's hours;
- 15 fails (fr103 37.4 %); **T = 20** gives agree <= 20 (inclusive): 41,415 clips, 88.97 h, 1,408 videos (95.8 %);
- per dir kept: fr000 67 %, fr100 58 %, fr101 55 %, fr102 59 %, fr103 51 %;
- label = the teacher text; manifest `mix/fr-yg/yg_agree20.json` sha256 158cb9cd.

**Sampler (IT `p1_additive.sh` rule, numerically):**
- PASSES 1.5 over the external hours gives W = 0.3267 (formula steps 11,883); then FIX_STEPS = 8000, exactly as
  it-w4-c250-8k.
- Source weights:

| source | weight |
|---|---|
| MLS-fr 40 h | 0.2356 |
| CV-fr | 0.2020 |
| VP-fr t50 | 0.1347 |
| FLEURS-fr | 0.1010 |
| Granary-fr agree <= 20 | 0.3267 |

- Expected exposure over 8000 x 16 draws, G1 vs FR-P1:

| source | G1 h | G1 passes | FR-P1 h | FR-P1 passes |
|---|---|---|---|---|
| MLS | 125.6 | 3.1 | 186.5 | 4.7 |
| CV | 36.7 | 0.9 | 54.5 | 1.4 |
| VP | 42.7 | 1.1 | 63.5 | 1.6 |
| FLEURS | 40.2 | 5.2 | 59.7 | 7.8 |
| Granary | 89.8 | 1.0 | -- | -- |
| **total processed audio** | **335.0** | | **364.2** | |

  G1 sees 8 % LESS audio than FR-P1 (Granary clips are short) and 33 % fewer A2 draws: a G1 gain is not an exposure
  effect.

**Everything else = FR-P1:**
- stock parakeet_realtime_eou_120m-v1 init;
- W4 (whole encoder 4e-4, dec / joint 3e-4, WSD 5 % warm-up, decay last 20 %), 8000 steps, bs 16, max-dur 30, seed 0;
- final.nemo = the last step; `--abort-wer 1000:85`; `--lang fr`; same harness-fr evals;
- no stage 2 / KD / CTC / LM / tokenizer change;
- FINAL-FR sealed.

Launch: `MIX=<above> TAG=fr-g1-w4-8k bash finetune/jobs/fr_p1.sh`.

**Reporting:** harness-fr macro + per source greedy / beam-8 / oracle-8 / CER / empty, raw EOF and 0.5 s flush.

**Class** (macro greedy, 0.5 s flush, FR-G1 - FR-P1 = 27.66):

| class | delta |
|---|---|
| LEVER | <= -3.0 |
| CLEAR | -1.5 .. -3.0 |
| WEAK | -0.5 .. -1.5 |
| NULL | within +-0.5 |
| HARM | > +0.5 |

Oracle is diagnostic.

**After:**
- LEVER / CLEAR: STOP. The next experiment (more Granary vs more exposure) gets its own pre-registration; no
  automatic 16k or 200-260 h pool.
- NULL / HARM: no compensation by hours or steps. First per source and greedy -> beam -> oracle.

**FR-G1 launch record (before any output).**
- Run: git 232cd40, utc 2026-10-19T16:45:41Z, train set 97,675 utts / 216.3 h (= A2-fr 56,260 + Granary 41,415),
  MIX as pre-registered, tmux `frg1`.
- Aborted earlier attempt (16:43 UTC):
  - the box checkout was stale (an untracked hand-copied granary_audit.py blocked `git pull`), so fr_p1.sh lacked
    the MIX override;
  - it started on A2-fr only (56,260 utts) and was killed within ~2 min, before any eval; its run dir and logs
    were deleted;
  - the launcher now refuses a dirty tree, a fr_p1.sh without MIX=, and a train set other than 97,675 utts.
