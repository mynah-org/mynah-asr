# FR false-endpoint autopsy (2026-10-21) and FR-v2 stage-2 proposal

Status: RC-FR v1 (6f00ac3) and FINAL-FR (93430d7) are CLOSED; nothing here modifies, reruns or reinterprets them.
Diagnostics only: no training, no serving threshold. FINAL-FR is used (spent) to understand the mechanism, never to
choose anything. Tools: `finetune/eou/diag/eou_autopsy.py`, `train_pause_audit.py`, `autopsy_report.py` (b86e9dc,
159c176). Artifacts: Hub `results/l40s-2026-10-21/autopsy/` (events, report.md, audit, punctuation check).

## 1. Method
`eou_autopsy.py` re-runs the frozen s2_reset protocol on FR-S2 (7814141d) and logs every model <EOU> (identity with
the frozen dumps: DEV 800 / 800, FINAL 2,113 / 2,113 clips, ids equal), classified by the frozen FE rule
(premature = >= 0.5 s of speech left), plus every internal pause >= 0.24 s that did NOT fire. 3,066 events on
2,874 clips: MLS false 149, MLS true 583, non-MLS false 86, non-MLS true 2,248.

## 2. Findings (OBSERVED, DEV + FINAL pooled; DEV and FINAL agree group by group)
1. **False <EOU> sit in LONG pauses and fire late in them.** Silence before the event p10 / p50 / p90:
   MLS false 0.00 / 0.88 / 1.44 s, MLS true 0.56 / 0.80 / 1.36 s. Whole pause around a false event p50 1.04 s (MLS),
   0.88 s (non-MLS). Speech resumes 0.16 s (MLS p50) after the false event. Position: mid-clip (rel_pos p50 0.59),
   4.5 s of speech left (p50), 18 words before it. 2 <EOU> per affected MLS clip (p50).
2. **The model's rule is a duration rule, and it is the same rule for true endpoints.** Hazard P(<EOU> | internal
   pause of length L): < 0.8 s: 0.2-1.6 % (short pauses are handled); 0.8-1.04 s: 6 % (MLS) / 9 % (other);
   1.04-1.6 s: **29 % / 17 %**; >= 1.6 s: **58 % / 43 %**. True endpoints fire after 0.80 s of silence (p50).
   A false endpoint is an internal pause longer than the model's own endpoint delay. MLS (audiobook reads) has many
   such pauses (220 internal pauses >= 1.04 s in the 800 MLS eval clips) and a slightly higher hazard at equal length.
3. **Not separable at the event.** EOU margin at the event: AUC(true > false) 0.60 (MLS 0.62). Frame phase
   (f % 2) 11 % vs 9 %: no chunk-phase effect. Pause level: MLS false pauses are room tone (-46 dB re peak, p50)
   vs -51.5 dB for true endpoints, but the true side includes the protocol's synthetic -75 dB quiet: not a usable cue.
4. **Mostly mid-sentence, not ambiguous sentence ends.** Non-MLS false events (punctuated references): a punctuation
   mark within +-1 word of the cut in 33 % (random cut in the same clips 22 %); sentence-final mark 22 %.
   (MLS has no punctuation.) HYPOTHESIS: most false endpoints are syntactically incomplete pauses.
5. **Acceptance rules (diagnostic only):** a confirmation window W ("report the <EOU> only if no speech within W")
   removes 40 / 60 / 69 / 82 % of false events (MLS 40 / 66 / 73 / 87 %) for W = 0.16 / 0.32 / 0.48 / 0.80 s, at a
   1:1 latency cost on every true endpoint (p50 0.72 -> 0.88 / 1.04 / 1.20 / 1.52 s). A minimum-pause rule removes
   only 32-38 % up to P = 0.8 s (false pauses are long). Window is a legitimate serving option with an explicit
   latency price; it would have to be chosen on a NEW dev bank, never on FINAL-FR.

## 3. Training construction (model-free, the v4 G1E mix, 3,000 clips per source, max-dur 20)
The v4 target is text + <EOU> at the end only; every internal silence is an implicit negative (`--concat-prob` 0).
| source | mix w | longest internal pause >= 0.56 / 0.8 / 1.04 / 1.6 s | pauses >= 0.56 s per h | trail s p50 |
|---|---|---|---|---|
| MLS-fr 40 h | 0.236 | 69.8 / 44.1 / 25.0 / 5.8 % | 332 | 0.32 |
| CV | 0.202 | 12.6 / 4.1 / 1.4 / 0.2 % | 98 | 0.64 |
| VP t50 | 0.135 | 14.4 / 4.6 / 1.9 / 0.4 % | 77 | 0.00 |
| FLEURS | 0.101 | 34.2 / 25.5 / 21.0 / 9.8 % | 135 | 1.60 |
| Granary agree<=20 | 0.327 | 11.1 / 5.5 / 3.1 / 0.9 % | 94 | 0.00 |
| mix-weighted | | 28.0 / 16.2 / **9.6 / 2.7 %** | | |
Positives in the same steps: 50 % of samples get U(1, 3) s of -90..-60 dB quiet + <EOU>.

**Verdict on the hypothesis "stage 2 lacks hard internal-pause negatives": PARTIALLY SUPPORTED, and narrowed.**
Short negatives are plentiful (and the model handles them: hazard < 2 % below 0.8 s). In the range where the false
endpoints live (>= 1 s) negatives are rare (9.6 % / 2.7 % of clips have one >= 1.04 / 1.6 s) while every padded
positive presents exactly that range (1-3 s) as a terminal silence: the data teach "about 1 s of silence => <EOU>".
Not measured: whether a causal 120M with 80 ms look-ahead can use pre-pause cues (prosody, syntactic incompleteness)
enough to wait; finding 4 says the cue exists in the text for most cases.

## 4. FR-v2 stage 2 proposal (NOT launched; each item needs approval and a pre-registration commit)
1. **Intervention (one variable): long internal-pause hard negatives.** During stage 2, with probability p = 0.3,
   take a training clip with an internal pause >= 0.24 s that is NOT within 0.5 s of the clip end, and stretch that
   pause to U(1.0, 2.5) s by inserting the clip's OWN room tone (loop of its quietest 80 ms frames), target
   unchanged (<EOU> only at the end). Everything else = frozen v4 (parent G1E 1271deff, mix, lr, steps 1500,
   pad 0.5 U(1, 3), seed 0, selection rule). Implementation: a new `--pause-stretch-prob/--pause-stretch-s` in
   two_stage.py, default off (v4 bit-identical when off: same RNG call order, checked by a dry audit).
2. **Causal comparison:** v2 vs frozen v4 (7814141d), same parent and seed; plus the serving-only reference
   "v4 + confirmation window W" with W chosen on DEV-EOU-v2 (does training beat a window at equal latency?).
3. **New DEV bank, DEV-EOU-fr-v2 (model-free, built and hashed before any decode):** long-form FR speech NOT in
   DEV-fr, FR-OOD1 or FINAL-FR (candidates: MLS-fr dev / test speakers not in those banks; VoxPopuli-fr sessions not
   used; mTEDx-fr talks not used, availability to verify): (a) internal-pause negatives = clips with >= 1 natural
   pause >= 0.8 s and speech after it (FE / hazard per pause bin); (b) endpoints = the same sources' clip ends
   (recall, latency); (c) the harness-fr DEV for WER retention.
4. **Fresh sealed FINAL-FR-v2:** disjoint speakers / talks / sessions from everything above, built model-free and
   sealed (SHA256SUMS) BEFORE any v2 decode; untouched by today's analysis.
5. **Pre-registered gates (proposal):** on DEV-EOU-v2 vs v4 -
   - primary: false-endpoint rate on the pause-negative bank, relative reduction >= 40 % (MLS-like pauses >= 1 s);
   - recall >= v4 - 1.0 pt; endpoint latency p50 <= v4 + 0.16 s and p90 <= v4 + 0.32 s;
   - harness-fr DEV reset-WER <= v4 + 0.3 and FE <= v4;
   - retention: FR-OOD1 reset-WER <= v4 + 0.5 (retention only, never selection);
   - classes PASS (all) / PARTIAL (primary met, one secondary missed by <= 2x its margin) / FAIL.
   Then RC-FR-v2 freeze and FINAL-FR-v2 once.

## 5. Probability choice rule (fixed 2026-10-21 ~17:05 UTC, BEFORE any silence-audit result is read)
Candidates p in {0.1, 0.2, 0.3} (`--pause-stretch-prob`, `--pause-stretch-s 1.0,2.5`). From `--silence-audit 400`
on the frozen v4 sampler (seed 0) at each p: N(p) = internal continuation pauses per hour of stage-2 audio in the
1.04-2.5 s band (bins 1.04-1.6 + 1.6-2.5), P(p) = terminal silences before <EOU> per hour in the same band.
**Choose the smallest p with N(p) >= 0.5 * P(p)** (negatives at least half as frequent as positives in the regime
where the false endpoints live); if no candidate reaches it, choose 0.3 (the largest candidate). No model run, no
DEV / FINAL result is involved. The audit at p = 0 must show the v4 baseline (and the explicit
`--pause-stretch-prob 0` batch audit must equal the frozen v4 hashes X 14abd104 / Y 5f8b73b6).

## 6. FR-v2 -- PRE-REGISTERED (2026-10-21 ~17:35 UTC, before launch; no v2 model result exists)

**Preflight (OBSERVED):**
- OFF identity: `--batch-audit 300` on ed01891 with the flag absent AND with `--pause-stretch-prob 0` = the frozen
  v4 sampler on e71fe7f: X 14abd1047c66d1df, Y 5f8b73b686b48e15 (4,800 samples), identical.
- Exposure (`--silence-audit 400`, 6,400 draws, ~17.8 h each), per hour of stage-2 audio, 1.04-2.5 s band:
  p = 0: negatives N 38.8 vs positives P 126.7; p = 0.1: N 56.5 vs P 125.6; **p = 0.2: N 74.0 vs P 124.9**;
  p = 0.3: N 89.5 vs P 123.7. The section-5 rule (smallest p with N >= 0.5 P) gives **p = 0.2**. Positives are
  unchanged by p (own RNG). Per bin at p = 0.2: internal pauses /h <0.8 540.3, 0.8-1.04 37.2, 1.04-1.6 41.2,
  1.6-2.5 32.8, >2.5 0.4; terminal silences before <EOU> /h 151.0 / 12.7 / 42.3 / 82.6 / 68.7.
- Banks: mTEDx part of DEV-EOU-FR-v2 / FINAL-EOU-FR-v2 built (DEV 174 internal by stratum 50 / 50 / 50 / 24 +
  100 endpoint; FINAL 168 + 100; talk-disjoint). MLS part: the last 8 MLS-fr train shards hold only speakers of our
  pools (0 rows kept). Rule fixed now: MLS-fr train speakers absent from all our MLS manifests are added with the same
  class / stratum / selection rules if >= 10 such speakers exist; otherwise the banks are mTEDx only (declared
  single-domain limitation). Both banks are sealed BEFORE any checkpoint (v4 or v2) is decoded on them; FINAL is
  tarred and its local audio removed. Order deviation from the steer, declared: the training is launched before the
  banks are sealed (the training does not read them; no decision depends on them until the seal exists).

**Run:** `LANG_=fr INIT=<G1E 1271deff> TAG=fr-s2v2-ps S2MIX=<G1E mix> S2EXTRA="--lr 2e-5 --pause-stretch-prob 0.2
--pause-stretch-s 1.0,2.5" PROBE=0 M1=0 bash finetune/jobs/s2_mix.sh` = the frozen v4 command (33a202f) plus the one
variable. Selection rule unchanged (v4: min in-run harness WER among evals with EOU >= 80 % and <= 10 % empty).
PROBE=0: the offline lm_fusion probes are diagnostics, not gates (can be run later).

**Evaluation (frozen s2_reset protocol, `--post-pad-s 2.0`, `--models s2_reset`):** v4 (7814141d) and v2 on DEV-EOU-FR-v2
(`eou_v2_score.py`); v2 on harness DEV-fr 4 x 200 against the reproduced v4 reference (25.64); FR-OOD1 retention once
on the selected v2 (v4 29.53).

**Gates (all required for PASS), v2 vs v4:** DEV-EOU-v2 internal-class FE (frozen FE rule) relative reduction >= 40 %;
endpoint-class recall loss <= 1.0 pp; endpoint latency p50 <= +0.16 s and p90 <= +0.32 s; DEV-fr reset-WER <= +0.3
and no DEV-fr source > +1.5; FR-OOD1 reset-WER <= +0.5. Reported by pause stratum (0.8-1.04 / 1.04-1.6 / 1.6-2.5 /
> 2.5). Diagnostic after the verdict: v4 native vs v4 + confirmation window W (W from DEV-EOU-v2 only) vs v2 native.
**PASS** -> freeze an RC-FR-v2 candidate, then FINAL-EOU-FR-v2 once. **FAIL** -> STOP (no v2b, no probability
change, no threshold, no extra steps, FINAL-EOU-FR-v2 untouched).

### Seal record (2026-10-21 17:28 UTC, after the v2 training, BEFORE any checkpoint is decoded on these banks)
- MLS-fr train: 142 / 142 speakers are in our pools -> by the rule above the banks are **mTEDx-fr only** (talk-
  disjoint from everything; single-domain limitation declared: no audiobook reads in DEV/FINAL-EOU-v2).
- DEV-EOU-FR-v2: 274 clips (internal 50 / 50 / 50 / 24 by stratum 0.8-1.04 / 1.04-1.6 / 1.6-2.5 / >2.5 s, endpoint
  100), SHA256SUMS sha256 851084f2918c3d77e9945f4599524dc865027b3a3721c552994fde2dd083a1f0. Hub data/eou-fr-v2-dev.
- FINAL-EOU-FR-v2: 268 clips (168 internal + 100 endpoint), SEAL.sha256 sha256
  70776709d422250250e7389f8f9fd80528ca64a3bdc74c1d259a148c8974cf9f (SHA256SUMS a474f449 + tarball); local audio
  removed; Hub data/eou-fr-v2-final. Never decoded.
- Decoded-PCM duplicates: 0 within the new banks, 0 against the 6,415 existing FR eval clips (DEV-fr, OOD1, FINAL-FR).
- FR-v2 training (782a7f6, p = 0.2): selected step 250 (v4: 250), in-run val WER 27.61, EOU 99.5 %, 1 empty.

### Result: FR-v2 (OBSERVED, 2026-10-21 17:28-17:35 UTC, eval git 8c96725) -- **FAIL (primary gate)**

DEV-EOU-FR-v2 (mTEDx-fr, 274 clips), frozen s2_reset protocol, v4 (7814141d) vs v2 (p = 0.2, selected step 250):

| | v4 | v2 | change |
|---|---|---|---|
| internal-class FE (primary) | 55.17 % | 59.77 % | **+8.3 % relative (worse; gate: >= 40 % reduction)** |
| internal 0.8-1.04 / 1.04-1.6 / 1.6-2.5 / >2.5 s | 24.0 / 52.0 / 70.0 / 95.8 % | 28.0 / 54.0 / 80.0 / 95.8 % | +16.7 / +3.8 / +14.3 / 0 % rel |
| <EOU> inside the target pause | 44.3 % | 48.9 % | |
| endpoint recall / p50 / p90 | 98.0 % / 0.80 / 1.68 s | 98.0 % / 0.80 / 1.62 s | 0 pp / 0 / -0.06 s |
| endpoint FE | 3.0 % | 2.0 % | |
| WER internal / endpoint | 25.02 / 32.26 | 25.54 / 32.84 | |

Retention (all PASS): DEV-fr reset-WER 25.48 vs 25.64 (-0.16; MLS -0.53, FLEURS +0.04, CV -0.15, VP -0.02);
DEV-fr harness FE 6.62 % vs 7.25 %, R 97.75 % vs 97.5 %, L 0.72 / 1.12 vs 0.72 / 1.20 s; FR-OOD1 29.62 vs 29.53 (+0.09).
Window frontier (diagnostic): v4 + W = 0.32 s reaches internal FE 33.3 % at p50 1.12 s; v2 does not move the frontier
(v2 + W = 0.32 s: 42.5 %).

Verdict: **FAIL** -> STOP as pre-registered (no v2b, no probability change, no threshold, no extra steps;
FINAL-EOU-FR-v2 stays sealed, never decoded). RC-FR v1 unchanged.
Reading (HYPOTHESIS, not tested today):
- the v4 selection rule picks step 250 for both runs, so v2 had seen ~800 stretched samples (250 x 16 x 0.2) when
  selected: the intervention may be under-exposed at the chosen checkpoint rather than ineffective;
- room-tone stretched pauses may be acoustically distinguishable from natural long pauses (the model can learn the
  artefact, not the rule);
- the mTEDx internal pauses are very hard for v4 too (55 % FE, 24 % already at 0.8-1.04 s): the long-pause regime
  may need lexical / prosodic context beyond a 120M causal encoder with 80 ms look-ahead.
Artifacts: Hub `runs/l40s-fr-s2v2-ps/` (final.nemo, metrics.json), `results/l40s-2026-10-21/fr-v2/` (dumps, scores,
gates, logs, scripts), `data/eou-fr-v2-dev`, `data/eou-fr-v2-final` (sealed).
