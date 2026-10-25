# SUB15: error budget of the ES-1 streaming greedy gap (18.09 -> 15.00) and the first experiment

Written 2026-10-25. CPU-only analysis on dgx2 (nice 19, no GPU, no decoding, ~/gpu.lock untouched); dgx1 not touched.
No sealed set was read or decoded. Every number is tagged **OBSERVED** (measured from an artifact named here) or
**HYPOTHESIS** (inference or prediction). UNKNOWN = the existing artifacts cannot answer it.

Goal of the line: macro WER **< 15.0 on the frozen harness-es, streaming GREEDY, 0.5 s post-pad flush** (the
pre-registered ES-1 primary), with EOU / empty / latency preserved. Beam or rescoring gains count only if they come
back into greedy (e.g. by distillation), or they are reported on their own line with their serving cost.

## 0. Reference numbers (OBSERVED, verified at the source)

harness-es = MLS-es test 200 / FLEURS es_419 eval 200 / CV17-es test 200 / VP-es test 164 (`two_stage.subsample`
rule `rs[::len//200][:200]`; seal 8e9c226f; 764 clips, 18,844 normalised reference words). Normaliser
`common/textnorm.py` lang `es` (lower, typographic apostrophes, NFKD de-accent, `[a-z' ]`, digits -> space).

| model | greedy 0.5 s | beam-8 | oracle-8 | source |
|---|---|---|---|---|
| ES-1 (172b1539) | **18.09** (MLS 14.86 / FLEURS 17.10 / CV 21.18 / VP 19.20; CER 7.43) | 16.55 (12.86 / 16.20 / 19.98 / 17.18) | 12.98 (10.48 / 13.00 / 14.09 / 14.34) | dgx2 `~/ft/es1-results/probe_harness_post0.5.json`; release-final-2026-10-22.md "ES-1 full plain qualification" |
| S5 seed 0 | 18.08 (14.54 / 17.04 / 21.98 / 18.77) | 16.18 | 12.70 | Hub `runs/dgx-cfs-s5-es/probe_harness_post0.5.json` (copy: dgx2 `~/ft/sub15/hub/`) |
| S5 seed 1 | 18.07 (13.56 / 17.24 / 23.03 / 18.44) | 16.64 | 12.93 | Hub `runs/dgx-cfs-s5-s1-es/probe_harness_post0.5.json` |

The 18.09 / 16.55 numbers in the brief are correct and are the 0.5 s-flush NeMo probe on that bank (raw EOF:
18.21 / 16.64 / 13.14). Mynah streaming: CPU f32 18.07, CUDA f32 18.07, CUDA bf16 (serving default) 18.09.

## 1. Inventory of per-utterance artifacts for the best ES checkpoints (OBSERVED)

| artifact | ES-1 | S5 s0 / s1 | S2 s0 / s1 |
|---|---|---|---|
| aggregate probe (greedy / beam-8 / oracle-8, per set, empty, len ratio) | yes | yes | yes |
| per-utterance **greedy** hyps, streaming | **yes**: dgx2 `~/ft/parity-es1/ws_bf16.jsonl` (CUDA bf16 server, 764 clips, = 18.09) and `cli.jsonl` (CPU f32, = 18.07); Hub `results/dgx-2026-10-23/es1-mynah/` | no | no |
| per-utterance refs (raw, cased, punctuated) + VAD speech mask + speech end | yes: `~/ft/parity-es1/clips.jsonl` (`text`, `sm`, `speech_end_s`, `orig` path) | (same bank) | (same bank) |
| per-utterance greedy from the NeMo probe | no (the probe does not dump greedy; `greedy_blank_sweep.py --dump` can) | no | no |
| per-utterance **beam-8 / N-best / oracle** | **no** (probe run without `--dump-nbest`; no ES `nbest_b8.jsonl` anywhere on dgx2 or the Hub) | no | no |
| durations, speaker ids | yes via the eval manifests (MLS speaker = file prefix, 20 in the bank; VP `vp:<id>`, 94; CV hashed, 197 = ~1 / clip; FLEURS none) | | |
| train text (for word frequencies) | yes: `train_40h_es.json`, `mix/es/train_cv.json`, `train_vp_t50.json`, `train_fleurs.json` (1.05 M words, 61,945 types) | | |
| in-run eval curve | `es1-fr_p1.log` (steps 100..12,670) | metrics.json | metrics.json |

Consequence: every per-utterance number below is on **ES-1 streaming greedy (Mynah bf16, 18.09)**. Greedy -> beam ->
oracle is available only as aggregates; which utterances beam fixes is UNKNOWN until measurement M1 (section 5).
Reproduction check: per set 14.83 / 17.08 / 21.23 / 19.23, macro 18.09 (NeMo probe 14.86 / 17.10 / 21.18 / 19.20;
33 / 764 clips differ bf16 vs f32, macro identical to 0.02).

Scripts and raw outputs: dgx2 `~/ft/sub15/sub15_budget.py`, `sub15_runs.py` -> `budget.json`, `budget.txt`,
`runs.txt` (copy of the scripts in this session's scratchpad; not committed).

## 2. Measurements (OBSERVED, ES-1 streaming greedy, 764 clips)

### 2.1 S / D / I and noise
| set | words | S % | D % | I % | WER | empty | bootstrap SD (utt resample) |
|---|---|---|---|---|---|---|---|
| MLS | 7,498 | 10.14 | 2.97 | 1.72 | 14.83 | 0 | 0.61 |
| FLEURS | 5,001 | 10.88 | 2.38 | **3.82** | 17.08 | 0 | 0.81 |
| CV | 2,002 | **15.08** | 4.65 | 1.50 | 21.23 | 1 | 1.37 |
| VP | 4,343 | 9.19 | **7.37** | 2.67 | 19.23 | 1 | 1.44 |
| pooled | 18,844 | 10.64 | 4.01 | 2.47 | 17.12 | 2 | macro **SD 0.55**, 95 % CI 17.05-19.25 |

Seed noise (S5 s0 vs s1, same recipe): macro 18.08 / 18.07 but per set up to 1.05 (CV) and 0.98 (MLS) apart.
A 3.1-point macro gain is ~5.6 bootstrap SD; per-set claims need two seeds.

### 2.2 Utterance-level concentration
- Worst 10 % of utterances (by error count) hold **30.5 %** of all errors (MLS 23.9, FLEURS 26.2, CV 30.4, VP 33.9 %).
- 12.2 % of utterances are error-free; utterance WER p50 15.1, p90 35.7; 43 utterances have utterance WER >= 50 %
  (CV 26, VP 11).
- Empty hypotheses: 2 (CV 7.3 s, VP 7.5 s; both normal speech from t = 0). Beam-8 has 0 empties on all four sets
  (aggregate): the empties are search failures of greedy (OBSERVED at aggregate level only).

### 2.3 Duration buckets (pooled WER, words)
0-4 s 23.0 (204), 4-6 s 20.4 (1,070), 6-10 s 19.1 (3,197), 10-15 s 16.3 (7,012), 15-20 s 16.2 (6,529), >= 20 s 18.4
(832). Short clips are worse but carry 6.8 % of the words; within MLS the curve is flat (14.6-15.1).

### 2.4 Speakers
MLS (20 speakers): the top 20 % of speakers hold 33.6 % of MLS errors (worst 4747 at 25.3 %). VP (94): top 20 % hold
45.5 % (driven by the gross failures of 2.5). CV / FLEURS: no reusable speaker ids in the bank.

### 2.5 Failure modes by run structure (macro points = mean over sets of errors / set words)
| class | MLS | FLEURS | CV | VP | **macro pts** |
|---|---|---|---|---|---|
| deletion runs >= 4 words (incl. the 2 empties) | 0.65 | 0 | 0.90 | **3.34** | **1.22** |
| of which a leading-prefix drop (>= 3 words deleted before the first match; 14 clips, 9 VP) | 0.59 | 0 | 0.60 | 2.12 | 0.83 |
| insertion runs >= 4 words | 0.05 | 0.80 | 0 | 0.76 | 0.40 |
| gross length mismatch (hyp / ref words < 0.6 or > 1.6; 13 clips) | 0.49 | 0.26 | 1.00 | 2.92 | 1.17 |

Prefix drops: in 13 / 14 clips the VAD mask of `clips.jsonl` puts speech at t = 0.0 s and the words-per-second of the
reference is normal (2.2-3.1 w/s), so the reference text is plausibly spoken (HYPOTHESIS: a cold-start / blank-lock
failure of the streaming greedy decoder, typical of clips cut mid-flow). Examples: VP 12.5 s "me alegro de tener la
oportunidad ... sorprender a las mujeres europeas ..." -> hyp starts at "sorprender"; MLS 19.5 s drops the first 18
words. Whether beam recovers them, or whether 0.5 s of leading silence removes them, is UNKNOWN (M1, M2).

### 2.6 Lexical: train-frequency of the reference word (S or D of that word)
| train count | share of ref words | word error % | share of all errors |
|---|---|---|---|
| 0 (OOV) | 3.1 % | **54.7** | 10.0 % |
| 1-9 | 8.9 % | 23.9 (3-9) - 37.5 (1-2) | 15.0 % |
| 10-99 | 19.1 % | 16.7 | 18.6 % |
| >= 100 | 68.9 % | 10.4-10.5 | 42.0 % |

- Rare words (<= 9) cost **4.68 macro points** of S+D (MLS 4.20, FLEURS 5.04, CV 7.44, VP 2.03); their EXCESS over
  the common-word error rate is ~3.4 macro points (the largest single lens, overlapping with names below).
- Named-entity proxy (capitalised non-initial words of the raw refs, CV + FLEURS only): word error 45.7 % (CV) and
  37.2 % (FLEURS) vs 16.9 / 12.0 % for other words; 1.6 macro points from these two sets alone.
- Coverage learning curve (train text resampled at 25 / 50 / 100 %): macro ref-OOV 6.70 -> 4.64 -> 3.06 %, rare (<= 9)
  share 23.1 -> 17.2 -> 12.5 % (x0.68 and x0.73 per doubling). HYPOTHESIS: 2x in-domain text -> OOV ~2.1 %, rare ~9 %.
- Function words (de, a, la, del, las, ...) are 30.1 % of all errors; top confusions las->la, del->de, al->el (final
  /s/ and contractions), top deletions de (73), a (57).

### 2.7 Normalisation and reference artifacts (report only; the official metric is unchanged)
| lens | MLS | FLEURS | CV | VP | macro pts |
|---|---|---|---|---|---|
| apostrophes dropped | 0 | 0.02 | 0 | 0 | ~0 |
| accents | 0 by construction (the official normaliser de-accents) | | | | 0 |
| **FLEURS digits**: 37 / 200 refs contain digits, which the normaliser deletes ("libre de metros" for "15 metros"); the model says the number: 80 inserted + 21 substituted number words | - | 2.02 | - | - | **0.51** |
| VP truncated reference (1 clip: 12-word ref for 17.9 s of speech, 29 insertions) | - | - | - | 0.67 | 0.17 |
| segmentation only (space-free concatenation of the error span equal: "con versiones", "sobre medicacion") | 1.75 | 1.04 | 0.70 | 0.37 | 0.97 |
| Spanish homophone spelling (same key under b/v, h, ll/y, c/s/z, qu/k, double letters) | 1.57 | 1.00 | 1.60 | 0.30 | 1.12 |
| one-character substitutions | 3.48 | 3.44 | 5.04 | 2.44 | 3.60 |

The digit and truncated-reference rows are a hard floor under the frozen metric (no correct transcript scores them).
Segmentation and homophone errors are acoustic hits with wrong orthography: model/text-prior addressable, not floor.
Also seen: MLS-train text abbreviations leak into hyps ("usted" -> "vd", 3x), archaic MLS spellings ("mesmo",
"deste"): ref-OOV vs the CV + VP + FLEURS train text is 9.6 % on MLS vs 3.7 % vs all train text.

## 3. Decomposition of 18.09 -> 15.00 (macro, ES-1)

| layer | macro | step | status |
|---|---|---|---|
| streaming greedy (primary) | **18.09** | | OBSERVED |
| search-recoverable: greedy -> beam-8 | 16.55 | **-1.54** (MLS -2.00, FLEURS -0.90, CV -1.20, VP -2.02); S5 s0 -1.90, s1 -1.43 | OBSERVED (aggregate); per-utterance UNKNOWN (M1) |
| ranking-recoverable: beam-8 -> oracle-8 | 12.98 | **-3.57** (MLS -2.38, FLEURS -3.20, CV -5.89, VP -2.84); S5 -3.48 / -3.71 | OBSERVED (aggregate; oracle picks with the reference, an optimistic bound) |
| normalisation / reference floor inside the residual | 0.7-1.5 | lower bound 0.68 = FLEURS digits 0.51 + VP truncated ref 0.17 (measured on greedy); upper part = other ref noise / ambiguous segmentation | lower bound OBSERVED on greedy; at oracle level UNKNOWN; upper part HYPOTHESIS |
| model / data residual (candidate quality) | ~11.5-12.3 | 12.98 - floor | HYPOTHESIS (by difference) |

Reading:
- The whole search gap (1.54) is smaller than the SUB15 gap (3.09). Even a perfect distillation of beam-8 into
  greedy lands at ~16.5. **SUB15 is not reachable through search or ranking alone: the candidate set (oracle-8) must
  move by ~2 points** (HYPOTHESIS: if the greedy/beam/oracle ratios stay as in ES-1 and S5, greedy 15.0 needs
  beam ~13.7 and oracle ~10.8).
- With the 0.7-1.5 floor, 15.0 means removing ~3.1 of the ~16.6-17.4 removable points, i.e. ~18-19 % relative.
- Overlapping lenses on greedy (not additive): rare / OOV words ~3.4 excess, function words ~5.1 pooled, deletion
  runs 1.22, homophone spelling 1.12, segmentation 0.97, NE proxy 1.6 (CV + FLEURS only), digits 0.51.
- Search / ranking gains reported separately must carry their cost: the beam-8 probe takes 13.6 s vs 7.0 s greedy for
  the 764 clips on the GB10 (batched NeMo); beam search was not measured on the Mynah streaming path (its serving cost there is UNKNOWN).

## 4. Candidate first experiments (at most 3, ranked by expected greedy gain per GB10-hour)

Prior evidence used (OBSERVED unless noted): ES S1 (0.6x horizon) +2.68 vs ES-1; FR G1E -> H2 (2.3 -> 4.6 passes on
216 h) -3.48 DEV; IT W4-C250 8k -> 16k -3.36 harness, but 16k ~ 36k (schedule saturation at the same mix); IT C250
vs C250-oldrep: new videos added nothing over recycling at equal exposure; IT C1 static difficulty curriculum NULL;
IT E1 feature KD NULL; IT greedy blank penalty MARGINAL (-0.04 harness); IT soups NULL; FR G1E Granary additive NULL
in-domain (OOD benefit). GB10 cost: S5 = 9,011 s for 12,670 steps (0.711 s / step), ES-1 13,326 s (1.05 s / step).

### Rank 1 (CHOSEN FIRST): ES-D2 -- double the in-domain mix at the frozen factory horizon rule
- **Hypothesis:** the ES recipe is exposure / coverage limited: 126 h physical, rare words carry ~3.4 excess points,
  CV clips are seen only 2.2 times, MLS has 86 speakers in 40 h cut from a 917 h pool.
- **Mechanism:** more distinct utterances and words at the same passes (lexical coverage, speakers) plus a longer
  horizon (the axis that moved oracle-8 in FR H2 and IT B2).
- **Intervention (one variable: the per-source budget 40 h -> 80 h; everything else frozen):** `data_mix.sh es` with
  budget 80 h for MLS-es (same 86 speakers, same cut rules), CV17-es (<= 20 clips / speaker, more shards), VP-es
  (<= 60 / speaker, more shards, the same Nemotron t50 filter; if the pool has < 80 h under the cap, take all and
  record it), FLEURS unchanged (6.62 h); weights frozen 0.35 / 0.30 / 0.20 / 0.15; S5 recipe (bucket-pool 32,
  unbucketed decay); steps = round(4.60 x 246.6 / 0.04592) = 24,703 -> **capped at 23,762** by the pre-registered
  rule (4.42 passes, ~1,090 audio-h seen); seed 0; same probe (greedy / beam-8 / oracle-8, 0.5 s flush, harness-es).
  The harness exclusion lists must be re-applied to the new cuts (no CV / VP / MLS eval text or speaker leak).
- **Predicted delta (HYPOTHESIS):** macro greedy **-1.3** (band -0.5 to -2.5), oracle-8 -1.0 to -2.0. Reasoning:
  one exposure doubling gave -13 to -14 % relative in FR H2 and IT B2 (would be -2.4) but IT saturated at higher
  passes (would be ~-0.3); lexical-coverage arithmetic alone (words leaving the <= 9 class at the 10-99 error rate)
  gives ~-0.7; the midpoint, -7 % relative, is -1.3. Not SUB15 alone; it would leave ~1.8 for a second lever.
- **GB10 hours:** training 23,762 x 0.711 s = **~4.7 h**, VP teacher filter on the added VP ~0.3 h (HYPOTHESIS),
  probes ~0.1 h -> **~5.1 GPU-h**; data download / cut on CPU + network beforehand (no GPU).
- **Falsifier:** macro greedy delta > -0.5 AND oracle-8 delta > -0.8 -> data / exposure is not the ES lever at this
  scale (go to distillation / signal, not more hours). FLEURS worse by > 0.8 -> FLEURS over-exposure (26 views per
  clip) confounds the reading.
- **Confound risk:** medium. Hours and steps move together by design (the factory rule); the separating control is
  ES-H3 (same 126 h, 23,762 steps), to run only if D2 wins. FLEURS exposure per clip doubles (14 -> ~26). New VP
  shards may be noisier. One seed: bootstrap SD 0.55, per-set seed spread up to ~1 -> confirm any WIN with seed 1 and
  `diag/paired_eval.py` on per-utterance outputs.
- **Streaming deployability:** identical architecture, tokenizer, decoder and greedy path: zero serving cost. Stage
  2 (`<EOU>`, v4 recipe unchanged) and the Mynah parity / EOU / empty / latency gates must be rerun on the winner.

### Rank 2: start-of-stream / deletion-run fix (conditional on M1 + M2)
- **Hypothesis:** long deletion runs (1.22 macro, VP 3.34) are a greedy cold-start / blank-lock failure on clips
  that begin in mid-speech (speech at t = 0 in 13 / 14 prefix drops), not missing evidence.
- **Mechanism:** depends on M2. If 0.5 s of leading silence removes the drops, training clips lack abrupt onsets;
  if beam-8 recovers them but pre-pad does not, it is a search failure that greedy can learn to avoid.
- **Intervention:** short continuation from S5 s0, 1,500 steps (W4 short decay), with leading-silence trimming to
  0-50 ms on 50 % of the training clips (VAD-based, transcript unchanged) vs a control continuation with no
  augmentation (same steps, same seed). Two arms.
- **Predicted delta (HYPOTHESIS):** -0.3 to -0.6 macro vs control (VP -1 to -2), lead-drop words -50 %.
- **GB10 hours:** 2 x ~18 min + probes = **~0.7 h** (+ M2). Highest gain per hour IF M2 confirms the mechanism, low
  absolute ceiling (<= 1.22).
- **Falsifier:** lead-drop words on the harness not reduced by >= 50 % or macro delta > -0.2 vs control.
- **Confound risk:** the continuation itself (handled by the control); VP noise (SD 1.44): read the targeted count,
  not only WER. Caveat: real streams usually start before speech, so part of this failure may be a harness artifact
  of cut segments; it still counts on the primary metric.
- **Deployability:** training-only, zero serving cost. A decode-side pre-roll would be a serving change, reported
  separately (extra ~0.5 s of encoder work per stream start).

### Rank 3: teacher-labelled heterogeneous Spanish data (additive, Nemotron agree filter)
- **Hypothesis:** out-of-bank vocabulary / speakers / styles (YODAS2-es manual subtitles or Granary-es, agree <= 15
  with the Nemotron 0.6B teacher) reduce rare-word and NE errors beyond the four A2 sources.
- **Intervention:** ES-1 mix + external pool at W ~0.3, horizon extended by the external hours (IT C250 recipe).
- **Predicted delta (HYPOTHESIS):** -0.5 to -1.5; lower prior than Rank 1 (FR G1E NULL in-domain; IT C250 gain
  was exposure, not new videos). Licences and the teacher's MLS / CV / FLEURS training contamination must be checked.
- **GB10 hours:** teacher pass ~1 h + training ~5 h = **~6 h**, plus a multi-hour CPU / network pool build.
- **Falsifier:** greedy delta > -0.5 or OOD-style gain only. **Confound:** pool noise, W and steps change together.
- **Deployability:** zero serving cost.

Considered and not ranked: beam -> greedy self-distillation / MWER (capped by the 1.54 search gap; no unlabelled ES
pool; NeMo has no RNNT MWER trainer; IT feature KD NULL), static hard-example replay / curriculum (IT C1 NULL),
confidence filtering (VP already t50; no evidence of CV / MLS label noise beyond the abbreviations above), CV-weighted
rebalance (S2 / S5 showed CV moves of +-1.9 inside seed noise; predicted <= 0.3 macro), greedy blank penalty (IT
MARGINAL; it trades D for S).

**Choice: Rank 1, ES-D2**, predicted -1.3 macro greedy (-0.5 to -2.5) for ~5.1 GB10-h. It is the only family with
both a measured precedent for moving oracle-8 by >= 1.5 and a ceiling compatible with SUB15. Rank 2 goes first only
if M1 + M2 (~10 GPU-min) show the deletion runs are a cheap, learnable cold-start failure: at ~0.7 GPU-h it is the
better first spend, though it cannot reach SUB15 alone.

## 5. Measurements that need the GPU (planned, NOT run)

All on dgx2 or dgx1 when free, under the GPU lock, harness-es only (development bank), never a sealed set.
`MF=~/ft/manifests MX=~/ft/mix/es VALS=mls=$MF/eval_mls_es.json,fleurs=$MF/eval_fleurs_es.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json`

- **M1, per-utterance N-best (ES-1, S5 s0, S5 s1):**
  `python finetune/eou/diag/lm_fusion_probe.py --nemo <final.nemo> --lang es --per-set 200 --vals $VALS --alphas 0 --beam 8 --post-pad-s 0.5 --dump-nbest ~/ft/sub15/nb_<tag> --out ~/ft/sub15/nb_<tag>/probe.json`
  plus a NeMo greedy dump: `python finetune/eou/diag/greedy_blank_sweep.py --nemo <final.nemo> --dev $VALS --per-set 200 --penalties 0 --post-pad-s 0.5 --dump ~/ft/sub15/nb_<tag>/greedy.jsonl --out ~/ft/sub15/nb_<tag>/sweep.json`
  then CPU: `nbest_gap.py --greedy greedy.jsonl --nbest nbest_b8.jsonl` (S_near / S_far / seg, same / ranking /
  no_good_hyp per set and bucket) and a join with section 2.5: do the prefix drops and empties survive beam-8 and
  oracle-8? (check that `--alphas 0` leaves the LM-free path, as in fr_p1.sh). ~2 GPU-min per model, **~6 min**.
- **M2, cold-start test:** `greedy_blank_sweep.py` as above with `--pre-pad-s 0.5` (and 1.0), ES-1 only: does
  leading silence remove the 14 prefix drops and 2 empties? **~2 GPU-min.**
- **M3 (only for Rank 3):** Nemotron 0.6B teacher greedy on harness-es (ceiling of seq-KD; contamination caveat).
  **~5 GPU-min.**
- **For every SUB15 candidate:** keep per-utterance greedy (Mynah CLI CPU f32 matches NeMo within 0.02) and the
  N-best dump, and gate with `diag/paired_eval.py` (paired bootstrap), two seeds before any < 15.0 claim.
