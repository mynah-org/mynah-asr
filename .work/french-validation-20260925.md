# French validation on the qualified serving configuration

Status: DONE 2026-09-25 (registered before any French measurement; results below)

Task: S12-16

## Question

EN+FR is the product minimum and French is unmeasured. With Nemotron 3.5
streaming 0.6B (fixed) and the qualified configuration (`MYNAH_ASR_STREAM_PAR=4
MYNAH_ASR_FIN_STACK=1`, 6x5, int8, L=3):
1. Is a French evaluation set trustworthy enough to call validated?
2. Is French quality acceptable, and is an explicit `lang=fr` better than
   `auto` when the client knows the language (it was for English: S13-10)?
3. Does the engine stay correct and performant under load in French?
Kept apart: nominal English capacity (C=144, 2026-09-24) and the fault
qualification (C=64/80) are NOT re-established or replaced by this work.

## Protocol (registered)

**Set.** `samples/stress-fr/` built by `tools/fetch_stress_bank.py --config fr_fr
--lang fr --out samples/stress-fr` (FLEURS fr_fr, CC-BY 4.0, same builder, same
seed 42, class bounds 8/20 s; long clips are concatenations, as for English).

**Trust checks, each one counted and reported, nothing silently dropped:**
- audio: 16 kHz mono PCM16, duration matches the manifest, peak not below
  -30 dBFS (the qualification's `--min-peak-dbfs`), not clipped;
- reference: non-empty, contains French orthography where expected, same
  normalisation as the English scorer (streaming_metrics, `fr`);
- language: Nemotron offline `lang=auto` language tag per clip; a clip not
  identified as French is flagged;
- reference/audio agreement: Nemotron OFFLINE (full context) `lang=fr` WER per
  clip; WER > 0.5 flags a suspect reference (the FLEURS retake artefact seen in
  English, M-6). Flagged clips are listed with the reason and excluded from the
  VALIDATED subset; the raw set's numbers are reported beside it.

**Amendment (2026-09-25, still before any French measurement).** A French
QUALITY baseline already exists and is reused rather than rebuilt:
`samples/eval-bank/` (`tools/fetch_eval_bank.py`, FLEURS **test** split, 200 FR +
200 EN, never tuned on), frozen in `cc28e08` (2026-09-22): stream int8 [56,3],
FR WER 0.1316 [0.1148, 0.1496], CER 0.0651, first published word right 86.0 %,
measured with the manifest's language, i.e. explicit `fr`. What S12-16 still
lacks is (a) French under LOAD and (b) `fr` vs `auto`.

**Quality A/B (configuration, not model selection).** `tools/eval/lang_gate.py`
on the eval-bank FR 200, stream int8 L=3: explicit `fr` (must reproduce the
frozen WER 0.1316 within its interval, else the run is void) vs `auto` (the
language overridden to `auto`, same clips), paired: WER/CER, insertions,
first-word correctness, first-text latency as lang_gate reports it.
Recommendation rule: explicit `fr` is recommended if its WER is not worse than
auto's and its first-word correctness is not worse.

**The stress-fr bank** (built for load) gets the trust checks below, and the
soak runs on its validated subset.

**Serving.** `tools/bench/v2_qualify.sh` with the French corpus, the qualified
flags and topology, `--soak-c 144 --soak-seconds 900 --soaks 1`, the clients
sending the recommended language, judged by `v2_verdict` (all rows incl. A, B).
This is a French CONSISTENCY soak at the qualified level, not a new
qualification (one 15-minute soak, not two 30-minute ones).

## Evidence (2026-09-25, Axion c4a-highcpu-32, Nemotron 3.5 int8, clean tree ac21e26)

Raw: `.work/evidence/fr-20260925/` (untracked: chain log, trust record, both
lang_gate runs, both v2_qualify runs with dumps/metrics/procsamples, verdicts).

**1. Trust checks, stress-fr (1587 clips).** 1143 kept, 444 excluded, every
exclusion counted: peak below -30 dBFS 306 (FLEURS levels vary; the builder
warned of the same count), no language tag emitted under `auto` 129 (the model
declined to tag, which is NOT a detection of another language; counted as
registered), reference without French orthography 35, offline `lang=fr` WER
above 0.5 19, detected English 1. Under `auto`: fr-FR 1457, no tag 129, en-US 1.
Offline `lang=fr` WER mean 0.0976 over all 1587, 0.0928 over the kept 1143.

**2. Quality A/B, eval-bank FR (FLEURS test, 200 clips), stream int8 L=3.**
Gate: explicit `fr` reproduces the frozen baseline within its interval (WER
0.1320 vs 0.1316 frozen in cc28e08, interval [0.1149, 0.1498]; the small
difference is FACT, not investigated -- the bank was re-fetched and the code
moved since 09-22, all flags off).

| | explicit `fr` | `auto` |
|---|---|---|
| WER mean [95 % CI] | **0.1320** [0.1149, 0.1498] | 0.1347 [0.1174, 0.1535] |
| WER format-free | **0.1033** | 0.1058 |
| CER mean | **0.0656** | 0.0666 |
| S / D / I (5283 ref words) | 482 / 53 / 135 | 497 / 59 / 132 |
| first published word correct | **86.0 %** (172) | 85.0 % (170) |
| speech before the first word, p95 | **1.46 s** | 1.51 s |

Paired per clip: explicit better on 14, worse on 3, equal on 183 (165/200
transcripts identical); first word right only with explicit 2, only with auto
0. Same direction as English (S13-10: `lang=en` recovered 9/36 first-word
errors, introduced 0).

**3. French under load, qualified flags, 6x5, clients `lang=fr`, corpus = 498
of the 1143 validated clips (sample 500, seed 42), one 900 s soak each.**

| | C=144 | C=128 |
|---|---|---|
| verdict (v2_verdict, all rows incl. A, B) | **NOT QUALIFIED** | **QUALIFIED** |
| failing row | 3: finalization p95 **504 ms** vs 500 | none |
| emission lag p95 / worst window | 278 / 280 ms | 153 / 158 ms |
| finalization p95 | 504 ms | 261 ms |
| backlog max | 0.384 s | 0.284 s |
| streams lost / books | 0 of 9386 / balanced every dump | 0 of 8459 / balanced |
| transcripts identical to unloaded | 498/498 | 498/498 |
| stalls > 640 ms / max lag | 0 / 638 ms | 0 / 421 ms |
| worker RSS growth | 1.024x | 1.033x |
| audio/wall | 132.6x | 119.4x |
| TTFP load penalty p95 | +241 ms | +123 ms |

English at the same flags for reference (2026-09-24, 2 x 30 min each): C=144
lag p95 243 / fin p95 428 ms, C=128 129 / 221 ms. French sits ~35 ms (lag) and
~40-75 ms (finalization) above English at the same concurrency; at C=144 that
takes finalization 4 ms over the registered bound. The bound was not moved.

## Conclusion

- The French set is validated with its exclusions stated (1143 of 1587).
- French quality on the frozen test bank reproduces (WER 0.132); **explicit
  `lang=fr` is recommended over `auto`** by the registered rule (better on
  every aggregate, 14 vs 3 clips, +2 first words, 0 lost) -- the same finding
  as English, so the recommendation for production is: **send the language when
  the client knows it**.
- Under load the engine is correct in French at both levels (0 lost, books
  balanced, transcripts byte-identical to unloaded). **French is consistent at
  C=128 (wide margins) and NOT at C=144**, where finalization p95 is 4 ms over
  the bound. Nomenclature, kept exact so it cannot drift:
  - **EN nominal qualification:** C=144 (2 x 30 min, 2026-09-24).
  - **FR consistency (one 15-minute soak per level, NOT a qualification):**
    C=128 PASS with wide margins; C=144 FAIL on finalization p95, 504 vs 500 ms
    (0.8 % over), with 0 sessions lost, identical transcripts and balanced
    books. This says "C=144 did not pass this run under the registered
    protocol", not that French is structurally unable to hold C=144; two
    independent reps would settle the variance if FR C=144 were ever needed.
  - **Recommended conservative EN+FR operating point:** C=128 -- a prudent
    choice, not a new `qualified_safe_concurrency`. Bears on S12-13.
- Scope: one 15-minute consistency soak per level, not a two-soak
  qualification; the English nominal (C=144/128) and fault (C=64/80)
  qualifications are unchanged by this work.
