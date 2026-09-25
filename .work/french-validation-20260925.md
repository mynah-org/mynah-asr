# French validation on the qualified serving configuration

Status: OPEN (registered 2026-09-25, before any French measurement)

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

## Evidence

(none yet)

## Conclusion

(open)
