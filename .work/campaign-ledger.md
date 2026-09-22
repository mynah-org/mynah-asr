# Campaign ledger

Task: C-1

One line per decision, newest phase last. FACT is measured. HYPOTHESIS is not.
A failed hypothesis stays here rather than disappearing, because the cost of
re-proposing it later is higher than the cost of a row.

Detail lives in the per-item notes; this is the index a new session reads first.

---

## Closed before this campaign (frozen, do not reopen without new evidence)

| id | claim | verdict |
|---|---|---|
| R-8 | initial predictor-state emission lock | **REJECTED** — injection delays emission (median +4 decisions), degrades CER, and disrupts equally after the first token |
| R-8 | the extreme blank logits are an int8 artefact | **REJECTED** — f32 reproduces them frame for frame |
| R-8 | forcing a first token as a fix | **REJECTED** — the only arm that advances emission (`▁`) drops first-word correctness 16/21 → 10/21, and at a deployable point doubles CER |
| R-5 | `[56,0]` as a quality-preserving latency win | **REJECTED** — mean CER 0.0405 → 0.0525 |
| R-10 | "`[56,3]`→`[56,0]` removes 240 ms" | **FALSE** — the attention is unmasked inside the chunk, so the expectation is `(q-1)/2 × 80` = **120 ms**; measured −0.100 s median |
| R-10 | cache fill depends on the preset | **FALSE** — `ceil(56/q)` steps of `q×80 ms` is **4480 ms in every preset** |
| R-11 | within-chunk position explains first-token timing | **UNTESTABLE** — 17/21 crossings at position 0 by construction of the greedy scan |
| R-11 | the within-chunk lookahead does real work | **FACT** — margin 2.26/4.58/4.88/6.11 across positions, 3101 balanced post-crossing decisions |
| R-13 | where the positions diverge | **FACT** — identical through the whole conformer stack; first divergence at the prompt projector's ReLU (input 1.13×, output 7.21×) |
| R-13 | the prompt one-hot dominates the ReLU gating | **HYPOTHESIS** — read from the code, not measured |
| R-13 | high encoder norm is harmful | **NOT ESTABLISHED** — co-occurrence with a large blank margin only |

---

## PHASE A — EN/FR quality gates

**EXPERIMENT A-1.** Is the committed corpus sufficient to claim English and
French quality? Run the new gate on it and look at the interval, not the point.

**RESULT A-1 — FACT. It is not, and now there is a number.**

| | n | WER mean | 95 % CI | width |
|---|---|---|---|---|
| EN | 4 | 0.0561 | [0.0000, 0.1122] | **0.1122** |
| FR | **2** | 0.1979 | [0.0625, 0.3333] | **0.2708** |

A French interval 27 points wide cannot distinguish a good model from a bad
one. Every quality statement made on this corpus so far was a DIAGNOSTIC
statement and must not be quoted as a quality claim.

**DECISION A-1.** Keep the 21-clip corpus for mechanism work, where it has been
excellent, and build a separate evaluation bank. FLEURS **test** split, English
and French, 200 utterances each (~34 min per language) — a split nothing in this
repo has ever been tuned on. Ground truth is the dataset transcription; the
model's own offline output is never an oracle.

**FACT A-2, found while building it.** FLEURS records the same sentence from
several speakers under one `fleurs_id`: 200 French records are only 100 distinct
ids. Naming the output by id silently overwrote half the bank and produced a
manifest with 400 rows over 209 files — some audio would have been scored
against another speaker's row. Caught by printing both counts; the fetcher now
names by source file and **asserts** the two counts agree.

**RESULT A-3 — FACT. The frozen `[56,3]` int8 baseline, 400 utterances.**

| | n | audio | WER mean | 95 % CI | CER mean | S / D / I | ref words |
|---|---|---|---|---|---|---|---|
| **EN** | 200 | 2054 s | **0.1211** | [0.1056, 0.1384] | 0.0713 | 388 / 63 / **105** | 4568 |
| **FR** | 200 | 2060 s | **0.1316** | [0.1148, 0.1496] | 0.0651 | 478 / 56 / **137** | 5283 |

| | empty | first published word CORRECT | prefix rewritten | speech → first word (med / p95) | speech → first CORRECT word |
|---|---|---|---|---|---|
| EN | 0/200 | **82.5 %** | 0/200 | 1.075 / 2.200 s | 1.020 s (n=165) |
| FR | 0/200 | **86.0 %** | 0/200 | 0.840 / 1.460 s | 0.810 s (n=172) |

WER by duration — **short utterances are the weak spot, and more so in French**:

| | < 6 s | 6–12 s | ≥ 12 s |
|---|---|---|---|
| EN | 0.147 (n=17) | 0.111 (n=132) | 0.138 (n=51) |
| FR | **0.212** (n=14) | 0.126 (n=138) | 0.125 (n=48) |

The interval went from 0.27 wide (FR, n=2) to **0.035** (n=200): the gate can now
see a 2-point WER move, which is what makes Phase C's candidates judgeable.

**Three things this baseline says that were not visible before.**

1. **Insertions outnumber deletions in both languages** — EN 105 against 63,
   FR 137 against 56. This engine leans toward saying too much, not toward
   truncating. Any latency change must be checked against the insertion count,
   not only against WER, because emitting earlier is exactly the pressure that
   would raise it.
2. **About one first published word in six is wrong** (17.5 % EN, 14.0 % FR) on
   real data, and this path never retracts. Q-2 measured this on 21 clips and it
   holds at n=400.
3. **Short utterances are the weak spot**, French especially. That is also where
   first-emission behaviour matters most, so the two problems overlap.

**NEXT.** Phase B: does the R-13 high-norm regime predict any of these?

---

## PHASE B — does the R-13 high-norm regime matter?

**EXPERIMENT B-1.** Per utterance on the 400-clip bank, the fraction of
pre-crossing decisions with `|enc| > 10`, joined to the outcomes the frozen gate
already scored. Split at each language's median.

**RESULT B-1 — FACT, and it is the opposite of a pathology.**

| outcome | EN low → high | FR low → high |
|---|---|---|
| WER (mean) | 0.1418 → **0.0943** | 0.1459 → **0.1173** |
| CER (mean) | 0.0834 → 0.0556 | 0.0659 → 0.0643 |
| first word CORRECT | 0.761 → **0.908** | 0.790 → **0.930** |
| speech → first word | 1.259 → 1.119 s | 0.940 → 0.952 s |

**High-norm utterances are BETTER, in both languages, on every outcome that
matters.** Nothing is normalised, clamped or compensated for. R-13 stands as a
fact about the representation and is closed as a defect.

**RESULT B-2 — the high norm is a PROXY.** `hi_frac` tracks leading silence
almost perfectly (mean onset: EN 0.059 s low against 0.525 s high; FR 0.275
against 1.438). Splitting by **onset alone** reproduces most of the effect:

| split by onset | EN short-lead → long-lead | FR short-lead → long-lead |
|---|---|---|
| first word CORRECT | 0.770 → **0.880** | 0.802 → **0.919** |
| WER | 0.1410 → **0.1013** | 0.1472 → **0.1157** |

Within the long-lead half `hi_frac` still separates (EN 0.843 → 0.967, n=70/30;
FR 0.882 → 0.958, n=51/48), so it is not a pure proxy — but the dominant
variable is how much audio the engine processed before speech began.

**DECISION B.** Close the high-norm branch. Promote what it uncovered:
**an utterance whose engine has been running before speech starts gets a better
first word and a lower WER, by 11–12 points and 3–4 points respectively.**

**Still a CORRELATION.** A recording with a clean lead-in may simply be a better
recording. That is what Phase C's first candidate tests causally.

**NEXT.** Phase C candidate 1: feed the encoder audio before the speech and see
whether the effect survives, on the same bank, each language on its own.

---

## PHASE C — responsiveness and quality candidates

**RESULT C-1 — REJECTED. The lead-in effect is not causal.**
Give all 400 utterances +2.0 s of leading silence, paired per utterance:

| | WER strict | WER format-free | first word CORRECT | insertions/utt |
|---|---|---|---|---|
| EN | 0.1211 → 0.1214 | 0.0916 → 0.0919 | **0.825 → 0.770** | 0.525 → 0.490 |
| FR | 0.1316 → 0.1332 | 0.1033 → 0.1045 | **0.860 → 0.845** | 0.685 → 0.665 |

WER does not move, and first-word correctness gets **worse** — 5.5 points in
English. Phase B's +11-point correlation was confounded: a recording that
happens to start with silence is a different recording, not a warmed-up engine.

*Not interpretable from this arm:* speech-to-first-word appears to rise by
0.36 s (EN) and 0.72 s (FR), but `clip_onset` estimates its noise floor from the
10th-percentile frame energy and digital silence drives that to zero, so the
detected onset moves. First-word CORRECTNESS is a pure text comparison and is
unaffected by this, which is why the rejection rests on it.

**RESULT C-1b — the dose-response settles it. More lead-in is monotonically
worse.**

| | base | +2.0 s | +4.5 s (full cache) |
|---|---|---|---|
| EN WER format-free | 0.0916 | 0.0919 | **0.1106** |
| EN first word CORRECT | 0.8250 | 0.7700 | **0.7136** |
| FR WER format-free | 0.1033 | 0.1045 | 0.1080 |
| FR first word CORRECT | 0.8600 | 0.8450 | 0.8450 |

At the dose that fills the encoder cache completely, English first-word
correctness falls **11 points**. A fully warm cache makes the first word worse,
which is the exact opposite of the Phase B correlation's direction.

**DECISION C-1.** Close the lead-in / cache-warmth branch. **R-12 is no longer
merely unjustified: the evidence points against it.** Nothing is prepended in
production, and cache priming does not get built. The correlation was a property
of which recordings have clean lead-ins, not of the engine's state.

**RESULT C-2 — FACT, and it changes every quality number so far.
A quarter of the bank was being scored on a convention, not on recognition.**

FLEURS references keep digits (`Since 1966 ... 400 Royal Bengal tigers`); this
checkpoint verbalises (`since nineteen sixty six ... four hundred royal bengal
tigers`). Both FLEURS columns keep digits -- `text_norm` only lowercases -- so
this is not a wrong-column mistake, it is a genuine convention mismatch.

| | n | WER strict | **WER format-free** | CER strict | **CER format-free** |
|---|---|---|---|---|---|
| EN, reference has a digit | 47 | 0.2461 | **0.1204** | 0.1933 | 0.0502 |
| EN, no digit | 153 | 0.0827 | 0.0827 | 0.0338 | 0.0338 |
| **EN all** | 200 | 0.1211 | **0.0916** | 0.0713 | **0.0377** |
| FR, reference has a digit | 44 | 0.2529 | **0.1241** | 0.1872 | 0.0542 |
| FR, no digit | 156 | 0.0974 | 0.0974 | 0.0306 | 0.0306 |
| **FR all** | 200 | 0.1316 | **0.1033** | 0.0651 | **0.0358** |

It also explains the insertion excess flagged in A-3: 76 of 105 English
insertions and 100 of 137 French ones are on digit utterances, because one
reference token `2007` becomes three hypothesis tokens.

**THIS IS NOT LOOSENING THE SCORER.** The words must still be right. The
reference's digits are expanded into every legitimate spoken rendering of *that*
number and the best is taken: `2007` may be "two thousand seven" or "twenty oh
seven", never "two thousand eight". A model that emits digits is not punished
either. Self-tested, including that a WRONG number is still an error. Both
scores are reported side by side, always; neither replaces the other.

**Recognition WER is therefore EN 9.2 % and FR 10.3 %**, not 12.1 and 13.2. The
remaining digit-utterance error (0.12 in both languages against 0.08/0.10
elsewhere) is real -- e.g. `Le 15 août` transcribed as `Le août`, a genuine
deletion -- so digit utterances are still harder, just not three times harder.

**NEXT.** Q2, the ceiling matrix: the same 400 utterances through int8 and f32,
streaming and offline, and the other declared presets, to find out whether 9-10 %
belongs to the checkpoint or to us.

---

## Q2, the ceiling matrix — the two arms that decided it

The matrix was asked one question: **is 9-10 % coming from the checkpoint,
quantization, streaming semantics, our implementation, or configuration?** Two
of the six arms answer it, and they finished first.

| | EN | FR |
|---|---|---|
| WER format-free, streaming `[56,3]` INT8 | 0.0916 | 0.1033 |
| WER format-free, offline INT8 | 0.0884 | 0.1011 |
| **streaming cost** | **+0.0032** | **+0.0022** |
| CER format-free delta | +0.0013 | +0.0009 |
| S/D/I streaming | 388/63/105 | 478/56/137 |
| S/D/I offline | 378/62/103 | 485/49/130 |

Per-utterance classification over the same 200 clips per language:

| | EN | FR |
|---|---|---|
| both correct | 25.0 % | 17.0 % |
| **both wrong in the identical way** | **45.0 %** | **45.5 %** |
| both wrong, differently | 25.5 % | 33.0 % |
| offline correct / streaming wrong | 4.0 % | 2.0 % |
| streaming correct / offline wrong | 0.5 % | 2.5 % |

**RESULT.** Streaming costs three tenths of a WER point. Nearly half the corpus
produces the *same error on both paths*, which is what an error inherited from
the checkpoint looks like and is not what a serving defect looks like. The
asymmetric cells are small and nearly balanced in French.

**DECISION.** There is no catastrophic engine/quantization/streaming quality gap
hiding inside 12-13 %. That was the one thing this matrix could have found that
would have outranked everything else, and it is not there. So the quality lane
closes as a research topic and WER/CER revert to what they should have been all
along: **regression gates on the engine**. The remaining arms (f32 streaming and
offline, lookahead 6 and 13) keep running and will say whether INT8 costs
recognition, but they cannot change this conclusion and nothing waits on them.

## PRIORITY CORRECTION — the mission is SERVER V2

Four days of engine investigation, a real scheduler freeze found and fixed, a
scoring convention corrected, curves up to C=64 — and still no answer to the
only question a deployment asks: *what concurrency can I promise continuously?*

`configs/perf/axion-c4a-highcpu32-nemotron-streaming.json` has said
`status: screened` and `long_soak_qualified: null` since 2026-09-20, and it says
why itself: the C=32 600 s soak passed every serving gate, lost no stream, and
did not qualify because **transcripts were never checked** -- there was no
manifest to check them against. There is one now.

| | |
|---|---|
| FACT | streaming costs +0.003 WER against offline; 45 % of errors are identical on both paths |
| DECISION | quality is a guardrail, not the mission; Server V2 qualification is the mission |
| EXPERIMENT | `.work/server-v2-qualification.md`: candidate frozen, twelve bounds registered before the first run |
| NEXT | ladder C=8/16/24/32 on Axion, then two independent 1800 s soaks at C=16 |
