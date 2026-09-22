# Is the initial RNNT state itself a barrier?

Task: R-8
Task: R-9

A second report on the same family of model (a NeMo streaming transducer,
fine-tuned elsewhere) describes a failure mode we have not tested for: outputs
that stay EMPTY, and a causal intervention that removes it. Forcing one token
into the decoder is reported to have turned **178 of 179 empty outputs into
outputs with text**, with the subsequent blank probability falling from ~0.96 to
~0.47.

This note records what that does and does not establish, why it is worth an
experiment HERE, and what that experiment must control for. Nothing is
implemented. Written 2026-09-22.

## What the report establishes, and what it does not

**It is an intervention, not a correlation, and that is what makes it worth
reading.** 178/179 is a large effect from a single manipulation of decoder
state. If the arms were otherwise identical it is strong evidence that something
about the INITIAL transducer state, not the acoustics, was holding those
utterances at blank.

**Not established by what was reported:**

- **Which token was forced.** "Force one token" and "force any token" are
  different claims by a wide margin. Only the second would mean the effect is a
  state transition rather than linguistic information, and that is exactly the
  distinction worth having.
- **What `0.18 vs 0.82` and `0.96 -> 0.47` are.** Softmax probabilities or
  logits, on which frames, aggregated how. Our own margins are logit
  differences; the two are not comparable without saying so.
- **That tokenizer fragmentation (~3 fragments/word) causes it.** Plausible and
  untested: it would need fragmentation correlated against empty rate and
  first-emission margin, controlling for language, domain and duration.
- **That duration is not a contributor.** The cited numbers can show that short
  duration is not SUFFICIENT to explain the failures. That is not the same as
  showing it does not contribute.
- **That the other model "has the same uncertainty underneath".** Output shape
  does not reveal a different architecture's internal distribution. If it is
  CTC or attention the manifestation of uncertainty is structurally different.
- **That FastEmit 0.02 fixes it.** FastEmit is designed to push a transducer
  toward earlier emission, so it is COHERENT with the diagnosis — which is a
  reason to measure first-emission after the fine-tune, not a reason to assume
  it. Final WER would not show this at all.

## Why it matters here, and where our evidence already points the same way

Q-2 left one observation we deliberately refused to build on:

> 2 of the 3 STABLE-WRONG tokens are the bare SentencePiece word mark `▁`,
> which is not a lexical token at all. With n=3 that is a hypothesis for a
> larger corpus, not a design.

The other report independently says the token involved is, most of the time,
"not even a word — the marker that just means a word starts here". **Two small,
independent observations on the same model family pointing at the same
SentencePiece boundary token near the first emission event.** That raises the
priority of looking. It proves nothing: ours is n=3, and we do not know how
theirs was counted.

**Our regime is NOT theirs, and this must not be blurred.** In our 21-utterance
corpus there are **no empty transcripts**. Our symptom is 0.50-0.97 s of speech
and 2-13 blank steps before the FIRST non-blank (R-2/R-3), not a failure to emit
at all. So we cannot reproduce their headline result: we have no empty
population. The question we can ask is the weaker and more interesting one —
**is part of our 0.7 s a state barrier rather than missing acoustic evidence?**
If it is, injecting state should reduce the NUMBER OF BLANK DECISIONS before the
first natural non-blank. If it is not, the 0.7 s stays attributed to evidence,
which is itself a result worth having.

## The hook already exists, exactly

`src/decoder.c`:

- `pred_step(dec, s, token)` is the ONLY mutator of predictor state (LSTM `h`/`c`,
  the cached predictor output `s->g`, `last_token`).
- It is called exactly twice in the streaming loop: once as SOS with `dec->blank`
  (line 256, zero state), and once per EMITTED token (line 326, with the comment
  "state advances only on emit"). Blank does not advance it.
- Emission and state advance are two separate statements:
  `tokens[n_out++] = best;` then `pred_step(dec, s, best);`.

**So "perturb the predictor without publishing" is `pred_step` without the
`tokens[n_out++]` line.** No redesign, one diagnostic hook, default OFF.

`s->g` is added to the encoder frame before the ReLU that feeds the joint, so a
state change moves every subsequent logit immediately — which is why the effect,
if it exists, should be visible within one or two decisions.

The intervention point can be pinned to an ABSOLUTE ENCODER FRAME: the trace
already prints `frame=%ld` as `s->t_abs + t + b`. Every arm is bit-identical up
to the injection, so a frame chosen from the baseline trace is the same point in
every arm by construction — no per-arm favourable choice is possible.

## R-9 first: the trace cannot answer the `▁` question yet

Today `MYNAH_ASR_TRACE_RNNT` prints `blank`, `best_nonblank` (id and score),
`margin` and the choice. `best_excluding` returns the best NON-BLANK, which may
itself be `▁`. To test the boundary-token hypothesis the trace must carry,
per frame: **P(blank), P(`▁`), P(best LEXICAL non-blank), with token ids and
ranks** — so that the sequence

    blank >> ▁ > lexical   ->   blank ~= ▁ >> lexical   ->   transition   ->   lexical sharpens

can be seen if it is there, and ruled out if it is not. `▁`'s id comes from the
pack's `tokens.json`, the same map `tools/eval/rnnt_earliness.py` already loads.

Without R-9 there is no way to choose arm E's token honestly or to read the
result.

## R-8: the intervention matrix

Diagnostic only, default OFF, development host only, **no Axion**. It is not an
optimisation and must not be described as one. The forced token is NEVER
published to a client; it perturbs predictor state and nothing else.

| arm | injected at the frozen point | what it isolates |
|---|---|---|
| A | nothing (baseline greedy) | the reference trajectory |
| B | the current best non-blank | the "publish the runner-up" state WITHOUT publishing it |
| C | a reference-compatible first token (oracle, diagnostic only) | does correct linguistic history help |
| D | a deliberately wrong lexical token | does wrongness hurt, or is any token enough |
| E | the bare word mark `▁` | is the boundary token a structural primer |
| F | a fixed unrelated non-blank control | does any non-blank unlock |
| **G** | **`dec->blank` itself** | **the tightest control: an extra `pred_step` with zero lexical content. Blank normally never advances the state, so this separates "a state transition" from "a non-blank token".** |

G is not in the original proposal and is the arm that decides the headline
question. If G unlocks as much as F, the mechanism is the state transition
itself and no token carries the effect.

**Second axis, also pre-registered: the injection point ladder.** Inject at
k = 1, 2 and 4 decisions before the baseline crossing. If the mechanism is a
barrier, injecting earlier should unlock earlier. If it is evidence, injecting
before the evidence exists should do nothing. Dose-response separates the two
hypotheses more sharply than any single point.

**Run it on utterances that already work normally, not only on slow ones.** An
intervention tested only where the baseline is bad measures regression to the
mean.

### What every arm records

- blank logit / margin trajectory after the injection;
- **number of blank decisions until the next NATURAL non-blank** (the primary
  outcome: it is a count, not a wall time, and cannot drift with the host);
- audio consumed until that next natural non-blank;
- whether sustained decoding starts, and the subsequent token rate;
- final transcript CER/WER **computed on the visible output only**, with the
  injected token removed from scoring — a hidden token still conditions
  everything after it, which is the point;
- divergence from the baseline transcript after the injection point;
- empty / non-empty outcome, so the comparison with the other report is
  possible if we ever have an empty population.

### The comparisons, and what each would mean

| comparison | reading |
|---|---|
| C vs D/F | does token CORRECTNESS matter, or only that something was injected |
| B vs C | is the model's own runner-up as good as the right answer |
| F/G vs A | can a wrong or contentless transition still unlock correct decoding |
| **G vs F** | **is it the token at all, or just the `pred_step`** |
| E vs lexical arms | is `▁` special, as two independent small observations hint |
| k-ladder | barrier (earlier injection, earlier unlock) vs evidence (no effect before the evidence exists) |

If nothing materially changes the next natural emission, **our first-word
latency is not explained by this mechanism** and the R-3 attribution — the model
is waiting for acoustic evidence — stands strengthened, not weakened. That is a
good outcome too.

### Two results, never inferred from each other

Record separately whether the intervention improves **subsequent decoding** and
whether it improves **final recognition quality**. A state unlock that produces
fluent wrong text is a finding, not a win. `tools/eval/partial_quality.py` is
the instrument for the second; the trace is the instrument for the first.

## Q-5: the metric set that would let us judge a fine-tuned pack

If a fine-tuned Nemotron ever lands here, CER alone will not show whether the
fine-tune moved THIS dynamic. The quality gate should also carry:

- **empty-transcript rate** (trivial to add to `partial_quality_summary`; today
  it would read 0/21 and that zero is itself worth recording);
- audio and speech to first non-blank;
- blank/non-blank margin at onset;
- first lexical token correctness (already there).

Then a FastEmit-trained checkpoint can be compared on the mechanism it claims to
change, instead of on a WER that would hide it.

## Scope discipline

Mac only. Behind a diagnostic flag, default OFF, registered in `src/flags.c`.
Production decoding unchanged. No forced token ever reaches a client. This is a
causal diagnostic; it does not become a serving feature without its own gate,
and Q-2 already showed that publishing early on persistence alone is a defect,
not a trade.


---

# R-8 RESULT (2026-09-22, development host, 21 clips, int8 unless stated)

Model `models_local/nemotron-3.5-asr-streaming-0.6b`, preset `[56,3]`, corpus
`samples/manifest.json` as the only oracle. Evidence:
`.work/evidence/r8-A-2026-09-22/`. No cloud machine was used.

**Baseline frozen and verified first.** All 21 clips reproduce the 2026-09-21
first-natural-token FRAME exactly. (A first comparison said every clip differed;
that was a parser fault of mine on the OLD trace format, whose `emit` line comes
AFTER the frames of its own step, so the audio it reported was one step early.
The frame index is unambiguous and matches 21/21.)

## FACT — the extreme blank regime is not int8, and not the predictor

R-9 now records, per decision: absolute frame, audio consumed, `|enc|`,
`|joint|`, predictor SOS/MOVED, post-intervention flag, and blank / word-mark /
best-lexical / best-non-blank each with id, logit and 1-based rank. The word
mark is resolved through `mynah_asr_tok_find` from the pack's own `tokens.json`
(id 2 here) and is never hard-coded.

**Not a quantisation artefact.** f32 produces the extreme values on the same
frames as int8: -969 / -972 / -947 against -946 / -959 / -946 on frames 2, 4, 5
of the same clip. That branch is closed.

**It is born in the encoder.** `|enc|` is ~2.7 on ordinary frames and ~21 on the
extreme ones, and `|joint|` follows (0.89 -> 17.2). Over six clips:

| audio | pred | cache | frames | \|enc\| median | >10 |
|---|---|---|---|---|---|
| silence | SOS | filling | 120 | **19.74** | 68.3 % |
| SPEECH | SOS | filling | 71 | 4.23 | 32.4 % |
| SPEECH | MOVED | FULL | 464 | 3.68 | **0.0 %** |

The encoder cannot see the predictor -- they are separate networks -- so the
"extreme ⟺ SOS" association reported from the saved traces was MEDIATED, not
direct. **The earlier note over-attributed it to the predictor; this corrects
it.**

## CAUSAL RESULT — holding the encoder fixed, the predictor does not explain it

The observational 2x2x2 could never fill the cell where the predictor has moved
and the encoder is still in its high-norm regime. The injection fills it:
inject at frame 4, inside the leading silence, and look only at frames with
`|enc| > 10`.

| arm | window | frames | \|enc\| med | \|joint\| med | \|blank\| med | >100 |
|---|---|---|---|---|---|---|
| A | pre | 115 | 20.70 | 17.00 | 953.5 | 100 % |
| G@4 | **post** | 109 | 20.65 | 19.48 | **1295.0** | **100 %** |

**Moving the predictor state does not remove the extreme regime. It slightly
amplifies it** -- consistent with `|g|` falling from 15.46 to 2.28, so the ReLU
suppresses less and the joint grows. An earlier reading of mine ("0 % extreme
after the injection") was the `|enc|` drop confound: injecting near the crossing
puts the post window past the leading silence. Held fixed, the effect vanishes.

## CAUSAL RESULT — the injection does not unlock emission

Every arm injected at the frozen point k=2 decisions before the baseline
crossing, mapped to an absolute encoder frame. The injected token is never
published and never counted as a natural emission.

| arm | injected | n | Δ blanks med | range | audio earlier/same/later | CER A | CER arm | text = A | first word correct |
|---|---|---|---|---|---|---|---|---|---|
| G | `blank` | 21 | **+4.0** | -2..+14 | 1 / 8 / 12 | 0.0286 | 0.0662 | 0/21 | — |
| B | best non-blank | 21 | +0.0 | -2..+5 | 7 / 11 / 3 | 0.0286 | 0.0370 | 3/21 | — |
| E | bare `▁` | 21 | **-2.0** | -2..+0 | **14 / 7 / 0** | 0.0286 | **0.0214** | 6/21 | **10/21** |
| F' | best lexical | 21 | +1.0 | -2..+5 | 5 / 12 / 4 | 0.0286 | 0.0545 | 0/21 | — |
| A | (baseline) | 21 | — | — | — | 0.0286 | — | — | **16/21** |

**G delays.** Median +4 blank decisions, and it changes the transcript in 21 of
21 while roughly doubling median CER.

**The post-first-token control says it is not special.** The same hidden
`pred_step(blank)` after the first natural token delays too: +2.0 at frame+4
(7 of 7 later) and +4.0 at frame+8 (11 of 11 later). An arbitrary predictor
perturbation disrupts decoding wherever it is applied. *Limit of the seam:* the
control was served on only 7 and 11 of 21 clips, because after the first token
the requested frame often falls inside a block that emits earlier and the loop
never reaches it. That is a limitation, not a result.

**Token identity matters, and orders the arms**: G +4, F' +1, B 0, E -2. One
`pred_step` is not equivalent to another.

## The one arm that advances emission fails the axis Q-2 said to watch

Arm E -- injecting the bare word mark -- is the only arm that never delays
(0 of 21 later) and the only one whose median CER improves. Δ blanks -2 means
the natural token arrives **at the injection frame itself**: give the model the
word-start marker and the word piece follows immediately.

**But the first published word is right in 10 of 21, against 16 of 21 in the
baseline.** Per clip: `les` -> `l`, `il` -> `i`, `w` -> `v`, `o` -> `lično`,
`в` -> `า`. The injected `▁` is suppressed from the output by design, so the
word boundary it represents never reaches the client and the first piece arrives
without it. The median CER improvement is partly the normaliser collapsing
whitespace; the first-token axis, which Q-2 established as the one that matters
for a transcript that revises nothing, moves the wrong way.

**And at a DEPLOYABLE point it collapses.** E at the fixed frame 4 -- no
baseline knowledge, identical definition on every clip -- gives Δ blanks -14
and 21 of 21 earlier, at median CER **0.0575**, double the baseline's 0.0286.
Forcing a word start before the evidence exists makes the model speak early and
wrong, which is the Q-2 finding again in a new costume.

**Oracle-relative and deployable are not the same experiment.** k=2 uses the
baseline crossing, i.e. future knowledge. Nothing in the k-relative rows may be
read as a proposed algorithm.

## INTERPRETATION

The first-emission-lock hypothesis, as it applies to THIS checkpoint and THIS
configuration, is **rejected**. Moving the predictor state does not release a
barrier: it delays emission, it degrades the transcript, and it does the same
thing after the first token, which is the signature of corrupting predictor
history rather than unlocking anything.

The extreme blank logits before the first word are an **encoder** phenomenon --
high-norm output while the stream opens on silence with a cold cache -- made
visible, not caused, by the suppressive SOS predictor output. They are a
signature, not a cause. R-3's attribution stands: the wait is the model wanting
acoustic evidence.

The word-mark result is the one thing worth keeping open, and it is a
TOKENIZER/predictor interaction, not a latency lever: the model appears to be
waiting to emit the word-start marker, and supplying it pulls the word piece
forward. That is interesting about the checkpoint. It is not a serving change:
the two ways of using it both fail, one on first-word correctness and one on CER.

## STILL UNKNOWN

- Why the encoder's output norm is ~7x larger on early silent frames with a cold
  cache. Not investigated; it may be benign (an unnormalised region of the
  representation) or it may be worth a look on its own.
- Whether arms C and D (oracle-correct and deliberately-wrong lexical tokens)
  separate further. They were not run: G, B, E and F' already answer "does
  identity matter" (yes) and "does any transition unlock" (no), and C/D need
  frozen selection rules written before results.
- Whether the post-first-token control holds on the 10-14 clips where the seam
  could not serve it.
- Whether a checkpoint trained with FastEmit behaves differently here. Q-5's
  metric set exists for exactly that comparison and has not been used.

## Not done, deliberately

No forced first token in serving, no blank bias, no `▁` publication rule, no
predictor priming, no dynamic lookahead. R-8 was a mechanism experiment and it
returned a negative result on its main hypothesis; the engineering choice is the
user's to make from this evidence.
