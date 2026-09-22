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
