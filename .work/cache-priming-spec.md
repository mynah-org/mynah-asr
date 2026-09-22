# What "prefilling the encoder cache" would actually mean

Task: R-12

**Nothing here has been run.** This is the inspection asked for before the
experiment is ever approved: what a prefill would do to the model's state, which
variants are legitimate, and whether any of them could be a production
optimisation rather than an intervention.

## The state a prefill would have to touch, from the code

| state | where | at stream open | read by |
|---|---|---|---|
| `k_cache`, `v_cache` | `[n_layers, left, d_model]`, `calloc` | all zeros | only the first `cache_valid` rows: `kk[0..valid)` |
| `cache_valid` | `mynah_asr_enc_stream_step` | 0, then `+= q`, saturating at `left = 56` | sets `K = valid + Q` |
| `conv_cache` | `[n_layers, conv_k-1, d_model]`, `calloc` | zeros | `stream_conv_module`, every step |
| `ss.cache[s]` | `[C_in, F]` per stage | zeros | `mynah_asr_ss_stream_step` |
| `ss.first` | flag | 1 | the one-off init pad of the first chunk |

Two consequences that decide the whole question:

1. **The zeros are already there and are already NOT attended to.** The caches
   are `calloc`'d, but the attention reads only `[0, valid)`. So a prefill is not
   "putting something in an empty buffer" — it is **raising `cache_valid`**, which
   is a claim about how much history exists.
2. **`cache_valid` is inside the positional arithmetic.** `K = valid + Q`, and
   `rel_shift` computes `base = K - 1 - valid - t`; the rel-pos table is indexed
   by `(Kmax - K)`. Raising `valid` therefore changes which relative-position
   rows every query reads, immediately and for every layer. It is not a neutral
   bookkeeping field.

## The four variants, and what each would violate

**V1 — zero-fill: set `cache_valid = 56` without processing anything.**
The attention then attends to 56 zero key/value vectors as if they were speech.
The conv and subsampling caches stay genuinely empty, so the layers disagree
about how much history exists. The relative positions assert 4.48 s of context
that never occurred. **Out-of-distribution on three counts at once.** It would
measure the attention's sensitivity to `valid`; it would say nothing about
whether a warm cache helps. Not a candidate.

**V2 — prime with the clip's own leading silence.**
Semantically clean, because it is just "the stream started earlier" — but it
changes the AUDIO as well as the cache, so it cannot separate the two. And a
server does not have the audio from before the stream opened. Useful only as an
upper bound, and it must be labelled as one.

**V3 — prime with unrelated room tone, then feed the clip untouched.**
No future speech leaks, no frame is duplicated, the positional semantics are
honest because those frames really were processed, and the target audio is
byte-identical to the baseline. **This is the only variant that is both honest
and informative.** It is still an intervention: the cache holds another
recording's noise floor, which is not what the model expects 4.48 s into an
utterance. Whether that matters is a measurement, not an assumption.

**V4 — duplicate the clip's own first frames.** Gives the model the same speech
twice and leaks future audio into the past. Excluded.

## Could V3 ever be a production optimisation?

The deployable form is "open every stream with a fixed silence preamble". Two
things have to be said about it, and the first is arithmetic, not opinion:

- **Filling the cache takes `left_ctx` encoder frames = 4480 ms of audio**
  (R-10, and it is preset-invariant). Fed in real time that is self-defeating:
  4.48 s of added latency to save at most 0.7 s.
- Fed **faster than real time at stream open** it costs compute, not latency:
  56 encoder frames is 14 steps at `[56,3]`, about 180 ms of CPU per stream at
  the measured warm step cost. That is a real per-stream admission cost and it
  lands on exactly the resource the cadence law `T(B) = a + b·B ≤ ρ·P` is about.
  It would have to be priced against capacity, not against latency alone.

So V3 is not obviously self-defeating, but it is not free either, and the thing
it buys is unknown.

## What must be proven BEFORE it is run

1. the target audio is byte-identical to the baseline run — asserted, not assumed;
2. no frame of the target audio is processed twice;
3. `cache_valid`, the conv cache and the subsampling cache advance **together**,
   by the same number of frames, so no layer believes in a different amount of
   history than another;
4. the relative-position window actually read is recorded before and after, so a
   change in positional semantics is observed rather than discovered later;
5. the resulting encoder state is compared against the state the same stream
   reaches naturally at the same `cache_valid` — if they are far apart, the
   prefill has created an out-of-distribution state and the result is about that,
   not about warm caches;
6. quality is gated by `tools/eval/partial_quality.py` on all six axes, because
   an earlier first word bought with a wrong first word is the Q-2 result again.

**A zero-filled or silence-filled cache is an intervention.** It becomes a
candidate optimisation only if 1–6 hold and the quality gate is green, and even
then the capacity cost above has to be paid for explicitly.
