# First-partial responsiveness: the other SLO

Status: OPEN (written 2026-09-21 on the development host, from measurements
taken on the Axion the same day; no part of the decomposition below has been
performed yet)

Task: R-1 .. R-6
Question: an interactive streaming ASR has two service levels, not one. Steady
state is emission lag, backlog, losses and throughput, and today's campaign
moved it a long way. Responsiveness is how long after someone starts speaking
the first word appears, and nothing has moved it because nothing has explained
it. What physically sets the floor, and how close can we get to the model's
earliest possible emission?

## Known facts

- **The campaign that opened this question is frozen as F29** in
  `docs/serving-findings.md`. Nothing below may be folded back into F29: that
  entry records the capacity campaign, this one is the responsiveness mission.
- **The two are independent, and the numbers prove it.** On the Axion,
  2026-09-21, arm C at C=60: emission lag p95 **223 ms** while TTFP p95 was
  **2628 ms**. Once a stream is in flight the server is reactive; the wait is
  in front of the first word, not between words.
- **There is a floor that load does not explain.** F25 recorded TTFP p95
  1544 ms at C=8 and 1657 ms at C=32 over 600 s soaks. Today's short probes
  read 2.5–4.5 s at every concurrency from 40 to 120, including rungs whose
  cadence lines were comfortable. Load makes it worse; load did not create it.
- **The previously recorded explanation is wrong.** "TTFP is leading silence"
  was struck in F25 for the corpus actually used, whose onsets are 0.02 / 0.18 /
  0.64 s. Measured again today with `tools/bench/clip_onset.py`, the sample bank
  starts speaking at 0.18–0.95 s, stable across a 14 dB range of the detector's
  margin. Silence is not seconds of the number.
- **TTFP now decomposes, and the instrument exists** (commit `99c518b`):
  `ttfp_from_onset_ms`, `first_delta_audio_s` (audio the server had CONSUMED
  when it emitted the first word) and `first_delta_lag_ms` (the server's own
  lateness on that frame). None of these has been read on a real run yet.
- Nemotron streams at lookahead 3, so the encoder chunk period is
  `(3+1) x 80 ms = 320 ms`. Some wait is arithmetic.

## Unknowns

1. What `first_delta_audio_s` actually is on this model. If the RNNT commits
   only after ~1.2 s of audio, most of the floor is the model and the search
   ends there. If it commits after ~0.4 s, a second of the floor is ours.
2. Where the rest of the path spends its milliseconds. No timestamp exists
   today between "audio arrived" and "the client saw text".
3. Whether anything in the serving path deliberately waits for a *better*
   partial, and whether an earlier unstable partial could be emitted and
   revised. Nemotron emits every delta as `final:true` today, so there is no
   revision channel and adding one is a protocol change, not a tuning knob.

## Files and functions to inspect (not yet inspected for this question)

`src/mynah_asr.c` (`mynah_asr_stream_feed`, `need_samples`, the mel pull and
the decode call), `src/features.c` (frame buffering before the first full
window), `src/subsampling.c` (the causal padding and how many frames the first
chunk needs), `src/encoder.c` (`cache_valid` priming: the first ~18 steps run
with a partially filled left cache), `src/decoder.c` (`greedy_decode_scratch`,
`max_symbols_per_step`, the blank policy that decides whether a step emits),
`server/sched.c` (slot admission, the batch collection window, when a delta is
queued), `server/stream_out.c` (coalescing and the writer's wake), and
`tools/bench/stream_load.py` (what the client calls t=0).

## The plan, in the order that cannot mislead

No optimisation before the C=1 waterfall reconciles numerically. The fastest
route to the real optimisation is, paradoxically, to refuse to start one.

**R-1. Prove the timestamp semantics before optimising anything.** From code,
not from names: source location and clock domain for `stream_open`,
`first_audio_sent`, `first_audio_received`, first speech / first non-silence if
it is observable at all, first features available, first encoder-eligible
chunk, first encoder admission, first encoder start and end, first
decoder/RNNT opportunity, first non-blank token, first non-empty partial
created, queued, written to the socket, and received by the harness client.
`sends[0][0]` is the completion of the first frame send; whether that is "audio
arrived at the server" is an assumption nobody has checked. Name three metrics
and keep them apart:

    TTFP-open   = first client-visible partial - stream open
    TTFP-audio  = first client-visible partial - first meaningful audio arrival
    TTFP-speech = first client-visible partial - speech onset

TTFP-speech is the UX metric. If speech onset is not currently measurable
correctly, say so and instrument it; never substitute stream-open silently.

**R-2. C=1 first, and explain every millisecond.** One stream, one clip. No
batching excuse, no contention excuse, no opening herd. A waterfall across
speech onset -> feature buffering -> model-required context and lookahead ->
scheduler wait -> encoder -> RNNT -> emission policy -> server output buffering
-> socket -> client, whose components reconcile with the measured first-partial
latency. Do not stop at "Nemotron needs context": measure how much it consumes.
Two explicit quantities:

    audio_seconds_consumed_at_first_nonblank = X
    wall_time_of_first_nonblank              = Y

If the first non-blank happens after 400 ms of audio but the client sees text
at 1.5 s, ~1.1 s is ours. If it cannot physically happen before ~1.3 s of
audio, that is a very different lower bound and the search ends much earlier.

**R-3. Classify every component before proposing anything.** REQUIRED BY MODEL /
REQUIRED BY CURRENT DECODING SEMANTICS / SERVING OVERHEAD / BENCHMARK ARTIFACT /
UNKNOWN. Audit by name: frontend window and chunk accumulation, subsampling,
lookahead and right context, cache priming, minimum encoder chunk, RNNT blank
behaviour, decoder commit and emission thresholds, partial suppression and
coalescing, scheduler cadence, batch collection window, output queues,
WebSocket buffering and flushing, the client read loop, and the harness ramp.

**R-4. `publication_delay`, as a first-class metric.** This is the discriminant:

    publication_delay = client_first_partial - model_first_hypothesis

Two readings, two different investigations:

    onset 0 ms, model hypothesis 450 ms, client 2600 ms
        -> ~2.15 s to find AFTER the model already had something to say:
           attack the serving and output path
    onset 0 ms, model hypothesis 2300 ms, client 2600 ms
        -> serving is not the culprit: chunking, lookahead and RNNT policy,
           and how much of those 2.3 s is actually necessary

**R-5. Optimise causally, one mechanism at a time.** Only after R-1..R-4 name
the dominant term. Hypotheses, not instructions: less initial accumulation, an
earlier first encoder chunk, bypassing the batch window for a stream that has
never emitted, prioritising first-token work, publishing an earlier unstable
partial and revising it, removing output coalescing, flushing the first partial
immediately, special first-chunk scheduling. Each one: `before -> mechanism ->
after`, same audio and build but the switch, transcript gate green. Never buy
TTFP with recognition quality, transcript semantics, established streams or
steady-state capacity. Shortening the model's required context is a numerics
change needing its own quality qualification, never a latency tweak.

**R-6. Load dependence last.** C1 -> C8 -> C32 -> near-knee, then synchronised
opening on its own. Report together, so responsiveness cannot be bought with
capacity: TTFP-speech p50/p95/p99, TTFP-open p50/p95, model first-hypothesis
latency, publication delay, steady emission lag, backlog, loss and rejection,
CER/WER, audio per wall second.

Excluding TTFP from the knee-search predicate (`--ignore-line`, commit
`c4d0491`) was the right call for a capacity bisection and must never turn into
its absence from production acceptance. Two SLO families, named and kept apart:

    responsiveness: first-useful-partial latency (open-relative AND
                    speech-relative), reported in every acceptance run
    steady state:   emission lag, backlog, losses, throughput

F29 is the proof that one can move a very long way without touching the other.

## Acceptance philosophy

No target invented from the current implementation. First establish the
physical and model lower bound, then set an aggressive target close to it. The
product behaviour wanted is: the user starts speaking, useful text appears as
soon as the model can safely provide it, the partial evolves while speech
continues, and the final result may arrive later. A 2-4 second first visible
partial makes an otherwise fast streaming ASR feel non-interactive.

Work offline on the development host wherever possible. No paid box until there
is a specific measurement that cannot be answered locally and its script is
already written.

## Conclusion

Not reached: nothing here has been measured yet. What today establishes is that
the question is separable and worth separating, because the steady-state
numbers moved by nearly a factor of two while the first-word wait did not move
at all.

## Next action

R-1 and R-2 on the development host, at C=1, with no paid machine running.
