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

## R-1 RESULT (2026-09-21, from code on the development host)

Every timestamp on the path, read from the source rather than from its name.
Nothing below was measured on a server; it is what the code does.

### The two clocks, and the one that was wrong

The server stamps everything with `mynah_asr_now()` (`server/slot.c:99`):
`clock_gettime(CLOCK_MONOTONIC)`. The harness stamped everything with
`time.monotonic()`. On Linux those are the same call. **On macOS they are not**:
`time.get_clock_info("monotonic")` reports `mach_absolute_time()`, which
excludes the time the machine spent asleep, while Darwin's `CLOCK_MONOTONIC`
includes it. Measured on this host, three samples, the C reading never landed
inside the Python window and the offset was **4.63 days**.

Durations inside one process are unaffected, so no number in F29 changes. A
client mark minus a server mark would have been nonsense, and R-2 is built
entirely out of such differences. `tools/bench/stream_load.py` now takes every
mark from `time.clock_gettime(time.CLOCK_MONOTONIC)` (`mono()`), which was
verified to land inside a window bracketing a C `clock_gettime` call, three
times out of three. On the box this is a no-op.

### What each mark actually means

| name | where | clock | what it really is |
|---|---|---|---|
| `t_start` | `stream_load.py`, `run_utterance` | client | **before** `ws_connect`: includes TCP connect and the WebSocket upgrade |
| `sends[i][0]` | same, after `ws_send` returns | client | the write of frame `i` **completed locally**; not server receipt |
| `sends[i][1]` | same | — | cumulative audio seconds handed to the socket, `(i+1)*frame/32000` |
| arrival | `server/main.c:968`, `last_activity` | server | after the whole WS frame was read **and unmasked**: a faithful "audio received" |
| per-sample arrival | `slot_record_arrival_locked`, `server/slot.c:249` | server | the same instant, recorded against the ring index so a chunk can be dated later |
| ready | `slot_refresh_ready_locked`, `server/slot.c:63` | server | the rising edge of "this slot holds a whole chunk"; re-stamped by `slot_take` when a whole chunk is still left behind |
| selected | `s->t_last_step`, `server/sched.c` `sched_stage` | server | the chunk was staged into the batch |
| model start | `t0` in `sched_step_batch` | server | the whole ready set enters the model together |
| result | `now` in `sched_on_result` | server | the model's callback fired — **before** framing, queueing and the socket |
| `lag_ms` on the wire | `server/sched.c:377` | server | `result − arrival of the last sample consumed`. It stops at the callback: framing, the output ring and the socket write are **not** in it |
| `audio_s` / `t1` on the wire | `src/mynah_asr.c`, `stream_decode_emit` | — | `samples_fed / sample_rate`: audio the model had **consumed** when it produced this text |
| `ev["t"]` | `stream_load.py`, reader thread | client | after `ws_recv` returned a complete message, before `json.loads` |

### The three TTFPs, as implemented

    ttfb_ms            = first ANY frame            − sends[0][0]
    ttfp_ms            = first delta with non-empty text − sends[0][0]
    ttfp_from_onset_ms = the same first delta − send time of the frame carrying the onset sample

So today's TTFP is **neither** open-relative nor speech-relative: it is relative
to the completion of the first audio write, which excludes connect and upgrade
and includes everything after. That is a defensible zero, and it is not what the
name says. `ttfp_from_onset_ms` is speech-relative but measured from when the
CLIENT sent the onset, which equals real time only while `paced` is true —
`analyze_utterance` already records `paced`, so the condition is checkable per
run rather than assumed.

A delta is emitted only when `total > s->chars_emitted`: a step that produces
only blanks emits nothing. Therefore `first_delta_audio_s`, already recorded by
the harness and never yet read, **is** `audio_seconds_consumed_at_first_nonblank`.

### What is NOT instrumented

Between "audio received" and "client sees text" the server publishes exactly
three instants: arrival, the result callback, and `lag_ms` which is their
difference. There is no timestamp for first feature frame, first encoder
admission, first encoder completion, first decoder emission, the framing, the
enqueue, or the socket write. **`publication_delay` (R-4) cannot be computed
from anything the server emits today.** It has to be instrumented.

`s->t_first_delta` (`server/slot.c:107`, set at `server/sched.c:403`) is
written on every stream and **read by nothing**. The server already knows when
it first spoke and throws it away.

### What the pipeline is NOT doing

Three suspects checked and cleared, so R-2 does not have to carry them:

- **The batch window does not delay a first chunk.** `sched_collect`
  (`server/sched.c:576`) sets `exempt` and breaks out of the wait when any live
  slot has `steps == 0`. `steps` is zeroed on slot open and on reset, and the
  harness opens a new socket per utterance, so every utterance's first chunk is
  exempt. The comment claims it; the code does it.
- **The scheduler has no tick.** Its idle wait is a plain `pthread_cond_wait`
  with no timeout (`server/sched.c:1054`), woken by the push doorbell.
- **Nagle is off on both sides** (`server/main.c:1190`, `stream_load.py`), and
  the writer signals its condvar on enqueue. No 40 ms ACK wait is hiding here.

### The model floor, derived (to be confirmed by measurement in R-2)

From `mynah.json`: `hop 160`, `n_fft 512`, `sub_factor 8`, default preset
`[56, 3]`. From `mynah_asr_enc_stream_need` (`src/encoder.c:729`) the first
chunk wants `1 + sub*right = 25` mel frames against `sub*(right+1) = 32`
afterwards, and `mynah_asr_mel_stream_samples_until` (`src/features.c:283`)
turns frame 24 into `24*160 + 512/2 = 4096` samples:

    first chunk      4096 samples = 256.0 ms of audio
    every chunk after                320.0 ms  (= P, the cadence period)

**The first admission needs 256 ms of audio, not 320.** Whether the RNNT
commits a character on that first step is exactly the open question, and
`first_delta_audio_s` answers it from a run that has already been instrumented.

## R-2 RESULT (2026-09-21, C=1, development host)

Eight clips, one stream at a time, each its own connection, onsets 0.15-3.45 s.
`MYNAH_ASR_TRACE_TTFP=14` for the server marks, `stream_load.py` for the client
marks, both on CLOCK_MONOTONIC. **DIAGNOSTIC (macOS, Accelerate, 8 threads): the
mechanism transfers, the milliseconds do not.** Evidence:
`.work/evidence/ttfp-c1-2026-09-21/`.

**The pairing is proved, not assumed.** For every utterance the server's marks
fall inside the client's own window, in order: client first write -> server
first audio (0.03-0.07 ms) -> first result -> first queued -> first send ->
client first partial (0.07-0.17 ms). The residual of the waterfall is an
algebraic identity and proves nothing; this ordering is the validation.

### Where the wait goes

| span | mean ms | share | owner |
|---|---|---|---|
| further audio the RNNT wanted, at 1x | 1350.0 | 68.6 % | MODEL |
| encoder+RNNT wall (incl. one cold step) | 371.1 | 18.9 % | MODEL |
| the first chunk's 256 ms of audio | 245.5 | 12.5 % | MODEL |
| framing + output ring + socket write | **0.07** | 0.0 % | SERVING |
| loopback, both directions | 0.1 | 0.0 % | CLIENT |

    publication_delay = client_first_partial - server_first_nonblank
                      = 0.12 .. 0.23 ms over eight utterances

**This is branch B.** Not A: the serving path between the model having something
to say and the client seeing it is two tenths of a millisecond. There is no hole
to find there.

### Per utterance

| clip | onset | TTFP-open | TTFP-speech | audio at 1st | speech at 1st | blank steps |
|---|---|---|---|---|---|---|
| en/1521 | 0.18 | 1533 | 1366 | 0.896 | 0.716 | 2 |
| en/1534 | 0.59 | 1572 | 986 | 1.536 | 0.946 | 4 |
| fr/1521 | 0.15 | 932 | 785 | 0.896 | 0.746 | 2 |
| de/1534 | 0.52 | 1249 | 742 | 1.216 | 0.696 | 3 |
| it/1521 | 0.95 | 1888 | 939 | 1.856 | 0.906 | 5 |
| es/1534 | 1.04 | 1574 | 546 | 1.536 | 0.496 | 4 |
| de/1521 | 3.45 | 4465 | 1019 | 4.416 | 0.966 | 13 |
| it/1534 | 1.85 | 2521 | 676 | 2.496 | 0.646 | 7 |

**FACT.** `audio_seconds_consumed_at_first_nonblank` is 0.896-4.416 s, and after
subtracting the leading silence it is **0.496-0.966 s of speech**, two to three
chunks. The RNNT emitted nothing on 2 to 13 steps first.

**FACT.** In seven of the eight, `TTFP-open` equals the audio consumed at the
first non-blank plus one step wall, to within a few ms: 1536+36=1572, 896+36=932,
1216+33=1249, 1856+32=1888, 1536+38=1574, 4416+49=4465, 2496+25=2521. The eighth
is the first stream the server ever served (below).

**FACT — the pipeline is not starved, it is waiting for audio.** Between steps
it idles 270-291 ms and then computes for 30-46 ms, against a 320 ms chunk
period. Nothing queues behind anything.

    en/1534   step  consumed_s  idle_before_ms  step_ms  emitted
                 1       0.256           247.0     36.6        0
                 2       0.576           284.8     46.1        0
                 3       0.896           270.6     34.0        0
                 4       1.216           282.8     35.9        0
                 5       1.536           288.9     45.1        1

**FACT — the first step a fresh server ever runs costs 1236 ms** against 30-46 ms
warm, and it blocks the two chunks behind it. That is the whole of the first
utterance's anomaly (compute 1284 ms against 111-539 ms for the rest). The Axion
soaks measured warm servers, so this is NOT the 2.6 s seen there; it is a real
first-request cost of its own, and it has never been named.

### The buckets

**FACT (measured here).** Publication is 0.2 ms. The serving path owns none of
the wait at C=1. The first chunk needs 256 ms of audio. The model consumed
0.50-0.97 s of speech before its first non-blank. Steady steps cost 30-46 ms
against a 320 ms period. A cold server pays 1236 ms on its first step.

**INFERENCE (follows from the above, not separately measured).** TTFP-speech at
C=1 is, to within one step wall, exactly the audio the RNNT consumes after
speech onset. **This locates the wait; it does not explain it.** A blank is an
observed output, not a root cause, and "the RNNT emitted blank" is not the same
claim as "the decoder deliberately waited". Which of the checkpoint, the
lookahead and cache semantics, the predictor/joiner, the blank policy,
`max_symbols_per_step` or our own publishing rule owns those steps is exactly
what R-3 has to separate, and nothing here decides it. The 2.6 s seen on the
Axion at C=60 is this floor plus load; how much of it is load is not
established by this run.

**UNKNOWN.** Whether 0.5-1.0 s of speech is inherent to this checkpoint or a
consequence of our blank policy, `max_symbols_per_step`, or the emit rule
`total > chars_emitted` — that is R-3 and it is a numerics question, not a
latency one. Whether the cold first step is allocation, page faults or a BLAS
first call. What any of these become under load (R-6).

**The largest removable component is not yet identified**, and the largest
component is not obviously removable: it is the model declining to commit.

## R-3 RESULT (2026-09-21, source inspection + local diagnostics)

Three clips traced frame by frame with `MYNAH_ASR_TRACE_RNNT=1` (onsets 0.18,
1.04, 3.45 s), on top of R-2's eight at chunk level. Axion stayed off.

### R-3B — there is no decoding policy to blame

**FACT, from source.** `mynah_asr_greedy_decode_scratch` (`src/decoder.c:190`)
is a plain greedy argmax: `argmax_bias(logits, head_b, V)`, blank if it wins.
There is **no blank threshold, no temperature, no suppression, no commit rule
and no confidence gate** anywhere on the path. `max_symbols_per_step` (10) is
checked only *after* an emission, so it cannot delay a first token.

**FACT — the publishing rule never fired.** The one suspect our side owned was
`total > s->chars_emitted` in `stream_decode_emit`: the detokeniser lifts a
language tag out of the text, so a step could decode a non-blank and publish
nothing. Instrumented, that case occurred **0 times in 3 clips**. Every token
the decoder produced became visible text.

**FACT — the first steps are STRONGLY blank, and the crossing is
NEAR-BOUNDARY.** In silence the blank margin is ~+100 (median +99 to +102 over
2, 12 and 43 silent frames). From speech onset it falls to ~+9 and decays to
the crossing:

    en/1521  onset 0.18s   +9.45 +7.72 +4.95 +4.47 +2.44  -1.56  TOKEN
    es/1534  onset 1.04s   +9.02 +9.76 +4.19 +2.55 +3.78  -2.61  TOKEN
    de/1521  onset 3.45s   +8.13 +8.34 +7.69 +7.91 +4.61 +4.74 +5.13 +5.83 +1.17  -0.45  TOKEN

The decay is not strictly monotonic. The crossing itself is a hair (-0.45 to
-2.61) and the frame before it is close (+1.17 to +3.78), but the five to eight
frames before *that* are not: +4 to +9. **Moving emission one frame earlier is a
small policy change; moving it three frames earlier is a large one.**

### R-3C — the lookahead is inside the number, not beside it

**FACT.** Preset `[56, 3]` means each encoder output attends 3 frames ahead, and
`q = right + 1 = 4` frames come out per step. So the decision taken "at encoder
frame f" consumed source audio through frame `f+3`: **240 ms more than the frame
index suggests**, and the token cannot exist earlier. It is then published
immediately (R-2: 0.2 ms).

This retires an attractive wrong reading. The first non-blank of en/1521 lands
on frame 8 (0.64 s by frame index) but is decided on audio through 0.96 s, which
is what the harness reports. **There is no publication delay to recover here.**
256 ms remains *earliest first-chunk eligibility only* and is NOT a model floor.

### R-3D — the wait tracks speech, not silence

**FACT.** Leading silence spans 0.18 to 3.45 s across the traced clips — a 19x
range — and the audio consumed after onset before the first non-blank is
**+0.46, +0.56, +0.79 s**. Over R-2's eight clips the same quantity is
**0.496 to 0.966 s**. It is approximately fixed after onset and does not track
the silence in front of it. Blank steps DO track silence (2 to 13) because
silence costs steps, not because it costs evidence.

**Not generalised further.** Eight FLEURS clips, read speech, six languages. A
corpus with breaths, hesitations or a quiet first word could behave differently
and nothing here says it would not.

### R-3E — cold start: UNKNOWN, and not reproducible on demand

**FACT.** Three fresh server processes, identical command, same clip: the first
model step cost **49.4 ms, 1010.6 ms, 43.8 ms**. The 1236 ms seen in R-2 is real
but **1 in 3, not deterministic**, which rules out the tidiest hypothesis: the
thread pool is built under `pthread_once` inside the first dispatch
(`src/threads.c:353`) and would therefore cost the same in every process.

**UNKNOWN.** First-touch page faults on the weights, memory pressure, or a lazy
initialisation elsewhere. Classified `COLD-START`, held apart from warm TTFP,
and deliberately NOT fixed: a warm-up that hides an unattributed second is worse
than the second.

**The smallest experiment that would decide it** (Linux, where the counters are
honest, and `getrusage` is already in `[DUMP]`): N fresh processes, minor and
major fault counts read immediately before and after the first step, crossed
with (a) page cache purged vs warm and (b) one synthetic dispatch before the
first stream. If the slow runs are the ones taking major faults, it is paging
and warm-up is the answer; if not, something is lazily initialised and the
counter will say where.

### R-3A — the exit table (warm C=1)

| component | measured | classification | evidence | removable | confidence |
|---|---|---|---|---|---|
| first chunk's audio | 245-249 ms | REAL-TIME INPUT REQUIREMENT | `enc_stream_need` = 25 mel frames | no: the audio does not exist yet | high |
| onset to first non-blank | 0.50-0.97 s | MODEL ARCHITECTURE / CHECKPOINT | blank margin +9 decaying to the crossing | only by changing what wins the argmax | high |
| of which right context | 240 ms | MODEL ARCHITECTURE (preset `[56,3]`) | `q = right+1`, att_context semantics | only with a different preset | high |
| step compute | 30-46 ms per 320 ms period | OUR IMPLEMENTATION | trace; idles 270-291 ms between steps | yes, but it buys no TTFP at C=1 | high |
| publication | 0.12-0.23 ms | OUR IMPLEMENTATION | R-2 | nothing to remove | high |
| first step, fresh process | 43.8 / 49.4 / 1010.6 ms | COLD-START | 3 fresh processes | unknown | low |
| blank logit outliers | -870 to -882, margin +100 | UNKNOWN | isolated frames | never changes a decision | low |

### Why does the first non-blank need 0.50-0.97 s of speech?

**Because a greedy argmax over this checkpoint keeps blank ahead of the best
token for six to ten encoder frames after onset, and each of those frames
already carries 240 ms of right context.** No threshold, suppression or
publishing rule participates: there are none in the code, and the one gate that
could have suppressed a token never did.

This is an answer about *these eight clips on this checkpoint*. It does not say
the checkpoint is right to wait, only that nothing between the argmax and the
socket is making it wait.

### What R-3 does NOT license

The two candidate levers are now named and both are numerics changes, not
latency tweaks: **a lower lookahead preset** (`[56,0]` exists) would cut the
240 ms of right context and shrink the chunk period from 320 ms to 80 ms, which
would also destroy the cadence budget F29 measured; **anything that lets a
non-blank win earlier** is a decoding change worth +1 to +4 of margin.

**And the baseline has an existing crack to close first.** Running the batched
gate for this work, `tests/test_stream_batch` reported, on int8 es-ES:

    off: Blavos dias, la reunion empieza a las VUELVE en la sala grande.
    one: Blavos dias, la reunion empieza a las NUEVE en la sala grande.

Streaming is right and offline is wrong, the test labels it pre-existing and not
a batching effect, and both paths mangle the first word. A responsiveness A/B
judged against a baseline that already disagrees with itself between the two
code paths would prove nothing. This belongs to Q-2.

Neither may be attempted before R-3F: a quality baseline able to detect a
premature or wrong first token, partial instability, final CER/WER regression,
and hallucination during leading silence. `configs/quality/` and
`tools/eval/cer_offline.py` cover the third only. **Q-2 is now a blocker for
R-5, not an independent item.**

## Axion run, 2026-09-21 — frozen as F30

Three experiments, then the box was powered off. Full numbers in
`docs/serving-findings.md` F30; what matters for this note:

- **R-3E is closed.** Cold start is major page faults reading the weights:
  3.26 s and ~6.5k major faults with the page cache dropped, 6 runs of 6; 36 ms
  and 0-2 major faults warm, 5 of 5. The second step is 12.7 ms in all 12 runs.
  A warm-up is now justified rather than guessed, and is deliberately NOT
  implemented here.
- **R-4 is answered on the production ISA.** `publication_delay` is
  **0.08-0.10 ms**. The output path is not a candidate for anything.
- **R-6's diagnostic half is done.** Steady-state TTFP p95 goes 1535 -> 1719 ms
  from C=1 to the knee while emission lag p95 goes 17 -> 268 ms. The audio the
  model needs after onset is **0.716 s at C=8, 32 and 65 alike**: the evidence
  requirement does not degrade under load.
- **The pairing had a real defect** and it was caught by its own causality
  check, not by taste: one mispaired row at C=32 and one at C=65 produced a
  negative publication delay. Pairs are now gated on clip duration, delta count
  and the full ordering, every rejected sample is counted, and all aggregates
  were regenerated from the raw artifacts.

**Not claimed.** That the opening wave explains F29's 2.5-4.5 s. It is strongly
consistent and there is no controlled A/B, so it stays an inference. The
established claim is the negative one: no steady-state load-dependent TTFP floor
of that size exists at these concurrencies.

**Not claimed either.** That 0.72 s of speech is an architectural bound. It is
a model-and-configuration floor under `[56,3]`, int8, greedy argmax.

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

R-3F, offline: the quality baseline that any emission change must be judged against —
premature or wrong first token, partial instability, final CER/WER, and
hallucination in leading silence. Q-2 is its vehicle and it now blocks R-5.
No optimisation before it exists.
