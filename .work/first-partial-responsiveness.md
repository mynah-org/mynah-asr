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
speech onset. Lowering it therefore means changing what the decoder waits for,
not making anything faster. The 2.6 s seen on the Axion at C=60 is this floor
plus load; how much of it is load is not established by this run.

**UNKNOWN.** Whether 0.5-1.0 s of speech is inherent to this checkpoint or a
consequence of our blank policy, `max_symbols_per_step`, or the emit rule
`total > chars_emitted` — that is R-3 and it is a numerics question, not a
latency one. Whether the cold first step is allocation, page faults or a BLAS
first call. What any of these become under load (R-6).

**The largest removable component is not yet identified**, and the largest
component is not obviously removable: it is the model declining to commit.

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

R-3: classify the 0.5-1.0 s of speech the RNNT consumes before its first
non-blank. It is a decoding-semantics question — blank policy, max_symbols,
the emit rule — and it is answered by reading code and by the oracle, not by a
latency experiment. Nothing is optimised until that classification exists.
