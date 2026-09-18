# S2-2 — one scheduler per worker, slots, the sink

Status: IMPLEMENTED (spec 2026-09-18; S2-2b landed the batched step 2026-09-18;
gates passed on the M1 dev host, Linux numbers pending)

Task: S2-2, S2-2b, S2-7
Question: replace "one blocking thread per connection" with one scheduler
thread that owns the model and drives B stream slots, with ingest threads that
never touch the model, so that admission, fairness, cancellation and lag
accounting happen at chunk boundaries and a later batched step has a place to
land.

## Known facts

- The library now offers what the scheduler needs (commit after 5b84137):
  `mynah_asr_stream_reset(s, lang)` (pooled slots), `mynah_asr_stream_need_samples(s)`
  (samples until the next chunk fires), `mynah_asr_stream_audio_seconds(s)`;
  `mynah_asr_stream_feed` fires at most one text callback per completed chunk,
  then possibly one `is_eou` callback; `mynah_asr_stream_finish` flushes the tail.
  A stream is Nemotron-only (`stream_open` returns NULL on offline models).
- Sibling contract (qwen-tts `qwen_batch_sink_t`, mynah-tts `mynah_graph_sink`):
  the engine loop asks the transport for work through a vtable and reports
  through it; admission at the top of every step; `cancelled()` polled once per
  slot per step, decided before any other policy; the "only this thread enters
  the model" invariant is asserted in every callback, not commented.
- `server/stream_out.{c,h}` (S2-4): bounded ring, detached writer, backpressure
  = cancel. `server/prefork.{c,h}` (S2-1): the router hands descriptors to
  `mynah_asr_prefork_recv_conn`; `mynah_asr_prefork_conn_done()` exactly once
  per connection; `mynah_asr_prefork_refusal_response()` builds a refusal body.
- v1 REST path: `handle_transcribe` parses multipart/WAV, resamples, optional
  LID, then `transcribe_batch_ts` through a 25 ms micro-batch thread. Word
  timestamps exist only on the offline API, not on streams.

## Design (v2.0: per-slot feed, no cross-stream batching yet)

**Superseded in part by S2-2b** (see the Evidence section of that name): step
(4) below fed each ready slot with its own `mynah_asr_stream_feed`; it now
stages the ready set and makes ONE `mynah_asr_stream_step_batch` call. The
order of the step, the fairness rule, the lag accounting and the finalize path
are unchanged, which is why the rest of this section still reads true.

Threads in a worker: `mynah-recv`/`mynah-accept` (main: takes descriptors, ring
of fds) · `mynah-http` × `--threads` (parse; for WebSocket they become the
**ingest** thread of that connection for its life) · **`mynah-sched` × 1** (the
only thread that calls any `mynah_asr_*` inference function after load) ·
`mynah-out` per connection (stream_out writer).

### Slot (`server/slot.{c,h}`)

```c
typedef struct {
    int id;
    enum { SLOT_FREE, SLOT_ACTIVE, SLOT_FINISHING, SLOT_DONE } state;
    mynah_asr_stream *stream;     /* pooled: opened once per slot, reset per session */
    int lookahead;                /* of the pooled stream (reset cannot change it) */
    /* PCM ring, float32, bounded (default 30 s). Ingest pushes, scheduler pops. */
    float *ring; size_t cap, head, len;
    /* arrival records (sample_end_index, t_monotonic) so a delta produced at
     * audio position p can be charged to the frame that carried sample p */
    struct { size_t end; double t; } arr[256]; int arr_head, arr_len;
    /* requests from ingest, read at the step boundary */
    int finalize_req, reset_req, cancel_req; char reset_lang[24];
    /* output */
    mynah_asr_stream_out *out; unsigned seq;
    /* per-session accounting for /health, the done frame and the tests */
    double t_open, t_first_audio, t_first_delta, lag_sum_ms, lag_max_ms;
    size_t samples_in, samples_consumed; int deltas, eous, steps;
    char lang[24];
    pthread_mutex_t mu; pthread_cond_t space;   /* ingest waits here when the ring is full */
} mynah_asr_slot;
```

Ingest → slot: `slot_push(slot, samples, n, t_now)` copies into the ring under
`mu`; when the ring is full it **blocks on `space`** (that is TCP backpressure:
a client faster than real time is throttled by not reading its socket, and can
never take more than one chunk per step from the others). `slot_request(slot,
finalize|reset lang|cancel)` sets a flag and signals the scheduler.

Scheduler → slot: `slot_take(slot, dst, n)` pops exactly `n` samples under `mu`,
signals `space`, and returns the arrival time of the last popped sample from
`arr`. Everything else the scheduler does with a slot happens without the lock
(the stream object is the scheduler's alone).

### Scheduler (`server/sched.{c,h}`)

```c
int  mynah_asr_sched_start(mynah_asr_model *m, int max_slots, int ring_seconds, int lookahead_default);
/* From an HTTP thread once the WebSocket handshake is accepted. Returns the slot
 * or NULL when every slot is taken (the caller has already been admitted by the
 * router; this is the worker's own cap and is answered 503 BEFORE the 101). */
mynah_asr_slot *mynah_asr_sched_open(int fd, const char *lang, int lookahead, char *err, size_t errcap);
/* Offline REST work still goes through the same thread: one call, bounded by
 * the model's segment limit; the HTTP thread blocks on the job. */
int  mynah_asr_sched_transcribe(const mynah_asr_offline_job *job);   /* wraps transcribe_batch_ts */
void mynah_asr_sched_stop(void);            /* stop admitting, finish the step, free slots */
void mynah_asr_sched_health(cJSON *into);   /* facts: slots active/cap, steps, lag p50/max, offline jobs */
```

Loop, one iteration = one **step**:

1. Poll `cancel_req`, `out->failed`, `peer_gone` for every active slot first;
   a cancelled slot is released within this step (no finish compute), counted
   `cancelled`, and its connection reported to the router (`conn_done` is the
   ingest thread's job when it returns, exactly once).
2. `reset_req` → `mynah_asr_stream_reset`, accounting reset, `seq` continues.
3. Ready set: a slot is ready when `ring.len >= need_samples`, or when
   `finalize_req` and the ring is non-empty (a short last piece), or when
   `finalize_req` and the ring is empty (run `stream_finish`).
4. For each ready slot in round-robin order (start index rotates per step):
   take exactly `need` samples (or all remaining on finalize), call
   `stream_feed`; the callback builds the frame (see below) and enqueues it on
   the slot's `stream_out`; `lag_ms = now - arrival_of_last_consumed_sample`.
   On finalize with empty ring: `stream_finish`, emit `done`, `stream_out_finish`,
   release the slot to FREE (stream object kept, reset lazily on next open).
5. Offline jobs: at most one `transcribe_batch_ts` call per step, aggregating
   what is queued up to `--batch`. (It stalls streams on this worker for its
   duration; v2.0 accepts that and says so in the banner; offline-heavy
   deployments use a separate worker group, S2-6.)
6. If nothing was ready and nothing is queued, wait on the scheduler condvar
   until ingest signals (push, request, new job); never spin, never sleep on a
   fixed tick — the first chunk of a new slot runs the moment it lands.

`sched_assert_thread()` is called at the top of every callback and every
`mynah_asr_*` call site in the module (compares `pthread_self()` with the
scheduler's id; aborts on mismatch in debug, logs once in release).

### Frames (S2-5, first cut)

Server → client text frames, all with `"seq"`, `"audio_s"` (consumed audio,
from `stream_audio_seconds`) and `"lag_ms"`:
`{"type":"delta","text":"...","final":true,"lang":"it-IT"}` ·
`{"type":"eou","t":12.34}` · `{"type":"done","lang":"it-IT","audio_s":...,
"steps":N,"lag_p50_ms":...,"lag_max_ms":...}` · `{"type":"error","code":...,
"message":...}`. For compatibility the v1 fields `"text"`, `"language"`,
`"audio_seconds"` are kept on `delta` for this release and removed in S2-5
proper (`tools/eval/ws_client.py` and `tools/bench/stream_load.py` read
`text`/`audio_seconds` today).

Client → server: binary = PCM s16le 16 kHz mono (any size); text
`{"type":"finalize"}` → the scheduler flushes and sends `done`, the connection
stays open for a `reset` (implicit: audio after `done` starts a new utterance);
`{"type":"reset","lang":"..."}`; close frame → finalize then close; ping → pong.

### Ingest (in `server/main.c`, replaces `handle_ws_stream`)

After the handshake is accepted **and** `sched_open` succeeded (order: reserve
the slot first; on failure answer `503` with the `server_at_capacity` body via
`mynah_asr_prefork_refusal_response` and return without upgrading), write the
101, start `stream_out` on the fd (it owns writes from here), then loop reading
frames with `SO_RCVTIMEO` = `--idle-ms` (default 60000): on a read timeout the
ingest thread requests `cancel` with reason `idle`; a frame larger than
`--max-frame-bytes` (1 MiB) is an `error` frame then cancel. When the slot
reaches DONE or the client closes, the ingest thread returns; `handle_conn`
closes nothing that `stream_out` owns (the fd belongs to the writer once
started; `handle_conn` skips its final `close(fd)` in that case).

### Admission inside the worker (S2-3, the part that lands here)

`--cap C` is the number of slots; a WebSocket beyond it is refused before the
upgrade; a REST request beyond `--max-pending` (default 2·threads) is refused
`server_at_capacity`. Per-stream caps: `--max-audio-seconds` (default 14400),
`--idle-ms` (default 60000), `--ring-seconds` (default 30). Graceful shutdown
(`SIGTERM`): stop taking connections → cancel slots with an `error` frame
`code:"shutting_down"` → stop the scheduler → join → close model.

## Acceptance gate (S2-7 included)

1. `tests/test_server.sh` WebSocket checks pass unchanged (v1 fields kept).
2. `tools/bench/stream_load.py --streams 4 --repeat 2` against the worker with
   Nemotron: **every stream's text byte-identical** to `mynah-asr transcribe`
   of the same clip (the tool's identity gate, plus a reference JSON produced
   by the CLI), TTFP and lag numbers printed; also under `--prefork 2`.
3. `tests/test_server_concurrency.sh` green (REST path still works).
4. A client that stops reading is cancelled within the send timeout and the
   other streams' lag p95 does not move (assert with two runs of the tool).
5. `grep -n pthread_mutex_lock server/*.c` shows only the fd ring, the slot
   rings, the offline job queue and the writer.
6. `make leaks` clean on `test_stream_out`; `make ubsan` clean on the server
   test; TSan once on `test_stream_out` and reported.

## Evidence

Implementation landed on top of `354743c`. Host: Apple M1 (arm64, 8 cpus),
macOS 25.5, Accelerate, `make` default flags (`-O3 -march=native`), model
`nemotron-3.5-asr-streaming-0.6b` `--quant int8` (pre-quantised checkpoint), the
110m `parakeet-tdt_ctc-110m-gguf` for the REST gate. **Development-machine
numbers: signals, not a serving point** (ENGINEERING §8) — the qualifying SOAK
belongs to S4 on the Linux box.

### What was built

- `server/slot.{c,h}` — the slot: float PCM ring sized by `--ring-seconds`,
  arrival records (`{end_sample, t_monotonic}`, one per push, merged into the
  newest when full so the oldest, which dates the audio about to be consumed, is
  never the one lost), blocking `slot_push`, `slot_take` returning the arrival
  of the last popped sample, request flags read at the step boundary, a
  per-session 8 ms lag histogram, `slot_wait_done` so a teardown costs one
  wakeup instead of a sleep loop.
- `server/sched.{c,h}` — one thread per worker, `sched_assert_thread()` at every
  callback and every inference call site (abort without `NDEBUG`, one line with
  it). Step order exactly as specified: cancel → reset → ready set round-robin,
  one chunk per slot per step → at most one batched offline call → park on a
  condvar. No tick. Offline REST work replaced the `batch_worker` thread; a
  batch is one kind AND one lookahead, because the batched call resolves the
  lookahead once for the whole batch and mixing them would make a transcript
  depend on who it travelled with (v1 took `jobs[0]->lookahead` for everyone).
- `server/main.c` — ingest loop, slot reserved before the 101, `503` /
  `400 model_not_streaming` before the upgrade, control messages, ping→pong
  through the writer, `SO_RCVTIMEO` + a 200 ms poll tick, oversized frame and
  idle caps. `g_inflight`, `inference_begin/end` and the BLAS coupling are gone:
  `mynah_asr_blas_set_concurrency(1)` once at start.
- The ingest reads through `dup(fd)` while the writer owns the original. The
  writer closes as soon as it has drained, which can happen while the ingest is
  parked in a read; reading a descriptor another thread is closing is how a
  server ends up serving a stranger after `accept()` reissues the number.
- s16→float conversion is byte-wise. The v1 code cast the payload to
  `const int16_t *`, which is unaligned and undefined; ubsan is right about it.

### What ran, and what it measured

`make` clean, no warnings. `grep -n pthread_mutex_lock server/*.c` → `main.c`
(the fd ring), `slot.c` (the slot rings), `sched.c` (the offline job queue, the
scheduler's wake flag and nothing else), `stream_out.c` (the writer). Counters
`/v1/health` reports are relaxed atomics, so a health probe never queues behind
a step.

```
sh tests/test_server_concurrency.sh <110m-gguf> 8307        -> 0, all 8 checks OK
sh tests/test_server.sh <nemotron> 8715                     -> 0 (ws-stream OK: v1 fields kept)
sh tests/test_serve_repro.sh <nemotron>                     -> 0 (16 concurrent responses byte-identical)
sh tests/test_server_stream.sh <nemotron> 8513              -> 0
make test                                                   -> 0 (77 where model-gated)
```

`tests/test_server_stream.sh`, the S2-7 gate, on this host:

| phase | result |
|---|---|
| 4 streams × 2 utterances, `--cap 4`, reference = `mynah-asr transcribe --quant int8 --lang auto` | MEASURED, 0 errors, 8/8, identity and reference clean. TTFP p50/p95 **1312 / 1655 ms**, emission lag p50/p95 **1152 / 2091 ms** |
| stalled reader beside 2 streams | the 2 streams MEASURED (TTFP p50 1091 ms, lag p50/p95 **226 / 390 ms**), stalled slot cancelled and reclaimed |
| `--prefork 2 --cap 2`, 2 streams × 2 | MEASURED, identity clean, TTFP p50/p95 **927 / 963 ms**, lag p50/p95 **140 / 159 ms**; SIGTERM leaves 0 survivors |

Capacity of one scheduler on this host, from `tools/bench/stream_load.py`
(`--repeat 2`, mixed it/en/de/fr, `--lookahead 3` = 330 ms chunks):

| streams | emission lag p50 / p95 | finalization p50 | wall vs audio |
|---|---|---|---|
| 1 | 145 / 156 ms | 100 ms | 5.4 s for 5.2 s |
| 2 | 261 / 728 ms | 166 ms | 10.5 s for 19.1 s |
| 4 | 1773 / 3042 ms | 2993 ms | 15.3 s for 34.0 s |

So `T_step(1) ≈ 145 ms` against a 330 ms period: **B=2 holds with margin on an
M1, B=4 does not** — the lag grows monotonically through the run, which is
exactly the backlog failure §3 of the design note predicts. The identity gate
still passes at B=4 (a slow server is still a correct one), which is why it is
the identity that gates and the cadence that is reported.

ubsan: `make clean` then a `-fsanitize=undefined -O2` build of the server and
the CLI, `tests/test_server_stream.sh` and `tests/test_server_concurrency.sh`
both green with **0 `runtime error` lines**. Rebuilt clean afterwards.

Re-run from the **committed** tree (ENGINEERING §12), `b8ff408`, clean working
directory, `make clean && make`, `mynah-asr-server` sha256
`4cb3e7d0d39607e0f671b4cfb1491ab3aebd0fb3ac202a2b19ec033df1dbdd2b`: the stream
gate, `tests/test_server_concurrency.sh` and `tests/test_server.sh` all exit 0.
The committed build's own numbers: 4 streams TTFP p50/p95 1449/1593 ms, lag
p50/p95 1228/2194 ms; prefork 2×2 TTFP 931/936 ms, lag 126/137 ms.

The control channel is not yet in a test (its gate belongs to S2-5), but it was
exercised by hand against this build: one socket, an unknown control message
answered with an `error` frame and the session surviving it, then three
utterances separated by `{"type":"finalize"}` and `{"type":"reset"}`, each
transcript byte-identical to the CLI's and `seq` continuing across them
(15 → 27 → 37). A test for that belongs with S2-5.

### What the stalled-reader phase actually proves

The client sets a 2 KiB receive buffer, sends 4.4 s of audio at 1x and never
reads. The slot is reclaimed ~3 s after it stops sending, and the server log
carries **no** `stream aborted` line — so the cap that fired was `--idle-ms`,
not the output ring and not `SO_SNDTIMEO`. That is the honest reading: JSON
deltas are small, and filling a loopback socket buffer with them takes far
longer than any test should wait. Slow-reader isolation as such (ring overflow →
cancel) is covered by `tests/test_stream_out.c` at the module level; what this
phase shows end to end is that a silent client is reclaimed on a bounded timer
and that the streams beside it do not move.

## Evidence — S2-2b (the step is one batched call), 2026-09-18, macOS arm64

Implementation on top of `5a6d612`, committed as `e605278`. Same host as the
S2-2 evidence above: Apple M1 (arm64, 8 cpus), macOS 25.5, Accelerate, `make`
default flags, `nemotron-3.5-asr-streaming-0.6b --quant int8`, binary
`mynah-asr-server` sha256 `a27160ab1cb7...`, `--threads 4 --batch 4`.
**Development-machine numbers: signals, not a serving point** (ENGINEERING §8).

### What changed

`sched_feed` is gone. The step now has three parts instead of one:

- `sched_stage(slot, avail, B)` takes exactly `need_samples` from the slot's ring
  (round-robin start, one chunk per ready slot, unchanged) and files the chunk as
  row `B` of the step's batch: `streams[]`, `samples[]` (the slot's own `take`
  scratch), `n_samples[]`, `userdata[]` = that slot's `emit_ctx`. Nothing is fed.
- `sched_step_batch(B)` makes ONE `mynah_asr_stream_step_batch` call for the whole
  set, under `sched_assert_thread`. A `-1` cancels every session in that set with
  `decode_failed`, which is what the per-slot feed did for its one slot.
- `sched_feed_tail` + `sched_finalize` keep the per-slot path for a finalizing
  stream: a last piece shorter than a chunk through `stream_feed`, the tail
  through `stream_finish` (the batched call takes whole chunks only and does not
  accept `is_last`). They run AFTER the batch, so a slot's own deltas keep their
  order.

Cancellation still runs before any other policy, resets still run before the
ready set is built, lag is still charged from the slot's own arrival record (the
callback of each row gets that slot's `emit_ctx`), and every frame still leaves
through that slot's `stream_out`. `mynah_asr_stream_batch_reserve(model, cap)` is
called once in `mynah_asr_sched_start`, before the scheduler thread exists, so
the step path allocates nothing; the per-step arrays are carved with the slot
table. A ready set larger than `MYNAH_ASR_STREAM_BATCH_MAX` becomes several calls
in the same step (the constant moved to the public header rather than being
copied here).

### Gate 1 — the same gate, visibly better lag

`tests/test_server_stream.sh` unchanged in its first three phases, both commits
built clean and run back to back on a QUIET host (loadavg ~2.5; an earlier pair
taken while a sibling process held a core is discarded, and that is why these
numbers differ from the ones in the S2-2 section above).

| phase | before `5a6d612` (per-slot feed) | after `e605278` (batched step) |
|---|---|---|
| 4 streams × 2, `--cap 4`: TTFP p50/p95 | 1367 / 1707 ms | **1026 / 1074 ms** |
| 4 streams × 2, `--cap 4`: emission lag p50/p95 | 1219 / **2193** ms | 145 / **253** ms |
| 4 streams: envelope | NOT STREAMABLE | **GOOD** |
| stalled reader beside 2 streams: lag p50/p95 | 252 / 444 ms | 112 / 194 ms |
| `--prefork 2 --cap 2`, 2 × 2: lag p50/p95 | 143 / 162 ms | 137 / 159 ms |

Identity, isolation and the prefork shutdown pass in both: a slow server is still
a correct one, which is why identity is the gate and cadence is the report.

### Gate 2 — the new "batched-identity" phase

Eight streams over five clips on ONE `--cap 8` process, `--repeat 1`, every text
byte-identical to the CLI's own streaming answer, 8/8 utterances, 0 errors. The
reference is `mynah-asr stream`, not `transcribe`: `test_es.wav`'s int8 streaming
transcript already differs from its offline one ("a las muertes" vs "a las
vuelve"), a pre-existing numerical difference recorded in `.work/stream-api-v2.md`
and nothing to do with batching; for the other four clips the two references are
the same bytes.

The phase then reads `/v1/health` and REQUIRES `batched_steps_total > 0` and
`rows_stacked_total > 0`, because an identity gate alone cannot tell the batched
path from the fallback — both are correct. On the quiet host:

```
    TTFP p50/p95 963/1323 ms   emission lag p50/p95 155/268 ms   envelope MARGINAL
    batched_steps_total 63   rows_stacked_total 256   ready_mean 1.57   step_wall_ms mean 108.9
      B=1  steps 35  wall mean 119.3 ms
      B=2  steps 20  wall mean  87.0 ms
      B=3  steps  8  wall mean 117.8 ms
```

### Gate 3 — WAVE, 8 streams × 2 utterances, `--cap 8`, one process

`python3 tools/bench/stream_load.py --mode wave --streams 8 --repeat 2 --clips
tests/audio/test_{it,en,de,fr,es}.wav --port 8901` against
`mynah-asr-server --threads 4 --batch 4 --quant int8 --cap 8`.
**macOS dev host, not a serving number** (WAVE screens, only a SOAK promotes).

| | before `5a6d612` | after `e605278` |
|---|---|---|
| utterances | 16/16 ok, 0 errors | 16/16 ok, 0 errors |
| TTFP p50/p95 | 1372 / 2053 ms | **944 / 1394 ms** |
| emission lag p50/p95 (client) | 1104 / **1952** ms | 143 / **258** ms |
| server `lag_ms` p50/p95 | 1104 / 1952 ms | 142 / 258 ms |
| finalization lag p95 | 2371 ms | 398 ms |
| backlog max p95 | 1.664 s | 0.284 s |
| wall | 24.5 s | 19.0 s |
| verdict | **NOT STREAMABLE** (4 of 4 limits failed) | **MARGINAL** (only TTFP p95 1394 vs 1160) |

`/v1/health` after the run:

```
batched_steps_total 134   rows_stacked_total 480   ready_sum 201   ready_mean 1.50
reserved_slots 8          step_wall_ms sum 14793.7 count 134 mean 110.4
by_b  B=1: 81 steps, 118.7 ms mean | B=2: 39 steps, 88.5 ms | B=3: 14 steps, 123.2 ms
```

480 stacked rows out of the ~868 encoder rows the run produced (a chunk is q = 4
rows at preset [56,3]): 39 × 2 × 4 + 14 × 3 × 4 = 480, i.e. **55 % of the work
went through the stacked path** although the mean ready set was only 1.5.

### Fitting `a` and `b`

Two fits, and they do not measure the same thing.

**In situ, from `by_b`** (the WAVE run plus the gate-2 phase, weighted): B = 2 →
88.0 ms over 59 steps, B = 3 → 121.3 ms over 22 steps, so **a ≈ 21 ms, b ≈ 33 ms**
and `B_max = (0.7·320 − a)/b ≈ 6`. B = 1 is deliberately excluded: it is the
library's single path, whose small-`T` int8 kernel is serial, so it costs MORE
(118.7 ms) than a batched B = 2 (88.0 ms) and a line through it has a negative
slope. This fit is weak and says so: B is not an independent variable here (the
scheduler batches whatever happens to be ready), and only 22 steps reached B = 3.

**Controlled, same host and binary**, `tests/test_stream_batch <model> --steptime`
(B is fixed, 5 clips cycled):

| B | single ms | batched ms | ratio | ms per stream |
|---|---|---|---|---|
| 1 | 109.86 | 106.40 | 0.97 | 106.40 |
| 2 | 192.85 | **84.20** | 0.44 | 42.10 |
| 4 | 342.87 | **128.29** | 0.37 | 32.07 |
| 8 | 689.04 | **221.05** | 0.32 | 27.63 |

Least squares over B ∈ [2,8]: **a ≈ 37.8 ms, b ≈ 22.9 ms**, so `B_max ≈ 8` with
ρ = 0.7 and P = 320 ms. That reproduces S1-4's own fit (a ≈ 37, b ≈ 23.7) on this
host, which is the point of quoting it: the server's step is the library's step
plus the server's own per-delta work, and the difference between the two fits
(b 33 vs 23 ms) is charged to what the server does INSIDE the call — the per-delta
cJSON build and enqueue run in the result callback, once per stream per step.
That is the allocation exception S2-2 already declared, and it is now measured
rather than suspected.

### Why the win is so large when the mean ready set is only 1.5

Because B is not a setting, it is a symptom. A worker that keeps up sees its
slots become ready one at a time (clients send 100 ms frames, chunks are 320 ms,
arrivals are staggered) and steps at B = 1. A worker that falls behind has
several slots ready at once and steps at B = 2, 3, …, which is exactly when the
stacked pass is 2–3× cheaper per stream. The batching therefore acts as a brake
on the backlog: it is strongest where the old code compounded. Before, 8 streams
drove the worker into a backlog it never recovered from (backlog max 1.66 s, lag
p95 1.95 s, and the run needed 24.5 s of wall to serve what it would have served
in ~19); after, the same run stays inside one chunk period.

### What ran, not what was configured

Every measurement above comes from a clean committed tree, binary
sha256 `a27160ab1cb7…` (`make clean && make` reproduces the same hash), with the
`[FLAGS]` / `[EFFECTIVE-CONFIG]` / `[SERVER-CONFIG]` / `[TOPOLOGY]` banner in the
run's own log: `build=v0.9.1-48-ge605278 blas=accelerate simd=neon+dotprod`,
`quant=int8 lookahead_default=3 chunk_ms=320 cap=8 http_threads=4 pool_threads=8`.

### The other gates

```
sh tests/test_server_stream.sh <nemotron> 8833        -> 0 (4 phases, batched-identity included)
sh tests/test_server_protocol.sh <nemotron> 8323      -> 0
sh tests/test_server_metrics.sh <110m-gguf> 8327      -> 0
sh tests/test_server_concurrency.sh <110m-gguf> 8331  -> 0
make test (PARAKEET110_DIR=<110m-gguf>)               -> 0 (21 skips, all model/uv-gated)
make check                                            -> 0 (plan, repo integrity, flag registry)
```

`make test` includes `make test-stream-batch-allocs` (0 allocations per batched
step) and `test-stream-allocs` (0 per chunk): the step path is still
allocation-free with the batch scratch reserved at start-up.

UBSAN: `make clean`, then an `-fsanitize=undefined -O2` build of the CLI and the
server (Makefile `ubsan` flags, no `-ffast-math`), all four phases of
`tests/test_server_stream.sh` green with **0 `runtime error` lines**, collected
with `UBSAN_OPTIONS=log_path=…` because the test deletes the directory it
redirects each server's stderr into — a diagnostic printed there would have gone
with it. Rebuilt clean afterwards (`make clean && make` returns the same binary
hash).

That run is also the cleanest evidence for the paragraph above: a sanitized
server is roughly three times slower per step, and its ready set rose from 1.6 to
**3.3** on the same 8-stream phase (17 of its 30 steps at B = 4), with 388 rows
stacked. The scheduler batches more exactly when a step costs more.

### What S2-2b does NOT answer

- **No Linux number.** `a`, `b`, and therefore the per-worker cap, are still S0/S4
  work on the Axion. Nothing above may be used to size a deployment.
- **The ready set is small by construction on a healthy worker** (mean 1.5 at 8
  streams). Coalescing — waiting a few milliseconds at the top of a step for more
  slots to become ready — would raise B and the stacked share, at the cost of
  adding latency on purpose. NOT done here (it changes the cadence policy, which
  is its own decision); recorded as a candidate for S2/S5.
- The in-situ `by_b` fit is too weak to size anything on its own; the controlled
  step table is the one to quote until a run holds a fixed B for long enough.
- `rows_stacked_total` is 0 on an f32 build with a BLAS that is not row-stable
  (OpenBLAS today): the counters make that visible, but the Linux side of it is
  S1-8.
- TSan was not run in this pass either (S2-2 gate item 6 is still open).

## Conclusion

S2-2 is implemented and its gate passes, with the caveats stated: the byte
identity holds at 4 concurrent streams and under `--prefork`, the transport
refuses before the upgrade, and one thread owns the model with the invariant
asserted rather than commented. Gate items 1, 2, 3, 5 pass; item 4's "a stopped
reader is cancelled within the send timeout" is served by the idle cap here and
is recorded as such, not as the send timeout; item 6's TSan pass was not run.

S2-2b closes the cadence half of it on this host. The scheduler's step is one
batched library call over the ready set; the transcripts did not move (the gate
that proves it now runs at 8 streams over 5 clips as well as at 4), the worker
holds 4 streams inside one chunk period where it used to run a 2.2 s backlog, and
the process says which path it took rather than being believed. What is still
missing is a Linux number and a TSan pass, not a mechanism.

## Unknowns

- No Linux number at all: `a` and `b` of the cadence law, and therefore the real
  per-worker slot cap, are S0/S4 work on the Axion. The M1 figures above must
  not be used to size anything.
- The per-delta frame is built with cJSON (one `cJSON_CreateObject` +
  `cJSON_PrintUnformatted` + `free` per frame). Everything else in the step path
  is allocation-free (ring, arrival records, chunk scratch and frame scratch are
  carved at slot init); this one is a declared exception for this cut and is the
  first thing S1-3 should take out.
- `peer_gone()` polls once per slot per step, so a worker at B=32 pays 32
  `poll()` syscalls per 330 ms. Not measured; suspected negligible, but suspected.
- A pooled stream is re-opened when a client asks for a lookahead different from
  the one that slot's stream holds (reset cannot resize the scratch). A
  deployment mixing lookaheads therefore pays an open per switch. Unmeasured.
- TSan on `test_stream_out` (gate item 6) was not run in this pass.
- The ingest thread's `dup` doubles the descriptors per stream; the fd ceiling
  of a many-stream worker has not been re-derived.

## Next action

S2-5 proper (drop the v1 frame fields, `t0` on deltas), S2-3's remaining rungs,
and the two things S2-2b measured rather than fixed: the per-delta cJSON build
inside the step (it is what makes the server's per-stream slope larger than the
library's) and the question of whether a worker should wait a few milliseconds
for a larger ready set. The Linux `a`/`b` and the TSan pass stay with S0/S4.
