# S2-2 — one scheduler per worker, slots, the sink

Status: IN PROGRESS (spec 2026-09-18; implementation delegated, owner reviews)

Task: S2-2, S2-7
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

## Evidence / Conclusion / Next action
Pending implementation.
