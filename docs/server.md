# mynah-asr-server — HTTP + WebSocket API

```sh
make && ./mynah-asr-server -m models/nemotron-3.5-asr-streaming-0.6b -p 8090 --threads 4
# Linux, many cores: add --prefork W (see below)
```

## Prefork: pinned worker processes (Linux production)

```sh
./mynah-asr-server --prefork-plan                 # the machine's topology and the W/T sweep, no model needed
./mynah-asr-server -m models/nemotron-3.5-asr-streaming-0.6b --quant int8 -p 8090 \
    --prefork 8 --prefork-threads 4 --threads 4 --cap 4
```

`--prefork W` forks W worker processes after the model is mapped (one physical
copy of the weights for the whole tree) and before any thread exists. On Linux
each worker is pinned to a contiguous **core-major** slice of the cpus this
process is allowed to use (`sched_getaffinity`, cgroup quota read and warned
about); elsewhere workers run unpinned and the banner says so. The parent never
runs inference: it accepts, picks the least-loaded worker with a free slot,
hands the descriptor over a socketpair with `SCM_RIGHTS`, and learns of a
finished connection by one byte coming back. `--cap C` is the number of
connections a worker holds at once (default `--threads`); a WebSocket stream
holds one for its whole life.

Admission is a ladder, outermost first, each rung with its own counter and its
own `error.code`: a full fleet refuses with `503 server_at_capacity` and
`Retry-After` (the listener is always polled, so overload is a visible refusal
and never a wait hidden in the kernel backlog); a bounded queue holds
`MYNAH_ASR_PREFORK_QUEUE` connections per live worker (default 1, `0` refuses at
once); an entry older than `MYNAH_ASR_PREFORK_QUEUE_MS` (default 2000) when it
reaches the head is refused `queued_too_long`; `MYNAH_ASR_PREFORK_SERVICE_MS`
(default 30000) is the per-request service cap. A refusal is written, the
socket is half-closed and the client's pending body drained before `close()`,
so the client reads the status instead of a connection reset. `SIGUSR1` on the
parent prints the per-worker table and forwards to every worker; `SIGTERM`
stops the fleet. `/v1/health` reports which worker answered.

Design and the measurements it rests on: `.work/serving-v2-design.md`,
`.work/server-prefork.md`, `.work/server-scheduler.md`. The prefork router, the
asynchronous writer and the per-worker scheduler are in; multi-model worker
groups and the batched stream step follow (`PLAN.md` S2).

## Endpoints

### POST /v1/audio/transcriptions — OpenAI-compatible

Multipart form-data (like the OpenAI/Whisper API) or raw `audio/wav` body:

```sh
curl -F file=@audio.wav -F language=auto http://localhost:8090/v1/audio/transcriptions
# {"text": "..."}

curl -F file=@audio.wav -F language=it-IT -F response_format=verbose_json ...
# {"text": "...", "task": "transcribe", "language": "it-IT", "duration": 5.23}

curl -X POST --data-binary @audio.wav -H 'Content-Type: audio/wav' ...
```

Fields: `file` (WAV PCM16, any sample rate — automatic resampling),
`language` (locale tag or `auto`, default auto), `response_format`
(`json` | `text` | `verbose_json`), `lookahead` (0|1|3|6|13, model default),
`target_language` (AED models only: output language ≠ source = translation).
With `verbose_json` (without batching) the response includes `words` with timestamps.

### POST /v1/audio/translations — speech translation (AED/Canary models only)

Same fields as `/transcriptions`; `target_language` defaults to **en**
(OpenAI/Whisper style). On non-AED models it responds 400.

```sh
# spoken de -> English text
curl -F file=@audio_de.wav -F language=de http://localhost:8090/v1/audio/translations
# spoken en -> German text, with metadata
curl -F file=@audio_en.wav -F language=en -F target_language=de \
     -F response_format=verbose_json .../v1/audio/translations
# {"text": "Hallo, ...", "task": "translate", "language": "en", "duration": 4.34}
```

#### `language=auto` on a Canary: `--lid-model`

On the AED models the source language is an INPUT — `auto` there means the model's
default (`en`), not detection. Start the server with a detector and every request
with `language=auto` gets one:

```sh
./mynah-asr-server -m models/canary-1b-v2 \
    --lid-model models/nemotron-3.5-asr-streaming-0.6b --quant int8

curl -F file=@unknown.wav -F target_language=en -F response_format=verbose_json \
     http://localhost:8090/v1/audio/translations
# {"text": "Hello, ...", "task": "translate", "language": "it", "duration": 5.23}
```

The detector reads a few seconds (~0.5 s of CPU with int8, independent of the file
length) and stays resident, serving every worker; the language it returns is adapted
to what the served model accepts and reported in `verbose_json`. Detecting **nothing**
(silence, a clip too short) is not an error: the request goes through with the model's
default. Detecting a language the served model **does not have** is — same `400` as
sending that language in the request, because the alternative is a fluent invented
transcript. Requests that name a language explicitly never pay for any of this.
Ignored (with a note) on a model that already detects by itself.

### GET /v1/audio/stream — WebSocket streaming (protocol v2)

Query: `?lang=auto&lookahead=3`.

**Client → server**

| frame | meaning |
|---|---|
| binary | PCM s16le 16 kHz mono, any size up to `--max-frame-bytes` |
| text `{"type":"finalize"}` | flush the tail, emit `done`, keep the socket: audio after that starts a new utterance |
| text `{"type":"reset","lang":"it-IT"}` | new utterance on the same socket, optionally a new language |
| ping | answered with a pong |
| close | finalize, emit `done`, close |

An unknown text message is answered with an `error` frame and does **not** end
the session. A binary frame larger than `--max-frame-bytes` does.

**Server → client** — every frame carries `seq` (per session, counting every
frame), `audio_s` (audio consumed when it was produced) and `lag_ms` (wall time
from the arrival of the last sample consumed to the frame being enqueued):

```json
{"type":"delta","text":"...","final":true,"lang":"it-IT","seq":3,"audio_s":1.32,"lag_ms":146,
 "language":"it-IT","audio_seconds":1.32}
{"type":"eou","t":4.10,"seq":9,"audio_s":4.16,"lag_ms":91}
{"type":"done","done":true,"lang":"it-IT","language":"it-IT","audio_s":5.21,"audio_seconds":5.21,
 "steps":16,"deltas":15,"lag_p50_ms":144,"lag_max_ms":358,"seq":17}
{"type":"error","code":"idle_timeout","message":"..."}
```

`text`, `language`, `audio_seconds` and `done:true` are the **v1 fields, kept
for one release** so existing clients keep working; they go away in S2-5 proper.
Nemotron deltas are always `final:true` (monotonic greedy, never retracted); the
field exists so an unstable-tail decoder can later send `final:false` without a
protocol change. `lag_p50_ms` in `done` is a median over 8 ms buckets, which is
the histogram the session keeps — a p50 to the nearest 8 ms, not a mean wearing
a median's name. An `error` frame raised by the transport (a bad control
message, an oversized frame) carries no `seq`: it is about the message the
client just sent, not about the audio.

**Error codes**: `idle_timeout` · `frame_too_large` · `peer_gone` ·
`shutting_down` · `decode_failed` · `reset_failed` · `unknown_message` ·
`unsupported_opcode` · `model_not_streaming`.

**Refusals happen before the upgrade**, as HTTP statuses, so a client reads a
status and not a transport error: `503` with `error.code`
`server_at_capacity` and `Retry-After` when every slot of this worker is taken,
`400` with `error.code` `model_not_streaming` on an offline-only model
(Parakeet, Canary — they have no cache-aware streaming presets and never will,
so there is no `Retry-After`).

Reference client (Python stdlib): `tools/eval/ws_client.py`. Load harness:
`tools/bench/stream_load.py`.

### GET /v1/models · GET /v1/health · OPTIONS (CORS)

`/v1/health` reports what the scheduler actually did, not what it was configured
to do:

```json
{"status":"ok","inflight":2,"blas_budget":8,"threads":8,"worker":-1,
 "slots":{"active":2,"cap":4},"steps":312,"deltas":270,"eous":0,"sessions":9,
 "cancelled":1,"offline_jobs":4,"offline_pending":0,
 "lag_p50_ms":144,"lag_max_ms":358.6,"streaming":true}
```

`inflight` = stream slots this worker is holding (a slot is held from the 101
until the ingest thread lets go) · `slots.cap` = `--cap` · `steps` = scheduler
steps that fed a chunk · `deltas`/`eous` = frames emitted · `sessions` = slots
claimed since start · `cancelled` = sessions ended by a cap or a dead peer ·
`offline_pending` = REST jobs waiting for a step · `lag_*` = emission lag since
start, from the same 8 ms histogram the `done` frame uses · `threads` =
`mynah_asr_num_threads()` · `blas_budget` = threads one inference may ask of
BLAS, now a constant (see below).

## Concurrency: one scheduler thread owns the model

The model is **read-only** (mmap'd weights) and shared. Inside a worker exactly
one thread — `mynah-sched` — calls the inference API, and the program asserts it
at every callback rather than promising it in a comment. The HTTP threads
(`mynah-http`) parse requests; for a WebSocket each becomes that connection's
ingest thread (`mynah-ingest`) for its life, converting PCM into the slot's
bounded ring. Output leaves on a per-connection writer (`mynah-out<fd>`).

One step: poll cancellation for every slot first, apply resets, then feed **one
encoder chunk to each ready slot in round-robin order**, then run at most one
batched offline call. A slot is ready when its ring holds
`mynah_asr_stream_need_samples()` — so a client uploading faster than real time
fills its ring, blocks on its own socket and can never take more than one chunk
per step from the others. When nothing is ready the scheduler waits on a
condvar: there is no tick, and the first chunk of a new stream runs the moment
it lands.

This is the design the sibling repos qualified, including the part they
falsified: a second submitter on the engine pool cost qwen-tts its cohort
(TTFA 174 → 1126 ms), so there is one submitter and only one.

**Offline REST work runs on the same thread**, batched **weight-stationary**:
padding-free packing of the frames of all queued requests, per-frame GEMM
(FFN/projections, >95% of FLOPs) on `[ΣT, d]` with weights read once per layer;
attention/conv stay per-sequence. Output identical to the B=1 path (verified).
A batch is one lookahead, because the batched call resolves it once for the
whole batch. Consequence to know: a long offline file stalls this worker's
streams for the duration of its step. v2.0 accepts that; offline-heavy
deployments get their own worker group (S2-6).

### Per-stream limits

| flag | default | what it does |
|---|---|---|
| `--cap C` | `--threads` | stream slots per worker; beyond it, `503 server_at_capacity` **before** the 101 |
| `--ring-seconds S` | 30 | PCM buffered per slot; beyond it the ingest stops reading and TCP throttles the client |
| `--idle-ms N` | 60000 | no audio for this long → the slot is cancelled with `idle_timeout` |
| `--max-frame-bytes N` | 1048576 | a larger WebSocket frame ends the stream with `frame_too_large` |
| `--max-pending N` | `2 × --threads` | offline requests queued for the scheduler; beyond it, `503 server_at_capacity` |

A client that stops **reading** is a different case from one that stops
**sending**, and it is worth being precise about which cap catches it. The
output ring is bounded and backpressure is cancellation, so a reader slow enough
to fill the ring or to hold a write past `SO_SNDTIMEO` (5 s) loses its stream —
but small JSON deltas take a long time to fill a socket buffer, so on a normal
loopback the cap that actually reclaims a silent client is `--idle-ms`. That is
what `tests/test_server_stream.sh` observes, and it is stated here rather than
claimed the other way round.

`SIGTERM` stops accepting, cancels live streams with an `error` frame carrying
`code:"shutting_down"`, lets the scheduler finish its step, joins it and frees
the weights.

Honest numbers on Apple Silicon (multithreaded Accelerate): batching ≈ thread pool for
throughput (a single GEMM already saturates the cores) — batching is worth ~1.4× over
sequential with a warm cache and reduces contention/footprint. The big gain is expected
on many-core x86/OpenBLAS and on future GPU backends (M5), where reading weights once
really matters. `--batch 1` disables it (back to per-request in the workers).

### BLAS threads: the knob stopped moving

Several inferences running at once must NOT each ask OpenBLAS for every core:
the concurrent calls thrash on its internal lock and aggregate throughput
collapses (measured on the A100 host: fine at 2 concurrent requests, collapsing
from 4 up). The v1 server therefore counted the inferences actually computing
and divided the budget between them.

With the scheduler there is **one inference in flight by construction**, so the
server sets `mynah_asr_blas_set_concurrency(1)` once at start and never touches
it again; `blas_budget` in `/v1/health` stays equal to `threads`, and a budget
that is anything else means something in the process is still driving the knob.
The library keeps the policy (`tests/test_threads` covers it) for embedders that
do run several inferences at once. An explicit `OPENBLAS_NUM_THREADS` still
wins, and `MYNAH_ASR_THREADS` sets the ceiling.

No effect with **Accelerate** (macOS), which nests through GCD and needs no
knob: there the budget is only bookkeeping, reported in `/v1/health`.

## Tests

- `make test-server` — REST (multipart, raw, verbose, errors), 4 concurrent
  requests, end-to-end WebSocket streaming. Skipped without the model.
- `make test-server-concurrency` — model-agnostic: concurrent REST, `inflight`
  and `blas_budget` back to rest, prefork byte-identity, a readable 503.
- `make test-server-stream` — the streaming gate: 4 real-time WebSocket streams
  whose text is byte-identical to `mynah-asr transcribe` of the same clip, a
  stalled reader taking only its own slot down, and the same identity under
  `--prefork 2 --cap 2`. Needs a streaming model, so it is skipped without one.

## Operational notes

- One model per process (`/v1/models` lists one).
- Timeouts/limits: body ≤ 200 MB, headers ≤ 64 KB, queue ≤ 128 connections,
  plus the per-stream caps in the table above.
- Threads, as `top -H` names them: `mynah-accept`/`mynah-recv` (the connection
  source) · `mynah-http` ×`--threads`, renamed `mynah-ingest` while one drives a
  stream · `mynah-sched` ×1 · `mynah-out<fd>` per streaming connection.
- TLS/auth out of scope: put behind a reverse proxy (nginx/caddy) in production.
