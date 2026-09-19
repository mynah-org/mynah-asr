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

`--cap` is the fleet's default; a `--model` group may set its own with `:cap=`,
and the ladder is evaluated per group (next section).

Design and the measurements it rests on: `.work/serving-v2-design.md`,
`.work/server-prefork.md`, `.work/server-scheduler.md`,
`.work/multi-model-serving.md`.

## Several models in one fleet (`--model`)

```sh
./mynah-asr-server \
    --model nemotron=models/nemotron-3.5-asr-streaming-0.6b:workers=6:cpus=24:cap=8:quant=int8 \
    --model parakeet=models/parakeet-tdt-0.6b-v3:workers=2:cpus=8:cap=4:quant=int8 \
    --default nemotron -p 8090
```

**One process holds one model.** That is the decision everything here rests on:
a batch is drawn from one process's slots, so it reads one set of weights *by
the shape of the address space* rather than by a check somebody has to remember.
So "several models" is several **worker groups**, never a registry inside one
address space — switching weights per request would cost a whole pass over them,
and the sibling repos costed exactly that and rejected it.

`--model name=dir[:workers=W][:cpus=N][:cap=C][:quant=int8|int4|f32][:lookahead=L]`
is repeatable and names one group:

| option | |
|---|---|
| `name=` | what a request routes by. Omitted (`--model dir`), the pack's own `name` from `mynah.json` |
| `:workers=W` | processes in the group. The counts must add up to `--prefork` |
| `:cpus=N` | cpus the group's workers share, carved contiguously out of the allowed mask in group order; each worker takes an equal share and, unless `--prefork-threads` is explicit, that share is its thread count |
| `:cap=C` | that group's stream slots per worker: rung 1 of the ladder, per group |
| `:quant=` | that group's quantization; without it, the fleet's `--quant` |
| `:lookahead=L` | the default streaming preset for that group's sessions when the client names none |

`--default <name>` picks the group a request that names no model gets; without
it, the first `--model`. The old `-m <dir>` still works and means exactly one
group, named by its own `mynah.json` — one group is the single-model server this
has always been, and every group-aware branch collapses.

**More than one `--model` implies `--prefork`**, because one process cannot hold
two models. If you did not give `--prefork`, the group count becomes it and the
banner says that it did:

```
mynah-asr-server: 2 model groups imply prefork; --prefork was not given, so it is 2
```

If you did give one and it does not fit the groups, that is an **error** and the
server refuses to start. Dropping a model the operator asked for, silently, is
the failure this whole feature exists to prevent.

```
prefork: models      2 resident, one model per worker (a batch is one process's slots,
                     so a batch is one model; `lang` stays a per-request parameter)
prefork:   nemotron                 workers 0-5 (6) · 24 cpus · 8 slots each (48) · rung2 queue per worker
prefork:   parakeet                 workers 6-7 (2) · 8 cpus · 4 slots each (8) · rung2 queue per worker
prefork:   default    nemotron (a request naming no model)
```

**Every model is opened in the parent, before the fork**, so the mapped weights
are one physical copy behind the whole tree; a child then **closes every group
but its own**, so one process can only ever run the model the router believes it
holds. A path that cannot be opened is a start-up failure, with nothing yet
forked and nothing yet listening — never a group that quietly answers 503
forever.

`--lid-model` is **not** a group. It is a detector in front of the models that
cannot detect a language themselves (Canary), resident in the parent like every
group's weights and inherited by every worker; nothing routes to it.

### How a request chooses its group

| where | shape |
|---|---|
| WebSocket | `GET /v1/audio/stream?model=nemotron&lang=it-IT` |
| REST, query | `POST /v1/audio/transcriptions?model=parakeet` |
| REST, form | a multipart field `model`, **before** the `file` part |
| nothing named | the `--default` group |

**Language is not a routing key here, and that is the difference from the TTS
siblings.** One Nemotron serves forty languages and the prompt is a per-request,
post-encoder one-hot, so a batch may freely mix languages and `lang` simply
travels with the request to the worker. Only the weights cannot be mixed.

The router classifies inside a **bounded, non-consuming `MSG_PEEK`** of at most
8 KiB: it reads the request line's query and, failing that, a multipart part
whose `Content-Disposition` names `model`. It never consumes a byte — the worker
still reads the whole request from the start — and it never interprets a method,
a route or a body's meaning. A connection whose prefix has not arrived yet is
parked with its own deadline (2 s) and looked at again, never waited on.

One consequence, stated rather than hidden: **a multipart `model` field that
arrives after the audio is past the peek bound** and the connection goes to the
default group. It is not then served by the wrong weights — the worker parses
the same field out of the whole body and refuses — so the cost is a wrong
refusal, never a wrong transcript. Put `model` before `file`, or in the query.

Names match case-insensitively, exactly first, then as a prefix of at least two
characters that matches exactly one group: `nemo` reaches `nemotron`. An
ambiguous prefix is refused rather than resolved.

### The refusals

| status | code | when |
|---|---|---|
| `404` | `model_not_found` | no group holds that name. The message names the accepted set; **no `Retry-After`**, because retrying will fail identically for as long as the server runs — residency is decided at start-up and printed, never grown on demand |
| `400` | `model_not_streaming` | a WebSocket to a group whose model cannot be streamed: it has no cache-aware presets, or it has them and uses something the incremental encoder does not implement (linear biases, a folded batch_norm, xscaling, symmetric conv padding, per-feature normalisation, a subsampling factor other than 8). The body says which. Before the upgrade, so the client reads a status and not a transport error |
| `503` | `server_at_capacity` | **that group** is full: every worker of it is at its slot cap and its share of the admission queue is full. A free slot in another group's worker is not capacity for this request |

```
$ curl -s "localhost:8090/v1/audio/transcriptions?model=whisper" -F file=@clip.wav
{"error":{"message":"no worker holds the requested model; this fleet serves: nemotron, parakeet",
          "type":"invalid_request_error","code":"model_not_found"}}
```

The admission ladder's rungs 1–3 are evaluated **per group**: rung 1 picks the
least-loaded worker *of that group* with a free slot, rung 2's bound is
`MYNAH_ASR_PREFORK_QUEUE × live workers of that group`, and rung 3's deadline is
checked at that group's own head. The queue is one array with a group tag,
scanned in arrival order per group, so every group has its own FIFO — a single
FIFO would let an entry for a saturated group block a ready entry for an idle one
behind it, which is a starvation channel introduced by the very partitioning
that exists to prevent starvation.

Gate: `tests/test_server_models.sh` (`make test-server-models`), which starts one
server with two groups and asserts all of the above, including that a REST
transcription to each group is byte-identical to that model's own CLI answer —
the two packs disagree about the clip, so a mis-route cannot pass.

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

**Query parameters.** Every key is validated **before** the upgrade; an unknown
key, or a value this server cannot honour, is an HTTP status naming the accepted
set — never a stream that silently runs with a default the client did not ask
for.

| key | default | accepted | refusal |
|---|---|---|---|
| `model` | the `--default` group | what `/v1/models` lists | `404 model_not_found` |
| `lang` | `auto` | any tag in the model's prompt dictionary | `400 language_not_served` |
| `lookahead` | model default | the model's presets (Nemotron: 0, 3, 6, 13) | `400 lookahead_not_available` |
| `format` | `s16le` | `s16le`, `f32le` | `400 unsupported_format` |
| `rate` | `16000` | `16000` | `400 unsupported_rate` |

Anything else is `400 unknown_query_parameter` with the list above in the
message.

**`rate=` is not resampled, and that is a decision, not an omission.** The
library's resampler (`mynah_asr_resample`) is a whole-buffer, stateless windowed
sinc: it keeps no filter history between calls, so running it per WebSocket
frame would inject a discontinuity at every frame boundary and change the
transcript in a way no identity gate would catch. The REST path resamples
because it has the whole file. Until a streaming resampler exists with its own
parity gate, a stream at another rate is refused and the client resamples.

**Client → server**

| frame | meaning |
|---|---|
| binary | PCM s16le 16 kHz mono, any size up to `--max-frame-bytes` |
| text `{"type":"finalize"}` | flush the tail, emit `done`, keep the socket: audio after that starts a new utterance |
| text `{"type":"reset","lang":"it-IT"}` | new utterance on the same socket, optionally a new language |
| ping | answered with a pong |
| close | finalize, emit `done`, close |

An unknown control `type` is answered with an `error` frame carrying
`unknown_control` and does **not** end the session; neither does a `reset`
naming a language this model does not hold (`language_not_served`) — a typo must
not cost a session. A binary frame larger than `--max-frame-bytes` does end it.
`finalize` followed by more audio is simply the next utterance on the same
socket: the stream is reset for you and `seq` keeps counting.

**The server pings** every `--ping-ms` (default 20000). A pong is not required —
`--idle-ms` is the rule and the ping is only there to keep middleboxes from
dropping an idle socket. Note the consequence, since it is measured rather than
assumed: idle is "no frame of ANY kind", and a pong is a frame, so a client that
answers the pings is never idle. `--idle-ms` reclaims a client that has stopped
answering altogether, which is the case it exists for.

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

**Error codes**: `idle_timeout` · `audio_limit` · `frame_too_large` ·
`peer_gone` · `shutting_down` · `decode_failed` · `unknown_control` ·
`language_not_served` · `unsupported_opcode` · `model_not_streaming`.
`audio_limit` is announced and then **finalised**: the audio already accepted is
still owed a transcript, so the cap flushes the tail, emits `done` and closes.

**Refusals happen before the upgrade**, as HTTP statuses, so a client reads a
status and not a transport error: `503` with `error.code` `server_at_capacity`
and `Retry-After` when every slot of this worker is taken, `400` with
`model_not_streaming` on an offline-only model (Parakeet, Canary — they have no
cache-aware streaming presets and never will, so there is no `Retry-After`), and
the query refusals in the table above.

Every refusal, on the WebSocket path and on the REST one, leaves through a
**lingering close**: the response is written, the socket is half-closed, whatever
the client still had in flight is drained, and only then is the descriptor
closed. A `close()` with unread bytes in the receive queue makes the kernel send
an RST, and the RST discards the response along with it — which is how a
documented 400 reaches a client as `ECONNRESET`. One implementation serves both
(`mynah_asr_prefork_linger_close`), so the worker refuses the way the router
does.

Reference client (Python stdlib): `tools/eval/ws_client.py`. Load harness:
`tools/bench/stream_load.py`.

### GET /v1/models · GET /v1/health · OPTIONS (CORS)

`/v1/models` is the **actual routing table**: one entry per `--model` group, with
the name a request routes by, the pack's engine, whether that group can stream,
and which one is the default. Every worker answers it identically — the names
survive the close of the other groups' weights — so it is the same list the
router routes by and the same one a `model_not_found` quotes.

```json
{"data":[{"id":"nemotron","object":"model","owned_by":"mynah",
          "engine":"nemotron-streaming","streaming":true,"default":true},
         {"id":"parakeet","object":"model","owned_by":"mynah",
          "engine":"parakeet-tdt","streaming":false}]}
```

`/v1/health` reports **facts**: what this process actually did. The
configuration it was given is on the `[SERVER-CONFIG]` banner line, printed once
at start — a number that is configuration has no business in a counter, and a
health endpoint that reports both is one that will be quoted for the wrong one.

```json
{"status":"ok","inflight":2,"blas_budget":8,"threads":8,"worker":-1,
 "slots":{"active":2,"cap":4},"steps":312,"deltas":270,"eous":0,"sessions":9,
 "cancelled":1,
 "cancelled_by":{"idle_timeout":1,"peer_gone":0,"frame_too_large":0,
                 "protocol_error":0,"shutting_down":0,"audio_limit":0,
                 "decode_failed":0,"other":0},
 "offline":{"queued":0,"done":4,"max_pending":8},
 "audio_seconds":5.229,
 "lag_ms":{"p50":144,"p95":312,"max":358.6,"count":270,"bucket_ms":8},
 "streaming":true,
 "batch":{"batched_steps_total":312,"rows_stacked_total":1248,"ready_sum":884,
          "ready_mean":2.83,"reserved_slots":8,
          "step_wall_ms":{"sum":79560.0,"count":312,"mean":255.0},
          "by_b":{"1":{"steps":80,"wall_ms_sum":14984.0,"wall_ms_mean":187.3},
                  "4":{"steps":132,"wall_ms_sum":45672.0,"wall_ms_mean":346.0}}},
 "model":{"name":"nemotron-3.5-asr-streaming-0.6b","engine":"nemotron-streaming",
          "quant":"f32","lookahead_default":3},
 "group":"nemotron",
 "groups":"nemotron=6 parakeet=2",
 "process":{"worker":-1,"pid":6392,"uptime_s":30.2,"pool_threads":8,
            "prefork_threads":0,"http_threads":4,"pinned":false,
            "cpu_mask":"unpinned","build":"v0.9.1-38-gd480467",
            "blas":"accelerate","simd":"neon+dotprod","int8_kernel":"neon-sdot",
            "int4_kernel":"neon-sdot-q4","int8_gemm":"off"},
 "refused":{"server_at_capacity":2,"unknown_query_parameter":1}}
```

| field | what it is |
|---|---|
| `inflight`, `slots` | stream slots this worker holds now; `cap` is `--cap` |
| `steps`, `deltas`, `eous` | scheduler steps that fed a chunk, frames emitted |
| `sessions` | slots claimed since start |
| `cancelled`, `cancelled_by` | sessions ended early, bucketed **by the `code` the client was sent** — the counter and the error frame can never name two different things |
| `offline` | REST jobs `queued` now, `done` since start, and `--max-pending` |
| `audio_seconds` | seconds of audio fed to the model, streams and REST alike |
| `lag_ms` | emission lag since start, from the 8 ms histogram (`bucket_ms`); `p50`/`p95` are therefore quantised to 8 ms, which is a measurement — a percentile computed from a mean is not |
| `batch` | what the batched stream step did (S2-2b). `batched_steps_total` counts the calls, one per step that had a ready set; `rows_stacked_total` is the library's own count of encoder rows that went through the STACKED path, so **0 next to a non-zero `batched_steps_total` means every step degraded to per-stream steps** rather than a silent fallback; `ready_mean` is `ready_sum / batched_steps_total`; `step_wall_ms` is time spent inside the call (model only: building the set and the frames are outside it); `by_b` is the same sum/count split by ready-set size, which is what the cadence law `T_step(B) = a + b·B` is fitted from |
| `model` | what THIS process holds, read from the pack's own `mynah.json` |
| `group`, `groups` | the `--model` group this worker serves, and the whole fleet's split (`name=workers`, space separated). `groups` is `""` in a single-model fleet, which is itself the answer. One probe therefore says both "what did I reach" and "what else is there" |
| `process.pinned`, `cpu_mask` | read BACK from the kernel (`sched_getaffinity`), never the mask that was requested. `false`/`unpinned` on macOS, which has no affinity API this server uses |
| `process.build/blas/simd/int8_kernel` | the same values `--dispatch-map` prints, from the same predicates (`src/qmat.c`) |
| `refused` | refusals **this worker** issued, by the `code` in the error body. The prefork router counts its own separately (`/metrics`, and its final table) |

## The start-up banner

Every server process prints four machine-readable lines to **stderr** before it
serves anything, unconditionally — there is no flag that turns them off, because
the one run whose banner is missing is the run that will be quoted
(`ENGINEERING.md` §5).

```
[FLAGS] v=1 MYNAH_ASR_THREADS=8
[EFFECTIVE-CONFIG] v=1 build=v0.9.1-38-gd480467 blas=accelerate simd=neon+dotprod MYNAH_ASR_THREADS=8->8(applied)
[SERVER-CONFIG] v=1 model_dir=... model=nemotron-3.5-asr-streaming-0.6b engine=nemotron-streaming
    quant=f32 lid_model=none streaming=yes lookahead_default=3 lookahead_presets=3,0,6,13
    chunk_ms=320 port=8397 cap=4 ring_s=30 idle_ms=60000 ping_ms=20000 max_audio_s=14400
    max_frame_bytes=1048576 max_pending=8 batch=8 http_threads=4 pool_threads=8
    blas_budget=8 prefork=single-process worker=-1 metrics=on
[TOPOLOGY] v=1 worker=-1 pid=6392 configured_mask=inherited actual_mask=unpinned threads=8 pinned=no
```

- `[FLAGS]` / `[EFFECTIVE-CONFIG]` come from `src/flags.c`, the same two lines
  the CLI prints: every registered environment variable that is **set**, and
  what this build on this host actually does with it (`applied`, `clamped`,
  `IGNORED: <why>`).
- `[SERVER-CONFIG]` is one token per field, whitespace inside a value replaced
  by `_`. `chunk_ms` is `(lookahead_default + 1) × encoder_frame_ms`, taken from
  the model's `mynah.json` — it is the cadence the emission-lag thresholds below
  are multiples of.
- `[TOPOLOGY]` is printed **per worker**, by the worker, right after it pins
  itself, and `configured_mask` vs `actual_mask` is the point: the second is read
  back from the kernel, so a cgroup, an inherited `taskset` or a failed
  `sched_setaffinity` shows up as a difference instead of as a slow server.
- `mynah-asr-server --dispatch-map [--json]` prints the same dispatch table the
  CLI does, resolved from the owners' predicates, and exits. A serving benchmark
  that cannot state its kernels is not a measurement.

## `/metrics` — Prometheus text, on its own port

Off by default. `--metrics-port N` turns it on; `--metrics-bind ADDR` (default
`127.0.0.1`) says where.

```
mynah-asr-server -m models/nemotron-3.5-asr-streaming-0.6b --metrics-port 9090
curl -s localhost:9090/metrics
```

**Never the service port.** A scrape must not queue behind a 200 MB upload, and
an observability surface should not be reachable from wherever the audio clients
are. **Never `SO_REUSEPORT`**: two processes silently sharing the port would
answer half the scrapes with the other one's counters under the same labels, so
a second bind on a live metrics port fails, loudly, with the port named.
`SO_REUSEADDR` *is* set, and only that — it lets a restart re-bind over the
`TIME_WAIT` of the scrapes the previous run answered.

**No thread, no background collection.** The page is rendered on request from
counters the process already keeps; the accept loop that owns the listener
serves it in a bounded slice (2 ms for the request bytes, 200 ms for the write,
the FIN and the drain). Nothing was added to the per-step path beyond two
relaxed atomic adds next to adds that were already there (audio samples and the
lag sum).

**Rate limit**: a token bucket, 5 scrapes/s with a burst of 10. Over that the
answer is `429` and the page is *not* rendered — rendering is the cost the limit
exists to bound.

What a worker exports (all series carry `worker`, `-1` = a single-process
server):

| series | |
|---|---|
| `mynah_asr_sessions_total` · `mynah_asr_steps_total` · `mynah_asr_deltas_total` · `mynah_asr_eou_total` | counters |
| `mynah_asr_audio_seconds_total` | **the throughput unit**: any RTF claim about this server has this as its denominator |
| `mynah_asr_offline_jobs_total` · `mynah_asr_offline_queued` | REST jobs done, and waiting now |
| `mynah_asr_cancelled_total{reason}` | `idle_timeout` `peer_gone` `frame_too_large` `protocol_error` `shutting_down` `audio_limit` `decode_failed` `other` |
| `mynah_asr_refused_total{code}` | this worker's own refusals, by the code in the error body |
| `mynah_asr_emission_lag_ms_sum` / `_count` / `_max` | a sum-and-count pair, not a histogram |
| `mynah_asr_emission_lag_over_ms_total{le}` | **exact** counts of deltas whose lag was **at least** the labelled ms. The thresholds are one and two chunk periods plus 1000 ms, rounded up to an 8 ms bucket edge — so on the v1 target they are `320`, `640`, `1000`, and the count is exact rather than interpolated |
| `mynah_asr_batched_steps_total` · `mynah_asr_batch_rows_stacked_total` · `mynah_asr_batch_ready_size_sum` | the batched step: calls, encoder rows actually stacked, and the summed ready-set size (÷ calls = the mean). `rows_stacked` at 0 with a non-zero call count is the visible fallback, not a healthy run |
| `mynah_asr_step_wall_ms_sum` / `_count` | milliseconds inside the batched step call. The per-`B` split is in `/v1/health` and deliberately not here: a `B` label would grow the label set with `--cap` |
| `mynah_asr_slots_active` · `mynah_asr_slots_cap` | gauges |
| `mynah_asr_build_info{build,blas,simd,int8_kernel}` | always 1; the labels are the point |
| `mynah_asr_uptime_seconds` | gauge |

**Under `--prefork` the PARENT answers**, and it says so in a comment line at
the top of the page. The router holds the routing table and never enters the
model, so it exports `mynah_asr_worker_up` / `_inflight` / `_slots` /
`_assigned_total` / `_completed_total` / `_over_service_cap_total` per worker,
its own `mynah_asr_refused_total{code}` and the admission-queue gauges — and
**not** sessions, steps, deltas, audio seconds or emission lag, because it does
not have them. In a **multi-model** fleet those per-worker series carry a second
label, `model`, naming the group that worker serves, so a scrape separates "the
parakeet group is saturated" from "the fleet is busy". It is added only there and
only then, because its value set is exactly the configured groups — bounded by
the CLI before the first request, which is what the cardinality rule asks. A worker's own scheduler counters are exported by that worker
only if it is given a metrics port of its own. Per-worker series are never
summed here: a fleet total hides the one worker that stopped.

**What is deliberately NOT exported.** No client-side latency: no TTFP, no stall
rate, no "safe play start". Those are measured at the far end of a socket this
process does not own, and a server that reports them is reporting a guess. They
belong to the benchmark harness (`tools/bench/`), which is the only thing that
may quote them. **Cardinality** is a contract: the only labels are `worker`,
`reason`, `code` and `le`, each from a fixed compile-time set. Never a language,
never a model path, never text, never anything a client can choose. `model` is
the one addition and it is not an exception to that: its values are the
`--model` names the operator typed, fixed at start-up, and a client cannot
invent one — a request naming a group that does not exist is refused before it
is ever counted under a label.

## `SIGUSR1` — one dump, to stderr

`kill -USR1 <pid>` makes a server print everything above once, bracketed, and
keep running. Sent to a prefork **parent** it prints the routing table and
forwards the signal to every worker, so one signal produces the whole machine's
view.

```
[DUMP] v=1 worker=0 seq=1 begin
[DUMP] worker=0 seq=1 process pid=1720 uptime_s=41.3 pool_threads=4 blas_budget=4 pinned=no mask=unpinned
[DUMP] worker=0 seq=1 build=... blas=accelerate simd=neon+dotprod int8_kernel=neon-sdot int8_gemm=off
[DUMP] worker=0 seq=1 model=... engine=... quant=f32 streaming=yes lookahead_default=3 chunk_ms=320
[DUMP] worker=0 seq=1 slots active=0 cap=2 sessions=1 steps=17 deltas=15 eous=0 audio_s=5.2
[DUMP] worker=0 seq=1 batch steps=17 rows_stacked=0 ready_mean=1.00 step_wall_ms_mean=187.3
[DUMP] worker=0 seq=1 offline queued=0 done=4 max_pending=4
[DUMP] worker=0 seq=1 cancelled=0 idle_timeout=0 peer_gone=0 ... other=0
[DUMP] worker=0 seq=1 refused none
[DUMP] worker=0 seq=1 lag_ms p50=144 p95=312 max=358.6 count=15 sum=2160 bucket_ms=8 over_320ms=0 over_640ms=0 over_1000ms=0
[DUMP] v=1 worker=0 seq=1 end
```

The whole dump is composed into one buffer and written once: W workers write to
the same stderr at the same moment, and a dump built out of a dozen `fprintf`
calls comes back with worker 0's line spliced into the middle of worker 1's —
which is what the first run of this actually produced. `seq` increments per
dump, so two dumps can never be read as one.

The handler is installed by `server/main.c` **before** the fork, and workers
inherit it. That is load-bearing rather than tidy: the default action for
`SIGUSR1` is to *terminate*, so a fleet without a handler loses every worker the
first time anyone asks it for statistics — which is exactly what happened the
first time this was exercised.

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
| `--idle-ms N` | 60000 | no frame of any kind for this long → the slot is cancelled with `idle_timeout` |
| `--ping-ms N` | 20000 | server-side WebSocket ping period; `0` never pings |
| `--max-audio-seconds S` | 14400 | audio one stream may send; beyond it, `error audio_limit`, then finalize and close. `0` = no cap |
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

Since S1-6 the **Linux default build has no OpenBLAS at all** (`BLAS=none`):
`src/sgemm.c` computes every f32 GEMM on the same pool as everything else, so
there is no second team to divide. `blas_budget` is then the pool width by
definition and `mynah_asr_blas_set_concurrency()` cannot move it —
`--dispatch-map` says so on the `threads.blas_budget` row, and
`OPENBLAS_NUM_THREADS` is reported IGNORED on the `[EFFECTIVE-CONFIG]` line.
The knob and the collapse above still apply to `make BLAS=openblas`, the
comparison build.

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
- `make test-server-metrics` — the observability surface, model-agnostic and
  REST-only: the four banner lines; `/metrics` and `/v1/health` agreeing on the
  same counters; `audio_seconds_total` equal to 4 x the clip; a second bind on
  the metrics port refused; a burst of 20 scrapes answered with 429s; a
  bracketed `[DUMP]` that does not kill the process; and, under `--prefork 2`,
  the router answering for both workers.
- `make test-server-protocol` — protocol v2 and the per-worker admission ladder:
  three utterances on ONE socket separated by `finalize`/`reset`, each
  byte-identical to the CLI with `seq` continuing; an unknown control message and
  an unserved `reset` language survived; the pre-upgrade `400` and `503` read as
  HTTP with their bodies and `Retry-After`; an idle client cancelled with
  `idle_timeout` and a stream cut off at `--max-audio-seconds` with
  `audio_limit`; the server's own pings observed; `SIGTERM` on a live stream
  answered with `shutting_down`, exit 0, no survivors. Model-gated.
- `make test-server-models` — several models in one fleet (S2-6): ONE server with
  two `--model` groups and `--prefork 2`. `/v1/models` lists exactly the two
  names; a REST transcription to each group (query and multipart field both) is
  byte-identical to THAT model's own CLI answer — the two packs disagree about
  the clip, which is what makes a mis-route impossible to pass; a WebSocket to
  the streaming group is byte-identical to `mynah-asr stream`; a WebSocket to the
  offline group is `400 model_not_streaming`; an unknown name is `404
  model_not_found` with the accepted set and no `Retry-After`; filling one
  group's slots refuses `server_at_capacity` while the other group still serves;
  `SIGTERM` leaves no worker. Needs BOTH a streaming and an offline model, and
  skips (77) without either.

## Operational notes

- One model per process, and several models are several worker groups
  (`--model`, above). `/v1/models` lists the groups; a worker holds exactly one
  of them and closes the rest at start-up.
- Timeouts/limits: body ≤ 200 MB, headers ≤ 64 KB, queue ≤ 128 connections,
  plus the per-stream caps in the table above.
- Threads, as `top -H` / `htop` / `ps -L` name them (verified, not intended):
  `mynah-accept` (single process) or `mynah-recv` (a prefork worker) — the
  connection source, one per process · `mynah-http` ×`--threads`, each renamed
  `mynah-ingest` for the life of a WebSocket it is driving and back afterwards ·
  `mynah-sched` ×1, the only thread that enters the model · `mynah-out<fd>`, one
  per streaming connection, named with its own descriptor so a stuck writer can
  be tied to a socket.
- TLS/auth out of scope: put behind a reverse proxy (nginx/caddy) in production.
