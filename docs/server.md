# mynah-asr-server — HTTP + WebSocket API

```sh
make && ./mynah-asr-server -m models/nemotron-3.5-asr-streaming-0.6b -p 8090 --threads 4
```

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

### GET /v1/audio/stream — WebSocket streaming

Query: `?lang=auto&lookahead=3`. Protocol:
- client → server: **binary** frames with PCM s16le 16 kHz mono (any size);
- server → client: JSON text frames `{"text": "<final delta>", "language": ...,
  "audio_seconds": ...}` as the text is finalized;
- on client close the server processes the tail, sends `{"done": true,
  "language": "..."}` and closes.

Reference client (Python stdlib): `tools/eval/ws_client.py`.

### GET /v1/models · GET /v1/health · OPTIONS (CORS)

`/v1/health` also reports the adaptive-BLAS state, handy under load:

```json
{"status":"ok","inflight":1,"blas_budget":5,"threads":10}
```

`inflight` = inferences computing right now (a batch counts as one), `threads` =
`mynah_asr_num_threads()`, `blas_budget` = threads one inference may ask of BLAS.
At rest `blas_budget == threads`.

## Concurrency

The model is **read-only** (mmap'd weights) and shared across workers: each request
only holds its own decode state (~12 MB per stream). `--threads N` = requests served
in parallel; excess requests queue up (503 beyond 128 queued). No model cloning,
no locks on the hot path.

**Cross-request batching** (`--batch N`, default 8): pending REST transcriptions
are aggregated (25 ms window) and processed **weight-stationary**: padding-free
packing of the frames of all requests, per-frame GEMM (FFN/projections, >95% of FLOPs)
on `[ΣT, d]` with weights read once per layer; attention/conv stay per-sequence.
Output identical to the B=1 path (verified).

Honest numbers on Apple Silicon (multithreaded Accelerate): batching ≈ thread pool for
throughput (a single GEMM already saturates the cores) — batching is worth ~1.4× over
sequential with a warm cache and reduces contention/footprint. The big gain is expected
on many-core x86/OpenBLAS and on future GPU backends (M5), where reading weights once
really matters. `--batch 1` disables it (back to per-request in the workers).

### Adaptive BLAS threads (OpenBLAS)

Several inferences running at once must NOT each ask OpenBLAS for every core: the
concurrent calls thrash on its internal lock and aggregate throughput collapses
(measured on the A100 host: fine at 2 concurrent requests, collapsing from 4 up;
capping the threads restored it). The server therefore counts the inferences
actually computing and sets the per-call budget to `threads / inflight` —
previously this needed `OPENBLAS_NUM_THREADS` tuned by hand.

The count is kept around the compute calls only, so a WebSocket stream sitting
idle between chunks does not hold a slot down, and the knob is written only when
the value really changes, so a steady-state server pays nothing for it. An
explicit `OPENBLAS_NUM_THREADS` in the environment still wins, and
`MYNAH_ASR_THREADS` sets the ceiling being divided.

No effect with **Accelerate** (macOS), which nests through GCD and needs no knob:
there the budget is only bookkeeping, reported in `/v1/health`. Consequence worth
stating plainly: the *policy* is unit-tested (`tests/test_threads`) and the
*accounting* is asserted end-to-end (`make test-server`), but the throughput win
itself has only been measured on the A100 host — re-measuring it needs a
many-core Linux box with OpenBLAS.

## Tests

`make test-server` — REST (multipart, raw, verbose, errors), 4 concurrent requests,
end-to-end WebSocket streaming. Automatically skipped if the model is not downloaded.

## Operational notes

- One model per process (`/v1/models` lists one).
- Timeouts/limits: body ≤ 200 MB, headers ≤ 64 KB, queue ≤ 128 connections.
- TLS/auth out of scope: put behind a reverse proxy (nginx/caddy) in production.
