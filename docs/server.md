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

## Tests

`make test-server` — REST (multipart, raw, verbose, errors), 4 concurrent requests,
end-to-end WebSocket streaming. Automatically skipped if the model is not downloaded.

## Operational notes

- One model per process (`/v1/models` lists one).
- Timeouts/limits: body ≤ 200 MB, headers ≤ 64 KB, queue ≤ 128 connections.
- TLS/auth out of scope: put behind a reverse proxy (nginx/caddy) in production.
