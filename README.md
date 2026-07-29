<p align="center">
  <img src="assets/mynah-logo3-generic.png" alt="Mynah ASR" width="320">
</p>

# Mynah ASR

[![CI](https://github.com/mynah-org/mynah-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/mynah-org/mynah-asr/actions/workflows/ci.yml)
[![Code Quality](https://github.com/mynah-org/mynah-asr/actions/workflows/codeql.yml/badge.svg)](https://github.com/mynah-org/mynah-asr/actions/workflows/codeql.yml)
[![Memory Safety](https://github.com/mynah-org/mynah-asr/actions/workflows/safety.yml/badge.svg)](https://github.com/mynah-org/mynah-asr/actions/workflows/safety.yml)
[![Release](https://img.shields.io/github/v/release/mynah-org/mynah-asr?color=blueviolet)](https://github.com/mynah-org/mynah-asr/releases/latest)
[![Models](https://img.shields.io/badge/models-10-blue)](docs/models.md)
[![Languages](https://img.shields.io/badge/languages-40-brightgreen)](docs/nemotron-languages.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A fast native C inference engine for speech recognition & translation** —
llama.cpp-style: streaming and offline, CPU/Metal/CUDA, no Python at runtime.

Today it runs the best open speech models (Parakeet, Canary, Nemotron); the
engine layer is model-agnostic and built to host more families tomorrow.

```
$ mynah-asr transcribe -m models/parakeet-tdt-0.6b-v3 -i audio.wav --timestamps
[5.2s audio | load 0.04s | inference 0.48s | RTF 0.092 | lang=auto]
  0.00   0.56  Ciao,
  0.64   0.72  questo
  ...

$ mynah-asr transcribe -m models/canary-1b-v2 -i italian.wav --lang it --target-lang en
The satellite, in space, receives the signal and then sends it back almost instantly.

$ rec -q -t raw -r 16000 -e signed -b 16 -c 1 - | mynah-asr stream -m models/nemotron-3.5-...
(live cache-aware transcription, selectable 80 ms – 1.12 s latency)
```

## What it is

Mynah is not "another speech recognizer": it is a **native runtime for modern ASR
models**, in the spirit of `llama.cpp` / `whisper.cpp`. One shared FastConformer
encoder, interchangeable decoders behind a vtable (`engine.h`), streaming as a
first-class citizen.

- **Pure C11**, zero runtime dependencies (only BLAS: Accelerate/OpenBLAS)
- **10 models across 4 decoder families** — RNNT, TDT, CTC and AED (Canary)
- **Speech translation**: Canary translates speech (25 EU languages ↔ English
  with canary-1b-v2, en↔de/es/fr with the flash models) — CLI `--target-lang`,
  server `/v1/audio/translations`
- **CPU-first** — warm offline RTF 0.015–0.06 on Apple Silicon (long audio);
  optional **Metal** backend (−25…45%) and **CUDA** (validated on A100: up to 2.9×
  vs a 22-core EPYC, identical transcripts); int8/int4
  with native SDOT/VNNI kernels
- **Cache-aware streaming** (Nemotron): runtime-selectable 80 ms–1.12 s latency,
  emitted text is never retracted, **byte-identical to the offline path**
- **40 languages** with language detection (Nemotron), 25 EU languages with
  PnC+ITN (Parakeet v3, canary-1b-v2)
- **Per-word timestamps**, automatic silence-based segmentation of long files
  (model-aware), weight-stationary batching, OpenAI-compatible REST + WebSocket
  server
- **Two weight containers**: safetensors (default, zero-copy mmap) or **GGUF**
  (F32/F16/BF16/Q8_0/Q4_0, `tools/export_gguf.py`) — same code path after load;
  can also **import third-party community GGUFs** without torch
  (`tools/import_gguf.py`, Parakeet/TDT — *WIP*)
- **Lightweight bindings**: Python (ctypes) and **Node** (koffi) over `libmynah_asr` —
  thin FFI wrappers, no build step committed to the repo
- Quality measured on **real audio**: CER 0.00–0.07 across 11 languages
  ([hear it below](#hear-it--11-languages-one-sentence)), translations scored
  against parallel references — `make test-samples`
- Python only as offline tooling (weight conversion, reference oracle, eval)

## Supported models

**v0.5.0, feature-complete toward v1** — 10 working models
(full catalog with verified configs: [docs/models.md](docs/models.md)):

| Model | What it does | Status |
|---|---|---|
| [nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | **streaming** cache-aware + offline, 40 languages, LID | ✅ |
| [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | offline TDT, 25 EU languages, PnC+ITN | ✅ |
| [parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | TDT+CTC 114M EN — the fastest | ✅ |
| [parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) / [ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) | offline EN | ✅ |
| [parakeet-rnnt-1.1b](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) / [ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | offline EN, 42 layers | ✅ |
| [canary-180m-flash](https://huggingface.co/nvidia/canary-180m-flash) / [1b-flash](https://huggingface.co/nvidia/canary-1b-flash) | ASR en/de/es/fr + **translation** + word timestamps | ✅ |
| [canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | ASR in **25 EU languages** + en↔24 translation, ITN | ✅ |

## Performance — one runtime, four backends

Every cell is an **RTF** (real-time factor) = *seconds of compute per second of
audio*, so **lower is faster**: RTF 0.05 means a 60-second recording is
transcribed in 3 seconds, i.e. **20× faster than realtime**. Warm runs on ~60 s
of audio, f32 weights, one request at a time. Full matrix, int8/int4 numbers and
methodology in [docs/benchmarks.md](docs/benchmarks.md).

| model | M1 CPU (NEON) | M1 Metal | EPYC 22c (x86 AVX2) | A100 (CUDA) | best ⇒ ×realtime |
|---|:---:|:---:|:---:|:---:|:---:|
| parakeet-tdt_ctc-110m | 0.015 | **0.010** | 0.015 | 0.011 | **100×** |
| parakeet-ctc-0.6b | 0.042 | **0.022** | 0.040 | **0.022** | **45×** |
| parakeet-tdt-0.6b-v3 | 0.047 | 0.030 | 0.065 | **0.028** | **36×** |
| nemotron-3.5-asr-streaming-0.6b¹ | 0.055 | 0.040 | 0.059 | **0.022** | **45×** |
| parakeet-rnnt-1.1b | 0.068 | 0.041 | 0.066 | **0.038** | **26×** |
| canary-1b-flash (AED, +translation) | 0.143 | 0.081 | 0.113 | **0.063** | **16×** |

Bold marks the fastest backend of the row; the last column is that same number
inverted (1 / RTF) for whoever reads speed as "× realtime" — e.g. the 110m
model transcribes an hour of audio in ~36 seconds on Metal.

¹ A100/EPYC measured after the banded-attention fix (2026-07-20); the M1
columns predate it and will improve on re-measure.
M1: 16 GB, ~65 s file (2026-07-18) · x86/CUDA: Ubuntu 24.04, 60 s LibriSpeech
(2026-07-20, TF32 tensor-core default — transcripts identical to CPU on
every model). Parallel requests on the A100 via the batch API: 110m ~94×,
nemotron ~46× realtime aggregate (`make bench-throughput`).

RAM: 110m 0.44 GB · 180m 0.71 · 0.6B ~2.4 · 1b-flash 3.3 · 1.1b 4.0 GB
(int8: ~⅓; on M1 int8 halves Canary's AED decode and triples Nemotron
streaming). Nemotron streaming: ~26 ms of compute per 80 ms chunk (9 ms int4).

Every numeric stage is validated against a numpy reference oracle
(`make test`: bit-exact mel, f32-tolerance encoder, streaming ≡ offline).

## Quickstart

**1 — Get the binary.** Either grab a
[prebuilt tarball](https://github.com/mynah-org/mynah-asr/releases/latest)
(linux-x86_64, linux-aarch64, darwin-arm64 — CLI + server + `libmynah_asr.a` +
header, with `SHA256SUMS`), or build in a few seconds:

```sh
git clone https://github.com/mynah-org/mynah-asr.git && cd mynah-asr
make          # macOS: Accelerate + Metal, zero deps
              # Linux: sudo apt install libopenblas-dev  (Fedora: openblas-devel)
```

**2 — Get a model.** `scripts/download_model.sh` fetches any supported
checkpoint straight from HuggingFace — the 10 models above plus 2 community
GGUF ports, no account and no token needed — and prints the exact convert
command for the one you picked:

```sh
scripts/download_model.sh --list            # alias, size and capabilities of each
scripts/download_model.sh                   # interactive menu
scripts/download_model.sh --model nemotron  # ...or by alias, for scripts and CI

# convert to mynah's format (offline tooling, run once per model)
cd tools && uv sync && uv run python convert_nemo.py ../models/nemotron-3.5-asr-streaming-0.6b && cd ..
```

| if you want… | alias | download |
|---|---|---|
| the lightest possible first run — community GGUF, no torch | `110m-gguf` | 90 MB |
| the fastest English model | `110m` | 0.5 GB |
| streaming + 40 languages with language detection | `nemotron` | 2.6 GB |
| offline, 25 EU languages, punctuation + ITN | `tdt-v3` | 2.4 GB |
| speech **translation**, en ↔ 24 languages | `canary-v2` | 3.9 GB |

**3 — Transcribe, or stream.**

```sh
# offline file (any sample rate: WAV is resampled automatically)
./mynah-asr transcribe -m models/nemotron-3.5-asr-streaming-0.6b -i file.wav --lang auto

# optional: int8 checkpoint — 0.79 GB instead of 2.55, instant load
./mynah-asr quantize -m models/nemotron-3.5-asr-streaming-0.6b --quant int8
./mynah-asr transcribe -m models/nemotron-3.5-asr-streaming-0.6b -i file.wav --quant int8

# live stream from mic or pipe (raw s16le 16 kHz mono on stdin)
ffmpeg -v quiet -i anything.mp3 -f s16le -ar 16000 -ac 1 - | \
  ./mynah-asr stream -m models/nemotron-3.5-asr-streaming-0.6b --lookahead 3
```

For mp3/m4a: `ffmpeg -i file.mp3 -ar 16000 -ac 1 out.wav`. Languages and latency
presets: [docs/nemotron-languages.md](docs/nemotron-languages.md) ·
[docs/streaming.md](docs/streaming.md).

## Hear it — 11 languages, one sentence

The quality suite runs on **real committed audio** ([samples/](samples/README.md)):
parallel [FLEURS](https://huggingface.co/datasets/google/fleurs) clips (CC-BY 4.0),
the *same sentence* read by native speakers in 11 languages. Click a clip to
download the WAV (release asset); the text is the reference transcription —
mynah's output matches it with **CER 0.00–0.07** (`make test-samples`).

| | clip | reference transcription |
|---|---|---|
| 🇮🇹 it | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/it_fleurs_1521.wav) | Il satellite nello spazio riceve il segnale e poi lo rimanda indietro quasi all'istante. |
| 🇺🇸 en | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/en_fleurs_1521.wav) | The satellite in space gets the call and then reflects it back down, almost instantly. |
| 🇩🇪 de | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/de_fleurs_1521.wav) | Der Satellit im Weltraum empfängt den Anruf und reflektiert ihn dann fast sofort zurück nach unten. |
| 🇪🇸 es | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/es_fleurs_1521.wav) | El satélite en el espacio recibe la llamada y, luego, la refleja de vuelta casi de forma instantánea. |
| 🇫🇷 fr | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/fr_fleurs_1521.wav) | Le satellite reçoit l'appel dans l'espace puis le renvoie sur Terre, presque instantanément. |
| 🇵🇹 pt | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/pt_fleurs_1521.wav) | O satélite no espaço recebe a chamada e depois a redireciona de volta, quase instantaneamente. |
| 🇳🇱 nl | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/nl_fleurs_1521.wav) | Zodra de ruimtesatelliet de oproep ontvangt, wordt deze meteen teruggezonden. |
| 🇵🇱 pl | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/pl_fleurs_1521.wav) | Połączenie trafia do satelity w przestrzeni kosmicznej, po czym niemal natychmiast odbija go z powrotem. |
| 🇷🇺 ru | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/ru_fleurs_1521.wav) | Спутник в космосе принимает звонок и практически мгновенно отражает его обратно вниз. |
| 🇺🇦 uk | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/uk_fleurs_1521.wav) | Супутник у космосі отримує виклик і потім майже одразу відображає його назад. |
| 🇯🇵 ja | [⬇ fleurs_1521](https://github.com/mynah-org/mynah-asr/releases/download/v0.1-samples/ja_fleurs_1521.wav) | 宇宙にある人工衛星は通話を受信して、ほぼ瞬時にそれを反映します。 |

Because the sentences are parallel, the English clip doubles as the reference
for scoring Canary's **speech translation** (e.g. it→en at the top of this page).
Longer clips (~2–5 min) exercise segmentation, timestamps and streaming.

## Languages

| engine | ASR | translation |
|---|---|---|
| Nemotron 3.5 (streaming) | **40 locales** in 3 tiers (19 transcription-ready — it-IT FLEURS WER 4.25%), auto language detection | — |
| Parakeet tdt-0.6b-v3 | **25 EU languages**, PnC + ITN | — |
| canary-1b-v2 | 25 EU languages, ITN | **en ↔ 24 languages** |
| canary-flash (180m/1b) | en, de, es, fr | en ↔ de/es/fr |
| Parakeet EN family | English | — |

Full locale tables with quality tiers and prompt ids:
[docs/nemotron-languages.md](docs/nemotron-languages.md).

## CLI

```
mynah-asr transcribe -m <model_dir> -i <file.wav>
    --lang <tag|auto>        source language (it-IT, en, auto for detection)
    --target-lang <xx>       AED/Canary: OUTPUT language ≠ source = translation
    --timestamps             one word per line: t0 t1 word
    --decoder default|ctc    CTC head of hybrid models (faster)
    --lookahead N            Nemotron streaming preset (0|1|3|6|13)
    --segment-sec S          per-segment limit (model-aware default: 30s/300s)
    --quant int8|int4        quantized checkpoint (or quantize at load)
    --backend cpu|metal|cuda GEMM backend (graceful CPU fallback)
    --caps auto|scalar|avx2|vnni   x86 SIMD level (default: cpuid)

mynah-asr stream -m <model_dir> [--lang auto] [--lookahead N] [--quant int8|int4]
    live transcription from stdin (raw s16le 16 kHz mono), text never retracted

mynah-asr quantize -m <model_dir> --quant int8|int4
    writes the pre-quantized checkpoint (⅓ of the RAM, zero-copy load)

mynah-asr --version
```

## Server (REST + WebSocket, OpenAI-compatible)

```sh
./mynah-asr-server -m models/canary-1b-v2 -p 8090 --threads 4 --batch 8

curl -F file=@audio.wav -F language=it http://localhost:8090/v1/audio/transcriptions
curl -F file=@audio_de.wav -F language=de http://localhost:8090/v1/audio/translations
# WebSocket streaming: GET /v1/audio/stream (PCM in, JSON deltas out)
```

`verbose_json` includes per-word timestamps; `--batch N` enables
weight-stationary micro-batching across concurrent requests.
Details: [docs/server.md](docs/server.md).

## Bindings (Python · Node)

Thin FFI wrappers over `libmynah_asr` (`make shared` first) — no build step in the repo:

```python
# Python — pure ctypes, zero dependencies: bindings/python/mynah_asr.py
from mynah_asr import MynahASR
m = MynahASR("models/parakeet-tdt-0.6b-v3")
text, words = m.transcribe("audio.wav", timestamps=True)
MynahASR("models/canary-1b-v2").transcribe("it.wav", lang="it>en")   # translation
```

```js
// Node — koffi FFI (npm i koffi): bindings/node/mynah_asr.js
const { MynahASR } = require("./mynah_asr");
const m = new MynahASR("models/parakeet-tdt-0.6b-v3");
const { text, words } = m.transcribe("audio.wav", { timestamps: true });
```

## C API (libmynah_asr)

```c
#include "mynah_asr.h"

mynah_asr_model *m = mynah_asr_load("models/parakeet-tdt-0.6b-v3");
char lang[16];
mynah_asr_word *words; int n_words;
char *text = mynah_asr_transcribe_ts(m, samples, n_samples, "auto", -1, lang,
                                     &words, &n_words);  /* or mynah_asr_transcribe */
printf("[%s] %s\n", lang, text);
for (int i = 0; i < n_words; i++)
    printf("%.2f-%.2f %s\n", words[i].t0, words[i].t1, words[i].word);
```

Complete buildable example: [`examples/minimal.c`](examples/minimal.c).
Reference: [docs/api.md](docs/api.md) · streaming: [docs/streaming.md](docs/streaming.md).
`make lib` builds `libmynah_asr.a`.

## Layout

```
src/        C runtime (libmynah_asr) — decoders behind a vtable (engine.h)
cli/        `mynah-asr` CLI
server/     `mynah-asr-server` REST + WebSocket
bindings/   Python (ctypes) & Node (koffi) over libmynah_asr
tools/      Python tooling (uv): weight converter, numpy oracle, eval
tests/      per-stage parity vs oracle + e2e (make test; skips without models)
samples/    real CC-BY audio (FLEURS, 11 languages) for the quality suite
docs/       architecture, models, languages, streaming, benchmarks
reference/  configs/tokenizers extracted from checkpoints (for development)
```

## Build & test

```sh
make              # CLI + server (separate build/, version from git)
make lib          # libmynah_asr.a        make shared   # .dylib/.so for bindings
make install      # PREFIX=/usr/local: bin + lib + include
make test         # parity vs oracle + e2e (exit 77 = skip without models)
make test-samples # quality on real audio: ASR CER + translations, cpu+metal
make test-server  # REST + concurrency + WebSocket + translations
make bench        # RTF on the fixtures   make leaks / make ubsan / make asan
make golden-dump  # regenerate reference dumps (requires tools/ + model)
```

## Documentation

| doc | contents |
|---|---|
| [docs/models.md](docs/models.md) | supported + candidate model catalog, verified configs & licenses |
| [docs/benchmarks.md](docs/benchmarks.md) | full RTF/RAM matrix, CPU vs Metal vs int8, methodology |
| [docs/streaming.md](docs/streaming.md) | cache-aware streaming, latency presets, WebSocket protocol |
| [docs/quantization.md](docs/quantization.md) | int8/int4 checkpoint format and SDOT/VNNI kernels |
| [docs/gguf.md](docs/gguf.md) | GGUF weight container (export, supported types, lookup order) |
| [docs/backends.md](docs/backends.md) | CPU SIMD dispatch, Metal, CUDA |
| [docs/server.md](docs/server.md) | REST + WebSocket server, OpenAI compatibility |
| [docs/api.md](docs/api.md) | C API reference (libmynah_asr) |
| [docs/nemotron-languages.md](docs/nemotron-languages.md) | the 40 Nemotron locales with quality tiers |
| [docs/nemotron-arch.md](docs/nemotron-arch.md) · [parakeet-tdt-arch.md](docs/parakeet-tdt-arch.md) · [canary-arch.md](docs/canary-arch.md) | verified model architectures |
| [docs/parakeet-en-family.md](docs/parakeet-en-family.md) · [canary-usage.md](docs/canary-usage.md) | per-family usage, features and limits |
| [docs/architecture-notes.md](docs/architecture-notes.md) | design decisions & implementation traps |
| [docs/prior-art.md](docs/prior-art.md) | the landscape: parakeet.cpp, sherpa-onnx, onnx-asr… |

## License

MIT. Model weights keep their respective licenses (Nemotron 3.5: OpenMDW-1.1).
