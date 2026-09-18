# Mynah ASR — repo contracts

`ENGINEERING.md` is normative: read it first. `PLAN.md` is the board (one line
per item); the detail of an item is its `.work/<item>.md` note, read only when
working on it. When docs and code diverge, trust the code and the converted
model's `mynah.json`.

## What this project is

Pure-C ASR runtime (llama.cpp/whisper.cpp style) for NVIDIA NeMo FastConformer
models: Nemotron 3.5 streaming 0.6B (cache-aware, 40 languages), the Parakeet
family, Canary. Current work is **serving v2**: one `mynah-asr-server` that
holds several models in one fleet and serves tens of concurrent real-time
streams on a Linux CPU box with no latency spikes and no stalls, built on the
design qwen-tts and mynah-tts already qualified
(`.work/serving-v2-design.md`). Production is Linux x86-64 and ARM64; macOS is
the development machine and its numbers are development signals.

## Map

- `src/` — `libmynah_asr` (C11). One module = one small cohesive `.c`+`.h`.
  No monolithic files, no placeholder files with names that lie.
- `cli/main.c` — `mynah-asr`. `server/` — `mynah-asr-server`.
- `tools/` — Python managed with **uv**; offline only (converter, oracle,
  eval, bench harness). Never required at runtime.
- `reference/<model>/` — configs and headers extracted from checkpoints.
- `models/` — downloaded weights, gitignored. `samples/` — committed FLEURS
  clips (CC-BY 4.0) used by the CER gate and the load bank.
- `docs/nemotron-arch.md` is THE architecture reference for the v1 target;
  `docs/models.md` the model catalogue; `docs/benchmarks.md` durable numbers.
- `.work/` — one note per board item; see `.work/README.md`.
- `third_party/ingot` — GGUF/safetensors reader (git subtree); `vendor/cJSON`.

## Rules

1. **Config-driven**: no model constants in `#define`s; everything comes from
   `mynah.json`.
2. **One code path** for offline and streaming, and in the server for REST
   and WebSocket: a file is a slot fed faster than real time.
3. Every numeric stage is **validated against the Python oracle** before
   anything is built on it; per-stage tolerances; transcripts byte-identical.
4. **A transcript never depends on batching, threads or ISA.** Every batched
   or threaded path ships with a bit-exactness gate against the single path.
5. **Load never enters the chunk size. When the machine is full, refuse.**
6. No build artefacts, models, WAVs or `.o` in the repo.
7. Small, frequent commits. **Everything written into the repo or published
   from it is English, always**: commits, comments, docs, CI, release notes.
   Chat with the user can be Italian; the repo cannot.
8. A/B tests use identical explicit parameters (model, quant, language,
   lookahead, threads, masks, seed) and prove dispatch from the binary.

## Build & test

- `make` (CLI + server) · `make lib` · `make test` (per-stage parity + e2e;
  exit 77 = skipped without model; `make golden-dump` regenerates) ·
  `make test-server`, `make test-server-concurrency` · `make bench`.
- `make check` — plan and repo integrity checkers.
- Memory/UB on macOS: `make leaks` + `make ubsan`. **Never ASan on macOS**
  (hangs with the large model); `make asan` is for the Linux CI.

## Known implementation traps

Full checklist in `docs/architecture-notes.md` §6: fc_factor 0.5 on the
macaron FFNs; rel_shift in rel-pos attention; blank = last index (13087 for
Nemotron); blank_as_pad; causal subsampling with asymmetric padding; conv norm
= layer_norm (streaming); mel normalize "NA" for Nemotron; dither 0. Build
traps: `-ffast-math` is load-bearing (never `±INFINITY`, sigmoid
hand-stabilised, finite sentinels); Apple's make 3.81 has one-second
timestamps, so a gate gets `make clean` first.
