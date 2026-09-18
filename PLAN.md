# Mynah ASR — plan

`ENGINEERING.md` is normative. This file is the **board**: one line per work
item, linking the `.work/` note that holds the detail. Read the note when you
pick the item up; never write detail here. Legend: `[ ]` open · `[~]` in
progress · `[x]` done · `[-]` dropped. `python3 tools/check_plan.py` must pass.

## Mission

A native C runtime for NeMo speech models (Nemotron streaming, Parakeet,
Canary) that is **usable as one concurrent server on a CPU box**: tens of
real-time streams and offline requests per host, several models in one fleet,
no latency spikes, no stalls, every claim provable from the process itself.
**Production is Linux x86-64 and ARM64 and every efficiency decision is taken
there; macOS is the development machine, nice to have, never the target.**

## Current trusted state (2026-09-18, HEAD after `5f0f802`)

- Runtime 0.9.1: six model families, streaming ≡ offline byte-identical,
  oracle parity per stage, int8/int4 (NEON SDOT, AVX-512 VNNI), Metal/CUDA
  opt-in, VAD, LID, word timestamps, GGUF import. Single-stream RTF 0.02–0.15
  on Apple Silicon and EPYC (`docs/benchmarks.md`).
- The server is a **prototype**: blocking pool, one thread per stream, ceiling
  = `--threads`, no timeouts, no metrics, one model per process. **No number
  exists for N concurrent streams on any host.** Audit and design:
  [`.work/serving-v2-design.md`](.work/serving-v2-design.md).
- Method adopted from qwen-tts / mynah-tts on 2026-09-18:
  [`.work/engineering-method.md`](.work/engineering-method.md).

## Work board

### Reference — read before starting anything
- [x] The v2 serving design, what it borrows, what is ASR-specific, and the
      ideas the siblings already falsified → [`.work/serving-v2-design.md`](.work/serving-v2-design.md)
- [x] The method → [`.work/engineering-method.md`](.work/engineering-method.md)
- [x] What the siblings measured as wins and losses, their flag machinery, kernel state per ISA → [`.work/sibling-wins-and-flags.md`](.work/sibling-wins-and-flags.md)
- [x] v1 history (Italian, closed, local only): the old plan and TODO live in the untracked private archive folder under the work notes

### S0 — Baseline (trimmed 2026-09-18: the v1 server is not worth measuring; S1/S2 start now)
- [~] S0-1 Build, gates and single-stream step cost (Q=4, int8, T=1..4, pinned, idle box) on the 32-core ARM host → [`.work/axion-first-run.md`](.work/axion-first-run.md)
- [~] S0-2 Streaming load tool v0: N WebSocket clients paced at 1x, stdlib only → [`.work/bench-harness-streaming.md`](.work/bench-harness-streaming.md)
- [-] S0-3 The v1 server under N streams — dropped: it is broken by construction and the number would not size v2 → [`.work/baseline-streaming-concurrency.md`](.work/baseline-streaming-concurrency.md)
- [ ] S0-4 Allocation count per chunk on Linux, measured not assumed (needs no box: any Linux) → [`.work/baseline-streaming-concurrency.md`](.work/baseline-streaming-concurrency.md)
- [ ] S0-5 **Linux functional gates on every server change** (glibc, OpenBLAS, `sched_setaffinity`, `POLLRDHUP`, `LD_PRELOAD` count): the GitHub CI on a pushed `serving-v2` branch, or the ARM box — both need the owner's go; until then every macOS-only result is labelled as such → [`.work/axion-first-run.md`](.work/axion-first-run.md)

### S1 — Library seams the server needs → [`.work/stream-api-v2.md`](.work/stream-api-v2.md)
- [x] S1-1 `mynah_asr_stream_reset` + `need_samples`; slots are pooled, not reopened (gate green 2026-09-18)
- [ ] S1-2 Deltas carry `t0`, `is_final`, `is_eou` and a chunk-arrival passthrough for `lag_ms`
- [x] S1-3 Allocation-free chunk: per-stream scratch, incremental detokenisation — 87 -> 0 allocations per chunk on macOS/Accelerate, gated by `make test-stream-allocs`; the Linux count stays S0-4
- [ ] S1-4 Batched stream step for B slots, byte-identical to B single steps
- [ ] S1-5 Lift the mynah-tts pool: spin-then-park, meter, after_fork, lane redirect → [`.work/threadpool-and-lane.md`](.work/threadpool-and-lane.md)
- [ ] S1-6 BLAS leaves the Linux worker: own sgemm, `BLAS=none` default, `openblas` as the comparison build → [`.work/threadpool-and-lane.md`](.work/threadpool-and-lane.md)
- [ ] S1-6a Interim, Linux only: OpenBLAS thread count inside a prefork worker (1 vs T; today pool T + OpenBLAS T = 2T threads on a T-cpu slice) measured on the box before BLAS leaves; int8 is the server default on Linux → [`.work/threadpool-and-lane.md`](.work/threadpool-and-lane.md)

### S2 — The server
- [~] S2-1 Prefork parent, core-major pinned workers, SCM_RIGHTS handoff, `--prefork-plan` (landed; Linux pinned run pending) → [`.work/server-prefork.md`](.work/server-prefork.md)
- [~] S2-2 One scheduler thread per worker owns the model; slots; ingest threads; one path for WS and REST (landed and gated on the M1 dev host: identity at 4 streams and under `--prefork`, ubsan clean; TSan and the Linux cadence numbers pending) → [`.work/server-scheduler.md`](.work/server-scheduler.md)
- [ ] S2-3 Admission ladder with per-reason statuses, timeouts, pings, lingering close, graceful shutdown → [`.work/server-admission.md`](.work/server-admission.md)
- [x] S2-4 Asynchronous bounded output writer; backpressure is cancel (module landed with a model-free self-test, TSan and leaks clean; the scheduler writes every frame through it) → [`.work/server-stream-out.md`](.work/server-stream-out.md)
- [ ] S2-5 WebSocket protocol v2: control messages, `seq`/`lag_ms`, refusals before the upgrade → [`.work/ws-protocol-v2.md`](.work/ws-protocol-v2.md)
- [ ] S2-6 Several models in one fleet: `--model name=dir:workers=:cpus=:cap=`, router by model, `/v1/models` from the table → [`.work/multi-model-serving.md`](.work/multi-model-serving.md)
- [ ] S2-7 Byte-identity gate: N concurrent streams == the same streams alone, also under `--prefork` → [`.work/server-scheduler.md`](.work/server-scheduler.md)

### S3 — The server proves what it runs → [`.work/observability.md`](.work/observability.md)
- [~] S3-1 Flag registry + `[FLAGS]`/`[EFFECTIVE-CONFIG]` lines in the library and the CLI (`--flags`, `MYNAH_ASR_VERBOSE=1`); the server banner and `[TOPOLOGY]` are still open
- [~] S3-2 `--dispatch-map [--json]` resolved from owner predicates, `IDLE HARDWARE` footer and ISA guard in `main()`; the server does not print it yet
- [ ] S3-3 `/v1/health` as facts; `/metrics` on its own port, per-worker series, threshold counters
- [ ] S3-4 `SIGUSR1` dump forwarded to workers; named threads

### S4 — Measure, then qualify → [`.work/bench-harness-streaming.md`](.work/bench-harness-streaming.md)
- [x] S4-1 Streaming metrics module: every metric defined once, known-answer self-test in `make test` ([`tools/bench/streaming_metrics.py`](tools/bench/streaming_metrics.py))
- [x] S4-2 WAVE / SOAK modes, mixed-length bank, manifest, drift gate, refusal on pacing failure ([`tools/bench/stream_load.py`](tools/bench/stream_load.py); exercised on the macOS dev host only — the qualifying run is S4-3)
- [ ] S4-3 First SOAK on the ARM box at the derived operating point; verdict by the envelope → [`.work/axion-first-run.md`](.work/axion-first-run.md)
- [x] S4-4 Serving operations doc: how a new box is approached, the vocabulary, the envelope → [`docs/serving.md`](docs/serving.md)

### S5 — Kernels for the batched step, ARM and x86 in one step → [`.work/cpu-kernels-arm-x86.md`](.work/cpu-kernels-arm-x86.md)
- [ ] S5-1 Weight-stationary int8 GEMM for T ≤ 64: SMMLA (i8mm) and VNNI, bit-exact vs single rows; no per-call dequant
- [ ] S5-2 Depthwise conv with transposed weights and SIMD FMA, gated by the profile
- [ ] S5-3 KleidiAI as an A/B arm behind `KLEIDI=1`
- [ ] S5-4 `SIMD=` profiles with the double test, build-flag stamp, link-only CI per ISA
- [ ] S5-5 Pool-meter run on the batched step; choose W×T from the break-even rule

### Hygiene → [`.work/repo-hygiene.md`](.work/repo-hygiene.md)
- [x] H-1 the per-language samples manifest under the test audio is generated by `make fetch-lang-samples`: say so or track it
- [x] H-2 Stale test-file reference in `tools/gen_kquant_ref.py`
- [~] H-3 `make check` (plan + repo + flag registry) green locally; still to add to CI

### Later, not now
- [ ] L-1 Cache-aware streaming for Parakeet / Canary (WS on them is a 400 in v2.0)
- [ ] L-2 Unstable-tail partials for Nemotron (a decoder change; protocol already has `final`)
- [ ] L-3 CUDA v2b resident encoder (from the v1 TODO; GPU serving is out of v2 scope)

## Durable contract (changes only when a decision changes)

1. Config-driven: every model number comes from the converted pack's
   `mynah.json`; no model constants in `#define`s.
2. One code path for offline and streaming, and for REST and WebSocket in the
   server: a file is a slot fed faster than real time.
3. Every numeric stage is validated against the Python oracle
   (`tools/oracle/`) before anything is built on it; per-stage tolerances,
   transcripts byte-identical.
4. A stream's transcript never depends on who it was batched with: every
   batched path ships with a bit-exactness gate against the single-item path.
5. CPU is the product. Linux x86-64 and ARM64 are production; macOS numbers are
   development signals. A backend claim names its gates (rule 4 of
   `ENGINEERING.md`).
6. Load never enters the chunk size. When the machine is full, refuse.
7. Python is offline tooling only (`uv`); never required at runtime.
8. Everything in the repo is English. Commit messages: imperative, what and why.
