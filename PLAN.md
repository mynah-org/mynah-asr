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
- [ ] S0-6 **The box day**: the ordered runbook for the Linux host — dispatch proof, identity gates, the two questions macOS could not ask, the qualification, multi-model → [`.work/box-day-plan.md`](.work/box-day-plan.md)
- [~] S0-1 Build, gates and single-stream step cost (Q=4, int8, T=1..4, pinned, idle box) on the 32-core ARM host → [`.work/axion-first-run.md`](.work/axion-first-run.md)
- [~] S0-2 Streaming load tool v0: N WebSocket clients paced at 1x, stdlib only → [`.work/bench-harness-streaming.md`](.work/bench-harness-streaming.md)
- [-] S0-3 The v1 server under N streams — dropped: it is broken by construction and the number would not size v2 → [`.work/baseline-streaming-concurrency.md`](.work/baseline-streaming-concurrency.md)
- [ ] S0-4 Allocation count per chunk on Linux, measured not assumed (needs no box: any Linux) → [`.work/baseline-streaming-concurrency.md`](.work/baseline-streaming-concurrency.md)
- [~] S0-5 **Linux functional gates on every server change**: PR #2 (`serving-v2`) runs the GitHub CI on ubuntu x86, ubuntu ARM and macOS, green 2026-09-18 including the prefork phases, ASan+UBSan and clang-tidy; still owed: masks read back on the ARM box → [`.work/axion-first-run.md`](.work/axion-first-run.md)

### S1 — Library seams the server needs → [`.work/stream-api-v2.md`](.work/stream-api-v2.md)
- [x] S1-1 `mynah_asr_stream_reset` + `need_samples`; slots are pooled, not reopened (gate green 2026-09-18)
- [x] S1-2 Deltas carry a real window: `t0` is the previous delta's `t1`, so the deltas PARTITION the stream and a client places text in time without accumulating anything of its own (it was always 0.0, which made `t1` the only usable field); `is_final`/`is_eou` were already there, and an `eou` is now a point (`t0 == t1`) rather than a window starting at 0. The WebSocket `delta` frame carries `t0`/`t1` beside `audio_s`. The chunk-arrival passthrough is NOT a library API: the server stages the chunk, so it owns the arrival — `emit_ctx.arrival` comes from the slot's own arrival record and the callback runs inside that step (`server/sched.c`). A second copy in the library would be a number the library cannot check → [`.work/stream-api-v2.md`](.work/stream-api-v2.md)
- [x] S1-3 Allocation-free chunk: per-stream scratch, incremental detokenisation — 87 -> 0 allocations per chunk on macOS/Accelerate, gated by `make test-stream-allocs`; the Linux count stays S0-4
- [x] S1-4 Batched stream step for B slots, byte-identical to B single steps: `mynah_asr_stream_step_batch` stacks the ready chunks as one `[Σq, d]` pass; int8 exact by construction (native per-row dot extended to any T, the dequant fallback never reached), f32 measured row-stable on Accelerate and OFF on OpenBLAS until its own gate runs; `tests/test_stream_batch.c` (0 of 266,240 floats differ at B=8, both dtypes) + `make test-stream-batch-allocs` (0 allocations/step); M1 dev signal 693.60 -> 227.00 ms per step at B=8
- [x] S1-5 The pool spins before it parks, with the meter: workers watch an atomic generation for `MYNAH_ASR_POOL_SPIN_US` (default 50, `0` = the pure condvar pool and the A/B arm) before sleeping, so the dozens of small GEMMs a stream step issues back to back stop paying a kernel round trip each. M1 dev signal, the sub-2M-MAC shapes: T=4 318 -> 219 us, T=8 355 -> 285 us; the large shapes do not move. `worker_spin_pct`/`caller_spin_pct` in the SIGUSR1 dump say whether the spin caught anything. **The A/B arm earned its keep immediately**: at `SPIN_US=0` the first dispatch DEADLOCKED (workers parked on the job condvar, caller on the completion condvar, 0% cpu) because a worker started from the live generation and could park on a job it was already counted in — invisible at the default spin, which catches the bump inside its window. `tests/test_threads.c` now gates the first dispatch at three spin budgets in pristine children, and the prefork `threadpool_after_fork` contract in a child of a warm pool. The lane redirect is not lifted: this runtime has no second lane to redirect → [`.work/threadpool-and-lane.md`](.work/threadpool-and-lane.md)
- [x] S1-6 BLAS leaves the worker: `src/sgemm.c` is ours, deterministic across thread counts and gated (`tests/test_sgemm.c`), behind one seam, and **`BLAS=none` is the Linux default since 2026-09-19**. What unblocked the decision was finding why `own` had lost: the DOT family — every `x @ W^T`, so every f32 linear, the joint head and the attention scores — ran one dot product at a time into ONE accumulator, i.e. FMA-latency bound at 8.7 GF/s on a core that does ~100. Register-tiled (`SG_DOT_MR x SG_DOT_NC_TILE`, a `1 x SG_DOT_NC_WIDE` strip for a gemv) it does 75-80, the panel shapes 48.9 -> 74.8 (`SG_PANEL_NV` derived instead of chosen), the LSTM gemv 8.2 -> 30.2; at 4-16 stacked rows — the streaming shapes — `own` now BEATS Accelerate's AMX. The tile is a schedule, not a formula: byte-identical to the untiled dot, gated for DOT/PANEL/NARROW by `tiling_identity` in `tests/test_sgemm.c`. The same instrument found a second defect: the parallel threshold was compared against total work rather than work PER THREAD, so the sub-2M-MAC shapes — where the streaming attention lives — cost 630 us at 8 threads against 318 at one; now 330. OpenBLAS keeps building as the comparison arm → [`.work/threadpool-and-lane.md`](.work/threadpool-and-lane.md)
- [ ] S1-6a Linux only, now a VERIFICATION rather than an open question, and two commands rather than a box-day: `MYNAH_ASR_GEMM_PROFILE=1` dumps the shapes and call counts the model really issued, `tests/bench_gemm_shapes.c` replays them against both arms interleaved in ONE process and weights each shape by its count. Run it on the Axion and on x86 with `make BLAS=openblas`; if OpenBLAS wins by enough to cost capacity the Makefile default goes back and the note says why. Also owed there: thread count inside a pinned prefork worker (2T vs T) and the OpenBLAS f32 row-stability gate that decides `MYNAH_ASR_BATCH_F32` → [`.work/threadpool-and-lane.md`](.work/threadpool-and-lane.md)
- [x] S1-7 Share the rel-pos K-projection across the streams of one batched pass that have the same `K`: computed once per (layer, K-group) into the batch scratch, bit-exact by construction and gated as such (0 of 266,240 floats differ at B=8, int8 and f32, with SHARED/PRIVATE/GROUP counters proving it happened); the single path is untouched and one allocation-free step is unchanged. M1 dev signal UNDER THIRD-PARTY LOAD, best-of-4 interleaved: slope `b` 22.7 -> 20.3 ms (-10.7%), B_max ~8.2 -> ~9.2 — a third of the 7.4 ms the isolated micro-benchmark had promised, discrepancy not explained. Only the largest K-group of a pass is shared; Linux unmeasured → [`.work/stream-api-v2.md`](.work/stream-api-v2.md)
- [ ] S1-8 Run the S1-4 gates on Linux: OpenBLAS f32 row stability (`MYNAH_ASR_BATCH_F32=1` with the stream-batch test, which decides the default there), the int8 identity, and the step table on Axion and on x86 → [`.work/stream-api-v2.md`](.work/stream-api-v2.md)

### S2 — The server
- [~] S2-1 Prefork parent, core-major pinned workers, SCM_RIGHTS handoff, `--prefork-plan` (landed; Linux pinned run pending) → [`.work/server-prefork.md`](.work/server-prefork.md)
- [~] S2-2 One scheduler thread per worker owns the model; slots; ingest threads; one path for WS and REST (landed and gated on the M1 dev host: identity at 4 and at 8 streams and under `--prefork`, ubsan clean, the step is one batched call; TSan and the Linux cadence numbers are what is left) → [`.work/server-scheduler.md`](.work/server-scheduler.md)
- [x] S2-2b The scheduler's step is ONE `mynah_asr_stream_step_batch` over the ready set, not one feed per slot: cancellation, fairness, per-slot lag and the finalize path unchanged, scratch reserved at start-up, `batched_steps_total`/`rows_stacked_total`/ready-set/`step_wall_ms` in `/v1/health` and `/metrics`. Quiet-host M1 pair, same gate back to back: 4 streams lag p50/p95 **1219/2193 -> 145/253 ms** (NOT STREAMABLE -> GOOD), WAVE 8x2 **1104/1952 -> 143/258 ms**; new `batched-identity` phase (8 streams, 5 clips, `--cap 8`) byte-identical and proving the stacked path ran → [`.work/server-scheduler.md`](.work/server-scheduler.md)
- [~] S2-3 Admission ladder with per-reason statuses, timeouts, pings, lingering close, graceful shutdown (landed and gated on the M1: every worker refusal goes through the lingering close, `--ping-ms`/`--max-audio-seconds`/idle caps announced with a code, SIGTERM drains with `shutting_down`, exit 0, `leaks --atExit` 0 leaks; the "lag does not move under a cap+1 refusal" line is served by the stalled-reader phase of the stream gate, not measured directly; rung 4 still not enforced at a step boundary; no Linux run) → [`.work/server-admission.md`](.work/server-admission.md)
- [x] S2-4 Asynchronous bounded output writer; backpressure is cancel (module landed with a model-free self-test, TSan and leaks clean; the scheduler writes every frame through it) → [`.work/server-stream-out.md`](.work/server-stream-out.md)
- [~] S2-5 WebSocket protocol v2: control messages, `seq`/`lag_ms`, refusals before the upgrade (query validated before the 101, `unknown_control` and `language_not_served` survive the session, `format=f32le`, `rate=` refused rather than resampled per frame, three utterances on one socket byte-identical to the CLI in `tests/test_server_protocol.sh`; the v1 aliases on `delta`/`done` and `t0`/`t1` are the last step) → [`.work/ws-protocol-v2.md`](.work/ws-protocol-v2.md)
- [~] S2-6 Several models in one fleet: `--model name=dir[:workers=][:cpus=][:cap=][:quant=][:lookahead=]` + `--default`, router by model (WS `?model=`, REST query or multipart field, inside the existing bounded `MSG_PEEK`), `model_not_found` 404 with the accepted set, per-group admission, `/v1/models` from the actual table, `group`/`groups` in `/v1/health`, a `model` label on the router's per-worker series (landed and gated on the M1: `tests/test_server_models.sh` runs ONE server with two groups under `--prefork 2` and proves each group's REST answer byte-identical to THAT model's CLI, the WS identity on the streaming group, 400 `model_not_streaming` on the offline one, 404 on an unknown name, capacity refused per group while the other still serves, survivors=0. NOT done: no Linux/pinned run, so `:cpus=` is a printed plan and not measured isolation; no Canary/AED group and no `--lid-model` in a multi-group fleet; no multi-group SOAK) → [`.work/multi-model-serving.md`](.work/multi-model-serving.md)
- [x] S2-7 Byte-identity gate: N concurrent streams == the same streams alone, also under `--prefork` (`tests/test_server_stream.sh`: 4 real-time streams x 2 utterances and `--prefork 2 --cap 2`, each text byte-identical to `mynah-asr transcribe` of the same clip; `tests/test_server_protocol.sh` adds three utterances on ONE socket) → [`.work/server-scheduler.md`](.work/server-scheduler.md)

### S3 — The server proves what it runs → [`.work/observability.md`](.work/observability.md)
- [x] S3-1 Flag registry + `[FLAGS]`/`[EFFECTIVE-CONFIG]` in library, CLI and server, `[SERVER-CONFIG] v=1` and a per-worker `[TOPOLOGY] v=1` with the mask read back from the kernel
- [x] S3-2 `--dispatch-map [--json]` resolved from owner predicates, `IDLE HARDWARE` footer, ISA guard first in both `main()`s; `mynah-asr-server --dispatch-map` prints the same table
- [x] S3-3 `/v1/health` as facts (cancels by reason, lag p50/p95, resolved kernels, actual mask); `/metrics` on its own port, token bucket, per-worker series, exact threshold counters, the router answering for the fleet
- [x] S3-5 `MYNAH_ASR_GEMM_PROFILE=1`: every call through the f32 seam recorded per distinct shape and dumped at exit (`[GEMM-PROFILE]`/`[GEMM-SHAPE]`), so "which shapes does this model issue, and how often" is a fact per family, per lookahead, per quantisation instead of a reading of the source → [`tests/bench_gemm_shapes.c`](tests/bench_gemm_shapes.c)
- [x] S3-4 `SIGUSR1` one-shot `[DUMP]` bracketed per worker, forwarded by the parent, handler installed before the fork; thread names verified and documented

### S6 — The fleet answers for itself → [`.work/fleet-observability.md`](.work/fleet-observability.md)
- [ ] S6-1 Fleet metrics through a shared slot table: one `mmap(MAP_SHARED)` before the fork, one fixed slot of atomics per worker, single writer, relaxed adds; the router renders every worker's SCHEDULER counters with a `worker` label beside the routing ones it already has. One scrape target, per-worker series preserved, nothing summed, no IPC in the step (the qwen-tts `qwen_metrics_slot_t` mechanism). Today the router exports routing facts only and the real counters are reachable only by giving each worker its own `--metrics-port`
- [ ] S6-2 Fleet `/v1/health` on the router, and `worker`/`counters_scope` disclosure on the worker's own: a probe currently lands on whichever worker the router picked, so it is a per-worker health wearing the server's name. Readiness and liveness stay ONE endpoint — only a server that HAS a scheduler can be unavailable for want of one
- [ ] S6-3 `configs/observability/`: `prometheus.yml`, `grafana-dashboard.json`, `alerts.yml`, README, and an `observability_up.sh` that provisions both on loopback. Panels translate one-for-one from the sibling once the vocabulary is ASR's; every alert fires on something a LISTENER would notice and every threshold is an exact count, never an estimated quantile
- [ ] S6-4 **Backlog**, the one counter this server is missing: audio received and not yet encoded, as exact threshold counters (`backlog_over_chunks_total`), not a gauge. It is the ASR mirror of the sibling's behind-realtime gap and the one server-side number that says a worker is falling behind before a client notices
- [~] S6-5 Quality, as opposed to speed: **landed** — every utterance gets a CER against the bank's human reference (`--transcripts`), the normaliser is defined once in the metrics module because a CER computed with a different one is a different number wearing the same name, and a run whose cadence held while the transcripts got worse is now **DEGRADED** rather than GOOD. TTFB (first frame of any kind) is reported beside TTFP (first word): together they say whether a wait is transport or model. Known-answer tests pin the normaliser, the edit distance, the rate and the verdict. STILL OPEN: a CER baseline (the `cer_p95` threshold is a placeholder guarding against collapse, not a quality target), and the CER drift track across commits. The load harness must REFUSE to print a KPI table when the batched-identity comparison fails (the sibling's `serve_identity_gate.py` rule — a number from a run whose transcripts moved is worse than no number); a CER drift track against the committed `samples/` bank; and named drift thresholds in the soak the way the sibling names its 30%/20%/resource-growth gates
- [ ] S6-6 A metrics-page watcher under `tools/` (not written yet): scrape the page under load and assert monotonic (no `_total` went backwards), conserved (`completed <= assigned`), balanced (every live worker received something). Model-free, and the only thing that exercises the metrics port under load

### S7 — C100 preconditions (code, before the box) → [`.work/fleet-observability.md`](.work/fleet-observability.md)
- [x] S7-1 The listen backlog is derived from the capacity the ladder already declares (`W x (slots + queue)`, doubled because the connections that will be REFUSED arrive in the same burst as the ones that will be served, floored at `SOMAXCONN`, capped at 4096, overridable with `MYNAH_ASR_LISTEN_BACKLOG`) and printed on `[SERVER-CONFIG]` beside the kernel `somaxconn`, with a WARNING when the kernel clamps it. The defect was reproduced before the fix: `listen(fd,64)` + 150 simultaneous connects completed 64; with the derived backlog it completes as many as `somaxconn` allows
- [x] S7-2 `RLIMIT_NOFILE` is raised toward the hard limit at start-up with a halving backoff and REPORTED AS IT BECAME, not as it was asked for; a loud WARNING names soft, hard and the need when the ceiling is below the configuration, because a stream refused by `accept()` with EMFILE is a refusal no rung of the ladder issued and no counter explains
- [ ] S7-3 The load generator is one PROCESS per stream, so at C100 it competes with the server for cores. The sibling MEASURED this on the same Axion: an unpinned generator cost worker 0 ~8% of first-audio latency. Either a second host generates the load, or the generator is pinned to a reserved slice excluded from the server's mask — and the harness records which, because a C100 number taken with the generator on the server's cores is not a C100 number

### S4 — Measure, then qualify → [`.work/bench-harness-streaming.md`](.work/bench-harness-streaming.md)
- [x] S4-1 Streaming metrics module: every metric defined once, known-answer self-test in `make test` ([`tools/bench/streaming_metrics.py`](tools/bench/streaming_metrics.py))
- [x] S4-2 WAVE / SOAK modes, mixed-length bank, manifest, drift gate, refusal on pacing failure ([`tools/bench/stream_load.py`](tools/bench/stream_load.py); exercised on the macOS dev host only — the qualifying run is S4-3)
- [ ] S4-3 First SOAK on the ARM box at the derived operating point; verdict by the envelope → [`.work/axion-first-run.md`](.work/axion-first-run.md)
- [x] S4-4 Serving operations doc: how a new box is approached, the vocabulary, the envelope → [`docs/serving.md`](docs/serving.md)

### S5 — Kernels for the batched step, ARM and x86 in one step → [`.work/cpu-kernels-arm-x86.md`](.work/cpu-kernels-arm-x86.md)
- [x] S5-1 Weight-stationary int8 GEMM for T <= 64, bit-exact vs single rows: SMMLA (2x2 tile), SDOT x4, EVEX and VEX VPDPBUSD, AVX2 — all behind target attributes in every build, chosen by a runtime probe, opt-out through `MYNAH_ASR_CAPS`. `--dispatch-map` splits `kernel.int8_dot` (one row) from `kernel.int8_rows` (the batched pass). It caught a real defect on the way (the `-ffast-math` epilogue grouping `s*ws*sx` differently at two inline sites, now pinned by a value barrier). **CLOSED 2026-09-19 on the Axion (Neoverse-V2, i8mm present)**: `simd=neon+dotprod+i8mm`, `kernel.int8_rows` resolves to `neon-smmla`, and `tests/test_qmat` reports `neon-smmla exact OK, 82 shape/T cases, 466 kernel blocks` — the kernel EXECUTES and is bit-exact, which is what `[~]` was waiting for. The x86 arms are still cross-compiled only and skip with a reason → [`.work/cpu-kernels-arm-x86.md`](.work/cpu-kernels-arm-x86.md)
- [ ] S5-2 Depthwise conv with transposed weights and SIMD FMA, gated by the profile
- [ ] S5-3 KleidiAI as an A/B arm behind `KLEIDI=1`
- [ ] S5-4 `SIMD=` profiles with the double test, build-flag stamp, link-only CI per ISA
- [ ] S5-5 Pool-meter run on the batched step; choose W×T from the break-even rule

### Hygiene → [`.work/repo-hygiene.md`](.work/repo-hygiene.md)
- [x] H-1 the per-language samples manifest under the test audio is generated by `make fetch-lang-samples`: say so or track it
- [x] H-2 Stale test-file reference in `tools/gen_kquant_ref.py`
- [x] H-3 `make check` (plan + repo + flag registry) green locally and a CI step (runs on the next push)

### Multi-model — the fleet beyond Nemotron → [`.work/multi-model-streaming.md`](.work/multi-model-streaming.md)
- [x] M-1 A model that declares streaming presets but uses something the incremental encoder does not implement (linear biases, a folded batch_norm, xscaling, symmetric conv padding, per-feature mel normalisation, a subsampling factor other than 8) is REFUSED with a named reason instead of emitting plausible wrong text: `mynah_asr_stream_unsupported` asks the loaded weights, the server asks it before the WebSocket upgrade (`400 model_not_streaming`, the reason in the body). No shipped pack trips it — that is the point
- [x] M-2 The model constants that were not: the subsampling factor is now DERIVED from the convolutions `src/subsampling.c` applies (`MYNAH_ASR_SS_STAGES`/`MYNAH_ASR_SS_FACTOR`) and a pack whose `encoder.subsampling_factor` disagrees is REFUSED at load rather than silently mis-timestamping every word; the CTC blank comes from `decoder.blank_id` (a pure-CTC pack declares it, a hybrid's auxiliary head keeps the last index because the pack's `blank_id` belongs to its RNNT/TDT decoder); the VAD reads its own `sample_rate`; and the server asks `mynah_asr_sample_rate()` instead of dividing by 16000 in six places — the `rate=` refusal now names the model's rate. No behaviour changes on any shipped pack (the converter writes `blank_id = vocab_size - 1`, every pack is 16 kHz and subsamples by 8): the point is that each of these was true by coincidence
- [ ] M-3 Converter builder for `nemotron-speech-streaming-en-0.6b` (cache-aware, left context 70): the only cache-aware model not served, and for a converter reason rather than an architectural one
- [ ] M-4 Buffered streaming for Parakeet behind an explicit stream mode, in its own group — **costs ~7x real time per stream** against ~1.05x for cache-aware Nemotron (derivation in the note); the cap comes from a measured step, and `docs/serving.md` states the multiplier before a line is written
- [-] M-5 Canary streaming: out of v2 by design — its AED decoder keeps no state between calls, so it re-decodes the window on top of the 7x encoder

### Later, not now
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
