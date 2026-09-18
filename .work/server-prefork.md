# S2 — prefork parent, pinned workers, fd handoff

Status: IN PROGRESS (landed 2026-09-18; Linux pinned run and WS-through-router pending an idle box)

Task: S2-1
Question: lift `server/prefork.{c,h}` from mynah-tts (zero model types, takes a
listen fd) and make it the process model of `mynah-asr-server` on Linux.

Known facts (from the sibling code and its measurements)
- Parent: owns the listener, `active[w]`, the CPU plan; never runs a model;
  the only bytes it writes to a client are a refusal.
- One `AF_UNIX` socketpair per worker; accepted fd crosses by `SCM_RIGHTS`;
  exactly one byte comes back per finished connection. No IPC protocol.
- Fork **after** the model is mmapped (COW-shared; Axion worker PSS 124 MB on
  a 567 MB RSS) and **before** any thread exists; from the main thread only.
- Order in the child, not optional: `close(listen_fd)` → `sched_setaffinity`
  → `threadpool_after_fork` → lane prepare → first dispatch builds the pool.
- Core-major ordering from sysfs (`physical_package_id`, `core_id`); print the
  mask actually set. Plan from `sched_getaffinity`, read cgroup `cpu.max`.
- `--prefork-plan` prints topology and the sweep procedure for choosing W and
  T; narrow workers beat wide ones monotonically on both sibling repos
  (16×2 STREAM_RTF p95 0.723 vs 1×32 1.616 at C48 on the Axion).
- Non-Linux: runs unpinned and says so in the banner; single-process mode
  keeps working for development on macOS.

Unknowns
- W×T for a 0.6B ASR encoder at Q=4 rows per stream: the sibling small-model
  answer was 16×2; the stream step here is bandwidth-bound per pass with
  batching amortising it, so the answer comes from S0/S4 and the pool meter,
  not from the sibling.

Plan
1. Copy `prefork.{c,h}` and `http_util.{c,h}` additions (thread naming);
   rename prefix; keep the refusal enum and per-reason counters.
2. `mynah-asr-server --prefork W [--prefork-threads T] [--cap C]`; the
   worker runs today's single-process server unchanged in the first commit,
   so the gate is "the same responses, byte-identical, through the parent".
3. `[TOPOLOGY] v=1 worker= pid= configured_mask= actual_mask= threads=` line.
4. Extend `tests/test_server_concurrency.sh` to run once more under
   `--prefork 2`.

Acceptance gate: N concurrent REST and WS requests through the parent produce
responses byte-identical to the same requests alone; a worker killed with
SIGKILL is reaped and its slots freed; a dead client leaves no orphan fd.

Evidence (2026-09-18, macOS dev machine, 8 cpus, unpinned by platform, model
parakeet-tdt_ctc-110m Q4_K_M GGUF under `models_local/`)
- `server/prefork.{c,h}` lifted from mynah-tts at 8aa714a with the
  `mynah_asr_` prefix; the decoder-lane and cost-map calls removed (reported
  as "requested but not implemented" if asked); the fork-precondition text
  and the lock inventory rewritten for this tree. `src/threads.c` gained
  `mynah_asr_threadpool_after_fork()` / `mynah_asr_blas_after_fork()`;
  `server/http_util.c` gained thread naming. `server/main.c`: `--prefork W
  --prefork-threads T --cap C --prefork-plan`, reserve-threads before the model
  opens, fork after listen, worker loop on the channel, `conn_done` exactly
  once per connection, queue-full refusal through `refuse_and_close`,
  `TCP_NODELAY` + blocking mode on every accepted descriptor, SIGINT/SIGTERM
  handler with a polled accept loop, `/v1/health.worker`. Fixed a latent double
  `close()` on the WebSocket path.
- `tests/test_server_concurrency.sh` now runs three phases: single process,
  the same requests through `--prefork 2 --cap 2` **byte-identical** (4/4),
  and `--prefork 1 --cap 1 MYNAH_ASR_PREFORK_QUEUE=0` with three concurrent
  requests: 1 served, 2 refused with a readable `503 server_at_capacity` +
  `Retry-After` (refusal latency 0.8 ms). Shutdown leaves no worker. CI runs
  this target on Linux x86, Linux ARM and macOS with the 110m model, so the
  pinned path is exercised there even before the box run.
- Worker SIGKILL: the parent logs "channel closed after 1 completed", the
  surviving worker serves the next three requests (queue peak 1), SIGTERM
  exits 0 with zero survivors.
- `make test` model-free tests green; the plan text (`--prefork-plan`)
  rewritten in ASR terms (bench for T*, stream_load for the W sweep).
Conclusion: the process model of v2 is in place and gated; on macOS it is a
handoff-and-routing test only (no affinity API), which the banner says.
Remaining for `[x]`: one pinned run on the Linux ARM box with the printed
masks checked against `taskset -p`, and a WebSocket stream through the
router with Nemotron (byte-identical to a direct one).
Next action: S2-2 (scheduler and slots); the box run when the owner frees it.
