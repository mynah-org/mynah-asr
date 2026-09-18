# S2 — prefork parent, pinned workers, fd handoff

Status: OPEN

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

Evidence / Conclusion / Next action: after S1-5.
