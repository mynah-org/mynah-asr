# S3 — the server proves what it runs

Status: OPEN

Task: S3-1, S3-2, S3-3, S3-4
Question: make every serving claim checkable from the process itself: which
kernels resolved, which cpus, which flags, how many streams, where the time
goes.

Plan
- S3-1 **effective-config banner** at start, unconditionally: model, quant,
  ISA kernels resolved by the dispatcher's own predicate (int8 dot = neon-sdot
  / i8mm / avx512vnni / avx2 / scalar; GEMM = own/openblas/accelerate),
  threads per worker, masks configured and actual, BLAS ownership, presets,
  VAD, and one line per env flag read: `requested | effective | reason`
  including `IGNORED: not compiled into this build`. `[TOPOLOGY] v=1` per worker.
- S3-2 `--dispatch-map [--json]`: rows `compiled · supported · env · resolved
  · reason`, resolved only from a registered predicate, a runtime call, or a
  pure gate, else UNKNOWN (a finding); `IDLE HARDWARE` footer (supported and
  not compiled: sve, sve2, i8mm, bf16 on the Axion today); a fatal ISA guard
  as the first statement of `main()` that fires only on a definite absence.
- S3-3 `/v1/health` reports **facts, not configuration**: groups, per-worker
  slots active/cap, queue depth, counters per refusal reason, warmups
  requested/done, lane width actually running. `/metrics` on its own port
  (never the service port, not `SO_REUSEPORT`, token-bucket 5/s, 429 without
  rendering): per-worker series never summed, `_sum/_count` pairs plus exact
  threshold counters (`ttfp_over_{250,500,1000}ms_total`,
  `emission_lag_over_{chunk}_total`), `audio_seconds_total` as the throughput
  unit, `build_info` labels; cardinality: `worker`, `group`, `route`,
  `outcome` only, never language or text.
- S3-4 `SIGUSR1` dump bracketed `[DUMP] seq=`; parent forwards to workers;
  threads named `mynah-sched`, `mynah-ingest`, `mynah-out`, `mynah-pool`.

Honest-metric boundary: the server exports what it observes (TTFP from first
audio byte received, emission lag from chunk arrival, finalization lag from
last byte, backlog seconds). It never exports a client-side latency.

Gate: `tests/test_server_metrics.sh` scrapes under load and checks the
per-worker sums equal the parent's counters; the banner of a run on the
Axion is pasted into `axion-first-run.md`.

Evidence (2026-09-18, S3-1 + S3-2, library + CLI only)

What exists now, on an M1 (`make`, `make test`, `make check` all green):

- `src/flags.{c,h}` — ONE registry of the 17 environment variables this tree
  reads (15 `MYNAH_ASR_*`, `OPENBLAS_NUM_THREADS`, `MALLOC_COUNT_OUT`), each with
  scope, default, description and an optional `inert` predicate returning the
  reason it cannot act in this build on this host.
  `mynah_asr_flags_print()` emits exactly two lines: `[FLAGS] v=1 ...` and
  `[EFFECTIVE-CONFIG] v=1 build=... blas=... simd=... NAME=req->eff(reason)`,
  reason being `applied`, `clamped`, or `IGNORED: <why>`. Measured on this host:
  `MYNAH_ASR_CAPS=vnni` -> `IGNORED: not an x86 build...`,
  `OPENBLAS_NUM_THREADS=3` -> `IGNORED: this build links Accelerate...`,
  `MYNAH_ASR_THREADS=999` -> `999->64(clamped)`.
- `tools/check_flag_registry.py`, wired into `make check`: fails when a name read
  through `getenv()`/`env_int()`/`mynah_asr_flag_*()` in `src/ cli/ server/
  tests/` is absent from the table, or when the table names a flag nothing reads.
  Currently "17 flags, all read, all described".
- `src/dispatch.{c,h}` — `--dispatch-map [--json]`: 8 rows
  (`kernel.int8_dot`, `kernel.int4_dot`, `kernel.int8_gemm`, `gemm.f32`,
  `backend.metal`, `backend.cuda`, `threads.pool`, `threads.blas_budget`),
  **0 UNKNOWN on this host**. `resolved` comes from an owner predicate exported
  by `src/qmat.c` (`mynah_asr_qmat_int8_kernel/_int4_kernel/_qgemm`), from a real
  runtime call, or from a pure gate; the tag in `reason` says which. On the M1 it
  reports `neon-sdot` / `neon-sdot-q4` / int8 GEMM OFF / accelerate / metal
  compiled with a device but not selected / cuda not compiled / pool 8.
  `IDLE HARDWARE:` reports dotprod present+USED, i8mm and bf16 absent, sve and
  sve2 unknown (Apple publishes no `hw.optional.arm.FEAT_SVE` key, and a missing
  key is not a zero) — each idle line names the kernel that would have to be
  written.
- `mynah_asr_isa_guard()` is the first statement of `main()`: exits 78 with one
  line when the binary carries an ISA the CPU DEFINITELY lacks. Tri-state
  probes, so UNKNOWN never refuses. It returns 0 on the build host, asserted by
  `tests/test_flags.c` (model-free, in `make test` and the TESTS list).
- Documented in `docs/backends.md` (one paragraph each).

What is left for server/ (another owner): print `[FLAGS]`/`[EFFECTIVE-CONFIG]`
and the dispatch map unconditionally at server start, add the `[TOPOLOGY] v=1`
per-worker line, and call the ISA guard in `server/main.c` too. The library side
needs nothing further: `mynah_asr_flags_print()` and `mynah_asr_dispatch_print()`
are already linked into `mynah-asr-server`.

Conclusion: the CLI can no longer be quoted without its own dispatch statement.
Next action: S3-3, plus the server banner wiring above.
