# S3 — the server proves what it runs

Status: DONE (S3-1..S3-4); the Linux/Axion banner is the one thing left

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

Evidence (2026-09-18, S3-1 server side + S3-3 + S3-4)

Host: M1, `make`, `make check`, `tests/test_server_{stream,protocol,concurrency,metrics}.sh`
all green. Nemotron 0.6B and parakeet-tdt_ctc-110m both exercised.

- **The banner**, four lines on stderr at start, unconditionally, from
  `server/obs.c`: `[FLAGS]` and `[EFFECTIVE-CONFIG]` (src/flags.c, unchanged),
  `[SERVER-CONFIG] v=1` (one token per field: model dir, the name and engine
  read from the model's own `mynah.json`, quant, streaming yes/no, the lookahead
  default and every preset, `chunk_ms`, cap, ring, idle/ping, max audio, max
  pending, batch, http threads, pool threads, BLAS budget, prefork W/T or
  `single-process`, worker index, metrics on/off/served-by-router) and
  `[TOPOLOGY] v=1 worker= pid= configured_mask= actual_mask= threads= pinned=`
  per worker, printed by the worker right after it pins itself, with the mask
  read BACK via `sched_getaffinity` (`unpinned` on macOS, which has no such
  API). Measured on the v1 target: `lookahead_presets=3,0,6,13 chunk_ms=320`,
  derived from `(lookahead_default + 1) x encoder_frame_ms` rather than assumed.
- **`mynah-asr-server --dispatch-map [--json]`** prints the CLI's table from the
  same code, and `mynah_asr_isa_guard()` is now the first statement of the
  server's `main()` too. 0 UNKNOWN rows on this host.
- **`/v1/health` as facts**: `cancelled_by{}` bucketed by the SAME `code` string
  the client's error frame carried (`idle_timeout` `peer_gone`
  `frame_too_large` `protocol_error` `shutting_down` `audio_limit`
  `decode_failed` `other`); `offline{queued,done,max_pending}`;
  `lag_ms{p50,p95,max,count,bucket_ms}` from the scheduler's 8 ms histogram;
  `audio_seconds`; `model{}` and `groups`; `process{}` with uptime, pool and
  prefork threads, `pinned` and `cpu_mask` from the kernel, build, blas, simd
  and the int8/int4 kernels from `src/qmat.c`'s own predicates; `refused{}` by
  code, counted at `refuse_json()`, the one funnel every worker refusal passes.
- **`/metrics`** (`--metrics-port N`, `--metrics-bind ADDR`, off by default):
  its own listener, `SO_REUSEADDR` and never `SO_REUSEPORT` (a second bind on a
  live port is refused -- asserted), no thread, rendered on request in the
  accept loop that owns the listener, bounded 2 ms for the request and 200 ms
  for write+FIN+drain, token bucket 5/s burst 10 with a 429 that does NOT
  render. Series: `sessions/steps/deltas/eou_total`,
  `audio_seconds_total` (the throughput unit), `offline_jobs_total`,
  `cancelled_total{reason}`, `refused_total{code}`,
  `emission_lag_ms_sum/_count/_max`, `emission_lag_over_ms_total{le}` EXACT
  (thresholds are chunk-period multiples rounded up to a bucket edge: 320/640/
  1000 on the v1 target), `slots_active/_cap`, `build_info{}`. Under
  `--prefork` the PARENT answers from its routing table (`worker_up`,
  `_inflight`, `_slots`, `_assigned_total`, `_completed_total`,
  `_over_service_cap_total`, its own `refused_total{code}`, the queue gauges)
  and says in a comment line that the scheduler counters do not exist in that
  process. Per-worker series are never summed. Labels are only
  `worker/reason/code/le`, all compile-time.
- **Honest-metric boundary held**: nothing client-side is exported. No TTFP, no
  stall rate, no safe-play-start -- asserted by the gate, which greps the
  non-comment lines of a live scrape for them.
- **`SIGUSR1`**: a bracketed `[DUMP] v=1 worker= seq= begin/end` of the same
  facts, composed into ONE buffer and written once. Two findings on the way,
  both now closed: (1) the single-process server had **no SIGUSR1 handler at
  all**, so `kill -USR1` killed it -- `server/main.c` now installs one before
  the fork and the workers inherit it, which is what `prefork.c`'s child path
  already assumed; (2) a dozen `fprintf`s per dump produced worker 0's line
  spliced into the middle of worker 1's under `--prefork 2`.
- **Thread names verified** on a live process: `mynah-accept` / `mynah-recv`,
  `mynah-http` (renamed `mynah-ingest` while driving a stream), `mynah-sched`,
  `mynah-out<fd>`. Listed in docs/server.md.

Cost on the step path: two relaxed atomic adds (audio samples, lag sum) next to
adds that were already there. Nothing else was added.

Gate: `tests/test_server_metrics.sh`, in `make test` and as
`make test-server-metrics`. Model-agnostic and REST-only (it runs on the 110m in
CI). It asserts the four banner lines; `/metrics` and `/v1/health` agreeing on
`offline.done` and `slots.cap`; `audio_seconds_total` within 3% of 4 x the
clip's real length (computed from the fixture, not hardcoded); sessions/steps/
deltas exactly 0 on a REST-only run; the exact threshold counters present; no
client-side latency in the page; the second bind refused; 20 back-to-back
scrapes producing 429s; one bracketed `[DUMP]` with the process still alive;
and under `--prefork 2` the router answering with two `worker` labels, both
workers' `[TOPOLOGY]` lines, both workers dumping on one signal and all three
processes surviving it.

Measured, single process, four REST requests of a 4.338 s clip on the 110m:
`mynah_asr_offline_jobs_total 4`, `mynah_asr_audio_seconds_total 17.353`,
`sessions/steps/deltas 0`. One WebSocket utterance on Nemotron (a client that
pushed the clip faster than real time, which is why the lag is large and honest):
`sessions 1 steps 17 deltas 15 audio_seconds 5.229`,
`emission_lag_ms_sum 62509.847 _count 15 _max 5339.664`,
`over_ms_total{le="320"|"640"|"1000"} = 15/15/15`, health
`lag_ms{p50:2048, p95:2048}` -- 2048 is the histogram's last bucket
("2048 ms or more"), which is the documented saturation, not a coincidence.

STILL OPEN / UNKNOWN
- Nothing here has run on Linux: `actual_mask`, `pinned` and the whole
  configured-vs-actual comparison are exactly the fields macOS cannot exercise.
  The Axion banner has not been pasted into `axion-first-run.md` yet.
- The gate scrapes at rest and after a small REST batch; it does not scrape
  UNDER load, so "the per-worker sums equal the parent's counters" from this
  note's original Gate line is NOT asserted -- and cannot be as designed, since
  the parent counts connections and a worker counts sessions. The honest version
  is what shipped: two views, each labelled with what it can see.
- `mynah_asr_refused_total` is exported by both the worker and the router under
  the same metric name with different label sets (`worker`+`code` vs `code`).
  They are never scraped from the same process, but a single Prometheus job that
  scrapes both a router and a worker would need `job`/`instance` to keep them
  apart.

Conclusion: every serving claim this repo makes can now be tied to a banner and
a scrape from the same process. Next action: S3-3's Linux half -- run the fleet
on the Axion, paste the banner, and check `pinned=yes` with a real mask.
