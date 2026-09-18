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

Evidence / Conclusion / Next action: after S2.
