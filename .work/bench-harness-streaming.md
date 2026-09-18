# S4 — the streaming load generator and the metric definitions

Status: OPEN

Task: S0-2, S4-1, S4-2, S4-4
Question: one tool that opens N WebSocket streams paced at exactly 1x, one
module that defines every metric once, and a protocol (WAVE / SOAK) whose
verdicts are comparable across hosts and commits.

Known facts
- `tools/eval/ws_client.py` is a single-connection stdlib client; the only
  concurrency tool is offline (`tests/bench_throughput`).
- Sibling harness properties to copy: one client **process** per stream (a
  GIL'd thread cannot pace 100 streams); marks are client-observed with the
  server-side `lag_ms` beside them; the harness refuses to print cadence
  percentiles when it could not keep 1x pacing (the analogue of the coalesced-
  read valve); percentiles nearest-rank over requests, never a ratio of
  percentiles; a manifest per run (commit, dirty state, binary sha256, host,
  masks, env, bank version + sha256, concurrency, duration, analysis windows);
  a mixed-length stratified bank with a seeded schedule; a drift gate across
  windows; WAVE may disqualify and never promote.
- Bank: the committed FLEURS clips under `samples/` (10 clips × 5 languages,
  CC-BY 4.0) plus `tests/audio/long_60s.wav`; classes short / medium / long;
  a conversational class needs a source with a licence we can commit.

Metrics (`tools/bench/streaming_metrics.py`, with a known-answer self-test)
- TTFP: first delta wall time − first audio byte sent.
- emission lag per delta: delta received − send time of the last sample of the
  chunk that produced it (client), and `lag_ms` (server); p50/p95 per stream
  and pooled.
- finalization lag: `done` received − last audio byte sent (after `finalize`).
- backlog: server-reported audio seconds received minus consumed, max.
- chunk service time: scheduler step wall / slots in the step, per worker.
- STREAM_RTF: compute seconds / audio seconds; capacity, not continuity.
- correctness: text equals the single-stream text for the same clip
  (byte-identical), else the run is invalid whatever the timing says.
- errors / rejects / timeouts; drift of every p95 across windows.

Plan
- S0-2 v0 tool: `tools/bench/stream_load.py --streams N --clips ... --pace 1.0`,
  stdlib only so it runs on the box without a venv; prints the metrics.
- S4-1 metrics module + self-test in `make test` (no model needed).
- S4-2 `--mode wave|soak --duration --bank --seed`, manifest, windows,
  drift, refusal on pacing failure, `results.json`; `make bench-stream-wave`,
  `make bench-stream-soak`.
- S4-4 `docs/serving.md`: how a new box is approached (inspect → build for the
  ISA → banner and dispatch map → wave → soak), the envelope, the vocabulary.

Gate: a soak on the Axion at the derived operating point promotes or fails
by the envelope in `serving-v2-design.md` §6, with the manifest in
`axion-first-run.md`.

Evidence: S0-2 v0 written as `tools/bench/stream_load.py` (commit 578a4fa): one
process per stream, real-time pacing, TTFP / emission lag / finalization lag,
pacing lateness gate, text-identity gate across streams, JSON with manifest.
Not yet exercised against a live server (no converted model on the laptop);
first run is S0-3 on the ARM box.
Conclusion: pending S0-3.
Next action: run it against the v1 server on the box at N = 1, 2, 4, 8, 16.
