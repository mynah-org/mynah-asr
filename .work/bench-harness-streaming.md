# S4 — the streaming load generator and the metric definitions

Status: S4-1, S4-2, S4-4 DONE; the qualifying run (S4-3) is open

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

## Evidence — S4-1 / S4-2 / S4-4 landed (commit 08026d3)

`tools/bench/streaming_metrics.py` is the single definition; `stream_load.py`
computes nothing itself and gained `--mode wave|soak`. The known-answer
`--self-test` (nearest-rank ranks, the frame-lookup arithmetic, a canonical
utterance whose four metrics are worked out by hand in the comments, and the
edge cases: single sample, no deltas, `done` missing, unpaced, empty run) runs
in `make test` without a model. `docs/serving.md` is the operations order.

### Evidence — macOS dev host, NOT a serving number

Every number below is DIAGNOSTIC and NON-QUALIFYING, for three independent
reasons, and none of them may be quoted as a capacity result:
1. Apple M1, 8 cpus, macOS, Accelerate — the product is Linux x86-64/ARM64.
2. Single process, no `--prefork`, no pinning (`--prefork-plan` says pinning is
   UNAVAILABLE on this platform).
3. The host was **not idle**: `load average 3.73` and a second, unrelated
   `mynah-asr-server` from a parallel agent session was live on port 8514
   throughout. ENGINEERING.md §10 says refuse to measure under another load;
   this run exists to prove the harness, not the server.

The purpose was to exercise every path of the tool against a live server:
pacing valve, rejection counting, windows, drift, manifest, envelope lines.
All of them fired.

Setup: `./mynah-asr-server -m <models_local>/nemotron-3.5-asr-streaming-0.6b
--quant int8 -p 8611 --cap 2` (one process, 8 threads, 2 stream slots, batch 8,
int8 checkpoint). Tree `08026d3`, clean. Client python 3.10.0. Bank:
`samples/*/fleurs_15*.wav tests/audio/test_*.wav`, `--class-bounds 7,11` ->
short 8 / medium 14 / long 5 clips, 3.6 s to 13.9 s. `samples/en/fleurs_long.wav`
(94.6 s) was excluded on purpose: a class must be shorter than the soak.
`/v1/health` steps 0 -> 356 -> 428 across the two runs and `sessions` ended at
15 = 13 soak utterances + 2 admitted wave utterances, so the client's
accounting and the server's agree.

**SOAK, concurrency 2, 60 s, `--warmup 10 --window 20 --seed 42`**

```
  wall 64.3 s
  utterances 10/10 ok, 0 errors, 0 rejected, 3 excluded by warm-up
  audio 83.7 s, 172 deltas, span 63.3 s
  pacing: max lateness 68.8 ms vs half a frame 50 ms -> NOT PACED (cadence is DIAGNOSTIC)
  TTFP                       p50/p95 ms: 1664 / 5575  (max 5575, n=10)   DIAGNOSTIC
  emission lag (client)      p50/p95 ms: 318 / 2041  (max 2896, n=172)   DIAGNOSTIC
  emission lag per-utt p95   p50/p95 ms: 599 / 2896  (max 2896, n=10)    DIAGNOSTIC
  server lag_ms              p50/p95 ms: 318 / 2041  (max 2895, n=172)   DIAGNOSTIC
  finalization lag           p50/p95 ms: 576 / 2897  (max 2897, n=10)    DIAGNOSTIC
  backlog max                p50/p95 s : 0.504 / 2.184 (max 2.184, n=10) DIAGNOSTIC
  pacing lateness            p50/p95 ms: 8 / 69  (max 69, n=10)
  drift emission_lag_ms     : max 70.6% (window 3 vs pooled p95 2041.0, 4 windows)
  drift server_lag_ms       : max 70.7% (window 3 vs pooled p95 2040.7, 4 windows)
  drift backlog_s           : max 67.2% (window 2 vs pooled p95 2.0, 4 windows)
  drift ttfp_ms             : max 66.6% (window 2 vs pooled p95 5574.9, 3 windows)
  [FAIL] TTFP p95 5575 (limit 520) · emission lag p95 2041 (320) ·
         finalization p95 2897 (500) · backlog max 2.184 s (0.640) ·
         drift 67-71 % on all four (limit 20)
  [INVALID] client could not pace at 1x: cadence percentiles are refused
  verdict: INVALID
  per-window emission lag p50/p95 ms:
    [ 0- 20 s] n=11   975 / 1248
    [20- 40 s] n=64   803 / 2195
    [40- 60 s] n=84   262 /  665
    [60- 80 s] n=13   263 /  600
```

The pacing valve is not a client artefact here: the client's `sendall` blocked
because the server's ingest stopped reading (the `--ring-seconds` backpressure
of `docs/server.md`), so the 68.8 ms lateness is the same event as the 2.2 s
backlog. The harness refused to quote a cadence percentile, which is exactly
the behaviour §4 of the method note asks for, and the verdict is INVALID rather
than NOT STREAMABLE: an unpaced client cannot testify about a server.

**WAVE, concurrency 4 against `--cap 2`, 1 utterance each**

```
  wall 12.0 s
  utterances 2/4 ok, 0 errors, 2 rejected, 0 excluded by warm-up
  audio 22.4 s, 49 deltas, span 12.0 s
  pacing: max lateness 6.2 ms vs half a frame 50 ms -> PACED
  TTFP                       p50/p95 ms: 1461 / 1632  (max 1632, n=2)
  emission lag (client)      p50/p95 ms:  177 /  259  (max  283, n=49)
  server lag_ms              p50/p95 ms:  177 /  259  (max  283, n=49)
  finalization lag           p50/p95 ms:   99 /  283  (max  283, n=2)
  backlog max                p50/p95 s :  0.284 / 0.284 (max 0.284, n=2)
  [FAIL] TTFP p95 1632 ms (limit 520)
  [PASS] emission lag p95 259 (320) · finalization p95 283 (500) · backlog 0.284 s (0.640)
  verdict: NOT STREAMABLE
  rejections (503 before the upgrade, counted, not errors): 2
```

The two streams above `--cap 2` were refused before the upgrade and counted as
REJECTIONS, not errors — the admission ladder behaving as `docs/server.md`
describes. The two admitted streams held 1x pacing and their emission lag sat
inside the envelope while TTFP did not: the first delta costs ~1.5 s on this
host with `lang=auto`. Whether that is LID on the first seconds, the first
encoder chunk on a cold slot, or both, is NOT determined by this run.

Teardown: SIGTERM to the saved PID, `survivors=0` for both the server and the
client processes.

Conclusion: the harness works end to end against a live server, including both
of its refusals (pacing, admission). It has produced no serving number.
Next action: S4-3 — run the same two commands on the ARM box from a clean
committed tree at the operating point derived by `--prefork-plan`, with a soak
of at least 10 minutes and the long class included.
