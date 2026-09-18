# S2 — admission ladder, timeouts, shutdown

Status: IMPLEMENTED (gate passed on the M1 dev host; the lag-under-refusal line
is served by an adjacent phase, see Evidence — Linux numbers pending)

Task: S2-3
Question: make every wait visible and every refusal a real HTTP status.

Known facts (sibling doctrine, measured)
- Rungs: (1) parent slot cap, least-loaded worker; the listener is polled
  unconditionally — gating it hid 97% of a 4.5 s tail before `accept()`;
  (2) bounded parent queue, default 1 per live worker; (3) queue deadline
  checked at **pop**, default 2000 ms; (4) per-request service cap at a step
  boundary. Plus the 4xx rung: `model_not_found` 404, `model_not_streaming`
  400, `language_not_served` 400, no `Retry-After` because retrying will fail
  identically.
- Refuse without RST: write the response → `shutdown(SHUT_WR)` → bounded
  non-blocking drain → close; a residual drain goes to the parent's poll set.
- Accepted sockets: `TCP_NODELAY`, `SO_RCVTIMEO`/`SO_SNDTIMEO` 30 s.
- Utilization-aware admission and fixed-target guards were falsified on the
  siblings: when the machine is full, refuse.
- Graceful shutdown order: stop accepting (poll with 200 ms timeout and re-read
  the flag; closing the listener from a handler does not wake `accept()`) →
  join ingest → stop the scheduler after the current step → answer 503 to
  anything never served → close model. SIGUSR1 must never kill a worker.

ASR-specific rungs
- WebSocket: refuse **before** the 101, as HTTP 503 + `Retry-After` or 4xx, so
  the client reads a status and not `ECONNRESET`.
- Per-stream caps: max audio seconds (default 4 h), max idle without audio
  (default 60 s, with server-side ping every 20 s), max PCM ring (30 s of
  audio; beyond that the ingest thread stops reading).

Plan / gate: `tests/test_server_admission.sh` drives cap+1 streams and asserts
exactly one 503 with `Retry-After`, that the refused client read the body, that
established streams' emission lag did not move (S4 tool), that a client which
stops sending is closed at the idle cap, and that SIGTERM during a stream
drains it and exits 0 with `make leaks --atExit` clean on macOS.

## Evidence

The gate landed as `tests/test_server_protocol.sh` rather than a separate
`test_server_admission.sh`: the admission rungs a worker owns and the protocol
that announces them are the same twelve assertions on the same sockets, and two
scripts would have started eight servers instead of four. The file's header
lists what it proves, in order.

Host: Apple M1 (arm64, 8 cpus), macOS 25.5, Accelerate, `make` default flags,
`nemotron-3.5-asr-streaming-0.6b --quant int8`, `parakeet-tdt_ctc-110m-gguf`
for the model-agnostic REST gate. **Development-machine numbers: signals, not a
serving point** (ENGINEERING §8).

### What was built (the rungs that were still open)

- **Every worker-side refusal now leaves without an RST.**
  `mynah_asr_prefork_linger_close(fd, response, len)` exposes the router's own
  write → `shutdown(SHUT_WR)` → bounded non-blocking drain → `close` sequence
  for responses the ladder does not name, and `mynah_asr_prefork_refuse_and_close`
  became a thin wrapper over the same `linger_run()`. In `server/main.c` the
  WebSocket slot-cap 503, the REST `--max-pending` 503, every pre-upgrade 4xx,
  the `Content-Length` refusals and the 404 all go through it. This is the
  single-process path as much as the prefork one: a worker is a worker.
  Before this, `handle_ws_stream` and `handle_transcribe` wrote the body and
  returned to a `close(fd)` in `handle_conn` with the request body still
  unread — the exact shape the note's "refuse without RST" line forbids.
- **Per-stream audio cap**: `--max-audio-seconds` (default 14400). On exceed the
  ingest enqueues `error audio_limit` and then requests FINALIZE|CLOSE, so the
  audio already accepted still gets its transcript and its `done`.
- **Server-side ping**: `--ping-ms` (default 20000, `0` disables), written
  through `stream_out` from the ingest thread's existing 200 ms tick — no timer
  thread, and a ping can never block the ingest.
- **Idle**: unchanged in mechanism (`--idle-ms`, default 60000), but stated
  correctly now: it is "no frame of ANY kind", and a pong is a frame. A client
  that answers the pings is therefore never idle by this cap; what the cap
  reclaims is a client that has stopped answering. `tests/ws_probe.py idle`
  uses `pong=False` so that what is asserted is exactly what is claimed.
- **Graceful shutdown**: the ingest no longer posts FINALIZE|CLOSE when it
  breaks out on `g_shutdown`. It used to race the scheduler's drain, and
  whichever won decided whether the client read `done` or
  `error shutting_down` — the two say opposite things about what happened.
  Now the drain alone speaks for a shutdown.

### What ran, and what it measured

```
sh tests/test_server_protocol.sh <nemotron> 8373      -> 0, 12/12 checks
sh tests/test_server_stream.sh   <nemotron> 8513      -> 0
sh tests/test_server.sh          <nemotron> 8615      -> 0
sh tests/test_server_concurrency.sh <110m>  8407      -> 0, 8/8 checks
```

The refusals, as the client reads them (from the test's own output):

```
HTTP/1.1 503 Service Unavailable | {"error":{"message":"every worker is at its
  slot cap and the admission queue is full","type":"server_error",
  "code":"server_at_capacity"}}                     + Retry-After
HTTP/1.1 400 Bad Request | {"error":{"message":"unknown query parameter
  'lookahed'; accepted: model, lang, lookahead, format, rate", ...
  "code":"unknown_query_parameter"}}
HTTP/1.1 400 Bad Request | {"error":{"message":"rate '44100' is not served;
  accepted: 16000 (resample client-side)", ... "code":"unsupported_rate"}}
```

Idle: with `--idle-ms 2000 --ping-ms 1000`, a client that never answers reads
`error idle_timeout` and the socket closes; a client on the same server that
does answer counts 2 server pings in 3 s and stays. Both in the same run, on
the same server, which is what makes the pong caveat a measurement rather than
an assertion.

Audio cap: with `--max-audio-seconds 1`, a client streaming a 4.4 s clip at 1×
reads `error audio_limit` and the session is finalised rather than dropped.

SIGTERM with a live stream (`--cap 2`, one stream paced at 1×, signal at
~1.5 s): the client reads `error shutting_down`, the server exits **0**, and
`pgrep` finds **0 survivors**.

`leaks --atExit` on the server binary, same scenario (one client streams
`test_it.wav` at 1×, SIGTERM at 1.5 s, the client reads `shutting_down`):

```
leaks Report Version: 4.0, multi-line stacks
Process 73657: 860 nodes malloced for 84 KB
Process 73657: 0 leaks for 0 total leaked bytes.
```


### Under `-fsanitize=undefined`

The repo's own ubsan flags (the Makefile's `ubsan` target: `-O2 -g
-fsanitize=undefined`, i.e. **without** `-march=native` and `-ffast-math`),
`UBSAN_OPTIONS=print_stacktrace=1:log_path=...`:

```
tests/test_server_protocol.sh    -> 0, 11/11 (this run predates the
                                        audio-limit check; the other 11 are
                                        the ones listed above)
tests/test_server_stream.sh      -> 0 (identity clean at 4 streams and prefork)
tests/test_server_concurrency.sh -> 0, 8/8
runtime error lines              -> 0
```

Recorded because it cost a detour: a first pass built the sanitizer on top of
the SHIPPING flags (`-O3 -march=native -ffast-math -fsanitize=undefined`) and
`test_server_stream.sh` failed identity on one clip out of four — `test_fr.wav`
lost its leading "Bonjour,". It reported 0 runtime errors, and the same tree
with the Makefile's ubsan flags is clean, so the drop is the sanitizer changing
vectorisation under `-ffast-math`, not this change. `CLAUDE.md` already says
`-ffast-math` is load-bearing; the lesson for the next agent is that it must not
be combined with a sanitizer, and that `make ubsan` is the gate precisely
because it drops it.

### From the committed tree

ENGINEERING §12: `make clean && make` on a clean tree with this item's code
committed and `git status` empty, then `tests/test_server_protocol.sh` — exit 0,
all twelve checks, same output as above.

A binary hash is deliberately NOT quoted here. `MYNAH_ASR_BUILD` is
`git describe --always --dirty`, so the commit id is compiled into the binary:
a sha256 written into the note that the same commit contains can never be
reproduced from that commit. Quoting one would look like evidence and be a
circular claim. The reproducible statement is the command above; a hash belongs
in a qualification run's artefact tree, where the binary and the note are
separate files.

## Conclusion

The rungs this item owns are in and gated: a refusal is an HTTP status with a
body the client can read, on every path and in both process shapes; the
per-stream caps are announced with a code before they act; SIGTERM drains,
exits 0 and leaks nothing.

## Unknowns

- The note's gate line "established streams' emission lag did not move" under a
  cap+1 refusal is **not** measured by this test. What exists is the adjacent
  phase in `tests/test_server_stream.sh`: two streams beside a stalled reader
  keep their cadence (lag p50/p95 1460/1806 ms in the run above) while the
  stalled slot is reclaimed. A refusal costs one `accept()` and one lingering
  close on an HTTP thread, so the expectation is that it is invisible to a live
  stream — expected, not measured.
- No Linux number. The lingering close's 500 ms budget and 64 KiB drain cap are
  the siblings' constants, not this repo's measurements.
- Rung 4 (`MYNAH_ASR_PREFORK_SERVICE_MS`) is still observed by the parent and
  not enforced by the worker at a step boundary; `prefork.h` says so and this
  change did not close it.
- `--max-audio-seconds` counts the whole connection, not the current utterance.
  For a long-lived socket doing many utterances that is the conservative
  reading; whether it is the one operators want is untested.

## Next action

S2-6 (worker groups), and the Linux run of the whole server suite on the Axion.
