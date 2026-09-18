# S2 — admission ladder, timeouts, shutdown

Status: OPEN

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

Evidence / Conclusion / Next action: after S2-2.
