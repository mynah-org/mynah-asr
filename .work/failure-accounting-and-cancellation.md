# Failure accounting, cancellation lifecycle and fault injection

Status: IN PROGRESS (2026-09-25). P0-a, P0-b, P0-c and the single-fault half of
P0-d done on macOS; open: the fault-injection soak on the box, the Nemotron
re-run of the fault suite, the full protocol/stream tests with Nemotron, and
qwen-tts.

Task: S12-17, S12-18, S12-19, S12-20

## Question

"0 streams lost" held for four 30-minute soaks when recounted from the raw
records (`.work/plateau-campaign-2026-09.md`, AUDIT 2026-09-24). But the
harness could have hidden failures, and it has never been shown what the
SERVER does when a failure is provoked on purpose. Two questions:

1. Can the harness miss a failure? (It can today: three real gaps.)
2. When a client or a component fails mid-utterance, does the server release
   everything — slot, request, model work, K/V and conv state, buffers — and
   leave the other streams unharmed?

## Known facts (audit of 2026-09-24)

Gaps in the counting code (`tools/bench/stream_load.py`,
`tools/bench/streaming_metrics.py`):

1. `ok` = no error and not rejected. An utterance where the server closes the
   socket WITHOUT a `done` frame (the reader breaks on opcode 0x8 and sets no
   error), or that receives no event at all, counts as OK; its finalization
   silently leaves the percentiles.
2. `aggregate()` drops the first `--warmup` seconds BEFORE counting `errored`:
   a failure in the first 30 s of a soak never reaches `counts.errors`, which
   is what v2_verdict bound 1 reads.
3. SOAK mode never asserts that every stream process lived to the deadline
   (`expected` is set only in WAVE mode): a dead client process stops producing
   records silently.
4. Harness artefact (not a failure): the schedule gives stream i the clips
   i, i+C, i+2C... of a bank alternating three length classes, so at C a
   multiple of 3 (96, 120, 144) each stream plays ONE class for the whole run.

Recount of the four qualification soaks, warm-up included: 0 errors, 0
rejections, 0 utterances without done/finalization, 0 empty transcripts, every
stream alive and sending a full 30 minutes of audio. The zeros stand; the code
must be fixed so the next run cannot hide them.

Related history: mynah-tts / qwen-tts has already seen a ZOMBIE inference
after a dropped connection (the model kept computing for a client that no
longer existed). That known failure mode is to become a regression test here,
and the same philosophy is to be applied back to qwen-tts.

## Plan (tomorrow, in this order)

### P0-a — Harness accounting (S12-17)
- An utterance with no `done` frame, or no events, is an ERROR
  (`server_disconnect` / `no_done`), never OK.
- Warm-up errors are COUNTED (reported separately from the KPI window), never
  dropped; v2_verdict bound 1 reads all of them.
- SOAK asserts every stream process lived to the deadline and reports the dead
  ones as `client_process_death`.
- The schedule stride cannot lock a stream to one length class (C multiple of 3).
- Each fix gets a case in the `streaming_metrics` self-test and a planted-fault
  check (the fix must turn a synthetic record into an error).

### P0-b — Granular counters and a conservation invariant (S12-18)
Server-side and client-side counters, not one `errors`:
`accepted`, `completed`, `client_disconnect`, `server_disconnect`,
`cancelled`, `rejected`, `timeout`, `protocol_error`, `worker_failure`,
`internal_error`, `active`.
Invariant checked at the end of every window and every run:

    accepted = completed + cancelled + failed + active

A request that disappears from the statistics becomes a gate failure.

### P0-c — Cancellation lifecycle audit, read in the code (S12-19)
Trace and document, with file:line, the path

    socket disconnect -> request cancellation -> scheduler/slot cancellation
      -> model work stops -> state freed (slot, K/V + conv caches, mel/VAD
         buffers, output ring, decoder state)

for every exit: clean close, RST mid-utterance, RST during finalization,
timeout, server-side cancel, worker death. Do NOT assume that destroying the
WebSocket implies the rest: on a CPU streaming server a zombie step keeps
consuming compute and slots for a client that is gone.

### P0-d — Fault injection (S12-20)
Provoked failure modes, each with verifiable invariants:
- client disconnect / RST mid-utterance, and during finalization
- client stops sending but keeps the socket open (idle timeout path)
- malformed / invalid WebSocket frames, protocol violations
- premature server close
- HTTP/WS 4xx / 5xx / 503 admission rejection, overload
- worker process death; server SIGTERM / restart
- load-generator (client) process death

Invariants per case: slot and request released; no zombie inference (model
work for the stream stops within one step); no K/V, conv, buffer or decoder
state left live; the right counter incremented; OTHER streams unperturbed
(their lag within the run's normal range); RSS and the active-stream count
return to baseline.

Then a **fault-injection soak**: C=64 or 80, 10-15 minutes, a deterministic
percentage of clients aborted at different points (before first partial,
mid-utterance, during finalization, idle-open). Pass = zero orphan work, zero
slot leak, no progressive RSS growth, capacity recovered after every abort,
healthy streams keep a reasonable latency, and the conservation invariant holds.
Worth more than another nominal soak.

## Priority for 2026-09-25

**P0** failure accounting + cancellation lifecycle + fault injection (this note)
-> **P1** first-word ~2.3 s p95 investigation (model side; serving and the
decoder commit policy are already ruled out, S13-5c) -> **P2** the TODOs
already on the board (promotion decision S12-13, cleanup S12-14, French set
and x86 S12-16).

## Evidence (2026-09-25, macOS dev machine)

Everything below ran on the M-series development Mac with the English-only
`parakeet-realtime-eou-120m` streaming pack (the Nemotron pack lives on a
volume that was not mounted). The fault suite is model-agnostic by design: it
compares a stream against the same binary's offline answer and reads the
server's own counters; the Nemotron re-run is an open item, not an assumption.

### P0-a — harness accounting (commit "Count every utterance the load harness starts...")

Each gap has a planted fault that the OLD code passes and the NEW code fails:

| gap | planted fault | old | new |
|---|---|---|---|
| no `done` | fake server closes before `done` (3 variants) | `ok 1, errors 0` | `server_disconnect` |
| no `done` | metrics fixture: 1 good + no-done + no-events | `ok 3, errors 0` | `ok 1, errors 2`, loss FAIL |
| warm-up | one error inside the warm-up | `errors 0`, PASS | `errors_warmup 1`, `errors_total 1`, FAIL |
| dead stream | real SIGKILL of one of 3 stream processes, 6 s soak | `10/10 ok`, PASS | `client_process_death 1`, `DIED: [2]`, FAIL |
| stride | C in {3,6,24,96,120,144} | every stream locked to 1 class | every stream plays every class; C coprime unchanged |
| conservation | an extra start with no record; a duplicated record | not checked | INVALID |

### P0-b — server session books (commit "Cancel streams whose client vanished...")

`sessions == completed + cancelled + aborted + slots.active`, counted once at
slot release, read under the lock that claims and releases: exact in every
snapshot (`balanced` in /v1/health, `mynah_asr_sessions_balanced`, the
`[DUMP] ... books` line). Router: `assigned == completed + lost + inflight` per
worker. v2_verdict row B fails on one unbalanced dump line or an abandoned
slot never recovered; row A on broken client conservation or a dead stream.

### P0-c — the cancellation lifecycle, in the code (line numbers at that commit)

    client socket                     ingest thread (server/main.c)
      close frame  ----------------->  0x8: FINALIZE|CLOSE            -> tail, done, "completed"
      finalize + FIN  -------------->  ws_read_loss: legal half-close -> tail, done, "completed"
      FIN / RST mid-utterance ------>  ws_read_exact EOF/ERROR (726) -> ws_read_loss (838)
                                         -> REQ_CANCEL peer (1097)     -> no tail, "peer_gone"
      silence, socket open  -------->  poll tick, --idle-ms (967)    -> REQ_CANCEL idle
      stall mid-frame  ------------->  SO_RCVTIMEO -> WS_READ_TIMEOUT  -> REQ_CANCEL idle
      RSV bits / bad control ------->  RFC 6455 check (1003)          -> REQ_CANCEL protocol_error
      frame > --max-frame-bytes ---->  (1018)                         -> REQ_CANCEL frame_too_large
      writer failed (EPIPE, overflow)  out_failed (940)               -> REQ_CANCEL peer
    slot (server/slot.c)
      mynah_asr_slot_request (374): CANCEL is sticky, wakes a pusher parked on
      a full ring (slot_push_open_locked, 278), so the ingest never sits on it.
    scheduler (server/sched.c), pass (1) before any work (970)
      REQ_CANCEL          -> sched_cancel (507): error frame, sched_close_session
      writer failed, or peer gone (peer_gone_ex, 1002; HARD hangups only while
      a tail is flushed: stream_out.c 401)       -> sched_close_session "peer_gone"
      step failure        -> sched_cancel "decode_failed" (803, 825)
      SIGTERM             -> sched_drain_on_stop (933): "shutting_down"
      sched_close_session (494): close frame, writer finish, slot_finish (slot.c
      252) records the outcome and moves to DONE in one lock
    model work stops: the cancelled slot is never staged again; granularity is
      ONE step (a batched step in flight completes, its rows are discarded by
      the closed session). Measured: <= 0.16 s of audio after the disconnect.
    state freed, by owner:
      PCM ring + arrivals: slot-owned, reset by mynah_asr_slot_release (209)
      output ring + socket: refcount 2 (writer + producer), stream_out.c 173;
        the writer closes the fd
      stream (K/V + conv caches, mel/VAD buffers, decoder state): POOLED with
        the slot by design, never freed per session; the next claim sets
        needs_reset (slot.c 190), so nothing of a cancelled session survives
        into the next one. Bounded memory, measured flat (abort-loop, leaks).
      counting: mynah_asr_sched_release (sched.c 305) -> count_outcome_locked
        (296), once, whoever releases
    ingest wait: 60 s + 5 s; if the scheduler still has not finished,
      mynah_asr_sched_abandon (main.c 1118): the scheduler releases slot and
      writer when it ends the session (abandoned.recovered). Before: the slot
      leaked for the life of the process.
    worker death (prefork.c 2658): the router charges its in-flight
      connections to `lost`, marks it down, routes to the rest. NOT respawned.

### P0-d — provoked failures (tests/test_server_faults.sh, make test-server-faults)

Measured BEFORE the server fix (same suite, commit 75e4bea binary):

| case | before | after |
|---|---|---|
| rst-mid / fin-mid (speech) | 0.16-0.32 s fed after, `peer_gone` +1 | 0.00 s |
| **rst-mid-silent / fin-mid-silent** | **27.4 / 27.1 s fed after, counted nowhere** | **0.00 s, `peer_gone` +1** |
| **close-then-rst-silent** | **27.4 s fed after, counted `completed`** | **0.00 s, `peer_gone` +1** |
| stall-mid-frame | finalized, no error frame, counted nowhere | `idle_timeout` |
| rsv-bits / bad-control | idle timeout after 2 s | `protocol_error` at once |
| books | not kept | balanced in every case, final 69 = 18 completed + 51 cancelled |

Speech hid the zombie: a delta written to a dead peer fails with EPIPE within
a write or two, so the output side cancelled it. Silence writes nothing.
Every zombie case first checks it is discriminating (>= 5 s still queued at the
disconnect; measured 27 s), so a fast machine cannot pass it vacuously.

After: 16 cases (plus worker-kill under `--prefork 2`) green 3/3 runs; green on
a UBSan build (no runtime error, 0 warnings in the new code); `leaks` on the
live server after ~70 aborted sessions: 0 leaks; abort-loop RSS 191 -> 192 MB.
worker-kill: the router charged 1 connection to `lost`, the other worker's
stream finished with the reference transcript, a new stream after the death was
served correctly. Regressions: test_stream_out, test-server-metrics,
test-server-concurrency OK; the protocol cases the change touches (audio_limit,
idle, pings, 503, SIGTERM frame and exit 0) OK on the EOU pack.

macOS facts found on the way (they decided the code, so they are kept):
a RST shows as POLLIN + a read error, not POLLHUP/POLLERR; a FIN raises POLLHUP
once POLLIN is requested; setsockopt(SO_LINGER) on a socket the peer already
reset returns EINVAL.

Findings not fixed (recorded, not assumed away):
- A prefork worker that dies is not respawned: the fleet runs on with W-1.
- With `--threads` (HTTP threads) <= `--cap`, the stream past the cap waits in
  accept() for a thread instead of reading its 503 (v2_qualify already warns).
- qwen-tts (read-only study of ../qwen-tts, 2026-09-25): cancel on disconnect is
  OFF by default (`QWEN_CANCEL_ON_DISCONNECT`), so the zombie is its default;
  the synchronous clone/design path ignores write errors; its terminal counters
  have no conservation invariant (queue-full, stale and fatal drains bypass
  them). Its `tests/cancel_correctness.py` (K1-K9, "zombie seconds") is the
  model this suite follows.

## Conclusion (so far)

The qualification zeros were real, but the server had a genuine zombie: a
client that vanished while sending silence (or reset after asking for the
tail) had up to --ring-seconds of audio run through the model for nobody, and
it was counted nowhere. Fixed, measured before/after, and gated. The books are
now exact on both sides, and a verdict fails when either side loses a session.

## Next action

1. Fault-injection SOAK mode in stream_load (abort percentage and points) —
   in progress; then 10-15 min at C=64/80 on the box.
2. Re-run `make test-server-faults`, test-server-protocol and
   test-server-stream with the Nemotron pack (mount the models volume).
3. qwen-tts: the same suite shape (silent-audio zombie, books, abandon), and
   decide with the user whether cancel-on-disconnect becomes its default.
4. Worker respawn in the prefork router: a decision, not a fix — ask.
