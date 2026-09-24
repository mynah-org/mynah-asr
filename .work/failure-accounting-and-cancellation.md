# Failure accounting, cancellation lifecycle and fault injection

Status: OPEN — P0 for 2026-09-25, before any further qualification and before
returning to first-word latency.

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

## Evidence

(none yet — this note is written before the work starts)

## Conclusion

(open)

## Next action

Start with P0-a in the harness, gated by planted-fault tests, then P0-c
(read-only trace of the cancellation path), then P0-b counters, then P0-d.
