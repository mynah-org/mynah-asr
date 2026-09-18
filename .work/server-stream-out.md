# S2 — asynchronous bounded output writer

Status: OPEN

Task: S2-4
Question: get socket writes off the scheduler thread so a slow or stopped
reader can only hurt its own stream.

Known facts
- mynah-tts `server/stream_out.{c,h}`: `<stddef.h>` only; one byte ring
  allocated at start (cap 1 MiB), producer converts outside the lock and
  memcpys under it, the detached writer writes from the ring without the lock
  (validated with ThreadSanitizer), refcount 2, overflow or send timeout =
  cancel the stream and wake the producer, peer-gone detection (`POLLRDHUP`)
  under the same mutex the writer takes to close. Measured: next client served
  after 66.5 s → 2.6 s once a reader stalled.
- For ASR the payload is JSON text frames, a few hundred bytes per chunk: the
  ring can be small (64 KiB) and overflow is a real signal of a dead reader.

Plan: copy, rename, replace PCM16 conversion with WebSocket text-frame
framing (server → client frames are unmasked; opcode 0x1). The scheduler's
`on_delta` only enqueues. `stream_out` owns the fd from admission to close,
including the close frame and `{"type":"done"}`.

Gate: a client that stops reading after the first delta is cancelled within
the send timeout with `failed_enqueues`/`failed` counted separately; a
concurrent normal client's emission lag p95 does not move; JSON bytes
byte-identical to the synchronous path on the fixtures.

Files/functions inspected
- sibling (read-only): `mynah-tts/server/stream_out.{c,h}` — ring, refcount,
  `mark_failed_locked`, `stream_out_peer_gone`, `write_all_or_gone`.
- this repo: `Makefile` (TESTS, the `tests/%` pattern rule, `leaks`),
  `tests/test_threads.c` (model-free self-test shape), `server/http_util.h`
  (header conventions), `server/main.c:695` (`SIGPIPE` already ignored).

Evidence (2026-09-18, S2-4 port)

Built: `server/stream_out.{c,h}` with the `mynah_asr_` prefix. The sibling's
design is kept intact — one byte ring allocated at start, producer copies under
the mutex, detached writer writes contiguous spans out of the ring WITHOUT the
mutex (the span is only returned to the producer after the write), refcount 2,
overflow / send timeout / socket error = mark failed and never block the
producer, peer-gone polled under the same mutex the writer takes to close.
What changed for ASR:
- the payload is opaque already-framed messages, not PCM16: no conversion
  buffer and no HTTP chunk framing, the ring is a plain byte stream and the
  caller owns the WebSocket frame;
- ring capacity and send timeout are arguments, not environment variables
  (default 64 KiB / 5000 ms) — for JSON deltas of a few hundred bytes, 64 KiB
  is hundreds of messages of backlog, so an overflow is a dead reader, not a
  burst;
- `failed` became an atomic read outside the mutex: the scheduler consults it
  once per slot per step and must not queue behind a memcpy;
- the failure cause is classified once, at `mark_failed_locked`, into
  `peer_gone` / `send_timeout` / overflow (errno 0), and exported in the stats
  instead of being re-derived from errno by the caller.

Measured, `tests/test_stream_out` (model-free, in `make test`), macOS arm64,
clang -O3, socketpair with 4 KiB socket buffers where a stall is needed:
- 1000 messages (140 716 B, 16..271 B each) through a 16 KiB ring: every byte
  arrived, byte-identical and in order, ring peak 16 354 B — i.e. the ring
  wrapped ~9 times, so the split-message path is exercised, not bypassed.
- stalled reader, SO_SNDTIMEO 500 ms: 64 × 4 KiB accepted with a worst enqueue
  of 0.007 ms (bound asserted: < 50 ms); the stream marked itself failed after
  1010 ms; the next enqueue returned -1 immediately with `failed_enqueues` = 1
  and `send_timeout` = 1, `peer_gone` = 0.
  Note worth keeping: detection costs ~2x the timeout, not 1x, because a
  `send()` that transfers a partial span before SO_SNDTIMEO expires returns
  that partial count rather than an error and the next `send()` restarts the
  timer. The guarantee is "a small multiple of the timeout"; the test bounds it
  at 4x.
- reader closes: `peer_gone` returns 1 without consuming input, `failed`
  becomes 1, the cause is classified as a hangup and not as a timeout.
- finish + release from both sides: `leaks --atExit -- tests/test_stream_out`
  reports "0 leaks for 0 total leaked bytes" (wired into `make leaks`).
- ring overflow (8 KiB ring, reader paused): the message that does not fit is
  refused, `failed_enqueues` = 1 with `peer_gone` = 0 and `send_timeout` = 0;
  what the writer had already put on the wire (2048 B) arrives as an exact
  prefix of what was enqueued — a cancelled stream is truncated, never
  scrambled.
- ThreadSanitizer, run once locally and not wired into the Makefile:
  `clang -std=c11 -O1 -g -fsanitize=thread server/stream_out.c
  tests/test_stream_out.c -lpthread` → the suite passes with no warning and no
  race reported. 10 consecutive plain runs: all OK (no flake).

Conclusion: the module and its gate are in. The acceptance gate above is only
partly discharged — the stalled-reader half is proven at unit level, the
"a concurrent normal client's emission lag p95 does not move" and the
"JSON bytes byte-identical to the synchronous path" halves need the scheduler
(S2-2) and stay open.

Next action: with S2-2 — wire `on_delta` to enqueue only, give `stream_out` the
fd from admission to close (close frame and `{"type":"done"}` included), and
re-run the gate end to end.
