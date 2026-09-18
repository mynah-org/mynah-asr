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

Evidence / Conclusion / Next action: with S2-2.
