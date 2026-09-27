/* Asynchronous bounded output writer for one streaming connection.
 *
 * Why this exists, measured in the sibling repo (mynah-tts, same design): the
 * engine used to write to the client socket from the thread that was running
 * the model, inside its critical section. One client that stopped reading
 * stalled that write until SO_SNDTIMEO expired and took every other request in
 * the process with it -- the next client was served after 66.5 s. With socket
 * I/O moved onto a thread of its own, and the engine thread only ever copying
 * bytes into a bounded queue, the same experiment served the next client after
 * 2.6 s. Nothing about the model changed; the scheduler simply stopped waiting
 * on a socket.
 *
 * Here the payload is not audio but opaque, already-framed messages: the caller
 * builds the WebSocket frame (or the HTTP bytes) and hands over the finished
 * bytes. This module knows nothing about framing, JSON or protocol version --
 * it moves bytes off the scheduler thread and reports when the peer stopped
 * deserving them.
 *
 * The contract, stated plainly because it is the part that bites:
 *
 *   - mynah_asr_stream_out_start() TAKES OWNERSHIP of the fd on success. The
 *     writer thread writes the queued messages and then closes the fd. Nobody
 *     else may write to it or close it. On failure the fd is untouched and
 *     still belongs to the caller.
 *   - The writer thread is detached, so it can outlive the request that
 *     started it. Lifetime is an atomic refcount of two: the producer drops
 *     its reference with mynah_asr_stream_out_release(), the writer drops its
 *     own when it is finished, and the last one out frees everything.
 *   - Backpressure is cancellation, not blocking. When the message does not
 *     fit, the enqueue fails and the stream is marked failed; the producer
 *     discovers this (from the return value or from _failed()) and abandons
 *     the stream. A reader too slow to keep up loses its own stream, never the
 *     process.
 *   - SIGPIPE must be ignored process-wide; the writer relies on EPIPE.
 *
 * The ring defaults to 64 KiB and the send timeout to 5000 ms. For ASR a
 * server->client message is a JSON delta of a few hundred bytes, so 64 KiB is
 * hundreds of messages of backlog: an overflow is not a burst, it is a reader
 * that has stopped.
 */
#ifndef MYNAH_ASR_SERVER_STREAM_OUT_H
#define MYNAH_ASR_SERVER_STREAM_OUT_H

#include <stddef.h>

typedef struct mynah_asr_stream_out mynah_asr_stream_out;

/* Counters for /health and the tests. */
typedef struct {
    size_t enqueued_msgs;    /* messages accepted into the ring */
    size_t enqueued_bytes;   /* their bytes */
    size_t written_bytes;    /* bytes actually written to the socket */
    size_t failed_enqueues;  /* refused: stream already gone, or did not fit */
    size_t peak_bytes;       /* deepest the ring ever got */
    int failed;              /* the stream was abandoned */
    int peer_gone;           /* ...because the peer hung up */
    int send_timeout;        /* ...because a send hit SO_SNDTIMEO */
} mynah_asr_stream_out_stats;

/* Starts the detached writer and hands it `fd`. `ring_bytes` is the queue
 * capacity, allocated once here (0 = 64 KiB default, clamped to [4 KiB, 64 MiB]).
 * `send_timeout_ms` is the SO_SNDTIMEO set on the fd (0 = 5000 ms).
 * Returns NULL without touching `fd` if the writer could not be started. */
mynah_asr_stream_out *mynah_asr_stream_out_start(int fd, size_t ring_bytes,
                                                 int send_timeout_ms);

/* Enqueues one already-framed message: the bytes are copied, so `msg` may be
 * reused immediately. Returns 0 when queued, -1 when the stream is failed or
 * cancelled, or when the message does not fit in what is left of the ring --
 * and then the stream is marked failed, because a backlog this deep means the
 * reader is gone and continuing would only cost the scheduler more copies.
 * Never blocks on the socket. */
int mynah_asr_stream_out_enqueue(mynah_asr_stream_out *o, const void *msg, size_t len);

/* 1 once the stream has been abandoned: peer hangup, socket error, send
 * timeout, or ring overflow. A lock-free atomic read, cheap enough for the
 * scheduler to consult at every step. */
/* When the writer's first send() to this socket completed, on CLOCK_MONOTONIC,
 * or 0.0 before it has. The last instant on the path to the client's first
 * visible text that this process can see: framing and the output ring are
 * behind it, the network and the client's read loop are in front. Diagnostic
 * only -- nothing in the serving path reads it. */
double mynah_asr_stream_out_first_send(const mynah_asr_stream_out *o);

int mynah_asr_stream_out_failed(const mynah_asr_stream_out *o);

/* 1 once the peer has hung up. Answers from the socket itself rather than from
 * a failed write, so a client that goes away during a long decode is noticed
 * while it happens instead of at the next message.
 *
 * It lives HERE, and not in the server's cancellation callback, because this
 * module owns the descriptor: the writer thread closes it and nothing outside
 * may name it. A poller elsewhere would be racing that close, and the number it
 * polled could by then belong to another client. The check runs under the same
 * mutex the writer takes to close, so the descriptor is guaranteed live for the
 * syscall.
 *
 * Detection marks the stream failed, so the writer wakes, stops and releases
 * the socket without waiting for the producer. The poll itself never blocks
 * (zero timeout); it is meant to be called once per scheduler step.
 *
 * Portability, stated rather than promised: POLLRDHUP is Linux-only, and
 * POLLHUP on a half-closed TCP socket is not portably raised. The fallback is
 * POLLIN plus a zero-length MSG_PEEK, which is what actually detects a closed
 * peer on macOS and BSD. A client that stops READING but keeps the socket open
 * is invisible to both, by construction -- that case is the ring's ceiling and
 * the send timeout, not this function's. */
int mynah_asr_stream_out_peer_gone(mynah_asr_stream_out *o);

/* The same, with `hard_only` set: only a reset or a socket error counts, never
 * a half-close. For a stream flushing its tail, whose client may legally have
 * shut its sending side and still be waiting for `done`. */
int mynah_asr_stream_out_peer_gone_ex(mynah_asr_stream_out *o, int hard_only);

/* No more messages will be enqueued: let the writer drain and close.
 * Non-blocking; the writer may still be draining when this returns. */
void mynah_asr_stream_out_finish(mynah_asr_stream_out *o);

/* Drops the producer's reference; the last reference frees. `o` must not be
 * touched afterwards. Never blocks on the writer. */
void mynah_asr_stream_out_release(mynah_asr_stream_out *o);

/* Snapshot of the counters. Safe at any time; the writer may still be
 * draining. */
void mynah_asr_stream_out_get_stats(const mynah_asr_stream_out *o,
                                    mynah_asr_stream_out_stats *s);

#endif

/* Who holds this ring's mutex, where it was taken and for how long (-1 when
 * free). Diagnostic only: three relaxed loads, no lock taken to ask. */
void mynah_asr_stream_out_owner(mynah_asr_stream_out *o, unsigned long *owner,
                                double *held_s, const char **where);
