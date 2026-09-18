/* Asynchronous bounded output writer. See stream_out.h for the ownership and
 * cancellation contract.
 *
 * The queue is a single byte ring allocated once at start, not a list of
 * malloc'd messages: a stream emits a delta every few milliseconds and the hot
 * path must not touch the allocator at all. The producer only holds the mutex
 * for the memcpy of an already-framed message into the ring.
 *
 * Messages are concatenated into the ring with no boundaries of their own:
 * the caller hands over finished frames, so byte order is the whole contract
 * and the writer never has to know where one message ends. The writer drains
 * the contiguous readable span and writes it straight out of the ring while
 * NOT holding the lock -- safe because the span is handed back to the producer
 * (queued -= span) only after the write has completed, so the region the
 * writer is reading is never one the producer may fill.
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE   /* POLLRDHUP, MSG_NOSIGNAL */
#endif

#include "stream_out.h"

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define STREAM_OUT_DEFAULT_BYTES      (64u * 1024u)
#define STREAM_OUT_MIN_BYTES          4096u
#define STREAM_OUT_MAX_BYTES          (1u << 26)   /* 64 MiB; allocated up front */
#define STREAM_OUT_DEFAULT_TIMEOUT_MS 5000
#define STREAM_OUT_MAX_TIMEOUT_MS     120000

/* EPIPE is the signal the writer wants; the process-wide SIGPIPE handler is the
 * server's job, but where the platform lets a single call opt out, it does. */
#ifdef MSG_NOSIGNAL
#define STREAM_OUT_SEND_FLAGS MSG_NOSIGNAL
#else
#define STREAM_OUT_SEND_FLAGS 0
#endif

struct mynah_asr_stream_out {
    /* Guarded by `mu` for READING from another thread; the writer thread is the
     * only one that mutates it, and only to -1 when it closes. Anyone asking
     * about the socket must hold `mu`, which is what keeps the close and a
     * peer-hangup poll from overlapping. */
    int fd;
    int send_timeout_ms;

    pthread_mutex_t mu;
    pthread_cond_t cv;

    /* Guarded by mu. */
    unsigned char *ring;
    size_t capacity;
    size_t head;              /* the writer's read cursor */
    size_t queued;            /* readable bytes; the tail is head + queued */
    size_t peak_bytes;
    size_t written_bytes;
    size_t enqueued_msgs;
    size_t enqueued_bytes;
    size_t failed_enqueues;
    int producer_done;
    int failure_errno;        /* 0 when the failure was not a socket error */
    int peer_gone;
    int send_timeout;

    /* Written under `mu` like the rest of the failure state, but read without
     * it: the scheduler consults it once per slot per step and must not queue
     * behind a writer that is busy memcpying. */
    _Atomic int failed;

    _Atomic int refs;         /* producer + detached writer */
};

/* ------------------------------------------------------------ configuration */

static size_t stream_out_capacity_bytes(size_t requested) {
    if (requested == 0) return STREAM_OUT_DEFAULT_BYTES;
    if (requested < STREAM_OUT_MIN_BYTES) return STREAM_OUT_MIN_BYTES;
    if (requested > STREAM_OUT_MAX_BYTES) return STREAM_OUT_MAX_BYTES;
    return requested;
}

static int stream_out_timeout_ms(int requested) {
    if (requested <= 0) return STREAM_OUT_DEFAULT_TIMEOUT_MS;
    if (requested > STREAM_OUT_MAX_TIMEOUT_MS) return STREAM_OUT_MAX_TIMEOUT_MS;
    return requested;
}

/* ------------------------------------------------------------------ socket */

/* Writes everything or reports the peer as gone. An SO_SNDTIMEO expiry arrives
 * as EAGAIN/EWOULDBLOCK: that is a client which has stopped reading, not a
 * transient condition to retry, so it ends the stream like any other failure.
 * Retrying would reinstate exactly the stall this module removes. */
static int write_all_or_gone(int fd, const void *data, size_t len) {
    const char *p = (const char *)data;
    while (len > 0) {
        const ssize_t n = send(fd, p, len, STREAM_OUT_SEND_FLAGS);
        if (n > 0) {
            p += (size_t)n;
            len -= (size_t)n;
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n == 0) errno = EPIPE;
        return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------- lifetime */

static void stream_out_destroy(mynah_asr_stream_out *o) {
    pthread_cond_destroy(&o->cv);
    pthread_mutex_destroy(&o->mu);
    free(o->ring);
    free(o);
}

void mynah_asr_stream_out_release(mynah_asr_stream_out *o) {
    if (o == NULL) return;
    if (atomic_fetch_sub(&o->refs, 1) != 1) return;
    stream_out_destroy(o);
}

/* Which recorded failures mean "the peer went away" as opposed to "the peer is
 * still there and not keeping up". The distinction is the whole point of
 * counting them apart: a hangup frees a slot nobody wanted, a slow reader is a
 * capacity problem, and an operator who sees them as one number cannot tell a
 * flaky client population from an undersized machine.
 *
 * Deliberately NOT in this set: errno 0, which is the ring-overflow
 * cancellation (a reader too slow, socket still open), and EAGAIN, which is an
 * SO_SNDTIMEO expiry -- same thing, arriving by a different route. */
static int failure_is_hangup(int err) {
    switch (err) {
        case EPIPE:
        case ECONNRESET:
        case ENOTCONN:
#ifdef ESHUTDOWN
        case ESHUTDOWN:
#endif
            return 1;
        default:
            return 0;
    }
}

/* Abandon the stream: the producer's next enqueue fails and the writer stops
 * without draining, because for a framed protocol an incomplete tail is how the
 * peer learns the stream was cut. */
static void mark_failed_locked(mynah_asr_stream_out *o, int err) {
    if (!atomic_load_explicit(&o->failed, memory_order_relaxed)) {
        o->failure_errno = err;
        if (failure_is_hangup(err)) o->peer_gone = 1;
        if (err == EAGAIN || err == EWOULDBLOCK) o->send_timeout = 1;
        atomic_store_explicit(&o->failed, 1, memory_order_release);
    }
    o->producer_done = 1;
    pthread_cond_broadcast(&o->cv);
}

/* ------------------------------------------------------------- the writer */

static void stream_out_name_thread(int fd) {
    /* One writer per stream, so the descriptor is the only identity available
     * and also the useful one: it is what `lsof` and the server's own logs
     * name. Set from this thread because that is the only spelling both
     * platforms share. */
    char name[16];
    snprintf(name, sizeof(name), "mynah-out%d", fd);
#if defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#else
    (void)name;
#endif
}

static void *stream_out_writer_main(void *arg) {
    mynah_asr_stream_out *o = (mynah_asr_stream_out *)arg;
    stream_out_name_thread(o->fd);

    for (;;) {
        pthread_mutex_lock(&o->mu);
        while (o->queued == 0 && !o->producer_done &&
               !atomic_load_explicit(&o->failed, memory_order_relaxed)) {
            pthread_cond_wait(&o->cv, &o->mu);
        }
        if (atomic_load_explicit(&o->failed, memory_order_relaxed)) {
            pthread_mutex_unlock(&o->mu);
            break;
        }
        if (o->queued == 0) { pthread_mutex_unlock(&o->mu); break; }
        /* Only the span up to the ring's end; a wrap becomes a second write. */
        size_t span = o->queued;
        if (span > o->capacity - o->head) span = o->capacity - o->head;
        const size_t offset = o->head;
        pthread_mutex_unlock(&o->mu);

        if (write_all_or_gone(o->fd, o->ring + offset, span) != 0) {
            const int err = errno;
            pthread_mutex_lock(&o->mu);
            mark_failed_locked(o, err);
            pthread_mutex_unlock(&o->mu);
            break;
        }

        pthread_mutex_lock(&o->mu);
        o->head = (o->head + span) % o->capacity;
        o->queued -= span;
        o->written_bytes += span;
        pthread_mutex_unlock(&o->mu);
    }

    /* Closed under the mutex, and the field retired to -1 in the same critical
     * section. mynah_asr_stream_out_peer_gone() may be polling this descriptor
     * from the scheduler thread; it holds `mu` while it does, so the close
     * either happens entirely before its poll (and it sees -1 and reports gone)
     * or entirely after (and its poll ran on a descriptor that was still ours).
     * Without the lock the number could be reissued by accept() between the
     * two, and the poll would be asking about a stranger's connection. */
    pthread_mutex_lock(&o->mu);
    const int doomed = o->fd;
    o->fd = -1;
    if (doomed >= 0) close(doomed);
    pthread_mutex_unlock(&o->mu);

    /* A stream that stops early must never be silent: a truncated message
     * sequence is otherwise indistinguishable from a short utterance. */
    if (atomic_load(&o->failed)) {
        mynah_asr_stream_out_stats st;
        mynah_asr_stream_out_get_stats(o, &st);
        pthread_mutex_lock(&o->mu);
        const int err = o->failure_errno;
        pthread_mutex_unlock(&o->mu);
        fprintf(stderr,
                "stream aborted after %zu bytes: %s "
                "(ring %zu peak, msgs=%zu, refused=%zu)\n",
                st.written_bytes,
                err != 0 ? strerror(err) : "client stopped reading",
                st.peak_bytes, st.enqueued_msgs, st.failed_enqueues);
    }

    mynah_asr_stream_out_release(o);
    return NULL;
}

/* ----------------------------------------------------------- the producer */

mynah_asr_stream_out *mynah_asr_stream_out_start(int fd, size_t ring_bytes,
                                                 int send_timeout_ms) {
    if (fd < 0) return NULL;
    mynah_asr_stream_out *o = (mynah_asr_stream_out *)calloc(1, sizeof(*o));
    if (o == NULL) return NULL;

    o->fd = fd;
    o->send_timeout_ms = stream_out_timeout_ms(send_timeout_ms);
    o->capacity = stream_out_capacity_bytes(ring_bytes);
    o->ring = (unsigned char *)malloc(o->capacity);
    if (o->ring == NULL) { free(o); return NULL; }

    if (pthread_mutex_init(&o->mu, NULL) != 0) {
        free(o->ring); free(o);
        return NULL;
    }
    if (pthread_cond_init(&o->cv, NULL) != 0) {
        pthread_mutex_destroy(&o->mu);
        free(o->ring); free(o);
        return NULL;
    }
    atomic_init(&o->failed, 0);
    atomic_init(&o->refs, 2);   /* producer + writer */

    struct timeval tv;
    tv.tv_sec = o->send_timeout_ms / 1000;
    tv.tv_usec = (o->send_timeout_ms % 1000) * 1000;
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#ifdef SO_NOSIGPIPE
    { const int on = 1; (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on)); }
#endif

    pthread_t thread;
    if (pthread_create(&thread, NULL, stream_out_writer_main, o) != 0) {
        /* The fd was never handed over: destroy never closes it, so it goes
         * back to the caller intact. */
        stream_out_destroy(o);
        return NULL;
    }
    pthread_detach(thread);
    return o;
}

int mynah_asr_stream_out_enqueue(mynah_asr_stream_out *o, const void *msg, size_t len) {
    if (o == NULL || msg == NULL || len == 0) return -1;

    pthread_mutex_lock(&o->mu);
    if (atomic_load_explicit(&o->failed, memory_order_relaxed) || o->producer_done) {
        ++o->failed_enqueues;
        pthread_mutex_unlock(&o->mu);
        return -1;
    }
    if (len > o->capacity - o->queued) {
        /* Backpressure is cancellation: a reader too slow to keep up loses its
         * own stream rather than holding the scheduler hostage. */
        ++o->failed_enqueues;
        mark_failed_locked(o, 0);
        pthread_mutex_unlock(&o->mu);
        return -1;
    }

    const size_t tail = (o->head + o->queued) % o->capacity;
    const size_t first = len < o->capacity - tail ? len : o->capacity - tail;
    memcpy(o->ring + tail, msg, first);
    if (first < len) {
        memcpy(o->ring, (const unsigned char *)msg + first, len - first);
    }
    o->queued += len;
    if (o->queued > o->peak_bytes) o->peak_bytes = o->queued;
    ++o->enqueued_msgs;
    o->enqueued_bytes += len;
    pthread_cond_signal(&o->cv);
    pthread_mutex_unlock(&o->mu);
    return 0;
}

int mynah_asr_stream_out_failed(const mynah_asr_stream_out *o) {
    if (o == NULL) return 1;
    return atomic_load_explicit(&o->failed, memory_order_acquire);
}

int mynah_asr_stream_out_peer_gone(mynah_asr_stream_out *o) {
    if (o == NULL) return 1;

    pthread_mutex_lock(&o->mu);
    /* Already failed. The answer comes from WHY, not from the socket: by the
     * time a write has returned EPIPE the descriptor may already be closed, and
     * in practice the writer usually discovers the hangup before this poll does
     * -- it is the thread actually touching the socket. Reporting only what the
     * poll caught would credit a fraction of the disconnects and file the rest
     * under "timed out". */
    if (atomic_load_explicit(&o->failed, memory_order_relaxed)) {
        const int gone = o->peer_gone;
        pthread_mutex_unlock(&o->mu);
        return gone;
    }
    const int fd = o->fd;
    if (fd < 0) { pthread_mutex_unlock(&o->mu); return 1; }

    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
#ifdef POLLRDHUP
    pfd.events |= POLLRDHUP;   /* Linux: the half-close, without a read */
#endif
    pfd.revents = 0;

    int gone = 0;
    const int ready = poll(&pfd, 1, 0);   /* zero timeout: never blocks on mu */
    if (ready < 0) {
        /* EINTR and EAGAIN are "ask again"; anything else means this descriptor
         * can no longer be polled, which is a dead stream. */
        gone = (errno != EINTR && errno != EAGAIN);
    } else if (ready > 0) {
        if ((pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
            gone = 1;
#ifdef POLLRDHUP
        } else if ((pfd.revents & POLLRDHUP) != 0) {
            gone = 1;
#endif
        } else if ((pfd.revents & POLLIN) != 0) {
            /* The portable half. Readable means either the peer sent something
             * (a control message the ingest side will read) or it closed. Only
             * a zero-length peek tells the two apart, and it cannot block
             * because poll() just said the socket is readable. */
            char probe;
            const ssize_t n = recv(fd, &probe, 1, MSG_PEEK);
            if (n == 0) {
                gone = 1;
            } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
                       errno != EINTR) {
                gone = 1;
            }
        }
    }

    /* Failing the stream here is what frees the socket promptly: the writer is
     * parked on the condvar waiting for messages nobody will read, and this
     * wakes it to close and go. The decoder is untouched -- it learns at its
     * own step boundary, through the server's cancellation callback. */
    if (gone) mark_failed_locked(o, EPIPE);
    pthread_mutex_unlock(&o->mu);
    return gone;
}

void mynah_asr_stream_out_finish(mynah_asr_stream_out *o) {
    if (o == NULL) return;
    pthread_mutex_lock(&o->mu);
    o->producer_done = 1;
    pthread_cond_broadcast(&o->cv);
    pthread_mutex_unlock(&o->mu);
}

void mynah_asr_stream_out_get_stats(const mynah_asr_stream_out *o,
                                    mynah_asr_stream_out_stats *s) {
    if (o == NULL || s == NULL) return;
    /* const in the signature is the caller's contract (reading stats changes
     * nothing observable); the counters still live under the mutex, so the
     * snapshot is consistent rather than torn. */
    mynah_asr_stream_out *m = (mynah_asr_stream_out *)(uintptr_t)o;
    pthread_mutex_lock(&m->mu);
    s->enqueued_msgs   = m->enqueued_msgs;
    s->enqueued_bytes  = m->enqueued_bytes;
    s->written_bytes   = m->written_bytes;
    s->failed_enqueues = m->failed_enqueues;
    s->peak_bytes      = m->peak_bytes;
    s->failed          = atomic_load_explicit(&m->failed, memory_order_relaxed);
    s->peer_gone       = m->peer_gone;
    s->send_timeout    = m->send_timeout;
    pthread_mutex_unlock(&m->mu);
}
