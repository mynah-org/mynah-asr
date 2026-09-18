/* Self-test of the asynchronous bounded output writer (no model required — it
 * runs in CI too).
 *
 * What it guards is the contract the scheduler depends on, not the arithmetic:
 *
 *   1. the bytes come out of the socket exactly as they went in, in order
 *      (the ring wraps, and a wrapped message must not be reordered);
 *   2. a reader that stops reading kills its OWN stream within the send
 *      timeout and never blocks the producer — the whole reason this module
 *      exists (66.5 s → 2.6 s for the next client, measured in mynah-tts);
 *   3. a reader that closes is noticed by the peer-gone poll, not only by the
 *      next failed write;
 *   4. finish + release from both sides frees everything (run under
 *      `leaks --atExit`: `make leaks`);
 *   5. an overflow refuses the message that does not fit, marks the stream
 *      failed, and leaves what was already written intact.
 *
 * Exit: 0 ok, 1 fail. */
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "../server/stream_out.h"

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("stream_out FAIL: %s\n", msg); failures = 1; } \
    else printf("stream_out ok:   %s\n", msg); } while (0)

/* ------------------------------------------------------------------ helpers */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void sleep_ms(int ms) {
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

/* Deterministic payload: message i, byte j. */
static unsigned char msg_byte(int i, size_t j) {
    return (unsigned char)((unsigned)i * 131u + (unsigned)j * 7u + 11u);
}

static size_t msg_len(int i) { return 16u + (size_t)((unsigned)i % 256u); }

static void fill_msg(unsigned char *buf, int i, size_t len) {
    for (size_t j = 0; j < len; ++j) buf[j] = msg_byte(i, j);
}

/* A socketpair with deliberately small buffers, so "the peer stopped reading"
 * is reachable in a test instead of megabytes away. */
static int pair_make(int sv[2], int bufbytes) {
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return -1;
    if (bufbytes > 0) {
        for (int k = 0; k < 2; ++k) {
            (void)setsockopt(sv[k], SOL_SOCKET, SO_SNDBUF, &bufbytes, sizeof(bufbytes));
            (void)setsockopt(sv[k], SOL_SOCKET, SO_RCVBUF, &bufbytes, sizeof(bufbytes));
        }
    }
    return 0;
}

/* Reads until EOF or until `cap` bytes have arrived; `idle_ms` without a byte
 * ends the read too, so a stuck stream fails the test instead of hanging it. */
static size_t read_until_eof(int fd, unsigned char *buf, size_t cap, int idle_ms) {
    size_t got = 0;
    for (;;) {
        struct pollfd pfd = { fd, POLLIN, 0 };
        const int r = poll(&pfd, 1, idle_ms);
        if (r <= 0) break;                    /* idle: nothing more is coming */
        if (got == cap) break;
        const ssize_t n = read(fd, buf + got, cap - got);
        if (n <= 0) break;                    /* 0 = the writer closed the fd */
        got += (size_t)n;
    }
    return got;
}

/* Waits until the ring has room for `len` (queued == enqueued - written). The
 * producer in production never waits — it loses the stream instead — but a test
 * about ordering must not turn into a test about overflow, so it paces itself
 * here rather than sizing the ring past the payload (a ring bigger than the
 * whole payload never wraps, and the wrap is the interesting part). */
static int wait_for_room(const mynah_asr_stream_out *o, size_t len, size_t capacity) {
    const double t0 = now_ms();
    for (;;) {
        mynah_asr_stream_out_stats st;
        mynah_asr_stream_out_get_stats(o, &st);
        if (st.failed) return 0;
        if (st.enqueued_bytes - st.written_bytes + len <= capacity) return 1;
        if (now_ms() - t0 > 5000.0) return 0;
        sleep_ms(1);
    }
}

typedef struct {
    int fd;
    unsigned char *buf;
    size_t cap;
    size_t got;
} reader_arg;

static void *reader_main(void *arg) {
    reader_arg *r = (reader_arg *)arg;
    r->got = read_until_eof(r->fd, r->buf, r->cap, 5000);
    return NULL;
}

/* --------------------------------------------------- 1. byte identity/order */

#define IDENTITY_MSGS 1000

static void test_identity(void) {
    int sv[2];
    if (pair_make(sv, 0) != 0) { CHECK(0, "socketpair"); return; }

    size_t total = 0;
    for (int i = 0; i < IDENTITY_MSGS; ++i) total += msg_len(i);
    unsigned char *expect = (unsigned char *)malloc(total);
    unsigned char *actual = (unsigned char *)malloc(total + 64);
    if (expect == NULL || actual == NULL) { CHECK(0, "alloc"); return; }
    size_t off = 0;
    for (int i = 0; i < IDENTITY_MSGS; ++i) {
        fill_msg(expect + off, i, msg_len(i));
        off += msg_len(i);
    }

    /* A 16 KiB ring against ~140 KiB of payload: the ring wraps about nine
     * times, so messages really do straddle the end of the buffer and the
     * two-memcpy enqueue and the two-span write are both exercised. */
    const size_t ring = 16u * 1024u;
    mynah_asr_stream_out *o = mynah_asr_stream_out_start(sv[0], ring, 2000);
    if (o == NULL) { CHECK(0, "start"); return; }

    reader_arg ra = { sv[1], actual, total + 64, 0 };
    pthread_t th;
    pthread_create(&th, NULL, reader_main, &ra);

    int queued = 0;
    off = 0;
    for (int i = 0; i < IDENTITY_MSGS; ++i) {
        if (!wait_for_room(o, msg_len(i), ring)) break;
        if (mynah_asr_stream_out_enqueue(o, expect + off, msg_len(i)) == 0) ++queued;
        off += msg_len(i);
    }
    mynah_asr_stream_out_finish(o);

    pthread_join(th, NULL);

    mynah_asr_stream_out_stats st;
    mynah_asr_stream_out_get_stats(o, &st);
    mynah_asr_stream_out_release(o);
    close(sv[1]);

    CHECK(queued == IDENTITY_MSGS, "1000 messages accepted");
    CHECK(ra.got == total, "every byte arrived");
    CHECK(ra.got == total && memcmp(expect, actual, total) == 0,
          "bytes identical and in order");
    CHECK(st.enqueued_msgs == IDENTITY_MSGS && st.enqueued_bytes == total,
          "enqueue counters match");
    CHECK(st.written_bytes == total, "written_bytes == enqueued_bytes");
    CHECK(st.failed == 0 && st.failed_enqueues == 0, "clean stream");
    CHECK(st.peak_bytes <= ring && total > ring, "the ring wrapped (payload > ring)");
    printf("        %zu bytes in %d messages through a %zu B ring, peak %zu B\n",
           total, IDENTITY_MSGS, ring, st.peak_bytes);

    free(expect);
    free(actual);
}

/* ------------------------------------------- 2. a reader that stops reading */

static void test_stalled_reader(void) {
    int sv[2];
    if (pair_make(sv, 4096) != 0) { CHECK(0, "socketpair"); return; }

    /* Ring big enough that nothing overflows: the failure under test is the
     * SO_SNDTIMEO expiry on the writer thread, not the ring ceiling. */
    mynah_asr_stream_out *o = mynah_asr_stream_out_start(sv[0], 1024u * 1024u, 500);
    if (o == NULL) { CHECK(0, "start"); return; }

    unsigned char chunk[4096];
    memset(chunk, 0xa5, sizeof(chunk));

    /* sv[1] is never read from. The socket buffers fill, the writer parks in
     * send(), and the producer must not notice. */
    double worst_ms = 0.0;
    int accepted = 0;
    for (int i = 0; i < 64; ++i) {
        const double t0 = now_ms();
        const int rc = mynah_asr_stream_out_enqueue(o, chunk, sizeof(chunk));
        const double dt = now_ms() - t0;
        if (dt > worst_ms) worst_ms = dt;
        if (rc == 0) ++accepted;
    }
    CHECK(accepted == 64, "producer accepted 256 KiB against a stalled reader");
    CHECK(worst_ms < 50.0, "no enqueue blocked (worst < 50 ms)");
    printf("        worst enqueue %.3f ms\n", worst_ms);

    /* Bound: 4x the timeout. Measured on macOS it lands near 2x, because a
     * send() that manages a partial transfer before SO_SNDTIMEO expires returns
     * that partial count instead of an error, and the next send starts the
     * timer again. The guarantee is "a small multiple of the timeout", never
     * "until the peer comes back". */
    const double t0 = now_ms();
    double failed_after = -1.0;
    while (now_ms() - t0 < 5000.0) {
        if (mynah_asr_stream_out_failed(o)) { failed_after = now_ms() - t0; break; }
        sleep_ms(5);
    }
    CHECK(failed_after >= 0.0 && failed_after < 2000.0,
          "stream failed within 4x the send timeout");
    printf("        failed after %.0f ms (SO_SNDTIMEO 500 ms)\n", failed_after);

    const double t1 = now_ms();
    const int rc = mynah_asr_stream_out_enqueue(o, chunk, sizeof(chunk));
    const double dt = now_ms() - t1;
    CHECK(rc == -1, "enqueue refused once failed");
    CHECK(dt < 50.0, "the refusal is immediate");

    mynah_asr_stream_out_stats st;
    mynah_asr_stream_out_get_stats(o, &st);
    CHECK(st.failed == 1 && st.failed_enqueues == 1, "failed_enqueues counted (1)");
    CHECK(st.send_timeout == 1, "failure classified as a send timeout");
    CHECK(st.peer_gone == 0, "a stalled reader is not a hangup");

    mynah_asr_stream_out_finish(o);
    mynah_asr_stream_out_release(o);
    close(sv[1]);
}

/* -------------------------------------------------- 3. a reader that closes */

static void test_peer_close(void) {
    int sv[2];
    if (pair_make(sv, 0) != 0) { CHECK(0, "socketpair"); return; }

    mynah_asr_stream_out *o = mynah_asr_stream_out_start(sv[0], 0, 1000);
    if (o == NULL) { CHECK(0, "start"); return; }

    close(sv[1]);

    int gone = 0;
    const double t0 = now_ms();
    while (now_ms() - t0 < 2000.0) {
        if (mynah_asr_stream_out_peer_gone(o)) { gone = 1; break; }
        sleep_ms(5);
    }
    CHECK(gone == 1, "peer_gone detected without consuming input");
    CHECK(mynah_asr_stream_out_failed(o) == 1, "peer hangup fails the stream");

    mynah_asr_stream_out_stats st;
    mynah_asr_stream_out_get_stats(o, &st);
    CHECK(st.peer_gone == 1 && st.send_timeout == 0, "failure classified as a hangup");
    CHECK(mynah_asr_stream_out_enqueue(o, "x", 1) == -1, "enqueue refused after hangup");

    mynah_asr_stream_out_finish(o);
    mynah_asr_stream_out_release(o);
}

/* ------------------------------------------- 4. finish + release both sides */

static void test_finish_release(void) {
    int sv[2];
    if (pair_make(sv, 0) != 0) { CHECK(0, "socketpair"); return; }

    mynah_asr_stream_out *o = mynah_asr_stream_out_start(sv[0], 8192, 1000);
    if (o == NULL) { CHECK(0, "start"); return; }

    unsigned char msg[64];
    size_t sent = 0;
    for (int i = 0; i < 10; ++i) {
        fill_msg(msg, i, sizeof(msg));
        if (mynah_asr_stream_out_enqueue(o, msg, sizeof(msg)) == 0) sent += sizeof(msg);
    }
    mynah_asr_stream_out_finish(o);
    mynah_asr_stream_out_release(o);   /* the producer is done with it here */

    /* EOF is the only synchronisation point the contract offers: the writer
     * closes the fd as its last act before dropping its own reference. The
     * short sleep covers the few instructions between that close and the free,
     * so `leaks --atExit` sees a settled process rather than a race. */
    unsigned char buf[1024];
    const size_t got = read_until_eof(sv[1], buf, sizeof(buf), 2000);
    close(sv[1]);
    sleep_ms(100);

    CHECK(got == sent, "drained everything before closing on finish");
}

/* ------------------------------------------------------- 5. ring overflow */

static void test_overflow(void) {
    int sv[2];
    if (pair_make(sv, 4096) != 0) { CHECK(0, "socketpair"); return; }

    const size_t ring = 8192;
    mynah_asr_stream_out *o = mynah_asr_stream_out_start(sv[0], ring, 5000);
    if (o == NULL) { CHECK(0, "start"); return; }

    unsigned char msg[512];
    static unsigned char expect[512 * 200];
    for (int i = 0; i < 200; ++i) fill_msg(expect + (size_t)i * sizeof(msg), i, sizeof(msg));

    /* First a handful the writer can actually place in the socket buffer, with
     * a pause so it does: without it the producer wins the race, fills the ring
     * before the writer is ever scheduled, and "what was written before the
     * overflow" is legitimately nothing. */
    int accepted = 0, refused_at = -1;
    for (int i = 0; i < 4; ++i) {
        if (mynah_asr_stream_out_enqueue(o, expect + (size_t)i * sizeof(msg),
                                         sizeof(msg)) == 0) ++accepted;
    }
    sleep_ms(100);
    const size_t drained = (size_t)accepted * sizeof(msg);

    for (int i = accepted; i < 200; ++i) {
        if (mynah_asr_stream_out_enqueue(o, expect + (size_t)i * sizeof(msg),
                                         sizeof(msg)) == 0) {
            ++accepted;
        } else {
            refused_at = i;
            break;
        }
    }
    CHECK(refused_at > 0 && refused_at < 200, "the message that does not fit is refused");
    CHECK(mynah_asr_stream_out_failed(o) == 1, "overflow fails the stream");

    mynah_asr_stream_out_stats st;
    mynah_asr_stream_out_get_stats(o, &st);
    CHECK(st.failed_enqueues == 1 && st.peer_gone == 0 && st.send_timeout == 0,
          "overflow counted as its own cause");
    printf("        %d of 200 messages accepted into a %zu B ring, peak %zu B\n",
           accepted, ring, st.peak_bytes);

    /* Now read: what the writer had already handed to the socket must be intact
     * and in order. A cancelled stream is truncated, never scrambled. */
    static unsigned char got_buf[512 * 200];
    const size_t got = read_until_eof(sv[1], got_buf, sizeof(got_buf), 500);
    CHECK(got >= drained, "the messages written before the overflow arrived");
    CHECK(got <= (size_t)accepted * sizeof(msg) &&
          memcmp(expect, got_buf, got) == 0,
          "what arrived is an exact prefix of what was enqueued");
    printf("        %zu B recovered of %zu B enqueued\n",
           got, (size_t)accepted * sizeof(msg));

    mynah_asr_stream_out_finish(o);
    mynah_asr_stream_out_release(o);
    close(sv[1]);
}

int main(void) {
    /* The writer relies on EPIPE, never on the signal. */
    signal(SIGPIPE, SIG_IGN);

    test_identity();
    test_stalled_reader();
    test_peer_close();
    test_finish_release();
    test_overflow();

    printf("test_stream_out: %s\n", failures ? "FAIL" : "OK");
    return failures;
}
