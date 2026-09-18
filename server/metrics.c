/* The /metrics listener. See metrics.h for why it is a separate port, why it
 * has no thread, and why SO_REUSEPORT is deliberately absent. */
#include "metrics.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------- the buffer */

void mynah_asr_metrics_buf_init(mynah_asr_metrics_buf *b) {
    b->p = NULL;
    b->len = 0;
    b->cap = 0;
    b->oom = 0;
}

void mynah_asr_metrics_buf_free(mynah_asr_metrics_buf *b) {
    free(b->p);
    mynah_asr_metrics_buf_init(b);
}

static int buf_reserve(mynah_asr_metrics_buf *b, size_t extra) {
    if (b->oom) return 0;
    if (b->len + extra + 1 <= b->cap) return 1;
    size_t want = b->cap ? b->cap : 8192;
    while (want < b->len + extra + 1) want *= 2;
    char *np = (char *)realloc(b->p, want);
    if (np == NULL) { b->oom = 1; return 0; }
    b->p = np;
    b->cap = want;
    return 1;
}

void mynah_asr_metrics_addf(mynah_asr_metrics_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char probe[512];
    const int n = vsnprintf(probe, sizeof(probe), fmt, ap);
    va_end(ap);
    if (n < 0) { b->oom = 1; return; }
    if ((size_t)n < sizeof(probe)) {
        if (!buf_reserve(b, (size_t)n)) return;
        memcpy(b->p + b->len, probe, (size_t)n);
        b->len += (size_t)n;
        b->p[b->len] = '\0';
        return;
    }
    if (!buf_reserve(b, (size_t)n)) return;
    va_start(ap, fmt);
    vsnprintf(b->p + b->len, b->cap - b->len, fmt, ap);
    va_end(ap);
    b->len += (size_t)n;
}

void mynah_asr_metrics_label(const char *in, char *out, size_t cap) {
    size_t k = 0;
    for (size_t i = 0; in != NULL && in[i] != '\0' && k + 2 < cap; i++) {
        const char c = in[i];
        if (c == '\\' || c == '"') { out[k++] = '\\'; out[k++] = c; }
        else if (c == '\n' || c == '\r') { out[k++] = '\\'; out[k++] = 'n'; }
        else out[k++] = c;
    }
    out[k] = '\0';
}

/* --------------------------------------------------------- the listener */

int mynah_asr_metrics_listen(const char *bind_addr, int port) {
    if (port <= 0) return -1;
    const char *addr = (bind_addr != NULL && bind_addr[0] != '\0') ? bind_addr
                                                                  : "127.0.0.1";
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "mynah-asr-server: metrics: socket: %s\n", strerror(errno));
        return -1;
    }
    /* SO_REUSEADDR and nothing else. See the header: SO_REUSEPORT would let a
     * second process silently share this port and answer half the scrapes with
     * its own counters under the same labels. */
    const int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (strcmp(addr, "0.0.0.0") == 0 || strcmp(addr, "*") == 0) {
        sa.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, addr, &sa.sin_addr) != 1) {
        fprintf(stderr, "mynah-asr-server: metrics: --metrics-bind '%s' is not an "
                        "IPv4 address\n", addr);
        close(fd);
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        fprintf(stderr, "mynah-asr-server: metrics: bind %s:%d failed: %s\n",
                addr, port, strerror(errno));
        close(fd);
        return -1;
    }
    if (listen(fd, 8) != 0) {
        fprintf(stderr, "mynah-asr-server: metrics: listen %s:%d failed: %s\n",
                addr, port, strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

/* --------------------------------------------------------- the rate limit */

static double mono_s(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* One bucket per process: there is one metrics listener per process by
 * construction (main.c binds at most one, the prefork parent binds at most
 * one), so a per-listener bucket would be the same state with more of it. */
static double g_tokens = MYNAH_ASR_METRICS_BURST;
static double g_last;

static int take_token(double now) {
    if (g_last == 0.0) g_last = now;
    g_tokens += (now - g_last) * MYNAH_ASR_METRICS_RATE;
    if (g_tokens > MYNAH_ASR_METRICS_BURST) g_tokens = MYNAH_ASR_METRICS_BURST;
    g_last = now;
    if (g_tokens < 1.0) return 0;
    g_tokens -= 1.0;
    return 1;
}

/* ------------------------------------------------------- the reply itself
 *
 * Write, FIN, drain, close -- the same sequence as a refusal in prefork.c, and
 * for the same reason: close() with the request still unread makes the kernel
 * send an RST, and the RST discards the response the scraper was about to
 * read. Bounded by MYNAH_ASR_METRICS_LINGER_MS end to end. */
static void reply_and_close(int fd, const char *body, size_t len, int code,
                            const char *status, const char *ctype) {
    char hdr[256];
    const int hn = snprintf(hdr, sizeof(hdr),
                            "HTTP/1.1 %d %s\r\n"
                            "Content-Type: %s\r\n"
                            "Content-Length: %zu\r\n"
                            "Connection: close\r\n\r\n",
                            code, status, ctype, len);
    const double deadline = mono_s() + MYNAH_ASR_METRICS_LINGER_MS / 1000.0;

    struct iovec_like { const char *p; size_t n; } parts[2] = {
        {hdr, hn > 0 ? (size_t)hn : 0}, {body, body != NULL ? len : 0}
    };
    for (int i = 0; i < 2; i++) {
        size_t sent = 0;
        while (sent < parts[i].n && mono_s() < deadline) {
            const ssize_t w = send(fd, parts[i].p + sent, parts[i].n - sent, 0);
            if (w > 0) { sent += (size_t)w; continue; }
            if (w < 0 && errno == EINTR) continue;
            if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                struct pollfd pfd = {.fd = fd, .events = POLLOUT, .revents = 0};
                const double left = (deadline - mono_s()) * 1000.0;
                if (left <= 0.0) break;
                if (poll(&pfd, 1, (int)left) <= 0) break;
                continue;
            }
            break;                       /* the scraper is gone */
        }
        if (sent < parts[i].n) break;
    }
    shutdown(fd, SHUT_WR);
    char scratch[2048];
    size_t drained = 0;
    while (mono_s() < deadline && drained < 64u * 1024u) {
        const ssize_t r = recv(fd, scratch, sizeof(scratch), 0);
        if (r > 0) { drained += (size_t)r; continue; }
        if (r == 0) break;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            struct pollfd pfd = {.fd = fd, .events = POLLIN, .revents = 0};
            const double left = (deadline - mono_s()) * 1000.0;
            if (left <= 0.0) break;
            if (poll(&pfd, 1, (int)left) <= 0) break;
            continue;
        }
        break;
    }
    close(fd);
}

void mynah_asr_metrics_service(int lfd, mynah_asr_metrics_render_fn render, void *ud) {
    if (lfd < 0) return;
    const int fd = accept(lfd, NULL, NULL);
    if (fd < 0) return;

    const int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    const int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    /* The request. Bounded by MYNAH_ASR_METRICS_POLL_MS in TOTAL, not per
     * read: a scraper that connects and then says nothing must not be able to
     * hold the loop that owns this listener. */
    char req[1024];
    size_t got = 0;
    const double until = mono_s() + MYNAH_ASR_METRICS_POLL_MS / 1000.0;
    for (;;) {
        const ssize_t r = recv(fd, req + got, sizeof(req) - 1 - got, 0);
        if (r > 0) {
            got += (size_t)r;
            req[got] = '\0';
            if (strstr(req, "\r\n") != NULL || got >= sizeof(req) - 1) break;
            continue;
        }
        if (r == 0) break;
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) break;
        const double left = (until - mono_s()) * 1000.0;
        if (left <= 0.0) break;
        struct pollfd pfd = {.fd = fd, .events = POLLIN, .revents = 0};
        if (poll(&pfd, 1, (int)left + 1) <= 0) break;
    }
    req[got] = '\0';
    if (got == 0) { close(fd); return; }   /* nothing said: nothing to answer */

    if (strncmp(req, "GET ", 4) != 0 || strstr(req, "/metrics") == NULL) {
        static const char body[] = "not found: this port serves GET /metrics only\n";
        reply_and_close(fd, body, sizeof(body) - 1, 404, "Not Found", "text/plain");
        return;
    }
    if (!take_token(mono_s())) {
        /* 429 WITHOUT rendering: the render is the cost this limit exists to
         * bound, so a limiter that renders first has limited nothing. */
        static const char body[] = "too many scrapes: 5/s, burst 10\n";
        reply_and_close(fd, body, sizeof(body) - 1, 429, "Too Many Requests",
                        "text/plain");
        return;
    }

    mynah_asr_metrics_buf b;
    mynah_asr_metrics_buf_init(&b);
    render(&b, ud);
    if (b.oom || b.p == NULL) {
        static const char body[] = "metrics render failed\n";
        mynah_asr_metrics_buf_free(&b);
        reply_and_close(fd, body, sizeof(body) - 1, 500, "Internal Server Error",
                        "text/plain");
        return;
    }
    reply_and_close(fd, b.p, b.len, 200, "OK",
                    "text/plain; version=0.0.4; charset=utf-8");
    mynah_asr_metrics_buf_free(&b);
}
