/* metrics.h — the /metrics endpoint: its own listener, its own port, and no
 * thread of its own.
 *
 * WHY A SEPARATE PORT.  The service port carries audio and answers clients; a
 * scrape must never queue behind a 200 MB upload, must never be reachable from
 * wherever the audio clients are, and must never be the thing that keeps the
 * listen backlog busy.  Default bind is 127.0.0.1 and the endpoint is OFF
 * unless --metrics-port is given: an observability surface that is on by
 * default is an attack surface nobody asked for.
 *
 * WHY NO SO_REUSEPORT.  Two processes silently sharing a metrics port would
 * hand a scraper whichever of them the kernel felt like, and the numbers would
 * be one process's counters labelled as the fleet's.  A second bind on a live
 * metrics port MUST fail, loudly, with the port named -- that is the whole
 * point of the option being absent.  SO_REUSEADDR *is* set, and only that:
 * it lets a restart re-bind over the TIME_WAIT of the scrapes the previous run
 * answered, and on both Linux and BSD it does NOT permit a second live
 * listener on the same address and port.
 *
 * WHY NO THREAD.  The page is rendered ON REQUEST from counters the process
 * already keeps.  There is nothing to collect in the background, so a
 * collector thread would only add a clock and a lock.  The owner of the
 * process's accept loop -- the single-process server's, or the prefork
 * parent's router -- puts this listener in its poll set and calls
 * mynah_asr_metrics_service() when it is readable.  Everything the call does
 * is bounded: MYNAH_ASR_METRICS_POLL_MS for the request bytes and
 * MYNAH_ASR_METRICS_LINGER_MS for the write, the FIN and the drain.  A scrape
 * therefore costs the accept loop a bounded slice, never a wait.
 *
 * RATE LIMIT.  A token bucket, 5 scrapes per second with a burst of 10.  Over
 * that the answer is 429 and the page is NOT rendered: rendering is the cost,
 * so a limiter that renders first limits nothing.
 */
#ifndef MYNAH_ASR_SERVER_METRICS_H
#define MYNAH_ASR_SERVER_METRICS_H

#include <stddef.h>

/* Budgets, named rather than sprinkled: every one of them bounds what one
 * scraper can take from the loop that owns the listener. */
#define MYNAH_ASR_METRICS_POLL_MS    2     /* waiting for the request bytes   */
#define MYNAH_ASR_METRICS_LINGER_MS  200   /* write + FIN + drain, in total   */
#define MYNAH_ASR_METRICS_RATE       5.0   /* scrapes per second              */
#define MYNAH_ASR_METRICS_BURST      10.0  /* bucket depth                    */

/* A growable text page.  One allocation that doubles; the render path runs on
 * a scrape, never on a step, so an allocation here costs a step nothing. */
typedef struct {
    char  *p;
    size_t len, cap;
    int    oom;      /* set once an append failed: the response is then 500 */
} mynah_asr_metrics_buf;

void mynah_asr_metrics_buf_init(mynah_asr_metrics_buf *b);
void mynah_asr_metrics_buf_free(mynah_asr_metrics_buf *b);
void mynah_asr_metrics_addf(mynah_asr_metrics_buf *b, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/* Binds and listens.  Returns the listening descriptor, or -1 after printing
 * one line that names the address, the port and the reason -- a metrics port
 * that is already taken is a configuration error the operator must see, not a
 * feature that quietly disappears. */
int mynah_asr_metrics_listen(const char *bind_addr, int port);

/* The page renderer, supplied by whoever owns the counters. */
typedef void (*mynah_asr_metrics_render_fn)(mynah_asr_metrics_buf *b, void *ud);

/* Accepts ONE pending connection on `lfd` and answers it, bounded as
 * described above.  Call it when poll() reports the listener readable.  A
 * request that is not GET /metrics gets a 404 without rendering; a request
 * over the rate gets a 429 without rendering. */
void mynah_asr_metrics_service(int lfd, mynah_asr_metrics_render_fn render, void *ud);

/* Escapes a value for a Prometheus label (backslash, quote, newline).  Label
 * VALUES here only ever come from the build and from a fixed set of reason
 * codes, but a build id is a git describe string and that is not this file's
 * to trust. */
void mynah_asr_metrics_label(const char *in, char *out, size_t cap);

#endif
