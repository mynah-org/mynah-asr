/* mynah-asr-server-cuda — one process, one GPU, the v2 wire protocol.
 *
 *   GET  /v1/audio/stream        WebSocket: s16le/f32le 16 kHz in, JSON frames out
 *   GET  /v1/health, /v1/models  facts; GET /metrics (also on --metrics-port)
 *
 * Design (.work/cuda-batched-streaming-server.md): ingest threads read the
 * sockets into per-slot PCM rings and never touch the engine; ONE engine thread
 * stages a chunk per ready slot, gathers a cohort (`--cohort-ms`), runs one
 * batched step and publishes the deltas through the v2 async writer
 * (server/stream_out.c, reused). Session books, the `[DUMP]` lines, /v1/health
 * and the frame formats are the CPU server's, so tools/bench/v2_qualify.sh,
 * v2_verdict.py, tests/ws_probe.py and tests/fault_probe.py run unchanged.
 *
 * What is deliberately NOT here: prefork (a CUDA context does not survive
 * fork), a hidden CPU fallback (a dead engine ends every session with
 * `internal_error` and the process exits 70), VAD/eou (S14-8), REST offline
 * transcription (the GPU server serves streams).
 *
 * This tree does not modify server/ or src/ (S14). */
#ifdef __linux__
#define _GNU_SOURCE   /* sched_getaffinity: the CPU budget of --host-threads auto */
#include <sched.h>
#endif
#include "asr_engine.h"
#include "hostprof.h"

#include "cJSON.h"
#include "http_util.h"
#include "mynah_asr.h"
#include "stream_out.h"
#include "ws.h"

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#ifndef MYNAH_ASR_BUILD
#define MYNAH_ASR_BUILD "dev"
#endif

/* ------------------------------------------------------------- constants */
#define LAG_BUCKETS 257
#define LAG_BUCKET_MS 8
#define REQ_HEAD_MAX 8192

enum { SLOT_FREE = 0, SLOT_CLAIMED, SLOT_ACTIVE, SLOT_DONE };
enum { REQ_FINALIZE = 1, REQ_CLOSE = 2, REQ_CANCEL = 4, REQ_RESET = 8 };
enum { CB_IDLE = 0, CB_PEER, CB_FRAME, CB_PROTOCOL, CB_SHUTDOWN, CB_AUDIO_LIMIT, CB_DECODE, CB_OTHER, CB__N };
static const char *const CANCEL_NAME[CB__N] = {
    "idle_timeout", "peer_gone", "frame_too_large", "protocol_error",
    "shutting_down", "audio_limit", "decode_failed", "other"};

/* ------------------------------------------------------------------ slot */
typedef struct {
    int id, state;
    pthread_mutex_t mu;
    pthread_cond_t done_cv;
    pthread_cond_t space;         /* the pusher waits here while the ring is full */
    /* PCM ring, ingest -> engine thread */
    float *ring;
    size_t ring_cap, ring_len, ring_head;
    double ring_arrival;          /* when the newest sample landed */
    double ready_since;           /* when the ring first held a whole chunk (0 = not) */
    double pre_ready;             /* --stage-ahead: the arrival of a chunk staged while
                                     the previous one was in flight (0 = none) */
    int req;                      /* REQ_* bits */
    int cancel_code;              /* CB_* when REQ_CANCEL */
    char reset_lang[16];
    /* session */
    char lang[16];
    int lookahead;
    mynah_asr_stream_out *out;
    unsigned long seq, steps, deltas;
    double lag_sum_ms, lag_max_ms;
    unsigned long lag_hist[LAG_BUCKETS];
    double t_open, t_first_audio;
    int first_text_counted;
    const char *outcome;          /* the code the session ended with, or NULL = completed */
    int done;                     /* the engine thread ended the session */
    int engine_open;              /* the engine slot holds this utterance */
    int abandoned;                /* the ingest gave up waiting: end_session releases */
} gslot;

/* ---------------------------------------------------------------- global */
static struct {
    /* config */
    const char *model_dir, *host, *engine_name, *precision, *gemm, *kv_dtype, *weights;
    int port, metrics_port, device, cap, cohort_ms, http_threads, idle_ms, ping_ms;
    int threads, ring_seconds, profile, profile_host, pass_lanes;
    int stage_ahead, host_threads, graphs, warmup;
    const char *graph_buckets;
    long max_frame_bytes;
    double max_audio_seconds;
    /* state */
    asr_engine *eng;
    asr_engine_facts facts;
    gslot *slots;
    pthread_mutex_t mu;           /* books + slot claims + engine wake */
    pthread_cond_t wake;
    _Atomic int shutdown;
    int listen_fd, metrics_fd;
    pthread_t engine_thread;
    /* books, under mu */
    unsigned long sessions, completed, cancelled, aborted;
    unsigned long abandoned, abandoned_recovered;
    unsigned long cancel_by[CB__N];
    int active;
    unsigned long steps, deltas, cohorts, eous;
    double audio_seconds;
    unsigned long lag_hist[LAG_BUCKETS];
    double lag_sum_ms, lag_max_ms;
    unsigned long refused_cap, refused_other;
    double t0;
    unsigned long dump_seq;
    /* cohort accounting */
    unsigned long cohort_lanes_sum;
    double cohort_wait_ms_sum;
    /* --profile-host (DIAGNOSTIC): the engine thread's wall per phase, under mu */
    hostprof hp;
    /* VRAM used on the device (all processes) when ready, and the most seen since */
    double vram_ready_mb, vram_peak_mb;
    /* accept() out of descriptors / kernel memory: backed off, under mu */
    unsigned long accept_backoffs;
    /* --stage-ahead: the window between submit and finish */
    unsigned long windows, window_staged;
    double window_ms_sum;
    double startup_ms;
} g;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int lag_bucket(double ms) {
    int b = (int)(ms / LAG_BUCKET_MS);
    if (b < 0) b = 0;
    if (b >= LAG_BUCKETS) b = LAG_BUCKETS - 1;
    return b;
}
static double hist_quantile(const unsigned long *h, double q) {
    unsigned long n = 0;
    for (int i = 0; i < LAG_BUCKETS; i++) n += h[i];
    if (n == 0) return 0.0;
    const double want = q * (double)n;
    unsigned long seen = 0;
    for (int i = 0; i < LAG_BUCKETS; i++) {
        seen += h[i];
        if ((double)seen >= want) return (double)i * LAG_BUCKET_MS;
    }
    return (double)(LAG_BUCKETS - 1) * LAG_BUCKET_MS;
}

/* ------------------------------------------------------------ socket io */
static int write_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        const ssize_t w = send(fd, p, n, 0);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return -1;
        p += w; n -= (size_t)w;
    }
    return 0;
}

/* write, half-close, drain briefly, close: a refusal the client can READ */
static void linger_close(int fd, const char *resp, size_t n) {
    if (resp && n) (void)write_all(fd, resp, n);
    shutdown(fd, SHUT_WR);
    struct pollfd p = {.fd = fd, .events = POLLIN, .revents = 0};
    char sink[512];
    const double t_end = now_s() + 0.3;
    for (;;) {
        const int rc = poll(&p, 1, 50);
        if (rc <= 0 || now_s() > t_end) break;
        const ssize_t r = recv(fd, sink, sizeof(sink), 0);
        if (r <= 0) break;
    }
    close(fd);
}

static void refuse_json(int fd, int code, const char *status, const char *type,
                        const char *errcode, const char *msg, int retry_after) {
    pthread_mutex_lock(&g.mu);
    if (retry_after > 0) g.refused_cap++; else g.refused_other++;
    pthread_mutex_unlock(&g.mu);
    char body[320], resp[1024], retry[48] = "";
    if (retry_after > 0) snprintf(retry, sizeof(retry), "Retry-After: %d\r\n", retry_after);
    const int blen = snprintf(body, sizeof(body),
        "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"code\":\"%s\"}}", msg, type, errcode);
    const int n = snprintf(resp, sizeof(resp),
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n%sConnection: close\r\n\r\n%s",
        code, status, blen, retry, body);
    linger_close(fd, resp, (size_t)n);
}

static void send_json(int fd, int code, cJSON *j) {
    char *s = cJSON_PrintUnformatted(j);
    if (!s) return;
    char head[256];
    const int n = snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        code, code == 200 ? "OK" : "Error", strlen(s));
    (void)write_all(fd, head, (size_t)n);
    (void)write_all(fd, s, strlen(s));
    free(s);
}

static void send_text(int fd, int code, const char *ctype, const char *body, size_t len) {
    char head[256];
    const int n = snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
        code, code == 200 ? "OK" : "Error", ctype, len);
    (void)write_all(fd, head, (size_t)n);
    (void)write_all(fd, body, len);
}

/* a client token on its way into a JSON message: only tag characters survive */
static void safe_token(const char *src, char *dst, size_t cap) {
    size_t k = 0;
    for (size_t i = 0; src && src[i] && k + 1 < cap; i++) {
        const char c = src[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
            c == '-' || c == '_' || c == '.' || c == '>')
            dst[k++] = c;
    }
    dst[k] = '\0';
}

/* ---------------------------------------------------------- slot books */
static void slot_release_locked(gslot *s) {
    /* counted ONCE, here: sessions = completed + cancelled + aborted + active */
    if (s->state == SLOT_CLAIMED) g.aborted++;
    else if (s->outcome == NULL) g.completed++;
    else {
        g.cancelled++;
        int b = CB_OTHER;
        for (int i = 0; i < CB__N; i++) if (strcmp(CANCEL_NAME[i], s->outcome) == 0) b = i;
        g.cancel_by[b]++;
    }
    g.active--;
    s->state = SLOT_FREE;
    s->out = NULL;
    s->req = 0; s->done = 0; s->outcome = NULL; s->engine_open = 0; s->abandoned = 0;
    s->ring_len = s->ring_head = 0; s->ready_since = 0.0; s->pre_ready = 0.0;
}

static gslot *slot_claim(const char *lang, int lookahead) {
    pthread_mutex_lock(&g.mu);
    gslot *got = NULL;
    for (int i = 0; i < g.cap && !got; i++)
        if (g.slots[i].state == SLOT_FREE) got = &g.slots[i];
    if (got) {
        got->state = SLOT_CLAIMED;
        snprintf(got->lang, sizeof(got->lang), "%s", lang);
        got->lookahead = lookahead;
        got->seq = got->steps = got->deltas = 0;
        got->lag_sum_ms = got->lag_max_ms = 0.0;
        memset(got->lag_hist, 0, sizeof(got->lag_hist));
        got->t_open = now_s(); got->t_first_audio = 0.0; got->first_text_counted = 0;
        got->outcome = NULL; got->done = 0; got->req = 0; got->cancel_code = 0;
        got->ring_len = got->ring_head = 0; got->ready_since = 0.0; got->pre_ready = 0.0; got->engine_open = 0;
        g.sessions++; g.active++;
    }
    pthread_mutex_unlock(&g.mu);
    return got;
}

static void slot_request(gslot *s, int bits, int cancel_code, const char *lang) {
    pthread_mutex_lock(&s->mu);
    s->req |= bits;
    if (bits & REQ_CANCEL) s->cancel_code = cancel_code;
    if (lang) snprintf(s->reset_lang, sizeof(s->reset_lang), "%s", lang);
    else if (bits & REQ_RESET) s->reset_lang[0] = '\0';
    pthread_mutex_unlock(&s->mu);
    pthread_mutex_lock(&g.mu);
    pthread_cond_broadcast(&g.wake);
    pthread_mutex_unlock(&g.mu);
}

/* ingest -> ring, BLOCKING while the ring is full, as the CPU server's slot
 * does (server/slot.c): a client ahead of real time is throttled through TCP,
 * never cut off. Returns the samples taken; fewer only when the session ended
 * under the pusher (done, cancelled, shutdown). */
static size_t slot_push(gslot *s, const float *pcm, size_t n, double now) {
    size_t off = 0;
    pthread_mutex_lock(&s->mu);
    while (off < n) {
        if (s->done || (s->req & REQ_CANCEL) || atomic_load(&g.shutdown)) break;
        const size_t room = s->ring_cap - s->ring_len;
        if (room == 0) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 200 * 1000000L;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&s->space, &s->mu, &ts);
            continue;
        }
        const size_t take = (n - off) < room ? (n - off) : room;
        for (size_t i = 0; i < take; i++)
            s->ring[(s->ring_head + s->ring_len + i) % s->ring_cap] = pcm[off + i];
        s->ring_len += take;
        s->ring_arrival = now;
        if (s->t_first_audio == 0.0) s->t_first_audio = now;
        off += take;
        pthread_mutex_unlock(&s->mu);
        pthread_mutex_lock(&g.mu);
        pthread_cond_broadcast(&g.wake);
        pthread_mutex_unlock(&g.mu);
        pthread_mutex_lock(&s->mu);
    }
    pthread_mutex_unlock(&s->mu);
    return off;
}

/* ------------------------------------------------------------- frames */
static void frame_send(gslot *s, cJSON *j) {
    if (!s->out) { cJSON_Delete(j); return; }
    char *txt = cJSON_PrintUnformatted(j);
    cJSON_Delete(j);
    if (!txt) return;
    unsigned char buf[8192];
    const size_t n = mynah_asr_ws_frame(buf, sizeof(buf), 0x1, txt, strlen(txt));
    if (n > 0) (void)mynah_asr_stream_out_enqueue(s->out, buf, n);
    else {
        /* a delta longer than the frame buffer: sent in its own allocation */
        const size_t need = strlen(txt) + 16;
        unsigned char *big = malloc(need);
        if (big) {
            const size_t m = mynah_asr_ws_frame(big, need, 0x1, txt, strlen(txt));
            if (m > 0) (void)mynah_asr_stream_out_enqueue(s->out, big, m);
            free(big);
        }
    }
    free(txt);
}
static void frame_common(cJSON *j, gslot *s, double audio_s, double lag_ms) {
    cJSON_AddNumberToObject(j, "seq", (double)(s->seq++));
    cJSON_AddNumberToObject(j, "audio_s", audio_s);
    cJSON_AddNumberToObject(j, "lag_ms", lag_ms);
}
static void frame_error(gslot *s, const char *code, const char *msg) {
    if (!s->out || mynah_asr_stream_out_failed(s->out)) return;
    cJSON *j = cJSON_CreateObject();
    cJSON_AddStringToObject(j, "type", "error");
    cJSON_AddStringToObject(j, "code", code);
    cJSON_AddStringToObject(j, "message", msg);
    frame_common(j, s, 0.0, 0.0);
    frame_send(s, j);
}
static void frame_delta(gslot *s, const asr_step_out *o, double lag_ms) {
    const char *lang = asr_engine_slot_lang(g.eng, s->id);
    if (!lang || !lang[0]) lang = s->lang;
    cJSON *j = cJSON_CreateObject();
    cJSON_AddStringToObject(j, "type", "delta");
    cJSON_AddStringToObject(j, "text", o->text);
    cJSON_AddBoolToObject(j, "final", 1);
    cJSON_AddStringToObject(j, "lang", lang);
    cJSON_AddNumberToObject(j, "t0", o->t0);
    cJSON_AddNumberToObject(j, "t1", o->t1);
    cJSON_AddStringToObject(j, "language", lang);
    cJSON_AddNumberToObject(j, "audio_seconds", o->t1);
    frame_common(j, s, o->t1, lag_ms);
    frame_send(s, j);
}
/* the model's own end of utterance (a pack with <EOU>/<EOB>): the CPU
 * server's eou frame (server/sched.c), field for field; the session goes on */
static void frame_eou(gslot *s, const asr_step_out *o, double lag_ms) {
    cJSON *j = cJSON_CreateObject();
    cJSON_AddStringToObject(j, "type", "eou");
    cJSON_AddNumberToObject(j, "t", o->eou_t);
    cJSON_AddStringToObject(j, "source", "model");
    cJSON_AddBoolToObject(j, "backchannel", o->eou == MYNAH_ASR_EOU_MODEL_BACKCHANNEL);
    frame_common(j, s, o->eou_t, lag_ms);
    frame_send(s, j);
}
static void frame_done(gslot *s) {
    const char *lang = asr_engine_slot_lang(g.eng, s->id);
    if (!lang || !lang[0]) lang = s->lang;
    const double audio_s = asr_engine_slot_audio_s(g.eng, s->id);
    cJSON *j = cJSON_CreateObject();
    cJSON_AddStringToObject(j, "type", "done");
    cJSON_AddBoolToObject(j, "done", 1);
    cJSON_AddStringToObject(j, "lang", lang);
    cJSON_AddStringToObject(j, "language", lang);
    cJSON_AddNumberToObject(j, "steps", (double)s->steps);
    cJSON_AddNumberToObject(j, "deltas", (double)s->deltas);
    cJSON_AddNumberToObject(j, "lag_p50_ms", hist_quantile(s->lag_hist, 0.5));
    cJSON_AddNumberToObject(j, "lag_max_ms", s->lag_max_ms);
    cJSON_AddNumberToObject(j, "audio_seconds", audio_s);
    frame_common(j, s, audio_s, 0.0);
    frame_send(s, j);
}

/* -------------------------------------------------------- engine thread */
static void end_session(gslot *s, const char *outcome) {
    /* engine-thread only: records the outcome, closes the writer, wakes the
     * ingest -- or, when the ingest already gave up on this slot, releases the
     * slot and the writer itself (the ingest cannot: the engine was still
     * using both). Exactly one side releases, decided under s->mu. */
    pthread_mutex_lock(&s->mu);
    if (s->outcome == NULL && outcome) s->outcome = outcome;
    s->state = SLOT_DONE;
    s->done = 1;
    mynah_asr_stream_out *out = s->out;
    if (out) mynah_asr_stream_out_finish(out);
    const int abandoned = s->abandoned;
    const char *ended = s->outcome ? s->outcome : "completed";
    pthread_cond_broadcast(&s->done_cv);
    pthread_cond_broadcast(&s->space);
    pthread_mutex_unlock(&s->mu);
    if (abandoned) {
        pthread_mutex_lock(&g.mu);
        slot_release_locked(s);
        g.abandoned_recovered++;
        pthread_mutex_unlock(&g.mu);
        if (out) mynah_asr_stream_out_release(out);
        fprintf(stderr, "mynah-asr-server-cuda: abandoned slot %d ended (%s) and released by the engine thread\n",
                s->id, ended);
    }
}

static void engine_dead_exit(void) {
    fprintf(stderr, "mynah-asr-server-cuda: the engine is dead (%s); ending every session with internal_error and exiting 70\n",
            asr_engine_error(g.eng));
    for (int i = 0; i < g.cap; i++) {
        gslot *s = &g.slots[i];
        if (s->state == SLOT_ACTIVE) { frame_error(s, "internal_error", "device failure"); end_session(s, "decode_failed"); }
    }
    fflush(stderr);
    _exit(70);
}

/* take `n` samples off the front of the slot's ring into dst (two memcpys) */
static void ring_take(gslot *s, float *dst, size_t n) {
    const size_t first = s->ring_cap - s->ring_head < n ? s->ring_cap - s->ring_head : n;
    memcpy(dst, s->ring + s->ring_head, first * sizeof(float));
    if (n > first) memcpy(dst + first, s->ring, (n - first) * sizeof(float));
    s->ring_head = (s->ring_head + n) % s->ring_cap;
    s->ring_len -= n;
}

/* The stage-ahead scan (--stage-ahead 1, or a feed team): every active slot
 * whose ring holds the whole next chunk has it taken and fed, the host mel of
 * all of them in ONE batch the engine may spread over its feed team. It runs
 * between asr_engine_step_submit and _finish, i.e. under the encoder pass, and
 * before the scan. It takes exactly what the scan would take (one chunk, the
 * engine's own need), so a slot's feed sequence -- hence its mel, its chunks
 * and its transcript -- is the same; only WHEN changes. Slots with a cancel,
 * a reset or a finalize pending are left to the scan, which handles them in
 * order (a finalize decides the tail; audio after a finalize belongs to the
 * next utterance, after the reset that publish does). Engine thread only. */
static struct {
    float *buf;
    size_t cap;
    int *slot;
    const float **pcm;
    size_t *off, *ns;
} g_stage;

static int stage_scan(void) {
    int k = 0;
    size_t used = 0;
    for (int i = 0; i < g.cap; i++) {
        gslot *s = &g.slots[i];
        pthread_mutex_lock(&s->mu);
        if (s->state != SLOT_ACTIVE || (s->req & (REQ_CANCEL | REQ_RESET | REQ_FINALIZE))) { pthread_mutex_unlock(&s->mu); continue; }
        const size_t need = asr_engine_slot_need_samples(g.eng, s->id);
        if (need == 0 || s->ring_len < need) { pthread_mutex_unlock(&s->mu); continue; }
        if (used + need > g_stage.cap) {
            size_t nc = g_stage.cap ? g_stage.cap : 65536;
            while (nc < used + need) nc *= 2;
            float *nb = realloc(g_stage.buf, nc * sizeof(float));
            if (!nb) { pthread_mutex_unlock(&s->mu); fprintf(stderr, "engine thread: out of memory\n"); _exit(70); }
            g_stage.buf = nb; g_stage.cap = nc;
        }
        ring_take(s, g_stage.buf + used, need);
        pthread_cond_broadcast(&s->space);
        /* a slot in flight keeps its cohort's ready_since until publish */
        if (s->ready_since == 0.0) s->ready_since = s->ring_arrival;
        else if (s->pre_ready == 0.0) s->pre_ready = s->ring_arrival;
        pthread_mutex_unlock(&s->mu);
        g_stage.slot[k] = s->id; g_stage.off[k] = used; g_stage.ns[k] = need;
        used += need;
        k++;
    }
    if (k == 0) return 0;
    for (int j = 0; j < k; j++) g_stage.pcm[j] = g_stage.buf + g_stage.off[j];
    if (asr_engine_slot_feed_batch(g.eng, k, g_stage.slot, g_stage.pcm, g_stage.ns) != 0) engine_dead_exit();
    pthread_mutex_lock(&g.mu);
    g.audio_seconds += (double)used / 16000.0;
    pthread_mutex_unlock(&g.mu);
    return k;
}

static void *engine_main(void *arg) {
    (void)arg;
    mynah_asr_thread_set_name("mynah-gpu");
    asr_step_req *reqs = calloc((size_t)g.cap, sizeof(*reqs));
    asr_step_out *outs = calloc((size_t)g.cap, sizeof(*outs));
    gslot **lane = calloc((size_t)g.cap, sizeof(*lane));
    double *arrival = calloc((size_t)g.cap, sizeof(double));
    float *stage = malloc(((size_t)g.ring_seconds * (size_t)g.facts.sample_rate + 1) * sizeof(float));
    g_stage.slot = calloc((size_t)g.cap, sizeof(int));
    g_stage.pcm = calloc((size_t)g.cap, sizeof(float *));
    g_stage.off = calloc((size_t)g.cap, sizeof(size_t));
    g_stage.ns = calloc((size_t)g.cap, sizeof(size_t));
    if (!reqs || !outs || !lane || !arrival || !stage || !g_stage.slot || !g_stage.pcm || !g_stage.off || !g_stage.ns) {
        fprintf(stderr, "engine thread: out of memory\n"); _exit(70);
    }
    /* --stage-ahead: the step in two halves around the stage-ahead scan */
    const int split = g.stage_ahead && asr_engine_can_split(g.eng);
    const int pre = g.stage_ahead || g.host_threads > 1;

    /* --profile-host: every segment of the loop is charged to one phase, so
     * the phases sum to the thread's wall; off, no clock is read for it */
    const int hpf = g.profile_host;
    hostprof_cycle hpc;
    memset(&hpc, 0, sizeof(hpc));
    asr_engine_stats es0, es1;
    if (hpf) {
        hpc.t = hostprof_now();
        pthread_mutex_lock(&g.mu); g.hp.t_start = hpc.t; pthread_mutex_unlock(&g.mu);
    }
#define HPL(ph) do { if (hpf) hostprof_lap(&hpc, (ph)); } while (0)

    while (!atomic_load(&g.shutdown)) {
        if (pre) { (void)stage_scan(); HPL(HP_WINDOW); }
        /* 1. requests: cancel / reset / finalize, and staging of ready chunks */
        int n = 0;
        double oldest = 0.0;
        const double now = now_s();
        for (int i = 0; i < g.cap; i++) {
            gslot *s = &g.slots[i];
            pthread_mutex_lock(&s->mu);
            if (s->state != SLOT_ACTIVE) { pthread_mutex_unlock(&s->mu); continue; }
            const int req = s->req;
            if (req & REQ_CANCEL) {
                const int code = s->cancel_code;
                pthread_mutex_unlock(&s->mu);
                if (code != CB_PEER) frame_error(s, CANCEL_NAME[code], "the session was cancelled");
                end_session(s, CANCEL_NAME[code]);
                continue;
            }
            /* peer gone: hard-only while finalizing (a legal half-close waits for done) */
            if (s->out && mynah_asr_stream_out_peer_gone_ex(s->out, (req & REQ_FINALIZE) != 0)) {
                pthread_mutex_unlock(&s->mu);
                end_session(s, "peer_gone");
                continue;
            }
            if (req & REQ_RESET) {
                s->req &= ~REQ_RESET;
                char lang[16];
                snprintf(lang, sizeof(lang), "%s", s->reset_lang[0] ? s->reset_lang : s->lang);
                pthread_mutex_unlock(&s->mu);
                if (asr_engine_slot_reset(g.eng, s->id, lang, s->lookahead) != 0) {
                    frame_error(s, "reset_failed", "the engine could not reset the slot");
                    end_session(s, "other");
                    continue;
                }
                snprintf(s->lang, sizeof(s->lang), "%s", lang);
                pthread_mutex_lock(&s->mu);
            }
            /* stage: exactly one chunk's worth when the ring holds it; on
             * finalize everything left */
            const size_t need = asr_engine_slot_need_samples(g.eng, s->id);
            const int fin = (req & REQ_FINALIZE) != 0;
            /* one chunk per step, finalize included: the audio counter then
             * says what the model consumed, not what a close frame handed over
             * (tests/fault_probe.py close-then-rst-silent measures exactly that);
             * only a tail shorter than a chunk goes in whole */
            size_t take = 0;
            if (need > 0 && s->ring_len >= need) take = need;
            else if (fin && need > 0 && s->ring_len > 0 && s->ring_len < need) take = s->ring_len;
            if (take > 0) {
                HPL(HP_SCAN);
                ring_take(s, stage, take);
                pthread_cond_broadcast(&s->space);
                if (s->ready_since == 0.0) s->ready_since = s->ring_arrival;
                HPL(HP_STAGE);
                pthread_mutex_unlock(&s->mu);
                if (asr_engine_slot_feed(g.eng, s->id, stage, take) != 0) engine_dead_exit();
                pthread_mutex_lock(&g.mu);
                g.audio_seconds += (double)take / 16000.0;
                pthread_mutex_unlock(&g.mu);
                if (hpf) { hostprof_lap(&hpc, HP_MEL); hpc.chunks++; }
                pthread_mutex_lock(&s->mu);
            }
            /* the engine is told to finalize only once the ring is empty:
             * a client that sends `finalize` right after blasting its audio
             * still has seconds of it queued here, and a finalize handed over
             * early would close the engine's mel stream in front of them */
            const int fin_now = fin && s->ring_len == 0;
            const int ready = asr_engine_slot_ready(g.eng, s->id);
            if (ready || fin_now) {
                reqs[n].slot = s->id; reqs[n].finalize = fin_now;
                lane[n] = s; arrival[n] = s->ready_since > 0.0 ? s->ready_since : s->ring_arrival;
                if (s->ready_since == 0.0) s->ready_since = now;
                if (oldest == 0.0 || s->ready_since < oldest) oldest = s->ready_since;
                n++;
            }
            pthread_mutex_unlock(&s->mu);
        }
        HPL(HP_SCAN);

        /* 2. the cohort policy: step when the oldest ready chunk has waited
         * --cohort-ms, or the set is the whole arena; otherwise wait */
        if (n == 0) {
            pthread_mutex_lock(&g.mu);
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 20 * 1000000L;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&g.wake, &g.mu, &ts);
            if (hpf) { hostprof_lap(&hpc, HP_IDLE); hostprof_fold(&g.hp, &hpc, 0, 0, 0.0, 0.0); }
            pthread_mutex_unlock(&g.mu);
            continue;
        }
        const double waited_ms = (now - oldest) * 1e3;
        if (n < g.cap && waited_ms < (double)g.cohort_ms) {
            pthread_mutex_lock(&g.mu);
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            long ns = (long)(((double)g.cohort_ms - waited_ms) * 1e6);
            if (ns < 1000000L) ns = 1000000L;
            ts.tv_nsec += ns;
            while (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&g.wake, &g.mu, &ts);
            if (hpf) { hostprof_lap(&hpc, HP_COHORT_WAIT); hostprof_fold(&g.hp, &hpc, 0, 0, 0.0, 0.0); }
            pthread_mutex_unlock(&g.mu);
            continue;
        }

        /* 3. one batched step */
        if (hpf) { asr_engine_get_stats(g.eng, &es0); hostprof_lap(&hpc, HP_SCAN); }
        const double t_step = now_s();
        double win_ms = 0.0;
        int win_k = 0;
        if (split) {
            if (asr_engine_step_submit(g.eng, reqs, n, outs) != 0) engine_dead_exit();
            HPL(HP_STEP);
            const double t_w = now_s();
            win_k = stage_scan();                  /* under the encoder pass */
            win_ms = (now_s() - t_w) * 1e3;
            HPL(HP_WINDOW);
            if (asr_engine_step_finish(g.eng, outs) != 0) engine_dead_exit();
        } else if (asr_engine_step(g.eng, reqs, n, outs) != 0) engine_dead_exit();
        const double t_end = now_s();
        double dev_wait_us = 0.0;
        if (hpf) {
            hostprof_lap(&hpc, HP_STEP);
            asr_engine_get_stats(g.eng, &es1);
            const int wph[3] = {ASR_HP_ENC_WAIT, ASR_HP_DEC_WAIT, ASR_HP_FINAL_WAIT};
            for (int w = 0; w < 3; w++) dev_wait_us += es1.hprof_us[wph[w]] - es0.hprof_us[wph[w]];
        }
        pthread_mutex_lock(&g.mu);
        if (split) { g.windows++; g.window_ms_sum += win_ms; g.window_staged += (unsigned long)win_k; }
        g.cohorts++;
        g.cohort_lanes_sum += (unsigned long)n;
        g.cohort_wait_ms_sum += waited_ms;
        pthread_mutex_unlock(&g.mu);

        /* 4. publish */
        for (int a = 0; a < n; a++) {
            gslot *s = lane[a];
            const asr_step_out *o = &outs[a];
            if (o->stepped) {
                pthread_mutex_lock(&s->mu);
                s->ready_since = s->pre_ready;     /* 0 unless staged ahead */
                s->pre_ready = 0.0;
                s->steps++;
                pthread_mutex_unlock(&s->mu);
                pthread_mutex_lock(&g.mu); g.steps++; pthread_mutex_unlock(&g.mu);
            }
            if (o->text && o->text[0]) {
                double lag_ms = arrival[a] > 0.0 ? (t_end - arrival[a]) * 1e3 : 0.0;
                if (lag_ms < 0.0) lag_ms = 0.0;
                frame_delta(s, o, lag_ms);
                s->deltas++;
                s->lag_sum_ms += lag_ms;
                if (lag_ms > s->lag_max_ms) s->lag_max_ms = lag_ms;
                s->lag_hist[lag_bucket(lag_ms)]++;
                pthread_mutex_lock(&g.mu);
                g.deltas++;
                g.lag_hist[lag_bucket(lag_ms)]++;
                g.lag_sum_ms += lag_ms;
                if (lag_ms > g.lag_max_ms) g.lag_max_ms = lag_ms;
                pthread_mutex_unlock(&g.mu);
            }
            if (o->eou) {
                double lag_ms = arrival[a] > 0.0 ? (t_end - arrival[a]) * 1e3 : 0.0;
                if (lag_ms < 0.0) lag_ms = 0.0;
                frame_eou(s, o, lag_ms);
                pthread_mutex_lock(&g.mu); g.eous++; pthread_mutex_unlock(&g.mu);
            }
            if (o->finished) {
                pthread_mutex_lock(&s->mu);
                const int close = (s->req & REQ_CLOSE) != 0;
                s->req &= ~(REQ_FINALIZE | REQ_CLOSE);
                pthread_mutex_unlock(&s->mu);
                frame_done(s);
                if (close) {
                    end_session(s, NULL);
                } else {
                    /* finalize without close: the next audio starts a new
                     * utterance on the same slot (the CPU server's reset) */
                    if (asr_engine_slot_reset(g.eng, s->id, s->lang, s->lookahead) != 0) {
                        frame_error(s, "reset_failed", "the engine could not reset the slot");
                        end_session(s, "other");
                    }
                }
            }
        }
        if (hpf) {
            hostprof_lap(&hpc, HP_PUBLISH);
            pthread_mutex_lock(&g.mu);
            hostprof_fold(&g.hp, &hpc, 1, n, dev_wait_us, t_step);
            hostprof_rusage(&g.hp);
            pthread_mutex_unlock(&g.mu);
        }
    }
#undef HPL
    /* shutdown: every live session is owed an error frame */
    for (int i = 0; i < g.cap; i++) {
        gslot *s = &g.slots[i];
        if (s->state == SLOT_ACTIVE) { frame_error(s, "shutting_down", "the server is shutting down"); end_session(s, "shutting_down"); }
    }
    free(reqs); free(outs); free(lane); free(arrival); free(stage);
    free(g_stage.buf); free(g_stage.slot); free(g_stage.pcm); free(g_stage.off); free(g_stage.ns);
    return NULL;
}

/* ------------------------------------------------------------ WebSocket */
typedef struct { char lang[16]; int lookahead, f32; } ws_params;

#define WS_QUERY_ACCEPTED "model, lang, lookahead, format, rate"

static int ws_parse_query(const char *query, ws_params *p, const char **code, char *msg, size_t cap) {
    snprintf(p->lang, sizeof(p->lang), "auto");
    p->lookahead = g.facts.default_lookahead;
    p->f32 = 0;
    if (!query || !query[0]) return 0;
    const char *q = query;
    while (*q) {
        const char *amp = strchr(q, '&');
        const size_t seg = amp ? (size_t)(amp - q) : strlen(q);
        const char *eq = memchr(q, '=', seg);
        const size_t klen = eq ? (size_t)(eq - q) : seg;
        const size_t vlen = eq ? seg - klen - 1 : 0;
        char key[40], val[64], tok[48];
        snprintf(key, sizeof(key), "%.*s", (int)(klen < sizeof(key) ? klen : sizeof(key) - 1), q);
        snprintf(val, sizeof(val), "%.*s", (int)(vlen < sizeof(val) ? vlen : sizeof(val) - 1), eq ? eq + 1 : "");
        if (klen == 0) {
        } else if (strcmp(key, "model") == 0) {
            if (val[0] && strcmp(val, g.facts.model_name) != 0) {
                safe_token(val, tok, sizeof(tok));
                snprintf(msg, cap, "model '%s' is not served; accepted: %s", tok, g.facts.model_name);
                *code = "model_not_found"; return 404;
            }
        } else if (strcmp(key, "lang") == 0) {
            if (val[0] == '\0') snprintf(p->lang, sizeof(p->lang), "auto");
            else if (asr_engine_lang_id(g.eng, val) < 0) {
                safe_token(val, tok, sizeof(tok));
                snprintf(msg, cap, "this model does not serve the language '%s'", tok);
                *code = "language_not_served"; return 400;
            } else snprintf(p->lang, sizeof(p->lang), "%s", val);
        } else if (strcmp(key, "lookahead") == 0) {
            if (val[0]) {
                char *end = NULL;
                const long want = strtol(val, &end, 10);
                if (!end || *end || !asr_engine_lookahead_ok(g.eng, (int)want)) {
                    char set[64] = ""; size_t w = 0;
                    for (int i = 0; i < g.facts.n_lookaheads && w + 8 < sizeof(set); i++)
                        w += (size_t)snprintf(set + w, sizeof(set) - w, "%s%d", i ? ", " : "", g.facts.lookaheads[i]);
                    safe_token(val, tok, sizeof(tok));
                    snprintf(msg, cap, "lookahead '%s' is not a preset of this model; accepted: %s", tok, set);
                    *code = "lookahead_not_available"; return 400;
                }
                p->lookahead = (int)want;
            }
        } else if (strcmp(key, "format") == 0) {
            if (strcmp(val, "f32le") == 0) p->f32 = 1;
            else if (val[0] && strcmp(val, "s16le") != 0) {
                safe_token(val, tok, sizeof(tok));
                snprintf(msg, cap, "format '%s' is not served; accepted: s16le, f32le", tok);
                *code = "unsupported_format"; return 400;
            }
        } else if (strcmp(key, "rate") == 0) {
            char want[16];
            snprintf(want, sizeof(want), "%d", g.facts.sample_rate);
            if (val[0] && strcmp(val, want) != 0) {
                safe_token(val, tok, sizeof(tok));
                snprintf(msg, cap, "rate '%s' is not served; accepted: %s (resample client-side)", tok, want);
                *code = "unsupported_rate"; return 400;
            }
        } else {
            safe_token(key, tok, sizeof(tok));
            snprintf(msg, cap, "unknown query parameter '%s'; accepted: %s", tok, WS_QUERY_ACCEPTED);
            *code = "unknown_query_parameter"; return 400;
        }
        if (!amp) break;
        q = amp + 1;
    }
    return 0;
}

typedef struct { int fd, f32; gslot *slot; mynah_asr_stream_out *out; int finalize_pending; } ws_ingest;

static void ws_enqueue(ws_ingest *w, int opcode, const void *payload, size_t len) {
    unsigned char buf[1024];
    const size_t n = mynah_asr_ws_frame(buf, sizeof(buf), opcode, payload, len);
    if (n > 0) (void)mynah_asr_stream_out_enqueue(w->out, buf, n);
}
static void ws_transport_error(ws_ingest *w, const char *code, const char *msg) {
    cJSON *j = cJSON_CreateObject();
    cJSON_AddStringToObject(j, "type", "error");
    cJSON_AddStringToObject(j, "code", code);
    cJSON_AddStringToObject(j, "message", msg);
    char *s = cJSON_PrintUnformatted(j);
    if (s) { ws_enqueue(w, 0x1, s, strlen(s)); free(s); }
    cJSON_Delete(j);
}
static void ws_control(ws_ingest *w, const uint8_t *payload, size_t len) {
    char buf[512];
    const size_t n = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
    memcpy(buf, payload, n); buf[n] = '\0';
    cJSON *j = cJSON_Parse(buf);
    const cJSON *t = j ? cJSON_GetObjectItem(j, "type") : NULL;
    const char *type = (t && cJSON_IsString(t)) ? t->valuestring : NULL;
    if (type && strcmp(type, "finalize") == 0) {
        w->finalize_pending = 1;
        slot_request(w->slot, REQ_FINALIZE, 0, NULL);
    } else if (type && strcmp(type, "reset") == 0) {
        const cJSON *l = cJSON_GetObjectItem(j, "lang");
        const char *lang = (l && cJSON_IsString(l)) ? l->valuestring : NULL;
        if (lang && lang[0] && asr_engine_lang_id(g.eng, lang) < 0) {
            char tok[48], m[160];
            safe_token(lang, tok, sizeof(tok));
            snprintf(m, sizeof(m), "this model does not serve the language '%s'", tok);
            ws_transport_error(w, "language_not_served", m);
        } else slot_request(w->slot, REQ_RESET, 0, lang ? lang : "");
    } else ws_transport_error(w, "unknown_control", "expected {\"type\":\"finalize\"} or {\"type\":\"reset\"}");
    cJSON_Delete(j);
}
static int ws_push_pcm(ws_ingest *w, const uint8_t *payload, size_t plen, double now) {
    const size_t width = w->f32 ? 4u : 2u;
    const size_t ns = plen / width;
    float f[4096];
    size_t off = 0;
    while (off < ns) {
        const size_t chunk = ns - off < 4096 ? ns - off : 4096;
        for (size_t i = 0; i < chunk; i++) {
            const size_t k = (off + i) * width;
            if (width == 2) {
                const int16_t v = (int16_t)((uint16_t)payload[k] | ((uint16_t)payload[k + 1] << 8));
                f[i] = (float)v / 32768.0f;
            } else {
                const uint32_t u = (uint32_t)payload[k] | ((uint32_t)payload[k + 1] << 8) |
                                   ((uint32_t)payload[k + 2] << 16) | ((uint32_t)payload[k + 3] << 24);
                memcpy(&f[i], &u, sizeof(f[i]));
            }
        }
        if (slot_push(w->slot, f, chunk, now) != chunk) return 0;
        off += chunk;
    }
    return 1;
}
static int ws_read_loss(int rr, int finalize_pending) {
    if (rr == WS_READ_TIMEOUT) return CB_IDLE;
    if (rr == WS_READ_EOF && finalize_pending) return -1;   /* legal half-close */
    return CB_PEER;
}

static int slot_wait_done(gslot *s, int ms) {
    pthread_mutex_lock(&s->mu);
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += ms / 1000;
    ts.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    while (!s->done) if (pthread_cond_timedwait(&s->done_cv, &s->mu, &ts) != 0) break;
    const int done = s->done;
    pthread_mutex_unlock(&s->mu);
    return done;
}

/* returns 1 when the fd is no longer the caller's */
static int handle_ws_stream(int fd, const char *headers, const char *query) {
    ws_params params;
    const char *qcode = "invalid_request";
    char qmsg[192] = "";
    const int qstatus = ws_parse_query(query, &params, &qcode, qmsg, sizeof(qmsg));
    if (qstatus != 0) {
        refuse_json(fd, qstatus, qstatus == 404 ? "Not Found" : "Bad Request", "invalid_request_error", qcode, qmsg, 0);
        return 1;
    }
    const char *k = strstr(headers, "Sec-WebSocket-Key:");
    if (!k) {
        refuse_json(fd, 400, "Bad Request", "invalid_request_error", "invalid_handshake",
                    "a WebSocket upgrade needs a Sec-WebSocket-Key header", 0);
        return 1;
    }
    char key[64] = {0};
    sscanf(k + 18, " %63[^\r\n]", key);
    if (asr_engine_dead(g.eng)) {
        refuse_json(fd, 503, "Service Unavailable", "server_error", "engine_dead", "the engine failed; restart the server", 0);
        return 1;
    }
    gslot *slot = slot_claim(params.lang, params.lookahead);
    if (!slot) {
        refuse_json(fd, 503, "Service Unavailable", "server_error", "server_at_capacity",
                    "every slot of this GPU is in use", 1);
        return 1;
    }
    /* the ingest's own descriptor BEFORE the upgrade: out of descriptors
     * (EMFILE) is then a 503 the client can read, not a silent close right
     * after a 101 */
    const int rfd = dup(fd);
    if (rfd < 0) {
        pthread_mutex_lock(&g.mu); slot_release_locked(slot); pthread_mutex_unlock(&g.mu);
        refuse_json(fd, 503, "Service Unavailable", "server_error", "server_at_capacity",
                    "the server is out of file descriptors", 1);
        return 1;
    }
    char accept[40];
    ws_accept_key(key, accept, sizeof(accept));
    char resp[256];
    const int rn = snprintf(resp, sizeof(resp),
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n", accept);
    if (write_all(fd, resp, (size_t)rn) != 0) {
        close(rfd);
        pthread_mutex_lock(&g.mu); slot_release_locked(slot); pthread_mutex_unlock(&g.mu);
        return 0;
    }
    struct timeval tv = {.tv_sec = g.idle_ms / 1000, .tv_usec = (g.idle_ms % 1000) * 1000};
    (void)setsockopt(rfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    mynah_asr_stream_out *out = mynah_asr_stream_out_start(fd, 0, 0);
    if (!out) { close(rfd); pthread_mutex_lock(&g.mu); slot_release_locked(slot); pthread_mutex_unlock(&g.mu); return 0; }

    /* arm: the engine slot starts a new utterance, the session becomes ACTIVE */
    pthread_mutex_lock(&slot->mu);
    slot->out = out;
    slot->state = SLOT_ACTIVE;
    slot->req = REQ_RESET;
    slot->reset_lang[0] = '\0';
    pthread_mutex_unlock(&slot->mu);
    pthread_mutex_lock(&g.mu); pthread_cond_broadcast(&g.wake); pthread_mutex_unlock(&g.mu);

    ws_ingest w = {.fd = rfd, .f32 = params.f32, .slot = slot, .out = out};
    mynah_asr_thread_set_name("mynah-ingest");
    int cancelled = 0, closed = 0, shutting = 0, lost = -1;
    size_t audio_samples = 0;
    double last_activity = now_s(), last_ping = last_activity;
    uint8_t *payload = NULL;
    while (!closed) {
        if (atomic_load(&g.shutdown)) { shutting = 1; break; }
        pthread_mutex_lock(&slot->mu);
        const int done = slot->done;
        pthread_mutex_unlock(&slot->mu);
        if (done) break;
        if (mynah_asr_stream_out_failed(out)) { lost = CB_PEER; break; }
        const double tick = now_s();
        if (g.ping_ms > 0 && (tick - last_ping) * 1000.0 >= (double)g.ping_ms) { ws_enqueue(&w, 0x9, "", 0); last_ping = tick; }
        struct pollfd pfd = {.fd = rfd, .events = POLLIN, .revents = 0};
        const int ready = poll(&pfd, 1, 200);
        if (ready < 0) { if (errno == EINTR) continue; lost = CB_PEER; break; }
        if (ready == 0) {
            if ((now_s() - last_activity) * 1000.0 >= (double)g.idle_ms) {
                slot_request(slot, REQ_CANCEL, CB_IDLE, NULL); cancelled = 1; break;
            }
            continue;
        }
        ws_frame_hdr h;
        int rr = ws_read_header(rfd, &h);
        if (rr != WS_READ_OK) { lost = ws_read_loss(rr, w.finalize_pending); if (lost < 0) { closed = 1; slot_request(slot, REQ_FINALIZE | REQ_CLOSE, 0, NULL); } break; }
        if (h.rsv != 0 || ((h.opcode & 0x8) && (h.len > 125 || !h.fin))) {
            ws_transport_error(&w, "protocol_error", h.rsv ? "reserved bits set: no extension was negotiated"
                                                            : "a control frame must be final and at most 125 bytes");
            slot_request(slot, REQ_CANCEL, CB_PROTOCOL, NULL); cancelled = 1; break;
        }
        if (h.len > (uint64_t)g.max_frame_bytes) {
            ws_transport_error(&w, "frame_too_large", "the frame exceeds --max-frame-bytes");
            slot_request(slot, REQ_CANCEL, CB_FRAME, NULL); cancelled = 1; break;
        }
        payload = malloc(h.len ? (size_t)h.len : 1);
        if (!payload) { lost = CB_PEER; break; }
        if ((rr = ws_read_exact(rfd, payload, (size_t)h.len)) != WS_READ_OK) { lost = ws_read_loss(rr, 0); break; }
        if (h.masked) for (uint64_t i = 0; i < h.len; i++) payload[i] ^= h.mask[i & 3];
        last_activity = now_s();
        switch (h.opcode) {
            case 0x8: slot_request(slot, REQ_FINALIZE | REQ_CLOSE, 0, NULL); closed = 1; break;
            case 0x9: ws_enqueue(&w, 0xA, payload, (size_t)h.len); break;
            case 0xA: break;
            case 0x1: ws_control(&w, payload, (size_t)h.len); break;
            case 0x0:
            case 0x2: {
                const size_t width = w.f32 ? 4u : 2u;
                if (h.len < width) break;
                w.finalize_pending = 0;
                if (!ws_push_pcm(&w, payload, (size_t)h.len, last_activity)) { closed = 1; break; }
                audio_samples += (size_t)h.len / width;
                if (g.max_audio_seconds > 0.0 && (double)audio_samples / (double)g.facts.sample_rate > g.max_audio_seconds) {
                    pthread_mutex_lock(&slot->mu); if (!slot->outcome) slot->outcome = "audio_limit"; pthread_mutex_unlock(&slot->mu);
                    ws_transport_error(&w, "audio_limit", "the stream reached --max-audio-seconds");
                    slot_request(slot, REQ_FINALIZE | REQ_CLOSE, 0, NULL); closed = 1;
                }
                break;
            }
            default: ws_transport_error(&w, "unsupported_opcode", "only text, binary, ping, pong and close are served"); break;
        }
        free(payload); payload = NULL;
    }
    free(payload);
    if (!cancelled && !shutting && !closed && lost >= 0) { slot_request(slot, REQ_CANCEL, lost, NULL); cancelled = 1; }
    if (!cancelled && !shutting && !closed) slot_request(slot, REQ_FINALIZE | REQ_CLOSE, 0, NULL);
    /* As in the CPU server: an unsolicited pong (no reply expected, so a
     * legally half-closed client is never asked to write) every 500 ms while
     * waiting for `done` gives a peer that vanished after its close frame or
     * finalize something to answer with a reset, so the tail is not flushed
     * for nobody. */
    int done = 0;
    for (int waited = 0; waited < 60000 && !done; waited += 250) {
        if (!cancelled && !shutting && waited % 500 == 0) ws_enqueue(&w, 0xA, "", 0);
        done = slot_wait_done(slot, 250);
    }
    if (!done) { slot_request(slot, REQ_CANCEL, CB_PEER, NULL); done = slot_wait_done(slot, 5000); }
    close(rfd);
    mynah_asr_thread_set_name("mynah-http");
    if (!done) {
        /* the engine still holds the slot and may still write through `out`:
         * releasing here would hand it freed memory. Hand the release over,
         * unless the session ended between the wait and this lock. */
        pthread_mutex_lock(&slot->mu);
        if (!slot->done) slot->abandoned = 1;
        const int handed = slot->abandoned;
        pthread_mutex_unlock(&slot->mu);
        if (handed) {
            pthread_mutex_lock(&g.mu); g.abandoned++; pthread_mutex_unlock(&g.mu);
            fprintf(stderr, "mynah-asr-server-cuda: slot %d did not finish; left to the engine thread to release\n", slot->id);
            return 1;
        }
    }
    pthread_mutex_lock(&g.mu);
    slot_release_locked(slot);
    pthread_mutex_unlock(&g.mu);
    mynah_asr_stream_out_release(out);
    return 1;
}

/* ------------------------------------------------------------ observability */
static void health_json(cJSON *j) {
    asr_engine_stats es;
    asr_engine_get_stats(g.eng, &es);
    asr_engine_facts f;
    asr_engine_get_facts(g.eng, &f);
    pthread_mutex_lock(&g.mu);
    cJSON_AddStringToObject(j, "status", asr_engine_dead(g.eng) ? "dead" : "ok");
    cJSON_AddNumberToObject(j, "inflight", g.active);
    cJSON_AddNumberToObject(j, "worker", -1);
    cJSON *sl = cJSON_AddObjectToObject(j, "slots");
    cJSON_AddNumberToObject(sl, "active", g.active);
    cJSON_AddNumberToObject(sl, "cap", g.cap);
    cJSON_AddNumberToObject(j, "steps", (double)g.steps);
    cJSON_AddNumberToObject(j, "deltas", (double)g.deltas);
    cJSON_AddNumberToObject(j, "eous", (double)g.eous);
    cJSON_AddNumberToObject(j, "sessions", (double)g.sessions);
    cJSON_AddNumberToObject(j, "completed", (double)g.completed);
    cJSON_AddNumberToObject(j, "aborted", (double)g.aborted);
    cJSON_AddNumberToObject(j, "cancelled", (double)g.cancelled);
    const int balanced = g.sessions == g.completed + g.cancelled + g.aborted + (unsigned long)g.active;
    cJSON_AddBoolToObject(j, "balanced", balanced);
    cJSON *ab = cJSON_AddObjectToObject(j, "abandoned");
    cJSON_AddNumberToObject(ab, "total", (double)g.abandoned);
    cJSON_AddNumberToObject(ab, "recovered", (double)g.abandoned_recovered);
    cJSON *cb = cJSON_AddObjectToObject(j, "cancelled_by");
    for (int i = 0; i < CB__N; i++) cJSON_AddNumberToObject(cb, CANCEL_NAME[i], (double)g.cancel_by[i]);
    cJSON_AddNumberToObject(j, "audio_seconds", g.audio_seconds);
    cJSON *lag = cJSON_AddObjectToObject(j, "lag_ms");
    cJSON_AddNumberToObject(lag, "p50", hist_quantile(g.lag_hist, 0.5));
    cJSON_AddNumberToObject(lag, "p95", hist_quantile(g.lag_hist, 0.95));
    cJSON_AddNumberToObject(lag, "max", g.lag_max_ms);
    unsigned long cnt = 0; for (int i = 0; i < LAG_BUCKETS; i++) cnt += g.lag_hist[i];
    cJSON_AddNumberToObject(lag, "count", (double)cnt);
    cJSON_AddNumberToObject(lag, "bucket_ms", LAG_BUCKET_MS);
    cJSON_AddNumberToObject(j, "lag_p50_ms", hist_quantile(g.lag_hist, 0.5));
    cJSON_AddNumberToObject(j, "lag_max_ms", g.lag_max_ms);
    cJSON_AddBoolToObject(j, "streaming", 1);
    cJSON *bt = cJSON_AddObjectToObject(j, "batch");
    cJSON_AddNumberToObject(bt, "batched_steps_total", (double)g.cohorts);
    cJSON_AddNumberToObject(bt, "rows_stacked_total", (double)es.rows);
    cJSON_AddNumberToObject(bt, "ready_sum", (double)g.cohort_lanes_sum);
    cJSON_AddNumberToObject(bt, "ready_mean", g.cohorts ? (double)g.cohort_lanes_sum / (double)g.cohorts : 0.0);
    cJSON_AddNumberToObject(bt, "cohort_wait_ms_mean", g.cohorts ? g.cohort_wait_ms_sum / (double)g.cohorts : 0.0);
    cJSON *sw = cJSON_AddObjectToObject(bt, "step_wall_ms");
    cJSON_AddNumberToObject(sw, "sum", es.step_wall_ms_sum);
    cJSON_AddNumberToObject(sw, "count", (double)es.steps);
    cJSON_AddNumberToObject(sw, "mean", es.steps ? es.step_wall_ms_sum / (double)es.steps : 0.0);
    cJSON *ge = cJSON_AddObjectToObject(j, "engine");
    cJSON_AddStringToObject(ge, "name", f.name);
    cJSON_AddStringToObject(ge, "device", f.device ? f.device : "");
    cJSON_AddStringToObject(ge, "precision", f.precision);
    cJSON_AddStringToObject(ge, "gemm", f.gemm);
    cJSON_AddNumberToObject(ge, "cohort_ms", g.cohort_ms);
    cJSON_AddNumberToObject(ge, "vram_total_bytes", (double)f.vram_total);
    cJSON_AddNumberToObject(ge, "vram_used_bytes", (double)f.vram_used);
    cJSON_AddNumberToObject(ge, "vram_weights_bytes", (double)f.vram_weights);
    cJSON_AddNumberToObject(ge, "vram_arena_bytes", (double)f.vram_arena);
    cJSON_AddNumberToObject(ge, "vram_ready_mb", g.vram_ready_mb);
    if ((double)f.vram_used / 1048576.0 > g.vram_peak_mb) g.vram_peak_mb = (double)f.vram_used / 1048576.0;
    cJSON_AddNumberToObject(ge, "vram_peak_seen_mb", g.vram_peak_mb);
    cJSON_AddNumberToObject(ge, "pass_lanes", f.pass_lanes);
    cJSON_AddNumberToObject(ge, "decode_iters_total", (double)es.decode_iters);
    cJSON_AddNumberToObject(ge, "h2d_bytes_total", es.h2d_bytes);
    cJSON_AddNumberToObject(ge, "d2h_bytes_total", es.d2h_bytes);
    cJSON_AddNumberToObject(ge, "device_errors_total", (double)es.errors);
    cJSON_AddNumberToObject(ge, "graphs", f.graphs);
    cJSON_AddNumberToObject(ge, "graph_passes_total", (double)es.graph_passes);
    cJSON_AddNumberToObject(ge, "eager_passes_total", (double)es.eager_passes);
    cJSON_AddNumberToObject(ge, "stage_ahead", g.stage_ahead);
    cJSON_AddNumberToObject(ge, "host_threads", g.host_threads);
    cJSON_AddNumberToObject(ge, "startup_ms", g.startup_ms);
    cJSON_AddNumberToObject(ge, "host_mel_ms_total", es.host_mel_ms);
    if (es.prof_passes > 0) {
        cJSON *pr = cJSON_AddObjectToObject(ge, "profile_ms");
        cJSON_AddNumberToObject(pr, "passes", (double)es.prof_passes);
        for (int i = 0; i < ASR_PROF_STAGES; i++) cJSON_AddNumberToObject(pr, ASR_PROF_NAME[i], es.prof_ms[i]);
    }
    if (g.profile_host) {
        /* DIAGNOSTIC: microseconds of the engine thread's wall per phase, summed */
        cJSON *hp = cJSON_AddObjectToObject(ge, "host_profile_us");
        cJSON_AddNumberToObject(hp, "cycles", (double)g.hp.cycles);
        cJSON_AddNumberToObject(hp, "lanes", (double)g.hp.lanes);
        cJSON_AddNumberToObject(hp, "device_wait", g.hp.dev_wait_us);
        for (int i = 0; i < HP__N; i++) cJSON_AddNumberToObject(hp, HP_NAME[i], g.hp.us[i]);
        cJSON *st = cJSON_AddObjectToObject(hp, "step");
        for (int i = 0; i < ASR_HPROF_PHASES; i++) cJSON_AddNumberToObject(st, ASR_HPROF_NAME[i], es.hprof_us[i]);
        cJSON_AddNumberToObject(st, "passes", (double)es.hprof_passes);
        cJSON_AddNumberToObject(st, "syncs", (double)es.hprof_syncs);
    }
    cJSON_AddNumberToObject(j, "accept_backoffs", (double)g.accept_backoffs);
    cJSON *refused = cJSON_AddObjectToObject(j, "refused");
    cJSON_AddNumberToObject(refused, "server_at_capacity", (double)g.refused_cap);
    cJSON_AddNumberToObject(refused, "other", (double)g.refused_other);
    pthread_mutex_unlock(&g.mu);
}

static size_t metrics_text(char *b, size_t cap) {
    asr_engine_stats es; asr_engine_get_stats(g.eng, &es);
    pthread_mutex_lock(&g.mu);
    const int balanced = g.sessions == g.completed + g.cancelled + g.aborted + (unsigned long)g.active;
    size_t k = 0;
#define M(...) do { if (k < cap) k += (size_t)snprintf(b + k, cap - k, __VA_ARGS__); if (k >= cap) k = cap - 1; } while (0)
    M("# TYPE mynah_asr_build_info gauge\nmynah_asr_build_info{build=\"%s\",engine=\"%s\"} 1\n", MYNAH_ASR_BUILD, g.facts.name);
    M("# TYPE mynah_asr_uptime_seconds gauge\nmynah_asr_uptime_seconds %.1f\n", now_s() - g.t0);
    M("# TYPE mynah_asr_sessions_total counter\nmynah_asr_sessions_total{worker=\"gpu\"} %lu\n", g.sessions);
    M("# TYPE mynah_asr_sessions_completed_total counter\nmynah_asr_sessions_completed_total{worker=\"gpu\"} %lu\n", g.completed);
    M("# TYPE mynah_asr_sessions_aborted_total counter\nmynah_asr_sessions_aborted_total{worker=\"gpu\"} %lu\n", g.aborted);
    M("# TYPE mynah_asr_sessions_active gauge\nmynah_asr_sessions_active{worker=\"gpu\"} %d\n", g.active);
    M("# TYPE mynah_asr_sessions_balanced gauge\nmynah_asr_sessions_balanced{worker=\"gpu\"} %d\n", balanced);
    M("# TYPE mynah_asr_cancelled_total counter\n");
    for (int i = 0; i < CB__N; i++) M("mynah_asr_cancelled_total{worker=\"gpu\",reason=\"%s\"} %lu\n", CANCEL_NAME[i], g.cancel_by[i]);
    M("# TYPE mynah_asr_refused_total counter\nmynah_asr_refused_total{code=\"server_at_capacity\"} %lu\nmynah_asr_refused_total{code=\"other\"} %lu\n", g.refused_cap, g.refused_other);
    M("# TYPE mynah_asr_steps_total counter\nmynah_asr_steps_total{worker=\"gpu\"} %lu\n", g.steps);
    M("# TYPE mynah_asr_deltas_total counter\nmynah_asr_deltas_total{worker=\"gpu\"} %lu\n", g.deltas);
    M("# TYPE mynah_asr_audio_seconds_total counter\nmynah_asr_audio_seconds_total{worker=\"gpu\"} %.3f\n", g.audio_seconds);
    unsigned long cnt = 0; for (int i = 0; i < LAG_BUCKETS; i++) cnt += g.lag_hist[i];
    M("# TYPE mynah_asr_emission_lag_ms_sum counter\nmynah_asr_emission_lag_ms_sum{worker=\"gpu\"} %.1f\n", g.lag_sum_ms);
    M("# TYPE mynah_asr_emission_lag_ms_count counter\nmynah_asr_emission_lag_ms_count{worker=\"gpu\"} %lu\n", cnt);
    M("# TYPE mynah_asr_gpu_cohorts_total counter\nmynah_asr_gpu_cohorts_total %lu\n", g.cohorts);
    M("# TYPE mynah_asr_gpu_cohort_lanes_sum counter\nmynah_asr_gpu_cohort_lanes_sum %lu\n", g.cohort_lanes_sum);
    M("# TYPE mynah_asr_gpu_step_wall_ms_sum counter\nmynah_asr_gpu_step_wall_ms_sum %.3f\n", es.step_wall_ms_sum);
    M("# TYPE mynah_asr_gpu_decode_iters_total counter\nmynah_asr_gpu_decode_iters_total %lu\n", es.decode_iters);
    M("# TYPE mynah_asr_gpu_h2d_bytes_total counter\nmynah_asr_gpu_h2d_bytes_total %.0f\n", es.h2d_bytes);
    M("# TYPE mynah_asr_gpu_d2h_bytes_total counter\nmynah_asr_gpu_d2h_bytes_total %.0f\n", es.d2h_bytes);
    M("# TYPE mynah_asr_gpu_device_errors_total counter\nmynah_asr_gpu_device_errors_total %lu\n", es.errors);
    M("# TYPE mynah_asr_accept_backoffs_total counter\nmynah_asr_accept_backoffs_total %lu\n", g.accept_backoffs);
    if (g.profile_host && k < cap)
        k += hostprof_metrics(&g.hp, es.hprof_us, ASR_HPROF_NAME, ASR_HPROF_PHASES, b + k, cap - k);
#undef M
    pthread_mutex_unlock(&g.mu);
    return k;
}

static void dump_stderr(void) {
    asr_engine_stats es; asr_engine_get_stats(g.eng, &es);
    asr_engine_facts f; asr_engine_get_facts(g.eng, &f);
    pthread_mutex_lock(&g.mu);
    const unsigned long n = ++g.dump_seq;
    const int balanced = g.sessions == g.completed + g.cancelled + g.aborted + (unsigned long)g.active;
    char b[8192]; size_t k = 0;
#define D(...) do { if (k < sizeof(b)) k += (size_t)snprintf(b + k, sizeof(b) - k, __VA_ARGS__); if (k >= sizeof(b)) k = sizeof(b) - 1; } while (0)
    D("[DUMP] v=1 worker=0 seq=%lu begin\n", n);
    D("[DUMP] worker=0 seq=%lu process pid=%d uptime_s=%.1f engine=%s device=%s\n", n, (int)getpid(), now_s() - g.t0, g.facts.name, g.facts.device ? g.facts.device : "-");
    D("[DUMP] worker=0 seq=%lu build=%s precision=%s gemm=%s cohort_ms=%d\n", n, MYNAH_ASR_BUILD, g.facts.precision, g.facts.gemm, g.cohort_ms);
    D("[DUMP] worker=0 seq=%lu model=%s engine=rnnt quant=%s streaming=yes lookahead_default=%d chunk_ms=%.0f\n",
      n, g.facts.model_name, g.facts.precision, g.facts.default_lookahead, (g.facts.default_lookahead + 1) * g.facts.frame_sec * 1000.0);
    D("[DUMP] worker=0 seq=%lu slots active=%d cap=%d sessions=%lu steps=%lu deltas=%lu eous=%lu audio_s=%.1f\n",
      n, g.active, g.cap, g.sessions, g.steps, g.deltas, g.eous, g.audio_seconds);
    D("[DUMP] worker=0 seq=%lu cohorts=%lu lanes_mean=%.2f wait_ms_mean=%.1f step_wall_ms_mean=%.2f decode_iters=%lu\n",
      n, g.cohorts, g.cohorts ? (double)g.cohort_lanes_sum / (double)g.cohorts : 0.0,
      g.cohorts ? g.cohort_wait_ms_sum / (double)g.cohorts : 0.0,
      es.steps ? es.step_wall_ms_sum / (double)es.steps : 0.0, es.decode_iters);
    D("[DUMP] worker=0 seq=%lu loop stage_ahead=%d host_threads=%d windows=%lu window_ms_mean=%.2f staged_in_window=%lu "
      "submit_ms_mean=%.2f finish_ms_mean=%.2f graph_passes=%lu eager_passes=%lu host_mel_ms=%.1f\n",
      n, g.stage_ahead, g.host_threads, g.windows, g.windows ? g.window_ms_sum / (double)g.windows : 0.0, g.window_staged,
      es.steps ? es.submit_ms_sum / (double)es.steps : 0.0, es.steps ? es.finish_ms_sum / (double)es.steps : 0.0,
      es.graph_passes, es.eager_passes, es.host_mel_ms);
    if (es.prof_passes > 0) {
        double tot = 0.0;
        for (int i = 0; i < ASR_PROF_STAGES; i++) tot += es.prof_ms[i];
        D("[DUMP] worker=0 seq=%lu profile passes=%lu device_ms=%.1f host_mel_ms=%.1f", n, es.prof_passes, tot, es.host_mel_ms);
        for (int i = 0; i < ASR_PROF_STAGES; i++) D(" %s=%.1f", ASR_PROF_NAME[i], es.prof_ms[i]);
        D("\n");
    }
    D("[DUMP] worker=0 seq=%lu books sessions=%lu completed=%lu cancelled=%lu aborted=%lu active=%d balanced=%d abandoned=%lu recovered=%lu\n",
      n, g.sessions, g.completed, g.cancelled, g.aborted, g.active, balanced, g.abandoned, g.abandoned_recovered);
    D("[DUMP] worker=0 seq=%lu cancelled=%lu", n, g.cancelled);
    for (int i = 0; i < CB__N; i++) D(" %s=%lu", CANCEL_NAME[i], g.cancel_by[i]);
    D("\n[DUMP] worker=0 seq=%lu refused server_at_capacity=%lu other=%lu\n", n, g.refused_cap, g.refused_other);
    unsigned long cnt = 0; for (int i = 0; i < LAG_BUCKETS; i++) cnt += g.lag_hist[i];
    D("[DUMP] worker=0 seq=%lu lag_ms p50=%.0f p95=%.0f max=%.1f count=%lu sum=%.0f bucket_ms=%d\n",
      n, hist_quantile(g.lag_hist, 0.5), hist_quantile(g.lag_hist, 0.95), g.lag_max_ms, cnt, g.lag_sum_ms, LAG_BUCKET_MS);
    {
        /* device-wide (every process on the GPU), from cudaMemGetInfo */
        const double used = (double)f.vram_used / 1048576.0;
        if (used > g.vram_peak_mb) g.vram_peak_mb = used;
        D("[DUMP] worker=0 seq=%lu vram used_mb=%.0f ready_mb=%.0f peak_seen_mb=%.0f total_mb=%.0f accept_backoffs=%lu\n",
          n, used, g.vram_ready_mb, g.vram_peak_mb, (double)f.vram_total / 1048576.0, g.accept_backoffs);
    }
    D("[DUMP] v=1 worker=0 seq=%lu end\n", n);
#undef D
    char *hb = NULL;
    size_t hk = 0;
    if (g.profile_host && (hb = malloc(16384)) != NULL) {
        /* the rusage in g.hp is the ENGINE thread's, refreshed by it each cycle */
        hk = hostprof_report(&g.hp, es.hprof_us, ASR_HPROF_NAME, ASR_HPROF_PHASES, es.hprof_passes,
                             es.hprof_syncs, n, hb, 16384);
    }
    pthread_mutex_unlock(&g.mu);
    fwrite(b, 1, k, stderr);
    if (hb) { fwrite(hb, 1, hk, stderr); free(hb); }
    fflush(stderr);
}

/* ---------------------------------------------------------------- HTTP */
static void serve_metrics_fd(int fd) {
    char req[2048];
    struct pollfd p = {.fd = fd, .events = POLLIN, .revents = 0};
    if (poll(&p, 1, 200) > 0) (void)recv(fd, req, sizeof(req), 0);
    char *body = malloc(65536);
    if (body) {
        const size_t n = metrics_text(body, 65536);
        send_text(fd, 200, "text/plain; version=0.0.4", body, n);
        free(body);
    }
    linger_close(fd, NULL, 0);
}

static void handle_http(int fd) {
    char hdr[REQ_HEAD_MAX + 1];
    size_t got = 0;
    struct timeval tv = {.tv_sec = 5, .tv_usec = 0};
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    while (got < REQ_HEAD_MAX) {
        const ssize_t r = recv(fd, hdr + got, REQ_HEAD_MAX - got, 0);
        if (r <= 0) { close(fd); return; }
        got += (size_t)r;
        hdr[got] = '\0';
        if (strstr(hdr, "\r\n\r\n")) break;
    }
    if (!strstr(hdr, "\r\n\r\n")) { refuse_json(fd, 400, "Bad Request", "invalid_request_error", "bad_request", "request head too large or incomplete", 0); return; }
    char method[8] = "", target[1024] = "";
    if (sscanf(hdr, "%7s %1023s", method, target) != 2) { refuse_json(fd, 400, "Bad Request", "invalid_request_error", "bad_request", "malformed request line", 0); return; }
    char *query = strchr(target, '?');
    if (query) *query++ = '\0';
    const char *path = target;
    if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/health") == 0) {
        cJSON *j = cJSON_CreateObject();
        health_json(j);
        send_json(fd, 200, j);
        cJSON_Delete(j);
        linger_close(fd, NULL, 0);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
        cJSON *j = cJSON_CreateObject();
        cJSON *arr = cJSON_AddArrayToObject(j, "data");
        cJSON *m = cJSON_CreateObject();
        cJSON_AddStringToObject(m, "id", g.facts.model_name);
        cJSON_AddStringToObject(m, "object", "model");
        cJSON_AddStringToObject(m, "owned_by", "mynah");
        cJSON_AddStringToObject(m, "engine", "rnnt");
        cJSON_AddBoolToObject(m, "streaming", 1);
        cJSON_AddBoolToObject(m, "default", 1);
        cJSON_AddItemToArray(arr, m);
        send_json(fd, 200, j);
        cJSON_Delete(j);
        linger_close(fd, NULL, 0);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/metrics") == 0) {
        serve_metrics_fd(fd);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/audio/stream") == 0) {
        if (!handle_ws_stream(fd, hdr, query)) close(fd);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/audio/transcriptions") == 0) {
        refuse_json(fd, 501, "Not Implemented", "invalid_request_error", "not_implemented",
                    "the GPU server serves streams; offline transcription is the CPU server's", 0);
    } else {
        refuse_json(fd, 404, "Not Found", "invalid_request_error", "not_found", "unknown route", 0);
    }
}

/* the HTTP pool: a bounded queue of accepted descriptors */
static struct { int *q; int cap, head, len; pthread_mutex_t mu; pthread_cond_t cv; } g_q;
static void *http_worker(void *arg) {
    (void)arg;
    mynah_asr_thread_set_name("mynah-http");
    for (;;) {
        pthread_mutex_lock(&g_q.mu);
        while (g_q.len == 0 && !atomic_load(&g.shutdown)) pthread_cond_wait(&g_q.cv, &g_q.mu);
        if (g_q.len == 0) { pthread_mutex_unlock(&g_q.mu); return NULL; }
        const int fd = g_q.q[g_q.head];
        g_q.head = (g_q.head + 1) % g_q.cap; g_q.len--;
        pthread_mutex_unlock(&g_q.mu);
        handle_http(fd);
    }
}

static int listen_on(const char *host, int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) != 0 || listen(fd, 512) != 0) { close(fd); return -1; }
    return fd;
}

/* The handler only sets flags: taking g.mu or g_q.mu here would self-deadlock
 * when the signal lands on a thread that already holds one (the qualification
 * harness sends SIGUSR1 under load). The accept loop wakes at least every
 * 250 ms, prints the dump and, on shutdown, broadcasts the condition vars. */
static volatile sig_atomic_t g_dump_req;

static void on_signal(int sig) {
    if (sig == SIGUSR1) { g_dump_req = 1; return; }
    atomic_store(&g.shutdown, 1);
}

/* Every stream holds two descriptors for its life (the socket and the dup the
 * ingest reads), plus the HTTP pool, the listeners, the CUDA driver's own and
 * stdio. The usual soft limit is 1024, so past ~500 streams accept() would
 * fail with EMFILE. Raise the SOFT limit to the HARD one (no privilege
 * needed; the same rule as the CPU server's raise_nofile_limit), halving the
 * ask when a sysctl below the hard limit refuses it, and say what it became. */
static void raise_nofile_limit(void) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) != 0) {
        fprintf(stderr, "mynah-asr-server-cuda: WARNING RLIMIT_NOFILE is unreadable (%s)\n", strerror(errno));
        return;
    }
    const rlim_t had = rl.rlim_cur;
    rlim_t want = rl.rlim_max;
#if defined(__APPLE__)
    if (want == RLIM_INFINITY || want > (rlim_t)OPEN_MAX) want = (rlim_t)OPEN_MAX;
#else
    if (want == RLIM_INFINITY) want = (rlim_t)1048576;
#endif
    for (rlim_t n = want; n > had; n = had + (n - had) / 2) {
        rl.rlim_cur = n;
        if (setrlimit(RLIMIT_NOFILE, &rl) == 0) break;
    }
    struct rlimit now;
    if (getrlimit(RLIMIT_NOFILE, &now) != 0) now = rl;
    const unsigned long long need = 2ull * (unsigned long long)g.cap + (unsigned long long)g.http_threads + 64ull;
    fprintf(stderr, "mynah-asr-server-cuda: RLIMIT_NOFILE soft %llu -> %llu (hard %llu); cap %d needs about %llu\n",
            (unsigned long long)had, (unsigned long long)now.rlim_cur, (unsigned long long)now.rlim_max, g.cap, need);
    if (now.rlim_cur != RLIM_INFINITY && (unsigned long long)now.rlim_cur < need)
        fprintf(stderr, "mynah-asr-server-cuda: WARNING the descriptor ceiling %llu is below what --cap %d needs (~%llu): "
                        "raise the hard limit (ulimit -Hn, LimitNOFILE=, --ulimit nofile) or lower --cap; past it accept() "
                        "fails with EMFILE and the server backs off\n",
                (unsigned long long)now.rlim_cur, g.cap, need);
}

/* The CPUs this process may really use: the affinity mask, capped by a
 * cgroup CPU quota (v2 cpu.max, else v1 cfs) -- a container's mask often
 * names every host CPU while its quota grants a fraction of them. */
static int usable_cpus(void) {
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) n = CPU_COUNT(&set);
#endif
    double quota = -1.0;
    FILE *f = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (f) {
        char q[32] = "";
        double period = 0.0;
        if (fscanf(f, "%31s %lf", q, &period) == 2 && strcmp(q, "max") != 0 && period > 0.0) quota = atof(q) / period;
        fclose(f);
    } else {
        FILE *fq = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r"), *fp = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
        double q = -1.0, period = 0.0;
        if (fq && fp && fscanf(fq, "%lf", &q) == 1 && fscanf(fp, "%lf", &period) == 1 && q > 0.0 && period > 0.0) quota = q / period;
        if (fq) fclose(fq);
        if (fp) fclose(fp);
    }
    if (quota > 0.0 && (int)quota < n) n = (int)quota < 1 ? 1 : (int)quota;
    return n < 1 ? 1 : n;
}

/* --host-threads auto: no team up to 8 usable CPUs (the engine thread, the
 * ingest and the writers need them), 2 up to 16, 4 above */
static int host_threads_auto(int cpus) { return cpus <= 8 ? 1 : cpus <= 16 ? 2 : 4; }

static void usage(void) {
    fprintf(stderr,
        "usage: mynah-asr-server-cuda -m <model_dir> [-p PORT] [--host H] [--engine cuda|cpu]\n"
        "       [--device N] [--cap N] [--cohort-ms MS] [--http-threads N] [--idle-ms MS]\n"
        "       [--ping-ms MS] [--max-frame-bytes N] [--max-audio-seconds S] [--metrics-port P]\n"
        "       [--ring-seconds 30] [--gemm auto|own|own-v2|splitk|own-tc|cublas] [--precision auto|f32|bf16 (bf16 = own-tc only; auto = bf16 own-tc on sm_80+, else f32 own)] [--engine-threads N (cpu engine pool)]\n"
        "       [--kv-dtype f32|bf16|int8 (K/V ring storage; default f32)] [--weights f32|int8]\n"
        "       [--profile-stages (DIAGNOSTIC: per-stage CUDA-event timing)] [--dispatch-map] [--version]\n"
        "       [--profile-host (DIAGNOSTIC: the engine thread's wall per phase, [HOSTP] lines)]\n"
        "       [--pass-lanes N|cap (lanes per encoder pass; default min(cap, 128))]\n"
        "       [--stage-ahead 0|1] [--host-threads N|auto] [--graphs off|buckets] [--graph-buckets 8,16,32,64,128]\n"
        "       [--warmup 0|1]  (cuda engine defaults: 1, auto, buckets, 8,16,32,64,128, 1; cpu engine: off)\n"
        "  --stage-ahead 1: stage and feed the next chunks while the encoder pass runs (byte-identical)\n"
        "  --host-threads: threads (engine thread included) for the host mel of a staged batch; auto =\n"
        "  1 up to 8 usable CPUs, 2 up to 16, 4 above (affinity and cgroup quota)\n"
        "  --graphs buckets: the encoder pass replays CUDA graphs captured at start-up per lane/row bucket\n"
        "  --warmup 1: before listening, run traffic-shaped steps at every lane bucket, then reset\n"
        "  --threads is the HTTP pool (v2 meaning): a WebSocket stream holds one of its threads for its life,\n"
        "  so it is the connection ceiling; default cap + 8.\n");
}

int main(int argc, char **argv) {
    g.host = "0.0.0.0"; g.port = 8291; g.metrics_port = 0; g.device = 0; g.cap = 128;
    g.cohort_ms = 40; g.http_threads = 0; g.idle_ms = 30000; g.ping_ms = 15000;
    g.max_frame_bytes = 1 << 20; g.max_audio_seconds = 0.0; g.engine_name = "cuda";
    /* precision and GEMM resolve below ("auto"); the serving-loop options
     * default on for the cuda engine (S14-13/S14-14: byte-identical to the
     * eager loop, gate C and the C=64 server wave) */
    g.precision = NULL; g.gemm = NULL; g.threads = 0; g.ring_seconds = 30;
    int stage_ahead_set = 0, graphs_set = 0, warmup_set = 0;
    int dispatch_map = 0;
    const double t_start = now_s();
    g.host_threads = 1;
    const char *host_threads_arg = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
#define ARG(name) (strcmp(a, name) == 0 && v && ++i)
        if (ARG("-m") || ARG("--model")) g.model_dir = v;
        else if (ARG("-p") || ARG("--port")) g.port = atoi(v);
        else if (ARG("--host")) g.host = v;
        else if (ARG("--engine")) g.engine_name = v;
        else if (ARG("--device")) g.device = atoi(v);
        else if (ARG("--cap")) g.cap = atoi(v);
        else if (ARG("--cohort-ms")) g.cohort_ms = atoi(v);
        else if (ARG("--http-threads") || ARG("--threads")) g.http_threads = atoi(v);
        else if (ARG("--engine-threads")) g.threads = atoi(v);
        else if (ARG("--ring-seconds")) g.ring_seconds = atoi(v);
        else if (strcmp(a, "--profile-stages") == 0) g.profile = 1;
        else if (strcmp(a, "--profile-host") == 0) g.profile_host = 1;
        else if (ARG("--pass-lanes")) g.pass_lanes = strcmp(v, "cap") == 0 ? -1 : atoi(v);
        else if (ARG("--idle-ms")) g.idle_ms = atoi(v);
        else if (ARG("--ping-ms")) g.ping_ms = atoi(v);
        else if (ARG("--max-frame-bytes")) g.max_frame_bytes = atol(v);
        else if (ARG("--max-audio-seconds")) g.max_audio_seconds = atof(v);
        else if (ARG("--metrics-port")) g.metrics_port = atoi(v);
        else if (ARG("--gemm")) g.gemm = v;
        else if (ARG("--precision")) g.precision = v;
        else if (ARG("--stage-ahead")) { g.stage_ahead = atoi(v); stage_ahead_set = 1; }
        else if (ARG("--host-threads")) host_threads_arg = v;
        else if (ARG("--graphs")) {
            graphs_set = 1;
            if (strcmp(v, "off") == 0) g.graphs = 0;
            else if (strcmp(v, "buckets") == 0) g.graphs = 1;
            else { fprintf(stderr, "--graphs must be off or buckets\n"); return 2; }
        }
        else if (ARG("--graph-buckets")) g.graph_buckets = v;
        else if (ARG("--warmup")) { g.warmup = atoi(v); warmup_set = 1; }
        else if (ARG("--kv-dtype")) g.kv_dtype = v;
        else if (ARG("--weights")) g.weights = v;
        else if (strcmp(a, "--dispatch-map") == 0) dispatch_map = 1;
        else if (strcmp(a, "--version") == 0) { printf("mynah-asr-server-cuda %s\n", MYNAH_ASR_BUILD); return 0; }
        else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) { usage(); return 0; }
        else { fprintf(stderr, "unknown or incomplete option: %s\n", a); usage(); return 2; }
#undef ARG
    }
    if (!g.model_dir) { usage(); return 2; }
    if (g.cap < 1) g.cap = 1;
    if (g.cohort_ms < 0) g.cohort_ms = 0;
    if (g.ring_seconds < 1) g.ring_seconds = 1;
    if (g.http_threads <= 0) g.http_threads = g.cap + 8;
    if (g.pass_lanes < 0) g.pass_lanes = g.cap;   /* --pass-lanes cap */
    const int cuda_engine = strcmp(g.engine_name, "cuda") == 0;
    /* --precision/--gemm auto (the default): the fixed-order bf16 tensor-core
     * GEMM on sm_80+ (S14-8c: batch-invariant, 498-clip bank WER 0.14469 vs
     * 0.14483 for f32, device time per pass -55%), the f32 own GEMM elsewhere,
     * with --weights int8 (an f32-activation arm), or on the cpu engine.
     * An explicit --precision or --gemm is honoured as given. */
    {
        const int auto_prec = g.precision == NULL || strcmp(g.precision, "auto") == 0;
        const int auto_gemm = g.gemm == NULL || strcmp(g.gemm, "auto") == 0;
        const int w8 = g.weights != NULL && strcmp(g.weights, "int8") == 0;
        const int tc = cuda_engine && !w8 && asr_engine_cuda_cc_major(g.device) >= 8 &&
                       (auto_prec || strcmp(g.precision, "bf16") == 0) &&
                       (auto_gemm || strcmp(g.gemm, "own-tc") == 0);
        if (auto_prec) g.precision = tc ? "bf16" : "f32";
        if (auto_gemm) g.gemm = (tc && strcmp(g.precision, "bf16") == 0) ? "own-tc" : "own";
    }
    if (cuda_engine) {
        if (!stage_ahead_set) g.stage_ahead = 1;
        if (!warmup_set) g.warmup = 1;
        /* graphs serve the own GEMMs only, and not under the event profile */
        if (!graphs_set) g.graphs = (strcmp(g.gemm, "cublas") != 0 && !g.profile) ? 1 : 0;
        if (host_threads_arg == NULL) host_threads_arg = "auto";
    } else if (host_threads_arg == NULL) {
        host_threads_arg = "1";
    }
    const int cpus = usable_cpus();
    if (strcmp(host_threads_arg, "auto") == 0) g.host_threads = host_threads_auto(cpus);
    else g.host_threads = atoi(host_threads_arg);
    if (g.host_threads < 1) g.host_threads = 1;
    g.stage_ahead = g.stage_ahead ? 1 : 0;
    g.warmup = g.warmup ? 1 : 0;

    signal(SIGPIPE, SIG_IGN);
    raise_nofile_limit();
    char err[512] = "";
    asr_engine_cfg cfg = {.model_dir = g.model_dir, .cap = g.cap, .device = g.device,
                          .precision = g.precision, .gemm = g.gemm, .threads = g.threads,
                          .profile = g.profile, .profile_host = g.profile_host, .pass_lanes = g.pass_lanes,
                          .graphs = g.graphs, .graph_buckets = g.graph_buckets,
                          .warmup = g.warmup, .host_threads = g.host_threads,
                          .kv_dtype = g.kv_dtype, .weights = g.weights};
    if (strcmp(g.engine_name, "cuda") == 0) g.eng = asr_engine_open_cuda(&cfg, err, sizeof(err));
    else if (strcmp(g.engine_name, "cpu") == 0) g.eng = asr_engine_open_cpu(&cfg, err, sizeof(err));
    else { fprintf(stderr, "mynah-asr-server-cuda: --engine must be cuda or cpu\n"); return 2; }
    if (!g.eng) {
        fprintf(stderr, "mynah-asr-server-cuda: the %s engine did not open: %s\n", g.engine_name, err);
        return strstr(err, "not compiled") ? 78 : 1;
    }
    asr_engine_get_facts(g.eng, &g.facts);
    /* what resolved: the reference engine has no feed team */
    if (g.facts.host_threads < 1) g.facts.host_threads = 1;
    g.host_threads = g.facts.host_threads;
    if (dispatch_map) {
        char buf[4096];
        const size_t n = asr_engine_dispatch_map(g.eng, buf, sizeof(buf));
        fwrite(buf, 1, n, stdout);
        printf("0 row(s) UNKNOWN\n");
        asr_engine_close(g.eng);
        return 0;
    }

    /* slots */
    g.slots = calloc((size_t)g.cap, sizeof(gslot));
    if (!g.slots) return 1;
    for (int i = 0; i < g.cap; i++) {
        gslot *s = &g.slots[i];
        s->id = i; s->state = SLOT_FREE;
        pthread_mutex_init(&s->mu, NULL);
        pthread_cond_init(&s->done_cv, NULL);
        pthread_cond_init(&s->space, NULL);
        s->ring_cap = (size_t)g.ring_seconds * (size_t)g.facts.sample_rate;
        s->ring = malloc(s->ring_cap * sizeof(float));
        if (!s->ring) return 1;
    }
    pthread_mutex_init(&g.mu, NULL);
    pthread_cond_init(&g.wake, NULL);
    g.t0 = now_s();

    g.listen_fd = listen_on(g.host, g.port);
    if (g.listen_fd < 0) { fprintf(stderr, "mynah-asr-server-cuda: cannot listen on %s:%d: %s\n", g.host, g.port, strerror(errno)); return 1; }
    g.metrics_fd = -1;
    if (g.metrics_port > 0) {
        g.metrics_fd = listen_on("127.0.0.1", g.metrics_port);
        if (g.metrics_fd < 0) { fprintf(stderr, "mynah-asr-server-cuda: cannot listen on the metrics port %d: %s\n", g.metrics_port, strerror(errno)); return 1; }
    }

    /* the effective-config banner: what RESOLVED, not what was asked */
    {
        char dm[4096];
        const size_t n = asr_engine_dispatch_map(g.eng, dm, sizeof(dm));
        asr_engine_get_facts(g.eng, &g.facts);   /* VRAM as of now: weights, arena, scratch, context */
        g.vram_ready_mb = g.vram_peak_mb = (double)g.facts.vram_used / 1048576.0;
        fprintf(stderr, "[SERVER-CONFIG] mynah-asr-server-cuda %s: engine=%s device=\"%s\" precision=%s gemm=%s model=%s "
                        "cap=%d cohort_ms=%d lookahead_default=%d presets=%d vram_weights_mb=%.0f vram_arena_mb=%.0f graphs=%d profile_stages=%s "
                        "profile_host=%s pass_lanes=%d vram_used_at_ready_mb=%.0f vram_total_mb=%.0f\n",
                MYNAH_ASR_BUILD, g.facts.name, g.facts.device ? g.facts.device : "-", g.facts.precision, g.facts.gemm,
                g.facts.model_name, g.cap, g.cohort_ms, g.facts.default_lookahead, g.facts.n_lookaheads,
                (double)g.facts.vram_weights / 1048576.0, (double)g.facts.vram_arena / 1048576.0, g.facts.graphs,
                g.profile ? "on (DIAGNOSTIC)" : "off", g.profile_host ? "on (DIAGNOSTIC)" : "off",
                g.facts.pass_lanes, g.vram_ready_mb, (double)g.facts.vram_total / 1048576.0);
        fwrite(dm, 1, n, stderr);
        char topo[1024];
        const size_t tn = host_topology_line(g.facts.pci_bus_id, topo, sizeof(topo));
        fwrite(topo, 1, tn, stderr);
        g.startup_ms = (now_s() - t_start) * 1e3;
        fprintf(stderr, "[SERVER-CONFIG] serving-loop stage_ahead=%d split=%s host_threads=%d usable_cpus=%d graphs=%d "
                        "graph_capture_ms=%.0f warmup=%d warmup_ms=%.0f vram_used_mb=%.0f vram_total_mb=%.0f startup_ms=%.0f\n",
                g.stage_ahead, asr_engine_can_split(g.eng) ? "yes" : "no", g.host_threads, cpus, g.facts.graphs,
                g.facts.graph_capture_ms, g.warmup, g.facts.warmup_ms, (double)g.facts.vram_used / 1048576.0,
                (double)g.facts.vram_total / 1048576.0, g.startup_ms);
        fprintf(stderr, "mynah-asr-server-cuda: listening on %s:%d (%d http threads, %d stream slots, one engine thread)\n",
                g.host, g.port, g.http_threads, g.cap);
        if (g.metrics_fd >= 0) fprintf(stderr, "mynah-asr-server-cuda: /metrics on 127.0.0.1:%d\n", g.metrics_port);
    }

    signal(SIGINT, on_signal); signal(SIGTERM, on_signal); signal(SIGUSR1, on_signal);

    g_q.cap = g.http_threads * 2 + 16;
    g_q.q = calloc((size_t)g_q.cap, sizeof(int));
    pthread_mutex_init(&g_q.mu, NULL); pthread_cond_init(&g_q.cv, NULL);
    pthread_t *pool = calloc((size_t)g.http_threads, sizeof(pthread_t));
    for (int i = 0; i < g.http_threads; i++) pthread_create(&pool[i], NULL, http_worker, NULL);
    pthread_create(&g.engine_thread, NULL, engine_main, NULL);

    while (!atomic_load(&g.shutdown)) {
        struct pollfd p[2] = {{.fd = g.listen_fd, .events = POLLIN, .revents = 0},
                              {.fd = g.metrics_fd, .events = POLLIN, .revents = 0}};
        const int rc = poll(p, g.metrics_fd >= 0 ? 2 : 1, 250);
        if (g_dump_req) { g_dump_req = 0; dump_stderr(); }
        if (rc <= 0) continue;
        if (p[0].revents & POLLIN) {
            const int fd = accept(g.listen_fd, NULL, NULL);
            if (fd < 0) {
                /* Out of descriptors or kernel memory is a capacity condition:
                 * the listener stays readable, so a bare `continue` would spin
                 * poll+accept at 100 % while the clients wait in the backlog
                 * with no word in the log. Say so at most once a second, count
                 * it, back off briefly; the streams in flight free theirs. */
                if (errno == EMFILE || errno == ENFILE || errno == ENOBUFS || errno == ENOMEM) {
                    static double last_note;
                    const double t = now_s();
                    pthread_mutex_lock(&g.mu); g.accept_backoffs++; pthread_mutex_unlock(&g.mu);
                    if (t - last_note >= 1.0) {
                        last_note = t;
                        fprintf(stderr, "mynah-asr-server-cuda: accept: %s; backing off (raise the open-file limit if this repeats)\n",
                                strerror(errno));
                    }
                    const struct timespec pause = {0, 20 * 1000 * 1000};
                    nanosleep(&pause, NULL);
                }
                continue;
            }
            pthread_mutex_lock(&g_q.mu);
            if (g_q.len >= g_q.cap) {
                pthread_mutex_unlock(&g_q.mu);
                /* every HTTP thread busy and the queue full: refuse, visibly */
                refuse_json(fd, 503, "Service Unavailable", "server_error", "server_at_capacity",
                            "every http thread is busy; raise --http-threads", 1);
                continue;
            }
            g_q.q[(g_q.head + g_q.len) % g_q.cap] = fd;
            g_q.len++;
            pthread_cond_signal(&g_q.cv);
            pthread_mutex_unlock(&g_q.mu);
        }
        if (g.metrics_fd >= 0 && (p[1].revents & POLLIN)) {
            const int fd = accept(g.metrics_fd, NULL, NULL);
            if (fd >= 0) serve_metrics_fd(fd);
        }
    }
    fprintf(stderr, "mynah-asr-server-cuda: shutting down\n");
    pthread_mutex_lock(&g_q.mu); pthread_cond_broadcast(&g_q.cv); pthread_mutex_unlock(&g_q.mu);
    pthread_mutex_lock(&g.mu); pthread_cond_broadcast(&g.wake); pthread_mutex_unlock(&g.mu);
    close(g.listen_fd);
    pthread_join(g.engine_thread, NULL);
    pthread_mutex_lock(&g_q.mu); pthread_cond_broadcast(&g_q.cv); pthread_mutex_unlock(&g_q.mu);
    for (int i = 0; i < g.http_threads; i++) pthread_join(pool[i], NULL);
    dump_stderr();
    asr_engine_close(g.eng);
    return 0;
}
