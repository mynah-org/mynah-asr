/* The scheduler thread. See sched.h for why there is exactly one of it. */
#include "sched.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "http_util.h"
#include "stream_out.h"

#define SCHED_MAX_OFFLINE_BATCH 64

static struct {
    mynah_asr_sched_config cfg;
    mynah_asr_slot *slots;
    int n_slots;
    int streaming;

    pthread_t thread;
    int started;
    pthread_t tid;                 /* the scheduler's own id, for the assert */
    _Atomic int tid_ready;
    _Atomic int stop;

    /* Per-step scratch, carved once: the step path allocates nothing of its own
     * (the per-delta JSON build is the one declared exception). */
    int *req;
    char (*req_lang)[MYNAH_ASR_SLOT_LANG_CAP];
    mynah_asr_slot_cancel *req_reason;
    mynah_asr_stream_out **req_out;
    size_t *req_avail;
    int *req_live;

    /* The one mutex in this module: offline job queue + the wake flag. */
    pthread_mutex_t mu;
    pthread_cond_t wake;           /* the scheduler parks here */
    pthread_cond_t job_done;       /* HTTP threads wait here for their job */
    int woken;
    mynah_asr_offline_job *q_head, *q_tail;
    int q_len;

    /* Counters. Written only by the scheduler, read by any HTTP thread serving
     * /v1/health: relaxed atomics rather than a lock, so a health probe can
     * never queue behind a step. */
    _Atomic unsigned long steps, deltas, eous, cancelled, sessions, offline_jobs;
    _Atomic unsigned long lag_hist[MYNAH_ASR_LAG_BUCKETS];
    _Atomic unsigned long lag_max_us;
    /* S3-3. Two more adds on paths that already do one, and one array indexed
     * by the reason the client was given. `audio_samples` is the throughput
     * unit: seconds of audio this process actually fed to the model, streams
     * and offline jobs alike, which is the only denominator an RTF claim about
     * a SERVER may use. */
    _Atomic unsigned long audio_samples;
    _Atomic unsigned long lag_sum_us;
    _Atomic unsigned long cancel_by[MYNAH_ASR_SCHED_CANCEL__COUNT];
} g;

/* ------------------------------------------------------- cancel buckets */

static const char *const CANCEL_BUCKET_NAME[MYNAH_ASR_SCHED_CANCEL__COUNT] = {
    "idle_timeout", "peer_gone", "frame_too_large", "protocol_error",
    "shutting_down", "audio_limit", "decode_failed", "other"
};

const char *mynah_asr_sched_cancel_bucket_name(int bucket) {
    if (bucket < 0 || bucket >= MYNAH_ASR_SCHED_CANCEL__COUNT) return "other";
    return CANCEL_BUCKET_NAME[bucket];
}

/* The bucket is chosen from the code the CLIENT was sent, so the counter and
 * the error frame can never name two different things. A code with no bucket
 * of its own (`reset_failed`, `model_not_streaming`) lands in `other` rather
 * than growing the label set: cardinality is a contract, see docs/server.md. */
static int cancel_bucket_of(const char *code) {
    if (code == NULL) return MYNAH_ASR_SCHED_CANCEL_OTHER;
    for (int i = 0; i < MYNAH_ASR_SCHED_CANCEL__COUNT - 1; i++)
        if (strcmp(code, CANCEL_BUCKET_NAME[i]) == 0) return i;
    return MYNAH_ASR_SCHED_CANCEL_OTHER;
}

static void count_cancel(const char *code) {
    atomic_fetch_add_explicit(&g.cancelled, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g.cancel_by[cancel_bucket_of(code)], 1,
                              memory_order_relaxed);
}

void mynah_asr_sched_note_cancel(const char *code) { count_cancel(code); }

/* ----------------------------------------------------------- the invariant */

void mynah_asr_sched_assert_thread(const char *where) {
    if (!atomic_load_explicit(&g.tid_ready, memory_order_acquire)) return;
    if (pthread_equal(pthread_self(), g.tid)) return;
#ifdef NDEBUG
    static _Atomic int said;
    int expect = 0;
    if (atomic_compare_exchange_strong(&said, &expect, 1))
        fprintf(stderr, "mynah-asr-server: %s ran off the scheduler thread; "
                        "the single-owner invariant is broken\n", where);
#else
    fprintf(stderr, "mynah-asr-server: %s ran off the scheduler thread\n", where);
    abort();
#endif
}

/* -------------------------------------------------------------- histograms */

static int lag_bucket(double ms) {
    if (ms <= 0.0) return 0;
    const long b = (long)(ms / (double)MYNAH_ASR_LAG_BUCKET_MS);
    return b >= MYNAH_ASR_LAG_BUCKETS - 1 ? MYNAH_ASR_LAG_BUCKETS - 1 : (int)b;
}

static double hist_p50_u(const unsigned *h) {
    unsigned long total = 0;
    for (int i = 0; i < MYNAH_ASR_LAG_BUCKETS; i++) total += h[i];
    if (total == 0) return 0.0;
    unsigned long seen = 0;
    for (int i = 0; i < MYNAH_ASR_LAG_BUCKETS; i++) {
        seen += h[i];
        if (seen * 2 >= total) return (double)i * MYNAH_ASR_LAG_BUCKET_MS;
    }
    return 0.0;
}

/* ------------------------------------------------------------ frame output */

static void sched_enqueue(mynah_asr_slot *s, int opcode, const char *payload,
                          size_t len) {
    if (s->out == NULL) return;
    size_t n = mynah_asr_ws_frame(s->frame, s->frame_cap, opcode, payload, len);
    if (n > 0) {
        (void)mynah_asr_stream_out_enqueue(s->out, s->frame, n);
        return;
    }
    /* Bigger than the slot's scratch. Not the per-chunk path (a delta is a few
     * dozen bytes); one allocation is the honest answer to a huge message. */
    unsigned char *big = (unsigned char *)malloc(len + 16);
    if (big == NULL) return;
    n = mynah_asr_ws_frame(big, len + 16, opcode, payload, len);
    if (n > 0) (void)mynah_asr_stream_out_enqueue(s->out, big, n);
    free(big);
}

static void sched_send_json(mynah_asr_slot *s, cJSON *j) {
    char *str = cJSON_PrintUnformatted(j);
    if (str != NULL) {
        sched_enqueue(s, 0x1, str, strlen(str));
        free(str);
    }
    cJSON_Delete(j);
}

/* Every server frame carries these three, so a client can compute cadence from
 * the wire alone: where it is in the stream, how much audio produced it, and how
 * long that audio waited between arriving and being answered. */
static void frame_common(cJSON *j, mynah_asr_slot *s, double audio_s, double lag_ms) {
    cJSON_AddNumberToObject(j, "seq", (double)(s->seq++));
    cJSON_AddNumberToObject(j, "audio_s", audio_s);
    cJSON_AddNumberToObject(j, "lag_ms", lag_ms);
}

static void sched_error_frame(mynah_asr_slot *s, const char *code, const char *msg) {
    if (s->out == NULL || mynah_asr_stream_out_failed(s->out)) return;
    cJSON *j = cJSON_CreateObject();
    cJSON_AddStringToObject(j, "type", "error");
    cJSON_AddStringToObject(j, "code", code);
    cJSON_AddStringToObject(j, "message", msg);
    frame_common(j, s, 0.0, 0.0);
    sched_send_json(s, j);
}

/* ------------------------------------------------------- the model callback */

typedef struct {
    mynah_asr_slot *slot;
    double arrival;    /* when the last sample consumed by this feed landed */
} emit_ctx;

static void sched_on_result(const mynah_asr_result *res, void *ud) {
    mynah_asr_sched_assert_thread("the stream result callback");
    emit_ctx *c = (emit_ctx *)ud;
    mynah_asr_slot *s = c->slot;
    if (s->out == NULL) return;

    const double now = mynah_asr_now();
    double lag_ms = c->arrival > 0.0 ? (now - c->arrival) * 1000.0 : 0.0;
    if (lag_ms < 0.0) lag_ms = 0.0;

    cJSON *j = cJSON_CreateObject();
    if (res->is_eou) {
        cJSON_AddStringToObject(j, "type", "eou");
        cJSON_AddNumberToObject(j, "t", res->t1);
        s->eous++;
        atomic_fetch_add_explicit(&g.eous, 1, memory_order_relaxed);
    } else {
        const char *lang = res->lang && res->lang[0] ? res->lang : s->lang;
        cJSON_AddStringToObject(j, "type", "delta");
        cJSON_AddStringToObject(j, "text", res->text ? res->text : "");
        cJSON_AddBoolToObject(j, "final", res->is_final ? 1 : 0);
        cJSON_AddStringToObject(j, "lang", lang);
        /* v1 fields, kept for one release: tools/eval/ws_client.py and
         * tools/bench/stream_load.py read `text` and `audio_seconds`. */
        cJSON_AddStringToObject(j, "language", lang);
        cJSON_AddNumberToObject(j, "audio_seconds", res->t1);
        if (s->t_first_delta == 0.0) s->t_first_delta = now;
        s->deltas++;
        s->lag_sum_ms += lag_ms;
        if (lag_ms > s->lag_max_ms) s->lag_max_ms = lag_ms;
        s->lag_hist[lag_bucket(lag_ms)]++;
        atomic_fetch_add_explicit(&g.deltas, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g.lag_hist[lag_bucket(lag_ms)], 1,
                                  memory_order_relaxed);
        const unsigned long us = (unsigned long)(lag_ms * 1000.0);
        atomic_fetch_add_explicit(&g.lag_sum_us, us, memory_order_relaxed);
        unsigned long prev = atomic_load_explicit(&g.lag_max_us, memory_order_relaxed);
        while (us > prev &&
               !atomic_compare_exchange_weak_explicit(&g.lag_max_us, &prev, us,
                                                      memory_order_relaxed,
                                                      memory_order_relaxed)) {}
    }
    frame_common(j, s, res->t1, lag_ms);
    sched_send_json(s, j);
}

/* ---------------------------------------------------------- slot lifecycle */

/* The pooled stream. Opened on first use and kept for the life of the slot;
 * a new session only resets it. Re-opened when a client asks for a different
 * lookahead, because that sizes the scratch and reset cannot change it. */
static int sched_ensure_stream(mynah_asr_slot *s) {
    mynah_asr_sched_assert_thread("mynah_asr_stream_open/reset");
    if (s->stream != NULL && s->stream_lookahead == s->lookahead) {
        if (s->needs_reset) {
            if (mynah_asr_stream_reset(s->stream, s->lang) != 0) return -1;
            s->needs_reset = 0;
        }
        return 0;
    }
    if (s->stream != NULL) {
        mynah_asr_stream_close(s->stream);
        s->stream = NULL;
    }
    s->stream = mynah_asr_stream_open(g.cfg.model, s->lang, s->lookahead);
    if (s->stream == NULL) return -1;
    s->stream_lookahead = s->lookahead;
    s->needs_reset = 0;
    return 0;
}

/* Hands the session back: the writer drains what is queued, closes the socket
 * and goes; the ingest thread sees DONE and lets the slot go FREE. The stream
 * object stays with the slot -- that is what "pooled" means. */
static void sched_close_session(mynah_asr_slot *s) {
    sched_enqueue(s, 0x8, "", 0);
    mynah_asr_stream_out_finish(s->out);
    mynah_asr_slot_set_state(s, MYNAH_ASR_SLOT_DONE);
}

static void sched_cancel(mynah_asr_slot *s, const char *code, const char *msg) {
    sched_error_frame(s, code, msg);
    count_cancel(code);
    sched_close_session(s);
}

static void sched_emit_done(mynah_asr_slot *s) {
    const char *lang = mynah_asr_stream_lang(s->stream);
    if (lang == NULL || lang[0] == '\0') lang = s->lang;
    cJSON *j = cJSON_CreateObject();
    cJSON_AddStringToObject(j, "type", "done");
    cJSON_AddBoolToObject(j, "done", 1);        /* v1 field, kept for one release */
    cJSON_AddStringToObject(j, "lang", lang);
    cJSON_AddStringToObject(j, "language", lang);
    cJSON_AddNumberToObject(j, "steps", (double)s->steps);
    cJSON_AddNumberToObject(j, "deltas", (double)s->deltas);
    cJSON_AddNumberToObject(j, "lag_p50_ms", hist_p50_u(s->lag_hist));
    cJSON_AddNumberToObject(j, "lag_max_ms", s->lag_max_ms);
    const double audio_s = mynah_asr_stream_audio_seconds(s->stream);
    cJSON_AddNumberToObject(j, "audio_seconds", audio_s);   /* v1 field */
    frame_common(j, s, audio_s, 0.0);
    sched_send_json(s, j);
}

/* --------------------------------------------------------------- one step */

/* Feeds exactly what the stream needs for its next chunk, or the tail on a
 * finalize. Returns 1 when it did work. */
static int sched_feed(mynah_asr_slot *s, size_t avail, int finalize) {
    mynah_asr_sched_assert_thread("mynah_asr_stream_feed");
    size_t need = mynah_asr_stream_need_samples(s->stream);
    if (need == 0) need = 1;

    if (avail >= need) {
        size_t want = need > s->take_cap ? s->take_cap : need;
        emit_ctx ctx = {.slot = s, .arrival = 0.0};
        const size_t got = mynah_asr_slot_take(s, s->take, want, &ctx.arrival);
        if (got == 0) return 0;
        if (ctx.arrival > 0.0) s->last_arrival = ctx.arrival;
        else ctx.arrival = s->last_arrival;
        s->steps++;
        atomic_fetch_add_explicit(&g.steps, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g.audio_samples, (unsigned long)got,
                                  memory_order_relaxed);
        if (mynah_asr_stream_feed(s->stream, s->take, got, sched_on_result, &ctx) != 0) {
            sched_cancel(s, "decode_failed", "the stream step failed");
            return 1;
        }
        return 1;
    }
    if (!finalize) return 0;

    if (avail > 0) {
        /* A last piece shorter than a chunk: hand it over now, the finalize
         * flag survives and the next pass runs the tail through finish(). */
        size_t want = avail > s->take_cap ? s->take_cap : avail;
        emit_ctx ctx = {.slot = s, .arrival = 0.0};
        const size_t got = mynah_asr_slot_take(s, s->take, want, &ctx.arrival);
        if (got == 0) return 0;
        if (ctx.arrival > 0.0) s->last_arrival = ctx.arrival;
        else ctx.arrival = s->last_arrival;
        s->steps++;
        atomic_fetch_add_explicit(&g.steps, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g.audio_samples, (unsigned long)got,
                                  memory_order_relaxed);
        if (mynah_asr_stream_feed(s->stream, s->take, got, sched_on_result, &ctx) != 0) {
            sched_cancel(s, "decode_failed", "the stream step failed");
            return 1;
        }
        return 1;
    }
    return 0;
}

static void sched_finalize(mynah_asr_slot *s, int close_after) {
    mynah_asr_sched_assert_thread("mynah_asr_stream_finish");
    mynah_asr_slot_set_state(s, MYNAH_ASR_SLOT_FINISHING);
    /* The tail is charged to the last sample that arrived, not to now: a
     * finalization lag measured from the finalize request would hide the wait
     * the audio actually had. */
    emit_ctx ctx = {.slot = s, .arrival = s->last_arrival};
    (void)mynah_asr_stream_finish(s->stream, sched_on_result, &ctx);
    sched_emit_done(s);
    if (close_after) {
        sched_close_session(s);
    } else {
        /* The socket stays open for another utterance; the next audio starts
         * from a reset stream, so a client need not reconnect. */
        s->needs_reset = 1;
        mynah_asr_slot_set_state(s, MYNAH_ASR_SLOT_ACTIVE);
    }
}

/* ------------------------------------------------------------ offline work */

static int sched_run_jobs(void) {
    mynah_asr_offline_job *batch[SCHED_MAX_OFFLINE_BATCH];
    int B = 0;

    pthread_mutex_lock(&g.mu);
    while (g.q_head != NULL && B < g.cfg.max_batch && B < SCHED_MAX_OFFLINE_BATCH) {
        /* A batch is one kind, one lookahead: the batched call resolves the
         * lookahead once for the whole batch, so mixing them would make a
         * request's text depend on who it travelled with. */
        if (B > 0 && (g.q_head->kind != batch[0]->kind ||
                      g.q_head->lookahead != batch[0]->lookahead)) break;
        mynah_asr_offline_job *j = g.q_head;
        g.q_head = j->next;
        if (g.q_head == NULL) g.q_tail = NULL;
        g.q_len--;
        j->next = NULL;
        batch[B++] = j;
        if (j->kind == MYNAH_ASR_JOB_DETECT_LANG) break;   /* one at a time */
    }
    pthread_mutex_unlock(&g.mu);
    if (B == 0) return 0;

    if (batch[0]->kind == MYNAH_ASR_JOB_DETECT_LANG) {
        mynah_asr_sched_assert_thread("mynah_asr_detect_lang");
        mynah_asr_offline_job *j = batch[0];
        j->detected[0] = '\0';
        j->rc = (g.cfg.lid != NULL &&
                 mynah_asr_detect_lang(g.cfg.lid, j->samples, j->n_samples,
                                       j->detected) == 0) ? 0 : -1;
    } else {
        mynah_asr_sched_assert_thread("mynah_asr_transcribe_batch_ts");
        const float *samples[SCHED_MAX_OFFLINE_BATCH];
        size_t ns[SCHED_MAX_OFFLINE_BATCH];
        const char *langs[SCHED_MAX_OFFLINE_BATCH];
        char *texts[SCHED_MAX_OFFLINE_BATCH];
        char louts[SCHED_MAX_OFFLINE_BATCH][16];
        mynah_asr_word *wordsv[SCHED_MAX_OFFLINE_BATCH];
        int nwordsv[SCHED_MAX_OFFLINE_BATCH];
        for (int b = 0; b < B; b++) {
            samples[b] = batch[b]->samples;
            ns[b] = batch[b]->n_samples;
            langs[b] = batch[b]->lang;
            texts[b] = NULL;
        }
        /* Words are always extracted: negligible next to inference, and a batch
         * can mix json and verbose_json requests. */
        mynah_asr_transcribe_batch_ts(g.cfg.model, samples, ns, B, langs,
                                      batch[0]->lookahead, texts, louts,
                                      wordsv, nwordsv);
        for (int b = 0; b < B; b++) {
            batch[b]->text = texts[b];
            batch[b]->words = wordsv[b];
            batch[b]->n_words = nwordsv[b];
            memcpy(batch[b]->lang_out, louts[b], sizeof(batch[b]->lang_out));
            batch[b]->rc = texts[b] != NULL ? 0 : -1;
        }
    }

    pthread_mutex_lock(&g.mu);
    for (int b = 0; b < B; b++) batch[b]->done = 1;
    atomic_fetch_add_explicit(&g.offline_jobs, (unsigned long)B, memory_order_relaxed);
    for (int b = 0; b < B; b++)
        atomic_fetch_add_explicit(&g.audio_samples, (unsigned long)batch[b]->n_samples,
                                  memory_order_relaxed);
    pthread_cond_broadcast(&g.job_done);
    pthread_mutex_unlock(&g.mu);
    return 1;
}

/* ------------------------------------------------------------- the thread */

static void sched_drain_on_stop(void) {
    for (int i = 0; i < g.n_slots; i++) {
        mynah_asr_slot *s = &g.slots[i];
        mynah_asr_stream_out *out = NULL;
        const mynah_asr_slot_state st = mynah_asr_slot_poll(s, &out, NULL);
        if (st == MYNAH_ASR_SLOT_FREE || st == MYNAH_ASR_SLOT_DONE || out == NULL) continue;
        sched_cancel(s, mynah_asr_slot_cancel_code(MYNAH_ASR_SLOT_CANCEL_SHUTDOWN),
                     "the server is shutting down");
    }
    pthread_mutex_lock(&g.mu);
    for (mynah_asr_offline_job *j = g.q_head; j != NULL; j = j->next) {
        j->rc = -1;
        j->done = 1;
    }
    g.q_head = g.q_tail = NULL;
    g.q_len = 0;
    pthread_cond_broadcast(&g.job_done);
    pthread_mutex_unlock(&g.mu);
}

static void *sched_main(void *arg) {
    (void)arg;
    g.tid = pthread_self();
    atomic_store_explicit(&g.tid_ready, 1, memory_order_release);
    mynah_asr_thread_set_name("mynah-sched");

    int rr = 0;
    while (!atomic_load_explicit(&g.stop, memory_order_acquire)) {
        pthread_mutex_lock(&g.mu);
        g.woken = 0;    /* anything signalled from here on re-arms the loop */
        pthread_mutex_unlock(&g.mu);

        int did = 0;

        /* (1) Cancellation, before any other policy. A slot the client has
         * abandoned must not pay for a finish it will never read. */
        for (int i = 0; i < g.n_slots; i++) {
            mynah_asr_slot *s = &g.slots[i];
            g.req_live[i] = 0;
            const mynah_asr_slot_state st =
                mynah_asr_slot_poll(s, &g.req_out[i], &g.req_avail[i]);
            if ((st != MYNAH_ASR_SLOT_ACTIVE && st != MYNAH_ASR_SLOT_FINISHING) ||
                g.req_out[i] == NULL)
                continue;
            g.req[i] = mynah_asr_slot_take_requests(s, g.req_lang[i], &g.req_reason[i]);
            g.req_live[i] = 1;

            const int finalizing = (g.req[i] & (MYNAH_ASR_SLOT_REQ_FINALIZE |
                                                MYNAH_ASR_SLOT_REQ_CLOSE)) != 0;
            if ((g.req[i] & MYNAH_ASR_SLOT_REQ_CANCEL) != 0) {
                sched_cancel(s, mynah_asr_slot_cancel_code(g.req_reason[i]),
                             "the stream was cancelled");
                g.req_live[i] = 0;
                did = 1;
            } else if (mynah_asr_stream_out_failed(g.req_out[i]) ||
                       /* Skipped while the tail is being flushed: a client that
                        * half-closes after asking to finalize is not gone, it is
                        * waiting for `done`. */
                       (!finalizing && mynah_asr_stream_out_peer_gone(g.req_out[i]))) {
                /* No frame goes out on this one -- the socket is the thing that
                 * failed -- but the reason is still the one the client would
                 * have been told, so it is counted in that bucket. */
                count_cancel(mynah_asr_slot_cancel_code(MYNAH_ASR_SLOT_CANCEL_PEER));
                sched_close_session(s);
                g.req_live[i] = 0;
                did = 1;
            }
        }

        /* (2) Resets, so the feed pass below sees a stream in the state the
         * client asked for. */
        for (int i = 0; i < g.n_slots; i++) {
            if (!g.req_live[i]) continue;
            mynah_asr_slot *s = &g.slots[i];
            if ((g.req[i] & MYNAH_ASR_SLOT_REQ_RESET) == 0) continue;
            if (g.req_lang[i][0] != '\0')
                snprintf(s->lang, sizeof(s->lang), "%s", g.req_lang[i]);
            s->needs_reset = 1;
            s->deltas = s->eous = s->steps = 0;
            s->lag_sum_ms = s->lag_max_ms = 0.0;
            s->t_first_delta = 0.0;
            memset(s->lag_hist, 0, sizeof(s->lag_hist));
            if (sched_ensure_stream(s) != 0) {
                sched_cancel(s, "reset_failed", "the language is not supported");
                g.req_live[i] = 0;
            }
            did = 1;
        }

        /* (3)+(4) The ready set, one chunk per ready slot, round-robin so a
         * stream that arrived first does not keep the head of the queue. */
        for (int k = 0; k < g.n_slots; k++) {
            const int i = (rr + k) % g.n_slots;
            if (!g.req_live[i]) continue;
            mynah_asr_slot *s = &g.slots[i];
            const int finalize = (g.req[i] & (MYNAH_ASR_SLOT_REQ_FINALIZE |
                                              MYNAH_ASR_SLOT_REQ_CLOSE)) != 0;
            const int close_after = (g.req[i] & MYNAH_ASR_SLOT_REQ_CLOSE) != 0;
            size_t avail = g.req_avail[i];
            if (avail == 0 && !finalize) continue;

            if (s->stream == NULL || s->needs_reset ||
                s->stream_lookahead != s->lookahead) {
                if (sched_ensure_stream(s) != 0) {
                    sched_cancel(s, "model_not_streaming",
                                 "this model has no cache-aware streaming");
                    g.req_live[i] = 0;
                    did = 1;
                    continue;
                }
            }
            if (avail > 0 || !finalize) {
                if (sched_feed(s, avail, finalize)) {
                    did = 1;
                    /* Keep the finalize pending: the tail runs when the ring
                     * is empty, on a later pass. */
                    if (finalize) mynah_asr_slot_request(s, g.req[i] &
                        (MYNAH_ASR_SLOT_REQ_FINALIZE | MYNAH_ASR_SLOT_REQ_CLOSE),
                        NULL, MYNAH_ASR_SLOT_CANCEL_NONE);
                    continue;
                }
            }
            if (finalize && mynah_asr_slot_available(s) == 0) {
                sched_finalize(s, close_after);
                did = 1;
            } else if (finalize) {
                mynah_asr_slot_request(s, g.req[i] &
                    (MYNAH_ASR_SLOT_REQ_FINALIZE | MYNAH_ASR_SLOT_REQ_CLOSE),
                    NULL, MYNAH_ASR_SLOT_CANCEL_NONE);
            }
        }
        rr = (rr + 1) % (g.n_slots > 0 ? g.n_slots : 1);

        /* (5) Offline REST work, at most one batched call per step. It stalls
         * the streams on this worker for its duration; v2.0 accepts that and
         * says so in the banner (S2-6 splits the groups). */
        if (sched_run_jobs()) did = 1;

        /* (6) Nothing ready and nothing queued: park. Never a fixed tick -- the
         * first chunk of a new stream runs the moment it lands. */
        pthread_mutex_lock(&g.mu);
        while (!did && !g.woken && g.q_head == NULL &&
               !atomic_load_explicit(&g.stop, memory_order_acquire))
            pthread_cond_wait(&g.wake, &g.mu);
        pthread_mutex_unlock(&g.mu);
    }

    sched_drain_on_stop();
    return NULL;
}

/* --------------------------------------------------------------- the API */

void mynah_asr_sched_wake(void) {
    pthread_mutex_lock(&g.mu);
    g.woken = 1;
    pthread_cond_signal(&g.wake);
    pthread_mutex_unlock(&g.mu);
}

int mynah_asr_sched_start(const mynah_asr_sched_config *cfg) {
    memset(&g, 0, sizeof(g));
    g.cfg = *cfg;
    if (g.cfg.slots < 1) g.cfg.slots = 1;
    if (g.cfg.ring_seconds < 1) g.cfg.ring_seconds = 1;
    if (g.cfg.max_batch < 1) g.cfg.max_batch = 1;
    if (g.cfg.max_pending < 1) g.cfg.max_pending = 1;
    g.n_slots = g.cfg.slots;

    /* A model with no streaming presets has no stream API at all; knowing it
     * here is what makes `model_not_streaming` a 400 before the upgrade. Read
     * from the config on the calling thread, before the scheduler exists: it
     * is a field read, not an inference. */
    int la[8];
    g.streaming = mynah_asr_lookaheads(g.cfg.model, la) > 0;

    if (pthread_mutex_init(&g.mu, NULL) != 0) return -1;
    if (pthread_cond_init(&g.wake, NULL) != 0) return -1;
    if (pthread_cond_init(&g.job_done, NULL) != 0) return -1;

    g.slots = (mynah_asr_slot *)calloc((size_t)g.n_slots, sizeof(*g.slots));
    g.req = (int *)calloc((size_t)g.n_slots, sizeof(int));
    g.req_lang = calloc((size_t)g.n_slots, MYNAH_ASR_SLOT_LANG_CAP);
    g.req_reason = calloc((size_t)g.n_slots, sizeof(*g.req_reason));
    g.req_out = calloc((size_t)g.n_slots, sizeof(*g.req_out));
    g.req_avail = calloc((size_t)g.n_slots, sizeof(*g.req_avail));
    g.req_live = (int *)calloc((size_t)g.n_slots, sizeof(int));
    if (!g.slots || !g.req || !g.req_lang || !g.req_reason || !g.req_out ||
        !g.req_avail || !g.req_live)
        return -1;

    const size_t ring = (size_t)g.cfg.ring_seconds * 16000u;
    /* One chunk of the widest preset is 8*(lookahead+1)+1 mel frames; 4 s of
     * scratch covers every preset the packs ship with, and a finalize tail is
     * bounded by the ring, not by this. */
    const size_t take = 4u * 16000u;
    for (int i = 0; i < g.n_slots; i++)
        if (mynah_asr_slot_init(&g.slots[i], i, ring, take) != 0) return -1;

    mynah_asr_slot_set_notify(mynah_asr_sched_wake);
    if (pthread_create(&g.thread, NULL, sched_main, NULL) != 0) return -1;
    g.started = 1;
    return 0;
}

int mynah_asr_sched_streaming(void) { return g.streaming; }

mynah_asr_slot *mynah_asr_sched_claim(const char *lang, int lookahead) {
    const double now = mynah_asr_now();
    for (int i = 0; i < g.n_slots; i++) {
        if (mynah_asr_slot_claim(&g.slots[i], lang, lookahead, NULL, now) == 0) {
            atomic_fetch_add_explicit(&g.sessions, 1, memory_order_relaxed);
            return &g.slots[i];
        }
    }
    return NULL;
}

int mynah_asr_sched_submit(mynah_asr_offline_job *job) {
    job->done = 0;
    job->rc = 0;
    job->next = NULL;
    pthread_mutex_lock(&g.mu);
    if (atomic_load_explicit(&g.stop, memory_order_acquire) ||
        g.q_len >= g.cfg.max_pending) {
        pthread_mutex_unlock(&g.mu);
        return -2;   /* refused: the caller answers 503, not 400 */
    }
    if (g.q_tail != NULL) g.q_tail->next = job;
    else g.q_head = job;
    g.q_tail = job;
    g.q_len++;
    g.woken = 1;
    pthread_cond_signal(&g.wake);
    while (!job->done) pthread_cond_wait(&g.job_done, &g.mu);
    pthread_mutex_unlock(&g.mu);
    return job->rc;
}

int mynah_asr_sched_active(void) {
    int n = 0;
    for (int i = 0; i < g.n_slots; i++)
        if (mynah_asr_slot_get_state(&g.slots[i]) != MYNAH_ASR_SLOT_FREE) n++;
    return n;
}

/* --------------------------------------------------------- the snapshot */

void mynah_asr_sched_stats_read(mynah_asr_sched_stats *out) {
    memset(out, 0, sizeof(*out));
    out->slots_active = mynah_asr_sched_active();
    out->slots_cap = g.n_slots;
    out->streaming = g.streaming;
    out->steps    = atomic_load_explicit(&g.steps, memory_order_relaxed);
    out->deltas   = atomic_load_explicit(&g.deltas, memory_order_relaxed);
    out->eous     = atomic_load_explicit(&g.eous, memory_order_relaxed);
    out->sessions = atomic_load_explicit(&g.sessions, memory_order_relaxed);
    out->cancelled = atomic_load_explicit(&g.cancelled, memory_order_relaxed);
    out->offline_done = atomic_load_explicit(&g.offline_jobs, memory_order_relaxed);
    for (int i = 0; i < MYNAH_ASR_SCHED_CANCEL__COUNT; i++)
        out->cancel_by[i] = atomic_load_explicit(&g.cancel_by[i], memory_order_relaxed);
    out->audio_seconds =
        (double)atomic_load_explicit(&g.audio_samples, memory_order_relaxed) / 16000.0;
    out->lag_count = out->deltas;
    out->lag_sum_ms =
        (double)atomic_load_explicit(&g.lag_sum_us, memory_order_relaxed) / 1000.0;
    out->lag_max_ms =
        (double)atomic_load_explicit(&g.lag_max_us, memory_order_relaxed) / 1000.0;
    for (int i = 0; i < MYNAH_ASR_LAG_BUCKETS; i++)
        out->lag_hist[i] = atomic_load_explicit(&g.lag_hist[i], memory_order_relaxed);
    pthread_mutex_lock(&g.mu);
    out->offline_pending = g.q_len;
    out->offline_max_pending = g.cfg.max_pending;
    pthread_mutex_unlock(&g.mu);
}

double mynah_asr_sched_lag_quantile(const unsigned long *hist, double q) {
    unsigned long total = 0;
    for (int i = 0; i < MYNAH_ASR_LAG_BUCKETS; i++) total += hist[i];
    if (total == 0) return 0.0;
    const double want = (double)total * q;
    unsigned long seen = 0;
    for (int i = 0; i < MYNAH_ASR_LAG_BUCKETS; i++) {
        seen += hist[i];
        if ((double)seen >= want) return (double)i * MYNAH_ASR_LAG_BUCKET_MS;
    }
    return (double)(MYNAH_ASR_LAG_BUCKETS - 1) * MYNAH_ASR_LAG_BUCKET_MS;
}

unsigned long mynah_asr_sched_lag_over(const unsigned long *hist, int ms) {
    if (ms <= 0) {
        unsigned long all = 0;
        for (int i = 0; i < MYNAH_ASR_LAG_BUCKETS; i++) all += hist[i];
        return all;
    }
    int first = ms / MYNAH_ASR_LAG_BUCKET_MS;
    if (ms % MYNAH_ASR_LAG_BUCKET_MS != 0) first++;   /* the caller rounded up */
    if (first >= MYNAH_ASR_LAG_BUCKETS) return hist[MYNAH_ASR_LAG_BUCKETS - 1];
    unsigned long n = 0;
    for (int i = first; i < MYNAH_ASR_LAG_BUCKETS; i++) n += hist[i];
    return n;
}

void mynah_asr_sched_health(cJSON *into) {
    /* FACTS, from one snapshot: what this worker DID, not what it was
     * configured to do. The configuration is on the [SERVER-CONFIG] banner
     * line, which is printed once and never has to be kept in sync with a
     * counter. */
    mynah_asr_sched_stats st;
    mynah_asr_sched_stats_read(&st);

    cJSON *sl = cJSON_AddObjectToObject(into, "slots");
    cJSON_AddNumberToObject(sl, "active", st.slots_active);
    cJSON_AddNumberToObject(sl, "cap", st.slots_cap);
    cJSON_AddNumberToObject(into, "steps", (double)st.steps);
    cJSON_AddNumberToObject(into, "deltas", (double)st.deltas);
    cJSON_AddNumberToObject(into, "eous", (double)st.eous);
    cJSON_AddNumberToObject(into, "sessions", (double)st.sessions);
    cJSON_AddNumberToObject(into, "cancelled", (double)st.cancelled);
    cJSON *cb = cJSON_AddObjectToObject(into, "cancelled_by");
    for (int i = 0; i < MYNAH_ASR_SCHED_CANCEL__COUNT; i++)
        cJSON_AddNumberToObject(cb, mynah_asr_sched_cancel_bucket_name(i),
                                (double)st.cancel_by[i]);
    cJSON_AddNumberToObject(into, "offline_jobs", (double)st.offline_done);
    cJSON_AddNumberToObject(into, "offline_pending", st.offline_pending);
    cJSON *off = cJSON_AddObjectToObject(into, "offline");
    cJSON_AddNumberToObject(off, "queued", st.offline_pending);
    cJSON_AddNumberToObject(off, "done", (double)st.offline_done);
    cJSON_AddNumberToObject(off, "max_pending", st.offline_max_pending);
    cJSON_AddNumberToObject(into, "audio_seconds", st.audio_seconds);

    cJSON *lag = cJSON_AddObjectToObject(into, "lag_ms");
    cJSON_AddNumberToObject(lag, "p50", mynah_asr_sched_lag_quantile(st.lag_hist, 0.50));
    cJSON_AddNumberToObject(lag, "p95", mynah_asr_sched_lag_quantile(st.lag_hist, 0.95));
    cJSON_AddNumberToObject(lag, "max", st.lag_max_ms);
    cJSON_AddNumberToObject(lag, "count", (double)st.lag_count);
    cJSON_AddNumberToObject(lag, "bucket_ms", MYNAH_ASR_LAG_BUCKET_MS);
    /* v1 field names, kept for one release: docs/server.md and the bench
     * harness read them flat. */
    cJSON_AddNumberToObject(into, "lag_p50_ms",
                            mynah_asr_sched_lag_quantile(st.lag_hist, 0.50));
    cJSON_AddNumberToObject(into, "lag_max_ms", st.lag_max_ms);
    cJSON_AddBoolToObject(into, "streaming", st.streaming ? 1 : 0);
}

void mynah_asr_sched_stop(void) {
    if (!g.started) return;
    atomic_store_explicit(&g.stop, 1, memory_order_release);
    mynah_asr_sched_wake();
    pthread_join(g.thread, NULL);
    g.started = 0;
    mynah_asr_slot_set_notify(NULL);

    /* The ingest threads are detached and hold pointers into this table. Give
     * them a bounded moment to see their slot DONE and let go; if one is still
     * in there, leave the table allocated. A leak at exit is a nuisance, a
     * table freed under a live reader is a crash. */
    for (int tries = 0; tries < 100 && mynah_asr_sched_active() > 0; tries++) {
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 10 * 1000000L};
        nanosleep(&ts, NULL);
    }
    if (mynah_asr_sched_active() > 0) {
        fprintf(stderr, "mynah-asr-server: %d stream slot(s) still held at exit; "
                        "leaving the table mapped\n", mynah_asr_sched_active());
        return;
    }
    for (int i = 0; i < g.n_slots; i++) mynah_asr_slot_destroy(&g.slots[i]);
    free(g.slots); g.slots = NULL;
    free(g.req); free(g.req_lang); free(g.req_reason);
    free(g.req_out); free(g.req_avail); free(g.req_live);
    g.req = NULL; g.req_lang = NULL; g.req_reason = NULL;
    g.req_out = NULL; g.req_avail = NULL; g.req_live = NULL;
    g.n_slots = 0;
}
