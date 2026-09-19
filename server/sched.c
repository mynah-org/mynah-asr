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

/* What a delta produced by one row of a step needs to know: whose session it
 * belongs to, and when the last sample it consumed arrived. The batched call
 * gets one of these per row as its userdata, so lag is charged per SLOT and is
 * the same number a per-slot feed would have charged. */
typedef struct {
    mynah_asr_slot *slot;
    double arrival;    /* when the last sample consumed by this feed landed */
} emit_ctx;

static struct {
    mynah_asr_sched_config cfg;
    mynah_asr_slot *slots;
    int n_slots;
    int streaming;
    int sample_rate;          /* from the model, not assumed */
    const char *stream_why;   /* why not, for the refusal body */

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

    /* The ready set of ONE step, as mynah_asr_stream_step_batch wants it
     * (S2-2b). Carved with the slot table so building the set is n_slots
     * pointer stores and never a malloc; `b_slot` keeps the owner of each row
     * so a failed pass can cancel exactly the sessions that were in it, and
     * `fin` lists the slots whose turn is the per-slot finalize path. */
    emit_ctx *b_ctx;
    mynah_asr_stream **b_stream;
    const float **b_samples;
    size_t *b_n;
    void **b_ud;
    mynah_asr_slot **b_slot;
    int *fin;
    int batch_reserved;            /* the scratch the library pre-carved, or 0 */
    unsigned long long rows_seen;  /* last mynah_asr_stream_batch_rows_stacked() */

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
    /* S2-2b. What the batched step did, and how long it took: four relaxed adds
     * per step (not per slot), plus one pair indexed by the ready-set size so
     * T_step(B) = a + b*B can be FITTED from a run instead of asserted. */
    _Atomic unsigned long batched_steps, ready_sum, step_wall_us;
    _Atomic unsigned long long rows_stacked;
    _Atomic unsigned long step_b_count[MYNAH_ASR_SCHED_B_BUCKETS];
    _Atomic unsigned long step_b_wall_us[MYNAH_ASR_SCHED_B_BUCKETS];
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
        /* The audio window this text covers, (t0, t1]. Deltas partition the
         * stream, so a client places text in time without keeping a running
         * total of its own. `audio_s` in frame_common is the same t1, kept
         * because every frame type carries it, including the ones with no
         * window (`eou`, `error`). */
        cJSON_AddNumberToObject(j, "t0", res->t0);
        cJSON_AddNumberToObject(j, "t1", res->t1);
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

/* Takes the samples this slot's next chunk needs and files it as one ROW of the
 * step's batch. Nothing is fed here: the whole ready set goes to the model in
 * one call below, which is what makes the 24 conformer layers read their
 * weights once for B streams instead of B times. Returns 1 when a row was
 * filed; `B` is the number of rows already in the set.
 *
 * The accounting is per SLOT and identical to the per-slot feed it replaces:
 * the slot's own arrival record dates the audio, and the delta callback of that
 * row gets that slot's emit_ctx. A transcript, and its lag, never depend on who
 * the chunk travelled with (ENGINEERING.md §9). */
static int sched_stage(mynah_asr_slot *s, size_t avail, int B) {
    size_t need = mynah_asr_stream_need_samples(s->stream);
    if (need == 0) need = 1;
    if (avail < need) return 0;

    const size_t want = need > s->take_cap ? s->take_cap : need;
    emit_ctx *c = &g.b_ctx[B];
    c->slot = s;
    c->arrival = 0.0;
    const size_t got = mynah_asr_slot_take(s, s->take, want, &c->arrival);
    if (got == 0) return 0;
    if (c->arrival > 0.0) s->last_arrival = c->arrival;
    else c->arrival = s->last_arrival;

    s->steps++;
    atomic_fetch_add_explicit(&g.steps, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g.audio_samples, (unsigned long)got,
                              memory_order_relaxed);
    g.b_stream[B] = s->stream;
    g.b_samples[B] = s->take;
    g.b_n[B] = got;
    g.b_ud[B] = c;
    g.b_slot[B] = s;
    return 1;
}

/* ONE call for the whole ready set. B == 1 takes the library's single path
 * verbatim, so a lone stream runs exactly the code it ran before S2-2b.
 *
 * A -1 is a failure of the pass, not of one row: every session in the set is
 * cancelled with `decode_failed`, which is what the per-slot feed did for its
 * one slot. */
static void sched_step_batch(int B) {
    mynah_asr_sched_assert_thread("mynah_asr_stream_step_batch");
    const double t0 = mynah_asr_now();
    const int rc = mynah_asr_stream_step_batch(g.b_stream, B, g.b_samples, g.b_n,
                                               sched_on_result, g.b_ud);
    const unsigned long us = (unsigned long)((mynah_asr_now() - t0) * 1e6 + 0.5);

    /* What the library ACTUALLY stacked, taken as a delta so this worker
     * reports its own steps and not a process-wide total. 0 over a run with
     * traffic means every step degraded to the single path (ENGINEERING.md §6). */
    const unsigned long long rows = mynah_asr_stream_batch_rows_stacked();
    if (rows > g.rows_seen)
        atomic_fetch_add_explicit(&g.rows_stacked, rows - g.rows_seen,
                                  memory_order_relaxed);
    g.rows_seen = rows;

    atomic_fetch_add_explicit(&g.batched_steps, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g.ready_sum, (unsigned long)B, memory_order_relaxed);
    atomic_fetch_add_explicit(&g.step_wall_us, us, memory_order_relaxed);
    const int bb = B < MYNAH_ASR_SCHED_B_BUCKETS ? B : MYNAH_ASR_SCHED_B_BUCKETS - 1;
    atomic_fetch_add_explicit(&g.step_b_count[bb], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g.step_b_wall_us[bb], us, memory_order_relaxed);

    if (rc == 0) return;
    for (int j = 0; j < B; j++) {
        mynah_asr_slot *s = g.b_slot[j];
        g.req_live[s->id] = 0;
        sched_cancel(s, "decode_failed", "the stream step failed");
    }
}

/* The last piece of a finalizing stream, shorter than a chunk. It does NOT go
 * through the batched call: that call takes whole chunks only, and what follows
 * this piece is the tail, which needs the causal right pad only
 * mynah_asr_stream_finish applies. Returns 1 when it fed something. */
static int sched_feed_tail(mynah_asr_slot *s, size_t avail) {
    mynah_asr_sched_assert_thread("mynah_asr_stream_feed");
    if (avail == 0) return 0;
    const size_t want = avail > s->take_cap ? s->take_cap : avail;
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

        /* (3) The ready set. Round-robin so a stream that arrived first does
         * not keep the head of the queue, one chunk per ready slot, and every
         * chunk STAGED rather than fed: the whole set goes to the model in one
         * call below. A slot that is finalizing with less than a chunk left is
         * put aside for the per-slot tail path, which runs after the batch so
         * the order of a slot's own deltas is unchanged. */
        int B = 0, n_fin = 0;
        for (int k = 0; k < g.n_slots; k++) {
            const int i = (rr + k) % g.n_slots;
            if (!g.req_live[i]) continue;
            mynah_asr_slot *s = &g.slots[i];
            const int finalize = (g.req[i] & (MYNAH_ASR_SLOT_REQ_FINALIZE |
                                              MYNAH_ASR_SLOT_REQ_CLOSE)) != 0;
            size_t avail = g.req_avail[i];
            if (avail == 0 && !finalize) continue;

            if (s->stream == NULL || s->needs_reset ||
                s->stream_lookahead != s->lookahead) {
                if (sched_ensure_stream(s) != 0) {
                    sched_cancel(s, "model_not_streaming",
                                 g.stream_why ? g.stream_why
                                              : "this model has no cache-aware streaming");
                    g.req_live[i] = 0;
                    did = 1;
                    continue;
                }
            }

            if (sched_stage(s, avail, B)) {
                B++;
                did = 1;
                /* Keep the finalize pending: the tail runs when the ring is
                 * empty, on a later pass. */
                if (finalize) mynah_asr_slot_request(s, g.req[i] &
                    (MYNAH_ASR_SLOT_REQ_FINALIZE | MYNAH_ASR_SLOT_REQ_CLOSE),
                    NULL, MYNAH_ASR_SLOT_CANCEL_NONE);
                /* MYNAH_ASR_STREAM_BATCH_MAX rows is one call's limit; a larger
                 * ready set is simply more calls in the same step. */
                if (B == MYNAH_ASR_STREAM_BATCH_MAX) {
                    sched_step_batch(B);
                    B = 0;
                }
                continue;
            }
            if (finalize) g.fin[n_fin++] = i;
        }

        /* (4) ONE batched step over the ready set. */
        if (B > 0) sched_step_batch(B);

        /* (5) Finalize: the short last piece, then the tail through
         * mynah_asr_stream_finish, per slot and never through the batch. */
        for (int t = 0; t < n_fin; t++) {
            const int i = g.fin[t];
            if (!g.req_live[i]) continue;
            mynah_asr_slot *s = &g.slots[i];
            const int close_after = (g.req[i] & MYNAH_ASR_SLOT_REQ_CLOSE) != 0;
            const size_t avail = mynah_asr_slot_available(s);
            if (avail > 0) {
                /* The piece goes now and the finalize stays pending: the tail
                 * runs on a later pass, when the ring is empty. */
                if (sched_feed_tail(s, avail)) did = 1;
                mynah_asr_slot_request(s, g.req[i] &
                    (MYNAH_ASR_SLOT_REQ_FINALIZE | MYNAH_ASR_SLOT_REQ_CLOSE),
                    NULL, MYNAH_ASR_SLOT_CANCEL_NONE);
                continue;
            }
            sched_finalize(s, close_after);
            did = 1;
        }
        rr = (rr + 1) % (g.n_slots > 0 ? g.n_slots : 1);

        /* (6) Offline REST work, at most one batched call per step. It stalls
         * the streams on this worker for its duration; v2.0 accepts that and
         * says so in the banner (S2-6 splits the groups). */
        if (sched_run_jobs()) did = 1;

        /* (7) Nothing ready and nothing queued: park. Never a fixed tick -- the
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

    /* A model with no streaming presets has no stream API at all, and a model
     * WITH presets may still use something the incremental encoder does not
     * implement (mynah_asr_stream_unsupported). Knowing both here is what makes
     * `model_not_streaming` a 400 before the upgrade, with the reason in it,
     * instead of a stream that dies after the 101 -- or, worse, one that runs
     * and is wrong. Read on the calling thread, before the scheduler exists:
     * field reads, not inferences. */
    g.sample_rate = mynah_asr_sample_rate(g.cfg.model);
    if (g.sample_rate <= 0) g.sample_rate = 16000;

    int la[8];
    g.stream_why = NULL;
    if (mynah_asr_lookaheads(g.cfg.model, la) <= 0) {
        g.streaming = 0;
        g.stream_why = "this model is offline-only (no cache-aware streaming presets)";
    } else if ((g.stream_why = mynah_asr_stream_unsupported(g.cfg.model)) != NULL) {
        g.streaming = 0;
    } else {
        g.streaming = 1;
    }

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
    g.b_ctx = calloc((size_t)g.n_slots, sizeof(*g.b_ctx));
    g.b_stream = calloc((size_t)g.n_slots, sizeof(*g.b_stream));
    g.b_samples = calloc((size_t)g.n_slots, sizeof(*g.b_samples));
    g.b_n = calloc((size_t)g.n_slots, sizeof(*g.b_n));
    g.b_ud = calloc((size_t)g.n_slots, sizeof(*g.b_ud));
    g.b_slot = calloc((size_t)g.n_slots, sizeof(*g.b_slot));
    g.fin = (int *)calloc((size_t)g.n_slots, sizeof(int));
    if (!g.slots || !g.req || !g.req_lang || !g.req_reason || !g.req_out ||
        !g.req_avail || !g.req_live || !g.b_ctx || !g.b_stream || !g.b_samples ||
        !g.b_n || !g.b_ud || !g.b_slot || !g.fin)
        return -1;

    const size_t ring = (size_t)g.cfg.ring_seconds * (size_t)g.sample_rate;
    /* One chunk of the widest preset is 8*(lookahead+1)+1 mel frames; 4 s of
     * scratch covers every preset the packs ship with, and a finalize tail is
     * bounded by the ring, not by this. */
    const size_t take = 4u * (size_t)g.sample_rate;
    for (int i = 0; i < g.n_slots; i++)
        if (mynah_asr_slot_init(&g.slots[i], i, ring, take) != 0) return -1;

    /* Pre-carve the batched step's scratch for the whole slot cap, ONCE, here:
     * after this the step path allocates nothing of its own. Called before the
     * scheduler thread exists, so it is still single-owner; a model with no
     * streaming presets has no batch scratch to carve and says so. */
    if (g.streaming) {
        const int cap = g.n_slots > MYNAH_ASR_STREAM_BATCH_MAX
                            ? MYNAH_ASR_STREAM_BATCH_MAX : g.n_slots;
        g.batch_reserved = mynah_asr_stream_batch_reserve(g.cfg.model, cap) == 0
                               ? cap : 0;
        if (g.batch_reserved == 0)
            fprintf(stderr, "mynah-asr-server: the batched-step scratch could not "
                            "be reserved for %d slots; steps will carve it on "
                            "first use\n", cap);
    }

    mynah_asr_slot_set_notify(mynah_asr_sched_wake);
    if (pthread_create(&g.thread, NULL, sched_main, NULL) != 0) return -1;
    g.started = 1;
    return 0;
}

int mynah_asr_sched_streaming(void) { return g.streaming; }

const char *mynah_asr_sched_stream_why(void) { return g.stream_why; }

int mynah_asr_sched_sample_rate(void) { return g.sample_rate > 0 ? g.sample_rate : 16000; }

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
        (double)atomic_load_explicit(&g.audio_samples, memory_order_relaxed) /
        (double)g.sample_rate;
    out->lag_count = out->deltas;
    out->lag_sum_ms =
        (double)atomic_load_explicit(&g.lag_sum_us, memory_order_relaxed) / 1000.0;
    out->lag_max_ms =
        (double)atomic_load_explicit(&g.lag_max_us, memory_order_relaxed) / 1000.0;
    for (int i = 0; i < MYNAH_ASR_LAG_BUCKETS; i++)
        out->lag_hist[i] = atomic_load_explicit(&g.lag_hist[i], memory_order_relaxed);
    out->batched_steps = atomic_load_explicit(&g.batched_steps, memory_order_relaxed);
    out->ready_sum = atomic_load_explicit(&g.ready_sum, memory_order_relaxed);
    out->rows_stacked = atomic_load_explicit(&g.rows_stacked, memory_order_relaxed);
    out->step_wall_ms_sum =
        (double)atomic_load_explicit(&g.step_wall_us, memory_order_relaxed) / 1000.0;
    out->step_wall_count = out->batched_steps;
    for (int i = 0; i < MYNAH_ASR_SCHED_B_BUCKETS; i++) {
        out->step_b_count[i] =
            atomic_load_explicit(&g.step_b_count[i], memory_order_relaxed);
        out->step_b_wall_ms[i] =
            (double)atomic_load_explicit(&g.step_b_wall_us[i], memory_order_relaxed)
            / 1000.0;
    }
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

    /* S2-2b: what the STEP did. `rows_stacked_total` is the library's own
     * count of encoder rows that went through the stacked path; 0 next to a
     * non-zero `batched_steps_total` means every step ran as single steps --
     * the fallback is visible, not silent (ENGINEERING.md §6). `by_b` is the
     * raw material of the cadence law: T_step(B) = a + b*B fitted over the
     * ready-set sizes this worker actually saw. */
    cJSON *bt = cJSON_AddObjectToObject(into, "batch");
    cJSON_AddNumberToObject(bt, "batched_steps_total", (double)st.batched_steps);
    cJSON_AddNumberToObject(bt, "rows_stacked_total", (double)st.rows_stacked);
    cJSON_AddNumberToObject(bt, "ready_sum", (double)st.ready_sum);
    cJSON_AddNumberToObject(bt, "ready_mean",
        st.batched_steps ? (double)st.ready_sum / (double)st.batched_steps : 0.0);
    cJSON_AddNumberToObject(bt, "reserved_slots", g.batch_reserved);
    cJSON *sw = cJSON_AddObjectToObject(bt, "step_wall_ms");
    cJSON_AddNumberToObject(sw, "sum", st.step_wall_ms_sum);
    cJSON_AddNumberToObject(sw, "count", (double)st.step_wall_count);
    cJSON_AddNumberToObject(sw, "mean",
        st.step_wall_count ? st.step_wall_ms_sum / (double)st.step_wall_count : 0.0);
    cJSON *byb = cJSON_AddObjectToObject(bt, "by_b");
    for (int i = 1; i < MYNAH_ASR_SCHED_B_BUCKETS; i++) {
        if (st.step_b_count[i] == 0) continue;
        char key[8];
        snprintf(key, sizeof(key), "%d", i);
        cJSON *e = cJSON_AddObjectToObject(byb, key);
        cJSON_AddNumberToObject(e, "steps", (double)st.step_b_count[i]);
        cJSON_AddNumberToObject(e, "wall_ms_sum", st.step_b_wall_ms[i]);
        cJSON_AddNumberToObject(e, "wall_ms_mean",
                                st.step_b_wall_ms[i] / (double)st.step_b_count[i]);
    }
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
    free(g.b_ctx); free(g.b_stream); free(g.b_samples); free(g.b_n);
    free(g.b_ud); free(g.b_slot); free(g.fin);
    g.b_ctx = NULL; g.b_stream = NULL; g.b_samples = NULL; g.b_n = NULL;
    g.b_ud = NULL; g.b_slot = NULL; g.fin = NULL;
    g.n_slots = 0;
}
