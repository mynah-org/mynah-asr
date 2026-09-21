/* The scheduler thread. See sched.h for why there is exactly one of it. */
#include "sched.h"

#include <errno.h>
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
/* Delay histogram: 100 us steps to 10 ms, 10 ms to 1 s, 100 ms to 10 s, then
 * one overflow bucket. 290 buckets, 2.3 KiB per histogram per worker. */
#define SCHED_DLY_BUCKETS 290

/* Cost and delay as a function of WHERE a row is inside its own utterance,
 * one bin per step (one chunk) for the first 40, then an overflow.
 *
 * This exists because a 7 s corpus and a 24 s corpus differ in two ways at
 * once -- how often the fixed finalization is paid, and how far a stream gets
 * from its own start -- and an aggregate cannot tell them apart. Position is
 * counted in STEPS, not seconds, because the chunk period is config-driven
 * (rule 1) and a seconds axis would bake the lookahead preset into the
 * instrumentation. s->steps is reset with the stream, so this is position
 * inside an utterance, which is the scale the encoder's left cache lives on. */
#define SCHED_POS_BINS 41

static int sched_dly_bucket(double sec) {
    const double us = sec * 1e6;
    if (!(us > 0.0)) return 0;
    if (us < 10000.0) return (int)(us / 100.0);
    if (us < 1000000.0) return 100 + (int)((us - 10000.0) / 10000.0);
    if (us < 10000000.0) return 199 + (int)((us - 1000000.0) / 100000.0);
    return SCHED_DLY_BUCKETS - 1;
}

/* The upper edge of a bucket, in ms: a percentile read off a histogram is
 * reported as the worst value the bucket can hold, never the best. */
static double sched_dly_upper_ms(int b) {
    if (b < 100) return (double)(b + 1) * 0.1;
    if (b < 199) return 10.0 + (double)(b - 100 + 1) * 10.0;
    if (b < SCHED_DLY_BUCKETS - 1) return 1000.0 + (double)(b - 199 + 1) * 100.0;
    return 10000.0;
}

static double sched_dly_pct(const unsigned long *h, double q) {
    unsigned long n = 0;
    for (int i = 0; i < SCHED_DLY_BUCKETS; i++) n += h[i];
    if (n == 0) return 0.0;
    const double want = q * (double)n;
    unsigned long c = 0;
    for (int i = 0; i < SCHED_DLY_BUCKETS; i++) {
        c += h[i];
        if ((double)c >= want) return sched_dly_upper_ms(i);
    }
    return sched_dly_upper_ms(SCHED_DLY_BUCKETS - 1);
}

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
    /* ready -> execution, bucketed. Three linear tiers rather than a log, so
     * the numbers a reader cares about keep their resolution: 100 us up to
     * 10 ms (a scheduler pass), 10 ms up to 1 s (a chunk period is 320), then
     * 100 ms up to 10 s (a stall). A log histogram would report the knee's
     * p95 to the nearest factor of two, which is the difference between
     * "inside the envelope" and "outside it". */
    unsigned long pos_rows[SCHED_POS_BINS];
    double pos_step_s[SCHED_POS_BINS];   /* this row's fair share of the step */
    double pos_dly_s[SCHED_POS_BINS];    /* ready -> model start, same rows    */
    unsigned long pos_bsum[SCHED_POS_BINS];
    int *b_pos;                          /* per staged row: its step index     */
    unsigned long h_ready_sel[SCHED_DLY_BUCKETS];
    unsigned long h_sel_start[SCHED_DLY_BUCKETS];
    unsigned long h_ready_start[SCHED_DLY_BUCKETS];
    unsigned long dly_samples;
    double *b_ready, *b_sel;   /* per staged row: when it became ready/selected */
    int *b_d0;                 /* per staged row: its delta count BEFORE the step
                                * (R-2 trace only; tells a blank step from an
                                * emitting one, which no counter did before) */
    /* The readiness predicate, audited against the scheduler's own answer.
     * The three-way split of wall time classifies an idle interval by asking
     * mynah_asr_slot_ready_count() > 0. If that predicate disagrees with what
     * the scheduler actually considers runnable, every conclusion drawn from
     * `runnable_idle` inherits the error, so it is checked rather than
     * trusted. */
    unsigned long pred_passes, pred_false_ready, pred_false_not_ready;
    unsigned long pred_diag_sum, pred_real_sum, pred_bad_passes;
    int *was_real, *was_diag;
    emit_ctx *b_ctx;
    mynah_asr_stream **b_stream;
    const float **b_samples;
    size_t *b_n;
    void **b_ud;
    mynah_asr_slot **b_slot;
    int *fin;
    int *fin_flag;                 /* who must not be made to wait (window) */
    int batch_reserved;            /* the scratch the library pre-carved, or 0 */
    /* Where the scheduler thread IS, and how long it has been there.
     *
     * A stall on 2026-09-20 froze sixteen slots for 28 s with runnable work
     * available, and the per-slot dump could say that execution had stopped but
     * not WHERE. The candidates are not close together -- parked on the condvar
     * with a lost wakeup, looping through the passes staging nothing, inside a
     * long step, or blocked in the offline path -- and they need different
     * fixes. A loop counter plus a phase marker separates them in one dump:
     * a static loop count with phase=park is a lost wakeup, a climbing one is a
     * livelock, and a static count with phase=step is the model. */
    _Atomic unsigned long loops;
    _Atomic int phase;
    _Atomic int phase_slot;
    _Atomic double phase_since;
    /* Wall time and call count per phase, written ONLY by the scheduler thread
     * (the macro runs nowhere else), so plain fields and no atomics: a reader
     * may see a torn double and will be one sample behind, which is the right
     * trade for a counter on a path that runs a few hundred times a second.
     *
     * This is what turns "the box sits at 45%" into an accounting: how much of
     * the wall is model execution, how much is the scheduler's own serial work,
     * and how much is parking with nothing to do -- which is not waste. */
    double w_model, w_model_solo, w_runnable_idle, w_no_work;
    double w_idle_by_phase[MYNAH_ASR_SCHED_PHASES];
    double fin_model_s, fin_rest_s;
    unsigned long fin_calls;
    double phase_wall_s[MYNAH_ASR_SCHED_PHASES];
    unsigned long phase_calls[MYNAH_ASR_SCHED_PHASES];
    /* Parks split by whether work existed. A park with a ready slot behind it
     * would be a scheduling defect; a park with nothing ready is the stream
     * cadence and cannot be optimised away. */
    unsigned long park_idle, park_ready, park_partial;
    double t_start;

    _Atomic unsigned long window_entered, window_filled;
    _Atomic unsigned long window_wait_us;
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

/* One store each, relaxed: this runs on every pass of the scheduler loop and
 * must not become a synchronisation point of its own. */
#define SCHED_PHASE(p) SCHED_PHASE_AT((p), -1)
/* `at` is the slot the pass is working on, so a phase that stops says WHICH
 * stream's lock it stopped on. Pass (1) is three different waits wearing one
 * name -- the slot lock, the peer check, the cancel path -- and the 2026-09-20
 * stall stopped in it for 26 s without saying which. */
#define SCHED_PHASE_AT(p, at) do { \
    const double now_ = mynah_asr_now(); \
    const int prev_ = atomic_load_explicit(&g.phase, memory_order_relaxed); \
    const double since_ = atomic_load_explicit(&g.phase_since, memory_order_relaxed); \
    if (prev_ > 0 && prev_ < MYNAH_ASR_SCHED_PHASES && since_ > 0.0) { \
        const double d_ = now_ - since_; \
        g.phase_wall_s[prev_] += d_; \
        g.phase_calls[prev_]++; \
        /* The accounting that matters, and it does NOT depend on which phase \
         * was running: model busy, or idle of the model with runnable work \
         * available, or idle with nothing to run. Counting only the parks that \
         * had a ready slot would answer a much weaker question -- the scheduler \
         * can spend its time in slot-poll, stage, finalize or cancel while a \
         * chunk sits ready, and that interval is avoidable idle just the same. */ \
        if (prev_ == 4) g.w_model += d_; \
        /* Phase 5 and phase 6 are the model too, and calling them idle was a \
         * real error in this file: finalize measured 99.8 % \
         * mynah_asr_stream_finish, so charging it to `runnable_idle` reported \
         * the model executing a tail as the model doing nothing. It made the \
         * duty read 0.82 at c=40 when model execution was 0.94, and it made \
         * 13-14 % of wall look like reclaimable waste when the reclaimable \
         * part is 1-3 %. They are kept in their OWN bucket rather than folded \
         * into w_model, because what distinguishes them is the thing worth \
         * fixing: this is model work at batch width one, while phase 4 runs \
         * the same model over a ready set averaging 4.7 rows. */ \
        else if (prev_ == 5 || prev_ == 6) g.w_model_solo += d_; \
        else if (mynah_asr_slot_ready_count() > 0) { \
            g.w_runnable_idle += d_; \
            /* WHICH phase is holding the model off work that is ready. The \
             * aggregate says 12-19 % of wall is avoidable idle and the phase \
             * table says finalize is 13-17 % of wall; those are two aggregates \
             * that happen to be close, which is a reason to measure and not a \
             * result. This attributes the interval to the phase that consumed \
             * it, and then the claim is a measurement. */ \
            g.w_idle_by_phase[prev_] += d_; \
        } else g.w_no_work += d_; \
    } \
    atomic_store_explicit(&g.phase, (p), memory_order_relaxed); \
    atomic_store_explicit(&g.phase_slot, (at), memory_order_relaxed); \
    atomic_store_explicit(&g.phase_since, now_, memory_order_relaxed); \
} while (0)

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
    /* AFTER the framing and the enqueue: what separates this from `now` above
     * is cJSON's printing and the memcpy into the output ring, which is
     * exactly the span R-1 found nobody could see. */
    if (s->t_first_queued == 0.0 && s->deltas > 0) s->t_first_queued = mynah_asr_now();
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

/* ---------------------------------------------------- R-2 first-partial trace
 *
 * MYNAH_ASR_TRACE_TTFP=N traces the first N steps of every stream on stderr.
 * It exists because of what R-1 found: between audio arriving and text leaving
 * this process the server published three instants and no more, so the wait in
 * front of the first word could not be attributed to anything. The trace adds
 * no work to an untraced run beyond one relaxed load per step, changes no
 * decision, and prints absolute CLOCK_MONOTONIC seconds so a mark here can be
 * subtracted from a mark taken in the client (which reads the same clock).
 *
 * A step that emits nothing is the point of the exercise: `emitted` is 0 on
 * every step the RNNT spent in blanks, and the line still carries the audio it
 * had consumed by then. */
static int g_trace_steps = -1;

static int sched_trace_steps(void) {
    if (g_trace_steps < 0) {
        const char *e = getenv("MYNAH_ASR_TRACE_TTFP");
        g_trace_steps = (e != NULL && *e != '\0') ? atoi(e) : 0;
        if (g_trace_steps < 0) g_trace_steps = 0;
    }
    return g_trace_steps;
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
    if (sched_trace_steps() > 0) {
        /* The whole server-side chain for this stream, in one line, in the
         * clock the client also reads. t_first_send is the writer's, so it is
         * read here rather than stamped here: by `done` the first delta is
         * long gone. A zero means the stage never happened. */
        fprintf(stderr,
                "[TTFP] slot=%d END open=%.6f first_audio=%.6f first_result=%.6f "
                "first_queued=%.6f first_send=%.6f steps=%d deltas=%d audio_s=%.4f\n",
                s->id, s->t_open, s->t_first_audio, s->t_first_delta,
                s->t_first_queued, mynah_asr_stream_out_first_send(s->out),
                s->steps, s->deltas, audio_s);
    }
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
    /* Publish it: the push and take paths keep this slot's readiness flag from
     * it, which is what makes "is there runnable work right now" one load. */
    mynah_asr_slot_set_need(s, need);
    if (avail < need) return 0;

    const size_t want = need > s->take_cap ? s->take_cap : need;
    emit_ctx *c = &g.b_ctx[B];
    c->slot = s;
    c->arrival = 0.0;
    /* Sampled BEFORE the take: the take itself re-stamps ready_since when it
     * leaves another whole chunk behind, so reading it afterwards would
     * measure zero on exactly the backlogged slots this is meant to expose. */
    const double t_rdy = mynah_asr_slot_ready_since(s);
    const size_t got = mynah_asr_slot_take(s, s->take, want, &c->arrival);
    if (got == 0) return 0;
    if (c->arrival > 0.0) s->last_arrival = c->arrival;
    else c->arrival = s->last_arrival;

    s->steps++;
    s->t_last_step = mynah_asr_now();
    if (t_rdy > 0.0 && s->t_last_step >= t_rdy) {
        g.h_ready_sel[sched_dly_bucket(s->t_last_step - t_rdy)]++;
        g.dly_samples++;
    }
    g.b_ready[B] = t_rdy;
    g.b_sel[B] = s->t_last_step;
    /* s->steps was incremented just above, so this row is step index steps-1
     * of its utterance: the first chunk after a reset is 0. */
    g.b_pos[B] = (int)(s->steps > 0 ? s->steps - 1 : 0);
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

/* Wait, briefly and boundedly, for more streams to become ready.
 *
 * The step is priced as a + b*B and `a` is paid per STEP, not per stream, so a
 * ready set of two costs almost as much as a ready set of sixteen. Clients push
 * their audio on independent phases, so on any given wake only the one or two
 * slots that have just crossed a whole chunk are stageable and the rest are a
 * few tens of milliseconds short. Measured on 24 Neoverse-V2 cores: mean ready
 * set 2.11 at c=8 and 3.73 at c=16, which is the fixed cost paid four times per
 * chunk period instead of once.
 *
 * Three exemptions, and they are what keep this from being a latency tax:
 *   - a stream that has never stepped: its first chunk is what TTFP measures,
 *     and it must never wait behind someone else's cadence;
 *   - a finalizing stream: it is asking to end, not to be batched;
 *   - a set that is already complete: if everyone live is ready there is
 *     nothing to wait for.
 * It also returns the moment the set fills, so on a machine whose streams share
 * a cadence the window costs nothing at all — it is a deadline, not a delay.
 *
 * Refreshes g.req_avail[] as it goes, because availability is what it is
 * waiting on. Returns the number of ready slots it ends up seeing. */
static int sched_collect(const int *finalizing) {
    const int win_ms = g.cfg.batch_window_ms;
    if (win_ms <= 0) return -1;

    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += (long)win_ms * 1000000L;
    deadline.tv_sec += deadline.tv_nsec / 1000000000L;
    deadline.tv_nsec %= 1000000000L;

    const double t0 = mynah_asr_now();
    int entered = 0, ready = 0;
    for (;;) {
        int live = 0, exempt = 0;
        ready = 0;
        for (int i = 0; i < g.n_slots; i++) {
            if (!g.req_live[i]) continue;
            mynah_asr_slot *s = &g.slots[i];
            live++;
            if (finalizing[i] || s->stream == NULL || s->needs_reset || s->steps == 0) {
                exempt = 1;
                continue;
            }
            size_t need = mynah_asr_stream_need_samples(s->stream);
            if (need == 0) need = 1;
            g.req_avail[i] = mynah_asr_slot_available(s);
            if (g.req_avail[i] >= need) ready++;
        }
        /* Nothing to gather, nothing to gather FOR, or someone here must not be
         * made to wait: step now. */
        if (exempt || ready == 0 || ready >= live) break;

        if (!entered) {
            entered = 1;
            atomic_fetch_add_explicit(&g.window_entered, 1, memory_order_relaxed);
        }
        pthread_mutex_lock(&g.mu);
        g.woken = 0;
        const int rc = pthread_cond_timedwait(&g.wake, &g.mu, &deadline);
        pthread_mutex_unlock(&g.mu);
        if (atomic_load_explicit(&g.stop, memory_order_acquire)) break;
        if (rc == ETIMEDOUT) break;
    }
    if (entered) {
        atomic_fetch_add_explicit(&g.window_wait_us,
                                  (unsigned long)((mynah_asr_now() - t0) * 1e6 + 0.5),
                                  memory_order_relaxed);
        /* "Filled" means it ended because the set was complete, not because the
         * clock ran out. The two are the whole diagnosis of the window. */
        int live = 0;
        for (int i = 0; i < g.n_slots; i++) if (g.req_live[i]) live++;
        if (ready >= live && live > 0)
            atomic_fetch_add_explicit(&g.window_filled, 1, memory_order_relaxed);
    }
    return ready;
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
    /* The model starts now, for every row in this set at once. A row staged
     * early in the pass has been waiting for the rest of the set; the last
     * row has not. Both are charged, because what a stream experiences is the
     * whole interval from the instant its chunk was executable. */
    for (int j = 0; j < B; j++) {
        if (g.b_sel[j] > 0.0 && t0 >= g.b_sel[j])
            g.h_sel_start[sched_dly_bucket(t0 - g.b_sel[j])]++;
        if (g.b_ready[j] > 0.0 && t0 >= g.b_ready[j])
            g.h_ready_start[sched_dly_bucket(t0 - g.b_ready[j])]++;
    }
    /* Before the call, so `emitted` below counts only this step's deltas. */
    const int trace_n = sched_trace_steps();
    if (trace_n > 0)
        for (int j = 0; j < B; j++) g.b_d0[j] = g.b_slot[j]->deltas;

    const int rc = mynah_asr_stream_step_batch(g.b_stream, B, g.b_samples, g.b_n,
                                               sched_on_result, g.b_ud);
    const double t_end = mynah_asr_now();
    const unsigned long us = (unsigned long)((t_end - t0) * 1e6 + 0.5);

    if (trace_n > 0) {
        for (int j = 0; j < B; j++) {
            mynah_asr_slot *s = g.b_slot[j];
            if ((int)s->steps > trace_n) continue;
            fprintf(stderr,
                    "[TTFP] slot=%d step=%d B=%d consumed_s=%.4f arrival=%.6f "
                    "ready=%.6f sel=%.6f mstart=%.6f mend=%.6f emitted=%d\n",
                    s->id, (int)s->steps, B,
                    s->stream ? mynah_asr_stream_audio_seconds(s->stream) : 0.0,
                    g.b_ctx[j].arrival, g.b_ready[j], g.b_sel[j], t0, t_end,
                    s->deltas - g.b_d0[j]);
        }
    }

    /* What the library ACTUALLY stacked, taken as a delta so this worker
     * reports its own steps and not a process-wide total. 0 over a run with
     * traffic means every step degraded to the single path (ENGINEERING.md §6). */
    const unsigned long long rows = mynah_asr_stream_batch_rows_stacked();
    if (rows > g.rows_seen)
        atomic_fetch_add_explicit(&g.rows_stacked, rows - g.rows_seen,
                                  memory_order_relaxed);
    g.rows_seen = rows;

    /* The step's wall, split evenly over its rows. Evenly is the only honest
     * split available: the rows go through the stacked layers together and
     * there is no per-row wall to read. It is exact for the mean of a bin once
     * rows of many widths have landed in it, which is what the bin reports. */
    if (B > 0) {
        const double per = (double)us * 1e-6 / (double)B;
        for (int j = 0; j < B; j++) {
            const int pb = g.b_pos[j] < SCHED_POS_BINS - 1 ? g.b_pos[j]
                                                           : SCHED_POS_BINS - 1;
            g.pos_rows[pb]++;
            g.pos_step_s[pb] += per;
            g.pos_bsum[pb] += (unsigned long)B;
            if (g.b_ready[j] > 0.0 && t0 >= g.b_ready[j])
                g.pos_dly_s[pb] += t0 - g.b_ready[j];
        }
    }
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
    /* Finalization is the largest serial phase at the knee -- 0.354 of wall at
     * c=36 on the 3x8 capacity curve -- and it is two very different costs
     * wearing one name: the model running the tail with its causal right pad,
     * and the teardown around it (the done frame, the close, the session
     * bookkeeping). They call for opposite fixes, so they are timed apart. */
    const double t_fin0 = mynah_asr_now();
    (void)mynah_asr_stream_finish(s->stream, sched_on_result, &ctx);
    const double t_fin1 = mynah_asr_now();
    g.fin_model_s += t_fin1 - t_fin0;
    g.fin_calls++;
    sched_emit_done(s);
    if (close_after) {
        sched_close_session(s);
    } else {
        /* The socket stays open for another utterance; the next audio starts
         * from a reset stream, so a client need not reconnect. */
        s->needs_reset = 1;
        mynah_asr_slot_set_state(s, MYNAH_ASR_SLOT_ACTIVE);
    }
    g.fin_rest_s += mynah_asr_now() - t_fin1;
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
    g.t_start = mynah_asr_now();
    atomic_store_explicit(&g.tid_ready, 1, memory_order_release);
    mynah_asr_thread_set_name("mynah-sched");

    int rr = 0;
    while (!atomic_load_explicit(&g.stop, memory_order_acquire)) {
        pthread_mutex_lock(&g.mu);
        g.woken = 0;    /* anything signalled from here on re-arms the loop */
        pthread_mutex_unlock(&g.mu);

        int did = 0;
        atomic_fetch_add_explicit(&g.loops, 1, memory_order_relaxed);
        SCHED_PHASE(1);

        /* (1) Cancellation, before any other policy. A slot the client has
         * abandoned must not pay for a finish it will never read. */
        for (int i = 0; i < g.n_slots; i++) {
            mynah_asr_slot *s = &g.slots[i];
            g.req_live[i] = 0;
            SCHED_PHASE_AT(1, i);
            const mynah_asr_slot_state st =
                mynah_asr_slot_poll(s, &g.req_out[i], &g.req_avail[i]);
            if ((st != MYNAH_ASR_SLOT_ACTIVE && st != MYNAH_ASR_SLOT_FINISHING) ||
                g.req_out[i] == NULL)
                continue;
            SCHED_PHASE_AT(8, i);
            g.req[i] = mynah_asr_slot_take_requests(s, g.req_lang[i], &g.req_reason[i]);
            g.req_live[i] = 1;

            const int finalizing = (g.req[i] & (MYNAH_ASR_SLOT_REQ_FINALIZE |
                                                MYNAH_ASR_SLOT_REQ_CLOSE)) != 0;
            /* Covers the whole decision block: the cancel test is a bit
             * check, but peer_gone takes the output ring's lock and
             * sched_cancel writes a frame through it. */
            SCHED_PHASE_AT(9, i);
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

        SCHED_PHASE(2);
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
        /* (2b) The collection window, when one is configured: a bounded wait
         * for the rest of the cadence's streams, so the step's fixed cost is
         * amortised over the set it was priced for. */
        if (g.cfg.batch_window_ms > 0) {
            for (int i = 0; i < g.n_slots; i++)
                g.fin_flag[i] = g.req_live[i] &&
                    (g.req[i] & (MYNAH_ASR_SLOT_REQ_FINALIZE |
                                 MYNAH_ASR_SLOT_REQ_CLOSE)) != 0;
            sched_collect(g.fin_flag);
        }

        SCHED_PHASE(3);
        /* Snapshot the diagnostic predicate BEFORE this pass takes anything:
         * a slot that was ready and got staged is not ready afterwards, and
         * comparing the two sides at different instants would manufacture a
         * disagreement that does not exist. */
        int diag_ready = 0;
        for (int i = 0; i < g.n_slots; i++) {
            g.was_diag[i] = mynah_asr_slot_is_ready(&g.slots[i]);
            g.was_real[i] = 0;
            diag_ready += g.was_diag[i];
        }
        int B = 0, n_fin = 0;
        for (int k = 0; k < g.n_slots; k++) {
            const int i = (rr + k) % g.n_slots;
            if (!g.req_live[i]) continue;
            mynah_asr_slot *s = &g.slots[i];
            const int finalize = (g.req[i] & (MYNAH_ASR_SLOT_REQ_FINALIZE |
                                              MYNAH_ASR_SLOT_REQ_CLOSE)) != 0;
            /* Availability is re-read rather than trusted from the poll in
             * pass (1): the collection window may have waited, and audio that
             * arrived during the wait is exactly what it was waiting for. */
            size_t avail = g.cfg.batch_window_ms > 0 ? mynah_asr_slot_available(s)
                                                     : g.req_avail[i];
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
                g.was_real[i] = 1;
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
            if (finalize) { g.fin[n_fin++] = i; g.was_real[i] = 1; }
        }

        /* The readiness predicate, audited against the scheduler's own answer.
         * Compared PER SLOT, not as two totals: two counts can agree while
         * naming different slots, and it is the slots that matter. A
         * false-not-ready files runnable time under `no_work`, which would
         * make the avoidable idle look smaller than it is; a false-ready does
         * the opposite. Neither is acceptable in a number used to choose what
         * to optimise, so the size of both is reported next to it. */
        {
            int real = 0, fr = 0, fnr = 0;
            for (int i = 0; i < g.n_slots; i++) {
                real += g.was_real[i];
                if (g.was_diag[i] && !g.was_real[i]) fr++;
                else if (!g.was_diag[i] && g.was_real[i]) fnr++;
            }
            g.pred_passes++;
            g.pred_diag_sum += (unsigned long)diag_ready;
            g.pred_real_sum += (unsigned long)real;
            g.pred_false_ready += (unsigned long)fr;
            g.pred_false_not_ready += (unsigned long)fnr;
            if (fr || fnr) g.pred_bad_passes++;
        }

        /* (4) ONE batched step over the ready set. */
        SCHED_PHASE(4);
        if (B > 0) sched_step_batch(B);

        SCHED_PHASE(5);
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
        SCHED_PHASE(6);
        if (sched_run_jobs()) did = 1;

        /* (7) Nothing ready and nothing queued: park. Never a fixed tick -- the
         * first chunk of a new stream runs the moment it lands. */
        /* Was there work when we decided to park? Counted from the availability
         * this pass already read, so it costs nothing extra. `partial` is a slot
         * holding audio that is not yet a whole chunk: real work, not yet
         * runnable, and the honest middle category between the two. */
        if (!did) {
            int ready = 0, partial = 0;
            for (int i = 0; i < g.n_slots; i++) {
                if (!g.req_live[i]) continue;
                mynah_asr_slot *s = &g.slots[i];
                if (s->stream == NULL) continue;
                size_t need = mynah_asr_stream_need_samples(s->stream);
                if (need == 0) need = 1;
                if (g.req_avail[i] >= need) ready++;
                else if (g.req_avail[i] > 0) partial++;
            }
            if (ready > 0) g.park_ready++;
            else if (partial > 0) g.park_partial++;
            else g.park_idle++;
        }
        SCHED_PHASE(7);
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
    g.b_ready = calloc((size_t)g.n_slots, sizeof(*g.b_ready));
    g.b_sel = calloc((size_t)g.n_slots, sizeof(*g.b_sel));
    g.b_d0 = (int *)calloc((size_t)g.n_slots, sizeof(int));
    g.b_pos = (int *)calloc((size_t)g.n_slots, sizeof(int));
    g.was_real = (int *)calloc((size_t)g.n_slots, sizeof(int));
    g.was_diag = (int *)calloc((size_t)g.n_slots, sizeof(int));
    g.b_slot = calloc((size_t)g.n_slots, sizeof(*g.b_slot));
    g.fin = (int *)calloc((size_t)g.n_slots, sizeof(int));
    g.fin_flag = (int *)calloc((size_t)g.n_slots, sizeof(int));
    if (!g.slots || !g.req || !g.req_lang || !g.req_reason || !g.req_out ||
        !g.req_avail || !g.req_live || !g.b_ctx || !g.b_stream || !g.b_samples ||
        !g.b_n || !g.b_ud || !g.b_slot || !g.fin || !g.fin_flag)
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

int mynah_asr_sched_slots_view(mynah_asr_sched_slot_view *out, int max) {
    const double now = mynah_asr_now();
    int n = 0;
    for (int i = 0; i < g.n_slots && n < max; i++) {
        mynah_asr_slot *s = &g.slots[i];
        mynah_asr_stream_out *o = NULL;
        size_t avail = 0;
        const mynah_asr_slot_state st = mynah_asr_slot_poll(s, &o, &avail);
        if (st != MYNAH_ASR_SLOT_ACTIVE && st != MYNAH_ASR_SLOT_FINISHING)
            continue;
        mynah_asr_sched_slot_view *v = &out[n++];
        v->id = i;
        v->state = (int)st;
        v->has_out = o != NULL;
        v->has_stream = s->stream != NULL;
        v->lookahead = s->lookahead;
        v->steps = (unsigned long)s->steps;
        v->deltas = (unsigned long)s->deltas;
        v->ring_samples = avail;
        v->need_samples = s->stream ? mynah_asr_stream_need_samples(s->stream) : 0;
        v->age_s = s->t_open > 0.0 ? now - s->t_open : -1.0;
        v->since_arrival_s = s->last_arrival > 0.0 ? now - s->last_arrival : -1.0;
        v->since_step_s = s->t_last_step > 0.0 ? now - s->t_last_step : -1.0;
        v->ready = s->stream && v->need_samples > 0 && avail >= v->need_samples;
        v->lag_max_ms = s->lag_max_ms;
        v->mu_owner = atomic_load_explicit(&s->mu_owner, memory_order_relaxed);
        {
            const double since = atomic_load_explicit(&s->mu_since, memory_order_relaxed);
            v->mu_held_s = v->mu_owner ? now - since : -1.0;
        }
        v->mu_where = atomic_load_explicit(&s->mu_where, memory_order_relaxed);
        mynah_asr_stream_out_owner(o, &v->out_owner, &v->out_held_s, &v->out_where);
    }
    return n;
}

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
    out->loops = atomic_load_explicit(&g.loops, memory_order_relaxed);
    out->phase = atomic_load_explicit(&g.phase, memory_order_relaxed);
    out->phase_slot = atomic_load_explicit(&g.phase_slot, memory_order_relaxed);
    for (int i = 0; i < MYNAH_ASR_SCHED_PHASES; i++) {
        out->phase_wall_s[i] = g.phase_wall_s[i];
        out->phase_calls[i] = g.phase_calls[i];
    }
    out->w_model = g.w_model;
    out->w_runnable_idle = g.w_runnable_idle;
    out->w_no_work = g.w_no_work;
    out->w_model_solo = g.w_model_solo;
    for (int i = 0; i < SCHED_POS_BINS; i++) {
        out->pos_rows[i] = g.pos_rows[i];
        out->pos_step_s[i] = g.pos_step_s[i];
        out->pos_dly_s[i] = g.pos_dly_s[i];
        out->pos_bsum[i] = g.pos_bsum[i];
    }
    mynah_asr_stream_batch_share_stats(&out->share_rows, &out->share_total);
    mynah_asr_stream_relpos_counts(&out->relpos_priv, &out->relpos_shared,
                               &out->relpos_group);
    mynah_asr_stream_step_profile(out->comp_ns, 11, &out->comp_rows,
                          &out->comp_frames, &out->comp_steps);
    out->dly_ready_sel_ms[0] = sched_dly_pct(g.h_ready_sel, 0.50);
    out->dly_ready_sel_ms[1] = sched_dly_pct(g.h_ready_sel, 0.95);
    out->dly_ready_sel_ms[2] = sched_dly_pct(g.h_ready_sel, 0.99);
    out->dly_sel_start_ms[0] = sched_dly_pct(g.h_sel_start, 0.50);
    out->dly_sel_start_ms[1] = sched_dly_pct(g.h_sel_start, 0.95);
    out->dly_sel_start_ms[2] = sched_dly_pct(g.h_sel_start, 0.99);
    out->dly_ready_start_ms[0] = sched_dly_pct(g.h_ready_start, 0.50);
    out->dly_ready_start_ms[1] = sched_dly_pct(g.h_ready_start, 0.95);
    out->dly_ready_start_ms[2] = sched_dly_pct(g.h_ready_start, 0.99);
    out->dly_samples = g.dly_samples;
    out->pred_passes = g.pred_passes;
    out->pred_bad_passes = g.pred_bad_passes;
    out->pred_false_ready = g.pred_false_ready;
    out->pred_false_not_ready = g.pred_false_not_ready;
    out->pred_diag_sum = g.pred_diag_sum;
    out->pred_real_sum = g.pred_real_sum;
    out->fin_model_s = g.fin_model_s;
    out->fin_rest_s = g.fin_rest_s;
    out->fin_calls = g.fin_calls;
    for (int i = 0; i < MYNAH_ASR_SCHED_PHASES; i++)
        out->w_idle_by_phase[i] = g.w_idle_by_phase[i];
    out->park_idle = g.park_idle;
    out->park_ready = g.park_ready;
    out->park_partial = g.park_partial;
    out->uptime_s = g.t_start > 0.0 ? mynah_asr_now() - g.t_start : -1.0;
    {
        const double since = atomic_load_explicit(&g.phase_since, memory_order_relaxed);
        out->phase_s = since > 0.0 ? mynah_asr_now() - since : -1.0;
    }
    out->window_entered = atomic_load_explicit(&g.window_entered, memory_order_relaxed);
    out->window_filled = atomic_load_explicit(&g.window_filled, memory_order_relaxed);
    out->window_wait_ms_sum =
        (double)atomic_load_explicit(&g.window_wait_us, memory_order_relaxed) / 1000.0;
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
    free(g.b_ud); free(g.b_slot); free(g.fin); free(g.fin_flag);
    /* The per-row accounting arrays, allocated beside the ones above and until
     * now not released with them. One-shot at teardown, but `make leaks` should
     * not have to know that. */
    free(g.b_ready); free(g.b_sel); free(g.b_pos); free(g.b_d0);
    free(g.was_real); free(g.was_diag);
    g.b_ready = NULL; g.b_sel = NULL; g.b_pos = NULL; g.b_d0 = NULL;
    g.was_real = NULL; g.was_diag = NULL;
    g.b_ctx = NULL; g.b_stream = NULL; g.b_samples = NULL; g.b_n = NULL;
    g.b_ud = NULL; g.b_slot = NULL; g.fin = NULL;
    g.n_slots = 0;
}
