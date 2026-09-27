/* Service-wide metrics for a prefork fleet. See fleet.h for the design. */
#include "fleet.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "http_util.h"   /* mynah_asr_thread_set_name */
#include "slot.h"

const double mynah_asr_fleet_ms_edges[MYNAH_ASR_FLEET_MS_EDGES] = {
    40, 80, 160, 320, 480, 640, 960, 1280, 1920, 2560, 3840, 5120, 10240
};
const double mynah_asr_fleet_s_edges[MYNAH_ASR_FLEET_S_EDGES] = {
    1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600
};

/* sched.h carries the histograms as plain arrays so it does not depend on this
 * header; these keep the two shapes from drifting apart. */
_Static_assert(sizeof(((mynah_asr_sched_stats *)0)->first_text_hist) /
               sizeof(unsigned long) == MYNAH_ASR_FLEET_MS_EDGES + 1,
               "sched first_text_hist must have one bucket per fleet ms edge + 1");
_Static_assert(sizeof(((mynah_asr_sched_stats *)0)->session_hist) /
               sizeof(unsigned long) == MYNAH_ASR_FLEET_S_EDGES + 1,
               "sched session_hist must have one bucket per fleet s edge + 1");

int mynah_asr_fleet_ms_bucket(double ms) {
    for (int i = 0; i < MYNAH_ASR_FLEET_MS_EDGES; i++)
        if (ms <= mynah_asr_fleet_ms_edges[i]) return i;
    return MYNAH_ASR_FLEET_MS_EDGES;
}

int mynah_asr_fleet_s_bucket(double s) {
    for (int i = 0; i < MYNAH_ASR_FLEET_S_EDGES; i++)
        if (s <= mynah_asr_fleet_s_edges[i]) return i;
    return MYNAH_ASR_FLEET_S_EDGES;
}

/* ------------------------------------------------------------ the pages */

/* One worker's page. `seq` is the seqlock: odd while the worker writes, and a
 * reader that sees it change (or odd) reads again. Lock-free atomics on a
 * MAP_SHARED mapping are process-shared by construction; nothing here is a
 * pthread object, so nothing needs PTHREAD_PROCESS_SHARED. */
typedef struct {
    _Atomic unsigned long seq;
    _Atomic int published;
    double t_pub;                  /* CLOCK_MONOTONIC: one clock, every process */
    mynah_asr_fleet_stats st;
} fleet_page;

static fleet_page *g_pages;
static int g_npages;

static double fleet_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int mynah_asr_fleet_map(int n) {
    if (n <= 0) return -1;
    void *p = mmap(NULL, (size_t)n * sizeof(fleet_page), PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
        perror("fleet: mmap");
        return -1;
    }
    memset(p, 0, (size_t)n * sizeof(fleet_page));
    g_pages = (fleet_page *)p;
    g_npages = n;
    return 0;
}

static void page_write(fleet_page *pg, const mynah_asr_fleet_stats *st) {
    const unsigned long s0 = atomic_load_explicit(&pg->seq, memory_order_relaxed);
    atomic_store_explicit(&pg->seq, s0 + 1, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    pg->st = *st;
    pg->t_pub = fleet_now();
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&pg->seq, s0 + 2, memory_order_release);
    atomic_store_explicit(&pg->published, 1, memory_order_release);
}

/* A consistent copy, or 0 when the page was never published. A worker killed
 * in the middle of a write leaves `seq` odd for ever: after a bounded number of
 * retries the torn copy is refused rather than summed. */
static int page_read(const fleet_page *pg, mynah_asr_fleet_stats *out, double *t_pub) {
    if (!atomic_load_explicit(&pg->published, memory_order_acquire)) return 0;
    for (int tries = 0; tries < 1000; tries++) {
        const unsigned long a = atomic_load_explicit(&pg->seq, memory_order_acquire);
        if (a & 1ul) continue;
        *out = pg->st;
        *t_pub = pg->t_pub;
        atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&pg->seq, memory_order_relaxed) == a) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------ collection */

void mynah_asr_fleet_collect(mynah_asr_fleet_stats *o) {
    memset(o, 0, sizeof(*o));
    mynah_asr_sched_stats st;
    mynah_asr_sched_stats_read(&st);
    o->sessions = st.sessions;
    o->completed = st.completed;
    o->aborted = st.aborted;
    o->cancelled = st.cancelled;
    memcpy(o->cancel_by, st.cancel_by, sizeof(o->cancel_by));
    o->abandoned = st.abandoned;
    o->abandoned_recovered = st.abandoned_recovered;
    o->active = st.slots_active;
    o->cap = st.slots_cap;
    o->active_peak = st.active_peak;
    o->balanced = st.balanced;
    o->steps = st.steps;
    o->deltas = st.deltas;
    o->audio_seconds = st.audio_seconds;
    o->model_busy_s = st.w_model + st.w_model_solo;
    o->uptime_s = st.uptime_s > 0.0 ? st.uptime_s : 0.0;
    /* The 8 ms lag histogram folded onto the fleet edges. Bucket i holds lags
     * in [8i, 8i+8); every edge is a multiple of 8, so "bucket start < edge"
     * is exactly "lag < edge" and nothing is interpolated. */
    for (int i = 0; i < MYNAH_ASR_LAG_BUCKETS; i++) {
        if (st.lag_hist[i] == 0) continue;
        const double lo = (double)(i * MYNAH_ASR_LAG_BUCKET_MS);
        int e = 0;
        while (e < MYNAH_ASR_FLEET_MS_EDGES && lo >= mynah_asr_fleet_ms_edges[e]) e++;
        o->lag[e] += st.lag_hist[i];
    }
    o->lag_sum_ms = st.lag_sum_ms;
    memcpy(o->first_text, st.first_text_hist, sizeof(o->first_text));
    o->first_text_sum_ms = st.first_text_sum_ms;
    memcpy(o->finalize, st.finalize_hist, sizeof(o->finalize));
    o->finalize_sum_ms = st.finalize_sum_ms;
    memcpy(o->session, st.session_hist, sizeof(o->session));
    o->session_sum_s = st.session_sum_s;

    /* Backlog: audio sitting in the rings, not yet fed. Read slot by slot under
     * each slot's own lock, off the scheduler thread. */
    if (st.slots_cap > 0) {
        mynah_asr_sched_slot_view *v =
            (mynah_asr_sched_slot_view *)calloc((size_t)st.slots_cap, sizeof(*v));
        if (v != NULL) {
            const int n = mynah_asr_sched_slots_view(v, st.slots_cap);
            const double sr = (double)mynah_asr_sched_sample_rate();
            for (int i = 0; i < n && i < st.slots_cap; i++) {
                const double b = (double)v[i].ring_samples / sr;
                o->backlog_s_sum += b;
                if (b > o->backlog_s_max) o->backlog_s_max = b;
            }
            free(v);
        }
    }
}

static void *publisher_main(void *arg) {
    const int idx = (int)(intptr_t)arg;
    mynah_asr_thread_set_name("mynah-fleet");
    unsigned long unbalanced = 0;
    for (;;) {
        mynah_asr_fleet_stats st;
        mynah_asr_fleet_collect(&st);
        /* The books are exact in every snapshot (sched.c, acct_mu); a snapshot
         * that reads unbalanced is a bug, and it is COUNTED, never smoothed. */
        if (!st.balanced) unbalanced++;
        st.unbalanced_snapshots = unbalanced;
        page_write(&g_pages[idx], &st);
        struct timespec ts = {.tv_sec = 0, .tv_nsec = MYNAH_ASR_FLEET_PUBLISH_MS * 1000000L};
        nanosleep(&ts, NULL);
    }
    return NULL;
}

void mynah_asr_fleet_publisher_start(int index) {
    if (g_pages == NULL || index < 0 || index >= g_npages) return;
    pthread_t t;
    if (pthread_create(&t, NULL, publisher_main, (void *)(intptr_t)index) == 0)
        pthread_detach(t);
    else
        fprintf(stderr, "fleet: worker %d could not start its publisher; the "
                        "service metrics will not include it\n", index);
}

/* ------------------------------------------------------------ the sum */

void mynah_asr_fleet_add(mynah_asr_fleet_stats *a, const mynah_asr_fleet_stats *b,
                         int with_gauges) {
    a->sessions += b->sessions;
    a->completed += b->completed;
    a->aborted += b->aborted;
    a->cancelled += b->cancelled;
    for (int i = 0; i < MYNAH_ASR_SCHED_CANCEL__COUNT; i++) a->cancel_by[i] += b->cancel_by[i];
    a->abandoned += b->abandoned;
    a->abandoned_recovered += b->abandoned_recovered;
    a->unbalanced_snapshots += b->unbalanced_snapshots;
    a->steps += b->steps;
    a->deltas += b->deltas;
    a->audio_seconds += b->audio_seconds;
    a->model_busy_s += b->model_busy_s;
    for (int i = 0; i <= MYNAH_ASR_FLEET_MS_EDGES; i++) {
        a->lag[i] += b->lag[i];
        a->first_text[i] += b->first_text[i];
        a->finalize[i] += b->finalize[i];
    }
    for (int i = 0; i <= MYNAH_ASR_FLEET_S_EDGES; i++) a->session[i] += b->session[i];
    a->lag_sum_ms += b->lag_sum_ms;
    a->first_text_sum_ms += b->first_text_sum_ms;
    a->finalize_sum_ms += b->finalize_sum_ms;
    a->session_sum_s += b->session_sum_s;
    if (!with_gauges) return;
    a->active += b->active;
    a->cap += b->cap;
    a->active_peak += b->active_peak;   /* sum of per-worker peaks: an upper bound */
    a->backlog_s_sum += b->backlog_s_sum;
    if (b->backlog_s_max > a->backlog_s_max) a->backlog_s_max = b->backlog_s_max;
    if (b->uptime_s > a->uptime_s) a->uptime_s = b->uptime_s;
}

int mynah_asr_fleet_sum(mynah_asr_fleet_stats *sum, const int *dead,
                        mynah_asr_fleet_router *r) {
    memset(sum, 0, sizeof(*sum));
    sum->balanced = 1;
    int seen = 0;
    const double now = fleet_now();
    r->snapshot_age_max_s = 0.0;
    r->sessions_lost = 0;
    for (int i = 0; i < g_npages; i++) {
        mynah_asr_fleet_stats st;
        double t_pub = 0.0;
        if (!page_read(&g_pages[i], &st, &t_pub)) continue;
        seen++;
        const int is_dead = dead != NULL && dead[i];
        /* A dead worker's counters stay (the totals never go backwards); its
         * gauges do not; what it still called active is lost. */
        mynah_asr_fleet_add(sum, &st, !is_dead);
        if (is_dead) {
            r->sessions_lost += (unsigned long)(st.active > 0 ? st.active : 0);
            continue;
        }
        if (!st.balanced) sum->balanced = 0;
        const double age = now - t_pub;
        if (age > r->snapshot_age_max_s) r->snapshot_age_max_s = age;
    }
    return seen;
}

/* ------------------------------------------------------------ rendering */

static void render_hist(mynah_asr_metrics_buf *b, const char *name, const char *help,
                        const unsigned long *h, int nedges, const double *edges,
                        double scale, double sum) {
    mynah_asr_metrics_addf(b, "# HELP %s %s\n# TYPE %s histogram\n", name, help, name);
    unsigned long cum = 0;
    for (int i = 0; i < nedges; i++) {
        cum += h[i];
        mynah_asr_metrics_addf(b, "%s_bucket{le=\"%g\"} %lu\n", name, edges[i] * scale, cum);
    }
    cum += h[nedges];
    mynah_asr_metrics_addf(b, "%s_bucket{le=\"+Inf\"} %lu\n%s_sum %.6f\n%s_count %lu\n",
                           name, cum, name, sum, name, cum);
}

void mynah_asr_fleet_render(mynah_asr_metrics_buf *b, const mynah_asr_fleet_stats *s,
                            const mynah_asr_fleet_router *r) {
    mynah_asr_metrics_addf(b,
        "# ---- SERVICE-WIDE (mynah_asr_fleet_*): every worker summed, no worker label.\n"
        "# Under --prefork these come from each worker's shared snapshot, at most\n"
        "# %d ms old; a dead worker's counters stay in the totals.\n",
        MYNAH_ASR_FLEET_PUBLISH_MS);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_fleet_workers worker processes configured.\n"
        "# TYPE mynah_asr_fleet_workers gauge\nmynah_asr_fleet_workers %d\n"
        "# HELP mynah_asr_fleet_workers_up worker processes alive.\n"
        "# TYPE mynah_asr_fleet_workers_up gauge\nmynah_asr_fleet_workers_up %d\n"
        "# HELP mynah_asr_fleet_worker_deaths_total worker processes that died.\n"
        "# TYPE mynah_asr_fleet_worker_deaths_total counter\n"
        "mynah_asr_fleet_worker_deaths_total %lu\n"
        "# HELP mynah_asr_fleet_snapshot_age_seconds oldest live worker snapshot.\n"
        "# TYPE mynah_asr_fleet_snapshot_age_seconds gauge\n"
        "mynah_asr_fleet_snapshot_age_seconds %.3f\n",
        r->workers, r->workers_up, r->worker_deaths, r->snapshot_age_max_s);

    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_fleet_sessions_total stream sessions accepted (a slot claimed).\n"
        "# TYPE mynah_asr_fleet_sessions_total counter\n"
        "mynah_asr_fleet_sessions_total %lu\n"
        "# HELP mynah_asr_fleet_sessions_completed_total sessions ended with done + close.\n"
        "# TYPE mynah_asr_fleet_sessions_completed_total counter\n"
        "mynah_asr_fleet_sessions_completed_total %lu\n"
        "# HELP mynah_asr_fleet_sessions_cancelled_total sessions ended early, by the\n"
        "# code the client was sent (peer_gone = the client disconnected).\n"
        "# TYPE mynah_asr_fleet_sessions_cancelled_total counter\n",
        s->sessions, s->completed);
    for (int i = 0; i < MYNAH_ASR_SCHED_CANCEL__COUNT; i++)
        mynah_asr_metrics_addf(b, "mynah_asr_fleet_sessions_cancelled_total{reason=\"%s\"} %lu\n",
                               mynah_asr_sched_cancel_bucket_name(i), s->cancel_by[i]);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_fleet_sessions_aborted_total sessions claimed and released\n"
        "# before they started (the 101 or the writer could not be set up).\n"
        "# TYPE mynah_asr_fleet_sessions_aborted_total counter\n"
        "mynah_asr_fleet_sessions_aborted_total %lu\n"
        "# HELP mynah_asr_fleet_sessions_lost_total sessions a worker still held when it\n"
        "# died, from its last snapshot.\n"
        "# TYPE mynah_asr_fleet_sessions_lost_total counter\n"
        "mynah_asr_fleet_sessions_lost_total %lu\n"
        "# HELP mynah_asr_fleet_sessions_active sessions holding a slot now.\n"
        "# TYPE mynah_asr_fleet_sessions_active gauge\n"
        "mynah_asr_fleet_sessions_active %d\n"
        "# HELP mynah_asr_fleet_slots stream slots of the live workers.\n"
        "# TYPE mynah_asr_fleet_slots gauge\nmynah_asr_fleet_slots %d\n"
        "# HELP mynah_asr_fleet_slots_peak sum of each live worker's peak slots held.\n"
        "# TYPE mynah_asr_fleet_slots_peak gauge\nmynah_asr_fleet_slots_peak %d\n",
        s->aborted, r->sessions_lost, s->active, s->cap, s->active_peak);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_fleet_books_balanced 1 when, in every live worker's snapshot,\n"
        "# sessions = completed + cancelled + aborted + active. 0 is a counting bug.\n"
        "# TYPE mynah_asr_fleet_books_balanced gauge\n"
        "mynah_asr_fleet_books_balanced %d\n"
        "# HELP mynah_asr_fleet_books_unbalanced_total snapshots whose books did not\n"
        "# balance. Anything but 0 is a bug.\n"
        "# TYPE mynah_asr_fleet_books_unbalanced_total counter\n"
        "mynah_asr_fleet_books_unbalanced_total %lu\n"
        "# HELP mynah_asr_fleet_slots_abandoned_total slots whose connection gave up on\n"
        "# the scheduler; _recovered_total the ones the scheduler released later.\n"
        "# TYPE mynah_asr_fleet_slots_abandoned_total counter\n"
        "mynah_asr_fleet_slots_abandoned_total %lu\n"
        "# TYPE mynah_asr_fleet_slots_abandoned_recovered_total counter\n"
        "mynah_asr_fleet_slots_abandoned_recovered_total %lu\n",
        s->balanced, s->unbalanced_snapshots, s->abandoned, s->abandoned_recovered);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_fleet_audio_seconds_total audio fed to the model.\n"
        "# TYPE mynah_asr_fleet_audio_seconds_total counter\n"
        "mynah_asr_fleet_audio_seconds_total %.3f\n"
        "# HELP mynah_asr_fleet_model_busy_seconds_total scheduler time with the model\n"
        "# running, summed over workers: rate() / workers is the model duty.\n"
        "# TYPE mynah_asr_fleet_model_busy_seconds_total counter\n"
        "mynah_asr_fleet_model_busy_seconds_total %.3f\n"
        "# HELP mynah_asr_fleet_steps_total model steps.\n"
        "# TYPE mynah_asr_fleet_steps_total counter\nmynah_asr_fleet_steps_total %lu\n"
        "# HELP mynah_asr_fleet_deltas_total transcript deltas sent.\n"
        "# TYPE mynah_asr_fleet_deltas_total counter\nmynah_asr_fleet_deltas_total %lu\n"
        "# HELP mynah_asr_fleet_backlog_seconds audio queued in the rings, not yet fed.\n"
        "# TYPE mynah_asr_fleet_backlog_seconds gauge\n"
        "mynah_asr_fleet_backlog_seconds %.3f\n"
        "# HELP mynah_asr_fleet_backlog_max_seconds the most any one stream has queued.\n"
        "# TYPE mynah_asr_fleet_backlog_max_seconds gauge\n"
        "mynah_asr_fleet_backlog_max_seconds %.3f\n",
        s->audio_seconds, s->model_busy_s, s->steps, s->deltas,
        s->backlog_s_sum, s->backlog_s_max);
    render_hist(b, "mynah_asr_fleet_emission_lag_seconds",
                "delta emission lag: arrival of the last sample it covers to the frame.",
                s->lag, MYNAH_ASR_FLEET_MS_EDGES, mynah_asr_fleet_ms_edges, 1e-3,
                s->lag_sum_ms * 1e-3);
    render_hist(b, "mynah_asr_fleet_first_text_seconds",
                "per session, first audio in to first text out. INCLUDES the audio before "
                "the first word: what a user waits, not a model latency.",
                s->first_text, MYNAH_ASR_FLEET_MS_EDGES, mynah_asr_fleet_ms_edges, 1e-3,
                s->first_text_sum_ms * 1e-3);
    render_hist(b, "mynah_asr_fleet_finalization_seconds",
                "per finalized utterance, arrival of its last sample to done.",
                s->finalize, MYNAH_ASR_FLEET_MS_EDGES, mynah_asr_fleet_ms_edges, 1e-3,
                s->finalize_sum_ms * 1e-3);
    render_hist(b, "mynah_asr_fleet_session_seconds",
                "per session, slot claimed to slot released, every outcome.",
                s->session, MYNAH_ASR_FLEET_S_EDGES, mynah_asr_fleet_s_edges, 1.0,
                s->session_sum_s);
}
