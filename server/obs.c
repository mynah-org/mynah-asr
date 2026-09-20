/* The worker's view of itself. See obs.h for the four renderings and the
 * honest-metric boundary. */
#include "obs.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>   /* RLIMIT_NOFILE, read back for the banner */
#include <time.h>
#include <unistd.h>

#include "../src/dispatch.h"
#include "../src/flags.h"
#include "../src/mynah_asr.h"
#include "../src/qmat.h"
#include "../src/threads.h"
#include "prefork.h"
#include "sched.h"
#include "slot.h"

static mynah_asr_obs_config g_cfg;
static double g_t0;

/* --------------------------------------------------------- refusal counters
 *
 * A worker refuses for reasons the ROUTER never sees (a bad Content-Length, an
 * unknown endpoint, a lookahead that is not a preset, its own connection queue
 * full), so the router's table cannot answer "why did this worker say no".
 * A small fixed table, filled on first use with the compile-time literals the
 * refusal sites pass; anything beyond its capacity is `other`, because the one
 * thing a label set may never do is grow with traffic. */
#define OBS_REFUSAL_MAX 24
static struct {
    pthread_mutex_t mu;
    const char *code[OBS_REFUSAL_MAX];
    unsigned long n[OBS_REFUSAL_MAX];
    int used;
    unsigned long other;
} g_ref = {PTHREAD_MUTEX_INITIALIZER, {NULL}, {0}, 0, 0};

void mynah_asr_obs_refused(const char *code) {
    if (code == NULL || code[0] == '\0') code = "other";
    pthread_mutex_lock(&g_ref.mu);
    for (int i = 0; i < g_ref.used; i++) {
        if (strcmp(g_ref.code[i], code) == 0) {
            g_ref.n[i]++;
            pthread_mutex_unlock(&g_ref.mu);
            return;
        }
    }
    if (g_ref.used < OBS_REFUSAL_MAX) {
        g_ref.code[g_ref.used] = code;   /* borrowed: every caller passes a literal */
        g_ref.n[g_ref.used] = 1;
        g_ref.used++;
    } else {
        g_ref.other++;
    }
    pthread_mutex_unlock(&g_ref.mu);
}

static double now_s(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void mynah_asr_obs_init(const mynah_asr_obs_config *cfg) {
    g_cfg = *cfg;
    g_t0 = now_s();
}

/* ----------------------------------------------------------- the thresholds
 *
 * Chunk-period multiples, because a lag of one chunk period is the cadence
 * working and a lag of two is the cadence slipping -- an absolute millisecond
 * ladder would mean something different on every preset. Rounded UP to a
 * histogram bucket edge so the counter is EXACT (the histogram quantises to
 * MYNAH_ASR_LAG_BUCKET_MS), and the rounded value is what the label says: a
 * label that names a threshold the counter does not use is a lie with a
 * number in it. */
#define OBS_LAG_THRESHOLDS 3
static void lag_thresholds(int out[OBS_LAG_THRESHOLDS]) {
    double chunk = g_cfg.chunk_ms;
    if (!(chunk > 0.0)) chunk = 320.0;   /* an offline-only model has no cadence */
    int t[OBS_LAG_THRESHOLDS] = {(int)(chunk + 0.5), (int)(2.0 * chunk + 0.5), 1000};
    for (int i = 0; i < OBS_LAG_THRESHOLDS; i++) {
        int v = t[i];
        if (v % MYNAH_ASR_LAG_BUCKET_MS != 0)
            v += MYNAH_ASR_LAG_BUCKET_MS - (v % MYNAH_ASR_LAG_BUCKET_MS);
        out[i] = v;
    }
}

/* ---------------------------------------------------------------- the banner */

/* A soft limit of RLIM_INFINITY is a real answer and prints as one; printing
 * the sentinel as a number would put 18446744073709551615 on a line an
 * operator is supposed to read. */
static void rlim_text(rlim_t v, char *out, size_t cap) {
    if (v == RLIM_INFINITY) snprintf(out, cap, "unlimited");
    else                    snprintf(out, cap, "%llu", (unsigned long long)v);
}

static void print_server_config(FILE *out) {
    char pres[64] = "none";
    if (g_cfg.n_lookaheads > 0) {
        size_t k = 0;
        pres[0] = '\0';
        for (int i = 0; i < g_cfg.n_lookaheads && k + 8 < sizeof(pres); i++)
            k += (size_t)snprintf(pres + k, sizeof(pres) - k, "%s%d",
                                  i ? "," : "", g_cfg.lookaheads[i]);
    }
    fprintf(out,
        "[SERVER-CONFIG] v=1 model_dir=%s model=%s group=%s engine=%s quant=%s "
        "lid_model=%s streaming=%s lookahead_default=%d lookahead_presets=%s "
        "chunk_ms=%.0f port=%d cap=%d ring_s=%d idle_ms=%d ping_ms=%d "
        "max_audio_s=%.0f max_frame_bytes=%zu max_pending=%d batch=%d "
        "http_threads=%d pool_threads=%d blas_budget=%d prefork=%s worker=%d "
        "metrics=%s\n",
        g_cfg.model_dir ? g_cfg.model_dir : "-",
        g_cfg.model_name ? g_cfg.model_name : "-",
        g_cfg.group ? g_cfg.group : "-",
        g_cfg.engine ? g_cfg.engine : "-",
        g_cfg.quant ? g_cfg.quant : "-",
        g_cfg.lid_dir ? g_cfg.lid_dir : "none",
        g_cfg.streaming ? "yes" : "no",
        g_cfg.lookahead_default, pres, g_cfg.chunk_ms,
        g_cfg.port, g_cfg.cap, g_cfg.ring_seconds, g_cfg.idle_ms, g_cfg.ping_ms,
        g_cfg.max_audio_seconds, g_cfg.max_frame_bytes, g_cfg.max_pending,
        g_cfg.batch, g_cfg.http_threads, mynah_asr_num_threads(),
        mynah_asr_blas_budget(),
        g_cfg.prefork_workers > 0 ? "yes" : "single-process",
        mynah_asr_prefork_worker_index(),
        /* A WORKER never answers a scrape: the router bound the port before the
         * fork and closed this process's copy of it. Saying "on" here would
         * point an operator at a port this pid does not hold. */
        g_cfg.metrics_port <= 0 ? "off"
            : mynah_asr_prefork_worker_index() >= 0 ? "served-by-router" : "on");
    if (g_cfg.prefork_workers > 0) {
        /* One field, one token: the plan is "a=2 b=1" and this line is parsed
         * by splitting on spaces, so the separator becomes a comma here. */
        char plan[256];
        const char *src = mynah_asr_prefork_model_plan();
        size_t k = 0;
        for (size_t i = 0; src[i] != '\0' && k + 1 < sizeof(plan); i++)
            plan[k++] = src[i] == ' ' ? ',' : src[i];
        if (k == 0 && sizeof(plan) > 1) plan[k++] = '-';
        plan[k] = '\0';
        fprintf(out, "[SERVER-CONFIG] v=1 prefork_workers=%d prefork_threads=%d "
                     "group_plan=%s\n",
                g_cfg.prefork_workers, g_cfg.prefork_threads, plan);
    }
    if (g_cfg.metrics_port > 0)
        fprintf(out, "[SERVER-CONFIG] v=1 metrics_bind=%s metrics_port=%d\n",
                g_cfg.metrics_bind ? g_cfg.metrics_bind : "127.0.0.1",
                g_cfg.metrics_port);
    /* S7-1/S7-2: the two ceilings a hundred streams arriving at once hit
     * first, and the two this process can be stopped by without saying so.
     * `listen_backlog` is what listen() was given, `somaxconn` what the kernel
     * will hold -- a backlog above it is a burst dropped as SYNs before
     * accept(), where no rung of the admission ladder and no counter here can
     * see it. `nofile_*` is READ BACK now rather than passed in: what matters
     * is the ceiling THIS process is running under, and a prefork worker
     * inherited it across the fork instead of raising it. A worker also does
     * not own the listener, so the backlog says whose it is, as `metrics=`
     * above does. */
    {
        struct rlimit rl;
        char soft[32] = "unknown", hard[32] = "unknown";
        if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
            rlim_text(rl.rlim_cur, soft, sizeof(soft));
            rlim_text(rl.rlim_max, hard, sizeof(hard));
        }
        fprintf(out, "[SERVER-CONFIG] v=1 listen_backlog=%d somaxconn=%d "
                     "listen_owner=%s nofile_soft=%s nofile_hard=%s\n",
                g_cfg.listen_backlog, g_cfg.listen_somaxconn,
                mynah_asr_prefork_worker_index() >= 0 ? "router" : "self",
                soft, hard);
    }
    fflush(out);
}

void mynah_asr_obs_banner(void) {
    /* The two lines src/flags.c owns: what the environment asked for and what
     * this build on this host can actually do with it. */
    mynah_asr_flags_print(stderr);
    print_server_config(stderr);
    /* A prefork worker already printed its own [TOPOLOGY] from inside the fork,
     * where the mask had just been set. A single process prints its own here. */
    if (mynah_asr_prefork_worker_index() < 0)
        mynah_asr_prefork_print_topology(stderr, mynah_asr_num_threads());
}

/* ---------------------------------------------------------------- /v1/health */

void mynah_asr_obs_health(cJSON *into) {
    char mask[256];
    const int pinned = mynah_asr_prefork_actual_mask(mask, sizeof(mask));

    cJSON *m = cJSON_AddObjectToObject(into, "model");
    cJSON_AddStringToObject(m, "name", g_cfg.model_name ? g_cfg.model_name : "-");
    cJSON_AddStringToObject(m, "engine", g_cfg.engine ? g_cfg.engine : "-");
    cJSON_AddStringToObject(m, "quant", g_cfg.quant ? g_cfg.quant : "-");
    cJSON_AddNumberToObject(m, "lookahead_default", g_cfg.lookahead_default);
    /* WHICH group this worker serves, and the whole fleet's plan beside it, so
     * one probe answers both "what did I reach" and "what else is there". The
     * plan is "" in a single-model fleet, which is itself the answer. */
    cJSON_AddStringToObject(into, "group",
                            g_cfg.group != NULL && g_cfg.group[0] != '\0'
                                ? g_cfg.group
                                : (g_cfg.model_name ? g_cfg.model_name : "-"));
    cJSON_AddStringToObject(into, "groups", mynah_asr_prefork_model_plan());

    cJSON *p = cJSON_AddObjectToObject(into, "process");
    cJSON_AddNumberToObject(p, "worker", mynah_asr_prefork_worker_index());
    cJSON_AddNumberToObject(p, "pid", (double)getpid());
    cJSON_AddNumberToObject(p, "uptime_s", now_s() - g_t0);
    cJSON_AddNumberToObject(p, "pool_threads", mynah_asr_num_threads());
    cJSON_AddNumberToObject(p, "prefork_threads", mynah_asr_prefork_worker_threads());
    cJSON_AddNumberToObject(p, "http_threads", g_cfg.http_threads);
    cJSON_AddBoolToObject(p, "pinned", pinned ? 1 : 0);
    cJSON_AddStringToObject(p, "cpu_mask", mask);
    cJSON_AddStringToObject(p, "build", mynah_asr_build_id());
    cJSON_AddStringToObject(p, "blas", mynah_asr_blas_provider());
    cJSON_AddStringToObject(p, "simd", mynah_asr_simd_profile());
    /* Resolved by the owner's predicate (src/qmat.c), never re-derived here:
     * ENGINEERING.md §5. */
    cJSON_AddStringToObject(p, "int8_kernel", mynah_asr_qmat_int8_kernel());
    cJSON_AddStringToObject(p, "int4_kernel", mynah_asr_qmat_int4_kernel());
    cJSON_AddStringToObject(p, "int8_gemm",
        mynah_asr_qmat_qgemm() > 0 ? "on"
        : mynah_asr_qmat_qgemm() == 0 ? "off" : "not-compiled");

    cJSON *r = cJSON_AddObjectToObject(into, "refused");
    pthread_mutex_lock(&g_ref.mu);
    for (int i = 0; i < g_ref.used; i++)
        cJSON_AddNumberToObject(r, g_ref.code[i], (double)g_ref.n[i]);
    if (g_ref.other > 0) cJSON_AddNumberToObject(r, "other", (double)g_ref.other);
    pthread_mutex_unlock(&g_ref.mu);
}

/* ------------------------------------------------------------------ /metrics */

void mynah_asr_obs_render_metrics(mynah_asr_metrics_buf *b, void *unused) {
    (void)unused;
    mynah_asr_sched_stats st;
    mynah_asr_sched_stats_read(&st);

    char wl[16];
    const int widx = mynah_asr_prefork_worker_index();
    snprintf(wl, sizeof(wl), "%d", widx);

    char build[128], blas[64], simd[64], k8[64];
    mynah_asr_metrics_label(mynah_asr_build_id(), build, sizeof(build));
    mynah_asr_metrics_label(mynah_asr_blas_provider(), blas, sizeof(blas));
    mynah_asr_metrics_label(mynah_asr_simd_profile(), simd, sizeof(simd));
    mynah_asr_metrics_label(mynah_asr_qmat_int8_kernel(), k8, sizeof(k8));

    mynah_asr_metrics_addf(b,
        "# This process's OWN counters, labelled with its worker index (-1 = a\n"
        "# single-process server). Per-worker series are never summed here: a\n"
        "# fleet total hides the one worker that stopped.\n"
        "# No client-side latency is exported -- no TTFP, no stall rate. Those are\n"
        "# measured at the far end of a socket this process does not own, and they\n"
        "# belong to the benchmark harness.\n");
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_build_info build, BLAS provider, compiled SIMD profile and\n"
        "# the int8 kernel src/qmat.c's own predicate resolved.\n"
        "# TYPE mynah_asr_build_info gauge\n"
        "mynah_asr_build_info{build=\"%s\",blas=\"%s\",simd=\"%s\",int8_kernel=\"%s\"} 1\n",
        build, blas, simd, k8);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_uptime_seconds seconds since this process started serving.\n"
        "# TYPE mynah_asr_uptime_seconds gauge\n"
        "mynah_asr_uptime_seconds %.3f\n", now_s() - g_t0);

    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_sessions_total stream slots claimed since start.\n"
        "# TYPE mynah_asr_sessions_total counter\n"
        "mynah_asr_sessions_total{worker=\"%s\"} %lu\n"
        "# HELP mynah_asr_steps_total scheduler steps that fed a chunk to the model.\n"
        "# TYPE mynah_asr_steps_total counter\n"
        "mynah_asr_steps_total{worker=\"%s\"} %lu\n"
        "# HELP mynah_asr_deltas_total transcript deltas emitted.\n"
        "# TYPE mynah_asr_deltas_total counter\n"
        "mynah_asr_deltas_total{worker=\"%s\"} %lu\n"
        "# HELP mynah_asr_eou_total end-of-utterance frames emitted.\n"
        "# TYPE mynah_asr_eou_total counter\n"
        "mynah_asr_eou_total{worker=\"%s\"} %lu\n",
        wl, st.sessions, wl, st.steps, wl, st.deltas, wl, st.eous);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_audio_seconds_total seconds of audio fed to the model,\n"
        "# streams and offline jobs alike. THE throughput unit: any RTF claim about\n"
        "# this server has this as its denominator.\n"
        "# TYPE mynah_asr_audio_seconds_total counter\n"
        "mynah_asr_audio_seconds_total{worker=\"%s\"} %.3f\n", wl, st.audio_seconds);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_offline_jobs_total offline (REST) jobs completed.\n"
        "# TYPE mynah_asr_offline_jobs_total counter\n"
        "mynah_asr_offline_jobs_total{worker=\"%s\"} %lu\n"
        "# HELP mynah_asr_offline_queued offline jobs waiting for a step right now.\n"
        "# TYPE mynah_asr_offline_queued gauge\n"
        "mynah_asr_offline_queued{worker=\"%s\"} %d\n",
        wl, st.offline_done, wl, st.offline_pending);

    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_cancelled_total sessions ended by a cap, a dead peer or a\n"
        "# shutdown, bucketed by the code the client was given.\n"
        "# TYPE mynah_asr_cancelled_total counter\n");
    for (int i = 0; i < MYNAH_ASR_SCHED_CANCEL__COUNT; i++)
        mynah_asr_metrics_addf(b,
            "mynah_asr_cancelled_total{worker=\"%s\",reason=\"%s\"} %lu\n",
            wl, mynah_asr_sched_cancel_bucket_name(i), st.cancel_by[i]);

    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_refused_total requests this worker refused, by the code in\n"
        "# the error body. Its own refusals only: the prefork router counts its own.\n"
        "# TYPE mynah_asr_refused_total counter\n");
    pthread_mutex_lock(&g_ref.mu);
    for (int i = 0; i < g_ref.used; i++) {
        char code[64];
        mynah_asr_metrics_label(g_ref.code[i], code, sizeof(code));
        mynah_asr_metrics_addf(b,
            "mynah_asr_refused_total{worker=\"%s\",code=\"%s\"} %lu\n",
            wl, code, g_ref.n[i]);
    }
    const unsigned long other = g_ref.other;
    pthread_mutex_unlock(&g_ref.mu);
    if (other > 0)
        mynah_asr_metrics_addf(b,
            "mynah_asr_refused_total{worker=\"%s\",code=\"other\"} %lu\n", wl, other);

    /* Emission lag: a _sum/_count pair and EXACT threshold counters, not a
     * histogram. A Prometheus histogram of a quantity we already keep as a
     * fixed 8 ms histogram would be a second quantisation of the same numbers,
     * and the three thresholds below are the ones an operator acts on. */
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_emission_lag_ms_sum total milliseconds between a sample\n"
        "# arriving in a slot's ring and the delta that answered it. Server-side by\n"
        "# construction: both ends of the interval are timestamps this process took.\n"
        "# TYPE mynah_asr_emission_lag_ms_sum counter\n"
        "mynah_asr_emission_lag_ms_sum{worker=\"%s\"} %.3f\n"
        "# HELP mynah_asr_emission_lag_ms_count deltas that contributed to the sum.\n"
        "# TYPE mynah_asr_emission_lag_ms_count counter\n"
        "mynah_asr_emission_lag_ms_count{worker=\"%s\"} %lu\n"
        "# HELP mynah_asr_emission_lag_ms_max the worst single emission lag so far.\n"
        "# TYPE mynah_asr_emission_lag_ms_max gauge\n"
        "mynah_asr_emission_lag_ms_max{worker=\"%s\"} %.3f\n",
        wl, st.lag_sum_ms, wl, st.lag_count, wl, st.lag_max_ms);

    int th[OBS_LAG_THRESHOLDS];
    lag_thresholds(th);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_emission_lag_over_ms_total deltas whose emission lag was AT\n"
        "# LEAST the labelled number of milliseconds. Exact, not interpolated: the\n"
        "# thresholds are bucket edges of the %d ms histogram, and the first two are\n"
        "# chunk-period multiples of this model's default preset.\n"
        "# TYPE mynah_asr_emission_lag_over_ms_total counter\n",
        MYNAH_ASR_LAG_BUCKET_MS);
    for (int i = 0; i < OBS_LAG_THRESHOLDS; i++)
        mynah_asr_metrics_addf(b,
            "mynah_asr_emission_lag_over_ms_total{worker=\"%s\",le=\"%d\"} %lu\n",
            wl, th[i], mynah_asr_sched_lag_over(st.lag_hist, th[i]));

    /* S2-2b: the batched step, as facts. A run can be asked whether it batched
     * at all (rows_stacked > 0), how large its ready set was on average
     * (ready_size_sum / batched_steps_total) and how long a step took
     * (step_wall_ms_sum / _count) -- the three numbers the cadence law
     * T_step(B) = a + b*B is fitted from. The per-B breakdown is in /v1/health
     * and NOT here: a `b` label would grow the label set with the slot cap. */
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_batched_steps_total calls to the batched stream step, one\n"
        "# per scheduler step that had a ready set (B=1 included: that is the\n"
        "# library's single path, reached through the same call).\n"
        "# TYPE mynah_asr_batched_steps_total counter\n"
        "mynah_asr_batched_steps_total{worker=\"%s\"} %lu\n"
        "# HELP mynah_asr_batch_rows_stacked_total encoder rows (one row = one encoder\n"
        "# frame of one stream) pushed through the STACKED encoder path. 0 next to a\n"
        "# non-zero batched_steps_total means every step degraded to single steps.\n"
        "# TYPE mynah_asr_batch_rows_stacked_total counter\n"
        "mynah_asr_batch_rows_stacked_total{worker=\"%s\"} %llu\n"
        "# HELP mynah_asr_batch_ready_size_sum streams in the ready set, summed over\n"
        "# batched steps; divided by batched_steps_total it is the mean ready set.\n"
        "# TYPE mynah_asr_batch_ready_size_sum counter\n"
        "mynah_asr_batch_ready_size_sum{worker=\"%s\"} %lu\n",
        wl, st.batched_steps, wl, st.rows_stacked, wl, st.ready_sum);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_step_wall_ms_sum milliseconds spent inside the batched step\n"
        "# call. Model time only: building the ready set and the frames are outside it.\n"
        "# TYPE mynah_asr_step_wall_ms_sum counter\n"
        "mynah_asr_step_wall_ms_sum{worker=\"%s\"} %.3f\n"
        "# HELP mynah_asr_step_wall_ms_count steps that contributed to the sum.\n"
        "# TYPE mynah_asr_step_wall_ms_count counter\n"
        "mynah_asr_step_wall_ms_count{worker=\"%s\"} %lu\n",
        wl, st.step_wall_ms_sum, wl, st.step_wall_count);

    /* The collection window, as a pair. `entered` alone says nothing: a window
     * that always fills is free (the streams share a cadence and it is a
     * deadline, never a delay) and one that always times out is either too long
     * or waiting for streams that are not coming. Only entered-vs-filled tells
     * them apart, and the wait sum prices it. */
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_batch_window_entered_total steps that waited for a wider\n"
        "# ready set (0 when --batch-window-ms is 0, which is the default).\n"
        "# TYPE mynah_asr_batch_window_entered_total counter\n"
        "mynah_asr_batch_window_entered_total{worker=\"%s\"} %lu\n"
        "# HELP mynah_asr_batch_window_filled_total windows that ended because every\n"
        "# live stream was ready, rather than because the deadline passed.\n"
        "# TYPE mynah_asr_batch_window_filled_total counter\n"
        "mynah_asr_batch_window_filled_total{worker=\"%s\"} %lu\n"
        "# HELP mynah_asr_batch_window_wait_ms_sum milliseconds spent waiting in those\n"
        "# windows. Divided by entered it is what a step actually paid to be wider.\n"
        "# TYPE mynah_asr_batch_window_wait_ms_sum counter\n"
        "mynah_asr_batch_window_wait_ms_sum{worker=\"%s\"} %.3f\n",
        wl, st.window_entered, wl, st.window_filled, wl, st.window_wait_ms_sum);

    /* Step cost BY READY-SET WIDTH. The capacity law T_step(B) = a + b*B prices
     * one step serving B streams, and its whole economy is that `a` is paid
     * once and amortised over B. Two servers with the same mean step time can
     * therefore have completely different capacity: one stepping B=16 once, one
     * stepping B=2 eight times. Only this pair of series tells them apart, and
     * it is what turns a + b*B from a bench model into a reading taken on the
     * serving path -- the bench measures the step, this measures the step AS
     * THE SCHEDULER ACTUALLY CALLS IT.
     *
     * Measured on 24 Neoverse-V2 cores (2026-09-20): with 16 streams in flight
     * the mean ready set was 3.73, so the worker paid `a` four times a period
     * and served a third of the predicted capacity. That gap was invisible in
     * every other counter on this page.
     *
     * Emitted per width with a `b` label rather than as a histogram: the widths
     * are not cumulative buckets, they are distinct step shapes, and a reader
     * must be able to take the mean of one without taking it apart from
     * another. The last bucket is a saturating ">=" for widths past the table. */
    for (int i = 1; i < MYNAH_ASR_SCHED_B_BUCKETS; i++) {
        if (st.step_b_count[i] == 0) continue;
        const int last = (i == MYNAH_ASR_SCHED_B_BUCKETS - 1);
        if (i == 1)
            mynah_asr_metrics_addf(b,
                "# HELP mynah_asr_step_b_count steps whose ready set had this width.\n"
                "# TYPE mynah_asr_step_b_count counter\n"
                "# HELP mynah_asr_step_b_wall_ms_sum milliseconds spent in steps of\n"
                "# that width; divided by the count it is T_step(B) as served.\n"
                "# TYPE mynah_asr_step_b_wall_ms_sum counter\n");
        mynah_asr_metrics_addf(b,
            "mynah_asr_step_b_count{worker=\"%s\",b=\"%s%d\"} %lu\n"
            "mynah_asr_step_b_wall_ms_sum{worker=\"%s\",b=\"%s%d\"} %.3f\n",
            wl, last ? ">=" : "", i, st.step_b_count[i],
            wl, last ? ">=" : "", i, st.step_b_wall_ms[i]);
    }

    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_slots_active stream slots this worker is holding now.\n"
        "# TYPE mynah_asr_slots_active gauge\n"
        "mynah_asr_slots_active{worker=\"%s\"} %d\n"
        "# HELP mynah_asr_slots_cap stream slots this worker will hold (--cap).\n"
        "# TYPE mynah_asr_slots_cap gauge\n"
        "mynah_asr_slots_cap{worker=\"%s\"} %d\n",
        wl, st.slots_active, wl, st.slots_cap);
}

/* ------------------------------------------------------------------- SIGUSR1 */

void mynah_asr_obs_dump(void) {
    static _Atomic unsigned long seq;
    const unsigned long n = atomic_fetch_add_explicit(&seq, 1, memory_order_relaxed) + 1;
    const int widx = mynah_asr_prefork_worker_index();

    mynah_asr_sched_stats st;
    mynah_asr_sched_stats_read(&st);
    char mask[256];
    const int pinned = mynah_asr_prefork_actual_mask(mask, sizeof(mask));
    int th[OBS_LAG_THRESHOLDS];
    lag_thresholds(th);

    /* COMPOSED FIRST, WRITTEN ONCE. The parent forwards SIGUSR1 to every
     * worker, so W processes write this at the same moment onto the same
     * stderr; a dump built out of a dozen fprintf() calls comes back with
     * worker 0's line spliced into the middle of worker 1's, which is exactly
     * what the first run of this produced. One buffer, one write: on a regular
     * file that is atomic, and on a pipe the ordering within a dump still
     * holds. Bracketed begin/end with the same seq so two dumps can never be
     * read as one. */
    /* Room for the per-slot block: 16 slots at ~150 bytes each on top of the
     * aggregate lines. OBS_ADD truncates rather than overruns, and a truncated
     * dump is exactly the artefact a stall investigation cannot afford. */
    char b[16384];
    size_t k = 0;
#define OBS_ADD(...) do { \
        if (k < sizeof(b)) \
            k += (size_t)snprintf(b + k, sizeof(b) - k, __VA_ARGS__); \
        if (k >= sizeof(b)) k = sizeof(b) - 1; \
    } while (0)

    OBS_ADD("[DUMP] v=1 worker=%d seq=%lu begin\n", widx, n);
    OBS_ADD("[DUMP] worker=%d seq=%lu process pid=%d uptime_s=%.1f "
            "pool_threads=%d blas_budget=%d pinned=%s mask=%s\n",
            widx, n, (int)getpid(), now_s() - g_t0, mynah_asr_num_threads(),
            mynah_asr_blas_budget(), pinned ? "yes" : "no", mask);
    OBS_ADD("[DUMP] worker=%d seq=%lu build=%s blas=%s simd=%s "
            "int8_kernel=%s int8_gemm=%s\n",
            widx, n, mynah_asr_build_id(), mynah_asr_blas_provider(),
            mynah_asr_simd_profile(), mynah_asr_qmat_int8_kernel(),
            mynah_asr_qmat_qgemm() > 0 ? "on"
                : mynah_asr_qmat_qgemm() == 0 ? "off" : "not-compiled");
    OBS_ADD("[DUMP] worker=%d seq=%lu model=%s engine=%s quant=%s streaming=%s "
            "lookahead_default=%d chunk_ms=%.0f\n",
            widx, n, g_cfg.model_name ? g_cfg.model_name : "-",
            g_cfg.engine ? g_cfg.engine : "-", g_cfg.quant ? g_cfg.quant : "-",
            g_cfg.streaming ? "yes" : "no", g_cfg.lookahead_default, g_cfg.chunk_ms);
    OBS_ADD("[DUMP] worker=%d seq=%lu slots active=%d cap=%d sessions=%lu "
            "steps=%lu deltas=%lu eous=%lu audio_s=%.1f\n",
            widx, n, st.slots_active, st.slots_cap, st.sessions, st.steps,
            st.deltas, st.eous, st.audio_seconds);
    {   /* The wall-time accounting. execution_duty is the share of the
         * scheduler's life spent inside the model; the rest is its own serial
         * work and parking. A park with nothing buffered is the stream cadence
         * and is not waste; a park with a READY slot behind it would be a
         * scheduling defect, which is why the three are counted apart. */
        static const char *const PN[] = {"?", "slot-poll", "reset", "stage", "step",
                                         "finalize", "offline", "park",
                                         "take-requests", "cancel", "peer-check"};
        double tot = 0.0;
        for (int i = 1; i < MYNAH_ASR_SCHED_PHASES; i++) tot += st.phase_wall_s[i];
        const double three = st.w_model + st.w_runnable_idle + st.w_no_work;
        OBS_ADD("[DUMP] worker=%d seq=%lu wall uptime_s=%.1f accounted_s=%.1f "
                "execution_duty=%.3f\n", widx, n, st.uptime_s, tot,
                tot > 0.0 ? st.phase_wall_s[4] / tot : 0.0);
        OBS_ADD("[DUMP] worker=%d seq=%lu split model_busy_s=%.2f (%.3f) "
                "runnable_idle_s=%.2f (%.3f) no_work_s=%.2f (%.3f)\n",
                widx, n, st.w_model, three > 0 ? st.w_model / three : 0.0,
                st.w_runnable_idle, three > 0 ? st.w_runnable_idle / three : 0.0,
                st.w_no_work, three > 0 ? st.w_no_work / three : 0.0);
        for (int i = 1; i < MYNAH_ASR_SCHED_PHASES; i++) {
            if (st.w_idle_by_phase[i] <= 0.0) continue;
            OBS_ADD("[DUMP] worker=%d seq=%lu idle_in %-13s wall_s=%8.2f share_of_idle=%.3f\n",
                    widx, n, PN[i], st.w_idle_by_phase[i],
                    st.w_runnable_idle > 0 ? st.w_idle_by_phase[i] / st.w_runnable_idle : 0.0);
        }
        for (int i = 1; i < MYNAH_ASR_SCHED_PHASES; i++) {
            if (st.phase_calls[i] == 0) continue;
            OBS_ADD("[DUMP] worker=%d seq=%lu phase %-13s wall_s=%8.2f share=%.3f "
                    "calls=%-9lu mean_us=%.1f\n", widx, n, PN[i], st.phase_wall_s[i],
                    tot > 0.0 ? st.phase_wall_s[i] / tot : 0.0, st.phase_calls[i],
                    1e6 * st.phase_wall_s[i] / (double)st.phase_calls[i]);
        }
        OBS_ADD("[DUMP] worker=%d seq=%lu park idle=%lu partial=%lu ready=%lu"
                "   (ready>0 would be a scheduling defect)\n",
                widx, n, st.park_idle, st.park_partial, st.park_ready);
    }
    {
        static const char *const PHASE[] = {"?", "slot-poll", "reset", "stage",
                                            "step", "finalize", "offline", "park",
                                            "take-requests", "cancel", "peer-check"};
        const int ph = st.phase >= 0 && st.phase <= 10 ? st.phase : 0;
        OBS_ADD("[DUMP] worker=%d seq=%lu sched loops=%lu phase=%s slot=%d "
                "phase_s=%.1f\n",
                widx, n, st.loops, PHASE[ph], st.phase_slot, st.phase_s);
    }
    OBS_ADD("[DUMP] worker=%d seq=%lu batch steps=%lu rows_stacked=%llu "
            "ready_mean=%.2f step_wall_ms_mean=%.1f\n",
            widx, n, st.batched_steps, st.rows_stacked,
            st.batched_steps ? (double)st.ready_sum / (double)st.batched_steps : 0.0,
            st.step_wall_count ? st.step_wall_ms_sum / (double)st.step_wall_count : 0.0);
    {   /* the pool meter: whether the spin-then-park pool is catching its
         * dispatches hot or paying a kernel round trip for each one. Under real
         * load this is the difference the small GEMMs of a stream step feel. */
        mynah_asr_pool_stats ps;
        mynah_asr_pool_stats_get(&ps);
        const unsigned long long ww = ps.worker_spin + ps.worker_park;
        const unsigned long long cc = ps.caller_spin + ps.caller_park;
        OBS_ADD("[DUMP] worker=%d seq=%lu pool spin_us=%d workers=%d dispatches=%llu "
                "inline=%llu worker_spin_pct=%.1f caller_spin_pct=%.1f\n",
                widx, n, ps.spin_us, ps.workers, ps.dispatches, ps.inline_runs,
                ww ? 100.0 * (double)ps.worker_spin / (double)ww : 0.0,
                cc ? 100.0 * (double)ps.caller_spin / (double)cc : 0.0);
    }
    {   /* Per-slot, because a stall is a property of PARTICULAR streams. The
         * aggregate says the fleet is behind; this says which ones, how much
         * audio is waiting in each ring, and whether the scheduler can even see
         * them (`out=0` is a slot it skips by design, mid-handshake). A stream
         * whose ring is full while its neighbours step is starved; one whose
         * ring is empty is simply not being fed. */
        mynah_asr_sched_slot_view sv[32];
        const int nv = mynah_asr_sched_slots_view(sv, 32);
        for (int i = 0; i < nv; i++)
            OBS_ADD("[DUMP] worker=%d seq=%lu slot id=%d state=%d out=%d stream=%d "
                    "la=%d ready=%d steps=%lu deltas=%lu ring_s=%.1f need_s=%.2f "
                    "age_s=%.1f since_rx_s=%.1f since_step_s=%.1f lag_max_ms=%.0f "
                    "mu_owner=%lx mu_held_s=%.1f mu_where=%s "
                    "out_owner=%lx out_held_s=%.1f out_where=%s\n",
                    widx, n, sv[i].id, sv[i].state, sv[i].has_out, sv[i].has_stream,
                    sv[i].lookahead, sv[i].ready, sv[i].steps, sv[i].deltas,
                    (double)sv[i].ring_samples / 16000.0,
                    (double)sv[i].need_samples / 16000.0,
                    sv[i].age_s, sv[i].since_arrival_s, sv[i].since_step_s,
                    sv[i].lag_max_ms, sv[i].mu_owner, sv[i].mu_held_s,
                    sv[i].mu_where ? sv[i].mu_where : "-",
                    sv[i].out_owner, sv[i].out_held_s,
                    sv[i].out_where ? sv[i].out_where : "-");
    }
    OBS_ADD("[DUMP] worker=%d seq=%lu offline queued=%d done=%lu max_pending=%d\n",
            widx, n, st.offline_pending, st.offline_done, st.offline_max_pending);
    OBS_ADD("[DUMP] worker=%d seq=%lu cancelled=%lu", widx, n, st.cancelled);
    for (int i = 0; i < MYNAH_ASR_SCHED_CANCEL__COUNT; i++)
        OBS_ADD(" %s=%lu", mynah_asr_sched_cancel_bucket_name(i), st.cancel_by[i]);
    OBS_ADD("\n[DUMP] worker=%d seq=%lu refused", widx, n);
    pthread_mutex_lock(&g_ref.mu);
    if (g_ref.used == 0 && g_ref.other == 0) OBS_ADD(" none");
    for (int i = 0; i < g_ref.used; i++) OBS_ADD(" %s=%lu", g_ref.code[i], g_ref.n[i]);
    if (g_ref.other > 0) OBS_ADD(" other=%lu", g_ref.other);
    pthread_mutex_unlock(&g_ref.mu);
    OBS_ADD("\n[DUMP] worker=%d seq=%lu lag_ms p50=%.0f p95=%.0f max=%.1f "
            "count=%lu sum=%.0f bucket_ms=%d",
            widx, n, mynah_asr_sched_lag_quantile(st.lag_hist, 0.50),
            mynah_asr_sched_lag_quantile(st.lag_hist, 0.95), st.lag_max_ms,
            st.lag_count, st.lag_sum_ms, MYNAH_ASR_LAG_BUCKET_MS);
    for (int i = 0; i < OBS_LAG_THRESHOLDS; i++)
        OBS_ADD(" over_%dms=%lu", th[i],
                mynah_asr_sched_lag_over(st.lag_hist, th[i]));
    OBS_ADD("\n[DUMP] v=1 worker=%d seq=%lu end\n", widx, n);
#undef OBS_ADD

    fwrite(b, 1, k, stderr);
    fflush(stderr);
}
