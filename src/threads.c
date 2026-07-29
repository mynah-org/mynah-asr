#include "threads.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

#define PF_MAX_THREADS 64

int mynah_asr_num_threads(void) {
    static int nth = 0;
    if (nth == 0) {
        const char *env = getenv("MYNAH_ASR_THREADS");
        long n = env ? atol(env) : sysconf(_SC_NPROCESSORS_ONLN);
        if (n < 1) n = 1;
        if (n > PF_MAX_THREADS) n = PF_MAX_THREADS;
        nth = (int)n;
    }
    return nth;
}

typedef struct {
    void (*fn)(void *, int);
    void *ctx;
    atomic_int next;
    int n;
} pf_state;

static void pf_run(pf_state *st) {
    for (;;) {
        const int i = atomic_fetch_add_explicit(&st->next, 1, memory_order_relaxed);
        if (i >= st->n) break;
        st->fn(st->ctx, i);
    }
}

/* ------------------------------------------------------------ persistent pool
 * Workers are created on the first parallel_for and sleep on a condvar: no
 * pthread_create/join in the hot path (before: thousands of spawns per batched
 * transcription). ONE dispatch at a time (g_pool_mu): when the pool is busy —
 * concurrent calls from the server workers — the caller runs inline and serial,
 * which is already parallel ACROSS requests (no oversubscription).
 * The workers are detached and live until process exit (like the BLAS pools). */
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_job_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_job_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER;
static pf_state *g_job;
static unsigned g_gen;
static int g_pending;
static int g_workers;
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;

static void *pool_worker(void *arg) {
    (void)arg;
    unsigned seen = 0;
    pthread_mutex_lock(&g_job_mu);
    for (;;) {
        while (g_gen == seen) pthread_cond_wait(&g_job_cv, &g_job_mu);
        seen = g_gen;
        pf_state *job = g_job;
        pthread_mutex_unlock(&g_job_mu);
        pf_run(job);
        pthread_mutex_lock(&g_job_mu);
        if (--g_pending == 0) pthread_cond_signal(&g_done_cv);
    }
    return NULL;   /* mai raggiunto */
}

static void pool_init(void) {
    const int nth = mynah_asr_num_threads();
    for (int k = 0; k < nth - 1; k++) {
        pthread_t tid;
        if (pthread_create(&tid, NULL, pool_worker, NULL) == 0) {
            pthread_detach(tid);
            g_workers++;
        }
    }
}

/* Inside a parallel_for the cores belong to the workers: if every worker calls
 * cblas with multi-threaded OpenBLAS the result is catastrophic oversubscription
 * (measured on a 22-core EPYC: batch 4×60 s = 257 s instead of ~10 s). BLAS is
 * forced single-threaded for the duration of the region and restored on exit.
 * Accelerate (macOS) handles the nesting through GCD and does not need this.
 * Weak symbol as in qwen-tts (qwen_tts_kernels.c): resolved only when linked
 * against OpenBLAS; an explicit OPENBLAS_NUM_THREADS in the environment always
 * wins. */
#if defined(__GNUC__) && !defined(__APPLE__)
extern void openblas_set_num_threads(int) __attribute__((weak));
#endif

static void blas_set_threads(int n) {
#if defined(__GNUC__) && !defined(__APPLE__)
    if (getenv("OPENBLAS_NUM_THREADS")) return;   /* an explicit choice by the user */
    if (openblas_set_num_threads) openblas_set_num_threads(n > 0 ? n : 1);
#else
    (void)n;
#endif
}

void mynah_asr_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx) {
    if (n <= 0) return;
    const int nth = mynah_asr_num_threads();
    if (nth <= 1 || n == 1) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    pthread_once(&g_pool_once, pool_init);

    pf_state st = {.fn = fn, .ctx = ctx, .n = n};
    atomic_init(&st.next, 0);
    /* BLAS quota per worker: with only 2 jobs each concurrent cblas can use half
     * the cores (measured on a 22-core EPYC, batch 2×60 s nemotron: 15.7→33.3×
     * aggregate realtime); from 3 workers up, concurrent OpenBLAS calls fight
     * over the internal lock and a quota >1 makes it WORSE (B=8: 34→11×) →
     * single-threaded. */
    const int active = n < nth ? n : nth;
    blas_set_threads(active <= 2 ? nth / active : 1);
    if (g_workers == 0 || pthread_mutex_trylock(&g_pool_mu) != 0) {
        pf_run(&st);              /* pool assente o occupato: inline */
        blas_set_threads(nth);
        return;
    }
    pthread_mutex_lock(&g_job_mu);
    g_job = &st;
    g_pending = g_workers;
    g_gen++;
    pthread_cond_broadcast(&g_job_cv);
    pthread_mutex_unlock(&g_job_mu);
    pf_run(&st);                  /* the caller does its share too */
    pthread_mutex_lock(&g_job_mu);
    while (g_pending > 0) pthread_cond_wait(&g_done_cv, &g_job_mu);
    pthread_mutex_unlock(&g_job_mu);
    pthread_mutex_unlock(&g_pool_mu);
    blas_set_threads(nth);
}
