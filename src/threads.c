#include "threads.h"

#include "backend.h"   /* mynah_asr_gemm_provider(): who owns the other pool */

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
 * Workers are created on the first parallel_for and then SPIN-THEN-PARK: they
 * watch an atomic generation counter for a bounded time before sleeping on the
 * condvar.  No pthread_create/join in the hot path, and no kernel round trip
 * between two dispatches that are microseconds apart.
 *
 * WHY THE SPIN.  The cost of a dispatch is not an abstraction: a condvar
 * broadcast wakes N threads through the kernel and the completion wakes the
 * caller back, and it is paid per GEMM.  Measured on the M1 dev host before
 * this change, the representative shapes under 2M MACs -- a streaming step's
 * attention GEMMs, dozens of them back to back -- cost MORE with eight threads
 * than with one.  src/sgemm.c now refuses to split work that small
 * (SG_PARALLEL_MIN_WORK per thread), which stops the pool being used where it
 * cannot pay; the spin is the other half, so that the work just above that line
 * is not eaten by the wake-up either.
 *
 * ONE dispatch at a time (g_pool_mu): when the pool is busy -- concurrent calls
 * from server workers -- the caller runs inline and serial, which is already
 * parallel ACROSS requests (no oversubscription).  The workers are detached and
 * live until process exit (like the BLAS pools).
 *
 * The spin budget is bounded in REAL TIME (MYNAH_ASR_POOL_SPIN_US, default 50)
 * and is per wait, so an idle worker burns at most that much of its own core
 * before parking: between two streaming chunks (320 ms apart at lookahead 3)
 * every worker is parked.  0 disables the spin and restores the pure condvar
 * pool, which is how the two are compared.
 *
 * WHAT IT COSTS TO GET WRONG.  Both rendezvous have a missed-wakeup hazard: a
 * worker that decides to park just as a job is published, and a caller that
 * parks just as the last worker finishes.  The generation store and the parked
 * count are therefore SEQUENTIALLY CONSISTENT on both sides -- with a single
 * total order, a worker that waited must have incremented `parked` before the
 * publisher's load of it -- and the completion side does not gamble at all: the
 * worker that takes the count to zero always takes the mutex and signals.  One
 * uncontended lock per dispatch against a lost wake-up is not a trade. */
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_job_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_job_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER;
static pf_state *g_job;
static atomic_uint g_gen;
static atomic_int g_pending;
static atomic_int g_parked;
static int g_workers;
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;

/* The meter.  A pool that silently stopped spinning, or one that spins and
 * never catches anything, both look like "it works"; these say which
 * (ENGINEERING.md §6). Relaxed: one per dispatch or per wait, never per task. */
static atomic_ullong g_m_dispatches, g_m_worker_spin, g_m_worker_park;
static atomic_ullong g_m_caller_spin, g_m_caller_park, g_m_inline;

void mynah_asr_pool_stats_get(mynah_asr_pool_stats *out) {
    if (!out) return;
    out->dispatches  = atomic_load_explicit(&g_m_dispatches, memory_order_relaxed);
    out->worker_spin = atomic_load_explicit(&g_m_worker_spin, memory_order_relaxed);
    out->worker_park = atomic_load_explicit(&g_m_worker_park, memory_order_relaxed);
    out->caller_spin = atomic_load_explicit(&g_m_caller_spin, memory_order_relaxed);
    out->caller_park = atomic_load_explicit(&g_m_caller_park, memory_order_relaxed);
    out->inline_runs = atomic_load_explicit(&g_m_inline, memory_order_relaxed);
    out->spin_us     = mynah_asr_pool_spin_us();
    out->workers     = g_workers;
}

void mynah_asr_pool_stats_reset(void) {
    atomic_store_explicit(&g_m_dispatches, 0, memory_order_relaxed);
    atomic_store_explicit(&g_m_worker_spin, 0, memory_order_relaxed);
    atomic_store_explicit(&g_m_worker_park, 0, memory_order_relaxed);
    atomic_store_explicit(&g_m_caller_spin, 0, memory_order_relaxed);
    atomic_store_explicit(&g_m_caller_park, 0, memory_order_relaxed);
    atomic_store_explicit(&g_m_inline, 0, memory_order_relaxed);
}

int mynah_asr_pool_spin_us(void) {
    static int us = -1;
    if (us < 0) {
        const char *env = getenv("MYNAH_ASR_POOL_SPIN_US");
        long v = env ? atol(env) : 50;
        if (v < 0) v = 0;
        if (v > 10000) v = 10000;   /* 10 ms: past this it is not a spin */
        us = (int)v;
    }
    return us;
}

static void pf_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

static double pf_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Spin until `pred(arg)` or the budget runs out. 1 = the predicate won. */
static int pf_spin_until(int (*pred)(void *), void *arg) {
    const int us = mynah_asr_pool_spin_us();
    if (us <= 0) return pred(arg);
    const double deadline = pf_now() + (double)us * 1e-6;
    for (;;) {
        for (int k = 0; k < 64; k++) {
            if (pred(arg)) return 1;
            pf_relax();
        }
        if (pf_now() >= deadline) return pred(arg);
    }
}

static int pred_gen_moved(void *arg) {
    return atomic_load_explicit(&g_gen, memory_order_acquire) != *(unsigned *)arg;
}
static int pred_done(void *arg) {
    (void)arg;
    return atomic_load_explicit(&g_pending, memory_order_acquire) <= 0;
}

static void *pool_worker(void *arg) {
    (void)arg;
    /* ZERO, not the current generation, and this is load-bearing.  pool_init
     * runs under pthread_once inside the first dispatch, so a worker can reach
     * its first read AFTER that dispatch has already published its job and
     * counted this thread in g_pending.  A worker that started from the live
     * generation would then see gen == seen, park, and wait for a job that was
     * published before it looked -- while the caller waits for a pending count
     * that will never reach zero.  Generations only ever increase and the first
     * job is generation 1, so starting from 0 means the first check always
     * finds work, which is exactly right for a thread the dispatcher is already
     * counting on.
     *
     * Found by running the A/B arm: with the default spin the worker catches
     * the bump inside its spin window and the race never fires.  The pure
     * condvar arm (MYNAH_ASR_POOL_SPIN_US=0) deadlocked on the first dispatch,
     * every time. A knob whose other setting is never exercised is a knob that
     * hides bugs. */
    unsigned seen = 0;
    for (;;) {
        if (!pf_spin_until(pred_gen_moved, &seen)) {
            pthread_mutex_lock(&g_job_mu);
            atomic_fetch_add(&g_parked, 1);              /* seq_cst on purpose */
            while (atomic_load(&g_gen) == seen)
                pthread_cond_wait(&g_job_cv, &g_job_mu);
            atomic_fetch_sub(&g_parked, 1);
            pthread_mutex_unlock(&g_job_mu);
            atomic_fetch_add_explicit(&g_m_worker_park, 1, memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(&g_m_worker_spin, 1, memory_order_relaxed);
        }
        seen = atomic_load_explicit(&g_gen, memory_order_acquire);
        pf_run(g_job);   /* published before the generation bump (release) */
        if (atomic_fetch_sub_explicit(&g_pending, 1, memory_order_acq_rel) == 1) {
            /* the caller may be parked and may have decided that after our last
             * look, so this never guesses */
            pthread_mutex_lock(&g_job_mu);
            pthread_cond_signal(&g_done_cv);
            pthread_mutex_unlock(&g_job_mu);
        }
    }
    return NULL;   /* never reached */
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

/* Per-call BLAS thread budget (see threads.h). 0 = never set -> num_threads.
 * `applied` caches what the knob was last given: openblas_set_num_threads is
 * process-global and not free, and parallel_for touches it twice per region, so
 * skipping the no-op writes matters. The mutex only serializes the rare change,
 * never the read path. */
static int g_blas_budget;
static int g_blas_applied;
static pthread_mutex_t g_blas_mu = PTHREAD_MUTEX_INITIALIZER;

/* IS THERE A SECOND POOL IN THIS PROCESS AT ALL?
 *
 * The whole budget mechanism exists because OpenBLAS runs its own team of
 * threads next to ours.  With BLAS=none there is no second team: src/sgemm.c
 * dispatches onto the SAME mynah_asr_parallel_for, so there is exactly one
 * pool and its width is the only number there is.  Splitting a budget then
 * would not divide anything — it would just print a smaller number on
 * /v1/health than the pool the process actually runs, which is precisely the
 * kind of plausible fiction ENGINEERING.md §6 forbids.
 *
 * So the public functions stay (the tests and the server read them) and become
 * HONEST rather than inert: with `own` the budget is the pool width, always,
 * and set_concurrency records the caller's declaration without pretending to
 * act on it.  Accelerate keeps its existing bookkeeping semantics unchanged —
 * it also takes no knob, but that behaviour is what the server's health output
 * and tests/test_server_concurrency.sh were written against, and changing it
 * is not this item's business.  The OpenBLAS path is untouched. */
static int blas_own_pool(void) {
    static int cached = -1;
    if (cached < 0) cached = strcmp(mynah_asr_gemm_provider(), "own") == 0;
    return cached;
}

static void blas_apply(int n) {
    if (blas_own_pool()) return;   /* nothing to apply it to */
    if (n < 1) n = 1;
    pthread_mutex_lock(&g_blas_mu);
    if (g_blas_applied != n) {
        g_blas_applied = n;
        blas_set_threads(n);
    }
    pthread_mutex_unlock(&g_blas_mu);
}

int mynah_asr_blas_budget(void) {
    if (blas_own_pool()) return mynah_asr_num_threads();
    pthread_mutex_lock(&g_blas_mu);
    const int b = g_blas_budget;
    pthread_mutex_unlock(&g_blas_mu);
    return b > 0 ? b : mynah_asr_num_threads();
}

void mynah_asr_blas_set_concurrency(int n_inflight) {
    const int nth = mynah_asr_num_threads();
    if (n_inflight < 1) n_inflight = 1;
    int budget = nth / n_inflight;
    if (budget < 1) budget = 1;
    pthread_mutex_lock(&g_blas_mu);
    g_blas_budget = budget;
    pthread_mutex_unlock(&g_blas_mu);
    blas_apply(budget);
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
     * the budget (measured on a 22-core EPYC, batch 2×60 s nemotron: 15.7→33.3×
     * aggregate realtime); from 3 workers up, concurrent OpenBLAS calls fight
     * over the internal lock and a quota >1 makes it WORSE (B=8: 34→11×) →
     * single-threaded. The ceiling is the budget, not nth: a server with several
     * inferences in flight has already lowered it, and restoring nth here would
     * silently undo that cap. */
    const int budget = mynah_asr_blas_budget();
    const int active = n < nth ? n : nth;
    blas_apply(active <= 2 ? budget / active : 1);
    if (g_workers == 0 || pthread_mutex_trylock(&g_pool_mu) != 0) {
        atomic_fetch_add_explicit(&g_m_inline, 1, memory_order_relaxed);
        pf_run(&st);              /* no pool or busy: run inline */
        blas_apply(budget);
        return;
    }
    atomic_fetch_add_explicit(&g_m_dispatches, 1, memory_order_relaxed);
    g_job = &st;
    atomic_store_explicit(&g_pending, g_workers, memory_order_relaxed);
    atomic_fetch_add(&g_gen, 1);                  /* seq_cst: publishes the job */
    if (atomic_load(&g_parked) > 0) {             /* seq_cst: pairs with the park */
        pthread_mutex_lock(&g_job_mu);
        pthread_cond_broadcast(&g_job_cv);
        pthread_mutex_unlock(&g_job_mu);
    }
    pf_run(&st);                  /* the caller does its share too */
    if (pf_spin_until(pred_done, NULL)) {
        atomic_fetch_add_explicit(&g_m_caller_spin, 1, memory_order_relaxed);
    } else {
        pthread_mutex_lock(&g_job_mu);
        while (atomic_load_explicit(&g_pending, memory_order_acquire) > 0)
            pthread_cond_wait(&g_done_cv, &g_job_mu);
        pthread_mutex_unlock(&g_job_mu);
        atomic_fetch_add_explicit(&g_m_caller_park, 1, memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_pool_mu);
    blas_apply(budget);
}

void mynah_asr_blas_after_fork(void) {
    pthread_mutex_init(&g_blas_mu, NULL);
    g_blas_applied = 0;
}

void mynah_asr_threadpool_after_fork(void) {
    pthread_mutex_init(&g_pool_mu, NULL);
    pthread_mutex_init(&g_job_mu, NULL);
    pthread_cond_init(&g_job_cv, NULL);
    pthread_cond_init(&g_done_cv, NULL);
    g_job = NULL;
    atomic_store(&g_gen, 0);
    atomic_store(&g_pending, 0);
    atomic_store(&g_parked, 0);
    g_workers = 0;
    {
        static const pthread_once_t fresh = PTHREAD_ONCE_INIT;
        memcpy(&g_pool_once, &fresh, sizeof(g_pool_once));
    }
    mynah_asr_blas_after_fork();
}
