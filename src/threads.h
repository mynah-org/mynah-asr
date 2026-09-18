/* Minimal pthread parallel-for for the independent CPU loops (mel frames,
 * depthwise channels, batch segments). Tasks must write to disjoint regions:
 * the result is BIT-IDENTICAL to the serial loop by construction (same code
 * per task, only on different threads).
 * Thread count: env MYNAH_ASR_THREADS, default = online cores. */
#ifndef MYNAH_ASR_THREADS_H
#define MYNAH_ASR_THREADS_H

int mynah_asr_num_threads(void);

/* Runs fn(ctx, i) for i in [0, n): the tasks are spread over
 * min(n, mynah_asr_num_threads()) threads (the caller takes part).
 * With n <= 1, or a single thread, it runs in place without spawning. */
void mynah_asr_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx);

/* BLAS thread budget — how many threads ONE inference may ask of BLAS.
 *
 * Default = mynah_asr_num_threads(): a single inference owns every core, which
 * is right for the CLI. A server running several inferences AT THE SAME TIME
 * must lower it: each call otherwise spawns nth BLAS threads and they fight
 * over the OpenBLAS internal lock (measured on the A100 host: from 4 concurrent
 * requests up, aggregate throughput collapses; capping the threads restored it,
 * which until now needed OPENBLAS_NUM_THREADS set by hand).
 *
 * Declare the concurrency, not the thread count: the policy (nth/n_inflight)
 * lives here so it stays in one place. mynah_asr_parallel_for takes the budget
 * as its ceiling and restores IT, not nth, when the region ends.
 *
 * OpenBLAS only: Accelerate nests through GCD and needs no knob, so on macOS
 * this only bookkeeps (mynah_asr_blas_budget stays truthful for tests/health).
 *
 * WITH NO BLAS IN THE PROCESS (make BLAS=none, the Linux default) there is no
 * second pool to budget: src/sgemm.c dispatches onto THIS pool, so
 * mynah_asr_blas_budget() is the pool width and nothing can move it.
 * set_concurrency still records the declaration — the server calls it and
 * /v1/health prints the result — but it does not invent a smaller number for a
 * team that does not exist. Ask mynah_asr_gemm_provider() (src/backend.h) which
 * of the three builds this is; do not infer it from the budget.
 *
 * Thread-safe; the knob is touched only when the value really changes, so a
 * steady-state server pays nothing. */
void mynah_asr_blas_set_concurrency(int n_inflight);
int mynah_asr_blas_budget(void);

/* Called in a forked CHILD before its first dispatch (server/prefork.c). The
 * pool's worker threads do not survive fork(); this forgets them and re-arms the
 * one-time init so the first parallel_for in the child builds a fresh pool of
 * mynah_asr_num_threads() workers, which prefork has just set through
 * MYNAH_ASR_THREADS. The mutexes are re-initialised rather than trusted, because a
 * fork taken while another thread held one would inherit it locked. Also resets
 * the cached BLAS thread count so the next apply really reaches the library.
 * Must be called from the only thread in the child. */
void mynah_asr_threadpool_after_fork(void);
void mynah_asr_blas_after_fork(void);

#endif
