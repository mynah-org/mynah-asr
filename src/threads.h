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
 * Thread-safe; the knob is touched only when the value really changes, so a
 * steady-state server pays nothing. */
void mynah_asr_blas_set_concurrency(int n_inflight);
int mynah_asr_blas_budget(void);

#endif
