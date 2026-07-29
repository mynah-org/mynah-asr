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

#endif
