/* Self-test of the BLAS thread budget (no model required — it runs in CI too).
 *
 * The arithmetic is trivial; what this guards is the interaction that is easy to
 * break: mynah_asr_parallel_for lowers the BLAS threads for the duration of a
 * parallel region and must restore the BUDGET, not mynah_asr_num_threads(). Get
 * that wrong and a server with several inferences in flight has its cap silently
 * undone after the first parallel region — invisible in output, only slower.
 *
 * MYNAH_ASR_THREADS is pinned so the expected values do not depend on the host's
 * core count (num_threads caches on first call, so it must be set before any).
 * Exit: 0 ok, 1 fail. */
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/backend.h"
#include "../src/threads.h"

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("threads FAIL: %s\n", msg); failures = 1; } \
    else printf("threads ok:   %s\n", msg); } while (0)

/* atomic because parallel_for really does call this from several threads: a plain
 * ++ loses updates (it passed at -O3 and failed under -O2/UBSan — a reminder that
 * the "disjoint writes" contract in threads.h is on the caller) */
static atomic_int calls;
static void noop(void *ctx, int i) { (void)ctx; (void)i; atomic_fetch_add(&calls, 1); }

int main(void) {
    setenv("MYNAH_ASR_THREADS", "8", 1);
    const int nth = mynah_asr_num_threads();
    CHECK(nth == 8, "MYNAH_ASR_THREADS honoured (8)");

    /* default: one inference owns every core (the CLI case) */
    CHECK(mynah_asr_blas_budget() == nth, "default budget == num_threads");

    /* WHAT THE BUDGET MEANS DEPENDS ON WHETHER THERE IS A SECOND POOL.
     * With a vendor BLAS linked the budget divides its team among the
     * inferences in flight. With BLAS=none there is no second team — sgemm
     * runs on this very pool — so the budget is the pool width and nothing may
     * move it. Asserting the division there would be asserting a fiction, so
     * the test asks the seam which build this is instead of assuming. */
    const int own_pool = strcmp(mynah_asr_gemm_provider(), "own") == 0;
    printf("threads: gemm provider = %s (%s)\n", mynah_asr_gemm_provider(),
           own_pool ? "one pool: the budget is its width"
                    : "a second BLAS pool exists: the budget divides it");

    if (own_pool) {
        int stable = 1;
        const int inflight[] = {1, 2, 3, 4, 8, 16, 0, -1};
        for (size_t i = 0; i < sizeof(inflight) / sizeof(inflight[0]); i++) {
            mynah_asr_blas_set_concurrency(inflight[i]);
            if (mynah_asr_blas_budget() != nth) stable = 0;
        }
        CHECK(stable, "no BLAS in the process: the budget stays the pool width "
                      "at every declared concurrency");
    } else {
        struct { int inflight, want; } cases[] = {
            {1, 8}, {2, 4}, {3, 2}, {4, 2}, {8, 1}, {16, 1},
            {0, 8}, {-1, 8},   /* nonsense in, clamped to "one caller" */
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            mynah_asr_blas_set_concurrency(cases[i].inflight);
            char msg[96];
            snprintf(msg, sizeof(msg), "concurrency %d -> budget %d",
                     cases[i].inflight, cases[i].want);
            CHECK(mynah_asr_blas_budget() == cases[i].want, msg);
        }
    }

    /* the regression this file exists for: a parallel region must leave the
     * budget where it found it (it may lower the knob while inside) */
    mynah_asr_blas_set_concurrency(4);
    const int before = mynah_asr_blas_budget();
    atomic_store(&calls, 0);
    mynah_asr_parallel_for(64, noop, NULL);
    CHECK(atomic_load(&calls) == 64, "parallel_for ran every task");
    CHECK(mynah_asr_blas_budget() == before, "parallel_for preserves the budget");

    /* and the same when it runs in place (n == 1 takes the early-return path) */
    mynah_asr_parallel_for(1, noop, NULL);
    CHECK(mynah_asr_blas_budget() == before, "in-place parallel_for preserves the budget");

    mynah_asr_blas_set_concurrency(1);
    CHECK(mynah_asr_blas_budget() == nth, "back to rest -> full budget");

    printf("test_threads: %s\n", failures ? "FAIL" : "OK");
    return failures;
}
