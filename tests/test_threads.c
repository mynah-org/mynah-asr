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
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

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

/* The pool's FIRST dispatch, in a fresh process, at a given spin budget.
 *
 * A deadlock gate, not a timing one, and it needs its own process because
 * MYNAH_ASR_POOL_SPIN_US and MYNAH_ASR_THREADS are both read once and cached.
 * It guards a failure that happens only on the FIRST dispatch: pool_init runs
 * under pthread_once INSIDE that dispatch, so a worker can reach its first look
 * at the generation counter after the job it is already counted in was
 * published. It cost a real hang -- every worker parked on the job condvar, the
 * caller parked on the completion condvar, 0% cpu -- and it fired only with the
 * spin DISABLED: at the default the worker catches the bump inside its spin
 * window and the race is invisible. The arm that exists for the A/B is exactly
 * the arm that has to be exercised.
 *
 * `after_fork` picks which contract is under test. 0: a pristine child, so this
 * must run BEFORE the parent has a pool or has cached a thread count. 1: the
 * child inherits a WARM pool whose worker threads did not survive fork() and
 * calls mynah_asr_threadpool_after_fork() -- the prefork server's path, whose
 * failure mode (proven here) is a hang and not a crash.
 *
 * The parent gives the child a deadline, because a deadlock is not an assertion
 * that fails: it is a test that never returns. */
static int pool_child(const char *spin_us, const char *threads, int after_fork) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        setenv("MYNAH_ASR_POOL_SPIN_US", spin_us, 1);
        setenv("MYNAH_ASR_THREADS", threads, 1);
        if (after_fork) mynah_asr_threadpool_after_fork();
        alarm(15);                      /* belt: SIGALRM if the pool wedges */
        atomic_store(&calls, 0);
        mynah_asr_parallel_for(256, noop, NULL);
        mynah_asr_parallel_for(256, noop, NULL);   /* and the second, warm */
        _exit(atomic_load(&calls) == 512 ? 0 : 2);
    }
    for (int i = 0; i < 200; i++) {     /* 20 s, past the child's own alarm */
        int st = 0;
        const pid_t r = waitpid(pid, &st, WNOHANG);
        if (r == pid) return (WIFEXITED(st) && WEXITSTATUS(st) == 0) ? 0 : 1;
        usleep(100000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, NULL, 0);
    return 1;
}

int main(void) {
    /* FIRST, before this process has a pool or a cached thread count: the
     * children below must be pristine for their environment to mean anything. */
    {
        static const char *spins[] = {"0", "50", "2000"};
        for (unsigned a = 0; a < sizeof(spins) / sizeof(spins[0]); a++) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "a fresh pool completes its first dispatch at "
                     "MYNAH_ASR_POOL_SPIN_US=%s", spins[a]);
            CHECK(pool_child(spins[a], "8", 0) == 0, msg);
        }
    }

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

    /* the prefork contract: this process now HAS a pool, whose threads will not
     * survive fork(). A child that calls threadpool_after_fork rebuilds one and
     * completes; one that does not would wait forever for workers that no
     * longer exist, which is how this gate was written in the first place. */
    CHECK(pool_child("0", "8", 1) == 0,
          "a child that calls threadpool_after_fork dispatches again (spin off)");
    CHECK(pool_child("50", "8", 1) == 0,
          "and with the spin on");

    printf("test_threads: %s\n", failures ? "FAIL" : "OK");
    return failures;
}
