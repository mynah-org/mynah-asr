/* Prefork serving topology. The contract and the "how to choose W" procedure
 * are in prefork.h; this file is the mechanism.
 *
 * Three properties are worth naming here because every failure mode of a
 * prefork server is one of them going wrong:
 *
 *   WHEN WE FORK. After the model model is open, so the mapped weights are one
 *   physical copy behind W address spaces; before any inference and before any
 *   other thread in this process exists, so no child can inherit a mutex that
 *   was locked by a thread the fork did not copy. src/threads.c registers a
 *   pthread_atfork child handler that rebuilds the pool, so a child whose pool
 *   was already running in the parent gets a fresh one rather than an empty
 *   one -- without it the first dispatch in a child broadcasts to nobody and
 *   then waits for it, which is a deadlock with no error message.
 *
 *   HOW THE DESCRIPTOR MOVES. The parent accepts and sends the descriptor over
 *   an AF_UNIX socketpair with SCM_RIGHTS, then closes its own copy. There is
 *   exactly one moment where both processes hold it, between sendmsg and
 *   close, and if sendmsg fails the parent still owns it and closes it. A
 *   descriptor is never in flight and unowned.
 *
 *   HOW LOAD IS COUNTED. The parent's only view of a worker is one byte per
 *   finished connection, sent back up the same socketpair. `active[w]` is
 *   dispatched minus finished, the routing decision is "smallest active", and
 *   a worker at `slots_per` is skipped. That makes the accounting the whole
 *   correctness story of the router: a worker path that forgets to report a
 *   finished connection leaks a slot permanently, and one that reports twice
 *   over-admits. See mynah_asr_prefork_conn_done()'s caller in server/main.c --
 *   there is one, on the job's last reference, which is the only place in that
 *   file where a connection is provably finished with.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "fleet.h"
#include "prefork.h"

#include "metrics.h"

#include "../src/flags.h"    /* build id, BLAS provider, SIMD profile */
#include "../src/qmat.h"     /* the int8 kernel predicate, for build_info */

#include "../src/threads.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <sys/sysctl.h>   /* kern.ipc.somaxconn: the cap listen() is clamped to */
#endif

#define PREFORK_MAX_WORKERS 256
/* Upper bound on the cpu ids we will enumerate. Sized for a large server, not
 * for CPU_SETSIZE: the list is a plan, and a plan over 1024 cpus is already
 * well past the point where one process should be routing for all of them. */
#define CPU_LIST_MAX 1024

/* Where the cpu topology lives. A macro and not a literal so a test can point
 * it at a fabricated tree: the SMT sibling ordering below is the one piece of
 * Linux-only logic complicated enough to be wrong, and on a machine with no
 * /sys it would otherwise ship having never been executed. */
#ifndef MYNAH_ASR_CPU_TOPOLOGY_ROOT
#define MYNAH_ASR_CPU_TOPOLOGY_ROOT "/sys/devices/system/cpu"
#endif

/* Same idea for the cgroup tree. Unlike the topology read, parsing `cpu.max`
 * is not Linux-only *code* -- it is an ordinary file read and a division, so
 * it compiles and RUNS everywhere and simply finds nothing where there is no
 * cgroupfs. Pointing this at a fabricated directory is therefore a real
 * execution of the parser on any platform, which is the only way the quota
 * arithmetic gets exercised on a developer machine at all. */
#ifndef MYNAH_ASR_CGROUP_ROOT
#define MYNAH_ASR_CGROUP_ROOT "/sys/fs/cgroup"
#endif

/* Both roots take an environment override in preference to the compile-time
 * default, so a test does not need its own build of this translation unit. */
#if defined(__linux__)   /* the only caller is the sysfs reader below */
static const char *topology_root(void) {
    const char *e = getenv("MYNAH_ASR_CPU_TOPOLOGY_ROOT");
    return (e != NULL && e[0] != '\0') ? e : MYNAH_ASR_CPU_TOPOLOGY_ROOT;
}
#endif
static const char *cgroup_root(void) {
    const char *e = getenv("MYNAH_ASR_CGROUP_ROOT");
    return (e != NULL && e[0] != '\0') ? e : MYNAH_ASR_CGROUP_ROOT;
}

/* ------------------------------------------------------------- cpu budget
 *
 * The residual the reference names and did not close: sched_getaffinity says
 * WHICH cpus we may use, the cgroup quota says HOW MUCH of them. They are
 * different limits and a container routinely sets only the second. A pod with
 * `cpu: "4"` and no cpuset has an affinity mask covering all 64 cores of its
 * host and a budget of four: planning W*T from the mask alone builds a server
 * that is throttled the instant it works, and the throttling appears as
 * latency with no CPU-bound thread to blame it on.
 *
 * We do not silently re-plan on the quota -- it is a rate over a period, not a
 * set of cpus, and W workers of T threads is still a legitimate shape under
 * one. We WARN, because "you asked for 16 threads inside a 4-cpu budget" is a
 * sentence an operator can act on and a 4x RTF regression is not.
 *
 * Returns 1 and fills *cpus when a finite quota is in force, 0 otherwise. */
static int cgroup_cpu_budget(double *cpus, char *detail, size_t detail_cap) {
    const char *root = cgroup_root();
    char path[512];
    if (detail != NULL && detail_cap > 0) detail[0] = '\0';

    /* cgroup v2: one file, "$MAX $PERIOD", MAX either a number or "max". */
    snprintf(path, sizeof(path), "%s/cpu.max", root);
    FILE *f = fopen(path, "r");
    if (f != NULL) {
        char quota[64];
        long period = 0;
        quota[0] = '\0';
        const int got = fscanf(f, "%63s %ld", quota, &period);
        fclose(f);
        if (got == 2 && period > 0 && strcmp(quota, "max") != 0) {
            const double q = atof(quota);
            if (q > 0.0) {
                *cpus = q / (double)period;
                if (detail != NULL) {
                    snprintf(detail, detail_cap, "cgroup v2 cpu.max %s %ld",
                             quota, period);
                }
                return 1;
            }
        }
        if (got >= 1 && strcmp(quota, "max") == 0) return 0;   /* explicitly none */
    }

    /* cgroup v1: two files, and a quota of -1 means unlimited. */
    long q = -1, period = 0;
    snprintf(path, sizeof(path), "%s/cpu/cpu.cfs_quota_us", root);
    f = fopen(path, "r");
    if (f == NULL) {
        snprintf(path, sizeof(path), "%s/cpu.cfs_quota_us", root);
        f = fopen(path, "r");
    }
    if (f != NULL) { if (fscanf(f, "%ld", &q) != 1) q = -1; fclose(f); }
    snprintf(path, sizeof(path), "%s/cpu/cpu.cfs_period_us", root);
    f = fopen(path, "r");
    if (f == NULL) {
        snprintf(path, sizeof(path), "%s/cpu.cfs_period_us", root);
        f = fopen(path, "r");
    }
    if (f != NULL) { if (fscanf(f, "%ld", &period) != 1) period = 0; fclose(f); }
    if (q > 0 && period > 0) {
        *cpus = (double)q / (double)period;
        if (detail != NULL) {
            snprintf(detail, detail_cap, "cgroup v1 cfs_quota/period %ld/%ld",
                     q, period);
        }
        return 1;
    }
    return 0;
}

/* Warns when the CPU budget contradicts the plan. Separate from the planner on
 * purpose: the plan is a statement about cpus, this is a statement about time,
 * and conflating them is how "we planned 32 and measured 4" happens. */
static void warn_cpu_budget(int workers, int threads, int ncpu, FILE *out) {
    double budget = 0.0;
    char detail[128];
    if (!cgroup_cpu_budget(&budget, detail, sizeof(detail))) return;

    fprintf(out, "prefork: cpu budget    %.2f cpus (%s)\n", budget, detail);
    const double want = (double)workers * (double)threads;
    if (want > budget + 0.01) {
        fprintf(out,
            "prefork: WARNING the CPU BUDGET CONTRADICTS THE PLAN. W*T = %d threads, "
            "but this cgroup allows %.2f cpus of runtime. The affinity mask says %d "
            "cpus and the quota says %.2f: the mask is WHICH cpus, the quota is HOW "
            "MUCH, and only the quota is enforced -- by throttling every worker at a "
            "period boundary with no busy thread to blame. Raise the quota, or plan "
            "W*T <= %.0f.\n",
            workers * threads, budget, ncpu, budget, budget);
    } else if ((double)ncpu > budget + 0.01) {
        fprintf(out,
            "prefork: note the affinity mask covers %d cpus but the budget is %.2f; "
            "the plan (W*T = %d) fits it. sysconf-based planning would not have.\n",
            ncpu, budget, workers * threads);
    }
}

/* ---------------------------------------------------- fork preconditions
 *
 * Two things must be true at the instant we fork, and neither fails loudly on
 * its own. They are checked here rather than trusted to the comment in
 * server/main.c that asserts them, because a comment is not a check and the
 * ordering it describes is one refactor away from being wrong.
 *
 * THE MUTEX QUESTION first, because it is what the audit is really about.
 * fork() copies the address space but only the calling thread. Every mutex in
 * the child is a bit-for-bit copy of its state at that instant: one another
 * thread held is inherited LOCKED, by a thread that does not exist in the
 * child, and can never be unlocked. Zeroing such a mutex is NOT a repair -- on
 * glibc a zeroed pthread_mutex_t happens to look unlocked, which is exactly
 * why the bug survives testing there, and on other implementations it is
 * simply corrupt. The only correct repair is pthread_mutex_init(); the only
 * way to not need one is to fork while single-threaded.
 *
 * The audit of THIS tree (mynah-asr, 2026-09-18):
 *
 *   src/threads.c  g_pool_mu, g_job_mu, g_job_cv, g_done_cv, g_pool_once
 *                  REINITIALIZED (not zeroed) by mynah_asr_threadpool_after_fork(),
 *                  called explicitly by the child below. There is no
 *                  pthread_atfork registration yet: the fork happens before
 *                  any thread exists, which is what makes the explicit call
 *                  sufficient, and check_fork_preconditions() verifies that.
 *   src/threads.c  g_blas_mu       REINITIALIZED by mynah_asr_blas_after_fork(),
 *                                  called from the pool hook so there is one
 *                                  handler to remember rather than two.
 *   src/qmat.c     g_stats_mutex   NOT repaired; safe only because no thread
 *                                  exists at the fork. Named here so the day
 *                                  a thread precedes the fork this is the
 *                                  first line somebody reads.
 *
 * Every lock this tree owns is therefore repaired. What is NOT repaired -- and
 * cannot be from here -- is a lock inside a library that spawned threads of its
 * own before the fork; a threaded BLAS is the one that actually happens, and on
 * a 32-core host it reaches the fork with 32 threads regardless of how narrow
 * the worker slices will be. That keeps "we fork from main, early" a
 * load-bearing invariant rather than a convention, so the thread count is
 * checked rather than assumed. */

/* Threads in this process, or -1 where the platform will not say. */
static int process_thread_count(void) {
#if defined(__linux__)
    DIR *d = opendir("/proc/self/task");
    if (d == NULL) return -1;
    int n = 0;
    const struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] != '.') ++n;
    }
    closedir(d);
    return n > 0 ? n : -1;
#elif defined(__APPLE__)
    thread_act_array_t list = NULL;
    mach_msg_type_number_t n = 0;
    if (task_threads(mach_task_self(), &list, &n) != KERN_SUCCESS) return -1;
    for (mach_msg_type_number_t i = 0; i < n; ++i) {
        mach_port_deallocate(mach_task_self(), list[i]);
    }
    vm_deallocate(mach_task_self(), (vm_address_t)list,
                  (vm_size_t)n * sizeof(*list));
    return (int)n;
#else
    return -1;
#endif
}

/* A resident GPU compute runtime, if one can be seen from inside the process.
 *
 * This is the reference's still-open bug and it earns the emphasis: a GPU
 * context does not survive fork() -- CUDA's own documentation says so. The
 * child inherits the handles and the device pointers but not the driver's
 * per-process state behind them, and what follows is not a crash. It is
 * kernels launched against a context that is no longer valid: WRONG AUDIO,
 * which nothing downstream of the decoder can tell from right audio. A wrong
 * answer is a far worse failure than a dead worker. `--backend cuda --prefork
 * N` must refuse, not try.
 *
 * cfg->gpu_backend_open is the authoritative signal and a caller that opens a
 * backend should set it. This scan is the backstop for a caller that does not:
 * a CUDA/NVIDIA runtime mapped into the process is unambiguous. Metal is
 * deliberately NOT grounds for refusal -- Metal.framework is pulled in
 * transitively by unrelated system frameworks, and a mapped library is not a
 * created device.
 *
 * Returns 1 and names what it found, 0 otherwise. */
static int gpu_runtime_resident(char *what, size_t cap) {
    static const char *const markers[] = {
        "libcuda.so", "libcudart", "libnvidia-ml", "libcuda.dylib", NULL
    };
#if defined(__linux__)
    FILE *f = fopen("/proc/self/maps", "r");
    if (f == NULL) return 0;
    char line[1024];
    while (fgets(line, sizeof(line), f) != NULL) {
        for (int i = 0; markers[i] != NULL; ++i) {
            if (strstr(line, markers[i]) != NULL) {
                snprintf(what, cap, "%s", markers[i]);
                fclose(f);
                return 1;
            }
        }
    }
    fclose(f);
    return 0;
#elif defined(__APPLE__)
    const uint32_t n = _dyld_image_count();
    for (uint32_t j = 0; j < n; ++j) {
        const char *name = _dyld_get_image_name(j);
        if (name == NULL) continue;
        for (int i = 0; markers[i] != NULL; ++i) {
            if (strstr(name, markers[i]) != NULL) {
                snprintf(what, cap, "%s", name);
                return 1;
            }
        }
    }
    return 0;
#else
    (void)what; (void)cap; (void)markers;
    return 0;
#endif
}

/* Everything that must hold before the first fork(). Returns 0 to proceed, -1
 * to refuse. Refusing is the point: these produce a WRONG ANSWER rather than a
 * crash, and a wrong answer that appears only under prefork is the most
 * expensive kind of bug this file could ship. */
static int check_fork_preconditions(const mynah_asr_prefork_config *cfg) {
    int bad = 0;

    char gpu[512];
    if (cfg->gpu_backend_open) {
        fprintf(stderr,
            "prefork: REFUSING TO FORK -- a GPU backend is open in this process.\n"
            "  A GPU context does not survive fork(): the child inherits the handles\n"
            "  but not the driver state behind them, so its kernels run against a\n"
            "  context that no longer exists. That does not crash. It produces wrong\n"
            "  audio, which nothing downstream can detect. Run prefork on the CPU\n"
            "  backend, or run the GPU backend without prefork.\n");
        bad = 1;
    } else if (gpu_runtime_resident(gpu, sizeof(gpu))) {
        fprintf(stderr,
            "prefork: REFUSING TO FORK -- a GPU compute runtime is mapped into this\n"
            "  process (%s). Even with no context created yet this is not a shape in\n"
            "  which forked workers can be trusted; see above. Set\n"
            "  MYNAH_ASR_PREFORK_ALLOW_GPU=1 only having established that no device state\n"
            "  exists.\n", gpu);
        if (getenv("MYNAH_ASR_PREFORK_ALLOW_GPU") == NULL) bad = 1;
        else fprintf(stderr, "prefork: MYNAH_ASR_PREFORK_ALLOW_GPU is set; continuing\n");
    }

    const int threads = process_thread_count();
    if (threads > 1) {
        /* CORRECTED. This warning used to name g_blas_mu (src/threads.c) and
         * g_stats_mutex (src/qmat.c) as having no atfork handler. They both
         * have one now, and a log line that describes a bug somebody already
         * fixed is worse than no line: the next reader budgets for a hang that
         * cannot happen and stops reading the rest of the warning, which is
         * still true.
         *
         * What remains true is the general statement. fork() copies one thread
         * and every lock in whatever state it was in, and the handlers only
         * cover the locks this repository knows about -- a threaded BLAS, an
         * allocator or any library that spawned a thread before the fork has
         * state no handler here touches. On this machine that is not
         * hypothetical: OpenBLAS builds a team sized to the HOST at its first
         * call, so a parent that has served a request anything reaches this line
         * with as many threads as the box has cores, whatever the worker slice
         * will be. The invariant is still "fork before any other thread
         * exists", and the count is printed because it is the evidence. */
        fprintf(stderr,
            "prefork: WARNING forking with %d threads in this process, not 1.\n"
            "  fork() copies one thread and every lock in whatever state it was in.\n"
            "  src/threads.c's pool locks and BLAS bookkeeping are reinitialised in\n"
            "  the child (mynah_asr_threadpool_after_fork), so those are covered --\n"
            "  but a library that spawned its own threads before this point (a\n"
            "  threaded BLAS is the usual one) has state no handler here can repair.\n"
            "  The fork belongs before the pool and the HTTP workers exist.\n"
            "  Set MYNAH_ASR_PREFORK_STRICT=1 to make this a refusal.\n", threads);
        if (getenv("MYNAH_ASR_PREFORK_STRICT") != NULL) {
            fprintf(stderr, "prefork: MYNAH_ASR_PREFORK_STRICT is set; refusing.\n");
            bad = 1;
        }
    } else if (threads < 0) {
        fprintf(stderr, "prefork: note this platform will not report its thread "
                        "count; the single-threaded-fork invariant is unchecked\n");
    }

    return bad ? -1 : 0;
}

/* ------------------------------------------------------------- cpu topology
 *
 * Everything in this section answers one question -- "which cpus may this
 * process use, and which of them are siblings of the same physical core" --
 * and answers it honestly on a platform that cannot tell us. */

/* Fills `out` with the cpu ids this process is allowed to run on and returns
 * how many. Plans on the MASK, not on the machine: sysconf() sees neither an
 * inherited taskset nor a cpuset cgroup, so a pinned or containerised run
 * would otherwise plan for cores it will never get. Returns the online count
 * with an identity list where no mask can be read. */
static int cpus_allowed(int *out, int max) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = 0;
        for (int c = 0; c < CPU_SETSIZE && n < max; ++c) {
            if (CPU_ISSET(c, &set)) out[n++] = c;
        }
        if (n > 0) return n;
    }
#endif
    {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (online < 1) online = 1;
        int n = (int)(online < (long)max ? online : (long)max);
        for (int i = 0; i < n; ++i) out[i] = i;
        return n;
    }
}

#if defined(__linux__)
static long sysfs_cpu_long(int cpu, const char *leaf) {
    char path[160];
    snprintf(path, sizeof(path), "%s/cpu%d/topology/%s",
             topology_root(), cpu, leaf);
    FILE *f = fopen(path, "r");
    if (f == NULL) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}
#endif

/* Reorders `cpus` so that the siblings of one physical core sit next to each
 * other, and returns 1 when it really did read the topology.
 *
 * This is the difference between isolation and the appearance of it. With SMT
 * enabled, Linux commonly enumerates the two threads of core 0 as cpu 0 and
 * cpu N/2. A contiguous slice of logical ids then hands worker 0 one thread of
 * each of the first N/2 cores and worker 1 the *other* thread of the same
 * cores: the two workers contend for every one of them while `taskset` shows
 * disjoint masks. Core-major order gives worker 0 cores 0..k with both their
 * threads instead. Identity order (and 0) when sysfs is unreadable, which is
 * every non-Linux platform. */
static int order_core_major(int *cpus, int n) {
#if defined(__linux__)
    int *pkg = (int *)malloc((size_t)n * sizeof(int));
    int *core = (int *)malloc((size_t)n * sizeof(int));
    int *out = (int *)malloc((size_t)n * sizeof(int));
    if (pkg == NULL || core == NULL || out == NULL) {
        free(pkg); free(core); free(out);
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        pkg[i] = (int)sysfs_cpu_long(cpus[i], "physical_package_id");
        core[i] = (int)sysfs_cpu_long(cpus[i], "core_id");
        if (pkg[i] < 0 || core[i] < 0) {
            free(pkg); free(core); free(out);
            return 0;
        }
    }
    int k = 0;
    for (int i = 0; i < n; ++i) {
        int seen = 0;
        for (int j = 0; j < i; ++j) {
            if (pkg[j] == pkg[i] && core[j] == core[i]) { seen = 1; break; }
        }
        if (seen) continue;
        out[k++] = cpus[i];
        for (int j = i + 1; j < n && k < n; ++j) {
            if (pkg[j] == pkg[i] && core[j] == core[i]) out[k++] = cpus[j];
        }
    }
    const int ok = (k == n);
    if (ok) memcpy(cpus, out, (size_t)n * sizeof(int));
    free(pkg); free(core); free(out);
    return ok;
#else
    (void)cpus; (void)n;
    return 0;
#endif
}

/* Distinct physical cores among `cpus`, or 0 when the topology is unreadable
 * (which is not the same as "one core" and must not be printed as such). */
static int count_physical_cores(const int *cpus, int n) {
#if defined(__linux__)
    int cores = 0;
    for (int i = 0; i < n; ++i) {
        const long pi = sysfs_cpu_long(cpus[i], "physical_package_id");
        const long ci = sysfs_cpu_long(cpus[i], "core_id");
        if (pi < 0 || ci < 0) return 0;
        int dup = 0;
        for (int j = 0; j < i; ++j) {
            if (sysfs_cpu_long(cpus[j], "physical_package_id") == pi &&
                sysfs_cpu_long(cpus[j], "core_id") == ci) { dup = 1; break; }
        }
        if (!dup) ++cores;
    }
    return cores;
#else
    (void)cpus; (void)n;
    return 0;
#endif
}

/* Pins this process to `cpus[first .. first+count)`. Returns 1 when the pin
 * really happened, 0 when the platform has no such call -- never a silent
 * no-op, because "pinned" is the load-bearing word in this whole file. */
static int pin_to_slice(const int *cpus, int first, int count,
                        char *desc, size_t desc_size) {
    desc[0] = '\0';
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    size_t used = 0;
    for (int i = first; i < first + count; ++i) {
        CPU_SET(cpus[i], &set);
        if (used + 8u < desc_size) {
            used += (size_t)snprintf(desc + used, desc_size - used, "%s%d",
                                     used > 0 ? "," : "", cpus[i]);
        }
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        snprintf(desc, desc_size, "unpinned (sched_setaffinity: %s)", strerror(errno));
        return 0;
    }
    return 1;
#else
    (void)cpus; (void)first; (void)count;
    snprintf(desc, desc_size, "unpinned (no cpu affinity API on this platform)");
    return 0;
#endif
}

/* --------------------------------------------------------- fd passing
 *
 * SCM_RIGHTS over an AF_UNIX socketpair. Portable POSIX: this is compiled and
 * exercised on macOS as well as Linux, which is why the untested surface of a
 * prefork build is only the two affinity/topology calls above. */

static int send_fd(int chan, int fd) {
    char dummy = 'F';
    struct iovec iov;
    iov.iov_base = &dummy;
    iov.iov_len = 1;

    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } control;
    memset(&control, 0, sizeof(control));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.buf;
    msg.msg_controllen = sizeof(control.buf);

    struct cmsghdr *cm = CMSG_FIRSTHDR(&msg);
    cm->cmsg_level = SOL_SOCKET;
    cm->cmsg_type = SCM_RIGHTS;
    cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &fd, sizeof(int));

    ssize_t n;
    do { n = sendmsg(chan, &msg, 0); } while (n < 0 && errno == EINTR);
    return n == 1 ? 0 : -1;
}

/* The descriptor, -1 when the channel is closed (the parent is gone), -2 when
 * the caller should simply try again. A short or control-less message is a
 * protocol violation and is reported as a closed channel rather than guessed
 * at: there is no sane recovery from a half-read SCM_RIGHTS. */
static int recv_fd(int chan) {
    char dummy = 0;
    struct iovec iov;
    iov.iov_base = &dummy;
    iov.iov_len = 1;

    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } control;
    memset(&control, 0, sizeof(control));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.buf;
    msg.msg_controllen = sizeof(control.buf);

    const ssize_t n = recvmsg(chan, &msg, 0);
    if (n < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return -2;
        return -1;
    }
    if (n == 0) return -1;
    struct cmsghdr *cm = CMSG_FIRSTHDR(&msg);
    if (cm == NULL || cm->cmsg_level != SOL_SOCKET || cm->cmsg_type != SCM_RIGHTS ||
        cm->cmsg_len != CMSG_LEN(sizeof(int))) {
        return -1;
    }
    int fd = -1;
    memcpy(&fd, CMSG_DATA(cm), sizeof(int));
    return fd;
}

/* ------------------------------------------------------------ worker state */

static int g_worker_index = -1;
static int g_worker_threads = 0;
static int g_worker_chan = -1;
static volatile sig_atomic_t g_dump_request = 0;

/* Both are written in the PARENT, before the fork, and read in the children.
 * That ordering is the reason a worker can never disagree with the router
 * about which model it holds: the assignment is one value computed once and
 * then copied by fork(), not a message that could be lost or reordered. */
static int  g_worker_group = 0;
static char g_group_plan[512];
/* The accepted set, comma separated, for the message of a `model_not_found`.
 * Published before the fork so the router and every worker quote one list. */
static char g_model_set[512];

/* The cpu list this worker was ASKED to take. Written by the child right after
 * pin_to_slice, so `configured` and `actual` can be printed side by side. */
static char g_configured_mask[192];

int mynah_asr_prefork_worker_index(void) { return g_worker_index; }
int mynah_asr_prefork_worker_threads(void) { return g_worker_threads; }
int mynah_asr_prefork_actual_mask(char *out, size_t cap) {
    if (out == NULL || cap == 0) return 0;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        size_t used = 0;
        int n = 0;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &set)) continue;
            ++n;
            if (used + 8u < cap)
                used += (size_t)snprintf(out + used, cap - used, "%s%d",
                                         used > 0 ? "," : "", c);
        }
        if (used == 0) snprintf(out, cap, "empty");
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (online < 1) online = 1;
        return n > 0 && n < (int)online;   /* a strict subset: really pinned */
    }
    snprintf(out, cap, "unpinned (sched_getaffinity: %s)", strerror(errno));
    return 0;
#else
    /* macOS has no affinity API this server would use (thread affinity tags are
     * hints to the scheduler, not a mask), so there is nothing to read back and
     * "unpinned" is the honest answer rather than a blank. */
    snprintf(out, cap, "unpinned");
    return 0;
#endif
}

/* One field, one token: whitespace becomes '_' so the line stays parseable by
 * splitting on spaces, exactly as [FLAGS] does with a value that contains one.
 * The phrases here are real ("unpinned (no cpu affinity API on this
 * platform)"), so this is not hypothetical tidiness. */
static void topo_token(const char *in, char *out, size_t cap) {
    size_t k = 0;
    for (size_t i = 0; in != NULL && in[i] != '\0' && k + 1 < cap; i++)
        out[k++] = (in[i] == ' ' || in[i] == '\t') ? '_' : in[i];
    if (k == 0 && cap > 1) out[k++] = '-';
    out[k] = '\0';
}

/* The machine-readable twin of the human "prefork: worker N pid ... cpus ..."
 * line. One line per worker, printed by the worker itself, with the mask read
 * BACK from the kernel next to the one that was planned: those two being
 * different is the failure this line exists to make visible. */
void mynah_asr_prefork_print_topology(FILE *out, int threads) {
    char actual[256], atok[256], ctok[256];
    const int pinned = mynah_asr_prefork_actual_mask(actual, sizeof(actual));
    topo_token(actual, atok, sizeof(atok));
    topo_token(g_configured_mask[0] != '\0' ? g_configured_mask : "inherited",
               ctok, sizeof(ctok));
    fprintf(out, "[TOPOLOGY] v=1 worker=%d pid=%d configured_mask=%s "
                 "actual_mask=%s threads=%d pinned=%s\n",
            g_worker_index, (int)getpid(), ctok, atok, threads,
            pinned ? "yes" : "no");
    fflush(out);
}
int mynah_asr_prefork_worker_model(void) { return g_worker_group; }
const char *mynah_asr_prefork_model_plan(void) { return g_group_plan; }
const char *mynah_asr_prefork_model_set(void) { return g_model_set; }

/* --------------------------------------------------------- model matching
 *
 * One implementation, used by the router and by the worker, because two
 * spellings of "does this request fit this model" that disagree means a request
 * the router accepts and the worker refuses -- a 404 for a model the server
 * demonstrably holds, which is the most confusing failure this feature could
 * produce. See the contract on the declaration in prefork.h. */
static int name_ci_equal(const char *a, const char *b) {
    while (*a != '\0' && *b != '\0') {
        const int ca = tolower((unsigned char)*a++);
        const int cb = tolower((unsigned char)*b++);
        if (ca != cb) return 0;
    }
    return *a == '\0' && *b == '\0';
}

static int name_ci_prefix(const char *prefix, const char *name) {
    while (*prefix != '\0') {
        if (*name == '\0') return 0;
        if (tolower((unsigned char)*prefix++) != tolower((unsigned char)*name++)) return 0;
    }
    return 1;
}

int mynah_asr_prefork_model_match(const char *const *names, int count,
                                 const char *want) {
    if (names == NULL || count <= 0 || want == NULL || want[0] == '\0') return -1;
    for (int i = 0; i < count; ++i) {
        if (names[i] != NULL && name_ci_equal(names[i], want)) return i;
    }
    /* A single character is not a model name, it is a coin flip. Two is the
     * shortest prefix worth resolving. */
    if (strlen(want) < 2u) return -1;
    int found = -1;
    for (int i = 0; i < count; ++i) {
        if (names[i] == NULL || !name_ci_prefix(want, names[i])) continue;
        if (found >= 0) return -2;            /* ambiguous: refuse, never guess */
        found = i;
    }
    return found;
}

static void on_usr1(int sig) {
    (void)sig;
    g_dump_request = 1;
}

/* Async-signal-safe by construction: one store to a volatile sig_atomic_t.
 * Exported so server/main.c can install the handler BEFORE the fork -- which
 * is what the long note in the child path below assumes, and what keeps a
 * worker from being killed by the default SIGUSR1 action. */
void mynah_asr_prefork_request_dump(void) { g_dump_request = 1; }

int mynah_asr_prefork_take_dump_request(void) {
    if (g_dump_request == 0) return 0;
    g_dump_request = 0;
    return 1;
}

/* The worker's half of the load accounting. One byte, best effort: if the
 * parent has already gone the write fails and the worker is shutting down
 * anyway. Deliberately not retried on a full pipe either -- the channel is a
 * stream socket with a socket buffer far larger than the number of slots a
 * worker can possibly have in flight, so a full buffer would mean the parent
 * has stopped reading, which is the same condition as it being gone. */
void mynah_asr_prefork_conn_done(void) {
    if (g_worker_chan < 0) return;
    const char b = 1;
    ssize_t n;
    do { n = write(g_worker_chan, &b, 1); } while (n < 0 && errno == EINTR);
    (void)n;
}

int mynah_asr_prefork_recv_conn(int chan_fd, int timeout_ms) {
    struct pollfd pfd;
    pfd.fd = chan_fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    const int ready = poll(&pfd, 1, timeout_ms);
    if (ready < 0) return errno == EINTR ? -2 : -1;
    if (ready == 0) return -2;
    if ((pfd.revents & POLLIN) == 0) return -1;   /* HUP/ERR: parent is gone */
    return recv_fd(chan_fd);
}

/* ------------------------------------------------------------ the ladder
 *
 * Resolving the four rungs. Every knob is "0 = unset" so a caller that memsets
 * mynah_asr_prefork_config and never hears of admission control still gets the
 * measured defaults, and every knob has an environment override because the
 * ladder has no command-line flags yet -- those belong to server/main.c's
 * argument parser, which this lane does not own. */

static double mono_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* Inherited by every worker across the fork, which is why a worker never has
 * to be told the cap and can never disagree with the parent about it. */
static int g_service_cap_ms = 0;

static int env_int(const char *name, int *out) {
    const char *e = getenv(name);
    if (e == NULL || e[0] == '\0') return 0;
    char *end = NULL;
    const long v = strtol(e, &end, 10);
    if (end == e) return 0;
    *out = (int)v;
    return 1;
}

void mynah_asr_prefork_apply_env(mynah_asr_prefork_config *cfg) {
    const char *e = getenv("MYNAH_ASR_PREFORK_QUEUE");
    if (e != NULL && e[0] != '\0') {
        if (strcmp(e, "unbounded") == 0) {
            cfg->queue_per_worker = MYNAH_ASR_PREFORK_QUEUE_UNBOUNDED;
        } else {
            int v = 0;
            if (env_int("MYNAH_ASR_PREFORK_QUEUE", &v)) {
                cfg->queue_per_worker = v > 0 ? v
                                     : (v == 0 ? MYNAH_ASR_PREFORK_QUEUE_NONE
                                               : MYNAH_ASR_PREFORK_QUEUE_UNBOUNDED);
            }
        }
    }
    int v = 0;
    /* A deadline or a cap of zero means DISABLED, not "use the default": an
     * operator who types 0 is turning the rung off. Unset is what asks for the
     * default, and unset is the absence of the variable. */
    if (env_int("MYNAH_ASR_PREFORK_QUEUE_MS", &v)) cfg->queue_deadline_ms = v > 0 ? v : -1;
    if (env_int("MYNAH_ASR_PREFORK_SERVICE_MS", &v)) cfg->service_cap_ms = v > 0 ? v : -1;
    /* Decoder lane (not implemented yet). Not validated here: src/threads.c owns the "is this split wide
     * enough to be faster than inline" question and answers it with a printed
     * reason, so a number that cannot work is refused where the measurement
     * that condemns it is documented, rather than silently corrected here. */
    if (env_int("MYNAH_ASR_LANE_SPLIT", &v)) cfg->lane_cpus = v > 0 ? v : 0;
}

/* -1 = unbounded, 0 = no queue, >0 = that many per live worker. */
static int resolve_queue_per_worker(const mynah_asr_prefork_config *cfg) {
    if (cfg->queue_per_worker == MYNAH_ASR_PREFORK_QUEUE_NONE) return 0;
    if (cfg->queue_per_worker < 0) return -1;
    if (cfg->queue_per_worker == 0) return MYNAH_ASR_PREFORK_QUEUE_DEFAULT;
    return cfg->queue_per_worker;
}
static int resolve_queue_deadline_ms(const mynah_asr_prefork_config *cfg) {
    if (cfg->queue_deadline_ms < 0) return 0;              /* disabled */
    if (cfg->queue_deadline_ms == 0) return MYNAH_ASR_PREFORK_QUEUE_DEADLINE_MS;
    return cfg->queue_deadline_ms;
}
static int resolve_service_cap_ms(const mynah_asr_prefork_config *cfg) {
    if (cfg->service_cap_ms < 0) return 0;                 /* disabled */
    if (cfg->service_cap_ms == 0) return MYNAH_ASR_PREFORK_SERVICE_CAP_MS;
    return cfg->service_cap_ms;
}

/* --------------------------------------------------- the kernel backlog
 *
 * See prefork.h. It lives here because it is sized from the ladder's own
 * arithmetic: a second copy of that arithmetic in server/main.c is how a
 * banner comes to print a number the listener was never given. */

int mynah_asr_prefork_max_connections(const mynah_asr_prefork_config *cfg, int workers) {
    mynah_asr_prefork_config local = *cfg;
    mynah_asr_prefork_apply_env(&local);
    if (workers <= 0) workers = local.workers > 0 ? local.workers : 1;
    if (workers > PREFORK_MAX_WORKERS) workers = PREFORK_MAX_WORKERS;

    int slots = local.slots_per > 0 ? local.slots_per : 1;
    /* A group with its own `:cap=` may hold MORE than the fleet's --cap, and
     * the largest is what the listener has to survive: a backlog one short is
     * a dropped SYN nobody counts, one too long is a few kernel entries. */
    for (int i = 0; local.model_slots != NULL && i < local.model_count; i++)
        if (local.model_slots[i] > slots) slots = local.model_slots[i];

    const int q_per = resolve_queue_per_worker(&local);
    /* UNBOUNDED is not a size. Sizing the backlog from "no bound" would invent
     * one, so an unbounded queue is counted as the default depth and the
     * queue's own shout below covers what that leaves uncovered. */
    const int queued = q_per > 0 ? q_per
                     : (q_per < 0 ? MYNAH_ASR_PREFORK_QUEUE_DEFAULT : 0);
    return workers * (slots + queued);
}

int mynah_asr_prefork_listen_backlog(const mynah_asr_prefork_config *cfg, int workers) {
    int n = 0;
    if (env_int("MYNAH_ASR_LISTEN_BACKLOG", &n) && n > 0) {
        /* Still capped: a cap an override can exceed is not a cap. */
        return n > MYNAH_ASR_PREFORK_BACKLOG_MAX ? MYNAH_ASR_PREFORK_BACKLOG_MAX : n;
    }
    /* Twice the capacity, because the clients that will be REFUSED arrive in
     * the same burst as the ones that will be served, and a refusal is only
     * visible if the connection was accepted first. */
    n = 2 * mynah_asr_prefork_max_connections(cfg, workers);
    if (n < SOMAXCONN) n = SOMAXCONN;   /* never below the platform's own idea of full */
    if (n > MYNAH_ASR_PREFORK_BACKLOG_MAX) n = MYNAH_ASR_PREFORK_BACKLOG_MAX;
    return n;
}

int mynah_asr_prefork_somaxconn(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/sys/net/core/somaxconn", "r");
    if (f != NULL) {
        int v = 0;
        const int got = fscanf(f, "%d", &v);
        fclose(f);
        if (got == 1 && v > 0) return v;
    }
#elif defined(__APPLE__)
    int v = 0;
    size_t vn = sizeof(v);
    if (sysctlbyname("kern.ipc.somaxconn", &v, &vn, NULL, 0) == 0 && v > 0) return v;
#endif
    return SOMAXCONN;   /* the header's constant: a floor, not a reading */
}

const char *mynah_asr_prefork_somaxconn_knob(void) {
#if defined(__linux__)
    return "net.core.somaxconn";
#elif defined(__APPLE__)
    return "kern.ipc.somaxconn";
#else
    return "the listen-backlog sysctl";
#endif
}

/* Prints the ladder that is actually in force, and shouts about an unbounded
 * queue. The shout is deliberately impossible to skim past: an unbounded
 * admission queue is not a generous setting, it is the listener-backlog
 * mistake wearing different clothes. The request still waits, the wait is
 * still unbounded, and the only thing that changed is that the wait now
 * happens somewhere this process can see and chooses not to act on. */
static void describe_ladder(int workers, int slots, int q_per, int deadline_ms,
                            int service_ms, int backlog, int somaxconn, FILE *out) {
    /* The backlog first, because it is the rung in FRONT of rung 1 and the
     * only one the kernel owns: what it drops is dropped before accept(), so
     * no counter below can see it. */
    fprintf(out, "prefork: admission   backlog %d", backlog);
    if (somaxconn > 0 && somaxconn < backlog) fprintf(out, " (kernel caps it at %d)", somaxconn);
    fprintf(out, " · rung1 slots %d/worker (%d total)",
            slots, slots * workers);
    if (q_per < 0)      fprintf(out, " · rung2 queue UNBOUNDED");
    else if (q_per == 0) fprintf(out, " · rung2 queue disabled (refuse immediately)");
    else fprintf(out, " · rung2 queue %d/worker (%d total)", q_per, q_per * workers);
    if (deadline_ms > 0) fprintf(out, " · rung3 deadline %d ms", deadline_ms);
    else fprintf(out, " · rung3 deadline off");
    if (service_ms > 0) fprintf(out, " · rung4 service cap %d ms", service_ms);
    else fprintf(out, " · rung4 service cap off");
    fprintf(out, "\n");

    if (q_per < 0) {
        fprintf(out,
"prefork: ############################################################\n"
"prefork: ##  THE ADMISSION QUEUE BOUND IS DISABLED.                ##\n"
"prefork: ##                                                        ##\n"
"prefork: ##  An unbounded queue does not add capacity. It converts ##\n"
"prefork: ##  a refusal the client can see into a wait it cannot,   ##\n"
"prefork: ##  which is precisely the failure this ladder exists to  ##\n"
"prefork: ##  prevent -- the reference measured p95 TTFB 4470 ms    ##\n"
"prefork: ##  with over 97%% of the tail invisible before accept().##\n"
"prefork: ##                                                        ##\n"
"prefork: ##  Admission capacity and sustained capacity are not the ##\n"
"prefork: ##  same number. Raising the bound until the rejects stop ##\n"
"prefork: ##  admitted C32 with zero refusals on the reference and  ##\n"
"prefork: ##  nothing above C20 sustained. The refusals were the    ##\n"
"prefork: ##  server telling the truth.                             ##\n"
"prefork: ##                                                        ##\n"
"prefork: ##  Memory now grows with arrival rate, and every queued  ##\n"
"prefork: ##  descriptor is an fd this process must hold.           ##\n"
"prefork: ############################################################\n");
    }
    if (deadline_ms <= 0 && q_per != 0) {
        fprintf(out,
            "prefork: WARNING rung 3 is off: a queued request has no deadline and will "
            "wait for a slot indefinitely. The queue bound is then the only thing "
            "limiting how long a client waits, and a bound is not a deadline.\n");
    }
}

/* ------------------------------------------------------------------ plan */

/* Resolves W and T against the cpu mask. Never invents W: the operator gives
 * it, because the whole point of prefork.h's procedure is that no constant in
 * this file could know it. T defaults to the slice width, which is the only
 * choice that leaves the machine exactly subscribed. */
static int plan_topology(const mynah_asr_prefork_config *cfg, int *cpus, int max_cpus,
                         int *out_workers, int *out_threads, int *out_per,
                         int *out_core_major) {
    const int ncpu = cpus_allowed(cpus, max_cpus);
    *out_core_major = order_core_major(cpus, ncpu);

    int workers = cfg->workers;
    if (workers < 1) workers = 1;
    if (workers > PREFORK_MAX_WORKERS) workers = PREFORK_MAX_WORKERS;

    const int per = ncpu / workers > 0 ? ncpu / workers : 1;
    int threads = cfg->threads_per > 0 ? cfg->threads_per : per;

    *out_workers = workers;
    *out_threads = threads;
    *out_per = per;
    return ncpu;
}

void mynah_asr_prefork_reserve_threads(mynah_asr_prefork_config *cfg) {
    int cpus[CPU_LIST_MAX];
    int workers = 0, threads = 0, per = 0, core_major = 0;
    (void)plan_topology(cfg, cpus, CPU_LIST_MAX, &workers, &threads, &per, &core_major);
    cfg->workers = workers;
    cfg->threads_per = threads;

    const char *existing = getenv("MYNAH_ASR_THREADS");
    if (existing != NULL && existing[0] != '\0') {
        const int have = atoi(existing);
        if (have != threads) {
            fprintf(stderr,
                    "prefork: MYNAH_ASR_THREADS=%s is set in the environment and wins; "
                    "the plan wanted %d threads per worker\n", existing, threads);
        }
        cfg->threads_per = have > 0 ? have : threads;
        return;
    }
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", threads);
    setenv("MYNAH_ASR_THREADS", buf, 1);
}

void mynah_asr_prefork_print_plan(const mynah_asr_prefork_config *cfg, FILE *out) {
    int cpus[CPU_LIST_MAX];
    int workers = 0, threads = 0, per = 0, core_major = 0;
    mynah_asr_prefork_config local = *cfg;
    mynah_asr_prefork_apply_env(&local);
    if (local.workers < 1) local.workers = 1;
    const int ncpu = plan_topology(&local, cpus, CPU_LIST_MAX, &workers, &threads,
                                   &per, &core_major);
    const int cores = count_physical_cores(cpus, ncpu);
    /* Only claim an SMT factor when the mask actually covers whole cores.
     * A partial mask (a taskset that took one sibling of some cores and both
     * of others) has no single factor, and rounding it down printed "x 1 SMT"
     * on a host where SMT was plainly on. */
    const int smt = (cores > 0 && ncpu % cores == 0) ? ncpu / cores : 0;

    fprintf(out, "prefork plan\n");
    fprintf(out, "  allowed cpus     %d", ncpu);
    if (smt > 0) fprintf(out, "  (%d physical cores x %d SMT)", cores, smt);
    else if (cores > 0) fprintf(out, "  (%d physical cores; the mask covers them "
                                     "unevenly, so there is no one SMT factor)", cores);
    else fprintf(out, "  (physical topology unavailable on this platform)");
    fprintf(out, "\n");
    fprintf(out, "  cpu order        %s\n",
            core_major ? "core-major: a worker's slice keeps SMT siblings together"
                       : "as enumerated: no sysfs topology, slices are logical ids");
    fprintf(out, "  pinning          %s\n",
#if defined(__linux__)
            "sched_setaffinity"
#else
            "UNAVAILABLE on this platform: workers will float across all cpus"
#endif
    );
    fprintf(out, "  asked for        W=%d workers", workers);
    if (cfg->threads_per > 0) fprintf(out, ", T=%d threads (explicit)", threads);
    else fprintf(out, ", T=%d threads (derived: %d cpus / %d workers)", threads, ncpu, workers);
    fprintf(out, ", %d slots each\n", cfg->slots_per > 0 ? cfg->slots_per : 1);
    if (workers * threads > ncpu) {
        fprintf(out, "  WARNING          W*T = %d > %d allowed cpus: oversubscribed, "
                     "workers will preempt each other\n", workers * threads, ncpu);
    } else if (workers * threads < ncpu) {
        fprintf(out, "  note             W*T = %d < %d allowed cpus: %d cpus idle\n",
                workers * threads, ncpu, ncpu - workers * threads);
    }
    if (cores > 0 && smt > 1) {
        fprintf(out, "  WARNING          SMT is on (%d threads per core). Two threads of "
                     "one core share its execution units; a measurement that does not "
                     "say whether SMT was on is not a measurement.\n", smt);
    }

    /* The slices themselves, because "pinned" is only meaningful if you can
     * see what to. On an SMT host this is also the line that shows the sibling
     * ordering working: worker 0 should hold BOTH threads of its cores, not
     * one thread of twice as many. */
    warn_cpu_budget(workers, threads, ncpu, out);
    describe_ladder(workers, cfg->slots_per > 0 ? cfg->slots_per : 1,
                    resolve_queue_per_worker(&local),
                    resolve_queue_deadline_ms(&local),
                    resolve_service_cap_ms(&local),
                    mynah_asr_prefork_listen_backlog(&local, workers),
                    mynah_asr_prefork_somaxconn(), out);

    for (int i = 0; i < workers; ++i) {
        const int first = i * per;
        const int count = (i + 1) * per <= ncpu ? per : ncpu - first;
        if (count <= 0) {
            fprintf(out, "  worker %-9d (no cpus left: W=%d exceeds the %d allowed)\n",
                    i, workers, ncpu);
            continue;
        }
        fprintf(out, "  worker %-9d cpus ", i);
        for (int c = first; c < first + count; ++c) {
            fprintf(out, "%s%d", c > first ? "," : "", cpus[c]);
        }
        fprintf(out, "\n");
    }

    fprintf(out,
        "\n"
        "how to choose W -- the procedure, not a number\n"
        "\n"
        "  W and T multiply to this machine. %d cpus admits", ncpu);
    {
        int printed = 0;
        for (int w = 1; w <= ncpu && printed < 6; w *= 2) {
            if (ncpu % w != 0) continue;
            fprintf(out, "%s %dx%d", printed ? "," : "", w, ncpu / w);
            ++printed;
        }
    }
    fprintf(out,
        " and they are not the same server.\n"
        "\n"
        "  STEP 1  find T*, the width at which ONE worker stops improving.\n"
        "          No prefork. Sweep the pool width and read the per-stream step\n"
        "          cost (mynah-asr bench, or one stream through the server):\n"
        "            for t in 1 2 4 8 16 32; do\n"
        "              MYNAH_ASR_THREADS=$t OPENBLAS_NUM_THREADS=$t ./mynah-asr bench -m MODEL -i long.wav --quant int8\n"
        "            done\n"
        "          T* is the SMALLEST t within ~10%% of the best RTF. The stream\n"
        "          step is bandwidth bound, so T* is usually far below %d.\n"
        "\n"
        "  STEP 2  sweep W at constant subscription: W*T = %d throughout.\n"
        "          For each W in {%d/T*/2, %d/T*, 2*%d/T*} (rounded to a divisor):\n"
        "            ./mynah-asr-server -m MODEL --prefork $W --prefork-threads $((%d/W)) &\n"
        "            python3 tools/bench/stream_load.py --streams N --clips ...   # N = W, 2W, 4W\n"
        "          Record per level: completed/launched, TTFP p50 and p95, emission\n"
        "          lag p50 and p95, finalization lag, rejects, and the RSS of the\n"
        "          whole process tree. Medians alone will flatter every configuration.\n"
        "\n"
        "  STEP 3  for the winner, raise concurrency until the verdict stops\n"
        "          being GOOD. THAT concurrency is this machine's answer. Publish\n"
        "          it with W, T, the cpu mask, SMT on/off, the model revision and\n"
        "          the ISA beside it -- it transfers to no other machine.\n"
        "\n"
        "  STEP 4  measure RSS after the sweep, never before: per-stream state and\n"
        "          scratch are built lazily on a worker's first request and are\n"
        "          NOT shared by the fork. Only the mapped model is. If the\n"
        "          total is far above (model + W * per-worker cache), something you\n"
        "          believed was shared is not.\n"
        "\n"
        "  Send SIGUSR1 to the parent at any point for the per-worker table\n"
        "  (dispatched, completed, in flight, mean in flight since the last dump).\n",
        ncpu, ncpu, ncpu, ncpu, ncpu, ncpu);
}

/* ------------------------------------------------------- refusal reasons
 *
 * One row per rung. The `code` is the contract: it appears as `error.code` in
 * the JSON body and as the counter name in every stats line, so a client
 * matching on it and an operator reading a dump are looking at the same
 * string. Append-only -- never renumber, never rename.
 *
 * Retry-After is per reason and not a constant, because the reasons say
 * different things about retrying. At-capacity is transient: come back. A
 * service-cap breach is a property of the request itself, and telling a client
 * to retry a request that will fail identically is how a refusal becomes a
 * retry storm. */
static const struct {
    const char *code;
    const char *status;
    const char *type;          /* error.type in the body, OpenAI vocabulary */
    const char *message;
    int         retry_after;   /* 0 = omit the header entirely */
} REFUSAL[MYNAH_ASR_PREFORK_REFUSE__COUNT] = {
    { "server_at_capacity", "503 Service Unavailable", "server_error",
      "every worker is at its slot cap and the admission queue is full", 1 },
    { "queued_too_long", "503 Service Unavailable", "server_error",
      "waited in the admission queue longer than the deadline", 1 },
    { "service_cap_exceeded", "503 Service Unavailable", "server_error",
      "the request ran past its per-request service cap", 0 },
    { "handoff_failed", "503 Service Unavailable", "server_error",
      "the chosen worker could not be handed the connection", 1 },
    /* The one refusal in this table that is the CLIENT's to fix, which is why
     * it is the one 404 and the one invalid_request_error. A 503 would tell a
     * client to retry, and the retry would fail identically for as long as the
     * server runs: residency is decided at startup and printed, never grown on
     * demand. The message names the accepted set (see below), and GET
     * /v1/models lists it too. */
    { "model_not_found", "404 Not Found", "invalid_request_error",
      "no worker holds the requested model", 0 },
};

const char *mynah_asr_prefork_refusal_code(mynah_asr_prefork_refusal reason) {
    if (reason < 0 || reason >= MYNAH_ASR_PREFORK_REFUSE__COUNT) return "unknown";
    return REFUSAL[reason].code;
}

size_t mynah_asr_prefork_refusal_response(mynah_asr_prefork_refusal reason,
                                      char *buf, size_t cap) {
    if (reason < 0 || reason >= MYNAH_ASR_PREFORK_REFUSE__COUNT) {
        reason = MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY;
    }
    /* Assembled rather than written out as a literal so Content-Length cannot
     * drift away from the body it describes. A hand-counted length is exactly
     * the constant that survives an edit to the message and then silently
     * truncates every refusal. */
    /* `model_not_found` is the one reason whose message is not a constant: a
     * client that named a model this fleet does not hold can only act on the
     * list of the ones it does, and making it ask a second endpoint for that
     * list is how a refusal becomes two round trips. The set comes from
     * g_model_set, which the parent published before the fork, so the router
     * and every worker quote the same line. */
    char body[640];
    const char *set = (reason == MYNAH_ASR_PREFORK_REFUSE_MODEL_NOT_FOUND)
                    ? g_model_set : "";
    char tail[440];
    tail[0] = '\0';
    if (set[0] != '\0') snprintf(tail, sizeof(tail), "; this fleet serves: %s", set);
    const int blen = snprintf(body, sizeof(body),
        "{\"error\":{\"message\":\"%s%s\",\"type\":\"%s\",\"code\":\"%s\"}}",
        REFUSAL[reason].message, tail, REFUSAL[reason].type, REFUSAL[reason].code);
    if (blen <= 0 || (size_t)blen >= sizeof(body)) return 0;

    char retry[40];
    retry[0] = '\0';
    if (REFUSAL[reason].retry_after > 0) {
        snprintf(retry, sizeof(retry), "Retry-After: %d\r\n",
                 REFUSAL[reason].retry_after);
    }
    const int n = snprintf(buf, cap,
        "HTTP/1.1 %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "%s"
        "Connection: close\r\n"
        "\r\n%s",
        REFUSAL[reason].status, blen, retry, body);
    if (n <= 0 || (size_t)n >= cap) return 0;
    return (size_t)n;
}

/* ------------------------------------------------- refusing without an RST
 *
 * See the long note on mynah_asr_prefork_refuse_and_close() in prefork.h. The
 * short version: write, shutdown(SHUT_WR), DRAIN, close -- and a close that
 * skips the drain sends an RST that destroys the response the client was about
 * to read. Both halves of that sequence are bounded, by bytes and by time,
 * because an unbounded drain is a way for one client to occupy the router. */

#define PF_LINGER_MAX   256          /* concurrent lingering closes            */
#define PF_LINGER_MS    500.0        /* per-fd budget for the whole sequence   */
#define PF_DRAIN_MAX    (64u * 1024u)/* bytes we will read before giving up    */
#define PF_QPOLL_MAX    64           /* queued fds watched for a hangup        */

typedef struct {
    int    fd;
    double deadline;      /* mono seconds */
    /* Sized for the longest refusal body, which is `model_not_found` with the
     * accepted set spelled out. A response that did not fit would be dropped
     * whole (refusal_response returns 0), never truncated. */
    char   msg[1024];
    size_t msg_len;
    size_t msg_sent;
    size_t drained;
    int    shut;          /* SHUT_WR already done */
} pf_linger;

static void set_nonblock(int fd) {
    const int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

static void linger_begin(pf_linger *L, int fd, mynah_asr_prefork_refusal reason,
                         double now) {
    memset(L, 0, sizeof(*L));
    L->fd = fd;
    L->deadline = now + PF_LINGER_MS / 1000.0;
    L->msg_len = mynah_asr_prefork_refusal_response(reason, L->msg, sizeof(L->msg));
    set_nonblock(fd);
}

/* Which direction this fd is waiting on right now. */
static short linger_events(const pf_linger *L) {
    return (L->msg_sent < L->msg_len) ? (short)POLLOUT : (short)POLLIN;
}

/* Advances one lingering close as far as it can go without blocking.
 * Returns 1 when the descriptor has been closed and the slot is free. */
static int linger_step(pf_linger *L, double now) {
    /* 1. the response itself */
    while (L->msg_sent < L->msg_len) {
        const ssize_t n = send(L->fd, L->msg + L->msg_sent,
                               L->msg_len - L->msg_sent, 0);
        if (n > 0) { L->msg_sent += (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (now >= L->deadline) break;
            return 0;                      /* wait for POLLOUT */
        }
        break;                             /* peer is gone: nothing to deliver */
    }

    /* 2. the FIN. This is what tells the client the response is complete, and
     *    it must come BEFORE the drain: a client waiting to send more will not
     *    finish until it learns we are done talking. */
    if (L->msg_sent >= L->msg_len && !L->shut) {
        shutdown(L->fd, SHUT_WR);
        L->shut = 1;
    }

    /* 3. the drain. The whole point of the exercise: unread bytes in the
     *    receive queue at close() time make the kernel send an RST instead of
     *    a FIN, and the RST discards our response along with it. */
    if (L->shut) {
        char scratch[4096];
        for (;;) {
            const ssize_t n = recv(L->fd, scratch, sizeof(scratch), 0);
            if (n > 0) {
                L->drained += (size_t)n;
                if (L->drained >= PF_DRAIN_MAX) break;   /* enough; not our body */
                continue;
            }
            if (n == 0) break;                           /* clean: peer closed */
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (now >= L->deadline) break;
                return 0;                                /* wait for POLLIN */
            }
            break;
        }
    }

    close(L->fd);
    L->fd = -1;
    return 1;
}

/* Drives one lingering close to completion on the CALLING thread, bounded by
 * PF_LINGER_MS. No globals are touched, so it is callable from any thread --
 * which matters because server/main.c refuses on an HTTP worker. It is
 * bounded-blocking rather than non-blocking: a worker thread can afford half a
 * second on a connection it is refusing, and the router cannot, which is why
 * the router uses refuse_park() below instead. */
static void linger_run(pf_linger *L) {
    for (;;) {
        const double now = mono_seconds();
        if (linger_step(L, now)) return;
        const double left_ms = (L->deadline - now) * 1000.0;
        if (left_ms <= 0.0) break;
        struct pollfd pfd;
        pfd.fd = L->fd;
        pfd.events = linger_events(L);
        pfd.revents = 0;
        const int r = poll(&pfd, 1, (int)left_ms);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
    }
    if (L->fd >= 0) close(L->fd);
}

void mynah_asr_prefork_refuse_and_close(int fd, mynah_asr_prefork_refusal reason) {
    if (fd < 0) return;
    pf_linger L;
    linger_begin(&L, fd, reason, mono_seconds());
    linger_run(&L);
}

int mynah_asr_prefork_linger_close(int fd, const char *response, size_t len) {
    if (fd < 0) return -1;
    pf_linger L;
    memset(&L, 0, sizeof(L));
    L.fd = fd;
    L.deadline = mono_seconds() + PF_LINGER_MS / 1000.0;
    const int fits = response != NULL && len > 0 &&
                     len <= (size_t)MYNAH_ASR_PREFORK_LINGER_MAX &&
                     len < sizeof(L.msg);
    if (fits) {
        memcpy(L.msg, response, len);
        L.msg_len = len;
    }
    set_nonblock(fd);
    linger_run(&L);   /* with msg_len 0 this is shutdown -> drain -> close */
    return fits ? 0 : -1;
}

int mynah_asr_prefork_service_cap_ms(void) { return g_service_cap_ms; }

/* ------------------------------------------------- classifying a connection
 *
 * Only reached when the fleet holds more than one model. The router has to know
 * which model a connection is for before it can choose a worker, because the
 * worker is the thing that holds the weights -- so this is the one place where
 * the parent looks at bytes a client sent.
 *
 * WHAT IT IS ALLOWED TO DO, stated as a boundary rather than as a description,
 * because "the parent parses a little HTTP" is a door that only opens wider:
 *
 *   - it PEEKS. MSG_PEEK consumes nothing, so the worker still reads the
 *     entire request from the start and there is no split-buffer to hand over
 *     with the descriptor. The fd remains the only thing that moves.
 *   - it reads a BOUNDED prefix and never more.
 *   - it looks for exactly four things: the end of the request line, the value
 *     of a query parameter named `model` inside it, the end of the header
 *     block, and a multipart part whose Content-Disposition names `model`.
 *     That is framing plus one parameter. It does not interpret a method, a
 *     route, a status or a body's meaning, and it never writes anything but a
 *     refusal.
 *   - it NEVER BLOCKS. Undecided means "park this and look again", handled by
 *     the caller's poll set with a deadline, because a router that waits on a
 *     slow client is rung 1's mistake wearing a different hat.
 *
 * WHY THOSE TWO PLACES AND NOT A JSON KEY. Both routes this server serves put
 * the model where a bounded prefix can see it: the WebSocket names it in the
 * query of its GET (`/v1/audio/stream?model=...`), and the REST endpoints take
 * it as a multipart field -- or in their own query, which is why the query is
 * searched for every method and not only for the upgrade. `lang` is NOT looked
 * at at all, and that is the difference from the TTS sibling this file came
 * from: one Nemotron serves forty languages, so a language decides nothing
 * about which process should run the request.
 *
 * The residual imprecision, named rather than hidden: a multipart `model` field
 * that arrives AFTER the audio is past the peek bound and will not be seen, so
 * such a request goes to the default group. It is not then served by the wrong
 * weights -- the worker parses the same field out of the whole body and refuses
 * with `model_not_found` if it is not its own -- so the cost is a wrong
 * refusal, never a wrong answer, and never a batch that mixed two models. Put
 * `model` before `file` in the form, or in the query, and it is always seen. */

#define PF_PEEK_MAX     8192u
#define PF_CLASSIFY_MS  2000.0       /* per-connection budget for deciding    */
#define PF_PENDING_MAX  64           /* connections awaiting classification   */

/* Negative results. Group indices are >= 0. */
#define PF_CLASS_DEFAULT   (-1)      /* named nothing: the default group      */
#define PF_CLASS_WAIT      (-2)      /* too little has arrived: park and retry*/
#define PF_CLASS_UNKNOWN   (-3)      /* named a model nobody holds: refuse    */

/* Case-insensitive search for `needle` within [begin, end). */
static const char *ci_find(const char *begin, const char *end, const char *needle) {
    const size_t n = strlen(needle);
    if (n == 0 || (size_t)(end - begin) < n) return NULL;
    for (const char *p = begin; (size_t)(end - p) >= n; ++p) {
        size_t i = 0;
        while (i < n && tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i])) ++i;
        if (i == n) return p;
    }
    return NULL;
}

/* The announced body length, or -1 when no Content-Length is present. Only the
 * framing number is read; nothing else about the header is interpreted. */
static long header_content_length(const char *begin, const char *end) {
    const char *p = ci_find(begin, end, "\ncontent-length:");
    if (p == NULL) return -1;
    p += strlen("\ncontent-length:");
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
    long value = 0;
    int digits = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        if (value > (2000000000L - 9) / 10) return -1;   /* absurd: ignore it */
        value = value * 10 + (*p++ - '0');
        ++digits;
    }
    return digits > 0 ? value : -1;
}

/* Percent-decoding, for the query only: a client that escaped a character in a
 * model name must reach the same group as one that did not. Anything malformed
 * is copied through as-is rather than guessed at. */
static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* The value of the query parameter `key` within [q, end), written into `out`.
 * Returns 1 when the parameter was present (the value may be empty), 0
 * otherwise. Matches only a WHOLE key -- at the start of the query or right
 * after an `&` -- so `?not_model=x` is never mistaken for `?model=x`. */
static int query_param(const char *q, const char *end, const char *key,
                       char *out, size_t cap) {
    const size_t klen = strlen(key);
    if (out == NULL || cap == 0) return 0;
    out[0] = '\0';
    for (const char *p = q; p < end; ) {
        const char *amp = (const char *)memchr(p, '&', (size_t)(end - p));
        const char *seg_end = amp != NULL ? amp : end;
        if ((size_t)(seg_end - p) > klen && p[klen] == '=' &&
            strncmp(p, key, klen) == 0) {
            const char *v = p + klen + 1;
            size_t used = 0;
            while (v < seg_end && used + 1u < cap) {
                if (*v == '%' && seg_end - v >= 3 &&
                    hexval((unsigned char)v[1]) >= 0 && hexval((unsigned char)v[2]) >= 0) {
                    out[used++] = (char)(hexval((unsigned char)v[1]) * 16 +
                                         hexval((unsigned char)v[2]));
                    v += 3;
                } else {
                    out[used++] = *v++;
                }
            }
            out[used] = '\0';
            return 1;
        }
        if (amp == NULL) break;
        p = amp + 1;
    }
    return 0;
}

/* The value of the multipart field named `model`, if a complete part carrying
 * it is inside [body, end). Returns 1 when it was found, 0 when it was not, and
 * -1 when a part that names it has arrived only partly -- which is the caller's
 * cue to wait for more bytes rather than to default.
 *
 * The `name="model"` match is required to sit inside a Content-Disposition
 * header, within the 128 bytes that precede it, so the same twelve bytes
 * appearing inside an uploaded file cannot decide where a request goes. */
static int multipart_model(const char *body, const char *end,
                           char *out, size_t cap) {
    if (out == NULL || cap == 0) return 0;
    out[0] = '\0';
    const char *k = ci_find(body, end, "name=\"model\"");
    if (k == NULL) return 0;
    const char *win = (k - body) > 128 ? k - 128 : body;
    if (ci_find(win, k, "content-disposition:") == NULL) return 0;
    const char *hdr_end = NULL;
    for (const char *p = k; p + 4 <= end; ++p) {
        if (p[0] == '\r' && p[1] == '\n' && p[2] == '\r' && p[3] == '\n') {
            hdr_end = p + 4;
            break;
        }
    }
    if (hdr_end == NULL) return -1;            /* the part is still arriving */
    const char *stop = NULL;
    for (const char *p = hdr_end; p + 2 <= end; ++p) {
        if (p[0] == '\r' && p[1] == '\n') { stop = p; break; }
    }
    if (stop == NULL) return -1;               /* the value is still arriving */
    const char *v = hdr_end;
    size_t used = 0;
    while (v < stop && used + 1u < cap) out[used++] = *v++;
    out[used] = '\0';
    return 1;
}

static int classify_model(int fd, const char *const *names, int count,
                          char *named, size_t named_cap) {
    if (named != NULL && named_cap > 0) named[0] = '\0';

    char buf[PF_PEEK_MAX + 1u];
    ssize_t got;
    do {
        got = recv(fd, buf, PF_PEEK_MAX, MSG_PEEK | MSG_DONTWAIT);
    } while (got < 0 && errno == EINTR);
    if (got < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return PF_CLASS_WAIT;
        return PF_CLASS_DEFAULT;    /* a broken socket is the worker's to report */
    }
    if (got == 0) return PF_CLASS_DEFAULT;            /* peer closed already */
    buf[got] = '\0';
    const int full = (size_t)got >= PF_PEEK_MAX;      /* our prefix is all we get */

    /* 1. THE QUERY. The request line first, because it is the one place both a
     *    WebSocket upgrade and a REST call can carry the name, and because it
     *    arrives in the very first packet of every client there is. */
    const char *eol = strstr(buf, "\r\n");
    if (eol == NULL) return full ? PF_CLASS_DEFAULT : PF_CLASS_WAIT;
    {
        const char *sp = (const char *)memchr(buf, ' ', (size_t)(eol - buf));
        const char *target = sp != NULL ? sp + 1 : NULL;
        const char *target_end = target != NULL
            ? (const char *)memchr(target, ' ', (size_t)(eol - target)) : NULL;
        if (target != NULL && target_end == NULL) target_end = eol;
        const char *qs = target != NULL
            ? (const char *)memchr(target, '?', (size_t)(target_end - target)) : NULL;
        if (qs != NULL &&
            query_param(qs + 1, target_end, "model", named, named_cap) &&
            named[0] != '\0') {
            const int at = mynah_asr_prefork_model_match(names, count, named);
            return at >= 0 ? at : PF_CLASS_UNKNOWN;
        }
        if (named != NULL && named_cap > 0) named[0] = '\0';
    }

    /* 2. THE MULTIPART FIELD. The header block must be complete before a body
     *    can exist. */
    const char *head_end = strstr(buf, "\r\n\r\n");
    if (head_end == NULL) return full ? PF_CLASS_DEFAULT : PF_CLASS_WAIT;
    const char *body = head_end + 4;
    const char *end = buf + got;

    const int mp = multipart_model(body, end, named, named_cap);
    if (mp == 1 && named[0] != '\0') {
        const int at = mynah_asr_prefork_model_match(names, count, named);
        return at >= 0 ? at : PF_CLASS_UNKNOWN;
    }
    if (named != NULL && named_cap > 0) named[0] = '\0';
    if (mp < 0) return full ? PF_CLASS_DEFAULT : PF_CLASS_WAIT;

    /* 3. Nothing named it yet. Decide whether more could still arrive. */
    const long announced = header_content_length(buf, head_end);
    if (announced < 0) {
        if (ci_find(buf, head_end, "\ntransfer-encoding:") == NULL) {
            return PF_CLASS_DEFAULT;     /* no body announced: none is coming */
        }
        return full ? PF_CLASS_DEFAULT : PF_CLASS_WAIT;   /* chunked: bounded */
    }
    const long have = (long)(got - (body - buf));
    if (have >= announced) return PF_CLASS_DEFAULT;   /* the whole body, no field */
    return full ? PF_CLASS_DEFAULT : PF_CLASS_WAIT;
}

/* ----------------------------------------------------------------- routing */

typedef struct {
    int    fd;
    double enqueued;      /* mono seconds, stamped at accept() */
    int    grp;           /* model group; 0 in a single-model fleet */
} pf_queued;

/* A connection accepted but not yet classified. Distinct from a queued one:
 * this is waiting for BYTES, not for a slot, so it is charged to no group's
 * capacity and refused by no rung. */
typedef struct {
    int    fd;
    double deadline;
} pf_pending;

typedef struct {
    pid_t pid;
    int chan;            /* parent's end of the socketpair */
    int grp;             /* model group; 0 in a single-model fleet */
    int slots;           /* this group's --cap: the rung-1 bound for this worker */
    int active;          /* dispatched minus finished */
    /* Two sets of counters on purpose. The `window_` ones are reset by every
     * SIGUSR1 dump, because "what happened in the last minute" is the question
     * a dump is asked; the totals are never reset, because a FINAL line that
     * prints the last window instead of the run reads as "this server did
     * nothing" to anyone who took a dump before shutting it down. */
    long long assigned;
    long long completed;
    long long window_assigned;
    long long window_completed;
    double area;         /* integral of `active` dt, for the mean in flight */
    /* Rung 4's watchdog. Dispatch timestamps of the connections this worker
     * still owes us a completion for, oldest first. Completions do not arrive
     * in dispatch order, so popping the head on every completion makes this an
     * approximation -- but it approximates in the safe direction: the head is
     * always the oldest outstanding dispatch, so "the head is past the cap" is
     * never a false alarm, it can only be late. */
    double *disp;
    int disp_cap, disp_head, disp_n;
    long long over_cap;  /* outstanding connections seen past the service cap */
    /* S12-18. Connections this worker held when its channel closed: the worker
     * died with them, so they are neither completed nor in flight. Before this
     * they were zeroed out of `active` and appeared nowhere. The router's books:
     *   assigned == completed + lost + active, per worker, always. */
    long long lost;
    int died;            /* the channel closed: this worker is gone for good */
} worker_state;

static void disp_push(worker_state *w, double t) {
    if (w->disp == NULL || w->disp_n >= w->disp_cap) return;
    w->disp[(w->disp_head + w->disp_n) % w->disp_cap] = t;
    ++w->disp_n;
}
static void disp_pop(worker_state *w, int n) {
    while (n-- > 0 && w->disp_n > 0) {
        w->disp_head = (w->disp_head + 1) % w->disp_cap;
        --w->disp_n;
    }
}

/* ------------------------------------------------------- the router's state
 *
 * Gathered into one struct so the admission path can be a function instead of
 * a macro. It has to be callable from two places now -- a fresh accept, and a
 * connection that has just finished being classified -- and two copies of the
 * rung arithmetic is exactly how the two paths would come to disagree about
 * capacity. */
typedef struct {
    worker_state *w;
    int           workers;
    int           slots;
    int           q_per;          /* -1 unbounded, 0 none, >0 per live worker */
    int           deadline_ms;
    int           groups;         /* model groups; 1 when single-model        */
    pf_queued   **q;              /* by pointer: rung 2 may grow the array    */
    int          *q_alloc;
    int          *q_n;
    long long    *queued_total;
    int          *queue_peak;
    long long    *dispatched;
    long long    *window_dispatched;
    long long    *refused;        /* MYNAH_ASR_PREFORK_REFUSE__COUNT entries      */
    long long    *window_refused;
    pf_linger    *linger;
    int          *linger_n;
    long long    *linger_forced;
} pf_router;

/* Refuses without blocking: the router is single threaded, so it does as much
 * of the write-shutdown-drain-close sequence as it can right now and parks the
 * remainder in the poll set. Counting happens here too, so a refusal cannot be
 * delivered without also being counted. */
static void router_refuse(pf_router *R, int fd, mynah_asr_prefork_refusal reason,
                          double now) {
    ++R->refused[reason];
    ++R->window_refused[reason];
    pf_linger L;
    linger_begin(&L, fd, reason, now);
    if (linger_step(&L, now)) return;
    if (*R->linger_n < PF_LINGER_MAX) {
        R->linger[(*R->linger_n)++] = L;
        return;
    }
    /* The set is full. Finish what we can without blocking and close; this is
     * the only path that can still produce an RST, so it is counted rather
     * than hidden. */
    ++(*R->linger_forced);
    L.deadline = 0.0;
    (void)linger_step(&L, now + 1.0);
}

/* Live workers holding `grp`. Recomputed rather than cached because a fleet
 * that has lost a worker has lost the slots behind that group's queue entries
 * too, and a bound that keeps promising capacity the machine no longer has is
 * how a degraded server turns a refusal into a wait. */
static int group_live(const pf_router *R, int grp) {
    int n = 0;
    for (int i = 0; i < R->workers; ++i) {
        if (R->w[i].pid > 0 && R->w[i].grp == grp) ++n;
    }
    return n;
}

/* Rung 1 within a group: the least-loaded worker of that group with a free
 * slot, or -1. A free slot in another group's worker is not capacity for this
 * request -- that worker does not have these weights. The cap compared against
 * is the WORKER's own, because --cap is per group. */
static int group_pick(const pf_router *R, int grp) {
    int best = -1;
    for (int i = 0; i < R->workers; ++i) {
        if (R->w[i].pid <= 0 || R->w[i].grp != grp) continue;
        if (R->w[i].active >= R->w[i].slots) continue;
        if (best < 0 || R->w[i].active < R->w[best].active) best = i;
    }
    return best;
}

static int group_queued(const pf_router *R, int grp) {
    int n = 0;
    for (int i = 0; i < *R->q_n; ++i) if ((*R->q)[i].grp == grp) ++n;
    return n;
}

static void queue_erase(pf_router *R, int at) {
    pf_queued *q = *R->q;
    memmove(&q[at], &q[at + 1], (size_t)(*R->q_n - at - 1) * sizeof(q[0]));
    --(*R->q_n);
}

/* Hands `fd` to `worker`, or refuses when the channel is broken. Takes
 * ownership of the descriptor either way. */
static void router_dispatch(pf_router *R, int worker, int fd, double now) {
    if (send_fd(R->w[worker].chan, fd) != 0) {
        fprintf(stderr, "prefork: handing fd to worker %d failed: %s\n",
                worker, strerror(errno));
        router_refuse(R, fd, MYNAH_ASR_PREFORK_REFUSE_HANDOFF_FAILED, now);
        return;
    }
    /* Our copy goes now: from here the worker is the only owner, and the
     * client sees a close only when the worker closes. */
    close(fd);
    ++R->w[worker].active;
    ++R->w[worker].assigned;
    ++R->w[worker].window_assigned;
    disp_push(&R->w[worker], now);
    ++(*R->dispatched);
    ++(*R->window_dispatched);
}

/* Rungs 1 and 2 for one freshly-decided connection. Takes ownership of `fd`. */
static void router_admit(pf_router *R, int fd, int grp, double now) {
    /* A new arrival may only go straight through when its GROUP's queue is
     * empty -- otherwise it would jump ahead of entries that have already
     * waited, and rung 3's deadline would start firing on requests that were
     * merely unlucky rather than late. Another group's queue is irrelevant:
     * those entries are not waiting for this worker. */
    if (group_queued(R, grp) == 0) {
        const int best = group_pick(R, grp);
        if (best >= 0) {
            router_dispatch(R, best, fd, now);
            return;
        }
    }

    /* RUNG 2, bounded per group because capacity is per group. */
    const int bound = (R->q_per < 0) ? -1 : R->q_per * group_live(R, grp);
    if (bound != 0 && (bound < 0 || group_queued(R, grp) < bound)) {
        if (*R->q_n >= *R->q_alloc) {
            const int grown = *R->q_alloc * 2;
            pf_queued *bigger = (pf_queued *)realloc(*R->q, (size_t)grown * sizeof(**R->q));
            if (bigger == NULL) {
                router_refuse(R, fd, MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY, now);
                return;
            }
            *R->q = bigger;
            *R->q_alloc = grown;
        }
        pf_queued *q = *R->q;
        q[*R->q_n].fd = fd;
        q[*R->q_n].enqueued = now;
        q[*R->q_n].grp = grp;
        ++(*R->q_n);
        ++(*R->queued_total);
        if (*R->q_n > *R->queue_peak) *R->queue_peak = *R->q_n;
        return;
    }

    router_refuse(R, fd, MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY, now);
}

/* RUNG 3, then RUNG 1: drain the queue, one group at a time.
 *
 * The deadline is checked at each GROUP's head, at pop time, never at push.
 * Checking at push can only refuse on a prediction about a wait that has not
 * happened; checking at pop refuses on a fact.
 *
 * Per group rather than over one FIFO, and that is not a detail: a single FIFO
 * would let an entry for a saturated group block a ready entry for an idle
 * one behind it, which is a starvation channel introduced by the partitioning
 * that exists to prevent starvation. Array order is arrival order, so scanning
 * it for one group's entries gives that group its own FIFO for free. */
static void router_drain(pf_router *R, double now) {
    for (int grp = 0; grp < R->groups; ++grp) {
        for (;;) {
            int head = -1;
            for (int i = 0; i < *R->q_n; ++i) {
                if ((*R->q)[i].grp == grp) { head = i; break; }
            }
            if (head < 0) break;

            if (R->deadline_ms > 0 &&
                (now - (*R->q)[head].enqueued) * 1000.0 >= (double)R->deadline_ms) {
                const int fd = (*R->q)[head].fd;
                queue_erase(R, head);
                router_refuse(R, fd, MYNAH_ASR_PREFORK_REFUSE_QUEUED_TOO_LONG, now);
                continue;
            }
            const int best = group_pick(R, grp);
            if (best < 0) break;              /* no slot: it stays queued */
            const int fd = (*R->q)[head].fd;
            queue_erase(R, head);
            router_dispatch(R, best, fd, now);
        }
    }
}

/* ------------------------------------------------- /metrics, the ROUTER's view
 *
 * What the parent can honestly say, and nothing else. It routes; it never runs
 * inference, so it holds no scheduler counter: no sessions, no steps, no
 * deltas, no emission lag. What it does hold is the routing table -- per worker
 * in-flight, assigned and completed -- and its own refusals by reason, and
 * those are exported with a `worker` label so a fleet is never summed into a
 * single number that hides one wedged process.
 *
 * The page says all of this in its own comment lines. A scrape that has to be
 * read next to a design document is a scrape that will be misread. */
typedef struct {
    const worker_state *w;
    int workers;
    int slots;
    /* The group names, for the `model` label. Bounded by the CLI -- one value
     * per configured group -- so the cardinality rule holds: every label this
     * file emits is drawn from a set fixed before the first request. NULL, or
     * a count below 2, means no label is emitted at all. */
    const char *const *models;
    int groups;
    long long dispatched, queued_total, client_gone;
    int queued_now, queue_peak;
    const long long *refused;
    const char *build, *blas, *simd, *int8_kernel;
    double uptime;
} pf_metrics_view;

/* `worker="3",model="parakeet"` -- or just `worker="3"` in a single-model
 * fleet, so a scrape of the server everybody already had is unchanged. One
 * place builds it because five series must agree about a worker's group. */
static void pf_worker_labels(const pf_metrics_view *v, int i, char *out, size_t cap) {
    if (v->groups > 1 && v->models != NULL) {
        char m[64];
        const int g = v->w[i].grp;
        mynah_asr_metrics_label(g >= 0 && g < v->groups && v->models[g] != NULL
                                    ? v->models[g] : "unknown", m, sizeof(m));
        snprintf(out, cap, "worker=\"%d\",model=\"%s\"", i, m);
    } else {
        snprintf(out, cap, "worker=\"%d\"", i);
    }
}

static void pf_render_metrics(mynah_asr_metrics_buf *b, void *ud) {
    const pf_metrics_view *v = (const pf_metrics_view *)ud;
    char build[128], blas[64], simd[64], k8[64];
    mynah_asr_metrics_label(v->build, build, sizeof(build));
    mynah_asr_metrics_label(v->blas, blas, sizeof(blas));
    mynah_asr_metrics_label(v->simd, simd, sizeof(simd));
    mynah_asr_metrics_label(v->int8_kernel, k8, sizeof(k8));

    mynah_asr_metrics_addf(b,
        "# This is the PREFORK ROUTER's view. This process routes connections and\n"
        "# never enters the model, so the scheduler counters -- sessions, steps,\n"
        "# deltas, audio seconds, emission lag -- do not exist here and are NOT\n"
        "# exported. Give a worker its own --metrics-port to scrape those.\n"
        "# Per-worker series are never summed: a fleet total hides a wedged worker.\n");
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_build_info build, BLAS provider, compiled SIMD profile and\n"
        "# the int8 kernel the dispatcher's own predicate resolved.\n"
        "# TYPE mynah_asr_build_info gauge\n"
        "mynah_asr_build_info{build=\"%s\",blas=\"%s\",simd=\"%s\",int8_kernel=\"%s\"} 1\n",
        build, blas, simd, k8);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_uptime_seconds seconds since this process started routing.\n"
        "# TYPE mynah_asr_uptime_seconds gauge\n"
        "mynah_asr_uptime_seconds %.3f\n", v->uptime);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_worker_up 1 while the router still holds this worker.\n"
        "# TYPE mynah_asr_worker_up gauge\n");
    for (int i = 0; i < v->workers; ++i) {
        char lbl[96];
        pf_worker_labels(v, i, lbl, sizeof(lbl));
        mynah_asr_metrics_addf(b, "mynah_asr_worker_up{%s} %d\n",
                               lbl, v->w[i].pid > 0 ? 1 : 0);
    }
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_worker_inflight connections dispatched to this worker and\n"
        "# not yet reported finished. Router-view: it counts CONNECTIONS, not the\n"
        "# worker's stream slots.\n"
        "# TYPE mynah_asr_worker_inflight gauge\n");
    for (int i = 0; i < v->workers; ++i) {
        char lbl[96];
        pf_worker_labels(v, i, lbl, sizeof(lbl));
        mynah_asr_metrics_addf(b, "mynah_asr_worker_inflight{%s} %d\n",
                               lbl, v->w[i].active);
    }
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_worker_slots the per-worker capacity the router admits\n"
        "# against (--cap), which is per model group.\n"
        "# TYPE mynah_asr_worker_slots gauge\n");
    for (int i = 0; i < v->workers; ++i) {
        char lbl[96];
        pf_worker_labels(v, i, lbl, sizeof(lbl));
        mynah_asr_metrics_addf(b, "mynah_asr_worker_slots{%s} %d\n",
                               lbl, v->w[i].slots);
    }
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_worker_assigned_total connections handed to this worker.\n"
        "# TYPE mynah_asr_worker_assigned_total counter\n");
    for (int i = 0; i < v->workers; ++i) {
        char lbl[96];
        pf_worker_labels(v, i, lbl, sizeof(lbl));
        mynah_asr_metrics_addf(b, "mynah_asr_worker_assigned_total{%s} %lld\n",
                               lbl, v->w[i].assigned);
    }
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_worker_completed_total connections this worker reported\n"
        "# finished.\n"
        "# TYPE mynah_asr_worker_completed_total counter\n");
    for (int i = 0; i < v->workers; ++i) {
        char lbl[96];
        pf_worker_labels(v, i, lbl, sizeof(lbl));
        mynah_asr_metrics_addf(b, "mynah_asr_worker_completed_total{%s} %lld\n",
                               lbl, v->w[i].completed);
    }
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_worker_lost_total connections this worker held when it died:\n"
        "# neither completed nor in flight. assigned = completed + lost + inflight.\n"
        "# TYPE mynah_asr_worker_lost_total counter\n");
    for (int i = 0; i < v->workers; ++i) {
        char lbl[96];
        pf_worker_labels(v, i, lbl, sizeof(lbl));
        mynah_asr_metrics_addf(b, "mynah_asr_worker_lost_total{%s} %lld\n",
                               lbl, v->w[i].lost);
    }
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_worker_over_service_cap_total outstanding connections this\n"
        "# worker was seen holding past --service-cap.\n"
        "# TYPE mynah_asr_worker_over_service_cap_total counter\n");
    for (int i = 0; i < v->workers; ++i) {
        char lbl[96];
        pf_worker_labels(v, i, lbl, sizeof(lbl));
        mynah_asr_metrics_addf(b,
            "mynah_asr_worker_over_service_cap_total{%s} %lld\n",
            lbl, v->w[i].over_cap);
    }

    {
        /* The service-wide series: every worker's shared snapshot summed. */
        int dead[256] = {0};
        mynah_asr_fleet_router fr;
        memset(&fr, 0, sizeof(fr));
        fr.prefork = 1;
        fr.workers = v->workers;
        for (int i = 0; i < v->workers; ++i) {
            if (i < 256) dead[i] = v->w[i].died;
            if (v->w[i].died) fr.worker_deaths++;
            else if (v->w[i].pid > 0) fr.workers_up++;
            fr.router_lost += v->w[i].lost;
        }
        mynah_asr_fleet_stats fs;
        mynah_asr_fleet_sum(&fs, dead, &fr);
        mynah_asr_fleet_render(b, &fs, &fr);
    }
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_connections_dispatched_total connections routed to a worker.\n"
        "# TYPE mynah_asr_connections_dispatched_total counter\n"
        "mynah_asr_connections_dispatched_total %lld\n", v->dispatched);
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_refused_total refusals by the code the client was given.\n"
        "# Router-owned: a refusal happens before a worker is chosen, so it carries\n"
        "# no worker label (handoff_failed excepted, and it is not broken out).\n"
        "# TYPE mynah_asr_refused_total counter\n");
    for (int r = 0; r < MYNAH_ASR_PREFORK_REFUSE__COUNT; ++r) {
        char code[64];
        mynah_asr_metrics_label(mynah_asr_prefork_refusal_code(
                                    (mynah_asr_prefork_refusal)r), code, sizeof(code));
        mynah_asr_metrics_addf(b, "mynah_asr_refused_total{code=\"%s\"} %lld\n",
                               code, v->refused[r]);
    }
    mynah_asr_metrics_addf(b,
        "# HELP mynah_asr_admission_queue_depth connections waiting for a slot now.\n"
        "# TYPE mynah_asr_admission_queue_depth gauge\n"
        "mynah_asr_admission_queue_depth %d\n"
        "# HELP mynah_asr_admission_queue_peak the deepest the queue has been.\n"
        "# TYPE mynah_asr_admission_queue_peak gauge\n"
        "mynah_asr_admission_queue_peak %d\n"
        "# HELP mynah_asr_admission_queued_total connections that had to wait.\n"
        "# TYPE mynah_asr_admission_queued_total counter\n"
        "mynah_asr_admission_queued_total %lld\n"
        "# HELP mynah_asr_client_gone_total queued clients that hung up before a slot.\n"
        "# TYPE mynah_asr_client_gone_total counter\n"
        "mynah_asr_client_gone_total %lld\n",
        v->queued_now, v->queue_peak, v->queued_total, v->client_gone);
}

static void dump_table(const worker_state *w, int workers, long long dispatched,
                       const long long *refused, int queued, double window) {
    fprintf(stderr, "[prefork] dispatched=%lld queued=%d", dispatched, queued);
    for (int r = 0; r < MYNAH_ASR_PREFORK_REFUSE__COUNT; ++r) {
        fprintf(stderr, " %s=%lld", REFUSAL[r].code, refused[r]);
    }
    for (int i = 0; i < workers; ++i) {
        if (w[i].pid <= 0) { fprintf(stderr, " w%d[dead]", i); continue; }
        fprintf(stderr, " w%d[pid=%d asg=%lld done=%lld inflight=%d mean=%.2f]",
                i, (int)w[i].pid, w[i].window_assigned, w[i].window_completed,
                w[i].active, window > 0.0 ? w[i].area / window : 0.0);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
}

mynah_asr_prefork_role mynah_asr_prefork_run(const mynah_asr_prefork_config *cfg,
                                     volatile sig_atomic_t *stop, int *chan_fd) {
    mynah_asr_prefork_config local = *cfg;
    mynah_asr_prefork_apply_env(&local);

    int cpus[CPU_LIST_MAX];
    int workers = 0, threads = 0, per = 0, core_major = 0;
    const int ncpu = plan_topology(&local, cpus, CPU_LIST_MAX, &workers, &threads,
                                   &per, &core_major);
    const int slots = local.slots_per > 0 ? local.slots_per : 1;
    const int q_per = resolve_queue_per_worker(&local);      /* -1 = unbounded */
    const int deadline_ms = resolve_queue_deadline_ms(&local);
    g_service_cap_ms = resolve_service_cap_ms(&local);

    /* ---- MODEL GROUPS ----
     *
     * One model per worker, workers divided into contiguous groups. Contiguous
     * rather than round-robin so that a group's workers get adjacent core
     * slices: the slices are already ordered core-major, and a group whose
     * workers are neighbours shares a memory path rather than straddling the
     * machine.
     *
     * `groups == 1` is the single-model fleet, and from here on every
     * model-aware branch collapses: one group, every worker in it, the
     * classifier never called. That is the shape that shipped before this
     * change and it must keep behaving identically. */
    const int groups = (local.models != NULL && local.model_count > 1)
                     ? local.model_count : 1;
    if (groups > workers) {
        fprintf(stderr, "prefork: %d models need at least %d workers, but W is %d. "
                        "A model with no worker could not be served and would have "
                        "to be dropped silently; refusing to start instead.\n",
                groups, groups, workers);
        return MYNAH_ASR_PREFORK_ERROR;
    }
    const int default_grp = (local.default_model > 0 && local.default_model < groups)
                          ? local.default_model : 0;

    /* How many workers each group gets. `model_workers` is the CLI's
     * `:workers=` and is honoured when it adds up; when it does not, the split
     * falls back to even and SAYS SO, because a silently corrected topology is
     * one an operator will quote from the flag rather than from the banner. */
    int grp_of[PREFORK_MAX_WORKERS];
    int grp_n[PREFORK_MAX_WORKERS];
    {
        int explicit_ok = 0;
        if (local.model_workers != NULL && groups > 1) {
            int sum = 0, bad = 0;
            for (int g = 0; g < groups; ++g) {
                if (local.model_workers[g] < 1) bad = 1;
                sum += local.model_workers[g];
            }
            if (!bad && sum == workers) explicit_ok = 1;
            else {
                fprintf(stderr, "prefork: the per-model worker counts sum to %d but W "
                                "is %d; splitting the %d workers evenly instead\n",
                        sum, workers, workers);
            }
        }
        const int base = workers / groups;
        const int extra = workers % groups;
        int at = 0;
        for (int g = 0; g < groups; ++g) {
            grp_n[g] = explicit_ok ? local.model_workers[g] : base + (g < extra ? 1 : 0);
            for (int k = 0; k < grp_n[g] && at < workers; ++k) grp_of[at++] = g;
        }
        while (at < workers) grp_of[at++] = groups - 1;   /* cannot happen; be total */
    }

    /* The per-worker plan: its group's slot cap, its cpu slice and its thread
     * count. Uniform unless `model_cpus` asked for something else, so a fleet
     * that does not use the per-group knobs gets byte-for-byte the topology it
     * got before they existed. */
    int slots_of[PREFORK_MAX_WORKERS];
    int cpu_first[PREFORK_MAX_WORKERS], cpu_n[PREFORK_MAX_WORKERS];
    int thr_of[PREFORK_MAX_WORKERS];
    {
        const int uneven = (local.model_cpus != NULL && groups > 1);
        int at = 0;   /* next free cpu index, walked in group order */
        for (int g = 0; g < groups; ++g) {
            /* A group with no explicit cpu budget gets its proportional share
             * of the mask, which is exactly the even split when nobody asked
             * for anything. */
            int cg = uneven && local.model_cpus[g] > 0
                   ? local.model_cpus[g]
                   : (ncpu * grp_n[g]) / (workers > 0 ? workers : 1);
            if (cg < grp_n[g]) cg = grp_n[g];        /* at least one cpu each */
            int per_g = cg / grp_n[g];
            if (per_g < 1) per_g = 1;
            for (int i = 0; i < workers; ++i) {
                if (grp_of[i] != g) continue;
                slots_of[i] = (local.model_slots != NULL && local.model_slots[g] > 0)
                            ? local.model_slots[g] : slots;
                cpu_first[i] = at < ncpu ? at : (ncpu > 0 ? ncpu - 1 : 0);
                cpu_n[i] = at + per_g <= ncpu ? per_g : (ncpu - at > 0 ? ncpu - at : 1);
                thr_of[i] = local.threads_per > 0 ? local.threads_per : cpu_n[i];
                at += per_g;
            }
        }
        if (!uneven) {
            /* The historical plan, restated exactly: contiguous slices of
             * `per` cpus in worker order. Recomputing it here rather than
             * trusting the loop above keeps the unchanged case unchanged. */
            for (int i = 0; i < workers; ++i) {
                cpu_first[i] = i * per;
                cpu_n[i] = (i + 1) * per <= ncpu ? per : (ncpu - i * per > 0 ? ncpu - i * per : 1);
                thr_of[i] = local.threads_per > 0 ? local.threads_per : threads;
            }
        }
    }

    g_group_plan[0] = '\0';
    g_model_set[0] = '\0';
    if (groups > 1) {
        size_t used = 0, sused = 0;
        for (int g = 0; g < groups; ++g) {
            const char *nm = local.models[g] != NULL ? local.models[g] : "?";
            const int m = snprintf(g_group_plan + used, sizeof(g_group_plan) - used,
                                   "%s%s=%d", used > 0 ? " " : "", nm, grp_n[g]);
            if (m <= 0 || (size_t)m >= sizeof(g_group_plan) - used) break;
            used += (size_t)m;
            const int s = snprintf(g_model_set + sused, sizeof(g_model_set) - sused,
                                   "%s%s", sused > 0 ? ", " : "", nm);
            if (s <= 0 || (size_t)s >= sizeof(g_model_set) - sused) break;
            sused += (size_t)s;
        }
    }

    if (!local.quiet) {
        const int cores = count_physical_cores(cpus, ncpu);
        fprintf(stderr, "prefork: %d workers x %d threads over %d allowed cpus "
                        "(%d per worker, %s), %d slots each\n",
                workers, threads, ncpu, per,
                core_major ? "core-major slices" : "logical slices", slots);
        if (cores > 0 && ncpu % cores == 0 && ncpu / cores > 1) {
            const int smt = ncpu / cores;
            fprintf(stderr, "prefork: SMT is on (%d per core); each slice covers "
                            "%d physical cores\n", smt,
                    per / smt > 0 ? per / smt : 1);
        }
        if (workers * threads > ncpu) {
            fprintf(stderr, "prefork: WARNING oversubscribed, W*T = %d > %d cpus\n",
                    workers * threads, ncpu);
        }
        warn_cpu_budget(workers, threads, ncpu, stderr);
        if (groups > 1) {
            /* Printed, never assumed: how the machine was divided is the first
             * thing anyone asks when one model is slow and another is not,
             * and it is a number this process chose rather than one the
             * operator typed. */
            fprintf(stderr, "prefork: models      %d resident, one model per worker "
                            "(a batch is one process's slots, so a batch is one "
                            "model; `lang` stays a per-request parameter)\n", groups);
            for (int g = 0; g < groups; ++g) {
                int first = -1, last = -1, cg = 0, gs = slots;
                for (int i = 0; i < workers; ++i) {
                    if (grp_of[i] != g) continue;
                    if (first < 0) first = i;
                    last = i;
                    cg += cpu_n[i];
                    gs = slots_of[i];
                }
                fprintf(stderr, "prefork:   %-24s workers %d-%d (%d) · %d cpus · "
                                "%d slots each (%d) · rung2 queue %s\n",
                        local.models[g] != NULL ? local.models[g] : "?",
                        first, last, grp_n[g], cg, gs, grp_n[g] * gs,
                        q_per < 0 ? "unbounded"
                                  : (q_per == 0 ? "disabled" : "per worker"));
            }
            fprintf(stderr, "prefork:   default    %s (a request naming no model)\n",
                    local.models[default_grp] != NULL ? local.models[default_grp] : "?");
        }
        describe_ladder(workers, slots, q_per, deadline_ms, g_service_cap_ms,
                        mynah_asr_prefork_listen_backlog(&local, workers),
                        mynah_asr_prefork_somaxconn(), stderr);
#if !defined(__linux__)
        fprintf(stderr, "prefork: WARNING this platform has no cpu affinity API. "
                        "Workers are NOT pinned: they float across every cpu and two "
                        "of them will share cores. The handoff and the routing are "
                        "real; the isolation the throughput claim rests on is not.\n");
#endif
        fflush(stderr);
    }

    /* Before anything is forked, and before the listening socket is committed
     * to a topology we cannot undo. */
    if (check_fork_preconditions(&local) != 0) return MYNAH_ASR_PREFORK_ERROR;

    /* /metrics, bound HERE and not in the child: a port that is already taken
     * must fail before there are W processes to clean up, and the descriptor
     * must exist before the fork so every child can close its copy. */
    int metrics_fd = -1;
    if (local.metrics_port > 0) {
        metrics_fd = mynah_asr_metrics_listen(local.metrics_bind, local.metrics_port);
        if (metrics_fd < 0) return MYNAH_ASR_PREFORK_ERROR;
        fprintf(stderr, "prefork: /metrics on %s:%d, answered by the ROUTER for the "
                        "whole fleet\n",
                local.metrics_bind != NULL && local.metrics_bind[0] != '\0'
                    ? local.metrics_bind : "127.0.0.1", local.metrics_port);
        fflush(stderr);
    }

    /* S12-21: one shared page per worker, mapped before the fork so every
     * child inherits it, for the service-wide /metrics (server/fleet.h). */
    if (mynah_asr_fleet_map(workers) != 0)
        fprintf(stderr, "prefork: WARNING no shared fleet pages: /metrics will carry "
                        "the router's view only\n");

    worker_state *w = (worker_state *)calloc((size_t)workers, sizeof(*w));
    if (w == NULL) return MYNAH_ASR_PREFORK_ERROR;
    for (int i = 0; i < workers; ++i) { w[i].pid = -1; w[i].chan = -1; }

    int live = 0;
    for (int i = 0; i < workers; ++i) {
        int sp[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) != 0) {
            perror("prefork: socketpair");
            break;
        }
        /* Flush before forking: a buffered line in the parent's stdio would be
         * duplicated into every child and printed W times. */
        fflush(stderr);
        fflush(stdout);
        const pid_t pid = fork();
        if (pid < 0) {
            perror("prefork: fork");
            close(sp[0]);
            close(sp[1]);
            break;
        }
        if (pid == 0) {
            /* ------------------------------------------------------- child */
            close(sp[0]);
            /* The parent's ends of every earlier worker's channel came along
             * with the fork. Holding them would keep a dead worker's channel
             * open from this process's point of view and, worse, keep the
             * parent's end alive after the parent exits, so the sibling never
             * sees EOF. */
            for (int j = 0; j < i; ++j) {
                if (w[j].chan >= 0) close(w[j].chan);
            }
            free(w);
            close(local.listen_fd);  /* a worker must never accept: the parent routes */
            /* And never answers a scrape the router is answering: two
             * processes on one metrics port would hand a scraper whichever of
             * them the kernel picked. */
            if (metrics_fd >= 0) close(metrics_fd);

            g_worker_index = i;
            g_worker_chan = sp[1];
            *chan_fd = sp[1];
            /* Which model this worker owns, decided in the parent and carried
             * across by fork() rather than sent as a message. The caller reads
             * it back with mynah_asr_prefork_worker_model() and keeps that one
             * model. */
            g_worker_group = grp_of[i];

            char slice[192];
            const int pinned = pin_to_slice(cpus, cpu_first[i], cpu_n[i],
                                            slice, sizeof(slice));
            snprintf(g_configured_mask, sizeof(g_configured_mask), "%s", slice);

            /* Belt and braces on the pool. src/threads.c registers a
             * pthread_atfork child handler, so this is already done; calling it
             * is idempotent and documented as such, and it keeps the invariant
             * visible at the one place a reader looks for it.
             *
             * g_blas_mu (src/threads.c) and g_stats_mutex (src/qmat.c) used
             * to be named here as having no after-fork repair. Both have an
             * atfork child handler now, so the note is kept only to say that
             * it is no longer the open question it was; what
             * check_fork_preconditions() above still verifies rather than
             * assumes is the thread count itself. */
            mynah_asr_threadpool_after_fork();


            /* A decoder lane (a private pinned sub-team for the per-stream
             * decode loops) is designed in .work/serving-v2-design.md and not
             * implemented yet. The request is reported and ignored rather than
             * silently honoured, because "I asked for a lane" and "I have a
             * lane" are different claims. */
            char lane_why[256];
            snprintf(lane_why, sizeof(lane_why), "%s",
                     local.lane_cpus > 0
                         ? "decoder lane requested but not implemented in this runtime; "
                           "the decode loops run inline on the engine pool"
                         : "");
            /* The pool resolves its width once and caches it, so MYNAH_ASR_THREADS
             * has to have been right BEFORE the model was opened -- which is
             * what mynah_asr_prefork_reserve_threads() is for. Re-apply it here for
             * the case where it had not been resolved yet, then report what is
             * actually in force rather than what was planned. */
            {
                char buf[16];
                snprintf(buf, sizeof(buf), "%d", thr_of[i]);
                setenv("MYNAH_ASR_THREADS", buf, 1);
                g_worker_threads = mynah_asr_num_threads();
                if (g_worker_threads != thr_of[i]) {
                    fprintf(stderr, "prefork: worker %d WANTED %d threads but the pool "
                                    "is already fixed at %d; export MYNAH_ASR_THREADS=%d "
                                    "before starting the server\n",
                            i, thr_of[i], g_worker_threads, thr_of[i]);
                }
            }
            /* SIGUSR1, carefully. A worker must NOT get a handler installed
             * here: it inherits the one server/main.c installed before the
             * fork, and that is the one that prints the worker's counters.
             * Installing a second would mean whichever ran last silently won.
             *
             * But the parent forwards SIGUSR1 to every worker on a dump, and
             * the DEFAULT action for SIGUSR1 is to terminate the process. A
             * caller that used this module without installing a handler would
             * therefore lose its entire fleet the first time anyone asked for
             * statistics -- which is exactly what happened the first time this
             * was exercised. So: leave an inherited handler alone, and ignore
             * the signal only where there is none. A dump must never be able
             * to kill a worker. */
            {
                struct sigaction cur;
                memset(&cur, 0, sizeof(cur));
                if (sigaction(SIGUSR1, NULL, &cur) == 0 &&
                    cur.sa_handler == SIG_DFL) {
                    signal(SIGUSR1, SIG_IGN);
                }
            }
            fprintf(stderr, "prefork: worker %d pid %d threads %d cpus %s%s%s%s\n",
                    i, (int)getpid(), g_worker_threads, slice,
                    groups > 1 ? " model " : "",
                    groups > 1 && local.models[grp_of[i]] != NULL
                        ? local.models[grp_of[i]] : "",
                    pinned ? "" : "  <-- NOT PINNED");
            /* Printed only when someone asked for a lane, or when one is up.
             * The line above states the slice this worker was GIVEN; this one
             * states how that slice was divided and, when it was not, why --
             * `configured` and `actual` being different claims is the whole
             * reason the topology is printed at all. */
            if (local.lane_cpus > 0) {
                fprintf(stderr, "prefork: worker %d %s\n", i, lane_why);
            }
            /* The same facts, machine-readable, with the mask read back from
             * the kernel rather than the one we asked for (S3-1). */
            mynah_asr_prefork_print_topology(stderr, g_worker_threads);
            fflush(stderr);
            return MYNAH_ASR_PREFORK_CHILD;
        }
        /* ------------------------------------------------------------ parent */
        close(sp[1]);
        w[i].pid = pid;
        w[i].chan = sp[0];
        w[i].grp = grp_of[i];
        w[i].slots = slots_of[i];
        ++live;
    }

    if (live == 0) {
        fprintf(stderr, "prefork: no worker could be started\n");
        if (metrics_fd >= 0) close(metrics_fd);
        free(w);
        return MYNAH_ASR_PREFORK_ERROR;
    }

    signal(SIGPIPE, SIG_IGN);
    signal(SIGUSR1, on_usr1);

    /* Per-worker watchdog rings, parent-only: allocated after the fork so no
     * child ever inherits or frees them. */
    for (int i = 0; i < workers; ++i) {
        w[i].disp_cap = w[i].slots + 1;
        w[i].disp = (double *)calloc((size_t)w[i].disp_cap, sizeof(double));
    }

    /* The admission queue (rung 2). Bounded at q_per per LIVE worker; the
     * allocation follows the bound, and only an explicitly unbounded queue
     * grows. */
    int q_alloc = (q_per < 0) ? 16 : (q_per > 0 ? q_per * workers : 1);
    pf_queued *q = (pf_queued *)calloc((size_t)q_alloc, sizeof(*q));
    int q_n = 0;

    const size_t pfd_cap = (size_t)workers + 2u + PF_QPOLL_MAX + PF_LINGER_MAX +
                           (size_t)PF_PENDING_MAX;   /* +1: the metrics listener */
    struct pollfd *pfd = (struct pollfd *)calloc(pfd_cap, sizeof(*pfd));
    if (pfd == NULL || q == NULL) {
        for (int i = 0; i < workers; ++i) if (w[i].pid > 0) kill(w[i].pid, SIGTERM);
        for (int i = 0; i < workers; ++i) free(w[i].disp);
        if (metrics_fd >= 0) close(metrics_fd);
        free(pfd); free(q); free(w);
        return MYNAH_ASR_PREFORK_ERROR;
    }

    pf_linger linger[PF_LINGER_MAX];
    int linger_n = 0;
    long long linger_forced = 0;

    long long dispatched = 0;
    long long window_dispatched = 0;
    long long refused[MYNAH_ASR_PREFORK_REFUSE__COUNT];
    long long window_refused[MYNAH_ASR_PREFORK_REFUSE__COUNT];
    memset(refused, 0, sizeof(refused));
    memset(window_refused, 0, sizeof(window_refused));
    long long queued_total = 0, client_gone = 0;
    int queue_peak = 0;

    double window_start = mono_seconds();
    double prev = window_start;

    /* Connections accepted but not yet classified. Only ever non-empty in a
     * multi-model fleet: with one group the classifier is never called. */
    pf_pending pending[PF_PENDING_MAX];
    int pending_n = 0;
    long long classify_timeouts = 0;

    pf_router R;
    memset(&R, 0, sizeof(R));
    R.w = w;
    R.workers = workers;
    R.slots = slots;
    R.q_per = q_per;
    R.deadline_ms = deadline_ms;
    R.groups = groups;
    R.q = &q;
    R.q_alloc = &q_alloc;
    R.q_n = &q_n;
    R.queued_total = &queued_total;
    R.queue_peak = &queue_peak;
    R.dispatched = &dispatched;
    R.window_dispatched = &window_dispatched;
    R.refused = refused;
    R.window_refused = window_refused;
    R.linger = linger;
    R.linger_n = &linger_n;
    R.linger_forced = &linger_forced;

    const double router_start = mono_seconds();
    while (*stop == 0 && live > 0) {
        int nf = 0, listen_slot = -1, metrics_slot = -1;
        int map[PREFORK_MAX_WORKERS];
        for (int i = 0; i < workers; ++i) {
            if (w[i].pid <= 0) continue;
            map[nf] = i;
            pfd[nf].fd = w[i].chan;
            pfd[nf].events = POLLIN;
            pfd[nf].revents = 0;
            ++nf;
        }
        const int worker_slots = nf;

        /* RUNG 1, AND THE WHOLE REASON THIS FILE WAS REVISITED. The listener
         * is in the poll set UNCONDITIONALLY -- not "while a slot is free".
         * Watching it only when there is room is the optimisation that reads
         * as free and costs a measurement campaign: a client left in the
         * kernel backlog has not been accepted, so it has no queue entry, no
         * deadline and no timestamp, and nothing in this process can see it
         * waiting. The reference measured p95 TTFB 4470 ms with >97% of the
         * tail before accept(). Accepting and refusing in 200 microseconds is
         * strictly better than an invisible four-second wait, and it is the
         * only way rungs 2 and 3 get to exist at all. */
        listen_slot = nf;
        pfd[nf].fd = local.listen_fd;
        pfd[nf].events = POLLIN;
        pfd[nf].revents = 0;
        ++nf;

        /* The metrics listener, in the SAME poll set: a scrape is then a
         * bounded slice of this loop rather than a thread with its own view of
         * a table this loop is writing. */
        if (metrics_fd >= 0) {
            metrics_slot = nf;
            pfd[nf].fd = metrics_fd;
            pfd[nf].events = POLLIN;
            pfd[nf].revents = 0;
            ++nf;
        }

        /* Queued clients, watched for a HANGUP ONLY. events is deliberately 0:
         * a queued client has already sent its request, so asking for POLLIN
         * would make poll() return immediately every single time and spin the
         * router at 100% CPU. POLLERR/POLLHUP/POLLNVAL are reported in revents
         * regardless of events (POSIX), which is exactly the subset we want. */
        const int q_watch = q_n < PF_QPOLL_MAX ? q_n : PF_QPOLL_MAX;
        const int q_first = nf;
        for (int i = 0; i < q_watch; ++i) {
            pfd[nf].fd = q[i].fd;
            pfd[nf].events = 0;
            pfd[nf].revents = 0;
            ++nf;
        }

        const int l_first = nf;
        for (int i = 0; i < linger_n; ++i) {
            pfd[nf].fd = linger[i].fd;
            pfd[nf].events = linger_events(&linger[i]);
            pfd[nf].revents = 0;
            ++nf;
        }

        /* Connections waiting to be classified, watched for POLLIN: unlike a
         * queued client these have NOT finished speaking -- the whole reason
         * they are here is that the bytes we need have not arrived -- so
         * POLLIN is the right event and it does not spin. */
        const int p_first = nf;
        for (int i = 0; i < pending_n; ++i) {
            pfd[nf].fd = pending[i].fd;
            pfd[nf].events = POLLIN;
            pfd[nf].revents = 0;
            ++nf;
        }

        /* Sleep no longer than the next thing that needs attention, so a queue
         * deadline is honoured even when nothing else happens. */
        int timeout = 200;
        {
            const double now = mono_seconds();
            if (q_n > 0 && deadline_ms > 0) {
                const double left = (double)deadline_ms - (now - q[0].enqueued) * 1000.0;
                if (left < timeout) timeout = left > 0.0 ? (int)left : 0;
            }
            for (int i = 0; i < linger_n; ++i) {
                const double left = (linger[i].deadline - now) * 1000.0;
                if (left < timeout) timeout = left > 0.0 ? (int)left : 0;
            }
            for (int i = 0; i < pending_n; ++i) {
                const double left = (pending[i].deadline - now) * 1000.0;
                if (left < timeout) timeout = left > 0.0 ? (int)left : 0;
            }
            if (timeout < 0) timeout = 0;
        }

        const int ready = poll(pfd, (nfds_t)nf, timeout);

        {   /* Integrate in-flight over wall time whether or not poll returned
             * anything: the mean is only meaningful if idle time counts. */
            const double now = mono_seconds();
            const double dt = now - prev;
            prev = now;
            for (int i = 0; i < workers; ++i) {
                if (w[i].pid > 0) w[i].area += (double)w[i].active * dt;
            }
        }

        if (mynah_asr_prefork_take_dump_request()) {
            dump_table(w, workers, window_dispatched, window_refused, q_n,
                       prev - window_start);
            /* Forward it: one `kill -USR1 <parent>` should produce the whole
             * machine's view, the routing table from here and each worker's own
             * counters from the worker. */
            for (int i = 0; i < workers; ++i) {
                if (w[i].pid > 0) kill(w[i].pid, SIGUSR1);
            }
            for (int i = 0; i < workers; ++i) {
                w[i].window_assigned = 0;
                w[i].window_completed = 0;
                w[i].area = 0.0;
            }
            window_dispatched = 0;
            memset(window_refused, 0, sizeof(window_refused));
            window_start = prev;
        }

        if (ready < 0) {
            if (errno == EINTR) continue;
            perror("prefork: poll");
            break;
        }

        const double now = prev;

        /* ---- lingering closes, first: they free descriptors ---- */
        for (int i = linger_n - 1; i >= 0; --i) {
            const int k = l_first + i;
            const int fired = (k < nf) && (pfd[k].revents != 0);
            if (!fired && now < linger[i].deadline) continue;
            if (linger_step(&linger[i], now)) {
                linger[i] = linger[linger_n - 1];
                --linger_n;
            }
        }

        /* ---- completions ---- */
        for (int k = 0; k < worker_slots; ++k) {
            if ((pfd[k].revents & (POLLIN | POLLHUP | POLLERR)) == 0) continue;
            const int i = map[k];
            char buf[256];
            ssize_t n;
            do { n = read(w[i].chan, buf, sizeof(buf)); } while (n < 0 && errno == EINTR);
            if (n > 0) {
                w[i].completed += n;
                w[i].window_completed += n;
                w[i].active -= (int)n;
                disp_pop(&w[i], (int)n);
                if (w[i].active < 0) {
                    /* Would mean a worker reported a connection it was never
                     * given. Clamp, but say so: it is an accounting bug and it
                     * silently inflates capacity. */
                    fprintf(stderr, "prefork: worker %d reported more completions than "
                                    "dispatches; slot accounting is wrong\n", i);
                    w[i].active = 0;
                    w[i].disp_n = 0;
                }
            } else if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
                fprintf(stderr, "prefork: worker %d (pid %d) channel closed after "
                                "%lld completed\n", i, (int)w[i].pid, w[i].completed);
                /* Reap it now rather than at shutdown. A worker that dies in
                 * the first minute of a week-long run would otherwise sit as a
                 * zombie for the rest of it, which is harmless but is exactly
                 * the sort of thing an operator finds and cannot explain. */
                (void)waitpid(w[i].pid, NULL, WNOHANG);
                close(w[i].chan);
                w[i].chan = -1;
                w[i].pid = -1;
                /* Its connections died with it: charged to `lost`, not dropped. */
                w[i].lost += w[i].active;
                w[i].died = 1;
                fprintf(stderr, "prefork: worker %d lost %d connection(s) in flight; "
                                "it is not respawned, the fleet now has %d worker(s)\n",
                        i, w[i].active, live - 1);
                w[i].active = 0;
                w[i].disp_n = 0;
                --live;
            }
        }

        /* ---- RUNG 4's watchdog: outstanding connections past the cap ----
         * The parent cannot refuse these; the descriptor has belonged to the
         * worker since the handoff. What it can do is make the breach VISIBLE,
         * which is the difference between "the server felt slow" and a counter
         * an operator can point at. Enforcement -- stopping at the next frame
         * boundary -- belongs to the inference loop; see the note on
         * mynah_asr_prefork_service_cap_ms(). */
        if (g_service_cap_ms > 0) {
            for (int i = 0; i < workers; ++i) {
                if (w[i].pid <= 0 || w[i].disp_n <= 0 || w[i].disp == NULL) continue;
                const double age = (now - w[i].disp[w[i].disp_head]) * 1000.0;
                if (age >= (double)g_service_cap_ms) {
                    ++w[i].over_cap;
                    ++refused[MYNAH_ASR_PREFORK_REFUSE_SERVICE_CAP];
                    ++window_refused[MYNAH_ASR_PREFORK_REFUSE_SERVICE_CAP];
                    /* Charge it once: re-stamp the head so the same connection
                     * is not counted again on every tick until it finishes. */
                    w[i].disp[w[i].disp_head] = now;
                    fprintf(stderr, "prefork: worker %d has a connection %.0f ms into "
                                    "service, past the %d ms cap\n",
                            i, age, g_service_cap_ms);
                }
            }
        }

        /* ---- queued clients that hung up while waiting ----
         * Not a refusal: nobody is owed a status. Counted separately so it can
         * never be mistaken for one. */
        for (int i = q_watch - 1; i >= 0; --i) {
            const int k = q_first + i;
            if (k >= nf) continue;
            if ((pfd[k].revents & (POLLERR | POLLHUP | POLLNVAL)) == 0) continue;
            close(q[i].fd);
            ++client_gone;
            memmove(&q[i], &q[i + 1], (size_t)(q_n - i - 1) * sizeof(q[0]));
            --q_n;
        }

        /* ---- classification: connections whose model is not known yet ----
         * Handled before the drain so a connection that becomes decidable this
         * tick gets its slot in the same tick. Walked backwards because a
         * decided entry is removed by swapping the tail into its place. */
        for (int i = pending_n - 1; i >= 0; --i) {
            const int k = p_first + i;
            const int fired = (k < nf) && (pfd[k].revents != 0);
            const int expired = now >= pending[i].deadline;
            if (!fired && !expired) continue;

            char named[64];
            int grp = classify_model(pending[i].fd, local.models, groups,
                                         named, sizeof(named));
            if (grp == PF_CLASS_WAIT) {
                if (!expired) continue;
                /* Out of time. Route it as unspecified rather than refusing:
                 * the client has not done anything wrong, it is merely slow,
                 * and the worker will read the whole request and refuse it
                 * properly if the model turns out not to be ours. */
                ++classify_timeouts;
                grp = PF_CLASS_DEFAULT;
            }
            const int fd = pending[i].fd;
            pending[i] = pending[pending_n - 1];
            --pending_n;

            if (grp == PF_CLASS_UNKNOWN) {
                fprintf(stderr, "prefork: no worker holds model '%.32s'; refusing\n",
                        named);
                router_refuse(&R, fd, MYNAH_ASR_PREFORK_REFUSE_MODEL_NOT_FOUND, now);
                continue;
            }
            router_admit(&R, fd, grp < 0 ? default_grp : grp, now);
        }

        /* ---- RUNG 3, then RUNG 1: drain the queue ---- */
        router_drain(&R, now);

        if (metrics_slot >= 0 && metrics_slot < nf &&
            (pfd[metrics_slot].revents & POLLIN) != 0) {
            pf_metrics_view view;
            memset(&view, 0, sizeof(view));
            view.w = w;
            view.workers = workers;
            view.slots = slots;
            view.models = groups > 1 ? local.models : NULL;
            view.groups = groups;
            view.dispatched = dispatched;
            view.queued_total = queued_total;
            view.client_gone = client_gone;
            view.queued_now = q_n;
            view.queue_peak = queue_peak;
            view.refused = refused;
            view.build = mynah_asr_build_id();
            view.blas = mynah_asr_blas_provider();
            view.simd = mynah_asr_simd_profile();
            view.int8_kernel = mynah_asr_qmat_int8_kernel();
            view.uptime = mono_seconds() - router_start;
            mynah_asr_metrics_service(metrics_fd, pf_render_metrics, &view);
        }

        if (listen_slot < 0 || listen_slot >= nf ||
            (pfd[listen_slot].revents & POLLIN) == 0) continue;

        const int cfd = accept(local.listen_fd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
                errno == ECONNABORTED) continue;
            perror("prefork: accept");
            break;
        }

        /* SINGLE-MODEL FLEET: straight to the rungs, exactly as before.
         * Nothing peeks, nothing parks, nothing about the connection is looked
         * at. This branch is the guarantee that adding models did not change
         * the server that was already running. */
        if (groups <= 1) {
            router_admit(&R, cfd, 0, now);
            continue;
        }

        /* MULTI-MODEL: decide now if the prefix is already here -- which it is
         * for virtually every request, since a client sends its request line
         * and headers in one go -- and park it only when it is not. */
        {
            char named[64];
            const int grp = classify_model(cfd, local.models, groups,
                                               named, sizeof(named));
            if (grp == PF_CLASS_UNKNOWN) {
                fprintf(stderr, "prefork: no worker holds model '%.32s'; refusing\n",
                        named);
                router_refuse(&R, cfd, MYNAH_ASR_PREFORK_REFUSE_MODEL_NOT_FOUND, now);
                continue;
            }
            if (grp != PF_CLASS_WAIT) {
                router_admit(&R, cfd, grp < 0 ? default_grp : grp, now);
                continue;
            }
            if (pending_n >= PF_PENDING_MAX) {
                /* The classification set is full. Do NOT block to decide and do
                 * NOT drop it: send it to the default group, where a worker
                 * that reads the whole request will either serve it or refuse
                 * it with the same `model_not_found` code. The bound stays a
                 * bound; what it costs is precision, not a connection. */
                router_admit(&R, cfd, default_grp, now);
                continue;
            }
            pending[pending_n].fd = cfd;
            pending[pending_n].deadline = now + PF_CLASSIFY_MS / 1000.0;
            ++pending_n;
        }
    }

    /* ------------------------------------------------------------- shutdown */
    /* Anything still queued was accepted and never answered. Tell it so rather
     * than dropping it: a client that gets a 503 at shutdown knows to retry
     * elsewhere, and one whose socket simply dies does not. */
    for (int i = 0; i < q_n; ++i) {
        mynah_asr_prefork_refuse_and_close(q[i].fd, MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY);
        ++refused[MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY];
    }
    q_n = 0;
    /* Same for a connection still waiting to be classified: it was accepted,
     * so it is owed an answer, and "we are going away" is at-capacity as far as
     * a client is concerned. Never a model refusal -- we never found out
     * what it was asking for, and inventing a reason is worse than a generic
     * one. */
    for (int i = 0; i < pending_n; ++i) {
        mynah_asr_prefork_refuse_and_close(pending[i].fd, MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY);
        ++refused[MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY];
    }
    pending_n = 0;
    /* Finish the lingering closes properly: the whole point is that the client
     * reads the status, and abandoning them here would reintroduce the RST at
     * exactly the moment an operator is most likely to be watching. */
    for (int i = 0; i < linger_n; ++i) {
        const double end = mono_seconds() + PF_LINGER_MS / 1000.0;
        for (;;) {
            const double t = mono_seconds();
            if (linger_step(&linger[i], t)) break;
            if (t >= end) { if (linger[i].fd >= 0) close(linger[i].fd); break; }
            struct pollfd one;
            one.fd = linger[i].fd;
            one.events = linger_events(&linger[i]);
            one.revents = 0;
            if (poll(&one, 1, 20) < 0 && errno != EINTR) {
                close(linger[i].fd);
                break;
            }
        }
    }
    linger_n = 0;

    for (int i = 0; i < workers; ++i) {
        if (w[i].pid > 0) kill(w[i].pid, SIGTERM);
    }
    for (int i = 0; i < workers; ++i) {
        if (w[i].pid <= 0) continue;
        int status = 0;
        pid_t got;
        do { got = waitpid(w[i].pid, &status, 0); } while (got < 0 && errno == EINTR);
        (void)got;
        (void)status;
        if (w[i].chan >= 0) close(w[i].chan);
        w[i].chan = -1;
    }

    /* Totals for the run, not for the window since the last dump. Each rung
     * gets its own line: "503 x 900" tells an operator nothing, while
     * "at_capacity 12, queued_too_long 888" says immediately that the queue
     * deadline is the thing to look at. */
    fprintf(stderr, "prefork: final  dispatched=%lld queued_total=%lld "
                    "queue_peak=%d client_gone=%lld\n",
            dispatched, queued_total, queue_peak, client_gone);
    if (groups > 1) {
        /* The plan, restated at the end beside the counters it explains, and
         * the number that says whether classification was ever in doubt. A
         * non-zero classify_timeout means some connection's prefix did not
         * arrive within the budget and was routed as unspecified -- rare, but
         * it is the one path where the router guesses, so it is counted rather
         * than described. */
        fprintf(stderr, "prefork: models %s  classify_timeout=%lld\n",
                g_group_plan, classify_timeouts);
    }
    long long refused_all = 0;
    for (int r = 0; r < MYNAH_ASR_PREFORK_REFUSE__COUNT; ++r) refused_all += refused[r];
    fprintf(stderr, "prefork: refused=%lld", refused_all);
    for (int r = 0; r < MYNAH_ASR_PREFORK_REFUSE__COUNT; ++r) {
        fprintf(stderr, "  %s=%lld", REFUSAL[r].code, refused[r]);
    }
    fprintf(stderr, "\n");
    if (linger_forced > 0) {
        fprintf(stderr, "prefork: WARNING %lld refusals were closed without a full "
                        "lingering close (the %d-slot set was full); those clients "
                        "may have seen a reset instead of the status\n",
                linger_forced, PF_LINGER_MAX);
    }
    /* A leftover `still-in-flight` here is the interesting number: it means a
     * worker was handed a connection it never reported finished, which is a
     * leaked slot. */
    for (int i = 0; i < workers; ++i) {
        fprintf(stderr, "  worker %d%s%s: assigned=%lld completed=%lld"
                        " still-in-flight=%d lost=%lld%s over-service-cap=%lld\n",
                i,
                groups > 1 ? " " : "",
                groups > 1 && local.models[w[i].grp] != NULL
                    ? local.models[w[i].grp] : "",
                w[i].assigned, w[i].completed, w[i].active, w[i].lost,
                w[i].died ? " (died)" : "", w[i].over_cap);
        free(w[i].disp);
    }
    close(local.listen_fd);
    if (metrics_fd >= 0) close(metrics_fd);
    free(pfd);
    free(q);
    free(w);
    return MYNAH_ASR_PREFORK_PARENT_DONE;
}

