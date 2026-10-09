/* gpu/server/hostprof.h — where the engine thread's wall goes (--profile-host),
 * and what host it runs on (the [TOPOLOGY] line).
 *
 * The GPU server has ONE engine thread: it scans the slots, copies ready
 * chunks out of the PCM rings, runs the host mel, waits for a cohort, steps
 * the engine and publishes the deltas, serially. While it does host work the
 * device is idle, so before any overlap work can be credited the loop's wall
 * has to be split into host time and device-wait time per phase. This module
 * keeps that split (DIAGNOSTIC, default off: with the profile off the loop
 * reads no clock it did not read before).
 *
 * Phases of the server loop (the engine reports its own phases inside the
 * step, asr_engine_stats.hprof_us, which this module prints next to these):
 *   scan         the slot scan: locks, requests, readiness checks
 *   stage_copy   ring -> stage buffer copies of ready chunks
 *   mel          asr_engine_slot_feed (the host streaming mel) + its book
 *   idle         waiting with nothing ready
 *   cohort_wait  waiting for the cohort timer (--cohort-ms)
 *   step         asr_engine_step, or submit + finish with --stage-ahead
 *                (split by the engine's own phases, which sum to it)
 *   publish      deltas, done frames, resets, lag books
 *   window       the stage-ahead scan (ring copies + host mel of the next
 *                chunks): between submit and finish, i.e. under the encoder
 *                pass, and the pre-scan at the top of the loop (--stage-ahead
 *                or a feed team); 0 with both off
 *
 * Threading: the engine thread accumulates a cycle locally and folds it into
 * the shared totals under the caller's lock; readers copy the totals under the
 * same lock. */
#ifndef MYNAH_ASR_GPU_HOSTPROF_H
#define MYNAH_ASR_GPU_HOSTPROF_H

#include <stddef.h>

enum { HP_SCAN = 0, HP_STAGE, HP_MEL, HP_IDLE, HP_COHORT_WAIT, HP_STEP, HP_PUBLISH, HP_WINDOW, HP__N };
extern const char *const HP_NAME[HP__N];

#define HP_HIST_BUCKETS 800      /* 250 us buckets: 0 .. 200 ms, the last one open */
#define HP_HIST_US 250.0

typedef struct {
    double us[HP__N];            /* engine-thread wall per server phase */
    unsigned long cycles;        /* engine steps (one cohort each) */
    unsigned long lanes;         /* lanes requested over those steps */
    unsigned long chunks;        /* ready chunks staged */
    double dev_wait_us;          /* device waits inside the steps (engine phases *_wait) */
    unsigned long hist_cycle[HP_HIST_BUCKETS];  /* step start -> next step start */
    unsigned long hist_host[HP_HIST_BUCKETS];   /* host busy per cycle (no waits) */
    double max_cycle_us, max_host_us;
    double t_start;              /* monotonic seconds when the profile started */
    /* getrusage(RUSAGE_THREAD) of the engine thread at its last fold (Linux) */
    double utime_s, stime_s;
    long nvcsw, nivcsw, minflt, majflt;
    int have_rusage;
} hostprof;

/* the engine thread's private accumulator for the cycle in progress */
typedef struct {
    double us[HP__N];
    unsigned long chunks;
    double t;                    /* the running lap clock */
    double cycle_start;          /* when the previous step started (0 = none) */
    double host_us;              /* host busy since the previous step started */
} hostprof_cycle;

double hostprof_now(void);
/* charge the time since c->t to `ph`, restart the lap clock */
void hostprof_lap(hostprof_cycle *c, int ph);
/* fold the cycle in progress into the totals (call under the shared lock);
 * `stepped`: a step ran in it, with `lanes` lanes and `dev_wait_us` of device
 * wait inside the step; `step_start` its start */
void hostprof_fold(hostprof *h, hostprof_cycle *c, int stepped, int lanes, double dev_wait_us,
                   double step_start);
/* the engine thread's rusage, read from the engine thread itself */
void hostprof_rusage(hostprof *h);

/* [HOSTP] lines: the server phases, the engine's phases (engine_us/passes/syncs
 * from asr_engine_stats, may be NULL), the tails and the thread's rusage.
 * Returns the bytes written into buf. */
size_t hostprof_report(const hostprof *h, const double *engine_us, const char *const *engine_names,
                       int n_engine, unsigned long passes, unsigned long syncs,
                       unsigned long seq, char *buf, size_t cap);
/* Prometheus counters, appended to buf */
size_t hostprof_metrics(const hostprof *h, const double *engine_us, const char *const *engine_names,
                        int n_engine, char *buf, size_t cap);
double hostprof_quantile(const unsigned long *hist, double q);

/* [TOPOLOGY] facts of this host as this process sees them: CPU model, the
 * affinity mask (count and list), the cgroup CPU quota, NUMA nodes, the
 * open-file limit, and the GPU's NUMA node when its PCI bus id is known. */
size_t host_topology_line(const char *gpu_pci_bus_id, char *buf, size_t cap);

#endif
