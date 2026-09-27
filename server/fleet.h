/* Service-wide metrics for a prefork fleet (S12-21).
 *
 * The problem this solves, measured before it existed: under --prefork the
 * router answers /metrics, and the router never enters the model, so the whole
 * scheduler side -- session books, cancels by reason, audio seconds, emission
 * lag -- was exported by NOBODY in the production topology. An operator could
 * see connections and admission and not one fact about what the service did
 * with them.
 *
 * The mechanism is the smallest one that stays correct when a worker dies:
 *
 *   - Before the first fork the router maps ONE shared page per worker
 *     (MAP_SHARED | MAP_ANON, so it needs no file and no name).
 *   - Each worker publishes a compact snapshot of its own counters into its
 *     page every MYNAH_ASR_FLEET_PUBLISH_MS, under a seqlock: the writer is the
 *     only writer, the reader retries a torn read, nobody ever blocks.
 *   - The router sums the pages when it is scraped. A dead worker's page is
 *     still mapped in the router, so its last snapshot stays in the totals:
 *     fleet counters never go backwards when a worker dies, and the sessions
 *     that page still called active are reported as LOST, not dropped.
 *
 * The price is staleness, bounded and exported: a snapshot is at most one
 * publish period old (mynah_asr_fleet_snapshot_age_seconds), and a worker
 * killed with SIGKILL loses what it did after its last publish -- except its
 * connections, which the router counts itself (mynah_asr_worker_lost_total).
 *
 * A single-process server renders the same series from its own live snapshot,
 * as a fleet of one, so a dashboard never needs to know which topology it is
 * looking at. Nothing here runs on a step: collection reads counters the
 * scheduler already keeps, on the publisher's own thread. */
#ifndef MYNAH_ASR_SERVER_FLEET_H
#define MYNAH_ASR_SERVER_FLEET_H

#include "metrics.h"
#include "sched.h"

#define MYNAH_ASR_FLEET_PUBLISH_MS 250

/* Latency histogram edges, ms. Every edge is a multiple of the scheduler's
 * 8 ms lag bucket, so the emission-lag histogram exported with these edges is
 * EXACT, not interpolated (see mynah_asr_sched_lag_over). */
#define MYNAH_ASR_FLEET_MS_EDGES 13
/* Session duration edges, seconds. */
#define MYNAH_ASR_FLEET_S_EDGES 12

extern const double mynah_asr_fleet_ms_edges[MYNAH_ASR_FLEET_MS_EDGES];
extern const double mynah_asr_fleet_s_edges[MYNAH_ASR_FLEET_S_EDGES];

/* The bucket a value falls in: the first edge it does not exceed, or the
 * overflow bucket (index = edge count). */
int mynah_asr_fleet_ms_bucket(double ms);
int mynah_asr_fleet_s_bucket(double s);

/* One worker's facts. Counters are monotonic for the life of the worker;
 * `active`, `cap`, `backlog_*` are gauges. Histograms are per-edge counts with
 * one extra bucket for everything past the last edge (NOT cumulative: the
 * renderer accumulates). */
typedef struct {
    unsigned long sessions, completed, aborted, cancelled;
    unsigned long cancel_by[MYNAH_ASR_SCHED_CANCEL__COUNT];
    unsigned long abandoned, abandoned_recovered;
    unsigned long unbalanced_snapshots;   /* books read unbalanced: a bug if > 0 */
    int active, cap, active_peak, balanced;
    unsigned long steps, deltas;
    double audio_seconds;
    double model_busy_s;                  /* the model running, batched or solo */
    double uptime_s;
    unsigned long lag[MYNAH_ASR_FLEET_MS_EDGES + 1];
    double lag_sum_ms;
    unsigned long first_text[MYNAH_ASR_FLEET_MS_EDGES + 1];
    double first_text_sum_ms;
    unsigned long finalize[MYNAH_ASR_FLEET_MS_EDGES + 1];
    double finalize_sum_ms;
    unsigned long session[MYNAH_ASR_FLEET_S_EDGES + 1];
    double session_sum_s;
    double backlog_s_sum, backlog_s_max;  /* audio queued, not yet fed */
} mynah_asr_fleet_stats;

/* What only the router knows, rendered next to the sums. A single-process
 * server passes workers = up = 1 and zeros. */
typedef struct {
    int workers, workers_up;
    unsigned long worker_deaths;
    long long router_lost;                /* connections held by dead workers */
    double snapshot_age_max_s;            /* oldest live snapshot */
    unsigned long sessions_lost;          /* active in a dead worker's last page */
    int prefork;
} mynah_asr_fleet_router;

/* Router, before the first fork: maps `n` pages. Returns 0 or -1 (the fleet
 * then runs without service-wide metrics, and says so). */
int mynah_asr_fleet_map(int n);

/* Worker, after the scheduler started: publishes into page `index` from a
 * thread of its own until the process exits. No-op when nothing was mapped. */
void mynah_asr_fleet_publisher_start(int index);

/* Router: sums every page. `dead[i]` != 0 marks worker i dead: its counters
 * stay in the sum, its gauges do not, its active sessions go to
 * `r->sessions_lost`. Fills r->snapshot_age_max_s from the live pages.
 * Returns the number of pages that had ever been published. */
int mynah_asr_fleet_sum(mynah_asr_fleet_stats *sum, const int *dead,
                        mynah_asr_fleet_router *r);

/* This process's own snapshot (the scheduler's counters, compacted). */
void mynah_asr_fleet_collect(mynah_asr_fleet_stats *out);

/* Adds `b` into `a`: counters and histograms summed, gauges summed except the
 * maxima, which take the max. */
void mynah_asr_fleet_add(mynah_asr_fleet_stats *a, const mynah_asr_fleet_stats *b,
                         int with_gauges);

/* The mynah_asr_fleet_* series, no worker label anywhere. */
void mynah_asr_fleet_render(mynah_asr_metrics_buf *b, const mynah_asr_fleet_stats *s,
                            const mynah_asr_fleet_router *r);

#endif
