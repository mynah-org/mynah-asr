/* obs.h — what a serving process can PROVE about itself.
 *
 * Four renderings of one set of facts, and one place that owns them:
 *
 *   the banner     printed once at start, unconditionally: [FLAGS],
 *                  [EFFECTIVE-CONFIG] (both from src/flags.c), [SERVER-CONFIG]
 *                  and [TOPOLOGY]. ENGINEERING.md §5 -- a run that is going to
 *                  be measured states its configuration in the same process,
 *                  before it serves anything.
 *   /v1/health     FACTS: what this worker did. Not configuration.
 *   /metrics       the same facts as Prometheus text, on their own port.
 *   SIGUSR1        the same facts once, to stderr, bracketed [DUMP] begin/end.
 *
 * WHY ONE MODULE. The banner is configuration and health is facts, and the two
 * must never be merged -- but they must never disagree either, and three
 * call sites that each read the counters themselves is how they come to. Every
 * renderer below reads ONE mynah_asr_sched_stats snapshot plus this module's
 * own config copy.
 *
 * THE HONEST-METRIC BOUNDARY. This process exports what it OBSERVES: audio it
 * received, steps it ran, deltas it emitted, the lag between a sample arriving
 * and the delta that answered it. It exports no client-side latency -- no TTFP,
 * no stall rate -- because those are measured at the far end of a socket this
 * process does not own, and a server that reports them is reporting a guess.
 * The harness (tools/bench/) owns those, and it is the only thing that may
 * quote them.
 *
 * CARDINALITY. The only labels anything here emits are `worker`, `reason`,
 * `code` and `le`, each drawn from a fixed compile-time set. Never a language,
 * never a model path, never anything a client can choose, and never text.
 */
#ifndef MYNAH_ASR_SERVER_OBS_H
#define MYNAH_ASR_SERVER_OBS_H

#include "../vendor/cJSON.h"
#include "metrics.h"

/* Everything the operator chose, captured once after the argument parse and
 * the model load. Strings are borrowed and must outlive the process, which
 * they do: they are argv, or literals, or the model's own config. */
typedef struct {
    const char *model_dir;
    const char *model_name;      /* "name" from mynah.json                    */
    const char *engine;          /* "engine" from mynah.json                  */
    const char *quant;           /* f32 | int8 | int4                         */
    const char *lid_dir;         /* --lid-model, or NULL                      */
    int   streaming;             /* the model has cache-aware presets         */
    int   lookahead_default;     /* the preset a session gets when it asks for none */
    int   lookaheads[8];
    int   n_lookaheads;
    double chunk_ms;             /* (lookahead+1) * encoder_frame_ms, 0 if none */
    int   port;
    int   cap;                   /* stream slots per worker                    */
    int   ring_seconds;
    int   idle_ms, ping_ms;
    double max_audio_seconds;
    size_t max_frame_bytes;
    int   max_pending;
    int   http_threads;
    int   batch;
    int   prefork_workers;       /* 0 = single process                         */
    int   prefork_threads;
    int   metrics_port;
    const char *metrics_bind;
} mynah_asr_obs_config;

/* Copies `cfg` and starts the uptime clock. Call once, before the banner. */
void mynah_asr_obs_init(const mynah_asr_obs_config *cfg);

/* [FLAGS] + [EFFECTIVE-CONFIG] + [SERVER-CONFIG] + [TOPOLOGY], in that order,
 * on stderr. Unconditional: there is no flag that turns the banner off, because
 * the one run whose banner is missing is the one that will be quoted. */
void mynah_asr_obs_banner(void);

/* The process facts /v1/health adds to the scheduler's own. */
void mynah_asr_obs_health(cJSON *into);

/* The worker's Prometheus page. Rendered on demand; allocates nothing that
 * outlives the call. */
void mynah_asr_obs_render_metrics(mynah_asr_metrics_buf *b, void *unused);

/* One-shot [DUMP] of the same facts to stderr, on SIGUSR1. `seq` increments
 * per dump so two dumps can never be mistaken for one. */
void mynah_asr_obs_dump(void);

/* A refusal this worker issued, by the `code` the client was given. The code
 * must be a compile-time literal: the table is bounded and anything it cannot
 * hold is counted as `other` rather than growing the label set. */
void mynah_asr_obs_refused(const char *code);

#endif
