/* The scheduler: ONE thread per worker process, and the only thread that calls
 * a mynah_asr_* inference function once the model is loaded.
 *
 * Why one thread and not a pool. The sibling repos measured it twice and the
 * answer did not move: a second submitter on the engine pool cost qwen-tts its
 * cohort (C4 STREAM .847 -> 1.296, TTFA 174 -> 1126 ms, context switches x2.9,
 * cores flat), and every variant that added a gate, a helper thread or a
 * parking policy on the same pool lost. What wins is removing work from the
 * serial path: socket writes went to `stream_out`, and here the model has one
 * owner, so a step is a step and never a queue behind another step.
 *
 * What runs on this thread: stream chunks (one per ready slot per step, and the
 * whole ready set in ONE mynah_asr_stream_step_batch call -- S2-2b) and the
 * offline REST jobs (at most one batched call per step).
 * What does NOT: reading sockets, writing sockets, parsing HTTP, JSON of the
 * REST responses. `mynah_asr_sched_assert_thread` is called at the top of every
 * callback and at every inference call site, so the invariant is checked by the
 * program rather than promised by this comment.
 *
 * One mutex lives in this module. It guards the offline job queue, the
 * scheduler's own wake flag and the counters `/v1/health` reports -- the three
 * things an HTTP thread and the scheduler share. Slot queues have their own
 * (slot.c) and the socket has the writer's (stream_out.c); there is no other.
 */
#ifndef MYNAH_ASR_SERVER_SCHED_H
#define MYNAH_ASR_SERVER_SCHED_H

#include "../src/mynah_asr.h"
#include "../vendor/cJSON.h"
#include "slot.h"

typedef struct {
    mynah_asr_model *model;
    mynah_asr_model *lid;        /* --lid-model detector, or NULL */
    int slots;                   /* --cap: streams this worker holds at once */
    int ring_seconds;            /* --ring-seconds: PCM buffered per slot */
    int max_batch;               /* --batch: offline jobs aggregated per step */
    int max_pending;             /* --max-pending: offline jobs queued */
    size_t out_ring_bytes;       /* per-connection output ring */
    int send_timeout_ms;         /* SO_SNDTIMEO on a stream's socket */
    /* --batch-window-ms: how long a step may WAIT for more streams to become
     * ready before it runs. 0 keeps the historical behaviour (step the moment
     * anything is ready).
     *
     * The capacity law T_step(B) = a + b*B prices ONE step serving B streams,
     * and its whole economy is that the fixed cost `a` is paid once and
     * amortised over B. Clients push audio on their own phase, so a scheduler
     * that steps on every arrival sees a ready set of two or three where B
     * should be sixteen: measured on 24 Neoverse-V2 cores, mean ready set 3.73
     * at c=16, i.e. `a` paid four times per chunk period.
     *
     * This does NOT change the chunk (repo rule 5: load never enters the chunk
     * size). It changes when the step runs, by a bounded amount, and it is
     * skipped entirely for a stream's first chunk and for a finalizing stream,
     * so neither first-partial latency nor a close pays for it. */
    int batch_window_ms;
} mynah_asr_sched_config;

typedef enum {
    MYNAH_ASR_JOB_TRANSCRIBE = 0,
    MYNAH_ASR_JOB_DETECT_LANG
} mynah_asr_job_kind;

/* One offline unit of work. The HTTP thread fills the inputs, submits, and
 * blocks until the scheduler has filled the outputs. Allocated on the caller's
 * stack: the queue links jobs, it never owns them. */
typedef struct mynah_asr_offline_job {
    mynah_asr_job_kind kind;
    const float *samples;
    size_t n_samples;
    char lang[MYNAH_ASR_SLOT_LANG_CAP];
    int lookahead;
    int want_words;

    char *text;                  /* TRANSCRIBE: malloc'd, the caller frees */
    char lang_out[16];
    mynah_asr_word *words;
    int n_words;
    char detected[16];           /* DETECT_LANG */
    int rc;                      /* 0 ok, -1 failed */

    int done;
    struct mynah_asr_offline_job *next;
} mynah_asr_offline_job;

/* Allocates the slot table and starts the thread. Returns 0 or -1. */
int mynah_asr_sched_start(const mynah_asr_sched_config *cfg);

/* 1 when this model has cache-aware streaming presets at all. Answered from the
 * model's config, so a WebSocket on an offline-only model is a 400 BEFORE the
 * upgrade instead of a stream that dies after it. */
int mynah_asr_sched_streaming(void);

/* When mynah_asr_sched_streaming() is 0, the reason, in a form that can go
 * straight into a refusal body: either the model has no presets at all, or it
 * has them and uses something the incremental encoder does not implement
 * (mynah_asr_stream_unsupported). NULL while streaming is available. */
const char *mynah_asr_sched_stream_why(void);

/* The sample rate of the model this worker loaded: what the WebSocket accepts,
 * what REST resamples to, and what every "samples -> seconds" in the server
 * divides by. From the pack (mynah_asr_sample_rate), 16000 before the
 * scheduler has started. */
int mynah_asr_sched_sample_rate(void);

/* Claims a slot for a new session, or NULL when this worker is full (the caller
 * answers 503 before the 101). The returned slot is not yet armed: the caller
 * completes the handshake, starts its writer and calls mynah_asr_slot_arm. */
mynah_asr_slot *mynah_asr_sched_claim(const char *lang, int lookahead);

/* Wakes the scheduler. Cheap and idempotent. */
void mynah_asr_sched_wake(void);

/* Runs one offline job on the scheduler thread and blocks until it is done.
 * Returns 0 when it ran and succeeded, -1 when it ran and failed (a 400: the
 * language, the audio) and -2 when it was refused before running because the
 * queue is at --max-pending or the server is stopping (a 503). The three are
 * kept apart because they are three different things for the client to do. */
int mynah_asr_sched_submit(mynah_asr_offline_job *job);

/* Facts for /v1/health, added to `into`. */
void mynah_asr_sched_health(cJSON *into);

/* ------------------------------------------------------ counters as FACTS
 *
 * Why a snapshot struct and not three readers: /v1/health, /metrics and the
 * SIGUSR1 dump all report the same numbers, and three copies of "load the
 * atomics" is how the three come to disagree about what `cancelled` means.
 * One read, three renderings.
 *
 * Every field below is a counter the scheduler ALREADY maintains on a path it
 * already runs; nothing here added work to a step beyond two relaxed adds
 * (audio samples and the lag sum) next to adds that were already there. */

/* Why a session ended, bucketed. The bucket is chosen from the same `code`
 * string the client was sent, so a counter can never name a reason the client
 * did not see. */
typedef enum {
    MYNAH_ASR_SCHED_CANCEL_IDLE = 0,      /* idle_timeout                     */
    MYNAH_ASR_SCHED_CANCEL_PEER,          /* peer_gone                        */
    MYNAH_ASR_SCHED_CANCEL_FRAME,         /* frame_too_large                  */
    MYNAH_ASR_SCHED_CANCEL_PROTOCOL,      /* protocol_error                   */
    MYNAH_ASR_SCHED_CANCEL_SHUTDOWN,      /* shutting_down                    */
    MYNAH_ASR_SCHED_CANCEL_AUDIO_LIMIT,   /* audio_limit (--max-audio-seconds)*/
    MYNAH_ASR_SCHED_CANCEL_DECODE,        /* decode_failed                    */
    MYNAH_ASR_SCHED_CANCEL_OTHER,         /* anything else, counted not lost  */
    MYNAH_ASR_SCHED_CANCEL__COUNT
} mynah_asr_sched_cancel_bucket;

const char *mynah_asr_sched_cancel_bucket_name(int bucket);

/* Step-wall accounting is kept per ready-set size B, because the cadence law
 * T_step(B) = a + b*B (.work/serving-v2-design.md §3) cannot be fitted from a
 * single mean: two sizes in one run give a and b, and a mean over a run whose
 * B moved gives neither. Index = B, the last bucket is "that many or more". */
#define MYNAH_ASR_SCHED_B_BUCKETS 33

typedef struct {
    int  slots_active, slots_cap, streaming;
    unsigned long steps, deltas, eous, sessions, cancelled;
    unsigned long offline_done;
    int  offline_pending, offline_max_pending;
    unsigned long cancel_by[MYNAH_ASR_SCHED_CANCEL__COUNT];
    double audio_seconds;          /* fed to the model: streams + offline jobs */
    unsigned long lag_count;       /* = deltas, but read in the same snapshot  */
    double lag_sum_ms, lag_max_ms;
    unsigned long lag_hist[MYNAH_ASR_LAG_BUCKETS];

    /* S2-2b: what the batched step actually did. `batched_steps` counts calls
     * to mynah_asr_stream_step_batch, `ready_sum` sums the B of those calls (so
     * the mean ready-set size is a division and not a third counter), and
     * `rows_stacked` accumulates the library's own rows-stacked deltas -- 0 on
     * a run with traffic means every step degraded to the single path, which is
     * the visible-fallback rule of ENGINEERING.md §6. */
    unsigned long batched_steps, ready_sum;
    unsigned long long rows_stacked;
    double step_wall_ms_sum;
    unsigned long step_wall_count;          /* = batched_steps */
    unsigned long step_b_count[MYNAH_ASR_SCHED_B_BUCKETS];
    double step_b_wall_ms[MYNAH_ASR_SCHED_B_BUCKETS];
    /* The collection window: how often it was entered, how long it waited in
     * total, and how often it ended because every live stream was ready rather
     * than because it ran out of time. A window that always times out is too
     * long or the streams are not on a common cadence; one that always fills is
     * free. Both are readable only as a pair. */
    /* The scheduler thread's own liveness: how many times the loop has gone
     * round, which phase it is in, and for how long. Static loops with phase 7
     * is a lost wakeup; climbing loops that stage nothing is a livelock; static
     * loops with phase 4 is a step that will not end. */
    unsigned long loops;
    int phase;                  /* 1 slot-poll 2 reset 3 stage 4 step 5 finalize
                                   6 offline 7 park 8 take-requests 9 cancel
                                   10 peer-check */
    int phase_slot;             /* which slot that pass was on (-1 = not a slot) */
    double phase_s;
    unsigned long window_entered, window_filled;
    double window_wait_ms_sum;
} mynah_asr_sched_stats;

/* One live slot, as the scheduler sees it. The aggregate counters say the fleet
 * is behind; only this says WHICH streams are behind and what state they are in
 * when they are. A stall on 2026-09-20 kept p50 emission lag at 70-95 ms for ten
 * minutes while p95 reached 55 s in three-minute episodes -- most streams fine,
 * a subset starved, self-healing. No aggregate can express that. */
typedef struct {
    int id;
    int state;                  /* mynah_asr_slot_state                        */
    int has_out;                /* the scheduler skips a slot with no output    */
    int has_stream;             /* NULL until the first step opens one          */
    int lookahead;
    unsigned long steps, deltas;
    size_t ring_samples;        /* audio received and not yet taken             */
    size_t need_samples;        /* what one step of this stream wants           */
    double age_s;               /* since the slot was claimed                   */
    double since_arrival_s;     /* since the last audio arrived (-1 = never)    */
    double since_step_s;        /* since the model last served it (-1 = never)  */
    int ready;                  /* the ring holds a whole chunk for this stream  */
    double lag_max_ms;
} mynah_asr_sched_slot_view;

/* Fills up to `max` views, returns how many slots were live. Safe from any
 * thread: it reads each slot under that slot's own lock, which is what the
 * scheduler's own poll does. */
int mynah_asr_sched_slots_view(mynah_asr_sched_slot_view *out, int max);

void mynah_asr_sched_stats_read(mynah_asr_sched_stats *out);

/* Quantile of the emission-lag histogram, in ms, quantised to the 8 ms bucket
 * the histogram keeps. A p95 to the nearest 8 ms is a measurement; a p95
 * interpolated out of a mean is not. */
double mynah_asr_sched_lag_quantile(const unsigned long *hist, double q);

/* EXACTLY how many deltas had a lag of `ms` or more. Exact, not estimated,
 * whenever `ms` is a multiple of MYNAH_ASR_LAG_BUCKET_MS -- which is why the
 * thresholds /metrics exports are rounded up to a bucket edge before they are
 * named in a label. */
unsigned long mynah_asr_sched_lag_over(const unsigned long *hist, int ms);

/* Counts a session ended by the INGEST side under a code the scheduler never
 * sees (today: `audio_limit`). Same buckets, same total, one funnel. */
void mynah_asr_sched_note_cancel(const char *code);

/* Slots not FREE right now -- what /v1/health reports as `inflight`. */
int mynah_asr_sched_active(void);

/* Stops admitting, cancels live slots with an `error` frame carrying
 * code "shutting_down", lets the step finish, joins and frees the table.
 * The model is the caller's to free afterwards. */
void mynah_asr_sched_stop(void);

/* Aborts (debug) or complains once (release) when a thread that is not the
 * scheduler reaches a model call. `where` names the call site. */
void mynah_asr_sched_assert_thread(const char *where);

#endif
