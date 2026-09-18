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
 * What runs on this thread: stream chunks (one per ready slot per step,
 * round-robin) and the offline REST jobs (at most one batched call per step).
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
