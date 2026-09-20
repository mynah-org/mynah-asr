/* One streaming session: the bounded PCM ring between an ingest thread and the
 * scheduler, plus everything the two threads have to agree on.
 *
 * The shape of the contract, because it is what keeps the scheduler honest:
 *
 *   - The INGEST thread (one per WebSocket connection, the HTTP thread renamed
 *     for the life of the stream) owns the slot's lifetime. It claims a free
 *     slot BEFORE the 101, pushes decoded PCM into the ring, posts control
 *     requests, and releases the slot when it returns. Nothing else ever moves
 *     a slot back to FREE, so a descriptor can never be handed to a new client
 *     while the previous ingest is still running.
 *   - The SCHEDULER thread owns `stream`, `out` and the per-session accounting
 *     once the slot is ACTIVE. It pops whole chunks, feeds the model, enqueues
 *     frames, and moves the slot to DONE. It never blocks on the ring and never
 *     blocks on a socket.
 *   - `mu` guards ONLY the queue and the request flags -- the two things both
 *     threads touch. Everything the scheduler alone reads and writes is outside
 *     it, so a step does not queue behind an ingest memcpy.
 *
 * Backpressure is the ring's whole purpose. When it is full the pusher WAITS on
 * `space`: a client uploading faster than real time is then throttled by its own
 * unread socket, and can never take more than one chunk per step away from the
 * other streams. That is fairness by construction rather than by policy, which
 * is the one thing the sibling repos never had to take back out.
 */
#ifndef MYNAH_ASR_SERVER_SLOT_H
#define MYNAH_ASR_SERVER_SLOT_H

#include <pthread.h>
#include <stddef.h>

#include "../src/mynah_asr.h"
#include "stream_out.h"

#define MYNAH_ASR_SLOT_LANG_CAP 24

/* Emission-lag histogram shape, shared by the per-session and the per-process
 * accounting: 257 buckets of 8 ms, the last one everything from 2048 ms up. */
#define MYNAH_ASR_LAG_BUCKETS 257
#define MYNAH_ASR_LAG_BUCKET_MS 8

typedef enum {
    MYNAH_ASR_SLOT_FREE = 0,   /* nobody's; claimable by an ingest thread     */
    MYNAH_ASR_SLOT_ACTIVE,     /* a session is running on the scheduler       */
    MYNAH_ASR_SLOT_FINISHING,  /* the tail is being flushed this step         */
    MYNAH_ASR_SLOT_DONE        /* the scheduler is finished; ingest may leave */
} mynah_asr_slot_state;

/* Requests an ingest thread posts and the scheduler reads at a step boundary.
 * Flags rather than a queue: a second finalize before the first was served is
 * the same finalize, and a cancel outranks everything whenever it arrives. */
enum {
    MYNAH_ASR_SLOT_REQ_FINALIZE = 1 << 0,
    MYNAH_ASR_SLOT_REQ_RESET    = 1 << 1,
    MYNAH_ASR_SLOT_REQ_CANCEL   = 1 << 2,
    /* Set with FINALIZE when the peer closed: emit `done`, then close. Without
     * it the socket stays open and the next audio starts a new utterance. */
    MYNAH_ASR_SLOT_REQ_CLOSE    = 1 << 3
};

/* Why a slot was cancelled -- the `code` of the error frame and the name of the
 * counter, kept as one value so the two can never disagree. */
typedef enum {
    MYNAH_ASR_SLOT_CANCEL_NONE = 0,
    MYNAH_ASR_SLOT_CANCEL_IDLE,        /* no audio for --idle-ms            */
    MYNAH_ASR_SLOT_CANCEL_FRAME,       /* a frame over --max-frame-bytes    */
    MYNAH_ASR_SLOT_CANCEL_PROTOCOL,    /* an unusable client message        */
    MYNAH_ASR_SLOT_CANCEL_PEER,        /* the peer hung up or stopped reading */
    MYNAH_ASR_SLOT_CANCEL_SHUTDOWN     /* SIGTERM while the stream was live  */
} mynah_asr_slot_cancel;

const char *mynah_asr_slot_cancel_code(mynah_asr_slot_cancel r);

/* When the last sample of a push landed. A delta produced at audio position p
 * is charged to the frame that CARRIED sample p, which is the only definition
 * of emission lag a client can reproduce from its own send timestamps. */
typedef struct {
    size_t end;   /* absolute index one past the last sample of this arrival */
    double t;     /* monotonic seconds at which it entered the ring          */
} mynah_asr_slot_arrival;

typedef struct mynah_asr_slot {
    int id;

    /* ---------------------------------------------------- guarded by `mu` */
    pthread_mutex_t mu;
    pthread_cond_t space;          /* the pusher waits here when the ring is full */
    mynah_asr_slot_state state;
    float *ring;                   /* allocated once at init, never resized */
    size_t cap, head, len;
    mynah_asr_slot_arrival *arr;   /* ditto */
    size_t arr_cap, arr_head, arr_len;
    size_t samples_in;             /* absolute: pushed since the session opened */
    size_t samples_consumed;       /* absolute: taken by the scheduler */
    int req;                       /* MYNAH_ASR_SLOT_REQ_* */
    mynah_asr_slot_cancel cancel_reason;
    char req_lang[MYNAH_ASR_SLOT_LANG_CAP];

    /* ------------------------------------ the scheduler's alone once ACTIVE */
    mynah_asr_stream *stream;      /* pooled: opened lazily, reset per session */
    int stream_lookahead;          /* the raw value it was opened with, or INT_MIN */
    mynah_asr_stream_out *out;     /* published by the ingest before ACTIVE */
    char lang[MYNAH_ASR_SLOT_LANG_CAP];
    int lookahead;                 /* what this session asked for */
    unsigned seq;
    int needs_reset;               /* a finalize without close: reset on next audio */
    double t_open, t_first_delta;
    double last_arrival;           /* of the newest sample the scheduler took */
    /* When the model last served this slot. `steps` says how many times; this
     * says how long ago, which is the only one of the two that can distinguish
     * a stream being starved right now from one that was simply admitted late. */
    double t_last_step;
    double lag_sum_ms, lag_max_ms;
    /* Emission lag of this session, as a fixed histogram so the `done` frame can
     * carry a median without keeping every sample: bucket = 8 ms, last bucket is
     * "2048 ms or more". The quantisation is stated in docs/server.md rather
     * than hidden -- a p50 to the nearest 8 ms is a measurement, a p50 from a
     * mean pretending to be one is not. */
    unsigned lag_hist[MYNAH_ASR_LAG_BUCKETS];
    int deltas, eous, steps;
    unsigned char *frame;          /* scratch for one outgoing WS frame */
    size_t frame_cap;
    float *take;                   /* scratch for one chunk of PCM */
    /* Who holds this slot's mutex, where they took it, and since when.
     *
     * The 2026-09-20 stall froze the scheduler inside its poll pass for 26 s and
     * the static audit found no lock-order cycle, which leaves a long critical
     * section -- and a long critical section has an owner that nothing in the
     * server could name. Two relaxed stores per acquisition, against the cost of
     * the mutex itself, buys a dump that says "waiting on slot 7, held by the
     * ingest thread in push for 26 s" instead of "waiting". */
    _Atomic unsigned long mu_owner;   /* a thread id, 0 = free                  */
    _Atomic double mu_since;          /* when it was taken                      */
    const char *_Atomic mu_where;     /* the call site that took it             */

    size_t take_cap;
} mynah_asr_slot;

/* Allocates the ring, the arrival records and the two scratch buffers. Nothing
 * in the per-step path allocates afterwards (the per-delta JSON build is the
 * declared exception). Returns 0, or -1 leaving the slot unusable but safe to
 * pass to _destroy. */
int mynah_asr_slot_init(mynah_asr_slot *s, int id, size_t ring_samples,
                        size_t take_samples);
void mynah_asr_slot_destroy(mynah_asr_slot *s);

/* Takes a FREE slot for a new session. `out` and the session parameters are
 * published under the same lock that makes the slot visible to the scheduler.
 * Returns 0, or -1 when the slot is busy. */
int mynah_asr_slot_claim(mynah_asr_slot *s, const char *lang, int lookahead,
                         mynah_asr_stream_out *out, double now);

/* Publishes the output writer on a slot already claimed. Claiming happens
 * BEFORE the 101 (a refusal must be an HTTP status, not a transport error) and
 * the writer only exists after it, so the scheduler ignores a claimed slot
 * until this call arms it. */
void mynah_asr_slot_arm(mynah_asr_slot *s, mynah_asr_stream_out *out);

/* The ingest thread letting go: back to FREE. The scheduler must already have
 * moved it to DONE or have been told to cancel; this only resets the queue. */
void mynah_asr_slot_release(mynah_asr_slot *s);

mynah_asr_slot_state mynah_asr_slot_get_state(mynah_asr_slot *s);

/* Everything a step needs about a slot, under one acquisition of its lock:
 * state, the armed writer (NULL while not armed) and the queued samples. */
mynah_asr_slot_state mynah_asr_slot_poll(mynah_asr_slot *s,
                                         mynah_asr_stream_out **out, size_t *avail);

/* The scheduler's "there is work" doorbell, called from inside a push or a
 * request while the slot lock is held. Set once at scheduler start; the lock
 * order is therefore always slot -> scheduler, and the scheduler never calls a
 * slot function while holding its own. */
void mynah_asr_slot_set_notify(void (*notify)(void));

/* Copies `n` samples into the ring, BLOCKING while it is full. Returns the
 * number accepted: less than `n` only when the session ended under us (cancel,
 * shutdown, or the scheduler finished the slot), which the ingest treats as
 * "stop reading this connection". */
size_t mynah_asr_slot_push(mynah_asr_slot *s, const float *samples, size_t n,
                           double now);

/* Scheduler side. Pops exactly min(n, len) samples and reports, through
 * `arrival`, when the last of them entered the ring. Returns the count. */
size_t mynah_asr_slot_take(mynah_asr_slot *s, float *dst, size_t n, double *arrival);

size_t mynah_asr_slot_available(mynah_asr_slot *s);

/* Posts a request. `lang` is only read for RESET. Wakes a blocked pusher when
 * the request ends the session, so the ingest never sits on a dead ring. */
void mynah_asr_slot_request(mynah_asr_slot *s, int flags, const char *lang,
                            mynah_asr_slot_cancel reason);

/* Reads and clears the pending request flags; `lang` (>= MYNAH_ASR_SLOT_LANG_CAP)
 * receives the reset language when RESET is among them. CANCEL is reported but
 * NOT cleared: it is terminal and every later look must still see it. */
int mynah_asr_slot_take_requests(mynah_asr_slot *s, char *lang,
                                 mynah_asr_slot_cancel *reason);

/* Moves the slot to a state the ingest can observe. The scheduler uses DONE to
 * say "I am finished with this session"; FINISHING is bookkeeping for /health. */
void mynah_asr_slot_set_state(mynah_asr_slot *s, mynah_asr_slot_state st);

/* Blocks until the scheduler has moved the slot to DONE, or the timeout runs
 * out. Returns 1 when DONE. The ingest thread uses it instead of polling, so a
 * connection's teardown costs one wakeup and not a sleep loop. */
int mynah_asr_slot_wait_done(mynah_asr_slot *s, int timeout_ms);

/* Monotonic seconds, the one clock the whole server measures lag with. */
double mynah_asr_now(void);

#endif
