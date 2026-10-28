/* gpu/server/recovery_pool.h — empty-turn recovery on spare engine slots (S18,
 * .work/empty-turn-recovery.md), the GPU server's form of src/recover.c.
 *
 * The decisions are those of src/recovery.h and the orchestration is the one
 * mynah_asr_recover_turn runs on the CPU library: the plan's attempts (noise
 * prefixes, then one replay of the previous turn's tail), each decoded from a
 * fresh slot, ownership of the replayed tokens by keep_from (cfg.rule, window
 * lookahead + 1), the gate, then the vote. What differs is only WHO decodes:
 * every attempt runs on one of the SPARE engine slots the server reserved at
 * open (never admitted to a session), and its chunks join the server's normal
 * cohort steps as extra lanes. The engine's contract 4 makes a lane's result
 * independent of its cohort, so an attempt's tokens are those of a decode
 * alone on that slot.
 *
 * Engine thread only; nothing here touches a socket or a session. A job is
 * started with every attempt it needs on free spare slots or not at all
 * (busy); it ends when every attempt finished (accepted or rejected), at its
 * deadline, or when its owner cancels it. Everything is allocated at open
 * except the detokenised texts of a finished job.
 *
 * Use, per engine step:
 *   k = rpool_lanes(p, reqs + n_live, room)   feeds the next chunk of up to
 *                                             `room` attempts, appends their
 *                                             requests (oldest job first)
 *   ... step reqs[0 .. n_live + k) ...
 *   rpool_stepped(p, reqs + n_live, outs + n_live, k)
 *   while (rpool_next_result(p, now, &res)) publish(res)
 */
#ifndef MYNAH_ASR_GPU_RECOVERY_POOL_H
#define MYNAH_ASR_GPU_RECOVERY_POOL_H

#include "asr_engine.h"
#include "recovery.h"
#include "tokenizer.h"

#include <stddef.h>

typedef struct rpool rpool;

/* The most spare slots a pool orders per step (rpool_lanes/rpool_next_result). */
#define RPOOL_MAX_SLOTS 1024

typedef struct {
    int first_slot, n_slots;     /* spare engine slots [first_slot, first_slot + n_slots) */
    double deadline_s;           /* a job older than this ends unpublished (<= 0 = none) */
    int sample_rate;             /* 16000 */
    int frame_samples;           /* one encoder frame in samples (1280 on Nemotron) */
    mynah_asr_recover_cfg cfg;   /* mynah_asr_recover_cfg_default unless a lab says otherwise */
} rpool_cfg;

typedef enum {
    RP_ACCEPTED = 0,             /* text is the turn's recovered transcript */
    RP_REJECTED,                 /* no text had the votes */
    RP_DEADLINE,                 /* the deadline passed first; nothing to publish */
} rpool_status;

typedef struct {
    int job;
    int owner;                   /* as given to rpool_start */
    unsigned gen;                /* as given to rpool_start */
    rpool_status status;
    int attempts;                /* attempts decoded to the end */
    int votes;                   /* attempts that agreed with the accepted text */
    int winner;                  /* index of the accepted attempt, -1 none */
    mynah_asr_attempt_kind kind; /* of the winner */
    const char *text;            /* "" unless accepted; valid until the next call */
    char lang[16];               /* the winner's language tag ("" none) */
    double age_s;                /* start to result */
} rpool_result;

typedef struct {
    unsigned long started, accepted_phase, accepted_replay, rejected, deadline, cancelled;
    unsigned long busy;          /* rpool_start refused: not enough free spare slots */
    unsigned long out_of_bounds; /* rpool_start refused: empty or longer than max_turn_s */
    unsigned long lanes;         /* attempt lanes requested into steps */
    unsigned long attempts;      /* attempts decoded to the end */
} rpool_stats;

/* NULL on failure (allocation). The tokenizer must outlive the pool. */
rpool *rpool_open(asr_engine *e, const mynah_asr_tokenizer *tk, const rpool_cfg *cfg);
void rpool_close(rpool *p);

/* Samples of the longest turn the pool recovers (cfg.max_turn_s). */
size_t rpool_max_turn_samples(const rpool *p);

/* Start recovering one EMPTY turn: cur/n_cur its samples, prev/n_prev the
 * previous turn's tail (NULL/0 = no replay vote), prev_text its transcript
 * (may be NULL), lang the session's tag, lookahead its preset. The samples are
 * copied. Returns the job id (>= 0), RP_START_BUSY when the attempts do not
 * all find a free spare slot, RP_START_BOUNDS when the turn is empty or longer
 * than cfg.max_turn_s, RP_START_ERROR when a slot reset failed. */
enum { RP_START_BUSY = -1, RP_START_BOUNDS = -2, RP_START_ERROR = -3 };
int rpool_start(rpool *p, int owner, unsigned gen, const char *lang, int lookahead,
                const float *cur, size_t n_cur, const float *prev, size_t n_prev,
                const char *prev_text, double now);

/* Drop a job (its owner went away or started a new utterance): its slots are
 * free again and it never produces a result. */
void rpool_cancel(rpool *p, int job);

/* Feed each unfinished attempt its next chunk (at most what the engine needs)
 * and append the requests of at most `room` attempts that can step, oldest job
 * first. Returns the count (0 when none), -1 when a feed failed (engine dead). */
int rpool_lanes(rpool *p, asr_step_req *reqs, int room);

/* After the step: outs[i] answers reqs[i] of the last rpool_lanes. */
void rpool_stepped(rpool *p, const asr_step_req *reqs, const asr_step_out *outs, int k);

/* 1 and *r filled when a job has ended (every attempt finished, or its
 * deadline passed), 0 when none has. The job's slots are free on return. */
int rpool_next_result(rpool *p, double now, rpool_result *r);

/* Jobs started and not yet returned or cancelled. */
int rpool_pending(const rpool *p);

void rpool_get_stats(const rpool *p, rpool_stats *s);

const char *rpool_kind_name(mynah_asr_attempt_kind k);

#endif
