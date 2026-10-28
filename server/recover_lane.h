/* Recovery lanes: where an EMPTY streaming turn is re-decoded, off the
 * scheduler thread (opt-in, --empty-recovery; .work/empty-turn-recovery.md).
 *
 * A lane is one low-priority thread that owns its pooled recoverers
 * (src/recover.h: one spare stream per lookahead preset, opened once) and holds
 * at most ONE job. The lanes together are the bounded queue: depth = lanes, and
 * a turn that finds every lane busy is not queued, it is skipped (the caller
 * emits `done` as it always did). When the machine is full, refuse.
 *
 * Threading contract, the same shape as the slot's:
 *   - The SCHEDULER submits (copying the turn's audio and context into the
 *     lane's own buffers, so the lane never reads a slot) and collects results.
 *     It never waits for a lane.
 *   - The LANE runs mynah_asr_recover_turn and nothing else: the single-stream
 *     feed/finish path on its own stream, never the batched step and never the
 *     model's batch scratch (the server refuses --empty-recovery under
 *     MYNAH_ASR_FIN_STACK=1, the one setting that would route a plain feed
 *     through that scratch). Its parallel_for calls run serially in place
 *     (mynah_asr_thread_set_serial), so it never takes the pool away from the
 *     scheduler's step. When it is done it rings the scheduler's doorbell.
 *   - A job is IDLE -> BUSY (scheduler) -> DONE (lane) -> IDLE (scheduler), each
 *     edge under the lane's mutex. While BUSY only the lane touches the buffers;
 *     while DONE only the scheduler reads the result. */
#ifndef MYNAH_ASR_SERVER_RECOVER_LANE_H
#define MYNAH_ASR_SERVER_RECOVER_LANE_H

#include <stddef.h>

#include "../src/mynah_asr.h"
#include "../src/recover.h"

typedef struct {
    int slot_id;
    unsigned gen;            /* the turn's generation, as submitted */
    int rc;                  /* mynah_asr_recover_turn's return: 0, or -1 failed */
    int accepted;
    mynah_asr_attempt_kind kind;
    char lang[16];           /* language tag of the accepted attempt ("" none) */
    const char *text;        /* accepted text; valid only inside the collect callback */
    double seconds;          /* wall time the lane spent on this turn */
} mynah_asr_rlane_result;

/* Opens `lanes` recoverers for `lookahead` (an explicit preset value) and starts
 * the threads; other presets are opened lazily by the lane that first meets
 * them. `wake` is called by a lane when it has a result. 0, or -1 (nothing left
 * running). */
int mynah_asr_rlanes_start(mynah_asr_model *m, int lanes, int lookahead,
                           const mynah_asr_recover_cfg *cfg, void (*wake)(void));

/* Joins the lanes (a lane in the middle of a turn finishes it first) and closes
 * the recoverers. Safe when never started. */
void mynah_asr_rlanes_stop(void);

/* Scheduler side. Hands one turn to an IDLE lane: cur/prev/prev_text are copied
 * (cur is at most cfg->max_turn_s of samples, prev at most the replay tail).
 * Returns 0 when a lane took it, -1 when every lane is busy or the turn does
 * not fit. */
int mynah_asr_rlanes_submit(int slot_id, unsigned gen, int lookahead, const char *lang,
                            const float *cur, size_t n_cur, const float *prev,
                            size_t n_prev, const char *prev_text);

/* Scheduler side. Calls fn once per finished job, then frees that lane.
 * Returns how many it collected. */
int mynah_asr_rlanes_collect(void (*fn)(const mynah_asr_rlane_result *r, void *ud),
                             void *ud);

#endif
