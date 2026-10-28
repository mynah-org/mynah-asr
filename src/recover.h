/* Recovering an EMPTY streaming turn on the CPU library (S18,
 * .work/empty-turn-recovery.md): the decisions of recovery.h run on a spare
 * stream the caller pools, so nothing here touches the live session.
 *
 * Off the hot path by design: a caller runs it only for a turn whose primary
 * transcript is empty, and a server runs it on its own lane, never inside the
 * batched step. Everything it needs is carved at open; a recovery allocates
 * only the detokenised texts of its attempts. */
#ifndef MYNAH_ASR_RECOVER_H
#define MYNAH_ASR_RECOVER_H

#include "mynah_asr.h"
#include "recovery.h"

/* The model's own tokenizer (owned by the model). */
const mynah_asr_tokenizer *mynah_asr_model_tokenizer(const mynah_asr_model *m);

typedef struct mynah_asr_recoverer mynah_asr_recoverer;

/* One pooled stream (~13 MB on Nemotron) plus scratch for the longest
 * attempt: cfg->max_turn_s of turn behind the longest prefix. NULL on failure. */
mynah_asr_recoverer *mynah_asr_recoverer_open(mynah_asr_model *m, int lookahead,
                                              const mynah_asr_recover_cfg *cfg);
void mynah_asr_recoverer_close(mynah_asr_recoverer *r);

typedef struct {
    int accepted;          /* 1 = text() is the turn's recovered transcript */
    int attempts;          /* decodes run (0 when the turn was out of bounds) */
    int votes;             /* attempts that agreed with the accepted text */
    int winner;            /* index of the accepted attempt, -1 none */
    mynah_asr_attempt_kind winner_kind;
    char lang[16];         /* language tag of the accepted attempt ("" none) */
} mynah_asr_recover_result;

/* Recover one EMPTY turn. cur: its samples (16 kHz mono). prev / n_prev: the
 * previous turn's samples, or its tail (NULL = no replay vote). prev_text: the
 * previous turn's transcript (echo gate, TEXT rule; may be NULL). lang: the
 * session's tag. A turn longer than cfg->max_turn_s is not tried. Returns 0
 * (accepted or not, see res), -1 on a decode or allocation failure. */
int mynah_asr_recover_turn(mynah_asr_recoverer *r, const char *lang, const float *cur,
                           size_t n_cur, const float *prev, size_t n_prev,
                           const char *prev_text, mynah_asr_recover_result *res);

/* The accepted text of the last call ("" when none); valid until the next. */
const char *mynah_asr_recover_text(const mynah_asr_recoverer *r);

#endif
