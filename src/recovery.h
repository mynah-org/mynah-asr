/* Empty-turn recovery: the decisions that need no model (S18,
 * .work/empty-turn-recovery.md).
 *
 * A context replay decodes [tail of the previous turn's audio][current turn]
 * from a fresh state. Two questions are answered here, as pure functions over
 * token ids, emission frames and text, so they can be tested without weights:
 *
 *   - which replayed tokens the current turn OWNS (the boundary), and
 *   - whether a recovered text is fit to publish (the gate).
 *
 * Word structure comes from the tokenizer's own piece helpers; nothing here
 * allocates. */
#ifndef MYNAH_ASR_RECOVERY_H
#define MYNAH_ASR_RECOVERY_H

#include "tokenizer.h"

/* Ownership of the replayed tokens.
 *
 * A WORD is a piece carrying the mark ▁ plus the unmarked pieces after it, and
 * it must hold a letter or digit; a marked run without one ("▁." , "▁" "?") is
 * punctuation and belongs to the word before it.
 *
 * FRAME: a token belongs to the current turn when it was emitted at an encoder
 *   frame >= keep_from. The naive rule, kept as the baseline: a transducer emits
 *   late, so it hands the current turn the context's sentence-final "?" or the
 *   second half of a word that started in the context.
 * WORD:  a WORD belongs to the turn in which its FIRST piece was emitted: the
 *   current turn starts at the first word opened at a frame >= keep_from.
 *   Punctuation and continuation pieces follow their word's owner.
 * TEXT:  WORD, then a leading current-turn word that completes the context's
 *   own transcript AND whose first piece was emitted less than lag_frames after
 *   keep_from is given back to the context (and so on, word by word): the
 *   transducer's emission lag at a turn end, measured at 0-1 frames on every
 *   such word (.work/empty-turn-recovery.md E12). Needs that transcript;
 *   without one it is WORD.
 *
 * The invariant (E12): ownership is per word; a word belongs to the turn in
 * which its first piece is emitted and its pieces and trailing punctuation
 * follow it, except that a word opened within one encoder chunk of the
 * boundary that completes the previous turn's transcript is the previous
 * turn's. Known cost, pinned by the tests: a current turn whose first word
 * repeats the previous turn's last word inside that chunk loses it. */
typedef enum {
    MYNAH_ASR_KEEP_FRAME = 0,
    MYNAH_ASR_KEEP_WORD = 1,
    MYNAH_ASR_KEEP_TEXT = 2,
} mynah_asr_keep_rule;

/* Index of the first token the current turn owns in tokens[0..n), or n when it
 * owns none. frames[i] is the absolute encoder frame where token i was emitted;
 * keep_from is the first encoder frame of the current turn's audio.
 * context_text is the previous turn's transcript (TEXT only; may be NULL).
 * lag_frames: TEXT's window, one encoder chunk (lookahead + 1); <= 0 = no window.
 * Special <...> tokens (language tags) never start the current turn. */
int mynah_asr_replay_keep_from(const mynah_asr_tokenizer *tk, const int *tokens,
                               const int *frames, int n, int keep_from,
                               mynah_asr_keep_rule rule, const char *context_text,
                               int lag_frames);

/* Why a recovered text was or was not accepted. */
typedef enum {
    MYNAH_ASR_GATE_OK = 0,
    MYNAH_ASR_GATE_EMPTY,   /* no letter or digit at all */
    MYNAH_ASR_GATE_LANG,    /* the model tagged another language than the session's */
    MYNAH_ASR_GATE_ECHO,    /* the text only repeats the end of the context transcript */
} mynah_asr_gate_verdict;

/* Gate a recovered text. session_lang: the tag the session asked for ("en-US",
 * "auto" or NULL = no constraint). detected_lang: the <xx-XX> tag the decode
 * emitted ("" or NULL = none); compared on the language base ("en-GB" ~ "en-US").
 * context_text may be NULL. */
mynah_asr_gate_verdict mynah_asr_recovery_gate(const char *text, const char *session_lang,
                                               const char *detected_lang,
                                               const char *context_text);

const char *mynah_asr_gate_name(mynah_asr_gate_verdict v);

/* 1 when the text holds at least one letter or digit: what "not empty" means
 * for a turn. A transcript of only punctuation or spaces is empty. */
int mynah_asr_text_has_content(const char *text);

/* 1 when the two texts hold the same words once case and punctuation are
 * ignored ("Yes, please." ~ "yes please"); two empty texts are not the same. */
int mynah_asr_text_same_words(const char *a, const char *b);

/* ------------------------------------------------ recovering an EMPTY turn
 * What the evidence supports (.work/empty-turn-recovery.md, E3/E5/E10/E11):
 * re-decoding the turn behind a short content-free prefix moves it against the
 * encoder's chunk grid and recovers part of the empties; one short replay of
 * the previous turn adds a vote; long previous-turn context suppresses the
 * current turn and is never used. No single attempt is published on its own:
 * a text is accepted only when min_votes attempts agree on its words, which is
 * what kept silence and noise from being "recovered".
 *
 * The plan, the audio of an attempt and the vote are pure, so a caller with
 * its own decoder (the GPU server's engine slots) runs exactly this logic. */
#define MYNAH_ASR_RECOVER_MAX_ATTEMPTS 8

typedef enum {
    MYNAH_ASR_ATTEMPT_PHASE = 0,   /* [low-level noise prefix][current turn] */
    MYNAH_ASR_ATTEMPT_REPLAY = 1,  /* [tail of the previous turn][current turn] */
} mynah_asr_attempt_kind;

typedef struct {
    mynah_asr_attempt_kind kind;
    int prefix_frames;             /* = keep_from of the attempt's decode */
} mynah_asr_attempt;

typedef struct {
    int phase_frames[MYNAH_ASR_RECOVER_MAX_ATTEMPTS];
    int n_phases;                  /* default 3: {3, 4, 6} frames = 240/320/480 ms */
    int replay_frames;             /* previous-turn tail as one more vote; 0 = off; default 6 */
    int min_votes;                 /* default 2 */
    mynah_asr_keep_rule rule;      /* ownership of the replayed tokens; default TEXT */
    int lag_frames;                /* TEXT's window; 0 = the stream's chunk (lookahead + 1) */
    double max_turn_s;             /* longer turns are not recovered; default 4 s */
} mynah_asr_recover_cfg;

void mynah_asr_recover_cfg_default(mynah_asr_recover_cfg *c);

/* The attempts for one turn, in order: every phase prefix, then the replay
 * when replay_frames > 0 and the previous turn has at least that much audio.
 * Writes at most MYNAH_ASR_RECOVER_MAX_ATTEMPTS; returns the count. */
int mynah_asr_recover_plan(const mynah_asr_recover_cfg *c, size_t n_prev, int frame_samples,
                           mynah_asr_attempt *out);

/* The samples of an attempt into out: prefix (deterministic noise at -60 dBFS
 * for PHASE, the last prefix_frames frames of prev for REPLAY), then cur.
 * out holds prefix_frames * frame_samples + n_cur floats. Returns the length. */
size_t mynah_asr_recover_build(const mynah_asr_attempt *a, int frame_samples, const float *cur,
                               size_t n_cur, const float *prev, size_t n_prev, float *out);

/* The accepted attempt: the first text whose words equal those of at least
 * min_votes of the n texts (itself included). NULL or contentless texts never
 * vote. Returns its index, or -1 when no text has the votes. */
int mynah_asr_recover_vote(const char *const *texts, int n, int min_votes);

/* The same decision taken early: after the first k of n attempts, is the
 * outcome of mynah_asr_recover_vote over all n already fixed whatever the
 * remaining attempts return? Returns the winner's index when it is (so the
 * rest need not run), -1 when no text can reach min_votes any more (early
 * reject: e.g. three empty attempts of four), -2 when it is still open.
 * Exact by construction, pinned exhaustively by tests/test_recovery.c. */
#define MYNAH_ASR_RECOVER_OPEN (-2)
int mynah_asr_recover_vote_partial(const char *const *texts, int k, int n, int min_votes);

#endif
