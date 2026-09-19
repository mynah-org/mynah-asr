/* Greedy CTC decoder: per-frame linear head over the ENCODER OUTPUT
 * (pre-projector), argmax + collapse of repeats + blank removal (= last index).
 * Serves the auxiliary head of hybrid models (parakeet-tdt_ctc-*) and, later,
 * the pure CTC Parakeets. See docs/parakeet-tdt-arch.md and
 * tools/oracle/model.py. */
#ifndef MYNAH_ASR_DECODER_CTC_H
#define MYNAH_ASR_DECODER_CTC_H

#include "weights.h"

typedef struct {
    const float *w, *b;    /* ctc_head [V, d, 1] + [V]; NULL when the model has no CTC */
    int vocab, d_in;       /* V includes the blank */
    int blank;             /* from decoder.blank_id; V-1 when the pack omits it */
} mynah_asr_ctc;

/* blank_id: the pack's `decoder.blank_id`, or < 0 to keep the historical
 * assumption that the blank is the LAST index.  It is true of every NeMo CTC
 * pack shipped so far and it is still only an assumption, which is why the
 * value now comes from the caller: the auxiliary head of a hybrid model has no
 * declared blank of its own (the pack's blank_id belongs to the RNNT/TDT
 * decoder) and passes < 0, while a pure-CTC pack passes what it declares.
 *
 * 0 = head found, -1 = absent (not an error: a model without CTC). */
int mynah_asr_ctc_init(mynah_asr_ctc *c, const mynah_asr_safetensors *st, int blank_id);

/* enc_out [T, d_in] -> raw head scores [T, vocab] (row-major, caller-allocated).
 * No softmax: for both argmax and Viterbi alignment the per-frame normalizer is
 * the same for every candidate and cancels (see src/align.c). 0 = ok. */
int mynah_asr_ctc_scores(const mynah_asr_ctc *c, const float *enc_out, int T, float *out);

/* enc_out [T, d_in] -> collapsed tokens in tokens[] (capacity cap); when
 * frames != NULL it writes the frame of each token's first argmax (timestamps).
 * Returns the token count. */
int mynah_asr_ctc_decode(const mynah_asr_ctc *c, const float *enc_out, int T,
                     int *tokens, int *frames, int cap);

#endif
