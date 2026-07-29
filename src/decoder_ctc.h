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
    int vocab, d_in;       /* V include il blank (ultimo indice) */
} mynah_asr_ctc;

/* 0 = head found, -1 = absent (not an error: a model without CTC). */
int mynah_asr_ctc_init(mynah_asr_ctc *c, const mynah_asr_safetensors *st);

/* enc_out [T, d_in] -> collapsed tokens in tokens[] (capacity cap); when
 * frames != NULL it writes the frame of each token's first argmax (timestamps).
 * Returns the token count. */
int mynah_asr_ctc_decode(const mynah_asr_ctc *c, const float *enc_out, int T,
                     int *tokens, int *frames, int cap);

#endif
