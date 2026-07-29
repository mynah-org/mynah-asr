/* Decoder CTC greedy: head lineare per-frame sull'ENCODER OUT (pre-projector),
 * argmax + collapse dei ripetuti + rimozione blank (= ultimo indice).
 * Serve la head ausiliaria dei modelli hybrid (parakeet-tdt_ctc-*) e, in futuro,
 * i Parakeet CTC puri. Vedi docs/parakeet-tdt-arch.md e tools/oracle/model.py. */
#ifndef MYNAH_ASR_DECODER_CTC_H
#define MYNAH_ASR_DECODER_CTC_H

#include "weights.h"

typedef struct {
    const float *w, *b;    /* ctc_head [V, d, 1] + [V]; NULL se il modello non ha CTC */
    int vocab, d_in;       /* V include il blank (ultimo indice) */
} mynah_asr_ctc;

/* 0 = head trovata, -1 = assente (non è un errore: modello senza CTC). */
int mynah_asr_ctc_init(mynah_asr_ctc *c, const mynah_asr_safetensors *st);

/* enc_out [T, d_in] -> token collassati in tokens[] (capienza cap); se
 * frames != NULL scrive il frame del primo argmax di ogni token (timestamp).
 * Ritorna il numero di token. */
int mynah_asr_ctc_decode(const mynah_asr_ctc *c, const float *enc_out, int T,
                     int *tokens, int *frames, int cap);

#endif
