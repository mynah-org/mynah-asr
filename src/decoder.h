/* Decoder RNNT/TDT: prediction network LSTM + joint, greedy decode.
 * Semantica NeMo/HF (docs/nemotron-arch.md, docs/parakeet-tdt-arch.md):
 * SOS = blank, lo stato LSTM avanza SOLO su emissione non-blank,
 * max_symbols_per_step limita le emissioni per frame.
 * TDT (n_durations > 0): la head emette [vocab | n_durations] logit; il frame
 * avanza della duration predetta a ogni step (blank con dur 0 -> forzata a 1).
 * Stato liftato e chunk-invariante: decodifica incrementale ≡ intera (prior-art). */
#ifndef MYNAH_ASR_DECODER_H
#define MYNAH_ASR_DECODER_H

#include "qmat.h"
#include "weights.h"

#define MYNAH_ASR_MAX_PRED_LAYERS 4
#define MYNAH_ASR_MAX_DURATIONS 16

typedef struct {
    const float *embedding;                 /* [vocab, H] */
    const float *w_ih[MYNAH_ASR_MAX_PRED_LAYERS], *w_hh[MYNAH_ASR_MAX_PRED_LAYERS];
    const float *b_ih[MYNAH_ASR_MAX_PRED_LAYERS], *b_hh[MYNAH_ASR_MAX_PRED_LAYERS];
    const float *proj_w, *proj_b;           /* decoder_projector [H, H] */
    mynah_asr_qmat head;                        /* joint.head [vocab(+dur), H] (int8 opz.) */
    const float *head_b;
    int vocab, hidden, n_layers, blank, max_symbols;
    /* TDT: vocab = righe della head; i token sono le prime vocab-n_durations */
    int n_durations;
    int durations[MYNAH_ASR_MAX_DURATIONS];
} mynah_asr_decoder;

typedef struct {
    float h[MYNAH_ASR_MAX_PRED_LAYERS][1024];   /* dimensione max; usa dec->hidden */
    float c[MYNAH_ASR_MAX_PRED_LAYERS][1024];
    float g[1024];                          /* output pred-net corrente (cache) */
    int last_token;                         /* -1 = non inizializzato */
    long t_abs;                             /* frame encoder assoluti già visti
                                               (per i timestamp nello streaming) */
} mynah_asr_dec_state;

/* durations: array TDT dal config (NULL/0 = RNNT puro). */
int mynah_asr_decoder_init(mynah_asr_decoder *dec, const mynah_asr_safetensors *st,
                       int blank, int max_symbols, int quantize,
                       const int *durations, int n_durations);

void mynah_asr_dec_state_reset(const mynah_asr_decoder *dec, mynah_asr_dec_state *s);

/* Greedy RNNT/TDT su enc [T, H]. Appende i token emessi a tokens[] (capienza cap)
 * e, se frames != NULL, il frame encoder ASSOLUTO di emissione di ciascun token
 * (base = somma dei T delle chiamate precedenti sullo stesso stato).
 * Ritorna il numero di token emessi. Lo stato è persistente tra chiamate
 * (streaming-ready): passare lo stesso stato per i chunk successivi. */
int mynah_asr_greedy_decode(const mynah_asr_decoder *dec, mynah_asr_dec_state *s,
                        const float *enc, int T, int *tokens, int *frames, int cap);

#endif
