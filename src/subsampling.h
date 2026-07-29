/* Subsampling FastConformer dw_striding 8x (3 stadi stride-2):
 * conv_in piena + ReLU, poi 2x (depthwise + pointwise + ReLU), flatten
 * channel-major, linear -> d_model. Padding causale (2,1) per Nemotron streaming
 * o simmetrico (1,1) per i modelli offline (Parakeet).
 * Vedi docs/nemotron-arch.md e docs/parakeet-tdt-arch.md. */
#ifndef MYNAH_ASR_SUBSAMPLING_H
#define MYNAH_ASR_SUBSAMPLING_H

#include "weights.h"

typedef struct {
    /* puntatori risolti UNA volta al load (decisione prior-art: niente lookup nel forward) */
    const mynah_asr_tensor *conv_in_w, *conv_in_b;         /* [C,1,3,3], [C] */
    const mynah_asr_tensor *dw_w[2], *dw_b[2];             /* [C,1,3,3], [C] */
    const mynah_asr_tensor *pw_w[2], *pw_b[2];             /* [C,C,1,1], [C] */
    const mynah_asr_tensor *lin_w, *lin_b;                 /* [d_model, C*F'], [d_model] */
    int channels;                                      /* 256 */
    int d_model;                                       /* 1024 */
    int causal;                                        /* 1 = pad (2,1), 0 = pad (1,1) */
} mynah_asr_subsampling;

/* Risolve i tensori dal safetensors. Supporta entrambi i naming HF:
 * Nemotron (conv_in/layers.{i}.depthwise_conv...) e Parakeet (layers.{0,2,3,5,6}).
 * causal di default dal naming (Nemotron 1, Parakeet 0); il config puo' sovrascrivere. */
int mynah_asr_subsampling_init(mynah_asr_subsampling *ss, const mynah_asr_safetensors *st);

/* feats [T, n_mels] float32 (solo frame validi) -> out [T_out, d_model] (malloc).
 * Scrive *t_out. NULL su errore. Padding tempo/freq secondo ss->causal. */
float *mynah_asr_subsampling_forward(const mynah_asr_subsampling *ss, const float *feats,
                                 int T, int n_mels, int *t_out);

/* ------------------------------------------------------- streaming (cache-aware)
 * Cache per stadio: l'ultimo time-frame dell'input (left_pad = k - stride = 1);
 * al primo chunk si antepone 1 zero extra (init_pad) => left effettivo 2, come
 * l'offline. Vedi docs/nemotron-arch.md (reference HF CausalConv2dCacheLayer). */
typedef struct {
    float *cache[3];        /* [C_in, F] ultimo frame input dello stadio */
    int cin[3], fdim[3];
    int first;
} mynah_asr_ss_stream;

int mynah_asr_ss_stream_init(mynah_asr_ss_stream *sst, const mynah_asr_subsampling *ss, int n_mels);
void mynah_asr_ss_stream_free(mynah_asr_ss_stream *sst);

/* mel chunk [n_mel, n_mels] (size ESATTA: primo = 1+8r, poi 8(r+1)) ->
 * out [q, d_model] con q = r+1 (buffer del caller). Ritorna q, -1 su errore. */
int mynah_asr_ss_stream_step(const mynah_asr_subsampling *ss, mynah_asr_ss_stream *sst,
                         const float *mel, int n_mel, int n_mels, int is_last, float *out);

#endif
