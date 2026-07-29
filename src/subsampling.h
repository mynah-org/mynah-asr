/* FastConformer dw_striding 8x subsampling (3 stride-2 stages):
 * full conv_in + ReLU, then 2x (depthwise + pointwise + ReLU), channel-major
 * flatten, linear -> d_model. Causal padding (2,1) for Nemotron streaming or
 * symmetric (1,1) for the offline models (Parakeet).
 * See docs/nemotron-arch.md and docs/parakeet-tdt-arch.md. */
#ifndef MYNAH_ASR_SUBSAMPLING_H
#define MYNAH_ASR_SUBSAMPLING_H

#include "weights.h"

typedef struct {
    /* pointers resolved ONCE at load (prior-art decision: no lookups in the forward) */
    const mynah_asr_tensor *conv_in_w, *conv_in_b;         /* [C,1,3,3], [C] */
    const mynah_asr_tensor *dw_w[2], *dw_b[2];             /* [C,1,3,3], [C] */
    const mynah_asr_tensor *pw_w[2], *pw_b[2];             /* [C,C,1,1], [C] */
    const mynah_asr_tensor *lin_w, *lin_b;                 /* [d_model, C*F'], [d_model] */
    int channels;                                      /* 256 */
    int d_model;                                       /* 1024 */
    int causal;                                        /* 1 = pad (2,1), 0 = pad (1,1) */
} mynah_asr_subsampling;

/* Resolve the tensors from the safetensors. Supports both HF namings:
 * Nemotron (conv_in/layers.{i}.depthwise_conv...) and Parakeet (layers.{0,2,3,5,6}).
 * causal defaults from the naming (Nemotron 1, Parakeet 0); the config can override. */
int mynah_asr_subsampling_init(mynah_asr_subsampling *ss, const mynah_asr_safetensors *st);

/* feats [T, n_mels] float32 (valid frames only) -> out [T_out, d_model] (malloc'd).
 * Writes *t_out. NULL on error. Time/freq padding according to ss->causal. */
float *mynah_asr_subsampling_forward(const mynah_asr_subsampling *ss, const float *feats,
                                 int T, int n_mels, int *t_out);

/* ------------------------------------------------------- streaming (cache-aware)
 * Per-stage cache: the last time-frame of the input (left_pad = k - stride = 1);
 * on the first chunk one extra zero is prepended (init_pad) => effective left 2,
 * like the offline path. See docs/nemotron-arch.md (HF CausalConv2dCacheLayer
 * reference). */
typedef struct {
    float *cache[3];        /* [C_in, F] ultimo frame input dello stadio */
    int cin[3], fdim[3];
    int first;
} mynah_asr_ss_stream;

int mynah_asr_ss_stream_init(mynah_asr_ss_stream *sst, const mynah_asr_subsampling *ss, int n_mels);
void mynah_asr_ss_stream_free(mynah_asr_ss_stream *sst);

/* mel chunk [n_mel, n_mels] (EXACT size: first = 1+8r, then 8(r+1)) ->
 * out [q, d_model] with q = r+1 (caller's buffer). Returns q, -1 on error. */
int mynah_asr_ss_stream_step(const mynah_asr_subsampling *ss, mynah_asr_ss_stream *sst,
                         const float *mel, int n_mel, int n_mels, int is_last, float *out);

#endif
