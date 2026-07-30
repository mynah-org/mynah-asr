/* Cache-aware FastConformer encoder (offline-chunked path; streaming in M1.3).
 * Pre-norm macaron block: ½FFN -> rel-pos MHSA -> Conv -> ½FFN -> output LN.
 * Numeric reference: tools/oracle/model.py + docs/nemotron-arch.md.
 * Every dimension is derived from the tensor shapes (config-driven). */
#ifndef MYNAH_ASR_ENCODER_H
#define MYNAH_ASR_ENCODER_H

#include "qmat.h"
#include "subsampling.h"
#include "weights.h"

typedef struct {
    const float *ln_ff1_w, *ln_ff1_b;
    mynah_asr_qmat ff1_w1, ff1_w2;
    const float *ln_att_w, *ln_att_b;
    mynah_asr_qmat q_w, k_w, v_w, o_w;
    const float *relk_w;   /* always f32: used with a large T=2L-1 on every chunk */
    const float *bias_u, *bias_v;
    const float *ln_conv_w, *ln_conv_b;
    mynah_asr_qmat pw1_w, pw2_w;
    const float *dw_w, *cnorm_w, *cnorm_b;
    /* linear/conv biases (use_bias true, e.g. parakeet-110m) — NULL when absent */
    const float *ff1_b1, *ff1_b2, *ff2_b1, *ff2_b2;
    const float *q_b, *k_b, *v_b, *o_b;
    const float *pw1_b, *dw_b, *pw2_b;
    /* conv norm = batch_norm (Parakeet): per-channel affine folded at load
     * (y = x*scale + shift); NULL => layer_norm with cnorm_w/b (Nemotron) */
    const float *cnorm_scale, *cnorm_shift;
    const float *ln_ff2_w, *ln_ff2_b;
    mynah_asr_qmat ff2_w1, ff2_w2;
    const float *ln_out_w, *ln_out_b;
} mynah_asr_enc_layer;

typedef struct {
    mynah_asr_subsampling ss;
    mynah_asr_enc_layer *layers;
    int n_layers, d_model, n_heads, d_head, ffn_dim, conv_k;
    int causal;            /* 1 = causal depthwise conv (Nemotron), 0 = 'same' (Parakeet) */
    float xscale;          /* layer input scale (sqrt(d_model) when xscaling, else 1) */
    float *bn_fold;        /* scale+shift buffer of the folded BN (NULL when layer_norm) */
    /* prompt and post-encoder projector, both optional (NULL when absent:
     * pure CTC models have no joint => d_out = d_model, out = encoder out) */
    const float *prompt_l1_w, *prompt_l1_b, *prompt_l2_w, *prompt_l2_b;
    const float *encproj_w, *encproj_b;
    int num_prompts, prompt_inter, d_out;
} mynah_asr_encoder;

/* quantize != 0: per-row INT8 on the large linears (FFN, attn q/k/v/o, pointwise
 * conv). Built at load time from the f32 weights; ~2.4x less resident memory. */
int mynah_asr_encoder_init(mynah_asr_encoder *enc, const mynah_asr_safetensors *st, int quantize);
void mynah_asr_encoder_free(mynah_asr_encoder *enc);

/* Relative positional embedding [2T-1, d_model] (interleaved sin/cos, pos T-1..-(T-1)).
 * Buffer allocated by the caller: (2T-1)*d_model floats. */
void mynah_asr_pos_emb(const mynah_asr_encoder *enc, int T, float *pe);

/* One conformer block, in-place on x [T, d_model]. left/right = att context;
 * left < 0 = full attention (offline models, att_context [-1,-1]). */
int mynah_asr_encoder_layer(const mynah_asr_encoder *enc, int li, float *x, int T,
                        const float *pe, int left_ctx, int right_ctx);

/* [Prompt one-hot + prompt_projector when present +] encoder_projector:
 * x [T,d_model] -> out [T,d_out]. prompt_id ignored when the model has no prompt. */
void mynah_asr_encoder_post(const mynah_asr_encoder *enc, const float *x, int T, int prompt_id,
                        float *out);

/* Full offline forward: valid feats [T_mel, n_mels] -> out [T_enc, d_out] (malloc'd). */
float *mynah_asr_encoder_forward(const mynah_asr_encoder *enc, const float *feats, int t_mel,
                             int n_mels, int prompt_id, int left_ctx, int right_ctx,
                             int *t_out);

/* Same as above but WITHOUT prompt/projector: raw encoder output [T_enc, d_model]
 * (the input of the CTC head in hybrid models). */
float *mynah_asr_encoder_forward_raw(const mynah_asr_encoder *enc, const float *feats, int t_mel,
                                 int n_mels, int left_ctx, int right_ctx, int *t_out);

/* Weight-stationary batched forward (variable lengths, padding-free packing):
 * the per-frame GEMMs (FFN, projections — >95% of the FLOPs) run over [ΣT, d]
 * reading the weights ONCE; attention and conv (per-sequence) iterate over the
 * segments.
 * outs[b] receives a malloc'd buffer [t_outs[b], d_out] (caller frees). 0 = ok. */
int mynah_asr_encoder_forward_batch(const mynah_asr_encoder *enc, const float *const *feats,
                                const int *t_mel, int batch, int n_mels,
                                const int *prompt_ids, int left_ctx, int right_ctx,
                                float **outs, int *t_outs);

/* --------------------------------------------------------- streaming cache-aware
 * Every mel chunk (first: 1+8r frames, then 8(r+1)) produces q = r+1 encoder
 * frames, which coincide with ONE chunk of the chunked_limited grid: the left
 * context in the cache (56 frames, always divisible by r+1) is EXACTLY the
 * allowed context
 * => full attention over [valid cache + chunk], no mask. See prior art §A. */
typedef struct {
    const mynah_asr_encoder *enc;
    mynah_asr_ss_stream ss;
    float *k_cache, *v_cache;   /* [n_layers, left, d_model] */
    float *conv_cache;          /* [n_layers, conv_k-1, d_model] */
    int left, right, q;         /* q = right+1 encoder frames per chunk */
    int cache_valid;            /* valid frames in the K/V cache (0..left) */
    /* hot-path scratch, ONE malloc at init (zero mallocs per chunk):
     * pointers carved out of scr. Sized for Qmax = q+2, Kmax = left+Qmax. */
    float *scr;
    float *sx, *stmp, *stmp2, *sxn, *skn;                    /* step */
    int sa_pe_K;                /* K of the last computed pos-emb (0 = never) */
    float *sa_pe, *sa_q, *sa_keys, *sa_rk, *sa_sc, *sa_bd,   /* attention */
          *sa_qb, *sa_ctx;
    float *sc_h2, *sc_gp, *sc_c, *sc_t;                      /* conv module */
} mynah_asr_enc_stream;

int mynah_asr_enc_stream_init(mynah_asr_enc_stream *es, const mynah_asr_encoder *enc,
                          int left_ctx, int right_ctx, int n_mels);
void mynah_asr_enc_stream_free(mynah_asr_enc_stream *es);

/* Mel frames required by the next chunk (first: 1+8r, then 8(r+1)). */
int mynah_asr_enc_stream_need(const mynah_asr_enc_stream *es);

/* Exact mel chunk [n_mel, n_mels] -> out [q, d_out] (caller buffer >= q*d_out).
 * Returns the number of encoder frames produced, -1 on error. */
int mynah_asr_enc_stream_step(mynah_asr_enc_stream *es, const float *mel, int n_mel,
                          int n_mels, int prompt_id, int is_last, float *out);

#endif
