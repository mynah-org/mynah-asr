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

/* Same, with caller-owned scratch (NULL = allocate internally, i.e. exactly
 * mynah_asr_encoder_post). Floats required: mynah_asr_encoder_post_scratch_floats. */
void mynah_asr_encoder_post_scratch(const mynah_asr_encoder *enc, const float *x, int T,
                                int prompt_id, float *out, float *scratch);
size_t mynah_asr_encoder_post_scratch_floats(const mynah_asr_encoder *enc, int T);

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
    float *ssilu;               /* SiLU exp buffer (>= Qmax * ffn_dim) */
    float *spost;               /* prompt + projector (encoder_post) */
} mynah_asr_enc_stream;

int mynah_asr_enc_stream_init(mynah_asr_enc_stream *es, const mynah_asr_encoder *enc,
                          int left_ctx, int right_ctx, int n_mels);
void mynah_asr_enc_stream_free(mynah_asr_enc_stream *es);
/* Back to the first-chunk state (empty caches), keeping every allocation. */
void mynah_asr_enc_stream_reset(mynah_asr_enc_stream *es);

/* Mel frames required by the next chunk (first: 1+8r, then 8(r+1)). */
int mynah_asr_enc_stream_need(const mynah_asr_enc_stream *es);

/* Exact mel chunk [n_mel, n_mels] -> out [q, d_out] (caller buffer >= q*d_out).
 * Returns the number of encoder frames produced, -1 on error. */
int mynah_asr_enc_stream_step(mynah_asr_enc_stream *es, const float *mel, int n_mel,
                          int n_mels, int prompt_id, int is_last, float *out);

/* ------------------------------------------------------ batched stream step (S1-4)
 * B streams whose chunk is complete, stacked as ONE [Σq_i, d] activation through
 * the 24 conformer layers: every per-row linear (FFN1, q/k/v/o, the two
 * pointwise convolutions, FFN2) reads the weights ONCE for all the rows, while
 * subsampling, the K/V cache, the relative-position attention and the conv cache
 * stay per stream (a loop over i).
 *
 * CONTRACT (ENGINEERING.md §9): out[i] is BYTE-IDENTICAL to what
 * mynah_asr_enc_stream_step would have written for that stream alone, and the
 * caches are left in exactly the same state. On the quantized path that is
 * guaranteed by construction (per-row integer accumulation, see
 * mynah_asr_qmat_mul_rows); on f32 it depends on the BLAS being row-stable in M,
 * which is a MEASURED property, not an assumption — see tests/test_stream_batch.
 *
 * Every stream must share the encoder AND the lookahead preset (same q): the
 * caller groups them. The scratch is carved once for (max_b, max_q) so a step
 * allocates nothing. One thread at a time per mynah_asr_enc_batch. */
typedef struct mynah_asr_enc_batch mynah_asr_enc_batch;

/* max_left = the largest left attention context a stream of this model uses:
 * it sizes the shared rel-pos projection buffer (S1-7), whose row count is
 * 2*(max_left + max_q + 2) - 1. */
mynah_asr_enc_batch *mynah_asr_enc_batch_new(const mynah_asr_encoder *enc, int max_b, int max_q,
                                         int max_left);
void mynah_asr_enc_batch_free(mynah_asr_enc_batch *bb);
/* Rows the scratch can hold (max_b * (max_q + 2)) — the caller checks before
 * splitting a large ready set into several passes. */
int mynah_asr_enc_batch_max_rows(const mynah_asr_enc_batch *bb);
int mynah_asr_enc_batch_max_b(const mynah_asr_enc_batch *bb);

/* mel[i] is the exact chunk of stream i (n_mel[i] frames, never a short tail:
 * is_last is always 0 here — a finalizing stream goes through the single step).
 * out[i] receives [q_out[i], d_out]. Returns 0, -1 on error. */
int mynah_asr_enc_stream_step_batch(mynah_asr_enc_batch *bb,
                                mynah_asr_enc_stream *const *ess, int B,
                                const float *const *mel, const int *n_mel, int n_mels,
                                const int *prompt_id, float *const *out, int *q_out);

/* f32 only: whether the batched path is allowed to stack the rows. cblas_sgemm
 * is not contractually row-stable in M, so the answer is a MEASUREMENT of this
 * build's BLAS (see the note in encoder.c); MYNAH_ASR_BATCH_F32=0|1 overrides it. */
int mynah_asr_enc_batch_f32_ok(void);

/* ------------------------------------------- rel-pos projection sharing (S1-7)
 * `rk = pe @ relk_wR` depends only on (layer, K) with K = cache_valid + q, so
 * every stream of a batched pass that is at the same K computes the SAME matrix.
 * The batched step computes it once per (layer, K-group) and lets the group read
 * it; a stream at a different K (a slot on its first chunks, cache_valid < left)
 * keeps its own. Bit-exact by construction: identical inputs, identical call.
 *
 * Which of the two actually happened is a counter, not a claim
 * (ENGINEERING.md §6): a run that silently stopped sharing shows SHARED = 0. */
enum {
    MYNAH_ASR_RELPOS_PRIVATE = 0, /* attention core that computed its own rk     */
    MYNAH_ASR_RELPOS_SHARED,      /* attention core that read a group's rk       */
    MYNAH_ASR_RELPOS_GROUP,       /* rk computed once for a group (layer x pass) */
    MYNAH_ASR_RELPOS__N
};
unsigned long long mynah_asr_enc_relpos_counter(int which);
void mynah_asr_enc_relpos_counters_reset(void);

#endif
