/* Encoder FastConformer cache-aware (path offline-chunked; streaming in M1.3).
 * Blocco pre-norm macaron: ½FFN -> MHSA rel-pos -> Conv -> ½FFN -> LN out.
 * Riferimento numerico: tools/oracle/model.py + docs/nemotron-arch.md.
 * Tutte le dimensioni derivate dalle shape dei tensori (config-driven). */
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
    const float *relk_w;   /* f32 sempre: usato con T=2L-1 grande a ogni chunk */
    const float *bias_u, *bias_v;
    const float *ln_conv_w, *ln_conv_b;
    mynah_asr_qmat pw1_w, pw2_w;
    const float *dw_w, *cnorm_w, *cnorm_b;
    /* bias dei linear/conv (use_bias true, es. parakeet-110m) — NULL se assenti */
    const float *ff1_b1, *ff1_b2, *ff2_b1, *ff2_b2;
    const float *q_b, *k_b, *v_b, *o_b;
    const float *pw1_b, *dw_b, *pw2_b;
    /* conv norm = batch_norm (Parakeet): affine per-canale foldata al load
     * (y = x*scale + shift); NULL => layer_norm con cnorm_w/b (Nemotron) */
    const float *cnorm_scale, *cnorm_shift;
    const float *ln_ff2_w, *ln_ff2_b;
    mynah_asr_qmat ff2_w1, ff2_w2;
    const float *ln_out_w, *ln_out_b;
} mynah_asr_enc_layer;

typedef struct {
    mynah_asr_subsampling ss;
    mynah_asr_enc_layer *layers;
    int n_layers, d_model, n_heads, d_head, ffn_dim, conv_k;
    int causal;            /* 1 = depthwise conv causale (Nemotron), 0 = 'same' (Parakeet) */
    float xscale;          /* scala input dei layer (sqrt(d_model) se xscaling, else 1) */
    float *bn_fold;        /* buffer scale+shift della BN foldata (NULL se layer_norm) */
    /* prompt e projector post-encoder, entrambi opzionali (NULL se assenti:
     * i modelli CTC puri non hanno joint => d_out = d_model, out = encoder out) */
    const float *prompt_l1_w, *prompt_l1_b, *prompt_l2_w, *prompt_l2_b;
    const float *encproj_w, *encproj_b;
    int num_prompts, prompt_inter, d_out;
} mynah_asr_encoder;

/* quantize != 0: INT8 per-riga sui grandi linear (FFN, attn q/k/v/o, pointwise
 * conv). Costruita al load dal f32; ~2.4x meno memoria residente. */
int mynah_asr_encoder_init(mynah_asr_encoder *enc, const mynah_asr_safetensors *st, int quantize);
void mynah_asr_encoder_free(mynah_asr_encoder *enc);

/* Positional embedding rel [2T-1, d_model] (interleaved sin/cos, pos T-1..-(T-1)).
 * Buffer allocato dal caller: (2T-1)*d_model float. */
void mynah_asr_pos_emb(const mynah_asr_encoder *enc, int T, float *pe);

/* Un blocco conformer in-place su x [T, d_model]. left/right = att context;
 * left < 0 = attention full (modelli offline, att_context [-1,-1]). */
int mynah_asr_encoder_layer(const mynah_asr_encoder *enc, int li, float *x, int T,
                        const float *pe, int left_ctx, int right_ctx);

/* [Prompt one-hot + prompt_projector se presente +] encoder_projector:
 * x [T,d_model] -> out [T,d_out]. prompt_id ignorato se il modello non ha prompt. */
void mynah_asr_encoder_post(const mynah_asr_encoder *enc, const float *x, int T, int prompt_id,
                        float *out);

/* Forward completo offline: feats [T_mel, n_mels] validi -> out [T_enc, d_out] (malloc). */
float *mynah_asr_encoder_forward(const mynah_asr_encoder *enc, const float *feats, int t_mel,
                             int n_mels, int prompt_id, int left_ctx, int right_ctx,
                             int *t_out);

/* Come sopra ma SENZA prompt/projector: encoder out grezzo [T_enc, d_model]
 * (input della head CTC dei modelli hybrid). */
float *mynah_asr_encoder_forward_raw(const mynah_asr_encoder *enc, const float *feats, int t_mel,
                                 int n_mels, int left_ctx, int right_ctx, int *t_out);

/* Forward batched weight-stationary (lunghezze variabili, packing senza padding):
 * le GEMM per-frame (FFN, proiezioni — >95% dei FLOP) girano su [ΣT, d] leggendo i
 * pesi UNA volta; attention e conv (per-sequenza) iterano sui segmenti.
 * outs[b] riceve un buffer malloc [t_outs[b], d_out] (caller free). 0 = ok. */
int mynah_asr_encoder_forward_batch(const mynah_asr_encoder *enc, const float *const *feats,
                                const int *t_mel, int batch, int n_mels,
                                const int *prompt_ids, int left_ctx, int right_ctx,
                                float **outs, int *t_outs);

/* --------------------------------------------------------- streaming cache-aware
 * Ogni chunk mel (primo: 1+8r frame, poi 8(r+1)) produce q = r+1 frame encoder,
 * che coincidono con UN chunk della griglia chunked_limited: il left context in
 * cache (56 frame, sempre divisibile per r+1) è ESATTAMENTE il contesto ammesso
 * => attention piena su [cache valida + chunk], niente mask. Vedi prior-art §A. */
typedef struct {
    const mynah_asr_encoder *enc;
    mynah_asr_ss_stream ss;
    float *k_cache, *v_cache;   /* [n_layers, left, d_model] */
    float *conv_cache;          /* [n_layers, conv_k-1, d_model] */
    int left, right, q;         /* q = right+1 frame encoder per chunk */
    int cache_valid;            /* frame validi nella cache K/V (0..left) */
    /* scratch del percorso caldo, UN malloc all'init (zero malloc per chunk):
     * puntatori ritagliati da scr. Dimensionati su Qmax = q+2, Kmax = left+Qmax. */
    float *scr;
    float *sx, *stmp, *stmp2, *sxn, *skn;                    /* step */
    int sa_pe_K;                /* K dell'ultima pos-emb calcolata (0 = mai) */
    float *sa_pe, *sa_q, *sa_keys, *sa_rk, *sa_sc, *sa_bd,   /* attention */
          *sa_qb, *sa_ctx;
    float *sc_h2, *sc_gp, *sc_c, *sc_t;                      /* conv module */
} mynah_asr_enc_stream;

int mynah_asr_enc_stream_init(mynah_asr_enc_stream *es, const mynah_asr_encoder *enc,
                          int left_ctx, int right_ctx, int n_mels);
void mynah_asr_enc_stream_free(mynah_asr_enc_stream *es);

/* Frame mel richiesti dal prossimo chunk (primo: 1+8r, poi 8(r+1)). */
int mynah_asr_enc_stream_need(const mynah_asr_enc_stream *es);

/* mel chunk [n_mel, n_mels] esatto -> out [q, d_out] (buffer caller >= q*d_out).
 * Ritorna il numero di frame encoder prodotti, -1 su errore. */
int mynah_asr_enc_stream_step(mynah_asr_enc_stream *es, const float *mel, int n_mel,
                          int n_mels, int prompt_id, int is_last, float *out);

#endif
