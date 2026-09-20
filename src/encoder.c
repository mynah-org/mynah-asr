#include "encoder.h"

#include "backend.h"
#include "threads.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ helpers */
static const float *T_(const mynah_asr_safetensors *st, const char *fmt, int li, const char *suffix) {
    char name[160];
    snprintf(name, sizeof(name), fmt, li, suffix);
    const mynah_asr_tensor *t = mynah_asr_st_get(st, name);
    return t ? (const float *)t->data : NULL;
}

static void layer_norm_f(const float *x, const float *w, const float *b, float *out, int T, int d) {
    for (int t = 0; t < T; t++) {
        const float *row = x + (size_t)t * (size_t)d;
        float *o = out + (size_t)t * (size_t)d;
        double mu = 0.0, var = 0.0;
        for (int i = 0; i < d; i++) mu += row[i];
        mu /= d;
        for (int i = 0; i < d; i++) { double c = row[i] - mu; var += c * c; }
        var /= d;
        const float inv = (float)(1.0 / sqrt(var + 1e-5));
        for (int i = 0; i < d; i++) o[i] = ((row[i] - (float)mu) * inv) * w[i] + b[i];
    }
}

static void silu_inplace(float *x, size_t n) { mynah_asr_silu(x, n); }

/* x[T, n] += b (broadcast over rows); no-op when b == NULL (models without bias) */
static void add_bias_rows(float *x, const float *b, int T, int n) {
    if (!b) return;
    for (int t = 0; t < T; t++) {
        float *row = x + (size_t)t * (size_t)n;
        for (int i = 0; i < n; i++) row[i] += b[i];
    }
}

/* out[T,n] = x[T,k] @ W[n,k]^T (row-major, PyTorch linear layout) */
static void matmul_wt(const float *x, const float *w, float *out, int T, int n, int k) {
    mynah_asr_gemm_wt(x, w, out, T, n, k);
}

/* ------------------------------------------------------------------- init */
int mynah_asr_encoder_init(mynah_asr_encoder *enc, const mynah_asr_safetensors *st, int quantize) {
    memset(enc, 0, sizeof(*enc));
    if (mynah_asr_subsampling_init(&enc->ss, st) != 0) return -1;

    /* dimensions from the shapes (ffn_dim after the first qmat init: in the
     * pre-quantized file the f32 of linear1 does not exist) */
    const mynah_asr_tensor *bu = mynah_asr_st_get(st, "encoder.layers.0.self_attn.bias_u");
    const mynah_asr_tensor *dw = mynah_asr_st_get(st, "encoder.layers.0.conv.depthwise_conv.weight");
    const mynah_asr_tensor *ep = mynah_asr_st_get(st, "encoder_projector.weight");  /* optional (pure CTC) */
    const mynah_asr_tensor *p1 = mynah_asr_st_get(st, "prompt_projector.linear_1.weight"); /* optional */
    if (!bu || !dw) return -1;

    enc->d_model = enc->ss.d_model;
    enc->n_heads = (int)bu->shape[0];
    enc->d_head = (int)bu->shape[1];
    enc->conv_k = (int)dw->shape[2];
    enc->d_out = ep ? (int)ep->shape[0] : enc->d_model;
    enc->causal = enc->ss.causal;  /* default from the naming; the config can override */
    enc->xscale = 1.0f;            /* xscaling from the config (mynah_asr.c) */
    if (p1) {
        enc->prompt_inter = (int)p1->shape[0];
        enc->num_prompts = (int)p1->shape[1] - enc->d_model;
    }

    /* count the layers */
    int n = 0;
    while (T_(st, "encoder.layers.%d.%s", n, "norm_out.weight")) n++;
    enc->n_layers = n;
    enc->layers = calloc((size_t)n, sizeof(mynah_asr_enc_layer));
    if (!enc->layers || n == 0) return -1;

    const char *F = "encoder.layers.%d.%s";
    for (int li = 0; li < n; li++) {
        mynah_asr_enc_layer *L = &enc->layers[li];
        L->ln_ff1_w = T_(st, F, li, "norm_feed_forward1.weight");
        L->ln_ff1_b = T_(st, F, li, "norm_feed_forward1.bias");
        L->ln_att_w = T_(st, F, li, "norm_self_att.weight");
        L->ln_att_b = T_(st, F, li, "norm_self_att.bias");
        L->relk_w = T_(st, F, li, "self_attn.relative_k_proj.weight");
        L->bias_u = T_(st, F, li, "self_attn.bias_u");
        L->bias_v = T_(st, F, li, "self_attn.bias_v");
        L->ln_conv_w = T_(st, F, li, "norm_conv.weight");
        L->ln_conv_b = T_(st, F, li, "norm_conv.bias");
        L->dw_w = T_(st, F, li, "conv.depthwise_conv.weight");
        L->cnorm_w = T_(st, F, li, "conv.norm.weight");
        L->cnorm_b = T_(st, F, li, "conv.norm.bias");
        L->ln_ff2_w = T_(st, F, li, "norm_feed_forward2.weight");
        L->ln_ff2_b = T_(st, F, li, "norm_feed_forward2.bias");
        L->ln_out_w = T_(st, F, li, "norm_out.weight");
        L->ln_out_b = T_(st, F, li, "norm_out.bias");
        /* optional biases (use_bias true, e.g. parakeet-110m): NULL when absent */
        L->ff1_b1 = T_(st, F, li, "feed_forward1.linear1.bias");
        L->ff1_b2 = T_(st, F, li, "feed_forward1.linear2.bias");
        L->ff2_b1 = T_(st, F, li, "feed_forward2.linear1.bias");
        L->ff2_b2 = T_(st, F, li, "feed_forward2.linear2.bias");
        L->q_b = T_(st, F, li, "self_attn.q_proj.bias");
        L->k_b = T_(st, F, li, "self_attn.k_proj.bias");
        L->v_b = T_(st, F, li, "self_attn.v_proj.bias");
        L->o_b = T_(st, F, li, "self_attn.o_proj.bias");
        L->pw1_b = T_(st, F, li, "conv.pointwise_conv1.bias");
        L->dw_b = T_(st, F, li, "conv.depthwise_conv.bias");
        L->pw2_b = T_(st, F, li, "conv.pointwise_conv2.bias");
        if (!L->ln_ff1_w || !L->relk_w || !L->cnorm_w || !L->ln_out_w) {
            fprintf(stderr, "encoder: missing tensors at layer %d\n", li);
            return -1;
        }
        /* large linears: qmat — looks for the pre-quantized form (.q8/.q4) first */
        int rc = 0;
        char qn[160];
        #define QM(field, suffix) \
            (snprintf(qn, sizeof(qn), "encoder.layers.%d." suffix, li), \
             mynah_asr_qmat_init_st(&L->field, st, qn, quantize))
        rc |= QM(ff1_w1, "feed_forward1.linear1.weight");
        rc |= QM(ff1_w2, "feed_forward1.linear2.weight");
        rc |= QM(ff2_w1, "feed_forward2.linear1.weight");
        rc |= QM(ff2_w2, "feed_forward2.linear2.weight");
        rc |= QM(q_w, "self_attn.q_proj.weight");
        rc |= QM(k_w, "self_attn.k_proj.weight");
        rc |= QM(v_w, "self_attn.v_proj.weight");
        rc |= QM(o_w, "self_attn.o_proj.weight");
        rc |= QM(pw1_w, "conv.pointwise_conv1.weight");
        rc |= QM(pw2_w, "conv.pointwise_conv2.weight");
        #undef QM
        if (rc != 0) {
            fprintf(stderr, "encoder: qmat init failed at layer %d\n", li);
            return -1;
        }
    }

    enc->ffn_dim = enc->layers[0].ff1_w1.n;

    /* conv norm = BatchNorm (Parakeet): fold the running stats into a per-channel
     * scale+shift (inference: y = (x-mu)/sqrt(var+eps)*gamma+beta, eps 1e-5) */
    if (mynah_asr_st_get(st, "encoder.layers.0.conv.norm.running_mean")) {
        const int d = enc->d_model;
        enc->bn_fold = malloc((size_t)n * 2u * (size_t)d * sizeof(float));
        if (!enc->bn_fold) return -1;
        for (int li = 0; li < n; li++) {
            mynah_asr_enc_layer *L = &enc->layers[li];
            const float *mu = T_(st, F, li, "conv.norm.running_mean");
            const float *var = T_(st, F, li, "conv.norm.running_var");
            if (!mu || !var) {
                fprintf(stderr, "encoder: missing BN running stats at layer %d\n", li);
                return -1;
            }
            float *scale = enc->bn_fold + (size_t)li * 2u * (size_t)d;
            float *shift = scale + d;
            for (int i = 0; i < d; i++) {
                scale[i] = (float)((double)L->cnorm_w[i] / sqrt((double)var[i] + 1e-5));
                shift[i] = L->cnorm_b[i] - mu[i] * scale[i];
            }
            L->cnorm_scale = scale;
            L->cnorm_shift = shift;
        }
    }

    if (p1) {
        enc->prompt_l1_w = (const float *)p1->data;
        enc->prompt_l1_b = (const float *)mynah_asr_st_get(st, "prompt_projector.linear_1.bias")->data;
        enc->prompt_l2_w = (const float *)mynah_asr_st_get(st, "prompt_projector.linear_2.weight")->data;
        enc->prompt_l2_b = (const float *)mynah_asr_st_get(st, "prompt_projector.linear_2.bias")->data;
    }
    if (ep) {
        enc->encproj_w = (const float *)ep->data;
        enc->encproj_b = (const float *)mynah_asr_st_get(st, "encoder_projector.bias")->data;
    }
    return 0;
}

void mynah_asr_encoder_free(mynah_asr_encoder *enc) {
    for (int li = 0; li < enc->n_layers && enc->layers; li++) {
        mynah_asr_enc_layer *L = &enc->layers[li];
        mynah_asr_qmat_free(&L->ff1_w1); mynah_asr_qmat_free(&L->ff1_w2);
        mynah_asr_qmat_free(&L->ff2_w1); mynah_asr_qmat_free(&L->ff2_w2);
        mynah_asr_qmat_free(&L->q_w); mynah_asr_qmat_free(&L->k_w);
        mynah_asr_qmat_free(&L->v_w); mynah_asr_qmat_free(&L->o_w);
        mynah_asr_qmat_free(&L->pw1_w); mynah_asr_qmat_free(&L->pw2_w);
    }
    free(enc->layers);
    free(enc->bn_fold);
    enc->layers = NULL;
    enc->bn_fold = NULL;
}

/* --------------------------------------------------------------- pos emb */
void mynah_asr_pos_emb(const mynah_asr_encoder *enc, int T, float *pe) {
    const int d = enc->d_model, P = 2 * T - 1;
    for (int p = 0; p < P; p++) {
        const double pos = (double)(T - 1 - p);
        for (int j = 0; j < d / 2; j++) {
            const double freq = pos * pow(10000.0, -2.0 * j / (double)d);
            pe[(size_t)p * (size_t)d + 2 * (size_t)j] = (float)sin(freq);
            pe[(size_t)p * (size_t)d + 2 * (size_t)j + 1] = (float)cos(freq);
        }
    }
}

/* ------------------------------------------------------------- attention */
/* Windowed attention (left >= 0) in BLOCKS of rows: the ac/bd/ctx GEMMs touch
 * only the band of keys visible to the block instead of the whole T×T — the cost
 * goes back to linear in T. Found on the first A100 bench (300 s: cuda RTF 0.052
 * vs 0.028 at 60 s, +1.3 GB RAM): at T=3750 the full GEMMs wasted ~2 TFLOP and
 * ~170 MB/layer to use ~60 columns per row. Same math as the full path:
 * bd_shifted[t,j] = bd[t, T-1+j-t], softmax only over [j0,j1). */
static void attention_banded(const mynah_asr_encoder *enc, const mynah_asr_enc_layer *L,
                             float *q, float *k, float *v, const float *pe,
                             float *ctx, int T, int left, int right) {
    const int d = enc->d_model, H = enc->n_heads, dk = enc->d_head, P = 2 * T - 1;
    const float scaling = 1.0f / sqrtf((float)dk);
    const int chunk = right + 1, lc = left / chunk;
    const int B = 128 / chunk > 0 ? 128 / chunk : 1;   /* query chunks per block */
    const int Rb_max = B * chunk;
    const int Wb_max = (lc + B) * chunk;
    const int pW_max = Wb_max + Rb_max - 1;

    /* rk is only needed in the band |j-t| < window+block around p = T-1 */
    int pu0 = T - (lc + B + 1) * chunk, pu1 = T + (B + 1) * chunk;
    if (pu0 < 0) pu0 = 0;
    if (pu1 > P) pu1 = P;
    const int pu = pu1 - pu0;

    float *rk = malloc((size_t)pu * (size_t)d * sizeof(float));
    float *scb = malloc((size_t)Rb_max * (size_t)Wb_max * sizeof(float));
    float *bdb = malloc((size_t)Rb_max * (size_t)pW_max * sizeof(float));
    float *qb = malloc((size_t)Rb_max * (size_t)dk * sizeof(float));
    float *cb = malloc((size_t)Rb_max * (size_t)dk * sizeof(float));
    if (!rk || !scb || !bdb || !qb || !cb) { free(rk); free(scb); free(bdb); free(qb); free(cb); return; }
    matmul_wt(pe + (size_t)pu0 * (size_t)d, L->relk_w, rk, pu, d, d);

    const int n_chunks = (T + chunk - 1) / chunk;
    for (int h = 0; h < H; h++) {
        const size_t ho = (size_t)h * (size_t)dk;
        for (int tc0 = 0; tc0 < n_chunks; tc0 += B) {
            const int t0 = tc0 * chunk;
            int t1 = (tc0 + B) * chunk;
            if (t1 > T) t1 = T;
            const int Rb = t1 - t0;
            int c0 = (tc0 - lc) * chunk;
            if (c0 < 0) c0 = 0;
            int c1 = (tc0 + B) * chunk;
            if (c1 > T) c1 = T;
            const int Wb = c1 - c0;
            const int p0 = T - 1 + c0 - (t1 - 1);
            const int pW = Wb + Rb - 1;

            /* bd_block[r, pl] = (q[t0+r]+bias_v) . rk[p0+pl] */
            for (int r = 0; r < Rb; r++)
                for (int i = 0; i < dk; i++)
                    qb[(size_t)r * (size_t)dk + (size_t)i] =
                        q[(size_t)(t0 + r) * (size_t)d + ho + (size_t)i] + L->bias_v[ho + (size_t)i];
            mynah_asr_gemm_f32(0, 1, Rb, pW, dk,
                               1.0f, qb, dk, rk + (size_t)(p0 - pu0) * (size_t)d + ho, d, 0.0f, bdb, pW);

            /* ac_block[r, jl] = (q[t0+r]+bias_u) . k[c0+jl] */
            for (int r = 0; r < Rb; r++)
                for (int i = 0; i < dk; i++)
                    qb[(size_t)r * (size_t)dk + (size_t)i] =
                        q[(size_t)(t0 + r) * (size_t)d + ho + (size_t)i] + L->bias_u[ho + (size_t)i];
            mynah_asr_gemm_f32(0, 1, Rb, Wb, dk,
                               1.0f, qb, dk, k + (size_t)c0 * (size_t)d + ho, d, 0.0f, scb, Wb);

            for (int r = 0; r < Rb; r++) {
                const int t = t0 + r;
                const int tc = t / chunk;
                int j0 = (tc - lc) * chunk;
                if (j0 < c0) j0 = c0;
                int j1 = (tc + 1) * chunk;
                if (j1 > c1) j1 = c1;
                float *srow = scb + (size_t)r * (size_t)Wb;
                const float *brow = bdb + (size_t)r * (size_t)pW;

                float maxv = -3.0e38f;
                for (int j = j0; j < j1; j++) {
                    const int jl = j - c0;
                    srow[jl] = (srow[jl] + brow[T - 1 + j - t - p0]) * scaling;
                    if (srow[jl] > maxv) maxv = srow[jl];
                }
                float sum = 0.0f;
                for (int j = j0; j < j1; j++) {
                    srow[j - c0] = expf(srow[j - c0] - maxv);
                    sum += srow[j - c0];
                }
                const float inv = 1.0f / sum;
                for (int jl = 0; jl < j0 - c0; jl++) srow[jl] = 0.0f;
                for (int j = j0; j < j1; j++) srow[j - c0] *= inv;
                for (int jl = j1 - c0; jl < Wb; jl++) srow[jl] = 0.0f;
            }

            /* ctx_block = scores_block @ v[c0:c1] (strided per head) */
            mynah_asr_gemm_f32(0, 0, Rb, dk, Wb,
                               1.0f, scb, Wb, v + (size_t)c0 * (size_t)d + ho, d, 0.0f, cb, dk);
            for (int r = 0; r < Rb; r++)
                memcpy(ctx + (size_t)(t0 + r) * (size_t)d + ho,
                       cb + (size_t)r * (size_t)dk, (size_t)dk * sizeof(float));
        }
    }
    free(rk); free(scb); free(bdb); free(qb); free(cb);
}

static void attention(const mynah_asr_encoder *enc, const mynah_asr_enc_layer *L, const float *x,
                      float *out, int T, const float *pe, int left, int right) {
    const int d = enc->d_model, H = enc->n_heads, dk = enc->d_head, P = 2 * T - 1;
    const float scaling = 1.0f / sqrtf((float)dk);
    const int chunk = left >= 0 ? right + 1 : 1, lc = left >= 0 ? left / chunk : 0;

    float *q = malloc(3 * (size_t)T * (size_t)d * sizeof(float));
    float *k = q + (size_t)T * (size_t)d;
    float *v = k + (size_t)T * (size_t)d;
    float *ctx0 = malloc((size_t)T * (size_t)d * sizeof(float));
    if (!q || !ctx0) { free(q); free(ctx0); return; }

    mynah_asr_qmat_qkv(&L->q_w, &L->k_w, &L->v_w, x, q, k, v, T);
    add_bias_rows(q, L->q_b, T, d);
    add_bias_rows(k, L->k_b, T, d);
    add_bias_rows(v, L->v_b, T, d);

    if (left >= 0) {
        attention_banded(enc, L, q, k, v, pe, ctx0, T, left, right);
        mynah_asr_qmat_mul(&L->o_w, ctx0, out, T);
        add_bias_rows(out, L->o_b, T, d);
        free(q); free(ctx0);
        return;
    }
    float *ctx = ctx0;

    float *rk = malloc((size_t)P * (size_t)d * sizeof(float));
    float *scores = malloc((size_t)T * (size_t)T * sizeof(float));
    float *bd = malloc((size_t)T * (size_t)P * sizeof(float));
    float *qb = malloc((size_t)T * (size_t)d * sizeof(float));
    if (!rk || !scores || !bd || !qb) { free(q); free(rk); free(scores); free(bd); free(qb); free(ctx); return; }
    matmul_wt(pe, L->relk_w, rk, P, d, d);

    for (int h = 0; h < H; h++) {
        const size_t ho = (size_t)h * (size_t)dk;
        /* q + bias_v (for matrix_bd): gemm over per-head strided views */
        for (int t = 0; t < T; t++)
            for (int i = 0; i < dk; i++)
                qb[(size_t)t * (size_t)dk + (size_t)i] = q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_v[ho + (size_t)i];
        /* bd_full[t, p] = qv[t] . rk[p]  — rk is strided per head: gemm with lda=d */
        mynah_asr_gemm_f32(0, 1, T, P, dk,
                           1.0f, qb, dk, rk + ho, d, 0.0f, bd, P);

        /* q + bias_u (for matrix_ac) */
        for (int t = 0; t < T; t++)
            for (int i = 0; i < dk; i++)
                qb[(size_t)t * (size_t)dk + (size_t)i] = q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_u[ho + (size_t)i];
        mynah_asr_gemm_f32(0, 1, T, T, dk,
                           1.0f, qb, dk, k + ho, d, 0.0f, scores, T);

        /* scores = (ac + rel_shift(bd)) * scaling + softmax, ONLY over the
         * chunked_limited window — which is contiguous per row: jc in [tc-lc, tc]
         * => j in [ (tc-lc)*chunk, (tc+1)*chunk ). No -inf (UB under -ffast-math),
         * no work on the masked entries (prior art §A: the window lives in the
         * loop bounds). left < 0: full attention (att_context [-1,-1], offline
         * models). */
        for (int t = 0; t < T; t++) {
            float *srow = scores + (size_t)t * (size_t)T;
            const float *brow = bd + (size_t)t * (size_t)P;
            int j0 = 0, j1 = T;
            if (left >= 0) {
                const int tc = t / chunk;
                j0 = (tc - lc) * chunk;
                if (j0 < 0) j0 = 0;
                j1 = (tc + 1) * chunk;
                if (j1 > T) j1 = T;
            }

            float maxv = -3.0e38f;
            for (int j = j0; j < j1; j++) {
                /* rel_shift in closed form: bd_shifted[t,j] = bd[t, T-1 + j - t] */
                srow[j] = (srow[j] + brow[T - 1 + j - t]) * scaling;
                if (srow[j] > maxv) maxv = srow[j];
            }
            float sum = 0.0f;
            for (int j = j0; j < j1; j++) {
                srow[j] = expf(srow[j] - maxv);
                sum += srow[j];
            }
            const float inv = 1.0f / sum;
            for (int j = 0; j < j0; j++) srow[j] = 0.0f;
            for (int j = j0; j < j1; j++) srow[j] *= inv;
            for (int j = j1; j < T; j++) srow[j] = 0.0f;
        }

        /* ctx_h = scores @ v_h (v is strided per head) */
        mynah_asr_gemm_f32(0, 0, T, dk, T,
                           1.0f, scores, T, v + ho, d, 0.0f, qb, dk);
        for (int t = 0; t < T; t++)
            memcpy(ctx + (size_t)t * (size_t)d + ho, qb + (size_t)t * (size_t)dk, (size_t)dk * sizeof(float));
    }

    mynah_asr_qmat_mul(&L->o_w, ctx, out, T);
    add_bias_rows(out, L->o_b, T, d);
    free(q); free(rk); free(scores); free(bd); free(qb); free(ctx);
}

/* ------------------------------------------------------------ conv module */
static void conv_module(const mynah_asr_encoder *enc, const mynah_asr_enc_layer *L, const float *x,
                        float *out, int T) {
    const int d = enc->d_model, k = enc->conv_k;
    float *h2 = malloc((size_t)T * 2u * (size_t)d * sizeof(float));
    float *g = malloc((size_t)T * (size_t)d * sizeof(float));
    float *c = malloc((size_t)T * (size_t)d * sizeof(float));
    if (!h2 || !g || !c) { free(h2); free(g); free(c); return; }

    /* pointwise_conv1 [2d, d, 1] as a linear, then GLU over the channels */
    mynah_asr_qmat_mul(&L->pw1_w, x, h2, T);
    add_bias_rows(h2, L->pw1_b, T, 2 * d);
    for (int t = 0; t < T; t++) {
        const float *a = h2 + (size_t)t * 2u * (size_t)d;
        const float *b = a + d;
        float *o = g + (size_t)t * (size_t)d;
        for (int i = 0; i < d; i++) o[i] = a[i] * mynah_asr_sigmoid(b[i]);
    }

    /* depthwise k — dw_w [d, 1, k]: causal (left pad k-1) or symmetric 'same'
     * (pad (k-1)/2 per side); out-of-range reads are zero via the loop bounds */
    const int pc = enc->causal ? k - 1 : (k - 1) / 2;
    for (int t = 0; t < T; t++) {
        float *o = c + (size_t)t * (size_t)d;
        if (L->dw_b) memcpy(o, L->dw_b, (size_t)d * sizeof(float));
        else memset(o, 0, (size_t)d * sizeof(float));
        int j0 = (t - pc < 0) ? (pc - t) : 0;
        int j1 = (t - pc + k > T) ? (T - t + pc) : k;
        for (int j = j0; j < j1; j++) {
            const float *src = g + (size_t)(t - pc + j) * (size_t)d;
            for (int i = 0; i < d; i++) o[i] += L->dw_w[(size_t)i * (size_t)k + (size_t)j] * src[i];
        }
    }

    if (L->cnorm_scale) {
        /* folded BatchNorm: per-channel affine */
        for (int t = 0; t < T; t++) {
            const float *src = c + (size_t)t * (size_t)d;
            float *o = g + (size_t)t * (size_t)d;
            for (int i = 0; i < d; i++) o[i] = src[i] * L->cnorm_scale[i] + L->cnorm_shift[i];
        }
    } else {
        layer_norm_f(c, L->cnorm_w, L->cnorm_b, g, T, d);
    }
    silu_inplace(g, (size_t)T * (size_t)d);
    mynah_asr_qmat_mul(&L->pw2_w, g, out, T);
    add_bias_rows(out, L->pw2_b, T, d);
    free(h2); free(g); free(c);
}

/* Pre-norm ½FFN: fused (qmat_ffn) for models without bias, unfused with biases.
 * ln -> linear1 (+b) -> SiLU -> linear2 (+b) into tmp [T,d]; tmp2 scratch [T,ffn]. */
static void half_ffn(const mynah_asr_encoder *enc, const mynah_asr_enc_layer *L,
                     const float *x, float *tmp, float *tmp2, int T, int which2) {
    const mynah_asr_qmat *w1 = which2 ? &L->ff2_w1 : &L->ff1_w1;
    const mynah_asr_qmat *w2 = which2 ? &L->ff2_w2 : &L->ff1_w2;
    const float *b1 = which2 ? L->ff2_b1 : L->ff1_b1;
    const float *b2 = which2 ? L->ff2_b2 : L->ff1_b2;
    const float *lnw = which2 ? L->ln_ff2_w : L->ln_ff1_w;
    const float *lnb = which2 ? L->ln_ff2_b : L->ln_ff1_b;
    layer_norm_f(x, lnw, lnb, tmp, T, enc->d_model);
    if (!b1 && !b2) {
        mynah_asr_qmat_ffn(w1, w2, tmp, tmp, T, tmp2);
        return;
    }
    mynah_asr_qmat_mul(w1, tmp, tmp2, T);
    add_bias_rows(tmp2, b1, T, enc->ffn_dim);
    silu_inplace(tmp2, (size_t)T * (size_t)enc->ffn_dim);
    mynah_asr_qmat_mul(w2, tmp2, tmp, T);
    add_bias_rows(tmp, b2, T, enc->d_model);
}

/* ------------------------------------------------------------------ layer */
int mynah_asr_encoder_layer(const mynah_asr_encoder *enc, int li, float *x, int T,
                        const float *pe, int left_ctx, int right_ctx) {
    const mynah_asr_enc_layer *L = &enc->layers[li];
    const int d = enc->d_model;
    const size_t n = (size_t)T * (size_t)d;
    float *tmp = malloc(n * sizeof(float));
    float *tmp2 = malloc((size_t)T * (size_t)enc->ffn_dim * sizeof(float));
    if (!tmp || !tmp2) { free(tmp); free(tmp2); return -1; }

    /* ½ FFN1 (fused when bias-free: a single sync on Metal) */
    half_ffn(enc, L, x, tmp, tmp2, T, 0);
    for (size_t i = 0; i < n; i++) x[i] += 0.5f * tmp[i];

    /* MHSA */
    float *xn = malloc(n * sizeof(float));
    if (!xn) { free(tmp); free(tmp2); return -1; }
    layer_norm_f(x, L->ln_att_w, L->ln_att_b, xn, T, d);
    attention(enc, L, xn, tmp, T, pe, left_ctx, right_ctx);
    for (size_t i = 0; i < n; i++) x[i] += tmp[i];

    /* Conv */
    layer_norm_f(x, L->ln_conv_w, L->ln_conv_b, xn, T, d);
    conv_module(enc, L, xn, tmp, T);
    for (size_t i = 0; i < n; i++) x[i] += tmp[i];

    /* ½ FFN2 */
    half_ffn(enc, L, x, tmp, tmp2, T, 1);
    for (size_t i = 0; i < n; i++) x[i] += 0.5f * tmp[i];

    /* output LN */
    layer_norm_f(x, L->ln_out_w, L->ln_out_b, xn, T, d);
    memcpy(x, xn, n * sizeof(float));
    free(tmp); free(tmp2); free(xn);
    return 0;
}

/* Full layer stack over x [T,d]. On Metal (f32 weights, k=9) everything runs on
 * GPU with a single sync (v4); otherwise a per-layer CPU loop. */
static int run_layers(const mynah_asr_encoder *enc, float *x, int T, const float *pe,
                      int left_ctx, int right_ctx) {
#ifdef MYNAH_ASR_METAL
    /* The Metal kernel covers both semantics: Nemotron (causal conv, LN, chunked
     * window) and Parakeet (conv 'same', folded BN, full attention, optional
     * biases). Only quantized weights and k != 9 stay on CPU. */
    if (mynah_asr_backend() == MYNAH_ASR_BACKEND_METAL && enc->conv_k == 9 &&
        enc->layers[0].q_w.qtype == MYNAH_ASR_Q_F32) {
        mynah_asr_metal_layer_w *ws = malloc((size_t)enc->n_layers * sizeof(*ws));
        if (ws) {
            for (int li = 0; li < enc->n_layers; li++) {
                const mynah_asr_enc_layer *L = &enc->layers[li];
                ws[li] = (mynah_asr_metal_layer_w){
                    .ln_ff1_w = L->ln_ff1_w, .ln_ff1_b = L->ln_ff1_b,
                    .ff1_w1 = L->ff1_w1.f32, .ff1_w2 = L->ff1_w2.f32,
                    .ln_att_w = L->ln_att_w, .ln_att_b = L->ln_att_b,
                    .wq = L->q_w.f32, .wk = L->k_w.f32, .wv = L->v_w.f32,
                    .wo = L->o_w.f32, .relk = L->relk_w,
                    .bias_u = L->bias_u, .bias_v = L->bias_v,
                    .ln_conv_w = L->ln_conv_w, .ln_conv_b = L->ln_conv_b,
                    .pw1 = L->pw1_w.f32, .dw9 = L->dw_w,
                    .cnorm_w = L->cnorm_w, .cnorm_b = L->cnorm_b,
                    .pw2 = L->pw2_w.f32,
                    .ln_ff2_w = L->ln_ff2_w, .ln_ff2_b = L->ln_ff2_b,
                    .ff2_w1 = L->ff2_w1.f32, .ff2_w2 = L->ff2_w2.f32,
                    .ln_out_w = L->ln_out_w, .ln_out_b = L->ln_out_b,
                    .ff1_b1 = L->ff1_b1, .ff1_b2 = L->ff1_b2,
                    .ff2_b1 = L->ff2_b1, .ff2_b2 = L->ff2_b2,
                    .q_b = L->q_b, .k_b = L->k_b, .v_b = L->v_b, .o_b = L->o_b,
                    .pw1_b = L->pw1_b, .dw_b = L->dw_b, .pw2_b = L->pw2_b,
                    .cnorm_scale = L->cnorm_scale, .cnorm_shift = L->cnorm_shift,
                };
            }
            const int conv_pad = enc->causal ? enc->conv_k - 1 : (enc->conv_k - 1) / 2;
            const int rc = mynah_asr_metal_encoder_layers(ws, enc->n_layers, x, pe, T,
                                                      enc->d_model, enc->n_heads,
                                                      enc->ffn_dim, left_ctx, right_ctx,
                                                      conv_pad);
            free(ws);
            if (rc == 0) return 0;
        }
    }
#endif
    for (int li = 0; li < enc->n_layers; li++)
        if (mynah_asr_encoder_layer(enc, li, x, T, pe, left_ctx, right_ctx) != 0)
            return -1;
    return 0;
}

/* -------------------------------------------------- prompt + projector */
size_t mynah_asr_encoder_post_scratch_floats(const mynah_asr_encoder *enc, int T) {
    if (!enc->encproj_w || !enc->prompt_l1_w) return 0;
    return (size_t)T * (size_t)(enc->d_model + enc->num_prompts)     /* cat   */
         + (size_t)T * (size_t)enc->prompt_inter                     /* mid   */
         + (size_t)T * (size_t)enc->d_model;                         /* fused */
}

void mynah_asr_encoder_post_scratch(const mynah_asr_encoder *enc, const float *x, int T,
                                int prompt_id, float *out, float *scratch) {
    const int d = enc->d_model, np = enc->num_prompts, di = enc->prompt_inter;
    const int dcat = d + np;

    if (!enc->encproj_w) {
        /* pure CTC: no joint, out = encoder out */
        memcpy(out, x, (size_t)T * (size_t)d * sizeof(float));
        return;
    }
    if (!enc->prompt_l1_w) {
        /* model without prompt (Parakeet): encoder_projector only */
        matmul_wt(x, enc->encproj_w, out, T, enc->d_out, d);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < enc->d_out; i++)
                out[(size_t)t * (size_t)enc->d_out + (size_t)i] += enc->encproj_b[i];
        return;
    }
    /* caller scratch (>= mynah_asr_encoder_post_scratch_floats) keeps the
     * streaming step allocation-free; NULL = allocate here, as before */
    const int owned = scratch == NULL;
    float *cat, *mid, *fused;
    if (owned) {
        cat = calloc((size_t)T * (size_t)dcat, sizeof(float));
        mid = malloc((size_t)T * (size_t)di * sizeof(float));
        fused = malloc((size_t)T * (size_t)d * sizeof(float));
        if (!cat || !mid || !fused) { free(cat); free(mid); free(fused); return; }
    } else {
        cat = scratch;
        mid = cat + (size_t)T * (size_t)dcat;
        fused = mid + (size_t)T * (size_t)di;
        memset(cat, 0, (size_t)T * (size_t)dcat * sizeof(float)); /* = the calloc above */
    }

    for (int t = 0; t < T; t++) {
        memcpy(cat + (size_t)t * (size_t)dcat, x + (size_t)t * (size_t)d, (size_t)d * sizeof(float));
        cat[(size_t)t * (size_t)dcat + (size_t)d + (size_t)prompt_id] = 1.0f;
    }
    matmul_wt(cat, enc->prompt_l1_w, mid, T, di, dcat);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < di; i++) {
            float *v = &mid[(size_t)t * (size_t)di + (size_t)i];
            *v += enc->prompt_l1_b[i];
            if (*v < 0.0f) *v = 0.0f;
        }
    matmul_wt(mid, enc->prompt_l2_w, fused, T, d, di);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++) fused[(size_t)t * (size_t)d + (size_t)i] += enc->prompt_l2_b[i];

    matmul_wt(fused, enc->encproj_w, out, T, enc->d_out, d);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < enc->d_out; i++) out[(size_t)t * (size_t)enc->d_out + (size_t)i] += enc->encproj_b[i];
    if (owned) { free(cat); free(mid); free(fused); }
}

void mynah_asr_encoder_post(const mynah_asr_encoder *enc, const float *x, int T, int prompt_id,
                        float *out) {
    mynah_asr_encoder_post_scratch(enc, x, T, prompt_id, out, NULL);
}

/* --------------------------------------------------------------- streaming */

int mynah_asr_enc_stream_init(mynah_asr_enc_stream *es, const mynah_asr_encoder *enc,
                          int left_ctx, int right_ctx, int n_mels) {
    memset(es, 0, sizeof(*es));
    es->enc = enc;
    es->left = left_ctx;
    es->right = right_ctx;
    es->q = right_ctx + 1;
    /* the largest mel chunk a step can be fed (see mynah_asr_enc_stream_need:
     * 1 + sub*right on the first chunk, sub*(right+1) afterwards) */
    const int max_n_mel = enc->ss.sub_factor * (right_ctx + 1) + 1;
    if (mynah_asr_ss_stream_init(&es->ss, &enc->ss, n_mels, max_n_mel) != 0) return -1;
    const size_t kv = (size_t)enc->n_layers * (size_t)left_ctx * (size_t)enc->d_model;
    const size_t cv = (size_t)enc->n_layers * (size_t)(enc->conv_k - 1) * (size_t)enc->d_model;
    es->k_cache = calloc(kv, sizeof(float));
    es->v_cache = calloc(kv, sizeof(float));
    es->conv_cache = calloc(cv, sizeof(float));
    if (!es->k_cache || !es->v_cache || !es->conv_cache) return -1;

    /* single scratch for the hot path (max sizes, reused on every chunk) */
    const size_t d = (size_t)enc->d_model, ck = (size_t)enc->conv_k;
    const size_t Qm = (size_t)es->q + 2, Km = (size_t)left_ctx + Qm, Pm = 2 * Km - 1;
    /* SiLU scratch: the FFN intermediate (Qm*ffn_dim) and the conv module's
     * (Qm*d) share it, so it is sized for the larger of the two */
    const size_t silu = Qm * (size_t)(enc->ffn_dim > enc->d_model ? enc->ffn_dim : enc->d_model);
    const size_t sz = Qm * d                  /* sx  */
                    + Qm * d                  /* stmp */
                    + Qm * (size_t)enc->ffn_dim /* stmp2 */
                    + Qm * d                  /* sxn */
                    + 2 * Qm * d              /* skn */
                    + Pm * d                  /* sa_pe */
                    + Qm * d                  /* sa_q */
                    + 2 * Km * d              /* sa_keys */
                    + Pm * d                  /* sa_rk */
                    + Qm * Km                 /* sa_sc */
                    + Qm * Pm                 /* sa_bd */
                    + Qm * d                  /* sa_qb */
                    + Qm * d                  /* sa_ctx */
                    + Qm * 2 * d              /* sc_h2 */
                    + (Qm + ck - 1) * d       /* sc_gp */
                    + Qm * d                  /* sc_c */
                    + Qm * d                  /* sc_t */
                    + silu                    /* ssilu */
                    + mynah_asr_encoder_post_scratch_floats(enc, (int)Qm); /* spost */
    es->scr = malloc(sz * sizeof(float));
    if (!es->scr) return -1;
    float *p = es->scr;
    #define CARVE(f, n) es->f = p; p += (n)
    CARVE(sx, Qm * d);
    CARVE(stmp, Qm * d);
    CARVE(stmp2, Qm * (size_t)enc->ffn_dim);
    CARVE(sxn, Qm * d);
    CARVE(skn, 2 * Qm * d);
    CARVE(sa_pe, Pm * d);
    CARVE(sa_q, Qm * d);
    CARVE(sa_keys, 2 * Km * d);
    CARVE(sa_rk, Pm * d);
    CARVE(sa_sc, Qm * Km);
    CARVE(sa_bd, Qm * Pm);
    CARVE(sa_qb, Qm * d);
    CARVE(sa_ctx, Qm * d);
    CARVE(sc_h2, Qm * 2 * d);
    CARVE(sc_gp, (Qm + ck - 1) * d);
    CARVE(sc_c, Qm * d);
    CARVE(sc_t, Qm * d);
    CARVE(ssilu, silu);
    CARVE(spost, mynah_asr_encoder_post_scratch_floats(enc, (int)Qm));
    #undef CARVE
    return 0;
}

void mynah_asr_enc_stream_free(mynah_asr_enc_stream *es) {
    mynah_asr_ss_stream_free(&es->ss);
    free(es->k_cache); free(es->v_cache); free(es->conv_cache); free(es->scr);
    es->k_cache = es->v_cache = es->conv_cache = es->scr = NULL;
}

void mynah_asr_enc_stream_reset(mynah_asr_enc_stream *es) {
    const mynah_asr_encoder *enc = es->enc;
    const size_t kv = (size_t)enc->n_layers * (size_t)es->left * (size_t)enc->d_model;
    const size_t cv = (size_t)enc->n_layers * (size_t)(enc->conv_k - 1) * (size_t)enc->d_model;
    mynah_asr_ss_stream_reset(&es->ss);
    memset(es->k_cache, 0, kv * sizeof(float));
    memset(es->v_cache, 0, kv * sizeof(float));
    memset(es->conv_cache, 0, cv * sizeof(float));
    es->cache_valid = 0;
    es->sa_pe_K = 0;
}

int mynah_asr_enc_stream_need(const mynah_asr_enc_stream *es) {
    const int sub = es->enc->ss.sub_factor;
    return es->cache_valid == 0 && es->ss.first ? 1 + sub * es->right
                                                : sub * (es->right + 1);
}

/* ------------------------------------------- rel-pos projection counters (S1-7)
 * One relaxed atomic per attention core (24 per stream per step): negligible
 * next to the GEMMs, and it is the only way a run can PROVE that the sharing
 * below actually happened instead of quietly degrading (ENGINEERING.md §6). */
static _Atomic unsigned long long g_relpos[MYNAH_ASR_RELPOS__N];
static void relpos_count(int which) {
    atomic_fetch_add_explicit(&g_relpos[which], 1ull, memory_order_relaxed);
}
unsigned long long mynah_asr_enc_relpos_counter(int which) {
    if (which < 0 || which >= MYNAH_ASR_RELPOS__N) return 0;
    return atomic_load_explicit(&g_relpos[which], memory_order_relaxed);
}
void mynah_asr_enc_relpos_counters_reset(void) {
    for (int i = 0; i < MYNAH_ASR_RELPOS__N; i++)
        atomic_store_explicit(&g_relpos[i], 0ull, memory_order_relaxed);
}

/* Streaming attention, WITHOUT the q and o projections: they are plain per-row
 * linears, so the batched step hoists them out and runs them over every stream's
 * rows at once (S1-4). Everything left here is per stream: the K/V cache, the
 * relative-position projection and the per-head softmax.
 * q [Q,d] is the already-projected query; ctx [Q,d] receives the context, which
 * the caller multiplies by o_w.
 * pe [2K-1, d] is computed by the caller (once per chunk, not per layer);
 * scratch preallocated in es (zero mallocs in the hot path).
 *
 * rk_in (S1-7): rk = pe @ relk_w^T for THIS layer and THIS K, already computed
 * by the caller because several streams of the pass share the same K. NULL = the
 * core computes its own into es->sa_rk, which is what the single-stream path
 * always does. Passing a shared buffer cannot change a single float: pe is a
 * pure function of K, relk_w is the layer's, so the two matmul_wt calls have
 * bit-identical inputs and the same shape. */
static void stream_attention_core(mynah_asr_enc_stream *es, const mynah_asr_enc_layer *L,
                                  const float *q, const float *kn, const float *vn,
                                  const float *pe, float *ctx, int Q,
                                  const float *k_cache, const float *v_cache, int valid,
                                  const float *rk_in) {
    const mynah_asr_encoder *enc = es->enc;
    const int d = enc->d_model, H = enc->n_heads, dk = enc->d_head;
    const int K = valid + Q, P = 2 * K - 1;
    const float scaling = 1.0f / sqrtf((float)dk);

    const float *rk = rk_in ? rk_in : es->sa_rk;
    float *scores = es->sa_sc, *bd = es->sa_bd;
    float *qb = es->sa_qb;

    {
        float *kk = es->sa_keys, *vv = es->sa_keys + (size_t)K * (size_t)d;

        /* keys = valid cache ++ new ones (the caller updates the cache) */
        memcpy(kk, k_cache, (size_t)valid * (size_t)d * sizeof(float));
        memcpy(kk + (size_t)valid * (size_t)d, kn, (size_t)Q * (size_t)d * sizeof(float));
        memcpy(vv, v_cache, (size_t)valid * (size_t)d * sizeof(float));
        memcpy(vv + (size_t)valid * (size_t)d, vn, (size_t)Q * (size_t)d * sizeof(float));

        if (rk_in) {
            relpos_count(MYNAH_ASR_RELPOS_SHARED);
        } else {
            matmul_wt(pe, L->relk_w, es->sa_rk, P, d, d);
            relpos_count(MYNAH_ASR_RELPOS_PRIVATE);
        }

        for (int h = 0; h < H; h++) {
            const size_t ho = (size_t)h * (size_t)dk;
            for (int t = 0; t < Q; t++)
                for (int i = 0; i < dk; i++)
                    qb[(size_t)t * (size_t)dk + (size_t)i] =
                        q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_v[ho + (size_t)i];
            mynah_asr_gemm_f32(0, 1, Q, P, dk,
                               1.0f, qb, dk, rk + ho, d, 0.0f, bd, P);

            for (int t = 0; t < Q; t++)
                for (int i = 0; i < dk; i++)
                    qb[(size_t)t * (size_t)dk + (size_t)i] =
                        q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_u[ho + (size_t)i];
            mynah_asr_gemm_f32(0, 1, Q, K, dk,
                               1.0f, qb, dk, kk + ho, d, 0.0f, scores, K);

            for (int t = 0; t < Q; t++) {
                float *srow = scores + (size_t)t * (size_t)K;
                const float *brow = bd + (size_t)t * (size_t)P;
                /* rel_shift: p = (K-1) - (valid + t) + j, j in [0, K) */
                const int base = K - 1 - valid - t;
                float maxv = -3.0e38f;
                for (int j = 0; j < K; j++) {
                    srow[j] = (srow[j] + brow[base + j]) * scaling;
                    if (srow[j] > maxv) maxv = srow[j];
                }
                float sum = 0.0f;
                for (int j = 0; j < K; j++) { srow[j] = expf(srow[j] - maxv); sum += srow[j]; }
                const float inv = 1.0f / sum;
                for (int j = 0; j < K; j++) srow[j] *= inv;
            }

            mynah_asr_gemm_f32(0, 0, Q, dk, K,
                               1.0f, scores, K, vv + ho, d, 0.0f, qb, dk);
            for (int t = 0; t < Q; t++)
                memcpy(ctx + (size_t)t * (size_t)d + ho, qb + (size_t)t * (size_t)dk,
                       (size_t)dk * sizeof(float));
        }
    }
}

/* Single-stream attention: the two hoisted linears around the core, in the same
 * order as before the split. */
static void stream_attention(mynah_asr_enc_stream *es, const mynah_asr_enc_layer *L,
                             const float *x, const float *kn, const float *vn,
                             const float *pe, float *out, int Q,
                             const float *k_cache, const float *v_cache, int valid) {
    mynah_asr_qmat_mul(&L->q_w, x, es->sa_q, Q);
    /* NULL: the single path always computes its own rk — unchanged by S1-7 */
    stream_attention_core(es, L, es->sa_q, kn, vn, pe, es->sa_ctx, Q,
                          k_cache, v_cache, valid, NULL);
    mynah_asr_qmat_mul(&L->o_w, es->sa_ctx, out, Q);
}

/* update a layer's K/V cache with the new k/v rows of the chunk */
static void update_kv_cache(float *cache, const float *fresh, int valid, int Q,
                            int left, int d) {
    const int total = valid + Q;
    const int keep = total < left ? total : left;
    const int from_old = keep - Q > 0 ? keep - Q : 0;      /* old rows to keep */
    const int drop_old = valid - from_old;                  /* old rows to drop */
    if (from_old > 0 && drop_old > 0)
        memmove(cache, cache + (size_t)drop_old * (size_t)d,
                (size_t)from_old * (size_t)d * sizeof(float));
    const int fresh_keep = keep - from_old;                 /* new rows to keep (<= Q) */
    memcpy(cache + (size_t)from_old * (size_t)d,
           fresh + (size_t)(Q - fresh_keep) * (size_t)d,
           (size_t)fresh_keep * (size_t)d * sizeof(float));
}

/* Streaming conv module WITHOUT the two pointwise convolutions (they are per-row
 * linears the batched step hoists, S1-4): GLU over h2 [Q,2d], the cached causal
 * depthwise, layer_norm and SiLU. cache [k-1, d] is prepended and refreshed with
 * the last k-1 rows. mid [Q,d] receives what pointwise_conv2 consumes. */
static void stream_conv_mid(mynah_asr_enc_stream *es, const mynah_asr_enc_layer *L,
                            const float *h2, float *mid, int Q, float *cache) {
    const mynah_asr_encoder *enc = es->enc;
    const int d = enc->d_model, k = enc->conv_k;
    float *gp = es->sc_gp, *c = es->sc_c;

    memcpy(gp, cache, (size_t)(k - 1) * (size_t)d * sizeof(float));
    for (int t = 0; t < Q; t++) {
        const float *a = h2 + (size_t)t * 2u * (size_t)d;
        const float *b = a + d;
        float *o = gp + (size_t)(k - 1 + t) * (size_t)d;
        for (int i = 0; i < d; i++) o[i] = a[i] * mynah_asr_sigmoid(b[i]);
    }
    /* update cache = last k-1 rows of gp */
    memcpy(cache, gp + (size_t)Q * (size_t)d, (size_t)(k - 1) * (size_t)d * sizeof(float));

    for (int t = 0; t < Q; t++) {
        float *o = c + (size_t)t * (size_t)d;
        memset(o, 0, (size_t)d * sizeof(float));
        for (int j = 0; j < k; j++) {
            const float *src = gp + (size_t)(t + j) * (size_t)d;
            for (int i = 0; i < d; i++) o[i] += L->dw_w[(size_t)i * (size_t)k + (size_t)j] * src[i];
        }
    }
    layer_norm_f(c, L->cnorm_w, L->cnorm_b, mid, Q, d);
    mynah_asr_silu_scratch(mid, (size_t)Q * (size_t)d, es->ssilu);
}

/* Single-stream conv module: the two hoisted pointwise convolutions around the
 * core, in the same order as before the split. */
static void stream_conv_module(mynah_asr_enc_stream *es, const mynah_asr_enc_layer *L,
                               const float *x, float *out, int Q, float *cache) {
    mynah_asr_qmat_mul(&L->pw1_w, x, es->sc_h2, Q);
    stream_conv_mid(es, L, es->sc_h2, es->sc_t, Q, cache);
    mynah_asr_qmat_mul(&L->pw2_w, es->sc_t, out, Q);
}

int mynah_asr_enc_stream_step(mynah_asr_enc_stream *es, const float *mel, int n_mel,
                          int n_mels, int prompt_id, int is_last, float *out) {
    const mynah_asr_encoder *enc = es->enc;
    const int d = enc->d_model;

    float *x = es->sx;
    const int Q = mynah_asr_ss_stream_step(&enc->ss, &es->ss, mel, n_mel, n_mels, is_last, x);
    if (Q <= 0) return -1;

    const size_t nd = (size_t)Q * (size_t)d;
    float *tmp = es->stmp, *tmp2 = es->stmp2, *xn = es->sxn, *kn = es->skn;

    /* pos emb for this step: depends only on K = valid + Q (the same for every
     * layer) and at steady state K is CONSTANT (cache full): recompute only when
     * it changes */
    const int pe_K = es->cache_valid + Q;
    if (pe_K != es->sa_pe_K) {
        mynah_asr_pos_emb(enc, pe_K, es->sa_pe);
        es->sa_pe_K = pe_K;
    }

    for (int li = 0; li < enc->n_layers; li++) {
        const mynah_asr_enc_layer *L = &enc->layers[li];
        float *kc = es->k_cache + (size_t)li * (size_t)es->left * (size_t)d;
        float *vc = es->v_cache + (size_t)li * (size_t)es->left * (size_t)d;
        float *cc = es->conv_cache + (size_t)li * (size_t)(enc->conv_k - 1) * (size_t)d;

        /* ½ FFN1 */
        layer_norm_f(x, L->ln_ff1_w, L->ln_ff1_b, tmp, Q, d);
        mynah_asr_qmat_mul(&L->ff1_w1, tmp, tmp2, Q);
        mynah_asr_silu_scratch(tmp2, (size_t)Q * (size_t)enc->ffn_dim, es->ssilu);
        mynah_asr_qmat_mul(&L->ff1_w2, tmp2, tmp, Q);
        for (size_t i = 0; i < nd; i++) x[i] += 0.5f * tmp[i];

        /* MHSA with cache: the chunk's k/v are needed to update the cache AFTER */
        layer_norm_f(x, L->ln_att_w, L->ln_att_b, xn, Q, d);
        mynah_asr_qmat_mul(&L->k_w, xn, kn, Q);
        mynah_asr_qmat_mul(&L->v_w, xn, kn + nd, Q);
        stream_attention(es, L, xn, kn, kn + nd, es->sa_pe, tmp, Q, kc, vc,
                         es->cache_valid);
        update_kv_cache(kc, kn, es->cache_valid, Q, es->left, d);
        update_kv_cache(vc, kn + nd, es->cache_valid, Q, es->left, d);
        for (size_t i = 0; i < nd; i++) x[i] += tmp[i];

        /* Conv with cache */
        layer_norm_f(x, L->ln_conv_w, L->ln_conv_b, xn, Q, d);
        stream_conv_module(es, L, xn, tmp, Q, cc);
        for (size_t i = 0; i < nd; i++) x[i] += tmp[i];

        /* ½ FFN2 + output LN */
        layer_norm_f(x, L->ln_ff2_w, L->ln_ff2_b, tmp, Q, d);
        mynah_asr_qmat_mul(&L->ff2_w1, tmp, tmp2, Q);
        mynah_asr_silu_scratch(tmp2, (size_t)Q * (size_t)enc->ffn_dim, es->ssilu);
        mynah_asr_qmat_mul(&L->ff2_w2, tmp2, tmp, Q);
        for (size_t i = 0; i < nd; i++) x[i] += 0.5f * tmp[i];
        layer_norm_f(x, L->ln_out_w, L->ln_out_b, xn, Q, d);
        memcpy(x, xn, nd * sizeof(float));
    }

    es->cache_valid = (es->cache_valid + Q < es->left) ? es->cache_valid + Q : es->left;

    mynah_asr_encoder_post_scratch(enc, x, Q, prompt_id, out, es->spost);
    return Q;
}

/* ---------------------------------------------------- batched stream step (S1-4)
 * See encoder.h for the contract. The layer body below is mynah_asr_enc_stream_step
 * with every per-row linear lifted to the stacked [R, d] buffer and every
 * per-sequence stage (subsampling, K/V cache, rel-pos attention, conv cache,
 * prompt+projector) left inside a loop over i. SiLU is called per stream over
 * its own rows rather than once over R, so that it is literally the same call
 * the single step makes.
 *
 * MYNAH_ASR_BATCH_MAX_B is a sanity bound, not a serving policy: the per-worker
 * slot cap comes from a measured T_step(B), see .work/serving-v2-design.md §3. */
#define MYNAH_ASR_BATCH_MAX_B 256

struct mynah_asr_enc_batch {
    const mynah_asr_encoder *enc;
    int max_b, max_q, max_rows, kmax;
    int max_p;                                      /* rows the shared rk holds */
    float *buf;                                     /* one allocation */
    float *xs, *tmp, *tmp2, *xn, *kn, *qs, *ctxs, *cin;
    float *rk_sh;                                   /* [max_p, d] shared rel-pos (S1-7) */
    int8_t *qx;                                     /* [max_rows, kmax] */
    float *sx;                                      /* [max_rows] */
    int *offs, *qq, *kks;                           /* [max_b]; kks = K per stream */
};

mynah_asr_enc_batch *mynah_asr_enc_batch_new(const mynah_asr_encoder *enc, int max_b, int max_q,
                                         int max_left) {
    if (!enc || max_b < 1 || max_b > MYNAH_ASR_BATCH_MAX_B || max_q < 1 || max_left < 0)
        return NULL;
    mynah_asr_enc_batch *bb = calloc(1, sizeof(*bb));
    if (!bb) return NULL;
    bb->enc = enc;
    bb->max_b = max_b;
    bb->max_q = max_q;
    /* the shared rel-pos projection is [2K-1, d] with K = cache_valid + q, so
     * the largest it can be is the same Kmax mynah_asr_enc_stream_init uses */
    bb->max_p = 2 * (max_left + max_q + 2) - 1;
    /* +2 per stream: the same slack mynah_asr_enc_stream_init carves (Qm = q+2),
     * so a chunk that subsamples to one or two extra frames still fits */
    bb->max_rows = max_b * (max_q + 2);
    const size_t d = (size_t)enc->d_model, ffn = (size_t)enc->ffn_dim;
    /* tmp2 doubles as the stacked pointwise_conv1 output [R, 2d] */
    const size_t wide = ffn > 2 * d ? ffn : 2 * d;
    bb->kmax = (int)(ffn > d ? ffn : d);
    const size_t R = (size_t)bb->max_rows;
    const size_t nf = R * d          /* xs   */
                    + R * d          /* tmp  */
                    + R * wide       /* tmp2 */
                    + R * d          /* xn   */
                    + 2 * R * d      /* kn   */
                    + R * d          /* qs   */
                    + R * d          /* ctxs */
                    + R * d          /* cin  */
                    + (size_t)bb->max_p * d /* rk_sh */
                    + R;             /* sx   */
    bb->buf = malloc(nf * sizeof(float));
    bb->qx = malloc(R * (size_t)bb->kmax);
    bb->offs = malloc((size_t)max_b * sizeof(int));
    bb->qq = malloc((size_t)max_b * sizeof(int));
    bb->kks = malloc((size_t)max_b * sizeof(int));
    if (!bb->buf || !bb->qx || !bb->offs || !bb->qq || !bb->kks) {
        mynah_asr_enc_batch_free(bb);
        return NULL;
    }
    float *p = bb->buf;
    #define BCARVE(f, n) bb->f = p; p += (n)
    BCARVE(xs, R * d);
    BCARVE(tmp, R * d);
    BCARVE(tmp2, R * wide);
    BCARVE(xn, R * d);
    BCARVE(kn, 2 * R * d);
    BCARVE(qs, R * d);
    BCARVE(ctxs, R * d);
    BCARVE(cin, R * d);
    BCARVE(rk_sh, (size_t)bb->max_p * d);
    BCARVE(sx, R);
    #undef BCARVE
    return bb;
}

void mynah_asr_enc_batch_free(mynah_asr_enc_batch *bb) {
    if (!bb) return;
    free(bb->buf); free(bb->qx); free(bb->offs); free(bb->qq); free(bb->kks);
    free(bb);
}

int mynah_asr_enc_batch_max_rows(const mynah_asr_enc_batch *bb) { return bb ? bb->max_rows : 0; }
int mynah_asr_enc_batch_max_b(const mynah_asr_enc_batch *bb) { return bb ? bb->max_b : 0; }

/* May the f32 path stack rows? For a VENDOR BLAS this is not a property of the
 * code: cblas_sgemm gives no guarantee that row t of C is the same bytes for
 * M = q and for M = Σq, so the answer there is a per-BLAS MEASUREMENT. It is
 * one question with three answers, one per provider, and each is stated with
 * its ground.
 *
 * accelerate — MEASURED 2026-09-18, macOS arm64, nemotron-3.5-asr-streaming-0.6b,
 *   preset [56,3]: the encoder output of every chunk of every stream is
 *   identical float for float between B single steps and one batched step —
 *   74,240 (B=2), 133,120 (B=4) and 266,240 (B=8) floats compared, 0 differ,
 *   and the K/V and conv caches match too (tests/test_stream_batch, "encoder
 *   bit-exact"). ON.
 *
 * own — TRUE BY CONSTRUCTION, and then measured anyway. The stacked GEMM is
 *   x @ W^T (mynah_asr_gemm_wt and the qmat f32 fallback), i.e. trans_b, which
 *   src/sgemm.c always sends to the DOT family: output element (t, j) is one
 *   dot product of row t of x with row j of W over the whole of k, and nothing
 *   in it — not the family choice, not the column grid, not the row block —
 *   depends on m. The same tests/test_stream_batch gate runs on this build and
 *   is what would catch it if that argument were ever falsified by a change.
 *   ON.
 *
 * openblas — still UNVERIFIED: the same gate has to run on Linux before the
 *   default moves, and until it does the f32 batched step degrades to
 *   per-stream single steps there — a visible fallback
 *   (mynah_asr_stream_batch_rows_stacked stays 0), not a silent one. OFF.
 *
 * MYNAH_ASR_BATCH_F32=0|1 forces either way, which is how the Linux gate runs.
 * The integer path never consults this: it is exact by construction. */
int mynah_asr_enc_batch_f32_ok(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("MYNAH_ASR_BATCH_F32");
        if (e) cached = (e[0] == '1');
        else cached = strcmp(mynah_asr_gemm_provider(), "openblas") != 0;
    }
    return cached;
}

/* Rows that took the shared rel-pos projection, against all rows stacked. */
static _Atomic unsigned long long g_share_rows, g_share_total;

void mynah_asr_enc_batch_share_stats(unsigned long long *shared,
                                 unsigned long long *total) {
    if (shared) *shared = atomic_load_explicit(&g_share_rows, memory_order_relaxed);
    if (total) *total = atomic_load_explicit(&g_share_total, memory_order_relaxed);
}

int mynah_asr_enc_stream_step_batch(mynah_asr_enc_batch *bb,
                                mynah_asr_enc_stream *const *ess, int B,
                                const float *const *mel, const int *n_mel, int n_mels,
                                const int *prompt_id, float *const *out, int *q_out) {
    if (!bb || B < 1 || B > bb->max_b) return -1;
    const mynah_asr_encoder *enc = bb->enc;
    const int d = enc->d_model, ffn = enc->ffn_dim, ck = enc->conv_k;

    /* 1. subsampling + pos-emb, per stream; the frames land stacked in xs */
    int R = 0;
    for (int i = 0; i < B; i++) {
        mynah_asr_enc_stream *es = ess[i];
        if (es->enc != enc) return -1;
        const int Q = mynah_asr_ss_stream_step(&enc->ss, &es->ss, mel[i], n_mel[i],
                                           n_mels, 0, es->sx);
        if (Q <= 0 || R + Q > bb->max_rows) return -1;
        memcpy(bb->xs + (size_t)R * (size_t)d, es->sx, (size_t)Q * (size_t)d * sizeof(float));
        bb->offs[i] = R;
        bb->qq[i] = Q;
        q_out[i] = Q;
        R += Q;
        const int pe_K = es->cache_valid + Q;
        if (pe_K != es->sa_pe_K) {
            mynah_asr_pos_emb(enc, pe_K, es->sa_pe);
            es->sa_pe_K = pe_K;
        }
        bb->kks[i] = pe_K;
    }

    /* S1-7: the K shared by the most streams of this pass. `rk = pe @ relk_w^T`
     * depends on nothing else, so that group computes it ONCE per layer and the
     * rest of the group reads it. Ties go to the lowest K so the choice does not
     * depend on the order the caller passed the streams in. A stream at another
     * K — a slot on its first chunks, cache_valid still below left — keeps the
     * private projection, exactly as before. */
    int k_sh = 0, k_sh_n = 0, lead = 0;
    for (int i = 0; i < B; i++) {
        int n = 0;
        for (int j = 0; j < B; j++) if (bb->kks[j] == bb->kks[i]) n++;
        if (n > k_sh_n || (n == k_sh_n && bb->kks[i] < k_sh)) { k_sh = bb->kks[i]; k_sh_n = n; }
    }
    for (int i = 0; i < B; i++) if (bb->kks[i] == k_sh) { lead = i; break; }
    /* sharing pays from two streams up, and only if the scratch really holds it
     * (it is sized for max_left + max_q + 2 — a defensive check, not a policy) */
    const int P_sh = 2 * k_sh - 1;
    const int share = k_sh_n >= 2 && P_sh <= bb->max_p;

    /* How often the pass actually gets to share, and over how many rows.
     * Sharing needs two streams at the SAME K, and K = cache_valid + Q is not
     * the same until a stream's left cache has filled -- so a fleet of short
     * utterances, each of which restarts its encoder stream, can spend its
     * whole life with every row on the private projection. Counting it is the
     * difference between knowing that and assuming it. */
    atomic_fetch_add_explicit(&g_share_rows, (unsigned long long)(share ? k_sh_n : 0),
                              memory_order_relaxed);
    atomic_fetch_add_explicit(&g_share_total, (unsigned long long)B,
                              memory_order_relaxed);

    const size_t nd = (size_t)R * (size_t)d;
    float *xs = bb->xs, *tmp = bb->tmp, *tmp2 = bb->tmp2, *xn = bb->xn;
    float *kn = bb->kn, *vn = bb->kn + nd;
    int8_t *qx = bb->qx;
    float *sx = bb->sx;

    /* 2. the layer stack */
    for (int li = 0; li < enc->n_layers; li++) {
        const mynah_asr_enc_layer *L = &enc->layers[li];

        /* ½ FFN1 — stacked */
        layer_norm_f(xs, L->ln_ff1_w, L->ln_ff1_b, tmp, R, d);
        mynah_asr_qmat_mul_rows(&L->ff1_w1, tmp, tmp2, R, qx, sx);
        for (int i = 0; i < B; i++)
            mynah_asr_silu_scratch(tmp2 + (size_t)bb->offs[i] * (size_t)ffn,
                               (size_t)bb->qq[i] * (size_t)ffn, ess[i]->ssilu);
        mynah_asr_qmat_mul_rows(&L->ff1_w2, tmp2, tmp, R, qx, sx);
        for (size_t j = 0; j < nd; j++) xs[j] += 0.5f * tmp[j];

        /* MHSA — q/k/v/o stacked, the cache and the rel-pos softmax per stream */
        layer_norm_f(xs, L->ln_att_w, L->ln_att_b, xn, R, d);
        mynah_asr_qmat_mul_rows(&L->k_w, xn, kn, R, qx, sx);
        mynah_asr_qmat_mul_rows(&L->v_w, xn, vn, R, qx, sx);
        mynah_asr_qmat_mul_rows(&L->q_w, xn, bb->qs, R, qx, sx);
        /* the group's rel-pos projection, once for this layer (S1-7). ess[lead]
         * is at K = k_sh, so its sa_pe holds pos_emb(k_sh) — the same bytes every
         * other member of the group would have fed to the same matmul_wt. */
        const float *rk_sh = NULL;
        if (share) {
            matmul_wt(ess[lead]->sa_pe, L->relk_w, bb->rk_sh, P_sh, d, d);
            relpos_count(MYNAH_ASR_RELPOS_GROUP);
            rk_sh = bb->rk_sh;
        }
        for (int i = 0; i < B; i++) {
            mynah_asr_enc_stream *es = ess[i];
            const size_t off = (size_t)bb->offs[i] * (size_t)d;
            const int Q = bb->qq[i];
            float *kc = es->k_cache + (size_t)li * (size_t)es->left * (size_t)d;
            float *vc = es->v_cache + (size_t)li * (size_t)es->left * (size_t)d;
            stream_attention_core(es, L, bb->qs + off, kn + off, vn + off, es->sa_pe,
                                  bb->ctxs + off, Q, kc, vc, es->cache_valid,
                                  bb->kks[i] == k_sh ? rk_sh : NULL);
            update_kv_cache(kc, kn + off, es->cache_valid, Q, es->left, d);
            update_kv_cache(vc, vn + off, es->cache_valid, Q, es->left, d);
        }
        mynah_asr_qmat_mul_rows(&L->o_w, bb->ctxs, tmp, R, qx, sx);
        for (size_t j = 0; j < nd; j++) xs[j] += tmp[j];

        /* Conv — the two pointwise convolutions stacked, the cached depthwise
         * per stream. tmp2 holds the stacked pointwise_conv1 output [R, 2d]. */
        layer_norm_f(xs, L->ln_conv_w, L->ln_conv_b, xn, R, d);
        mynah_asr_qmat_mul_rows(&L->pw1_w, xn, tmp2, R, qx, sx);
        for (int i = 0; i < B; i++) {
            mynah_asr_enc_stream *es = ess[i];
            const size_t off = (size_t)bb->offs[i] * (size_t)d;
            float *cc = es->conv_cache + (size_t)li * (size_t)(ck - 1) * (size_t)d;
            stream_conv_mid(es, L, tmp2 + 2u * off, bb->cin + off, bb->qq[i], cc);
        }
        mynah_asr_qmat_mul_rows(&L->pw2_w, bb->cin, tmp, R, qx, sx);
        for (size_t j = 0; j < nd; j++) xs[j] += tmp[j];

        /* ½ FFN2 + output LN — stacked */
        layer_norm_f(xs, L->ln_ff2_w, L->ln_ff2_b, tmp, R, d);
        mynah_asr_qmat_mul_rows(&L->ff2_w1, tmp, tmp2, R, qx, sx);
        for (int i = 0; i < B; i++)
            mynah_asr_silu_scratch(tmp2 + (size_t)bb->offs[i] * (size_t)ffn,
                               (size_t)bb->qq[i] * (size_t)ffn, ess[i]->ssilu);
        mynah_asr_qmat_mul_rows(&L->ff2_w2, tmp2, tmp, R, qx, sx);
        for (size_t j = 0; j < nd; j++) xs[j] += 0.5f * tmp[j];
        layer_norm_f(xs, L->ln_out_w, L->ln_out_b, xn, R, d);
        memcpy(xs, xn, nd * sizeof(float));
    }

    /* 3. cache bookkeeping and the per-stream prompt + projector */
    for (int i = 0; i < B; i++) {
        mynah_asr_enc_stream *es = ess[i];
        const int Q = bb->qq[i];
        es->cache_valid = (es->cache_valid + Q < es->left) ? es->cache_valid + Q : es->left;
        mynah_asr_encoder_post_scratch(enc, xs + (size_t)bb->offs[i] * (size_t)d, Q,
                                   prompt_id[i], out[i], es->spost);
    }
    return 0;
}

/* --------------------------------------------------------- batched forward
 * Padding-free packing: x = the frames of every sequence concatenated [ΣT, d].
 * FFN/LN run over the whole packed buffer (weight-stationary); attention and
 * conv, which depend on per-sequence causality, iterate over the segments —
 * IN PARALLEL (independent segments, disjoint outputs: bit-identical to the
 * serial loop). This is what makes batching pay off on many-core: BLAS
 * parallelizes the packed GEMMs, we parallelize the segments. */
typedef struct {
    const mynah_asr_encoder *enc;
    const mynah_asr_enc_layer *L;
    const float *xn;
    float *tmp;
    const int *t_enc, *offs;
    float *const *pes;
    int left, right, is_conv;
} seg_par;

static void seg_worker(void *ctx, int b) {
    const seg_par *sp = ctx;
    const size_t off = (size_t)sp->offs[b] * (size_t)sp->enc->d_model;
    if (sp->is_conv)
        conv_module(sp->enc, sp->L, sp->xn + off, sp->tmp + off, sp->t_enc[b]);
    else
        attention(sp->enc, sp->L, sp->xn + off, sp->tmp + off, sp->t_enc[b],
                  sp->pes[b], sp->left, sp->right);
}

static int encoder_layer_batch(const mynah_asr_encoder *enc, int li, float *x,
                               const int *t_enc, int batch, float *const *pes,
                               int left, int right) {
    const mynah_asr_enc_layer *L = &enc->layers[li];
    const int d = enc->d_model;
    int T_total = 0;
    for (int b = 0; b < batch; b++) T_total += t_enc[b];
    const size_t n = (size_t)T_total * (size_t)d;

    float *tmp = malloc(n * sizeof(float));
    float *tmp2 = malloc((size_t)T_total * (size_t)enc->ffn_dim * sizeof(float));
    float *xn = malloc(n * sizeof(float));
    if (!tmp || !tmp2 || !xn) { free(tmp); free(tmp2); free(xn); return -1; }

    /* ½ FFN1 — packed (fused when bias-free) */
    half_ffn(enc, L, x, tmp, tmp2, T_total, 0);
    for (size_t i = 0; i < n; i++) x[i] += 0.5f * tmp[i];

    int *offs = malloc((size_t)batch * sizeof(int));
    if (!offs) { free(tmp); free(tmp2); free(xn); return -1; }
    for (int b = 0, off = 0; b < batch; off += t_enc[b], b++) offs[b] = off;
    seg_par sp = {.enc = enc, .L = L, .xn = xn, .tmp = tmp, .t_enc = t_enc,
                  .offs = offs, .pes = pes, .left = left, .right = right};

    /* MHSA — segments in parallel */
    layer_norm_f(x, L->ln_att_w, L->ln_att_b, xn, T_total, d);
    sp.is_conv = 0;
    mynah_asr_parallel_for(batch, seg_worker, &sp);
    for (size_t i = 0; i < n; i++) x[i] += tmp[i];

    /* Conv — segments in parallel (causal per segment) */
    layer_norm_f(x, L->ln_conv_w, L->ln_conv_b, xn, T_total, d);
    sp.is_conv = 1;
    mynah_asr_parallel_for(batch, seg_worker, &sp);
    for (size_t i = 0; i < n; i++) x[i] += tmp[i];
    free(offs);

    /* ½ FFN2 + output LN — packed (fused when bias-free) */
    half_ffn(enc, L, x, tmp, tmp2, T_total, 1);
    for (size_t i = 0; i < n; i++) x[i] += 0.5f * tmp[i];
    layer_norm_f(x, L->ln_out_w, L->ln_out_b, xn, T_total, d);
    memcpy(x, xn, n * sizeof(float));

    free(tmp); free(tmp2); free(xn);
    return 0;
}

int mynah_asr_encoder_forward_batch(const mynah_asr_encoder *enc, const float *const *feats,
                                const int *t_mel, int batch, int n_mels,
                                const int *prompt_ids, int left_ctx, int right_ctx,
                                float **outs, int *t_outs) {
    const int d = enc->d_model;
    float **seq = calloc((size_t)batch, sizeof(float *));
    float **pes = calloc((size_t)batch, sizeof(float *));
    if (!seq || !pes) { free(seq); free(pes); return -1; }

    int T_total = 0, rc = -1;
    for (int b = 0; b < batch; b++) {
        seq[b] = mynah_asr_subsampling_forward(&enc->ss, feats[b], t_mel[b], n_mels, &t_outs[b]);
        if (!seq[b]) goto done;
        if (enc->xscale != 1.0f)
            for (size_t i = 0; i < (size_t)t_outs[b] * (size_t)d; i++) seq[b][i] *= enc->xscale;
        pes[b] = malloc((size_t)(2 * t_outs[b] - 1) * (size_t)d * sizeof(float));
        if (!pes[b]) goto done;
        mynah_asr_pos_emb(enc, t_outs[b], pes[b]);
        T_total += t_outs[b];
    }

    float *x = malloc((size_t)T_total * (size_t)d * sizeof(float));
    if (!x) goto done;
    for (int b = 0, off = 0; b < batch; off += t_outs[b], b++) {
        memcpy(x + (size_t)off * (size_t)d, seq[b],
               (size_t)t_outs[b] * (size_t)d * sizeof(float));
        free(seq[b]);
        seq[b] = NULL;
    }

#ifdef MYNAH_ASR_METAL
    /* On Metal each segment runs the whole encoder on GPU (resident weights =
     * weight-stationary anyway); packing only pays off on the CPU GEMMs.
     * Same gate as run_layers. */
    if (mynah_asr_backend() == MYNAH_ASR_BACKEND_METAL && enc->conv_k == 9 &&
        enc->layers[0].q_w.qtype == MYNAH_ASR_Q_F32) {
        for (int b = 0, off = 0; b < batch; off += t_outs[b], b++)
            if (run_layers(enc, x + (size_t)off * (size_t)d, t_outs[b], pes[b],
                           left_ctx, right_ctx) != 0) {
                free(x);
                goto done;
            }
    } else
#endif
    for (int li = 0; li < enc->n_layers; li++)
        if (encoder_layer_batch(enc, li, x, t_outs, batch, pes, left_ctx, right_ctx) != 0) {
            free(x);
            goto done;
        }

    rc = 0;
    for (int b = 0, off = 0; b < batch; off += t_outs[b], b++) {
        outs[b] = malloc((size_t)t_outs[b] * (size_t)enc->d_out * sizeof(float));
        if (!outs[b]) { rc = -1; continue; }
        mynah_asr_encoder_post(enc, x + (size_t)off * (size_t)d, t_outs[b], prompt_ids[b], outs[b]);
    }
    free(x);

done:
    for (int b = 0; b < batch; b++) { free(seq[b]); free(pes[b]); }
    free(seq); free(pes);
    return rc;
}

/* ---------------------------------------------------------------- forward */
static float *forward_core(const mynah_asr_encoder *enc, const float *feats, int t_mel,
                           int n_mels, int prompt_id, int left_ctx, int right_ctx,
                           int *t_out, int do_post) {
    int T;
    float *x = mynah_asr_subsampling_forward(&enc->ss, feats, t_mel, n_mels, &T);
    if (!x) return NULL;
    if (enc->xscale != 1.0f)
        for (size_t i = 0; i < (size_t)T * (size_t)enc->d_model; i++) x[i] *= enc->xscale;

    float *pe = malloc((size_t)(2 * T - 1) * (size_t)enc->d_model * sizeof(float));
    if (!pe) { free(x); return NULL; }
    mynah_asr_pos_emb(enc, T, pe);

    if (run_layers(enc, x, T, pe, left_ctx, right_ctx) != 0) {
        free(x); free(pe);
        return NULL;
    }
    free(pe);
    *t_out = T;
    if (!do_post) return x;                /* raw encoder out [T, d_model] (CTC) */

    float *out = malloc((size_t)T * (size_t)enc->d_out * sizeof(float));
    if (!out) { free(x); return NULL; }
    mynah_asr_encoder_post(enc, x, T, prompt_id, out);
    free(x);
    return out;
}

float *mynah_asr_encoder_forward(const mynah_asr_encoder *enc, const float *feats, int t_mel,
                             int n_mels, int prompt_id, int left_ctx, int right_ctx,
                             int *t_out) {
    return forward_core(enc, feats, t_mel, n_mels, prompt_id, left_ctx, right_ctx,
                        t_out, 1);
}

float *mynah_asr_encoder_forward_raw(const mynah_asr_encoder *enc, const float *feats, int t_mel,
                                 int n_mels, int left_ctx, int right_ctx, int *t_out) {
    return forward_core(enc, feats, t_mel, n_mels, 0, left_ctx, right_ctx, t_out, 0);
}
