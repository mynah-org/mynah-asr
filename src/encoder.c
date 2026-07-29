#include "encoder.h"

#include "backend.h"
#include "threads.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MYNAH_ASR_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

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
    const mynah_asr_tensor *ep = mynah_asr_st_get(st, "encoder_projector.weight");  /* opzionale (CTC puro) */
    const mynah_asr_tensor *p1 = mynah_asr_st_get(st, "prompt_projector.linear_1.weight"); /* opzionale */
    if (!bu || !dw) return -1;

    enc->d_model = enc->ss.d_model;
    enc->n_heads = (int)bu->shape[0];
    enc->d_head = (int)bu->shape[1];
    enc->conv_k = (int)dw->shape[2];
    enc->d_out = ep ? (int)ep->shape[0] : enc->d_model;
    enc->causal = enc->ss.causal;  /* default dal naming; il config puo' sovrascrivere */
    enc->xscale = 1.0f;            /* xscaling dal config (mynah.c) */
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
    const int B = 128 / chunk > 0 ? 128 / chunk : 1;   /* chunk di query per blocco */
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
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Rb, pW, dk,
                        1.0f, qb, dk, rk + (size_t)(p0 - pu0) * (size_t)d + ho, d, 0.0f, bdb, pW);

            /* ac_block[r, jl] = (q[t0+r]+bias_u) . k[c0+jl] */
            for (int r = 0; r < Rb; r++)
                for (int i = 0; i < dk; i++)
                    qb[(size_t)r * (size_t)dk + (size_t)i] =
                        q[(size_t)(t0 + r) * (size_t)d + ho + (size_t)i] + L->bias_u[ho + (size_t)i];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Rb, Wb, dk,
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
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Rb, dk, Wb,
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
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, P, dk,
                    1.0f, qb, dk, rk + ho, d, 0.0f, bd, P);

        /* q + bias_u (for matrix_ac) */
        for (int t = 0; t < T; t++)
            for (int i = 0; i < dk; i++)
                qb[(size_t)t * (size_t)dk + (size_t)i] = q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_u[ho + (size_t)i];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, dk,
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
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, dk, T,
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
void mynah_asr_encoder_post(const mynah_asr_encoder *enc, const float *x, int T, int prompt_id,
                        float *out) {
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
    float *cat = calloc((size_t)T * (size_t)dcat, sizeof(float));
    float *mid = malloc((size_t)T * (size_t)di * sizeof(float));
    float *fused = malloc((size_t)T * (size_t)d * sizeof(float));
    if (!cat || !mid || !fused) { free(cat); free(mid); free(fused); return; }

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
    free(cat); free(mid); free(fused);
}

/* --------------------------------------------------------------- streaming */

int mynah_asr_enc_stream_init(mynah_asr_enc_stream *es, const mynah_asr_encoder *enc,
                          int left_ctx, int right_ctx, int n_mels) {
    memset(es, 0, sizeof(*es));
    es->enc = enc;
    es->left = left_ctx;
    es->right = right_ctx;
    es->q = right_ctx + 1;
    if (mynah_asr_ss_stream_init(&es->ss, &enc->ss, n_mels) != 0) return -1;
    const size_t kv = (size_t)enc->n_layers * (size_t)left_ctx * (size_t)enc->d_model;
    const size_t cv = (size_t)enc->n_layers * (size_t)(enc->conv_k - 1) * (size_t)enc->d_model;
    es->k_cache = calloc(kv, sizeof(float));
    es->v_cache = calloc(kv, sizeof(float));
    es->conv_cache = calloc(cv, sizeof(float));
    if (!es->k_cache || !es->v_cache || !es->conv_cache) return -1;

    /* single scratch for the hot path (max sizes, reused on every chunk) */
    const size_t d = (size_t)enc->d_model, ck = (size_t)enc->conv_k;
    const size_t Qm = (size_t)es->q + 2, Km = (size_t)left_ctx + Qm, Pm = 2 * Km - 1;
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
                    + Qm * d;                 /* sc_t */
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
    #undef CARVE
    return 0;
}

void mynah_asr_enc_stream_free(mynah_asr_enc_stream *es) {
    mynah_asr_ss_stream_free(&es->ss);
    free(es->k_cache); free(es->v_cache); free(es->conv_cache); free(es->scr);
    es->k_cache = es->v_cache = es->conv_cache = es->scr = NULL;
}

int mynah_asr_enc_stream_need(const mynah_asr_enc_stream *es) {
    const int sub = 8; /* subsampling_factor: 3 stadi stride-2 */
    return es->cache_valid == 0 && es->ss.first ? 1 + sub * es->right
                                                : sub * (es->right + 1);
}

/* Streaming attention: Q query rows, K/V = [valid cache ; chunk], no mask
 * (the cache holds exactly the left context the chunked grid allows).
 * pe [2K-1, d] is computed by the caller (once per chunk, not per layer);
 * scratch preallocated in es (zero mallocs in the hot path). */
static void stream_attention(mynah_asr_enc_stream *es, const mynah_asr_enc_layer *L,
                             const float *x, const float *kn, const float *vn,
                             const float *pe, float *out, int Q,
                             const float *k_cache, const float *v_cache, int valid) {
    const mynah_asr_encoder *enc = es->enc;
    const int d = enc->d_model, H = enc->n_heads, dk = enc->d_head;
    const int K = valid + Q, P = 2 * K - 1;
    const float scaling = 1.0f / sqrtf((float)dk);

    float *rk = es->sa_rk, *scores = es->sa_sc, *bd = es->sa_bd;
    float *qb = es->sa_qb, *ctx = es->sa_ctx;

    {
        float *q = es->sa_q;
        float *kk = es->sa_keys, *vv = es->sa_keys + (size_t)K * (size_t)d;
        mynah_asr_qmat_mul(&L->q_w, x, q, Q);

        /* keys = valid cache ++ new ones (the caller updates the cache) */
        memcpy(kk, k_cache, (size_t)valid * (size_t)d * sizeof(float));
        memcpy(kk + (size_t)valid * (size_t)d, kn, (size_t)Q * (size_t)d * sizeof(float));
        memcpy(vv, v_cache, (size_t)valid * (size_t)d * sizeof(float));
        memcpy(vv + (size_t)valid * (size_t)d, vn, (size_t)Q * (size_t)d * sizeof(float));

        matmul_wt(pe, L->relk_w, rk, P, d, d);

        for (int h = 0; h < H; h++) {
            const size_t ho = (size_t)h * (size_t)dk;
            for (int t = 0; t < Q; t++)
                for (int i = 0; i < dk; i++)
                    qb[(size_t)t * (size_t)dk + (size_t)i] =
                        q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_v[ho + (size_t)i];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q, P, dk,
                        1.0f, qb, dk, rk + ho, d, 0.0f, bd, P);

            for (int t = 0; t < Q; t++)
                for (int i = 0; i < dk; i++)
                    qb[(size_t)t * (size_t)dk + (size_t)i] =
                        q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_u[ho + (size_t)i];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q, K, dk,
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

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Q, dk, K,
                        1.0f, scores, K, vv + ho, d, 0.0f, qb, dk);
            for (int t = 0; t < Q; t++)
                memcpy(ctx + (size_t)t * (size_t)d + ho, qb + (size_t)t * (size_t)dk,
                       (size_t)dk * sizeof(float));
        }
        mynah_asr_qmat_mul(&L->o_w, ctx, out, Q);
    }
}

/* update a layer's K/V cache with the new k/v rows of the chunk */
static void update_kv_cache(float *cache, const float *fresh, int valid, int Q,
                            int left, int d) {
    const int total = valid + Q;
    const int keep = total < left ? total : left;
    const int from_old = keep - Q > 0 ? keep - Q : 0;      /* righe vecchie da tenere */
    const int drop_old = valid - from_old;                  /* righe vecchie da scartare */
    if (from_old > 0 && drop_old > 0)
        memmove(cache, cache + (size_t)drop_old * (size_t)d,
                (size_t)from_old * (size_t)d * sizeof(float));
    const int fresh_keep = keep - from_old;                 /* righe nuove da tenere (<= Q) */
    memcpy(cache + (size_t)from_old * (size_t)d,
           fresh + (size_t)(Q - fresh_keep) * (size_t)d,
           (size_t)fresh_keep * (size_t)d * sizeof(float));
}

/* Streaming conv module: cache [k-1, d] prepended, refreshed with the last k-1 rows. */
static void stream_conv_module(mynah_asr_enc_stream *es, const mynah_asr_enc_layer *L,
                               const float *x, float *out, int Q, float *cache) {
    const mynah_asr_encoder *enc = es->enc;
    const int d = enc->d_model, k = enc->conv_k;
    float *h2 = es->sc_h2, *gp = es->sc_gp, *c = es->sc_c;

    mynah_asr_qmat_mul(&L->pw1_w, x, h2, Q);
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
    layer_norm_f(c, L->cnorm_w, L->cnorm_b, es->sc_t, Q, d);
    silu_inplace(es->sc_t, (size_t)Q * (size_t)d);
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
        silu_inplace(tmp2, (size_t)Q * (size_t)enc->ffn_dim);
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
        silu_inplace(tmp2, (size_t)Q * (size_t)enc->ffn_dim);
        mynah_asr_qmat_mul(&L->ff2_w2, tmp2, tmp, Q);
        for (size_t i = 0; i < nd; i++) x[i] += 0.5f * tmp[i];
        layer_norm_f(x, L->ln_out_w, L->ln_out_b, xn, Q, d);
        memcpy(x, xn, nd * sizeof(float));
    }

    es->cache_valid = (es->cache_valid + Q < es->left) ? es->cache_valid + Q : es->left;

    mynah_asr_encoder_post(enc, x, Q, prompt_id, out);
    return Q;
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
    if (!do_post) return x;                /* encoder out grezzo [T, d_model] (CTC) */

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
