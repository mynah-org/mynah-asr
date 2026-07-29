#include "decoder_aed.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MYNAH_ASR_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* y = W x + b (T=1 sui qmat: kernel dot diretto per int8/int4) */
static void mv(const mynah_asr_qmat *w, const float *b, const float *x, float *y) {
    mynah_asr_qmat_mul(w, x, y, 1);
    if (b)
        for (int i = 0; i < w->n; i++) y[i] += b[i];
}

static void ln(const float *x, const float *w, const float *b, float *y, int d) {
    double mu = 0.0, var = 0.0;
    for (int i = 0; i < d; i++) mu += x[i];
    mu /= d;
    for (int i = 0; i < d; i++) { const double c = x[i] - mu; var += c * c; }
    const float inv = (float)(1.0 / sqrt(var / d + 1e-5));
    for (int i = 0; i < d; i++) y[i] = (float)((x[i] - mu) * inv) * w[i] + b[i];
}

static void softmax_inplace(float *s, int n) {
    float m = s[0];
    for (int i = 1; i < n; i++) if (s[i] > m) m = s[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { s[i] = expf(s[i] - m); sum += s[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) s[i] *= inv;
}

#define T_(st, name) ((const float *)(mynah_asr_st_get((st), (name)) ? mynah_asr_st_get((st), (name))->data : NULL))

int mynah_asr_aed_init(mynah_asr_aed *a, const mynah_asr_safetensors *st,
                   int n_layers, int n_heads, int max_seq, int max_gen_delta,
                   int quantize) {
    memset(a, 0, sizeof(*a));
    const mynah_asr_tensor *emb = mynah_asr_st_get(st, "aed.embedding.weight");
    if (!emb) return -1;
    a->emb = (const float *)emb->data;
    a->vocab = (int)emb->shape[0];
    a->d = (int)emb->shape[1];
    a->n_layers = n_layers;
    a->n_heads = n_heads;
    a->max_seq = max_seq;
    a->max_gen_delta = max_gen_delta;
    a->pos = T_(st, "aed.pos_enc");
    a->embln_w = T_(st, "aed.emb_norm.weight");
    a->embln_b = T_(st, "aed.emb_norm.bias");
    a->fin_w = T_(st, "aed.final_norm.weight");
    a->fin_b = T_(st, "aed.final_norm.bias");
    a->head_b = T_(st, "aed.head.bias");
    if (!a->pos || !a->embln_w || !a->fin_w || !a->head_b) return -1;
    if (mynah_asr_qmat_init_st(&a->head, st, "aed.head.weight", quantize) != 0) return -1;
    if (mynah_asr_qmat_init_st(&a->proj, st, "enc_dec_proj.weight", quantize) == 0) {
        a->proj_b = T_(st, "enc_dec_proj.bias");
        a->d_enc = a->proj.k;
    } else {
        a->d_enc = a->d;                /* enc hidden == dec hidden: identity */
    }

    a->layers = calloc((size_t)n_layers, sizeof(*a->layers));
    if (!a->layers) return -1;
    for (int li = 0; li < n_layers; li++) {
        mynah_asr_aed_layer *L = &a->layers[li];
        char n[96];
#define GF(field, suffix) \
        snprintf(n, sizeof(n), "aed.layers.%d." suffix, li); \
        L->field = T_(st, n); \
        if (!L->field) { fprintf(stderr, "mynah-asr: aed is missing %s\n", n); return -1; }
#define GQ(field, suffix) \
        snprintf(n, sizeof(n), "aed.layers.%d." suffix, li); \
        if (mynah_asr_qmat_init_st(&L->field, st, n, quantize) != 0) { \
            fprintf(stderr, "mynah-asr: aed is missing %s\n", n); return -1; }
        GF(ln_self_w, "ln_self.weight")   GF(ln_self_b, "ln_self.bias")
        GQ(sq, "self_attn.q_proj.weight") GF(sq_b, "self_attn.q_proj.bias")
        GQ(sk, "self_attn.k_proj.weight") GF(sk_b, "self_attn.k_proj.bias")
        GQ(sv, "self_attn.v_proj.weight") GF(sv_b, "self_attn.v_proj.bias")
        GQ(so, "self_attn.o_proj.weight") GF(so_b, "self_attn.o_proj.bias")
        GF(ln_cross_w, "ln_cross.weight") GF(ln_cross_b, "ln_cross.bias")
        GQ(cq, "cross_attn.q_proj.weight") GF(cq_b, "cross_attn.q_proj.bias")
        GQ(ck, "cross_attn.k_proj.weight") GF(ck_b, "cross_attn.k_proj.bias")
        GQ(cv, "cross_attn.v_proj.weight") GF(cv_b, "cross_attn.v_proj.bias")
        GQ(co, "cross_attn.o_proj.weight") GF(co_b, "cross_attn.o_proj.bias")
        GF(ln_ffn_w, "ln_ffn.weight")     GF(ln_ffn_b, "ln_ffn.bias")
        GQ(ff1, "ffn.linear1.weight")     GF(ff1_b, "ffn.linear1.bias")
        GQ(ff2, "ffn.linear2.weight")     GF(ff2_b, "ffn.linear2.bias")
#undef GF
#undef GQ
    }
    a->ffn = a->layers[0].ff1.n;
    return 0;
}

void mynah_asr_aed_free(mynah_asr_aed *a) {
    for (int li = 0; li < a->n_layers && a->layers; li++) {
        mynah_asr_aed_layer *L = &a->layers[li];
        mynah_asr_qmat_free(&L->sq); mynah_asr_qmat_free(&L->sk);
        mynah_asr_qmat_free(&L->sv); mynah_asr_qmat_free(&L->so);
        mynah_asr_qmat_free(&L->cq); mynah_asr_qmat_free(&L->ck);
        mynah_asr_qmat_free(&L->cv); mynah_asr_qmat_free(&L->co);
        mynah_asr_qmat_free(&L->ff1); mynah_asr_qmat_free(&L->ff2);
    }
    mynah_asr_qmat_free(&a->head);
    mynah_asr_qmat_free(&a->proj);
    free(a->layers);
    a->layers = NULL;
}

/* attention di UNA query su n_kv chiavi/valori [n_kv, d] (righe contigue),
 * per-head, score scalati 1/sqrt(dk). out [d]. scores: scratch [n_kv]. */
static void attend(const float *q, const float *K, const float *V, int n_kv,
                   int H, int dk, float *scores, float *out) {
    const int d = H * dk;
    const float scale = 1.0f / sqrtf((float)dk);
    for (int h = 0; h < H; h++) {
        const float *qh = q + h * dk;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n_kv, dk, scale, K + h * dk, d,
                    qh, 1, 0.0f, scores, 1);
        softmax_inplace(scores, n_kv);
        cblas_sgemv(CblasRowMajor, CblasTrans, n_kv, dk, 1.0f, V + h * dk, d,
                    scores, 1, 0.0f, out + h * dk, 1);
    }
}

static void add_bias_rows(float *x, const float *b, int T, int d) {
    if (!b) return;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++) x[(size_t)t * (size_t)d + i] += b[i];
}

int mynah_asr_aed_decode(const mynah_asr_aed *a, const float *enc, int T,
                     const int *prompt, int n_prompt, int eos,
                     int *tokens, int cap) {
    const int d = a->d, H = a->n_heads, dk = d / H, nl = a->n_layers;
    if (n_prompt <= 0 || !prompt) return -1;   /* serve almeno un token di avvio */
    /* budget di generazione = cap del caller (che sa se i timestamp sono attivi) */
    int max_len = n_prompt + cap;
    if (max_len > a->max_seq) max_len = a->max_seq;

    /* proiezione encoder -> spazio decoder */
    float *encp;
    if (a->proj.n) {
        encp = malloc((size_t)T * (size_t)d * sizeof(float));
        if (!encp) return -1;
        mynah_asr_qmat_mul(&a->proj, enc, encp, T);
        add_bias_rows(encp, a->proj_b, T, d);
    } else {
        encp = (float *)enc;
    }

    /* cross K/V per layer (una volta) + cache self K/V + scratch */
    const size_t td = (size_t)T * (size_t)d, md = (size_t)max_len * (size_t)d;
    float *ckv = malloc(2u * (size_t)nl * td * sizeof(float));
    float *skv = malloc(2u * (size_t)nl * md * sizeof(float));
    float *scr = malloc((size_t)(6 * d + a->ffn + (T > max_len ? T : max_len)) * sizeof(float));
    float *logits = malloc((size_t)a->vocab * sizeof(float));
    int rc = -1, n_out = 0;
    if (!ckv || !skv || !scr || !logits) goto done;
    float *x = scr, *xn = scr + d, *q = scr + 2 * d, *att = scr + 3 * d,
          *tmp = scr + 4 * d, *fin = scr + 5 * d, *ff = scr + 6 * d,
          *scores = scr + 6 * d + a->ffn;

    for (int li = 0; li < nl; li++) {
        const mynah_asr_aed_layer *L = &a->layers[li];
        float *CK = ckv + (size_t)(2 * li) * td, *CV = CK + td;
        mynah_asr_qmat_mul(&L->ck, encp, CK, T);
        mynah_asr_qmat_mul(&L->cv, encp, CV, T);
        add_bias_rows(CK, L->ck_b, T, d);
        add_bias_rows(CV, L->cv_b, T, d);
    }

    int cur = prompt[0];
    for (int p = 0; p < max_len; p++) {
        /* embedding + pos enc (buffer dal ckpt) + LN */
        const float *e = a->emb + (size_t)cur * d, *pe = a->pos + (size_t)p * d;
        for (int i = 0; i < d; i++) x[i] = e[i] + pe[i];
        ln(x, a->embln_w, a->embln_b, x, d);

        for (int li = 0; li < nl; li++) {
            const mynah_asr_aed_layer *L = &a->layers[li];
            float *SK = skv + (size_t)(2 * li) * md, *SV = SK + md;
            /* self-attention causale: nuova k/v in cache, attention su [0..p] */
            ln(x, L->ln_self_w, L->ln_self_b, xn, d);
            mv(&L->sq, L->sq_b, xn, q);
            mv(&L->sk, L->sk_b, xn, SK + (size_t)p * d);
            mv(&L->sv, L->sv_b, xn, SV + (size_t)p * d);
            attend(q, SK, SV, p + 1, H, dk, scores, att);
            mv(&L->so, L->so_b, att, tmp);
            for (int i = 0; i < d; i++) x[i] += tmp[i];
            /* cross-attention sull'encoder proiettato */
            ln(x, L->ln_cross_w, L->ln_cross_b, xn, d);
            mv(&L->cq, L->cq_b, xn, q);
            attend(q, ckv + (size_t)(2 * li) * td, ckv + (size_t)(2 * li) * td + td,
                   T, H, dk, scores, att);
            mv(&L->co, L->co_b, att, tmp);
            for (int i = 0; i < d; i++) x[i] += tmp[i];
            /* FFN ReLU */
            ln(x, L->ln_ffn_w, L->ln_ffn_b, xn, d);
            mv(&L->ff1, L->ff1_b, xn, ff);
            for (int i = 0; i < a->ffn; i++) if (ff[i] < 0.0f) ff[i] = 0.0f;
            mv(&L->ff2, L->ff2_b, ff, tmp);
            for (int i = 0; i < d; i++) x[i] += tmp[i];
        }

        if (p + 1 < n_prompt) {          /* ancora nel prompt: niente sampling */
            cur = prompt[p + 1];
            continue;
        }
        ln(x, a->fin_w, a->fin_b, fin, d);
        mv(&a->head, a->head_b, fin, logits);
        int best = 0;
        for (int i = 1; i < a->vocab; i++) if (logits[i] > logits[best]) best = i;
        if (best == eos) break;
        if (n_out >= cap) break;
        tokens[n_out++] = best;
        cur = best;
    }
    rc = n_out;

done:
    if (encp != enc) free(encp);
    free(ckv); free(skv); free(scr); free(logits);
    return rc;
}
