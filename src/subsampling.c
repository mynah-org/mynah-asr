#include "subsampling.h"
#include "threads.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MYNAH_ASR_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

int mynah_asr_subsampling_init(mynah_asr_subsampling *ss, const mynah_asr_safetensors *st) {
    memset(ss, 0, sizeof(*ss));
    ss->conv_in_w = mynah_asr_st_get(st, "encoder.subsampling.conv_in.weight");
    if (ss->conv_in_w) {
        /* Nemotron naming: conv_in + layers.{i}.{depthwise,pointwise}_conv (causal) */
        ss->causal = 1;
        ss->conv_in_b = mynah_asr_st_get(st, "encoder.subsampling.conv_in.bias");
        for (int i = 0; i < 2; i++) {
            char name[96];
            snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.depthwise_conv.weight", i);
            ss->dw_w[i] = mynah_asr_st_get(st, name);
            snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.depthwise_conv.bias", i);
            ss->dw_b[i] = mynah_asr_st_get(st, name);
            snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.pointwise_conv.weight", i);
            ss->pw_w[i] = mynah_asr_st_get(st, name);
            snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.pointwise_conv.bias", i);
            ss->pw_b[i] = mynah_asr_st_get(st, name);
            if (!ss->dw_w[i] || !ss->dw_b[i] || !ss->pw_w[i] || !ss->pw_b[i]) return -1;
        }
    } else {
        /* Parakeet naming: flat Sequential layers.{0,2,3,5,6} (offline, symmetric) */
        ss->causal = 0;
        static const int dw_idx[2] = {2, 5}, pw_idx[2] = {3, 6};
        char name[96];
        ss->conv_in_w = mynah_asr_st_get(st, "encoder.subsampling.layers.0.weight");
        ss->conv_in_b = mynah_asr_st_get(st, "encoder.subsampling.layers.0.bias");
        for (int i = 0; i < 2; i++) {
            snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.weight", dw_idx[i]);
            ss->dw_w[i] = mynah_asr_st_get(st, name);
            snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.bias", dw_idx[i]);
            ss->dw_b[i] = mynah_asr_st_get(st, name);
            snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.weight", pw_idx[i]);
            ss->pw_w[i] = mynah_asr_st_get(st, name);
            snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.bias", pw_idx[i]);
            ss->pw_b[i] = mynah_asr_st_get(st, name);
            if (!ss->dw_w[i] || !ss->dw_b[i] || !ss->pw_w[i] || !ss->pw_b[i]) return -1;
        }
    }
    ss->lin_w = mynah_asr_st_get(st, "encoder.subsampling.linear.weight");
    ss->lin_b = mynah_asr_st_get(st, "encoder.subsampling.linear.bias");
    if (!ss->conv_in_w || !ss->conv_in_b || !ss->lin_w || !ss->lin_b) return -1;
    ss->channels = (int)ss->conv_in_w->shape[0];
    ss->d_model = (int)ss->lin_w->shape[0];
    return 0;
}

/* Depthwise worker over a slice of channels (independent channels, disjoint
 * output -> bit-identical to the serial loop). Pad allocated per slice. */
#include <stdatomic.h>
typedef struct {
    const float *x, *w, *b;
    float *out;
    int T, F, Tp, Fp2, To, Fo, pl, pf, s, C_out, slices;
    atomic_int failed;
} dw_par;

static void dw_slice_worker(void *ctx, int wslice) {
    dw_par *dp = ctx;
    const int c0 = (int)((long)dp->C_out * wslice / dp->slices);
    const int c1 = (int)((long)dp->C_out * (wslice + 1) / dp->slices);
    const int T = dp->T, F = dp->F, Tp = dp->Tp, Fp2 = dp->Fp2;
    const int To = dp->To, Fo = dp->Fo, pl = dp->pl, pf = dp->pf, s = dp->s;

    float *pad = malloc((size_t)Tp * (size_t)Fp2 * sizeof(float));
    if (!pad) { atomic_store(&dp->failed, 1); return; }
    for (int co = c0; co < c1; co++) {
        const float *src = dp->x + (size_t)co * (size_t)T * (size_t)F;
        const float *ker = dp->w + (size_t)co * 9u;
        memset(pad, 0, (size_t)Tp * (size_t)Fp2 * sizeof(float));
        for (int t = 0; t < T; t++)
            memcpy(pad + (size_t)(t + pl) * (size_t)Fp2 + (size_t)pf,
                   src + (size_t)t * (size_t)F, (size_t)F * sizeof(float));
        float *dst = dp->out + (size_t)co * (size_t)To * (size_t)Fo;
        for (int to = 0; to < To; to++) {
            const float *p0 = pad + (size_t)(to * s) * (size_t)Fp2;
            const float *p1 = p0 + Fp2, *p2 = p1 + Fp2;
            float *drow = dst + (size_t)to * (size_t)Fo;
            for (int fo = 0; fo < Fo; fo++) {
                const int f = fo * s;
                const float acc = ker[0] * p0[f] + ker[1] * p0[f + 1] + ker[2] * p0[f + 2]
                                + ker[3] * p1[f] + ker[4] * p1[f + 1] + ker[5] * p1[f + 2]
                                + ker[6] * p2[f] + ker[7] * p2[f + 1] + ker[8] * p2[f + 2];
                drow[fo] = dp->b[co] + acc;
            }
        }
    }
    free(pad);
}

/* k3 s2 conv with parametric time (pl_t, pr_t) and freq (pl_f, pr_f) padding.
 * x [C_in, T, F] -> out [C_out, T', F']. Causal: (2,1); symmetric: (1,1);
 * streaming: time (0,0) (the input already carries cache+init on the left).
 * depthwise: C_in == C_out, weight [C,1,3,3]. full: weight [C_out, C_in, 3, 3]. */
static void conv2d_s2(const float *x, int C_in, int T, int F, int pl_t, int pr_t,
                      int pl_f, int pr_f, const float *w, const float *b, int C_out,
                      int depthwise, float *out, int *To_, int *Fo_) {
    const int k = 3, s = 2, pl = pl_t, pr = pr_t;
    const int Tp = T + pl + pr, Fp = F + pl_f + pr_f;
    const int To = (Tp - k) / s + 1, Fo = (Fp - k) / s + 1;
    *To_ = To; *Fo_ = Fo;

    /* full conv with C_in=1 (stage 0, ~80% of the subsampling MACs): im2col
     * P [9, S] + GEMM W [C_out, 9] @ P -> [C_out, S]. Same math as the direct
     * loop (out-of-bounds reads become explicit zeros). */
    if (!depthwise && C_in == 1) {
        const size_t S = (size_t)To * (size_t)Fo;
        float *P = malloc(9u * S * sizeof(float));
        if (P) {
            for (int dt = 0; dt < k; dt++) {
                for (int df = 0; df < k; df++) {
                    float *row = P + (size_t)(dt * 3 + df) * S;
                    for (int to = 0; to < To; to++) {
                        const int t = to * s - pl + dt;
                        float *r = row + (size_t)to * (size_t)Fo;
                        if (t < 0 || t >= T) {
                            memset(r, 0, (size_t)Fo * sizeof(float));
                            continue;
                        }
                        const float *sr = x + (size_t)t * (size_t)F;
                        for (int fo = 0; fo < Fo; fo++) {
                            const int fq = fo * s - pl_f + df;
                            r[fo] = (fq < 0 || fq >= F) ? 0.0f : sr[fq];
                        }
                    }
                }
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C_out, (int)S, 9,
                        1.0f, w, 9, P, (int)S, 0.0f, out, (int)S);
            for (int co = 0; co < C_out; co++) {
                float *dst = out + (size_t)co * S;
                for (size_t i = 0; i < S; i++) dst[i] += b[co];
            }
            free(P);
            return;
        }
        /* malloc fallita: prosegue col loop diretto */
    }

    /* depthwise: explicitly padded channel -> unrolled 3x3 with no branch in the
     * hot loop (the pad zeros make the sum identical to the bounds-checked one).
     * Independent channels -> parallel slices when the work is large
     * (offline); i chunk streaming restano seriali (overhead > lavoro). */
    if (depthwise) {
        dw_par dp = {.x = x, .w = w, .b = b, .out = out, .T = T, .F = F, .Tp = Tp,
                     .Fp2 = F + pl_f + pr_f, .To = To, .Fo = Fo, .pl = pl, .pf = pl_f,
                     .s = s, .C_out = C_out, .slices = 1};
        atomic_init(&dp.failed, 0);
        const size_t work = (size_t)C_out * (size_t)To * (size_t)Fo;
        if (work > 500000)
            dp.slices = mynah_asr_num_threads() < C_out ? mynah_asr_num_threads() : C_out;
        mynah_asr_parallel_for(dp.slices, dw_slice_worker, &dp);
        if (!atomic_load(&dp.failed)) return;
        /* a failed malloc in a worker: fall back to the direct loop */
    }

    for (int co = 0; co < C_out; co++) {
        float *dst = out + (size_t)co * (size_t)To * (size_t)Fo;
        for (int i = 0; i < To * Fo; i++) dst[i] = b[co];
    }
    /* direct loop (3x3 kernel, stride 2) — clear and cache-friendly at these sizes */
    for (int co = 0; co < C_out; co++) {
        const int ci_lo = depthwise ? co : 0;
        const int ci_hi = depthwise ? co + 1 : C_in;
        float *dst = out + (size_t)co * (size_t)To * (size_t)Fo;
        for (int ci = ci_lo; ci < ci_hi; ci++) {
            const float *src = x + (size_t)ci * (size_t)T * (size_t)F;
            const float *ker = depthwise ? w + (size_t)co * 9u
                                         : w + ((size_t)co * (size_t)C_in + (size_t)ci) * 9u;
            for (int to = 0; to < To; to++) {
                const int t0 = to * s - pl;
                for (int fo = 0; fo < Fo; fo++) {
                    const int f0 = fo * s - pl_f;
                    float acc = 0.0f;
                    for (int dt = 0; dt < k; dt++) {
                        const int t = t0 + dt;
                        if (t < 0 || t >= T) continue;
                        for (int df = 0; df < k; df++) {
                            const int fq = f0 + df;
                            if (fq < 0 || fq >= F) continue;
                            acc += ker[dt * 3 + df] * src[(size_t)t * (size_t)F + (size_t)fq];
                        }
                    }
                    dst[(size_t)to * (size_t)Fo + (size_t)fo] += acc;
                }
            }
        }
    }
}

static void relu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

float *mynah_asr_subsampling_forward(const mynah_asr_subsampling *ss, const float *feats,
                                 int T, int n_mels, int *t_out) {
    const int C = ss->channels;

    /* work buffers sized on the first stage (the largest one) */
    int To = 0, Fo = 0;
    const int To_max = (T + 3 - 3) / 2 + 1, Fo_max = (n_mels + 3 - 3) / 2 + 1;
    float *a = malloc((size_t)C * (size_t)To_max * (size_t)Fo_max * sizeof(float));
    float *bbuf = malloc((size_t)C * (size_t)To_max * (size_t)Fo_max * sizeof(float));
    if (!a || !bbuf) { free(a); free(bbuf); return NULL; }

    /* time/freq padding: causal (2,1) or symmetric 'same' (1,1) */
    const int pl = ss->causal ? 2 : 1, pr = 1;

    /* stadio 0: conv piena 1 -> C su [1, T, n_mels] */
    conv2d_s2(feats, 1, T, n_mels, pl, pr, pl, pr,
              (const float *)ss->conv_in_w->data, (const float *)ss->conv_in_b->data,
              C, 0, a, &To, &Fo);
    relu_inplace(a, (size_t)C * (size_t)To * (size_t)Fo);

    /* stadi 1..2: depthwise + pointwise + ReLU */
    for (int i = 0; i < 2; i++) {
        int To2, Fo2;
        conv2d_s2(a, C, To, Fo, pl, pr, pl, pr,
                  (const float *)ss->dw_w[i]->data, (const float *)ss->dw_b[i]->data,
                  C, 1, bbuf, &To2, &Fo2);
        /* pointwise 1x1: out[co, t, f] = sum_ci w[co, ci] * b[ci, t, f]  (GEMM [C, C] x [C, S]) */
        const size_t S = (size_t)To2 * (size_t)Fo2;
        const float *pw = (const float *)ss->pw_w[i]->data; /* [C, C, 1, 1] -> [C, C] */
        const float *pb = (const float *)ss->pw_b[i]->data;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C, (int)S, C,
                    1.0f, pw, C, bbuf, (int)S, 0.0f, a, (int)S);
        for (int co = 0; co < C; co++) {
            float *row = a + (size_t)co * S;
            for (size_t j = 0; j < S; j++) row[j] += pb[co];
        }
        relu_inplace(a, (size_t)C * S);
        To = To2; Fo = Fo2;
    }

    /* flatten channel-major [T', C*F'] + linear -> d_model */
    const int CF = C * Fo;
    float *flat = malloc((size_t)To * (size_t)CF * sizeof(float));
    float *out = malloc((size_t)To * (size_t)ss->d_model * sizeof(float));
    if (!flat || !out) { free(a); free(bbuf); free(flat); free(out); return NULL; }
    for (int t = 0; t < To; t++)
        for (int c = 0; c < C; c++)
            memcpy(flat + (size_t)t * (size_t)CF + (size_t)c * (size_t)Fo,
                   a + ((size_t)c * (size_t)To + (size_t)t) * (size_t)Fo,
                   (size_t)Fo * sizeof(float));

    /* out = flat @ W^T + b — W [d_model, CF] row-major => GEMM with B transposed */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, To, ss->d_model, CF,
                1.0f, flat, CF, (const float *)ss->lin_w->data, CF, 0.0f, out, ss->d_model);
    const float *lb = (const float *)ss->lin_b->data;
    for (int t = 0; t < To; t++)
        for (int d = 0; d < ss->d_model; d++) out[(size_t)t * (size_t)ss->d_model + (size_t)d] += lb[d];

    free(a); free(bbuf); free(flat);
    *t_out = To;
    return out;
}

/* ------------------------------------------------------------------ streaming */

int mynah_asr_ss_stream_init(mynah_asr_ss_stream *sst, const mynah_asr_subsampling *ss, int n_mels) {
    memset(sst, 0, sizeof(*sst));
    sst->first = 1;
    int F = n_mels;
    for (int s = 0; s < 3; s++) {
        sst->cin[s] = (s == 0) ? 1 : ss->channels;
        sst->fdim[s] = F;
        sst->cache[s] = calloc((size_t)sst->cin[s] * (size_t)F, sizeof(float));
        if (!sst->cache[s]) return -1;
        F = F / 2 + 1; /* (F + 3 - 3)/2 + 1 */
    }
    return 0;
}

void mynah_asr_ss_stream_free(mynah_asr_ss_stream *sst) {
    for (int s = 0; s < 3; s++) { free(sst->cache[s]); sst->cache[s] = NULL; }
}

/* Antepone [init?1:0 zeri][cache 1] al chunk sull'asse tempo, conv valida s2,
 * updates cache = last frame of the chunk. x [C, T, F] -> out [C_out, To, Fo]. */
static void stream_stage(const float *x, int C_in, int T, int F, int first, int last,
                         float *cache, const float *w, const float *b, int C_out,
                         int depthwise, float *out, int *To_, int *Fo_) {
    const int lp = first ? 2 : 1; /* cache(1) + init_pad(1) al primo chunk */
    const int rp = last ? 1 : 0;  /* last chunk: causal right-pad, as offline */
    const int Tp = T + lp + rp;
    float *xp = malloc((size_t)C_in * (size_t)Tp * (size_t)F * sizeof(float));
    if (!xp) { *To_ = 0; *Fo_ = 0; return; }
    for (int c = 0; c < C_in; c++) {
        float *dst = xp + (size_t)c * (size_t)Tp * (size_t)F;
        memset(dst, 0, (size_t)(lp - 1) * (size_t)F * sizeof(float));
        memcpy(dst + (size_t)(lp - 1) * (size_t)F, cache + (size_t)c * (size_t)F,
               (size_t)F * sizeof(float));
        memcpy(dst + (size_t)lp * (size_t)F, x + (size_t)c * (size_t)T * (size_t)F,
               (size_t)T * (size_t)F * sizeof(float));
        if (rp) memset(dst + (size_t)(lp + T) * (size_t)F, 0, (size_t)F * sizeof(float));
    }
    /* refresh the cache with the last frame of the chunk */
    for (int c = 0; c < C_in; c++)
        memcpy(cache + (size_t)c * (size_t)F,
               x + ((size_t)c * (size_t)T + (size_t)(T - 1)) * (size_t)F,
               (size_t)F * sizeof(float));

    conv2d_s2(xp, C_in, Tp, F, 0, 0, 2, 1, w, b, C_out, depthwise, out, To_, Fo_);
    free(xp);
}

int mynah_asr_ss_stream_step(const mynah_asr_subsampling *ss, mynah_asr_ss_stream *sst,
                         const float *mel, int n_mel, int n_mels, int is_last, float *out) {
    const int C = ss->channels;
    int To = 0, Fo = 0;
    const int cap_t = n_mel / 2 + 2, cap_f = n_mels / 2 + 2;
    float *a = malloc((size_t)C * (size_t)cap_t * (size_t)cap_f * sizeof(float));
    float *bbuf = malloc((size_t)C * (size_t)cap_t * (size_t)cap_f * sizeof(float));
    if (!a || !bbuf) { free(a); free(bbuf); return -1; }

    stream_stage(mel, 1, n_mel, n_mels, sst->first, is_last, sst->cache[0],
                 (const float *)ss->conv_in_w->data, (const float *)ss->conv_in_b->data,
                 C, 0, a, &To, &Fo);
    relu_inplace(a, (size_t)C * (size_t)To * (size_t)Fo);

    for (int i = 0; i < 2; i++) {
        int To2, Fo2;
        stream_stage(a, C, To, Fo, sst->first, is_last, sst->cache[i + 1],
                     (const float *)ss->dw_w[i]->data, (const float *)ss->dw_b[i]->data,
                     C, 1, bbuf, &To2, &Fo2);
        const size_t S = (size_t)To2 * (size_t)Fo2;
        const float *pw = (const float *)ss->pw_w[i]->data;
        const float *pb = (const float *)ss->pw_b[i]->data;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C, (int)S, C,
                    1.0f, pw, C, bbuf, (int)S, 0.0f, a, (int)S);
        for (int co = 0; co < C; co++) {
            float *row = a + (size_t)co * S;
            for (size_t j = 0; j < S; j++) row[j] += pb[co];
        }
        relu_inplace(a, (size_t)C * S);
        To = To2; Fo = Fo2;
    }
    sst->first = 0;

    /* flatten channel-major + linear */
    const int CF = C * Fo;
    float *flat = malloc((size_t)To * (size_t)CF * sizeof(float));
    if (!flat) { free(a); free(bbuf); return -1; }
    for (int t = 0; t < To; t++)
        for (int c = 0; c < C; c++)
            memcpy(flat + (size_t)t * (size_t)CF + (size_t)c * (size_t)Fo,
                   a + ((size_t)c * (size_t)To + (size_t)t) * (size_t)Fo,
                   (size_t)Fo * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, To, ss->d_model, CF,
                1.0f, flat, CF, (const float *)ss->lin_w->data, CF, 0.0f, out, ss->d_model);
    const float *lb = (const float *)ss->lin_b->data;
    for (int t = 0; t < To; t++)
        for (int d = 0; d < ss->d_model; d++) out[(size_t)t * (size_t)ss->d_model + (size_t)d] += lb[d];

    free(a); free(bbuf); free(flat);
    return To;
}
