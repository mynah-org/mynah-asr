#include "decoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MYNAH_ASR_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

int mynah_asr_decoder_init(mynah_asr_decoder *dec, const mynah_asr_safetensors *st,
                       int blank, int max_symbols, int quantize,
                       const int *durations, int n_durations) {
    memset(dec, 0, sizeof(*dec));
    if (n_durations > MYNAH_ASR_MAX_DURATIONS) return -1;
    dec->n_durations = n_durations;
    for (int i = 0; i < n_durations; i++) dec->durations[i] = durations[i];
    const mynah_asr_tensor *emb = mynah_asr_st_get(st, "decoder.embedding.weight");
    const mynah_asr_tensor *proj_w = mynah_asr_st_get(st, "decoder.decoder_projector.weight");
    const mynah_asr_tensor *proj_b = mynah_asr_st_get(st, "decoder.decoder_projector.bias");
    const mynah_asr_tensor *head_b = mynah_asr_st_get(st, "joint.head.bias");
    if (!emb || !proj_w || !proj_b || !head_b) return -1;

    dec->embedding = (const float *)emb->data;
    dec->proj_w = (const float *)proj_w->data;
    dec->proj_b = (const float *)proj_b->data;
    dec->head_b = (const float *)head_b->data;
    dec->vocab = (int)emb->shape[0];
    dec->hidden = (int)emb->shape[1];
    if (mynah_asr_qmat_init_st(&dec->head, st, "joint.head.weight", quantize) != 0) return -1;
    dec->blank = blank;
    dec->max_symbols = max_symbols;

    int n = 0;
    for (; n < MYNAH_ASR_MAX_PRED_LAYERS; n++) {
        char name[64];
        snprintf(name, sizeof(name), "decoder.lstm.weight_ih_l%d", n);
        const mynah_asr_tensor *w = mynah_asr_st_get(st, name);
        if (!w) break;
        dec->w_ih[n] = (const float *)w->data;
        /* partial layer (corrupt checkpoint): clean error instead of a null deref */
        const mynah_asr_tensor *whh, *bih, *bhh;
        snprintf(name, sizeof(name), "decoder.lstm.weight_hh_l%d", n);
        whh = mynah_asr_st_get(st, name);
        snprintf(name, sizeof(name), "decoder.lstm.bias_ih_l%d", n);
        bih = mynah_asr_st_get(st, name);
        snprintf(name, sizeof(name), "decoder.lstm.bias_hh_l%d", n);
        bhh = mynah_asr_st_get(st, name);
        if (!whh || !bih || !bhh) {
            fprintf(stderr, "mynah-asr: LSTM layer %d is incomplete in the checkpoint\n", n);
            return -1;
        }
        dec->w_hh[n] = (const float *)whh->data;
        dec->b_ih[n] = (const float *)bih->data;
        dec->b_hh[n] = (const float *)bhh->data;
    }
    dec->n_layers = n;
    return (n > 0 && dec->hidden <= 1024) ? 0 : -1;
}

void mynah_asr_dec_state_reset(const mynah_asr_decoder *dec, mynah_asr_dec_state *s) {
    memset(s, 0, sizeof(*s));
    (void)dec;
    s->last_token = -1;
}

static inline float sigmoid_f(float x) { return mynah_asr_sigmoid(x); }

/* One stacked-LSTM step + projector: input = embedding[token]. Updates h/c and s->g. */
static void pred_step(const mynah_asr_decoder *dec, mynah_asr_dec_state *s, int token) {
    const int H = dec->hidden;
    const float *x = dec->embedding + (size_t)token * (size_t)H;
    float z[4 * 1024];

    for (int l = 0; l < dec->n_layers; l++) {
        for (int i = 0; i < 4 * H; i++) z[i] = dec->b_ih[l][i] + dec->b_hh[l][i];
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, H, 1.0f, dec->w_ih[l], H,
                    x, 1, 1.0f, z, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, H, 1.0f, dec->w_hh[l], H,
                    s->h[l], 1, 1.0f, z, 1);
        for (int i = 0; i < H; i++) {
            const float ig = sigmoid_f(z[i]);
            const float fg = sigmoid_f(z[H + i]);
            const float gg = tanhf(z[2 * H + i]);
            const float og = sigmoid_f(z[3 * H + i]);
            s->c[l][i] = fg * s->c[l][i] + ig * gg;
            s->h[l][i] = og * tanhf(s->c[l][i]);
        }
        x = s->h[l];
    }
    /* decoder_projector */
    memcpy(s->g, dec->proj_b, (size_t)H * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, H, H, 1.0f, dec->proj_w, H,
                s->h[dec->n_layers - 1], 1, 1.0f, s->g, 1);
    s->last_token = token;
}

/* BLOCKED greedy: g changes only on a non-blank emission, so the frames of a
 * blank run share the same g and the joint head (the dominant cost: a V x H
 * matvec per frame) batches into ONE GEMM [B, V] — the same weight matrix read
 * once per block instead of once per frame. The block is adaptive (it doubles
 * over blank runs and restarts short after an emission) so it does not waste
 * rows past the first emitting frame. Semantics identical to the per-frame loop
 * (the inner loop over an emitting frame stays scalar). */
#define DEC_BMAX 32

/* argmax of (logits + bias) on the fly: same float arithmetic as summing into a
 * buffer and then searching for the maximum (v computed once per k), without the
 * V float writes/re-reads per row. */
static int argmax_bias(const float *lg, const float *bias, int V) {
    int best = 0;
    float bv = lg[0] + bias[0];
    for (int k = 1; k < V; k++) {
        const float v = lg[k] + bias[k];
        if (v > bv) { bv = v; best = k; }
    }
    return best;
}

/* TDT greedy (ParakeetTDTGenerationMixin / NeMo GreedyTDTInfer): the head emits
 * [vocab | n_durations] logits; at EVERY step the frame advances by the predicted
 * duration (argmax over the last ND logits) — a blank with duration 0 advances by
 * 1, a non-blank with duration 0 re-emits on the same frame (guarded by
 * max_symbols). No blocking over blank runs: TDT already skips frames (dur>1) and
 * the visited grid is not contiguous. */
static int greedy_decode_tdt(const mynah_asr_decoder *dec, mynah_asr_dec_state *s,
                             const float *enc, int T, int *tokens, int *frames, int cap) {
    const int H = dec->hidden, V = dec->vocab, ND = dec->n_durations;
    const int VL = V + ND;
    float joint[1024];
    float *logits = malloc((size_t)VL * sizeof(float));
    if (!logits) return 0;

    /* f32 head for the BLAS GEMM (deterministic across backends); when quantized
     * and T is large it is dequantized once per call, like the RNNT path */
    const float *W = dec->head.qtype == MYNAH_ASR_Q_F32 ? dec->head.f32 : NULL;
    float *wd = NULL;
    if (!W && T > 16) {
        wd = malloc((size_t)VL * (size_t)H * sizeof(float));
        if (wd) { mynah_asr_qmat_dequant(&dec->head, wd); W = wd; }
    }

    if (s->last_token < 0) pred_step(dec, s, dec->blank); /* SOS = blank, stato zero */

    int n_out = 0, t = 0, emitted_here = 0;
    while (t < T) {
        const float *e = enc + (size_t)t * (size_t)H;
        for (int i = 0; i < H; i++) {
            const float v = e[i] + s->g[i];
            joint[i] = v > 0.0f ? v : 0.0f;                /* ReLU */
        }
        if (W)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, VL, H,
                        1.0f, joint, H, W, H, 0.0f, logits, VL);
        else
            mynah_asr_qmat_mul(&dec->head, joint, logits, 1);

        const int k = argmax_bias(logits, dec->head_b, V);
        int dur = dec->durations[argmax_bias(logits + V, dec->head_b + V, ND)];
        if (k != dec->blank) {
            if (n_out < cap) {
                if (frames) frames[n_out] = (int)(s->t_abs + t);
                tokens[n_out++] = k;
            }
            pred_step(dec, s, k);                          /* stato avanza solo su emit */
            emitted_here++;
            if (dur == 0 && emitted_here >= dec->max_symbols)
                dur = 1;                                   /* sblocca il frame (NeMo) */
        } else if (dur == 0) {
            dur = 1;                                       /* blank: avanzamento minimo */
        }
        if (dur > 0) emitted_here = 0;
        t += dur;
    }
    s->t_abs += T;
    free(logits); free(wd);
    return n_out;
}

int mynah_asr_greedy_decode(const mynah_asr_decoder *dec, mynah_asr_dec_state *s,
                        const float *enc, int T, int *tokens, int *frames, int cap) {
    if (dec->n_durations > 0)
        return greedy_decode_tdt(dec, s, enc, T, tokens, frames, cap);
    const int H = dec->hidden, V = dec->vocab;
    float joint[1024];
    float *jin = malloc((size_t)DEC_BMAX * (size_t)H * sizeof(float));
    float *logits = malloc((size_t)DEC_BMAX * (size_t)V * sizeof(float));
    if (!jin || !logits) { free(jin); free(logits); return 0; }
    int n_out = 0;

    /* The head as an f32 matrix for a direct BLAS GEMM (CPU: deterministic across
     * backends — decoding NEVER goes through the GPU). When quantized and T is
     * large (offline) it is dequantized ONCE per call: 33 MB read/written once
     * against the int8 matrix re-read for every frame. For small T (streaming
     * chunks) qmat's per-frame quantized dot stays. */
    const float *W = dec->head.qtype == MYNAH_ASR_Q_F32 ? dec->head.f32 : NULL;
    float *wd = NULL;
    if (!W && T > 16) {
        wd = malloc((size_t)V * (size_t)H * sizeof(float));
        if (wd) { mynah_asr_qmat_dequant(&dec->head, wd); W = wd; }
    }

    if (s->last_token < 0) pred_step(dec, s, dec->blank); /* SOS = blank, stato zero */

    int t = 0, B = 4;
    while (t < T) {
        const int Bc = (T - t) < B ? (T - t) : B;
        for (int b = 0; b < Bc; b++) {
            const float *e = enc + (size_t)(t + b) * (size_t)H;
            float *ji = jin + (size_t)b * (size_t)H;
            for (int i = 0; i < H; i++) {
                const float v = e[i] + s->g[i];
                ji[i] = v > 0.0f ? v : 0.0f;               /* ReLU */
            }
        }
        if (W)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Bc, V, H,
                        1.0f, jin, H, W, H, 0.0f, logits, V);
        else
            mynah_asr_qmat_mul(&dec->head, jin, logits, Bc);

        int first = -1, best = -1;
        for (int b = 0; b < Bc; b++) {
            const int am = argmax_bias(logits + (size_t)b * (size_t)V, dec->head_b, V);
            if (am != dec->blank) { first = b; best = am; break; }
        }
        if (first < 0) {                                   /* run di soli blank */
            t += Bc;
            if (B < DEC_BMAX) B *= 2;
            continue;
        }

        /* frame t+first emits: scalar inner loop (the first emission is already known) */
        t += first;
        const float *e = enc + (size_t)t * (size_t)H;
        for (int emitted = 0; emitted < dec->max_symbols; emitted++) {
            if (emitted > 0) {                             /* iterazione 0: dal batch */
                for (int i = 0; i < H; i++) {
                    const float v = e[i] + s->g[i];
                    joint[i] = v > 0.0f ? v : 0.0f;
                }
                if (W)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, V, H,
                                1.0f, joint, H, W, H, 0.0f, logits, V);
                else
                    mynah_asr_qmat_mul(&dec->head, joint, logits, 1);
                best = argmax_bias(logits, dec->head_b, V);
                if (best == dec->blank) break;
            }
            if (n_out < cap) {
                if (frames) frames[n_out] = (int)(s->t_abs + t);
                tokens[n_out++] = best;
            }
            pred_step(dec, s, best);                       /* stato avanza solo su emit */
        }
        t++;
        B = 4;                                             /* riparte corto dopo un emit */
    }
    s->t_abs += T;
    free(jin); free(logits); free(wd);
    return n_out;
}
