#include "decoder_ctc.h"

#include <stdlib.h>

#ifdef MYNAH_ASR_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

int mynah_asr_ctc_init(mynah_asr_ctc *c, const mynah_asr_safetensors *st) {
    c->w = NULL;
    c->b = NULL;
    const mynah_asr_tensor *w = mynah_asr_st_get(st, "ctc_head.weight");
    const mynah_asr_tensor *b = mynah_asr_st_get(st, "ctc_head.bias");
    if (!w || !b) return -1;
    c->w = (const float *)w->data;      /* [V, d, 1] (conv1d k=1) == linear [V, d] */
    c->b = (const float *)b->data;
    c->vocab = (int)w->shape[0];
    c->d_in = (int)w->shape[1];
    return 0;
}

int mynah_asr_ctc_scores(const mynah_asr_ctc *c, const float *enc_out, int T, float *out) {
    if (!c || !c->w || !enc_out || !out || T <= 0) return -1;
    const int V = c->vocab, d = c->d_in;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, V, d,
                1.0f, enc_out, d, c->w, d, 0.0f, out, V);
    for (int t = 0; t < T; t++) {
        float *row = out + (size_t)t * (size_t)V;
        for (int k = 0; k < V; k++) row[k] += c->b[k];
    }
    return 0;
}

int mynah_asr_ctc_decode(const mynah_asr_ctc *c, const float *enc_out, int T,
                     int *tokens, int *frames, int cap) {
    const int V = c->vocab, d = c->d_in, blank = V - 1;
    float *logits = malloc((size_t)T * (size_t)V * sizeof(float));
    if (!logits) return 0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, V, d,
                1.0f, enc_out, d, c->w, d, 0.0f, logits, V);

    int n_out = 0, prev = -1;
    for (int t = 0; t < T; t++) {
        const float *lg = logits + (size_t)t * (size_t)V;
        int best = 0;
        float bv = lg[0] + c->b[0];
        for (int k = 1; k < V; k++) {
            const float v = lg[k] + c->b[k];
            if (v > bv) { bv = v; best = k; }
        }
        if (best != prev && best != blank && n_out < cap) {
            if (frames) frames[n_out] = t;
            tokens[n_out++] = best;
        }
        prev = best;
    }
    free(logits);
    return n_out;
}
