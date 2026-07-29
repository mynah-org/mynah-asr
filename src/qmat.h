/* Weight matrix with INT8/INT4 quantization (at load, or pre-quantized on disk).
 * Policy taken from prior art (parakeet.cpp/qwen-tts): ONLY the large linears
 * consumed by the GEMMs are quantized; 2D conv, LSTM, norms, biases and
 * embeddings stay f32.
 *
 * Schemes:
 *  - INT8: symmetric per-row, scales [n]
 *  - INT4: symmetric per-group (32 values), nibbles packed 2-per-byte,
 *          scales [n * k/32] (Q4_0 style)
 *
 * Product dispatch:
 *  - f32: cblas_sgemm
 *  - quantized, small T (streaming/decode): direct dot kernel (bandwidth bound)
 *  - quantized, large T (offline/batch): dequant into scratch + sgemm */
#ifndef MYNAH_ASR_QMAT_H
#define MYNAH_ASR_QMAT_H

#include <stddef.h>
#include <stdint.h>

#include "weights.h"

enum { MYNAH_ASR_Q_F32 = 0, MYNAH_ASR_Q_INT8 = 1, MYNAH_ASR_Q_INT4 = 2 };

#include <math.h>

/* STABLE sigmoid: never expf(large positive argument) -> inf. Under -ffast-math
 * an inf is UB: gcc on x86 vectorizes expf through libmvec and the inf becomes a
 * NaN (seen in CI on 2026-07-18: encoder NaNs on linux x86 only; clang/ARM
 * survived by luck). expf(x) with x <= 0 can never overflow. */
static inline float mynah_asr_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

#define MYNAH_ASR_Q4_GROUP 32

typedef struct {
    const float *f32;       /* original weights [n, k] — used when qtype == F32 */
    const int8_t *q8;       /* INT8 [n, k] */
    const uint8_t *q4;      /* INT4 packed [n, k/2] */
    const float *scales;    /* INT8: [n]; INT4: [n, k/32] */
    void *owned_q, *owned_s;/* buffer nostri (quantizzazione al load), da liberare */
    int qtype;
    int n, k;
} mynah_asr_qmat;

/* Init: look first for the pre-quantized form in the safetensors
 * ("<name>.q8"/"<name>.q4" + "<name>.scales" — zero-copy from the mmap), then for
 * the f32 tensor "<name>" (quantized on the fly when qtype != F32).
 * 0 = ok, -1 = tensor missing. */
int mynah_asr_qmat_init_st(mynah_asr_qmat *m, const mynah_asr_safetensors *st, const char *name,
                       int qtype);

/* Direct init from f32 (used by the quantize tool and by the fallbacks). */
int mynah_asr_qmat_init(mynah_asr_qmat *m, const float *w, int n, int k, int qtype);

void mynah_asr_qmat_free(mynah_asr_qmat *m);

/* out[T, n] = x[T, k] @ W^T (PyTorch linear layout). */
void mynah_asr_qmat_mul(const mynah_asr_qmat *m, const float *x, float *out, int T);

/* FFN: out = SiLU(x @ W1^T) @ W2^T. scratch >= T*w1->n floats.
 * When both are F32 it uses the backend's fused path (Metal: a single GPU sync). */
void mynah_asr_qmat_ffn(const mynah_asr_qmat *w1, const mynah_asr_qmat *w2, const float *x,
                    float *out, int T, float *scratch);

/* q/k/v over the same input; when all F32 it uses the backend's fused multi-GEMM. */
void mynah_asr_qmat_qkv(const mynah_asr_qmat *wq, const mynah_asr_qmat *wk, const mynah_asr_qmat *wv,
                    const float *x, float *oq, float *ok, float *ov, int T);

/* Dequantize the whole matrix into the caller's wd [n, k] f32 buffer (a plain
 * copy when already f32). Lets many calls reuse one f32 GEMM without a
 * per-call dequant (e.g. the joint head in blocked greedy decoding). */
void mynah_asr_qmat_dequant(const mynah_asr_qmat *m, float *wd);

/* Runtime x86 SIMD caps (qwen-tts --caps pattern): "auto" (default, cpuid),
 * "scalar", "avx2", "vnni"; env MYNAH_ASR_CAPS as an alternative to the flag. A
 * level above what the CPU supports is downgraded with a note. Returns the
 * effective level (0 scalar, 1 avx2, 2 vnni). On ARM it is a no-op: NEON/SDOT
 * are compile-time (Apple Silicon always has dotprod). */
int mynah_asr_set_caps(const char *name);

/* Quantize an f32 [n,k] buffer into out_q/out_scales (caller-owned buffers):
 * INT8: out_q [n*k] int8, out_scales [n]
 * INT4: out_q [n*k/2] uint8, out_scales [n*k/32]
 * Used by the `mynah-asr quantize` tool. */
void mynah_asr_quantize_int8(const float *w, int n, int k, int8_t *out_q, float *out_scales);
void mynah_asr_quantize_int4(const float *w, int n, int k, uint8_t *out_q, float *out_scales);

#endif
