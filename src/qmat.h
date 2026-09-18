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
 *  - f32: the sgemm seam (src/backend.c: Accelerate, OpenBLAS or our own)
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
    void *owned_q, *owned_s;/* our buffers (quantized at load), to free */
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

/* ------------------------------------------------- row-stable stacked product
 * Same product, but with the contract the batched stream step needs: output row
 * t is BYTE-IDENTICAL to the same input row multiplied alone. Rows stacked from
 * several streams therefore give exactly what those streams produce on their
 * own, whatever B is (ENGINEERING.md §9: a transcript never depends on batching).
 *
 * Quantized weights: the native int8xint8 per-row dot kernel at ANY T, as a
 * threaded loop over blocks of WEIGHT rows (weight-stationary: each weight row
 * is read once and dotted against every activation row). The accumulation is
 * integer and per row, so it is order-independent by construction — this is
 * what makes the identity exact rather than approximate. The T > QMAT_SMALL_T
 * dequant+sgemm fallback of mynah_asr_qmat_mul is NEVER taken here (it would
 * change the numerics AND malloc per call).
 *
 * F32 weights: one sgemm through the seam (src/backend.c) over the whole
 * [T, k] block. Row stability across M is a property of the PROVIDER, not a
 * guarantee — measure it per platform (tests/test_stream_batch) before relying
 * on it.
 *
 * qx/sx: caller-owned scratch, >= T*m->k int8 and >= T floats, so the call
 * allocates nothing. NULL is allowed and falls back to the per-row path.
 * Returns the path taken: one of MYNAH_ASR_QC_*. */
int mynah_asr_qmat_mul_rows(const mynah_asr_qmat *m, const float *x, float *out, int T,
                        int8_t *qx, float *sx);

/* Which implementation ran, counted per call (ENGINEERING.md §6: a fallback is
 * visible). Cheap relaxed atomics; read with mynah_asr_qmat_counter. */
enum {
    MYNAH_ASR_QC_F32 = 0,      /* the sgemm seam on f32 weights                  */
    MYNAH_ASR_QC_DOT,          /* native int8 per-row dot, serial small-T path   */
    MYNAH_ASR_QC_DOT_ROWS,     /* native int8 per-row dot, weight-stationary     */
    MYNAH_ASR_QC_GENERIC,      /* per-row f32xint8 fallback (no native kernel)   */
    MYNAH_ASR_QC_QGEMM,        /* opt-in MYNAH_ASR_QGEMM=1 threaded int8 GEMM    */
    MYNAH_ASR_QC_DEQUANT,      /* dequant whole matrix + sgemm (mallocs!)        */
    MYNAH_ASR_QC__N
};
unsigned long long mynah_asr_qmat_counter(int which);
void mynah_asr_qmat_counters_reset(void);

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

/* Dispatch predicates OWNED by src/qmat.c (S3-2). src/dispatch.c calls these
 * instead of re-deriving "compiled && supported", which is the guess that lets
 * a report and a README agree and both be wrong. Pure readers, no side effect
 * beyond the same one-time cpuid the first kernel call would do anyway.
 *   mynah_asr_qmat_int8_kernel  "neon-sdot" | "avx512vnni" | "avx2" |
 *                               "neon-f32" | "scalar"
 *   mynah_asr_qmat_int4_kernel  "neon-sdot-q4" | "avx2-q4" | "neon-f32-q4" |
 *                               "scalar"
 *   mynah_asr_qmat_qgemm        1 on, 0 off, -1 no native int8 kernel compiled
 *   mynah_asr_caps_detected/_effective  x86 SIMD level, -1 on a non-x86 build */
const char *mynah_asr_qmat_int8_kernel(void);
const char *mynah_asr_qmat_int4_kernel(void);
int         mynah_asr_qmat_qgemm(void);
int         mynah_asr_caps_detected(void);
int         mynah_asr_caps_effective(void);
const char *mynah_asr_caps_name(int level);

/* Quantize an f32 [n,k] buffer into out_q/out_scales (caller-owned buffers):
 * INT8: out_q [n*k] int8, out_scales [n]
 * INT4: out_q [n*k/2] uint8, out_scales [n*k/32]
 * Used by the `mynah-asr quantize` tool. */
void mynah_asr_quantize_int8(const float *w, int n, int k, int8_t *out_q, float *out_scales);
void mynah_asr_quantize_int4(const float *w, int n, int k, uint8_t *out_q, float *out_scales);

#endif
