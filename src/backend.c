#include "backend.h"
#include "qmat.h"
#include "sgemm.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* THE THREE PROVIDERS, and the one place they are chosen.
 *
 * MYNAH_ASR_ACCELERATE is "the Accelerate framework is linked" — true for
 * every macOS build, because Metal and vForce need it.  It is deliberately NOT
 * the same question as "cblas_sgemm is the f32 provider": keying a fast path
 * on the framework when what it needed was a GEMM (or the other way round) is
 * how a default flip silently switches something off on the production target.
 * MYNAH_ASR_BLAS_ACCELERATE / MYNAH_ASR_BLAS_OPENBLAS are the provider; when
 * neither is defined, src/sgemm.c is, and no cblas symbol is referenced
 * anywhere in the binary. */
#ifdef MYNAH_ASR_ACCELERATE
#include <Accelerate/Accelerate.h>   /* vForce (vvexpf) in every macOS build */
#endif
#if defined(MYNAH_ASR_BLAS_ACCELERATE)
#define MYNAH_ASR_GEMM_CBLAS 1
#define MYNAH_ASR_GEMM_PROVIDER_NAME "accelerate"
#elif defined(MYNAH_ASR_BLAS_OPENBLAS)
#include <cblas.h>
#define MYNAH_ASR_GEMM_CBLAS 1
#define MYNAH_ASR_GEMM_PROVIDER_NAME "openblas"
#else
#define MYNAH_ASR_GEMM_CBLAS 0
#define MYNAH_ASR_GEMM_PROVIDER_NAME "own"
#endif

const char *mynah_asr_gemm_provider(void) { return MYNAH_ASR_GEMM_PROVIDER_NAME; }

#ifdef MYNAH_ASR_METAL
int mynah_asr_metal_available(void);
int mynah_asr_metal_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);
int mynah_asr_metal_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                       float *out, int T, int k);
int mynah_asr_metal_gemm3_wt(const float *x, const float *wa, const float *wb,
                         const float *wc, float *oa, float *ob, float *oc,
                         int T, int n, int k);
#endif
#ifdef MYNAH_ASR_CUDA
int mynah_asr_cuda_available(void);
int mynah_asr_cuda_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);
#endif

static int g_backend = MYNAH_ASR_BACKEND_CPU;

/* below this row threshold the GEMM stays on CPU: the GPU round-trip does not pay */
#define METAL_MIN_T 24

int mynah_asr_set_backend(const char *name) {
    if (name && strcmp(name, "cuda") == 0) {
#ifdef MYNAH_ASR_CUDA
        if (mynah_asr_cuda_available()) {
            g_backend = MYNAH_ASR_BACKEND_CUDA;
            fprintf(stderr, "mynah-asr: CUDA backend active (large GEMMs on GPU, T>=%d)\n",
                    METAL_MIN_T);
            return g_backend;
        }
        fprintf(stderr, "mynah-asr: CUDA requested but no device available -> CPU\n");
#else
        fprintf(stderr, "mynah-asr: CUDA requested but not compiled in (make cuda) -> CPU\n");
#endif
        g_backend = MYNAH_ASR_BACKEND_CPU;
        return g_backend;
    }
    if (name && strcmp(name, "metal") == 0) {
#ifdef MYNAH_ASR_METAL
        if (mynah_asr_metal_available()) {
            g_backend = MYNAH_ASR_BACKEND_METAL;
            fprintf(stderr, "mynah-asr: Metal backend active (large GEMMs on GPU, T>=%d)\n",
                    METAL_MIN_T);
            return g_backend;
        }
        fprintf(stderr, "mynah-asr: Metal requested but no device available -> CPU\n");
#else
        fprintf(stderr, "mynah-asr: Metal requested but not compiled into this build -> CPU\n");
#endif
    }
    g_backend = MYNAH_ASR_BACKEND_CPU;
    return g_backend;
}

int mynah_asr_backend(void) { return g_backend; }

/* The seam.  `own` maps 1:1 onto mynah_asr_sgemm_f32, which follows the same
 * argument contract on purpose, so this is a type conversion and not a
 * translation layer. */
void mynah_asr_gemm_f32(int trans_a, int trans_b, int m, int n, int k,
                        float alpha, const float *a, int lda,
                        const float *b, int ldb, float beta, float *c, int ldc) {
#if MYNAH_ASR_GEMM_CBLAS
    cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans,
                trans_b ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b,
                ldb, beta, c, ldc);
#else
    mynah_asr_sgemm_f32(trans_a, trans_b, (size_t)m, (size_t)n, (size_t)k, alpha,
                        a, (size_t)lda, b, (size_t)ldb, beta, c, (size_t)ldc);
#endif
}

/* GEMV.  On `own` it is written as a ONE-ROW GEMM rather than as a fourth
 * kernel, and the transpose flag decides which family serves it:
 *
 *   trans == 0  y^T = x^T * A^T  -> trans_b, the DOT family: one vectorized
 *               dot per output element over a contiguous row of A.
 *   trans != 0  y^T = x^T * A    -> no transpose, the MATVEC family: A's
 *               columns are contiguous, so it vectorizes across the output.
 *
 * Either way the reduction is over the whole of the contracted dimension in
 * one invocation, so the answer does not move with the pool width. */
void mynah_asr_gemv_f32(int trans, int rows, int cols, float alpha,
                        const float *a, int lda, const float *x,
                        float beta, float *y) {
#if MYNAH_ASR_GEMM_CBLAS
    cblas_sgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans, rows, cols,
                alpha, a, lda, x, 1, beta, y, 1);
#else
    if (trans)
        mynah_asr_sgemm_f32(0, 0, 1, (size_t)cols, (size_t)rows, alpha, x,
                            (size_t)rows, a, (size_t)lda, beta, y, (size_t)cols);
    else
        mynah_asr_sgemm_f32(0, 1, 1, (size_t)rows, (size_t)cols, alpha, x,
                            (size_t)cols, a, (size_t)lda, beta, y, (size_t)rows);
#endif
}

void mynah_asr_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k) {
#ifdef MYNAH_ASR_METAL
    if (g_backend == MYNAH_ASR_BACKEND_METAL &&
        mynah_asr_metal_gemm_wt(x, w, out, T, n, k) == 0)
        return;
#endif
#ifdef MYNAH_ASR_CUDA
    if (g_backend == MYNAH_ASR_BACKEND_CUDA && T >= METAL_MIN_T &&
        mynah_asr_cuda_gemm_wt(x, w, out, T, n, k) == 0)
        return;
#endif
    mynah_asr_gemm_f32(0, 1, T, n, k, 1.0f, x, k, w, k, 0.0f, out, n);
}

/* In-place SiLU. On Accelerate: vvexpf (vForce) batches the exp — in the
 * 2026-07-19 profile scalar expf accounted for ~10% of the samples. The clamp at
 * 87 avoids inf (the -ffast-math lesson: inf = UB, see mynah_asr_sigmoid);
 * exp(-x) with the clamp can never overflow in f32. The scalar fallback is
 * unchanged.
 *
 * Gated on MYNAH_ASR_ACCELERATE, the FRAMEWORK, not on the BLAS provider: this
 * is vForce, not cblas, and BLAS=none on macOS must not silently change the
 * activation's arithmetic. Otherwise an accelerate-vs-none transcript diff
 * would have two possible causes and the A/B would prove nothing. */
void mynah_asr_silu_scratch(float *x, size_t n, float *scratch) {
#ifdef MYNAH_ASR_ACCELERATE
    if (n >= 256) {
        /* caller scratch (>= n floats) keeps the streaming step allocation-free;
         * NULL = allocate here, as before */
        float *t = scratch;
        const int owned = t == NULL;
        if (owned) t = malloc(n * sizeof(float));
        if (t) {
            for (size_t i = 0; i < n; i++) {
                const float v = -x[i];
                t[i] = v > 87.0f ? 87.0f : v;
            }
            for (size_t off = 0; off < n; off += (size_t)1 << 30) {
                const int chunk = (int)(n - off > (size_t)1 << 30 ? (size_t)1 << 30 : n - off);
                vvexpf(t + off, t + off, &chunk);
            }
            for (size_t i = 0; i < n; i++) x[i] = x[i] / (1.0f + t[i]);
            if (owned) free(t);
            return;
        }
    }
#else
    (void)scratch;
#endif
    for (size_t i = 0; i < n; i++) x[i] = x[i] * mynah_asr_sigmoid(x[i]);
}

void mynah_asr_silu(float *x, size_t n) { mynah_asr_silu_scratch(x, n, NULL); }

void mynah_asr_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                  float *out, int T, int k, float *scratch) {
#ifdef MYNAH_ASR_METAL
    if (g_backend == MYNAH_ASR_BACKEND_METAL &&
        mynah_asr_metal_ffn_wt(x, w1, n1, w2, n2, out, T, k) == 0)
        return;
#endif
    mynah_asr_gemm_wt(x, w1, scratch, T, n1, k);
    mynah_asr_silu(scratch, (size_t)T * (size_t)n1);
    mynah_asr_gemm_wt(scratch, w2, out, T, n2, n1);
}

void mynah_asr_gemm3_wt(const float *x, const float *wa, const float *wb, const float *wc,
                    float *oa, float *ob, float *oc, int T, int n, int k) {
#ifdef MYNAH_ASR_METAL
    if (g_backend == MYNAH_ASR_BACKEND_METAL &&
        mynah_asr_metal_gemm3_wt(x, wa, wb, wc, oa, ob, oc, T, n, k) == 0)
        return;
#endif
    mynah_asr_gemm_wt(x, wa, oa, T, n, k);
    mynah_asr_gemm_wt(x, wb, ob, T, n, k);
    mynah_asr_gemm_wt(x, wc, oc, T, n, k);
}
