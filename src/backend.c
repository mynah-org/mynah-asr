#include "backend.h"
#include "qmat.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#ifdef MYNAH_ASR_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

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
            fprintf(stderr, "mynah-asr: backend CUDA attivo (GEMM grandi su GPU, T>=%d)\n",
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
            fprintf(stderr, "mynah-asr: backend Metal attivo (GEMM grandi su GPU, T>=%d)\n",
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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, n, k,
                1.0f, x, k, w, k, 0.0f, out, n);
}

/* In-place SiLU. On Accelerate: vvexpf (vForce) batches the exp — in the
 * 2026-07-19 profile scalar expf accounted for ~10% of the samples. The clamp at
 * 87 avoids inf (the -ffast-math lesson: inf = UB, see mynah_asr_sigmoid);
 * exp(-x) with the clamp can never overflow in f32. The scalar fallback is
 * unchanged. */
void mynah_asr_silu(float *x, size_t n) {
#ifdef MYNAH_ASR_BLAS_ACCELERATE
    if (n >= 256) {
        float *t = malloc(n * sizeof(float));
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
            free(t);
            return;
        }
    }
#endif
    for (size_t i = 0; i < n; i++) x[i] = x[i] * mynah_asr_sigmoid(x[i]);
}

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
