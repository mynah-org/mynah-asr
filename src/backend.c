#include "backend.h"
#include "qmat.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#ifdef MYNAH_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#ifdef MYNAH_METAL
int mynah_metal_available(void);
int mynah_metal_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);
int mynah_metal_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                       float *out, int T, int k);
int mynah_metal_gemm3_wt(const float *x, const float *wa, const float *wb,
                         const float *wc, float *oa, float *ob, float *oc,
                         int T, int n, int k);
#endif
#ifdef MYNAH_CUDA
int mynah_cuda_available(void);
int mynah_cuda_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);
#endif

static int g_backend = MYNAH_BACKEND_CPU;

/* sotto questa soglia di righe la GEMM resta su CPU: il round-trip GPU non paga */
#define METAL_MIN_T 24

int mynah_set_backend(const char *name) {
    if (name && strcmp(name, "cuda") == 0) {
#ifdef MYNAH_CUDA
        if (mynah_cuda_available()) {
            g_backend = MYNAH_BACKEND_CUDA;
            fprintf(stderr, "mynah: backend CUDA attivo (GEMM grandi su GPU, T>=%d)\n",
                    METAL_MIN_T);
            return g_backend;
        }
        fprintf(stderr, "mynah: CUDA requested but no device available -> CPU\n");
#else
        fprintf(stderr, "mynah: CUDA requested but not compiled in (make cuda) -> CPU\n");
#endif
        g_backend = MYNAH_BACKEND_CPU;
        return g_backend;
    }
    if (name && strcmp(name, "metal") == 0) {
#ifdef MYNAH_METAL
        if (mynah_metal_available()) {
            g_backend = MYNAH_BACKEND_METAL;
            fprintf(stderr, "mynah: backend Metal attivo (GEMM grandi su GPU, T>=%d)\n",
                    METAL_MIN_T);
            return g_backend;
        }
        fprintf(stderr, "mynah: Metal requested but no device available -> CPU\n");
#else
        fprintf(stderr, "mynah: Metal requested but not compiled into this build -> CPU\n");
#endif
    }
    g_backend = MYNAH_BACKEND_CPU;
    return g_backend;
}

int mynah_backend(void) { return g_backend; }

void mynah_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k) {
#ifdef MYNAH_METAL
    if (g_backend == MYNAH_BACKEND_METAL &&
        mynah_metal_gemm_wt(x, w, out, T, n, k) == 0)
        return;
#endif
#ifdef MYNAH_CUDA
    if (g_backend == MYNAH_BACKEND_CUDA && T >= METAL_MIN_T &&
        mynah_cuda_gemm_wt(x, w, out, T, n, k) == 0)
        return;
#endif
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, n, k,
                1.0f, x, k, w, k, 0.0f, out, n);
}

/* SiLU in-place. Su Accelerate: vvexpf (vForce) batcha l'exp — dal profilo
 * 2026-07-19 expf scalare pesava ~10% dei sample. Il clamp a 87 evita inf
 * (lezione -ffast-math: inf = UB, vedi mynah_sigmoid); exp(-x) con clamp
 * non overflowa mai in f32. Fallback scalare identico a prima. */
void mynah_silu(float *x, size_t n) {
#ifdef MYNAH_BLAS_ACCELERATE
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
    for (size_t i = 0; i < n; i++) x[i] = x[i] * mynah_sigmoid(x[i]);
}

void mynah_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                  float *out, int T, int k, float *scratch) {
#ifdef MYNAH_METAL
    if (g_backend == MYNAH_BACKEND_METAL &&
        mynah_metal_ffn_wt(x, w1, n1, w2, n2, out, T, k) == 0)
        return;
#endif
    mynah_gemm_wt(x, w1, scratch, T, n1, k);
    mynah_silu(scratch, (size_t)T * (size_t)n1);
    mynah_gemm_wt(scratch, w2, out, T, n2, n1);
}

void mynah_gemm3_wt(const float *x, const float *wa, const float *wb, const float *wc,
                    float *oa, float *ob, float *oc, int T, int n, int k) {
#ifdef MYNAH_METAL
    if (g_backend == MYNAH_BACKEND_METAL &&
        mynah_metal_gemm3_wt(x, wa, wb, wc, oa, ob, oc, T, n, k) == 0)
        return;
#endif
    mynah_gemm_wt(x, wa, oa, T, n, k);
    mynah_gemm_wt(x, wb, ob, T, n, k);
    mynah_gemm_wt(x, wc, oc, T, n, k);
}
