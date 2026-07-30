/* CUDA backend for the large GEMMs — pattern from qwen-tts (qwen_tts_cuda.c):
 * a shared RESIDENT WEIGHT CACHE (every weight pointer uploaded to the device
 * ONCE) + a PER-THREAD CONTEXT (cuBLAS handle, stream, I/O buffers in TLS):
 * N concurrent server requests issue GEMMs on independent streams without
 * serializing — before this there was a global mutex around the whole call and
 * multi-request throughput was flat (measured on an A100, 2026-07-20).
 *
 * Validated on hardware 2026-07-20 (A100-SXM4-40GB, CUDA 12.8, Ubuntu 24.04):
 * transcriptions identical to CPU on all 10 supported models, e2e green,
 * RTF in docs/benchmarks.md. Automatic CPU fallback on any error.
 * The TLS contexts live until the thread exits (server: persistent pool); no
 * explicit cleanup, like the BLAS pools.
 *
 * v2a precision (2026-07-20): TF32 math mode by default (tensor cores, f32
 * I/O — MYNAH_ASR_CUDA_TF32=0 for strict f32) and resident bf16 weights with
 * cublasGemmEx, opt-in through MYNAH_ASR_CUDA_BF16=1 (half the VRAM and half
 * the PCIe traffic on x; 8-bit mantissa: see the validation gate in
 * docs/benchmarks.md). */
#ifdef MYNAH_ASR_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_ready = -1;
static int g_bf16 = 0;  /* MYNAH_ASR_CUDA_BF16=1: resident bf16 weights + GemmEx */
static int g_tf32 = 1;  /* MYNAH_ASR_CUDA_TF32=0 turns the TF32 tensor cores off */
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; /* init + weight cache */

typedef struct {
    const void *host_ptr;
    void *dev;   /* f32, or bf16 if g_bf16 */
} wc_ent;
static wc_ent *g_wc;
static int g_wc_n, g_wc_cap;

/* f32 -> bf16 round-to-nearest-even (finite weights: no NaN/inf cases) */
static void f32_to_bf16(const float *src, unsigned short *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned int u;
        memcpy(&u, &src[i], 4);
        u += 0x7FFFu + ((u >> 16) & 1u);
        dst[i] = (unsigned short)(u >> 16);
    }
}

/* per-thread context: handle+stream+buffers. Created lazily on the first GEMM. */
typedef struct {
    cublasHandle_t blas;
    cudaStream_t stream;
    float *in, *out;
    size_t in_cap, out_cap;
    unsigned short *stage;  /* host staging for the x -> bf16 conversion */
    size_t stage_cap;
    int state; /* 0 = to initialize, 1 = ok, -1 = failed (stays on CPU) */
} cu_tls;
static __thread cu_tls t_cu;

extern "C" int mynah_asr_cuda_available(void) {
    pthread_mutex_lock(&g_mu);
    if (g_ready < 0) {
        int n = 0;
        g_ready = cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
        const char *bf = getenv("MYNAH_ASR_CUDA_BF16");
        g_bf16 = bf && *bf == '1';
        const char *tf = getenv("MYNAH_ASR_CUDA_TF32");
        g_tf32 = !(tf && *tf == '0');
        if (g_ready && g_bf16)
            fprintf(stderr, "mynah-asr: CUDA bf16 (resident bf16 weights, GemmEx)\n");
    }
    pthread_mutex_unlock(&g_mu);
    return g_ready == 1;
}

static cu_tls *tls_ctx(void) {
    if (t_cu.state == 0) {
        t_cu.state = cublasCreate(&t_cu.blas) == CUBLAS_STATUS_SUCCESS &&
                     cudaStreamCreate(&t_cu.stream) == cudaSuccess ? 1 : -1;
        if (t_cu.state == 1) {
            cublasSetStream(t_cu.blas, t_cu.stream);
            /* TF32: tensor cores with f32 I/O (10-bit mantissa in the product).
             * Validated on the e2e fixtures; MYNAH_ASR_CUDA_TF32=0 for strict f32. */
            if (g_tf32) cublasSetMathMode(t_cu.blas, CUBLAS_TF32_TENSOR_OP_MATH);
        }
    }
    return t_cu.state == 1 ? &t_cu : NULL;
}

/* upload once, shared across threads: a lock-free lookup is not worth it (the
 * cache is append-only but realloc moves the array) -> a mutex, held only for
 * the cache. With g_bf16 the weight is converted f32->bf16 on upload (half the
 * VRAM). */
static void *weight_dev(const void *w, size_t n_elems) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_wc_n; i++)
        if (g_wc[i].host_ptr == w) {
            void *d = g_wc[i].dev;
            pthread_mutex_unlock(&g_mu);
            return d;
        }
    void *d = NULL;
    int ok = 0;
    if (g_bf16) {
        unsigned short *tmp = (unsigned short *)malloc(n_elems * 2u);
        if (tmp) {
            f32_to_bf16((const float *)w, tmp, n_elems);
            ok = cudaMalloc(&d, n_elems * 2u) == cudaSuccess &&
                 cudaMemcpy(d, tmp, n_elems * 2u, cudaMemcpyHostToDevice) == cudaSuccess;
            free(tmp);
        }
    } else {
        ok = cudaMalloc(&d, n_elems * 4u) == cudaSuccess &&
             cudaMemcpy(d, w, n_elems * 4u, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) {
        if (d) cudaFree(d);
        pthread_mutex_unlock(&g_mu);
        return NULL;
    }
    if (g_wc_n == g_wc_cap) {
        g_wc_cap = g_wc_cap ? g_wc_cap * 2 : 256;
        g_wc = (wc_ent *)realloc(g_wc, (size_t)g_wc_cap * sizeof(wc_ent));
    }
    g_wc[g_wc_n++] = (wc_ent){w, d};
    pthread_mutex_unlock(&g_mu);
    return d;
}

static float *io_dev(float **slot, size_t *cap, size_t bytes) {
    if (!*slot || *cap < bytes) {
        size_t want = *cap ? *cap : (1u << 20);
        while (want < bytes) want *= 2;
        if (*slot) cudaFree(*slot);
        if (cudaMalloc(slot, want) != cudaSuccess) { *slot = NULL; *cap = 0; return NULL; }
        *cap = want;
    }
    return *slot;
}

/* out[T,n] = x[T,k] @ W[n,k]^T. 0 = ok, -1 = fallback CPU. */
extern "C" int mynah_asr_cuda_gemm_wt(const float *x, const float *w, float *out,
                                  int T, int n, int k) {
    if (!mynah_asr_cuda_available()) return -1;
    cu_tls *c = tls_ctx();
    if (!c) return -1;

    const size_t xe = (size_t)T * (size_t)k;
    void *dw = weight_dev(w, (size_t)n * (size_t)k);
    float *dx = io_dev(&c->in, &c->in_cap, xe * (g_bf16 ? 2u : 4u));
    float *dy = io_dev(&c->out, &c->out_cap, (size_t)T * (size_t)n * 4u);
    if (!dw || !dx || !dy) return -1;

    if (g_bf16) {
        /* x -> bf16 on the host: half the bytes over PCIe, tensor cores in GemmEx */
        if (!c->stage || c->stage_cap < xe) {
            size_t want = c->stage_cap ? c->stage_cap : (1u << 19);
            while (want < xe) want *= 2;
            free(c->stage);
            c->stage = (unsigned short *)malloc(want * 2u);
            if (!c->stage) { c->stage_cap = 0; return -1; }
            c->stage_cap = want;
        }
        f32_to_bf16(x, c->stage, xe);
        if (cudaMemcpyAsync(dx, c->stage, xe * 2u,
                            cudaMemcpyHostToDevice, c->stream) != cudaSuccess) return -1;
    } else {
        if (cudaMemcpyAsync(dx, x, xe * 4u,
                            cudaMemcpyHostToDevice, c->stream) != cudaSuccess) return -1;
    }

    /* cuBLAS is column-major: out_rm[T,n] = x_rm[T,k] @ W_rm[n,k]^T is equivalent to
     * out_cm[n,T] = W_cm[k,n]^T @ x_cm[k,T] -> gemm(op=T, op=N, m=n, n=T, k=k) */
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st;
    if (g_bf16)
        st = cublasGemmEx(c->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, T, k, &alpha,
                          dw, CUDA_R_16BF, k, dx, CUDA_R_16BF, k, &beta,
                          dy, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    else
        st = cublasSgemm(c->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, T, k,
                         &alpha, (const float *)dw, k, dx, k, &beta, dy, n);
    if (st != CUBLAS_STATUS_SUCCESS) return -1;
    if (cudaMemcpyAsync(out, dy, (size_t)T * (size_t)n * 4u,
                        cudaMemcpyDeviceToHost, c->stream) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(c->stream) != cudaSuccess) return -1;
    return 0;
}

#endif /* MYNAH_ASR_CUDA */
