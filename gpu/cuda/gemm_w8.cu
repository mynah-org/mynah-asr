/* gpu/cuda/gemm_w8.cu — C[M,N] = A[M,K] · (s[n] * Q[N,K])^T, weight-only int8
 * (--weights int8), row-stable by construction.
 *
 * Q is the per-row symmetric int8 the CPU path uses (mynah_asr_quantize_int8:
 * s[n] = max|W[n,:]| / 127, Q = round-half-away(W / s)), so the GPU and the CPU
 * int8 packs hold the same codes. Activations stay f32: element (m, n) is ONE
 * fmaf chain acc = fmaf(A[m][k], (float)Q[n][k], acc) over k = 0..K-1 in
 * order (the v1 tiling of gemm.cu, GEMM_BK = 16, same zero padding), then
 * acc * s[n], then + bias, + C, activation. Nothing about the tiling or M can
 * change a bit, so a stream's output does not depend on its cohort (contract
 * 4). It is NOT bit-identical to the f32 kernels: a numerical change gated by
 * the bank WER (gpu/tools/transcript_ab.py).
 *
 * What it buys: the weights are a quarter of the bytes in VRAM and on every
 * pass's weight reads (the label loop's head GEMM [V, H] is read once per
 * iteration at small M, where it is bandwidth-bound). */
#include "kernels.cuh"

#include <math.h>
#include <stdint.h>

#define W8_BK 16
#define W8_BM 64
#define W8_BN 64
#define W8_THREADS 256

__device__ __forceinline__ float w8_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__global__ void __launch_bounds__(W8_THREADS)
gemm_w8_kernel(const float *__restrict__ A, int lda, const int8_t *__restrict__ Q,
               const float *__restrict__ S, const float *__restrict__ bias, float *__restrict__ C,
               int ldc, int M, int N, int K, int accumulate, int act) {
    __shared__ float As[W8_BK][W8_BM + 4];
    __shared__ float Ws[W8_BK][W8_BN + 4];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * W8_BM, n0 = blockIdx.x * W8_BN;

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    const int lrow = tid >> 2, lk = (tid & 3) * 4;
    const bool k4 = (K & 3) == 0;

    for (int k0 = 0; k0 < K; k0 += W8_BK) {
        {
            const int m = m0 + lrow;
            const float *src = A + (size_t)m * (size_t)lda + k0 + lk;
            const bool rin = m < M;
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int k = k0 + lk + u;
                As[lk + u][lrow] = (rin && k < K) ? src[u] : 0.0f;
            }
        }
        {
            const int n = n0 + lrow;
            const int8_t *src = Q + (size_t)n * (size_t)K + k0 + lk;
            if (n < N && k4 && k0 + lk + 3 < K) {
                const char4 c = *(const char4 *)src;
                Ws[lk + 0][lrow] = (float)c.x; Ws[lk + 1][lrow] = (float)c.y;
                Ws[lk + 2][lrow] = (float)c.z; Ws[lk + 3][lrow] = (float)c.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int k = k0 + lk + u;
                    Ws[lk + u][lrow] = (n < N && k < K) ? (float)src[u] : 0.0f;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < W8_BK; k++) {
            float a[4], w[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[k][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; j++) w[j] = Ws[k][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], w[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
        float *crow = C + (size_t)m * (size_t)ldc;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            float v = acc[i][j] * S[n];
            if (bias) v += bias[n];
            if (accumulate) v += crow[n];
            if (act == 1) v = v > 0.0f ? v : 0.0f;
            else if (act == 2) v = v * w8_sigmoid(v);
            crow[n] = v;
        }
    }
}

cudaError_t k_gemm_w8(const float *A, int lda, const int8_t *Q, const float *S, const float *bias,
                      float *C, int ldc, int M, int N, int K, int accumulate, int act,
                      cudaStream_t s) {
    if (M <= 0 || N <= 0) return cudaSuccess;
    if (((uintptr_t)Q & 3) != 0) return cudaErrorInvalidValue;
    dim3 grid((unsigned)((N + W8_BN - 1) / W8_BN), (unsigned)((M + W8_BM - 1) / W8_BM));
    gemm_w8_kernel<<<grid, W8_THREADS, 0, s>>>(A, lda, Q, S, bias, C, ldc, M, N, K, accumulate, act);
    return cudaGetLastError();
}
