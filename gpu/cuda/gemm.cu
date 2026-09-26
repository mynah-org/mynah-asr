/* gpu/cuda/gemm.cu — C[M,N] = A[M,K] · W[N,K]^T, row-stable by construction.
 *
 * WHY NOT cuBLAS HERE. Durable contract 4: a stream's transcript never depends
 * on who it was batched with. cuBLAS chooses its kernel (tiling, split-K) per
 * shape, so row i of a product at M = 3 and row i of the same product at
 * M = 200 are not guaranteed the same bits -- the property S1-4 had to MEASURE
 * per BLAS on the CPU. This kernel makes it a property of the code: every output
 * element is one fma chain over k in tile order 0, 16, 32, ..., and the chain
 * does not know M, N or the block it runs in. cuBLAS stays available as the
 * comparison arm (engine.cu, MYNAH_ASR_CUDA_GEMM=cublas) under the same gate.
 *
 * Shape: 64x64 block tile, BK = 16, 256 threads, a 4x4 micro-tile per thread.
 * Both operands are K-contiguous (activations [M,K], PyTorch linear weights
 * [N,K]) so both tiles load coalesced along k and are stored transposed in
 * shared memory. Out-of-range rows/cols are zero-filled on load and skipped on
 * store; a partial last k-tile is zero-filled, and fma(0, 0, acc) == acc. */
#include "kernels.cuh"

#include <math.h>

#define GEMM_BM 64
#define GEMM_BN 64
#define GEMM_BK 16
#define GEMM_THREADS 256

__device__ __forceinline__ float gemm_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__global__ void __launch_bounds__(GEMM_THREADS)
gemm_wt_kernel(const float *__restrict__ A, int lda, const float *__restrict__ W,
               const float *__restrict__ bias, float *__restrict__ C, int ldc,
               int M, int N, int K, int accumulate, int act) {
    __shared__ float As[GEMM_BK][GEMM_BM + 4];
    __shared__ float Ws[GEMM_BK][GEMM_BN + 4];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;          /* 16 x 16 threads */
    const int m0 = blockIdx.y * GEMM_BM, n0 = blockIdx.x * GEMM_BN;

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    /* load mapping: 256 threads move a 64x16 tile as 64 rows x 4 float4-ish
     * quads; thread -> (row = tid / 4, k4 = (tid % 4) * 4) */
    const int lrow = tid >> 2, lk = (tid & 3) * 4;

    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
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
            const float *src = W + (size_t)n * (size_t)K + k0 + lk;
            const bool rin = n < N;
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int k = k0 + lk + u;
                Ws[lk + u][lrow] = (rin && k < K) ? src[u] : 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < GEMM_BK; k++) {
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
            float v = acc[i][j];
            if (bias) v += bias[n];
            if (accumulate) v += crow[n];
            if (act == 1) v = v > 0.0f ? v : 0.0f;
            else if (act == 2) v = v * gemm_sigmoid(v);
            crow[n] = v;
        }
    }
}

cudaError_t k_gemm_wt(const float *A, int lda, const float *W, const float *bias,
                      float *C, int ldc, int M, int N, int K, int accumulate, int act,
                      cudaStream_t s) {
    if (M <= 0 || N <= 0) return cudaSuccess;
    dim3 grid((unsigned)((N + GEMM_BN - 1) / GEMM_BN), (unsigned)((M + GEMM_BM - 1) / GEMM_BM));
    gemm_wt_kernel<<<grid, GEMM_THREADS, 0, s>>>(A, lda, W, bias, C, ldc, M, N, K,
                                                 accumulate, act);
    return cudaGetLastError();
}
