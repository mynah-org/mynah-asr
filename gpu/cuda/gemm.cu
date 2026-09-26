/* gpu/cuda/gemm.cu — C[M,N] = A[M,K] · W[N,K]^T, row-stable by construction.
 *
 * WHY NOT cuBLAS HERE. Durable contract 4: a stream's transcript never depends
 * on who it was batched with. cuBLAS chooses its kernel (tiling, split-K) per
 * shape; MEASURED on the L4 (tests/test_cuda_kernels, 2026-09-26) it is not
 * row-stable on any of the 12 shapes the model issues, from M = 1. It stays the
 * comparison arm (--gemm cublas), never the default.
 *
 * THE INVARIANT every kernel in this file keeps: output element (m, n) is ONE
 * fmaf chain acc = fmaf(A[m][k], W[n][k], acc) over k = 0, 1, ..., K-1 in that
 * order, starting from 0.0f, walked in tiles of GEMM_BK = 16 with the same zero
 * padding of the last tile, then + bias, + C (accumulate), activation. Nothing
 * else about the kernel -- block tile, thread microtile, which rows share a
 * block, how many blocks there are -- can change a bit of the result. So any
 * tiling may be chosen per call (the v2 dispatcher below picks one from M, N and
 * the SM count to fill the card at small cohorts), and every one of them is
 * byte-identical to v1 and to itself at any M. The test gates that too.
 *
 * v1 (S14-2): one 64x64 tile, 4x4 microtile, one sync pair per k-tile. Kept as
 * the reference and as an A/B arm. Profiled on the L4 (S14-6b): ~27 % of the
 * card's sustained FP32 and too few blocks at small M (16 blocks for a
 * 1024-wide output at M <= 64 on 58 SMs).
* v2 (S14-8a, REJECTED as the default 2026-09-26, kept as the own-v2 arm): templated block tile and microtile, double-buffered shared
 * memory (one barrier per k-tile), interleaved thread->column mapping so the
 * shared loads are bank-conflict-free and the stores coalesced, and a tile
 * chosen per call to give the card at least two waves of blocks. */
#include "kernels.cuh"

#include <math.h>

#define GEMM_BK 16

__device__ __forceinline__ float gemm_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ float gemm_epilogue(float v, const float *bias, const float *crow, int n,
                                               int accumulate, int act) {
    if (bias) v += bias[n];
    if (accumulate) v += crow[n];
    if (act == 1) v = v > 0.0f ? v : 0.0f;
    else if (act == 2) v = v * gemm_sigmoid(v);
    return v;
}

/* ------------------------------------------------------------------ v1 */
#define GEMM_BM 64
#define GEMM_BN 64
#define GEMM_THREADS 256

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
            crow[n] = gemm_epilogue(acc[i][j], bias, crow, n, accumulate, act);
        }
    }
}

cudaError_t k_gemm_wt_v1(const float *A, int lda, const float *W, const float *bias,
                         float *C, int ldc, int M, int N, int K, int accumulate, int act,
                         cudaStream_t s) {
    if (M <= 0 || N <= 0) return cudaSuccess;
    dim3 grid((unsigned)((N + GEMM_BN - 1) / GEMM_BN), (unsigned)((M + GEMM_BM - 1) / GEMM_BM));
    gemm_wt_kernel<<<grid, GEMM_THREADS, 0, s>>>(A, lda, W, bias, C, ldc, M, N, K, accumulate, act);
    return cudaGetLastError();
}

/* ------------------------------------------------------------------ v2 */
/* BM x BN block tile, TM x TN per thread; threads = (BM/TM) * (BN/TN).
 * Thread (ty, tx) owns rows ty + i*(BM/TM) and columns tx + j*(BN/TN):
 * interleaved, so a warp's shared reads hit consecutive words.
 *
 * Software pipeline: the NEXT k-tile is fetched from global memory into
 * registers (float4 along k when the operands are 16-byte aligned), the
 * CURRENT tile is computed from shared memory while those loads are in flight,
 * and only then are the registers written to the other shared buffer. Loading
 * straight from global into shared (the first v2) makes every thread wait for
 * its load before it can compute, and measured slower than v1. */
template <int BM, int BN, int TM, int TN, bool VEC>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_wt_t(const float *__restrict__ A, int lda, const float *__restrict__ W,
          const float *__restrict__ bias, float *__restrict__ C, int ldc,
          int M, int N, int K, int accumulate, int act) {
    constexpr int NTX = BN / TN, NTY = BM / TM, NT = NTX * NTY;
    constexpr int LA = (BM * 4 + NT - 1) / NT;      /* float4 loads per thread, A */
    constexpr int LW = (BN * 4 + NT - 1) / NT;      /* float4 loads per thread, W */
    __shared__ float As[2][GEMM_BK][BM + 4];
    __shared__ float Ws[2][GEMM_BK][BN + 4];
    const int tid = threadIdx.x, tx = tid % NTX, ty = tid / NTX;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    float4 ra[LA], rw[LW];
    /* quad q of a tile: row q / 4, k offset (q % 4) * 4 */
    auto fetch = [&](int k0) {
#pragma unroll
        for (int l = 0; l < LA; l++) {
            const int q = tid + l * NT, r = q >> 2, kk = (q & 3) * 4, m = m0 + r, k = k0 + kk;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q < BM * 4 && m < M) {
                const float *p = A + (size_t)m * (size_t)lda + k;
                if (VEC && k + 3 < K) v = *reinterpret_cast<const float4 *>(p);
                else { if (k < K) v.x = p[0]; if (k + 1 < K) v.y = p[1]; if (k + 2 < K) v.z = p[2]; if (k + 3 < K) v.w = p[3]; }
            }
            ra[l] = v;
        }
#pragma unroll
        for (int l = 0; l < LW; l++) {
            const int q = tid + l * NT, r = q >> 2, kk = (q & 3) * 4, n = n0 + r, k = k0 + kk;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q < BN * 4 && n < N) {
                const float *p = W + (size_t)n * (size_t)K + k;
                if (VEC && k + 3 < K) v = *reinterpret_cast<const float4 *>(p);
                else { if (k < K) v.x = p[0]; if (k + 1 < K) v.y = p[1]; if (k + 2 < K) v.z = p[2]; if (k + 3 < K) v.w = p[3]; }
            }
            rw[l] = v;
        }
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int l = 0; l < LA; l++) {
            const int q = tid + l * NT, r = q >> 2, kk = (q & 3) * 4;
            if (q < BM * 4) { As[buf][kk][r] = ra[l].x; As[buf][kk + 1][r] = ra[l].y; As[buf][kk + 2][r] = ra[l].z; As[buf][kk + 3][r] = ra[l].w; }
        }
#pragma unroll
        for (int l = 0; l < LW; l++) {
            const int q = tid + l * NT, r = q >> 2, kk = (q & 3) * 4;
            if (q < BN * 4) { Ws[buf][kk][r] = rw[l].x; Ws[buf][kk + 1][r] = rw[l].y; Ws[buf][kk + 2][r] = rw[l].z; Ws[buf][kk + 3][r] = rw[l].w; }
        }
    };

    int buf = 0;
    fetch(0);
    stash(0);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        const bool more = k0 + GEMM_BK < K;
        if (more) fetch(k0 + GEMM_BK);             /* in flight during the compute */
#pragma unroll
        for (int k = 0; k < GEMM_BK; k++) {
            float a[TM], w[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[buf][k][ty + i * NTY];
#pragma unroll
            for (int j = 0; j < TN; j++) w[j] = Ws[buf][k][tx + j * NTX];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], w[j], acc[i][j]);
        }
        /* the other buffer was last read before the previous barrier */
        if (more) stash(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        const int m = m0 + ty + i * NTY;
        if (m >= M) continue;
        float *crow = C + (size_t)m * (size_t)ldc;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const int n = n0 + tx + j * NTX;
            if (n >= N) continue;
            crow[n] = gemm_epilogue(acc[i][j], bias, crow, n, accumulate, act);
        }
    }
}

template <int BM, int BN, int TM, int TN>
static cudaError_t launch_t(const float *A, int lda, const float *W, const float *bias, float *C,
                            int ldc, int M, int N, int K, int accumulate, int act, cudaStream_t s) {
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM));
    /* float4 loads need 16-byte rows: both base pointers aligned and both row
     * strides multiples of 4 floats. Same arithmetic either way. */
    const bool vec = ((((size_t)A) | ((size_t)W)) & 15u) == 0 && (lda & 3) == 0 && (K & 3) == 0;
    if (vec)
        gemm_wt_t<BM, BN, TM, TN, true><<<grid, (BM / TM) * (BN / TN), 0, s>>>(A, lda, W, bias, C, ldc, M, N, K, accumulate, act);
    else
        gemm_wt_t<BM, BN, TM, TN, false><<<grid, (BM / TM) * (BN / TN), 0, s>>>(A, lda, W, bias, C, ldc, M, N, K, accumulate, act);
    return cudaGetLastError();
}

/* the v2 configurations, largest reuse first */
enum { G2_128x64 = 0, G2_64x128, G2_64x64, G2_32x64, G2_16x64, G2_16x32, G2__N };
static const int G2_BM[G2__N] = {128, 64, 64, 32, 16, 16};
static const int G2_BN[G2__N] = {64, 128, 64, 64, 64, 32};

static cudaError_t launch_cfg(int cfg, const float *A, int lda, const float *W, const float *bias,
                              float *C, int ldc, int M, int N, int K, int accumulate, int act,
                              cudaStream_t s) {
    switch (cfg) {
        case G2_128x64: return launch_t<128, 64, 8, 4>(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
        case G2_64x128: return launch_t<64, 128, 4, 8>(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
        case G2_64x64: return launch_t<64, 64, 4, 4>(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
        case G2_32x64: return launch_t<32, 64, 2, 4>(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
        case G2_16x64: return launch_t<16, 64, 1, 4>(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
        default: return launch_t<16, 32, 1, 2>(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
    }
}

static int g_sms = 0;
static int g_force_cfg = -1;   /* the test pins a configuration; -1 = choose */

/* the largest tile that still gives the card two waves of blocks; if none
 * does, the one with the most blocks. Pure performance: every configuration
 * returns the same bits (the invariant above). */
static int choose_cfg(int M, int N) {
    if (g_force_cfg >= 0) return g_force_cfg;
    if (g_sms <= 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || g_sms <= 0)
            g_sms = 40;
    }
    int best = G2__N - 1;
    long best_blocks = -1;
    for (int c = 0; c < G2__N; c++) {
        const long blocks = (long)((N + G2_BN[c] - 1) / G2_BN[c]) * (long)((M + G2_BM[c] - 1) / G2_BM[c]);
        if (blocks >= 2L * g_sms) return c;
        if (blocks > best_blocks) { best_blocks = blocks; best = c; }
    }
    return best;
}

void k_gemm_force_config(int cfg) { g_force_cfg = cfg; }
int k_gemm_config_count(void) { return G2__N; }
const char *k_gemm_config_name(int cfg) {
    static const char *const nm[G2__N] = {"128x64/8x4", "64x128/4x8", "64x64/4x4", "32x64/2x4", "16x64/1x4", "16x32/1x2"};
    return cfg >= 0 && cfg < G2__N ? nm[cfg] : "?";
}

cudaError_t k_gemm_wt_v2(const float *A, int lda, const float *W, const float *bias,
                         float *C, int ldc, int M, int N, int K, int accumulate, int act,
                         cudaStream_t s) {
    if (M <= 0 || N <= 0) return cudaSuccess;
    return launch_cfg(choose_cfg(M, N), A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
}

/* The default. MEASURED on the L4 2026-09-26 (tests/test_cuda_kernels --bench):
 * v2 is byte-identical to v1 but not faster -- its best tile wins 10-15 % on a
 * few shapes and loses on most, and both stay 2-4x behind cuBLAS. The reason
 * is structural, not tuning: one sequential fma chain per output element (the
 * row-stability invariant) leaves too few independent chains at small cohorts
 * with a large K, where cuBLAS splits K. So the default stays v1 and v2 is the
 * --gemm own-v2 arm; the next GEMM lever changes the accumulation order and is
 * therefore a numerical change with its own gate (see the S14 note). */
cudaError_t k_gemm_wt(const float *A, int lda, const float *W, const float *bias,
                      float *C, int ldc, int M, int N, int K, int accumulate, int act,
                      cudaStream_t s) {
    if (g_force_cfg >= 0) return k_gemm_wt_v2(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
    return k_gemm_wt_v1(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
}
