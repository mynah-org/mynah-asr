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

#include <cuda_bf16.h>
#include <mma.h>

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

/* ------------------------------------------------------------ split-K (S14-8b)
 * THE CONTRACT CHANGES HERE, deliberately (decision 2026-09-26): the result is
 * no longer v1's single chain, so it is not bit-identical to v1. What it keeps
 * is the property serving needs -- the same input row gives the same output row
 * whatever else is in the cohort -- by construction:
 *
 *   S, the number of splits, is a function of the weight's shape (N, K) ONLY,
 *   never of M. Split s is the ascending fma chain over its own k range
 *   [s*KC, min((s+1)*KC, K)), KC a multiple of 16, starting from 0.0f, written
 *   to its own partial buffer. The reduction adds the partials in the fixed
 *   order ((P0 + P1) + P2) + ..., then bias, accumulate, activation.
 *
 * Nothing in either kernel reads M except to skip rows past the end, so a row's
 * bits cannot depend on how many other rows share the call. Gated by the same
 * cohort-size sweep as the other kernels; the change against v1 is a numerical
 * change and carries the transcript and WER gates of the S14 note. */
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_part_kernel(const float *__restrict__ A, int lda, const float *__restrict__ W,
                 float *__restrict__ P, int M, int N, int K, int KC) {
    __shared__ float As[GEMM_BK][GEMM_BM + 4];
    __shared__ float Ws[GEMM_BK][GEMM_BN + 4];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * GEMM_BM, n0 = blockIdx.x * GEMM_BN, sp = blockIdx.z;
    const int kb = sp * KC, ke = min(K, kb + KC);
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    const int lrow = tid >> 2, lk = (tid & 3) * 4;
    for (int k0 = kb; k0 < ke; k0 += GEMM_BK) {
        {
            const int m = m0 + lrow;
            const float *src = A + (size_t)m * (size_t)lda + k0 + lk;
            const bool rin = m < M;
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int k = k0 + lk + u;
                As[lk + u][lrow] = (rin && k < ke) ? src[u] : 0.0f;
            }
        }
        {
            const int n = n0 + lrow;
            const float *src = W + (size_t)n * (size_t)K + k0 + lk;
            const bool rin = n < N;
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int k = k0 + lk + u;
                Ws[lk + u][lrow] = (rin && k < ke) ? src[u] : 0.0f;
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
    float *Ps = P + (size_t)sp * (size_t)M * (size_t)N;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j;
            if (n < N) Ps[(size_t)m * N + n] = acc[i][j];
        }
    }
}

__global__ void gemm_reduce_kernel(const float *__restrict__ P, int S, int M, int N,
                                   const float *__restrict__ bias, float *__restrict__ C, int ldc,
                                   int accumulate, int act) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= N || m >= M) return;
    const size_t mn = (size_t)M * (size_t)N, o = (size_t)m * N + n;
    float v = P[o];
    for (int sp = 1; sp < S; sp++) v += P[(size_t)sp * mn + o];      /* fixed order */
    float *crow = C + (size_t)m * (size_t)ldc;
    crow[n] = gemm_epilogue(v, bias, crow, n, accumulate, act);
}

/* S from the weight's shape only: enough (N-tile x split) blocks to give the
 * card two waves when the cohort is one M-tile, each split at least 256 deep
 * and a multiple of 16. The table is a pure function of (N, K, SM count); the
 * SM count is fixed for the life of the process. */
int k_gemm_splits(int N, int K) {
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) sms = 40;
    const int ntiles = (N + GEMM_BN - 1) / GEMM_BN;
    int S = (2 * sms + ntiles - 1) / ntiles;
    const int smax = K / 256;
    if (S > smax) S = smax;
    if (S < 1) S = 1;
    return S;
}

size_t k_gemm_splitk_workspace_floats(int Mmax, int N, int K) {
    const int S = k_gemm_splits(N, K);
    return S > 1 ? (size_t)S * (size_t)Mmax * (size_t)N : 0;
}

cudaError_t k_gemm_wt_splitk(const float *A, int lda, const float *W, const float *bias,
                             float *C, int ldc, int M, int N, int K, int accumulate, int act,
                             float *ws, size_t ws_floats, cudaStream_t s) {
    if (M <= 0 || N <= 0) return cudaSuccess;
    const int S = k_gemm_splits(N, K);
    if (S <= 1) return k_gemm_wt_v1(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, s);
    if (!ws || ws_floats < (size_t)S * (size_t)M * (size_t)N) return cudaErrorInvalidValue;
    int KC = (K + S - 1) / S;
    KC = (KC + GEMM_BK - 1) / GEMM_BK * GEMM_BK;
    dim3 grid((unsigned)((N + GEMM_BN - 1) / GEMM_BN), (unsigned)((M + GEMM_BM - 1) / GEMM_BM), (unsigned)S);
    gemm_part_kernel<<<grid, GEMM_THREADS, 0, s>>>(A, lda, W, ws, M, N, K, KC);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    dim3 g2((unsigned)((N + 255) / 256), (unsigned)M);
    gemm_reduce_kernel<<<g2, 256, 0, s>>>(ws, S, M, N, bias, C, ldc, accumulate, act);
    return cudaGetLastError();
}

/* ------------------------------------------------- bf16 tensor cores (own-tc)
 * --precision bf16 --gemm own-tc. Ported from mynah-tts's fixed-order prefill
 * tile (MYNAH_CUDA_PREFILL_BF16TC). A NUMERICAL CHANGE against every f32 arm:
 * the weights are a resident bf16 copy (round to nearest even, once at open),
 * the activations are rounded to bf16 as they are staged into shared memory,
 * the products accumulate in fp32 on the tensor cores (WMMA 16x16x16).
 *
 * What it keeps, by construction, is contract 4 -- a row's bits do not depend
 * on how many rows share the call -- and, new against split-K, they do not
 * depend on the card either:
 *
 *   S, the number of K splits, is tc_splits(N, K): a pure function of the
 *   weight's shape and two constants of this file. Never M, never the SM
 *   count, never the tile. Split s covers [s*KC, min((s+1)*KC, K)), KC a
 *   multiple of TC_BK. Inside a split, element (m, n) is the WMMA
 *   accumulator chain over the 32-wide K slabs in ascending order and, inside
 *   a slab, the two 16-deep MMA steps in ascending order, from 0.0f. A tensor
 *   core MMA computes each output element from its own row of A and column of
 *   B only, so the rows sharing a fragment cannot change it. Partials are
 *   added in split order ((P0 + P1) + P2) + ..., then bias, + C, activation
 *   -- the same reduce kernel as split-K.
 *
 * The block tile (which warp computes an element, how many blocks there are)
 * changes no bit, so it is chosen per call from M and N for speed; the test
 * pins every configuration and compares bytes. Needs sm_80+ (bf16 WMMA); the
 * engine refuses the arm on an older card instead of falling back. */
#define TC_BK 32
#define TC_LD (TC_BK + 8)         /* bf16 elements per shared row: 80 bytes, 16-byte multiple */
#define TC_SLD 20                 /* per-warp 16x16 fp32 staging row, padded */
#define TC_SPLIT_TARGET 128       /* (64-wide N tiles x splits) aimed at for one M tile */
#define TC_SPLIT_MIN_DEPTH 256    /* no split shallower than this */

__device__ __forceinline__ unsigned short tc_bf16_bits(float x) {
    __nv_bfloat16 h = __float2bfloat16_rn(x);
    return *reinterpret_cast<unsigned short *>(&h);
}

__global__ void k_f32_to_bf16_kernel(const float *__restrict__ src, unsigned short *__restrict__ dst, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = tc_bf16_bits(src[i]);
}

cudaError_t k_f32_to_bf16(const float *src, unsigned short *dst, size_t n, cudaStream_t s) {
    if (n == 0) return cudaSuccess;
    k_f32_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, dst, n);
    return cudaGetLastError();
}

/* WM x WN warps, each 32 x 32 outputs (2 x 2 fragments). */
template <int WM, int WN, bool VEC>
__global__ void __launch_bounds__(WM * WN * 32)
gemm_tc_kernel(const float *__restrict__ A, int lda, const unsigned short *__restrict__ W,
               const float *__restrict__ bias, float *__restrict__ C, int ldc,
               int M, int N, int K, int KC, float *__restrict__ P, int accumulate, int act) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    constexpr int BM = WM * 32, BN = WN * 32, NT = WM * WN * 32;
    constexpr int LA = BM * (TC_BK / 4) / NT;   /* float4 of A per thread per slab */
    constexpr int LW = BN * (TC_BK / 8) / NT;   /* 8 x bf16 of W per thread per slab */
    static_assert(LA * NT == BM * (TC_BK / 4) && LW * NT == BN * (TC_BK / 8), "tile/threads");
    __shared__ __align__(32) unsigned short As[BM * TC_LD];
    __shared__ __align__(32) unsigned short Ws[BN * TC_LD];
    __shared__ __align__(32) float St[WM * WN * 16 * TC_SLD];
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WN, wn = warp % WN;
    const int kbeg = blockIdx.z * KC, kend = min(K, kbeg + KC);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

    float4 ra[LA];
    uint4 rw[LW];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int l = 0; l < LA; l++) {
            const int q = tid + l * NT, r = q / (TC_BK / 4), kk = (q % (TC_BK / 4)) * 4;
            const int m = m0 + r, k = k0 + kk;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m < M) {
                const float *p = A + (size_t)m * (size_t)lda + k;
                if (VEC) { if (k < kend) v = *reinterpret_cast<const float4 *>(p); }
                else { if (k < kend) v.x = p[0]; if (k + 1 < kend) v.y = p[1]; if (k + 2 < kend) v.z = p[2]; if (k + 3 < kend) v.w = p[3]; }
            }
            ra[l] = v;
        }
#pragma unroll
        for (int l = 0; l < LW; l++) {
            const int q = tid + l * NT, r = q / (TC_BK / 8), kk = (q % (TC_BK / 8)) * 8;
            const int n = n0 + r, k = k0 + kk;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (n < N) {
                const unsigned short *p = W + (size_t)n * (size_t)K + k;
                if (VEC) { if (k < kend) v = *reinterpret_cast<const uint4 *>(p); }
                else {
                    unsigned short h[8];
#pragma unroll
                    for (int u = 0; u < 8; u++) h[u] = k + u < kend ? p[u] : (unsigned short)0;
                    v.x = (unsigned)h[0] | ((unsigned)h[1] << 16); v.y = (unsigned)h[2] | ((unsigned)h[3] << 16);
                    v.z = (unsigned)h[4] | ((unsigned)h[5] << 16); v.w = (unsigned)h[6] | ((unsigned)h[7] << 16);
                }
            }
            rw[l] = v;
        }
    };
    auto stash = [&]() {
#pragma unroll
        for (int l = 0; l < LA; l++) {
            const int q = tid + l * NT, r = q / (TC_BK / 4), kk = (q % (TC_BK / 4)) * 4;
            uint2 pk;
            pk.x = (unsigned)tc_bf16_bits(ra[l].x) | ((unsigned)tc_bf16_bits(ra[l].y) << 16);
            pk.y = (unsigned)tc_bf16_bits(ra[l].z) | ((unsigned)tc_bf16_bits(ra[l].w) << 16);
            *reinterpret_cast<uint2 *>(As + r * TC_LD + kk) = pk;
        }
#pragma unroll
        for (int l = 0; l < LW; l++) {
            const int q = tid + l * NT, r = q / (TC_BK / 8), kk = (q % (TC_BK / 8)) * 8;
            *reinterpret_cast<uint4 *>(Ws + r * TC_LD + kk) = rw[l];
        }
    };

    if (kbeg < kend) fetch(kbeg);
    for (int k0 = kbeg; k0 < kend; k0 += TC_BK) {
        stash();
        __syncthreads();
        if (k0 + TC_BK < kend) fetch(k0 + TC_BK);    /* in flight during the MMAs */
#pragma unroll
        for (int kk = 0; kk < TC_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; i++)
                wmma::load_matrix_sync(a[i], reinterpret_cast<const __nv_bfloat16 *>(As + (wm * 32 + i * 16) * TC_LD + kk), TC_LD);
#pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::load_matrix_sync(b[j], reinterpret_cast<const __nv_bfloat16 *>(Ws + (wn * 32 + j * 16) * TC_LD + kk), TC_LD);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    float *st = St + warp * 16 * TC_SLD;
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(st, acc[i][j], TC_SLD, wmma::mem_row_major);
            __syncwarp();
            const int rb = m0 + wm * 32 + i * 16, cb = n0 + wn * 32 + j * 16;
            for (int e = lane; e < 256; e += 32) {
                const int r = e >> 4, c = e & 15, m = rb + r, n = cb + c;
                if (m >= M || n >= N) continue;
                const float v = st[r * TC_SLD + c];
                if (P) P[(size_t)blockIdx.z * (size_t)M * (size_t)N + (size_t)m * N + n] = v;
                else {
                    float *crow = C + (size_t)m * (size_t)ldc;
                    crow[n] = gemm_epilogue(v, bias, crow, n, accumulate, act);
                }
            }
            __syncwarp();
        }
#endif
}

/* the block tiles, largest first; pure performance (see above) */
enum { TC_64x128 = 0, TC_64x64, TC_32x64, TC__N };
static int g_tc_force = -1;

static int tc_choose(int M, int N, int S) {
    if (g_tc_force >= 0) return g_tc_force;
    if (M <= 32) return TC_32x64;
    const long blocks = (long)((N + 127) / 128) * (long)((M + 63) / 64) * S;
    return blocks >= 2L * TC_SPLIT_TARGET ? TC_64x128 : TC_64x64;
}

template <int WM, int WN>
static void tc_launch(bool vec, const float *A, int lda, const unsigned short *W, const float *bias,
                      float *C, int ldc, int M, int N, int K, int KC, int S, float *P,
                      int accumulate, int act, cudaStream_t s) {
    dim3 grid((unsigned)((N + WN * 32 - 1) / (WN * 32)), (unsigned)((M + WM * 32 - 1) / (WM * 32)), (unsigned)S);
    if (vec) gemm_tc_kernel<WM, WN, true><<<grid, WM * WN * 32, 0, s>>>(A, lda, W, bias, C, ldc, M, N, K, KC, P, accumulate, act);
    else gemm_tc_kernel<WM, WN, false><<<grid, WM * WN * 32, 0, s>>>(A, lda, W, bias, C, ldc, M, N, K, KC, P, accumulate, act);
}

/* S from the weight's shape and this file's two constants ONLY: enough
 * (64-wide N tile x split) blocks to reach TC_SPLIT_TARGET for a one-tile
 * cohort, no split shallower than TC_SPLIT_MIN_DEPTH. The same S on an L4, an
 * A100 or a MIG slice: S14-11 does not apply to this arm. */
int k_gemm_tc_splits(int N, int K) {
    const int ntiles = (N + 63) / 64;
    int S = (TC_SPLIT_TARGET + ntiles - 1) / ntiles;
    const int smax = K / TC_SPLIT_MIN_DEPTH;
    if (S > smax) S = smax;
    if (S < 1) S = 1;
    return S;
}

size_t k_gemm_tc_workspace_floats(int Mmax, int N, int K) {
    const int S = k_gemm_tc_splits(N, K);
    return S > 1 ? (size_t)S * (size_t)Mmax * (size_t)N : 0;
}

void k_gemm_tc_force_config(int cfg) { g_tc_force = cfg; }
int k_gemm_tc_config_count(void) { return TC__N; }
const char *k_gemm_tc_config_name(int cfg) {
    static const char *const nm[TC__N] = {"tc64x128", "tc64x64", "tc32x64"};
    return cfg >= 0 && cfg < TC__N ? nm[cfg] : "?";
}

cudaError_t k_gemm_wt_tc(const float *A, int lda, const unsigned short *W, const float *bias,
                         float *C, int ldc, int M, int N, int K, int accumulate, int act,
                         float *ws, size_t ws_floats, cudaStream_t s) {
    if (M <= 0 || N <= 0) return cudaSuccess;
    const int S = k_gemm_tc_splits(N, K);
    if (S > 1 && (!ws || ws_floats < (size_t)S * (size_t)M * (size_t)N)) return cudaErrorInvalidValue;
    int KC = (K + S - 1) / S;
    KC = (KC + TC_BK - 1) / TC_BK * TC_BK;
    /* 16-byte loads need 16-byte rows in both operands; same arithmetic either way */
    const bool vec = (((size_t)A & 15u) == 0) && (((size_t)W & 15u) == 0) && (lda & 3) == 0 && (K & 7) == 0;
    float *P = S > 1 ? ws : nullptr;
    switch (tc_choose(M, N, S)) {
        case TC_64x128: tc_launch<2, 4>(vec, A, lda, W, bias, C, ldc, M, N, K, KC, S, P, accumulate, act, s); break;
        case TC_64x64: tc_launch<2, 2>(vec, A, lda, W, bias, C, ldc, M, N, K, KC, S, P, accumulate, act, s); break;
        default: tc_launch<1, 2>(vec, A, lda, W, bias, C, ldc, M, N, K, KC, S, P, accumulate, act, s); break;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess || S == 1) return e;
    dim3 g2((unsigned)((N + 255) / 256), (unsigned)M);
    gemm_reduce_kernel<<<g2, 256, 0, s>>>(ws, S, M, N, bias, C, ldc, accumulate, act);
    return cudaGetLastError();
}
