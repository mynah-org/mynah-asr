/* tests/gemm_shapes_bench.cu — own-tc (bf16 tensor cores) on the REAL shapes
 * of one encoder pass and one label-loop iteration, against cuBLAS bf16 as the
 * ceiling and against a roofline.
 *
 * WHY THIS EXISTS. test_cuda_kernels --bench-tc times own-tc at a handful of M
 * on warm, L2-resident operands, which says neither how far a STAGE is from
 * what the card can do nor why. This tool takes the (M, N, K, calls) list a
 * pass actually issues (gpu/cuda/engine.cu enqueue_encoder / pass_decode, at
 * the steady chunk of the given lookahead), times each shape with the weights
 * COLD the way a layer sees them (operands rotated over enough copies to
 * exceed the L2 several times: 24 layers of weights never fit), and prints per
 * shape: time, TFLOP/s, effective GB/s over the minimum bytes, the split count
 * S = k_gemm_tc_splits(N, K), the bytes the split-K partials add, the roofline
 * time and the percent of it, and cuBLAS bf16 (CUBLAS_COMPUTE_32F) on the
 * same shape. A footer weights every shape by its calls per pass and sums by
 * engine stage, to compare with `test_cuda_stream --bench C` stage ms.
 *
 * The roofline is max(FLOP / peak, min_bytes / bw), min_bytes = bf16 weights
 * once + f32 A once + f32 C once (what the engine's interface moves). bw is
 * MEASURED here by a copy kernel unless --bw-gbs is given; the tensor peak is
 * an ASSUMPTION passed by --peak-tflops (GB10: not measured by this tool; the
 * cuBLAS column is the practical ceiling).
 *
 * usage: tests/gemm_shapes_bench [--model nemotron|eou] [--lanes 32,128]
 *            [--q 4] [--dec-iters 8] [--reps 20] [--peak-tflops 62]
 *            [--bw-gbs 0 (=measure)] [--only NAME] [--no-cublas] [--csv]
 *   --only NAME   time one shape (e.g. ff_w1) at each --lanes: the ncu target
 *   --q           rows per lane per pass (lookahead + 1; Nemotron [56,3] = 4,
 *                 EOU [70,1] = 2); the subsampling M follows the steady chunk
 *   --dec-iters   label-loop iterations per pass, to weight the decoder rows
 *                 (read decode_iters / passes from test_cuda_stream --bench)
 * exit 0 ok; 1 device error or a numeric mismatch vs cuBLAS; 77 no device. */
#include "../gpu/cuda/kernels.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "cuda: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublas: %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

struct model_dims {
    const char *name;
    int d, L, ffn, C, Fo1, Fo2, np, inter, dout, Hdec, V, pred_layers;
};
/* docs/nemotron-arch.md; EOU 120M = 17 x 512, 1-layer 640 predictor, no prompt
 * projector (NOTE: the CUDA engine at main refuses a pack without one, so the
 * EOU rows are a model of the shapes, not of a pass it can run today) */
static const model_dims NEMOTRON = {"nemotron", 1024, 24, 4096, 256, 33, 17, 128, 2048, 640, 640, 13088, 2};
static const model_dims EOU120M = {"eou", 512, 17, 2048, 256, 33, 17, 0, 0, 640, 640, 1027, 1};

struct shape {
    const char *stage, *name;
    int M, N, K, calls;
};

static std::vector<shape> pass_shapes(const model_dims &m, int lanes, int q, int dec_iters) {
    std::vector<shape> v;
    const int R = lanes * q;
    /* steady chunk: to[1] = 2q stage-1 frames, to[2] = q (engine.cu ss_geometry) */
    const int P1 = lanes * 2 * q * m.Fo1, P2 = lanes * q * m.Fo2, CF = m.C * m.Fo2;
    v.push_back({"subsample", "ss_pw1", P1, m.C, m.C, 1});
    v.push_back({"subsample", "ss_pw2", P2, m.C, m.C, 1});
    v.push_back({"subsample", "ss_lin", R, m.d, CF, 1});
    v.push_back({"ffn1", "ff_w1", R, m.ffn, m.d, m.L});
    v.push_back({"ffn1", "ff_w2", R, m.d, m.ffn, m.L});
    v.push_back({"att_proj", "qkv", R, m.d, m.d, 3 * m.L});
    v.push_back({"att_out", "o", R, m.d, m.d, m.L});
    v.push_back({"conv", "pw1", R, 2 * m.d, m.d, m.L});
    v.push_back({"conv", "pw2", R, m.d, m.d, m.L});
    v.push_back({"ffn2", "ff_w1", R, m.ffn, m.d, m.L});
    v.push_back({"ffn2", "ff_w2", R, m.d, m.ffn, m.L});
    if (m.np > 0) {
        v.push_back({"post", "prompt_l1", R, m.inter, m.d + m.np, 1});
        v.push_back({"post", "prompt_l2", R, m.d, m.inter, 1});
    }
    v.push_back({"post", "encproj", R, m.dout, m.d, 1});
    /* the label loop: M = active lanes (upper bound: all of them) */
    v.push_back({"dec_joint", "head", lanes, m.V, m.Hdec, dec_iters});
    v.push_back({"dec_pred", "lstm", lanes, 4 * m.Hdec, m.Hdec, 2 * m.pred_layers * dec_iters});
    v.push_back({"dec_pred", "proj", lanes, m.Hdec, m.Hdec, dec_iters});
    return v;
}

/* mirror of gemm.cu tc_choose() for the report only (it is static there) */
static const char *tile_of(int M, int N, int S) {
    if (M <= 32) return "32x64";
    const long blocks = (long)((N + 127) / 128) * (long)((M + 63) / 64) * S;
    return blocks >= 2L * 128 ? "64x128" : "64x64";
}

__global__ void copy_kernel(const float4 *__restrict__ s, float4 *__restrict__ d, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) d[i] = s[i];
}
/* the split reduce's access pattern (S strided reads, one write) for a model
 * of its share; the real kernel is gemm.cu gemm_reduce_kernel */
__global__ void reduce_model_kernel(const float *__restrict__ P, int S, int M, int N, float *__restrict__ C) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= N || m >= M) return;
    const size_t mn = (size_t)M * N, o = (size_t)m * N + n;
    float v = P[o];
    for (int s = 1; s < S; s++) v += P[(size_t)s * mn + o];
    C[o] = v;
}
__global__ void fill_kernel(float *p, size_t n, unsigned seed) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { unsigned x = (unsigned)i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15; p[i] = ((x & 0xFFFF) / 32768.0f - 1.0f) * 0.5f; }
}

static double measure_bw(void) {
    const size_t bytes = (size_t)512 << 20, n = bytes / 16;
    float4 *a, *b;
    CU(cudaMalloc(&a, bytes)); CU(cudaMalloc(&b, bytes));
    CU(cudaMemset(a, 0, bytes)); CU(cudaMemset(b, 0, bytes));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int r = 0; r < 6; r++) {
        cudaEventRecord(e0);
        copy_kernel<<<1024, 256>>>(a, b, n);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (r > 0) best = std::min(best, ms);
    }
    CU(cudaGetLastError());
    cudaFree(a); cudaFree(b);
    return 2.0 * (double)bytes / (best * 1e-3) / 1e9;
}

struct arena {
    /* rotated copies: copy c of A/W/C at c * stride */
    float *A = nullptr, *C = nullptr, *P = nullptr, *A32tmp = nullptr;
    unsigned short *W16 = nullptr;
    __nv_bfloat16 *A16 = nullptr;
    int copies = 1;
};

static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char **argv) {
    const char *model = "nemotron", *lanes_s = "32,128", *only = nullptr;
    int q = -1, dec_iters = 8, reps = 20, use_cublas = 1, csv = 0;
    double peak = 62.0, bw = 0.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--lanes") && i + 1 < argc) lanes_s = argv[++i];
        else if (!strcmp(argv[i], "--q") && i + 1 < argc) q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dec-iters") && i + 1 < argc) dec_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--peak-tflops") && i + 1 < argc) peak = atof(argv[++i]);
        else if (!strcmp(argv[i], "--bw-gbs") && i + 1 < argc) bw = atof(argv[++i]);
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--no-cublas")) use_cublas = 0;
        else if (!strcmp(argv[i], "--csv")) csv = 1;
        else { fprintf(stderr, "unknown argument %s (see the header of tests/gemm_shapes_bench.cu)\n", argv[i]); return 2; }
    }
    const model_dims &m = !strcmp(model, "eou") ? EOU120M : NEMOTRON;
    if (q <= 0) q = !strcmp(model, "eou") ? 2 : 4;
    if (reps < 3) reps = 3;

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("SKIP gemm_shapes_bench: no CUDA device\n"); return 77; }
    cudaDeviceProp p; CU(cudaGetDeviceProperties(&p, 0));
    if (p.major < 8) { printf("SKIP: %s is sm_%d%d, own-tc needs sm_80+\n", p.name, p.major, p.minor); return 77; }
    int l2 = 0; cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
    const double bw_meas = measure_bw();
    if (bw <= 0.0) bw = bw_meas;
    printf("gemm_shapes_bench on %s sm_%d%d, %d SMs, L2 %.1f MiB; copy-kernel bw %.1f GB/s; roofline bw %.1f GB/s, "
           "tensor peak %.1f TFLOP/s (ASSUMED, --peak-tflops)\n", p.name, p.major, p.minor, p.multiProcessorCount,
           l2 / 1048576.0, bw_meas, bw, peak);
    printf("model %s q=%d dec_iters=%d reps=%d, weights cold (rotated copies)\n", m.name, q, dec_iters, reps);

    std::vector<int> lanes_v;
    for (const char *s = lanes_s; *s;) { lanes_v.push_back(atoi(s)); const char *c = strchr(s, ','); if (!c) break; s = c + 1; }

    cublasHandle_t h = nullptr;
    if (use_cublas) CB(cublasCreate(&h));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int bad = 0;

    if (csv) printf("CSV,model,lanes,stage,name,M,N,K,calls,S,tile,own_ms,own_tflops,own_gbs,roof_ms,pct_roof,partial_mb,reduce_model_ms,cublas_ms,own_over_cublas,maxdiff\n");
    for (int lanes : lanes_v) {
        const std::vector<shape> shp = pass_shapes(m, lanes, q, dec_iters);
        struct agg { double own, roof, cub; };
        std::vector<std::pair<const char *, agg>> st;
        if (!csv)
            printf("\n== lanes %d (R = %d rows)\n%-10s %-10s %6s %6s %6s %5s %2s %-6s %9s %7s %7s %9s %6s %8s %9s %9s %6s\n", lanes, lanes * q,
                   "stage", "gemm", "M", "N", "K", "calls", "S", "tile", "own ms", "TFLOP/s", "GB/s", "roof ms", "%roof", "part MB", "red.model", "cublas ms", "own/cb");
        for (const shape &s : shp) {
            if (only && strcmp(only, s.name) != 0) continue;
            const int M = s.M, N = s.N, K = s.K, S = k_gemm_tc_splits(N, K);
            const size_t aN = (size_t)M * K, wN = (size_t)N * K, cN = (size_t)M * N;
            const size_t wsf = k_gemm_tc_workspace_floats(M, N, K);
            const double minbytes = 2.0 * wN + 4.0 * aN + 4.0 * cN, flop = 2.0 * M * (double)N * K;
            const double partial = S > 1 ? 2.0 * 4.0 * S * (double)cN : 0.0;
            /* enough copies that one rotation is >= 4x the L2 (min 2, max 48) */
            const double per = 2.0 * wN + 4.0 * aN + 4.0 * cN;
            int copies = (int)ceil(4.0 * (double)l2 / per);
            copies = std::max(2, std::min(48, copies));
            float *A, *C, *P = nullptr, *Wf; unsigned short *W16; __nv_bfloat16 *A16 = nullptr;
            CU(cudaMalloc(&A, aN * copies * 4)); CU(cudaMalloc(&C, cN * copies * 4));
            CU(cudaMalloc(&W16, wN * copies * 2)); CU(cudaMalloc(&Wf, wN * 4));
            if (wsf) CU(cudaMalloc(&P, wsf * 4));
            fill_kernel<<<(unsigned)((aN * copies + 255) / 256), 256>>>(A, aN * copies, 1u);
            fill_kernel<<<(unsigned)((wN + 255) / 256), 256>>>(Wf, wN, 2u);
            for (int c = 0; c < copies; c++) CU(k_f32_to_bf16(Wf, W16 + (size_t)c * wN, wN, 0));
            if (use_cublas) {
                CU(cudaMalloc(&A16, aN * copies * 2));
                /* the same RNE rounding own-tc applies while staging A */
                CU(k_f32_to_bf16(A, (unsigned short *)A16, aN * copies, 0));
            }
            CU(cudaDeviceSynchronize());

            auto run_own = [&](int c) {
                return k_gemm_wt_tc(A + (size_t)c * aN, K, W16 + (size_t)c * wN, nullptr, C + (size_t)c * cN, N, M, N, K, 0, 0, P, wsf, 0);
            };
            auto run_cub = [&](int c) {
                const float al = 1.0f, be = 0.0f;
                return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &al, W16 + (size_t)c * wN, CUDA_R_16BF, K,
                                    A16 + (size_t)c * aN, CUDA_R_16BF, K, &be, C + (size_t)c * cN, CUDA_R_32F, N,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            };
            /* numeric sanity on copy 0: same bf16 operands, fp32 accumulate in
             * a different order -> small differences only */
            double maxdiff = 0.0;
            if (use_cublas) {
                CU(run_own(0)); CU(cudaDeviceSynchronize());
                std::vector<float> co(cN), cc(cN);
                CU(cudaMemcpy(co.data(), C, cN * 4, cudaMemcpyDeviceToHost));
                CB(run_cub(0)); CU(cudaDeviceSynchronize());
                CU(cudaMemcpy(cc.data(), C, cN * 4, cudaMemcpyDeviceToHost));
                double scale = 0.0;
                for (size_t i = 0; i < cN; i += 7) { maxdiff = std::max(maxdiff, (double)fabsf(co[i] - cc[i])); scale = std::max(scale, (double)fabsf(cc[i])); }
                if (maxdiff > 1e-3 * std::max(1.0, scale)) { bad++; printf("MISMATCH %s M=%d: own-tc vs cublas max|diff| %.3g (|ref| %.3g)\n", s.name, M, maxdiff, scale); }
            }
            auto timeit = [&](int arm) {
                std::vector<float> t;
                for (int r = 0; r < reps + 2; r++) {
                    const int c = r % copies;
                    cudaEventRecord(e0);
                    if (arm == 0) CU(run_own(c)); else CB(run_cub(c));
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float ms; cudaEventElapsedTime(&ms, e0, e1);
                    if (r >= 2) t.push_back(ms);
                }
                return median(t);
            };
            const float t_own = timeit(0);
            const float t_cub = use_cublas ? timeit(1) : 0.0f;
            float t_red = 0.0f;
            if (S > 1) {
                std::vector<float> t;
                for (int r = 0; r < reps + 2; r++) {
                    cudaEventRecord(e0);
                    reduce_model_kernel<<<dim3((unsigned)((N + 255) / 256), (unsigned)M), 256>>>(P, S, M, N, C);
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float ms; cudaEventElapsedTime(&ms, e0, e1);
                    if (r >= 2) t.push_back(ms);
                }
                CU(cudaGetLastError());
                t_red = median(t);
            }
            const double roof = std::max(flop / (peak * 1e12), minbytes / (bw * 1e9)) * 1e3;
            const double tfl = flop / (t_own * 1e-3) / 1e12, gbs = minbytes / (t_own * 1e-3) / 1e9;
            if (csv)
                printf("CSV,%s,%d,%s,%s,%d,%d,%d,%d,%d,%s,%.4f,%.2f,%.1f,%.4f,%.1f,%.1f,%.4f,%.4f,%.2f,%.3g\n", m.name, lanes, s.stage, s.name, M, N, K,
                       s.calls, S, tile_of(M, N, S), t_own, tfl, gbs, roof, 100.0 * roof / t_own, partial / 1e6, t_red, t_cub,
                       t_cub > 0 ? t_own / t_cub : 0.0, maxdiff);
            else
                printf("%-10s %-10s %6d %6d %6d %5d %2d %-6s %9.4f %7.1f %7.1f %9.4f %5.0f%% %8.1f %9.4f %9.4f %6.2f\n", s.stage, s.name, M, N, K,
                       s.calls, S, tile_of(M, N, S), t_own, tfl, gbs, roof, 100.0 * roof / t_own, partial / 1e6, t_red, t_cub,
                       t_cub > 0 ? t_own / t_cub : 0.0);
            bool found = false;
            for (auto &a : st)
                if (!strcmp(a.first, s.stage)) { a.second.own += t_own * s.calls; a.second.roof += roof * s.calls; a.second.cub += t_cub * s.calls; found = true; }
            if (!found) st.push_back({s.stage, {t_own * (double)s.calls, roof * s.calls, (double)t_cub * s.calls}});
            cudaFree(A); cudaFree(C); cudaFree(W16); cudaFree(Wf); if (P) cudaFree(P); if (A16) cudaFree(A16);
        }
        if (!only) {
            double to = 0, tr = 0, tc = 0;
            printf("-- per pass, GEMM time only (calls x median), lanes %d: compare with test_cuda_stream --bench %d stage ms\n", lanes, lanes);
            for (auto &a : st) {
                printf("STAGE %-10s own-tc %8.3f ms  roofline %8.3f ms (%3.0f%%)  cublas %8.3f ms\n", a.first, a.second.own, a.second.roof,
                       100.0 * a.second.roof / a.second.own, a.second.cub);
                to += a.second.own; tr += a.second.roof; tc += a.second.cub;
            }
            printf("STAGE %-10s own-tc %8.3f ms  roofline %8.3f ms (%3.0f%%)  cublas %8.3f ms\n", "TOTAL", to, tr, 100.0 * tr / to, tc);
        }
    }
    if (h) cublasDestroy(h);
    printf("%s\n", bad ? "FAIL: own-tc and cuBLAS disagree on a shape" : "done");
    return bad ? 1 : 0;
}
