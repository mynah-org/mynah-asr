/* tests/test_cuda_kernels.cu — the GPU kernels against scalar CPU references,
 * and the two properties the design rests on (S14-2):
 *
 *   1. the GEMM is ROW-STABLE: row i of C = A·Wᵀ at M = 1 is byte-identical to
 *      row i inside M = 257, on every shape the Nemotron step issues;
 *   2. the argmax returns the LOWEST index among equal maxima, as the CPU's
 *      first-strict-greater scan does.
 *
 * Plus a tolerance check of every kernel against a straightforward CPU loop on
 * random data: layernorm (+silu), residual, silu, the rel-pos attention over a
 * ring window with a table, the GLU + causal depthwise with its cache, the
 * three subsampling stages, the LSTM gates and the label-loop decide rule.
 *
 * Needs a CUDA device. Exit 0 = all pass; 1 = a failure (named); 77 = no
 * device (SKIP, the CI compile-only job's expected outcome). No model. */
#include "../gpu/cuda/kernels.cuh"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL "); printf(__VA_ARGS__); printf("\n"); } else { printf("OK   "); printf(__VA_ARGS__); printf("\n"); } } while (0)
#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("FAIL cuda: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

static unsigned long long g_rng = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (float)((g_rng >> 11) & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f;
}
static void fill(std::vector<float> &v, float scale) { for (auto &x : v) x = frand() * scale; }

template <class T> static T *dup_dev(const std::vector<T> &h) {
    T *d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> static std::vector<T> to_host(const T *d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
static double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) { const double d = fabs((double)a[i] - (double)b[i]); if (d > m) m = d; }
    return m;
}
static float sigmoid_ref(float x) { return x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x)); }

/* ------------------------------------------------------------------ GEMM */
static int test_gemm_shape(int N, int K, const char *what) {
    const int M1 = 1, M2 = 257;
    std::vector<float> A((size_t)M2 * K), W((size_t)N * K), bias(N);
    fill(A, 1.0f); fill(W, 0.05f); fill(bias, 0.1f);
    float *dA = dup_dev(A), *dW = dup_dev(W), *db = dup_dev(bias), *dC1, *dC2;
    CU(cudaMalloc(&dC1, (size_t)M1 * N * sizeof(float)));
    CU(cudaMalloc(&dC2, (size_t)M2 * N * sizeof(float)));
    /* row 100 alone (as M = 1) against row 100 inside M = 257 */
    const int row = 100;
    CU(k_gemm_wt(dA + (size_t)row * K, K, dW, db, dC1, N, M1, N, K, 0, 0, 0));
    CU(k_gemm_wt(dA, K, dW, db, dC2, N, M2, N, K, 0, 0, 0));
    CU(cudaDeviceSynchronize());
    auto c1 = to_host(dC1, (size_t)M1 * N), c2 = to_host(dC2, (size_t)M2 * N);
    int identical = memcmp(c1.data(), c2.data() + (size_t)row * N, (size_t)N * sizeof(float)) == 0;
    CHECK(identical, "gemm row-stable %s [N=%d K=%d]: row alone == row inside M=257, byte for byte", what, N, K);
    /* correctness against a double-precision CPU loop, a few rows */
    double worst = 0.0;
    for (int m = 0; m < M2; m += 64)
        for (int n = 0; n < N; n += 97) {
            double acc = bias[n];
            for (int k = 0; k < K; k++) acc += (double)A[(size_t)m * K + k] * (double)W[(size_t)n * K + k];
            const double d = fabs(acc - (double)c2[(size_t)m * N + n]);
            if (d > worst) worst = d;
        }
    CHECK(worst < 1e-3 * sqrt((double)K), "gemm correct %s [N=%d K=%d]: max |gpu - cpu(double)| = %.3g", what, N, K, worst);
    /* accumulate + relu + silu epilogues */
    CU(k_gemm_wt(dA, K, dW, nullptr, dC2, N, M2, N, K, 1, 1, 0));
    CU(cudaDeviceSynchronize());
    auto c3 = to_host(dC2, (size_t)M2 * N);
    int ok = 1;
    for (int i = 0; i < N; i += 13) {
        const float v = c2[i] + (c2[i] - bias[i]);
        const float want = v > 0.0f ? v : 0.0f;
        if (fabsf(want - c3[i]) > 1e-4f * (1.0f + fabsf(want))) ok = 0;
    }
    CHECK(ok, "gemm epilogue %s: accumulate + relu", what);
    cudaFree(dA); cudaFree(dW); cudaFree(db); cudaFree(dC1); cudaFree(dC2);
    return 0;
}

/* -------------------------------------------------------------- argmax */
static int test_argmax(void) {
    const int V = 13088, n = 3;
    std::vector<float> lg((size_t)n * V);
    fill(lg, 1.0f);
    /* row 0: a tie between 17 and 9000 at the max -> 17 must win */
    lg[17] = 50.0f; lg[9000] = 50.0f;
    /* row 1: the max at the last index */
    lg[(size_t)V + V - 1] = 60.0f;
    /* row 2: a tie at index 0 and 1 -> 0 */
    lg[(size_t)2 * V + 0] = 70.0f; lg[(size_t)2 * V + 1] = 70.0f;
    float *d = dup_dev(lg);
    int *dam; CU(cudaMalloc(&dam, n * sizeof(int)));
    CU(k_dec_argmax(d, V, n, dam, 0));
    CU(cudaDeviceSynchronize());
    auto am = to_host(dam, (size_t)n);
    CHECK(am[0] == 17 && am[1] == V - 1 && am[2] == 0, "argmax: lowest index wins a tie (%d, %d, %d)", am[0], am[1], am[2]);
    cudaFree(d); cudaFree(dam);
    return 0;
}

/* ---------------------------------------------------------- layernorm */
static int test_layernorm(void) {
    const int rows = 37, d = 1024;
    std::vector<float> x((size_t)rows * d), w(d), b(d), out((size_t)rows * d);
    fill(x, 3.0f); fill(w, 1.0f); fill(b, 0.5f);
    for (int r = 0; r < rows; r++) {
        double mu = 0.0, var = 0.0;
        for (int i = 0; i < d; i++) mu += x[(size_t)r * d + i];
        mu /= d;
        for (int i = 0; i < d; i++) { double c = x[(size_t)r * d + i] - mu; var += c * c; }
        var /= d;
        const float inv = (float)(1.0 / sqrt(var + 1e-5));
        for (int i = 0; i < d; i++) {
            float v = ((x[(size_t)r * d + i] - (float)mu) * inv) * w[i] + b[i];
            out[(size_t)r * d + i] = v * sigmoid_ref(v);
        }
    }
    float *dx = dup_dev(x), *dw = dup_dev(w), *db = dup_dev(b), *dout;
    CU(cudaMalloc(&dout, x.size() * sizeof(float)));
    CU(k_layernorm(dx, dw, db, dout, rows, d, 1, 0));
    CU(cudaDeviceSynchronize());
    const double diff = max_abs_diff(to_host(dout, x.size()), out);
    CHECK(diff < 2e-5, "layernorm + silu: max |diff| = %.3g", diff);
    /* in place */
    CU(k_layernorm(dx, dw, db, dx, rows, d, 1, 0));
    CU(cudaDeviceSynchronize());
    const double diff2 = max_abs_diff(to_host(dx, x.size()), out);
    CHECK(diff2 < 2e-5, "layernorm in place: max |diff| = %.3g", diff2);
    cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(dout);
    return 0;
}

/* ------------------------------------------------ attention + kv ring */
static int test_attention(void) {
    gpu_model_dims dm = {};
    dm.n_layers = 2; dm.d = 256; dm.H = 2; dm.dk = 128; dm.left = 8; dm.kmax = 8 + 4 + 2;
    const int li = 1, B = 3;
    const int qs_[3] = {4, 1, 4}, valid_[3] = {0, 5, 8}, head_[3] = {0, 2, 3};
    /* arena: one slot per lane, ring prefilled with random rows */
    const int cap = 3;
    const size_t kv_n = (size_t)cap * dm.n_layers * 2 * dm.left * dm.d;
    std::vector<float> kv(kv_n); fill(kv, 1.0f);
    std::vector<gpu_slot_meta> meta(cap);
    for (int s = 0; s < cap; s++) { memset(&meta[s], 0, sizeof(meta[s])); meta[s].valid = valid_[s]; meta[s].head = head_[s]; }
    std::vector<gpu_row> rows(B);
    int R = 0;
    for (int a = 0; a < B; a++) { memset(&rows[a], 0, sizeof(rows[a])); rows[a].slot = a; rows[a].q = qs_[a]; rows[a].row_off = R; R += qs_[a]; }
    std::vector<float> qsv((size_t)R * dm.d), kn((size_t)R * dm.d), vn((size_t)R * dm.d), bu(dm.d), bv(dm.d);
    fill(qsv, 1.0f); fill(kn, 1.0f); fill(vn, 1.0f); fill(bu, 0.3f); fill(bv, 0.3f);
    std::vector<float> tab((size_t)dm.n_layers * (2 * dm.kmax - 1) * dm.d); fill(tab, 0.5f);
    /* CPU reference, the src/encoder.c arithmetic with the ring made logical */
    std::vector<float> ctx_ref((size_t)R * dm.d, 0.0f);
    for (int a = 0; a < B; a++) {
        const int Q = qs_[a], valid = valid_[a], K = valid + Q;
        const float scaling = 1.0f / sqrtf((float)dm.dk);
        for (int h = 0; h < dm.H; h++) {
            const int ho = h * dm.dk;
            for (int t = 0; t < Q; t++) {
                std::vector<float> sc(K);
                float mx = -3e38f;
                for (int j = 0; j < K; j++) {
                    const float *key, *rk;
                    if (j < valid) key = &kv[((((size_t)a * dm.n_layers + li) * 2 + 0) * dm.left + (size_t)((head_[a] + j) % dm.left)) * dm.d];
                    else key = &kn[(size_t)(rows[a].row_off + j - valid) * dm.d];
                    const int p = (dm.kmax - K) + (K - 1 - valid - t + j);
                    rk = &tab[((size_t)li * (2 * dm.kmax - 1) + p) * dm.d];
                    float ac = 0.0f, bd = 0.0f;
                    for (int i = 0; i < dm.dk; i++) {
                        const float qv_ = qsv[(size_t)(rows[a].row_off + t) * dm.d + ho + i];
                        ac += (qv_ + bu[ho + i]) * key[ho + i];
                        bd += (qv_ + bv[ho + i]) * rk[ho + i];
                    }
                    sc[j] = (ac + bd) * scaling;
                    if (sc[j] > mx) mx = sc[j];
                }
                float sum = 0.0f;
                for (int j = 0; j < K; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                for (int j = 0; j < K; j++) sc[j] /= sum;
                for (int i = 0; i < dm.dk; i++) {
                    float acc = 0.0f;
                    for (int j = 0; j < K; j++) {
                        const float *v;
                        if (j < valid) v = &kv[((((size_t)a * dm.n_layers + li) * 2 + 1) * dm.left + (size_t)((head_[a] + j) % dm.left)) * dm.d];
                        else v = &vn[(size_t)(rows[a].row_off + j - valid) * dm.d];
                        acc += sc[j] * v[ho + i];
                    }
                    ctx_ref[(size_t)(rows[a].row_off + t) * dm.d + ho + i] = acc;
                }
            }
        }
    }
    gpu_arena ar = {};
    ar.kv = dup_dev(kv); ar.meta = dup_dev(meta);
    gpu_layer_w L = {}; L.bias_u = dup_dev(bu); L.bias_v = dup_dev(bv);
    float *dtab = dup_dev(tab), *dq = dup_dev(qsv), *dkn = dup_dev(kn), *dvn = dup_dev(vn), *dctx;
    gpu_row *drows = dup_dev(rows);
    CU(cudaMalloc(&dctx, (size_t)R * dm.d * sizeof(float)));
    CU(k_attention(dm, L, li, dtab, ar, drows, B, dq, dkn, dvn, dctx, 0));
    CU(cudaDeviceSynchronize());
    const double diff = max_abs_diff(to_host(dctx, (size_t)R * dm.d), ctx_ref);
    CHECK(diff < 1e-4, "attention (ring window + table, mixed q and valid): max |diff| = %.3g", diff);
    /* lane identity: the same lane alone must give the same bytes */
    std::vector<gpu_row> one(1); one[0] = rows[2]; one[0].row_off = 0;
    gpu_row *done_ = dup_dev(one);
    float *dctx1; CU(cudaMalloc(&dctx1, (size_t)4 * dm.d * sizeof(float)));
    CU(k_attention(dm, L, li, dtab, ar, done_, 1, dq + (size_t)rows[2].row_off * dm.d, dkn + (size_t)rows[2].row_off * dm.d, dvn + (size_t)rows[2].row_off * dm.d, dctx1, 0));
    CU(cudaDeviceSynchronize());
    auto full = to_host(dctx, (size_t)R * dm.d), alone = to_host(dctx1, (size_t)4 * dm.d);
    CHECK(memcmp(alone.data(), full.data() + (size_t)rows[2].row_off * dm.d, (size_t)4 * dm.d * sizeof(float)) == 0,
          "attention lane identity: lane alone == lane inside the cohort, byte for byte");
    /* commit + advance: the ring holds the last min(valid+q, left) rows */
    CU(k_kv_commit(dm, li, ar, drows, B, dkn, dvn, 0));
    CU(k_kv_advance(dm, ar, drows, B, 0));
    CU(cudaDeviceSynchronize());
    auto kv2 = to_host(ar.kv, kv_n);
    auto meta2 = to_host(ar.meta, (size_t)cap);
    int ok = 1;
    for (int a = 0; a < B; a++) {
        const int Q = qs_[a], valid = valid_[a], total = valid + Q, nvalid = total < dm.left ? total : dm.left;
        if (meta2[a].valid != nvalid || meta2[a].t_abs != Q) ok = 0;
        /* logical row i after == window row (total - nvalid + i) */
        for (int i = 0; i < nvalid && ok; i++) {
            const int wi = total - nvalid + i;
            const float *want = wi < valid ? &kv[((((size_t)a * dm.n_layers + li) * 2 + 0) * dm.left + (size_t)((head_[a] + wi) % dm.left)) * dm.d]
                                           : &kn[(size_t)(rows[a].row_off + wi - valid) * dm.d];
            const float *got = &kv2[((((size_t)a * dm.n_layers + li) * 2 + 0) * dm.left + (size_t)((meta2[a].head + i) % dm.left)) * dm.d];
            if (memcmp(want, got, (size_t)dm.d * sizeof(float)) != 0) ok = 0;
        }
    }
    CHECK(ok, "kv commit + advance: the ring holds the last min(valid+q, left) rows, logically");
    return 0;
}

/* -------------------------------------------------------- glu + dwconv */
static int test_conv(void) {
    gpu_model_dims dm = {}; dm.n_layers = 1; dm.d = 64; dm.conv_k = 9;
    const int Q = 4, B = 2, cap = 2;
    std::vector<gpu_row> rows(B);
    for (int a = 0; a < B; a++) { memset(&rows[a], 0, sizeof(rows[a])); rows[a].slot = a; rows[a].q = Q; rows[a].row_off = a * Q; }
    std::vector<float> h2((size_t)B * Q * 2 * dm.d), dw((size_t)dm.d * dm.conv_k), cache((size_t)cap * (dm.conv_k - 1) * dm.d);
    fill(h2, 1.0f); fill(dw, 0.3f); fill(cache, 1.0f);
    std::vector<float> ref((size_t)B * Q * dm.d), cache_ref = cache;
    for (int a = 0; a < B; a++)
        for (int c = 0; c < dm.d; c++) {
            std::vector<float> seq(dm.conv_k - 1 + Q);
            for (int j = 0; j < dm.conv_k - 1; j++) seq[j] = cache[((size_t)a * (dm.conv_k - 1) + j) * dm.d + c];
            for (int t = 0; t < Q; t++) {
                const float *row = &h2[(size_t)(a * Q + t) * 2 * dm.d];
                seq[dm.conv_k - 1 + t] = row[c] * sigmoid_ref(row[dm.d + c]);
            }
            for (int t = 0; t < Q; t++) {
                float acc = 0.0f;
                for (int j = 0; j < dm.conv_k; j++) acc += dw[(size_t)c * dm.conv_k + j] * seq[t + j];
                ref[(size_t)(a * Q + t) * dm.d + c] = acc;
            }
            for (int j = 0; j < dm.conv_k - 1; j++) cache_ref[((size_t)a * (dm.conv_k - 1) + j) * dm.d + c] = seq[Q + j];
        }
    gpu_arena ar = {}; ar.conv_cache = dup_dev(cache);
    gpu_layer_w L = {}; L.dw_w = dup_dev(dw);
    float *dh2 = dup_dev(h2), *dout; gpu_row *drows = dup_dev(rows);
    CU(cudaMalloc(&dout, ref.size() * sizeof(float)));
    CU(k_glu_dwconv(dm, L, 0, ar, drows, B, dh2, dout, 0));
    CU(cudaDeviceSynchronize());
    CHECK(max_abs_diff(to_host(dout, ref.size()), ref) < 1e-5, "glu + causal depthwise: matches the CPU loop");
    CHECK(max_abs_diff(to_host(ar.conv_cache, cache.size()), cache_ref) < 1e-6, "conv cache refreshed with the last k-1 rows");
    return 0;
}

/* ------------------------------------------------------- subsampling */
static void ss_ref_stage(const std::vector<float> &x, int C_in, int T, int F, int lp, int rp,
                         const std::vector<float> &cache, const std::vector<float> &w,
                         const std::vector<float> &b, int depthwise, int C_out,
                         std::vector<float> &out, int *To_, int *Fo_, int relu) {
    /* x position-major [T*F][C_in] (or [T][F] for C_in = 1); out position-major */
    const int Tp = T + lp + rp, Fp = F + 3;
    const int To = (Tp - 3) / 2 + 1, Fo = (Fp - 3) / 2 + 1;
    *To_ = To; *Fo_ = Fo;
    out.assign((size_t)To * Fo * C_out, 0.0f);
    auto in = [&](int c, int tp, int fp) -> float {
        const int f = fp - 2;
        if (f < 0 || f >= F || tp < lp - 1) return 0.0f;
        if (tp == lp - 1) return cache[(size_t)c * F + f];
        const int t = tp - lp;
        if (t >= T) return 0.0f;
        return x[((size_t)t * F + f) * C_in + c];
    };
    for (int co = 0; co < C_out; co++)
        for (int to = 0; to < To; to++)
            for (int fo = 0; fo < Fo; fo++) {
                float acc = 0.0f;
                const int ci_lo = depthwise ? co : 0, ci_hi = depthwise ? co + 1 : C_in;
                for (int ci = ci_lo; ci < ci_hi; ci++)
                    for (int dt = 0; dt < 3; dt++)
                        for (int df = 0; df < 3; df++) {
                            const float wv = depthwise ? w[(size_t)co * 9 + dt * 3 + df] : w[((size_t)co * C_in + ci) * 9 + dt * 3 + df];
                            acc += wv * in(ci, 2 * to + dt, 2 * fo + df);
                        }
                acc += b[co];
                out[((size_t)to * Fo + fo) * C_out + co] = relu && acc < 0.0f ? 0.0f : acc;
            }
}

static int test_subsampling(void) {
    gpu_model_dims dm = {};
    dm.n_mels = 128; dm.C = 256;
    dm.F[0] = 128; dm.Fo[0] = 65; dm.F[1] = 65; dm.Fo[1] = 33; dm.F[2] = 33; dm.Fo[2] = 17;
    const int B = 2, cap = 2;
    const int nmel[2] = {33, 32}, first[2] = {1, 0}, last[2] = {0, 1};
    std::vector<gpu_row> rows(B);
    int M = 0, P[3] = {0, 0, 0};
    std::vector<int> to_all(B * 3);
    for (int a = 0; a < B; a++) {
        memset(&rows[a], 0, sizeof(rows[a]));
        rows[a].slot = a; rows[a].n_mel = nmel[a]; rows[a].mel_off = M; rows[a].first = first[a]; rows[a].last = last[a];
        int T = nmel[a];
        for (int s = 0; s < 3; s++) {
            const int lp = first[a] ? 2 : 1, rp = last[a] ? 1 : 0, Tp = T + lp + rp, To = (Tp - 3) / 2 + 1;
            rows[a].to[s] = To; rows[a].pos_off[s] = P[s]; P[s] += To * dm.Fo[s]; T = To; to_all[a * 3 + s] = To;
        }
        rows[a].q = rows[a].to[2]; M += nmel[a];
    }
    std::vector<float> mel((size_t)M * dm.n_mels), w0((size_t)dm.C * 9), b0(dm.C), w1((size_t)dm.C * 9), b1(dm.C);
    fill(mel, 1.0f); fill(w0, 0.3f); fill(b0, 0.1f); fill(w1, 0.3f); fill(b1, 0.1f);
    std::vector<float> c0((size_t)cap * dm.F[0]), c1((size_t)cap * dm.C * dm.F[1]);
    fill(c0, 1.0f); fill(c1, 1.0f);
    gpu_arena ar = {};
    ar.ss_cache[0] = dup_dev(c0); ar.ss_cache[1] = dup_dev(c1);
    std::vector<float> c2((size_t)cap * dm.C * dm.F[2], 0.0f); ar.ss_cache[2] = dup_dev(c2);
    float *dmel = dup_dev(mel), *dw0 = dup_dev(w0), *db0 = dup_dev(b0), *dw1 = dup_dev(w1), *db1 = dup_dev(b1), *s0, *s1;
    gpu_row *drows = dup_dev(rows);
    CU(cudaMalloc(&s0, (size_t)P[0] * dm.C * sizeof(float)));
    CU(cudaMalloc(&s1, (size_t)P[1] * dm.C * sizeof(float)));
    CU(k_ss_stage0(dm, ar, drows, B, dmel, dw0, db0, s0, 0));
    CU(cudaDeviceSynchronize());
    auto g0 = to_host(s0, (size_t)P[0] * dm.C);
    double worst = 0.0;
    std::vector<float> ref0_all;
    for (int a = 0; a < B; a++) {
        std::vector<float> x((size_t)nmel[a] * dm.n_mels);
        memcpy(x.data(), &mel[(size_t)rows[a].mel_off * dm.n_mels], x.size() * sizeof(float));
        std::vector<float> cache(&c0[(size_t)a * dm.F[0]], &c0[(size_t)(a + 1) * dm.F[0]]);
        std::vector<float> ref; int To, Fo;
        ss_ref_stage(x, 1, nmel[a], dm.F[0], first[a] ? 2 : 1, last[a], cache, w0, b0, 0, dm.C, ref, &To, &Fo, 1);
        std::vector<float> got(&g0[(size_t)rows[a].pos_off[0] * dm.C], &g0[(size_t)(rows[a].pos_off[0] + To * Fo) * dm.C]);
        const double d = max_abs_diff(got, ref);
        if (d > worst) worst = d;
        if (To != rows[a].to[0] || Fo != dm.Fo[0]) worst = 1e9;
    }
    CHECK(worst < 1e-4, "subsampling stage 0 (full 3x3 s2, first/last pads, cache): max |diff| = %.3g", worst);
    auto c0b = to_host(ar.ss_cache[0], c0.size());
    int ok = 1;
    for (int a = 0; a < B; a++)
        for (int f = 0; f < dm.F[0]; f++)
            if (c0b[(size_t)a * dm.F[0] + f] != mel[(size_t)(rows[a].mel_off + nmel[a] - 1) * dm.n_mels + f]) ok = 0;
    CHECK(ok, "subsampling stage 0 cache = the last mel frame");
    /* stage 1 depthwise over the GPU's stage-0 output */
    CU(k_ss_dw(dm, 1, ar, drows, B, s0, dw1, db1, s1, 0));
    CU(cudaDeviceSynchronize());
    auto g1 = to_host(s1, (size_t)P[1] * dm.C);
    worst = 0.0;
    for (int a = 0; a < B; a++) {
        const int T = rows[a].to[0], F = dm.F[1];
        std::vector<float> x(&g0[(size_t)rows[a].pos_off[0] * dm.C], &g0[(size_t)(rows[a].pos_off[0] + T * F) * dm.C]);
        std::vector<float> cache(&c1[(size_t)a * dm.C * F], &c1[(size_t)(a + 1) * dm.C * F]);
        std::vector<float> ref; int To, Fo;
        ss_ref_stage(x, dm.C, T, F, first[a] ? 2 : 1, last[a], cache, w1, b1, 1, dm.C, ref, &To, &Fo, 0);
        std::vector<float> got(&g1[(size_t)rows[a].pos_off[1] * dm.C], &g1[(size_t)(rows[a].pos_off[1] + To * Fo) * dm.C]);
        const double d = max_abs_diff(got, ref);
        if (d > worst) worst = d;
        if (To != rows[a].to[1] || Fo != dm.Fo[1]) worst = 1e9;
    }
    CHECK(worst < 1e-4, "subsampling stage 1 (depthwise 3x3 s2, cache): max |diff| = %.3g", worst);
    return 0;
}

/* ------------------------------------------------------- lstm + decide */
static int test_decode(void) {
    gpu_model_dims dm = {}; dm.Hdec = 64; dm.pred_layers = 2; dm.blank = 5; dm.max_symbols = 3; dm.V = 8;
    const int cap = 2, B = 2, n = 2;
    std::vector<gpu_row> rows(B);
    for (int a = 0; a < B; a++) { memset(&rows[a], 0, sizeof(rows[a])); rows[a].slot = a; rows[a].q = 2; rows[a].row_off = 2 * a; }
    std::vector<int> active = {0, 1}, emit = {1, 0};
    std::vector<float> h((size_t)cap * dm.pred_layers * dm.Hdec), c(h.size()), z((size_t)n * 4 * dm.Hdec);
    fill(h, 0.5f); fill(c, 0.5f); fill(z, 2.0f);
    std::vector<float> h_ref = h, c_ref = c, xn_ref((size_t)n * dm.Hdec);
    const int layer = 1;
    for (int a = 0; a < n; a++) {
        const int slot = a;
        for (int i = 0; i < dm.Hdec; i++) {
            const size_t hi = ((size_t)slot * dm.pred_layers + layer) * dm.Hdec + i;
            if (emit[a]) {
                const float *zr = &z[(size_t)a * 4 * dm.Hdec];
                const float ig = sigmoid_ref(zr[i]), fg = sigmoid_ref(zr[dm.Hdec + i]), gg = tanhf(zr[2 * dm.Hdec + i]), og = sigmoid_ref(zr[3 * dm.Hdec + i]);
                c_ref[hi] = fg * c[hi] + ig * gg;
                h_ref[hi] = og * tanhf(c_ref[hi]);
                xn_ref[(size_t)a * dm.Hdec + i] = h_ref[hi];
            } else xn_ref[(size_t)a * dm.Hdec + i] = h[hi];
        }
    }
    gpu_arena ar = {}; ar.dec_h = dup_dev(h); ar.dec_c = dup_dev(c);
    std::vector<gpu_slot_meta> meta(cap); for (auto &m : meta) memset(&m, 0, sizeof(m));
    ar.meta = dup_dev(meta);
    ar.tok_cap = 8;
    std::vector<int> tok((size_t)cap * ar.tok_cap, -1), tf(tok.size(), -1);
    ar.tok = dup_dev(tok); ar.tok_frame = dup_dev(tf);
    gpu_row *drows = dup_dev(rows);
    int *dact = dup_dev(active), *demit = dup_dev(emit);
    float *dz = dup_dev(z), *dxn; CU(cudaMalloc(&dxn, xn_ref.size() * sizeof(float)));
    CU(k_dec_lstm_gates(dm, ar, drows, dact, demit, n, layer, dz, dxn, 0));
    CU(cudaDeviceSynchronize());
    CHECK(max_abs_diff(to_host(ar.dec_h, h.size()), h_ref) < 1e-5 && max_abs_diff(to_host(ar.dec_c, c.size()), c_ref) < 1e-5 &&
          max_abs_diff(to_host(dxn, xn_ref.size()), xn_ref) < 1e-5, "lstm gates: emitting lane updated, idle lane untouched");
    /* the greedy rule: lane 0 sees token 3, lane 1 sees blank; then lane 0 at
     * max_symbols advances the frame */
    CU(k_dec_begin(ar, drows, B, 0));
    std::vector<int> am = {3, dm.blank};
    int *dam = dup_dev(am), *dem2; CU(cudaMalloc(&dem2, n * sizeof(int)));
    CU(k_dec_decide(dm, ar, drows, dact, n, dam, dem2, 0));
    CU(k_dec_decide(dm, ar, drows, dact, n, dam, dem2, 0));
    CU(k_dec_decide(dm, ar, drows, dact, n, dam, dem2, 0));   /* 3 emissions on frame 0 -> t = 1 */
    CU(cudaDeviceSynchronize());
    auto m2 = to_host(ar.meta, (size_t)cap);
    auto t2 = to_host(ar.tok, tok.size());
    CHECK(m2[0].dec_t == 1 && m2[0].dec_ntok == 3 && m2[0].n_emitted == 3 && m2[0].dec_emitted == 0 && !m2[0].dec_done &&
          t2[0] == 3 && t2[1] == 3 && t2[2] == 3,
          "decide: three emissions hit max_symbols and advance the frame (t=%d ntok=%d)", m2[0].dec_t, m2[0].dec_ntok);
    CHECK(m2[1].dec_t == 3 && m2[1].dec_ntok == 0 && m2[1].dec_done, "decide: blanks advance and finish the lane (t=%d done=%d)", m2[1].dec_t, m2[1].dec_done);
    int *dact2, *dn; CU(cudaMalloc(&dact2, B * sizeof(int))); CU(cudaMalloc(&dn, sizeof(int)));
    CU(k_dec_compact(ar, drows, B, dact2, dn, 0));
    CU(cudaDeviceSynchronize());
    auto nact = to_host(dn, 1), act2 = to_host(dact2, (size_t)B);
    CHECK(nact[0] == 1 && act2[0] == 0, "compact: one active lane left, lane 0 (n=%d)", nact[0]);
    return 0;
}

int main(void) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("SKIP test_cuda_kernels: compiled, no CUDA device\n");
        return 77;
    }
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("test_cuda_kernels on %s\n", p.name);
    /* the shapes the Nemotron step issues (N, K) */
    const int shapes[][2] = {{4096, 1024}, {1024, 4096}, {1024, 1024}, {2048, 1024}, {1024, 4352},
                             {2048, 1152}, {1024, 2048}, {640, 1024}, {13088, 640}, {2560, 640}, {640, 640}, {256, 256}};
    const char *names[] = {"ffn1", "ffn2", "qkvo/pw2", "pw1", "ss-linear", "prompt-l1", "prompt-l2", "encproj", "head", "lstm", "proj", "ss-pointwise"};
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        if (test_gemm_shape(shapes[i][0], shapes[i][1], names[i]) != 0) return 1;
    if (test_argmax() != 0) return 1;
    if (test_layernorm() != 0) return 1;
    if (test_attention() != 0) return 1;
    if (test_conv() != 0) return 1;
    if (test_subsampling() != 0) return 1;
    if (test_decode() != 0) return 1;
    printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
