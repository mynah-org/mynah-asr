/* gpu/cuda/kernels.cu — every kernel of the resident streaming step except the
 * GEMM (gemm.cu). Per-stream kernels take one block per lane and read the
 * lane's geometry from its gpu_row; per-row kernels take one block per row.
 * Reductions use fixed-order trees or per-thread sequential chains, so two runs
 * of the same lane with the same inputs produce the same bits whatever else is
 * in the cohort (the gate: tests/test_cuda_stream, "batch identity").
 *
 * Numeric forms follow src/encoder.c, src/subsampling.c and src/decoder.c
 * statement for statement where the order of operations is visible (the
 * LayerNorm's double accumulators, the stable sigmoid, first-index argmax);
 * CPU <-> GPU equality is still a tolerance/transcript gate, because expf,
 * tanhf and the reduction orders are not the CPU's. */
#include "kernels.cuh"

#include <math.h>

/* ------------------------------------------------------------------ helpers */

__device__ __forceinline__ float stable_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* fixed-order block sum in double, 256 threads or fewer */
__device__ double block_sum_d(double v, double *sh) {
    const int tid = threadIdx.x, n = blockDim.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = n >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    const double r = sh[0];
    __syncthreads();
    return r;
}

/* ----------------------------------------------------------- elementwise */

__global__ void layernorm_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                 const float *__restrict__ b, float *__restrict__ out,
                                 int d, int silu_after) {
    __shared__ double sh[256];
    const float *row = x + (size_t)blockIdx.x * (size_t)d;
    float *o = out + (size_t)blockIdx.x * (size_t)d;
    double mu = 0.0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) mu += (double)row[i];
    mu = block_sum_d(mu, sh) / (double)d;
    double var = 0.0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const double c = (double)row[i] - mu;
        var += c * c;
    }
    var = block_sum_d(var, sh) / (double)d;
    const float inv = (float)(1.0 / sqrt(var + 1e-5));
    const float muf = (float)mu;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = ((row[i] - muf) * inv) * w[i] + b[i];
        if (silu_after) v = v * stable_sigmoid(v);
        o[i] = v;
    }
}

cudaError_t k_layernorm(const float *x, const float *w, const float *b, float *out,
                        int rows, int d, int silu_after, cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    layernorm_kernel<<<(unsigned)rows, 256, 0, s>>>(x, w, b, out, d, silu_after);
    return cudaGetLastError();
}

__global__ void residual_kernel(float *__restrict__ x, const float *__restrict__ y,
                                float alpha, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += alpha * y[i];
}
cudaError_t k_residual(float *x, const float *y, float alpha, size_t n, cudaStream_t s) {
    if (n == 0) return cudaSuccess;
    residual_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, y, alpha, n);
    return cudaGetLastError();
}

__global__ void silu_kernel(float *__restrict__ x, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float v = x[i]; x[i] = v * stable_sigmoid(v); }
}
cudaError_t k_silu(float *x, size_t n, cudaStream_t s) {
    if (n == 0) return cudaSuccess;
    silu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, n);
    return cudaGetLastError();
}

__global__ void prompt_cat_kernel(const float *__restrict__ x, const gpu_row *__restrict__ rows,
                                  int d, int np, float *__restrict__ cat) {
    const gpu_row r = rows[blockIdx.x];
    const int dcat = d + np;
    for (int t = 0; t < r.q; t++) {
        const size_t ro = (size_t)(r.row_off + t);
        float *o = cat + ro * (size_t)dcat;
        const float *xi = x + ro * (size_t)d;
        for (int i = threadIdx.x; i < dcat; i += blockDim.x)
            o[i] = i < d ? xi[i] : (i - d == r.prompt ? 1.0f : 0.0f);
    }
}
cudaError_t k_prompt_cat(const float *x, const gpu_row *rows, int B, int d, int np,
                         float *cat, cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    prompt_cat_kernel<<<(unsigned)B, 256, 0, s>>>(x, rows, d, np, cat);
    return cudaGetLastError();
}

/* -------------------------------------------------------------- attention */

/* Row j of the [valid ++ fresh] window of layer li for this lane, K or V. */
__device__ __forceinline__ const float *kv_window_row(const gpu_model_dims &dm,
                                                      const gpu_arena &ar, int li, int which,
                                                      const gpu_row &r, const gpu_slot_meta &m,
                                                      const float *fresh, int j) {
    if (j < m.valid) {
        const int phys = (m.head + j) % dm.left;
        const size_t base = (((size_t)r.slot * dm.n_layers + li) * 2 + which) *
                            (size_t)dm.left * (size_t)dm.d;
        return ar.kv + base + (size_t)phys * (size_t)dm.d;
    }
    return fresh + (size_t)(r.row_off + (j - m.valid)) * (size_t)dm.d;
}

/* dynamic shared: qu[q][dk], qv[q][dk], sc[q][K] */
__global__ void attention_kernel(gpu_model_dims dm, gpu_layer_w L, int li,
                                 const float *__restrict__ relpos_tab, gpu_arena ar,
                                 const gpu_row *__restrict__ rows, const float *__restrict__ qs,
                                 const float *__restrict__ kn, const float *__restrict__ vn,
                                 float *__restrict__ ctx) {
    extern __shared__ float sm[];
    const gpu_row r = rows[blockIdx.x];
    const gpu_slot_meta m = ar.meta[r.slot];
    const int h = blockIdx.y, dk = dm.dk, Q = r.q, K = m.valid + Q;
    const size_t ho = (size_t)h * (size_t)dk;
    float *qu = sm, *qv = sm + (size_t)Q * dk, *sc = qv + (size_t)Q * dk;
    const float scaling = 1.0f / sqrtf((float)dk);

    for (int e = threadIdx.x; e < Q * dk; e += blockDim.x) {
        const int t = e / dk, i = e - t * dk;
        const float qv_ = qs[(size_t)(r.row_off + t) * dm.d + ho + i];
        qu[e] = qv_ + L.bias_u[ho + i];
        qv[e] = qv_ + L.bias_v[ho + i];
    }
    __syncthreads();

    /* rk rows for this K: table row p = (kmax - K) + (K - 1 - valid - t + j) */
    const float *tab = relpos_tab + (size_t)li * (size_t)(2 * dm.kmax - 1) * (size_t)dm.d;
    for (int e = threadIdx.x; e < Q * K; e += blockDim.x) {
        const int t = e / K, j = e - t * K;
        const float *key = kv_window_row(dm, ar, li, 0, r, m, kn, j) + ho;
        const int p = (dm.kmax - K) + (K - 1 - m.valid - t + j);
        const float *rk = tab + (size_t)p * dm.d + ho;
        const float *qut = qu + (size_t)t * dk, *qvt = qv + (size_t)t * dk;
        float ac = 0.0f, bd = 0.0f;
        for (int i = 0; i < dk; i++) {
            ac = fmaf(qut[i], key[i], ac);
            bd = fmaf(qvt[i], rk[i], bd);
        }
        sc[e] = (ac + bd) * scaling;
    }
    __syncthreads();

    /* softmax per row: one warp per row, fixed-order shuffles */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int t = warp; t < Q; t += nw) {
        float *srow = sc + (size_t)t * K;
        float mx = -3.0e38f;
        for (int j = lane; j < K; j += 32) mx = fmaxf(mx, srow[j]);
        mx = warp_max(mx);
        float sum = 0.0f;
        for (int j = lane; j < K; j += 32) { const float v = expf(srow[j] - mx); srow[j] = v; sum += v; }
        sum = warp_sum(sum);
        const float inv = 1.0f / sum;
        for (int j = lane; j < K; j += 32) srow[j] *= inv;
    }
    __syncthreads();

    /* ctx[t][i] = sum_j P[t][j] * V[j][i], sequential in j */
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        for (int t = 0; t < Q; t++) {
            const float *srow = sc + (size_t)t * K;
            float acc = 0.0f;
            for (int j = 0; j < K; j++) {
                const float *v = kv_window_row(dm, ar, li, 1, r, m, vn, j) + ho;
                acc = fmaf(srow[j], v[i], acc);
            }
            ctx[(size_t)(r.row_off + t) * dm.d + ho + i] = acc;
        }
    }
}

cudaError_t k_attention(const gpu_model_dims dm, const gpu_layer_w L, int li,
                        const float *relpos_tab, const gpu_arena ar, const gpu_row *rows,
                        int B, const float *qs, const float *kn, const float *vn, float *ctx,
                        cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    const int qmax = GPU_QMAX_HARD, kmax = dm.left + qmax;
    const size_t shm = ((size_t)2 * qmax * dm.dk + (size_t)qmax * kmax) * sizeof(float);
    int threads = ((dm.dk + 31) / 32) * 32;
    if (threads < 128) threads = 128;
    if (threads > 1024) return cudaErrorInvalidValue;
    dim3 grid((unsigned)B, (unsigned)dm.H);
    attention_kernel<<<grid, (unsigned)threads, shm, s>>>(dm, L, li, relpos_tab, ar, rows,
                                                          qs, kn, vn, ctx);
    return cudaGetLastError();
}

__global__ void kv_commit_kernel(gpu_model_dims dm, int li, gpu_arena ar,
                                 const gpu_row *__restrict__ rows, const float *__restrict__ kn,
                                 const float *__restrict__ vn) {
    const gpu_row r = rows[blockIdx.x];
    const gpu_slot_meta m = ar.meta[r.slot];
    const size_t kbase = (((size_t)r.slot * dm.n_layers + li) * 2 + 0) * (size_t)dm.left * dm.d;
    const size_t vbase = (((size_t)r.slot * dm.n_layers + li) * 2 + 1) * (size_t)dm.left * dm.d;
    for (int t = 0; t < r.q; t++) {
        const int phys = (m.head + m.valid + t) % dm.left;
        const size_t src = (size_t)(r.row_off + t) * dm.d;
        for (int c = threadIdx.x; c < dm.d; c += blockDim.x) {
            ar.kv[kbase + (size_t)phys * dm.d + c] = kn[src + c];
            ar.kv[vbase + (size_t)phys * dm.d + c] = vn[src + c];
        }
    }
}
cudaError_t k_kv_commit(const gpu_model_dims dm, int li, const gpu_arena ar,
                        const gpu_row *rows, int B, const float *kn, const float *vn,
                        cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    kv_commit_kernel<<<(unsigned)B, 256, 0, s>>>(dm, li, ar, rows, kn, vn);
    return cudaGetLastError();
}

__global__ void kv_advance_kernel(gpu_model_dims dm, gpu_arena ar,
                                  const gpu_row *__restrict__ rows, int B) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= B) return;
    const gpu_row r = rows[a];
    gpu_slot_meta *m = &ar.meta[r.slot];
    const int total = m->valid + r.q;
    const int nvalid = total < dm.left ? total : dm.left;
    m->head = (m->head + (total - nvalid)) % dm.left;
    m->valid = nvalid;
    m->t_abs += r.q;
}
cudaError_t k_kv_advance(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                         int B, cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    kv_advance_kernel<<<(unsigned)((B + 127) / 128), 128, 0, s>>>(dm, ar, rows, B);
    return cudaGetLastError();
}

/* ------------------------------------------------------------------- conv */

__global__ void glu_dwconv_kernel(gpu_model_dims dm, gpu_layer_w L, int li, gpu_arena ar,
                                  const gpu_row *__restrict__ rows, const float *__restrict__ h2,
                                  float *__restrict__ out) {
    const gpu_row r = rows[blockIdx.x];
    const int d = dm.d, k = dm.conv_k, Q = r.q;
    float *cache = ar.conv_cache +
                   ((size_t)r.slot * dm.n_layers + li) * (size_t)(k - 1) * (size_t)d;
    float seq[GPU_QMAX_HARD + 15];
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        for (int j = 0; j < k - 1; j++) seq[j] = cache[(size_t)j * d + c];
        for (int t = 0; t < Q; t++) {
            const float *row = h2 + (size_t)(r.row_off + t) * (size_t)(2 * d);
            seq[k - 1 + t] = row[c] * stable_sigmoid(row[d + c]);
        }
        const float *w = L.dw_w + (size_t)c * k;
        for (int t = 0; t < Q; t++) {
            float acc = 0.0f;
            for (int j = 0; j < k; j++) acc = fmaf(w[j], seq[t + j], acc);
            out[(size_t)(r.row_off + t) * d + c] = acc;
        }
        for (int j = 0; j < k - 1; j++) cache[(size_t)j * d + c] = seq[Q + j];
    }
}
cudaError_t k_glu_dwconv(const gpu_model_dims dm, const gpu_layer_w L, int li,
                         const gpu_arena ar, const gpu_row *rows, int B, const float *h2,
                         float *out, cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    if (dm.conv_k > 16) return cudaErrorInvalidValue;
    glu_dwconv_kernel<<<(unsigned)B, 256, 0, s>>>(dm, L, li, ar, rows, h2, out);
    return cudaGetLastError();
}

/* ------------------------------------------------------------ subsampling
 * Padded input of a stage, on the time axis: [init zero (first only)] [cache
 * row] [T rows of x] [zero (last only)], and (2,1) on the frequency axis. */

__device__ __forceinline__ float ss_in0(const gpu_row &r, const gpu_model_dims &dm,
                                        const float *__restrict__ cache0,
                                        const float *__restrict__ mel, int tp, int fp) {
    const int f = fp - 2, lp = r.first ? 2 : 1;
    if (f < 0 || f >= dm.F[0]) return 0.0f;
    if (tp < lp - 1) return 0.0f;
    if (tp == lp - 1) return cache0[f];
    const int t = tp - lp;
    if (t >= r.n_mel) return 0.0f;                    /* the causal right pad */
    return mel[(size_t)(r.mel_off + t) * dm.n_mels + f];
}

/* stage 0: 8 positions per iteration; threads 0..71 stage the 9 taps of each */
__global__ void ss_stage0_kernel(gpu_model_dims dm, gpu_arena ar, const gpu_row *__restrict__ rows,
                                 const float *__restrict__ mel, const float *__restrict__ w,
                                 const float *__restrict__ b, float *__restrict__ out) {
    __shared__ float taps[8][9];
    const gpu_row r = rows[blockIdx.x];
    const int C = dm.C, To = r.to[0], Fo = dm.Fo[0], npos = To * Fo;
    const float *cache0 = ar.ss_cache[0] + (size_t)r.slot * dm.F[0];
    for (int p0 = 0; p0 < npos; p0 += 8) {
        if (threadIdx.x < 72) {
            const int pi = threadIdx.x / 9, tap = threadIdx.x - pi * 9;
            const int pos = p0 + pi;
            float v = 0.0f;
            if (pos < npos) {
                const int to = pos / Fo, fo = pos - to * Fo;
                v = ss_in0(r, dm, cache0, mel, 2 * to + tap / 3, 2 * fo + tap % 3);
            }
            taps[pi][tap] = v;
        }
        __syncthreads();
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float *wc = w + (size_t)c * 9;
            for (int pi = 0; pi < 8; pi++) {
                const int pos = p0 + pi;
                if (pos >= npos) break;
                float acc = 0.0f;
                for (int j = 0; j < 9; j++) acc = fmaf(wc[j], taps[pi][j], acc);
                acc += b[c];
                out[(size_t)(r.pos_off[0] + pos) * C + c] = acc > 0.0f ? acc : 0.0f;
            }
        }
        __syncthreads();
    }
    /* refresh: the last mel frame of the chunk */
    if (r.n_mel > 0)
        for (int f = threadIdx.x; f < dm.F[0]; f += blockDim.x)
            ar.ss_cache[0][(size_t)r.slot * dm.F[0] + f] =
                mel[(size_t)(r.mel_off + r.n_mel - 1) * dm.n_mels + f];
}
cudaError_t k_ss_stage0(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                        int B, const float *mel, const float *w, const float *b, float *out,
                        cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    ss_stage0_kernel<<<(unsigned)B, 256, 0, s>>>(dm, ar, rows, mel, w, b, out);
    return cudaGetLastError();
}

/* stages 1, 2: depthwise, one thread per channel, positions in a loop */
__global__ void ss_dw_kernel(gpu_model_dims dm, int st, gpu_arena ar, const gpu_row *__restrict__ rows,
                             const float *__restrict__ in, const float *__restrict__ w,
                             const float *__restrict__ b, float *__restrict__ out) {
    const gpu_row r = rows[blockIdx.x];
    const int C = dm.C, F = dm.F[st], Fo = dm.Fo[st], Fprev = dm.Fo[st - 1];
    const int T = r.to[st - 1], To = r.to[st], lp = r.first ? 2 : 1;
    float *cache = ar.ss_cache[st] + (size_t)r.slot * (size_t)C * F;
    const float *prev = in + (size_t)r.pos_off[st - 1] * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float *wc = w + (size_t)c * 9;
        for (int to = 0; to < To; to++) {
            for (int fo = 0; fo < Fo; fo++) {
                float acc = 0.0f;
                for (int dt = 0; dt < 3; dt++) {
                    const int tp = 2 * to + dt;
                    for (int df = 0; df < 3; df++) {
                        const int f = 2 * fo + df - 2;
                        float v = 0.0f;
                        if (f >= 0 && f < F && tp >= lp - 1) {
                            if (tp == lp - 1) v = cache[(size_t)c * F + f];
                            else if (tp - lp < T) v = prev[((size_t)(tp - lp) * Fprev + f) * C + c];
                        }
                        acc = fmaf(wc[dt * 3 + df], v, acc);
                    }
                }
                out[(size_t)(r.pos_off[st] + to * Fo + fo) * C + c] = acc + b[c];
            }
        }
    }
    __syncthreads();
    if (T > 0)
        for (int c = threadIdx.x; c < C; c += blockDim.x)
            for (int f = 0; f < F; f++)
                cache[(size_t)c * F + f] = prev[((size_t)(T - 1) * Fprev + f) * C + c];
}
cudaError_t k_ss_dw(const gpu_model_dims dm, int stage, const gpu_arena ar, const gpu_row *rows,
                    int B, const float *in, const float *w, const float *b, float *out,
                    cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    if (stage < 1 || stage >= GPU_SS_STAGES) return cudaErrorInvalidValue;
    ss_dw_kernel<<<(unsigned)B, 256, 0, s>>>(dm, stage, ar, rows, in, w, b, out);
    return cudaGetLastError();
}

__global__ void ss_flatten_kernel(gpu_model_dims dm, const gpu_row *__restrict__ rows,
                                  const float *__restrict__ s2, float *__restrict__ flat) {
    const gpu_row r = rows[blockIdx.x];
    const int C = dm.C, Fo = dm.Fo[GPU_SS_STAGES - 1], CF = C * Fo;
    for (int t = 0; t < r.q; t++)
        for (int idx = threadIdx.x; idx < CF; idx += blockDim.x) {
            const int c = idx / Fo, f = idx - c * Fo;
            flat[(size_t)(r.row_off + t) * CF + idx] =
                s2[(size_t)(r.pos_off[GPU_SS_STAGES - 1] + t * Fo + f) * C + c];
        }
}
cudaError_t k_ss_flatten(const gpu_model_dims dm, const gpu_row *rows, int B, const float *s2,
                         float *flat, cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    ss_flatten_kernel<<<(unsigned)B, 256, 0, s>>>(dm, rows, s2, flat);
    return cudaGetLastError();
}

/* ----------------------------------------------------------------- decode */

__global__ void dec_begin_kernel(gpu_arena ar, const gpu_row *__restrict__ rows, int B) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= B) return;
    gpu_slot_meta *m = &ar.meta[rows[a].slot];
    m->dec_t = 0; m->dec_emitted = 0; m->dec_done = rows[a].q <= 0; m->dec_ntok = 0;
}
cudaError_t k_dec_begin(const gpu_arena ar, const gpu_row *rows, int B, cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    dec_begin_kernel<<<(unsigned)((B + 127) / 128), 128, 0, s>>>(ar, rows, B);
    return cudaGetLastError();
}

__global__ void dec_compact_kernel(gpu_arena ar, const gpu_row *__restrict__ rows, int B,
                                   int *__restrict__ active, int *__restrict__ n_out) {
    __shared__ int flag[1024];
    const int tid = threadIdx.x;
    for (int a = tid; a < 1024; a += blockDim.x)
        flag[a] = (a < B && !ar.meta[rows[a].slot].dec_done) ? 1 : 0;
    __syncthreads();
    /* exclusive scan, Hillis-Steele over 1024 (deterministic, integer) */
    __shared__ int scan[1024];
    for (int a = tid; a < 1024; a += blockDim.x) scan[a] = flag[a];
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        int vals[4];
        for (int a = tid, i = 0; a < 1024; a += blockDim.x, i++)
            vals[i] = a >= off ? scan[a - off] : 0;
        __syncthreads();
        for (int a = tid, i = 0; a < 1024; a += blockDim.x, i++) scan[a] += vals[i];
        __syncthreads();
    }
    for (int a = tid; a < B; a += blockDim.x)
        if (flag[a]) active[scan[a] - 1] = a;
    if (tid == 0) *n_out = B > 0 ? scan[B - 1] : 0;
}
cudaError_t k_dec_compact(const gpu_arena ar, const gpu_row *rows, int B, int *active,
                          int *n_out, cudaStream_t s) {
    if (B > 1024) return cudaErrorInvalidValue;
    dec_compact_kernel<<<1, 256, 0, s>>>(ar, rows, B, active, n_out);
    return cudaGetLastError();
}

__global__ void dec_joint_kernel(gpu_model_dims dm, gpu_arena ar, const gpu_row *__restrict__ rows,
                                 const int *__restrict__ active, const float *__restrict__ enc,
                                 float *__restrict__ jin) {
    const int a = blockIdx.x;
    const gpu_row r = rows[active[a]];
    const gpu_slot_meta m = ar.meta[r.slot];
    const float *e = enc + (size_t)(r.row_off + m.dec_t) * dm.dout;
    const float *g = ar.dec_g + (size_t)r.slot * dm.Hdec;
    float *o = jin + (size_t)a * dm.Hdec;
    for (int i = threadIdx.x; i < dm.Hdec; i += blockDim.x) {
        const float v = e[i] + g[i];
        o[i] = v > 0.0f ? v : 0.0f;
    }
}
cudaError_t k_dec_joint(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                        const int *active, int n, const float *enc, float *jin,
                        cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    dec_joint_kernel<<<(unsigned)n, 128, 0, s>>>(dm, ar, rows, active, enc, jin);
    return cudaGetLastError();
}

__global__ void dec_argmax_kernel(const float *__restrict__ logits, int V, int *__restrict__ am) {
    __shared__ float bv[256];
    __shared__ int bi[256];
    const float *row = logits + (size_t)blockIdx.x * V;
    float best = -3.0e38f;
    int besti = 0x7fffffff;
    for (int k = threadIdx.x; k < V; k += blockDim.x) {
        const float v = row[k];
        if (v > best) { best = v; besti = k; }
    }
    bv[threadIdx.x] = best; bi[threadIdx.x] = besti;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float v2 = bv[threadIdx.x + s];
            const int i2 = bi[threadIdx.x + s];
            if (v2 > bv[threadIdx.x] || (v2 == bv[threadIdx.x] && i2 < bi[threadIdx.x])) {
                bv[threadIdx.x] = v2; bi[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) am[blockIdx.x] = bi[0] == 0x7fffffff ? 0 : bi[0];
}
cudaError_t k_dec_argmax(const float *logits, int V, int n, int *am, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    dec_argmax_kernel<<<(unsigned)n, 256, 0, s>>>(logits, V, am);
    return cudaGetLastError();
}

__global__ void dec_decide_kernel(gpu_model_dims dm, gpu_arena ar, const gpu_row *__restrict__ rows,
                                  const int *__restrict__ active, int n, const int *__restrict__ am,
                                  int *__restrict__ emit) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    const gpu_row r = rows[active[a]];
    gpu_slot_meta *m = &ar.meta[r.slot];
    const int tok = am[a];
    int e = 0;
    if (tok == dm.blank) {
        m->dec_t++;
        m->dec_emitted = 0;
    } else {
        if (m->dec_ntok < ar.tok_cap) {
            ar.tok[(size_t)r.slot * ar.tok_cap + m->dec_ntok] = tok;
            ar.tok_frame[(size_t)r.slot * ar.tok_cap + m->dec_ntok] = (int)(m->t_abs + m->dec_t);
            m->dec_ntok++;
        }
        m->n_emitted++;
        m->dec_emitted++;
        e = 1;
        if (m->dec_emitted >= dm.max_symbols) { m->dec_t++; m->dec_emitted = 0; }
    }
    if (m->dec_t >= r.q) m->dec_done = 1;
    emit[a] = e;
}
cudaError_t k_dec_decide(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                         const int *active, int n, const int *am, int *emit, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    dec_decide_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(dm, ar, rows, active, n, am, emit);
    return cudaGetLastError();
}

__global__ void dec_gather_emb_kernel(const float *__restrict__ emb, int H, const int *__restrict__ am,
                                      float *__restrict__ x) {
    const int a = blockIdx.x;
    const float *src = emb + (size_t)am[a] * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) x[(size_t)a * H + i] = src[i];
}
cudaError_t k_dec_gather_emb(const float *emb, int H, const int *am, int n, float *x,
                             cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    dec_gather_emb_kernel<<<(unsigned)n, 128, 0, s>>>(emb, H, am, x);
    return cudaGetLastError();
}

__global__ void dec_gather_h_kernel(gpu_model_dims dm, gpu_arena ar, const gpu_row *__restrict__ rows,
                                    const int *__restrict__ active, int layer,
                                    float *__restrict__ h_rows) {
    const int a = blockIdx.x;
    const int slot = rows[active[a]].slot;
    const float *h = ar.dec_h + ((size_t)slot * dm.pred_layers + layer) * dm.Hdec;
    for (int i = threadIdx.x; i < dm.Hdec; i += blockDim.x) h_rows[(size_t)a * dm.Hdec + i] = h[i];
}
cudaError_t k_dec_gather_h(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                           const int *active, int n, int layer, float *h_rows, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    dec_gather_h_kernel<<<(unsigned)n, 128, 0, s>>>(dm, ar, rows, active, layer, h_rows);
    return cudaGetLastError();
}

__global__ void dec_lstm_gates_kernel(gpu_model_dims dm, gpu_arena ar, const gpu_row *__restrict__ rows,
                                      const int *__restrict__ active, const int *__restrict__ emit,
                                      int layer, const float *__restrict__ z, float *__restrict__ x_next) {
    const int a = blockIdx.x;
    const int H = dm.Hdec;
    const int slot = rows[active[a]].slot;
    float *h = ar.dec_h + ((size_t)slot * dm.pred_layers + layer) * H;
    float *c = ar.dec_c + ((size_t)slot * dm.pred_layers + layer) * H;
    const float *zr = z + (size_t)a * 4 * H;
    const int does = emit[a];
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        if (does) {
            const float ig = stable_sigmoid(zr[i]);
            const float fg = stable_sigmoid(zr[H + i]);
            const float gg = tanhf(zr[2 * H + i]);
            const float og = stable_sigmoid(zr[3 * H + i]);
            const float cn = fg * c[i] + ig * gg;
            const float hn = og * tanhf(cn);
            c[i] = cn; h[i] = hn;
            x_next[(size_t)a * H + i] = hn;
        } else {
            x_next[(size_t)a * H + i] = h[i];
        }
    }
}
cudaError_t k_dec_lstm_gates(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                             const int *active, const int *emit, int n, int layer, const float *z,
                             float *x_next, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    dec_lstm_gates_kernel<<<(unsigned)n, 128, 0, s>>>(dm, ar, rows, active, emit, layer, z, x_next);
    return cudaGetLastError();
}

__global__ void dec_commit_g_kernel(gpu_model_dims dm, gpu_arena ar, const gpu_row *__restrict__ rows,
                                    const int *__restrict__ active, const int *__restrict__ emit,
                                    const int *__restrict__ am, const float *__restrict__ gtmp) {
    const int a = blockIdx.x;
    if (!emit[a]) return;
    const int slot = rows[active[a]].slot;
    float *g = ar.dec_g + (size_t)slot * dm.Hdec;
    for (int i = threadIdx.x; i < dm.Hdec; i += blockDim.x) g[i] = gtmp[(size_t)a * dm.Hdec + i];
    if (threadIdx.x == 0) ar.meta[slot].last_token = am[a];
}
cudaError_t k_dec_commit_g(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                           const int *active, const int *emit, const int *am, int n,
                           const float *gtmp, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    dec_commit_g_kernel<<<(unsigned)n, 128, 0, s>>>(dm, ar, rows, active, emit, am, gtmp);
    return cudaGetLastError();
}

/* ------------------------------------------------------------------ slots */

__global__ void slots_reset_kernel(gpu_model_dims dm, gpu_arena ar, const int *__restrict__ slots,
                                   const float *__restrict__ sos_h, const float *__restrict__ sos_c,
                                   const float *__restrict__ sos_g) {
    const int slot = slots[blockIdx.x];
    const size_t nconv = (size_t)dm.n_layers * (size_t)(dm.conv_k - 1) * dm.d;
    float *cc = ar.conv_cache + (size_t)slot * nconv;
    for (size_t i = threadIdx.x; i < nconv; i += blockDim.x) cc[i] = 0.0f;
    for (int st = 0; st < GPU_SS_STAGES; st++) {
        const size_t n = (size_t)(st == 0 ? 1 : dm.C) * dm.F[st];
        float *c = ar.ss_cache[st] + (size_t)slot * n;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) c[i] = 0.0f;
    }
    const size_t nh = (size_t)dm.pred_layers * dm.Hdec;
    for (size_t i = threadIdx.x; i < nh; i += blockDim.x) {
        ar.dec_h[(size_t)slot * nh + i] = sos_h[i];
        ar.dec_c[(size_t)slot * nh + i] = sos_c[i];
    }
    for (int i = threadIdx.x; i < dm.Hdec; i += blockDim.x)
        ar.dec_g[(size_t)slot * dm.Hdec + i] = sos_g[i];
    if (threadIdx.x == 0) {
        gpu_slot_meta *m = &ar.meta[slot];
        m->valid = 0; m->head = 0; m->t_abs = 0;
        m->last_token = dm.blank; m->n_emitted = 0;
        m->dec_t = 0; m->dec_emitted = 0; m->dec_done = 1; m->dec_ntok = 0;
    }
}
cudaError_t k_slots_reset(const gpu_model_dims dm, const gpu_arena ar, const int *slots, int n,
                          const float *sos_h, const float *sos_c, const float *sos_g,
                          cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    slots_reset_kernel<<<(unsigned)n, 256, 0, s>>>(dm, ar, slots, sos_h, sos_c, sos_g);
    return cudaGetLastError();
}
