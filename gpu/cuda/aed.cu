/* gpu/cuda/aed.cu — the offline AED engine on one GPU (gpu/aed_gpu.h).
 *
 * Design: .work/canary-180m-l4.md section 5. One device, one stream, one
 * thread. Weights are uploaded once at open, every scratch buffer is sized at
 * open for `max_items` segments of at most T_max encoder frames; a call
 * allocates nothing on the device. Rows of a wave are PACKED ([sum T, d], no
 * padding): every per-segment kernel reads its segment's offset and length from
 * small tables, every GEMM is row-stable, so a segment's result does not depend
 * on what it was packed with. */
extern "C" {
#include "../aed_gpu.h"
#include "../../src/decoder_aed.h"
#include "../../src/encoder.h"
#include "../../src/weights.h"
#include "../../vendor/cJSON.h"
}
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>
#include <unordered_map>
#include <vector>

#define AED_THREADS 128      /* attention / norm blocks: a power of two */

/* ------------------------------------------------------------------ types */

struct enc_lw {
    const float *ln_ff1_w, *ln_ff1_b, *ff1_w1, *ff1_b1, *ff1_w2, *ff1_b2;
    const float *ln_att_w, *ln_att_b, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    const float *bias_u, *bias_v;
    const float *ln_conv_w, *ln_conv_b, *pw1_w, *pw1_b, *dw_w, *dw_b;
    const float *cn_scale, *cn_shift, *cn_w, *cn_b, *pw2_w, *pw2_b;
    const float *ln_ff2_w, *ln_ff2_b, *ff2_w1, *ff2_b1, *ff2_w2, *ff2_b2;
    const float *ln_out_w, *ln_out_b;
};

struct asr_aed_gpu {
    asr_aed_gpu_cfg cfg;
    int device = 0;
    cudaStream_t stream = nullptr;
    std::string devname;
    char err[512] = {0};
    int dead = 0;
    /* the pack, through the library's own loaders */
    cJSON *jcfg = nullptr;
    mynah_asr_safetensors *st = nullptr;
    mynah_asr_encoder enc;
    int enc_ok = 0;
    /* encoder geometry */
    int L = 0, d = 0, H = 0, dk = 0, ffn = 0, K = 0, pc = 0, d_out = 0;
    int t_max = 0, max_items = 0, r_max = 0;
    int bn = 0;                 /* conv norm = folded batch_norm (else layer_norm) */
    /* device weights */
    std::vector<enc_lw> lw;
    float *d_rk = nullptr;      /* [L][2 t_max - 1][d]: the rel-pos table */
    const float *d_ep_w = nullptr, *d_ep_b = nullptr;
    /* encoder scratch */
    float *xs = nullptr, *xn = nullptr, *tmp = nullptr, *tmp2 = nullptr, *qs = nullptr,
          *ks = nullptr, *vs = nullptr, *ctx = nullptr, *enc_out = nullptr;
    int *d_ints = nullptr, *h_ints = nullptr;   /* row_seq [r_max] | seq_off | seq_T */
    float *h_x = nullptr, *h_out = nullptr;     /* pinned staging */
    /* the last encode: what aed_decode may find resident */
    int last_n = 0;
    std::vector<int> last_T, last_off;
    /* GEMM arms */
    int tc = 0;
    std::unordered_map<const float *, const unsigned short *> w16;
    float *tc_ws = nullptr;
    size_t tc_ws_floats = 0;
    size_t vram_weights = 0, vram_scratch = 0, vram_total = 0;
    asr_aed_gpu_stats st_ = {};
};
typedef asr_aed_gpu aed_gpu;

/* ---------------------------------------------------------------- helpers */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static void set_dead(aed_gpu *e, const char *what, cudaError_t c) {
    snprintf(e->err, sizeof(e->err), "%s: %s", what, cudaGetErrorString(c));
    e->dead = 1;
    e->st_.errors++;
    fprintf(stderr, "mynah-asr-server-cuda: DEVICE ERROR (aed) %s\n", e->err);
}

#define CK(e, what, call) do { cudaError_t c_ = (call); if (c_ != cudaSuccess) { set_dead((e), (what), c_); return -1; } } while (0)

static int dalloc(aed_gpu *e, void **p, size_t bytes, size_t *acct, const char *what) {
    cudaError_t c = cudaMalloc(p, bytes > 0 ? bytes : 4);
    if (c != cudaSuccess) { set_dead(e, what, c); return -1; }
    if (acct) *acct += bytes;
    return 0;
}

/* a host f32 tensor -> device; NULL host = NULL device (an optional tensor) */
static const float *up_opt(aed_gpu *e, const float *host, size_t n, const char *what) {
    if (!host || e->dead) return nullptr;
    void *dp = nullptr;
    if (dalloc(e, &dp, n * sizeof(float), &e->vram_weights, what) != 0) return nullptr;
    cudaError_t c = cudaMemcpy(dp, host, n * sizeof(float), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) { set_dead(e, what, c); return nullptr; }
    return (const float *)dp;
}
static const float *up_req(aed_gpu *e, const float *host, size_t n, const char *what) {
    if (!host) {
        if (!e->err[0]) snprintf(e->err, sizeof(e->err), "%s: missing f32 tensor (the GPU uploads f32)", what);
        return nullptr;
    }
    return up_opt(e, host, n, what);
}
static const float *qf32(const mynah_asr_qmat *m) {
    return m->qtype == MYNAH_ASR_Q_F32 ? m->f32 : nullptr;
}

/* C[M,N] = A[M,K] W[N,K]^T (+bias) (+C) (act): the own row-stable f32 kernel,
 * or own-tc (bf16 weights, f32 accumulate; split count from the shape only) */
static int gemm(aed_gpu *e, const float *A, int lda, const float *W, const float *bias, float *C,
                int ldc, int M, int N, int K, int accumulate, int act) {
    if (M <= 0) return 0;
    if (e->tc) {
        auto it = e->w16.find(W);
        if (it == e->w16.end()) {
            snprintf(e->err, sizeof(e->err), "own-tc: a GEMM weight has no bf16 copy (N=%d K=%d)", N, K);
            e->dead = 1;
            e->st_.errors++;
            return -1;
        }
        CK(e, "gemm tc", k_gemm_wt_tc(A, lda, it->second, bias, C, ldc, M, N, K, accumulate, act,
                                      e->tc_ws, e->tc_ws_floats, e->stream));
        return 0;
    }
    CK(e, "gemm", k_gemm_wt(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, e->stream));
    return 0;
}

/* own-tc: a bf16 copy of a GEMM weight, keyed by its f32 device pointer, and
 * the split-K workspace for the largest M this weight is called with */
static int tc_prepare(aed_gpu *e, const float *W, int N, int K, int Mmax) {
    if (!e->tc || !W) return 0;
    const size_t ws = k_gemm_tc_workspace_floats(Mmax, N, K);
    if (ws > e->tc_ws_floats) e->tc_ws_floats = ws;
    if (e->w16.count(W)) return 0;
    void *dp = nullptr;
    if (dalloc(e, &dp, (size_t)N * (size_t)K * sizeof(unsigned short), &e->vram_weights, "bf16 weight") != 0) return -1;
    CK(e, "bf16 weight", k_f32_to_bf16(W, (unsigned short *)dp, (size_t)N * (size_t)K, e->stream));
    e->w16[W] = (const unsigned short *)dp;
    return 0;
}

/* -------------------------------------------------------------- kernels */

__device__ __forceinline__ float aed_sigmoid(float x) {
    /* the library's stable sigmoid (src/qmat.h mynah_asr_sigmoid) */
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float ex = expf(x);
    return ex / (1.0f + ex);
}

/* Fixed-shape tree reductions over a block of AED_THREADS: the partial a
 * thread holds depends only on the thread index, so the result depends only on
 * the row's own data. red: AED_THREADS floats of shared memory. */
__device__ float block_max(float v, float *red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = AED_THREADS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}
__device__ float block_sum(float v, float *red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = AED_THREADS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

/* Full-context rel-pos self-attention of the encoder, one block per (row,
 * head). Row r belongs to segment b = row_seq[r] (rows seq_off[b] ..
 * seq_off[b] + seq_T[b]), query position t = r - seq_off[b]:
 *   s[j] = ((q + u) . k_j + (q + v) . rk(T)[T-1+j-t]) * scale,  j in [0, T)
 * with rk(T)[p] = RK[p + kmax - T] (the library's table), i.e. RK[kmax-1+j-t];
 * softmax over j; ctx = sum_j p_j v_j. src/encoder.c attention(), left < 0.
 * Shared: qu[dk] qv[dk] red[AED_THREADS] s[T]. */
__global__ void __launch_bounds__(AED_THREADS)
enc_attn_kernel(const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
                const float *__restrict__ bu, const float *__restrict__ bv, const float *__restrict__ rk,
                int kmax, const int *__restrict__ row_seq, const int *__restrict__ seq_off,
                const int *__restrict__ seq_T, int d, int dk, float scale, float *__restrict__ ctx) {
    extern __shared__ float sm[];
    float *qu = sm, *qv = sm + dk, *red = sm + 2 * dk, *sc = red + AED_THREADS;
    const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int b = row_seq[r], off = seq_off[b], T = seq_T[b], t = r - off, ho = h * dk;
    for (int c = tid; c < dk; c += AED_THREADS) {
        const float qq = q[(size_t)r * d + ho + c];
        qu[c] = qq + bu[ho + c];
        qv[c] = qq + bv[ho + c];
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (int j = tid; j < T; j += AED_THREADS) {
        const float *kr = k + (size_t)(off + j) * d + ho;
        const float *pr = rk + (size_t)(kmax - 1 + j - t) * d + ho;
        float ac = 0.0f, bd = 0.0f;
        for (int c = 0; c < dk; c++) {
            ac = fmaf(qu[c], kr[c], ac);
            bd = fmaf(qv[c], pr[c], bd);
        }
        const float s = (ac + bd) * scale;
        sc[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int j = tid; j < T; j += AED_THREADS) {
        const float ex = expf(sc[j] - mx);
        sc[j] = ex;
        sum += ex;
    }
    sum = block_sum(sum, red);
    const float inv = 1.0f / sum;
    for (int j = tid; j < T; j += AED_THREADS) sc[j] *= inv;
    __syncthreads();
    for (int c = tid; c < dk; c += AED_THREADS) {
        float acc = 0.0f;
        for (int j = 0; j < T; j++) acc = fmaf(sc[j], v[(size_t)(off + j) * d + ho + c], acc);
        ctx[(size_t)r * d + ho + c] = acc;
    }
}

/* GLU over the channels: g[r][c] = h2[r][c] * sigmoid(h2[r][d + c]) */
__global__ void glu_kernel(const float *__restrict__ h2, int d, size_t n, float *__restrict__ g) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t r = i / (size_t)d, c = i % (size_t)d;
    const float a = h2[r * 2 * (size_t)d + c], bg = h2[r * 2 * (size_t)d + (size_t)d + c];
    g[i] = a * aed_sigmoid(bg);
}

/* Depthwise conv over time inside each segment ('same': pad (K-1)/2 per side,
 * or causal: pad K-1 left), + bias, then the folded batch_norm affine and SiLU
 * when `scale` is given (else the caller applies LayerNorm+SiLU). Out-of-range
 * taps are skipped, as src/encoder.c conv_module bounds its loop. */
__global__ void dwconv_kernel(const float *__restrict__ g, const int *__restrict__ row_seq,
                              const int *__restrict__ seq_off, const int *__restrict__ seq_T,
                              const float *__restrict__ w, const float *__restrict__ bias,
                              const float *__restrict__ scale, const float *__restrict__ shift,
                              int K, int pc, int d, float *__restrict__ out) {
    const int r = blockIdx.x;
    const int b = row_seq[r], off = seq_off[b], T = seq_T[b], t = r - off;
    const int j0 = t - pc < 0 ? pc - t : 0;
    const int j1 = t - pc + K > T ? T - t + pc : K;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float acc = bias ? bias[c] : 0.0f;
        for (int j = j0; j < j1; j++)
            acc += w[(size_t)c * K + j] * g[(size_t)(off + t - pc + j) * d + c];
        if (scale) {
            acc = acc * scale[c] + shift[c];
            acc = acc * aed_sigmoid(acc);
        }
        out[(size_t)r * d + c] = acc;
    }
}

/* ------------------------------------------------------------ the encoder */

/* xs [R, d] (packed, uploaded) -> enc_out [R, d_out]; tables on the device */
static int enc_layers(aed_gpu *e, int R) {
    const int d = e->d, ffn = e->ffn;
    const size_t P = (size_t)(2 * e->t_max - 1);
    const int *row_seq = e->d_ints, *seq_off = e->d_ints + e->r_max, *seq_T = seq_off + e->max_items;
    const float scale = 1.0f / sqrtf((float)e->dk);
    const size_t smem = (size_t)(2 * e->dk + AED_THREADS + e->t_max) * sizeof(float);
    for (int li = 0; li < e->L; li++) {
        const enc_lw &W = e->lw[(size_t)li];
        /* ½ FFN1 */
        CK(e, "ln", k_layernorm(e->xs, W.ln_ff1_w, W.ln_ff1_b, e->xn, R, d, 0, e->stream));
        if (gemm(e, e->xn, d, W.ff1_w1, W.ff1_b1, e->tmp2, ffn, R, ffn, d, 0, 2) != 0) return -1;
        if (gemm(e, e->tmp2, ffn, W.ff1_w2, W.ff1_b2, e->tmp, d, R, d, ffn, 0, 0) != 0) return -1;
        CK(e, "residual", k_residual(e->xs, e->tmp, 0.5f, (size_t)R * d, e->stream));
        /* MHSA, full context, rel-pos */
        CK(e, "ln", k_layernorm(e->xs, W.ln_att_w, W.ln_att_b, e->xn, R, d, 0, e->stream));
        if (gemm(e, e->xn, d, W.q_w, W.q_b, e->qs, d, R, d, d, 0, 0) != 0) return -1;
        if (gemm(e, e->xn, d, W.k_w, W.k_b, e->ks, d, R, d, d, 0, 0) != 0) return -1;
        if (gemm(e, e->xn, d, W.v_w, W.v_b, e->vs, d, R, d, d, 0, 0) != 0) return -1;
        enc_attn_kernel<<<dim3((unsigned)R, (unsigned)e->H), AED_THREADS, smem, e->stream>>>(
            e->qs, e->ks, e->vs, W.bias_u, W.bias_v, e->d_rk + (size_t)li * P * (size_t)d, e->t_max,
            row_seq, seq_off, seq_T, d, e->dk, scale, e->ctx);
        CK(e, "enc attention", cudaGetLastError());
        if (gemm(e, e->ctx, d, W.o_w, W.o_b, e->xs, d, R, d, d, 1, 0) != 0) return -1;
        /* conv module */
        CK(e, "ln", k_layernorm(e->xs, W.ln_conv_w, W.ln_conv_b, e->xn, R, d, 0, e->stream));
        if (gemm(e, e->xn, d, W.pw1_w, W.pw1_b, e->tmp2, 2 * d, R, 2 * d, d, 0, 0) != 0) return -1;
        {
            const size_t n = (size_t)R * d;
            glu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, e->stream>>>(e->tmp2, d, n, e->tmp);
            CK(e, "glu", cudaGetLastError());
        }
        dwconv_kernel<<<(unsigned)R, 256, 0, e->stream>>>(e->tmp, row_seq, seq_off, seq_T, W.dw_w, W.dw_b,
                                                         e->bn ? W.cn_scale : nullptr, W.cn_shift,
                                                         e->K, e->pc, d, e->xn);
        CK(e, "dwconv", cudaGetLastError());
        if (!e->bn) CK(e, "conv ln", k_layernorm(e->xn, W.cn_w, W.cn_b, e->xn, R, d, 1, e->stream));
        if (gemm(e, e->xn, d, W.pw2_w, W.pw2_b, e->xs, d, R, d, d, 1, 0) != 0) return -1;
        /* ½ FFN2 */
        CK(e, "ln", k_layernorm(e->xs, W.ln_ff2_w, W.ln_ff2_b, e->xn, R, d, 0, e->stream));
        if (gemm(e, e->xn, d, W.ff2_w1, W.ff2_b1, e->tmp2, ffn, R, ffn, d, 0, 2) != 0) return -1;
        if (gemm(e, e->tmp2, ffn, W.ff2_w2, W.ff2_b2, e->tmp, d, R, d, ffn, 0, 0) != 0) return -1;
        CK(e, "residual", k_residual(e->xs, e->tmp, 0.5f, (size_t)R * d, e->stream));
        /* output LN, in place */
        CK(e, "ln", k_layernorm(e->xs, W.ln_out_w, W.ln_out_b, e->xs, R, d, 0, e->stream));
    }
    /* post: the encoder projector when the pack has one, else identity */
    if (e->d_ep_w) {
        if (gemm(e, e->xs, d, e->d_ep_w, e->d_ep_b, e->enc_out, e->d_out, R, e->d_out, d, 0, 0) != 0) return -1;
    }
    return 0;
}

/* the offload's encode hook (src/mynah_asr.h mynah_asr_offload) */
static int hook_encode(void *ud, const float *const *feats, const int *t_mel, int n, int n_mels,
                       const int *prompt_ids, float **outs, int *t_outs) {
    aed_gpu *e = (aed_gpu *)ud;
    (void)prompt_ids;           /* no prompt projector: refused at open */
    for (int b = 0; b < n; b++) outs[b] = nullptr;
    e->last_n = 0;
    if (e->dead) return -1;
    if (n > e->max_items) {
        snprintf(e->err, sizeof(e->err), "a wave of %d segments exceeds --batch %d", n, e->max_items);
        return -1;
    }
    cudaSetDevice(e->device);
    const int d = e->d;
    int *row_seq = e->h_ints, *seq_off = e->h_ints + e->r_max, *seq_T = seq_off + e->max_items;
    /* subsampling on the host, the library's own code, then the packed rows */
    const double t0 = now_ms();
    int R = 0;
    std::vector<int> T((size_t)n, 0);
    for (int b = 0; b < n; b++) {
        int Tb = 0;
        float *x = mynah_asr_subsampling_forward(&e->enc.ss, feats[b], t_mel[b], n_mels, &Tb);
        if (!x) { snprintf(e->err, sizeof(e->err), "subsampling failed (segment %d)", b); return -1; }
        if (Tb > e->t_max) {
            free(x);
            snprintf(e->err, sizeof(e->err), "a segment of %d encoder frames exceeds T_max %d", Tb, e->t_max);
            return -1;
        }
        if (e->enc.xscale != 1.0f)
            for (size_t i = 0; i < (size_t)Tb * (size_t)d; i++) x[i] *= e->enc.xscale;
        memcpy(e->h_x + (size_t)R * d, x, (size_t)Tb * (size_t)d * sizeof(float));
        free(x);
        seq_off[b] = R;
        seq_T[b] = Tb;
        for (int t = 0; t < Tb; t++) row_seq[R + t] = b;
        T[(size_t)b] = Tb;
        R += Tb;
    }
    const double t1 = now_ms();
    e->st_.host_ss_ms += t1 - t0;
    if (R > 0) {
        CK(e, "h2d", cudaMemcpyAsync(e->xs, e->h_x, (size_t)R * d * sizeof(float), cudaMemcpyHostToDevice, e->stream));
        CK(e, "h2d", cudaMemcpyAsync(e->d_ints, e->h_ints, (size_t)(e->r_max + 2 * e->max_items) * sizeof(int),
                                     cudaMemcpyHostToDevice, e->stream));
        if (enc_layers(e, R) != 0) return -1;
        CK(e, "d2h", cudaMemcpyAsync(e->h_out, e->enc_out, (size_t)R * e->d_out * sizeof(float),
                                     cudaMemcpyDeviceToHost, e->stream));
        CK(e, "sync", cudaStreamSynchronize(e->stream));
    }
    for (int b = 0; b < n; b++) {
        const size_t bytes = (size_t)T[(size_t)b] * (size_t)e->d_out * sizeof(float);
        outs[b] = (float *)malloc(bytes > 0 ? bytes : sizeof(float));
        if (!outs[b]) {
            for (int i = 0; i < b; i++) { free(outs[i]); outs[i] = nullptr; }
            snprintf(e->err, sizeof(e->err), "out of host memory");
            return -1;
        }
        if (bytes) memcpy(outs[b], e->h_out + (size_t)seq_off[b] * e->d_out, bytes);
        t_outs[b] = T[(size_t)b];
    }
    /* resident for aed_decode (from_encode) */
    e->last_n = n;
    e->last_T.assign(T.begin(), T.end());
    e->last_off.assign(seq_off, seq_off + n);
    e->st_.enc_calls++;
    e->st_.enc_segments += (unsigned long)n;
    e->st_.enc_rows += (unsigned long)R;
    e->st_.enc_ms += now_ms() - t1;
    return 0;
}

/* --------------------------------------------------------------------- open */

static cJSON *load_json(const char *dir, const char *file) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *f = fopen(path, "rb");
    if (!f) return nullptr;
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = len > 0 ? (char *)malloc((size_t)len + 1) : nullptr;
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); fclose(f); return nullptr; }
    buf[len] = '\0';
    fclose(f);
    cJSON *j = cJSON_Parse(buf);
    free(buf);
    return j;
}
static int jint(const cJSON *o, const char *k, int dflt) {
    const cJSON *j = o ? cJSON_GetObjectItem(o, k) : nullptr;
    return j && cJSON_IsNumber(j) ? j->valueint : dflt;
}
static const char *jstr(const cJSON *o, const char *k) {
    const cJSON *j = o ? cJSON_GetObjectItem(o, k) : nullptr;
    return j && cJSON_IsString(j) ? j->valuestring : nullptr;
}

static int upload_encoder(aed_gpu *e) {
    const mynah_asr_encoder &en = e->enc;
    const size_t d = (size_t)e->d, ffn = (size_t)e->ffn, K = (size_t)e->K;
    e->lw.resize((size_t)e->L);
    for (int li = 0; li < e->L; li++) {
        const mynah_asr_enc_layer *S = &en.layers[li];
        enc_lw &W = e->lw[(size_t)li];
        memset(&W, 0, sizeof(W));
#define REQ(dst, src, n) do { W.dst = up_req(e, (src), (n), #dst); if (!W.dst) return -1; } while (0)
#define OPT(dst, src, n) do { W.dst = up_opt(e, (src), (n), #dst); if (e->dead) return -1; } while (0)
        REQ(ln_ff1_w, S->ln_ff1_w, d); REQ(ln_ff1_b, S->ln_ff1_b, d);
        REQ(ff1_w1, qf32(&S->ff1_w1), ffn * d); OPT(ff1_b1, S->ff1_b1, ffn);
        REQ(ff1_w2, qf32(&S->ff1_w2), d * ffn); OPT(ff1_b2, S->ff1_b2, d);
        REQ(ln_att_w, S->ln_att_w, d); REQ(ln_att_b, S->ln_att_b, d);
        REQ(q_w, qf32(&S->q_w), d * d); OPT(q_b, S->q_b, d);
        REQ(k_w, qf32(&S->k_w), d * d); OPT(k_b, S->k_b, d);
        REQ(v_w, qf32(&S->v_w), d * d); OPT(v_b, S->v_b, d);
        REQ(o_w, qf32(&S->o_w), d * d); OPT(o_b, S->o_b, d);
        REQ(bias_u, S->bias_u, d); REQ(bias_v, S->bias_v, d);
        REQ(ln_conv_w, S->ln_conv_w, d); REQ(ln_conv_b, S->ln_conv_b, d);
        REQ(pw1_w, qf32(&S->pw1_w), 2 * d * d); OPT(pw1_b, S->pw1_b, 2 * d);
        REQ(dw_w, S->dw_w, d * K); OPT(dw_b, S->dw_b, d);
        if (e->bn) { REQ(cn_scale, S->cnorm_scale, d); REQ(cn_shift, S->cnorm_shift, d); }
        else { REQ(cn_w, S->cnorm_w, d); REQ(cn_b, S->cnorm_b, d); }
        REQ(pw2_w, qf32(&S->pw2_w), d * d); OPT(pw2_b, S->pw2_b, d);
        REQ(ln_ff2_w, S->ln_ff2_w, d); REQ(ln_ff2_b, S->ln_ff2_b, d);
        REQ(ff2_w1, qf32(&S->ff2_w1), ffn * d); OPT(ff2_b1, S->ff2_b1, ffn);
        REQ(ff2_w2, qf32(&S->ff2_w2), d * ffn); OPT(ff2_b2, S->ff2_b2, d);
        REQ(ln_out_w, S->ln_out_w, d); REQ(ln_out_b, S->ln_out_b, d);
#undef REQ
#undef OPT
    }
    if (en.encproj_w) {
        e->d_ep_w = up_req(e, en.encproj_w, (size_t)e->d_out * d, "encoder_projector");
        e->d_ep_b = up_req(e, en.encproj_b, (size_t)e->d_out, "encoder_projector.bias");
        if (!e->d_ep_w || !e->d_ep_b) return -1;
    }
    /* the rel-pos table: the library builds and CHECKS it (row identity
     * against a direct per-T projection); when it refuses (MYNAH_ASR_RELPOS_TABLE=0,
     * or a provider that fails the check) the device builds it from the same pe */
    const size_t P = (size_t)(2 * e->t_max - 1);
    if (dalloc(e, (void **)&e->d_rk, (size_t)e->L * P * d * sizeof(float), &e->vram_weights, "relpos table") != 0) return -1;
    if (mynah_asr_enc_relpos_table_init(&e->enc, e->t_max) == 0 && e->enc.relpos_tab && e->enc.relpos_kmax == e->t_max) {
        CK(e, "relpos table", cudaMemcpy(e->d_rk, e->enc.relpos_tab, (size_t)e->L * P * d * sizeof(float),
                                         cudaMemcpyHostToDevice));
        free(e->enc.relpos_tab);           /* resident on the device now */
        e->enc.relpos_tab = nullptr;
        e->enc.relpos_kmax = 0;
    } else {
        float *pe = (float *)malloc(P * d * sizeof(float));
        if (!pe) { snprintf(e->err, sizeof(e->err), "out of host memory (pe)"); return -1; }
        mynah_asr_pos_emb(&e->enc, e->t_max, pe);
        float *dpe = nullptr, *drel = nullptr;
        if (dalloc(e, (void **)&dpe, P * d * sizeof(float), nullptr, "pe") != 0 ||
            dalloc(e, (void **)&drel, d * d * sizeof(float), nullptr, "relk") != 0) { free(pe); return -1; }
        CK(e, "pe", cudaMemcpy(dpe, pe, P * d * sizeof(float), cudaMemcpyHostToDevice));
        free(pe);
        for (int li = 0; li < e->L; li++) {
            CK(e, "relk", cudaMemcpy(drel, en.layers[li].relk_w, d * d * sizeof(float), cudaMemcpyHostToDevice));
            CK(e, "relpos gemm", k_gemm_wt(dpe, (int)d, drel, nullptr, e->d_rk + (size_t)li * P * d, (int)d,
                                           (int)P, (int)d, (int)d, 0, 0, e->stream));
            CK(e, "relpos sync", cudaStreamSynchronize(e->stream));
        }
        cudaFree(dpe);
        cudaFree(drel);
    }
    return 0;
}

static int alloc_encoder_scratch(aed_gpu *e) {
    const size_t R = (size_t)e->r_max, d = (size_t)e->d;
    const size_t wide = (size_t)(e->ffn > 2 * e->d ? e->ffn : 2 * e->d);
    float **bufs[] = {&e->xs, &e->xn, &e->tmp, &e->qs, &e->ks, &e->vs, &e->ctx};
    for (float **b : bufs)
        if (dalloc(e, (void **)b, R * d * sizeof(float), &e->vram_scratch, "encoder scratch") != 0) return -1;
    if (dalloc(e, (void **)&e->tmp2, R * wide * sizeof(float), &e->vram_scratch, "encoder scratch") != 0) return -1;
    if (e->d_ep_w) {
        if (dalloc(e, (void **)&e->enc_out, R * (size_t)e->d_out * sizeof(float), &e->vram_scratch, "encoder out") != 0) return -1;
    } else {
        e->enc_out = e->xs;
    }
    const size_t ints = (size_t)e->r_max + 2u * (size_t)e->max_items;
    if (dalloc(e, (void **)&e->d_ints, ints * sizeof(int), &e->vram_scratch, "row tables") != 0) return -1;
    CK(e, "pinned", cudaMallocHost((void **)&e->h_ints, ints * sizeof(int)));
    CK(e, "pinned", cudaMallocHost((void **)&e->h_x, R * d * sizeof(float)));
    CK(e, "pinned", cudaMallocHost((void **)&e->h_out, R * (size_t)e->d_out * sizeof(float)));
    return 0;
}

static int prepare_tc(aed_gpu *e) {
    if (!e->tc) return 0;
    const int R = e->r_max, d = e->d, ffn = e->ffn;
    for (const enc_lw &W : e->lw) {
        if (tc_prepare(e, W.ff1_w1, ffn, d, R) || tc_prepare(e, W.ff1_w2, d, ffn, R) ||
            tc_prepare(e, W.q_w, d, d, R) || tc_prepare(e, W.k_w, d, d, R) || tc_prepare(e, W.v_w, d, d, R) ||
            tc_prepare(e, W.o_w, d, d, R) || tc_prepare(e, W.pw1_w, 2 * d, d, R) || tc_prepare(e, W.pw2_w, d, d, R) ||
            tc_prepare(e, W.ff2_w1, ffn, d, R) || tc_prepare(e, W.ff2_w2, d, ffn, R))
            return -1;
    }
    if (tc_prepare(e, e->d_ep_w, e->d_out, d, R)) return -1;
    return 0;
}

static void aed_close(aed_gpu *e);

extern "C" asr_aed_gpu *asr_aed_gpu_open(const asr_aed_gpu_cfg *cfg, char *err, size_t errcap) {
    aed_gpu *e = new aed_gpu();
    memset(&e->enc, 0, sizeof(e->enc));
    e->cfg = *cfg;
    e->max_items = cfg->max_items > 0 ? cfg->max_items : 1;
    const char *prec = cfg->precision ? cfg->precision : "f32";
    const char *gm = cfg->gemm ? cfg->gemm : "own";
    if (strcmp(prec, "f32") == 0 && strcmp(gm, "own") == 0) e->tc = 0;
    else if (strcmp(prec, "bf16") == 0 && strcmp(gm, "own-tc") == 0) e->tc = 1;
    else {
        snprintf(err, errcap, "the AED engine serves --precision f32 --gemm own (the reference) and "
                              "--precision bf16 --gemm own-tc (the A/B arm); got %s/%s", prec, gm);
        delete e;
        return nullptr;
    }
    if (cfg->decode_on_gpu) {
        snprintf(err, errcap, "the GPU AED decoder is not built yet: run with --aed-decoder host "
                              "(the GPU encoder, the library's CPU decoder)");
        delete e;
        return nullptr;
    }
    int ndev = 0;
    cudaError_t c = cudaGetDeviceCount(&ndev);
    if (c != cudaSuccess || ndev <= 0) {
        snprintf(err, errcap, "no CUDA device: %s", c == cudaSuccess ? "count is 0" : cudaGetErrorString(c));
        delete e;
        return nullptr;
    }
    if (cfg->device < 0 || cfg->device >= ndev) {
        snprintf(err, errcap, "device %d does not exist (%d device(s))", cfg->device, ndev);
        delete e;
        return nullptr;
    }
    e->device = cfg->device;
    if ((c = cudaSetDevice(e->device)) != cudaSuccess || (c = cudaStreamCreate(&e->stream)) != cudaSuccess) {
        snprintf(err, errcap, "cudaSetDevice/StreamCreate: %s", cudaGetErrorString(c));
        delete e;
        return nullptr;
    }
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    if (cudaGetDeviceProperties(&prop, e->device) == cudaSuccess) e->devname = prop.name;
    if (e->tc && prop.major < 8) {
        snprintf(err, errcap, "--gemm own-tc needs bf16 tensor cores (sm_80+); %s is sm_%d%d", prop.name, prop.major, prop.minor);
        aed_close(e);
        return nullptr;
    }
#define FAIL(...) do { snprintf(err, errcap, __VA_ARGS__); aed_close(e); return nullptr; } while (0)
    e->jcfg = load_json(cfg->model_dir, "mynah.json");
    if (!e->jcfg) FAIL("%s/mynah.json is missing or not JSON", cfg->model_dir);
    const char *wfile = jstr(e->jcfg, "weights");
    if (!wfile) FAIL("mynah.json has no \"weights\"");
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", cfg->model_dir, wfile);
    e->st = mynah_asr_st_open_quiet(path);
    if (!e->st) FAIL("the f32 weights %s are missing: the GPU uploads f32", path);
    if (mynah_asr_encoder_init(&e->enc, e->st, 0) != 0) FAIL("encoder init failed (not a FastConformer pack?)");
    e->enc_ok = 1;
    /* causality and xscaling from the config, exactly as mynah_asr_load_quant */
    const cJSON *jenc = cJSON_GetObjectItem(e->jcfg, "encoder");
    const char *jsub = jstr(jenc, "subsampling");
    if (jsub) { e->enc.causal = strstr(jsub, "causal") != nullptr; e->enc.ss.causal = e->enc.causal; }
    const cJSON *jxs = jenc ? cJSON_GetObjectItem(jenc, "xscaling") : nullptr;
    if (jxs && cJSON_IsTrue(jxs)) e->enc.xscale = sqrtf((float)e->enc.d_model);
    if (e->enc.prompt_l1_w) FAIL("the pack has a language-prompt projector: not an AED encoder this engine serves");

    e->L = e->enc.n_layers; e->d = e->enc.d_model; e->H = e->enc.n_heads; e->dk = e->enc.d_head;
    e->ffn = e->enc.ffn_dim; e->K = e->enc.conv_k; e->d_out = e->enc.d_out;
    e->pc = e->enc.causal ? e->K - 1 : (e->K - 1) / 2;
    e->bn = e->enc.layers[0].cnorm_scale != nullptr;
    if (e->H * e->dk != e->d) FAIL("encoder heads %d x %d != d_model %d", e->H, e->dk, e->d);

    /* T_max: the planner's largest segment (1.1 x the limit, src/mynah_asr.c
     * plan_segments) through the three stride-2 stages, rounded up */
    const cJSON *jf = cJSON_GetObjectItem(e->jcfg, "features");
    const int sr = jint(jf, "sample_rate", 16000), hop = jint(jf, "hop_length", 160);
    const double seg = cfg->seg_sec > 0.0 ? cfg->seg_sec : 30.0;
    int T = (int)(1.1 * seg * (double)sr / (double)hop) + 4;
    for (int s = 0; s < 3; s++) T = T / 2 + 1;
    e->t_max = T + 2;
    e->r_max = e->max_items * e->t_max;
    if ((size_t)(2 * e->dk + AED_THREADS + e->t_max) * sizeof(float) > 48u * 1024u)
        FAIL("T_max %d needs more shared memory than a block has (segment limit too long)", e->t_max);

    size_t freeb = 0;
    cudaMemGetInfo(&freeb, &e->vram_total);
    if (upload_encoder(e) != 0 || alloc_encoder_scratch(e) != 0 || prepare_tc(e) != 0)
        FAIL("%s", e->err[0] ? e->err : "device upload/allocation failed");
    if (e->tc && e->tc_ws_floats > 0 &&
        dalloc(e, (void **)&e->tc_ws, e->tc_ws_floats * sizeof(float), &e->vram_scratch, "own-tc workspace") != 0)
        FAIL("%s", e->err);
    c = cudaStreamSynchronize(e->stream);
    if (c != cudaSuccess) FAIL("open: %s", cudaGetErrorString(c));
#undef FAIL
    return e;
}

static void aed_close(aed_gpu *e) {
    if (!e) return;
    if (e->stream) cudaStreamDestroy(e->stream);
    if (e->h_ints) cudaFreeHost(e->h_ints);
    if (e->h_x) cudaFreeHost(e->h_x);
    if (e->h_out) cudaFreeHost(e->h_out);
    /* device buffers go with the context (the process closes once, on exit;
     * cudaDeviceReset does the bulk free, as the streaming engine) */
    if (e->enc_ok) mynah_asr_encoder_free(&e->enc);
    if (e->st) mynah_asr_st_close(e->st);
    if (e->jcfg) cJSON_Delete(e->jcfg);
    cudaDeviceReset();
    delete e;
}

extern "C" void asr_aed_gpu_close(asr_aed_gpu *e) {
    if (!e) return;
    cudaSetDevice(e->device);
    aed_close(e);
}

extern "C" void asr_aed_gpu_offload(asr_aed_gpu *e, mynah_asr_offload *out) {
    memset(out, 0, sizeof(*out));
    out->encode = hook_encode;
    out->aed_decode = nullptr;     /* the library's CPU decoder (stage 2) */
    out->ud = e;
}

extern "C" void asr_aed_gpu_get_facts(const asr_aed_gpu *e, asr_aed_gpu_facts *f) {
    memset(f, 0, sizeof(*f));
    f->device = e->devname.c_str();
    f->precision = e->tc ? "bf16" : "f32";
    f->gemm = e->tc ? "own-tc" : "own-rowstable";
    f->decoder = "host";
    f->max_items = e->max_items;
    f->t_max = e->t_max;
    f->enc_layers = e->L;
    f->d_model = e->d;
    f->vram_total = e->vram_total;
    f->vram_weights = e->vram_weights;
    f->vram_scratch = e->vram_scratch;
    size_t freeb = 0, totb = 0;
    if (cudaMemGetInfo(&freeb, &totb) == cudaSuccess) f->vram_used = totb - freeb;
}

extern "C" void asr_aed_gpu_get_stats(const asr_aed_gpu *e, asr_aed_gpu_stats *s) { *s = e->st_; }
extern "C" int asr_aed_gpu_dead(const asr_aed_gpu *e) { return e->dead; }
extern "C" const char *asr_aed_gpu_error(const asr_aed_gpu *e) { return e->err; }

extern "C" size_t asr_aed_gpu_dispatch_map(const asr_aed_gpu *e, char *buf, size_t cap) {
    const char *g = e->tc ? "own-tc bf16 (k_gemm_wt_tc, splits from the shape)" : "own f32 row-stable (k_gemm_wt)";
    const int n = snprintf(buf, cap,
        "engine        cuda aed-offline  device=%s max_items=%d t_max=%d\n"
        "host          wav, segmentation, mel, prompt, detok      library (src/mynah_asr.c)\n"
        "subsampling   dw_striding 8x                              library on the host (src/subsampling.c)\n"
        "enc linears   ffn, q/k/v/o, pointwise conv                %s\n"
        "enc attention full context, rel-pos table                 enc_attn_kernel (one block per row x head)\n"
        "enc conv      glu + 'same' depthwise k%d + %s + silu     glu_kernel, dwconv_kernel\n"
        "enc norms     layer_norm (mean/var in double)             k_layernorm\n"
        "decoder       aed greedy                                  library on the host (src/decoder_aed.c)\n",
        e->devname.c_str(), e->max_items, e->t_max, g, e->K, e->bn ? "folded batch_norm" : "layer_norm");
    return n < 0 ? 0 : ((size_t)n < cap ? (size_t)n : cap - 1);
}
