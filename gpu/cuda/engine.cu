/* gpu/cuda/engine.cu — the resident CUDA engine behind gpu/engine.h.
 *
 * One GPU, one stream, one thread. Weights are uploaded once; every slot's
 * state (K/V ring, conv cache, subsampling caches, predictor h/c/g, decode
 * metadata) lives in a device arena allocated at open; a step stacks the
 * cohort's chunks into one [R, d] activation, runs the 24 layers with the
 * per-row GEMMs over R and the per-stream kernels over the lanes, then decodes
 * every lane with the label loop. Host <-> device traffic per pass: the packed
 * mel and the row descriptors up, the tokens and the slot metadata down.
 *
 * Nothing is allocated after open. A cohort larger than the scratch budget is
 * split into passes, never refused and never grown. A device error marks the
 * engine dead and every later call fails visibly (ENGINEERING.md §6). */
extern "C" {
#include "../asr_engine.h"
#include "../pack.h"
}
#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>
#include <vector>

/* ------------------------------------------------------------------ types */

struct slot_host {
    mynah_asr_mel_stream mel = {};
    int mel_inited = 0;
    float *mel_buf = nullptr;   /* [mel_cap, n_mels] */
    int mel_cap = 0, mel_have = 0;
    int first = 1, lookahead = 0, prompt = 0, in_use = 0, finished = 0, mel_finished = 0;
    unsigned long samples_fed = 0;
    mynah_asr_detok detok = {};
    int detok_inited = 0;
    size_t chars_emitted = 0;
    double emitted_t1 = 0.0;
    char lang[16] = {0};
    std::vector<int> tokens;
    int reset_pending = 0;
};

struct cuda_engine {
    asr_engine base;
    asr_engine_cfg cfg;
    asr_pack pack;
    int cap;
    gpu_model_dims dm;
    std::vector<gpu_layer_w> L;
    /* device weights */
    float *d_relpos = nullptr;
    float *d_ss_in_w = nullptr, *d_ss_in_b = nullptr, *d_ss_dw_w[2] = {nullptr, nullptr},
          *d_ss_dw_b[2] = {nullptr, nullptr}, *d_ss_pw_w[2] = {nullptr, nullptr},
          *d_ss_pw_b[2] = {nullptr, nullptr}, *d_ss_lin_w = nullptr, *d_ss_lin_b = nullptr;
    float *d_pl1_w = nullptr, *d_pl1_b = nullptr, *d_pl2_w = nullptr, *d_pl2_b = nullptr,
          *d_ep_w = nullptr, *d_ep_b = nullptr;
    float *d_emb = nullptr, *d_wih[MYNAH_ASR_MAX_PRED_LAYERS] = {}, *d_whh[MYNAH_ASR_MAX_PRED_LAYERS] = {},
          *d_bsum[MYNAH_ASR_MAX_PRED_LAYERS] = {}, *d_proj_w = nullptr, *d_proj_b = nullptr,
          *d_head_w = nullptr, *d_head_b = nullptr;
    float *d_sos_h = nullptr, *d_sos_c = nullptr, *d_sos_g = nullptr;
    gpu_arena ar = {};
    /* scratch budgets and buffers */
    int Bmax = 0, Rmax = 0, Mmax = 0, Pmax[GPU_SS_STAGES] = {0, 0, 0};
    float *xs = nullptr, *tmp = nullptr, *tmp2 = nullptr, *xn = nullptr, *kn = nullptr,
          *vn = nullptr, *qs = nullptr, *ctx = nullptr, *h2 = nullptr, *cmid = nullptr,
          *cat = nullptr, *mid = nullptr, *fused = nullptr, *enc = nullptr;
    float *s0 = nullptr, *s1a = nullptr, *s1 = nullptr, *s2a = nullptr, *s2 = nullptr,
          *flat = nullptr, *d_mel = nullptr;
    gpu_row *d_rows = nullptr, *h_rows = nullptr;
    float *h_mel = nullptr;
    float *jin = nullptr, *logits = nullptr, *z = nullptr, *xin = nullptr, *xnext = nullptr,
          *hrows = nullptr, *gtmp = nullptr;
    int *d_active = nullptr, *d_n_active = nullptr, *d_am = nullptr, *d_emit = nullptr,
        *d_resets = nullptr;
    int *h_n_active = nullptr, *h_resets = nullptr, *h_tok = nullptr;
    gpu_slot_meta *h_meta = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    int use_cublas = 0;
    std::vector<slot_host> slots;
    std::vector<int> pending_resets;
    asr_engine_stats st = {};
    size_t vram_weights = 0, vram_arena = 0, vram_total = 0;
    char err[512] = {0};
    int dead = 0;
    std::string devname;
    /* S14-6b: CUDA events at stage boundaries, one pass at a time */
    int prof = 0;
    static const int EV_MAX = 4096;
    cudaEvent_t ev[4096];
    int ev_tag[4096];
    int nev = 0;
};

enum { PS_H2D = 0, PS_SS, PS_FFN1, PS_ATT_PROJ, PS_ATT_CORE, PS_ATT_OUT, PS_CONV, PS_FFN2,
       PS_POST, PS_DEC_JOINT, PS_DEC_PRED, PS_DEC_SYNC, PS_D2H, PS_OTHER };

/* mark: from here on the stream's time is charged to `tag` */
static inline void prof_mark(cuda_engine *e, int tag) {
    if (!e->prof || e->nev >= cuda_engine::EV_MAX) return;
    cudaEventRecord(e->ev[e->nev], e->stream);
    e->ev_tag[e->nev++] = tag;
}
/* after the pass's final sync: each interval charged to the tag that opened it */
static void prof_collect(cuda_engine *e) {
    if (!e->prof) return;
    for (int i = 0; i + 1 < e->nev; i++) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, e->ev[i], e->ev[i + 1]) == cudaSuccess)
            e->st.prof_ms[e->ev_tag[i]] += ms;
    }
    e->st.prof_passes++;
    e->nev = 0;
}

/* ---------------------------------------------------------------- helpers */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void set_err(cuda_engine *e, const char *what, cudaError_t c) {
    snprintf(e->err, sizeof(e->err), "%s: %s", what, cudaGetErrorString(c));
    e->dead = 1;
    e->st.errors++;
    fprintf(stderr, "mynah-asr-server-cuda: DEVICE ERROR %s\n", e->err);
}

#define CK(e, what, call) do { cudaError_t c_ = (call); if (c_ != cudaSuccess) { set_err((e), (what), c_); return -1; } } while (0)

static int dmalloc(cuda_engine *e, void **p, size_t bytes, size_t *acct, const char *what) {
    cudaError_t c = cudaMalloc(p, bytes);
    if (c != cudaSuccess) { set_err(e, what, c); return -1; }
    if (acct) *acct += bytes;
    return 0;
}

static float *upload(cuda_engine *e, const float *host, size_t n, const char *what) {
    if (!host) { snprintf(e->err, sizeof(e->err), "%s: missing tensor", what); return nullptr; }
    void *d = nullptr;
    if (dmalloc(e, &d, n * sizeof(float), &e->vram_weights, what) != 0) return nullptr;
    cudaError_t c = cudaMemcpy(d, host, n * sizeof(float), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) { set_err(e, what, c); return nullptr; }
    return (float *)d;
}

static const float *qf32(const mynah_asr_qmat *m) {
    return m->qtype == MYNAH_ASR_Q_F32 ? m->f32 : nullptr;
}

/* the three stride-2 stages on the time axis, as src/subsampling.c computes them */
static void ss_geometry(int n_mel, int first, int last, int to[GPU_SS_STAGES]) {
    int T = n_mel;
    for (int s = 0; s < GPU_SS_STAGES; s++) {
        const int lp = first ? 2 : 1, rp = last ? 1 : 0;
        const int Tp = T + lp + rp;
        const int To = Tp >= 3 ? (Tp - 3) / 2 + 1 : 0;
        to[s] = To;
        T = To;
    }
}

/* ------------------------------------------------------------------ GEMM
 * The own kernel by default (row-stable by construction); cuBLAS as the
 * measured comparison arm (--gemm cublas), pedantic f32, no TF32. */
__global__ void bias_act_kernel(float *__restrict__ C, int ldc, const float *__restrict__ bias,
                                int M, int N, int act) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= N || m >= M) return;
    float v = C[(size_t)m * ldc + n];
    if (bias) v += bias[n];
    if (act == 1) v = v > 0.0f ? v : 0.0f;
    else if (act == 2) {
        const float sg = v >= 0.0f ? 1.0f / (1.0f + expf(-v)) : expf(v) / (1.0f + expf(v));
        v = v * sg;
    }
    C[(size_t)m * ldc + n] = v;
}

static int gemm(cuda_engine *e, const float *A, int lda, const float *W, const float *bias,
                float *C, int ldc, int M, int N, int K, int accumulate, int act) {
    if (M <= 0) return 0;
    if (!e->use_cublas) {
        CK(e, "gemm", k_gemm_wt(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, e->stream));
        return 0;
    }
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    cublasStatus_t st = cublasSgemm(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K,
                                    A, lda, &beta, C, ldc);
    if (st != CUBLAS_STATUS_SUCCESS) {
        snprintf(e->err, sizeof(e->err), "cublasSgemm failed (%d)", (int)st);
        e->dead = 1; e->st.errors++;
        return -1;
    }
    if (bias || act) {
        dim3 grid((unsigned)((N + 255) / 256), (unsigned)M);
        bias_act_kernel<<<grid, 256, 0, e->stream>>>(C, ldc, bias, M, N, act);
        CK(e, "gemm epilogue", cudaGetLastError());
    }
    return 0;
}

/* ------------------------------------------------------------------- open */

static int slot_host_init(cuda_engine *e, slot_host *s) {
    if (mynah_asr_mel_stream_init(&s->mel, &e->pack.feat) != 0) return -1;
    s->mel_inited = 1;
    /* a chunk of the largest preset, plus one more chunk of slack */
    s->mel_cap = 2 * asr_pack_chunk_mel(&e->pack, e->pack.qmax - 1, 1) + 8;
    s->mel_buf = (float *)calloc((size_t)s->mel_cap * e->pack.feat.n_mels, sizeof(float));
    if (!s->mel_buf) return -1;
    if (mynah_asr_detok_init(&s->detok, 4096) != 0) return -1;
    s->detok_inited = 1;
    s->lookahead = e->pack.default_right;
    s->prompt = e->pack.default_prompt;
    s->first = 1;
    return 0;
}

static void slot_host_free(slot_host *s) {
    if (s->mel_inited) mynah_asr_mel_stream_free(&s->mel);
    if (s->detok_inited) mynah_asr_detok_free(&s->detok);
    free(s->mel_buf);
    s->tokens.clear();
    s->tokens.shrink_to_fit();
    s->mel_inited = s->detok_inited = 0;
    s->mel_buf = nullptr;
}

static int alloc_scratch(cuda_engine *e) {
    const gpu_model_dims &dm = e->dm;
    const int qmax = e->pack.qmax;
    /* budget: a pass of up to Bmax lanes at the LARGEST preset; a cohort that
     * needs more is split into passes */
    e->Bmax = e->cap < 128 ? e->cap : 128;
    e->Rmax = e->Bmax * qmax;
    const int melmax = asr_pack_chunk_mel(&e->pack, qmax - 1, 1);
    e->Mmax = e->Bmax * melmax;
    int to[GPU_SS_STAGES];
    ss_geometry(melmax, 1, 1, to);
    for (int s = 0; s < GPU_SS_STAGES; s++) e->Pmax[s] = e->Bmax * to[s] * dm.Fo[s];

    size_t *acct = &e->vram_arena;
    const size_t R = (size_t)e->Rmax, d = (size_t)dm.d;
#define DM(ptr, floats, what) if (dmalloc(e, (void **)&(ptr), (floats) * sizeof(float), acct, what) != 0) return -1
    DM(e->xs, R * d, "xs"); DM(e->tmp, R * d, "tmp"); DM(e->tmp2, R * (size_t)dm.ffn, "tmp2");
    DM(e->xn, R * d, "xn"); DM(e->kn, R * d, "kn"); DM(e->vn, R * d, "vn"); DM(e->qs, R * d, "qs");
    DM(e->ctx, R * d, "ctx"); DM(e->h2, R * 2 * d, "h2"); DM(e->cmid, R * d, "cmid");
    DM(e->cat, R * (size_t)(dm.d + dm.np), "cat"); DM(e->mid, R * (size_t)dm.inter, "mid");
    DM(e->fused, R * d, "fused"); DM(e->enc, R * (size_t)dm.dout, "enc");
    DM(e->s0, (size_t)e->Pmax[0] * dm.C, "s0"); DM(e->s1a, (size_t)e->Pmax[1] * dm.C, "s1a");
    DM(e->s1, (size_t)e->Pmax[1] * dm.C, "s1"); DM(e->s2a, (size_t)e->Pmax[2] * dm.C, "s2a");
    DM(e->s2, (size_t)e->Pmax[2] * dm.C, "s2");
    DM(e->flat, R * (size_t)dm.C * dm.Fo[GPU_SS_STAGES - 1], "flat");
    DM(e->d_mel, (size_t)e->Mmax * dm.n_mels, "mel");
    const size_t B = (size_t)e->Bmax, H = (size_t)dm.Hdec;
    DM(e->jin, B * H, "jin"); DM(e->logits, B * (size_t)dm.V, "logits"); DM(e->z, B * 4 * H, "z");
    DM(e->xin, B * H, "xin"); DM(e->xnext, B * H, "xnext"); DM(e->hrows, B * H, "hrows");
    DM(e->gtmp, B * H, "gtmp");
#undef DM
    if (dmalloc(e, (void **)&e->d_rows, B * sizeof(gpu_row), acct, "rows") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_active, B * sizeof(int), acct, "active") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_am, B * sizeof(int), acct, "am") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_emit, B * sizeof(int), acct, "emit") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_n_active, sizeof(int), acct, "n_active") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_resets, (size_t)e->cap * sizeof(int), acct, "resets") != 0) return -1;
    CK(e, "pinned rows", cudaHostAlloc((void **)&e->h_rows, B * sizeof(gpu_row), cudaHostAllocPortable));
    CK(e, "pinned mel", cudaHostAlloc((void **)&e->h_mel, (size_t)e->Mmax * dm.n_mels * sizeof(float), cudaHostAllocPortable));
    CK(e, "pinned n_active", cudaHostAlloc((void **)&e->h_n_active, sizeof(int), cudaHostAllocPortable));
    CK(e, "pinned resets", cudaHostAlloc((void **)&e->h_resets, (size_t)e->cap * sizeof(int), cudaHostAllocPortable));
    CK(e, "pinned tok", cudaHostAlloc((void **)&e->h_tok, (size_t)e->cap * e->ar.tok_cap * sizeof(int), cudaHostAllocPortable));
    CK(e, "pinned meta", cudaHostAlloc((void **)&e->h_meta, (size_t)e->cap * sizeof(gpu_slot_meta), cudaHostAllocPortable));
    return 0;
}

static int alloc_arena(cuda_engine *e) {
    const gpu_model_dims &dm = e->dm;
    const size_t cap = (size_t)e->cap;
    size_t *acct = &e->vram_arena;
    if (dmalloc(e, (void **)&e->ar.kv, cap * dm.n_layers * 2 * dm.left * dm.d * sizeof(float), acct, "kv arena") != 0) return -1;
    if (dmalloc(e, (void **)&e->ar.conv_cache, cap * dm.n_layers * (dm.conv_k - 1) * dm.d * sizeof(float), acct, "conv cache") != 0) return -1;
    for (int s = 0; s < GPU_SS_STAGES; s++) {
        const size_t n = (size_t)(s == 0 ? 1 : dm.C) * dm.F[s];
        if (dmalloc(e, (void **)&e->ar.ss_cache[s], cap * n * sizeof(float), acct, "subsampling cache") != 0) return -1;
    }
    if (dmalloc(e, (void **)&e->ar.dec_h, cap * dm.pred_layers * dm.Hdec * sizeof(float), acct, "dec h") != 0) return -1;
    if (dmalloc(e, (void **)&e->ar.dec_c, cap * dm.pred_layers * dm.Hdec * sizeof(float), acct, "dec c") != 0) return -1;
    if (dmalloc(e, (void **)&e->ar.dec_g, cap * dm.Hdec * sizeof(float), acct, "dec g") != 0) return -1;
    if (dmalloc(e, (void **)&e->ar.meta, cap * sizeof(gpu_slot_meta), acct, "meta") != 0) return -1;
    e->ar.tok_cap = e->pack.qmax * dm.max_symbols;
    if (dmalloc(e, (void **)&e->ar.tok, cap * e->ar.tok_cap * sizeof(int), acct, "tok") != 0) return -1;
    if (dmalloc(e, (void **)&e->ar.tok_frame, cap * e->ar.tok_cap * sizeof(int), acct, "tok frame") != 0) return -1;
    CK(e, "arena memset", cudaMemset(e->ar.kv, 0, cap * dm.n_layers * 2 * dm.left * dm.d * sizeof(float)));
    CK(e, "arena memset", cudaMemset(e->ar.meta, 0, cap * sizeof(gpu_slot_meta)));
    return 0;
}

static int upload_weights(cuda_engine *e) {
    const mynah_asr_encoder &enc = e->pack.enc;
    const mynah_asr_decoder &dec = e->pack.dec;
    const gpu_model_dims &dm = e->dm;
    const size_t d = (size_t)dm.d;
    e->L.resize((size_t)dm.n_layers);
    for (int li = 0; li < dm.n_layers; li++) {
        const mynah_asr_enc_layer &s = enc.layers[li];
        gpu_layer_w &L = e->L[(size_t)li];
#define UP(dst, src, n, what) do { (dst) = upload(e, (src), (n), what); if (!(dst)) return -1; } while (0)
        UP(L.ln_ff1_w, s.ln_ff1_w, d, "ln_ff1_w"); UP(L.ln_ff1_b, s.ln_ff1_b, d, "ln_ff1_b");
        UP(L.ff1_w1, qf32(&s.ff1_w1), (size_t)dm.ffn * d, "ff1_w1");
        UP(L.ff1_w2, qf32(&s.ff1_w2), d * (size_t)dm.ffn, "ff1_w2");
        UP(L.ln_att_w, s.ln_att_w, d, "ln_att_w"); UP(L.ln_att_b, s.ln_att_b, d, "ln_att_b");
        UP(L.q_w, qf32(&s.q_w), d * d, "q_w"); UP(L.k_w, qf32(&s.k_w), d * d, "k_w");
        UP(L.v_w, qf32(&s.v_w), d * d, "v_w"); UP(L.o_w, qf32(&s.o_w), d * d, "o_w");
        UP(L.bias_u, s.bias_u, d, "bias_u"); UP(L.bias_v, s.bias_v, d, "bias_v");
        UP(L.ln_conv_w, s.ln_conv_w, d, "ln_conv_w"); UP(L.ln_conv_b, s.ln_conv_b, d, "ln_conv_b");
        UP(L.pw1_w, qf32(&s.pw1_w), 2 * d * d, "pw1_w");
        UP(L.dw_w, s.dw_w, d * (size_t)dm.conv_k, "dw_w");
        UP(L.cnorm_w, s.cnorm_w, d, "cnorm_w"); UP(L.cnorm_b, s.cnorm_b, d, "cnorm_b");
        UP(L.pw2_w, qf32(&s.pw2_w), d * d, "pw2_w");
        UP(L.ln_ff2_w, s.ln_ff2_w, d, "ln_ff2_w"); UP(L.ln_ff2_b, s.ln_ff2_b, d, "ln_ff2_b");
        UP(L.ff2_w1, qf32(&s.ff2_w1), (size_t)dm.ffn * d, "ff2_w1");
        UP(L.ff2_w2, qf32(&s.ff2_w2), d * (size_t)dm.ffn, "ff2_w2");
        UP(L.ln_out_w, s.ln_out_w, d, "ln_out_w"); UP(L.ln_out_b, s.ln_out_b, d, "ln_out_b");
    }
    UP(e->d_relpos, enc.relpos_tab, (size_t)dm.n_layers * (size_t)(2 * dm.kmax - 1) * d, "relpos table");
    const mynah_asr_subsampling &ss = enc.ss;
    const size_t C = (size_t)dm.C;
    UP(e->d_ss_in_w, (const float *)ss.conv_in_w->data, C * 9, "ss conv_in_w");
    UP(e->d_ss_in_b, (const float *)ss.conv_in_b->data, C, "ss conv_in_b");
    for (int i = 0; i < 2; i++) {
        UP(e->d_ss_dw_w[i], (const float *)ss.dw_w[i]->data, C * 9, "ss dw_w");
        UP(e->d_ss_dw_b[i], (const float *)ss.dw_b[i]->data, C, "ss dw_b");
        UP(e->d_ss_pw_w[i], (const float *)ss.pw_w[i]->data, C * C, "ss pw_w");
        UP(e->d_ss_pw_b[i], (const float *)ss.pw_b[i]->data, C, "ss pw_b");
    }
    UP(e->d_ss_lin_w, (const float *)ss.lin_w->data, d * C * (size_t)dm.Fo[GPU_SS_STAGES - 1], "ss lin_w");
    UP(e->d_ss_lin_b, (const float *)ss.lin_b->data, d, "ss lin_b");
    UP(e->d_pl1_w, enc.prompt_l1_w, (size_t)dm.inter * (size_t)(dm.d + dm.np), "prompt_l1_w");
    UP(e->d_pl1_b, enc.prompt_l1_b, (size_t)dm.inter, "prompt_l1_b");
    UP(e->d_pl2_w, enc.prompt_l2_w, d * (size_t)dm.inter, "prompt_l2_w");
    UP(e->d_pl2_b, enc.prompt_l2_b, d, "prompt_l2_b");
    UP(e->d_ep_w, enc.encproj_w, (size_t)dm.dout * d, "encproj_w");
    UP(e->d_ep_b, enc.encproj_b, (size_t)dm.dout, "encproj_b");
    const size_t H = (size_t)dm.Hdec;
    UP(e->d_emb, dec.embedding, (size_t)dm.V * H, "embedding");
    for (int l = 0; l < dm.pred_layers; l++) {
        UP(e->d_wih[l], dec.w_ih[l], 4 * H * H, "w_ih");
        UP(e->d_whh[l], dec.w_hh[l], 4 * H * H, "w_hh");
        std::vector<float> bsum(4 * H);
        for (size_t i = 0; i < 4 * H; i++) bsum[i] = dec.b_ih[l][i] + dec.b_hh[l][i];
        UP(e->d_bsum[l], bsum.data(), 4 * H, "b_ih + b_hh");
    }
    UP(e->d_proj_w, dec.proj_w, H * H, "proj_w"); UP(e->d_proj_b, dec.proj_b, H, "proj_b");
    UP(e->d_head_w, qf32(&dec.head), (size_t)dm.V * H, "head");
    UP(e->d_head_b, dec.head_b, (size_t)dm.V, "head_b");
    /* the SOS predictor state: one pred_step(blank) from zeros, computed by the
     * library's own decoder on the host (T = 0 runs exactly that and nothing else) */
    {
        mynah_asr_dec_state *st = (mynah_asr_dec_state *)calloc(1, sizeof(*st));
        if (!st) return -1;
        mynah_asr_dec_state_reset(&dec, st);
        (void)mynah_asr_greedy_decode(&dec, st, nullptr, 0, nullptr, nullptr, 0);
        std::vector<float> h(H * dm.pred_layers), c(H * dm.pred_layers);
        for (int l = 0; l < dm.pred_layers; l++) {
            memcpy(&h[(size_t)l * H], st->h[l], H * sizeof(float));
            memcpy(&c[(size_t)l * H], st->c[l], H * sizeof(float));
        }
        UP(e->d_sos_h, h.data(), H * dm.pred_layers, "sos h");
        UP(e->d_sos_c, c.data(), H * dm.pred_layers, "sos c");
        UP(e->d_sos_g, st->g, H, "sos g");
        free(st);
    }
#undef UP
    return 0;
}

static const asr_engine_ops *cuda_ops(void);
static void cuda_close(cuda_engine *e);

extern "C" asr_engine *asr_engine_open_cuda(const asr_engine_cfg *cfg, char *err, size_t errcap) {
    cuda_engine *e = new cuda_engine();
    e->base.ops = cuda_ops();
    e->cfg = *cfg;
    e->cap = cfg->cap > 0 ? cfg->cap : 1;
    if (cfg->precision && strcmp(cfg->precision, "f32") != 0) {
        snprintf(err, errcap, "precision '%s' is not implemented yet (S14-8); f32 only", cfg->precision);
        delete e; return nullptr;
    }
    e->use_cublas = cfg->gemm && strcmp(cfg->gemm, "cublas") == 0;
    e->prof = cfg->profile ? 1 : 0;
    if (cfg->gemm && !e->use_cublas && strcmp(cfg->gemm, "own") != 0) {
        snprintf(err, errcap, "gemm '%s' is not one of own, cublas", cfg->gemm);
        delete e; return nullptr;
    }
    int ndev = 0;
    cudaError_t c = cudaGetDeviceCount(&ndev);
    if (c != cudaSuccess || ndev <= 0) {
        snprintf(err, errcap, "no CUDA device: %s", c == cudaSuccess ? "count is 0" : cudaGetErrorString(c));
        delete e; return nullptr;
    }
    if (cfg->device < 0 || cfg->device >= ndev) {
        snprintf(err, errcap, "device %d does not exist (%d device(s))", cfg->device, ndev);
        delete e; return nullptr;
    }
    if ((c = cudaSetDevice(cfg->device)) != cudaSuccess ||
        (c = cudaStreamCreate(&e->stream)) != cudaSuccess) {
        snprintf(err, errcap, "cudaSetDevice/StreamCreate: %s", cudaGetErrorString(c));
        delete e; return nullptr;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, cfg->device) == cudaSuccess) e->devname = prop.name;
    if (e->prof)
        for (int i = 0; i < cuda_engine::EV_MAX; i++)
            if (cudaEventCreate(&e->ev[i]) != cudaSuccess) { snprintf(err, errcap, "cudaEventCreate failed"); delete e; return nullptr; }
    if (e->use_cublas) {
        if (cublasCreate(&e->blas) != CUBLAS_STATUS_SUCCESS) {
            snprintf(err, errcap, "cublasCreate failed"); cuda_close(e); return nullptr;
        }
        cublasSetStream(e->blas, e->stream);
        cublasSetMathMode(e->blas, CUBLAS_PEDANTIC_MATH);
    }
    if (asr_pack_open(&e->pack, cfg->model_dir, err, errcap) != 0) { cuda_close(e); return nullptr; }

    const mynah_asr_encoder &enc = e->pack.enc;
    gpu_model_dims &dm = e->dm;
    memset(&dm, 0, sizeof(dm));
    dm.n_layers = enc.n_layers; dm.d = enc.d_model; dm.H = enc.n_heads; dm.dk = enc.d_head;
    dm.ffn = enc.ffn_dim; dm.conv_k = enc.conv_k; dm.left = e->pack.left_ctx; dm.kmax = enc.relpos_kmax;
    dm.n_mels = e->pack.feat.n_mels; dm.C = enc.ss.channels;
    dm.F[0] = dm.n_mels;
    for (int s = 0; s < GPU_SS_STAGES; s++) {
        dm.Fo[s] = (dm.F[s] + 3 - 3) / 2 + 1;
        if (s + 1 < GPU_SS_STAGES) dm.F[s + 1] = dm.Fo[s];
    }
    dm.dout = enc.d_out; dm.np = enc.num_prompts; dm.inter = enc.prompt_inter;
    dm.V = e->pack.dec.vocab; dm.Hdec = e->pack.dec.hidden; dm.pred_layers = e->pack.dec.n_layers;
    dm.blank = e->pack.dec.blank; dm.max_symbols = e->pack.dec.max_symbols;
    if (dm.H * dm.dk != dm.d || dm.Hdec != dm.dout || dm.kmax < dm.left + e->pack.qmax ||
        e->pack.qmax > GPU_QMAX_HARD || dm.left + e->pack.qmax > GPU_KMAX_HARD || !enc.prompt_l1_w) {
        snprintf(err, errcap, "model geometry outside what the kernels serve (H*dk=%d d=%d Hdec=%d dout=%d kmax=%d left=%d qmax=%d prompt=%s)",
                 dm.H * dm.dk, dm.d, dm.Hdec, dm.dout, dm.kmax, dm.left, e->pack.qmax, enc.prompt_l1_w ? "yes" : "no");
        cuda_close(e); return nullptr;
    }
    /* the subsampling linear must match the flatten the kernels produce */
    if ((int)enc.ss.lin_w->shape[1] != dm.C * dm.Fo[GPU_SS_STAGES - 1]) {
        snprintf(err, errcap, "subsampling linear expects %d inputs, the kernels flatten %d",
                 (int)enc.ss.lin_w->shape[1], dm.C * dm.Fo[GPU_SS_STAGES - 1]);
        cuda_close(e); return nullptr;
    }
    size_t freeb = 0, totb = 0;
    cudaMemGetInfo(&freeb, &totb);
    e->vram_total = totb;
    if (upload_weights(e) != 0 || alloc_arena(e) != 0 || alloc_scratch(e) != 0) {
        snprintf(err, errcap, "%s", e->err[0] ? e->err : "device allocation failed");
        cuda_close(e); return nullptr;
    }
    e->slots.resize((size_t)e->cap);
    for (int i = 0; i < e->cap; i++)
        if (slot_host_init(e, &e->slots[(size_t)i]) != 0) {
            snprintf(err, errcap, "host slot state allocation failed"); cuda_close(e); return nullptr;
        }
    /* every slot starts reset */
    for (int i = 0; i < e->cap; i++) e->h_resets[i] = i;
    c = cudaMemcpyAsync(e->d_resets, e->h_resets, (size_t)e->cap * sizeof(int), cudaMemcpyHostToDevice, e->stream);
    if (c == cudaSuccess) c = k_slots_reset(dm, e->ar, e->d_resets, e->cap, e->d_sos_h, e->d_sos_c, e->d_sos_g, e->stream);
    if (c == cudaSuccess) c = cudaStreamSynchronize(e->stream);
    if (c != cudaSuccess) { snprintf(err, errcap, "initial reset: %s", cudaGetErrorString(c)); cuda_close(e); return nullptr; }
    return &e->base;
}

static void cuda_close(cuda_engine *e) {
    if (!e) return;
    for (auto &s : e->slots) slot_host_free(&s);
    if (e->blas) cublasDestroy(e->blas);
    if (e->stream) cudaStreamDestroy(e->stream);
    /* the device allocations go with the context; a server closes once, on
     * its way out, and cudaDeviceReset does the bulk free */
    if (e->h_rows) cudaFreeHost(e->h_rows);
    if (e->h_mel) cudaFreeHost(e->h_mel);
    if (e->h_n_active) cudaFreeHost(e->h_n_active);
    if (e->h_resets) cudaFreeHost(e->h_resets);
    if (e->h_tok) cudaFreeHost(e->h_tok);
    if (e->h_meta) cudaFreeHost(e->h_meta);
    asr_pack_close(&e->pack);
    cudaDeviceReset();
    delete e;
}

/* ----------------------------------------------------------------- facts */

static void cuda_facts(const cuda_engine *e, asr_engine_facts *f) {
    memset(f, 0, sizeof(*f));
    f->name = "cuda";
    f->device = e->devname.c_str();
    f->precision = "f32";
    f->gemm = e->use_cublas ? "cublas-pedantic (measured arm)" : "own-rowstable";
    f->model_name = e->pack.name;
    f->cap = e->cap; f->qmax = e->pack.qmax;
    f->n_lookaheads = e->pack.n_lookaheads;
    for (int i = 0; i < e->pack.n_lookaheads; i++) f->lookaheads[i] = e->pack.lookaheads[i];
    f->default_lookahead = e->pack.default_right;
    f->sample_rate = e->pack.feat.sample_rate; f->n_mels = e->pack.feat.n_mels;
    f->frame_sec = e->pack.frame_sec;
    size_t freeb = 0, totb = 0;
    if (cudaMemGetInfo(&freeb, &totb) == cudaSuccess) { f->vram_total = totb; f->vram_used = totb - freeb; }
    f->vram_arena = e->vram_arena; f->vram_weights = e->vram_weights;
    f->graphs = 0;
}

static void cuda_stats(const cuda_engine *e, asr_engine_stats *s) { *s = e->st; }
static int cuda_lang_id(const cuda_engine *e, const char *lang) { return asr_pack_lang_id(&e->pack, lang); }
static int cuda_lookahead_ok(const cuda_engine *e, int la) { return asr_pack_lookahead_ok(&e->pack, la); }
static int cuda_dead(const cuda_engine *e) { return e->dead; }
static const char *cuda_error(const cuda_engine *e) { return e->err; }

static size_t cuda_dispatch_map(const cuda_engine *e, char *buf, size_t cap) {
    const char *g = e->use_cublas ? "cublas-sgemm-pedantic" : "own-rowstable-f32";
    return (size_t)snprintf(buf, cap,
        "engine            cuda           %s\n"
        "gemm              %-14s row-stable-by-construction=%s\n"
        "subsampling       direct-3x3-s2  position-major, pointwise via gemm\n"
        "layernorm         own            double accumulators, fixed tree\n"
        "attention         own            rel-pos table, ring window, warp softmax\n"
        "conv              own            glu + causal depthwise k=%d, cached\n"
        "decoder           own            label loop, first-index argmax, lstm x%d\n"
        "mel               host           src/features.c (double fft), phase 1\n"
        "precision         f32            weights f32 resident, no tf32\n"
        "graphs            off            phase 1\n",
        e->devname.c_str(), g, e->use_cublas ? "no(measured)" : "yes", e->dm.conv_k, e->dm.pred_layers);
}

/* ----------------------------------------------------------------- slots */

static int cuda_slot_reset(cuda_engine *e, int slot, const char *lang, int lookahead) {
    if (e->dead || slot < 0 || slot >= e->cap) return -1;
    const int prompt_id = asr_pack_lang_id(&e->pack, lang);
    if (!asr_pack_lookahead_ok(&e->pack, lookahead) || prompt_id < 0) return -1;
    slot_host &s = e->slots[(size_t)slot];
    mynah_asr_mel_stream_reset(&s.mel);
    s.mel_have = 0; s.first = 1; s.lookahead = lookahead; s.prompt = prompt_id;
    s.in_use = 1; s.finished = 0; s.mel_finished = 0; s.samples_fed = 0;
    mynah_asr_detok_reset(&s.detok);
    s.chars_emitted = 0; s.emitted_t1 = 0.0; s.lang[0] = '\0';
    s.tokens.clear();
    /* once per slot until the next step applies it: repeated resets of an
     * idle slot must not grow the list past the pinned buffer (cap entries) */
    if (!s.reset_pending) { s.reset_pending = 1; e->pending_resets.push_back(slot); }
    return 0;
}

static int slot_need_mel(const cuda_engine *e, const slot_host &s) {
    return asr_pack_chunk_mel(&e->pack, s.lookahead, s.first);
}

static size_t cuda_slot_need_samples(const cuda_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap) return 0;
    const slot_host &s = e->slots[(size_t)slot];
    const int need = slot_need_mel(e, s);
    if (s.mel_have >= need) return 0;
    const long frame = s.mel.next_frame + (need - s.mel_have) - 1;
    return mynah_asr_mel_stream_samples_until(&s.mel, frame);
}

/* pull every ready mel frame into the slot's buffer */
static void slot_pull_mel(cuda_engine *e, slot_host &s, const float *pcm, size_t n) {
    const int nm = e->pack.feat.n_mels;
    const int room = s.mel_cap - s.mel_have;
    if (room <= 0) { if (n) mynah_asr_mel_stream_feed(&s.mel, pcm, n, nullptr, 0); return; }
    const int got = n ? mynah_asr_mel_stream_feed(&s.mel, pcm, n, s.mel_buf + (size_t)s.mel_have * nm, room)
                      : mynah_asr_mel_stream_feed(&s.mel, nullptr, 0, s.mel_buf + (size_t)s.mel_have * nm, room);
    if (got > 0) s.mel_have += got;
}

static int cuda_slot_feed(cuda_engine *e, int slot, const float *pcm, size_t n) {
    if (e->dead || slot < 0 || slot >= e->cap) return -1;
    slot_host &s = e->slots[(size_t)slot];
    s.samples_fed += (unsigned long)n;
    const double t0 = now_s();
    slot_pull_mel(e, s, pcm, n);
    e->st.host_mel_ms += (now_s() - t0) * 1e3;
    return 0;
}

static int cuda_slot_ready(const cuda_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap) return 0;
    const slot_host &s = e->slots[(size_t)slot];
    return s.mel_have >= slot_need_mel(e, s);
}
static double cuda_slot_audio_s(const cuda_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap) return 0.0;
    return (double)e->slots[(size_t)slot].samples_fed / (double)e->pack.feat.sample_rate;
}
static const char *cuda_slot_text(const cuda_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap) return "";
    const slot_host &s = e->slots[(size_t)slot];
    return s.detok.buf ? s.detok.buf : "";
}
static const char *cuda_slot_lang(const cuda_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap) return "";
    return e->slots[(size_t)slot].lang;
}

/* ------------------------------------------------------------------ step */

struct pass_lane {
    int req;                    /* index into reqs/outs */
    int slot, n_mel, first, last, q;
};

static int run_pass(cuda_engine *e, const std::vector<pass_lane> &lanes, const asr_step_req *reqs,
                    asr_step_out *outs) {
    const gpu_model_dims &dm = e->dm;
    const int B = (int)lanes.size();
    if (B == 0) return 0;
    const int nm = dm.n_mels;
    /* descriptors and the packed mel */
    int R = 0, M = 0, P[GPU_SS_STAGES] = {0, 0, 0};
    for (int a = 0; a < B; a++) {
        const pass_lane &l = lanes[(size_t)a];
        gpu_row &r = e->h_rows[a];
        memset(&r, 0, sizeof(r));
        r.slot = l.slot; r.q = l.q; r.n_mel = l.n_mel; r.mel_off = M; r.first = l.first; r.last = l.last;
        r.prompt = e->slots[(size_t)l.slot].prompt; r.row_off = R;
        int to[GPU_SS_STAGES];
        ss_geometry(l.n_mel, l.first, l.last, to);
        for (int s = 0; s < GPU_SS_STAGES; s++) { r.to[s] = to[s]; r.pos_off[s] = P[s]; P[s] += to[s] * dm.Fo[s]; }
        memcpy(e->h_mel + (size_t)M * nm, e->slots[(size_t)l.slot].mel_buf, (size_t)l.n_mel * nm * sizeof(float));
        M += l.n_mel; R += l.q;
    }
    prof_mark(e, PS_H2D);
    CK(e, "h2d rows", cudaMemcpyAsync(e->d_rows, e->h_rows, (size_t)B * sizeof(gpu_row), cudaMemcpyHostToDevice, e->stream));
    CK(e, "h2d mel", cudaMemcpyAsync(e->d_mel, e->h_mel, (size_t)M * nm * sizeof(float), cudaMemcpyHostToDevice, e->stream));
    e->st.h2d_bytes += (double)B * sizeof(gpu_row) + (double)M * nm * sizeof(float);

    /* subsampling */
    prof_mark(e, PS_SS);
    CK(e, "ss0", k_ss_stage0(dm, e->ar, e->d_rows, B, e->d_mel, e->d_ss_in_w, e->d_ss_in_b, e->s0, e->stream));
    CK(e, "ss1 dw", k_ss_dw(dm, 1, e->ar, e->d_rows, B, e->s0, e->d_ss_dw_w[0], e->d_ss_dw_b[0], e->s1a, e->stream));
    if (gemm(e, e->s1a, dm.C, e->d_ss_pw_w[0], e->d_ss_pw_b[0], e->s1, dm.C, P[1], dm.C, dm.C, 0, 1) != 0) return -1;
    CK(e, "ss2 dw", k_ss_dw(dm, 2, e->ar, e->d_rows, B, e->s1, e->d_ss_dw_w[1], e->d_ss_dw_b[1], e->s2a, e->stream));
    if (gemm(e, e->s2a, dm.C, e->d_ss_pw_w[1], e->d_ss_pw_b[1], e->s2, dm.C, P[2], dm.C, dm.C, 0, 1) != 0) return -1;
    CK(e, "ss flatten", k_ss_flatten(dm, e->d_rows, B, e->s2, e->flat, e->stream));
    const int CF = dm.C * dm.Fo[GPU_SS_STAGES - 1];
    if (gemm(e, e->flat, CF, e->d_ss_lin_w, e->d_ss_lin_b, e->xs, dm.d, R, dm.d, CF, 0, 0) != 0) return -1;

    /* the conformer stack */
    const size_t nd = (size_t)R * dm.d;
    for (int li = 0; li < dm.n_layers; li++) {
        const gpu_layer_w &L = e->L[(size_t)li];
        prof_mark(e, PS_FFN1);
        CK(e, "ln ff1", k_layernorm(e->xs, L.ln_ff1_w, L.ln_ff1_b, e->tmp, R, dm.d, 0, e->stream));
        if (gemm(e, e->tmp, dm.d, L.ff1_w1, nullptr, e->tmp2, dm.ffn, R, dm.ffn, dm.d, 0, 2) != 0) return -1;
        if (gemm(e, e->tmp2, dm.ffn, L.ff1_w2, nullptr, e->tmp, dm.d, R, dm.d, dm.ffn, 0, 0) != 0) return -1;
        CK(e, "res ff1", k_residual(e->xs, e->tmp, 0.5f, nd, e->stream));

        prof_mark(e, PS_ATT_PROJ);
        CK(e, "ln att", k_layernorm(e->xs, L.ln_att_w, L.ln_att_b, e->xn, R, dm.d, 0, e->stream));
        if (gemm(e, e->xn, dm.d, L.k_w, nullptr, e->kn, dm.d, R, dm.d, dm.d, 0, 0) != 0) return -1;
        if (gemm(e, e->xn, dm.d, L.v_w, nullptr, e->vn, dm.d, R, dm.d, dm.d, 0, 0) != 0) return -1;
        if (gemm(e, e->xn, dm.d, L.q_w, nullptr, e->qs, dm.d, R, dm.d, dm.d, 0, 0) != 0) return -1;
        prof_mark(e, PS_ATT_CORE);
        CK(e, "attention", k_attention(dm, L, li, e->d_relpos, e->ar, e->d_rows, B, e->qs, e->kn, e->vn, e->ctx, e->stream));
        CK(e, "kv commit", k_kv_commit(dm, li, e->ar, e->d_rows, B, e->kn, e->vn, e->stream));
        prof_mark(e, PS_ATT_OUT);
        if (gemm(e, e->ctx, dm.d, L.o_w, nullptr, e->tmp, dm.d, R, dm.d, dm.d, 0, 0) != 0) return -1;
        CK(e, "res att", k_residual(e->xs, e->tmp, 1.0f, nd, e->stream));

        prof_mark(e, PS_CONV);
        CK(e, "ln conv", k_layernorm(e->xs, L.ln_conv_w, L.ln_conv_b, e->xn, R, dm.d, 0, e->stream));
        if (gemm(e, e->xn, dm.d, L.pw1_w, nullptr, e->h2, 2 * dm.d, R, 2 * dm.d, dm.d, 0, 0) != 0) return -1;
        CK(e, "glu dwconv", k_glu_dwconv(dm, L, li, e->ar, e->d_rows, B, e->h2, e->cmid, e->stream));
        CK(e, "conv norm", k_layernorm(e->cmid, L.cnorm_w, L.cnorm_b, e->tmp, R, dm.d, 1, e->stream));
        if (gemm(e, e->tmp, dm.d, L.pw2_w, nullptr, e->xn, dm.d, R, dm.d, dm.d, 0, 0) != 0) return -1;
        CK(e, "res conv", k_residual(e->xs, e->xn, 1.0f, nd, e->stream));

        prof_mark(e, PS_FFN2);
        CK(e, "ln ff2", k_layernorm(e->xs, L.ln_ff2_w, L.ln_ff2_b, e->tmp, R, dm.d, 0, e->stream));
        if (gemm(e, e->tmp, dm.d, L.ff2_w1, nullptr, e->tmp2, dm.ffn, R, dm.ffn, dm.d, 0, 2) != 0) return -1;
        if (gemm(e, e->tmp2, dm.ffn, L.ff2_w2, nullptr, e->tmp, dm.d, R, dm.d, dm.ffn, 0, 0) != 0) return -1;
        CK(e, "res ff2", k_residual(e->xs, e->tmp, 0.5f, nd, e->stream));
        CK(e, "ln out", k_layernorm(e->xs, L.ln_out_w, L.ln_out_b, e->xs, R, dm.d, 0, e->stream));
    }
    prof_mark(e, PS_POST);
    CK(e, "kv advance", k_kv_advance(dm, e->ar, e->d_rows, B, e->stream));

    /* prompt + projector */
    CK(e, "prompt cat", k_prompt_cat(e->xs, e->d_rows, B, dm.d, dm.np, e->cat, e->stream));
    if (gemm(e, e->cat, dm.d + dm.np, e->d_pl1_w, e->d_pl1_b, e->mid, dm.inter, R, dm.inter, dm.d + dm.np, 0, 1) != 0) return -1;
    if (gemm(e, e->mid, dm.inter, e->d_pl2_w, e->d_pl2_b, e->fused, dm.d, R, dm.d, dm.inter, 0, 0) != 0) return -1;
    if (gemm(e, e->fused, dm.d, e->d_ep_w, e->d_ep_b, e->enc, dm.dout, R, dm.dout, dm.d, 0, 0) != 0) return -1;

    /* the label loop */
    CK(e, "dec begin", k_dec_begin(e->ar, e->d_rows, B, e->stream));
    const int H = dm.Hdec;
    const int iter_cap = e->pack.qmax * (dm.max_symbols + 1) + 2;
    for (int it = 0; it < iter_cap; it++) {
        prof_mark(e, PS_DEC_SYNC);
        CK(e, "dec compact", k_dec_compact(e->ar, e->d_rows, B, e->d_active, e->d_n_active, e->stream));
        CK(e, "d2h n_active", cudaMemcpyAsync(e->h_n_active, e->d_n_active, sizeof(int), cudaMemcpyDeviceToHost, e->stream));
        CK(e, "sync", cudaStreamSynchronize(e->stream));
        const int n = *e->h_n_active;
        if (n <= 0) break;
        e->st.decode_iters++;
        prof_mark(e, PS_DEC_JOINT);
        CK(e, "dec joint", k_dec_joint(dm, e->ar, e->d_rows, e->d_active, n, e->enc, e->jin, e->stream));
        if (gemm(e, e->jin, H, e->d_head_w, e->d_head_b, e->logits, dm.V, n, dm.V, H, 0, 0) != 0) return -1;
        CK(e, "dec argmax", k_dec_argmax(e->logits, dm.V, n, e->d_am, e->stream));
        CK(e, "dec decide", k_dec_decide(dm, e->ar, e->d_rows, e->d_active, n, e->d_am, e->d_emit, e->stream));
        /* predictor step over the active lanes; only the emitting lanes commit */
        prof_mark(e, PS_DEC_PRED);
        CK(e, "dec emb", k_dec_gather_emb(e->d_emb, H, e->d_am, n, e->xin, e->stream));
        float *x = e->xin, *xn2 = e->xnext;
        for (int l = 0; l < dm.pred_layers; l++) {
            CK(e, "dec gather h", k_dec_gather_h(dm, e->ar, e->d_rows, e->d_active, n, l, e->hrows, e->stream));
            if (gemm(e, x, H, e->d_wih[l], e->d_bsum[l], e->z, 4 * H, n, 4 * H, H, 0, 0) != 0) return -1;
            if (gemm(e, e->hrows, H, e->d_whh[l], nullptr, e->z, 4 * H, n, 4 * H, H, 1, 0) != 0) return -1;
            CK(e, "dec gates", k_dec_lstm_gates(dm, e->ar, e->d_rows, e->d_active, e->d_emit, n, l, e->z, xn2, e->stream));
            float *t = x; x = xn2; xn2 = t;
        }
        if (gemm(e, x, H, e->d_proj_w, e->d_proj_b, e->gtmp, H, n, H, H, 0, 0) != 0) return -1;
        CK(e, "dec commit g", k_dec_commit_g(dm, e->ar, e->d_rows, e->d_active, e->d_emit, e->d_am, n, e->gtmp, e->stream));
    }
    /* results: tokens and metadata of every slot (small), once */
    prof_mark(e, PS_D2H);
    CK(e, "d2h tok", cudaMemcpyAsync(e->h_tok, e->ar.tok, (size_t)e->cap * e->ar.tok_cap * sizeof(int), cudaMemcpyDeviceToHost, e->stream));
    CK(e, "d2h meta", cudaMemcpyAsync(e->h_meta, e->ar.meta, (size_t)e->cap * sizeof(gpu_slot_meta), cudaMemcpyDeviceToHost, e->stream));
    prof_mark(e, PS_OTHER);
    CK(e, "sync", cudaStreamSynchronize(e->stream));
    prof_collect(e);
    e->st.d2h_bytes += (double)e->cap * e->ar.tok_cap * sizeof(int) + (double)e->cap * sizeof(gpu_slot_meta);
    e->st.rows += (unsigned long)R;
    e->st.lanes += (unsigned long)B;

    for (int a = 0; a < B; a++) {
        const pass_lane &l = lanes[(size_t)a];
        slot_host &s = e->slots[(size_t)l.slot];
        asr_step_out &o = outs[l.req];
        const int ntok = e->h_meta[l.slot].dec_ntok;
        const int *tok = e->h_tok + (size_t)l.slot * e->ar.tok_cap;
        char lang_tmp[16] = "";
        const size_t before = s.chars_emitted;
        const char *text = mynah_asr_detok_append(&s.detok, &e->pack.tok, tok, ntok, lang_tmp);
        for (int k = 0; k < ntok; k++) s.tokens.push_back(tok[k]);
        if (lang_tmp[0]) memcpy(s.lang, lang_tmp, sizeof(s.lang));
        o.n_tokens = ntok;
        o.stepped = 1;
        const double t1 = (double)s.samples_fed / (double)e->pack.feat.sample_rate;
        if (text && strlen(text) > before) {
            o.text = text + before;
            o.t0 = s.emitted_t1; o.t1 = t1;
            s.chars_emitted = strlen(text);
            s.emitted_t1 = t1;
        } else {
            o.text = ""; o.t0 = s.emitted_t1; o.t1 = t1;
        }
        /* consume the chunk */
        s.mel_have -= l.n_mel;
        if (s.mel_have > 0)
            memmove(s.mel_buf, s.mel_buf + (size_t)l.n_mel * nm, (size_t)s.mel_have * nm * sizeof(float));
        s.first = 0;
        if (l.last || (reqs[l.req].finalize && s.mel_finished && s.mel_have == 0)) {
            s.finished = 1;
            o.finished = 1;
        }
    }
    return 0;
}

static int cuda_step(cuda_engine *e, const asr_step_req *reqs, int n, asr_step_out *outs) {
    if (e->dead) return -1;
    const double t0 = now_s();
    for (int i = 0; i < n; i++) {
        outs[i].text = ""; outs[i].t0 = outs[i].t1 = 0.0;
        outs[i].n_tokens = 0; outs[i].finished = 0; outs[i].stepped = 0;
    }
    /* pending resets first: a slot reset since the last step starts clean */
    if (!e->pending_resets.empty()) {
        const int nr = (int)e->pending_resets.size();
        for (int i = 0; i < nr; i++) { e->h_resets[i] = e->pending_resets[(size_t)i]; e->slots[(size_t)e->h_resets[i]].reset_pending = 0; }
        CK(e, "h2d resets", cudaMemcpyAsync(e->d_resets, e->h_resets, (size_t)nr * sizeof(int), cudaMemcpyHostToDevice, e->stream));
        CK(e, "slots reset", k_slots_reset(e->dm, e->ar, e->d_resets, nr, e->d_sos_h, e->d_sos_c, e->d_sos_g, e->stream));
        e->pending_resets.clear();
    }
    /* decide, per request, what this step consumes */
    std::vector<pass_lane> lanes;
    lanes.reserve((size_t)n);
    int R = 0, M = 0, P[GPU_SS_STAGES] = {0, 0, 0};
    for (int i = 0; i < n; i++) {
        const int slot = reqs[i].slot;
        if (slot < 0 || slot >= e->cap) continue;
        slot_host &s = e->slots[(size_t)slot];
        if (!s.in_use || s.finished) { if (s.finished) outs[i].finished = 1; continue; }
        if (reqs[i].finalize && !s.mel_finished) {
            /* the leftover frames of the utterance, as mynah_asr_stream_finish does */
            const int room = s.mel_cap - s.mel_have;
            const int got = room > 0 ? mynah_asr_mel_stream_finish(&s.mel, s.mel_buf + (size_t)s.mel_have * e->dm.n_mels, room) : 0;
            if (got > 0) s.mel_have += got;
            s.mel_finished = 1;
        }
        const int need = slot_need_mel(e, s);
        pass_lane l = {i, slot, 0, s.first, 0, 0};
        if (s.mel_have >= need) {
            l.n_mel = need; l.last = 0;
        } else if (reqs[i].finalize && s.mel_have > 0) {
            l.n_mel = s.mel_have; l.last = 1;
        } else if (reqs[i].finalize) {
            s.finished = 1; outs[i].finished = 1; outs[i].t0 = outs[i].t1 = (double)s.samples_fed / e->pack.feat.sample_rate;
            continue;
        } else {
            continue;                                /* not ready: nothing to do */
        }
        int to[GPU_SS_STAGES];
        ss_geometry(l.n_mel, l.first, l.last, to);
        l.q = to[GPU_SS_STAGES - 1];
        if (l.q <= 0) {
            /* a tail too short to yield a frame: it is consumed and finishes */
            s.mel_have = 0; s.finished = 1; outs[i].finished = 1;
            outs[i].t0 = outs[i].t1 = (double)s.samples_fed / e->pack.feat.sample_rate;
            continue;
        }
        /* would this lane overflow the pass budget? then run what we have */
        int fits = (int)lanes.size() < e->Bmax && R + l.q <= e->Rmax && M + l.n_mel <= e->Mmax;
        for (int st = 0; st < GPU_SS_STAGES && fits; st++)
            if (P[st] + to[st] * e->dm.Fo[st] > e->Pmax[st]) fits = 0;
        if (!fits) {
            if (run_pass(e, lanes, reqs, outs) != 0) return -1;
            lanes.clear(); R = M = 0; P[0] = P[1] = P[2] = 0;
        }
        lanes.push_back(l);
        R += l.q; M += l.n_mel;
        for (int st = 0; st < GPU_SS_STAGES; st++) P[st] += to[st] * e->dm.Fo[st];
    }
    if (run_pass(e, lanes, reqs, outs) != 0) return -1;
    e->st.steps++;
    e->st.step_wall_ms_sum += (now_s() - t0) * 1e3;
    return 0;
}

/* ------------------------------------------------------------- the ops table */
#define CE(e) ((cuda_engine *)(e))
#define CCE(e) ((const cuda_engine *)(e))
static void ops_close(asr_engine *e) { cuda_close(CE(e)); }
static void ops_facts(const asr_engine *e, asr_engine_facts *f) { cuda_facts(CCE(e), f); }
static void ops_stats(const asr_engine *e, asr_engine_stats *s) { cuda_stats(CCE(e), s); }
static int ops_lang_id(const asr_engine *e, const char *l) { return cuda_lang_id(CCE(e), l); }
static int ops_lookahead_ok(const asr_engine *e, int la) { return cuda_lookahead_ok(CCE(e), la); }
static int ops_slot_reset(asr_engine *e, int s, const char *l, int la) { return cuda_slot_reset(CE(e), s, l, la); }
static size_t ops_slot_need(const asr_engine *e, int s) { return cuda_slot_need_samples(CCE(e), s); }
static int ops_slot_feed(asr_engine *e, int s, const float *p, size_t n) { return cuda_slot_feed(CE(e), s, p, n); }
static int ops_slot_ready(const asr_engine *e, int s) { return cuda_slot_ready(CCE(e), s); }
static double ops_slot_audio(const asr_engine *e, int s) { return cuda_slot_audio_s(CCE(e), s); }
static const char *ops_slot_text(const asr_engine *e, int s) { return cuda_slot_text(CCE(e), s); }
static const char *ops_slot_lang(const asr_engine *e, int s) { return cuda_slot_lang(CCE(e), s); }
static int ops_step(asr_engine *e, const asr_step_req *r, int n, asr_step_out *o) { return cuda_step(CE(e), r, n, o); }
static int ops_dead(const asr_engine *e) { return cuda_dead(CCE(e)); }
static const char *ops_error(const asr_engine *e) { return cuda_error(CCE(e)); }
static size_t ops_dispatch(const asr_engine *e, char *b, size_t c) { return cuda_dispatch_map(CCE(e), b, c); }
static const asr_engine_ops CUDA_OPS = {
    ops_close, ops_facts, ops_stats, ops_lang_id, ops_lookahead_ok, ops_slot_reset, ops_slot_need,
    ops_slot_feed, ops_slot_ready, ops_slot_audio, ops_slot_text, ops_slot_lang, ops_step, ops_dead,
    ops_error, ops_dispatch,
};
static const asr_engine_ops *cuda_ops(void) { return &CUDA_OPS; }
