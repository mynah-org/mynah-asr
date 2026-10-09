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

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <string>
#include <unordered_map>
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

/* one lane of a pass, decided at submit */
struct pass_lane {
    int req;                    /* index into reqs/outs */
    int slot, n_mel, first, last, q;
    double t1;                  /* audio fed when the chunk was taken: the delta's t1 */
    int fin;                    /* this chunk ends the utterance */
};

/* one encoder pass: its lanes, its counts, and how it launches */
struct pass_rec {
    std::vector<pass_lane> lanes;
    int R = 0, M = 0, P[GPU_SS_STAGES] = {0, 0, 0};
    int Bk = 0, gi = -1;        /* graph: lanes launched (real + padding), exec index; -1 = eager */
};

struct cuda_engine;
static void slot_feed_one(cuda_engine *e, int slot, const float *pcm, size_t n);

/* The feed team (--host-threads): persistent helpers that run the host mel of
 * a slot_feed_batch next to the caller. Slots are claimed one at a time with an
 * atomic counter; each slot's state is touched by exactly one thread; helpers
 * park on a condition variable between batches and make no CUDA call. The
 * mel of a slot does not depend on which thread computed it. */
struct feed_team;
struct team_arg { feed_team *t; int who; };
struct feed_team {
    cuda_engine *e = nullptr;
    int helpers = 0;
    std::vector<pthread_t> th;
    std::vector<team_arg> args;
    pthread_mutex_t mu;
    pthread_cond_t go, done;
    unsigned long gen = 0;
    int quit = 0, reported = 0;
    int n = 0;
    const int *slots = nullptr;
    const float *const *pcm = nullptr;
    const size_t *ns = nullptr;
    std::atomic<int> next{0};
    std::vector<double> ms;     /* per participant: [0] the caller, [1..] helpers */
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
    int gemm_v2 = 0;   /* --gemm own-v2: S14-8a, byte-identical, not faster (kept as an arm) */
    int gemm_splitk = 0; /* --gemm splitk: S14-8b, row-stable by construction, NOT bit-identical to v1 */
    /* --weights int8: the quantised matrices. The f32 pointer slots of those
     * weights (gpu_layer_w, d_head_w) hold the int8 device pointer instead, and
     * gemm() routes any W found here to k_gemm_w8; nothing dereferences those
     * slots as f32. */
    struct w8_mat { const int8_t *q; const float *s; };
    int weights_int8 = 0;
    std::unordered_map<const void *, w8_mat> w8;
    float *sk_ws = nullptr;
    size_t sk_ws_floats = 0;
    /* --precision bf16 --gemm own-tc: a resident bf16 copy of every GEMM weight,
     * keyed by its f32 device pointer (the call sites stay as they are) */
    int gemm_tc = 0;
    std::unordered_map<const float *, const unsigned short *> w16;
    float *tc_ws = nullptr;
    size_t tc_ws_floats = 0;
    std::vector<slot_host> slots;
    std::vector<int> pending_resets;
    asr_engine_stats st = {};
    size_t vram_weights = 0, vram_arena = 0, vram_total = 0, vram_kv = 0;
    char err[512] = {0};
    int dead = 0;
    std::string devname;
    char pci_bus_id[32] = {0};
    int device = 0;   /* cudaSetDevice is per host thread: every op re-binds it */
    /* S14-6b: CUDA events at stage boundaries, one pass at a time */
    int prof = 0;
    static const int EV_MAX = 4096;
    cudaEvent_t ev[4096];
    int ev_tag[4096];
    int nev = 0;
    /* --profile-host: host wall per phase of a step (asr_engine_stats.hprof_*) */
    int hprof = 0;
    /* the split step: passes decided at submit, each with its own pinned
     * staging slab (h_rows + p*Bmax, h_mel + p*Mmax*n_mels), so nothing a
     * later pass or the next step stages can overwrite what is in flight */
    int arena_slots = 0;        /* cap, + 1 scratch slot (index cap) when graphs pad lanes */
    int npass_max = 0;
    std::vector<pass_rec> passes;
    int n_passes = 0, inflight = 0;
    double submit_ms = 0.0;
    /* CUDA graphs of the encoder pass, one per (lane bucket, row bucket) */
    int graphs = 0, capturing = 0;
    int RG = 64;                /* row bucket step = the GEMM's M tile */
    std::vector<int> gb;        /* lane buckets, ascending, the last = Bmax */
    std::vector<int> gfirst, gnr; /* per lane bucket: first exec index, row buckets */
    std::vector<cudaGraphExec_t> gx;
    double capture_ms = 0.0, warmup_ms = 0.0;
    feed_team *team = nullptr;
    int host_threads = 1;
    double hp_in = 0.0;         /* --profile-host: phase total at submit */
};

enum { PS_H2D = 0, PS_SS, PS_FFN1, PS_ATT_PROJ, PS_ATT_CORE, PS_ATT_OUT, PS_CONV, PS_FFN2,
       PS_POST, PS_DEC_JOINT, PS_DEC_PRED, PS_DEC_SYNC, PS_D2H, PS_OTHER };

/* mark: from here on the stream's time is charged to `tag` */
static inline void prof_mark(cuda_engine *e, int tag) {
    if (!e->prof || e->capturing || e->nev >= cuda_engine::EV_MAX) return;
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

/* --profile-host: charge the time since `t` to phase `ph` and return now.
 * With the profile off neither reads a clock: one predictable branch. */
static inline double hp_now(const cuda_engine *e) { return e->hprof ? now_s() : 0.0; }
static inline double hp_lap(cuda_engine *e, int ph, double t) {
    if (!e->hprof) return 0.0;
    const double n = now_s();
    e->st.hprof_us[ph] += (n - t) * 1e6;
    return n;
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

/* --weights int8: per-row symmetric codes and scales on the device, the same
 * codes the CPU int8 path holds (a pre-quantised pack's own, else quantised
 * here with the library's quantiser). Returns the int8 pointer, typed as the
 * f32 slot it replaces; gemm() finds it in e->w8. */
static const float *upload_w8_raw(cuda_engine *e, const float *w32, const int8_t *q8, const float *sc,
                                  int n, int k, const char *what) {
    std::vector<int8_t> q;
    std::vector<float> s;
    if (!q8) {
        if (!w32) { snprintf(e->err, sizeof(e->err), "%s: missing tensor", what); return nullptr; }
        q.resize((size_t)n * (size_t)k);
        s.resize((size_t)n);
        mynah_asr_quantize_int8(w32, n, k, q.data(), s.data());
        q8 = q.data(); sc = s.data();
    }
    void *dq = nullptr, *ds = nullptr;
    if (dmalloc(e, &dq, (size_t)n * (size_t)k, &e->vram_weights, what) != 0) return nullptr;
    if (dmalloc(e, &ds, (size_t)n * sizeof(float), &e->vram_weights, what) != 0) return nullptr;
    cudaError_t c = cudaMemcpy(dq, q8, (size_t)n * (size_t)k, cudaMemcpyHostToDevice);
    if (c == cudaSuccess) c = cudaMemcpy(ds, sc, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (c != cudaSuccess) { set_err(e, what, c); return nullptr; }
    e->w8[dq] = cuda_engine::w8_mat{(const int8_t *)dq, (const float *)ds};
    return (const float *)dq;
}

/* a qmat-backed linear: int8 when --weights int8, else the f32 upload */
static const float *upload_lin(cuda_engine *e, const mynah_asr_qmat *m, int n, int k, const char *what) {
    if (e->weights_int8) {
        if (m->qtype == MYNAH_ASR_Q_INT8 && m->n == n && m->k == k)
            return upload_w8_raw(e, nullptr, m->q8, m->scales, n, k, what);
        return upload_w8_raw(e, qf32(m), nullptr, nullptr, n, k, what);
    }
    return upload(e, qf32(m), (size_t)n * (size_t)k, what);
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
    if (e->gemm_tc) {
        auto it = e->w16.find(W);
        if (it == e->w16.end()) {
            snprintf(e->err, sizeof(e->err), "own-tc: a GEMM weight has no bf16 copy (N=%d K=%d)", N, K);
            e->dead = 1; e->st.errors++;
            return -1;
        }
        CK(e, "gemm tc", k_gemm_wt_tc(A, lda, it->second, bias, C, ldc, M, N, K, accumulate, act, e->tc_ws, e->tc_ws_floats, e->stream));
        return 0;
    }
    if (e->weights_int8) {
        auto it = e->w8.find((const void *)W);
        if (it != e->w8.end()) {
            CK(e, "gemm w8", k_gemm_w8(A, lda, it->second.q, it->second.s, bias, C, ldc, M, N, K, accumulate, act, e->stream));
            return 0;
        }
    }
    if (!e->use_cublas) {
        if (e->gemm_splitk) CK(e, "gemm", k_gemm_wt_splitk(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, e->sk_ws, e->sk_ws_floats, e->stream));
        else if (e->gemm_v2) CK(e, "gemm", k_gemm_wt_v2(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, e->stream));
        else CK(e, "gemm", k_gemm_wt(A, lda, W, bias, C, ldc, M, N, K, accumulate, act, e->stream));
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
    const int lanes = e->cfg.pass_lanes > 0 ? e->cfg.pass_lanes : 128;
    e->Bmax = e->cap < lanes ? e->cap : lanes;
    e->Rmax = e->Bmax * qmax;
    /* the per-lane bound is the larger of the first chunk (1 + sub*L) and a
     * steady one (sub*(L+1), the larger), with the widest padding, so that a
     * lane alone always fits an empty pass whatever Bmax is */
    const int mel_first = asr_pack_chunk_mel(&e->pack, qmax - 1, 1);
    const int mel_steady = asr_pack_chunk_mel(&e->pack, qmax - 1, 0);
    const int melmax = mel_first > mel_steady ? mel_first : mel_steady;
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
    if (e->gemm_splitk || e->gemm_tc) {
        /* every (M bound, N, K) the pass issues; the largest S*M*N wins */
        const int R_ = e->Rmax, B_ = e->Bmax, P1 = e->Pmax[1], P2 = e->Pmax[2];
        const int CF_ = dm.C * dm.Fo[GPU_SS_STAGES - 1];
        const int shp[][3] = {{P1, dm.C, dm.C}, {P2, dm.C, dm.C}, {R_, dm.d, CF_}, {R_, dm.ffn, dm.d}, {R_, dm.d, dm.ffn},
                              {R_, dm.d, dm.d}, {R_, 2 * dm.d, dm.d}, {R_, dm.inter, dm.d + dm.np}, {R_, dm.d, dm.inter},
                              {R_, dm.dout, dm.d}, {B_, dm.V, dm.Hdec}, {B_, 4 * dm.Hdec, dm.Hdec}, {B_, dm.Hdec, dm.Hdec}};
        for (size_t i = 0; i < sizeof(shp) / sizeof(shp[0]); i++) {
            if (e->gemm_tc) {
                const size_t f = k_gemm_tc_workspace_floats(shp[i][0], shp[i][1], shp[i][2]);
                if (f > e->tc_ws_floats) e->tc_ws_floats = f;
            } else {
                const size_t f = k_gemm_splitk_workspace_floats(shp[i][0], shp[i][1], shp[i][2]);
                if (f > e->sk_ws_floats) e->sk_ws_floats = f;
            }
        }
        if (e->sk_ws_floats && dmalloc(e, (void **)&e->sk_ws, e->sk_ws_floats * sizeof(float), acct, "split-K workspace") != 0) return -1;
        if (e->tc_ws_floats && dmalloc(e, (void **)&e->tc_ws, e->tc_ws_floats * sizeof(float), acct, "own-tc split workspace") != 0) return -1;
    }
    if (dmalloc(e, (void **)&e->d_rows, B * sizeof(gpu_row), acct, "rows") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_active, B * sizeof(int), acct, "active") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_am, B * sizeof(int), acct, "am") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_emit, B * sizeof(int), acct, "emit") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_n_active, sizeof(int), acct, "n_active") != 0) return -1;
    if (dmalloc(e, (void **)&e->d_resets, (size_t)e->arena_slots * sizeof(int), acct, "resets") != 0) return -1;
    /* a step splits into at most ceil(cap / Bmax) passes (a pass closes only
     * when Bmax lanes are in, every budget being Bmax x the per-lane bound);
     * one slab of slack */
    e->npass_max = (e->cap + e->Bmax - 1) / e->Bmax + 1;
    const size_t NP = (size_t)e->npass_max;
    CK(e, "pinned rows", cudaHostAlloc((void **)&e->h_rows, NP * B * sizeof(gpu_row), cudaHostAllocPortable));
    CK(e, "pinned mel", cudaHostAlloc((void **)&e->h_mel, NP * (size_t)e->Mmax * dm.n_mels * sizeof(float), cudaHostAllocPortable));
    CK(e, "pinned n_active", cudaHostAlloc((void **)&e->h_n_active, sizeof(int), cudaHostAllocPortable));
    CK(e, "pinned resets", cudaHostAlloc((void **)&e->h_resets, (size_t)e->arena_slots * sizeof(int), cudaHostAllocPortable));
    e->passes.resize(NP);
    for (auto &p : e->passes) p.lanes.reserve(B);
    CK(e, "pinned tok", cudaHostAlloc((void **)&e->h_tok, (size_t)e->cap * e->ar.tok_cap * sizeof(int), cudaHostAllocPortable));
    CK(e, "pinned meta", cudaHostAlloc((void **)&e->h_meta, (size_t)e->cap * sizeof(gpu_slot_meta), cudaHostAllocPortable));
    return 0;
}

static int alloc_arena(cuda_engine *e) {
    const gpu_model_dims &dm = e->dm;
    const size_t cap = (size_t)e->arena_slots;
    size_t *acct = &e->vram_arena;
    const size_t kv_n = cap * dm.n_layers * 2 * dm.left * dm.d;
    const size_t kv_esz = e->ar.kv_dtype == GPU_KV_INT8 ? 1 : e->ar.kv_dtype == GPU_KV_BF16 ? 2 : sizeof(float);
    void *kvp = nullptr;
    if (dmalloc(e, &kvp, kv_n * kv_esz, acct, "kv arena") != 0) return -1;
    CK(e, "arena memset", cudaMemset(kvp, 0, kv_n * kv_esz));
    e->vram_kv = kv_n * kv_esz;
    if (e->ar.kv_dtype == GPU_KV_F32) e->ar.kv = (float *)kvp;
    else e->ar.kvq = kvp;
    if (e->ar.kv_dtype == GPU_KV_INT8) {
        const size_t ns = cap * dm.n_layers * 2 * dm.left * dm.H;
        if (dmalloc(e, (void **)&e->ar.kv_scale, ns * sizeof(float), acct, "kv scales") != 0) return -1;
        CK(e, "arena memset", cudaMemset(e->ar.kv_scale, 0, ns * sizeof(float)));
        e->vram_kv += ns * sizeof(float);
    }
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
#define UL(dst, qm, n, k, what) do { (dst) = upload_lin(e, (qm), (n), (k), what); if (!(dst)) return -1; } while (0)
        UP(L.ln_ff1_w, s.ln_ff1_w, d, "ln_ff1_w"); UP(L.ln_ff1_b, s.ln_ff1_b, d, "ln_ff1_b");
        UL(L.ff1_w1, &s.ff1_w1, dm.ffn, dm.d, "ff1_w1");
        UL(L.ff1_w2, &s.ff1_w2, dm.d, dm.ffn, "ff1_w2");
        UP(L.ln_att_w, s.ln_att_w, d, "ln_att_w"); UP(L.ln_att_b, s.ln_att_b, d, "ln_att_b");
        UL(L.q_w, &s.q_w, dm.d, dm.d, "q_w"); UL(L.k_w, &s.k_w, dm.d, dm.d, "k_w");
        UL(L.v_w, &s.v_w, dm.d, dm.d, "v_w"); UL(L.o_w, &s.o_w, dm.d, dm.d, "o_w");
        UP(L.bias_u, s.bias_u, d, "bias_u"); UP(L.bias_v, s.bias_v, d, "bias_v");
        UP(L.ln_conv_w, s.ln_conv_w, d, "ln_conv_w"); UP(L.ln_conv_b, s.ln_conv_b, d, "ln_conv_b");
        UL(L.pw1_w, &s.pw1_w, 2 * dm.d, dm.d, "pw1_w");
        UP(L.dw_w, s.dw_w, d * (size_t)dm.conv_k, "dw_w");
        UP(L.cnorm_w, s.cnorm_w, d, "cnorm_w"); UP(L.cnorm_b, s.cnorm_b, d, "cnorm_b");
        UL(L.pw2_w, &s.pw2_w, dm.d, dm.d, "pw2_w");
        UP(L.ln_ff2_w, s.ln_ff2_w, d, "ln_ff2_w"); UP(L.ln_ff2_b, s.ln_ff2_b, d, "ln_ff2_b");
        UL(L.ff2_w1, &s.ff2_w1, dm.ffn, dm.d, "ff2_w1");
        UL(L.ff2_w2, &s.ff2_w2, dm.d, dm.ffn, "ff2_w2");
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
    { const float *hw_ = upload_lin(e, &dec.head, dm.V, dm.Hdec, "head"); if (!hw_) return -1; e->d_head_w = (float *)hw_; }
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
#undef UL
    return 0;
}

/* own-tc: the bf16 copy of one GEMM weight [N, K], converted on the device
 * (round to nearest even) from the f32 upload; the f32 copy stays resident */
static int tc_weight(cuda_engine *e, const float *W, size_t n) {
    if (!W || e->w16.count(W)) return 0;
    void *d = nullptr;
    if (dmalloc(e, &d, n * sizeof(unsigned short), &e->vram_weights, "bf16 weight") != 0) return -1;
    CK(e, "bf16 weight", k_f32_to_bf16(W, (unsigned short *)d, n, e->stream));
    e->w16[W] = (const unsigned short *)d;
    return 0;
}

static int tc_weights(cuda_engine *e) {
    const gpu_model_dims &dm = e->dm;
    const size_t d = (size_t)dm.d, ffn = (size_t)dm.ffn, C = (size_t)dm.C, H = (size_t)dm.Hdec;
    for (const gpu_layer_w &L : e->L) {
        if (tc_weight(e, L.ff1_w1, ffn * d) || tc_weight(e, L.ff1_w2, d * ffn) ||
            tc_weight(e, L.q_w, d * d) || tc_weight(e, L.k_w, d * d) || tc_weight(e, L.v_w, d * d) ||
            tc_weight(e, L.o_w, d * d) || tc_weight(e, L.pw1_w, 2 * d * d) || tc_weight(e, L.pw2_w, d * d) ||
            tc_weight(e, L.ff2_w1, ffn * d) || tc_weight(e, L.ff2_w2, d * ffn))
            return -1;
    }
    for (int i = 0; i < 2; i++)
        if (tc_weight(e, e->d_ss_pw_w[i], C * C)) return -1;
    if (tc_weight(e, e->d_ss_lin_w, d * C * (size_t)dm.Fo[GPU_SS_STAGES - 1]) ||
        tc_weight(e, e->d_pl1_w, (size_t)dm.inter * (size_t)(dm.d + dm.np)) ||
        tc_weight(e, e->d_pl2_w, d * (size_t)dm.inter) || tc_weight(e, e->d_ep_w, (size_t)dm.dout * d) ||
        tc_weight(e, e->d_proj_w, H * H) || tc_weight(e, e->d_head_w, (size_t)dm.V * H))
        return -1;
    for (int l = 0; l < dm.pred_layers; l++)
        if (tc_weight(e, e->d_wih[l], 4 * H * H) || tc_weight(e, e->d_whh[l], 4 * H * H)) return -1;
    CK(e, "bf16 weights", cudaStreamSynchronize(e->stream));
    return 0;
}

static const asr_engine_ops *cuda_ops(void);
static void cuda_close(cuda_engine *e);
static int parse_buckets(cuda_engine *e, const char *list, char *err, size_t errcap);
static int graphs_build(cuda_engine *e, char *err, size_t errcap);
static int warmup_run(cuda_engine *e);
static feed_team *team_open(cuda_engine *e, int helpers);
static void team_close(feed_team *t);

extern "C" int asr_engine_cuda_cc_major(int device) {
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) return -1;
    return major;
}

extern "C" asr_engine *asr_engine_open_cuda(const asr_engine_cfg *cfg, char *err, size_t errcap) {
    cuda_engine *e = new cuda_engine();
    e->base.ops = cuda_ops();
    e->cfg = *cfg;
    e->cap = cfg->cap > 0 ? cfg->cap : 1;
    /* bf16 exists only as the own-tc arm, and own-tc only in bf16: one flag
     * never silently changes what the other means */
    const int want_bf16 = cfg->precision && strcmp(cfg->precision, "bf16") == 0;
    e->gemm_tc = cfg->gemm && strcmp(cfg->gemm, "own-tc") == 0;
    if (cfg->precision && strcmp(cfg->precision, "f32") != 0 && !want_bf16) {
        snprintf(err, errcap, "precision '%s' is not one of f32, bf16", cfg->precision);
        delete e; return nullptr;
    }
    if (want_bf16 != e->gemm_tc) {
        snprintf(err, errcap, "--precision bf16 goes with --gemm own-tc and only with it (got precision=%s gemm=%s)",
                 cfg->precision ? cfg->precision : "f32", cfg->gemm ? cfg->gemm : "own");
        delete e; return nullptr;
    }
    e->use_cublas = cfg->gemm && strcmp(cfg->gemm, "cublas") == 0;
    e->gemm_v2 = cfg->gemm && strcmp(cfg->gemm, "own-v2") == 0;
    e->gemm_splitk = cfg->gemm && strcmp(cfg->gemm, "splitk") == 0;
    e->prof = cfg->profile ? 1 : 0;
    if (!cfg->weights || strcmp(cfg->weights, "f32") == 0) e->weights_int8 = 0;
    else if (strcmp(cfg->weights, "int8") == 0) e->weights_int8 = 1;
    else {
        snprintf(err, errcap, "weights '%s' is not one of f32, int8", cfg->weights);
        delete e; return nullptr;
    }
    if (!cfg->kv_dtype || strcmp(cfg->kv_dtype, "f32") == 0) e->ar.kv_dtype = GPU_KV_F32;
    else if (strcmp(cfg->kv_dtype, "bf16") == 0) e->ar.kv_dtype = GPU_KV_BF16;
    else if (strcmp(cfg->kv_dtype, "int8") == 0) e->ar.kv_dtype = GPU_KV_INT8;
    else {
        snprintf(err, errcap, "kv dtype '%s' is not one of f32, bf16, int8", cfg->kv_dtype);
        delete e; return nullptr;
    }
    e->hprof = cfg->profile_host ? 1 : 0;
    /* the label loop's lane compaction is one block over at most 1024 lanes */
    if (cfg->pass_lanes < 0 || cfg->pass_lanes > 1024) {
        snprintf(err, errcap, "pass lanes %d outside 1..1024 (0 = the default min(cap, 128))", cfg->pass_lanes);
        delete e; return nullptr;
    }
    if (cfg->graphs && (e->use_cublas || e->prof)) {
        snprintf(err, errcap, "--graphs buckets serves the own GEMMs without --profile-stages (cuBLAS and the event profile are eager-only)");
        delete e; return nullptr;
    }
    /* graphs pad a pass with q = 0 lanes on one extra scratch slot (index cap) */
    e->arena_slots = e->cap + (cfg->graphs ? 1 : 0);
    if (cfg->gemm && !e->use_cublas && strcmp(cfg->gemm, "own") != 0 && strcmp(cfg->gemm, "own-v2") != 0 && strcmp(cfg->gemm, "splitk") != 0 && !e->gemm_tc) {
        snprintf(err, errcap, "gemm '%s' is not one of own, own-v2, splitk, own-tc, cublas", cfg->gemm);
    if (e->weights_int8 && e->gemm_tc) {
        snprintf(err, errcap, "--weights int8 and --gemm own-tc are separate arms (one GEMM per weight)");
        delete e; return nullptr;
    }
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
    e->device = cfg->device;
    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, cfg->device) == cudaSuccess) e->devname = prop.name;
    if (cudaDeviceGetPCIBusId(e->pci_bus_id, (int)sizeof(e->pci_bus_id), cfg->device) != cudaSuccess) e->pci_bus_id[0] = '\0';
    if (e->gemm_tc && prop.major < 8) {
        snprintf(err, errcap, "--gemm own-tc needs bf16 tensor cores (sm_80+); %s is sm_%d%d", prop.name, prop.major, prop.minor);
        cuda_close(e); return nullptr;
    }
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
    if (upload_weights(e) != 0 || (e->gemm_tc && tc_weights(e) != 0) || alloc_arena(e) != 0 || alloc_scratch(e) != 0) {
        snprintf(err, errcap, "%s", e->err[0] ? e->err : "device allocation failed");
        cuda_close(e); return nullptr;
    }
    e->slots.resize((size_t)e->cap);
    for (int i = 0; i < e->cap; i++)
        if (slot_host_init(e, &e->slots[(size_t)i]) != 0) {
            snprintf(err, errcap, "host slot state allocation failed"); cuda_close(e); return nullptr;
        }
    /* every slot starts reset (the scratch slot too) */
    for (int i = 0; i < e->arena_slots; i++) e->h_resets[i] = i;
    c = cudaMemcpyAsync(e->d_resets, e->h_resets, (size_t)e->arena_slots * sizeof(int), cudaMemcpyHostToDevice, e->stream);
    if (c == cudaSuccess) c = k_slots_reset(dm, e->ar, e->d_resets, e->arena_slots, e->d_sos_h, e->d_sos_c, e->d_sos_g, e->stream);
    if (c == cudaSuccess) c = cudaStreamSynchronize(e->stream);
    if (c != cudaSuccess) { snprintf(err, errcap, "initial reset: %s", cudaGetErrorString(c)); cuda_close(e); return nullptr; }
    if ((cfg->graphs || cfg->warmup) && parse_buckets(e, cfg->graph_buckets, err, errcap) != 0) { cuda_close(e); return nullptr; }
    if (cfg->graphs && graphs_build(e, err, errcap) != 0) { cuda_close(e); return nullptr; }
    if (cfg->host_threads > 1) {
        e->team = team_open(e, cfg->host_threads - 1);
        e->host_threads = 1 + e->team->helpers;
    }
    if (cfg->warmup && warmup_run(e) != 0) {
        snprintf(err, errcap, "warm-up: %s", e->err[0] ? e->err : "failed");
        cuda_close(e); return nullptr;
    }
    return &e->base;
}

static void cuda_close(cuda_engine *e) {
    if (!e) return;
    team_close(e->team);
    e->team = nullptr;
    for (auto x : e->gx) cudaGraphExecDestroy(x);
    e->gx.clear();
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
    static const char *const PREC[3][3] = {{"f32", "f32+kv-bf16", "f32+kv-int8"},
                                           {"w8a32", "w8a32+kv-bf16", "w8a32+kv-int8"},
                                           {"bf16", "bf16+kv-bf16", "bf16+kv-int8"}};
    f->precision = PREC[e->gemm_tc ? 2 : e->weights_int8][e->ar.kv_dtype];
    f->gemm = e->use_cublas ? "cublas-pedantic (measured NOT row-stable)" : e->gemm_tc ? "own-tc-bf16-rowstable" : e->gemm_splitk ? "own-splitk-rowstable" : e->gemm_v2 ? "own-rowstable-v2" : "own-rowstable-v1";
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
    f->graphs = (int)e->gx.size();
    f->graph_capture_ms = e->capture_ms;
    f->warmup_ms = e->warmup_ms;
    f->host_threads = e->host_threads;
    f->pass_lanes = e->Bmax;
    f->pci_bus_id = e->pci_bus_id;
}

static void cuda_stats(const cuda_engine *e, asr_engine_stats *s) { *s = e->st; }
static int cuda_lang_id(const cuda_engine *e, const char *lang) { return asr_pack_lang_id(&e->pack, lang); }
static int cuda_lookahead_ok(const cuda_engine *e, int la) { return asr_pack_lookahead_ok(&e->pack, la); }
static int cuda_dead(const cuda_engine *e) { return e->dead; }
static const char *cuda_error(const cuda_engine *e) { return e->err; }

static size_t cuda_dispatch_map(const cuda_engine *e, char *buf, size_t cap) {
    const char *g = e->use_cublas ? "cublas-sgemm-pedantic" : e->gemm_tc ? "own-tc-bf16" : e->gemm_splitk ? "own-splitk-rowstable" : e->gemm_v2 ? "own-rowstable-v2" : "own-rowstable-v1";
    char gl[256] = "graphs            off            launched kernel by kernel\n";
    if (e->graphs) {
        int k = snprintf(gl, sizeof(gl), "graphs            buckets        %d execs, encoder pass, lanes", (int)e->gx.size());
        for (size_t i = 0; i < e->gb.size() && k < (int)sizeof(gl) - 40; i++) k += snprintf(gl + k, sizeof(gl) - (size_t)k, "%s%d", i ? "," : " ", e->gb[i]);
        snprintf(gl + k, sizeof(gl) - (size_t)k, ", rows step %d; label loop eager\n", e->RG);
    }
    size_t k0 = (size_t)snprintf(buf, cap,
        "engine            cuda           %s\n"
        "gemm              %-14s row-stable-by-construction=%s%s\n"
        "subsampling       direct-3x3-s2  position-major, pointwise via gemm\n"
        "layernorm         own            double accumulators, fixed tree\n"
        "attention         own            rel-pos table, ring window, warp softmax\n"
        "conv              own            glu + causal depthwise k=%d, cached\n"
        "decoder           own            label loop, first-index argmax, lstm x%d\n"
        "mel               host           src/features.c (double fft), phase 1\n"
        "precision         %-14s %s\n"
        "kv ring           %-14s %s, %.2f MiB per slot\n"
        "%s"
        "host mel          %-14s %d thread(s), caller included\n",
        e->devname.c_str(), g, e->use_cublas ? "NO(measured)" : "yes",
        e->gemm_tc ? " splits=f(N,K) only (not the SM count), fixed-order reduction, wmma bf16 x bf16 -> f32"
                   : e->gemm_splitk ? " splits=f(N,K) fixed-order reduction, NOT bit-identical to v1" : "",
        e->dm.conv_k, e->dm.pred_layers, e->gemm_tc ? "bf16" : e->weights_int8 ? "w8a32" : "f32",
        e->gemm_tc ? "gemm weights bf16 resident (+ f32 copy), activations rounded to bf16, f32 accumulate; rest f32"
        : e->weights_int8 ? "encoder linears + joint head int8 per-row (own w8 gemm, row-stable), rest f32; NOT bit-identical to f32"
                          : "weights f32 resident, no tf32",
        e->ar.kv_dtype == GPU_KV_INT8 ? "int8" : e->ar.kv_dtype == GPU_KV_BF16 ? "bf16" : "f32",
        e->ar.kv_dtype == GPU_KV_INT8 ? "per (position, head) scale max|x|/127, NOT bit-identical to f32"
        : e->ar.kv_dtype == GPU_KV_BF16 ? "round-to-nearest-even, NOT bit-identical to f32" : "reference",
        (double)e->vram_kv / (double)(e->cap > 0 ? e->cap : 1) / 1048576.0,
        gl, e->team ? "feed-team" : "inline", e->host_threads);
    return k0;
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

/* the per-slot body of a feed: touches that slot's host state only (the feed
 * team runs it for distinct slots in parallel) */
static void slot_feed_one(cuda_engine *e, int slot, const float *pcm, size_t n) {
    slot_host &s = e->slots[(size_t)slot];
    s.samples_fed += (unsigned long)n;
    slot_pull_mel(e, s, pcm, n);
}

static int cuda_slot_feed(cuda_engine *e, int slot, const float *pcm, size_t n) {
    if (e->dead || slot < 0 || slot >= e->cap) return -1;
    const double t0 = now_s();
    slot_feed_one(e, slot, pcm, n);
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

/* The encoder half of a pass: from the staged rows and mel on the device to
 * the projector output `enc`, over B lanes, R stacked rows and P1/P2
 * subsampling positions (the M of the two pointwise GEMMs). The eager path
 * passes the pass's real counts. A graph is captured with a bucket's counts:
 * its padding lanes have q = 0 on the scratch slot (every per-lane kernel
 * loops over the lane's own q / to[] / n_mel, so they do nothing but read the
 * scratch slot), and its padding rows and positions belong to no lane: the
 * per-row kernels and the GEMMs compute them and nobody reads them. A real
 * row's bytes do not change, because every GEMM is row-stable in M by
 * construction (contract 4, the test_cuda_kernels byte gate) and every other
 * kernel is per row or per lane. */
static int enqueue_encoder(cuda_engine *e, int B, int R, int P1, int P2) {
    const gpu_model_dims &dm = e->dm;
    /* subsampling */
    prof_mark(e, PS_SS);
    CK(e, "ss0", k_ss_stage0(dm, e->ar, e->d_rows, B, e->d_mel, e->d_ss_in_w, e->d_ss_in_b, e->s0, e->stream));
    CK(e, "ss1 dw", k_ss_dw(dm, 1, e->ar, e->d_rows, B, e->s0, e->d_ss_dw_w[0], e->d_ss_dw_b[0], e->s1a, e->stream));
    if (gemm(e, e->s1a, dm.C, e->d_ss_pw_w[0], e->d_ss_pw_b[0], e->s1, dm.C, P1, dm.C, dm.C, 0, 1) != 0) return -1;
    CK(e, "ss2 dw", k_ss_dw(dm, 2, e->ar, e->d_rows, B, e->s1, e->d_ss_dw_w[1], e->d_ss_dw_b[1], e->s2a, e->stream));
    if (gemm(e, e->s2a, dm.C, e->d_ss_pw_w[1], e->d_ss_pw_b[1], e->s2, dm.C, P2, dm.C, dm.C, 0, 1) != 0) return -1;
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
    return 0;
}

/* the graph bucket's padded counts (see enqueue_encoder): P1 of a lane is at
 * most 2q + 1 stage-1 frames, P2 exactly q, so (2Rk + Bk) and Rk bound them */
static void bucket_counts(const cuda_engine *e, int Bk, int Rk, int *P1, int *P2) {
    const int a = (2 * Rk + Bk) * e->dm.Fo[1], b = Rk * e->dm.Fo[2];
    *P1 = a < e->Pmax[1] ? a : e->Pmax[1];
    *P2 = b < e->Pmax[2] ? b : e->Pmax[2];
}
static int bucket_rows(const cuda_engine *e, int bi, int ri) {
    const int cap_ = e->gb[(size_t)bi] * e->pack.qmax < e->Rmax ? e->gb[(size_t)bi] * e->pack.qmax : e->Rmax;
    const int r = (ri + 1) * e->RG;
    return r < cap_ ? r : cap_;
}

/* Stage pass p into its pinned slab and take its chunks out of the slots: the
 * descriptors, the packed mel, the per-lane results known now (t1, fin), and
 * the launch plan (a graph bucket when one covers it). After this the slots'
 * mel buffers hold only what comes AFTER the chunk in flight. */
static void pass_stage(cuda_engine *e, int p, const asr_step_req *reqs) {
    const gpu_model_dims &dm = e->dm;
    const int nm = dm.n_mels;
    pass_rec &ps = e->passes[(size_t)p];
    gpu_row *hr = e->h_rows + (size_t)p * e->Bmax;
    float *hm = e->h_mel + (size_t)p * e->Mmax * nm;
    const int B = (int)ps.lanes.size();
    int R = 0, M = 0, P[GPU_SS_STAGES] = {0, 0, 0};
    for (int a = 0; a < B; a++) {
        pass_lane &l = ps.lanes[(size_t)a];
        slot_host &s = e->slots[(size_t)l.slot];
        gpu_row &r = hr[a];
        memset(&r, 0, sizeof(r));
        r.slot = l.slot; r.q = l.q; r.n_mel = l.n_mel; r.mel_off = M; r.first = l.first; r.last = l.last;
        r.prompt = s.prompt; r.row_off = R;
        int to[GPU_SS_STAGES];
        ss_geometry(l.n_mel, l.first, l.last, to);
        for (int st = 0; st < GPU_SS_STAGES; st++) { r.to[st] = to[st]; r.pos_off[st] = P[st]; P[st] += to[st] * dm.Fo[st]; }
        memcpy(hm + (size_t)M * nm, s.mel_buf, (size_t)l.n_mel * nm * sizeof(float));
        M += l.n_mel; R += l.q;
        /* consume the chunk */
        s.mel_have -= l.n_mel;
        if (s.mel_have > 0)
            memmove(s.mel_buf, s.mel_buf + (size_t)l.n_mel * nm, (size_t)s.mel_have * nm * sizeof(float));
        s.first = 0;
        l.t1 = (double)s.samples_fed / (double)e->pack.feat.sample_rate;
        l.fin = l.last || (reqs[l.req].finalize && s.mel_finished && s.mel_have == 0);
        if (l.fin) s.finished = 1;
    }
    ps.R = R; ps.M = M;
    for (int st = 0; st < GPU_SS_STAGES; st++) ps.P[st] = P[st];
    ps.gi = -1; ps.Bk = B;
    if (!e->graphs || B == 0) return;
    int bi = 0;
    while (bi < (int)e->gb.size() && e->gb[(size_t)bi] < B) bi++;
    if (bi == (int)e->gb.size()) return;
    int ri = (R + e->RG - 1) / e->RG - 1;
    if (ri >= e->gnr[(size_t)bi]) ri = e->gnr[(size_t)bi] - 1;
    const int Bk = e->gb[(size_t)bi], Rk = bucket_rows(e, bi, ri);
    int P1k, P2k;
    bucket_counts(e, Bk, Rk, &P1k, &P2k);
    if (R > Rk || P[1] > P1k || P[2] > P2k) return;   /* not covered: eager */
    for (int a = B; a < Bk; a++) {
        gpu_row &r = hr[a];
        memset(&r, 0, sizeof(r));
        r.slot = e->cap;                               /* the scratch slot */
        r.row_off = R;
        for (int st = 0; st < GPU_SS_STAGES; st++) r.pos_off[st] = P[st];
    }
    ps.gi = e->gfirst[(size_t)bi] + ri;
    ps.Bk = Bk;
}

/* H2D of pass p's slab, then its encoder: the graph or the kernels */
static int pass_launch(cuda_engine *e, int p) {
    const pass_rec &ps = e->passes[(size_t)p];
    const int nm = e->dm.n_mels;
    const int nrows = ps.gi >= 0 ? ps.Bk : (int)ps.lanes.size();
    prof_mark(e, PS_H2D);
    CK(e, "h2d rows", cudaMemcpyAsync(e->d_rows, e->h_rows + (size_t)p * e->Bmax, (size_t)nrows * sizeof(gpu_row), cudaMemcpyHostToDevice, e->stream));
    CK(e, "h2d mel", cudaMemcpyAsync(e->d_mel, e->h_mel + (size_t)p * e->Mmax * nm, (size_t)ps.M * nm * sizeof(float), cudaMemcpyHostToDevice, e->stream));
    e->st.h2d_bytes += (double)nrows * sizeof(gpu_row) + (double)ps.M * nm * sizeof(float);
    if (ps.gi >= 0) {
        CK(e, "graph launch", cudaGraphLaunch(e->gx[(size_t)ps.gi], e->stream));
        e->st.graph_passes++;
        return 0;
    }
    e->st.eager_passes++;
    return enqueue_encoder(e, (int)ps.lanes.size(), ps.R, ps.P[1], ps.P[2]);
}

/* the label loop of pass p (its encoder enqueued), the D2H, the wait, and the
 * host half: detokenise each lane into outs */
static int pass_decode(cuda_engine *e, int p, asr_step_out *outs) {
    const gpu_model_dims &dm = e->dm;
    const pass_rec &ps = e->passes[(size_t)p];
    const int B = (int)ps.lanes.size();
    double hp = hp_now(e);
    CK(e, "dec begin", k_dec_begin(e->ar, e->d_rows, B, e->stream));
    const int H = dm.Hdec;
    const int iter_cap = e->pack.qmax * (dm.max_symbols + 1) + 2;
    for (int it = 0; it < iter_cap; it++) {
        prof_mark(e, PS_DEC_SYNC);
        CK(e, "dec compact", k_dec_compact(e->ar, e->d_rows, B, e->d_active, e->d_n_active, e->stream));
        CK(e, "d2h n_active", cudaMemcpyAsync(e->h_n_active, e->d_n_active, sizeof(int), cudaMemcpyDeviceToHost, e->stream));
        hp = hp_lap(e, ASR_HP_DEC_LAUNCH, hp);
        CK(e, "sync", cudaStreamSynchronize(e->stream));
        hp = hp_lap(e, it == 0 ? ASR_HP_ENC_WAIT : ASR_HP_DEC_WAIT, hp);
        if (e->hprof) e->st.hprof_syncs++;
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
    hp = hp_lap(e, ASR_HP_DEC_LAUNCH, hp);
    CK(e, "sync", cudaStreamSynchronize(e->stream));
    hp = hp_lap(e, ASR_HP_FINAL_WAIT, hp);
    if (e->hprof) { e->st.hprof_syncs++; e->st.hprof_passes++; }
    prof_collect(e);
    e->st.d2h_bytes += (double)e->cap * e->ar.tok_cap * sizeof(int) + (double)e->cap * sizeof(gpu_slot_meta);
    e->st.rows += (unsigned long)ps.R;
    e->st.lanes += (unsigned long)B;

    for (int a = 0; a < B; a++) {
        const pass_lane &l = ps.lanes[(size_t)a];
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
        if (text && strlen(text) > before) {
            o.text = text + before;
            o.t0 = s.emitted_t1; o.t1 = l.t1;
            s.chars_emitted = strlen(text);
            s.emitted_t1 = l.t1;
        } else {
            o.text = ""; o.t0 = s.emitted_t1; o.t1 = l.t1;
        }
        if (l.fin) o.finished = 1;
    }
    (void)hp_lap(e, ASR_HP_DETOK, hp);
    return 0;
}

static int cuda_step_submit(cuda_engine *e, const asr_step_req *reqs, int n, asr_step_out *outs) {
    if (e->dead) return -1;
    if (e->inflight) {
        snprintf(e->err, sizeof(e->err), "step_submit with a step in flight (a caller bug)");
        e->dead = 1; e->st.errors++;
        return -1;
    }
    const double t0 = now_s();
    /* --profile-host: the phase total before this step; finish charges the
     * step's untimed remainder to pass_build so the phases sum to submit +
     * finish (the caller's work between them is not the step's) */
    if (e->hprof) { e->hp_in = 0.0; for (int k = 0; k < ASR_HPROF_PHASES; k++) e->hp_in += e->st.hprof_us[k]; }
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
    double hp = hp_lap(e, ASR_HP_RESET, t0);
    /* decide, per request, what this step consumes, and split into passes */
    e->n_passes = 0;
    e->passes[0].lanes.clear();
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
        pass_lane l = {i, slot, 0, s.first, 0, 0, 0.0, 0};
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
        /* would this lane overflow the pass budget? then it opens the next pass */
        std::vector<pass_lane> &cur = e->passes[(size_t)e->n_passes].lanes;
        int fits = (int)cur.size() < e->Bmax && R + l.q <= e->Rmax && M + l.n_mel <= e->Mmax;
        for (int st = 0; st < GPU_SS_STAGES && fits; st++)
            if (P[st] + to[st] * e->dm.Fo[st] > e->Pmax[st]) fits = 0;
        if (!fits) {
            if (e->n_passes + 2 > e->npass_max) {
                snprintf(e->err, sizeof(e->err), "a step needs more than %d passes (a budget bug)", e->npass_max - 1);
                e->dead = 1; e->st.errors++;
                return -1;
            }
            e->n_passes++;
            e->passes[(size_t)e->n_passes].lanes.clear();
            R = M = 0; P[0] = P[1] = P[2] = 0;
        }
        e->passes[(size_t)e->n_passes].lanes.push_back(l);
        R += l.q; M += l.n_mel;
        for (int st = 0; st < GPU_SS_STAGES; st++) P[st] += to[st] * e->dm.Fo[st];
    }
    if (!e->passes[(size_t)e->n_passes].lanes.empty()) e->n_passes++;
    for (int p = 0; p < e->n_passes; p++) pass_stage(e, p, reqs);
    hp = hp_lap(e, ASR_HP_PASS_BUILD, hp);
    if (e->n_passes > 0 && pass_launch(e, 0) != 0) return -1;
    (void)hp_lap(e, ASR_HP_ENQUEUE, hp);
    e->inflight = 1;
    e->submit_ms = (now_s() - t0) * 1e3;
    return 0;
}

static int cuda_step_finish(cuda_engine *e, asr_step_out *outs) {
    if (e->dead) return -1;
    if (!e->inflight) return 0;
    const double t0 = now_s();
    for (int p = 0; p < e->n_passes; p++) {
        if (p > 0) {
            const double hp = hp_now(e);
            if (pass_launch(e, p) != 0) return -1;
            (void)hp_lap(e, ASR_HP_ENQUEUE, hp);
        }
        if (pass_decode(e, p, outs) != 0) return -1;
    }
    e->inflight = 0;
    const double fin_ms = (now_s() - t0) * 1e3;
    if (e->hprof) {
        /* whatever submit + finish spent outside the timed phases (output
         * init, pass splitting, the loop itself) is pass building: the eight
         * phases then sum to the step wall exactly */
        double hp_out = 0.0;
        for (int k = 0; k < ASR_HPROF_PHASES; k++) hp_out += e->st.hprof_us[k];
        e->st.hprof_us[ASR_HP_PASS_BUILD] += (e->submit_ms + fin_ms) * 1e3 - (hp_out - e->hp_in);
    }
    e->st.steps++;
    e->st.submit_ms_sum += e->submit_ms;
    e->st.finish_ms_sum += fin_ms;
    e->st.step_wall_ms_sum += e->submit_ms + fin_ms;
    return 0;
}

static int cuda_step(cuda_engine *e, const asr_step_req *reqs, int n, asr_step_out *outs) {
    if (cuda_step_submit(e, reqs, n, outs) != 0) return -1;
    return cuda_step_finish(e, outs);
}

/* ------------------------------------------------------------- feed team */

static void team_work(feed_team *t, int who) {
    double ms = 0.0;
    for (;;) {
        const int i = t->next.fetch_add(1);
        if (i >= t->n) break;
        const double t0 = now_s();
        slot_feed_one(t->e, t->slots[i], t->pcm[i], t->ns[i]);
        ms += (now_s() - t0) * 1e3;
    }
    t->ms[(size_t)who] += ms;
}

static void *team_main(void *arg) {
    team_arg *a = (team_arg *)arg;
    feed_team *t = a->t;
    unsigned long seen = 0;
    pthread_mutex_lock(&t->mu);
    for (;;) {
        while (!t->quit && t->gen == seen) pthread_cond_wait(&t->go, &t->mu);
        if (t->quit) break;
        seen = t->gen;
        pthread_mutex_unlock(&t->mu);
        team_work(t, a->who);
        pthread_mutex_lock(&t->mu);
        if (++t->reported == t->helpers) pthread_cond_signal(&t->done);
    }
    pthread_mutex_unlock(&t->mu);
    return nullptr;
}

static feed_team *team_open(cuda_engine *e, int helpers) {
    feed_team *t = new feed_team();
    t->e = e;
    t->helpers = helpers;
    pthread_mutex_init(&t->mu, nullptr);
    pthread_cond_init(&t->go, nullptr);
    pthread_cond_init(&t->done, nullptr);
    t->ms.assign((size_t)helpers + 1, 0.0);
    t->args.resize((size_t)helpers);
    t->th.resize((size_t)helpers);
    for (int i = 0; i < helpers; i++) {
        t->args[(size_t)i] = {t, i + 1};
        if (pthread_create(&t->th[(size_t)i], nullptr, team_main, &t->args[(size_t)i]) != 0) {
            t->helpers = i;                       /* the ones that started */
            t->th.resize((size_t)i);
            break;
        }
    }
    return t;
}

static void team_close(feed_team *t) {
    if (!t) return;
    pthread_mutex_lock(&t->mu);
    t->quit = 1;
    pthread_cond_broadcast(&t->go);
    pthread_mutex_unlock(&t->mu);
    for (auto &th : t->th) pthread_join(th, nullptr);
    pthread_mutex_destroy(&t->mu);
    pthread_cond_destroy(&t->go);
    pthread_cond_destroy(&t->done);
    delete t;
}

/* returns the summed mel milliseconds of every participant */
static double team_run(feed_team *t, int n, const int *slots, const float *const *pcm, const size_t *ns) {
    pthread_mutex_lock(&t->mu);
    t->n = n; t->slots = slots; t->pcm = pcm; t->ns = ns;
    t->next.store(0);
    t->reported = 0;
    for (auto &m : t->ms) m = 0.0;
    t->gen++;
    pthread_cond_broadcast(&t->go);
    pthread_mutex_unlock(&t->mu);
    team_work(t, 0);
    pthread_mutex_lock(&t->mu);
    while (t->reported < t->helpers) pthread_cond_wait(&t->done, &t->mu);
    double sum = 0.0;
    for (double m : t->ms) sum += m;
    pthread_mutex_unlock(&t->mu);
    return sum;
}

static int cuda_slot_feed_batch(cuda_engine *e, int n, const int *slots, const float *const *pcm, const size_t *ns) {
    if (e->dead) return -1;
    for (int i = 0; i < n; i++)
        if (slots[i] < 0 || slots[i] >= e->cap) return -1;
    /* below a few slots the wake-up costs more than the mel */
    if (!e->team || n < 4) {
        for (int i = 0; i < n; i++)
            if (cuda_slot_feed(e, slots[i], pcm[i], ns[i]) != 0) return -1;
        return 0;
    }
    e->st.host_mel_ms += team_run(e->team, n, slots, pcm, ns);
    return 0;
}

/* --------------------------------------------------------- graphs, warm-up */

static int parse_buckets(cuda_engine *e, const char *list, char *err, size_t errcap) {
    const char *spec = list && list[0] ? list : "8,16,32,64,128";
    std::vector<int> v;
    const char *p = spec;
    while (*p) {
        char *end = nullptr;
        const long b = strtol(p, &end, 10);
        if (end == p || b <= 0 || b > 1024) { snprintf(err, errcap, "graph buckets '%s': not a list of lane counts 1..1024", spec); return -1; }
        v.push_back(b < e->Bmax ? (int)b : e->Bmax);
        p = end;
        if (*p == ',') p++;
        else if (*p) { snprintf(err, errcap, "graph buckets '%s': not a comma list", spec); return -1; }
    }
    v.push_back(e->Bmax);                         /* every cohort is covered */
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    e->gb = v;
    return 0;
}

/* One capture per (lane bucket, row bucket), at open, before any client: the
 * row buckets step by the GEMM's M tile up to the lane bucket's largest
 * possible rows (Bk x qmax, at most Rmax). A capture records launches and
 * runs nothing; every workspace already exists (nothing is allocated after
 * open), so a graph's pointers stay valid for the engine's life. */
static int graphs_build(cuda_engine *e, char *err, size_t errcap) {
    const double t0 = now_s();
    for (size_t bi = 0; bi < e->gb.size(); bi++) {
        const int Bk = e->gb[bi];
        const int rcap = Bk * e->pack.qmax < e->Rmax ? Bk * e->pack.qmax : e->Rmax;
        const int nr = (rcap + e->RG - 1) / e->RG;
        e->gfirst.push_back((int)e->gx.size());
        e->gnr.push_back(nr);
        for (int ri = 0; ri < nr; ri++) {
            const int Rk = bucket_rows(e, (int)bi, ri);
            int P1k, P2k;
            bucket_counts(e, Bk, Rk, &P1k, &P2k);
            cudaGraph_t g = nullptr;
            cudaGraphExec_t x = nullptr;
            cudaError_t c = cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeThreadLocal);
            if (c != cudaSuccess) { snprintf(err, errcap, "graph capture begin: %s", cudaGetErrorString(c)); return -1; }
            e->capturing = 1;
            const int rc = enqueue_encoder(e, Bk, Rk, P1k, P2k);
            e->capturing = 0;
            c = cudaStreamEndCapture(e->stream, &g);
            if (rc != 0 || c != cudaSuccess) {
                snprintf(err, errcap, "graph capture B=%d R=%d: %s", Bk, Rk, rc != 0 ? e->err : cudaGetErrorString(c));
                if (g) cudaGraphDestroy(g);
                return -1;
            }
            c = cudaGraphInstantiate(&x, g, 0);
            cudaGraphDestroy(g);
            if (c == cudaSuccess) c = cudaGraphUpload(x, e->stream);
            if (c != cudaSuccess) { snprintf(err, errcap, "graph instantiate B=%d R=%d: %s", Bk, Rk, cudaGetErrorString(c)); return -1; }
            e->gx.push_back(x);
        }
    }
    const cudaError_t c = cudaStreamSynchronize(e->stream);
    if (c != cudaSuccess) { snprintf(err, errcap, "graph upload: %s", cudaGetErrorString(c)); return -1; }
    e->graphs = 1;
    e->capture_ms = (now_s() - t0) * 1e3;
    return 0;
}

/* every slot back to the state open left it in: host mel, buffer, decode,
 * text; device caches and decoder at SOS (the scratch slot included) */
static int slots_reset_all(cuda_engine *e) {
    for (auto &s : e->slots) {
        mynah_asr_mel_stream_reset(&s.mel);
        s.mel_have = 0; s.first = 1; s.lookahead = e->pack.default_right; s.prompt = e->pack.default_prompt;
        s.in_use = 0; s.finished = 0; s.mel_finished = 0; s.samples_fed = 0;
        mynah_asr_detok_reset(&s.detok);
        s.chars_emitted = 0; s.emitted_t1 = 0.0; s.lang[0] = '\0';
        s.tokens.clear();
        s.reset_pending = 0;
    }
    e->pending_resets.clear();
    for (int i = 0; i < e->arena_slots; i++) e->h_resets[i] = i;
    CK(e, "reset h2d", cudaMemcpyAsync(e->d_resets, e->h_resets, (size_t)e->arena_slots * sizeof(int), cudaMemcpyHostToDevice, e->stream));
    CK(e, "reset", k_slots_reset(e->dm, e->ar, e->d_resets, e->arena_slots, e->d_sos_h, e->d_sos_c, e->d_sos_g, e->stream));
    CK(e, "reset sync", cudaStreamSynchronize(e->stream));
    return 0;
}

/* The warm-up: before the first client, traffic shaped like the real one at
 * every lane bucket -- default preset, a first chunk, a steady chunk and a
 * finalising tail per lane, through the same step a client gets (graphs
 * included) -- so no lazily loaded kernel, first graph replay or first-use
 * cost lands on a client. Then every slot is reset to the state open left it
 * in, and the counters are zeroed: the warm-up is not traffic. */
static int warmup_run(cuda_engine *e) {
    const double t0 = now_s();
    std::vector<int> buckets = e->gb;
    if (buckets.empty()) buckets.push_back(e->Bmax);
    std::vector<float> pcm;
    std::vector<asr_step_req> reqs((size_t)e->cap);
    std::vector<asr_step_out> outs((size_t)e->cap);
    unsigned int rng = 12345u;
    for (int Bk : buckets) {
        const int L = Bk < e->cap ? Bk : e->cap;
        for (int i = 0; i < L; i++)
            if (cuda_slot_reset(e, i, nullptr, e->pack.default_right) != 0) return -1;
        /* round 0: first chunks; 1: steady; 2: half a chunk, then finalize */
        for (int round = 0; round < 3; round++) {
            for (int i = 0; i < L; i++) {
                size_t need = cuda_slot_need_samples(e, i);
                if (round == 2) need = need / 2 + 1;
                if (pcm.size() < need) pcm.resize(need);
                for (size_t k = 0; k < need; k++) {
                    rng = rng * 1664525u + 1013904223u;
                    pcm[k] = (float)((int)(rng >> 9) - (1 << 22)) * (0.05f / (float)(1 << 22));
                }
                if (cuda_slot_feed(e, i, pcm.data(), need) != 0) return -1;
                reqs[(size_t)i].slot = i;
                reqs[(size_t)i].finalize = round == 2;
            }
            if (cuda_step(e, reqs.data(), L, outs.data()) != 0) return -1;
        }
        /* drain whatever the finalize left */
        for (int guard = 0; guard < 4; guard++) {
            int left = 0;
            for (int i = 0; i < L; i++) left += !e->slots[(size_t)i].finished;
            if (!left) break;
            if (cuda_step(e, reqs.data(), L, outs.data()) != 0) return -1;
        }
    }
    if (slots_reset_all(e) != 0) return -1;
    e->st = asr_engine_stats();
    e->warmup_ms = (now_s() - t0) * 1e3;
    return 0;
}

/* ------------------------------------------------------------- the ops table */
#define CE(e) ((cuda_engine *)(e))
#define CCE(e) ((const cuda_engine *)(e))
/* The engine is opened on the main thread but stepped on the engine thread
 * and queried from the HTTP threads; the current device is per host thread,
 * so every op that touches the runtime binds the engine's device first
 * (otherwise --device N != 0 launches on device 0). */
static void bind_device(const cuda_engine *e) { cudaSetDevice(e->device); }
static void ops_close(asr_engine *e) { bind_device(CE(e)); cuda_close(CE(e)); }
static void ops_facts(const asr_engine *e, asr_engine_facts *f) { bind_device(CCE(e)); cuda_facts(CCE(e), f); }
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
static int ops_step(asr_engine *e, const asr_step_req *r, int n, asr_step_out *o) { bind_device(CE(e)); return cuda_step(CE(e), r, n, o); }
static int ops_submit(asr_engine *e, const asr_step_req *r, int n, asr_step_out *o) { bind_device(CE(e)); return cuda_step_submit(CE(e), r, n, o); }
static int ops_finish(asr_engine *e, asr_step_out *o) { bind_device(CE(e)); return cuda_step_finish(CE(e), o); }
static int ops_feed_batch(asr_engine *e, int n, const int *s, const float *const *p, const size_t *ns) { return cuda_slot_feed_batch(CE(e), n, s, p, ns); }
static int ops_dead(const asr_engine *e) { return cuda_dead(CCE(e)); }
static const char *ops_error(const asr_engine *e) { return cuda_error(CCE(e)); }
static size_t ops_dispatch(const asr_engine *e, char *b, size_t c) { return cuda_dispatch_map(CCE(e), b, c); }
static const asr_engine_ops CUDA_OPS = {
    ops_close, ops_facts, ops_stats, ops_lang_id, ops_lookahead_ok, ops_slot_reset, ops_slot_need,
    ops_slot_feed, ops_slot_ready, ops_slot_audio, ops_slot_text, ops_slot_lang, ops_step, ops_dead,
    ops_error, ops_dispatch, ops_submit, ops_finish, ops_feed_batch,
};
static const asr_engine_ops *cuda_ops(void) { return &CUDA_OPS; }
