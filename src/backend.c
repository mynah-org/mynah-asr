#include "backend.h"
#include "qmat.h"
#include "sgemm.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

/* THE THREE PROVIDERS, and the one place they are chosen.
 *
 * MYNAH_ASR_ACCELERATE is "the Accelerate framework is linked" — true for
 * every macOS build, because Metal and vForce need it.  It is deliberately NOT
 * the same question as "cblas_sgemm is the f32 provider": keying a fast path
 * on the framework when what it needed was a GEMM (or the other way round) is
 * how a default flip silently switches something off on the production target.
 * MYNAH_ASR_BLAS_ACCELERATE / MYNAH_ASR_BLAS_OPENBLAS are the provider; when
 * neither is defined, src/sgemm.c is, and no cblas symbol is referenced
 * anywhere in the binary. */
#ifdef MYNAH_ASR_ACCELERATE
#include <Accelerate/Accelerate.h>   /* vForce (vvexpf) in every macOS build */
#endif
#if defined(MYNAH_ASR_BLAS_ACCELERATE)
#define MYNAH_ASR_GEMM_CBLAS 1
#define MYNAH_ASR_GEMM_PROVIDER_NAME "accelerate"
#elif defined(MYNAH_ASR_BLAS_OPENBLAS)
#include <cblas.h>
#define MYNAH_ASR_GEMM_CBLAS 1
#define MYNAH_ASR_GEMM_PROVIDER_NAME "openblas"
#else
#define MYNAH_ASR_GEMM_CBLAS 0
#define MYNAH_ASR_GEMM_PROVIDER_NAME "own"
#endif

const char *mynah_asr_gemm_provider(void) { return MYNAH_ASR_GEMM_PROVIDER_NAME; }

/* ------------------------------------------------------- the shape profiler
 *
 * MYNAH_ASR_GEMM_PROFILE=1 makes this seam record, per DISTINCT shape, how
 * many times it was called and how long those calls took, and dump the table
 * at exit.  It exists because "which GEMM provider should ship" is not a
 * question about GEMM in general: it is a question about the twenty-odd shapes
 * THIS model issues, and those differ per family (Nemotron's streaming step,
 * Parakeet's offline pass and Canary's AED decode do not stack the same
 * matrices).  The dump is the input to tests/bench_gemm_shapes, which replays
 * the same table against whichever provider a build linked — so an A/B between
 * `own` and OpenBLAS is one run of the model plus two replays, instead of a
 * day of end-to-end sweeps in which nobody can say WHICH shape moved.
 *
 * The times recorded here are indicative: two clock reads bracket calls that
 * can be under a microsecond.  The MEASUREMENT is the replay; what this table
 * owns is the call COUNTS and the shape set, which carry no timing error.
 *
 * Off by default, and when off it costs one predictable branch on a static. */
#define GP_SLOTS 512
enum { GP_GEMM = 0, GP_GEMV = 1 };
typedef struct {
    int kind, ta, tb, m, n, k;
    unsigned long long calls, ns;
} gp_entry;

static gp_entry g_gp[GP_SLOTS];
static int g_gp_used = 0;          /* distinct shapes recorded            */
static unsigned long long g_gp_lost = 0;  /* calls dropped: table full    */
static pthread_mutex_t g_gp_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_gp_dumped = 0;

static int gp_enabled(void) {
    static int on = -1;            /* benign race: every racer computes the same */
    if (on < 0) {
        const char *e = getenv("MYNAH_ASR_GEMM_PROFILE");
        on = (e && *e && *e != '0') ? 1 : 0;
    }
    return on;
}

static unsigned long long gp_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
}

static int gp_cmp(const void *a, const void *b) {
    const gp_entry *x = (const gp_entry *)a, *y = (const gp_entry *)b;
    if (x->ns != y->ns) return x->ns > y->ns ? -1 : 1;   /* most expensive first */
    return 0;
}

/* One line per shape, in a form tests/bench_gemm_shapes parses back.  Written
 * to stderr like every other banner in this runtime, prefixed with the pid so
 * that a prefork server's workers do not read as one process. */
static void gp_dump(void) {
    pthread_mutex_lock(&g_gp_mu);
    if (g_gp_dumped || g_gp_used == 0) { pthread_mutex_unlock(&g_gp_mu); return; }
    g_gp_dumped = 1;
    gp_entry snap[GP_SLOTS];
    const int n = g_gp_used;
    memcpy(snap, g_gp, (size_t)n * sizeof(snap[0]));
    const unsigned long long lost = g_gp_lost;
    pthread_mutex_unlock(&g_gp_mu);

    qsort(snap, (size_t)n, sizeof(snap[0]), gp_cmp);
    unsigned long long calls = 0, ns = 0;
    for (int i = 0; i < n; i++) { calls += snap[i].calls; ns += snap[i].ns; }
    fprintf(stderr, "[GEMM-PROFILE] v=1 pid=%d provider=%s shapes=%d calls=%llu ns=%llu dropped=%llu\n",
            (int)getpid(), MYNAH_ASR_GEMM_PROVIDER_NAME, n, calls, ns, lost);
    for (int i = 0; i < n; i++) {
        if (snap[i].kind == GP_GEMV)
            fprintf(stderr, "[GEMM-SHAPE] v=1 kind=gemv trans=%d rows=%d cols=%d lda=%d calls=%llu ns=%llu\n",
                    snap[i].ta, snap[i].m, snap[i].n, snap[i].k, snap[i].calls, snap[i].ns);
        else
            fprintf(stderr, "[GEMM-SHAPE] v=1 kind=gemm ta=%d tb=%d m=%d n=%d k=%d calls=%llu ns=%llu\n",
                    snap[i].ta, snap[i].tb, snap[i].m, snap[i].n, snap[i].k,
                    snap[i].calls, snap[i].ns);
    }
    fflush(stderr);
}

static void gp_record(int kind, int ta, int tb, int m, int n, int k,
                      unsigned long long ns) {
    pthread_mutex_lock(&g_gp_mu);
    for (int i = 0; i < g_gp_used; i++) {
        gp_entry *e = &g_gp[i];
        if (e->kind == kind && e->ta == ta && e->tb == tb &&
            e->m == m && e->n == n && e->k == k) {
            e->calls++; e->ns += ns;
            pthread_mutex_unlock(&g_gp_mu);
            return;
        }
    }
    if (g_gp_used >= GP_SLOTS) {
        g_gp_lost++;
        pthread_mutex_unlock(&g_gp_mu);
        return;
    }
    if (g_gp_used == 0) atexit(gp_dump);
    gp_entry *e = &g_gp[g_gp_used++];
    e->kind = kind; e->ta = ta; e->tb = tb; e->m = m; e->n = n; e->k = k;
    e->calls = 1; e->ns = ns;
    pthread_mutex_unlock(&g_gp_mu);
}

/* Dump now instead of at exit.  A prefork worker is killed rather than
 * returning from main, and a profile that only exists for processes that exit
 * cleanly would silently describe the parent alone. */
void mynah_asr_gemm_profile_dump(void) { if (gp_enabled()) gp_dump(); }

#ifdef MYNAH_ASR_METAL
int mynah_asr_metal_available(void);
int mynah_asr_metal_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);
int mynah_asr_metal_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                       float *out, int T, int k);
int mynah_asr_metal_gemm3_wt(const float *x, const float *wa, const float *wb,
                         const float *wc, float *oa, float *ob, float *oc,
                         int T, int n, int k);
#endif
#ifdef MYNAH_ASR_CUDA
int mynah_asr_cuda_available(void);
int mynah_asr_cuda_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);
#endif

static int g_backend = MYNAH_ASR_BACKEND_CPU;

/* below this row threshold the GEMM stays on CPU: the GPU round-trip does not pay */
#define METAL_MIN_T 24

int mynah_asr_set_backend(const char *name) {
    if (name && strcmp(name, "cuda") == 0) {
#ifdef MYNAH_ASR_CUDA
        if (mynah_asr_cuda_available()) {
            g_backend = MYNAH_ASR_BACKEND_CUDA;
            fprintf(stderr, "mynah-asr: CUDA backend active (large GEMMs on GPU, T>=%d)\n",
                    METAL_MIN_T);
            return g_backend;
        }
        fprintf(stderr, "mynah-asr: CUDA requested but no device available -> CPU\n");
#else
        fprintf(stderr, "mynah-asr: CUDA requested but not compiled in (make cuda) -> CPU\n");
#endif
        g_backend = MYNAH_ASR_BACKEND_CPU;
        return g_backend;
    }
    if (name && strcmp(name, "metal") == 0) {
#ifdef MYNAH_ASR_METAL
        if (mynah_asr_metal_available()) {
            g_backend = MYNAH_ASR_BACKEND_METAL;
            fprintf(stderr, "mynah-asr: Metal backend active (large GEMMs on GPU, T>=%d)\n",
                    METAL_MIN_T);
            return g_backend;
        }
        fprintf(stderr, "mynah-asr: Metal requested but no device available -> CPU\n");
#else
        fprintf(stderr, "mynah-asr: Metal requested but not compiled into this build -> CPU\n");
#endif
    }
    g_backend = MYNAH_ASR_BACKEND_CPU;
    return g_backend;
}

int mynah_asr_backend(void) { return g_backend; }

/* The provider call itself, split out so the profiler can bracket it without
 * duplicating the #if: one body, two entry paths. */
static void gemm_f32_call(int trans_a, int trans_b, int m, int n, int k,
                          float alpha, const float *a, int lda,
                          const float *b, int ldb, float beta, float *c, int ldc);
static void gemv_f32_call(int trans, int rows, int cols, float alpha,
                          const float *a, int lda, const float *x,
                          float beta, float *y);

/* The seam.  `own` maps 1:1 onto mynah_asr_sgemm_f32, which follows the same
 * argument contract on purpose, so this is a type conversion and not a
 * translation layer. */
void mynah_asr_gemm_f32(int trans_a, int trans_b, int m, int n, int k,
                        float alpha, const float *a, int lda,
                        const float *b, int ldb, float beta, float *c, int ldc) {
    if (gp_enabled()) {
        const unsigned long long t0 = gp_now_ns();
        gemm_f32_call(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        gp_record(GP_GEMM, trans_a, trans_b, m, n, k, gp_now_ns() - t0);
        return;
    }
    gemm_f32_call(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static void gemm_f32_call(int trans_a, int trans_b, int m, int n, int k,
                          float alpha, const float *a, int lda,
                          const float *b, int ldb, float beta, float *c, int ldc) {
#if MYNAH_ASR_GEMM_CBLAS
    cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans,
                trans_b ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b,
                ldb, beta, c, ldc);
#else
    mynah_asr_sgemm_f32(trans_a, trans_b, (size_t)m, (size_t)n, (size_t)k, alpha,
                        a, (size_t)lda, b, (size_t)ldb, beta, c, (size_t)ldc);
#endif
}

/* GEMV.  On `own` it is written as a ONE-ROW GEMM rather than as a fourth
 * kernel, and the transpose flag decides which family serves it:
 *
 *   trans == 0  y^T = x^T * A^T  -> trans_b, the DOT family: one vectorized
 *               dot per output element over a contiguous row of A.
 *   trans != 0  y^T = x^T * A    -> no transpose, the MATVEC family: A's
 *               columns are contiguous, so it vectorizes across the output.
 *
 * Either way the reduction is over the whole of the contracted dimension in
 * one invocation, so the answer does not move with the pool width. */
void mynah_asr_gemv_f32(int trans, int rows, int cols, float alpha,
                        const float *a, int lda, const float *x,
                        float beta, float *y) {
    if (gp_enabled()) {
        const unsigned long long t0 = gp_now_ns();
        gemv_f32_call(trans, rows, cols, alpha, a, lda, x, beta, y);
        gp_record(GP_GEMV, trans, 0, rows, cols, lda, gp_now_ns() - t0);
        return;
    }
    gemv_f32_call(trans, rows, cols, alpha, a, lda, x, beta, y);
}

static void gemv_f32_call(int trans, int rows, int cols, float alpha,
                          const float *a, int lda, const float *x,
                          float beta, float *y) {
#if MYNAH_ASR_GEMM_CBLAS
    cblas_sgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans, rows, cols,
                alpha, a, lda, x, 1, beta, y, 1);
#else
    if (trans)
        mynah_asr_sgemm_f32(0, 0, 1, (size_t)cols, (size_t)rows, alpha, x,
                            (size_t)rows, a, (size_t)lda, beta, y, (size_t)cols);
    else
        mynah_asr_sgemm_f32(0, 1, 1, (size_t)rows, (size_t)cols, alpha, x,
                            (size_t)cols, a, (size_t)lda, beta, y, (size_t)rows);
#endif
}

void mynah_asr_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k) {
#ifdef MYNAH_ASR_METAL
    if (g_backend == MYNAH_ASR_BACKEND_METAL &&
        mynah_asr_metal_gemm_wt(x, w, out, T, n, k) == 0)
        return;
#endif
#ifdef MYNAH_ASR_CUDA
    if (g_backend == MYNAH_ASR_BACKEND_CUDA && T >= METAL_MIN_T &&
        mynah_asr_cuda_gemm_wt(x, w, out, T, n, k) == 0)
        return;
#endif
    mynah_asr_gemm_f32(0, 1, T, n, k, 1.0f, x, k, w, k, 0.0f, out, n);
}

/* In-place SiLU. On Accelerate: vvexpf (vForce) batches the exp — in the
 * 2026-07-19 profile scalar expf accounted for ~10% of the samples. The clamp at
 * 87 avoids inf (the -ffast-math lesson: inf = UB, see mynah_asr_sigmoid);
 * exp(-x) with the clamp can never overflow in f32. The scalar fallback is
 * unchanged.
 *
 * Gated on MYNAH_ASR_ACCELERATE, the FRAMEWORK, not on the BLAS provider: this
 * is vForce, not cblas, and BLAS=none on macOS must not silently change the
 * activation's arithmetic. Otherwise an accelerate-vs-none transcript diff
 * would have two possible causes and the A/B would prove nothing. */
void mynah_asr_silu_scratch(float *x, size_t n, float *scratch) {
#ifdef MYNAH_ASR_ACCELERATE
    if (n >= 256) {
        /* caller scratch (>= n floats) keeps the streaming step allocation-free;
         * NULL = allocate here, as before */
        float *t = scratch;
        const int owned = t == NULL;
        if (owned) t = malloc(n * sizeof(float));
        if (t) {
            for (size_t i = 0; i < n; i++) {
                const float v = -x[i];
                t[i] = v > 87.0f ? 87.0f : v;
            }
            for (size_t off = 0; off < n; off += (size_t)1 << 30) {
                const int chunk = (int)(n - off > (size_t)1 << 30 ? (size_t)1 << 30 : n - off);
                vvexpf(t + off, t + off, &chunk);
            }
            for (size_t i = 0; i < n; i++) x[i] = x[i] / (1.0f + t[i]);
            if (owned) free(t);
            return;
        }
    }
#else
    (void)scratch;
#endif
    for (size_t i = 0; i < n; i++) x[i] = x[i] * mynah_asr_sigmoid(x[i]);
}

void mynah_asr_silu(float *x, size_t n) { mynah_asr_silu_scratch(x, n, NULL); }

void mynah_asr_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                  float *out, int T, int k, float *scratch) {
#ifdef MYNAH_ASR_METAL
    if (g_backend == MYNAH_ASR_BACKEND_METAL &&
        mynah_asr_metal_ffn_wt(x, w1, n1, w2, n2, out, T, k) == 0)
        return;
#endif
    mynah_asr_gemm_wt(x, w1, scratch, T, n1, k);
    mynah_asr_silu(scratch, (size_t)T * (size_t)n1);
    mynah_asr_gemm_wt(scratch, w2, out, T, n2, n1);
}

void mynah_asr_gemm3_wt(const float *x, const float *wa, const float *wb, const float *wc,
                    float *oa, float *ob, float *oc, int T, int n, int k) {
#ifdef MYNAH_ASR_METAL
    if (g_backend == MYNAH_ASR_BACKEND_METAL &&
        mynah_asr_metal_gemm3_wt(x, wa, wb, wc, oa, ob, oc, T, n, k) == 0)
        return;
#endif
    mynah_asr_gemm_wt(x, wa, oa, T, n, k);
    mynah_asr_gemm_wt(x, wb, ob, T, n, k);
    mynah_asr_gemm_wt(x, wc, oc, T, n, k);
}
