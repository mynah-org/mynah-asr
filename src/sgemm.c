/*
 * mynah_asr_sgemm_f32 — our own f32 GEMM.  See sgemm.h for why it exists and
 * for the shapes this runtime actually issues.
 *
 * ------------------------------------------------------------------ layout
 *
 * Everything here is row-major and follows cblas_sgemm's argument contract
 * exactly, because the point is to be droppable into the seam that used it.
 * op(A) is m x k, op(B) is k x n, C is m x n.
 *
 * -------------------------------------------------------------- determinism
 *
 * THE RESULT DOES NOT DEPEND ON THE THREAD COUNT, by construction, not by
 * tolerance.  Three properties give that, and all three are load-bearing:
 *
 *  1. The reduction over k is NEVER split.  One output element is accumulated
 *     by exactly one micro-kernel invocation, over p = 0..k-1 in order, in a
 *     register.  There is no k-blocking anywhere in this file.
 *  2. The COLUMN blocking depends only on the shape (n, k) and the compiled
 *     ISA — never on mynah_asr_num_threads().  So which column group, and
 *     hence which micro-kernel instantiation, covers a given column is fixed.
 *  3. The ROW blocking is the only axis the thread count touches, and the row
 *     block is always rounded UP to SG_MR.  Every block therefore starts at a
 *     multiple of SG_MR, only the final block can carry a ragged remainder,
 *     and that remainder is m % SG_MR whatever the thread count is.  So a
 *     given row is handled by the 4-row strip or the 1-row strip regardless of
 *     how the work was split.
 *
 * Property 3 is the subtle one.  Without the round-up, splitting m=6 into two
 * blocks of 3 would send rows 0..2 through the 1-row kernel where a single
 * block sent them through the 4-row kernel — two different function
 * instantiations of the same macro, which clang is free to contract
 * differently under -ffast-math.  The self-test checks the property directly,
 * with memcmp.
 *
 * It matters here for a second reason.  ENGINEERING.md §9: a serving change
 * never changes a transcript, and the server runs the same stream step under
 * one worker or eight.  A GEMM whose answer moved with the pool width would
 * make that gate unwinnable.
 *
 * ------------------------------------------------------- numerical contract
 *
 * Against mynah_asr_sgemm_f32_reference() the blocked kernels differ only by
 * FMA contraction: the accumulation order over p is identical (sequential, one
 * accumulator per output element).  They are NOT asserted bit-identical to it,
 * because -ffast-math implies -fassociative-math and clang regroups a
 * multi-factor float product differently in two textually identical code
 * paths; the self-test uses a stated relative bound instead.  The DOT family
 * is a deliberate exception and is looser still: a vector dot sums SG_LANES
 * partial accumulators and then folds them, which is a different association
 * from the reference's single running sum.  Two invocations of the SAME
 * instantiation with different block offsets ARE asserted bit-identical,
 * because that is the same machine code with different arguments, and that
 * assertion is the proof of the determinism claim above.
 *
 * Against Accelerate or OpenBLAS the difference is larger, because those DO
 * reassociate and block over k: replacing them changes this runtime's f32
 * output slightly.  That is a numerical change and is qualified as one —
 * against the oracle's per-stage tolerances, never waved through.  The int8
 * path is unaffected: its per-row integer dot is exact by construction.
 */
#include "sgemm.h"

#include "threads.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ======================================================================
 * ISA abstraction
 *
 * One micro-kernel body, three instruction sets.  The scalar build is not a
 * separate algorithm: SG_LANES == 1 makes a "vector" one float, so the scalar
 * path executes the same loop nest with the same accumulation order.  There is
 * no second formula to keep in sync.
 * ====================================================================== */
#if !defined(MYNAH_ASR_SGEMM_NO_SIMD) && (defined(__ARM_NEON) || defined(__aarch64__))
#include <arm_neon.h>
#define SG_ISA_NAME "neon"
#define SG_LANES 4
typedef float32x4_t sg_vec;
#define sg_zero()         vdupq_n_f32(0.0f)
#define sg_load(p)        vld1q_f32(p)
#define sg_store(p, v)    vst1q_f32((p), (v))
#define sg_dup(x)         vdupq_n_f32(x)
#define sg_mul(a, b)      vmulq_f32((a), (b))
#define sg_fma(acc, a, b) vfmaq_f32((acc), (a), (b))
#define sg_hadd(v)        vaddvq_f32(v)
/* 32 architectural v-registers: half of them may hold accumulators. */
#define SG_ACC_VECS 16

#elif !defined(MYNAH_ASR_SGEMM_NO_SIMD) && defined(__AVX2__)
#include <immintrin.h>
#define SG_ISA_NAME "avx2"
#define SG_LANES 8
typedef __m256 sg_vec;
#define sg_zero()      _mm256_setzero_ps()
#define sg_load(p)     _mm256_loadu_ps(p)
#define sg_store(p, v) _mm256_storeu_ps((p), (v))
#define sg_dup(x)      _mm256_set1_ps(x)
#define sg_mul(a, b)   _mm256_mul_ps((a), (b))
#if defined(__FMA__)
#define sg_fma(acc, a, b) _mm256_fmadd_ps((a), (b), (acc))
#else
#define sg_fma(acc, a, b) _mm256_add_ps((acc), _mm256_mul_ps((a), (b)))
#endif
static inline float sg_hadd_avx2(__m256 v) {
    float lane[8];
    _mm256_storeu_ps(lane, v);
    return lane[0] + lane[1] + lane[2] + lane[3] + lane[4] + lane[5] +
           lane[6] + lane[7];
}
#define sg_hadd(v) sg_hadd_avx2(v)
/* 16 architectural ymm registers: half of them may hold accumulators. */
#define SG_ACC_VECS 8

#else
#define SG_ISA_NAME "scalar"
#define SG_LANES 1
typedef float sg_vec;
#define sg_zero()         0.0f
#define sg_load(p)        (*(p))
#define sg_store(p, v)    (*(p) = (v))
#define sg_dup(x)         (x)
#define sg_mul(a, b)      ((a) * (b))
#define sg_fma(acc, a, b) ((acc) + (a) * (b))
#define sg_hadd(v)        (v)
#define SG_ACC_VECS 16
#endif

/* Micro-kernel rows.  Four independent accumulator chains per column vector is
 * enough to cover the FMA latency on every core this runtime targets, and it
 * keeps the b operand loaded once per four FMAs. */
#define SG_MR 4

/* The narrow/panel boundary, DERIVED (see mynah_asr_sgemm_narrow_max in
 * sgemm.h): the widest n whose whole C row block still fits in the accumulator
 * budget.  NEON 16, AVX2 16, scalar 4. */
#define SG_NV_MAX     (SG_ACC_VECS / SG_MR)
#define SG_NARROW_MAX (SG_NV_MAX * SG_LANES)

/* Panel family: accumulator vectors per column group.  DERIVED, like the
 * narrow boundary, from the accumulator budget: SG_NV_MAX is the widest group
 * whose whole C row block still fits in the register file (NEON 4, AVX2 2,
 * scalar 4), and a narrower group would leave registers idle while paying more
 * passes over op(A).
 *
 * It used to be 2 everywhere, "the conservative end", chosen rather than
 * measured.  Measured (M1, one thread, the attention-context shapes of
 * tests/bench_gemm_shapes --demo, 26 of them): 2 -> 48.9 GF/s mean, 4 -> 73.0,
 * the total time of those shapes down 34%.  A development signal, and one that
 * cannot change a result: the group width decides how many output columns a
 * micro-kernel covers at once, never how an element accumulates, so the answer
 * is byte-identical either way (gated by tiling_identity in tests/test_sgemm.c).
 * The AVX2 value does not move -- there SG_NV_MAX was already 2. */
#define SG_PANEL_NV SG_NV_MAX
#define SG_NR       (SG_PANEL_NV * SG_LANES)

/* op(B) panel budget for the panel family, in floats: the panel is k x nc and
 * we want it to stay in L2 while the rows of op(A) sweep past.  128 KiB leaves
 * room for the A strip and the C block in a 256-512 KiB private L2.  A
 * COST-MODEL estimate, not a measurement; it has never been swept. */
#define SG_PANEL_FLOATS 32768u

/* DOT family register tile: MROWS rows of op(A) against NCOLS rows of B, all
 * MROWS*NCOLS dot products accumulated at once.  The tile is what makes this
 * family fast, and it is NOT a numerics change: each output element still
 * accumulates over k in SG_LANES steps into ONE vector accumulator, folds with
 * the same sg_hadd and finishes with the same scalar tail as sg_dot, so the
 * bytes are identical to computing the dots one at a time.  What the tile buys
 * is (a) MROWS*NCOLS independent FMA chains instead of one, which is the whole
 * story — a single chain is FMA-LATENCY bound and reaches a few percent of the
 * core's throughput — and (b) each loaded vector feeding several FMAs.
 *
 * MROWS*NCOLS must stay within the accumulator budget, with room for the
 * MROWS + NCOLS operand vectors: NEON 4x4 = 16 of 32 registers, AVX2 4x2 = 8
 * of 16, scalar 4x4 in whatever the compiler has. */
#ifndef SG_DOT_MR
#define SG_DOT_MR 4
#endif
#ifndef SG_DOT_NC_TILE
#define SG_DOT_NC_TILE (SG_ACC_VECS / SG_DOT_MR)
#endif

/* Fewer than SG_DOT_MR rows left — which is the WHOLE of a gemv, and a gemv
 * through this family is the RNNT prediction network, once per emitted token.
 * With one row there is nothing to tile against, so the independent chains have
 * to come from the column axis instead: one row against SG_DOT_NC_WIDE rows of
 * B.  Same accumulator budget, same per-element arithmetic. */
#define SG_DOT_NC_WIDE (SG_ACC_VECS / 2)

/* Task granularity for the two families that have no row axis to split.
 * m == 1 parallelises over columns only; the dot family is coarse because each
 * task is n*m calls into sg_dot. */
#define SG_MATVEC_NC 256u
#define SG_DOT_NC    64u

/* Below this much work the blocked path's own bookkeeping — the plan, the job
 * struct, the pool's dispatch check — is a visible fraction of the arithmetic,
 * so the reference is both simpler and faster.  4096 MACs is a 16x16x16 GEMM.
 * Cost-model estimate; the self-test only depends on it through the coverage
 * refusal, never on its exact value. */
#define SG_MIN_BLOCKED_WORK 4096u

/* Below this much work PER THREAD the GEMM runs as ONE task, inline.
 *
 * Per thread, not in total, because a dispatch does not cost a fixed amount:
 * mynah_asr_parallel_for wakes min(tasks, width) workers and waits for all of
 * them, so the overhead grows with the pool width while the arithmetic per
 * worker shrinks.  A flat threshold therefore gets WORSE the wider the pool,
 * which is the opposite of what it is for.
 *
 * Measured on the M1 dev host (tests/bench_gemm_shapes --demo, the 49
 * representative shapes under 2M MACs -- which is where a streaming step's
 * attention GEMMs live): with the flat 131072 threshold those shapes cost
 * 318 us at MYNAH_ASR_THREADS=1 and 630 us at 8.  Twice as slow for having
 * eight cores.  The crossover on the same host is around 750K MACs at width 8,
 * i.e. about 131072 per thread, which is what the constant already said -- it
 * was just being compared against the wrong side of the multiplication.
 *
 * Still a cost-model estimate in the sense that the dispatch cost is per host;
 * the Linux box re-derives the constant, not the shape of the rule. */
#define SG_PARALLEL_MIN_WORK 131072u

/* ======================================================================
 * Counters
 *
 * What RAN, not what could run.  Relaxed atomics, one increment per GEMM call
 * (order tens per stream step), never per element and never inside a loop.
 * ====================================================================== */
typedef struct {
    atomic_ullong calls;
    atomic_ullong reference;
    atomic_ullong dot;
    atomic_ullong matvec;
    atomic_ullong narrow;
    atomic_ullong panel;
    atomic_ullong refused;
} sg_counters;

static sg_counters g_sg;

static void sg_bump(atomic_ullong *c) {
    atomic_fetch_add_explicit(c, 1ull, memory_order_relaxed);
}

static unsigned long long sg_read(atomic_ullong *c) {
    return atomic_load_explicit(c, memory_order_relaxed);
}

void mynah_asr_sgemm_stats_get(mynah_asr_sgemm_stats *out) {
    if (out == NULL) return;
    out->calls     = sg_read(&g_sg.calls);
    out->reference = sg_read(&g_sg.reference);
    out->dot       = sg_read(&g_sg.dot);
    out->matvec    = sg_read(&g_sg.matvec);
    out->narrow    = sg_read(&g_sg.narrow);
    out->panel     = sg_read(&g_sg.panel);
    out->refused   = sg_read(&g_sg.refused);
}

void mynah_asr_sgemm_stats_reset(void) {
    atomic_store_explicit(&g_sg.calls, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.reference, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.dot, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.matvec, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.narrow, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.panel, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.refused, 0ull, memory_order_relaxed);
}

/* ======================================================================
 * Shape histogram — what ACTUALLY ran, per shape
 *
 * sgemm.h states the shapes this runtime issues, read out of the call sites.
 * A scaling investigation needs the histogram AS EXECUTED, with per-shape wall
 * time and the task count each call asked the pool for, because "how many
 * dispatches" and "how long is one dispatch" are the two numbers that decide
 * whether a barrier is the ceiling.  Deriving them from the source is exactly
 * the mistake src/dispatch.h exists to prevent.
 *
 * A fixed table, no allocation, linear probe on a 5-tuple key.  OFF unless
 * MYNAH_ASR_SGEMM_PROFILE is set to something other than "0"; when off this
 * costs one relaxed load and a predictable branch per GEMM CALL (not per
 * element, not per task).  A full table stops recording and says so rather
 * than evicting, because a histogram that silently drops its tail is worse
 * than one that refuses.
 * ====================================================================== */
#define SG_HIST_MAX 96

typedef struct {
    atomic_int used;
    int trans_a, trans_b;
    size_t m, n, k;
    unsigned long long calls;
    unsigned long long tasks;     /* summed: tasks asked of the pool */
    unsigned long long ns;        /* summed wall, caller's view      */
    unsigned long long ns_min;
    unsigned long long ns_max;
    int family;
} sg_hist_row;

static sg_hist_row g_sg_hist[SG_HIST_MAX];
static atomic_int g_sg_hist_full;
static atomic_flag g_sg_hist_lock = ATOMIC_FLAG_INIT;
static atomic_int g_sg_prof_state = -1;
static atomic_int g_sg_prof_hooked;

static void sg_hist_report(void);

static int sg_prof_on(void) {
    int state = atomic_load_explicit(&g_sg_prof_state, memory_order_relaxed);
    if (state >= 0) return state;
    const char *env = getenv("MYNAH_ASR_SGEMM_PROFILE");
    state = (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    atomic_store_explicit(&g_sg_prof_state, state, memory_order_relaxed);
    if (state) {
        int expected = 0;
        if (atomic_compare_exchange_strong(&g_sg_prof_hooked, &expected, 1))
            (void)atexit(sg_hist_report);
    }
    return state;
}

static unsigned long long sg_now(void) {
    if (!sg_prof_on()) return 0ull;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ull +
           (unsigned long long)ts.tv_nsec;
}

/* One spinlock, taken once per GEMM call and only while profiling.  The
 * alternative — per-row atomics — would make the min/max racy for no benefit:
 * this path never runs in production. */
static void sg_hist_add(unsigned long long start, int trans_a, int trans_b,
                        size_t m, size_t n, size_t k, int family, size_t tasks) {
    if (start == 0ull) return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const unsigned long long dt =
        (unsigned long long)ts.tv_sec * 1000000000ull +
        (unsigned long long)ts.tv_nsec - start;
    while (atomic_flag_test_and_set_explicit(&g_sg_hist_lock,
                                             memory_order_acquire)) { }
    sg_hist_row *slot = NULL;
    for (int i = 0; i < SG_HIST_MAX; ++i) {
        sg_hist_row *r = &g_sg_hist[i];
        if (!atomic_load_explicit(&r->used, memory_order_relaxed)) {
            slot = r;
            break;
        }
        if (r->trans_a == trans_a && r->trans_b == trans_b && r->m == m &&
            r->n == n && r->k == k) {
            slot = r;
            break;
        }
    }
    if (slot == NULL) {
        atomic_store_explicit(&g_sg_hist_full, 1, memory_order_relaxed);
    } else if (!atomic_load_explicit(&slot->used, memory_order_relaxed)) {
        slot->trans_a = trans_a; slot->trans_b = trans_b;
        slot->m = m; slot->n = n; slot->k = k;
        slot->calls = 1ull; slot->tasks = (unsigned long long)tasks;
        slot->ns = dt; slot->ns_min = dt; slot->ns_max = dt;
        slot->family = family;
        atomic_store_explicit(&slot->used, 1, memory_order_relaxed);
    } else {
        slot->calls += 1ull;
        slot->tasks += (unsigned long long)tasks;
        slot->ns += dt;
        if (dt < slot->ns_min) slot->ns_min = dt;
        if (dt > slot->ns_max) slot->ns_max = dt;
    }
    atomic_flag_clear_explicit(&g_sg_hist_lock, memory_order_release);
}

static void sg_hist_report(void) {
    unsigned long long calls = 0, ns = 0, tasks = 0;
    for (int i = 0; i < SG_HIST_MAX; ++i) {
        if (!atomic_load_explicit(&g_sg_hist[i].used, memory_order_relaxed))
            continue;
        calls += g_sg_hist[i].calls;
        ns    += g_sg_hist[i].ns;
        tasks += g_sg_hist[i].tasks;
    }
    fprintf(stderr,
            "[SGEMM-SHAPES] isa=%s threads=%d narrow_max=%zu -- AS EXECUTED, "
            "not as predicted.\n"
            "  ns is the CALLER's wall for the whole call: wake-up + its own "
            "share + barrier.  tasks/call is what the pool was asked for.\n",
            SG_ISA_NAME, mynah_asr_num_threads(), (size_t)SG_NARROW_MAX);
    if (atomic_load_explicit(&g_sg_hist_full, memory_order_relaxed))
        fprintf(stderr, "  WARNING: table full, some shapes not recorded.\n");
    fprintf(stderr, "  %2s %2s %6s %6s %6s %8s %9s %7s %10s %9s %9s %7s\n",
            "tA", "tB", "m", "n", "k", "calls", "family", "tasks", "total_ms",
            "mean_us", "min_us", "%ns");
    for (int i = 0; i < SG_HIST_MAX; ++i) {
        sg_hist_row *r = &g_sg_hist[i];
        if (!atomic_load_explicit(&r->used, memory_order_relaxed)) continue;
        fprintf(stderr,
                "  %2d %2d %6zu %6zu %6zu %8llu %9s %7.1f %10.3f %9.2f %9.2f "
                "%6.2f%%\n",
                r->trans_a, r->trans_b, r->m, r->n, r->k, r->calls,
                mynah_asr_sgemm_family_name((mynah_asr_sgemm_family)r->family),
                (double)r->tasks / (double)r->calls, (double)r->ns / 1e6,
                (double)r->ns / (double)r->calls / 1e3,
                (double)r->ns_min / 1e3,
                ns ? 100.0 * (double)r->ns / (double)ns : 0.0);
    }
    fprintf(stderr, "  totals: calls=%llu wall=%.3f ms tasks=%llu (%.1f/call)\n",
            calls, (double)ns / 1e6, tasks,
            calls ? (double)tasks / (double)calls : 0.0);
}

/* ======================================================================
 * The reference — the definition of correctness
 *
 * The vectorize(disable) pragma is not decoration: -ffast-math would otherwise
 * let clang reassociate this reduction, and then "the reference" would be one
 * more vectorised kernel rather than the thing the others are judged against.
 * ====================================================================== */
void mynah_asr_sgemm_f32_reference(int trans_a, int trans_b,
                                   size_t m, size_t n, size_t k,
                                   float alpha,
                                   const float *a, size_t lda,
                                   const float *b, size_t ldb,
                                   float beta,
                                   float *c, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#elif defined(__GNUC__)
#pragma GCC novector
#endif
            for (size_t p = 0; p < k; ++p) {
                const float av = trans_a ? a[p * lda + i] : a[i * lda + p];
                const float bv = trans_b ? b[j * ldb + p] : b[p * ldb + j];
                sum += av * bv;
            }
            float *cp = c + i * ldc + j;
            /* beta == 0 must not READ C: cblas guarantees an uninitialised or
             * NaN-carrying C is overwritten. */
            *cp = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * *cp;
        }
    }
}

/* ======================================================================
 * Micro-kernels
 *
 * One body, instantiated for {1, 4} rows x {1, 2, 4} column vectors.  Each
 * invocation owns MROWS * NVEC * SG_LANES output elements, accumulates every
 * one of them over the whole of k in registers, and stores once.
 *
 * `ap` walks op(A): ars is the step between rows, acs the step along k.  With
 * trans_a == 0 that is (lda, 1) and with trans_a == 1 it is (1, lda), so no
 * operand is ever copied or transposed — the transpose is two strides.
 * ====================================================================== */

#define SG_STORE_TILE(MROWS, NVEC, ACC, ALPHA, BETA, C, LDC)                   \
    do {                                                                       \
        const sg_vec sg_a_ = sg_dup(ALPHA);                                    \
        if ((BETA) == 0.0f) {                                                  \
            for (int r_ = 0; r_ < (MROWS); ++r_)                               \
                for (int v_ = 0; v_ < (NVEC); ++v_)                            \
                    sg_store((C) + (size_t)r_ * (LDC) +                        \
                                 (size_t)v_ * SG_LANES,                        \
                             sg_mul((ACC)[r_][v_], sg_a_));                    \
        } else {                                                               \
            const sg_vec sg_b_ = sg_dup(BETA);                                 \
            for (int r_ = 0; r_ < (MROWS); ++r_)                               \
                for (int v_ = 0; v_ < (NVEC); ++v_) {                          \
                    float *cp_ = (C) + (size_t)r_ * (LDC) +                    \
                                 (size_t)v_ * SG_LANES;                        \
                    sg_store(cp_, sg_fma(sg_mul((ACC)[r_][v_], sg_a_),         \
                                         sg_b_, sg_load(cp_)));                \
                }                                                              \
        }                                                                      \
    } while (0)

#define SG_DEFINE_MICRO(NAME, MROWS, NVEC)                                     \
    static void NAME(size_t k, const float *ap, size_t ars, size_t acs,        \
                     const float *bp, size_t ldbp, float alpha, float beta,    \
                     float *c, size_t ldc) {                                   \
        sg_vec acc[MROWS][NVEC];                                               \
        const float *arow[MROWS];                                              \
        for (int r = 0; r < (MROWS); ++r) {                                    \
            arow[r] = ap + (size_t)r * ars;                                    \
            for (int v = 0; v < (NVEC); ++v) acc[r][v] = sg_zero();            \
        }                                                                      \
        for (size_t p = 0; p < k; ++p) {                                       \
            sg_vec bv[NVEC];                                                   \
            for (int v = 0; v < (NVEC); ++v)                                   \
                bv[v] = sg_load(bp + (size_t)v * SG_LANES);                    \
            for (int r = 0; r < (MROWS); ++r) {                                \
                const sg_vec av = sg_dup(*arow[r]);                            \
                arow[r] += acs;                                                \
                for (int v = 0; v < (NVEC); ++v)                               \
                    acc[r][v] = sg_fma(acc[r][v], av, bv[v]);                  \
            }                                                                  \
            bp += ldbp;                                                        \
        }                                                                      \
        SG_STORE_TILE(MROWS, NVEC, acc, alpha, beta, c, ldc);                  \
    }

SG_DEFINE_MICRO(sg_micro4_1, SG_MR, 1)
SG_DEFINE_MICRO(sg_micro4_2, SG_MR, 2)
SG_DEFINE_MICRO(sg_micro4_4, SG_MR, 4)
SG_DEFINE_MICRO(sg_micro1_1, 1, 1)
SG_DEFINE_MICRO(sg_micro1_2, 1, 2)
SG_DEFINE_MICRO(sg_micro1_4, 1, 4)

/* Column remainder: fewer than SG_LANES columns left.  Scalar, and it walks k in
 * the same ORDER as the vector bodies -- but not necessarily with the same
 * ROUNDING, and the difference is worth stating because a comment here used to
 * claim otherwise.  `sg_fma` is an explicitly fused intrinsic; `acc += av * b`
 * below is fused only if the compiler contracts it, which depends on the ISA
 * (AVX2 does not imply FMA) and on the flags (-ffp-contract).  So a column in
 * this remainder can differ in the last bit from the same column computed
 * inside a vector group.  That is allowed: both are valid roundings of the same
 * sum, the choice is deterministic for a given shape and build, and NOTHING in
 * this runtime varies n for a call site.  What must not move is the answer for
 * a row when OTHER rows join it (batching) or when the pool width changes --
 * both gated in tests/test_sgemm.c, neither affected by this.
 * Never reached on the scalar build, where SG_LANES is 1 and there is no
 * remainder. */
static void sg_micro_tail(size_t rows, size_t cols, size_t k, const float *ap,
                          size_t ars, size_t acs, const float *bp, size_t ldbp,
                          float alpha, float beta, float *c, size_t ldc) {
    float acc[SG_MR][SG_LANES];
    for (size_t r = 0; r < rows; ++r)
        for (size_t j = 0; j < cols; ++j) acc[r][j] = 0.0f;
    for (size_t p = 0; p < k; ++p) {
        for (size_t r = 0; r < rows; ++r) {
            const float av = ap[r * ars + p * acs];
            for (size_t j = 0; j < cols; ++j)
                acc[r][j] += av * bp[p * ldbp + j];
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t j = 0; j < cols; ++j) {
            float *cp = c + r * ldc + j;
            *cp = (beta == 0.0f) ? alpha * acc[r][j]
                                 : alpha * acc[r][j] + beta * *cp;
        }
    }
}

/* A strip of MROWS rows across `cols` columns.  The column decomposition is
 * greedy over the available instantiations and depends only on (cols, nv_max),
 * which is why it is thread-count independent. */
#define SG_DEFINE_STRIP(NAME, MROWS, M4, M2, M1)                               \
    static void NAME(size_t k, size_t cols, size_t nv_max, const float *ap,    \
                     size_t ars, size_t acs, const float *bp, size_t ldbp,     \
                     float alpha, float beta, float *c, size_t ldc) {          \
        size_t j = 0;                                                          \
        if (nv_max >= 4u)                                                      \
            for (; j + 4u * SG_LANES <= cols; j += 4u * SG_LANES)              \
                M4(k, ap, ars, acs, bp + j, ldbp, alpha, beta, c + j, ldc);    \
        if (nv_max >= 2u)                                                      \
            for (; j + 2u * SG_LANES <= cols; j += 2u * SG_LANES)              \
                M2(k, ap, ars, acs, bp + j, ldbp, alpha, beta, c + j, ldc);    \
        for (; j + SG_LANES <= cols; j += SG_LANES)                            \
            M1(k, ap, ars, acs, bp + j, ldbp, alpha, beta, c + j, ldc);        \
        if (j < cols)                                                          \
            sg_micro_tail(MROWS, cols - j, k, ap, ars, acs, bp + j, ldbp,      \
                          alpha, beta, c + j, ldc);                            \
    }

SG_DEFINE_STRIP(sg_strip4, SG_MR, sg_micro4_4, sg_micro4_2, sg_micro4_1)
SG_DEFINE_STRIP(sg_strip1, 1, sg_micro1_4, sg_micro1_2, sg_micro1_1)

/* The DOT family's kernel.  mynah-tts calls its kernels.c mynah_dot_f32 here;
 * this repo has no f32 kernel module and one caller, so the dot is expressed
 * in the SAME SG_* macros as the micro-kernels above — one ISA abstraction in
 * this file rather than two spellings of the same reduction.
 *
 * The horizontal fold at the end is the one place in this file where the
 * association differs from the reference, which is why the self-test bounds
 * the DOT family relatively and not bit-exactly.  It does NOT weaken the
 * determinism claim: the fold is inside one invocation, the same one whatever
 * the thread count, and k is never split. */
static float sg_dot(const float *a, const float *b, size_t n) {
    sg_vec acc = sg_zero();
    size_t i = 0;
    for (; i + SG_LANES <= n; i += SG_LANES)
        acc = sg_fma(acc, sg_load(a + i), sg_load(b + i));
    float sum = sg_hadd(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

/* The same reduction, MROWS x NCOLS of them at once.  Element (r, j) walks the
 * identical sequence sg_dot() walks — same SG_LANES stride, same single
 * accumulator, same fold, same scalar tail — so this kernel and sg_dot are
 * interchangeable to the bit.  Only the schedule differs. */
#define SG_DEFINE_DOT_TILE(NAME, MROWS, NCOLS)                                 \
    static void NAME(size_t k, const float *ap, size_t lda, const float *bp,   \
                     size_t ldb, float alpha, float beta, float *c,            \
                     size_t ldc) {                                             \
        sg_vec acc[MROWS][NCOLS];                                              \
        for (int r = 0; r < (MROWS); ++r)                                      \
            for (int j = 0; j < (NCOLS); ++j) acc[r][j] = sg_zero();           \
        size_t p = 0;                                                          \
        for (; p + SG_LANES <= k; p += SG_LANES) {                             \
            sg_vec av[MROWS], bv[NCOLS];                                       \
            for (int r = 0; r < (MROWS); ++r)                                  \
                av[r] = sg_load(ap + (size_t)r * lda + p);                     \
            for (int j = 0; j < (NCOLS); ++j)                                  \
                bv[j] = sg_load(bp + (size_t)j * ldb + p);                     \
            for (int r = 0; r < (MROWS); ++r)                                  \
                for (int j = 0; j < (NCOLS); ++j)                              \
                    acc[r][j] = sg_fma(acc[r][j], av[r], bv[j]);               \
        }                                                                      \
        for (int r = 0; r < (MROWS); ++r)                                      \
            for (int j = 0; j < (NCOLS); ++j) {                                \
                float sum = sg_hadd(acc[r][j]);                                \
                const float *arow = ap + (size_t)r * lda;                      \
                const float *brow = bp + (size_t)j * ldb;                      \
                for (size_t q = p; q < k; ++q) sum += arow[q] * brow[q];       \
                float *cp = c + (size_t)r * ldc + j;                           \
                *cp = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * *cp; \
            }                                                                  \
    }

SG_DEFINE_DOT_TILE(sg_dot_tile, SG_DOT_MR, SG_DOT_NC_TILE)
SG_DEFINE_DOT_TILE(sg_dot_row, 1, SG_DOT_NC_WIDE)

/* ======================================================================
 * The job and its tiles
 * ====================================================================== */
typedef struct {
    int trans_a;
    int trans_b;
    size_t m, n, k;
    float alpha, beta;
    const float *a;
    size_t lda;
    const float *b;
    size_t ldb;
    float *c;
    size_t ldc;
    /* plan */
    mynah_asr_sgemm_family family;
    size_t nv_max;    /* accumulator vectors per column group */
    size_t nc;        /* columns per task                     */
    size_t row_block; /* rows per task, a multiple of SG_MR    */
    size_t grid_n;    /* column blocks                        */
    size_t grid_m;    /* row blocks                           */
    int serial;       /* the plan asked for one task: run inline */
} sg_job;

/* op(B) not transposed: columns of op(B) are contiguous, so the micro-kernels
 * vectorise across n and no operand is copied. */
static void sg_tile_nn(const sg_job *j, size_t i0, size_t rows, size_t j0,
                       size_t cols) {
    const size_t ars = j->trans_a ? 1u : j->lda;
    const size_t acs = j->trans_a ? j->lda : 1u;
    const float *abase = j->a + (j->trans_a ? i0 : i0 * j->lda);
    const float *bbase = j->b + j0;
    float *cbase = j->c + i0 * j->ldc + j0;

    size_t i = 0;
    for (; i + SG_MR <= rows; i += SG_MR)
        sg_strip4(j->k, cols, j->nv_max, abase + i * ars, ars, acs, bbase,
                  j->ldb, j->alpha, j->beta, cbase + i * j->ldc, j->ldc);
    for (; i < rows; ++i)
        sg_strip1(j->k, cols, j->nv_max, abase + i * ars, ars, acs, bbase,
                  j->ldb, j->alpha, j->beta, cbase + i * j->ldc, j->ldc);
}

/* op(B) transposed: op(B)'s column j is B's ROW j, contiguous in k.  There is
 * nothing to vectorise across n, so each output element is one dot product.
 * Only reached with trans_a == 0, where op(A)'s rows are contiguous too; the
 * transposed-both case has no contiguous axis at all and the predicate sends
 * it to the reference. */
static void sg_tile_dot(const sg_job *j, size_t i0, size_t rows, size_t j0,
                        size_t cols) {
    const float *abase = j->a + i0 * j->lda;
    const float *bbase = j->b + j0 * j->ldb;
    float *cbase = j->c + i0 * j->ldc + j0;

    size_t i = 0;
    for (; i + SG_DOT_MR <= rows; i += SG_DOT_MR) {
        size_t jj = 0;
        for (; jj + SG_DOT_NC_TILE <= cols; jj += SG_DOT_NC_TILE)
            sg_dot_tile(j->k, abase + i * j->lda, j->lda, bbase + jj * j->ldb,
                        j->ldb, j->alpha, j->beta, cbase + i * j->ldc + jj,
                        j->ldc);
        /* column remainder: sg_dot computes exactly what the tile would have */
        for (; jj < cols; ++jj)
            for (size_t r = 0; r < SG_DOT_MR; ++r) {
                const float s = sg_dot(abase + (i + r) * j->lda,
                                       bbase + jj * j->ldb, j->k);
                float *cp = cbase + (i + r) * j->ldc + jj;
                *cp = (j->beta == 0.0f) ? j->alpha * s
                                        : j->alpha * s + j->beta * *cp;
            }
    }
    for (; i < rows; ++i) {
        size_t jj = 0;
        for (; jj + SG_DOT_NC_WIDE <= cols; jj += SG_DOT_NC_WIDE)
            sg_dot_row(j->k, abase + i * j->lda, j->lda, bbase + jj * j->ldb,
                       j->ldb, j->alpha, j->beta, cbase + i * j->ldc + jj, j->ldc);
        for (; jj < cols; ++jj) {
            const float s = sg_dot(abase + i * j->lda, bbase + jj * j->ldb, j->k);
            float *cp = cbase + i * j->ldc + jj;
            *cp = (j->beta == 0.0f) ? j->alpha * s : j->alpha * s + j->beta * *cp;
        }
    }
}

static void sg_task(void *ctx, int index) {
    const sg_job *j = (const sg_job *)ctx;
    const size_t bi = (size_t)index / j->grid_n;
    const size_t bj = (size_t)index % j->grid_n;
    const size_t i0 = bi * j->row_block;
    const size_t j0 = bj * j->nc;
    if (i0 >= j->m || j0 >= j->n) return;
    size_t rows = j->row_block;
    if (i0 + rows > j->m) rows = j->m - i0;
    size_t cols = j->nc;
    if (j0 + cols > j->n) cols = j->n - j0;
    if (j->family == MYNAH_ASR_SGEMM_FAMILY_DOT) sg_tile_dot(j, i0, rows, j0, cols);
    else sg_tile_nn(j, i0, rows, j0, cols);
}

/* ======================================================================
 * The predicate
 *
 * One implementation, called by the runtime AND by the dispatch report, so the
 * two cannot disagree.  src/dispatch.h's central rule: a report that
 * recomputes the condition can agree with the source and both be wrong.
 * ====================================================================== */
static int sg_work(size_t m, size_t n, size_t k, size_t *out) {
    if (m != 0 && n > (size_t)-1 / m) return -1;
    size_t mn = m * n;
    if (mn != 0 && k > (size_t)-1 / mn) return -1;
    *out = mn * k;
    return 0;
}

mynah_asr_sgemm_family mynah_asr_sgemm_family_for(int trans_a, int trans_b,
                                                  size_t m, size_t n, size_t k,
                                                  const char **why) {
    const char *reason = "";
    mynah_asr_sgemm_family family = MYNAH_ASR_SGEMM_FAMILY_REFERENCE;
    size_t work = 0;
    const int huge = sg_work(m, n, k, &work) != 0;

    if (m == 0u || n == 0u || k == 0u) {
        reason = "degenerate shape: nothing to accumulate, the reference "
                 "applies beta and returns";
    } else if (!huge && work < SG_MIN_BLOCKED_WORK) {
        reason = "below SG_MIN_BLOCKED_WORK (4096 MACs): the plan and the pool "
                 "check cost more than the arithmetic";
    } else if (trans_a && trans_b) {
        reason = "op(A) and op(B) both transposed: neither operand has a "
                 "contiguous axis a kernel here can use. Not reached by any "
                 "call site in this repo";
    } else if (trans_b) {
        family = MYNAH_ASR_SGEMM_FAMILY_DOT;
        reason = "op(B) transposed: its columns are B's rows, contiguous in k, "
                 "so each output element is one dot over the whole of k. This "
                 "is x @ W^T -- every f32 linear, the joint head and the "
                 "attention scores -- and it is why stacking B streams' rows is "
                 "row-stable by construction here";
    } else if (m == 1u) {
        family = MYNAH_ASR_SGEMM_FAMILY_MATVEC;
        reason = "m == 1: one output row, no row axis to block. The LSTM "
                 "pred-net and the VAD gemv land here";
    } else if (n <= SG_NARROW_MAX) {
        family = MYNAH_ASR_SGEMM_FAMILY_NARROW;
        reason = "n fits the accumulator budget, so the whole C row block stays "
                 "in registers and op(A) is streamed once";
    } else {
        family = MYNAH_ASR_SGEMM_FAMILY_PANEL;
        reason = "n exceeds the accumulator budget: ordinary panel GEMM, column "
                 "panels sized to keep the op(B) panel in L2";
    }
    if (why != NULL) *why = reason;
    return family;
}

const char *mynah_asr_sgemm_family_name(mynah_asr_sgemm_family family) {
    switch (family) {
    case MYNAH_ASR_SGEMM_FAMILY_REFERENCE: return "reference";
    case MYNAH_ASR_SGEMM_FAMILY_DOT:       return "dot";
    case MYNAH_ASR_SGEMM_FAMILY_MATVEC:    return "matvec";
    case MYNAH_ASR_SGEMM_FAMILY_NARROW:    return "narrow";
    case MYNAH_ASR_SGEMM_FAMILY_PANEL:     return "panel";
    default:                               return "?";
    }
}

const char *mynah_asr_sgemm_isa_name(void) { return SG_ISA_NAME; }

size_t mynah_asr_sgemm_narrow_max(void) { return (size_t)SG_NARROW_MAX; }

/* ======================================================================
 * Planning and dispatch
 * ====================================================================== */

/* Column blocking: a function of the shape and the compiled ISA only.  If this
 * ever learns about mynah_asr_num_threads() the determinism argument above is
 * void. */
static void sg_plan_columns(sg_job *j) {
    switch (j->family) {
    case MYNAH_ASR_SGEMM_FAMILY_NARROW:
        j->nv_max = SG_NV_MAX;
        j->nc = j->n; /* <= SG_NARROW_MAX by the predicate: one group */
        break;
    case MYNAH_ASR_SGEMM_FAMILY_MATVEC:
        j->nv_max = SG_NV_MAX;
        j->nc = SG_MATVEC_NC;
        break;
    case MYNAH_ASR_SGEMM_FAMILY_DOT:
        j->nv_max = 0u; /* unused: sg_tile_dot has no column groups */
        j->nc = SG_DOT_NC;
        break;
    case MYNAH_ASR_SGEMM_FAMILY_PANEL:
    default: {
        j->nv_max = SG_PANEL_NV;
        size_t nc = (j->k != 0u) ? SG_PANEL_FLOATS / j->k : j->n;
        nc -= nc % SG_NR;
        if (nc < SG_NR) nc = SG_NR;
        j->nc = nc;
        break;
    }
    }
    if (j->nc == 0u || j->nc > j->n) j->nc = j->n;
    j->grid_n = (j->n + j->nc - 1u) / j->nc;
}

/* Row blocking: the ONLY axis the thread count touches, and always rounded up
 * to SG_MR so which strip covers a row never moves.  `force_tasks` is the
 * self-test's handle on this; 0 means "ask the pool". */
static void sg_plan_rows(sg_job *j, size_t force_tasks) {
    size_t work = 0;
    const int huge = sg_work(j->m, j->n, j->k, &work) != 0;
    size_t want = 1u;
    if (force_tasks > 0u) {
        want = force_tasks;
    } else {
        /* How many tasks the WORK can carry, not how many the pool would like:
         * every task must be worth at least SG_PARALLEL_MIN_WORK, because below
         * that its share of the dispatch costs more than its arithmetic.  The
         * count then degrades smoothly -- a GEMM worth three tasks gets three,
         * on a four-core box and on a thirty-two-core one alike -- instead of
         * falling off a cliff at some multiple of the pool width.  A flat
         * threshold was wrong in the other direction (it got worse the wider
         * the pool); `threads * MIN_WORK` fixed the pathology but would have
         * left medium GEMMs single-threaded on a 32-core Neoverse, which is the
         * box this is for. */
        const int threads = mynah_asr_num_threads();
        const size_t cap = (size_t)threads * 2u;
        if (threads > 1) {
            size_t by_work = huge ? cap : work / SG_PARALLEL_MIN_WORK;
            if (by_work > cap) by_work = cap;
            if (by_work > 1u) want = by_work;
        }
    }
    /* `want == 1` means the work cannot fill even two tasks and this GEMM
     * should not touch the pool at all.  That intent is easy to lose: the
     * COLUMN grid is planned from the shape alone, so a matvec with nc = 256
     * over n = 1920 still produces grid_n = 8 tasks.  Running the same tasks
     * inline, in order, is bit-identical (same task code, disjoint outputs) and
     * is what `want == 1` always meant. */
    j->serial = (want == 1u);

    size_t grid_m = (want + j->grid_n - 1u) / j->grid_n;
    if (grid_m == 0u) grid_m = 1u;
    const size_t max_row_blocks = (j->m + SG_MR - 1u) / SG_MR;
    if (grid_m > max_row_blocks) grid_m = max_row_blocks;
    if (grid_m == 0u) grid_m = 1u;

    size_t row_block = (j->m + grid_m - 1u) / grid_m;
    row_block += (SG_MR - row_block % SG_MR) % SG_MR; /* round up to SG_MR */
    if (row_block == 0u) row_block = SG_MR;
    j->row_block = row_block;
    j->grid_m = (j->m + row_block - 1u) / row_block;
}

static int sg_validate(int trans_a, int trans_b, size_t m, size_t n, size_t k,
                       const float *a, size_t lda, const float *b, size_t ldb,
                       const float *c, size_t ldc) {
    if (m == 0u || n == 0u) return 0; /* no output elements: nothing to check */
    if (c == NULL || ldc < n) return -1;
    if (k == 0u) return 0;            /* a and b are not dereferenced */
    if (a == NULL || b == NULL) return -1;
    if (trans_a ? (lda < m) : (lda < k)) return -1;
    if (trans_b ? (ldb < k) : (ldb < n)) return -1;
    return 0;
}

static void sg_count(mynah_asr_sgemm_family family) {
    switch (family) {
    case MYNAH_ASR_SGEMM_FAMILY_REFERENCE: sg_bump(&g_sg.reference); break;
    case MYNAH_ASR_SGEMM_FAMILY_DOT:       sg_bump(&g_sg.dot);       break;
    case MYNAH_ASR_SGEMM_FAMILY_MATVEC:    sg_bump(&g_sg.matvec);    break;
    case MYNAH_ASR_SGEMM_FAMILY_NARROW:    sg_bump(&g_sg.narrow);    break;
    case MYNAH_ASR_SGEMM_FAMILY_PANEL:     sg_bump(&g_sg.panel);     break;
    default: break;
    }
}

/* Can this family serve this operand layout at all?  A forced family that
 * silently degrades would make the self-test's comparison vacuous, so the
 * degradation is reported through `ran` rather than hidden. */
static int sg_family_fits(mynah_asr_sgemm_family family, int trans_a,
                          int trans_b, size_t m, size_t n, size_t k) {
    if (m == 0u || n == 0u || k == 0u)
        return family == MYNAH_ASR_SGEMM_FAMILY_REFERENCE;
    switch (family) {
    case MYNAH_ASR_SGEMM_FAMILY_REFERENCE: return 1;
    case MYNAH_ASR_SGEMM_FAMILY_DOT:       return trans_b != 0 && trans_a == 0;
    case MYNAH_ASR_SGEMM_FAMILY_MATVEC:    return trans_b == 0 && m == 1u;
    case MYNAH_ASR_SGEMM_FAMILY_NARROW:    return trans_b == 0 && n <= SG_NARROW_MAX;
    case MYNAH_ASR_SGEMM_FAMILY_PANEL:     return trans_b == 0;
    default:                               return 0;
    }
}

static int sg_dispatch(mynah_asr_sgemm_family want, int forced,
                       size_t force_tasks, mynah_asr_sgemm_family *ran,
                       int trans_a, int trans_b, size_t m, size_t n, size_t k,
                       float alpha, const float *a, size_t lda, const float *b,
                       size_t ldb, float beta, float *c, size_t ldc) {
    if (ran != NULL) *ran = MYNAH_ASR_SGEMM_FAMILY_REFERENCE;
    const unsigned long long t_prof = sg_now();
    sg_bump(&g_sg.calls);
    if (sg_validate(trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc) != 0) {
        sg_bump(&g_sg.refused);
        return -1;
    }
    if (m == 0u || n == 0u) return 0;

    mynah_asr_sgemm_family family =
        forced ? want
               : mynah_asr_sgemm_family_for(trans_a, trans_b, m, n, k, NULL);
    if (!sg_family_fits(family, trans_a, trans_b, m, n, k))
        family = MYNAH_ASR_SGEMM_FAMILY_REFERENCE;
    if (ran != NULL) *ran = family;
    sg_count(family);

    if (family == MYNAH_ASR_SGEMM_FAMILY_REFERENCE) {
        mynah_asr_sgemm_f32_reference(trans_a, trans_b, m, n, k, alpha, a, lda,
                                      b, ldb, beta, c, ldc);
        sg_hist_add(t_prof, trans_a, trans_b, m, n, k, (int)family, 1u);
        return 0;
    }

    sg_job job;
    memset(&job, 0, sizeof job);
    job.trans_a = trans_a;
    job.trans_b = trans_b;
    job.m = m; job.n = n; job.k = k;
    job.alpha = alpha; job.beta = beta;
    job.a = a; job.lda = lda;
    job.b = b; job.ldb = ldb;
    job.c = c; job.ldc = ldc;
    job.family = family;
    sg_plan_columns(&job);
    sg_plan_rows(&job, force_tasks);

    /* A clamped task count would leave output blocks uncomputed, so an absurd
     * grid collapses to a single tile instead of being truncated.  Unreachable
     * for any shape this runtime produces; it is here because "clamp and hope"
     * is how a silently wrong result gets shipped. */
    size_t tasks = job.grid_m * job.grid_n;
    if (job.grid_m != 0u && tasks / job.grid_m != job.grid_n) tasks = 0u;
    if (tasks == 0u || tasks > (size_t)INT_MAX) {
        job.nc = job.n;
        job.grid_n = 1u;
        job.row_block = job.m + (SG_MR - job.m % SG_MR) % SG_MR;
        job.grid_m = 1u;
        tasks = 1u;
    }
    if (job.serial) {
        for (size_t t = 0; t < tasks; ++t) sg_task(&job, (int)t);
    } else {
        mynah_asr_parallel_for((int)tasks, sg_task, &job);
    }
    sg_hist_add(t_prof, trans_a, trans_b, m, n, k, (int)family, tasks);
    return 0;
}

int mynah_asr_sgemm_f32(int trans_a, int trans_b, size_t m, size_t n, size_t k,
                        float alpha, const float *a, size_t lda, const float *b,
                        size_t ldb, float beta, float *c, size_t ldc) {
    return sg_dispatch(MYNAH_ASR_SGEMM_FAMILY_REFERENCE, 0, 0u, NULL, trans_a,
                       trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int mynah_asr_sgemm_f32_forced(mynah_asr_sgemm_family want,
                               mynah_asr_sgemm_family *ran, int trans_a,
                               int trans_b, size_t m, size_t n, size_t k,
                               float alpha, const float *a, size_t lda,
                               const float *b, size_t ldb, float beta, float *c,
                               size_t ldc) {
    return sg_dispatch(want, 1, 0u, ran, trans_a, trans_b, m, n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

int mynah_asr_sgemm_f32_tasks(size_t tasks, int trans_a, int trans_b, size_t m,
                              size_t n, size_t k, float alpha, const float *a,
                              size_t lda, const float *b, size_t ldb,
                              float beta, float *c, size_t ldc) {
    return sg_dispatch(MYNAH_ASR_SGEMM_FAMILY_REFERENCE, 0, tasks, NULL,
                       trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta,
                       c, ldc);
}

/* ======================================================================
 * Self-test
 *
 * Model-free.  The reference is the oracle; every compiled path is compared
 * against it over the REAL stream-step shapes and over the edge cases the
 * blocking can get wrong.
 *
 * TOLERANCE.  Relative, not bit-identical.  For the blocked families (NARROW,
 * PANEL, MATVEC) the accumulation ORDER over p is the same as the reference's
 * and the residual is FMA contraction at worst.  The DOT family folds SG_LANES
 * partial sums, which is a genuinely different association.
 *
 * MEASURED 2026-09-18, macOS arm64 / NEON, against a DOUBLE-PRECISION oracle
 * computed outside this file, on the widest shape the runtime issues
 * (m = 64, n = 1024, k = 4096, trans_b, values in [-1, 1)):
 *
 *     reference vs f64   6.81e-05        kernel vs f64   3.85e-05
 *     kernel vs reference                                6.55e-05
 *
 * So the 6.55e-05 the self-test reports is NOT the kernel drifting: the kernel
 * is nearly twice as CLOSE to the exact answer as the scalar reference is, and
 * the gap between them is dominated by the reference's own sequential
 * summation over 4096 terms.  On the same shape with trans_b == 0 — the NARROW
 * and PANEL path — the kernel is bit-identical to the reference (deviation
 * exactly 0).  1e-04 is therefore a bound on a KNOWN quantity, roughly 1.5x the
 * observed worst case, and it must not be tightened without re-running that
 * comparison: the number that would trip first is the reference's error, not a
 * bug.  The observed maximum is reported in the error string when the bound is
 * exceeded, and through mynah_asr_sgemm_self_test_worst() when it is not.
 *
 * DETERMINISM.  The task-count sweep IS asserted bit-identical, because there
 * the two arms are the same machine code with different block offsets, not two
 * textually identical source paths compiled twice.
 * ====================================================================== */

#define SG_TEST_TOL 1.0e-4f

static float g_sg_worst = -1.0f;

float mynah_asr_sgemm_self_test_worst(void) { return g_sg_worst; }
float mynah_asr_sgemm_self_test_bound(void) { return SG_TEST_TOL; }

typedef struct {
    size_t m, n, k;
    int trans_a;
    const char *note;
} sg_case;

/* The shapes this runtime issues, read out of the call sites on 2026-09-18
 * (Nemotron 0.6b streaming, d_model 1024, ffn 4096, vocab 13088, dk 64).
 * m is the stacked row count of a batched stream step: 1..64. */
static const sg_case g_measured[] = {
    {1,   1024, 1024, 0, "stream step, B=1: one linear row, x @ W^T"},
    {8,   1024, 1024, 0, "stream step, B=1 chunk of 8 rows"},
    {16,  4096, 1024, 0, "FFN up, one chunk"},
    {16,  1024, 4096, 0, "FFN down, one chunk"},
    {32,  1024, 1024, 0, "batched stream step, B=4"},
    {64,  1024, 1024, 0, "batched stream step, B=8"},
    {64,  4096, 1024, 0, "batched FFN up, B=8"},
    {64,  1024, 4096, 0, "batched FFN down, B=8"},
    {1,   13088, 1024, 0, "joint head / CTC head, one frame"},
    {8,   13088, 1024, 0, "joint head, a blank run of 8 frames"},
    {56,  1024, 64,   0, "attention: context = scores @ v, n = dk"},
    {56,  120,  64,   0, "attention: scores = q @ k^T over the window"},
    {176, 1024, 1029, 0, "subsampling flatten linear, offline chunk"},
};

/* Edge cases: k not a multiple of the unroll, m or n equal to 1, shapes that
 * straddle SG_LANES and SG_MR, and transposed-A pairs. */
static const sg_case g_edges[] = {
    {1,  1,   1,   0, "1x1x1"},
    {1,  17,  13,  0, "single row, ragged n and k"},
    {17, 1,   13,  0, "single column"},
    {17, 13,  1,   0, "k == 1"},
    {5,  7,   11,  0, "every dimension prime"},
    {33, 31,  29,  0, "all three straddle SG_MR and SG_LANES"},
    {64, 16,  63,  0, "narrow family, k not a multiple of 4"},
    {64, 17,  63,  0, "one column past the narrow boundary"},
    {4,  64,  64,  0, "exactly SG_MR rows"},
    {6,  64,  64,  0, "SG_MR + 2 rows: the ragged row tail"},
    {64, 1024, 1023, 0, "the stacked shape with k not a multiple of 4"},
    {70, 130, 70,  1, "transposed A, all ragged"},
    {33, 31,  29,  1, "transposed A, prime"},
};

static void sg_fill(float *p, size_t n, unsigned seed) {
    /* xorshift32: reproducible across platforms and libcs, which a rand()
     * based fixture is not. */
    unsigned s = seed | 1u;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (float)((double)(s >> 8) / 8388608.0 - 1.0); /* [-1, 1) */
    }
}

/* Returns the largest relative deviation, or a negative value on a non-finite
 * result (which no tolerance may excuse). */
static float sg_maxdev(const float *got, const float *want, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(got[i])) return -1.0f;
        const float d = fabsf(got[i] - want[i]);
        const float scale = 1.0f + fabsf(want[i]);
        const float rel = d / scale;
        if (rel > worst) worst = rel;
    }
    return worst;
}

typedef struct {
    float *a, *b, *c, *ref;
    size_t cap_a, cap_b, cap_c;
    unsigned long long ran[5];
    float worst;
} sg_test_ctx;

static int sg_case_run(sg_test_ctx *t, const sg_case *sc, int trans_b,
                       float alpha, float beta, char *error, size_t cap) {
    const size_t m = sc->m, n = sc->n, k = sc->k;
    const size_t na = m * k, nb = n * k, nc = m * n;
    if (na > t->cap_a || nb > t->cap_b || nc > t->cap_c) {
        snprintf(error, cap, "sgemm self-test fixture too small for %zux%zux%zu",
                 m, n, k);
        return -1;
    }
    const size_t lda = sc->trans_a ? m : k;
    const size_t ldb = trans_b ? k : n;

    sg_fill(t->a, na, (unsigned)(m * 7919u + k));
    sg_fill(t->b, nb, (unsigned)(n * 104729u + k * 31u));
    sg_fill(t->c, nc, 0xC0FFEEu);
    memcpy(t->ref, t->c, nc * sizeof(float));
    mynah_asr_sgemm_f32_reference(sc->trans_a, trans_b, m, n, k, alpha, t->a,
                                  lda, t->b, ldb, beta, t->ref, n);

    /* The predicate's own choice, plus every family forced.  A family that
     * cannot serve this layout reports REFERENCE through `ran` and is not
     * counted as coverage, so a vacuous comparison cannot look like a pass. */
    for (int f = -1; f <= (int)MYNAH_ASR_SGEMM_FAMILY_PANEL; ++f) {
        /* Asked BEFORE the call, not after: a family that cannot serve this
         * layout degrades to the reference inside sg_dispatch, and comparing
         * the reference against itself is both vacuous and, on the widest
         * shapes, the single most expensive thing this test could do. */
        if (f == (int)MYNAH_ASR_SGEMM_FAMILY_REFERENCE)
            continue; /* the oracle: comparing it with itself proves nothing.
                       * Its coverage comes from the degenerate and below-
                       * threshold cases, where the dispatcher picks it. */
        if (f >= 0 && !sg_family_fits((mynah_asr_sgemm_family)f, sc->trans_a,
                                      trans_b, m, n, k))
            continue;
        float *out = t->c;
        /* Restore C: beta != 0 reads it, so every arm must start from the same
         * bytes the reference started from. */
        sg_fill(out, nc, 0xC0FFEEu);
        mynah_asr_sgemm_family ran = MYNAH_ASR_SGEMM_FAMILY_REFERENCE;
        int rc;
        if (f < 0) {
            rc = mynah_asr_sgemm_f32(sc->trans_a, trans_b, m, n, k, alpha, t->a,
                                     lda, t->b, ldb, beta, out, n);
            ran = mynah_asr_sgemm_family_for(sc->trans_a, trans_b, m, n, k, NULL);
            if (!sg_family_fits(ran, sc->trans_a, trans_b, m, n, k))
                ran = MYNAH_ASR_SGEMM_FAMILY_REFERENCE;
        } else {
            rc = mynah_asr_sgemm_f32_forced((mynah_asr_sgemm_family)f, &ran,
                                            sc->trans_a, trans_b, m, n, k,
                                            alpha, t->a, lda, t->b, ldb, beta,
                                            out, n);
            if ((int)ran != f) continue; /* layout refused: not coverage */
        }
        if (rc != 0) {
            snprintf(error, cap, "sgemm %s returned -1 on %zux%zux%zu (%s)",
                     f < 0 ? "dispatch"
                           : mynah_asr_sgemm_family_name(
                                 (mynah_asr_sgemm_family)f),
                     m, n, k, sc->note);
            return -1;
        }
        t->ran[(int)ran]++;
        const float dev = sg_maxdev(out, t->ref, nc);
        if (dev < 0.0f) {
            snprintf(error, cap,
                     "sgemm %s produced a non-finite value on %zux%zux%zu (%s)",
                     mynah_asr_sgemm_family_name(ran), m, n, k, sc->note);
            return -1;
        }
        if (dev > t->worst) t->worst = dev;
        if (dev > SG_TEST_TOL) {
            snprintf(error, cap,
                     "sgemm %s deviates %.3g (bound %.3g) on m=%zu n=%zu k=%zu "
                     "transA=%d transB=%d alpha=%g beta=%g (%s)",
                     mynah_asr_sgemm_family_name(ran), (double)dev,
                     (double)SG_TEST_TOL, m, n, k, sc->trans_a, trans_b,
                     (double)alpha, (double)beta, sc->note);
            return -1;
        }
    }
    return 0;
}

/* The determinism proof: the same GEMM planned as one task and as many, byte
 * for byte.  Legitimate as a bit-identity assertion because both arms enter
 * the SAME compiled micro-kernel instantiations — only the block offsets
 * differ — which is exactly the property the SG_MR round-up in sg_plan_rows
 * exists to guarantee. */
static int sg_case_tasks(sg_test_ctx *t, const sg_case *sc, int trans_b,
                         char *error, size_t cap) {
    const size_t m = sc->m, n = sc->n, k = sc->k;
    const size_t nc = m * n;
    const size_t lda = sc->trans_a ? m : k;
    const size_t ldb = trans_b ? k : n;
    sg_fill(t->a, m * k, (unsigned)(m * 7919u + k));
    sg_fill(t->b, n * k, (unsigned)(n * 104729u + k * 31u));

    static const size_t task_counts[] = {1u, 2u, 3u, 5u, 64u};
    for (size_t ti = 0; ti < sizeof task_counts / sizeof task_counts[0]; ++ti) {
        float *out = (ti == 0u) ? t->ref : t->c;
        sg_fill(out, nc, 0x5EEDu);
        if (mynah_asr_sgemm_f32_tasks(task_counts[ti], sc->trans_a, trans_b, m,
                                      n, k, 1.0f, t->a, lda, t->b, ldb, 0.5f,
                                      out, n) != 0) {
            snprintf(error, cap, "sgemm task sweep failed on %zux%zux%zu", m, n, k);
            return -1;
        }
        if (ti == 0u) continue;
        if (memcmp(t->ref, t->c, nc * sizeof(float)) != 0) {
            snprintf(error, cap,
                     "sgemm is not task-count independent: %zu tasks differ "
                     "from 1 task on m=%zu n=%zu k=%zu transB=%d (%s)",
                     task_counts[ti], m, n, k, trans_b, sc->note);
            return -1;
        }
    }
    return 0;
}

/* The same GEMM at 1, 2, 4 and 8 POOL threads, byte for byte.  The task sweep
 * above proves the plan is stable; this proves the plan is the only thing the
 * thread count touches, by moving the real pool width under a fixed shape.
 * MYNAH_ASR_THREADS is read once and cached by mynah_asr_num_threads(), so the
 * width cannot be changed from inside this process — the sweep therefore runs
 * the grid the pool WOULD plan at each width and compares those.  The caller
 * (tests/test_sgemm.c) re-executes the whole self-test at four pinned widths,
 * which is the part that moves the real threads. */
static int sg_case_threadwidths(sg_test_ctx *t, const sg_case *sc, int trans_b,
                                char *error, size_t cap) {
    const size_t m = sc->m, n = sc->n, k = sc->k;
    const size_t nc = m * n;
    const size_t lda = sc->trans_a ? m : k;
    const size_t ldb = trans_b ? k : n;
    sg_fill(t->a, m * k, (unsigned)(m * 7919u + k));
    sg_fill(t->b, n * k, (unsigned)(n * 104729u + k * 31u));

    static const int widths[] = {1, 2, 4, 8};
    for (size_t wi = 0; wi < sizeof widths / sizeof widths[0]; ++wi) {
        float *out = (wi == 0u) ? t->ref : t->c;
        sg_fill(out, nc, 0xBEEFu);
        /* the grid a pool of `widths[wi]` threads would ask for */
        const size_t tasks = (size_t)widths[wi] * 2u;
        if (mynah_asr_sgemm_f32_tasks(tasks, sc->trans_a, trans_b, m, n, k,
                                      1.0f, t->a, lda, t->b, ldb, 0.0f, out,
                                      n) != 0) {
            snprintf(error, cap, "sgemm width sweep failed on %zux%zux%zu", m, n, k);
            return -1;
        }
        if (wi == 0u) continue;
        if (memcmp(t->ref, t->c, nc * sizeof(float)) != 0) {
            snprintf(error, cap,
                     "sgemm is not thread-count independent: the grid for %d "
                     "threads differs from the grid for 1 on m=%zu n=%zu k=%zu "
                     "transB=%d (%s)",
                     widths[wi], m, n, k, trans_b, sc->note);
            return -1;
        }
    }
    return 0;
}

int mynah_asr_sgemm_self_test(char *error, size_t error_capacity) {
    char scratch[256];
    if (error == NULL || error_capacity == 0) {
        error = scratch;
        error_capacity = sizeof scratch;
    }
    error[0] = '\0';

    size_t max_a = 0, max_b = 0, max_c = 0;
    const size_t n_meas = sizeof g_measured / sizeof g_measured[0];
    const size_t n_edge = sizeof g_edges / sizeof g_edges[0];
    for (size_t i = 0; i < n_meas + n_edge; ++i) {
        const sg_case *sc = (i < n_meas) ? &g_measured[i] : &g_edges[i - n_meas];
        if (sc->m * sc->k > max_a) max_a = sc->m * sc->k;
        if (sc->n * sc->k > max_b) max_b = sc->n * sc->k;
        if (sc->m * sc->n > max_c) max_c = sc->m * sc->n;
    }

    sg_test_ctx t;
    memset(&t, 0, sizeof t);
    t.cap_a = max_a; t.cap_b = max_b; t.cap_c = max_c;
    t.a = (float *)malloc(max_a * sizeof(float));
    t.b = (float *)malloc(max_b * sizeof(float));
    t.c = (float *)malloc(max_c * sizeof(float));
    t.ref = (float *)malloc(max_c * sizeof(float));
    if (t.a == NULL || t.b == NULL || t.c == NULL || t.ref == NULL) {
        free(t.a); free(t.b); free(t.c); free(t.ref);
        snprintf(error, error_capacity, "sgemm self-test: out of memory");
        return -1;
    }

    int rc = 0;
    /* The real shapes, in the transpose the call sites use: x @ W^T is
     * trans_b, the attention context and the subsampling pointwise are not.
     * beta 0 (write) and beta 1 (accumulate) both occur. */
    for (size_t i = 0; i < n_meas && rc == 0; ++i) {
        rc = sg_case_run(&t, &g_measured[i], 1, 1.0f, 0.0f, error, error_capacity);
        if (rc == 0)
            rc = sg_case_run(&t, &g_measured[i], 0, 1.0f, 0.0f, error, error_capacity);
        if (rc == 0)
            rc = sg_case_run(&t, &g_measured[i], 0, 1.0f, 1.0f, error, error_capacity);
    }
    /* Edge cases across both transpose flags and a non-unit alpha/beta. */
    for (size_t i = 0; i < n_edge && rc == 0; ++i) {
        rc = sg_case_run(&t, &g_edges[i], 0, 1.0f, 0.0f, error, error_capacity);
        if (rc == 0)
            rc = sg_case_run(&t, &g_edges[i], 0, -0.75f, 2.5f, error, error_capacity);
        if (rc == 0 && !g_edges[i].trans_a)
            rc = sg_case_run(&t, &g_edges[i], 1, 1.0f, 0.0f, error, error_capacity);
        if (rc == 0 && !g_edges[i].trans_a)
            rc = sg_case_run(&t, &g_edges[i], 1, 0.5f, -1.25f, error, error_capacity);
    }
    /* Task-count and thread-width independence, on shapes that really block in
     * both axes and in both transposes (so the DOT family is covered too). */
    for (size_t i = 0; i < n_edge && rc == 0; ++i) {
        if (g_edges[i].m * g_edges[i].n * g_edges[i].k < 1024u) continue;
        rc = sg_case_tasks(&t, &g_edges[i], 0, error, error_capacity);
        if (rc == 0 && !g_edges[i].trans_a)
            rc = sg_case_tasks(&t, &g_edges[i], 1, error, error_capacity);
        if (rc == 0)
            rc = sg_case_threadwidths(&t, &g_edges[i], 0, error, error_capacity);
        if (rc == 0 && !g_edges[i].trans_a)
            rc = sg_case_threadwidths(&t, &g_edges[i], 1, error, error_capacity);
    }
    if (rc == 0) {
        static const sg_case sweep[] = {
            {64, 1024, 1024, 0, "the stacked stream-step shape"},
            {16, 4096, 1024, 0, "the FFN up shape"},
        };
        for (size_t i = 0; i < 2u && rc == 0; ++i) {
            rc = sg_case_tasks(&t, &sweep[i], 1, error, error_capacity);
            if (rc == 0)
                rc = sg_case_tasks(&t, &sweep[i], 0, error, error_capacity);
            if (rc == 0)
                rc = sg_case_threadwidths(&t, &sweep[i], 1, error, error_capacity);
            if (rc == 0)
                rc = sg_case_threadwidths(&t, &sweep[i], 0, error, error_capacity);
        }
    }

    /* Coverage refusal.  A self-test in which a family never ran proves nothing
     * about that family, and reporting PASS would be exactly the silent
     * fallback ENGINEERING.md §6 forbids. */
    if (rc == 0) {
        static const char *names[5] = {"reference", "dot", "matvec", "narrow",
                                       "panel"};
        for (int f = 0; f <= (int)MYNAH_ASR_SGEMM_FAMILY_PANEL; ++f) {
            if (t.ran[f] == 0ull) {
                snprintf(error, error_capacity,
                         "sgemm self-test covered no %s case: the comparison "
                         "would be vacuous, so this is a failure and not a pass",
                         names[f]);
                rc = -1;
                break;
            }
        }
    }

    g_sg_worst = t.worst;
    free(t.a); free(t.b); free(t.c); free(t.ref);
    return rc;
}
