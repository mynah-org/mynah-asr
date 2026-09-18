#include "qmat.h"

#include "backend.h"
#include "dispatch.h"   /* mynah_asr_cpu_has(): ONE tri-state CPU probe (§6) */
#include "threads.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef MYNAH_ASR_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* ------------------------------------------------------------- quantizers */
void mynah_asr_quantize_int8(const float *w, int n, int k, int8_t *out_q, float *out_scales) {
    for (int i = 0; i < n; i++) {
        const float *row = w + (size_t)i * (size_t)k;
        float amax = 0.0f;
        for (int j = 0; j < k; j++) {
            const float a = fabsf(row[j]);
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        out_scales[i] = scale;
        const float inv = 1.0f / scale;
        int8_t *qrow = out_q + (size_t)i * (size_t)k;
        for (int j = 0; j < k; j++) {
            const float v = row[j] * inv;
            qrow[j] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        }
    }
}

void mynah_asr_quantize_int4(const float *w, int n, int k, uint8_t *out_q, float *out_scales) {
    const int G = MYNAH_ASR_Q4_GROUP;
    const int groups = k / G;
    for (int i = 0; i < n; i++) {
        const float *row = w + (size_t)i * (size_t)k;
        uint8_t *qrow = out_q + (size_t)i * (size_t)(k / 2);
        float *srow = out_scales + (size_t)i * (size_t)groups;
        for (int g = 0; g < groups; g++) {
            const float *grp = row + g * G;
            float amax = 0.0f;
            for (int j = 0; j < G; j++) {
                const float a = fabsf(grp[j]);
                if (a > amax) amax = a;
            }
            const float scale = amax > 0.0f ? amax / 7.0f : 1.0f;
            srow[g] = scale;
            const float inv = 1.0f / scale;
            for (int j = 0; j < G; j += 2) {
                float v0 = grp[j] * inv, v1 = grp[j + 1] * inv;
                int q0 = (int)(v0 >= 0.0f ? v0 + 0.5f : v0 - 0.5f);
                int q1 = (int)(v1 >= 0.0f ? v1 + 0.5f : v1 - 0.5f);
                if (q0 < -8) q0 = -8;
                if (q0 > 7) q0 = 7;
                if (q1 < -8) q1 = -8;
                if (q1 > 7) q1 = 7;
                qrow[(g * G + j) / 2] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
            }
        }
    }
}

/* --------------------------------------------------------------------- init */
static void release_f32_pages(const float *w, size_t bytes) {
    /* mmap pages of the quantized f32: clean and re-readable — on Linux DONTNEED
     * releases them immediately, on macOS they stay reclaimable under pressure
     * (RSS accounting does not drop: known limitation, see TODO M5) */
    const long pg = sysconf(_SC_PAGESIZE);
    uintptr_t lo = ((uintptr_t)w + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    uintptr_t hi = ((uintptr_t)w + bytes) & ~((uintptr_t)pg - 1);
    if (hi > lo) madvise((void *)lo, hi - lo, MADV_DONTNEED);
}

int mynah_asr_qmat_init(mynah_asr_qmat *m, const float *w, int n, int k, int qtype) {
    memset(m, 0, sizeof(*m));
    m->f32 = w;
    m->n = n;
    m->k = k;
    m->qtype = MYNAH_ASR_Q_F32;
    if (qtype == MYNAH_ASR_Q_F32 || !w) return 0;

    if (qtype == MYNAH_ASR_Q_INT8) {
        int8_t *q = malloc((size_t)n * (size_t)k);
        float *s = malloc((size_t)n * sizeof(float));
        if (!q || !s) { free(q); free(s); return -1; }
        mynah_asr_quantize_int8(w, n, k, q, s);
        m->q8 = q;
        m->scales = s;
        m->owned_q = q;
        m->owned_s = s;
        m->qtype = MYNAH_ASR_Q_INT8;
    } else {
        if (k % MYNAH_ASR_Q4_GROUP != 0) return 0;   /* stays f32 */
        uint8_t *q = malloc((size_t)n * (size_t)k / 2);
        float *s = malloc((size_t)n * (size_t)(k / MYNAH_ASR_Q4_GROUP) * sizeof(float));
        if (!q || !s) { free(q); free(s); return -1; }
        mynah_asr_quantize_int4(w, n, k, q, s);
        m->q4 = q;
        m->scales = s;
        m->owned_q = q;
        m->owned_s = s;
        m->qtype = MYNAH_ASR_Q_INT4;
    }
    release_f32_pages(w, (size_t)n * (size_t)k * 4u);
    return 0;
}

int mynah_asr_qmat_init_st(mynah_asr_qmat *m, const mynah_asr_safetensors *st, const char *name,
                       int qtype) {
    memset(m, 0, sizeof(*m));
    char qname[192];

    if (qtype != MYNAH_ASR_Q_F32) {
        snprintf(qname, sizeof(qname), "%s.%s", name, qtype == MYNAH_ASR_Q_INT8 ? "q8" : "q4");
        const mynah_asr_tensor *tq = mynah_asr_st_get(st, qname);
        snprintf(qname, sizeof(qname), "%s.scales", name);
        const mynah_asr_tensor *ts = mynah_asr_st_get(st, qname);
        if (tq && ts) {                       /* pre-quantized: zero-copy from the mmap */
            m->n = (int)tq->shape[0];
            m->scales = (const float *)ts->data;
            if (qtype == MYNAH_ASR_Q_INT8) {
                m->k = (int)tq->shape[1];
                m->q8 = (const int8_t *)tq->data;
                m->qtype = MYNAH_ASR_Q_INT8;
            } else {
                m->k = (int)tq->shape[1] * 2;
                m->q4 = (const uint8_t *)tq->data;
                m->qtype = MYNAH_ASR_Q_INT4;
            }
            return 0;
        }
    }
    const mynah_asr_tensor *tf = mynah_asr_st_get(st, name);
    if (!tf || tf->shape[0] <= 0) return -1;
    return mynah_asr_qmat_init(m, (const float *)tf->data, (int)tf->shape[0],
                           (int)(tf->n_elems / (size_t)tf->shape[0]), qtype);
}

void mynah_asr_qmat_free(mynah_asr_qmat *m) {
    free(m->owned_q);
    free(m->owned_s);
    memset(m, 0, sizeof(*m));
}

/* ------------------------------------------------------------------ dequant */
static void dequant_row(const mynah_asr_qmat *m, int i, float *dst) {
    if (m->qtype == MYNAH_ASR_Q_INT8) {
        const int8_t *qrow = m->q8 + (size_t)i * (size_t)m->k;
        const float s = m->scales[i];
        for (int j = 0; j < m->k; j++) dst[j] = (float)qrow[j] * s;
    } else {
        const uint8_t *qrow = m->q4 + (size_t)i * (size_t)(m->k / 2);
        const float *srow = m->scales + (size_t)i * (size_t)(m->k / MYNAH_ASR_Q4_GROUP);
        for (int g = 0; g < m->k / MYNAH_ASR_Q4_GROUP; g++) {
            const float s = srow[g];
            for (int j = 0; j < MYNAH_ASR_Q4_GROUP; j += 2) {
                const uint8_t b = qrow[(g * MYNAH_ASR_Q4_GROUP + j) / 2];
                dst[g * MYNAH_ASR_Q4_GROUP + j] = (float)((int)(b & 0x0F) - 8) * s;
                dst[g * MYNAH_ASR_Q4_GROUP + j + 1] = (float)((int)(b >> 4) - 8) * s;
            }
        }
    }
}

void mynah_asr_qmat_dequant(const mynah_asr_qmat *m, float *wd) {
    if (m->qtype == MYNAH_ASR_Q_F32) {
        memcpy(wd, m->f32, (size_t)m->n * (size_t)m->k * sizeof(float));
        return;
    }
    for (int i = 0; i < m->n; i++) dequant_row(m, i, wd + (size_t)i * (size_t)m->k);
}

/* row threshold: below -> direct dot (bandwidth-bound), above -> dequant+GEMM */
#define QMAT_SMALL_T 16
#define QMAT_K_MAX 8192

/* ------------------------------------------------------------- path counters
 * One relaxed atomic increment per product: negligible next to a GEMM, and it
 * is the only way a test can PROVE which kernel a run took (ENGINEERING.md §5).
 * Process-wide, like the caps cache. */
static _Atomic unsigned long long g_qc[MYNAH_ASR_QC__N];
static inline void qc(int which) {
    atomic_fetch_add_explicit(&g_qc[which], 1ull, memory_order_relaxed);
}
unsigned long long mynah_asr_qmat_counter(int which) {
    if (which < 0 || which >= MYNAH_ASR_QC__N) return 0;
    return atomic_load_explicit(&g_qc[which], memory_order_relaxed);
}
void mynah_asr_qmat_counters_reset(void) {
    for (int i = 0; i < MYNAH_ASR_QC__N; i++)
        atomic_store_explicit(&g_qc[i], 0ull, memory_order_relaxed);
}

/* ...and which MICRO-KERNEL did the arithmetic inside that path (S5-1). */
static _Atomic unsigned long long g_qk[MYNAH_ASR_QK__N];
static inline void qk(int which) {
    atomic_fetch_add_explicit(&g_qk[which], 1ull, memory_order_relaxed);
}
unsigned long long mynah_asr_qmat_kernel_counter(int which) {
    if (which < 0 || which >= MYNAH_ASR_QK__N) return 0;
    return atomic_load_explicit(&g_qk[which], memory_order_relaxed);
}
void mynah_asr_qmat_kernel_counters_reset(void) {
    for (int i = 0; i < MYNAH_ASR_QK__N; i++)
        atomic_store_explicit(&g_qk[i], 0ull, memory_order_relaxed);
}
const char *mynah_asr_qmat_kernel_name(int which) {
    switch (which) {
        case MYNAH_ASR_QK_SCALAR:       return "scalar";
        case MYNAH_ASR_QK_AVX2:         return "avx2";
        case MYNAH_ASR_QK_AVXVNNI:      return "avxvnni";
        case MYNAH_ASR_QK_AVX512VNNI:   return "avx512vnni";
        case MYNAH_ASR_QK_NEON_SDOT:    return "neon-sdot";
        case MYNAH_ASR_QK_NEON_SMMLA:   return "neon-smmla";
        case MYNAH_ASR_QK_DOT_PER_ROW:  return "dot-per-row";
        default:                        return "?";
    }
}

/* THE ONE DEFINITION of the int8 epilogue: int32 accumulator, per-row weight
 * scale, per-row activation scale. EVERY int8 kernel in this file ends here.
 *
 * WHY THE BARRIER. `(float)s * ws * sx` is three factors, and under
 * -ffast-math the compiler may group them either way. It does not choose the
 * same way everywhere: with the expression written out longhand in each
 * kernel, Apple clang 21 -O3 -march=native -ffast-math grouped it one way when
 * dot_q8_sdot was inlined into the per-row loop and the OTHER way when the
 * same function was inlined into the SMMLA kernel's odd-row tail — 0.867895186
 * against 0.867895246 for the same weight row and the same activation row,
 * with the integers identical. mynah-tts hit this first (71 of 192 rows, up to
 * 2 ULP, GCC 15) and it is not cosmetic: the grouping is chosen by WHICH
 * KERNEL served the row, the kernel is chosen by the tile position, and the
 * tile position is chosen by how many other streams happened to be in the
 * batch. A request's transcript would depend on who it was batched with
 * (ENGINEERING.md §9).
 *
 * The barrier removes the freedom instead of hoping. WHICH grouping to pin was
 * not a matter of taste: it had to be the one the shipped binary already
 * produced, or this would be a numerical change smuggled in as a refactor. It
 * was measured, not assumed — dumping both entries (offline
 * mynah_asr_qmat_mul, including its T>16 arm, and the stacked
 * mynah_asr_qmat_mul_rows) over 7 shapes x 9 values of T, 791,448 bytes of
 * output, before and after. Left-to-right `(s*ws)*sx`, which is what the
 * source text says, DIFFERS. `s*(ws*sx)` is byte-identical: -ffast-math had
 * quietly been hoisting the scale product all along, and that is now the
 * pinned definition rather than a compiler mood. One FP move per output
 * element. */
static inline float qmat_fp_barrier(float v) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__("" : "+w"(v));
#elif defined(__x86_64__) || defined(__i386__)
    __asm__("" : "+x"(v));
#else
    volatile float t = v;
    v = t;
#endif
    return v;
}

static inline float qmat_q8_epilogue(int32_t s, float ws, float sx) {
    return (float)s * qmat_fp_barrier(ws * sx);
}

/* -------------------------------------------------- activation quantization
 * Per-row absmax -> int8 (qwen-tts recipe, quality verified in production):
 * enables the native int8xint8 dot (SDOT/VNNI) without a per-weight dequant. */
static float quantize_act_int8(int8_t *qx, const float *x, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) { memset(qx, 0, (size_t)n); return 0.0f; }
    const float inv = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        const float v = x[i] * inv;
        int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qx[i] = (int8_t)q;
    }
    return amax / 127.0f;
}

/* ------------------------------------- the activation pass for T rows (S5-1)
 * Rows are independent (per-row absmax), so putting them on the pool is
 * bit-identical by construction — the same reason mynah_asr_parallel_for is
 * safe for the weight-row blocks.
 *
 * WHEN IT PAYS, and the measurement behind the constant. mynah-tts measured
 * ~20 us of wake-up latency per parallel region and derived the rule that a
 * region must be worth >= 200 us before it is worth dispatching; the sibling's
 * own activation-quantisation pass sat at a flat 3.8 ms while the GEMM beside
 * it threaded 16.7 -> 7.7 ms, which is why this was named as the serial
 * fraction to attack. OURS IS NOT THAT PASS. Measured on the M1 dev host
 * (Apple clang 21, -O3 -march=native -ffast-math, isolated loop over
 * quantize_act_int8): 0.167 ns/element at k=1024 and 0.106 ns/element at
 * k=4096 — 10.9 us and 27.8 us for a whole T=64 stack. Against the 200 us
 * rule the break-even is ~1.9 M elements, so:
 *
 *   QMAT_ACT_PAR_ELEMS = 1<<21 = 2,097,152 elements (T*k)
 *
 * and on the shapes the batched stream step actually produces (T = 4..64,
 * k = 1024 or 4096, i.e. at most 262,144 elements) the pool is NEVER engaged:
 * the pass is ~0.4% of a call whose GEMM is n times larger. The threshold is
 * kept, not deleted, because it is the crossover a bigger T or a host with a
 * slower core would cross, and because a constant with a measurement behind it
 * is the difference between a policy and a guess. Linux owes its own number:
 * the ns/element above is an Apple-core figure. */
#define QMAT_ACT_PAR_ELEMS (1 << 21)

typedef struct {
    const float *x;
    int8_t *qx;
    float *sx;
    int k;
} actq_ctx;

static void actq_row(void *ctx, int t) {
    const actq_ctx *c = ctx;
    c->sx[t] = quantize_act_int8(c->qx + (size_t)t * (size_t)c->k,
                                 c->x + (size_t)t * (size_t)c->k, c->k);
}

static void quantize_act_rows(int8_t *qx, float *sx, const float *x, int T, int k) {
    if ((long long)T * (long long)k >= QMAT_ACT_PAR_ELEMS && T > 1 &&
        mynah_asr_num_threads() > 1) {
        actq_ctx c = {.x = x, .qx = qx, .sx = sx, .k = k};
        mynah_asr_parallel_for(T, actq_row, &c);
        return;
    }
    for (int t = 0; t < T; t++)
        sx[t] = quantize_act_int8(qx + (size_t)t * (size_t)k, x + (size_t)t * (size_t)k, k);
}

/* ---------------------------------------------------- SDOT kernel (ARMv8.2+)
 * Native int8xint8: 4 MACs per lane per instruction, zero f32 conversions.
 * Pattern from qwen-tts (int8_matvec_sdot), extended to q4 with vld2q_s8 for the
 * even/odd deinterleave of the nibbles (order inside the dot does not matter, but
 * the SDOT lanes must line up element by element). */
#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>

static float dot_q8_sdot(const int8_t *qx, float sx, const int8_t *w, float ws, int k) {
    int32x4_t acc = vdupq_n_s32(0);
    int j = 0;
    for (; j + 15 < k; j += 16)
        acc = vdotq_s32(acc, vld1q_s8(w + j), vld1q_s8(qx + j));
    int32_t s = vaddvq_s32(acc);
    for (; j < k; j++) s += (int32_t)w[j] * qx[j];
    return qmat_q8_epilogue(s, ws, sx);   /* ONE grouping, see qmat_q8_epilogue */
}

static float dot_q4_sdot(const int8_t *qx, float sx, const uint8_t *q,
                         const float *scales, int k) {
    const int8x16_t off = vdupq_n_s8(8);
    const uint8x16_t maskv = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    for (int g = 0; g < k / MYNAH_ASR_Q4_GROUP; g++) {
        const uint8x16_t b = vld1q_u8(q + g * 16);
        const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, maskv)), off);
        const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), off);
        const int8x16x2_t xg = vld2q_s8(qx + g * 32);  /* val[0]=even, val[1]=odd */
        int32x4_t ig = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, xg.val[0]), hi, xg.val[1]);
        acc += (float)vaddvq_s32(ig) * scales[g];
    }
    return acc * sx;
}

/* Weight-stationary SDOT: ONE weight row against FOUR activation rows, so the
 * 16-byte weight vector is loaded once per four dot products instead of once
 * per dot product. Integer accumulation, so the four results are exactly the
 * four dot_q8_sdot() would have produced (int32 addition is associative and
 * the products are identical) — asserted with `==` in tests/test_qmat. */
static void dots_q8_sdot_x4(const int8_t *w, const int8_t *x0, const int8_t *x1,
                            const int8_t *x2, const int8_t *x3, int k, int32_t s[4]) {
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int j = 0;
    for (; j + 15 < k; j += 16) {
        const int8x16_t wv = vld1q_s8(w + j);
        a0 = vdotq_s32(a0, wv, vld1q_s8(x0 + j));
        a1 = vdotq_s32(a1, wv, vld1q_s8(x1 + j));
        a2 = vdotq_s32(a2, wv, vld1q_s8(x2 + j));
        a3 = vdotq_s32(a3, wv, vld1q_s8(x3 + j));
    }
    s[0] = vaddvq_s32(a0);
    s[1] = vaddvq_s32(a1);
    s[2] = vaddvq_s32(a2);
    s[3] = vaddvq_s32(a3);
    for (; j < k; j++) {
        const int32_t wj = w[j];
        s[0] += wj * x0[j];
        s[1] += wj * x1[j];
        s[2] += wj * x2[j];
        s[3] += wj * x3[j];
    }
}
#define MYNAH_ASR_HAVE_SDOT 1
#endif

/* ------------------------------------------------------ ARM i8mm (SMMLA), S5-1
 * SMMLA is a 2x2x8 outer product: one instruction consumes TWO weight rows and
 * TWO activation rows and issues 32 MACs against SDOT's 16. The second
 * activation row is the whole point, which is also the honest limit of the
 * instruction: a one-row decode dot has nothing to put in the other half, so
 * it is wired into the weight-stationary batched pass and NOWHERE ELSE.
 *
 * vmmlaq_s32(r, a, b) fills r with the four dot products of the two 8-byte
 * halves of `a` against the two of `b`:
 *     r[0] += a.lo . b.lo   r[1] += a.lo . b.hi
 *     r[2] += a.hi . b.lo   r[3] += a.hi . b.hi
 * so a = (w[i], w[i+1]) and b = (x[t], x[t+1]) is the whole 2x2 tile with two
 * 64-bit loads per operand and NO weight repacking: the row-major int8 layout
 * already is what SMMLA wants.
 *
 * COMPILED INTO EVERY aarch64 BUILD through the target attribute and selected
 * by a runtime probe — never by a -march gate, which is how the sibling repo
 * once shipped an ISA claim it could not honour (ENGINEERING.md §5). The probe
 * is mynah_asr_cpu_has("i8mm"), the SAME tri-state detector the dispatch report
 * reads (§6: one source, never a second copy); UNKNOWN is not a yes.
 *
 * clang has always accepted `target("+i8mm")` with the i8mm intrinsics; GCC
 * grew the aarch64 target-attribute machinery for them in 10. */
#if defined(MYNAH_ASR_HAVE_SDOT) && defined(__aarch64__) && \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 10))
#define MYNAH_ASR_HAVE_I8MM 1

/* s = { w0.x0, w0.x1, w1.x0, w1.x1 } — exactly the four integers four
 * dot_q8_sdot() calls would have accumulated. */
__attribute__((target("+i8mm")))
static void dots_q8_smmla_2x2(const int8_t *w0, const int8_t *w1,
                              const int8_t *x0, const int8_t *x1, int k, int32_t s[4]) {
    int32x4_t acc_a = vdupq_n_s32(0), acc_b = vdupq_n_s32(0);
    int j = 0;
    for (; j + 16 <= k; j += 16) {
        acc_a = vmmlaq_s32(acc_a, vcombine_s8(vld1_s8(w0 + j), vld1_s8(w1 + j)),
                                  vcombine_s8(vld1_s8(x0 + j), vld1_s8(x1 + j)));
        acc_b = vmmlaq_s32(acc_b, vcombine_s8(vld1_s8(w0 + j + 8), vld1_s8(w1 + j + 8)),
                                  vcombine_s8(vld1_s8(x0 + j + 8), vld1_s8(x1 + j + 8)));
    }
    for (; j + 8 <= k; j += 8)
        acc_a = vmmlaq_s32(acc_a, vcombine_s8(vld1_s8(w0 + j), vld1_s8(w1 + j)),
                                  vcombine_s8(vld1_s8(x0 + j), vld1_s8(x1 + j)));
    const int32x4_t acc = vaddq_s32(acc_a, acc_b);
    s[0] = vgetq_lane_s32(acc, 0);
    s[1] = vgetq_lane_s32(acc, 1);
    s[2] = vgetq_lane_s32(acc, 2);
    s[3] = vgetq_lane_s32(acc, 3);
    for (; j < k; j++) {
        const int32_t a0 = x0[j], a1 = x1[j];
        s[0] += (int32_t)w0[j] * a0;
        s[1] += (int32_t)w0[j] * a1;
        s[2] += (int32_t)w1[j] * a0;
        s[3] += (int32_t)w1[j] * a1;
    }
}
#endif

/* --------------------------------------------- x86 kernels (q8 VNNI/AVX2, q4 AVX2)
 * RUNTIME DISPATCH (qwen-tts --caps pattern): the kernels are always compiled
 * with a target attribute (no -march needed: multi-target release binaries), and
 * the selection happens through cpuid+xgetbv on the first call.
 * Override with mynah_asr_set_caps("scalar"|"avx2"|"vnni") or env MYNAH_ASR_CAPS.
 * dpbusd/maddubs multiply u8 x s8: ua = qx+128 with a -128*Σw correction
 * (qwen-tts int8_matvec_vnni pattern). Validated by tests/test_qmat in CI. */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#define MYNAH_ASR_HAVE_X86 1

enum { MYNAH_ASR_CAPS_SCALAR = 0, MYNAH_ASR_CAPS_AVX2 = 1, MYNAH_ASR_CAPS_VNNI = 2 };

static unsigned long long xgetbv0(void) {
    unsigned lo, hi;
    __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}

/* The TWO VPDPBUSD encodings are separate CPU features and must be asked for
 * separately: AVX-512 VNNI (EVEX, zmm) and AVX-VNNI (VEX, ymm — the
 * AVX-512-less client parts). Issuing the EVEX kernel because the VEX feature
 * is present would be a SIGILL, so the caps LADDER says "this CPU has a VNNI
 * of some encoding" and these two say which one may actually be emitted. */
static int x86_have_avx512vnni(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 27))) return 0;  /* OSXSAVE */
    const unsigned long long xcr0 = xgetbv0();
    if ((xcr0 & 0xE6) != 0xE6) return 0;                     /* xmm+ymm+zmm+opmask */
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    return ((b >> 16) & 1) && ((b >> 30) & 1) && ((c >> 11) & 1); /* f, bw, vnni */
}

static int x86_have_avxvnni(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 27))) return 0;
    if ((xgetbv0() & 0x6) != 0x6) return 0;                  /* xmm+ymm from the OS */
    if (!__get_cpuid_count(7, 1, &a, &b, &c, &d)) return 0;
    return (a >> 4) & 1;                                     /* leaf 7.1 EAX[4] */
}

static int x86_detect_caps(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 27))) /* OSXSAVE */
        return MYNAH_ASR_CAPS_SCALAR;
    const unsigned long long xcr0 = xgetbv0();
    if ((xcr0 & 0x6) != 0x6) return MYNAH_ASR_CAPS_SCALAR;        /* xmm+ymm from the OS */
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d) || !((b >> 5) & 1)) /* AVX2 */
        return MYNAH_ASR_CAPS_SCALAR;
    const int avx512f = (b >> 16) & 1, avx512bw = (b >> 30) & 1, vnni = (c >> 11) & 1;
    if (avx512f && avx512bw && vnni && (xcr0 & 0xE0) == 0xE0) /* zmm+opmask */
        return MYNAH_ASR_CAPS_VNNI;
    /* S5-1: an AVX-VNNI part reaches the same ladder rung. The per-row dot
     * below still asks x86_have_avx512vnni() before emitting the EVEX kernel,
     * so this widening changes NO existing dispatch: on an AVX-512 VNNI host
     * the level and the encoding agree exactly as before, and on an AVX-VNNI
     * host the per-row dot stays on AVX2 while the weight-stationary kernel
     * gets its VEX twin. */
    if (x86_have_avxvnni()) return MYNAH_ASR_CAPS_VNNI;
    return MYNAH_ASR_CAPS_AVX2;
}

static int g_x86_caps = -1;

static int x86_caps(void) {
    if (g_x86_caps < 0) {
        const char *env = getenv("MYNAH_ASR_CAPS");
        if (env) mynah_asr_set_caps(env);
        if (g_x86_caps < 0) g_x86_caps = x86_detect_caps();
    }
    return g_x86_caps;
}

/* "May this build emit that encoding on this host, at this caps level?" —
 * the single question every x86 int8 kernel site asks. MYNAH_ASR_CAPS=avx2
 * turns both off, MYNAH_ASR_CAPS=scalar turns the native dot off entirely. */
static int x86_use_avx512vnni(void) {
    static int have = -1;
    if (have < 0) have = x86_have_avx512vnni();
    return have && x86_caps() >= MYNAH_ASR_CAPS_VNNI;
}
static int x86_use_avxvnni(void) {
    static int have = -1;
    if (have < 0) have = x86_have_avxvnni();
    return have && !x86_use_avx512vnni() && x86_caps() >= MYNAH_ASR_CAPS_VNNI;
}

__attribute__((target("avx512f,avx512bw,avx512vnni")))
static float dot_q8_vnni(const int8_t *qx, float sx, const int8_t *w, float ws, int k) {
    const __m512i v128 = _mm512_set1_epi8((char)128);
    const __m512i ones = _mm512_set1_epi8(1);
    __m512i acc = _mm512_setzero_si512(), wsum = _mm512_setzero_si512();
    int j = 0;
    for (; j + 64 <= k; j += 64) {
        const __m512i ua = _mm512_add_epi8(_mm512_loadu_si512((const void *)(qx + j)), v128);
        const __m512i wv = _mm512_loadu_si512((const void *)(w + j));
        acc = _mm512_dpbusd_epi32(acc, ua, wv);
        wsum = _mm512_dpbusd_epi32(wsum, ones, wv);
    }
    int s = _mm512_reduce_add_epi32(acc) - 128 * _mm512_reduce_add_epi32(wsum);
    for (; j < k; j++) s += (int)w[j] * qx[j];
    return qmat_q8_epilogue(s, ws, sx);   /* ONE grouping, see qmat_q8_epilogue */
}

__attribute__((target("avx2")))
static inline int hsum_i32_avx2(__m256i v) {
    __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

/* sign/abs trick (llama.cpp): maddubs(|w|, qx*sign(w)) — pairs <= 32258, so no
 * int16 saturation and no correction term */
__attribute__((target("avx2")))
static float dot_q8_avx2(const int8_t *qx, float sx, const int8_t *w, float ws, int k) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    int j = 0;
    for (; j + 32 <= k; j += 32) {
        const __m256i xv = _mm256_loadu_si256((const __m256i *)(qx + j));
        const __m256i wv = _mm256_loadu_si256((const __m256i *)(w + j));
        const __m256i aw = _mm256_sign_epi8(wv, wv);   /* |w| as u8 */
        const __m256i sxv = _mm256_sign_epi8(xv, wv);  /* qx * sign(w) */
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sxv), ones16));
    }
    __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    int s = _mm_cvtsi128_si32(lo);
    for (; j < k; j++) s += (int)w[j] * qx[j];
    return qmat_q8_epilogue(s, ws, sx);   /* ONE grouping, see qmat_q8_epilogue */
}

/* q4 on x86 (plain AVX2, used on VNNI machines too). The ACTIVATIONS are
 * pre-permuted per group ([even(16) | odd(16)]) once per row, so the nibble
 * unpack [lo(16) | hi(16)] lines up without a shuffle in the inner loop.
 * |w| <= 8 => maddubs can never saturate (8*127*2 = 2032). */

/* xq -> xq_perm: for each group of 32, the even elements first, then the odd */
static void q4_permute_act(const int8_t *qx, int8_t *xp, int k) {
    for (int g = 0; g < k / MYNAH_ASR_Q4_GROUP; g++) {
        const int8_t *src = qx + g * 32;
        int8_t *dst = xp + g * 32;
        for (int j = 0; j < 16; j++) {
            dst[j] = src[2 * j];
            dst[16 + j] = src[2 * j + 1];
        }
    }
}

__attribute__((target("avx2")))
static float dot_q4_x86(const int8_t *xp /* permuted */, float sx, const uint8_t *q,
                        const float *scales, int k) {
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m128i off8 = _mm_set1_epi8(8);
    const __m256i ones16 = _mm256_set1_epi16(1);
    float acc = 0.0f;
    for (int g = 0; g < k / MYNAH_ASR_Q4_GROUP; g++) {
        const __m128i b = _mm_loadu_si128((const __m128i *)(q + g * 16));
        const __m128i lo = _mm_sub_epi8(_mm_and_si128(b, mask4), off8);
        const __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(b, 4), mask4), off8);
        const __m256i wv = _mm256_set_m128i(hi, lo);              /* [lo(16) | hi(16)] */
        const __m256i xv = _mm256_loadu_si256((const __m256i *)(xp + g * 32));
        const __m256i aw = _mm256_sign_epi8(wv, wv);
        const __m256i sxv = _mm256_sign_epi8(xv, wv);
        const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sxv), ones16);
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p32),
                                  _mm256_extracti128_si256(p32, 1));
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        acc += (float)_mm_cvtsi128_si32(s) * scales[g];
    }
    return acc * sx;
}

/* ------------------------------------------- x86 weight-stationary int8 (S5-1)
 * ONE weight row against FOUR activation rows: the weight vector is loaded
 * once per four dot products, and on the VNNI leaves the per-row weight sum
 * that pays for the unsigned encoding is computed once instead of four times.
 *
 * THE ENCODING, stated once. VPDPBUSD (and maddubs) multiply UNSIGNED bytes by
 * SIGNED bytes, so the activation is shifted to u8 as `x + 128` and the shift
 * is taken back out with the per-weight-row correction
 *     sum_j (x_j + 128) * w_j  -  128 * sum_j w_j  =  sum_j x_j * w_j
 * exactly as dot_q8_vnni does — the same definition, hoisted so the rowsum is
 * amortised over the four activation rows. Every term is integer, so the
 * result is bit-identical to the per-row dot and to the scalar reference.
 *
 * The AVX2 leaf uses the sign/abs trick instead (llama.cpp, and already in
 * dot_q8_avx2): maddubs(|w|, x*sign(w)) needs no correction term and cannot
 * saturate, because a pair sums to at most 2*128*127 = 32512 < 32767. */

__attribute__((target("avx512f,avx512bw,avx512vnni")))
static void dots_q8_avx512vnni_x4(const int8_t *w, const int8_t *x0, const int8_t *x1,
                                  const int8_t *x2, const int8_t *x3, int k, int32_t s[4]) {
    const __m512i v128 = _mm512_set1_epi8((char)128);
    const __m512i ones = _mm512_set1_epi8(1);
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    __m512i wsum = _mm512_setzero_si512();
    int j = 0;
    for (; j + 64 <= k; j += 64) {
        const __m512i wv = _mm512_loadu_si512((const void *)(w + j));
        a0 = _mm512_dpbusd_epi32(a0, _mm512_add_epi8(_mm512_loadu_si512((const void *)(x0 + j)), v128), wv);
        a1 = _mm512_dpbusd_epi32(a1, _mm512_add_epi8(_mm512_loadu_si512((const void *)(x1 + j)), v128), wv);
        a2 = _mm512_dpbusd_epi32(a2, _mm512_add_epi8(_mm512_loadu_si512((const void *)(x2 + j)), v128), wv);
        a3 = _mm512_dpbusd_epi32(a3, _mm512_add_epi8(_mm512_loadu_si512((const void *)(x3 + j)), v128), wv);
        wsum = _mm512_dpbusd_epi32(wsum, ones, wv);
    }
    const int corr = 128 * _mm512_reduce_add_epi32(wsum);
    s[0] = _mm512_reduce_add_epi32(a0) - corr;
    s[1] = _mm512_reduce_add_epi32(a1) - corr;
    s[2] = _mm512_reduce_add_epi32(a2) - corr;
    s[3] = _mm512_reduce_add_epi32(a3) - corr;
    for (; j < k; j++) {
        const int wj = w[j];
        s[0] += wj * x0[j];
        s[1] += wj * x1[j];
        s[2] += wj * x2[j];
        s[3] += wj * x3[j];
    }
}

/* The VEX twin for the AVX-512-less client parts. GCC named the VEX intrinsic
 * _mm256_dpbusd_avx_epi32 in 11 (clang in 12) to keep it distinct from the
 * AVX512VL one of the same shape; older compilers simply do not get this leaf
 * and the dispatcher falls to AVX2. */
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 11)
#define MYNAH_ASR_HAVE_AVXVNNI_KERNEL 1
__attribute__((target("avx2,avxvnni")))
static void dots_q8_avxvnni_x4(const int8_t *w, const int8_t *x0, const int8_t *x1,
                               const int8_t *x2, const int8_t *x3, int k, int32_t s[4]) {
    const __m256i v128 = _mm256_set1_epi8((char)128);
    const __m256i ones = _mm256_set1_epi8(1);
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    __m256i wsum = _mm256_setzero_si256();
    int j = 0;
    for (; j + 32 <= k; j += 32) {
        const __m256i wv = _mm256_loadu_si256((const __m256i *)(w + j));
        a0 = _mm256_dpbusd_avx_epi32(a0, _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(x0 + j)), v128), wv);
        a1 = _mm256_dpbusd_avx_epi32(a1, _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(x1 + j)), v128), wv);
        a2 = _mm256_dpbusd_avx_epi32(a2, _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(x2 + j)), v128), wv);
        a3 = _mm256_dpbusd_avx_epi32(a3, _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(x3 + j)), v128), wv);
        wsum = _mm256_dpbusd_avx_epi32(wsum, ones, wv);
    }
    const int corr = 128 * hsum_i32_avx2(wsum);
    s[0] = hsum_i32_avx2(a0) - corr;
    s[1] = hsum_i32_avx2(a1) - corr;
    s[2] = hsum_i32_avx2(a2) - corr;
    s[3] = hsum_i32_avx2(a3) - corr;
    for (; j < k; j++) {
        const int wj = w[j];
        s[0] += wj * x0[j];
        s[1] += wj * x1[j];
        s[2] += wj * x2[j];
        s[3] += wj * x3[j];
    }
}
#endif

__attribute__((target("avx2")))
static void dots_q8_avx2_x4(const int8_t *w, const int8_t *x0, const int8_t *x1,
                            const int8_t *x2, const int8_t *x3, int k, int32_t s[4]) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    int j = 0;
    for (; j + 32 <= k; j += 32) {
        const __m256i wv = _mm256_loadu_si256((const __m256i *)(w + j));
        const __m256i aw = _mm256_sign_epi8(wv, wv);              /* |w| as u8 */
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(
                 aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(x0 + j)), wv)), ones16));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(
                 aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(x1 + j)), wv)), ones16));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(
                 aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(x2 + j)), wv)), ones16));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(
                 aw, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(x3 + j)), wv)), ones16));
    }
    s[0] = hsum_i32_avx2(a0);
    s[1] = hsum_i32_avx2(a1);
    s[2] = hsum_i32_avx2(a2);
    s[3] = hsum_i32_avx2(a3);
    for (; j < k; j++) {
        const int wj = w[j];
        s[0] += wj * x0[j];
        s[1] += wj * x1[j];
        s[2] += wj * x2[j];
        s[3] += wj * x3[j];
    }
}
#endif /* x86 */

/* ------------------------------------------------------- the ARM caps ladder
 * The same opt-out vocabulary as x86, with the ARM rungs: scalar (no native
 * int8 dot at all) < sdot < smmla. It exists so `MYNAH_ASR_CAPS=sdot` can take
 * the SMMLA kernel out of a run WITHOUT a rebuild — the sibling's lesson that
 * a kernel you cannot turn off is a kernel you cannot A/B (and the only way to
 * answer "is the new one really what moved the number?"). */
#ifdef MYNAH_ASR_HAVE_SDOT
enum { MYNAH_ASR_ARM_SCALAR = 0, MYNAH_ASR_ARM_SDOT = 1, MYNAH_ASR_ARM_SMMLA = 2 };

static int arm_detect_caps(void) {
#ifdef MYNAH_ASR_HAVE_I8MM
    /* tri-state: only an explicit YES is a yes. An UNKNOWN (a key the kernel
     * does not publish, a leaf a hypervisor hid) means we do not issue SMMLA. */
    if (mynah_asr_cpu_has("i8mm") == MYNAH_ASR_CPU_YES) return MYNAH_ASR_ARM_SMMLA;
#endif
    return MYNAH_ASR_ARM_SDOT;
}

static int g_arm_caps = -1;

static int arm_caps(void) {
    if (g_arm_caps < 0) {
        const char *env = getenv("MYNAH_ASR_CAPS");
        if (env) mynah_asr_set_caps(env);
        if (g_arm_caps < 0) g_arm_caps = arm_detect_caps();
    }
    return g_arm_caps;
}
#endif

int mynah_asr_set_caps(const char *name) {
#ifdef MYNAH_ASR_HAVE_X86
    const int detected = x86_detect_caps();
    int want = detected;
    if (name && strcmp(name, "scalar") == 0) want = MYNAH_ASR_CAPS_SCALAR;
    else if (name && strcmp(name, "avx2") == 0) want = MYNAH_ASR_CAPS_AVX2;
    else if (name && strcmp(name, "vnni") == 0) want = MYNAH_ASR_CAPS_VNNI;
    else if (name && strcmp(name, "auto") != 0)
        fprintf(stderr, "mynah-asr: unknown caps '%s' (scalar|avx2|vnni|auto) -> auto\n", name);
    if (want > detected) {
        fprintf(stderr, "mynah-asr: caps '%s' not supported by this CPU -> level %d\n",
                name, detected);
        want = detected;
    }
    g_x86_caps = want;
    return g_x86_caps;
#elif defined(MYNAH_ASR_HAVE_SDOT)
    const int detected = arm_detect_caps();
    int want = detected;
    if (name && strcmp(name, "scalar") == 0) want = MYNAH_ASR_ARM_SCALAR;
    else if (name && strcmp(name, "sdot") == 0) want = MYNAH_ASR_ARM_SDOT;
    else if (name && strcmp(name, "smmla") == 0) want = MYNAH_ASR_ARM_SMMLA;
    else if (name && strcmp(name, "auto") != 0)
        fprintf(stderr, "mynah-asr: unknown caps '%s' (scalar|sdot|smmla|auto) -> auto\n", name);
    if (want > detected) {
        fprintf(stderr, "mynah-asr: caps '%s' not supported by this CPU -> level %d\n",
                name, detected);
        want = detected;
    }
    g_arm_caps = want;
    return g_arm_caps;
#else
    (void)name;   /* no native int8 kernel in this build: nothing to pick */
    return 0;
#endif
}

/* ------------------------------------------------------------- NEON kernels */
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

/* f32 x int8 dot (whole row, single scale) */
static float dot_q8_neon(const float *x, const int8_t *q, int k) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    for (int j = 0; j < k; j += 16) {
        const int8x16_t qb = vld1q_s8(q + j);
        const int16x8_t lo = vmovl_s8(vget_low_s8(qb));
        const int16x8_t hi = vmovl_s8(vget_high_s8(qb));
        acc0 = vfmaq_f32(acc0, vld1q_f32(x + j),      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
        acc1 = vfmaq_f32(acc1, vld1q_f32(x + j + 4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
        acc2 = vfmaq_f32(acc2, vld1q_f32(x + j + 8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
        acc3 = vfmaq_f32(acc3, vld1q_f32(x + j + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

/* f32 x int4 dot (groups of 32 packed nibbles: byte j = elements 2j | 2j+1<<4).
 * vld2q_f32 deinterleaves x into even/odd lanes, aligned with the lo/hi nibbles. */
static float dot_q4_neon(const float *x, const uint8_t *q, const float *scales,
                         int k) {
    const int8x16_t off = vdupq_n_s8(8);
    const uint8x16_t maskv = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    for (int g = 0; g < k / MYNAH_ASR_Q4_GROUP; g++) {
        const uint8x16_t b = vld1q_u8(q + g * 16);
        const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, maskv)), off);
        const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), off);

        const float32x4x2_t x0 = vld2q_f32(x + g * 32);       /* even/odd 0..7  */
        const float32x4x2_t x1 = vld2q_f32(x + g * 32 + 8);   /* 8..15  */
        const float32x4x2_t x2 = vld2q_f32(x + g * 32 + 16);  /* 16..23 */
        const float32x4x2_t x3 = vld2q_f32(x + g * 32 + 24);  /* 24..31 */

        const int16x8_t lo16a = vmovl_s8(vget_low_s8(lo));
        const int16x8_t lo16b = vmovl_s8(vget_high_s8(lo));
        const int16x8_t hi16a = vmovl_s8(vget_low_s8(hi));
        const int16x8_t hi16b = vmovl_s8(vget_high_s8(hi));

        float32x4_t ga = vmulq_f32(x0.val[0], vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16a))));
        ga = vfmaq_f32(ga, x0.val[1], vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16a))));
        ga = vfmaq_f32(ga, x1.val[0], vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16a))));
        ga = vfmaq_f32(ga, x1.val[1], vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16a))));
        ga = vfmaq_f32(ga, x2.val[0], vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16b))));
        ga = vfmaq_f32(ga, x2.val[1], vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16b))));
        ga = vfmaq_f32(ga, x3.val[0], vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16b))));
        ga = vfmaq_f32(ga, x3.val[1], vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16b))));

        acc += vaddvq_f32(ga) * scales[g];
    }
    return acc;
}
#define MYNAH_ASR_HAVE_NEON 1
#endif

/* --------------------------------------------- int8xint8 GEMM for large T
 * A block of QGEMM_ROWS weight rows per task: each weight row is loaded once and
 * dotted against all T activations (already quantized, hot in cache). Tasks write
 * disjoint columns of out -> bit-identical to the serial version. */
#define QGEMM_ROWS 32

/* ====================================== which int8 micro-kernel runs (S5-1) ==
 * Three predicates, never two (the sibling's rule): COMPILED is a build gate,
 * SUPPORTED is a runtime probe, and USE is the policy that combines them with
 * the caps ladder. `force` is the fourth, and it is a TEST hook only: it lets
 * one process run the same shapes through every compiled kernel and require
 * the results to be bit-identical, which is the only way both arms of a
 * fallback are ever proven in the same binary. */

static int q8_leaf_auto(void) {
#if defined(MYNAH_ASR_HAVE_I8MM)
    if (arm_caps() >= MYNAH_ASR_ARM_SMMLA) return MYNAH_ASR_QK_NEON_SMMLA;
#endif
#if defined(MYNAH_ASR_HAVE_SDOT)
    return arm_caps() >= MYNAH_ASR_ARM_SDOT ? MYNAH_ASR_QK_NEON_SDOT : MYNAH_ASR_QK_SCALAR;
#elif defined(MYNAH_ASR_HAVE_X86)
    if (x86_use_avx512vnni()) return MYNAH_ASR_QK_AVX512VNNI;
#if defined(MYNAH_ASR_HAVE_AVXVNNI_KERNEL)
    if (x86_use_avxvnni()) return MYNAH_ASR_QK_AVXVNNI;
#endif
    return x86_caps() >= MYNAH_ASR_CAPS_AVX2 ? MYNAH_ASR_QK_AVX2 : MYNAH_ASR_QK_SCALAR;
#else
    return MYNAH_ASR_QK_SCALAR;
#endif
}

int mynah_asr_qmat_kernel_available(int which, const char **reason) {
    const char *r = NULL;
    int v = -1;
    switch (which) {
        case MYNAH_ASR_QK_SCALAR:
            v = 1;   /* the f32xint8 per-row fallback is always in the binary */
            break;
        case MYNAH_ASR_QK_DOT_PER_ROW:
            /* the pre-S5-1 loop still runs a NATIVE per-row dot, so it is off
             * at caps=scalar exactly like the kernels that replaced it */
#if defined(MYNAH_ASR_HAVE_SDOT)
            v = arm_caps() >= MYNAH_ASR_ARM_SDOT;
            if (!v) r = "MYNAH_ASR_CAPS=scalar forbids the native int8 dot";
#elif defined(MYNAH_ASR_HAVE_X86)
            v = x86_caps() >= MYNAH_ASR_CAPS_AVX2;
            if (!v) r = "this CPU has no usable AVX2, or MYNAH_ASR_CAPS=scalar";
#else
            r = "no native int8 dot in this build";
#endif
            break;
        case MYNAH_ASR_QK_NEON_SDOT:
#if defined(MYNAH_ASR_HAVE_SDOT)
            v = arm_caps() >= MYNAH_ASR_ARM_SDOT;
            if (!v) r = "MYNAH_ASR_CAPS=scalar forbids the native int8 dot";
#else
            r = "not a dotprod build (__ARM_FEATURE_DOTPROD undefined)";
#endif
            break;
        case MYNAH_ASR_QK_NEON_SMMLA:
#if defined(MYNAH_ASR_HAVE_I8MM)
            v = arm_caps() >= MYNAH_ASR_ARM_SMMLA;
            if (!v)
                r = mynah_asr_cpu_has("i8mm") == MYNAH_ASR_CPU_YES
                        ? "MYNAH_ASR_CAPS caps the ladder below smmla"
                        : "this CPU does not report FEAT_I8MM: compiled here, "
                          "never issued here";
#else
            r = "no i8mm kernel in this build (not aarch64, or a compiler older "
                "than GCC 10)";
#endif
            break;
        case MYNAH_ASR_QK_AVX512VNNI:
#if defined(MYNAH_ASR_HAVE_X86)
            v = x86_use_avx512vnni();
            if (!v) r = "this CPU has no usable AVX-512 VNNI, or MYNAH_ASR_CAPS "
                        "caps the ladder below vnni";
#else
            r = "not an x86 build";
#endif
            break;
        case MYNAH_ASR_QK_AVXVNNI:
#if defined(MYNAH_ASR_HAVE_AVXVNNI_KERNEL)
            v = x86_use_avxvnni();
            if (!v) r = "this CPU has no AVX-VNNI, or AVX-512 VNNI outranks it "
                        "here, or MYNAH_ASR_CAPS caps the ladder below vnni";
#elif defined(MYNAH_ASR_HAVE_X86)
            r = "no AVX-VNNI kernel in this build (compiler older than GCC 11)";
#else
            r = "not an x86 build";
#endif
            break;
        case MYNAH_ASR_QK_AVX2:
#if defined(MYNAH_ASR_HAVE_X86)
            v = x86_caps() >= MYNAH_ASR_CAPS_AVX2;
            if (!v) r = "this CPU has no usable AVX2, or MYNAH_ASR_CAPS=scalar";
#else
            r = "not an x86 build";
#endif
            break;
        default:
            r = "no such kernel id";
            break;
    }
    if (reason) *reason = r;
    return v;
}

/* Test-only pin. Relaxed atomic: it is written once from one thread before a
 * call and read by the pool workers of that call. */
static _Atomic int g_q8_force = -1;

int mynah_asr_qmat_kernel_force(int which) {
    if (which < 0) {
        atomic_store_explicit(&g_q8_force, -1, memory_order_relaxed);
        return q8_leaf_auto();
    }
    if (which >= MYNAH_ASR_QK__N) return -1;
    if (mynah_asr_qmat_kernel_available(which, NULL) != 1) return -1;
    atomic_store_explicit(&g_q8_force, which, memory_order_relaxed);
    return which;
}

static int q8_leaf(void) {
    const int f = atomic_load_explicit(&g_q8_force, memory_order_relaxed);
    return f >= 0 ? f : q8_leaf_auto();
}

int mynah_asr_qmat_kernel_resolved(void) { return q8_leaf(); }

#if defined(MYNAH_ASR_HAVE_SDOT) || defined(MYNAH_ASR_HAVE_X86)
#include "threads.h"

typedef struct {
    const mynah_asr_qmat *m;
    const int8_t *qx;      /* [T, k] int8 activations (x86-q4: pre-permuted) */
    const float *sx;       /* [T] activation scales */
    float *out;            /* [T, n] */
    int T;
    int leaf;              /* MYNAH_ASR_QK_*, resolved ONCE by the caller so no
                            * worker ever calls getenv() or cpuid */
} qgemm_ctx;

/* The weight-stationary sweep, written once and instantiated per leaf: one
 * weight row, FOUR activation rows per pass of the k loop, the T remainder
 * finished by the per-row dot this leaf falls back to (which is also what
 * makes the remainder bit-identical for free). */
#define QMAT_WS_X4(DOTS_X4, DOT_ROW)                                              \
    do {                                                                          \
        const int k = m->k, n = m->n, T = c->T;                                   \
        for (int i = i0; i < i1; i++) {                                           \
            const int8_t *w = m->q8 + (size_t)i * (size_t)k;                      \
            const float ws = m->scales[i];                                        \
            int t = 0;                                                            \
            for (; t + 4 <= T; t += 4) {                                          \
                const int8_t *x0 = c->qx + (size_t)t * (size_t)k;                 \
                int32_t s[4];                                                     \
                DOTS_X4(w, x0, x0 + k, x0 + 2 * k, x0 + 3 * k, k, s);             \
                c->out[(size_t)(t + 0) * (size_t)n + i] =                         \
                    qmat_q8_epilogue(s[0], ws, c->sx[t + 0]);                      \
                c->out[(size_t)(t + 1) * (size_t)n + i] =                         \
                    qmat_q8_epilogue(s[1], ws, c->sx[t + 1]);                      \
                c->out[(size_t)(t + 2) * (size_t)n + i] =                         \
                    qmat_q8_epilogue(s[2], ws, c->sx[t + 2]);                      \
                c->out[(size_t)(t + 3) * (size_t)n + i] =                         \
                    qmat_q8_epilogue(s[3], ws, c->sx[t + 3]);                      \
            }                                                                     \
            for (; t < T; t++)                                                    \
                c->out[(size_t)t * (size_t)n + i] =                               \
                    DOT_ROW(c->qx + (size_t)t * (size_t)k, c->sx[t], w, ws, k);    \
        }                                                                         \
    } while (0)

#if defined(MYNAH_ASR_HAVE_I8MM)
/* The 2x2 tile: two weight rows and two activation rows per SMMLA. The odd
 * activation row and the odd weight row finish on SDOT — a single row has
 * nothing to put in the other half of the tile. */
static void qgemm_q8_smmla(const qgemm_ctx *c, int i0, int i1) {
    const mynah_asr_qmat *m = c->m;
    const int k = m->k, n = m->n, T = c->T;
    int i = i0;
    for (; i + 2 <= i1; i += 2) {
        const int8_t *w0 = m->q8 + (size_t)i * (size_t)k;
        const int8_t *w1 = w0 + k;
        const float ws0 = m->scales[i], ws1 = m->scales[i + 1];
        int t = 0;
        for (; t + 2 <= T; t += 2) {
            const int8_t *x0 = c->qx + (size_t)t * (size_t)k;
            int32_t s[4];   /* { w0.x0, w0.x1, w1.x0, w1.x1 } */
            dots_q8_smmla_2x2(w0, w1, x0, x0 + k, k, s);
            c->out[(size_t)(t + 0) * (size_t)n + i]     = qmat_q8_epilogue(s[0], ws0, c->sx[t]);
            c->out[(size_t)(t + 0) * (size_t)n + i + 1] = qmat_q8_epilogue(s[2], ws1, c->sx[t]);
            c->out[(size_t)(t + 1) * (size_t)n + i]     = qmat_q8_epilogue(s[1], ws0, c->sx[t + 1]);
            c->out[(size_t)(t + 1) * (size_t)n + i + 1] = qmat_q8_epilogue(s[3], ws1, c->sx[t + 1]);
        }
        for (; t < T; t++) {
            const int8_t *x0 = c->qx + (size_t)t * (size_t)k;
            c->out[(size_t)t * (size_t)n + i]     = dot_q8_sdot(x0, c->sx[t], w0, ws0, k);
            c->out[(size_t)t * (size_t)n + i + 1] = dot_q8_sdot(x0, c->sx[t], w1, ws1, k);
        }
    }
    for (; i < i1; i++) {
        const int8_t *w = m->q8 + (size_t)i * (size_t)k;
        const float ws = m->scales[i];
        for (int t = 0; t < T; t++)
            c->out[(size_t)t * (size_t)n + i] =
                dot_q8_sdot(c->qx + (size_t)t * (size_t)k, c->sx[t], w, ws, k);
    }
}
#endif

#if defined(MYNAH_ASR_HAVE_SDOT)
static void qgemm_q8_sdot(const qgemm_ctx *c, int i0, int i1) {
    const mynah_asr_qmat *m = c->m;
    QMAT_WS_X4(dots_q8_sdot_x4, dot_q8_sdot);
}
#endif

#if defined(MYNAH_ASR_HAVE_X86)
static void qgemm_q8_avx512vnni(const qgemm_ctx *c, int i0, int i1) {
    const mynah_asr_qmat *m = c->m;
    QMAT_WS_X4(dots_q8_avx512vnni_x4, dot_q8_vnni);
}
static void qgemm_q8_avx2(const qgemm_ctx *c, int i0, int i1) {
    const mynah_asr_qmat *m = c->m;
    QMAT_WS_X4(dots_q8_avx2_x4, dot_q8_avx2);
}
#if defined(MYNAH_ASR_HAVE_AVXVNNI_KERNEL)
/* The VEX twin's remainder goes through the AVX2 dot, not through dot_q8_vnni:
 * an AVX-VNNI part need not have AVX-512 at all. */
static void qgemm_q8_avxvnni(const qgemm_ctx *c, int i0, int i1) {
    const mynah_asr_qmat *m = c->m;
    QMAT_WS_X4(dots_q8_avxvnni_x4, dot_q8_avx2);
}
#endif
#endif

static void qgemm_block(void *ctx, int blk) {
    const qgemm_ctx *c = ctx;
    const mynah_asr_qmat *m = c->m;
    const int i0 = blk * QGEMM_ROWS;
    const int i1 = i0 + QGEMM_ROWS < m->n ? i0 + QGEMM_ROWS : m->n;
    if (m->qtype == MYNAH_ASR_Q_INT8) {
        qk(c->leaf);
        switch (c->leaf) {
#if defined(MYNAH_ASR_HAVE_I8MM)
            case MYNAH_ASR_QK_NEON_SMMLA: qgemm_q8_smmla(c, i0, i1); return;
#endif
#if defined(MYNAH_ASR_HAVE_SDOT)
            case MYNAH_ASR_QK_NEON_SDOT:  qgemm_q8_sdot(c, i0, i1); return;
#endif
#if defined(MYNAH_ASR_HAVE_X86)
            case MYNAH_ASR_QK_AVX512VNNI: qgemm_q8_avx512vnni(c, i0, i1); return;
            case MYNAH_ASR_QK_AVX2:       qgemm_q8_avx2(c, i0, i1); return;
#if defined(MYNAH_ASR_HAVE_AVXVNNI_KERNEL)
            case MYNAH_ASR_QK_AVXVNNI:    qgemm_q8_avxvnni(c, i0, i1); return;
#endif
#endif
            default: break;   /* MYNAH_ASR_QK_DOT_PER_ROW: the pre-S5-1 loop */
        }
    }
    for (int i = i0; i < i1; i++) {
        if (m->qtype == MYNAH_ASR_Q_INT8) {
            const int8_t *qrow = m->q8 + (size_t)i * (size_t)m->k;
            const float ws = m->scales[i];
            for (int t = 0; t < c->T; t++) {
                const int8_t *qxt = c->qx + (size_t)t * (size_t)m->k;
#ifdef MYNAH_ASR_HAVE_SDOT
                c->out[(size_t)t * (size_t)m->n + i] = dot_q8_sdot(qxt, c->sx[t], qrow, ws, m->k);
#else
                c->out[(size_t)t * (size_t)m->n + i] = x86_use_avx512vnni()
                    ? dot_q8_vnni(qxt, c->sx[t], qrow, ws, m->k)
                    : dot_q8_avx2(qxt, c->sx[t], qrow, ws, m->k);
#endif
            }
        } else {
            const int groups = m->k / MYNAH_ASR_Q4_GROUP;
            const uint8_t *qrow = m->q4 + (size_t)i * (size_t)(m->k / 2);
            const float *srow = m->scales + (size_t)i * (size_t)groups;
            for (int t = 0; t < c->T; t++) {
                const int8_t *qxt = c->qx + (size_t)t * (size_t)m->k;
#ifdef MYNAH_ASR_HAVE_SDOT
                c->out[(size_t)t * (size_t)m->n + i] = dot_q4_sdot(qxt, c->sx[t], qrow, srow, m->k);
#else
                c->out[(size_t)t * (size_t)m->n + i] = dot_q4_x86(qxt, c->sx[t], qrow, srow, m->k);
#endif
            }
        }
    }
}
#endif

/* ------------------------------------------------- dispatch predicates (S3-2)
 * The dispatch report NEVER re-derives which kernel runs from "compiled &&
 * supported": it asks the owner of the decision, and the owner is this file.
 * Everything below is a read of the same macros and the same cached x86 level
 * that mynah_asr_qmat_mul() branches on a few lines further down, so the answer
 * cannot drift from the code that produces the numbers. No behaviour change:
 * these are pure readers. */

/* The MYNAH_ASR_QGEMM gate, hoisted out of mynah_asr_qmat_mul so the report and
 * the hot path read ONE definition (and one cached value). */
#if defined(MYNAH_ASR_HAVE_SDOT) || defined(MYNAH_ASR_HAVE_X86)
static int qmat_qgemm_env(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("MYNAH_ASR_QGEMM");
        v = e && e[0] == '1';
    }
    return v;
}
#endif

/* 1 = the threaded int8xint8 GEMM may run, 0 = off by default or by the env,
 * -1 = this build has no native int8 kernel to run it with. */
int mynah_asr_qmat_qgemm(void) {
#if defined(MYNAH_ASR_HAVE_SDOT) || defined(MYNAH_ASR_HAVE_X86)
    return qmat_qgemm_env();
#else
    return -1;
#endif
}

const char *mynah_asr_caps_name(int level) {
#ifdef MYNAH_ASR_HAVE_X86
    switch (level) {
        case 0: return "scalar";
        case 1: return "avx2";
        case 2: return "vnni";
        default: return "n/a";
    }
#elif defined(MYNAH_ASR_HAVE_SDOT)
    switch (level) {
        case 0: return "scalar";
        case 1: return "sdot";
        case 2: return "smmla";
        default: return "n/a";
    }
#else
    (void)level;
    return "n/a";
#endif
}

int mynah_asr_caps_detected(void) {
#ifdef MYNAH_ASR_HAVE_X86
    return x86_detect_caps();
#elif defined(MYNAH_ASR_HAVE_SDOT)
    return arm_detect_caps();   /* sdot always, smmla when FEAT_I8MM says YES */
#else
    return -1;   /* no native int8 kernel in this build: no level to pick */
#endif
}

int mynah_asr_caps_effective(void) {
#ifdef MYNAH_ASR_HAVE_X86
    return x86_caps();   /* reads MYNAH_ASR_CAPS once, exactly as the kernels do */
#elif defined(MYNAH_ASR_HAVE_SDOT)
    return arm_caps();
#else
    return -1;
#endif
}

/* The int8 micro-kernel this host resolves to, from the S5-1 vocabulary:
 * neon-smmla | neon-sdot | avx512vnni | avxvnni | avx2 | neon-f32 | scalar.
 *
 * It answers by CALLING the same q8_leaf() the block worker branches on, so
 * the report cannot drift from the arithmetic. One nuance the dispatch row's
 * `reason` has to carry: SMMLA is issued by the weight-stationary batched pass
 * only — a one-row dot would leave half the 2x2 tile empty, so with i8mm
 * present the single-row projection still runs SDOT. */
const char *mynah_asr_qmat_int8_kernel(void) {
#if defined(MYNAH_ASR_HAVE_SDOT) || defined(MYNAH_ASR_HAVE_X86)
    const int leaf = q8_leaf();
    if (leaf != MYNAH_ASR_QK_SCALAR && leaf != MYNAH_ASR_QK_DOT_PER_ROW)
        return mynah_asr_qmat_kernel_name(leaf);
#endif
#if defined(MYNAH_ASR_HAVE_NEON)
    return "neon-f32";      /* widen + FMA: vectorized, but no integer unit */
#else
    return "scalar";
#endif
}

/* The one-row dot: the T<=QMAT_SMALL_T branch of mynah_asr_qmat_mul and the
 * T remainder of every weight-stationary sweep. Never SMMLA (see above). */
const char *mynah_asr_qmat_int8_dot_kernel(void) {
#if defined(MYNAH_ASR_HAVE_SDOT)
    return arm_caps() >= MYNAH_ASR_ARM_SDOT ? "neon-sdot" : "neon-f32";
#elif defined(MYNAH_ASR_HAVE_X86)
    if (x86_use_avx512vnni()) return "avx512vnni";
    if (x86_caps() >= MYNAH_ASR_CAPS_AVX2) return "avx2";
    return "scalar";
#elif defined(MYNAH_ASR_HAVE_NEON)
    return "neon-f32";
#else
    return "scalar";
#endif
}

const char *mynah_asr_qmat_int4_kernel(void) {
#if defined(MYNAH_ASR_HAVE_SDOT)
    return "neon-sdot-q4";
#elif defined(MYNAH_ASR_HAVE_X86)
    const int c = x86_caps();
    if (c >= MYNAH_ASR_CAPS_AVX2) return "avx2-q4";   /* no VNNI q4 kernel exists */
    return "scalar";
#elif defined(MYNAH_ASR_HAVE_NEON)
    return "neon-f32-q4";
#else
    return "scalar";
#endif
}

void mynah_asr_qmat_mul(const mynah_asr_qmat *m, const float *x, float *out, int T) {
    if (m->qtype == MYNAH_ASR_Q_F32) {
        qc(MYNAH_ASR_QC_F32);
        mynah_asr_gemm_wt(x, m->f32, out, T, m->n, m->k);
        return;
    }
    if (T <= QMAT_SMALL_T) {
        /* native int8xint8 dot (SDOT at compile time on ARM; VNNI/AVX2 at
         * RUNTIME on x86): quantize the activations once per row (per-row
         * absmax, qwen-tts recipe) */
#if defined(MYNAH_ASR_HAVE_SDOT) || defined(MYNAH_ASR_HAVE_X86)
#ifdef MYNAH_ASR_HAVE_X86
        const int caps = x86_caps();
        const int native = caps >= MYNAH_ASR_CAPS_AVX2 && m->k <= QMAT_K_MAX;
#else
        const int native = arm_caps() >= MYNAH_ASR_ARM_SDOT && m->k <= QMAT_K_MAX;
#endif
        if (native) {
            qc(MYNAH_ASR_QC_DOT);
            /* one activation row at a time: SMMLA would leave half its tile
             * empty, so this path stays on SDOT/VNNI/AVX2 even on a host whose
             * batched pass is SMMLA. The counter says so rather than letting a
             * reader assume the two paths share a kernel. */
            if (m->qtype == MYNAH_ASR_Q_INT8)
#if defined(MYNAH_ASR_HAVE_SDOT)
                qk(MYNAH_ASR_QK_NEON_SDOT);
#else
                qk(x86_use_avx512vnni() ? MYNAH_ASR_QK_AVX512VNNI : MYNAH_ASR_QK_AVX2);
#endif
            int8_t qx[QMAT_K_MAX];
            for (int t = 0; t < T; t++) {
                const float *xr = x + (size_t)t * (size_t)m->k;
                float *o = out + (size_t)t * (size_t)m->n;
                const float sx = quantize_act_int8(qx, xr, m->k);
                if (m->qtype == MYNAH_ASR_Q_INT8) {
                    for (int i = 0; i < m->n; i++) {
                        const int8_t *qrow = m->q8 + (size_t)i * (size_t)m->k;
#ifdef MYNAH_ASR_HAVE_SDOT
                        o[i] = dot_q8_sdot(qx, sx, qrow, m->scales[i], m->k);
#else
                        /* x86_use_avx512vnni() and not `caps >= VNNI`: since
                         * S5-1 the ladder's top rung also covers AVX-VNNI
                         * parts, and issuing an EVEX kernel there is a SIGILL */
                        o[i] = x86_use_avx512vnni()
                                   ? dot_q8_vnni(qx, sx, qrow, m->scales[i], m->k)
                                   : dot_q8_avx2(qx, sx, qrow, m->scales[i], m->k);
#endif
                    }
                } else {
                    const int groups = m->k / MYNAH_ASR_Q4_GROUP;
#ifdef MYNAH_ASR_HAVE_SDOT
                    for (int i = 0; i < m->n; i++)
                        o[i] = dot_q4_sdot(qx, sx, m->q4 + (size_t)i * (size_t)(m->k / 2),
                                           m->scales + (size_t)i * (size_t)groups, m->k);
#else
                    int8_t xp[QMAT_K_MAX];
                    q4_permute_act(qx, xp, m->k);
                    for (int i = 0; i < m->n; i++)
                        o[i] = dot_q4_x86(xp, sx, m->q4 + (size_t)i * (size_t)(m->k / 2),
                                          m->scales + (size_t)i * (size_t)groups, m->k);
#endif
                }
            }
            return;
        }
#endif
        qc(MYNAH_ASR_QC_GENERIC);
        if (m->qtype == MYNAH_ASR_Q_INT8) {
            for (int t = 0; t < T; t++) {
                const float *xr = x + (size_t)t * (size_t)m->k;
                float *o = out + (size_t)t * (size_t)m->n;
                for (int i = 0; i < m->n; i++) {
                    const int8_t *qrow = m->q8 + (size_t)i * (size_t)m->k;
#ifdef MYNAH_ASR_HAVE_NEON
                    o[i] = dot_q8_neon(xr, qrow, m->k) * m->scales[i];
#else
                    float acc = 0.0f;
                    for (int j = 0; j < m->k; j++) acc += xr[j] * (float)qrow[j];
                    o[i] = acc * m->scales[i];
#endif
                }
            }
        } else {
            for (int t = 0; t < T; t++) {
                const float *xr = x + (size_t)t * (size_t)m->k;
                float *o = out + (size_t)t * (size_t)m->n;
                const int groups = m->k / MYNAH_ASR_Q4_GROUP;
                for (int i = 0; i < m->n; i++) {
                    const uint8_t *qrow = m->q4 + (size_t)i * (size_t)(m->k / 2);
                    const float *srow = m->scales + (size_t)i * (size_t)groups;
#ifdef MYNAH_ASR_HAVE_NEON
                    o[i] = dot_q4_neon(xr, qrow, srow, m->k);
#else
                    float acc = 0.0f;
                    for (int g = 0; g < groups; g++) {
                        float ga = 0.0f;
                        const float *xg = xr + g * MYNAH_ASR_Q4_GROUP;
                        for (int j = 0; j < MYNAH_ASR_Q4_GROUP; j += 2) {
                            const uint8_t b = qrow[(g * MYNAH_ASR_Q4_GROUP + j) / 2];
                            ga += xg[j] * (float)((int)(b & 0x0F) - 8);
                            ga += xg[j + 1] * (float)((int)(b >> 4) - 8);
                        }
                        acc += ga * srow[g];
                    }
                    o[i] = acc;
#endif
                }
            }
        }
        return;
    }
    /* Large T, OPT-IN through env MYNAH_ASR_QGEMM=1: threaded int8xint8 GEMM —
     * activations quantized ONCE (T int8 rows + scales), parallel-for in blocks
     * over the weight rows (the weight is read once as int8 = 4x less bandwidth
     * than the f32 dequant, activations hot in cache).
     * MEASURED 2026-07-18 on M-series: it LOSES against dequant+sgemm (the AMX
     * inside Accelerate dominates) and the activation quant changes the numerics
     * -> default OFF. The expected upside is on x86 VNNI (no AMX): validate
     * there before considering a per-platform default. */
#if defined(MYNAH_ASR_HAVE_SDOT) || defined(MYNAH_ASR_HAVE_X86)
    const int g_qgemm = qmat_qgemm_env();
#ifdef MYNAH_ASR_HAVE_X86
    const int gnative = g_qgemm && x86_caps() >= MYNAH_ASR_CAPS_AVX2 && m->k <= QMAT_K_MAX;
#else
    /* arm_caps() and not just the compile gate: MYNAH_ASR_CAPS=scalar has to
     * take the native int8 kernels out of THIS path too, or the opt-out means
     * something different depending on which entry the caller used */
    const int gnative = g_qgemm && arm_caps() >= MYNAH_ASR_ARM_SDOT && m->k <= QMAT_K_MAX;
#endif
    if (gnative) {
        int8_t *qx = malloc((size_t)T * (size_t)m->k);
        float *sx = malloc((size_t)T * sizeof(float));
        if (qx && sx) {
            for (int t = 0; t < T; t++)
                sx[t] = quantize_act_int8(qx + (size_t)t * (size_t)m->k,
                                          x + (size_t)t * (size_t)m->k, m->k);
#if defined(MYNAH_ASR_HAVE_X86) && !defined(MYNAH_ASR_HAVE_SDOT)
            if (m->qtype == MYNAH_ASR_Q_INT4) {
                /* the q4 AVX2 kernel wants the activations pre-permuted */
                for (int t = 0; t < T; t++) {
                    int8_t xp[QMAT_K_MAX];
                    q4_permute_act(qx + (size_t)t * (size_t)m->k, xp, m->k);
                    memcpy(qx + (size_t)t * (size_t)m->k, xp, (size_t)m->k);
                }
            }
#endif
            qc(MYNAH_ASR_QC_QGEMM);
            qgemm_ctx c = {.m = m, .qx = qx, .sx = sx, .out = out, .T = T,
                           .leaf = q8_leaf()};
            mynah_asr_parallel_for((m->n + QGEMM_ROWS - 1) / QGEMM_ROWS, qgemm_block, &c);
            free(qx);
            free(sx);
            return;
        }
        free(qx);
        free(sx);
    }
#endif
    /* per-call dequant + GEMM: fallback (no native kernels available) */
    qc(MYNAH_ASR_QC_DEQUANT);
    float *wd = malloc((size_t)m->n * (size_t)m->k * sizeof(float));
    if (!wd) return;
    for (int i = 0; i < m->n; i++) dequant_row(m, i, wd + (size_t)i * (size_t)m->k);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, m->n, m->k,
                1.0f, x, m->k, wd, m->k, 0.0f, out, m->n);
    free(wd);
}

/* ----------------------------------------------- row-stable stacked product
 * See qmat.h for the contract. The point of this entry is that it is the ONLY
 * one the batched stream step calls: mynah_asr_qmat_mul above keeps its own
 * dispatch (and therefore the offline numerics) untouched. */
int mynah_asr_qmat_mul_rows(const mynah_asr_qmat *m, const float *x, float *out, int T,
                        int8_t *qx, float *sx) {
    if (T <= 0) return MYNAH_ASR_QC_DOT_ROWS;
    if (m->qtype == MYNAH_ASR_Q_F32) {
        qc(MYNAH_ASR_QC_F32);
        mynah_asr_gemm_wt(x, m->f32, out, T, m->n, m->k);
        return MYNAH_ASR_QC_F32;
    }
#if defined(MYNAH_ASR_HAVE_SDOT) || defined(MYNAH_ASR_HAVE_X86)
    const int leaf = q8_leaf();
    int native = m->k <= QMAT_K_MAX && qx != NULL && sx != NULL;
#ifdef MYNAH_ASR_HAVE_X86
    native = native && x86_caps() >= MYNAH_ASR_CAPS_AVX2;
#else
    native = native && arm_caps() >= MYNAH_ASR_ARM_SDOT;
#endif
    /* MYNAH_ASR_CAPS=scalar (or a pinned scalar leaf) takes int8 off the native
     * kernels entirely; int4 keeps its own gate, it has no S5-1 kernel. */
    if (m->qtype == MYNAH_ASR_Q_INT8 && leaf == MYNAH_ASR_QK_SCALAR) native = 0;
    if (native) {
        qc(MYNAH_ASR_QC_DOT_ROWS);
        /* Exactly the activation quantization of the small-T path, per row —
         * on the pool above the threshold, because it is the SERIAL FRACTION of
         * an int8 call and the sibling measured it staying flat at 3.8 ms while
         * the GEMM next to it threaded 16.7 -> 7.7 ms. Each row is independent
         * (per-row absmax), so the pool changes nothing numerically. */
        quantize_act_rows(qx, sx, x, T, m->k);
#if defined(MYNAH_ASR_HAVE_X86) && !defined(MYNAH_ASR_HAVE_SDOT)
        if (m->qtype == MYNAH_ASR_Q_INT4) {
            /* the q4 AVX2 kernel wants the activations pre-permuted */
            for (int t = 0; t < T; t++) {
                int8_t xp[QMAT_K_MAX];
                q4_permute_act(qx + (size_t)t * (size_t)m->k, xp, m->k);
                memcpy(qx + (size_t)t * (size_t)m->k, xp, (size_t)m->k);
            }
        }
#endif
        qgemm_ctx c = {.m = m, .qx = qx, .sx = sx, .out = out, .T = T, .leaf = leaf};
        mynah_asr_parallel_for((m->n + QGEMM_ROWS - 1) / QGEMM_ROWS, qgemm_block, &c);
        return MYNAH_ASR_QC_DOT_ROWS;
    }
#else
    (void)qx; (void)sx;
#endif
    /* No native kernel (or k > QMAT_K_MAX): T products of ONE row each. Bit-exact
     * against the single path by construction — it IS the single path, T times —
     * and it still never reaches the dequant+malloc fallback. */
    for (int t = 0; t < T; t++)
        mynah_asr_qmat_mul(m, x + (size_t)t * (size_t)m->k,
                           out + (size_t)t * (size_t)m->n, 1);
    return MYNAH_ASR_QC_GENERIC;
}

/* ---------------------------------------------------------- fused helpers */
void mynah_asr_qmat_ffn(const mynah_asr_qmat *w1, const mynah_asr_qmat *w2, const float *x,
                    float *out, int T, float *scratch) {
    if (w1->qtype == MYNAH_ASR_Q_F32 && w2->qtype == MYNAH_ASR_Q_F32) {
        mynah_asr_ffn_wt(x, w1->f32, w1->n, w2->f32, w2->n, out, T, w1->k, scratch);
        return;
    }
    mynah_asr_qmat_mul(w1, x, scratch, T);
    mynah_asr_silu(scratch, (size_t)T * (size_t)w1->n);
    mynah_asr_qmat_mul(w2, scratch, out, T);
}

void mynah_asr_qmat_qkv(const mynah_asr_qmat *wq, const mynah_asr_qmat *wk, const mynah_asr_qmat *wv,
                    const float *x, float *oq, float *ok, float *ov, int T) {
    if (wq->qtype == MYNAH_ASR_Q_F32 && wk->qtype == MYNAH_ASR_Q_F32 && wv->qtype == MYNAH_ASR_Q_F32) {
        mynah_asr_gemm3_wt(x, wq->f32, wk->f32, wv->f32, oq, ok, ov, T, wq->n, wq->k);
        return;
    }
    mynah_asr_qmat_mul(wq, x, oq, T);
    mynah_asr_qmat_mul(wk, x, ok, T);
    mynah_asr_qmat_mul(wv, x, ov, T);
}
