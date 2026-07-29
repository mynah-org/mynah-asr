#include "qmat.h"

#include "backend.h"

#include <math.h>
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
        if (k % MYNAH_ASR_Q4_GROUP != 0) return 0;   /* resta f32 */
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
        if (tq && ts) {                       /* pre-quantizzato: zero-copy dal mmap */
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
    return (float)s * ws * sx;
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
        const int8x16x2_t xg = vld2q_s8(qx + g * 32);  /* val[0]=pari, val[1]=dispari */
        int32x4_t ig = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, xg.val[0]), hi, xg.val[1]);
        acc += (float)vaddvq_s32(ig) * scales[g];
    }
    return acc * sx;
}
#define MYNAH_ASR_HAVE_SDOT 1
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

static int x86_detect_caps(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 27))) /* OSXSAVE */
        return MYNAH_ASR_CAPS_SCALAR;
    const unsigned long long xcr0 = xgetbv0();
    if ((xcr0 & 0x6) != 0x6) return MYNAH_ASR_CAPS_SCALAR;        /* xmm+ymm dall'OS */
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d) || !((b >> 5) & 1)) /* AVX2 */
        return MYNAH_ASR_CAPS_SCALAR;
    const int avx512f = (b >> 16) & 1, avx512bw = (b >> 30) & 1, vnni = (c >> 11) & 1;
    if (avx512f && avx512bw && vnni && (xcr0 & 0xE0) == 0xE0) /* zmm+opmask */
        return MYNAH_ASR_CAPS_VNNI;
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
    return (float)s * ws * sx;
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
        const __m256i aw = _mm256_sign_epi8(wv, wv);   /* |w| come u8 */
        const __m256i sxv = _mm256_sign_epi8(xv, wv);  /* qx * sign(w) */
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sxv), ones16));
    }
    __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    int s = _mm_cvtsi128_si32(lo);
    for (; j < k; j++) s += (int)w[j] * qx[j];
    return (float)s * ws * sx;
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
static float dot_q4_x86(const int8_t *xp /* permutate */, float sx, const uint8_t *q,
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
#endif /* x86 */

int mynah_asr_set_caps(const char *name) {
#ifdef MYNAH_ASR_HAVE_X86
    const int detected = x86_detect_caps();
    int want = detected;
    if (name && strcmp(name, "scalar") == 0) want = MYNAH_ASR_CAPS_SCALAR;
    else if (name && strcmp(name, "avx2") == 0) want = MYNAH_ASR_CAPS_AVX2;
    else if (name && strcmp(name, "vnni") == 0) want = MYNAH_ASR_CAPS_VNNI;
    else if (name && strcmp(name, "auto") != 0)
        fprintf(stderr, "mynah-asr: caps ignoti '%s' (scalar|avx2|vnni|auto) -> auto\n", name);
    if (want > detected) {
        fprintf(stderr, "mynah-asr: caps '%s' not supported by this CPU -> level %d\n",
                name, detected);
        want = detected;
    }
    g_x86_caps = want;
    return g_x86_caps;
#else
    (void)name;   /* ARM: NEON/SDOT are compile-time (Apple always has dotprod) */
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

        const float32x4x2_t x0 = vld2q_f32(x + g * 32);       /* pari/dispari 0..7  */
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

#if defined(MYNAH_ASR_HAVE_SDOT) || defined(MYNAH_ASR_HAVE_X86)
#include "threads.h"

typedef struct {
    const mynah_asr_qmat *m;
    const int8_t *qx;      /* [T, k] attivazioni int8 (x86-q4: pre-permutate) */
    const float *sx;       /* [T] scale attivazioni */
    float *out;            /* [T, n] */
    int T;
} qgemm_ctx;

static void qgemm_block(void *ctx, int blk) {
    const qgemm_ctx *c = ctx;
    const mynah_asr_qmat *m = c->m;
    const int i0 = blk * QGEMM_ROWS;
    const int i1 = i0 + QGEMM_ROWS < m->n ? i0 + QGEMM_ROWS : m->n;
#ifdef MYNAH_ASR_HAVE_X86
    const int caps = x86_caps();
#endif
    for (int i = i0; i < i1; i++) {
        if (m->qtype == MYNAH_ASR_Q_INT8) {
            const int8_t *qrow = m->q8 + (size_t)i * (size_t)m->k;
            const float ws = m->scales[i];
            for (int t = 0; t < c->T; t++) {
                const int8_t *qxt = c->qx + (size_t)t * (size_t)m->k;
#ifdef MYNAH_ASR_HAVE_SDOT
                c->out[(size_t)t * (size_t)m->n + i] = dot_q8_sdot(qxt, c->sx[t], qrow, ws, m->k);
#else
                c->out[(size_t)t * (size_t)m->n + i] = caps >= MYNAH_ASR_CAPS_VNNI
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

void mynah_asr_qmat_mul(const mynah_asr_qmat *m, const float *x, float *out, int T) {
    if (m->qtype == MYNAH_ASR_Q_F32) {
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
        const int native = m->k <= QMAT_K_MAX;
#endif
        if (native) {
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
                        o[i] = caps >= MYNAH_ASR_CAPS_VNNI
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
    static int g_qgemm = -1;
    if (g_qgemm < 0) {
        const char *e = getenv("MYNAH_ASR_QGEMM");
        g_qgemm = e && e[0] == '1';
    }
#ifdef MYNAH_ASR_HAVE_X86
    const int gnative = g_qgemm && x86_caps() >= MYNAH_ASR_CAPS_AVX2 && m->k <= QMAT_K_MAX;
#else
    const int gnative = g_qgemm && m->k <= QMAT_K_MAX;
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
            qgemm_ctx c = {.m = m, .qx = qx, .sx = sx, .out = out, .T = T};
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
    float *wd = malloc((size_t)m->n * (size_t)m->k * sizeof(float));
    if (!wd) return;
    for (int i = 0; i < m->n; i++) dequant_row(m, i, wd + (size_t)i * (size_t)m->k);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, m->n, m->k,
                1.0f, x, m->k, wd, m->k, 0.0f, out, m->n);
    free(wd);
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
