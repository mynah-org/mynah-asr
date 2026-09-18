/* Self-test of the quantized kernels (no model required — it runs in CI too).
 *
 * TWO GATES, and they answer different questions.
 *
 * 1. QUANTIZATION ERROR (the original gate): the small-T path and the
 *    dequant+GEMM path against an f32 reference, with the tolerance the format
 *    itself costs (q8 ~1%, q4 ~4%). A kernel bug — signs, lane order,
 *    saturation — lands orders of magnitude outside it.
 *
 * 2. KERNEL IDENTITY (S5-1): every int8 micro-kernel compiled into this binary
 *    must produce EXACTLY the same int32 accumulator and EXACTLY the same f32
 *    output as the per-row dot, for the same (weight row, activation row).
 *    Not a tolerance — `==`. Integer accumulation is exact and order-free, so
 *    anything else is a bug; and the requirement is architectural, not
 *    cosmetic: a row whose answer depended on which micro-kernel or which tile
 *    position served it would make a transcript depend on who the stream was
 *    batched with (ENGINEERING.md §9, PLAN.md contract 4).
 *
 *    Both arms run IN ONE PROCESS through mynah_asr_qmat_kernel_force(), so a
 *    fallback is proven by execution and not by reading the dispatcher. A
 *    kernel this host cannot run is printed as SKIPPED with the reason — never
 *    silently counted as a pass, which is how a gate comes to guard nothing.
 *
 * Usage: test_qmat [--bench]
 * Exit: 0 ok, 1 fail. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/qmat.h"

static unsigned long rng_state = 42;
static float frand(void) { /* xorshift, reproducible everywhere */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return ((float)(rng_state % 200001) / 100000.0f) - 1.0f;
}

static double rel_err(const float *a, const float *ref, int n) {
    double d2 = 0.0, r2 = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)ref[i];
        d2 += d * d;
        r2 += (double)ref[i] * (double)ref[i];
    }
    return sqrt(d2 / (r2 > 0 ? r2 : 1.0));
}

static int check(const char *name, int qtype, int n, int k, int T, double tol) {
    float *w = malloc((size_t)n * (size_t)k * sizeof(float));
    float *x = malloc((size_t)T * (size_t)k * sizeof(float));
    float *ref = calloc((size_t)T * (size_t)n, sizeof(float));
    float *out = calloc((size_t)T * (size_t)n, sizeof(float));
    for (int i = 0; i < n * k; i++) w[i] = frand();
    for (int i = 0; i < T * k; i++) x[i] = frand();

    for (int t = 0; t < T; t++)          /* f32 reference computed in double */
        for (int i = 0; i < n; i++) {
            double acc = 0.0;
            for (int j = 0; j < k; j++)
                acc += (double)x[t * k + j] * (double)w[(size_t)i * (size_t)k + j];
            ref[t * n + i] = (float)acc;
        }

    mynah_asr_qmat m;
    mynah_asr_qmat_init(&m, w, n, k, qtype);
    mynah_asr_qmat_mul(&m, x, out, T);
    const double e = rel_err(out, ref, T * n);
    printf("%-22s n=%d k=%d T=%-3d rel_err=%.4f (tol %.3f) %s\n",
           name, n, k, T, e, tol, e <= tol ? "OK" : "FAIL");
    mynah_asr_qmat_free(&m);
    free(w); free(x); free(ref); free(out);
    return e <= tol ? 0 : 1;
}

/* ===================================================== gate 2: identity ==== */

/* Every kernel id the identity gate walks, reference arm FIRST: that arm is
 * the pre-S5-1 per-row dot loop, i.e. the numerics the runtime already ships
 * and the transcripts in tests/golden were produced with. */
static const int KERNELS[] = {
    MYNAH_ASR_QK_DOT_PER_ROW,
    MYNAH_ASR_QK_NEON_SMMLA,
    MYNAH_ASR_QK_NEON_SDOT,
    MYNAH_ASR_QK_AVX512VNNI,
    MYNAH_ASR_QK_AVXVNNI,
    MYNAH_ASR_QK_AVX2,
};
#define N_KERNELS ((int)(sizeof(KERNELS) / sizeof(KERNELS[0])))

/* The adversarial byte alphabet. -128 exists only on the weight side: the
 * activation quantizer clamps to [-127,127] by construction, so a test that
 * fed -128 activations would be testing a value the runtime cannot produce. */
static const int W_EXTREMES[] = {-128, -127, -1, 0, 1, 127};
static const int X_EXTREMES[] = {-127, -1, 0, 1, 127};

/* T values: around every unroll boundary this file's kernels have (4 for the
 * x4 sweeps, 2 for the SMMLA tile, 16 for QMAT_SMALL_T) and past them. */
static const int TS[] = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64};
#define N_TS ((int)(sizeof(TS) / sizeof(TS[0])))

typedef struct { int n, k; } shape;
/* k off every unroll (16 SDOT / 8 SMMLA / 32 AVX2 / 64 AVX-512), n off the
 * 32-row block and off the SMMLA row pair, then the real batched-step shapes. */
static const shape SHAPES[] = {
    {33, 7}, {33, 37}, {96, 64}, {31, 65}, {65, 255}, {64, 1024},
};
#define N_SHAPES ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))

/* PASS 1 — the integers.
 *
 * Weight scales are forced to 1 and the activations are built so their per-row
 * absmax is exactly 127, which makes the activation scale exactly 1: the f32
 * output is then (float)s, exact for |s| < 2^24, so a difference here is an
 * INTEGER difference and nothing else. The bound is asserted, not assumed —
 * a shape that overflowed it would turn this gate into a float gate without
 * saying so. */
static int identity_ints(int kern, int n, int k, int T, const char **why) {
    int fail = 0;
    int8_t *q8 = malloc((size_t)n * (size_t)k);
    float *ones = malloc((size_t)n * sizeof(float));
    float *x = malloc((size_t)T * (size_t)k * sizeof(float));
    int8_t *qx = malloc((size_t)T * (size_t)k);
    float *sx = malloc((size_t)T * sizeof(float));
    float *out = calloc((size_t)T * (size_t)n, sizeof(float));
    if (!q8 || !ones || !x || !qx || !sx || !out) { *why = "out of memory"; fail = 1; goto done; }

    for (size_t i = 0; i < (size_t)n * (size_t)k; i++)
        q8[i] = (int8_t)W_EXTREMES[(i * 7u + i / (size_t)k) % 6u];
    for (int i = 0; i < n; i++) ones[i] = 1.0f;
    for (int t = 0; t < T; t++) {
        float *row = x + (size_t)t * (size_t)k;
        for (int j = 0; j < k; j++)
            row[j] = (float)X_EXTREMES[((size_t)j * 3u + (size_t)t) % 5u];
        row[0] = 127.0f;        /* pins this row's absmax, hence sx == 1 */
    }

    mynah_asr_qmat m;
    memset(&m, 0, sizeof(m));
    m.q8 = q8;
    m.scales = ones;
    m.qtype = MYNAH_ASR_Q_INT8;
    m.n = n;
    m.k = k;
    mynah_asr_qmat_mul_rows(&m, x, out, T, qx, sx);

    for (int t = 0; t < T && !fail; t++) {
        if (sx[t] != 1.0f) { *why = "the activation scale is not exactly 1"; fail = 1; break; }
        for (int i = 0; i < n; i++) {
            long long s = 0;
            const int8_t *w = q8 + (size_t)i * (size_t)k;
            const int8_t *a = qx + (size_t)t * (size_t)k;
            for (int j = 0; j < k; j++) s += (long long)w[j] * (long long)a[j];
            if (s > 16777215LL || s < -16777215LL) {
                *why = "the reference sum left the exactly-representable float range";
                fail = 1;
                break;
            }
            if (out[(size_t)t * (size_t)n + i] != (float)s) {
                static char buf[192];
                snprintf(buf, sizeof(buf),
                         "int32 differs at row %d activation %d: scalar %lld vs kernel %.1f",
                         i, t, s, (double)out[(size_t)t * (size_t)n + i]);
                *why = buf;
                fail = 1;
                break;
            }
        }
    }
done:
    (void)kern;
    free(q8); free(ones); free(x); free(qx); free(sx); free(out);
    return fail;
}

/* PASS 2 — the float epilogue, against the reference arm.
 *
 * Real per-row weight scales, a real per-row activation scale: the arithmetic
 * the serving path does. Bit-identical, not bounded — the sibling's incident
 * was exactly here, two kernels whose textually identical `s * ws * sx` the
 * optimizer grouped differently, which made a row's answer depend on where it
 * sat in the batch. */
static int identity_floats(int kern, int n, int k, int T, const float *w,
                           const float *x, const float *ref, const char **why) {
    int fail = 0;
    int8_t *qx = malloc((size_t)T * (size_t)k);
    float *sx = malloc((size_t)T * sizeof(float));
    float *out = calloc((size_t)T * (size_t)n, sizeof(float));
    float *wc = malloc((size_t)n * (size_t)k * sizeof(float));
    if (!qx || !sx || !out || !wc) { *why = "out of memory"; fail = 1; goto done; }
    memcpy(wc, w, (size_t)n * (size_t)k * sizeof(float));

    mynah_asr_qmat m;
    mynah_asr_qmat_init(&m, wc, n, k, MYNAH_ASR_Q_INT8);
    mynah_asr_qmat_mul_rows(&m, x, out, T, qx, sx);
    mynah_asr_qmat_free(&m);

    for (size_t i = 0; i < (size_t)T * (size_t)n; i++) {
        if (memcmp(&out[i], &ref[i], sizeof(float)) != 0) {
            static char buf[224];
            snprintf(buf, sizeof(buf),
                     "float epilogue differs at row %d activation %d: reference "
                     "%.9g vs kernel %.9g — the integers agreed, so the two "
                     "kernels are grouping the scale product differently",
                     (int)(i % (size_t)n), (int)(i / (size_t)n),
                     (double)ref[i], (double)out[i]);
            *why = buf;
            fail = 1;
            break;
        }
    }
done:
    (void)kern;
    free(qx); free(sx); free(out); free(wc);
    return fail;
}

/* The reference arm's own output for one (shape, T), produced by the pre-S5-1
 * per-row loop. Caller owns `ref`. */
static int reference_floats(int n, int k, int T, const float *w, const float *x, float *ref) {
    int8_t *qx = malloc((size_t)T * (size_t)k);
    float *sx = malloc((size_t)T * sizeof(float));
    float *wc = malloc((size_t)n * (size_t)k * sizeof(float));
    if (!qx || !sx || !wc) { free(qx); free(sx); free(wc); return -1; }
    memcpy(wc, w, (size_t)n * (size_t)k * sizeof(float));
    mynah_asr_qmat m;
    mynah_asr_qmat_init(&m, wc, n, k, MYNAH_ASR_Q_INT8);
    mynah_asr_qmat_mul_rows(&m, x, ref, T, qx, sx);
    mynah_asr_qmat_free(&m);
    free(qx); free(sx); free(wc);
    return 0;
}

static int gate_identity(void) {
    int fails = 0;
    printf("\nint8 kernel identity (S5-1): every compiled kernel == the per-row dot, bit for bit\n");

    /* the reference arm has to exist before anything can be compared to it */
    const char *r0 = NULL;
    if (mynah_asr_qmat_kernel_available(MYNAH_ASR_QK_DOT_PER_ROW, &r0) != 1) {
        printf("  reference arm 'dot-per-row' unavailable — SKIP: %s. Nothing runs "
               "a native int8 kernel here, so there is nothing to be identical to.\n",
               r0 ? r0 : "no reason given");
        return 0;
    }

    for (int ki = 0; ki < N_KERNELS; ki++) {
        const int kern = KERNELS[ki];
        const char *why = NULL;
        const int avail = mynah_asr_qmat_kernel_available(kern, &why);
        if (avail != 1) {
            printf("  %-12s SKIPPED (%s): %s\n", mynah_asr_qmat_kernel_name(kern),
                   avail < 0 ? "not compiled" : "compiled, not runnable here",
                   why ? why : "no reason given");
            continue;
        }
        if (mynah_asr_qmat_kernel_force(kern) != kern) {
            printf("  %-12s FAIL: available but could not be pinned\n",
                   mynah_asr_qmat_kernel_name(kern));
            fails++;
            continue;
        }
        mynah_asr_qmat_kernel_counters_reset();

        int cases = 0, bad = 0;
        const char *msg = NULL;
        for (int si = 0; si < N_SHAPES && !bad; si++) {
            const int n = SHAPES[si].n, k = SHAPES[si].k;
            float *w = malloc((size_t)n * (size_t)k * sizeof(float));
            float *x = malloc((size_t)TS[N_TS - 1] * (size_t)k * sizeof(float));
            float *ref = malloc((size_t)TS[N_TS - 1] * (size_t)n * sizeof(float));
            if (!w || !x || !ref) { free(w); free(x); free(ref); msg = "out of memory"; bad = 1; break; }
            rng_state = 1234u + (unsigned long)si;
            for (size_t i = 0; i < (size_t)n * (size_t)k; i++) w[i] = frand();
            for (size_t i = 0; i < (size_t)TS[N_TS - 1] * (size_t)k; i++) x[i] = frand();
            for (int ti = 0; ti < N_TS && !bad; ti++) {
                const int T = TS[ti];
                if (identity_ints(kern, n, k, T, &msg)) { bad = 1; break; }
                /* the reference is produced by the reference arm, then the
                 * kernel under test is pinned back */
                if (mynah_asr_qmat_kernel_force(MYNAH_ASR_QK_DOT_PER_ROW) < 0 ||
                    reference_floats(n, k, T, w, x, ref) != 0 ||
                    mynah_asr_qmat_kernel_force(kern) != kern) {
                    msg = "could not build the reference"; bad = 1; break;
                }
                if (identity_floats(kern, n, k, T, w, x, ref, &msg)) { bad = 1; break; }
                cases++;
            }
            free(w); free(x); free(ref);
        }

        /* the shapes the batched stream step really produces */
        const shape real[] = {{1024, 1024}, {1024, 4096}};
        const int real_T[] = {4, 17, 64, 8};
        for (int si = 0; si < 2 && !bad; si++) {
            const int n = real[si].n, k = real[si].k;
            const int Tmax = si == 0 ? 64 : 8;
            float *w = malloc((size_t)n * (size_t)k * sizeof(float));
            float *x = malloc((size_t)Tmax * (size_t)k * sizeof(float));
            float *ref = malloc((size_t)Tmax * (size_t)n * sizeof(float));
            if (!w || !x || !ref) { free(w); free(x); free(ref); msg = "out of memory"; bad = 1; break; }
            rng_state = 777u + (unsigned long)si;
            for (size_t i = 0; i < (size_t)n * (size_t)k; i++) w[i] = frand();
            for (size_t i = 0; i < (size_t)Tmax * (size_t)k; i++) x[i] = frand();
            for (int ti = 0; ti < (si == 0 ? 3 : 1) && !bad; ti++) {
                const int T = si == 0 ? real_T[ti] : real_T[3];
                if (mynah_asr_qmat_kernel_force(MYNAH_ASR_QK_DOT_PER_ROW) < 0 ||
                    reference_floats(n, k, T, w, x, ref) != 0 ||
                    mynah_asr_qmat_kernel_force(kern) != kern) {
                    msg = "could not build the reference"; bad = 1; break;
                }
                if (identity_floats(kern, n, k, T, w, x, ref, &msg)) { bad = 1; break; }
                cases++;
            }
            free(w); free(x); free(ref);
        }

        const unsigned long long ran = mynah_asr_qmat_kernel_counter(kern);
        if (!bad && ran == 0 && kern != MYNAH_ASR_QK_DOT_PER_ROW) {
            msg = "the counter says this kernel never actually ran";
            bad = 1;
        }
        printf("  %-12s %s  %d shape/T cases, %llu kernel blocks\n",
               mynah_asr_qmat_kernel_name(kern), bad ? "FAIL" : "exact OK", cases, ran);
        if (bad) {
            printf("      %s\n", msg ? msg : "no detail");
            fails++;
        }
    }
    mynah_asr_qmat_kernel_force(-1);
    return fails;
}

/* ======================================================== the A/B bench ==== */

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Paired in-process A/B of the weight-stationary leaves against the per-row
 * dot they replace, on the shapes the batched stream step produces. A dev
 * signal, never a serving number (ENGINEERING.md §8). */
static void bench(void) {
    const shape shapes[] = {{1024, 1024}, {1024, 4096}};
    const int Ts[] = {1, 2, 4, 8, 16, 32, 64};
    /* the auto leaf has to be read BEFORE anything is pinned: once arm 0 has
     * forced dot-per-row, "resolved" is dot-per-row and the A/B compares a
     * thing with itself — which is exactly what a 1.00 ratio everywhere means */
    const int ws_leaf = mynah_asr_qmat_kernel_force(-1);
    printf("\nint8 weight-stationary A/B (paired, one process) — DEV SIGNAL, not a serving number\n");
    printf("  resolved kernel here: %s (%s)\n", mynah_asr_qmat_int8_kernel(),
           mynah_asr_qmat_kernel_name(ws_leaf));
    for (int si = 0; si < 2; si++) {
        const int n = shapes[si].n, k = shapes[si].k;
        float *w = malloc((size_t)n * (size_t)k * sizeof(float));
        float *x = malloc((size_t)64 * (size_t)k * sizeof(float));
        float *out = malloc((size_t)64 * (size_t)n * sizeof(float));
        int8_t *qx = malloc((size_t)64 * (size_t)k);
        float *sx = malloc(64 * sizeof(float));
        if (!w || !x || !out || !qx || !sx) { free(w); free(x); free(out); free(qx); free(sx); return; }
        rng_state = 99u + (unsigned long)si;
        for (size_t i = 0; i < (size_t)n * (size_t)k; i++) w[i] = frand();
        for (size_t i = 0; i < 64u * (size_t)k; i++) x[i] = frand();
        mynah_asr_qmat m;
        mynah_asr_qmat_init(&m, w, n, k, MYNAH_ASR_Q_INT8);

        printf("  n=%d k=%d\n", n, k);
        printf("    %3s | %12s | %12s | %6s\n", "T", "per-row ms", "wt-stat ms", "ratio");
        for (int ti = 0; ti < 7; ti++) {
            const int T = Ts[ti];
            double best[2] = {1e9, 1e9};
            for (int arm = 0; arm < 2; arm++) {
                const int kern = arm == 0 ? MYNAH_ASR_QK_DOT_PER_ROW : ws_leaf;
                if (arm == 1 && kern == MYNAH_ASR_QK_DOT_PER_ROW) { best[1] = best[0]; continue; }
                if (mynah_asr_qmat_kernel_force(kern) < 0) continue;
                for (int r = 0; r < 3; r++) mynah_asr_qmat_mul_rows(&m, x, out, T, qx, sx);
                for (int r = 0; r < 7; r++) {
                    const double t0 = now_s();
                    mynah_asr_qmat_mul_rows(&m, x, out, T, qx, sx);
                    const double el = now_s() - t0;
                    if (el < best[arm]) best[arm] = el;
                }
            }
            printf("    %3d | %12.3f | %12.3f | %6.2f\n", T, best[0] * 1e3, best[1] * 1e3,
                   best[1] > 0 ? best[0] / best[1] : 0.0);
        }
        mynah_asr_qmat_kernel_force(-1);
        mynah_asr_qmat_free(&m);
        free(w); free(x); free(out); free(qx); free(sx);
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench();
        return 0;
    }
    int fails = 0;
    /* small-T: direct dot path (SDOT/VNNI/AVX2/NEON) */
    fails += check("q8 small-T", MYNAH_ASR_Q_INT8, 96, 1024, 1, 0.03);
    fails += check("q8 small-T multi", MYNAH_ASR_Q_INT8, 64, 4096, 4, 0.03);
    fails += check("q4 small-T", MYNAH_ASR_Q_INT4, 96, 1024, 1, 0.08);
    fails += check("q4 small-T multi", MYNAH_ASR_Q_INT4, 64, 4096, 4, 0.08);
    /* large T: dequant+GEMM path */
    fails += check("q8 large-T", MYNAH_ASR_Q_INT8, 96, 1024, 48, 0.02);
    fails += check("q4 large-T", MYNAH_ASR_Q_INT4, 96, 1024, 48, 0.08);
    /* f32 passthrough */
    fails += check("f32 passthrough", MYNAH_ASR_Q_F32, 96, 1024, 8, 1e-6);

    fails += gate_identity();

    if (fails) { fprintf(stderr, "FAIL (%d)\n", fails); return 1; }
    printf("OK\n");
    return 0;
}
