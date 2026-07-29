/* Self-test of the quantized kernels (no model required — it runs in CI too):
 * compares the small-T path (SDOT/VNNI/AVX2/NEON/scalar) and the dequant+GEMM
 * path against the f32 reference, on reproducible random data.
 * Tolerances = the expected quantization error (q8 ~1%, q4 ~4%): a kernel bug
 * (signs, lane order, saturation) produces errors orders of magnitude larger.
 * Exit: 0 ok, 1 fail. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/qmat.h"

static unsigned long rng_state = 42;
static float frand(void) { /* xorshift, riproducibile ovunque */
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

int main(void) {
    int fails = 0;
    /* small-T: percorso dot diretto (SDOT/VNNI/AVX2/NEON) */
    fails += check("q8 small-T", MYNAH_ASR_Q_INT8, 96, 1024, 1, 0.03);
    fails += check("q8 small-T multi", MYNAH_ASR_Q_INT8, 64, 4096, 4, 0.03);
    fails += check("q4 small-T", MYNAH_ASR_Q_INT4, 96, 1024, 1, 0.08);
    fails += check("q4 small-T multi", MYNAH_ASR_Q_INT4, 64, 4096, 4, 0.08);
    /* T grande: percorso dequant+GEMM */
    fails += check("q8 large-T", MYNAH_ASR_Q_INT8, 96, 1024, 48, 0.02);
    fails += check("q4 large-T", MYNAH_ASR_Q_INT4, 96, 1024, 48, 0.08);
    /* f32 passthrough */
    fails += check("f32 passthrough", MYNAH_ASR_Q_F32, 96, 1024, 8, 1e-6);
    if (fails) { fprintf(stderr, "FAIL (%d)\n", fails); return 1; }
    printf("OK\n");
    return 0;
}
