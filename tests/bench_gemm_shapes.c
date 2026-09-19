/* bench_gemm_shapes — the f32 provider A/B, shape by shape, in ONE process.
 *
 * WHY THIS EXISTS.  "Should OpenBLAS leave the worker?" has been an open
 * question with no measurement behind it, and the way it was going to be
 * answered — build twice, run the whole server on each, compare cadence
 * percentiles — cannot say WHICH shape moved, costs a box-day, and compares two
 * processes that never saw the same page cache or the same thermal state.
 *
 * This tool answers the same question the other way round.  `own`
 * (src/sgemm.c) is compiled into EVERY build, whatever BLAS= selected, so both
 * arms are callable from one process:
 *
 *     arm A   mynah_asr_sgemm_f32   — ours, always present
 *     arm B   mynah_asr_gemm_f32    — the seam, i.e. whatever this build linked
 *
 * They are timed INTERLEAVED, batch by batch, on the same operands, and each
 * arm keeps its best batch: the comparison is a difference measured under one
 * set of conditions rather than two absolute numbers measured under two
 * (~/.claude: measure differences, not values).  Correctness is checked first —
 * an arm that is fast because it computed something else is not an arm.
 *
 * WHICH SHAPES.  Not invented here: `MYNAH_ASR_GEMM_PROFILE=1` makes
 * src/backend.c record every call through the seam per distinct shape and dump
 * the table at exit, so the input is what the model ACTUALLY issued, with the
 * call counts that weight it.  Per family, per lookahead, per quantisation:
 *
 *     MYNAH_ASR_GEMM_PROFILE=1 ./mynah-asr transcribe -m <model> a.wav 2> shapes.txt
 *     tests/bench_gemm_shapes shapes.txt
 *
 * With counts, the footer gives the only number that decides the question: the
 * modelled cost of ONE run's worth of GEMM under each arm.  Without a profile
 * (--demo) it runs a representative table and reports per-shape ratios only,
 * which is enough to find where `own` loses but not to claim a verdict.
 *
 * Model-free by construction: it links libmynah_asr but loads nothing.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "backend.h"
#include "sgemm.h"
#include "threads.h"

enum { K_GEMM = 0, K_GEMV = 1 };

typedef struct {
    int kind;
    int ta, tb;                 /* gemv: ta is `trans`, tb unused        */
    int m, n, k;                /* gemv: m=rows, n=cols, k=lda           */
    unsigned long long calls;   /* 0 when the shape came from --demo     */
    double ns_own, ns_seam;     /* best observed, per call               */
    double worst_rel;           /* largest relative difference own↔seam  */
} shape;

#define MAX_SHAPES 512
static shape g_s[MAX_SHAPES];
static int g_n;

/* ------------------------------------------------------------------ input */

/* The representative table: the shapes src/sgemm.h documents for a 0.6B
 * FastConformer streaming step (d_model 1024, ffn 4096, 8 heads, 24 layers,
 * subsampling 8x, lookahead 3 -> 4 encoder frames per chunk), plus the offline
 * row counts.  DERIVED FROM THE ARCHITECTURE, NOT OBSERVED: it exists so the
 * tool is runnable and testable on a host with no model, and so the harness
 * itself is exercised in CI.  A verdict quotes a profile, never this. */
static void load_demo(void) {
    static const int rows[] = {4, 8, 16, 32, 64, 256};
    /* x @ W^T — every f32 linear (attention projections, FFN, the joint head) */
    static const int lin[][2] = {{1024, 1024}, {4096, 1024}, {1024, 4096}, {13088, 640}};
    for (unsigned r = 0; r < sizeof(rows) / sizeof(rows[0]); r++)
        for (unsigned l = 0; l < sizeof(lin) / sizeof(lin[0]); l++) {
            if (g_n >= MAX_SHAPES) return;
            g_s[g_n++] = (shape){K_GEMM, 0, 1, rows[r], lin[l][0], lin[l][1], 0, 0, 0, 0};
        }
    /* attention: scores q @ k^T (trans_b, k = dk) and context scores @ v */
    static const int dk[] = {64, 128};
    static const int win[] = {60, 120, 320};
    for (unsigned r = 0; r < 4; r++)
        for (unsigned d = 0; d < 2; d++)
            for (unsigned w = 0; w < 3; w++) {
                if (g_n + 1 >= MAX_SHAPES) return;
                g_s[g_n++] = (shape){K_GEMM, 0, 1, rows[r], win[w], dk[d], 0, 0, 0, 0};
                g_s[g_n++] = (shape){K_GEMM, 0, 0, rows[r], dk[d], win[w], 0, 0, 0, 0};
            }
    /* subsampling im2col / pointwise: no transpose, n = frames */
    g_s[g_n++] = (shape){K_GEMM, 0, 0, 256, 400, 9, 0, 0, 0, 0};
    g_s[g_n++] = (shape){K_GEMM, 0, 0, 256, 200, 256, 0, 0, 0, 0};
    g_s[g_n++] = (shape){K_GEMM, 0, 1, 50, 1024, 2304, 0, 0, 0, 0};
    /* the LSTM prediction network and the VAD LSTM: gemv */
    g_s[g_n++] = (shape){K_GEMV, 0, 0, 2560, 640, 640, 0, 0, 0, 0};
    g_s[g_n++] = (shape){K_GEMV, 0, 0, 640, 640, 640, 0, 0, 0, 0};
}

static int parse_kv(const char *line, const char *key, int *out) {
    char pat[64];
    snprintf(pat, sizeof(pat), " %s=", key);
    const char *p = strstr(line, pat);
    if (!p) return 0;
    *out = atoi(p + strlen(pat));
    return 1;
}

static int parse_kv_ull(const char *line, const char *key, unsigned long long *out) {
    char pat[64];
    snprintf(pat, sizeof(pat), " %s=", key);
    const char *p = strstr(line, pat);
    if (!p) return 0;
    *out = strtoull(p + strlen(pat), NULL, 10);
    return 1;
}

/* Reads the [GEMM-SHAPE] lines of a MYNAH_ASR_GEMM_PROFILE dump.  Every other
 * line is ignored, so the file can be the raw stderr of the run. */
static int load_profile(const char *path) {
    FILE *f = strcmp(path, "-") == 0 ? stdin : fopen(path, "r");
    if (!f) { fprintf(stderr, "bench_gemm_shapes: cannot open %s\n", path); return -1; }
    char line[1024];
    int taken = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "[GEMM-SHAPE]", 12) != 0) continue;
        if (g_n >= MAX_SHAPES) break;
        shape s;
        memset(&s, 0, sizeof(s));
        unsigned long long calls = 0;
        if (strstr(line, "kind=gemv")) {
            s.kind = K_GEMV;
            if (!parse_kv(line, "trans", &s.ta) || !parse_kv(line, "rows", &s.m) ||
                !parse_kv(line, "cols", &s.n) || !parse_kv(line, "lda", &s.k))
                continue;
        } else {
            s.kind = K_GEMM;
            if (!parse_kv(line, "ta", &s.ta) || !parse_kv(line, "tb", &s.tb) ||
                !parse_kv(line, "m", &s.m) || !parse_kv(line, "n", &s.n) ||
                !parse_kv(line, "k", &s.k))
                continue;
        }
        parse_kv_ull(line, "calls", &calls);
        s.calls = calls;
        g_s[g_n++] = s;
        taken++;
    }
    if (f != stdin) fclose(f);
    if (taken == 0)
        fprintf(stderr, "bench_gemm_shapes: no [GEMM-SHAPE] line in %s "
                        "(run with MYNAH_ASR_GEMM_PROFILE=1)\n", path);
    return taken;
}

/* ------------------------------------------------------------------ timing */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Deterministic filler: the same bytes for both arms and for every run, so a
 * difference between arms is never a difference between inputs. */
static void fill(float *p, size_t n, unsigned seed) {
    unsigned x = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        p[i] = (float)((int)(x >> 8) % 2001 - 1000) * 0.001f;
    }
}

static void run_own(const shape *s, const float *a, const float *b, float *c) {
    if (s->kind == K_GEMV) {
        /* the seam's gemv contract, expressed the way backend.c expresses it */
        if (s->ta)
            mynah_asr_sgemm_f32(0, 0, 1, (size_t)s->n, (size_t)s->m, 1.0f, b,
                                (size_t)s->m, a, (size_t)s->k, 0.0f, c, (size_t)s->n);
        else
            mynah_asr_sgemm_f32(0, 1, 1, (size_t)s->m, (size_t)s->n, 1.0f, b,
                                (size_t)s->n, a, (size_t)s->k, 0.0f, c, (size_t)s->m);
        return;
    }
    const size_t lda = s->ta ? (size_t)s->m : (size_t)s->k;
    const size_t ldb = s->tb ? (size_t)s->k : (size_t)s->n;
    mynah_asr_sgemm_f32(s->ta, s->tb, (size_t)s->m, (size_t)s->n, (size_t)s->k,
                        1.0f, a, lda, b, ldb, 0.0f, c, (size_t)s->n);
}

static void run_seam(const shape *s, const float *a, const float *b, float *c) {
    if (s->kind == K_GEMV) {
        mynah_asr_gemv_f32(s->ta, s->m, s->n, 1.0f, a, s->k, b, 0.0f, c);
        return;
    }
    const int lda = s->ta ? s->m : s->k;
    const int ldb = s->tb ? s->k : s->n;
    mynah_asr_gemm_f32(s->ta, s->tb, s->m, s->n, s->k, 1.0f, a, lda, b, ldb,
                       0.0f, c, s->n);
}

static size_t a_elems(const shape *s) {
    if (s->kind == K_GEMV) return (size_t)s->m * (size_t)(s->k > s->n ? s->k : s->n);
    return (size_t)s->m * (size_t)s->k;
}
static size_t b_elems(const shape *s) {
    if (s->kind == K_GEMV) return (size_t)(s->ta ? s->m : s->n);
    return (size_t)s->k * (size_t)s->n;
}
static size_t c_elems(const shape *s) {
    if (s->kind == K_GEMV) return (size_t)(s->ta ? s->n : s->m);
    return (size_t)s->m * (size_t)s->n;
}
static double flops(const shape *s) {
    if (s->kind == K_GEMV) return 2.0 * (double)s->m * (double)s->n;
    return 2.0 * (double)s->m * (double)s->n * (double)s->k;
}

/* One shape: correctness first, then `batches` interleaved timing rounds.
 * Each round runs `reps` calls of each arm; the arm keeps its BEST round. */
static int bench_one(shape *s, int batches, double target_s) {
    float *a = malloc(a_elems(s) * sizeof(float));
    float *b = malloc(b_elems(s) * sizeof(float));
    float *c1 = malloc(c_elems(s) * sizeof(float));
    float *c2 = malloc(c_elems(s) * sizeof(float));
    if (!a || !b || !c1 || !c2) { free(a); free(b); free(c1); free(c2); return -1; }
    fill(a, a_elems(s), 12345u);
    fill(b, b_elems(s), 6789u);

    run_own(s, a, b, c1);
    run_seam(s, a, b, c2);
    double worst = 0.0;
    const size_t cn = c_elems(s);
    for (size_t i = 0; i < cn; i++) {
        const double d = fabs((double)c1[i] - (double)c2[i]);
        const double m = fabs((double)c1[i]) > 1.0 ? fabs((double)c1[i]) : 1.0;
        if (d / m > worst) worst = d / m;
    }
    s->worst_rel = worst;

    /* how many calls make one round long enough to be above clock noise */
    const double t0 = now_s();
    run_own(s, a, b, c1);
    const double one = now_s() - t0;
    int reps = (int)(target_s / (one > 1e-9 ? one : 1e-9));
    if (reps < 1) reps = 1;
    if (reps > 200000) reps = 200000;

    double best_own = 1e30, best_seam = 1e30;
    for (int r = 0; r < batches; r++) {
        double t = now_s();
        for (int i = 0; i < reps; i++) run_own(s, a, b, c1);
        const double d_own = (now_s() - t) / reps;
        t = now_s();
        for (int i = 0; i < reps; i++) run_seam(s, a, b, c2);
        const double d_seam = (now_s() - t) / reps;
        if (d_own < best_own) best_own = d_own;
        if (d_seam < best_seam) best_seam = d_seam;
    }
    s->ns_own = best_own * 1e9;
    s->ns_seam = best_seam * 1e9;
    free(a); free(b); free(c1); free(c2);
    return 0;
}

/* ------------------------------------------------------------------- main */

static int cmp_cost(const void *x, const void *y) {
    const shape *a = (const shape *)x, *b = (const shape *)y;
    const double ca = a->ns_seam * (double)(a->calls ? a->calls : 1);
    const double cb = b->ns_seam * (double)(b->calls ? b->calls : 1);
    return ca < cb ? 1 : (ca > cb ? -1 : 0);
}

static void usage(void) {
    fprintf(stderr,
        "usage: bench_gemm_shapes [<profile>|-] [--demo] [--batches N] [--target-ms N]\n"
        "\n"
        "  <profile>     stderr of a run with MYNAH_ASR_GEMM_PROFILE=1 (the shapes and\n"
        "                the call counts the model actually issued); - reads stdin\n"
        "  --demo        no profile: a representative table, per-shape ratios only\n"
        "  --batches N   interleaved timing rounds per shape (default 5)\n"
        "  --target-ms N how long one round should take (default 30)\n");
}

int main(int argc, char **argv) {
    const char *profile = NULL;
    int demo = 0, batches = 5;
    double target_s = 0.030;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--demo") == 0) demo = 1;
        else if (strcmp(argv[i], "--batches") == 0 && i + 1 < argc) batches = atoi(argv[++i]);
        else if (strcmp(argv[i], "--target-ms") == 0 && i + 1 < argc) target_s = atoi(argv[++i]) / 1000.0;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { usage(); return 0; }
        else if (argv[i][0] != '-' || strcmp(argv[i], "-") == 0) profile = argv[i];
        else { usage(); return 2; }
    }
    if (!profile && !demo) { usage(); return 2; }
    if (batches < 1) batches = 1;

    if (profile) { if (load_profile(profile) <= 0) return 1; }
    else load_demo();

    const char *prov = mynah_asr_gemm_provider();
    const int same = strcmp(prov, "own") == 0;

    printf("[GEMM-BENCH] v=1 provider=%s isa=%s threads=%d shapes=%d source=%s batches=%d\n",
           prov, mynah_asr_sgemm_isa_name(), mynah_asr_num_threads(), g_n,
           profile ? profile : "demo", batches);
    if (same)
        printf("[GEMM-BENCH] NOTE this build linked no BLAS: both arms are the same code, "
               "so the ratio column is noise, not a comparison\n");

    for (int i = 0; i < g_n; i++)
        if (bench_one(&g_s[i], batches, target_s) != 0) {
            fprintf(stderr, "bench_gemm_shapes: out of memory on shape %d\n", i);
            return 1;
        }

    qsort(g_s, (size_t)g_n, sizeof(g_s[0]), cmp_cost);

    printf("%-4s %2s %2s %6s %6s %6s %12s %11s %11s %7s %9s %s\n",
           "kind", "ta", "tb", "m", "n", "k", "calls", "own_ns", "seam_ns",
           "own/seam", "own_GF/s", "family");
    double tot_own = 0, tot_seam = 0, worst = 0;
    int weighted = 0;
    for (int i = 0; i < g_n; i++) {
        const shape *s = &g_s[i];
        const char *why = NULL;
        const mynah_asr_sgemm_family fam =
            s->kind == K_GEMV
                ? mynah_asr_sgemm_family_for(0, s->ta ? 0 : 1, 1, (size_t)(s->ta ? s->n : s->m),
                                             (size_t)(s->ta ? s->m : s->n), &why)
                : mynah_asr_sgemm_family_for(s->ta, s->tb, (size_t)s->m, (size_t)s->n,
                                             (size_t)s->k, &why);
        const double w = (double)(s->calls ? s->calls : 0);
        if (s->calls) { tot_own += w * s->ns_own; tot_seam += w * s->ns_seam; weighted = 1; }
        if (s->worst_rel > worst) worst = s->worst_rel;
        printf("%-4s %2d %2d %6d %6d %6d %12llu %11.1f %11.1f %7.2f %9.1f %s\n",
               s->kind == K_GEMV ? "gemv" : "gemm", s->ta, s->tb, s->m, s->n, s->k,
               s->calls, s->ns_own, s->ns_seam,
               s->ns_seam > 0 ? s->ns_own / s->ns_seam : 0.0,
               s->ns_own > 0 ? flops(s) / s->ns_own : 0.0,
               mynah_asr_sgemm_family_name(fam));
    }

    printf("[GEMM-BENCH] v=1 worst_rel_diff=%.3g (own vs %s, over every shape)\n", worst, prov);
    if (weighted)
        printf("[GEMM-BENCH-TOTAL] v=1 provider=%s modelled_own_ms=%.3f modelled_seam_ms=%.3f "
               "ratio=%.3f\n",
               prov, tot_own * 1e-6, tot_seam * 1e-6,
               tot_seam > 0 ? tot_own / tot_seam : 0.0);
    else
        printf("[GEMM-BENCH-TOTAL] v=1 provider=%s modelled=NONE "
               "(no call counts: a shape table without a profile cannot be weighted)\n", prov);
    return 0;
}
