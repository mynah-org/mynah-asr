/* Self-test of src/sgemm.c — our own f32 GEMM. No model required, so it runs
 * in CI on every platform and in every BLAS build.
 *
 * WHAT IT PROVES, and what would be worthless without it:
 *
 *  1. Every COMPILED family against the always-compiled scalar reference, over
 *     the shapes this runtime really issues (m = 1..64 stacked rows, n and k in
 *     {1024, 4096, 13088}, the attention window, the subsampling flatten) and
 *     the edge cases the blocking can get wrong: k not a multiple of the
 *     unroll, m == 1, n == 1, beta 0 and non-zero, alpha != 1, both transpose
 *     flags. mynah_asr_sgemm_self_test() REFUSES if a family never ran: a
 *     forced path that silently degraded to the reference would make the whole
 *     comparison vacuous, and "PASS" would then mean nothing.
 *
 *  2. The result does not depend on the thread count. The library-side sweep
 *     plans the same GEMM for 1, 2, 3, 5 and 64 blocks and memcmps them; this
 *     file adds the part a library function cannot do — it re-runs the ENTIRE
 *     self-test in a child process at MYNAH_ASR_THREADS = 1, 2, 4 and 8, with
 *     real threads in the pool, and compares the transcript of each run. The
 *     width is read once and cached by mynah_asr_num_threads(), so a fork is
 *     the only way to move it honestly inside one test.
 *
 *  3. That the seam agrees with the family kernels: mynah_asr_gemm_f32 and
 *     mynah_asr_gemv_f32 are checked against the reference too, so a mistake in
 *     the argument mapping (a transposed flag, a swapped leading dimension, the
 *     gemv rewritten as a one-row GEMM) fails HERE and not in a transcript.
 *     That check runs in all three BLAS builds and so also guards the vendor
 *     arms: it is the only test that looks at the seam itself.
 *
 * Exit: 0 ok, 1 fail. */
#include "../src/backend.h"
#include "../src/sgemm.h"
#include "../src/threads.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("sgemm FAIL: %s\n", msg); failures = 1; } \
    else printf("sgemm ok:   %s\n", msg); } while (0)

static void fill(float *p, size_t n, unsigned seed) {
    unsigned s = seed | 1u;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (float)((double)(s >> 8) / 8388608.0 - 1.0);
    }
}

static float maxdev(const float *got, const float *want, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(got[i])) return -1.0f;
        const float rel = fabsf(got[i] - want[i]) / (1.0f + fabsf(want[i]));
        if (rel > worst) worst = rel;
    }
    return worst;
}

/* ------------------------------------------------------------------ the seam
 * These run in EVERY build: with a vendor BLAS they check that provider's
 * answer against the reference too, which is how the argument mapping stays
 * honest on the arm that is not the default. */
#define SEAM_TOL 2.0e-4f

static void seam_gemm(int trans_a, int trans_b, int m, int n, int k,
                      float alpha, float beta) {
    const size_t na = (size_t)m * (size_t)k, nb = (size_t)n * (size_t)k;
    const size_t ncell = (size_t)m * (size_t)n;
    float *a = malloc(na * sizeof(float));
    float *b = malloc(nb * sizeof(float));
    float *c = malloc(ncell * sizeof(float));
    float *ref = malloc(ncell * sizeof(float));
    if (!a || !b || !c || !ref) { failures = 1; free(a); free(b); free(c); free(ref); return; }
    const int lda = trans_a ? m : k;
    const int ldb = trans_b ? k : n;
    fill(a, na, 11u); fill(b, nb, 22u); fill(c, ncell, 33u);
    memcpy(ref, c, ncell * sizeof(float));
    mynah_asr_sgemm_f32_reference(trans_a, trans_b, (size_t)m, (size_t)n,
                                  (size_t)k, alpha, a, (size_t)lda, b,
                                  (size_t)ldb, beta, ref, (size_t)n);
    mynah_asr_gemm_f32(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, n);
    const float dev = maxdev(c, ref, ncell);
    char msg[160];
    snprintf(msg, sizeof(msg),
             "seam gemm tA=%d tB=%d %dx%dx%d alpha=%g beta=%g dev=%.3g (bound %.0e)",
             trans_a, trans_b, m, n, k, (double)alpha, (double)beta,
             (double)dev, (double)SEAM_TOL);
    CHECK(dev >= 0.0f && dev <= SEAM_TOL, msg);
    free(a); free(b); free(c); free(ref);
}

static void seam_gemv(int trans, int rows, int cols, float alpha, float beta) {
    const size_t na = (size_t)rows * (size_t)cols;
    const size_t nx = (size_t)(trans ? rows : cols);
    const size_t ny = (size_t)(trans ? cols : rows);
    float *a = malloc(na * sizeof(float));
    float *x = malloc(nx * sizeof(float));
    float *y = malloc(ny * sizeof(float));
    float *ref = malloc(ny * sizeof(float));
    if (!a || !x || !y || !ref) { failures = 1; free(a); free(x); free(y); free(ref); return; }
    fill(a, na, 44u); fill(x, nx, 55u); fill(y, ny, 66u);
    memcpy(ref, y, ny * sizeof(float));
    /* y = alpha*op(A)*x + beta*y, written as a one-row GEMM on the reference:
     * y^T = alpha * x^T * op(A)^T. */
    if (trans)
        mynah_asr_sgemm_f32_reference(0, 0, 1, (size_t)cols, (size_t)rows, alpha,
                                      x, (size_t)rows, a, (size_t)cols, beta,
                                      ref, (size_t)cols);
    else
        mynah_asr_sgemm_f32_reference(0, 1, 1, (size_t)rows, (size_t)cols, alpha,
                                      x, (size_t)cols, a, (size_t)cols, beta,
                                      ref, (size_t)rows);
    mynah_asr_gemv_f32(trans, rows, cols, alpha, a, cols, x, beta, y);
    const float dev = maxdev(y, ref, ny);
    char msg[160];
    snprintf(msg, sizeof(msg),
             "seam gemv trans=%d %dx%d alpha=%g beta=%g dev=%.3g (bound %.0e)",
             trans, rows, cols, (double)alpha, (double)beta, (double)dev,
             (double)SEAM_TOL);
    CHECK(dev >= 0.0f && dev <= SEAM_TOL, msg);
    free(a); free(x); free(y); free(ref);
}

/* ------------------------------------------------- the thread-width sweep
 * A fresh PROCESS per width, and re-exec rather than plain fork, because
 * mynah_asr_num_threads() caches on its first call: a forked child inherits
 * the parent's already-cached width, sets MYNAH_ASR_THREADS, and would then
 * report a sweep it never ran — the exact silent-fallback shape this file
 * exists to catch. The child runs the whole self-test AND writes the bytes of
 * one fixed GEMM to a pipe, so the parent compares the RESULT and not just the
 * pass/fail. */
#define SWEEP_M 64u
#define SWEEP_N 1024u
#define SWEEP_K 1024u

/* The child half: `<binary> --sweep`, with MYNAH_ASR_THREADS already in the
 * environment and stdout replaced by the pipe. */
static int sweep_child(void) {
    char err[256];
    if (mynah_asr_sgemm_self_test(err, sizeof err) != 0) {
        fprintf(stderr, "  MYNAH_ASR_THREADS=%d: %s\n", mynah_asr_num_threads(), err);
        return 3;
    }
    const size_t na = SWEEP_M * SWEEP_K, nb = SWEEP_N * SWEEP_K;
    const size_t nc = SWEEP_M * SWEEP_N;
    float *a = malloc(na * sizeof(float));
    float *b = malloc(nb * sizeof(float));
    float *c = malloc(nc * sizeof(float));
    int rc = 0;
    if (!a || !b || !c) {
        rc = 4;
    } else {
        fill(a, na, 77u); fill(b, nb, 88u); fill(c, nc, 99u);
        /* trans_b: the stacked stream-step shape, the one the batched encoder
         * relies on being row-stable. */
        mynah_asr_sgemm_f32(0, 1, SWEEP_M, SWEEP_N, SWEEP_K, 1.0f, a, SWEEP_K, b,
                            SWEEP_K, 0.5f, c, SWEEP_N);
        const char *p = (const char *)c;
        size_t left = nc * sizeof(float);
        while (left > 0) {
            const ssize_t n = write(STDOUT_FILENO, p, left);
            if (n <= 0) { rc = 5; break; }
            p += n; left -= (size_t)n;
        }
    }
    /* The child must free before it exits, even though the process is about to
     * die: under ASan the leak check runs at exit and turns a harmless leak into
     * a non-zero status, which the parent reads as "the sweep failed at this
     * width". A test that leaks cannot be run by the leak checker. */
    free(a); free(b); free(c);
    return rc;
}

static int run_width(const char *self, int width, float *out) {
    int fds[2];
    if (pipe(fds) != 0) return -1;
    const pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return -1; }
    if (pid == 0) {
        close(fds[0]);
        if (dup2(fds[1], STDOUT_FILENO) < 0) _exit(6);
        close(fds[1]);
        char w[16];
        snprintf(w, sizeof(w), "%d", width);
        setenv("MYNAH_ASR_THREADS", w, 1);
        char *const argv[] = {(char *)self, (char *)"--sweep", NULL};
        execv(self, argv);
        _exit(7);
    }
    close(fds[1]);
    char *p = (char *)out;
    size_t left = SWEEP_M * SWEEP_N * sizeof(float);
    while (left > 0) {
        const ssize_t n = read(fds[0], p, left);
        if (n <= 0) break;
        p += n; left -= (size_t)n;
    }
    close(fds[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return -1;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return -1;
    return left == 0 ? 0 : -1;
}


/* ------------------------------------------------------------ tiling identity
 *
 * Every family here computes several output elements at once: the DOT family
 * an SG_DOT_MR x SG_DOT_NC_TILE block of dot products (and a 1 x
 * SG_DOT_NC_WIDE strip for the rows under MR, which is the whole of a gemv);
 * the panel and narrow families a row strip against SG_PANEL_NV column
 * vectors.  Those widths are a SCHEDULE, not a formula: an element accumulates
 * over the whole of k into one accumulator, folds the same way and finishes
 * with the same scalar tail whatever block it lands in.
 *
 * So the tiled answer must be the SAME BYTES as the untiled one.  If it is
 * not, an element's value depends on how many columns happened to sit beside
 * it -- and since a batched stream step stacks streams into the same GEMM,
 * that is exactly rule 4 of CLAUDE.md ("a transcript never depends on who it
 * was batched with") failing at the bottom of the stack.
 *
 * The comparison: the whole GEMM, against the same GEMM taken one column at a
 * time, with the family FORCED on both sides so the width is the only thing
 * that differs (left to the predicate, a one-column GEMM is small enough to
 * fall to the reference, and then the test would be comparing two algorithms).
 * m and n are ragged on purpose -- 9 rows is two DOT tiles plus one strip, 37
 * columns leaves a remainder on every path -- and k is not a multiple of
 * SG_LANES, so the scalar tail runs as well. */
static void tiling_identity(mynah_asr_sgemm_family want, int trans_b, size_t n,
                            const char *label) {
    const size_t m = 9, k = 77;
    const size_t ldb = trans_b ? k : n;
    float *a = malloc(m * k * sizeof(float));
    float *b = malloc(n * k * sizeof(float));
    float *c_tiled = malloc(m * n * sizeof(float));
    float *c_col = malloc(m * sizeof(float));
    if (!a || !b || !c_tiled || !c_col) {
        printf("sgemm FAIL: out of memory in the tiling gate\n");
        failures = 1;
        free(a); free(b); free(c_tiled); free(c_col);
        return;
    }
    fill(a, m * k, 991u);
    fill(b, n * k, 4241u);

    mynah_asr_sgemm_family ran = MYNAH_ASR_SGEMM_FAMILY_REFERENCE;
    mynah_asr_sgemm_f32_forced(want, &ran, 0, trans_b, m, n, k, 1.0f, a, k, b,
                               ldb, 0.0f, c_tiled, n);
    if (ran != want) {
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "the %s tiling gate could not force its family (ran %s): "
                 "not compiled on this ISA", label,
                 mynah_asr_sgemm_family_name(ran));
        printf("sgemm SKIP: %s\n", msg);
        free(a); free(b); free(c_tiled); free(c_col);
        return;
    }

    size_t differing = 0;
    for (size_t j = 0; j < n; ++j) {
        /* one column: for the transposed families that is row j of B, for the
         * others column j of B, which keeps its ldb and moves the base */
        const float *bj = trans_b ? b + j * k : b + j;
        mynah_asr_sgemm_f32_forced(want, NULL, 0, trans_b, m, 1, k, 1.0f, a, k,
                                   bj, ldb, 0.0f, c_col, 1);
        for (size_t i = 0; i < m; ++i)
            if (memcmp(&c_tiled[i * n + j], &c_col[i], sizeof(float)) != 0) differing++;
    }
    {
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "%s: the tiled block is byte-identical to one column at a time "
                 "(%zu of %zu elements differ, %zux%zux%zu)",
                 label, differing, m * n, m, n, k);
        CHECK(differing == 0, msg);
    }
    free(a); free(b); free(c_tiled); free(c_col);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--sweep") == 0) return sweep_child();
    printf("sgemm: provider=%s isa=%s narrow_max=%zu\n",
           mynah_asr_gemm_provider(), mynah_asr_sgemm_isa_name(),
           mynah_asr_sgemm_narrow_max());

    /* 1. every family against the reference, plus the coverage refusal */
    char err[256];
    err[0] = '\0';
    const int rc = mynah_asr_sgemm_self_test(err, sizeof err);
    if (rc != 0) printf("  %s\n", err);
    CHECK(rc == 0, "every compiled family matches the scalar reference");
    {
        char msg[128];
        snprintf(msg, sizeof(msg), "worst relative deviation %.3g (bound %.0e)",
                 (double)mynah_asr_sgemm_self_test_worst(),
                 (double)mynah_asr_sgemm_self_test_bound());
        CHECK(mynah_asr_sgemm_self_test_worst() >= 0.0f &&
              mynah_asr_sgemm_self_test_worst() <=
                  mynah_asr_sgemm_self_test_bound(), msg);
    }

    /* 2. the predicate: the shapes this runtime issues land where the header
     * says they do. Asked of the predicate, never restated as a condition. */
    CHECK(mynah_asr_sgemm_family_for(0, 1, 64, 1024, 1024, NULL) ==
              MYNAH_ASR_SGEMM_FAMILY_DOT,
          "the stacked linear (x @ W^T) is the DOT family, so it is row-stable in m");
    CHECK(mynah_asr_sgemm_family_for(0, 1, 1, 1024, 1024, NULL) ==
              MYNAH_ASR_SGEMM_FAMILY_DOT,
          "and so is the same linear at m == 1: the family does not move with m");
    CHECK(mynah_asr_sgemm_family_for(0, 0, 1, 2560, 640, NULL) ==
              MYNAH_ASR_SGEMM_FAMILY_MATVEC,
          "the LSTM gemv is the MATVEC family");
    CHECK(mynah_asr_sgemm_family_for(0, 0, 56, 64, 1024, NULL) ==
              MYNAH_ASR_SGEMM_FAMILY_NARROW ||
          mynah_asr_sgemm_family_for(0, 0, 56, 64, 1024, NULL) ==
              MYNAH_ASR_SGEMM_FAMILY_PANEL,
          "the attention context GEMM blocks (narrow or panel, per ISA)");
    CHECK(mynah_asr_sgemm_family_for(0, 0, 2, 2, 2, NULL) ==
              MYNAH_ASR_SGEMM_FAMILY_REFERENCE,
          "a tiny GEMM stays on the reference instead of paying for a plan");

    /* 37 columns leaves a remainder on every path; NARROW only serves
     * n <= mynah_asr_sgemm_narrow_max(), so it is asked its own width. */
    tiling_identity(MYNAH_ASR_SGEMM_FAMILY_DOT, 1, 37, "DOT");
    tiling_identity(MYNAH_ASR_SGEMM_FAMILY_PANEL, 0, 37, "PANEL");
    tiling_identity(MYNAH_ASR_SGEMM_FAMILY_NARROW, 0,
                    mynah_asr_sgemm_narrow_max() - 3u, "NARROW");

    /* 3. the seam, in whatever build this is */
    seam_gemm(0, 1, 64, 1024, 1024, 1.0f, 0.0f);
    seam_gemm(0, 1, 1, 1024, 1024, 1.0f, 0.0f);
    seam_gemm(0, 1, 17, 130, 63, -0.75f, 2.5f);
    seam_gemm(0, 0, 56, 64, 120, 1.0f, 0.0f);
    seam_gemm(0, 0, 33, 31, 29, 0.5f, -1.25f);
    seam_gemv(0, 2560, 640, 1.0f, 1.0f);
    seam_gemv(0, 640, 640, 1.0f, 0.0f);
    seam_gemv(1, 137, 64, 1.0f, 0.0f);
    seam_gemv(0, 137, 64, 0.125f, 0.0f);

    /* 4. thread-count independence with REAL threads: 1, 2, 4, 8 */
    {
        const size_t nc = SWEEP_M * SWEEP_N;
        float *base = malloc(nc * sizeof(float));
        float *got = malloc(nc * sizeof(float));
        if (!base || !got) { failures = 1; }
        else {
            static const int widths[] = {1, 2, 4, 8};
            int ok = run_width(argv[0], widths[0], base) == 0;
            CHECK(ok, "self-test passes at MYNAH_ASR_THREADS=1");
            for (size_t i = 1; ok && i < sizeof widths / sizeof widths[0]; ++i) {
                char msg[128];
                if (run_width(argv[0], widths[i], got) != 0) {
                    snprintf(msg, sizeof(msg),
                             "self-test passes at MYNAH_ASR_THREADS=%d", widths[i]);
                    CHECK(0, msg);
                    continue;
                }
                snprintf(msg, sizeof(msg),
                         "MYNAH_ASR_THREADS=%d gives byte-identical output to 1 "
                         "thread (%zux%zux%zu, transB)",
                         widths[i], (size_t)SWEEP_M, (size_t)SWEEP_N,
                         (size_t)SWEEP_K);
                CHECK(memcmp(base, got, nc * sizeof(float)) == 0, msg);
            }
        }
        free(base); free(got);
    }

    printf("test_sgemm: %s\n", failures ? "FAIL" : "OK");
    return failures;
}
