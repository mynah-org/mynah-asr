/* sgemm.h — mynah-asr's own f32 GEMM, so that no external BLAS has to be in
 * the process.
 *
 * WHY THIS EXISTS.  The first reason is OWNERSHIP (.work/threadpool-and-lane.md,
 * standing decision).  OpenBLAS brings its own thread pool with its
 * own policies: inside a prefork worker pinned to T cpus our pool builds T
 * threads and OpenBLAS builds T more, so the worker runs 2T threads on T cpus
 * and the two pools take turns owning the cores (measured on the Axion: one
 * worker pinned to 8 cpus held 63 threads with OpenBLAS linked and 32
 * without).  On top of that comes an env var that has to be ABSENT for a
 * profile to be valid, and a team size that ignores our own pool.  Every one
 * of those is a trap that has to be rechecked on every host, forever.
 * Production is Linux, where the BLAS is OpenBLAS, so removing it is a
 * production decision.  `BLAS=openblas` stays as the comparison arm.
 *
 * The second reason is that the kernels are now worth linking.  Until
 * 2026-09-19 the DOT family below computed ONE dot product at a time into ONE
 * accumulator: a single FMA dependency chain, latency-bound at ~8.7 GF/s on an
 * M1 core that can do ~100.  It is register-tiled now (SG_DOT_MR x
 * SG_DOT_NC_TILE at once, a 1 x SG_DOT_NC_WIDE strip for a gemv), which is a
 * schedule change and not an arithmetic one — each element still accumulates
 * over the whole of k into one accumulator, in the same order, so the answer is
 * byte-identical and tests/test_sgemm.c gates exactly that.  75-80 GF/s, and at
 * the shapes a streaming step issues (4-16 stacked rows) faster than
 * Accelerate's AMX, which only pays from m >= 16.  tests/bench_gemm_shapes is
 * where that is measured, one process, both arms, shape by shape.
 *
 * THIS FILE IS NOT THE SEAM.  src/backend.c is: `mynah_asr_gemm_f32` and
 * `mynah_asr_gemv_f32` are the one door every f32 GEMM in this runtime goes
 * through, and they select Accelerate, OpenBLAS or this file.  Nothing outside
 * backend.c includes <cblas.h> or calls into here.
 *
 * WHAT THE SHAPES ACTUALLY ARE (src/encoder.c, src/subsampling.c,
 * src/decoder.c, src/qmat.c, read 2026-09-18, Nemotron 0.6b streaming):
 *
 *   x @ W^T, the linears and the joint head — trans_b, m = B*Q rows (1..64 in
 *       a stream step, T in the offline path), n and k in {1024, 4096, 13088}.
 *       This is the whole of mynah_asr_gemm_wt and of qmat's f32 fallback.
 *   attention scores  q @ k^T — trans_b, m = rows in the chunk, n = window,
 *       k = dk (64).
 *   attention context scores @ v — NO transpose, n = dk.
 *   subsampling pointwise / im2col — NO transpose, n = frames.
 *   the LSTM pred-net and the VAD LSTM — gemv, m = 1 wearing a GEMM's clothes.
 *
 * The families below are exactly those answers.  The threshold between narrow
 * and panel is DERIVED from the register file (mynah_asr_sgemm_narrow_max()),
 * not chosen.
 *
 * ROW STABILITY, which this runtime needs and a vendor BLAS does not give.
 * src/encoder.c stacks B streams' rows into one GEMM only where row t of C is
 * the same bytes for m = q and for m = sum(q) — for Accelerate that had to be
 * MEASURED (see mynah_asr_enc_batch_f32_ok).  Here it is a property of the
 * code for the stacked shape: that GEMM is `x @ W^T`, trans_b, which is the
 * DOT family, where each output element is one dot product over the whole of
 * k and nothing about it depends on m.  The measurement is still run, because
 * a claim the test does not exercise is a claim nobody checks.
 *
 * DEVIATION FROM THE SIBLING.  mynah-tts' sgemm.c calls mynah_dot_f32 out of
 * its kernels.c for the DOT family.  mynah-asr has no f32 kernel module and
 * one caller, so the dot is written here in terms of the SAME SG_* ISA macros
 * as the micro-kernels — one ISA abstraction in this file rather than two.
 * The conv-tap fusion is not lifted: this runtime has no conv1d tap GEMM (the
 * subsampling convolutions go through im2col and one GEMM), and a fused path
 * with no call site is a feature nothing tests.
 */
#ifndef MYNAH_ASR_SGEMM_H
#define MYNAH_ASR_SGEMM_H

#include <stddef.h>

/* ------------------------------------------------------------------------
 * The call
 * ------------------------------------------------------------------------ */

/* C[m,n] = alpha * op(A) * op(B) + beta * C, row-major.
 *
 * Identical semantics to cblas_sgemm(CblasRowMajor, ...) for the arguments
 * this repo uses, including both transpose flags, alpha, beta and the leading
 * dimensions:
 *
 *   trans_a == 0  op(A) is A, m x k, row stride lda   (lda >= k)
 *   trans_a != 0  op(A) is A^T, A is k x m, lda >= m
 *   trans_b == 0  op(B) is B, k x n, ldb >= n
 *   trans_b != 0  op(B) is B^T, B is n x k, ldb >= k
 *
 * beta == 0 means C is WRITTEN, never read: an uninitialised or NaN-carrying
 * C must not poison the result.  That is cblas' contract.
 *
 * Work is split over disjoint output blocks on this repo's pool
 * (mynah_asr_parallel_for).  The reduction over k is never split, and the
 * row-block size is always rounded up to the micro-kernel's row count, so the
 * result DOES NOT DEPEND ON THE THREAD COUNT — see the determinism note at the
 * top of sgemm.c.
 *
 * Allocates nothing.  Returns 0, or -1 on a null pointer or a leading
 * dimension too small for the declared shape. */
int mynah_asr_sgemm_f32(int trans_a, int trans_b,
                        size_t m, size_t n, size_t k,
                        float alpha,
                        const float *a, size_t lda,
                        const float *b, size_t ldb,
                        float beta,
                        float *c, size_t ldc);

/* The definition of correctness: the naive i,j,p triple loop, always compiled,
 * never vectorised, never threaded.  Every kernel in sgemm.c is checked
 * against THIS by mynah_asr_sgemm_self_test(), and it is also the scalar
 * fallback on a target with no vector ISA — so the reference exists once
 * rather than twice.  Same arguments and semantics as mynah_asr_sgemm_f32. */
void mynah_asr_sgemm_f32_reference(int trans_a, int trans_b,
                                   size_t m, size_t n, size_t k,
                                   float alpha,
                                   const float *a, size_t lda,
                                   const float *b, size_t ldb,
                                   float beta,
                                   float *c, size_t ldc);

/* ------------------------------------------------------------------------
 * Families
 *
 * A family is a RESIDENCY STRATEGY, not a tuning constant.  The micro-kernel
 * is shared on purpose (one formula, one reference); what differs is which
 * operand stays in registers and how many times the other one is streamed.
 * ------------------------------------------------------------------------ */
typedef enum {
    /* Degenerate (m, n or k == 0), too small to pay for blocking, or a
     * transpose combination no kernel here handles.  Runs the reference. */
    MYNAH_ASR_SGEMM_FAMILY_REFERENCE = 0,
    /* op(B) transposed: op(B)'s columns are B's ROWS, contiguous in k.  There
     * is nothing to vectorise across the n axis, so this is n*m dot products.
     * Every f32 linear in this runtime (x @ W^T) lands here. */
    MYNAH_ASR_SGEMM_FAMILY_DOT = 1,
    /* m == 1.  One output row: no row blocking exists, and a gemv must not pay
     * for a blocked kernel. */
    MYNAH_ASR_SGEMM_FAMILY_MATVEC = 2,
    /* n <= mynah_asr_sgemm_narrow_max(): the ENTIRE C row block fits in the
     * register file, so op(A) is streamed exactly once and the n columns never
     * leave registers.  The attention context GEMM (n = dk = 64 head-strided,
     * blocked to nc) and the narrow subsampling shapes land here. */
    MYNAH_ASR_SGEMM_FAMILY_NARROW = 3,
    /* Everything wider: an ordinary panel GEMM, column panels sized so the
     * op(B) panel stays in cache while the rows of op(A) sweep past it. */
    MYNAH_ASR_SGEMM_FAMILY_PANEL = 4
} mynah_asr_sgemm_family;

/* The predicate.  Exported so the dispatch report can CALL it instead of
 * restating "n <= 16" — a report that recomputes the condition can agree with
 * the source and both be wrong (src/dispatch.h, the central rule).  `why`
 * (optional) receives a static string naming the clause that decided. */
mynah_asr_sgemm_family mynah_asr_sgemm_family_for(int trans_a, int trans_b,
                                                  size_t m, size_t n, size_t k,
                                                  const char **why);

const char *mynah_asr_sgemm_family_name(mynah_asr_sgemm_family family);

/* Which micro-kernel ISA this translation unit compiled: "neon", "avx2" or
 * "scalar".  A fact about the build, answered by the file that owns it. */
const char *mynah_asr_sgemm_isa_name(void);

/* The narrow/panel boundary, in columns.  DERIVED, not chosen:
 *
 *     narrow_max = lanes * (accumulator budget / micro-kernel rows)
 *
 * with the accumulator budget set to half the architectural vector register
 * file so the b operand and the broadcast a values still have somewhere to
 * live.  NEON (32 regs, 4 lanes): 4 * (16/4) = 16.  AVX2 (16 regs, 8 lanes):
 * 8 * (8/4) = 16.  Scalar: 1 * (16/4) = 4. */
size_t mynah_asr_sgemm_narrow_max(void);

/* ------------------------------------------------------------------------
 * What actually ran
 * ------------------------------------------------------------------------ */

/* Per-family call counts since process start.  Counters, not predicates:
 * --dispatch-map builds its table before any model is loaded, so at that
 * moment no GEMM has run and the row must say so rather than present a clean
 * zero as health. */
typedef struct {
    unsigned long long calls;
    unsigned long long reference;
    unsigned long long dot;
    unsigned long long matvec;
    unsigned long long narrow;
    unsigned long long panel;
    unsigned long long refused;   /* bad arguments, nothing computed */
} mynah_asr_sgemm_stats;

void mynah_asr_sgemm_stats_get(mynah_asr_sgemm_stats *out);
void mynah_asr_sgemm_stats_reset(void);

/* ------------------------------------------------------------------------
 * Self-test
 * ------------------------------------------------------------------------ */

/* Model-free.  Runs every COMPILED path against mynah_asr_sgemm_f32_reference()
 * over the real stream-step shapes plus the edge cases the blocking can get
 * wrong: k not a multiple of the unroll, m == 1, n == 1, beta zero and
 * non-zero, alpha != 1, both transpose flags, and a thread-count sweep that is
 * asserted BYTE-identical.  Refuses (returns -1) if a family never actually
 * ran, because a forced path that silently degraded to the reference would
 * make the comparison vacuous.  0 = pass. */
int mynah_asr_sgemm_self_test(char *error, size_t error_capacity);

/* Force one family regardless of the predicate, for the self-test.  `ran`
 * (optional) receives the family that was actually executed, which may be
 * REFERENCE when the requested one cannot serve this operand layout.  Not for
 * production callers: mynah_asr_sgemm_f32 is the entry point. */
int mynah_asr_sgemm_f32_forced(mynah_asr_sgemm_family want,
                               mynah_asr_sgemm_family *ran,
                               int trans_a, int trans_b,
                               size_t m, size_t n, size_t k,
                               float alpha,
                               const float *a, size_t lda,
                               const float *b, size_t ldb,
                               float beta,
                               float *c, size_t ldc);

/* Run the GEMM with the row grid planned for exactly `tasks` blocks instead of
 * for the pool width.  The determinism proof in the self-test is built on this
 * and on nothing else; production never calls it. */
int mynah_asr_sgemm_f32_tasks(size_t tasks,
                              int trans_a, int trans_b,
                              size_t m, size_t n, size_t k,
                              float alpha,
                              const float *a, size_t lda,
                              const float *b, size_t ldb,
                              float beta,
                              float *c, size_t ldc);

/* Largest relative deviation the last self-test observed, or -1 before one has
 * run, and the bound it was judged against.  Exported so the dispatch row can
 * print the MEASURED margin rather than just PASS: "7.1e-07 against a 1e-04
 * bound" is a number a future tightening can be argued from; "PASS" is not. */
float mynah_asr_sgemm_self_test_worst(void);
float mynah_asr_sgemm_self_test_bound(void);

#endif /* MYNAH_ASR_SGEMM_H */
