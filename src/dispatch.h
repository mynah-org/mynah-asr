/* dispatch.h — the dispatch report: every ISA / BLAS / backend / pool decision
 * this binary can make, RESOLVED for this host and this environment.
 *
 * WHY THIS EXISTS.  ENGINEERING.md §5: "Never infer the active kernel from a
 * flag, a build target or a filename."  The sibling repo learned that the hard
 * way — mynah-tts src/dispatch.h opens with the incident where a README and a
 * commit message attributed an EPYC result to "AVX-512 VNNI" on a build whose
 * int8 dot was the AVX2 widen-then-madd pair.  The number was real; the
 * attribution was invented.  This file is the same defence for mynah-asr: a
 * result may only ever be attributed to the kernel that `--dispatch-map`
 * printed in the SAME run.
 *
 * SIX COLUMNS, per logical feature:
 *
 *   feature    the stable dotted id
 *   compiled   the implementation is in THIS binary (build-time gate)
 *   supported  this CPU/host actually has it (runtime probe; "?" = cannot tell)
 *   env        the environment variable that steers it, or "unset"
 *   resolved   what the runtime decision is NOW
 *   reason     which branch decided it, and where that branch lives
 *
 * THE CENTRAL RULE, taken from mynah-tts: `resolved` is NEVER re-derived here.
 * `compiled && supported` is not an answer — it is exactly the guess that lets
 * a report and a README agree and both be wrong.  A row's `resolved` may come
 * from three places, and `reason` says which:
 *
 *   [predicate]  a predicate exported by the translation unit that OWNS the
 *                decision (src/qmat.c: mynah_asr_qmat_int8_kernel() and
 *                friends).  Always preferred.
 *   [runtime]    a real exported call this file makes and believes:
 *                mynah_asr_num_threads(), mynah_asr_blas_budget(),
 *                mynah_asr_backend(), mynah_asr_metal_available(), sysctl,
 *                cpuid, getauxval.
 *   [gate]       a PURE compile-time gate with no runtime fallback — "CUDA is
 *                not in this binary" is a complete answer, not a guess.
 *
 * Anything else resolves to UNKNOWN and the footer counts those rows.  UNKNOWN
 * is a finding (a predicate somebody still has to export), not a failure.
 *
 * WHY NO REGISTRATION TABLE.  mynah-tts routes predicates through
 * mynah_dispatch_register_probe() because its owners are many and some sit
 * behind link-time islands.  Here the owners are three files, all linked into
 * libmynah_asr.a, so the direct call IS the owner's predicate with one less
 * indirection to get wrong.  The rule that matters — the report asks the
 * owner, it never guesses — is unchanged.
 */
#ifndef MYNAH_ASR_DISPATCH_H
#define MYNAH_ASR_DISPATCH_H

#include <stdio.h>
#include <stddef.h>

/* ------------------------------------------------------------------------
 * The compile gates of this runtime, in one place.
 * Each macro is 1 or 0 and mirrors, expression for expression, the gate named
 * in its comment.  Nothing else in this header tests a raw __ARM_* / __AVX*.
 * ------------------------------------------------------------------------ */

#if defined(__ARM_NEON) || defined(__aarch64__)
#define MYNAH_ASR_DISPATCH_HAS_NEON 1          /* src/qmat.c MYNAH_ASR_HAVE_NEON */
#else
#define MYNAH_ASR_DISPATCH_HAS_NEON 0
#endif

#if defined(__ARM_FEATURE_DOTPROD)
#define MYNAH_ASR_DISPATCH_HAS_DOTPROD 1       /* src/qmat.c MYNAH_ASR_HAVE_SDOT */
#else
#define MYNAH_ASR_DISPATCH_HAS_DOTPROD 0
#endif

/* Compiled for i8mm is NOT the same as "there is an i8mm kernel": src/qmat.c
 * has no SMMLA path at all, so this macro exists only for the ISA guard, and
 * the IDLE HARDWARE footer says so in words rather than leaving a blank the
 * reader fills in optimistically. */
#if defined(__ARM_FEATURE_MATMUL_INT8)
#define MYNAH_ASR_DISPATCH_BUILT_I8MM 1
#else
#define MYNAH_ASR_DISPATCH_BUILT_I8MM 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define MYNAH_ASR_DISPATCH_HAS_X86 1           /* src/qmat.c MYNAH_ASR_HAVE_X86 */
#else
#define MYNAH_ASR_DISPATCH_HAS_X86 0
#endif

/* src/qmat.c carries dot_q8_vnni behind __attribute__((target(...))), so on x86
 * it is compiled into EVERY build and chosen at runtime by cpuid — "compiled"
 * is a property of the target and the compiler, not of CFLAGS.  That is
 * precisely why a -march flag must never be read as a kernel claim. */
#if MYNAH_ASR_DISPATCH_HAS_X86
#define MYNAH_ASR_DISPATCH_HAS_VNNI_KERNEL 1
#define MYNAH_ASR_DISPATCH_HAS_AVX2_KERNEL 1
#else
#define MYNAH_ASR_DISPATCH_HAS_VNNI_KERNEL 0
#define MYNAH_ASR_DISPATCH_HAS_AVX2_KERNEL 0
#endif

/* What the COMPILER was allowed to emit everywhere (-march=native by default
 * in the Makefile).  Used only by the ISA guard: these are the instruction
 * sets a stray autovectorized loop may contain. */
#if defined(__AVX2__)
#define MYNAH_ASR_DISPATCH_BUILT_AVX2 1
#else
#define MYNAH_ASR_DISPATCH_BUILT_AVX2 0
#endif
#if defined(__AVX512F__)
#define MYNAH_ASR_DISPATCH_BUILT_AVX512F 1
#else
#define MYNAH_ASR_DISPATCH_BUILT_AVX512F 0
#endif

#if defined(MYNAH_ASR_BLAS_ACCELERATE)
#define MYNAH_ASR_DISPATCH_HAS_ACCELERATE 1
#else
#define MYNAH_ASR_DISPATCH_HAS_ACCELERATE 0
#endif
#if defined(MYNAH_ASR_BLAS_OPENBLAS)
#define MYNAH_ASR_DISPATCH_HAS_OPENBLAS 1
#else
#define MYNAH_ASR_DISPATCH_HAS_OPENBLAS 0
#endif
#if defined(MYNAH_ASR_METAL)
#define MYNAH_ASR_DISPATCH_HAS_METAL 1
#else
#define MYNAH_ASR_DISPATCH_HAS_METAL 0
#endif
#if defined(MYNAH_ASR_CUDA)
#define MYNAH_ASR_DISPATCH_HAS_CUDA 1
#else
#define MYNAH_ASR_DISPATCH_HAS_CUDA 0
#endif

/* ------------------------------------------------------------------------
 * CPU feature probe — TRI-STATE on purpose
 *
 * YES / NO / UNKNOWN, never a two-valued answer.  A feature this process
 * cannot interrogate (no sysctl key, a leaf a hypervisor hid, an emulator) is
 * not an absence, and the ISA guard refusing to start over our own ignorance
 * would be a worse bug than the SIGILL it prevents.
 *
 * Names: "dotprod" "i8mm" "bf16" "sve" "sve2" on ARM;
 *        "avx2" "avx512f" "avx512vnni" "avxvnni" "amx" on x86.
 * An unknown name answers UNKNOWN.
 * ------------------------------------------------------------------------ */
enum { MYNAH_ASR_CPU_NO = 0, MYNAH_ASR_CPU_YES = 1, MYNAH_ASR_CPU_UNKNOWN = -1 };
int mynah_asr_cpu_has(const char *feature);

/* ------------------------------------------------------------------------
 * Rows
 * ------------------------------------------------------------------------ */
typedef enum {
    MYNAH_ASR_DISPATCH_SRC_PREDICATE = 0,
    MYNAH_ASR_DISPATCH_SRC_RUNTIME   = 1,
    MYNAH_ASR_DISPATCH_SRC_GATE      = 2,
    MYNAH_ASR_DISPATCH_SRC_UNKNOWN   = 3
} mynah_asr_dispatch_source;

#define MYNAH_ASR_DISPATCH_MAX_ROWS 32

typedef struct {
    const char *id;            /* component.feature                          */
    const char *compiled;      /* "yes" / "no" / "-"                          */
    const char *supported;     /* "yes" / "no" / "?" / "-"                    */
    const char *env_name;      /* steering variable, or NULL                  */
    char        env_value[64]; /* raw value, or "unset"                       */
    char        resolved[32];  /* a kernel name, ON/OFF, a number, or UNKNOWN */
    char        reason[200];
    int         source;        /* mynah_asr_dispatch_source                   */
} mynah_asr_dispatch_row;

/* Fill `rows`; returns the number written, or -1.  Collecting opens the GPU
 * backends (a real runtime call is the point), so it is not free: call once. */
int mynah_asr_dispatch_collect(mynah_asr_dispatch_row *rows, int capacity);

/* Print the table (human, or `json != 0` as one JSON document) followed by the
 * IDLE HARDWARE footer.  `out` NULL means stderr.  Returns the number of rows
 * whose `resolved` is UNKNOWN — i.e. the number of predicates still missing. */
int mynah_asr_dispatch_print(FILE *out, int json);

/* THE ISA GUARD — the first statement of main().
 *
 * Returns 0 when this binary can run on this CPU, or 78 (EX_CONFIG) after
 * printing ONE line on stderr naming the instruction set the binary was built
 * with and the feature this CPU reports missing — the whole diagnosis that an
 * over-built binary otherwise delivers as a bare SIGILL.
 *
 * It fires only on a DEFINITE absence (MYNAH_ASR_CPU_NO).  UNKNOWN never
 * refuses.  It allocates nothing and opens nothing. */
int mynah_asr_isa_guard(void);

/* Model-free check of the report machinery: the table collects, ids are
 * unique, every row has a source, no row is left UNKNOWN without saying which
 * predicate is missing.  0 = ok, -1 after writing `error`. */
int mynah_asr_dispatch_self_test(char *error, size_t error_capacity);

#endif
