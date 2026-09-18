/* flags.c — the registry. See flags.h for why it exists.
 *
 * ADDING A FLAG: add its getenv() where it belongs, then add ONE row here.
 * tools/check_flag_registry.py (run by `make check`) fails the build if you do
 * one without the other, in either direction. */
#include "flags.h"

#include "dispatch.h"
#include "qmat.h"
#include "threads.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------- inert predicates
 * "This flag cannot do anything in THIS binary on THIS host, and here is why."
 * The reason is what the banner prints after IGNORED:, so it has to be a
 * sentence an operator can act on, not a restatement of the name. */

static const char *inert_caps(void) {
#if MYNAH_ASR_DISPATCH_HAS_X86
    return NULL;
#else
    return "not an x86 build: NEON/SDOT are chosen at compile time, there is no "
           "runtime SIMD level to pick";
#endif
}

static const char *inert_qgemm(void) {
    /* Asks the owner (src/qmat.c) rather than re-deriving the gate here. */
    return mynah_asr_qmat_qgemm() < 0
               ? "no native int8 kernel in this build (neither ARM dotprod nor "
                 "x86), so T>16 always dequantizes and calls sgemm"
               : NULL;
}

static const char *inert_openblas_threads(void) {
#if MYNAH_ASR_DISPATCH_HAS_ACCELERATE
    return "this build links Accelerate; openblas_set_num_threads is never "
           "called and Accelerate nests through GCD";
#else
    return NULL;
#endif
}

static const char *inert_metal(void) {
#if MYNAH_ASR_DISPATCH_HAS_METAL
    return NULL;
#else
    return "Metal is not compiled into this build";
#endif
}

static const char *inert_cuda(void) {
#if MYNAH_ASR_DISPATCH_HAS_CUDA
    return NULL;
#else
    return "CUDA is not compiled into this build (make cuda)";
#endif
}

/* --------------------------------------------------- effective-value hooks */

static int eff_threads(const char *requested, char *out, size_t cap) {
    const int effective = mynah_asr_num_threads();
    snprintf(out, cap, "%d", effective);
    return atoi(requested) != effective;   /* 1 = clamped to [1,64] */
}

static int eff_caps(const char *requested, char *out, size_t cap) {
    const int level = mynah_asr_caps_effective();
    const char *name = mynah_asr_caps_name(level);
    snprintf(out, cap, "%s", name);
    if (strcmp(requested, "auto") == 0) return 0;
    return strcmp(requested, name) != 0;   /* 1 = the CPU could not honour it */
}

/* =============================================================== the table
 * Scope, default, description, inert predicate, effective-value hook.
 * Order: what steers a measurement first, then the process, then the server,
 * then the diagnostics. */
static const mynah_asr_flag g_flags[] = {
    {"MYNAH_ASR_CAPS", MYNAH_ASR_FLAG_KERNEL, "auto (cpuid)",
     "x86 SIMD level for the int8/int4 dot kernels: auto|scalar|avx2|vnni",
     inert_caps, eff_caps},

    {"MYNAH_ASR_QGEMM", MYNAH_ASR_FLAG_KERNEL, "0 (off)",
     "1 enables the threaded int8xint8 GEMM for T>16 instead of dequant+sgemm",
     inert_qgemm, NULL},

    {"MYNAH_ASR_CUDA_BF16", MYNAH_ASR_FLAG_KERNEL, "0 (off)",
     "1 lets cuBLAS use bf16 compute for the large GEMMs",
     inert_cuda, NULL},

    {"MYNAH_ASR_CUDA_TF32", MYNAH_ASR_FLAG_KERNEL, "0 (off)",
     "1 lets cuBLAS use TF32 compute for the large GEMMs",
     inert_cuda, NULL},

    {"MYNAH_ASR_THREADS", MYNAH_ASR_FLAG_RUNTIME, "online cores",
     "width of the parallel-for pool, clamped to [1,64]",
     NULL, eff_threads},

    {"OPENBLAS_NUM_THREADS", MYNAH_ASR_FLAG_RUNTIME, "unset (the pool decides)",
     "when set, src/threads.c never touches openblas_set_num_threads again",
     inert_openblas_threads, NULL},

    {"MYNAH_ASR_VERBOSE", MYNAH_ASR_FLAG_RUNTIME, "0 (quiet)",
     "1 prints the [FLAGS] and [EFFECTIVE-CONFIG] lines on stderr before a run",
     NULL, NULL},

    {"MYNAH_ASR_PREFORK_QUEUE", MYNAH_ASR_FLAG_SERVER, "the config default",
     "per-worker queue depth, or 'unbounded' (server only)",
     NULL, NULL},

    {"MYNAH_ASR_PREFORK_QUEUE_MS", MYNAH_ASR_FLAG_SERVER, "the config default",
     "queue deadline in ms; 0 disables the rung (server only)",
     NULL, NULL},

    {"MYNAH_ASR_PREFORK_SERVICE_MS", MYNAH_ASR_FLAG_SERVER, "the config default",
     "per-request service cap in ms; 0 disables it (server only)",
     NULL, NULL},

    {"MYNAH_ASR_PREFORK_ALLOW_GPU", MYNAH_ASR_FLAG_SERVER, "unset (refuse)",
     "set to allow prefork with a GPU backend, which forks a GPU context "
     "(server only)",
     NULL, NULL},

    {"MYNAH_ASR_PREFORK_STRICT", MYNAH_ASR_FLAG_SERVER, "unset (warn)",
     "set to turn a prefork topology warning into a refusal (server only)",
     NULL, NULL},

    {"MYNAH_ASR_LANE_SPLIT", MYNAH_ASR_FLAG_SERVER, "0 (no lane)",
     "cpus reserved for the ingest/output lane (server only)",
     NULL, NULL},

    {"MYNAH_ASR_CPU_TOPOLOGY_ROOT", MYNAH_ASR_FLAG_DEBUG, "/sys/devices/system/cpu",
     "sysfs root the topology reader parses; exists so tests can fake a machine",
     NULL, NULL},

    {"MYNAH_ASR_CGROUP_ROOT", MYNAH_ASR_FLAG_DEBUG, "/sys/fs/cgroup",
     "cgroup root the quota reader parses; exists so tests can fake a slice",
     NULL, NULL},

    {"MYNAH_ASR_METAL_PROF", MYNAH_ASR_FLAG_DEBUG, "unset (quiet)",
     "set to print per-encoder Metal timings on stderr",
     inert_metal, NULL},

    {"MALLOC_COUNT_OUT", MYNAH_ASR_FLAG_DEBUG, "unset (no file)",
     "where tests/malloc_count.c, the PRELOADED allocation counter, writes its "
     "census; nothing inside the runtime reads it",
     NULL, NULL},
};

#define N_FLAGS ((int)(sizeof(g_flags) / sizeof(g_flags[0])))

int mynah_asr_flags_count(void) { return N_FLAGS; }

const mynah_asr_flag *mynah_asr_flags_get(int i) {
    return (i >= 0 && i < N_FLAGS) ? &g_flags[i] : NULL;
}

const mynah_asr_flag *mynah_asr_flags_find(const char *name) {
    if (name == NULL) return NULL;
    for (int i = 0; i < N_FLAGS; i++)
        if (strcmp(g_flags[i].name, name) == 0) return &g_flags[i];
    return NULL;
}

const char *mynah_asr_flag_scope_name(int scope) {
    switch (scope) {
        case MYNAH_ASR_FLAG_RUNTIME: return "runtime";
        case MYNAH_ASR_FLAG_KERNEL:  return "kernel";
        case MYNAH_ASR_FLAG_SERVER:  return "server";
        case MYNAH_ASR_FLAG_DEBUG:   return "debug";
        default:                     return "?";
    }
}

const char *mynah_asr_flag_str(const char *name, const char *dflt) {
    const char *v = mynah_asr_flags_find(name) ? getenv(name) : NULL;
    return (v && v[0]) ? v : dflt;
}

int mynah_asr_flag_int(const char *name, int dflt) {
    const char *v = mynah_asr_flag_str(name, NULL);
    return v ? atoi(v) : dflt;
}

/* ------------------------------------------------------------ build facts */

const char *mynah_asr_build_id(void) {
#ifdef MYNAH_ASR_BUILD
    return MYNAH_ASR_BUILD;
#else
    return "dev";
#endif
}

const char *mynah_asr_blas_provider(void) {
#if MYNAH_ASR_DISPATCH_HAS_ACCELERATE
    return "accelerate";
#elif MYNAH_ASR_DISPATCH_HAS_OPENBLAS
    return "openblas";
#else
    return "none";
#endif
}

/* The ISA the COMPILER was allowed to emit, plus the target-attributed kernels
 * that are in the binary whatever -march said. Deliberately not a claim about
 * what ran: that is the dispatch map's job. */
const char *mynah_asr_simd_profile(void) {
    static char buf[80];
    if (buf[0]) return buf;
    size_t o = 0;
#define ADD(s)                                                                 \
    do {                                                                       \
        const int w = snprintf(buf + o, sizeof(buf) - o, "%s%s", o ? "+" : "", (s)); \
        if (w > 0) o += (size_t)w;                                             \
    } while (0)
#if MYNAH_ASR_DISPATCH_HAS_NEON
    ADD("neon");
#endif
#if MYNAH_ASR_DISPATCH_HAS_DOTPROD
    ADD("dotprod");
#endif
#if MYNAH_ASR_DISPATCH_BUILT_I8MM
    ADD("i8mm");
#endif
#if MYNAH_ASR_DISPATCH_HAS_X86
    ADD("x86-64");
#endif
#if MYNAH_ASR_DISPATCH_BUILT_AVX2
    ADD("avx2");
#endif
#if MYNAH_ASR_DISPATCH_BUILT_AVX512F
    ADD("avx512f");
#endif
#if MYNAH_ASR_DISPATCH_HAS_VNNI_KERNEL
    ADD("vnni-kernel");
#endif
#undef ADD
    if (buf[0] == '\0') snprintf(buf, sizeof(buf), "scalar");
    return buf;
}

/* ---------------------------------------------------------------- the two
 * lines. Exactly two, always both, whatever is set. */

static void sanitize(char *s) {
    for (; *s; s++)
        if (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') *s = '_';
}

void mynah_asr_flags_print(FILE *out) {
    if (out == NULL) out = stderr;

    fputs("[FLAGS] v=1", out);
    for (int i = 0; i < N_FLAGS; i++) {
        const char *v = getenv(g_flags[i].name);
        if (v == NULL || v[0] == '\0') continue;
        char val[128];
        snprintf(val, sizeof(val), "%s", v);
        sanitize(val);
        fprintf(out, " %s=%s", g_flags[i].name, val);
    }
    fputc('\n', out);

    fprintf(out, "[EFFECTIVE-CONFIG] v=1 build=%s blas=%s simd=%s",
            mynah_asr_build_id(), mynah_asr_blas_provider(), mynah_asr_simd_profile());
    for (int i = 0; i < N_FLAGS; i++) {
        const mynah_asr_flag *f = &g_flags[i];
        const char *v = getenv(f->name);
        if (v == NULL || v[0] == '\0') continue;   /* unset: there is no request */
        char req[128];
        snprintf(req, sizeof(req), "%s", v);
        sanitize(req);

        const char *why = f->inert ? f->inert() : NULL;
        if (why) {
            char reason[192];
            snprintf(reason, sizeof(reason), "%s", why);
            sanitize(reason);
            fprintf(out, " %s=%s->none(IGNORED:_%s)", f->name, req, reason);
            continue;
        }
        char eff[128];
        int clamped = 0;
        if (f->effective) {
            clamped = f->effective(req, eff, sizeof(eff));
            sanitize(eff);
        } else {
            snprintf(eff, sizeof(eff), "%s", req);
        }
        fprintf(out, " %s=%s->%s(%s)", f->name, req, eff, clamped ? "clamped" : "applied");
    }
    fputc('\n', out);
}
