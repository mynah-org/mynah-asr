/* flags.c — the registry. See flags.h for why it exists.
 *
 * ADDING A FLAG: add its getenv() where it belongs, then add ONE row here.
 * tools/check_flag_registry.py (run by `make check`) fails the build if you do
 * one without the other, in either direction. */
#include "flags.h"

#include "backend.h"
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
#if MYNAH_ASR_DISPATCH_HAS_X86 || MYNAH_ASR_DISPATCH_HAS_DOTPROD
    /* S5-1 gave ARM a ladder of its own (scalar < sdot < smmla), so the flag
     * acts on both architectures now; it used to be inert off x86. */
    return NULL;
#else
    return "this build has no native int8 kernel (neither ARM dotprod nor x86), "
           "so there is no runtime SIMD level to pick";
#endif
}

static const char *inert_qgemm(void) {
    /* Asks the owner (src/qmat.c) rather than re-deriving the gate here. */
    return mynah_asr_qmat_qgemm() < 0
               ? "no native int8 kernel in this build (neither ARM dotprod nor "
                 "x86), so T>16 always dequantizes and calls sgemm"
               : NULL;
}

/* Asks the seam which provider was linked instead of re-deriving it from a
 * #if: src/backend.c owns that branch. */
static const char *inert_openblas_threads(void) {
    const char *p = mynah_asr_gemm_provider();
    if (strcmp(p, "openblas") == 0) return NULL;
    if (strcmp(p, "accelerate") == 0)
        return "this build links Accelerate; openblas_set_num_threads is never "
               "called and Accelerate nests through GCD";
    return "this build links no BLAS at all: the f32 GEMMs run on src/sgemm.c "
           "over the one pool, and there is no OpenBLAS team to size";
}

static const char *inert_sgemm_profile(void) {
    return strcmp(mynah_asr_gemm_provider(), "own") == 0
               ? NULL
               : "the f32 GEMMs go to a vendor BLAS in this build, so "
                 "src/sgemm.c never runs and has no shapes to record";
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
    {"MYNAH_ASR_CAPS", MYNAH_ASR_FLAG_KERNEL, "auto (from the CPU)",
     "SIMD level for the int8/int4 kernels, and the opt-out for every one of them:"
     " x86 auto|scalar|avx2|vnni (vnni = either VPDPBUSD encoding),"
     " ARM auto|scalar|sdot|smmla; a level above what the CPU has is downgraded with a note",
     inert_caps, eff_caps},

    {"MYNAH_ASR_BATCH_F32", MYNAH_ASR_FLAG_KERNEL, "auto (1 on accelerate and own, 0 on openblas)",
     "1/0 forces the f32 stream step to stack B streams' rows into one sgemm or to step them one by one;"
     " the default is on only where sgemm is proven row-stable -- measured for Accelerate, true by construction for our own"
     " (the stacked GEMM is x @ W^T, the DOT family, one dot per output element over the whole of k) -- and off on"
     " OpenBLAS, where it is unverified; int8 never consults it, its per-row dot is exact by construction",
     NULL, NULL},

    {"MYNAH_ASR_RELPOS_TABLE", MYNAH_ASR_FLAG_KERNEL, "1 (on, when the model streams)",
     "1/0: whether the rel-pos projection comes from the per-layer table built at load."
     " rk = pe(K) @ relk_w^T depends on neither the stream nor the audio nor the cache contents,"
     " and pe(K)[p] depends only on pos = K-1-p, so one table per layer is a window away from"
     " every K. Off, every stream projects for itself every step, which is what the S1-7 group"
     " sharing was trying to soften. The two arms must be byte-identical: that is what"
     " tests/test_stream_batch gates, and the table refuses to install itself at load if a"
     " probe window does not match a direct projection byte for byte",
     NULL, NULL},

    {"MYNAH_ASR_STACK_SOLO", MYNAH_ASR_FLAG_KERNEL, "1 (on)",
     "1/0: whether a ready set of ONE goes through the stacked encoder path."
     " The group is per lookahead preset, so this is not only B==1: eight streams on eight"
     " presets are eight groups of one. On, a lone chunk is one [R, d] GEMM; off, it is a chain"
     " of [1, d] GEMVs, which measured 86.7 ms against 16.8 on a 24-core Neoverse-V2."
     " Off is the pre-2026-09-20 behaviour and exists so the A/B is one binary",
     NULL, NULL},

    {"MYNAH_ASR_SGEMM_PROFILE", MYNAH_ASR_FLAG_DEBUG, "unset (off)",
     "src/sgemm.c: record every f32 GEMM shape AS EXECUTED (family, tasks asked of the pool, wall)"
     " and print the table at exit; off it costs one relaxed load per GEMM call",
     inert_sgemm_profile, NULL},

    {"MYNAH_ASR_BATCH_ALLOCS", MYNAH_ASR_FLAG_DEBUG, "unset",
     "tests/test_stream_batch: run only the zero-allocation gate of the batched step (same as --allocs)",
     NULL, NULL},

    {"MYNAH_ASR_POOL_SPIN_US", MYNAH_ASR_FLAG_RUNTIME, "50",
     "microseconds a pool thread spins watching for the next dispatch before parking on the condvar;"
     " 0 parks immediately (the pure condvar pool, the A/B arm), clamped to 10000",
     NULL, NULL},

    {"MYNAH_ASR_GEMM_PROFILE", MYNAH_ASR_FLAG_DEBUG, "unset",
     "1 records every f32 GEMM/GEMV through the backend seam per distinct shape and dumps the table at exit"
     " ([GEMM-PROFILE]/[GEMM-SHAPE] on stderr); the replay input of tests/bench_gemm_shapes",
     NULL, NULL},

    {"MYNAH_ASR_STEP_TIME", MYNAH_ASR_FLAG_DEBUG, "unset",
     "tests/test_stream_batch: print per-step wall time of single vs batched steps (a dev signal)",
     NULL, NULL},

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

    {"MYNAH_ASR_LISTEN_BACKLOG", MYNAH_ASR_FLAG_SERVER,
     "2 * (workers * (slots + queue)), floored at SOMAXCONN, capped at 4096",
     "listen() backlog for the service port, overriding the value derived from the"
     " admission ladder; the kernel still clamps it to somaxconn, and both numbers"
     " are on the [SERVER-CONFIG] banner (server only)",
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

/* One answer, from the file that owns the branch (src/backend.c): "accelerate",
 * "openblas" or "own".  It used to be a second copy of the compile gates here,
 * which is the exact shape of bug src/dispatch.h opens with — a report and a
 * source that agree and are both wrong. */
const char *mynah_asr_blas_provider(void) { return mynah_asr_gemm_provider(); }

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

void mynah_asr_flags_print_all(FILE *out) {
    if (!out) return;
    const int n = mynah_asr_flags_count();
    fprintf(out, "[FLAG-REGISTRY] v=1 count=%d\n", n);
    fprintf(out, "%-28s %-8s %-38s %s\n", "NAME", "SCOPE", "DEFAULT", "DESCRIPTION");
    for (int i = 0; i < n; i++) {
        const mynah_asr_flag *f = mynah_asr_flags_get(i);
        if (!f) continue;
        const char *inert = f->inert ? f->inert() : NULL;
        const char *set = getenv(f->name);
        fprintf(out, "%-28s %-8s %-38s %s\n", f->name,
                mynah_asr_flag_scope_name(f->scope), f->dflt ? f->dflt : "-",
                f->desc ? f->desc : "");
        if (inert)
            fprintf(out, "%-28s %-8s IGNORED HERE: %s\n", "", "", inert);
        if (set)
            fprintf(out, "%-28s %-8s SET IN THIS ENVIRONMENT: %s\n", "", "", set);
    }
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
