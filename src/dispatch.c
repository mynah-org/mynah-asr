/* dispatch.c — see dispatch.h for the rule this file exists to enforce. */
#include "dispatch.h"

#include "backend.h"
#include "flags.h"
#include "qmat.h"
#include "sgemm.h"
#include "threads.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__linux__) && (defined(__aarch64__) || defined(__arm__))
#include <sys/auxv.h>
#endif
#if MYNAH_ASR_DISPATCH_HAS_X86
#include <cpuid.h>
#endif

#if MYNAH_ASR_DISPATCH_HAS_METAL
int mynah_asr_metal_available(void);
#endif
#if MYNAH_ASR_DISPATCH_HAS_CUDA
int mynah_asr_cuda_available(void);
#endif

/* ===================================================== CPU feature probes ==
 * Tri-state. The only rule: never turn "I could not ask" into "no". */

#if defined(__APPLE__) && defined(__aarch64__)
/* Apple publishes hw.optional.arm.FEAT_* with an explicit 0 for a feature the
 * silicon lacks but the kernel knows about (FEAT_I8MM, FEAT_BF16 are 0 on M1).
 * A key that does not exist at all — FEAT_SVE on every Apple part to date — is
 * a question this process cannot answer, not a zero: ENOENT stays UNKNOWN. */
static int sysctl_feat(const char *key) {
    int v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(key, &v, &sz, NULL, 0) != 0) return MYNAH_ASR_CPU_UNKNOWN;
    return v ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
}
#endif

#if MYNAH_ASR_DISPATCH_HAS_X86
static unsigned long long x86_xgetbv0(void) {
    unsigned lo, hi;
    __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}

/* Same shape as src/qmat.c x86_detect_caps(): a feature bit without the OS
 * state to hold its registers is not usable, and reporting it as present would
 * make the IDLE HARDWARE footer lie in the one direction that costs a SIGILL. */
static int x86_feat(const char *f) {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return MYNAH_ASR_CPU_UNKNOWN;
    const int osxsave = (c >> 27) & 1;
    const unsigned long long xcr0 = osxsave ? x86_xgetbv0() : 0;
    const int ymm = osxsave && (xcr0 & 0x6) == 0x6;
    const int zmm = osxsave && (xcr0 & 0xE6) == 0xE6;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return MYNAH_ASR_CPU_UNKNOWN;
    const unsigned b7 = b, c7 = c, d7 = d;
    if (strcmp(f, "avx2") == 0) return (ymm && ((b7 >> 5) & 1)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
    if (strcmp(f, "avx512f") == 0) return (zmm && ((b7 >> 16) & 1)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
    if (strcmp(f, "avx512vnni") == 0) return (zmm && ((c7 >> 11) & 1)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
    if (strcmp(f, "amx") == 0)  /* AMX-TILE (24) + AMX-INT8 (25), edx of leaf 7.0 */
        return (((d7 >> 24) & 1) && ((d7 >> 25) & 1)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
    if (strcmp(f, "avxvnni") == 0) {
        unsigned a1, b1, c1, d1;
        if (!__get_cpuid_count(7, 1, &a1, &b1, &c1, &d1)) return MYNAH_ASR_CPU_UNKNOWN;
        return (ymm && ((a1 >> 4) & 1)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
    }
    return MYNAH_ASR_CPU_UNKNOWN;
}
#endif

int mynah_asr_cpu_has(const char *feature) {
    if (feature == NULL) return MYNAH_ASR_CPU_UNKNOWN;
#if defined(__APPLE__) && defined(__aarch64__)
    if (strcmp(feature, "dotprod") == 0) return sysctl_feat("hw.optional.arm.FEAT_DotProd");
    if (strcmp(feature, "i8mm") == 0) return sysctl_feat("hw.optional.arm.FEAT_I8MM");
    if (strcmp(feature, "bf16") == 0) return sysctl_feat("hw.optional.arm.FEAT_BF16");
    if (strcmp(feature, "sve") == 0) return sysctl_feat("hw.optional.arm.FEAT_SVE");
    if (strcmp(feature, "sve2") == 0) return sysctl_feat("hw.optional.arm.FEAT_SVE2");
    return MYNAH_ASR_CPU_UNKNOWN;
#elif defined(__linux__) && defined(__aarch64__)
    {
        const unsigned long h1 = getauxval(AT_HWCAP), h2 = getauxval(AT_HWCAP2);
        if (strcmp(feature, "dotprod") == 0) return (h1 & (1UL << 20)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
        if (strcmp(feature, "sve") == 0) return (h1 & (1UL << 22)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
        if (strcmp(feature, "sve2") == 0) return (h2 & (1UL << 1)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
        if (strcmp(feature, "i8mm") == 0) return (h2 & (1UL << 13)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
        if (strcmp(feature, "bf16") == 0) return (h2 & (1UL << 14)) ? MYNAH_ASR_CPU_YES : MYNAH_ASR_CPU_NO;
        return MYNAH_ASR_CPU_UNKNOWN;
    }
#elif MYNAH_ASR_DISPATCH_HAS_X86
    return x86_feat(feature);
#else
    (void)feature;
    return MYNAH_ASR_CPU_UNKNOWN;
#endif
}

static const char *tri_name(int t) {
    return t == MYNAH_ASR_CPU_YES ? "yes" : t == MYNAH_ASR_CPU_NO ? "no" : "?";
}

/* What the CPU offers the quantized kernels, asked of the SAME detector the
 * kernels use (src/qmat.c x86_detect_caps) rather than of a second cpuid here:
 * on x86 the highest SIMD level detected, on ARM whether dotprod is there. */
static const char *quant_supported(void) {
#if MYNAH_ASR_DISPATCH_HAS_X86
    return mynah_asr_caps_name(mynah_asr_caps_detected());
#else
    return tri_name(mynah_asr_cpu_has("dotprod"));
#endif
}

/* ============================================================ the rows ==== */

static void row_env(mynah_asr_dispatch_row *r, const char *name) {
    r->env_name = name;
    const char *v = name ? getenv(name) : NULL;
    snprintf(r->env_value, sizeof(r->env_value), "%s", v && v[0] ? v : "unset");
    for (char *p = r->env_value; *p; p++)
        if (*p == ' ' || *p == '\t') *p = '_';
}

static void row_set(mynah_asr_dispatch_row *r, const char *id, const char *compiled,
                    const char *supported, const char *env, const char *resolved,
                    int source, const char *reason) {
    memset(r, 0, sizeof(*r));
    r->id = id;
    r->compiled = compiled;
    r->supported = supported;
    r->source = source;
    snprintf(r->resolved, sizeof(r->resolved), "%s", resolved);
    snprintf(r->reason, sizeof(r->reason), "%s", reason);
    row_env(r, env);
}

static const char *src_tag(int source) {
    switch (source) {
        case MYNAH_ASR_DISPATCH_SRC_PREDICATE: return "[predicate]";
        case MYNAH_ASR_DISPATCH_SRC_RUNTIME:   return "[runtime]";
        case MYNAH_ASR_DISPATCH_SRC_GATE:      return "[gate]";
        default:                               return "[UNKNOWN]";
    }
}

int mynah_asr_dispatch_collect(mynah_asr_dispatch_row *rows, int capacity) {
    if (rows == NULL || capacity < 8) return -1;
    int n = 0;

    /* --- int8 dot: the kernel every streaming projection runs (T <= 16) ---
     * Owner: src/qmat.c. We ASK it; we do not rebuild its branch here. */
    row_set(&rows[n++], "kernel.int8_dot",
            MYNAH_ASR_DISPATCH_HAS_DOTPROD ? "neon-sdot"
              : MYNAH_ASR_DISPATCH_HAS_X86 ? "avx512vnni+avx2"
              : MYNAH_ASR_DISPATCH_HAS_NEON ? "neon-f32" : "scalar",
            quant_supported(),
            "MYNAH_ASR_CAPS", mynah_asr_qmat_int8_kernel(),
            MYNAH_ASR_DISPATCH_SRC_PREDICATE,
            "src/qmat.c mynah_asr_qmat_int8_kernel(); the T<=16 branch of "
            "mynah_asr_qmat_mul, k<=8192");

    row_set(&rows[n++], "kernel.int4_dot",
            MYNAH_ASR_DISPATCH_HAS_DOTPROD ? "neon-sdot"
              : MYNAH_ASR_DISPATCH_HAS_X86 ? "avx2"
              : MYNAH_ASR_DISPATCH_HAS_NEON ? "neon-f32" : "scalar",
            quant_supported(),
            "MYNAH_ASR_CAPS", mynah_asr_qmat_int4_kernel(),
            MYNAH_ASR_DISPATCH_SRC_PREDICATE,
            "src/qmat.c mynah_asr_qmat_int4_kernel(); no VNNI q4 kernel exists, "
            "VNNI hosts run the AVX2 one");

    /* --- int8 GEMM for large T: opt-in, and the reason it is opt-in ------- */
    {
        const int q = mynah_asr_qmat_qgemm();
        row_set(&rows[n++], "kernel.int8_gemm",
                q < 0 ? "no" : "yes",
                quant_supported(),
                "MYNAH_ASR_QGEMM",
                q < 0 ? "not-compiled" : q ? "ON" : "OFF",
                q < 0 ? MYNAH_ASR_DISPATCH_SRC_GATE : MYNAH_ASR_DISPATCH_SRC_PREDICATE,
                q < 0 ? "no native int8 kernel in this build: T>16 dequantizes and "
                        "calls sgemm"
                      : "src/qmat.c mynah_asr_qmat_qgemm(); default OFF, it loses to "
                        "dequant+sgemm on Apple AMX");
    }

    /* --- f32 GEMM provider: a pure build gate, and a complete answer ------ */
    row_set(&rows[n++], "gemm.f32", mynah_asr_gemm_provider(), "-", NULL,
            mynah_asr_gemm_provider(), MYNAH_ASR_DISPATCH_SRC_GATE,
            "src/backend.c mynah_asr_gemm_provider(); make BLAS=none|openblas|"
            "accelerate picks it, every f32 GEMM and GEMV in the runtime goes "
            "through that one seam, and 'own' means no cblas symbol is linked "
            "at all");

    /* --- our own sgemm: the family predicate, the counters, the self-test --
     * Three facts, and none of them re-derived: the family comes from the
     * predicate the runtime itself calls, on a shape this runtime really
     * issues, and the counters are what the dispatcher incremented. */
    if (strcmp(mynah_asr_gemm_provider(), "own") == 0) {
        mynah_asr_sgemm_stats st;
        mynah_asr_sgemm_stats_get(&st);
        const char *why = NULL;
        const mynah_asr_sgemm_family head =
            mynah_asr_sgemm_family_for(0, 1, 64u, 1024u, 1024u, &why);
        char res[32];
        snprintf(res, sizeof(res), "%s", mynah_asr_sgemm_isa_name());
        char reason[200];
        snprintf(reason, sizeof(reason),
                 "src/sgemm.c mynah_asr_sgemm_isa_name(): %s micro-kernels, 4 "
                 "rows, narrow/panel boundary at n=%zu derived from the "
                 "register file; family_for(64,1024,1024,transB) says '%s'",
                 mynah_asr_sgemm_isa_name(), mynah_asr_sgemm_narrow_max(),
                 mynah_asr_sgemm_family_name(head));
        row_set(&rows[n++], "gemm.f32_kernel", "yes", "yes",
                "MYNAH_ASR_SGEMM_PROFILE", res,
                MYNAH_ASR_DISPATCH_SRC_PREDICATE, reason);

        char calls[32];
        if (st.calls == 0ull) {
            snprintf(calls, sizeof(calls), "n/a");
            snprintf(reason, sizeof(reason),
                     "src/sgemm.c counters: no GEMM has run in this process "
                     "yet, so there is nothing to report. --dispatch-map loads "
                     "no model; read this row after a transcription");
        } else {
            snprintf(calls, sizeof(calls), "%llu calls", st.calls);
            snprintf(reason, sizeof(reason),
                     "src/sgemm.c counters: %llu dot, %llu narrow, %llu panel, "
                     "%llu matvec, %llu reference, %llu refused",
                     st.dot, st.narrow, st.panel, st.matvec, st.reference,
                     st.refused);
        }
        row_set(&rows[n++], "gemm.f32_families", "yes", "-", NULL, calls,
                MYNAH_ASR_DISPATCH_SRC_PREDICATE, reason);
    }

    /* --- Metal: compiled gate + a real device open ------------------------ */
#if MYNAH_ASR_DISPATCH_HAS_METAL
    {
        const int dev = mynah_asr_metal_available();
        const int on = mynah_asr_backend() == MYNAH_ASR_BACKEND_METAL;
        row_set(&rows[n++], "backend.metal", "yes", dev ? "yes" : "no", NULL,
                on ? "ON" : "OFF", MYNAH_ASR_DISPATCH_SRC_RUNTIME,
                dev ? "mynah_asr_metal_available() opened a device; selected with "
                      "--backend metal, GEMMs with T>=24"
                    : "mynah_asr_metal_available() found no device");
    }
#else
    row_set(&rows[n++], "backend.metal", "no", "-", NULL, "OFF",
            MYNAH_ASR_DISPATCH_SRC_GATE,
            "MYNAH_ASR_METAL not defined: no Metal code in this binary, so the "
            "host's GPU cannot be the answer");
#endif

#if MYNAH_ASR_DISPATCH_HAS_CUDA
    {
        const int dev = mynah_asr_cuda_available();
        const int on = mynah_asr_backend() == MYNAH_ASR_BACKEND_CUDA;
        row_set(&rows[n++], "backend.cuda", "yes", dev ? "yes" : "no", "MYNAH_ASR_CUDA_TF32",
                on ? "ON" : "OFF", MYNAH_ASR_DISPATCH_SRC_RUNTIME,
                dev ? "mynah_asr_cuda_available() found a device; selected with "
                      "--backend cuda"
                    : "mynah_asr_cuda_available() found no device");
    }
#else
    row_set(&rows[n++], "backend.cuda", "no", "-", "MYNAH_ASR_CUDA_TF32", "OFF",
            MYNAH_ASR_DISPATCH_SRC_GATE,
            "MYNAH_ASR_CUDA not defined (make cuda): no cuBLAS call in this binary");
#endif

    /* --- the pool and the BLAS quota: real calls, not the env -------------- */
    {
        char w[32];
        snprintf(w, sizeof(w), "%d", mynah_asr_num_threads());
        row_set(&rows[n++], "threads.pool", "yes", "yes", "MYNAH_ASR_THREADS", w,
                MYNAH_ASR_DISPATCH_SRC_RUNTIME,
                "src/threads.c mynah_asr_num_threads(); persistent pool, width "
                "clamped to [1,64]");
        snprintf(w, sizeof(w), "%d", mynah_asr_blas_budget());
        const int own = strcmp(mynah_asr_gemm_provider(), "own") == 0;
        row_set(&rows[n++], "threads.blas_budget", "yes",
                MYNAH_ASR_DISPATCH_HAS_OPENBLAS ? "yes" : "-",
                "OPENBLAS_NUM_THREADS", w, MYNAH_ASR_DISPATCH_SRC_RUNTIME,
                own ? "src/threads.c mynah_asr_blas_budget(); there is no second "
                      "pool in this build, so the budget IS the pool width and "
                      "mynah_asr_blas_set_concurrency() cannot move it"
                    : "src/threads.c mynah_asr_blas_budget(); Accelerate nests "
                      "through GCD and takes no knob, so on macOS this is "
                      "bookkeeping");
    }

    return n;
}

/* ====================================================== IDLE HARDWARE ===== */

/* A feature the CPU has that this binary never issues an instruction for.
 * Each zero is a statement: `want` names what would have to be WRITTEN, so the
 * footer is a work list and not a boast. */
typedef struct {
    const char *name;
    int         used;        /* 1 = a kernel in this binary really uses it   */
    const char *want;        /* what to write to stop it being idle          */
} idle_feature;

static const idle_feature *idle_table(int *count) {
#if defined(__aarch64__) || defined(__arm__)
    static const idle_feature arm[] = {
        {"dotprod", MYNAH_ASR_DISPATCH_HAS_DOTPROD,
         "SDOT int8/int4 dots in src/qmat.c (dot_q8_sdot, dot_q4_sdot)"},
        {"i8mm", 0,
         "an SMMLA int8 GEMM in src/qmat.c: 2x2 tiles for the T>16 path, the "
         "one kernel that would make MYNAH_ASR_QGEMM worth defaulting to"},
        {"bf16", 0,
         "a BFDOT/BFMMLA f32 GEMM in src/backend.c, or bf16 weights in "
         "src/weights.c"},
        {"sve", 0, "a vector-length-agnostic dot in src/qmat.c"},
        {"sve2", 0, "an SVE2 int8 dot in src/qmat.c"},
    };
    *count = (int)(sizeof(arm) / sizeof(arm[0]));
    return arm;
#elif MYNAH_ASR_DISPATCH_HAS_X86
    static const idle_feature x86[] = {
        {"avx2", MYNAH_ASR_DISPATCH_HAS_AVX2_KERNEL,
         "dot_q8_avx2 / dot_q4_x86 in src/qmat.c"},
        {"avx512f", 0,
         "an AVX-512 f32 kernel; today only the int8 dot is AVX-512 (VNNI)"},
        {"avx512vnni", MYNAH_ASR_DISPATCH_HAS_VNNI_KERNEL,
         "dot_q8_vnni in src/qmat.c (VPDPBUSD)"},
        {"avxvnni", 0,
         "a VEX-encoded VPDPBUSD twin of dot_q8_vnni for the AVX-512-less "
         "client parts"},
        {"amx", 0,
         "a tile-based int8 GEMM in src/qmat.c: no tile op exists anywhere in "
         "this runtime"},
    };
    *count = (int)(sizeof(x86) / sizeof(x86[0]));
    return x86;
#else
    *count = 0;
    return NULL;
#endif
}

/* ============================================================== printing == */

static void json_escape(FILE *out, const char *s) {
    for (; s && *s; s++) {
        if (*s == '"' || *s == '\\') fprintf(out, "\\%c", *s);
        else if (*s == '\n') fputs("\\n", out);
        else fputc(*s, out);
    }
}

int mynah_asr_dispatch_print(FILE *out, int json) {
    if (out == NULL) out = stderr;
    mynah_asr_dispatch_row rows[MYNAH_ASR_DISPATCH_MAX_ROWS];
    const int n = mynah_asr_dispatch_collect(rows, MYNAH_ASR_DISPATCH_MAX_ROWS);
    if (n < 0) return -1;

    int unknown = 0;
    for (int i = 0; i < n; i++)
        if (strcmp(rows[i].resolved, "UNKNOWN") == 0) unknown++;

    int n_idle = 0;
    const idle_feature *idle = idle_table(&n_idle);

    if (json) {
        fprintf(out, "{\"v\":1,\"build\":\"");
        json_escape(out, mynah_asr_build_id());
        fprintf(out, "\",\"blas\":\"%s\",\"simd\":\"%s\",\"rows\":[",
                mynah_asr_blas_provider(), mynah_asr_simd_profile());
        for (int i = 0; i < n; i++) {
            fprintf(out, "%s{\"feature\":\"%s\",\"compiled\":\"%s\",\"supported\":\"%s\","
                         "\"env\":\"%s\",\"env_value\":\"%s\",\"resolved\":\"%s\","
                         "\"source\":\"%s\",\"reason\":\"",
                    i ? "," : "", rows[i].id, rows[i].compiled, rows[i].supported,
                    rows[i].env_name ? rows[i].env_name : "", rows[i].env_value,
                    rows[i].resolved, src_tag(rows[i].source));
            json_escape(out, rows[i].reason);
            fputs("\"}", out);
        }
        fprintf(out, "],\"idle_hardware\":[");
        for (int i = 0; i < n_idle; i++) {
            const int have = mynah_asr_cpu_has(idle[i].name);
            fprintf(out, "%s{\"feature\":\"%s\",\"present\":\"%s\",\"used\":%d,"
                         "\"would_need\":\"",
                    i ? "," : "", idle[i].name, tri_name(have), idle[i].used);
            json_escape(out, idle[i].want);
            fputs("\"}", out);
        }
        fprintf(out, "],\"unknown_rows\":%d}\n", unknown);
        return unknown;
    }

    fprintf(out, "[DISPATCH-MAP] v=1 build=%s blas=%s simd=%s\n",
            mynah_asr_build_id(), mynah_asr_blas_provider(), mynah_asr_simd_profile());
    fprintf(out, "%-20s %-16s %-9s %-26s %-13s %s\n",
            "feature", "compiled", "supported", "env", "resolved", "reason");
    for (int i = 0; i < n; i++) {
        char env[96];
        if (rows[i].env_name)
            snprintf(env, sizeof(env), "%s=%s", rows[i].env_name, rows[i].env_value);
        else
            snprintf(env, sizeof(env), "-");
        fprintf(out, "%-20s %-16s %-9s %-26s %-13s %s %s\n", rows[i].id,
                rows[i].compiled, rows[i].supported, env, rows[i].resolved,
                src_tag(rows[i].source), rows[i].reason);
    }

    fprintf(out, "\nIDLE HARDWARE: what this CPU has that this binary never issues\n");
    for (int i = 0; i < n_idle; i++) {
        const int have = mynah_asr_cpu_has(idle[i].name);
        const char *state = have == MYNAH_ASR_CPU_YES
                                ? (idle[i].used ? "present, USED" : "present, IDLE")
                            : have == MYNAH_ASR_CPU_NO ? "absent" : "unknown";
        fprintf(out, "  %-11s %-14s %s\n", idle[i].name, state,
                have == MYNAH_ASR_CPU_YES && idle[i].used ? idle[i].want
                : have == MYNAH_ASR_CPU_NO ? "-"
                : idle[i].want);
    }
    fprintf(out, "\n%d row(s) UNKNOWN%s\n", unknown,
            unknown ? " — each is a predicate its owner still has to export" : "");
    return unknown;
}

/* ============================================================= ISA guard == */

int mynah_asr_isa_guard(void) {
    const char *built = NULL, *missing = NULL;

    /* Only DEFINITE absences, and only for instruction sets the compiler was
     * allowed to emit anywhere in the binary (-march=native), not for the
     * target-attributed kernels that dispatch on cpuid by themselves. */
#if MYNAH_ASR_DISPATCH_HAS_DOTPROD
    if (mynah_asr_cpu_has("dotprod") == MYNAH_ASR_CPU_NO) {
        built = "__ARM_FEATURE_DOTPROD (SDOT)";
        missing = "dotprod";
    }
#endif
#if MYNAH_ASR_DISPATCH_BUILT_I8MM
    if (missing == NULL && mynah_asr_cpu_has("i8mm") == MYNAH_ASR_CPU_NO) {
        built = "__ARM_FEATURE_MATMUL_INT8 (SMMLA)";
        missing = "i8mm";
    }
#endif
#if MYNAH_ASR_DISPATCH_BUILT_AVX512F
    if (missing == NULL && mynah_asr_cpu_has("avx512f") == MYNAH_ASR_CPU_NO) {
        built = "__AVX512F__";
        missing = "avx512f";
    }
#endif
#if MYNAH_ASR_DISPATCH_BUILT_AVX2
    if (missing == NULL && mynah_asr_cpu_has("avx2") == MYNAH_ASR_CPU_NO) {
        built = "__AVX2__";
        missing = "avx2";
    }
#endif
    if (missing == NULL) return 0;

    fprintf(stderr,
            "mynah-asr: this binary was compiled with %s but this CPU reports no "
            "%s — it would die with SIGILL. Rebuild without -march=native (make "
            "CFLAGS='-std=c11 -O3 ...') or run it on a host that has %s.\n",
            built, missing, missing);
    return 78;   /* EX_CONFIG */
}

/* ============================================================ self test === */

int mynah_asr_dispatch_self_test(char *error, size_t cap) {
    mynah_asr_dispatch_row rows[MYNAH_ASR_DISPATCH_MAX_ROWS];
    const int n = mynah_asr_dispatch_collect(rows, MYNAH_ASR_DISPATCH_MAX_ROWS);
    if (n <= 0) {
        snprintf(error, cap, "collect returned %d", n);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (rows[i].id == NULL || rows[i].id[0] == '\0') {
            snprintf(error, cap, "row %d has no id", i);
            return -1;
        }
        if (rows[i].resolved[0] == '\0' || rows[i].reason[0] == '\0') {
            snprintf(error, cap, "row %s has an empty resolved/reason", rows[i].id);
            return -1;
        }
        if (rows[i].source == MYNAH_ASR_DISPATCH_SRC_UNKNOWN &&
            strcmp(rows[i].resolved, "UNKNOWN") != 0) {
            snprintf(error, cap, "row %s claims an answer from no source", rows[i].id);
            return -1;
        }
        for (int j = 0; j < i; j++)
            if (strcmp(rows[i].id, rows[j].id) == 0) {
                snprintf(error, cap, "duplicate row id %s", rows[i].id);
                return -1;
            }
    }
    return 0;
}
