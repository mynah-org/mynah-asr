/* Model-free self-test of the flag registry and the dispatch report (S3-1/S3-2).
 *
 * What it guards is not arithmetic, it is a promise: that the two lines a
 * measurement run prints about itself are complete and honest. Concretely —
 * the registry has no duplicate and no undescribed row; the banner really
 * prints the [FLAGS] / [EFFECTIVE-CONFIG] prefixes; a flag that CANNOT act in
 * this build is reported IGNORED with the reason rather than echoed back as if
 * it had been applied; and the dispatch map resolves every row from a
 * predicate, a runtime call or a pure gate, never from compiled && supported.
 *
 * Exit: 0 ok, 1 fail. Runs in CI with no model and no network. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/dispatch.h"
#include "../src/flags.h"

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("flags FAIL: %s\n", msg); failures = 1; } \
    else printf("flags ok:   %s\n", msg); } while (0)

/* The banner goes to a FILE*, so capture it the portable way: a temp file. */
static char *capture(void (*fn)(FILE *), size_t *len) {
    FILE *f = tmpfile();
    if (!f) return NULL;
    fn(f);
    fflush(f);
    const long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    const size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);
    if (len) *len = got;
    return buf;
}

static void print_flags(FILE *f) { mynah_asr_flags_print(f); }
static void print_dispatch(FILE *f) { mynah_asr_dispatch_print(f, 0); }
static void print_dispatch_json(FILE *f) { mynah_asr_dispatch_print(f, 1); }

int main(void) {
    /* ---- the table itself ------------------------------------------------ */
    const int n = mynah_asr_flags_count();
    CHECK(n > 0, "the registry is not empty");

    int described = 1, duplicated = 0, scoped = 1;
    for (int i = 0; i < n; i++) {
        const mynah_asr_flag *f = mynah_asr_flags_get(i);
        if (!f || !f->name || !f->desc || f->desc[0] == '\0' || !f->dflt) described = 0;
        if (!f || strcmp(mynah_asr_flag_scope_name(f->scope), "?") == 0) scoped = 0;
        for (int j = 0; j < i; j++) {
            const mynah_asr_flag *g = mynah_asr_flags_get(j);
            if (f && g && strcmp(f->name, g->name) == 0) duplicated = 1;
        }
    }
    CHECK(described, "every entry has a name, a default and a description");
    CHECK(!duplicated, "no duplicate name in the registry");
    CHECK(scoped, "every entry has a known scope");
    CHECK(mynah_asr_flags_find("MYNAH_ASR_THREADS") != NULL, "lookup by name works");
    CHECK(mynah_asr_flags_find("MYNAH_ASR_NOT_A_FLAG") == NULL, "an unknown name is not found");

    /* The registry-mediated read must refuse a name nobody registered, rather
     * than quietly becoming a second, undocumented getenv. */
    setenv("MYNAH_ASR_NOT_A_FLAG", "7", 1);
    CHECK(mynah_asr_flag_int("MYNAH_ASR_NOT_A_FLAG", 3) == 3, /* check_flag_registry: ignore */
          "mynah_asr_flag_int ignores an unregistered name");
    setenv("MYNAH_ASR_THREADS", "4", 1);
    CHECK(mynah_asr_flag_int("MYNAH_ASR_THREADS", 0) == 4,
          "mynah_asr_flag_int reads a registered name");

    /* ---- the two lines --------------------------------------------------- */
    size_t len = 0;
    char *out = capture(print_flags, &len);
    CHECK(out != NULL, "the banner is printable");
    if (out) {
        CHECK(strncmp(out, "[FLAGS] v=1", 11) == 0, "line 1 is [FLAGS] v=1");
        const char *l2 = strchr(out, '\n');
        CHECK(l2 && strncmp(l2 + 1, "[EFFECTIVE-CONFIG] v=1 ", 23) == 0,
              "line 2 is [EFFECTIVE-CONFIG] v=1");
        int lines = 0;
        for (const char *p = out; *p; p++) if (*p == '\n') lines++;
        CHECK(lines == 2, "exactly two lines, both terminated");
        CHECK(strstr(out, "build=") && strstr(out, "blas=") && strstr(out, "simd="),
              "line 2 carries build, blas and simd");
        CHECK(strstr(out, "MYNAH_ASR_THREADS=4->") != NULL,
              "a set flag appears as requested->effective");
        free(out);
    }

    /* An inert flag must be named IGNORED with its reason. Pick one that is
     * inert on THIS build: MYNAH_ASR_CAPS off x86, OPENBLAS_NUM_THREADS against
     * Accelerate, MYNAH_ASR_METAL_PROF without Metal, MYNAH_ASR_CUDA_TF32
     * without CUDA. At least one of those is always inert somewhere, and the
     * test only asserts about the ones that really are. */
    setenv("MYNAH_ASR_CAPS", "vnni", 1);
    setenv("OPENBLAS_NUM_THREADS", "3", 1);
    setenv("MYNAH_ASR_METAL_PROF", "1", 1);
    setenv("MYNAH_ASR_CUDA_TF32", "1", 1);
    out = capture(print_flags, &len);
    CHECK(out != NULL, "the banner is printable with flags set");
    if (out) {
        int checked = 0, wrong = 0;
        static const char *names[] = {"MYNAH_ASR_CAPS", "OPENBLAS_NUM_THREADS",
                                      "MYNAH_ASR_METAL_PROF", "MYNAH_ASR_CUDA_TF32"};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            const mynah_asr_flag *f = mynah_asr_flags_find(names[i]);
            if (!f) { wrong = 1; continue; }
            char needle[64];
            snprintf(needle, sizeof(needle), " %s=", f->name);
            const char *at = strstr(out, needle);
            if (!at) { wrong = 1; continue; }
            /* the [FLAGS] line lists it; the [EFFECTIVE-CONFIG] one judges it */
            const char *l2 = strchr(out, '\n');
            const char *tok = l2 ? strstr(l2, needle) : NULL;
            if (!tok) { wrong = 1; continue; }
            const char *end = strchr(tok + 1, ' ');
            size_t span = end ? (size_t)(end - tok) : strlen(tok);
            char tokbuf[512];
            if (span >= sizeof(tokbuf)) span = sizeof(tokbuf) - 1;
            memcpy(tokbuf, tok, span);
            tokbuf[span] = '\0';
            const int says_ignored = strstr(tokbuf, "(IGNORED:_") != NULL;
            if (f->inert && f->inert() != NULL) {
                checked++;
                if (!says_ignored) wrong = 1;
            } else if (says_ignored) {
                wrong = 1;   /* a flag that DOES act must never be called ignored */
            }
        }
        CHECK(checked > 0, "at least one flag is inert on this build and was checked");
        CHECK(!wrong, "every inert flag is reported IGNORED with a reason, and no other");
        free(out);
    }
    unsetenv("MYNAH_ASR_CAPS");
    unsetenv("OPENBLAS_NUM_THREADS");
    unsetenv("MYNAH_ASR_METAL_PROF");
    unsetenv("MYNAH_ASR_CUDA_TF32");

    /* ---- the dispatch map ------------------------------------------------ */
    char err[256] = "";
    CHECK(mynah_asr_dispatch_self_test(err, sizeof(err)) == 0, err[0] ? err : "dispatch self-test");

    mynah_asr_dispatch_row rows[MYNAH_ASR_DISPATCH_MAX_ROWS];
    const int nr = mynah_asr_dispatch_collect(rows, MYNAH_ASR_DISPATCH_MAX_ROWS);
    CHECK(nr >= 8, "the map has a row per logical feature");
    int unknown = 0;
    for (int i = 0; i < nr; i++)
        if (strcmp(rows[i].resolved, "UNKNOWN") == 0) {
            unknown++;
            printf("            UNKNOWN row: %s (%s)\n", rows[i].id, rows[i].reason);
        }
    /* UNKNOWN is a finding, not a failure: it is named above and counted here,
     * which is the whole contract of dispatch.h. */
    printf("flags note: %d UNKNOWN row(s) on this host\n", unknown);

    out = capture(print_dispatch, &len);
    CHECK(out && strncmp(out, "[DISPATCH-MAP] v=1 ", 19) == 0, "the map prints its header");
    CHECK(out && strstr(out, "IDLE HARDWARE:") != NULL, "the map prints the IDLE HARDWARE footer");
    free(out);
    out = capture(print_dispatch_json, &len);
    CHECK(out && out[0] == '{' && strstr(out, "\"idle_hardware\"") != NULL,
          "--json emits one JSON document with the footer");
    free(out);

    /* The guard must not refuse on the very host that built this binary. */
    CHECK(mynah_asr_isa_guard() == 0, "the ISA guard passes on the build host");

    printf(failures ? "flags: FAIL\n" : "flags: ok\n");
    return failures;
}
