/* flags.h — ONE registry of every environment variable this runtime reads.
 *
 * WHY THIS EXISTS.  ENGINEERING.md §5: a benchmark is invalid until dispatch is
 * proven, and part of "dispatch" is the environment the process was started
 * with.  Until now that environment was discoverable only by grepping for
 * getenv(), which is how a flag that does nothing in a given build (MYNAH_ASR_CAPS
 * on ARM, OPENBLAS_NUM_THREADS against Accelerate, MYNAH_ASR_QGEMM without a
 * native int8 kernel) can be set, believed, and quoted in a result.
 *
 * The registry is therefore a table, not a convention: one row per variable,
 * with a scope, a default, a one-line description, and an OPTIONAL `inert`
 * predicate that returns the REASON the flag has no effect in THIS binary on
 * THIS host.  `tools/check_flag_registry.py` fails the build when a name read
 * through getenv() anywhere under src/ cli/ server/ tests/ is missing here, or
 * when this table names a flag nothing reads.
 *
 * WHAT IT IS NOT.  It is not a configuration system: the call sites keep their
 * own getenv() and their own caching.  A row describes a read that happens
 * elsewhere; mynah_asr_flag_int()/_str() are conveniences for new call sites,
 * not a required funnel.  The registry's guarantee is completeness of the
 * DESCRIPTION, which is what a result banner needs.
 */
#ifndef MYNAH_ASR_FLAGS_H
#define MYNAH_ASR_FLAGS_H

#include <stdio.h>
#include <stddef.h>

/* Where the flag acts.  `kernel` steers a compute kernel and therefore a
 * measurement; `runtime` steers the process; `server` is read only by
 * mynah-asr-server; `debug` exists for diagnostics and test harnesses. */
enum {
    MYNAH_ASR_FLAG_RUNTIME = 0,
    MYNAH_ASR_FLAG_KERNEL  = 1,
    MYNAH_ASR_FLAG_SERVER  = 2,
    MYNAH_ASR_FLAG_DEBUG   = 3
};

/* Returns NULL when the flag really acts in this build on this host, or a
 * short static reason why it cannot.  A predicate may probe the CPU; it must
 * not allocate and must not change any dispatch state. */
typedef const char *(*mynah_asr_flag_inert_fn)(void);

/* Writes into `out` the value the runtime will ACTUALLY use given `requested`.
 * Returns 0 when the request was applied verbatim, 1 when it was clamped or
 * downgraded.  NULL means "the request is the effective value". */
typedef int (*mynah_asr_flag_eff_fn)(const char *requested, char *out, size_t cap);

typedef struct {
    const char             *name;
    int                     scope;   /* MYNAH_ASR_FLAG_*                      */
    const char             *dflt;    /* what happens when it is unset         */
    const char             *desc;    /* one line, English, no trailing stop   */
    mynah_asr_flag_inert_fn inert;   /* optional                              */
    mynah_asr_flag_eff_fn   effective; /* optional                            */
} mynah_asr_flag;

int                    mynah_asr_flags_count(void);
const mynah_asr_flag  *mynah_asr_flags_get(int i);
const mynah_asr_flag  *mynah_asr_flags_find(const char *name);
const char            *mynah_asr_flag_scope_name(int scope);

/* Registry-mediated reads, for call sites that want them: identical semantics
 * to getenv()+atoi()/getenv(), except that an unregistered name is a
 * programming error the check script catches. */
int         mynah_asr_flag_int(const char *name, int dflt);
const char *mynah_asr_flag_str(const char *name, const char *dflt);

/* Exactly two lines, both machine-readable, both terminated by '\n':
 *
 *   [FLAGS] v=1 <NAME>=<value> ...              every registered flag SET in
 *                                               the environment (nothing else)
 *   [EFFECTIVE-CONFIG] v=1 build=<git rev> blas=<accelerate|openblas|none>
 *       simd=<compiled ISA profile> <NAME>=<requested>-><effective>(<reason>) ...
 *
 * where <reason> is `applied`, `clamped`, or `IGNORED: <inert reason>`.  Only
 * flags that are SET get a token on the second line: an unset flag has no
 * request to report, and inventing one would be the same fiction this file
 * exists to prevent.  Whitespace inside a value is replaced by '_' so the
 * lines stay token-separated. */
void mynah_asr_flags_print(FILE *out);

/* The build's BLAS provider and compiled SIMD profile, as printed above.
 * Pure compile-time gates, exported so the dispatch report shares one
 * definition with the banner instead of keeping a second copy. */
const char *mynah_asr_blas_provider(void);
const char *mynah_asr_simd_profile(void);
const char *mynah_asr_build_id(void);

#endif
