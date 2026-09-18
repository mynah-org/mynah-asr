/* malloc_count — a preloaded/inserted shared library that counts the process's
 * malloc/calloc/realloc calls and prints the total when the process exits.
 *
 * It exists for one gate: the per-chunk streaming step must allocate NOTHING
 * after warm-up (S1-3). The count is a whole-process total; the per-chunk number
 * is obtained by differencing two runs of the SAME command on inputs of
 * different length (tests/test_stream_allocs.sh), which needs no hook inside the
 * library under test — nothing in src/ knows this file exists.
 *
 *   macOS : DYLD_INSERT_LIBRARIES + DYLD_FORCE_FLAT_NAMESPACE, __DATA,__interpose
 *           tuples. The replacements call malloc_zone_* (what libSystem's malloc
 *           calls underneath) so they can never re-enter their own interposition.
 *           free() is NOT interposed: default-zone blocks free normally.
 *   Linux : LD_PRELOAD + dlsym(RTLD_NEXT). dlsym itself allocates while we are
 *           still resolving it, so those first allocations are served from a
 *           small static arena and counted like any other.
 *
 * Output: "[malloc_count] calls=<n>" on stderr, or into $MALLOC_COUNT_OUT.
 * Also exported as mynah_asr_malloc_count() for a driver that wants to sample it. */

#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static atomic_ulong g_calls;

unsigned long mynah_asr_malloc_count(void);
unsigned long mynah_asr_malloc_count(void) {
    return atomic_load_explicit(&g_calls, memory_order_relaxed);
}

static inline void bump(void) { atomic_fetch_add_explicit(&g_calls, 1, memory_order_relaxed); }

#if defined(__APPLE__)

#include <malloc/malloc.h>

#define DYLD_INTERPOSE(_replacement, _replacee)                                       \
    __attribute__((used)) static struct {                                             \
        const void *replacement;                                                      \
        const void *replacee;                                                         \
    } _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = {        \
        (const void *)(unsigned long)&_replacement,                                    \
        (const void *)(unsigned long)&_replacee}

static void *mc_malloc(size_t n) {
    bump();
    return malloc_zone_malloc(malloc_default_zone(), n);
}
static void *mc_calloc(size_t n, size_t sz) {
    bump();
    return malloc_zone_calloc(malloc_default_zone(), n, sz);
}
static void *mc_realloc(void *p, size_t n) {
    bump();
    return malloc_zone_realloc(malloc_default_zone(), p, n);
}

DYLD_INTERPOSE(mc_malloc, malloc);
DYLD_INTERPOSE(mc_calloc, calloc);
DYLD_INTERPOSE(mc_realloc, realloc);

#else /* Linux and other ELF hosts: LD_PRELOAD */

#include <dlfcn.h>

static void *(*real_malloc)(size_t);
static void *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);

/* dlsym() allocates: serve it from here until the real symbols are resolved */
#define BOOT_ARENA 65536
static char g_boot[BOOT_ARENA];
static size_t g_boot_used;
static int g_resolving;

static void *boot_alloc(size_t n) {
    n = (n + 15u) & ~(size_t)15;
    if (g_boot_used + n > BOOT_ARENA) return NULL;
    void *p = g_boot + g_boot_used;
    g_boot_used += n;
    return p;
}
static int from_boot(const void *p) {
    return (const char *)p >= g_boot && (const char *)p < g_boot + BOOT_ARENA;
}

static void resolve(void) {
    if (real_malloc || g_resolving) return;
    g_resolving = 1;
    real_malloc = (void *(*)(size_t))dlsym(RTLD_NEXT, "malloc");
    real_calloc = (void *(*)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
    real_realloc = (void *(*)(void *, size_t))dlsym(RTLD_NEXT, "realloc");
    g_resolving = 0;
}

void *malloc(size_t n) {
    bump();
    resolve();
    if (!real_malloc) return boot_alloc(n);
    return real_malloc(n);
}

void *calloc(size_t n, size_t sz) {
    bump();
    resolve();
    if (!real_calloc) {
        void *p = boot_alloc(n * sz);
        if (p) memset(p, 0, n * sz);
        return p;
    }
    return real_calloc(n, sz);
}

void *realloc(void *p, size_t n) {
    bump();
    resolve();
    if (from_boot(p)) {                 /* grow out of the arena */
        void *q = real_malloc ? real_malloc(n) : boot_alloc(n);
        if (q && p) memcpy(q, p, n);    /* the arena block is at least as small */
        return q;
    }
    if (!real_realloc) return boot_alloc(n);
    return real_realloc(p, n);
}

void free(void *p) {
    if (from_boot(p)) return;           /* arena memory is never returned */
    static void (*real_free)(void *);
    if (!real_free) real_free = (void (*)(void *))dlsym(RTLD_NEXT, "free");
    if (real_free) real_free(p);
}

#endif

__attribute__((destructor)) static void mc_report(void) {
    const unsigned long n = mynah_asr_malloc_count();
    const char *path = getenv("MALLOC_COUNT_OUT");
    if (path && path[0]) {
        FILE *f = fopen(path, "w");
        if (f) {
            fprintf(f, "%lu\n", n);
            fclose(f);
            return;
        }
    }
    fprintf(stderr, "[malloc_count] calls=%lu\n", n);
}
