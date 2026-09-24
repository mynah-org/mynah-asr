/* kvcache.h — the streaming attention K/V cache, and three physical layouts of
 * the SAME logical contents (CACHE-RING-1, S13-1d / S10-5).
 *
 * THE LOGICAL CONTRACT, identical in every layout.  Per layer and per tensor
 * (K, V) the cache holds `valid` rows, 0 <= valid <= left: the projected keys or
 * values of the last `valid` encoder frames, oldest first.  A chunk of Q fresh
 * rows is attended over the window [cache ++ fresh] (valid + Q rows), and after
 * the chunk the cache holds the LAST min(valid + Q, left) rows of that window.
 *
 * THE LAYOUTS, which differ only in where the rows live and in what is copied:
 *
 *   shift (0, default)  physical row i == logical row i.  The window is
 *                       GATHERED into the caller's scratch every layer
 *                       (valid + Q rows), and the commit memmoves the kept old
 *                       rows down to row 0 (left - Q rows once saturated).
 *                       This is the behaviour before CACHE-RING-1, byte for
 *                       byte, and it stays the default until an A/B says
 *                       otherwise.
 *   ring (1)            logical row i lives at physical (head + i) % left.
 *                       The window is gathered as two spans plus the fresh
 *                       rows -- the same bytes as shift -- and the commit
 *                       writes only the Q fresh rows at the tail.  Removes
 *                       the memmove, keeps the gather.
 *   slide (2)           each layer's arena holds cap = left + slack rows and
 *                       logical row i lives at physical head + i.  The fresh
 *                       rows are written straight after the valid ones, so the
 *                       window IS the arena at `head` and nothing is gathered;
 *                       committing is moving `head`.  When the next chunk
 *                       would run off the end, the valid rows are moved back
 *                       to row 0 once (`prepare`), every ~slack/Q chunks.
 *                       Removes the memmove AND the gather, for slack extra
 *                       rows per layer per tensor.
 *
 * WHY NO LAYOUT CAN CHANGE A FLOAT.  The attention consumes the window as a
 * contiguous [valid + Q, d] row-major block with leading dimension d in every
 * layout; the values in it are the same values in the same order.  Only the
 * address differs, and every address is a multiple of d floats from an
 * allocation base.  tests/test_kv_layout asserts this with memcmp on the
 * logical contents and on every encoder output of every chunk.
 *
 * The softmax is untouched: the window is ONE contiguous block in every layout,
 * so no layout ever evaluates attention over split spans.
 *
 * Per-stream, single-threaded: a stream is only ever stepped by one thread.
 */
#ifndef MYNAH_ASR_KVCACHE_H
#define MYNAH_ASR_KVCACHE_H

#include <stddef.h>

enum {
    MYNAH_ASR_KV_SHIFT = 0,
    MYNAH_ASR_KV_RING  = 1,
    MYNAH_ASR_KV_SLIDE = 2,
    MYNAH_ASR_KV__N
};

typedef struct {
    int layout;                 /* MYNAH_ASR_KV_*                               */
    int n_layers, left, d;
    int qmax;                   /* the largest chunk a step may append          */
    int cap;                    /* physical rows per layer per tensor           */
    int valid;                  /* logical rows held, 0..left                   */
    int head;                   /* physical row of logical row 0 (ring, slide)  */
    float *k, *v;               /* [n_layers, cap, d]                           */
    unsigned long long bytes;   /* bytes copied by this cache (instrumentation):
                                   gather + commit + compaction                 */
    unsigned long long flushed; /* part of `bytes` already in the process total */
} mynah_asr_kv;

/* MYNAH_ASR_KV_LAYOUT = shift|ring|slide (or 0|1|2); unset = shift.  Read once. */
int         mynah_asr_kv_layout_default(void);
const char *mynah_asr_kv_layout_name(int layout);

int  mynah_asr_kv_init(mynah_asr_kv *kv, int layout, int n_layers, int left, int qmax, int d);
void mynah_asr_kv_free(mynah_asr_kv *kv);
/* Empty cache, every allocation kept. */
void mynah_asr_kv_reset(mynah_asr_kv *kv);

/* Once per step, BEFORE any layer: makes room for Q fresh rows (slide
 * compaction; a no-op elsewhere).  -1 when Q is outside 1..qmax. */
int  mynah_asr_kv_prepare(mynah_asr_kv *kv, int Q);

/* Per layer and tensor (which: 0 = K, 1 = V): the contiguous window
 * [valid + Q, d] = cache ++ fresh.  shift/ring gather into `scratch` (room for
 * left + qmax rows) and return it; slide writes `fresh` into the arena and
 * returns the arena. */
const float *mynah_asr_kv_window(mynah_asr_kv *kv, int li, int which, const float *fresh,
                                 int Q, float *scratch);

/* Per layer and tensor, AFTER the attention has consumed the window: stores
 * the fresh rows.  Does not change valid/head (the other layers of this step
 * still need the pre-step geometry); `advance` does. */
void mynah_asr_kv_commit(mynah_asr_kv *kv, int li, int which, const float *fresh, int Q);

/* Once per step, AFTER every layer. */
void mynah_asr_kv_advance(mynah_asr_kv *kv, int Q);

/* Process-wide bytes copied by every cache, flushed once per step by
 * `advance` (one relaxed atomic per stream step, not per layer), and the layout
 * of the most recently initialised cache. The server dump prints both: proof
 * of which layout RAN, and the traffic measured rather than computed. */
unsigned long long mynah_asr_kv_bytes_total(void);
int                mynah_asr_kv_layout_last(void);

/* Logical row i (0 = oldest) of a layer's K (which 0) or V (which 1). */
const float *mynah_asr_kv_row(const mynah_asr_kv *kv, int li, int which, int i);

/* The logical contents, [n_layers, valid, d] into dst (n_layers*left*d floats
 * suffice).  For tests: two layouts are equal iff these bytes are. */
void mynah_asr_kv_logical(const mynah_asr_kv *kv, int which, float *dst);

#endif
