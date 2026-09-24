/* kvcache.c — see kvcache.h for the logical contract and the three layouts. */
#include "kvcache.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static _Atomic unsigned long long g_bytes;
static _Atomic int g_layout_last = -1;

unsigned long long mynah_asr_kv_bytes_total(void) {
    return atomic_load_explicit(&g_bytes, memory_order_relaxed);
}

int mynah_asr_kv_layout_last(void) {
    return atomic_load_explicit(&g_layout_last, memory_order_relaxed);
}

int mynah_asr_kv_layout_default(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("MYNAH_ASR_KV_LAYOUT");
        v = MYNAH_ASR_KV_SHIFT;
        if (e && (!strcmp(e, "ring") || !strcmp(e, "1"))) v = MYNAH_ASR_KV_RING;
        else if (e && (!strcmp(e, "slide") || !strcmp(e, "2"))) v = MYNAH_ASR_KV_SLIDE;
    }
    return v;
}

const char *mynah_asr_kv_layout_name(int layout) {
    switch (layout) {
        case MYNAH_ASR_KV_SHIFT: return "shift";
        case MYNAH_ASR_KV_RING:  return "ring";
        case MYNAH_ASR_KV_SLIDE: return "slide";
        default:                 return "?";
    }
}

static size_t rows_bytes(const mynah_asr_kv *kv, int rows) {
    return (size_t)rows * (size_t)kv->d * sizeof(float);
}

static float *layer_base(const mynah_asr_kv *kv, int li, int which) {
    return (which ? kv->v : kv->k) + (size_t)li * (size_t)kv->cap * (size_t)kv->d;
}

int mynah_asr_kv_init(mynah_asr_kv *kv, int layout, int n_layers, int left, int qmax, int d) {
    memset(kv, 0, sizeof(*kv));
    if (layout < 0 || layout >= MYNAH_ASR_KV__N || n_layers < 1 || left < 0 || qmax < 1 || d < 1)
        return -1;
    /* a ring over zero rows has no head to advance: left 0 keeps nothing, which
     * is exactly what shift does, so it is shift */
    if (left == 0 && layout == MYNAH_ASR_KV_RING) layout = MYNAH_ASR_KV_SHIFT;
    kv->layout = layout;
    kv->n_layers = n_layers;
    kv->left = left;
    kv->d = d;
    kv->qmax = qmax;
    /* slide: slack >= qmax so one chunk always fits after a compaction, and
     * slack >= left so a compaction (left rows) happens at most every left/Q
     * chunks -- amortised, Q rows per chunk, the same as the fresh write */
    kv->cap = layout == MYNAH_ASR_KV_SLIDE ? left + (left > qmax ? left : qmax) : left;
    const size_t n = (size_t)n_layers * (size_t)(kv->cap > 0 ? kv->cap : 1) * (size_t)d;
    kv->k = calloc(n, sizeof(float));
    kv->v = calloc(n, sizeof(float));
    if (!kv->k || !kv->v) { mynah_asr_kv_free(kv); return -1; }
    atomic_store_explicit(&g_layout_last, layout, memory_order_relaxed);
    return 0;
}

void mynah_asr_kv_free(mynah_asr_kv *kv) {
    free(kv->k); free(kv->v);
    kv->k = kv->v = NULL;
}

void mynah_asr_kv_reset(mynah_asr_kv *kv) {
    const size_t n = (size_t)kv->n_layers * (size_t)(kv->cap > 0 ? kv->cap : 1) * (size_t)kv->d;
    memset(kv->k, 0, n * sizeof(float));
    memset(kv->v, 0, n * sizeof(float));
    kv->valid = 0;
    kv->head = 0;
}

int mynah_asr_kv_prepare(mynah_asr_kv *kv, int Q) {
    if (Q < 1 || Q > kv->qmax) return -1;
    if (kv->layout != MYNAH_ASR_KV_SLIDE || kv->head + kv->valid + Q <= kv->cap) return 0;
    /* compaction: the valid rows back to row 0, every layer, both tensors.
     * memmove: with slack < left the two ranges could overlap */
    for (int li = 0; li < kv->n_layers; li++)
        for (int w = 0; w < 2; w++) {
            float *b = layer_base(kv, li, w);
            memmove(b, b + (size_t)kv->head * (size_t)kv->d, rows_bytes(kv, kv->valid));
        }
    kv->bytes += 2ull * (unsigned long long)kv->n_layers * rows_bytes(kv, kv->valid);
    kv->head = 0;
    return 0;
}

const float *mynah_asr_kv_window(mynah_asr_kv *kv, int li, int which, const float *fresh,
                                 int Q, float *scratch) {
    float *b = layer_base(kv, li, which);
    const size_t d = (size_t)kv->d;
    if (kv->layout == MYNAH_ASR_KV_SLIDE) {
        float *w = b + (size_t)kv->head * d;
        memcpy(w + (size_t)kv->valid * d, fresh, rows_bytes(kv, Q));
        kv->bytes += rows_bytes(kv, Q);
        return w;
    }
    if (kv->layout == MYNAH_ASR_KV_RING) {
        const int n1 = kv->valid < kv->cap - kv->head ? kv->valid : kv->cap - kv->head;
        memcpy(scratch, b + (size_t)kv->head * d, rows_bytes(kv, n1));
        memcpy(scratch + (size_t)n1 * d, b, rows_bytes(kv, kv->valid - n1));
    } else {
        memcpy(scratch, b, rows_bytes(kv, kv->valid));
    }
    memcpy(scratch + (size_t)kv->valid * d, fresh, rows_bytes(kv, Q));
    kv->bytes += rows_bytes(kv, kv->valid + Q);
    return scratch;
}

void mynah_asr_kv_commit(mynah_asr_kv *kv, int li, int which, const float *fresh, int Q) {
    float *b = layer_base(kv, li, which);
    const size_t d = (size_t)kv->d;
    const int left = kv->left, valid = kv->valid;
    if (kv->layout == MYNAH_ASR_KV_SLIDE) return;         /* the window placed them */

    if (kv->layout == MYNAH_ASR_KV_RING) {
        /* Physical row of absolute frame a is a % left, so the fresh row j goes
         * to (tail + j) % left with tail = (head + valid) % left. A chunk longer
         * than the ring keeps only its last `left` rows. At most two spans. */
        const int keep = Q < left ? Q : left;
        const float *src = fresh + (size_t)(Q - keep) * d;
        const int pos = (kv->head + valid + (Q - keep)) % left;
        const int n1 = keep < left - pos ? keep : left - pos;
        memcpy(b + (size_t)pos * d, src, rows_bytes(kv, n1));
        memcpy(b, src + (size_t)n1 * d, rows_bytes(kv, keep - n1));
        kv->bytes += rows_bytes(kv, keep);
        return;
    }

    /* shift: the pre-CACHE-RING-1 update_kv_cache, verbatim */
    const int total = valid + Q;
    const int keep = total < left ? total : left;
    const int from_old = keep - Q > 0 ? keep - Q : 0;      /* old rows to keep */
    const int drop_old = valid - from_old;                  /* old rows to drop */
    if (from_old > 0 && drop_old > 0) {
        memmove(b, b + (size_t)drop_old * d, rows_bytes(kv, from_old));
        kv->bytes += rows_bytes(kv, from_old);
    }
    const int fresh_keep = keep - from_old;                 /* new rows to keep (<= Q) */
    memcpy(b + (size_t)from_old * d, fresh + (size_t)(Q - fresh_keep) * d,
           rows_bytes(kv, fresh_keep));
    kv->bytes += rows_bytes(kv, fresh_keep);
}

void mynah_asr_kv_advance(mynah_asr_kv *kv, int Q) {
    const int total = kv->valid + Q;
    const int nv = total < kv->left ? total : kv->left;
    if (kv->layout == MYNAH_ASR_KV_RING && kv->left > 0)
        kv->head = (kv->head + total - nv) % kv->left;
    else if (kv->layout == MYNAH_ASR_KV_SLIDE)
        kv->head += total - nv;
    kv->valid = nv;
    atomic_fetch_add_explicit(&g_bytes, kv->bytes - kv->flushed, memory_order_relaxed);
    kv->flushed = kv->bytes;
}

const float *mynah_asr_kv_row(const mynah_asr_kv *kv, int li, int which, int i) {
    const float *b = layer_base(kv, li, which);
    int p = i;
    if (kv->layout == MYNAH_ASR_KV_RING) p = (kv->head + i) % kv->left;
    else if (kv->layout == MYNAH_ASR_KV_SLIDE) p = kv->head + i;
    return b + (size_t)p * (size_t)kv->d;
}

void mynah_asr_kv_logical(const mynah_asr_kv *kv, int which, float *dst) {
    for (int li = 0; li < kv->n_layers; li++)
        for (int i = 0; i < kv->valid; i++)
            memcpy(dst + ((size_t)li * (size_t)kv->valid + (size_t)i) * (size_t)kv->d,
                   mynah_asr_kv_row(kv, li, which, i), rows_bytes(kv, 1));
}
