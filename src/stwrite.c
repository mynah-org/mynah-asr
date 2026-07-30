#include "stwrite.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[192], dtype[8];
    int64_t shape[4];
    int n_dims;
    const void *data;
    size_t bytes, offset;
} stw_entry;

struct mynah_asr_stw {
    stw_entry *entries;
    size_t n, cap, total;
};

mynah_asr_stw *mynah_asr_stw_new(void) { return calloc(1, sizeof(mynah_asr_stw)); }

int mynah_asr_stw_add(mynah_asr_stw *w, const char *name, const char *dtype,
                  const int64_t *shape, int n_dims, const void *data, size_t bytes) {
    if (w->n == w->cap) {
        w->cap = w->cap ? w->cap * 2 : 64;
        stw_entry *ne = realloc(w->entries, w->cap * sizeof(stw_entry));
        if (!ne) return -1;
        w->entries = ne;
    }
    stw_entry *e = &w->entries[w->n++];
    memset(e, 0, sizeof(*e));
    snprintf(e->name, sizeof(e->name), "%s", name);
    snprintf(e->dtype, sizeof(e->dtype), "%s", dtype);
    e->n_dims = n_dims > 4 ? 4 : n_dims;
    for (int i = 0; i < e->n_dims; i++) e->shape[i] = shape[i];
    e->data = data;
    e->bytes = bytes;
    e->offset = w->total;
    w->total += bytes;
    return 0;
}

/* Printf-style append into *buf starting at *len, growing *cap when
 * needed. The return value of vsnprintf is ALWAYS compared against the remaining
 * space BEFORE advancing *len: no silent truncation and no (cap - len) arithmetic
 * that could underflow. Returns 0, or -1 on OOM. */
static int stw_appendf(char **buf, size_t *cap, size_t *len, const char *fmt, ...) {
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        int need = vsnprintf(*buf + *len, *cap - *len, fmt, ap);
        va_end(ap);
        if (need < 0) return -1;
        if ((size_t)need < *cap - *len) { *len += (size_t)need; return 0; }
        /* truncated: double until it fits (need + NUL) and retry */
        size_t ncap = *cap ? *cap : 256;
        while (ncap <= *len + (size_t)need) ncap *= 2;
        char *nb = realloc(*buf, ncap);
        if (!nb) return -1;
        *buf = nb;
        *cap = ncap;
    }
}

int mynah_asr_stw_write(mynah_asr_stw *w, const char *path) {
    /* hand-built JSON header: the names are controlled (HF tensors), so there is
     * no exotic escaping to handle. The buffer grows on its own through
     * stw_appendf, so the initial capacity is only an estimate. */
    size_t hcap = 256 + w->n * 256;
    char *hdr = malloc(hcap);
    if (!hdr) return -1;
    size_t hl = 0;
    int err = stw_appendf(&hdr, &hcap, &hl, "{");
    for (size_t i = 0; i < w->n && !err; i++) {
        const stw_entry *e = &w->entries[i];
        err = stw_appendf(&hdr, &hcap, &hl,
                          "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[", i ? "," : "",
                          e->name, e->dtype);
        for (int d = 0; d < e->n_dims && !err; d++)
            err = stw_appendf(&hdr, &hcap, &hl, "%s%lld", d ? "," : "",
                              (long long)e->shape[d]);
        if (!err)
            err = stw_appendf(&hdr, &hcap, &hl, "],\"data_offsets\":[%zu,%zu]}",
                              e->offset, e->offset + e->bytes);
    }
    if (!err) err = stw_appendf(&hdr, &hcap, &hl, "}");
    /* pad to a multiple of 8 with spaces (safetensors convention) */
    while (!err && hl % 8 != 0) err = stw_appendf(&hdr, &hcap, &hl, " ");
    if (err) { free(hdr); return -1; }

    FILE *f = fopen(path, "wb");
    if (!f) { free(hdr); return -1; }
    uint64_t hlen = (uint64_t)hl;
    int ok = fwrite(&hlen, 8, 1, f) == 1 && fwrite(hdr, 1, hl, f) == hl;
    for (size_t i = 0; ok && i < w->n; i++)
        ok = fwrite(w->entries[i].data, 1, w->entries[i].bytes, f) == w->entries[i].bytes;
    free(hdr);
    return fclose(f) == 0 && ok ? 0 : -1;
}

void mynah_asr_stw_free(mynah_asr_stw *w) {
    if (!w) return;
    free(w->entries);
    free(w);
}
