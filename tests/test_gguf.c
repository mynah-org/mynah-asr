/* Self-test of the GGUF loader (no model required — it runs in CI too):
 * builds synthetic GGUFs in /tmp and checks
 *   1. parsing of the valid file: reversed dims (ggml ne[0] = the fastest one),
 *      F32 zero-copy bit-esatto, dequant F16/BF16/Q8_0/Q4_0 entro tolleranza;
 *   2. that malformed files (magic, v1, truncated, offset past EOF, absurd
 *      string, alignment not a power of 2, unknown ggml type) are rejected
 *      cleanly (NULL, no crash) — the same class of harness used to validate
 *      the original parser (keyra).
 * Exit: 0 ok, 1 fail. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/weights.h"

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("gguf FAIL: %s\n", msg); failures = 1; } \
    else printf("gguf ok:   %s\n", msg); } while (0)

/* --------------------------------------------------- costruzione file */
typedef struct { unsigned char *p; size_t len, cap; } buf;

static void put(buf *b, const void *src, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2 + 256;
        b->p = realloc(b->p, b->cap);
    }
    memcpy(b->p + b->len, src, n);
    b->len += n;
}
static void put_u32(buf *b, uint32_t v) { unsigned char x[4] = {v, v >> 8, v >> 16, v >> 24}; put(b, x, 4); }
static void put_u64(buf *b, uint64_t v) {
    unsigned char x[8];
    for (int i = 0; i < 8; i++) x[i] = (unsigned char)(v >> (8 * i));
    put(b, x, 8);
}
static void put_str(buf *b, const char *s) { put_u64(b, strlen(s)); put(b, s, strlen(s)); }
static void pad_to(buf *b, size_t align) {
    static const unsigned char z[64];
    while (b->len % align != 0) put(b, z, 1);
}

static uint16_t f32_to_f16(float f) { /* solo per i fixture (valori normali) */
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int e = (int)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t frac = (bits >> 13) & 0x3ffu;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)e << 10) | frac);
}

static const char *write_tmp(const buf *b, char *path_out) {
    strcpy(path_out, "/tmp/mynah_asr_test_gguf_XXXXXX");
    int fd = mkstemp(path_out);
    if (fd < 0) return NULL;
    ssize_t w = write(fd, b->p, b->len);
    close(fd);
    return w == (ssize_t)b->len ? path_out : NULL;
}

/* v3 header with 1 metadata entry (general.alignment=32) */
static void hdr(buf *b, uint32_t version, uint64_t n_tensors) {
    put_u32(b, 0x46554747u);
    put_u32(b, version);
    put_u64(b, n_tensors);
    put_u64(b, 1);
    put_str(b, "general.alignment");
    put_u32(b, 4);
    put_u32(b, 32);
}

static void tinfo(buf *b, const char *name, int rank, const uint64_t *ne,
                  uint32_t type, uint64_t off) {
    put_str(b, name);
    put_u32(b, (uint32_t)rank);
    for (int i = 0; i < rank; i++) put_u64(b, ne[i]);
    put_u32(b, type);
    put_u64(b, off);
}

/* ------------------------------------------------------------- test */
static float src_val(int i) { return (float)(i - 32) * 0.03125f; } /* [-1, 1] */

int main(void) {
    char path[64], junk[64];

    /* ---- file valido: f32 4x8 + f16/bf16/q8_0/q4_0 da 64 elementi + q4_k 256 ---- */
    enum { N = 64, NK = 256 };
    float vals[N];
    for (int i = 0; i < N; i++) vals[i] = src_val(i);

    buf b = {0};
    hdr(&b, 3, 6);
    const uint64_t ne_f32[2] = {8, 4};    /* ggml: ne[0]=8 veloce -> mynah [4][8] */
    const uint64_t ne_v[1] = {N};
    const uint64_t ne_k[1] = {NK};
    /* payload: f32 (128B) | f16 (128B) | bf16 (128B) | q8_0 (68B, pad) | q4_0 | q4_k */
    tinfo(&b, "t.f32", 2, ne_f32, 0, 0);
    tinfo(&b, "t.f16", 1, ne_v, 1, 128);
    tinfo(&b, "t.bf16", 1, ne_v, 30, 256);
    tinfo(&b, "t.q8", 1, ne_v, 8, 384);
    tinfo(&b, "t.q4", 1, ne_v, 2, 480);   /* 384+68 -> pad a 480 */
    tinfo(&b, "t.q4k", 1, ne_k, 12, 544); /* 480+36 -> pad a 544, 1 super-blocco */
    pad_to(&b, 32);

    for (int i = 0; i < 32; i++) put(&b, &vals[i], 4);          /* t.f32: 4x8 = 32 elem */
    for (int i = 0; i < N; i++) { uint16_t h = f32_to_f16(vals[i]); put(&b, &h, 2); }
    for (int i = 0; i < N; i++) {
        uint32_t bits;
        memcpy(&bits, &vals[i], 4);
        uint16_t h = (uint16_t)(bits >> 16);                     /* bf16 = troncamento */
        put(&b, &h, 2);
    }
    for (int blk = 0; blk < 2; blk++) {                          /* q8_0: d=1/127 * amax */
        float amax = 0;
        for (int i = 0; i < 32; i++) if (fabsf(vals[blk * 32 + i]) > amax) amax = fabsf(vals[blk * 32 + i]);
        float d = amax / 127.0f;
        uint16_t dh = f32_to_f16(d);
        put(&b, &dh, 2);
        for (int i = 0; i < 32; i++) {
            signed char q = (signed char)lrintf(vals[blk * 32 + i] / (d > 0 ? d : 1));
            put(&b, &q, 1);
        }
    }
    pad_to(&b, 32);                                              /* q4_0 a offset 480 */
    for (int blk = 0; blk < 2; blk++) {
        float maxs = 0;
        for (int i = 0; i < 32; i++) if (fabsf(vals[blk * 32 + i]) > fabsf(maxs)) maxs = vals[blk * 32 + i];
        float d = maxs / -8.0f;
        uint16_t dh = f32_to_f16(d);
        put(&b, &dh, 2);
        for (int i = 0; i < 16; i++) {
            int lo = (int)lrintf(vals[blk * 32 + i] / (d != 0 ? d : 1)) + 8;
            int hi = (int)lrintf(vals[blk * 32 + i + 16] / (d != 0 ? d : 1)) + 8;
            lo = lo < 0 ? 0 : lo > 15 ? 15 : lo;
            hi = hi < 0 ? 0 : hi > 15 ? 15 : hi;
            unsigned char p = (unsigned char)(lo | (hi << 4));
            put(&b, &p, 1);
        }
    }

    /* q4_k at offset 544: 1 super-block with known sc/min/q; the expectation is
     * computed from the spec formula x = d*sc*q - dmin*m (it validates the
     * layout: the 6-bit packing and the i|i+32 nibble order per group of 64) */
    static const float KD = 0.5f, KDMIN = 0.25f;
    unsigned char ksc[8], kmn[8], kq[NK];
    float expect_k[NK];
    for (int j = 0; j < 8; j++) { ksc[j] = (unsigned char)(j + 1); kmn[j] = (unsigned char)(8 - j); }
    for (int i = 0; i < NK; i++) kq[i] = (unsigned char)((i * 7) % 16);
    for (int i = 0; i < NK; i++) {
        int j = i / 32;
        expect_k[i] = KD * (float)ksc[j] * (float)kq[i] - KDMIN * (float)kmn[j];
    }
    pad_to(&b, 32);
    {
        uint16_t dh = f32_to_f16(KD), dminh = f32_to_f16(KDMIN);
        put(&b, &dh, 2);
        put(&b, &dminh, 2);
        unsigned char scales[12];
        for (int j = 0; j < 4; j++) {                     /* pack 6-bit ggml */
            scales[j]     = (unsigned char)(ksc[j] | ((ksc[j + 4] >> 4) << 6));
            scales[j + 4] = (unsigned char)(kmn[j] | ((kmn[j + 4] >> 4) << 6));
            scales[j + 8] = (unsigned char)((ksc[j + 4] & 0x0f) | ((kmn[j + 4] & 0x0f) << 4));
        }
        put(&b, scales, 12);
        for (int base = 0; base < NK; base += 64)          /* nibble i | i+32 */
            for (int i = 0; i < 32; i++) {
                unsigned char p = (unsigned char)(kq[base + i] | (kq[base + i + 32] << 4));
                put(&b, &p, 1);
            }
    }

    CHECK(write_tmp(&b, path) != NULL, "fixture scritto");
    mynah_asr_safetensors *st = mynah_asr_st_open(path);
    CHECK(st != NULL, "GGUF valido aperto (via mynah_asr_st_open)");
    if (st) {
        CHECK(mynah_asr_st_count(st) == 6, "count == 6");
        const mynah_asr_tensor *f32 = mynah_asr_st_get(st, "t.f32");
        CHECK(f32 && f32->n_dims == 2 && f32->shape[0] == 4 && f32->shape[1] == 8,
              "dims ggml invertite -> [4][8]");
        CHECK(f32 && f32->n_elems == 32 && memcmp(f32->data, vals, 32 * 4) == 0,
              "f32 zero-copy bit-esatto");
        struct { const char *name; double tol; } deq[] = {
            {"t.f16", 1e-3}, {"t.bf16", 8e-3}, {"t.q8", 1.5e-2}, {"t.q4", 1e-1},
        };
        for (size_t k = 0; k < sizeof(deq) / sizeof(deq[0]); k++) {
            const mynah_asr_tensor *t = mynah_asr_st_get(st, deq[k].name);
            double worst = 1e9;
            if (t && t->n_elems == N) {
                worst = 0;
                for (int i = 0; i < N; i++) {
                    double e = fabs((double)((const float *)t->data)[i] - (double)vals[i]);
                    if (e > worst) worst = e;
                }
            }
            char msg[128];
            snprintf(msg, sizeof(msg), "dequant %s (max err %.2e, tol %.0e)",
                     deq[k].name, worst, deq[k].tol);
            CHECK(worst <= deq[k].tol, msg);
        }
        const mynah_asr_tensor *tqk = mynah_asr_st_get(st, "t.q4k");
        double worst_k = 1e9;
        if (tqk && tqk->n_elems == NK) {
            worst_k = 0;
            for (int i = 0; i < NK; i++) {
                double e = fabs((double)((const float *)tqk->data)[i] - (double)expect_k[i]);
                if (e > worst_k) worst_k = e;
            }
        }
        char kmsg[128];
        snprintf(kmsg, sizeof(kmsg), "dequant t.q4k vs formula spec (max err %.2e)", worst_k);
        CHECK(worst_k <= 1e-6, kmsg);   /* d/dmin esatti in f16: atteso 0 */
        mynah_asr_st_close(st);
    }
    unlink(path);

    /* ---- malformati: tutti rifiutati senza crash ---- */
    struct { const char *what; buf f; } bad[7];
    memset(bad, 0, sizeof(bad));

    bad[0].what = "wrong magic";
    hdr(&bad[0].f, 3, 0);
    bad[0].f.p[0] = 'X';

    bad[1].what = "version 1 (u32 layout)";
    hdr(&bad[1].f, 1, 0);

    bad[2].what = "truncated mid-header";
    hdr(&bad[2].f, 3, 1);
    bad[2].f.len = 30;

    bad[3].what = "tensor offset past EOF";
    hdr(&bad[3].f, 3, 1);
    { const uint64_t ne[1] = {8}; tinfo(&bad[3].f, "t", 1, ne, 0, 1u << 30); pad_to(&bad[3].f, 32); }

    bad[4].what = "absurd string length";
    hdr(&bad[4].f, 3, 1);
    put_u64(&bad[4].f, (uint64_t)1 << 60);

    bad[5].what = "alignment not a power of 2";
    put_u32(&bad[5].f, 0x46554747u);
    put_u32(&bad[5].f, 3);
    put_u64(&bad[5].f, 0);
    put_u64(&bad[5].f, 1);
    put_str(&bad[5].f, "general.alignment");
    put_u32(&bad[5].f, 4);
    put_u32(&bad[5].f, 33);

    bad[6].what = "unsupported ggml type (Q6_K: interop milestone)";
    hdr(&bad[6].f, 3, 1);
    { const uint64_t ne[1] = {256}; tinfo(&bad[6].f, "t", 1, ne, 14, 0); pad_to(&bad[6].f, 32);
      static const unsigned char blk[210] = {0}; put(&bad[6].f, blk, 210); }

    for (size_t k = 0; k < sizeof(bad) / sizeof(bad[0]); k++) {
        CHECK(write_tmp(&bad[k].f, junk) != NULL, "fixture malformato scritto");
        fprintf(stderr, "--- expected error (%s):\n", bad[k].what);
        mynah_asr_safetensors *s = mynah_asr_st_open(junk);
        CHECK(s == NULL, bad[k].what);
        if (s) mynah_asr_st_close(s);
        unlink(junk);
        free(bad[k].f.p);
    }
    free(b.p);

    printf(failures ? "test_gguf: FAIL\n" : "test_gguf: OK\n");
    return failures;
}
