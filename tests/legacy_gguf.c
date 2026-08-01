#include "legacy_gguf.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Value types of the GGUF metadata (the KV store in the header). */
enum {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4,
    GGUF_I32 = 5, GGUF_F32 = 6, GGUF_BOOL = 7, GGUF_STRING = 8, GGUF_ARRAY = 9,
    GGUF_U64 = 10, GGUF_I64 = 11, GGUF_F64 = 12
};

/* ggml tensor types (only the ones the runtime accepts; anything else -> error). */
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q8_0 = 8,
       GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13,
       GGML_Q6_K = 14, GGML_BF16 = 30 };

struct legacy_asr_gguf {
    int fd;
    uint64_t size;
    uint64_t data_base;
    uint64_t alignment;
    unsigned char *map;
    legacy_asr_tensor *tensors;
    char **names;          /* owned by us (in safetensors they point into the JSON) */
    float **dequant;       /* f32 buffers of the non-F32 tensors (NULL when zero-copy) */
    uint32_t *ggml_type;
    size_t count;
};

/* Overflow-checked arithmetic: a hostile header must not drag us into UB. */
static int add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (b > UINT64_MAX - a) return -1;
    *out = a + b;
    return 0;
}
static int mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return -1;
    *out = a * b;
    return 0;
}

typedef struct { int fd; uint64_t size; uint64_t pos; } reader;

static int rd_bytes(reader *r, void *dst, uint64_t n) {
    uint64_t end;
    if (add_u64(r->pos, n, &end) != 0 || end > r->size) return -1;
    unsigned char *out = dst;
    while (n != 0) {
        size_t chunk = n > (1u << 20) ? (1u << 20) : (size_t)n;
        ssize_t got = pread(r->fd, out, chunk, (off_t)r->pos);
        if (got <= 0) return -1;
        r->pos += (uint64_t)got;
        out += got;
        n -= (uint64_t)got;
    }
    return 0;
}

/* explicit little-endian reads: independent of host endianness */
static int rd_u32(reader *r, uint32_t *v) {
    unsigned char b[4];
    if (rd_bytes(r, b, 4) != 0) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}
static int rd_u64(reader *r, uint64_t *v) {
    unsigned char b[8];
    if (rd_bytes(r, b, 8) != 0) return -1;
    *v = 0;
    for (int i = 7; i >= 0; i--) *v = (*v << 8) | b[i];
    return 0;
}

static int rd_string(reader *r, char **out) {
    uint64_t len;
    if (rd_u64(r, &len) != 0 || len > (1u << 20) || len > r->size - r->pos) return -1;
    char *s = malloc((size_t)len + 1);
    if (!s) return -1;
    if (rd_bytes(r, s, len) != 0) { free(s); return -1; }
    s[len] = '\0';
    *out = s;
    return 0;
}

static int skip_value(reader *r, uint32_t type, unsigned depth) {
    if (depth > 32) return -1;
    uint64_t n = 0;
    switch (type) {
    case GGUF_U8: case GGUF_I8: case GGUF_BOOL: n = 1; break;
    case GGUF_U16: case GGUF_I16: n = 2; break;
    case GGUF_U32: case GGUF_I32: case GGUF_F32: n = 4; break;
    case GGUF_U64: case GGUF_I64: case GGUF_F64: n = 8; break;
    case GGUF_STRING: {
        uint64_t len;
        if (rd_u64(r, &len) != 0 || len > r->size - r->pos) return -1;
        n = len;
        break;
    }
    case GGUF_ARRAY: {
        uint32_t elem;
        uint64_t cnt;
        if (rd_u32(r, &elem) != 0 || rd_u64(r, &cnt) != 0 || cnt > 100000000u) return -1;
        for (uint64_t i = 0; i < cnt; i++)
            if (skip_value(r, elem, depth + 1) != 0) return -1;
        return 0;
    }
    default: return -1;
    }
    if (n > r->size - r->pos) return -1;
    r->pos += n;
    return 0;
}

/* Of the metadata only general.alignment matters: the model config does NOT come
 * from the GGUF but from mynah.json (extra KVs are skipped, not an error). */
static int rd_metadata(legacy_asr_gguf *g, reader *r, uint64_t count) {
    if (count > 100000000u) return -1;
    for (uint64_t i = 0; i < count; i++) {
        char *key = NULL;
        uint32_t type;
        if (rd_string(r, &key) != 0 || rd_u32(r, &type) != 0) { free(key); return -1; }
        int rc = 0;
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_U32) {
            uint32_t a = 0;   /* on a read error rc aborts, but don't copy
                                 garbage into the alignment either way */
            rc = rd_u32(r, &a);
            g->alignment = a;
        } else if (strcmp(key, "general.alignment") == 0 && type == GGUF_U64) {
            rc = rd_u64(r, &g->alignment);
        } else {
            rc = skip_value(r, type, 0);
        }
        free(key);
        if (rc != 0) return -1;
    }
    return 0;
}

static int type_geometry(uint32_t type, uint64_t *block_elems, uint64_t *block_bytes) {
    switch (type) {
    case GGML_F32:  *block_elems = 1;  *block_bytes = 4;  return 0;
    case GGML_F16:  case GGML_BF16: *block_elems = 1; *block_bytes = 2; return 0;
    case GGML_Q8_0: *block_elems = 32; *block_bytes = 34; return 0;
    case GGML_Q4_0: *block_elems = 32; *block_bytes = 18; return 0;
    case GGML_Q2_K: *block_elems = 256; *block_bytes = 84; return 0;
    case GGML_Q3_K: *block_elems = 256; *block_bytes = 110; return 0;
    case GGML_Q4_K: *block_elems = 256; *block_bytes = 144; return 0;
    case GGML_Q5_K: *block_elems = 256; *block_bytes = 176; return 0;
    case GGML_Q6_K: *block_elems = 256; *block_bytes = 210; return 0;
    default: return -1;
    }
}

static float f16_to_f32(uint16_t v) {
    uint32_t sign = (uint32_t)(v & 0x8000u) << 16;
    int e = (v >> 10) & 0x1f;
    uint32_t frac = v & 0x03ffu;
    uint32_t bits;
    if (e == 0) {
        if (frac == 0) bits = sign;
        else { /* subnormal: normalize */
            e = 1;
            while ((frac & 0x0400u) == 0) { frac <<= 1; e--; }
            bits = sign | ((uint32_t)(e + 112) << 23) | ((frac & 0x03ffu) << 13);
        }
    } else if (e == 31) {
        bits = sign | 0x7f800000u | (frac << 13);
    } else {
        bits = sign | ((uint32_t)(e + 112) << 23) | (frac << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t ld_u16(const unsigned char *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }

/* Q4_K/Q5_K: decoding of the 8 six-bit scales/mins packed into 12 bytes
 * (ggml layout, get_scale_min_k4; ported from keyra/gguf_quant.c, validated
 * there against real files). */
static void q4k_scale_min(const unsigned char *scales, int idx,
                          unsigned char *sc, unsigned char *mn) {
    if (idx < 4) {
        *sc = scales[idx] & 63u;
        *mn = scales[idx + 4] & 63u;
    } else {
        *sc = (unsigned char)((scales[idx + 4] & 0x0fu) | ((scales[idx - 4] >> 6) << 4));
        *mn = (unsigned char)((scales[idx + 4] >> 4) | ((scales[idx] >> 6) << 4));
    }
}

/* ggml -> f32 dequant. The blocks are sequential over the row-major tensor
 * (the fastest dim ne[0] is a multiple of 32: checked beforehand). */
static void dequant(uint32_t type, const unsigned char *src, size_t n_elems, float *dst) {
    switch (type) {
    case GGML_F16:
        for (size_t i = 0; i < n_elems; i++) dst[i] = f16_to_f32(ld_u16(src + 2 * i));
        break;
    case GGML_BF16:
        for (size_t i = 0; i < n_elems; i++) {
            uint32_t bits = (uint32_t)ld_u16(src + 2 * i) << 16;
            memcpy(&dst[i], &bits, 4);
        }
        break;
    case GGML_Q8_0:  /* 34B block: d f16 + 32 int8 */
        for (size_t b = 0; b < n_elems / 32; b++) {
            const unsigned char *blk = src + b * 34;
            const float d = f16_to_f32(ld_u16(blk));
            const signed char *q = (const signed char *)(blk + 2);
            for (int i = 0; i < 32; i++) dst[b * 32 + i] = d * (float)q[i];
        }
        break;
    case GGML_Q4_0:  /* 18B block: d f16 + 16B nibbles (j and j+16 packed) */
        for (size_t b = 0; b < n_elems / 32; b++) {
            const unsigned char *blk = src + b * 18;
            const float d = f16_to_f32(ld_u16(blk));
            const unsigned char *q = blk + 2;
            for (int i = 0; i < 16; i++) {
                dst[b * 32 + i]      = d * (float)((int)(q[i] & 0x0f) - 8);
                dst[b * 32 + i + 16] = d * (float)((int)(q[i] >> 4) - 8);
            }
        }
        break;
    case GGML_Q2_K:  /* 84B super-block: 16B scales/mins (4+4 bits) + 64B of 2-bit
                      * quants + d,dmin f16. x = d*sc*q - dmin*m over 16 sub-blocks
                      * of 16; the 2-bit fields of one byte belong to FOUR different
                      * sub-blocks, advancing by shift (0,2,4,6) */
        for (size_t b = 0; b < n_elems / 256; b++) {
            const unsigned char *blk = src + b * 84;
            const unsigned char *scales = blk;
            const unsigned char *q = blk + 16;
            const float d = f16_to_f32(ld_u16(blk + 80));
            const float dmin = f16_to_f32(ld_u16(blk + 82));
            float *out = dst + b * 256;
            int is = 0, o = 0;
            for (int base = 0; base < 256; base += 128) {
                for (int shift = 0; shift < 8; shift += 2) {
                    for (int half = 0; half < 32; half += 16) {
                        const unsigned char sc = scales[is++];
                        const float dl = d * (float)(sc & 0x0f);
                        const float ml = dmin * (float)(sc >> 4);
                        for (int l = 0; l < 16; l++)
                            out[o++] = dl * (float)((q[base / 4 + half + l] >> shift) & 3) - ml;
                    }
                }
            }
        }
        break;
    case GGML_Q3_K:  /* 110B super-block: 32B high-bit mask + 64B of 2-bit quants +
                      * 12B of 6-bit scales + d f16. x = d*(sc-32)*(q - 4*(hbit==0)):
                      * the high bit is INVERTED, so a clear bit subtracts 4 */
        for (size_t b = 0; b < n_elems / 256; b++) {
            const unsigned char *blk = src + b * 110;
            const unsigned char *hmask = blk;
            const unsigned char *q = blk + 32;
            const unsigned char *sc6 = blk + 96;
            const float d = f16_to_f32(ld_u16(blk + 108));
            float *out = dst + b * 256;
            /* the 16 six-bit scales live in 12 bytes: low 4 bits in [0..7], the
             * missing 2 high bits packed 4-per-byte in [8..11] */
            signed char sc[16];
            for (int i = 0; i < 16; i++) {
                const int lo = (i < 8) ? (sc6[i] & 0x0f) : (sc6[i - 8] >> 4);
                const int hi = (sc6[8 + (i % 4)] >> (2 * (i / 4))) & 3;
                sc[i] = (signed char)((lo | (hi << 4)) - 32);
            }
            int is = 0, o = 0;
            unsigned char m = 1;
            for (int base = 0; base < 256; base += 128) {
                for (int shift = 0; shift < 8; shift += 2) {
                    for (int half = 0; half < 32; half += 16) {
                        const float dl = d * (float)sc[is++];
                        for (int l = 0; l < 16; l++) {
                            const int lo2 = (q[base / 4 + half + l] >> shift) & 3;
                            /* hmask is 32 bytes = one bit per element: the BYTE index
                             * stays in 0..31 and the walking bit `m` (1..128) is what
                             * distinguishes the two 128-element halves */
                            const int sub = (hmask[half + l] & m) ? 0 : 4;
                            out[o++] = dl * (float)(lo2 - sub);
                        }
                    }
                    m <<= 1;
                }
            }
        }
        break;
    case GGML_Q4_K:  /* 144B super-block: d,dmin f16 + 12B scales/mins + 128B nibbles
                      * x = d*sc*q - dmin*m over 8 sub-blocks of 32 */
        for (size_t b = 0; b < n_elems / 256; b++) {
            const unsigned char *blk = src + b * 144;
            const float d = f16_to_f32(ld_u16(blk));
            const float dmin = f16_to_f32(ld_u16(blk + 2));
            const unsigned char *scales = blk + 4;
            const unsigned char *q = blk + 16;
            float *out = dst + b * 256;
            for (int base = 0, si = 0; base < 256; base += 64, si += 2) {
                unsigned char sc0, mn0, sc1, mn1;
                q4k_scale_min(scales, si, &sc0, &mn0);
                q4k_scale_min(scales, si + 1, &sc1, &mn1);
                const float d0 = d * sc0, m0 = dmin * mn0;
                const float d1 = d * sc1, m1 = dmin * mn1;
                for (int i = 0; i < 32; i++) {
                    out[base + i]      = d0 * (float)(q[i] & 0x0f) - m0;
                    out[base + i + 32] = d1 * (float)(q[i] >> 4) - m1;
                }
                q += 32;
            }
        }
        break;
    case GGML_Q5_K:  /* 176B super-block: Q4_K + a 5th bit per quant in qh[32].
                      * x = d*sc*(q|bit<<4) - dmin*m; the qh bit pair advances by
                      * 2 every group of 64 (l:1,2 -> 4,8 -> 16,32 -> 64,128) */
        for (size_t b = 0; b < n_elems / 256; b++) {
            const unsigned char *blk = src + b * 176;
            const float d = f16_to_f32(ld_u16(blk));
            const float dmin = f16_to_f32(ld_u16(blk + 2));
            const unsigned char *scales = blk + 4;
            const unsigned char *qh = blk + 16;
            const unsigned char *q = blk + 48;
            float *out = dst + b * 256;
            unsigned u1 = 1, u2 = 2;
            for (int base = 0, si = 0; base < 256; base += 64, si += 2) {
                unsigned char sc0, mn0, sc1, mn1;
                q4k_scale_min(scales, si, &sc0, &mn0);
                q4k_scale_min(scales, si + 1, &sc1, &mn1);
                const float d0 = d * sc0, m0 = dmin * mn0;
                const float d1 = d * sc1, m1 = dmin * mn1;
                for (int i = 0; i < 32; i++) {
                    const int hi0 = (qh[i] & u1) ? 16 : 0;
                    const int hi1 = (qh[i] & u2) ? 16 : 0;
                    out[base + i]      = d0 * (float)((q[i] & 0x0f) + hi0) - m0;
                    out[base + i + 32] = d1 * (float)((q[i] >> 4) + hi1) - m1;
                }
                q += 32;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        break;
    case GGML_Q6_K:  /* 210B super-block: ql[128] low nibbles + qh[64] bit pairs +
                      * 16 SIGNED 8-bit scales + d f16. x = d*sc*(q-32), no mins.
                      * Two halves of 128 elements, each with 8 scales. */
        for (size_t b = 0; b < n_elems / 256; b++) {
            const unsigned char *blk = src + b * 210;
            const unsigned char *ql = blk;
            const unsigned char *qh = blk + 128;
            const signed char *sc = (const signed char *)(blk + 192);
            const float d = f16_to_f32(ld_u16(blk + 208));
            float *out = dst + b * 256;
            for (int half = 0; half < 2; half++) {
                for (int i = 0; i < 32; i++) {
                    const int is = i / 16;
                    const int q1 = (int)((ql[i]      & 0x0f) | (((qh[i] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql[i + 32] & 0x0f) | (((qh[i] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((ql[i]      >> 4)   | (((qh[i] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql[i + 32] >> 4)   | (((qh[i] >> 6) & 3) << 4)) - 32;
                    out[i]      = d * (float)sc[is]     * (float)q1;
                    out[i + 32] = d * (float)sc[is + 2] * (float)q2;
                    out[i + 64] = d * (float)sc[is + 4] * (float)q3;
                    out[i + 96] = d * (float)sc[is + 6] * (float)q4;
                }
                out += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        break;
    }
}

static uint64_t align_up(uint64_t v, uint64_t a, int *valid) {
    /* alignment: a power of 2, within reason (spec default: 32) */
    if (a == 0 || a > (1u << 20) || (a & (a - 1)) != 0 || v > UINT64_MAX - a + 1) {
        *valid = 0;
        return 0;
    }
    *valid = 1;
    return (v + a - 1) & ~(a - 1);
}

legacy_asr_gguf *legacy_asr_gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "gguf: cannot open %s\n", path); return NULL; }
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < 24) { close(fd); return NULL; }

    legacy_asr_gguf *g = calloc(1, sizeof(*g));
    if (!g) { close(fd); return NULL; }
    g->fd = fd;
    g->size = (uint64_t)sb.st_size;
    g->alignment = 32;

    reader r = {fd, g->size, 0};
    uint32_t magic, version;
    uint64_t n_tensors, n_meta;
    if (rd_u32(&r, &magic) != 0 || magic != 0x46554747u /* "GGUF" LE */ ||
        rd_u32(&r, &version) != 0 || version < 2 || version > 3 ||  /* v1: u32 layout, unsupported */
        rd_u64(&r, &n_tensors) != 0 || rd_u64(&r, &n_meta) != 0 ||
        n_tensors > 1000000u || rd_metadata(g, &r, n_meta) != 0) {
        fprintf(stderr, "gguf: invalid header in %s\n", path);
        goto fail;
    }

    g->count = (size_t)n_tensors;
    g->tensors = calloc(g->count ? g->count : 1, sizeof(*g->tensors));
    g->names = calloc(g->count ? g->count : 1, sizeof(*g->names));
    g->dequant = calloc(g->count ? g->count : 1, sizeof(*g->dequant));
    g->ggml_type = calloc(g->count ? g->count : 1, sizeof(*g->ggml_type));
    if (!g->tensors || !g->names || !g->dequant || !g->ggml_type) goto fail;

    /* tensor info: name, dims (ggml convention: ne[0] = the fastest one ->
     * reversed with respect to row-major safetensors), type, relative offset */
    uint64_t *offsets = calloc(g->count ? g->count : 1, sizeof(*offsets));
    if (!offsets) goto fail;
    for (size_t i = 0; i < g->count; i++) {
        legacy_asr_tensor *t = &g->tensors[i];
        uint32_t rank, type;
        uint64_t ne[8];
        if (rd_string(&r, &g->names[i]) != 0 || rd_u32(&r, &rank) != 0 || rank > 8) goto fail_off;
        for (uint32_t d = 0; d < rank; d++)
            if (rd_u64(&r, &ne[d]) != 0) goto fail_off;
        if (rd_u32(&r, &type) != 0 || rd_u64(&r, &offsets[i]) != 0) goto fail_off;

        uint64_t be, bb, elems = 1;
        for (uint32_t d = 0; d < rank; d++)
            if (ne[d] == 0 || mul_u64(elems, ne[d], &elems) != 0) goto fail_off;
        if (type_geometry(type, &be, &bb) != 0 || elems % be != 0 ||
            (be > 1 && rank > 0 && ne[0] % be != 0)) { /* blocks never straddle rows */
            fprintf(stderr, "gguf: tensor '%s' has unsupported ggml type %u\n",
                    g->names[i], type);
            goto fail_off;
        }
        t->name = g->names[i];
        t->dtype = LEGACY_ASR_DT_F32;                /* after the load it is always f32 */
        t->n_dims = (int)rank;
        for (uint32_t d = 0; d < rank; d++) t->shape[d] = (int64_t)ne[rank - 1 - d];
        t->n_elems = (size_t)elems;
        g->ggml_type[i] = type;
        (void)bb;   /* the real byte count is validated after the mmap, against the file range */
    }

    int valid;
    g->data_base = align_up(r.pos, g->alignment, &valid);
    if (!valid || g->data_base > g->size) goto fail_off;

    g->map = mmap(NULL, (size_t)g->size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (g->map == MAP_FAILED) { g->map = NULL; goto fail_off; }

    for (size_t i = 0; i < g->count; i++) {
        legacy_asr_tensor *t = &g->tensors[i];
        uint64_t be, bb, bytes, abs_off, end;
        /* unknown ggml type: without this check we divided by a `be` that was
         * never written (UB) — caught by GCC 13 on the first Linux build */
        if (type_geometry(g->ggml_type[i], &be, &bb) != 0 ||
            mul_u64(t->n_elems / be, bb, &bytes) != 0 ||
            add_u64(g->data_base, offsets[i], &abs_off) != 0 ||
            add_u64(abs_off, bytes, &end) != 0 || end > g->size ||
            offsets[i] % g->alignment != 0) {
            fprintf(stderr, "gguf: payload of tensor '%s' lies outside the file\n", t->name);
            goto fail_off;
        }
        const unsigned char *src = g->map + abs_off;
        if (g->ggml_type[i] == GGML_F32) {
            t->data = src;                       /* zero-copy from the mmap */
        } else {                                 /* dequant to f32 at load */
            if (t->n_elems > SIZE_MAX / sizeof(float)) goto fail_off;
            g->dequant[i] = malloc(t->n_elems * sizeof(float));
            if (!g->dequant[i]) goto fail_off;
            dequant(g->ggml_type[i], src, t->n_elems, g->dequant[i]);
            t->data = g->dequant[i];
        }
    }
    free(offsets);
    return g;

fail_off:
    free(offsets);
fail:
    fprintf(stderr, "gguf: failed to load %s\n", path);
    legacy_asr_gguf_close(g);
    return NULL;
}

void legacy_asr_gguf_close(legacy_asr_gguf *g) {
    if (!g) return;
    for (size_t i = 0; i < g->count; i++) {
        if (g->names) free(g->names[i]);
        if (g->dequant) free(g->dequant[i]);
    }
    free(g->names);
    free(g->dequant);
    free(g->ggml_type);
    free(g->tensors);
    if (g->map) munmap(g->map, (size_t)g->size);
    if (g->fd >= 0) close(g->fd);
    free(g);
}

const legacy_asr_tensor *legacy_asr_gguf_tensors(const legacy_asr_gguf *g, size_t *count) {
    if (count) *count = g ? g->count : 0;
    return g ? g->tensors : NULL;
}
