/* A/B gate for the ingot migration. Deleted when this branch is merged.
 *
 * The reader this branch replaces and ingot, linked into one binary — the
 * prefixes do not collide — asked the same questions about the same file.
 *
 * mynah-asr reads BOTH containers, and they behave differently on purpose:
 *
 *   safetensors — dtype and pointer are handed over untouched, so the 212 I8
 *                 tensors of an int8 checkpoint reach qmat.c as raw bytes.
 *                 Compared here byte for byte.
 *   GGUF        — F32 is zero-copy, everything else is dequantized to f32 at
 *                 load and reported as F32. Compared here as floats, which is
 *                 also a cross-check of two independently written decoders:
 *                 src/gguf.c's and ingot's.
 *
 * Fixtures are built in a temp directory, so this runs with no model on disk.
 * Pass paths as arguments to run it over real checkpoints as well.
 *
 * SPDX-License-Identifier: MIT */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ingot/wfile.h"
#include "ingot/write.h"
#include "legacy_weights.h"

static int checks = 0;
static int failures = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        checks++;                                                              \
        if (cond) { printf("  ok:   "); }                                      \
        else { failures++; printf("  FAIL: "); }                               \
        printf(__VA_ARGS__);                                                    \
        printf("\n");                                                          \
    } while (0)

static unsigned char byte_at(size_t i) {
    return (unsigned char)((i * 167u + 13u) ^ (i >> 3));
}

/* The two dtype vocabularies, so a mismatch is a finding and not a surprise. */
static ingot_dtype expected_ingot_dtype(legacy_asr_dtype d) {
    switch (d) {
    case LEGACY_ASR_DT_F32:  return INGOT_DT_F32;
    case LEGACY_ASR_DT_F64:  return INGOT_DT_F64;
    case LEGACY_ASR_DT_BF16: return INGOT_DT_BF16;
    case LEGACY_ASR_DT_F16:  return INGOT_DT_F16;
    case LEGACY_ASR_DT_I8:   return INGOT_DT_I8;
    case LEGACY_ASR_DT_U8:   return INGOT_DT_U8;
    case LEGACY_ASR_DT_I64:  return INGOT_DT_I64;
    }
    return INGOT_DT_UNKNOWN;
}

static size_t dtype_size(legacy_asr_dtype d) {
    switch (d) {
    case LEGACY_ASR_DT_F64: case LEGACY_ASR_DT_I64: return 8;
    case LEGACY_ASR_DT_F32: return 4;
    case LEGACY_ASR_DT_BF16: case LEGACY_ASR_DT_F16: return 2;
    case LEGACY_ASR_DT_I8: case LEGACY_ASR_DT_U8: return 1;
    }
    return 0;
}

/* ── safetensors ─────────────────────────────────────────────────────────── */

typedef struct {
    const char *name;
    const char *dtype;
    size_t rank;
    size_t shape[4];
    size_t elem_size;
} fixture_tensor;

/* Every dtype the old reader accepts, including the I8 an int8 checkpoint is
 * full of and the I64 that BatchNorm's num_batches_tracked uses. */
static const fixture_tensor ST_FIXTURE[] = {
    {"encoder.layers.0.weight", "F32",  2, {3, 5, 0, 0}, 4},
    {"encoder.layers.0.bias",   "F32",  1, {7, 0, 0, 0}, 4},
    {"encoder.q.weight",        "I8",   2, {4, 8, 0, 0}, 1},
    {"encoder.q.scale",         "F32",  1, {4, 0, 0, 0}, 4},
    {"decoder.embed",           "BF16", 2, {2, 6, 0, 0}, 2},
    {"decoder.half",            "F16",  1, {5, 0, 0, 0}, 2},
    {"norm.num_batches_tracked","I64",  1, {1, 0, 0, 0}, 8},
    {"mask.flags",              "U8",   1, {3, 0, 0, 0}, 1},
    {"deep.tensor",             "F32",  4, {2, 2, 2, 2}, 4},
};
static const size_t ST_COUNT = sizeof(ST_FIXTURE) / sizeof(ST_FIXTURE[0]);

static size_t elements(const fixture_tensor *t) {
    size_t n = 1;
    for (size_t d = 0; d < t->rank; d++) n *= t->shape[d];
    return n;
}

static int write_st_fixture(const char *path) {
    char header[4096];
    size_t used = 0, offset = 0;
    used += (size_t)snprintf(header + used, sizeof(header) - used, "{");
    for (size_t i = 0; i < ST_COUNT; i++) {
        const size_t nbytes = elements(&ST_FIXTURE[i]) * ST_FIXTURE[i].elem_size;
        used += (size_t)snprintf(header + used, sizeof(header) - used,
                                 "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                                 i > 0 ? "," : "", ST_FIXTURE[i].name, ST_FIXTURE[i].dtype);
        for (size_t d = 0; d < ST_FIXTURE[i].rank; d++)
            used += (size_t)snprintf(header + used, sizeof(header) - used, "%s%zu",
                                     d > 0 ? "," : "", ST_FIXTURE[i].shape[d]);
        used += (size_t)snprintf(header + used, sizeof(header) - used,
                                 "],\"data_offsets\":[%zu,%zu]}", offset, offset + nbytes);
        offset += nbytes;
    }
    used += (size_t)snprintf(header + used, sizeof(header) - used, "}");
    while ((used + 8u) % 8u != 0u) header[used++] = ' ';

    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;
    unsigned char len_le[8];
    for (size_t i = 0; i < 8; i++) len_le[i] = (unsigned char)(((uint64_t)used >> (8u * i)) & 0xffu);
    if (fwrite(len_le, 1, 8, f) != 8 || fwrite(header, 1, used, f) != used) { fclose(f); return -1; }
    for (size_t i = 0, written = 0; i < ST_COUNT; i++) {
        const size_t nbytes = elements(&ST_FIXTURE[i]) * ST_FIXTURE[i].elem_size;
        for (size_t b = 0; b < nbytes; b++, written++) {
            const unsigned char v = byte_at(written);
            if (fwrite(&v, 1, 1, f) != 1) { fclose(f); return -1; }
        }
    }
    return fclose(f) == 0 ? 0 : -1;
}

static void gate_safetensors(const char *path) {
    printf("safetensors: dtypes and pointers must be handed over untouched\n");
    legacy_asr_safetensors *old = legacy_asr_st_open(path);
    CHECK(old != NULL, "the old reader opens it");

    char err[256] = {0};
    ingot_wfile *new = NULL;
    CHECK(ingot_wfile_open(&new, path, err, sizeof err) == 0 && new != NULL,
          "ingot opens it (%s)", err);
    if (old == NULL || new == NULL) { if (old) legacy_asr_st_close(old); return; }

    CHECK(legacy_asr_st_count(old) == ingot_wfile_count(new),
          "same tensor count (%zu / %zu)", legacy_asr_st_count(old), ingot_wfile_count(new));

    for (size_t i = 0; i < ST_COUNT; i++) {
        const char *name = ST_FIXTURE[i].name;
        const legacy_asr_tensor *a = legacy_asr_st_get(old, name);
        const ingot_wtensor *b = ingot_wfile_find(new, name);
        CHECK(a != NULL && b != NULL, "%s: both find it", name);
        if (a == NULL || b == NULL) continue;

        CHECK(expected_ingot_dtype(a->dtype) == b->dtype, "%s: same dtype", name);
        CHECK((size_t)a->n_dims == (size_t)b->rank, "%s: same rank (%d / %u)",
              name, a->n_dims, b->rank);
        int shape_ok = 1;
        for (int d = 0; d < a->n_dims; d++)
            if ((uint64_t)a->shape[d] != b->shape[d]) shape_ok = 0;
        CHECK(shape_ok, "%s: same shape", name);
        CHECK(a->n_elems == (size_t)b->nelem, "%s: same element count", name);

        const size_t nbytes = a->n_elems * dtype_size(a->dtype);
        CHECK(a->data != NULL && b->data != NULL && memcmp(a->data, b->data, nbytes) == 0,
              "%s: %zu payload bytes identical, unconverted", name, nbytes);
    }

    /* The property qmat.c depends on: an I8 tensor must NOT come back as f32. */
    const ingot_wtensor *q = ingot_wfile_find(new, "encoder.q.weight");
    CHECK(q != NULL && q->dtype == INGOT_DT_I8,
          "an I8 tensor stays I8 — qmat.c reads those bytes directly");

    legacy_asr_st_close(old);
    ingot_wfile_close(new);
}

/* ── GGUF ────────────────────────────────────────────────────────────────── */

/* Built with ingot's writer so the fixture has real quantized blocks. The old
 * reader dequantizes them at load; ingot decodes them on request. Two decoders
 * written from the same spec by different hands agreeing is the point. */
static int write_gguf_fixture(const char *path, size_t rows, size_t cols) {
    float *values = malloc(rows * cols * sizeof(float));
    if (values == NULL) return -1;
    for (size_t i = 0; i < rows * cols; i++)
        values[i] = (float)((int)(byte_at(i)) - 128) / 64.0f;

    ingot_gguf_writer *w = ingot_gguf_writer_new();
    if (w == NULL) { free(values); return -1; }
    char err[256] = {0};
    /* ne is in GGML order: ne[0] is the fastest dimension, so a [rows][cols]
     * row-major matrix is {cols, rows}. */
    const uint64_t ne[2] = {(uint64_t)cols, (uint64_t)rows};
    int rc = ingot_gguf_kv_string(w, "general.architecture", "parity") != 0 ||
             ingot_gguf_add_f32(w, "plain.f32",  INGOT_TYPE_F32,  2, ne, values) != 0 ||
             ingot_gguf_add_f32(w, "block.q8_0", INGOT_TYPE_Q8_0, 2, ne, values) != 0 ||
             ingot_gguf_add_f32(w, "block.q4_k", INGOT_TYPE_Q4_K, 2, ne, values) != 0 ||
             ingot_gguf_add_f32(w, "block.q6_k", INGOT_TYPE_Q6_K, 2, ne, values) != 0 ||
             ingot_gguf_writer_save(w, path, err, sizeof err) != 0;
    if (rc != 0) printf("        writer: %s\n", err);
    ingot_gguf_writer_free(w);
    free(values);
    return rc == 0 ? 0 : -1;
}

static void gate_gguf(const char *path, size_t nelem) {
    printf("GGUF: the eager dequant policy, and two independent decoders\n");
    legacy_asr_safetensors *old = legacy_asr_st_open(path);   /* sniffs the magic */
    CHECK(old != NULL, "the old reader opens the GGUF through the same API");

    char err[256] = {0};
    ingot_wfile *new = NULL;
    CHECK(ingot_wfile_open(&new, path, err, sizeof err) == 0 && new != NULL,
          "ingot opens it (%s)", err);
    if (old == NULL || new == NULL) { if (old) legacy_asr_st_close(old); return; }

    CHECK(ingot_wfile_container(new) == INGOT_CONTAINER_GGUF, "ingot says it is a GGUF");

    static const char *names[] = {"plain.f32", "block.q8_0", "block.q4_k", "block.q6_k"};
    float *decoded = malloc(nelem * sizeof(float));
    if (decoded == NULL) { CHECK(0, "out of memory"); goto out; }

    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        const legacy_asr_tensor *a = legacy_asr_st_get(old, names[i]);
        const ingot_wtensor *b = ingot_wfile_find(new, names[i]);
        CHECK(a != NULL && b != NULL, "%s: both find it", names[i]);
        if (a == NULL || b == NULL) continue;

        /* The old policy: whatever the block type, it is f32 after the load. */
        CHECK(a->dtype == LEGACY_ASR_DT_F32, "%s: the old reader reports F32", names[i]);
        CHECK(a->n_elems == (size_t)b->nelem, "%s: same element count", names[i]);

        CHECK(ingot_wfile_to_f32(new, b, decoded) == 0, "%s: ingot decodes it", names[i]);
        const float *want = a->data;
        size_t differing = 0;
        double worst = 0.0;
        for (size_t k = 0; k < a->n_elems; k++) {
            if (decoded[k] != want[k]) {
                differing++;
                const double delta = (double)decoded[k] - (double)want[k];
                if (delta > worst || -delta > worst) worst = delta < 0 ? -delta : delta;
            }
        }
        CHECK(differing == 0, "%s: %zu values match bit for bit%s", names[i], a->n_elems,
              differing == 0 ? "" : " — worst delta above");
        if (differing != 0)
            printf("        %zu of %zu differ, worst %.9g\n", differing, a->n_elems, worst);
    }
    free(decoded);
out:
    legacy_asr_st_close(old);
    ingot_wfile_close(new);
}

/* ── a real checkpoint ───────────────────────────────────────────────────── */

static void gate_real(const char *path) {
    printf("real checkpoint: %s\n", path);
    legacy_asr_safetensors *old = legacy_asr_st_open(path);
    char err[256] = {0};
    ingot_wfile *new = NULL;
    const int rc = ingot_wfile_open(&new, path, err, sizeof err);
    CHECK(old != NULL, "the old reader opens it");
    CHECK(rc == 0 && new != NULL, "ingot opens it (%s)", err);
    if (old == NULL || new == NULL) { if (old) legacy_asr_st_close(old); if (new) ingot_wfile_close(new); return; }

    const size_t total = legacy_asr_st_count(old);
    CHECK(total == ingot_wfile_count(new), "same tensor count (%zu / %zu)",
          total, ingot_wfile_count(new));

    size_t missing = 0, dtype_bad = 0, shape_bad = 0, bytes_bad = 0, compared = 0;
    unsigned long long compared_bytes = 0;
    for (size_t i = 0; i < total; i++) {
        const legacy_asr_tensor *a = legacy_asr_st_at(old, i);
        if (a == NULL) continue;
        const ingot_wtensor *b = ingot_wfile_find(new, a->name);
        if (b == NULL) { if (!missing) printf("        first missing: %s\n", a->name); missing++; continue; }
        /* A GGUF block type: the old reader already decoded it to f32 at load,
         * so the comparison is float against float and doubles as a check of
         * two independently written decoders. Everything else is raw bytes. */
        const int is_block = b->ggml_type >= 0 && b->dtype != INGOT_DT_F32;
        if (!is_block && expected_ingot_dtype(a->dtype) != b->dtype) {
            if (!dtype_bad) printf("        first dtype mismatch: %s\n", a->name);
            dtype_bad++; continue;
        }
        if (is_block && a->dtype != LEGACY_ASR_DT_F32) {
            if (!dtype_bad) printf("        block type not decoded to f32: %s\n", a->name);
            dtype_bad++; continue;
        }
        int ok = (size_t)a->n_dims == (size_t)b->rank && a->n_elems == (size_t)b->nelem;
        for (int d = 0; ok && d < a->n_dims; d++)
            if ((uint64_t)a->shape[d] != b->shape[d]) ok = 0;
        if (!ok) { if (!shape_bad) printf("        first shape mismatch: %s\n", a->name); shape_bad++; continue; }

        int same;
        if (is_block) {
            float *decoded = malloc(a->n_elems * sizeof(float));
            if (decoded == NULL) { bytes_bad++; continue; }
            same = ingot_wfile_to_f32(new, b, decoded) == 0 &&
                   memcmp(decoded, a->data, a->n_elems * sizeof(float)) == 0;
            compared_bytes += (unsigned long long)a->n_elems * sizeof(float);
            free(decoded);
        } else {
            const size_t nbytes = a->n_elems * dtype_size(a->dtype);
            const unsigned char *pa = a->data, *pb = b->data;
            if (nbytes <= 1024u * 1024u) { same = memcmp(pa, pb, nbytes) == 0; compared_bytes += nbytes; }
            else {
                const size_t edge = 4096;
                same = memcmp(pa, pb, edge) == 0 &&
                       memcmp(pa + nbytes - edge, pb + nbytes - edge, edge) == 0;
                compared_bytes += 2u * edge;
            }
        }
        if (!same) { if (!bytes_bad) printf("        first payload mismatch: %s\n", a->name); bytes_bad++; continue; }
        compared++;
    }
    CHECK(missing == 0, "%zu tensors: every name found (%zu missing)", total, missing);
    CHECK(dtype_bad == 0, "dtypes agree on all %zu (%zu differ)", total, dtype_bad);
    CHECK(shape_bad == 0, "rank, shape and element count agree (%zu differ)", shape_bad);
    CHECK(bytes_bad == 0, "payloads agree (%zu differ, %llu bytes compared)", bytes_bad, compared_bytes);
    CHECK(compared == total, "every tensor was compared (%zu / %zu)", compared, total);

    legacy_asr_st_close(old);
    ingot_wfile_close(new);
}

int main(int argc, char **argv) {
    for (int a = 1; a < argc; a++) gate_real(argv[a]);

    char dir[] = "/tmp/mynah_asr_parity_XXXXXX";
    if (mkdtemp(dir) == NULL) { fprintf(stderr, "no temp directory\n"); return 1; }
    char st_path[512], gguf_path[512];
    snprintf(st_path, sizeof st_path, "%s/fixture.safetensors", dir);
    snprintf(gguf_path, sizeof gguf_path, "%s/fixture.gguf", dir);

    if (write_st_fixture(st_path) != 0) { CHECK(0, "cannot write the safetensors fixture"); }
    else gate_safetensors(st_path);

    const size_t rows = 4, cols = 256;
    if (write_gguf_fixture(gguf_path, rows, cols) != 0) { CHECK(0, "cannot write the GGUF fixture"); }
    else gate_gguf(gguf_path, rows * cols);

    remove(st_path);
    remove(gguf_path);
    rmdir(dir);

    printf("\n%d checks, %d failures\n", checks, failures);
    if (failures == 0) printf("PARITY GATE GREEN\n");
    return failures != 0;
}
