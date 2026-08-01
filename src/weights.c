/* Weight loading over ingot (third_party/ingot).
 *
 * This replaces the hand-rolled safetensors reader that used to live here and
 * the GGUF reader that used to live in src/gguf.c. The API above it does not
 * change: mynah_asr_st_open() still sniffs the magic and opens either
 * container, so encoder and decoder still never learn where the weights came
 * from.
 *
 * The two containers behave differently, and that is preserved exactly:
 *
 *   safetensors — dtype and pointer are handed over untouched. An int8
 *                 checkpoint's I8 tensors must reach qmat.c as raw bytes.
 *   GGUF        — F32 is zero-copy; every other block type is decoded to f32
 *                 at load into a buffer this handle owns, and reported as F32.
 *                 That was the policy of src/gguf.c, and the engine relies on
 *                 it: after a load, a GGUF tensor is always f32.
 *
 * The eager decode is a policy of THIS engine, not of the library, and keeping
 * it here is the point: a 110M-1B encoder can afford it, and the library stays
 * usable by consumers that cannot. */
#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/wfile.h"

struct mynah_asr_safetensors {
    ingot_wfile *w;
    mynah_asr_tensor *views;   /* parallel to ingot's tensor order */
    float **owned;             /* decoded f32 for the GGUF block types */
    size_t count;
};

static int dtype_from_ingot(ingot_dtype d, mynah_asr_dtype *out) {
    switch (d) {
    case INGOT_DT_F32:  *out = MYNAH_ASR_DT_F32;  return 0;
    case INGOT_DT_F64:  *out = MYNAH_ASR_DT_F64;  return 0;
    case INGOT_DT_BF16: *out = MYNAH_ASR_DT_BF16; return 0;
    case INGOT_DT_F16:  *out = MYNAH_ASR_DT_F16;  return 0;
    case INGOT_DT_I8:   *out = MYNAH_ASR_DT_I8;   return 0;
    case INGOT_DT_U8:   *out = MYNAH_ASR_DT_U8;   return 0;
    case INGOT_DT_I64:  *out = MYNAH_ASR_DT_I64;  return 0;
    default: return -1;
    }
}

static void st_free(mynah_asr_safetensors *st) {
    if (st == NULL) return;
    if (st->owned != NULL)
        for (size_t i = 0; i < st->count; i++) free(st->owned[i]);
    free(st->owned);
    free(st->views);
    ingot_wfile_close(st->w);
    free(st);
}

/* `quiet` keeps the old open_quiet() contract: a missing file is not news. */
static mynah_asr_safetensors *st_open(const char *path, int quiet) {
    char err[256] = {0};
    mynah_asr_safetensors *st = calloc(1, sizeof(*st));
    if (st == NULL) {
        if (!quiet) fprintf(stderr, "weights: out of memory\n");
        return NULL;
    }
    if (ingot_wfile_open(&st->w, path, err, sizeof err) != 0) {
        if (!quiet) fprintf(stderr, "weights: %s\n", err);
        free(st);
        return NULL;
    }

    st->count = ingot_wfile_count(st->w);
    st->views = calloc(st->count ? st->count : 1, sizeof(*st->views));
    st->owned = calloc(st->count ? st->count : 1, sizeof(*st->owned));
    if (st->views == NULL || st->owned == NULL) {
        if (!quiet) fprintf(stderr, "weights: out of memory\n");
        st_free(st);
        return NULL;
    }

    for (size_t i = 0; i < st->count; i++) {
        const ingot_wtensor *t = ingot_wfile_at(st->w, i);
        mynah_asr_tensor *v = &st->views[i];
        v->name = t->name;
        v->n_dims = (int)t->rank;
        v->n_elems = (size_t)t->nelem;
        for (uint32_t d = 0; d < t->rank && d < 8; d++) v->shape[d] = (int64_t)t->shape[d];

        const int is_gguf_block = t->ggml_type >= 0 && t->dtype != INGOT_DT_F32;
        if (!is_gguf_block) {
            if (dtype_from_ingot(t->dtype, &v->dtype) != 0) {
                if (!quiet)
                    fprintf(stderr, "weights: '%s' has a dtype this engine cannot use (%s)\n",
                            t->name, ingot_dtype_name(t->dtype));
                st_free(st);
                return NULL;
            }
            v->data = t->data;                       /* zero-copy from the mmap */
            continue;
        }

        /* A GGUF block type: decode now, exactly as the old loader did. */
        if ((size_t)t->nelem > SIZE_MAX / sizeof(float)) { st_free(st); return NULL; }
        st->owned[i] = malloc((size_t)t->nelem * sizeof(float));
        if (st->owned[i] == NULL || ingot_wfile_to_f32(st->w, t, st->owned[i]) != 0) {
            if (!quiet)
                fprintf(stderr, "weights: cannot decode '%s' (%s)\n",
                        t->name, ingot_type_name(t->ggml_type));
            st_free(st);
            return NULL;
        }
        v->dtype = MYNAH_ASR_DT_F32;                 /* after the load it is always f32 */
        v->data = st->owned[i];
    }
    return st;
}

mynah_asr_safetensors *mynah_asr_st_open(const char *path) { return st_open(path, 0); }
mynah_asr_safetensors *mynah_asr_st_open_quiet(const char *path) { return st_open(path, 1); }

void mynah_asr_st_close(mynah_asr_safetensors *st) { st_free(st); }

size_t mynah_asr_st_count(const mynah_asr_safetensors *st) {
    return st != NULL ? st->count : 0;
}

const mynah_asr_tensor *mynah_asr_st_at(const mynah_asr_safetensors *st, size_t i) {
    return (st != NULL && i < st->count) ? &st->views[i] : NULL;
}

const mynah_asr_tensor *mynah_asr_st_get(const mynah_asr_safetensors *st, const char *name) {
    if (st == NULL || name == NULL) return NULL;
    const ingot_wtensor *t = ingot_wfile_find(st->w, name);   /* O(1), was a strcmp sweep */
    if (t == NULL) return NULL;

    /* ingot hands back a pointer into its own tensor array and the views here
     * are built in the same order, so the index is the offset. The equality
     * check makes that assumption self-verifying: if it ever stops holding,
     * the scan below is still correct, only slower. */
    const ingot_wtensor *base = ingot_wfile_at(st->w, 0);
    if (base != NULL) {
        const size_t index = (size_t)(t - base);
        if (index < st->count && ingot_wfile_at(st->w, index) == t) return &st->views[index];
    }
    for (size_t i = 0; i < st->count; i++)
        if (strcmp(st->views[i].name, name) == 0) return &st->views[i];
    return NULL;
}
