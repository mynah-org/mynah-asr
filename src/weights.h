/* safetensors loader: file mmap + name -> tensor index.
 * Tensor names are the HF ones verbatim (decision from docs/prior-art.md). */
#ifndef MYNAH_ASR_WEIGHTS_H
#define MYNAH_ASR_WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

typedef enum { MYNAH_ASR_DT_F32, MYNAH_ASR_DT_F64, MYNAH_ASR_DT_BF16, MYNAH_ASR_DT_F16, MYNAH_ASR_DT_I8, MYNAH_ASR_DT_U8, MYNAH_ASR_DT_I64 } mynah_asr_dtype;

typedef struct {
    const char *name;      /* punta dentro l'header JSON (vive quanto il file) */
    mynah_asr_dtype dtype;
    int n_dims;
    int64_t shape[8];
    const void *data;      /* mmap, read-only */
    size_t n_elems;
} mynah_asr_tensor;

typedef struct mynah_asr_safetensors mynah_asr_safetensors;

mynah_asr_safetensors *mynah_asr_st_open(const char *path);
mynah_asr_safetensors *mynah_asr_st_open_quiet(const char *path); /* niente errore se assente */
void mynah_asr_st_close(mynah_asr_safetensors *st);

/* Exact-name lookup. NULL when absent. */
const mynah_asr_tensor *mynah_asr_st_get(const mynah_asr_safetensors *st, const char *name);
size_t mynah_asr_st_count(const mynah_asr_safetensors *st);
const mynah_asr_tensor *mynah_asr_st_at(const mynah_asr_safetensors *st, size_t i);

#endif
