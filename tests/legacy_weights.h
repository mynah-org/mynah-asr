/* safetensors loader: file mmap + name -> tensor index.
 * Tensor names are the HF ones verbatim (decision from docs/prior-art.md). */
#ifndef LEGACY_ASR_WEIGHTS_H
#define LEGACY_ASR_WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

typedef enum { LEGACY_ASR_DT_F32, LEGACY_ASR_DT_F64, LEGACY_ASR_DT_BF16, LEGACY_ASR_DT_F16, LEGACY_ASR_DT_I8, LEGACY_ASR_DT_U8, LEGACY_ASR_DT_I64 } legacy_asr_dtype;

typedef struct {
    const char *name;      /* points into the JSON header (lives as long as the file) */
    legacy_asr_dtype dtype;
    int n_dims;
    int64_t shape[8];
    const void *data;      /* mmap, read-only */
    size_t n_elems;
} legacy_asr_tensor;

typedef struct legacy_asr_safetensors legacy_asr_safetensors;

legacy_asr_safetensors *legacy_asr_st_open(const char *path);
legacy_asr_safetensors *legacy_asr_st_open_quiet(const char *path); /* no error when absent */
void legacy_asr_st_close(legacy_asr_safetensors *st);

/* Exact-name lookup. NULL when absent. */
const legacy_asr_tensor *legacy_asr_st_get(const legacy_asr_safetensors *st, const char *name);
size_t legacy_asr_st_count(const legacy_asr_safetensors *st);
const legacy_asr_tensor *legacy_asr_st_at(const legacy_asr_safetensors *st, size_t i);

#endif
