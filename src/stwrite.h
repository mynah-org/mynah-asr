/* Minimal safetensors writer (JSON header + contiguous data).
 * Used by `mynah-asr quantize` for the pre-quantized checkpoints. */
#ifndef MYNAH_ASR_STWRITE_H
#define MYNAH_ASR_STWRITE_H

#include <stddef.h>
#include <stdint.h>

typedef struct mynah_asr_stw mynah_asr_stw;

mynah_asr_stw *mynah_asr_stw_new(void);
/* data must stay valid until mynah_asr_stw_write. dtype: "F32", "I8", "U8". */
int mynah_asr_stw_add(mynah_asr_stw *w, const char *name, const char *dtype,
                  const int64_t *shape, int n_dims, const void *data, size_t bytes);
int mynah_asr_stw_write(mynah_asr_stw *w, const char *path);
void mynah_asr_stw_free(mynah_asr_stw *w);

#endif
