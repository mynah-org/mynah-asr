/* GGUF loader: an ALTERNATIVE weight container to safetensors (same
 * mynah_asr_tensor struct, so encoder/decoder never know which file the weights
 * came from — a single code path after load). GGUF carries weights only: the
 * config stays in mynah.json (config-driven rule), mel filters and tokenizer are
 * unchanged.
 *
 * Supported ggml types: F32 (zero-copy from the mmap), F16/BF16/Q8_0/Q4_0 and
 * the K-quants Q4_K/Q5_K/Q6_K (dequantized to f32 into buffers at load).
 * GGUF v2/v3 only (v1 has a different, u32 layout). Origin: the keyra parser
 * (../keyra), exercised with a harness of malformed files. */
#ifndef MYNAH_ASR_GGUF_H
#define MYNAH_ASR_GGUF_H

#include <stddef.h>

#include "weights.h"

typedef struct mynah_asr_gguf mynah_asr_gguf;

mynah_asr_gguf *mynah_asr_gguf_open(const char *path);   /* NULL on error (stderr) */
void mynah_asr_gguf_close(mynah_asr_gguf *g);

/* Internal tensor array (lives as long as the handle). */
const mynah_asr_tensor *mynah_asr_gguf_tensors(const mynah_asr_gguf *g, size_t *count);

#endif
