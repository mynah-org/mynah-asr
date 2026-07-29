/* Loader GGUF: container ALTERNATIVO a safetensors per i pesi (stessa struct
 * mynah_asr_tensor, quindi encoder/decoder non sanno da che file arrivano — un
 * solo code path dopo il load). Il GGUF trasporta solo pesi: la config resta
 * mynah.json (regola config-driven), mel filters e tokenizer restano invariati.
 *
 * Tipi ggml supportati: F32 (zero-copy dal mmap), F16/BF16/Q8_0/Q4_0
 * (dequantizzati a f32 in buffer al load). Q4_K/Q6_K: rimandati al milestone
 * interop parakeet.cpp. Solo GGUF v2/v3 (la v1 ha un layout diverso, u32).
 * Origine: parser di keyra (../keyra), verificato con harness su file malformati. */
#ifndef MYNAH_ASR_GGUF_H
#define MYNAH_ASR_GGUF_H

#include <stddef.h>

#include "weights.h"

typedef struct mynah_asr_gguf mynah_asr_gguf;

mynah_asr_gguf *mynah_asr_gguf_open(const char *path);   /* NULL su errore (stderr) */
void mynah_asr_gguf_close(mynah_asr_gguf *g);

/* Array interno dei tensori (vive quanto il handle). */
const mynah_asr_tensor *mynah_asr_gguf_tensors(const mynah_asr_gguf *g, size_t *count);

#endif
