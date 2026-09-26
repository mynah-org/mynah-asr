/* gpu/pack.h — a converted Nemotron pack opened for the GPU engine: the f32
 * tensors (mmapped), the encoder/decoder descriptors the library builds from
 * them, the tokenizer, the mel config and the streaming presets.
 *
 * This is the library's own loading code used READ-ONLY (src/encoder.h,
 * src/decoder.h, src/weights.h, src/tokenizer.h, src/features.h), not a copy
 * of it: the descriptors are the same structs `mynah_asr_load` fills, so the
 * GPU uploads exactly the tensors the CPU path reads. What is NOT reused is
 * `mynah_asr_model` itself, which is opaque and carries the CPU stream state. */
#ifndef MYNAH_ASR_GPU_PACK_H
#define MYNAH_ASR_GPU_PACK_H

/* relative paths on purpose: nvcc must not get -I../src, because src/features.h
 * would shadow glibc's <features.h> for every system header (seen in CI) */
#include "../vendor/cJSON.h"
#include "../src/decoder.h"
#include "../src/encoder.h"
#include "../src/features.h"
#include "../src/tokenizer.h"
#include "../src/weights.h"

typedef struct {
    cJSON *cfg;
    mynah_asr_safetensors *weights, *mel_filters;
    mynah_asr_encoder enc;      /* f32, quantize = 0 */
    mynah_asr_decoder dec;      /* f32 head */
    mynah_asr_tokenizer tok;
    mynah_asr_feat_cfg feat;
    char name[128];
    int left_ctx, default_right;
    int lookaheads[8], n_lookaheads, qmax;
    int default_prompt;
    double frame_sec;
} asr_pack;

/* Opens `dir`. Refuses (with the reason in err) anything the streaming step
 * cannot serve: no presets, biases, batch_norm, xscaling, non-causal conv,
 * per-feature normalisation, a subsampling factor other than 8, a TDT head, an
 * int8-only pack (the GPU uploads f32 in phase 1). Builds the rel-pos table. */
int asr_pack_open(asr_pack *p, const char *dir, char *err, size_t errcap);
void asr_pack_close(asr_pack *p);

/* Language tag -> prompt id, -1 when the pack does not serve it; NULL/"auto"
 * -> the pack's default prompt. */
int asr_pack_lang_id(const asr_pack *p, const char *lang);
int asr_pack_lookahead_ok(const asr_pack *p, int lookahead);

/* Mel frames one chunk needs at lookahead r: 1 + 8r the first time, 8(r+1)
 * after (src/encoder.c, mynah_asr_enc_stream_need). */
int asr_pack_chunk_mel(const asr_pack *p, int lookahead, int first);

#endif
