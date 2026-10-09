/* gpu/aed_gpu.h — the offline AED engine on one GPU (Canary), as an offload of
 * the library's offline path (src/mynah_asr.h, mynah_asr_offload).
 *
 * The library keeps the host half of a request -- WAV, segmentation, the mel
 * front end, the canary2 prompt, the generation budget, detokenisation and the
 * stitching of segments -- and calls the two hooks this engine provides:
 *
 *   encode      the FastConformer encoder of a wave of segments. The dw_striding
 *               subsampling runs on the host with the library's own code (it is
 *               ~10 % of the encoder FLOPs; moving it is a later step), the
 *               conformer layers on the device over the PACKED rows [sum T, d]:
 *               every GEMM row-stable (own f32, or own-tc bf16), the full-context
 *               rel-pos attention and the 'same' depthwise conv per segment.
 *   aed_decode  the Transformer decoder, greedy, every segment of the wave
 *               stepped together: per step one row per live segment through the
 *               self-attention (per-segment K/V cache), the cross-attention over
 *               that segment's encoder rows, the FFN, the head and the argmax; the
 *               stopping rule (EOS, the per-segment cap, max_seq) is the host's,
 *               the library's own (src/decoder_aed.c).
 *
 * Batch invariance (PLAN.md durable contract 4) holds by construction: no row
 * of any GEMM depends on the other rows of its call (gpu/cuda/gemm.cu), and every
 * attention, convolution and norm reads only its own segment's rows. Gated by
 * tests/test_cuda_aed.c (batch identity) rather than trusted.
 *
 * One thread at a time (the library's contract for an offloaded model). A device
 * error marks the engine dead; every later call fails visibly. */
#ifndef MYNAH_ASR_GPU_AED_GPU_H
#define MYNAH_ASR_GPU_AED_GPU_H

#include <stddef.h>

#include "../src/mynah_asr.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct asr_aed_gpu asr_aed_gpu;

typedef struct {
    const char *model_dir;   /* the converted pack (f32 weights) */
    int device;
    int max_items;           /* segments per wave: sizes every scratch buffer */
    double seg_sec;          /* the library's per-segment limit (sizes T_max) */
    const char *precision;   /* "f32" (default) | "bf16" (own-tc, sm_80+) */
    const char *gemm;        /* "own" (default with f32) | "own-tc" (with bf16) */
    int decode_on_gpu;       /* 1 = both hooks; 0 = encoder only, the library's
                                CPU decoder decodes (the stage-2 arm) */
} asr_aed_gpu_cfg;

typedef struct {
    const char *device, *precision, *gemm, *decoder;
    int max_items, t_max, s_max;
    int enc_layers, d_model, dec_layers, d_dec, vocab;
    size_t vram_total, vram_used, vram_weights, vram_scratch;
} asr_aed_gpu_facts;

typedef struct {
    unsigned long enc_calls, enc_segments, enc_rows;
    unsigned long dec_calls, dec_segments, dec_steps, dec_tokens, dec_uploads;
    double host_ss_ms, enc_ms, dec_ms;
    unsigned long errors;
} asr_aed_gpu_stats;

/* NULL with the reason in err (the cpu-only build says "not compiled"). */
asr_aed_gpu *asr_aed_gpu_open(const asr_aed_gpu_cfg *cfg, char *err, size_t errcap);
void asr_aed_gpu_close(asr_aed_gpu *e);
/* The offload table bound to this engine, for mynah_asr_set_offload. */
void asr_aed_gpu_offload(asr_aed_gpu *e, mynah_asr_offload *out);
void asr_aed_gpu_get_facts(const asr_aed_gpu *e, asr_aed_gpu_facts *f);
void asr_aed_gpu_get_stats(const asr_aed_gpu *e, asr_aed_gpu_stats *s);
int asr_aed_gpu_dead(const asr_aed_gpu *e);
const char *asr_aed_gpu_error(const asr_aed_gpu *e);
/* One line per operation: the kernel that runs it (ENGINEERING.md §5). */
size_t asr_aed_gpu_dispatch_map(const asr_aed_gpu *e, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif
