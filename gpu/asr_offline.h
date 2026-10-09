/* gpu/asr_offline.h — the OFFLINE engine seam of the GPU server: a sibling of
 * gpu/asr_engine.h for packs that cannot stream (an attention encoder-decoder,
 * Canary). Why a sibling and not the streaming interface: asr_engine is a
 * per-slot, per-chunk step whose every call returns the text a chunk appended;
 * an AED decoder keeps no state between calls and only produces text once it
 * has the whole segment, so the honest unit of work is a REQUEST (a file), and
 * the honest answer is its final transcript. Forcing it behind slot_feed/step
 * would mean either fake partials or a step that does nothing until the end.
 *
 * One implementation (gpu/offline_engine.c) with two compute backends:
 *
 *   cpu    the library alone (mynah_asr_transcribe_batch): the REFERENCE, the
 *          same call the CPU server's scheduler makes. Chosen only by
 *          `--engine cpu`; never a fallback.
 *   cuda   the same library call with the GPU offload installed
 *          (gpu/aed_gpu.h): the host half is the library's, byte for byte;
 *          the encoder and the AED decode run on the device.
 *
 * Threading: one thread at a time (the server's batcher thread). */
#ifndef MYNAH_ASR_GPU_ASR_OFFLINE_H
#define MYNAH_ASR_GPU_ASR_OFFLINE_H

#include <stddef.h>

#include "../src/mynah_asr.h"

typedef struct asr_offline asr_offline;

typedef struct {
    const char *model_dir;
    const char *engine;      /* "cuda" | "cpu" */
    int device;
    int max_items;           /* requests per batch (and segments per GPU wave) */
    const char *precision;   /* cuda: "f32" | "bf16" */
    const char *gemm;        /* cuda: "own" | "own-tc" */
    const char *decoder;     /* cuda: "gpu" (default) | "host" (the library's CPU
                                decoder after the GPU encoder: the stage-2 arm) */
    int threads;             /* library pool threads (host half); 0 = default */
} asr_offline_cfg;

typedef struct {
    const char *name;        /* "cuda" | "cpu" */
    const char *device, *precision, *gemm, *decoder;
    const char *model_name;
    int sample_rate, can_translate, max_items;
    double seg_sec;
    size_t vram_total, vram_used, vram_weights, vram_scratch;
} asr_offline_facts;

typedef struct {
    unsigned long batches, items, failed_batches, retried_items;
    double wall_ms;          /* inside asr_offline_transcribe, total */
    unsigned long enc_calls, enc_rows, dec_calls, dec_steps, dec_tokens;
    double host_ss_ms, enc_ms, dec_ms;
    unsigned long device_errors;
} asr_offline_stats;

asr_offline *asr_offline_open(const asr_offline_cfg *cfg, char *err, size_t errcap);
void asr_offline_close(asr_offline *o);
void asr_offline_get_facts(const asr_offline *o, asr_offline_facts *f);
void asr_offline_get_stats(const asr_offline *o, asr_offline_stats *s);
/* 1 when `lang` ("en", "auto", "fr>en", ...) would be accepted; a config lookup */
int asr_offline_lang_ok(const asr_offline *o, const char *lang);

/* n requests (16 kHz mono f32), each with its language tag ("src" or
 * "src>tgt"). texts[i] receives the final transcript (malloc'd) or NULL when
 * THAT request failed; langs_out (optional) the language reported. words/n_words
 * (optional, per item; NULL = text only) ask for word timestamps, which on an
 * AED pack take the library's single path, as on the CPU server.
 *
 * The library's batch call is all-or-nothing; when it fails, every item is
 * retried ALONE, so one bad request never fails the others -- and since a
 * transcript does not depend on its batch, the retried texts are the ones the
 * batch would have produced. Returns the number of failed items, or -1 when
 * the device is dead. */
int asr_offline_transcribe(asr_offline *o, int n, const float *const *pcm, const size_t *ns,
                           const char *const *langs, char **texts, char (*langs_out)[16],
                           mynah_asr_word **words, int *n_words);
int asr_offline_dead(const asr_offline *o);
const char *asr_offline_error(const asr_offline *o);
size_t asr_offline_dispatch_map(const asr_offline *o, char *buf, size_t cap);

#endif
