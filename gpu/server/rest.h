/* gpu/server/rest.h — the OFFLINE serving mode of mynah-asr-server-cuda, for a
 * pack that cannot stream (asr_pack_mode == ASR_PACK_OFFLINE_AED, Canary).
 *
 * Contract (the CPU server's for the same pack, server/main.c):
 *   POST /v1/audio/transcriptions  multipart (file, language, response_format
 *                                  json|text|verbose_json, model) or a raw WAV body;
 *                                  the FINAL transcript of the file, nothing before it
 *   POST /v1/audio/translations    the same + target_language (default "en")
 *   GET  /v1/audio/stream          400 model_not_streaming BEFORE the upgrade: the
 *                                  pack has no streaming API and the server never
 *                                  fakes incremental partials
 *
 * Requests are admitted into a bounded queue (--cap; full = 503
 * server_at_capacity with Retry-After, rule 5: refuse, never stretch). One
 * batcher thread gathers what is queued within --cohort-ms (or until --batch
 * requests are waiting) and transcribes them in one asr_offline call: the GPU
 * steps their encoders and their decoders together. A transcript does not
 * depend on what it was batched with (tests/test_cuda_aed.c gates it). */
#ifndef MYNAH_ASR_GPU_SERVER_REST_H
#define MYNAH_ASR_GPU_SERVER_REST_H

#include <stddef.h>

#include "asr_offline.h"
#include "cJSON.h"

typedef struct {
    asr_offline *eng;
    int cap;                 /* requests admitted (queued + in flight) */
    int batch;               /* requests per batch (<= the engine's max_items) */
    int window_ms;           /* gather window after the first queued request */
    double max_audio_seconds;/* 0 = no limit */
} rest_cfg;

int rest_start(const rest_cfg *cfg);
void rest_stop(void);       /* wakes and joins the batcher; queued requests fail */
/* POST transcriptions (translate 0) / translations (1). `head` holds `got`
 * bytes read so far, the request head ending at `head_len`. Always consumes
 * (answers and closes) fd. */
void rest_handle(int fd, const char *head, size_t head_len, size_t got, const char *query,
                 int translate);
void rest_health_json(cJSON *j);
size_t rest_metrics_text(char *b, size_t cap);
size_t rest_dump(char *b, size_t cap, unsigned long seq);

#endif
