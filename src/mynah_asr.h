/* Mynah — a lightweight native C runtime for streaming and offline ASR.
 *
 * Public API of libmynah_asr. Current phase: offline-chunked transcription (M1.2).
 * The streaming API (mynah_asr_stream_*) lands with M1.3.
 */
#ifndef MYNAH_ASR_H
#define MYNAH_ASR_H

#include <stddef.h>
#include <stdbool.h>

#define MYNAH_ASR_VERSION_MAJOR 0
#define MYNAH_ASR_VERSION_MINOR 9
#define MYNAH_ASR_VERSION_PATCH 1
#define MYNAH_ASR_VERSION "0.9.1"

#ifdef __cplusplus
extern "C" {
#endif

/* Incremental transcription result (streaming and offline).
 * text is UTF-8 and only valid for the duration of the callback. */
typedef struct {
    const char *text;      /* text of the segment/partial                   */
    double      t0, t1;    /* time window in seconds (when available) */
    bool        is_final;  /* false = partial (may still change), true = commit */
    const char *lang;      /* detected language tag, NULL when unavailable   */
    bool        is_eou;    /* the VAD saw the utterance END at t1 (text is "") */
} mynah_asr_result;

typedef void (*mynah_asr_result_cb)(const mynah_asr_result *res, void *userdata);

const char *mynah_asr_version(void);

/* ------------------------------------------------------------------- model */
typedef struct mynah_asr_model mynah_asr_model;

/* Load a converted model (a directory with mynah.json + model.safetensors
 * + tokens.json + mel_filters.safetensors). NULL on error. */
mynah_asr_model *mynah_asr_load(const char *model_dir);

/* Like mynah_asr_load but quantized: MYNAH_ASR_QUANT_INT8 builds per-row INT8 for
 * the large linears at load time (~2.4x less RAM, near-identical quality). */
enum { MYNAH_ASR_QUANT_F32 = 0, MYNAH_ASR_QUANT_INT8 = 1, MYNAH_ASR_QUANT_INT4 = 2 };
mynah_asr_model *mynah_asr_load_quant(const char *model_dir, int quant);
void mynah_asr_free(mynah_asr_model *m);

/* Resolve a language tag ("it-IT", "auto", ...) into a prompt id. -1 if unknown. */
int mynah_asr_lang_id(const mynah_asr_model *m, const char *lang);

/* ------------------------------------------------------- language identification
 * Only some models can be ASKED which language they heard: Nemotron emits the
 * locale as a token ("auto" prompt), Parakeet detects it internally but never
 * says so, and the AED models (Canary) take the source language as an INPUT and
 * predict nothing — feed one the wrong source and the text comes out wrong, so
 * "auto" on those means "the model's default" (en), not detection.
 *
 * Hence the pairing: run a model that CAN detect over a short prefix, hand the
 * tag to the one that cannot. */

/* 1 when this model reports the language it detected (usable as a detector). */
int mynah_asr_can_detect_lang(const mynah_asr_model *m);

/* Language of `samples`, from a SHORT prefix (a few seconds starting at the first
 * speech, retried once on a longer window when nothing comes out) — the text is
 * decoded and thrown away, only the tag is kept. Writes the locale ("it-IT") into
 * out (>= 16 bytes) and returns 0; -1 when the model has no detection or nothing
 * was detected (out is then ""), which the caller is expected to handle by
 * falling back rather than failing. */
int mynah_asr_detect_lang(mynah_asr_model *m, const float *samples, size_t n_samples,
                          char *out);

/* Adapt a tag to what THIS model accepts and write it into out (>= 16 bytes):
 * the detector's locale ("it-IT") becomes what the target takes ("it" on Canary,
 * whose set of languages is smaller). Returns 0, or -1 when the model does not
 * support that language at all — the case a detect->translate pipeline must
 * report instead of silently transcribing as something else. */
int mynah_asr_map_lang(const mynah_asr_model *m, const char *tag, char *out);

/* Select the decoder for subsequent transcriptions: "default" (the model's
 * RNNT/TDT) or "ctc" (the auxiliary head of hybrid models, faster, offline only).
 * -1 when the model does not support the requested decoder. */
int mynah_asr_set_decoder(mynah_asr_model *m, const char *name);

/* AED models only (Canary): OUTPUT language of subsequent transcriptions
 * ("en", "de", ...). Different from the source = speech translation. NULL or "" =
 * same as the source (ASR). -1 when unsupported by the model.
 * PER-CALL alternative (thread-safe, for server/batch): lang "src>tgt" in
 * mynah_asr_transcribe* (e.g. "en>de"); it wins over mynah_asr_set_target_lang. */
int mynah_asr_set_target_lang(mynah_asr_model *m, const char *lang);

/* 1 when the model supports speech translation (AED engine). */
int mynah_asr_can_translate(const mynah_asr_model *m);

/* Per-segment duration limit for offline transcription: longer audio is split at
 * the energy minimum (silence) near the boundary and transcribed as independent
 * segments (text and timestamps concatenated). MODEL-AWARE default: 30 s for
 * full-attention/AED models (Parakeet, Canary — trained on short utterances: long
 * segments degrade quality), 300 s for windowed-attention models (Nemotron).
 * sec >= 5; 0 = restore the default. */
void mynah_asr_set_segment_limit(mynah_asr_model *m, double sec);

/* The limit currently in force, in seconds (the resolved default if never set). */
double mynah_asr_segment_limit(const mynah_asr_model *m);

/* Attaches a Silero VAD (the directory produced by tools/convert_silero.py) to
 * the offline path: segments are then cut on speech boundaries and the silence
 * between them is never fed to the encoder. OPT-IN — it changes both the timing
 * and, potentially, the text, so it is never enabled behind the caller's back.
 *
 * vad_dir NULL detaches it. Returns 0, or -1 when the directory cannot be loaded
 * (the model keeps working without a VAD). Call before transcribing, not
 * concurrently: like set_target_lang, it mutates the model.
 *
 * Consequence worth knowing: audio in which the VAD finds no speech at all is not
 * decoded at all and comes back as an EMPTY string. Usually what you want, but it
 * means a file the VAD misjudges transcribes to nothing rather than to something
 * wrong. (On the models tested, silence already decoded to "" either way.) */
int mynah_asr_enable_vad(mynah_asr_model *m, const char *vad_dir);

/* 1 when a VAD is attached. */
int mynah_asr_has_vad(const mynah_asr_model *m);

/* Lookaheads (right context) valid for the model, e.g. {3,0,6,13}. */
int mynah_asr_lookaheads(const mynah_asr_model *m, int out[8]);

/* The sample rate this model's features were built for, from the pack.  Every
 * NeMo speech model published so far is 16 kHz; a caller that hardcodes it is
 * right today and wrong the first time that stops being true, so the server
 * asks instead. */
int mynah_asr_sample_rate(const mynah_asr_model *m);

/* Offline transcription: float32 [-1,1] 16 kHz mono samples.
 * lang: tag ("auto" for detection). lookahead: -1 = the model's default.
 * Returns UTF-8 text (malloc'd, freed by the caller); when lang_out != NULL
 * (>= 16 bytes) the detected language is written there. NULL on error. */
char *mynah_asr_transcribe(mynah_asr_model *m, const float *samples, size_t n_samples,
                       const char *lang, int lookahead, char *lang_out);

/* A word with its time window (from the greedy emission encoder frames:
 * resolution = 1 encoder frame, typically 80 ms). */
typedef struct {
    char  *word;           /* UTF-8, markers stripped (malloc'd) */
    double t0, t1;         /* seconds from the start of audio  */
} mynah_asr_word;

/* Like mynah_asr_transcribe, and additionally writes into *words a malloc'd array
 * of *n_words timestamped words (free with mynah_asr_words_free).
 * words == NULL = text only. */
char *mynah_asr_transcribe_ts(mynah_asr_model *m, const float *samples, size_t n_samples,
                          const char *lang, int lookahead, char *lang_out,
                          mynah_asr_word **words, int *n_words);

void mynah_asr_words_free(mynah_asr_word *words, int n_words);

/* Weight-stationary BATCH transcription: N requests processed together — the
 * weights (2.5 GB) are read once per layer instead of N times. Variable lengths,
 * no padding. texts[i] receives the text (malloc'd, freed by the caller);
 * langs_out[i] (>= 16 bytes each) is optional.
 *
 * Items longer than the segment limit are split exactly as mynah_asr_transcribe
 * splits them, so both entry points return the same text (they did not before:
 * the batch path encoded each item whole and long audio came out worse).
 *
 * An item with 0 samples is NOT a failure: it transcribes to "" and leaves the
 * rest of the batch alone (verified in every position by tests/test_batch.c).
 *
 * Two configurations cannot be batched without changing the answer, and are run one
 * item at a time instead of quietly returning something else: a hybrid model
 * switched to "ctc" (that engine needs the raw encoder output, the batched forward
 * produces the projected one) and `words` on an AED model (asking for timestamps
 * changes the prompt on the flash models, and canary-1b-v2 needs its aligner). The
 * text is then identical to the single path by construction; only the
 * weight-stationary speedup is lost.
 *
 * Returns 0 on success, -1 on failure — and failure is ALL-OR-NOTHING: every
 * texts[]/words[] slot comes back NULL/0, not just the item that failed. On
 * success every texts[i] is non-NULL (possibly ""). Every output slot is
 * initialized before anything that can fail, so a -1 never leaves a caller's
 * array holding uninitialized pointers. */
/* Like mynah_asr_transcribe_batch, plus words[b]/n_words[b] per item (arrays of
 * `batch` slots provided by the caller; each words[b] must be freed with
 * mynah_asr_words_free). words == NULL = text only. */
int mynah_asr_transcribe_batch_ts(mynah_asr_model *m, const float *const *samples,
                              const size_t *n_samples, int batch, const char *const *langs,
                              int lookahead, char **texts, char (*langs_out)[16],
                              mynah_asr_word **words, int *n_words);
int mynah_asr_transcribe_batch(mynah_asr_model *m, const float *const *samples,
                           const size_t *n_samples, int batch, const char *const *langs,
                           int lookahead, char **texts, char (*langs_out)[16]);

/* --------------------------------------------------------------- streaming
 * Cache-aware, latency = (lookahead+1) * 80 ms. Text emitted through the callback
 * is ALWAYS final (monotonic greedy, never retracted): is_final = true. */
typedef struct mynah_asr_stream mynah_asr_stream;

/* Inherits endpointing from the model: if mynah_asr_enable_vad was called, the
 * stream opens its OWN VAD instance (the VAD carries LSTM state, so it cannot be
 * shared between streams or with the offline path) and reports each end of
 * utterance with an extra callback carrying text = "", is_eou = true and t1 = the
 * time speech stopped.
 *
 * Endpointing does NOT touch decoding: the transcribed text is byte-identical with
 * and without a VAD, it only tells you where utterances end (gated in
 * tests/test_e2e.sh). The report arrives ~160 ms of audio after speech actually
 * stopped — min_silence_ms plus frame quantization — plus however much audio you
 * feed per call, so smaller feeds mean lower endpoint latency. Measured on
 * nemotron: 160 ms at 32 ms chunks, 176-212 ms at 100 ms, 312-376 ms at 250 ms.
 * Note the text for the audio just before an endpoint may still arrive AFTER it:
 * the encoder works in chunks and the VAD is ahead of the decoder. */
mynah_asr_stream *mynah_asr_stream_open(mynah_asr_model *m, const char *lang, int lookahead);

/* Why the cache-aware streaming path cannot serve this model, or NULL when it
 * can.  Two separate questions, and callers that refuse before a WebSocket
 * upgrade need both: mynah_asr_lookaheads() == 0 says the pack was never
 * trained cache-aware, while this says the pack IS cache-aware but uses
 * something the incremental encoder does not implement (linear biases, a folded
 * batch_norm, xscaling, symmetric conv padding, per-feature normalisation, a
 * subsampling factor other than 8).  The offline path serves all of those; the
 * streaming step, written for Nemotron, does not, and would emit plausible
 * WRONG text instead of failing.  See .work/multi-model-streaming.md.
 *
 * Reads the loaded weights, so it answers for the model in memory rather than
 * for what the config claims.  The string is static, English, and safe to put
 * in a refusal body. */
const char *mynah_asr_stream_unsupported(const mynah_asr_model *m);

/* MYNAH_ASR_STACK_SOLO: whether a ready set of one takes the stacked encoder
 * path (default 1). Exposed so the flag registry and the tests can read the
 * same answer the step does. */
int mynah_asr_stack_solo(void);

/* Feed float32 16 kHz mono samples; the callback receives the text deltas. */
int mynah_asr_stream_feed(mynah_asr_stream *s, const float *samples, size_t n,
                      mynah_asr_result_cb cb, void *userdata);

/* End of stream: processes the tail (last chunk padded) and emits the rest. */
int mynah_asr_stream_finish(mynah_asr_stream *s, mynah_asr_result_cb cb, void *userdata);

/* Language detected so far ("" when not emitted yet). */
const char *mynah_asr_stream_lang(const mynah_asr_stream *s);

/* A new utterance on the SAME stream object: every cache, the decoder state,
 * the VAD state and the emitted-text bookkeeping go back to what open() gave,
 * without freeing or allocating anything (~13 MB per stream on nemotron, and a
 * server pools its slots). lang = NULL keeps the current language prompt; the
 * lookahead cannot change (it sizes the scratch). Returns 0, or -1 when lang is
 * not supported (the stream is then unchanged). Gate: reset+feed is byte-identical
 * to close+open+feed (tests/test_streaming.c). */
int mynah_asr_stream_reset(mynah_asr_stream *s, const char *lang);

/* Samples the stream still needs before its next chunk is complete and the
 * callback can fire: a scheduler that feeds exactly this much per step gives
 * every stream one encoder chunk per step, no more. 0 never happens after a
 * feed (a complete chunk is consumed by the feed that completes it). */
size_t mynah_asr_stream_need_samples(const mynah_asr_stream *s);

/* Audio seconds fed so far. */
double mynah_asr_stream_audio_seconds(const mynah_asr_stream *s);

/* ------------------------------------------------------- batched step (S1-4)
 * Feed B streams at once and run ONE encoder pass for all of them: the chunks
 * that completed are stacked as a single [Sum q_i, d] activation, so each
 * conformer layer's linears (FFN1, q/k/v/o, the pointwise convolutions, FFN2)
 * read the weights once instead of B times. Attention (per-stream K/V cache and
 * relative positions), the conv cache and the subsampling stay per stream, and
 * so do the greedy decode and the callbacks.
 *
 * This exists because streaming ASR chunks arrive on the real-time grid: at the
 * default preset every live stream has a chunk ready every 320 ms, so batching
 * inside one worker is structural, not opportunistic.
 *
 * samples[i] / n_samples[i] is what to feed stream i — typically exactly
 * mynah_asr_stream_need_samples(streams[i]), which gives every stream exactly one
 * encoder chunk per call. A stream whose chunk does not complete is simply fed
 * (its mel accumulates) and produces no callback; more than one complete chunk
 * is handled by repeating the pass. userdata may be NULL, else it is an array of
 * B pointers, userdata[i] going to stream i's callback.
 *
 * IDENTITY (ENGINEERING.md §9): every stream's text is byte-identical to the same
 * clip fed through mynah_asr_stream_feed alone, whatever B is and whoever it was
 * batched with. Streams may mix languages and clips freely. Streams with
 * DIFFERENT lookahead presets are grouped automatically and run as one pass per
 * group. All streams must belong to the same model.
 *
 * B == 1 takes exactly the single-stream path. The f32 weights path may also
 * degrade to per-stream steps (a vendor sgemm is not row-stable in M on every
 * BLAS — see mynah_asr_stream_batch_rows_stacked to check what a run actually
 * did).
 *
 * THREADING: one thread at a time per model (the serving-v2 scheduler owns it).
 * Returns 0, -1 on error.
 *
 * ALLOCATION: the batch scratch is carved on the first call and grown if a later
 * call brings a larger B. Call mynah_asr_stream_batch_reserve(m, max_b) once at
 * start-up and the step allocates nothing after that. */
int mynah_asr_stream_step_batch(mynah_asr_stream *const *streams, int B,
                            const float *const *samples, const size_t *n_samples,
                            mynah_asr_result_cb cb, void *const *userdata);

/* Largest B one call accepts; a larger ready set is split into calls of this
 * size by the caller. A bound on the fixed per-call arrays, not a serving
 * policy: the per-worker slot cap comes from a measured T_step(B)
 * (.work/serving-v2-design.md §3). */
#define MYNAH_ASR_STREAM_BATCH_MAX 256

/* Pre-carve the batched-step scratch for up to max_b streams (sized for the
 * model's largest lookahead preset). 0 = ok, -1 = not a streaming model / OOM. */
int mynah_asr_stream_batch_reserve(mynah_asr_model *m, int max_b);

/* Rows this process has pushed through the STACKED encoder path (a row = one
 * encoder frame of one stream). 0 after a run means every step degraded to the
 * single path — the visible-fallback rule of ENGINEERING.md §6. */
unsigned long long mynah_asr_stream_batch_rows_stacked(void);

void mynah_asr_stream_close(mynah_asr_stream *s);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_ASR_H */
