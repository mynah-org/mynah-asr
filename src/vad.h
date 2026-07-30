/* Silero VAD v5 (MIT, Silero Team) — speech probability per fixed-size frame.
 * Reference for every number here: docs/vad-silero.md; the weights come from
 * tools/convert_silero.py, the parity oracle is onnxruntime (tests/test_vad.sh).
 *
 * Streaming by construction: one frame in, one probability out, with the LSTM
 * state and the 64-sample lookbehind carried across calls — the same object
 * serves the offline and the streaming path (repo rule 2). */
#ifndef MYNAH_ASR_VAD_H
#define MYNAH_ASR_VAD_H

#include <stddef.h>

typedef struct mynah_asr_vad mynah_asr_vad;

/* Loads <model_dir>/{mynah.json,model.safetensors}. NULL on failure (quiet when
 * the directory does not exist: the VAD is optional). */
mynah_asr_vad *mynah_asr_vad_open(const char *model_dir);
void mynah_asr_vad_close(mynah_asr_vad *v);

/* Frame size in samples at 16 kHz: the model is defined for exactly this many
 * (512 = 32 ms) and nothing else. */
int mynah_asr_vad_frame_samples(const mynah_asr_vad *v);

/* Speech probability in [0, 1] for one frame, -1.0f on error. `n` shorter than
 * the frame is zero-padded (the tail of a file); longer is an error. */
float mynah_asr_vad_feed(mynah_asr_vad *v, const float *samples, size_t n);

/* Forgets the LSTM state and the lookbehind: use between unrelated streams. */
void mynah_asr_vad_reset(mynah_asr_vad *v);

/* Decision defaults from mynah.json (silero's own get_speech_timestamps). */
double mynah_asr_vad_threshold(const mynah_asr_vad *v);
double mynah_asr_vad_neg_threshold(const mynah_asr_vad *v);
int mynah_asr_vad_min_silence_ms(const mynah_asr_vad *v);
int mynah_asr_vad_min_speech_ms(const mynah_asr_vad *v);
int mynah_asr_vad_speech_pad_ms(const mynah_asr_vad *v);

/* ------------------------------------------------------- speech segmentation
 * Probabilities are not decisions: turning them into "here is speech" needs
 * hysteresis, and the reference for it is silero's own get_speech_timestamps.
 * The policy is a plain struct, separate from the network, for two reasons: the
 * numbers travel in mynah.json instead of being magic constants, and the logic
 * is then testable with synthetic probabilities and no checkpoint at all
 * (tests/test_vadseg.c).
 *
 * Not implemented: silero's max_speech_duration_s (default infinity). Bounding
 * segment length is the ASR side's job — it already does it in plan_segments,
 * with limits that depend on the model. */
typedef struct { size_t t0, t1; } mynah_asr_vad_span;   /* sample offsets, [t0, t1) */

typedef struct {
    int sample_rate, frame_samples;
    double threshold, neg_threshold;
    int min_speech_ms, min_silence_ms, speech_pad_ms;
} mynah_asr_vad_policy;

mynah_asr_vad_policy mynah_asr_vad_policy_of(const mynah_asr_vad *v);

/* Incremental decision state: the same code serves the offline scan and (later)
 * streaming endpointing, instead of two implementations that drift apart. */
typedef struct { int triggered; long start, temp_end, i; } mynah_asr_vad_seg;

void mynah_asr_vad_seg_reset(mynah_asr_vad_seg *s);

/* One probability in. Returns 1 and fills `out` when a speech span just closed
 * (which happens min_silence_ms AFTER the speech actually ended), else 0. */
int mynah_asr_vad_seg_feed(const mynah_asr_vad_policy *p, mynah_asr_vad_seg *s,
                           float prob, mynah_asr_vad_span *out);

/* Closes a span still open at the end of the audio. 1 if it emitted one. */
int mynah_asr_vad_seg_finish(const mynah_asr_vad_policy *p, mynah_asr_vad_seg *s,
                             size_t n_samples, mynah_asr_vad_span *out);

/* Widens spans by speech_pad_ms, splitting the difference when two spans are
 * closer than twice the padding (silero's rule, in place). */
void mynah_asr_vad_pad_spans(const mynah_asr_vad_policy *p, mynah_asr_vad_span *spans,
                             int n, size_t n_samples);

/* Whole buffer -> padded speech spans, equivalent to silero's
 * get_speech_timestamps. Returns the number of spans, or -1 (more than `cap`
 * spans, or a failure): an upper bound for cap is n_samples / (2 * frame) + 1.
 * Resets the model state first, so it is independent of previous calls. */
int mynah_asr_vad_speech_spans(mynah_asr_vad *v, const float *samples, size_t n_samples,
                               mynah_asr_vad_span *out, int cap);

#endif
