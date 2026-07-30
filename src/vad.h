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

#endif
