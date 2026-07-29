/* Audio input: WAV PCM16 (mono/stereo -> mono). For now 16 kHz is required;
 * the resampler lands with a dedicated task (TODO 0.5). */
#ifndef MYNAH_ASR_AUDIO_H
#define MYNAH_ASR_AUDIO_H

#include <stddef.h>

/* Load a PCM16 WAV. Returns float32 samples in [-1,1] (malloc'd, freed by the
 * caller) and writes *n_samples and *sample_rate. NULL on error (message on
 * stderr). */
float *mynah_asr_wav_load(const char *path, size_t *n_samples, int *sample_rate);

/* Same as above but from an in-memory buffer (e.g. an HTTP upload). */
float *mynah_asr_wav_parse(const unsigned char *data, size_t len, size_t *n_samples,
                       int *sample_rate);

/* Windowed-sinc resampling (Hann, 32 taps per side) to sr_out. Returns the new
 * buffer (malloc'd) and writes *n_out. NULL on error. */
float *mynah_asr_resample(const float *in, size_t n_in, int sr_in, int sr_out, size_t *n_out);

#endif
