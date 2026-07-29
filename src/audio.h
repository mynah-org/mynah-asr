/* Lettura audio: WAV PCM16 (mono/stereo -> mono). Per ora richiede 16 kHz;
 * il resampler arriva con un task dedicato (TODO 0.5). */
#ifndef MYNAH_ASR_AUDIO_H
#define MYNAH_ASR_AUDIO_H

#include <stddef.h>

/* Carica un WAV PCM16. Ritorna campioni float32 in [-1,1] (malloc, caller free)
 * e scrive *n_samples e *sample_rate. NULL su errore (messaggio su stderr). */
float *mynah_asr_wav_load(const char *path, size_t *n_samples, int *sample_rate);

/* Come sopra ma da buffer in memoria (es. upload HTTP). */
float *mynah_asr_wav_parse(const unsigned char *data, size_t len, size_t *n_samples,
                       int *sample_rate);

/* Resampling windowed-sinc (Hann, 32 tap per lato) a sr_out. Ritorna il nuovo
 * buffer (malloc) e scrive *n_out. NULL su errore. */
float *mynah_asr_resample(const float *in, size_t n_in, int sr_in, int sr_out, size_t *n_out);

#endif
