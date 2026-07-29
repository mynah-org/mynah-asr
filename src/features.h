/* Log-mel config-driven — replica dei feature extractor HF NeMo
 * (vedi docs/nemotron-arch.md, docs/parakeet-tdt-arch.md, tools/oracle/features.py):
 * preemph -> center pad costante -> Hann simmetrica paddata a n_fft -> |rfft|^2
 * -> filterbank mel -> log(x + guard) -> normalizzazione opzionale.
 * normalize "NA" (Nemotron) o "per_feature" (Parakeet: media/std per bin sui
 * frame validi, ddof=1, x = (x-mu)/(std+1e-5)). */
#ifndef MYNAH_ASR_FEATURES_H
#define MYNAH_ASR_FEATURES_H

#include <stddef.h>

typedef struct {
    int sample_rate;       /* 16000 */
    int n_mels;            /* 128 */
    int n_fft;             /* 512 */
    int win_length;        /* 400 */
    int hop_length;        /* 160 */
    double preemphasis;    /* 0.97 */
    double log_zero_guard; /* 2^-24 */
    int normalize_per_feature; /* 0 = NA, 1 = per_feature (solo offline) */
    const float *mel_fb;   /* [n_fft/2+1, n_mels] da mel_filters.safetensors */
    const float *window;   /* [win_length] */
} mynah_asr_feat_cfg;

/* Calcola il log-mel offline. Ritorna feats [T, n_mels] float32 (malloc, caller
 * free), scrive *n_frames (= 1 + S/hop) e *valid_frames (= S/hop; i frame oltre
 * il valid sono azzerati, come nel feature extractor HF). NULL su errore. */
float *mynah_asr_log_mel(const mynah_asr_feat_cfg *cfg, const float *audio, size_t n_samples,
                     int *n_frames, int *valid_frames);

/* ------------------------------------------------------------- mel streaming
 * Incrementale, identico bit-a-bit all'offline sui frame validi (possibile solo
 * perché Nemotron non normalizza — vedi docs/prior-art.md §A.7).
 * Il frame t copre i campioni [t*hop-256, t*hop+256): è pronto quando il segnale
 * arriva a t*hop+256; a finish() si emettono i frame residui (< S/hop) leggendo
 * gli zeri del pad destro. */
typedef struct {
    const mynah_asr_feat_cfg *cfg;
    double *buf;            /* finestra scorrevole di segnale preemfatizzato   */
    size_t buf_len, buf_cap;
    size_t base;            /* indice assoluto del campione buf[0]             */
    size_t total;           /* campioni totali visti                           */
    double *win;            /* finestra Hann center-paddata a n_fft (precomp.) */
    int *mel_lo, *mel_hi;   /* range bin non-zero per filtro (precomputati)    */
    float last_raw;         /* carry per la preemphasis tra feed               */
    long next_frame;        /* prossimo frame mel da emettere                  */
    int finished;
} mynah_asr_mel_stream;

int mynah_asr_mel_stream_init(mynah_asr_mel_stream *ms, const mynah_asr_feat_cfg *cfg);
void mynah_asr_mel_stream_free(mynah_asr_mel_stream *ms);

/* Aggiunge campioni; scrive in *out (capienza cap_frames righe da n_mels) i frame
 * mel diventati pronti. Ritorna il numero di frame scritti. */
int mynah_asr_mel_stream_feed(mynah_asr_mel_stream *ms, const float *audio, size_t n,
                          float *out, int cap_frames);

/* Fine stream: emette i frame residui fino a S/hop (esclusi). */
int mynah_asr_mel_stream_finish(mynah_asr_mel_stream *ms, float *out, int cap_frames);

#endif
