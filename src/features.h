/* Config-driven log-mel — a replica of the HF NeMo feature extractors
 * (see docs/nemotron-arch.md, docs/parakeet-tdt-arch.md, tools/oracle/features.py):
 * preemph -> constant center pad -> symmetric Hann padded to n_fft -> |rfft|^2
 * -> mel filterbank -> log(x + guard) -> optional normalization.
 * normalize "NA" (Nemotron) or "per_feature" (Parakeet: per-bin mean/std over the
 * valid frames, ddof=1, x = (x-mu)/(std+1e-5)). */
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
    int normalize_per_feature; /* 0 = NA, 1 = per_feature (offline only) */
    const float *mel_fb;   /* [n_fft/2+1, n_mels] da mel_filters.safetensors */
    const float *window;   /* [win_length] */
} mynah_asr_feat_cfg;

/* Compute the offline log-mel. Returns feats [T, n_mels] float32 (malloc'd, freed
 * by the caller), writes *n_frames (= 1 + S/hop) and *valid_frames (= S/hop;
 * frames past valid are zeroed, as in the HF feature extractor). NULL on error. */
float *mynah_asr_log_mel(const mynah_asr_feat_cfg *cfg, const float *audio, size_t n_samples,
                     int *n_frames, int *valid_frames);

/* ------------------------------------------------------------- streaming mel
 * Incremental, bit-identical to the offline path on the valid frames (only
 * possible because Nemotron does not normalize — see docs/prior-art.md §A.7).
 * Frame t covers samples [t*hop-256, t*hop+256): it is ready once the signal
 * reaches t*hop+256; finish() emits the leftover frames (< S/hop) by reading
 * the zeros of the right pad. */
typedef struct {
    const mynah_asr_feat_cfg *cfg;
    double *buf;            /* finestra scorrevole di segnale preemfatizzato   */
    size_t buf_len, buf_cap;
    size_t base;            /* absolute index of sample buf[0]                 */
    size_t total;           /* total samples seen                              */
    double *win;            /* finestra Hann center-paddata a n_fft (precomp.) */
    int *mel_lo, *mel_hi;   /* range bin non-zero per filtro (precomputati)    */
    float last_raw;         /* carry per la preemphasis tra feed               */
    long next_frame;        /* prossimo frame mel da emettere                  */
    int finished;
} mynah_asr_mel_stream;

int mynah_asr_mel_stream_init(mynah_asr_mel_stream *ms, const mynah_asr_feat_cfg *cfg);
void mynah_asr_mel_stream_free(mynah_asr_mel_stream *ms);

/* Push samples; writes into *out (capacity cap_frames rows of n_mels) the mel
 * frames that became ready. Returns the number of frames written. */
int mynah_asr_mel_stream_feed(mynah_asr_mel_stream *ms, const float *audio, size_t n,
                          float *out, int cap_frames);

/* End of stream: emits the leftover frames up to S/hop (exclusive). */
int mynah_asr_mel_stream_finish(mynah_asr_mel_stream *ms, float *out, int cap_frames);

#endif
