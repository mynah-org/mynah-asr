#include "audio.h"

#include <math.h>

#ifndef M_PI                       /* non-ISO: glibc lo nasconde con -std=c11 */
#define M_PI 3.14159265358979323846
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

float *mynah_asr_wav_parse(const unsigned char *data, size_t len, size_t *n_samples,
                       int *sample_rate) {
    if (len < 44 || memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "audio: non è un WAV RIFF\n");
        return NULL;
    }
    int channels = 0, bits = 0, sr = 0;
    const uint8_t *pcm = NULL;
    uint32_t pcm_len = 0;

    size_t off = 12;
    while (off + 8 <= len) {
        const uint8_t *chdr = data + off;
        uint32_t sz = rd_u32(chdr + 4);
        const uint8_t *body = chdr + 8;
        if (off + 8 + sz > len) break;
        if (memcmp(chdr, "fmt ", 4) == 0 && sz >= 16) {
            uint16_t audio_format = rd_u16(body);
            channels = rd_u16(body + 2);
            sr = (int)rd_u32(body + 4);
            bits = rd_u16(body + 14);
            if (audio_format != 1 /* PCM */) {
                fprintf(stderr, "audio: unsupported WAV format %u (PCM only)\n", audio_format);
                return NULL;
            }
        } else if (memcmp(chdr, "data", 4) == 0) {
            pcm = body;
            pcm_len = sz;
        }
        off += 8 + sz + (sz & 1); /* chunk dispari: pad byte */
    }

    if (!pcm || channels <= 0 || bits != 16) {
        fprintf(stderr, "audio: expected WAV PCM16 with fmt+data chunks (ch=%d bits=%d)\n",
                channels, bits);
        return NULL;
    }

    size_t frames = pcm_len / (size_t)(channels * 2);
    float *out = malloc(frames * sizeof(float));
    if (!out) return NULL;

    /* lettura per byte (LE): un chunk data a offset dispari renderebbe il cast
     * a int16_t* disallineato (UB tecnico, ultima voce dell'audit 2026-07-18) */
    const uint8_t *s = pcm;
    for (size_t i = 0; i < frames; i++) {
        int32_t acc = 0;
        for (int c = 0; c < channels; c++) {
            const uint8_t *p = s + 2 * (i * (size_t)channels + (size_t)c);
            acc += (int16_t)(p[0] | (p[1] << 8));
        }
        out[i] = (float)acc / (float)channels / 32768.0f;
    }
    *n_samples = frames;
    *sample_rate = sr;
    return out;
}

float *mynah_asr_wav_load(const char *path, size_t *n_samples, int *sample_rate) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "audio: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) { fclose(f); return NULL; }
    unsigned char *buf = malloc((size_t)len);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "audio: read failed for %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    float *out = mynah_asr_wav_parse(buf, (size_t)len, n_samples, sample_rate);
    free(buf);
    return out;
}

float *mynah_asr_resample(const float *in, size_t n_in, int sr_in, int sr_out, size_t *n_out) {
    if (sr_in == sr_out) {
        float *copy = malloc(n_in * sizeof(float));
        if (!copy) return NULL;
        memcpy(copy, in, n_in * sizeof(float));
        *n_out = n_in;
        return copy;
    }
    const double ratio = (double)sr_out / (double)sr_in;
    /* when downsampling the sinc must be cut at the new Nyquist */
    const double fc = ratio < 1.0 ? ratio : 1.0;
    const int taps = 32;
    const size_t N = (size_t)((double)n_in * ratio);
    float *out = malloc(N * sizeof(float));
    if (!out) return NULL;

    for (size_t i = 0; i < N; i++) {
        const double center = (double)i / ratio;   /* posizione nel segnale sorgente */
        const long i0 = (long)floor(center) - taps + 1;
        double acc = 0.0, wsum = 0.0;
        for (long j = i0; j < i0 + 2 * taps; j++) {
            const double x = ((double)j - center) * fc;
            const double sinc = x == 0.0 ? 1.0 : sin(M_PI * x) / (M_PI * x);
            const double hw = 0.5 + 0.5 * cos(M_PI * ((double)j - center) / (double)taps);
            const double w = sinc * hw;
            wsum += w;
            if (j >= 0 && (size_t)j < n_in) acc += w * (double)in[j];
        }
        out[i] = (float)(acc / (wsum > 1e-9 ? wsum : 1.0));
    }
    *n_out = N;
    return out;
}
