/* log-mel parity: C vs the numpy oracle.
 * Usage: test_features <model_dir> <file.wav> <golden_dir>
 * Exit: 0 ok, 1 mismatch, 77 skip (model or goldens missing). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/audio.h"
#include "../src/features.h"
#include "../src/weights.h"

double *npy_load_f(const char *path, size_t *n_elems); /* tests/npy.c */
int test_model_cfg(const char *model_dir, int *normalize_pf, int *left, int *right,
                   int *prompt_it); /* tests/testcfg.c */

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <model_dir> <wav> <golden_dir>\n", argv[0]); return 2; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/mel_filters.safetensors", argv[1]);
    mynah_asr_safetensors *mf = mynah_asr_st_open(path);
    if (!mf) return 77;

    snprintf(path, sizeof(path), "%s/mel.npy", argv[3]);
    size_t n_golden;
    double *golden = npy_load_f(path, &n_golden);
    if (!golden) { mynah_asr_st_close(mf); return 77; }

    size_t n_samples; int sr;
    float *audio = mynah_asr_wav_load(argv[2], &n_samples, &sr);
    if (!audio || sr != 16000) { fprintf(stderr, "wav non valido o non 16 kHz\n"); return 2; }

    const mynah_asr_tensor *fb = mynah_asr_st_get(mf, "mel_fb");
    const mynah_asr_tensor *win = mynah_asr_st_get(mf, "window");
    if (!fb || !win) { fprintf(stderr, "mel_filters.safetensors incompleto\n"); return 2; }

    int norm_pf = 0, left, right, prompt;
    if (test_model_cfg(argv[1], &norm_pf, &left, &right, &prompt) != 0) return 77;
    mynah_asr_feat_cfg cfg = {
        .sample_rate = 16000, .n_mels = (int)fb->shape[1], .n_fft = (int)(fb->shape[0] - 1) * 2,
        .win_length = (int)win->shape[0], .hop_length = 160,
        .preemphasis = 0.97, .log_zero_guard = pow(2.0, -24.0),
        .normalize_per_feature = norm_pf,
        .mel_fb = (const float *)fb->data, .window = (const float *)win->data,
    };

    int T, valid;
    float *feats = mynah_asr_log_mel(&cfg, audio, n_samples, &T, &valid);
    if (!feats) return 2;

    size_t n_c = (size_t)T * (size_t)cfg.n_mels;
    if (n_c != n_golden) {
        fprintf(stderr, "FAIL: shape C %d x %d = %zu vs golden %zu\n", T, cfg.n_mels, n_c, n_golden);
        return 1;
    }

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < n_c; i++) {
        double d = fabs((double)feats[i] - golden[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
    }
    double tol = 5e-4; /* feature tolerance vs the reference (prior art §B.4) */
    printf("mel parity: T=%d valid=%d n_mels=%d | max|d|=%.3e mean|d|=%.3e (tol %.0e)\n",
           T, valid, cfg.n_mels, max_diff, sum_diff / (double)n_c, tol);

    free(audio); free(feats); free(golden); mynah_asr_st_close(mf);
    if (max_diff > tol) { fprintf(stderr, "FAIL\n"); return 1; }
    printf("OK\n");
    return 0;
}
