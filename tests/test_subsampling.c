/* Subsampling parity: C (f32+BLAS) vs the numpy oracle (f64).
 * Usage: test_subsampling <model_dir> <file.wav> <golden_dir>
 * Exit: 0 ok, 1 mismatch, 77 skip. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/audio.h"
#include "../src/features.h"
#include "../src/subsampling.h"
#include "../src/weights.h"

double *npy_load_f(const char *path, size_t *n_elems); /* shared from npy.c */
int test_model_cfg(const char *model_dir, int *normalize_pf, int *left, int *right,
                   int *prompt_it); /* tests/testcfg.c */

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <model_dir> <wav> <golden_dir>\n", argv[0]); return 2; }
    char path[1024];

    snprintf(path, sizeof(path), "%s/subsampling.npy", argv[3]);
    size_t n_golden;
    double *golden = npy_load_f(path, &n_golden);
    if (!golden) return 77;

    snprintf(path, sizeof(path), "%s/mel_filters.safetensors", argv[1]);
    mynah_asr_safetensors *mf = mynah_asr_st_open(path);
    snprintf(path, sizeof(path), "%s/model.safetensors", argv[1]);
    mynah_asr_safetensors *st = mynah_asr_st_open(path);
    if (!mf || !st) { free(golden); return 77; }

    size_t n_samples; int sr;
    float *audio = mynah_asr_wav_load(argv[2], &n_samples, &sr);
    if (!audio || sr != 16000) return 2;

    const mynah_asr_tensor *fb = mynah_asr_st_get(mf, "mel_fb");
    const mynah_asr_tensor *win = mynah_asr_st_get(mf, "window");
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

    mynah_asr_subsampling ss;
    if (mynah_asr_subsampling_init(&ss, st) != 0) { fprintf(stderr, "subsampling init failed\n"); return 2; }

    int t_out;
    float *out = mynah_asr_subsampling_forward(&ss, feats, valid, cfg.n_mels, &t_out);
    if (!out) return 2;

    size_t n_c = (size_t)t_out * (size_t)ss.d_model;
    if (n_c != n_golden) {
        fprintf(stderr, "FAIL: shape C %d x %d = %zu vs golden %zu\n", t_out, ss.d_model, n_c, n_golden);
        return 1;
    }
    double max_diff = 0.0, mean = 0.0, scale = 0.0;
    for (size_t i = 0; i < n_c; i++) {
        double d = fabs((double)out[i] - golden[i]);
        if (d > max_diff) max_diff = d;
        if (fabs(golden[i]) > scale) scale = fabs(golden[i]);
        mean += d;
    }
    /* C is f32+BLAS, the oracle is f64: the tolerance is relative to the value
     * scale (here |max| ~1e3, so ~1e-2 absolute). Prior art uses 4e-5..6e-3 per
     * stage. */
    double tol = 1e-5 * scale;
    printf("subsampling parity: T=%d d=%d | max|d|=%.3e mean|d|=%.3e scale=%.1f (tol %.1e)\n",
           t_out, ss.d_model, max_diff, mean / (double)n_c, scale, tol);

    free(audio); free(feats); free(out); free(golden);
    mynah_asr_st_close(mf); mynah_asr_st_close(st);
    if (max_diff > tol) { fprintf(stderr, "FAIL\n"); return 1; }
    printf("OK\n");
    return 0;
}
