/* Encoder parity: C (f32+BLAS) vs the numpy oracle (f64), layer by layer.
 * Compares layer_0, layer_12, layer_23, enc_proj (dumps from make golden-dump, it-IT).
 * Usage: test_encoder <model_dir> <file.wav> <golden_dir>
 * Exit: 0 ok, 1 mismatch, 77 skip. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/audio.h"
#include "../src/encoder.h"
#include "../src/features.h"
#include "../src/weights.h"

double *npy_load_f(const char *path, size_t *n_elems);
int test_model_cfg(const char *model_dir, int *normalize_pf, int *left, int *right,
                   int *prompt_it); /* tests/testcfg.c */

static int check(const char *name, const float *c, const char *golden_dir, size_t n_expect,
                 double rel_tol) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", golden_dir, name);
    size_t n_g;
    double *g = npy_load_f(path, &n_g);
    if (!g) { printf("skip %s (dump missing)\n", name); return 0; }
    if (n_g != n_expect) { fprintf(stderr, "FAIL %s: size %zu vs %zu\n", name, n_expect, n_g); return 1; }
    double max_d = 0.0, scale = 0.0, mean = 0.0;
    for (size_t i = 0; i < n_g; i++) {
        double d = fabs((double)c[i] - g[i]);
        if (d > max_d) max_d = d;
        if (fabs(g[i]) > scale) scale = fabs(g[i]);
        mean += d;
    }
    free(g);
    double tol = rel_tol * scale;
    printf("%-12s max|d|=%.3e mean|d|=%.3e scale=%.1f (tol %.1e) %s\n",
           name, max_d, mean / (double)n_g, scale, tol, max_d <= tol ? "OK" : "FAIL");
    return max_d <= tol ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <model_dir> <wav> <golden_dir>\n", argv[0]); return 2; }
    char path[1024];

    snprintf(path, sizeof(path), "%s/enc_proj.npy", argv[3]);
    size_t n_tmp;
    double *probe = npy_load_f(path, &n_tmp);
    if (!probe) return 77;
    free(probe);

    snprintf(path, sizeof(path), "%s/mel_filters.safetensors", argv[1]);
    mynah_asr_safetensors *mf = mynah_asr_st_open(path);
    snprintf(path, sizeof(path), "%s/model.safetensors", argv[1]);
    mynah_asr_safetensors *st = mynah_asr_st_open(path);
    if (!mf || !st) return 77;

    size_t n_samples; int sr;
    float *audio = mynah_asr_wav_load(argv[2], &n_samples, &sr);
    if (!audio || sr != 16000) return 2;

    const mynah_asr_tensor *fb = mynah_asr_st_get(mf, "mel_fb");
    const mynah_asr_tensor *win = mynah_asr_st_get(mf, "window");
    int norm_pf = 0, left, right, prompt_it;
    if (test_model_cfg(argv[1], &norm_pf, &left, &right, &prompt_it) != 0) return 77;
    mynah_asr_feat_cfg fcfg = {
        .sample_rate = 16000, .n_mels = (int)fb->shape[1], .n_fft = (int)(fb->shape[0] - 1) * 2,
        .win_length = (int)win->shape[0], .hop_length = 160,
        .preemphasis = 0.97, .log_zero_guard = pow(2.0, -24.0),
        .normalize_per_feature = norm_pf,
        .mel_fb = (const float *)fb->data, .window = (const float *)win->data,
    };
    int T_mel, valid;
    float *feats = mynah_asr_log_mel(&fcfg, audio, n_samples, &T_mel, &valid);

    mynah_asr_encoder enc;
    if (mynah_asr_encoder_init(&enc, st, 0) != 0) { fprintf(stderr, "encoder init failed\n"); return 2; }
    printf("encoder: %d layer, d=%d, heads=%d, ffn=%d, conv_k=%d, d_out=%d, causal=%d, "
           "att [%d,%d]\n", enc.n_layers, enc.d_model, enc.n_heads, enc.ffn_dim, enc.conv_k,
           enc.d_out, enc.causal, left, right);

    /* manual forward so the intermediate layers can be compared (context/prompt
     * from the config: Nemotron default preset it-IT, Parakeet full [-1,-1] with
     * no prompt) */
    int T;
    float *x = mynah_asr_subsampling_forward(&enc.ss, feats, valid, fcfg.n_mels, &T);
    float *pe = malloc((size_t)(2 * T - 1) * (size_t)enc.d_model * sizeof(float));
    mynah_asr_pos_emb(&enc, T, pe);

    int fails = 0;
    const size_t nd = (size_t)T * (size_t)enc.d_model;
    for (int li = 0; li < enc.n_layers; li++) {
        mynah_asr_encoder_layer(&enc, li, x, T, pe, left, right);
        char name[32];
        snprintf(name, sizeof(name), "layer_%d", li);
        if (li == 0 || li == enc.n_layers / 2 || li == enc.n_layers - 1)
            fails += check(name, x, argv[3], nd, 1e-4);
    }
    free(pe);

    float *out = malloc((size_t)T * (size_t)enc.d_out * sizeof(float));
    mynah_asr_encoder_post(&enc, x, T, prompt_it, out);
    fails += check("enc_proj", out, argv[3], (size_t)T * (size_t)enc.d_out, 1e-4);

    free(audio); free(feats); free(x); free(out);
    mynah_asr_encoder_free(&enc); mynah_asr_st_close(mf); mynah_asr_st_close(st);
    if (fails) { fprintf(stderr, "FAIL (%d stadi)\n", fails); return 1; }
    printf("OK\n");
    return 0;
}
