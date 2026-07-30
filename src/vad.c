#include "vad.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qmat.h"                 /* mynah_asr_sigmoid */
#include "weights.h"
#include "../vendor/cJSON.h"

#ifdef MYNAH_ASR_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* Structural cap, not a model constant: silero v5 has 4 encoder blocks and a
 * checkpoint with more would need a config change anyway. */
#define VAD_MAX_ENC 8

struct mynah_asr_vad {
    cJSON *cfg;
    mynah_asr_safetensors *w;

    /* geometry, all from mynah.json / the tensor shapes */
    int frame, context, pad, n_fft, hop, n_bins, stft_frames, hidden, n_enc;
    double threshold, neg_threshold;
    int min_speech_ms, min_silence_ms, speech_pad_ms;

    const float *basis;                       /* [2*n_bins, n_fft] */
    const float *enc_w[VAD_MAX_ENC], *enc_b[VAD_MAX_ENC];
    int enc_cin[VAD_MAX_ENC], enc_cout[VAD_MAX_ENC], enc_k[VAD_MAX_ENC];
    int enc_stride[VAD_MAX_ENC], enc_pl[VAD_MAX_ENC], enc_pr[VAD_MAX_ENC];
    const float *w_ih, *w_hh, *b_ih, *b_hh;   /* LSTM, gate order [i,f,g,o] */
    const float *head_w;                      /* [1, hidden, 1] conv 1x1 */
    float head_b;

    /* carried state: this is what makes the module streaming */
    float *lookbehind;                        /* [context] */
    float *h, *c;                             /* [hidden] */

    /* scratch, sized once at open */
    float *buf;                               /* [context + frame + pad] */
    float *win;                               /* [stft_frames * n_fft] */
    float *spec;                              /* [2*n_bins * stft_frames] */
    float *act[2];                            /* ping-pong activations */
    float *col;                               /* im2col */
    float *z;                                 /* [4*hidden] */
};

static const cJSON *obj(const cJSON *o, const char *k) { return cJSON_GetObjectItem(o, k); }

static int cfg_int(const cJSON *o, const char *k, int *bad) {
    const cJSON *j = obj(o, k);
    if (!cJSON_IsNumber(j)) { fprintf(stderr, "mynah-asr: vad: mynah.json: bad \"%s\"\n", k); *bad = 1; return 0; }
    return j->valueint;
}

static double cfg_num(const cJSON *o, const char *k, int *bad) {
    const cJSON *j = obj(o, k);
    if (!cJSON_IsNumber(j)) { fprintf(stderr, "mynah-asr: vad: mynah.json: bad \"%s\"\n", k); *bad = 1; return 0.0; }
    return j->valuedouble;
}

/* Loads the [n] int array `k`; returns the count read, -1 on a shape mismatch. */
static int cfg_ints(const cJSON *o, const char *k, int *out, int n) {
    const cJSON *a = obj(o, k);
    if (!cJSON_IsArray(a) || cJSON_GetArraySize(a) != n) return -1;
    for (int i = 0; i < n; i++) {
        const cJSON *e = cJSON_GetArrayItem(a, i);
        if (!cJSON_IsNumber(e)) return -1;
        out[i] = e->valueint;
    }
    return n;
}

static const float *tensor(const mynah_asr_safetensors *st, const char *name, int n_dims,
                           int64_t *shape, int *bad) {
    const mynah_asr_tensor *t = mynah_asr_st_get(st, name);
    if (!t || t->dtype != MYNAH_ASR_DT_F32 || t->n_dims != n_dims) {
        fprintf(stderr, "mynah-asr: vad: missing or non-f32 tensor \"%s\"\n", name);
        *bad = 1;
        return NULL;
    }
    if (shape) for (int i = 0; i < n_dims; i++) shape[i] = t->shape[i];
    return (const float *)t->data;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = len >= 0 ? malloc((size_t)len + 1) : NULL;
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); fclose(f); return NULL; }
    buf[len] = '\0';
    fclose(f);
    return buf;
}

/* out[t_out] = ceil-free conv1d output length */
static int conv_out(int t_in, int k, int stride, int pl, int pr) {
    return (t_in + pl + pr - k) / stride + 1;
}

mynah_asr_vad *mynah_asr_vad_open(const char *model_dir) {
    char path[1024];
    mynah_asr_vad *v = calloc(1, sizeof(*v));
    if (!v) return NULL;

    snprintf(path, sizeof(path), "%s/mynah.json", model_dir);
    char *json = read_file(path);
    if (!json) goto fail;                       /* quiet: the VAD is optional */
    v->cfg = cJSON_Parse(json);
    free(json);
    if (!v->cfg) { fprintf(stderr, "mynah-asr: vad: %s is not valid JSON\n", path); goto fail; }

    const cJSON *vc = obj(v->cfg, "vad");
    if (!cJSON_IsObject(vc)) {
        fprintf(stderr, "mynah-asr: vad: %s has no \"vad\" section (not a VAD model?)\n", path);
        goto fail;
    }
    int bad = 0;
    v->frame = cfg_int(vc, "frame_samples", &bad);
    v->context = cfg_int(vc, "context_samples", &bad);
    v->pad = cfg_int(vc, "reflect_pad", &bad);
    v->n_fft = cfg_int(vc, "n_fft", &bad);
    v->hop = cfg_int(vc, "hop_length", &bad);
    v->n_bins = cfg_int(vc, "n_bins", &bad);
    v->hidden = cfg_int(vc, "lstm_hidden", &bad);
    v->threshold = cfg_num(vc, "threshold", &bad);
    v->neg_threshold = cfg_num(vc, "neg_threshold", &bad);
    v->min_speech_ms = cfg_int(vc, "min_speech_ms", &bad);
    v->min_silence_ms = cfg_int(vc, "min_silence_ms", &bad);
    v->speech_pad_ms = cfg_int(vc, "speech_pad_ms", &bad);
    if (bad) goto fail;

    const cJSON *strides = obj(vc, "encoder_strides");
    v->n_enc = cJSON_IsArray(strides) ? cJSON_GetArraySize(strides) : 0;
    if (v->n_enc < 1 || v->n_enc > VAD_MAX_ENC) {
        fprintf(stderr, "mynah-asr: vad: %d encoder blocks (1..%d supported)\n", v->n_enc, VAD_MAX_ENC);
        goto fail;
    }
    int kernels[VAD_MAX_ENC], pads[2 * VAD_MAX_ENC];
    if (cfg_ints(vc, "encoder_strides", v->enc_stride, v->n_enc) < 0 ||
        cfg_ints(vc, "encoder_kernels", kernels, v->n_enc) < 0) {
        fprintf(stderr, "mynah-asr: vad: encoder_strides/encoder_kernels do not match\n");
        goto fail;
    }
    const cJSON *jpads = obj(vc, "encoder_pads");
    if (!cJSON_IsArray(jpads) || cJSON_GetArraySize(jpads) != v->n_enc) {
        fprintf(stderr, "mynah-asr: vad: encoder_pads does not match encoder_strides\n");
        goto fail;
    }
    for (int i = 0; i < v->n_enc; i++) {
        const cJSON *p = cJSON_GetArrayItem(jpads, i);
        if (!cJSON_IsArray(p) || cJSON_GetArraySize(p) != 2) {
            fprintf(stderr, "mynah-asr: vad: encoder_pads[%d] is not a [left, right] pair\n", i);
            goto fail;
        }
        pads[2 * i] = cJSON_GetArrayItem(p, 0)->valueint;
        pads[2 * i + 1] = cJSON_GetArrayItem(p, 1)->valueint;
    }

    /* ---- weights ------------------------------------------------------- */
    const char *wfile = cJSON_IsString(obj(v->cfg, "weights")) ? obj(v->cfg, "weights")->valuestring
                                                              : "model.safetensors";
    snprintf(path, sizeof(path), "%s/%s", model_dir, wfile);
    v->w = mynah_asr_st_open(path);
    if (!v->w) goto fail;

    int64_t shape[3];
    v->basis = tensor(v->w, "stft.forward_basis_buffer", 3, shape, &bad);
    if (!bad && (shape[0] != 2 * (int64_t)v->n_bins || shape[1] != 1 || shape[2] != v->n_fft)) {
        fprintf(stderr, "mynah-asr: vad: STFT basis %lld x %lld x %lld contradicts the config\n",
                (long long)shape[0], (long long)shape[1], (long long)shape[2]);
        bad = 1;
    }
    for (int i = 0; i < v->n_enc; i++) {
        char name[96];
        snprintf(name, sizeof(name), "encoder.%d.reparam_conv.weight", i);
        v->enc_w[i] = tensor(v->w, name, 3, shape, &bad);
        if (bad) break;
        v->enc_cout[i] = (int)shape[0];
        v->enc_cin[i] = (int)shape[1];
        v->enc_k[i] = (int)shape[2];
        v->enc_pl[i] = pads[2 * i];
        v->enc_pr[i] = pads[2 * i + 1];
        if (v->enc_k[i] != kernels[i]) {
            fprintf(stderr, "mynah-asr: vad: encoder.%d kernel %d != config %d\n",
                    i, v->enc_k[i], kernels[i]);
            bad = 1;
            break;
        }
        snprintf(name, sizeof(name), "encoder.%d.reparam_conv.bias", i);
        v->enc_b[i] = tensor(v->w, name, 1, NULL, &bad);
    }
    v->w_ih = tensor(v->w, "decoder.rnn.weight_ih", 2, shape, &bad);
    if (!bad && (shape[0] != 4 * (int64_t)v->hidden || shape[1] != v->hidden)) {
        fprintf(stderr, "mynah-asr: vad: LSTM weight_ih contradicts lstm_hidden=%d\n", v->hidden);
        bad = 1;
    }
    v->w_hh = tensor(v->w, "decoder.rnn.weight_hh", 2, NULL, &bad);
    v->b_ih = tensor(v->w, "decoder.rnn.bias_ih", 1, NULL, &bad);
    v->b_hh = tensor(v->w, "decoder.rnn.bias_hh", 1, NULL, &bad);
    v->head_w = tensor(v->w, "decoder.decoder.2.weight", 3, NULL, &bad);
    const float *hb = tensor(v->w, "decoder.decoder.2.bias", 1, NULL, &bad);
    if (bad) goto fail;
    v->head_b = hb[0];

    /* ---- geometry check: the encoder must collapse to ONE timestep ------ */
    v->stft_frames = conv_out(v->context + v->frame + v->pad, v->n_fft, v->hop, 0, 0);
    int t = v->stft_frames, cin = v->n_bins;
    size_t act_max = (size_t)v->n_bins * (size_t)v->stft_frames, col_max = 0;
    for (int i = 0; i < v->n_enc; i++) {
        if (v->enc_cin[i] != cin) {
            fprintf(stderr, "mynah-asr: vad: encoder.%d takes %d channels, %d arrive\n",
                    i, v->enc_cin[i], cin);
            goto fail;
        }
        const int t_out = conv_out(t, v->enc_k[i], v->enc_stride[i], v->enc_pl[i], v->enc_pr[i]);
        if (t_out < 1) {
            fprintf(stderr, "mynah-asr: vad: encoder.%d yields %d timesteps\n", i, t_out);
            goto fail;
        }
        const size_t need = (size_t)v->enc_cin[i] * (size_t)v->enc_k[i] * (size_t)t_out;
        if (need > col_max) col_max = need;
        if ((size_t)v->enc_cout[i] * (size_t)t_out > act_max)
            act_max = (size_t)v->enc_cout[i] * (size_t)t_out;
        cin = v->enc_cout[i];
        t = t_out;
    }
    if (t != 1 || cin != v->hidden) {
        fprintf(stderr, "mynah-asr: vad: encoder ends at %d x %d, expected %d x 1\n",
                cin, t, v->hidden);
        goto fail;
    }

    v->lookbehind = calloc((size_t)v->context, sizeof(float));
    v->h = calloc((size_t)v->hidden, sizeof(float));
    v->c = calloc((size_t)v->hidden, sizeof(float));
    v->buf = calloc((size_t)(v->context + v->frame + v->pad), sizeof(float));
    v->win = calloc((size_t)v->stft_frames * (size_t)v->n_fft, sizeof(float));
    v->spec = calloc(2 * (size_t)v->n_bins * (size_t)v->stft_frames, sizeof(float));
    v->act[0] = calloc(act_max, sizeof(float));
    v->act[1] = calloc(act_max, sizeof(float));
    v->col = calloc(col_max, sizeof(float));
    v->z = calloc(4 * (size_t)v->hidden, sizeof(float));
    if (!v->lookbehind || !v->h || !v->c || !v->buf || !v->win || !v->spec ||
        !v->act[0] || !v->act[1] || !v->col || !v->z) goto fail;
    return v;

fail:
    mynah_asr_vad_close(v);
    return NULL;
}

void mynah_asr_vad_close(mynah_asr_vad *v) {
    if (!v) return;
    if (v->cfg) cJSON_Delete(v->cfg);
    if (v->w) mynah_asr_st_close(v->w);
    free(v->lookbehind); free(v->h); free(v->c); free(v->buf);
    free(v->win); free(v->spec); free(v->act[0]); free(v->act[1]);
    free(v->col); free(v->z);
    free(v);
}

int mynah_asr_vad_frame_samples(const mynah_asr_vad *v) { return v ? v->frame : 0; }
double mynah_asr_vad_threshold(const mynah_asr_vad *v) { return v ? v->threshold : 0.5; }
double mynah_asr_vad_neg_threshold(const mynah_asr_vad *v) { return v ? v->neg_threshold : 0.35; }
int mynah_asr_vad_min_silence_ms(const mynah_asr_vad *v) { return v ? v->min_silence_ms : 100; }
int mynah_asr_vad_min_speech_ms(const mynah_asr_vad *v) { return v ? v->min_speech_ms : 250; }
int mynah_asr_vad_speech_pad_ms(const mynah_asr_vad *v) { return v ? v->speech_pad_ms : 30; }

void mynah_asr_vad_reset(mynah_asr_vad *v) {
    if (!v) return;
    memset(v->lookbehind, 0, (size_t)v->context * sizeof(float));
    memset(v->h, 0, (size_t)v->hidden * sizeof(float));
    memset(v->c, 0, (size_t)v->hidden * sizeof(float));
}

/* conv1d over [cin, t_in] -> [cout, t_out], NCHW weights [cout, cin, k].
 * im2col + one GEMM: t_out is 1..4 here, so the cost is the weight read, and a
 * single GEMM keeps the accumulation order stable across BLAS backends. */
static void conv1d(const float *in, int cin, int t_in, const float *w, const float *b,
                   int cout, int k, int stride, int pl, int pr, float *col, float *out) {
    const int t_out = conv_out(t_in, k, stride, pl, pr);
    for (int ci = 0; ci < cin; ci++)
        for (int kk = 0; kk < k; kk++) {
            float *dst = col + ((size_t)ci * (size_t)k + (size_t)kk) * (size_t)t_out;
            for (int t = 0; t < t_out; t++) {
                const int s = t * stride - pl + kk;
                dst[t] = (s >= 0 && s < t_in) ? in[(size_t)ci * (size_t)t_in + (size_t)s] : 0.0f;
            }
        }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cout, t_out, cin * k,
                1.0f, w, cin * k, col, t_out, 0.0f, out, t_out);
    for (int o = 0; o < cout; o++) {
        float *row = out + (size_t)o * (size_t)t_out;
        for (int t = 0; t < t_out; t++) {
            const float x = row[t] + b[o];
            row[t] = x > 0.0f ? x : 0.0f;              /* ReLU after every block */
        }
    }
}

float mynah_asr_vad_feed(mynah_asr_vad *v, const float *samples, size_t n) {
    if (!v || !samples || n > (size_t)v->frame) return -1.0f;
    const int C = v->context, F = v->frame, P = v->pad, N = C + F;

    /* input = lookbehind | frame (zero-padded tail) | reflect pad on the right,
     * exactly what silero's wrapper feeds the graph (docs/vad-silero.md) */
    memcpy(v->buf, v->lookbehind, (size_t)C * sizeof(float));
    memcpy(v->buf + C, samples, n * sizeof(float));
    if (n < (size_t)F) memset(v->buf + C + n, 0, ((size_t)F - n) * sizeof(float));
    for (int j = 0; j < P; j++) v->buf[N + j] = v->buf[N - 2 - j];
    memcpy(v->lookbehind, v->buf + N - C, (size_t)C * sizeof(float));

    /* STFT as a conv: the basis is a plain buffer, so windows x basis is a GEMM */
    const int T0 = v->stft_frames, B = v->n_bins;
    for (int f = 0; f < T0; f++)
        memcpy(v->win + (size_t)f * (size_t)v->n_fft, v->buf + (size_t)f * (size_t)v->hop,
               (size_t)v->n_fft * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 2 * B, T0, v->n_fft,
                1.0f, v->basis, v->n_fft, v->win, v->n_fft, 0.0f, v->spec, T0);
    float *mag = v->act[0];
    for (int b = 0; b < B; b++)
        for (int f = 0; f < T0; f++) {
            const float re = v->spec[(size_t)b * (size_t)T0 + (size_t)f];
            const float im = v->spec[(size_t)(b + B) * (size_t)T0 + (size_t)f];
            mag[(size_t)b * (size_t)T0 + (size_t)f] = sqrtf(re * re + im * im);
        }

    int cur = 0, t = T0, cin = B;
    for (int i = 0; i < v->n_enc; i++) {
        conv1d(v->act[cur], cin, t, v->enc_w[i], v->enc_b[i], v->enc_cout[i], v->enc_k[i],
               v->enc_stride[i], v->enc_pl[i], v->enc_pr[i], v->col, v->act[1 - cur]);
        t = conv_out(t, v->enc_k[i], v->enc_stride[i], v->enc_pl[i], v->enc_pr[i]);
        cin = v->enc_cout[i];
        cur = 1 - cur;
    }

    /* one LSTM step (PyTorch convention: gate order [i,f,g,o], BOTH biases —
     * the same as the RNNT prediction net, see docs/architecture-notes.md §6) */
    const int H = v->hidden;
    for (int i = 0; i < 4 * H; i++) v->z[i] = v->b_ih[i] + v->b_hh[i];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, H, 1.0f, v->w_ih, H, v->act[cur], 1, 1.0f, v->z, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, H, 1.0f, v->w_hh, H, v->h, 1, 1.0f, v->z, 1);
    for (int i = 0; i < H; i++) {
        const float ig = mynah_asr_sigmoid(v->z[i]);
        const float fg = mynah_asr_sigmoid(v->z[H + i]);
        const float gg = tanhf(v->z[2 * H + i]);
        const float og = mynah_asr_sigmoid(v->z[3 * H + i]);
        v->c[i] = fg * v->c[i] + ig * gg;
        v->h[i] = og * tanhf(v->c[i]);
    }

    /* head: ReLU(h) -> conv 1x1 -> sigmoid (the ReLU is easy to miss: it lives
     * in the ONNX between the LSTM output and decoder.decoder.2) */
    float acc = v->head_b;
    for (int i = 0; i < H; i++)
        if (v->h[i] > 0.0f) acc += v->head_w[i] * v->h[i];
    return mynah_asr_sigmoid(acc);
}
