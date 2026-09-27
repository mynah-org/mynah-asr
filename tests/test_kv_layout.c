/* CACHE-RING-1 gate: the three K/V cache layouts (src/kvcache.h) are the SAME
 * cache. Asserted with memcmp, never with a tolerance.
 *
 * Part 1, no model (always runs). The data structure alone, against an
 * independent reference: the logical cache after n appended rows IS the last
 * min(n, left) of them, oldest first, and the window of a chunk IS that ++ the
 * chunk. Rows carry their absolute index, layer and tensor in their bits, so a
 * row in the wrong place, the wrong layer or the wrong tensor cannot compare
 * equal. Geometries cover left 0/1/prime/production, Q 1..qmax including
 * Q > left, empty -> partial -> exactly full -> first wrap -> thousands of
 * wraps, random chunk lengths from a fixed seed, and resets at random steps.
 * Fails on the FIRST differing row and says where.
 *
 * Part 2, with a streaming model (<model_dir>; exit 77 without one). One long
 * input -- every committed clip, concatenated until it crosses the cache many
 * times over -- through one encoder stream per layout, stepped in lockstep.
 * Every encoder output of every chunk and the full logical K/V cache after
 * every chunk must be byte-identical across layouts; then the same three
 * streams go through ONE batched step together (a batch mixing layouts) and
 * must reproduce the single-stream outputs byte for byte. int8, at lookahead 0
 * (Q = 1) and at the model's default.
 *
 * Part 3, `--bench [streams] [steps]` (dev signal on macOS; the Axion number is
 * the one that counts): the cache operations alone, at the production geometry
 * of the model in <model_dir> (or Nemotron's, 24 x 56 x 1024, without one),
 * exactly as a step issues them -- prepare, then per layer window K, window V,
 * commit K, commit V, then advance -- round-robin over `streams` saturated
 * caches so the working set is a server's, not one L2's. Prints ns/step and
 * bytes/step per layout. No attention arithmetic: this measures the mechanism.
 *
 * Usage: test_kv_layout [<model_dir>] [--bench [streams] [steps]]
 * Exit: 0 ok, 1 mismatch, 2 usage/IO, 77 skip (part 2 without a model).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/audio.h"
#include "../src/encoder.h"
#include "../src/features.h"
#include "../src/kvcache.h"
#include "../src/weights.h"

int test_model_cfg(const char *model_dir, int *normalize_pf, int *left, int *right,
                   int *prompt_it); /* tests/testcfg.c */

static const char *const LNAME[MYNAH_ASR_KV__N] = {"shift", "ring", "slide"};

/* ----------------------------------------------------------------- part 1 */

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}

/* the value of column c of absolute row a, layer li, tensor w: unique bits */
static float cell(long a, int li, int w, int c) {
    const uint32_t u = (uint32_t)(a * 7919 + li * 104729 + w * 1299709 + c * 15485863);
    float f;
    const uint32_t bits = 0x3F800000u | (u & 0x007FFFFFu);     /* [1, 2), never NaN */
    memcpy(&f, &bits, sizeof f);
    return f;
}

static void fill_fresh(float *fresh, long a0, int Q, int li, int w, int d) {
    for (int j = 0; j < Q; j++)
        for (int c = 0; c < d; c++) fresh[(size_t)j * d + c] = cell(a0 + j, li, w, c);
}

/* the row that should be there, compared bit for bit */
static int row_is(const float *row, long a, int li, int w, int d) {
    for (int c = 0; c < d; c++) {
        const float want = cell(a, li, w, c);
        if (memcmp(&row[c], &want, sizeof want) != 0) return 0;
    }
    return 1;
}

static int structure_one(int left, int qmax, int steps, int d, int nl) {
    mynah_asr_kv kv[MYNAH_ASR_KV__N];
    for (int L = 0; L < MYNAH_ASR_KV__N; L++)
        if (mynah_asr_kv_init(&kv[L], L, nl, left, qmax, d) != 0) return -1;
    float *fresh = malloc((size_t)qmax * d * sizeof(float));
    float *scr = malloc((size_t)(left + qmax) * d * sizeof(float));
    if (!fresh || !scr) return -1;

    long n = 0;                /* rows appended since the last reset */
    int wraps = 0, resets = 0;
    for (int s = 0; s < steps; s++) {
        if (s > 0 && rnd() % 997 == 0) {                 /* reset, mid-life */
            for (int L = 0; L < MYNAH_ASR_KV__N; L++) mynah_asr_kv_reset(&kv[L]);
            n = 0;
            resets++;
        }
        /* the first 200 steps at qmax, the next 200 at 1, then random */
        const int Q = s < 200 ? qmax : s < 400 ? 1 : 1 + (int)(rnd() % (uint32_t)qmax);
        const long valid = n < left ? n : left;
        for (int L = 0; L < MYNAH_ASR_KV__N; L++) {
            mynah_asr_kv *k = &kv[L];
            if (k->valid != valid) {
                printf("  FAIL structure left=%d qmax=%d step %d %s: valid %d, want %ld\n",
                       left, qmax, s, LNAME[L], k->valid, valid);
                return 1;
            }
            if (mynah_asr_kv_prepare(k, Q) != 0) return -1;
            for (int li = 0; li < nl; li++)
                for (int w = 0; w < 2; w++) {
                    fill_fresh(fresh, n, Q, li, w, d);
                    const float *win = mynah_asr_kv_window(k, li, w, fresh, Q, scr);
                    for (int i = 0; i < valid + Q; i++)
                        if (!row_is(win + (size_t)i * d, n - valid + i, li, w, d)) {
                            printf("  FAIL structure left=%d qmax=%d step %d %s layer %d %c: "
                                   "window row %d is not absolute row %ld\n",
                                   left, qmax, s, LNAME[L], li, w ? 'V' : 'K', i, n - valid + i);
                            return 1;
                        }
                }
            for (int li = 0; li < nl; li++)
                for (int w = 0; w < 2; w++) {
                    fill_fresh(fresh, n, Q, li, w, d);
                    mynah_asr_kv_commit(k, li, w, fresh, Q);
                }
            mynah_asr_kv_advance(k, Q);
            const long nn = n + Q, nv = nn < left ? nn : left;
            for (int li = 0; li < nl; li++)
                for (int w = 0; w < 2; w++)
                    for (int i = 0; i < nv; i++)
                        if (!row_is(mynah_asr_kv_row(k, li, w, i), nn - nv + i, li, w, d)) {
                            printf("  FAIL structure left=%d qmax=%d step %d %s layer %d %c: "
                                   "logical row %d is not absolute row %ld\n",
                                   left, qmax, s, LNAME[L], li, w ? 'V' : 'K', i, nn - nv + i);
                            return 1;
                        }
        }
        if (left > 0 && (n / left) != ((n + Q) / left)) wraps++;
        n += Q;
    }
    printf("  structure left=%-3d qmax=%-2d d=%d layers=%d: %d steps, %d wraps, %d resets, "
           "3 layouts == reference OK\n", left, qmax, d, nl, steps, wraps, resets);
    for (int L = 0; L < MYNAH_ASR_KV__N; L++) mynah_asr_kv_free(&kv[L]);
    free(fresh); free(scr);
    return 0;
}

static int part1(void) {
    static const int geo[][2] = {
        {56, 6}, {56, 16}, {70, 4}, {0, 3}, {1, 1}, {1, 4}, {3, 5}, {7, 3}, {8, 4}, {13, 13},
    };
    for (size_t g = 0; g < sizeof geo / sizeof geo[0]; g++) {
        const int rc = structure_one(geo[g][0], geo[g][1], 5000, 3, 2);
        if (rc != 0) return rc < 0 ? 2 : 1;
    }
    return 0;
}

/* ----------------------------------------------------------------- part 2 */

static const char *const CLIPS[5] = {
    "tests/audio/test_it.wav", "tests/audio/test_en.wav", "tests/audio/test_de.wav",
    "tests/audio/test_fr.wav", "tests/audio/test_es.wav",
};

typedef struct {
    mynah_asr_safetensors *st, *mf;
    mynah_asr_encoder enc;
    mynah_asr_feat_cfg fcfg;
    int left, right, prompt;
} encfix;

static int encfix_open(encfix *e, const char *dir, int quantize) {
    char path[1024];
    memset(e, 0, sizeof(*e));
    snprintf(path, sizeof(path), "%s/mel_filters.safetensors", dir);
    e->mf = mynah_asr_st_open(path);
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    e->st = mynah_asr_st_open(path);
    if (!e->mf || !e->st) return -1;
    const mynah_asr_tensor *fb = mynah_asr_st_get(e->mf, "mel_fb");
    const mynah_asr_tensor *win = mynah_asr_st_get(e->mf, "window");
    int norm_pf = 0;
    if (!fb || !win || test_model_cfg(dir, &norm_pf, &e->left, &e->right, &e->prompt) != 0)
        return -1;
    if (e->left < 0 || e->right < 0) return 77;          /* offline-only model */
    e->fcfg = (mynah_asr_feat_cfg){
        .sample_rate = 16000, .n_mels = (int)fb->shape[1], .n_fft = (int)(fb->shape[0] - 1) * 2,
        .win_length = (int)win->shape[0], .hop_length = 160,
        .preemphasis = 0.97, .log_zero_guard = pow(2.0, -24.0),
        .normalize_per_feature = norm_pf,
        .mel_fb = (const float *)fb->data, .window = (const float *)win->data,
    };
    return mynah_asr_encoder_init(&e->enc, e->st, quantize);
}

static void encfix_close(encfix *e) {
    mynah_asr_encoder_free(&e->enc);
    mynah_asr_st_close(e->st);
    mynah_asr_st_close(e->mf);
}

/* every committed clip, end to end, repeated until at least min_s seconds */
static float *long_audio(double min_s, size_t *n_out) {
    float *all = NULL;
    size_t n = 0;
    while ((double)n < min_s * 16000.0) {
        for (int c = 0; c < 5; c++) {
            size_t cn = 0;
            int sr = 0;
            float *a = mynah_asr_wav_load(CLIPS[c], &cn, &sr);
            if (!a || sr != 16000) { free(a); free(all); return NULL; }
            float *g = realloc(all, (n + cn) * sizeof(float));
            if (!g) { free(a); free(all); return NULL; }
            all = g;
            memcpy(all + n, a, cn * sizeof(float));
            n += cn;
            free(a);
        }
    }
    *n_out = n;
    return all;
}

static int model_one(encfix *e, const float *feats, int valid, int right) {
    const int nm = e->fcfg.n_mels, d = e->enc.d_model, d_out = e->enc.d_out, nl = e->enc.n_layers;
    const int q = right + 1;
    mynah_asr_enc_stream es[MYNAH_ASR_KV__N];
    for (int L = 0; L < MYNAH_ASR_KV__N; L++)
        if (mynah_asr_enc_stream_init_layout(&es[L], &e->enc, e->left, right, nm, L) != 0) return -1;
    const size_t per = (size_t)(q + 2) * (size_t)d_out;
    const size_t cf = (size_t)nl * (size_t)e->left * (size_t)d;
    const int max_steps = valid / (8 * q) + 2;
    float *ref = malloc((size_t)max_steps * per * sizeof(float));     /* shift, single */
    int *qref = calloc((size_t)max_steps, sizeof(int));
    float *out = malloc(per * sizeof(float));
    float *la = malloc(cf * sizeof(float)), *lb = malloc(cf * sizeof(float));
    if (!ref || !qref || !out || !la || !lb) return -1;

    /* single path, the three layouts in lockstep */
    int pos = 0, steps = 0, wraps;
    for (;;) {
        const int need = mynah_asr_enc_stream_need(&es[0]);
        if (pos + need > valid || steps >= max_steps) break;
        for (int L = 0; L < MYNAH_ASR_KV__N; L++) {
            float *o = L == 0 ? ref + (size_t)steps * per : out;
            const int got = mynah_asr_enc_stream_step(&es[L], feats + (size_t)pos * nm, need, nm,
                                                      e->prompt, 0, o);
            if (got <= 0) return -1;
            if (L == 0) { qref[steps] = got; continue; }
            if (got != qref[steps] || memcmp(o, ref + (size_t)steps * per,
                                             (size_t)got * (size_t)d_out * sizeof(float)) != 0) {
                printf("  FAIL model right=%d step %d: %s encoder output differs from shift\n",
                       right, steps, LNAME[L]);
                return 1;
            }
        }
        for (int w = 0; w < 2; w++) {
            mynah_asr_kv_logical(&es[0].kv, w, la);
            const size_t lf = (size_t)nl * (size_t)es[0].kv.valid * (size_t)d;
            for (int L = 1; L < MYNAH_ASR_KV__N; L++) {
                mynah_asr_kv_logical(&es[L].kv, w, lb);
                if (es[L].kv.valid != es[0].kv.valid || memcmp(la, lb, lf * sizeof(float)) != 0) {
                    printf("  FAIL model right=%d step %d: %s logical %c cache differs from shift\n",
                           right, steps, LNAME[L], w ? 'V' : 'K');
                    return 1;
                }
            }
        }
        pos += need;
        steps++;
    }
    const long frames = es[0].t_abs;
    wraps = e->left > 0 ? (int)(frames / e->left) : 0;

    /* batched path: the three layouts in ONE batch, against the single outputs */
    mynah_asr_enc_batch *bb = mynah_asr_enc_batch_new(&e->enc, MYNAH_ASR_KV__N, q, e->left);
    if (!bb) return -1;
    for (int L = 0; L < MYNAH_ASR_KV__N; L++) mynah_asr_enc_stream_reset(&es[L]);
    float *bo[MYNAH_ASR_KV__N];
    for (int L = 0; L < MYNAH_ASR_KV__N; L++) bo[L] = malloc(per * sizeof(float));
    pos = 0;
    int bsteps = 0;
    for (; bsteps < steps; bsteps++) {
        const int need = mynah_asr_enc_stream_need(&es[0]);
        mynah_asr_enc_stream *sub[MYNAH_ASR_KV__N];
        const float *mel[MYNAH_ASR_KV__N];
        int n_mel[MYNAH_ASR_KV__N], pr[MYNAH_ASR_KV__N], qq[MYNAH_ASR_KV__N];
        for (int L = 0; L < MYNAH_ASR_KV__N; L++) {
            sub[L] = &es[L];
            mel[L] = feats + (size_t)pos * nm;
            n_mel[L] = need;
            pr[L] = e->prompt;
        }
        if (mynah_asr_enc_stream_step_batch(bb, sub, MYNAH_ASR_KV__N, mel, n_mel, nm, pr, bo, qq) != 0)
            return -1;
        for (int L = 0; L < MYNAH_ASR_KV__N; L++)
            if (qq[L] != qref[bsteps] || memcmp(bo[L], ref + (size_t)bsteps * per,
                                                (size_t)qq[L] * (size_t)d_out * sizeof(float)) != 0) {
                printf("  FAIL model right=%d batched step %d: %s differs from single shift\n",
                       right, bsteps, LNAME[L]);
                return 1;
            }
        pos += need;
    }
    printf("  model int8 left=%d right=%d (Q=%d): %d chunks, %ld frames, %d cache turnovers: "
           "single 3 layouts EXACT, logical K/V EXACT every chunk, mixed-layout batch EXACT OK\n",
           e->left, right, q, steps, frames, wraps);
    mynah_asr_enc_batch_free(bb);
    for (int L = 0; L < MYNAH_ASR_KV__N; L++) { mynah_asr_enc_stream_free(&es[L]); free(bo[L]); }
    free(ref); free(qref); free(out); free(la); free(lb);
    return 0;
}

static int part2(const char *dir) {
    encfix e;
    const int rc = encfix_open(&e, dir, 1);
    if (rc == 77) { printf("  SKIP part 2: %s is not a streaming model\n", dir); return 77; }
    if (rc != 0) { printf("  SKIP part 2: cannot open %s\n", dir); return 77; }
    size_t n = 0;
    float *audio = long_audio(60.0, &n);
    if (!audio) { encfix_close(&e); return 2; }
    int t_mel = 0, valid = 0;
    float *feats = mynah_asr_log_mel(&e.fcfg, audio, n, &t_mel, &valid);
    free(audio);
    if (!feats) { encfix_close(&e); return 2; }
    printf("  input: %.1f s of audio, %d mel frames\n", (double)n / 16000.0, valid);
    int out = 0;
    const int rights[2] = {0, e.right};
    for (int r = 0; r < (e.right == 0 ? 1 : 2) && out == 0; r++) {
        const int m = model_one(&e, feats, valid, rights[r]);
        if (m != 0) out = m < 0 ? 2 : 1;
    }
    free(feats);
    encfix_close(&e);
    return out;
}

/* ----------------------------------------------------------------- part 3 */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int part3(const char *dir, int streams, int steps) {
    int nl = 24, left = 56, d = 1024, Q = 4;
    if (dir) {
        int npf, l, r, p;
        encfix e;
        if (test_model_cfg(dir, &npf, &l, &r, &p) == 0 && l > 0 && r >= 0 &&
            encfix_open(&e, dir, 1) == 0) {
            nl = e.enc.n_layers; d = e.enc.d_model; left = l; Q = r + 1;
            encfix_close(&e);
        }
    }
    printf("  bench geometry: layers=%d left=%d d=%d Q=%d, %d saturated streams round-robin, "
           "%d steps per layout\n", nl, left, d, Q, streams, steps);
    printf("  logical cache per stream: %.2f MB (K+V)\n",
           2.0 * nl * left * d * sizeof(float) / 1e6);
    mynah_asr_kv *kv = calloc((size_t)streams, sizeof(*kv));
    float *fresh = malloc((size_t)Q * d * sizeof(float));
    float *scr = malloc((size_t)(left + Q) * d * sizeof(float) * 2);
    if (!kv || !fresh || !scr) return 2;
    for (int i = 0; i < Q * d; i++) fresh[i] = (float)i * 1e-3f;
    for (int rep = 0; rep < 2; rep++)                      /* interleaved: A B C A B C */
        for (int L = 0; L < MYNAH_ASR_KV__N; L++) {
            for (int s = 0; s < streams; s++)
                if (mynah_asr_kv_init(&kv[s], L, nl, left, Q, d) != 0) return 2;
            /* saturate, and past the first wrap / compaction */
            for (int s = 0; s < streams; s++)
                for (int k = 0; k < 2 * left / Q + 3; k++) {
                    mynah_asr_kv_prepare(&kv[s], Q);
                    for (int li = 0; li < nl; li++) {
                        mynah_asr_kv_window(&kv[s], li, 0, fresh, Q, scr);
                        mynah_asr_kv_window(&kv[s], li, 1, fresh, Q, scr + (size_t)(left + Q) * d);
                        mynah_asr_kv_commit(&kv[s], li, 0, fresh, Q);
                        mynah_asr_kv_commit(&kv[s], li, 1, fresh, Q);
                    }
                    mynah_asr_kv_advance(&kv[s], Q);
                }
            unsigned long long b0 = 0;
            for (int s = 0; s < streams; s++) b0 += kv[s].bytes;
            volatile float sink = 0.0f;
            const double t0 = now_s();
            for (int k = 0; k < steps; k++) {
                mynah_asr_kv *c = &kv[k % streams];
                mynah_asr_kv_prepare(c, Q);
                for (int li = 0; li < nl; li++) {
                    const float *wk = mynah_asr_kv_window(c, li, 0, fresh, Q, scr);
                    const float *wv = mynah_asr_kv_window(c, li, 1, fresh, Q,
                                                          scr + (size_t)(left + Q) * d);
                    sink += wk[(size_t)left * d] + wv[0];       /* the window is consumed */
                    mynah_asr_kv_commit(c, li, 0, fresh, Q);
                    mynah_asr_kv_commit(c, li, 1, fresh, Q);
                }
                mynah_asr_kv_advance(c, Q);
            }
            const double dt = now_s() - t0;
            unsigned long long b1 = 0;
            for (int s = 0; s < streams; s++) b1 += kv[s].bytes;
            const double bps = (double)(b1 - b0) / steps;
            printf("  [rep %d] %-5s %9.1f us/step  %7.2f MB copied/step  %6.1f GB/s  "
                   "footprint %.2f MB/stream\n", rep, LNAME[L], dt / steps * 1e6, bps / 1e6,
                   bps * steps / dt / 1e9, 2.0 * nl * kv[0].cap * d * sizeof(float) / 1e6);
            for (int s = 0; s < streams; s++) mynah_asr_kv_free(&kv[s]);
        }
    free(kv); free(fresh); free(scr);
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    int bench = 0, streams = 16, steps = 20000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) {
            bench = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') streams = atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') steps = atoi(argv[++i]);
        } else dir = argv[i];
    }
    if (bench) return part3(dir, streams > 0 ? streams : 1, steps > 0 ? steps : 1);

    printf("test_kv_layout part 1: data structure vs reference\n");
    const int p1 = part1();
    if (p1 != 0) { printf("FAIL\n"); return p1; }
    if (!dir) { printf("OK (part 1 only: no model dir)\n"); return 0; }
    printf("test_kv_layout part 2: encoder, %s\n", dir);
    const int p2 = part2(dir);
    if (p2 == 77) return 77;
    printf(p2 == 0 ? "OK\n" : "FAIL\n");
    return p2;
}
