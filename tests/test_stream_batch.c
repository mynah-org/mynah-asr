/* S1-4 gate: the batched stream step is byte-identical to B single steps.
 * A transcript must never depend on who a stream was batched with
 * (ENGINEERING.md §9), so this is asserted with == and not with a tolerance.
 *
 * Gate A level 2 (strict). The encoder output of every chunk of every stream,
 * float for float, plus the final K/V and conv caches, between B single
 * mynah_asr_enc_stream_step calls and one mynah_asr_enc_stream_step_batch over
 * the same chunks. int8 and f32, B = 2, 4, 8. This is also the MEASUREMENT that
 * decides whether the f32 path may stack rows on this BLAS: cblas_sgemm is not
 * contractually row-stable in M.
 *
 * Gate A level 1 (end to end). B streams over different clips, fed one chunk
 * each per step through the public mynah_asr_stream_step_batch (need_samples,
 * so every stream takes exactly one encoder chunk per step). The clips have
 * different lengths, so the streams finish at different steps and the batch
 * shrinks while it runs. Each stream's final text must equal a single-stream
 * mynah_asr_stream_feed run of the same clip, and mynah_asr_transcribe of it
 * wherever streaming and offline already agree at that quant. The qmat counters
 * say which PATH ran and the S5-1 kernel counters say which MICRO-KERNEL did
 * the arithmetic inside it (ENGINEERING.md §5-6): the T > 16 dequant+sgemm
 * fallback must never be reached, and the kernel --dispatch-map names must be
 * the one that actually executed, or an identical transcript would be proving
 * nothing about a kernel that quietly fell back.
 *
 * Gate B, `--steptime` (dev signal, NOT a serving claim). Wall per step for
 * B = 1,2,4,8, single-stream vs batched, printed as a table.
 *
 * Gate C, `--allocs` (needs tests/libmalloc_count inserted): the batched step
 * allocates nothing after warm-up. Driven by `make test-stream-batch-allocs`.
 *
 * Usage: test_stream_batch <model_dir> [--steptime|--allocs]
 * Exit: 0 ok, 1 mismatch, 2 usage/IO, 77 skip (no model / not a streaming model).
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>

#include "../src/audio.h"
#include "../src/encoder.h"
#include "../src/features.h"
#include "../src/mynah_asr.h"
#include "../src/threads.h"
#include "../src/qmat.h"
#include "../src/weights.h"

int test_model_cfg(const char *model_dir, int *normalize_pf, int *left, int *right,
                   int *prompt_it); /* tests/testcfg.c */

#define MAX_B 8
#define TEXT_CAP 8192

static const char *const CLIPS[5] = {
    "tests/audio/test_it.wav", "tests/audio/test_en.wav", "tests/audio/test_de.wav",
    "tests/audio/test_fr.wav", "tests/audio/test_es.wav",
};
static const char *const LANGS[5] = {"it-IT", "en-US", "de-DE", "fr-FR", "es-ES"};

typedef struct {
    char text[TEXT_CAP];
    size_t n;
} sink;

static void collect(const mynah_asr_result *res, void *ud) {
    sink *k = ud;
    const size_t len = strlen(res->text);
    if (k->n + len + 1 >= TEXT_CAP) return;
    memcpy(k->text + k->n, res->text, len + 1);
    k->n += len;
}

typedef struct {
    float *audio;
    size_t n;
    const char *lang;
    const char *path;
} clip;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* One stream alone, paced by need_samples: the reference every batched run is
 * compared against. */
static int run_single(mynah_asr_model *m, const clip *c, char *out) {
    sink k = {.n = 0};
    k.text[0] = '\0';
    mynah_asr_stream *s = mynah_asr_stream_open(m, c->lang, -1);
    if (!s) return -1;
    size_t fed = 0;
    while (fed < c->n) {
        size_t need = mynah_asr_stream_need_samples(s);
        if (need == 0) { mynah_asr_stream_close(s); return -1; }
        if (need > c->n - fed) need = c->n - fed;
        if (mynah_asr_stream_feed(s, c->audio + fed, need, collect, &k) != 0) {
            mynah_asr_stream_close(s);
            return -1;
        }
        fed += need;
    }
    mynah_asr_stream_finish(s, collect, &k);
    mynah_asr_stream_close(s);
    memcpy(out, k.text, k.n + 1);
    return 0;
}

/* B streams through the batched API, one chunk each per step. Streams whose
 * audio is exhausted leave the ready set, so the batch really does shrink. */
static int run_batched(mynah_asr_model *m, const clip *cs, int B, char texts[][TEXT_CAP],
                       double *wall_per_step, long *steps_out) {
    mynah_asr_stream *ss[MAX_B];
    sink sinks[MAX_B];
    size_t fed[MAX_B];
    for (int i = 0; i < B; i++) {
        ss[i] = mynah_asr_stream_open(m, cs[i].lang, -1);
        if (!ss[i]) return -1;
        sinks[i].n = 0;
        sinks[i].text[0] = '\0';
        fed[i] = 0;
    }

    double wall = 0.0;
    long steps = 0;
    for (;;) {
        mynah_asr_stream *act[MAX_B];
        const float *bufs[MAX_B];
        size_t ns[MAX_B];
        void *uds[MAX_B];
        int n_act = 0;
        for (int i = 0; i < B; i++) {
            if (fed[i] >= cs[i].n) continue;
            size_t need = mynah_asr_stream_need_samples(ss[i]);
            if (need > cs[i].n - fed[i]) need = cs[i].n - fed[i];
            act[n_act] = ss[i];
            bufs[n_act] = cs[i].audio + fed[i];
            ns[n_act] = need;
            uds[n_act] = &sinks[i];
            fed[i] += need;
            n_act++;
        }
        if (n_act == 0) break;
        const double t0 = now_s();
        if (mynah_asr_stream_step_batch(act, n_act, bufs, ns, collect, uds) != 0) return -1;
        wall += now_s() - t0;
        steps++;
    }
    for (int i = 0; i < B; i++) {
        mynah_asr_stream_finish(ss[i], collect, &sinks[i]);
        mynah_asr_stream_close(ss[i]);
        memcpy(texts[i], sinks[i].text, sinks[i].n + 1);
    }
    if (wall_per_step) *wall_per_step = steps ? wall / (double)steps : 0.0;
    if (steps_out) *steps_out = steps;
    return 0;
}

/* B copies of the SAME work through B separate single-stream runs, timed the
 * same way: the "single" column of the step-time table. */
static double run_single_steps_timed(mynah_asr_model *m, const clip *cs, int B, long *steps_out) {
    mynah_asr_stream *ss[MAX_B];
    size_t fed[MAX_B];
    for (int i = 0; i < B; i++) {
        ss[i] = mynah_asr_stream_open(m, cs[i].lang, -1);
        if (!ss[i]) return -1.0;
        fed[i] = 0;
    }
    double wall = 0.0;
    long steps = 0;
    for (;;) {
        int any = 0;
        const double t0 = now_s();
        for (int i = 0; i < B; i++) {
            if (fed[i] >= cs[i].n) continue;
            size_t need = mynah_asr_stream_need_samples(ss[i]);
            if (need > cs[i].n - fed[i]) need = cs[i].n - fed[i];
            mynah_asr_stream_feed(ss[i], cs[i].audio + fed[i], need, NULL, NULL);
            fed[i] += need;
            any = 1;
        }
        if (!any) break;
        wall += now_s() - t0;
        steps++;
    }
    for (int i = 0; i < B; i++) mynah_asr_stream_close(ss[i]);
    if (steps_out) *steps_out = steps;
    return steps ? wall / (double)steps : 0.0;
}


/* ------------------------------------------------------------- Gate A, level 2
 * Transcripts agreeing is necessary but weak: a greedy argmax hides small float
 * differences. This gate compares the ENCODER OUTPUT of every chunk of every
 * stream, float for float (memcmp), plus the final K/V and conv caches, between
 * B single steps and one batched step over the same chunks. That is the "==" the
 * batching rule asks for (CLAUDE.md rule 4). */
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
    if (e->left < 0 || e->right < 0) return -1;      /* offline-only model */
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

static size_t cache_floats(const encfix *e) {
    return (size_t)e->enc.n_layers * (size_t)e->left * (size_t)e->enc.d_model;
}
static size_t ccache_floats(const encfix *e) {
    return (size_t)e->enc.n_layers * (size_t)(e->enc.conv_k - 1) * (size_t)e->enc.d_model;
}

static int bitexact_one(encfix *ep, int quantize, clip *cs, int n_clips, int B) {
    encfix e = *ep;
    const int nm = e.fcfg.n_mels, d_out = e.enc.d_out, q = e.right + 1;

    float *feats[MAX_B];
    int valid[MAX_B], pos[MAX_B];
    mynah_asr_enc_stream es[MAX_B];
    for (int i = 0; i < B; i++) {
        const clip *c = &cs[i % n_clips];
        int t_mel = 0;
        feats[i] = mynah_asr_log_mel(&e.fcfg, c->audio, c->n, &t_mel, &valid[i]);
        pos[i] = 0;
        if (!feats[i] || mynah_asr_enc_stream_init(&es[i], &e.enc, e.left, e.right, nm) != 0)
            return -1;
    }
    /* room for every chunk of every stream: (q+2) rows is what the scratch allows */
    const size_t per_step = (size_t)(q + 2) * (size_t)d_out;
    const int max_steps = 64;
    float *A = calloc((size_t)B * (size_t)max_steps * per_step, sizeof(float));
    float *Bo = calloc((size_t)B * (size_t)max_steps * per_step, sizeof(float));
    float *kA = calloc((size_t)B * cache_floats(&e), sizeof(float));
    float *vA = calloc((size_t)B * cache_floats(&e), sizeof(float));
    float *cA = calloc((size_t)B * ccache_floats(&e), sizeof(float));
    /* 0 = "this stream had no chunk at that step": the streams run out at
     * different steps, so the grids are ragged and only the common cells count */
    int qA[MAX_B * 64] = {0}, qB[MAX_B * 64] = {0}, nsteps = 0;
    if (!A || !Bo || !kA || !vA || !cA) return -1;

    /* pass 1: every stream on its own, mynah_asr_enc_stream_step */
    for (;;) {
        int any = 0;
        for (int i = 0; i < B; i++) {
            const int need = mynah_asr_enc_stream_need(&es[i]);
            if (pos[i] + need > valid[i]) continue;
            const int got = mynah_asr_enc_stream_step(&es[i], feats[i] + (size_t)pos[i] * (size_t)nm,
                                                  need, nm, e.prompt, 0,
                                                  A + ((size_t)i * max_steps + nsteps) * per_step);
            if (got < 0) return -1;
            qA[i * 64 + nsteps] = got;
            pos[i] += need;
            any = 1;
        }
        if (!any) break;
        if (++nsteps >= max_steps) break;
    }
    for (int i = 0; i < B; i++) {
        memcpy(kA + (size_t)i * cache_floats(&e), es[i].k_cache, cache_floats(&e) * sizeof(float));
        memcpy(vA + (size_t)i * cache_floats(&e), es[i].v_cache, cache_floats(&e) * sizeof(float));
        memcpy(cA + (size_t)i * ccache_floats(&e), es[i].conv_cache, ccache_floats(&e) * sizeof(float));
    }

    /* pass 2: the same chunks, batched */
    mynah_asr_enc_batch *bb = mynah_asr_enc_batch_new(&e.enc, B, q, e.left);
    if (!bb) return -1;
    mynah_asr_enc_relpos_counters_reset();       /* S1-7: shared vs private rk */
    for (int i = 0; i < B; i++) { mynah_asr_enc_stream_reset(&es[i]); pos[i] = 0; }
    int used_batched = 0;
    for (int step = 0; step < nsteps; step++) {
        mynah_asr_enc_stream *sub[MAX_B];
        const float *mel[MAX_B];
        int n_mel[MAX_B], pr[MAX_B], qq[MAX_B], idx[MAX_B];
        float *outs[MAX_B];
        int g = 0;
        for (int i = 0; i < B; i++) {
            const int need = mynah_asr_enc_stream_need(&es[i]);
            if (pos[i] + need > valid[i]) continue;
            sub[g] = &es[i];
            mel[g] = feats[i] + (size_t)pos[i] * (size_t)nm;
            n_mel[g] = need;
            pr[g] = e.prompt;
            outs[g] = Bo + ((size_t)i * max_steps + step) * per_step;
            idx[g] = i;
            pos[i] += need;
            g++;
        }
        if (g == 0) break;
        if (mynah_asr_enc_stream_step_batch(bb, sub, g, mel, n_mel, nm, pr, outs, qq) != 0) {
            mynah_asr_enc_batch_free(bb);
            return -1;
        }
        if (g > 1) used_batched = 1;
        for (int j = 0; j < g; j++) qB[idx[j] * 64 + step] = qq[j];
    }

    /* compare: every frame of every chunk, then the caches */
    size_t diff_floats = 0, cmp_floats = 0;
    int qdiff = 0;
    for (int i = 0; i < B; i++)
        for (int step = 0; step < nsteps; step++) {
            if (qA[i * 64 + step] == 0) continue;            /* stream already done */
            if (qA[i * 64 + step] != qB[i * 64 + step]) { qdiff = 1; continue; }
            const size_t n = (size_t)qA[i * 64 + step] * (size_t)d_out;
            const float *a = A + ((size_t)i * max_steps + step) * per_step;
            const float *b = Bo + ((size_t)i * max_steps + step) * per_step;
            for (size_t j = 0; j < n; j++) if (memcmp(&a[j], &b[j], sizeof(float)) != 0) diff_floats++;
            cmp_floats += n;
        }
    size_t cdiff = 0;
    for (int i = 0; i < B; i++) {
        const float *k = kA + (size_t)i * cache_floats(&e), *v = vA + (size_t)i * cache_floats(&e);
        const float *c = cA + (size_t)i * ccache_floats(&e);
        if (memcmp(k, es[i].k_cache, cache_floats(&e) * sizeof(float)) != 0) cdiff++;
        if (memcmp(v, es[i].v_cache, cache_floats(&e) * sizeof(float)) != 0) cdiff++;
        if (memcmp(c, es[i].conv_cache, ccache_floats(&e) * sizeof(float)) != 0) cdiff++;
    }

    /* S1-7: the sharing must have happened, and it must not have changed a
     * single float above — the two assertions belong to the same line. */
    const unsigned long long rp_sh = mynah_asr_enc_relpos_counter(MYNAH_ASR_RELPOS_SHARED);
    const unsigned long long rp_pv = mynah_asr_enc_relpos_counter(MYNAH_ASR_RELPOS_PRIVATE);
    const unsigned long long rp_gr = mynah_asr_enc_relpos_counter(MYNAH_ASR_RELPOS_GROUP);
    const int shared_ok = rp_sh > 0;

    const int ok = !qdiff && diff_floats == 0 && cdiff == 0 && used_batched && shared_ok;
    printf("  [%s] encoder bit-exact B=%d: %d steps, %zu floats compared, %zu differ, "
           "%zu/%d caches differ%s | rel-pos shared %llu private %llu group %llu%s | %s\n",
           quantize ? "int8" : "f32", B, nsteps, cmp_floats, diff_floats, cdiff, 3 * B,
           qdiff ? ", FRAME COUNT DIFFERS" : "", rp_sh, rp_pv, rp_gr,
           shared_ok ? "" : " NO SHARING", ok ? "EXACT OK" : "FAIL");

    mynah_asr_enc_batch_free(bb);
    for (int i = 0; i < B; i++) { mynah_asr_enc_stream_free(&es[i]); free(feats[i]); }
    free(A); free(Bo); free(kA); free(vA); free(cA);
    return ok ? 0 : 1;
}

/* One encoder build (an int8 one quantizes the whole model at load), several B. */
static int gate_bitexact(const char *dir, int quantize, clip *cs, int n_clips,
                         const int *Bs, int n_b) {
    encfix e;
    if (encfix_open(&e, dir, quantize) != 0) { encfix_close(&e); return -1; }
    int rc = 0;
    for (int i = 0; i < n_b && rc >= 0; i++) {
        const int r = bitexact_one(&e, quantize, cs, n_clips, Bs[i]);
        if (r < 0) rc = -1; else rc |= r;
    }
    encfix_close(&e);
    return rc;
}

/* --------------------------------------------------------------- Gate C, allocs
 * The batched step must allocate NOTHING after warm-up (same rule as the single
 * step, S1-3). tests/libmalloc_count exports its counter, so when the library is
 * inserted (DYLD_INSERT_LIBRARIES / LD_PRELOAD) this samples it around a run of
 * batched steps instead of differencing two processes. Without the interposer
 * the symbol is absent and the gate skips. */
typedef unsigned long (*mc_fn)(void);

static int gate_allocs(const char *dir, clip *cs, int n_clips, int B) {
    /* dlsym rather than a weak reference: the counter lives in an inserted
     * library, so it is either already in the process or it is not */
    const mc_fn mcount = (mc_fn)dlsym(RTLD_DEFAULT, "mynah_asr_malloc_count");
    if (!mcount) {
        printf("  allocs: SKIP (malloc_count not inserted)\n");
        return 77;
    }
    mynah_asr_model *m = mynah_asr_load_quant(dir, MYNAH_ASR_QUANT_INT8);
    if (!m) return -1;
    if (mynah_asr_stream_batch_reserve(m, B) != 0) { mynah_asr_free(m); return -1; }

    /* the longest clip for every slot, so no stream runs out mid-measurement */
    int longest = 0;
    for (int i = 1; i < n_clips; i++) if (cs[i].n > cs[longest].n) longest = i;
    mynah_asr_stream *ss[MAX_B];
    const float *bufs[MAX_B];
    size_t ns[MAX_B], fed = 0;
    for (int i = 0; i < B; i++) {
        ss[i] = mynah_asr_stream_open(m, cs[longest].lang, -1);
        if (!ss[i]) { mynah_asr_free(m); return -1; }
    }

    unsigned long a0 = 0;
    int steps = 0, measured = 0;
    const int warmup = 4;
    for (;;) {
        size_t need = mynah_asr_stream_need_samples(ss[0]);
        if (fed + need > cs[longest].n) break;
        for (int i = 0; i < B; i++) { bufs[i] = cs[longest].audio + fed; ns[i] = need; }
        if (mynah_asr_stream_step_batch(ss, B, bufs, ns, NULL, NULL) != 0) { mynah_asr_free(m); return -1; }
        fed += need;
        steps++;
        if (steps == warmup) a0 = mcount();
        else if (steps > warmup) measured++;
    }
    const unsigned long a1 = mcount();
    for (int i = 0; i < B; i++) mynah_asr_stream_close(ss[i]);
    mynah_asr_free(m);

    if (measured <= 0) { printf("  allocs: SKIP (clip too short)\n"); return 77; }
    const unsigned long delta = a1 - a0;
    printf("  allocs: B=%d, %d steps after warm-up, %lu allocations (%.2f per step) | %s\n",
           B, measured, delta, (double)delta / measured, delta == 0 ? "OK" : "FAIL");
    return delta == 0 ? 0 : 1;
}

static const char *quant_name(int q) { return q == MYNAH_ASR_QUANT_INT8 ? "int8" : "f32"; }

static int gate_identity(const char *dir, int quant, clip *cs, int n_clips) {
    mynah_asr_model *m = mynah_asr_load_quant(dir, quant);
    if (!m) { fprintf(stderr, "batch: load failed (%s)\n", quant_name(quant)); return -1; }

    /* references: offline transcribe + a single paced stream, per clip.
     * streaming == offline is test_streaming's gate, not this one: where the
     * two already differ at this quant (a numerical property of the model, not
     * of batching) the offline comparison is reported and skipped, and the
     * batched == single gate still has to hold exactly. */
    char off[5][TEXT_CAP], one[5][TEXT_CAP];
    int off_eq[5];
    for (int i = 0; i < n_clips; i++) {
        char *t = mynah_asr_transcribe(m, cs[i].audio, cs[i].n, cs[i].lang, -1, NULL);
        if (!t) { mynah_asr_free(m); return -1; }
        snprintf(off[i], TEXT_CAP, "%s", t);
        free(t);
        if (run_single(m, &cs[i], one[i]) != 0) { mynah_asr_free(m); return -1; }
        off_eq[i] = strcmp(off[i], one[i]) == 0;
        if (!off_eq[i])
            printf("  [%s] %s: single stream != offline (pre-existing, not a batching "
                   "effect; the batched gate below is unaffected)\n    off: %s\n    one: %s\n",
                   quant_name(quant), cs[i].lang, off[i], one[i]);
    }

    int fail = 0;
    const int Bs[5] = {1, 2, 3, 4, 8};
    for (int bi = 0; bi < 5; bi++) {
        const int B = Bs[bi];
        clip use[MAX_B];
        for (int i = 0; i < B; i++) use[i] = cs[i % n_clips];
        char texts[MAX_B][TEXT_CAP];

        mynah_asr_qmat_counters_reset();
        mynah_asr_qmat_kernel_counters_reset();
        mynah_asr_enc_relpos_counters_reset();
        const unsigned long long rows0 = mynah_asr_stream_batch_rows_stacked();
        long steps = 0;
        if (run_batched(m, use, B, texts, NULL, &steps) != 0) { mynah_asr_free(m); return -1; }
        const unsigned long long rows = mynah_asr_stream_batch_rows_stacked() - rows0;

        int ok = 1;
        for (int i = 0; i < B; i++) {
            const int c = i % n_clips;
            if (strcmp(texts[i], one[c]) != 0 || (off_eq[c] && strcmp(texts[i], off[c]) != 0)) {
                ok = 0;
                printf("    stream %d (%s) DIFFERS\n      want: %s\n      got : %s\n",
                       i, use[i].lang, one[c], texts[i]);
            }
        }
        const unsigned long long rp_sh = mynah_asr_enc_relpos_counter(MYNAH_ASR_RELPOS_SHARED);
        const unsigned long long rp_pv = mynah_asr_enc_relpos_counter(MYNAH_ASR_RELPOS_PRIVATE);
        const unsigned long long rp_gr = mynah_asr_enc_relpos_counter(MYNAH_ASR_RELPOS_GROUP);
        printf("  [%s] B=%d: %2ld steps, %4llu stacked rows | dot %llu  dot_rows %llu  "
               "f32 %llu  generic %llu  qgemm %llu  DEQUANT %llu | relpos sh %llu pv %llu "
               "gr %llu | %s\n",
               quant_name(quant), B, steps, rows,
               mynah_asr_qmat_counter(MYNAH_ASR_QC_DOT),
               mynah_asr_qmat_counter(MYNAH_ASR_QC_DOT_ROWS),
               mynah_asr_qmat_counter(MYNAH_ASR_QC_F32),
               mynah_asr_qmat_counter(MYNAH_ASR_QC_GENERIC),
               mynah_asr_qmat_counter(MYNAH_ASR_QC_QGEMM),
               mynah_asr_qmat_counter(MYNAH_ASR_QC_DEQUANT),
               rp_sh, rp_pv, rp_gr,
               ok ? "IDENTICAL OK" : "DIFFERENT FAIL");
        if (!ok) fail = 1;
        /* S1-7: at B > 1 the streams reach a common K within a few steps, so a
         * run that took the stacked path and never shared means the sharing
         * silently stopped working. (rows == 0 = the whole batched path
         * degraded — an f32 build with MYNAH_ASR_BATCH_F32=0 — and the line
         * above already reports that.) */
        if (B > 1 && rows > 0 && rp_sh == 0) {
            printf("  [%s] B=%d: the rel-pos projection was never shared FAIL\n",
                   quant_name(quant), B);
            fail = 1;
        }
        /* the dequant+sgemm fallback mallocs per call and changes the numerics:
         * reaching it from a stream step is a failure, not a slow path */
        if (mynah_asr_qmat_counter(MYNAH_ASR_QC_DEQUANT) != 0) {
            printf("  [%s] B=%d: reached the dequant+sgemm fallback FAIL\n", quant_name(quant), B);
            fail = 1;
        }
        if (quant == MYNAH_ASR_QUANT_INT8 && B > 1 && rows == 0) {
            printf("  [%s] B=%d: nothing went through the stacked path FAIL\n",
                   quant_name(quant), B);
            fail = 1;
        }
        /* S5-1: WHICH micro-kernel did the arithmetic. A transcript that is
         * identical while the kernel silently fell back to the pre-S5-1 loop
         * would be a green gate over a dead kernel (ENGINEERING.md §5). */
        if (quant == MYNAH_ASR_QUANT_INT8) {
            printf("  [%s] B=%d: int8 kernels", quant_name(quant), B);
            for (int ki = 0; ki < MYNAH_ASR_QK__N; ki++) {
                const unsigned long long c = mynah_asr_qmat_kernel_counter(ki);
                if (c) printf("  %s %llu", mynah_asr_qmat_kernel_name(ki), c);
            }
            printf("  | resolved %s\n", mynah_asr_qmat_int8_kernel());
            const int want = mynah_asr_qmat_kernel_resolved();
            if (B > 1 && rows > 0 && mynah_asr_qmat_kernel_counter(want) == 0) {
                printf("  [%s] B=%d: the resolved kernel %s never ran FAIL\n",
                       quant_name(quant), B, mynah_asr_qmat_kernel_name(want));
                fail = 1;
            }
        }
    }
    mynah_asr_free(m);
    return fail;
}

/* Gate B: step wall for B = 1,2,4,8, single vs batched. A development signal on
 * this host, never a serving number (ENGINEERING.md §8: WAVE screens, SOAK
 * promotes; this is neither). */
static void gate_steptime(const char *dir, clip *cs, int n_clips) {
    mynah_asr_model *m = mynah_asr_load_quant(dir, MYNAH_ASR_QUANT_INT8);
    if (!m) return;
    /* The platform is READ, not asserted. This line said "macOS dev signal" on
     * every host until it was run on Linux and printed that about a Neoverse:
     * a label that names the wrong machine is worse than no label, because it
     * travels into a note as if it were provenance. */
    {
        struct utsname un;
        const char *host = (uname(&un) == 0) ? un.sysname : "unknown-platform";
        printf("\n  step time, int8, %s, threads=%d — a DIAGNOSTIC signal, NOT a serving\n"
               "  claim: one process, unpinned, and this harness does not check that the\n"
               "  box is idle. B=1 carries the warm-up and is not part of any fit.\n",
               host, mynah_asr_num_threads());
    }
    printf("  %3s | %10s | %10s | %6s | %s\n", "B", "single ms", "batched ms", "ratio", "ms/stream");
    const int Bs[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int bi = 0; bi < 8; bi++) {
        const int B = Bs[bi];
        clip use[MAX_B];
        for (int i = 0; i < B; i++) use[i] = cs[i % n_clips];
        char texts[MAX_B][TEXT_CAP];
        long s1 = 0, s2 = 0;
        const double single = run_single_steps_timed(m, use, B, &s1);
        double batched = 0.0;
        if (run_batched(m, use, B, texts, &batched, &s2) != 0) break;
        printf("  %3d | %10.2f | %10.2f | %6.2f | %.2f\n", B, single * 1e3, batched * 1e3,
               single > 0 ? batched / single : 0.0, batched * 1e3 / B);
    }
    mynah_asr_free(m);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model_dir> [wav] [golden]\n", argv[0]); return 2; }
    char path[1024];
    snprintf(path, sizeof(path), "%s/mynah.json", argv[1]);
    FILE *probe = fopen(path, "rb");
    if (!probe) return 77;
    fclose(probe);

    /* the gate is written for the cache-aware streaming model; an offline-only
     * model has no stream API at all, so it skips rather than fails */
    mynah_asr_model *probe_m = mynah_asr_load(argv[1]);
    if (!probe_m) return 77;
    mynah_asr_stream *ps = mynah_asr_stream_open(probe_m, "auto", -1);
    const int streaming = ps != NULL;
    mynah_asr_stream_close(ps);
    mynah_asr_free(probe_m);
    if (!streaming) return 77;

    clip cs[5];
    int n_clips = 0;
    for (int i = 0; i < 5; i++) {
        int sr = 0;
        size_t n = 0;
        float *a = mynah_asr_wav_load(CLIPS[i], &n, &sr);
        if (!a || sr != 16000) { free(a); continue; }
        cs[n_clips].audio = a;
        cs[n_clips].n = n;
        cs[n_clips].lang = LANGS[i];
        cs[n_clips].path = CLIPS[i];
        n_clips++;
    }
    if (n_clips < 2) {
        fprintf(stderr, "batch: need at least 2 clips in tests/audio\n");
        return 77;
    }
    printf("stream batch: %d clips,", n_clips);
    for (int i = 0; i < n_clips; i++)
        printf(" %s=%.2fs", cs[i].lang, (double)cs[i].n / 16000.0);
    printf("\n");

    /* `--steptime` (or MYNAH_ASR_STEP_TIME) runs only Gate B, so the table can be
     * re-measured without paying for the identity gates */
    const int steptime = (argc > 2 && strcmp(argv[2], "--steptime") == 0) ||
                         getenv("MYNAH_ASR_STEP_TIME") != NULL;
    if (steptime) {
        gate_steptime(argv[1], cs, n_clips);
        for (int i = 0; i < n_clips; i++) free(cs[i].audio);
        return 0;
    }
    if ((argc > 2 && strcmp(argv[2], "--allocs") == 0) || getenv("MYNAH_ASR_BATCH_ALLOCS")) {
        const int rc = gate_allocs(argv[1], cs, n_clips, 4);
        for (int i = 0; i < n_clips; i++) free(cs[i].audio);
        return rc < 0 ? 2 : (rc == 77 ? 77 : rc);
    }

    int fail = 0;
    /* level 2 first: the strict gate */
    const int bex_b[3] = {2, 4, MAX_B};
    for (int qz = 1; qz >= 0; qz--) {
        const int rc = gate_bitexact(argv[1], qz, cs, n_clips, bex_b, 3);
        if (rc < 0) { for (int i = 0; i < n_clips; i++) free(cs[i].audio); return 2; }
        fail |= rc;
    }

    const int quants[2] = {MYNAH_ASR_QUANT_INT8, MYNAH_ASR_QUANT_F32};
    for (int qi = 0; qi < 2; qi++) {
        const int rc = gate_identity(argv[1], quants[qi], cs, n_clips);
        if (rc < 0) { for (int i = 0; i < n_clips; i++) free(cs[i].audio); return 2; }
        fail |= rc;
    }
    printf("stream batch identity: %s\n", fail ? "DIFFERENT FAIL" : "IDENTICAL OK");
    for (int i = 0; i < n_clips; i++) free(cs[i].audio);
    return fail;
}
