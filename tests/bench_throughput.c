/* Batch throughput: replicates the same wav B times (B = 1,2,4,...,max-batch)
 * and measures the aggregate audio-s/s of the weight-stationary batch
 * transcription (mynah_asr_transcribe_batch). Meant to estimate how many
 * parallel requests a GPU backend (make cuda / metal) can sustain, but it runs
 * on cpu too. It also checks that every batch item produces the B=1 text.
 *
 * With --threads the bench instead simulates N CONCURRENT server-style requests:
 * N pthreads over the same shared model, each transcribing the wav R times
 * (sweeping N = 1,2,4,...,max). On CUDA it measures the gain of the per-thread
 * contexts (independent streams) over the old global mutex.
 *
 * Usage: bench_throughput <model_dir> <wav> [--lang l] [--backend cpu|metal|cuda]
 *                         [--max-batch N] [--runs R] [--threads N]
 * Exit: 0 ok, 1 text mismatch, 2 bad usage, 77 model/wav missing. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/audio.h"
#include "../src/backend.h"
#include "../src/mynah_asr.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* --------------------------- --threads mode (concurrent requests) */
typedef struct {
    mynah_asr_model *m;
    const float *samples;
    size_t ns;
    const char *lang;
    const char *ref;
    int reps;
    int mismatch;
} thr_arg;

static void *thr_worker(void *p) {
    thr_arg *a = (thr_arg *)p;
    for (int r = 0; r < a->reps; r++) {
        char *t = mynah_asr_transcribe(a->m, a->samples, a->ns, a->lang, -1, NULL);
        if (!t || strcmp(t, a->ref) != 0) a->mismatch = 1;
        free(t);
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *wav = NULL, *lang = "auto";
    int max_batch = 32, runs = 2, max_threads = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) lang = argv[++i];
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) mynah_asr_set_backend(argv[++i]);
        else if (strcmp(argv[i], "--max-batch") == 0 && i + 1 < argc) max_batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) max_threads = atoi(argv[++i]);
        else if (!model_dir) model_dir = argv[i];
        else if (!wav) wav = argv[i];
        else { fprintf(stderr, "argomento ignoto: %s\n", argv[i]); return 2; }
    }
    if (!model_dir || !wav) {
        fprintf(stderr, "usage: %s <model_dir> <wav> [--lang l] [--backend b] "
                        "[--max-batch N] [--runs R]\n", argv[0]);
        return 2;
    }
    if (max_batch < 1) max_batch = 1;
    if (runs < 1) runs = 1;

    mynah_asr_model *m = mynah_asr_load(model_dir);
    if (!m) return 77;

    size_t ns;
    int sr;
    float *samples = mynah_asr_wav_load(wav, &ns, &sr);
    if (!samples) { mynah_asr_free(m); return 77; }
    if (sr != 16000) {
        size_t n_rs;
        float *rs = mynah_asr_resample(samples, ns, sr, 16000, &n_rs);
        free(samples);
        if (!rs) { mynah_asr_free(m); return 77; }
        samples = rs; ns = n_rs;
    }
    const double dur = (double)ns / 16000.0;

    /* B=1 reference (and warm-up: weights on device, buffers allocated) */
    char *ref = mynah_asr_transcribe(m, samples, ns, lang, -1, NULL);
    if (!ref) { free(samples); mynah_asr_free(m); return 1; }

    if (max_threads > 0) {
        /* N concurrent server-style requests, each doing `runs` transcriptions */
        printf("# %s | %s %.1fs | lang=%s | reps/thread=%d (concurrent transcribe)\n",
               model_dir, wav, dur, lang, runs);
        printf("%8s %10s %12s %14s\n", "threads", "wall-s", "s-req", "xRT-aggr");
        int rc = 0;
        pthread_t tids[256];
        thr_arg targs[256];
        if (max_threads > 256) max_threads = 256;
        for (int N = 1; N <= max_threads && rc == 0; N *= 2) {
            for (int i = 0; i < N; i++)
                targs[i] = (thr_arg){m, samples, ns, lang, ref, runs, 0};
            double t0 = now_sec();
            for (int i = 0; i < N; i++) pthread_create(&tids[i], NULL, thr_worker, &targs[i]);
            for (int i = 0; i < N; i++) pthread_join(tids[i], NULL);
            double dt = now_sec() - t0;
            for (int i = 0; i < N; i++)
                if (targs[i].mismatch) { fprintf(stderr, "MISMATCH threads=%d\n", N); rc = 1; }
            const double reqs = (double)N * (double)runs;
            printf("%8d %10.3f %12.3f %14.1f\n", N, dt, dt / reqs, reqs * dur / dt);
            fflush(stdout);
        }
        free(ref); free(samples); mynah_asr_free(m);
        return rc;
    }

    const float **sv = malloc((size_t)max_batch * sizeof *sv);
    size_t *nv = malloc((size_t)max_batch * sizeof *nv);
    const char **lv = malloc((size_t)max_batch * sizeof *lv);
    char **out = malloc((size_t)max_batch * sizeof *out);
    if (!sv || !nv || !lv || !out) return 1;
    for (int b = 0; b < max_batch; b++) { sv[b] = samples; nv[b] = ns; lv[b] = lang; }

    printf("# %s | %s %.1fs | lang=%s | runs=%d (best)\n", model_dir, wav, dur, lang, runs);
    printf("%6s %10s %12s %14s\n", "batch", "wall-s", "s-item", "xRT-aggr");
    int rc = 0;
    for (int B = 1; B <= max_batch && rc == 0; B *= 2) {
        double best = 1e30;
        for (int r = 0; r < runs; r++) {
            for (int b = 0; b < B; b++) out[b] = NULL;
            double t0 = now_sec();
            if (mynah_asr_transcribe_batch(m, sv, nv, B, lv, -1, out, NULL) != 0) {
                fprintf(stderr, "FAIL: transcribe_batch B=%d\n", B);
                rc = 1;
                break;
            }
            double dt = now_sec() - t0;
            if (dt < best) best = dt;
            for (int b = 0; b < B; b++) {
                if (out[b] && strcmp(out[b], ref) != 0) {
                    fprintf(stderr, "MISMATCH B=%d item %d:\n  ref:   %s\n  item:  %s\n",
                            B, b, ref, out[b]);
                    rc = 1;
                }
                free(out[b]);
            }
        }
        if (rc == 0)
            printf("%6d %10.3f %12.3f %14.1f\n",
                   B, best, best / B, (double)B * dur / best);
        fflush(stdout);
    }

    free(ref); free(samples);
    free(sv); free(nv); free(lv); free(out);
    mynah_asr_free(m);
    return rc;
}
