/* tests/test_cuda_aed.c — the GPU AED engine (gpu/cuda/aed.cu) against the
 * library, on an AED pack (Canary). .work/canary-180m-l4.md section 5.6.
 *
 *   A  parity: every clip's final transcript from the cuda offline engine is
 *      byte-identical to the cpu offline engine's (the library alone, the same
 *      mynah_asr_transcribe_batch the CPU server makes). With --precision bf16
 *      the bytes may differ: the gate is then the CER against the f32 library
 *      (mean <= --max-cer), and every differing clip is printed with its CER.
 *   B  batch identity (PLAN.md durable contract 4): each clip alone, all clips
 *      in one call, and all clips in reversed order give the same bytes on the
 *      GPU. Never relaxed, at any precision.
 *   C  the single path: a verbose (word timestamps) request on an AED pack takes
 *      the library's single path with the timestamp prompt; its text through the
 *      offload equals the library's (f32), and its word count too.
 *
 * Clips: the committed FLEURS clips (samples/en, samples/fr), each with its
 * language, plus two translation requests (en>de, and fr>en, which the 180M
 * answers with an immediate EOS: an empty transcript is a case too).
 *
 * Usage: test_cuda_aed <model_dir> [--precision f32|bf16] [--aed-decoder gpu|host]
 *                      [--device N] [--max-cer X] [wav:lang ...]
 * Exit: 0 ok, 1 fail, 77 skip (no pack, or no CUDA device / not compiled). */
#include "asr_offline.h"

#include "audio.h"
#include "mynah_asr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CLIPS 32

static const char *const DEFAULT_CLIPS[] = {
    "samples/en/fleurs_1521.wav:en", "samples/en/fleurs_1534.wav:en", "samples/en/fleurs_long.wav:en",
    "samples/fr/fleurs_1521.wav:fr", "samples/fr/fleurs_1534.wav:fr",
    "samples/en/fleurs_1534.wav:en>de", "samples/fr/fleurs_1521.wav:fr>en",
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

/* UTF-8 -> code points (invalid bytes count as one each) */
static int utf8_cps(const char *s, unsigned *out, int cap) {
    int n = 0;
    const unsigned char *p = (const unsigned char *)s;
    while (*p && n < cap) {
        unsigned c = *p;
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        unsigned v = len == 1 ? c : c & (0x7Fu >> len);
        for (int i = 1; i < len && p[i]; i++) v = (v << 6) | (p[i] & 0x3Fu);
        for (int i = 0; i < len && *p; i++) p++;
        out[n++] = v;
    }
    return n;
}

/* character error rate of hyp against ref (Levenshtein over code points) */
static double cer(const char *ref, const char *hyp) {
    static unsigned a[20000], b[20000];
    static int row0[20001], row1[20001];
    const int n = utf8_cps(ref, a, 20000), m = utf8_cps(hyp, b, 20000);
    if (n == 0) return m == 0 ? 0.0 : 1.0;
    for (int j = 0; j <= m; j++) row0[j] = j;
    for (int i = 1; i <= n; i++) {
        row1[0] = i;
        for (int j = 1; j <= m; j++) {
            int best = row0[j - 1] + (a[i - 1] != b[j - 1]);
            if (row0[j] + 1 < best) best = row0[j] + 1;
            if (row1[j - 1] + 1 < best) best = row1[j - 1] + 1;
            row1[j] = best;
        }
        memcpy(row0, row1, (size_t)(m + 1) * sizeof(int));
    }
    return (double)row0[m] / (double)n;
}

typedef struct {
    char path[512], lang[24];
    float *pcm;
    size_t n;
} clip;

static int run(asr_offline *o, int n, clip *const *cs, char **texts) {
    const float *pcm[MAX_CLIPS];
    size_t ns[MAX_CLIPS];
    const char *langs[MAX_CLIPS];
    for (int i = 0; i < n; i++) { pcm[i] = cs[i]->pcm; ns[i] = cs[i]->n; langs[i] = cs[i]->lang; }
    return asr_offline_transcribe(o, n, pcm, ns, langs, texts, NULL, NULL, NULL);
}

static void free_texts(char **t, int n) {
    for (int i = 0; i < n; i++) { free(t[i]); t[i] = NULL; }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_dir> [--precision f32|bf16] [--aed-decoder gpu|host] "
                        "[--device N] [--max-cer X] [wav:lang ...]\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1], *prec = "f32", *dec = "gpu";
    int device = 0;
    double max_cer = 0.02;
    const char *specs[MAX_CLIPS];
    int n = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) prec = argv[++i];
        else if (strcmp(argv[i], "--aed-decoder") == 0 && i + 1 < argc) dec = argv[++i];
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-cer") == 0 && i + 1 < argc) max_cer = atof(argv[++i]);
        else if (n < MAX_CLIPS) specs[n++] = argv[i];
    }
    if (n == 0)
        for (size_t i = 0; i < sizeof(DEFAULT_CLIPS) / sizeof(DEFAULT_CLIPS[0]); i++) specs[n++] = DEFAULT_CLIPS[i];
    char probe[1024];
    snprintf(probe, sizeof(probe), "%s/mynah.json", dir);
    FILE *pf = fopen(probe, "rb");
    if (!pf) { printf("SKIP: no pack at %s\n", dir); return 77; }
    fclose(pf);

    static clip clips[MAX_CLIPS];
    clip *all[MAX_CLIPS], *rev[MAX_CLIPS];
    for (int i = 0; i < n; i++) {
        const char *colon = strrchr(specs[i], ':');
        if (!colon) { fprintf(stderr, "clip spec '%s' is not wav:lang\n", specs[i]); return 2; }
        snprintf(clips[i].path, sizeof(clips[i].path), "%.*s", (int)(colon - specs[i]), specs[i]);
        snprintf(clips[i].lang, sizeof(clips[i].lang), "%s", colon + 1);
        int sr = 0;
        clips[i].pcm = mynah_asr_wav_load(clips[i].path, &clips[i].n, &sr);
        if (!clips[i].pcm) { printf("SKIP: cannot read %s\n", clips[i].path); return 77; }
        if (sr != 16000) {
            size_t n2 = 0;
            float *rs = mynah_asr_resample(clips[i].pcm, clips[i].n, sr, 16000, &n2);
            free(clips[i].pcm);
            clips[i].pcm = rs;
            clips[i].n = n2;
        }
        all[i] = &clips[i];
        rev[n - 1 - i] = &clips[i];
    }

    char err[512] = "";
    char *ref[MAX_CLIPS] = {0}, *ref_w = NULL;
    int ref_nw = 0;
    {   /* the reference: the library alone, f32 */
        asr_offline_cfg cc = {.model_dir = dir, .engine = "cpu", .max_items = n};
        asr_offline *cpu = asr_offline_open(&cc, err, sizeof(err));
        if (!cpu) { printf("SKIP: the cpu engine did not open: %s\n", err); return 77; }
        const double t0 = now_ms();
        if (run(cpu, n, all, ref) != 0) { printf("FAIL: the reference failed on some clip\n"); return 1; }
        printf("reference (library f32, cpu): %d clips in %.0f ms\n", n, now_ms() - t0);
        const float *p0 = clips[0].pcm;
        const char *l0 = clips[0].lang;
        mynah_asr_word *w = NULL;
        if (asr_offline_transcribe(cpu, 1, &p0, &clips[0].n, &l0, &ref_w, NULL, &w, &ref_nw) != 0) {
            printf("FAIL: the reference verbose request failed\n");
            return 1;
        }
        mynah_asr_words_free(w, ref_nw);
        asr_offline_close(cpu);
    }

    asr_offline_cfg gc = {.model_dir = dir, .engine = "cuda", .device = device, .max_items = n,
                          .precision = prec, .gemm = strcmp(prec, "bf16") == 0 ? "own-tc" : "own",
                          .decoder = dec};
    asr_offline *gpu = asr_offline_open(&gc, err, sizeof(err));
    if (!gpu) {
        if (strstr(err, "no CUDA device") || strstr(err, "not compiled")) { printf("SKIP: %s\n", err); return 77; }
        printf("FAIL: the cuda engine did not open: %s\n", err);
        return 1;
    }
    asr_offline_facts f;
    asr_offline_get_facts(gpu, &f);
    printf("engine: %s device=%s precision=%s gemm=%s decoder=%s max_items=%d\n", f.name, f.device,
           f.precision, f.gemm, f.decoder, f.max_items);
    int fail = 0;

    /* B first: all together, reversed, each alone */
    char *tall[MAX_CLIPS] = {0}, *trev[MAX_CLIPS] = {0}, *tone[MAX_CLIPS] = {0};
    double t0 = now_ms();
    if (run(gpu, n, all, tall) != 0) { printf("FAIL: the batched cuda call failed: %s\n", asr_offline_error(gpu)); return 1; }
    const double ms_all = now_ms() - t0;
    if (run(gpu, n, rev, trev) != 0) { printf("FAIL: the reversed cuda call failed\n"); return 1; }
    t0 = now_ms();
    for (int i = 0; i < n; i++)
        if (run(gpu, 1, &all[i], &tone[i]) != 0) { printf("FAIL: clip %s alone failed\n", clips[i].path); return 1; }
    const double ms_one = now_ms() - t0;
    int b_ok = 0;
    for (int i = 0; i < n; i++) {
        const int same = strcmp(tall[i], tone[i]) == 0 && strcmp(trev[n - 1 - i], tone[i]) == 0;
        b_ok += same;
        if (!same)
            printf("  B DIFF %s [%s]\n    alone   : %s\n    batched : %s\n    reversed: %s\n", clips[i].path,
                   clips[i].lang, tone[i], tall[i], trev[n - 1 - i]);
    }
    printf("B batch identity: %d/%d identical (batched %.0f ms, one by one %.0f ms)\n", b_ok, n, ms_all, ms_one);
    if (b_ok != n) fail = 1;

    /* A: against the library */
    int a_same = 0;
    double cer_sum = 0.0;
    for (int i = 0; i < n; i++) {
        const double c = cer(ref[i], tall[i]);
        cer_sum += c;
        if (strcmp(ref[i], tall[i]) == 0) a_same++;
        else printf("  A DIFF %s [%s] CER %.4f\n    library: %s\n    cuda   : %s\n", clips[i].path, clips[i].lang,
                    c, ref[i], tall[i]);
    }
    const double cer_mean = cer_sum / (double)n;
    const int f32 = strcmp(prec, "f32") == 0;
    printf("A parity with the library (%s): %d/%d byte-identical, mean CER %.4f%s\n", prec, a_same, n, cer_mean,
           f32 ? " (gate: all identical)" : "");
    if (f32 ? a_same != n : cer_mean > max_cer) fail = 1;

    /* C: the single path through the offload (word timestamps) */
    {
        char *tw = NULL;
        mynah_asr_word *w = NULL;
        int nw = 0;
        const float *p0 = clips[0].pcm;
        const char *l0 = clips[0].lang;
        if (asr_offline_transcribe(gpu, 1, &p0, &clips[0].n, &l0, &tw, NULL, &w, &nw) != 0 || !tw) {
            printf("FAIL: C the verbose request failed on the GPU\n");
            fail = 1;
        } else {
            const int same = strcmp(tw, ref_w) == 0 && nw == ref_nw;
            printf("C single path (words): %s (words %d vs %d)\n", same ? "identical" : "DIFFERENT", nw, ref_nw);
            if (!same) {
                printf("    library: %s\n    cuda   : %s\n", ref_w, tw);
                if (f32) fail = 1;
            }
        }
        free(tw);
        mynah_asr_words_free(w, nw);
    }

    asr_offline_stats s;
    asr_offline_get_stats(gpu, &s);
    printf("stats: enc_calls=%lu enc_rows=%lu dec_calls=%lu dec_steps=%lu dec_tokens=%lu host_ss_ms=%.0f "
           "enc_ms=%.0f dec_ms=%.0f\n", s.enc_calls, s.enc_rows, s.dec_calls, s.dec_steps, s.dec_tokens,
           s.host_ss_ms, s.enc_ms, s.dec_ms);
    free_texts(tall, n); free_texts(trev, n); free_texts(tone, n); free_texts(ref, n);
    free(ref_w);
    asr_offline_close(gpu);
    for (int i = 0; i < n; i++) free(clips[i].pcm);
    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail;
}
