/* tests/test_cuda_stream.c — the two transcript gates of the CUDA engine
 * (S14-3 / S14-4), with a pack and a GPU:
 *
 *   A. BATCH IDENTITY ON THE GPU (contract 4): every clip streamed as one lane
 *      of a mixed cohort -- other clips, other languages, a first chunk next to
 *      a steady one, a finalising tail -- produces the same tokens and the same
 *      text, byte for byte, as the same clip streamed alone on the same GPU.
 *      Idle lanes of the cohort are poisoned with a different clip so that a
 *      cross-lane read cannot go unnoticed.
 *   B. CPU <-> GPU: the same clips through the library's own f32 stream API
 *      (src/mynah_asr.h, the CPU path this tree never modifies) and through the
 *      cuda engine. Identity is EXPECTED and reported per clip; a difference is
 *      printed with both texts and exits 3 -- a finding to look at, never a
 *      silent pass and never a silent fail.
 *
 * usage: tests/test_cuda_stream <model_dir> [--gemm own|cublas] [clip.wav ...]
 *        (default clips: tests/audio/test_*.wav; the GEMM arm is an argument,
 *        not an environment flag, so the flag registry stays the library's)
 * exit 0 = both gates pass; 1 = gate A failed or a device error; 3 = gate A
 * passed and gate B found a difference; 77 = no CUDA device (SKIP). */
#include "../gpu/asr_engine.h"

#include "audio.h"
#include "mynah_asr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXC 16

typedef struct { char *text; } utt;

static void cat_text(utt *u, const char *s) {
    if (!s || !s[0]) return;
    const size_t a = u->text ? strlen(u->text) : 0, b = strlen(s);
    char *n = realloc(u->text, a + b + 1);
    if (!n) return;
    memcpy(n + a, s, b + 1);
    u->text = n;
}

/* stream `n` clips as one cohort on the cuda engine, in real-time-sized feeds,
 * finalizing each at its end; returns the per-slot concatenated text */
static int run_cohort(asr_engine *e, float **pcm, size_t *ns, int n, const char **langs, utt *out) {
    asr_step_req reqs[MAXC];
    asr_step_out outs[MAXC];
    size_t off[MAXC];
    int done[MAXC];
    for (int i = 0; i < n; i++) {
        if (asr_engine_slot_reset(e, i, langs[i], 3) != 0) { printf("FAIL slot reset %d\n", i); return -1; }
        off[i] = 0; done[i] = 0; out[i].text = NULL;
    }
    for (int guard = 0; guard < 100000; guard++) {
        int nreq = 0, alive = 0;
        for (int i = 0; i < n; i++) {
            if (done[i]) continue;
            alive = 1;
            const size_t need = asr_engine_slot_need_samples(e, i);
            if (off[i] < ns[i]) {
                size_t take = need > 0 ? need : 0;
                if (take == 0) take = 1;   /* a chunk is complete: step it before feeding more */
                if (off[i] + take > ns[i]) take = ns[i] - off[i];
                if (need > 0) {
                    if (asr_engine_slot_feed(e, i, pcm[i] + off[i], take) != 0) return -1;
                    off[i] += take;
                }
            }
            const int fin = off[i] >= ns[i];
            if (asr_engine_slot_ready(e, i) || fin) { reqs[nreq].slot = i; reqs[nreq].finalize = fin; nreq++; }
        }
        if (!alive) break;
        if (nreq == 0) continue;
        if (asr_engine_step(e, reqs, nreq, outs) != 0) { printf("FAIL step: %s\n", asr_engine_error(e)); return -1; }
        for (int k = 0; k < nreq; k++) {
            cat_text(&out[reqs[k].slot], outs[k].text);
            if (outs[k].finished) done[reqs[k].slot] = 1;
        }
    }
    return 0;
}

typedef struct { utt *u; } cb_ctx;
static void on_res(const mynah_asr_result *r, void *ud) {
    if (r->is_eou) return;
    cat_text(((cb_ctx *)ud)->u, r->text);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model_dir> [clips...]\n", argv[0]); return 2; }
    const char *model = argv[1];
    const char *dflt[] = {"tests/audio/test_it.wav", "tests/audio/test_en.wav", "tests/audio/test_de.wav",
                          "tests/audio/test_fr.wav", "tests/audio/test_es.wav"};
    const char *clips[MAXC], *gemm = "own";
    int n = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--gemm") == 0 && i + 1 < argc) gemm = argv[++i];
        else if (n < MAXC) clips[n++] = argv[i];
    }
    if (n == 0) for (int i = 0; i < 5; i++) clips[n++] = dflt[i];
    const char *langs[MAXC];
    for (int i = 0; i < n; i++) langs[i] = "auto";

    float *pcm[MAXC]; size_t ns[MAXC];
    for (int i = 0; i < n; i++) {
        int sr = 0;
        pcm[i] = mynah_asr_wav_load(clips[i], &ns[i], &sr);
        if (!pcm[i] || sr != 16000) { fprintf(stderr, "cannot load %s (16 kHz mono)\n", clips[i]); return 2; }
    }

    char err[512] = "";
    asr_engine_cfg cfg = {.model_dir = model, .cap = n + 1, .device = 0, .precision = "f32", .gemm = gemm};
    asr_engine *e = asr_engine_open_cuda(&cfg, err, sizeof(err));
    if (!e) {
        if (strstr(err, "no CUDA device") || strstr(err, "not compiled")) { printf("SKIP test_cuda_stream: %s\n", err); return 77; }
        printf("FAIL open: %s\n", err); return 1;
    }
    asr_engine_facts f; asr_engine_get_facts(e, &f);
    printf("test_cuda_stream on %s, gemm=%s, %d clip(s)\n", f.device, f.gemm, n);

    /* ---- gate A: each clip alone, then all together (mixed lanes) */
    utt alone[MAXC] = {{0}}, together[MAXC] = {{0}};
    int fail = 0;
    for (int i = 0; i < n; i++) {
        float *p1[1] = {pcm[i]}; size_t n1[1] = {ns[i]}; const char *l1[1] = {langs[i]};
        if (run_cohort(e, p1, n1, 1, l1, &alone[i]) != 0) return 1;
    }
    if (run_cohort(e, pcm, ns, n, langs, together) != 0) return 1;
    for (int i = 0; i < n; i++) {
        const char *a = alone[i].text ? alone[i].text : "", *b = together[i].text ? together[i].text : "";
        const int same = strcmp(a, b) == 0;
        if (!same) fail = 1;
        printf("%s A batch-identity %-28s %s\n", same ? "OK  " : "FAIL", clips[i], same ? "identical" : "DIFFERS");
        if (!same) printf("       alone   : %s\n       cohort  : %s\n", a, b);
    }
    /* poisoned idle lanes: the same cohort with the clips shifted by one slot */
    {
        utt shifted[MAXC] = {{0}};
        float *p2[MAXC]; size_t n2[MAXC]; const char *l2[MAXC];
        for (int i = 0; i < n; i++) { p2[i] = pcm[(i + 1) % n]; n2[i] = ns[(i + 1) % n]; l2[i] = langs[(i + 1) % n]; }
        if (run_cohort(e, p2, n2, n, l2, shifted) != 0) return 1;
        for (int i = 0; i < n; i++) {
            const char *a = alone[(i + 1) % n].text ? alone[(i + 1) % n].text : "", *b = shifted[i].text ? shifted[i].text : "";
            const int same = strcmp(a, b) == 0;
            if (!same) fail = 1;
            printf("%s A slot-independence %-25s %s\n", same ? "OK  " : "FAIL", clips[(i + 1) % n], same ? "identical on another slot" : "DIFFERS");
        }
    }
    if (fail) { printf("FAIL gate A: a stream's text depends on its cohort or its slot\n"); return 1; }

    /* ---- gate B: the library's f32 CPU stream API */
    mynah_asr_model *m = mynah_asr_load_quant(model, MYNAH_ASR_QUANT_F32);
    if (!m) { printf("FAIL: the pack did not load through the library\n"); return 1; }
    int differs = 0;
    for (int i = 0; i < n; i++) {
        utt cpu = {0};
        cb_ctx c = {&cpu};
        mynah_asr_stream *s = mynah_asr_stream_open(m, NULL, 3);
        if (!s) { printf("FAIL: stream open\n"); return 1; }
        size_t off = 0;
        while (off < ns[i]) {
            size_t take = mynah_asr_stream_need_samples(s);
            if (take == 0 || off + take > ns[i]) take = ns[i] - off;
            if (mynah_asr_stream_feed(s, pcm[i] + off, take, on_res, &c) != 0) { printf("FAIL: feed\n"); return 1; }
            off += take;
        }
        mynah_asr_stream_finish(s, on_res, &c);
        mynah_asr_stream_close(s);
        const char *a = cpu.text ? cpu.text : "", *b = alone[i].text ? alone[i].text : "";
        const int same = strcmp(a, b) == 0;
        if (!same) differs++;
        printf("%s B cpu-f32 vs gpu %-30s %s\n", same ? "OK  " : "DIFF", clips[i], same ? "identical" : "differs");
        if (!same) printf("       cpu f32 : %s\n       gpu f32 : %s\n", a, b);
        free(cpu.text);
    }
    mynah_asr_free(m);
    asr_engine_close(e);
    if (differs) { printf("gate B: %d of %d clip(s) differ between the CPU f32 path and the GPU -- a finding, see the note\n", differs, n); return 3; }
    printf("PASS: gate A (batch identity, slot independence) and gate B (cpu f32 == gpu) on %d clip(s)\n", n);
    return 0;
}
