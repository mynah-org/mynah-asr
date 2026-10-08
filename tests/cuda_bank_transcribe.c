/* tests/cuda_bank_transcribe.c — a whole bank through the cuda engine, offline.
 *
 * The quality evidence of a numerical change (ENGINEERING.md §9) without a
 * real-time load run: every clip of a list is streamed through the engine in
 * cohorts of --cohort clips, fed as fast as the engine takes it, finalised at
 * its end, and the transcripts are written as {clip: text} -- the format
 * gpu/tools/transcript_ab.py compares and gpu_qualify.sh writes to
 * reference.json. Contract 4 makes the cohort size irrelevant to a transcript,
 * so two runs at different --cohort must produce identical files: that is the
 * bank-wide batch-identity check (`cmp` the two outputs).
 *
 * usage: tests/cuda_bank_transcribe <model_dir> --list clips.txt --json out.json
 *            [--cohort 64] [--gemm own|splitk|own-tc] [--precision f32|bf16]
 *            [--lookahead 3] [--lang auto] [--profile 1]
 * --profile 1: the engine's per-stage CUDA-event profile (DIAGNOSTIC: it
 * adds events to the stream), printed per pass at the end with rows per pass.
 * exit 0 = every clip transcribed; 1 = an engine error; 2 = usage/input. */
#include "../gpu/asr_engine.h"

#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXCLIPS 4096
#define MAXC 256

static void cat_text(char **t, const char *s) {
    if (!s || !s[0]) return;
    const size_t a = *t ? strlen(*t) : 0, b = strlen(s);
    char *n = realloc(*t, a + b + 1);
    if (!n) return;
    memcpy(n + a, s, b + 1);
    *t = n;
}

static int run_cohort(asr_engine *e, float **pcm, size_t *ns, int n, const char *lang, int la, char **out) {
    asr_step_req reqs[MAXC];
    asr_step_out outs[MAXC];
    size_t off[MAXC];
    int done[MAXC];
    for (int i = 0; i < n; i++) {
        if (asr_engine_slot_reset(e, i, lang, la) != 0) { fprintf(stderr, "slot reset %d failed\n", i); return -1; }
        off[i] = 0; done[i] = 0; out[i] = NULL;
    }
    for (;;) {
        int nreq = 0, alive = 0;
        for (int i = 0; i < n; i++) {
            if (done[i]) continue;
            alive = 1;
            const size_t need = asr_engine_slot_need_samples(e, i);
            if (off[i] < ns[i] && need > 0) {
                size_t take = need;
                if (off[i] + take > ns[i]) take = ns[i] - off[i];
                if (asr_engine_slot_feed(e, i, pcm[i] + off[i], take) != 0) return -1;
                off[i] += take;
            }
            const int fin = off[i] >= ns[i];
            if (asr_engine_slot_ready(e, i) || fin) { reqs[nreq].slot = i; reqs[nreq].finalize = fin; nreq++; }
        }
        if (!alive) break;
        if (nreq == 0) continue;
        if (asr_engine_step(e, reqs, nreq, outs) != 0) { fprintf(stderr, "step: %s\n", asr_engine_error(e)); return -1; }
        for (int k = 0; k < nreq; k++) {
            cat_text(&out[reqs[k].slot], outs[k].text);
            if (outs[k].finished) done[reqs[k].slot] = 1;
        }
    }
    return 0;
}

static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (; s && *s; s++) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model_dir> --list clips.txt --json out.json [...]\n", argv[0]); return 2; }
    const char *model = argv[1], *list = NULL, *json = NULL, *gemm = "own", *precision = "f32", *lang = "auto";
    int cohort = 64, la = 3, prof = 0;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--list")) list = argv[i + 1];
        else if (!strcmp(argv[i], "--json")) json = argv[i + 1];
        else if (!strcmp(argv[i], "--gemm")) gemm = argv[i + 1];
        else if (!strcmp(argv[i], "--precision")) precision = argv[i + 1];
        else if (!strcmp(argv[i], "--cohort")) cohort = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--lookahead")) la = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--lang")) lang = argv[i + 1];
        else if (!strcmp(argv[i], "--profile")) prof = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!list || !json || cohort < 1 || cohort > MAXC) { fprintf(stderr, "--list and --json are required; 1 <= --cohort <= %d\n", MAXC); return 2; }
    static char *names[MAXCLIPS];
    int nclip = 0;
    FILE *lf = fopen(list, "r");
    if (!lf) { perror(list); return 2; }
    char line[4096];
    while (nclip < MAXCLIPS && fgets(line, sizeof(line), lf)) {
        line[strcspn(line, "\r\n")] = 0;
        if (line[0]) names[nclip++] = strdup(line);
    }
    fclose(lf);

    char err[512] = "";
    asr_engine_cfg cfg = {.model_dir = model, .cap = cohort, .device = 0, .precision = precision, .gemm = gemm, .profile = prof};
    asr_engine *e = asr_engine_open_cuda(&cfg, err, sizeof(err));
    if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
    asr_engine_facts f; asr_engine_get_facts(e, &f);
    fprintf(stderr, "cuda_bank_transcribe on %s, precision=%s gemm=%s, %d clip(s), cohort %d, lookahead %d, lang %s\n",
            f.device, f.precision, f.gemm, nclip, cohort, la, lang);

    FILE *jf = fopen(json, "w");
    if (!jf) { perror(json); return 2; }
    fprintf(jf, "{\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double audio_s = 0.0;
    for (int b = 0; b < nclip; b += cohort) {
        const int n = nclip - b < cohort ? nclip - b : cohort;
        float *pcm[MAXC]; size_t ns[MAXC]; char *out[MAXC];
        for (int i = 0; i < n; i++) {
            int sr = 0;
            pcm[i] = mynah_asr_wav_load(names[b + i], &ns[i], &sr);
            if (!pcm[i] || sr != 16000) { fprintf(stderr, "cannot load %s (16 kHz mono)\n", names[b + i]); return 2; }
            audio_s += (double)ns[i] / 16000.0;
        }
        if (run_cohort(e, pcm, ns, n, lang, la, out) != 0) return 1;
        for (int i = 0; i < n; i++) {
            fprintf(jf, " ");
            json_str(jf, names[b + i]);
            fprintf(jf, ": ");
            json_str(jf, out[i] ? out[i] : "");
            fprintf(jf, "%s\n", b + i + 1 < nclip ? "," : "");
            free(out[i]); free(pcm[i]);
        }
    }
    fprintf(jf, "}\n");
    fclose(jf);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double wall = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    asr_engine_stats st; asr_engine_get_stats(e, &st);
    fprintf(stderr, "done: %d clips, %.0f s of audio in %.1f s wall (%.1fx), %lu steps, step wall mean %.2f ms\n",
            nclip, audio_s, wall, audio_s / wall, st.steps, st.steps ? st.step_wall_ms_sum / (double)st.steps : 0.0);
    if (st.prof_passes > 0) {
        const double np = (double)st.prof_passes;
        double tot = 0.0, dec = 0.0;
        for (int i = 0; i < ASR_PROF_STAGES; i++) {
            tot += st.prof_ms[i];
            if (!strncmp(ASR_PROF_NAME[i], "dec", 3) || !strcmp(ASR_PROF_NAME[i], "d2h")) dec += st.prof_ms[i];
        }
        fprintf(stderr, "profile: %lu passes, %.1f rows/pass, %.1f lanes/pass, device %.2f ms/pass (encoder+projector %.2f, decoder+d2h %.2f)\n  ",
                st.prof_passes, (double)st.rows / np, (double)st.lanes / np, tot / np, (tot - dec) / np, dec / np);
        for (int i = 0; i < ASR_PROF_STAGES; i++) fprintf(stderr, " %s=%.2f", ASR_PROF_NAME[i], st.prof_ms[i] / np);
        fprintf(stderr, "\n");
    }
    asr_engine_close(e);
    return 0;
}
