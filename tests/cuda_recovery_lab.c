/* tests/cuda_recovery_lab.c — the empty-turn recovery lab (S18) through a GPU
 * server engine (.work/empty-turn-recovery.md).
 *
 * Same items, same arms and the same JSONL as tests/recovery_lab (the CPU
 * library path), so tools/eval/recovery_lab.py reports both unchanged. Every
 * decode is one utterance on one engine slot -- the previous turn alone, the
 * current turn (primary), the current turn with "auto" (retry) and each replay
 * [last N ms of the previous turn][current turn] -- run in cohorts of --cohort
 * slots. Contract 4 of the engine makes the cohort irrelevant to a transcript.
 * Emission frames come from asr_engine_slot_tokens (cfg.record_frames).
 *
 * usage: tests/cuda_recovery_lab <model_dir> <items.tsv> [--ctx-ms 480,960,...]
 *            [--cohort 64] [--engine cuda|cpu] [--precision f32|bf16]
 *            [--gemm own|...] [--weights f32|int8] [--lookahead 3] [--dump-tokens 1]
 * --dump-tokens 1: every token with its emission frame (tests/recovery_lab.c)
 * items.tsv: id  lang  ctx_wav  cur_wav  cur_start_s  cur_end_s  (tests/recovery_lab.c)
 * exit 0 ok; 1 engine error; 2 usage/input. */
#include "../gpu/asr_engine.h"

#include "audio.h"
#include "recovery.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXC 256
#define MAX_CTX 8
#define BOUNDARY_SHOW 6
#define FRAME_SAMPLES_DEFAULT 1280

typedef struct {
    const char *lang;
    float *pcm;              /* owned */
    size_t n;
    /* results */
    char *text, lang_out[16];
    int *tok, *fr, ntok;
} job;

typedef struct {
    char id[128], lang[32], ctx_wav[1024], cur_wav[1024];
    double t0, t1;
    int ctx_job, primary, retry, replay[MAX_CTX];
    int keep_from[MAX_CTX], ctx_ms_real[MAX_CTX];
} item;

/* A JSON string. A vocabulary piece can be part of a multi-byte character
 * (byte-fallback pieces): bytes that do not form valid UTF-8 are written as
 * the text "<0xNN>", so the line stays valid JSON and the byte stays visible. */
static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p;) {
        const int len = p[0] < 0x80 ? 1 : (p[0] & 0xE0) == 0xC0 ? 2 : (p[0] & 0xF0) == 0xE0 ? 3
                      : (p[0] & 0xF8) == 0xF0 ? 4 : 0;
        int ok = len > 0;
        for (int k = 1; ok && k < len; k++) ok = (p[k] & 0xC0) == 0x80;
        if (!ok) { fprintf(f, "<0x%02X>", p[0]); p++; continue; }
        if (len > 1) { fwrite(p, 1, (size_t)len, f); p += len; continue; }
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p < 0x20) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
        p++;
    }
    fputc('"', f);
}

static void dump_tokens(const mynah_asr_tokenizer *tk, const int *tok, const int *fr, int n) {
    putchar('[');
    for (int i = 0; i < n; i++) {
        printf("%s[%d,", i ? "," : "", fr[i]);
        json_str(stdout, (tok[i] >= 0 && tok[i] < tk->n_pieces) ? tk->pieces[tok[i]] : "?");
        putchar(']');
    }
    putchar(']');
}

static float *load16k(const char *path, size_t *n) {
    int sr = 0;
    float *x = mynah_asr_wav_load(path, n, &sr);
    if (!x || sr == 16000) return x;
    size_t m = 0;
    float *y = mynah_asr_resample(x, *n, sr, 16000, &m);
    free(x);
    *n = m;
    return y;
}

/* One cohort: jobs[ids[0..n)] on slots 0..n-1, fed as fast as the engine
 * takes them, finalised at their end; results copied out before any reset. */
static int run_cohort(asr_engine *e, job *jobs, const int *ids, int n, int la) {
    asr_step_req reqs[MAXC];
    asr_step_out outs[MAXC];
    size_t off[MAXC];
    int done[MAXC];
    for (int i = 0; i < n; i++) {
        if (asr_engine_slot_reset(e, i, jobs[ids[i]].lang, la) != 0) {
            fprintf(stderr, "slot reset %d (%s) failed\n", i, jobs[ids[i]].lang);
            return -1;
        }
        off[i] = 0; done[i] = 0;
    }
    for (;;) {
        int nreq = 0, alive = 0;
        for (int i = 0; i < n; i++) {
            if (done[i]) continue;
            alive = 1;
            job *j = &jobs[ids[i]];
            const size_t need = asr_engine_slot_need_samples(e, i);
            if (off[i] < j->n && need > 0) {
                size_t take = need;
                if (off[i] + take > j->n) take = j->n - off[i];
                if (asr_engine_slot_feed(e, i, j->pcm + off[i], take) != 0) return -1;
                off[i] += take;
            }
            const int fin = off[i] >= j->n;
            if (asr_engine_slot_ready(e, i) || fin) { reqs[nreq].slot = i; reqs[nreq].finalize = fin; nreq++; }
        }
        if (!alive) break;
        if (nreq == 0) continue;
        if (asr_engine_step(e, reqs, nreq, outs) != 0) { fprintf(stderr, "step: %s\n", asr_engine_error(e)); return -1; }
        for (int k = 0; k < nreq; k++)
            if (outs[k].finished) done[reqs[k].slot] = 1;
    }
    for (int i = 0; i < n; i++) {
        job *j = &jobs[ids[i]];
        j->text = strdup(asr_engine_slot_text(e, i));
        snprintf(j->lang_out, sizeof(j->lang_out), "%s", asr_engine_slot_lang(e, i));
        const int *t = NULL, *f = NULL;
        const int nt = asr_engine_slot_tokens(e, i, &t, &f);
        if (nt < 0) { fprintf(stderr, "the engine does not report emission frames\n"); return -1; }
        j->ntok = nt;
        j->tok = malloc((size_t)(nt > 0 ? nt : 1) * sizeof(int));
        j->fr = malloc((size_t)(nt > 0 ? nt : 1) * sizeof(int));
        if (nt > 0) { memcpy(j->tok, t, (size_t)nt * sizeof(int)); memcpy(j->fr, f, (size_t)nt * sizeof(int)); }
        free(j->pcm);
        j->pcm = NULL;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir> <items.tsv> [--ctx-ms 480,960] [--cohort 64] "
                        "[--engine cuda|cpu] [--precision f32] [--gemm own] [--weights f32] [--lookahead 3]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1], *items_path = argv[2], *engine = "cuda", *precision = "f32",
               *gemm = "own", *weights = NULL;
    int dump = 0, cohort = 64, la = 3, ctx_ms[MAX_CTX] = {500, 1000, 2000, 3000}, n_ctx = 4;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--ctx-ms")) {
            n_ctx = 0;
            for (char *t = strtok(argv[i + 1], ","); t && n_ctx < MAX_CTX; t = strtok(NULL, ",")) ctx_ms[n_ctx++] = atoi(t);
        } else if (!strcmp(argv[i], "--cohort")) cohort = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--engine")) engine = argv[i + 1];
        else if (!strcmp(argv[i], "--precision")) precision = argv[i + 1];
        else if (!strcmp(argv[i], "--gemm")) gemm = argv[i + 1];
        else if (!strcmp(argv[i], "--weights")) weights = argv[i + 1];
        else if (!strcmp(argv[i], "--lookahead")) la = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--dump-tokens")) dump = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (cohort < 1 || cohort > MAXC) { fprintf(stderr, "1 <= --cohort <= %d\n", MAXC); return 2; }

    char tok_path[1200];
    snprintf(tok_path, sizeof(tok_path), "%s/tokens.json", model);
    mynah_asr_tokenizer tk;
    if (mynah_asr_tokenizer_load(&tk, tok_path) != 0) { fprintf(stderr, "tokens.json\n"); return 2; }

    /* ---- items ---- */
    FILE *in = fopen(items_path, "r");
    if (!in) { perror(items_path); return 2; }
    int cap_items = 256, n_items = 0, cap_jobs = 4096, n_jobs = 0;
    item *items = calloc((size_t)cap_items, sizeof(item));
    job *jobs = calloc((size_t)cap_jobs, sizeof(job));
    char line[4096];
    while (fgets(line, sizeof(line), in)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (!line[0] || line[0] == '#') continue;
        char *f[6];
        int nf = 0;
        for (char *t = strtok(line, "\t"); t && nf < 6; t = strtok(NULL, "\t")) f[nf++] = t;
        if (nf < 6) continue;
        if (n_items == cap_items) { cap_items *= 2; items = realloc(items, (size_t)cap_items * sizeof(item)); }
        item *it = &items[n_items++];
        memset(it, 0, sizeof(*it));
        snprintf(it->id, sizeof(it->id), "%s", f[0]);
        snprintf(it->lang, sizeof(it->lang), "%s", f[1]);
        snprintf(it->ctx_wav, sizeof(it->ctx_wav), "%s", f[2]);
        snprintf(it->cur_wav, sizeof(it->cur_wav), "%s", f[3]);
        it->t0 = atof(f[4]);
        it->t1 = atof(f[5]);
    }
    fclose(in);

    char err[512] = "";
    asr_engine_cfg cfg = {.model_dir = model, .cap = cohort, .device = 0, .precision = precision,
                          .gemm = gemm, .weights = weights, .threads = 1, .record_frames = 1};
    asr_engine *e = !strcmp(engine, "cpu") ? asr_engine_open_cpu(&cfg, err, sizeof(err))
                                           : asr_engine_open_cuda(&cfg, err, sizeof(err));
    if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
    asr_engine_facts fa;
    asr_engine_get_facts(e, &fa);
    const int fs = fa.frame_sec > 0 ? (int)(fa.frame_sec * 16000.0 + 0.5) : FRAME_SAMPLES_DEFAULT;
    fprintf(stderr, "cuda_recovery_lab: engine %s on %s, precision=%s gemm=%s, %d items, cohort %d, "
                    "lookahead %d, frame %d samples\n", fa.name, fa.device, fa.precision, fa.gemm,
            n_items, cohort, la, fs);

    /* ---- jobs: one previous-turn decode per (clip, language), then per item ---- */
    float **ctx_pcm = calloc((size_t)n_items, sizeof(float *));
    size_t *ctx_n = calloc((size_t)n_items, sizeof(size_t));
#define NEWJOB() (n_jobs == cap_jobs ? (cap_jobs *= 2, jobs = realloc(jobs, (size_t)cap_jobs * sizeof(job)), \
                  memset(jobs + n_jobs, 0, (size_t)(cap_jobs - n_jobs) * sizeof(job)), n_jobs++) : n_jobs++)
    for (int i = 0; i < n_items; i++) {
        item *it = &items[i];
        it->ctx_job = -1;
        for (int k = 0; k < i; k++)
            if (!strcmp(items[k].ctx_wav, it->ctx_wav) && !strcmp(items[k].lang, it->lang)) {
                it->ctx_job = items[k].ctx_job;
                ctx_pcm[i] = ctx_pcm[k];
                ctx_n[i] = ctx_n[k];
                break;
            }
        if (it->ctx_job >= 0) continue;
        ctx_pcm[i] = load16k(it->ctx_wav, &ctx_n[i]);
        if (!ctx_pcm[i]) { fprintf(stderr, "%s: cannot load %s\n", it->id, it->ctx_wav); return 2; }
        const int j = NEWJOB();
        jobs[j].lang = it->lang;
        jobs[j].n = ctx_n[i];
        jobs[j].pcm = malloc(ctx_n[i] * sizeof(float));
        memcpy(jobs[j].pcm, ctx_pcm[i], ctx_n[i] * sizeof(float));
        it->ctx_job = j;
    }
    for (int i = 0; i < n_items; i++) {
        item *it = &items[i];
        size_t nall = 0;
        float *all = load16k(it->cur_wav, &nall);
        if (!all) { fprintf(stderr, "%s: cannot load %s\n", it->id, it->cur_wav); return 2; }
        size_t a = (size_t)(it->t0 * 16000.0), b = it->t1 > 0 ? (size_t)(it->t1 * 16000.0) : nall;
        if (b > nall) b = nall;
        if (a > b) a = b;
        const float *cur = all + a;
        const size_t ncur = b - a;
        int j = NEWJOB();
        jobs[j].lang = it->lang; jobs[j].n = ncur;
        jobs[j].pcm = malloc((ncur ? ncur : 1) * sizeof(float));
        memcpy(jobs[j].pcm, cur, ncur * sizeof(float));
        it->primary = j;
        it->retry = -1;
        if (strncmp(it->lang, "auto", 4) != 0) {
            j = NEWJOB();
            jobs[j].lang = "auto"; jobs[j].n = ncur;
            jobs[j].pcm = malloc((ncur ? ncur : 1) * sizeof(float));
            memcpy(jobs[j].pcm, cur, ncur * sizeof(float));
            it->retry = j;
        }
        for (int c = 0; c < n_ctx; c++) {
            size_t np = ctx_ms[c] < 0 ? ctx_n[i] : (size_t)ctx_ms[c] * 16u;
            if (np > ctx_n[i]) np = ctx_n[i];
            np -= np % (size_t)fs;
            it->keep_from[c] = (int)(np / (size_t)fs);
            it->ctx_ms_real[c] = (int)(np / 16u);
            j = NEWJOB();
            jobs[j].lang = it->lang; jobs[j].n = np + ncur;
            jobs[j].pcm = malloc((np + ncur ? np + ncur : 1) * sizeof(float));
            memcpy(jobs[j].pcm, ctx_pcm[i] + (ctx_n[i] - np), np * sizeof(float));
            memcpy(jobs[j].pcm + np, cur, ncur * sizeof(float));
            it->replay[c] = j;
        }
        free(all);
    }
    fprintf(stderr, "cuda_recovery_lab: %d decodes\n", n_jobs);

    /* ---- run ---- */
    int ids[MAXC];
    for (int b = 0; b < n_jobs; b += cohort) {
        const int n = n_jobs - b < cohort ? n_jobs - b : cohort;
        for (int i = 0; i < n; i++) ids[i] = b + i;
        if (run_cohort(e, jobs, ids, n, la) != 0) return 1;
    }

    /* ---- JSONL, the CPU lab's format ---- */
    static const char *rn[3] = {"frame", "word", "text"};
    for (int i = 0; i < n_items; i++) {
        const item *it = &items[i];
        const char *ctx_text = jobs[it->ctx_job].text;
        const job *p = &jobs[it->primary];
        printf("{\"id\":");
        json_str(stdout, it->id);
        printf(",\"lang\":");
        json_str(stdout, it->lang);
        printf(",\"cur_s\":%.3f,\"ctx_text\":", (double)p->n / 16000.0);
        json_str(stdout, ctx_text);
        printf(",\"primary\":");
        json_str(stdout, p->text);
        printf(",\"primary_lang\":");
        json_str(stdout, p->lang_out);
        printf(",\"empty\":%d", !mynah_asr_text_has_content(p->text));
        if (dump) {
            const job *cj = &jobs[it->ctx_job];
            printf(",\"ctx_samples\":%zu,\"ctx_tokens\":", cj->n);
            dump_tokens(&tk, cj->tok, cj->fr, cj->ntok);
            printf(",\"primary_tokens\":");
            dump_tokens(&tk, p->tok, p->fr, p->ntok);
        }
        if (it->retry >= 0) {
            const job *r = &jobs[it->retry];
            printf(",\"retry\":");
            json_str(stdout, r->text);
            printf(",\"retry_lang\":");
            json_str(stdout, r->lang_out);
            printf(",\"retry_gate\":\"%s\"",
                   mynah_asr_gate_name(mynah_asr_recovery_gate(r->text, it->lang, r->lang_out, ctx_text)));
        }
        printf(",\"replay\":[");
        for (int c = 0; c < n_ctx; c++) {
            const job *r = &jobs[it->replay[c]];
            const int kf = it->keep_from[c];
            printf("%s{\"ctx_ms\":%d,\"keep_from\":%d,\"full\":", c ? "," : "", it->ctx_ms_real[c], kf);
            json_str(stdout, r->text);
            printf(",\"around\":[");
            int first = 1;
            for (int k = 0; k < r->ntok; k++) {
                if (r->fr[k] < kf - BOUNDARY_SHOW || r->fr[k] > kf + BOUNDARY_SHOW) continue;
                printf("%s[%d,", first ? "" : ",", r->fr[k]);
                json_str(stdout, (r->tok[k] >= 0 && r->tok[k] < tk.n_pieces) ? tk.pieces[r->tok[k]] : "?");
                printf("]");
                first = 0;
            }
            printf("]");
            if (dump) {
                printf(",\"prefix_samples\":%zu,\"tokens\":", r->n - p->n);
                dump_tokens(&tk, r->tok, r->fr, r->ntok);
            }
            for (int q = 0; q < 3; q++) {
                const int k = mynah_asr_replay_keep_from(&tk, r->tok, r->fr, r->ntok, kf,
                                                         (mynah_asr_keep_rule)q, ctx_text, la + 1);
                char *kt = mynah_asr_detokenize(&tk, r->tok + k, r->ntok - k, NULL);
                printf(",\"%s\":", rn[q]);
                json_str(stdout, kt);
                printf(",\"%s_gate\":\"%s\"", rn[q],
                       mynah_asr_gate_name(mynah_asr_recovery_gate(kt, it->lang, r->lang_out, ctx_text)));
                free(kt);
            }
            printf("}");
        }
        printf("]}\n");
    }
    asr_engine_stats st;
    asr_engine_get_stats(e, &st);
    fprintf(stderr, "cuda_recovery_lab: %d items, %d decodes, %lu steps\n", n_items, n_jobs, st.steps);
    asr_engine_close(e);
    mynah_asr_tokenizer_free(&tk);
    return 0;
}
