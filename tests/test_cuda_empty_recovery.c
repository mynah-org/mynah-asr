/* tests/test_cuda_empty_recovery.c — the GPU server's empty-turn recovery
 * (gpu/server/recovery_pool.c, S18) against the CPU reference, on a pack.
 *
 * The pool is the code the server runs: attempts on spare engine slots, their
 * lanes added to the steps of live slots exactly as gpu/server/main.c adds
 * them (lowest priority, at most --rows per step, only into the room a full
 * live cohort of --cap lanes would use). For every turn the same recovery is
 * also run by the library (mynah_asr_recover_turn, src/recover.c) and the two
 * decisions are compared. Three checks:
 *
 *   1. AGREEMENT: accept/reject and the accepted words equal the CPU
 *      reference's, on the items of an items.tsv (the S18 lab format; the
 *      previous turn is the item's ctx clip: its last replay_frames frames and
 *      the CPU's transcript of it). Up to numerics (bf16, own vs CPU kernels):
 *      reported per turn; --min-agree sets the gate (default 0.9).
 *   2. NEVER ON SILENCE OR NOISE: digital silence and white noise at -60, -45
 *      and -30 dBFS, 0.4-3.2 s, with and without a previous turn, never accept
 *      (GPU or CPU). Gate: 0.
 *   3. LIVE LANES UNCHANGED: --live slots decode real clips alongside the
 *      recovery lanes; their transcripts equal a run of the same clips without
 *      any recovery, byte for byte. Gate: identity.
 *
 * usage: tests/test_cuda_empty_recovery <model_dir> <items.tsv> [--engine cuda|cpu]
 *            [--precision f32|bf16] [--gemm own|own-tc] [--lookahead 3] [--live 8]
 *            [--cap 16] [--slots 8] [--rows 4] [--max-items N] [--min-agree 0.9]
 *            [--dump-dir D]
 * --dump-dir: also write every item's previous and current turn as the exact
 * 16 kHz f32 samples decoded here (D/<n>.ctx.f32, D/<n>.cur.f32) and
 * D/items.tsv naming them, so a client can send the server the same audio
 * (gpu/tools/recovery_ws_check.py --items D/items.tsv, format=f32le).
 * items.tsv: id lang ctx_wav cur_wav cur_start_s cur_end_s (tests/recovery_lab.c)
 * stdout: one JSON line per turn; stderr: the summary and the gates.
 * exit 0 all gates pass; 1 engine/model error; 2 usage/input; 3 a gate failed. */
#include "../gpu/asr_engine.h"
#include "../gpu/server/recovery_pool.h"

#include "audio.h"
#include "mynah_asr.h"
#include "recover.h"
#include "recovery.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXL 256

typedef struct {
    char id[128], lang[32];
    float *cur, *prev;           /* owned */
    size_t n_cur, n_prev;
    char *prev_text;             /* owned (NULL = none) */
    int synthetic;               /* silence/noise: must never be accepted */
    int cpu_empty;               /* the CPU's primary decode had no content */
    /* results */
    int cpu_acc, gpu_acc, gpu_done, cpu_votes, gpu_votes, cpu_attempts, gpu_attempts;
    mynah_asr_attempt_kind cpu_kind, gpu_kind;
    char *cpu_text, *gpu_text;
} turn;

static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p < 0x20) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
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

static char *cpu_decode(mynah_asr_stream *s, const char *lang, const float *x, size_t n) {
    if (mynah_asr_stream_reset(s, lang) != 0) return NULL;
    if (n && mynah_asr_stream_feed(s, x, n, NULL, NULL) != 0) return NULL;
    if (mynah_asr_stream_finish(s, NULL, NULL) != 0) return NULL;
    return strdup(mynah_asr_stream_text(s));
}

/* deterministic white noise at `dbfs` (uniform, so peak = 10^(dbfs/20)) */
static void noise(float *x, size_t n, double dbfs, unsigned seed) {
    const float amp = dbfs <= -200.0 ? 0.0f : (float)pow(10.0, dbfs / 20.0);
    unsigned st = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        x[i] = amp * ((float)(st >> 8) / 8388608.0f - 1.0f);
    }
}

/* live lanes: clip i on slot i, fed as fast as the engine takes it */
typedef struct {
    const float *pcm;
    size_t n, off;
    int done;
    char *text;
} live;

static int live_lanes(asr_engine *e, live *lv, int nl, asr_step_req *reqs) {
    int k = 0;
    for (int i = 0; i < nl; i++) {
        live *l = &lv[i];
        if (l->done) continue;
        const size_t need = asr_engine_slot_need_samples(e, i);
        if (l->off < l->n && need > 0) {
            size_t take = need;
            if (l->off + take > l->n) take = l->n - l->off;
            if (asr_engine_slot_feed(e, i, l->pcm + l->off, take) != 0) return -1;
            l->off += take;
        }
        const int fin = l->off >= l->n;
        if (asr_engine_slot_ready(e, i) || fin) { reqs[k].slot = i; reqs[k].finalize = fin; k++; }
    }
    return k;
}

static int live_after(asr_engine *e, live *lv, const asr_step_req *reqs, const asr_step_out *outs, int k) {
    for (int a = 0; a < k; a++)
        if (outs[a].finished) {
            live *l = &lv[reqs[a].slot];
            l->done = 1;
            l->text = strdup(asr_engine_slot_text(e, reqs[a].slot));
        }
    return 0;
}

/* One run: live clips on slots [0, nl) and, when turns != NULL, every turn
 * recovered through the pool, admitted as spare slots free up. */
static int run(asr_engine *e, rpool *rp, live *lv, int nl, const char *live_lang, int la,
               int cap, int rows, turn *turns, int nt, unsigned long *steps_mixed) {
    asr_step_req reqs[MAXL];
    asr_step_out outs[MAXL];
    for (int i = 0; i < nl; i++) {
        if (asr_engine_slot_reset(e, i, live_lang, la) != 0) return -1;
        lv[i].off = 0; lv[i].done = 0; free(lv[i].text); lv[i].text = NULL;
    }
    int next = 0, left = turns ? nt : 0;
    for (;;) {
        /* admit as many turns as the spare slots take, in order */
        while (turns && next < nt) {
            turn *t = &turns[next];
            const int j = rpool_start(rp, next, 0, t->lang, la, t->cur, t->n_cur,
                                      t->n_prev ? t->prev : NULL, t->n_prev, t->prev_text, 0.0);
            if (j == RP_START_BUSY) break;
            if (j == RP_START_ERROR) return -1;
            if (j == RP_START_BOUNDS) { t->gpu_done = 1; left--; }
            next++;
        }
        int n = live_lanes(e, lv, nl, reqs);
        if (n < 0) return -1;
        int nrec = 0;
        if (rp && rpool_pending(rp) > 0) {
            int room = cap - n;
            if (room > rows) room = rows;
            nrec = rpool_lanes(rp, reqs + n, room);
            if (nrec < 0) return -1;
        }
        if (n + nrec == 0) {
            if (left == 0) break;
            fprintf(stderr, "stall: %d turns left, nothing to step\n", left);
            return -1;
        }
        if (asr_engine_step(e, reqs, n + nrec, outs) != 0) { fprintf(stderr, "step: %s\n", asr_engine_error(e)); return -1; }
        if (n > 0 && nrec > 0) (*steps_mixed)++;
        live_after(e, lv, reqs, outs, n);
        if (nrec > 0) {
            rpool_stepped(rp, reqs + n, outs + n, nrec);
            rpool_result r;
            while (rpool_next_result(rp, 0.0, &r)) {
                turn *t = &turns[r.owner];
                t->gpu_done = 1;
                t->gpu_acc = r.status == RP_ACCEPTED;
                t->gpu_kind = r.kind;
                t->gpu_votes = r.votes;
                t->gpu_attempts = r.attempts;
                t->gpu_text = strdup(r.text);
                left--;
            }
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir> <items.tsv> [--engine cuda|cpu] [--precision f32|bf16] [--gemm own]\n"
                        "          [--lookahead 3] [--live 8] [--cap 16] [--slots 8] [--rows 4] [--max-items N] [--min-agree 0.9]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1], *items_path = argv[2], *engine = "cuda", *precision = "f32", *gemm = "own";
    int la = 3, nl = 8, cap = 16, slots = 8, rows = 4, max_items = 1 << 30;
    double min_agree = 0.9;
    const char *dump_dir = NULL;
    for (int i = 3; i + 1 < argc; i += 2) {
        const char *k = argv[i], *v = argv[i + 1];
        if (!strcmp(k, "--engine")) engine = v;
        else if (!strcmp(k, "--precision")) precision = v;
        else if (!strcmp(k, "--gemm")) gemm = v;
        else if (!strcmp(k, "--lookahead")) la = atoi(v);
        else if (!strcmp(k, "--live")) nl = atoi(v);
        else if (!strcmp(k, "--cap")) cap = atoi(v);
        else if (!strcmp(k, "--slots")) slots = atoi(v);
        else if (!strcmp(k, "--rows")) rows = atoi(v);
        else if (!strcmp(k, "--max-items")) max_items = atoi(v);
        else if (!strcmp(k, "--min-agree")) min_agree = atof(v);
        else if (!strcmp(k, "--dump-dir")) dump_dir = v;
        else { fprintf(stderr, "unknown option %s\n", k); return 2; }
    }
    if (nl < 0 || cap < nl || cap + rows > MAXL || slots < 1 || rows < 1) {
        fprintf(stderr, "need 0 <= --live <= --cap, --cap + --rows <= %d, --slots/--rows >= 1\n", MAXL);
        return 2;
    }

    /* ---- the CPU reference ---- */
    mynah_asr_model *m = mynah_asr_load(model);
    if (!m) { fprintf(stderr, "cannot load %s\n", model); return 1; }
    mynah_asr_recover_cfg rcfg;
    mynah_asr_recover_cfg_default(&rcfg);
    mynah_asr_recoverer *rec = mynah_asr_recoverer_open(m, la, &rcfg);
    mynah_asr_stream *cs = mynah_asr_stream_open(m, "auto", la);
    if (!rec || !cs) { fprintf(stderr, "cpu recoverer/stream open failed\n"); return 1; }
    const int fs = mynah_asr_frame_samples(m);
    const size_t tail = (size_t)rcfg.replay_frames * (size_t)fs;
    char tok_path[1200];
    snprintf(tok_path, sizeof(tok_path), "%s/tokens.json", model);
    mynah_asr_tokenizer tk;
    if (mynah_asr_tokenizer_load(&tk, tok_path) != 0) { fprintf(stderr, "tokens.json\n"); return 2; }

    /* ---- turns: the items, then silence and noise ---- */
    int cap_t = 64, nt = 0;
    turn *turns = calloc((size_t)cap_t, sizeof(turn));
    const char *live_wav[MAXL];
    int n_live_wav = 0;
    FILE *in = fopen(items_path, "r");
    if (!in) { perror(items_path); return 2; }
    char line[4096], ctx_loaded[1024] = "", ctx_lang[32] = "";
    float *ctx = NULL;
    size_t n_ctx = 0;
    char *ctx_text = NULL;
    int n_items = 0;
    FILE *dump_tsv = NULL;
    if (dump_dir) {
        char pth[1200];
        snprintf(pth, sizeof(pth), "%s/items.tsv", dump_dir);
        if (!(dump_tsv = fopen(pth, "w"))) { perror(pth); return 2; }
    }
    while (n_items < max_items && fgets(line, sizeof(line), in)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (!line[0] || line[0] == '#') continue;
        char *f[6];
        int nf = 0;
        for (char *t = strtok(line, "\t"); t && nf < 6; t = strtok(NULL, "\t")) f[nf++] = t;
        if (nf < 6) continue;
        if (strcmp(ctx_loaded, f[2]) != 0 || strcmp(ctx_lang, f[1]) != 0) {
            free(ctx); free(ctx_text);
            ctx = load16k(f[2], &n_ctx);
            ctx_text = ctx ? cpu_decode(cs, f[1], ctx, n_ctx) : NULL;
            if (!ctx || !ctx_text) { fprintf(stderr, "%s: cannot load/decode %s\n", f[0], f[2]); return 2; }
            snprintf(ctx_loaded, sizeof(ctx_loaded), "%s", f[2]);
            snprintf(ctx_lang, sizeof(ctx_lang), "%s", f[1]);
            if (n_live_wav < nl) {
                int dup = 0;
                for (int q = 0; q < n_live_wav; q++) dup |= !strcmp(live_wav[q], f[2]);
                if (!dup) live_wav[n_live_wav++] = strdup(f[2]);
            }
        }
        size_t nall = 0;
        float *all = load16k(f[3], &nall);
        if (!all) { fprintf(stderr, "%s: cannot load %s\n", f[0], f[3]); return 2; }
        size_t a = (size_t)(atof(f[4]) * 16000.0), b = atof(f[5]) > 0 ? (size_t)(atof(f[5]) * 16000.0) : nall;
        if (b > nall) b = nall;
        if (a > b) a = b;
        if (nt == cap_t) { cap_t *= 2; turns = realloc(turns, (size_t)cap_t * sizeof(turn)); memset(turns + nt, 0, (size_t)(cap_t - nt) * sizeof(turn)); }
        turn *t = &turns[nt++];
        snprintf(t->id, sizeof(t->id), "%s", f[0]);
        snprintf(t->lang, sizeof(t->lang), "%s", f[1]);
        t->n_cur = b - a;
        t->cur = malloc((t->n_cur ? t->n_cur : 1) * sizeof(float));
        memcpy(t->cur, all + a, t->n_cur * sizeof(float));
        free(all);
        t->n_prev = n_ctx < tail ? n_ctx : tail;
        t->prev = malloc((t->n_prev ? t->n_prev : 1) * sizeof(float));
        memcpy(t->prev, ctx + (n_ctx - t->n_prev), t->n_prev * sizeof(float));
        t->prev_text = strdup(ctx_text);
        if (dump_tsv) {
            char pc[1200], pu[1200];
            snprintf(pc, sizeof(pc), "%s/%d.ctx.f32", dump_dir, n_items);
            snprintf(pu, sizeof(pu), "%s/%d.cur.f32", dump_dir, n_items);
            FILE *fc = fopen(pc, "wb"), *fu = fopen(pu, "wb");
            if (!fc || !fu) { perror(dump_dir); return 2; }
            fwrite(ctx, sizeof(float), n_ctx, fc);
            fwrite(t->cur, sizeof(float), t->n_cur, fu);
            fclose(fc); fclose(fu);
            fprintf(dump_tsv, "%s\t%s\t%s\t%s\t0\t0\n", t->id, t->lang, pc, pu);
        }
        char *pt = cpu_decode(cs, t->lang, t->cur, t->n_cur);
        t->cpu_empty = pt && !mynah_asr_text_has_content(pt);
        free(pt);
        n_items++;
    }
    fclose(in);
    if (dump_tsv) fclose(dump_tsv);
    const double durs[4] = {0.4, 0.8, 1.6, 3.2}, dbs[4] = {-1000.0, -60.0, -45.0, -30.0};
    for (int d = 0; d < 4; d++)
        for (int q = 0; q < 4; q++)
            for (int wp = 0; wp < 2; wp++) {
                if (nt == cap_t) { cap_t *= 2; turns = realloc(turns, (size_t)cap_t * sizeof(turn)); memset(turns + nt, 0, (size_t)(cap_t - nt) * sizeof(turn)); }
                turn *t = &turns[nt++];
                snprintf(t->id, sizeof(t->id), "synthetic/%s/%.1fs/%s", q == 0 ? "silence" : q == 1 ? "noise-60" : q == 2 ? "noise-45" : "noise-30",
                         durs[d], wp ? "prev" : "noprev");
                snprintf(t->lang, sizeof(t->lang), "%s", n_items ? turns[0].lang : "auto");
                t->synthetic = 1;
                t->n_cur = (size_t)(durs[d] * 16000.0);
                t->cur = malloc(t->n_cur * sizeof(float));
                noise(t->cur, t->n_cur, dbs[q], (unsigned)(d * 16 + q * 2 + wp + 1));
                if (wp && n_items) {
                    t->n_prev = turns[0].n_prev;
                    t->prev = malloc((t->n_prev ? t->n_prev : 1) * sizeof(float));
                    memcpy(t->prev, turns[0].prev, t->n_prev * sizeof(float));
                    t->prev_text = strdup(turns[0].prev_text);
                }
            }
    fprintf(stderr, "test_cuda_empty_recovery: %d items + %d silence/noise turns\n", n_items, nt - n_items);

    /* ---- CPU reference decisions ---- */
    for (int i = 0; i < nt; i++) {
        turn *t = &turns[i];
        mynah_asr_recover_result rr;
        if (mynah_asr_recover_turn(rec, t->lang, t->cur, t->n_cur, t->n_prev ? t->prev : NULL, t->n_prev,
                                   t->prev_text, &rr) != 0) { fprintf(stderr, "%s: cpu recover failed\n", t->id); return 1; }
        t->cpu_acc = rr.accepted; t->cpu_kind = rr.winner_kind; t->cpu_votes = rr.votes; t->cpu_attempts = rr.attempts;
        t->cpu_text = strdup(mynah_asr_recover_text(rec));
    }

    /* ---- the engine: live slots [0, cap), spare slots [cap, cap + slots) ---- */
    char err[512] = "";
    asr_engine_cfg cfg = {.model_dir = model, .cap = cap + slots, .device = 0, .precision = precision,
                          .gemm = gemm, .threads = 1, .record_frames = 1, .record_frames_from = cap};
    asr_engine *e = !strcmp(engine, "cpu") ? asr_engine_open_cpu(&cfg, err, sizeof(err))
                                           : asr_engine_open_cuda(&cfg, err, sizeof(err));
    if (!e) { fprintf(stderr, "open: %s\n", err); return 1; }
    asr_engine_facts fa;
    asr_engine_get_facts(e, &fa);
    rpool_cfg pc;
    memset(&pc, 0, sizeof(pc));
    pc.first_slot = cap; pc.n_slots = slots; pc.deadline_s = 0.0;
    pc.sample_rate = fa.sample_rate;
    pc.frame_samples = (int)(fa.frame_sec * (double)fa.sample_rate + 0.5);
    pc.cfg = rcfg;
    if (pc.frame_samples != fs) { fprintf(stderr, "frame samples: engine %d, library %d\n", pc.frame_samples, fs); return 1; }
    rpool *rp = rpool_open(e, &tk, &pc);
    if (!rp) { fprintf(stderr, "rpool_open failed\n"); return 1; }
    fprintf(stderr, "test_cuda_empty_recovery: engine %s on %s precision=%s gemm=%s, live %d, cap %d, spare %d, rows %d, lookahead %d\n",
            fa.name, fa.device, fa.precision, fa.gemm, n_live_wav, cap, slots, rows, la);

    live lv[MAXL];
    memset(lv, 0, sizeof(lv));
    const int nlv = n_live_wav;
    for (int i = 0; i < nlv; i++) {
        lv[i].pcm = load16k(live_wav[i], &lv[i].n);
        if (!lv[i].pcm) return 2;
    }
    const char *live_lang = n_items ? turns[0].lang : "auto";
    unsigned long mixed = 0, mixed0 = 0;
    char *ref[MAXL] = {0};
    if (run(e, NULL, lv, nlv, live_lang, la, cap, rows, NULL, 0, &mixed0) != 0) return 1;
    for (int i = 0; i < nlv; i++) { ref[i] = lv[i].text; lv[i].text = NULL; }
    if (run(e, rp, lv, nlv, live_lang, la, cap, rows, turns, nt, &mixed) != 0) return 1;

    /* ---- report and gates ---- */
    int agree_dec = 0, agree_txt = 0, n_cmp = 0, emp = 0, emp_agree = 0, cpu_acc = 0, gpu_acc = 0;
    int syn_cpu = 0, syn_gpu = 0, live_bad = 0;
    for (int i = 0; i < nt; i++) {
        turn *t = &turns[i];
        if (!t->gpu_text) t->gpu_text = strdup("");
        const int dec = t->cpu_acc == t->gpu_acc;
        const int txt = dec && (!t->cpu_acc || mynah_asr_text_same_words(t->cpu_text, t->gpu_text));
        if (t->synthetic) { syn_cpu += t->cpu_acc; syn_gpu += t->gpu_acc; }
        else {
            n_cmp++; agree_dec += dec; agree_txt += txt; cpu_acc += t->cpu_acc; gpu_acc += t->gpu_acc;
            if (t->cpu_empty) { emp++; emp_agree += txt; }
        }
        printf("{\"id\":");
        json_str(stdout, t->id);
        printf(",\"lang\":");
        json_str(stdout, t->lang);
        printf(",\"cur_s\":%.3f,\"synthetic\":%d,\"cpu_primary_empty\":%d,\"agree\":%d,"
               "\"cpu\":{\"accepted\":%d,\"kind\":\"%s\",\"votes\":%d,\"attempts\":%d,\"text\":",
               (double)t->n_cur / 16000.0, t->synthetic, t->cpu_empty, txt, t->cpu_acc,
               t->cpu_acc ? rpool_kind_name(t->cpu_kind) : "", t->cpu_votes, t->cpu_attempts);
        json_str(stdout, t->cpu_text);
        printf("},\"gpu\":{\"accepted\":%d,\"kind\":\"%s\",\"votes\":%d,\"attempts\":%d,\"text\":",
               t->gpu_acc, t->gpu_acc ? rpool_kind_name(t->gpu_kind) : "", t->gpu_votes, t->gpu_attempts);
        json_str(stdout, t->gpu_text);
        printf("}}\n");
    }
    for (int i = 0; i < nlv; i++) {
        const int same = ref[i] && lv[i].text && !strcmp(ref[i], lv[i].text);
        if (!same) {
            live_bad++;
            fprintf(stderr, "LIVE DIFF slot %d (%s):\n  alone: %s\n  mixed: %s\n", i, live_wav[i], ref[i] ? ref[i] : "(null)",
                    lv[i].text ? lv[i].text : "(null)");
        }
    }
    rpool_stats st;
    rpool_get_stats(rp, &st);
    const double frac = n_cmp ? (double)agree_txt / (double)n_cmp : 1.0;
    fprintf(stderr, "test_cuda_empty_recovery: items %d: decision agree %d/%d, words agree %d/%d (%.3f); accepted cpu %d gpu %d; "
                    "cpu-primary-empty %d (agree %d)\n", n_cmp, agree_dec, n_cmp, agree_txt, n_cmp, frac, cpu_acc, gpu_acc, emp, emp_agree);
    fprintf(stderr, "test_cuda_empty_recovery: silence/noise %d turns: accepted cpu %d gpu %d\n", nt - n_items, syn_cpu, syn_gpu);
    fprintf(stderr, "test_cuda_empty_recovery: pool started %lu busy %lu bounds %lu lanes %lu attempts %lu; steps mixing live+recovery %lu; "
                    "live clips %d identical %d\n", st.started, st.busy, st.out_of_bounds, st.lanes, st.attempts, mixed, nlv, nlv - live_bad);
    const int ok_agree = frac >= min_agree, ok_syn = syn_gpu == 0 && syn_cpu == 0, ok_live = live_bad == 0;
    fprintf(stderr, "GATE agreement >= %.2f: %s\nGATE silence/noise never accepted: %s\nGATE live lanes byte-identical: %s\n",
            min_agree, ok_agree ? "PASS" : "FAIL", ok_syn ? "PASS" : "FAIL", ok_live ? "PASS" : "FAIL");
    rpool_close(rp);
    asr_engine_close(e);
    mynah_asr_recoverer_close(rec);
    mynah_asr_stream_close(cs);
    mynah_asr_free(m);
    mynah_asr_tokenizer_free(&tk);
    return ok_agree && ok_syn && ok_live ? 0 : 3;
}
