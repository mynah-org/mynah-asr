/* gpu/engine_cpu.c — the REFERENCE engine: the library's public stream API
 * (src/mynah_asr.h) behind gpu/engine.h, so the GPU server can be exercised on
 * a machine without a GPU and so the cuda engine has an in-process reference
 * for its transcripts. Chosen only by `--engine cpu`; the cuda engine never
 * falls back to it.
 *
 * It is NOT the CPU product server: no prefork, no pinned pool, no qualified
 * numbers. It runs the same library the CPU server runs, one batched
 * `mynah_asr_stream_step_batch` per cohort, which is what makes its transcripts
 * the reference (they are the CPU server's transcripts by construction). */
#include "asr_engine.h"

#include "backend.h"
#include "cJSON.h"
#include "mynah_asr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    mynah_asr_stream *stream;
    int in_use, finished, lookahead;
    char lang[16];
    /* text produced by the step in flight: the callbacks append here */
    char *delta;
    size_t delta_len, delta_cap;
    double t0, t1;
    int n_cb;
} cpu_slot;

typedef struct cpu_engine cpu_engine;
struct cpu_engine {
    asr_engine base;
    asr_engine_cfg cfg;
    mynah_asr_model *m;
    int cap;
    cpu_slot *slots;
    int lookaheads[8], n_lookaheads, default_lookahead;
    double frame_sec;
    asr_engine_stats st;
    char model_name[128];
    char err[256];
};

static void on_result(const mynah_asr_result *res, void *ud) {
    cpu_slot *s = (cpu_slot *)ud;
    if (res->is_eou || !res->text) return;
    const size_t n = strlen(res->text);
    if (s->delta_len + n + 1 > s->delta_cap) {
        size_t nc = s->delta_cap ? s->delta_cap : 256;
        while (nc < s->delta_len + n + 1) nc *= 2;
        char *nb = realloc(s->delta, nc);
        if (!nb) return;
        s->delta = nb; s->delta_cap = nc;
    }
    memcpy(s->delta + s->delta_len, res->text, n + 1);
    s->delta_len += n;
    if (s->n_cb == 0) s->t0 = res->t0;
    s->t1 = res->t1;
    s->n_cb++;
    if (res->lang && res->lang[0]) snprintf(s->lang, sizeof(s->lang), "%s", res->lang);
}

static const asr_engine_ops *cpu_ops(void);
static void cpu_close(cpu_engine *e);

asr_engine *asr_engine_open_cpu(const asr_engine_cfg *cfg, char *err, size_t errcap) {
    cpu_engine *e = calloc(1, sizeof(*e));
    if (!e) { snprintf(err, errcap, "out of memory"); return NULL; }
    e->base.ops = cpu_ops();
    e->cfg = *cfg;
    e->cap = cfg->cap > 0 ? cfg->cap : 1;
    if (cfg->threads > 0) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", cfg->threads);
        setenv("MYNAH_ASR_THREADS", buf, 1);
    }
    /* int8 to match the CPU product's numerics; the pack decides what exists */
    e->m = mynah_asr_load_quant(cfg->model_dir, MYNAH_ASR_QUANT_INT8);
    if (!e->m) { snprintf(err, errcap, "the pack at %s did not load", cfg->model_dir); free(e); return NULL; }
    const char *why = mynah_asr_stream_unsupported(e->m);
    if (why) { snprintf(err, errcap, "%s", why); mynah_asr_free(e->m); free(e); return NULL; }
    e->n_lookaheads = mynah_asr_lookaheads(e->m, e->lookaheads);
    if (e->n_lookaheads <= 0) { snprintf(err, errcap, "the pack has no streaming presets"); mynah_asr_free(e->m); free(e); return NULL; }
    /* name, default preset and frame length: the public API does not expose
     * them, and the library is not modified for this tree -- read mynah.json */
    e->default_lookahead = e->lookaheads[0];
    e->frame_sec = 0.08;
    snprintf(e->model_name, sizeof(e->model_name), "%s", cfg->model_dir);
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/mynah.json", cfg->model_dir);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            const long len = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = malloc((size_t)len + 1);
            if (buf && fread(buf, 1, (size_t)len, f) == (size_t)len) {
                buf[len] = '\0';
                cJSON *j = cJSON_Parse(buf);
                const cJSON *nm = j ? cJSON_GetObjectItem(j, "name") : NULL;
                if (nm && cJSON_IsString(nm)) snprintf(e->model_name, sizeof(e->model_name), "%s", nm->valuestring);
                const cJSON *js = j ? cJSON_GetObjectItem(j, "streaming") : NULL;
                const cJSON *di = js ? cJSON_GetObjectItem(js, "default_preset_index") : NULL;
                if (di && cJSON_IsNumber(di) && di->valueint >= 0 && di->valueint < e->n_lookaheads)
                    e->default_lookahead = e->lookaheads[di->valueint];
                const cJSON *fm = js ? cJSON_GetObjectItem(js, "encoder_frame_ms") : NULL;
                if (fm && cJSON_IsNumber(fm)) e->frame_sec = fm->valuedouble / 1000.0;
                cJSON_Delete(j);
            }
            free(buf);
            fclose(f);
        }
    }
    e->slots = calloc((size_t)e->cap, sizeof(cpu_slot));
    if (!e->slots) { snprintf(err, errcap, "out of memory"); mynah_asr_free(e->m); free(e); return NULL; }
    if (mynah_asr_stream_batch_reserve(e->m, e->cap < MYNAH_ASR_STREAM_BATCH_MAX ? e->cap : MYNAH_ASR_STREAM_BATCH_MAX) != 0) {
        snprintf(err, errcap, "batch scratch reserve failed"); cpu_close(e); return NULL;
    }
    return &e->base;
}

static void cpu_close(cpu_engine *e) {
    if (!e) return;
    for (int i = 0; i < e->cap; i++) {
        if (e->slots[i].stream) mynah_asr_stream_close(e->slots[i].stream);
        free(e->slots[i].delta);
    }
    free(e->slots);
    if (e->m) mynah_asr_free(e->m);
    free(e);
}

static void cpu_facts(const cpu_engine *e, asr_engine_facts *f) {
    memset(f, 0, sizeof(*f));
    f->name = "cpu";
    f->device = mynah_asr_gemm_provider();
    f->precision = "int8";
    f->gemm = "library (src/qmat.c, src/sgemm.c)";
    f->model_name = e->model_name;
    f->cap = e->cap;
    f->n_lookaheads = e->n_lookaheads;
    f->qmax = 1;
    for (int i = 0; i < e->n_lookaheads; i++) {
        f->lookaheads[i] = e->lookaheads[i];
        if (e->lookaheads[i] + 1 > f->qmax) f->qmax = e->lookaheads[i] + 1;
    }
    f->default_lookahead = e->default_lookahead;
    f->sample_rate = mynah_asr_sample_rate(e->m);
    f->n_mels = 0;
    f->frame_sec = e->frame_sec;
}

static void cpu_stats(const cpu_engine *e, asr_engine_stats *s) { *s = e->st; }
static int cpu_lang_id(const cpu_engine *e, const char *lang) {
    if (!lang || !lang[0] || strcmp(lang, "auto") == 0) return 0;
    return mynah_asr_lang_id(e->m, lang);
}
static int cpu_lookahead_ok(const cpu_engine *e, int la) {
    for (int i = 0; i < e->n_lookaheads; i++)
        if (e->lookaheads[i] == la) return 1;
    return 0;
}
static int cpu_dead(const cpu_engine *e) { (void)e; return 0; }
static const char *cpu_error(const cpu_engine *e) { return e->err; }

static size_t cpu_dispatch_map(const cpu_engine *e, char *buf, size_t cap) {
    (void)e;
    return (size_t)snprintf(buf, cap,
        "engine            cpu            reference: src/mynah_asr.h stream API\n"
        "gemm              %-14s the library's own seam\n"
        "precision         int8           the CPU product's numerics\n",
        mynah_asr_gemm_provider());
}

static int cpu_slot_reset(cpu_engine *e, int slot, const char *lang_in, int lookahead) {
    if (slot < 0 || slot >= e->cap) return -1;
    cpu_slot *s = &e->slots[slot];
    /* the library resolves the prompt from the tag itself */
    const char *lang = (lang_in && lang_in[0] && strcmp(lang_in, "auto") != 0) ? lang_in : NULL;
    snprintf(s->lang, sizeof(s->lang), "%s", lang ? lang : "");
    if (s->stream && s->lookahead != lookahead) { mynah_asr_stream_close(s->stream); s->stream = NULL; }
    if (!s->stream) {
        s->stream = mynah_asr_stream_open(e->m, lang, lookahead);
        if (!s->stream) return -1;
        s->lookahead = lookahead;
    } else if (mynah_asr_stream_reset(s->stream, lang) != 0) {
        return -1;
    }
    s->in_use = 1; s->finished = 0; s->delta_len = 0; s->n_cb = 0; s->t0 = s->t1 = 0.0;
    return 0;
}

typedef struct { float *pcm; size_t n, cap; } stage_buf;
static stage_buf *g_stage;   /* [cap], lazily */

/* what the library still needs for its next chunk, MINUS what is already
 * staged here: a slot with a whole chunk staged needs nothing more, so the
 * server stages exactly one chunk per step (the cuda engine's mel buffer
 * gives the same answer for the same reason) */
static size_t cpu_slot_need_samples(const cpu_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap || !e->slots[slot].stream) return 0;
    const size_t need = mynah_asr_stream_need_samples(e->slots[slot].stream);
    const size_t have = g_stage ? g_stage[slot].n : 0;
    return have >= need ? 0 : need - have;
}

/* The library consumes a chunk inside feed; the engine contract is "feed, then
 * step consumes". Here feed only stores nothing: the step feeds the library. So
 * the samples are kept in a per-slot staging buffer until the step. */
static int cpu_slot_feed(cpu_engine *e, int slot, const float *pcm, size_t n) {
    if (slot < 0 || slot >= e->cap) return -1;
    if (!g_stage) { g_stage = calloc((size_t)e->cap, sizeof(stage_buf)); if (!g_stage) return -1; }
    stage_buf *b = &g_stage[slot];
    if (b->n + n > b->cap) {
        size_t nc = b->cap ? b->cap : 16000;
        while (nc < b->n + n) nc *= 2;
        float *nb = realloc(b->pcm, nc * sizeof(float));
        if (!nb) return -1;
        b->pcm = nb; b->cap = nc;
    }
    memcpy(b->pcm + b->n, pcm, n * sizeof(float));
    b->n += n;
    return 0;
}

static int cpu_slot_ready(const cpu_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap || !e->slots[slot].stream || !g_stage) return 0;
    return g_stage[slot].n >= mynah_asr_stream_need_samples(e->slots[slot].stream) &&
           mynah_asr_stream_need_samples(e->slots[slot].stream) > 0;
}
static double cpu_slot_audio_s(const cpu_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap || !e->slots[slot].stream) return 0.0;
    return mynah_asr_stream_audio_seconds(e->slots[slot].stream) +
           (g_stage ? (double)g_stage[slot].n / (double)mynah_asr_sample_rate(e->m) : 0.0);
}
static const char *cpu_slot_text(const cpu_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap || !e->slots[slot].stream) return "";
    return mynah_asr_stream_text(e->slots[slot].stream);
}
static const char *cpu_slot_lang(const cpu_engine *e, int slot) {
    if (slot < 0 || slot >= e->cap) return "";
    return e->slots[slot].lang;
}

static int cpu_step(cpu_engine *e, const asr_step_req *reqs, int n, asr_step_out *outs) {
    mynah_asr_stream *streams[MYNAH_ASR_STREAM_BATCH_MAX];
    const float *samples[MYNAH_ASR_STREAM_BATCH_MAX];
    size_t ns[MYNAH_ASR_STREAM_BATCH_MAX];
    void *ud[MYNAH_ASR_STREAM_BATCH_MAX];
    int idx[MYNAH_ASR_STREAM_BATCH_MAX];
    int B = 0;
    for (int i = 0; i < n; i++) {
        outs[i].text = ""; outs[i].t0 = outs[i].t1 = 0.0;
        outs[i].n_tokens = 0; outs[i].finished = 0; outs[i].stepped = 0;
        const int slot = reqs[i].slot;
        if (slot < 0 || slot >= e->cap) continue;
        cpu_slot *s = &e->slots[slot];
        if (!s->stream || !s->in_use || s->finished) { if (s->finished) outs[i].finished = 1; continue; }
        s->delta_len = 0; s->n_cb = 0;
        if (s->delta) s->delta[0] = '\0';
        const size_t need = mynah_asr_stream_need_samples(s->stream);
        const size_t have = g_stage ? g_stage[slot].n : 0;
        if (reqs[i].finalize && (have < need || need == 0)) {
            /* less than a chunk left: the tail, and the utterance is finished.
             * With a whole chunk still staged the slot takes the normal path
             * below, ONE chunk per step, so the server's peer check runs between
             * chunks and a reset client stops the work (S12-19), exactly as the
             * cuda engine's one-chunk-per-step contract does. */
            if (have > 0 && mynah_asr_stream_feed(s->stream, g_stage[slot].pcm, have, on_result, s) != 0) return -1;
            if (g_stage) g_stage[slot].n = 0;
            if (mynah_asr_stream_finish(s->stream, on_result, s) != 0) return -1;
            s->finished = 1;
            outs[i].finished = 1; outs[i].stepped = 1;
            outs[i].text = s->delta ? s->delta : ""; outs[i].t0 = s->t0; outs[i].t1 = s->t1;
            if (s->n_cb == 0) outs[i].t0 = outs[i].t1 = mynah_asr_stream_audio_seconds(s->stream);
            e->st.lanes++;
            continue;
        }
        if (have < need || need == 0 || B >= MYNAH_ASR_STREAM_BATCH_MAX) continue;
        streams[B] = s->stream; samples[B] = g_stage[slot].pcm; ns[B] = need; ud[B] = s; idx[B] = i;
        B++;
    }
    if (B > 0) {
        if (mynah_asr_stream_step_batch(streams, B, samples, ns, on_result, ud) != 0) return -1;
        for (int b = 0; b < B; b++) {
            const int i = idx[b], slot = reqs[i].slot;
            cpu_slot *s = &e->slots[slot];
            stage_buf *sb = &g_stage[slot];
            memmove(sb->pcm, sb->pcm + ns[b], (sb->n - ns[b]) * sizeof(float));
            sb->n -= ns[b];
            outs[i].stepped = 1;
            outs[i].text = s->delta ? s->delta : "";
            outs[i].t0 = s->t0; outs[i].t1 = s->t1;
            if (s->n_cb == 0) outs[i].t0 = outs[i].t1 = mynah_asr_stream_audio_seconds(s->stream);
            e->st.lanes++;
        }
        e->st.rows += (unsigned long)B;
    }
    e->st.steps++;
    return 0;
}

/* ------------------------------------------------------------- the ops table */
#define CE(e) ((cpu_engine *)(e))
#define CCE(e) ((const cpu_engine *)(e))
static void ops_close(asr_engine *e) { cpu_close(CE(e)); }
static void ops_facts(const asr_engine *e, asr_engine_facts *f) { cpu_facts(CCE(e), f); }
static void ops_stats(const asr_engine *e, asr_engine_stats *s) { cpu_stats(CCE(e), s); }
static int ops_lang_id(const asr_engine *e, const char *l) { return cpu_lang_id(CCE(e), l); }
static int ops_lookahead_ok(const asr_engine *e, int la) { return cpu_lookahead_ok(CCE(e), la); }
static int ops_slot_reset(asr_engine *e, int s, const char *l, int la) { return cpu_slot_reset(CE(e), s, l, la); }
static size_t ops_slot_need(const asr_engine *e, int s) { return cpu_slot_need_samples(CCE(e), s); }
static int ops_slot_feed(asr_engine *e, int s, const float *p, size_t n) { return cpu_slot_feed(CE(e), s, p, n); }
static int ops_slot_ready(const asr_engine *e, int s) { return cpu_slot_ready(CCE(e), s); }
static double ops_slot_audio(const asr_engine *e, int s) { return cpu_slot_audio_s(CCE(e), s); }
static const char *ops_slot_text(const asr_engine *e, int s) { return cpu_slot_text(CCE(e), s); }
static const char *ops_slot_lang(const asr_engine *e, int s) { return cpu_slot_lang(CCE(e), s); }
static int ops_step(asr_engine *e, const asr_step_req *r, int n, asr_step_out *o) { return cpu_step(CE(e), r, n, o); }
static int ops_dead(const asr_engine *e) { return cpu_dead(CCE(e)); }
static const char *ops_error(const asr_engine *e) { return cpu_error(CCE(e)); }
static size_t ops_dispatch(const asr_engine *e, char *b, size_t c) { return cpu_dispatch_map(CCE(e), b, c); }
static const asr_engine_ops CPU_OPS = {
    ops_close, ops_facts, ops_stats, ops_lang_id, ops_lookahead_ok, ops_slot_reset, ops_slot_need,
    ops_slot_feed, ops_slot_ready, ops_slot_audio, ops_slot_text, ops_slot_lang, ops_step, ops_dead,
    ops_error, ops_dispatch,
};
static const asr_engine_ops *cpu_ops(void) { return &CPU_OPS; }
