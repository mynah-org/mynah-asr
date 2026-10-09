/* gpu/offline_engine.c — see asr_offline.h. */
#include "asr_offline.h"

#include "aed_gpu.h"
#include "../vendor/cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct asr_offline {
    asr_offline_cfg cfg;
    mynah_asr_model *m;
    asr_aed_gpu *gpu;            /* NULL on the cpu engine */
    char model_name[128];
    char err[512];
    asr_offline_stats st;
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

/* the pack's display name, read the way gpu/pack.c reads it */
static void read_name(const char *dir, char *out, size_t cap) {
    snprintf(out, cap, "%s", dir);
    char path[1024];
    snprintf(path, sizeof(path), "%s/mynah.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = len > 0 ? malloc((size_t)len + 1) : NULL;
    if (buf && fread(buf, 1, (size_t)len, f) == (size_t)len) {
        buf[len] = '\0';
        cJSON *j = cJSON_Parse(buf);
        const cJSON *nm = j ? cJSON_GetObjectItem(j, "name") : NULL;
        if (nm && cJSON_IsString(nm)) snprintf(out, cap, "%s", nm->valuestring);
        cJSON_Delete(j);
    }
    free(buf);
    fclose(f);
}

asr_offline *asr_offline_open(const asr_offline_cfg *cfg, char *err, size_t errcap) {
    const int cuda = cfg->engine && strcmp(cfg->engine, "cuda") == 0;
    if (!cuda && !(cfg->engine && strcmp(cfg->engine, "cpu") == 0)) {
        snprintf(err, errcap, "engine '%s' is not one of cuda, cpu", cfg->engine ? cfg->engine : "?");
        return NULL;
    }
    asr_offline *o = calloc(1, sizeof(*o));
    if (!o) { snprintf(err, errcap, "out of memory"); return NULL; }
    o->cfg = *cfg;
    if (o->cfg.max_items < 1) o->cfg.max_items = 1;
    if (!o->cfg.decoder) o->cfg.decoder = "gpu";
    if (strcmp(o->cfg.decoder, "gpu") != 0 && strcmp(o->cfg.decoder, "host") != 0) {
        snprintf(err, errcap, "aed decoder '%s' is not one of gpu, host", o->cfg.decoder);
        free(o);
        return NULL;
    }
    if (cfg->threads > 0) {      /* the library pool reads it at first use */
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", cfg->threads);
        setenv("MYNAH_ASR_THREADS", buf, 1);
    }
    read_name(cfg->model_dir, o->model_name, sizeof(o->model_name));
    /* f32: the reference arithmetic, and the weights the GPU uploads */
    o->m = mynah_asr_load_quant(cfg->model_dir, MYNAH_ASR_QUANT_F32);
    if (!o->m) {
        snprintf(err, errcap, "the library could not load %s (see the lines above)", cfg->model_dir);
        free(o);
        return NULL;
    }
    if (!mynah_asr_can_translate(o->m)) {
        /* the capability this engine exists for; a CTC/RNNT pack has the
         * streaming engine (and an offline CTC/RNNT GPU path is not built) */
        snprintf(err, errcap, "%s is not an AED pack: the offline engine serves attention "
                              "encoder-decoders (Canary)", cfg->model_dir);
        mynah_asr_free(o->m);
        free(o);
        return NULL;
    }
    if (cuda) {
        asr_aed_gpu_cfg gc = {.model_dir = cfg->model_dir, .device = cfg->device,
                              .max_items = o->cfg.max_items,
                              .seg_sec = mynah_asr_segment_limit(o->m),
                              .precision = cfg->precision, .gemm = cfg->gemm,
                              .decode_on_gpu = strcmp(o->cfg.decoder, "gpu") == 0};
        o->gpu = asr_aed_gpu_open(&gc, err, errcap);
        if (!o->gpu) {
            mynah_asr_free(o->m);
            free(o);
            return NULL;
        }
        mynah_asr_offload off;
        asr_aed_gpu_offload(o->gpu, &off);
        mynah_asr_set_offload(o->m, &off);
    }
    return o;
}

void asr_offline_close(asr_offline *o) {
    if (!o) return;
    if (o->m) mynah_asr_set_offload(o->m, NULL);
    asr_aed_gpu_close(o->gpu);
    mynah_asr_free(o->m);
    free(o);
}

void asr_offline_get_facts(const asr_offline *o, asr_offline_facts *f) {
    memset(f, 0, sizeof(*f));
    f->name = o->gpu ? "cuda" : "cpu";
    f->model_name = o->model_name;
    f->sample_rate = mynah_asr_sample_rate(o->m);
    f->can_translate = mynah_asr_can_translate(o->m);
    f->max_items = o->cfg.max_items;
    f->seg_sec = mynah_asr_segment_limit(o->m);
    if (o->gpu) {
        asr_aed_gpu_facts g;
        asr_aed_gpu_get_facts(o->gpu, &g);
        f->device = g.device; f->precision = g.precision; f->gemm = g.gemm; f->decoder = g.decoder;
        f->vram_total = g.vram_total; f->vram_used = g.vram_used;
        f->vram_weights = g.vram_weights; f->vram_scratch = g.vram_scratch;
    } else {
        f->device = "cpu"; f->precision = "f32"; f->gemm = "sgemm-own"; f->decoder = "host";
    }
}

void asr_offline_get_stats(const asr_offline *o, asr_offline_stats *s) {
    *s = o->st;
    if (o->gpu) {
        asr_aed_gpu_stats g;
        asr_aed_gpu_get_stats(o->gpu, &g);
        s->enc_calls = g.enc_calls; s->enc_rows = g.enc_rows;
        s->dec_calls = g.dec_calls; s->dec_steps = g.dec_steps; s->dec_tokens = g.dec_tokens;
        s->host_ss_ms = g.host_ss_ms; s->enc_ms = g.enc_ms; s->dec_ms = g.dec_ms;
        s->device_errors = g.errors;
    }
}

int asr_offline_lang_ok(const asr_offline *o, const char *lang) {
    return mynah_asr_lang_supported(o->m, lang);
}

int asr_offline_dead(const asr_offline *o) { return o->gpu ? asr_aed_gpu_dead(o->gpu) : 0; }

const char *asr_offline_error(const asr_offline *o) {
    if (o->gpu && asr_aed_gpu_dead(o->gpu)) return asr_aed_gpu_error(o->gpu);
    return o->err;
}

size_t asr_offline_dispatch_map(const asr_offline *o, char *buf, size_t cap) {
    if (o->gpu) return asr_aed_gpu_dispatch_map(o->gpu, buf, cap);
    const int n = snprintf(buf, cap,
        "offline  host: wav, segmentation, mel, prompt, detok    library (src/mynah_asr.c)\n"
        "encoder  conformer layers                                library sgemm (cpu reference)\n"
        "decoder  aed greedy                                      library (src/decoder_aed.c)\n");
    return n < 0 ? 0 : ((size_t)n < cap ? (size_t)n : cap - 1);
}

int asr_offline_transcribe(asr_offline *o, int n, const float *const *pcm, const size_t *ns,
                           const char *const *langs, char **texts, char (*langs_out)[16],
                           mynah_asr_word **words, int *n_words) {
    const double t0 = now_ms();
    for (int i = 0; i < n; i++) {
        texts[i] = NULL;
        if (words) { words[i] = NULL; n_words[i] = 0; }
        if (langs_out) langs_out[i][0] = '\0';
    }
    if (n <= 0) return 0;
    if (asr_offline_dead(o)) return -1;
    o->st.batches++;
    o->st.items += (unsigned long)n;
    int failed = 0;
    if (mynah_asr_transcribe_batch_ts(o->m, pcm, ns, n, langs, -1, texts, langs_out,
                                      words, n_words) != 0) {
        o->st.failed_batches++;
        if (asr_offline_dead(o)) { o->st.wall_ms += now_ms() - t0; return -1; }
        /* all-or-nothing: retry each item alone, so one bad request (a language
         * the pack refuses at prompt time, an oversized segment) fails alone */
        for (int i = 0; i < n; i++) {
            o->st.retried_items++;
            char *t = NULL;
            char lo[1][16] = {{0}};
            mynah_asr_word *w = NULL;
            int nw = 0;
            if (mynah_asr_transcribe_batch_ts(o->m, &pcm[i], &ns[i], 1, langs ? &langs[i] : NULL,
                                              -1, &t, lo, words ? &w : NULL,
                                              words ? &nw : NULL) == 0) {
                texts[i] = t;
                if (langs_out) memcpy(langs_out[i], lo[0], sizeof(lo[0]));
                if (words) { words[i] = w; n_words[i] = nw; }
            } else {
                failed++;
            }
            if (asr_offline_dead(o)) { o->st.wall_ms += now_ms() - t0; return -1; }
        }
    }
    o->st.wall_ms += now_ms() - t0;
    return failed;
}
