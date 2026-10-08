/* gpu/pack.c — see pack.h. */
#include "pack.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cJSON *load_json(const char *dir, const char *file) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); fclose(f); return NULL; }
    buf[len] = '\0';
    fclose(f);
    cJSON *j = cJSON_Parse(buf);
    free(buf);
    return j;
}

static int jint(const cJSON *o, const char *k, int dflt) {
    const cJSON *j = o ? cJSON_GetObjectItem(o, k) : NULL;
    return j && cJSON_IsNumber(j) ? j->valueint : dflt;
}
static double jnum(const cJSON *o, const char *k, double dflt) {
    const cJSON *j = o ? cJSON_GetObjectItem(o, k) : NULL;
    return j && cJSON_IsNumber(j) ? j->valuedouble : dflt;
}
static const char *jstr(const cJSON *o, const char *k) {
    const cJSON *j = o ? cJSON_GetObjectItem(o, k) : NULL;
    return j && cJSON_IsString(j) ? j->valuestring : NULL;
}

#define FAIL(...) do { snprintf(err, errcap, __VA_ARGS__); asr_pack_close(p); return -1; } while (0)

int asr_pack_open(asr_pack *p, const char *dir, char *err, size_t errcap) {
    memset(p, 0, sizeof(*p));
    char path[1024];
    p->cfg = load_json(dir, "mynah.json");
    if (!p->cfg) FAIL("%s/mynah.json is missing or not JSON", dir);
    const char *nm = jstr(p->cfg, "name");
    snprintf(p->name, sizeof(p->name), "%s", nm ? nm : dir);

    const char *wfile = jstr(p->cfg, "weights");
    if (!wfile) FAIL("mynah.json has no \"weights\"");
    snprintf(path, sizeof(path), "%s/%s", dir, wfile);
    p->weights = mynah_asr_st_open_quiet(path);
    if (!p->weights) FAIL("the f32 weights %s are missing: the GPU engine uploads f32 (S14-8 adds bf16/int8)", path);
    snprintf(path, sizeof(path), "%s/mel_filters.safetensors", dir);
    p->mel_filters = mynah_asr_st_open_quiet(path);
    if (!p->mel_filters) FAIL("%s is missing", path);

    if (mynah_asr_encoder_init(&p->enc, p->weights, 0) != 0) FAIL("encoder init failed (not a FastConformer pack?)");
    const cJSON *jenc = cJSON_GetObjectItem(p->cfg, "encoder");
    const char *jsub = jstr(jenc, "subsampling");
    if (jsub) { p->enc.causal = strstr(jsub, "causal") != NULL; p->enc.ss.causal = p->enc.causal; }
    const cJSON *jxs = jenc ? cJSON_GetObjectItem(jenc, "xscaling") : NULL;
    if (jxs && cJSON_IsTrue(jxs)) p->enc.xscale = sqrtf((float)p->enc.d_model);

    const cJSON *jdec = cJSON_GetObjectItem(p->cfg, "decoder");
    const char *dtype = jstr(jdec, "type");
    if (!dtype || strcmp(dtype, "rnnt_lstm") != 0)
        FAIL("decoder type '%s' is not rnnt_lstm: the GPU engine serves the RNNT greedy decoder only", dtype ? dtype : "?");
    if (cJSON_GetObjectItem(jdec, "durations")) FAIL("a TDT head is not served by the GPU engine (RNNT only)");
    const int blank = jint(jdec, "blank_id", -1), max_sym = jint(jdec, "max_symbols_per_step", -1);
    if (blank < 0 || max_sym <= 0) FAIL("decoder.blank_id / max_symbols_per_step missing");
    if (mynah_asr_decoder_init(&p->dec, p->weights, blank, max_sym, 0, NULL, 0) != 0) FAIL("decoder init failed");

    snprintf(path, sizeof(path), "%s/tokens.json", dir);
    if (mynah_asr_tokenizer_load(&p->tok, path) != 0) FAIL("%s failed to load", path);

    const cJSON *jf = cJSON_GetObjectItem(p->cfg, "features");
    const mynah_asr_tensor *fb = mynah_asr_st_get(p->mel_filters, "mel_fb");
    const mynah_asr_tensor *win = mynah_asr_st_get(p->mel_filters, "window");
    if (!jf || !fb || !win) FAIL("features section or mel filters missing");
    const char *norm = jstr(jf, "normalize");
    p->feat = (mynah_asr_feat_cfg){
        .sample_rate = jint(jf, "sample_rate", 0), .n_mels = jint(jf, "n_mels", 0),
        .n_fft = jint(jf, "n_fft", 0), .win_length = jint(jf, "win_length", 0),
        .hop_length = jint(jf, "hop_length", 0), .preemphasis = jnum(jf, "preemphasis", 0.0),
        .log_zero_guard = jnum(jf, "log_zero_guard", 0.0),
        .normalize_per_feature = norm && strcmp(norm, "per_feature") == 0,
        .mel_fb = (const float *)fb->data, .window = (const float *)win->data,
    };
    if (p->feat.sample_rate <= 0 || p->feat.hop_length <= 0 || p->feat.n_mels <= 0) FAIL("features config incomplete");
    if (p->feat.normalize_per_feature) FAIL("per-feature mel normalisation is not a streaming front end");
    const int sf = jint(jenc, "subsampling_factor", p->enc.ss.sub_factor);
    if (sf != p->enc.ss.sub_factor) FAIL("the pack declares subsampling_factor %d, the runtime subsamples by %d", sf, p->enc.ss.sub_factor);
    p->frame_sec = (double)p->feat.hop_length * (double)p->enc.ss.sub_factor / (double)p->feat.sample_rate;

    /* the streaming-path gates of src/mynah_asr.c (mynah_asr_stream_unsupported) */
    {
        const mynah_asr_enc_layer *L = &p->enc.layers[0];
        if (L->q_b || L->k_b || L->v_b || L->o_b || L->ff1_b1 || L->ff1_b2 || L->ff2_b1 ||
            L->ff2_b2 || L->pw1_b || L->pw2_b || L->dw_b)
            FAIL("the pack has linear biases (use_bias) and the streaming step does not add them");
        if (p->enc.bn_fold) FAIL("the pack folds a batch_norm into the conv module; the streaming step applies layer_norm");
        if (p->enc.xscale != 1.0f) FAIL("the pack scales the encoder input (xscaling)");
        if (!p->enc.causal) FAIL("the pack uses symmetric conv padding; the streaming step is causal-only");
        if (!p->enc.encproj_w) FAIL("the pack has no encoder projector (a CTC-only model?)");
    }

    p->left_ctx = -1; p->default_right = -1;
    const cJSON *js = cJSON_GetObjectItem(p->cfg, "streaming");
    const cJSON *presets = js ? cJSON_GetObjectItem(js, "att_context_presets") : NULL;
    if (!presets) FAIL("the pack has no streaming presets (not cache-aware)");
    const int def = jint(js, "default_preset_index", 0);
    int i = 0;
    for (cJSON *q = presets->child; q && i < 8; q = q->next, i++) {
        const cJSON *pl = cJSON_GetArrayItem(q, 0), *pr = cJSON_GetArrayItem(q, 1);
        if (!pl || !pr) FAIL("malformed streaming preset");
        p->lookaheads[i] = pr->valueint;
        if (i == def) { p->left_ctx = pl->valueint; p->default_right = pr->valueint; }
    }
    p->n_lookaheads = i;
    if (p->left_ctx < 0) FAIL("default preset index out of range");
    p->qmax = 1;
    for (i = 0; i < p->n_lookaheads; i++)
        if (p->lookaheads[i] + 1 > p->qmax) p->qmax = p->lookaheads[i] + 1;
    for (i = 0; i < p->n_lookaheads; i++)
        if (p->left_ctx % (p->lookaheads[i] + 1) != 0)
            FAIL("left context %d is not divisible by the chunk %d of preset %d (src/encoder.h invariant)", p->left_ctx, p->lookaheads[i] + 1, i);

    const cJSON *jprompt = cJSON_GetObjectItem(p->cfg, "prompt");
    p->default_prompt = jint(jprompt, "default_id", -1);
    if (p->enc.prompt_l1_w && p->default_prompt < 0) FAIL("the pack has a prompt projector but no default prompt id");

    if (mynah_asr_enc_relpos_table_init(&p->enc, p->left_ctx + p->qmax + 2) != 0 || !p->enc.relpos_tab)
        FAIL("the rel-pos table could not be built (the GPU attention reads it)");
    return 0;
}

void asr_pack_close(asr_pack *p) {
    mynah_asr_qmat_free(&p->dec.head);
    mynah_asr_encoder_free(&p->enc);
    mynah_asr_tokenizer_free(&p->tok);
    if (p->weights) mynah_asr_st_close(p->weights);
    if (p->mel_filters) mynah_asr_st_close(p->mel_filters);
    if (p->cfg) cJSON_Delete(p->cfg);
    memset(p, 0, sizeof(*p));
}

int asr_pack_lang_id(const asr_pack *p, const char *lang) {
    if (!lang || !lang[0] || strcmp(lang, "auto") == 0) return p->default_prompt;
    const cJSON *jprompt = cJSON_GetObjectItem(p->cfg, "prompt");
    const cJSON *dict = jprompt ? cJSON_GetObjectItem(jprompt, "dictionary") : NULL;
    const cJSON *e = dict ? cJSON_GetObjectItem(dict, lang) : NULL;
    return e && cJSON_IsNumber(e) ? e->valueint : -1;
}

int asr_pack_lookahead_ok(const asr_pack *p, int lookahead) {
    for (int i = 0; i < p->n_lookaheads; i++)
        if (p->lookaheads[i] == lookahead) return 1;
    return 0;
}

int asr_pack_chunk_mel(const asr_pack *p, int lookahead, int first) {
    const int sub = p->enc.ss.sub_factor;
    return first ? 1 + sub * lookahead : sub * (lookahead + 1);
}
