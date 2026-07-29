#include "mynah_asr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend.h"
#include "decoder.h"
#include "decoder_aed.h"
#include "decoder_ctc.h"
#include "encoder.h"
#include "engine.h"
#include "features.h"
#include "threads.h"
#include "tokenizer.h"
#include "weights.h"
#include "../vendor/cJSON.h"

const char *mynah_asr_version(void) { return MYNAH_ASR_VERSION; }

#define MYNAH_ASR_AED_PROMPT_MAX 16

struct mynah_asr_model {
    cJSON *cfg;                     /* mynah.json (vivo per il prompt dictionary) */
    mynah_asr_safetensors *weights;
    mynah_asr_safetensors *mel_filters;
    mynah_asr_encoder enc;
    mynah_asr_decoder dec;
    mynah_asr_ctc ctc;                  /* head CTC (hybrid o CTC puro); w NULL se assente */
    const mynah_asr_engine *engine;     /* decoder attivo (vtable, vedi engine.h)  */
    const mynah_asr_engine *engine_dflt;/* engine di default del modello           */
    mynah_asr_aed aed;                  /* decoder AED (Canary); layers NULL se assente */
    int is_aed, aed_eos, aed_ts;    /* aed_ts: supporta i token <|timestamp|> */
    char aed_target[8];             /* lingua di uscita ("" = come la sorgente) */
    mynah_asr_tokenizer tok;
    mynah_asr_feat_cfg feat;
    int left_ctx, default_right;    /* att context dal preset di default */
    int lookaheads[8], n_lookaheads;
    int default_prompt;
    double frame_sec;               /* durata di un frame encoder (hop*sub/sr) */
    double seg_sec;                 /* limite per segmento offline (default 300 s) */
};

/* Default per-segment limit for offline decoding. Full-attention/AED models
 * (Parakeet, Canary) are trained on SHORT utterances: past ~30 s the quality
 * collapses (measured on a 305 s FLEURS file: CER 0.68 as a single segment, 0.29
 * at 60 s, 0.05 at 30 s; Canary de>en overlap 0.16 -> 0.76). Nemotron has
 * windowed (chunked) attention: it does not degrade, so for it the limit only
 * bounds memory. */
#define MYNAH_ASR_SEG_DEFAULT 300.0
#define MYNAH_ASR_SEG_OFFLINE 30.0

static cJSON *load_json(const char *dir, const char *file) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "mynah-asr: missing %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); fclose(f); return NULL; }
    buf[len] = '\0';
    fclose(f);
    cJSON *j = cJSON_Parse(buf);
    free(buf);
    return j;
}

/* Safe accessors over mynah.json: a missing key or the wrong type -> a message
 * on stderr + *bad = 1 (the load fails cleanly instead of dereferencing NULL). */
static const cJSON *jneed(const cJSON *o, const char *k, int *bad) {
    const cJSON *j = o ? cJSON_GetObjectItem(o, k) : NULL;
    if (!j) {
        fprintf(stderr, "mynah-asr: mynah.json: missing \"%s\"\n", k);
        if (bad) *bad = 1;
    }
    return j;
}

static int jint(const cJSON *o, const char *k, int *bad) {
    const cJSON *j = jneed(o, k, bad);
    if (j && !cJSON_IsNumber(j)) { fprintf(stderr, "mynah-asr: \"%s\" non numerico\n", k); if (bad) *bad = 1; }
    return j && cJSON_IsNumber(j) ? j->valueint : 0;
}

static double jnum(const cJSON *o, const char *k, int *bad) {
    const cJSON *j = jneed(o, k, bad);
    if (j && !cJSON_IsNumber(j)) { fprintf(stderr, "mynah-asr: \"%s\" non numerico\n", k); if (bad) *bad = 1; }
    return j && cJSON_IsNumber(j) ? j->valuedouble : 0.0;
}

static const char *jstr(const cJSON *o, const char *k, int *bad) {
    const cJSON *j = jneed(o, k, bad);
    if (j && !cJSON_IsString(j)) { fprintf(stderr, "mynah-asr: \"%s\" non stringa\n", k); if (bad) *bad = 1; }
    return j && cJSON_IsString(j) ? j->valuestring : "";
}

static int aed_build_prompt(const mynah_asr_model *m, const char *lang, int *ids, int want_ts);

/* -------------------------------------------------------- concrete engines
 * They decode the encoder output of one segment into tokens (see engine.h).
 * They live here because they consume the model internals; when an engine with
 * its own state arrives (Whisper-style) it will move to its own module. */

static int eng_decode_rnnt(struct mynah_asr_model *m, const float *enc, int T,
                           const char *lang, int want_ts, int **tokens, int **frames) {
    (void)lang;
    mynah_asr_dec_state *s = malloc(sizeof(*s));
    if (!s) return -1;
    mynah_asr_dec_state_reset(&m->dec, s);
    const int cap = T * m->dec.max_symbols;
    *tokens = malloc((size_t)cap * sizeof(int));
    *frames = want_ts && *tokens ? malloc((size_t)cap * sizeof(int)) : NULL;
    const int n = *tokens ? mynah_asr_greedy_decode(&m->dec, s, enc, T, *tokens,
                                                *frames, cap) : -1;
    free(s);
    return n;
}

static int eng_decode_ctc(struct mynah_asr_model *m, const float *enc, int T,
                          const char *lang, int want_ts, int **tokens, int **frames) {
    (void)lang;
    *tokens = malloc((size_t)T * sizeof(int));       /* CTC: <= 1 token/frame */
    *frames = want_ts && *tokens ? malloc((size_t)T * sizeof(int)) : NULL;
    return *tokens ? mynah_asr_ctc_decode(&m->ctc, enc, T, *tokens, *frames, T) : -1;
}

static int eng_decode_aed(struct mynah_asr_model *m, const float *enc, int T,
                          const char *lang, int want_ts, int **tokens, int **frames) {
    int pids[MYNAH_ASR_AED_PROMPT_MAX];
    const int ts = want_ts && m->aed_ts;             /* v2: niente <|timestamp|> */
    const int n_p = aed_build_prompt(m, lang, pids, ts);
    if (n_p <= 0) return -1;
    /* with timestamps every word costs 2 extra <|N|> tokens */
    const int cap = (ts ? 3 * T : T) + m->aed.max_gen_delta;
    *frames = NULL;                                  /* times live in the <|N|> tokens */
    *tokens = malloc((size_t)cap * sizeof(int));
    return *tokens ? mynah_asr_aed_decode(&m->aed, enc, T, pids, n_p, m->aed_eos,
                                      *tokens, cap) : -1;
}

static const mynah_asr_engine ENG_RNNT = {"rnnt-tdt", 0, eng_decode_rnnt};
static const mynah_asr_engine ENG_CTC = {"ctc", 1, eng_decode_ctc};
static const mynah_asr_engine ENG_AED = {"aed", 0, eng_decode_aed};

mynah_asr_model *mynah_asr_load(const char *model_dir) { return mynah_asr_load_quant(model_dir, MYNAH_ASR_QUANT_F32); }

mynah_asr_model *mynah_asr_load_quant(const char *model_dir, int quant) {
    mynah_asr_model *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    char path[1024];

    m->cfg = load_json(model_dir, "mynah.json");
    if (!m->cfg) goto fail;
    int bad = 0;

    /* pre-quantized on disk? (mynah-asr quantize) -> instant load, no f32 */
    const char *wfile = jstr(m->cfg, "weights", &bad);
    if (bad) goto fail;
    m->weights = NULL;
    if (quant != MYNAH_ASR_QUANT_F32) {
        snprintf(path, sizeof(path), "%s/model.%s.safetensors", model_dir,
                 quant == MYNAH_ASR_QUANT_INT8 ? "int8" : "int4");
        m->weights = mynah_asr_st_open_quiet(path);
        if (m->weights)
            fprintf(stderr, "mynah-asr: checkpoint pre-quantizzato %s\n", path);
    }
    if (!m->weights) {
        snprintf(path, sizeof(path), "%s/%s", model_dir, wfile);
        m->weights = mynah_asr_st_open_quiet(path);
        if (!m->weights) {   /* container alternativo GGUF (tools/export_gguf.py) */
            snprintf(path, sizeof(path), "%s/model.gguf", model_dir);
            m->weights = mynah_asr_st_open_quiet(path);
            if (m->weights)
                fprintf(stderr, "mynah-asr: pesi dal container GGUF %s\n", path);
        }
        if (!m->weights) {   /* riapre il path primario per il messaggio d'errore */
            snprintf(path, sizeof(path), "%s/%s", model_dir, wfile);
            m->weights = mynah_asr_st_open(path);
        }
    }
    snprintf(path, sizeof(path), "%s/mel_filters.safetensors", model_dir);
    m->mel_filters = mynah_asr_st_open(path);
    if (!m->weights || !m->mel_filters) goto fail;

    if (mynah_asr_encoder_init(&m->enc, m->weights, quant) != 0) {
        fprintf(stderr, "mynah-asr: encoder init failed\n");
        goto fail;
    }

    /* causality and xscaling from the config (init infers from the naming; the config wins) */
    const cJSON *jenc = cJSON_GetObjectItem(m->cfg, "encoder");
    const cJSON *jsub = jenc ? cJSON_GetObjectItem(jenc, "subsampling") : NULL;
    if (jsub && cJSON_IsString(jsub)) {
        m->enc.causal = strstr(jsub->valuestring, "causal") != NULL;
        m->enc.ss.causal = m->enc.causal;
    }
    const cJSON *jxs = jenc ? cJSON_GetObjectItem(jenc, "xscaling") : NULL;
    if (jxs && cJSON_IsTrue(jxs))
        m->enc.xscale = sqrtf((float)m->enc.d_model);

    const cJSON *jdec = jneed(m->cfg, "decoder", &bad);
    const char *dec_type = jstr(jdec, "type", &bad);
    if (bad) goto fail;
    if (strcmp(dec_type, "aed_transformer") == 0) {
        const int a_layers = jint(jdec, "n_layers", &bad);
        const int a_heads = jint(jdec, "n_heads", &bad);
        const int a_maxseq = jint(jdec, "max_seq", &bad);
        const int a_delta = jint(jdec, "max_generation_delta", &bad);
        if (bad) goto fail;
        if (mynah_asr_aed_init(&m->aed, m->weights, a_layers, a_heads, a_maxseq,
                           a_delta, quant) != 0) {
            fprintf(stderr, "mynah-asr: AED decoder init failed\n");
            goto fail;
        }
        m->is_aed = 1;
        m->engine_dflt = &ENG_AED;
    } else if (strcmp(dec_type, "ctc") == 0) {
        /* pure CTC: no prednet/joint, the head IS the decoder */
        if (mynah_asr_ctc_init(&m->ctc, m->weights) != 0) {
            fprintf(stderr, "mynah-asr: missing CTC head\n");
            goto fail;
        }
        m->engine_dflt = &ENG_CTC;
    } else {
        int durations[MYNAH_ASR_MAX_DURATIONS];
        int n_durations = 0;
        const cJSON *jdur = cJSON_GetObjectItem(jdec, "durations");
        for (cJSON *d = jdur ? jdur->child : NULL; d && n_durations < MYNAH_ASR_MAX_DURATIONS; d = d->next)
            durations[n_durations++] = d->valueint;
        const int blank_id = jint(jdec, "blank_id", &bad);
        const int max_sym = jint(jdec, "max_symbols_per_step", &bad);
        if (bad) goto fail;
        if (mynah_asr_decoder_init(&m->dec, m->weights, blank_id, max_sym,
                               quant, durations, n_durations) != 0) {
            fprintf(stderr, "mynah-asr: decoder init failed\n");
            goto fail;
        }
        mynah_asr_ctc_init(&m->ctc, m->weights);   /* head ausiliaria hybrid, opzionale */
        m->engine_dflt = &ENG_RNNT;
    }
    m->engine = m->engine_dflt;

    snprintf(path, sizeof(path), "%s/tokens.json", model_dir);
    if (mynah_asr_tokenizer_load(&m->tok, path) != 0) goto fail;

    /* feature config */
    const cJSON *jf = jneed(m->cfg, "features", &bad);
    const mynah_asr_tensor *fb = mynah_asr_st_get(m->mel_filters, "mel_fb");
    const mynah_asr_tensor *win = mynah_asr_st_get(m->mel_filters, "window");
    if (!fb || !win) goto fail;
    const cJSON *jnorm = jf ? cJSON_GetObjectItem(jf, "normalize") : NULL;
    m->feat = (mynah_asr_feat_cfg){
        .sample_rate = jint(jf, "sample_rate", &bad),
        .n_mels = jint(jf, "n_mels", &bad),
        .n_fft = jint(jf, "n_fft", &bad),
        .win_length = jint(jf, "win_length", &bad),
        .hop_length = jint(jf, "hop_length", &bad),
        .preemphasis = jnum(jf, "preemphasis", &bad),
        .log_zero_guard = jnum(jf, "log_zero_guard", &bad),
        .normalize_per_feature = jnorm && cJSON_IsString(jnorm) &&
                                 strcmp(jnorm->valuestring, "per_feature") == 0,
        .mel_fb = (const float *)fb->data,
        .window = (const float *)win->data,
    };
    if (bad || m->feat.sample_rate <= 0 || m->feat.hop_length <= 0) goto fail;

    const cJSON *jsf = jenc ? cJSON_GetObjectItem(jenc, "subsampling_factor") : NULL;
    m->frame_sec = (double)m->feat.hop_length * (jsf ? jsf->valueint : 8)
                   / (double)m->feat.sample_rate;
    m->seg_sec = 0.0;   /* risolto DOPO il parse della sezione streaming */

    /* streaming presets [[left, right], ...] — the section is absent for offline
     * models (Parakeet): full attention [-1,-1], no stream API */
    m->left_ctx = -1;
    m->default_right = -1;
    const cJSON *js = cJSON_GetObjectItem(m->cfg, "streaming");
    if (js) {
        const cJSON *presets = jneed(js, "att_context_presets", &bad);
        const int def = jint(js, "default_preset_index", &bad);
        if (bad) goto fail;
        int i = 0;
        for (cJSON *p = presets->child; p && i < 8; p = p->next, i++) {
            const cJSON *pl = cJSON_GetArrayItem(p, 0), *pr = cJSON_GetArrayItem(p, 1);
            if (!pl || !pr) { fprintf(stderr, "mynah-asr: preset streaming malformato\n"); goto fail; }
            m->lookaheads[i] = pr->valueint;
            if (i == def) {
                m->left_ctx = pl->valueint;
                m->default_right = m->lookaheads[i];
            }
        }
        m->n_lookaheads = i;
    }
    /* model-aware segmentation default (see the comment on MYNAH_ASR_SEG_OFFLINE) */
    m->seg_sec = m->n_lookaheads > 0 ? MYNAH_ASR_SEG_DEFAULT : MYNAH_ASR_SEG_OFFLINE;

    /* prompt (Nemotron): absent in models with implicit LID (Parakeet);
     * for AED (Canary) the prompt section is the DECODER's (no default_id) */
    const cJSON *jprompt = cJSON_GetObjectItem(m->cfg, "prompt");
    const cJSON *jdid = jprompt ? cJSON_GetObjectItem(jprompt, "default_id") : NULL;
    m->default_prompt = jdid ? jdid->valueint : -1;
    if (m->is_aed) {
        const cJSON *jeos = cJSON_GetObjectItem(jdec, "eos_token");
        m->aed_eos = jeos ? mynah_asr_tok_find(&m->tok, jeos->valuestring) : -1;
        if (m->aed_eos < 0) { fprintf(stderr, "mynah-asr: AED EOS not found\n"); goto fail; }
        /* generative <|N|> timestamps: not every AED supports them (v2 uses an
         * external aligner) — capability read from mynah.json, default yes */
        const cJSON *jts = cJSON_GetObjectItem(jdec, "timestamp_tokens");
        m->aed_ts = !(jts && cJSON_IsFalse(jts));
    }
    return m;

fail:
    mynah_asr_free(m);
    return NULL;
}

void mynah_asr_free(mynah_asr_model *m) {
    if (!m) return;
    mynah_asr_qmat_free(&m->dec.head);
    mynah_asr_aed_free(&m->aed);
    mynah_asr_encoder_free(&m->enc);
    mynah_asr_tokenizer_free(&m->tok);
#ifdef MYNAH_ASR_METAL
    /* the cached GPU weights point into the mmap we are about to close */
    mynah_asr_metal_weights_evict();
#endif
    mynah_asr_st_close(m->weights);
    mynah_asr_st_close(m->mel_filters);
    cJSON_Delete(m->cfg);
    free(m);
}

int mynah_asr_lang_id(const mynah_asr_model *m, const char *lang) {
    const cJSON *jprompt = cJSON_GetObjectItem(m->cfg, "prompt");
    if (!jprompt) return -1;
    const cJSON *dict = cJSON_GetObjectItem(jprompt, "dictionary");
    const cJSON *e = dict ? cJSON_GetObjectItem(dict, lang) : NULL;
    return e ? e->valueint : -1;
}

void mynah_asr_set_segment_limit(mynah_asr_model *m, double sec) {
    const double def = m->n_lookaheads > 0 ? MYNAH_ASR_SEG_DEFAULT : MYNAH_ASR_SEG_OFFLINE;
    m->seg_sec = sec <= 0.0 ? def : (sec < 5.0 ? 5.0 : sec);
}

int mynah_asr_set_decoder(mynah_asr_model *m, const char *name) {
    if (strcmp(name, "ctc") == 0) {
        if (!m->ctc.w) {
            fprintf(stderr, "mynah-asr: this model has no CTC head\n");
            return -1;
        }
        m->engine = &ENG_CTC;
        return 0;
    }
    m->engine = m->engine_dflt;   /* CTC puro: il default È già la head CTC */
    return strcmp(name, "default") == 0 ? 0 : -1;
}

/* prompt id for the request: -1 = model without prompt (valid, e.g. Parakeet),
 * -2 = unknown language for a model WITH a prompt (an error). */
static int resolve_prompt(const mynah_asr_model *m, const char *lang) {
    if (m->default_prompt < 0) return -1;
    if (!lang) return m->default_prompt;
    const int id = mynah_asr_lang_id(m, lang);
    return id < 0 ? -2 : id;
}

static int aed_build_prompt(const mynah_asr_model *m, const char *lang, int *ids, int want_ts);


int mynah_asr_lookaheads(const mynah_asr_model *m, int out[8]) {
    memcpy(out, m->lookaheads, sizeof(m->lookaheads));
    return m->n_lookaheads;
}

/* per-item batch worker (features and decode are independent across items:
 * disjoint regions, read-only model -> parallel_for is safe) */
typedef struct {
    mynah_asr_model *m;
    const float *const *samples;
    const size_t *n_samples;
    const char *const *langs;
    float **feats, **encs;
    int *valids, *t_encs;
    char **texts;
    char (*langs_out)[16];
    mynah_asr_word **words;    /* per-item, NULL = niente timestamp */
    int *n_words;
} batch_ctx;

static void batch_feat_worker(void *ctx, int b) {
    batch_ctx *c = ctx;
    int T_mel;
    c->feats[b] = mynah_asr_log_mel(&c->m->feat, c->samples[b], c->n_samples[b],
                                &T_mel, &c->valids[b]);
}

static void batch_decode_worker(void *ctx, int b) {
    batch_ctx *c = ctx;
    mynah_asr_model *m = c->m;
    if (!c->encs[b]) return;
    /* the batched encoder produces the POST-projector output: the CTC engine
     * (which wants the raw one) is valid only when post == raw (pure CTC, no
     * projector) */
    const mynah_asr_engine *eng = m->engine;
    if (eng->raw_encoder && m->ctc.d_in != m->enc.d_out)
        eng = m->engine_dflt;
    /* words in the batch ONLY for frame-based engines (RNNT/TDT/CTC): the
     * emission frames are a side channel that does not change the tokens. For AED
     * want_ts changes the PROMPT (<|N|> mode) and therefore the text: inside a
     * batch — where json and verbose requests coexist — that would stay
     * non-deterministic, so no AED words here (the non-batch path has them, on
     * request). */
    const int want_ts = c->words != NULL && !m->is_aed;
    int *tokens = NULL, *frames = NULL;
    const int n_tok = eng->decode(m, c->encs[b], c->t_encs[b],
                                  c->langs ? c->langs[b] : NULL, want_ts,
                                  &tokens, &frames);
    if (n_tok >= 0)
        c->texts[b] = mynah_asr_detokenize(&m->tok, tokens, n_tok,
                                       c->langs_out ? c->langs_out[b] : NULL);
    if (n_tok >= 0 && c->texts[b] && c->words && frames)
        mynah_asr_detokenize_words(&m->tok, tokens, frames, n_tok, m->frame_sec,
                               &c->words[b], &c->n_words[b]);
    free(tokens);
    free(frames);
}

int mynah_asr_transcribe_batch(mynah_asr_model *m, const float *const *samples,
                           const size_t *n_samples, int batch, const char *const *langs,
                           int lookahead, char **texts, char (*langs_out)[16]) {
    return mynah_asr_transcribe_batch_ts(m, samples, n_samples, batch, langs, lookahead,
                                     texts, langs_out, NULL, NULL);
}

int mynah_asr_transcribe_batch_ts(mynah_asr_model *m, const float *const *samples,
                              const size_t *n_samples, int batch, const char *const *langs,
                              int lookahead, char **texts, char (*langs_out)[16],
                              mynah_asr_word **words, int *n_words) {
    const int right = lookahead >= 0 ? lookahead : m->default_right;
    int rc = -1;

    float **feats = calloc((size_t)batch, sizeof(float *));
    float **encs = calloc((size_t)batch, sizeof(float *));
    int *valids = calloc((size_t)batch, sizeof(int));
    int *prompts = calloc((size_t)batch, sizeof(int));
    int *t_encs = calloc((size_t)batch, sizeof(int));
    if (!feats || !encs || !valids || !prompts || !t_encs) goto done;

    batch_ctx c = {.m = m, .samples = samples, .n_samples = n_samples, .langs = langs,
                   .feats = feats, .encs = encs, .valids = valids, .t_encs = t_encs,
                   .texts = texts, .langs_out = langs_out,
                   .words = words, .n_words = n_words};

    for (int b = 0; b < batch; b++) {
        texts[b] = NULL;
        if (words) { words[b] = NULL; n_words[b] = 0; }
        prompts[b] = resolve_prompt(m, langs ? langs[b] : NULL);
        if (prompts[b] == -2) goto done;
    }
    mynah_asr_parallel_for(batch, batch_feat_worker, &c);
    for (int b = 0; b < batch; b++)
        if (!feats[b]) goto done;

    if (mynah_asr_encoder_forward_batch(&m->enc, (const float *const *)feats, valids, batch,
                                    m->feat.n_mels, prompts, m->left_ctx, right,
                                    encs, t_encs) != 0)
        goto done;

    mynah_asr_parallel_for(batch, batch_decode_worker, &c);
    rc = 0;

done:
    for (int b = 0; b < batch; b++) {
        if (feats) free(feats[b]);
        if (encs) free(encs[b]);
    }
    free(feats); free(encs); free(valids); free(prompts); free(t_encs);
    return rc;
}

/* ----------------------------------------------------------------- streaming */

struct mynah_asr_stream {
    mynah_asr_model *m;
    mynah_asr_mel_stream mel;
    mynah_asr_enc_stream es;
    mynah_asr_dec_state dec;
    int prompt;
    float *mel_buf;             /* buffer del chunk mel corrente */
    int mel_have;
    float *enc_buf;             /* [q, d_out] output encoder per chunk */
    int *tokens;
    int n_tokens, cap_tokens;
    size_t chars_emitted;       /* byte di testo già consegnati alla callback */
    char lang[16];
    size_t samples_fed;
};

mynah_asr_stream *mynah_asr_stream_open(mynah_asr_model *m, const char *lang, int lookahead) {
    if (m->n_lookaheads == 0) {
        fprintf(stderr, "mynah-asr: this model is offline-only (no cache-aware streaming)\n");
        return NULL;
    }
    const int prompt = resolve_prompt(m, lang);
    if (prompt == -2) { fprintf(stderr, "mynah-asr: language '%s' is not supported\n", lang); return NULL; }
    const int right = lookahead >= 0 ? lookahead : m->default_right;

    mynah_asr_stream *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->m = m;
    s->prompt = prompt;
    if (mynah_asr_mel_stream_init(&s->mel, &m->feat) != 0 ||
        mynah_asr_enc_stream_init(&s->es, &m->enc, m->left_ctx, right, m->feat.n_mels) != 0) {
        mynah_asr_stream_close(s);
        return NULL;
    }
    mynah_asr_dec_state_reset(&m->dec, &s->dec);
    const int max_chunk = 8 * (right + 1) + 1;
    s->mel_buf = malloc((size_t)max_chunk * (size_t)m->feat.n_mels * sizeof(float));
    s->enc_buf = malloc((size_t)s->es.q * (size_t)m->enc.d_out * sizeof(float));
    s->cap_tokens = 4096;
    s->tokens = malloc((size_t)s->cap_tokens * sizeof(int));
    if (!s->mel_buf || !s->enc_buf || !s->tokens) { mynah_asr_stream_close(s); return NULL; }
    return s;
}

void mynah_asr_stream_close(mynah_asr_stream *s) {
    if (!s) return;
    mynah_asr_mel_stream_free(&s->mel);
    mynah_asr_enc_stream_free(&s->es);
    free(s->mel_buf); free(s->enc_buf); free(s->tokens);
    free(s);
}

const char *mynah_asr_stream_lang(const mynah_asr_stream *s) { return s->lang; }

/* Encode the current mel chunk, decode it, emit the text delta. */
static int stream_flush_chunk(mynah_asr_stream *s, int n_mel, int is_last,
                              mynah_asr_result_cb cb, void *ud) {
    mynah_asr_model *m = s->m;
    const int q = mynah_asr_enc_stream_step(&s->es, s->mel_buf, n_mel, m->feat.n_mels,
                                        s->prompt, is_last, s->enc_buf);
    if (q < 0) return -1;

    if (s->n_tokens + q * m->dec.max_symbols > s->cap_tokens) {
        s->cap_tokens = (s->cap_tokens + q * m->dec.max_symbols) * 2;
        int *nb = realloc(s->tokens, (size_t)s->cap_tokens * sizeof(int));
        if (!nb) return -1;
        s->tokens = nb;
    }
    s->n_tokens += mynah_asr_greedy_decode(&m->dec, &s->dec, s->enc_buf, q,
                                       s->tokens + s->n_tokens, NULL,
                                       s->cap_tokens - s->n_tokens);
    s->mel_have = 0;

    if (cb) {
        char lang_tmp[16] = "";
        char *text = mynah_asr_detokenize(&m->tok, s->tokens, s->n_tokens, lang_tmp);
        if (!text) return -1;
        if (lang_tmp[0]) memcpy(s->lang, lang_tmp, sizeof(s->lang));
        const size_t total = strlen(text);
        if (total > s->chars_emitted) {
            const double t1 = (double)s->samples_fed / (double)m->feat.sample_rate;
            mynah_asr_result res = {
                .text = text + s->chars_emitted,
                .t0 = 0.0, .t1 = t1,
                .is_final = true,
                .lang = s->lang[0] ? s->lang : NULL,
            };
            cb(&res, ud);
            s->chars_emitted = total;
        }
        free(text);
    }
    return 0;
}

int mynah_asr_stream_feed(mynah_asr_stream *s, const float *samples, size_t n,
                      mynah_asr_result_cb cb, void *ud) {
    mynah_asr_model *m = s->m;
    s->samples_fed += n;
    const float *src = samples;
    size_t left = n;
    int first_pass = 1;

    for (;;) {
        const int need = mynah_asr_enc_stream_need(&s->es);
        const int got = mynah_asr_mel_stream_feed(&s->mel, first_pass ? src : NULL,
                                              first_pass ? left : 0,
                                              s->mel_buf + (size_t)s->mel_have * (size_t)m->feat.n_mels,
                                              need - s->mel_have);
        first_pass = 0;
        s->mel_have += got;
        if (s->mel_have < need) break;          /* servono altri campioni */
        if (stream_flush_chunk(s, need, 0, cb, ud) != 0) return -1;
    }
    return 0;
}

int mynah_asr_stream_finish(mynah_asr_stream *s, mynah_asr_result_cb cb, void *ud) {
    mynah_asr_model *m = s->m;
    for (;;) {
        const int need = mynah_asr_enc_stream_need(&s->es);
        const int got = mynah_asr_mel_stream_finish(&s->mel,
                                                s->mel_buf + (size_t)s->mel_have * (size_t)m->feat.n_mels,
                                                need - s->mel_have);
        s->mel_have += got;
        if (s->mel_have == 0) break;
        if (s->mel_have < need) {
            /* tail: short chunk with causal right pad (is_last) — same as offline */
            if (stream_flush_chunk(s, s->mel_have, 1, cb, ud) != 0) return -1;
            break;
        }
        /* full chunk: is_last only when the mel stream has nothing left after it */
        if (stream_flush_chunk(s, need, 0, cb, ud) != 0) return -1;
    }
    return 0;
}

/* canary2 prompt -> global ids (template from mynah.json, tokens as strings;
 * empty slots = 0 tokens). lang is the source ("auto"/NULL -> "en"); the
 * "src>tgt" form (e.g. "en>de") requests translation PER CALL (thread-safe, used
 * by the server); otherwise the target comes from mynah_asr_set_target_lang
 * ("" = same as source). want_ts: timestamp slot active (the model brackets every
 * word with <|N|>). Returns the token count, -1 = unknown language. */
static int aed_build_prompt(const mynah_asr_model *m, const char *lang, int *ids, int want_ts) {
    char src[8] = "en", tgt_buf[8] = "";
    if (lang && *lang && strncmp(lang, "auto", 4) != 0) {
        int i = 0;
        for (; lang[i] && lang[i] != '-' && lang[i] != '_' && lang[i] != '>' && i < 7; i++)
            src[i] = (char)(lang[i] >= 'A' && lang[i] <= 'Z' ? lang[i] + 32 : lang[i]);
        src[i] = '\0';
    }
    const char *gt = lang ? strchr(lang, '>') : NULL;
    if (gt && gt[1]) {
        int i = 0;
        for (gt++; gt[i] && gt[i] != '-' && gt[i] != '_' && i < 7; i++)
            tgt_buf[i] = (char)(gt[i] >= 'A' && gt[i] <= 'Z' ? gt[i] + 32 : gt[i]);
        tgt_buf[i] = '\0';
    }
    const char *tgt = tgt_buf[0] ? tgt_buf : (m->aed_target[0] ? m->aed_target : src);
    const cJSON *jp = cJSON_GetObjectItem(m->cfg, "prompt");
    const cJSON *jl = cJSON_GetObjectItem(jp, "languages");
    int ok_src = 0, ok_tgt = 0;
    for (const cJSON *e = jl ? jl->child : NULL; e; e = e->next) {
        ok_src |= strcmp(e->valuestring, src) == 0;
        ok_tgt |= strcmp(e->valuestring, tgt) == 0;
    }
    if (!ok_src || !ok_tgt) {
        fprintf(stderr, "mynah-asr: language '%s' is not supported by the model\n",
                ok_src ? tgt : src);
        return -1;
    }
    const cJSON *jd = cJSON_GetObjectItem(jp, "defaults");
    const cJSON *jt = jp ? cJSON_GetObjectItem(jp, "template") : NULL;
    if (!jt) { fprintf(stderr, "mynah-asr: missing prompt.template\n"); return -1; }
    int n = 0;
    for (const cJSON *e = jt->child; e && n < MYNAH_ASR_AED_PROMPT_MAX; e = e->next) {
        if (!cJSON_IsString(e)) continue;
        const char *item = e->valuestring;
        char tok[32];
        if (item[0] == '{') {
            char slot[24];
            snprintf(slot, sizeof(slot), "%.*s", (int)strlen(item) - 2, item + 1);
            if (strcmp(slot, "source_lang") == 0) snprintf(tok, sizeof(tok), "<|%s|>", src);
            else if (strcmp(slot, "target_lang") == 0) snprintf(tok, sizeof(tok), "<|%s|>", tgt);
            else if (want_ts && strcmp(slot, "timestamp") == 0)
                snprintf(tok, sizeof(tok), "<|timestamp|>");
            else {
                const cJSON *dv = cJSON_GetObjectItem(jd, slot);
                if (!dv || !cJSON_IsString(dv) || !dv->valuestring[0]) continue;
                snprintf(tok, sizeof(tok), "%s", dv->valuestring);
            }
        } else {
            snprintf(tok, sizeof(tok), "%s", item);
        }
        const int id = mynah_asr_tok_find(&m->tok, tok);
        if (id < 0) { fprintf(stderr, "mynah-asr: token prompt '%s' ignoto\n", tok); return -1; }
        ids[n++] = id;
    }
    return n;
}

int mynah_asr_can_translate(const mynah_asr_model *m) { return m->is_aed; }

int mynah_asr_set_target_lang(mynah_asr_model *m, const char *lang) {
    if (!lang || !*lang) { m->aed_target[0] = '\0'; return 0; }
    if (!m->is_aed) {
        fprintf(stderr, "mynah-asr: this model does not support translation (AED only)\n");
        return -1;
    }
    const cJSON *jl = cJSON_GetObjectItem(cJSON_GetObjectItem(m->cfg, "prompt"), "languages");
    for (const cJSON *e = jl ? jl->child : NULL; e; e = e->next)
        if (strcmp(e->valuestring, lang) == 0) {
            snprintf(m->aed_target, sizeof(m->aed_target), "%s", lang);
            return 0;
        }
    fprintf(stderr, "mynah-asr: target language '%s' is not supported\n", lang);
    return -1;
}

/* Piece "<|N|>" (digits only) -> N, otherwise -1. Overflow clamp: no real audio
 * exceeds ~10^7 frames (9 days), a hostile tokens.json can. */
static int aed_ts_frame(const char *piece) {
    if (strncmp(piece, "<|", 2) != 0) return -1;
    const char *p = piece + 2;
    if (*p < '0' || *p > '9') return -1;
    int v = 0;
    for (; *p >= '0' && *p <= '9'; p++) {
        if (v > 10000000) return -1;
        v = v * 10 + (*p - '0');
    }
    return strcmp(p, "|>") == 0 ? v : -1;
}

/* Timestamped words from AED tokens in the canary format: <|t0|> pieces <|t1|>
 * per word (consecutive <|N|> close the previous word and open the next one).
 * 0 = ok. */
static int aed_words_from_tokens(const mynah_asr_tokenizer *tk, const int *toks, int n,
                                 double frame_sec, mynah_asr_word **out, int *n_out) {
    *out = NULL;
    *n_out = 0;
    mynah_asr_word *ws = malloc((size_t)(n + 1) * sizeof(*ws));
    char buf[256];
    size_t blen = 0;
    int nw = 0, open_ts = -1, word_open = 0;
    if (!ws) return -1;

    for (int i = 0; i < n; i++) {
        if (toks[i] < 0 || toks[i] >= tk->n_pieces) continue;
        const char *piece = tk->pieces[toks[i]];
        const int ts = aed_ts_frame(piece);
        if (ts >= 0) {
            if (word_open && blen > 0) {          /* chiude la parola corrente */
                buf[blen] = '\0';
                ws[nw].word = strdup(buf[0] == ' ' ? buf + 1 : buf);
                ws[nw].t0 = (open_ts >= 0 ? open_ts : 0) * frame_sec;
                ws[nw].t1 = ts * frame_sec;
                if (ws[nw].word) nw++;
                blen = 0;
                word_open = 0;
            } else {
                open_ts = ts;                     /* apre la prossima */
            }
            continue;
        }
        if (piece[0] == '<') continue;            /* altri speciali */
        if (strncmp(piece, "\xe2\x96\x81", 3) == 0) {   /* ▁: nuova parola */
            if (word_open && blen > 0) {          /* senza ts di chiusura: chiudi */
                buf[blen] = '\0';
                ws[nw].word = strdup(buf[0] == ' ' ? buf + 1 : buf);
                ws[nw].t0 = (open_ts >= 0 ? open_ts : 0) * frame_sec;
                ws[nw].t1 = ws[nw].t0;
                if (ws[nw].word) nw++;
                blen = 0;
            }
            word_open = 1;
            if (blen < sizeof(buf) - 2) { buf[blen++] = ' '; }
            piece += 3;
        } else if (!word_open) {
            word_open = 1;                        /* continuazione senza ▁ */
        }
        const size_t pl = strlen(piece);
        if (blen + pl < sizeof(buf) - 1) { memcpy(buf + blen, piece, pl); blen += pl; }
    }
    if (word_open && blen > 0) {
        buf[blen] = '\0';
        ws[nw].word = strdup(buf[0] == ' ' ? buf + 1 : buf);
        ws[nw].t0 = (open_ts >= 0 ? open_ts : 0) * frame_sec;
        ws[nw].t1 = ws[nw].t0;
        if (ws[nw].word) nw++;
    }
    if (nw == 0) { free(ws); return 0; }
    *out = ws;
    *n_out = nw;
    return 0;
}

/* Transcription of ONE segment (the whole audio, or a slice between silences). */
static char *transcribe_segment(mynah_asr_model *m, const float *samples, size_t n_samples,
                                int prompt, int right, const char *lang, char *lang_out,
                                mynah_asr_word **words, int *n_words) {
    if (words) { *words = NULL; *n_words = 0; }
    int T_mel, valid;
    float *feats = mynah_asr_log_mel(&m->feat, samples, n_samples, &T_mel, &valid);
    if (!feats) return NULL;

    const mynah_asr_engine *eng = m->engine;
    int T_enc;
    float *enc = eng->raw_encoder
        ? mynah_asr_encoder_forward_raw(&m->enc, feats, valid, m->feat.n_mels,
                                    m->left_ctx, right, &T_enc)
        : mynah_asr_encoder_forward(&m->enc, feats, valid, m->feat.n_mels, prompt,
                                m->left_ctx, right, &T_enc);
    free(feats);
    if (!enc) return NULL;

    int *tokens = NULL, *frames = NULL;
    const int n_tok = eng->decode(m, enc, T_enc, lang, words != NULL,
                                  &tokens, &frames);
    free(enc);
    if (n_tok < 0) { free(tokens); free(frames); return NULL; }

    char *text = mynah_asr_detokenize(&m->tok, tokens, n_tok, lang_out);
    if (text && words) {
        if (frames)
            mynah_asr_detokenize_words(&m->tok, tokens, frames, n_tok, m->frame_sec,
                                   words, n_words);
        else if (m->is_aed && m->aed_ts)   /* AED: tempi nei token <|N|> */
            aed_words_from_tokens(&m->tok, tokens, n_tok, m->frame_sec,
                                  words, n_words);
    }
    free(tokens);
    free(frames);
    return text;
}

/* Split point: energy minimum (RMS over 20 ms windows) in [lo, hi). */
static size_t find_split_point(const float *s, size_t lo, size_t hi, int sr) {
    const size_t win = (size_t)sr / 50, hop = win / 2;
    double best = 1e300;
    size_t best_pos = hi;
    for (size_t p = lo; p + win <= hi; p += hop) {
        double e = 0.0;
        for (size_t i = 0; i < win; i++) {
            const double v = s[p + i];
            e += v * v;
        }
        if (e < best) { best = e; best_pos = p + win / 2; }
    }
    return best_pos;
}

char *mynah_asr_transcribe_ts(mynah_asr_model *m, const float *samples, size_t n_samples,
                          const char *lang, int lookahead, char *lang_out,
                          mynah_asr_word **words, int *n_words) {
    if (words) { *words = NULL; *n_words = 0; }
    if (lang_out) lang_out[0] = '\0';
    const int prompt = resolve_prompt(m, lang);
    if (prompt == -2) { fprintf(stderr, "mynah-asr: language '%s' is not supported\n", lang); return NULL; }
    const int right = lookahead >= 0 ? lookahead : m->default_right;

    const int sr = m->feat.sample_rate;
    const size_t seg_max = (size_t)(m->seg_sec * sr);
    if (n_samples <= seg_max + seg_max / 10)   /* +10%: non spezzare per poco */
        return transcribe_segment(m, samples, n_samples, prompt, right, lang, lang_out,
                                  words, n_words);

    /* long audio: independent segments split on silence, results concatenated */
    char *text = NULL;
    size_t text_len = 0;
    size_t cur = 0;
    while (cur < n_samples) {
        size_t end = n_samples;
        if (n_samples - cur > seg_max + seg_max / 10) {
            /* look for silence in the last 20 s of the window (at least 1 s in) */
            const size_t hi = cur + seg_max;
            size_t lo = seg_max > 20u * (size_t)sr ? hi - 20u * (size_t)sr : cur + (size_t)sr;
            if (lo <= cur) lo = cur + 1;
            end = find_split_point(samples, lo, hi, sr);
        }

        char seg_lang[16] = "";
        mynah_asr_word *sw = NULL;
        int sn = 0;
        char *seg = transcribe_segment(m, samples + cur, end - cur, prompt, right,
                                       lang, seg_lang, words ? &sw : NULL, &sn);
        if (!seg) { free(text); if (words) mynah_asr_words_free(*words, *n_words); return NULL; }
        if (lang_out && seg_lang[0] && cur == 0) memcpy(lang_out, seg_lang, sizeof(seg_lang));

        const size_t sl = strlen(seg);
        char *nt = realloc(text, text_len + sl + 2);
        if (!nt) {
            free(seg); free(text); mynah_asr_words_free(sw, sn);
            if (words) { mynah_asr_words_free(*words, *n_words); *words = NULL; *n_words = 0; }
            return NULL;
        }
        text = nt;
        if (text_len > 0 && sl > 0) text[text_len++] = ' ';
        memcpy(text + text_len, seg, sl + 1);
        text_len += sl;
        free(seg);

        if (words && sn > 0) {
            mynah_asr_word *nw = realloc(*words, ((size_t)*n_words + (size_t)sn) * sizeof(mynah_asr_word));
            if (!nw) {
                mynah_asr_words_free(sw, sn); free(text);
                /* a failed realloc does NOT free the old block */
                mynah_asr_words_free(*words, *n_words); *words = NULL; *n_words = 0;
                return NULL;
            }
            *words = nw;
            const double off = (double)cur / sr;
            for (int i = 0; i < sn; i++) {
                nw[*n_words + i] = sw[i];
                nw[*n_words + i].t0 += off;
                nw[*n_words + i].t1 += off;
            }
            *n_words += sn;
            free(sw);   /* le stringhe sono state trasferite */
        } else if (sw) {
            mynah_asr_words_free(sw, sn);
        }
        cur = end;
    }
    return text ? text : calloc(1, 1);
}

char *mynah_asr_transcribe(mynah_asr_model *m, const float *samples, size_t n_samples,
                       const char *lang, int lookahead, char *lang_out) {
    return mynah_asr_transcribe_ts(m, samples, n_samples, lang, lookahead, lang_out,
                               NULL, NULL);
}
