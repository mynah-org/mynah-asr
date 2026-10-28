/* Recovering an EMPTY streaming turn on the CPU library (recover.h). */
#include "recover.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mynah_asr_recoverer {
    mynah_asr_model *m;
    mynah_asr_stream *s;
    mynah_asr_recover_cfg cfg;
    int frame_samples;
    float *buf;                 /* [longest prefix + max turn] */
    size_t buf_cap, max_turn;
    char *texts[MYNAH_ASR_RECOVER_MAX_ATTEMPTS];
    char langs[MYNAH_ASR_RECOVER_MAX_ATTEMPTS][16];
    int winner;
    int lag;                    /* TEXT's window in frames: cfg, or the stream's chunk */
};

mynah_asr_recoverer *mynah_asr_recoverer_open(mynah_asr_model *m, int lookahead,
                                              const mynah_asr_recover_cfg *cfg) {
    mynah_asr_recoverer *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->m = m;
    r->cfg = *cfg;
    r->frame_samples = mynah_asr_frame_samples(m);
    r->winner = -1;
    int longest = cfg->replay_frames;
    for (int i = 0; i < cfg->n_phases; i++)
        if (cfg->phase_frames[i] > longest) longest = cfg->phase_frames[i];
    r->max_turn = (size_t)(cfg->max_turn_s * (double)mynah_asr_sample_rate(m));
    r->buf_cap = (size_t)longest * (size_t)r->frame_samples + r->max_turn;
    r->buf = malloc(r->buf_cap * sizeof(float));
    r->s = mynah_asr_stream_open(m, "auto", lookahead);
    if (!r->buf || !r->s) { mynah_asr_recoverer_close(r); return NULL; }
    r->lag = cfg->lag_frames > 0 ? cfg->lag_frames : mynah_asr_stream_chunk_frames(r->s);
    return r;
}

void mynah_asr_recoverer_close(mynah_asr_recoverer *r) {
    if (!r) return;
    for (int i = 0; i < MYNAH_ASR_RECOVER_MAX_ATTEMPTS; i++) free(r->texts[i]);
    mynah_asr_stream_close(r->s);
    free(r->buf);
    free(r);
}

const char *mynah_asr_recover_text(const mynah_asr_recoverer *r) {
    return (r->winner >= 0 && r->texts[r->winner]) ? r->texts[r->winner] : "";
}

int mynah_asr_recover_turn(mynah_asr_recoverer *r, const char *lang, const float *cur,
                           size_t n_cur, const float *prev, size_t n_prev,
                           const char *prev_text, mynah_asr_recover_result *res) {
    memset(res, 0, sizeof(*res));
    res->winner = r->winner = -1;
    for (int i = 0; i < MYNAH_ASR_RECOVER_MAX_ATTEMPTS; i++) { free(r->texts[i]); r->texts[i] = NULL; }
    if (n_cur == 0 || n_cur > r->max_turn) return 0;

    mynah_asr_attempt plan[MYNAH_ASR_RECOVER_MAX_ATTEMPTS];
    const int n = mynah_asr_recover_plan(&r->cfg, prev ? n_prev : 0, r->frame_samples, plan);
    const mynah_asr_tokenizer *tk = mynah_asr_model_tokenizer(r->m);
    for (int a = 0; a < n; a++) {
        const size_t len = mynah_asr_recover_build(&plan[a], r->frame_samples, cur, n_cur,
                                                   prev, n_prev, r->buf);
        if (mynah_asr_stream_reset(r->s, lang) != 0) return -1;
        if (mynah_asr_stream_feed(r->s, r->buf, len, NULL, NULL) != 0 ||
            mynah_asr_stream_finish(r->s, NULL, NULL) != 0) return -1;
        const int *tok, *fr;
        const int nt = mynah_asr_stream_tokens(r->s, &tok, &fr);
        const int k = mynah_asr_replay_keep_from(tk, tok, fr, nt, plan[a].prefix_frames,
                                                 r->cfg.rule, prev_text, r->lag);
        char *text = mynah_asr_detokenize(tk, tok + k, nt - k, NULL);
        if (!text) return -1;
        snprintf(r->langs[a], sizeof(r->langs[a]), "%s", mynah_asr_stream_lang(r->s));
        if (mynah_asr_recovery_gate(text, lang, r->langs[a], prev_text) != MYNAH_ASR_GATE_OK)
            text[0] = '\0';   /* a rejected attempt keeps no vote */
        r->texts[a] = text;
        res->attempts++;
    }
    const int w = mynah_asr_recover_vote((const char *const *)r->texts, n, r->cfg.min_votes);
    if (w < 0) return 0;
    r->winner = res->winner = w;
    res->accepted = 1;
    res->winner_kind = plan[w].kind;
    for (int a = 0; a < n; a++)
        res->votes += mynah_asr_text_has_content(r->texts[a]) &&
                      mynah_asr_text_same_words(r->texts[w], r->texts[a]);
    snprintf(res->lang, sizeof(res->lang), "%s", r->langs[w]);
    return 0;
}
