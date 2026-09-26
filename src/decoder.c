#include "decoder.h"

#include "backend.h"   /* the f32 seam: mynah_asr_gemm_f32, mynah_asr_gemv_f32 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int mynah_asr_decoder_init(mynah_asr_decoder *dec, const mynah_asr_safetensors *st,
                       int blank, int max_symbols, int quantize,
                       const int *durations, int n_durations) {
    memset(dec, 0, sizeof(*dec));
    if (n_durations > MYNAH_ASR_MAX_DURATIONS) return -1;
    dec->n_durations = n_durations;
    for (int i = 0; i < n_durations; i++) dec->durations[i] = durations[i];
    const mynah_asr_tensor *emb = mynah_asr_st_get(st, "decoder.embedding.weight");
    const mynah_asr_tensor *proj_w = mynah_asr_st_get(st, "decoder.decoder_projector.weight");
    const mynah_asr_tensor *proj_b = mynah_asr_st_get(st, "decoder.decoder_projector.bias");
    const mynah_asr_tensor *head_b = mynah_asr_st_get(st, "joint.head.bias");
    if (!emb || !proj_w || !proj_b || !head_b) return -1;

    dec->embedding = (const float *)emb->data;
    dec->proj_w = (const float *)proj_w->data;
    dec->proj_b = (const float *)proj_b->data;
    dec->head_b = (const float *)head_b->data;
    dec->vocab = (int)emb->shape[0];
    dec->hidden = (int)emb->shape[1];
    if (mynah_asr_qmat_init_st(&dec->head, st, "joint.head.weight", quantize) != 0) return -1;
    dec->blank = blank;
    dec->max_symbols = max_symbols;

    int n = 0;
    for (; n < MYNAH_ASR_MAX_PRED_LAYERS; n++) {
        char name[64];
        snprintf(name, sizeof(name), "decoder.lstm.weight_ih_l%d", n);
        const mynah_asr_tensor *w = mynah_asr_st_get(st, name);
        if (!w) break;
        dec->w_ih[n] = (const float *)w->data;
        /* partial layer (corrupt checkpoint): clean error instead of a null deref */
        const mynah_asr_tensor *whh, *bih, *bhh;
        snprintf(name, sizeof(name), "decoder.lstm.weight_hh_l%d", n);
        whh = mynah_asr_st_get(st, name);
        snprintf(name, sizeof(name), "decoder.lstm.bias_ih_l%d", n);
        bih = mynah_asr_st_get(st, name);
        snprintf(name, sizeof(name), "decoder.lstm.bias_hh_l%d", n);
        bhh = mynah_asr_st_get(st, name);
        if (!whh || !bih || !bhh) {
            fprintf(stderr, "mynah-asr: LSTM layer %d is incomplete in the checkpoint\n", n);
            return -1;
        }
        dec->w_hh[n] = (const float *)whh->data;
        dec->b_ih[n] = (const float *)bih->data;
        dec->b_hh[n] = (const float *)bhh->data;
    }
    dec->n_layers = n;
    return (n > 0 && dec->hidden <= 1024) ? 0 : -1;
}

void mynah_asr_dec_state_reset(const mynah_asr_decoder *dec, mynah_asr_dec_state *s) {
    memset(s, 0, sizeof(*s));
    (void)dec;
    s->last_token = -1;
}

static inline float sigmoid_f(float x) { return mynah_asr_sigmoid(x); }

/* R-8's predictor probe, defined below beside the rest of the experiment. */
static int pred_trace(void);
static void pred_dump(const mynah_asr_decoder *dec, const mynah_asr_dec_state *s,
                      const char *what, int token);

/* One stacked-LSTM step + projector: input = embedding[token]. Updates h/c and s->g. */
static void pred_step(const mynah_asr_decoder *dec, mynah_asr_dec_state *s, int token) {
    const int H = dec->hidden;
    const float *x = dec->embedding + (size_t)token * (size_t)H;
    float z[4 * 1024];

    for (int l = 0; l < dec->n_layers; l++) {
        for (int i = 0; i < 4 * H; i++) z[i] = dec->b_ih[l][i] + dec->b_hh[l][i];
        mynah_asr_gemv_f32(0, 4 * H, H, 1.0f, dec->w_ih[l], H, x, 1.0f, z);
        mynah_asr_gemv_f32(0, 4 * H, H, 1.0f, dec->w_hh[l], H, s->h[l], 1.0f, z);
        for (int i = 0; i < H; i++) {
            const float ig = sigmoid_f(z[i]);
            const float fg = sigmoid_f(z[H + i]);
            const float gg = tanhf(z[2 * H + i]);
            const float og = sigmoid_f(z[3 * H + i]);
            s->c[l][i] = fg * s->c[l][i] + ig * gg;
            s->h[l][i] = og * tanhf(s->c[l][i]);
        }
        x = s->h[l];
    }
    /* decoder_projector */
    memcpy(s->g, dec->proj_b, (size_t)H * sizeof(float));
    mynah_asr_gemv_f32(0, H, H, 1.0f, dec->proj_w, H, s->h[dec->n_layers - 1],
                       1.0f, s->g);
    s->last_token = token;
    if (pred_trace()) pred_dump(dec, s, "after pred_step", token);
}

/* BLOCKED greedy: g changes only on a non-blank emission, so the frames of a
 * blank run share the same g and the joint head (the dominant cost: a V x H
 * matvec per frame) batches into ONE GEMM [B, V] — the same weight matrix read
 * once per block instead of once per frame. The block is adaptive (it doubles
 * over blank runs and restarts short after an emission) so it does not waste
 * rows past the first emitting frame. Semantics identical to the per-frame loop
 * (the inner loop over an emitting frame stays scalar). */
#define DEC_BMAX 32

/* argmax of (logits + bias) on the fly: same float arithmetic as summing into a
 * buffer and then searching for the maximum (v computed once per k), without the
 * V float writes/re-reads per row. */
/* ------------------------------------------------- R-3B: why a step is blank
 *
 * MYNAH_ASR_TRACE_RNNT=1 prints, for every encoder frame the greedy loop looks
 * at, the blank score, the best NON-blank score and their margin. R-2 proved
 * the wait before the first word is spent in blanks; a blank is an observed
 * output, not a cause, and the margin is what separates "the checkpoint is far
 * from emitting" from "a numerical hair decided it".
 *
 * Diagnostic only: it reads the same logits the decision reads and changes no
 * decision. Off, it costs one relaxed load per block. */
static int dec_trace(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("MYNAH_ASR_TRACE_RNNT");
        v = (e != NULL && *e != '\0' && *e != '0') ? 1 : 0;
    }
    return v;
}

/* Best score over the vocabulary excluding `skip`, with the head bias applied,
 * exactly as argmax_bias sees it. */
static int best_excluding(const float *lg, const float *bias, int V, int skip,
                          float *score) {
    int best = -1;
    float bv = 0.0f;
    for (int k = 0; k < V; k++) {
        if (k == skip) continue;
        const float v = lg[k] + bias[k];
        if (best < 0 || v > bv) { bv = v; best = k; }
    }
    *score = bv;
    return best;
}

/* Same, excluding two ids: blank and the word mark, so "best LEXICAL" means a
 * piece that is neither. -1 when the vocabulary is only those two. */
static int best_excluding2(const float *lg, const float *bias, int V, int s1, int s2,
                           float *score) {
    int best = -1;
    float bv = 0.0f;
    for (int k = 0; k < V; k++) {
        if (k == s1 || k == s2) continue;
        const float v = lg[k] + bias[k];
        if (best < 0 || v > bv) { bv = v; best = k; }
    }
    *score = bv;
    return best;
}

/* 1-based rank of `id` by score: how many tokens beat it, plus one. O(V) and
 * only ever called from the trace. */
static int rank_of(const float *lg, const float *bias, int V, int id) {
    if (id < 0 || id >= V) return -1;
    const float mine = lg[id] + bias[id];
    int above = 0;
    for (int k = 0; k < V; k++)
        if (k != id && lg[k] + bias[k] > mine) above++;
    return above + 1;
}

/* The audio a streaming caller had consumed when it handed us this block.
 * Diagnostic only: the decoder has no notion of samples, and reconstructing
 * this from the frame index outside would re-derive a number the stream layer
 * already knows exactly. Set to a negative value when nobody said. */
static double g_trace_audio_s = -1.0;

void mynah_asr_dec_trace_audio(double audio_s) { g_trace_audio_s = audio_s; }

/* ------------------------------------------------------- R-8: predictor probe
 *
 * WHY THIS EXISTS AT ALL. It is tempting to describe arm G -- feeding blank
 * through pred_step() once more -- as "the same content, only the state moves".
 * The implementation does not warrant that phrasing without a check, so here is
 * the check. mynah_asr_dec_state_reset() zeroes h, c and g; the greedy loop then
 * calls pred_step(blank) once, so the SOS representation IS pred_step(blank)
 * APPLIED TO A ZERO STATE. A second pred_step(blank) feeds the SAME embedding
 * through the SAME weights but from a DIFFERENT recurrent state, and an LSTM is
 * not idempotent. G therefore holds the token identity fixed and moves the
 * state; it does not "return to SOS" and it is not a no-op. This prints the
 * numbers that say so. */
static void pred_dump(const mynah_asr_decoder *dec, const mynah_asr_dec_state *s,
                      const char *what, int token) {
    double hn = 0.0, cn = 0.0, gn = 0.0;
    unsigned long hs = 1469598103934665603UL;
    for (int l = 0; l < dec->n_layers; l++)
        for (int i = 0; i < dec->hidden; i++) {
            hn += (double)s->h[l][i] * s->h[l][i];
            cn += (double)s->c[l][i] * s->c[l][i];
        }
    for (int i = 0; i < dec->hidden; i++) {
        gn += (double)s->g[i] * s->g[i];
        unsigned char b[4];
        memcpy(b, &s->g[i], 4);
        for (int k = 0; k < 4; k++) { hs ^= b[k]; hs *= 1099511628211UL; }
    }
    fprintf(stderr, "[PRED] %-22s token=%d |h|=%.6f |c|=%.6f |g|=%.6f g_fnv=%016lx\n",
            what, token, sqrt(hn), sqrt(cn), sqrt(gn), hs);
}

static int pred_trace(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("MYNAH_ASR_TRACE_PRED");
        v = (e != NULL && *e != '\0' && *e != '0') ? 1 : 0;
    }
    return v;
}

/* --------------------------------------------------------- R-8: the injection
 *
 * MYNAH_ASR_RNNT_INJECT="frame=<n>,mode=<blank|wmark|best|lex|id>[,id=<n>]"
 *
 * At the named ABSOLUTE encoder frame, feed one token through pred_step() and
 * do not publish it. The frame comes from OUTSIDE, computed by the harness from
 * the frozen baseline trace, because the decoder has no idea where a baseline
 * crossing was and choosing the point in here would be choosing it per run.
 *
 * What it must not do, and what keeps it honest:
 *   - the token is never written into `tokens[]`, so no client and no scorer
 *     ever sees it (it is not passed to the emit path at all);
 *   - `n_emitted` is not touched, so the trace's SOS/MOVED label keeps meaning
 *     "a NATURAL token was emitted", not "something happened";
 *   - the encoder, its caches and the audio position are not reachable from
 *     here, so they cannot move;
 *   - the frames of the current block BEFORE the injection point are decided
 *     first, normally, and only then is the state perturbed and the block
 *     recomputed from that frame -- the block's logits were computed with the
 *     old g and must not be reused across the perturbation;
 *   - if a natural token is emitted before the injection frame is reached, the
 *     injection is abandoned and says so: the point was mis-specified for that
 *     utterance, which is a fact about the run and not something to paper over.
 *
 * Default OFF. Diagnostic. This is a mechanism experiment, not a fast path. */
typedef struct {
    int armed, done, mode, id, noted;
    long frame, at;
} inject_cfg;
enum { INJ_BLANK = 0, INJ_WMARK, INJ_BEST, INJ_LEX, INJ_ID };
static inject_cfg *inject_conf(void);

static inject_cfg *inject_conf(void) {
    static inject_cfg c;
    static int parsed = 0;
    if (parsed) return &c;
    parsed = 1;
    c.frame = -1;
    c.id = -1;
    const char *e = getenv("MYNAH_ASR_RNNT_INJECT");
    if (!e || !*e) return &c;
    const char *p = strstr(e, "frame=");
    if (p) c.frame = atol(p + 6);
    p = strstr(e, "id=");
    if (p) c.id = atoi(p + 3);
    if (strstr(e, "mode=wmark")) c.mode = INJ_WMARK;
    else if (strstr(e, "mode=best")) c.mode = INJ_BEST;
    else if (strstr(e, "mode=lex")) c.mode = INJ_LEX;
    else if (strstr(e, "mode=id")) c.mode = INJ_ID;
    else c.mode = INJ_BLANK;
    c.armed = c.frame >= 0;
    return &c;
}

/* The token this arm injects at this frame. -1 when the arm cannot be served
 * (no word mark in the vocabulary, no id given), which aborts the injection
 * rather than silently substituting another arm. */
static int inject_token(const mynah_asr_decoder *dec, const inject_cfg *c,
                        const float *lb) {
    float sc = 0.0f;
    switch (c->mode) {
    case INJ_WMARK: return dec->word_mark;
    case INJ_BEST:  return best_excluding(lb, dec->head_b, dec->vocab, dec->blank, &sc);
    case INJ_LEX:   return best_excluding2(lb, dec->head_b, dec->vocab,
                                           dec->blank, dec->word_mark, &sc);
    case INJ_ID:    return (c->id >= 0 && c->id < dec->vocab) ? c->id : -1;
    default:        return dec->blank;
    }
}

/* One decision, fully described (R-9).
 *
 * WHAT THIS HAS TO SEPARATE. Before the first emitted token the blank logit
 * sits in a regime it never returns to (median |blank| 914 against 27 after;
 * .work/first-partial-responsiveness.md). Three explanations were still open
 * from saved traces: the audio is silent, the encoder cache is filling, or the
 * predictor has not moved. The trace cannot decide that on its own -- R-8's
 * intervention can -- but it CAN stop conflating the things it prints:
 *
 *   blank   the score the decision is actually made against
 *   wmark   the bare SentencePiece word mark: not a lexical token at all, and
 *           2 of Q-2's 3 stably-wrong runner-ups were exactly this
 *   lex     the best piece that is neither blank nor the word mark
 *   nb      the best non-blank, which MAY BE the word mark -- this is what the
 *           old trace called best_nonblank, and why it could not answer the
 *           question it was being asked
 *
 * Ranks are 1-based over the whole vocabulary, so "blank rank 1, lex rank 3"
 * reads without knowing the scores. `pred` is SOS while no natural token has
 * been emitted. */
static void dec_trace_line(const mynah_asr_decoder *dec, const mynah_asr_dec_state *s,
                           const float *lb, long frame, int chosen, int post,
                           const float *enc_frame) {
    const int V = dec->vocab;
    const float *bias = dec->head_b;
    const int wm = dec->word_mark;
    float nb_sc = 0.0f, lex_sc = 0.0f, wm_sc = 0.0f;
    const int nb = best_excluding(lb, bias, V, dec->blank, &nb_sc);
    const int lex = best_excluding2(lb, bias, V, dec->blank, wm, &lex_sc);
    if (wm >= 0 && wm < V) wm_sc = lb[wm] + bias[wm];
    const float blank_sc = lb[dec->blank] + bias[dec->blank];
    /* |enc| and |joint| decide whether an out-of-scale logit was born in the
     * encoder or in the joint. They are not decoration: the pre-first-token
     * blank logits reach -900 in f32 as well as int8, so the question "which
     * stage produced that magnitude" is the whole question. */
    double en = 0.0, jn = 0.0;
    for (int i = 0; i < dec->hidden; i++) {
        const float e = enc_frame ? enc_frame[i] : 0.0f;
        const float j = e + s->g[i];
        en += (double)e * e;
        if (j > 0.0f) jn += (double)j * j;
    }
    fprintf(stderr,
            "[RNNT] frame=%ld audio_s=%.4f enc=%.2f joint=%.2f pred=%s post=%d"
            " blank=%d:%.4f:r%d wmark=%d:%.4f:r%d lex=%d:%.4f:r%d nb=%d:%.4f:r%d"
            " chose=%d margin_lex=%.4f margin_nb=%.4f\n",
            frame, g_trace_audio_s, sqrt(en), sqrt(jn),
            s->n_emitted ? "MOVED" : "SOS", post,
            dec->blank, blank_sc, rank_of(lb, bias, V, dec->blank),
            wm, wm_sc, rank_of(lb, bias, V, wm),
            lex, lex_sc, rank_of(lb, bias, V, lex),
            nb, nb_sc, rank_of(lb, bias, V, nb),
            chosen, blank_sc - lex_sc, blank_sc - nb_sc);
}

/* MYNAH_ASR_BLANK_BIAS=<delta> (RESEARCH, default off): until a stream's FIRST
 * natural emission, a decision whose blank beats the best non-blank by less
 * than delta commits that non-blank instead. It is the rule the S13-5b replay
 * applied to stored logits, made live, so the replay is its gate: the live
 * first frame must equal the replayed one. Unset or 0 leaves every decision
 * untouched (the branch below is never taken). RNNT greedy only. */
static float blank_bias(void) {
    static float v = -1.0f;
    if (v < 0.0f) {
        const char *e = getenv("MYNAH_ASR_BLANK_BIAS");
        v = e ? (float)atof(e) : 0.0f;
        if (!(v > 0.0f)) v = 0.0f;
    }
    return v;
}

static int argmax_bias(const float *lg, const float *bias, int V) {
    int best = 0;
    float bv = lg[0] + bias[0];
    for (int k = 1; k < V; k++) {
        const float v = lg[k] + bias[k];
        if (v > bv) { bv = v; best = k; }
    }
    return best;
}

/* TDT greedy (ParakeetTDTGenerationMixin / NeMo GreedyTDTInfer): the head emits
 * [vocab | n_durations] logits; at EVERY step the frame advances by the predicted
 * duration (argmax over the last ND logits) — a blank with duration 0 advances by
 * 1, a non-blank with duration 0 re-emits on the same frame (guarded by
 * max_symbols). No blocking over blank runs: TDT already skips frames (dur>1) and
 * the visited grid is not contiguous. */
static int greedy_decode_tdt(const mynah_asr_decoder *dec, mynah_asr_dec_state *s,
                             const float *enc, int T, int *tokens, int *frames, int cap,
                             float *scratch) {
    const int H = dec->hidden, V = dec->vocab, ND = dec->n_durations;
    const int VL = V + ND;
    float joint[1024];
    const int owned = scratch == NULL;
    float *logits = owned ? malloc((size_t)VL * sizeof(float))
                          : scratch + (size_t)DEC_BMAX * (size_t)H;
    if (!logits) return 0;

    /* f32 head for the BLAS GEMM (deterministic across backends); when quantized
     * and T is large it is dequantized once per call, like the RNNT path */
    const float *W = dec->head.qtype == MYNAH_ASR_Q_F32 ? dec->head.f32 : NULL;
    float *wd = NULL;
    if (!W && T > 16) {
        wd = malloc((size_t)VL * (size_t)H * sizeof(float));
        if (wd) { mynah_asr_qmat_dequant(&dec->head, wd); W = wd; }
    }

    if (s->last_token < 0) pred_step(dec, s, dec->blank); /* SOS = blank, zero state */

    int n_out = 0, t = 0, emitted_here = 0;
    while (t < T) {
        const float *e = enc + (size_t)t * (size_t)H;
        for (int i = 0; i < H; i++) {
            const float v = e[i] + s->g[i];
            joint[i] = v > 0.0f ? v : 0.0f;                /* ReLU */
        }
        if (W)
            mynah_asr_gemm_f32(0, 1, 1, VL, H,
                               1.0f, joint, H, W, H, 0.0f, logits, VL);
        else
            mynah_asr_qmat_mul(&dec->head, joint, logits, 1);

        const int k = argmax_bias(logits, dec->head_b, V);
        int dur = dec->durations[argmax_bias(logits + V, dec->head_b + V, ND)];
        if (k != dec->blank) {
            if (n_out < cap) {
                if (frames) frames[n_out] = (int)(s->t_abs + t);
                tokens[n_out++] = k;
            }
            pred_step(dec, s, k);                          /* state advances only on emit */
            emitted_here++;
            if (dur == 0 && emitted_here >= dec->max_symbols)
                dur = 1;                                   /* unblocks the frame (NeMo) */
        } else if (dur == 0) {
            dur = 1;                                       /* blank: minimal advance */
        }
        if (dur > 0) emitted_here = 0;
        t += dur;
    }
    s->t_abs += T;
    if (owned) free(logits);
    free(wd);
    return n_out;
}

size_t mynah_asr_greedy_scratch_floats(const mynah_asr_decoder *dec) {
    return (size_t)DEC_BMAX * (size_t)dec->hidden
         + (size_t)DEC_BMAX * (size_t)(dec->vocab + dec->n_durations)
         /* S10-2: the quantised activations and their scales for the weight-
          * stationary joint head. int8 rounded up to whole floats. */
         + (((size_t)DEC_BMAX * (size_t)dec->hidden + sizeof(float) - 1) / sizeof(float))
         + (size_t)DEC_BMAX;
}

int mynah_asr_greedy_decode_scratch(const mynah_asr_decoder *dec, mynah_asr_dec_state *s,
                                const float *enc, int T, int *tokens, int *frames, int cap,
                                float *scratch) {
    if (dec->n_durations > 0)
        return greedy_decode_tdt(dec, s, enc, T, tokens, frames, cap, scratch);
    const int H = dec->hidden, V = dec->vocab;
    float joint[1024];
    /* caller scratch (>= mynah_asr_greedy_scratch_floats) keeps the streaming
     * step allocation-free; NULL = allocate here, as before */
    const int owned = scratch == NULL;
    float *jin = owned ? malloc((size_t)DEC_BMAX * (size_t)H * sizeof(float)) : scratch;
    float *logits = owned ? malloc((size_t)DEC_BMAX * (size_t)V * sizeof(float))
                          : scratch + (size_t)DEC_BMAX * (size_t)H;
    if (!jin || !logits) { if (owned) { free(jin); free(logits); } return 0; }
    /* S10-2 scratch for the weight-stationary head; NULL when the caller owns
     * nothing, and mynah_asr_qmat_mul_rows then takes its per-row path. */
    float *hsx = owned ? NULL : logits + (size_t)DEC_BMAX * (size_t)V;
    int8_t *hqx = hsx ? (int8_t *)(hsx + DEC_BMAX) : NULL;
    int n_out = 0;

    /* The head as an f32 matrix for a direct BLAS GEMM (CPU: deterministic across
     * backends — decoding NEVER goes through the GPU). When quantized and T is
     * large (offline) it is dequantized ONCE per call: 33 MB read/written once
     * against the int8 matrix re-read for every frame. For small T (streaming
     * chunks) qmat's per-frame quantized dot stays. */
    const float *W = dec->head.qtype == MYNAH_ASR_Q_F32 ? dec->head.f32 : NULL;
    float *wd = NULL;
    if (!W && T > 16) {
        wd = malloc((size_t)V * (size_t)H * sizeof(float));
        if (wd) { mynah_asr_qmat_dequant(&dec->head, wd); W = wd; }
    }

    if (s->last_token < 0) pred_step(dec, s, dec->blank); /* SOS = blank, zero state */

    int t = 0, B = 4;
    while (t < T) {
        const int Bc = (T - t) < B ? (T - t) : B;
        for (int b = 0; b < Bc; b++) {
            const float *e = enc + (size_t)(t + b) * (size_t)H;
            float *ji = jin + (size_t)b * (size_t)H;
            for (int i = 0; i < H; i++) {
                const float v = e[i] + s->g[i];
                ji[i] = v > 0.0f ? v : 0.0f;               /* ReLU */
            }
        }
        if (W)
            mynah_asr_gemm_f32(0, 1, Bc, V, H,
                               1.0f, jin, H, W, H, 0.0f, logits, V);
        else
            /* S10-2: Bc is at most 4, which put this through the T <= 16 branch
             * of mynah_asr_qmat_mul: a loop that quantises one row, then walks
             * the whole head computing one dot per output. The head is 8.4 MiB
             * of int8, so it was re-read from memory once per FRAME, four times
             * per step per stream, on the scheduler thread with the pool idle.
             * The rows entry point is weight-stationary and exact by the same
             * per-row-dot construction (S1-4, qmat.h): the head is read once
             * for the block. */
            mynah_asr_qmat_mul_rows(&dec->head, jin, logits, Bc, hqx, hsx);

        inject_cfg *inj = inject_conf();
        int first = -1, best = -1, hit = -1;
        if (inj->armed && !inj->done)
            for (int b = 0; b < Bc; b++)
                if ((long)(s->t_abs + t + b) == inj->frame) { hit = b; break; }

        for (int b = 0; b < Bc; b++) {
            const float *lb = logits + (size_t)b * (size_t)V;
            int am = argmax_bias(lb, dec->head_b, V);
            if (am == dec->blank && s->n_emitted == 0 && blank_bias() > 0.0f) {
                float nbs;
                const int nb = best_excluding(lb, dec->head_b, V, dec->blank, &nbs);
                if (nb >= 0 && (lb[dec->blank] + dec->head_b[dec->blank]) - nbs < blank_bias())
                    am = nb;
            }
            if (dec_trace())
                dec_trace_line(dec, s, lb, (long)(s->t_abs + t + b), am,
                               (inj->armed && inj->done) ? 1 : 0,
                               enc + (size_t)(t + b) * (size_t)H);
            if (am != dec->blank) { first = b; best = am; break; }
            /* The injection frame, reached with every earlier frame in this
             * block decided normally: perturb, then recompute FROM HERE. The
             * logits already in `logits` were produced with the old g. */
            if (b == hit) {
                const int tok = inject_token(dec, inj, lb);
                if (tok < 0) {
                    fprintf(stderr, "[INJECT] ABORTED frame=%ld: this arm has no token\n",
                            inj->frame);
                    inj->armed = 0;
                } else {
                    if (pred_trace()) pred_dump(dec, s, "before injection", s->last_token);
                    pred_step(dec, s, tok);
                    inj->done = 1;
                    inj->at = s->t_abs + t + b;
                    fprintf(stderr, "[INJECT] frame=%ld token=%d (not published, "
                                    "n_emitted still %d)\n", inj->at, tok, s->n_emitted);
                }
                first = -2;                       /* redo this frame with the new g */
                break;
            }
        }
        if (first == -2) { t += hit; B = 4; continue; }
        /* A natural token before the requested frame is NOT a reason to give up:
         * the post-first-token perturbation is a deliberate control (does an
         * arbitrary pred_step always disrupt decoding, or is the pre-first-token
         * window special?). Say it happened and keep the injection armed. */
        if (inj->armed && !inj->done && first >= 0 && !inj->noted &&
            (long)(s->t_abs + t + first) < inj->frame) {
            fprintf(stderr, "[INJECT] note: a natural token was emitted at frame %ld, "
                            "before the requested frame %ld -- this is the post-token control\n",
                    (long)(s->t_abs + t + first), inj->frame);
            inj->noted = 1;
        }
        if (first < 0) {                                   /* all-blank run */
            t += Bc;
            if (B < DEC_BMAX) B *= 2;
            continue;
        }

        /* frame t+first emits: scalar inner loop (the first emission is already known) */
        t += first;
        const float *e = enc + (size_t)t * (size_t)H;
        for (int emitted = 0; emitted < dec->max_symbols; emitted++) {
            if (emitted > 0) {                             /* iteration 0: from the batch */
                for (int i = 0; i < H; i++) {
                    const float v = e[i] + s->g[i];
                    joint[i] = v > 0.0f ? v : 0.0f;
                }
                if (W)
                    mynah_asr_gemm_f32(0, 1, 1, V, H,
                                       1.0f, joint, H, W, H, 0.0f, logits, V);
                else
                    mynah_asr_qmat_mul(&dec->head, joint, logits, 1);
                best = argmax_bias(logits, dec->head_b, V);
                if (best == dec->blank) break;
            }
            if (n_out < cap) {
                if (frames) frames[n_out] = (int)(s->t_abs + t);
                tokens[n_out++] = best;
            }
            s->n_emitted++;                                /* NATURAL emissions only */
            pred_step(dec, s, best);                       /* state advances only on emit */
        }
        t++;
        B = 4;                                             /* restart short after an emit */
    }
    s->t_abs += T;
    if (owned) { free(jin); free(logits); }
    free(wd);
    return n_out;
}

int mynah_asr_greedy_decode(const mynah_asr_decoder *dec, mynah_asr_dec_state *s,
                        const float *enc, int T, int *tokens, int *frames, int cap) {
    return mynah_asr_greedy_decode_scratch(dec, s, enc, T, tokens, frames, cap, NULL);
}

int mynah_asr_dec_diag_prime(void) {
    (void)blank_bias();                        /* resolve it before any parallel decode */
    const int t = dec_trace(), p = pred_trace();
    const inject_cfg *c = inject_conf();
    const char *e = getenv("MYNAH_ASR_RNNT_INJECT");
    (void)c;
    return t || p || (e && *e);
}
