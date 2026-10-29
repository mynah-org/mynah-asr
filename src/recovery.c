/* Empty-turn recovery: replay ownership and the publish gate (recovery.h). */
#include "recovery.h"

#include <string.h>

/* ------------------------------------------------------------- content */

/* Is this code point punctuation, a symbol or a space rather than content?
 * ASCII by class; above it, the punctuation blocks and the few punctuation
 * marks that live inside a script's own block (the ones this vocabulary
 * carries: Arabic comma/question mark, Devanagari danda, ...). Everything else
 * non-ASCII counts as content, combining marks included: they never stand
 * alone in a word. */
static int cp_is_content(unsigned c) {
    if (c < 0x80)
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    if (c < 0xC0 || c == 0xD7 || c == 0xF7) return 0;                 /* Latin-1 punctuation */
    if (c == 0x37E || c == 0x387 || (c >= 0x55A && c <= 0x55F) || c == 0x589) return 0;
    if (c == 0x5BE || c == 0x5C0 || c == 0x5C3 || c == 0x5C6) return 0;
    if (c == 0x60C || c == 0x61B || c == 0x61F || (c >= 0x66A && c <= 0x66D) || c == 0x6D4) return 0;
    if (c == 0x964 || c == 0x965 || c == 0xE2F || c == 0xE5A || c == 0xE5B) return 0;
    if ((c >= 0x1361 && c <= 0x1368) || (c >= 0x17D4 && c <= 0x17DA)) return 0;
    if (c >= 0x2000 && c <= 0x2BFF) return 0;       /* general punctuation .. symbols, ▁ */
    if (c >= 0x3000 && c <= 0x303F) return 0;       /* CJK punctuation */
    if (c == 0x30FB || c == 0xFFFD) return 0;
    if (c >= 0xFF00 && c <= 0xFF65)                 /* fullwidth: keep the letters and digits */
        return (c >= 0xFF10 && c <= 0xFF19) || (c >= 0xFF21 && c <= 0xFF3A) ||
               (c >= 0xFF41 && c <= 0xFF5A);
    return 1;
}

/* One UTF-8 code point at s (< end); returns the bytes consumed, >= 1. */
static int utf8_next(const char *s, const char *end, unsigned *cp) {
    const unsigned char *u = (const unsigned char *)s;
    const int len = u[0] < 0x80 ? 1 : (u[0] & 0xE0) == 0xC0 ? 2 : (u[0] & 0xF0) == 0xE0 ? 3
                  : (u[0] & 0xF8) == 0xF0 ? 4 : 0;
    if (len == 0 || len > end - s) { *cp = 0xFFFD; return 1; }
    unsigned v = len == 1 ? u[0] : u[0] & (0x7F >> len);
    for (int i = 1; i < len; i++) {
        if ((u[i] & 0xC0) != 0x80) { *cp = 0xFFFD; return 1; }
        v = (v << 6) | (u[i] & 0x3F);
    }
    *cp = v;
    return len;
}

static int span_has_content(const char *s, const char *end) {
    while (s < end) {
        unsigned cp;
        s += utf8_next(s, end, &cp);
        if (cp_is_content(cp)) return 1;
    }
    return 0;
}

int mynah_asr_text_has_content(const char *text) {
    return text && span_has_content(text, text + strlen(text));
}

/* ---------------------------------------------------------------- words */

static const char *piece_of(const mynah_asr_tokenizer *tk, int id) {
    return (id >= 0 && id < tk->n_pieces) ? tk->pieces[id] : NULL;
}

static int is_text(const char *p) { return p && !mynah_asr_piece_is_special(p); }

/* Scripts written without spaces between words: there an unmarked piece is no
 * word's tail, the mark only comes before a digit, a Latin word or a sentence. */
static int cp_is_unspaced(unsigned c) {
    return (c >= 0x0E00 && c <= 0x0EFF) ||  /* Thai, Lao */
           (c >= 0x1000 && c <= 0x109F) ||  /* Myanmar */
           (c >= 0x1780 && c <= 0x17FF) ||  /* Khmer */
           (c >= 0x3040 && c <= 0x30FF) ||  /* Hiragana, Katakana */
           (c >= 0x3400 && c <= 0x4DBF) ||  /* CJK Extension A */
           (c >= 0x4E00 && c <= 0x9FFF) ||  /* CJK Unified Ideographs */
           (c >= 0xF900 && c <= 0xFAFF);    /* CJK Compatibility Ideographs */
}

/* Does token i start a word run: a marked text piece, or the text piece right
 * after a special one (a language tag ends whatever was before it)? */
static int starts_run(const mynah_asr_tokenizer *tk, const int *tokens, int i) {
    const char *p = piece_of(tk, tokens[i]);
    if (!is_text(p)) return 0;
    return mynah_asr_piece_has_mark(p) || (i > 0 && !is_text(piece_of(tk, tokens[i - 1])));
}

/* Do the text pieces of tokens[a..b) hold content (unspaced: content of an
 * unspaced script)? */
static int run_has(const mynah_asr_tokenizer *tk, const int *tokens, int a, int b, int unspaced) {
    for (int j = a; j < b; j++) {
        const char *q = piece_of(tk, tokens[j]);
        if (!is_text(q)) continue;
        for (const char *s = q, *end = q + strlen(q); s < end;) {
            unsigned cp;
            s += utf8_next(s, end, &cp);
            if (unspaced ? cp_is_unspaced(cp) : cp_is_content(cp)) return 1;
        }
    }
    return 0;
}

/* The end of the run that token i starts: the next run start, or n. */
static int run_end(const mynah_asr_tokenizer *tk, const int *tokens, int n, int i) {
    int j = i + 1;
    while (j < n && !starts_run(tk, tokens, j)) j++;
    return j;
}

/* Does token i open a WORD (recovery.h): a run start whose run holds content? */
static int opens_word(const mynah_asr_tokenizer *tk, const int *tokens, int n, int i) {
    return starts_run(tk, tokens, i) && run_has(tk, tokens, i, run_end(tk, tokens, n, i), 0);
}

/* Is the word that tokens[0..f) end on still open and lexical, so that the
 * unmarked pieces from f on continue it? Not when nothing comes before f, when a
 * special piece ends it, or when its run holds no content yet: a bare mark
 * emitted just before the boundary opened the CURRENT turn's first word. */
static int context_word_open(const mynah_asr_tokenizer *tk, const int *tokens, int f) {
    if (f == 0 || !is_text(piece_of(tk, tokens[f - 1]))) return 0;
    int j = f - 1;
    while (j > 0 && !starts_run(tk, tokens, j)) j--;
    return run_has(tk, tokens, j, f, 0);
}

/* WORD: the first token the current turn owns. */
static int word_start(const mynah_asr_tokenizer *tk, const int *tokens, const int *frames,
                      int n, int keep_from) {
    int f = 0;
    while (f < n && frames[f] < keep_from) f++;
    if (f == n) return n;
    /* the unmarked pieces before the first run start at or after the boundary */
    int b = f;
    while (b < n && !starts_run(tk, tokens, b)) b++;
    if (b > f && run_has(tk, tokens, f, b, 0) &&
        (!context_word_open(tk, tokens, f) || run_has(tk, tokens, f, b, 1)))
        return f;  /* they are the current turn's (its word opened before the boundary) */
    for (int i = b; i < n; i++)
        if (opens_word(tk, tokens, n, i)) return i;
    return n;
}

/* A word reduced for comparison: its content bytes, ASCII lower-cased.
 * Bounded; a longer word compares on its first bytes, enough to tell words
 * apart here. */
#define NWORD_CAP 64
typedef struct { char w[NWORD_CAP]; int len; } nword;

static void nword_add(nword *w, const char *s, const char *end) {
    while (s < end) {
        unsigned cp;
        const int k = utf8_next(s, end, &cp);
        if (cp_is_content(cp) && w->len + k < NWORD_CAP)
            for (int b = 0; b < k; b++)
                w->w[w->len++] = (k == 1 && s[b] >= 'A' && s[b] <= 'Z') ? (char)(s[b] + 32) : s[b];
        s += k;
    }
}

static int nword_eq(const nword *a, const nword *b) {
    return a->len == b->len && memcmp(a->w, b->w, (size_t)a->len) == 0;
}

/* The last up-to-cap content words of a text (split at spaces), in order;
 * *total receives how many content words it has in all. */
static int tail_words(const char *text, nword *out, int cap, int *total) {
    int n = 0, all = 0;
    for (const char *s = text ? text : ""; *s;) {
        while (*s == ' ') s++;
        const char *b = s;
        while (*s && *s != ' ') s++;
        nword w = {{0}, 0};
        nword_add(&w, b, s);
        if (w.len == 0) continue;
        all++;
        if (n == cap) { memmove(out, out + 1, (size_t)(cap - 1) * sizeof(nword)); n--; }
        out[n++] = w;
    }
    if (total) *total = all;
    return n;
}

/* ------------------------------------------------------------ ownership */

/* At most this many context words are looked for past the boundary: a
 * transducer's emission lag is a fraction of a second, a few words at most. */
#define TEXT_STRIP_MAX 4

int mynah_asr_replay_keep_from(const mynah_asr_tokenizer *tk, const int *tokens,
                               const int *frames, int n, int keep_from,
                               mynah_asr_keep_rule rule, const char *context_text,
                               int lag_frames) {
    int i = 0;
    if (rule == MYNAH_ASR_KEEP_FRAME) {
        while (i < n && !(frames[i] >= keep_from && is_text(piece_of(tk, tokens[i])))) i++;
        return i;
    }
    i = word_start(tk, tokens, frames, n, keep_from);
    if (rule == MYNAH_ASR_KEEP_WORD || i == n || !context_text) return i;

    /* TEXT: the first words the current turn owns, w0..; give back the
     * longest prefix w0..wj-1 equal to the context's last j words. A word is
     * compared only when it is whole: the next word started or the tokens end. */
    nword ctx[TEXT_STRIP_MAX], kw[TEXT_STRIP_MAX];
    const int m = tail_words(context_text, ctx, TEXT_STRIP_MAX, NULL);
    int start[TEXT_STRIP_MAX + 1], k = 0;
    start[k++] = i;  /* the first owned word, which may open with an unmarked piece */
    for (int t = i + 1; t < n && k <= TEXT_STRIP_MAX; t++)
        if (opens_word(tk, tokens, n, t)) start[k++] = t;
    /* only words opened inside the lag window may be given back */
    int in_window = k;
    if (lag_frames > 0)
        for (int q = 0; q < k; q++)
            if (frames[start[q]] >= keep_from + lag_frames) { in_window = q; break; }
    const int whole = k > TEXT_STRIP_MAX ? TEXT_STRIP_MAX : k;
    for (int q = 0; q < whole; q++) {
        const int end = q + 1 < k ? start[q + 1] : n;
        kw[q].len = 0;
        for (int t = start[q]; t < end; t++) {
            const char *p = piece_of(tk, tokens[t]);
            if (is_text(p)) nword_add(&kw[q], p, p + strlen(p));
        }
    }
    int best = 0;
    for (int j = 1; j <= whole && j <= m && j <= in_window; j++) {
        int same = 1;
        for (int q = 0; q < j && same; q++) same = nword_eq(&kw[q], &ctx[m - j + q]);
        if (same) best = j;
    }
    if (best == 0) return i;
    return best < k ? start[best] : n;
}

/* ----------------------------------------------------------------- gate */

/* "en-US" and "en-GB" share the base "en". */
static int same_base(const char *a, const char *b) {
    size_t i = 0;
    while (a[i] >= 'a' && a[i] <= 'z' && b[i] == a[i]) i++;
    return i > 0 && !(a[i] >= 'a' && a[i] <= 'z') && !(b[i] >= 'a' && b[i] <= 'z');
}

mynah_asr_gate_verdict mynah_asr_recovery_gate(const char *text, const char *session_lang,
                                               const char *detected_lang,
                                               const char *context_text) {
    if (!mynah_asr_text_has_content(text)) return MYNAH_ASR_GATE_EMPTY;
    if (session_lang && *session_lang && strncmp(session_lang, "auto", 4) != 0 &&
        detected_lang && *detected_lang && !same_base(session_lang, detected_lang))
        return MYNAH_ASR_GATE_LANG;
    if (context_text) {
        nword tw[TEXT_STRIP_MAX], cw[TEXT_STRIP_MAX];
        int total = 0;
        const int nt = tail_words(text, tw, TEXT_STRIP_MAX, &total);
        const int nc = tail_words(context_text, cw, TEXT_STRIP_MAX, NULL);
        if (total == nt && nt <= nc) {
            int same = 1;
            for (int q = 0; q < nt && same; q++) same = nword_eq(&tw[q], &cw[nc - nt + q]);
            if (same) return MYNAH_ASR_GATE_ECHO;
        }
    }
    return MYNAH_ASR_GATE_OK;
}

const char *mynah_asr_gate_name(mynah_asr_gate_verdict v) {
    switch (v) {
    case MYNAH_ASR_GATE_OK: return "ok";
    case MYNAH_ASR_GATE_EMPTY: return "empty";
    case MYNAH_ASR_GATE_LANG: return "lang";
    case MYNAH_ASR_GATE_ECHO: return "echo";
    }
    return "?";
}

/* ------------------------------------------------------------- recovery */

/* Words of a whole text, normalised, for an equality check. */
#define SAME_WORDS_MAX 64
static int all_words(const char *text, nword *out, int cap) {
    int n = 0;
    for (const char *s = text ? text : ""; *s && n < cap;) {
        while (*s == ' ') s++;
        const char *b = s;
        while (*s && *s != ' ') s++;
        nword w = {{0}, 0};
        nword_add(&w, b, s);
        if (w.len > 0) out[n++] = w;
    }
    return n;
}

int mynah_asr_text_same_words(const char *a, const char *b) {
    nword wa[SAME_WORDS_MAX], wb[SAME_WORDS_MAX];
    const int na = all_words(a, wa, SAME_WORDS_MAX), nb = all_words(b, wb, SAME_WORDS_MAX);
    if (na == 0 || na != nb) return 0;
    for (int i = 0; i < na; i++)
        if (!nword_eq(&wa[i], &wb[i])) return 0;
    return 1;
}

void mynah_asr_recover_cfg_default(mynah_asr_recover_cfg *c) {
    memset(c, 0, sizeof(*c));
    c->phase_frames[0] = 3;
    c->phase_frames[1] = 4;
    c->phase_frames[2] = 6;
    c->n_phases = 3;
    c->replay_frames = 6;
    c->min_votes = 2;
    c->rule = MYNAH_ASR_KEEP_TEXT;
    c->lag_frames = 0;
    c->max_turn_s = 4.0;
}

int mynah_asr_recover_plan(const mynah_asr_recover_cfg *c, size_t n_prev, int frame_samples,
                           mynah_asr_attempt *out) {
    int n = 0;
    for (int i = 0; i < c->n_phases && n < MYNAH_ASR_RECOVER_MAX_ATTEMPTS; i++)
        if (c->phase_frames[i] > 0)
            out[n++] = (mynah_asr_attempt){MYNAH_ASR_ATTEMPT_PHASE, c->phase_frames[i]};
    if (c->replay_frames > 0 && n < MYNAH_ASR_RECOVER_MAX_ATTEMPTS &&
        n_prev >= (size_t)c->replay_frames * (size_t)frame_samples)
        out[n++] = (mynah_asr_attempt){MYNAH_ASR_ATTEMPT_REPLAY, c->replay_frames};
    return n;
}

/* Deterministic low-level noise: a fixed xorshift stream, 12 uniforms summed
 * (a Gaussian to a good approximation), scaled to an RMS of 1e-3 (-60 dBFS).
 * The same samples for every attempt and every call: the attempt's text
 * depends only on the turn. */
#define RECOVER_NOISE_RMS 1e-3f

size_t mynah_asr_recover_build(const mynah_asr_attempt *a, int frame_samples, const float *cur,
                               size_t n_cur, const float *prev, size_t n_prev, float *out) {
    const size_t np = (size_t)a->prefix_frames * (size_t)frame_samples;
    if (a->kind == MYNAH_ASR_ATTEMPT_REPLAY && prev && n_prev >= np) {
        memcpy(out, prev + (n_prev - np), np * sizeof(float));
    } else {
        unsigned x = 0x9E3779B9u;
        for (size_t i = 0; i < np; i++) {
            float g = 0.0f;
            for (int k = 0; k < 12; k++) {
                x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                g += (float)(x >> 8) * (1.0f / 16777216.0f);
            }
            out[i] = (g - 6.0f) * RECOVER_NOISE_RMS;
        }
    }
    if (n_cur) memcpy(out + np, cur, n_cur * sizeof(float));
    return np + n_cur;
}

int mynah_asr_recover_vote(const char *const *texts, int n, int min_votes) {
    for (int i = 0; i < n; i++) {
        if (!mynah_asr_text_has_content(texts[i])) continue;
        int votes = 0;
        for (int j = 0; j < n; j++)
            votes += mynah_asr_text_has_content(texts[j]) && mynah_asr_text_same_words(texts[i], texts[j]);
        if (votes >= min_votes) return i;
    }
    return -1;
}

int mynah_asr_recover_vote_partial(const char *const *texts, int k, int n, int min_votes) {
    const int left = n - k;
    /* votes so far of the text of attempt i (0 when it carries no content) */
    int votes[MYNAH_ASR_RECOVER_MAX_ATTEMPTS] = {0};
    for (int i = 0; i < k; i++) {
        if (!mynah_asr_text_has_content(texts[i])) continue;
        for (int j = 0; j < k; j++)
            votes[i] += mynah_asr_text_has_content(texts[j]) && mynah_asr_text_same_words(texts[i], texts[j]);
    }
    /* the full vote takes the FIRST index reaching min_votes: attempt i is the
     * final winner once it has the votes and no earlier contentful attempt can
     * still reach them with the attempts left */
    for (int i = 0; i < k; i++) {
        if (!mynah_asr_text_has_content(texts[i])) continue;
        if (votes[i] >= min_votes) return i;
        if (votes[i] + left >= min_votes) return MYNAH_ASR_RECOVER_OPEN;
    }
    /* nobody so far can win; a text not seen yet needs min_votes fresh attempts */
    return left >= min_votes ? MYNAH_ASR_RECOVER_OPEN : -1;
}
