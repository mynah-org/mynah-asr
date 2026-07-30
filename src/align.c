#include "align.h"

#include <stdlib.h>

/* Viterbi over the standard CTC extended sequence: blank, tok0, blank, tok1, ...
 * blank — S = 2N+1 states. Allowed moves into state s: stay (s), advance from
 * s-1, and skip from s-2 but only when s is a label different from the label at
 * s-2 (otherwise two identical labels would merge into one).
 *
 * Scores may be raw logits. A CTC path picks exactly ONE label per frame, so the
 * log_softmax normalizer contributes sum_t logsumexp_v(scores[t][v]) to EVERY
 * path: it shifts all paths by the same constant and cannot change which one is
 * best. Skipping the normalization saves a T x V pass and, more importantly,
 * removes a numerical stability question from the middle of the alignment.
 *
 * -ffast-math is on for this repo, so infinities are not something to rely on:
 * unreachable states carry a large finite sentinel instead. */
#define ALIGN_NEG_BIG (-1.0e30f)
#define ALIGN_UNREACHABLE (-1.0e29f)      /* anything below this is "no path" */

int mynah_asr_align_min_frames(const int *tokens, int n_tokens) {
    if (n_tokens <= 0) return 0;
    int need = n_tokens;
    for (int i = 1; i < n_tokens; i++)
        if (tokens[i] == tokens[i - 1]) need++;   /* a blank must separate them */
    return need;
}

int mynah_asr_align_ctc(const float *scores, int T, int V, const int *tokens, int n_tokens,
                        int blank, int *t0, int *t1) {
    if (!scores || !tokens || !t0 || !t1 || T <= 0 || V <= 0 || n_tokens <= 0) return -1;
    if (blank < 0 || blank >= V) return -1;
    for (int i = 0; i < n_tokens; i++)
        if (tokens[i] < 0 || tokens[i] >= V || tokens[i] == blank) return -1;
    if (T < mynah_asr_align_min_frames(tokens, n_tokens)) return -1;

    const int S = 2 * n_tokens + 1;
    /* ext[s]: the label state s emits (blank on even positions) */
    int *ext = malloc((size_t)S * sizeof(int));
    float *cur = malloc((size_t)S * sizeof(float));
    float *prv = malloc((size_t)S * sizeof(float));
    /* backpointer per (frame, state): 0 = stay, 1 = from s-1, 2 = from s-2 */
    unsigned char *bp = malloc((size_t)T * (size_t)S);
    if (!ext || !cur || !prv || !bp) { free(ext); free(cur); free(prv); free(bp); return -1; }

    for (int s = 0; s < S; s++) ext[s] = (s % 2 == 0) ? blank : tokens[s / 2];

    for (int s = 0; s < S; s++) prv[s] = ALIGN_NEG_BIG;
    prv[0] = scores[ext[0]];                          /* frame 0: blank ... */
    prv[1] = scores[ext[1]];                          /* ... or the first token */
    for (int s = 0; s < S; s++) bp[s] = 0;

    for (int t = 1; t < T; t++) {
        const float *row = scores + (size_t)t * (size_t)V;
        unsigned char *bpt = bp + (size_t)t * (size_t)S;
        /* a path cannot be further along than one state per frame, and it must
         * be close enough to the end to still finish: pruning these keeps the
         * loop honest about impossible states instead of relying on sentinels */
        const int s_lo = (S - 2 * (T - t) > 0) ? S - 2 * (T - t) : 0;
        const int s_hi = (2 * t + 1 < S - 1) ? 2 * t + 1 : S - 1;
        for (int s = 0; s < s_lo; s++) { cur[s] = ALIGN_NEG_BIG; bpt[s] = 0; }
        for (int s = s_hi + 1; s < S; s++) { cur[s] = ALIGN_NEG_BIG; bpt[s] = 0; }
        for (int s = s_lo; s <= s_hi; s++) {
            float best = prv[s];
            unsigned char from = 0;
            if (s >= 1 && prv[s - 1] > best) { best = prv[s - 1]; from = 1; }
            if (s >= 2 && ext[s] != blank && ext[s] != ext[s - 2] && prv[s - 2] > best) {
                best = prv[s - 2];
                from = 2;
            }
            cur[s] = best > ALIGN_UNREACHABLE ? best + row[ext[s]] : ALIGN_NEG_BIG;
            bpt[s] = from;
        }
        float *swap = prv; prv = cur; cur = swap;
    }

    /* the path ends on the last token or on the trailing blank */
    int s = (prv[S - 1] >= prv[S - 2]) ? S - 1 : S - 2;
    if (prv[s] <= ALIGN_UNREACHABLE) { free(ext); free(cur); free(prv); free(bp); return -1; }

    /* backtrack, filling the runs from the end: state 2j+1 is token j, and the
     * path visits it over one contiguous stretch of frames */
    for (int j = 0; j < n_tokens; j++) { t0[j] = -1; t1[j] = -1; }
    for (int t = T - 1; t >= 0; t--) {
        if (s % 2 == 1) {
            const int j = s / 2;
            if (t1[j] < 0) t1[j] = t;
            t0[j] = t;
        }
        s -= bp[(size_t)t * (size_t)S + (size_t)s];
    }

    free(ext); free(cur); free(prv); free(bp);
    for (int j = 0; j < n_tokens; j++)
        if (t0[j] < 0) return -1;                     /* every token must be placed */
    return 0;
}
