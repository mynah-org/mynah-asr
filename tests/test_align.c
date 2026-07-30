/* CTC forced alignment (src/align.c) — no model needed, so this runs everywhere.
 *
 * The point of aligning synthetic posteriors is that the answer is known in
 * advance: the fixtures put a label's score high over an exact frame range, so
 * the test asserts the recovered range, not just "it returned 0". Then the
 * awkward cases that a hand-rolled Viterbi gets wrong: two identical labels in a
 * row (a blank must separate them), two different labels with no blank frame
 * between them (needs the skip transition), an alignment that does not fit in the
 * frames available, and flat scores where every path ties.
 *
 * Exit: 0 ok, 1 fail. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/align.h"

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("align FAIL: %s\n", msg); failures = 1; } \
    else printf("align ok:   %s\n", msg); } while (0)

#define V 5
#define BLANK 4

/* scores[T][V]: `want[t]` is the favoured label of frame t */
static float *build(const int *want, int T, float peak) {
    float *s = calloc((size_t)T * V, sizeof(float));
    for (int t = 0; t < T; t++) s[(size_t)t * V + want[t]] = peak;
    return s;
}

static int check_run(const char *what, const int *t0, const int *t1, int j, int a, int b) {
    char msg[160];
    snprintf(msg, sizeof(msg), "%s: token %d on frames [%d, %d] (got [%d, %d])",
             what, j, a, b, t0[j], t1[j]);
    const int ok = t0[j] == a && t1[j] == b;
    CHECK(ok, msg);
    return ok;
}

int main(void) {
    char msg[160];

    /* ---- 1. known alignment, blanks in between ------------------------- */
    {
        const int want[9] = {BLANK, BLANK, 0, 0, BLANK, BLANK, 1, 2, 2};
        const int tokens[3] = {0, 1, 2};
        float *s = build(want, 9, 10.0f);
        int t0[3], t1[3];
        CHECK(mynah_asr_align_ctc(s, 9, V, tokens, 3, BLANK, t0, t1) == 0, "3 tokens over 9 frames");
        check_run("spaced", t0, t1, 0, 2, 3);
        check_run("spaced", t0, t1, 1, 6, 6);
        check_run("spaced", t0, t1, 2, 7, 8);
        free(s);
    }

    /* ---- 2. repeated label: the blank between them is mandatory -------- */
    {
        const int want[3] = {1, BLANK, 1};
        const int tokens[2] = {1, 1};
        float *s = build(want, 3, 10.0f);
        int t0[2], t1[2];
        CHECK(mynah_asr_align_min_frames(tokens, 2) == 3, "repeated label needs 3 frames");
        CHECK(mynah_asr_align_ctc(s, 3, V, tokens, 2, BLANK, t0, t1) == 0, "\"a a\" over 3 frames");
        check_run("repeat", t0, t1, 0, 0, 0);
        check_run("repeat", t0, t1, 1, 2, 2);
        /* the same tokens in 2 frames cannot be aligned at all */
        CHECK(mynah_asr_align_ctc(s, 2, V, tokens, 2, BLANK, t0, t1) == -1,
              "\"a a\" in 2 frames is refused");
        free(s);
    }

    /* ---- 2b. the blank between identical labels is not negotiable -------
     * Scores that make the ILLEGAL path much more attractive: favouring the
     * label at frames 0 and 1 lets a broken Viterbi skip straight from token 0
     * to token 1 (score 30) instead of paying for the mandatory blank (score
     * 10). Case 2 above does not catch that — the legal path happens to win
     * there anyway — so without this fixture the rule is untested. */
    {
        const int want[3] = {1, 1, BLANK};
        const int tokens[2] = {1, 1};
        float *s = build(want, 3, 10.0f);
        int t0[2], t1[2];
        CHECK(mynah_asr_align_ctc(s, 3, V, tokens, 2, BLANK, t0, t1) == 0, "\"a a\" with a tempting skip");
        check_run("no-merge", t0, t1, 0, 0, 0);
        check_run("no-merge", t0, t1, 1, 2, 2);
        free(s);
    }

    /* ---- 3. adjacent distinct labels: needs the skip transition -------- */
    {
        const int want[2] = {0, 1};
        const int tokens[2] = {0, 1};
        float *s = build(want, 2, 10.0f);
        int t0[2], t1[2];
        CHECK(mynah_asr_align_min_frames(tokens, 2) == 2, "distinct labels need 2 frames");
        CHECK(mynah_asr_align_ctc(s, 2, V, tokens, 2, BLANK, t0, t1) == 0, "\"a b\" over 2 frames");
        check_run("skip", t0, t1, 0, 0, 0);
        check_run("skip", t0, t1, 1, 1, 1);
        free(s);
    }

    /* ---- 4. a long token run is not split -------------------------------
     * one label held for 5 frames must come back as one run, not five */
    {
        const int want[7] = {BLANK, 3, 3, 3, 3, 3, BLANK};
        const int tokens[1] = {3};
        float *s = build(want, 7, 10.0f);
        int t0[1], t1[1];
        CHECK(mynah_asr_align_ctc(s, 7, V, tokens, 1, BLANK, t0, t1) == 0, "1 token over 7 frames");
        check_run("held", t0, t1, 0, 1, 5);
        free(s);
    }

    /* ---- 5. flat scores: every path ties, the result must still be valid */
    {
        const int tokens[4] = {0, 1, 0, 2};
        float *s = calloc(40 * V, sizeof(float));
        int t0[4], t1[4];
        CHECK(mynah_asr_align_ctc(s, 40, V, tokens, 4, BLANK, t0, t1) == 0, "flat scores align");
        int monotone = 1;
        for (int j = 0; j < 4; j++) {
            if (t0[j] < 0 || t1[j] < t0[j] || t1[j] >= 40) monotone = 0;
            if (j && t0[j] <= t1[j - 1]) monotone = 0;
        }
        CHECK(monotone, "flat scores give a monotone, in-range alignment");
        free(s);
    }

    /* ---- 6. random stress: never crash, always monotone or -1 ---------- */
    {
        srand(20260730);
        int bad_shape = 0, refused = 0;
        for (int it = 0; it < 500; it++) {
            const int T = 1 + rand() % 60, n = 1 + rand() % 12;
            float *s = malloc((size_t)T * V * sizeof(float));
            int *tok = malloc((size_t)n * sizeof(int));
            int *t0 = malloc((size_t)n * sizeof(int)), *t1 = malloc((size_t)n * sizeof(int));
            for (int i = 0; i < T * V; i++) s[i] = (float)(rand() % 2000 - 1000) / 100.0f;
            for (int i = 0; i < n; i++) {
                tok[i] = rand() % (V - 1);           /* never the blank */
            }
            const int rc = mynah_asr_align_ctc(s, T, V, tok, n, BLANK, t0, t1);
            if (rc == 0) {
                for (int j = 0; j < n; j++) {
                    if (t0[j] < 0 || t1[j] < t0[j] || t1[j] >= T) bad_shape = 1;
                    if (j && t0[j] <= t1[j - 1]) bad_shape = 1;
                }
                if (T < mynah_asr_align_min_frames(tok, n)) bad_shape = 1;
            } else {
                refused++;
                if (T >= mynah_asr_align_min_frames(tok, n)) bad_shape = 1;  /* it did fit! */
            }
            free(s); free(tok); free(t0); free(t1);
        }
        snprintf(msg, sizeof(msg), "500 random cases monotone (%d refused as too short)", refused);
        CHECK(!bad_shape, msg);
        CHECK(refused > 0 && refused < 500, "the random sweep covers both outcomes");
    }

    /* ---- 7. bad arguments are refused, not crashed ---------------------- */
    {
        const int want[4] = {0, BLANK, 1, BLANK};
        const int tokens[2] = {0, 1};
        const int with_blank[2] = {0, BLANK};
        const int out_of_range[2] = {0, V};
        float *s = build(want, 4, 10.0f);
        int t0[2], t1[2];
        CHECK(mynah_asr_align_ctc(NULL, 4, V, tokens, 2, BLANK, t0, t1) == -1, "NULL scores");
        CHECK(mynah_asr_align_ctc(s, 4, V, NULL, 2, BLANK, t0, t1) == -1, "NULL tokens");
        CHECK(mynah_asr_align_ctc(s, 4, V, tokens, 0, BLANK, t0, t1) == -1, "0 tokens");
        CHECK(mynah_asr_align_ctc(s, 0, V, tokens, 2, BLANK, t0, t1) == -1, "0 frames");
        CHECK(mynah_asr_align_ctc(s, 4, V, tokens, 2, V, t0, t1) == -1, "blank outside the vocab");
        CHECK(mynah_asr_align_ctc(s, 4, V, with_blank, 2, BLANK, t0, t1) == -1,
              "a blank among the tokens");
        CHECK(mynah_asr_align_ctc(s, 4, V, out_of_range, 2, BLANK, t0, t1) == -1,
              "a token outside the vocab");
        CHECK(mynah_asr_align_min_frames(NULL, 0) == 0, "min_frames of nothing is 0");
        free(s);
    }

    return failures;
}
