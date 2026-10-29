/* Empty-turn recovery: replay boundary and publish gate (src/recovery.c).
 *
 * Model-free: the vocabulary and the emission frames are written by hand, so
 * each case states exactly what a transducer could emit around the boundary
 * between the replayed context and the current turn, and which text each rule
 * hands to the current turn.
 *
 * Exit: 0 ok, 1 fail. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/recovery.h"
#include "../src/tokenizer.h"

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("recovery FAIL: %s\n", msg); failures = 1; } \
    else printf("recovery ok:   %s\n", msg); } while (0)

#define WM "\xe2\x96\x81"      /* ▁ U+2581 */

static const char *const VOCAB[] = {
    "<unk>", WM "How", WM "do", WM "I", WM "open", WM "a", WM "joint", WM "acc", "ount",
    "?", ".", "!", WM "Yes", WM "please", WM ".", WM, "J", "o", "<en-US>", WM "account",
    ",", WM "Bonjour", WM "\xc2\xbf", "Qu\xc3\xa9", WM "?",
    "ye", "s", "\xe6\x88\x91", "\xe5\xa5\xbd", "\xe7\x9a\x84", "3",
};
enum {
    UNK, HOW, DO, I_, OPEN, A, JOINT, ACC, OUNT, QM, DOT, EXCL, YES, PLEASE, WM_DOT, WM_BARE,
    J, O, TAG_EN, ACCOUNT, COMMA, BONJOUR, WM_INVQ, QUE, WM_QM,
    YE, S_, ZH_WO, ZH_HAO, ZH_DE, DIGIT3, NVOCAB
};

static int load_vocab(mynah_asr_tokenizer *tk, char *path) {
    snprintf(path, 64, "/tmp/mynah_asr_rec_%d.json", (int)getpid());
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fputc('[', f);
    for (int i = 0; i < NVOCAB; i++) fprintf(f, "%s\"%s\"", i ? "," : "", VOCAB[i]);
    fputc(']', f);
    fclose(f);
    return mynah_asr_tokenizer_load(tk, path);
}

/* The text the current turn gets under a rule. */
static char *kept_text(const mynah_asr_tokenizer *tk, const int *tok, const int *fr, int n,
                       int keep_from, mynah_asr_keep_rule rule, const char *ctx) {
    const int i = mynah_asr_replay_keep_from(tk, tok, fr, n, keep_from, rule, ctx, 4);
    return mynah_asr_detokenize(tk, tok + i, n - i, NULL);
}

static void expect(const mynah_asr_tokenizer *tk, const char *name, const int *tok, const int *fr,
                   int n, int keep_from, const char *ctx, const char *want_frame,
                   const char *want_word, const char *want_text) {
    const char *want[3] = {want_frame, want_word, want_text};
    const char *rname[3] = {"FRAME", "WORD", "TEXT"};
    for (int r = 0; r < 3; r++) {
        char *got = kept_text(tk, tok, fr, n, keep_from, (mynah_asr_keep_rule)r, ctx);
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: %s -> \"%s\" (want \"%s\")", name, rname[r],
                 got ? got : "(null)", want[r]);
        CHECK(got && strcmp(got, want[r]) == 0, msg);
        free(got);
    }
}

#define N(a) ((int)(sizeof(a) / sizeof((a)[0])))

int main(void) {
    char path[64], msg[256];
    mynah_asr_tokenizer tk;
    if (load_vocab(&tk, path) != 0) { printf("recovery FAIL: vocab\n"); return 1; }
    const char *ctx_q = "How do I open a joint account?";

    /* 1. context ends "?": the transducer emits the "?" one frame past the
     *    boundary. FRAME hands it to the current turn. */
    {
        const int t[] = {HOW, DO, I_, OPEN, A, JOINT, ACC, OUNT, QM, YES, PLEASE};
        const int f[] = {1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14};
        expect(&tk, "context '?' past boundary", t, f, N(t), 10, ctx_q,
               "? Yes please", "Yes please", "Yes please");
    }
    /* 2. the same with "." and "!" */
    {
        const int t[] = {OPEN, A, JOINT, ACCOUNT, DOT, YES};
        const int f[] = {1, 2, 3, 5, 10, 12};
        expect(&tk, "context '.' past boundary", t, f, N(t), 10, "open a joint account.",
               ". Yes", "Yes", "Yes");
        const int t2[] = {OPEN, A, JOINT, ACCOUNT, EXCL, YES};
        expect(&tk, "context '!' past boundary", t2, f, N(t2), 10, "open a joint account!",
               "! Yes", "Yes", "Yes");
    }
    /* 3. a word split across the boundary: "acc" before, "ount?" after */
    {
        const int t[] = {OPEN, A, JOINT, ACC, OUNT, QM, YES};
        const int f[] = {1, 2, 3, 6, 9, 10, 12};
        expect(&tk, "split word past boundary", t, f, N(t), 8, ctx_q,
               "ount? Yes", "Yes", "Yes");
    }
    /* 4. the context's LAST WORD emitted whole past the boundary (emission lag):
     *    only the transcript of the context can tell it apart. */
    {
        const int t[] = {OPEN, A, JOINT, ACCOUNT, QM, YES, PLEASE};
        const int f[] = {1, 2, 3, 11, 12, 13, 14};
        expect(&tk, "delayed lexical word past boundary", t, f, N(t), 10, ctx_q,
               "account? Yes please", "account? Yes please", "Yes please");
        /* without the context transcript TEXT is WORD */
        char *got = kept_text(&tk, t, f, N(t), 10, MYNAH_ASR_KEEP_TEXT, NULL);
        CHECK(got && strcmp(got, "account? Yes please") == 0, "TEXT without context = WORD");
        free(got);
    }
    /* 5. no punctuation on the context, silence at the boundary */
    {
        const int t[] = {OPEN, A, JOINT, ACCOUNT, YES};
        const int f[] = {1, 2, 3, 4, 30};
        expect(&tk, "no punctuation, silence at boundary", t, f, N(t), 25, "open a joint account",
               "Yes", "Yes", "Yes");
    }
    /* 6. very short current turn: one word, and nothing at all */
    {
        const int t[] = {JOINT, ACCOUNT, QM, YES};
        const int f[] = {1, 2, 3, 11};
        expect(&tk, "one-word current turn", t, f, N(t), 10, ctx_q, "Yes", "Yes", "Yes");
        const int t2[] = {JOINT, ACCOUNT, QM};
        const int f2[] = {1, 11, 12};
        expect(&tk, "current turn decodes nothing new", t2, f2, N(t2), 10, ctx_q,
               "account?", "account?", "");
    }
    /* 7. a language tag past the boundary is not a first token */
    {
        const int t[] = {TAG_EN, ACCOUNT, QM, TAG_EN, YES};
        const int f[] = {0, 3, 10, 10, 11};
        expect(&tk, "language tag past boundary", t, f, N(t), 10, ctx_q, "? Yes", "Yes", "Yes");
    }
    /* 8. TEXT strips only a prefix that matches the context's tail ... */
    {
        const int t[] = {ACCOUNT, YES, ACCOUNT};
        const int f[] = {2, 11, 12};
        expect(&tk, "TEXT keeps words that do not complete the context", t, f, N(t), 10, ctx_q,
               "Yes account", "Yes account", "Yes account");
    }
    /* ... and its known cost: a current turn that STARTS by repeating the
     * context's last word loses that word. Pinned so a change is deliberate. */
    {
        const int t[] = {JOINT, ACCOUNT, PLEASE};
        const int f[] = {2, 11, 12};
        expect(&tk, "TEXT cost: a repeated last word is stripped", t, f, N(t), 10, ctx_q,
               "account please", "account please", "please");
    }
    /* 9. pieces: "▁." does not open a word; a bare ▁ before characters does */
    {
        const int t[] = {ACCOUNT, WM_DOT, WM_BARE, J, O};
        const int f[] = {1, 10, 11, 11, 12};
        expect(&tk, "'▁.' is punctuation, bare ▁ + chars is a word", t, f, N(t), 10, NULL,
               ". Jo", "Jo", "Jo");
        const int t2[] = {ACCOUNT, COMMA, YES};
        const int f2[] = {1, 10, 11};
        expect(&tk, "',' past boundary", t2, f2, N(t2), 10, NULL, ", Yes", "Yes", "Yes");
    }
    /* 10. everything before the boundary -> n */
    {
        const int t[] = {OPEN, ACCOUNT};
        const int f[] = {1, 2};
        const int i = mynah_asr_replay_keep_from(&tk, t, f, N(t), 10, MYNAH_ASR_KEEP_WORD, NULL, 4);
        CHECK(i == 2, "nothing past the boundary -> n");
        CHECK(mynah_asr_replay_keep_from(&tk, t, f, 0, 0, MYNAH_ASR_KEEP_TEXT, ctx_q, 4) == 0,
              "zero tokens -> 0");
    }

    /* 11. THE regression: the context is a question, the current turn's primary
     *     decode was EMPTY, the replay is context + current. Whatever rule ships,
     *     the current turn's published text must not start with the context's
     *     "?" -- by OWNERSHIP (the "?" continues "account", a context word), not
     *     by trimming punctuation: case 12 shows a current-turn "¿" survives. */
    {
        const int t[] = {HOW, DO, I_, OPEN, A, JOINT, ACCOUNT, QM, YES};
        const int f[] = {1, 2, 3, 4, 5, 6, 8, 10, 12};
        for (int r = MYNAH_ASR_KEEP_WORD; r <= MYNAH_ASR_KEEP_TEXT; r++) {
            char *got = kept_text(&tk, t, f, N(t), 10, (mynah_asr_keep_rule)r, ctx_q);
            snprintf(msg, sizeof(msg), "REGRESSION '?': rule %d publishes \"%s\", not \"?...\"", r,
                     got ? got : "(null)");
            CHECK(got && got[0] != '?' && strcmp(got, "Yes") == 0, msg);
            free(got);
        }
        /* and the naive rule is the one that fails it: recorded, not shipped */
        char *got = kept_text(&tk, t, f, N(t), 10, MYNAH_ASR_KEEP_FRAME, ctx_q);
        CHECK(got && got[0] == '?', "FRAME rule leaks the context's '?' (the bug the rules fix)");
        free(got);
    }
    /* 12. current-turn leading punctuation is kept: "▁¿" opens the word "¿Qué" */
    {
        const int t[] = {ACCOUNT, QM, WM_INVQ, QUE, QM};
        const int f[] = {3, 10, 11, 12, 13};
        expect(&tk, "current-turn '¿Qué?' keeps its own '¿'", t, f, N(t), 10, ctx_q,
               "? \xc2\xbfQu\xc3\xa9?", "\xc2\xbfQu\xc3\xa9?", "\xc2\xbfQu\xc3\xa9?");
    }
    /* 13. a marked punctuation piece ("▁?", French spacing) belongs to the word before */
    {
        const int t[] = {BONJOUR, WM_QM, YES};
        const int f[] = {5, 10, 11};
        expect(&tk, "'▁?' past boundary follows its word", t, f, N(t), 10, "Bonjour ?",
               "? Yes", "Yes", "Yes");
    }
    /* 14. a token exactly ON the boundary frame is the current turn's; one frame
     *     before is the context's; one frame after is the current turn's */
    {
        const int t[] = {ACCOUNT, YES, PLEASE};
        const int on[] = {3, 10, 11}, before[] = {3, 9, 11}, after[] = {3, 11, 12};
        expect(&tk, "word opened exactly on keep_from", t, on, N(t), 10, NULL,
               "Yes please", "Yes please", "Yes please");
        expect(&tk, "word opened one frame before keep_from", t, before, N(t), 10, NULL,
               "please", "please", "please");
        expect(&tk, "word opened one frame after keep_from", t, after, N(t), 10, NULL,
               "Yes please", "Yes please", "Yes please");
    }
    /* 15. punctuation-only current output: nothing the current turn owns */
    {
        const int t[] = {JOINT, ACCOUNT, QM, DOT};
        const int f[] = {1, 3, 11, 12};
        expect(&tk, "punctuation-only output past boundary", t, f, N(t), 10, ctx_q,
               "?.", "", "");
        const int t2[] = {WM_DOT, EXCL};
        const int f2[] = {11, 12};
        expect(&tk, "punctuation-only current turn, no context tokens", t2, f2, N(t2), 10, NULL,
               ".!", "", "");
    }

    /* 16. class C from the real replays (E12): the context ends "... thank you."
     *     and the decoder opens "you" ON the boundary frame (0-1 frames late);
     *     its "." waits for the next word. WORD keeps "you", TEXT gives it back. */
    {
        const int t[] = {OPEN, A, JOINT, ACCOUNT, YES, DOT, YES, PLEASE};
        const int f[] = {50, 55, 60, 70, 74, 88, 88, 90};
        /* the context text ends with "yes." here: "you" is spelled YES in this vocabulary */
        expect(&tk, "C: previous word opened 1 frame after keep_from", t, f, N(t), 73,
               "open a joint account yes.", "Yes. Yes please", "Yes. Yes please", "Yes please");
    }
    /* 17. TEXT's window: the same repeated word opened one chunk (4 frames) or
     *     more after the boundary is the current turn's -- the known cost is
     *     bounded to the first chunk. */
    {
        const int t[] = {JOINT, ACCOUNT, PLEASE};
        const int in_w[] = {2, 13, 14}, out_w[] = {2, 14, 15};
        expect(&tk, "TEXT window: repeat opened 3 frames after keep_from is given back", t, in_w, N(t), 10,
               ctx_q, "account please", "account please", "please");
        expect(&tk, "TEXT window: repeat opened 4 frames after keep_from stays current", t, out_w, N(t), 10,
               ctx_q, "account please", "account please", "account please");
        const int k0 = mynah_asr_replay_keep_from(&tk, t, out_w, N(t), 10, MYNAH_ASR_KEEP_TEXT, ctx_q, 0);
        char *got = mynah_asr_detokenize(&tk, t + k0, N(t) - k0, NULL);
        CHECK(got && strcmp(got, "please") == 0, "TEXT without a window (lag 0) gives back any repeat");
        free(got);
    }
    /* 18. the current turn's first word opens with a bare ▁ emitted one frame
     *     BEFORE the boundary (the chunk's lookahead lets the decoder start it
     *     early) and its letters after it: "▁" | "ye" "s". The word is the
     *     current turn's: a mark with no content yet does not make the context's
     *     last word open. Without this, WORD skips the unmarked "ye" "s". */
    {
        const int t[] = {JOINT, ACCOUNT, QM, WM_BARE, YE, S_, PLEASE};
        const int f[] = {1, 2, 3, 9, 10, 10, 12};
        expect(&tk, "bare ▁ before the boundary, letters after: 'yes' is kept", t, f, N(t), 10,
               ctx_q, "yes please", "yes please", "yes please");
        /* ... while a lexical context word's tail after the boundary still goes back */
        const int t2[] = {JOINT, ACC, OUNT, QM, YES};
        const int f2[] = {1, 2, 10, 10, 12};
        expect(&tk, "a lexical context word's tail is still given back", t2, f2, N(t2), 10, ctx_q,
               "ount? Yes", "Yes", "Yes");
    }
    /* 19. a language tag ends the context's word: the unmarked piece after it
     *     opens the current turn's word ("." "<en-US>" | "J" "o"). */
    {
        const int t[] = {ACCOUNT, DOT, TAG_EN, J, O, PLEASE};
        const int f[] = {1, 2, 3, 10, 10, 12};
        expect(&tk, "a tag ends the context's word: 'Jo' after it is kept", t, f, N(t), 10, NULL,
               "Jo please", "Jo please", "Jo please");
        const int t2[] = {ACCOUNT, QM, WM_BARE, TAG_EN, J, O};
        const int f2[] = {1, 10, 10, 10, 11, 11};
        expect(&tk, "'?' + bare ▁ + tag past the boundary open nothing", t2, f2, N(t2), 10, NULL,
               "? Jo", "Jo", "Jo");
    }
    /* 20. a script written without spaces: the kept chunks are the inside of one
     *     long run, so its unmarked pieces are the current turn's, also when a
     *     later mark (before a digit) follows. */
    {
        const int t[] = {WM_BARE, ZH_WO, ZH_HAO, ZH_DE};
        const int f[] = {1, 2, 10, 11};
        expect(&tk, "unspaced script: the run after the boundary is kept", t, f, N(t), 10,
               "\xe6\x88\x91", "\xe5\xa5\xbd\xe7\x9a\x84", "\xe5\xa5\xbd\xe7\x9a\x84",
               "\xe5\xa5\xbd\xe7\x9a\x84");
        const int t2[] = {WM_BARE, ZH_WO, ZH_HAO, ZH_DE, WM_BARE, DIGIT3};
        const int f2[] = {1, 2, 10, 11, 12, 12};
        expect(&tk, "unspaced script with a later mark: the leading run is kept", t2, f2, N(t2), 10,
               "\xe6\x88\x91", "\xe5\xa5\xbd\xe7\x9a\x84 3", "\xe5\xa5\xbd\xe7\x9a\x84 3",
               "\xe5\xa5\xbd\xe7\x9a\x84 3");
    }
    /* 21. only the context's tail after the boundary: the current turn owns
     *     nothing (EMPTY), as before -- no fallback to the whole slice. */
    {
        const int t[] = {JOINT, ACC, OUNT, QM, WM_QM};
        const int f[] = {1, 2, 10, 10, 11};
        expect(&tk, "only the context's tail past the boundary -> nothing", t, f, N(t), 10, ctx_q,
               "ount? ?", "", "");
    }

    /* ---- gate ---------------------------------------------------------- */
    struct { const char *text, *lang, *det, *ctx; mynah_asr_gate_verdict want; } g[] = {
        {"", "en-US", NULL, NULL, MYNAH_ASR_GATE_EMPTY},
        {" ? . ,", "en-US", NULL, NULL, MYNAH_ASR_GATE_EMPTY},
        {"\xe3\x80\x82\xef\xbc\x9f", "ja-JP", NULL, NULL, MYNAH_ASR_GATE_EMPTY},
        {"Yes please", "en-US", "", ctx_q, MYNAH_ASR_GATE_OK},
        {"Bonjour", "fr-FR", "fr-FR", NULL, MYNAH_ASR_GATE_OK},
        {"Hello", "en-US", "en-GB", NULL, MYNAH_ASR_GATE_OK},
        {"Bonjour", "en-US", "fr-FR", NULL, MYNAH_ASR_GATE_LANG},
        {"Bonjour", "auto", "fr-FR", NULL, MYNAH_ASR_GATE_OK},
        {"Bonjour", NULL, "fr-FR", NULL, MYNAH_ASR_GATE_OK},
        {"\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82", "ru-RU", "ru-RU", NULL, MYNAH_ASR_GATE_OK},
        {"\xe3\x81\x93\xe3\x82\x93\xe4\xb8\x96\xe7\x95\x8c", "ja-JP", "", NULL, MYNAH_ASR_GATE_OK},
        {"123", "en-US", "", NULL, MYNAH_ASR_GATE_OK},
        {"account?", "en-US", "", ctx_q, MYNAH_ASR_GATE_ECHO},
        {"Joint account.", "en-US", "", ctx_q, MYNAH_ASR_GATE_ECHO},
        {"account please", "en-US", "", ctx_q, MYNAH_ASR_GATE_OK},
        {"How do I open a joint account?", "en-US", "", ctx_q, MYNAH_ASR_GATE_OK},
    };
    for (int i = 0; i < N(g); i++) {
        const mynah_asr_gate_verdict v = mynah_asr_recovery_gate(g[i].text, g[i].lang, g[i].det, g[i].ctx);
        snprintf(msg, sizeof(msg), "gate \"%s\" [%s, tagged %s] -> %s (want %s)", g[i].text,
                 g[i].lang ? g[i].lang : "-", g[i].det && *g[i].det ? g[i].det : "-",
                 mynah_asr_gate_name(v), mynah_asr_gate_name(g[i].want));
        CHECK(v == g[i].want, msg);
    }
    CHECK(!mynah_asr_text_has_content("?") && mynah_asr_text_has_content("a") &&
          !mynah_asr_text_has_content(NULL), "has_content");

    /* ---- recovery: plan, attempt audio, vote ------------------------- */
    {
        mynah_asr_recover_cfg c;
        mynah_asr_recover_cfg_default(&c);
        mynah_asr_attempt pl[MYNAH_ASR_RECOVER_MAX_ATTEMPTS];
        const int fs = 1280;
        int n = mynah_asr_recover_plan(&c, 6 * fs, fs, pl);
        CHECK(n == 4 && pl[0].kind == MYNAH_ASR_ATTEMPT_PHASE && pl[0].prefix_frames == 3 &&
              pl[1].prefix_frames == 4 && pl[2].prefix_frames == 6 &&
              pl[3].kind == MYNAH_ASR_ATTEMPT_REPLAY && pl[3].prefix_frames == 6,
              "plan: phases 3/4/6 frames, then a 6-frame replay");
        CHECK(mynah_asr_recover_plan(&c, 6 * fs - 1, fs, pl) == 3, "plan: no replay vote without 6 frames of previous turn");
        c.replay_frames = 0;
        CHECK(mynah_asr_recover_plan(&c, 100 * fs, fs, pl) == 3, "plan: replay off");

        float cur[100], prev[8 * 1280], out[8 * 1280 + 100], out2[8 * 1280 + 100];
        for (int i = 0; i < 100; i++) cur[i] = 0.5f;
        for (int i = 0; i < 8 * fs; i++) prev[i] = (float)i;
        const mynah_asr_attempt ph = {MYNAH_ASR_ATTEMPT_PHASE, 3}, rp = {MYNAH_ASR_ATTEMPT_REPLAY, 6};
        size_t len = mynah_asr_recover_build(&ph, fs, cur, 100, prev, 8 * fs, out);
        (void)mynah_asr_recover_build(&ph, fs, cur, 100, NULL, 0, out2);
        double e = 0.0;
        for (int i = 0; i < 3 * fs; i++) e += (double)out[i] * out[i];
        const double rms = sqrt(e / (3 * fs));
        CHECK(len == (size_t)(3 * fs + 100) && out[3 * fs] == 0.5f && out[3 * fs + 99] == 0.5f,
              "phase attempt: prefix of 3 frames, then the turn");
        CHECK(memcmp(out, out2, sizeof(float) * (size_t)(3 * fs)) == 0, "phase prefix is deterministic");
        snprintf(msg, sizeof(msg), "phase prefix is -60 dBFS noise (rms %.5f)", rms);
        CHECK(rms > 0.0008 && rms < 0.0012, msg);
        len = mynah_asr_recover_build(&rp, fs, cur, 100, prev, 8 * fs, out);
        CHECK(len == (size_t)(6 * fs + 100) && out[0] == (float)(2 * fs) && out[6 * fs - 1] == (float)(8 * fs - 1) &&
              out[6 * fs] == 0.5f, "replay attempt: the previous turn's last 6 frames, then the turn");

        const char *v1[] = {"", "Yes please.", "yes, please", "No."};
        CHECK(mynah_asr_recover_vote(v1, 4, 2) == 1, "vote: two attempts agree on the words");
        const char *v2[] = {"a", "b", "c"};
        CHECK(mynah_asr_recover_vote(v2, 3, 2) == -1, "vote: no agreement -> nothing accepted");
        const char *v3[] = {NULL, "?", "?", "x"};
        CHECK(mynah_asr_recover_vote(v3, 4, 2) == -1, "vote: contentless texts never vote");
        const char *v4[] = {"Uh.", "uh", "Uh?"};
        CHECK(mynah_asr_recover_vote(v4, 3, 3) == 0, "vote: three of three");
        /* early decision == the full vote, exhaustively: every sequence of 4
         * attempts over {empty, a, b, c} and every prefix the partial vote
         * calls final must give the full vote's answer for EVERY completion */
        {
            const char *alpha[4] = {"", "a", "b", "c"};
            int bad = 0, final_early = 0, reject_early = 0;
            for (int code = 0; code < 256; code++) {
                const char *seq[4];
                for (int i = 0; i < 4; i++) seq[i] = alpha[(code >> (2 * i)) & 3];
                for (int k = 0; k <= 4; k++) {
                    const int d = mynah_asr_recover_vote_partial(seq, k, 4, 2);
                    if (d == MYNAH_ASR_RECOVER_OPEN) { if (k == 4) bad++; continue; }
                    if (k < 4) { final_early += d >= 0; reject_early += d == -1; }
                    for (int rest = 0; rest < (1 << (2 * (4 - k))); rest++) {
                        const char *full[4];
                        for (int i = 0; i < 4; i++)
                            full[i] = i < k ? seq[i] : alpha[(rest >> (2 * (i - k))) & 3];
                        if (mynah_asr_recover_vote(full, 4, 2) != d) bad++;
                    }
                    break;
                }
            }
            snprintf(msg, sizeof(msg), "partial vote is exact: 256 sequences, %d early accepts, "
                     "%d early rejects, %d mismatches", final_early, reject_early, bad);
            CHECK(bad == 0 && final_early > 0 && reject_early > 0, msg);
            const char *e3[] = {"", "", ""};
            CHECK(mynah_asr_recover_vote_partial(e3, 3, 4, 2) == -1, "three empty attempts of four: rejected without the fourth");
            const char *a2[] = {"Yes.", "yes"};
            CHECK(mynah_asr_recover_vote_partial(a2, 2, 4, 2) == 0, "the first two agree: accepted after two");
            const char *xy[] = {"x", "y", "y"};
            CHECK(mynah_asr_recover_vote_partial(xy, 3, 4, 2) == MYNAH_ASR_RECOVER_OPEN,
                  "x y y: still open (a fourth x would make x the winner)");
        }
        CHECK(mynah_asr_text_same_words("Yes, please.", "yes please") && !mynah_asr_text_same_words("yes", "yes please") &&
              !mynah_asr_text_same_words("", ""), "same_words");
    }

    mynah_asr_tokenizer_free(&tk);
    unlink(path);
    printf(failures ? "recovery: FAIL\n" : "recovery: PASS\n");
    return failures;
}
