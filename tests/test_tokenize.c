/* text -> piece ids (src/tokenizer.c), the direction the forced aligner needs.
 *
 * Model-free: the vocabularies here are written by hand, so every expectation is
 * exact instead of "looks plausible". The property that matters is the ROUND TRIP —
 * detokenize(tokenize(text)) == text — because that is what makes the alignment
 * meaningful: the ids have to spell the words whose times we are looking for.
 *
 * Exit: 0 ok, 1 fail. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/tokenizer.h"

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("tokenize FAIL: %s\n", msg); failures = 1; } \
    else printf("tokenize ok:   %s\n", msg); } while (0)

#define WM "\xe2\x96\x81"      /* ▁ U+2581 */

/* Builds a tokens.json out of the given pieces and loads it. */
static int load_vocab(mynah_asr_tokenizer *tk, const char *const *pieces, int n, char *path) {
    snprintf(path, 64, "/tmp/mynah_asr_tok_%d.json", (int)getpid());
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fputc('[', f);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s\"", i ? "," : "");
        for (const char *p = pieces[i]; *p; p++) {
            if ((unsigned char)*p < 0x20 || *p == '"' || *p == '\\') fprintf(f, "\\u%04x", *p);
            else fputc(*p, f);
        }
        fputc('"', f);
    }
    fputc(']', f);
    fclose(f);
    return mynah_asr_tokenizer_load(tk, path);
}

static int roundtrip(mynah_asr_tokenizer *tk, const char *text, const char *want) {
    int ids[128], skipped = 0;
    char msg[512];
    const int n = mynah_asr_tokenize(tk, text, ids, 128, &skipped);
    if (n < 0) { CHECK(0, "tokenize returned -1"); return 0; }
    char *back = mynah_asr_detokenize(tk, ids, n, NULL);
    const int ok = back && strcmp(back, want) == 0;
    snprintf(msg, sizeof(msg), "\"%s\" -> %d ids -> \"%s\" (%d bytes skipped)",
             text, n, back ? back : "(null)", skipped);
    CHECK(ok, msg);
    free(back);
    return n;
}

int main(void) {
    char path[64], msg[256];

    /* ---- 1. a vocabulary that can spell the text exactly ---------------- */
    {
        static const char *const pieces[] = {
            WM "hello", WM "world", WM "a", "b", "c", WM "the", WM "cat",
            WM "s", "at", WM "ing", "s", "<unk>", "<blank>",
        };
        mynah_asr_tokenizer tk;
        CHECK(load_vocab(&tk, pieces, 13, path) == 0, "hand-written vocabulary loaded");
        roundtrip(&tk, "hello world", "hello world");
        roundtrip(&tk, "the cat", "the cat");
        /* longest match must prefer "▁hello" over "▁h"-less alternatives, and a
         * continuation piece must be used when no ▁ variant fits */
        const int n = roundtrip(&tk, "the cats", "the cats");
        snprintf(msg, sizeof(msg), "\"the cats\" costs 3 pieces (got %d)", n);
        CHECK(n == 3, msg);
        mynah_asr_tokenizer_free(&tk);
        unlink(path);
    }

    /* ---- 2. characters the vocabulary cannot spell are skipped, counted -- */
    {
        static const char *const pieces[] = {WM "ok", "<blank>"};
        mynah_asr_tokenizer tk;
        int ids[16], skipped = -1;
        CHECK(load_vocab(&tk, pieces, 2, path) == 0, "tiny vocabulary loaded");
        const int n = mynah_asr_tokenize(&tk, "ok zzz", ids, 16, &skipped);
        snprintf(msg, sizeof(msg), "unknown bytes skipped and reported (%d ids, %d skipped)",
                 n, skipped);
        CHECK(n == 1 && skipped == 3, msg);
        mynah_asr_tokenizer_free(&tk);
        unlink(path);
    }

    /* ---- 3. word starts: repeated words must repeat the ▁ piece ---------- */
    {
        static const char *const pieces[] = {WM "go", "go", "<blank>"};
        mynah_asr_tokenizer tk;
        int ids[16], skipped = 0;
        CHECK(load_vocab(&tk, pieces, 3, path) == 0, "start/continuation vocabulary loaded");
        const int n = mynah_asr_tokenize(&tk, "go go", ids, 16, &skipped);
        snprintf(msg, sizeof(msg), "\"go go\" -> [%d, %d], both word starts",
                 n > 0 ? ids[0] : -1, n > 1 ? ids[1] : -1);
        CHECK(n == 2 && ids[0] == 0 && ids[1] == 0, msg);
        /* "gogo" instead has one start and one continuation */
        const int n2 = mynah_asr_tokenize(&tk, "gogo", ids, 16, &skipped);
        snprintf(msg, sizeof(msg), "\"gogo\" -> [%d, %d], start then continuation",
                 n2 > 0 ? ids[0] : -1, n2 > 1 ? ids[1] : -1);
        CHECK(n2 == 2 && ids[0] == 0 && ids[1] == 1, msg);
        mynah_asr_tokenizer_free(&tk);
        unlink(path);
    }

    /* ---- 4. cap and bad arguments --------------------------------------- */
    {
        static const char *const pieces[] = {WM "a", WM "b", "<blank>"};
        mynah_asr_tokenizer tk;
        int ids[2], skipped = 0;
        CHECK(load_vocab(&tk, pieces, 3, path) == 0, "cap vocabulary loaded");
        CHECK(mynah_asr_tokenize(&tk, "a b a b", ids, 2, &skipped) == 2, "cap is respected");
        CHECK(mynah_asr_tokenize(&tk, "a", NULL, 4, &skipped) == -1, "no output buffer");
        CHECK(mynah_asr_tokenize(&tk, NULL, ids, 2, &skipped) == -1, "no text");
        CHECK(mynah_asr_tokenize(&tk, "a", ids, 0, &skipped) == -1, "zero capacity");
        CHECK(mynah_asr_tokenize(&tk, "", ids, 2, &skipped) == 0, "empty text -> 0 ids");
        mynah_asr_tokenizer_free(&tk);
        unlink(path);
    }

    /* ---- 5. specials are never emitted ---------------------------------
     * The aligner's vocabulary contains <unk>/<blank>; a tokenization that used
     * them would put non-text into the alignment. */
    {
        static const char *const pieces[] = {"<unk>", WM "x", "<blank>"};
        mynah_asr_tokenizer tk;
        int ids[16], skipped = 0;
        CHECK(load_vocab(&tk, pieces, 3, path) == 0, "vocabulary with specials loaded");
        const int n = mynah_asr_tokenize(&tk, "x ???", ids, 16, &skipped);
        int used_special = 0;
        for (int i = 0; i < n; i++) if (ids[i] == 0 || ids[i] == 2) used_special = 1;
        snprintf(msg, sizeof(msg), "specials never emitted (%d ids, %d skipped)", n, skipped);
        CHECK(!used_special && skipped == 3, msg);
        mynah_asr_tokenizer_free(&tk);
        unlink(path);
    }

    return failures;
}
