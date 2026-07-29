#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../vendor/cJSON.h"

int mynah_asr_tokenizer_load(mynah_asr_tokenizer *tk, const char *path) {
    memset(tk, 0, sizeof(*tk));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tokenizer: cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); fclose(f); return -1; }
    buf[len] = '\0';
    fclose(f);

    cJSON *arr = cJSON_Parse(buf);
    free(buf);
    if (!arr || !cJSON_IsArray(arr)) { cJSON_Delete(arr); return -1; }

    tk->n_pieces = cJSON_GetArraySize(arr);
    tk->pieces = calloc((size_t)tk->n_pieces, sizeof(char *));
    int i = 0;
    for (cJSON *it = arr->child; it; it = it->next, i++)
        tk->pieces[i] = strdup(cJSON_IsString(it) ? it->valuestring : "");
    cJSON_Delete(arr);
    return 0;
}

void mynah_asr_tokenizer_free(mynah_asr_tokenizer *tk) {
    for (int i = 0; i < tk->n_pieces; i++) free(tk->pieces[i]);
    free(tk->pieces);
    tk->pieces = NULL;
}

void mynah_asr_words_free(mynah_asr_word *words, int n_words) {
    if (!words) return;
    for (int i = 0; i < n_words; i++) free(words[i].word);
    free(words);
}

/* piece "speciale" <...>: tag lingua o marcatore di controllo -> non è testo */
static int piece_is_special(const char *p, size_t pl) {
    return pl >= 2 && p[0] == '<' && p[pl - 1] == '>';
}

/* il piece inizia con ▁ (U+2581, e2 96 81)? */
static int piece_starts_word(const char *p) {
    return (unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 &&
           (unsigned char)p[2] == 0x81;
}

int mynah_asr_detokenize_words(const mynah_asr_tokenizer *tk, const int *tokens,
                           const int *frames, int n, double frame_sec,
                           mynah_asr_word **out, int *n_out) {
    *out = NULL;
    *n_out = 0;
    mynah_asr_word *words = calloc((size_t)(n > 0 ? n : 1), sizeof(mynah_asr_word));
    char *wbuf = malloc(256);
    if (!words || !wbuf) { free(words); free(wbuf); return -1; }
    size_t wcap = 256, wlen = 0;
    int nw = 0, first_frame = 0, last_frame = 0;

    for (int i = 0; i <= n; i++) {
        const char *p = NULL;
        size_t pl = 0;
        int special = 1, starts = 0;
        if (i < n && tokens[i] >= 0 && tokens[i] < tk->n_pieces) {
            p = tk->pieces[tokens[i]];
            pl = strlen(p);
            special = piece_is_special(p, pl);
            starts = pl >= 3 && piece_starts_word(p);
        }
        /* chiudi la parola corrente a fine sequenza o all'inizio della prossima */
        if (wlen > 0 && (i == n || (!special && starts))) {
            char *w = malloc(wlen + 1);
            if (!w) { mynah_asr_words_free(words, nw); free(wbuf); return -1; }
            memcpy(w, wbuf, wlen);
            w[wlen] = '\0';
            words[nw].word = w;
            words[nw].t0 = first_frame * frame_sec;
            words[nw].t1 = (last_frame + 1) * frame_sec;
            nw++;
            wlen = 0;
        }
        if (i == n || special) continue;

        if (wlen == 0) first_frame = frames[i];
        last_frame = frames[i];
        if (wlen + pl + 1 > wcap) {
            wcap = (wcap + pl + 1) * 2;
            char *nb = realloc(wbuf, wcap);
            if (!nb) { mynah_asr_words_free(words, nw); free(wbuf); return -1; }
            wbuf = nb;
        }
        /* copia saltando il ▁ iniziale (e ogni ▁ interno -> niente: è un confine) */
        for (size_t j = starts ? 3 : 0; j < pl; j++) wbuf[wlen++] = p[j];
    }
    free(wbuf);
    *out = words;
    *n_out = nw;
    return 0;
}

char *mynah_asr_detokenize(const mynah_asr_tokenizer *tk, const int *tokens, int n,
                       char *lang_out) {
    size_t cap = 256, len = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    if (lang_out) lang_out[0] = '\0';

    for (int i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= tk->n_pieces) continue;
        const char *p = tk->pieces[tokens[i]];
        const size_t pl = strlen(p);

        /* token speciale <...>: tag lingua (contiene '-') o marcatore -> strip */
        if (pl >= 2 && p[0] == '<' && p[pl - 1] == '>') {
            if (lang_out && memchr(p, '-', pl) && pl - 2 < 16) {
                memcpy(lang_out, p + 1, pl - 2);
                lang_out[pl - 2] = '\0';
            }
            continue;
        }
        if (len + pl * 3 + 1 > cap) {
            cap = (cap + pl * 3 + 1) * 2;
            char *nb = realloc(out, cap);
            if (!nb) { free(out); return NULL; }
            out = nb;
        }
        /* copia sostituendo ▁ (U+2581, e2 96 81) con spazio */
        for (size_t j = 0; j < pl; ) {
            if (j + 2 < pl && (unsigned char)p[j] == 0xE2 && (unsigned char)p[j + 1] == 0x96 &&
                (unsigned char)p[j + 2] == 0x81) {
                out[len++] = ' ';
                j += 3;
            } else {
                out[len++] = p[j++];
            }
        }
    }
    out[len] = '\0';

    /* Il tag lingua può arrivare anche "spelled-out" in pezzi BPE normali quando
     * il token singolo non esiste nel vocab (es. <sl-SI> vs token <sl-SL>):
     * rimuovi i pattern <xx-XX> inline e usali come lang se non già rilevata. */
    for (char *p = out; (p = strchr(p, '<')) != NULL;) {
        char *close = strchr(p, '>');
        if (!close || (size_t)(close - p) > 12 || !memchr(p, '-', (size_t)(close - p))) {
            p++;
            continue;
        }
        if (lang_out && !lang_out[0] && (size_t)(close - p - 1) < 16) {
            memcpy(lang_out, p + 1, (size_t)(close - p - 1));
            lang_out[close - p - 1] = '\0';
        }
        memmove(p, close + 1, strlen(close + 1) + 1);
        /* comprimi l'eventuale doppio spazio risultante */
        if (p > out && p[-1] == ' ' && p[0] == ' ')
            memmove(p, p + 1, strlen(p + 1) + 1);
    }
    len = strlen(out);

    /* strip degli spazi iniziali (come l'oracolo: lstrip) */
    size_t skip = 0;
    while (out[skip] == ' ') skip++;
    if (skip) memmove(out, out + skip, len - skip + 1);
    /* strip degli spazi finali */
    len = strlen(out);
    while (len > 0 && out[len - 1] == ' ') out[--len] = '\0';
    return out;
}

int mynah_asr_tok_find(const mynah_asr_tokenizer *tk, const char *piece) {
    for (int i = 0; i < tk->n_pieces; i++)
        if (strcmp(tk->pieces[i], piece) == 0) return i;
    return -1;
}
