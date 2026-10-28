/* Empty-turn recovery lab (S18-1b, .work/empty-turn-recovery.md).
 *
 * Research tool, not a gate: built on demand (make tests/recovery_lab), never by
 * `make test`. For each item it decodes, through the streaming path exactly as a
 * server turn would:
 *
 *   ctx      the previous turn (a whole clip) with the session language: its
 *            transcript is what the TEXT rule and the echo gate get;
 *   primary  the current turn [start, end) of another clip, from a reset stream;
 *   retry    the current turn again with the "auto" prompt (when the session
 *            language is not auto already);
 *   replay   for each context length: [last N ms of ctx][current turn] from a
 *            reset stream with the session language, then the three boundary
 *            rules and the gate on what each keeps. ctx_ms -1 = the whole
 *            previous clip (context prepended, as a continuing stream would).
 *
 * Every arm runs on every item, empty primary or not, so the boundary is
 * measured on all the data; whether recovery would have FIRED is the primary's
 * emptiness, recorded with it.
 *
 * Input (argv[2] or stdin): one item per line, tab-separated:
 *   id  lang  ctx_wav  cur_wav  cur_start_s  cur_end_s
 * Output: one JSON object per item on stdout.
 *
 * Usage: recovery_lab MODEL_DIR [items.tsv] [--quant int8] [--lookahead L]
 *                     [--ctx-ms 500,1000,2000,3000] [--dump-tokens] [--recover]
 * --recover: for a primary-EMPTY turn, also the library's own recovery
 * (mynah_asr_recover_turn, default config) with the whole previous clip as
 * the previous turn: "recover": {accepted, text, votes, attempts, kind}.
 * --dump-tokens: also every token with its emission frame for the previous
 * turn alone ("ctx_tokens", frames from the clip's start), the primary and each
 * replay ("tokens"), plus "ctx_samples" and "prefix_samples" to map a frame of
 * the previous turn decoded alone onto the replay. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/audio.h"
#include "../src/mynah_asr.h"
#include "../src/recover.h"
#include "../src/recovery.h"
#include "../src/tokenizer.h"

#define MAX_CTX 8
#define BOUNDARY_SHOW 6   /* frames either side of keep_from dumped per replay */


/* A JSON string. A vocabulary piece can be part of a multi-byte character
 * (byte-fallback pieces): bytes that do not form valid UTF-8 are written as
 * the text "<0xNN>", so the line stays valid JSON and the byte stays visible. */
static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p;) {
        const int len = p[0] < 0x80 ? 1 : (p[0] & 0xE0) == 0xC0 ? 2 : (p[0] & 0xF0) == 0xE0 ? 3
                      : (p[0] & 0xF8) == 0xF0 ? 4 : 0;
        int ok = len > 0;
        for (int k = 1; ok && k < len; k++) ok = (p[k] & 0xC0) == 0x80;
        if (!ok) { fprintf(f, "<0x%02X>", p[0]); p++; continue; }
        if (len > 1) { fwrite(p, 1, (size_t)len, f); p += len; continue; }
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p < 0x20) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
        p++;
    }
    fputc('"', f);
}

static void dump_tokens(FILE *f, const mynah_asr_tokenizer *tk, const int *tok, const int *fr, int n) {
    fputc('[', f);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s[%d,", i ? "," : "", fr[i]);
        json_str(f, (tok[i] >= 0 && tok[i] < tk->n_pieces) ? tk->pieces[tok[i]] : "?");
        fputc(']', f);
    }
    fputc(']', f);
}

/* Decode samples on a reset stream; the stream keeps tokens/frames/text. */
static int decode(mynah_asr_stream *s, const char *lang, const float *a, size_t na,
                  const float *b, size_t nb) {
    if (mynah_asr_stream_reset(s, lang) != 0) return -1;
    if (na && mynah_asr_stream_feed(s, a, na, NULL, NULL) != 0) return -1;
    if (nb && mynah_asr_stream_feed(s, b, nb, NULL, NULL) != 0) return -1;
    return mynah_asr_stream_finish(s, NULL, NULL);
}

static float *load16k(const char *path, size_t *n) {
    int sr = 0;
    float *x = mynah_asr_wav_load(path, n, &sr);
    if (!x) return NULL;
    if (sr != 16000) {
        size_t m = 0;
        float *y = mynah_asr_resample(x, *n, sr, 16000, &m);
        free(x);
        *n = m;
        return y;
    }
    return x;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL_DIR [items.tsv] [--quant int8] [--lookahead L] "
                        "[--ctx-ms 500,1000,2000,3000]\n", argv[0]);
        return 2;
    }
    const char *model_dir = argv[1], *items_path = NULL;
    int quant = MYNAH_ASR_QUANT_F32, lookahead = -1, dump = 0, recover = 0;
    int ctx_ms[MAX_CTX] = {500, 1000, 2000, 3000}, n_ctx = 4;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--quant") && i + 1 < argc) {
            quant = !strcmp(argv[++i], "int8") ? MYNAH_ASR_QUANT_INT8 : MYNAH_ASR_QUANT_F32;
        } else if (!strcmp(argv[i], "--lookahead") && i + 1 < argc) {
            lookahead = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--recover")) {
            recover = 1;
        } else if (!strcmp(argv[i], "--dump-tokens")) {
            dump = 1;
        } else if (!strcmp(argv[i], "--ctx-ms") && i + 1 < argc) {
            n_ctx = 0;
            for (char *t = strtok(argv[++i], ","); t && n_ctx < MAX_CTX; t = strtok(NULL, ","))
                ctx_ms[n_ctx++] = atoi(t);
        } else if (!items_path) {
            items_path = argv[i];
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }

    mynah_asr_model *m = mynah_asr_load_quant(model_dir, quant);
    if (!m) return 1;
    char tok_path[1024];
    snprintf(tok_path, sizeof(tok_path), "%s/tokens.json", model_dir);
    mynah_asr_tokenizer tk;
    if (mynah_asr_tokenizer_load(&tk, tok_path) != 0) { fprintf(stderr, "tokens.json\n"); return 1; }
    const int fs = mynah_asr_frame_samples(m);
    mynah_asr_stream *s = mynah_asr_stream_open(m, "auto", lookahead);
    if (!s) return 1;
    mynah_asr_recover_cfg rcfg;
    mynah_asr_recover_cfg_default(&rcfg);
    mynah_asr_recoverer *rec = recover ? mynah_asr_recoverer_open(m, lookahead, &rcfg) : NULL;
    if (recover && !rec) return 1;

    FILE *in = items_path ? fopen(items_path, "r") : stdin;
    if (!in) { perror(items_path); return 1; }

    char line[4096], ctx_loaded[1024] = "", ctx_lang[32] = "";
    float *ctx = NULL;
    size_t n_ctx_s = 0;
    char *ctx_text = NULL, *ctx_toks = NULL;
    int n_items = 0;
    while (fgets(line, sizeof(line), in)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (!line[0] || line[0] == '#') continue;
        char *f[6];
        int nf = 0;
        for (char *t = strtok(line, "\t"); t && nf < 6; t = strtok(NULL, "\t")) f[nf++] = t;
        if (nf < 6) { fprintf(stderr, "bad line (%d fields)\n", nf); continue; }
        const char *id = f[0], *lang = f[1], *ctx_wav = f[2], *cur_wav = f[3];
        const double t0 = atof(f[4]), t1 = atof(f[5]);

        /* the previous turn: decoded once per (clip, language) */
        if (strcmp(ctx_loaded, ctx_wav) != 0 || strcmp(ctx_lang, lang) != 0) {
            free(ctx); free(ctx_text);
            ctx = load16k(ctx_wav, &n_ctx_s);
            if (!ctx || decode(s, lang, ctx, n_ctx_s, NULL, 0) != 0) {
                fprintf(stderr, "%s: ctx decode failed\n", id);
                ctx_loaded[0] = '\0';
                ctx_text = NULL;
                continue;
            }
            ctx_text = strdup(mynah_asr_stream_text(s));
            free(ctx_toks);
            ctx_toks = NULL;
            if (dump) {
                size_t len = 0;
                FILE *m = open_memstream(&ctx_toks, &len);
                const int *ct, *cf;
                const int cn = mynah_asr_stream_tokens(s, &ct, &cf);
                dump_tokens(m, &tk, ct, cf, cn);
                fclose(m);
            }
            snprintf(ctx_loaded, sizeof(ctx_loaded), "%s", ctx_wav);
            snprintf(ctx_lang, sizeof(ctx_lang), "%s", lang);
        }
        size_t n_cur_all = 0;
        float *cur_all = load16k(cur_wav, &n_cur_all);
        if (!cur_all) continue;
        size_t a = (size_t)(t0 * 16000.0), b = t1 > 0 ? (size_t)(t1 * 16000.0) : n_cur_all;
        if (b > n_cur_all) b = n_cur_all;
        if (a > b) a = b;
        const float *cur = cur_all + a;
        const size_t n_cur = b - a;

        printf("{\"id\":");
        json_str(stdout, id);
        printf(",\"lang\":");
        json_str(stdout, lang);
        printf(",\"cur_s\":%.3f,\"ctx_text\":", (double)n_cur / 16000.0);
        json_str(stdout, ctx_text);

        /* primary */
        if (decode(s, lang, cur, n_cur, NULL, 0) != 0) { printf("}\n"); free(cur_all); continue; }
        const char *pt = mynah_asr_stream_text(s);
        const int empty = !mynah_asr_text_has_content(pt);
        printf(",\"primary\":");
        json_str(stdout, pt);
        printf(",\"primary_lang\":");
        json_str(stdout, mynah_asr_stream_lang(s));
        printf(",\"empty\":%d", empty);
        if (rec && empty) {
            mynah_asr_recover_result rr;
            if (mynah_asr_recover_turn(rec, lang, cur, n_cur, ctx, n_ctx_s, ctx_text, &rr) == 0) {
                printf(",\"recover\":{\"accepted\":%d,\"votes\":%d,\"attempts\":%d,\"kind\":\"%s\",\"text\":",
                       rr.accepted, rr.votes, rr.attempts,
                       !rr.accepted ? "" : rr.winner_kind == MYNAH_ASR_ATTEMPT_REPLAY ? "replay" : "phase");
                json_str(stdout, mynah_asr_recover_text(rec));
                printf("}");
            }
        }
        if (dump) {
            const int *pt_tok, *pt_fr;
            const int pn = mynah_asr_stream_tokens(s, &pt_tok, &pt_fr);
            printf(",\"ctx_samples\":%zu,\"ctx_tokens\":%s,\"primary_tokens\":", n_ctx_s,
                   ctx_toks ? ctx_toks : "[]");
            dump_tokens(stdout, &tk, pt_tok, pt_fr, pn);
        }

        /* retry with auto */
        if (strncmp(lang, "auto", 4) != 0) {
            if (decode(s, "auto", cur, n_cur, NULL, 0) == 0) {
                const char *rt = mynah_asr_stream_text(s);
                printf(",\"retry\":");
                json_str(stdout, rt);
                printf(",\"retry_lang\":");
                json_str(stdout, mynah_asr_stream_lang(s));
                printf(",\"retry_gate\":\"%s\"",
                       mynah_asr_gate_name(mynah_asr_recovery_gate(rt, lang, mynah_asr_stream_lang(s), ctx_text)));
            }
        }

        /* replays */
        printf(",\"replay\":[");
        for (int c = 0; c < n_ctx; c++) {
            /* ctx_ms < 0: the whole previous clip, i.e. the turn decoded as the
             * continuation of a stream that heard all of the context */
            size_t np = ctx_ms[c] < 0 ? n_ctx_s : (size_t)ctx_ms[c] * 16u;
            if (np > n_ctx_s) np = n_ctx_s;
            np -= np % (size_t)fs;                  /* the boundary on a whole encoder frame */
            const int keep_from = (int)(np / (size_t)fs);
            if (decode(s, lang, ctx + (n_ctx_s - np), np, cur, n_cur) != 0) continue;
            const int *tok, *fr;
            const int n = mynah_asr_stream_tokens(s, &tok, &fr);
            printf("%s{\"ctx_ms\":%d,\"keep_from\":%d,\"full\":", c ? "," : "",
                   (int)(np / 16u), keep_from);
            json_str(stdout, mynah_asr_stream_text(s));
            /* the pieces around the boundary, with their frames */
            printf(",\"around\":[");
            int first = 1;
            for (int i = 0; i < n; i++) {
                if (fr[i] < keep_from - BOUNDARY_SHOW || fr[i] > keep_from + BOUNDARY_SHOW) continue;
                printf("%s[%d,", first ? "" : ",", fr[i]);
                json_str(stdout, (tok[i] >= 0 && tok[i] < tk.n_pieces) ? tk.pieces[tok[i]] : "?");
                printf("]");
                first = 0;
            }
            printf("]");
            if (dump) {
                printf(",\"prefix_samples\":%zu,\"tokens\":", np);
                dump_tokens(stdout, &tk, tok, fr, n);
            }
            static const char *rn[3] = {"frame", "word", "text"};
            for (int r = 0; r < 3; r++) {
                const int k = mynah_asr_replay_keep_from(&tk, tok, fr, n, keep_from,
                                                         (mynah_asr_keep_rule)r, ctx_text,
                                                         mynah_asr_stream_chunk_frames(s));
                char *kt = mynah_asr_detokenize(&tk, tok + k, n - k, NULL);
                printf(",\"%s\":", rn[r]);
                json_str(stdout, kt);
                printf(",\"%s_gate\":\"%s\"", rn[r],
                       mynah_asr_gate_name(mynah_asr_recovery_gate(kt, lang, mynah_asr_stream_lang(s), ctx_text)));
                free(kt);
            }
            printf("}");
        }
        printf("]}\n");
        fflush(stdout);
        free(cur_all);
        n_items++;
    }
    fprintf(stderr, "recovery_lab: %d items\n", n_items);
    free(ctx); free(ctx_text);
    if (in != stdin) fclose(in);
    mynah_asr_recoverer_close(rec);
    mynah_asr_stream_close(s);
    mynah_asr_tokenizer_free(&tk);
    mynah_asr_free(m);
    return 0;
}
