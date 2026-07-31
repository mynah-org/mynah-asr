/* mynah-asr-server — OpenAI-compatible HTTP API + WebSocket streaming.
 *
 * Endpoints:
 *   POST /v1/audio/translations     as above + target_language (default en) — AED only
 *   POST /v1/audio/transcriptions   multipart (file=..., language, response_format,
 *                                   lookahead) -> json | text | verbose_json
 *   GET  /v1/models, /v1/health     info
 *   GET  /v1/audio/stream           WebSocket: binary s16le 16kHz in -> JSON deltas out
 *                                   (query: ?lang=auto&lookahead=3)
 *
 * Concurrency: the model is read-only (mmap'd weights) and shared; each request
 * only owns its decode state -> a pool of worker threads, no clones.
 * Cross-request batching (B>1 kernels) is in the backlog (see TODO M4).
 */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../src/audio.h"
#include "../src/backend.h"
#include "../src/mynah_asr.h"
#include "../src/qmat.h"      /* mynah_asr_set_caps (--caps) */
#include "../src/threads.h"   /* mynah_asr_blas_set_concurrency (inflight -> BLAS) */
#include "../vendor/cJSON.h"
#include "http_util.h"

#define MAX_HDR (64 * 1024)
#define MAX_BODY (200u * 1024 * 1024)
#define QUEUE_CAP 128

static mynah_asr_model *g_model;
static mynah_asr_model *g_lid;       /* --lid-model: language detector for language=auto */
static const char *g_model_name = "nemotron-3.5-asr-streaming-0.6b";
static int g_max_batch = 8;          /* --batch N; 1 = disabled */

/* Inferences computing RIGHT NOW (a batch call counts as one, however many items
 * it carries). Only the server knows this number, and BLAS needs it: N calls
 * each asking for every core thrash on the OpenBLAS lock instead of going
 * faster. Kept around the compute calls only, so an idle WebSocket stream — open
 * but between chunks — does not hold a slot down. */
static int g_inflight;
static pthread_mutex_t g_inflight_mu = PTHREAD_MUTEX_INITIALIZER;

static void inference_begin(void) {
    pthread_mutex_lock(&g_inflight_mu);
    const int n = ++g_inflight;
    pthread_mutex_unlock(&g_inflight_mu);
    mynah_asr_blas_set_concurrency(n);
}

static void inference_end(void) {
    pthread_mutex_lock(&g_inflight_mu);
    const int n = --g_inflight;
    pthread_mutex_unlock(&g_inflight_mu);
    mynah_asr_blas_set_concurrency(n > 0 ? n : 1);
}

static int inflight_now(void) {
    pthread_mutex_lock(&g_inflight_mu);
    const int n = g_inflight;
    pthread_mutex_unlock(&g_inflight_mu);
    return n;
}
static int g_quant = MYNAH_ASR_QUANT_F32;

/* --------------------------------------------------- micro-batching scheduler
 * Connections wrap their work into jobs; a dedicated thread aggregates the
 * pending jobs (a 25 ms window, or a full batch) and calls
 * mynah_asr_transcribe_batch: the weights are read once per layer for the whole
 * batch. */
typedef struct trx_job {
    const float *samples;
    size_t n_samples;
    char lang[24];
    int lookahead;
    char *text;                 /* result (malloc'd) */
    char lang_out[16];
    mynah_asr_word *words;          /* per-word timestamps (malloc'd, may be NULL) */
    int n_words;
    int done;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct trx_job *next;
} trx_job;

static trx_job *bq_head, *bq_tail;
static int bq_len;
static pthread_mutex_t bq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t bq_cv = PTHREAD_COND_INITIALIZER;

static void batch_submit_and_wait(trx_job *j) {
    pthread_mutex_lock(&bq_mu);
    if (bq_tail) bq_tail->next = j;
    else bq_head = j;
    bq_tail = j;
    bq_len++;
    pthread_cond_broadcast(&bq_cv);
    pthread_mutex_unlock(&bq_mu);

    pthread_mutex_lock(&j->mu);
    while (!j->done) pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);
}

static void *batch_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&bq_mu);
        while (!bq_head) pthread_cond_wait(&bq_cv, &bq_mu);

        /* aggregation window: wait up to 25 ms for more jobs to arrive */
        struct timespec dl;
        clock_gettime(CLOCK_REALTIME, &dl);
        dl.tv_nsec += 25 * 1000000;
        if (dl.tv_nsec >= 1000000000) { dl.tv_sec++; dl.tv_nsec -= 1000000000; }
        while (bq_len < g_max_batch &&
               pthread_cond_timedwait(&bq_cv, &bq_mu, &dl) == 0) {}

        trx_job *jobs[64];
        int B = 0;
        while (bq_head && B < g_max_batch && B < 64) {
            jobs[B++] = bq_head;
            bq_head = bq_head->next;
        }
        if (!bq_head) bq_tail = NULL;
        bq_len -= B;
        pthread_mutex_unlock(&bq_mu);
        if (B == 0) continue;   /* never happens with the wait on bq_head, but makes
                                   the bound provable (GCC 13 maybe-uninitialized) */

        const float *samples[64];
        size_t ns[64];
        const char *langs[64];
        char *texts[64];
        char louts[64][16];
        mynah_asr_word *wordsv[64];
        int nwordsv[64];
        int lookahead = jobs[0]->lookahead;   /* batch homogeneous on the first */
        for (int b = 0; b < B; b++) {
            samples[b] = jobs[b]->samples;
            ns[b] = jobs[b]->n_samples;
            langs[b] = jobs[b]->lang;
            texts[b] = NULL;
        }
        /* words are always extracted: negligible next to inference, and a batch
         * can mix json and verbose_json requests */
        inference_begin();
        mynah_asr_transcribe_batch_ts(g_model, samples, ns, B, langs, lookahead,
                                  texts, louts, wordsv, nwordsv);
        inference_end();

        for (int b = 0; b < B; b++) {
            pthread_mutex_lock(&jobs[b]->mu);
            jobs[b]->text = texts[b];
            jobs[b]->words = wordsv[b];
            jobs[b]->n_words = nwordsv[b];
            memcpy(jobs[b]->lang_out, louts[b], sizeof(jobs[b]->lang_out));
            jobs[b]->done = 1;
            pthread_cond_signal(&jobs[b]->cv);
            pthread_mutex_unlock(&jobs[b]->mu);
        }
    }
    return NULL;
}

/* ------------------------------------------------------------ connection queue */
static int q_fds[QUEUE_CAP];
static int q_head, q_tail, q_len;
static pthread_mutex_t q_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_cv = PTHREAD_COND_INITIALIZER;

static void q_push(int fd) {
    pthread_mutex_lock(&q_mu);
    if (q_len == QUEUE_CAP) {   /* full: reject right away */
        pthread_mutex_unlock(&q_mu);
        const char *msg = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
        if (write(fd, msg, strlen(msg)) < 0) { /* best-effort: we close anyway */ }
        close(fd);
        return;
    }
    q_fds[q_tail] = fd;
    q_tail = (q_tail + 1) % QUEUE_CAP;
    q_len++;
    pthread_cond_signal(&q_cv);
    pthread_mutex_unlock(&q_mu);
}

static int q_pop(void) {
    pthread_mutex_lock(&q_mu);
    while (q_len == 0) pthread_cond_wait(&q_cv, &q_mu);
    int fd = q_fds[q_head];
    q_head = (q_head + 1) % QUEUE_CAP;
    q_len--;
    pthread_mutex_unlock(&q_mu);
    return fd;
}

/* ------------------------------------------------------------------ I/O utils */
static int write_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static void send_response(int fd, int code, const char *status, const char *ctype,
                          const char *body, size_t body_len) {
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: %s\r\n"
                     "Content-Length: %zu\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "Access-Control-Allow-Headers: *\r\n"
                     "Connection: close\r\n\r\n",
                     code, status, ctype, body_len);
    write_all(fd, hdr, (size_t)n);
    if (body_len) write_all(fd, body, body_len);
}

static void send_json(int fd, int code, cJSON *j) {
    char *s = cJSON_PrintUnformatted(j);
    send_response(fd, code, code == 200 ? "OK" : "Error", "application/json",
                  s, strlen(s));
    free(s);
}

static void send_error(int fd, int code, const char *msg) {
    cJSON *j = cJSON_CreateObject();
    cJSON *e = cJSON_AddObjectToObject(j, "error");
    cJSON_AddStringToObject(e, "message", msg);
    send_json(fd, code, j);
    cJSON_Delete(j);
}

/* ------------------------------------------------------------------ multipart */
typedef struct {
    const uint8_t *file;
    size_t file_len;
    char language[24], response_format[24], target_language[8];
    int lookahead;
} form_data;

static void parse_multipart(const uint8_t *body, size_t len, const char *boundary,
                            form_data *out) {
    char sep[80];
    const size_t sep_len = (size_t)snprintf(sep, sizeof(sep), "--%s", boundary);
    const uint8_t *p = body;
    size_t remain = len;

    for (;;) {
        const uint8_t *part = mynah_asr_memmem(p, remain, (const uint8_t *)sep, sep_len);
        if (!part) break;
        part += sep_len;
        remain = len - (size_t)(part - body);
        if (remain < 4 || part[0] == '-') break;   /* final --boundary-- */

        const uint8_t *hdr_end = mynah_asr_memmem(part, remain, (const uint8_t *)"\r\n\r\n", 4);
        if (!hdr_end) break;
        const uint8_t *data = hdr_end + 4;

        const uint8_t *next = mynah_asr_memmem(data, len - (size_t)(data - body),
                                           (const uint8_t *)sep, sep_len);
        if (!next) break;
        size_t data_len = (size_t)(next - data);
        if (data_len >= 2) data_len -= 2;          /* the \r\n before the boundary */

        char hdrs[512] = {0};
        size_t hl = (size_t)(hdr_end - part);
        if (hl >= sizeof(hdrs)) hl = sizeof(hdrs) - 1;
        memcpy(hdrs, part, hl);

        char name[64] = {0};
        const char *nm = strstr(hdrs, "name=\"");
        if (nm) sscanf(nm + 6, "%63[^\"]", name);

        if (strcmp(name, "file") == 0) {
            out->file = data;
            out->file_len = data_len;
        } else if (strcmp(name, "language") == 0 && data_len < sizeof(out->language)) {
            memcpy(out->language, data, data_len);
            out->language[data_len] = '\0';
        } else if (strcmp(name, "response_format") == 0 && data_len < sizeof(out->response_format)) {
            memcpy(out->response_format, data, data_len);
            out->response_format[data_len] = '\0';
        } else if (strcmp(name, "target_language") == 0 && data_len < sizeof(out->target_language)) {
            memcpy(out->target_language, data, data_len);
            out->target_language[data_len] = '\0';
        } else if (strcmp(name, "lookahead") == 0 && data_len < 8) {
            char tmp[8] = {0};
            memcpy(tmp, data, data_len);
            out->lookahead = atoi(tmp);
        }
        p = next;
        remain = len - (size_t)(p - body);
    }
}

/* -------------------------------------- POST /transcriptions and /translations
 * translate = 1: the /v1/audio/translations endpoint (AED models only) — the
 * output language is target_language (default "en", OpenAI/Whisper style); the
 * per-request translation travels inside lang as "src>tgt" (thread-safe). */
static void handle_transcribe(int fd, const char *headers, const uint8_t *body,
                              size_t body_len, int translate) {
    form_data f = {.lookahead = -1, .language = "auto", .response_format = "json"};

    const char *ct = strstr(headers, "Content-Type:");
    if (!ct) ct = strstr(headers, "content-type:");
    char boundary[72] = {0};
    if (ct) {
        const char *b = strstr(ct, "boundary=");
        if (b) sscanf(b + 9, "%71[^\r\n; ]", boundary);
    }
    if (boundary[0]) {
        parse_multipart(body, body_len, boundary, &f);
    } else {
        f.file = body;           /* raw body: direct audio/wav */
        f.file_len = body_len;
    }
    if (!f.file || f.file_len < 44) { send_error(fd, 400, "missing audio file (multipart 'file' or raw WAV body)"); return; }

    char src_lang[24];
    snprintf(src_lang, sizeof(src_lang), "%s", f.language);
    const char *tgt = NULL;
    if (translate || f.target_language[0]) {
        if (!mynah_asr_can_translate(g_model)) {
            send_error(fd, 400, "this model does not support translation (an AED engine is required, e.g. Canary)");
            return;
        }
        tgt = f.target_language[0] ? f.target_language : "en";
    }

    size_t n_samples;
    int sr;
    float *samples = mynah_asr_wav_parse(f.file, f.file_len, &n_samples, &sr);
    if (!samples) { send_error(fd, 400, "invalid WAV (PCM16 required)"); return; }
    if (sr != 16000) {
        size_t n2;
        float *rs = mynah_asr_resample(samples, n_samples, sr, 16000, &n2);
        free(samples);
        if (!rs) { send_error(fd, 500, "resampling failed"); return; }
        samples = rs;
        n_samples = n2;
    }

    /* language=auto on a model that cannot detect (Canary: the source language is an
     * INPUT, "auto" there means "en") — the --lid-model detector answers first, on a
     * few seconds of audio, and its answer becomes the source language. When it has
     * no answer, or names a language this model does not have, the request still
     * goes through with the model's default rather than failing. */
    if (g_lid && strcmp(src_lang, "auto") == 0 && !mynah_asr_can_detect_lang(g_model)) {
        char tag[16] = "", mapped[16] = "";
        inference_begin();
        const int got = mynah_asr_detect_lang(g_lid, samples, n_samples, tag) == 0;
        inference_end();
        if (got && mynah_asr_map_lang(g_model, tag, mapped) == 0) {
            snprintf(src_lang, sizeof(src_lang), "%s", mapped);
        } else if (got) {   /* same 400 as naming that language in the request */
            char msg[128];
            snprintf(msg, sizeof(msg), "detected language '%s', which this model does "
                                       "not support", tag);
            free(samples);
            send_error(fd, 400, msg);
            return;
        }
    }
    if (tgt) snprintf(f.language, sizeof(f.language), "%s>%s", src_lang, tgt);
    else     snprintf(f.language, sizeof(f.language), "%s", src_lang);

    char lang_out[16] = "";
    char *text;
    mynah_asr_word *words = NULL;
    int n_words = 0;
    const int want_words = strcmp(f.response_format, "verbose_json") == 0;
    if (g_max_batch > 1) {
        trx_job j = {.samples = samples, .n_samples = n_samples, .lookahead = f.lookahead};
        snprintf(j.lang, sizeof(j.lang), "%s", f.language);
        pthread_mutex_init(&j.mu, NULL);
        pthread_cond_init(&j.cv, NULL);
        batch_submit_and_wait(&j);
        text = j.text;
        memcpy(lang_out, j.lang_out, sizeof(lang_out));
        if (want_words) {
            words = j.words;
            n_words = j.n_words;
        } else {
            mynah_asr_words_free(j.words, j.n_words);
        }
        pthread_mutex_destroy(&j.mu);
        pthread_cond_destroy(&j.cv);
    } else {
        inference_begin();
        text = mynah_asr_transcribe_ts(g_model, samples, n_samples, f.language, f.lookahead,
                                   lang_out, want_words ? &words : NULL, &n_words);
        inference_end();
    }
    const double duration = (double)n_samples / 16000.0;
    free(samples);
    if (!text) { send_error(fd, 400, "transcription failed (unsupported language?)"); return; }

    if (strcmp(f.response_format, "text") == 0) {
        send_response(fd, 200, "OK", "text/plain; charset=utf-8", text, strlen(text));
    } else {
        cJSON *j = cJSON_CreateObject();
        cJSON_AddStringToObject(j, "text", text);
        if (strcmp(f.response_format, "verbose_json") == 0) {
            cJSON_AddStringToObject(j, "task", translate || f.target_language[0]
                                              ? "translate" : "transcribe");
            cJSON_AddStringToObject(j, "language", lang_out[0] ? lang_out : src_lang);
            cJSON_AddNumberToObject(j, "duration", duration);
            if (words) {   /* per-word timestamps (from the batch scheduler too) */
                cJSON *jw = cJSON_AddArrayToObject(j, "words");
                for (int i = 0; i < n_words; i++) {
                    cJSON *w = cJSON_CreateObject();
                    cJSON_AddStringToObject(w, "word", words[i].word);
                    cJSON_AddNumberToObject(w, "start", words[i].t0);
                    cJSON_AddNumberToObject(w, "end", words[i].t1);
                    cJSON_AddItemToArray(jw, w);
                }
            }
        }
        send_json(fd, 200, j);
        cJSON_Delete(j);
    }
    mynah_asr_words_free(words, n_words);
    free(text);
}

/* ------------------------------------------------------------------ WebSocket */
static int ws_send_frame(int fd, int opcode, const void *data, size_t len) {
    uint8_t hdr[10];
    size_t hl = 2;
    hdr[0] = (uint8_t)(0x80 | opcode);
    if (len < 126) {
        hdr[1] = (uint8_t)len;
    } else if (len < 65536) {
        hdr[1] = 126;
        hdr[2] = (uint8_t)(len >> 8);
        hdr[3] = (uint8_t)len;
        hl = 4;
    } else {
        hdr[1] = 127;
        for (int i = 0; i < 8; i++) hdr[2 + i] = (uint8_t)((uint64_t)len >> (56 - 8 * i));
        hl = 10;
    }
    if (write_all(fd, hdr, hl) != 0) return -1;
    return write_all(fd, data, len);
}

static int read_exact(int fd, uint8_t *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, buf + got, n - got);
        if (r <= 0) return -1;
        got += (size_t)r;
    }
    return 0;
}

typedef struct { int fd; int failed; } ws_ctx;

static void ws_on_text(const mynah_asr_result *res, void *ud) {
    ws_ctx *c = ud;
    if (c->failed) return;
    cJSON *j = cJSON_CreateObject();
    cJSON_AddStringToObject(j, "text", res->text);
    if (res->lang) cJSON_AddStringToObject(j, "language", res->lang);
    cJSON_AddNumberToObject(j, "audio_seconds", res->t1);
    char *s = cJSON_PrintUnformatted(j);
    if (ws_send_frame(c->fd, 0x1, s, strlen(s)) != 0) c->failed = 1;
    free(s);
    cJSON_Delete(j);
}

static void handle_ws_stream(int fd, const char *headers, const char *query) {
    const char *k = strstr(headers, "Sec-WebSocket-Key:");
    if (!k) { send_error(fd, 400, "invalid WebSocket handshake"); return; }
    char key[64] = {0};
    sscanf(k + 18, " %63[^\r\n]", key);

    char accept_src[128];
    snprintf(accept_src, sizeof(accept_src), "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key);
    uint8_t sha[20];
    mynah_asr_sha1((const uint8_t *)accept_src, strlen(accept_src), sha);
    char accept[40];
    mynah_asr_b64(sha, 20, accept);

    char resp[256];
    int n = snprintf(resp, sizeof(resp),
                     "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                     "Connection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n", accept);
    if (write_all(fd, resp, (size_t)n) != 0) return;

    /* parameters from the query string */
    char lang[24] = "auto";
    int lookahead = -1;
    if (query) {
        const char *ql = strstr(query, "lang=");
        if (ql) sscanf(ql + 5, "%23[^&\n ]", lang);
        const char *qk = strstr(query, "lookahead=");
        if (qk) lookahead = atoi(qk + 10);
    }

    mynah_asr_stream *s = mynah_asr_stream_open(g_model, lang, lookahead);
    if (!s) { close(fd); return; }
    ws_ctx ctx = {.fd = fd};

    float fbuf[65536];
    for (;;) {
        uint8_t h[2];
        if (read_exact(fd, h, 2) != 0) break;
        const int opcode = h[0] & 0x0F;
        const int masked = h[1] & 0x80;
        uint64_t plen = h[1] & 0x7F;
        if (plen == 126) {
            uint8_t e[2];
            if (read_exact(fd, e, 2) != 0) break;
            plen = ((uint64_t)e[0] << 8) | e[1];
        } else if (plen == 127) {
            uint8_t e[8];
            if (read_exact(fd, e, 8) != 0) break;
            plen = 0;
            for (int i = 0; i < 8; i++) plen = (plen << 8) | e[i];
        }
        uint8_t mask[4] = {0};
        if (masked && read_exact(fd, mask, 4) != 0) break;
        if (plen > sizeof(fbuf) * 2) break;   /* unreasonable frame */

        uint8_t *payload = malloc(plen ? plen : 1);
        if (!payload || read_exact(fd, payload, plen) != 0) { free(payload); break; }
        if (masked)
            for (uint64_t i = 0; i < plen; i++) payload[i] ^= mask[i & 3];

        if (opcode == 0x8) { free(payload); break; }             /* close */
        if (opcode == 0x9) {                                     /* ping -> pong */
            ws_send_frame(fd, 0xA, payload, plen);
            free(payload);
            continue;
        }
        if (opcode == 0x2 && plen >= 2) {                        /* PCM s16le */
            const size_t ns = plen / 2;
            const int16_t *pcm = (const int16_t *)payload;
            size_t off = 0;
            /* one frame = one compute burst: counted as a whole, not per 64k
             * chunk, so a busy stream does not churn the BLAS knob */
            inference_begin();
            while (off < ns) {
                const size_t chunk = ns - off < 65536 ? ns - off : 65536;
                for (size_t i = 0; i < chunk; i++) fbuf[i] = (float)pcm[off + i] / 32768.0f;
                mynah_asr_stream_feed(s, fbuf, chunk, ws_on_text, &ctx);
                off += chunk;
            }
            inference_end();
        }
        free(payload);
        if (ctx.failed) break;
    }

    inference_begin();
    mynah_asr_stream_finish(s, ws_on_text, &ctx);
    inference_end();
    cJSON *done = cJSON_CreateObject();
    cJSON_AddBoolToObject(done, "done", 1);
    if (mynah_asr_stream_lang(s)[0]) cJSON_AddStringToObject(done, "language", mynah_asr_stream_lang(s));
    char *ds = cJSON_PrintUnformatted(done);
    ws_send_frame(fd, 0x1, ds, strlen(ds));
    free(ds);
    cJSON_Delete(done);
    ws_send_frame(fd, 0x8, "", 0);
    mynah_asr_stream_close(s);
}

/* ------------------------------------------------------------------- routing */
static void handle_conn(int fd) {
    char hdr[MAX_HDR + 1];
    size_t got = 0;
    const char *hdr_end = NULL;
    while (got < MAX_HDR) {
        ssize_t r = read(fd, hdr + got, MAX_HDR - got);
        if (r <= 0) { close(fd); return; }
        got += (size_t)r;
        hdr[got] = '\0';
        if ((hdr_end = strstr(hdr, "\r\n\r\n")) != NULL) break;
    }
    if (!hdr_end) { close(fd); return; }
    const size_t hdr_len = (size_t)(hdr_end - hdr) + 4;

    char method[8] = {0}, path[512] = {0};
    sscanf(hdr, "%7s %511s", method, path);
    char *query = strchr(path, '?');
    if (query) *query++ = '\0';

    if (strcmp(method, "OPTIONS") == 0) {
        send_response(fd, 204, "No Content", "text/plain", "", 0);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/health") == 0) {
        /* inflight/blas_budget are the adaptive BLAS state: useful to see under
         * load, and what lets a test assert the accounting returns to rest */
        cJSON *j = cJSON_CreateObject();
        cJSON_AddStringToObject(j, "status", "ok");
        cJSON_AddNumberToObject(j, "inflight", inflight_now());
        cJSON_AddNumberToObject(j, "blas_budget", mynah_asr_blas_budget());
        cJSON_AddNumberToObject(j, "threads", mynah_asr_num_threads());
        send_json(fd, 200, j);
        cJSON_Delete(j);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
        cJSON *j = cJSON_CreateObject();
        cJSON *arr = cJSON_AddArrayToObject(j, "data");
        cJSON *m = cJSON_CreateObject();
        cJSON_AddStringToObject(m, "id", g_model_name);
        cJSON_AddStringToObject(m, "object", "model");
        cJSON_AddItemToArray(arr, m);
        send_json(fd, 200, j);
        cJSON_Delete(j);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/audio/stream") == 0) {
        handle_ws_stream(fd, hdr, query);
    } else if (strcmp(method, "POST") == 0 &&
               (strcmp(path, "/v1/audio/transcriptions") == 0 ||
                strcmp(path, "/v1/audio/translations") == 0)) {
        const int translate = strcmp(path, "/v1/audio/translations") == 0;
        size_t content_len = 0;
        const char *cl = strstr(hdr, "Content-Length:");
        if (!cl) cl = strstr(hdr, "content-length:");
        if (cl) content_len = strtoull(cl + 15, NULL, 10);
        if (content_len == 0 || content_len > MAX_BODY) {
            send_error(fd, 400, "missing or oversized Content-Length");
            close(fd);
            return;
        }
        uint8_t *body = malloc(content_len);
        if (!body) { send_error(fd, 500, "out of memory"); close(fd); return; }
        size_t have = got - hdr_len;
        if (have > content_len) have = content_len;
        memcpy(body, hdr + hdr_len, have);
        while (have < content_len) {
            ssize_t r = read(fd, body + have, content_len - have);
            if (r <= 0) break;
            have += (size_t)r;
        }
        if (have == content_len) handle_transcribe(fd, hdr, body, content_len, translate);
        else send_error(fd, 400, "incomplete body");
        free(body);
    } else {
        send_error(fd, 404, "not found");
    }
    close(fd);
}

static void *worker(void *arg) {
    (void)arg;
    for (;;) handle_conn(q_pop());
    return NULL;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *lid_dir = NULL;
    int port = 8090, n_threads = 4;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--lid-model") == 0 && i + 1 < argc) lid_dir = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) g_max_batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            i++;
            g_quant = strcmp(argv[i], "int8") == 0 ? MYNAH_ASR_QUANT_INT8
                    : strcmp(argv[i], "int4") == 0 ? MYNAH_ASR_QUANT_INT4 : MYNAH_ASR_QUANT_F32;
        }
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) mynah_asr_set_backend(argv[++i]);
        else if (strcmp(argv[i], "--caps") == 0 && i + 1 < argc) mynah_asr_set_caps(argv[++i]);
        else {
            fprintf(stderr, "usage: mynah-asr-server -m <model_dir> [-p 8090] [--threads 4] "
                            "[--batch 8] [--backend cpu|metal|cuda] [--caps auto|scalar|avx2|vnni]\n"
                            "       [--lid-model <dir>]  detector for language=auto on models that\n"
                            "                            cannot detect it themselves (Canary)\n");
            return 2;
        }
    }
    if (!model_dir) {
        fprintf(stderr, "usage: mynah-asr-server -m <model_dir> [-p 8090] [--threads 4] [--batch 8]\n");
        return 2;
    }
    if (g_max_batch < 1) g_max_batch = 1;
    if (g_max_batch > 64) g_max_batch = 64;

    signal(SIGPIPE, SIG_IGN);
    g_model = mynah_asr_load_quant(model_dir, g_quant);
    if (!g_model) return 1;
    if (lid_dir) {
        g_lid = mynah_asr_load_quant(lid_dir, g_quant);
        if (!g_lid) return 1;
        if (!mynah_asr_can_detect_lang(g_lid)) {
            fprintf(stderr, "mynah-asr-server: %s cannot detect the language "
                            "(use a model with an 'auto' prompt, e.g. nemotron)\n", lid_dir);
            return 1;
        }
        if (mynah_asr_can_detect_lang(g_model)) {   /* it would never be consulted */
            fprintf(stderr, "mynah-asr-server: --lid-model ignored, the served model "
                            "detects the language itself\n");
            mynah_asr_free(g_lid);
            g_lid = NULL;
        }
    }

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port),
                               .sin_addr.s_addr = htonl(INADDR_ANY)};
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(srv, 64) != 0) {
        fprintf(stderr, "mynah-asr-server: bind/listen failed on port %d\n", port);
        return 1;
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_t t;
        pthread_create(&t, NULL, worker, NULL);
        pthread_detach(t);
    }
    if (g_max_batch > 1) {
        pthread_t t;
        pthread_create(&t, NULL, batch_worker, NULL);
        pthread_detach(t);
    }
    fprintf(stderr, "mynah-asr-server %s: listening on :%d (%d workers, batch %d)\n"
                    "  POST /v1/audio/transcriptions | GET /v1/audio/stream (WS) | /v1/models | /v1/health\n",
            mynah_asr_version(), port, n_threads, g_max_batch);

    for (;;) {
        int fd = accept(srv, NULL, NULL);
        if (fd < 0) continue;
        q_push(fd);
    }
}
