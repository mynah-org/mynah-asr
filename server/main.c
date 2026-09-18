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
 * Concurrency (serving v2, S2-2): ONE scheduler thread owns the model and drives
 * every inference -- stream chunks and offline REST jobs alike. The HTTP threads
 * parse, and for a WebSocket each becomes the INGEST thread of that connection
 * for its life: it decodes PCM into the slot's bounded ring and never touches
 * the model. Output leaves on a per-connection writer (server/stream_out.c), so
 * a client that stops reading loses its own stream and nobody else's.
 * See .work/server-scheduler.md for the design and the evidence behind it.
 */
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "../src/audio.h"
#include "../src/backend.h"
#include "../src/mynah_asr.h"
#include "../src/qmat.h"      /* mynah_asr_set_caps (--caps) */
#include "../src/threads.h"   /* mynah_asr_blas_set_concurrency */
#include "../vendor/cJSON.h"
#include "http_util.h"
#include "prefork.h"
#include "sched.h"
#include "slot.h"
#include "stream_out.h"

#define MAX_HDR (64 * 1024)
#define MAX_BODY (200u * 1024 * 1024)
#define QUEUE_CAP 128

static mynah_asr_model *g_model;
static mynah_asr_model *g_lid;       /* --lid-model: language detector for language=auto */
static const char *g_model_name = "nemotron-3.5-asr-streaming-0.6b";
static int g_max_batch = 8;          /* --batch N; 1 = disabled */
static int g_quant = MYNAH_ASR_QUANT_F32;

/* Per-stream service limits (S2-3, the part that lands with the scheduler). */
static int g_idle_ms = 60000;                      /* --idle-ms */
static int g_ring_seconds = 30;                    /* --ring-seconds */
static size_t g_max_frame_bytes = 1024u * 1024u;   /* --max-frame-bytes */
static int g_max_pending;                          /* --max-pending, 2*threads */

/* Set by SIGINT/SIGTERM; both accept loops poll with a timeout and re-read it,
 * because closing the listening socket from a handler does not wake a blocking
 * accept(). The handler is installed without SA_RESTART on purpose. */
static volatile sig_atomic_t g_shutdown;
static void on_stop(int sig) { (void)sig; g_shutdown = 1; }

/* ------------------------------------------------------------ connection queue */
static int q_fds[QUEUE_CAP];
static int q_head, q_tail, q_len;
static pthread_mutex_t q_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_cv = PTHREAD_COND_INITIALIZER;

static void q_push(int fd) {
    pthread_mutex_lock(&q_mu);
    if (q_len == QUEUE_CAP) {   /* full: refuse right away */
        pthread_mutex_unlock(&q_mu);
        /* Not write+close: with the request body still unread in the socket
         * that sends an RST and the client sees a reset, never the 503.
         * refuse_and_close does shutdown -> bounded drain -> close and emits the
         * same error.code as the parent's ladder. It takes the descriptor; the
         * slot the router charged for it is given back separately. */
        mynah_asr_prefork_refuse_and_close(fd, MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY);
        mynah_asr_prefork_conn_done();
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
        /* The detector is a model too, so it runs where every model runs. */
        mynah_asr_offline_job dj;
        memset(&dj, 0, sizeof(dj));
        dj.kind = MYNAH_ASR_JOB_DETECT_LANG;
        dj.samples = samples;
        dj.n_samples = n_samples;
        const int got = mynah_asr_sched_submit(&dj) == 0 && dj.detected[0] != '\0';
        if (got) snprintf(tag, sizeof(tag), "%s", dj.detected);
        /* map_lang is a table lookup on the model's config, not inference: it
         * stays on this thread rather than costing the scheduler a step. */
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
    {
        /* One path for every offline request: the scheduler runs it, batching
         * what is queued at the step boundary. `--batch 1` still goes through
         * here, as a batch of one -- the code path a transcript takes must not
         * depend on a flag. */
        mynah_asr_offline_job j;
        memset(&j, 0, sizeof(j));
        j.kind = MYNAH_ASR_JOB_TRANSCRIBE;
        j.samples = samples;
        j.n_samples = n_samples;
        j.lookahead = f.lookahead;
        j.want_words = want_words;
        snprintf(j.lang, sizeof(j.lang), "%s", f.language);
        const int rc = mynah_asr_sched_submit(&j);
        if (rc == -2) {
            free(samples);
            char buf[1024];
            const size_t n = mynah_asr_prefork_refusal_response(
                MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY, buf, sizeof(buf));
            if (n) write_all(fd, buf, n);
            return;
        }
        text = j.text;
        memcpy(lang_out, j.lang_out, sizeof(lang_out));
        if (want_words) {
            words = j.words;
            n_words = j.n_words;
        } else {
            mynah_asr_words_free(j.words, j.n_words);
        }
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

/* ------------------------------------------------------------------ WebSocket
 * The INGEST side of a stream (protocol v2, .work/ws-protocol-v2.md). This
 * thread reads frames, converts PCM into the slot's bounded ring and posts
 * control requests. It never calls the model -- the scheduler does -- and it
 * never writes to the socket: once the writer is started the descriptor belongs
 * to it, so everything going out is enqueued instead.
 *
 * Two descriptors for one socket, and the reason is worth stating: the writer
 * closes the fd it owns as soon as it has drained, which can happen while this
 * thread is parked in a read. Reading from a descriptor another thread is
 * closing is how a server ends up serving a stranger's connection after the
 * number is reissued by accept(). So the ingest reads through its own dup and
 * closes it itself; the socket lives until both are gone.
 */
typedef struct {
    int fd;                      /* the ingest's own dup of the client socket */
    mynah_asr_slot *slot;
    mynah_asr_stream_out *out;
} ws_ingest;

static int ws_read_exact(int fd, uint8_t *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        const ssize_t r = recv(fd, buf + got, n - got, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;   /* EOF, error, or the SO_RCVTIMEO backstop */
        got += (size_t)r;
    }
    return 0;
}

static void ws_enqueue(ws_ingest *w, int opcode, const void *payload, size_t len) {
    unsigned char buf[1024];
    const size_t n = mynah_asr_ws_frame(buf, sizeof(buf), opcode, payload, len);
    if (n > 0) (void)mynah_asr_stream_out_enqueue(w->out, buf, n);
}

/* A transport-level complaint: it carries no `seq`, because it is not part of
 * the stream's sequence -- the scheduler owns that counter, and this frame is
 * about the message the client just sent, not about the audio. */
static void ws_transport_error(ws_ingest *w, const char *code, const char *msg) {
    cJSON *j = cJSON_CreateObject();
    cJSON_AddStringToObject(j, "type", "error");
    cJSON_AddStringToObject(j, "code", code);
    cJSON_AddStringToObject(j, "message", msg);
    char *s = cJSON_PrintUnformatted(j);
    if (s != NULL) {
        ws_enqueue(w, 0x1, s, strlen(s));
        free(s);
    }
    cJSON_Delete(j);
}

static void ws_control(ws_ingest *w, const uint8_t *payload, size_t len) {
    char buf[512];
    const size_t n = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
    memcpy(buf, payload, n);
    buf[n] = '\0';
    cJSON *j = cJSON_Parse(buf);
    const cJSON *t = j ? cJSON_GetObjectItem(j, "type") : NULL;
    const char *type = (t != NULL && cJSON_IsString(t)) ? t->valuestring : NULL;

    if (type != NULL && strcmp(type, "finalize") == 0) {
        mynah_asr_slot_request(w->slot, MYNAH_ASR_SLOT_REQ_FINALIZE, NULL,
                               MYNAH_ASR_SLOT_CANCEL_NONE);
    } else if (type != NULL && strcmp(type, "reset") == 0) {
        const cJSON *l = cJSON_GetObjectItem(j, "lang");
        mynah_asr_slot_request(w->slot, MYNAH_ASR_SLOT_REQ_RESET,
                               (l != NULL && cJSON_IsString(l)) ? l->valuestring : NULL,
                               MYNAH_ASR_SLOT_CANCEL_NONE);
    } else {
        /* An unknown message is an error frame, never a disconnect: a client
         * that learns a newer keyword must not lose its session over it. */
        ws_transport_error(w, "unknown_message",
                           "expected {\"type\":\"finalize\"} or {\"type\":\"reset\"}");
    }
    cJSON_Delete(j);
}

/* s16le -> float, byte by byte: the payload is not guaranteed to be aligned for
 * an int16_t, and reading it as one is undefined behaviour that ubsan is right
 * to complain about. Returns 0 when the session ended under us. */
static int ws_push_pcm(ws_ingest *w, const uint8_t *payload, size_t plen, double now) {
    const size_t ns = plen / 2;
    float f[4096];
    size_t off = 0;
    while (off < ns) {
        const size_t chunk = ns - off < 4096 ? ns - off : 4096;
        for (size_t i = 0; i < chunk; i++) {
            const size_t k = (off + i) * 2;
            const int16_t v = (int16_t)((uint16_t)payload[k] |
                                        ((uint16_t)payload[k + 1] << 8));
            f[i] = (float)v / 32768.0f;
        }
        if (mynah_asr_slot_push(w->slot, f, chunk, now) != chunk) return 0;
        off += chunk;
    }
    return 1;
}

/* Returns 1 when the descriptor was handed to the writer (the caller must not
 * close it), 0 when it is still the caller's. */
static int handle_ws_stream(int fd, const char *headers, const char *query) {
    char lang[MYNAH_ASR_SLOT_LANG_CAP] = "auto";
    int lookahead = -1;
    if (query != NULL) {
        const char *ql = strstr(query, "lang=");
        if (ql) sscanf(ql + 5, "%23[^&\n ]", lang);
        const char *qk = strstr(query, "lookahead=");
        if (qk) lookahead = atoi(qk + 10);
    }

    /* Rung 4xx, before the upgrade: an offline-only model will never grow a
     * stream API, so this is not a 503 and carries no Retry-After. */
    if (!mynah_asr_sched_streaming()) {
        cJSON *j = cJSON_CreateObject();
        cJSON *e = cJSON_AddObjectToObject(j, "error");
        cJSON_AddStringToObject(e, "message",
            "this model is offline-only (no cache-aware streaming presets)");
        cJSON_AddStringToObject(e, "type", "invalid_request_error");
        cJSON_AddStringToObject(e, "code", "model_not_streaming");
        send_json(fd, 400, j);
        cJSON_Delete(j);
        return 0;
    }

    const char *k = strstr(headers, "Sec-WebSocket-Key:");
    if (!k) { send_error(fd, 400, "invalid WebSocket handshake"); return 0; }
    char key[64] = {0};
    sscanf(k + 18, " %63[^\r\n]", key);

    /* The slot is reserved BEFORE the 101: a client that is refused must read an
     * HTTP status, not discover the refusal as a transport error after the
     * upgrade. This is the worker's own cap; the router has its own rung. */
    mynah_asr_slot *slot = mynah_asr_sched_claim(lang, lookahead);
    if (slot == NULL) {
        char buf[1024];
        const size_t n = mynah_asr_prefork_refusal_response(
            MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY, buf, sizeof(buf));
        if (n) write_all(fd, buf, n);
        return 0;
    }

    char accept_src[128];
    snprintf(accept_src, sizeof(accept_src),
             "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key);
    uint8_t sha[20];
    mynah_asr_sha1((const uint8_t *)accept_src, strlen(accept_src), sha);
    char accept[40];
    mynah_asr_b64(sha, 20, accept);
    char resp[256];
    const int rn = snprintf(resp, sizeof(resp),
                            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                            "Connection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n",
                            accept);
    if (write_all(fd, resp, (size_t)rn) != 0) {
        mynah_asr_slot_release(slot);
        return 0;
    }

    const int rfd = dup(fd);
    if (rfd < 0) { mynah_asr_slot_release(slot); return 0; }
    struct timeval tv = {.tv_sec = g_idle_ms / 1000,
                         .tv_usec = (g_idle_ms % 1000) * 1000};
    (void)setsockopt(rfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    mynah_asr_stream_out *out = mynah_asr_stream_out_start(fd, 0, 0);
    if (out == NULL) {   /* the fd was never handed over: still the caller's */
        close(rfd);
        mynah_asr_slot_release(slot);
        return 0;
    }
    ws_ingest w = {.fd = rfd, .slot = slot, .out = out};
    mynah_asr_slot_arm(slot, out);
    mynah_asr_thread_set_name("mynah-ingest");

    int cancelled = 0, closed = 0;
    double last_activity = mynah_asr_now();
    uint8_t *payload = NULL;
    while (!closed) {
        if (g_shutdown) break;
        if (mynah_asr_slot_get_state(slot) == MYNAH_ASR_SLOT_DONE) break;
        if (mynah_asr_stream_out_failed(out)) break;

        /* A short tick rather than one long blocking read: the loop has to
         * notice a finished slot, a cancelled stream and SIGTERM, and none of
         * those arrive on this socket. SO_RCVTIMEO above is the backstop for a
         * frame that starts and never finishes. */
        struct pollfd pfd = {.fd = rfd, .events = POLLIN, .revents = 0};
        const int ready = poll(&pfd, 1, 200);
        if (ready < 0) { if (errno == EINTR) continue; break; }
        if (ready == 0) {
            if ((mynah_asr_now() - last_activity) * 1000.0 >= (double)g_idle_ms) {
                mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_CANCEL, NULL,
                                       MYNAH_ASR_SLOT_CANCEL_IDLE);
                cancelled = 1;
                break;
            }
            continue;
        }

        uint8_t h[2];
        if (ws_read_exact(rfd, h, 2) != 0) break;
        const int opcode = h[0] & 0x0F;
        const int masked = h[1] & 0x80;
        uint64_t plen = h[1] & 0x7F;
        if (plen == 126) {
            uint8_t e[2];
            if (ws_read_exact(rfd, e, 2) != 0) break;
            plen = ((uint64_t)e[0] << 8) | e[1];
        } else if (plen == 127) {
            uint8_t e[8];
            if (ws_read_exact(rfd, e, 8) != 0) break;
            plen = 0;
            for (int i = 0; i < 8; i++) plen = (plen << 8) | e[i];
        }
        uint8_t mask[4] = {0};
        if (masked && ws_read_exact(rfd, mask, 4) != 0) break;
        if (plen > (uint64_t)g_max_frame_bytes) {
            ws_transport_error(&w, "frame_too_large",
                               "the frame exceeds --max-frame-bytes");
            mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_CANCEL, NULL,
                                   MYNAH_ASR_SLOT_CANCEL_FRAME);
            cancelled = 1;
            break;
        }

        payload = (uint8_t *)malloc((size_t)plen ? (size_t)plen : 1);
        if (payload == NULL || ws_read_exact(rfd, payload, (size_t)plen) != 0) break;
        if (masked)
            for (uint64_t i = 0; i < plen; i++) payload[i] ^= mask[i & 3];
        last_activity = mynah_asr_now();

        switch (opcode) {
            case 0x8:   /* close: flush the tail, answer `done`, then go */
                mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_FINALIZE |
                                             MYNAH_ASR_SLOT_REQ_CLOSE, NULL,
                                       MYNAH_ASR_SLOT_CANCEL_NONE);
                closed = 1;
                break;
            case 0x9:   /* ping -> pong, through the writer like everything else */
                ws_enqueue(&w, 0xA, payload, (size_t)plen);
                break;
            case 0xA:
                break;  /* pong: liveness, nothing to do */
            case 0x1:
                ws_control(&w, payload, (size_t)plen);
                break;
            case 0x0:   /* a continuation of a binary frame; the only kind any
                         * client here sends, and PCM concatenates */
            case 0x2:
                if (plen >= 2 && !ws_push_pcm(&w, payload, (size_t)plen, last_activity))
                    closed = 1;
                break;
            default:
                ws_transport_error(&w, "unsupported_opcode",
                                   "only text, binary, ping, pong and close are served");
                break;
        }
        free(payload);
        payload = NULL;
    }
    free(payload);

    /* However the loop ended, the session must end explicitly: without a
     * finalize the tail is never flushed and the client never sees `done`. */
    if (!cancelled)
        mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_FINALIZE |
                                     MYNAH_ASR_SLOT_REQ_CLOSE, NULL,
                               MYNAH_ASR_SLOT_CANCEL_NONE);
    mynah_asr_sched_wake();

    int done = mynah_asr_slot_wait_done(slot, 60000);
    if (!done) {
        mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_CANCEL, NULL,
                               MYNAH_ASR_SLOT_CANCEL_PEER);
        done = mynah_asr_slot_wait_done(slot, 5000);
    }
    close(rfd);
    mynah_asr_thread_set_name("mynah-http");
    if (done) {
        mynah_asr_slot_release(slot);
        mynah_asr_stream_out_release(out);
    } else {
        /* The scheduler still owns both. Letting go here would hand it a freed
         * writer; the slot stays charged instead, which /v1/health shows. */
        fprintf(stderr, "mynah-asr-server: slot %d did not finish; "
                        "leaving it to the scheduler\n", slot->id);
    }
    return 1;
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
        cJSON *j = cJSON_CreateObject();
        cJSON_AddStringToObject(j, "status", "ok");
        /* `inflight` is now what it says: slots this worker is holding. With one
         * inference in flight by construction, `blas_budget` no longer moves --
         * it is reported because a budget that is not the thread count means
         * something in the process is still driving the knob. */
        cJSON_AddNumberToObject(j, "inflight", mynah_asr_sched_active());
        cJSON_AddNumberToObject(j, "blas_budget", mynah_asr_blas_budget());
        cJSON_AddNumberToObject(j, "threads", mynah_asr_num_threads());
        /* prefork: which worker answered, so a probe under load can tell "one
         * worker wedged" from "the server is slow" */
        cJSON_AddNumberToObject(j, "worker", mynah_asr_prefork_worker_index());
        mynah_asr_sched_health(j);
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
        /* The writer owns the descriptor from the 101 on; closing it here would
         * be a second close of a number that may already belong to someone. */
        if (handle_ws_stream(fd, hdr, query)) return;
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
    mynah_asr_thread_set_name("mynah-http");
    for (;;) {
        handle_conn(q_pop());
        /* exactly once per connection: the router's only view of our load */
        mynah_asr_prefork_conn_done();
    }
    return NULL;
}

/* Every accepted descriptor, whichever process accepted it: blocking (BSD hands
 * down the listener's O_NONBLOCK, Linux does not, SCM_RIGHTS keeps whatever the
 * parent's listener had) and Nagle off, because a JSON delta is a small write
 * followed by nothing -- exactly the shape Nagle holds back. */
static void prepare_client_fd(int fd) {
    const int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) fcntl(fd, F_SETFL, fl & ~O_NONBLOCK);
    const int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
}

static void usage(void) {
    fprintf(stderr,
        "usage: mynah-asr-server -m <model_dir> [-p 8090] [--threads 4] [--batch 8] [--quant int8|int4]\n"
        "       [--backend cpu|metal|cuda] [--caps auto|scalar|avx2|vnni]\n"
        "       [--lid-model <dir>]  detector for language=auto on models that\n"
        "                            cannot detect it themselves (Canary)\n"
        "       [--prefork W] [--prefork-threads T] [--cap C]\n"
        "                            W pinned worker processes (Linux: core-major cpu slices),\n"
        "                            T threads each (default: cpus/W), C stream slots per\n"
        "                            worker before it refuses 503 (default: --threads)\n"
        "       --prefork-plan       print the machine's topology and the W/T sweep, exit\n"
        "       [--idle-ms 60000]    a stream with no audio for this long is cancelled\n"
        "       [--ring-seconds 30]  PCM buffered per stream before the client is throttled\n"
        "       [--max-frame-bytes 1048576]  a larger WebSocket frame ends the stream\n"
        "       [--max-pending N]    offline requests queued (default 2*--threads), 503 beyond\n"
        "  env: MYNAH_ASR_PREFORK_QUEUE (queued per worker, default 1; 0 = refuse at once),\n"
        "       MYNAH_ASR_PREFORK_QUEUE_MS (queue deadline, default 2000),\n"
        "       MYNAH_ASR_PREFORK_SERVICE_MS (service cap, default 30000)\n");
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *lid_dir = NULL;
    int port = 8090, n_threads = 4;
    int prefork_workers = 0, prefork_threads = 0, cap = 0, plan_only = 0;
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
        else if (strcmp(argv[i], "--prefork") == 0 && i + 1 < argc) prefork_workers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prefork-threads") == 0 && i + 1 < argc) prefork_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cap") == 0 && i + 1 < argc) cap = atoi(argv[++i]);
        else if (strcmp(argv[i], "--idle-ms") == 0 && i + 1 < argc) g_idle_ms = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ring-seconds") == 0 && i + 1 < argc) g_ring_seconds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-frame-bytes") == 0 && i + 1 < argc)
            g_max_frame_bytes = (size_t)strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--max-pending") == 0 && i + 1 < argc) g_max_pending = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prefork-plan") == 0) plan_only = 1;
        else { usage(); return 2; }
    }
    if (g_max_batch < 1) g_max_batch = 1;
    if (g_max_batch > 64) g_max_batch = 64;
    if (n_threads < 1) n_threads = 1;
    if (cap <= 0) cap = n_threads;   /* one slot per HTTP thread: a WS stream holds one */
    if (g_idle_ms < 1000) g_idle_ms = 1000;
    if (g_ring_seconds < 1) g_ring_seconds = 1;
    if (g_max_frame_bytes < 4096) g_max_frame_bytes = 4096;
    if (g_max_pending <= 0) g_max_pending = 2 * n_threads;

    /* A plan is about the machine, not the model: it must work before anyone
     * has downloaded the weights. */
    mynah_asr_prefork_config pf;
    memset(&pf, 0, sizeof(pf));
    pf.listen_fd = -1;
    pf.workers = prefork_workers;
    pf.threads_per = prefork_threads;
    pf.slots_per = cap;
    if (plan_only) {
        mynah_asr_prefork_print_plan(&pf, stdout);
        return 0;
    }
    if (!model_dir) { usage(); return 2; }
    if (prefork_workers < 0) { usage(); return 2; }
    /* BEFORE the model is opened: the pool resolves its width once, on first
     * use, so MYNAH_ASR_THREADS has to be right before anything can dispatch. */
    if (prefork_workers > 0) {
        mynah_asr_prefork_reserve_threads(&pf);
        prefork_workers = pf.workers;
    }

    signal(SIGPIPE, SIG_IGN);
    {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = on_stop;        /* no SA_RESTART: accept/poll must return EINTR */
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);
    }
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

    /* ---- prefork: fork W workers now, with the model mapped and no thread
     * yet. The parent routes until shutdown and never enters the model; a
     * worker continues below with chan_fd as its only source of connections. */
    int chan_fd = -1;
    if (prefork_workers > 0) {
        pf.listen_fd = srv;
        const mynah_asr_prefork_role role = mynah_asr_prefork_run(&pf, &g_shutdown, &chan_fd);
        if (role == MYNAH_ASR_PREFORK_ERROR) { close(srv); mynah_asr_free(g_model); return 1; }
        if (role == MYNAH_ASR_PREFORK_PARENT_DONE) { mynah_asr_free(g_model); return 0; }
        srv = -1;   /* a worker never accepts: the router closed our copy */
    }
    mynah_asr_thread_set_name(chan_fd >= 0 ? "mynah-recv" : "mynah-accept");

    /* One inference in flight, always: the scheduler. Nothing in the server
     * moves this knob again -- the adaptive policy existed because N request
     * threads entered the model at once, and now none do. */
    mynah_asr_blas_set_concurrency(1);

    /* After the fork, so the thread lives in the worker and not in the router. */
    mynah_asr_sched_config sc;
    memset(&sc, 0, sizeof(sc));
    sc.model = g_model;
    sc.lid = g_lid;
    sc.slots = cap;
    sc.ring_seconds = g_ring_seconds;
    sc.max_batch = g_max_batch;
    sc.max_pending = g_max_pending;
    if (mynah_asr_sched_start(&sc) != 0) {
        fprintf(stderr, "mynah-asr-server: the scheduler failed to start\n");
        return 1;
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_t t;
        pthread_create(&t, NULL, worker, NULL);
        pthread_detach(t);
    }
    if (chan_fd >= 0)
        fprintf(stderr, "mynah-asr-server %s: prefork worker %d ready (%d http threads, "
                        "%d stream slots, batch %d, streaming %s)\n",
                mynah_asr_version(), mynah_asr_prefork_worker_index(), n_threads, cap,
                g_max_batch, mynah_asr_sched_streaming() ? "yes" : "no (offline-only model)");
    else
        fprintf(stderr, "mynah-asr-server %s: listening on :%d (%d http threads, "
                        "%d stream slots, batch %d, streaming %s)\n"
                        "  one scheduler thread owns the model; offline jobs share its steps\n"
                        "  POST /v1/audio/transcriptions | GET /v1/audio/stream (WS) | /v1/models | /v1/health\n",
                mynah_asr_version(), port, n_threads, cap, g_max_batch,
                mynah_asr_sched_streaming() ? "yes" : "no (offline-only model)");

    /* Where a connection comes from is the ONLY difference between the single
     * process server and a prefork worker: a worker has no listening socket,
     * the router accepted, chose it and passed the descriptor down a
     * socketpair. Everything after this loop is the same code in both shapes. */
    while (!g_shutdown) {
        if (mynah_asr_prefork_take_dump_request())
            fprintf(stderr, "[stats] worker %d slots %d/%d blas_budget %d\n",
                    mynah_asr_prefork_worker_index(), mynah_asr_sched_active(), cap,
                    mynah_asr_blas_budget());
        int fd;
        if (chan_fd >= 0) {
            fd = mynah_asr_prefork_recv_conn(chan_fd, 200);
            if (fd == -2) continue;            /* timeout or signal: re-read the flag */
            if (fd < 0) {
                fprintf(stderr, "mynah-asr-server: worker %d: the router closed the channel; "
                                "shutting down\n", mynah_asr_prefork_worker_index());
                break;
            }
        } else {
            struct pollfd pfd = {.fd = srv, .events = POLLIN};
            const int ready = poll(&pfd, 1, 200);
            if (ready < 0) { if (errno == EINTR) continue; break; }
            if (ready == 0) continue;
            fd = accept(srv, NULL, NULL);
            if (fd < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK || errno == ECONNABORTED) continue;
                break;
            }
        }
        prepare_client_fd(fd);
        q_push(fd);
    }
    /* Graceful shutdown, in the order the siblings settled on: stop accepting,
     * cancel the live streams with an `error` frame the client can read, let the
     * scheduler finish the step it is in, join, and only then let go of the
     * weights. The writers are detached, so a short grace gives them time to put
     * those last frames on the wire. */
    if (srv >= 0) close(srv);
    if (chan_fd >= 0) close(chan_fd);
    mynah_asr_sched_stop();
    { struct timespec ts = {.tv_sec = 0, .tv_nsec = 200 * 1000000L}; nanosleep(&ts, NULL); }
    mynah_asr_free(g_lid);
    mynah_asr_free(g_model);
    fprintf(stderr, "mynah-asr-server: worker %d stopped\n", mynah_asr_prefork_worker_index());
    return 0;
}
