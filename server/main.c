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
#include <limits.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>   /* RLIMIT_NOFILE: the descriptor ceiling (S7-2) */
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "../src/audio.h"
#include "../src/backend.h"
#include "../src/dispatch.h"  /* --dispatch-map, mynah_asr_isa_guard */
#include "../src/flags.h"     /* the [FLAGS]/[EFFECTIVE-CONFIG] banner lines */
#include "../src/mynah_asr.h"
#include "../src/qmat.h"      /* mynah_asr_set_caps (--caps) */
#include "../src/threads.h"   /* mynah_asr_blas_set_concurrency */
#include "../vendor/cJSON.h"
#include "http_util.h"
#include "metrics.h"
#include "obs.h"
#include "prefork.h"
#include "sched.h"
#include "slot.h"
#include "stream_out.h"

#define MAX_HDR (64 * 1024)
#define MAX_BODY (200u * 1024 * 1024)
#define QUEUE_CAP 128

static mynah_asr_model *g_model;
static mynah_asr_model *g_lid;       /* --lid-model: language detector for language=auto */
/* Read from the model's own mynah.json at start: a server that names a model
 * it is not serving is a server whose /v1/models and whose banner are both
 * wrong. The literal below is only the answer for a directory with no name. */
static char g_model_name[96] = "unknown";
static char g_model_engine[64] = "unknown";
static double g_encoder_frame_ms;      /* streaming.encoder_frame_ms, 0 if none */
static int    g_default_preset_index;
static int g_max_batch = 8;          /* --batch N; 1 = disabled */
/* int8, not f32, and this is a SERVING default rather than a library one.
 *
 * Measured on the M1 dev host, one stream, tests/audio/test_en.wav with an
 * explicit language: f32 RTF 1.528, int8 RTF 0.682. The f32 default did not
 * sustain a single real-time stream there, and every capacity number this repo
 * has -- including the Axion step table (a = 31.7 ms, b = 4.58 ms/stream, about
 * 49 streams in one 32-thread worker) -- was taken at int8. The shipped default
 * and the measurements disagreed, which is the worse of the two problems.
 *
 * The quality cost is documented and small (docs/quantization.md, 34 locales,
 * 102 samples): mean CER 0.133 -> 0.145, identical on about 90% of languages,
 * one genuine regression (ro-RO, already borderline at f32). On a CPU box that
 * trade is not close.
 *
 * The CLI keeps f32: there it is the reference the other modes are checked
 * against, and a tool that quietly quantised would make its own oracle useless.
 * `--quant f32` restores it here, and the [SERVER-CONFIG] banner prints what
 * was actually loaded, so no run has to infer it from this line. */
static int g_quant = MYNAH_ASR_QUANT_INT8;

/* ------------------------------------------------------------ model groups
 *
 * SEVERAL MODELS IN ONE FLEET (S2-6). One process holds ONE model -- that is
 * the decision the whole serving design rests on, because a batch is drawn
 * from one process's slots and therefore reads one set of weights by the shape
 * of the address space rather than by a check somebody has to remember. So
 * "several models" is several WORKER GROUPS, not a registry inside one
 * address space.
 *
 * The table below is the fleet's plan, and it is the SAME table in every
 * process: the parent builds it, opens every model in it, and forks. A child
 * then keeps exactly one model -- its group's -- and closes the rest, so the
 * mapped weights of the model it serves are COW-shared with the parent and the
 * weights of the others cost it nothing. The table's NAMES survive that close
 * in every worker, which is what lets any worker answer /v1/models for the
 * whole fleet and refuse a request that named a model it does not hold.
 *
 * Note what is NOT a group: a LANGUAGE. One Nemotron serves forty of them and
 * the prompt is a per-request post-encoder one-hot, so a batch may freely mix
 * languages and `lang` travels with the request. Routing is by model only. */
#define MAX_GROUPS 16

typedef struct {
    char  name[96];           /* what `?model=` and `--default` match against */
    char  pack[96];           /* "name" from the pack's own mynah.json        */
    const char *dir;          /* the pack; borrowed from argv                 */
    int   workers;            /* :workers= — 0 = unset                        */
    int   cpus;               /* :cpus=    — 0 = unset                        */
    int   cap;                /* :cap=     — 0 = the fleet's --cap            */
    int   quant;              /* :quant=   — the fleet's --quant when unset   */
    int   lookahead;          /* :lookahead= — -1 = the pack's own default    */
    mynah_asr_model *model;   /* opened in the PARENT, before the fork        */
    char  engine[64];
    double encoder_frame_ms;
    int   default_preset_index;
} model_group;

static model_group g_groups[MAX_GROUPS];
static int g_n_groups;
static int g_group;                    /* the group THIS process serves */
static int g_default_group;            /* --default: the group a request naming none gets */

/* Defined with the rest of the group machinery, below the argument parser it
 * belongs to; declared here because both request paths -- the WebSocket query
 * and the REST form -- ask it the same question before they do any work. */
static int check_model_name(const char *want, char *msg, size_t cap);

/* Per-stream service limits (S2-3). Every one of them bounds what ONE connection
 * can cost this worker: time without saying anything, audio, frame size,
 * buffered PCM. A cap that fires is always announced -- an `error` frame with a
 * code, then a close -- because a stream that simply stops is indistinguishable
 * from a bug in the client. */
static int g_idle_ms = 60000;                      /* --idle-ms */
static int g_ping_ms = 20000;                      /* --ping-ms, 0 = no ping */
static int g_ring_seconds = 30;                    /* --ring-seconds */
static double g_max_audio_seconds = 14400.0;       /* --max-audio-seconds */
static size_t g_max_frame_bytes = 1024u * 1024u;   /* --max-frame-bytes */
static int g_max_pending;                          /* --max-pending, 2*threads */

/* Set by SIGINT/SIGTERM; both accept loops poll with a timeout and re-read it,
 * because closing the listening socket from a handler does not wake a blocking
 * accept(). The handler is installed without SA_RESTART on purpose. */
static volatile sig_atomic_t g_shutdown;
static void on_stop(int sig) { (void)sig; g_shutdown = 1; }

/* SIGUSR1 = "print what you know, once" (S3-4). The handler only raises a
 * flag; the dump itself is written by the accept loop, where fprintf is
 * allowed. Installed BEFORE the fork on purpose: a worker inherits it, and
 * without a handler the default action for SIGUSR1 would kill the worker the
 * first time anyone asked the fleet for statistics. */
static void on_usr1(int sig) { (void)sig; mynah_asr_prefork_request_dump(); }

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
        mynah_asr_obs_refused(mynah_asr_prefork_refusal_code(
                                  MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY));
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

/* ---------------------------------------------------------------- refusing
 * A refusal is a response the client must be able to READ. Writing it and
 * closing is not enough: with the request body (or the next pipelined request)
 * still unread in the receive queue, close() makes the kernel send an RST, and
 * the RST discards the response along with it -- which is how a documented 400
 * reaches a client as ECONNRESET. So every refusal leaves through the same
 * lingering close the router uses: write, shutdown(SHUT_WR), bounded
 * non-blocking drain, close. Takes ownership of `fd`.
 *
 * `msg` is built here and never echoes a client string verbatim: what came from
 * the query goes through refuse_token() first. */
static void refuse_json(int fd, int code, const char *status, const char *type,
                        const char *errcode, const char *msg) {
    /* Counted HERE, at the one funnel every refusal this worker issues passes
     * through, so a counted refusal and a delivered refusal are the same event.
     * `errcode` is a compile-time literal at every call site. */
    mynah_asr_obs_refused(errcode);
    char body[288], resp[MYNAH_ASR_PREFORK_LINGER_MAX];
    const int blen = snprintf(body, sizeof(body),
        "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"code\":\"%s\"}}",
        msg, type, errcode);
    if (blen <= 0 || (size_t)blen >= sizeof(body)) {
        mynah_asr_prefork_linger_close(fd, NULL, 0);
        return;
    }
    const int n = snprintf(resp, sizeof(resp),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n%s",
        code, status, blen, body);
    if (n <= 0 || (size_t)n >= sizeof(resp)) {
        mynah_asr_prefork_linger_close(fd, NULL, 0);
        return;
    }
    mynah_asr_prefork_linger_close(fd, resp, (size_t)n);
}

/* A client-supplied token on its way into a JSON error message. Only what a
 * query key or a language tag can legitimately contain survives, and the result
 * is short: a message that quoted the raw bytes would let the client choose
 * where the JSON string ends. */
static void refuse_token(const char *src, char *dst, size_t cap) {
    size_t k = 0;
    for (size_t i = 0; src != NULL && src[i] != '\0' && k + 1 < cap; i++) {
        const char c = src[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '>')
            dst[k++] = c;
    }
    dst[k] = '\0';
}

/* ------------------------------------------------------------------ multipart */
typedef struct {
    const uint8_t *file;
    size_t file_len;
    char language[24], response_format[24], target_language[8];
    /* The group the client asked for. Read here as well as by the router,
     * because the router classifies on a bounded prefix and this worker reads
     * the whole body: the two must agree, and where they cannot, this one is
     * the one that can refuse rather than answer from the wrong weights. */
    char model[96];
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
        } else if (strcmp(name, "model") == 0 && data_len < sizeof(out->model)) {
            memcpy(out->model, data, data_len);
            out->model[data_len] = '\0';
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
 * per-request translation travels inside lang as "src>tgt" (thread-safe).
 *
 * Returns 1 when it took ownership of `fd` (a refusal closed it), 0 when the
 * descriptor is still the caller's. */
static int handle_transcribe(int fd, const char *headers, const uint8_t *body,
                             size_t body_len, const char *query, int translate) {
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
    if (!f.file || f.file_len < 44) { send_error(fd, 400, "missing audio file (multipart 'file' or raw WAV body)"); return 0; }

    /* WHICH MODEL, checked before any work: the request may name it in the form
     * or in the query, and either way this worker holds exactly one. The router
     * normally decided from the same value; this is what makes a request that
     * slipped past the classifier a refusal instead of a wrong transcript. */
    {
        char want[96] = "";
        if (f.model[0] != '\0') snprintf(want, sizeof(want), "%s", f.model);
        else if (query != NULL) {
            const char *q = query;
            while (*q != '\0') {
                const char *amp = strchr(q, '&');
                const size_t seg = amp != NULL ? (size_t)(amp - q) : strlen(q);
                if (seg > 6 && strncmp(q, "model=", 6) == 0) {
                    const size_t vlen = seg - 6;
                    snprintf(want, sizeof(want), "%.*s",
                             (int)(vlen < sizeof(want) ? vlen : sizeof(want) - 1), q + 6);
                    break;
                }
                if (amp == NULL) break;
                q = amp + 1;
            }
        }
        /* 192, not more: refuse_json assembles the body into 288 bytes and a
         * message that does not fit is dropped whole rather than truncated, so
         * the cap here is what keeps the refusal deliverable. */
        char msg[192];
        const int st = check_model_name(want, msg, sizeof(msg));
        if (st != 0) {
            refuse_json(fd, st, "Not Found", "invalid_request_error",
                        "model_not_found", msg);
            return 1;
        }
    }

    if (f.lookahead < 0) f.lookahead = g_groups[g_group].lookahead;   /* :lookahead= */

    char src_lang[24];
    snprintf(src_lang, sizeof(src_lang), "%s", f.language);
    const char *tgt = NULL;
    if (translate || f.target_language[0]) {
        if (!mynah_asr_can_translate(g_model)) {
            send_error(fd, 400, "this model does not support translation (an AED engine is required, e.g. Canary)");
            return 0;
        }
        tgt = f.target_language[0] ? f.target_language : "en";
    }

    size_t n_samples;
    int sr;
    float *samples = mynah_asr_wav_parse(f.file, f.file_len, &n_samples, &sr);
    if (!samples) { send_error(fd, 400, "invalid WAV (PCM16 required)"); return 0; }
    const int want_sr = mynah_asr_sched_sample_rate();
    if (sr != want_sr) {
        size_t n2;
        float *rs = mynah_asr_resample(samples, n_samples, sr, want_sr, &n2);
        free(samples);
        if (!rs) { send_error(fd, 500, "resampling failed"); return 0; }
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
            return 0;
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
            /* --max-pending, and the same discipline as every other refusal:
             * writing a 503 and closing on a socket that may still hold a
             * pipelined request sends an RST, and the client reads the reset
             * instead of the status it was told to back off on. */
            free(samples);
            mynah_asr_obs_refused(mynah_asr_prefork_refusal_code(
                                      MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY));
            mynah_asr_prefork_refuse_and_close(
                fd, MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY);
            return 1;
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
    const double duration = (double)n_samples / (double)want_sr;
    free(samples);
    if (!text) { send_error(fd, 400, "transcription failed (unsupported language?)"); return 0; }

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
    return 0;
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
/* The accepted query keys in ONE place, so the 400 that refuses an unknown one
 * names exactly what the server will read. */
#define WS_QUERY_ACCEPTED "model, lang, lookahead, format, rate"

typedef struct {
    char lang[MYNAH_ASR_SLOT_LANG_CAP];
    int lookahead;               /* -1 = the model's default preset */
    int f32;                     /* format=f32le; the default is s16le */
} ws_params;

/* Parses and VALIDATES the query of GET /v1/audio/stream. Returns 0, or the HTTP
 * status to refuse with, filling `*code` and `msg` (both go into the body).
 * Everything here happens BEFORE the 101: a client that misspells a parameter
 * reads a status naming the accepted set, instead of getting a stream that
 * silently runs with a default it never asked for.
 *
 * `rate=`: only the model's own sample rate is accepted (16000 for every NeMo
 * speech pack shipped so far), and the refusal says so rather than
 * resampling. The library's resampler (src/audio.h, mynah_asr_resample) is a
 * WHOLE-BUFFER stateless windowed sinc: it keeps no filter history across calls,
 * so running it per WebSocket frame would inject a discontinuity at every frame
 * boundary and change the transcript in a way no identity gate would catch. A
 * streaming resampler is separate work with its own parity gate; until it
 * exists the honest answer is a 400 that names what is served.
 *
 * The lookups are reads of the model's config -- the prompt dictionary and the
 * preset table -- not inference, so they stay on this thread rather than
 * costing the scheduler a step (handle_transcribe resolves map_lang the same
 * way, for the same reason). */
static int ws_parse_query(const char *query, ws_params *p, const char **code,
                          char *msg, size_t msgcap) {
    snprintf(p->lang, sizeof(p->lang), "auto");
    /* The group's `:lookahead=`, when it named one: a per-group default preset
     * is what lets two models in one fleet be served at two different latencies
     * without every client having to know which. -1 keeps the pack's own
     * default, which is what a group that named none gets. */
    p->lookahead = g_groups[g_group].lookahead;
    p->f32 = 0;
    if (query == NULL || query[0] == '\0') return 0;

    const char *q = query;
    while (*q != '\0') {
        const char *amp = strchr(q, '&');
        const size_t seg = amp != NULL ? (size_t)(amp - q) : strlen(q);
        const char *eq = (const char *)memchr(q, '=', seg);
        const size_t klen = eq != NULL ? (size_t)(eq - q) : seg;
        const size_t vlen = eq != NULL ? seg - klen - 1 : 0;
        char key[40], val[64], tok[48];
        snprintf(key, sizeof(key), "%.*s",
                 (int)(klen < sizeof(key) ? klen : sizeof(key) - 1), q);
        snprintf(val, sizeof(val), "%.*s",
                 (int)(vlen < sizeof(val) ? vlen : sizeof(val) - 1),
                 eq != NULL ? eq + 1 : "");

        if (klen == 0) {
            /* an empty segment ("a=1&&b=2"): nothing to read, nothing to refuse */
        } else if (strcmp(key, "model") == 0) {
            /* The router already chose this worker from this very parameter, so
             * in a well-classified fleet this check never fires. It is here for
             * the two shapes where it can: a single-process server, and a
             * connection the router had to default because the name was not in
             * the prefix it could see. Answering from the wrong weights is the
             * failure that must never happen, so the worker checks too. */
            const int st = check_model_name(val, msg, msgcap);
            if (st != 0) { *code = "model_not_found"; (void)tok; return st; }
        } else if (strcmp(key, "lang") == 0) {
            if (val[0] == '\0') {
                snprintf(p->lang, sizeof(p->lang), "auto");
            } else if (mynah_asr_lang_id(g_model, val) < 0) {
                refuse_token(val, tok, sizeof(tok));
                snprintf(msg, msgcap, "this model does not serve the language '%s'", tok);
                *code = "language_not_served";
                return 400;
            } else {
                snprintf(p->lang, sizeof(p->lang), "%s", val);
            }
        } else if (strcmp(key, "lookahead") == 0) {
            if (val[0] != '\0') {
                char *end = NULL;
                const long want = strtol(val, &end, 10);
                int la[8];
                const int n = mynah_asr_lookaheads(g_model, la);
                int ok = end != NULL && *end == '\0';
                if (ok) {
                    ok = 0;
                    for (int i = 0; i < n; i++)
                        if ((long)la[i] == want) ok = 1;
                }
                if (!ok) {
                    char set[64];
                    size_t w = 0;
                    set[0] = '\0';
                    for (int i = 0; i < n && w + 8 < sizeof(set); i++)
                        w += (size_t)snprintf(set + w, sizeof(set) - w, "%s%d",
                                              i > 0 ? ", " : "", la[i]);
                    refuse_token(val, tok, sizeof(tok));
                    snprintf(msg, msgcap,
                             "lookahead '%s' is not a preset of this model; accepted: %s",
                             tok, set[0] != '\0' ? set : "none");
                    *code = "lookahead_not_available";
                    return 400;
                }
                p->lookahead = (int)want;
            }
        } else if (strcmp(key, "format") == 0) {
            if (strcmp(val, "f32le") == 0) {
                p->f32 = 1;
            } else if (val[0] != '\0' && strcmp(val, "s16le") != 0) {
                refuse_token(val, tok, sizeof(tok));
                snprintf(msg, msgcap, "format '%s' is not served; accepted: s16le, f32le",
                         tok);
                *code = "unsupported_format";
                return 400;
            }
        } else if (strcmp(key, "rate") == 0) {
            char want[16];
            snprintf(want, sizeof(want), "%d", mynah_asr_sched_sample_rate());
            if (val[0] != '\0' && strcmp(val, want) != 0) {
                refuse_token(val, tok, sizeof(tok));
                snprintf(msg, msgcap,
                         "rate '%s' is not served; accepted: %s (resample client-side)",
                         tok, want);
                *code = "unsupported_rate";
                return 400;
            }
        } else {
            refuse_token(key, tok, sizeof(tok));
            snprintf(msg, msgcap, "unknown query parameter '%s'; accepted: %s",
                     tok, WS_QUERY_ACCEPTED);
            *code = "unknown_query_parameter";
            return 400;
        }

        if (amp == NULL) break;
        q = amp + 1;
    }
    return 0;
}

typedef struct {
    int fd;                      /* the ingest's own dup of the client socket */
    int f32;                     /* the session's binary format (format=f32le) */
    mynah_asr_slot *slot;
    mynah_asr_stream_out *out;
    /* The client asked for the tail ({"type":"finalize"}) and has sent no audio
     * since. Only then is an EOF a legal half-close waiting for `done`; any
     * other EOF is a client that went away mid-utterance. */
    int finalize_pending;
} ws_ingest;

/* How a read of the client socket ended. The three failures used to be one
 * `-1`, and the ingest answered all three with a finalize: a client that
 * vanished mid-utterance then had the rest of its ring -- up to --ring-seconds
 * of audio -- run through the model for nobody, and was counted as nothing.
 * Measured before the fix (tests/fault_probe.py, rst-mid-silent): 27.4 s of
 * audio fed after a RST. */
enum { WS_READ_OK = 0, WS_READ_EOF = -1, WS_READ_TIMEOUT = -2, WS_READ_ERROR = -3 };

static int ws_read_exact(int fd, uint8_t *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        const ssize_t r = recv(fd, buf + got, n - got, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r == 0) return WS_READ_EOF;
        if (r < 0) {
            /* SO_RCVTIMEO, the backstop for a frame that starts and never
             * finishes: an idle client, not a broken one. */
            if (errno == EAGAIN || errno == EWOULDBLOCK) return WS_READ_TIMEOUT;
            return WS_READ_ERROR;
        }
        got += (size_t)r;
    }
    return WS_READ_OK;
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
        w->finalize_pending = 1;
        mynah_asr_slot_request(w->slot, MYNAH_ASR_SLOT_REQ_FINALIZE, NULL,
                               MYNAH_ASR_SLOT_CANCEL_NONE);
    } else if (type != NULL && strcmp(type, "reset") == 0) {
        const cJSON *l = cJSON_GetObjectItem(j, "lang");
        const char *lang = (l != NULL && cJSON_IsString(l)) ? l->valuestring : NULL;
        /* A language this model does not hold is refused HERE, from the model's
         * own prompt dictionary (a table lookup, not inference). Letting it
         * reach the scheduler would fail mynah_asr_stream_reset and take the
         * session down with it, and a typo must not cost a session. */
        if (lang != NULL && lang[0] != '\0' && mynah_asr_lang_id(g_model, lang) < 0) {
            char tok[48], m[160];
            refuse_token(lang, tok, sizeof(tok));
            snprintf(m, sizeof(m), "this model does not serve the language '%s'", tok);
            ws_transport_error(w, "language_not_served", m);
        } else {
            mynah_asr_slot_request(w->slot, MYNAH_ASR_SLOT_REQ_RESET, lang,
                                   MYNAH_ASR_SLOT_CANCEL_NONE);
        }
    } else {
        /* An unknown control message is an error frame, never a disconnect: a
         * client that learns a newer keyword must not lose its session over it. */
        ws_transport_error(w, "unknown_control",
                           "expected {\"type\":\"finalize\"} or {\"type\":\"reset\"}");
    }
    cJSON_Delete(j);
}

/* The wire format -> float, byte by byte in both cases: the payload sits at
 * whatever offset the WebSocket header left it at, and reading it through an
 * int16_t or a float pointer is an unaligned access -- undefined behaviour that
 * ubsan is right to complain about. Returns 0 when the session ended under us. */
static int ws_push_pcm(ws_ingest *w, const uint8_t *payload, size_t plen, double now) {
    const size_t width = w->f32 ? 4u : 2u;
    const size_t ns = plen / width;
    float f[4096];
    size_t off = 0;
    while (off < ns) {
        const size_t chunk = ns - off < 4096 ? ns - off : 4096;
        for (size_t i = 0; i < chunk; i++) {
            const size_t k = (off + i) * width;
            if (width == 2) {
                const int16_t v = (int16_t)((uint16_t)payload[k] |
                                            ((uint16_t)payload[k + 1] << 8));
                f[i] = (float)v / 32768.0f;
            } else {
                const uint32_t u = (uint32_t)payload[k] |
                                   ((uint32_t)payload[k + 1] << 8) |
                                   ((uint32_t)payload[k + 2] << 16) |
                                   ((uint32_t)payload[k + 3] << 24);
                memcpy(&f[i], &u, sizeof(f[i]));
            }
        }
        if (mynah_asr_slot_push(w->slot, f, chunk, now) != chunk) return 0;
        off += chunk;
    }
    return 1;
}

/* What a failed read means for the session. A timeout is an idle client. An
 * EOF right after {"type":"finalize"} is the legal half-close of a client that
 * is still reading (NONE: finalize as usual); any other EOF, and every socket
 * error, is a client that is gone. */
static mynah_asr_slot_cancel ws_read_loss(int rr, int finalize_pending) {
    if (rr == WS_READ_TIMEOUT) return MYNAH_ASR_SLOT_CANCEL_IDLE;
    if (rr == WS_READ_EOF && finalize_pending) return MYNAH_ASR_SLOT_CANCEL_NONE;
    return MYNAH_ASR_SLOT_CANCEL_PEER;
}

/* Returns 1 when the descriptor is no longer the caller's -- handed to the
 * writer, or consumed by a lingering refusal -- and 0 when it is still theirs. */
static int handle_ws_stream(int fd, const char *headers, const char *query) {
    /* Rung 4xx, before the upgrade: an offline-only model will never grow a
     * stream API, so this is not a 503 and carries no Retry-After. Asked first
     * because on such a model there is no preset table to validate against. */
    if (!mynah_asr_sched_streaming()) {
        const char *why = mynah_asr_sched_stream_why();
        refuse_json(fd, 400, "Bad Request", "invalid_request_error",
                    "model_not_streaming",
                    why ? why
                        : "this model is offline-only (no cache-aware streaming presets)");
        return 1;
    }

    ws_params params;
    const char *qcode = "invalid_request";
    char qmsg[192];
    qmsg[0] = '\0';
    const int qstatus = ws_parse_query(query, &params, &qcode, qmsg, sizeof(qmsg));
    if (qstatus != 0) {
        refuse_json(fd, qstatus, qstatus == 404 ? "Not Found" : "Bad Request",
                    "invalid_request_error", qcode, qmsg);
        return 1;
    }

    const char *k = strstr(headers, "Sec-WebSocket-Key:");
    if (!k) {
        refuse_json(fd, 400, "Bad Request", "invalid_request_error",
                    "invalid_handshake",
                    "a WebSocket upgrade needs a Sec-WebSocket-Key header");
        return 1;
    }
    char key[64] = {0};
    sscanf(k + 18, " %63[^\r\n]", key);

    /* The slot is reserved BEFORE the 101: a client that is refused must read an
     * HTTP status, not discover the refusal as a transport error after the
     * upgrade. This is the worker's own cap; the router has its own rung. The
     * refusal leaves through the lingering close for the same reason the body
     * and the Retry-After exist at all -- they are worthless if the client
     * reads ECONNRESET instead. */
    mynah_asr_slot *slot = mynah_asr_sched_claim(params.lang, params.lookahead);
    if (slot == NULL) {
        mynah_asr_obs_refused(mynah_asr_prefork_refusal_code(
                                  MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY));
        mynah_asr_prefork_refuse_and_close(fd, MYNAH_ASR_PREFORK_REFUSE_AT_CAPACITY);
        return 1;
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
    /* Every exit before the writer is armed goes through sched_release, which
     * counts the session as `aborted`: claimed, never started. */
    if (write_all(fd, resp, (size_t)rn) != 0) {
        mynah_asr_sched_release(slot);
        return 0;
    }

    const int rfd = dup(fd);
    if (rfd < 0) { mynah_asr_sched_release(slot); return 0; }
    struct timeval tv = {.tv_sec = g_idle_ms / 1000,
                         .tv_usec = (g_idle_ms % 1000) * 1000};
    (void)setsockopt(rfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    mynah_asr_stream_out *out = mynah_asr_stream_out_start(fd, 0, 0);
    if (out == NULL) {   /* the fd was never handed over: still the caller's */
        close(rfd);
        mynah_asr_sched_release(slot);
        return 0;
    }
    ws_ingest w = {.fd = rfd, .f32 = params.f32, .slot = slot, .out = out};
    mynah_asr_slot_arm(slot, out);
    mynah_asr_thread_set_name("mynah-ingest");

    int cancelled = 0, closed = 0, shutting = 0;
    /* Why the read loop ended, when it was not a close frame: the session is
     * then cancelled under this reason instead of finalized. */
    mynah_asr_slot_cancel lost = MYNAH_ASR_SLOT_CANCEL_NONE;
    size_t audio_samples = 0;
    const double t_start = mynah_asr_now();
    double last_activity = t_start, last_ping = t_start;
    uint8_t *payload = NULL;
    while (!closed) {
        if (g_shutdown) { shutting = 1; break; }
        if (mynah_asr_slot_get_state(slot) == MYNAH_ASR_SLOT_DONE) break;
        if (mynah_asr_stream_out_failed(out)) { lost = MYNAH_ASR_SLOT_CANCEL_PEER; break; }

        /* The server's own liveness probe, on this thread's tick rather than on
         * a timer thread. It goes out through the writer like every other frame,
         * so it can never block the ingest, and no pong is required: a pong is a
         * courtesy, `--idle-ms` is the rule. */
        const double tick = mynah_asr_now();
        if (g_ping_ms > 0 && (tick - last_ping) * 1000.0 >= (double)g_ping_ms) {
            ws_enqueue(&w, 0x9, "", 0);
            last_ping = tick;
        }

        /* A short tick rather than one long blocking read: the loop has to
         * notice a finished slot, a cancelled stream and SIGTERM, and none of
         * those arrive on this socket. SO_RCVTIMEO above is the backstop for a
         * frame that starts and never finishes. */
        struct pollfd pfd = {.fd = rfd, .events = POLLIN, .revents = 0};
        const int ready = poll(&pfd, 1, 200);
        if (ready < 0) {
            if (errno == EINTR) continue;
            lost = MYNAH_ASR_SLOT_CANCEL_PEER;
            break;
        }
        if (ready == 0) {
            /* Idle is measured against ANY frame, not just audio: a client that
             * keeps the socket open and says nothing at all is the one holding
             * a slot for free, whatever it intended to send. */
            if ((mynah_asr_now() - last_activity) * 1000.0 >= (double)g_idle_ms) {
                mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_CANCEL, NULL,
                                       MYNAH_ASR_SLOT_CANCEL_IDLE);
                cancelled = 1;
                break;
            }
            continue;
        }

        uint8_t h[2];
        int rr = ws_read_exact(rfd, h, 2);
        if (rr != WS_READ_OK) { lost = ws_read_loss(rr, w.finalize_pending); break; }
        const int opcode = h[0] & 0x0F;
        const int masked = h[1] & 0x80;
        uint64_t plen = h[1] & 0x7F;
        if (plen == 126) {
            uint8_t e[2];
            if ((rr = ws_read_exact(rfd, e, 2)) != WS_READ_OK) {
                lost = ws_read_loss(rr, 0);
                break;
            }
            plen = ((uint64_t)e[0] << 8) | e[1];
        } else if (plen == 127) {
            uint8_t e[8];
            if ((rr = ws_read_exact(rfd, e, 8)) != WS_READ_OK) {
                lost = ws_read_loss(rr, 0);
                break;
            }
            plen = 0;
            for (int i = 0; i < 8; i++) plen = (plen << 8) | e[i];
        }
        uint8_t mask[4] = {0};
        if (masked && (rr = ws_read_exact(rfd, mask, 4)) != WS_READ_OK) {
            lost = ws_read_loss(rr, 0);
            break;
        }
        /* RFC 6455 5.2 and 5.5: reserved bits with no negotiated extension, a
         * control frame over 125 bytes or fragmented -- the connection must be
         * failed. Before this they were read as audio or ignored, and a client
         * speaking another dialect got an idle timeout instead of a reason. */
        if ((h[0] & 0x70) != 0 ||
            ((opcode & 0x8) != 0 && (plen > 125 || (h[0] & 0x80) == 0))) {
            ws_transport_error(&w, "protocol_error",
                               (h[0] & 0x70) != 0
                                   ? "reserved bits set: no extension was negotiated"
                                   : "a control frame must be final and at most 125 bytes");
            mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_CANCEL, NULL,
                                   MYNAH_ASR_SLOT_CANCEL_PROTOCOL);
            cancelled = 1;
            break;
        }
        if (plen > (uint64_t)g_max_frame_bytes) {
            ws_transport_error(&w, "frame_too_large",
                               "the frame exceeds --max-frame-bytes");
            mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_CANCEL, NULL,
                                   MYNAH_ASR_SLOT_CANCEL_FRAME);
            cancelled = 1;
            break;
        }

        payload = (uint8_t *)malloc((size_t)plen ? (size_t)plen : 1);
        if (payload == NULL) { lost = MYNAH_ASR_SLOT_CANCEL_PEER; break; }
        if ((rr = ws_read_exact(rfd, payload, (size_t)plen)) != WS_READ_OK) {
            lost = ws_read_loss(rr, 0);
            break;
        }
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
            case 0x2: {
                const size_t width = w.f32 ? 4u : 2u;
                if (plen < width) break;
                w.finalize_pending = 0;
                if (!ws_push_pcm(&w, payload, (size_t)plen, last_activity)) {
                    closed = 1;
                    break;
                }
                audio_samples += (size_t)plen / width;
                if (g_max_audio_seconds > 0.0 &&
                    (double)audio_samples / (double)mynah_asr_sched_sample_rate() >
                        g_max_audio_seconds) {
                    /* Announced, then finalised rather than dropped: the audio
                     * already accepted is still owed a transcript, so the cap
                     * flushes the tail, emits `done` and closes. */
                    /* Ended by the INGEST, so the scheduler never sees a
                     * cancel for it: counted here, in the same buckets, under
                     * the same code the client was just given. */
                    mynah_asr_slot_set_outcome(slot, "audio_limit");
                    ws_transport_error(&w, "audio_limit",
                                       "the stream reached --max-audio-seconds");
                    mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_FINALIZE |
                                                 MYNAH_ASR_SLOT_REQ_CLOSE, NULL,
                                           MYNAH_ASR_SLOT_CANCEL_NONE);
                    closed = 1;
                }
                break;
            }
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
     * finalize the tail is never flushed and the client never sees `done`.
     * SIGTERM is the exception -- the scheduler's drain owes that client an
     * `error shutting_down`, and a finalize racing it would answer `done`
     * instead, which tells the client the opposite of what happened. */
    /* ...and a client that went away without a close frame is owed nothing:
     * no tail, no `done`. It is cancelled here, so the model stops at the next
     * step boundary instead of draining the ring for nobody. */
    if (!cancelled && !shutting && !closed && lost != MYNAH_ASR_SLOT_CANCEL_NONE) {
        mynah_asr_slot_request(slot, MYNAH_ASR_SLOT_REQ_CANCEL, NULL, lost);
        cancelled = 1;
    }
    if (!cancelled && !shutting)
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
    if (done || !mynah_asr_sched_abandon(slot)) {
        mynah_asr_sched_release(slot);
        mynah_asr_stream_out_release(out);
    } else {
        /* The scheduler still owns both. Letting go here would hand it a freed
         * writer, so the release becomes the scheduler's: it frees slot and
         * writer when it finally ends the session (abandoned.recovered in
         * /v1/health). Until then the slot is charged, and counted. */
        fprintf(stderr, "mynah-asr-server: slot %d did not finish; "
                        "left to the scheduler to release\n", slot->id);
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
        hdr_end = strstr(hdr, "\r\n\r\n");
        if (hdr_end != NULL) break;
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
        mynah_asr_obs_health(j);
        send_json(fd, 200, j);
        cJSON_Delete(j);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
        /* THE ACTUAL TABLE, not this worker's own model. Every worker holds the
         * fleet's plan -- the names survive the close of the other groups'
         * weights -- so whichever worker answers, the list is the same list the
         * router routes by and the same one a `model_not_found` quotes. This is
         * what replaced a hard-coded name. */
        cJSON *j = cJSON_CreateObject();
        cJSON *arr = cJSON_AddArrayToObject(j, "data");
        for (int i = 0; i < g_n_groups; i++) {
            cJSON *m = cJSON_CreateObject();
            cJSON_AddStringToObject(m, "id", g_groups[i].name);
            cJSON_AddStringToObject(m, "object", "model");
            cJSON_AddStringToObject(m, "owned_by", "mynah");
            cJSON_AddStringToObject(m, "engine", g_groups[i].engine);
            /* True of the GROUP, answered from its pack rather than from this
             * process: a client choosing where to open a WebSocket needs to
             * know which groups can stream before it tries one. */
            cJSON_AddBoolToObject(m, "streaming",
                                  g_groups[i].encoder_frame_ms > 0.0 ? 1 : 0);
            if (i == g_default_group) cJSON_AddBoolToObject(m, "default", 1);
            cJSON_AddItemToArray(arr, m);
        }
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
            /* The body is still in the socket: a plain close here would RST it
             * away together with the 400 the client is waiting to read. */
            refuse_json(fd, 400, "Bad Request", "invalid_request_error",
                        "invalid_content_length",
                        "missing or oversized Content-Length");
            return;
        }
        uint8_t *body = malloc(content_len);
        if (!body) {
            refuse_json(fd, 500, "Internal Server Error", "server_error",
                        "out_of_memory", "out of memory");
            return;
        }
        size_t have = got - hdr_len;
        if (have > content_len) have = content_len;
        memcpy(body, hdr + hdr_len, have);
        while (have < content_len) {
            ssize_t r = read(fd, body + have, content_len - have);
            if (r <= 0) break;
            have += (size_t)r;
        }
        if (have == content_len) {
            const int taken = handle_transcribe(fd, hdr, body, content_len, query, translate);
            free(body);
            if (taken) return;
        } else {
            free(body);
            refuse_json(fd, 400, "Bad Request", "invalid_request_error",
                        "incomplete_body", "the body was shorter than Content-Length");
            return;
        }
    } else {
        refuse_json(fd, 404, "Not Found", "invalid_request_error", "not_found",
                    "no such endpoint");
        return;
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

/* The model's own mynah.json: its name, its engine, and the streaming shape
 * the lag thresholds are derived from. Read directly rather than through the
 * library, because the server needs it before it needs anything else and
 * because these three fields are description, not weights. A directory without
 * them still serves; it just cannot name itself. */
static void read_model_meta(model_group *g) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/mynah.json", g->dir);
    FILE *f = fopen(path, "rb");
    if (f == NULL) return;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || n > 8 * 1024 * 1024) { fclose(f); return; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (buf == NULL) { fclose(f); return; }
    const size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    cJSON *j = cJSON_Parse(buf);
    free(buf);
    if (j == NULL) return;
    const cJSON *v = cJSON_GetObjectItem(j, "name");
    /* The pack's own name becomes the group's name when the CLI did not give
     * one: a fleet started with a bare `-m dir` must still be routable by the
     * name /v1/models prints, and that name can only come from here. */
    if (cJSON_IsString(v)) {
        snprintf(g->pack, sizeof(g->pack), "%s", v->valuestring);
        if (g->name[0] == '\0')
            snprintf(g->name, sizeof(g->name), "%s", v->valuestring);
    }
    v = cJSON_GetObjectItem(j, "engine");
    if (cJSON_IsString(v)) snprintf(g->engine, sizeof(g->engine), "%s", v->valuestring);
    const cJSON *st = cJSON_GetObjectItem(j, "streaming");
    if (cJSON_IsObject(st)) {
        v = cJSON_GetObjectItem(st, "encoder_frame_ms");
        if (cJSON_IsNumber(v)) g->encoder_frame_ms = v->valuedouble;
        v = cJSON_GetObjectItem(st, "default_preset_index");
        if (cJSON_IsNumber(v)) g->default_preset_index = v->valueint;
    }
    cJSON_Delete(j);
}

/* ------------------------------------------------------ --model, the parser
 *
 * `name=dir[:workers=W][:cpus=N][:cap=C][:quant=int8|int4|f32][:lookahead=L]`
 * and, for the shape `-m` has always had, a bare `dir` whose name then comes
 * from the pack's own mynah.json.
 *
 * The directory is everything up to the first `:` that starts a KNOWN option,
 * so a path containing a colon still works. A `:word=` that is NOT one of the
 * five options is an ERROR rather than something silently folded into the
 * path: a typo in `:wrokers=4` that quietly produced a one-worker group is
 * exactly the kind of mis-plan this repository refuses to ship.
 *
 * `spec` is modified in place (it is argv, which lives as long as the process)
 * and `g->dir` points into it. Returns 0, or -1 having said why on stderr. */
static const char *const OPT_KEYS[] = { "workers", "cpus", "cap", "quant",
                                        "lookahead", NULL };

/* The option starting at `s[at]`, or -1 when nothing does. -2 means "this
 * looks like an option and is not one", which is a typo and not a path. */
static int opt_at(const char *s, size_t at) {
    if (s[at] != ':') return -1;
    for (int k = 0; OPT_KEYS[k] != NULL; k++) {
        const size_t n = strlen(OPT_KEYS[k]);
        if (strncmp(s + at + 1, OPT_KEYS[k], n) == 0 && s[at + 1 + n] == '=') return k;
    }
    size_t j = at + 1;
    while ((s[j] >= 'a' && s[j] <= 'z') || s[j] == '_' || s[j] == '-') ++j;
    return (j > at + 1 && s[j] == '=') ? -2 : -1;
}

static int parse_quant_name(const char *v, int *out) {
    if (strcmp(v, "int8") == 0) { *out = MYNAH_ASR_QUANT_INT8; return 0; }
    if (strcmp(v, "int4") == 0) { *out = MYNAH_ASR_QUANT_INT4; return 0; }
    if (strcmp(v, "f32") == 0)  { *out = MYNAH_ASR_QUANT_F32;  return 0; }
    return -1;
}

#define MAX_SPEC_OPTS 8

static int parse_model_spec(char *spec, model_group *g) {
    memset(g, 0, sizeof(*g));
    g->quant = -1;
    g->lookahead = -1;

    char *rest = spec;
    char *eq = strchr(spec, '=');
    char *colon = strchr(spec, ':');
    /* `name=` only when the `=` comes before any option separator: in
     * `dir:cap=4` the first `=` belongs to an option, not to a name. */
    if (eq != NULL && (colon == NULL || eq < colon)) {
        *eq = '\0';
        if (spec[0] == '\0') {
            fprintf(stderr, "mynah-asr-server: --model: empty name before '='\n");
            return -1;
        }
        snprintf(g->name, sizeof(g->name), "%s", spec);
        rest = eq + 1;
    }

    int keys[MAX_SPEC_OPTS];
    size_t at[MAX_SPEC_OPTS];
    int n_opt = 0;
    for (size_t j = 0; rest[j] != '\0'; j++) {
        const int k = opt_at(rest, j);
        if (k == -1) continue;
        if (k == -2) {
            fprintf(stderr, "mynah-asr-server: --model '%s': unknown option at ':%.16s'; "
                            "accepted: workers, cpus, cap, quant, lookahead\n",
                    spec, rest + j + 1);
            return -1;
        }
        if (n_opt >= MAX_SPEC_OPTS) {
            fprintf(stderr, "mynah-asr-server: --model '%s': too many options\n", spec);
            return -1;
        }
        keys[n_opt] = k;
        at[n_opt] = j;
        ++n_opt;
    }
    for (int a = 0; a < n_opt; a++) rest[at[a]] = '\0';   /* each value now ends at a NUL */
    for (int a = 0; a < n_opt; a++) {
        char *val = rest + at[a] + 1 + strlen(OPT_KEYS[keys[a]]) + 1;
        if (val[0] == '\0') {
            fprintf(stderr, "mynah-asr-server: --model '%s': ':%s=' has no value\n",
                    spec, OPT_KEYS[keys[a]]);
            return -1;
        }
        switch (keys[a]) {
            case 0: g->workers = atoi(val); break;
            case 1: g->cpus = atoi(val); break;
            case 2: g->cap = atoi(val); break;
            case 3:
                if (parse_quant_name(val, &g->quant) != 0) {
                    fprintf(stderr, "mynah-asr-server: --model '%s': quant '%s' is not "
                                    "one of int8, int4, f32\n", spec, val);
                    return -1;
                }
                break;
            case 4: g->lookahead = atoi(val); break;
            default: break;
        }
    }
    g->dir = rest;
    if (g->dir[0] == '\0') {
        fprintf(stderr, "mynah-asr-server: --model: empty model directory\n");
        return -1;
    }
    if (g->workers < 0 || g->cpus < 0 || g->cap < 0) {
        fprintf(stderr, "mynah-asr-server: --model '%s': workers/cpus/cap cannot be "
                        "negative\n", spec);
        return -1;
    }
    return 0;
}

/* The group names, as prefork wants them and as every refusal quotes them. */
static const char *g_group_names[MAX_GROUPS];
static const char *const *group_name_table(void) {
    for (int i = 0; i < g_n_groups; i++) g_group_names[i] = g_groups[i].name;
    return g_group_names;
}

/* A model name a client asked for, checked against the fleet from INSIDE a
 * worker. Returns 0 when it is this worker's own (or unspecified), and 404
 * with `msg` filled otherwise.
 *
 * Two different failures share the 404, and the message is what separates
 * them, because the action a client must take differs:
 *   - a name nobody serves: the accepted set is named, and that is the end;
 *   - a name ANOTHER group serves, which means the router could not see it in
 *     the bounded prefix it classifies on -- a multipart `model` field sitting
 *     after the audio. Serving it here would answer from the wrong weights,
 *     which is the one outcome that must never happen, so it is refused and
 *     the message says where to put the name instead. */
static int check_model_name(const char *want, char *msg, size_t cap) {
    if (want == NULL || want[0] == '\0') return 0;
    const int at2 = mynah_asr_prefork_model_match(group_name_table(), g_n_groups, want);
    if (at2 == g_group) return 0;
    char tok[64];
    refuse_token(want, tok, sizeof(tok));
    const char *set = mynah_asr_prefork_model_set();
    if (at2 >= 0) {
        snprintf(msg, cap, "'%s' is another group of this fleet, and this request "
                           "reached '%s': name the model in ?model= or before the file "
                           "in the form, where the router can see it",
                 tok, g_groups[g_group].name);
    } else if (set != NULL && set[0] != '\0') {
        snprintf(msg, cap, "no model '%s' here; this fleet serves: %s", tok, set);
    } else {
        snprintf(msg, cap, "no model '%s' here; this server serves '%s'",
                 tok, g_groups[g_group].name);
    }
    return 404;
}

/* RLIM_INFINITY is a real answer and prints as one: the sentinel rendered as a
 * number is 18446744073709551615 on a line meant to be read. */
static void nofile_text(rlim_t v, char *out, size_t cap) {
    if (v == RLIM_INFINITY) snprintf(out, cap, "unlimited");
    else                    snprintf(out, cap, "%llu", (unsigned long long)v);
}

/* ---------------------------------------------------- the descriptor ceiling
 *
 * S7-2. A stream costs TWO descriptors -- the client socket and the dup() the
 * ingest side reads from -- so a hundred streams is over two hundred, plus the
 * listener, the metrics listener and one socketpair end per worker. A process
 * takes whatever soft limit it inherited (256 on some systems, 1024 on most),
 * and when that runs out accept() fails with EMFILE: a refusal with no rung,
 * no counter and no code, which is exactly the unexplained refusal this raise
 * exists to prevent.
 *
 * Raise the SOFT limit toward the HARD one and never above it -- crossing the
 * hard limit needs a privilege a server must not require -- then report what
 * it ACTUALLY became. Called before the models are opened and before the fork,
 * so every worker inherits the result rather than repeating it. */
static void raise_nofile_limit(int max_connections, int workers) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) != 0) {
        fprintf(stderr, "mynah-asr-server: WARNING RLIMIT_NOFILE is unreadable (%s): "
                        "this process cannot say what its descriptor ceiling is\n",
                strerror(errno));
        return;
    }
    const rlim_t had = rl.rlim_cur;
    rlim_t want = rl.rlim_max;
#if defined(__APPLE__)
    /* Darwin refuses RLIM_INFINITY here, and anything above
     * kern.maxfilesperproc; OPEN_MAX is the most a process may ask for. */
    if (want == RLIM_INFINITY || want > (rlim_t)OPEN_MAX) want = (rlim_t)OPEN_MAX;
#else
    /* Linux refuses a soft limit above fs.nr_open with EPERM, and
     * RLIM_INFINITY is above it by construction. */
    if (want == RLIM_INFINITY) want = (rlim_t)1048576;
#endif
    /* Halving backoff rather than one attempt: the hard limit says what the
     * kernel allows in principle, and a sysctl below it (Darwin's
     * maxfilesperproc, a container's fs.nr_open) turns the honest ask into
     * EINVAL. Landing above what we inherited beats landing on nothing. */
    for (rlim_t n = want; n > had; n = had + (n - had) / 2) {
        rl.rlim_cur = n;
        if (setrlimit(RLIMIT_NOFILE, &rl) == 0) break;
    }

    struct rlimit now;
    if (getrlimit(RLIMIT_NOFILE, &now) != 0) now = rl;
    char soft[32], hard[32], before[32];
    nofile_text(now.rlim_cur, soft, sizeof(soft));
    nofile_text(now.rlim_max, hard, sizeof(hard));
    nofile_text(had, before, sizeof(before));
    /* Two per connection, plus one channel per worker and the two listeners;
     * the constant is slack for the model files, the pack's JSON and stdio. */
    const rlim_t need = (rlim_t)2 * (rlim_t)max_connections + (rlim_t)workers + 32;
    fprintf(stderr, "mynah-asr-server: RLIMIT_NOFILE soft %s -> %s (hard %s); "
                    "this configuration admits %d connections at 2 descriptors each, "
                    "so it needs about %llu\n",
            before, soft, hard, max_connections, (unsigned long long)need);
    if (now.rlim_cur != RLIM_INFINITY && now.rlim_cur < need) {
        fprintf(stderr,
            "mynah-asr-server: WARNING the descriptor ceiling is BELOW this "
            "configuration. soft=%s hard=%s, needed about %llu. %s Raise the hard "
            "limit (ulimit -Hn, LimitNOFILE= in the unit file) or lower "
            "--cap/--prefork. Left as it is, the streams past the ceiling are "
            "refused by accept() with EMFILE -- a refusal no rung of the admission "
            "ladder issued and no counter in this process explains.\n",
            soft, hard, (unsigned long long)need,
            now.rlim_cur >= now.rlim_max
                ? "The soft limit is at the hard one, so nothing here can raise it "
                  "further."
                : "The kernel refused to raise it to the hard limit, which means a "
                  "sysctl below it (kern.maxfilesperproc, fs.nr_open) is the real "
                  "ceiling.");
    }
    fflush(stderr);
}

static void usage(void) {
    fprintf(stderr,
        "usage: mynah-asr-server -m <model_dir> [-p 8090] [--threads 4] [--batch 8] [--quant int8|int4]\n"
        "       [--backend cpu|metal|cuda] [--caps auto|scalar|avx2|vnni]\n"
        "       [--model name=dir[:workers=W][:cpus=N][:cap=C][:quant=int8|int4|f32][:lookahead=L]]\n"
        "                            one MODEL GROUP, repeatable. One process holds one model,\n"
        "                            so several --model imply --prefork: if you do not give one\n"
        "                            the group count becomes it. Requests choose a group with\n"
        "                            ?model= (WebSocket and REST) or a multipart `model` field;\n"
        "                            an unknown name is 404 model_not_found. -m is one group\n"
        "                            named by its own mynah.json.\n"
        "       [--default <name>]   the group a request naming no model gets (default: the first)\n"
        "       [--lid-model <dir>]  detector for language=auto on models that\n"
        "                            cannot detect it themselves (Canary)\n"
        "       [--prefork W] [--prefork-threads T] [--cap C]\n"
        "                            W pinned worker processes (Linux: core-major cpu slices),\n"
        "                            T threads each (default: cpus/W), C stream slots per\n"
        "                            worker before it refuses 503 (default: --threads)\n"
        "       --prefork-plan       print the machine's topology and the W/T sweep, exit\n"
        "       [--batch-window-ms N]  a step may wait up to N ms for more streams\n"
        "                            to become ready, so its fixed cost is amortised over\n"
        "                            the set it was priced for (0 = step as soon as\n"
        "                            anything is ready). Never waits for a stream's first\n"
        "                            chunk or for one that is finalizing\n"
        "       [--idle-ms 60000]    a stream silent for this long is cancelled (idle_timeout)\n"
        "       [--ping-ms 20000]    server-side WebSocket ping period (0 = never)\n"
        "       [--max-audio-seconds 14400]  audio one stream may send (0 = no cap)\n"
        "       [--ring-seconds 30]  PCM buffered per stream before the client is throttled\n"
        "       [--max-frame-bytes 1048576]  a larger WebSocket frame ends the stream\n"
        "       [--max-pending N]    offline requests queued (default 2*--threads), 503 beyond\n"
        "       [--metrics-port N]   Prometheus text on its OWN port (off by default)\n"
        "       [--metrics-bind ADDR]  where that port listens (default 127.0.0.1)\n"
        "       --dispatch-map [--json]  which kernel/backend/pool this binary resolved, exit\n"
        "  env: MYNAH_ASR_PREFORK_QUEUE (queued per worker, default 1; 0 = refuse at once),\n"
        "       MYNAH_ASR_PREFORK_QUEUE_MS (queue deadline, default 2000),\n"
        "       MYNAH_ASR_PREFORK_SERVICE_MS (service cap, default 30000),\n"
        "       MYNAH_ASR_LISTEN_BACKLOG (listen() backlog; default 2*workers*(slots+queue),\n"
        "         floored at SOMAXCONN, capped at 4096, and clamped by the kernel's somaxconn)\n");
}

int main(int argc, char **argv) {
    /* FIRST statement, exactly as in cli/main.c: a binary built for an ISA this
     * CPU lacks says so in one line instead of dying with a bare SIGILL inside
     * a kernel three frames down. Fires only on a DEFINITE absence (S3-2). */
    const int isa = mynah_asr_isa_guard();
    if (isa != 0) return isa;

    /* The same dispatch report the CLI prints, from the same code: a server
     * benchmark that cannot state its kernels is not a measurement
     * (ENGINEERING.md §5). */
    if (argc >= 2 && strcmp(argv[1], "--dispatch-map") == 0) {
        const int json = argc >= 3 && strcmp(argv[2], "--json") == 0;
        return mynah_asr_dispatch_print(stdout, json) < 0 ? 1 : 0;
    }

    const char *model_dir = NULL, *lid_dir = NULL, *default_name = NULL;
    int port = 8090, n_threads = 4;
    int batch_window_ms = 0;   /* --batch-window-ms: 0 = step the moment anything is ready */
    int prefork_workers = 0, prefork_threads = 0, cap = 0, plan_only = 0;
    int metrics_port = 0;
    const char *metrics_bind = "127.0.0.1";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            if (g_n_groups >= MAX_GROUPS) {
                fprintf(stderr, "mynah-asr-server: at most %d --model groups\n",
                        MAX_GROUPS);
                return 2;
            }
            if (parse_model_spec(argv[++i], &g_groups[g_n_groups]) != 0) return 2;
            ++g_n_groups;
        }
        else if (strcmp(argv[i], "--default") == 0 && i + 1 < argc) default_name = argv[++i];
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
        else if (strcmp(argv[i], "--batch-window-ms") == 0 && i + 1 < argc)
            batch_window_ms = atoi(argv[++i]);
        else if (strcmp(argv[i], "--idle-ms") == 0 && i + 1 < argc) g_idle_ms = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ping-ms") == 0 && i + 1 < argc) g_ping_ms = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-audio-seconds") == 0 && i + 1 < argc)
            g_max_audio_seconds = atof(argv[++i]);
        else if (strcmp(argv[i], "--ring-seconds") == 0 && i + 1 < argc) g_ring_seconds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-frame-bytes") == 0 && i + 1 < argc)
            g_max_frame_bytes = (size_t)strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--max-pending") == 0 && i + 1 < argc) g_max_pending = atoi(argv[++i]);
        else if (strcmp(argv[i], "--metrics-port") == 0 && i + 1 < argc) metrics_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--metrics-bind") == 0 && i + 1 < argc) metrics_bind = argv[++i];
        else if (strcmp(argv[i], "--prefork-plan") == 0) plan_only = 1;
        else { usage(); return 2; }
    }
    if (g_max_batch < 1) g_max_batch = 1;
    if (g_max_batch > 64) g_max_batch = 64;
    if (n_threads < 1) n_threads = 1;
    if (cap <= 0) cap = n_threads;   /* one slot per HTTP thread: a WS stream holds one */
    if (g_idle_ms < 1000) g_idle_ms = 1000;
    if (g_ping_ms < 0) g_ping_ms = 0;              /* 0 = never ping */
    if (g_ping_ms > 0 && g_ping_ms < 100) g_ping_ms = 100;
    if (g_max_audio_seconds < 0.0) g_max_audio_seconds = 0.0;   /* 0 = no cap */
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
    pf.metrics_port = metrics_port;
    pf.metrics_bind = metrics_bind;
    if (plan_only) {
        mynah_asr_prefork_print_plan(&pf, stdout);
        return 0;
    }
    /* ---- the fleet's plan, resolved before anything is opened -------------
     *
     * `-m dir` is one more group named by its own pack, which is what keeps
     * every existing invocation working unchanged: one group is the
     * single-model server this file has always been, and every group-aware
     * branch below collapses. */
    if (model_dir != NULL) {
        if (g_n_groups >= MAX_GROUPS) {
            fprintf(stderr, "mynah-asr-server: at most %d model groups\n", MAX_GROUPS);
            return 2;
        }
        memset(&g_groups[g_n_groups], 0, sizeof(g_groups[0]));
        g_groups[g_n_groups].dir = model_dir;
        g_groups[g_n_groups].quant = -1;
        g_groups[g_n_groups].lookahead = -1;
        ++g_n_groups;
    }
    if (g_n_groups == 0) { usage(); return 2; }
    if (prefork_workers < 0) { usage(); return 2; }

    for (int i = 0; i < g_n_groups; i++) {
        read_model_meta(&g_groups[i]);
        if (g_groups[i].name[0] == '\0') {
            /* A pack with no name of its own still has to be routable, so the
             * directory's last component becomes the name and the banner shows
             * what happened. Never a made-up "model0": a name nobody can guess
             * is a name nobody can send. */
            const char *base = strrchr(g_groups[i].dir, '/');
            base = (base != NULL && base[1] != '\0') ? base + 1 : g_groups[i].dir;
            snprintf(g_groups[i].name, sizeof(g_groups[i].name), "%s", base);
            fprintf(stderr, "mynah-asr-server: %s has no \"name\" in mynah.json; "
                            "this group is called '%s'\n",
                    g_groups[i].dir, g_groups[i].name);
        }
        if (g_groups[i].quant < 0) g_groups[i].quant = g_quant;
        for (int j = 0; j < i; j++) {
            if (strcmp(g_groups[i].name, g_groups[j].name) != 0) continue;
            fprintf(stderr, "mynah-asr-server: two groups are both called '%s'. "
                            "A name is how a request chooses its weights, so two "
                            "groups with one name is a routing table that cannot "
                            "answer; give one of them `--model othername=%s`\n",
                    g_groups[i].name, g_groups[i].dir);
            return 2;
        }
    }

    /* --default: which group a request that names no model gets. */
    g_default_group = 0;
    if (default_name != NULL) {
        const int at = mynah_asr_prefork_model_match(group_name_table(), g_n_groups,
                                                     default_name);
        if (at < 0) {
            fprintf(stderr, "mynah-asr-server: --default '%s' names no group%s\n",
                    default_name, at == -2 ? " unambiguously" : "");
            return 2;
        }
        g_default_group = at;
    }

    /* ---- how many workers each group gets --------------------------------
     *
     * More than one model IMPLIES prefork, because one process holds one model:
     * there is no shape in which a single process serves two groups. When
     * --prefork was not given the group count becomes it, and the banner says
     * that it did rather than leaving an operator to infer a topology. When it
     * WAS given and does not fit the groups, that is an error: dropping a model
     * the operator asked for, silently, is the failure this whole feature is
     * supposed to make impossible. */
    int grp_workers[MAX_GROUPS], grp_slots[MAX_GROUPS], grp_cpus[MAX_GROUPS];
    int have_cpus = 0, have_slots = 0;
    for (int i = 0; i < g_n_groups; i++) {
        grp_workers[i] = g_groups[i].workers;
        grp_slots[i] = g_groups[i].cap > 0 ? g_groups[i].cap : cap;
        grp_cpus[i] = g_groups[i].cpus;
        if (g_groups[i].cap > 0) have_slots = 1;
        if (g_groups[i].cpus > 0) have_cpus = 1;
    }
    if (g_n_groups > 1) {
        int want = 0, unspec = 0;
        for (int i = 0; i < g_n_groups; i++) {
            if (grp_workers[i] > 0) want += grp_workers[i];
            else ++unspec;
        }
        if (prefork_workers == 0) {
            prefork_workers = want + unspec;
            fprintf(stderr, "mynah-asr-server: %d model groups imply prefork; "
                            "--prefork was not given, so it is %d "
                            "(one process holds one model)\n",
                    g_n_groups, prefork_workers);
        }
        if (prefork_workers < g_n_groups) {
            fprintf(stderr, "mynah-asr-server: --prefork %d cannot hold %d model "
                            "groups; a group with no worker could not be served and "
                            "would have to be dropped silently. Ask for at least "
                            "%d workers.\n",
                    prefork_workers, g_n_groups, g_n_groups);
            return 2;
        }
        if (unspec == 0) {
            if (want != prefork_workers) {
                fprintf(stderr, "mynah-asr-server: the per-model `:workers=` counts sum "
                                "to %d but --prefork says %d. They are the same number; "
                                "drop one of them rather than leaving the server to "
                                "guess which you meant.\n", want, prefork_workers);
                return 2;
            }
        } else {
            const int left = prefork_workers - want;
            if (left < unspec) {
                fprintf(stderr, "mynah-asr-server: `:workers=` already claims %d of the "
                                "%d workers, leaving %d for the %d groups that named "
                                "none. Raise --prefork to at least %d.\n",
                        want, prefork_workers, left, unspec, want + unspec);
                return 2;
            }
            const int base = left / unspec, extra = left % unspec;
            int k = 0;
            for (int i = 0; i < g_n_groups; i++) {
                if (grp_workers[i] > 0) continue;
                grp_workers[i] = base + (k < extra ? 1 : 0);
                ++k;
            }
        }
    } else if (grp_workers[0] > 0 && prefork_workers == 0) {
        prefork_workers = grp_workers[0];   /* `--model x=dir:workers=4` alone */
    }

    pf.workers = prefork_workers;
    /* BEFORE any model is opened: the pool resolves its width once, on first
     * use, so MYNAH_ASR_THREADS has to be right before anything can dispatch. */
    if (prefork_workers > 0) {
        mynah_asr_prefork_reserve_threads(&pf);
        prefork_workers = pf.workers;
    }

    /* The fleet's plan is complete BEFORE the listener is bound, because the
     * backlog is sized from it: a group's own `:cap=` is part of how many
     * connections this server will hold at once. `listen_fd` is the one field
     * that cannot be filled yet and is set where the socket is bound. */
    pf.models = group_name_table();
    pf.model_count = g_n_groups;
    pf.default_model = g_default_group;
    pf.model_workers = g_n_groups > 1 ? grp_workers : NULL;
    pf.model_slots = have_slots ? grp_slots : NULL;
    pf.model_cpus = have_cpus ? grp_cpus : NULL;

    /* S7-1/S7-2, resolved here: before any model is opened, before the fork and
     * before the listener exists, so the ceiling the parent raises is the
     * ceiling every worker inherits. W is 1 when there is no prefork, and the
     * single-process server sizes its queue by the same arithmetic. */
    const int listen_workers = prefork_workers > 0 ? prefork_workers : 1;
    const int max_conns = mynah_asr_prefork_max_connections(&pf, listen_workers);
    const int backlog = mynah_asr_prefork_listen_backlog(&pf, listen_workers);
    const int somaxconn = mynah_asr_prefork_somaxconn();
    raise_nofile_limit(max_conns, listen_workers);

    signal(SIGPIPE, SIG_IGN);
    {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = on_stop;        /* no SA_RESTART: accept/poll must return EINTR */
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);
        /* Restartable: a dump request must not turn a read() in an ingest
         * thread into an EINTR the stream then has to recover from. */
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = on_usr1;
        sa.sa_flags = SA_RESTART;
        sigaction(SIGUSR1, &sa, NULL);
    }

    /* ---- every model, opened HERE, in the parent, before the fork ---------
     *
     * Two things follow from that and both are load bearing. The weights are
     * mapped once and the fork makes them COW-shared, so N groups cost one copy
     * of each pack rather than one per worker; and a path that cannot be opened
     * is a STARTUP failure, with nothing yet forked and nothing yet listening,
     * rather than a group that quietly answers 503 forever. */
    for (int i = 0; i < g_n_groups; i++) {
        g_groups[i].model = mynah_asr_load_quant(g_groups[i].dir, g_groups[i].quant);
        if (g_groups[i].model == NULL) {
            fprintf(stderr, "mynah-asr-server: could not open the model for group "
                            "'%s' (%s); refusing to start\n",
                    g_groups[i].name, g_groups[i].dir);
            for (int j = 0; j < i; j++) mynah_asr_free(g_groups[j].model);
            return 1;
        }
    }
    if (lid_dir) {
        /* NOT a group: a detector in front of the models that cannot detect.
         * It is resident in the parent like every group's weights, so every
         * worker inherits it COW and none of them routes to it. */
        g_lid = mynah_asr_load_quant(lid_dir, g_groups[g_default_group].quant);
        if (!g_lid) {
            for (int i = 0; i < g_n_groups; i++) mynah_asr_free(g_groups[i].model);
            return 1;
        }
        if (!mynah_asr_can_detect_lang(g_lid)) {
            fprintf(stderr, "mynah-asr-server: %s cannot detect the language "
                            "(use a model with an 'auto' prompt, e.g. nemotron)\n", lid_dir);
            mynah_asr_free(g_lid);
            for (int i = 0; i < g_n_groups; i++) mynah_asr_free(g_groups[i].model);
            return 1;
        }
    }

    /* S7-1. The backlog is derived from what this fleet will admit (above) and
     * printed on the banner, never a literal: a hundred clients arriving
     * together against the old listen(srv, 64) were dropped as SYNs before
     * accept(), and the run then measured the accept queue rather than the
     * server. This is the ONE place either shape binds -- with --prefork the
     * router keeps this descriptor across the fork and every worker closes its
     * copy -- so the single process and the fleet get the same queue. */
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port),
                               .sin_addr.s_addr = htonl(INADDR_ANY)};
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(srv, backlog) != 0) {
        fprintf(stderr, "mynah-asr-server: bind/listen failed on port %d\n", port);
        return 1;
    }
    if (somaxconn > 0 && somaxconn < backlog) {
        /* listen() clamps in silence, so this is the only place the difference
         * can be stated. Above the clamp a burst is dropped as SYNs before
         * accept(), which makes the concurrency number a measurement of the
         * accept queue -- ENGINEERING.md §7, a contradiction nobody benchmarks. */
        fprintf(stderr,
            "mynah-asr-server: WARNING the listen backlog was sized at %d for this "
            "capacity and the kernel caps it at %d (%s). Connections arriving "
            "together past %d are dropped as SYNs before accept(), where nothing in "
            "this process can count them: raise the sysctl, or read any concurrency "
            "above %d as a measurement of the accept queue.\n",
            backlog, somaxconn, mynah_asr_prefork_somaxconn_knob(),
            somaxconn, somaxconn);
    }

    /* ---- prefork: fork W workers now, with every model mapped and no thread
     * yet. The parent routes until shutdown and never enters a model; a worker
     * continues below with chan_fd as its only source of connections, and with
     * exactly one of the groups still open. */
    int chan_fd = -1;
    if (prefork_workers > 0) {
        pf.listen_fd = srv;
        const mynah_asr_prefork_role role = mynah_asr_prefork_run(&pf, &g_shutdown, &chan_fd);
        if (role == MYNAH_ASR_PREFORK_ERROR) {
            close(srv);
            for (int i = 0; i < g_n_groups; i++) mynah_asr_free(g_groups[i].model);
            mynah_asr_free(g_lid);
            return 1;
        }
        if (role == MYNAH_ASR_PREFORK_PARENT_DONE) {
            for (int i = 0; i < g_n_groups; i++) mynah_asr_free(g_groups[i].model);
            mynah_asr_free(g_lid);
            return 0;
        }
        srv = -1;   /* a worker never accepts: the router closed our copy */
        g_group = mynah_asr_prefork_worker_model();
    } else {
        g_group = g_default_group;
    }
    if (g_group < 0 || g_group >= g_n_groups) g_group = 0;

    /* ONE MODEL PER PROCESS, made true rather than assumed: every other group's
     * weights are closed right here, in the child, so this address space can
     * only ever run the model the router believes it holds. The table's NAMES
     * stay, which is what lets this worker answer /v1/models for the fleet and
     * refuse a request that named a model it is not holding. */
    for (int i = 0; i < g_n_groups; i++) {
        if (i == g_group) continue;
        mynah_asr_free(g_groups[i].model);
        g_groups[i].model = NULL;
    }
    g_model = g_groups[g_group].model;
    g_quant = g_groups[g_group].quant;
    if (g_groups[g_group].cap > 0) cap = g_groups[g_group].cap;
    g_encoder_frame_ms = g_groups[g_group].encoder_frame_ms;
    g_default_preset_index = g_groups[g_group].default_preset_index;
    snprintf(g_model_name, sizeof(g_model_name), "%s",
             g_groups[g_group].pack[0] != '\0' ? g_groups[g_group].pack
                                               : g_groups[g_group].name);
    snprintf(g_model_engine, sizeof(g_model_engine), "%s",
             g_groups[g_group].engine[0] != '\0' ? g_groups[g_group].engine : "unknown");
    model_dir = g_groups[g_group].dir;
    if (g_lid != NULL && mynah_asr_can_detect_lang(g_model)) {
        /* It would never be consulted by THIS group, so this worker lets go of
         * it. Per worker and not per fleet: in a mixed fleet the group that
         * cannot detect still needs it, and only this process knows which it
         * is. The parent's copy is what the others inherited. */
        fprintf(stderr, "mynah-asr-server: --lid-model is not consulted by group '%s', "
                        "which detects the language itself\n", g_groups[g_group].name);
        mynah_asr_free(g_lid);
        g_lid = NULL;
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
    sc.batch_window_ms = batch_window_ms;
    if (mynah_asr_sched_start(&sc) != 0) {
        fprintf(stderr, "mynah-asr-server: the scheduler failed to start\n");
        return 1;
    }

    /* ENGINEERING.md §5: unconditionally, in the same process, before anything
     * is served -- what the environment asked for, what this build does with
     * it, the whole server configuration, and the cpu mask actually in force. */
    {
        mynah_asr_obs_config oc;
        memset(&oc, 0, sizeof(oc));
        oc.model_dir = model_dir;
        oc.model_name = g_model_name;
        oc.group = g_groups[g_group].name;
        oc.engine = g_model_engine;
        oc.quant = g_quant == MYNAH_ASR_QUANT_INT8 ? "int8"
                 : g_quant == MYNAH_ASR_QUANT_INT4 ? "int4" : "f32";
        oc.lid_dir = lid_dir;
        oc.streaming = mynah_asr_sched_streaming();
        oc.n_lookaheads = mynah_asr_lookaheads(g_model, oc.lookaheads);
        if (oc.n_lookaheads > 0) {
            const int d = g_default_preset_index >= 0 &&
                          g_default_preset_index < oc.n_lookaheads
                              ? g_default_preset_index : 0;
            oc.lookahead_default = oc.lookaheads[d];
            /* One chunk is (lookahead + 1) encoder frames. That is the cadence
             * the emission-lag thresholds are multiples of, so it is computed
             * from the model rather than assumed. */
            if (g_encoder_frame_ms > 0.0)
                oc.chunk_ms = (double)(oc.lookahead_default + 1) * g_encoder_frame_ms;
        }
        oc.port = port;
        oc.cap = cap;
        oc.ring_seconds = g_ring_seconds;
        oc.idle_ms = g_idle_ms;
        oc.ping_ms = g_ping_ms;
        oc.max_audio_seconds = g_max_audio_seconds;
        oc.max_frame_bytes = g_max_frame_bytes;
        oc.max_pending = g_max_pending;
        oc.http_threads = n_threads;
        oc.batch = g_max_batch;
        oc.prefork_workers = prefork_workers;
        oc.prefork_threads = mynah_asr_prefork_worker_threads();
        oc.metrics_port = metrics_port;
        oc.metrics_bind = metrics_bind;
        oc.listen_backlog = backlog;
        oc.listen_somaxconn = somaxconn;
        mynah_asr_obs_init(&oc);
        mynah_asr_obs_banner();
    }

    /* /metrics, in a SINGLE-PROCESS server only: with --prefork the parent owns
     * the port and answers for the fleet (see prefork.h), and it bound the
     * listener before the fork. A bind that fails is fatal here -- an
     * observability port that silently did not open is worse than none. */
    int metrics_fd = -1;
    if (metrics_port > 0 && prefork_workers == 0) {
        metrics_fd = mynah_asr_metrics_listen(metrics_bind, metrics_port);
        if (metrics_fd < 0) {
            mynah_asr_sched_stop();
            mynah_asr_free(g_lid);
            mynah_asr_free(g_model);
            return 1;
        }
        fprintf(stderr, "mynah-asr-server: /metrics on %s:%d\n",
                metrics_bind, metrics_port);
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_t t;
        pthread_create(&t, NULL, worker, NULL);
        pthread_detach(t);
    }
    if (chan_fd >= 0)
        fprintf(stderr, "mynah-asr-server %s: prefork worker %d ready, group '%s' "
                        "(%d http threads, %d stream slots, batch %d, streaming %s)\n",
                mynah_asr_version(), mynah_asr_prefork_worker_index(),
                g_groups[g_group].name, n_threads, cap,
                g_max_batch, mynah_asr_sched_streaming() ? "yes" : "no (offline-only model)");
    else
        /* The backlog belongs on this line too: a single-process server has no
         * admission ladder to print it beside, and it is the first limit a
         * burst of connections meets. */
        fprintf(stderr, "mynah-asr-server %s: listening on :%d (%d http threads, "
                        "%d stream slots, batch %d, backlog %d, batch-window %d ms, "
                        "streaming %s)\n"
                        "  one scheduler thread owns the model; offline jobs share its steps\n"
                        "  POST /v1/audio/transcriptions | GET /v1/audio/stream (WS) | /v1/models | /v1/health\n",
                mynah_asr_version(), port, n_threads, cap, g_max_batch, backlog,
                batch_window_ms,
                mynah_asr_sched_streaming() ? "yes" : "no (offline-only model)");

    /* Where a connection comes from is the ONLY difference between the single
     * process server and a prefork worker: a worker has no listening socket,
     * the router accepted, chose it and passed the descriptor down a
     * socketpair. Everything after this loop is the same code in both shapes. */
    while (!g_shutdown) {
        /* SIGUSR1 (S3-4). The parent forwards it to every worker, so one
         * `kill -USR1 <parent>` produces the routing table from the router and
         * this block from each worker. */
        if (mynah_asr_prefork_take_dump_request()) mynah_asr_obs_dump();
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
            /* The service listener and, when it exists, the metrics listener in
             * ONE poll: a scrape is then a bounded slice of this loop rather
             * than a thread reading counters the loop is writing. */
            struct pollfd pfd[2];
            pfd[0].fd = srv; pfd[0].events = POLLIN; pfd[0].revents = 0;
            pfd[1].fd = metrics_fd; pfd[1].events = POLLIN; pfd[1].revents = 0;
            const int nfd = metrics_fd >= 0 ? 2 : 1;
            const int ready = poll(pfd, (nfds_t)nfd, 200);
            if (ready < 0) { if (errno == EINTR) continue; break; }
            if (ready == 0) continue;
            if (nfd == 2 && (pfd[1].revents & POLLIN) != 0)
                mynah_asr_metrics_service(metrics_fd, mynah_asr_obs_render_metrics, NULL);
            if ((pfd[0].revents & POLLIN) == 0) continue;
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
    if (metrics_fd >= 0) close(metrics_fd);
    mynah_asr_sched_stop();
    { struct timespec ts = {.tv_sec = 0, .tv_nsec = 200 * 1000000L}; nanosleep(&ts, NULL); }
    mynah_asr_free(g_lid);
    mynah_asr_free(g_model);
    fprintf(stderr, "mynah-asr-server: worker %d stopped\n", mynah_asr_prefork_worker_index());
    return 0;
}
