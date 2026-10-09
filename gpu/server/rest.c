/* gpu/server/rest.c — see rest.h. */
#include "rest.h"

#include "audio.h"
#include "http_io.h"
#include "http_util.h"
#include "mynah_asr.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define REST_MAX_BODY (200u * 1024u * 1024u)   /* the CPU server's MAX_BODY */

typedef struct rest_job {
    float *pcm;
    size_t n;
    char lang[24];
    int want_words;
    /* results, written by the batcher under R.mu */
    char *text;
    char lang_out[16];
    mynah_asr_word *words;
    int n_words;
    int status;               /* 0 ok, 1 failed (this request), 2 device dead, 3 shutting down */
    int done;
    double t_enq, t_start, t_end;
    struct rest_job *next;
} rest_job;

static struct {
    rest_cfg cfg;
    asr_offline_facts facts;
    pthread_mutex_t mu;
    pthread_cond_t cv;        /* the batcher waits here for work */
    pthread_cond_t done_cv;   /* the HTTP threads wait here for their job */
    rest_job *head, *tail;
    int queued, inflight, stop, dead, started;
    pthread_t th;
    /* books, under mu: requests = completed + failed + refused + active */
    unsigned long requests, completed, failed, refused_cap, refused_bad, refused_dead;
    unsigned long batches, batch_items;
    double audio_seconds, queue_ms_sum, service_ms_sum, total_ms_sum, total_ms_max;
} R;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ batcher */
static void run_group(rest_job **jobs, int n) {
    if (n <= 0) return;
    const float **pcm = calloc((size_t)n, sizeof(*pcm));
    size_t *ns = calloc((size_t)n, sizeof(*ns));
    const char **langs = calloc((size_t)n, sizeof(*langs));
    char **texts = calloc((size_t)n, sizeof(*texts));
    char (*lo)[16] = calloc((size_t)n, sizeof(*lo));
    const int ww = jobs[0]->want_words;
    mynah_asr_word **words = ww ? calloc((size_t)n, sizeof(*words)) : NULL;
    int *nw = ww ? calloc((size_t)n, sizeof(int)) : NULL;
    int rc = -2;
    if (pcm && ns && langs && texts && lo && (!ww || (words && nw))) {
        for (int i = 0; i < n; i++) { pcm[i] = jobs[i]->pcm; ns[i] = jobs[i]->n; langs[i] = jobs[i]->lang; }
        rc = asr_offline_transcribe(R.cfg.eng, n, pcm, ns, langs, texts, lo, words, nw);
    }
    for (int i = 0; i < n; i++) {
        rest_job *j = jobs[i];
        if (rc == -1) j->status = 2;
        else if (rc == -2 || !texts || !texts[i]) j->status = 1;
        else {
            j->text = texts[i];
            memcpy(j->lang_out, lo[i], sizeof(j->lang_out));
            if (ww) { j->words = words[i]; j->n_words = nw[i]; }
        }
    }
    free(pcm); free(ns); free(langs); free(texts); free(lo); free(words); free(nw);
}

static void *batcher_main(void *arg) {
    (void)arg;
    mynah_asr_thread_set_name("mynah-batcher");
    rest_job **take = calloc((size_t)R.cfg.batch, sizeof(*take));
    rest_job **grp = calloc((size_t)R.cfg.batch, sizeof(*grp));
    if (!take || !grp) { fprintf(stderr, "mynah-asr-server-cuda: batcher: out of memory\n"); _exit(70); }
    pthread_mutex_lock(&R.mu);
    for (;;) {
        while (!R.head && !R.stop) pthread_cond_wait(&R.cv, &R.mu);
        if (R.stop) break;
        /* the gather window: from the OLDEST queued request, so a lone request
         * waits at most window_ms, and a full batch goes at once */
        const double deadline = R.head->t_enq + (double)R.cfg.window_ms * 1e-3;
        while (R.queued < R.cfg.batch && !R.stop) {
            const double left = deadline - now_s();
            if (left <= 0.0) break;
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            long long ns = (long long)ts.tv_nsec + (long long)(left * 1e9);
            ts.tv_sec += (time_t)(ns / 1000000000LL);
            ts.tv_nsec = (long)(ns % 1000000000LL);
            pthread_cond_timedwait(&R.cv, &R.mu, &ts);
        }
        if (R.stop) break;
        int n = 0;
        while (R.head && n < R.cfg.batch) {
            rest_job *j = R.head;
            R.head = j->next;
            if (!R.head) R.tail = NULL;
            j->next = NULL;
            R.queued--;
            take[n++] = j;
        }
        R.inflight += n;
        R.batches++;
        R.batch_items += (unsigned long)n;
        const double t0 = now_s();
        for (int i = 0; i < n; i++) take[i]->t_start = t0;
        pthread_mutex_unlock(&R.mu);

        /* word timestamps change an AED prompt, so they never share a call
         * with plain requests (the library runs them on its single path) */
        for (int pass = 0; pass < 2; pass++) {
            int g = 0;
            for (int i = 0; i < n; i++)
                if (take[i]->want_words == pass) grp[g++] = take[i];
            run_group(grp, g);
        }

        pthread_mutex_lock(&R.mu);
        const double t1 = now_s();
        int dead = 0;
        for (int i = 0; i < n; i++) {
            take[i]->t_end = t1;
            take[i]->done = 1;
            if (take[i]->status == 2) dead = 1;
        }
        R.inflight -= n;
        if (dead) R.dead = 1;
        pthread_cond_broadcast(&R.done_cv);
        if (dead) {
            /* the engine is dead: fail what is queued, let the HTTP threads
             * answer, then exit 70 like the streaming mode (no hidden fallback) */
            for (rest_job *j = R.head; j; j = j->next) { j->status = 2; j->done = 1; }
            R.head = R.tail = NULL;
            R.queued = 0;
            pthread_cond_broadcast(&R.done_cv);
            pthread_mutex_unlock(&R.mu);
            fprintf(stderr, "mynah-asr-server-cuda: the offline engine is dead (%s); exiting 70\n",
                    asr_offline_error(R.cfg.eng));
            const struct timespec pause = {0, 300 * 1000 * 1000};
            nanosleep(&pause, NULL);
            fflush(stderr);
            _exit(70);
        }
    }
    /* stopping: whatever is still queued is answered as shutting down */
    for (rest_job *j = R.head; j; j = j->next) { j->status = 3; j->done = 1; }
    R.head = R.tail = NULL;
    R.queued = 0;
    pthread_cond_broadcast(&R.done_cv);
    pthread_mutex_unlock(&R.mu);
    free(take);
    free(grp);
    return NULL;
}

int rest_start(const rest_cfg *cfg) {
    memset(&R, 0, sizeof(R));
    R.cfg = *cfg;
    if (R.cfg.batch < 1) R.cfg.batch = 1;
    if (R.cfg.cap < R.cfg.batch) R.cfg.cap = R.cfg.batch;
    if (R.cfg.window_ms < 0) R.cfg.window_ms = 0;
    asr_offline_get_facts(cfg->eng, &R.facts);
    pthread_mutex_init(&R.mu, NULL);
    pthread_cond_init(&R.cv, NULL);
    pthread_cond_init(&R.done_cv, NULL);
    if (pthread_create(&R.th, NULL, batcher_main, NULL) != 0) return -1;
    R.started = 1;
    return 0;
}

void rest_stop(void) {
    if (!R.started) return;
    pthread_mutex_lock(&R.mu);
    R.stop = 1;
    pthread_cond_broadcast(&R.cv);
    pthread_mutex_unlock(&R.mu);
    pthread_join(R.th, NULL);
    R.started = 0;
}

/* ---------------------------------------------------------------- multipart */
typedef struct {
    const uint8_t *file;
    size_t file_len;
    char language[24], response_format[24], target_language[8], model[96];
} form_data;

/* The CPU server's parser (server/main.c parse_multipart), same fields. */
static void parse_multipart(const uint8_t *body, size_t len, const char *boundary, form_data *out) {
    char sep[80];
    const size_t sep_len = (size_t)snprintf(sep, sizeof(sep), "--%s", boundary);
    const uint8_t *p = body;
    size_t remain = len;
    for (;;) {
        const uint8_t *part = mynah_asr_memmem(p, remain, (const uint8_t *)sep, sep_len);
        if (!part) break;
        part += sep_len;
        remain = len - (size_t)(part - body);
        if (remain < 4 || part[0] == '-') break;
        const uint8_t *hdr_end = mynah_asr_memmem(part, remain, (const uint8_t *)"\r\n\r\n", 4);
        if (!hdr_end) break;
        const uint8_t *data = hdr_end + 4;
        const uint8_t *next = mynah_asr_memmem(data, len - (size_t)(data - body), (const uint8_t *)sep, sep_len);
        if (!next) break;
        size_t data_len = (size_t)(next - data);
        if (data_len >= 2) data_len -= 2;
        char hdrs[512] = {0};
        size_t hl = (size_t)(hdr_end - part);
        if (hl >= sizeof(hdrs)) hl = sizeof(hdrs) - 1;
        memcpy(hdrs, part, hl);
        char name[64] = {0};
        const char *nm = strstr(hdrs, "name=\"");
        if (nm) sscanf(nm + 6, "%63[^\"]", name);
#define FIELD(fname, dst) \
        else if (strcmp(name, fname) == 0 && data_len < sizeof(dst)) { memcpy(dst, data, data_len); dst[data_len] = '\0'; }
        if (strcmp(name, "file") == 0) { out->file = data; out->file_len = data_len; }
        FIELD("model", out->model)
        FIELD("language", out->language)
        FIELD("response_format", out->response_format)
        FIELD("target_language", out->target_language)
#undef FIELD
        p = next;
        remain = len - (size_t)(p - body);
    }
}

static void query_value(const char *query, const char *key, char *out, size_t cap) {
    out[0] = '\0';
    const size_t kl = strlen(key);
    for (const char *q = query; q && *q;) {
        const char *amp = strchr(q, '&');
        const size_t seg = amp ? (size_t)(amp - q) : strlen(q);
        if (seg > kl && strncmp(q, key, kl) == 0 && q[kl] == '=') {
            const size_t vl = seg - kl - 1;
            snprintf(out, cap, "%.*s", (int)(vl < cap ? vl : cap - 1), q + kl + 1);
            return;
        }
        if (!amp) break;
        q = amp + 1;
    }
}

static void count_refusal(int bad) {
    pthread_mutex_lock(&R.mu);
    R.requests++;
    if (bad) R.refused_bad++;
    pthread_mutex_unlock(&R.mu);
}

static void bad_request(int fd, const char *code, const char *msg) {
    count_refusal(1);
    refuse_json(fd, 400, "Bad Request", "invalid_request_error", code, msg, 0);
}

/* ------------------------------------------------------------------ handler */
void rest_handle(int fd, const char *head, size_t head_len, size_t got, const char *query,
                 int translate) {
    size_t content_len = 0;
    const char *cl = strstr(head, "Content-Length:");
    if (!cl) cl = strstr(head, "content-length:");
    if (cl) content_len = strtoull(cl + 15, NULL, 10);
    if (content_len == 0 || content_len > REST_MAX_BODY) {
        bad_request(fd, "invalid_content_length", "missing or oversized Content-Length");
        return;
    }
    uint8_t *body = malloc(content_len);
    if (!body) {
        count_refusal(1);
        refuse_json(fd, 500, "Internal Server Error", "server_error", "out_of_memory", "out of memory", 0);
        return;
    }
    size_t have = got > head_len ? got - head_len : 0;
    if (have > content_len) have = content_len;
    memcpy(body, head + head_len, have);
    while (have < content_len) {
        const ssize_t r = recv(fd, body + have, content_len - have, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        have += (size_t)r;
    }
    if (have != content_len) {
        free(body);
        bad_request(fd, "incomplete_body", "the body was shorter than Content-Length");
        return;
    }

    form_data f;
    memset(&f, 0, sizeof(f));
    snprintf(f.language, sizeof(f.language), "auto");
    snprintf(f.response_format, sizeof(f.response_format), "json");
    const char *ct = strstr(head, "Content-Type:");
    if (!ct) ct = strstr(head, "content-type:");
    char boundary[72] = {0};
    if (ct) {
        const char *b = strstr(ct, "boundary=");
        if (b) sscanf(b + 9, "%71[^\r\n; ]", boundary);
    }
    if (boundary[0]) parse_multipart(body, content_len, boundary, &f);
    else { f.file = body; f.file_len = content_len; }
    if (!f.file || f.file_len < 44) {
        free(body);
        bad_request(fd, "missing_file", "missing audio file (multipart 'file' or raw WAV body)");
        return;
    }
    {   /* WHICH MODEL: this process holds exactly one */
        char want[96];
        if (f.model[0]) snprintf(want, sizeof(want), "%s", f.model);
        else query_value(query, "model", want, sizeof(want));
        if (want[0] && strcmp(want, R.facts.model_name) != 0) {
            free(body);
            count_refusal(1);
            refuse_json(fd, 404, "Not Found", "invalid_request_error", "model_not_found",
                        "this server holds one model; GET /v1/models names it", 0);
            return;
        }
    }
    if (strcmp(f.response_format, "json") != 0 && strcmp(f.response_format, "text") != 0 &&
        strcmp(f.response_format, "verbose_json") != 0) {
        free(body);
        bad_request(fd, "invalid_response_format", "response_format is one of json, text, verbose_json");
        return;
    }
    const char *tgt = NULL;
    if (translate || f.target_language[0]) {
        if (!R.facts.can_translate) {
            free(body);
            bad_request(fd, "translation_not_supported", "this model does not support translation");
            return;
        }
        tgt = f.target_language[0] ? f.target_language : "en";
    }
    char lang[24];
    if (tgt) snprintf(lang, sizeof(lang), "%.10s>%.8s", f.language, tgt);
    else snprintf(lang, sizeof(lang), "%s", f.language);
    if (!asr_offline_lang_ok(R.cfg.eng, lang)) {
        free(body);
        bad_request(fd, "language_not_supported", "the model does not serve this language (pair)");
        return;
    }

    size_t n = 0;
    int sr = 0;
    float *pcm = mynah_asr_wav_parse(f.file, f.file_len, &n, &sr);
    free(body);
    if (!pcm) { bad_request(fd, "invalid_audio", "invalid WAV (PCM16 required)"); return; }
    if (sr != R.facts.sample_rate) {
        size_t n2 = 0;
        float *rs = mynah_asr_resample(pcm, n, sr, R.facts.sample_rate, &n2);
        free(pcm);
        if (!rs) {
            count_refusal(1);
            refuse_json(fd, 500, "Internal Server Error", "server_error", "resample_failed", "resampling failed", 0);
            return;
        }
        pcm = rs;
        n = n2;
    }
    const double duration = (double)n / (double)R.facts.sample_rate;
    if (R.cfg.max_audio_seconds > 0.0 && duration > R.cfg.max_audio_seconds) {
        free(pcm);
        bad_request(fd, "audio_too_long", "the file is longer than --max-audio-seconds");
        return;
    }

    rest_job *j = calloc(1, sizeof(*j));
    if (!j) {
        free(pcm);
        count_refusal(1);
        refuse_json(fd, 500, "Internal Server Error", "server_error", "out_of_memory", "out of memory", 0);
        return;
    }
    j->pcm = pcm;
    j->n = n;
    snprintf(j->lang, sizeof(j->lang), "%s", lang);
    j->want_words = strcmp(f.response_format, "verbose_json") == 0;

    /* admission: queued + in flight <= cap, else refuse now (rule 5) */
    pthread_mutex_lock(&R.mu);
    R.requests++;
    if (R.dead || R.stop || R.queued + R.inflight >= R.cfg.cap) {
        const int dead = R.dead || R.stop;
        if (dead) R.refused_dead++; else R.refused_cap++;
        pthread_mutex_unlock(&R.mu);
        free(pcm);
        free(j);
        if (dead) refuse_json(fd, 503, "Service Unavailable", "server_error", "engine_dead",
                              "the engine failed or the server is stopping", 0);
        else refuse_json(fd, 503, "Service Unavailable", "server_error", "server_at_capacity",
                         "every batch slot is taken; retry", 1);
        return;
    }
    j->t_enq = now_s();
    if (R.tail) R.tail->next = j; else R.head = j;
    R.tail = j;
    R.queued++;
    pthread_cond_signal(&R.cv);
    while (!j->done) pthread_cond_wait(&R.done_cv, &R.mu);
    if (j->status == 0) {
        R.completed++;
        R.audio_seconds += duration;
        R.queue_ms_sum += (j->t_start - j->t_enq) * 1e3;
        R.service_ms_sum += (j->t_end - j->t_start) * 1e3;
        const double tot = (j->t_end - j->t_enq) * 1e3;
        R.total_ms_sum += tot;
        if (tot > R.total_ms_max) R.total_ms_max = tot;
    } else {
        R.failed++;
    }
    pthread_mutex_unlock(&R.mu);
    free(pcm);
    j->pcm = NULL;

    if (j->status == 2 || j->status == 3) {
        refuse_json(fd, 500, "Internal Server Error", "server_error",
                    j->status == 2 ? "internal_error" : "shutting_down",
                    j->status == 2 ? "device failure" : "the server is stopping", 0);
    } else if (j->status != 0) {
        refuse_json(fd, 400, "Bad Request", "invalid_request_error", "transcription_failed",
                    "transcription failed (unsupported language, or a segment the engine cannot hold)", 0);
    } else if (strcmp(f.response_format, "text") == 0) {
        send_text(fd, 200, "text/plain; charset=utf-8", j->text, strlen(j->text));
        linger_close(fd, NULL, 0);
    } else {
        cJSON *o = cJSON_CreateObject();
        cJSON_AddStringToObject(o, "text", j->text);
        if (j->want_words) {
            cJSON_AddStringToObject(o, "task", tgt ? "translate" : "transcribe");
            cJSON_AddStringToObject(o, "language", j->lang_out[0] ? j->lang_out : f.language);
            cJSON_AddNumberToObject(o, "duration", duration);
            if (j->words) {
                cJSON *jw = cJSON_AddArrayToObject(o, "words");
                for (int i = 0; i < j->n_words; i++) {
                    cJSON *w = cJSON_CreateObject();
                    cJSON_AddStringToObject(w, "word", j->words[i].word);
                    cJSON_AddNumberToObject(w, "start", j->words[i].t0);
                    cJSON_AddNumberToObject(w, "end", j->words[i].t1);
                    cJSON_AddItemToArray(jw, w);
                }
            }
        }
        send_json(fd, 200, o);
        cJSON_Delete(o);
        linger_close(fd, NULL, 0);
    }
    free(j->text);
    mynah_asr_words_free(j->words, j->n_words);
    free(j);
}

/* ------------------------------------------------------------ observability */
void rest_health_json(cJSON *j) {
    asr_offline_stats es;
    asr_offline_get_stats(R.cfg.eng, &es);
    asr_offline_facts f;
    asr_offline_get_facts(R.cfg.eng, &f);
    pthread_mutex_lock(&R.mu);
    cJSON_AddStringToObject(j, "status", R.dead || asr_offline_dead(R.cfg.eng) ? "dead" : "ok");
    cJSON_AddStringToObject(j, "mode", "offline");
    cJSON_AddBoolToObject(j, "streaming", 0);
    cJSON_AddNumberToObject(j, "inflight", R.inflight);
    cJSON_AddNumberToObject(j, "queued", R.queued);
    cJSON_AddNumberToObject(j, "cap", R.cfg.cap);
    cJSON *rq = cJSON_AddObjectToObject(j, "requests");
    cJSON_AddNumberToObject(rq, "total", (double)R.requests);
    cJSON_AddNumberToObject(rq, "completed", (double)R.completed);
    cJSON_AddNumberToObject(rq, "failed", (double)R.failed);
    cJSON_AddNumberToObject(rq, "refused_capacity", (double)R.refused_cap);
    cJSON_AddNumberToObject(rq, "refused_invalid", (double)R.refused_bad);
    cJSON_AddNumberToObject(rq, "refused_dead", (double)R.refused_dead);
    const unsigned long active = (unsigned long)(R.queued + R.inflight);
    cJSON_AddBoolToObject(rq, "balanced", R.requests == R.completed + R.failed + R.refused_cap +
                                                       R.refused_bad + R.refused_dead + active);
    cJSON_AddNumberToObject(j, "audio_seconds", R.audio_seconds);
    cJSON *bt = cJSON_AddObjectToObject(j, "batch");
    cJSON_AddNumberToObject(bt, "batches_total", (double)R.batches);
    cJSON_AddNumberToObject(bt, "items_mean", R.batches ? (double)R.batch_items / (double)R.batches : 0.0);
    cJSON_AddNumberToObject(bt, "max", R.cfg.batch);
    cJSON_AddNumberToObject(bt, "window_ms", R.cfg.window_ms);
    cJSON *lat = cJSON_AddObjectToObject(j, "latency_ms");
    const double c = R.completed ? (double)R.completed : 1.0;
    cJSON_AddNumberToObject(lat, "queue_mean", R.queue_ms_sum / c);
    cJSON_AddNumberToObject(lat, "service_mean", R.service_ms_sum / c);
    cJSON_AddNumberToObject(lat, "total_mean", R.total_ms_sum / c);
    cJSON_AddNumberToObject(lat, "total_max", R.total_ms_max);
    cJSON *ge = cJSON_AddObjectToObject(j, "engine");
    cJSON_AddStringToObject(ge, "name", f.name);
    cJSON_AddStringToObject(ge, "kind", "aed-offline");
    cJSON_AddStringToObject(ge, "device", f.device ? f.device : "");
    cJSON_AddStringToObject(ge, "precision", f.precision ? f.precision : "");
    cJSON_AddStringToObject(ge, "gemm", f.gemm ? f.gemm : "");
    cJSON_AddStringToObject(ge, "decoder", f.decoder ? f.decoder : "");
    cJSON_AddNumberToObject(ge, "max_items", f.max_items);
    cJSON_AddNumberToObject(ge, "segment_seconds", f.seg_sec);
    cJSON_AddNumberToObject(ge, "vram_total_bytes", (double)f.vram_total);
    cJSON_AddNumberToObject(ge, "vram_used_bytes", (double)f.vram_used);
    cJSON_AddNumberToObject(ge, "vram_weights_bytes", (double)f.vram_weights);
    cJSON_AddNumberToObject(ge, "vram_scratch_bytes", (double)f.vram_scratch);
    cJSON_AddNumberToObject(ge, "engine_wall_ms_total", es.wall_ms);
    cJSON_AddNumberToObject(ge, "failed_batches_total", (double)es.failed_batches);
    cJSON_AddNumberToObject(ge, "retried_items_total", (double)es.retried_items);
    cJSON_AddNumberToObject(ge, "enc_calls_total", (double)es.enc_calls);
    cJSON_AddNumberToObject(ge, "enc_rows_total", (double)es.enc_rows);
    cJSON_AddNumberToObject(ge, "dec_calls_total", (double)es.dec_calls);
    cJSON_AddNumberToObject(ge, "dec_steps_total", (double)es.dec_steps);
    cJSON_AddNumberToObject(ge, "dec_tokens_total", (double)es.dec_tokens);
    cJSON_AddNumberToObject(ge, "host_subsampling_ms_total", es.host_ss_ms);
    cJSON_AddNumberToObject(ge, "enc_ms_total", es.enc_ms);
    cJSON_AddNumberToObject(ge, "dec_ms_total", es.dec_ms);
    cJSON_AddNumberToObject(ge, "device_errors_total", (double)es.device_errors);
    pthread_mutex_unlock(&R.mu);
}

size_t rest_metrics_text(char *b, size_t cap) {
    asr_offline_stats es;
    asr_offline_get_stats(R.cfg.eng, &es);
    pthread_mutex_lock(&R.mu);
    size_t k = 0;
#define M(...) do { if (k < cap) k += (size_t)snprintf(b + k, cap - k, __VA_ARGS__); if (k >= cap) k = cap - 1; } while (0)
    M("# TYPE mynah_asr_offline_requests_total counter\n");
    M("mynah_asr_offline_requests_total{outcome=\"completed\"} %lu\n", R.completed);
    M("mynah_asr_offline_requests_total{outcome=\"failed\"} %lu\n", R.failed);
    M("mynah_asr_offline_requests_total{outcome=\"refused_capacity\"} %lu\n", R.refused_cap);
    M("mynah_asr_offline_requests_total{outcome=\"refused_invalid\"} %lu\n", R.refused_bad);
    M("mynah_asr_offline_requests_total{outcome=\"refused_dead\"} %lu\n", R.refused_dead);
    M("# TYPE mynah_asr_offline_inflight gauge\nmynah_asr_offline_inflight %d\n", R.inflight);
    M("# TYPE mynah_asr_offline_queued gauge\nmynah_asr_offline_queued %d\n", R.queued);
    M("# TYPE mynah_asr_offline_batches_total counter\nmynah_asr_offline_batches_total %lu\n", R.batches);
    M("# TYPE mynah_asr_offline_batch_items_total counter\nmynah_asr_offline_batch_items_total %lu\n", R.batch_items);
    M("# TYPE mynah_asr_audio_seconds_total counter\nmynah_asr_audio_seconds_total{worker=\"gpu\"} %.3f\n", R.audio_seconds);
    M("# TYPE mynah_asr_offline_latency_ms_sum counter\nmynah_asr_offline_latency_ms_sum %.1f\n", R.total_ms_sum);
    M("# TYPE mynah_asr_gpu_aed_dec_steps_total counter\nmynah_asr_gpu_aed_dec_steps_total %lu\n", es.dec_steps);
    M("# TYPE mynah_asr_gpu_aed_enc_rows_total counter\nmynah_asr_gpu_aed_enc_rows_total %lu\n", es.enc_rows);
    M("# TYPE mynah_asr_gpu_device_errors_total counter\nmynah_asr_gpu_device_errors_total %lu\n", es.device_errors);
#undef M
    pthread_mutex_unlock(&R.mu);
    return k;
}

size_t rest_dump(char *b, size_t cap, unsigned long seq) {
    asr_offline_stats es;
    asr_offline_get_stats(R.cfg.eng, &es);
    pthread_mutex_lock(&R.mu);
    const int n = snprintf(b, cap,
        "[DUMP] worker=0 seq=%lu offline model=%s engine=%s requests=%lu completed=%lu failed=%lu "
        "refused_cap=%lu refused_invalid=%lu refused_dead=%lu queued=%d inflight=%d batches=%lu "
        "items_mean=%.2f audio_s=%.1f latency_mean_ms=%.1f latency_max_ms=%.1f enc_ms=%.1f dec_ms=%.1f "
        "host_ss_ms=%.1f dec_steps=%lu\n",
        seq, R.facts.model_name, R.facts.name, R.requests, R.completed, R.failed, R.refused_cap,
        R.refused_bad, R.refused_dead, R.queued, R.inflight, R.batches,
        R.batches ? (double)R.batch_items / (double)R.batches : 0.0, R.audio_seconds,
        R.completed ? R.total_ms_sum / (double)R.completed : 0.0, R.total_ms_max, es.enc_ms,
        es.dec_ms, es.host_ss_ms, es.dec_steps);
    pthread_mutex_unlock(&R.mu);
    return n < 0 ? 0 : ((size_t)n < cap ? (size_t)n : cap - 1);
}
