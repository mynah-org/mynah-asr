/* mynah-asr — CLI. Subcommands: transcribe (offline). stream lands with M1.3. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>   /* getrusage: peak RAM in `bench` */

#include "mynah_asr.h"
#include "audio.h"
#include "backend.h"
#include "qmat.h"      /* mynah_asr_set_caps (--caps) */

static void usage(void) {
    printf("mynah-asr %s — native ASR runtime for NeMo speech models\n\n", mynah_asr_version());
    printf("Usage: mynah-asr <command> [options]\n\n");
    printf("Commands:\n");
    printf("  transcribe -m <model_dir> -i <file.wav> [--lang auto] [--lookahead N] [--quant int8]\n");
    printf("             [--target-lang xx]   (AED/Canary: translation, e.g. --lang en --target-lang de)\n");
    printf("             [--timestamps] [--decoder default|ctc]\n");
    printf("             offline transcription (WAV PCM16 16 kHz);\n");
    printf("             --timestamps prints one word per line: t0 t1 word\n");
    printf("             --decoder ctc uses the CTC head of hybrid models (tdt_ctc)\n");
    printf("  stream     -m <model_dir> [--lang auto] [--quant int8|int4]\n");
    printf("             live streaming from stdin (raw s16le 16 kHz mono)\n");
    printf("  quantize   -m <model_dir> --quant int8|int4\n");
    printf("             writes the pre-quantized checkpoint (instant load)\n");
    printf("  bench      -m <model_dir> [-i file.wav] [--runs N] [--warmup W] [--quant int8]\n");
    printf("             warm RTF (min/median over N runs) + peak RAM, one model at a time\n");
    printf("  --version                                 prints the version\n\n");
    printf("Common options (transcribe/stream):\n");
    printf("  --backend cpu|metal|cuda    GEMM backend (falls back to CPU if unavailable)\n");
    printf("  --caps auto|scalar|avx2|vnni  x86 SIMD level (default: cpuid; env MYNAH_ASR_CAPS)\n");
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmd_transcribe(int argc, char **argv) {
    const char *model_dir = NULL, *wav = NULL, *lang = "auto", *decoder = "default";
    const char *target_lang = NULL;
    int lookahead = -1, quant = MYNAH_ASR_QUANT_F32, timestamps = 0;
    double segment_sec = 0.0;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) wav = argv[++i];
        else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) lang = argv[++i];
        else if (strcmp(argv[i], "--target-lang") == 0 && i + 1 < argc) target_lang = argv[++i];
        else if (strcmp(argv[i], "--timestamps") == 0) timestamps = 1;
        else if (strcmp(argv[i], "--decoder") == 0 && i + 1 < argc) decoder = argv[++i];
        else if (strcmp(argv[i], "--segment-sec") == 0 && i + 1 < argc) segment_sec = atof(argv[++i]);
        else if (strcmp(argv[i], "--lookahead") == 0 && i + 1 < argc) lookahead = atoi(argv[++i]);
        else if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            i++;
            quant = strcmp(argv[i], "int8") == 0 ? MYNAH_ASR_QUANT_INT8
                  : strcmp(argv[i], "int4") == 0 ? MYNAH_ASR_QUANT_INT4 : MYNAH_ASR_QUANT_F32;
        }
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) mynah_asr_set_backend(argv[++i]);
        else if (strcmp(argv[i], "--caps") == 0 && i + 1 < argc) mynah_asr_set_caps(argv[++i]);
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
    }
    if (!model_dir || !wav) { usage(); return 2; }

    double t0 = now_sec();
    mynah_asr_model *m = mynah_asr_load_quant(model_dir, quant);
    if (!m) return 1;
    if (mynah_asr_set_decoder(m, decoder) != 0) { mynah_asr_free(m); return 1; }
    if (target_lang && mynah_asr_set_target_lang(m, target_lang) != 0) { mynah_asr_free(m); return 1; }
    if (segment_sec > 0.0) mynah_asr_set_segment_limit(m, segment_sec);
    double t_load = now_sec() - t0;

    size_t n_samples;
    int sr;
    float *samples = mynah_asr_wav_load(wav, &n_samples, &sr);
    if (!samples) { mynah_asr_free(m); return 1; }
    if (sr != 16000) {
        fprintf(stderr, "[resample %d Hz -> 16000 Hz]\n", sr);
        size_t n_rs;
        float *rs = mynah_asr_resample(samples, n_samples, sr, 16000, &n_rs);
        free(samples);
        if (!rs) { mynah_asr_free(m); return 1; }
        samples = rs;
        n_samples = n_rs;
    }

    char lang_out[16];
    mynah_asr_word *words = NULL;
    int n_words = 0;
    t0 = now_sec();
    char *text = mynah_asr_transcribe_ts(m, samples, n_samples, lang, lookahead, lang_out,
                                     timestamps ? &words : NULL, &n_words);
    double t_run = now_sec() - t0;
    if (!text) { free(samples); mynah_asr_free(m); return 1; }

    const double dur = (double)n_samples / 16000.0;
    fprintf(stderr, "[%.1fs audio | load %.2fs | inference %.2fs | RTF %.3f | lang=%s]\n",
            dur, t_load, t_run, t_run / dur, lang_out[0] ? lang_out : lang);
    if (timestamps) {
        for (int i = 0; i < n_words; i++)
            printf("%6.2f %6.2f  %s\n", words[i].t0, words[i].t1, words[i].word);
        mynah_asr_words_free(words, n_words);
    } else {
        printf("%s\n", text);
    }

    free(text); free(samples); mynah_asr_free(m);
    return 0;
}

static void print_partial(const mynah_asr_result *res, void *ud) {
    (void)ud;
    fputs(res->text, stdout);
    fflush(stdout);
}

static int cmd_stream(int argc, char **argv) {
    const char *model_dir = NULL, *lang = "auto";
    int lookahead = -1, quant = MYNAH_ASR_QUANT_F32;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) lang = argv[++i];
        else if (strcmp(argv[i], "--lookahead") == 0 && i + 1 < argc) lookahead = atoi(argv[++i]);
        else if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            i++;
            quant = strcmp(argv[i], "int8") == 0 ? MYNAH_ASR_QUANT_INT8
                  : strcmp(argv[i], "int4") == 0 ? MYNAH_ASR_QUANT_INT4 : MYNAH_ASR_QUANT_F32;
        }
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) mynah_asr_set_backend(argv[++i]);
        else if (strcmp(argv[i], "--caps") == 0 && i + 1 < argc) mynah_asr_set_caps(argv[++i]);
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
    }
    if (!model_dir) { usage(); return 2; }

    mynah_asr_model *m = mynah_asr_load_quant(model_dir, quant);
    if (!m) return 1;
    mynah_asr_stream *s = mynah_asr_stream_open(m, lang, lookahead);
    if (!s) { mynah_asr_free(m); return 1; }
    fprintf(stderr, "[stream: raw s16le 16 kHz mono from stdin, lang=%s]\n", lang);

    short pcm[1600]; /* 100 ms per read */
    float buf[1600];
    size_t got;
    while ((got = fread(pcm, sizeof(short), 1600, stdin)) > 0) {
        for (size_t i = 0; i < got; i++) buf[i] = (float)pcm[i] / 32768.0f;
        if (mynah_asr_stream_feed(s, buf, got, print_partial, NULL) != 0) break;
    }
    mynah_asr_stream_finish(s, print_partial, NULL);
    printf("\n");
    if (mynah_asr_stream_lang(s)[0]) fprintf(stderr, "[lang=%s]\n", mynah_asr_stream_lang(s));

    mynah_asr_stream_close(s);
    mynah_asr_free(m);
    return 0;
}

/* ------------------------------------------------------------------ quantize */
#include "qmat.h"
#include "stwrite.h"
#include "weights.h"

/* allowlist: the same linears the runtime consumes through qmat (2D or [n,k,1]) */
static int quantizable(const char *name, const mynah_asr_tensor *t) {
    if (strstr(name, "relative_k_proj")) return 0;
    const int is_2dish = t->n_dims == 2 || (t->n_dims == 3 && t->shape[2] == 1);
    if (!is_2dish || t->dtype != MYNAH_ASR_DT_F32) return 0;
    if (strstr(name, "feed_forward") && strstr(name, ".linear")) return 1;
    if (strstr(name, "self_attn.") && strstr(name, "_proj.weight")) return 1;
    if (strstr(name, "conv.pointwise_conv") && !strstr(name, "subsampling")) return 1;
    if (strcmp(name, "joint.head.weight") == 0) return 1;
    /* AED decoder (Canary): self/cross attention, FFN, head, encoder projection */
    if (strstr(name, "cross_attn.") && strstr(name, "_proj.weight")) return 1;
    if (strstr(name, "aed.") && strstr(name, "ffn.linear")) return 1;
    if (strcmp(name, "aed.head.weight") == 0) return 1;
    if (strcmp(name, "enc_dec_proj.weight") == 0) return 1;
    return 0;
}

static int cmd_quantize(int argc, char **argv) {
    const char *model_dir = NULL, *qs = "int8";
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) qs = argv[++i];
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
    }
    const int qtype = strcmp(qs, "int4") == 0 ? MYNAH_ASR_Q_INT4
                    : strcmp(qs, "int8") == 0 ? MYNAH_ASR_Q_INT8 : MYNAH_ASR_Q_F32;
    if (!model_dir || qtype == MYNAH_ASR_Q_F32) {
        fprintf(stderr, "usage: mynah-asr quantize -m <model_dir> --quant int8|int4\n");
        return 2;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    mynah_asr_safetensors *st = mynah_asr_st_open_quiet(path);
    if (!st) {   /* container alternativo GGUF (stesso fallback di mynah_asr_load) */
        snprintf(path, sizeof(path), "%s/model.gguf", model_dir);
        st = mynah_asr_st_open_quiet(path);
    }
    if (!st) {
        fprintf(stderr, "quantize: neither model.safetensors nor model.gguf in %s\n", model_dir);
        return 1;
    }

    mynah_asr_stw *w = mynah_asr_stw_new();
    void **to_free = calloc(2 * mynah_asr_st_count(st), sizeof(void *));
    size_t n_free = 0, n_quant = 0;
    double t0 = now_sec();

    for (size_t idx = 0; idx < mynah_asr_st_count(st); idx++) {
        const mynah_asr_tensor *t = mynah_asr_st_at(st, idx);
        if (t->dtype == MYNAH_ASR_DT_I64) continue; /* BN num_batches_tracked: inutile a runtime */
        if (!quantizable(t->name, t)) {
            const size_t bytes = t->n_elems * 4u; /* solo F32 nel file sorgente */
            mynah_asr_stw_add(w, t->name, "F32", t->shape, t->n_dims, t->data, bytes);
            continue;
        }
        const int n = (int)t->shape[0], k = (int)(t->n_elems / (size_t)t->shape[0]);
        char qname[224];
        if (qtype == MYNAH_ASR_Q_INT8) {
            int8_t *q = malloc((size_t)n * (size_t)k);
            float *s = malloc((size_t)n * sizeof(float));
            mynah_asr_quantize_int8((const float *)t->data, n, k, q, s);
            int64_t qshape[2] = {n, k}, sshape[1] = {n};
            snprintf(qname, sizeof(qname), "%s.q8", t->name);
            mynah_asr_stw_add(w, qname, "I8", qshape, 2, q, (size_t)n * (size_t)k);
            snprintf(qname, sizeof(qname), "%s.scales", t->name);
            mynah_asr_stw_add(w, qname, "F32", sshape, 1, s, (size_t)n * 4u);
            to_free[n_free++] = q;
            to_free[n_free++] = s;
        } else {
            const int groups = k / MYNAH_ASR_Q4_GROUP;
            uint8_t *q = malloc((size_t)n * (size_t)k / 2);
            float *s = malloc((size_t)n * (size_t)groups * sizeof(float));
            mynah_asr_quantize_int4((const float *)t->data, n, k, q, s);
            int64_t qshape[2] = {n, k / 2}, sshape[2] = {n, groups};
            snprintf(qname, sizeof(qname), "%s.q4", t->name);
            mynah_asr_stw_add(w, qname, "U8", qshape, 2, q, (size_t)n * (size_t)k / 2);
            snprintf(qname, sizeof(qname), "%s.scales", t->name);
            mynah_asr_stw_add(w, qname, "F32", sshape, 2, s, (size_t)n * (size_t)groups * 4u);
            to_free[n_free++] = q;
            to_free[n_free++] = s;
        }
        n_quant++;
    }

    snprintf(path, sizeof(path), "%s/model.%s.safetensors", model_dir, qs);
    const int rc = mynah_asr_stw_write(w, path);
    const double dt = now_sec() - t0;

    for (size_t i = 0; i < n_free; i++) free(to_free[i]);
    free(to_free);
    mynah_asr_stw_free(w);
    mynah_asr_st_close(st);

    if (rc != 0) { fprintf(stderr, "quantize: write failed\n"); return 1; }
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    const double gb = (double)ftell(f) / 1e9;
    fclose(f);
    printf("OK %s: %zu tensors quantized to %s, %.2f GB (%.1fs)\n", path, n_quant, qs, gb, dt);
    printf("Usage: mynah-asr transcribe|stream --quant %s (or mynah-asr-server --quant %s)\n", qs, qs);
    return 0;
}

/* Peak process RSS in GB (getrusage: bytes on macOS, KB on Linux). */
static double peak_ram_gb(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#ifdef __APPLE__
    return (double)ru.ru_maxrss / 1073741824.0;   /* byte */
#else
    return (double)ru.ru_maxrss / 1048576.0;       /* KB   */
#endif
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* bench: warm RTF (min/median over N timed runs, after W warm-ups) + peak RAM.
 * Same metric as `make bench` (tests/bench.sh) but in-process, one model at a
 * time and with explicit A/B parameters (repo rule 6). */
static int cmd_bench(int argc, char **argv) {
    const char *model_dir = NULL, *wav = "tests/audio/test_it.wav", *lang = "auto";
    int lookahead = -1, quant = MYNAH_ASR_QUANT_F32, runs = 5, warmup = 2;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) wav = argv[++i];
        else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) lang = argv[++i];
        else if (strcmp(argv[i], "--lookahead") == 0 && i + 1 < argc) lookahead = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            i++;
            quant = strcmp(argv[i], "int8") == 0 ? MYNAH_ASR_QUANT_INT8
                  : strcmp(argv[i], "int4") == 0 ? MYNAH_ASR_QUANT_INT4 : MYNAH_ASR_QUANT_F32;
        }
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) mynah_asr_set_backend(argv[++i]);
        else if (strcmp(argv[i], "--caps") == 0 && i + 1 < argc) mynah_asr_set_caps(argv[++i]);
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
    }
    if (!model_dir) { usage(); return 2; }
    if (runs < 1) runs = 1;
    if (warmup < 0) warmup = 0;

    double t0 = now_sec();
    mynah_asr_model *m = mynah_asr_load_quant(model_dir, quant);
    if (!m) return 1;
    const double t_load = now_sec() - t0;

    size_t n_samples;
    int sr;
    float *samples = mynah_asr_wav_load(wav, &n_samples, &sr);
    if (!samples) { mynah_asr_free(m); return 1; }
    if (sr != 16000) {
        size_t n_rs;
        float *rs = mynah_asr_resample(samples, n_samples, sr, 16000, &n_rs);
        free(samples);
        if (!rs) { mynah_asr_free(m); return 1; }
        samples = rs; n_samples = n_rs;
    }
    const double dur = (double)n_samples / 16000.0;

    fprintf(stderr, "[bench] model=%s quant=%s lookahead=%d | audio %.1fs | load %.2fs | warmup %d runs %d\n",
            model_dir, quant == MYNAH_ASR_QUANT_INT8 ? "int8" : quant == MYNAH_ASR_QUANT_INT4 ? "int4" : "f32",
            lookahead, dur, t_load, warmup, runs);

    char lang_out[16] = {0};
    for (int w = 0; w < warmup; w++) {
        char *t = mynah_asr_transcribe(m, samples, n_samples, lang, lookahead, lang_out);
        if (!t) { free(samples); mynah_asr_free(m); return 1; }
        free(t);
    }

    double *rtf = malloc((size_t)runs * sizeof *rtf);
    if (!rtf) { free(samples); mynah_asr_free(m); return 1; }
    for (int r = 0; r < runs; r++) {
        t0 = now_sec();
        char *t = mynah_asr_transcribe(m, samples, n_samples, lang, lookahead, lang_out);
        const double dt = now_sec() - t0;
        if (!t) { free(rtf); free(samples); mynah_asr_free(m); return 1; }
        free(t);
        rtf[r] = dt / dur;
        fprintf(stderr, "  run %2d  %.3fs  RTF %.3f\n", r + 1, dt, rtf[r]);
    }

    qsort(rtf, (size_t)runs, sizeof *rtf, cmp_double);
    const double median = runs & 1 ? rtf[runs / 2] : (rtf[runs / 2 - 1] + rtf[runs / 2]) / 2.0;
    printf("[bench] RTF warm: min %.3f  median %.3f  | peak RAM %.2f GB | lang=%s\n",
           rtf[0], median, peak_ram_gb(), lang_out[0] ? lang_out : lang);

    free(rtf); free(samples); mynah_asr_free(m);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--version") == 0) {
#ifdef MYNAH_ASR_BUILD
        printf("%s (%s)\n", mynah_asr_version(), MYNAH_ASR_BUILD);
#else
        printf("%s\n", mynah_asr_version());
#endif
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "transcribe") == 0) return cmd_transcribe(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "stream") == 0) return cmd_stream(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "quantize") == 0) return cmd_quantize(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) return cmd_bench(argc - 2, argv + 2);
    usage();
    return argc < 2 ? 0 : 1;
}
