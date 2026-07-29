/* Streaming ≡ offline: alimenta il WAV in pezzi da ~37 ms e confronta il testo
 * finale con la trascrizione offline (stessa matematica: devono coincidere).
 * Uso: test_streaming <model_dir> <wav> <golden_dir(ignorata)>
 * Exit: 0 ok, 1 mismatch, 77 skip. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/audio.h"
#include "../src/mynah_asr.h"

static char stream_text[8192];

static void collect(const mynah_asr_result *res, void *ud) {
    (void)ud;
    strncat(stream_text, res->text, sizeof(stream_text) - strlen(stream_text) - 1);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <model_dir> <wav> <golden_dir>\n", argv[0]); return 2; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/mynah.json", argv[1]);
    FILE *probe = fopen(path, "rb");
    if (!probe) return 77;
    fclose(probe);

    mynah_asr_model *m = mynah_asr_load(argv[1]);
    if (!m) return 77;

    size_t n_samples; int sr;
    float *audio = mynah_asr_wav_load(argv[2], &n_samples, &sr);
    if (!audio || sr != 16000) return 2;

    /* offline (lookahead default) */
    char *offline = mynah_asr_transcribe(m, audio, n_samples, "it-IT", -1, NULL);
    if (!offline) return 2;

    /* streaming a pezzi di 600 campioni (37.5 ms, non allineati a nulla) */
    mynah_asr_stream *s = mynah_asr_stream_open(m, "it-IT", -1);
    if (!s) return 2;
    for (size_t off = 0; off < n_samples; off += 600) {
        const size_t n = (n_samples - off) < 600 ? (n_samples - off) : 600;
        if (mynah_asr_stream_feed(s, audio + off, n, collect, NULL) != 0) return 2;
    }
    mynah_asr_stream_finish(s, collect, NULL);
    mynah_asr_stream_close(s);

    const int same = strcmp(offline, stream_text) == 0;
    printf("streaming  offline : %s\n", offline);
    printf("streaming  chunked : %s\n", stream_text);
    printf("streaming parity: %s\n", same ? "IDENTICI OK" : "DIVERSI FAIL");

    free(offline); free(audio); mynah_asr_free(m);
    return same ? 0 : 1;
}
