/* Streaming ≡ offline: feeds the WAV in ~37 ms pieces and compares the final
 * text with the offline transcription (same math: they must match).
 * Usage: test_streaming <model_dir> <wav> <golden_dir(ignored)>
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

    /* offline (default lookahead) */
    char *offline = mynah_asr_transcribe(m, audio, n_samples, "it-IT", -1, NULL);
    if (!offline) return 2;

    /* streaming in 600-sample pieces (37.5 ms, deliberately unaligned) */
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
    printf("streaming parity: %s\n", same ? "IDENTICAL OK" : "DIFFERENT FAIL");
    int fail = !same;

    /* reset ≡ reopen: pollute a stream with the second half of the audio, reset
     * it, then feed the whole file paced by need_samples (one chunk per feed) */
    s = mynah_asr_stream_open(m, "it-IT", -1);
    if (!s) return 2;
    if (mynah_asr_stream_feed(s, audio + n_samples / 2, n_samples - n_samples / 2, NULL, NULL) != 0) return 2;
    if (mynah_asr_stream_reset(s, NULL) != 0) return 2;
    stream_text[0] = '\0';
    size_t fed = 0, feeds = 0, flushes = 0;
    while (fed < n_samples) {
        size_t need = mynah_asr_stream_need_samples(s);
        if (need == 0) { printf("streaming need_samples: 0 after a feed FAIL\n"); fail = 1; break; }
        if (need > n_samples - fed) need = n_samples - fed;
        const size_t before = strlen(stream_text);
        if (mynah_asr_stream_feed(s, audio + fed, need, collect, NULL) != 0) return 2;
        if (strlen(stream_text) != before) flushes++;
        fed += need;
        feeds++;
    }
    mynah_asr_stream_finish(s, collect, NULL);
    const int same2 = strcmp(offline, stream_text) == 0;
    printf("streaming reset+paced (%zu feeds, %zu with text): %s\n", feeds, flushes,
           same2 ? "IDENTICAL OK" : "DIFFERENT FAIL");
    if (!same2) { printf("  got: %s\n", stream_text); fail = 1; }
    /* a wrong language is refused and leaves the stream usable */
    if (mynah_asr_stream_reset(s, "xx-YY") != -1) { printf("streaming reset bad lang: accepted FAIL\n"); fail = 1; }
    mynah_asr_stream_close(s);

    free(offline); free(audio); mynah_asr_free(m);
    return fail;
}
