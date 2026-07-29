/* Batch ≡ single: transcribes 3 fixtures in a weight-stationary batch and
 * compares against the B=1 path (they must match: the packed GEMMs produce the
 * same rows). Also measures the batch vs sequential time ratio.
 * Usage: test_batch <model_dir> <ignored_wav> <ignored_golden>
 * Exit: 0 ok, 1 mismatch, 77 skip. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/audio.h"
#include "../src/mynah_asr.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model_dir> ...\n", argv[0]); return 2; }
    char path[1024];
    snprintf(path, sizeof(path), "%s/mynah.json", argv[1]);
    FILE *probe = fopen(path, "rb");
    if (!probe) return 77;
    fclose(probe);

    const char *wavs[] = {"tests/audio/test_it.wav", "tests/audio/test_en.wav",
                          "tests/audio/test_de.wav", "tests/audio/test_fr.wav"};
    const char *langs[] = {"it-IT", "en-US", "de-DE", "fr-FR"};
    const int B = 4;

    mynah_asr_model *m = mynah_asr_load(argv[1]);
    if (!m) return 77;

    float *samples[4];
    size_t ns[4];
    for (int b = 0; b < B; b++) {
        int sr;
        samples[b] = mynah_asr_wav_load(wavs[b], &ns[b], &sr);
        if (!samples[b] || sr != 16000) { fprintf(stderr, "missing fixture %s\n", wavs[b]); return 2; }
    }

    /* reference: sequential B=1 */
    char *ref[4];
    double t0 = now_sec();
    for (int b = 0; b < B; b++)
        ref[b] = mynah_asr_transcribe(m, samples[b], ns[b], langs[b], -1, NULL);
    const double t_seq = now_sec() - t0;

    /* batch */
    char *out[4] = {0};
    t0 = now_sec();
    if (mynah_asr_transcribe_batch(m, (const float *const *)samples, ns, B, langs, -1,
                               out, NULL) != 0) {
        fprintf(stderr, "FAIL: transcribe_batch error\n");
        return 1;
    }
    const double t_batch = now_sec() - t0;

    int fails = 0;
    for (int b = 0; b < B; b++) {
        const int same = ref[b] && out[b] && strcmp(ref[b], out[b]) == 0;
        if (!same) {
            fprintf(stderr, "FAIL item %d:\n  seq:   %s\n  batch: %s\n", b,
                    ref[b] ? ref[b] : "(null)", out[b] ? out[b] : "(null)");
            fails++;
        }
        free(ref[b]); free(out[b]); free(samples[b]);
    }
    printf("batch parity: %d/%d identici | seq %.2fs vs batch %.2fs (%.2fx)\n",
           B - fails, B, t_seq, t_batch, t_seq / t_batch);
    mynah_asr_free(m);
    return fails ? 1 : 0;
}
