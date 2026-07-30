/* Batch ≡ single: transcribes 4 fixtures in a weight-stationary batch and
 * compares against the B=1 path (they must match: the packed GEMMs produce the
 * same rows). Also measures the batch vs sequential time ratio.
 *
 * Phase 2 covers audio LONGER than the offline segment limit, mixed with short
 * items. That case used to diverge: the single path split long audio on silence
 * (full-attention models collapse past ~30 s) while the batch path encoded the
 * item whole, so the same file gave a different — and worse — answer depending on
 * which entry point you called. It also exercises the wave boundary and the
 * per-item stitching, since one long item contributes several segments.
 *
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
    printf("batch parity: %d/%d identical | seq %.2fs vs batch %.2fs (%.2fx)\n",
           B - fails, B, t_seq, t_batch, t_seq / t_batch);

    /* ---- phase 2: an item longer than the segment limit, mixed with short ones ---- */
    const char *long_wav = "samples/en/fleurs_long.wav";   /* ~95 s */
    size_t ln = 0;
    int lsr = 0;
    float *lsamples = mynah_asr_wav_load(long_wav, &ln, &lsr);
    if (!lsamples || lsr != 16000) {
        printf("batch long-item SKIP (%s missing)\n", long_wav);
        free(lsamples);
        mynah_asr_free(m);
        return fails ? 1 : 0;
    }

    /* Force a small limit instead of trusting the model's default: Nemotron's is
     * 300 s, so a 95 s file would be ONE segment and this phase would be a green
     * gate over the very path it exists to protect. At 5 s the same file yields
     * ~19 segments on every model. */
    const double saved_limit = mynah_asr_segment_limit(m);
    mynah_asr_set_segment_limit(m, 5.0);
    const double limit = mynah_asr_segment_limit(m);
    const double dur = (double)ln / 16000.0;
    if (dur <= limit * 1.1) {   /* +10%: same "don't split a small overrun" rule */
        fprintf(stderr, "FAIL: %s is %.0f s, not longer than the %.0f s limit — "
                        "phase 2 would not segment anything\n", long_wav, dur, limit);
        return 1;
    }

    float *s2[3];
    size_t n2[3];
    const char *l2[3] = {"en-US", "en-US", "en-US"};
    int sr2 = 0;
    s2[0] = lsamples;                                  n2[0] = ln;
    s2[1] = mynah_asr_wav_load("tests/audio/test_en.wav", &n2[1], &sr2);
    s2[2] = mynah_asr_wav_load(long_wav, &n2[2], &sr2);  /* twice: crosses the wave edge */
    if (!s2[1] || !s2[2]) { fprintf(stderr, "FAIL: fixtures for phase 2\n"); return 1; }

    char *ref2[3], *out2[3] = {0};
    for (int b = 0; b < 3; b++)
        ref2[b] = mynah_asr_transcribe(m, s2[b], n2[b], l2[b], -1, NULL);
    if (mynah_asr_transcribe_batch(m, (const float *const *)s2, n2, 3, l2, -1, out2, NULL) != 0) {
        fprintf(stderr, "FAIL: transcribe_batch error (phase 2)\n");
        return 1;
    }
    for (int b = 0; b < 3; b++) {
        const int same = ref2[b] && out2[b] && strcmp(ref2[b], out2[b]) == 0;
        if (!same) {
            fprintf(stderr, "FAIL long-item %d (%.0f s):\n  seq:   %s\n  batch: %s\n",
                    b, (double)n2[b] / 16000.0, ref2[b] ? ref2[b] : "(null)",
                    out2[b] ? out2[b] : "(null)");
            fails++;
        }
        free(ref2[b]);
        free(out2[b]);
    }
    free(s2[0]); free(s2[1]); free(s2[2]);
    /* segments per long item, so the log shows the path was really exercised */
    printf("batch long-item parity: %s (%.0f s item / %.0f s limit = ~%d segments,"
           " batch of 3 crosses the wave edge)\n",
           fails ? "FAIL" : "OK", dur, limit, (int)(dur / limit) + 1);
    mynah_asr_set_segment_limit(m, saved_limit);

    mynah_asr_free(m);
    return fails ? 1 : 0;
}
