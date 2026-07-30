/* Silero VAD parity: src/vad.c against onnxruntime running the real ONNX
 * (tools/oracle/vad.py). Driven by tests/test_vad.sh, which produces the dump.
 *
 * Compared over the WHOLE frame sequence, not one frame: the LSTM state is
 * carried, so a wrong state update shows up as drift and only a full run catches
 * it. The fixture is also checked to actually swing between speech and silence —
 * a constant-output bug must not pass because the audio never changes.
 *
 * Exit: 0 ok, 1 mismatch, 77 skip (VAD model not converted). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/audio.h"
#include "../src/vad.h"

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("vad FAIL: %s\n", msg); failures = 1; } \
    else printf("vad ok:   %s\n", msg); } while (0)

double *npy_load_f(const char *path, size_t *n_elems);   /* shared from npy.c */

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <vad_dir> <file.wav> <probs.npy>\n", argv[0]);
        return 1;
    }
    const char *vad_dir = argv[1], *wav_path = argv[2], *npy_path = argv[3];

    char cfg[1024];
    snprintf(cfg, sizeof(cfg), "%s/mynah.json", vad_dir);
    FILE *probe = fopen(cfg, "rb");
    if (!probe) return 77;                     /* not converted: skip, like every model-gated test */
    fclose(probe);

    size_t n_ref = 0;
    double *ref = npy_load_f(npy_path, &n_ref);
    if (!ref) { fprintf(stderr, "vad: cannot read %s\n", npy_path); return 1; }

    size_t n_samples = 0;
    int sr = 0;
    float *audio = mynah_asr_wav_load(wav_path, &n_samples, &sr);
    if (!audio) { free(ref); fprintf(stderr, "vad: cannot read %s\n", wav_path); return 1; }
    if (sr != 16000) { free(ref); free(audio); fprintf(stderr, "vad: %s is not 16 kHz\n", wav_path); return 1; }

    mynah_asr_vad *v = mynah_asr_vad_open(vad_dir);
    CHECK(v != NULL, "VAD model loaded");
    if (!v) { free(ref); free(audio); return 1; }

    const int F = mynah_asr_vad_frame_samples(v);
    CHECK(F > 0, "frame size known");
    const size_t n_frames = (n_samples + (size_t)F - 1) / (size_t)F;   /* zero-padded tail */
    char msg[256];
    snprintf(msg, sizeof(msg), "%zu frames from %zu samples, oracle has %zu",
             n_frames, n_samples, n_ref);
    CHECK(n_frames == n_ref, msg);

    /* the fixture must exercise both decisions, or a constant output would pass */
    double rmin = 1.0, rmax = 0.0;
    for (size_t i = 0; i < n_ref; i++) {
        if (ref[i] < rmin) rmin = ref[i];
        if (ref[i] > rmax) rmax = ref[i];
    }
    snprintf(msg, sizeof(msg), "fixture swings speech/silence (oracle min %.3f, max %.3f)", rmin, rmax);
    CHECK(rmin < 0.1 && rmax > 0.9, msg);

    double worst = 0.0;
    size_t worst_i = 0, n_cmp = n_frames < n_ref ? n_frames : n_ref;
    for (size_t i = 0; i < n_cmp; i++) {
        const size_t off = i * (size_t)F;
        const size_t n = n_samples - off < (size_t)F ? n_samples - off : (size_t)F;
        const float p = mynah_asr_vad_feed(v, audio + off, n);
        if (p < 0.0f) { snprintf(msg, sizeof(msg), "feed failed at frame %zu", i); CHECK(0, msg); break; }
        const double e = fabs((double)p - ref[i]);
        if (e > worst) { worst = e; worst_i = i; }
    }
    snprintf(msg, sizeof(msg), "%zu frames vs onnxruntime (worst abs err %.2e at frame %zu)",
             n_cmp, worst, worst_i);
    CHECK(worst <= 1e-5, msg);

    /* reset must return the module to its initial state: the first frame decoded
     * again has to give the exact same probability as at the start */
    mynah_asr_vad_reset(v);
    const float again = mynah_asr_vad_feed(v, audio, n_samples < (size_t)F ? n_samples : (size_t)F);
    snprintf(msg, sizeof(msg), "reset restores frame 0 (%.6f vs oracle %.6f)", again, ref[0]);
    CHECK(fabs((double)again - ref[0]) <= 1e-5, msg);

    mynah_asr_vad_close(v);
    free(ref);
    free(audio);
    return failures;
}
