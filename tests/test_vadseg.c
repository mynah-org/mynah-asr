/* VAD speech segmentation (the hysteresis in src/vad.c) with synthetic
 * probabilities — no checkpoint, so it runs everywhere including CI.
 *
 * Probabilities are not decisions, and the decision logic is where the off-by-one
 * bugs live: when a span opens, when it closes (min_silence_ms AFTER the speech
 * stopped, not when it stopped), which spans are dropped for being too short, and
 * how padding widens them. Every expectation below is computed by hand from the
 * policy, in samples, so a wrong boundary shows up as a wrong number rather than
 * as "still plausible".
 *
 * Exit: 0 ok, 1 fail. */
#include <stdio.h>
#include <string.h>

#include "../src/vad.h"

static int failures;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("vadseg FAIL: %s\n", msg); failures = 1; } \
    else printf("vadseg ok:   %s\n", msg); } while (0)

#define FRAME 512
#define SR 16000

/* the defaults tools/convert_silero.py writes into mynah.json */
static mynah_asr_vad_policy policy(void) {
    mynah_asr_vad_policy p;
    p.sample_rate = SR;
    p.frame_samples = FRAME;
    p.threshold = 0.5;
    p.neg_threshold = 0.35;
    p.min_speech_ms = 250;      /* 4000 samples */
    p.min_silence_ms = 100;     /* 1600 samples */
    p.speech_pad_ms = 30;       /*  480 samples */
    return p;
}

/* Runs a probability sequence through the stepper; returns the span count. */
static int run(const mynah_asr_vad_policy *p, const float *probs, int n,
               size_t n_samples, mynah_asr_vad_span *out, int cap, int pad) {
    mynah_asr_vad_seg seg;
    mynah_asr_vad_seg_reset(&seg);
    int k = 0;
    for (int i = 0; i < n; i++) {
        mynah_asr_vad_span s;
        if (mynah_asr_vad_seg_feed(p, &seg, probs[i], &s) && k < cap) out[k++] = s;
    }
    mynah_asr_vad_span s;
    if (mynah_asr_vad_seg_finish(p, &seg, n_samples, &s) && k < cap) out[k++] = s;
    if (pad) mynah_asr_vad_pad_spans(p, out, k, n_samples);
    return k;
}

static void fill(float *probs, int n, float v) { for (int i = 0; i < n; i++) probs[i] = v; }

static int span_is(const char *what, const mynah_asr_vad_span *s, size_t a, size_t b) {
    char msg[192];
    snprintf(msg, sizeof(msg), "%s: [%zu, %zu) (got [%zu, %zu))", what, a, b, s->t0, s->t1);
    const int ok = s->t0 == a && s->t1 == b;
    CHECK(ok, msg);
    return ok;
}

int main(void) {
    const mynah_asr_vad_policy p = policy();
    char msg[192];
    float probs[64];
    mynah_asr_vad_span sp[8];

    /* ---- 1. one span, unpadded boundaries ------------------------------
     * speech on frames 2..11 (start = 2*512 = 1024, 10 frames = 5120 > 4000),
     * silence from frame 12 (temp_end = 6144); the close needs 1600 samples of
     * silence, so it fires at the first frame with 512*i - 6144 >= 1600, i.e.
     * i = 16 (8192 - 6144 = 2048). The span END is temp_end, not frame 16. */
    fill(probs, 24, 0.0f);
    for (int i = 2; i <= 11; i++) probs[i] = 0.9f;
    {
        const int n = run(&p, probs, 24, 24 * FRAME, sp, 8, 0);
        snprintf(msg, sizeof(msg), "one span found (%d)", n);
        CHECK(n == 1, msg);
        if (n == 1) span_is("unpadded", &sp[0], 1024, 6144);
    }

    /* ---- 2. the same run, padded (silero widens by 480 both sides) ------ */
    {
        const int n = run(&p, probs, 24, 24 * FRAME, sp, 8, 1);
        if (n == 1) span_is("padded", &sp[0], 1024 - 480, 6144 + 480);
    }

    /* ---- 3. speech shorter than min_speech_ms is dropped ---------------
     * frames 2..8: end = 9*512 = 4608, start = 1024, length 3584 < 4000 */
    fill(probs, 24, 0.0f);
    for (int i = 2; i <= 8; i++) probs[i] = 0.9f;
    {
        const int n = run(&p, probs, 24, 24 * FRAME, sp, 8, 1);
        snprintf(msg, sizeof(msg), "3584-sample span dropped (%d spans)", n);
        CHECK(n == 0, msg);
    }

    /* ---- 4. a short dip below neg_threshold does not split a span -------
     * frames 12-13 silent = 1024 samples < 1600, so the span survives and ends
     * only after the real silence at frame 20 (temp_end = 10240) */
    fill(probs, 30, 0.0f);
    for (int i = 2; i <= 19; i++) probs[i] = 0.9f;
    probs[12] = probs[13] = 0.05f;
    {
        const int n = run(&p, probs, 30, 30 * FRAME, sp, 8, 0);
        snprintf(msg, sizeof(msg), "a 2-frame dip keeps one span (%d)", n);
        CHECK(n == 1, msg);
        if (n == 1) span_is("dip", &sp[0], 1024, 10240);
    }

    /* ---- 5. the dead zone between the thresholds holds the state -------
     * 0.4 is below threshold but above neg_threshold: it neither opens nor
     * closes. A span already open must stay open through it forever. */
    fill(probs, 30, 0.4f);
    for (int i = 2; i <= 6; i++) probs[i] = 0.9f;
    {
        const int n = run(&p, probs, 30, 30 * FRAME, sp, 8, 0);
        snprintf(msg, sizeof(msg), "0.4 never closes a span (%d)", n);
        CHECK(n == 1, msg);
        if (n == 1) span_is("dead zone", &sp[0], 1024, 30 * FRAME);   /* closed by finish */
    }
    fill(probs, 30, 0.4f);
    {
        const int n = run(&p, probs, 30, 30 * FRAME, sp, 8, 0);
        snprintf(msg, sizeof(msg), "0.4 never opens one either (%d)", n);
        CHECK(n == 0, msg);
    }

    /* ---- 6. two spans, and the padding does not make them overlap ------ */
    fill(probs, 40, 0.0f);
    for (int i = 2; i <= 11; i++) probs[i] = 0.9f;      /* [1024, 6144) */
    for (int i = 22; i <= 31; i++) probs[i] = 0.9f;     /* [11264, 16384) */
    {
        const int n = run(&p, probs, 40, 40 * FRAME, sp, 8, 1);
        snprintf(msg, sizeof(msg), "two spans (%d)", n);
        CHECK(n == 2, msg);
        if (n == 2) {
            span_is("pair[0]", &sp[0], 1024 - 480, 6144 + 480);
            span_is("pair[1]", &sp[1], 11264 - 480, 16384 + 480);
            CHECK(sp[0].t1 < sp[1].t0, "padded spans stay disjoint");
        }
    }

    /* ---- 7. padding wider than half the gap: split the difference ------
     * With the default 30 ms this branch is unreachable — a close needs 1600
     * samples of silence, always more than 2*480 — so it is only exercised with
     * a larger pad. 100 ms pad = 1600, gap = 11264 - 6144 = 5120 ... still too
     * big; 200 ms pad = 3200 gives 2*pad = 6400 > 5120, so both spans move by
     * 5120/2 = 2560. */
    {
        mynah_asr_vad_policy wide = p;
        wide.speech_pad_ms = 200;
        const int n = run(&wide, probs, 40, 40 * FRAME, sp, 8, 1);
        if (n == 2) {
            span_is("shared[0]", &sp[0], 0, 6144 + 2560);      /* 1024 - 3200 clamps to 0 */
            span_is("shared[1]", &sp[1], 11264 - 2560, 16384 + 3200);
            CHECK(sp[0].t1 == sp[1].t0, "the shared gap leaves no hole");
        }
    }

    /* ---- 8. clamping at both ends of the audio -------------------------
     * speech from frame 0 to the last frame: padding must not go below 0 nor
     * past n_samples. */
    fill(probs, 20, 0.9f);
    {
        const int n = run(&p, probs, 20, 20 * FRAME, sp, 8, 1);
        if (n == 1) span_is("full-length", &sp[0], 0, 20 * FRAME);
    }

    /* ---- 9. finish() drops a too-short tail ---------------------------- */
    fill(probs, 20, 0.0f);
    for (int i = 17; i <= 19; i++) probs[i] = 0.9f;     /* start 8704, tail 1536 */
    {
        const int n = run(&p, probs, 20, 20 * FRAME, sp, 8, 1);
        snprintf(msg, sizeof(msg), "1536-sample tail dropped (%d)", n);
        CHECK(n == 0, msg);
    }

    /* ---- 10. reset really resets -------------------------------------- */
    {
        mynah_asr_vad_seg seg;
        mynah_asr_vad_span s;
        fill(probs, 20, 0.9f);
        mynah_asr_vad_seg_reset(&seg);
        for (int i = 0; i < 20; i++) mynah_asr_vad_seg_feed(&p, &seg, probs[i], &s);
        CHECK(seg.triggered == 1, "state is triggered mid-speech");
        mynah_asr_vad_seg_reset(&seg);
        CHECK(seg.triggered == 0 && seg.i == 0 && seg.temp_end == -1, "reset clears the state");
        CHECK(mynah_asr_vad_seg_finish(&p, &seg, 20 * FRAME, &s) == 0,
              "finish after reset emits nothing");
    }

    /* ---- 11. null arguments are refused ------------------------------- */
    {
        mynah_asr_vad_seg seg;
        mynah_asr_vad_span s;
        mynah_asr_vad_seg_reset(&seg);
        CHECK(mynah_asr_vad_seg_feed(NULL, &seg, 0.9f, &s) == 0, "feed with no policy");
        CHECK(mynah_asr_vad_seg_feed(&p, NULL, 0.9f, &s) == 0, "feed with no state");
        CHECK(mynah_asr_vad_seg_feed(&p, &seg, 0.9f, NULL) == 0, "feed with nowhere to write");
        CHECK(mynah_asr_vad_seg_finish(&p, NULL, 100, &s) == 0, "finish with no state");
        mynah_asr_vad_pad_spans(&p, NULL, 3, 100);      /* must not crash */
        mynah_asr_vad_pad_spans(&p, sp, 0, 100);
        CHECK(1, "pad_spans survives empty input");
    }

    return failures;
}
