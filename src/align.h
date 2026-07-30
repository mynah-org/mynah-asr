/* CTC forced alignment (Viterbi): given a token sequence that is already known
 * to be the answer, find WHERE in time each token sits.
 *
 * This is what gives canary-1b-v2 word timestamps: an AED decoder generates the
 * text but its attention is not monotonic, so the times cannot come out of the
 * decoder itself (docs/canary-arch.md). NVIDIA ships a separate CTC aligner in
 * the .nemo for exactly this; the text comes from the AED, the times from
 * aligning that text against the aligner's per-frame posteriors.
 *
 * Independent of any model: it takes a score matrix and a label sequence, which
 * is why tests/test_align.c can exercise it with synthetic posteriors and a
 * known answer, with no checkpoint at all. */
#ifndef MYNAH_ASR_ALIGN_H
#define MYNAH_ASR_ALIGN_H

/* Aligns `tokens` (n_tokens labels, none of them `blank`) over `scores`, a
 * [T, V] row-major matrix of per-frame per-label scores — raw logits are fine,
 * see the note in align.c: the softmax normalizer is the same for every path and
 * cancels in the argmax, so no log_softmax pass is needed.
 *
 * Writes, for every token, the first and last frame of the run it occupies:
 * t0[j] <= t1[j] < t0[j+1], every frame index in [0, T).
 *
 * Returns 0 on success, -1 on bad arguments or when the alignment is impossible
 * (T shorter than the tokens plus the blanks that repeated labels require).
 * Never allocates more than O(T * n_tokens) bytes. */
int mynah_asr_align_ctc(const float *scores, int T, int V, const int *tokens, int n_tokens,
                        int blank, int *t0, int *t1);

/* Minimum number of frames any CTC path through `tokens` needs: one frame per
 * token plus one blank between every pair of identical neighbours. */
int mynah_asr_align_min_frames(const int *tokens, int n_tokens);

#endif
