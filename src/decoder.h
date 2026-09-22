/* RNNT/TDT decoder: LSTM prediction network + joint, greedy decode.
 * NeMo/HF semantics (docs/nemotron-arch.md, docs/parakeet-tdt-arch.md):
 * SOS = blank, the LSTM state advances ONLY on a non-blank emission,
 * max_symbols_per_step caps the emissions per frame.
 * TDT (n_durations > 0): the head emits [vocab | n_durations] logits; the frame
 * advances by the predicted duration at each step (blank with dur 0 -> forced to 1).
 * Lifted, chunk-invariant state: incremental decoding is equivalent to whole-input
 * decoding (prior art). */
#ifndef MYNAH_ASR_DECODER_H
#define MYNAH_ASR_DECODER_H

#include "qmat.h"
#include "weights.h"

#define MYNAH_ASR_MAX_PRED_LAYERS 4
#define MYNAH_ASR_MAX_DURATIONS 16

typedef struct {
    const float *embedding;                 /* [vocab, H] */
    const float *w_ih[MYNAH_ASR_MAX_PRED_LAYERS], *w_hh[MYNAH_ASR_MAX_PRED_LAYERS];
    const float *b_ih[MYNAH_ASR_MAX_PRED_LAYERS], *b_hh[MYNAH_ASR_MAX_PRED_LAYERS];
    const float *proj_w, *proj_b;           /* decoder_projector [H, H] */
    mynah_asr_qmat head;                        /* joint.head [vocab(+dur), H] (opt. int8) */
    const float *head_b;
    int vocab, hidden, n_layers, blank, max_symbols;
    /* TDT: vocab = rows of the head; the tokens are the first vocab-n_durations */
    int n_durations;
    int durations[MYNAH_ASR_MAX_DURATIONS];
    /* R-9, diagnostic only: id of the bare SentencePiece word mark, RESOLVED
     * from the pack's own tokens.json by the loader and never hard-coded (it is
     * 2 in the Nemotron pack and there is no reason another pack agrees). -1
     * when the vocabulary has no such piece. Read only by the trace. */
    int word_mark;
} mynah_asr_decoder;

typedef struct {
    float h[MYNAH_ASR_MAX_PRED_LAYERS][1024];   /* max size; use dec->hidden */
    float c[MYNAH_ASR_MAX_PRED_LAYERS][1024];
    float g[1024];                          /* current pred-net output (cache) */
    int last_token;                         /* -1 = not initialized */
    /* Natural non-blank emissions so far. The predictor is at its SOS
     * representation exactly while this is 0, which `last_token` cannot say on
     * its own once a diagnostic intervention has fed a token through
     * pred_step() without emitting it. */
    int n_emitted;
    long t_abs;                             /* absolute encoder frames seen so far
                                               (for streaming timestamps) */
} mynah_asr_dec_state;

/* durations: TDT array from the config (NULL/0 = pure RNNT). */
int mynah_asr_decoder_init(mynah_asr_decoder *dec, const mynah_asr_safetensors *st,
                       int blank, int max_symbols, int quantize,
                       const int *durations, int n_durations);

void mynah_asr_dec_state_reset(const mynah_asr_decoder *dec, mynah_asr_dec_state *s);

/* Greedy RNNT/TDT over enc [T, H]. Appends the emitted tokens to tokens[]
 * (capacity cap) and, when frames != NULL, the ABSOLUTE encoder frame each token
 * was emitted at (base = sum of the T of previous calls on the same state).
 * Returns the number of emitted tokens. The state persists across calls
 * (streaming-ready): pass the same state for the following chunks. */
int mynah_asr_greedy_decode(const mynah_asr_decoder *dec, mynah_asr_dec_state *s,
                        const float *enc, int T, int *tokens, int *frames, int cap);

/* Same, with caller-owned scratch for the joint input and the logits, so a
 * streaming step never allocates. NULL = allocate internally, i.e. exactly
 * mynah_asr_greedy_decode. Floats required: mynah_asr_greedy_scratch_floats.
 * NOTE: the quantized head is still dequantized into a temporary when T > 16
 * (offline-sized calls only; a streaming chunk is q = right+1 frames). */
size_t mynah_asr_greedy_scratch_floats(const mynah_asr_decoder *dec);
/* R-9 diagnostic: tell the trace how much audio the caller had consumed when it
 * handed over this block. The decoder has no notion of samples; the stream layer
 * knows it exactly, and reconstructing it from frame indices outside would be a
 * second, worse answer. No effect unless MYNAH_ASR_TRACE_RNNT is on. */
void mynah_asr_dec_trace_audio(double audio_s);

int mynah_asr_greedy_decode_scratch(const mynah_asr_decoder *dec, mynah_asr_dec_state *s,
                                const float *enc, int T, int *tokens, int *frames, int cap,
                                float *scratch);

#endif
