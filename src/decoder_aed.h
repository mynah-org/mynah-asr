/* Decoder AED (Canary): Transformer pre-LN con self-attention causale (KV cache)
 * e cross-attention sull'encoder out proiettato. Greedy autoregressivo col prompt
 * canary2; semantica in docs/canary-arch.md, riferimento oracle/model.py.
 * I grandi linear passano da mynah_asr_qmat: --quant int8/int4 e checkpoint
 * pre-quantizzati coprono anche il decoder. */
#ifndef MYNAH_ASR_DECODER_AED_H
#define MYNAH_ASR_DECODER_AED_H

#include "qmat.h"
#include "weights.h"

typedef struct {
    const float *ln_self_w, *ln_self_b;
    mynah_asr_qmat sq, sk, sv, so;
    const float *sq_b, *sk_b, *sv_b, *so_b;
    const float *ln_cross_w, *ln_cross_b;
    mynah_asr_qmat cq, ck, cv, co;
    const float *cq_b, *ck_b, *cv_b, *co_b;
    const float *ln_ffn_w, *ln_ffn_b;
    mynah_asr_qmat ff1, ff2;
    const float *ff1_b, *ff2_b;
} mynah_asr_aed_layer;

typedef struct {
    int n_layers, d, n_heads, ffn, vocab, max_seq, max_gen_delta;
    int d_enc;                       /* input di enc_dec_proj (= d se identity) */
    mynah_asr_qmat proj;                 /* enc_dec_proj (n == 0 = identity)        */
    const float *proj_b;
    const float *emb;                /* [vocab, d]                              */
    const float *pos;                /* [max_seq, d] buffer dal ckpt (già /√d)  */
    const float *embln_w, *embln_b;
    const float *fin_w, *fin_b;      /* final_layer_norm                        */
    mynah_asr_qmat head;                 /* [vocab, d]                              */
    const float *head_b;
    mynah_asr_aed_layer *layers;
} mynah_asr_aed;

/* 0 = ok, -1 = pesi AED assenti (modello non-AED). */
int mynah_asr_aed_init(mynah_asr_aed *a, const mynah_asr_safetensors *st,
                   int n_layers, int n_heads, int max_seq, int max_gen_delta,
                   int quantize);
void mynah_asr_aed_free(mynah_asr_aed *a);

/* Greedy: enc [T, d_enc] -> token generati (senza prompt né EOS) in tokens[cap].
 * Ritorna il numero di token, -1 su errore. */
int mynah_asr_aed_decode(const mynah_asr_aed *a, const float *enc, int T,
                     const int *prompt, int n_prompt, int eos,
                     int *tokens, int cap);

#endif
