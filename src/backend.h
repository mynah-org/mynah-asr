/* Backend di calcolo per le GEMM grandi: CPU (BLAS) o Metal/MPS su macOS.
 * Pattern qwen-tts: richiesta -> resolve() -> nota -> fallback graceful CPU.
 * Il backend è una scelta di processo (mynah_asr_set_backend prima del load). */
#ifndef MYNAH_ASR_BACKEND_H
#define MYNAH_ASR_BACKEND_H

#include <stddef.h>

enum { MYNAH_ASR_BACKEND_CPU = 0, MYNAH_ASR_BACKEND_METAL = 1, MYNAH_ASR_BACKEND_CUDA = 2 };

/* "cpu" | "metal" | "cuda". Ritorna il backend EFFETTIVO dopo il resolve (fallback CPU
 * con nota su stderr se quello richiesto non è disponibile). */
int mynah_asr_set_backend(const char *name);
int mynah_asr_backend(void);

/* out[T,n] = x[T,k] @ W[n,k]^T — dispatch: Metal per T grandi se attivo, else BLAS.
 * W deve essere stabile per la vita del processo (i pesi mmap lo sono): su Metal
 * viene copiato UNA volta in un MTLBuffer residente (cache per-pointer). */
void mynah_asr_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);

/* FFN fusa: out[T,n2] = SiLU(x @ W1^T) @ W2^T. scratch: >= T*n1 float (usato
 * solo nel fallback CPU). Su Metal l'intermedio resta in GPU (un solo sync). */
void mynah_asr_silu(float *x, size_t n);   /* x = x*sigmoid(x), vettorizzata su Accelerate */
void mynah_asr_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                  float *out, int T, int k, float *scratch);

/* Tre GEMM sullo stesso input (q/k/v): su Metal un solo sync. */
void mynah_asr_gemm3_wt(const float *x, const float *wa, const float *wb, const float *wc,
                    float *oa, float *ob, float *oc, int T, int n, int k);

#ifdef MYNAH_ASR_METAL
/* v4: encoder intero su GPU — stream residuo f32 residente, LN/residual/softmax
 * in shader, un solo sync per forward. Pesi f32 host-stabili (convertiti f16 in
 * cache residente alla prima chiamata). Ritorna -1 -> fallback CPU. */
typedef struct {
    const float *ln_ff1_w, *ln_ff1_b, *ff1_w1, *ff1_w2;
    const float *ln_att_w, *ln_att_b, *wq, *wk, *wv, *wo, *relk, *bias_u, *bias_v;
    const float *ln_conv_w, *ln_conv_b, *pw1, *dw9, *cnorm_w, *cnorm_b, *pw2;
    const float *ln_ff2_w, *ln_ff2_b, *ff2_w1, *ff2_w2;
    const float *ln_out_w, *ln_out_b;
    /* opzionali (NULL se assenti): bias dei linear (use_bias true: 110m, rnnt/ctc)
     * e BatchNorm foldata del conv module (Parakeet: sostituisce cnorm_w/b) */
    const float *ff1_b1, *ff1_b2, *ff2_b1, *ff2_b2;
    const float *q_b, *k_b, *v_b, *o_b;
    const float *pw1_b, *dw_b, *pw2_b;
    const float *cnorm_scale, *cnorm_shift;
} mynah_asr_metal_layer_w;

/* conv_pad: padding sinistro della depthwise (k-1 causale, (k-1)/2 'same').
 * left < 0 = attention full (modelli offline). */
int mynah_asr_metal_encoder_layers(const mynah_asr_metal_layer_w *Ls, int n_layers,
                               float *x, const float *pe, int T, int d, int H,
                               int ffn, int left, int right, int conv_pad);

/* Svuota la cache dei pesi GPU (chiamata da mynah_asr_free: i puntatori host mmap
 * cessano di essere validi e un load successivo può riusarne gli indirizzi). */
void mynah_asr_metal_weights_evict(void);
#endif

#endif
