/* Compute backend for the large GEMMs: CPU (BLAS) or Metal/MPS on macOS.
 * qwen-tts pattern: request -> resolve() -> note -> graceful CPU fallback.
 * The backend is a process-wide choice (mynah_asr_set_backend before load). */
#ifndef MYNAH_ASR_BACKEND_H
#define MYNAH_ASR_BACKEND_H

#include <stddef.h>

enum { MYNAH_ASR_BACKEND_CPU = 0, MYNAH_ASR_BACKEND_METAL = 1, MYNAH_ASR_BACKEND_CUDA = 2 };

/* "cpu" | "metal" | "cuda". Returns the EFFECTIVE backend after resolve (CPU
 * fallback, with a note on stderr, when the requested one is unavailable). */
int mynah_asr_set_backend(const char *name);
int mynah_asr_backend(void);

/* ------------------------------------------------------------- the f32 seam
 *
 * EVERY f32 GEMM and GEMV in this runtime goes through the three entry points
 * below, and they are the ONLY place that knows which provider computes it:
 * `accelerate` (macOS, BLAS=accelerate), `openblas` (BLAS=openblas) or `own`
 * (src/sgemm.c, the Linux default).  No other translation unit includes
 * <cblas.h> or <Accelerate/Accelerate.h> for arithmetic — that is what makes
 * `BLAS=none` a build rather than a patch, and what makes
 * mynah_asr_gemm_provider() a fact instead of a guess (ENGINEERING.md §6).
 *
 * The provider this binary linked, as one of those three words.  A pure
 * compile-time gate answered by the file that owns the branch; src/flags.c and
 * src/dispatch.c both READ it rather than re-deriving it. */
const char *mynah_asr_gemm_provider(void);

/* C[m,n] = alpha * op(A) * op(B) + beta * C, row-major — cblas_sgemm's
 * contract, including beta == 0 meaning "C is written, never read". */
void mynah_asr_gemm_f32(int trans_a, int trans_b, int m, int n, int k,
                        float alpha, const float *a, int lda,
                        const float *b, int ldb, float beta, float *c, int ldc);

/* y = alpha * op(A) * x + beta * y, A row-major [rows, cols] with row stride
 * lda, x and y unit-stride — cblas_sgemv(CblasRowMajor, ..., incx=1, incy=1).
 * trans == 0: y has `rows` elements and x has `cols`; trans != 0: the reverse. */
void mynah_asr_gemv_f32(int trans, int rows, int cols, float alpha,
                        const float *a, int lda, const float *x,
                        float beta, float *y);

/* out[T,n] = x[T,k] @ W[n,k]^T — dispatch: Metal/CUDA for large T when active,
 * else mynah_asr_gemm_f32.  W must stay stable for the life of the process
 * (mmap'd weights are): on Metal it is copied ONCE into a resident MTLBuffer
 * (per-pointer cache). */
void mynah_asr_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);

/* Fused FFN: out[T,n2] = SiLU(x @ W1^T) @ W2^T. scratch: >= T*n1 floats (used
 * only in the CPU fallback). On Metal the intermediate stays on GPU (one sync). */
/* x = x*sigmoid(x); vectorized through Accelerate's vForce where that
 * framework is present, which on macOS is EVERY build — vvexpf is not the BLAS
 * and does not move with BLAS=none, so the two macOS builds run the same
 * arithmetic here and a transcript difference between them can only come from
 * the GEMM. */
void mynah_asr_silu(float *x, size_t n);
/* Same, with caller-owned scratch (>= n floats) for the vectorized path, so a
 * hot loop never allocates. scratch == NULL behaves exactly like mynah_asr_silu.
 * Identical arithmetic in both forms (the scratch only replaces the malloc). */
void mynah_asr_silu_scratch(float *x, size_t n, float *scratch);
void mynah_asr_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                  float *out, int T, int k, float *scratch);

/* Three GEMMs over the same input (q/k/v): a single sync on Metal. */
void mynah_asr_gemm3_wt(const float *x, const float *wa, const float *wb, const float *wc,
                    float *oa, float *ob, float *oc, int T, int n, int k);

#ifdef MYNAH_ASR_METAL
/* v4: the whole encoder on GPU — resident f32 residual stream, LN/residual/softmax
 * in shaders, one sync per forward. Host-stable f32 weights (converted to f16 in a
 * resident cache on the first call). Returns -1 -> CPU fallback. */
typedef struct {
    const float *ln_ff1_w, *ln_ff1_b, *ff1_w1, *ff1_w2;
    const float *ln_att_w, *ln_att_b, *wq, *wk, *wv, *wo, *relk, *bias_u, *bias_v;
    const float *ln_conv_w, *ln_conv_b, *pw1, *dw9, *cnorm_w, *cnorm_b, *pw2;
    const float *ln_ff2_w, *ln_ff2_b, *ff2_w1, *ff2_w2;
    const float *ln_out_w, *ln_out_b;
    /* optional (NULL when absent): linear biases (use_bias true: 110m, rnnt/ctc)
     * and the folded BatchNorm of the conv module (Parakeet: replaces cnorm_w/b) */
    const float *ff1_b1, *ff1_b2, *ff2_b1, *ff2_b2;
    const float *q_b, *k_b, *v_b, *o_b;
    const float *pw1_b, *dw_b, *pw2_b;
    const float *cnorm_scale, *cnorm_shift;
} mynah_asr_metal_layer_w;

/* conv_pad: left padding of the depthwise conv (k-1 causal, (k-1)/2 'same').
 * left < 0 = full attention (offline models). */
int mynah_asr_metal_encoder_layers(const mynah_asr_metal_layer_w *Ls, int n_layers,
                               float *x, const float *pe, int T, int d, int H,
                               int ffn, int left, int right, int conv_pad);

/* Drop the GPU weight cache (called by mynah_asr_free: the mmap'd host pointers
 * stop being valid and a later load may reuse the same addresses). */
void mynah_asr_metal_weights_evict(void);
#endif

#endif
