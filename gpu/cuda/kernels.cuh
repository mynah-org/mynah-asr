/* gpu/cuda/kernels.cuh — device types shared by the kernels, the engine and
 * the kernel test, and the host-side launch wrappers.
 *
 * Every wrapper launches on the stream it is given and returns the launch's
 * cudaError_t; nothing here synchronises. Every per-stream kernel reads its
 * lane's geometry from a gpu_row descriptor, never from a grid dimension, so a
 * cohort may mix chunk sizes, first chunks and finalising tails
 * (.work/cuda-batched-streaming-server.md §3). */
#ifndef MYNAH_ASR_GPU_KERNELS_CUH
#define MYNAH_ASR_GPU_KERNELS_CUH

#include <cuda_runtime.h>
#include <stddef.h>

#define GPU_SS_STAGES 3
#define GPU_QMAX_HARD 32      /* a bound on q for the shared-memory tiles */
#define GPU_KMAX_HARD 160     /* a bound on valid + q for the score tile   */

/* One lane of a cohort: geometry the per-stream kernels read. Built on the
 * host per step, copied once. */
struct gpu_row {
    int slot;                  /* arena slot                                   */
    int q;                     /* encoder frames this chunk produces           */
    int n_mel, mel_off;        /* mel rows of the chunk, offset in the packed  */
                               /* [Σ n_mel, n_mels] upload                     */
    int first, last;           /* subsampling init pad / causal right pad      */
    int prompt;                /* language prompt id (one-hot column)          */
    int row_off;               /* first row in the stacked [R, d] activation   */
    int to[GPU_SS_STAGES];     /* time frames out of each subsampling stage    */
    int pos_off[GPU_SS_STAGES];/* first position of this lane in each stage's  */
                               /* position-major [Σpos, C] buffer              */
};

/* Device-resident per-slot state that is not a tensor. */
struct gpu_slot_meta {
    int valid, head;           /* K/V ring: logical rows held, physical row 0 */
    long long t_abs;           /* encoder frames produced so far              */
    int last_token, n_emitted; /* predictor: last fed token, natural emissions */
    /* the current step's label loop */
    int dec_t, dec_emitted, dec_done, dec_ntok;
};

/* The weights of one conformer layer, device pointers, f32. */
struct gpu_layer_w {
    const float *ln_ff1_w, *ln_ff1_b, *ff1_w1, *ff1_w2;
    const float *ln_att_w, *ln_att_b, *q_w, *k_w, *v_w, *o_w, *bias_u, *bias_v;
    const float *ln_conv_w, *ln_conv_b, *pw1_w, *dw_w, *cnorm_w, *cnorm_b, *pw2_w;
    const float *ln_ff2_w, *ln_ff2_b, *ff2_w1, *ff2_w2, *ln_out_w, *ln_out_b;
};

/* Everything a kernel needs to know about the model, by value. */
struct gpu_model_dims {
    int n_layers, d, H, dk, ffn, conv_k, left, kmax;
    int n_mels, C;             /* subsampling channels (256)                  */
    int F[GPU_SS_STAGES], Fo[GPU_SS_STAGES];  /* freq in / out per stage       */
    int dout, np, inter;       /* joint dim, prompts, prompt projector inner  */
    int V, Hdec, pred_layers, blank, max_symbols;
};

/* The arena: one pointer per tensor, indexed by slot inside the kernels. */
struct gpu_arena {
    float *kv;                 /* [cap][L][2][left][d]                        */
    float *conv_cache;         /* [cap][L][k-1][d]                            */
    float *ss_cache[GPU_SS_STAGES]; /* [cap][C_in_s][F_s]                     */
    float *dec_h, *dec_c;      /* [cap][pred_layers][Hdec]                    */
    float *dec_g;              /* [cap][Hdec]                                 */
    gpu_slot_meta *meta;       /* [cap]                                       */
    int *tok, *tok_frame;      /* [cap][tok_cap]                              */
    int tok_cap;
};

/* ------------------------------------------------------------------ GEMM
 * C[M,N] = A[M,K] · W[N,K]^T (+ bias[N]) (+ C when accumulate), then act
 * (0 none, 1 ReLU, 2 SiLU). A and C row-major with the leading dimensions
 * given; W row-major [N,K] with ld K. ROW-STABLE BY CONSTRUCTION: element
 * (m,n) is one fixed-order fma chain over k, whatever M and N are. */
cudaError_t k_gemm_wt(const float *A, int lda, const float *W, const float *bias,
                      float *C, int ldc, int M, int N, int K, int accumulate, int act,
                      cudaStream_t s);
/* v1, the S14-2 kernel: the bit reference every v2 configuration must equal,
 * and the A/B arm (--gemm own-v1). */
cudaError_t k_gemm_wt_v1(const float *A, int lda, const float *W, const float *bias,
                         float *C, int ldc, int M, int N, int K, int accumulate, int act,
                         cudaStream_t s);
/* tests: pin one v2 configuration (-1 = the dispatcher chooses) */
void k_gemm_force_config(int cfg);
int k_gemm_config_count(void);
const char *k_gemm_config_name(int cfg);

/* ------------------------------------------------------------- elementwise */
/* out[r] = LN(x[r]) * w + b over d, eps 1e-5, mean/var in double as the CPU;
 * silu_after applies x*sigmoid(x) to the result. out may alias x. */
cudaError_t k_layernorm(const float *x, const float *w, const float *b, float *out,
                        int rows, int d, int silu_after, cudaStream_t s);
/* x[i] += alpha * y[i] */
cudaError_t k_residual(float *x, const float *y, float alpha, size_t n, cudaStream_t s);
cudaError_t k_silu(float *x, size_t n, cudaStream_t s);
/* cat[r] = [x[r] (d) | one-hot(prompt of r's lane) (np)] */
cudaError_t k_prompt_cat(const float *x, const gpu_row *rows, int B, int d, int np,
                         float *cat, cudaStream_t s);

/* --------------------------------------------------------------- attention
 * For every lane and head: scores over [ring window ++ fresh rows] with the
 * rel-pos table rows for K = valid + q, rel_shift by index, softmax, P·V into
 * ctx. Reads valid/head from the slot's meta; never writes the ring. */
cudaError_t k_attention(const gpu_model_dims dm, const gpu_layer_w L, int li,
                        const float *relpos_tab, const gpu_arena ar,
                        const gpu_row *rows, int B, const float *qs, const float *kn,
                        const float *vn, float *ctx, cudaStream_t s);
/* Appends the q fresh K and V rows of every lane into layer li's ring. */
cudaError_t k_kv_commit(const gpu_model_dims dm, int li, const gpu_arena ar,
                        const gpu_row *rows, int B, const float *kn, const float *vn,
                        cudaStream_t s);
/* Once per step after every layer: valid/head/t_abs per lane. */
cudaError_t k_kv_advance(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                         int B, cudaStream_t s);

/* -------------------------------------------------------------------- conv
 * GLU over h2 [R, 2d], the cached causal depthwise (k taps, cache k-1 rows per
 * layer per slot, refreshed here), into out [R, d]. LayerNorm + SiLU follow as
 * k_layernorm(..., silu_after = 1). */
cudaError_t k_glu_dwconv(const gpu_model_dims dm, const gpu_layer_w L, int li,
                         const gpu_arena ar, const gpu_row *rows, int B, const float *h2,
                         float *out, cudaStream_t s);

/* ------------------------------------------------------------- subsampling
 * Stage 0: full 3x3 s2 conv, C_in = 1, from the packed mel with the one-frame
 * time cache prepended (and the init/right pads from the lane's flags); ReLU.
 * Output position-major [Σpos0, C]. Refreshes the stage cache. */
cudaError_t k_ss_stage0(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                        int B, const float *mel, const float *w, const float *b,
                        float *out, cudaStream_t s);
/* Stages 1 and 2: depthwise 3x3 s2 over the previous stage's position-major
 * output with the time cache; NO ReLU here (it follows the pointwise GEMM). */
cudaError_t k_ss_dw(const gpu_model_dims dm, int stage, const gpu_arena ar,
                    const gpu_row *rows, int B, const float *in, const float *w,
                    const float *b, float *out, cudaStream_t s);
/* flat[row_off + t][c*Fo2 + f] = s2[pos_off2 + t*Fo2 + f][c] */
cudaError_t k_ss_flatten(const gpu_model_dims dm, const gpu_row *rows, int B,
                         const float *s2, float *flat, cudaStream_t s);

/* ------------------------------------------------------------------ decode */
/* Per lane: dec_t = 0, dec_emitted = 0, dec_done = (q == 0), dec_ntok = 0. */
cudaError_t k_dec_begin(const gpu_arena ar, const gpu_row *rows, int B, cudaStream_t s);
/* active[0..n) = lanes not done; *n_out = n. One block. */
cudaError_t k_dec_compact(const gpu_arena ar, const gpu_row *rows, int B, int *active,
                          int *n_out, cudaStream_t s);
/* jin[a] = ReLU(enc[row_off + dec_t] + g[slot]) for the n active lanes */
cudaError_t k_dec_joint(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                        const int *active, int n, const float *enc, float *jin,
                        cudaStream_t s);
/* am[a] = argmax over logits[a] (V), the LOWEST index among equal maxima */
cudaError_t k_dec_argmax(const float *logits, int V, int n, int *am, cudaStream_t s);
/* The greedy rule per active lane; emit[a] = 1 when a token was appended. */
cudaError_t k_dec_decide(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                         const int *active, int n, const int *am, int *emit,
                         cudaStream_t s);
/* x[a] = embedding[am[a]] (every active lane; masked lanes' rows are unused) */
cudaError_t k_dec_gather_emb(const float *emb, int H, const int *am, int n, float *x,
                             cudaStream_t s);
/* h_rows[a] = dec_h[slot][layer] for the active lanes */
cudaError_t k_dec_gather_h(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                           const int *active, int n, int layer, float *h_rows,
                           cudaStream_t s);
/* LSTM gates from z[a][4H] (i,f,g,o) for lanes with emit[a]: updates dec_c and
 * dec_h of `layer` in place and writes the new h into x_next[a]. */
cudaError_t k_dec_lstm_gates(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                             const int *active, const int *emit, int n, int layer,
                             const float *z, float *x_next, cudaStream_t s);
/* For lanes with emit[a]: g[slot] = gtmp[a]; last_token = am[a]; n_emitted++ */
cudaError_t k_dec_commit_g(const gpu_model_dims dm, const gpu_arena ar, const gpu_row *rows,
                           const int *active, const int *emit, const int *am, int n,
                           const float *gtmp, cudaStream_t s);

/* ------------------------------------------------------------------- slots */
/* Empty caches, decoder at the SOS state (copied from the model's precomputed
 * SOS h/c/g), meta reset, for the listed slots. */
cudaError_t k_slots_reset(const gpu_model_dims dm, const gpu_arena ar, const int *slots,
                          int n, const float *sos_h, const float *sos_c, const float *sos_g,
                          cudaStream_t s);

#endif
