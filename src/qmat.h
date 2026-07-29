/* Matrice pesi con quantizzazione INT8/INT4 (al load o pre-quantizzata su disco).
 * Politica da prior-art (parakeet.cpp/qwen-tts): si quantizzano SOLO i grandi
 * linear consumati dalle GEMM; conv 2D, LSTM, norm, bias, embedding restano f32.
 *
 * Schemi:
 *  - INT8: per-riga simmetrica, scale [n]
 *  - INT4: per-gruppo (32 valori) simmetrica, nibble packed 2-per-byte,
 *          scale [n * k/32] (stile Q4_0)
 *
 * Dispatch del prodotto:
 *  - f32: cblas_sgemm
 *  - quantizzata, T piccolo (streaming/decode): kernel dot diretto (bandwidth)
 *  - quantizzata, T grande (offline/batch): dequant in scratch + sgemm */
#ifndef MYNAH_ASR_QMAT_H
#define MYNAH_ASR_QMAT_H

#include <stddef.h>
#include <stdint.h>

#include "weights.h"

enum { MYNAH_ASR_Q_F32 = 0, MYNAH_ASR_Q_INT8 = 1, MYNAH_ASR_Q_INT4 = 2 };

#include <math.h>

/* Sigmoide STABILE: mai expf(argomento positivo grande) -> inf. Con -ffast-math
 * l'inf è UB: gcc x86 vettorizza expf via libmvec e l'inf diventa NaN (visto in
 * CI 2026-07-18: encoder NaN solo su linux x86; clang/ARM sopravvivevano per
 * caso). expf(x) con x <= 0 non overflowa mai. */
static inline float mynah_asr_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

#define MYNAH_ASR_Q4_GROUP 32

typedef struct {
    const float *f32;       /* pesi originali [n, k] — usati se qtype == F32 */
    const int8_t *q8;       /* INT8 [n, k] */
    const uint8_t *q4;      /* INT4 packed [n, k/2] */
    const float *scales;    /* INT8: [n]; INT4: [n, k/32] */
    void *owned_q, *owned_s;/* buffer nostri (quantizzazione al load), da liberare */
    int qtype;
    int n, k;
} mynah_asr_qmat;

/* Inizializza cercando prima la forma pre-quantizzata nel safetensors
 * ("<name>.q8"/"<name>.q4" + "<name>.scales" — zero-copy dal mmap), poi il tensore
 * f32 "<name>" (quantizzato al volo se qtype != F32). 0 = ok, -1 tensore assente. */
int mynah_asr_qmat_init_st(mynah_asr_qmat *m, const mynah_asr_safetensors *st, const char *name,
                       int qtype);

/* Inizializzazione diretta da f32 (usata dal tool quantize e dai fallback). */
int mynah_asr_qmat_init(mynah_asr_qmat *m, const float *w, int n, int k, int qtype);

void mynah_asr_qmat_free(mynah_asr_qmat *m);

/* out[T, n] = x[T, k] @ W^T (layout linear PyTorch). */
void mynah_asr_qmat_mul(const mynah_asr_qmat *m, const float *x, float *out, int T);

/* FFN: out = SiLU(x @ W1^T) @ W2^T. scratch >= T*w1->n float.
 * Se entrambe F32 usa il path fuso del backend (Metal: un solo sync GPU). */
void mynah_asr_qmat_ffn(const mynah_asr_qmat *w1, const mynah_asr_qmat *w2, const float *x,
                    float *out, int T, float *scratch);

/* q/k/v sullo stesso input; se tutte F32 usa il multi-GEMM fuso del backend. */
void mynah_asr_qmat_qkv(const mynah_asr_qmat *wq, const mynah_asr_qmat *wk, const mynah_asr_qmat *wv,
                    const float *x, float *oq, float *ok, float *ov, int T);

/* Dequantizza l'intera matrice in wd [n, k] f32 del caller (no-op copia se già
 * f32). Per riusare una GEMM f32 su molte chiamate senza dequant per-chiamata
 * (es. joint head nel greedy a blocchi). */
void mynah_asr_qmat_dequant(const mynah_asr_qmat *m, float *wd);

/* Caps SIMD x86 a runtime (pattern --caps di qwen-tts): "auto" (default, cpuid),
 * "scalar", "avx2", "vnni"; env MYNAH_ASR_CAPS come alternativa al flag. Un livello
 * superiore a quello della CPU viene declassato con nota. Ritorna il livello
 * effettivo (0 scalar, 1 avx2, 2 vnni). Su ARM è un no-op: NEON/SDOT sono
 * compile-time (Apple Silicon ha sempre dotprod). */
int mynah_asr_set_caps(const char *name);

/* Quantizza un buffer f32 [n,k] in out_q/out_scales (buffer del caller):
 * INT8: out_q [n*k] int8, out_scales [n]
 * INT4: out_q [n*k/2] uint8, out_scales [n*k/32]
 * Usata dal tool `mynah-asr quantize`. */
void mynah_asr_quantize_int8(const float *w, int n, int k, int8_t *out_q, float *out_scales);
void mynah_asr_quantize_int4(const float *w, int n, int k, uint8_t *out_q, float *out_scales);

#endif
