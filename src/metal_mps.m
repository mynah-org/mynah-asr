/* Metal backend v4 — the whole encoder on GPU, one sync per forward.
 *
 * History: v1 one GEMM = one sync (~10/layer, lost against AMX). v2 resident
 * fp16 weights + fused FFN/QKV (~5/layer). v3 attention and conv module
 * entirely on GPU (4/layer). v4: the residual stream stays on GPU too — f32
 * for numeric fidelity (the blocks work in f16 as in v3, but the LNs between
 * blocks and the residual adds accumulate in f32, like the CPU). The
 * f32<->f16 conversions happen inside the shaders (LN reads f32 and writes
 * f16): the CPU only does two memcpys (upload x, download x) for the whole
 * encoder. One command buffer per layer, committed without waiting: the GPU
 * runs layer i while the CPU encodes layer i+1; only the last one is waited on.
 *
 * Below MYNAH_ASR_METAL_MIN_T (streaming chunks) it returns -1 -> CPU. */
#ifdef MYNAH_ASR_METAL

#include "backend.h"

#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MYNAH_ASR_METAL_MIN_T 24

typedef _Float16 f16;

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_silu, g_glu, g_dwconv, g_lnorm, g_addbias, g_smax,
                                   g_cvt, g_resadd, g_ln32h, g_ln32f, g_caffine;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    const void *host_ptr;
    void *buf;              /* id<MTLBuffer> fp16 (retained) */
} wc_ent;
static wc_ent *g_wc;
static int g_wc_n, g_wc_cap;

/* reusable I/O buffers (protected by g_mu) */
static id<MTLBuffer> g_in, g_mid, g_out, g_out2, g_out3;
static size_t g_in_cap, g_mid_cap, g_out_cap, g_out2_cap, g_out3_cap;
/* v4: f32 residual stream, pe staging, f16 pe, block input/output, attn scratch */
static id<MTLBuffer> g_x32, g_pe32, g_pe16, g_xn, g_blk, g_att;
static size_t g_x32_cap, g_pe32_cap, g_pe16_cap, g_xn_cap, g_blk_cap, g_att_cap;

static const char *SHADER_SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void silu(device half *x [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
    "    float v = float(x[i]);\n"
    "    x[i] = half(v / (1.0f + exp(-v)));\n"
    "}\n"
    /* GLU over the channels: x [T,2d] -> g [T,d] */
    "kernel void glu(device const half *x [[buffer(0)]], device half *g [[buffer(1)]],\n"
    "                constant uint &d [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    uint t = i / d, c = i % d;\n"
    "    float a = float(x[t * 2 * d + c]);\n"
    "    float b = float(x[t * 2 * d + d + c]);\n"
    "    g[i] = half(a / (1.0f + exp(-b)));\n"
    "}\n"
    /* depthwise conv k=9: c[t,i] = b[i] + sum_j w[i,j] * g[t-pad+j, i]
       (pad 8 = causale, 4 = 'same'; has_b 0 -> bias non letto) */
    "kernel void dwconv9(device const half *g [[buffer(0)]], device const half *w [[buffer(1)]],\n"
    "                    device half *c [[buffer(2)]], constant uint &d [[buffer(3)]],\n"
    "                    constant uint &T [[buffer(4)]], constant uint &pad [[buffer(5)]],\n"
    "                    device const half *b [[buffer(6)]], constant uint &has_b [[buffer(7)]],\n"
    "                    uint i [[thread_position_in_grid]]) {\n"
    "    uint t = i / d, ch = i % d;\n"
    "    float acc = has_b ? float(b[ch]) : 0.0f;\n"
    "    for (uint j = 0; j < 9; j++) {\n"
    "        int src = (int)t - (int)pad + (int)j;\n"
    "        if (src >= 0 && src < (int)T) acc += float(w[ch * 9 + j]) * float(g[(uint)src * d + ch]);\n"
    "    }\n"
    "    c[i] = half(acc);\n"
    "}\n"
    /* per-channel affine (folded BatchNorm): x = x * s[ch] + b[ch] */
    "kernel void caffine(device half *x [[buffer(0)]], device const half *s [[buffer(1)]],\n"
    "                    device const half *b [[buffer(2)]], constant uint &d [[buffer(3)]],\n"
    "                    uint i [[thread_position_in_grid]]) {\n"
    "    uint ch = i % d;\n"
    "    x[i] = half(float(x[i]) * float(s[ch]) + float(b[ch]));\n"
    "}\n"
    /* Per-row reduction shared by the layernorms: one THREADGROUP per row
       (letture coalescenti + simd_sum), non un thread per riga (stride d,
       ~128B di traffico per elemento: era il collo di bottiglia della v4). */
    "static float row_moment(float part, threadgroup float *sh, uint tid, uint tpt) {\n"
    "    part = simd_sum(part);\n"
    "    uint nsg = (tpt + 31) / 32;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if ((tid & 31u) == 0u) sh[tid / 32] = part;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    float tot = 0.0f;\n"
    "    for (uint i = 0; i < nsg; i++) tot += sh[i];\n"
    "    return tot;\n"
    "}\n"
    /* per-row f16 layernorm (the conv module's internal norm) */
    "kernel void lnorm(device half *x [[buffer(0)]], device const half *gm [[buffer(1)]],\n"
    "                  device const half *bt [[buffer(2)]], constant uint &d [[buffer(3)]],\n"
    "                  uint tg [[threadgroup_position_in_grid]],\n"
    "                  uint tid [[thread_position_in_threadgroup]],\n"
    "                  uint tpt [[threads_per_threadgroup]]) {\n"
    "    device half *row = x + tg * d;\n"
    "    threadgroup float sh[32];\n"
    "    float s = 0.0f;\n"
    "    for (uint i = tid; i < d; i += tpt) s += float(row[i]);\n"
    "    float mu = row_moment(s, sh, tid, tpt) / d;\n"
    "    float v = 0.0f;\n"
    "    for (uint i = tid; i < d; i += tpt) { float c = float(row[i]) - mu; v += c * c; }\n"
    "    float inv = rsqrt(row_moment(v, sh, tid, tpt) / d + 1e-5f);\n"
    "    for (uint i = tid; i < d; i += tpt)\n"
    "        row[i] = half((float(row[i]) - mu) * inv * float(gm[i]) + float(bt[i]));\n"
    "}\n"
    /* q + per-head bias (broadcast over T): out[t, h*dk+i] = q[..] + bias[h*dk+i] */
    "kernel void addbias(device const half *q [[buffer(0)]], device const half *b [[buffer(1)]],\n"
    "                    device half *o [[buffer(2)]], constant uint &d [[buffer(3)]],\n"
    "                    uint i [[thread_position_in_grid]]) {\n"
    "    o[i] = half(float(q[i]) + float(b[i % d]));\n"
    "}\n"
    /* softmax with rel_shift in closed form and the chunked_limited window
       (chunk == 0 -> attention full, modelli offline).
       scores [H,T,K] (ac), bd [H,T,P]; P = 2L-1, gq = cached + t (qui cached=0). */
    "kernel void smax_shift(device half *sc [[buffer(0)]], device const half *bd [[buffer(1)]],\n"
    "                       constant uint &T [[buffer(2)]], constant uint &K [[buffer(3)]],\n"
    "                       constant uint &P [[buffer(4)]], constant float &scale [[buffer(5)]],\n"
    "                       constant uint &chunk [[buffer(6)]], constant uint &lc [[buffer(7)]],\n"
    "                       uint idx [[thread_position_in_grid]]) {\n"
    "    uint h = idx / T, t = idx % T;\n"
    "    device half *srow = sc + (h * T + t) * K;\n"
    "    device const half *brow = bd + (h * T + t) * P;\n"
    "    uint j0 = 0, j1 = K;\n"
    "    if (chunk > 0) {\n"
    "        uint tc = t / chunk;\n"
    "        j0 = (tc >= lc) ? (tc - lc) * chunk : 0;\n"
    "        j1 = min((tc + 1) * chunk, K);\n"
    "    }\n"
    "    int base = (int)K - 1 - (int)t;\n"          /* p = K-1 + j - t */
    "    float mx = -3.0e38f;\n"
    "    for (uint j = j0; j < j1; j++) {\n"
    "        float v = (float(srow[j]) + float(brow[base + (int)j])) * scale;\n"
    "        srow[j] = half(v);\n"
    "        if (v > mx) mx = v;\n"
    "    }\n"
    "    float sum = 0.0f;\n"
    "    for (uint j = j0; j < j1; j++) { float e = exp(float(srow[j]) - mx); srow[j] = half(e); sum += e; }\n"
    "    float inv = 1.0f / sum;\n"
    "    for (uint j = 0; j < j0; j++) srow[j] = half(0.0f);\n"
    "    for (uint j = j0; j < j1; j++) srow[j] = half(float(srow[j]) * inv);\n"
    "    for (uint j = j1; j < K; j++) srow[j] = half(0.0f);\n"
    "}\n"
    /* --- v4: f32 residual stream --- */
    "kernel void cvt32to16(device const float *x [[buffer(0)]], device half *o [[buffer(1)]],\n"
    "                      uint i [[thread_position_in_grid]]) { o[i] = half(x[i]); }\n"
    /* residual add: x32 += alpha * y16 */
    "kernel void resadd(device float *x [[buffer(0)]], device const half *y [[buffer(1)]],\n"
    "                   constant float &alpha [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    x[i] += alpha * float(y[i]);\n"
    "}\n"
    /* per-row layernorm: f32 stream -> f16 block input */
    "kernel void ln32h(device const float *x [[buffer(0)]], device half *o [[buffer(1)]],\n"
    "                  device const half *gm [[buffer(2)]], device const half *bt [[buffer(3)]],\n"
    "                  constant uint &d [[buffer(4)]],\n"
    "                  uint tg [[threadgroup_position_in_grid]],\n"
    "                  uint tid [[thread_position_in_threadgroup]],\n"
    "                  uint tpt [[threads_per_threadgroup]]) {\n"
    "    device const float *row = x + tg * d;\n"
    "    threadgroup float sh[32];\n"
    "    float s = 0.0f;\n"
    "    for (uint i = tid; i < d; i += tpt) s += row[i];\n"
    "    float mu = row_moment(s, sh, tid, tpt) / d;\n"
    "    float v = 0.0f;\n"
    "    for (uint i = tid; i < d; i += tpt) { float c = row[i] - mu; v += c * c; }\n"
    "    float inv = rsqrt(row_moment(v, sh, tid, tpt) / d + 1e-5f);\n"
    "    for (uint i = tid; i < d; i += tpt)\n"
    "        o[tg * d + i] = half((row[i] - mu) * inv * float(gm[i]) + float(bt[i]));\n"
    "}\n"
    /* in-place f32 layernorm (norm_out between layers) */
    "kernel void ln32f(device float *x [[buffer(0)]], device const half *gm [[buffer(1)]],\n"
    "                  device const half *bt [[buffer(2)]], constant uint &d [[buffer(3)]],\n"
    "                  uint tg [[threadgroup_position_in_grid]],\n"
    "                  uint tid [[thread_position_in_threadgroup]],\n"
    "                  uint tpt [[threads_per_threadgroup]]) {\n"
    "    device float *row = x + tg * d;\n"
    "    threadgroup float sh[32];\n"
    "    float s = 0.0f;\n"
    "    for (uint i = tid; i < d; i += tpt) s += row[i];\n"
    "    float mu = row_moment(s, sh, tid, tpt) / d;\n"
    "    float v = 0.0f;\n"
    "    for (uint i = tid; i < d; i += tpt) { float c = row[i] - mu; v += c * c; }\n"
    "    float inv = rsqrt(row_moment(v, sh, tid, tpt) / d + 1e-5f);\n"
    "    for (uint i = tid; i < d; i += tpt)\n"
    "        row[i] = (row[i] - mu) * inv * float(gm[i]) + float(bt[i]);\n"
    "}\n";

int mynah_asr_metal_available(void) {
    static int checked = 0;
    pthread_mutex_lock(&g_mu);
    if (!checked) {
        g_dev = MTLCreateSystemDefaultDevice();
        if (g_dev) {
            g_queue = [g_dev newCommandQueue];
            NSError *err = nil;
            id<MTLLibrary> lib = [g_dev newLibraryWithSource:
                [NSString stringWithUTF8String:SHADER_SRC] options:nil error:&err];
            if (!lib && err)
                fprintf(stderr, "mynah metal: shader: %s\n",
                        err.localizedDescription.UTF8String);
            id<MTLComputePipelineState> __strong *ps[] = {&g_silu, &g_glu, &g_dwconv,
                                                          &g_lnorm, &g_addbias, &g_smax,
                                                          &g_cvt, &g_resadd, &g_ln32h,
                                                          &g_ln32f, &g_caffine};
            const char *names[] = {"silu", "glu", "dwconv9", "lnorm", "addbias",
                                   "smax_shift", "cvt32to16", "resadd", "ln32h", "ln32f",
                                   "caffine"};
            for (int i = 0; i < 11 && lib; i++) {
                id<MTLFunction> fn = [lib newFunctionWithName:
                    [NSString stringWithUTF8String:names[i]]];
                if (fn) *ps[i] = [g_dev newComputePipelineStateWithFunction:fn error:&err];
            }
        }
        checked = 1;
    }
    pthread_mutex_unlock(&g_mu);
    return g_dev != nil && g_queue != nil && g_silu != nil && g_glu != nil &&
           g_dwconv != nil && g_lnorm != nil && g_addbias != nil && g_smax != nil &&
           g_cvt != nil && g_resadd != nil && g_ln32h != nil && g_ln32f != nil &&
           g_caffine != nil;
}

/* bulk f32<->f16 conversions through vImage (NEON, ~20 GB/s vs a scalar loop) */
static void cvt_f32_to_f16(const float *src, void *dst, size_t n) {
    vImage_Buffer s = {.data = (void *)src, .height = 1, .width = n, .rowBytes = n * 4};
    vImage_Buffer d = {.data = dst, .height = 1, .width = n, .rowBytes = n * 2};
    if (vImageConvert_PlanarFtoPlanar16F(&s, &d, kvImageNoFlags) != kvImageNoError) {
        f16 *o = (f16 *)dst;
        for (size_t i = 0; i < n; i++) o[i] = (f16)src[i];
    }
}

/* f32 weight -> resident fp16 MTLBuffer (one-off conversion, per-pointer cache) */
static id<MTLBuffer> weight_buffer_f16(const float *w, size_t n_elems) {
    for (int i = 0; i < g_wc_n; i++)
        if (g_wc[i].host_ptr == w) return (__bridge id<MTLBuffer>)g_wc[i].buf;
    id<MTLBuffer> buf = [g_dev newBufferWithLength:n_elems * sizeof(f16)
                                           options:MTLResourceStorageModeShared];
    if (!buf) return nil;
    cvt_f32_to_f16(w, buf.contents, n_elems);
    if (g_wc_n == g_wc_cap) {
        const int cap = g_wc_cap ? g_wc_cap * 2 : 256;
        wc_ent *wc = realloc(g_wc, (size_t)cap * sizeof(wc_ent));
        if (!wc) return buf;      /* niente cache: il buffer resta valido per la chiamata */
        g_wc = wc;
        g_wc_cap = cap;
    }
    g_wc[g_wc_n++] = (wc_ent){.host_ptr = w, .buf = (void *)CFBridgingRetain(buf)};
    return buf;
}

/* Drop the resident weight cache. Call it when the host pointers stop being valid
 * (mynah_asr_free: the safetensors are munmap'd and a later load may reuse the
 * same virtual addresses -> the GPU would use the OLD weights).
 * Subsequent forwards re-upload the weights on the first call. */
void mynah_asr_metal_weights_evict(void) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_wc_n; i++) CFBridgingRelease(g_wc[i].buf);
    free(g_wc);
    g_wc = NULL;
    g_wc_n = g_wc_cap = 0;
    pthread_mutex_unlock(&g_mu);
}

static id<MTLBuffer> io_buffer(id<MTLBuffer> __strong *slot, size_t *cap, size_t bytes) {
    if (*slot == nil || *cap < bytes) {
        size_t want = *cap ? *cap : (1u << 20);
        while (want < bytes) want *= 2;
        *slot = [g_dev newBufferWithLength:want options:MTLResourceStorageModeShared];
        *cap = want;
    }
    return *slot;
}

static void f32_to_f16(const float *src, id<MTLBuffer> dst, size_t n) {
    cvt_f32_to_f16(src, dst.contents, n);
}

static void f16_to_f32(id<MTLBuffer> src, float *dst, size_t n) {
    vImage_Buffer s = {.data = src.contents, .height = 1, .width = n, .rowBytes = n * 2};
    vImage_Buffer d = {.data = dst, .height = 1, .width = n, .rowBytes = n * 4};
    if (vImageConvert_Planar16FtoPlanarF(&s, &d, kvImageNoFlags) != kvImageNoError) {
        const f16 *p = (const f16 *)src.contents;
        for (size_t i = 0; i < n; i++) dst[i] = (float)p[i];
    }
}

static MPSMatrix *mat16(id<MTLBuffer> buf, int rows, int cols) {
    MPSMatrixDescriptor *d = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)rows columns:(NSUInteger)cols
                        rowBytes:(NSUInteger)cols * sizeof(f16)
                        dataType:MPSDataTypeFloat16];
    return [[MPSMatrix alloc] initWithBuffer:buf descriptor:d];
}

/* strided MPS view (a per-head slice, or a slot inside a scratch buffer) */
static MPSMatrix *mat16_off(id<MTLBuffer> buf, size_t byte_off, int rows, int cols,
                            int row_stride_elems) {
    MPSMatrixDescriptor *d = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)rows columns:(NSUInteger)cols
                        rowBytes:(NSUInteger)row_stride_elems * sizeof(f16)
                        dataType:MPSDataTypeFloat16];
    return [[MPSMatrix alloc] initWithBuffer:buf offset:byte_off descriptor:d];
}

/* Cache of MPSMatrixMultiplication kernels keyed by shape (T,n,k,transB): the
 * shapes repeat identically for every layer and every forward of the same length
 * — without the cache one MPS object was allocated for EVERY encoded GEMM
 * (hundreds per forward). Protected by g_mu (already held while encoding). */
typedef struct { int T, n, k, tb; void *mm; } mm_ent;
static mm_ent g_mmc[512];
static int g_mmc_n;

static MPSMatrixMultiplication *mm_cached(int T, int n, int k, int transB) {
    for (int i = 0; i < g_mmc_n; i++)
        if (g_mmc[i].T == T && g_mmc[i].n == n && g_mmc[i].k == k && g_mmc[i].tb == transB)
            return (__bridge MPSMatrixMultiplication *)g_mmc[i].mm;
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_dev transposeLeft:NO transposeRight:(transB ? YES : NO)
            resultRows:(NSUInteger)T resultColumns:(NSUInteger)n
       interiorColumns:(NSUInteger)k alpha:1.0 beta:0.0];
    if (g_mmc_n < (int)(sizeof(g_mmc) / sizeof(g_mmc[0])))
        g_mmc[g_mmc_n++] = (mm_ent){T, n, k, transB, (void *)CFBridgingRetain(mm)};
    return mm;
}

static void encode_gemm(id<MTLCommandBuffer> cb, MPSMatrix *a, MPSMatrix *b,
                        MPSMatrix *c, int T, int n, int k) {
    [mm_cached(T, n, k, 1) encodeToCommandBuffer:cb leftMatrix:a rightMatrix:b
                                    resultMatrix:c];
}

static void encode_silu(id<MTLCommandBuffer> cb, id<MTLBuffer> buf, size_t n) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_silu];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(n, g_silu.maxTotalThreadsPerThreadgroup), 1, 1)];
    [enc endEncoding];
}

/* out[T,n] = x[T,k] @ W^T — a single fp16 GEMM */
int mynah_asr_metal_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k) {
    if (T < MYNAH_ASR_METAL_MIN_T || !mynah_asr_metal_available()) return -1;
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    @autoreleasepool {
        id<MTLBuffer> wb = weight_buffer_f16(w, (size_t)n * (size_t)k);
        id<MTLBuffer> xb = io_buffer(&g_in, &g_in_cap, (size_t)T * (size_t)k * sizeof(f16));
        id<MTLBuffer> ob = io_buffer(&g_out, &g_out_cap, (size_t)T * (size_t)n * sizeof(f16));
        if (wb && xb && ob) {
            f32_to_f16(x, xb, (size_t)T * (size_t)k);
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            encode_gemm(cb, mat16(xb, T, k), mat16(wb, n, k), mat16(ob, T, n), T, n, k);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusCompleted) {
                f16_to_f32(ob, out, (size_t)T * (size_t)n);
                rc = 0;
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}

/* fused FFN: out[T,n2] = SiLU(x[T,k] @ W1^T) @ W2^T — a single wait */
int mynah_asr_metal_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                       float *out, int T, int k) {
    if (T < MYNAH_ASR_METAL_MIN_T || !mynah_asr_metal_available()) return -1;
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    @autoreleasepool {
        id<MTLBuffer> w1b = weight_buffer_f16(w1, (size_t)n1 * (size_t)k);
        id<MTLBuffer> w2b = weight_buffer_f16(w2, (size_t)n2 * (size_t)n1);
        id<MTLBuffer> xb = io_buffer(&g_in, &g_in_cap, (size_t)T * (size_t)k * sizeof(f16));
        id<MTLBuffer> mb = io_buffer(&g_mid, &g_mid_cap, (size_t)T * (size_t)n1 * sizeof(f16));
        id<MTLBuffer> ob = io_buffer(&g_out, &g_out_cap, (size_t)T * (size_t)n2 * sizeof(f16));
        if (w1b && w2b && xb && mb && ob) {
            f32_to_f16(x, xb, (size_t)T * (size_t)k);
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            encode_gemm(cb, mat16(xb, T, k), mat16(w1b, n1, k), mat16(mb, T, n1), T, n1, k);
            encode_silu(cb, mb, (size_t)T * (size_t)n1);
            encode_gemm(cb, mat16(mb, T, n1), mat16(w2b, n2, n1), mat16(ob, T, n2), T, n2, n1);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusCompleted) {
                f16_to_f32(ob, out, (size_t)T * (size_t)n2);
                rc = 0;
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}

/* QKV: three GEMMs over the same input, a single wait */
int mynah_asr_metal_gemm3_wt(const float *x, const float *wa, const float *wb_,
                         const float *wc_, float *oa, float *ob_, float *oc,
                         int T, int n, int k) {
    if (T < MYNAH_ASR_METAL_MIN_T || !mynah_asr_metal_available()) return -1;
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    @autoreleasepool {
        id<MTLBuffer> ba = weight_buffer_f16(wa, (size_t)n * (size_t)k);
        id<MTLBuffer> bb = weight_buffer_f16(wb_, (size_t)n * (size_t)k);
        id<MTLBuffer> bc = weight_buffer_f16(wc_, (size_t)n * (size_t)k);
        id<MTLBuffer> xb = io_buffer(&g_in, &g_in_cap, (size_t)T * (size_t)k * sizeof(f16));
        id<MTLBuffer> o1 = io_buffer(&g_out, &g_out_cap, (size_t)T * (size_t)n * sizeof(f16));
        id<MTLBuffer> o2 = io_buffer(&g_out2, &g_out2_cap, (size_t)T * (size_t)n * sizeof(f16));
        id<MTLBuffer> o3 = io_buffer(&g_out3, &g_out3_cap, (size_t)T * (size_t)n * sizeof(f16));
        if (ba && bb && bc && xb && o1 && o2 && o3) {
            f32_to_f16(x, xb, (size_t)T * (size_t)k);
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            MPSMatrix *mx = mat16(xb, T, k);
            encode_gemm(cb, mx, mat16(ba, n, k), mat16(o1, T, n), T, n, k);
            encode_gemm(cb, mx, mat16(bb, n, k), mat16(o2, T, n), T, n, k);
            encode_gemm(cb, mx, mat16(bc, n, k), mat16(o3, T, n), T, n, k);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusCompleted) {
                f16_to_f32(o1, oa, (size_t)T * (size_t)n);
                f16_to_f32(o2, ob_, (size_t)T * (size_t)n);
                f16_to_f32(o3, oc, (size_t)T * (size_t)n);
                rc = 0;
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}

/* ------------------------------------------------------------ v4: encoder */

/* helper: 1D dispatch of an already-configured compute kernel */
static void run1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, size_t n) {
    [enc setComputePipelineState:ps];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(n, ps.maxTotalThreadsPerThreadgroup), 1, 1)];
}

/* LN from the f32 residual: to16 -> writes f16 into o (the block input);
 * otherwise in-place f32 (norm_out). gm/bt are resident f16. */
static void encode_ln32(id<MTLCommandBuffer> cb, id<MTLBuffer> x32, id<MTLBuffer> o16,
                        id<MTLBuffer> gm, id<MTLBuffer> bt, int T, int d) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    const uint32_t du = (uint32_t)d;
    if (o16) {
        [enc setComputePipelineState:g_ln32h];
        [enc setBuffer:x32 offset:0 atIndex:0];
        [enc setBuffer:o16 offset:0 atIndex:1];
        [enc setBuffer:gm offset:0 atIndex:2];
        [enc setBuffer:bt offset:0 atIndex:3];
        [enc setBytes:&du length:4 atIndex:4];
    } else {
        [enc setComputePipelineState:g_ln32f];
        [enc setBuffer:x32 offset:0 atIndex:0];
        [enc setBuffer:gm offset:0 atIndex:1];
        [enc setBuffer:bt offset:0 atIndex:2];
        [enc setBytes:&du length:4 atIndex:3];
    }
    /* one threadgroup per row (coalesced reads) */
    [enc dispatchThreadgroups:MTLSizeMake((size_t)T, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

/* x32 += alpha * blk16 */
static void encode_resadd(id<MTLCommandBuffer> cb, id<MTLBuffer> x32, id<MTLBuffer> y16,
                          float alpha, size_t n) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_resadd];
    [enc setBuffer:x32 offset:0 atIndex:0];
    [enc setBuffer:y16 offset:0 atIndex:1];
    [enc setBytes:&alpha length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

/* bias broadcast over rows, in place: buf[t*d+i] += bias[i] (no-op when bias is NULL) */
static void encode_addbias(id<MTLCommandBuffer> cb, id<MTLBuffer> buf, size_t byte_off,
                           const float *bias, int T, int d) {
    if (!bias) return;
    id<MTLBuffer> bb = weight_buffer_f16(bias, (size_t)d);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    const uint32_t du = (uint32_t)d;
    [enc setComputePipelineState:g_addbias];
    [enc setBuffer:buf offset:byte_off atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:buf offset:byte_off atIndex:2];
    [enc setBytes:&du length:4 atIndex:3];
    [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

/* fp16 FFN: out16 = SiLU(xn @ W1^T + b1) @ W2^T + b2 (intermediate in g_mid) */
static void encode_ffn16(id<MTLCommandBuffer> cb, id<MTLBuffer> xn, id<MTLBuffer> out16,
                         id<MTLBuffer> w1, id<MTLBuffer> w2, const float *b1,
                         const float *b2, int T, int k, int ffn) {
    encode_gemm(cb, mat16(xn, T, k), mat16(w1, ffn, k), mat16(g_mid, T, ffn), T, ffn, k);
    encode_addbias(cb, g_mid, 0, b1, T, ffn);
    encode_silu(cb, g_mid, (size_t)T * (size_t)ffn);
    encode_gemm(cb, mat16(g_mid, T, ffn), mat16(w2, k, ffn), mat16(out16, T, k), T, k, ffn);
    encode_addbias(cb, out16, 0, b2, T, k);
}

/* fp16 attention as in v3: qkv -> +bias_u/v -> rk = pe@relk^T -> per-head ac/bd
 * (GEMMs over strided views) -> softmax+rel_shift+window -> ctx -> o_proj.
 * Scratch in g_att: q,k,v,qu,qv [5*td] ++ rk [pd] ++ ctx [td] ++ scores ++ bd. */
static void encode_att16(id<MTLCommandBuffer> cb, id<MTLBuffer> xn, id<MTLBuffer> out16,
                         const mynah_asr_metal_layer_w *L, int T, int d, int H,
                         int left, int right) {
    const int dk = d / H, P = 2 * T - 1, K = T;
    const size_t td = (size_t)T * (size_t)d * sizeof(f16);
    const size_t pd = (size_t)P * (size_t)d * sizeof(f16);
    const size_t rk_off = 5 * td, ctx_off = rk_off + pd;
    const size_t sc_off = ctx_off + td;
    const size_t bd_off = sc_off + (size_t)H * (size_t)T * (size_t)K * sizeof(f16);

    id<MTLBuffer> bwq = weight_buffer_f16(L->wq, (size_t)d * (size_t)d);
    id<MTLBuffer> bwk = weight_buffer_f16(L->wk, (size_t)d * (size_t)d);
    id<MTLBuffer> bwv = weight_buffer_f16(L->wv, (size_t)d * (size_t)d);
    id<MTLBuffer> bwo = weight_buffer_f16(L->wo, (size_t)d * (size_t)d);
    id<MTLBuffer> brel = weight_buffer_f16(L->relk, (size_t)d * (size_t)d);
    id<MTLBuffer> bbu = weight_buffer_f16(L->bias_u, (size_t)d);
    id<MTLBuffer> bbv = weight_buffer_f16(L->bias_v, (size_t)d);
    id<MTLBuffer> sb = g_att;

    MPSMatrix *mx = mat16(xn, T, d);
    encode_gemm(cb, mx, mat16(bwq, d, d), mat16_off(sb, 0 * td, T, d, d), T, d, d);
    encode_gemm(cb, mx, mat16(bwk, d, d), mat16_off(sb, 1 * td, T, d, d), T, d, d);
    encode_gemm(cb, mx, mat16(bwv, d, d), mat16_off(sb, 2 * td, T, d, d), T, d, d);
    encode_addbias(cb, sb, 0 * td, L->q_b, T, d);
    encode_addbias(cb, sb, 1 * td, L->k_b, T, d);
    encode_addbias(cb, sb, 2 * td, L->v_b, T, d);
    encode_gemm(cb, mat16(g_pe16, P, d), mat16(brel, d, d),
                mat16_off(sb, rk_off, P, d, d), P, d, d);

    {   /* qu = q + bias_u ; qv = q + bias_v */
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const uint32_t du = (uint32_t)d;
        [enc setComputePipelineState:g_addbias];
        [enc setBuffer:sb offset:0 atIndex:0];
        [enc setBuffer:bbu offset:0 atIndex:1];
        [enc setBuffer:sb offset:3 * td atIndex:2];
        [enc setBytes:&du length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setBuffer:bbv offset:0 atIndex:1];
        [enc setBuffer:sb offset:4 * td atIndex:2];
        [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    /* per head: ac[h] = qu_h @ k_h^T ; bd[h] = qv_h @ rk_h^T (strided views) */
    for (int h = 0; h < H; h++) {
        const size_t ho = (size_t)h * (size_t)dk * sizeof(f16);
        MPSMatrix *quh = mat16_off(sb, 3 * td + ho, T, dk, d);
        MPSMatrix *qvh = mat16_off(sb, 4 * td + ho, T, dk, d);
        MPSMatrix *kh = mat16_off(sb, 1 * td + ho, T, dk, d);
        MPSMatrix *rkh = mat16_off(sb, rk_off + ho, P, dk, d);
        MPSMatrix *ach = mat16_off(sb, sc_off + (size_t)h * (size_t)T * (size_t)K * sizeof(f16), T, K, K);
        MPSMatrix *bdh = mat16_off(sb, bd_off + (size_t)h * (size_t)T * (size_t)P * sizeof(f16), T, P, P);
        encode_gemm(cb, quh, kh, ach, T, K, dk);
        encode_gemm(cb, qvh, rkh, bdh, T, P, dk);
    }

    {   /* softmax + rel_shift + finestra */
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const uint32_t Tu = (uint32_t)T, Ku = (uint32_t)K, Pu = (uint32_t)P;
        /* left < 0: full attention -> chunk 0 (whole window inside the shader) */
        const uint32_t chunk = left >= 0 ? (uint32_t)(right + 1) : 0u;
        const uint32_t lc = left >= 0 ? (uint32_t)(left / (right + 1)) : 0u;
        const float scale = 1.0f / sqrtf((float)dk);
        [enc setComputePipelineState:g_smax];
        [enc setBuffer:sb offset:sc_off atIndex:0];
        [enc setBuffer:sb offset:bd_off atIndex:1];
        [enc setBytes:&Tu length:4 atIndex:2];
        [enc setBytes:&Ku length:4 atIndex:3];
        [enc setBytes:&Pu length:4 atIndex:4];
        [enc setBytes:&scale length:4 atIndex:5];
        [enc setBytes:&chunk length:4 atIndex:6];
        [enc setBytes:&lc length:4 atIndex:7];
        [enc dispatchThreads:MTLSizeMake((size_t)H * (size_t)T, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
    }

    /* ctx_h = probs_h @ v_h (output into a strided [T,dk] view inside [T,d]) */
    for (int h = 0; h < H; h++) {
        const size_t ho = (size_t)h * (size_t)dk * sizeof(f16);
        MPSMatrix *ph = mat16_off(sb, sc_off + (size_t)h * (size_t)T * (size_t)K * sizeof(f16), T, K, K);
        MPSMatrix *vh = mat16_off(sb, 2 * td + ho, T, dk, d);
        MPSMatrix *ch = mat16_off(sb, ctx_off + ho, T, dk, d);
        [mm_cached(T, dk, K, 0) encodeToCommandBuffer:cb leftMatrix:ph rightMatrix:vh
                                         resultMatrix:ch];
    }

    /* out = ctx @ wo^T (+ o_b) */
    encode_gemm(cb, mat16_off(sb, ctx_off, T, d, d), mat16(bwo, d, d),
                mat16(out16, T, d), T, d, d);
    encode_addbias(cb, out16, 0, L->o_b, T, d);
}

/* fp16 conv module as in v3: pw1(+b) -> GLU -> dwconv9(+b, causal or 'same' pad)
 * -> LN | BN-affine -> SiLU -> pw2(+b).
 * Scratch in g_mid: h2 [T,2d] @0, g [T,d] @2td, c [T,d] @3td. */
static void encode_conv16(id<MTLCommandBuffer> cb, id<MTLBuffer> xn, id<MTLBuffer> out16,
                          const mynah_asr_metal_layer_w *L, int T, int d, int conv_pad) {
    const size_t td = (size_t)T * (size_t)d * sizeof(f16);
    id<MTLBuffer> b1 = weight_buffer_f16(L->pw1, (size_t)2 * (size_t)d * (size_t)d);
    id<MTLBuffer> bw = weight_buffer_f16(L->dw9, (size_t)d * 9u);
    id<MTLBuffer> bg = L->cnorm_scale ? weight_buffer_f16(L->cnorm_scale, (size_t)d)
                                      : weight_buffer_f16(L->cnorm_w, (size_t)d);
    id<MTLBuffer> bb = L->cnorm_scale ? weight_buffer_f16(L->cnorm_shift, (size_t)d)
                                      : weight_buffer_f16(L->cnorm_b, (size_t)d);
    id<MTLBuffer> b2 = weight_buffer_f16(L->pw2, (size_t)d * (size_t)d);
    /* optional dwconv bias: when absent bw is bound instead (never read, has_b=0) */
    id<MTLBuffer> bdw = L->dw_b ? weight_buffer_f16(L->dw_b, (size_t)d) : bw;
    id<MTLBuffer> mb = g_mid;

    const uint32_t du = (uint32_t)d, Tu = (uint32_t)T;
    const uint32_t pad = (uint32_t)conv_pad, has_dwb = L->dw_b ? 1u : 0u;
    encode_gemm(cb, mat16(xn, T, d), mat16(b1, 2 * d, d), mat16(mb, T, 2 * d), T, 2 * d, d);
    encode_addbias(cb, mb, 0, L->pw1_b, T, 2 * d);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    /* GLU: mb[0..2td) -> mb[2td..3td) */
    [enc setComputePipelineState:g_glu];
    [enc setBuffer:mb offset:0 atIndex:0];
    [enc setBuffer:mb offset:2 * td atIndex:1];
    [enc setBytes:&du length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    /* dwconv: g -> c */
    [enc setComputePipelineState:g_dwconv];
    [enc setBuffer:mb offset:2 * td atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:mb offset:3 * td atIndex:2];
    [enc setBytes:&du length:4 atIndex:3];
    [enc setBytes:&Tu length:4 atIndex:4];
    [enc setBytes:&pad length:4 atIndex:5];
    [enc setBuffer:bdw offset:0 atIndex:6];
    [enc setBytes:&has_dwb length:4 atIndex:7];
    [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (L->cnorm_scale) {
        /* folded BatchNorm: per-channel affine in place on c */
        [enc setComputePipelineState:g_caffine];
        [enc setBuffer:mb offset:3 * td atIndex:0];
        [enc setBuffer:bg offset:0 atIndex:1];
        [enc setBuffer:bb offset:0 atIndex:2];
        [enc setBytes:&du length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        /* in-place LN on c (one threadgroup per row) */
        [enc setComputePipelineState:g_lnorm];
        [enc setBuffer:mb offset:3 * td atIndex:0];
        [enc setBuffer:bg offset:0 atIndex:1];
        [enc setBuffer:bb offset:0 atIndex:2];
        [enc setBytes:&du length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((size_t)T, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc setComputePipelineState:g_silu];
    [enc setBuffer:mb offset:3 * td atIndex:0];
    [enc dispatchThreads:MTLSizeMake((size_t)T * (size_t)d, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    encode_gemm(cb, mat16_off(mb, 3 * td, T, d, d), mat16(b2, d, d), mat16(out16, T, d), T, d, d);
    encode_addbias(cb, out16, 0, L->pw2_b, T, d);
}

/* Whole encoder on GPU: x [T,d] f32 updated in place, a single final wait.
 * Per layer: LN->FFN1(+0.5) -> LN->MHSA(+1) -> LN->conv(+1) -> LN->FFN2(+0.5)
 * -> output LN. f32 residual (like the CPU), f16 blocks (like v3). */
int mynah_asr_metal_encoder_layers(const mynah_asr_metal_layer_w *Ls, int n_layers,
                               float *x, const float *pe, int T, int d, int H,
                               int ffn, int left, int right, int conv_pad) {
    if (T < MYNAH_ASR_METAL_MIN_T || n_layers <= 0 || !mynah_asr_metal_available()) return -1;
    const int P = 2 * T - 1;
    const double tf0 = CFAbsoluteTimeGetCurrent();
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    @autoreleasepool {
        const size_t nd = (size_t)T * (size_t)d;
        const size_t td = nd * sizeof(f16);
        const size_t pd = (size_t)P * (size_t)d * sizeof(f16);
        const int conv_cols = 4 * d;                    /* h2 + g + c */
        const size_t mid_bytes = (size_t)T * (size_t)(ffn > conv_cols ? ffn : conv_cols)
                                 * sizeof(f16);
        const size_t att_bytes = 6 * td + pd +
            (size_t)H * (size_t)T * ((size_t)T + (size_t)P) * sizeof(f16);

        id<MTLBuffer> x32 = io_buffer(&g_x32, &g_x32_cap, nd * sizeof(float));
        id<MTLBuffer> pe32 = io_buffer(&g_pe32, &g_pe32_cap, (size_t)P * (size_t)d * sizeof(float));
        id<MTLBuffer> pe16 = io_buffer(&g_pe16, &g_pe16_cap, pd);
        id<MTLBuffer> xn = io_buffer(&g_xn, &g_xn_cap, td);
        id<MTLBuffer> blk = io_buffer(&g_blk, &g_blk_cap, td);
        id<MTLBuffer> mid = io_buffer(&g_mid, &g_mid_cap, mid_bytes);
        id<MTLBuffer> att = io_buffer(&g_att, &g_att_cap, att_bytes);
        int ok = (x32 && pe32 && pe16 && xn && blk && mid && att);

        /* the weights must be cached BEFORE encoding (weight_buffer_f16 can fail) */
        for (int li = 0; ok && li < n_layers; li++) {
            const mynah_asr_metal_layer_w *L = &Ls[li];
            const float *big[] = {L->ff1_w1, L->ff1_w2, L->ff2_w1, L->ff2_w2,
                                  L->wq, L->wk, L->wv, L->wo, L->relk, L->pw1, L->pw2};
            const size_t bign[] = {(size_t)ffn * (size_t)d, (size_t)d * (size_t)ffn,
                                   (size_t)ffn * (size_t)d, (size_t)d * (size_t)ffn,
                                   (size_t)d * (size_t)d, (size_t)d * (size_t)d,
                                   (size_t)d * (size_t)d, (size_t)d * (size_t)d,
                                   (size_t)d * (size_t)d, 2 * (size_t)d * (size_t)d,
                                   (size_t)d * (size_t)d};
            for (int i = 0; i < 11 && ok; i++) ok = weight_buffer_f16(big[i], bign[i]) != nil;
            const float *small[] = {L->ln_ff1_w, L->ln_ff1_b, L->ln_att_w, L->ln_att_b,
                                    L->ln_conv_w, L->ln_conv_b,
                                    L->cnorm_scale ? L->cnorm_scale : L->cnorm_w,
                                    L->cnorm_scale ? L->cnorm_shift : L->cnorm_b,
                                    L->ln_ff2_w, L->ln_ff2_b, L->ln_out_w, L->ln_out_b,
                                    L->bias_u, L->bias_v};
            for (int i = 0; i < 14 && ok; i++) ok = weight_buffer_f16(small[i], (size_t)d) != nil;
            ok = ok && weight_buffer_f16(L->dw9, (size_t)d * 9u) != nil;
            /* optional biases (use_bias true) */
            const float *ob[] = {L->q_b, L->k_b, L->v_b, L->o_b, L->dw_b, L->pw2_b};
            for (int i = 0; i < 6 && ok; i++)
                if (ob[i]) ok = weight_buffer_f16(ob[i], (size_t)d) != nil;
            if (ok && L->pw1_b) ok = weight_buffer_f16(L->pw1_b, 2u * (size_t)d) != nil;
            if (ok && L->ff1_b1) ok = weight_buffer_f16(L->ff1_b1, (size_t)ffn) != nil;
            if (ok && L->ff1_b2) ok = weight_buffer_f16(L->ff1_b2, (size_t)d) != nil;
            if (ok && L->ff2_b1) ok = weight_buffer_f16(L->ff2_b1, (size_t)ffn) != nil;
            if (ok && L->ff2_b2) ok = weight_buffer_f16(L->ff2_b2, (size_t)d) != nil;
        }

        if (ok) {
            memcpy(x32.contents, x, nd * sizeof(float));
            memcpy(pe32.contents, pe, (size_t)P * (size_t)d * sizeof(float));

            const double t0 = CFAbsoluteTimeGetCurrent();
            id<MTLCommandBuffer> last = nil;
            for (int li = 0; li < n_layers; li++) {
                const mynah_asr_metal_layer_w *L = &Ls[li];
                id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                if (li == 0) {   /* pe f32 -> f16 una volta per forward */
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setBuffer:pe32 offset:0 atIndex:0];
                    [enc setBuffer:pe16 offset:0 atIndex:1];
                    run1d(enc, g_cvt, (size_t)P * (size_t)d);
                    [enc endEncoding];
                }
                #define LNW(f) weight_buffer_f16(L->f, (size_t)d)
                /* ½ FFN1 */
                encode_ln32(cb, x32, xn, LNW(ln_ff1_w), LNW(ln_ff1_b), T, d);
                encode_ffn16(cb, xn, blk, weight_buffer_f16(L->ff1_w1, (size_t)ffn * (size_t)d),
                             weight_buffer_f16(L->ff1_w2, (size_t)d * (size_t)ffn),
                             L->ff1_b1, L->ff1_b2, T, d, ffn);
                encode_resadd(cb, x32, blk, 0.5f, nd);
                /* MHSA */
                encode_ln32(cb, x32, xn, LNW(ln_att_w), LNW(ln_att_b), T, d);
                encode_att16(cb, xn, blk, L, T, d, H, left, right);
                encode_resadd(cb, x32, blk, 1.0f, nd);
                /* Conv */
                encode_ln32(cb, x32, xn, LNW(ln_conv_w), LNW(ln_conv_b), T, d);
                encode_conv16(cb, xn, blk, L, T, d, conv_pad);
                encode_resadd(cb, x32, blk, 1.0f, nd);
                /* ½ FFN2 + output LN (in place, f32) */
                encode_ln32(cb, x32, xn, LNW(ln_ff2_w), LNW(ln_ff2_b), T, d);
                encode_ffn16(cb, xn, blk, weight_buffer_f16(L->ff2_w1, (size_t)ffn * (size_t)d),
                             weight_buffer_f16(L->ff2_w2, (size_t)d * (size_t)ffn),
                             L->ff2_b1, L->ff2_b2, T, d, ffn);
                encode_resadd(cb, x32, blk, 0.5f, nd);
                encode_ln32(cb, x32, nil, LNW(ln_out_w), LNW(ln_out_b), T, d);
                #undef LNW
                [cb commit];        /* niente wait: la GPU parte, la CPU encoda il prossimo */
                last = cb;
            }
            const double t1 = CFAbsoluteTimeGetCurrent();
            [last waitUntilCompleted];
            if (getenv("MYNAH_ASR_METAL_PROF"))
                fprintf(stderr, "metal v4: T=%d encode %.3fs wait %.3fs gpu(last) %.3fs\n",
                        T, t1 - t0, CFAbsoluteTimeGetCurrent() - t1,
                        last.GPUEndTime - last.GPUStartTime);
            if (last.status == MTLCommandBufferStatusCompleted) {
                memcpy(x, x32.contents, nd * sizeof(float));
                rc = 0;
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
    if (getenv("MYNAH_ASR_METAL_PROF"))
        fprintf(stderr, "metal v4: funzione intera %.3fs\n",
                CFAbsoluteTimeGetCurrent() - tf0);
    return rc;
}

#endif /* MYNAH_ASR_METAL */
