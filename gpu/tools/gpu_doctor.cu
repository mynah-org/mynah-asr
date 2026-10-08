/* gpu/tools/gpu_doctor.cu — does this (rented, containerised) GPU deliver what
 * its name promises? Run it on a new box BEFORE any number is taken there.
 *
 * Measures, with the device otherwise idle:
 *   - VRAM: total and free as the driver reports them, and how much can
 *     ACTUALLY be allocated (a provider cap or another tenant shows here);
 *   - device memory bandwidth: a read+write copy kernel over 1 GiB, best of 20;
 *   - FP32 throughput: a register-resident FMA chain, best of 10;
 *   - host<->device bandwidth, pinned, 256 MiB each way.
 * The clocks, P-state, power cap and throttle reasons are read while the
 * compute kernel runs by `nvidia-smi` from the wrapper script
 * (gpu/tools/gpu_doctor.sh), because NVML is not linked here.
 *
 * It prints the measured values next to the datasheet peak it is given with
 * --peak-bw-gbs and --peak-tflops, and a verdict per line. It never decides for
 * the operator; a ratio below the stated floor is flagged SUSPECT. */
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "cuda: %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__global__ void copy_kernel(const float4 *__restrict__ a, float4 *__restrict__ b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) b[i] = a[i];
}

__global__ void fma_kernel(float *out, int iters, float seed) {
    float a0 = seed + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    const float m = 0.999999f, c = 1e-7f;
    for (int i = 0; i < iters; i++) {
        a0 = fmaf(a0, m, c); a1 = fmaf(a1, m, c); a2 = fmaf(a2, m, c); a3 = fmaf(a3, m, c);
        a4 = fmaf(a4, m, c); a5 = fmaf(a5, m, c); a6 = fmaf(a6, m, c); a7 = fmaf(a7, m, c);
    }
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

static float time_ms(cudaEvent_t a, cudaEvent_t b) { float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms; }

int main(int argc, char **argv) {
    double peak_bw = 0.0, peak_tf = 0.0, burn_s = 0.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--peak-bw-gbs") && i + 1 < argc) peak_bw = atof(argv[++i]);
        else if (!strcmp(argv[i], "--peak-tflops") && i + 1 < argc) peak_tf = atof(argv[++i]);
        else if (!strcmp(argv[i], "--burn-seconds") && i + 1 < argc) burn_s = atof(argv[++i]);
    }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { printf("no CUDA device\n"); return 77; }
    cudaDeviceProp p;
    CU(cudaGetDeviceProperties(&p, 0));
    printf("device            %s, sm_%d%d, %d SMs, %.0f MiB, ECC %s, bus %d-bit\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.totalGlobalMem / 1048576.0, p.ECCEnabled ? "on" : "off", p.memoryBusWidth);
    int clk = 0, mclk = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0);
    printf("attr clocks       sm %.0f MHz (max), mem %.0f MHz\n", clk / 1000.0, mclk / 1000.0);

    size_t freeb = 0, totb = 0;
    CU(cudaMemGetInfo(&freeb, &totb));
    printf("vram (driver)     total %.0f MiB, free %.0f MiB, used by others %.0f MiB\n", totb / 1048576.0,
           freeb / 1048576.0, (totb - freeb) / 1048576.0);
    /* how much can really be allocated, in 256 MiB steps */
    {
        void *blk[512]; int k = 0; size_t got = 0;
        while (k < 512) { if (cudaMalloc(&blk[k], 256u << 20) != cudaSuccess) break; got += 256u << 20; k++; }
        cudaGetLastError();
        for (int i = 0; i < k; i++) cudaFree(blk[i]);
        printf("vram (allocated)  %.0f MiB in 256 MiB blocks (%.1f%% of total)  %s\n", got / 1048576.0,
               100.0 * got / totb, got >= 0.9 * totb ? "OK" : "SUSPECT: less than 90% allocatable");
    }

    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));

    /* device bandwidth */
    {
        const size_t bytes = 1ull << 30, nv = bytes / sizeof(float4);
        float4 *a, *b;
        CU(cudaMalloc(&a, bytes)); CU(cudaMalloc(&b, bytes));
        CU(cudaMemset(a, 1, bytes));
        const int blocks = p.multiProcessorCount * 8;
        copy_kernel<<<blocks, 256>>>(a, b, nv);
        CU(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 20; r++) {
            cudaEventRecord(e0);
            copy_kernel<<<blocks, 256>>>(a, b, nv);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            const float ms = time_ms(e0, e1); if (ms < best) best = ms;
        }
        const double gbs = 2.0 * bytes / (best / 1e3) / 1e9;
        printf("device bandwidth  %.1f GB/s (read+write, best of 20)", gbs);
        if (peak_bw > 0) printf("  = %.0f%% of %.0f GB/s peak  %s", 100.0 * gbs / peak_bw, peak_bw,
                                gbs >= 0.70 * peak_bw ? "OK" : "SUSPECT: below 70% of peak");
        printf("\n");
        cudaFree(a); cudaFree(b);
    }

    /* FP32 */
    {
        const int blocks = p.multiProcessorCount * 16, threads = 256, iters = 1 << 16;
        float *out;
        CU(cudaMalloc(&out, (size_t)blocks * threads * sizeof(float)));
        fma_kernel<<<blocks, threads>>>(out, 1024, 1.0f);
        CU(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 10; r++) {
            cudaEventRecord(e0);
            fma_kernel<<<blocks, threads>>>(out, iters, 1.0f);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            const float ms = time_ms(e0, e1); if (ms < best) best = ms;
        }
        const double flops = 2.0 * 8.0 * (double)iters * blocks * threads;
        const double tf = flops / (best / 1e3) / 1e12;
        printf("fp32 fma          %.2f TFLOP/s (best of 10)", tf);
        if (peak_tf > 0) printf("  = %.0f%% of %.1f TFLOP/s peak  %s", 100.0 * tf / peak_tf, peak_tf,
                                tf >= 0.80 * peak_tf ? "OK" : "SUSPECT: below 80% of peak");
        printf("\n");
        /* a sustained burn so the wrapper can sample clocks, power and throttle */
        if (burn_s > 0) {
            printf("burn              %.0f s of fma for the clock/power sampler...\n", burn_s);
            fflush(stdout);
            cudaEventRecord(e0);
            float el = 0;
            while (el < burn_s * 1e3) {
                fma_kernel<<<blocks, threads>>>(out, iters, 1.0f);
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                el = time_ms(e0, e1);
            }
            printf("burn done\n");
        }
        cudaFree(out);
    }

    /* host <-> device, pinned */
    {
        const size_t bytes = 256u << 20;
        void *h, *d;
        CU(cudaHostAlloc(&h, bytes, cudaHostAllocPortable)); CU(cudaMalloc(&d, bytes));
        memset(h, 1, bytes);
        float bh = 1e30f, bd = 1e30f;
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(e0); cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = time_ms(e0, e1); if (ms < bh) bh = ms;
            cudaEventRecord(e0); cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost); cudaEventRecord(e1); cudaEventSynchronize(e1);
            ms = time_ms(e0, e1); if (ms < bd) bd = ms;
        }
        printf("pcie pinned       h2d %.1f GB/s, d2h %.1f GB/s\n", bytes / (bh / 1e3) / 1e9, bytes / (bd / 1e3) / 1e9);
        cudaFreeHost(h); cudaFree(d);
    }
    return 0;
}
