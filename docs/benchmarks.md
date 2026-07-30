# Benchmarks — measured RTF and RAM

> Measurements from **2026-07-18** on M-series (16 GB), models on local disk,
> **warm** runs (2 warm-ups before measuring — see the methodology note at the
> bottom). Reproducible with `make bench` (short fixtures) + manual runs on the
> long file. RTF = inference time / audio duration (lower = faster; 0.05 ≈ 20×
> faster than realtime).

## Short fixtures (4–5 s, `make bench`) — f32 CPU

| model | RTF | RAM |
|---|---|---|
| parakeet-tdt_ctc-110m | 0.022 | 0.44 GB |
| canary-180m-flash | 0.127 | 0.71 GB |
| parakeet-tdt-0.6b-v3 | 0.097 | 2.35 GB |
| nemotron-3.5-asr-streaming-0.6b | 0.113 | 2.38 GB |
| parakeet-rnnt-1.1b / ctc-1.1b | 0.19 | 4.0 GB |
| canary-1b-flash | 0.37 | 3.3 GB |

On short clips the fixed costs dominate (prompt/decoder warm-up): the real
picture is below, on long audio.

## Long audio (~65 s) — CPU vs Metal vs int8

| model | f32 CPU | Metal | int8 CPU |
|---|---|---|---|
| parakeet-tdt_ctc-110m | 0.015 | 0.010 | 0.015 |
| canary-180m-flash | 0.060 | 0.054 | **0.030** |
| parakeet-tdt-0.6b-v3 | 0.047 | **0.030** | 0.046 |
| nemotron-3.5-asr-streaming-0.6b | 0.055 | **0.040** | 0.054 |
| parakeet-rnnt-0.6b¹ | 0.050 | 0.028 | — |
| parakeet-ctc-0.6b¹ | 0.042 | 0.022 | — |
| parakeet-rnnt-1.1b¹ | 0.068 | 0.041 | — |
| parakeet-ctc-1.1b¹ | 0.062 | 0.033 | — |
| canary-1b-flash¹ | 0.143 | 0.081 | 0.133² |
| canary-1b-v2 | to be measured³ | — | — |

¹ measured before the move to the NAS (same day, same protocol).
² 4 s fixture (long run not re-measured); the AED int8/f32 ratio (~2×) applies
here as well.
³ encoder = 1b-flash (32L) + 8L decoder (2× the flash's): expected ~1.3-1.5× the
1b-flash. Measure with weights on local disk (the NAS numbers don't count).

Key takeaways:
- **Metal** wins on the encoder of every model (−25…45%); the gain shrinks
  where the decoder dominates (canary: the AED decode stays on CPU).
- **int8** is ~neutral on the offline encoder (dequant+sgemm on AMX matches f32)
  but **halves Canary's AED decode** (SDOT dot kernel on the T=1 path) and
  triples Nemotron streaming speed (weight bandwidth).
- int8 checkpoints on disk: 110m 0.15 GB · canary-180m 0.22 GB · 0.6b ~0.8 GB ·
  canary-1b 0.98 GB · 1.1b ~1.2 GB — zero-copy load ~instant.

## Linux x86 + CUDA — EPYC 7642 (22 vcpu, AVX2) / A100-SXM4-40GB — 2026-07-20

First hardware validation of `make cuda` (Ubuntu 24.04, CUDA 12.8, OpenBLAS).
**All 10 supported models + the 2 community GGUFs: e2e green, CPU and CUDA
transcripts byte-identical.** Same protocol as above (`mynah-asr bench`, f32,
2 warm-ups + 5 runs short / 1 + 3 long); long audio = LibriSpeech dev-clean
(CC-BY 4.0) concatenated to 60 s and 300 s, 16 kHz mono — not committed,
regenerate with `scripts/make_long_fixtures.sh`.
Raw TSVs in [bench/](bench/). CUDA backend = big-GEMM offload to cuBLAS
(T≥24), the rest stays on CPU.

### RTF (median, warm) — CPU (22-core AVX2) vs CUDA

| model | 5 s cpu | 5 s cuda | 60 s cpu | 60 s cuda | 300 s cpu | 300 s cuda |
|---|---|---|---|---|---|---|
| parakeet-tdt_ctc-110m | 0.016 | 0.011 | 0.015 | 0.011 | 0.015 | 0.011 |
| 110m **GGUF Q4** | 0.017 | 0.011 | 0.015 | 0.011 | — | — |
| parakeet-tdt-0.6b-v3 | 0.045 | 0.025 | 0.065 | 0.028 | 0.044 | 0.024 |
| tdt-v3 **GGUF Q4** | 0.047 | 0.026 | 0.067 | 0.031 | — | — |
| nemotron-3.5-0.6b¹ | 0.051 | 0.028 | 0.059 | 0.022 | 0.054 | **0.028** |
| parakeet-rnnt-0.6b | 0.049 | 0.026 | 0.043 | 0.023 | 0.042 | 0.024 |
| parakeet-ctc-0.6b | 0.047 | 0.026 | 0.040 | 0.022 | 0.040 | 0.023 |
| parakeet-rnnt-1.1b | 0.080 | 0.057 | 0.066 | 0.038 | 0.066 | 0.038 |
| parakeet-ctc-1.1b | 0.084 | 0.050 | 0.071 | 0.037 | 0.066 | 0.037 |
| canary-180m-flash | 0.060 | 0.051 | 0.046 | 0.044 | 0.045 | 0.043 |
| canary-1b-flash | 0.103 | 0.072 | 0.113 | 0.063 | 0.090 | 0.067 |
| canary-1b-v2 | 0.130 | 0.115 | 0.111 | 0.084 | 0.135 | 0.096 |

¹ after the banded-attention fix (below) and with the TF32 default (v2a),
re-measured last; pre-fix 300 s was 0.083 cpu / **0.052** cuda.
All other CUDA columns were measured with strict f32 **before** TF32 became
the default: expect them ~4–8% better on the GEMM-heavy models (the server
was returned before a full TF32 re-sweep — worth redoing on the next GPU
rental together with the M1 local-disk pass).

### Throughput — `tests/bench_throughput` (batch API, same 60 s wav ×B)

| | B=1 | B=4 | B=16 | B=64 |
|---|---|---|---|---|
| nemotron cuda | 45× | 24× | 42× | **46×** |
| nemotron cpu | 16× | 14× | 28× | — |
| 110m cuda (4.3 s wav) | 87× | 85× | 86× | **94×** |
| 110m cpu (4.3 s wav) | 51× | 36× | 61× | — |

Aggregate ×realtime (B·audio/wall). Same-day follow-ups already in tree:
per-thread CUDA contexts (TLS handle/stream/buffers, shared weight cache)
replaced the global mutex, and the per-worker BLAS quota fixed the B=2 dip
(15.7→**30.8×**; B=8 unchanged at ~33×).

### Concurrent requests (server-style) — `bench_throughput --threads N`

N pthreads, each transcribing the 60 s wav on the shared model (nemotron,
CUDA): 1×=35, 2×=24.6 aggregate — but **N≥4 collapses** (1.9×): N callers ×
22-thread OpenBLAS thrash on the CPU-side attention. The knob is pinning
BLAS for the whole process: with `OPENBLAS_NUM_THREADS=3` → N=4 24.5×,
N=8 19.6×, no collapse (single-request latency drops to 17× — the usual
latency/throughput tradeoff). Adaptive per-request BLAS sizing in
mynah-asr-server is future work; for parallel throughput today, the batch API
is the fast path.

### CUDA v2a — GEMM precision (same day)

TF32 tensor-core math is now the **default** on CUDA: 300 s transcript
byte-identical to CPU on nemotron, all e2e green, ~4–8% RTF
(`MYNAH_ASR_CUDA_TF32=0` restores strict f32). Resident **bf16** weights via
cublasGemmEx are available with `MYNAH_ASR_CUDA_BF16=1`: half the VRAM and half
the PCIe on activations, but perf-neutral today (the host-side f32→bf16
conversion offsets the transfer saving) and with minor wording differences
on 300 s audio — it stays opt-in until the resident-activation encoder
(v2b in TODO) makes weight bandwidth the bottleneck.

### Two bugs found (and fixed) by these benchmarks

- **Banded windowed attention** (`encoder.c`): the offline path computed full
  T×T (and T×(2T−1) rel-pos) GEMMs even for windowed models that only read
  ~60 columns per row. On nemotron at 300 s (T=3750) that wasted ~2 TFLOP and
  ~170 MB/layer: RTF cuda 0.052→0.028, cpu 0.083→0.054. Parity vs oracle,
  streaming and batch outputs unchanged.
- **Nested OpenBLAS** (`threads.c`): on Linux OpenBLAS defaults to all cores;
  called from `mynah_asr_parallel_for` workers it oversubscribes catastrophically
  (batch 4×60 s: **257 s → 10 s** after pinning BLAS to 1 thread inside
  parallel regions — weak-symbol pattern from qwen-tts, `OPENBLAS_NUM_THREADS`
  still wins). macOS/Accelerate is unaffected.

~~Known issue: `mynah_asr_transcribe_batch` skips long-file segmentation~~ **fixed
2026-07-30**: batching now happens over SEGMENTS, planned by the same code the
single path uses, so a long item gives the same text through either entry point
(it used to come out worse through the batch). Guarded by phase 2 of
`tests/test_batch`, which forces a 5 s limit so the case is exercised on every
model rather than only on the full-attention ones.

## Streaming (Nemotron, cache-aware)

~26 ms of compute per 80 ms chunk (f32), ~9 ms with int4+SDOT — realtime with
ample headroom at every latency preset (0/1/3/6/13 lookahead chunks).

## Methodology and known pitfalls

- **Always 2+ warm-ups**: the first run pays the mmap weight page-in (seconds).
- On machines with little RAM (16 GB) models ≥3 GB can pay swap on the first
  runs, especially with Metal (+~half the weights in f16 GPU buffers): trust
  only the steady-state runs.
- **Weights over the network (SMB/NAS) don't hold the page cache** like local
  files: RTF up to 15× worse. Benchmark ONLY with models on local disk.
- NEVER ±INFINITY in code with `-ffast-math` (cost 6.5× in RTF — see
  architecture-notes §6).
