# `gpu/` — the CUDA batched streaming server (S14)

A SEPARATE tree: one process on one GPU serves the v2 wire protocol with the
weights and every stream's state resident in VRAM and one batched step per
cohort. Nothing under `src/` or `server/` is modified for it; those are used
read-only as libraries. Design, cost model, decisions and gates:
[`.work/cuda-batched-streaming-server.md`](../.work/cuda-batched-streaming-server.md).

```
gpu/asr_engine.h        the seam: server <-> engine (ops table, one thread)
gpu/pack.{c,h}          a converted pack opened for the GPU (library loaders, read-only)
gpu/engine_cpu.c        the REFERENCE engine: the library's stream API behind the seam
gpu/cuda/kernels.cuh    device types (rows, slot meta, arena) and the launch wrappers
gpu/cuda/gemm.cu        C = A·Wᵀ, row-stable by construction (the default GEMM)
gpu/cuda/kernels.cu     layernorm, silu, rel-pos attention on the K/V ring, GLU + causal
                        depthwise with cache, subsampling stages, label-loop decode, reset
gpu/cuda/engine.cu      the resident engine: upload, arena, cohort step, decode loop
gpu/cuda/engine_stub.c  "not compiled" when built without nvcc
gpu/server/main.c       mynah-asr-server-cuda: ingest, cohort scheduler, books, protocol
gpu/server/ws.{c,h}     the RFC 6455 server side (client frames)
tests/test_cuda_kernels.cu   kernels vs CPU references; the GEMM row-stability BYTE gate
tests/test_cuda_stream.c     gate A (batch identity on the GPU), gate B (CPU f32 == GPU)
```

## Build

```
make lib                          # the library, at the top
make -C gpu                       # mynah-asr-server-cuda            (needs nvcc)
make -C gpu cpu                   # mynah-asr-server-cuda-cpuonly    (no nvcc: cpu engine + stub)
make -C gpu test-kernels          # tests/test_cuda_kernels          (GPU)
make -C gpu test-stream           # tests/test_cuda_stream           (GPU + pack)
```

`CUDA_ARCH` defaults to `sm_80 sm_86 sm_89 sm_90`; add `sm_120` with CUDA
12.8+. `native` is for local bring-up only.

## Run

```
./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b --cap 128 --cohort-ms 40 -p 8291 --metrics-port 9291
./mynah-asr-server-cuda -m ... --engine cpu        # the reference engine, any machine
./mynah-asr-server-cuda -m ... --dispatch-map      # what each op resolves to, then exit
```

The banner prints what RESOLVED (engine, device, precision, GEMM, VRAM, cohort);
`/v1/health` carries the session books, the lag histogram and the engine's
counters; `SIGUSR1` prints the `[DUMP]` lines `tools/bench/v2_verdict.py` reads.
`tools/bench/v2_qualify.sh`, `tests/ws_probe.py` and `tests/fault_probe.py`
run against it unchanged.

## Quantised arms (default off)

Two independent switches, both numerical changes (not bit-identical to f32),
both batch-invariant by construction (a lane's bytes never depend on its
cohort or slot; gate A of `tests/test_cuda_stream` holds byte for byte):

| flag | what is stored | how it is read |
|---|---|---|
| `--kv-dtype f32` (default) | the K/V ring f32, 10.5 MiB per slot | reference |
| `--kv-dtype bf16` | ring bf16 (round to nearest even), 5.25 MiB per slot | widened in the attention kernel |
| `--kv-dtype int8` | ring int8 + one f32 scale per (position, head), `scale = max\|x\|/127`, 2.71 MiB per slot | `code * scale` in the attention kernel |
| `--weights int8` | the 24 layers' FFN, q/k/v/o, pointwise-conv linears and the joint head as the CPU's per-row int8 codes (`mynah_asr_quantize_int8`) | `gpu/cuda/gemm_w8.cu`: the v1 fixed-order fma chain over the codes, times the row scale |

The fresh rows of a chunk are attended in f32 and quantised once on commit, so
a row reads the same values on every later chunk, alone or batched. The
subsampling, prompt projector, LSTM and depthwise weights stay f32. The
dispatch map's `precision` and `kv ring` lines and the banner's `precision=`
(`f32`, `f32+kv-int8`, `w8a32`, `w8a32+kv-int8`, ...) say what resolved.
Evidence: [`.work/cuda-quant.md`](../.work/cuda-quant.md).

## Status

See the board (`PLAN.md` S14) and the note's Evidence section. Until S14-6 runs
on a GPU, the cuda engine is compile-verified only; the server is protocol- and
fault-verified with the cpu reference engine.
