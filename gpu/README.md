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

## Serving-loop options (all default OFF until screened)

| flag | what it does | identity |
|---|---|---|
| `--stage-ahead 1` | the step runs as `submit` (lanes decided, chunks taken out of the slots, staged into a per-pass pinned slab, H2D + encoder enqueued) and `finish` (label loop, wait, detokenise); between the two the engine thread stages and feeds the NEXT whole chunk of every slot that has one, i.e. the host mel runs under the encoder pass | byte-identical: each slot's feed sequence is the one the scan would make, only earlier |
| `--host-threads N\|auto` | a feed team: the host mel of a staged batch runs on N threads (the engine thread included); `auto` = 1 up to 8 usable CPUs, 2 up to 16, 4 above, counting the affinity mask capped by the cgroup quota | byte-identical: one slot, one thread, same mel |
| `--graphs buckets` (`--graph-buckets 8,16,32,64,128`) | the encoder pass (subsampling, 24 layers, projector) replays a CUDA graph captured at start-up per (lane bucket, row bucket of 64); padding lanes have q = 0 on a scratch slot, padding rows belong to no lane; the label loop stays eager | byte-identical: the GEMMs are row-stable in M, every other kernel is per row or per lane |
| `--warmup 1` | before listening, steps shaped like real traffic (first, steady and tail chunks of the default preset at every lane bucket) on the idle slots, then every slot reset and the counters zeroed | byte-identical (gate C) |

`tests/test_cuda_stream` gate C re-runs the gate-A cohorts with each option and
with all of them, and requires the same per-step output (text, t0/t1, token
count, finish) as the default engine. The banner's `[SERVER-CONFIG]
serving-loop` line reports what resolved, the start-up time, the capture and
warm-up times and the VRAM in use at ready; `[DUMP] ... loop` reports the
window, the submit/finish split and graph vs eager passes.

## Status

See the board (`PLAN.md` S14) and the note's Evidence section. Until S14-6 runs
on a GPU, the cuda engine is compile-verified only; the server is protocol- and
fault-verified with the cpu reference engine.
