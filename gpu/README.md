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

At start the server raises its open-file soft limit to the hard one and says
what it got (each stream holds two descriptors); when `accept()` still runs
out of descriptors or kernel memory it logs once a second, counts
`accept_backoffs` (`/v1/health`, `/metrics`, `[DUMP] ... vram`) and backs off
20 ms instead of spinning. The banner also carries `pass_lanes` (the lanes
one encoder pass holds, `--pass-lanes N|cap`, default min(cap, 128), at most
1024; a larger cohort runs as several passes, which never changes a row's
result: `tests/test_cuda_stream <pack> --pass-lanes 2` gates the split path), `vram_used_at_ready_mb` (device-wide) and a
`[TOPOLOGY]` line (CPU model, usable CPUs, affinity, cgroup quota, NUMA nodes,
the GPU's PCI id and NUMA node, the open-file limit).

### Where the engine thread's time goes: `--profile-host` (DIAGNOSTIC)

```
./mynah-asr-server-cuda -m ... --profile-host      # then SIGUSR1, or stop it
```

Every segment of the engine loop is charged to one phase, so the phases sum
to the thread's wall: `scan`, `stage_copy`, `mel`, `idle`, `cohort_wait`,
`step`, `publish`; and inside the step the engine's own phases `reset`,
`pass_build`, `enqueue` (host time to queue H2D + encoder + projector),
`enc_wait` (blocked until the encoder pass is done), `dec_launch`, `dec_wait`,
`final_wait`, `detok`. Each `[DUMP]` (SIGUSR1, shutdown) is followed by
cumulative `[HOSTP] seq=N` lines: the host / device-wait / cohort-wait / idle
split, per phase total, us per cycle, us per lane and share of wall, the tail
of the cycle and host-busy times, and the engine thread's rusage. `/v1/health`
(`engine.host_profile_us`) and `/metrics` (`mynah_asr_gpu_hostp_us_total`)
carry the same totals. Off (the default) the loop reads no extra clock. A run
with the profile on is DIAGNOSTIC, never a latency headline.

`gpu/tools/hostp_level.py server.log --from 1 --to 2` turns two dumps into one
level's profile.

### Bench hygiene

- `gpu/tools/topology.sh [--server-cpus L] [--gen-cpus L]`: the `[TOPOLOGY]`
  line of a run (GPU, driver, the GPU's NUMA node, CPU, usable CPUs incl. the
  cgroup quota, governor, loadavg, the NUMA nodes of the pinned sets) and a
  WARNING when server and generator share CPUs or the server is off the GPU's
  node.
- `gpu/tools/gpu_sample.py run|summary`: nvidia-smi samples during a level
  (SM %, power, temperature, SM clock, throttle reasons, VRAM) and one `[GPU]`
  summary line per level.
- `gpu/tools/gpu_knee.sh`: a WAVE-class screen over a few levels, fresh server
  per level, with `PIN`/`CLIPIN` (taskset for server / load generator),
  `SRV_ARGS` (e.g. `--profile-host`), and per level the client latency line,
  the `[GPU]` line, the server's VRAM and the `[HOSTP-LEVEL]` profile of the
  measured window. `gpu_qualify.sh` records the same topology and `[GPU]`
  lines and takes `--server-cpus` / `--gen-cpus`.

## Status

See the board (`PLAN.md` S14) and the note's Evidence section. Until S14-6 runs
on a GPU, the cuda engine is compile-verified only; the server is protocol- and
fault-verified with the cpu reference engine.
