# `gpu/` — the CUDA batched streaming server (S14)

A SEPARATE tree: one process on one GPU serves the v2 wire protocol with the
weights and every stream's state resident in VRAM and one batched step per
cohort. `server/` is used read-only; `src/` is used as a library, plus one
optional seam the offline AED mode installs (`mynah_asr_offload`, below). Design, cost model, decisions and gates:
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

gpu/asr_offline.h       the OFFLINE seam (an AED pack): request in, final transcript out
gpu/offline_engine.c    the library model + (cuda) the GPU offload installed in it
gpu/aed_gpu.h           the GPU AED engine: the library's offload hooks on one device
gpu/cuda/aed.cu         encoder layers (packed rows, full rel-pos attention, 'same'
                        depthwise + folded batch_norm) and the batched greedy AED decoder
gpu/cuda/aed_stub.c     "not compiled" when built without nvcc
gpu/server/rest.{c,h}   the offline mode: REST, admission, the batcher thread, books
tests/test_cuda_aed.c        parity with the library, batch identity, the word path
tests/test_cuda_aed_server.sh  the offline mode end to end against the CLI
```

## Build

```
make lib                          # the library, at the top
make -C gpu                       # mynah-asr-server-cuda            (needs nvcc)
make -C gpu cpu                   # mynah-asr-server-cuda-cpuonly    (no nvcc: cpu engine + stub)
make -C gpu test-kernels          # tests/test_cuda_kernels          (GPU)
make -C gpu test-stream           # tests/test_cuda_stream           (GPU + pack)
make -C gpu test-aed              # tests/test_cuda_aed              (GPU + an AED pack)
```

`CUDA_ARCH` defaults to `sm_80 sm_86 sm_89 sm_90`; add `sm_120` with CUDA
12.8+. `native` is for local bring-up only.

## Run

```
./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b --cap 128 --cohort-ms 40 -p 8291 --metrics-port 9291
./mynah-asr-server-cuda -m ... --engine cpu        # the reference engine, any machine
./mynah-asr-server-cuda -m ... --dispatch-map      # what each op resolves to, then exit
```

A pack without Nemotron's language-prompt projector (`parakeet-realtime-eou-120m`:
one preset `[70, 1]`, English only) is served too: the encoder output goes
straight into the encoder projector, as `src/encoder.c` does, and the dispatch
map adds a `post-encoder encproj-only` row. Such a pack serves no `lang`: a
query or `reset` without one (or with `lang=` empty) runs the model, any
explicit tag -- `auto` included -- is refused with 400 `language_not_served`,
as the CPU server refuses it. Drive it with `--lang ''` / `LANG_Q=''` and
lookahead 1 in `gpu_qualify.sh` / `gpu_knee.sh`, and
`tests/test_cuda_stream <pack> --lookahead 1`.

On a pack whose vocabulary has `<EOU>`/`<EOB>` the model's own end of
utterance is an `eou` frame (`"source":"model"`), exactly as the CPU server
sends it: the label loop emits the token like any other, the host finds it in
the pass's tokens, reports the end of the encoder frame that emitted it
(`tok_frame`, downloaded only for such a pack) and queues the slot's model-state
reset (K/V ring, conv and subsampling caches, predictor at SOS) for the next
submit; the mel stream, the chunk cadence and the text go on. `/v1/health` and
`[DUMP]` count them in `eous`.

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
`step`, `publish`, `window` (the stage-ahead scan between submit and finish,
under the encoder pass, which `step` then excludes); and inside the step the engine's own phases `reset`,
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

## Defaults of the CUDA server (2026-10-09)

- `--precision auto --gemm auto`: the fixed-order bf16 tensor-core GEMM
  (`bf16` + `own-tc`) on sm_80 and newer, the f32 `own` GEMM on older cards,
  with `--weights int8`, or on the cpu engine. It is a numerical change from
  f32, batch-invariant byte for byte, with the 498-clip bank inside the WER
  bound (0.14469 vs 0.14483 corpus WER) and -55% device time per encoder pass;
  on an L40S it moved the screening knee from below C416 to above C512
  (`.work/l40s-2026-10-08-asr-cuda.md`). `--precision f32 --gemm own` restores
  the previous default.
- The serving-loop options below default ON for the cuda engine
  (`--stage-ahead 1 --host-threads auto --graphs buckets --warmup 1`): all
  byte-identical. `--stage-ahead 0 --host-threads 1 --graphs off --warmup 0`
  restores the eager loop. `--graphs` stays off with `--gemm cublas` or
  `--profile-stages`.
- Off: `--kv-dtype`, `--weights int8` (capacity levers, see below) and
  `--profile-host` (diagnostic).

## Serving-loop options

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

## Quantised arms (default off)

To improve before they can be defaults: `--weights int8` runs on the v1 f32
kernel, so it competes with the bf16 tensor-core GEMM instead of adding to it;
the lasting form decodes int8 codes inside the tensor-core tile load (and fuses
dequant, bias and activation). `--kv-dtype int8` cuts per-stream VRAM by 69%
but does not speed a pass up; it pays once the cap is VRAM-bound.

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

## Offline mode: AED packs (Canary)

The mode comes from the pack (`decoder.type` `aed_transformer`), never from a
flag. Such a pack cannot stream -- its decoder keeps no state between calls --
so the server takes the CPU server's contract for it: `POST
/v1/audio/transcriptions` (and `/translations`) answers the final transcript,
and `GET /v1/audio/stream` is refused before the upgrade with `400
model_not_streaming`. Nothing is ever sent before the transcript is final.

```
./mynah-asr-server-cuda -m models/canary-180m-flash --cap 64 --batch 16 --cohort-ms 30 -p 8297
./mynah-asr-server-cuda -m models/canary-180m-flash --engine cpu          # the library alone
./mynah-asr-server-cuda -m models/canary-180m-flash --aed-decoder host    # GPU encoder, CPU decoder
./mynah-asr-server-cuda -m models/canary-180m-flash --precision bf16 --gemm own-tc   # the A/B arm
```

`--cap` bounds the requests admitted (queued + in flight; past it 503
`server_at_capacity`), `--batch` the requests per GPU batch, `--cohort-ms` the
gather window from the oldest queued request. The host half of a request --
WAV, segmentation, mel, the canary2 prompt, the generation budget,
detokenisation, stitching -- is the library's own code; the GPU replaces the
encoder and the AED decode through `mynah_asr_offload` (`src/mynah_asr.h`).
Defaults to f32 own (bf16 own-tc only when asked: no quality gate yet).
Design and gates: `.work/canary-180m-l4.md` section 5.

## Status

See the board (`PLAN.md` S14) and the note's Evidence section. Until S14-6 runs
on a GPU, the cuda engine is compile-verified only; the server is protocol- and
fault-verified with the cpu reference engine.
