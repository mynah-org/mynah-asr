# S14 — CUDA batched streaming server: Nemotron on one GPU, device-resident

Status: IN PROGRESS (opened 2026-09-26; design accepted for implementation the
same day; nothing measured on a GPU yet)

Task: S14-1 … S14-9 (board section S14)
Question: can ONE process on ONE GPU serve at least the C=128 real-time
Nemotron streams the qualified CPU fleet serves on 30 Axion cores, on a cloud
GPU instance that has FEW vCPUs (4–8) and a 24 GB class card, with the same
wire protocol, the same harness, the same bounds and the same transcripts?

## Scope and the one rule that shapes everything

**DECISION.** The CUDA path is a separate tree and a separate binary. The
qualified CPU path (`src/` stream step, `server/` v2 prefork scheduler,
C=144 EN / C=128 EN+FR) is not modified for this work and does not carry a
GPU branch. The GPU server reuses the CPU code READ-ONLY where it is a
library (pack loading and tensor access, tokenizer and incremental
detokeniser, streaming mel, `server/http_util.c`, `server/stream_out.c`) and
re-implements everything the GPU design does differently. Its design is free
to diverge from `serving-v2-design.md` where the machine is different, and
this note says where and why.

What is the same by construction, because it is what makes the result
comparable and the harness reusable:

- the model pack (`mynah.json`, same tensors, same presets, same tokenizer);
- the wire protocol of `ws-protocol-v2.md` (`/v1/audio/stream`, `delta`/`eou`/
  `done`/`error` frames with `seq`, `audio_s`, `lag_ms`, `t0`/`t1`; refusals
  as HTTP statuses before the upgrade);
- `/v1/health`, `/metrics`, the SIGUSR1 `[DUMP] … books` line and the session
  books, so `tools/bench/v2_qualify.sh` and `v2_verdict.py` run unchanged;
- the twelve registered bounds and the WAVE/SOAK vocabulary (ENGINEERING.md §8).

## Known facts (from the code, not from docs)

The single-stream step this port must reproduce, `src/encoder.c:1096`
(`mynah_asr_enc_stream_step`) and `src/mynah_asr.c` (`stream_flush_chunk`,
`stream_decode`):

1. **Mel** (`src/features.c:229`, `mynah_asr_mel_stream_feed`): pre-emphasis
   in double, 512-point radix-2 FFT in double, mel projection, `log(x+guard)`,
   no normalisation. Frame t is ready once `t*160 + 256` samples exist.
2. **Chunk geometry**: a chunk is `1 + 8r` mel frames the first time, then
   `8(r+1)`, producing `q = r + 1` encoder frames (`r` = lookahead: 0/3/6/13 in
   the Nemotron pack, so q ∈ {1, 4, 7, 14}); the default preset `[56,3]` is
   32 mel frames = 320 ms of audio per chunk, 4 encoder frames of 80 ms.
3. **Subsampling** (`src/subsampling.c:375`): three stride-2 3×3 stages on
   `[C, T, F]` (C=1 → 256 → 256 → 256), each with a ONE-frame time cache per
   stage (the first chunk prepends one extra zero frame; the LAST chunk
   appends one zero frame, `is_last`), frequency padding (2,1); stage 0 is a
   full conv (C_in=1), stages 1–2 are depthwise then pointwise `[C,C]` + ReLU;
   then channel-major flatten `[To, 256·17 = 4352]` and a linear to d=1024.
4. **24 conformer layers**, pre-norm macaron, d=1024, 8 heads of 128, FFN 4096,
   conv kernel 9, no biases, LayerNorm inside the conv module:
   - ½FFN1: `LN → W1 [4096,1024] → SiLU → W2 [1024,4096]`, `x += 0.5·y`;
   - MHSA: `LN → K,V [1024,1024]` of the chunk; attention of the chunk's Q
     rows over the window `[valid cache rows ++ Q fresh rows]` (K = valid + Q,
     valid ≤ 56) with Transformer-XL relative positions: `scores = (q+u)·kᵀ +
     rel_shift((q+v)·rkᵀ)`, `rk` read from the load-time table
     `relpos_tab[layer][2·kmax−1, d]` at offset `kmax − K` (S10-1, F28), then
     `O [1024,1024]`; the cache commits the Q fresh K/V rows and keeps the
     last min(valid+Q, 56) rows (`src/kvcache.h`);
   - conv: `LN → PW1 [2048,1024] → GLU → causal depthwise k=9 with a
     per-layer cache of the last 8 rows → LayerNorm → SiLU → PW2 [1024,1024]`;
   - ½FFN2 as FFN1, then the layer's output LN.
5. **Post**: `cat(x, one-hot(prompt)) [Q,1152] → L1 [2048,1152] + b → ReLU →
   L2 [1024,2048] + b → encproj [640,1024] + b` (`src/encoder.c:584`).
6. **Greedy RNNT** (`src/decoder.c:445`): per encoder frame `logits =
   head(ReLU(enc_t + g)) + head_b` over V=13088, argmax (first index wins a
   tie); blank advances the frame; a token is emitted, appended, and ONLY then
   the predictor moves: `embedding[token] → 2-layer LSTM (hidden 640) →
   projector [640,640] + b → g`; at most `max_symbols` (10) per frame. SOS is
   one `pred_step(blank)` on a zero state. The CPU blocks blank runs into one
   GEMM of up to 32 frames; that is an optimisation, not semantics.
7. **State per stream** (Nemotron 0.6B, f32): K/V `24·2·56·1024·4 B` = 11.0 MB;
   conv cache `24·8·1024·4` = 0.79 MB; three subsampling caches `[C, F]` under
   100 KB; predictor `h,c [2][640]`, `g [640]`, `last_token`, `n_emitted`,
   `t_abs`; mel stream (host). ≈ 12 MB per stream; 256 streams ≈ 3 GB.
8. **Weights**: 24 × (16.8M FFN + 4.2M attention + 1.05M relk + 3.1M conv) ≈
   600M encoder parameters + subsampling 4.5M + prompt 4.5M + head 8.4M +
   LSTM 6.5M: **2.4 GB in f32, 1.2 GB in bf16, 0.6 GB in int8**. The rel-pos
   table is 14 MiB. Everything fits a 24 GB card many times over.
9. **The CPU cadence law and where its cost is** (F23, F27/F28, S12-7c): on
   the CPU the per-stream term `b` dominates (70 %) and the fixed term `a` is
   the weight walk of one pass. On a GPU the two terms are of the same order
   and both are byte counts, not guesses: one pass over 2.4 GB of f32
   weights costs ≈ 8 ms on a 300 GB/s card (L4), ≈ 4 ms in bf16, ≈ 2 ms in
   int8, whatever B is; the per-stream arithmetic of one chunk is ≈ 4.6 GFLOP
   (2 · 0.6e9 parameters · 4 frames), so a cohort of 16 streams is ≈ 75
   GFLOP — ≈ 2.5 ms at an L4's 30 TFLOP/s f32 peak, ≈ 15 ms for a plain
   f32 kernel at a fifth of peak, ≈ 1 ms on bf16 tensor cores. So at a
   cohort of 16 the fixed weight walk and the compute are both a few
   milliseconds, and 8 cohorts per 320 ms period leave the card well under
   half busy in f32 and under a fifth in bf16. **HYPOTHESIS until S14-6
   measures `a_gpu` and `b_gpu` on a named card**; the design rests on the
   byte counts, the numbers do not enter any claim.
9b. **What NVIDIA's own stack sustains for this model class** (Riva/NIM
   performance pages, primary; details in the prior-art section): Nemotron
   streaming RNNT at a 160 ms chunk, 512 concurrent streams on one L40 with
   188 ms average latency; Conformer-CTC at 160 ms about 230 streams on
   every card from a T4 to an H100, which says that ceiling was the CPU
   decoder, not the GPU. C=128 at a 320 ms chunk on a 24 GB card is inside
   what the architecture has already demonstrated; whether THIS
   implementation reaches it is the question S14-7 answers.
10. **What the siblings learned on CUDA** (`../mynah-tts` branch
    `pocket-cuda-streaming-parity`, `../qwen-tts`; read-only audit 2026-09-26):
    a per-op offload seam is round-trip-bound and cannot be fixed
    incrementally; resident weights + resident per-request state + one dense
    `[B, D]` activation with per-row state pointers in pinned tables is the
    pattern that worked; CUDA graphs keyed by batch width gave 5–8 %, not the
    30 % the launch count suggested; TF32/bf16 tensor cores CHANGED the
    output and are off by default; a runtime-bounded per-lane accumulator
    array spilled to local memory and produced wrong results (template the
    lane count); reserve every workspace BEFORE graph capture; never fork
    after CUDA init (`prefork.c` refuses); `-arch=native` is for local
    bring-up only; the batch-vs-single gate on the GPU is BIT-EXACT with
    poisoned idle lanes, while CPU↔GPU is a tolerance / transcript gate.

## Prior art consulted (2026-09-26, primary sources; what transfers)

- **NeMo cache-aware streaming** (`conformer_encoder.py`, `streaming_utils.py`,
  `mixins.py`): caches are batched tensors `[L, B, 56, d]` and `[L, B, d, K-1]`
  with a per-stream `cache_last_channel_len`; `CacheAwareStreamingAudioBuffer`
  steps every stream of a batch in lockstep and pads shorter streams; NeMo
  captures the steady-state chunk step as one CUDA graph
  (`set_streaming_cuda_graphs`). NVIDIA shipped and fixed a batching bug of
  exactly the kind contract 4 forbids: a normalisation read the padded width,
  so a short stream's features depended on its batch neighbour (NeMo Speech
  PR #16287). Our per-row descriptors make every per-stream kernel read its
  own lengths; the gate 2a is what proves it.
- **RNNT label-looping and speed-of-light decoding** (Bataev, Interspeech
  2024; Galvez et al., Interspeech 2024; `rnnt_label_looping.py`): the
  decoder loop batched over lanes with masked updates and per-lane time
  indices; predictor calls = the longest hypothesis in the batch; CUDA-graph
  conditional nodes remove the host sync entirely, with a fallback that lets
  the host drive the while loop. Section 3 step 4 is that algorithm; the
  host-driven variant first, the conditional-node graph as a phase-2 lever.
- **Riva / NIM**: `max_batch_size=32` per acoustic-model instance, Nemotron
  streaming registered with `att_context_size [70,1]` and 80 ms timesteps;
  the featurizer runs on the GPU (`use_streaming_torch_fe`); the two
  published operating points are low-latency (160 ms) and high-throughput
  (960 ms). We publish one (320 ms, the pack's default preset) and keep the
  preset per stream.
- **Triton sequence batching** (`model_config.proto`): slot-indexed state,
  READY/START/END controls, `max_queue_delay_microseconds`, a slot reclaimed
  on idle timeout, refuse when slots are exhausted. Our arena IS that model.
- **sherpa-onnx online server**: a timer every `loop_interval_ms` (10)
  gathers every READY stream up to `max_batch_size` and decodes them in one
  call. Our cohort is the same idea with the wait bounded by the oldest ready
  chunk rather than a fixed tick.
- **vLLM / vLLM-Omni**: no NeMo encoder, no cache-aware streaming; the
  Realtime API extends one request's prompt per audio chunk. What transfers
  is the discipline: one engine step, a fixed compute budget per step, graphs
  captured at bucketed widths and replayed with padding, refuse rather than
  degrade when memory is exhausted.
- **NeMo-Speech.cpp** (NVIDIA, ggml): CUDA backend, a bounded worker pool,
  independent per-connection state; cross-session GPU batching of streaming
  ASR is not documented and no throughput number is published.
- **Kernels**: NeMo's own SDPA branch shows the rel-pos recipe — `matrix_bd`
  after `rel_shift` is an additive bias to a standard attention — and Kestrel
  fused shift + both products + softmax into one op; ggml-cuda's depthwise
  conv is a direct one-thread-per-output kernel. No FasterTransformer or
  TensorRT-LLM Conformer rel-pos kernel exists to lift.
- **Instances** (4 vCPU class): AWS g6.xlarge L4 24 GB 300 GB/s; g5.xlarge
  A10G 24 GB 600 GB/s; g4dn.xlarge T4 16 GB; GCP g2-standard-4 L4. The
  design assumes 4 vCPUs carry the WebSocket ingest and, in phase 1, the mel.

## Unknowns (each becomes a measurement, none is assumed)

- `a_gpu` (fixed cost of one batched pass) and `b_gpu` (per-stream cost:
  attention, conv, subsampling, decode) per card; the knee they imply with
  cohort batching (below).
- Whether f32 CUDA transcripts equal the CPU f32 transcripts on `samples/`
  (the 2026-07-20 cuBLAS offload was identical on 10 models; the new kernels
  reorder sums and use CUDA's `expf`/`tanhf`, so this is re-measured, not
  inherited).
- Whether the GPU decode loop (a host-visible "lanes still active" flag per
  iteration) is cheap enough at C=128, or needs a graph with a fixed
  iteration budget.
- Host cost of mel + WebSocket ingest for 128 streams on 4 vCPUs (≈ 0.65
  core for the double-precision mel alone by the CPU numbers).
- The cohort interval's effect on emission lag and on GPU duty.

## Files/functions inspected

`src/encoder.{c,h}` (stream step, attention core, conv mid, post, rel-pos
table), `src/kvcache.h`, `src/subsampling.{c,h}` (stream_stage, conv2d_s2),
`src/features.{c,h}` (mel stream), `src/decoder.{c,h}` (pred_step, greedy),
`src/mynah_asr.{c,h}` (stream struct, feed, step_batch, finish),
`src/backend.h`, `src/cuda_gemm.cu` (the 2026-07 offload seam),
`server/sched.c` (the CPU scheduler's step call), `Makefile` (cuda target),
`.github/workflows/ci.yml` (compile-only CUDA job), `docs/nemotron-arch.md`,
`docs/serving-findings.md` F20/F23/F27/F28, `.work/serving-v2-design.md`.
Siblings: `../mynah-tts/gpu/cuda/backend_cuda.cu`, `src/backend.{c,h}`,
`.work/pocket-tts-cuda-streaming-parity.md`; `../qwen-tts/qwen_tts_cuda_talker.cu`,
`docs/serving/gpu-cuda.md`, `.work/cuda-parity-track-20260915.md`.

## The design

### 1. Topology

```
mynah-asr-server-cuda  (ONE process, ONE GPU, no prefork)
  listener ─ accept ─ HTTP/WS handshake ─ admission (slot cap → 503 + Retry-After)
  ingest threads ×k    WS framing, s16→f32, streaming mel on the host into the
                       slot's mel ring (a host-resident copy of the CPU mel: bit
                       identical to the CPU's, by reuse); never touch the GPU
  gpu thread ×1        the ONLY thread that touches the device (asserted):
                       loop { wait for a ready slot; cohort gather; ONE batched
                              step; hand tokens to the slots; publish deltas }
  stream_out ×1/conn   the v2 async writer, reused as-is (bounded ring, peer-gone)
```

Why one process: a CUDA context does not survive `fork()` and the GPU is one
device; the CPU design's reason for prefork (one worker per cache domain) does
not exist here. Why one GPU thread: the same reason the CPU scheduler is one
thread — the model's state has one owner, and the step is one submission.

### 2. Device-resident everything

- **Weights** uploaded once at load, in the layout the kernels read (`[N,K]`
  row-major, K contiguous, as in the pack); f32 in phase 1, bf16 opt-in in
  phase 2 with its own CER gate, the pack's own int8 rows + scales in phase 3.
  The rel-pos table is built by the CPU code at load (identical values) and
  uploaded.
- **Slot arena**: `--cap N` slots allocated at start as `[N][…]` device
  arrays: K/V ring `[layer][2][56][1024]` with `head`/`valid` per slot, conv
  cache `[layer][8][1024]`, subsampling caches, predictor `h/c/g`, decode
  metadata. A slot is `reset`, never freed. Nothing is allocated after start
  (the qwen-tts and mynah-tts rule: allocations inside a capture break it, and
  a step that allocates is a step whose latency has a tail).
- **Activations**: one dense scratch `[Rmax, ·]` with `Rmax = Σ q_i` over the
  largest cohort, carved once. Rows of different streams are stacked; per-row
  state pointers live in a **step descriptor table** in pinned host memory,
  copied H2D as part of the step: `slot, q, n_mel, first, last, prompt, K,
  valid, head, row_off`.
- **Host↔device traffic per step**: H2D = the mel chunks (16 KB per stream)
  and the descriptor table; D2H = emitted tokens (`≤ q·max_symbols` ints per
  lane) and counts. ≈ 2 MB per 320 ms at C=128. Nothing else crosses.

### 3. One step, heterogeneous by construction

The CPU batched step groups streams by lookahead preset because its stacked
GEMMs and its per-stream loops were written for one q. On the GPU every
per-stream kernel reads its own `q_i`, `n_mel_i`, `first_i`, `last_i` from the
descriptor, and the row-stacked GEMMs do not care what q a row belongs to.
So **a cohort mixes presets, first chunks, steady chunks and finalising
chunks in one pass**: there is no separate finalize path (F20's serial tail
does not exist here) and no per-preset grouping.

Per step, in order, all on one CUDA stream:

1. `ss_stage0/1/2` (per stream, direct 3×3 s2 convolutions with the time
   cache prepended and the `first`/`last` padding from the descriptor;
   pointwise `[C,C]` as one GEMM over all streams' positions concatenated
   along S; ReLU) → flatten → row-stable GEMM `[ΣQ,4352]·lin_wᵀ + b`.
2. 24 × { `layernorm` · GEMM · `silu` · GEMM · `residual(0.5)` ·
   `layernorm` · GEMM K,V · GEMM Q · `attention_relpos` (one block per
   (stream, head): scores over the ring window + the table rows, fused
   rel_shift + softmax in shared memory, `P·V`; commits the fresh K/V rows
   into the ring) · GEMM O · `residual` · `layernorm` · GEMM PW1 ·
   `glu_dwconv_causal` (per stream: GLU, cache prepend, k=9 depthwise,
   cache refresh) · `layernorm_silu` · GEMM PW2 · `residual` · `layernorm` ·
   GEMM · `silu` · GEMM · `residual(0.5)` · `layernorm` }.
3. `kv_advance` (per slot: `valid`, `head`), `prompt_onehot_cat`, GEMM L1 +
   ReLU, GEMM L2, GEMM encproj → `enc [ΣQ, 640]`.
4. **Batched greedy RNNT** (label-looping over lanes): per lane `t, emitted,
   done` on device; iterate { `joint_relu` over active lanes → row-stable
   GEMM `[B_act, 13088]` → `argmax_first` per row (lowest index wins a tie,
   as the CPU) → `decide` (blank: `t++`, `emitted=0`; token: append, mark
   lane for a predictor step; `max_symbols` reached: `t++`) → `pred_step`
   over the emitting lanes (embedding gather, two LSTM layers as GEMMs over
   the emitting rows + a gates kernel, projector) → `n_active` D2H } until
   `n_active == 0`. Tokens and per-lane counts D2H once.

The SOS `pred_step(blank)` runs at slot reset; `t_abs` is per lane.

### 4. GEMMs are row-stable BY CONSTRUCTION, and cuBLAS is the comparison arm

Durable contract 4: a transcript never depends on who a stream was batched
with. cuBLAS is not contractually row-stable in M (its heuristics pick the
kernel per shape, split-K included), exactly the property S1-4 had to
MEASURE per BLAS on the CPU. **DECISION**: the GPU GEMMs of this path are our
own kernels — a register-tiled `C[M,N] = A[M,K]·W[N,K]ᵀ` with the k-loop in one
fixed order per output element and no split-K — so row `i` of the output is
a function of row `i` of the input alone, whatever M is. That is the same
argument `src/sgemm.c`'s DOT family rests on, and it is CHECKED, not trusted:
`tests/test_cuda_kernels` compares every output row at M=1 against the same
row inside M=257 byte for byte, on every shape the model issues. A cuBLAS
arm (`MYNAH_ASR_CUDA_GEMM=cublas`, `CUBLAS_PEDANTIC_MATH`) exists for A/B and
is subject to the same measurement; it is never the default until its gate
passes on the card in use.

A hand-written f32 kernel at a fifth of peak puts the compute of a cohort of
16 at roughly the cost of the weight walk (fact 9); that is affordable for
the first qualification, and the bf16 tensor-core variant of the same
kernel (phase 2, same fixed k-order, same gate) is the lever if the profile
says the GEMMs dominate. Both read the weights once per pass.

### 5. Cohort batching, and why the CPU's falsification does not transfer

The CPU design rejected a batch collection window (F4: it widened the ready
set and did not move the ceiling), because on the CPU the fixed term `a` was
16 % of the cost and a window only buys `a`. On the GPU `a` — one walk over
the weights — IS the cost (fact 9): a step at B=3 costs the same bandwidth
as a step at B=48. With arrivals at random phases on the 320 ms grid, a
run-when-ready loop at C=128 would run ≈ 40 passes per period at B≈3 and
spend the whole period streaming weights.

**DECISION**: the GPU thread gathers a **cohort**: it steps when the oldest
ready chunk has waited `--cohort-ms` (default 40) OR the ready set has
reached the arena's `Rmax`, whichever first. At 40 ms that is ≤ 8 passes
per period (≤ 32 ms of bf16 weight traffic, ≤ 64 ms in f32, per 320 ms) and
adds ≤ 40 ms to the emission lag against a 320 ms bound. **HYPOTHESIS to
measure**: emission lag p95 and GPU duty over `--cohort-ms ∈ {0, 20, 40,
80}` at C=128, same bank, same seed; the default is chosen from that ladder,
and the ladder is recorded here before the runs.

Fairness stays by construction: a slot contributes at most one chunk per
step (its mel ring holds the rest); a client faster than real time is
throttled by TCP as in v2.

### 6. Admission and the books

Admission ladder: slot cap (the arena) → 503 + `Retry-After` before the
upgrade; `model_not_found`/`language_not_served`/`model_not_streaming` as in
v2. Session books identical in shape to S12-18 (`sessions = completed +
cancelled + aborted + active`, counted once at release), the `[DUMP] … books`
line, `mynah_asr_sessions_*` and the fleet series names with a fleet of one,
`/v1/health` with the same fields plus `gpu` facts (device name, VRAM total /
used, arena slots, resident precision, GEMM provider, cohort-ms, graph mode).
The effective-config banner prints every one of those with
`requested | effective | reason`, and `--dispatch-map` prints the kernel per
operation (ENGINEERING.md §5/§6: a silent fallback is a failure — this path
has NO CPU fallback; a device error fails the step and cancels the cohort's
sessions with `internal_error`, visibly).

Cancellation: peer gone → the slot is marked and excluded from the next
cohort; its device state is reset at release; no zombie step (S12-19's
invariant, tested by `tests/fault_probe.py` against this server unchanged).

### 7. Gates, in the order they run

1. `tests/test_cuda_kernels` (GPU, no model): every kernel against a scalar
   CPU reference on random data with a stated tolerance; the GEMM
   row-stability byte gate over the model's shapes; the argmax tie rule; the
   ring-window gather against the logical contents (`mynah_asr_kv_logical`
   as the oracle).
2. `tests/test_cuda_stream` (GPU + pack): (a) **batch identity on the GPU**:
   B streams in one cohort against each stream alone on the same GPU, byte
   identical encoder output, tokens and caches, with idle lanes poisoned;
   (b) **CPU↔GPU**: transcripts of the `samples/` clips through
   `mynah_asr_stream_feed` (CPU f32) and through the GPU step; expected
   identical, reported per clip; a difference is a finding with a CER, never
   a silent pass and never a silent fail.
3. `tests/test_gpu_server_*.sh`: `tests/ws_probe.py` and `tests/fault_probe.py`
   against `mynah-asr-server-cuda` (protocol-level, unchanged).
4. CI: a compile-only job in an `nvidia/cuda` devel container with an
   explicit `-arch` matrix (`sm_80 sm_86 sm_89` on 12.6; `sm_120` on 12.8+),
   full link, and `--gpu-self-test` that must print "compiled, no device"
   rather than "not compiled". No model, no GPU on the runner.
5. Qualification: `tools/bench/v2_qualify.sh` with the GPU server binary on
   the target instance class, C=128 first (two 30-minute soaks, different
   seeds, all twelve bounds, transcript parity against the GPU's own C=1
   reference AND reported against the CPU reference), then a ladder upward.
   The number quoted is the SOAK, never a WAVE.

### Acceptance gate of S14 as a whole

`mynah-asr-server-cuda` QUALIFIED at C=128 on a named instance class with
the registered bounds, transcripts identical across streams and to its own
unloaded reference, and the CPU↔GPU transcript comparison on the bank
reported (identity or WER delta), with dispatch proven from the process.
Anything short of that is "compile-verified" or "runtime-verified", never
qualified.

## Phases (each a board item with its own gate)

- **S14-1** this note, the cost model, the decisions, the board.
- **S14-2** kernels + row-stable GEMM + `test_cuda_kernels` + the CI
  container job. Compile-verified on CI; runtime-verified on a GPU.
- **S14-3** resident model + slot arena + batched encoder step + gate 2a.
- **S14-4** batched greedy RNNT on device + gate 2b.
- **S14-5** the server binary: protocol-compatible, cohort scheduler,
  admission, books, health/metrics/dump, banner, dispatch map, fault probes.
- **S14-6** bring-up on a GPU box: parity on `samples/` and the stress bank,
  `a_gpu`/`b_gpu` measured, the cohort ladder.
- **S14-7** qualification C=128 on the target instance class, then the ladder.
- **S14-8** phase 2 levers, each behind a flag with its own gate: bf16
  resident weights (CER gate), mel on the device (cuFFT; parity gate on the
  mel and on transcripts), CUDA graphs keyed by cohort width, int8 head and
  int8 encoder from the pack's own rows (integer accumulation: exact batch
  identity and a chance of CPU==GPU byte identity on int8).
- **S14-9** multi-GPU: one process per GPU behind the v2 router is the
  natural shape (SCM_RIGHTS handoff exists); not before S14-7.

## S14-6b — profile C=128 against C=160 before any optimisation (registered 2026-09-26, before the data)

DECISION: the qualified f32 build of 2026-09-26 (`075c8d2`) is frozen as the
**CUDA F32 v1 baseline**. No lever of S14-8 is written until this profile says
which one; afterwards every lever is one change, measured as an A/B against
this baseline on the same bank, with transcript parity (gates A and B) before
any timing is read. Precision (bf16/fp16/int8 on tensor cores) is a numerical
change: it carries its own CER/WER gate on the bank and is never promoted on
speed alone (ENGINEERING.md §9).

Instrument: `mynah-asr-server-cuda --profile-stages` (DIAGNOSTIC, default off)
records CUDA events at every stage boundary of a pass and charges each
interval to the stage that opened it; the host mel front end is timed in the
engine; both reach `/v1/health` (`engine.profile_ms`) and the `[DUMP] profile`
line. Stages: h2d, subsample, ffn1, att_proj (LN + K/V/Q GEMMs), att_core
(attention kernel + ring commit), att_out (O GEMM + residual), conv, ffn2,
post (advance + prompt + projector), dec_joint (joint + head GEMM + argmax +
decide), dec_pred (embedding + LSTM + projector), dec_sync (the per-iteration
compaction, D2H of the active count and the host round trip), d2h, other.
Nsight Systems is not installed on the box and hardware counters may be
restricted in the container; `ncu` on single kernels is a follow-up only if
the stage table points at one kernel.

Runs: C=128 and C=160, 120 s each, fresh server, same bank rule as the
qualification, `--cohort-ms 40`, profile ON (so the absolute latencies of
these runs are not quoted: the events add a little work per pass). Recorded
per run: the stage table per pass and per lane, passes per second, lanes and
rows per pass, cohort wait, engine-thread busy fraction (step wall / wall),
host mel ms per audio-second, GPU util and clocks from nvidia-smi.

Decision table, fixed now:

| if the profile shows | then the first lever is |
|---|---|
| step wall < ~50 % of wall at C=160 while lag fails | serving: the cohort policy, not the GPU (the engine thread waits) |
| host mel or staging > ~25 % of the engine thread | move the mel to the device (cuFFT), S14-8 |
| dec_sync + dec_joint + dec_pred > ~30 % of device time | the label loop: fewer host syncs (graph conditional nodes or a fixed iteration budget) |
| the GEMM stages dominate and ms per pass is flat in rows | small-M GEMM efficiency (tile shape, split-K with a fixed order) before any precision change |
| the GEMM stages dominate and scale with rows | tensor cores: bf16 weights with f32 accumulation, behind the CER gate |
| many tiny stages of similar size, no dominant one | launch overhead: CUDA graphs keyed by cohort width, fusion of LN/residual |

## S14-6b RESULT (2026-09-26, L4) and the first GEMM lever

**Profile** (`--profile-stages`, DIAGNOSTIC; `.work/evidence/gpu-l4-20260926/`),
frozen F32 v1, cohort 40 ms:

| stage | C=128, 95 rows/pass | C=160, 163 rows/pass |
|---|---|---|
| FFN1 + FFN2 | 26.1 ms (48 %) | 34.9 ms (48 %) |
| attention projections + O | 8.6 ms (16 %) | 11.0 ms (15 %) |
| attention core | 6.7 ms (12 %) | 10.8 ms (15 %) |
| conv (2 GEMMs + depthwise) | 6.0 ms (11 %) | 7.9 ms (11 %) |
| decoder: joint + predictor + sync | 5.4 ms (10 %) | 6.1 ms (8 %) |
| subsampling, post, transfers | 1.8 ms (3 %) | 2.3 ms (3 %) |
| device total per pass | 54.5 ms | 73.0 ms |

Step wall equals device time (55.0 against 54.5 ms), host mel is ~11 % of one
thread, the label loop's host syncs are 0.4 %: by the registered table the
serving, host-mel, label-loop and launch rows are ruled out, and the lever is
the GEMM (~75 % of device time). FFN1 fitted over the two points: ~7 ms fixed
per pass (2.7 ms of it f32 weight bandwidth) + 0.062 ms per row, i.e. ~6.5
TFLOP/s effective, 27 % of the card's sustained FP32.

**The cuBLAS arm** (same profile, `--gemm cublas`): device per pass 29-34 ms,
FFN1 ~3.1 ms fixed + 0.036 ms per row; lag p95 82 ms and fin p95 112 ms at
C=160 in a DIAGNOSTIC run. **But cuBLAS is MEASURED not row-stable on the L4**:
on all 12 model shapes rows differ bitwise between cohort sizes from M=1
(`tests/test_cuda_kernels`, now an INFO line). Gate A had passed with cuBLAS on
5 clips only because the differing bits did not change those transcripts.
**DECISION: cuBLAS cannot be the default (contract 4).**

**S14-8a, GEMM v2 — REJECTED as the default.** Same per-element chain as v1
(byte-identical, gated for 6 tile configurations x 10 cohort sizes x the
epilogues), templated tiles, register-staged double buffering, float4 loads,
a tile chosen per call. Measured: its best tile wins 10-15 % on a few shapes
and loses on most; both v1 and v2 stay 2-4x behind cuBLAS. **FACT, and it
bounds this whole family:** one sequential fma chain per output element leaves
M x N independent chains, too few at small cohorts with a large K (FFN2:
K = 4096, N = 1024), which is exactly where cuBLAS splits K. No tiling of the
v1 chain reaches cuBLAS. v1 stays the default; v2 is the `--gemm own-v2` arm.

**What the next GEMM lever must be, therefore:** a different accumulation
ORDER — a split-K with a split fixed by K alone and a fixed-order reduction
(row-stable by construction: nothing depends on M), or a pinned cuBLASLt
algorithm (row stability then MEASURED with the same sweep), and later bf16
tensor cores. Every one of these changes the bits relative to v1, so it is a
numerical change (ENGINEERING.md §9): gate A (batch identity), gate B (CPU f32
transcripts), a new unloaded reference, and a WER/CER check on the bank before
any speed is quoted. That is a decision for the owner, recorded as S14-8b.

## S14-8b split-K (decision 2026-09-26: split-K first, then bf16)

Contract, stated before the numbers: same input row, same output row whatever
the cohort; NOT bit-identical to v1 (the association of the sums changes).
Split count S = f(N, K) only, each split the ascending chain over its k range,
partials reduced in a fixed order, then the epilogue.

**Kernel gates** (`tests/test_cuda_kernels`, L4): row-stable byte for byte on
all 12 shapes over 12 cohort sizes; error against a double reference smaller
than v1's on every shape (e.g. FFN2 1.25e-6 against 5.43e-6); 1.1-3.5x faster
than v1 (FFN2 2-3x, the projections 1.5-3.5x), 1.3-1.7x behind cuBLAS in the
serving range.

**Transcript gates:** gate A (batch identity, slot independence) and gate B
(CPU f32 == GPU) green on the 5 committed clips. **Bank quality gate**
(`gpu/tools/transcript_ab.py`, registered bound: corpus WER not worse than v1
by more than +0.002): the unloaded split-K reference over the 498 qualification
clips is **byte-identical to v1's on 498 of 498 clips**; WER corpus 0.14483,
mean 0.10515, CER mean 0.06737 for both. PASS.

**Profile** (DIAGNOSTIC): device per pass 37.6 ms at C=128 (v1 54.5, -31 %),
FFN1+FFN2 15.7 ms (v1 26.1, -40 %); attention core becomes the second largest
stage (16-18 %). **WAVE ladder** (120 s, qualification bank): C=128 lag p95
118 / fin 159 ms; **C=144 118->148 / 198 ms, backlog 0.384 s, clean** (v1 failed
C=144); C=160 fails (492 / 645 ms); C=192 fails. FACT about the knee: at C=160
one 120 s run passed (profile, 173 / 220 ms) and another failed (ladder,
492 / 645 ms) -- near the knee the queue is bistable across runs, which is why
only the 2x30-minute soaks decide.

## Explicit non-goals and rejected shortcuts

- No per-op offload of the CPU step (the 2026-07 `cuda_gemm.cu` seam stays
  as it is for the CLI; it is not the serving path).
- No change to any transcript-bearing default of the CPU path; no shared
  scheduler; no `#ifdef CUDA` inside `src/encoder.c` or `server/sched.c`.
- No TF32 by default; no bf16 activations without the CER gate.
- No hidden CPU fallback in the GPU server.
- No prefork after CUDA init; no `-arch=native` in CI or in a release.
- No number quoted from this note: every figure above is a byte count or a
  HYPOTHESIS until S14-6 measures it on a named card.

## Evidence

### 2026-09-26, development host (macOS, no GPU): the server, on the cpu reference engine

What ran: `make -C gpu cpu` (the server linked with `gpu/engine_cpu.c` and the
"not compiled" cuda stub), `models_local/nemotron-3.5-asr-streaming-0.6b`
(pre-quantized int8 through the library), `--cap 4 --threads 12
--engine-threads 4`, plus two more instances for the short-timeout and cap-1
cases. The CPU server's own probes, unchanged:

| probe | result |
|---|---|
| `ws_probe utterances` (3 clips, finalize/reset on one socket) | OK, byte-identical to the CLI, `seq` continuing |
| `ws_probe control` / `bad-lang` | OK, error frame and the session survives |
| `ws_probe http` unknown query key -> 400 | OK |
| `ws_probe idle --wait 8` (idle-ms 2000) | OK, `idle_timeout` |
| `ws_probe ping --wait 3 --min-pings 2` (ping-ms 1000) | OK, 2 pings |
| `ws_probe http` second stream at cap 1 -> 503 | OK, `server_at_capacity` + `Retry-After` |
| `ws_probe hold --expect-code audio_limit` (max-audio-seconds 1) | OK |
| `fault_probe suite` (16 cases: RST/FIN mid-utterance and silent, close-then-RST, unacked RST, half-close, idle-open, stall, oversize, rsv-bits, bad-control, garbage, capacity, neighbours, 40-abort loop) | 0 failed invariants after the fixes below; books balanced at every check; audio fed after the peer went away 0.00 s in every zombie case; RSS +0.7 MB over 40 aborts |

Three defects the suite found in the new code, each fixed before this line
was written: (1) the PCM ring returned a partial push instead of BLOCKING like
`server/slot.c`, which cut off a client ahead of real time (`neighbours`); (2)
the cpu reference engine's `need_samples` ignored what was already staged, so
the engine thread staged a blasted utterance whole and the audio counter said
"fed" before the model ran (`*-silent` discriminating checks, and a 28 s finalize
that could not see a RST); (3) `--threads` doubled as the engine pool size, so
`--threads 4` with cap 4 starved the HTTP pool (`capacity`). None of them was
in the CUDA kernels; all three were in the seam and the server, which is
exactly what the cpu reference engine exists to expose on a machine without a
GPU.

Not run: anything on a GPU. The cuda engine (`gpu/cuda/*.cu`) is
compile-verified by the CI job only (run 36240377764 green: nvcc 12.6 built
and linked the server and both tests for sm_80/86/89/90, the kernel test
reported SKIP 77 with no device, the server refused with "no CUDA device"), `tests/test_cuda_kernels` and
`tests/test_cuda_stream` have never executed. This is the state S14-6 starts
from.

### 2026-09-26, first GPU: NVIDIA L4 on a rented container (S14-6)

Box: one L4 (sm_89, 22.5 GiB with ECC, driver 595.84, CUDA 12.8), an
unprivileged container with a 24.5-CPU cgroup quota on a shared 128-core host
(the loadavg the container sees, ~28, is the HOST's). Build `CUDA_ARCH=sm_89`
from a clean clone of the branch; raw evidence in
`.work/evidence/gpu-l4-20260926/` (untracked).

**Is the GPU what it claims** (`gpu/tools/gpu_doctor.sh`, idle box):

| check | measured | reading |
|---|---|---|
| P-state | P8 idle, P0 under load, memory at 6251 MHz | full clocks |
| power limit | 72 W = default = max | not lowered by the provider |
| MIG / other processes | none / none | the whole GPU |
| VRAM allocatable | 22272 of 22565 MiB (98.7 %) | no cap |
| device bandwidth (copy r+w) | 210 GB/s, 70 % of 300 | normal for GDDR6 |
| FP32 FMA sustained | 23.8 TFLOP/s, 78 % of 30.3 | `sw_power_cap` at 72 W holds the SM at ~1570 of 2040 MHz: the L4's own TDP, the same on any L4 |
| PCIe pinned | 25.4 / 22.1 GB/s | Gen4 x16 |

**Gates, first run on a device, all green:** `tests/test_cuda_kernels` 51/51,
including the GEMM row-stability BYTE gate on all 12 model shapes and lane
identity of the attention; `tests/test_cuda_stream` gate A (batch identity and
slot independence, 5 clips, 5 languages) and **gate B: the GPU f32 transcripts
are byte-identical to the library's CPU f32 stream path on all 5 clips**, both
with the own GEMM and with the cuBLAS arm.

**Two defects the protocol probes found on the device** (the engine-level test
could not): the server handed `finalize` to the engine with audio still queued
(empty transcripts), and repeated resets of an idle slot overflowed the pinned
reset list (device error, engine dead, exit 70). Both fixed in `aea98e9`; then
7/7 protocol probes and the 16-case fault suite green on the GPU (0 failed
invariants, books balanced, 0.00 s fed after a peer left, RSS flat over 40
aborts).

**WAVE ladder** (screening, never promotes; 120 s per rung, fresh server,
`lang=en`, lookahead 3, `--cohort-ms 40`, stress-en all 1587 clips, generator
on the same box; stream_load's own TTFP line is the provisional envelope, which
v2_verdict reports and does not gate):

| C | lag p95 (client) | fin p95 | backlog max | lost | cohorts | lanes/cohort | step ms | GPU util |
|---|---|---|---|---|---|---|---|---|
| 32 | 86 | 128 | 0.084 s | 0 | 2585 | 5.0 | 42.5 | 68 % |
| 64 | 103 | 140 | 0.184 s | 0 | 2535 | 10.2 | 47.4 | 82 % |
| 96 | 137 | 191 | 0.184 s | 0 | 2078 | 18.8 | 58.9 | 81 % |
| 128 | 178 | 237 | 0.384 s | 0 | 1800 | 28.7 | 66.3 | 78 % |
| 160 | **625** | **800** | 0.864 s | 0 | 1128 | 56.1 | 100.9 | 78 % |

Reading, labelled: **RESULT (WAVE)** C=128 is inside the latency bounds
(320 / 500 / 0.640) and C=160 is out, the same knee region as the 30-core Axion
fleet (C=144 qualified, 152 fails). **FACT** the fixed cost of a step is large:
42 ms at 5 lanes, and only +59 ms from 5 to 56 lanes, so a lane costs ~1.2 ms
while the pass costs ~36 ms before any lane — the byte count predicted 8 ms of
f32 weight traffic, so the fixed term is kernel inefficiency, not bandwidth.
**HYPOTHESIS** (S14-8): the own GEMM launches N/64 × M/64 blocks, i.e. 16 blocks
for a 1024-wide output at small M on 58 SMs, and the label loop synchronises
once per iteration; a small-M tile and fewer syncs are the first levers, and
the GPU utilisation figure (68-82 %) is the nvidia-smi "a kernel was running"
fraction, not SM occupancy.

### 2026-09-26, the V2 qualification of the frozen CUDA F32 v1 baseline (L4)

`gpu/tools/gpu_qualify.sh`, commit `075c8d2`, binary `23c9efd5…`, bank
`04a7753aa1e80f9a` (498 clips, the SAME bank as the CPU C=128/144/152 runs),
`lang=auto`, lookahead 3, `--cohort-ms 40`, cap 192; unloaded reference at C=4
from the GPU server itself; verdict by the unmodified `tools/bench/v2_verdict.py`.
Evidence: `.work/evidence/gpu-l4-20260926/qual-20260926T134405Z/`.

**RESULT, stated exactly:** CUDA F32 v1: C128 short-run passed all latency
gates; formal 2x30m qualification failed because soak #1 exceeded backlog_max
by 44 ms (0.684 vs 0.640 s). All other reported gates passed. Optimization and
profile campaign continues from the frozen baseline. C=128 is therefore NOT
"qualified" and NOT "safe" on this GPU build.

| run | lost | lag p95 | fin p95 | backlog max | drift worst | parity | TTFP paired p95 | audio/wall | verdict |
|---|---|---|---|---|---|---|---|---|---|
| ladder C=96 (90 s) | 0 | 114 | 158 | 0.184 | 116 | PASS | +60 | 62.5x | pass (no trend over 90 s) |
| ladder C=128 | 0 | 165 | 224 | 0.344 | 178 | PASS | +94 | 89.2x | pass (no trend over 90 s) |
| ladder C=144 | 0 | 286 | 369 | **0.784** | **352** | PASS | +184 | 103.5x | NOT QUALIFIED |
| ladder C=160 | 0 | **611** | **814** | **0.884** | **675** | PASS | +422 | 110.3x | NOT QUALIFIED |
| soak 1 C=128, 1800 s, seed 42 | 0 / 17016 | 177 | 237 | **0.684** | 213 | PASS | +104 | 123.3x | NOT QUALIFIED (backlog only) |
| soak 2 C=128, 1800 s, seed 43 | 0 / 17023 | 164 | 223 | 0.584 | 180 | PASS | +91 | 123.8x | QUALIFIED |

Both soaks: books balanced in all 63 dumps, RSS 1.021x, one process throughout,
fairness 1.04x / 1.02x, 0 deltas over 1280 ms. For orientation only (not a
claim; different hardware class and different cost): the 30-core Axion CPU
fleet qualified C=128 at lag p95 129 / fin 221 ms and ~124 audio-s/s on the
same bank.

**FACT from the server's own counters** (cohorts, lanes per cohort, step wall,
cohort wait): the single engine thread spends 86 % of wall inside `step` at
C=128 (33 lanes, 68-71 ms per step) and the cohort wait averages ~80 ms against
the 40 ms setting, i.e. a cohort waits for the previous step to finish. The
serial step chain, not the GPU, is saturating; what inside the step costs the
time is S14-6b's question.

Per GPU phase, this section will record the box, the commit, the binary hash,
`nvidia-smi` identity, the `--dispatch-map` output and the artefact path
(untracked under `.work/evidence/`).

## Conclusion

Design accepted 2026-09-26 for implementation as S14-2 … S14-5 (code) with
compile-only verification on CI, then S14-6/S14-7 on a GPU box the moment
one is available. The claim on the table — C=128 on a small-vCPU GPU
instance — is a hypothesis with a byte-count argument behind it, not a
result.

## Next action

Implement S14-2 and S14-3 (`gpu/cuda/`), S14-4, then S14-5 (`gpu/server/`);
extend CI; then bring-up on a GPU box (S14-6).
