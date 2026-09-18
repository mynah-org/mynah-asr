# Serving v2 — a concurrent streaming ASR server on CPU

Status: OPEN (design accepted 2026-09-18; implementation tracked as S0–S5 on the board)

Task: design reference for epics S0–S5
Question: how does one `mynah-asr-server` process tree serve Nemotron (streaming,
40 languages), Parakeet and Canary at the same time, with tens of concurrent
real-time streams on a 32-core Linux box, without latency spikes or stalls?

## 1. Where we start (audit of HEAD `5f0f802`, 2026-09-18)

The compute core is solid: streaming ≡ offline byte-identical, oracle parity per
stage, int8/int4 with NEON SDOT and AVX-512 VNNI dot kernels, single-stream cost
about 26 ms per 80 ms chunk in f32 and 9 ms in int4 on Apple Silicon. The
**server is a 743-line prototype** and is the part that is rewritten:

- fixed blocking worker pool (`--threads`, default 4); a WebSocket stream holds a
  worker for its whole life, so the stream ceiling *is* the thread count;
- no timeouts, no server pings, no graceful shutdown, `Connection: close`;
- REST micro-batching by a single thread with a 25 ms window; streams bypass it;
  the batch takes `lookahead` from its first job;
- BLAS thread count derived as `ncpu / inflight`, process-global, no-op on
  Accelerate, disabled if `OPENBLAS_NUM_THREADS` is set; `parallel_for` runs
  inline and serial as soon as two requests overlap;
- about 55–60 malloc/free pairs per streaming chunk around an allocation-free
  encoder step (subsampling, encoder post, decode, SiLU on Accelerate, full
  re-detokenisation each chunk);
- one model per process; `/v1/models` hard-codes the Nemotron name;
- `/v1/health` has four numbers; nothing else is observable;
- the streaming API has no `reset`, does not surface `is_final`/`is_eou` over
  the wire, and has no batched stream step, so streams cannot share a GEMM;
- no tool opens N WebSockets at real time and reports per-chunk latency; the
  only concurrency number in the repo is offline, on an A100 host.

Full audit with `file:line` pointers is in the session that produced this note;
the load-bearing facts are repeated in the task notes.

## 2. What the sibling repos already learned (do not re-learn)

qwen-tts reached qualified points of C11–C12 real-time TTS streams on 32 Zen5
cores and C12 on 32 Graviton cores; mynah-tts ported the design and measured a
GOOD 600 s soak at C99 on a 32-core Axion with a small model. Their doctrine, in
the order that mattered:

1. **Server design, pinned core split and batching first; dataflow second;
   kernels third.** A 36x kernel win left the machine at two or three streams.
2. **Narrow pinned workers beat wide pools monotonically.** 16×2 beat 1×32 by
   2.2x on the same box; a wide pool wakes 28 threads to have 20 run a chunk and
   pays 31% of its time in barriers. One worker per cache domain (CCX / L3).
3. **Every change that removed work from the serial critical path won. Every
   change that added a second submitter, a gate or a parking policy on the
   same pool lost.** Falsified, with numbers: priority helper threads (TTFA
   435→2379 ms), same-pool second consumer (167→1200 ms), utilization-aware
   admission (established streams' stall rate 0→50%), lead/credit fairness
   gates, capping admissions per iteration, shrinking the quantum under load.
4. **When the machine is full, refuse.** A visible 503 beats an invisible wait;
   never gate the listener on free slots (97% of a 4.5 s tail hid before
   `accept()`).
5. **Output must never be on the inference thread**; a slow reader stalls
   everyone otherwise (66.5 s → 2.6 s to serve the next client once fixed).
6. **A mean rate fails last.** STREAM_RTF < 1 with 25% of streams stalling is a
   real measured state; choose serving points on cadence percentiles, never on
   aggregate RTF.
7. **The workload mix is a first-class variable**: the same C99 was GOOD on a
   medium-length bank and NOT STREAMABLE on a mixed one.
8. **Prove dispatch inside the timed run** or the number is not a measurement.

## 3. ASR is not TTS: what changes

| | TTS (siblings) | streaming ASR (here) |
|---|---|---|
| who paces | the server produces audio; the client consumes at 1x | **the client produces audio at 1x**; the server consumes |
| the failure | player underrun (stall) | **backlog**: audio received but not yet encoded grows, emission lag grows without bound, then timeouts |
| natural quantum | decode frames per emit (1/2/4/8 ramp) | the encoder chunk of the lookahead preset: 80/160/**320**/560/1120 ms of audio, fixed by the model pack |
| arrivals | requests arrive at random; batching across them is opportunistic (2.7% coincidence) | **chunks arrive on the real-time grid**; every 320 ms every live stream has one chunk ready. Within-worker batching of the stream step is natural, not opportunistic |
| the per-slot serial stage | speech decoder (80% of marginal cost) | RNNT greedy loop: LSTM pred-net + joint per emitted token, per stream, a few ms |
| language | one model per language, a batch cannot mix | one Nemotron for 40 languages, prompt is a per-row post-encoder one-hot: **a batch can mix languages** |
| partial results | audio chunks | Nemotron emits **finals only** (monotonic greedy); a "partial" is a text delta |
| memory per slot | KV of an AR transformer | ~13 MB per stream (K/V cache 11 MB + conv cache + scratch), model shared |

**The ASR cadence law.** A worker holding B real-time streams at chunk period
`P` (320 ms at the default preset) keeps up only if the batched step for B
streams plus their B decode loops fits in `P` with margin:

```
T_step(B) = a + b·B  ≤  ρ · P       (ρ ≈ 0.7 for jitter headroom)
```

`a` is the fixed cost of one pass over the weights (bandwidth-bound), `b` the
per-stream cost (decode loop, per-row activation work). The capacity of a
worker is the largest B that satisfies it; the capacity of the box is the sum
over workers. Both `a` and `b` are **measured** per host and per ISA in S0/S4,
never predicted, and the per-worker slot cap is derived from them.

Consequences that shape the design:

- **Load never enters the chunk size.** A stream's quantum is its preset. When
  a worker cannot hold another stream, the parent refuses; it does not degrade
  everyone.
- **Fairness is by construction**: each slot consumes at most one chunk per
  step. A client that uploads a file faster than real time fills its bounded
  input ring and is then throttled by TCP; it cannot take more than one chunk
  per step from the others.
- **The first delta is the TTFP** (time to first partial): the first step after
  the first full chunk runs as soon as that chunk lands, never waiting for a
  tick, and a stream joins the running batch at the next step boundary.

## 4. The architecture

```
parent (router, never touches a model)
  listen ─ accept ─ MSG_PEEK ≤ 8 KiB → model group (query ?model= / multipart field / default)
        → least-loaded worker of that group with active < cap → SCM_RIGHTS → active[w]++
        ← one byte per finished connection ← active[w]--
  refuses with a real HTTP status + Retry-After, shutdown(SHUT_WR) + bounded drain, never RST
  /metrics on its own port; SIGUSR1 dump forwarded to every worker; waitpid(WNOHANG) reaper

worker group "nemotron" (W workers, one model mmapped COW, each pinned to a core-major slice)
  ingest threads ×k   WS framing, s16→f32 into the slot's bounded PCM ring, control messages,
                      REST parse into a job. Never touch the model.
  scheduler ×1        the ONLY thread that enters the model (asserted, not commented):
                      loop { admit at step boundary; gather slots with a ready chunk;
                             batched mel (cheap, per slot); batched enc_stream_step [Σ Q, d];
                             per-slot greedy decode; emit deltas to stream_out;
                             poll cancelled() once per slot per step }
  stream_out ×1/conn  detached writer, bounded byte ring, backpressure = cancel, peer-gone detection
  decoder lane        optional pinned sub-team for the decode loops (OFF by default; measured before ON)

worker group "parakeet-tdt" / "canary-1b-v2" (offline-only models)
  same skeleton; a job is a file, the scheduler feeds it as segments of bounded length per step
  (the existing transcribe_batch on segments), so one long file costs the others one step, not a minute.
```

Rules the code enforces, taken from the siblings verbatim where possible:

- **Prefork, core-major, from the inherited affinity mask** (`sched_getaffinity`,
  cgroup `cpu.max` read and warned about). Pin *before* creating any thread.
  Print the mask actually set, not the one configured.
- **One process, one model.** Multi-model is worker groups, not an in-process
  registry: `--model name=dir[:workers=W][:cpus=N][:cap=C][:quant=int8]`,
  repeatable. A batch is drawn from one process, so it is one model by the
  shape of the address space.
- **Admission ladder** (per group): slot cap → bounded parent queue (default 1
  per live worker) → queue deadline checked at pop → per-stream service cap
  (max audio seconds, max idle) → `model_not_found` 404 / `model_not_streaming`
  400 / `language_not_served` 400, the only refusals without `Retry-After`.
  The listener is always in the poll set.
- **Sink contract** between transport and engine: `next_job / on_delta /
  on_done / cancelled / running`. The greedy loop never sees a socket.
- **Allocation-free step.** The per-chunk mallocs go into per-slot scratch
  carved once at slot open; slots are pooled and `reset`, not freed and
  reopened.
- **Header at admission, `TCP_NODELAY` everywhere, `SO_RCVTIMEO`/`SO_SNDTIMEO`,
  server-side WS ping**, graceful shutdown that drains and answers 503 to what
  it never served.
- **Warm-up through the request path's own reset**, so the first real request
  is not different from every one after it.
- **Effective-config banner** at start: model, quant, ISA kernels resolved (from
  the dispatcher's own predicate), threads per worker, masks, BLAS ownership,
  every env flag read with `requested | effective | reason`.
- **BLAS leaves the worker.** A second thread pool inside the address space is
  a permanent source of non-reproducible measurement (63 threads on an 8-cpu
  slice with OpenBLAS linked vs 32 without, measured on the Axion). Own
  f32 GEMM (lift `src/sgemm.c` from mynah-tts) plus int8 batched kernels;
  keep `BLAS=openblas` as a comparison build forever, never the default.

## 5. Wire protocol (WebSocket, `/v1/audio/stream`)

Query: `model=`, `lang=`, `lookahead=`, `format=s16le` (16 kHz mono; other
rates resampled server-side and said so). Client → server: binary frames of
PCM (any size, bounded ring), text frames `{"type":"finalize"}` and
`{"type":"reset"}`. Server → client, every frame carrying `seq`, `audio_s`
(audio consumed so far) and `lag_ms` (wall time from the last sample of the
chunk that produced this text arriving to this frame being written):
`{"type":"delta","text":...,"final":true,"lang":...}`,
`{"type":"eou","t":...}`, `{"type":"done",...}`,
`{"type":"error","code":...,"message":...}`. Refusals happen **before** the
upgrade completes, as HTTP 503/400/404 with the OpenAI error shape, so a client
sees a status and not a transport error. Detail: `ws-protocol-v2.md`.

## 6. What is measured, and what promotes

Defined once in `tools/bench/streaming_metrics.py` (S4), imported by every
harness. Per stream: **TTFP** (first delta after first audio byte), **emission
lag** p50/p95 (server-side `lag_ms`, and client-observed end-to-end),
**finalization lag** (last audio → `done`), **backlog** (audio received, not yet
encoded) max, **chunk service time** p50/p95 per worker, **STREAM_RTF**
(compute seconds / audio seconds, a capacity number, not a promise), rejects /
errors / timeouts, drift across windows. Vocabulary: WAVE screens and may
disqualify; **a SOAK at fixed concurrency for ≥ 10 minutes on a mixed-length
bank with a drift gate is the only thing that promotes**; POISSON for overload;
DIAGNOSTIC never quotes a number. The harness refuses to print cadence
percentiles when the client could not pace at 1x.

Provisional envelope (recalibrated 2026-09-18 on the M1: TTFP carries the model's
emission delay, measured ~950 ms at lookahead 3 with an explicit language on a
clip that speaks from t=0, so TTFP p95 < 3 chunk periods + 200 ms and the
server's own cost is the emission-lag line); emission lag p95 < 1 chunk period; finalization lag p95 < 500 ms;
backlog max < 2 chunks; zero rejects at the operating point; a new arrival
raises no established stream's emission lag p95 above the gate; a stopped
reader affects only its own stream.

## 7. Order of work and why

S0 measure the current server as-is under N real-time streams on the Axion:
the baseline, `a` and `b`, the arrival-phase distribution that justifies
within-worker batching. **Nothing else starts before this number exists.**
S1 library seams the server needs (reset, sink, exposed final/eou, batched
stream step, allocation-free chunk, pool lift, BLAS decision).
S2 the server: prefork → scheduler → admission → stream_out → protocol →
multi-model. Each step keeps `make test-server-concurrency` green and adds
its own byte-identity gate (N concurrent streams == the same streams alone).
S3 observability; S4 the harness and the first SOAK; S5 kernels for the
batched stream step (SMMLA / VNNI / KleidiAI), gated by the S0 profile that
blames them.

Out of scope for v2.0, tracked separately: cache-aware streaming for
Parakeet/Canary (offline-only in v2.0, WS on them is a 400); unstable-tail
partials for Nemotron (a decoder change); beam search; GPU serving.

## 8. Ideas already falsified in the siblings — a re-try needs a new argument

Wide pool one process per box · a second submitter on the engine pool ·
priority or "helper" threads on a full machine · utilization-aware admission
· lead/credit gates parking ready work · a larger quantum for a better RTF ·
shrinking the quantum under load · capping admissions per iteration ·
per-stream bump arenas before measuring allocation counts · cache tricks (NTA
prefetch, hot workers) · elastic re-slicing of CPUs on every completion.
The one that is *different* for ASR and worth measuring: cross-stream batching
of the encoder step inside a worker, because arrivals here are on a grid.
