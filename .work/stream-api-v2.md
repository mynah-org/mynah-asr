# S1 — library seams the v2 server needs

Status: OPEN

Task: S1-1, S1-2, S1-3, S1-4, S1-7, S1-8
Question: which changes to `libmynah_asr` are required so a single scheduler
thread can drive B streams as slots, allocation-free, with cross-stream
batching of the encoder step, without changing any transcript?

Known facts (HEAD 5f0f802)
- `mynah_asr_stream_open/feed/finish/close` only; no reset. A session restart
  re-allocates ~13 MB and re-opens a VAD instance.
- Deltas are always final (`is_final=true`); `t0` is always 0; `is_eou` is
  emitted with empty text and the server drops the distinction.
- Feed re-detokenises the whole token history each chunk and mallocs the text.
- `mynah_asr_enc_stream_step` is allocation-free, but subsampling, encoder_post,
  greedy decode and (macOS) SiLU allocate per chunk.
- `mynah_asr_encoder_forward_batch` exists for offline only; there is no batched
  stream step. Weight-stationary batching of B slots' chunks would turn 24×B
  skinny GEMMs of Q rows into 24 GEMMs of B·Q rows.
- A model is shareable read-only across threads; the mutators
  (`set_decoder`, `set_target_lang`, `set_segment_limit`, `enable_vad`) are not
  thread-safe and must be called before serving.

Unknowns
- Whether a batched stream step can be made **byte-identical** to B single
  steps (it must: the transcript of a stream must not depend on who it was
  batched with). BLAS sgemm on a [B·Q, d] matrix may differ in accumulation
  order from Q rows alone; the own-GEMM path (S1-6) has the deterministic
  row-block contract, so S1-4 may depend on it.
- Whether the decode loop should move to a lane; only after S0 blames it.

Plan
- S1-1 `mynah_asr_stream_reset(s, lang, lookahead)`: clears caches, decoder
  state, VAD state, counters; keeps the allocations. Slots are pooled by the
  server. Gate: reset+feed ≡ close+open+feed byte-identical on the fixtures.
- S1-2 result struct gains `t0` per delta (frame of the first emitted token),
  `chunk_arrival` passthrough (caller-provided timestamp of the last sample of
  the chunk, returned on the delta so the server can compute `lag_ms`), and
  `is_eou` stays; server surfaces `final`, `eou`, `t0/t1`. No text change.
- S1-3 allocation-free chunk: per-stream scratch for subsampling/post/decode
  carved at open; incremental detokenisation (emit only new pieces; keep the
  ▁/UTF-8 boundary logic); SiLU scratch. Gate: LD_PRELOAD count per chunk = 0
  after warm-up, transcript byte-identical (`make test`).
- S1-4 `mynah_asr_enc_stream_step_batch(streams[], B)`: one pass over the
  weights per layer for B slots, then per-slot decode. Gate: for B ∈ {2,4,8},
  ragged presets allowed or refused explicitly, output byte-identical to B
  single steps on the same slots; the refused path is the old path.

Acceptance gate: `make test` and `make test-server-concurrency` green; the
new tests above green; `docs/api.md` updated; no transcript in `tests/` changed.

Evidence
- S1-1 DONE 2026-09-18 (commit 794f1c7): `mynah_asr_stream_reset(s, lang)`,
  `mynah_asr_stream_need_samples(s)`, `mynah_asr_stream_audio_seconds(s)`;
  mel/subsampling/encoder stream resets added underneath. Gate in
  tests/test_streaming.c with the local Nemotron f32: a stream polluted with
  the second half of the clip, reset, then fed exactly `need_samples` per call
  produced text byte-identical to the offline path (17 feeds, 14 of them with
  a delta, i.e. one encoder chunk per feed); a bad language is refused and the
  stream stays usable. S1-2 turned out to need no library change for the
  server (the scheduler knows when each sample arrived; `t1` maps a delta to
  its audio position).
- S1-3 DONE 2026-09-18: see the section below.
- S1-4 DONE 2026-09-18: see the section below.
- S1-7 DONE 2026-09-18: see the section below. Bit-exact as predicted (0 of
  266,240 floats differ at B = 8, both dtypes, counters proving the sharing);
  the slope `b` measured 22.7 -> 20.3 ms on the M1 under third-party load, a
  third of the 7.4 ms the isolated micro-benchmark had promised.
Conclusion: the seams the scheduler needs exist.
Next action: S2 wires the batched step into the scheduler (recipe below).

## Evidence — S1-3 (allocation-free chunk), 2026-09-18, macOS arm64 (Accelerate)

Measurement (`tests/malloc_count.c` inserted with `DYLD_INSERT_LIBRARIES` +
`DYLD_FORCE_FLAT_NAMESPACE`, `tests/test_stream_allocs.sh`): the counter is a
whole-process total, so the per-chunk number is the DIFFERENCE between two runs
of the same command on the same model with inputs of different length — model
load, stream open, WAV load and the final flush are identical in both, so the
difference is exactly what the extra chunks cost. Command in both runs:
`./mynah-asr stream -m <nemotron-3.5-asr-streaming-0.6b> -i <wav> --quant int8`,
preset [56,3] (q = 4 encoder frames = 320 ms per chunk), `tests/audio/test_it.wav`
(5.229 s) against its first half (2.615 s), i.e. 8 steady-state chunks.

| tree | 2.615 s | 5.229 s | second half | per chunk |
|---|---|---|---|---|
| `354743c` (before) | 45975 | 46671 | 696 | **87** |
| this change (after) | 45171 | 45171 | 0 | **0** |

The 87 are exactly the audited sites: SiLU on Accelerate 24 × (2 FFN + 1 conv
module) = 72 · subsampling 9 (`a`, `bbuf`, `flat`, `xp` × 3 stages, the stage-0
im2col `P`, the depthwise pad × 2 stages — the audit listed 6, the im2col and
the two pads were inside `conv2d_s2`) · `mynah_asr_encoder_post` 3 (`cat`,
`mid`, `fused`) · `mynah_asr_greedy_decode` 2 (`jin`, `logits`) ·
`mynah_asr_detokenize` 1 (the whole transcript, re-decoded every chunk).

What was done: every one of those buffers is now carved at stream open —
`mynah_asr_ss_stream` grew its own `scr` (sized from `max_n_mel = 8*(right+1)+1`,
`sflat` aliasing `sb`), `mynah_asr_enc_stream` grew `ssilu` and `spost`,
`mynah_asr_stream` grew `dec_scr` and a `mynah_asr_detok`. The offline entry
points (`mynah_asr_silu`, `mynah_asr_encoder_post`, `mynah_asr_greedy_decode`,
`mynah_asr_subsampling_forward`) keep allocating: they call the same code with a
NULL scratch, so their arithmetic and their outputs are untouched.

Detokenisation is now incremental (`mynah_asr_detok_append`): the raw transcript
is kept UNSTRIPPED in a per-stream buffer and the ▁-expansion, the inline
`<xx-XX>` strip and the leading/trailing space strip are reproduced as a view
over it, so at every chunk the bytes equal `mynah_asr_detokenize` over the whole
history. Two cases needed care and both are covered by a model-free test
(`tests/test_tokenize.c` §6, every prefix × chunkings of 1..4 tokens):
a language tag spelled out across several tokens (the pass restarts at the first
`'<'` it could not resolve) and the double-space collapse after a tag stripped at
the very end of the buffer (undecidable until the next chunk arrives, so it is
deferred rather than skipped — this was a real divergence the test caught).
There is NO fallback to the whole-history path: no case needed one.

What remains: the transcript buffer (reserved 8192 B) and the token array
(4096 ids) still realloc geometrically if a stream outgrows them, as does the mel
stream's own sample buffer — amortised, not per chunk, and the measurement above
shows none of them fires on a 5 s clip. `mynah_asr_greedy_decode` still
dequantizes the head into a temporary when T > 16; no Nemotron preset reaches
that (the largest, [56,13], gives q = 14), but a model with a larger lookahead
would allocate 1 per chunk there. `MYNAH_ASR_QGEMM=1` (off by default) allocates
inside `mynah_asr_qmat_mul`. Linux is NOT measured here: the counter has an
`LD_PRELOAD`/`dlsym(RTLD_NEXT)` arm and the test skips (77) where interposition
does not work, but the numbers above are macOS/Accelerate only — S0-4 is the
Linux count.

## Evidence — S1-4 (batched stream step), 2026-09-18, macOS arm64 (Accelerate)

### What was built

`mynah_asr_stream_step_batch(streams, B, samples, n_samples, cb, userdata)`
(`src/mynah_asr.h`). It feeds B streams and runs ONE encoder pass over the
chunks that completed: the rows of every completed chunk are stacked into a
single `[Σq_i, d]` activation, so each of the 24 conformer layers runs its
per-row linears — FFN1 ×2, q/k/v/o, pointwise_conv1/2, FFN2 ×2, ten products
per layer — once for all the rows instead of once per stream. Per stream and
inside a loop over `i`: subsampling, the K/V cache, the relative-position
attention (rel-pos projection, per-head softmax, `rel_shift`), the conv cache,
the prompt + projector, the greedy decode and the callbacks. `B == 1` calls
`mynah_asr_stream_feed` verbatim. Streams whose chunk did not complete are just
fed (their mel accumulates). Streams with different lookahead presets are
GROUPED (one stacked pass per preset), not refused. Languages may mix freely:
the prompt is a per-row post-encoder one-hot.

Underneath: `mynah_asr_enc_stream_step_batch` + `mynah_asr_enc_batch`
(`src/encoder.h`), and `mynah_asr_qmat_mul_rows` (`src/qmat.h`), which extends
the native int8×int8 per-row dot path to ANY `T` as a threaded loop over blocks
of weight rows — it reuses `qgemm_block` and the existing SDOT/VNNI/AVX2 dot
kernels, so **no new SIMD kernel was written** (that is S5-1). `stream_attention`
and `stream_conv_module` were split into a core plus the hoisted linears, and
the single path is those pieces reassembled in the same order.

`mynah_asr_qmat_mul` itself is UNCHANGED: the offline dispatch, and therefore
the offline numerics, are untouched. `make test` transcripts are identical.

### Gate A — identity (`tests/test_stream_batch`, model-gated, 77 without Nemotron)

Two levels. Level 2 is the strict one: the encoder output of every chunk of
every stream, float for float (`memcmp`), plus the final K/V and conv caches,
between B single `mynah_asr_enc_stream_step` calls and one batched step over the
same chunks. Level 1 is the transcript: B streams over
`tests/audio/test_{it,en,de,fr,es}.wav` (5.23 / 4.34 / 3.88 / 3.57 / 3.74 s, so
the streams finish at different steps and the batch shrinks) fed one chunk each
per step through the public API with `need_samples`, compared against
`mynah_asr_transcribe` and against a single-stream `stream_feed` run.

Model `nemotron-3.5-asr-streaming-0.6b`, preset [56,3] (q = 4 encoder frames =
320 ms per chunk), 17 steps per clip.

| quant | B | floats compared | differ | caches differ |
|---|---|---|---|---|
| int8 | 2 | 74,240 | **0** | 0 / 6 |
| int8 | 4 | 133,120 | **0** | 0 / 12 |
| int8 | 8 | 266,240 | **0** | 0 / 24 |
| f32 | 2 | 74,240 | **0** | 0 / 6 |
| f32 | 4 | 133,120 | **0** | 0 / 12 |
| f32 | 8 | 266,240 | **0** | 0 / 24 |

Transcripts, B ∈ {1,2,3,4,8}, both dtypes: IDENTICAL for every stream, to the
single-stream run and to the offline transcription. One caveat recorded rather
than hidden: at **int8** the offline transcription of `test_es.wav` already
differs from the streaming one ("a las vuelve" vs "a las nueve") — a pre-existing
streaming-vs-offline numerical difference at int8, visible in `make test`'s own
e2e line, NOT a batching effect. The gate prints it and still requires
batched == single exactly.

### WHAT PATH RAN (counters, `mynah_asr_qmat_counter`)

`src/qmat.c` now counts every product by implementation. int8 run, per B
(figures from one gate run):

| B | stacked rows | native dot (serial, T≤16) | native dot (weight-stationary) | DEQUANT+sgemm |
|---|---|---|---|---|
| 1 | 0 | 4173 | 0 | **0** |
| 2 | 104 | 1362 | 3120 | **0** |
| 4 | 196 | 1972 | 3120 | **0** |
| 8 | 416 | 2508 | 3840 | **0** |

The `T > 16` dequant+sgemm fallback — which mallocs `n*k*4` per call and changes
the numerics — is never reached: 0 at every B, asserted by the gate. The serial
column stays non-zero because streams run out of audio at different steps, so
the last streams of a group step alone (g == 1 → the single path), which is the
intended behaviour and is itself covered by the identity gate.

### The f32 question, MEASURED

`cblas_sgemm` gives no guarantee that row `t` of C is the same bytes for `M = q`
and for `M = Σq`, so this was measured, not assumed. On **macOS arm64 /
Accelerate** it IS row-stable for these shapes: 0 of 266,240 floats differ at
B = 8 (table above), caches included. The f32 batched path is therefore ON for
Accelerate — **identical on macOS/Accelerate; OpenBLAS UNVERIFIED (Linux gate
pending)**, so on a non-Accelerate build the default is OFF and the f32 batched
step degrades to per-stream single steps through the same API. The degradation
is visible, not silent: `mynah_asr_stream_batch_rows_stacked()` stays 0.
`MYNAH_ASR_BATCH_F32=0|1` forces either way — that is how the Linux gate runs.
The integer path never consults this switch: it is exact by construction
(per-row integer accumulation is order-independent, the activation quantization
is per row, and each output row is one dot of the same two int8 vectors).

### Gate B — step time (macOS dev signal, NOT a serving claim)

`tests/test_stream_batch <model> --steptime`. Apple M1, 8 cores, int8, preset
[56,3] (P = 320 ms of audio per step per stream), 5 clips cycled. Wall per step,
B separate `stream_feed` calls vs one `stream_step_batch`:

| B | single ms | batched ms | ratio | ms per stream |
|---|---|---|---|---|
| 1 | 109.72 | 107.27 | 0.98 | 107.27 |
| 2 | 194.03 | **84.57** | 0.44 | 42.29 |
| 4 | 352.27 | **133.38** | 0.38 | 33.35 |
| 8 | 693.60 | **227.00** | 0.33 | 28.37 |

B = 1 is the same code on both sides, so 0.98 is the noise floor. Fitting the
cadence law of `serving-v2-design.md` §3 over B ∈ [2,8]: **a ≈ 37 ms, b ≈ 23.7
ms**, i.e. with ρ = 0.7 and P = 320 ms this host holds B ≈ 7 per worker.

Two honest caveats. (1) The win is NOT only weight-stationarity: the old
small-`T` int8 path is fully serial, and the new row loop is threaded, so part
of the 3.06× at B = 8 is threading the linears for the first time. (2) The
absolute numbers are macOS/Accelerate on a dev box; the Linux/Axion and x86
numbers are S0/S4's job.

### Where the per-stream slope `b` goes — a finding, not a change

`b ≈ 23.7 ms` is large, and a third of it is one avoidable GEMM. In
`stream_attention_core` every layer computes `rk = pe @ relk_wᵀ` with
`P = 2K-1 = 119` rows and d = 1024 — 24 × 119 × 1024 × 1024 MACs per stream per
step. Measured on this host in isolation (Accelerate sgemm, same shape, scratch
micro-benchmark): **7.4 ms for the 24 layers**, i.e. ≈ 31 % of `b`. It does not
depend on the stream at all: at steady state every stream has the same
`K = left + q`, so every stream recomputes the SAME matrix. Sharing it across
the streams of a pass that have equal `K` would be bit-exact by construction
(identical inputs, identical call) and should move B_max on this host from ≈ 7
to ≈ 9. NOT done here — S1-4 is scoped to "attention stays per stream" —
recorded as a new board item (S1-7). **Done in S1-7, see below: the sharing is
bit-exact as predicted, but the saving measured end to end is 2.4 ms of the
slope, not 7.4.**

### Allocation

`make test-stream-batch-allocs`: `tests/libmalloc_count` inserted, its counter
sampled in-process (the batched API has no CLI entry to difference two
processes with). B = 4, 12 steps after a 4-step warm-up: **0 allocations**. The
batch scratch is one `malloc` carved by `mynah_asr_enc_batch_new`, sized for
(max_b, max_q + 2); `mynah_asr_stream_batch_reserve(m, max_b)` pre-carves it at
start-up so the first real step does not. Everything else reuses the per-stream
scratch S1-3 already carved.

### What remains unknown

- Linux: OpenBLAS f32 row stability; the int8 identity and the step table on
  Axion / x86 (the counters and the gate are portable, they have not been run).
- x86 int4 in the stacked path: `mynah_asr_qmat_mul_rows` permutes the
  activations in place for the AVX2 q4 kernel; that arm is compiled but not
  exercised here (this host is ARM, and the model is int8).
- Above B ≈ 8 nothing was measured; `MYNAH_ASR_STREAM_BATCH_MAX` is 256 as a
  bound on fixed arrays, not a capacity claim.
- Whether threading the linears inside the step competes with the scheduler's
  other work (S1-5 pool, S5-5 pool meter).

## Evidence — S1-7 (shared rel-pos projection), 2026-09-18, macOS arm64 (Accelerate)

### What was built

In `mynah_asr_enc_stream_step_batch` the rel-pos projection `rk = pe @ relk_w^T`
is now computed ONCE per (layer, K) for the streams of the pass that are at the
same `K = cache_valid + q`, instead of once per (layer, stream).

- `stream_attention_core` gained a last parameter `rk_in`. NULL = compute your
  own into `es->sa_rk`, which is exactly what the code did before and what the
  **single-stream path always passes** — `mynah_asr_enc_stream_step` is byte for
  byte the same computation it was.
- The batched step records each stream's `K` in `bb->kks[]`, picks the `K` shared
  by the most streams (ties to the lowest `K`, so the choice does not depend on
  the order the caller passed the streams in), and, when at least two streams
  are at it, projects once per layer into `bb->rk_sh` from the leader's `sa_pe`.
  Every stream at that `K` reads that buffer; a stream at another `K` — a slot on
  its first chunks, `cache_valid` still below `left` — keeps its private one.
- `bb->rk_sh` is carved inside the existing single `malloc` of
  `mynah_asr_enc_batch_new`, which now takes `max_left` so it can size it
  (`2*(max_left + max_q + 2) - 1` rows × d). `mynah_asr_stream_batch_reserve`
  passes the model's `left_ctx`. Nothing is allocated per step.

Why this cannot change a float: `pe` is `mynah_asr_pos_emb(enc, K)`, a pure
function of `K`; `relk_w` is the layer's weight. Two streams at the same `K`
therefore feed bit-identical inputs of the same shape to the same `matmul_wt`.
Sharing the result is not an approximation, it is common-subexpression
elimination across streams. The gate below asserts it anyway (rule 4).

### WHAT PATH RAN — the counters (`mynah_asr_enc_relpos_counter`)

Three counters, one relaxed atomic per attention core: `SHARED` (a core that read
the group's `rk`), `PRIVATE` (a core that computed its own), `GROUP` (a hoisted
projection, i.e. layer × pass). A run that silently stopped sharing shows
SHARED = 0, and the identity gate fails on it (ENGINEERING.md §6).

Level 1, int8, `tests/test_stream_batch` (5 clips of different length, so streams
leave the batch as they run out of audio):

| B | SHARED | PRIVATE | GROUP | reading |
|---|---|---|---|---|
| 1 | 0 | 408 | 0 | B=1 is the single path verbatim: 17 chunks × 24 layers |
| 2 | 624 | 120 | 312 | 13 shared passes × 2 streams; 5 steps ran with one stream left |
| 3 | 912 | 144 | 312 | |
| 4 | 1176 | 168 | 312 | |
| 8 | 2496 | 192 | 384 | 16 shared passes × 8; PRIVATE = 8 tail steps × 24 |

PRIVATE never drops to 0 and should not: when a group shrinks to one stream the
library takes the single path, which is the intended behaviour. The numbers are
identical at f32.

### Gate — identity, UNCHANGED at both levels

`tests/test_stream_batch <model>`, same two levels as S1-4, now also asserting
`SHARED > 0` whenever the stacked path ran at B > 1.

| quant | B | floats compared | differ | caches differ | SHARED |
|---|---|---|---|---|---|
| int8 | 2 | 74,240 | **0** | 0 / 6 | 624 |
| int8 | 4 | 133,120 | **0** | 0 / 12 | 1176 |
| int8 | 8 | 266,240 | **0** | 0 / 24 | 2496 |
| f32 | 2 | 74,240 | **0** | 0 / 6 | 624 |
| f32 | 4 | 133,120 | **0** | 0 / 12 | 1176 |
| f32 | 8 | 266,240 | **0** | 0 / 24 | 2496 |

Level 1 (transcripts through the public API): IDENTICAL for every stream at
B ∈ {1,2,3,4,8}, int8 and f32, against the single-stream run and against
`mynah_asr_transcribe`. The `DEQUANT` fallback stays 0. The pre-existing
`test_es.wav` int8 streaming-vs-offline difference is still reported and is still
not a batching effect.

`make test` green (`MODEL_DIR` = the local Nemotron pack); the Parakeet/Canary
e2e lines and the golden-dump stages SKIP because those models are not present
on this box. `make test-stream-batch-allocs`: **0 allocations per step** at B = 4,
12 steps after warm-up — the shared buffer did not add one.

UBSan: `tests/test_stream_batch` rebuilt with the Makefile's `ubsan` flags
(`-O2 -g -fsanitize=undefined -fno-omit-frame-pointer`, no `-march=native`, no
`-ffast-math`) and run once on the full gate: **exit 0, no diagnostic**, and both
identity levels still pass in that build — the sharing is bit-exact there too,
not only under `-O3 -ffast-math`.

### Gate B — step time (macOS dev signal, NOT a serving claim)

Apple M1, 8 cores, int8, preset [56,3] (P = 320 ms of audio per step per stream),
5 clips cycled, `tests/test_stream_batch <model> --steptime` (the table now runs
B = 1..8, it used to run 1,2,4,8).

**The box was not idle.** Another agent was building and testing in a sibling
worktree throughout, loadavg 4–11 on 8 cores, and single `--steptime` runs
disagreed with each other by up to 2× — far more than the effect being measured.
So the protocol is: the two binaries (`5a6d612` and this change, same compiler
and flags) alternate, 4 repetitions each, and the estimate per (arm, B) is the
**minimum** over repetitions, which is the value closest to the uncontended step
time. The sanity check that this works: the `single ms` column, which this change
does not touch, comes out the same in both arms (109.92 vs 109.21 at B = 1,
694.13 vs 693.90 at B = 8, ≤ 0.6 %). Everything below is still a DEV SIGNAL under
load, never a serving number (ENGINEERING.md §8).

| B | single ms | batched before | batched after | delta | ms per stream after |
|---|---|---|---|---|---|
| 1 | 109.9 | 107.77 | 107.69 | −0.1 % | 107.69 |
| 2 | 194.1 | 84.34 | **80.42** | −4.6 % | 40.21 |
| 3 | 273.6 | 107.16 | **99.47** | −7.2 % | 33.16 |
| 4 | 347.2 | 128.67 | **117.13** | −9.0 % | 29.28 |
| 5 | 419.4 | 150.67 | **137.17** | −9.0 % | 27.43 |
| 6 | 526.7 | 174.35 | **150.96** | −13.4 % | 25.16 |
| 7 | 615.0 | 198.62 | **175.09** | −11.8 % | 25.01 |
| 8 | 694.1 | 219.87 | **207.83** | −5.5 % | 25.98 |

B = 1 is the same code on both sides and moves by 0.1 %: the noise floor of the
protocol. Fitting `T_step(B) = a + b·B` over B ∈ [2,8]:

|  | a | b |
|---|---|---|
| before (`5a6d612`) | 38.5 ms | 22.69 ms |
| after (S1-7) | 37.0 ms | **20.26 ms** |

i.e. the per-stream slope drops by **2.4 ms (−10.7 %)**, and with ρ = 0.7 and
P = 320 ms this host's `B_max = (ρP − a)/b` moves from **8.2 to 9.2**.

### The prediction was 7.4 ms; the measurement is 2.4 ms

Honest discrepancy, stated rather than smoothed. The S1-4 finding measured the
same GEMM shape (P = 119 × 1024 × 1024, 24 layers) **in isolation** in a scratch
micro-benchmark and got 7.4 ms. Inside the real step the same work is worth about
a third of that. Not investigated here, so the cause is a hypothesis, not a
result: inside the step `pe` and `relk_w` are already resident and the machine is
already saturated by the stacked linears, so the isolated timing measured an
idle-machine cost that the step never actually paid. Whatever the reason, the
number to trust is the end-to-end one: **b 22.7 → 20.3 ms**. The direction of the
S1-4 prediction (B_max ≈ 7 → ≈ 9) survives; the size of the win does not.

The B = 8 point is the weakest: −5.5 % against −13 % at B = 6 and B = 7, and the
two low-load repetitions disagreed there (one showed no gain at all). At B = 8 the
batch saturates all 8 cores and the contention from the other worktree bites
hardest. Do not read the B = 8 cell as a measurement.

### What remains unknown

- The real size of the win on an idle box, and on Linux/OpenBLAS and Axion/x86
  (the counters and the gate are portable; they have not been run there). The
  M1 numbers above were taken under third-party load.
- Why the isolated 7.4 ms does not show up in the step. A DIAGNOSTIC run with
  the projection timed in place would settle it; not done.
- Only ONE K-group is shared per pass, the largest. A pass holding two large
  groups at different `K` — many slots admitted in two waves — shares only the
  bigger one; the rest fall back. Not measured, because the scheduler that would
  produce that shape is S2. Generalising needs one buffer per group.
- The counters are in `src/encoder.h`, so the gate can read them; they are NOT
  surfaced in the server's `/metrics` yet. That is S3 work on `server/`, which
  this change deliberately did not touch.

## Integration recipe for the scheduler (S2, NOT done here)

`server/sched.c` is another agent's file; this is the contract it should use.

1. **Reserve once**, right after the model is loaded in the worker and before
   any slot is served: `mynah_asr_stream_batch_reserve(model, cap)` with `cap`
   the worker's slot cap. After that the step allocates nothing. Warm up through
   the same call so the first real step is not different.
2. **The ready set maps 1:1 onto one call.** Today `sched_feed(slot, avail,
   finalize)` is called per slot in the step loop. Split it: a first pass over
   the slots computes, per slot, `need = mynah_asr_stream_need_samples(s->stream)`
   and takes `got` samples from the ring when `avail >= need`; collect those
   slots into `streams[]`, `samples[]`, `n_samples[]`, `userdata[]` (the
   per-slot `emit_ctx`, so `lag_ms` accounting is unchanged — the callback is
   still invoked once per completed chunk, from the same thread). Then ONE
   `mynah_asr_stream_step_batch(streams, B, samples, n_samples,
   sched_on_result, userdata)`. Slots with `avail < need` are simply not in the
   set; they are not fed at all, exactly as now.
3. **Finalize and short tails do NOT go through it.** A slot that is finalizing,
   or whose last piece is shorter than a chunk, keeps the existing per-slot
   path: `mynah_asr_stream_feed` for the partial piece and
   `mynah_asr_stream_finish` for the tail (the tail is a short chunk with the
   causal right pad — `is_last`, which the batched step deliberately does not
   accept). Do those after the batched call, in the same step, so the ordering
   of deltas within a slot is unchanged.
4. **Cancellation** stays where it is: poll `cancelled()` once per slot per step
   BEFORE building the ready set, and drop a cancelled slot from the set rather
   than cancelling mid-pass. A `-1` from the batched call is a worker-level
   failure: cancel every slot in that set with `decode_failed`, as `sched_feed`
   does today for one.
5. **Do not group by language or by model** — a batch already is one model
   (one process, one model) and languages mix by design. Group by lookahead
   preset only if you want the grouping visible in the metrics; the library
   groups internally either way.
6. **Counters to surface in `/metrics`**: `mynah_asr_stream_batch_rows_stacked()`
   (0 means every step degraded to the single path — on an OpenBLAS f32 build
   that is expected, elsewhere it is a bug) and, for a dispatch banner,
   `mynah_asr_qmat_counter(MYNAH_ASR_QC_DOT_ROWS)` vs
   `MYNAH_ASR_QC_DEQUANT` (the latter must stay 0 in a serving process).
7. **Assert the single-owner invariant** around the batched call the same way
   `mynah_asr_sched_assert_thread` does around `stream_feed`: one thread per
   model may run a batched step, because the batch scratch lives on the model.

---

## 2026-09-19 — S1-2 closed: the delta window

`t0` was `0.0` on every streaming delta, which left `t1` as the only usable
field and forced any consumer that wanted to place text in time to keep its own
running total — and to get it right, which it cannot, because it does not know
how the runtime chunked the audio.

Deltas now partition the stream: `t0` is the `t1` of the previous delta of the
same stream (`emitted_t1`, reset with the stream). The WebSocket `delta` frame
carries both beside `audio_s`, which stays because `eou` and `error` frames have
no window and still need to say where they are.

Stated where it is easy to get wrong (`docs/api.md`, `docs/server.md`): this is
the audio WINDOW the text was produced from, not an alignment. A token may have
been spoken slightly before the window it is reported in — the encoder works in
chunks and the decoder trails it. Word-level times remain the offline path's
`mynah_asr_transcribe_ts`.

An `eou` result is now a point, `t0 == t1 == eou_sec`, instead of a window
starting at zero.

**The chunk-arrival passthrough asked for in the item was deliberately NOT
added.** The server stages the chunk, so the server owns its arrival:
`emit_ctx.arrival` comes from the slot's arrival record and the result callback
runs synchronously inside that step (`server/sched.c`). A library-level copy
would be a number the library has no way to check, duplicated in the one place
that already has the right one.
