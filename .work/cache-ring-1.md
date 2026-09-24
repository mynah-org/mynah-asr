# CACHE-RING-1 — does removing the K/V cache memmove matter?

Status: IN PROGRESS (contract registered, implementation and local correctness
done; Axion measurement pending)

Task: S13-1d, S10-5

Everything above the line marked **RESULTS** was written before the first Axion
measurement. Thresholds are not moved after seeing numbers.

## Question

S13-1c found that NVIDIA's NeMo-Speech.cpp advances a ring head where we
`memmove` the saturated streaming K/V cache. Arithmetic in the S13 ledger puts
our shift at ~11 MB per chunk per stream, ~2.75 GB/s at the qualified C=80.
S12-7c records that CPU occupancy plateaus at 21.6-23.5 cores of 30 at
saturation, mechanism UNKNOWN.

1. Can the physical shift be removed without changing model semantics?
2. Is the output equivalent (and at what strength)?
3. Does the cache-copy traffic disappear (measured, not computed)?
4. Does it move CPU utilisation?
5. Does it move throughput / safe concurrency / backlog?
6. Does it change first-token latency? (Expected: no. This is NOT a
   first-word experiment; R-14/R-15/S13-5 closed that separately.)
7. Is the added complexity justified by the gain?

## HYPOTHESIS

Removing the physical K/V shift reduces memory traffic and may raise CPU
utilisation and/or saturated streaming throughput, because a bandwidth-bound
copy that scales with streams leaves cores stalled.

## NULL / REJECTION CONDITION

If a layout that is output-equivalent and provably removes the copy does not
materially move throughput, cores used, lag or backlog (thresholds below), the
memmove is rejected as an important cause of the S12-7c plateau and recorded as
such.

## Known facts before the experiment (and a prediction they force)

- **FACT, already measured — the direct cost is small.** The server profile
  times the shift apart from the attention (`kv_cache` component,
  `src/encoder.c` batched step). Axion, 2026-09-23, 6x5, C=64, current
  relpos-table build (`.work/evidence/serving-v2/20260923T113713Z/server-ladder-C64.log`,
  all six workers): **kv_cache = 0.21-0.24 ms per row, share 0.012-0.014 of
  the encoder step** (18.2 ms/row). F27 (earlier build) had 0.13-0.16 ms, 0.6 %.
- **FACT — the shift is half the copying, not all of it.**
  `stream_attention_core` already GATHERED `valid cache ++ fresh rows` into the
  contiguous scratch `sa_keys` on every layer (56+4 rows x 1024 f32 x 2
  tensors x 24 layers = 11.8 MB/step), then `update_kv_cache` memmoved 52 rows
  (10.2 MB/step). The gather sits inside the `attn` component (0.078-0.080).
  The audit of 2026-09-20 had already flagged both (S10-5: "~27 MB moved per
  row-step").
- **Arithmetic on those facts.** At C=80 and 320 ms, 250 row-steps/s x
  0.23 ms = **~0.06 core-equivalents** for the shift. Its DIRECT removal cannot
  recover the ~7 unused cores. The hypothesis survives only through an
  INDIRECT route: bandwidth/cache contention slowing the GEMMs of other
  streams on the same memory system, which the component timer cannot see.
- **PREDICTION (registered): NEUTRAL** for the memmove alone. `slide` also
  removes the gather and should be a few percent of the encoder step at most.
  The experiment is still worth running because only the system-level A/B can
  see the indirect route, and because rejection needs a measurement.

## Cache lifecycle, traced before the change

Every access site, found by grep over `src/ server/ cli/ tests/ tools/`:

| event | where | what |
|---|---|---|
| allocation | `mynah_asr_enc_stream_init` | `k_cache`, `v_cache` calloc `[n_layers, left, d]` each |
| dimensions | `mynah.json` streaming preset | Nemotron 24 x 56 x 1024; EOU 120M 17 x 70 x 512 |
| valid count | `es->cache_valid` | 0..left, `+= Q` saturating, updated ONCE per step after all layers |
| initial fill | first steps | `valid < left`, K = valid + Q grows, pos-emb / rel-pos rows follow K |
| saturated update | `update_kv_cache` | memmove `from_old` rows to row 0, memcpy last Q fresh rows |
| reader | `stream_attention_core` | memcpy `cache[0:valid] ++ fresh` into `sa_keys`, GEMMs over it |
| writers | single `mynah_asr_enc_stream_step`, batched `mynah_asr_enc_stream_step_batch` | both call `update_kv_cache` per layer, K then V |
| reset | `mynah_asr_enc_stream_reset` | memset both, `cache_valid = 0` (per utterance in the server) |
| destroy | `mynah_asr_enc_stream_free` | free |
| clone/copy | none in src/ or server/ | — |
| tests | `tests/test_stream_batch` | memcmp of the PHYSICAL cache, single vs batched |
| batch/worker | one stream is stepped by one scheduler thread; batching stacks rows, the cache loop is per stream | no sharing |

Assumptions that logical row 0 is physical row 0: the gather (`memcpy` from
`k_cache` base), the shift itself, and the test's physical memcmp. The rel-pos
`base = K-1-valid-t` depends on `valid` and on the WINDOW order, not on where
the cache lives, so it is layout-independent as long as the window is logical.
`need()`, the pos-emb K, the R-13 probe and relpos sharing read `cache_valid`
only.

**Old invariant**: physical `cache[0:valid]` == logical history, oldest ->
newest.

**New invariant**: logical row i of layer l, tensor w is `kv_row(kv, l, w, i)`;
the window handed to the attention is a CONTIGUOUS `[valid + Q, d]` block equal
to `logical[0:valid] ++ fresh` in every layout.

## Design (`src/kvcache.[ch]`)

Three layouts of the same logical cache, one runtime flag
`MYNAH_ASR_KV_LAYOUT=shift|ring|slide`, default `shift` (the old path, byte
for byte), so the A/B is one binary and the experiment can be rejected by
deleting a module.

| layout | window | commit | copied/step (Nemotron, saturated) | memory |
|---|---|---|---|---|
| shift | gather 60 rows | memmove 52 + write 4 | 22.8 MB | 11.0 MB/stream |
| ring  | gather 2 spans + fresh (same bytes) | write 4 at the tail, modulo once per chunk | 12.6 MB | 11.0 MB/stream |
| slide | none: fresh written after the valid rows, window = arena | move `head` | 1.6 MB (fresh + compaction every 14 chunks) | 22.0 MB/stream |

Reader alternatives considered (brief section 4):

- **A, two-span attention without a gather.** Scores are per column, so the QK
  GEMM splits exactly; but `scores x V` split into two GEMMs changes the
  accumulation order over K — NOT bit-exact, and it is a kernel change. Rejected
  while an exact alternative exists.
- **B, gather into contiguous scratch.** Already what the code did. `ring`
  keeps it: it removes the memmove and nothing else — the clean test of the
  hypothesis as stated.
- **C, ring-aware traversal.** Modulo inside the GEMM operand. Rejected, same
  accumulation problem as A plus per-element index arithmetic.
- **D, contiguous arena with amortised compaction (`slide`).** Double the cache
  memory, and no copy at all in the steady state. It is S10-5 as the audit
  specified it. Kept as a second treatment: if even removing ~21 of 22.8 MB per
  step moves nothing, the rejection is strong.

Softmax: untouched. The window is ONE contiguous block in every layout, so the
attention is evaluated over one logical vector exactly as before; no span is
ever softmaxed alone.

Geometry cases: `valid < left`, `valid == left`, Q = 1, production Q, Q > left
(ring keeps the last `left` rows of the chunk), wraparound (at most two spans
for both the window and the tail write), reset (head = valid = 0), `left == 0`
(ring degenerates to shift). Slide slack = max(left, qmax), so a chunk always
fits after a compaction and compactions happen at most every left/Q chunks.

## Correctness gates (local)

- `tests/test_kv_layout` part 1 (no model, runs in `make test` always):
  3 layouts against an independent reference ("the last min(n, left) appended
  rows"), rows whose bits encode (absolute index, layer, tensor), window and
  logical contents checked after every step, 10 geometries x 5000 steps,
  random Q from a fixed seed, random resets. **Mutation-checked**: an
  off-by-one in the ring tail, a wrong slide head and a wrong ring span each
  fail on the first bad row.
- `tests/test_kv_layout` part 2 (model): ~62 s of concatenated clips through
  one stream per layout in lockstep; every encoder output of every chunk and
  the full logical K/V after every chunk memcmp'd; then the three layouts in
  ONE mixed batched step against the single outputs.
- `tests/test_stream_batch` compares LOGICAL caches now (was physical), and
  runs green with every layout.

## Axion protocol (registered)

Methodology reused from the frozen 6x5 ladder (manifest of
`20260923T113713Z`): `tools/bench/v2_qualify.sh --phase ladder`, `-W 6 -T 5
-C 96`, server cpus 0-29, generator 30-31, `--corpus
samples/stress-en/manifest.json --corpus-sample 500 --corpus-seed 42`, int8,
lookahead 3, `--http-threads 32`, fresh fleet per rung, reference reused with
`--reference-file` from the latest C=80 soak so parity is checked under load.
Only `MYNAH_ASR_KV_LAYOUT` differs between arms, proven per run by the
`[EFFECTIVE-CONFIG]` banner and by the dump's `kv_copy layout=` line.

1. **Pre-flight**: SHA, dirty=0, disk, RAM, no stale processes/tmux, clean build.
2. **Nemotron correctness on the box** (the pack is not on the dev machine):
   `tests/test_kv_layout <nemotron>` and `tests/test_stream_batch` under all
   three layouts. Any failure = INVALID, stop.
3. **Phase A — mechanism**: `tests/test_kv_layout <nemotron> --bench 80 5000`,
   interleaved x2. Question: did the ~10 MB/step physical movement disappear?
4. **Phase B — C=1 control** is covered by the reference pass semantics plus
   phase 2's bit-exactness; a single-stream CLI RTF per layout on one long
   clip is recorded as the C1 point. A large C1 gain would be SUSPICIOUS.
5. **Phase C — saturation A/B**, 120 s rungs, interleaved
   `shift ring slide shift ring slide`, at **C=96** (last clean rung, equal
   offered load: the cores-used and lag question) and **C=112** (overload:
   the capacity question, audio/wall).

## Registered thresholds

Noise is taken from the two replicates of each arm, not assumed. For a metric
m, `spread(m)` = the larger |rep1 - rep2| of the arms compared.

- **Material throughput effect**: C=112 audio/wall differs by >= 3 % AND by
  more than 2 x spread.
- **Material cores effect**: cores used at C=96 differ by >= 1.0 core AND by
  more than 2 x spread.
- **Material latency effect**: C=96 emission-lag p95 or backlog max differ by
  >= 20 % AND by more than 2 x spread.
- **Safe-concurrency effect**: a rung that is bad for shift in both reps and
  clean for the treatment in both reps.

Decision, per treatment arm against shift:

- **INVALID**: any parity/bit-exactness failure. Numbers discarded.
- **WIN**: >= 5 % throughput or a safe-concurrency step, parity intact.
- **SMALL WIN**: material by the rules above but below the WIN bar.
- **NEUTRAL**: nothing material. The memmove (ring) / the cache copying as a
  whole (slide) is rejected as an important cause of the plateau.
- **LOSS**: a material regression on any of the above.

---

## RESULTS

(pending)
