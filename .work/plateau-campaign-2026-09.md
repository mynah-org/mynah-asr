# Plateau campaign, Axion 2026-09-24 — why ~8 of 30 cores sit idle at saturation

Status: DONE 2026-09-24 — plateau explained and removed; research candidate QUALIFIED at C=128 and C=144 (2 x 30 min each); default OFF pending a product decision (see DECISION RECORD at the end)

Task: S12-7c, S10-4, S10-3

One note for the short falsification experiments aimed at S12-7c, run on the
same box, topology and harness as CACHE-RING-1 (`.work/cache-ring-1.md`):
6 workers x 5 threads on cpus 0-29, generator on 30-31, Nemotron int8,
lookahead 3, `--batch-window-ms 0`, stress-en 500-clip sample, frozen C=80
reference for parity, 120 s rungs, fresh fleet per rung, shift K/V layout.
Raw evidence untracked under `.work/evidence/`.

Already rejected before this note: the K/V memmove (CACHE-RING-1) and early
batch release (S13-1e, the wait does not exist).

## Where the cores go — existing evidence, no new run

Mined from the 2026-09-23 6x5 ladder (C=64..120, 180 s rungs) and the two
C=80 soaks; cores from procsample cputimes, the rest from each worker's last
`[DUMP]`, mean of six workers.

| C | cores (of 30) | model_duty | park | ready B | dispatches/step | worker_spin % |
|---|---|---|---|---|---|---|
| 80 soak | 21.6-22.0 | 0.874-0.880 | 0.12 | 1.47 | 259 | 79 |
| 96 | 23.5 | 0.973 | 0.025 | 2.24 | 266 | 69.5 |
| 104 | 22.9 | 0.991 | 0.008 | 4.37 | 286 | 58.7 |
| 112 | 21.5 | 0.996 | 0.003 | 11.0 | 349 | 47.4 |
| 120 | 20.8 | 0.996 | 0.003 | 16.7 | 378 | 45.0 |

**FACT.** At saturation the scheduler thread is inside the model 97-99.6 % of
wall and parks < 3 %: the missing 1.1-1.5 cores per worker are INSIDE model
execution. `caller_spin` 93-99 % says the caller almost never waits for the
workers (the pooled GEMMs are balanced); `worker_spin` falling 85 -> 45 % as C
rises says the workers increasingly wait for the CALLER, across serial gaps
longer than the 50 us spin.

**DERIVED, not measured directly.** Treating model time as either caller-only
(1 core) or pooled (5 cores), `1 + 4(1-s)` = cores per worker gives a serial
fraction s ~ 0.24 of wall at C=96 and 0.38-0.47 at C=120. It grows with B
because the per-stream loops run serially and scale with B while the GEMMs
amortise.

**Code map** (`src/encoder.c`, batched step): on the pool are only the ten
`qmat_mul_rows` GEMMs per layer (one dispatch and one barrier each, 240 per
step) and a few sgemms in subsampling/post. Serial on the scheduler thread:
five layer norms per layer, SiLU per stream, residual adds, activation
quantisation (the 2^21 threshold is never reached at R*k <= 262k), the
per-stream attention core (18 M MAC + 46 k expf per stream per step; its
small GEMMs never reach the pool), K/V commit, per-stream conv mid (GLU,
depthwise, LN, SiLU), per-stream subsampling glue, tail, and the decoder
loop. Fully serial components alone (attn + kv_cache + tail) are ~12 % of
encoder time at C=120; the LN/SiLU/residual share is hidden inside ffn/conv.

## S10-4 — pool spin 50 vs 500 us

- **HYPOTHESIS.** Pool threads park between dispatches (54 % of worker waits
  at C=112) and the futex wake on the next dispatch costs width; a longer spin
  removes it and moves throughput or latency.
- **TREATMENT.** `MYNAH_ASR_POOL_SPIN_US=500` against the default 50, C=112,
  one pair, commit `161ae9f`.

| spin | worker_spin % | caller_spin % | dispatches/s/worker | audio/wall | cores | lag p95 | backlog max | fin p95 | lost |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 48.4 | 95.2 | 2430 | 81.45 | 21.50 | 581 | 0.824 | 1089 | 0 |
| 500 | **95.1** | 99.9 | 2655 | 81.36 | **26.99** | 513 | 0.724 | 981 | 0 |

- **RESULT.** The treatment took (48 -> 95 % spin hits). Throughput did not
  move (-0.1 %); the 5.5 extra cores are spinning, not work. Lag p95 -12 % is
  inside the range the shift arm itself produced at C=112 across three runs
  today (537 / 559 / 581). Parity intact.
- **DECISION.** REJECTED: park/wake is not what limits the plateau. Not
  repeated: nothing approached a registered threshold. The spin arm filling
  exactly the missing cores is itself evidence for the serial-gap reading:
  those cores are pool threads waiting for the caller.
- **NEXT.** S10-3.

## S10-3 — per-stream stages over streams on the pool

- **HYPOTHESIS.** The serial per-stream loops are a large part of the gaps;
  running them over streams on the pool raises cores used and throughput at
  overload, where B is wide.
- **TREATMENT.** `MYNAH_ASR_STREAM_PAR=1` (commit `bdeefc9`): the attention
  core + K/V commit loop and the conv-mid loop become one `parallel_for` over
  streams each. Bit-exact by construction; `test_stream_batch` (int8/f32,
  B=2..8, Nemotron on the box and EOU 120M locally) and `test_kv_layout`
  (mixed batch) IDENTICAL with the flag on.
- **Registered thresholds** (same as CACHE-RING-1): throughput >= 3 % and
  > 2 x spread; latency >= 20 % and > 2 x spread; cores >= 1.0. Cores alone is
  never a win. Upper bound of the direct gain: attn + kv + conv-mid serial time
  at B ~ 10 shared over 5 threads.

### Level 1 — RESULT (commit `bdeefc9`, C=112 x2 in ABBA order, C=80 x1)

| STREAM_PAR | C | rep | audio/wall | cores | a/w per core | lag p50/p95/p99 | backlog max | fin p95 | ready B | disp/s/w | worker_spin % | lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 112 | 1 | 82.19 | 21.56 | 3.81 | 296 / 546 / 744 | 0.744 | 1051 | 10.7 | 2483 | 48.6 | 0 |
| 0 | 112 | 2 | 81.51 | 21.56 | 3.78 | 298 / 552 / 739 | 0.824 | 1065 | 9.0 | 2477 | 48.5 | 0 |
| 1 | 112 | 1 | 83.81 | 24.79 | 3.38 | 93 / 247 / 350 | 0.684 | 506 | 3.8 | 5761 | 73.9 | 0 |
| 1 | 112 | 2 | 82.23 | 24.77 | 3.32 | 93 / 242 / 341 | 0.504 | 499 | 3.7 | 5703 | 73.9 | 0 |
| 0 | 80 | 1 | 56.16 | 17.78 | 3.16 | 43 / 109 / 163 | 0.384 | 215 | 1.58 | 6435 | 80.0 | 0 |
| 1 | 80 | 1 | 56.10 | 17.68 | 3.17 | 40 / 99 / 156 | 0.284 | 209 | 1.40 | 7052 | 85.3 | 0 |

Parity `identity_fail 0, reference_fail 0` on every rung; the banner names
`MYNAH_ASR_STREAM_PAR` on each. Component `attn` 1.39 -> 0.63 ms/row of wall
at C=112 (the region now runs on five threads).

Against the registered thresholds, C=112, means of two reps:

- **lag p95 546/552 -> 247/242: -55 %**, 2 x spread = 12 ms. MATERIAL.
- lag p99 -53 %, fin p95 -52 %, lag p50 297 -> 93.
- backlog max 0.784 -> 0.594: -24 %, but 2 x spread = 0.36 s. Not material.
- audio/wall +1.4 %: not material (C=112 is still largely paced: a/w / C =
  0.73, the same ratio as C=80, so throughput has little room to move).
- cores +3.2: real, and not a win by itself.
- **Serving gates** (v2_qualify's own bounds; TTFP fails on every rung at
  every C including C=80, the known corpus-onset artefact with no registered
  bound): shift FAILS emission lag and finalization at C=112 in both reps;
  level 1 PASSES all of them in rep 2 and misses finalization (506 against
  500) and backlog (0.684 against 0.640) by a hair in rep 1. So the
  registered safe-concurrency step (bad in both reps -> clean in both reps) is
  NOT met — it is 3 of 4.
- **Throughput per core FALLS 12 %** at C=112 (3.80 -> 3.35 audio-s per
  core-s): the fleet buys its latency with ~3 more cores, most of it pool
  threads now busy (and spinning between the doubled dispatches: 2480 -> 5730
  per second per worker) inside a region that used to be serial. At C=80
  it is flat (3.16 / 3.17) and nothing material moves.

**DECISION.** Level 1 is a **SMALL WIN** by the registered rules: a large,
reproducible overload-latency gain (-55 % p95, a near step in safe
concurrency), at a cost in efficiency, with no throughput gain. It is the
first treatment in this campaign that moves the core plateau materially
(21.6 -> 24.8 cores doing work that was serial), which confirms the serial
per-stream loops as a real part of the S12-7c mechanism. Kept, default off.

**NEXT.** Level 2 measured as 1 vs 2, not 0 vs 2.

### Level 2 — RESULT, measured as 1 vs 2 (commit `249e6a5`, C=112, ABBA x2)

Level 2 = level 1 + SiLU per stream + each residual add fused into the
following layer norm, row-parallel. Bit-exact on Nemotron on the box
(`test_stream_batch`, `test_kv_layout`), and a planted 0.5 -> 0.5001 in the
fused add fails the gate, so the new path is exercised.

| level | rep | audio/wall | cores | a/w per core | lag p50/p95/p99 | backlog max | fin p95 | ready B | disp/s/w | worker_spin % |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 83.90 | 24.79 | 3.38 | 94 / 246 / 353 | 0.584 | 513 | 3.7 | 5747 | 74.0 |
| 2 | 1 | 82.68 | 25.59 | 3.23 | 67 / 185 / 257 | 0.484 | 377 | 2.6 | 9671 | 91.5 |
| 2 | 2 | 82.43 | 25.63 | 3.22 | 67 / 187 / 274 | 0.544 | 368 | 2.6 | 9743 | 91.5 |
| 1 | 2 | 83.69 | 24.77 | 3.38 | 97 / 243 / 340 | 0.564 | 488 | 3.8 | 5688 | 73.8 |

- Level 2 over level 1: **lag p95 -24 %** (2 x spread = 6 ms) MATERIAL; fin
  p95 -26 %, lag p99 -23 %. Backlog -13 % (inside 2 x spread) and audio/wall
  -1.5 % not material; cores +0.8; throughput per core -5 %.
- **Cumulative against shift** (shift at C=112: five runs today, lag p95
  537-581, all failing the lag and finalization gates): lag p95 ~549 -> 186
  (-66 %), fin p95 ~1058 -> 373 (-65 %). Level 2 passes every serving gate at
  C=112 in both reps (TTFP excepted, the corpus-onset artefact that fails at
  every C). That is the registered safe-concurrency step. Caveat: the shift
  arm was not interleaved with the level-2 reps; it was interleaved with level
  1, and it has been stable across five runs on this box today.
- **Cost**: throughput per core 3.80 (shift) -> 3.38 (1) -> 3.22 (2). The
  latency is bought with pool threads doing work that was serial, plus four
  times the dispatches (2480 -> 9700 per second per worker), each a barrier.
  The fleet is not doing MORE audio per second at C=112, because C=112 is
  still mostly paced; what changes is that it stops falling behind.

**DECISION.** Level 2 kept as an incremental SMALL WIN on top of level 1, and
together they are the first treatment that WINS by the registered rules (a
safe-concurrency step at C=112). Default stays off: promotion is a product
decision and needs its own qualification.

**NEXT.** Where does the knee move? C=128, shift against level 2.

### Knee probe — C=128, shift against level 2 (one pair)

| arm | audio/wall | cores | a/w per core | lag p50/p95/p99 | backlog max | fin p95 | utterances | ready B |
|---|---|---|---|---|---|---|---|---|
| shift | 77.49 | 20.97 | 3.70 | 1116 / 2781 / 3467 | 5.944 | 4219 | 862 | 12.8 |
| level 2 | **83.41** | 20.91 | **3.99** | 196 / 422 / 566 | 0.764 | 855 | 996 | 8.6 |

**RESULT.** Both arms are past their knee (level 2 fails finalization, 855
against 500, and lag p95 422 against 320). Shift COLLAPSES — a 5.9 s backlog,
lag p95 2.8 s, 862 utterances completed; level 2 degrades gracefully with
**+7.6 % audio/wall, +16 % completed utterances and +8 % throughput per core at
the same cores**. One pair only: the effect is an order of magnitude beyond any
spread measured today, so it was not repeated.

**Reading.** At C=112 the treatment bought latency with cores (per-core
efficiency down) because the fleet was still keeping up; at C=128 the serial
path is the binding constraint, and removing part of it is pure capacity. The
knee moves from between C=96 and C=104 (shift, S12-7) to between C=112 and
C=128 (level 2). Parity intact on every rung.

## Conclusion so far

- **S12-7c, mechanism.** The plateau is mostly the scheduler thread running
  per-stream and elementwise stages alone while the pool waits: the evidence
  mining derives a serial fraction of ~0.24 of wall at C=96 and 0.38-0.47 at
  C=120; spinning the pool (S10-4) fills the idle cores without doing work;
  moving the per-stream loops onto the pool (S10-3) moves cores, latency and,
  past the knee, throughput. The K/V memmove (CACHE-RING-1) and the batch
  window (S13-1e) are ruled out.
- **Product decision pending, not taken here:** promoting
  `MYNAH_ASR_STREAM_PAR=2` to default changes production behaviour and needs
  its own qualification (C=112 soak pair under v2_qualify/v2_promote).
- **Remaining serial stages, planned but not built** (read-only plan): the
  tail stacked across streams (~0.3 ms/row, low risk), the mel/VAD front end
  over streams (low risk), subsampling over streams (its GEMMs are already
  pooled: measure first), decode + emit split into a parallel decode and an
  ordered publish (largest, medium risk: trace statics, head bandwidth).

### Level 3 — RESULT, measured as 2 vs 3 (commit `7accee0`, C=112, ABBA x2)

| level | rep | audio/wall | utt | cores | a/w per core | lag p50/p95/p99 | fin p95 | backlog max | disp/s/w |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 82.30 | 899 | 25.61 | 3.21 | 67 / 184 / 270 | 367 | 0.564 | 9647 |
| 3 | 1 | 82.34 | 900 | 25.01 | 3.29 | 71 / 183 / 264 | 357 | 0.564 | 9111 |
| 3 | 2 | 82.86 | 900 | 24.99 | 3.32 | 72 / 184 / 258 | 349 | 0.464 | 8996 |
| 2 | 2 | 82.74 | 899 | 25.57 | 3.24 | 67 / 187 / 268 | 372 | 0.544 | 9681 |

Every serving gate passes in all four; parity intact. Level 3 over level 2:
throughput flat, lag p95 -1 %, fin p95 -4.5 %, cores -0.6 at equal throughput
(+2.4 % per core). Nothing clears a registered threshold: **NEUTRAL**. Kept in
the tree (bit-exact, default off) but NOT carried into the level-4 measurement:
`MYNAH_ASR_STREAM_PAR_DECODE=0` switches it off under level 4.

### Level 4 — mechanism, measured before any benchmark

Local probe, Nemotron geometry (24 x 1024, Q=4), B=8, 30 steps; counts are
exact, times are a dev signal only.

| level | act-quant calls on caller / step | rows on caller | caller ms / step | rows quantised in regions | GEMMs on pre-quantised rows | pool dispatches / step |
|---|---|---|---|---|---|---|
| 0 | 240 | 7680 | 3.5 | 0 | 0 | 240 |
| 2 | 240 | 7680 | 4.7 | 0 | 0 | 456 |
| 3 | 240 | 7680 | 4.7 | 0 | 0 | 456 |
| 4 | **0** | **0** | **0** | 6144 | 240 | **456** |

The serial quantisation passes disappear entirely; q/k/v share one pass (7680
-> 6144 rows); no dispatch is added (level 2 had added 9 per layer). The
server dump carries the same counters as an `actq` line.

### Level 4 — RESULT at C=112, measured as 2 vs 4 without level 3 (commit `758a10c`, ABBA x2)

| arm | rep | audio/wall | cores | a/w per core | lag p50/p95/p99 | fin p95 | backlog max | disp/s/w | caller act-quant ms / worker / 120 s |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 82.31 | 25.60 | 3.22 | 67 / 184 / 271 | 390 | 0.644 | 9780 | ~3670 |
| 4 | 1 | 82.84 | 25.38 | 3.26 | 62 / 175 / 247 | 340 | 0.584 | 10114 | ~430 |
| 4 | 2 | 83.03 | 25.37 | 3.27 | 62 / 175 / 247 | 346 | 0.384 | 10069 | ~425 |
| 2 | 2 | 82.66 | 25.52 | 3.24 | 68 / 189 / 276 | 391 | 0.564 | 9622 | ~3680 |

- **Mechanism on the box**: caller-side activation quantisation falls 88 %
  (the rest is B=1 steps, finalization and the decoder, which level 4 does not
  touch); ~3.2 s of scheduler time per worker per 120 s removed (~2.7 %).
- System level: audio/wall +0.5 %, lag p95 -6 % (2 x spread = 10 ms, so
  outside noise but under the 20 % threshold), fin p95 -12 %, lag p99 -10 %,
  cores -0.2. **NOT MATERIAL at C=112** by the registered rules.
- C=112 no longer discriminates: both arms pass every gate with ready B ~2.6,
  i.e. neither is saturated. C=128 (level 2's knee) is the discriminating rung.

### Level 4 — RESULT at C=128 (level 2's knee), 2 vs 4 without level 3, ABBA x2

| arm | rep | audio/wall | utt | cores | a/w per core | lag p50/p95/p99 | fin p95 | backlog max | ready B |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 84.91 | 1003 | 21.03 | 4.04 | 147 / 330 / 474 | 655 | 0.684 | 6.5 |
| 2 | 1 | 84.03 | 994 | 20.90 | 4.02 | 198 / 413 / 555 | 849 | 0.684 | 8.4 |
| 2 | 2 | 83.32 | 997 | 20.96 | 3.98 | 197 / 413 / 566 | 858 | 0.784 | 8.0 |
| 4 | 2 | 84.12 | 1004 | 20.98 | 4.01 | 148 / 337 / 468 | 689 | 0.704 | 7.7 |

- lag p95 **-19.3 %** (2 x spread = 14 ms, so outside noise) against a 20 %
  threshold; fin p95 -21 %; lag p99 -16 %; audio/wall +1.0 % (inside 2 x
  spread); backlog -5 %; cores and per-core efficiency unchanged. Both arms
  still fail the lag and finalization gates at C=128: no safe-concurrency step.
- **DECISION: NOT MATERIAL by the registered rules** — the threshold is not
  moved after the fact. Recorded as directionally consistent at both rungs
  (-6 % at C=112, -19 % at C=128) at no efficiency cost; kept in the tree,
  default off.
- **Observation for the next item**: at C=128 level 2 and level 4 both sit at
  ~21 cores of 30 again (25.6 at C=112), with ready B 7-8. The plateau
  reappears one rung later, consistent with per-step serial work that still
  grows with B (decode, tail, finalization at B=1).
- **Consistency check queued**: level 3 was judged neutral only at C=112, the
  rung shown here not to discriminate. It is re-measured at C=128 on top of
  level 4 (`STREAM_PAR_DECODE` 1 vs 0).

### Level 3 re-measured at C=128, on top of level 4 (`STREAM_PAR_DECODE` 1 vs 0, ABBA x2)

| arm | rep | audio/wall | utt | cores | a/w per core | lag p50/p95/p99 | fin p95 | backlog max | ready B |
|---|---|---|---|---|---|---|---|---|---|
| 4 + 3 | 1 | 86.30 | 1014 | 21.16 | 4.08 | 111 / 261 / 366 | 515 | 0.584 | 5.1 |
| 4 | 1 | 84.99 | 1005 | 21.03 | 4.04 | 144 / 340 / 491 | 719 | 0.644 | 6.9 |
| 4 | 2 | 85.01 | 1005 | 21.10 | 4.03 | 142 / 337 / 465 | 703 | 0.584 | 7.0 |
| 4 + 3 | 2 | 86.04 | 1013 | 21.14 | 4.07 | 111 / 261 / 370 | 516 | 0.584 | 5.0 |

- lag p95 **-22.9 %** (2 x spread = 6 ms): **MATERIAL**; fin p95 -27.5 %, lag
  p99 -23 %, audio/wall +1.4 % (outside 2 x spread, under 3 %), cores flat.
- **The C=112 verdict on level 3 is superseded**: that rung did not saturate
  level 2, so it could not see a per-stream decode that only binds when B is
  wide (ready B 7 at C=128 against 2.7 at C=112). Recorded as a method lesson:
  a treatment of a B-scaling serial stage is judged at a rung where B is wide.
- With levels 2+3+4 the C=128 fleet passes the emission-lag gate (261 < 320)
  and backlog, and misses finalization by 15 ms (515 / 516 against 500).

### The ladder so far, C=128 (lag p95 / fin p95 / audio/wall)

| config | lag p95 | fin p95 | audio/wall | source |
|---|---|---|---|---|
| shift (off) | 2781 | 4219 | 77.49 | one pair |
| level 2 | 413 | 853 | 83.68 | ABBA x2 |
| level 2 + 4 | 334 | 672 | 84.52 | ABBA x2 |
| level 2 + 3 + 4 | **261** | **515** | **86.17** | ABBA x2 |

**Where the frontier is now.** Finalization is the gate that still fails at
C=128, and it is the one path none of the levels touches: a finishing stream
runs its last chunks through the SINGLE-stream path, one slot at a time on the
scheduler thread (~61 ms per call, 8 % of wall at C=112, `phase finalize`),
while every other stream waits. That is the next evidence-backed candidate.
Cores used are back at ~21 of 30 at C=128.

## FIN-STACK — finalization through the stacked path (registered before the run)

- **FACT** (C=128, levels 2+3+4, worker 0 dumps): `finalize model_mean_ms`
  58.9-59.9 over 183-205 calls, 9 % of worker wall, all of it `model_solo`
  (the rest of the worker waits). A finalizing stream's last pieces and its
  padded tail go through `stream_flush_chunk` -> the single-stream step, whose
  int8 products take the serial small-T path (`QC_DOT`) on the scheduler
  thread: one core busy, four pool threads idle, for ~59 ms.
- **HYPOTHESIS.** Routing those chunks through the stacked B=1 path (measured
  4.5x faster per step than the GEMV chain in the S1 work) shortens every
  finalization and every solo block, and moves fin p95 and lag at C=128.
- **TREATMENT.** `MYNAH_ASR_FIN_STACK=1` (commit `3afcb07`) on top of
  `MYNAH_ASR_STREAM_PAR=4` (levels 2+3+4), C=128, ABBA x2. Gate before the
  run: CLI deltas byte-identical flag off/on (12 local runs + 6 corpus clips
  on the box), and a planted `is_last=0` in the batched step changes deltas.
- **Mechanism proof expected in the dump**: `finalize model_mean_ms` well
  below 59.
- **Thresholds** as registered for the campaign (lag p95 or backlog >= 20 % and
  > 2 x spread; throughput >= 3 %; the fin p95 gate at 500 ms is reported).

### FIN-STACK — RESULT (commit `3afcb07`, C=128, on STREAM_PAR=4, ABBA x2)

Box gate first: CLI deltas byte-identical with the flag off and on for 6
corpus clips on the qualified Nemotron pack.

| FIN_STACK | rep | finalize model ms / call | audio/wall | utt | cores | a/w per core | lag p50/p95/p99 | fin p95 | backlog max | ready B |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 17.0 | 88.53 | 1036 | 22.54 | 3.93 | 76 / 126 / 161 | 220 | 0.184 | 3.4 |
| 0 | 1 | 61.0 | 85.93 | 1012 | 21.04 | 4.08 | 111 / 264 / 369 | 541 | 0.584 | 5.2 |
| 0 | 2 | 59.6 | 85.99 | 1010 | 21.04 | 4.09 | 112 / 269 / 381 | 537 | 0.784 | 5.1 |
| 1 | 2 | 17.5 | 88.50 | 1034 | 22.49 | 3.94 | 76 / 128 / 164 | 220 | 0.244 | 3.3 |

- **Mechanism**: a finalization's model time 60 -> 17 ms (-71 %).
- lag p95 **-52 %** (2 x spread = 10 ms), fin p95 -59 %, backlog max -69 %,
  lag p99 -57 %: MATERIAL. Audio/wall +3.0 % (2.97 %, on the bar; +2.4 %
  completed utterances); cores +1.5; throughput per core -3.8 %.
- **Serving gates**: without the flag the fleet fails finalization at C=128 in
  both reps; with it, it passes EVERY gate in both. That is the registered
  safe-concurrency step: **WIN**.
- Parity `identity_fail 0, reference_fail 0` on all four; no stream lost.

**Reading.** The single-stream step was the last big serial block: every
finalization held the scheduler thread for ~60 ms on one core while the rest
of the worker's streams waited, and at C=128 a worker finalizes ~1.7 streams
per second. Both the finalizing stream and everyone queued behind it paid for
it.

### Knee re-measured with the full stack (STREAM_PAR=4 + FIN_STACK=1, one rep per rung)

| C | audio/wall | utt | cores (of 30) | lag p95 | fin p95 | backlog max | ready B | gates |
|---|---|---|---|---|---|---|---|---|
| 144 | 110.12 | 1441 | **28.31** | 241 | 429 | 0.384 | 6.8 | all PASS |
| 160 | 109.74 | 1196 | 28.49 | 761 | 1338 | 1.084 | 15.0 | lag, fin, backlog FAIL |

Throughput saturates at ~110 audio-s per second (C=144 and C=160 agree), with
the fleet using 28.3-28.5 of its 30 cpus. One rep per rung: a screening point,
not a qualification.

## Campaign conclusion

| config | knee (last clean rung / first bad) | cores used near the knee | source |
|---|---|---|---|
| shipped (shift, all off) | C=96 / C=104 | 21.5-23.5 of 30 | S12-7, today's shift rungs |
| STREAM_PAR=2 | C=112 / C=128 | 25.6 at C=112 | ABBA x2 |
| STREAM_PAR=4 | (C=128 fails finalization) | 21 at C=128 | ABBA x2 |
| STREAM_PAR=4 + FIN_STACK | **C=144 / C=160** | **28.3** | ABBA x2 at C=128, one rep at 144/160 |

- **S12-7c is answered**: the ~8 idle cores were serial work on each worker's
  scheduler thread — per-stream encoder stages, per-stream front end and
  decode, activation quantisation, and above all finalization on the
  single-stream path — while the pool waited. Every treatment that moved it
  onto the pool moved the plateau; the ones that did not (K/V memmove, pool
  spin, batch window) did not.
- Per treatment, at the rung that discriminates it: level 2 WIN (C=112 step),
  level 3 MATERIAL at C=128 (neutral at C=112), level 4 -19 % lag p95 at C=128
  (just under the bar), FIN-STACK WIN (C=128 step). Parity intact on every
  rung of the campaign; bit-exact gates for each level, each mutation-checked.
- **Method lessons.** (1) A treatment of a stage that scales with B must be
  judged at a rung where B is wide; C=112 hid level 3 once level 2 had made it
  unsaturated. (2) Throughput per core is reported beside cores: level 2 bought
  C=112 latency with -15 % efficiency, FIN-STACK at C=128 with -4 %.
- **Not done, needs a decision**: promoting any of these flags to default
  changes production and needs its own qualification (v2_qualify soak pair at
  the new operating point, v2_promote). The research configuration is
  `MYNAH_ASR_STREAM_PAR=4 MYNAH_ASR_FIN_STACK=1`.

## QUALIFICATION of the research candidate (registered bounds, v2_qualify / v2_verdict)

Candidate frozen at commit `dbae3fe` (dirty 0), `MYNAH_ASR_STREAM_PAR=4
MYNAH_ASR_FIN_STACK=1`, every other flag unset; Nemotron int8 pack
2f5e1434..., lookahead 3, batch window 0, 6x5 on cpus 0-29, generator 30-31,
stress-en 500-clip sample (bank 04a7753aa1e80f9a, 498 clips), fresh server per
soak, soak seeds 43/44. Provenance per run in the evidence (untracked,
`.work/evidence/qual-20260924/`).

### C=128 — QUALIFIED (both soaks)

| bound | soak 1 | soak 2 | limit |
|---|---|---|---|
| utterances / established streams lost | 17007 / 0 | 17005 / 0 | 0 |
| emission lag p95 | 129 ms | 129 ms | 320 ms |
| finalization p95 | 221 ms | 221 ms | 500 ms |
| backlog max | 0.284 s | 0.284 s | 0.640 s |
| 503 refusals | 0 | 0 | — |
| worst steady 60 s window p95 / trend | 132 ms / +0.2 % | 132 ms / +0.5 % | 320 ms / +50 % |
| stalls >640 ms over published deltas | 0 / 467968 | 0 / 467879 | — |
| TTFP load penalty, paired p50/p95/p99 | +59 / +103 / +124 ms | +59 / +102 / +126 ms | GOOD <= 250 |
| worker RSS growth | 1.034x | 1.025x | 1.15x |
| transcript parity (every stream = unloaded reference) | PASS | PASS | — |
| audio/wall | 123.68x | 123.65x | — |

The raw stream_load screen "TTFP p95 2881 vs 1160" fails at every C including
the unloaded C=4 pass (2821 ms): it is the corpus onset, not a registered
bound, and v2_verdict's paired bound 13 is the gate (as for the frozen C=80).

### Quality regression gate — PASS, by identity

The candidate's unloaded transcripts (C=4 pass of this qualification) are
byte-identical to the FROZEN shipped-build reference (2026-09-23 C=80
qualification) on 498 of 498 clips. Scored with streaming_metrics (the scorer
reproduces the frozen S13-3c numbers exactly first):

| subset | n | WER mean | corpus WER | p50 / p95 | CER mean | WER-ff | empty / truncated |
|---|---|---|---|---|---|---|---|
| all | 498 | 0.10634 | 0.14618 | 0.0714 / 0.3333 | 0.06878 | 0.07736 | 0 / 1 |
| original | 349 | 0.10398 | 0.10856 | 0.0667 / 0.3333 | 0.06866 | 0.07435 | 0 / 0 |
| composed | 149 | 0.11186 | 0.18436 | 0.0789 / 0.2778 | 0.06906 | 0.08441 | 0 / 1 |

Paired per-clip delta against the frozen result: 0 worse, 0 better, 0.00000.
Under load the soaks' parity bound compares every stream to this same
reference, so the loaded transcripts are identical to the shipped build too.

### C=144 — QUALIFIED (both soaks), with thin finalization margin

Same frozen build and flags; the unloaded reference is the C=128 one (same
build, byte-identical file), which v2_qualify's `--reference-file` supports.

| bound | soak 1 | soak 2 | limit |
|---|---|---|---|
| utterances / established streams lost | 23371 / 0 | 23367 / 0 | 0 |
| emission lag p95 | 243 ms | 243 ms | 320 ms |
| finalization p95 | **428 ms** | **429 ms** | 500 ms |
| backlog max | 0.384 s | 0.464 s | 0.640 s |
| TTFP load penalty, paired p50/p95/p99 | +142 / +205 / +229 ms | +142 / +206 / +230 ms | GOOD <= 250 |
| deltas late >320 ms / >640 ms (of ~531k) | 7494 / 0 (worst 582) | 7729 / 0 (worst 581) | — |
| transcript parity | PASS | PASS | — |
| audio/wall | 135.16x | 135.63x | — |

### Both points, per soak

| C | soak | cores of 30 | audio/wall | a/w per core | lag p50/p95/p99 | fin p50/p95/p99 | backlog p95/max |
|---|---|---|---|---|---|---|---|
| 128 | 1 | 26.21 | 123.68 | 4.72 | 78 / 129 / 164 | 145 / 221 / 256 | 0.184 / 0.284 |
| 128 | 2 | 26.20 | 123.65 | 4.72 | 78 / 129 / 164 | 146 / 221 / 256 | 0.184 / 0.284 |
| 144 | 1 | 26.93 | 135.16 | 5.02 | 165 / 243 / 343 | 303 / 428 / 476 | 0.284 / 0.384 |
| 144 | 2 | 27.36 | 135.63 | 4.96 | 165 / 243 / 344 | 304 / 429 / 478 | 0.284 / 0.464 |

`v2_promote` dry-run (no `--apply`, nothing written) accepts both runs.

## DECISION RECORD (immutable, 2026-09-24)

Three different claims, kept apart:

**1. Research capacity (screening, short runs).**
- Shipped build: last clean C=96, first bad C=104 (S12-7), 21.5-23.5 of 30
  cores near the knee.
- Research stack `MYNAH_ASR_STREAM_PAR=4 MYNAH_ASR_FIN_STACK=1`: screening
  last good **C=144**, first bad **C=160**; 28.3-28.5 of 30 cores; throughput
  saturates at ~110 audio-s per second in the 2-minute screen.

**2. Qualified serving capacity (2 x 30 min soaks, fresh server each,
registered bounds, v2_verdict).**
- C=128: QUALIFIED x2. lag p95 129 ms (bound 320), fin p95 221 ms (500),
  backlog max 0.284 s (0.640), 0 lost, parity PASS, 123.7 audio-s/s,
  26.2 cores.
- C=144: QUALIFIED x2. lag p95 243 ms, fin p95 428/429 ms, backlog max
  0.384/0.464 s, TTFP penalty p95 +205/+206 ms (GOOD <= 250), 0 lost, parity
  PASS, 135.2-135.6 audio-s/s, 26.9-27.4 cores.
- **qualified_safe_concurrency = 144** for the research candidate (was 80
  for the shipped build, 2026-09-23). Margins at 144 are thin on
  finalization (86 % of the bound) and the TTFP penalty (82 % of GOOD); 128
  has wide margins everywhere. C=160 is the first failed screening point.
- Production flags of the candidate: `MYNAH_ASR_STREAM_PAR=4
  MYNAH_ASR_FIN_STACK=1`, everything else as the frozen C=80 profile (6x5 on
  cpus 0-29, generator 30-31, int8, lookahead 3, batch window 0,
  `--http-threads 32`). Build `dbae3fe`.

**3. Model/ASR quality.** Serving is transcript-preserving: the candidate's
transcripts are byte-identical to the shipped build on 498/498 clips and every
loaded stream matches them. WER mean 0.10634 / corpus 0.14618 on the 498;
0.10398 / 0.10856 on the 349 originals — unchanged, by identity. First-text
latency is a separate matter (S13-5c): unchanged by serving work, and the
blank bias is rejected.

**Still default OFF / research only.** Every flag above: making them the
default is a product decision, not taken. `v2_promote` dry-run accepts both
runs; nothing was applied. Also research-only: `MYNAH_ASR_KV_LAYOUT`
(ring/slide), `MYNAH_ASR_STREAM_PAR_DECODE`, `MYNAH_ASR_BLANK_BIAS`.

**The plateau campaign is closed.** No further micro-optimisation unless a
qualification exposes a concrete failure mechanism.

## AUDIT 2026-09-24 — is "0 lost, 0 errors" real? (read after the decision record)

**Question.** Does the harness actually count every way a streaming utterance
can fail, or could the qualification's zeros be an accounting artefact?

**Gaps found in the counting code** (`tools/bench/stream_load.py`,
`tools/bench/streaming_metrics.py`):
1. `ok` = no error and not rejected. An utterance where the server closes the
   socket WITHOUT a `done` frame (the reader breaks on opcode 0x8 and sets no
   error), or that receives no event at all, counts as OK; its finalization
   silently leaves the percentiles.
2. Warm-up hides errors: `aggregate()` drops the first `--warmup` seconds
   BEFORE counting `errored`, so a failure in the first 30 s of a soak is not
   in `counts.errors` (v2_verdict bound 1 reads that count).
3. In SOAK mode nothing checks that every stream process lived to the end: a
   stream process that dies stops producing records silently (`expected` is
   only set in WAVE mode).
4. Harness artefact, not a failure: the schedule gives stream i the clips
   i, i+C, i+2C... of a bank that alternates the three length classes, so
   when C is a multiple of 3 (96, 120, **144**) every stream plays ONE class
   for the whole run (C=144: stream 62 only long clips, 81 utterances; stream 99
   only short, 261). The fleet mix stays one third per class; the per-
   utterance mix shifts toward short clips, i.e. more opens and finalizations
   per second — a harder load, not an easier one.

**FACT — none of 1-3 happened in the qualification.** Re-counted from the raw
per-utterance records of all four soaks, INCLUDING warm-up:

| soak | utterances (all) | errors | rejected | OK without done / fin | empty text | streams alive to the end |
|---|---|---|---|---|---|---|
| C128 #1 | 17348 | 0 | 0 | 0 | 0 | 128 / 128 |
| C128 #2 | 17347 | 0 | 0 | 0 | 0 | 128 / 128 |
| C144 #1 | 23844 | 0 | 0 | 0 | 0 | 144 / 144 |
| C144 #2 | 23839 | 0 | 0 | 0 | 0 | 144 / 144 |

Every stream sent 1735-1822 s of audio in 1800 s (no stream starved); the
slowest stream's lag p95 is 273 ms against a median stream of 245 ms at C=144
(141 vs 128 at C=128). The zeros stand; the counting is still to be fixed so
that the NEXT run cannot hide them (S12-17).
