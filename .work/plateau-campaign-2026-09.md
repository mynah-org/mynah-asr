# Plateau campaign, Axion 2026-09-24 — why ~8 of 30 cores sit idle at saturation

Status: IN PROGRESS

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

(results pending)
