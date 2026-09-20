# Serving findings that later work must not silently revert

Status: frozen 2026-09-20 from the Axion (GCP c4a-highcpu-32, Neoverse-V2)
campaign. Every entry is labelled **FACT** (measured, or proven from the code),
**HYPOTHESIS** (plausible, needs an experiment), **RESULT** (what an experiment
returned) or **DECISION** (what changed because of it).

**How this file changes.** A FACT here is retired only by a measurement that
contradicts it, recorded the same way — with the host, the commit, the workload
and the failure dimensions. A conclusion that merely *reads better* does not
retire one. Several entries below exist because an earlier conclusion in this
repo was flattering and wrong, and the flattering version survived for days.

---

## F1 — The prefork wins of 2026-09-20 were invalid: those configurations lost streams

**FACT.** The first W x T sweep read as `8x3 holds c=32` and `4x6 holds c=24`
against `1x24 holds c=16`. Every winning rung was losing active streams to
`reader: timed out`:

```
  8x3  c=16  31/32   c=24  45/48   c=32  60/64
  4x6  c=16  29/32   c=24  43/48
  2x12 c=8    9/16
  1x24 c=8   16/16   c=16  32/32   c=24  48/48   (zero losses at every rung)
```

The harness scored those losses as a MARGINAL beside the percentiles, so the
table said the narrow topologies were better while they were dropping work.

## F2 — A dropped active stream is a hard failure

**DECISION**, and it is now enforced rather than remembered: lost utterances are
an envelope line with a limit of zero (`tools/bench/streaming_metrics.py`), so a
rung that loses one stream cannot be GOOD or MARGINAL.

A 503 is deliberately **not** counted there and still reads MARGINAL. Refused at
the door and dropped mid-sentence are different outcomes: the first is the
admission ladder working as designed, the second is a session a caller lost.

## F3 — 1x24 is the only topology that has reached the ceiling without losing streams

**FACT**, re-scored under F2:

| topology | as first scored | with losses counted |
|---|---|---|
| **1x24** | c=16 | **c=16, zero losses at every rung** |
| 4x6 | c=24 | c=8 |
| 8x3 | c=32 | c=8 |
| 2x12 | c=8 | c=0 |

**This is not yet a recommendation of one wide worker.** It is a statement that
no prefork topology has demonstrated a loss-free ceiling here. Prefork is
**suspect**, not a tuning target, until F8 is explained.

## F4 — The batch collection window widened the ready set and did not move the ceiling

**RESULT.** `--batch-window-ms` on the same ladder, 24 cpus:

| window | mean ready set at c=16 | emission lag p95 | ceiling |
|---|---|---|---|
| 0 ms | 3.69 | 288 ms | c=16 |
| 40 ms | 5.85 | 302 ms | c=16 |
| 80 ms | 7.98 | 370 ms | c=8 |

The ready set nearly doubled for no gain, and the wider window cost a rung.

## F5 — Fixed-cost amortisation is NOT demonstrated to be the limiting factor

**DECISION** following F4. The capacity law `T_step(B) = a + b*B` prices one
step serving B streams and its economy is that `a` is paid once. F4 falsifies
the claim that this is what limits this server. `.work/where-to-attack.md` §A3
and §A5 were written on the opposite assumption and must be re-argued from
measurement before any further work is spent on `a` — including the bf16/SVE2
and weight-layout lines, whose whole justification was that `a` is memory
traffic.

Related **FACT**: `a` fell from 84.1 ms to 30.8 ms between 8 and 24 cores. A
fixed cost dominated by weight traffic would not scale like that.

## F6 — Marginal per-stream cost rises with concurrency, and the isolated bench understates it

**FACT**, fitted from `mynah_asr_step_b_wall_ms_sum / mynah_asr_step_b_count` —
the step as the scheduler actually calls it:

| where | a | b per stream |
|---|---|---|
| bench, isolated | 30.8 ms | 4.58 ms |
| serving path at c=16 | 39.5 ms | **8.57 ms** |
| serving path at c=24 | 31.4 ms | **10.13 ms** |

So `T(B)` measured in isolation is not `T(B, C)` under service load, and the
error is about 2x and grows. **No capacity number may be derived from the
isolated step table alone.**

**HYPOTHESIS, untested**: the growth in `b` is LLC pressure, DRAM bandwidth,
thread-team wake cost, allocator traffic, per-stream feature/decoder work
competing with the encoder GEMM, or mixed lookahead presets fragmenting the
batch. Candidates, not conclusions.

## F7 — B=1 was never a warm-up anomaly

**FACT.** The step table's B=1 row was excluded from every calibration for days
under a note saying it "carries the warm-up". It does not: B=1 steps in a
long-running server, far past any first touch, cost the same. The row was a slow
path.

```
  B=1 through the single path   86.7 ms   86.7 ms per stream
  B=2 through the stacked path  38.4 ms   19.2 ms per stream
```

Two streams together cost less in absolute terms than one alone.

## F8 — There were TWO singleton gates, and the second is the general one

**FACT**, from `src/mynah_asr.c`:

1. `if (B == 1) return mynah_asr_stream_feed(...)` — the explicit B==1 exit.
2. `if (batched && g > 1)` inside the round loop — a **per-lookahead-preset**
   group of one took the same old path.

The second matters more: the group is per preset, so eight streams on eight
different presets are eight groups of one, each on the row-by-row path, **at any
concurrency**. A fleet serving mixed presets was paying this everywhere.

## F9 — With singleton groups stacked, transcripts are identical and the counters prove the path moved

**RESULT.** `tests/test_stream_batch`:

```
  before  [int8] B=1: 17 steps,  0 stacked rows | dot 4173  dot_rows 0    | IDENTICAL OK
  after   [int8] B=1: 17 steps, 64 stacked rows | dot  333  dot_rows 3840 | IDENTICAL OK
```

**And the gate that passed it was vacuous before.** It compared B=1 against the
single path while B=1 *was* the single path. Removing the exit is what gave that
comparison something to say. A green gate is not evidence until someone has
checked that it exercises the path it claims to.

Step table after, same box:

```
  B=1 16.75 ms (5.2x)   B=2 25.82   B=4 40.42   B=8 72.22 (unchanged)
```

Server A/B, one binary, `MYNAH_ASR_STACK_SOLO` as the only difference, arms
alternated, c=8 fresh server each time:

```
  ON   105 / 104 / 103 ms emission lag p95, 16/16 every rung
  OFF  175 / 175 / 43793 ms, the last losing 3 of 16
```

---

## F10 — At its own ceiling this server is at about 30% CPU

**FACT**, `htop` during a c=16 rung: cpus 0-23 at ~30% each, 24-31 idle. About
seven cores of work spread over twenty-four, while the envelope breaks.

**Consequence**: the measured ceiling is a property of how the machine is used,
not of the machine. Any statement of the form "this box holds N streams" is
provisional until utilisation is accounted for.

**HYPOTHESIS, under test**: with `ready_mean = 2.13`, a step is a two-row GEMM
handed to 24 threads, so the cost is rendezvous rather than arithmetic — the
pool meter over the same run reports 389 197 dispatches in 146 s at 81% spin
wins. If true, T should be sized to the batch width and not to the core count.
The experiment is a step table at T = 2, 4, 8, 12, 16, 24 per B.

## F11 — `reader: timed out` is concurrency-dependent, probabilistic, and inside the server

**FACT**, six rungs at each width on one box, same everything:

```
  c=12   6 runs, 6 clean            lag 143-147 ms
  c=16   6 runs, 4 clean, 2 stalled lag 201-218 ms clean, 29 200-29 336 ms stalled
```

Never at c=12; about one run in three at c=16.

**FACT**, from the server's own counters at a stall: `lag_ms max=29960`, and that
histogram measures *sample received to delta emitted*. The audio was inside the
server. Not the socket, not the writer, not the network.

**FACT**: `pool inline=0` at the same moment — the thread pool never degraded to
serial, which rules out the single-slot trylock fallback the sibling engine
flagged.

**FALSIFIED HYPOTHESIS**: "a rung fails when it asks for more streams than that
server has ever held". The increase rung passed and the identical repeat failed.

**Not yet explained.** Until it is, prefork is not promoted in any profile and
the timeout value is not raised.

---

Evidence: `.work/axion-session-2026-09-20.md`,
`configs/perf/axion-c4a-highcpu32-nemotron-streaming.json`, and on the box
`~/hunt/`, `~/ladder2/`, `~/sweep/`, `~/window_ab/`, `~/solo_ab/`.
