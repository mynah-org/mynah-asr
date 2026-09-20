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

**Updated the same day by a later RESULT**: with the singleton fix and the
corrected gate, **2x12 held c=24 with zero losses at every rung** (79/130/192 ms
at c=8/16/24) and 1x24 held c=16, while 4x6 and 8x3 still lost streams. So a
prefork topology HAS now reached a ladder ceiling loss-free — and F12 explains
why that is not enough to promote it.

Prefork remains **suspect**, not a tuning target, until F11 is explained.

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

**That hypothesis is FALSIFIED.** It was: "a step is a two-row GEMM handed to 24
threads, so the cost is rendezvous rather than arithmetic; T should be sized to
the batch width". The experiment was a step table at T = 2, 4, 8, 12, 16, 24 for
each B, on the idle box.

**RESULT** — the step scales to 24 threads at every width, B=1 included:

```
         T=2     T=4     T=8    T=12    T=16    T=24    T*
  B=1   80.0    46.3    27.8    22.6    18.4    16.7    24
  B=2  103.2    61.0    38.9    32.6    27.7    25.5    16
  B=4  144.8    85.4    55.6    48.1    41.6    40.2    16
  B=8  245.4   146.5    97.9    84.8    75.0    73.3    16
```

T* (the smallest T within 10% of the best) is 16-24, not 4 or 8. **The machine
is used effectively while a step is executing**, and the extra threads are work,
not rendezvous.

**FACT, and this is what the 30% actually is**: from one run's own counters,
`batch steps=1276` at `step_wall_ms_mean=31.4` over `uptime 146.8 s` is 40 s of
model execution in 147 s of wall — an **execution duty cycle of about 27%**. The
low average CPU is a DUTY-CYCLE observation, not evidence of poor intra-step
scaling.

**Do not resurrect the rendezvous explanation** without a measurement that
contradicts this table. And note what the 27% does NOT yet say: idle time is
only waste if runnable work existed during it, and that has not been measured —
a stream with less than a chunk of audio is not runnable work.

## F11 — `reader: timed out` is concurrency-dependent, probabilistic, and inside the server

**FACT**, six rungs at each width on one box, same everything:

```
  c=12   6 runs, 6 clean            lag 143-147 ms
  c=16   6 runs, 4 clean, 2 stalled lag 201-218 ms clean, 29 200-29 336 ms stalled
```

Never at c=12; about one run in three at c=16.

**FACT**, from the server's own counters at a stall: `lag_ms max=29960`, and that
histogram measures *sample received to delta emitted*. So the audio was inside
the server when the delta was finally produced. That bounds where to look; it
does **not** by itself exonerate the socket, the writer or backpressure, because
a blocked output path can hold the ring and stop the ingest that feeds it.

**FACT**: `pool inline=0` at the same moment — the thread pool never degraded to
serial, which rules out the single-slot trylock fallback the sibling engine
flagged.

**FALSIFIED HYPOTHESIS**: "a rung fails when it asks for more streams than that
server has ever held". The increase rung passed and the identical repeat failed.

**FACT**, from the 600 s soak at 2x12, c=24 (1105/1118 utterances, 13 lost):
degradation is **strongly non-uniform and episodic**. Median service stays
healthy throughout while the tail and the backlog explode for minutes, new
streams still start, and the system recovers without a restart.

```
  window        p50 / p95 ms
  0-240 s       59-69 / 153-163      healthy
  240-420 s     94-95 / 36647-55509  episode, three minutes
  420-600 s     70-71 / 158-165      recovered by itself
  660-720 s     22164 / 55981        again
  [PASS] drift ttfp_ms p95 8%      [FAIL] backlog max 60.2 s
```

**HYPOTHESIS, under test and NOT yet a fact**: a subset of established slots is
being starved of scheduler/model service. `tools/bench/stall_timeline.py` and the
per-slot SIGUSR1 dump exist to separate that from the alternatives, which are
equally live until the evidence lands:

| class | shape | what it would mean |
|---|---|---|
| A | ring full, ready, visible, no model progress | scheduler/model-service starvation |
| B | ring full, ready, NOT scheduler-visible | readiness/bookkeeping bug |
| C | ring empty, RX stopped | ingest/transport/backpressure — not the scheduler |
| D | ring has data, never reaches the step requirement | chunk accounting |
| E | no slot on the worker progresses | worker/model execution stall |
| F | every slot degrades together | capacity or a global execution problem |

## F13 — The stall is the scheduler thread blocking inside its own poll pass

**RESULT**, 2026-09-20, twelve rungs at c=16 with SIGUSR1 every 5 s through each
rung. Two episodes captured, both identical in shape.

Per-slot, at the moment of the episode:

```
  seq  out rdy  steps  ring_s  need_s  since_rx  since_step  lag_max
   18    1   0     17     0.1    0.32       0.2         0.2      433
   19    1   0     23     0.1    0.32       3.3         3.3      433   onset
   24    1   0     23     0.1    0.32      28.3        28.3      433
   25    1   0     12     0.0    0.32       0.1         0.1      214   slot RE-CLAIMED
```

**CLASS E** by `tools/bench/stall_timeline.py`: all sixteen live slots stopped
together while work was runnable — one of them held 4.9 s of audio with
`ready=1, out=1`, unserved for 26 s. Execution stopped; it was not starved of
work.

And the scheduler's own liveness names the place:

```
  seq 18  loops=4726 phase=poll phase_s=1.2
  seq 19  loops=4726 phase=poll phase_s=6.2
  seq 23  loops=4726 phase=poll phase_s=26.2
  seq 24  loops=4730 phase=step phase_s=0.2      <- released
```

The loop counter is **frozen** and the phase is **poll**. That is pass (1) of
`sched_main`, and it excludes the three alternatives by construction:

| if it were | the dump would say |
|---|---|
| a lost wakeup | `phase=park`, loops frozen |
| a livelock in the passes | loops **climbing**, nothing staged |
| a step that will not end | `phase=step`, loops frozen |

It says none of those. The scheduler is **blocked on a lock inside the poll
pass**, for 26 s, while every stream on the worker waits behind it.

**FACT, from reading the code rather than measuring it**: pass (1) can only
block on a mutex. `mynah_asr_stream_out_peer_gone()` polls with a **zero**
timeout and cannot block on the socket, and `mynah_asr_stream_out_enqueue()`
cancels the stream on ring overflow rather than blocking — the failure mode the
sibling engine measured at 66.5 s does not exist here. That leaves the slot lock
`s->mu` and the output-ring lock `o->mu`.

**HYPOTHESIS, not yet proven**: the holder is the ingest thread, which takes
`s->mu` and then rings the scheduler's doorbell — `mynah_asr_slot_push()` calls
`g_notify()` **while still holding `s->mu`**, and `g_notify` takes `g.mu`. That
is a lock order (`s->mu` then `g.mu`) opposite to nothing the scheduler does
today, so it is not yet an ABBA proof; it is the first place to look.

**EXPERIMENT, ready and not yet run** (the box was handed to another campaign):
the phase marker now records the SLOT INDEX and splits pass (1) into slot-poll,
take-requests and the cancel/peer-check block. The next episode says which
stream's lock, and a lock has an owner.

**Recovery is not self-healing.** At seq 25 the slot's `steps` counter goes
BACKWARDS, 23 to 12: the slot was released and re-claimed by the next rung's
stream. The episode ends when the clients give up, not when the server recovers.
The earlier reading of the 600 s soak as "recovers without a restart" was the
turnover of streams, and is corrected here.

**Not yet explained at the level of the responsible line.** Until it is, prefork
is not promoted in any profile and the timeout value is not raised.

## F12 — A short ladder cannot qualify a topology

**DECISION**, forced by the soak. 2x12 passed every rung of a screening ladder
at c=8, 16, 24 with zero losses and the best latency of any topology measured,
and then lost 13 established streams in a ten-minute soak at its own best rung.
A ladder rung is twenty seconds; the episodes are three minutes long, so a
ladder cannot see them at all.

So: **a prefork topology is not promoted into a profile on ladder evidence.** It
needs a long soak with zero lost established streams. 2x12 is the current best
CANDIDATE on this box and is deliberately not in
`configs/perf/axion-c4a-highcpu32-nemotron-streaming.json`.

---

Evidence: `.work/axion-session-2026-09-20.md`,
`configs/perf/axion-c4a-highcpu32-nemotron-streaming.json`, and on the box
`~/hunt/`, `~/ladder2/`, `~/sweep/`, `~/window_ab/`, `~/solo_ab/`.

---

## F14 — No lock-order cycle among the AUDITED paths; the stall is not yet explained

**FACT**, from reading every acquisition of the three locks rather than assuming:

| edge | where | inverse found? |
|---|---|---|
| `s->mu` → `g.mu` | `mynah_asr_slot_push()` rings the doorbell with the slot lock held: `g_notify()` → `mynah_asr_sched_wake()` → `g.mu` | **none** |
| `g.mu` → anything | every `g.mu` critical section in `sched.c` touches only the offline queue and the wake flags | never takes a slot or ring lock |
| `o->mu` → `close(fd)` | the writer retires the descriptor under the ring lock | `close()` cannot block: no `SO_LINGER` is set |
| `s->mu` → `cond_wait` | `slot_push` on a full ring | releases the mutex while it waits |

**No lock-order cycle was found among the audited `g.mu`, `s->mu` and `o->mu`
acquisition paths.** That is the whole claim. It does **not** exclude a wait
cycle through another primitive — the condition variables `s->space`, `o->cv`,
`g.wake`, `g.job_done`, the thread pool's own rendezvous — nor a dependency
outside the three locks audited here. Those are un-audited, not cleared.

What it does mean is that a critical section held long enough to stop everything
behind it is now at least as likely as an inversion, and it is the cheaper of
the two to falsify.

Two candidates were cleared by reading, not by measuring:

- `mynah_asr_stream_out_finish()` does **not** join the writer — the writer is
  detached and there is no `pthread_join` in `server/stream_out.c` at all.
- `SO_SNDTIMEO` defaults to **5 s** (`STREAM_OUT_DEFAULT_TIMEOUT_MS`), and the
  observed freeze is 26-28 s, so a single send timeout does not explain it.

**HYPOTHESIS**: some holder of `s->mu` or `o->mu` keeps it for tens of seconds,
and nothing in the server can currently name that holder.

**EXPERIMENT, built and ready, not yet run** (the box was handed to another
campaign): every acquisition of a slot's mutex now records the owning thread,
the call site that took it, and when — **three** relaxed stores against the cost
of the mutex itself — and both waits clear the owner while they wait, so a dump
cannot name a thread that is holding nothing.

**The instrument was audited before it was trusted**, and the audit found a real
defect in it: `mynah_asr_slot_wait_done()` used a raw `pthread_cond_timedwait`
that released `s->mu` without clearing the owner. That is the call the INGEST
thread makes at the end of a stream — precisely the moment the stall occurs — so
the first dump would have accused a thread that was holding nothing, at exactly
the point where the accusation would have looked most convincing. Both waits are
wrapped now, the owner is restored on every return including a timeout, and the
four raw primitives that remain in `server/slot.c` are the ones inside the
wrappers themselves. SIGUSR1 prints
`mu_owner=<tid> mu_held_s=<s> mu_where=<function>` on every live slot, beside
the scheduler's own `phase=<sub-phase> slot=<index>`.

The next episode therefore answers three questions in one dump: which lock the
scheduler is waiting for, who owns it, and since when. **No scheduler behaviour
is changed until those three agree with the graph above.**
