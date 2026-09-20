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

---

## F15 — The stall, explained: a blocking peek on a socket that has two readers

**FACT.** `mynah_asr_stream_out_peer_gone()` polls the connection and, on
`POLLIN`, peeks one byte to tell a control message from a hangup. The comment
beside that peek claimed it could not block because `poll()` had just reported
the socket readable.

**That invariant is false in this server**, and the reason is three lines away in
`server/main.c`: the connection is `dup()`ed. The ingest thread reads WebSocket
frames from `rfd` continuously; `stream_out` owns `fd`. **One socket, two
readers.** So:

```
  poll() reports POLLIN
  -> the ingest thread drains the socket
  -> recv(fd, MSG_PEEK) finds nothing
  -> blocking descriptor, no SO_RCVTIMEO on this dup
  -> the scheduler waits for the client to send again
```

and it waits **holding that ring's mutex and the scheduler loop**, so every
stream on the worker stops behind it.

**EVIDENCE** (Axion, commit `fb74287`, c=16, `--repeat 6` on a 4.3 s clip to
raise session turnover, SIGUSR1 every 2 s):

```
  sched  loops=7857  phase=cancel  slot=15  phase_s=0.9 → 25.9   (loop frozen)
  slot15 out_owner=f2d20e339160  out_held_s=25.9
         out_where=mynah_asr_stream_out_peer_gone
         mu_owner=0                       (no slot mutex held by anyone)
```

`out_held_s` and `phase_s` grow in lockstep, and the owner is **the scheduler
thread itself**. It is not waiting for another thread's lock: it is inside the
function, blocked in a syscall, holding the lock. Four captures across three
runs, same shape; the episode always ends when the clients give up.

**DECISION — the minimal fix**: `MSG_PEEK | MSG_DONTWAIT`. The `EAGAIN` branch
below already meant "not gone", which is exactly right: the other reader got
there first, so ask again next pass. No timeout was raised, no buffer grown, no
retry added.

**RETIRED by this result:**

- *"a subset of established slots is being starved"* — the scheduler was not
  choosing badly, it was not running at all.
- *"the ingest thread holds the slot lock"* — `mu_owner=0` on every slot, in
  every capture after the slot mutex was instrumented.
- *"the output path cancels rather than blocks, so it cannot hold the
  scheduler"* — true of `enqueue`, false of `peer_gone`, which is in the same
  file and was cleared by the same reading. **Reading one function does not
  clear its neighbours.**

## F16 — What the per-thread CPU numbers do and do not say

**FACT**, `/proc/<pid>/task/*/stat` over one c=16 run: the scheduler thread
accumulated **174.8 s** of CPU, every pool worker **94.5 s**.

**NOT ESTABLISHED, and previously overstated here**: that this decomposes into
"94.5 s parallel plus 80 s serial". CPU time accumulated by different threads
cannot be partitioned that way without knowing which phases each thread
participated in and when. The scheduler thread also runs work inside parallel
regions as the calling thread, so part of its excess is parallel work, not
serial.

**And the explanation of 1x24 vs 2x12 is a HYPOTHESIS, not a measurement.**
"Two workers duplicate a serial bottleneck" is one candidate among several:
temporal hole-filling between independent domains, different ready-set dynamics,
different batch widths, reduced rendezvous, cache/scheduling effects, or a
combination. §3 and §4 of the campaign plan exist to separate them, and nothing
here selects one.

Related **FACT** worth keeping separate from all of that: the eight idle cpus
visible in `htop` during these runs are **reserved by the harness**
(`taskset -c 0-23` for the server, `24-31` for the generator). They are
experimental design, not a server defect: a generator sharing the server's cores
measures the generator.

---

## F17 — The fix, verified: before/after on the same reproducer and two 600 s soaks

**EVIDENCE**, Axion, commit `2ebf087`, load generator pinned off the server cores.

**The reproducer that caught it** (c=16, `--repeat 6` on a 4.3 s clip, high session
turnover, SIGUSR1 every 2 s):

| | rungs | utterances offered | lost | freeze episodes |
|---|---|---|---|---|
| before | 11 | 1056 | 2 | 2 |
| **after** | **22** | **2112** | **0** | **0** |

Emission lag p95 steady at 433-479 ms across all 22 rungs.

**600 s soaks**, same commit, same corpus, same client placement:

| | 1x24, c=16 | 2x12, c=24 |
|---|---|---|
| utterances | 242/242 | **1243/1243** |
| **lost** | **0** | **0** |
| emission lag p95 | 87 ms | 153 ms (limit 320) |
| finalization p95 | 324 ms | 277 ms (limit 500) |
| backlog max | 0.484 s | 0.484 s (limit 0.640) |
| steady drift / trend | 36 % / −14 % | **10 % / −9 %** |
| **useful audio per wall second** | **12.63** | **20.36** |

The same 2x12 c=24 configuration before the fix: **13 streams lost**, emission lag
p95 **24 363 ms**, backlog **60.2 s**. The multi-second freeze was never a
property of the model's capacity — it was a blocking `recv()` in the control
path that stopped the scheduler outright.

**FACT**: 2x12 at c=24 delivers **61 % more useful audio per wall second** than
1x24 at c=16, with zero losses in ten minutes and every envelope line inside.

**FACT about the gate, not the server**: the 20 % drift threshold discriminates
between these two runs *after* ramp and drain windows are excluded — 2x12 reads
10 % and passes, 1x24 reads 36 % and does not, and 1x24's steady windows really
do vary more (74-116 ms around a pooled 87) than 2x12's (144-168 around 153).
The threshold was contaminated, not too tight, and it has not been changed.

**NOT YET PROMOTED.** One soak is not a qualification: the profile requires a
second independent long soak at the same operating point, and the cause of
1x24's wider steady spread is unexplained. 2x12 c=24 remains the CANDIDATE.

**HYPOTHESIS for why 2x12 wins, explicitly not selected**: duplication of a
serial bottleneck, temporal hole-filling between independent domains, different
ready-set dynamics, different step widths, reduced rendezvous, cache effects, or
a combination. The execution-duty experiment exists to separate them, and no
number here chooses one.

---

## F18 — The 3x8 capacity curve on the Axion: C_safe 32, knee 36, overload 48

**EVIDENCE**, Axion (GCP `c4a-highcpu-32`, Neoverse-V2, 32 vCPU), commit
`a941390`, 2026-09-20T15:19Z. Concurrency is the **only** independent variable:
every rung is a fresh server, same commit, same model
(`nemotron-3.5-asr-streaming-0.6b`, `--quant int8`), same corpus
(`fleurs_1521` 7.4 s / `fleurs_1534` 11.9 s / `test_en` 4.3 s, bank
`short,medium`, seed 42), same pacing (`lookahead 3`, real time), same affinity
(server `taskset -c 0-23`, generator `taskset -c 24-31`), same warm-up (25 s)
and same run length (90 s).

Topology: **3 workers x 8 threads**, `--threads 96 --cap 96`.

| C | est. | lost | rej | audio/wall | ttfp p95 | lag p50/p95 | backlog | duty | r_idle | no_work | finalize | meanB | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 86 | 0 | 0 | 7.62 | 1545 | 38/75 | 0.184 | 0.247 | 0.248 | 0.505 | 0.086 | 1.02 | HEALTHY |
| 12 | 129 | 0 | 0 | 11.55 | 1529 | 37/80 | 0.164 | 0.352 | 0.201 | 0.446 | 0.119 | 1.06 | HEALTHY |
| 16 | 169 | 0 | 0 | 15.17 | 1546 | 38/93 | 0.184 | 0.473 | 0.202 | 0.325 | 0.167 | 1.11 | HEALTHY |
| 20 | 210 | 0 | 0 | 19.07 | 1552 | 39/103 | 0.244 | 0.589 | 0.192 | 0.219 | 0.209 | 1.17 | HEALTHY |
| 24 | 252 | 0 | 0 | 22.65 | 1559 | 51/129 | 0.244 | 0.677 | 0.162 | 0.161 | 0.251 | 1.32 | HEALTHY |
| 28 | 292 | 0 | 0 | 26.33 | 1597 | 70/173 | 0.284 | 0.755 | 0.145 | 0.100 | 0.293 | 1.59 | HEALTHY |
| **32** | 329 | **0** | 0 | **29.58** | 1636 | 94/**231** | **0.464** | 0.793 | 0.135 | 0.072 | 0.319 | 2.11 | **HEALTHY** |
| **36** | 369 | 0 | 0 | 32.90 | 1744 | 145/**345** | 0.564 | 0.813 | 0.133 | 0.055 | 0.354 | 3.12 | **NEAR KNEE** |
| 40 | 399 | 0 | 0 | 35.03 | 1924 | 234/601 | 0.904 | 0.817 | 0.136 | 0.047 | 0.373 | 4.71 | NEAR KNEE |
| **48** | 420 | 0 | 0 | 39.76 | 2113 | 856/**1379** | **1.604** | 0.820 | 0.144 | 0.036 | 0.423 | 11.05 | **OVERLOADED** |
| 56 | 432 | 0 | 0 | 38.47 | 2448 | 1580/3265 | 3.544 | 0.817 | 0.142 | 0.041 | 0.413 | 13.65 | OVERLOADED |
| 64 | 449 | 0 | 0 | 40.84 | 2755 | 2273/4763 | 4.564 | 0.818 | 0.144 | 0.038 | 0.434 | 15.34 | OVERLOADED |

Envelope: emission lag p95 <= 320 ms (= `(lookahead+1) x 80 ms`), backlog max
<= 0.640 s, lost established streams = 0.

**The `finalize` column of this table is WITHDRAWN — see F21.** The harness
that printed it computed the share against the wrong denominator and overstated
it by about 3x. Recomputed from the raw per-worker dumps, finalize is 0.106 of
wall at C = 32, 0.118 at C = 36 and 0.125 at C = 40; the other nine rungs cannot
be recomputed because a later run overwrote their server logs, so their values
are struck rather than corrected. Every other column here was recomputed from
the same raw dumps and is unchanged: `duty` and `runnable_idle` at C = 32/36/40
reproduce to within 0.003.

**FACT**: on this machine, this build and this workload, **C_safe = 32**
(lag p95 231 ms, backlog 0.464 s, zero loss), the **latency knee begins at
C = 36** (345 ms, first crossing of the envelope), **C = 40 still completes all
its work** but at 601 ms and 0.904 s backlog, and **C = 48 is overloaded**
(backlog 2.5x the limit).

**FACT**: useful throughput ceiling is approximately **40 audio-s/wall-s**. From
C = 48 upward the curve is flat (39.76, 38.47, 40.84) while latency grows by
3.5x — more offered concurrency buys latency, not work.

**FACT**: **zero lost established streams and zero admission rejections at every
rung, including C = 64.** Above the knee this server degrades by getting slow,
not by dropping work. That is the post-F15 behaviour and it is the behaviour
rule 5 asks for.

**This is a short-run result, not a production capacity claim.** Rungs are 90 s.
F12 says in this same file that a short ladder cannot qualify a topology.
C = 32 is `measured_short_run_safe`; it becomes `long_soak_qualified` only after
an independent long soak at that point.

**NOT universally optimal.** 3x8 is the current best measured candidate **on
this machine and this workload**. The topology comparison at equal concurrency
has not been run since F17, and no result here licenses 3x8 on another box.

---

## F19 — RETIRED by F21: "duty plateaus at 0.82 while runnable work exists"

**EVIDENCE**: the same F18 run, three-way split of scheduler wall (commit
`bc572f0` instrumentation: `model_busy` / `runnable_idle` / `no_work`, computed
independently of phase).

From C = 36 upward, with concurrency, backlog and mean batch size all still
rising, the split stops moving:

| C | duty | runnable_idle | no_work | meanB | backlog |
|---|---|---|---|---|---|
| 32 | 0.793 | 0.135 | 0.072 | 2.11 | 0.464 |
| 36 | 0.813 | 0.133 | 0.055 | 3.12 | 0.564 |
| 40 | 0.817 | 0.136 | 0.047 | 4.71 | 0.904 |
| 48 | 0.820 | 0.144 | 0.036 | 11.05 | 1.604 |
| 56 | 0.817 | 0.142 | 0.041 | 13.65 | 3.544 |
| 64 | 0.818 | 0.144 | 0.038 | 15.34 | 4.564 |

**RETIRED, 2026-09-20, by F21.** The measurement is sound; the reading of it was
not. `w_model` counted **only phase 4**, the batched step. Phase 5, which is
99.8 % `mynah_asr_stream_finish` — the model executing a tail — was charged to
`runnable_idle`. So the sentence this entry used to carry, "13-14 % with work
runnable and the model not executing", described the model *executing*.

Corrected, from the same raw dumps: model execution is **0.896 of wall at
C = 32, 0.928 at C = 36 and 0.940 at C = 40**, and the genuinely reclaimable
idle is **1-3 %**, not 13-14 %.

**What survives**: the phase-4 plateau itself is real and reproduces — the
batched step holds 0.790 / 0.810 / 0.815 of wall at C = 32/36/40 while
concurrency, backlog and mean batch all keep rising. What does NOT survive is
the claim that the remainder is available to reclaim. It is mostly the same
model, running at batch width one.

**And this answers F10 the other way round.** Cores 0-23 are not idle at the
knee: the fleet executes the model for 94 % of scheduler wall at C = 40. The
serving problem is not utilisation, it is **efficiency** — 12.5 % of the wall
runs the model one stream at a time while the batched path over the same
weights averages 4.71 rows.

**NOT ESTABLISHED**: that finalize *causes* the plateau. F20 shows finalize is
large and correlates with `runnable_idle`, and the phase attribution charges
63-96 % of the avoidable idle to it. Correlation between two aggregate shares is
not causality, and the readiness predicate used by the diagnostic has not yet
been validated against the scheduler's own runnable predicate.

---

## F20 — Finalize is 100 % model tail and 0 % teardown, at a flat 82 ms per call

**EVIDENCE**, Axion, commit `0761a35` (which times `mynah_asr_stream_finish`
apart from the teardown around it), 2026-09-20T15:48Z, same 3x8 setup as F18,
rungs at the knee:

| C | finalize model | teardown | calls | model ms/call | teardown ms/call | finalize share of wall |
|---|---|---|---|---|---|---|
| 32 | 31.5 s (**99.8 %**) | 0.0 s (0 %) | 384 | **82.1** | 0.0 | **0.106** |
| 36 | 35.5 s (**99.8 %**) | 0.0 s (0 %) | 432 | **82.3** | 0.0 | **0.118** |
| 40 | 38.3 s (**99.8 %**) | 0.0 s (0 %) | 468 | **81.9** | 0.0 | **0.125** |

The share column is the corrected one (F21): the harness printed 0.319 / 0.354 /
0.373 against the wrong denominator. The 99.8 % is `mynah_asr_stream_finish`
against the whole of scheduler phase 5, recomputed from the raw per-worker
dumps — so `sched_feed_tail` and the availability scan that share the phase cost
0.2 % of it between them, and the tail really is all of it.

**FACT**: the teardown — the done frame, the session close, the bookkeeping, the
reset — costs **nothing measurable**. Every microsecond charged to finalize is
the model running inside `mynah_asr_stream_finish`.

**FACT**: the per-call cost is **flat at 82 ms across C = 32, 36 and 40**, while
the mean batch of ordinary steps over the same rungs goes 2.11 -> 3.12 -> 4.71.
Finalize does not amortise with load **because it is not batched**: `sched.c`
runs it one slot at a time in phase 5, and `mynah_asr_stream_finish` is a loop of
`stream_flush_chunk` calls at B = 1. The comment in `sched_feed_tail` says so
outright — "never through the batch".

**FACT**: finalize is therefore a fixed cost **per utterance**, not per second
and not per stream-hour. Its share of scheduler wall is set by the utterance
rate. At C = 40 the workload's mean utterance is ~6.8 audio-s, so the server pays
82 ms of exclusive B = 1 model time for every ~6.8 s of audio it transcribes —
**12.5 % of all scheduler wall**, spent at a batch width of one while the ready
set holds 4.71 rows on average.

**ARITHMETIC, stated before the experiment rather than after it.** Removing
finalize entirely can return at most 12.5 % of wall at C = 40, and only if the
tail work vanished rather than moved. Going from C = 32 to C = 40 is +25 %
offered load. So batching finalization, on its own, is **predicted not to be
enough to make C = 40 healthy**. If the probe below shows C = 40 healthy with
finalize near zero, that prediction is wrong and finalize is worth attacking
hard; if it shows C = 40 still outside the envelope, finalize is not the
barrier and the limit is in phase 4. Either way the answer arrives before any
code is written.

**DECISION**: this is the candidate mechanism, and it is testable without
touching the code. Finalize's share depends on utterance length and on nothing
else the server controls, so slicing one source file into 7 s / 24 s / 94 s
clips varies the finalization rate ~13x with the audio content, model, quant,
topology, affinity, pacing, seed and run length all held identical. If finalize
limits the knee, C = 40 returns inside the envelope as the utterances grow. That
probe runs before any finalize optimisation is written.

**NOT ESTABLISHED, and must not be asserted until the probe returns**: that
batching finalization would move C_safe. A cheaper finalize that does not move
the capacity curve is not a serving improvement (F18's envelope is the
criterion, not the profiler's percentages).

**CONSTRAINT on any fix**: rule 4. The tail carries `is_last = 1` and the causal
right pad; any batched finalization ships with a bit-exactness gate against the
serial path, or it does not ship.

---

## F21 — The avoidable idle was mostly the model: a denominator and a predicate, both wrong

**How it was caught**: not by a new experiment. By recomputing F18's own table
from the raw per-worker `[DUMP]` lines instead of trusting the campaign script
that printed it, before building an optimisation on the number.

**DEFECT 1, the denominator.** The capacity harness computed finalize's share of
wall against a denominator roughly 3x too small, so it reported 0.319 / 0.354 /
0.373 at C = 32/36/40 where the raw dumps say **0.106 / 0.118 / 0.125**. Every
other column of that table recomputes to within 0.003, so the error is confined
to that one column. Nine of the twelve rungs cannot be recomputed: a later run
reused the output directory and overwrote their server logs. Those nine values
are struck, not corrected.

**DEFECT 2, and the one that mattered.** The three-way split of scheduler wall
counted **only phase 4** as model execution:

```c
if (prev_ == 4) g.w_model += d_;
else if (mynah_asr_slot_ready_count() > 0) g.w_runnable_idle += d_;
else g.w_no_work += d_;
```

Phase 5 is finalization, and F20 measured it at **99.8 % `mynah_asr_stream_finish`**
— the model, executing. So every tail the fleet ran was booked as *the model
not executing while work was ready*. The instrumentation was measuring
correctly and the label was false.

**FACT**, recomputed from the raw dumps of the same run, commit `a941390`:

| C | phase 4 (batched step) | phase 5 (finalize) | runnable idle | no work | **model execution** |
|---|---|---|---|---|---|
| 32 | 0.790 | 0.106 | 0.135 | 0.075 | **0.896** |
| 36 | 0.810 | 0.118 | 0.132 | 0.058 | **0.928** |
| 40 | 0.815 | 0.125 | 0.137 | 0.048 | **0.940** |

`runnable_idle` minus `finalize` is **0.029 / 0.014 / 0.012**. That residue is
the avoidable idle. Everything else in that bucket was the model.

**This retires the central claim of F19.** There is no 13-14 % of reclaimable
wall on this box. At C = 40 the fleet executes the model for **94 % of scheduler
wall**, and the question is not how to keep it busier but how to make 12.5 % of
that work stop running at batch width one.

**DECISION, in the code**: `w_model_solo` is now a bucket of its own, holding
phase 5 and phase 6 — model execution *outside* the batched ready set. It is
deliberately not folded into `w_model`, because the distinction between them is
exactly the thing worth fixing. The dump prints `execution_duty` (phase 4) and
`model_duty` (phases 4+5+6) side by side: one number can never separate "the box
is idle" from "the box is busy at batch width one", and this repo reported the
second as the first for a day.

**DECISION, in method**: a number that selects what to optimise gets recomputed
from the raw artefact before it is acted on. Both defects here survived a
12-rung campaign, a findings entry, a profile and a commit, and neither would
have been caught by re-running the campaign — only by recomputing it.

**FACT about the readiness predicate, from the audit added the same day**: the
diagnostic predicate has **false_ready 0 and false_not_ready 6 of 65** on a
3-stream smoke. Its errors are one-sided, so the measured `runnable_idle` is a
floor and never a ceiling. That is the opposite direction from the error above
and does not offset it.

---

## F22 — The BEFORE ladder, with the accounting fixed: the box is compute-bound, not scheduling-bound

**EVIDENCE**, Axion, commit `a48d443`, 2026-09-20T16:42Z-17:10Z. Same protocol as
F18 — fresh server per rung, 3x8, cpus 0-23 server / 24-31 generator, 25 s
warm-up discarded, 90 s measured, mixed corpus — with two differences: the build
carries the ready-to-execution histograms and the corrected split, and the
generator is given `--seed 42`. Every share is computed from the raw per-worker
dumps, not from a ratio the server printed (F21).

| C | audio/wall | lag p50/p95 | backlog | step | finalize | r_idle | no_work | **model** | meanB | rdy->start p50/95/99 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 7.70 | 38/75 | 0.144 | 0.248 | 0.028 | 0.236 | 0.488 | 0.276 | 1.02 | 0/34/44 | HEALTHY |
| 12 | 11.43 | 37/78 | 0.184 | 0.356 | 0.040 | 0.189 | 0.416 | 0.396 | 1.06 | 0/40/60 | HEALTHY |
| 16 | 15.02 | 39/92 | 0.184 | 0.477 | 0.056 | 0.161 | 0.306 | 0.533 | 1.12 | 0/40/77 | HEALTHY |
| 20 | 18.76 | 40/105 | 0.224 | 0.591 | 0.070 | 0.124 | 0.215 | 0.661 | 1.19 | 4/47/80 | HEALTHY |
| 24 | 22.65 | 52/133 | 0.244 | 0.684 | 0.084 | 0.082 | 0.150 | 0.768 | 1.30 | 13/63/93 | HEALTHY |
| 28 | 26.29 | 70/170 | 0.284 | 0.758 | 0.095 | 0.051 | 0.095 | 0.854 | 1.57 | 20/84/134 | HEALTHY |
| **32** | **29.58** | 100/**236** | **0.444** | 0.798 | 0.108 | 0.030 | 0.064 | **0.905** | 2.17 | 27/**114**/171 | **HEALTHY** |
| **36** | 32.86 | 150/**373** | 0.604 | 0.811 | 0.121 | 0.021 | 0.048 | 0.931 | 3.26 | 43/**173**/243 | **NEAR KNEE** |
| 40 | 35.17 | 249/667 | 0.964 | 0.815 | 0.125 | 0.014 | 0.046 | 0.940 | 4.77 | 80/277/347 | NEAR KNEE |
| **48** | 39.36 | 920/**1475** | **1.664** | 0.817 | 0.146 | 0.007 | 0.030 | 0.963 | 11.41 | 280/437/480 | **OVERLOADED** |
| 56 | 38.62 | 1611/3306 | 3.564 | 0.819 | 0.143 | 0.005 | 0.033 | 0.962 | 13.77 | 360/516/620 | OVERLOADED |
| 64 | 40.21 | 2335/4926 | 4.824 | 0.818 | 0.146 | 0.002 | 0.034 | 0.964 | 16.76 | 406/586/704 | OVERLOADED |

**FACT, the serving result reproduces F18.** C_safe = 32, knee at 36, overload
at 48, ceiling ~40 audio-s/wall-s, zero lost established streams and zero
rejections at every rung. Throughput per rung agrees within about 1 %. The
instrumentation did not move the answer, which is what makes this a usable
BEFORE.

**FACT, and it closes a line of investigation: `runnable_idle` falls
monotonically to 0.002.** 0.236 at C = 8, 0.030 at C = 32, 0.014 at C = 40,
**0.002 at C = 64**. It does not bottom out and rise again — that shape was
entirely the accounting defect of F21. Model execution runs the other way:
0.276 at C = 8 to **0.964 at C = 64**.

**Therefore this box is compute-bound at the knee, not scheduling-bound.** At
C = 40 the fleet executes the model 94 % of scheduler wall; the whole
non-model remainder is 6 %, and only 1.4 points of it is avoidable. **No
scheduling change can move the knee.** The only lever left is making the model's
work cheaper per second of audio.

**FACT: the ready-to-execution delay grows smoothly and does not change regime
at the knee.** p95 reads 84 / 114 / 173 / 277 / 437 ms at C = 28/32/36/40/48 —
accelerating, but with no discontinuity anywhere near C = 32-36. It is a
queueing curve approaching saturation, so the delay is a CONSEQUENCE of the
knee and not its cause. At C = 40 it is 277 ms of a 320 ms budget, against an
emission lag p95 of 667 ms: queueing is about 40 % of what a client sees.

**FACT: the readiness predicate's error is one-sided at every rung.**
`false_ready` is 0-10 per rung against `false_not_ready` 200-1326, on 2528-13729
runnable slots. So the measured `runnable_idle` is a floor, never a ceiling —
and since the floor is now 0.002, the conclusion above only gets stronger.

**CAVEAT, stated rather than explained away**: this ladder passed `--seed 42`
and F18 did not, so the two drew different clip schedules. At equal audio per
wall second this run completed more utterances (385 vs 329 at C = 32), i.e.
shorter ones, i.e. more finalizations — and its lag p95 is correspondingly worse
at the high rungs (667 vs 601 at C = 40). That difference is NOT attributed to
the instrumentation, because nothing here measures the instrumentation's cost.
It is the reason both sides of the BEFORE/AFTER must be this build.

**DECISION**: this table, not F18, is the baseline the optimisation must move.

---

## F23 — The cadence law for 3x8, fitted and validated: it predicts the measured knee

**EVIDENCE**, Axion, commit `a48d443`, the F22 ladder's own dumps. Every worker
of every rung reports the mean ready set it stepped over and the mean wall that
step took, so each (worker, rung) is one point of `T_step(B) = a + b·B`. The
three workers of a rung sit at different B — they are independent domains fed by
independent clients — so the fit is not purely a restatement of load.

**FACT**: over 36 points, `T_step(B) = 14.52 + 19.00·B` ms, **R² = 0.9994**.

Per-row cost falls from 30.6 ms at B = 1.0 to 19.7 ms at B = 17, exactly as a
fixed cost amortised over a widening set.

**VALIDATION — the law was not tuned to the answer.** Put the fitted constants
into the cadence budget `N·(a/B + b + finalize) ≤ P` with `P = 320 ms` and the
mean batch each rung actually ran:

| at the batch of | a/B | b | finalize | per stream per period | predicted N |
|---|---|---|---|---|---|
| C = 32 (B = 2.17) | 6.69 | 19.00 | 3.86 | 29.55 ms | **32 streams** |
| C = 36 (B = 3.26) | 4.45 | 19.00 | 3.86 | 27.31 ms | **35 streams** |
| C = 40 (B = 4.77) | 3.05 | 19.00 | 3.86 | 25.90 ms | **37 streams** |

Measured: C_safe **32**, knee **36**. The law lands on both. This is the first
capacity prediction in this repo that matches the measurement without being
fitted to it.

**FACT — where the cost actually is**, at the knee's B = 3.26:

| component | ms per stream per period | share |
|---|---|---|
| **b — the model itself, 8 threads** | **19.00** | **70 %** |
| a/B — fixed step cost, amortised | 4.45 | 16 % |
| finalization tail, alone at B = 1 | 3.86 | 14 % |

**This ranks the levers, and it retires a hope.** Widening the ready set can
return at most the 16 %, and only by waiting — 40 ms of collection window is
12.5 % of the 320 ms budget, so the window roughly pays for itself and no more.
That is the same conclusion the pre-F15 window A/B reached by measurement, and
the fitted law now explains *why* it was right even though the run that produced
it was contaminated.

**PREDICTION, recorded before the change is written** (and it confirms F20's,
which was a hand-wave; this one has constants):

| finalization | per stream per period | knee moves to |
|---|---|---|
| as today, alone at B = 1 | 27.31 ms | 35 streams |
| **batched at the marginal cost** | **24.56 ms** | **39 streams** |
| free (an upper bound nothing can beat) | 23.45 ms | 41 streams |

So **batching the tail is predicted to move the knee from ~35 to ~39, and to
leave C = 40 just outside.** Making C = 40 healthy needs a second lever as well.
If the measurement lands far from 39, the law is wrong and that is worth more
than the optimisation.

**OPEN, and now the largest single item on this box**: `b` is 70 % of the cost
and it is pure model compute on 8 threads. This profile's own hardware section
records `bf16`, `sve` and `sve2` as present on this Neoverse-V2 and **idle** —
the binary issues none of them. A 20 % reduction in `b` is worth as much as
batching finalization, and nothing here has measured whether it is available.
That is a kernel question, not a serving question, and it is recorded so it
stops being invisible.

**NOT TRANSFERABLE**: a and b are functions of how many cores walk the weights.
These are 8-thread constants. The 1x24 constants in the profile (`a = 39.5`,
`b = 8.57`) are a different function and the two must never be mixed; that 3
workers at `b = 19.0` beat 1 worker at `b = 8.57` by about 35 % in rows per ms
is the same fact the domain study reported as 0.826 against 0.537 audio-s per
cpu.

**METHOD NOTE, because the first attempt at this fit was wrong.** `rows_stacked
/ steps` is NOT the mean ready set: it is the encoder's own row counter and
counts rows per internal call, reading 23 rows per step at C = 40 where the
ready set was 5.8. Fitting on it gave `b = 4.75` — a law four times too flat,
which would have predicted about 100 concurrent streams on a box that knees at
36. The field to use is `ready_mean`, and both it and the step mean must be
multiplied back by their step counts before being differenced, because a mean
cannot be subtracted.

---

## F24 — Cost by position: no growth with history, and the long-clip benefit is finalization, not the step

**WHY THIS RUN EXISTS.** The 7 s / 24 s / 94 s probe varies two things at once —
how often the fixed finalization is paid, and how far a stream gets from its own
start. If later steps of a long stream got progressively more expensive (the
failure mode the sibling TTS engine had, where a growing cache makes every step
dearer), then "long clips serve better" would be the opposite of the truth and
optimising finalization would be chasing the wrong sign.

**EVIDENCE**, Axion, commit `d0c36f4`, 3x8, cpus 0-23 / generator 24-31, 120 s
rungs, three corpora at C = 32 and 40. The scheduler charges every batched row to
its step index inside its own utterance and reports rows, the row's fair share
of the step wall, its ready-to-model-start delay and the batch it travelled in.
`mix` is the control for synchronisation: its clips are 4.3/7.4/11.9 s and
desynchronise the fleet by themselves (measured start-time spread p50 43.5 s),
while `s07` and `s24` are equal-length slices of one file and hold it in phase.

**Step cost by position, C = 32, ms per row:**

| step | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mix | 20.9 | 21.8 | 23.6 | 25.7 | 27.8 | 29.1 | 29.9 | 27.9 | 25.9 | 28.5 | 27.1 | 28.2 |
| s24 | 15.9 | 18.9 | 22.9 | 25.3 | 27.9 | 27.9 | 28.9 | 26.6 | 25.9 | 27.3 | 26.9 | 28.6 |
| s07 | 18.5 | 19.9 | 22.5 | 25.6 | 28.7 | 30.0 | 30.5 | 28.4 | 27.0 | 27.0 | 24.8 | — |

**FACT — the duration pathology is excluded.** Cost rises over the first ~12
steps (3.8 s) from about 17 ms to about 29 ms and is then FLAT: to step 36
(11.5 s) in the server data, and to 24 s in the client's own per-position lag
(F-probe: 80/199/100/99/102/103/106/103 at C = 32). It is the bounded left
context filling and then saturating, which is what a cache-aware model is
designed to do. Nothing grows with stream history.

**FACT — the profile is the same in all three corpora**, within 1-2 ms at every
position, at two concurrencies, with contents ranging from near-silence to
continuous speech. It is a property of the stream, not of the corpus, the clip
length, or whether the fleet is in phase.

**FACT — utterance length does not change the mean cost of a row.** Row-weighted
over a whole utterance: **25.99** ms (s07, 7 s), **26.31** (mix, 6.8 s mean),
**26.64** (s24, 24 s) at C = 32; 22.37 / 22.41 / 22.64 at C = 40. Within 2 %.
**So the benefit of long clips is not in the model step at all.**

**FACT, and it kills a hypothesis of mine before it became code.** The shared
relative-position projection was the proposed mechanism: while a stream's left
cache is filling, its K differs from everyone else's and the batch cannot share
the projection, so short utterances — which restart the encoder stream every
time — would pay it privately for their whole life. The share does move exactly
as predicted, **0.316 (s07) -> 0.661 (s24) at C = 32, and 0.440 -> 0.821 at
C = 40** — and the mean step cost does not move with it (25.99 -> 26.64). **The
mechanism is real and buys nothing measurable. It is not a lever.** Also note
the warm-up region is the CHEAP part (16-21 ms against a 27 ms plateau), so the
earlier guess that short utterances suffer by living inside it had the sign
backwards.

**FACT — where the corpora actually differ is QUEUEING.** Mean
ready-to-model-start, same rungs: **28.4** ms (s07), 37.1 (mix), **22.1** (s24)
at C = 32; **82.3** / 106.1 / **42.6** at C = 40. At C = 40 a mixed-corpus row
waits 106 ms of a 320 ms budget; the 24 s corpus waits 43.

**CONCLUSION — the decomposition asked for.** Of the improvement from 7 s to
24 s utterances, essentially **all** of it is finalization amortised. Per-row
model cost is unchanged, rel-pos sharing doubles and returns nothing, and the
only cost that scales with utterance rate is the 82 ms tail — 3.86 ms per stream
per period at a 6.8 s mean utterance against 1.09 ms at 24 s (F23). That
difference reappears as queueing delay, which is what the client sees as
emission lag.

**DECISION**: finalization is confirmed as the target, on evidence rather than
on its size in a profile. F23's prediction stands unchanged and unhedged —
batching the tail moves the knee from about 35 to about 39, and leaves C = 40
just outside.

---

## F25 — C=32 holds for ten minutes with zero losses, and is still NOT qualified

**EVIDENCE**, Axion, commit `d0c36f4`, 3x8, cpus 0-23 server / 24-31 generator,
600 s closed loop at C = 32, mixed corpus, seed 42, one fresh server.

```
utterances 2362/2362 ok, 0 errors, 0 rejected, 136 excluded by warm-up
audio 17776.7 s, 39201 deltas, span 611.8 s, pacing PACED (max lateness 1.5 ms)
[PASS    ] utterances lost                      0    (limit 0)
[PASS    ] emission lag p95 (pooled)          229 ms (limit 320)
[PASS    ] finalization lag p95               390 ms (limit 500)
[PASS    ] backlog max                      0.444 s  (limit 0.640)
[PASS    ] drift / trend, all four series      10 % / 3-7 %  (limits 20 / 15)
[MARGINAL] TTFP p95                          1657 ms (limit 1160)
verdict: MARGINAL
```

Per-window emission lag p95 over the ten minutes: 237, 252, 235, 235, 221, 228,
222, 225, 224, 222 ms. Flat.

**FACT**: at C = 32 this fleet sustained **30.71 audio-s per wall second for ten
minutes with zero lost established streams and zero rejections**, every serving
envelope inside its limit and no drift.

**FACT — the strongest confirmation yet that the box is compute-bound.** Server
accounting over the same window: step 0.852, finalize 0.113, **model 0.965**,
runnable_idle **0.015**, no_work 0.020, mean batch 2.25, ready-to-model-start
37/104/147 ms. Over a long run, with no ramp in the window, avoidable idle is
**1.5 % of scheduler wall**.

**NOT PROMOTED to `long_soak_qualified`, for two reasons, neither of them
softened.**

**1. Transcripts were not checked.** `quality: 0 utterance(s) scored against a
reference, 2362 without one`. The profile's own condition is "zero lost streams
**and transcripts checked**", and the reference manifest does not exist in this
repo — `make fetch-stress-bank` has never been run. A soak that proves timing
while proving nothing about what was transcribed is half a qualification.

**2. TTFP fails its limit, and the explanation previously recorded for that is
wrong.** This file and the Axion profile said TTFP was measuring leading
silence. For THIS corpus that is false: `test_en` starts speaking at 0.02 s,
`fleurs_1521` at 0.18 s and `fleurs_1534` at 0.64 s, against a TTFP p95 of
1657 ms. About **1 second of pipeline latency is unexplained** and is being
dismissed by an explanation that only fits `fleurs_long.wav`.

**FACT: the TTFP floor is load-independent.** Across the F22 ladder it reads
1544 ms at C = 8, 1545 at C = 16, 1646 at C = 32, then 1925 at C = 40 and
2827 at C = 64. So it has a floor of about 1545 ms present on an almost idle
fleet, plus a queueing term that only appears past the knee. **The MARGINAL
verdict at C = 32 is therefore not a statement about capacity** — but the floor
is a real, user-visible 1.5 s to first partial and it is NOT explained. Roughly
1.0 s of it is about three chunk periods, which is the shape of an encoder
filling context before the decoder will emit, but that is a guess and is
labelled as one.

**DECISION**: the TTFP gate is NOT relaxed and the limit is NOT moved. Two
things are owed instead — a corpus with a reference manifest so transcripts can
be scored, and a measurement of where the 1545 ms floor is spent. Until both
exist, C = 32 stays `measured_short_run_safe`.

---

## F26 — The step, split by component: attention dominates, but the first reading was confounded

**EVIDENCE**, Axion, commit `4badd67`, 3x8, cpus 0-23 / generator 24-31, 120 s
rungs, mixed corpus, seed 42. Three concurrencies on purpose: the stacked
per-row linears read the weights once for the whole set so their per-row cost
falls as B grows, while anything that loops per stream cannot amortise.

| component | C=16, B=1.07 | C=32, B=2.16 | C=40, B=5.74 |
|---|---|---|---|
| **attn+cache** | **15.188 (54.2 %)** | **10.252 (43.8 %)** | **7.503 (38.4 %)** |
| ffn2 | 3.724 (13.3 %) | 3.278 (14.0 %) | 2.932 (15.0 %) |
| ffn1 | 3.648 (13.0 %) | 3.219 (13.8 %) | 2.892 (14.8 %) |
| relpos (shared) | 0.371 (1.3 %) | 2.239 (9.6 %) | 2.205 (11.3 %) |
| conv | 1.938 (6.9 %) | 1.766 (7.5 %) | 1.620 (8.3 %) |
| qkv | 1.322 (4.7 %) | 1.051 (4.5 %) | 0.935 (4.8 %) |
| subsample | 0.964 (3.4 %) | 0.939 (4.0 %) | 0.915 (4.7 %) |
| o_proj | 0.668 (2.4 %) | 0.463 (2.0 %) | 0.365 (1.9 %) |
| tail | 0.206 (0.7 %) | 0.195 (0.8 %) | 0.179 (0.9 %) |
| **encoder total** | **28.03 ms/row** | **23.40** | **19.55** |

**FACT**: `attn+cache` is the largest component at every batch width, 38-54 % of
the encoder step. At the operating point that matters — C = 32, where the ladder
also measured B = 2.17 — it is **10.25 ms of 23.40 ms per row**, larger than the
finalization tail and the amortised fixed cost put together.

**CROSS-CHECK, and it holds.** The fitted law (F23) says `T_step(B)/B = a/B + b`
= 21.5 ms/row at B = 5.74; the encoder components sum to 19.55. The ~2 ms
difference is the decoder, the emission and the rest of
`mynah_asr_stream_step_batch`, which are outside the encoder call. Two
independent instruments agree to within 10 %.

**RESULT NOT YET ATTRIBUTABLE — the first reading of this table was wrong, and
the table says so itself.** Two columns move in ways a per-stream loop cannot:
`attn+cache` halves as B grows (15.19 -> 7.50) while `relpos` rises sixfold
(0.37 -> 2.21) and then stops amortising. Both are one mechanism. Inside
`stream_attention_core`:

```c
if (rk_in) { relpos_count(MYNAH_ASR_RELPOS_SHARED); }
else       { matmul_wt(pe, L->relk_w, es->sa_rk, P, d, d);   /* PRIVATE */
             relpos_count(MYNAH_ASR_RELPOS_PRIVATE); }
```

When no other stream in the pass shares a stream's K, that stream computes the
projection **itself, inside the attention**. So the work was never shrinking as
the batch widened — it was **migrating from `attn` to `relpos`**. Reading the
fall in `attn` as "attention parallelises with B" would have pointed an
optimisation at the wrong component entirely.

**DECISION**: the projection is now timed where it happens and subtracted from
the attention, so the two names stop trading cost, and the private/shared/group
call counts are reported beside the table. `attn` is then attention, and the
question "how much of the 19 ms/row is the rel-pos projection" becomes a
measurement rather than a subtraction between two runs.

**METHOD NOTE**: the split of `attn` from the k/v cache shift is timed INSIDE
the original loop, not by splitting the loop in two. A stream's attention and
its own cache update run back to back over the same cache lines; splitting the
loop to get a clean timer boundary would have changed the locality of the thing
being measured. `tests/test_stream_batch` stays IDENTICAL OK, so rule 4 holds.

**STILL OPEN, and not to be guessed at**: how much of the marginal per-row cost
`b` is attention arithmetic, how much is moving `left x d` floats of k/v cache
per layer per stream, and how much is the private projection. The next run
answers all three. No optimisation is chosen before it lands.

---

## F27 — The dominant cost is a projection recomputed per stream that depends on neither

**EVIDENCE**, Axion, commit `68ad60b` (gate `tests/test_stream_batch`: IDENTICAL
OK, int8 and f32, B = 1..8), 3x8, 120 s rungs, mixed corpus, seed 42.

| component | C=16, B=1.07 | C=32, B=2.12 | C=40, B=5.61 |
|---|---|---|---|
| **relpos_priv** | **14.383 (50.1 %)** | **9.562 (39.5 %)** | **6.818 (33.4 %)** |
| ffn2 | 3.696 (12.9 %) | 3.268 (13.5 %) | 2.925 (14.3 %) |
| ffn1 | 3.625 (12.6 %) | 3.208 (13.3 %) | 2.884 (14.1 %) |
| relpos (shared) | 0.376 (1.3 %) | 2.187 (9.0 %) | 2.240 (11.0 %) |
| conv | 1.911 (6.7 %) | 1.757 (7.3 %) | 1.615 (7.9 %) |
| **attn** | **1.418 (4.9 %)** | **1.425 (5.9 %)** | **1.429 (7.0 %)** |
| qkv | 1.285 (4.5 %) | 1.034 (4.3 %) | 0.923 (4.5 %) |
| subsample | 0.962 (3.4 %) | 0.944 (3.9 %) | 0.916 (4.5 %) |
| o_proj | 0.664 (2.3 %) | 0.468 (1.9 %) | 0.365 (1.8 %) |
| tail | 0.203 (0.7 %) | 0.192 (0.8 %) | 0.177 (0.9 %) |
| **kv_cache** | **0.161 (0.6 %)** | **0.140 (0.6 %)** | **0.134 (0.7 %)** |
| encoder total | 28.69 ms/row | 24.19 | 20.43 |
| attention cores computing their own projection | 93.6 % | 70.5 % | 55.6 % |

**FACT — the attention is not the cost.** `stream_attention_core`, once the
projection is timed out of it, is **1.42 ms/row and FLAT** across all three batch
widths (1.418 / 1.425 / 1.429). The k/v cache shift, which looked like a
plausible memory-bound suspect, is **0.14 ms/row — six tenths of one percent**.
F26's "attn+cache dominates" was entirely the projection hiding inside it.

**FACT — the dominant cost is the relative-position projection.** At the
operating point C = 32, `relpos_priv` + `relpos` is **11.75 ms of 24.19 ms per
row: 48.6 % of the whole encoder step.** Per projection the cost is ~0.542 ms
private and ~0.805 ms shared (the shared one covers a larger P, since the K most
streams agree on is the saturated one), and at C = 32 a row pays ~17.7 private
projections of the 24 layers it passes through.

**FACT — it is recomputed, not computed.** From `encoder.c`, `rk = pe @ relk_w`
where `pe = mynah_asr_pos_emb(enc, K)`. It depends on **(encoder, layer, K) and on
nothing else** — not on the stream, not on the audio, not on the cache contents.
`pe` is already memoised per stream (`if (pe_K != es->sa_pe_K)`), `relk_w` is a
constant layer weight, and at steady state every stream's left cache is full so
K is the same constant for all of them. The projection is nevertheless rebuilt
from scratch on every step by every stream that does not happen to share it.

**HYPOTHESIS, with its prediction stated before any code is written.** Memoise
`rk` per (layer, K) at the model level rather than sharing it per pass. Sharing
within a pass can only divide the cost by the group size — at B = 2.12 a group of
two halves it, which is why `relpos` does not amortise in the table above.
Memoising across steps removes it entirely at steady state.

- Arithmetic unchanged (same inputs, same deterministic matmul), so rule 4 is
  satisfiable by construction and must still be proven by the gate.
- Memory: `rk` is `[2K-1, d]` floats per layer. At K ≈ 74 and d = 512 that is
  ~300 KiB per layer, **~7 MiB per model** for the one saturated K, shared by
  every stream of the worker. Warm-up Ks need a small bounded set, not one entry
  per stream.
- Predicted effect at C = 32: the encoder step falls from 24.19 to ~12.4 ms/row.
  Through the cadence law (F23) the per-stream cost per period goes from
  ~29.6 ms to ~17.8 ms, moving the predicted knee from **35 to about 54
  streams**.

**That number is deliberately recorded before the change exists.** It is far
larger than anything else measured today — batching finalization was worth 35 ->
39 — and a prediction that large is exactly the kind that turns out to be wrong.
If the ladder does not move, the law or the reasoning above is broken and that
is the finding.

**This also explains F24 without contradicting it.** Doubling the *share* ratio
bought nothing measurable there, and here sharing is confirmed to be the weak
lever: it divides by the group size, and groups are small at the batch widths
this box actually runs. Memoisation is a different mechanism, not a stronger
version of the same one.

**NOT AN ISA QUESTION.** No SVE, SVE2 or bf16 is involved. The largest single
item in this serving path is work that does not need doing at all.
