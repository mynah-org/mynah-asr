# The Axion session, day two: the first ladder measured nothing, and why

Status: OPEN (the fix and the two structural findings are done; the numbers are
the run in progress)

Task: S4-3, S5-2, M-4
Question: what concurrency does the streaming server hold on this box — and for
which models is that question even defined?

## 1. Yesterday's ladder produced no data at all

`~/wave.log` on the box, five rungs, every one of them:

```
  utterances 0/40 ok, 40 errors, 0 rejected
  [INVALID ] no utterance completed
  verdict: INVALID
  error stream 0 rep 0: connect: [Errno 111] Connection refused
```

`~/wave-server.log` holds the answer and it is one line long — the server's
usage text. The launcher passed `--lookahead 3`, which **is not a server flag**:
lookahead is a per-group suffix (`--model name=dir:lookahead=L`) or a per-request
query parameter (`?lookahead=3`), because it is a property of the stream and not
of the process. The server printed usage, exited, and the ladder then ran for
0.1 s per rung against a closed port and wrote five well-formed JSON artefacts
containing nothing.

Two lessons, and the second is the one that matters:

- A wrong flag was cheap. **A ladder with no readiness probe was not.** Five
  rungs, five verdicts, five artefacts, and no server. The output was shaped
  exactly like a result.
- `sleep 25` is not a readiness check. `tools/bench/box_session.sh` now polls
  `/v1/health` until it answers, refuses with the server's own log when the
  process died, and never starts a rung against a port that is not serving.

## 2. Parakeet does not stream here, and that is a config fact, not an opinion

`mynah_asr_stream_unsupported()` refuses the converted
`parakeet-tdt_ctc-110m` pack on three independent counts, each read from the
loaded weights or the pack config:

| what the pack has | what the streaming step requires |
|---|---|
| `use_bias: true` (linear biases) | the streaming step adds no biases |
| `conv_norm: batch_norm` | the streaming step applies layer_norm |
| `att_context_style: regular` | causal, asymmetric padding |

```
$ ./mynah-asr stream -m models_local/parakeet-tdt_ctc-110m-gguf -i tests/audio/test_en.wav
mynah-asr: this model is offline-only (no cache-aware streaming)
```

So **"what concurrency does Parakeet hold on the streaming server" has no
answer**: the server refuses the WebSocket upgrade with a named reason
(`server/sched.c` asks the same function before the upgrade). This is not a gap
to be closed by tuning — it is the same conclusion `.work/where-to-attack.md` §2
reached from the arithmetic, arriving from the other direction: Nemotron is the
streaming product, Parakeet is the throughput product.

The question that IS defined for Parakeet is offline throughput, and it is
answered on `POST /v1/audio/transcriptions`. `tools/bench/rest_load.py` (new)
climbs a concurrency ladder there and reports two numbers that must not be
confused: **xRT** (audio seconds accepted per wall second — the fleet's
throughput, what pays for hardware) and **per-request latency** (what a caller
waits). Throughput flattens at the knee while latency keeps climbing; the tool
names the knee and refuses to choose it for you.

First signal, development Mac, 4 threads, 2 clips:

```
     C    ok  ref  err  wall_s     xRT   p50_ms   p95_ms
     1     2    0    0    0.17   69.53     59.3    109.4
     2     4    0    0    0.28   83.74    142.8    163.6
     4     8    0    0    0.49   95.48    248.2    266.1
```

and `transcribe` on that pack runs at **RTF 0.024** against Nemotron int8's
0.682 on the same host — 28x cheaper per second of audio, which is exactly what
5.65x fewer encoder MACs plus no re-encoding of context buys when there is no
cadence to hold.

## 3. What the session runs, in order

`tools/bench/box_session.sh`, one command, budgeted:

1. **P0 preflight** — uname, nproc, loadavg, who else is logged in. Refuses at
   loadavg >= 2.0, the same threshold `box_qualify.sh` and `box_doctor.sh` use.
2. **P1 doctor** — read-only description of the machine.
3. **P2 dispatch map** — ENGINEERING.md §5: a benchmark is invalid until
   dispatch is proven.
4. **P3 calibration** — the box's own step table, fitted to `a` and `b`, and the
   prediction `B_max = (rho*P - a)/b` that follows from them.
5. **P4 WAVE ladder** — climbs until the envelope breaks, stops there, and
   reports the last rung that held.
6. **P5 SOAK** at that rung, with a metrics scrape at the end.
7. **P6 REST ladder** for the offline-only model.

The cpu split is mandatory and stated in the result: the generator is one
process per stream, so at C=100 it is 100 processes, and a capacity number taken
with the generator sharing the server's cores is not a capacity number. Server
on one slice, generator on the rest, and what is reported is the capacity OF
THAT SLICE — never scaled up to the whole box.

Evidence: `~/wave.log`, `~/wave-server.log` on the box (the invalid ladder);
`tools/bench/box_session.sh`, `tools/bench/rest_load.py`.
Conclusion: (pending the run)
Next action: run it, then close S8-4 by writing the measured capacity next to
the predicted one.

---

## 4. The ladder, measured: c=16 on 24 cores, and c=20 breaks three times out of three

One server, cpus 0-23, generator on 24-31, int8, lookahead 3, `--cap 64`,
counters sampled every 2 s through each rung:

```
  c=8   lag 176  ttfp 1660  fin 378   MARGINAL        16/16   steps+387  flat 0s
  c=12  lag 223  ttfp 1691  fin 387   MARGINAL        24/24   steps+608  flat 0s
  c=16  lag 296  ttfp 1719  fin 707   MARGINAL        32/32   steps+786  flat 0s
  c=20  lag 500  ttfp 1756  fin 887   NOT STREAMABLE  40/40   steps+1010 flat 0s
  confirm c=20 on a clean server: lag 529, then lag 688 — NOT STREAMABLE both times
```

Three things to read out of that table beyond the headline.

- **It is saturation, not the anomaly.** `steps_total` never has a flat stretch,
  and at c=20 every utterance still completes (40/40) — they are late, not lost.
- **TTFP is flat across the whole ladder** (1660 → 1756 ms) while emission lag
  triples. A load-induced latency grows with load; this one does not, so TTFP
  here is a property of the CORPUS and the model's own latency, not of capacity.
  The clips' speech starts at 0.18 s, 0.64 s and 0.02 s, and `samples/en/fleurs_long.wav`
  does not start until **23.2 s** — a TTFP gate measured from stream open is
  measuring the leading silence. It must be measured from speech onset, or the
  soak corpus must be trimmed; otherwise every FLEURS clip fails it.
- **Predicted 45, measured 16.** That is the S8-4 number, and the rest of this
  section is where the missing 2.8x went.

## 5. Where the factor of three goes: the ready set is 2 to 4 streams, not 8 to 16

The capacity law `T_step(B) = a + b*B <= rho*P` prices ONE step serving B
streams. Its whole economy is that `a = 30.8 ms` is paid once per step and
amortised over B. The server exports what B actually was —
`batch_ready_size_sum / batched_steps_total` — and it says:

| streams in flight | mean ready-set B | batched steps per 320 ms period | fixed cost per period |
|---|---|---|---|
| 8 | **2.11** | 3.8 | **117 ms** (vs 31 if B=8) |
| 16 | **3.73** | 4.3 | **132 ms** (vs 31 if B=16) |

So at c=16 the worker spends 132 ms of every 320 ms period re-walking the
weights, four times over, because the ready set never fills. That is the 2.8x,
almost exactly, and it is arithmetic rather than inference.

The mechanism is in `sched_main` (`server/sched.c` §3): the loop parks on a
condvar and steps **as soon as anything is ready**. Clients send 100 ms frames
independently, so at any wake only the one or two slots that have just crossed a
whole chunk are staged; the rest are a few tens of milliseconds short and get
their own step moments later. Nothing is wrong with any individual step — the
batching is bit-exact and `MYNAH_ASR_STREAM_BATCH_MAX` is 256, so nothing is
clipping the set. The set is simply small when it is looked at.

Note what this does NOT say: it is not pool contention, not the kernels, not
SMMLA, and not the f32 remainder. Every one of those was a candidate this
morning and the counter settles it without another run.

The fix has a shape and a constraint. A bounded collection window — when the
first slot becomes ready, wait up to W ms (or until N are ready) before
stepping — is the standard continuous-batching answer, and the slack exists at
low concurrency where it costs nothing. The constraint is repo rule 5: **load
never enters the chunk size.** A collection window does not change the chunk; it
changes when the step runs, and it must be bounded so the emission-lag envelope
still holds at the top of the ladder. What W should be, and whether the sibling
engines already answer this, is the next question rather than a guess.

Evidence: `~/ladder2/` (the ladder and its per-rung metrics), `~/sweep/m-*.txt`
(the batch counters quoted above), `~/repro8/` (c=8 four times).
Conclusion: the streaming ceiling on 24 Neoverse-V2 cores is **c=16 today**, and
the gap to the predicted 45 is a scheduling gap, not a compute gap.
Next action: the W x T sweep (running), then a collection window with a measured W.
