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
