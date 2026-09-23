# Server V2 — qualification of the current build as a product

The question this note exists to answer, and which four days of engine
investigation did not:

> If I deploy this on a CPU machine, what concurrency can I promise
> continuously without stalls, dropped established streams, corrupted
> streaming output, or unbounded latency?

`configs/perf/axion-c4a-highcpu32-nemotron-streaming.json` carries
`status: screened` and `long_soak_qualified: null`. It says why in its own
words: the 2026-09-20 C=32 600 s soak passed every serving gate and lost no
stream, and still did not qualify, because **transcripts were never checked** —
"0 utterances scored against a reference, because no manifest exists in this
repo". That hole is now closed, so the run can be redone properly.

**C=16 is the hard product minimum.** It is a target, not an axiom: if the
model genuinely needs more compute than a 16-core box has, the measurement says
so and we report the safe point we measured instead of the one we wanted. On
**32 vCPU** C=16 is expected to qualify with margin, and that is the claim under
test here.

Everything below the line marked REGISTERED was written before the first
measured run of this campaign.

---

## V2-1 — the frozen candidate

| | |
|---|---|
| commit | `067e9f3` (`v0.9.1-167-g067e9f3`) |
| tree | clean, fresh `git clone` over HTTPS, `make clean && make -j32` |
| host | GCP `c4a-highcpu-32`, Arm Neoverse-V2 (Google Axion), 32 vCPU / 32 physical cores, 1 NUMA node, L3 80 MiB, 62 GiB RAM, SMT off |
| model | `nemotron-3.5-asr-streaming-0.6b`, INT8 (`model.int8.safetensors`, pre-quantized) |
| preset | cache-aware streaming, lookahead 3 → chunk period 320 ms |
| server topology | `--prefork 3 --prefork-threads 8 --cap 96`, `--batch-window-ms 0` |
| pinning | server `taskset -c 0-23`; cpus **24-31 reserved for the load generator** |
| metrics | `--metrics-port 9910` |
| env | `MYNAH_ASR_THREADS`, `MYNAH_ASR_CAPS`, `OPENBLAS_NUM_THREADS`, `MYNAH_ASR_POOL_SPIN_US` all ABSENT; `MYNAH_ASR_STACK_SOLO` at its default (1) |

Resolution proven by the binary, not asserted (`--dispatch-map`, **0 rows
UNKNOWN**):

    kernel.int8_rows  neon-smmla      kernel.int8_dot  neon-sdot
    gemm.f32          own             gemm.f32_kernel  neon
    blas=own  simd=neon+dotprod+i8mm

`bf16`, `sve`, `sve2` are present and never issued. Three unspent levers, and
the dispatch map prints them as IDLE so they stay visible.

None of these may change during qualification. A run with any of them different
is a different candidate.

### Corpus

27 committed clips, **3.57 s to 13.92 s, mean 8.7 s** — every `samples/*/` and
`tests/audio/` WAV except the three over 90 s. Mixed durations and mixed
languages, so the fleet is not fed a synchronized equal-length herd.

This is **not** the corpus of the 2026-09-20 ladder, which used three clips
(7.4, 11.9, 4.3 s; mean 6.8 s). Numbers from this campaign are therefore not
directly comparable to that curve, and nothing here is allowed to quote it as a
baseline. Finalization is a per-utterance cost, so a longer mean utterance moves
the knee to the RIGHT of the historical one; that is a corpus difference, not an
improvement.

---

## V2-2 — REGISTERED: what SAFE means, written before the results

Four of these come straight from the profile's own `gates` block and are not
re-derived here. The rest exist because the previous soak had no equivalent and
could not have caught what it was asked to catch.

| # | bound | value | where it comes from |
|---|---|---|---|
| 1 | established streams lost | **0** | absolute. A stream accepted and then dropped is a product failure at any concurrency |
| 2 | emission lag p95 | **≤ 320 ms** | profile gate. `(lookahead + 1) × 80 ms` is the cadence the model itself sets; past it a stream is falling behind the audio |
| 3 | finalization p95 | **≤ 500 ms** | profile gate |
| 4 | backlog max | **≤ 0.64 s** | profile gate, two chunk periods |
| 5 | 503 admission refusals | counted **separately**; expected 0 at C ≤ cap | a 503 before the upgrade is the admission ladder working. It is never a loss and never excuses one |
| 6 | per-window drift | **every** 60 s window must satisfy bound 2 | whole-run percentiles hide a bad minute. If it fails at minute 23, the run failed |
| 7 | monotonic trend | last-third window p95 ≤ **1.5 ×** first-third window p95 | a soak that is slowly getting worse has not qualified, however good its average |
| 8 | server-side stall | **zero** 30 s intervals in which slots were active and the worker's `model_busy_s` did not advance | SIGUSR1 `[DUMP]`, sampled every 30 s. F15 was a freeze of tens of seconds |
| 9 | client-observable stall | **max** emission lag over all established streams ≤ **3000 ms** | the max, not a percentile. ~9 cadence periods. "Eventually recovered" is still a stall |
| 10 | quality parity | every probe transcript under load **byte-identical** to the unloaded C=1 transcript of the same clip on the same binary | rule 4: a transcript never depends on batching, threads or ISA. Any divergence is a server bug until disproven |
| 11 | worker RSS | end ≤ **1.15 ×** post-warm-up | a leak over 30 minutes is a leak |
| 12 | worker deaths / restarts | **0** | a fleet that silently replaces a worker has not held the load |

## V2-2b — REGISTERED 2026-09-23: TTFP, for CAPACITY rungs only

Registered **after** the C=16 qualification ran and deliberately **not applied to
it**: that run was registered against twelve bounds before it started, and
adding a thirteenth once the results are known is the thing the frozen-line rule
exists to forbid. `v2_verdict` prints bounds 13 and 14 as NOT REGISTERED on a
run that carries no unloaded baseline, which is a statement, not a silent pass.

Raw `open -> first partial` is not the server's gate. It sums three unrelated
things: the silence a clip leads with, the speech this checkpoint wants before
it will commit to a word, and the delay the fleet adds. On this corpus the first
two dominate — the C=16 soaks read 2430 ms of raw TTFP and **816 ms median /
1251 ms p95 measured from speech onset**. Gating on the raw figure would fail a
healthy server for the corpus's room tone.

So the server is gated on the **paired** penalty, per clip:

    penalty_i(C) = ttfp_i(loaded) - ttfp_i(unloaded, same clip, same build)

A clip that leads with 700 ms of silence carries those 700 ms on both sides and
vanishes from the difference. Subtracting two p95s would not do this: it would
compare a loaded corpus with an unloaded one and call the gap the server.

| # | bound | GOOD | DEGRADED | BAD |
|---|---|---|---|---|
| 13 | paired TTFP load penalty, p95 | **<= 250 ms** | 250-500 ms | > 500 ms |
| 14 | paired `ready -> first partial` penalty, p95 | **<= one cadence** | — | > `(lookahead+1)x80` ms |

Bound 14 uses `first_delta_lag_ms`: the server's own lateness on the frame whose
audio the model actually consumed for its first partial. It excludes leading
silence and excludes the speech the model wanted, because it is measured against
the audio the server *had*. Losing a whole model cadence there, before a single
word is published, is a concrete queueing signal — and the cadence is the same
`(lookahead+1) x 80 ms` the emission-lag bound already uses, not a new taste.

**Reported, never the server's gate:** `speech -> first partial`, p50/p95, with
GOOD <= 1500 ms, WARN <= 1800, BAD above. These are reasonable for THIS
checkpoint and THIS preset given its observed floor, and they are not claimed as
universal ASR thresholds. This line belongs to the product and the checkpoint;
bounds 13 and 14 belong to the fleet.

A capacity rung is SERVER GOOD only when the twelve original bounds pass **and**
13 and 14 pass.

**TTFP has no gate**, and that is deliberate. The profile refuses to defend a
number for it: there is a load-independent floor near 1545 ms on this build and
roughly a second of it is unexplained. It is reported as a secondary
responsiveness figure and it cannot pass or fail this qualification.

Quality here is a **regression gate, not a research campaign**. Bound 10 asks
whether concurrency alters recognition. It does not ask whether Nemotron is
good: that is settled separately and is not this run's business.

Thresholds are not moved after seeing results. A run that violates one is
reported as violating it.

---

## V2-3 — ladder

`C = 8, 16, 24, 32`, one fresh server per rung, warm-up discarded, extended
around the knee only if the knee turns out to sit between two rungs. The ladder
**screens**; it never promotes. Its only job is to show whether C=16 clears the
envelope with enough room to bother soaking it.

## V2-4 — the qualification itself

**Two independent 1800 s soaks at C=16, a fresh server each time**, analysed in
60 s windows against bounds 1-12, with the worst window reported and not only
the pooled percentiles. Higher operating points are qualified after C=16, never
instead of it.

## V2-5 — liveness

Periodic SIGUSR1 dumps throughout every soak, so bound 8 has evidence rather
than an absence of complaints, and `tools/bench/stall_timeline.py` has something
to classify if bound 9 trips.

## V2-6 — load correctness

The unloaded C=1 pass over the corpus produces the reference transcripts, from
the **server**, not from the CLI — the question is whether the server's own
scheduling changes its own output. Those clips are then replayed throughout the
soak and compared byte for byte (bound 10).

---

# RESULT — 2026-09-22, GCP c4a-highcpu-32 (Axion Neoverse-V2)

Commit `067e9f3`, clean HTTPS clone, `make clean && make -j32`, `make test` RC=0
with `streaming parity: IDENTICAL OK`, dispatch 0 rows UNKNOWN
(`neon-smmla` / `neon-sdot` / sgemm `own`), 3 workers x 8 threads pinned inside
cpus 0-23 with every mask proven disjoint, generator alone on 24-31.

## V2-4 — two independent 1800 s soaks at C=16: QUALIFIED

| bound | soak 1 | soak 2 | limit |
|---|---|---|---|
| 1 established streams lost | **0** | **0** | 0 |
| 2 emission lag p95 | **65 ms** | **66 ms** | 320 |
| 3 finalization p95 | 99 ms | 98 ms | 500 |
| 4 backlog max | 0.184 s | 0.184 s | 0.640 |
| 5 503 refusals | 0 | 0 | counted apart |
| 6 worst 60 s window | 71 ms (w13) | 71 ms (w13) | 320 |
| 7 trend, last third vs first | **+1.9 %** | **-0.6 %** | +50 % |
| 8 server-side stall | 0 of 177 intervals | 0 of 177 | 0 |
| 9 max emission lag | 233 ms | 218 ms | 3000 |
| 10 transcript parity | **PASS** | **PASS** | byte-identical |
| 11 worker RSS growth | **1.007x** | 1.009x | 1.15x |
| 12 worker deaths | 0 of 60 samples | 0 of 60 | 0 |

7738 utterances, 56 488 audio-seconds — 15.7 hours of speech. **Zero stalls at
every threshold over 125 948 published deltas**; the latest delta of the whole
campaign arrived 233 ms behind its audio against a 320 ms cadence.
`audio/wall` 15.63x and 15.61x: the fleet carried sixteen real-time streams.
Fairness 1.23x and 1.27x worst-stream-over-median. TTFP p95 2426/2430 ms,
REPORTED and ungated. Peak 6 active slots per worker against a ceiling of 32,
so neither run was throttled by its own connection limit.

The two runs used different seeds on fresh fleets and agree to 1 ms of p95.

**This closes the hole that kept the profile at `screened` since 2026-09-20**:
transcripts were checked, 27 clips, every one identical across streams and
identical to the unloaded C=1 reference taken from the server itself.

## V2-3 — ladder, 90 s per rung, fresh fleet each

| C | emission lag p95 | finalization p95 | backlog max | max lag | stalls | parity | peak slots/worker |
|---|---|---|---|---|---|---|---|
| 8 | 55 ms | 87 ms | 0.104 s | 113 ms | 0 | PASS | 3 |
| 16 | 71 ms | ~100 ms | 0.184 s | 165 ms | 0 | PASS | 6 |
| 24 | 73 ms | 114 ms | 0.184 s | 202 ms | 0 | PASS | **8 (at the ceiling)** |
| 32 | 73 ms | 119 ms | 0.184 s | 213 ms | 0 | PASS | **8 (INVALID)** |

C=24 and C=32 are **not** capacity measurements. `--threads` is the HTTP pool and
a WebSocket stream holds one of its threads for its whole life, so the fleet had
3 x 8 = 24 connections while `--cap` advertised 96 slots. Both rungs pinned at 8
active slots per worker; C=32 ran 24 streams, produced FEWER utterances than
C=24 (253 against 270) and lower throughput (18.75 against 19.13 audio/wall),
with no error and no 503 — a stream that never connects is neither. The upward
ladder with the fix (`--http-threads 32`, ceiling 96) had started when the box
became unreachable.

## Why the box looks idle at C=16, with the arithmetic

From the worker's own `/v1/health` during soak 1:

    batched_steps_total 7670   ready_mean 1.031   step_wall_ms mean 15.85
    by_b   B=1: 7479 steps 15.46 ms   B=2: 154 / 28.08   B=3: 26 / 40.25   B=4: 11 / 49.70
    audio_seconds / steps = 2570 / 8246 = 0.3117 s  -> one step advances one 320 ms chunk

One stream costs **15.85 ms of compute per 320 ms of audio = 4.95 % of a worker**.
16 streams over 3 workers is 5.33 each, so **26.4 % duty** — which is what the
SIGUSR1 dump reports (`model_duty` 0.277 at 240 s, 0.281 at 961 s, flat) and what
`htop` shows. The fleet is not slow at C=16; C=16 asks it for a quarter of itself.
A worker saturates near 320/15.85 ~ 20 streams.

Two open items fell out of this and are NOT conclusions:

* **97.5 % of steps run at batch 1** (`ready_mean` 1.03). With
  `--batch-window-ms 0` a step fires the instant one stream is ready, so a ready
  set never accumulates. The by-B table says batching pays: 15.46 ms/stream at
  B=1 against 12.43 at B=4, **-20 %**. The profile records that a 40 ms window did
  not move the ceiling, but that was measured before the F15 stall fix and on a
  different corpus. Worth re-testing.
* **`runnable_idle` 0.223, of which 99.9 % in `park`**, with `ready_sel`
  0.1/20.0/50.0 ms. This is NOT yet called waste: this profile has already been
  wrong about this exact figure once, by charging the finalization tail to it,
  and the correction left only 1-3 % genuinely avoidable. It needs the same
  recomputation from raw dumps first.

The 50 % `no_work` is not a defect and is not recoverable: streams arrive in real
time and the server cannot encode audio nobody has spoken yet.

## Evidence

`gabrielemastrapasqua@34.136.69.91:~/asr-evidence/v2/`
`20260922T162708Z` (freeze, reference, first ladder) and
`20260922T163950Z` (the two C=16 soaks, with `server-soak*.log`,
`procsample-soak*.txt`, `masks-soak*.txt`, `onsets.json`, `bank.txt`).

**NOT ARCHIVED.** The instance became unreachable — no ICMP, port 22 filtered —
before the copy. The numbers above are the verdict tool's own output, recorded
verbatim; the per-utterance JSON behind them is on that disk. If the instance
was stopped the disk survives and the evidence is one `scp` away; if it was
deleted the raw records are gone and only this table remains.

**The profile therefore stays `screened`.** `tools/bench/v2_promote.py` derives a
promotion from artefacts and refuses without them, and hand-writing the block it
would have written is exactly what that tool exists to prevent. The measurement
passed; the archive is what is missing, and it is one command once the box
returns.
