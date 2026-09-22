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
