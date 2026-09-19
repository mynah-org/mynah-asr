# Fleet observability and the road to C100

Status: OPEN (written 2026-09-19 on the dev host, from a read of this server's
metrics/health surface and of qwen-tts's; no measurement here)

Task: S6-1 … S6-6, and the C100 preconditions
Question: this server already proves what it runs. What it does NOT do is answer
"how is the FLEET doing" from one place, and it exports nothing that would catch
"it got faster and worse". qwen-tts answers both; what exactly is missing here,
and what does 100 concurrent streams on a 32-core Axion need before it is worth
booking box time?

## 1. What exists here already

`/metrics` on its own port, OFF unless `--metrics-port`, bound to 127.0.0.1, no
`SO_REUSEPORT` (a second bind must fail loudly), token bucket 5/s burst 10 that
refuses BEFORE rendering, no collector thread — the page is rendered on the
accept loop that owns the listener, with bounded poll and linger budgets.
`/v1/health` as facts. `[SERVER-CONFIG]`, per-worker `[TOPOLOGY]` with the mask
read back from the kernel, `SIGUSR1` `[DUMP]`.

The counters are the right ones for an ASR: `audio_seconds_total` as the
denominator of every RTF claim, `emission_lag_ms_{sum,count,max}` plus
`emission_lag_over_ms_total` as EXACT threshold counters rather than estimated
quantiles, `batched_steps_total` / `rows_stacked_total` / `batch_ready_size_sum`,
`step_wall_ms_*`, `cancelled_total` by reason, `refused_total` by reason.

The doctrine is already the same as the sibling's and is written into the page
itself: client-side latency (TTFP, stall rate) is NOT exported, because it is
measured at the far end of a socket this process does not own and belongs to the
harness.

## 2. The gap: nothing answers for the fleet

In prefork there are two views and they do not meet.

- **The router** exports routing facts only — `worker_up`, `worker_inflight`,
  `worker_slots`, `worker_assigned_total`, `worker_completed_total` — and says
  so in the page: *"This process routes connections and never enters the model,
  so the scheduler counters … do not exist here and are NOT exported."*
- **The scheduler counters** live inside each worker process and are reachable
  only by giving each worker its own `--metrics-port`.

So with N workers there is no single scrape target that says how the fleet is
serving. And `/v1/health` is served by the worker, so a probe lands on whichever
worker the router picked: a per-worker health wearing the server's name.

**qwen-tts solved exactly this and the mechanism is worth copying verbatim**
(`qwen_tts_server.c`): a `MAP_SHARED|MAP_ANONYMOUS` mmap of one fixed slot per
worker, allocated BEFORE the fork loop so children inherit it; each child binds
its own slot; writes are relaxed `atomic_fetch_add` on a single-writer slot; the
parent renders every slot with a `worker="i"` label. Two properties make it
right rather than merely convenient:

1. **Nothing is ever summed.** Their comment says it best: *"Per-worker series
   are emitted and never summed. A reader can always add them up; it cannot take
   an average apart again."* That is the same rule this page already states —
   the difference is that here it is used to justify not exporting them at all.
2. **No IPC on the hot path.** A worker's write is one relaxed add to a
   cache line it owns; the parent reads whenever a scrape arrives.

Note their health endpoint has the SAME per-worker limitation and does not hide
it: it carries `worker` and a literal `counters_scope` string ("this worker
only; the limits above are the server's"). That disclosure costs nothing and
this server does not have it.

## 3. What is missing, ordered by value

### S6-1 Fleet metrics through a shared slot table
One `mmap` before the fork, one slot per worker, single writer, relaxed adds;
the router renders every worker's scheduler counters with a `worker` label
beside the routing ones it already has. One scrape target, per-worker series
preserved, no summing, no IPC in the step. The slot struct is a fixed table of
`_Atomic unsigned long long`, so its layout is a compile-time fact and a worker
built from a different commit cannot be silently misread.

### S6-2 Fleet `/v1/health` on the router, and scope disclosure on the worker's
The router knows `worker_up`, `inflight`, `slots` and, with S6-1, every worker's
counters: it can answer for the fleet without entering the model. The worker's
own `/v1/health` gains `worker` and `counters_scope`. Readiness and liveness
stay one endpoint — the sibling's reasoning applies unchanged: only a server
that HAS a scheduler can be unavailable for want of one, and a 503 from a
healthy single-worker server is how a load balancer takes a good host out of
rotation.

### S6-3 `configs/observability/`
`prometheus.yml`, `grafana-dashboard.json`, `alerts.yml`, a README and an
`observability_up.sh` that provisions both on loopback. The panels transfer
almost one-for-one once the vocabulary is translated:

| qwen-tts panel | ASR equivalent |
|---|---|
| Realtime streams sustained | `rate(audio_seconds_total[1m])` — unchanged, same meaning |
| Time to first audio | emission lag mean = `rate(lag_sum)/rate(lag_count)` |
| First audio past 1s | `rate(emission_lag_over_ms_total{threshold="…"})` |
| Chunks arriving behind realtime | **backlog** (S6-4) — the ASR analogue |
| How requests end | `rate(cancelled_total)` by reason |
| Saturation | `worker_inflight / worker_slots` |
| Imbalance (max−min inflight) | unchanged |

Alerts follow the same rule they state: every one fires on something a LISTENER
would notice, and every threshold is an exact count, never an estimated
quantile.

### S6-4 The one counter this server is missing: backlog
qwen-tts exports `stream_gap_behind_realtime_total` — chunks that arrived later
than real time — and refuses to export a stall rate, because a stall is an event
in a player. The ASR mirror image is **backlog**: audio received and not yet
encoded. It is defined in the harness (`§6` of the serving design) and is the
one server-side number that says "this worker is falling behind" before any
client notices. Exact threshold counters, not a gauge: `backlog_over_chunks_total`.

### S6-5 Quality, as opposed to speed
Today this server can say how fast it is and nothing about whether it is still
RIGHT. The sibling's answer is three layers, and each has an ASR analogue that
costs little because the gates already exist offline:

- **Identity at runtime.** `tests/test_stream_batch.c` proves the batched step
  is byte-identical to B single steps. There is no counter that says the
  property held in THIS process on THIS run. A cheap version: the harness's
  batched-identity phase already compares transcripts against the CLI — make
  the load harness refuse to print a KPI table when that comparison fails, the
  way `serve_identity_gate.py` does ("the gate that must pass BEFORE any KPI
  table is printed"). A performance number from a run whose transcripts moved
  is worse than no number.
- **CER drift against a fixed bank.** `samples/` carries committed FLEURS clips
  and the CER gate exists. Nothing tracks it across commits, so "faster and
  worse" is currently invisible.
- **Drift inside the soak.** `stream_load.py` has a drift gate; the sibling's
  has NAMED thresholds (TTFA p50/p95 ≤30 %, stream-RTF p50 ≤20 %, plus a
  text-mix distance so a changing input mix is not read as latency drift) and a
  resource-growth verdict (anon +10 %, PSS +10 %, RSS +15 %, threads +2, fds
  +4). Ours should say its numbers out loud the same way.

### S6-6 `tools/metrics_watch.py`
Scrape the page while load runs and assert three properties: **monotonic** (no
`_total` went backwards), **conserved** (`completed ≤ assigned`), **balanced**
(every live worker received something). Model-free, catches a wedged worker and
a mis-rendered page, and it is the only test that exercises the metrics port
under load.

## 4. C100 — what 100 concurrent streams need before box time

The cadence law is `T_step(B) = a + b·B ≤ ρ·P`, `P = 320 ms` at lookahead 3.
Capacity is `floor(CPUS / T) × B_max(T)` and `T*` is unknown, so C100 is not a
number yet — it is the output of the WAVE sweep in `.work/box-day-plan.md`
step 5. The M1 development signal (Accelerate, before this week's kernel work)
was `a ≈ 37.8 ms`, `b ≈ 20.3 ms` → `B_max ≈ 10` per worker, i.e. ~10 workers for
100 streams, ~3 cpus each on a 32-core box. That is precisely the regime where
`a` and `b` get worse, and it is also the regime where SMMLA — compiled,
validated, never executed — could move `b` the other way.

Four things must be true before the sweep is worth running, and three of them
are code:

1. **`listen(srv, 64)`** (`server/main.c`). 100 clients connecting together
   overflow the backlog and the kernel drops SYNs; the run would measure the
   accept queue, not the server. The backlog belongs in the admission banner
   with the rest of the ladder, where it can be read rather than inferred.
2. **`RLIMIT_NOFILE` is whatever was inherited.** 100 streams plus their writer
   descriptors plus the metrics port against a default 256 on some systems is a
   silent ceiling. Raise it explicitly at start-up and print what it became.
3. **The load generator is one PROCESS per stream.** 100 Python processes on the
   box under test steal cores from the server and invalidate the measurement.
   The sibling measured this on THE SAME MACHINE: an unpinned load generator
   cost worker 0 about 8 % of its first-audio latency on a 32-core Axion. So
   either a second host generates the load, or the generator is pinned to a
   reserved slice and the reserved cores are excluded from the server's mask —
   and the harness must record which, because a C100 number taken with the
   generator sharing the server's cores is not a C100 number.
4. **Memory is not a constraint and should be said so once**: per stream the KV
   cache is `24 × 56 × 1024 × 4 × 2 ≈ 11 MB`, the conv cache 0.8 MB, the decoder
   scratch `32 × (640 + 13088) × 4 ≈ 1.7 MB`, plus the PCM ring — about 14 MB,
   so ~1.4 GB at C100 on top of the COW-shared weights.

## 5. What this is NOT

None of S6 changes a transcript, and none of it belongs in the step. The metrics
page is an OPERATIONS signal; acceptance percentiles keep coming from the
client-side harness over raw samples. That line is already in this server's page
and in the sibling's docs, and it is what stops a dashboard from quietly
becoming the gate.

Evidence: (none — analysis only)
Conclusion: (pending)
Next action: S6-1, because everything else in this note either renders it or
checks it.
