# `.work/` — task detail notes

`PLAN.md` is the board: one line per work item, with a link to the note here
that holds the detail. Nothing in `PLAN.md` should need more than one line.
`ENGINEERING.md` is the normative method; this folder is where its rules are
applied to one task at a time.

## Rules

1. **One note per work item.** Name it after the item, not after a date:
   `server-prefork.md`, not `2026-09-18-server.md`. A dated name is allowed
   only for a measurement campaign on one host (`axion-first-run-2026-09.md`).
2. **A note is written before the work starts**, not after. It states the
   problem, the evidence, the plan, and the acceptance gate. If it cannot state
   a gate, the item is not ready to be worked on.
3. **A note is falsifiable.** Record what was measured, on what machine, with
   what flags and at what commit. Record rejected ideas and *why* they were
   rejected: a negative result that is not written down gets re-tried.
4. **`PLAN.md` never grows a log.** Session checkpoints, measurements and
   post-mortems live here. Historical measurements that are still true belong in
   `docs/benchmarks.md`; this folder is for work in flight and for the reasoning
   behind decisions.
5. **Close a note, do not delete it.** Mark the status at the top
   (`OPEN` / `IN PROGRESS` / `DONE <date>` / `REJECTED <date>`) and leave it.
6. **Fixed skeleton.** Every task note opens with: Task · Question · Known facts
   · Unknowns · Files/functions inspected · Evidence · Conclusion · Next action.
   A note never becomes a second global plan.
7. **Public-safe.** Notes are tracked. No credentials, no private hostnames or
   addresses, no SSH commands or key paths, no absolute personal paths, no raw
   profiler dumps. Raw evidence lives untracked under `.work/evidence/` and
   `.work/private/` and is only *summarised* here.
8. `python3 tools/check_plan.py` must pass before any plan edit is reported as
   done: every path the board names exists, every task id is unique, and every
   `Task:` line in a note names a task the board has.

## Index

| Note | What it covers |
|---|---|
| [engineering-method.md](engineering-method.md) | The method behind `ENGINEERING.md`: cost model before code, every tool declares a refusal, the completion rule |
| [sibling-wins-and-flags.md](sibling-wins-and-flags.md) | What qwen-tts and mynah-tts measured as wins and losses, their feature-flag machinery, kernel state per ISA, and ten recommendations for this server |
| [serving-v2-design.md](serving-v2-design.md) | **The v2 concurrent streaming server design**, what it borrows from qwen-tts and mynah-tts, what is ASR-specific, and the ideas already falsified there |
| [baseline-streaming-concurrency.md](baseline-streaming-concurrency.md) | S0 — measure the current server under N concurrent streams before changing anything |
| [stream-api-v2.md](stream-api-v2.md) | S1 — library changes the server needs: reset, sink contract, partial/final/EOU exposure, allocation-free chunk, batched stream step |
| [threadpool-and-lane.md](threadpool-and-lane.md) | S1 — lift the mynah-tts pool (spin-then-park, meter, lane redirect); the BLAS ownership decision |
| [server-prefork.md](server-prefork.md) | S2 — prefork parent, core-major pinning, SCM_RIGHTS handoff, one model group per worker set |
| [server-scheduler.md](server-scheduler.md) | S2 — one scheduler thread per worker owns the model; stream slots; admission at the step boundary |
| [server-admission.md](server-admission.md) | S2 — the admission ladder, per-reason refusals, lingering close, timeouts |
| [server-stream-out.md](server-stream-out.md) | S2 — async bounded output writer; backpressure is cancel, never block |
| [ws-protocol-v2.md](ws-protocol-v2.md) | S2 — WebSocket wire protocol: control messages, partial/final/eou frames, sequence numbers |
| [multi-model-serving.md](multi-model-serving.md) | S2 — Nemotron + Parakeet + Canary in one fleet: worker groups, routing by model and language |
| [observability.md](observability.md) | S3 — effective-config banner, dispatch map, `/health` as facts, `/metrics`, SIGUSR1 dump, thread names |
| [bench-harness-streaming.md](bench-harness-streaming.md) | S4 — the streaming load generator and the single definition of every ASR streaming metric; WAVE/SOAK protocol |
| [cpu-kernels-arm-x86.md](cpu-kernels-arm-x86.md) | S5 — batched int8 for the stream step (SMMLA/VNNI), KleidiAI, own sgemm, dispatch report and ISA guard |
| [axion-first-run.md](axion-first-run.md) | Campaign — first build and baseline on the 32-core Neoverse-V2 box |
| [repo-hygiene.md](repo-hygiene.md) | Small integrity items the checkers found |
| [serving-audit-metrics-tests-hotpath.md](serving-audit-metrics-tests-hotpath.md) | Audit of 2026-09-20: what the hot path wastes beyond F27, what the harness measures and misses, quality as a regression system, the three test tiers, what the sibling still teaches |
| `private/archive-2026-07-v1-plan.md`, `private/archive-2026-07-v1-todo.md` | The v1 plan and TODO (Italian, July 2026), local untracked history; superseded by the board |
