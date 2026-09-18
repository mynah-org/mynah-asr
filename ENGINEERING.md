# Engineering rules

Normative for every coding agent and every human working in this repository.
`CLAUDE.md` and `AGENTS.md` only point here. Keep this file under ~150 lines.

The principle: the agent must not remember the project; the repository must
make forgetting impossible. `PLAN.md` says what remains, this file says how we
work, `.work/` notes hold the detail of one task, and **the program itself
proves which path it is running**. The reasoning behind these rules, with the
measurements that produced them, is in `.work/engineering-method.md`.

## 1. Work from a small board

Before non-trivial work read `PLAN.md`. It is a board, not a diary.

- One line per item with an id (`S2-3`, `H-1`), linking its `.work/` note.
  One item = one independently verifiable outcome.
- Mark `[x]` only after the gate in the note passed. Never rewrite history in
  the board; close notes, do not delete them.
- Load a note only when working on its item. `python3 tools/check_plan.py`
  passes before any board edit is reported complete: every path exists, every
  id is unique, every `Task:` in a note names a board item. Never create a
  placeholder link or an empty note for work not yet written.

## 2. Separate plan from evidence

A `.work/` note opens with: Task · Question · Known facts · Unknowns ·
Files/functions inspected · Evidence · Conclusion · Next action, and states an
acceptance gate before the work starts. A note never becomes a second plan.

Tracked and public-safe: `ENGINEERING.md`, `AGENTS.md`, the reviewed
`.work/*.md` notes, `docs/`, code. Local by standing decision (gitignored):
`PLAN.md` and `CLAUDE.md`; the board is therefore checked locally, and a
`.work/` note must stand on its own for a reader without the board. Untracked
and private: the raw evidence the notes summarise (`.work/evidence/`,
`.work/private/`), profiler dumps, benchmark artefact trees. Never `git add .work/` wholesale; before
staging a note read the staged content and reject credentials, private
hostnames or addresses, SSH commands and key paths, absolute personal paths,
raw dumps. A tracked note summarises private evidence; it never reproduces it.

## 3. Code is the source of truth

Docs, tables and old benchmark pages may be stale. For an implementation or
audit question inspect the code, then git history, then docs. Never claim
backend or kernel support from documentation alone.

## 4. Backend claims require a matrix

Never call a change "cross-platform", "ARM", "x86" or "VNNI" without reading
its gates: `#if`, runtime predicates, dispatch tables, fallbacks. Classify
runtime and kernel work as **A** common runtime · **B** common design,
backend-specific implementation · **C** backend/ISA-specific. Backends: x86
AVX2 · AVX-512 VNNI · Arm NEON / DOTPROD / i8mm (SMMLA) · Apple Arm
(Accelerate, Metal) · CUDA. For a change to allocation, threads, scheduling,
dispatch, BLAS ownership, batching or stream lifecycle, state which backends
actually receive it. Common-looking code is not common behaviour.

## 5. A benchmark is invalid until dispatch is proven

Before every performance run record, from the process being measured: git
commit and dirty state, binary hash, compile SIMD target, CPU model and
flags, worker topology and masks, exact command and environment, and the
resolved kernel per operation as the binary itself prints it (effective-config
banner, `--dispatch-map`). Never infer the active kernel from a flag, a
build target or a filename. An explicit request that resolves to a different
path stops the run; a fallback run is never reported as a measurement of the
requested path.

## 6. Fallbacks must be visible

Every important optimised operation exposes which implementation ran: native
optimised · project generic · external BLAS/Accelerate · scalar fallback ·
unsupported. Capability predicates are the single source the dispatcher and
the reports read; never keep a second copy. A silent fallback in a
qualification run is a failure.

## 7. Never benchmark a contradiction

VNNI requested on a binary without the kernel; a batched path claimed at a
concurrency whose B never reaches its gate; a soak with another load on the
box; server parity shown with the CLI's file splitter. Whenever a
contradiction is mechanically detectable, add a preflight gate that refuses.

## 8. Performance reports use explicit terminology

WAVE (screening; may disqualify, never promotes) · SOAK (closed loop at fixed
concurrency for minutes with a drift gate; the only production gate) ·
POISSON (overload) · DIAGNOSTIC (instrumented, never a headline number).
Never a bare "RTF" and never an unqualified "concurrency N". For streaming
report separately: TTFP p50/p95, emission lag p50/p95 (client-observed and
server `lag_ms`), finalization lag p95, backlog max, chunk service time,
STREAM_RTF (capacity, not continuity), rejects/errors/timeouts, wall
duration, drift, artefact path. Metric definitions live in one module
(`tools/bench/streaming_metrics.py`) that every harness imports.

## 9. Transcripts are the quality gate

A serving change never changes a transcript. Every batched, threaded or
re-ordered path ships with a byte-identity gate against the single-item path
(`make test`, `make test-server-concurrency`, the S2-7 gate). A numerical
change (int8 where there was f32, a different accumulation order, a fused
approximation) is separate work with its own CER gate on `samples/` and is
never promoted on speed alone. Production defaults change only by decision.

## 10. Run lifecycle is strict

A persistent server is never inside a bare `wait`. Save client PIDs and wait
only for them, bounded timeouts, confirm counters advance and CPU load is
real shortly after launch, refuse to start a measurement while `loadavg`
shows another load, terminate samplers and server explicitly, report
survivors=0. Warm-up goes through the request path's own reset.

## 11. No opportunistic scope expansion

While executing item X do not start Y, do not change numerical precision, do
not change production defaults, do not build a new profiling framework, do
not refactor unrelated code. Record discoveries as new board items.

## 12. What is committed is what was built

Canonical SOAK evidence comes from a clean committed tree: build that exact
commit. A dirty-tree binary serves development, WAVE and DIAGNOSTIC only and
is labelled NON-QUALIFYING. Tracked scripts never depend on untracked local
files (`python3 tools/check_repo_integrity.py`). Commit messages: English,
imperative, what and why, no tool attribution.

## 13. One implementation owner; parallel agents provide evidence

Canonical state is a hash. Before delegating analysis emit
`CANONICAL_HEAD=$(git rev-parse HEAD)` and require the analyst to prove the
same HEAD and a clean tree. Bound the question so the answer is directly
consumable. Analysts return EVIDENCE / INTERPRETATION / CONFIDENCE /
CONTRADICTIONS / RECOMMENDATION; read the evidence before the recommendation.
Never block on an analyst when the hypothesis is narrow, reversible,
parity-testable and cheap to measure.

## 14. Completion rule

An item is complete only when its report contains: WHAT CHANGED · WHAT PATH
ACTUALLY RAN · WHAT WAS MEASURED · WHAT REMAINS UNKNOWN · VERDICT: PROMOTE /
KEEP / INCONCLUSIVE / REJECT. Unknowns are stated, not omitted. A skipped
model-gated test is reported with the exact command and reason, never
replaced by a claim.
