# The method — how not to fool ourselves

Status: REFERENCE (read before starting any serving or kernel work)

Source: qwen-tts `ENGINEERING.md` and `docs/ENGINEERING-METHOD.md` (lineage
declared there: Abrash's *Graphics Programming Black Book*), as re-read for
mynah-tts in its `.work/engineering-method.md`. This is the ASR copy. Almost
none of it is code, and it is the cheapest thing in the whole v2 to adopt.

## 1. The four opening lines

> Do not optimize the code. Optimize the work.
> Do not trust expected behaviour. Observe the machine.
> Do not celebrate a faster component until the system is faster.
> Every unexplained millisecond is an engineering question.

With the disclaimer that prevents the usual misreading: this is not a
low-level-programming aesthetic. Hand-written SIMD is not the point. The rule
exists to make us **more suspicious, especially of results we like**.

## 2. The organising principle

The agent must not remember the project; the repository must make forgetting
impossible. `PLAN.md` says what remains, `ENGINEERING.md` says how we work,
`.work/` holds the detail of one task, and **the program itself proves which
path it is executing**. That last clause is why a dispatch report, a kernel
census and an effective-config banner exist at all.

## 3. Rules with the reason attached

- **Cost model before code.** Current cost · suspected cause · proposed
  transformation · **maximum plausible saving** · new work introduced · risk ·
  **the smallest experiment able to kill the idea**. If the maximum plausible
  saving is too small for the product goal, do not optimise. The most expensive
  error recorded in the sibling repo: a 36x kernel win that took RTF to 0.245
  while the machine still served two or three real-time streams, because the
  limit was never the kernel.
- **The win ordering for a CPU streaming server is (1) server design, pinned
  core split and batching, (2) dataflow, (3) kernels.** The instinct is the
  reverse. Both sibling repos measured this.
- **A label is not a causal explanation.** `encoder`, `decode`, `GEMM`,
  `queue` are boxes to open, not answers.
- **Which control would make my preferred explanation look stupid if it were
  wrong?** Ask before every experiment. Measure differences, not absolutes,
  when system noise is in the number.
- **The claim chain**: microbenchmark → component → critical path → server →
  concurrency → sustained workload → quality. Never skip a level. A wave screen
  never promotes; only a soak with a drift gate does.
- **Numerical changes are separate work.** int8 where there was f32, a
  different accumulation order, a fused approximation: each needs its own
  quality gate (WER/CER against the oracle transcripts) and is never promoted on
  speed alone. The sibling repo rejected an arm that passed every serving gate
  because its audio-quality gate failed.
- **A benchmark is invalid until dispatch is proven.** Never infer the active
  kernel from a flag, a build target or a filename. The binary prints which
  kernel resolved, from the same predicate the dispatcher uses. An explicit
  request that resolves to something else stops the run; a fallback run is
  never reported as a measurement of the requested path.
- **Fallbacks must be visible.** Every optimised operation can say which of
  {native optimised, project generic, external BLAS, scalar fallback,
  unsupported} ran. A silent fallback in a qualification run is a failure.
- **Never benchmark a contradiction.** Where a contradiction is mechanically
  detectable, add a preflight gate that refuses to start.
- **What is committed is what was built.** Canonical soak evidence comes from a
  clean committed tree; a dirty-tree binary is NON-QUALIFYING.
- **No opportunistic scope expansion.** Discoveries become board items, not
  detours.
- **One implementation owner.** Parallel agents supply evidence
  (EVIDENCE / INTERPRETATION / CONFIDENCE / CONTRADICTIONS / RECOMMENDATION)
  against a canonical HEAD hash; they never own the tree or the plan.
- **Completion rule.** WHAT CHANGED · WHAT PATH ACTUALLY RAN · WHAT WAS
  MEASURED · WHAT REMAINS UNKNOWN · VERDICT: PROMOTE / KEEP / INCONCLUSIVE /
  REJECT. Unknowns are stated, not omitted.

## 4. Every tool declares a refusal

The most copyable property of the sibling toolchains, and it needs no kernel
work: a census exits non-zero if an operation resolved UNKNOWN; a harness
refuses to print cadence percentiles when the client coalesced too many reads;
a comparator refuses to difference two arms whose ISA, threads or flags differ
or are both unrecorded ("equality of two unknowns is not sameness"); a doctor
labels every value MEASURED / CACHED / TRANSFERRED / PREDICTED / UNKNOWN. A
number that cannot be supported is not printed.

## 5. Vocabulary for performance reports

WAVE (screening: C requests at once, wait, repeat) · SOAK (the production gate:
closed loop at fixed concurrency for minutes, with a drift gate across windows)
· POISSON (overload, open arrivals) · DIAGNOSTIC (instrumented, never a headline
number). Never a bare "RTF" and never an unqualified "concurrency N". For
streaming ASR the metrics are defined once, in one module every harness imports
(see `bench-harness-streaming.md`): TTFP, partial gap, finalization lag,
chunk service time, STREAM_RTF, drift, rejects/errors/timeouts.

## 6. Local traps already paid for

- Apple ships GNU Make 3.81 with whole-second timestamps; a tight edit-build-run
  loop can run the old object. Any gate that decides something gets a
  `make clean` first.
- `-ffast-math` is load-bearing here: never `±INFINITY` in code (cost 6.5x RTF
  once), sigmoid hand-stabilised, sentinel instead of `-inf` in Viterbi.
- A test that passes while the function under test is never exercised is worse
  than no test (twice in this repo: a batch test that never segmented, a server
  test the CI never ran). Make a gate fail on purpose once.
