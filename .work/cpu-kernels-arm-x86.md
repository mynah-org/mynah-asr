# S5 — CPU kernels for the batched stream step, ARM and x86 together

Status: OPEN (S5-1 landed on macOS/ARM, `[~]` until Linux executes SMMLA and VNNI)

Task: S5-1, S5-2, S5-3, S5-4, S5-5
Question: once the server batches B streams into one step, the hot shape is
`[B·Q ≤ 64, k] × [n, k]ᵀ` with int8 weights. Today that shape leaves the
native int8 path (`T > 16` dequantises the whole matrix and mallocs per call).

Known facts (src/qmat.c at HEAD)
- Native int8×int8 dot per row for `T ≤ 16`, `k ≤ 8192`: NEON SDOT
  (`dot_q8_sdot`, `dot_q4_sdot`), AVX-512 VNNI (`dot_q8_vnni`), AVX2
  (`dot_q8_avx2`), runtime CPUID dispatch on x86, compile-time on ARM.
- `T > 16`: threaded int8 GEMM is opt-in and off (`MYNAH_ASR_QGEMM=1`, lost to
  AMX on Apple); default is dequant + sgemm with `malloc(n*k*4)` per call.
- No SMMLA (i8mm), no SVE, no BF16 kernels; depthwise conv is a scalar triple
  loop; layer_norm SIMD was archived as not worth it.
- Siblings: mynah-tts has `matvec_q8_pair_i8mm` (two activations per SMMLA,
  one weight load serving two rows) proven **bit-identical to SDOT** for the
  same (row, activation), because otherwise a request's output would depend on
  who it was batched with; exported int8 primitives (`act_quantize` with one
  encoding definition: signed for SDOT, unsigned x+128 for VPDPBUSD,
  `pack_q8` with per-row scale + rowsum, `dots_i8`, `epilogue`). qwen-tts has
  the full VNNI/AMX matmat family, KleidiAI (`qsi8cxp` i8mm GEMM, requires
  i8mm at build), and the three-predicate dispatch (`compiled / supported /
  use(gate, B, rows, cols)`) with a `force()` pin for in-process paired A/B.
- Measured elsewhere: the activation-quantisation pass was the serial
  fraction of an int8 conv call (flat 3.8 ms while the GEMM went 16.7 → 7.7
  ms); put it on the pool. The smaller the model, the larger the share of
  preparation over arithmetic.
- Axion has i8mm, bf16, sve2 idle; the binary uses SDOT only.

Plan (each gated by the S0/S4 profile that blames it, in this order)
- S5-1 weight-stationary int8 GEMM for `T ≤ 64`: ARM SMMLA with the SDOT
  fallback, x86 VNNI (`vpdpbusd`) with the AVX2 `maddubs` fallback; activation
  quantisation on the pool; **bit-exact against T single-row calls** (integer
  accumulation, asserted with `==`); removes the dequant+malloc path from the
  server's shapes. Gate: `test_qmat` extended, transcripts byte-identical,
  step time at B=8 before/after on Axion and on an x86 box.
- S5-2 depthwise conv: weights transposed to `[k][d]` at load, NEON/AVX2 FMA
  over `d`. Gate: parity vs scalar, per-layer µs before/after.
- S5-3 KleidiAI behind `KLEIDI=1` as an A/B arm for the same shapes; promoted
  only if it beats S5-1 on the soak envelope, not on a microbenchmark.
- S5-4 build: `SIMD=auto|portable|neon|avx2|avx512` with the double test
  (cpuinfo and a compiler probe), build-flag stamp so objects cannot mix,
  link-only CI job per ISA ("guards the guard"), release flags per target.
- S5-5 pool meter run on the batched step to pick W×T from the 200 µs
  break-even rule, then re-run the S4 ladder.

Numerical rule: int8 for the encoder already exists and is qualified by the
CER samples; any new arithmetic (bf16, different accumulation) is a separate
quality-gated item.

Next action for S5-2..S5-5: after S4's first soak.

---

## Evidence — S5-1 (weight-stationary int8 kernels), 2026-09-18

Acceptance gate stated before the work: every int8 micro-kernel compiled into
the binary gives **exactly** the same int32 accumulator and the same f32 output
as the per-row dot, for random and adversarial data, asserted with `==`; both
arms proven by execution in one process; `--dispatch-map` names the resolved
kernel; the `IDLE HARDWARE` footer stops naming a feature once a kernel issues
it; `make test` and `make check` green; one UBSan run.

### What was written (`src/qmat.c`, `src/qmat.h`)

The batched stream step's hot shape is `C = A·Bᵀ`, `A = [T, k]` with
`T = Σq` over the ready streams (4..64 rows) and `B = [n, k]` int8 with one f32
scale per row, `n, k ∈ {1024, 4096}`. It was served by a threaded loop of
per-row dot products: each weight row was re-walked once per activation row.
S5-1 adds a weight-stationary family — one weight row (or, on i8mm, a PAIR of
weight rows) held while several activation rows sweep through it:

| kernel id | instruction | tile | gate |
|---|---|---|---|
| `neon-smmla` | `SMMLA` (`vmmlaq_s32`) | 2 weight × 2 activation rows, 32 MACs/instr | `__attribute__((target("+i8mm")))` + `mynah_asr_cpu_has("i8mm") == YES` |
| `neon-sdot` | `SDOT` | 1 weight × 4 activation rows, one weight load per 4 dots | `__ARM_FEATURE_DOTPROD` |
| `avx512vnni` | EVEX `VPDPBUSD` | 1 × 4, rowsum hoisted | `target("avx512f,avx512bw,avx512vnni")` + CPUID |
| `avxvnni` | VEX `VPDPBUSD` | 1 × 4, rowsum hoisted | `target("avx2,avxvnni")` + CPUID leaf 7.1 EAX[4] |
| `avx2` | `maddubs`+`madd` | 1 × 4, `\|w\|` computed once per weight vector | `target("avx2")` + CPUID |
| `dot-per-row` | the pre-S5-1 loop | — | kept as the reference arm and the fallback |

Every one of them is compiled into **every** build of its architecture through
a target attribute and selected at runtime — never by a `-march` gate, which is
how the sibling once shipped an ISA claim it could not honour. Proof, from a
compile with **no** `-march` at all: the aarch64 object contains 13 `smmla`;
the x86-64 object contains 7 EVEX `vpdpbusd %zmm` and 5 `{vex} vpdpbusd %ymm`.

The unsigned-activation correction has one definition: VPDPBUSD and `maddubs`
want u8, so the activation is `x + 128` and `128·Σⱼwⱼ` is subtracted per weight
row — hoisted out of the four activation rows in the weight-stationary form,
which is where it stops being a per-dot tax. Every term is integer.

`MYNAH_ASR_CAPS` is the opt-out and grew an ARM ladder to be one:
x86 `auto|scalar|avx2|vnni` (`vnni` = either VPDPBUSD encoding), ARM
`auto|scalar|sdot|smmla`. No new flag was registered. The x86 ladder's top rung
now also covers AVX-VNNI parts, so **the per-row dot asks
`x86_use_avx512vnni()` rather than `caps >= VNNI`** — issuing an EVEX kernel on
a VEX-only part is a SIGILL, and that is the one place the widening could have
bitten.

`--dispatch-map` gained a row. `kernel.int8_dot` is the one-row dot,
`kernel.int8_rows` is the weight-stationary stacked product: different
questions, different predicates (`mynah_asr_qmat_int8_dot_kernel()` and
`mynah_asr_qmat_int8_kernel()`), because on an i8mm host they differ — a single
activation row leaves half an SMMLA tile empty, so the one-row dot stays SDOT.
Per-kernel counters (`mynah_asr_qmat_kernel_counter`) count one increment per
weight-row block, so a gate can prove which kernel executed.

### THE FINDING: the float epilogue was never pinned, and the gate caught it

The int32 side is exact and order-free, so it was never the risk. The risk was
the epilogue `(float)s * ws * sx` — three factors, and under `-ffast-math` the
compiler is free to group them either way. It does not choose the same way
everywhere. The identity gate, run with a scalar model of SMMLA's documented
lane mapping (see below), failed on its first shape:

```
neon-smmla   FAIL  float epilogue differs at row 2 activation 0:
             reference 0.867895186 vs kernel 0.867895246
             — the integers agreed
```

Apple clang 21, `-O3 -march=native -ffast-math`: `dot_q8_sdot` inlined into the
per-row loop grouped one way, **the same function** inlined into the SMMLA
kernel's odd-row tail grouped the other. The grouping was therefore chosen by
which kernel served the row, the kernel by the tile position, and the tile
position by how many other streams happened to be in the batch — a transcript
that depends on who you were batched with (ENGINEERING.md §9). This is the
mynah-tts incident (71 of 192 rows, 2 ULP, GCC 15) reproduced independently on
a different compiler and a different architecture: it is not a GCC quirk.

Fixed at the source, not gated around: one `qmat_q8_epilogue()` with a value
barrier, used by every int8 kernel in the file including the three pre-existing
per-row dots. WHICH grouping to pin was measured, not chosen — it had to be the
one the shipped binary already produced, or the "refactor" would have been a
silent numerical change. Both entries (`mynah_asr_qmat_mul`, including its
`T>16` arm, and `mynah_asr_qmat_mul_rows`) were dumped over 7 shapes × 9 values
of T, 791,448 bytes, before and after:

| pinned grouping | vs the pre-S5-1 bytes |
|---|---|
| `((float)s * ws) * sx` — what the source text says | **DIFFERS** |
| `(float)s * (ws * sx)` | **byte-identical** |

`-ffast-math` had been hoisting the scale product all along. That is now the
definition instead of a compiler mood, and it is the same in a `-ffast-math`
build and in the UBSan build, which it was not before.

### Identity, measured (`tests/test_qmat`, model-free, runs in CI)

82 shape/T cases per kernel: 6 shapes with `k` off every unroll (7, 37, 64, 65,
255, 1024) and `n` off the 32-row block and the SMMLA row pair (33, 96, 31, 65,
64) × T ∈ {1,2,3,4,7,8,15,16,17,31,32,63,64}, plus the real batched-step shapes
`n=1024 k=1024` at T ∈ {4,17,64} and `n=1024 k=4096` at T=8. Two passes:

- **integers** — weight scales forced to 1 and activations built so the per-row
  absmax is exactly 127 (so `sx == 1` exactly, asserted); the f32 output is then
  `(float)s`, exact below 2²⁴ (asserted too, so the gate cannot quietly become a
  float gate). Weights walk `{-128,-127,-1,0,1,127}`; activations walk
  `{-127,-1,0,1,127}` — `-128` is on the weight side only, because the
  activation quantiser clamps to `[-127,127]` and a test feeding `-128` would be
  testing a value the runtime cannot produce.
- **floats** — real per-row scales and a real activation scale, `memcmp` against
  the reference arm.

Both arms run in one process through `mynah_asr_qmat_kernel_force()`, and each
kernel's own counter must be non-zero or the case is a FAIL: a gate that passes
because the kernel never ran is worse than no gate. A kernel this host cannot
run prints SKIPPED **with the reason**.

```
int8 kernel identity (S5-1): every compiled kernel == the per-row dot, bit for bit
  dot-per-row  exact OK  82 shape/T cases, 763 kernel blocks
  neon-smmla   SKIPPED (compiled, not runnable here): this CPU does not report
               FEAT_I8MM: compiled here, never issued here
  neon-sdot    exact OK  82 shape/T cases, 466 kernel blocks
  avx512vnni   SKIPPED (not compiled): not an x86 build
  avxvnni      SKIPPED (not compiled): not an x86 build
  avx2         SKIPPED (not compiled): not an x86 build
```

Also green at `MYNAH_ASR_CAPS=sdot` (same table) and at `MYNAH_ASR_CAPS=scalar`
(the reference arm itself is then off, and the gate says so and skips rather
than comparing nothing to nothing).

### SMMLA: what was proven here and what was not

**This Mac is an Apple M1: `hw.optional.arm.FEAT_DotProd = 1`,
`hw.optional.arm.FEAT_I8MM = 0`.** `./mynah-asr --dispatch-map` says
`kernel.int8_rows compiled=neon-smmla+sdot supported=sdot resolved=neon-sdot`
and the footer says `i8mm absent`. The SMMLA path **cannot execute on this
host**. Said plainly rather than implied.

What WAS proven locally, so that CI is confirming rather than discovering:

1. It compiles and the instruction is really emitted: 13 `smmla` in
   `build/src/qmat.o`, from a plain `make` on a CPU without the feature.
2. The blocking is right. A scratch copy of `src/qmat.c` (not committed) with
   `vmmlaq_s32` replaced by a scalar model of its documented lane mapping
   (`r[0]+=a.lo·b.lo, r[1]+=a.lo·b.hi, r[2]+=a.hi·b.lo, r[3]+=a.hi·b.hi`) and
   the runtime probe forced to YES runs the whole identity gate:
   `neon-smmla exact OK 82 shape/T cases, 466 kernel blocks`. That covers the
   2×2 tile indexing, the odd-activation-row tail, the odd-weight-row tail and
   the `k` remainder — i.e. everything except the semantics of one documented
   instruction. It is also what produced the epilogue failure above.

What is NOT proven here: that real i8mm silicon implements those lanes as
documented. That is a Linux run.

### Activation quantisation on the pool, and the threshold

The pass is now `quantize_act_rows()` and goes on `mynah_asr_parallel_for` when
`T·k >= QMAT_ACT_PAR_ELEMS`. Rows are independent (per-row absmax), so the pool
is bit-identical by construction.

The threshold is measured, not assumed. mynah-tts' rule is that a parallel
region must be worth ≥ 200 µs to pay its ~20 µs of wake-up latency. Isolated
loop over `quantize_act_int8` on this host (Apple clang 21, `-O3 -march=native
-ffast-math`): **0.167 ns/element at k=1024, 0.106 ns/element at k=4096** —
10.9 µs and 27.8 µs for a whole T=64 stack. Break-even ≈ 1.9 M elements, so
`QMAT_ACT_PAR_ELEMS = 1<<21 = 2,097,152`.

**On the shapes the batched stream step produces (T ≤ 64, k ≤ 4096, i.e. at
most 262,144 elements) that threshold is never crossed, and it should not be.**
The sibling's pass was flat at 3.8 ms while its GEMM threaded 16.7 → 7.7 ms;
ours is ~140× smaller and structurally so: quantisation is O(T·k) against a
GEMM that is O(T·k·n) with n ≥ 256, i.e. ≲ 0.4% of the call. **Activation
staging is not our serial fraction.** The threshold is kept rather than
deleted, because it is the crossover a much larger T or a slower core would
reach, and because a constant with a measurement behind it is a policy rather
than a guess. The ns/element is an Apple-core number; Linux owes its own.

### Step time

**The `--steptime` table could not be produced on this host.** `models/` on
this machine is a tree of symlinks into `/Volumes/shared`, which is not
mounted, so every model-gated target skips:

```
$ ./tests/test_stream_batch models/nemotron-3.5-asr-streaming-0.6b --steptime
$ echo $?
77
```

`make test` is green with every model-gated test reported as SKIP for that
reason. The B ∈ {1,2,4,8} identity run with the counters is therefore **owed**,
not done.

What could be measured is the kernel itself, model-free and paired in one
process (`tests/test_qmat --bench`). macOS dev signal, **host load stated**:
Apple M1, 8 cores, `load averages: 3.2–4.2` — a loaded box, so these are a
direction, not a number. Best-of-7 per point, both arms interleaved in the same
process, three repetitions of the whole table.

`n=1024, k=1024` (ms per `mynah_asr_qmat_mul_rows` call, speed-up = per-row ÷ weight-stationary):

| T | per-row | neon-sdot ws | speed-up | across 3 runs |
|---|---|---|---|---|
| 1 | 0.038 | 0.027 | 1.41 | 0.91–1.41 (noise: T=1 is all tail, the two arms run the same code) |
| 2 | 0.043 | 0.039 | 1.10 | 1.04–1.28 |
| 4 | 0.058 | 0.047 | 1.23 | 1.23–1.51 |
| 8 | 0.109 | 0.068 | 1.60 | 1.60–2.03 |
| 16 | 0.168 | 0.089 | 1.89 | 1.65–2.01 |
| 32 | 0.224 | 0.127 | 1.76 | 1.75–2.07 |
| 64 | 0.439 | 0.200 | 2.20 | 2.08–2.20 |

`n=1024, k=4096`:

| T | per-row | neon-sdot ws | speed-up | across 3 runs |
|---|---|---|---|---|
| 1 | 0.073 | 0.060 | 1.22 | 1.06–1.22 |
| 2 | 0.100 | 0.093 | 1.08 | 0.91–1.08 |
| 4 | 0.176 | 0.085 | 2.07 | 2.07–2.24 |
| 8 | 0.305 | 0.136 | 2.24 | 2.24–2.48 |
| 16 | 0.568 | 0.238 | 2.39 | 2.39–2.40 |
| 32 | 1.117 | 0.455 | 2.45 | 2.45–2.59 |
| 64 | 2.179 | 0.968 | 2.25 | 2.23–2.25 |

Read honestly: at T = 1–2 the weight-stationary sweep IS the per-row dot (every
row falls into the T remainder) and the ratio is noise around 1.0; from T = 4 —
the smallest batch the scheduler actually stacks — it is **1.2–2.6×**, on SDOT,
with no i8mm involved. The crossover is a measured model, not a capability
filter, and on this ISA it sits at T ≈ 4. That number is what the SMMLA arm has
to be compared against on Linux, not against the per-row loop.

### Gates run on the tree that was left

- `tests/test_qmat` — green, table above; also at `MYNAH_ASR_CAPS=sdot` and
  `MYNAH_ASR_CAPS=scalar`.
- `make test` — green; every model-gated test SKIP (77), reason above.
- `make check` — green (41 tasks, 38 scripts, 20 flags).
- UBSan: `make clean && make CFLAGS="-std=c11 -O2 -g -fsanitize=undefined
  -fno-omit-frame-pointer ..." tests/test_qmat && tests/test_qmat` — clean, no
  diagnostics, identity still exact. Never with `-ffast-math`.
- Cross-compile of `src/qmat.c`, `src/dispatch.c`, `src/flags.c` to x86-64
  (`-target x86_64-apple-macos13`, no `-march`): no warnings, both VPDPBUSD
  encodings and `vpmaddubsw` present in the object.
- `tests/test_stream_batch` — **SKIP 77**, no model on this host.

### What Linux still owes

1. **The ARM i8mm arm must actually execute.** `ubuntu-24.04-arm` is
   Neoverse-class and has i8mm; CI now runs `tests/test_qmat` there at
   `MYNAH_ASR_CAPS=scalar|sdot|smmla` and prints `--dispatch-map`, so the log
   is the evidence. The line to look for is
   `neon-smmla   exact OK  82 shape/T cases, N kernel blocks` — if it says
   SKIPPED, the runner does not have the feature and the claim is still open.
2. **The x86 arms must actually execute.** `ubuntu-latest` has at least AVX2
   and usually AVX-512 VNNI; AVX-VNNI (VEX) needs an Alder Lake / Sapphire
   Rapids runner and may well print SKIPPED. Same rule: the printed table is
   the claim.
3. **`tests/test_stream_batch <model>` at B ∈ {1,2,4,8}, int8** — identical at
   both levels with the new kernel counters non-zero. Nothing here ran it.
4. **`--steptime` before/after on a quiet Linux box**, and the SMMLA arm
   against the SDOT weight-stationary arm (not against the per-row loop).
5. **`QMAT_ACT_PAR_ELEMS` re-derived** from a Linux ns/element figure.
6. **The epilogue grouping re-verified under GCC.** The barrier pins
   `s*(ws*sx)`; that this is also what GCC's `-ffast-math` was producing on
   Linux is an assumption until a before/after byte dump says so. If it is not,
   the numbers move on Linux — and the transcripts with them.

## Evidence — S5-1, the gates the implementing agent could not run (owner, 2026-09-18)

The agent that wrote the kernels had no converted model in its worktree, so the
model-gated half of the gate was owed. Run here on the merged tree (`make` at
`v0.9.1-56`, Apple M1, Accelerate, `models_local/nemotron-3.5-asr-streaming-0.6b`):

- `tests/test_qmat` kernel identity: `dot-per-row exact OK 82 shape/T cases, 763
  kernel blocks` · `neon-sdot exact OK 82 shape/T cases, 466 kernel blocks` ·
  `neon-smmla SKIPPED (compiled, not runnable here): this CPU does not report
  FEAT_I8MM` · the three x86 kernels `SKIPPED (not compiled)`. The skips carry
  their reason, which is the point: a claim that was not executed says so.
- `tests/test_stream_batch` with the model: **IDENTICAL at both levels** for
  B ∈ {1,2,3,4,8}, int8 and f32, and the per-kernel counters prove the resolved
  kernel ran (`int8 kernels neon-sdot 171700 | resolved neon-sdot` at B=4).
  `DEQUANT 0` everywhere.
- `tests/test_streaming`: parity IDENTICAL and reset+paced IDENTICAL.
- Transcripts of `test_it.wav` unchanged in int8 and f32 against the pre-merge
  tree, which is the check the epilogue fix needed: pinning
  `(float)s*(ws*sx)` moved no shipped byte.
- In-situ step table: taken at **loadavg 15 on 8 cores** (another agent was
  building), so the absolute milliseconds are NOT comparable with the S1-7 table
  and are not quoted. What survives the load, because both arms ran inside the
  same process under the same load, is the ratio: batched/single 0.53 at B=2,
  0.37 at B=4, 0.27 at B=8. DIAGNOSTIC.

What Linux still owes for this item is unchanged and listed above: SMMLA on real
i8mm silicon, the three x86 kernels, and a before/after byte dump of the float
epilogue under GCC's `-ffast-math` (the barrier pins one grouping; that GCC chose
the same one is an assumption until dumped).
