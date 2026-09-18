# Sibling wins, feature flags and kernel state — what transfers to the v2 ASR server

Status: REFERENCE (analysis 2026-09-18, qwen-tts @e391ec5 mynah-tts @368021a)

Task: reference note for S1–S5. Question: of everything the two sibling repos
measured, what is a *result* we must not re-learn, what is an *implementation*
that failed and could still work, and what is our own. Read `serving-v2-design.md`
first; this note is its evidence appendix and does not restate it.

Hosts named below, as the sibling notes name them: **Turin** = AWS c8a.8xlarge,
AMD EPYC 9R45, 32 physical cores, 4 CCX, SMT off, one NUMA node. **SPR** = GCP
c4-standard-24, Xeon Platinum 8581C, 12 physical cores online, SMT off, 2x6
prefork. **Axion** = GCP 32-core Neoverse-V2. **M1** = Apple Silicon dev box.
Vocabulary is ENGINEERING.md §8: WAVE screens, SOAK promotes, DIAGNOSTIC never
quotes a headline.

---

## 1. THE WINS

`STREAM` = STREAM_RTF p95 unless stated. `prebuf` = required_prebuffer p95.
`safe` = safe_play_start p95. Verdicts are the sibling's own.

### 1.1 The C12-WIN ladder (Turin, 4x8 one worker per CCX, cap 4, q4 units, DL-2 elastic lane, RES1_V2, VNNI per-item int8 decoder, native BF16 prefill)

Control: C12 wave STREAM .772/.841, TTFA p95 272 ms, prebuf 256, safe 469,
stall@250/@500 = 0. 10-min control soak pooled .854/.923, short .867/.966,
conversational .856/.919. Goal: STREAM p95 ≤ 0.90 including the short tail.

| tried | measured effect (host) | why it won / lost | transfers to streaming ASR? |
|---|---|---|---|
| **Prefill helper** `QWEN_PREFILL_HELPER=1` (admission off the loop, LOW pool priority) | Turin 10-min soak: pooled STREAM .923→.915, **short .966→.915**; TTFA p95 **172→683 ms**, safe 417→922, stall@250 0→0.8% (short 1.6%) | Mechanism VALIDATED (inline admission *is* the short tail, ≈0.05 of STREAM); the LOW-priority one-shot implementation starves on a busy pool | **maybe** — our admission is a slot open, not a prompt prefill; but the rule "a second submitter on the engine pool loses" is ours too |
| **Fixed-B3 overlap diagnostic** | 1,349 active=3 iterations: decoder overlaps **54.9% of wall**; CP p50 **22.2→36.5 ms** (p95 22.7→37.6), Talker flat; decoder unit 49.2 ms = conv_stack 43.3 + transformer 5.2 | Confirms the causal model; the tax is L3 residency loss, not arithmetic | **yes** — same shape: a batched encoder step next to a per-stream decode loop on one cache domain |
| **Pool spin** `QWEN_POOL_SPIN` 4096 / 16384 / 65536 | Δ STREAM p95 ≤ +0.018 = run noise on Turin; but on c8a 2x8 C4 the *soak* had 65536 worth **+11%** while the wave said neutral | Spin is a per-host constant with a soak-only signature, not a lever to tune per experiment | **maybe** — set it once per host from a soak, never screen it |
| **BF16 pre-up** `QWEN_SD_BF16_PREUP=1` (44 MB persistent bf16 copies of decoder transformer + in/out proj, bf16-packed activations) | 3-wave C12 screen pooled STREAM **.842→.825**, short .843→.829, conv .817→.793, medium .814→.798, prebuf 276→250, safe 492→457. Decoder ms unchanged (56.4 vs 56.0); CP **did not** drop (5.1–5.5 → 6.0–6.2) | **NO-GO**: paired same-generation audio gate failed, `mel_corr=0.97890 < 0.98`. It also targeted the 5 ms transformer, not the 43 ms conv stack | **no** as implemented (bf16 *activations*); **maybe** as weight-only bf16 with f32 accumulate |
| **ConvT one-GEMM** (zero-filled `[k·in_ch][full_len]` panel + one f32 GEMM) | Bit-identical (`max_abs=0`). Slower at **every** point: +14–18% at chunks 1–2, +26–32% at chunks 4–8, B1–B4 (B1/chunk4 56.54→71.47 ms; B4/chunk8 389.65→501.06) | **REJECTED**: the zero-expanded panel multiplies FLOPs and traffic by ≈ k/stride. The *idea* is not dead — the un-expanded `[k·out_ch][in_ch]` form with a fused epilogue was never built | **yes, as a warning** — never express a strided op by expanding the input |
| **Allocation-only glue** (calloc→alloc for fully-overwritten temporaries) | No resolvable q4 movement B1–B4; expected ≤1.5 ms ≈ 3%, inside the ~0.8 ms bench noise | INCONCLUSIVE/reverted. Allocation churn was never the dominant term here | **maybe** — our 55–60 malloc/chunk is a different order; measure the count first (S0-4) |
| **RES1_V2 split-input** (direct `tail`+`new` instead of a contiguous `[tail\|new]`) | Exact suffix parity (`max_abs=0`); **+0.3–2.0% q4, +4.7–5.9% q8** across B1–B4 | REJECTED: the estimated ~95 MB/unit saving never appeared, so the estimate was wrong | **maybe** — the same "avoid the concat, choose (src,stride) once per row" trick is right for our streaming conv cache, but prove the traffic first |
| **First-ramp 1,4** | Knob written, removed, **never run** | Recorded as NOT TESTED rather than as a verdict | n/a |
| **Admission slicing** `QWEN_PREFILL_SLICE=N` (WIN-10) | Built default-off. State oracle: slices 2/3/4/5/7/16/32/48 vs unsliced → `dec_x` max abs **0.000e+00**, **0** K/V rows differing. vs monolithic: 7.96e-04 (38 pos) / 1.686e-03 (107 pos) — below one bf16 ulp (3.91e-03). **x86 never run**: no `admit_ms`, no A/B, no soak | Resume is bit-exact; a 1-token slice takes the M=1 matvec path (different accumulation order) and broke parity until the rule absorbed a 1-token remainder. The spec's own mel-corr ≥0.99 gate was **withdrawn as unsatisfiable** for any token-outer sliced prefill | **yes, as method**: the state oracle, not the audio, is what decides a slicing change. Our analogue: N-concurrent transcripts byte-identical to the same stream alone |
| **Conv-stack ConvT + weight-only bf16** (WIN-11, spec only) | Cost target: unit 49.2 ms = res1 17.1, convt 8.6, res2 4.0, resadd 3.5, alloc 1.5, final 1.3, snake 1.0, transformer 5.2, convnext+init ≈7. Weight streams: ConvT block0 (1536→768, k=16) **75 MB f32**, convnext pw 67 MB, init conv 44 MB. GO gates: ≥2 ms step A, ≥4 ms A+B | **NOT RUN.** The reasoning is the transferable part: the bandwidth-bound layer is the one with tiny N (block 0 streams 75 MB for 16 columns) | **yes** — our `[Σ Q ≤ 64, k] × [n,k]ᵀ` batched step has exactly this shape at small B |
| **VNNI residual glue** `QWEN_SD_GLUE` (WIN-12, spec only) | Removes 6 of ~10 full-size passes per residual unit (≈29 MB of traffic at block 3 against ≈4.5 ms of conv work); target ≥3 ms/unit, **exact parity** expected | **NOT RUN** on x86. Note the forbidden list: no per-element branch in the row loop, no new concat, no calloc where the consumer writes everything | **yes** — the "one (src,stride) choice per row, never per element" rule is directly portable |

**Ladder verdict:** no qualified C12 win. The remaining mechanism is a
decoder-in-flight CP tax, not spin, not layout, not allocation.

### 1.2 P2/P4 mechanisms (SPR, 2x6, cap 2, AMX Design-D, ragged threshold 2)

| tried | measured effect | why | transfers? |
|---|---|---|---|
| **Fused residual** `QWEN_SD_FUSED_RESIDUAL` | C4 STREAM **.831→.788**, TOTAL .905→.862, prebuf **389→266**, safe **568→426**, stall@250 **17%→0%**. 5-min C4 SOAK (119 req): STREAM .8304/.8933, TTFA p95 526, prebuf 554.5, safe 916.8, max gap 966.9, stall@500 0.84%, windows .8987/.9045/.8617/.8838, PASS | The one class of change that wins: it **shortens the inline critical-path call**, so throughput and cadence improve *together* | **yes** — anything removed from the batched step wins twice |
| **C5/C6 capacity screen** with the same binary | STREAM p95 .852/.802 (fine) but **TTFB p95 4.28 s**, TTFA p95 4.44/4.77 s, safe 4.60/5.0 s | The failure above capacity is **pre-admission slot wait**, not compute. RTF alone selects the wrong point | **yes** — our refusal must happen at accept, not in a queue |
| **Prefork admission bound** `--max-queue 0` | C5 wave: 8 accepted, **2 rejected with HTTP 503**; accepted TTFB 68/187 ms, TTFA 423/562, STREAM .764/.788. Control (listener gated when full): **no rejects**, TTFB/TTFA p95 ≈ **4470/4635 ms** | 97% of the tail hid *before* `accept()`, where no child deadline can see it. Keep the listener polled, refuse explicitly | **yes** — already doctrine in our §4 admission ladder; this is the number behind it |
| **Same-pool decoder consumer** `QWEN_DECODER_THREAD=1` | C4 STREAM **.847→1.296** (+53%), TTFA p95 **174→1126 ms**, prebuf 425→1251, max gap 511→1286, context switches **×2.9**, cores flat (9.0→8.8), decoder calls still `group=1` | A second submitter pays `submit_mtx` serialization *and* loses the ragged cohort. One sampled 1-frame call reached **1.4 s** | **yes** — one thread enters the model per worker, asserted |
| **Ragged claim-first allocation** | KPI-neutral (C4 .877→.817, C3 .794→.809 — inside wave variation). 4,392 panels, 618 allocations vs 702 implied = **12% fewer events, 24.6% less scratch** (732.4 MB vs est. 971.1 MB) | RETAIN as hygiene, **not** a serving win. Claiming work before allocating for it is free correctness | **yes**, free |
| **Ragged panel scratch reuse** `QWEN_SD_RAG_PANEL_SCRATCH` | C3 .846→.826 but **C4 .810→.843**, prebuf **344→420** | REJECTED and removed: per-pthread retained buffers bought nothing and carry a lifetime/PSS risk | **no** — do not build per-thread caches before a measured allocation cost |
| **Ragged panel parallelization** (P2) | C4 p95 **1.338→0.870** | The single largest structural win of the whole campaign | **yes** — parallelise *inside* one call before adding threads outside it |
| **Ragged threshold 2** | C4 p95 1.0035→0.954 | Small panels are allowed on the pool | maybe |
| **Low-N M split / 1x12 cap 4** | M split C4 .971→**1.106**; single-process 1x12 with real `items=4` worksets **1.125/1.277** | REJECTED: manufacturing tile tasks on a glue-bound stage adds dispatch and burst | **yes, as a warning** |

### 1.3 The decoder lane (DL-1…DL-4, one CCX = 8 cores, 1.7B int8)

Laws measured first: inline `T(B) = 40 + 13.5·B` ms, STREAM p95
**.674/.861/.991/1.172** at B1..B4; decoder = **9.7 ms per decoded slot-frame
(72% of per-slot cost)**, Talker +1.6, CP +1.8, serial 0.05. Talker+CP
26.4+20.8 ms on 2 threads ≈ 26.2+18.1 on 8 → **the weight stream saturates a
CCX at 2–4 threads**. Decoder ms/frame vs threads: 33 (1) / 18.5 (2) / 12.9 (4)
/ 7.6–9.7 (8). Two weight streams on one CCX **halve each other** (1.28 vs 0.97).

| arm | measured | reading | transfers? |
|---|---|---|---|
| **DL-1 static 4+4 lane** | STREAM p95 B2 .866→**.774** (−11%), B3 1.020→**.871** (−15%), B4 1.203→**.997** (−17%); iteration wall p95 **192→77 ms** at B4; stall@250 at B4 **100%→0%**; mailbox wait 1.6/2.4/3.6 ms per frame; decoder team **0.54–0.82 core-eq of 4** | Gate (B3 ≤ .85, B4 ≤ .92) **NOT met** and FAIL not triggered. The decoder left the critical path and is *under-used*; the new wall is the **step side on 4 threads** (+27–29%: Talker+CP 54.0→69.8 ms at B4) | **yes** — our optional decoder lane must be default-off and justified by a profile, exactly as §4 says |
| **5+3 and 6+2** | fixed B4 **1.113 / 1.364**; long B4 1.015 / 1.263 (prebuf 5.9 s); mailbox wait 8.0 / 16.6 ms per frame | The decoder needs **≥4 cores** to stay hidden; returning threads to the step side buys 3–7 ms only | yes |
| **DL-2 elastic 8↔4+4** | B4 **.987** fixed, long B4 **.895** (inline 1.107); Talker+CP at B4 **69.5 elastic vs 69.8 static** | **The static-partition tax is falsified.** The inflation lives only in the windows where the decoder runs: CP loses L3 residency | yes |
| **DL-3 falsifiers of the overlap tax** | CP in overlap **35.7 ms** vs 23.5 without. DIRECT_CONVT/DWCONV/INPUT **36.3** (no effect); 1-frame sub-quantum **38.2**, STREAM **1.475**; 2-frame 36.8 / 1.132; q8 units 34.2 / .864 long but **prebuf 806 ms, stall@250 100%**; NTA prefetch 37.2; hot lane workers 35.7; lane-sized panels 35.7 | The tax is **not** the live set, the weight cacheability, the handoff, park latency or the unit size. Only the *fraction of overlapped iterations* scales it | **yes** — do not spend on cache tricks; spend on fewer overlapped iterations |
| **DL-4 multi-slot cohort** (ARM, S=2..3) | Exact kernel/ABI oracles pass; short WAVE **2.8–3.8% slower** ON than OFF at cohort 2 and 3, both 0.6B and 1.7B | Implemented, opt-in, negative on this screen. Not a verdict on the mechanism — a verdict on that implementation | **maybe** — but note: ASR arrivals are on a grid, theirs are not (see F3) |

### 1.4 Pool and admission levers

| lever | measured | verdict |
|---|---|---|
| **Hard lead gate** `QWEN_STREAM_LEAD_GATE` (park a stream above 250 ms lead) | **95.8%** of checks suppressed; cores 7.3→5.2; STREAM p95 **.838→.986**; stalls unchanged | REJECT. In a lockstep worker there is no other work to give the freed time to, and parking one stream drops the other to B=1 (+20% per slot). Production rate was clamped to playback rate **by construction** |
| **Utilization-aware admission** `QWEN_ADMIT_UTIL` (LS-4), thresholds 40/60/80 ms | All 3 fifth arrivals admitted in every arm; real B3 occupancy proven (737/754/737 `n_active=3` iterations). **Established four**: STREAM p95 .835 → **.993 / 1.004 / .985**; prebuf 165 → **659 / 801 / 529** ms; safe 936 → 1369/1577/1241; max gap 391 → 704/948/657; **stall@250 0/12 → 6/12 in every arm**; stall@500 0 → 2/12 at limits 40 and 60. The fifth itself: STREAM ≈ .96, TTFA 394–479 ms | **FAIL.** The admitted stream became interactive and damaged everyone else. Local gap after the fifth = 2–3× the p95 before it |
| **F3 cross-worker cohort coincidence** | Steady state, 1,112 ready events: useful B≥3 within ±0.25 ms **1.62%**, ±1 ms **2.70%**, ±2 ms **3.60%**, ±8 ms **11.69%** — against a pre-declared 25% floor | NOT JUSTIFIED. **This is the one result that does not transfer**: TTS arrivals are random, ASR chunks arrive on the real-time grid, so within-worker batching is structural for us, not opportunistic |

### 1.5 Cadence and quantum — the law that selects the operating point

`lead ≥ C × w ≈ ρ_f × (chunk audio)`. Chunk sweep (SPR, short wave), prebuf p95:
**8 → 0.450 s, 12 → 0.757, 16 → 0.785, 24 → 0.927, 32 → 1.222** while STREAM p95
moved only **0.919 → 0.880**. Mixed bank at chunk 32: underrun p50/p95
**2.15 / 2.55 s**. Chunk 32 is an RTF artifact; `safe_play_start ≈ 3.0 s`.

| quantum screen | STREAM p95 | prebuf p95 | safe p95 | max gap p95 | stall@250 | req/s |
|---|---|---|---|---|---|---|
| quantum floor, pre-fused, C4: q1 | 1.005 | 224 | 535 | 273 | 0% | — |
| q2 | .892 | 149 | 568 | 300 | 0% | — |
| q4 | .862 | 277 | 683 | 366 | 0% | — |
| q8 | .817 | 333 | 606 | 602 | **50%** | — |
| **F1**, fused-on, C4: q2 | .914 | 276 | 450 | 314 | 0% | 1.57 |
| **F1 q4 (promoted)** | **.868** | **201** | 362 | 344 | **0%** | 1.64 |
| F1 q8 | .822 | 306 | 476 | 531 | **25%** | 1.74 |

q1 is unaffordable (STREAM crosses 1.0 while call count explodes: 81 chunks vs 15
at q8). **Aggregate RTF alone selects the wrong point every time.**

### 1.6 The AMX Amdahl accounting — why kernels came third

`amx_dispatch_share` ≈100% of residual conv1 calls, 0% of Talker/CP at B<4;
`amx_matrix_mac_share` ≈55% of request MACs; `amx_addressable_mac_share` ≈70%;
**`amx_request_wall_share` ≈10–20%** → infinite AMX speed saves at most ~15% of
wall. The decoder is **glue-bound**: movement, preparation, barriers and
transcendentals around the tiles (≈41 pool rendezvous and ≈110 BLAS calls per
call; ~1 GB of fp32 im2col per 32-frame chunk per stream). The old "14% AMX MAC
share" census is **invalidated and must not be cited**.

---

## 2. FEATURE FLAGS AND EFFECTIVE CONFIG

### 2.1 How qwen-tts does it (five parts, each earned by an incident)

1. **One registry, one array.** `g_qwen_reported_flags[]` in `qwen_tts_kernels.c`
   — ~200 `QWEN_*` names in commented groups (kernel dispatch · gates/tiling ·
   KleidiAI · prefill · code predictor · decoder/streaming · server/admission ·
   threading/pool · precision · GPU · diagnostics). It is a *declaration*, not a
   parser.
2. **`[FLAGS]` line**, printed by `qwen_provenance_report()` at startup:
   `[FLAGS] v=1 pid=<pid> NAME=VAL NAME=VAL …` — **only the variables actually
   present in the environment**, as the operator typed them. A profile is then
   verified against a *log* by `perf_profile.py check-flags <profile> <log>`,
   which fails if the line is absent ("the engine never declared, so nothing
   about this run's configuration can be verified") or if any profile value
   differs from what the engine reported.
3. **`[EFFECTIVE-CONFIG]`** (`--effective-config`, and dumped by the server at
   start), `qwen_tts_dispatch.c:201`. Per flag the operator set: `requested ·
   effective · scope/reason`. Four ways a row resolves, in priority order:
   (a) an **owner-declared inert predicate** — `qwen_pool_flag_inert()`
   ("GCD dispatch has no spin loop to tune"), `qwen_kleidi_flag_inert()`
   ("KleidiAI is not compiled into this build"); (b) a **gate lookup**
   (`qwen_flag_gate_status`) that answers "is the kernel this flag steers
   compiled?"; (c) a special case that calls the real resolver
   (`QWEN_SD_POOL` → `qwen_sd_pool_mode()`); (d) the **static scope table**.
   The footer counts set flags and ignored flags and prints
   `WARNING: an ignored flag is a configuration that is not being applied.`
   It also reports `OPENBLAS_NUM_THREADS` as `OVERRIDDEN` plus a
   `blas.ownership` row, because that env is the one that can put a second
   scheduler in the process.
4. **Generated scope table.** `qwen_flag_scope.h` (238 lines, "Generated by
   tools/flag_parity.py — do not edit"): `{name, scope_bitmask}` over
   ARM/AVX2/AVX512F/VNNI/AMX/APPLE/GPU, derived from the preprocessor guards
   around each read site plus one call hop. The build scope is computed from
   `#if`s at compile time; `scope & build == 0` → the flag is IGNORED and the
   row says which families it would have reached. **The static table is never
   trusted over an owner predicate** — `QWEN_POOL_SPIN` is read in portable
   code and so "reaches" everything, yet is inert on GCD.
5. **The registry check.** `tools/check_flag_registry.py` greps every `QWEN_*`
   string literal in `*.c *.h *.cu *.m` (excluding the registry itself) and
   diffs against the registry: `undeclared` (read but never declared → cannot be
   verified in a run) and `stale` (declared but no longer read → the register
   advertises a flag that does nothing). Both are FAIL. It works because every
   flag is read either by `getenv()` on a literal or through helpers that take
   the name as a literal.
6. **Profiles.** `configs/perf/*.json` (22 of them) + `schema.json`.
   `tools/perf_profile.py` subcommands: `validate` (schema + semantics, all
   profiles), `show`, `command` (the full env-prefixed command line),
   `server-env`, `new --like` (a skeleton whose every unmeasured field says
   `unspecified` / `PLACEHOLDER`, because copying a profile turns another
   machine's measurement into a claim about this one), **`forbidden-env`** (keys
   declared with `value: null` must be **absent**, not merely unset — the case
   that matters is `OPENBLAS_NUM_THREADS`), and `check-flags`.
   The control-plane design (`config-control-plane.md`) adds the layering that
   is still open there: `common defaults → Linux CPU server → backend family →
   host topology → experiment override`, one canonical resolver/launcher, and a
   resolved object `{host, server, effective, config_hash, binary_hash,
   model_hash}` per run. The incident behind it: the *correct* GCP profile
   existed, versioned, and three campaigns were run at 2x8 from memory with both
   workers on the same 12 physical cores. "The JSON was right and the experiment
   was wrong" → remove the path that can ignore it.

### 2.2 Minimal concrete proposal for mynah-asr

We already read 11 `MYNAH_ASR_*` names and `OPENBLAS_NUM_THREADS`
(`src/qmat.c`: `MYNAH_ASR_CAPS`, `MYNAH_ASR_QGEMM`; `src/threads.c`:
`MYNAH_ASR_THREADS`; `src/cuda_gemm.cu`: `_CUDA_BF16`, `_CUDA_TF32`;
`src/metal_mps.m`: `_METAL_PROF`; `server/prefork.c`: `_CPU_TOPOLOGY_ROOT`,
`_CGROUP_ROOT`, `_PREFORK_ALLOW_GPU`, `_PREFORK_STRICT`, `_PREFORK_QUEUE`).
None is declared anywhere. Build the registry **now**, while it is 11 rows.

**One table in one file** — `src/flags.c` + `src/flags.h`, owned by nobody else:

```c
typedef struct {
    const char *name;        /* "MYNAH_ASR_QGEMM"                            */
    unsigned    scope;       /* MYNAH_FSCOPE_ARM|X86|VNNI|SMMLA|APPLE|CUDA|ALL */
    const char *deflt;       /* "0" — the compiled default, as a string      */
    const char *desc;        /* one line, operator-facing                    */
    const char *(*inert)(void); /* NULL, or a reason this build ignores it   */
} mynah_asr_flag_t;
```

- `[FLAGS] v=1 pid=… NAME=VAL …` — only what is set, printed once at startup by
  the CLI, by each prefork worker **and by the parent**, to stderr. Copy the
  format verbatim so a future `check-flags` is the same 20 lines of Python.
- `[EFFECTIVE-CONFIG] v=1 build_scope=… flags=N` — one row per **set** flag:
  `name · requested · effective · reason`, plus three non-`MYNAH_ASR_` rows we
  need regardless: `blas.ownership` (engine vs BLAS), `OPENBLAS_NUM_THREADS`
  (present → say it was overridden, per §4 of the design), and
  `threads.effective` (mask actually set, not the one configured). Footer:
  count set, count ignored, and the WARNING line when ignored > 0.
- **Inert predicates are owner-declared**, returning a sentence:
  `MYNAH_ASR_QGEMM` → `"no native int8 GEMM in this build"` when neither
  `MYNAH_ASR_HAVE_SDOT` nor `MYNAH_ASR_HAVE_X86`; `MYNAH_ASR_CAPS=vnni` on a
  CPU without AVX-512VNNI → `"this CPU reaches avx2"`; every `_CUDA_*`/`_METAL_*`
  on a CPU-only build; every `_PREFORK_*` on macOS. **Never** re-derive a row
  from `compiled && supported`.
- `tools/check_flag_registry.py` — copy it almost verbatim (regex the registry
  block out of `src/flags.c`, glob `src/*.c src/*.h server/*.c cli/*.c *.cu *.m`,
  report `undeclared` and `stale`, exit non-zero). Wire it into `make test`
  beside `check_plan.py` and `check_repo_integrity.py`. It costs ~60 lines and
  it is the only thing that keeps §4 of the design ("every env flag read with
  requested | effective | reason") true a year from now.

**Copy also:** `perf_profile.py forbidden-env` semantics (a profile key with
`null` means the variable must be **absent**) — we have exactly the
`OPENBLAS_NUM_THREADS` problem, measured on the Axion in mynah-tts: **63 threads
on an 8-cpu slice with OpenBLAS linked vs 32 without**. And the `new --like`
skeleton rule: unmeasured fields say `unspecified`, never a copied number.

**Skip, and why:**
- **The generated scope header** (`flag_parity.py` + `qwen_flag_scope.h`). It
  exists because ~200 flags are read across 13k-line files. At 11–25 flags the
  scope belongs in the same table as the name, hand-written and reviewed. Adding
  a generator now buys drift risk and a build step for nothing.
- **A 22-profile hierarchy with a schema.** We have one target host family. Two
  profiles (`axion-32c`, `x86-vnni`) + `common-control` is the whole need; grow
  to a resolver only when a *second* host contradicts the first.
- **Diagnostics in the registry.** qwen-tts declares `QWEN_*_TRACE`/`_DEBUG`/
  `_JSON` and then has to say "never in a run that produces a number". Simpler
  here: keep every diagnostic under a single `MYNAH_ASR_TRACE=<subsystem list>`
  so a qualification preflight can refuse on **one** name.
- **`--dispatch-map` as a second report.** qwen-tts has both because they were
  written years apart. We should have **one** report with the five columns
  (`compiled · supported · env · resolved · reason`) and print the flag rows in
  the same table; mynah-tts's `src/dispatch.h` is the better model of the two —
  it names its `resolved` provenance (`[predicate] / [runtime] / [gate]`),
  refuses `compiled && supported` as an answer, counts `UNKNOWN` rows as a
  to-do list rather than hiding them, carries a **drift canary** that compares a
  mirrored compile gate against what the owning TU actually does, and prints an
  **`IDLE HARDWARE:`** footer (units this CPU has and this binary has no kernel
  for — on the Axion it printed 5: `sve sve2 svei8mm svebf16 bf16`). That footer
  is the S5 backlog, generated.

---

## 3. KERNEL STATE PER ISA

`opt` = ISA-specific kernel · `simd` = vectorized but no integer/matrix unit ·
`twin` = fixed-B software fallback · `—` = none · `gated` = present, not default.

| op | AVX2 | AVX-512 (no VNNI) | AVX-512 VNNI | AMX | NEON | NEON+DOTPROD | i8mm / KleidiAI | repo |
|---|---|---|---|---|---|---|---|---|
| int8 GEMV | simd (widen+FMA) | **same AVX2 body** | **opt** (`qwen_matvec_int8`) | not a GEMV path | simd | **opt** SDOT | KAI native GEMV | qwen |
| int8 GEMM B>1 | opt (emulated `maddubs`+`madd`, B2..16) | **falls to AVX2** | opt (row/tile families B2..16) | opt, B≥3 + rows/cols floors | twin | SDOT loop, **opt-in** `QWEN_INT8_SDOT_MM` | KAI persistent RHS/LHS pack, first candidate | qwen |
| Q4 GEMV / GEMM | simd / `maddubs` | same AVX2 | opt / opt | Q4 matmat, B≥4, 32-col floors | simd / twin | SDOT / no matrix-unit Q4 | KAI Q4 packed RHS | qwen |
| BF16 GEMV / GEMM | convert+FMA / twin | convert+FMA / twin | same unless AVX-512-BF16 | AMX BF16 B≥4 | convert+FMA / twin | BFDOT only with the BF16 extension | KAI BFMMLA family | qwen |
| conv1d int8 (decoder) | — (f32 im2col + SGEMM) | — | **opt, default on** | Design-D + fused residual | — | available, **policy-gated** | KAI prepared paths | qwen |
| Q8_0 repack | — | — | — | — | n/a | **opt/gated** (4-row + SDOT) | SDOT GEMV, i8mm B>1 | qwen |
| int8 dot (T ≤ 16) | `dot_q8_avx2` | AVX2 body | **`dot_q8_vnni`** (VPDPBUSD) | — | simd | **`dot_q8_sdot` / `dot_q4_sdot`** | — | **asr** |
| int8 GEMM (T > 16) | dequant + sgemm + `malloc(n·k·4)` **per call** | same | same | — | same | same, `MYNAH_ASR_QGEMM=1` opt-in (lost to Apple AMX) | — | **asr** |
| depthwise conv | scalar triple loop | scalar | scalar | — | scalar | scalar | — | **asr** |
| int8 dot | `dot4_u8_vex` (AVX-VNNI) | — | **`dot4_u8_evex`** (AVX512-VNNI) | **0, no tile ops** | simd | SDOT | **`matvec_q8_pair_i8mm`**, bit-identical to SDOT | tts |
| f16 weight cache | **F16C/AVX2** + scalar half | — | — | — | NEON converts | — | — | tts |
| f32 GEMM | `src/sgemm.c` (own) — **the Linux default, `BLAS=none`** | | | | | | | tts |

**Gating/dispatch design points that matter for S5:**

- **Three predicates, never two.** `compiled` (build gate) · `supported`
  (runtime probe) · `use(gate, B, rows, cols)` (the policy). qwen-tts adds a
  `force()` pin so an in-process paired A/B can pick a leaf deterministically.
  `resolved` is obtained by **calling** the owner's predicate, never re-derived.
- **Target attributes beat build flags.** mynah-tts carries VNNI and i8mm behind
  `__attribute__((target(...)))` so they are compiled into *every* x86/aarch64
  build and chosen by CPUID — which is why `SIMD=avx512` could once imply a
  kernel that did not exist. Our `src/qmat.c` already does this
  (`target("avx512f,avx512bw,avx512vnni")` on `dot_q8_vnni`); keep it for SMMLA.
- **Bit-identity is an architectural requirement, not a nicety.**
  `matvec_q8_pair_i8mm` is bit-identical to SDOT *for the same (row,
  activation)* — otherwise a request's output would depend on who it was batched
  with. For us that is the S2-7 gate: N concurrent streams must produce the same
  transcript as the same streams alone. Integer accumulation makes `==`
  assertable; use `==`, not a tolerance.
- **Crossover is a measured model, not a capability filter.** The legacy audit's
  finding: "there is no measured cost model that asks *for this backend, shape,
  thread count and B, does gather + quantize + matmat beat B GEMVs?*" On M1 the
  int8 batched twin measured **0.70–1.18×** against sequential SDOT. So
  `T ≤ 64 → weight-stationary GEMM` must be a measured threshold per ISA, not a
  constant.
- **Preparation is the serial fraction.** Their activation-quantisation pass was
  flat at **3.8 ms** while the GEMM went 16.7 → 7.7 ms with threads. Put the
  activation quantisation on the pool in S5-1 or the kernel win will not appear.
  Same lesson from the x86 dataflow research: the ranked #1 hypothesis is
  "FP32 gather/staging plus two-pass dynamic activation preparation around each
  small-B projection", above any missing matrix unit.
- **KleidiAI needs both features at build time.** `__ARM_FEATURE_MATMUL_INT8`
  **and** `__ARM_FEATURE_DOTPROD`; a dotprod-only box gets no KAI at all. Treat
  it as an A/B arm (`KLEIDI=1`), promoted only on the soak envelope.
- **Persistent packing is not proven by "the pack happened once."** The open
  oracle is whether the selected hot kernel actually *reads the packed bytes and
  avoids all other layout work*. Prepack also costs resident bytes — measure PSS.
- **Own sgemm vs vendor BLAS is an ownership decision, not a speed one.** 63 vs
  32 threads on an 8-cpu slice; `BLAS=none` also happened to be 9% faster, "but
  that is the tiebreak, not the argument." Keep `BLAS=openblas` as a comparison
  build forever.
- **Threading has a break-even.** mynah-tts measured ~20 µs of wake-up latency
  per parallel region → **a region must be worth ≥200 µs** to pay for itself;
  PocketTTS matvecs are not (int8 1→4 threads: 0.191 → **0.276**). Our S5-5 pool
  meter should use exactly this rule to pick W×T.

---

## 4. mynah-tts / PocketTTS server ideas not already in `serving-v2-design.md`

Each with the note that carries it.

1. **`stream_out` as a single-byte ring**, not a list of malloc'd chunks: one allocation at start, PCM conversion into reused scratch *outside* the lock, memcpy into the ring under it, writer writes from the ring *without* it — safe because `queued` only shrinks after a write completes (ThreadSanitizer-validated); it also coalesces a backlog into one frame. → `streaming-server-v2.md`.
2. **Why the slow reader was unbounded**: the old `write_all` restarted a fresh `SO_SNDTIMEO` on every partial send, so a *slow* (not stopped) reader stalls forever — 66.50 s measured; async writer 4.81 s at 1 MiB, **2.64 s at 64 KiB** because it reaches the cancel path. Queue size changes the answer. → same note.
3. **An aborted stream is never silent**: log byte count, queue high-water mark, refusal count — that is what separates "client left" from "we dropped it". → same note.
4. **`job_claim_fd` as an `atomic_exchange`**: first caller gets the fd, everyone else gets −1; the defence against a second close landing on a number `accept()` has since reused. → `streaming-server-v2.md` §E5-1.
5. **Exact route matching**: `strncmp` on the request-line prefix means `POST /v1/audio/streamXYZ` matches, and once the fd changes owner that is an fd nobody closes. Port an `http_precheck`. → risk 7.
6. **A wall-clock deadline separate from the step budget**: `max_steps` bounds steps, not time; with continuous admission a pathological slot holds its share forever. → risk 9.
7. **A distinct "cancelled" outcome**: two uncoordinated minimum-length thresholds made a cancel-before-minimum report as a *synthesis error*. → risk 11.
8. **`after_fork` is a prerequisite**: a `PTHREAD_ONCE_INIT` pool in a forked child believes it is initialised, inherits `workers > 0` with no live workers, broadcasts to nobody and waits forever; fork only from the main thread before any inference. → risk 2.
9. **A `trylock`-based parallel-for degrades to serial silently** — it looks like "the kernel got slow". Corollary they state: do not parallelise slots *above* `parallel_for`. (We have the same defect.) → risk 1.
10. **Any per-thread cache keyed by `pthread_t`** forces the constraint that threads entering the model are persistent and fixed in number, or it leaks per dead thread. → risk 4.
11. **Scratch sized on the initial batch is the silent failure**: their `batch_scratch_init` returned early at `batch ≤ 1`, and the NULL it left turned into a *silent fallback to the per-row path, losing all batching*. Size from `model->info`, allocate once for the maximum. → step 2.
12. **A chunk constant does not survive a second model**: `STREAM_EMIT_FRAMES 16` = 744 ms at 21.5 fps but **1.28 s** at 12.5 Hz. Keep the quantum derived from the model pack. → step 6.
13. **Warm prefill is starved of threads, not of parallelism**: 11.3 ms genuinely serial and flat in core count, projections divide **14×**. → `prefill-ttfa.md`.
14. **Weight packing is a `serve()`-entry cost, not a request cost**: 215 ms constant on the first call under one mutex, invisible in any steady-state profile — and **warm up before the fork**, or every worker builds its own 151 MB copy. → same note.
15. **Model/language as a routing key validated before admission** (unknown → 400, not a 500 mid-stream); the scheduler iterates **over groups, not slots**, because every switch re-reads the full weight set (≈110 MB int8 at ~60 GB/s ≈ 2 ms, the order of a whole frame) — hence one process per language as the default. → `streaming-server-v2.md` §multi-language.
16. **Count allocations before building an arena**: their LD_PRELOAD census found **2,786 per request, constant over an 8× range in frames**, 3 mmap / 2 munmap, so the reference's arena item was closed as *not-applicable*. Ours is per chunk, so it is real — but S0-4 should produce the same constant-count check. → `linux-build-and-dispatch.md` §E4-18.
17. **Serving-profile hygiene to copy verbatim**: refuse to guess a sample rate; a non-streaming route is **INCONCLUSIVE**, never GOOD; incomplete levels print `1/4 completed` and exit non-zero instead of averaging over survivors; report the coalesced-read fraction (~85% on loopback) so cadence reads as an upper bound; print p50/p95/mean/**sd**/min/max with the sample count — their C4 TTFA sd was **2246 ms against a mean of 2896**, i.e. bimodal, which no percentile pair shows. → `streaming-server-v2.md` §how this gets measured.
18. **Two playback-simulator bugs inherited from qwen-tts and fixed there**: a jitter buffer smaller than the first chunk absorbs nothing, and `stall_count` counted a persistently-late stream as *one infinite stall* instead of one episode per gap. → same note, §cadence law.
19. **The mutex baseline is the model of how to publish a "before"**: C1/C2/C4 TTFA p95 **110 / 1847 / 5596 ms** while STREAM_RTF never moved (0.313–0.345) and prebuffer/stall stayed at zero — "the mutex shows up exactly where the cadence law says it will, and nowhere else." → same note, §baseline measured.

## 5. TEN RECOMMENDATIONS FOR THE v2 IMPLEMENTATION ORDER

1. **Land the flag registry + `[FLAGS]` + `[EFFECTIVE-CONFIG]` + the registry
   check before S2 code.** Evidence: `QWEN_NO_SIMD_QUANT` was inert on ARM and
   nothing said so; the correct GCP profile existed while three campaigns ran on
   the wrong topology from memory. At 11 flags this is an afternoon; at 200 it is
   a generator.
2. **Then the effective-config banner including `blas.ownership` and the
   *actual* mask.** Evidence: 63 threads on an 8-cpu slice with OpenBLAS linked
   vs 32 without, measured on the Axion. Print the mask set, not configured.
3. **Build the scheduler as the single thread that enters the model, and assert
   it.** Evidence: `QWEN_DECODER_THREAD` — C4 STREAM .847→**1.296**, TTFA
   174→1126 ms, context switches ×2.9, cores flat. A second submitter pays
   `submit_mtx` and loses the cohort.
4. **Admission: keep the listener polled and refuse at the parent with a status.**
   Evidence: gating the listener hid a **4.47 s** TTFB p95 before `accept()`;
   with `--max-queue 0` the same overload produced 8 accepted at TTFB p95 187 ms
   and 2 honest 503s.
5. **`stream_out` before the batched step.** Evidence: 66.50 s → 4.81 s → 2.64 s
   for the second client; and output isolation cost nothing in KPI (C4
   .821→.830) while proving slow-reader isolation with byte-identical audio.
   Implement it as the single-byte ring with cancel-not-block backpressure.
6. **Allocation-free step next, but only after S0-4 counts the allocations.**
   Evidence *for*: our 55–60 malloc/free per chunk sit around an allocation-free
   encoder step. Evidence *against overreach*: ragged scratch reuse was
   REJECTED (C4 .810→.843) and mynah-tts closed its arena item as
   not-applicable at a constant 2,786 allocs/request. Carve per-slot scratch at
   slot open; do not build a per-thread cache.
7. **Make the within-worker batched step the S2 centrepiece — this is the one
   place we differ.** Evidence: F3 measured only **2.7%** useful cross-worker
   coincidence within ±1 ms for TTS, which killed global batching there; our
   chunks arrive on the real-time grid, so at the 320 ms preset every live
   stream has a chunk ready every step. Gate it with byte-identity (N streams
   together == N streams alone), asserted with `==` on integer accumulations.
8. **Choose the operating point on cadence percentiles and fix the quantum to
   the preset; never let load change it.** Evidence: chunk sweep prebuf p95
   0.450 → 1.222 s while STREAM p95 moved only .919 → .880; q8 at C4 had the
   best RTF (.817) and a **50% stall@250**; the hard lead gate turned a .838
   server into a .986 one by clamping production to playback.
9. **Do not implement a decoder lane until an S0/S4 profile blames the RNNT
   loop, and then default it off.** Evidence: DL-1 removed the decoder from the
   critical path perfectly (iteration wall p95 192→77 ms, stall@250 100%→0% at
   B4) and **still missed its gate** because the step side on 4 threads inflated
   +27–29%; 5+3 and 6+2 were worse; the elastic arm proved the tax is L3
   pollution during overlap, unfixable by NTA prefetch, hot workers, sub-quantum
   units or direct conv paths. Our equivalent risk: a lane for the greedy loop
   that steals threads from the batched encoder step.
10. **S5 kernels last, and start with the weight-stationary int8 GEMM for
    T ≤ 64 with the activation quantisation on the pool.** Evidence:
    `amx_request_wall_share` ≈10–20% → infinite matrix-unit speed saves ≈15% of
    wall; the ranked #1 x86 inefficiency is activation staging/preparation, not
    a missing unit; their activation-quant pass was flat at 3.8 ms while the
    GEMM threaded 16.7→7.7 ms. Corollaries: never expand a strided input into a
    zero-filled panel (ConvT one-GEMM, +14–32%); prove the crossover per ISA
    (M1 measured 0.70–1.18× for the batched twin); and let the `IDLE HARDWARE:`
    footer generate the backlog — on the Axion it names `sve sve2 svei8mm
    svebf16 bf16` while our binary uses SDOT only.

---

Sources (repo-relative). qwen-tts `.work/`: `c12-win-{track,checkpoint,step1-3,bf16-preup,convt-one-gemm,glue-vnni,forensic-audit}-20260909.md`, `c12-win-admission-slicing-20260910.md`, `c12-win-{conv-stack,vnni-glue}-implementation.md`, `p2-cross-backend-runtime.md`, `p3-runtime-knob-parity.md`, `p4-{fused-residual,prefork-admission-bound,same-pool-decoder,rag-claim-first,rag-panel-scratch}-20260907.md`, `f1-fused-quantum-20260908.md`, `f3-cross-worker-cohort-coincidence-20260908.md`, `decoder-quantum-floor-20260907.md`, `ls4-utilization-aware-admission-20260908.md`, `dl1-decoder-lane-split-20260909.md`, `post-p4-evidence-synthesis-20260908.md`, `professional-streaming-architecture.md`, `config-control-plane.md`, `x86-dataflow-research.md`, `arm-linux-v2-parity-implementation-20260911.md`, `cross-isa-operational-parity-20260908.md`, `legacy-cpu-v2-audit-20260916.md`. qwen-tts code: `qwen_flag_scope.h`, `qwen_tts_dispatch.c:1-60,150-280`, `qwen_tts_kernels.c:297-390`, `qwen_tts_thread.c:887-910`, `qwen_tts_kleidi.c:1239-1255`, `tools/check_flag_registry.py`, `tools/perf_profile.py:200-345`, `configs/perf/`, `docs/BENCHMARKING.md`. mynah-tts `.work/`: `streaming-server-v2.md`, `serving-doctrine.md`, `pocket-tts-engine.md`, `pocket-tts-vs-mynah.md`, `decode-gang.md`, `prefill-ttfa.md`, `dtype-and-fallbacks.md`, `linux-build-and-dispatch.md`, `cpu-kernels-arm-x86.md`; code: `src/dispatch.h`, `Makefile:21-98`.
