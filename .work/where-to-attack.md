# Where to attack next: Nemotron first, Parakeet second

Status: OPEN (written 2026-09-19 from the Axion step table and local measurements;
the ranked list below is a plan, not a result)

Task: input to S4-3, S5-2..S5-5, M-4, and a new S8 (the advisor)
Question: given `a = 31.7 ms` and `b = 4.58 ms` on the Axion, where does the next
factor of two come from — and what should a new box be told to run?

## 0. The one number everything hangs off

`T_step(B) = a + b·B`, with `a = 31.7 ms` FIXED and `b = 4.58 ms` per stream
(Axion, int8, one 32-thread process, DIAGNOSTIC). Two consequences that are not
obvious and that shape every item below:

- **`a` is 12 % of a 320 ms chunk period, paid whether the worker serves 1
  stream or 49.** It is the cost of walking the model once. `b` is what an extra
  stream costs once the weights are already being walked.
- Therefore **every worker in a prefork fleet pays `a` again.** Eight workers
  spend `8 × 31.7 = 254 ms` of each 320 ms period on fixed cost before serving
  anyone. This is the opposite of the sibling TTS intuition, where a request is
  compute-bound and narrow workers are free.

That asymmetry — a big fixed cost amortised over B — is the shape of a
**memory-bound** step: the int8 weights are 0.79 GB, far past this box's 80 MiB
L3, so every step streams them from DRAM once and reuses them across the B rows.
HYPOTHESIS, not yet measured: `a` is dominated by weight traffic. It is cheap to
falsify (below) and almost everything else follows from whether it is true.

## 1. Ranked attacks

### A1. The server's default quantisation is f32, and that is wrong for CPU serving

`server/main.c`: `static int g_quant = MYNAH_ASR_QUANT_F32;`

Measured locally (M1, 8 threads, `tests/audio/test_en.wav`, explicit language):

| quant | RTF |
|---|---|
| f32 | **1.528** |
| int8 | **0.682** |

int8 is **2.24× faster**, and f32 does not sustain even ONE real-time stream on
that host. The quality cost is already documented and small
(`docs/quantization.md`, 34 locales, 102 samples): mean CER 0.133 → 0.145,
identical on ~90 % of languages, one genuine regression (ro-RO, already
borderline at f32).

So the shipped default is a configuration in which a single stream does not run
in real time on a development machine — and every capacity number in this repo,
including the 49 from the Axion, is an **int8** number. The default and the
measurement disagree. This is a one-line change with a large effect and it is
the first thing to do.

### A2. Lookahead is a capacity knob and nobody is treating it as one

`P = (lookahead + 1) × encoder_frame_ms`, so at lookahead 3, `P = 320 ms`. The
cadence law says capacity is `(ρP − a) / b`:

| lookahead | P | B_max at ρ=0.8 (Axion numbers) |
|---|---|---|
| 0 | 80 ms | 7 |
| 1 | 160 ms | 21 |
| 3 | 320 ms | **49** |
| 6 | 560 ms | 91 |
| 13 | 1120 ms | 189 |

**Going from lookahead 3 to 6 nearly doubles capacity for 240 ms more latency.**
That is a product decision, not a tuning detail, and it belongs in the advisor's
output as an explicit trade rather than buried in a preset index. Note the shape:
capacity is linear in P but `a` is subtracted first, so the gain per unit of
latency is largest at small P — the jump from lookahead 0 to 1 triples capacity.

### A3. Fewer, wider workers — and the arithmetic says so

Because every worker pays `a`, the fleet's aggregate fixed cost is `W · a` per
period. Capacity is `W · (ρP − a) / b`, which is maximised by making W SMALL,
right up to the point where one worker can no longer use all the cores. The
counter-pressure is real (a wide worker has more pool contention, one wedged
worker loses more streams, NUMA), which is exactly why it is a sweep — but the
sweep should start at **W=2, T=16** and work down in width, not at W=8, T=4.

This prediction is falsifiable and cheap: run the same WAVE at 2×16, 4×8 and
8×4 and compare capacity at equal envelope. If capacity is flat in W, `a` is not
what I think it is.

### A4. Test the memory-bound hypothesis before optimising for it

Three cheap experiments, in order:

1. **`a` versus thread count.** If `a` is weight traffic, it will not improve
   much from 8 to 32 threads. If it halves, it is compute and A3 is wrong.
2. **int4 encoder weights.** Halves the bytes. If `a` nearly halves, the
   hypothesis is confirmed and A5 becomes the main line. `docs/quantization.md`
   already says int4 hurts multilingual quality (mean CER 0.227) — this is a
   measurement of the MECHANISM, not a proposal to ship int4.
3. **`perf stat -e` DRAM counters for one step** on the box, which answers it
   directly rather than by inference.

### A5. If it is memory-bound: mixed precision, weight layout, and prefetch

Only worth doing once A4 says so, and in this order:

- **Mixed precision by tensor role.** The FFN weights are the bulk (16.8 M of
  the 23.1 M MACs per layer per frame) and are the most redundant; attention
  projections are small and sensitive. int4 FFN + int8 attention could take a
  large bite out of `a` at a fraction of int4's quality cost. This has never
  been tried here and the quantiser already supports per-tensor choices.
- **Walk order.** The step visits 24 layers in sequence, each streamed once.
  There is nothing to reorder within a step, but ACROSS the B streams the
  current code already stacks rows — which is why `b` is small. Nothing to win
  here; noted so nobody spends a week on it.
- **Prefetch / pre-faulting.** The weights are mmap'd. A first-touch pass at
  load (or `MAP_POPULATE`) removes page faults from the first steps; huge pages
  remove TLB pressure from every step. On a 0.79 GB working set with 4 KiB pages
  that is ~200 k TLB entries against a few hundred with 2 MiB pages. Cheap to
  try, and the box doctor already reports whether hugepages are available.

### A6. SMMLA does not cover the small-T path

From the Axion counters at B=8: `dot 2509` (SDOT, the T ≤ 16 branch) beside
`dot_rows 3840` (SMMLA). At B=1–2 most of the work goes through SDOT, because
the 2×2 SMMLA tile needs at least two rows. That is exactly the low-concurrency
regime where latency matters most. Worth measuring how much of `b` at small B is
this.

### A7. The RNNT prediction network is f32 and per emitted token

`src/decoder.c` runs the LSTM through `mynah_asr_gemv_f32`: 2 layers × 2 ×
(4H × H) at H = 640 ≈ 3.3 M MACs per emitted token, in f32, on the path that
runs once per token per stream. The joint head is already int8. Quantising the
pred-net is unexplored and it is the only remaining f32 weight matrix in the
streaming hot path.

## 2. Parakeet: lighter, and that does not mean more streams

The arithmetic, from the two `mynah.json` files:

| | Nemotron 0.6b | Parakeet-110m |
|---|---|---|
| layers × d_model × ffn | 24 × 1024 × 4096 | 17 × 512 × 2048 |
| encoder MACs / frame | 553.6 M | **98.0 M** (5.65× less) |
| joint head MACs / token | 8.38 M | 0.66 M (12.8× less) |
| cache-aware streaming | **yes** | **no** |

At the same cadence a Parakeet-shaped encoder would hold ~277 streams where
Nemotron holds 49. **But Parakeet is not cache-aware**, so streaming it means
re-encoding a window every chunk (`.work/multi-model-streaming.md` §4):

| buffering | re-encode cost | net vs Nemotron |
|---|---|---|
| left 10 s, chunk 2 s, right 2 s (NeMo default) | 7.0× | **0.81× — WORSE** |
| left 6 s, chunk 2 s, right 1 s | 4.5× | 1.25× |
| left 4 s, chunk 2 s, right 1 s | 3.5× | 1.61× |

So the headline is: **at NeMo's own buffering defaults, the model that is 5.65×
lighter is a WORSE streaming server than Nemotron.** The 5.65× is spent, and
more, on re-encoding context that Nemotron keeps in a cache. Aggressive
buffering claws it back to ~1.6×, at an accuracy cost nobody here has measured —
and parakeet-tdt-0.6b-v3 (24 layers, d_model 1024) has no advantage to spend at
all.

Where Parakeet IS the right answer, unchanged: **offline and batch**, where a
file is fed faster than real time and there is no re-encoding penalty. There it
is 5.65× cheaper and the fleet already serves it today. The honest position is
that Nemotron is the streaming product and Parakeet is the throughput product,
and the advisor should say which one a given workload wants.

## 3. What the advisor must do (new: S8)

The sibling's doctor "often gets it right" because it does not guess a machine's
constants — this one must not either. The design that follows from everything
above:

1. **Calibrate, do not assume.** Run the step table at B = 1, 2, 4, 8 on the
   actual box and fit `a` and `b`, discarding B=1 as warm-up. Thirty seconds,
   and it is what makes the prediction machine-specific instead of a spec-sheet
   reading.
2. **Predict**, per quantisation and per lookahead, with the arithmetic shown
   and labelled a prediction: `B_max = (ρP − a) / b` per worker, `W · B_max` for
   the fleet.
3. **Recommend** a starting topology (W × T with W small, per A3), `--cap`,
   `--quant int8`, the lookahead that meets the caller's latency budget, and
   the descriptor/backlog limits the C100 work already derives — as a
   ready-to-paste command line.
4. **Refuse** when an input is missing rather than substituting a default: no
   dispatch map, no prediction; a loaded box, no calibration.
5. **Close the loop**: a WAVE or SOAK artifact carries the measured capacity, so
   the advisor can print `predicted 49, measured 38` next time. A predictor that
   never learns it was wrong is a horoscope.

Evidence: the Axion step table (`.work/axion-session-2026-09-19.md` §5), the
local f32/int8 RTF pair above, `docs/quantization.md` §CER regression, the two
`mynah.json` files.
Conclusion: A1 is a one-line change with a 2.24× effect and should land first;
A2 is the biggest single lever and is a product decision; A3 and A4 are one
experiment each and they decide whether A5 is worth a week.
Next action: A1, then the advisor's calibration step, then A4 on the box.

---

## 2026-09-19 (later) — the advisor calibrated itself, and the M1 constants moved

`tools/bench/box_advisor.py` ran its own calibration on the development Mac:

```
[CALIBRATED] a = 42.4 ms fixed, b = 12.00 ms per stream
             (fit over B=2..8, B=1 discarded: warm-up)
             worst residual 3.6 ms (4 % of the mean step)
step rows    B=1 106.59 | B=2 63.19 | B=4 90.58 | B=6 111.71 | B=8 137.57
```

Two things worth recording.

**B=1 is anomalously high on both machines** — 106.59 ms here against 63.19 at
B=2, exactly the pattern the Axion showed (88.11 vs 37.75). That is the first
step paying the first touch of the weights, and it is why the fit discards it.
Two machines showing the same shape is what turns a rule of thumb into a rule.

**`b` halved against the note from the previous day.** `.work/stream-api-v2.md`
records `a = 37.8, b = 20.3` for the same Mac on 2026-09-18; today the same tool
on the same machine fits `b = 12.00`. The tempting story is that the week's work
— the DOT register tile, the derived panel width, the parallel threshold, the
spin-then-park pool — landed on the int8 step. It is probably part of it. But
yesterday's pair was taken UNDER THIRD-PARTY LOAD, best-of-4 interleaved, and
today's is a single run at loadavg 1.95, so **the two were not measured under
comparable conditions and neither supersedes the other**. Both are in the
transferred table with their conditions attached, and the discrepancy is
recorded rather than resolved by choosing the flattering one.

**The prediction for this Mac is 18 streams at lookahead 3, and the only server
evidence says >= 8.** The 2026-09-18 WAVE held 8 streams at emission lag p95
258 ms, inside the 320 ms gate, with no ceiling established. So 18 is unverified
and probably optimistic: the step table carries no ingest thread, no WebSocket
framing, no writer and no admission. That gap — between what the compute can do
and what the server delivers — is the single most useful thing the first WAVE
will measure, on either machine.
