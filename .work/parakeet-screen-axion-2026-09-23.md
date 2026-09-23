# Parakeet TDT 110m — fast characterisation on Axion, 2026-09-23

Status: **DONE 2026-09-23** (screen only — nothing here promotes a production
concurrency, and by design nothing here was run long enough to)

Task: M-6
Question: the fleet has served exactly one model in anger. What does the other
supported family actually cost and deliver on the same box, the same corpus and
the same scorers — and is its serving semantics even the same question?

## Known facts before measuring

- **Parakeet does not stream, and this is enforced in code, not in a doc.**
  `mynah_asr_stream_unsupported()` refuses every converted Parakeet pack
  (linear biases, folded batch_norm, symmetric conv padding) and the server
  declines the WebSocket upgrade with `400 model_not_streaming` (M-1, M-4).
  The pack confirms it independently: `att_context_style: "regular"`, i.e.
  non-causal full attention.
- Therefore **"concurrency" means a different thing for the two models.** For
  Nemotron it is "how many real-time streams does this box hold"; for Parakeet
  it is "how many files at once before latency stops being worth it".
  `tools/bench/rest_load.py` exists for precisely that distinction and its own
  docstring states it.
- **Any comparison must therefore be labelled, never merged.** A Parakeet
  throughput number and a Nemotron concurrency number are not the same units.

## Provenance (A1)

| | |
|---|---|
| commit | `5b934b7` (clean tree, clean rebuild, `build/` removed first) |
| binary | `mynah-asr` sha256 `37df098d…`, `mynah-asr-server` sha256 `bc168f31…` |
| dispatch | `blas=own simd=neon+dotprod+i8mm`, int8 dot `neon-sdot`, int8 rows `neon-smmla`, **0 rows UNKNOWN** |
| host | ARM Neoverse-V2, 32 vCPU, 1 thread/core |
| Parakeet pack | `parakeet-tdt_ctc-110m`, `arch fastconformer_tdt`, engine `parakeet-tdt`, 17L x 512, TDT head (`durations [0,1,2,3,4]`), vocab 1025, **weights imported from a Q4_K_M GGUF** |
| Nemotron pack | `nemotron-3.5-asr-streaming-0.6b`, `--quant int8`, preset `[56,3]` |
| corpus | the qualification's own `bank.txt`: 498 clips, 6585.2 s, seeded class-balanced sample of `samples/stress-en`, identified by per-clip sha256 |
| affinity | proven per process with `taskset -cp`, six workers on disjoint 5-cpu slices inside 0-29, generator on 30-31 |

Both models were scored by **the same functions** — `wer`, `cer`,
`wer_format_free`, `cer_format_free`, `normalise` imported from
`tools/bench/streaming_metrics.py`, never reimplemented — on **the same 498
clips**, in the same session, on the same box.

## Evidence — quality (A2)

Offline whole-file for both, so the two offline columns are a true A/B; the
streaming column is the frozen C=80 run's unloaded reference transcripts,
**scored, not recomputed**.

| | Parakeet TDT 110m, Q4_K_M, offline | Nemotron 0.6b, int8, offline | Nemotron 0.6b, int8, **streaming [56,3]** |
|---|---|---|---|
| clips scored | 498 | 498 | 498 |
| errors / empty transcripts | 0 / 0 | 0 / 0 | 0 / 0 |
| WER mean | **0.0916** | 0.1045 | 0.1063 |
| WER p50 | **0.0556** | 0.0714 | 0.0714 |
| WER p95 | **0.3333** | 0.3214 | 0.3333 |
| WER corpus-weighted | **0.1306** | 0.1443 | 0.1462 |
| WER format-free, mean | **0.0641** | 0.0759 | 0.0774 |
| CER mean | **0.0622** | 0.0674 | 0.0688 |
| CER format-free, mean | **0.0303** | 0.0349 | 0.0357 |
| wall for 6585 s of audio, 24 x 1 thread | **26.3 s** | 135.9 s | — |

Split by provenance, so a synthetic concatenation is never read as original
material (`composed` in the manifest): **FLEURS original** (349 clips) WER mean
0.0907 Parakeet / 0.1014 Nemotron; **synthetic stress-bank concatenations**
(149) 0.0937 / 0.1116. The stress bank does not distort the comparison.

By length class, Parakeet WER mean: short 0.0686, medium 0.0946, long 0.1117.

**Thread invariance was gated, not assumed.** The screen runs clips in parallel
at one thread each; 8 clips per model were re-run serially at full width and
required to be byte-identical. 0 mismatches in both models (rule 4).

### RESULT — a corpus artefact found while reading the worst rows

`long_0021_f708.wav` is **not** a synthetic concatenation (`fleurs_ids: [708]`,
one `source_file`) yet its 30.48 s of audio contains the same sentence twice,
while the FLEURS reference carries it once. Both models transcribe both takes
and both are charged WER > 1. This is a **retake inside the original FLEURS
recording**, i.e. a reference artefact, and it sets a floor on the absolute WER
of this bank's long class for every model. It cancels in the comparison, since
both models see the same clips, and it is the reason the corpus-weighted WER
(0.131 / 0.144) sits well above the per-clip median (0.056 / 0.071).

## Evidence — offline capacity (A3)

Fresh six-worker fleet per rung, 5 threads each, cpus 0-29, generator pinned to
30-31, 60 clips, affinity proven per rung. `xRT` = audio seconds accepted per
wall second. Latency is submit to final JSON.

| C | requests | refused | errors | wall s | xRT | lat p50 ms | lat p95 ms | cores measured | RSS total MB |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 60 | 0 | 0 | 21.89 | 40.3 | 283 | 802 | 3.78 | 1249 |
| 4 | 32 | 0 | 0 | 5.35 | 148.2 | 603 | 859 | 13.84 | 1323 |
| 8 | 64 | 0 | 0 | 7.91 | 198.4 | 614 | 1423 | 18.67 | 1453 |
| 16 | 128 | 0 | 0 | 15.05 | 207.3 | 1361 | 2708 | 18.79 | 1865 |
| 32 | 256 | 0 | 0 | 21.02 | 227.3 | 2529 | 3161 | 19.82 | 2251 |

**FACT — C=1.** One request at a time costs 283 ms median, 802 ms p95, and runs
at 40x real time using **3.78 of the 5 threads** its worker owns. Resident set
is ~208 MB per worker with the pack loaded.

**FACT — throughput saturates between C=4 and C=8 and latency pays after it.**
xRT gains 3.7x from C=1 to C=4, a further 34 % to C=8, then **4.5 % to C=16 and
9.7 % to C=32** — while median latency goes 614 -> 2529 ms, 4.1x, over the same
stretch. There is no rung in this ladder where the fleet failed: **0 errors and
0 refusals everywhere**, so the ladder found a *knee*, not a *limit*.

**FACT — cores plateau far below the slice.** Measured cpu-seconds over wall put
the fleet at 18.7-19.8 cores of the 30 it is pinned to, from C=8 upward, and
adding concurrency does not recover them. This is the **same shape as S12-7c on
Nemotron** (23.1 of 30 at saturation, falling under overload) on a different
model, a different decoder family and a different code path — offline
weight-stationary rather than cache-aware streaming. That makes a
Nemotron-specific explanation less likely; it does not identify the cause.

## Conclusion

Labelled, and deliberately not a verdict on which model is better:

- **Serving semantics are not comparable.** Nemotron is cache-aware real-time
  streaming, qualified at C=80 concurrent live streams. Parakeet is whole-file
  offline; it has no streaming mode here and the runtime refuses to pretend.
- **On this English corpus Parakeet 110m is the more accurate model** — WER mean
  0.0916 against 0.1045 offline / 0.1063 streaming, and the format-free figures
  keep the ordering — **while being 5.2x faster and a fifth of the parameters.**
  The obvious caveats are that Parakeet 110m is English-only against Nemotron's
  40 locales, and that the two carry different quantisation schemes (Q4_K_M
  against int8), neither of which this screen isolates.
- **Streaming costs Nemotron almost nothing in quality**: mean WER 0.1045
  offline against 0.1063 streaming on identical clips, +0.0018.
- **What each is for, on this evidence**: Parakeet for batch/asynchronous
  English transcription where latency is a throughput question; Nemotron where
  a caller needs words while the speaker is still talking, or a language other
  than English.

## Next action

None for Parakeet. This is a screen: rungs are 8-60 requests, not soaks, and
nothing here may be promoted. If Parakeet ever becomes a product lane the
sequence is the one S12 used — freeze, register bounds, discovery ladder,
then two independent long soaks — and not this note.
