# CUDA quantised arms: int8/bf16 K/V ring and int8 weights

Status: IN PROGRESS (2026-10-08; code, unit gates, batch identity and the bank
WER done; serving A/B not run)

Task: S14-8
Question: can the per-stream state and the weights of the CUDA engine be
stored in fewer bytes, so more streams fit and each pass reads less, without
breaking batch identity (contract 4) or the bank WER bound (+0.002 corpus)?

## Known facts

- The per-slot K/V ring is `[24 layers][K|V][56 positions][1024]` f32 =
  10.5 MiB per slot, ~93 % of a slot's arena (conv cache 0.75 MiB, the rest
  is KB). At `--cap 128` the rings alone are 1.31 GiB.
- The GEMMs are ~75 % of device time at C=128 (the S14-6b stage profile);
  the attention core ~12 %.
- The CPU path already ships per-row symmetric int8 weights
  (`docs/quantization.md`, `mynah_asr_quantize_int8`).

## What was built (all default off)

- `--kv-dtype bf16|int8` (`gpu/cuda/kernels.cu`): ring stored bf16 (RNE) or
  int8 with one f32 scale per (position, head), `scale = max|x| / 127`,
  `code = rint(x / scale)` clamped to ±127 — the Pocket CUDA engine's int8 KV
  record, scales kept in a separate plane. Fresh rows are attended in f32 and
  quantised once on commit (one warp per row and head, fixed-order max); the
  attention kernel is templated on the storage and reads `code * scale`.
  Per slot: bf16 5.25 MiB, int8 2.71 MiB (-74 %).
- `--weights int8` (`gpu/cuda/gemm_w8.cu`): the 24 layers' FFN, q/k/v/o and
  pointwise-conv linears and the joint head as the CPU's per-row int8 codes
  (an int8 pack's own, else quantised at open with the library quantiser).
  The kernel is v1's fixed-order fma chain over `(float)code`, times the row
  scale before the epilogue: row-stable by construction. Subsampling, prompt
  projector, LSTM, depthwise stay f32.

## Evidence (NVIDIA L4, sm_89, CUDA 12.8, Nemotron 3.5 streaming 0.6B, gemm own v1)

Unit (`tests/test_cuda_kernels`): PASS, 0 failures. Attention + commit +
advance for f32/bf16/int8 rings against a host codec (int8 codes and scales
bit for bit, lane alone == lane in cohort byte for byte). w8 GEMM on all 12
model shapes: row-stable over 12 cohort sizes byte for byte; arithmetic error
vs double on the decoded weights no worse than v1; quantisation error vs the
f32 weights 0.003-0.013 absolute on outputs up to 0.6-3.3.

Batch identity (`tests/test_cuda_stream`): gate A PASS on every arm (5 test
clips alone == cohort, slot independence); f32 gate B (CPU == GPU) still PASS.
On the bank subset: the first 16 clips alone == cohort and all 227 clips
identical on a shifted slot, every arm.

Bank WER: `samples/stress-en`, every 7th manifest clip (227 clips, short,
medium, long), lookahead 3, `gpu/tools/transcript_ab.py` vs the f32 arm:

| arm | identical to f32 | corpus WER | utt-mean WER | CER mean | delta vs f32 | gate (+0.002) |
|---|---|---|---|---|---|---|
| f32 | 227/227 | 0.43453 | 0.11496 | 0.07598 | — | — |
| kv bf16 | 227/227 | 0.43453 | 0.11496 | 0.07598 | +0.00000 | PASS |
| kv int8 | 224/227 | 0.43473 | 0.11518 | 0.07603 | +0.00020 | PASS |
| w8 | 167/227 | 0.43302 | 0.11243 | 0.07417 | -0.00152 | PASS |
| w8 + kv int8 | 168/227 | 0.43322 | 0.11263 | 0.07427 | -0.00131 | PASS |

(The corpus WER is dominated by the composed long clips, the same for every
arm; the deltas are what the gate reads.)

Pass bench (`tests/test_cuda_stream --bench C --steps 40`, 8 long clips
cycled over C lanes, all C lanes in every pass, steady chunks after 2 warm
steps, `--profile-stages` events; 38 full passes per arm; one short run per
arm on a shared box, so differences under ~2 % are noise). Every lane in one
pass is heavier than a serving cohort at the same C (the 40 ms cohort policy
splits lanes over passes), so read the deltas, not the absolute ms:

| arm | C | weights MiB | arena MiB | VRAM used at ready MiB | device ms / pass | vs f32 | ffn1+ffn2 | att_core |
|---|---|---|---|---|---|---|---|---|
| f32 | 64 | 2351 | 1199 | 3786 | 90.5 | — | 42.6 | 16.8 |
| kv int8 | 64 | 2351 | 700 | 3288 | 89.5 | -1.1 % | 42.3 | 16.5 |
| w8 | 64 | 673 | 1199 | 2108 | 80.8 | -10.7 % | 37.2 | 16.2 |
| w8 + kv int8 | 64 | 673 | 700 | 1610 | 80.0 | -11.7 % | 36.9 | 15.8 |
| f32 | 128 | 2351 | 2397 | 4988 | 156.4 | — | 73.5 | 33.1 |
| kv int8 | 128 | 2351 | 1400 | 3992 | 156.4 | 0.0 % | 73.8 | 32.6 |
| w8 | 128 | 673 | 2397 | 3310 | 143.0 | -8.5 % | 64.7 | 32.9 |
| w8 + kv int8 | 128 | 673 | 1400 | 2314 | 139.7 | -10.7 % | 63.5 | 31.7 |

VRAM per stream: the arena grows by 18.7 MiB per slot at f32 between C=64
and C=128 (ring 10.5 + conv cache 0.75 + per-lane scratch, Bmax tracks the
cap up to 128); with `--kv-dtype int8` by 10.9 MiB (-7.8 MiB per slot, the
ring's 10.5 -> 2.71). Above `--cap 128` the scratch stops growing and the
per-slot cost is the state alone: ~11.3 MiB f32 vs ~3.5 MiB with an int8
ring (-69 %).

## Conclusion

- `--kv-dtype int8` does what it is for: the per-slot state drops 69 %
  (ring 10.5 -> 2.71 MiB), WER within the gate (+0.0002, 224/227 clips
  byte-identical), batch-invariant. It does NOT make the pass faster on the
  L4: the attention core is not bound by the ring's bytes (it reads 56 rows
  per lane per head from L2-friendly records; the kernel is latency bound),
  so -1 % / 0 % device time. Its value is capacity in VRAM: more slots per
  card, which matters once the GEMMs stop being the ceiling (bf16 tensor
  cores) or on a card where `--cap` is VRAM-limited. bf16 ring: identical
  transcripts on all 227 clips, half the bytes; the safe choice if int8's
  three changed clips matter.
- `--weights int8` is the arm that moves device time: -8.5 % to -10.7 % per
  pass (the f32 GEMM's weight reads shrink 4x; FFN -12 %), weights 2351 ->
  673 MiB, WER -0.0015 (better on 14 clips, worse on 11 of 60 changed).
  It is a weight-only path over the v1 tiling, so it competes with the
  split-K and bf16 tensor-core GEMMs rather than stacking on them; the
  durable form is int8 codes decoded inside the tensor-core tile load.
- Recommendation: keep both off by default until the serving A/B; first
  candidate for a default is `--kv-dtype int8` (or bf16) together with the
  bf16 tensor-core GEMM, when the cap is raised past what f32 rings allow.

## Next action

- A/B on a serving screen (L4 / L40S): `--kv-dtype int8` at the current knee
  and at a higher `--cap`; the full 498-clip bank against a fresh unloaded
  reference before any default change.
- With the bf16 tensor-core GEMM: an int8-weight variant of that kernel
  (decode to bf16 in the tile load), where the weight bytes matter more.
