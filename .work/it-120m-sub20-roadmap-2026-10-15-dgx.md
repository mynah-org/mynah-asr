# IT 120M — method lab on the DGX (GB10), 2026-10-15

Continuation of `it-120m-sub20-roadmap-2026-10-12.md`. Schedule scaling C250 20k -> 28k -> 36k is in diminishing
returns (clean greedy 26.48 -> 24.70 -> 23.73, 5 L40S-hours for the last -1.0). This session asks a different
question: which training-free or cheap change buys what more steps buy, and which recipe change gives a steeper
curve at 1-4k steps. Training-free first, micro-FT second; the tokenizer stays as is (762/1026 new rows, confounded).

## DGX qualification (OBSERVED)

- Box: DGX Spark GB10 (sm_121, aarch64, 121 GB unified memory, CUDA 13.0 driver), user-space venv
  `~/nemo-venv`: NeMo 3.0.0 + torch 2.11.0+cu128 (the L40S stack). numba-cuda 0.30.4 needs the CUDA 12.9 toolchain
  for sm_121 (nvcc/nvrtc/runtime/nvjitlink-cu12 12.9.x; with 12.8 NVVM rejects compute_121a, mixing cu13 NVVM with
  torch's cu12 nvJitLink fails the link). RNNT loss gate: `warprnnt_numba` forward/backward OK.
- Throughput, C250 replica (real durations/texts/weights of the 36k mix, synthetic audio, same recipe and LR
  schedule, 600 steps): **121x RT, 41.7 steps/min, peak 35.6 GB, GPU util ~95 %** vs L40S C250-36k 360x RT,
  ~120 steps/min: the GB10 is **3.0x slower**; compute-bound, not loader-bound. A 36k run = ~14-15 h here.
- Sanity FT: stock -> 800 steps on FLEURS-it train (2,175 clips), 200 held-out clips: WER 56.7 -> 46.3 -> 43.3
  -> 43.0 (steps 200/400/600/800), CER 17.6 -> 12.7, no empties/NaN; 187x RT on these shorter clips.
- Verdict: method lab (decoding, analysis, soups, 1-4k-step screens), not the long-run machine.

## Pre-registration (2026-10-15, before any number on the eval banks)

Checkpoint `plain-it-c250-36k` (clean 23.73 / 22.00 / 15.11, harness 22.27 / 20.53 / 16.25). Dev = the harness
(mls, fleurs, cv, vp; 200 per set); report = clean-it (tedx 909, italic 475), read ONCE at the dev-chosen setting.

1. **Greedy blank penalty** (`diag/greedy_blank_sweep.py`; serving decoder; parity gate PASSED: at P = 0 it
   reproduces NeMo greedy_batch exactly, 10.72 / 2.98 / len 0.981 on 200 FLEURS clips). Grid
   P in {-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2}, chosen by harness macro greedy WER.
   adopt: clean greedy -0.5 or better AND no clean set worse by > 0.3; marginal: -0.2..-0.5; null: < 0.2.
2. **Model soup** (`diag/soup.py`, uniform weights; all from the same stock init): 28k+36k and 20k+28k+36k.
   Selected on harness greedy; clean read once for the selected soup. keep: harness AND clean greedy -0.3 or
   better vs 36k; null otherwise. (No intermediate checkpoints of the 36k run were archived: only finals.)
3. **Error decomposition** of the 36k (`diag/nbest_gap.py` over the greedy dump + the beam-8 N-best dump):
   descriptive, no gate. Splits the greedy -> beam -> oracle@8 mass into search / ranking / no-good-hypothesis,
   and errors into near (ending, spelling, elision) vs far substitutions, deletions, insertions, segmentation,
   by set and clip duration. It decides which micro-FT arms are worth running; the E3-8k autopsy
   (88 % model errors, LM rescoring not transferable to TEDx) is the prior.
