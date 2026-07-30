# Silero VAD — verified architecture (port target)

Status: **architecture verified, not yet implemented.** This file is the reference
for the C port, in the same role `docs/nemotron-arch.md` plays for the ASR model:
every number here was read out of the real checkpoint or confirmed by running it,
not taken from a blog post.

## Why

Today "silence handling" is one energy heuristic: `find_split_point`
(src/mynah_asr.c) picks the RMS minimum over 20 ms windows to choose *where* to
cut a long file. Nothing detects speech. A real VAD buys three things:

1. **Skip silence offline** — large RTF win on real recordings (meetings, voice
   notes), where a good fraction of the audio is nobody talking.
2. **Better long-file segmentation** — cut on actual speech boundaries instead of
   the quietest 20 ms window, which is only a proxy.
3. **Endpointing / EOU in streaming** — decide that an utterance ended, rather
   than inferring it from decoder output.

## Model

- Source: `https://github.com/snakers4/silero-vad`, file
  `src/silero_vad/data/silero_vad.onnx` (2.3 MB, v5 layout).
- **License: MIT** (Silero Team) — compatible with this repo's MIT.
- ~375 K parameters on the 16 kHz path ≈ 1.5 MB in f32. Two branches live in the
  same file (8 kHz and 16 kHz) selected by an ONNX `If` on the `sr` input; we only
  need the 16 kHz one.
- Weights are stored as `Constant` **nodes inside the `If` subgraphs**, not as
  graph initializers — a converter that only walks `graph.initializer` finds
  nothing. Names are clean (`encoder.0.reparam_conv.weight`, `decoder.rnn.*`).

## Forward pass, 16 kHz

Input: exactly **512 samples** (32 ms) per call, plus LSTM state `[2, 1, 128]`
(h and c). Output: one speech probability, plus the new state. 1024 samples per
call is rejected by the model, so the frame size is fixed, not a suggestion.

| # | Step | Shape / params | Notes |
|---|---|---|---|
| 1 | reflect pad | 64 samples | ONNX `Pad` mode `reflect` (confirmed) |
| 2 | STFT as Conv1d | weight `[258, 1, 256]`, kernel 256, **stride 128**, no pad | 258 = 129 real + 129 imag; the basis is a plain buffer → a matmul is enough |
| 3 | magnitude | `Pow(2) + Add + Sqrt` | → 129 bins |
| 4 | `encoder.0` | Conv1d `[128, 129, 3]`, s1, pad 1 | + ReLU |
| 5 | `encoder.1` | Conv1d `[64, 128, 3]`, **s2**, pad 1 | + ReLU |
| 6 | `encoder.2` | Conv1d `[64, 64, 3]`, **s2**, pad 1 | + ReLU |
| 7 | `encoder.3` | Conv1d `[128, 64, 3]`, s1, pad 1 | + ReLU |
| 8 | `decoder.rnn` | LSTM 128→128: `weight_ih/hh [512, 128]`, `bias_ih/hh [512]` | 512 = 4 gates × 128, **stateful across calls** |
| 9 | `decoder.decoder.2` | Conv1d `[1, 128, 1]` + bias `[1]` | 1×1, then `Sigmoid`, then `ReduceMean` over axis 1 |

Nothing here is new to this codebase: conv1d exists (subsampling), and the LSTM is
the same shape and the same PyTorch convention as the RNNT prediction network —
gate order `[i, f, g, o]` with **both** biases (see docs/architecture-notes.md §6).
So the port is mechanical; the risk is in the details, not the structure.

## Oracle

`onnxruntime` runs this file directly, so the reference is the **actual Silero
implementation**, not a numpy rewrite of it that could repeat my own
misreading — the same reasoning as the llama.cpp K-quant parity
(tests/test_kquant.sh). Confirmed working:

```python
import numpy as np, onnxruntime as ort
s = ort.InferenceSession("silero_vad.onnx", providers=["CPUExecutionProvider"])
state = np.zeros((2, 1, 128), dtype=np.float32)
prob, state = s.run(None, {"input": chunk_512.reshape(1, 512),
                           "state": state, "sr": np.array(16000, dtype=np.int64)})
```

Parity target: per-frame probability, tolerance on the order of 1e-5 relative
(f32 conv + LSTM accumulation), asserted over a real fixture's whole frame
sequence — not one frame, since the LSTM state makes errors accumulate.

## Plan

1. `tools/convert_silero.py` — ONNX → mynah layout (safetensors + config), pulling
   the weights out of the `If` subgraph constants and keeping only the 16 kHz
   branch. Must NOT need torch.
2. `tools/oracle/vad.py` — thin wrapper over onnxruntime that dumps the per-frame
   probabilities of a wav, for the parity test.
3. `src/vad.c/.h` — the forward pass above, streaming by construction (512-sample
   frames, carried state), reusing the existing conv/LSTM kernels.
4. `tests/test_vad.c` — parity against the oracle dump over a full fixture;
   skips at 77 without the converted model, like every other model-gated test.
5. Integration, one at a time and each with its own gate: offline silence skip →
   segmentation on speech boundaries → streaming endpointing. Each changes output,
   so each needs a before/after quality number (WER/CER on the samples), not just
   "it runs".

Step 5 is where the value is, but steps 1-4 are what make it safe to attempt.
