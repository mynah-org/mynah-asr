# Silero VAD — verified architecture (port target)

Status: **ported and at parity** (`src/vad.c`, `tests/test_vad.sh`). This file is
the reference for the C port, in the same role `docs/nemotron-arch.md` plays for
the ASR model: every number here was read out of the real checkpoint or confirmed
by running it, not taken from a blog post.

Two things in the first version of this file were wrong, and reading the graph
node by node before writing any C is what caught them: the ONNX input is **576
samples, not 512** (see "Forward pass" below), and there is a **ReLU between the
LSTM output and the final 1×1 conv**. Both are load-bearing: sabotaging either one
moves the output by ~0.1 and ~0.9 absolute respectively.

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
- **309 633 parameters** on the 16 kHz path (15 tensors, 1.2 MB in f32 — counted by
  the converter, not estimated). Two branches live in the same file (8 kHz and
  16 kHz) selected by an ONNX `If` on the `sr` input; we only need the 16 kHz one.
  The 8 kHz twin is the same graph with 65 bins, kernel 128, stride 64.
- Weights are stored as `Constant` **nodes inside the `If` subgraphs**, not as
  graph initializers — a converter that only walks `graph.initializer` finds
  nothing. Names are clean (`encoder.0.reparam_conv.weight`, `decoder.rnn.*`).

## Forward pass, 16 kHz

The **API** frame is exactly **512 samples** (32 ms) — silero's own wrapper rejects
anything else. The **graph** input is **576**: the wrapper prepends the 64 samples
of context it kept from the previous call (zeros on the first), so the model always
sees `context(64) | chunk(512)`. Plus LSTM state `[2, 1, 128]` (h and c). Output:
one speech probability and the new state.

That 64-sample lookbehind is the whole reason the VAD is streaming-shaped: our
`mynah_asr_vad_feed` carries it exactly like the wrapper does, and dropping it
costs 0.39 absolute on a real fixture (measured, by removing it on purpose).

| # | Step | Shape / params | Notes |
|---|---|---|---|
| 1 | reflect pad | 64 samples **on the right only** | ONNX `Pad` mode `reflect`, pads `[0,0,0,64]` → 640 samples |
| 2 | STFT as Conv1d | weight `[258, 1, 256]`, kernel 256, **stride 128**, no pad | 258 = 129 real + 129 imag; the basis is a plain buffer → a matmul is enough |
| 3 | magnitude | `Pow(2) + Add + Sqrt` | → 129 bins × 4 frames |
| 4 | `encoder.0` | Conv1d `[128, 129, 3]`, s1, pad 1 | + ReLU |
| 5 | `encoder.1` | Conv1d `[64, 128, 3]`, **s2**, pad 1 | + ReLU |
| 6 | `encoder.2` | Conv1d `[64, 64, 3]`, **s2**, pad 1 | + ReLU |
| 7 | `encoder.3` | Conv1d `[128, 64, 3]`, s1, pad 1 | + ReLU |
| 8 | `decoder.rnn` | LSTM 128→128: `weight_ih/hh [512, 128]`, `bias_ih/hh [512]` | 512 = 4 gates × 128, **stateful across calls**; the 4 → 1 collapse happens in steps 5-6, so the LSTM takes exactly one step per chunk |
| 9 | **ReLU** | on the LSTM output | `decoder.decoder.1` — easy to miss, it sits between the LSTM and the conv |
| 10 | `decoder.decoder.2` | Conv1d `[1, 128, 1]` + bias `[1]` | 1×1, then `Sigmoid`, then `ReduceMean` over axis 1 (a no-op with one timestep) |

Nothing here is new to this codebase: conv1d exists (subsampling), and the LSTM is
the same shape and the same PyTorch convention as the RNNT prediction network —
gate order `[i, f, g, o]` with **both** biases (see docs/architecture-notes.md §6).
So the port is mechanical; the risk is in the details, not the structure.

`src/vad.c` reads all of the above out of `mynah.json` (channels, kernels, strides,
pads, hop, bins, hidden) and **refuses to load** a checkpoint whose geometry does
not collapse to one timestep — the frame arithmetic is checked at open, not assumed.

## Oracle

`onnxruntime` runs this file directly, so the reference is the **actual Silero
implementation**, not a numpy rewrite of it that could repeat my own
misreading — the same reasoning as the llama.cpp K-quant parity
(tests/test_kquant.sh). Confirmed working:

`tools/oracle/vad.py` is that wrapper. The one thing it has to get right is the
framing, which is NOT in the graph, so it does not get to be trusted either:
`--verify-official` replays the same audio through silero's own `OnnxWrapper` and
compares. Result on `samples/en/fleurs_1521.wav`: **232 frames, max abs diff 0.0** —
identical, so the framing is silero's, not mine.

Parity target: per-frame probability, 1e-5 absolute, asserted over a real fixture's
whole frame sequence — not one frame, since the LSTM state makes errors accumulate.
Measured (macOS/Accelerate, f32):

| fixture | frames | worst abs err |
|---|---|---|
| `samples/en/fleurs_1521.wav` (7.4 s) | 232 | 6.9e-07 |
| `samples/long/en_long.wav` (305 s) | 9532 | 6.1e-06 |

The drift over 5 minutes stays bounded, which is the property that matters for a
recurrent module. Speed: 305 s of audio in 0.36 s single-threaded (~840× realtime),
so the VAD is free relative to the ASR it is meant to save work for.

Deliberately broken once each, to prove the test is not green for free: no reflect
pad → 1.1e-01, no head ReLU → 9.2e-01, lookbehind dropped → 3.9e-01.

## Plan

1. **done** — `tools/convert_silero.py`: ONNX → mynah layout, weights pulled out of
   the `If` subgraph constants, 16 kHz branch only, no torch. Reads its
   hyperparameters from the graph (conv attributes, pad chain, tensor shapes)
   rather than from this file, and asserts the geometry collapses to one timestep.
2. **done** — `tools/oracle/vad.py`: onnxruntime wrapper, `--verify-official` cross-check.
3. **done** — `src/vad.c/.h`: the forward pass above, streaming by construction
   (512-sample frames, carried LSTM state + 64-sample lookbehind), config-driven.
4. **done** — `tests/test_vad.c` + `tests/test_vad.sh`: full-sequence parity, exit 77
   without the checkpoint (`make fetch-vad` gets it), `make test-vad` on its own.
   Clean under `make ubsan` and `make leaks` (0 leaks).
5. Integration, one at a time and each with its own gate: offline silence skip →
   segmentation on speech boundaries → streaming endpointing. Each changes output,
   so each needs a before/after quality number (WER/CER on the samples), not just
   "it runs".

Step 5 is where the value is, but steps 1-4 are what make it safe to attempt.

## Using it

```sh
make fetch-vad          # 2.3 MB from the silero-vad repo into models/silero-vad/
make test-vad           # converts if needed, then parity vs onnxruntime
```
The C side is `src/vad.h`: `open` → `feed(512 samples)` → probability, `reset`
between unrelated streams. The decision defaults (threshold, min speech/silence,
speech pad) travel in `mynah.json`, taken from silero's own `get_speech_timestamps`,
so the hysteresis logic added in step 5 has no magic numbers of its own.
