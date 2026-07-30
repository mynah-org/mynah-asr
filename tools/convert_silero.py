#!/usr/bin/env python3
"""Convert Silero VAD (silero_vad.onnx, MIT) into the Mynah layout.

Input:  the ONNX file shipped in silero-vad/src/silero_vad/data/silero_vad.onnx
Output (into <out_dir>):
  - mynah.json          config that drives src/vad.c
  - model.safetensors   the 16 kHz branch weights, f32

Only the 16 kHz branch is kept: the file holds both sample rates behind an ONNX
`If` on the `sr` input, and the runtime is 16 kHz-only anyway.

Two traps, both paid for already (docs/vad-silero.md):
  - the weights are `Constant` NODES inside the `If` subgraphs, not
    graph.initializer — walking initializers finds exactly nothing;
  - the ONNX input is 576 samples, not 512: the official wrapper prepends 64
    samples of context to each 512-sample chunk. The 512 is the API contract,
    576 is what the graph sees.

Hyperparameters are read out of the graph (conv kernel/stride/pads, the reflect
pad amount, tensor shapes) instead of being typed in from the doc, and the frame
arithmetic is asserted to collapse to exactly one LSTM step per chunk.

Usage: uv run --extra vad python convert_silero.py <silero_vad.onnx> <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

SAMPLE_RATE = 16000
# Chunk size and carried context: neither is in the graph, both are enforced by
# silero's own wrapper (`Provided number of samples is ...` / `torch.cat([context,
# x])`), so they belong here as the model's contract. The assertions below check
# the graph agrees with them.
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64

WEIGHTS = [
    "stft.forward_basis_buffer",
    "encoder.0.reparam_conv.weight", "encoder.0.reparam_conv.bias",
    "encoder.1.reparam_conv.weight", "encoder.1.reparam_conv.bias",
    "encoder.2.reparam_conv.weight", "encoder.2.reparam_conv.bias",
    "encoder.3.reparam_conv.weight", "encoder.3.reparam_conv.bias",
    "decoder.rnn.weight_ih", "decoder.rnn.weight_hh",
    "decoder.rnn.bias_ih", "decoder.rnn.bias_hh",
    "decoder.decoder.2.weight", "decoder.decoder.2.bias",
]


def constants(graph) -> dict[str, np.ndarray]:
    """name -> array for every Constant node of a subgraph (prefix stripped).

    ONNX inlining prefixes the original name with `If_0_then_branch__Inline_0__`;
    the tail is the clean PyTorch name.
    """
    out = {}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            name = node.output[0].split("__")[-1]
            out[name] = numpy_helper.to_array(attr.t)
    return out


def attrs(node) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def branch_16k(graph):
    """The `If` subgraph selected when sr == 16000."""
    eq = {n.output[0]: n for n in graph.node if n.op_type == "Equal"}
    const = {n.output[0]: numpy_helper.to_array(n.attribute[0].t)
             for n in graph.node if n.op_type == "Constant"}
    for node in graph.node:
        if node.op_type != "If":
            continue
        cond = eq.get(node.input[0])
        if cond is None:
            continue
        rates = [int(const[i]) for i in cond.input if i in const]
        if SAMPLE_RATE not in rates:
            continue
        for a in node.attribute:
            if a.name == "then_branch":
                return a.g
    raise SystemExit("convert_silero: no `If sr == 16000` branch found — wrong file?")


def convs(graph) -> list[tuple[str, dict]]:
    """Conv nodes in graph order, with the weight name each one uses."""
    return [(n.input[1].split("__")[-1], attrs(n)) for n in graph.node if n.op_type == "Conv"]


def reflect_pad(graph) -> int:
    """The right-side reflect pad of the STFT, from the `[0, N]` constant feeding it.

    The Pad amount is built by a chain of Concat/Reshape/Slice/Transpose ops (torch
    export noise) starting from a single 2-element constant [0, N]; N is the pad.
    """
    pads = [n for n in graph.node if n.op_type == "Pad"]
    if len(pads) != 1:
        raise SystemExit(f"convert_silero: expected exactly one Pad, found {len(pads)}")
    if attrs(pads[0]).get("mode") != b"reflect":
        raise SystemExit("convert_silero: STFT pad is not `reflect` any more")
    # walk back from the Pad's `pads` input to the constants it is built from; the
    # amount is the only [begin, end] pair among them (the others are reshape
    # targets and reverse-slice bounds, which carry negative values)
    producer = {o: n for n in graph.node for o in n.output}
    seen, stack, cands = set(), [pads[0].input[1]], []
    while stack:
        name = stack.pop()
        node = producer.get(name)
        if node is None or name in seen:
            continue
        seen.add(name)
        if node.op_type == "Constant":
            v = numpy_helper.to_array(node.attribute[0].t)
            if v.dtype == np.int64 and v.shape == (2,) and v.min() >= 0:
                cands.append(v)
        stack.extend(node.input)
    if len(cands) != 1:
        raise SystemExit(f"convert_silero: {len(cands)} pad candidates, expected 1")
    lo, hi = int(cands[0][0]), int(cands[0][1])
    if lo != 0:
        raise SystemExit(f"convert_silero: unexpected left pad {lo} (the graph pads right only)")
    return hi


def conv_out(t_in: int, kernel: int, stride: int, pads: list[int]) -> int:
    return (t_in + pads[0] + pads[1] - kernel) // stride + 1


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-1])
    onnx_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    g = branch_16k(onnx.load(str(onnx_path)).graph)
    tensors = constants(g)
    missing = [w for w in WEIGHTS if w not in tensors]
    if missing:
        raise SystemExit(f"convert_silero: missing weights {missing}")
    weights = {w: np.ascontiguousarray(tensors[w], dtype=np.float32) for w in WEIGHTS}

    # ---- geometry, read from the graph -------------------------------------
    conv_list = convs(g)
    stft, encoder, head = conv_list[0], conv_list[1:5], conv_list[5]
    if stft[0] != "stft.forward_basis_buffer" or len(conv_list) != 6:
        raise SystemExit(f"convert_silero: unexpected conv layout {[c[0] for c in conv_list]}")

    n_fft = int(stft[1]["kernel_shape"][0])
    hop = int(stft[1]["strides"][0])
    pad = reflect_pad(g)
    n_bins = weights["stft.forward_basis_buffer"].shape[0] // 2    # 129 real + 129 imag
    enc_strides = [int(c[1]["strides"][0]) for c in encoder]
    enc_pads = [[int(p) for p in c[1]["pads"]] for c in encoder]
    enc_kernels = [int(c[1]["kernel_shape"][0]) for c in encoder]
    lstm_hidden = weights["decoder.rnn.weight_hh"].shape[1]

    # ---- the frame arithmetic must collapse to ONE LSTM step per chunk -----
    n_in = CONTEXT_SAMPLES + FRAME_SAMPLES + pad
    t = conv_out(n_in, n_fft, hop, [0, 0])
    stft_frames = t
    for k, s, p in zip(enc_kernels, enc_strides, enc_pads):
        t = conv_out(t, k, s, p)
    if t != 1:
        raise SystemExit(f"convert_silero: the encoder yields {t} timesteps, expected 1")
    if weights["encoder.0.reparam_conv.weight"].shape[1] != n_bins:
        raise SystemExit("convert_silero: encoder.0 does not consume the magnitude bins")
    if weights["decoder.rnn.weight_ih"].shape != (4 * lstm_hidden, lstm_hidden):
        raise SystemExit("convert_silero: unexpected LSTM shapes")

    cfg = {
        "mynah_asr_format": 1,
        "name": out_dir.name,
        "arch": "silero_vad_v5",
        "engine": "vad",
        "weights": "model.safetensors",
        "license": "MIT (Silero Team)",
        "sample_rate": SAMPLE_RATE,
        "vad": {
            # framing: the caller feeds frame_samples, the model sees
            # context_samples + frame_samples + reflect_pad
            "frame_samples": FRAME_SAMPLES,
            "context_samples": CONTEXT_SAMPLES,
            "reflect_pad": pad,
            "n_fft": n_fft,
            "hop_length": hop,
            "n_bins": n_bins,
            "stft_frames": stft_frames,
            "encoder_channels": [int(weights[f"encoder.{i}.reparam_conv.weight"].shape[0])
                                 for i in range(4)],
            "encoder_kernels": enc_kernels,
            "encoder_strides": enc_strides,
            "encoder_pads": enc_pads,
            "lstm_hidden": int(lstm_hidden),
            # decision defaults, from silero's own get_speech_timestamps
            "threshold": 0.5,
            "neg_threshold": 0.35,
            "min_speech_ms": 250,
            "min_silence_ms": 100,
            "speech_pad_ms": 30,
        },
    }

    save_file(weights, str(out_dir / "model.safetensors"))
    (out_dir / "mynah.json").write_text(json.dumps(cfg, indent=1) + "\n")
    n_params = sum(int(a.size) for a in weights.values())
    print(f"OK {out_dir.name} [vad]: mynah.json, model.safetensors "
          f"({len(weights)} tensors, {n_params} params, "
          f"{n_in} samples in -> {stft_frames} stft frames -> 1 step)")


if __name__ == "__main__":
    main()
