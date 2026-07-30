#!/usr/bin/env python3
"""Silero VAD oracle: per-frame speech probabilities, straight from the ONNX.

The reference here is the REAL implementation run by onnxruntime, not a numpy
rewrite of it — the same reasoning as the llama.cpp K-quant parity: if the same
misreading goes into both the C code and the fixtures, a shared bug passes.

The only thing this file has to get right is the framing, and that is not in the
graph: silero's own wrapper feeds `context(64) + chunk(512)` samples per call,
context starting at zeros and carried from the tail of the previous input, with
the audio zero-padded to a whole number of 512-sample chunks. `--verify-official`
checks exactly that against silero's `OnnxWrapper` instead of trusting this note.

Usage:
  uv run --extra vad python -m oracle.vad <silero_vad.onnx> <file.wav> [--out probs.npy]
  uv run --with silero-vad --extra vad python -m oracle.vad <onnx> <wav> --verify-official
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FRAME = 512
CONTEXT = 64
LSTM_HIDDEN = 128


def frame_probs(onnx_path: str, audio: np.ndarray) -> np.ndarray:
    """[n_frames] float32 speech probabilities, n_frames = ceil(len(audio) / 512)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"], sess_options=opts)

    x = np.asarray(audio, dtype=np.float32)
    tail = (-len(x)) % FRAME
    if tail:
        x = np.concatenate([x, np.zeros(tail, dtype=np.float32)])

    state = np.zeros((2, 1, LSTM_HIDDEN), dtype=np.float32)
    ctx = np.zeros(CONTEXT, dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    out = np.empty(len(x) // FRAME, dtype=np.float32)
    for i in range(len(out)):
        chunk = np.concatenate([ctx, x[i * FRAME:(i + 1) * FRAME]])
        prob, state = sess.run(None, {"input": chunk.reshape(1, -1), "state": state, "sr": sr})
        out[i] = prob[0, 0]
        ctx = chunk[-CONTEXT:]
    return out


def load_wav(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SAMPLE_RATE, f"16 kHz required, got {sr}"
    return audio


def official_spans(onnx_path: str, audio: np.ndarray) -> np.ndarray:
    """[n, 2] sample offsets from silero's own get_speech_timestamps.

    Needs silero-vad (and therefore torch), so this is not part of the default
    parity run — see `make test-vad-spans`. The thresholds are the ones the
    converter writes into mynah.json, so C and reference decide on equal terms.
    """
    import torch
    from silero_vad.utils_vad import OnnxWrapper, get_speech_timestamps

    model = OnnxWrapper(onnx_path, force_onnx_cpu=True)
    spans = get_speech_timestamps(
        torch.from_numpy(audio), model,
        threshold=0.5, neg_threshold=0.35, sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=250, min_silence_duration_ms=100, speech_pad_ms=30,
    )
    return np.array([[s["start"], s["end"]] for s in spans], dtype=np.float64).reshape(-1, 2)


def verify_official(onnx_path: str, audio: np.ndarray, probs: np.ndarray) -> None:
    """Cross-check the framing against silero's own wrapper (needs silero-vad)."""
    import torch
    from silero_vad.utils_vad import OnnxWrapper

    ref = OnnxWrapper(onnx_path, force_onnx_cpu=True)
    got = ref.audio_forward(torch.from_numpy(audio), SAMPLE_RATE).numpy().reshape(-1)
    if got.shape != probs.shape:
        raise SystemExit(f"framing mismatch: official {got.shape} vs oracle {probs.shape}")
    err = float(np.abs(got - probs).max())
    print(f"[verify-official] {len(got)} frames, max abs diff {err:.3e}")
    if err > 1e-6:
        raise SystemExit("framing mismatch: the oracle does not reproduce the official wrapper")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("wav")
    ap.add_argument("--out", default=None, help="write the probabilities as .npy")
    ap.add_argument("--spans", default=None,
                    help="write silero's own speech spans (sample offsets) as .npy")
    ap.add_argument("--verify-official", action="store_true",
                    help="cross-check the framing against silero's own wrapper")
    args = ap.parse_args()

    audio = load_wav(args.wav)
    probs = frame_probs(args.onnx, audio)
    if args.verify_official:
        verify_official(args.onnx, audio, probs)
    if args.out:
        np.save(args.out, probs)
    if args.spans:
        spans = official_spans(args.onnx, audio)
        np.save(args.spans, spans)
        print(f"[spans] {len(spans)} speech spans from silero's get_speech_timestamps")
    speech = float((probs > 0.5).mean())
    print(f"[{len(audio) / SAMPLE_RATE:.1f}s audio | {len(probs)} frames | "
          f"{speech * 100:.1f}% above 0.5 | mean {probs.mean():.3f}]")


if __name__ == "__main__":
    main()
