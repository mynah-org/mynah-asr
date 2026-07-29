#!/usr/bin/env python3
"""Offline transcription with the numpy oracle (slow: it is a reference, not a runtime).

Usage:
  uv run python -m oracle.transcribe <model_dir> <file.wav> [--lang it-IT] [--lookahead N]
                                     [--dump-dir DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from safetensors.numpy import load_file

from oracle.features import log_mel
from oracle.model import Oracle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("wav")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--lookahead", type=int, default=None, help="right context (0|1|3|6|13)")
    ap.add_argument("--dump-dir", default=None, help="dump the intermediate activations as .npy")
    ap.add_argument("--decoder", choices=["default", "ctc"], default="default",
                    help="ctc: the auxiliary head of hybrid models (tdt_ctc)")
    ap.add_argument("--target-lang", default=None,
                    help="AED (Canary): output language (different from --lang = translation)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"servono 16 kHz, ricevuti {sr} (resampla con ffmpeg/sox)"

    oracle = Oracle(model_dir)
    prompt_id = oracle.cfg["prompt"]["dictionary"][args.lang] if oracle.has_prompt else None

    mf = load_file(model_dir / "mel_filters.safetensors")
    t0 = time.time()
    feats, valid = log_mel(audio, mf["mel_fb"], mf["window"],
                           normalize=oracle.cfg["features"]["normalize"])
    dumps: dict | None = {} if args.dump_dir else None
    if dumps is not None:
        dumps["mel"] = feats.copy()

    dumps = dumps if dumps is not None else ({} if args.decoder == "ctc" else None)
    enc = oracle.encode(feats[:valid], prompt_id, lookahead=args.lookahead, dumps=dumps)
    if args.decoder == "ctc":
        tokens = oracle.greedy_decode_ctc(dumps["encoder_out"])
    elif oracle.cfg["decoder"]["type"] == "aed_transformer":
        src = "en" if args.lang == "auto" else args.lang.split("-")[0]
        tokens = oracle.greedy_decode_aed(enc, src, args.target_lang or src)
    else:
        tokens = oracle.greedy_decode(enc)
    text, lang = oracle.detokenize(tokens)
    dt = time.time() - t0

    if args.dump_dir:
        out = Path(args.dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, arr in dumps.items():
            np.save(out / f"{name}.npy", arr)
        print(f"[dump] {len(dumps)} stadi in {out}")

    dur = len(audio) / sr
    print(f"[{dur:.1f}s audio | {dt:.1f}s oracolo | lang={lang or args.lang} | {len(tokens)} token]")
    print(text)


if __name__ == "__main__":
    main()
