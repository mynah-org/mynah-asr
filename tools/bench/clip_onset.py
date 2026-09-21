#!/usr/bin/env python3
"""clip_onset.py — where does the speech in each clip actually begin.

    python3 tools/bench/clip_onset.py samples/*/fleurs_*.wav --json onsets.json

WHY. "Time to first word" measured from the moment the socket opened charges
the server for whatever silence the clip starts with. A bank whose clips lead
with two seconds of room tone will report a two-second latency the server never
caused, and the fix is not a better threshold, it is knowing where the speech
starts. `streaming_metrics.analyze_utterance` takes this map and reports TTFP
from the onset beside TTFP from the stream open; the two together say how much
of the wait was the audio and how much was the engine.

METHOD, and its limits. Short-time RMS over 10 ms frames; the noise floor is
the 10th percentile of frame energy; the onset is the first frame that exceeds
the floor by `--margin` dB and is followed by `--hold` ms that also exceed it,
which is what stops a click or a breath from being called speech. It is an
ENERGY onset, not a speech detector: it will mark loud non-speech, and on a
clip that fades in it marks the crossing rather than the intent. Where that
matters, run the repo's VAD instead and feed its first span. Recorded here so
that the number carries its method.

Standard library only: this runs on a bare box beside the load generator.
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import sys
import wave


def onset_s(path, frame_ms=10.0, margin_db=12.0, hold_ms=60.0):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            return None                       # 16-bit PCM only, by design
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    pcm = array.array("h")
    pcm.frombytes(raw)
    if ch > 1:
        pcm = array.array("h", pcm[::ch])     # first channel, no mixing
    step = max(1, int(sr * frame_ms / 1000.0))
    rms = []
    for i in range(0, len(pcm) - step + 1, step):
        acc = 0
        for v in pcm[i:i + step]:
            acc += v * v
        rms.append(math.sqrt(acc / step))
    if not rms:
        return None
    ordered = sorted(rms)
    floor = ordered[max(0, int(0.10 * len(ordered)) - 1)]
    floor = max(floor, 1.0)                   # a digitally silent lead-in
    thr = floor * (10.0 ** (margin_db / 20.0))
    hold = max(1, int(hold_ms / frame_ms))
    for i in range(len(rms) - hold + 1):
        if all(rms[i + k] > thr for k in range(hold)):
            return i * frame_ms / 1000.0
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--frame-ms", type=float, default=10.0)
    ap.add_argument("--margin-db", type=float, default=12.0)
    ap.add_argument("--hold-ms", type=float, default=60.0)
    ap.add_argument("--root", help="strip this prefix from the keys written")
    ap.add_argument("--json", help="write the map here")
    a = ap.parse_args()

    out, missed = {}, []
    for c in a.clips:
        try:
            v = onset_s(c, a.frame_ms, a.margin_db, a.hold_ms)
        except (OSError, wave.Error, EOFError) as e:
            print(f"  {c}: {e}", file=sys.stderr); v = None
        key = os.path.relpath(c, a.root) if a.root else c
        if v is None:
            missed.append(key)
        else:
            out[key] = round(v, 3)
    for k in sorted(out):
        print(f"  {out[k]:6.3f} s  {k}")
    if missed:
        print(f"  no onset found in {len(missed)} clip(s): {', '.join(missed[:5])}",
              file=sys.stderr)
    if out:
        vs = sorted(out.values())
        print(f"\n  {len(out)} clip(s): median {vs[len(vs)//2]:.3f} s, "
              f"max {vs[-1]:.3f} s  (method: {a.margin_db:.0f} dB over the 10th-percentile "
              f"frame energy, held {a.hold_ms:.0f} ms)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print(f"  -> {a.json}")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
