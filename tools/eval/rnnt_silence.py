#!/usr/bin/env python3
"""rnnt_silence.py — does this decoder talk into the silence, and what is the
blank logit doing before the first word?

    python3 tools/eval/rnnt_silence.py .work/evidence/q2-earliness-2026-09-21

Two questions, from SAVED traces only: no model, no server, no box. It reads the
`MYNAH_ASR_TRACE_RNNT` captures beside their transcripts and the committed WAVs.

**1. False non-blanks in the leading silence.** R-3F owed this and Q-2 said the
trace could already answer it. A decision taken on audio that is entirely before
speech onset, that chooses a token, is the engine inventing words out of room
tone. Reported as a count with its denominator, per clip and pooled, and split
between DECODED (the RNNT chose a token) and PUBLISHED (a delta reached the
client) — R-3B showed the two differ, because the detokeniser lifts the language
tag out of the text and `total > chars_emitted` is what gates a delta.

**2. The regime the blank logit is in before the first emission.** This started
as a suspicion about a trace line and it is the more interesting half. The
classification is a 2x2x2, because three things change together at the start of
an utterance and it is easy to credit the wrong one:

    audio           silence | SPEECH      per-frame RMS against the clip's own
                                          noise floor, the same +12 dB rule as
                                          tools/bench/clip_onset.py
    predictor       SOS     | moved       pred_step() runs ONLY on an emitted
                                          token (src/decoder.c:326), so `s->g`
                                          is FROZEN at the SOS state for the
                                          whole pre-first-token window
    encoder cache   filling | FULL        cache_valid climbs to left_ctx over
                                          the first left_ctx encoder frames

Only the full table can say which factor the blank magnitude tracks. A single
split cannot: before the first token all three are unusual at once.

Onsets come from tools/bench/clip_onset.py — an ENERGY onset, with its limits
recorded there. Frame times are reconstructed from the `emit` lines' `audio_s`
and an 80 ms encoder frame, so they are chunk-quantised: good enough to call a
frame silent, not an alignment.

Exit: 0 ok · 2 usage · 77 no traces found.
"""
from __future__ import annotations

import argparse
import array
import glob
import json
import math
import os
import re
import statistics as st
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import clip_onset  # noqa: E402

FRAME = re.compile(r"\[RNNT\] frame=(\d+) blank=([-\d.]+) best_nonblank=(\d+):([-\d.]+) "
                   r"margin=([-\d.]+) chose=(\w+)")
EMIT = re.compile(r"\[RNNT\] emit q=(\d+) audio_s=([\d.]+) tokens_added=(\d+) "
                  r"chars=(\d+) chars_emitted=(\d+)")
CLIP = re.compile(r"\[stream: (\S+) in")
ENC_FRAME_S = 0.080


def parse(path):
    clip, steps, pend = None, [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = CLIP.search(line)
        if m:
            clip = m.group(1)
            continue
        m = FRAME.search(line)
        if m:
            pend.append({"frame": int(m.group(1)), "blank": float(m.group(2)),
                         "margin": float(m.group(5)), "tok": m.group(6) == "TOKEN"})
            continue
        m = EMIT.search(line)
        if m:
            steps.append({"audio_s": float(m.group(2)), "added": int(m.group(3)),
                          "chars": int(m.group(4)), "emitted": int(m.group(5)),
                          "frames": pend})
            pend = []
    return clip, steps


def envelope(path, frame_ms=10.0):
    """Per-frame RMS in dB and the clip's own noise floor, clip_onset's method."""
    w = wave.open(path, "rb")
    n, sr = w.getnframes(), w.getframerate()
    raw = w.readframes(n)
    w.close()
    a = array.array("h")
    a.frombytes(raw)
    step = max(int(sr * frame_ms / 1000.0), 1)
    e = []
    for i in range(0, len(a) - step, step):
        acc = 0.0
        for v in a[i:i + step]:
            acc += float(v) * float(v)
        e.append(10.0 * math.log10(acc / step + 1e-9))
    if not e:
        return None
    return e, sorted(e)[int(0.10 * len(e))], frame_ms / 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="directories of *.rnnt.err captures")
    ap.add_argument("--left-ctx", type=int, default=56,
                    help="encoder frames after which cache_valid saturates")
    ap.add_argument("--margin-db", type=float, default=12.0)
    ap.add_argument("--extreme", type=float, default=100.0,
                    help="|blank| above this is called the extreme regime")
    ap.add_argument("--json", help="write the per-frame classification here")
    a = ap.parse_args()

    out = {}
    for d in a.dirs:
        files = sorted(glob.glob(os.path.join(d, "*.rnnt.err")))
        if not files:
            continue
        cells, clips, pooled = {}, [], {"dec": 0, "pub": 0, "n": 0, "frames": 0, "steps": 0}
        for f in files:
            clip, steps = parse(f)
            if not clip or not os.path.exists(clip):
                continue
            on = clip_onset.onset_s(clip, margin_db=a.margin_db)
            env = envelope(clip)
            txt = f.replace(".rnnt.err", ".stream.txt")
            text = open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else ""
            ft = next((s["audio_s"] for s in steps if any(x["tok"] for x in s["frames"])), None)

            pre = [s for s in steps if s["audio_s"] <= on + 1e-9]
            dec = sum(1 for s in pre for x in s["frames"] if x["tok"])
            pub = sum(1 for s in pre if s["chars"] > s["emitted"])
            clips.append({"clip": os.path.basename(f)[:-9], "onset": on, "pre_steps": len(pre),
                          "decoded": dec, "published": pub, "first_token_s": ft,
                          "empty": not text})
            pooled["n"] += 1
            pooled["dec"] += dec > 0
            pooled["pub"] += pub > 0
            pooled["steps"] += len(pre)

            for s in steps:
                q = len(s["frames"])
                for j, x in enumerate(s["frames"]):
                    pooled["frames"] += 1
                    t1 = s["audio_s"] - (q - 1 - j) * ENC_FRAME_S
                    lab_a = "SPEECH"
                    if env:
                        e, floor, fs = env
                        i0, i1 = int(max(t1 - ENC_FRAME_S, 0.0) / fs), int(t1 / fs) + 1
                        seg = e[max(i0, 0):min(i1, len(e))]
                        if seg and max(seg) < floor + a.margin_db:
                            lab_a = "silence"
                    lab_p = "SOS" if (ft is None or s["audio_s"] < ft) else "moved"
                    lab_c = "FULL" if x["frame"] >= a.left_ctx else "filling"
                    cells.setdefault((lab_a, lab_p, lab_c), []).append(
                        (abs(x["blank"]), x["margin"], x["tok"]))

        if not clips:
            continue
        out[d] = {"clips": clips, "pooled": pooled,
                  "cells": {"/".join(k): len(v) for k, v in cells.items()}}

        print(f"\n===== {d}   ({pooled['n']} clips, {pooled['frames']} decisions)")
        print(f"  {'clip':22} {'onset':>6} {'pre-steps':>9} {'decoded':>8} "
              f"{'published':>9} {'1st tok':>8} {'empty':>6}")
        for c in clips:
            ft = f"{c['first_token_s']:.2f}" if c['first_token_s'] else "-"
            print(f"  {c['clip']:22} {c['onset']:6.2f} {c['pre_steps']:9d} "
                  f"{c['decoded']:8d} {c['published']:9d} {ft:>8} "
                  f"{('YES' if c['empty'] else 'no'):>6}")
        print(f"  -> clips that DECODED a token on pre-onset audio only : "
              f"{pooled['dec']}/{pooled['n']}   ({pooled['steps']} such steps in total)")
        print(f"  -> clips that PUBLISHED text on pre-onset audio only  : "
              f"{pooled['pub']}/{pooled['n']}")
        print(f"  -> empty transcripts                                  : "
              f"{sum(1 for c in clips if c['empty'])}/{pooled['n']}")

        print(f"\n  blank magnitude by audio x predictor x encoder cache")
        print(f"  {'audio':8} {'pred':6} {'cache':8} {'frames':>7} {'|blank| med':>12} "
              f"{'p90':>9} {'extreme':>8} {'TOKEN':>6}")
        for k in sorted(cells, key=lambda k: -len(cells[k])):
            v = cells[k]
            b = sorted(x[0] for x in v)
            ext = 100.0 * sum(1 for x in v if x[0] > a.extreme) / len(v)
            print(f"  {k[0]:8} {k[1]:6} {k[2]:8} {len(v):>7} {st.median(b):12.1f} "
                  f"{b[int(0.9 * len(b))]:9.1f} {ext:7.1f}% {sum(1 for x in v if x[2]):6d}")

    if not out:
        print("rnnt_silence: no usable traces (need *.rnnt.err beside their clips) — SKIP",
              file=sys.stderr)
        return 77
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
