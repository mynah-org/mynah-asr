#!/usr/bin/env python3
"""chunk_phase.py — is the within-chunk lookahead doing what the preset claims?

    python3 tools/eval/chunk_phase.py .work/evidence/r8-A-2026-09-22

R-11, a falsification test for the lookahead explanation of the first-word wait.
Saved traces only: no model, no server, no machine.

Under `[56,3]` the four frames of a chunk are decided at the same instant, having
consumed the same total audio, and differ only in where their own centre sits:
position 0 carries 240 ms of future context, position 3 carries none. The
prediction and the confound are pre-registered in
`.work/chunk-phase-prereg.md`, written before this was run.

P1: `speech_at_frame` at the crossing should rise with position, spread ~240 ms.
P2: within a chunk, `margin_lex` should be roughly flat across positions if the
    lookahead is strong, and fall steeply with position if each frame is
    dominated by its own past.

THE CONFOUND, restated here so no reader misses it: `F mod q` is chosen by the
decoder, not assigned. The greedy loop scans a chunk in order and breaks at the
first non-blank, so the lowest position wins whenever more than one would fire.
Over-representation of low positions is expected under either hypothesis and is
not evidence. Nothing here is a causal claim.

Exit: 0 ok, 77 no traces.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import clip_onset  # noqa: E402

FRAME = re.compile(r"\[RNNT\] frame=(\d+) audio_s=([\d.-]+) enc=([\d.]+) joint=([\d.]+) "
                   r"pred=(\w+) post=(\d) blank=(\d+):([-\d.]+):r(\d+) "
                   r"wmark=(-?\d+):([-\d.]+):r(-?\d+) lex=(\d+):([-\d.]+):r(\d+) "
                   r"nb=(\d+):([-\d.]+):r(\d+) chose=(\d+) "
                   r"margin_lex=([-\d.]+) margin_nb=([-\d.]+)")
CLIP = re.compile(r"\[stream: (\S+) in")


def parse(path):
    clip, rows = None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = CLIP.search(line)
        if m:
            clip = m.group(1)
            continue
        m = FRAME.search(line)
        if m:
            g = m.groups()
            rows.append({"f": int(g[0]), "audio_s": float(g[1]), "enc": float(g[2]),
                         # group order: frame,audio,enc,joint,pred,post,
                         # blank(id,score,rank), wmark(3), lex(3), nb(3),
                         # chose, margin_lex, margin_nb
                         "blank_id": g[6], "chose": g[18],
                         "m_lex": float(g[19]), "m_nb": float(g[20])})
    return clip, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--frame-ms", type=float, default=80.0)
    ap.add_argument("--chunks-before", type=int, default=3,
                    help="how many pre-crossing chunks P2 pools")
    a = ap.parse_args()
    q, fms = a.q, a.frame_ms / 1000.0

    files = sorted(glob.glob(os.path.join(a.dir, "*.rnnt.err")))
    if not files:
        print("chunk_phase: no traces — SKIP", file=sys.stderr)
        return 77

    rows, pairs = [], []
    for path in files:
        clip, tr = parse(path)
        if not clip or not tr or not os.path.exists(clip):
            continue
        onset = clip_onset.onset_s(clip)
        blank = tr[0]["blank_id"]
        cross = next((r for r in tr if r["chose"] != blank), None)
        if cross is None:
            continue
        F = cross["f"]
        rows.append({
            "clip": os.path.basename(path)[:-9], "onset": onset, "F": F, "t": F % q,
            "audio_at": cross["audio_s"],
            "frame_time": (F + 1) * fms,
            "speech_at_frame": (F + 1) * fms - onset,
            "speech_at_audio": cross["audio_s"] - onset,
            "m_lex": cross["m_lex"],
        })
        # P2: complete pre-crossing chunks, paired by chunk
        by_chunk = {}
        for r in tr:
            if r["f"] >= F - (F % q):          # the crossing chunk itself is excluded
                continue
            by_chunk.setdefault(r["f"] // q, {})[r["f"] % q] = r
        full = [c for c in sorted(by_chunk) if len(by_chunk[c]) == q]
        for c in full[-a.chunks_before:]:
            pairs.append({"clip": rows[-1]["clip"], "chunk": c,
                          "m": [by_chunk[c][t]["m_lex"] for t in range(q)],
                          "enc": [by_chunk[c][t]["enc"] for t in range(q)]})

    print(f"=== R-11, {len(rows)} clips, q={q}, {a.frame_ms:.0f} ms per encoder frame")
    print(f"future context by position: "
          + ", ".join(f"t={t}: {(q-1-t)*a.frame_ms:.0f} ms" for t in range(q)))

    print(f"\n--- RAW, all {len(rows)} rows before any aggregate")
    print(f"  {'clip':20} {'onset':>6} {'F':>4} {'F%q':>4} {'audio@':>7} {'frame_t':>8} "
          f"{'speech@frame':>13} {'speech@audio':>13} {'margin':>8}")
    for r in sorted(rows, key=lambda r: (r["t"], r["clip"])):
        print(f"  {r['clip']:20} {r['onset']:6.2f} {r['F']:>4} {r['t']:>4} "
              f"{r['audio_at']:7.2f} {r['frame_time']:8.2f} {r['speech_at_frame']:13.3f} "
              f"{r['speech_at_audio']:13.3f} {r['m_lex']:8.2f}")

    print(f"\n--- P1: speech consumed at the crossing FRAME, by chunk position")
    print(f"  {'t':>2} {'future':>7} {'n':>3} {'median':>8} {'mean':>8} {'min':>7} {'max':>7}")
    meds = {}
    for t in range(q):
        v = sorted(r["speech_at_frame"] for r in rows if r["t"] == t)
        if not v:
            print(f"  {t:>2} {(q-1-t)*a.frame_ms:6.0f}m {0:>3}   (empty)")
            continue
        meds[t] = st.median(v)
        print(f"  {t:>2} {(q-1-t)*a.frame_ms:6.0f}m {len(v):>3} {st.median(v):8.3f} "
              f"{st.mean(v):8.3f} {v[0]:7.3f} {v[-1]:7.3f}")
    if len(meds) >= 2:
        lo, hi = min(meds), max(meds)
        spread = meds[hi] - meds[lo]
        exp = (hi - lo) * a.frame_ms / 1000.0
        print(f"\n  predicted spread t={lo} -> t={hi} if the lookahead is at full scale: "
              f"{exp:+.3f} s")
        print(f"  observed spread of medians:                                  {spread:+.3f} s"
              f"   ({100*spread/exp:.0f}% of predicted)" if exp else "")
        order = [meds[t] for t in sorted(meds)]
        print(f"  monotone increasing with position? "
              f"{'yes' if all(x <= y for x, y in zip(order, order[1:])) else 'NO'}")

    print(f"\n--- P2: margin_lex by position INSIDE a chunk, paired "
          f"({len(pairs)} complete pre-crossing chunks, last {a.chunks_before} per clip)")
    if pairs:
        print(f"  {'t':>2} {'future':>7} {'median margin':>14} {'median |enc|':>13}")
        for t in range(q):
            print(f"  {t:>2} {(q-1-t)*a.frame_ms:6.0f}m "
                  f"{st.median([p['m'][t] for p in pairs]):14.2f} "
                  f"{st.median([p['enc'][t] for p in pairs]):13.2f}")
        d = [p["m"][q - 1] - p["m"][0] for p in pairs]
        print(f"\n  within-chunk drop, margin(t={q-1}) - margin(t=0): "
              f"median {st.median(d):+.2f}, mean {st.mean(d):+.2f}, "
              f"negative in {sum(1 for x in d if x < 0)}/{len(d)} chunks")
        mono = sum(1 for p in pairs if all(x >= y for x, y in zip(p["m"], p["m"][1:])))
        print(f"  monotonically falling with position in {mono}/{len(pairs)} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
