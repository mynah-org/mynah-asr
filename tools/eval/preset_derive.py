#!/usr/bin/env python3
"""preset_derive.py — what a lookahead preset ACTUALLY costs, derived from the
pack and the code rather than from the name of the preset.

    python3 tools/eval/preset_derive.py models_local/<pack> [--evidence DIR:DIR]

WHY THIS EXISTS.  "[56,3] -> [56,0] removes 240 ms of right context" was carried
forward through two plans and it is wrong by a factor of two. The right context
in this implementation is not a mask: `es->right` is read in exactly two places
in src/encoder.c -- its assignment and mynah_asr_enc_stream_need -- and never
reaches the attention. The streaming attention is UNMASKED over
[valid cache + chunk] (grep the block: no mask, no -INFINITY, and the softmax
runs over j in [0, K) with K = valid + Q), so a frame at position t inside a
chunk of q frames attends to the q-1-t frames AFTER it in the same chunk.

Future context is therefore a property of a frame's POSITION IN ITS CHUNK, not
of the preset alone: with q = 4 the four frames get 240, 160, 80 and 0 ms. The
LAST frame of every chunk has no lookahead at all. The expectation over
positions is (q-1)/2 * 80 ms -- 120 ms at [56,3], not 240.

Everything below is computed from the pack's mynah.json (rule 1: no model
constants in the tool) and from the four functions that decide:

    mynah_asr_enc_stream_need()      mel frames a step consumes
    mynah_asr_mel_stream_samples_until()   frame -> samples: f*hop + n_fft/2
    mynah_asr_ss_stream_step()       sub_factor mel frames -> 1 encoder frame
    mynah_asr_enc_stream_step()      cache_valid += q, saturating at left_ctx

Exit: 0 ok, 2 usage, 77 no pack.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys


def derive(cfg, right):
    sub = cfg["sub"]
    hop, nfft, sr = cfg["hop"], cfg["nfft"], cfg["sr"]
    left, efms = cfg["left"], cfg["efms"]
    q = right + 1
    first_mel = 1 + sub * right                      # enc_stream_need, first chunk
    first_samples = (first_mel - 1) * hop + nfft // 2
    steady_mel = sub * (right + 1)                   # enc_stream_need, steady
    period_ms = steady_mel * hop * 1000.0 / sr
    fut = [(q - 1 - t) * efms for t in range(q)]
    steps_to_fill = -(-left // q)                    # ceil
    return {
        "right": right, "q": q,
        "first_mel": first_mel,
        "first_ms": first_samples * 1000.0 / sr,
        "steady_mel": steady_mel,
        "period_ms": period_ms,
        "future_ms": fut,
        "future_mean_ms": sum(fut) / len(fut),
        "steps_to_fill": steps_to_fill,
        "cache_fill_ms": steps_to_fill * period_ms,
        # src/encoder.h states the invariant the unmasked attention rests on:
        # "the left context in the cache (56 frames, always divisible by r+1) is
        # EXACTLY the allowed context". A q that does not divide left_ctx leaves
        # the cache boundary off the chunk grid, and the claim stops holding.
        "divides_left": left % q == 0,
    }


def first_token_audio(path):
    """Audio consumed at the first NATURAL non-blank, from a trace of either
    generation: the new format carries audio_s per decision, the old one only on
    the `emit` line that FOLLOWS its own step (which is why a naive reader of the
    old format is one step early)."""
    new = re.compile(r"\[RNNT\] frame=\d+ audio_s=([\d.-]+) .*chose=(\d+)")
    old = re.compile(r"\[RNNT\] frame=\d+ blank=[-\d.]+ best_nonblank=\d+:[-\d.]+ "
                     r"margin=[-\d.]+ chose=(\w+)")
    emit = re.compile(r"\[RNNT\] emit q=(\d+) audio_s=([\d.]+)")
    blank, pend_q, seen = None, None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = new.search(line)
        if m:
            if blank is None:
                blank = m.group(2)
            if m.group(2) != blank:
                return float(m.group(1))
            continue
        m = old.search(line)
        if m:
            seen.append(m.group(1) == "TOKEN")
            continue
        m = emit.search(line)
        if m and any(seen):
            return float(m.group(2))          # this step's audio, printed after it
        if m:
            seen = []
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("--also", type=int, nargs="*", default=[],
                    help="derive these right values too, even if the pack does not declare them")
    ap.add_argument("--evidence", help="DIR_A:DIR_B of saved traces, to check the derivation")
    a = ap.parse_args()
    p = os.path.join(a.pack, "mynah.json")
    if not os.path.exists(p):
        print(f"preset_derive: no pack at {a.pack} — SKIP", file=sys.stderr)
        return 77
    m = json.load(open(p, encoding="utf-8"))
    f, e, s = m["features"], m["encoder"], m["streaming"]
    presets = s["att_context_presets"]
    cfg = {"sub": e["subsampling_factor"], "hop": f["hop_length"], "nfft": f["n_fft"],
           "sr": f["sample_rate"], "left": presets[0][0], "efms": s["encoder_frame_ms"]}
    print(f"pack {os.path.basename(a.pack.rstrip('/'))}: sub_factor {cfg['sub']}, hop {cfg['hop']}, "
          f"n_fft {cfg['nfft']}, encoder frame {cfg['efms']:.0f} ms, left_ctx {cfg['left']}")
    print(f"DECLARED presets: {presets}   default index {s.get('default_preset_index')}")
    extra = [r for r in a.also if not any(r == pr[1] for pr in presets)]
    if extra:
        print(f"NOT declared by this pack, derived anyway and OFF-DISTRIBUTION: "
              f"{[[cfg['left'], r] for r in extra]}")
    print()
    print(f"  {'preset':10} {'q':>2} {'1st chunk':>11} {'steady':>8} {'cadence P':>10} "
          f"{'future ctx per chunk position':>32} {'mean':>7} {'cache fill':>11}")
    rows = []
    for r in sorted({pr[1] for pr in presets} | set(a.also)):
        d = derive(cfg, r)
        rows.append(d)
        fut = "/".join(f"{x:.0f}" for x in d["future_ms"])
        if len(fut) > 32:
            fut = fut[:29] + "..."
        print(f"  [{cfg['left']},{r}]{'':<{max(0,10-len(str(cfg['left']))-len(str(r))-3)}} "
              f"{d['q']:>2} {d['first_mel']:>4} mel {d['first_ms']:>5.0f}ms "
              f"{d['steady_mel']:>5} mel {d['period_ms']:>8.0f}ms {fut:>32} "
              f"{d['future_mean_ms']:>6.0f}ms {d['cache_fill_ms']:>9.0f}ms"
              f"{'' if d['divides_left'] else '   <-- q does not divide left_ctx'}")
    print()
    print("  READ THE COLUMNS, NOT THE PRESET NAME:")
    print("   * 'future ctx per chunk position' is what each of the q frames in a chunk")
    print("     actually sees ahead of itself. The LAST frame of every chunk has 0 ms,")
    print("     in every preset. The preset's number is only the FIRST frame's share.")
    print("   * 'mean' is the expectation over positions, which is what a first token")
    print("     landing at an arbitrary position pays: (q-1)/2 x encoder_frame_ms.")
    print("   * a preset whose q does not divide left_ctx breaks the invariant")
    print("     src/encoder.h rests its UNMASKED attention on. Every preset this")
    print("     pack declares divides it; [56,2] does not, which is a reason from")
    print("     the code not to reach for it, on top of it being untrained here.")
    print("   * 'cache fill' is left_ctx frames of audio and is THE SAME in every")
    print("     preset, because left_ctx counts encoder frames, not steps. Changing")
    print("     the preset does not shorten the cold-cache window at all.")

    if a.evidence:
        da, db = a.evidence.split(":")
        pairs = []
        for fa in sorted(glob.glob(os.path.join(da, "*.rnnt.err"))):
            fb = os.path.join(db, os.path.basename(fa))
            if not os.path.exists(fb):
                continue
            ta, tb = first_token_audio(fa), first_token_audio(fb)
            if ta is not None and tb is not None:
                pairs.append((os.path.basename(fa)[:-9], ta, tb))
        if pairs:
            d = [b - aa for _, aa, b in pairs]
            print(f"\n  MEASURED, {len(pairs)} clips: audio at the first natural token")
            print(f"  {'clip':20} {'A':>7} {'B':>7} {'delta':>8}")
            for n, aa, b in pairs:
                print(f"  {n:20} {aa:7.2f} {b:7.2f} {b-aa:+8.2f}")
            print(f"  median {st.median(d):+.3f} s   mean {st.mean(d):+.3f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
