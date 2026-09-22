#!/usr/bin/env python3
"""norm_predicts.py — does the high-norm encoder regime predict a product failure?

    python3 tools/eval/norm_predicts.py -m models_local/<pack> \
        --gate .work/evidence/gates/nemotron-56-3-int8.json

PHASE B. R-13 localised a real thing: before the first token, the prompt
projector's ReLU turns a 1.13x input spread into a 7.21x output spread, and the
encoder output the joint reads reaches a norm around 20 instead of 3. That is a
FACT about the representation. It is NOT a reason to touch it.

The only question that licenses an intervention is whether the regime PREDICTS
something a user would notice:

    a wrong first published word
    a later first correct word
    a worse final WER / CER

Measured per utterance on the EN/FR evaluation bank, each language on its own,
against the frozen gate run so the outcomes are exactly the ones the gate
scored. If it predicts nothing, the branch closes and nothing is normalised,
clamped or compensated for.

Exit: 0 ok, 2 usage, 77 missing inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import streaming_metrics as M          # noqa: E402

FR = re.compile(r"\[RNNT\] frame=(\d+) audio_s=[\d.-]+ enc=([\d.]+) joint=[\d.]+ "
                r"pred=(\w+) post=\d blank=(\d+):")
CH = re.compile(r"chose=(\d+) margin_lex=([-\d.]+)")


def trace_features(binary, model, quant, clip, lang, hi):
    """Pre-crossing high-norm fraction, and the margin, for one utterance."""
    env = dict(os.environ, MYNAH_ASR_TRACE_RNNT="1")
    p = subprocess.run([binary, "stream", "-m", model, "-i", clip, "--quant", quant,
                        "--lang", lang, "--deltas"], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        return None
    blank, pre, margins = None, [], []
    for line in p.stderr.splitlines():
        m, c = FR.search(line), CH.search(line)
        if not m or not c:
            continue
        if blank is None:
            blank = m.group(4)
        if c.group(1) != blank:
            break                                    # the crossing: stop
        pre.append(float(m.group(2)))
        margins.append(float(c.group(2)))
    if not pre:
        return None
    return {"n_pre": len(pre), "hi_frac": sum(1 for e in pre if e > hi) / len(pre),
            "enc_median": st.median(pre), "margin_median": st.median(margins)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--gate", required=True, help="the frozen lang_gate JSON")
    ap.add_argument("--root", default="samples/eval-bank")
    ap.add_argument("--binary", default="./mynah-asr")
    ap.add_argument("--hi", type=float, default=10.0, help="|enc| above this is the regime")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--json")
    a = ap.parse_args()
    if not os.path.exists(a.gate):
        print(f"norm_predicts: no gate run at {a.gate} — SKIP", file=sys.stderr)
        return 77
    gate = json.load(open(a.gate, encoding="utf-8"))

    out = {}
    for lang, blk in sorted(gate["langs"].items()):
        rows = blk["utterances"][: a.limit] if a.limit else blk["utterances"]
        joined = []
        for r in rows:
            f = trace_features(a.binary, a.model, a.quant,
                               os.path.join(a.root, r["file"]), lang, a.hi)
            if f:
                joined.append({**r, **f})
        out[lang] = joined
        if not joined:
            continue
        print(f"\n=== {lang.upper()}   {len(joined)} utterances with a pre-crossing window")
        hf = sorted(x["hi_frac"] for x in joined)
        print(f"  high-norm fraction before the first token: median {st.median(hf):.2f}, "
              f"p10 {hf[len(hf)//10]:.2f}, p90 {hf[9*len(hf)//10]:.2f}")
        cut = st.median(hf)
        lo = [x for x in joined if x["hi_frac"] <= cut]
        hi = [x for x in joined if x["hi_frac"] > cut]
        print(f"  split at the median: {len(lo)} low, {len(hi)} high")
        print(f"  {'outcome':34} {'LOW norm':>12} {'HIGH norm':>12} {'delta':>10}")
        def show(label, fn, fmt="{:.4f}", guard=None):
            L = [fn(x) for x in lo if (guard is None or guard(x)) and fn(x) is not None]
            H = [fn(x) for x in hi if (guard is None or guard(x)) and fn(x) is not None]
            if not L or not H:
                print(f"  {label:34} {'—':>12} {'—':>12}")
                return
            a_, b_ = st.mean(L), st.mean(H)
            print(f"  {label:34} {fmt.format(a_):>12} {fmt.format(b_):>12} "
                  f"{b_-a_:+10.4f}")
        show("WER (mean)", lambda x: x["wer"])
        show("CER (mean)", lambda x: x["cer"])
        show("first word correct (rate)",
             lambda x: 1.0 if x["first_word_correct"] else 0.0,
             guard=lambda x: x["first_word_correct"] is not None)
        show("speech before first word (s)", lambda x: x["speech_at_first_word"], "{:.3f}")
        show("speech before first CORRECT (s)", lambda x: x["speech_at_first_correct_word"], "{:.3f}")
        show("blank margin before crossing", lambda x: x["margin_median"], "{:.2f}")
        show("pre-crossing decisions (count)", lambda x: float(x["n_pre"]), "{:.1f}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
