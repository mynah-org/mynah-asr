#!/usr/bin/env python3
"""cer_offline.py — CER against human references on an idle box, and the
baseline a later run is compared against.

    python3 tools/eval/cer_offline.py --model models/nemotron-... --quant int8 \
        --manifest samples/manifest.json --json cer.json
    python3 tools/eval/cer_offline.py ... --baseline configs/quality/nemotron-int8.json

WHY IT IS SEPARATE FROM THE LOAD RUN.  A CER measured while the box is
saturated answers two questions at once and separates neither: a transcript can
degrade because the model is wrong or because the server dropped audio, and the
load run cannot tell them apart.  This runs first, alone, and gives the number
the loaded run has to match.

WHY A BASELINE AND NOT A CONSTANT.  The suites in this repo gate on absolute
thresholds -- 0.20 here, 0.30 there, 0.25 in the streaming harness.  A model
whose CER goes from 0.13 to 0.19 passes every one of them while having got
much worse.  A baseline says what THIS model at THIS quant scored when it was
known good, and the gate is the distance from it.  The margin defaults to 0.02,
which is above the int8-vs-f32 delta this repo has recorded (0.133 -> 0.145
over 34 locales, docs/quantization.md) and below any regression worth shipping.

Every CER, every normaliser and every percentile comes from
tools/bench/streaming_metrics.py.  This file runs the CLI and does arithmetic
on what comes back.

Exit: 0 within the baseline (or no baseline given) · 1 a regression · 2 usage
      · 77 the model or the manifest is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
from streaming_metrics import cer, normalise, pct, wer  # noqa: E402


def load_manifest(path):
    """[(clip_path, reference_text, lang), ...] from a bank manifest."""
    with open(path) as f:
        doc = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    out = []
    if isinstance(doc, dict) and isinstance(doc.get("samples"), list):
        for e in doc["samples"]:
            fn, txt = e.get("file"), e.get("text")
            if fn and txt:
                out.append((os.path.join(base, fn), txt, e.get("lang")))
    elif isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, str):
                out.append((os.path.join(base, k), v, None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--lookahead", type=int, default=-1,
                    help="-1 = offline transcribe; >=0 = the streaming path at that lookahead")
    ap.add_argument("--mynah-asr", default="./mynah-asr")
    ap.add_argument("--limit", type=int, default=0, help="0 = every clip in the manifest")
    ap.add_argument("--json", help="write the result here")
    ap.add_argument("--baseline", help="a previous result to compare against")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="how much worse than the baseline mean CER is still allowed")
    a = ap.parse_args()

    if not os.path.isdir(a.model):
        print(f"cer_offline: {a.model} is not a directory", file=sys.stderr)
        return 77
    if not os.path.exists(a.manifest):
        print(f"cer_offline: {a.manifest} does not exist "
              f"(for the stress bank: make fetch-stress-bank)", file=sys.stderr)
        return 77
    if not os.path.exists(a.mynah_asr):
        print(f"cer_offline: {a.mynah_asr} is not built", file=sys.stderr)
        return 2

    items = load_manifest(a.manifest)
    items = [it for it in items if os.path.exists(it[0])]
    if not items:
        print("cer_offline: the manifest named no clip that exists", file=sys.stderr)
        return 77
    if a.limit > 0:
        items = items[:a.limit]

    mode = "stream" if a.lookahead >= 0 else "transcribe"
    print(f"cer_offline: {len(items)} clip(s), {a.model}, quant {a.quant}, mode {mode}"
          + (f", lookahead {a.lookahead}" if a.lookahead >= 0 else ""))

    rows, t0 = [], time.monotonic()
    for clip, ref, lang in items:
        cmd = [a.mynah_asr, mode, "-m", a.model, "-i", clip, "--quant", a.quant]
        if a.lookahead >= 0:
            cmd += ["--lookahead", str(a.lookahead)]
        if lang:
            cmd += ["--lang", lang]      # the CLI resolves "it" to "it-IT" itself
        r = subprocess.run(cmd, capture_output=True, text=True)
        hyp = (r.stdout or "").strip()
        if r.returncode != 0:
            err = (r.stderr or "")[-200:]
            # A language this pack does not serve is a refusal, not a defect: it
            # is counted apart so a baseline is not read as having two failures.
            kind = "unsupported_language" if "is not supported" in err else "error"
            rows.append({"clip": clip, "lang": lang, kind: err, "cer": None})
            continue
        rows.append({"clip": clip, "lang": lang, "cer": cer(hyp, ref), "wer": wer(hyp, ref),
                     "ref_chars": len(normalise(ref)), "hyp": hyp[:400]})
    wall = time.monotonic() - t0

    scored = [r for r in rows if r.get("cer") is not None]
    errs = [r for r in rows if r.get("error")]
    unsup = [r for r in rows if r.get("unsupported_language")]
    if not scored:
        print("cer_offline: nothing scored", file=sys.stderr)
        return 1
    cers = [r["cer"] for r in scored]
    wers = [r["wer"] for r in scored if r.get("wer") is not None]
    # Weighted by reference length: the mean of per-clip rates over-weights the
    # short clips, and a corpus CER is errors over characters.
    tot_chars = sum(r["ref_chars"] for r in scored)
    weighted = (sum(r["cer"] * r["ref_chars"] for r in scored) / tot_chars) if tot_chars else None
    worst = max(scored, key=lambda r: r["cer"])

    result = {
        "model": os.path.basename(os.path.abspath(a.model)), "quant": a.quant,
        "mode": mode, "lookahead": a.lookahead,
        "manifest": os.path.relpath(os.path.abspath(a.manifest), ROOT),
        "clips": len(rows), "scored": len(scored), "errors": len(errs),
        "unsupported_language": len(unsup),
        "cer_mean": sum(cers) / len(cers), "cer_weighted": weighted,
        "cer_p50": pct(cers, 50), "cer_p95": pct(cers, 95), "cer_max": max(cers),
        "wer_mean": (sum(wers) / len(wers)) if wers else None,
        # the LAST TWO path components, never the bare file name: this repo's own
        # bank has eleven fleurs_1521.wav and naming one of them tells nobody which
        "worst": {"clip": "/".join(worst["clip"].replace("\\", "/").split("/")[-2:]),
                  "cer": worst["cer"]},
        "wall_s": wall, "rows": rows,
    }

    print(f"  scored {len(scored)}/{len(rows)}"
          + (f", {len(errs)} error(s)" if errs else "")
          + (f", {len(unsup)} clip(s) in a language this pack does not serve" if unsup else ""))
    print(f"  CER mean {result['cer_mean']:.4f}  weighted {weighted:.4f}"
          f"  p95 {result['cer_p95']:.4f}  max {result['cer_max']:.4f}")
    if result["wer_mean"] is not None:
        print(f"  WER mean {result['wer_mean']:.4f}")
    print(f"  worst clip: {result['worst']['clip']} at {result['worst']['cer']:.4f}")
    print(f"  {wall:.0f} s")

    rc = 0
    if a.baseline:
        if not os.path.exists(a.baseline):
            print(f"  no baseline at {a.baseline}: writing this run is the way to create one")
        else:
            with open(a.baseline) as f:
                b = json.load(f)
            same = all(b.get(k) == result.get(k) for k in ("model", "quant", "mode",
                                                           "lookahead", "manifest"))
            if not same:
                print("  REFUSED to compare: the baseline was taken with a different "
                      "model/quant/mode/lookahead/manifest. Two unknowns are not a comparison.")
                print(f"    baseline: {[b.get(k) for k in ('model','quant','mode','lookahead','manifest')]}")
                print(f"    this run: {[result.get(k) for k in ('model','quant','mode','lookahead','manifest')]}")
                rc = 1
            else:
                d = result["cer_mean"] - b["cer_mean"]
                verdict = "PASS" if d <= a.margin + 1e-12 else "REGRESSION"
                print(f"  vs baseline {b['cer_mean']:.4f}: delta {d:+.4f} "
                      f"(margin {a.margin:+.4f}) -> {verdict}")
                if verdict != "PASS":
                    rc = 1
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        print(f"  -> {a.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
