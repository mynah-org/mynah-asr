#!/usr/bin/env python3
"""compare_runs.py — do two runs agree, and may they be compared at all.

    python3 tools/bench/compare_runs.py soak-1.json soak-2.json
    python3 tools/bench/compare_runs.py before.json after.json --allow tree,dirty

WHY.  One soak is an anecdote.  The 2026-09-20 campaign ran each rung once and
then reasoned about differences of a few per cent between rungs, which is
inside what a single run of this harness varies by -- nobody knew that, because
no rung was ever repeated.  Two identical runs give the noise floor, and a
difference smaller than the noise floor is not a difference.

WHAT IT REFUSES.  It compares two runs only when their manifests agree on
everything that changes a number: model, quant, lookahead, concurrency, mode,
bank hash, frame size, pace, the commit.  Otherwise it says which key differs
and stops.  Comparing two runs that differ in an unknown way is the mistake
this file exists to prevent; `--allow` lets the caller name the keys it MEANT
to change (a before/after on the same bank differs in `tree`, and that is the
point of it).

Percentile definitions come from streaming_metrics.  Nothing is recomputed.

Exit: 0 they agree within tolerance · 1 they do not, or may not be compared
      · 2 usage/IO.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Manifest keys that must match, because each of them moves the numbers.
PINNED = ("model", "quant", "lookahead", "concurrency", "mode", "bank_hash",
          "frame_ms", "pace", "tree", "duration", "repeat")

# The lines worth comparing, and how they are read out of the summary.
LINES = (
    ("ttfp_ms", "p95", "TTFP p95", "ms"),
    ("emission_lag_ms", "p50", "emission lag p50", "ms"),
    ("emission_lag_ms", "p95", "emission lag p95", "ms"),
    ("emission_lag_ms", "p99", "emission lag p99", "ms"),
    ("finalization_lag_ms", "p95", "finalization p95", "ms"),
    ("backlog_max_s", "max", "backlog max", "s"),
    ("audio_per_wall", "max", "audio per wall s", "x"),
    ("cer", "p95", "CER p95", ""),
    ("max_delta_gap_ms", "max", "longest gap", "ms"),
)


def load(path):
    with open(path) as f:
        return json.load(f)


def manifest_of(doc):
    m = doc.get("manifest") or {}
    out = {k: m.get(k) for k in PINNED}
    # the bank is identified by the hash of its clips when the harness recorded one
    bank = m.get("bank")
    if isinstance(bank, dict) and out.get("bank_hash") is None:
        out["bank_hash"] = bank.get("hash") or bank.get("sha256")
    return out


def val(doc, key, field):
    s = ((doc.get("summary") or {}).get("metrics") or {}).get(key) or {}
    return s.get(field)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--allow", default="",
                    help="comma-separated manifest keys this comparison MEANT to change")
    ap.add_argument("--tolerance-pct", type=float, default=10.0,
                    help="relative difference still called agreement")
    ap.add_argument("--json")
    args = ap.parse_args()

    try:
        da, db = load(args.a), load(args.b)
    except (OSError, ValueError) as e:
        print(f"compare_runs: {e}", file=sys.stderr)
        return 2

    allow = {k.strip() for k in args.allow.split(",") if k.strip()}
    ma, mb = manifest_of(da), manifest_of(db)
    differ = [k for k in PINNED
              if ma.get(k) is not None and mb.get(k) is not None
              and ma[k] != mb[k] and k not in allow]
    print(f"A {os.path.basename(args.a)}")
    print(f"B {os.path.basename(args.b)}")
    for k in PINNED:
        if ma.get(k) is not None or mb.get(k) is not None:
            mark = "  <- differs" if k in differ else (
                "  <- allowed" if (k in allow and ma.get(k) != mb.get(k)) else "")
            if mark or ma.get(k) == mb.get(k):
                print(f"  {k:<12} {ma.get(k)}  |  {mb.get(k)}{mark}")
    if differ:
        print(f"\nREFUSED: these runs differ in {', '.join(differ)} and were not declared "
              f"as changed. Two unknowns are not a comparison; pass --allow "
              f"{','.join(differ)} if that is what you meant to vary.")
        return 1

    print(f"\n  {'line':<20} {'A':>12} {'B':>12} {'delta':>10} {'rel':>8}")
    rows, worst, worst_name = [], 0.0, None
    for key, field, title, unit in LINES:
        x, y = val(da, key, field), val(db, key, field)
        if x is None or y is None:
            continue
        d = y - x
        rel = abs(d) / abs(x) * 100.0 if abs(x) > 1e-12 else (0.0 if abs(d) < 1e-12 else float("inf"))
        rows.append({"line": title, "a": x, "b": y, "delta": d, "rel_pct": rel})
        if rel > worst:
            worst, worst_name = rel, title
        fmt = "{:>12.3f}" if unit == "s" or unit == "" or unit == "x" else "{:>12.0f}"
        print(f"  {title:<20}" + fmt.format(x) + fmt.format(y)
              + fmt.replace(">12", ">10").format(d) + f" {rel:>7.1f}%")

    ca = (da.get("summary") or {}).get("counts") or {}
    cb = (db.get("summary") or {}).get("counts") or {}
    lost = (ca.get("errors") or 0, cb.get("errors") or 0)
    print(f"\n  lost streams: {lost[0]} | {lost[1]}")

    ok = True
    if lost[0] or lost[1]:
        print("  DISAGREE: a run lost a stream. That disqualifies the point, not the comparison.")
        ok = False
    if worst_name is None:
        print("  no line had data in both runs")
        ok = False
    elif worst > args.tolerance_pct:
        print(f"  DISAGREE: '{worst_name}' moved {worst:.1f}% "
              f"(tolerance {args.tolerance_pct:.0f}%).")
        print("  Two identical runs that disagree by this much mean the noise floor of this")
        print("  harness at this operating point is at least that wide: no difference smaller")
        print("  than it may be called a result.")
        ok = False
    else:
        print(f"  AGREE: the widest move is '{worst_name}' at {worst:.1f}% "
              f"(tolerance {args.tolerance_pct:.0f}%).")
        print(f"  That is the noise floor of this harness here: a later change has to beat it.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"a": args.a, "b": args.b, "manifest_a": ma, "manifest_b": mb,
                       "allowed": sorted(allow), "lines": rows,
                       "worst_rel_pct": worst, "worst_line": worst_name,
                       "agree": ok}, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
