#!/usr/bin/env python3
"""In-run trajectory of a mixed run against a reference run, at equal steps AND at equal A2 exposure.

    python3 traj_compare.py --log /root/ft/logs/it-p1-add25.log --ref /root/ft/runs/plain-it-e3-s8000/metrics.json \
        --a2-share 0.75 [--last 1]

Reads the VAL lines of a running two_stage.py log; prints the harness macro (mls/fleurs/cv/vp, recomputed even when
the run evaluates extra sets), the reference at the same step and at the nearest OBSERVED step to the A2-equivalent
step (steps x a2-share; never interpolated), and any extra val sets (e.g. tedx/italic) as they appear.
"""
import argparse
import json
import re

H = ("mls", "fleurs", "cv", "vp")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--ref", required=True, help="metrics.json of the reference run")
    ap.add_argument("--a2-share", type=float, default=0.75)
    ap.add_argument("--last", type=int, default=0)
    a = ap.parse_args()
    ref = {r["step"]: r for r in json.load(open(a.ref))["log"] if "val_wer" in r}
    st, rows = None, []
    for l in open(a.log):
        m = re.search(r"step (\d+)/", l)
        if m:
            st = int(m.group(1))
        if l.lstrip().startswith("VAL"):
            rows.append((st, json.loads(l[l.index("{"):])))
    for st, d in rows[-a.last:]:
        h = sum(d[f"val_{k}_wer"] for k in H) / 4
        a2 = int(st * a.a2_share)
        near = min(ref, key=lambda s: abs(s - a2))
        same = f"{ref[st]['val_wer']:.2f}" if st in ref else "-"
        extra = " ".join(f"{k[4:-4]} {v:.2f}" for k, v in d.items()
                         if k.endswith("_wer") and k[4:-4] not in H and k != "val_wer")
        print(f"step {st} (A2-eq {a2}): harness {h:.2f} | ref same step {same} | ref @{near} (observed, nearest A2-eq) "
              f"{ref[near]['val_wer']:.2f} | {extra} | empty {d['val_empty']}")


if __name__ == "__main__":
    main()
