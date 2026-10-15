#!/usr/bin/env python3
"""Compare method-tournament arms: convergence curve, cost, loss/gradient behaviour and error composition.

    python3 arm_report.py --arms r0-ctl,r2-ctc01 --runs ~/ft/runs --logs ~/ft/logs --dumps ~/ft/analysis/arms \
        [--posts 0,0.5] [--dev mls,fleurs,cv,vp] [--report tedx,italic] [--md out.md]

Per arm (two_stage.py run plain-<arm>, its log <arm>.log, greedy_blank_sweep.py dumps <arm>_post<P>.jsonl):
  curve      in-run macro val WER / CER at every eval step, plus per-set WER (the harness, raw EOF);
  cost       optimizer steps, processed audio hours, wall seconds, audio hours per wall hour;
  training   mean loss / RNNT / CTC terms and encoder gradient norm (pre-clip) over log windows;
  errors     per set and decoding (raw EOF, trailing flush): WER, S_near, S_far, D, I per 100 reference words,
             empty-hypothesis rate (nbest_gap.py classes: near = within 2 chars or a shared >= 4-letter prefix).
The dev sets choose; the report sets (clean-it) are post-hoc and printed apart.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbest_gap import classify  # noqa: E402

STEP_RE = re.compile(r"step (\d+)/\d+ loss ([\d.]+)(?: \(rnnt ([\d.]+) ctc ([\d.]+)\))?(?: enc-gn ([\d.]+))?")


def log_windows(path, edges):
    rows = []
    for l in open(path):
        m = STEP_RE.search(l)
        if m:
            rows.append(tuple(None if x is None else float(x) for x in m.groups()))
    out = []
    for lo, hi in zip(edges, edges[1:]):
        w = [r for r in rows if lo < r[0] <= hi]
        if not w:
            continue
        avg = lambda i: round(sum(r[i] for r in w) / len(w), 2) if all(r[i] is not None for r in w) else None
        out.append({"steps": f"{lo}-{hi}", "loss": avg(1), "rnnt": avg(2), "ctc": avg(3), "enc_gn": avg(4)})
    return out


def errors(dump, post):
    agg = defaultdict(lambda: defaultdict(float))
    for l in open(dump):
        x = json.loads(l)
        h = x["hyp"].get(post) or x["hyp"].get(str(float(post))) or next(iter(x["hyp"].values()))
        c = classify(x["ref"], h)
        a = agg[x["set"]]
        for k in ("S_near", "S_far", "D", "I", "err", "n"):
            a[k] += c[k]
        a["clips"] += 1; a["empty"] += not h.strip()
    return {s: {"wer": round(100 * a["err"] / a["n"], 2), **{k: round(100 * a[k] / a["n"], 2) for k in ("S_near", "S_far", "D", "I")},
                "empty_pct": round(100 * a["empty"] / a["clips"], 1)} for s, a in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--posts", default="0,0.5")
    ap.add_argument("--dev", default="mls,fleurs,cv,vp")
    ap.add_argument("--report", default="tedx,italic")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    arms, dev, rep = a.arms.split(","), a.dev.split(","), a.report.split(",")
    L = []
    p = lambda s="": L.append(s)
    p("### Convergence (in-run harness val, raw EOF, greedy)\n")
    p("| arm | step | macro WER | CER | " + " | ".join(dev) + " |")
    p("|---|---|---|---|" + "---|" * len(dev))
    cost = {}
    for arm in arms:
        d = json.load(open(f"{a.runs}/plain-{arm}/metrics.json"))
        for r in d["log"]:
            if "val_wer" in r:
                p(f"| {arm} | {r['step']} | {r['val_wer']} | {r['val_cer']} | " + " | ".join(str(r.get(f"val_{s}_wer", "")) for s in dev) + " |")
        cost[arm] = (max(r["step"] for r in d["log"]), d["audio_h_seen"], d["wall_s"], d["audio_h_per_gpu_h"])
    p("\n### Cost\n")
    p("| arm | steps | audio h | wall s | audio h / wall h |")
    p("|---|---|---|---|---|")
    for arm, (st, ah, ws, rt) in cost.items():
        p(f"| {arm} | {st} | {ah} | {ws} | {rt} |")
    p("\n### Training signals (window means; enc-gn = encoder grad norm before clipping)\n")
    p("| arm | steps | loss | rnnt | ctc | enc-gn |")
    p("|---|---|---|---|---|---|")
    for arm in arms:
        st = cost[arm][0]
        edges = [0, 100, 250, 500, 1000, 2000, 4000, 8000]
        for w in log_windows(f"{a.logs}/{arm}.log", [e for e in edges if e < st] + [st]):
            p(f"| {arm} | {w['steps']} | {w['loss']} | {w['rnnt'] if w['rnnt'] is not None else w['loss']} | {w['ctc'] or ''} | {w['enc_gn']} |")
    for post in a.posts.split(","):
        for title, sets in (("dev (harness)", dev), ("report (clean-it, post hoc)", rep)):
            p(f"\n### Errors at the last step, {title}, trailing silence {post} s (per 100 ref words)\n")
            p("| arm | set | WER | S_near | S_far | D | I | empty % |")
            p("|---|---|---|---|---|---|---|---|")
            for arm in arms:
                f = f"{a.dumps}/{arm}_post{post}.jsonl"
                if not os.path.exists(f):
                    continue
                e = errors(f, "0.0")
                for s in sets:
                    if s in e:
                        x = e[s]
                        p(f"| {arm} | {s} | {x['wer']} | {x['S_near']} | {x['S_far']} | {x['D']} | {x['I']} | {x['empty_pct']} |")
                ws = [e[s]["wer"] for s in sets if s in e]
                if ws:
                    p(f"| {arm} | **macro** | **{round(sum(ws) / len(ws), 2)}** | | | | | |")
    txt = "\n".join(L)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
