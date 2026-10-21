#!/usr/bin/env python3
"""How many stage-2 training examples teach "internal pause, speech continues, no <EOU>"? (CPU, model-free)

    train_pause_audit.py --mix "a.json:w,b.json:w,..." [--n 3000] [--max-dur 20] --out audit.json

The stage-2 v4 target is text + <EOU> at the very end (two_stage.py --eou-append 1); every silence INSIDE a clip is
therefore an implicit negative. With the frozen energy rule (80 ms frames, speech = within 35 dB of the clip peak)
this measures, per mix source (up to --n clips, the v4 --max-dur filter applied) and mix-weighted:
  - internal pauses (a silence run with speech before AND after): count per clip, duration p50 / p90 / p99, share of
    clips whose longest internal pause is >= 0.24 / 0.4 / 0.56 / 0.8 / 1.04 / 1.6 s, their level (dB re peak);
  - the terminal silence after the last speech frame (natural trail; v4 adds U(1, 3) s at -90..-60 dB on 50 %).
"""
import argparse
import json
import random

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mix", required=True)
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--max-dur", type=float, default=20.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
FR = 0.08
TH = (0.24, 0.4, 0.56, 0.8, 1.04, 1.6)


def runs(x):
    n = int(FR * 16000)
    f = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10); db -= db.max()
    sm = db > -35
    on = np.nonzero(sm)[0]
    if not len(on):
        return [], 0.0, []
    internal, lv, j, last = [], [], on[0], on[-1]
    while j <= last:
        if not sm[j]:
            k = j
            while not sm[k]:
                k += 1
            internal.append((k - j) * FR); lv.append(float(db[j:k].mean())); j = k
        else:
            j += 1
    return internal, (len(sm) - 1 - last) * FR, lv


res, W = {}, {}
for spec in a.mix.split(","):
    p, w = spec.rsplit(":", 1)
    name = p.split("/")[-1].replace(".json", "")
    rows = [json.loads(l) for l in open(p)]
    rows = [r for r in rows if r.get("duration", 0) <= a.max_dur]
    random.Random(0).shuffle(rows)
    rows = rows[: a.n]
    per, allp, trail, levels, hours = [], [], [], [], 0.0
    for r in rows:
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        ip, tr, lv = runs(x)
        per.append(max(ip) if ip else 0.0); allp += ip; trail.append(tr); levels += lv; hours += len(x) / 16000 / 3600
    per, allp, trail = np.array(per), np.array(allp), np.array(trail)
    q = lambda v, qs: [round(float(np.percentile(v, x)), 2) for x in qs] if len(v) else None
    res[name] = {"weight": float(w), "clips": len(rows), "h": round(hours, 2),
                 "internal_pauses_per_clip": round(len(allp) / len(rows), 2),
                 "internal_pause_s_p50_p90_p99": q(allp, (50, 90, 99)),
                 "clips_with_longest_internal_pause_ge_s_%": {str(t): round(100 * float((per >= t).mean()), 1) for t in TH},
                 "internal_pauses_ge_0.56s_per_hour": round(float((allp >= 0.56).sum()) / max(hours, 1e-9), 1),
                 "internal_pause_db_re_peak_p50": round(float(np.median(levels)), 1) if levels else None,
                 "trail_s_p10_p50_p90": q(trail, (10, 50, 90))}
    W[name] = float(w)
    print("==", name, json.dumps(res[name]), flush=True)
tw = sum(W.values())
res["MIX_WEIGHTED"] = {"clips_with_longest_internal_pause_ge_s_%": {
    str(t): round(sum(W[k] * res[k]["clips_with_longest_internal_pause_ge_s_%"][str(t)] for k in W) / tw, 1) for t in TH}}
print("== MIX_WEIGHTED", json.dumps(res["MIX_WEIGHTED"]), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
