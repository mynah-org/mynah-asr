#!/usr/bin/env python3
"""Score s2_reset.py dumps on DEV-EOU-FR-v2 (internal continuation vs endpoint), frozen definitions, plus the
diagnostic confirmation-window frontier.

    eou_v2_score.py --manifest eou_all.json --dumps v4=d4.jsonl,v2=d2.jsonl [--base v4] --out score.json

From the dump (protocol s2_reset; frozen FE / recall / latency rules of s2_reset.py, energy endpoint of the original
clip): per class (internal / endpoint), source and stratum: clips, reset-WER, empty, FE % (an <EOU> with >= 0.5 s of
speech left), FE_any %, recall % and latency p50 / p90 (endpoint class), and for the internal class the share of clips
with an <EOU> inside the TARGET pause (the clip's longest internal pause, from the manifest; [start, end + 0.16 s]).
Relative change vs --base. Window frontier (diagnostic, never a selection): an <EOU> is reported only if no speech
within W s after it -> a premature <EOU> is cancelled iff speech resumes within W; every true endpoint is W later.
"""
import argparse
import json

import numpy as np
import soundfile as sf

FR = 0.08
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--manifest", required=True)
ap.add_argument("--dumps", required=True)
ap.add_argument("--base", default="v4")
ap.add_argument("--out", required=True)
a = ap.parse_args()
man = {r["audio_filepath"]: r for r in map(json.loads, open(a.manifest))}


def lev(x, y):
    p = list(range(len(y) + 1))
    for i, u in enumerate(x, 1):
        c = [i]
        for j, v in enumerate(y, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (u != v)))
        p = c
    return p[-1]


sm_cache = {}


def speech_mask(path):
    if path not in sm_cache:
        x, _ = sf.read(path, dtype="float32")
        n = int(FR * 16000)
        f = x[: len(x) // n * n].reshape(-1, n)
        db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
        sm_cache[path] = db > db.max() - 35
    return sm_cache[path]


def per_clip(d):
    r = man[d["audio"]]
    ef = d["eou_frames"]; end = d["speech_end_s"]; sm = speech_mask(d["audio"])
    prem = [f for f in ef if f * FR < end - 0.5]
    tgt = None
    if r["class"] == "internal":
        ps = [p for p in r["internal_pauses"] if p[1] - p[0] >= 0.8]
        s, e = max(ps, key=lambda p: p[1] - p[0])
        tgt = any(s <= f * FR <= e + 0.16 for f in ef)
    resume = []
    for f in prem:
        nx = np.nonzero(sm[f + 1:])[0]
        resume.append((nx[0] + 1) * FR if len(nx) else 99.0)
    ref, hyp = d["ref"].split(), d["hyp"].split()
    return {"class": r["class"], "source": r["source"], "stratum": r["stratum"], "err": lev(ref, hyp), "N": len(ref),
            "empty": not hyp, "fe": bool(prem), "fe_any": bool(d["fe_any"]), "true": d["true_eou_s"] is not None,
            "lat": (d["true_eou_s"] - end) if d["true_eou_s"] is not None else None, "tgt": tgt, "resume": resume}


def agg(rs, W=0.0):
    if not rs:
        return None
    n = len(rs)
    fe = [any(t >= W for t in r["resume"]) if W else r["fe"] for r in rs]
    lat = [r["lat"] + W for r in rs if r["lat"] is not None]
    o = {"clips": n, "wer": round(100 * sum(r["err"] for r in rs) / max(1, sum(r["N"] for r in rs)), 2),
         "empty": sum(r["empty"] for r in rs), "FE%": round(100 * sum(fe) / n, 2),
         "FE_any%": round(100 * sum(r["fe_any"] for r in rs) / n, 2) if not W else None,
         "recall%": round(100 * sum(r["true"] for r in rs) / n, 2),
         "lat_p50_p90": [round(float(np.percentile(lat, q)), 2) for q in (50, 90)] if lat else None}
    if rs[0]["class"] == "internal" and not W:
        o["eou_in_target_pause%"] = round(100 * sum(bool(r["tgt"]) for r in rs) / n, 2)
    return o


res = {}
for spec in a.dumps.split(","):
    name, p = spec.split("=", 1)
    rows = [per_clip(d) for d in map(json.loads, open(p)) if d["protocol"] == "s2_reset" and d["audio"] in man]
    I = [r for r in rows if r["class"] == "internal"]; E = [r for r in rows if r["class"] == "endpoint"]
    out = {"internal": agg(I), "endpoint": agg(E), "all_wer": agg(rows)["wer"],
           "internal_by_stratum": {s: agg([r for r in I if r["stratum"] == s]) for s in sorted({r["stratum"] for r in I})},
           "by_source": {s: {"internal": agg([r for r in I if r["source"] == s]), "endpoint": agg([r for r in E if r["source"] == s])}
                         for s in sorted({r["source"] for r in rows})},
           "window_frontier": {str(W): {"internal_FE%": agg(I, W)["FE%"], "endpoint_recall%": agg(E, W)["recall%"],
                                        "endpoint_lat_p50_p90": agg(E, W)["lat_p50_p90"]} for W in (0.16, 0.32, 0.48, 0.64, 0.8, 1.04)}}
    res[name] = out
    print(f"== {name}: internal {json.dumps(out['internal'])}\n   endpoint {json.dumps(out['endpoint'])}\n   all WER {out['all_wer']}", flush=True)
    for s, v in out["internal_by_stratum"].items():
        print(f"   {name} internal {s}: {json.dumps(v)}", flush=True)
    print(f"   {name} window frontier {json.dumps(out['window_frontier'])}", flush=True)
if a.base in res:
    b = res[a.base]
    for k, v in res.items():
        if k == a.base:
            continue
        rel = lambda x, y: round(100 * (x - y) / y, 1) if y else None  # noqa: E731
        v["vs_base"] = {"internal_FE_rel%": rel(v["internal"]["FE%"], b["internal"]["FE%"]),
                        "internal_FE_rel%_by_stratum": {s: rel(v["internal_by_stratum"][s]["FE%"], b["internal_by_stratum"][s]["FE%"])
                                                        for s in v["internal_by_stratum"] if s in b["internal_by_stratum"]},
                        "endpoint_recall_pp": round(v["endpoint"]["recall%"] - b["endpoint"]["recall%"], 2),
                        "endpoint_lat_p50_d": round(v["endpoint"]["lat_p50_p90"][0] - b["endpoint"]["lat_p50_p90"][0], 2),
                        "endpoint_lat_p90_d": round(v["endpoint"]["lat_p50_p90"][1] - b["endpoint"]["lat_p50_p90"][1], 2)}
        print(f"== {k} vs {a.base}: {json.dumps(v['vs_base'])}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
