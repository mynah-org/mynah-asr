#!/usr/bin/env python3
"""Data gate before any GPU run on a multi-domain mix: per source and per eval set.

    $PY mix_stats.py --mix a.json:0.35,b.json:0.30 [--evals mls=x.json,cv=y.json] \
        [--steps 4000 --bs 16] [--sample 300] [--json out.json]

Prints, per manifest: utterances, hours, unique speakers, source sample-rate
distribution (the `src_sr` field written by prepare_mix.py; MLS/FLEURS from
prepare_it.py are 16 kHz natively), duration and level (sample peak / RMS dBFS
on --sample random clips) p10/p50/p90, missing audio files, three text examples
raw and as trained (the de-accented a-z normalisation of two_stage.py), and for
the mix: the weight, the expected hours drawn per source over --steps x --bs, and
how many passes over each source that means.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter

import numpy as np
import soundfile as sf


def norm(t):   # = two_stage.norm
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def pct(v, qs=(0.1, 0.5, 0.9)):
    v = sorted(v)
    return [round(v[min(len(v) - 1, int(q * len(v)))], 1) for q in qs] if v else None


def stats(name, path, sample, seed):
    rows = [json.loads(l) for l in open(path)]
    rng = random.Random(seed)
    miss = sum(1 for r in rows if not os.path.exists(r["audio_filepath"]))
    pk, rms = [], []
    for r in rng.sample(rows, min(sample, len(rows))):
        try:
            x, _ = sf.read(r["audio_filepath"], dtype="float32")
        except Exception:  # noqa: BLE001
            continue
        pk.append(float(20 * np.log10(np.abs(x).max() + 1e-9)))
        rms.append(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)))
    srs = Counter(r.get("src_sr", "16000 (native)") for r in rows)
    spk = {r.get("speaker") for r in rows if r.get("speaker")}
    ex = rng.sample(rows, min(3, len(rows)))
    s = {"name": name, "path": path, "n_utts": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
         "speakers": len(spk) or None, "missing_audio": miss, "src_sr": dict(srs.most_common()),
         "dur_s_p10_p50_p90": pct([r["duration"] for r in rows]), "peak_dbfs_p10_p50_p90": pct(pk),
         "rms_dbfs_p10_p50_p90": pct(rms), "mean_dur_s": round(float(np.mean([r["duration"] for r in rows])), 2),
         "examples": [{"raw": r["text"][:90], "trained": norm(r["text"])[:90]} for r in ex]}
    print(f"-- {name}: {s['n_utts']} utts {s['hours']} h, speakers {s['speakers']}, missing {miss}, src_sr {s['src_sr']}")
    print(f"   dur p10/50/90 {s['dur_s_p10_p50_p90']} s, peak dBFS {s['peak_dbfs_p10_p50_p90']}, RMS dBFS {s['rms_dbfs_p10_p50_p90']}")
    for e in s["examples"]:
        print(f"   raw: {e['raw']}\n   trn: {e['trained']}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--evals", default="")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {"train": [], "eval": []}
    mix = [(p, float(w)) for p, w in (x.rsplit(":", 1) for x in a.mix.split(","))]
    tot_w = sum(w for _, w in mix)
    print("== TRAIN sources")
    for p, w in mix:
        s = stats(os.path.basename(p).replace(".json", ""), p, a.sample, a.seed)
        s["weight"] = round(w / tot_w, 3)
        drawn_h = s["weight"] * a.steps * a.bs * s["mean_dur_s"] / 3600
        s["drawn_h"] = round(drawn_h, 1); s["passes"] = round(drawn_h / max(1e-9, s["hours"]), 2)
        out["train"].append(s)
    print(f"== MIX over {a.steps} steps x bs {a.bs}")
    for s in out["train"]:
        print(f"   {s['name']:16s} weight {s['weight']:.3f}  {s['hours']:7.2f} h  drawn ~{s['drawn_h']:6.1f} h  = {s['passes']:.2f} passes")
    if a.evals:
        print("== EVAL sets")
        for x in a.evals.split(","):
            n, p = x.split("=", 1)
            out["eval"].append(stats(n, p, a.sample, a.seed))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
