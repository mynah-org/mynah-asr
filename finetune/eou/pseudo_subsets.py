#!/usr/bin/env python3
"""Training subsets from a teacher-scored pseudo-label pool (teacher_ws.py output), BEFORE training.

    python3 pseudo_subsets.py --pool train_yg.teacher.json --out DIR [--thresholds 10,15,30]
        [--max-s 30] [--a2 "mls.json:0.35,cv.json:0.30,vp.json:0.20,fleurs.json:0.15"] [--bs 16]

Label = the teacher's text (Nemotron), not the pool's reference (Granary-Whisper); the
agreement WER (teacher vs reference) is the quality score. Clips whose teacher text contains
digits are dropped and counted (the a-z targets cannot spell them). Writes, all NESTED by
construction (same clips, same order):
  yg_agree<=T.json   for each threshold T
  yg_all.json        every clip that passes the digit / empty / duration checks
  yg_rand_eq<=T.json a random sample of yg_all with the SAME hours as the middle threshold:
                     cleaner labels vs more hours, separated
and prints, per subset: clips, hours, videos, duration p10/p50/p90/max, and the expected
audio seconds per optimizer step of a 50 % A2 / 50 % subset mix vs A2 alone (bs --bs), so a
B2 vs B2+Granary comparison declares how much speech per step changed.
"""
import argparse
import json
import random
import re
from pathlib import Path


def pct(v, qs=(0.1, 0.5, 0.9)):
    v = sorted(v)
    return [round(v[min(len(v) - 1, int(q * len(v)))], 1) for q in qs] + [round(v[-1], 1)] if v else None


def mean_dur(path, max_s):
    d = [json.loads(l)["duration"] for l in open(path)]
    d = [x for x in d if x <= max_s]
    return sum(d) / max(1, len(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="10,15,30")
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--a2", default="")
    ap.add_argument("--a2-max-s", type=float, default=20.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261010)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rs = [json.loads(l) for l in open(a.pool)]
    random.Random(a.seed).shuffle(rs)   # one fixed order for every subset
    n0 = len(rs)
    digits = [r for r in rs if re.search(r"\d", r.get("teacher_raw", ""))]
    keep = [r for r in rs if r.get("teacher") and not re.search(r"\d", r.get("teacher_raw", "")) and r["duration"] <= a.max_s
            and r.get("teacher_wer") is not None]
    print(f"== pool {n0} clips {sum(r['duration'] for r in rs) / 3600:.1f} h; dropped: teacher digits {len(digits)}, "
          f"empty {sum(1 for r in rs if not r.get('teacher'))}, > {a.max_s:g} s {sum(1 for r in rs if r['duration'] > a.max_s)}")

    def write(name, xs):
        with open(out / f"{name}.json", "w") as f:
            for r in xs:
                row = {k: v for k, v in r.items() if k not in ("teacher", "teacher_raw", "teacher_lang", "_wps")}
                row["text"] = r["teacher"]; row["ref_granary"] = r["text"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return xs

    subsets = {"yg_all": write("yg_all", keep)}
    ths = [float(t) for t in a.thresholds.split(",")]
    for t in ths:
        subsets[f"yg_agree<={t:g}"] = write(f"yg_agree<={t:g}", [r for r in keep if r["teacher_wer"] <= t])
    mid = ths[len(ths) // 2]
    target = sum(r["duration"] for r in subsets[f"yg_agree<={mid:g}"])
    rnd, acc = [], 0.0
    for r in keep:   # the same shuffled order: a random, unfiltered draw of equal hours
        if acc >= target:
            break
        rnd.append(r); acc += r["duration"]
    subsets[f"yg_rand_eq<={mid:g}"] = write(f"yg_rand_eq<={mid:g}", rnd)

    a2_mean = None
    if a.a2:
        parts = [(p, float(w)) for p, w in (x.rsplit(":", 1) for x in a.a2.split(","))]
        tw = sum(w for _, w in parts)
        a2_mean = sum(w / tw * mean_dur(p, a.a2_max_s) for p, w in parts)
        print(f"   A2 mix alone: mean clip {a2_mean:.1f} s -> {a.bs * a2_mean:.0f} audio s / step")
    from collections import Counter
    base_v = Counter(r.get("speaker") for r in keep)
    print("| subset | clips | h | videos | videos losing > 50 % of their clips | top-10 % videos' share of h | dur p10/p50/p90/max s | 50/50 with A2: audio s / step (x A2) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, xs in subsets.items():
        d = [r["duration"] for r in xs]
        m = sum(d) / max(1, len(d))
        mix = f"{a.bs * (a2_mean + m) / 2:.0f} ({(a2_mean + m) / 2 / a2_mean:.2f}x)" if a2_mean else "-"
        cv = Counter(r.get("speaker") for r in xs)
        lost = sum(1 for v, c in base_v.items() if cv.get(v, 0) < 0.5 * c)
        hv = sorted((sum(r["duration"] for r in xs if r.get("speaker") == v) for v in cv), reverse=True) if len(cv) < 20000 else []
        top = sum(hv[: max(1, len(hv) // 10)]) / max(1e-9, sum(hv)) if hv else float("nan")
        print(f"| {name} | {len(xs)} | {sum(d) / 3600:.1f} | {len(cv)} | {lost} | {100 * top:.0f} % | {pct(d)} | {mix} |")


if __name__ == "__main__":
    main()
