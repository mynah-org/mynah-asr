#!/usr/bin/env python3
"""Difficulty buckets for a training mix, and a curriculum mix that re-weights them INSIDE each source.

    python3 curriculum_mix.py --mix "mls.json:0.1964,...,yg6.json:0.2503" --student student.jsonl --out DIR \
        [--mult "hard=2,solved=0.5,mid=1,suspect=1,lowcred=1"]

Per clip (student = a reference checkpoint's greedy WER against the training label, student_score.py):
  credibility, estimated WITHOUT the student:
    Granary pools (label = Nemotron text): two-teacher agreement (teacher vs Granary-Whisper WER, `teacher_wer`)
        <= 5 high, else mid (the agree<=15 subsets hold nothing above 15);
    VP (human label, teacher-scored): teacher_wer <= 10 high, <= 25 mid, else low;
    MLS / CV / FLEURS (human labels, no teacher): high, unless the label rate is implausible
    (chars per second outside the source's p1..p99) or the label is empty -> low.
  buckets (first match):
    suspect   VP only: teacher and student agree with each other (WER <= 10) but both disagree with the label
              (teacher_wer >= 30): label / segmentation suspect;
    lowcred   credibility low;
    solved    student_wer <= 5;
    hard      credibility high and student_wer >= 30   (the "learnable-hard" target);
    mid       everything else.
Without --mult: writes stats only (clips, hours, mean student / teacher WER, share of the source's clips, per
source x bucket). With --mult: one manifest per source x bucket (rows unchanged) in DIR and the MIX string in
DIR/mix.txt; each source keeps its total weight, split over its buckets as n_b * mult_b (the trainer draws a
source by weight, then a clip uniformly), so ONLY the difficulty distribution inside a source changes.
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BUCKETS = ["hard", "mid", "solved", "lowcred", "suspect"]


def lev(x, y):
    prev = list(range(len(y) + 1))
    for i, p in enumerate(x, 1):
        cur = [i]
        for j, q in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != q)))
        prev = cur
    return prev[-1]


def src_name(path):   # it-yg/subsets/yg_agree<=15.json and it-yg6/... must not collide
    q = Path(path)
    d = q.parent.parent.name if q.parent.name == "subsets" else q.parent.name
    return f"{d}-{q.stem}".replace("<=", "le")


def src_kind(path):
    p = path.lower()
    if "yg" in Path(p).parent.parent.name or "agree" in p:
        return "granary"
    if "vp" in Path(p).name:
        return "vp"
    return "human"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mult", default="", help="bucket multipliers, e.g. hard=2,solved=0.5 (others 1): shown as weight after")
    ap.add_argument("--write", type=int, default=0, help="1 = write the bucket manifests and mix.txt (needs --mult)")
    ap.add_argument("--max-dur", type=float, default=30.0)
    a = ap.parse_args()
    st = {}
    for l in open(a.student):
        r = json.loads(l); st[r["audio_filepath"]] = r
    mix = [(p, float(w)) for p, w in (x.rsplit(":", 1) for x in a.mix.split(","))]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    table, buckets_of = [], {}
    for p, w in mix:
        kind = src_kind(p)
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if r.get("duration", 0) <= a.max_dur]
        miss = sum(1 for r in rows if r["audio_filepath"] not in st)
        if miss:
            sys.exit(f"== {p}: {miss} clips without a student score")
        cps = np.array([len(st[r["audio_filepath"]]["ref"]) / max(0.1, r["duration"]) for r in rows])
        lo, hi = np.percentile(cps, 1), np.percentile(cps, 99)
        bk = defaultdict(list)
        for r, c in zip(rows, cps):
            s = st[r["audio_filepath"]]; sw = s["student_wer"]; tw = r.get("teacher_wer")
            if kind == "granary":
                cred = "high" if tw is not None and tw <= 5 else "mid"
            elif kind == "vp":
                cred = "high" if tw is not None and tw <= 10 else ("mid" if tw is not None and tw <= 25 else "low")
            else:
                cred = "high" if s["ref_words"] > 0 and lo <= c <= hi else "low"
            if kind == "vp" and tw is not None and tw >= 30 and r.get("teacher") is not None and \
                    100 * lev(s["student"].split(), r["teacher"].split()) / max(1, len(r["teacher"].split())) <= 10:
                b = "suspect"
            elif cred == "low":
                b = "lowcred"
            elif sw <= 5:
                b = "solved"
            elif cred == "high" and sw >= 30:
                b = "hard"
            else:
                b = "mid"
            bk[b].append((r, sw, tw))
        n = len(rows)
        for b in BUCKETS:
            xs = bk.get(b, [])
            if not xs:
                continue
            tws = [t for _, _, t in xs if t is not None]
            sts = [st[r["audio_filepath"]] for r, _, _ in xs]
            rw = sum(x["ref_words"] for x in sts)
            table.append({"source": src_name(p), "bucket": b, "clips": len(xs),
                          "median_s": round(float(np.median([r["duration"] for r, _, _ in xs])), 1),
                          "hyp_ref_word_ratio": round(sum(len(x["student"].split()) for x in sts) / max(1, rw), 3),
                          "empty_pct": round(100 * sum(1 for x in sts if not x["student"].strip()) / len(xs), 1),
                          "hours": round(sum(r["duration"] for r, _, _ in xs) / 3600, 2),
                          "student_wer": round(float(np.mean([s for _, s, _ in xs])), 1),
                          "teacher_wer": round(float(np.mean(tws)), 1) if tws else None,
                          "share_clips": round(100 * len(xs) / n, 1)})
        buckets_of[p] = (w, bk)
        mult = {k: float(v) for k, v in (x.split("=") for x in a.mult.split(","))} if a.mult else {}
        Z = sum(len(bk[b]) * mult.get(b, 1.0) for b in bk)
        for t in table:
            if t["source"] == src_name(p):
                t["w_before"] = round(w * t["clips"] / n, 4)
                t["w_after"] = round(w * t["clips"] * mult.get(t["bucket"], 1.0) / Z, 4)
    hdr = ("| source | bucket | clips | hours | median s | student WER | teacher/label WER | hyp/ref words | empty % "
           "| % source clips | weight before | weight after |")
    lines = [hdr, "|" + "---|" * 12] + [
        f"| {t['source']} | {t['bucket']} | {t['clips']} | {t['hours']} | {t['median_s']} | {t['student_wer']} | "
        f"{t['teacher_wer'] if t['teacher_wer'] is not None else '-'} | {t['hyp_ref_word_ratio']} | {t['empty_pct']} | "
        f"{t['share_clips']} | {t['w_before']} | {t['w_after']} |" for t in table]
    tot = defaultdict(lambda: [0, 0.0])
    for t in table:
        tot[t["bucket"]][0] += t["clips"]; tot[t["bucket"]][1] += t["hours"]
    wb = defaultdict(lambda: [0.0, 0.0])
    for t in table:
        wb[t["bucket"]][0] += t["w_before"]; wb[t["bucket"]][1] += t["w_after"]
    lines += ["", "| bucket | clips | hours | draw share before | draw share after |", "|---|---|---|---|---|"] + [
        f"| {b} | {tot[b][0]} | {tot[b][1]:.1f} | {100 * wb[b][0]:.1f} % | {100 * wb[b][1]:.1f} % |" for b in BUCKETS if b in tot]
    (out / "stats.md").write_text("\n".join(lines) + "\n"); json.dump(table, open(out / "stats.json", "w"), indent=1)
    print("\n".join(lines))
    if not a.write:
        return
    mult = {k: float(v) for k, v in (x.split("=") for x in a.mult.split(","))}
    parts, eff = [], []
    for p, (w, bk) in buckets_of.items():
        Z = sum(len(bk[b]) * mult[b] for b in bk)
        for b in BUCKETS:
            if b not in bk or mult[b] == 0:
                continue
            mp = out / f"{src_name(p)}-{b}.json"
            with open(mp, "w") as f:
                for r, _, _ in bk[b]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            wb = w * len(bk[b]) * mult[b] / Z
            parts.append(f"{mp}:{wb:.6f}"); eff.append((src_name(p), b, round(100 * wb / w, 1)))
    (out / "mix.txt").write_text(",".join(parts) + "\n")
    print("\n== draw share inside each source after --mult (%):", eff)
    print(f"== wrote {len(parts)} bucket manifests and {out / 'mix.txt'}")


if __name__ == "__main__":
    main()
