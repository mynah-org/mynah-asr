#!/usr/bin/env python3
"""Phase D: selected-new (D1) vs random-new (D0) clips from ONE new teacher-labelled Granary pool (it-yg7).

    stats:  python3 d_select.py --pool yg_all.json --student student_yg7.jsonl --out DIR
    write:  python3 d_select.py --pool yg_all.json --student student_yg7.jsonl --out DIR --hours H \
                --base "mls.json:0.1964,cv.json:0.1684,vp.json:0.1122,fleurs.json:0.0842" \
                --ext "yg_agree<=15.json,yg6_agree<=15.json" --ext-w 0.4388

--pool = pseudo_subsets.py's yg_all.json (label `text` = teacher text, `ref_granary` = Granary-Whisper text,
`teacher_wer` = teacher vs Granary-Whisper, the two-teacher agreement); --student = student_score.py on that pool
(student vs label). Buckets (first match; the thresholds are curriculum_mix.py's, not tuned here):
  E  excluded     label hallucination heuristics (pseudo_subsets.h_flags on the label: rep / long / wps) or an
                  empty student hypothesis (whole-clip deletion: boundary / music / non-speech suspects);
  D  suspicious   teacher_wer >= 30 and the student agrees with Granary-Whisper (WER <= 10): label suspect;
  C  ambiguous    teacher_wer > 15 (outside the agree<=15 cut of the C250 pools);
  B  solved       student_wer <= 5;
  A  learnable    teacher_wer <= 5 and student_wer >= 30 (teacher credible, student fails);
  M  mid          everything else.
Credibility (teacher_wer, h_flags) never looks at the student. With --hours H:
  D1 = H hours drawn at random (--seed) from A;
  D0 = H hours drawn at random from the GATE = teacher_wer <= 5 and no h_flags (A's student-free conditions),
       independent of the student, matched to D1 per duration bin (hours per bin = D1's), so D0 and D1 differ
       only in the student-error selection. D0 may contain D1 clips; the overlap is reported.
Each arm's mix = --base (A2, unchanged) + the --ext manifests + the D set, the external weight --ext-w split in
proportion to hours (the C250 rule; the D set counts H hours in both arms, so the weights are identical), written as DIR/mix_D0.txt and DIR/mix_D1.txt (same weights by construction).
"""
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pseudo_subsets import h_flags   # noqa: E402

BINS = [1, 2, 4, 8, 15, 30.01]
ORDER = ["A", "B", "M", "C", "D", "E"]
NAMES = {"A": "learnable", "B": "solved", "M": "mid", "C": "ambiguous", "D": "suspicious", "E": "excluded"}


def lev(x, y):
    prev = list(range(len(y) + 1))
    for i, p in enumerate(x, 1):
        cur = [i]
        for j, q in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != q)))
        prev = cur
    return prev[-1]


def dbin(d):
    for i in range(len(BINS) - 1):
        if d < BINS[i + 1]:
            return i
    return len(BINS) - 2


def hours(path, max_s):
    return sum(r["duration"] for r in map(json.loads, open(path)) if r["duration"] <= max_s) / 3600


def row_stats(xs):
    d = np.array([r["duration"] for r in xs])
    rw = sum(r["_s"]["ref_words"] for r in xs)
    hv = Counter()
    for r in xs:
        hv[r.get("speaker")] += r["duration"]
    top = sorted(hv.values(), reverse=True)
    k = max(1, len(top) // 10)
    return {"clips": len(xs), "hours": round(d.sum() / 3600, 2),
            "dur_p10_p50_p90": [round(float(np.percentile(d, q)), 1) for q in (10, 50, 90)],
            "student_wer": round(float(np.mean([r["_s"]["student_wer"] for r in xs])), 1),
            "teacher_wer": round(float(np.mean([r["teacher_wer"] for r in xs])), 1),
            "empty_pct": round(100 * sum(1 for r in xs if not r["_s"]["student"].strip()) / len(xs), 1),
            "hyp_ref": round(sum(len(r["_s"]["student"].split()) for r in xs) / max(1, rw), 3),
            "videos": len(hv), "top10pct_videos_h_share": round(100 * sum(top[:k]) / max(1e-9, sum(top)), 1)}


def draw(xs, target_s, rnd):
    xs = xs[:]; rnd.shuffle(xs); out, acc = [], 0.0
    for r in xs:
        if acc >= target_s:
            break
        out.append(r); acc += r["duration"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=0.0, help="H: write D0 / D1 of H hours each (0 = stats only)")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--base", default="", help="A2 part of the mix, unchanged (path:weight,...)")
    ap.add_argument("--ext", default="", help="existing external manifests sharing --ext-w with the D set")
    ap.add_argument("--ext-w", type=float, default=0.4388)
    a = ap.parse_args()
    st = {}
    for l in open(a.student):
        r = json.loads(l); st[r["audio_filepath"]] = r
    rows = [r for r in map(json.loads, open(a.pool)) if r["duration"] <= a.max_s]
    miss = sum(1 for r in rows if r["audio_filepath"] not in st)
    if miss:
        sys.exit(f"== {miss} pool clips without a student score")
    bk = defaultdict(list)
    for r in rows:
        s = st[r["audio_filepath"]]; r["_s"] = s; tw = r["teacher_wer"]
        r["_flags"] = h_flags({"teacher": r["text"], "duration": r["duration"]})
        sg = r.get("ref_granary") or ""
        swg = 100 * lev(sg.split(), s["student"].split()) / max(1, len(sg.split())) if sg else None
        if r["_flags"] or not s["student"].strip():
            b = "E"
        elif tw >= 30 and swg is not None and swg <= 10:
            b = "D"
        elif tw > 15:
            b = "C"
        elif s["student_wer"] <= 5:
            b = "B"
        elif tw <= 5 and s["student_wer"] >= 30:
            b = "A"
        else:
            b = "M"
        bk[b].append(r)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tab = {b: row_stats(bk[b]) for b in ORDER if bk[b]}
    tot = row_stats(rows)
    lines = ["| bucket | clips | hours | dur p10/p50/p90 s | student WER | teacher/label WER | empty % | hyp/ref words "
             "| videos | top-10 % videos' share of h |", "|" + "---|" * 10]
    for b, t in list(tab.items()) + [("all", tot)]:
        lines.append(f"| {b} {NAMES.get(b, '')} | {t['clips']} | {t['hours']} | {'/'.join(map(str, t['dur_p10_p50_p90']))} "
                     f"| {t['student_wer']} | {t['teacher_wer']} | {t['empty_pct']} | {t['hyp_ref']} | {t['videos']} "
                     f"| {t['top10pct_videos_h_share']} |")
    gate = [r for r in rows if r["teacher_wer"] <= 5 and not r["_flags"]]
    lines += ["", "| duration bin s | A clips | A hours | gate clips | gate hours |", "|---|---|---|---|---|"]
    for i in range(len(BINS) - 1):
        ab = [r for r in bk["A"] if dbin(r["duration"]) == i]; gb = [r for r in gate if dbin(r["duration"]) == i]
        lines.append(f"| {BINS[i]}-{BINS[i + 1]:.0f} | {len(ab)} | {sum(r['duration'] for r in ab) / 3600:.2f} "
                     f"| {len(gb)} | {sum(r['duration'] for r in gb) / 3600:.2f} |")
    fc = Counter(f for r in rows for f in r["_flags"])
    lines += ["", f"label flags (E): {dict(fc)}; student-empty clips: {sum(1 for r in rows if not r['_s']['student'].strip())}"]
    (out / "d_stats.md").write_text("\n".join(lines) + "\n")
    json.dump({"buckets": tab, "all": tot}, open(out / "d_stats.json", "w"), indent=1)
    print("\n".join(lines))
    if a.hours <= 0:
        return
    rnd = random.Random(a.seed)
    A = bk["A"]; ah = sum(r["duration"] for r in A) / 3600
    if a.hours > ah:
        sys.exit(f"== H {a.hours} h > bucket A {ah:.2f} h")
    d1 = draw(A, a.hours * 3600, rnd)
    d0 = []
    for i in range(len(BINS) - 1):   # D1's hours per duration bin, drawn from the gate without looking at the student
        t = sum(r["duration"] for r in d1 if dbin(r["duration"]) == i)
        d0 += draw([r for r in gate if dbin(r["duration"]) == i], t, rnd) if t > 0 else []
    ov = len({r["audio_filepath"] for r in d0} & {r["audio_filepath"] for r in d1})
    sets = {"D0": d0, "D1": d1}
    for n, xs in sets.items():
        with open(out / f"{n}.json", "w") as f:
            for r in xs:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    summ = {n: row_stats(xs) for n, xs in sets.items()}
    summ["overlap_clips"] = ov; summ["seed"] = a.seed; summ["H"] = a.hours
    ext = [p for p in a.ext.split(",") if p]
    eh = [hours(p, a.max_s) for p in ext]
    for n in sets:
        hs = eh + [a.hours]; ps = ext + [str(out / f"{n}.json")]
        parts = ([a.base] if a.base else []) + [f"{p}:{a.ext_w * h / sum(hs):.4f}" for p, h in zip(ps, hs)]
        (out / f"mix_{n}.txt").write_text(",".join(parts) + "\n")
        summ[n]["mix"] = ",".join(parts)
    json.dump(summ, open(out / "d_sets.json", "w"), indent=1)
    for n in sets:
        t = summ[n]
        print(f"== {n}: {t['clips']} clips {t['hours']} h, dur {t['dur_p10_p50_p90']}, student {t['student_wer']}, "
              f"teacher {t['teacher_wer']}, videos {t['videos']}\n   mix {t['mix']}")
    print(f"== overlap D0/D1: {ov} clips ({100 * ov / max(1, len(d1)):.1f} % of D1)")


if __name__ == "__main__":
    main()
