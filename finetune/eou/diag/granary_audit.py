#!/usr/bin/env python3
"""Model-free audit of a teacher-scored Granary pool BEFORE choosing a training subset (no student model involved).

    granary_audit.py --pool mix/fr-yg/train_yg.teacher.json --lang fr --leak "a.json,b.json,..." [--train "c.json,..."] \
        [--thresholds 5,10,15,20,30] [--max-s 30] [--json out.json]

agree = teacher_wer (teacher vs the pool reference, teacher_ws.py, normalised with --lang). Per threshold T (agree <= T,
inclusive) and for the whole pool: clips, hours, videos, share of the pool's videos kept, clips per video p50 / p90,
duration p10 / p50 / p90, hours per Granary dir. Exclusions counted, not applied to the pool file:
  digit   teacher text with a digit (the a-z targets cannot spell it; the IT pipeline drops these)
  empty   empty teacher text;  noref  no reference (agree None)
  dup     normalised teacher text already seen in the pool (first occurrence kept)
  leak    a word 8-gram (or the whole text when < 8 words) shared with any reference of --leak (dev / FINAL banks)
  train   the same rule against --train (the plain mix the subset is ADDED to: overlap, informational)
"""
import argparse, collections, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import textnorm  # noqa: E402


def grams(ws, n=8):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)} if len(ws) >= n else {" ".join(ws)}


def index(paths, nf):
    g = set()
    for p in filter(None, paths.split(",")):
        for l in open(p):
            if l.strip():
                g |= grams(nf(json.loads(l).get("text", "")).split())
    return g


def pc(v, qs=(10, 50, 90)):
    v = sorted(v)
    return [round(v[min(len(v) - 1, int(q / 100 * len(v)))], 1) for q in qs] if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--leak", default="")
    ap.add_argument("--train", default="")
    ap.add_argument("--thresholds", default="5,10,15,20,30")
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    nf = lambda t: textnorm.norm(t or "", a.lang)  # noqa: E731
    leak, train = index(a.leak, nf), index(a.train, nf)
    rs = [json.loads(l) for l in open(a.pool) if l.strip()]
    vid = lambda r: str(r.get("speaker") or r.get("original_audio_id") or r["audio_filepath"])  # noqa: E731
    gdir = lambda r: (re.search(r"/(\w\w\d{3})[_/]", r["audio_filepath"]) or re.search(r"(\w\w\d{3})", r["audio_filepath"])).group(1)  # noqa: E731
    allv = {vid(r) for r in rs}
    drop, seen, ok = collections.Counter(), set(), []
    for r in rs:
        t = (r.get("teacher") or "").strip()
        nt = nf(t)
        if not nt:
            drop["empty"] += 1; continue
        if re.search(r"\d", t):
            drop["digit"] += 1; continue
        if r.get("teacher_wer") is None:
            drop["noref"] += 1; continue
        if r["duration"] > a.max_s:
            drop["duration"] += 1; continue
        if nt in seen:
            drop["dup"] += 1; continue
        seen.add(nt)
        g = grams(nt.split())
        if g & leak:
            drop["leak"] += 1; continue
        r["_train_overlap"] = bool(g & train)
        ok.append(r)
    out = {"pool_clips": len(rs), "pool_hours": round(sum(r["duration"] for r in rs) / 3600, 2), "pool_videos": len(allv),
           "dropped": dict(drop), "agree_p10_p25_p50_p75_p90": pc([r["teacher_wer"] for r in ok], (10, 25, 50, 75, 90)),
           "teacher_lang": dict(collections.Counter(r.get("teacher_lang", "?") for r in ok).most_common(5)), "subsets": {}}
    print(f"== pool {out['pool_clips']} clips {out['pool_hours']} h {out['pool_videos']} videos; dropped {dict(drop)}; "
          f"agree p10/p25/p50/p75/p90 {out['agree_p10_p25_p50_p75_p90']}; teacher_lang {out['teacher_lang']}")
    base_h, base_v = collections.defaultdict(float), collections.defaultdict(set)
    for r in ok:
        base_h[gdir(r)] += r["duration"] / 3600; base_v[gdir(r)].add(vid(r))
    print("| agree <= | clips | h | videos | videos kept % | h / video mean | clips/video p50/p90 | dur p10/p50/p90 s | train-overlap clips | per dir: h (% of dir h kept) / videos |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for T in [float(x) for x in a.thresholds.split(",")] + [float("inf")]:
        s = [r for r in ok if r["teacher_wer"] <= T]
        if not s:
            continue
        per = collections.Counter(vid(r) for r in s)
        hd, vd = collections.defaultdict(float), collections.defaultdict(set)
        for r in s:
            hd[gdir(r)] += r["duration"] / 3600; vd[gdir(r)].add(vid(r))
        row = {"clips": len(s), "hours": round(sum(r["duration"] for r in s) / 3600, 2), "videos": len(per),
               "videos_kept_pct": round(100 * len(per) / len(allv), 1), "clips_per_video_p50_p90": pc(list(per.values()), (50, 90)),
               "dur_p10_p50_p90": pc([r["duration"] for r in s]), "train_overlap": sum(r["_train_overlap"] for r in s),
               "h_per_video_mean": round(sum(r["duration"] for r in s) / 3600 / len(per), 3),
               "per_dir": {k: {"h": round(hd[k], 1), "h_kept_pct": round(100 * hd[k] / base_h[k], 1), "videos": len(vd[k]),
                               "videos_kept_pct": round(100 * len(vd[k]) / len(base_v[k]), 1)} for k in sorted(hd)}}
        dirs = "; ".join(f"{k} {v['h']} ({v['h_kept_pct']}%) / {v['videos']}" for k, v in row["per_dir"].items())
        out["subsets"]["all" if T == float("inf") else str(int(T))] = row
        print(f"| {'all' if T == float('inf') else int(T)} | {row['clips']} | {row['hours']} | {row['videos']} | {row['videos_kept_pct']} | "
              f"{row['h_per_video_mean']} | {row['clips_per_video_p50_p90']} | {row['dur_p10_p50_p90']} | {row['train_overlap']} | "
              f"{dirs} |", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
