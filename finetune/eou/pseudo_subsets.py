#!/usr/bin/env python3
"""Training subsets from a teacher-scored pseudo-label pool (teacher_ws.py output), BEFORE training.

    python3 pseudo_subsets.py --pool train_yg.teacher.json --out DIR [--thresholds 10,15,30]
        [--max-s 30] [--a2 "mls.json:0.35,cv.json:0.30,vp.json:0.20,fleurs.json:0.15"] [--bs 16]
        [--verbalize 1 [--drop-flags ...]] [--no-ref-keep heuristics] [--leak-refs dev.json,test.json,...]

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

Options (all off by default: the output is then unchanged):
  --verbalize 1     teacher_raw -> verbalize_it.verbalize_ex instead of the digit drop: clips
                    whose verbalisation carries a guess flag in --drop-flags (verbalize_it's
                    default set) are dropped and counted; the other symbol/digit clips keep the
                    verbalised label, and teacher_wer is recomputed against the verbalised
                    reference (like verbalize_it.py --manifest).
  --no-ref-keep heuristics
                    clips without a reference (teacher_wer None: --yg-split-long pieces) are
                    dropped today; with this they are kept, in EVERY subset (the agreement
                    threshold cannot apply), iff they pass the hallucination heuristics:
                    rep  a word repeated > 4 times in a row, or a 2-6 word n-gram repeated
                         >= 4 times in a row;  long  a word > 22 letters;
                    wps  words per second outside 1..4.5.
                    The flags are computed for ALL clips, written as `h_flags` and counted;
                    clips with a reference are only flagged (their agreement score rules).
  --leak-refs a.json,b.json
                    drop every clip whose normalised teacher text, reference or ref_parent
                    shares a word 8-gram with any reference text of those manifests (dev/test
                    of every eval set); references shorter than 8 words match only as the exact
                    whole text. Count reported.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verbalize_it import _HAS_SYM, lev, norm, verbalize, verbalize_ex  # noqa: E402

VB_DROP = "phone,longdigits,dotnum,slash,date2y,residual"   # verbalize_it.py --drop-flags default


def pct(v, qs=(0.1, 0.5, 0.9)):
    v = sorted(v)
    return [round(v[min(len(v) - 1, int(q * len(v)))], 1) for q in qs] + [round(v[-1], 1)] if v else None


def mean_dur(path, max_s):
    d = [json.loads(l)["duration"] for l in open(path)]
    d = [x for x in d if x <= max_s]
    return sum(d) / max(1, len(d))


def h_flags(r):
    """Hallucination heuristics on the teacher label (H1-H3-like) + speaking rate."""
    w = r.get("teacher", "").split()
    fl = []
    if any(len(set(w[i:i + 5])) == 1 for i in range(len(w) - 4)) or any(
            w[i:i + n] * 4 == w[i:i + 4 * n] for n in range(2, 7) for i in range(len(w) - 4 * n + 1)):
        fl.append("rep")
    if any(len(x) > 22 for x in w):
        fl.append("long")
    if not 1.0 <= len(w) / max(1e-3, r["duration"]) <= 4.5:
        fl.append("wps")
    return fl


def grams(t, n=8):
    w = t.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def leak_index(paths):
    """-> (8-gram set, set of whole short references) of every reference of the manifests."""
    g, short = set(), set()
    for p in paths:
        txt = open(p, encoding="utf-8").read().strip()
        try:
            rs = json.loads(txt)
            rs = rs if isinstance(rs, list) else [rs]
        except ValueError:
            rs = [json.loads(l) for l in txt.splitlines() if l.strip()]
        for r in rs:
            for t in {norm(r.get("text") or ""), norm(verbalize(r.get("text") or ""))}:
                if len(t.split()) >= 8:
                    g |= grams(t)
                elif t:
                    short.add(t)
    return g, short


def leaks(r, idx):
    for t in (r.get("teacher") or "", norm(r.get("text") or ""), norm(r.get("ref_parent") or "")):
        if t and (t in idx[1] or not grams(t).isdisjoint(idx[0])):
            return True
    return False


def vb_pool(rs, drop):
    """verbalize_it.rewrite_manifest on rows in memory: -> (kept rows, n dropped, {flag: n})."""
    out, nd, fc = [], 0, {}
    for r in rs:
        raw = r.get("teacher_raw") or ""
        if raw and _HAS_SYM.search(raw):
            vb, fl = verbalize_ex(raw)
            for f in fl:
                fc[f] = fc.get(f, 0) + 1
            if drop & set(fl):
                nd += 1
                continue
            r["teacher_pre_vb"], r["teacher_wer_pre_vb"] = r.get("teacher"), r.get("teacher_wer")
            r["teacher_raw_vb"], r["teacher_vb_flags"], r["teacher"] = vb, fl, norm(vb)
            ref = norm(verbalize(r.get("text") or ""))
            if ref:
                r["teacher_wer"] = round(100 * lev(ref.split(), r["teacher"].split()) / max(1, len(ref.split())), 1)
        out.append(r)
    return out, nd, fc


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
    ap.add_argument("--verbalize", type=int, default=0, choices=(0, 1), help="1: verbalise teacher digits instead of dropping")
    ap.add_argument("--drop-flags", default=VB_DROP, help="--verbalize 1: drop clips carrying any of these guess flags")
    ap.add_argument("--no-ref-keep", default="none", choices=("none", "heuristics"),
                    help="clips without a reference: drop (none) or keep those passing the heuristics")
    ap.add_argument("--leak-refs", default="", help="eval manifests (comma list): drop clips sharing an 8-gram with them")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rs = [json.loads(l) for l in open(a.pool)]
    random.Random(a.seed).shuffle(rs)   # one fixed order for every subset
    n0, h0 = len(rs), sum(r['duration'] for r in rs) / 3600
    if a.verbalize:
        rs, nd, fc = vb_pool(rs, {f for f in a.drop_flags.split(",") if f})
        print(f"== verbalize: {nd} clips dropped by guess flags {a.drop_flags or '-'}; flags {json.dumps(fc)}")
    raw = ((lambda r: r.get("teacher_raw_vb", r.get("teacher_raw", ""))) if a.verbalize  # noqa: E731
           else (lambda r: r.get("teacher_raw", "")))
    heur = a.no_ref_keep == "heuristics"
    if heur:
        from collections import Counter
        for r in rs:
            r["h_flags"] = h_flags(r)
        hc = Counter(f for r in rs if r.get("teacher") for f in r["h_flags"])
        noref = [r for r in rs if r.get("teacher") and r.get("teacher_wer") is None]
        print(f"== heuristics on {sum(1 for r in rs if r.get('teacher'))} non-empty clips: {dict(hc)}; clips without a "
              f"reference {len(noref)}, of which failing {sum(1 for r in noref if r['h_flags'])}")
    digits = [r for r in rs if re.search(r"\d", raw(r))]
    keep = [r for r in rs if r.get("teacher") and not re.search(r"\d", raw(r)) and r["duration"] <= a.max_s
            and (r.get("teacher_wer") is not None or (heur and not r["h_flags"]))]
    print(f"== pool {n0} clips {h0:.1f} h; dropped: teacher digits {len(digits)}, "
          f"empty {sum(1 for r in rs if not r.get('teacher'))}, > {a.max_s:g} s {sum(1 for r in rs if r['duration'] > a.max_s)}")
    if a.leak_refs:
        idx = leak_index([p for p in a.leak_refs.split(",") if p])
        n = len(keep)
        keep = [r for r in keep if not leaks(r, idx)]
        print(f"== leak guard: {len(idx[0])} ref 8-grams + {len(idx[1])} short refs; dropped {n - len(keep)} clips")

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
        subsets[f"yg_agree<={t:g}"] = write(f"yg_agree<={t:g}", [r for r in keep if r["teacher_wer"] is None or r["teacher_wer"] <= t])
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
        hv = Counter()
        for r in xs:
            hv[r.get("speaker")] += r["duration"]
        hv = sorted(hv.values(), reverse=True)
        top = sum(hv[: max(1, len(hv) // 10)]) / max(1e-9, sum(hv)) if hv else float("nan")
        print(f"| {name} | {len(xs)} | {sum(d) / 3600:.1f} | {len(cv)} | {lost} | {100 * top:.0f} % | {pct(d)} | {mix} |")


if __name__ == "__main__":
    main()
