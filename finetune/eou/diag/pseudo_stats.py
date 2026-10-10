#!/usr/bin/env python3
"""Distributions of a teacher-labelled (pseudo-label) manifest, BEFORE any filter is chosen.

    python3 pseudo_stats.py --manifest clips_00000000.teacher.json [--subs subs_00000000.json] [--json out.json]

Per clip (teacher_ws.py output): detected language (teacher_lang), empty teacher text,
words per second, characters per word, longest repeated n-gram run (degenerate loops),
duration; printed as counts and percentiles, with the hours each candidate gate would keep.
With a reference text in the manifest (e.g. Granary's Whisper label) the teacher_wer
distribution is the two-teacher AGREEMENT, with the hours each threshold keeps.
With --subs (YODAS2 video-level subtitles): per video, the WER between the concatenated
teacher text (clips in time order) and the subtitle text -- a video-level check only
(music / other language / auto-captions), never a clip filter.
"""
import argparse
import collections
import json
import re
import unicodedata


def norm(t):
    t = "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def lev(x, y):
    p = list(range(len(y) + 1))
    for i, u in enumerate(x, 1):
        c = [i]
        for j, v in enumerate(y, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (u != v)))
        p = c
    return p[-1]


def max_repeat(words, nmax=4):
    best = 1
    for n in range(1, nmax + 1):
        for i in range(0, max(0, len(words) - n)):
            k = 1
            while words[i + k * n: i + (k + 1) * n] == words[i: i + n] and i + (k + 1) * n <= len(words):
                k += 1
            best = max(best, k)
    return best


def pct(v, qs=(0.1, 0.5, 0.9)):
    v = sorted(v)
    return [round(v[min(len(v) - 1, int(q * len(v)))], 2) for q in qs] if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--subs", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--lang", default="it")
    a = ap.parse_args()
    rs = [json.loads(l) for l in open(a.manifest)]
    H = lambda xs: round(sum(r["duration"] for r in xs) / 3600, 2)  # noqa: E731
    for r in rs:
        w = r.get("teacher", "").split()
        r["_wps"] = len(w) / max(0.1, r["duration"])
        r["_cpw"] = sum(map(len, w)) / max(1, len(w))
        r["_rep"] = max_repeat(w) if w else 0
    out = {"n": len(rs), "hours": H(rs)}
    langs = collections.Counter(r.get("teacher_lang", "?") for r in rs)
    out["lang"] = {k: {"clips": v, "hours": H([r for r in rs if r.get("teacher_lang", "?") == k])} for k, v in langs.most_common(8)}
    nonempty = [r for r in rs if r.get("teacher")]
    out["empty"] = {"clips": len(rs) - len(nonempty), "hours": round(out["hours"] - H(nonempty), 2)}
    out["dur_s"] = pct([r["duration"] for r in rs])
    out["words_per_s"] = pct([r["_wps"] for r in nonempty])
    out["chars_per_word"] = pct([r["_cpw"] for r in nonempty])
    out["repeat_run_ge3"] = sum(r["_rep"] >= 3 for r in nonempty)
    lang0 = (args_lang := a.lang)
    it = [r for r in nonempty if (r.get("teacher_lang") or "").split("-")[0] == lang0]   # the server says "it" or "it-IT"
    gate = [r for r in it if 1.0 <= r["_wps"] <= 4.5 and r["_rep"] < 3]
    agree = [r for r in rs if r.get("teacher_wer") is not None]
    if agree:   # a reference text exists (e.g. Granary's Whisper label): teacher-vs-reference agreement
        out["agreement_wer"] = {"p10_p50_p90": pct([r["teacher_wer"] for r in agree]),
                                "hours_at_or_below": {t: H([r for r in agree if r["teacher_wer"] <= t]) for t in (5, 10, 15, 20, 30, 50)}}
        d = sorted(r["duration"] for r in agree); q1, q2 = d[len(d) // 3], d[2 * len(d) // 3]
        out["agreement_by_duration"] = {f"{lo:.1f}-{hi:.1f}s": {"clips": len(x), "p50": (pct([r["teacher_wer"] for r in x]) or [None] * 3)[1],
                                                               "share_le_15": round(sum(r["teacher_wer"] <= 15 for r in x) / max(1, len(x)), 3)}
                                        for lo, hi in ((0, q1), (q1, q2), (q2, 1e9))
                                        for x in [[r for r in agree if lo <= r["duration"] < hi]]}
    out["candidate_gate"] = {"rule": f"lang {lang0}, non-empty, 1.0 <= words/s <= 4.5, no 3x repeated n-gram", "clips": len(gate), "hours": H(gate)}
    if a.subs:
        subs = {}
        for v in json.load(open(a.subs)):
            subs[v["audio_id"]] = norm(" ".join(v["text"].values()))
        byv = collections.defaultdict(list)
        for r in rs:
            byv[r.get("video")].append(r)
        vw = []
        for vid, xs in byv.items():
            t = " ".join(r.get("teacher", "") for r in sorted(xs, key=lambda r: r.get("start", 0))).split()
            s = subs.get(vid, "").split()
            if s:
                vw.append((vid, round(100 * lev(s, t) / max(1, len(s)), 1), H(xs), collections.Counter(r.get("teacher_lang") for r in xs).most_common(1)[0][0]))
        vw.sort(key=lambda x: x[1])
        out["video_teacher_vs_subs_wer"] = {"videos": len(vw), "p10_p50_p90": pct([x[1] for x in vw]),
                                            "worst": vw[-8:], "best": vw[:4]}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    ex = sorted(nonempty, key=lambda r: r["_rep"], reverse=True)[:3] + it[:4]
    for r in ex:
        print(f"   [{r.get('teacher_lang')}] {r['duration']:5.1f}s wps {r['_wps']:.1f} rep {r['_rep']} | {r.get('teacher_raw', r.get('teacher', ''))[:110]}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
