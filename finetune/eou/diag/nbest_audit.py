#!/usr/bin/env python3
"""Ranking / search audit of an RNNT N-best dump (lm_fusion_probe.py --dump-nbest), offline, standard library only.

    python3 nbest_audit.py --dir /root/ft/analysis/K/e3s8k [--report mls,fleurs,cv,vp] [--tune mls_tune,fleurs_tune]
        [--out audit.json]

For every nbest_b<K>.jsonl in --dir:
1. beam WER (rank 1), oracle@k for k = 1..K (the best of the top-k by beam rank) per report set and macro:
   how much better material the candidate list holds, and how deep it sits;
2. winner vs oracle (clips where a lower-ranked hypothesis has fewer word errors): substitutions / deletions /
   insertions of each, word-length difference, score margin, oracle rank -> what the ranking gets wrong;
3. rescoring grid: score' = score + lam * tokens + mu * words, and score / tokens ** alpha (length
   normalisation), CHOSEN on the --tune sets only (MLS/FLEURS val clips outside the harness subsample) and
   REPORTED on the --report sets (the harness clips, never used for the choice). The same grid chosen on the
   report sets themselves is printed as an optimistic in-sample bound, labelled as such.
WER is pooled per set (errors / reference words), macro = mean over --report sets, like two_stage.py.
"""
import argparse
import glob
import json
import os
import re


def align(r, h):
    """-> (substitutions, deletions, insertions) of hypothesis h against reference r (word lists)."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    s = de = ins = 0
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            s += r[i - 1] != h[j - 1]; i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            de += 1; i -= 1
        else:
            ins += 1; j -= 1
    return s, de, ins


def load(path):
    clips = []
    for l in open(path):
        c = json.loads(l)
        rw = c["ref"].split()
        for h in c["hyps"]:
            h["words"] = len(h["text"].split())
            h["sdi"] = align(rw, h["text"].split())
            h["err"] = sum(h["sdi"])
        c["nref"] = len(rw)
        clips.append(c)
    return clips


def wer(clips, pick):
    """pooled WER (%) per set of the hypothesis chosen by pick(clip) -> hyp."""
    e, n = {}, {}
    for c in clips:
        e[c["set"]] = e.get(c["set"], 0) + pick(c)["err"]
        n[c["set"]] = n.get(c["set"], 0) + c["nref"]
    return {s: 100 * e[s] / max(1, n[s]) for s in e}


def macro(per, sets):
    return sum(per[s] for s in sets) / len(sets)


def rescorer(kind, p):
    if kind == "linear":
        lam, mu = p
        return lambda c: max(c["hyps"], key=lambda h: h["score"] + lam * h["ntok"] + mu * h["words"])
    return lambda c: max(c["hyps"], key=lambda h: h["score"] / max(1, h["ntok"]) ** p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--report", default="mls,fleurs,cv,vp")
    ap.add_argument("--tune", default="mls_tune,fleurs_tune")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rep, tun = a.report.split(","), a.tune.split(",")
    grid = [("linear", (lam, mu)) for lam in (-1, -0.5, 0, 0.5, 1, 1.5, 2, 3) for mu in (-1, 0, 1, 2, 3, 4)] + \
           [("norm", al) for al in (0.25, 0.5, 0.75, 1.0)]
    res = {}
    files = sorted(glob.glob(os.path.join(a.dir, "nbest_b*.jsonl")), key=lambda p: int(re.search(r"b(\d+)", p).group(1)))
    for path in files:
        K = int(re.search(r"b(\d+)", path).group(1))
        clips = load(path)
        R = [c for c in clips if c["set"] in rep]
        T = [c for c in clips if c["set"] in tun]
        out = {"clips_report": len(R), "clips_tune": len(T)}
        beam = wer(R, lambda c: c["hyps"][0])
        out["beam"] = {**{s: round(beam[s], 2) for s in rep}, "macro": round(macro(beam, rep), 2)}
        out["oracle_at_k"] = {}
        for k in sorted({1, 2, 4, 8, 16, 32, K}):
            if k > K:
                continue
            o = wer(R, lambda c, k=k: min(c["hyps"][:k], key=lambda h: h["err"]))
            out["oracle_at_k"][k] = {**{s: round(o[s], 2) for s in rep}, "macro": round(macro(o, rep), 2)}
        # winner vs oracle on the report clips where the oracle beats rank 1
        diff = [c for c in R if min(h["err"] for h in c["hyps"]) < c["hyps"][0]["err"]]
        agg = {"clips": len(diff), "of": len(R)}
        for s in rep + ["all"]:
            cs = [c for c in diff if s == "all" or c["set"] == s]
            if not cs:
                continue
            w = [c["hyps"][0] for c in cs]
            o = [min(c["hyps"], key=lambda h: (h["err"], c["hyps"].index(h))) for c in cs]
            agg[s] = {
                "n": len(cs),
                "winner_SDI": [sum(h["sdi"][i] for h in w) for i in range(3)],
                "oracle_SDI": [sum(h["sdi"][i] for h in o) for i in range(3)],
                "words_winner_minus_oracle": round(sum(x["words"] - y["words"] for x, y in zip(w, o)) / len(cs), 3),
                "score_margin_mean": round(sum(x["score"] - y["score"] for x, y in zip(w, o)) / len(cs), 3),
                "oracle_rank_mean": round(sum(c["hyps"].index(y) + 1 for c, y in zip(cs, o)) / len(cs), 2),
            }
        out["winner_vs_oracle"] = agg
        # rescoring: choose on tune, report on harness
        if T:
            tw = {g: macro(wer(T, rescorer(*g)), [s for s in tun if any(c["set"] == s for c in T)]) for g in grid}
            best = min(tw, key=tw.get)
            rw = wer(R, rescorer(*best))
            out["rescore_tuned"] = {"chosen_on_tune": [best[0], best[1]], "tune_macro": round(tw[best], 2),
                                    "tune_macro_rank1": round(macro(wer(T, lambda c: c["hyps"][0]), [s for s in tun if any(c["set"] == s for c in T)]), 2),
                                    **{s: round(rw[s], 2) for s in rep}, "macro": round(macro(rw, rep), 2)}
        ins = {g: macro(wer(R, rescorer(*g)), rep) for g in grid}
        bi = min(ins, key=ins.get)
        out["rescore_in_sample_OPTIMISTIC"] = {"grid": [bi[0], bi[1]], "macro": round(ins[bi], 2)}
        res[K] = out
        print(f"== beam {K}: report {len(R)} clips, tune {len(T)}")
        print("   beam   ", out["beam"])
        for k, v in out["oracle_at_k"].items():
            print(f"   oracle@{k:<2}", v)
        print("   winner vs oracle:", json.dumps(agg.get("all")), "clips", agg["clips"], "/", agg["of"])
        for s in rep:
            if s in agg:
                print(f"     {s}:", json.dumps(agg[s]))
        if "rescore_tuned" in out:
            print("   rescore (chosen on tune, reported on harness):", out["rescore_tuned"])
        print("   rescore in-sample (OPTIMISTIC bound):", out["rescore_in_sample_OPTIMISTIC"])
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
