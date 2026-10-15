#!/usr/bin/env python3
"""Where the greedy -> beam -> oracle@K word errors are, per set and clip-duration bucket (CPU, standard library).

    python3 nbest_gap.py --greedy greedy.jsonl [--penalty 0] --nbest nbest_b8.jsonl [--out gap.json]

Inputs: greedy_blank_sweep.py --dump (the greedy hypothesis at --penalty) and lm_fusion_probe.py --dump-nbest
(rank 1 = beam winner, oracle = fewest word errors in the list), joined on the audio path. Same normalised text.
Per system (greedy, beam, oracle) and set: WER and its split into
  D, I            deletions / insertions per 100 reference words;
  S_near          substitutions within 2 characters, or sharing a >= 4-letter prefix (ending, accent-free spelling,
                  apostrophe/elision: the morphology / orthography class);
  S_far           every other substitution (a different word);
  seg             a reference word split into two hypothesis words or two merged (joined pair within 1 char),
                  counted once per event (each event is also 2 errors among S/D/I above).
Clip classes on the K-best list:
  same            beam winner already has the oracle's error count;
  ranking         a lower-ranked hypothesis has fewer errors (the list holds better text, the score picks worse);
  no_good_hyp     even the oracle has clip WER >= 30 % (nothing in the K-best is close: acoustics/representation).
Greedy -> oracle error mass is split into greedy->beam (search) and beam->oracle (ranking), per set and bucket.
"""
import argparse
import json
from collections import defaultdict

BUCKETS = ((0, 5, "<5s"), (5, 10, "5-10s"), (10, 20, "10-20s"), (20, 1e9, ">=20s"))


def char_lev(x, y):
    prev = list(range(len(y) + 1))
    for i, p in enumerate(x, 1):
        cur = [i]
        for j, q in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != q)))
        prev = cur
    return prev[-1]


def ops(r, h):
    """Word alignment -> list of (op, ref_word, hyp_word), op in = S D I."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    out = []; i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            out.append(("=" if r[i - 1] == h[j - 1] else "S", r[i - 1], h[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append(("D", r[i - 1], None)); i -= 1
        else:
            out.append(("I", None, h[j - 1])); j -= 1
    return out[::-1]


def near(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return char_lev(a, b) <= 2 or k >= 4


def classify(ref, hyp):
    r, h = ref.split(), hyp.split()
    al = ops(r, h)
    c = defaultdict(int)
    for op, rw, hw in al:
        if op == "S":
            c["S_near" if near(rw, hw) else "S_far"] += 1
        elif op in "DI":
            c[op] += 1
    # segmentation: S+I or S+D adjacent pairs whose joined words match the other side within 1 char
    for (o1, r1, h1), (o2, r2, h2) in zip(al, al[1:]):
        if o1 == "S" and o2 == "I" and char_lev(r1, h1 + h2) <= 1 or o1 == "I" and o2 == "S" and char_lev(r2, h1 + h2) <= 1:
            c["seg"] += 1
        if o1 == "S" and o2 == "D" and char_lev(r1 + r2, h1) <= 1 or o1 == "D" and o2 == "S" and char_lev(r1 + r2, h2) <= 1:
            c["seg"] += 1
    c["err"] = c["S_near"] + c["S_far"] + c["D"] + c["I"]
    c["n"] = len(r)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", required=True)
    ap.add_argument("--penalty", default="0.0")
    ap.add_argument("--nbest", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    g = {}
    for l in open(a.greedy):
        x = json.loads(l)
        g[x["path"]] = x
    rows = []
    for l in open(a.nbest):
        x = json.loads(l)
        if x["path"] not in g:
            continue
        gx = g[x["path"]]
        hyp_g = gx["hyp"].get(a.penalty, gx["hyp"].get(str(float(a.penalty))))
        texts = [h["text"] for h in x["hyps"]]
        cl = [classify(x["ref"], t) for t in texts]
        oi = min(range(len(cl)), key=lambda i: (cl[i]["err"], i))
        rows.append({"set": x["set"], "dur": gx["duration"], "g": classify(x["ref"], hyp_g), "b": cl[0], "o": cl[oi], "orank": oi})
    print(f"joined {len(rows)} clips")
    res = {}
    keys = ("S_near", "S_far", "D", "I", "seg")

    def summ(sel):
        N = sum(r["g"]["n"] for r in sel) or 1
        out = {"clips": len(sel), "words": N}
        for s in "gbo":
            e = sum(r[s]["err"] for r in sel)
            out[{"g": "greedy", "b": "beam", "o": "oracle"}[s]] = {"wer": round(100 * e / N, 2),
                **{k: round(100 * sum(r[s][k] for r in sel) / N, 2) for k in keys}}
        same = sum(r["b"]["err"] == r["o"]["err"] for r in sel)
        nogood = sum(r["o"]["err"] / max(1, r["o"]["n"]) >= 0.30 for r in sel)
        out["clips_same"] = same; out["clips_ranking"] = len(sel) - same; out["clips_no_good_hyp"] = nogood
        ge, be, oe = (sum(r[s]["err"] for r in sel) for s in "gbo")
        out["gap_words"] = {"greedy_to_beam": ge - be, "beam_to_oracle": be - oe, "oracle_residual": oe}
        out["no_good_hyp_share_of_oracle_errors"] = round(
            sum(r["o"]["err"] for r in sel if r["o"]["err"] / max(1, r["o"]["n"]) >= 0.30) / max(1, oe), 3)
        return out

    for s in sorted({r["set"] for r in rows}):
        sel = [r for r in rows if r["set"] == s]
        res[s] = summ(sel)
        res[s]["by_duration"] = {name: summ([r for r in sel if lo <= r["dur"] < hi]) for lo, hi, name in BUCKETS}
    for s, v in res.items():
        gr, be, orc = v["greedy"], v["beam"], v["oracle"]
        print(f"\n== {s}: {v['clips']} clips {v['words']} words | ranking clips {v['clips_ranking']} | no-good-hyp clips "
              f"{v['clips_no_good_hyp']} ({100 * v['no_good_hyp_share_of_oracle_errors']:.0f}% of oracle errors) | gap {v['gap_words']}")
        print(f"   {'':8} {'WER':>6} {'S_near':>7} {'S_far':>6} {'D':>5} {'I':>5} {'seg':>5}")
        for name, x in (("greedy", gr), ("beam", be), ("oracle", orc)):
            print(f"   {name:8} {x['wer']:6.2f} {x['S_near']:7.2f} {x['S_far']:6.2f} {x['D']:5.2f} {x['I']:5.2f} {x['seg']:5.2f}")
        for b, x in v["by_duration"].items():
            if x["clips"]:
                print(f"   {b:7} n={x['clips']:4} greedy {x['greedy']['wer']:6.2f} (D {x['greedy']['D']:5.2f}) beam {x['beam']['wer']:6.2f} "
                      f"oracle {x['oracle']['wer']:6.2f} no-good {x['clips_no_good_hyp']}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
