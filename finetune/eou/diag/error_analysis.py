#!/usr/bin/env python3
"""Where the student's word errors come from, per model and val set (CPU only).

    error_analysis.py --dir ceiling --models P40,A0,A2 --teacher Nemotron \\
        [--lexicon mix/it/train_cv.json,manifests/train_40h.json,...] [--json out.json]

Reads paired_eval.py's DIR/sub_<set>.json (refs) + DIR/hyp_<model>_<set>.json and the
teacher's DIR/teacher_<set>.json. Same a-z normalisation as two_stage.py. Per model and set:
  WER split into S / D / I (word alignment backtrace), CER, WER/CER ratio;
  substitutions by character distance: near-miss (1 char), 2 chars, far (>= 3);
  segmentation errors: a ref word split into two hyp words or two ref words merged
      (the joined pair matches the other side within 1 char);
  non-words: hyp words absent from the lexicon (every training transcript), i.e. the
      model spelled a sound instead of a word -- the signature of a weak internal LM /
      output units, not of hearing;
  WER with spaces ignored (CER of the de-spaced strings) to show the space share.
The question it answers (S15-14): is the ~3x WER/CER of the student near-misses and
non-words (output representation / predictor) or far substitutions and deletions (acoustics)?
"""
import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path


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


def align(r, h):
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
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("=" if r[i - 1] == h[j - 1] else "S", r[i - 1], h[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", r[i - 1], None)); i -= 1
        else:
            ops.append(("I", None, h[j - 1])); j -= 1
    return ops[::-1]


def segmentation(ops):
    """Count S+I / I+S (one ref word -> two hyp words) and S+D / D+S (two ref -> one hyp) pairs
    whose joined side matches the other within 1 char."""
    seg = 0
    for a, b in zip(ops, ops[1:]):
        kinds = {a[0], b[0]}
        if kinds == {"S", "I"}:
            ref = a[1] or b[1]; hyp = (a[2] or "") + (b[2] or "")
        elif kinds == {"S", "D"}:
            hyp = a[2] or b[2]; ref = (a[1] or "") + (b[1] or "")
        else:
            continue
        if lev(ref, hyp) <= 1:
            seg += 1
    return seg


def analyse(rows, hyp, lex):
    c = collections.Counter(); conf = collections.Counter()
    for r in rows:
        ref, h = norm(r["text"]), norm(hyp.get(r["audio_filepath"], ""))
        rw, hw = ref.split(), h.split()
        ops = align(rw, hw)
        c["words"] += len(rw); c["chars"] += len(ref); c["cerr"] += lev(ref, h)
        c["chars_ns"] += len(ref.replace(" ", "")); c["cerr_ns"] += lev(ref.replace(" ", ""), h.replace(" ", ""))
        c["seg"] += segmentation(ops)
        for op, a, b in ops:
            c[op] += 1
            if op == "S":
                k = lev(a, b); c["S1" if k == 1 else "S2" if k == 2 else "S3+"] += 1
                conf[(a, b)] += 1
            if b is not None:
                c["hw"] += 1; c["nonword"] += lex is not None and b not in lex
    w = c["words"] or 1; s = c["S"] or 1
    errs = c["S"] + c["D"] + c["I"]
    out = {"wer": 100 * errs / w, "cer": 100 * c["cerr"] / max(1, c["chars"]),
           "S": 100 * c["S"] / w, "D": 100 * c["D"] / w, "I": 100 * c["I"] / w,
           "S_near1": 100 * c["S1"] / s, "S_2": 100 * c["S2"] / s, "S_far": 100 * c["S3+"] / s,
           "seg_pairs_per100w": 100 * c["seg"] / w,
           "cer_nospace": 100 * c["cerr_ns"] / max(1, c["chars_ns"]),
           "top_conf": [f"{a}->{b} x{n}" for (a, b), n in conf.most_common(8)]}
    out["wer_cer"] = out["wer"] / max(1e-9, out["cer"])
    if lex is not None:
        out["nonword_pct_of_hyp"] = 100 * c["nonword"] / max(1, c["hw"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--models", default="P40,A0,A2")
    ap.add_argument("--teacher", default="Nemotron")
    ap.add_argument("--sets", default="mls,fleurs,cv,vp")
    ap.add_argument("--lexicon", default="", help="comma list of manifests whose 'text' builds the word list")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    D = Path(a.dir)
    lex = None
    if a.lexicon:
        lex = set()
        for p in a.lexicon.split(","):
            for l in open(p):
                lex.update(norm(json.loads(l).get("text", "")).split())
        print(f"== lexicon {len(lex)} words from {len(a.lexicon.split(','))} manifests")
    res = {}
    for s in a.sets.split(","):
        rows = [json.loads(l) for l in open(D / f"sub_{s}.json")]
        hyps = {m: json.load(open(D / f"hyp_{m}_{s}.json")) for m in a.models.split(",") if m}
        if a.teacher:
            hyps[a.teacher] = {r["audio_filepath"]: r.get("teacher", "") for r in map(json.loads, open(D / f"teacher_{s}.json"))}
        for m, h in hyps.items():
            res[f"{m}/{s}"] = analyse(rows, h, lex)
    cols = ["wer", "cer", "wer_cer", "S", "D", "I", "S_near1", "S_2", "S_far", "seg_pairs_per100w", "cer_nospace"]
    if lex is not None:
        cols.append("nonword_pct_of_hyp")
    print("| model/set | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for k, v in res.items():
        print(f"| {k} | " + " | ".join(f"{v[c]:.2f}" for c in cols) + " |")
    for k, v in res.items():
        print(f"== {k} top confusions: {'; '.join(v['top_conf'])}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
