#!/usr/bin/env python3
"""Where do premature <EOU> cuts fall in the sentence? Punctuation-based classification against a random control.

    cut_class.py --manifest eou_all.json --dump v11_eouv2.jsonl --tokenizer <dir with tokenizer.model> [--json out.json]

For every clip of an s2_reset dump (--dump, the frozen protocol) whose decode has a premature <EOU> (dump field
"premature"), each internal cut = the boundary between two consecutive <EOU> segments (ids split at 1024; pieces
< 1024 decoded with the stock SPE) is located in the PUNCTUATED reference (the manifest "text", raw):
- the raw text is split on whitespace; each raw token keeps a flag "punctuation after it" (any of , ; : . ! ? … or a
  closing quote / bracket followed by one of those, or a guillemet / dash that closes a clause) and is normalised
  with textnorm.norm(tok, "fr") (a token that normalises to nothing attaches its punctuation to the previous word);
- the normalised word list must equal the dump's "ref" word list, else the clip is counted as `unaligned` and skipped;
- hyp words before the cut are mapped to a reference boundary j (number of reference words before the cut) by a
  Levenshtein backtrace between the full hyp and the reference;
- the cut is AT-PUNCT if any of the boundaries j-1, j, j+1 carries punctuation (+-1 word tolerance), else MID-PHRASE.
Control: for the same clips, every internal reference word boundary (1 .. n-1) classified with the same +-1 rule,
weighted so each clip contributes as many control draws as it has cuts (the expectation, no sampling).
Prints and writes: cuts, mid-phrase share, control mid-phrase share, excess (pp), sentence-final share (. ! ? …).
Pre-registered use (report section 11, FR-E0): the truncated-prefix arm is admitted only if mid-phrase >= 40 % AND
excess >= +10 pp on DEV-EOU-FR-v2 for RC-FR v1.1.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "common")]
import textnorm  # noqa: E402

PUNCT = set(",;:.!?…")
FINAL = set(".!?…")
CLOSERS = set("»\"')]”’")


def ref_words(raw):
    """[(word, punct_after, final_after)] from the raw punctuated text."""
    out = []
    for tok in raw.replace("«", " ").replace("—", " — ").replace("–", " – ").split():
        core = tok.rstrip("".join(CLOSERS))
        p = bool(core) and core[-1] in PUNCT or tok in ("—", "–")
        f = bool(core) and core[-1] in FINAL
        ws = textnorm.norm(tok, "fr").split()
        if not ws:
            if out and (p or f):
                w, p0, f0 = out[-1]
                out[-1] = (w, p0 or p, f0 or f)
            continue
        for k, w in enumerate(ws):
            last = k == len(ws) - 1
            out.append((w, p and last, f and last))
    return out


def align_map(hyp, ref):
    """For each hyp prefix length k (0..len(hyp)), the number of ref words aligned before it (Levenshtein backtrace)."""
    n, m = len(hyp), len(ref)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]))
    i, j, mp = n, m, {n: m}
    while i > 0:
        if j > 0 and D[i][j] == D[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]):
            i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
            continue
        mp.setdefault(i, j)
    return [mp.get(k, 0) for k in range(n + 1)]


def at_punct(rw, j):
    """boundary j = after ref word j-1; +-1 word tolerance."""
    return any(1 <= b <= len(rw) and rw[b - 1][1] for b in (j - 1, j, j + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tokenizer", required=True, help="dir holding the stock SPE tokenizer.model")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=os.path.join(a.tokenizer, "tokenizer.model"))
    man = {}
    for line in open(a.manifest):
        r = json.loads(line)
        man[os.path.basename(r["audio_filepath"])] = r
    cuts = mid = final = 0
    ctrl_mid = 0.0
    unaligned = clips = 0
    examples = []
    for line in open(a.dump):
        x = json.loads(line)
        if not x.get("premature"):
            continue
        r = man.get(os.path.basename(x["audio"]))
        if r is None:
            unaligned += 1
            continue
        rw = ref_words(r["text"])
        if [w for w, _, _ in rw] != x["ref"].split():
            unaligned += 1
            continue
        segs, cur = [], []
        for t in x["ids"]:
            if t == 1024:
                segs.append(cur); cur = []
            elif t < 1024:
                cur.append(t)
        segs.append(cur)
        words = [textnorm.norm(sp.decode(s), "fr").split() for s in segs]
        hyp = [w for s in words for w in s]
        mp = align_map(hyp, [w for w, _, _ in rw])
        k, n_here = 0, 0
        for s in words[:-1]:
            k += len(s)
            if k == 0 or k == len(hyp):
                continue
            j = mp[k]
            if j <= 0 or j >= len(rw):
                continue
            cuts += 1; n_here += 1
            ap_ = at_punct(rw, j)
            mid += not ap_
            final += any(1 <= b <= len(rw) and rw[b - 1][2] for b in (j - 1, j, j + 1))
            if len(examples) < 40:
                examples.append(("AT  " if ap_ else "MID ") + " ".join(w for w, _, _ in rw[max(0, j - 4):j]) + " || "
                                + " ".join(w for w, _, _ in rw[j:j + 4]))
        if n_here:
            clips += 1
            inner = range(1, len(rw))
            ctrl_mid += n_here * sum(not at_punct(rw, b) for b in inner) / max(1, len(inner))
    res = {"dump": a.dump, "clips_with_cuts": clips, "cuts": cuts, "unaligned_clips": unaligned,
           "mid_phrase%": round(100 * mid / max(1, cuts), 1), "control_mid_phrase%": round(100 * ctrl_mid / max(1, cuts), 1),
           "sentence_final%": round(100 * final / max(1, cuts), 1)}
    res["excess_pp"] = round(res["mid_phrase%"] - res["control_mid_phrase%"], 1)
    res["rule_truncated_prefix_admitted"] = res["mid_phrase%"] >= 40 and res["excess_pp"] >= 10
    print("== CUT-CLASS", json.dumps(res))
    for e in examples:
        print("  ", e)
    if a.json:
        json.dump({**res, "examples": examples}, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
