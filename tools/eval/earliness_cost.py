#!/usr/bin/env python3
"""Was the earliness paid for?

Two models that differ in when they commit also differ in quality, and a pair
of corpus averages cannot tell you whether the second follows from the first.
This asks the paired question instead, per clip and from ONE run per model:

    does a clip where model B commits much earlier than model A
    also tend to be a clip where model B is more wrong?

Input: two `tools/eval/first_emission.py` outputs over the SAME bank, which
carry the timing and the transcript score for each clip from the same pass.

    python3 tools/eval/earliness_cost.py a.json b.json [--label-a ...] [--json out]

WHAT THIS CANNOT SHOW. Two different checkpoints differ in depth, width,
training data, training objective, cadence and vocabulary at once. A
correlation here is a property of the PAIR, never evidence that emitting
earlier causes an error. It is read to decide whether the trade is worth
investigating, not to conclude that it exists.
"""
from __future__ import annotations
import argparse, json, math, os, sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from streaming_metrics import pct  # noqa: E402  one definition of a percentile


def load(path):
    d = json.load(open(path))
    return d, {r["clip"]: r for r in d["rows"]}


def sign_test(diffs, eps=1e-9):
    """Better / worse / tied and the normal approximation over the non-ties."""
    lo = sum(1 for x in diffs if x < -eps)
    hi = sum(1 for x in diffs if x > eps)
    m = lo + hi
    z = (lo - m / 2) / ((m ** 0.5) / 2) if m else 0.0
    return lo, hi, len(diffs) - m, z


def spearman(xs, ys):
    """Rank correlation: the relationship need not be linear, and WER is a
    bounded ratio with a spike at zero, so Pearson would be the wrong tool."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):           # average ranks over ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return None, None
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    rho = num / den if den else 0.0
    # Fisher z, two-sided; n is large here so the approximation is fair
    z = 0.5 * math.log((1 + rho) / (1 - rho)) * math.sqrt(n - 3) if abs(rho) < 1 else float("inf")
    return rho, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default=None); ap.add_argument("--label-b", default=None)
    ap.add_argument("--json")
    ap.add_argument("--drop-composed", action="store_true",
                    help="keep only original single-utterance recordings. A bank that "
                         "concatenates two turns into one file is structurally unfair to "
                         "a model trained to END an utterance, and mixing the two "
                         "measures the corpus rather than the model")
    o = ap.parse_args()

    da, A = load(o.a); db, B = load(o.b)
    la = o.label_a or da.get("model", "A"); lb = o.label_b or db.get("model", "B")
    common = [c for c in A if c in B]
    scored = [c for c in common
              if A[c].get("wer") is not None and B[c].get("wer") is not None
              and A[c].get("speech_to_first_nonblank_ms") is not None
              and B[c].get("speech_to_first_nonblank_ms") is not None]
    if o.drop_composed:
        n0 = len(scored)
        scored = [c for c in scored if not A[c].get("composed")]
        print(f"  --drop-composed: {n0 - len(scored)} synthetic concatenation(s) removed")

    print(f"{la}  vs  {lb}")
    print(f"  {len(common)} clip(s) in both runs, {len(scored)} with both a timing and a score\n")

    # ---- corpus level, each on its own
    print("CORPUS")
    for lbl, d in ((la, da), (lb, db)):
        print(f"  {lbl:<34} WER mean {d['wer']['mean']:.4f}  p50 {d['wer']['p50']:.4f}"
              f"  corpus {d['wer_corpus']:.4f}   CER mean {d['cer']['mean']:.4f}"
              f"   empty {d.get('empty', 0)}  errors {d['errors']}")
    print()

    dt = [B[c]["speech_to_first_nonblank_ms"] - A[c]["speech_to_first_nonblank_ms"] for c in scored]
    dw = [B[c]["wer"] - A[c]["wer"] for c in scored]
    dc = [B[c]["cer"] - A[c]["cer"] for c in scored]
    dwf = [B[c]["wer_ff"] - A[c]["wer_ff"] for c in scored
           if A[c].get("wer_ff") is not None and B[c].get("wer_ff") is not None]

    print(f"PAIRED  ({lb} minus {la}; negative = {lb} earlier / better)")
    for name, v in (("first non-blank, ms", dt), ("WER", dw), ("CER", dc), ("WER format-free", dwf)):
        lo, hi, tie, z = sign_test(v)
        print(f"  {name:<22} p50 {pct(v,50):+9.4f}  mean {sum(v)/len(v):+9.4f}   "
              f"{lb} better {lo:4d} / worse {hi:4d} / tied {tie:4d}   z={z:+.2f}")
    print()

    # ---- THE question: is a bigger head start bought with a bigger error?
    rho, z = spearman(dt, dw)
    print("IS THE EARLINESS PAID FOR?")
    print(f"  Spearman rho(delta first non-blank, delta WER) = {rho:+.4f}   z={z:+.2f}")
    print("  A POSITIVE rho would mean: the more time it saves (more negative dt),")
    print("  the LOWER its WER delta -- i.e. earliness comes with no penalty here.")
    print("  A NEGATIVE rho would mean the head start is bought with errors.\n")

    q = sorted(dt)
    edges = [pct(q, 25), pct(q, 50), pct(q, 75)]
    print(f"  by head-start quartile (dt cut at {edges[0]:.0f} / {edges[1]:.0f} / {edges[2]:.0f} ms)")
    print(f"  {'bucket':<22} {'n':>4} {'dt p50':>9} {'dWER mean':>10} {'dWER p50':>9} {'dCER mean':>10}")
    buckets = [("Q1 earliest", lambda x: x <= edges[0]),
               ("Q2", lambda x: edges[0] < x <= edges[1]),
               ("Q3", lambda x: edges[1] < x <= edges[2]),
               ("Q4 least early", lambda x: x > edges[2])]
    rows_out = []
    for name, f in buckets:
        idx = [i for i, x in enumerate(dt) if f(x)]
        if not idx:
            continue
        bt = [dt[i] for i in idx]; bw = [dw[i] for i in idx]; bc = [dc[i] for i in idx]
        print(f"  {name:<22} {len(idx):>4} {pct(bt,50):>9.0f} {sum(bw)/len(bw):>10.4f} "
              f"{pct(bw,50):>9.4f} {sum(bc)/len(bc):>10.4f}")
        rows_out.append({"bucket": name, "n": len(idx), "dt_p50": pct(bt, 50),
                         "dwer_mean": sum(bw) / len(bw), "dcer_mean": sum(bc) / len(bc)})
    print()

    # ---- stratification the manifest already supports
    print("BY LENGTH CLASS")
    print(f"  {'class':<10} {'n':>4} {'dt p50':>9} {'dWER mean':>10} {'A WER':>8} {'B WER':>8}")
    for k in ("short", "medium", "long"):
        idx = [i for i, c in enumerate(scored) if A[c].get("class") == k]
        if not idx:
            continue
        print(f"  {k:<10} {len(idx):>4} {pct([dt[i] for i in idx],50):>9.0f} "
              f"{sum(dw[i] for i in idx)/len(idx):>10.4f} "
              f"{sum(A[scored[i]]['wer'] for i in idx)/len(idx):>8.4f} "
              f"{sum(B[scored[i]]['wer'] for i in idx)/len(idx):>8.4f}")
    print()
    print("ORIGINAL FLEURS vs SYNTHETIC CONCATENATIONS")
    for name, want in (("original", False), ("composed", True)):
        idx = [i for i, c in enumerate(scored) if bool(A[c].get("composed")) is want]
        if not idx:
            continue
        print(f"  {name:<10} {len(idx):>4} dt p50 {pct([dt[i] for i in idx],50):>7.0f}  "
              f"dWER mean {sum(dw[i] for i in idx)/len(idx):+.4f}")

    if o.json:
        json.dump({"a": la, "b": lb, "n": len(scored),
                   "spearman_dt_dwer": rho, "spearman_z": z,
                   "paired": {"dt_p50": pct(dt, 50), "dwer_mean": sum(dw) / len(dw),
                              "dcer_mean": sum(dc) / len(dc)},
                   "buckets": rows_out}, open(o.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
