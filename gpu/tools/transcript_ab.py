#!/usr/bin/env python3
"""transcript_ab.py -- the quality gate of a numerical change (S14-8b).

Two sets of unloaded transcripts of the SAME bank ({clip: text}, as
gpu/tools/gpu_qualify.sh and tools/bench/v2_qualify.sh write them to
reference.json), A = the baseline, B = the arm. Reports:

  * how many clips transcribe byte-identically, and the ones that do not,
    side by side with the human reference;
  * WER and CER of each set against the bank's human references (strict, and
    number-format-free, as tools/bench/streaming_metrics.py defines them),
    utterance mean and corpus (total edits / total reference words);
  * a paired count: on the differing clips, how many B made better, worse, same.

    gpu/tools/transcript_ab.py A/reference.json B/reference.json \
        --manifest samples/stress-en/manifest.json [--lang en] [--max-worse-wer-delta 0.002]

Exit 0 when B's corpus WER is not worse than A's by more than
--max-worse-wer-delta (registered before the run: 0.002 absolute, i.e. 0.2
points), 1 otherwise. It never decides alone: the diff is printed so a person
reads what changed."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "bench"))
import streaming_metrics as M  # noqa: E402


def load_refs(manifest):
    d = json.load(open(manifest))
    clips = d["clips"] if isinstance(d, dict) and "clips" in d else d
    if isinstance(clips, dict):
        clips = list(clips.values())
    out = {}
    for c in clips:
        f = c.get("file") or c.get("clip")
        t = c.get("text") or c.get("transcript") or c.get("reference")
        if f and t:
            out[f] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--max-worse-wer-delta", type=float, default=0.002)
    ap.add_argument("--show", type=int, default=15)
    x = ap.parse_args()
    A, B = json.load(open(x.a)), json.load(open(x.b))
    refs = load_refs(x.manifest)
    clips = sorted(set(A) & set(B))
    if len(clips) != len(A) or len(clips) != len(B):
        print(f"WARNING: the two sets cover {len(A)} and {len(B)} clips, {len(clips)} in common")
    same = [c for c in clips if A[c] == B[c]]
    diff = [c for c in clips if A[c] != B[c]]
    print(f"clips {len(clips)}: identical {len(same)}, different {len(diff)}")

    def score(T):
        ws, cs, wf = [], [], []
        edits = words = 0
        for c in clips:
            r = M.reference_for(refs, c)
            if r is None:
                continue
            w = M.wer(T[c], r)
            if w is None:
                continue
            n = len(M.normalise(r).split())
            ws.append(w); edits += w * n; words += n
            cs.append(M.cer(T[c], r))
            f = M.wer_format_free(T[c], r, x.lang)
            if f is not None:
                wf.append(f)
        return {"n": len(ws), "wer_mean": sum(ws) / len(ws), "wer_corpus": edits / words,
                "cer_mean": sum(cs) / len(cs), "wer_ff_mean": sum(wf) / len(wf) if wf else None}

    sa, sb = score(A), score(B)
    for tag, s in (("A", sa), ("B", sb)):
        print(f"{tag}: n={s['n']}  WER corpus {s['wer_corpus']:.5f}  mean {s['wer_mean']:.5f}  "
              f"CER mean {s['cer_mean']:.5f}  WER format-free mean {s['wer_ff_mean']:.5f}")
    better = worse = even = 0
    for c in diff:
        r = M.reference_for(refs, c)
        if r is None:
            continue
        wa, wb = M.wer(A[c], r), M.wer(B[c], r)
        if wb < wa: better += 1
        elif wb > wa: worse += 1
        else: even += 1
    print(f"on the {len(diff)} differing clips, B vs A: better {better}, worse {worse}, same WER {even}")
    for c in diff[: x.show]:
        r = M.reference_for(refs, c)
        print(f"--- {c}\n  ref: {r}\n  A  : {A[c]}\n  B  : {B[c]}")
    delta = sb["wer_corpus"] - sa["wer_corpus"]
    ok = delta <= x.max_worse_wer_delta
    print(f"corpus WER delta B-A {delta:+.5f} vs allowed +{x.max_worse_wer_delta:.5f}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
