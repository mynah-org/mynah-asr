#!/usr/bin/env python3
"""Label-quality probe of a manifest with an independent teacher (Nemotron 0.6B in Mynah, CPU).

    python3 teacher_check.py --manifest train_vp.json --n 300 [--model models/nemotron-3.5-asr-streaming-0.6b]
        [--lang it] [--jobs 24] [--json out.json]     (run from the mynah-asr checkout)

A random sample of --n clips (fixed seed) is transcribed by the teacher; per clip the
accent-insensitive WER of the teacher against the manifest text. A high-WER tail means
either a hard domain for the teacher too or labels/segments that do not match the audio;
comparing two manifests of the same language (e.g. VoxPopuli vs Common Voice train)
separates the two. Prints the WER distribution and the worst clips (ref vs teacher).
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import subprocess
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="models/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest)]
    rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))

    def run(r):
        o = subprocess.run(["./mynah-asr", "transcribe", "-m", a.model, "-i", r["audio_filepath"], "--lang", a.lang],
                           capture_output=True, text=True, timeout=600)
        t = o.stdout.strip().splitlines()[-1] if o.stdout.strip() else ""
        ref = norm(r["text"])
        return {"audio_filepath": r["audio_filepath"], "duration": r["duration"], "speaker": r.get("speaker"), "ref": ref,
                "teacher": norm(t), "teacher_wer": round(100 * lev(ref.split(), norm(t).split()) / max(1, len(ref.split())), 1)}

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run, rows))
    w = sorted(x["teacher_wer"] for x in res)
    pct = lambda q: w[min(len(w) - 1, int(q * len(w)))]  # noqa: E731
    pooled = 100 * sum(lev(x["ref"].split(), x["teacher"].split()) for x in res) / max(1, sum(len(x["ref"].split()) for x in res))
    print(f"== {a.manifest}: n={len(res)} teacher pooled WER {pooled:.1f}%, per-clip p50 {pct(.5)} p75 {pct(.75)} p90 {pct(.9)}; "
          f">50 %: {sum(v > 50 for v in w)}  >80 %: {sum(v > 80 for v in w)}  empty teacher: {sum(not x['teacher'] for x in res)}")
    for x in sorted(res, key=lambda x: -x["teacher_wer"])[:12]:
        print(f"   {x['teacher_wer']:6.1f}% {x['duration']:5.1f}s | ref {x['ref'][:70]}\n          teacher {x['teacher'][:70]}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
