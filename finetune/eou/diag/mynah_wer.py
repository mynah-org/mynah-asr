#!/usr/bin/env python3
"""WER / CER / empty / EOU of a Mynah pack in STREAMING (the serving semantics: the engine
resets encoder + decoder after each model <EOU> and keeps the stream open) on a NeMo manifest.

    python3 mynah_wer.py --pack models/parakeet-realtime-eou-120m-it --manifest eval_vp.json [--val-n 200] [--jobs 14]

Run from the mynah-asr checkout (CPU, ./mynah-asr stream --quant f32 --deltas). The text is the
concatenation of every `final` segment (across EOU resets); the same subsample (rs[::len//n][:n])
and accent-insensitive normalisation as two_stage.py, so the numbers sit next to its NeMo eval.
"""
import argparse
import concurrent.futures as cf
import json
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
    ap.add_argument("--pack", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-n", type=int, default=200); ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    rs = [json.loads(l) for l in open(a.manifest)]
    rs = rs[:: max(1, len(rs) // a.val_n)][: a.val_n]

    def run(r):
        out = subprocess.run(["./mynah-asr", "stream", "-m", a.pack, "-i", r["audio_filepath"], "--quant", "f32", "--deltas"],
                             capture_output=True, text=True, timeout=600).stdout
        finals, eous = [], 0
        for l in out.splitlines():
            if l.startswith("{"):
                o = json.loads(l)
                if o.get("type") == "final":
                    finals.append(o.get("text", ""))
                if o.get("type") == "eou" and o.get("source") == "model":
                    eous += 1
        return r, " ".join(finals), eous

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run, rs))
    we = wn = ce = cn = emp = eo = multi = 0
    for r, h, e in res:
        ref, h = norm(r["text"]), norm(h)
        we += lev(ref.split(), h.split()); wn += len(ref.split()); ce += lev(ref, h); cn += len(ref)
        emp += not h; eo += e > 0; multi += e > 1
    print(f"MYNAH {a.pack} on {a.manifest}: n={len(res)} WER {100 * we / wn:.2f} CER {100 * ce / cn:.2f} "
          f"empty {emp} clips_with_model_eou {eo} clips_with_2+_eou {multi}", flush=True)


if __name__ == "__main__":
    main()
