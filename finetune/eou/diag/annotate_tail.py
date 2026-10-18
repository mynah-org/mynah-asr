#!/usr/bin/env python3
"""Annotate stage-2 manifests with `trail_s` = natural trailing non-speech (model-free), for two_stage.py --eou-min-tail.

    annotate_tail.py --mix "mls=a.json,cv=b.json,..." --out DIR [--min-tail 0.5] [--max-dur 20] [--workers 12]

Speech mask = s2_endpoints.py's (80 ms frames; speech = within 35 dB of the clip's loudest frame AND >= 10 dB above
its 10th-percentile frame); trail_s = clip end - end of the last speech frame. Every row is kept, in order, with
`trail_s` added (DIR/<name>.json); nothing is filtered (two_stage.py only drops the <EOU> of unpadded samples with
trail_s < --eou-min-tail). Report per source and aggregate: clips | hours | accepted % (trail_s >= min-tail) |
hours accepted | trail_s p10/p50/p90 | duration p10/p50/p90 of accepted vs rejected (rows <= max-dur, the ones
stage 2 trains on).
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

FR = 0.08
N = int(FR * 16000)


def trail(path):
    try:
        x, _ = sf.read(path, dtype="float32")
    except Exception:  # noqa: BLE001
        return None
    x = x.mean(1) if x.ndim > 1 else x
    f = x[: len(x) // N * N].reshape(-1, N)
    if not len(f):
        return 0.0
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    on = np.nonzero(db > max(db.max() - 35, np.percentile(db, 10) + 10))[0]
    return round(float((len(f) - 1 - on[-1]) * FR), 2) if len(on) else round(len(f) * FR, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-tail", type=float, default=0.5)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pc = lambda v: "/".join(f"{np.percentile(v, q):.2f}" for q in (10, 50, 90)) if len(v) else "-"  # noqa: E731
    print("| source | clips <= max-dur | h | accepted % | h accepted | trail_s p10/p50/p90 | dur accepted p10/p50/p90 | dur rejected p10/p50/p90 |")
    print("|---|---|---|---|---|---|---|---|")
    tot = {"n": 0, "h": 0.0, "acc": 0, "hacc": 0.0}
    with Pool(a.workers) as pool:
        for spec in a.mix.split(","):
            name, path = spec.split("=", 1)
            rows = [json.loads(l) for l in open(path)]
            tails = pool.map(trail, [r["audio_filepath"] for r in rows], chunksize=64)
            bad = sum(t is None for t in tails)
            assert not bad, f"{name}: {bad} unreadable clips"
            with open(f"{a.out}/{name}.json", "w") as f:
                for r, t in zip(rows, tails):
                    f.write(json.dumps({**r, "trail_s": t}, ensure_ascii=False) + "\n")
            use = [(r["duration"], t) for r, t in zip(rows, tails) if r["duration"] <= a.max_dur]
            d = np.array([u[0] for u in use]); t = np.array([u[1] for u in use]); ok = t >= a.min_tail
            print(f"| {name} | {len(use)} | {d.sum() / 3600:.1f} | {100 * ok.mean():.1f} | {d[ok].sum() / 3600:.1f} | {pc(t)} | "
                  f"{pc(d[ok])} | {pc(d[~ok])} |", flush=True)
            tot["n"] += len(use); tot["h"] += d.sum() / 3600; tot["acc"] += int(ok.sum()); tot["hacc"] += d[ok].sum() / 3600
    print(f"| all | {tot['n']} | {tot['h']:.1f} | {100 * tot['acc'] / tot['n']:.1f} | {tot['hacc']:.1f} | | | |")


if __name__ == "__main__":
    main()
