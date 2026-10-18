#!/usr/bin/env python3
"""How the stage-2 training clips END (model-free): does stage 2 teach "pause / cut = <EOU>"?

    s2_endpoints.py --mix "mls=a.json,cv=b.json,..." [--n 600] [--max-dur 20] [--pad-prob 0.5 --pad-min 1 --pad-max 3] \
        [--patho frames.json] [--out e.json]

two_stage.py --eou-append puts <EOU> after EVERY stage-2 target; with probability --pad-prob the clip gets U(pad-min,
pad-max) s of near-digital quiet appended, otherwise the <EOU> sits on the clip's NATURAL tail. Per source (n clips
<= max-dur, fixed seed) an energy speech mask on 80 ms frames (speech = within 35 dB of the clip's loudest frame and
>= 10 dB above its 10th-percentile frame) gives:
  - natural trailing non-speech (clip end - last speech frame): p10 / p50 / p90, shares < 0.25 / < 0.5 / < 1 / > 3 s,
    and "cut" = < 0.1 s (the clip ends on speech);
  - the same for the EOU labels actually trained (mixture: 1 - pad-prob natural, pad-prob natural + U(pad));
  - internal pauses (non-speech runs between speech frames, i.e. followed by more speech): per clip >= 0.25 / 0.5 /
    1 / 2 s -- the only "pause -> continue" negatives stage 2 sees;
  - pause-length competition: for pauses in [0.08, 2.1] s (the range of the pathological premature EOUs), how many
    are followed by <EOU> (trailing, unpadded) vs by speech (internal), per source.
--patho: the pauses at which s2 fired a premature <EOU> (s2_frames.py output) placed in the trained-EOU distribution.
"""
import argparse
import json
import random

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mix", required=True)
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--max-dur", type=float, default=20.0)
ap.add_argument("--pad-prob", type=float, default=0.5)
ap.add_argument("--pad-min", type=float, default=1.0)
ap.add_argument("--pad-max", type=float, default=3.0)
ap.add_argument("--patho", default="")
ap.add_argument("--out", default="")
a = ap.parse_args()
FR = 0.08
N = int(FR * 16000)


def mask(x):
    f = x[: len(x) // N * N].reshape(-1, N)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    return db > max(db.max() - 35, np.percentile(db, 10) + 10)


def runs(m):
    """internal non-speech runs (between speech frames) in seconds, and trailing / leading non-speech."""
    on = np.nonzero(m)[0]
    if not len(on):
        return [], len(m) * FR, len(m) * FR
    inner, k = [], 0
    for t in range(on[0], on[-1] + 1):
        if not m[t]:
            k += 1
        elif k:
            inner.append(k * FR); k = 0
    return inner, (len(m) - 1 - on[-1]) * FR, on[0] * FR


pct = lambda v, q: round(float(np.percentile(v, q)), 2) if len(v) else None  # noqa: E731
share = lambda v, f: round(100 * float(np.mean([f(x) for x in v])), 1) if len(v) else None  # noqa: E731
rnd = random.Random(0)
res, allt, alli = {}, [], []
for spec in a.mix.split(","):
    name, path = spec.split("=", 1)
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("duration", 0) <= a.max_dur]
    rows = rnd.sample(rows, min(a.n, len(rows)))
    trail, inner_all, per_clip = [], [], {0.25: [], 0.5: [], 1.0: [], 2.0: []}
    for r in rows:
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        inner, tr, _ = runs(mask(x))
        trail.append(tr); inner_all += inner
        for th in per_clip:
            per_clip[th].append(sum(p >= th for p in inner))
    trained = [t if rnd.random() >= a.pad_prob else t + rnd.uniform(a.pad_min, a.pad_max) for t in trail]
    comp_trail = sum(1 for t in trail if 0.08 <= t <= 2.1) * (1 - a.pad_prob)   # unpadded EOU labels on such tails
    comp_inner = sum(1 for p in inner_all if 0.08 <= p <= 2.1)
    res[name] = {
        "clips": len(rows),
        "natural_trail_p10_p50_p90": [pct(trail, q) for q in (10, 50, 90)],
        "natural_trail_<0.1s_cut%": share(trail, lambda t: t < 0.1),
        "natural_trail_<0.25%": share(trail, lambda t: t < 0.25), "natural_trail_<0.5%": share(trail, lambda t: t < 0.5),
        "natural_trail_<1%": share(trail, lambda t: t < 1.0), "natural_trail_>3%": share(trail, lambda t: t > 3.0),
        "trained_EOU_tail_<0.5%": share(trained, lambda t: t < 0.5), "trained_EOU_tail_<1%": share(trained, lambda t: t < 1.0),
        "internal_pauses_per_clip_>=0.25/0.5/1/2": [round(float(np.mean(per_clip[t])), 2) for t in (0.25, 0.5, 1.0, 2.0)],
        "pause_0.08-2.1s_followed_by_EOU_vs_speech": [round(comp_trail, 1), comp_inner],
    }
    allt += trail; alli += inner_all
    print("==", name, json.dumps(res[name]), flush=True)

if a.patho:
    pa = [c["pause_after_stop_s"] for c in json.load(open(a.patho))["clips"]]
    res["_patho"] = {"premature_EOU_pauses_s": sorted(pa),
                     "share_of_natural_train_tails_<=_median_patho_pause%": share(allt, lambda t: t <= float(np.median(pa))),
                     "share_of_internal_train_pauses_<=_median_patho_pause%": share(alli, lambda p: p <= float(np.median(pa)))}
    print("== patho", json.dumps(res["_patho"]), flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
