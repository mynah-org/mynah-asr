#!/usr/bin/env python3
"""Sanity report of a built Granary pool BEFORE its teacher pass: is it the pool we meant (definition, not bytes)?

    python3 pool_sanity.py --pool train_yg.json --old "it-yg=a.json,it-yg6=b.json" [--expect "clips=156247,h=500.7,videos=9441,shards=201"] [--hash 1]

Prints clips, physical hours, unique videos, shard count and index range per Granary dir, video and shard overlap
with each --old pool, duration p10/p50/p90/max, clips over 30 s, duplicate paths, and (--hash 1) duplicate audio
(sha1 of the decoded samples' file bytes, 16 threads). With --expect, the relative difference to each expected count
is shown (a rebuild of the same shard spread should be within ~1 %).
Clip names are <dir>_<shard>_<video>_<...>.<ext> (prepare_mix.py's Granary cut).
"""
import argparse, hashlib, json, re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

NAME = re.compile(r"^(it\d{3})_(\d{8})_(.{11})_")


def parse(path):
    m = NAME.match(Path(path).name)
    return (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def info(rows):
    sh, vids = defaultdict(set), set()
    for r in rows:
        d, s, v = parse(r["audio_filepath"])
        if d:
            sh[d].add(int(s)); vids.add(v)
    return sh, vids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--old", default="")
    ap.add_argument("--expect", default="")
    ap.add_argument("--hash", type=int, default=0)
    a = ap.parse_args()
    rows = load(a.pool)
    d = np.array([r["duration"] for r in rows])
    sh, vids = info(rows)
    paths = Counter(r["audio_filepath"] for r in rows)
    rep = {"clips": len(rows), "h": round(d.sum() / 3600, 1), "videos": len(vids),
           "shards": sum(len(v) for v in sh.values()),
           "shards_per_dir": {k: {"n": len(v), "min": min(v), "max": max(v)} for k, v in sorted(sh.items())},
           "dur_p10_p50_p90_max": [round(float(np.percentile(d, q)), 1) for q in (10, 50, 90)] + [round(float(d.max()), 1)],
           "clips_over_30s": int((d > 30).sum()), "dup_paths": sum(c - 1 for c in paths.values() if c > 1),
           "unparsed_names": sum(1 for r in rows if parse(r["audio_filepath"])[0] is None)}
    for spec in (x for x in a.old.split(",") if x):
        n, p = spec.split("=", 1)
        osh, ov = info(load(p))
        rep[f"overlap_{n}"] = {"videos": len(vids & ov),
                               "shards": sum(len(sh.get(k, set()) & s) for k, s in osh.items()),
                               "old_videos": len(ov)}
    if a.hash:
        def h(p):
            return hashlib.sha1(open(p, "rb").read()).hexdigest()
        with ThreadPoolExecutor(16) as ex:
            hs = Counter(ex.map(h, list(paths)))
        rep["dup_audio"] = sum(c - 1 for c in hs.values() if c > 1)
    if a.expect:
        rep["vs_expected_pct"] = {k: round(100 * (rep[k] - float(v)) / float(v), 2)
                                  for k, v in (x.split("=") for x in a.expect.split(","))}
    print(json.dumps(rep, indent=1))
    print("| clips | physical h | videos | shards | dur p10/p50/p90/max s | > 30 s | dup paths | dup audio |")
    print(f"| {rep['clips']} | {rep['h']} | {rep['videos']} | {rep['shards']} | {'/'.join(map(str, rep['dur_p10_p50_p90_max']))} "
          f"| {rep['clips_over_30s']} | {rep['dup_paths']} | {rep.get('dup_audio', '-')} |")


if __name__ == "__main__":
    main()
