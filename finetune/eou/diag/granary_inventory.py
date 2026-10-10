#!/usr/bin/env python3
"""Per-language inventory of espnet/yodas-granary (Granary's YODAS: pre-segmented clips, Whisper-v3 labels).

    $PY granary_inventory.py [--langs it,fr,de] [--json out.json]

For every data/<lang><NNN>/{asr_only,ast} directory: shard count and bytes (HF tree API). Hours,
clips and distinct videos are CALIBRATED on the first `ast` shard of each language by reading only
the `duration` and `original_audio_id` columns (HfFileSystem range reads, no audio), then scaled by
bytes; the estimate is labelled as such. Licence: CC-BY-3.0 (dataset card). Needs /root/.hf_token.
"""
import argparse
import collections
import json
import re
import urllib.request

REPO = "espnet/yodas-granary"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    tok = open("/root/.hf_token").read().strip()
    fs = HfFileSystem(token=tok)

    def tree(p):
        req = urllib.request.Request(f"https://huggingface.co/api/datasets/{REPO}/tree/main/{p}", headers={"Authorization": f"Bearer {tok}"})
        return json.load(urllib.request.urlopen(req, timeout=60))

    dirs = [d["path"].split("/")[1] for d in tree("data") if d["type"] == "directory"]
    by = collections.defaultdict(list)
    for d in dirs:
        by[re.match(r"[a-z]+", d).group()].append(d)
    want = a.langs.split(",") if a.langs else sorted(by)
    rows = []
    for lang in want:
        tot = {"asr_only": [0, 0], "ast": [0, 0]}; first = None
        for d in by.get(lang, []):
            for kind in tot:
                try:
                    fl = [x for x in tree(f"data/{d}/{kind}") if x["type"] == "file"]
                except Exception:  # noqa: BLE001
                    continue
                tot[kind][0] += len(fl); tot[kind][1] += sum(x.get("size", 0) for x in fl)
                if kind == "ast" and first is None and fl:
                    first = fl[0]
        cal = None
        if first:
            with fs.open(f"datasets/{REPO}/{first['path']}") as f:
                t = pq.read_table(f, columns=["duration", "original_audio_id"]).to_pylist()
            h = sum(float(r["duration"]) for r in t) / 3600
            cal = {"shard": first["path"], "MB": round(first["size"] / 1e6), "hours": round(h, 2), "clips": len(t),
                   "videos": len({r["original_audio_id"] for r in t}), "GB_per_h": round(first["size"] / 1e9 / max(h, 1e-9), 3)}
        gb = (tot["asr_only"][1] + tot["ast"][1]) / 1e9
        est_h = gb / cal["GB_per_h"] if cal else None
        rows.append({"lang": lang, "dirs": by.get(lang, []), "shards_asr_only": tot["asr_only"][0], "shards_ast": tot["ast"][0],
                     "GB": round(gb, 1), "est_hours": round(est_h) if est_h else None, "calibration": cal,
                     "est_clips": round(est_h * cal["clips"] / cal["hours"]) if cal else None, "licence": "CC-BY-3.0"})
        r = rows[-1]
        print(f"{lang:4s} dirs {len(r['dirs']):2d} shards {r['shards_ast']:5d} ast + {r['shards_asr_only']:5d} asr_only  "
              f"{r['GB']:8.1f} GB  ~{r['est_hours'] or 0:7d} h (est)  cal: {cal and (cal['hours'], cal['clips'], cal['videos'])}", flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
