#!/usr/bin/env python3
"""WER/CER (with S/D/I) of any Canary .nemo on the frozen Italian eval.

    $PY eval_it.py --nemo $FT/runs/A-5h/final.nemo
    $PY eval_it.py --nemo $FT/models/canary-180m-flash-it.nemo --tag it-init --limit 200

Sets: FLEURS it_it test (all) + MLS it test (all) [+ CV it test 2,000 if it
was built], scored separately and pooled, with the Open ASR Leaderboard
multilingual normaliser (lang=it). `--en` adds the FLEURS en_us sanity clips
(en/en, leaderboard English normaliser) for the forgetting check.
`--levels` (default -3,-20,-40) repeats the IT sets (and EN with --en) with
every clip re-levelled to that peak dBFS and re-quantised to PCM16: level
robustness before/after fine-tuning. Use `--levels ''` to skip.
Decoding: the model's own decoding config (beam_size 1 for Canary 180M),
prompt source=target=it, pnc=no, fp32 unless --bf16.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "common")]
import ftlib  # noqa: E402

FT = ftlib.FT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--sets", default=None, help="comma list of name=manifest; default: the frozen IT eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--en", action="store_true", help="also score the EN sanity clips")
    ap.add_argument("--out", default=None)
    ap.add_argument("--levels", default="-3,-20,-40", help="peak dBFS list for the level sweep ('' = skip)")
    a = ap.parse_args()

    import torch

    man = FT / "manifests"
    if a.sets:
        sets = dict(x.split("=", 1) for x in a.sets.split(","))
    else:
        sets = {"fleurs_it": man / "eval_fleurs_it.json", "mls_it": man / "eval_mls_it.json"}
        if (man / "eval_cv_it.json").exists():
            sets["cv_it"] = man / "eval_cv_it.json"
    tag = a.tag or Path(a.nemo).parent.name + "-" + Path(a.nemo).stem
    out = Path(a.out or FT / "evals" / tag)
    t0 = time.time()
    m = ftlib.load_model(a.nemo)
    res = {"nemo": a.nemo, "tag": tag, "env": ftlib.env_info(), "bf16": a.bf16, "limit": a.limit,
           a.lang: ftlib.eval_sets(m, sets, a.lang, pnc="no", batch_size=a.batch_size, bf16=a.bf16, limit=a.limit,
                                   dump_dir=out)}
    if a.en:
        res["en"] = ftlib.eval_sets(m, {"fleurs_en": man / "en_sanity.json"}, "en", pnc="no",
                                    batch_size=a.batch_size, bf16=a.bf16, dump_dir=out)
    res["preprocessor"] = ftlib.preprocessor_info(m)
    levels = ftlib.parse_levels(a.levels)
    if levels:
        lsets = dict(sets)
        if a.limit:  # keep the sweep on the same subset as the main pass
            lsets = {}
            for k, v in sets.items():
                p = out / f"limit_{k}.json"
                ftlib.write_manifest(p, ftlib.read_manifest(v)[: a.limit])
                lsets[k] = p
        res["level_sweep"] = {"levels_peak_dbfs": levels, a.lang: ftlib.level_sweep(m, lsets, a.lang, levels, a.batch_size, out)}
        if a.en:
            res["level_sweep"]["en"] = ftlib.level_sweep(m, {"fleurs_en": man / "en_sanity.json"}, "en", levels, a.batch_size)
    res["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["wall_s"] = round(time.time() - t0, 1)
    ftlib.write_json(out / "eval.json", res)
    print(f"wrote {out}/eval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
