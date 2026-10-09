#!/usr/bin/env python3
"""C0: zero-shot probe of the UNTOUCHED Canary 180M on the frozen Italian eval.

Arms (source_lang = target_lang = the arm's language, pnc=no):
  it  <|it|> exists in spl_tokens but there is no Italian text sub-tokenizer,
      so the decoder can only emit en/de/es/fr pieces: expected poor; this is
      the floor the fine-tune has to beat.
  es, fr  "nearest language" floors on the same Italian audio.
Plus EN sanity on the FLEURS en_us clips (en/en) -- the reference for the
forgetting check of every fine-tuned run -- also re-levelled to fixed peaks
(--levels) to see whether the untouched model degrades on quiet audio, and the
preprocessor's `normalize` setting.

Writes $FT/probe/zeroshot.json and per-utterance hypotheses.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftlib  # noqa: E402

FT = ftlib.FT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", default=str(FT / "models" / "canary-180m-flash.nemo"))
    ap.add_argument("--arms", default="it,es,fr")
    ap.add_argument("--limit", type=int, default=None, help="utterances per eval set (default: all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=str(FT / "probe"))
    ap.add_argument("--levels", default="-3,-20,-40", help="EN level sweep on the untouched model ('' = skip)")
    a = ap.parse_args()

    import torch

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    m = ftlib.load_model(a.nemo)
    man = FT / "manifests"
    sets = {"fleurs_it": man / "eval_fleurs_it.json", "mls_it": man / "eval_mls_it.json"}
    if (man / "eval_cv_it.json").exists():
        sets["cv_it"] = man / "eval_cv_it.json"
    res = {"nemo": a.nemo, "env": ftlib.env_info(), "arms": {}}
    for arm in a.arms.split(","):
        print(f"== arm src=tgt={arm} on Italian audio", flush=True)
        out = {}
        for name, path in sets.items():
            rows = ftlib.read_manifest(path)[: a.limit] if a.limit else ftlib.read_manifest(path)
            hyps = ftlib.decode(m, rows, arm, arm, pnc="no", batch_size=a.batch_size)
            sc, per = ftlib.score([(r["text"], h) for r, h in zip(rows, hyps)], "it")  # always scored as Italian
            out[name] = sc
            ftlib.write_manifest(Path(a.out) / f"hyp_{arm}_{name}.jsonl",
                                 [{"ref": r["text"], "hyp": h, **u} for r, h, u in zip(rows, hyps, per)])
            print(f"  [{arm}/{name}] WER {sc['wer']:.2f} CER {sc['cer']:.2f} S/D/I "
                  f"{sc['words']['S']}/{sc['words']['D']}/{sc['words']['I']} empty={sc['empty_hyp']}", flush=True)
            for r, h in list(zip(rows, hyps))[:3]:
                print(f"    ref {r['text'][:80]!r}\n    hyp {h[:80]!r}")
        res["arms"][arm] = out
    print("== EN sanity (en/en)", flush=True)
    res["en"] = ftlib.eval_sets(m, {"fleurs_en": man / "en_sanity.json"}, "en", pnc="no",
                                batch_size=a.batch_size, dump_dir=Path(a.out) / "en")
    res["preprocessor"] = ftlib.preprocessor_info(m)
    print(f"  preprocessor: {res['preprocessor']}")
    res["en_level_sweep"] = ftlib.level_sweep(m, {"fleurs_en": man / "en_sanity.json"}, "en",
                                              ftlib.parse_levels(a.levels), a.batch_size)
    res["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["wall_s"] = round(time.time() - t0, 1)
    ftlib.write_json(Path(a.out) / "zeroshot.json", res)
    print(f"wrote {a.out}/zeroshot.json ({res['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
