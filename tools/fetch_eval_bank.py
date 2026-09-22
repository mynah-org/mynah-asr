#!/usr/bin/env python3
"""fetch_eval_bank.py — the EN/FR quality bank, with ground truth.

    cd tools && uv run python fetch_eval_bank.py --dry-run
    cd tools && uv run python fetch_eval_bank.py --n 200

WHY.  `samples/` holds 2 French and 2 short English utterances. That is a
DIAGNOSTIC corpus: it has told us a great deal about mechanisms, and it cannot
support a sentence like "top English and French quality". With n = 2 the 95 %
interval on a word error rate covers essentially the whole range; a gate built
on it would pass anything.

WHAT IT BUILDS.  FLEURS (google/fleurs, CC-BY 4.0) **test** split — a split the
repo has not used for anything, so nothing was tuned on it — English and French,
N utterances each, 16 kHz mono PCM16, with the dataset's own transcription as
ground truth. Same source and same streaming-tar mechanism as
`fetch_stress_bank.py`; the tarball is never stored.

Ground truth is the oracle. The model's own offline output is never used as one.

Output: `samples/eval-bank/<lang>/fleurs_<id>.wav` and a manifest in the shape
`samples/manifest.json` uses, so every existing tool reads it unchanged.
Gitignored: rule 6, no audio in the repo.

Exit: 0 ok, 2 usage.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
CACHE = Path("/tmp/fleurs-eval")
SR = 16000
LANGS = {"en": "en_us", "fr": "fr_fr"}


def load_records(cfg, split):
    path = CACHE / f"{cfg}_{split}.tsv"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", f"{BASE}/{cfg}/{split}.tsv", "-o", str(path)],
                       check=True)
    recs, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) < 7 or r[1] in seen:
                continue
            seen.add(r[1])
            recs.append({"fleurs_id": int(r[0]), "source_file": r[1], "text": r[2],
                         "text_norm": r[3], "duration_sec": int(r[5]) / float(SR),
                         "gender": r[6].strip().lower()})
    # deterministic, and not the TSV's order
    recs.sort(key=lambda r: (r["fleurs_id"], r["source_file"]))
    return recs


def extract(cfg, split, files, dest):
    want = [f for f in files if not (dest / split / f).exists()]
    if not want:
        return
    dest.mkdir(parents=True, exist_ok=True)
    fd, listfile = tempfile.mkstemp(suffix=".txt", prefix="fleurs-eval-")
    with os.fdopen(fd, "w") as f:
        f.write("".join(f"{split}/{n}\n" for n in want))
    print(f"  streaming {cfg}/{split}.tar.gz for {len(want)} member(s)…", flush=True)
    curl = subprocess.Popen(["curl", "-sSL", f"{BASE}/{cfg}/audio/{split}.tar.gz"],
                            stdout=subprocess.PIPE)
    try:
        subprocess.run(["tar", "-xzf", "-", "-C", str(dest), "-T", listfile],
                       stdin=curl.stdout, check=True)
    finally:
        if curl.stdout:
            curl.stdout.close()
        curl.wait()
        os.unlink(listfile)


def to_wav(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                    "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)], check=True)
    with wave.open(str(dst)) as w:
        return w.getnframes() / float(w.getframerate())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="utterances per language")
    ap.add_argument("--split", default="test")
    ap.add_argument("--langs", default="en,fr")
    ap.add_argument("--out", default=str(ROOT / "samples" / "eval-bank"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    samples = []
    for lang in a.langs.split(","):
        cfg = LANGS.get(lang)
        if not cfg:
            raise SystemExit(f"no FLEURS config for {lang!r}")
        recs = load_records(cfg, a.split)[: a.n]
        tot = sum(r["duration_sec"] for r in recs)
        print(f"{lang} ({cfg}/{a.split}): {len(recs)} utterances, "
              f"{tot/60:.1f} min, median {sorted(r['duration_sec'] for r in recs)[len(recs)//2]:.1f} s")
        if a.dry_run:
            continue
        tmp = CACHE / cfg
        extract(cfg, a.split, [r["source_file"] for r in recs], tmp)
        for r in recs:
            dst = out / lang / f"fleurs_{r['fleurs_id']}.wav"
            if not dst.exists():
                dur = to_wav(tmp / a.split / r["source_file"], dst)
            else:
                with wave.open(str(dst)) as w:
                    dur = w.getnframes() / float(w.getframerate())
            samples.append({"file": f"{lang}/fleurs_{r['fleurs_id']}.wav", "lang": lang,
                            "fleurs_id": r["fleurs_id"], "duration_sec": round(dur, 3),
                            "text": r["text"], "text_norm": r["text_norm"],
                            "gender": r["gender"], "split": a.split})
    if a.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    man = {"source": "FLEURS (google/fleurs) — Conneau et al., CC-BY 4.0",
           "url": "https://huggingface.co/datasets/google/fleurs",
           "note": f"{a.split} split, untouched by any tuning in this repo; "
                   "the dataset transcription is the ONLY oracle",
           "samples": samples}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"wrote {out/'manifest.json'}: {len(samples)} utterances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
