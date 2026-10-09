#!/usr/bin/env python3
"""Multilingual replay data for the Canary 180M Italian fine-tune.

Long Italian-only training erodes the base languages (FLEURS en WER 6.9 -> 40
after 3,120 steps with a frozen encoder). This step builds a small replay pool
of the model's original languages so train_it.py --replay-ratio can mix
them back in, plus per-language forgetting probes.

Source: FLEURS (google/fleurs, CC-BY-4.0), same tsv + streamed-tarball path as
prepare_it.py (fleurs_tsv / fleurs_stream; tarballs are streamed, never stored).
  train  en_us, de_de, es_419, fr_fr: capped at --hours-per-lang (env REPLAY_H,
         default 2 h) each; 1-20 s clips; deterministic (seed) order that takes
         one reading of every sentence before a second one (more sentences per
         hour). Written to $FT/audio/replay_<lang>/.
  test   de_de, es_419, fr_fr: the first --eval-clips (100) clips sorted by
         sentence id, like the existing en_sanity probe. $FT/audio/fleurs_<lang>/.
Outputs ($FT/manifests): replay_train.json (all four languages; canary2 rows
with source_lang = target_lang = the row's language, pnc=yes as for the cased
FLEURS it rows), eval_fleurs_de.json, eval_fleurs_es.json, eval_fleurs_fr.json,
and $FT/replay_hours.json (exact seconds from the written wavs).

Disk: ~115 MB per audio hour -> ~0.9 GB for 4 x 2 h + ~60 MB of eval clips.
Network: each train tarball is streamed until all its planned clips are found
(usually most of the tarball, ~1.5-3 GB per language, nothing kept on disk).

  $PY data_replay.py --dry-run      # tsv metadata + plan only (prints hours)
  $PY data_replay.py --synthetic    # no network: fabricated tsv rows, plan only
  REPLAY_H=2 $PY data_replay.py     # plan + audio + manifests (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "common")]
import ftlib  # noqa: E402
import prepare_it as data_it  # noqa: E402

# FLEURS config -> canary2 language code (all four are sub-tokenizers of the base model)
LANGS = {"en_us": "en", "de_de": "de", "es_419": "es", "fr_fr": "fr"}
EVAL_LANGS = ("de_de", "es_419", "fr_fr")  # en already has manifests/en_sanity.json


def synthetic_rows(cfg, split, n=1500):
    rng = random.Random(f"{cfg}-{split}")
    return sorted(({"sid": i // 3, "file": f"{cfg}_{split}_{i}.wav", "raw": f"Sentence {i // 3}.", "norm": "",
                    "duration": rng.uniform(0.5, 25.0), "gender": "female"} for i in range(n)),
                  key=lambda r: (r["sid"], r["file"]))


def plan_train(rows, cap_s, seed, min_dur, max_dur):
    """Deterministic: shuffle, then order by (reading index of the sentence, shuffled position)."""
    rows = [r for r in rows if min_dur <= r["duration"] <= max_dur]
    random.Random(seed).shuffle(rows)
    seen = {}
    keyed = []
    for i, r in enumerate(rows):
        k = seen.get(r["sid"], 0)
        seen[r["sid"]] = k + 1
        keyed.append((k, i, r))
    out, acc = [], 0.0
    for _, _, r in sorted(keyed, key=lambda t: (t[0], t[1])):
        if acc + r["duration"] > cap_s:
            continue
        out.append(r)
        acc += r["duration"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours-per-lang", type=float, default=float(os.environ.get("REPLAY_H", "2") or 2))
    ap.add_argument("--eval-clips", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="fabricated tsv rows, no network; implies --dry-run")
    a = ap.parse_args()

    if a.synthetic:
        a.dry_run = True
    FT = ftlib.FT
    cap_s = a.hours_per_lang * 3600
    plan = {"hours_per_lang": a.hours_per_lang, "seed": a.seed, "train": {}, "eval": {}}
    for cfg, lang in LANGS.items():
        tr = synthetic_rows(cfg, "train") if a.synthetic else data_it.fleurs_tsv(cfg, "train")
        plan["train"][cfg] = plan_train(tr, cap_s, a.seed, a.min_dur, a.max_dur)
        if cfg in EVAL_LANGS:
            te = synthetic_rows(cfg, "test", 400) if a.synthetic else data_it.fleurs_tsv(cfg, "test")
            plan["eval"][cfg] = te[: a.eval_clips]
    for cfg, rows in plan["train"].items():
        n_sid = len({r["sid"] for r in rows})
        print(f"  replay train {cfg:7s} {len(rows):5d} utts {sum(r['duration'] for r in rows) / 3600:6.3f} h "
              f"({n_sid} sentences)")
    for cfg, rows in plan["eval"].items():
        print(f"  eval   test  {cfg:7s} {len(rows):5d} utts {sum(r['duration'] for r in rows) / 60:6.1f} min")
    if a.dry_run:
        return 0

    t0 = time.time()
    for cfg, lang in LANGS.items():
        n = data_it.fleurs_stream(cfg, "train", [r["file"] for r in plan["train"][cfg]], f"replay_{lang}")
        print(f"  {cfg} train: wrote {n} ({time.time() - t0:.0f}s)", flush=True)
    for cfg in EVAL_LANGS:
        n = data_it.fleurs_stream(cfg, "test", [r["file"] for r in plan["eval"][cfg]], f"fleurs_{LANGS[cfg]}")
        print(f"  {cfg} test: wrote {n} ({time.time() - t0:.0f}s)", flush=True)

    import soundfile as sf

    def rows_for(units, cfg, src):
        lang, rows, miss = LANGS[cfg], [], 0
        for r in units:
            uid = r["file"].rsplit(".", 1)[0]
            p = data_it.wav_path(src, uid)
            if not p.exists():
                miss += 1
                continue
            audio, sr = sf.read(str(p), dtype="float32")
            # FLEURS raw text is cased + punctuated -> pnc=yes (same as the FLEURS it rows)
            rows.append(ftlib.canary_row(p, len(audio) / sr, r["raw"], lang, "yes", corpus=f"fleurs_{lang}",
                                         utt_id=uid, peak_dbfs=ftlib.peak_dbfs(audio)))
        return rows, miss

    man = FT / "manifests"
    hours = {"note": "seconds summed from the written 16 kHz wav files", "hours_per_lang_cap": a.hours_per_lang}
    train = []
    for cfg, units in plan["train"].items():
        rows, miss = rows_for(units, cfg, f"replay_{LANGS[cfg]}")
        train += rows
        s = sum(r["duration"] for r in rows)
        hours[f"replay_{LANGS[cfg]}"] = {"n_utts": len(rows), "seconds": round(s, 3), "hours": round(s / 3600, 4),
                                         "missing_audio": miss}
    ftlib.write_manifest(man / "replay_train.json", train)
    s = sum(r["duration"] for r in train)
    hours["replay_train"] = {"n_utts": len(train), "seconds": round(s, 3), "hours": round(s / 3600, 4)}
    for cfg in EVAL_LANGS:
        lang = LANGS[cfg]
        rows, miss = rows_for(plan["eval"][cfg], cfg, f"fleurs_{lang}")
        ftlib.write_manifest(man / f"eval_fleurs_{lang}.json", rows)
        hours[f"eval_fleurs_{lang}"] = {"n_utts": len(rows), "seconds": round(sum(r["duration"] for r in rows), 3),
                                        "missing_audio": miss}
    ftlib.write_json(FT / "replay_hours.json", hours)
    for k, v in hours.items():
        if isinstance(v, dict):
            print(f"  {k:18s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
