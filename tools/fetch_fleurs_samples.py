#!/usr/bin/env python3
"""Build samples/ from the FLEURS dataset (google/fleurs, CC-BY 4.0).

PARALLEL sentences across languages (same FLEURS id = same sentence, translated):
the English one is the reference for scoring Canary's speech translation.
I wav (16 kHz mono PCM16) vengono COMMITTATI nel repo: pochi e leggeri.

Usage: uv run python fetch_fleurs_samples.py          # from the tools/ dir
Riscarica i tsv, estrae in streaming SOLO i wav scelti dai dev.tar.gz
(--fast-read), scrive samples/<lang>/fleurs_<id>.wav + manifest.json.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
CACHE = Path("/tmp/fleurs")

# mynah language -> FLEURS config. The first 5 cover Canary (translation) and the
# historical fixtures; the others exercise Nemotron's 40 locales and v3's 25 EU
# languages (alphabets included: Cyrillic, Japanese).
LANGS = {"it": "it_it", "en": "en_us", "de": "de_de", "es": "es_419", "fr": "fr_fr",
         "pt": "pt_br", "nl": "nl_nl", "pl": "pl_pl", "ru": "ru_ru", "uk": "uk_ua",
         "ja": "ja_jp"}
# FLEURS ids of the chosen sentences (parallel across all languages, 7-13 s,
# contenuti distintivi: 1521 satellite, 1534 Timbuctù)
IDS = ["1521", "1534"]
# long EN clip (~90 s): dev sentences concatenated with 0.6 s pauses — an exact
# reference for silence segmentation, long timestamps and streaming
LONG_N = 8
LONG_PAUSE = 0.6
# LONG clips (WAV PCM16, committed: ~5 min EN + ~2 min DE ≈ 14 MB) for the long
# transcribe / long translate tests — lossless from the FLEURS sources, no ffmpeg
# dependency in the tests. de: only PARALLEL sentences (ids present in en too) so
# the translation reference is the concatenation of the en ones.
LONG_TARGETS = {"en": 300.0, "de": 120.0}


def tsv_rows(cfg: str) -> dict[str, dict]:
    path = CACHE / f"{cfg}.tsv"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", f"{BASE}/{cfg}/dev.tsv", "-o", str(path)], check=True)
    rows: dict[str, dict] = {}
    with open(path) as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) >= 7 and r[0] not in rows:
                rows[r[0]] = {"file": r[1], "raw": r[2], "dur": int(r[5]) / 16000.0}
    return rows


def extract(cfg: str, files: list[str]) -> None:
    outdir = CACHE / cfg
    if all((outdir / "dev" / f).exists() for f in files):
        return
    outdir.mkdir(parents=True, exist_ok=True)
    pats = [f"*/{f}" for f in files]
    curl = subprocess.Popen(["curl", "-sL", f"{BASE}/{cfg}/audio/dev.tar.gz"],
                            stdout=subprocess.PIPE)
    subprocess.run(["tar", "-xzf", "-", "-C", str(outdir), "--fast-read", *pats],
                   stdin=curl.stdout, check=True)
    curl.stdout.close()
    curl.wait()


def main() -> None:
    samples_dir = ROOT / "samples"
    manifest: dict = {
        "source": "FLEURS (google/fleurs) — Conneau et al., CC-BY 4.0",
        "url": "https://huggingface.co/datasets/google/fleurs",
        "note": "sentences parallel across languages: same id = same sentence; English "
                "is the reference for speech translation",
        "samples": [],
    }
    texts = {lang: tsv_rows(cfg) for lang, cfg in LANGS.items()}
    for lang, cfg in LANGS.items():
        rows = texts[lang]
        missing = [i for i in IDS if i not in rows]
        if missing:
            sys.exit(f"{cfg}: ids missing from the dev set: {missing}")
        extract(cfg, [rows[i]["file"] for i in IDS])
        (samples_dir / lang).mkdir(parents=True, exist_ok=True)
        for i in IDS:
            src = CACHE / cfg / "dev" / rows[i]["file"]
            dst = samples_dir / lang / f"fleurs_{i}.wav"
            # FLEURS ships float32: rewritten as PCM16 (half the size,
            # universal support — the runtime reads WAV PCM16)
            import soundfile as sf
            audio, sr = sf.read(src, dtype="float32")
            sf.write(dst, audio, sr, subtype="PCM_16")
            manifest["samples"].append({
                "file": f"{lang}/fleurs_{i}.wav",
                "lang": lang,
                "fleurs_id": int(i),
                "duration_sec": round(rows[i]["dur"], 1),
                "text": rows[i]["raw"],
                "en_ref": texts["en"][i]["raw"],
            })
    # long EN clip: the first LONG_N dev sentences (8-14 s each) concatenated
    import numpy as np
    import soundfile as sf
    en = texts["en"]
    long_ids = [i for i in sorted(en) if 8.0 <= en[i]["dur"] <= 14.0][:LONG_N]
    extract(LANGS["en"], [en[i]["file"] for i in long_ids])
    pieces, refs = [], []
    for i in long_ids:
        audio, sr = sf.read(CACHE / LANGS["en"] / "dev" / en[i]["file"], dtype="float32")
        pieces += [audio, np.zeros(int(LONG_PAUSE * sr), dtype="float32")]
        refs.append(en[i]["raw"])
    long_audio = np.concatenate(pieces)
    sf.write(samples_dir / "en" / "fleurs_long.wav", long_audio, 16000, subtype="PCM_16")
    manifest["samples"].append({
        "file": "en/fleurs_long.wav",
        "lang": "en",
        "long": True,
        "duration_sec": round(len(long_audio) / 16000.0, 1),
        "text": " ".join(refs),
    })

    # clip lunghi WAV
    (samples_dir / "long").mkdir(exist_ok=True)
    for lang, target in LONG_TARGETS.items():
        rows = texts[lang]
        pool = [i for i in sorted(rows) if 6.0 <= rows[i]["dur"] <= 16.0
                and (lang == "en" or i in texts["en"])]
        ids, tot = [], 0.0
        for i in pool:
            if tot >= target:
                break
            ids.append(i)
            tot += rows[i]["dur"] + LONG_PAUSE
        extract(LANGS[lang], [rows[i]["file"] for i in ids])
        pieces, refs, en_refs = [], [], []
        for i in ids:
            audio, sr = sf.read(CACHE / LANGS[lang] / "dev" / rows[i]["file"],
                                dtype="float32")
            pieces += [audio, np.zeros(int(LONG_PAUSE * sr), dtype="float32")]
            refs.append(rows[i]["raw"])
            if lang != "en":
                en_refs.append(texts["en"][i]["raw"])
        dst = samples_dir / "long" / f"{lang}_long.wav"
        sf.write(dst, np.concatenate(pieces), 16000, subtype="PCM_16")
        entry = {"file": f"long/{lang}_long.wav", "lang": lang, "long": True,
                 "duration_sec": round(tot, 1), "text": " ".join(refs)}
        if en_refs:
            entry["en_ref"] = " ".join(en_refs)
        manifest["samples"].append(entry)

    (samples_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    total = sum(s["duration_sec"] for s in manifest["samples"])
    print(f"OK samples/: {len(manifest['samples'])} clip, {total:.0f}s totali")


if __name__ == "__main__":
    main()
