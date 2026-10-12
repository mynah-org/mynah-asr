#!/usr/bin/env python3
"""clean-it: a small Italian HOLDOUT bank, source/speaker/video-disjoint from every training mix and not named
in the teacher's training data (MLS, CommonVoice, FLEURS, VoxPopuli/Europarl, Granary are; these are not).

    python3 prepare_clean_it.py --out /root/ft/manifests --audio /dev/shm/clean-it [--tedx-per-talk 60]
        [--italic-per-speaker 70]

Two sub-banks, reported separately (very different domains):
  eval_tedx_it.json    Multilingual TEDx it (deepdml/mtedx, config it), test + valid talks: subtitles of
                       talks; 1-20 s, no digit (the scorer deletes digits), none of []()&<> (speaker tags,
                       "(Applausi)", editorial insertions), >= 3 words; at most N segments per talk, ordered
                       by sha1(id). The 16 talk ids (YouTube ids) are absent from the Granary-it yodas/ytc
                       manifests (agent L, 2026-10-12). Licence CC BY-NC-ND 4.0: private evaluation only, never
                       redistribute the segments or manifests.
  eval_italic_it.json  ITALIC hard_speaker test (Zenodo 8040649, CC BY 4.0): read voice-assistant commands by
                       7 held-out volunteers; >= 2 words, no digit, identical texts deduplicated, at most N
                       per speaker by sha1(id).
Text: the curly apostrophe is mapped to ' (the scorer keeps ' and would split l'uomo otherwise); otherwise raw.
Audio: 16 kHz mono PCM16 wav. Downloads happen on the box; the ITALIC zip (1.3 GB) is deleted afterwards.
"""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

TEDX = "https://huggingface.co/datasets/deepdml/mtedx/resolve/main/it/{split}-00000-of-00001.parquet"
ITALIC = "https://zenodo.org/records/8040649/files/italic.zip"


def fetch(url, dst):
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        subprocess.run(["curl", "-fsSL", "--retry", "5", "-C", "-", "-o", dst, url], check=True)
    return dst


def sha(x):
    return hashlib.sha1(str(x).encode()).hexdigest()


def write(audio_dir, name, x, sr):
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        import torch
        import torchaudio.functional as F
        x = F.resample(torch.from_numpy(x.astype(np.float32)), sr, 16000).numpy()
    p = Path(audio_dir) / f"{name}.wav"
    sf.write(p, x, 16000, subtype="PCM_16")
    return str(p), len(x) / 16000


def tedx(a, tmp):
    import pyarrow.parquet as pq
    rows = []
    for split in ("test", "valid"):
        t = pq.read_table(fetch(TEDX.format(split=split), f"{tmp}/mtedx_{split}.parquet")).to_pylist()
        for r in t:
            txt = r["transcript"].replace("’", "'").strip()
            if not (1.0 <= r["duration"] <= 20.0) or re.search(r"\d|[\[\]()&<>]", txt) or len(txt.split()) < 3:
                continue
            rows.append((r, txt, split))
    keep, per = [], {}
    for r, txt, split in sorted(rows, key=lambda z: sha(z[0]["id"])):
        if per.get(r["talk_id"], 0) >= a.tedx_per_talk:
            continue
        per[r["talk_id"]] = per.get(r["talk_id"], 0) + 1
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        path, dur = write(a.audio, f"tedx_{r['id']}", x, sr)
        keep.append({"audio_filepath": path, "duration": round(dur, 3), "text": txt, "corpus": "mtedx",
                     "split": split, "talk_id": r["talk_id"], "utt_id": r["id"]})
    return keep


def italic(a, tmp):
    z = zipfile.ZipFile(fetch(ITALIC, f"{tmp}/italic.zip"))
    meta = [json.loads(l) for l in z.read("zenodo_dataset/hard_speaker_test.json").decode().splitlines() if l.strip()]
    seen, per, keep = set(), {}, []
    for r in sorted(meta, key=lambda m: sha(m["id"])):
        txt = r["utt"].strip()
        if len(txt.split()) < 2 or re.search(r"\d", txt) or txt in seen or per.get(r["speaker_id"], 0) >= a.italic_per_speaker:
            continue
        seen.add(txt); per[r["speaker_id"]] = per.get(r["speaker_id"], 0) + 1
        x, sr = sf.read(io.BytesIO(z.read(f"zenodo_dataset/recordings/{r['id']}.wav")), dtype="float32")
        path, dur = write(a.audio, f"italic_{r['id']}", x, sr)
        keep.append({"audio_filepath": path, "duration": round(dur, 3), "text": txt, "corpus": "italic",
                     "speaker": r["speaker_id"], "utt_id": r["id"]})
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--tmp", default="/root/clean-it-dl")
    ap.add_argument("--tedx-per-talk", type=int, default=60)
    ap.add_argument("--italic-per-speaker", type=int, default=70)
    a = ap.parse_args()
    for d in (a.out, a.audio, a.tmp):
        os.makedirs(d, exist_ok=True)
    for name, fn in (("eval_tedx_it", tedx), ("eval_italic_it", italic)):
        rs = fn(a, a.tmp)
        with open(f"{a.out}/{name}.json", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        groups = len({r.get("talk_id") or r.get("speaker") for r in rs})
        print(f"== {name}: {len(rs)} utts {sum(r['duration'] for r in rs) / 3600:.2f} h, {groups} talks/speakers")
    for f in Path(a.tmp).glob("*"):
        f.unlink()
    print("== CLEAN-IT-DONE")


if __name__ == "__main__":
    main()
