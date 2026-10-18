#!/usr/bin/env python3
"""FINAL-IT: the one-shot Italian final test set (pre-registered 2026-10-18; selection is model-free).

    python3 prepare_final_it.py --out /root/ft/final-it --audio /root/disk/final-it [--n 400] \
        --exclude-texts "/root/ft/manifests/*.json,/root/ft/mix/it/*.json,/root/ft/mix/it-yg*/subsets/*.json"

Six sub-banks, each from a split that no run of ours has trained on, selected on or scored with:
  mls     MLS-it dev (facebook/multilingual_librispeech italian/dev); speakers asserted absent from train + test
  cv      Common Voice 17 it validation (fixie-ai/common_voice_17_0); client_ids in train / test dropped
  vp      VoxPopuli it validation (facebook/voxpopuli); speaker_ids in train / test dropped
  fleurs  FLEURS it_it dev (google/fleurs); sentence ids asserted absent from train / test; one reading per
          sentence (FLEURS has no speaker id)
  tedx    Multilingual TEDx it TRAIN shards 0-1 (deepdml/mtedx): talks absent from the mTEDx test / valid talks
          (= the clean-it dev bank) AND from every Granary-it original_audio_id (espnet/yodas-granary data/it*)
  italic  ITALIC hard_speaker VALIDATION (Zenodo 8040649): speakers asserted absent from hard_speaker test (dev)
Common rules (identical for every bank, fixed before any clip is seen):
  - 1.0 <= duration <= 20 s; no digit and none of []()&<> in the text; >= 2 words after the scorer's norm()
    (>= 3 for tedx); the curly apostrophe -> ';
  - natural ending: whole read / dictated utterances end naturally; tedx segments must end with . ? ! or ...;
  - dedup: normalised text unique inside FINAL-IT and absent from every --exclude-texts manifest (training pools
    and dev banks); decoded-sample duplicates are checked afterwards with pool_sanity.py --xhash;
  - order: candidates sorted by sha1("<bank>:<id>"), then taken round-robin over speakers / talks (max diversity),
    at most --n clips per bank (tedx at most --tedx-per-talk per talk).
No ASR model, teacher, student or WER is involved; the only signal measured on the audio is an energy endpoint
(20 ms frames, last frame within 35 dB of the clip peak) -> `trail_s` (trailing non-speech), recorded, never filtered.
Outputs ($OUT): final_<bank>.json, final_all.json, the EOF variant final_trim_<bank>.json (same clips cut at the
energy endpoint + 50 ms: the finalisation / flush probe), stats.md, SHA256SUMS (manifests + every wav).
Licences: mTEDx CC BY-NC-ND 4.0 (private evaluation only, never redistribute), ITALIC CC BY 4.0, MLS / FLEURS
CC BY 4.0, CV / VP CC0. Needs /root/.hf_token (Granary listing).
"""
import argparse
import collections
import csv
import glob
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import unicodedata
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HF = "https://huggingface.co/datasets"
BAD = re.compile(r"\d|[\[\]()&<>]")


def norm(t):   # = two_stage.norm (the scorer)
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


def sha1(x):
    return hashlib.sha1(str(x).encode()).hexdigest()


def fetch(url, dst, tries=20):
    head = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    size = [int(x) for x in re.findall(r"(?im)^content-length:\s*(\d+)", head)]
    want = size[-1] if size else None
    for _ in range(tries):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if have and (want is None or have >= want):
            return dst
        subprocess.run(["curl", "-fsSL", "--retry", "5", "-C", "-", "-o", dst, url])
    raise RuntimeError(f"download incomplete: {url}")


class Hub:
    def __init__(self):
        from huggingface_hub import HfFileSystem
        self.tok = open("/root/.hf_token").read().strip()
        self.fs = HfFileSystem(token=self.tok)

    def tree(self, repo, p):
        req = urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}/tree/main/{p}",
                                     headers={"Authorization": f"Bearer {self.tok}"})
        return json.load(urllib.request.urlopen(req, timeout=120))

    def column(self, repo, path, cols):   # column projection: only these column chunks are read
        import pyarrow.parquet as pq
        with self.fs.open(f"datasets/{repo}/{path}") as f:
            return pq.ParquetFile(f).read(columns=cols).to_pylist()

    def columns(self, repo, paths, cols, workers=16):
        with ThreadPoolExecutor(workers) as ex:
            return [r for rs in ex.map(lambda p: self.column(repo, p, cols), paths) for r in rs]

    def files(self, repo, d, prefix):
        return sorted(e["path"] for e in self.tree(repo, d) if e["type"] == "file" and
                      e["path"].rsplit("/", 1)[-1].startswith(prefix) and e["path"].endswith(".parquet"))


def to16k(x, sr):
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        import torch
        import torchaudio.functional as F
        x = F.resample(torch.from_numpy(x.astype(np.float32)), sr, 16000).numpy()
    return x.astype(np.float32)


def endpoint(x):
    """energy endpoint: end (s) of the last 20 ms frame within 35 dB of the clip's loudest frame."""
    n = 320
    f = x[: len(x) // n * n].reshape(-1, n)
    if not len(f):
        return len(x) / 16000
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    on = np.nonzero(db > db.max() - 35)[0]
    return (on[-1] + 1) * n / 16000 if len(on) else len(x) / 16000


class Bank:
    """collect candidates, then select round-robin over groups in sha1 order; audio is decoded only when kept."""
    def __init__(self, name, a, seen_texts):
        self.name, self.a, self.seen, self.cand, self.drop = name, a, seen_texts, [], collections.Counter()

    def offer(self, uid, group, text, dur, load, min_words=2, ending=None):
        text = text.replace("’", "'").strip()
        nt = norm(text)
        if not (1.0 <= dur <= 20.0):
            self.drop["duration"] += 1
        elif BAD.search(text):
            self.drop["digit/markup"] += 1
        elif len(nt.split()) < min_words:
            self.drop["short"] += 1
        elif ending is not None and not ending(text):
            self.drop["no natural ending"] += 1
        elif nt in self.seen["exclude"]:
            self.drop["text in train/dev"] += 1
        else:
            self.cand.append((sha1(f"{self.name}:{uid}"), uid, str(group), text, nt, load))

    def select(self, n, per_group=10 ** 9):
        by = collections.defaultdict(list)
        for c in sorted(self.cand):
            by[c[2]].append(c)
        order, rows, used = sorted(by, key=lambda g: by[g][0][0]), [], collections.Counter()
        while len(rows) < n and any(by[g] for g in order):
            for g in order:
                while by[g] and len(rows) < n:
                    _, uid, grp, text, nt, load = by[g].pop(0)
                    if nt in self.seen["final"] or used[grp] >= per_group:
                        self.drop["dup text in FINAL" if nt in self.seen["final"] else "group cap"] += 1
                        continue
                    x = load()
                    if x is None or not (1.0 <= len(x) / 16000 <= 20.0):
                        self.drop["audio"] += 1
                        continue
                    self.seen["final"].add(nt); used[grp] += 1
                    rows.append(self.write(uid, grp, text, x))
                    break
        print(f"== {self.name}: {len(rows)} clips from {len(self.cand)} candidates, {len(used)} groups; dropped "
              f"{dict(self.drop)}", flush=True)
        return rows

    def write(self, uid, grp, text, x):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", uid)
        p = Path(self.a.audio) / self.name / f"{safe}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, x, 16000, subtype="PCM_16")
        end = endpoint(x)
        cut = x[: min(len(x), int((end + 0.05) * 16000))]
        pt = Path(self.a.audio) / f"{self.name}_trim" / f"{safe}.wav"
        pt.parent.mkdir(parents=True, exist_ok=True)
        sf.write(pt, cut, 16000, subtype="PCM_16")
        return {"audio_filepath": str(p), "duration": round(len(x) / 16000, 3), "text": text, "bank": self.name,
                "group": grp, "utt_id": uid, "trail_s": round(len(x) / 16000 - end, 3),
                "trim_filepath": str(pt), "trim_duration": round(len(cut) / 16000, 3)}


def parquet_audio(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def decode(b):
    x, sr = sf.read(io.BytesIO(b), dtype="float32")
    return to16k(x, sr)


def mls(a, hub, seen):
    R = "facebook/multilingual_librispeech"
    spk = {str(r["speaker_id"]) for r in hub.columns(R, hub.files(R, "italian", "train") + hub.files(R, "italian", "test"),
                                                     ["speaker_id"])}
    rows = parquet_audio(fetch(f"{HF}/{R}/resolve/main/italian/dev-00000-of-00001.parquet", f"{a.tmp}/mls_dev.parquet"))
    leak = {str(r["speaker_id"]) for r in rows} & spk
    assert not leak, f"MLS dev speakers in train/test: {leak}"
    b = Bank("mls", a, seen)
    for r in rows:
        b.offer(str(r["id"]), r["speaker_id"], r["transcript"], float(r["audio_duration"]),
                lambda r=r: decode(r["audio"]["bytes"]))
    return b.select(a.n)


def cv(a, hub, seen):
    R = "fixie-ai/common_voice_17_0"
    ex = {r["client_id"] for r in hub.columns(R, hub.files(R, "it", "train") + hub.files(R, "it", "test"), ["client_id"])}
    b = Bank("cv", a, seen)
    for i, p in enumerate(hub.files(R, "it", "validation")):
        for r in parquet_audio(fetch(f"{HF}/{R}/resolve/main/{p}", f"{a.tmp}/cv_val_{i}.parquet")):
            if r["client_id"] in ex:
                b.drop["speaker in train/test"] += 1
                continue
            ab = r["audio"]["bytes"]   # no duration column: read from the container header, decoded only when kept
            b.offer(Path(r["path"]).stem, r["client_id"], r["sentence"], sf.info(io.BytesIO(ab)).duration,
                    lambda ab=ab: decode(ab))
    return b.select(a.n)


def vp(a, hub, seen):
    R = "facebook/voxpopuli"
    ex = {str(r["speaker_id"]) for r in hub.columns(R, hub.files(R, "it", "train") + hub.files(R, "it", "test"), ["speaker_id"])}
    rows = parquet_audio(fetch(f"{HF}/{R}/resolve/main/it/validation-00000-of-00001.parquet", f"{a.tmp}/vp_val.parquet"))
    b = Bank("vp", a, seen)
    for r in rows:
        if str(r["speaker_id"]) in ex:
            b.drop["speaker in train/test"] += 1
            continue
        ab = r["audio"]["bytes"]
        b.offer(str(r["audio_id"]), r["speaker_id"], r["normalized_text"], sf.info(io.BytesIO(ab)).duration,
                lambda ab=ab: decode(ab))
    return b.select(a.n)


def fleurs(a, hub, seen):
    base = f"{HF}/google/fleurs/resolve/main/data/it_it"

    def tsv(split):
        p = fetch(f"{base}/{split}.tsv", f"{a.tmp}/fleurs_{split}.tsv")
        return [r for r in csv.reader(open(p, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 7]
    other = {int(r[0]) for s in ("train", "test") for r in tsv(s)}
    dev = tsv("dev")
    leak = {int(r[0]) for r in dev} & other
    assert not leak, f"FLEURS dev sentence ids in train/test: {sorted(leak)[:5]}"
    audio = {}
    with urllib.request.urlopen(f"{base}/audio/dev.tar.gz", timeout=300) as resp, tarfile.open(fileobj=resp, mode="r|gz") as t:
        for m in t:
            if m.isfile():
                audio[m.name.rsplit("/", 1)[-1]] = t.extractfile(m).read()
    b = Bank("fleurs", a, seen)
    for r in dev:
        if r[1] in audio:
            b.offer(r[1].rsplit(".", 1)[0], r[0], r[2], int(r[5]) / 16000, lambda f=r[1]: decode(audio[f]))
    return b.select(a.n, per_group=1)


def granary_ids(hub):
    R = "espnet/yodas-granary"
    dirs = [d["path"] for d in hub.tree(R, "data") if d["type"] == "directory" and re.match(r"data/it\d", d["path"])]
    paths = []
    for d in dirs:
        for kind in ("asr_only", "ast"):
            try:
                paths += [e["path"] for e in hub.tree(R, f"{d}/{kind}") if e["path"].endswith(".parquet")]
            except Exception:  # noqa: BLE001  (a dir may have only one kind)
                pass
    ids = {str(r["original_audio_id"]) for r in hub.columns(R, paths, ["original_audio_id"], workers=24)}
    toks = {t for i in ids for t in re.findall(r"[A-Za-z0-9_-]{11}", i)}
    print(f"== granary-it: {len(dirs)} dirs, {len(paths)} shards, {len(ids)} videos", flush=True)
    return ids | toks


def tedx(a, hub, seen):
    R = "deepdml/mtedx"
    dev_talks = {r["talk_id"] for p in ("it/test-00000-of-00001.parquet", "it/valid-00000-of-00001.parquet")
                 for r in hub.column(R, p, ["talk_id"])}
    gran = granary_ids(hub)
    b = Bank("tedx", a, seen)
    for i in (0, 1):
        p = f"it/train-{i:05d}-of-00030.parquet"
        for r in parquet_audio(fetch(f"{HF}/{R}/resolve/main/{p}", f"{a.tmp}/mtedx_train_{i}.parquet")):
            if r["talk_id"] in dev_talks:
                b.drop["dev talk"] += 1
                continue
            if r["talk_id"] in gran:
                b.drop["talk in Granary-it"] += 1
                continue
            b.offer(r["id"], r["talk_id"], r["transcript"], float(r["duration"]),
                    lambda r=r: decode(r["audio"]["bytes"]), min_words=3,
                    ending=lambda t: bool(re.search(r"([.?!]|\.\.\.|…)[\"'»”]?$", t)))
    return b.select(a.n, per_group=a.tedx_per_talk)


def italic(a, hub, seen):
    z = zipfile.ZipFile(fetch("https://zenodo.org/records/8040649/files/italic.zip", f"{a.tmp}/italic.zip"))
    js = lambda n: [json.loads(l) for l in z.read(f"zenodo_dataset/{n}").decode().splitlines() if l.strip()]  # noqa: E731
    names = [n.split("/")[-1] for n in z.namelist()]
    assert "hard_speaker_validation.json" in names, f"no hard_speaker_validation.json in ITALIC: {[n for n in names if n.endswith('.json')]}"
    val, test = js("hard_speaker_validation.json"), js("hard_speaker_test.json")
    leak = {r["speaker_id"] for r in val} & {r["speaker_id"] for r in test}
    assert not leak, f"ITALIC validation speakers in the dev test: {leak}"
    b = Bank("italic", a, seen)
    for r in val:
        b.offer(r["id"], r["speaker_id"], r["utt"], float(r.get("duration") or 5.0),
                lambda r=r: decode(z.read(f"zenodo_dataset/recordings/{r['id']}.wav")))
    return b.select(a.n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--tmp", default="/root/final-it-dl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tedx-per-talk", type=int, default=25)
    ap.add_argument("--exclude-texts", required=True, help="comma-separated globs of training / dev manifests")
    ap.add_argument("--banks", default="mls,cv,vp,fleurs,tedx,italic")
    a = ap.parse_args()
    for d in (a.out, a.audio, a.tmp):
        os.makedirs(d, exist_ok=True)
    ex, nman = set(), 0
    for g in a.exclude_texts.split(","):
        for p in sorted(glob.glob(g)):
            nman += 1
            for l in open(p):
                if l.strip():
                    ex.add(norm(json.loads(l).get("text", "")))
    print(f"== exclusion texts: {len(ex)} from {nman} manifests", flush=True)
    seen, hub, allr = {"exclude": ex, "final": set()}, Hub(), []
    fns = {"mls": mls, "cv": cv, "vp": vp, "fleurs": fleurs, "tedx": tedx, "italic": italic}
    for name in a.banks.split(","):
        rows = fns[name](a, hub, seen)
        allr += rows
        for tag, key, dk in (("final", "audio_filepath", "duration"), ("final_trim", "trim_filepath", "trim_duration")):
            with open(f"{a.out}/{tag}_{name}.json", "w") as f:
                for r in rows:
                    f.write(json.dumps({**r, "audio_filepath": r[key], "duration": r[dk]}, ensure_ascii=False) + "\n")
    with open(f"{a.out}/final_all.json", "w") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{a.out}/stats.md", "w") as f:
        f.write("| bank | clips | h | groups | dur p10/p50/p90 s | trail_s p10/p50/p90 | trail < 0.2 s | words p50 |\n|---|---|---|---|---|---|---|---|\n")
        for name in a.banks.split(","):
            rs = [r for r in allr if r["bank"] == name]
            if not rs:
                continue
            d, t = np.array([r["duration"] for r in rs]), np.array([r["trail_s"] for r in rs])
            w = np.median([len(norm(r["text"]).split()) for r in rs])
            f.write(f"| {name} | {len(rs)} | {d.sum() / 3600:.2f} | {len({r['group'] for r in rs})} | "
                    f"{'/'.join(f'{v:.1f}' for v in np.percentile(d, [10, 50, 90]))} | "
                    f"{'/'.join(f'{v:.2f}' for v in np.percentile(t, [10, 50, 90]))} | {(t < 0.2).mean() * 100:.0f} % | {w:.0f} |\n")
    print(open(f"{a.out}/stats.md").read())
    wavs = sorted(str(p) for p in Path(a.audio).rglob("*.wav"))
    with open(f"{a.out}/SHA256SUMS", "w") as f:
        for p in sorted(glob.glob(f"{a.out}/final*.json")) + wavs:
            f.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {p}\n")
    print(f"== FINAL-IT-DONE {len(allr)} clips, {len(wavs)} wavs hashed")


if __name__ == "__main__":
    main()
