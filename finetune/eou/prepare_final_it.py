#!/usr/bin/env python3
"""FINAL-IT / FINAL-FR: the one-shot final test sets (IT pre-registered 2026-10-18; selection is model-free).

    python3 prepare_final_it.py --out /root/ft/final-it --audio /root/disk/final-it [--n 400] \
        --exclude-texts "/root/ft/manifests/*.json,/root/ft/mix/it/*.json,/root/ft/mix/it-yg*/subsets/*.json"
    python3 prepare_final_it.py --lang fr --out /root/ft/final-fr --audio /root/disk/final-fr \
        --exclude-texts "/root/ft/manifests/*fr*.json,/root/ft/mix/fr/*.json" --vp-exclude <our FR VP manifests>
    ... --plan-only: metadata only (column projection, no audio download / decode, nothing sealed): per-bank
        candidate counts, drops and the selection order -> $OUT/plan_<bank>.json (duration filter unchecked where the
        source has no duration column: cv, vp, smassive; FLEURS audio presence not checked)

--lang it (default) is the frozen FINAL-IT build, byte for byte (sources, paths, norm(), regexes unchanged).

FR-OOD1 (2026-10-20, a DEV bank, not FINAL): --banks tedxdev,smtrain --seed-final <FINAL-FR final_*.json> with the same
  rules: tedxdev = mTEDx fr test + valid talks (asserted absent from mTEDx train; Granary-fr ids dropped), smtrain =
  Speech-MASSIVE fr-FR train-* (FINAL's validation speakers dropped).
IT: six sub-banks, each from a split that no run of ours has trained on, selected on or scored with:
  mls     MLS-it dev (facebook/multilingual_librispeech italian/dev); speakers asserted absent from train + test
  cv      Common Voice 17 it validation (fixie-ai/common_voice_17_0); client_ids in train / test dropped
  vp      VoxPopuli it validation (facebook/voxpopuli); speaker_ids in train / test dropped (--vp-exclude: only the
          speakers of our VP pools / bank; VP's own splits share speakers)
  fleurs  FLEURS it_it dev (google/fleurs); sentence ids asserted absent from train / test; one reading per
          sentence (FLEURS has no speaker id)
  tedx    Multilingual TEDx it TRAIN shards 0-1 (deepdml/mtedx): talks absent from the mTEDx test / valid talks
          (= the clean-it dev bank) AND from every Granary-it original_audio_id (espnet/yodas-granary data/it*)
  italic  ITALIC hard_speaker VALIDATION (Zenodo 8040649): speakers asserted absent from hard_speaker test (dev)
FR (--lang fr, same rules; release doc 2026-10-19 sections F / I): mls = MLS-fr dev (french/), cv = CV17 fr validation,
  vp = VoxPopuli fr validation, fleurs = FLEURS fr_fr dev, tedx = mTEDx fr TRAIN shards 0-1 (of the listed shards, 41
  for fr) minus fr test / valid talks and every Granary-fr original_audio_id (data/fr\\d: fr000, fr100-fr103), and
  smassive = Speech-MASSIVE fr-FR VALIDATION (FBK-MT/Speech-MASSIVE; group = speaker): speakers of fr-FR train /
          train_115 (and of the gated FBK-MT/Speech-MASSIVE-test when readable) dropped, recordings not human-validated
          (is_validated false) dropped. The text norm is common/textnorm.norm(t, "fr") (oe / ae, typographic
          apostrophes); tedx natural ending also accepts the French spaced closing quote ("... . »").
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
CC BY 4.0, CV / VP CC0, Speech-MASSIVE CC BY-NC-SA 4.0 (private evaluation only, never redistribute).
Uses /root/.hf_token when present (anonymous Hub reads otherwise: every source is public; only the optional
Speech-MASSIVE-test speaker list is gated).
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
import sys
import tarfile
import time
import unicodedata
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HF = "https://huggingface.co/datasets"
BAD = re.compile(r"\d|[\[\]()&<>]")
# per-language sources; "it" = the frozen 2026-10-18 strings. hf = the language dir of CV / VP / mTEDx / Granary
SRC = {"it": {"mls": "italian", "hf": "it", "fleurs": "it_it", "banks": "mls,cv,vp,fleurs,tedx,italic", "tag": "FINAL-IT"},
       "fr": {"mls": "french", "hf": "fr", "fleurs": "fr_fr", "smassive": "fr-FR", "banks": "mls,cv,vp,fleurs,tedx,smassive",
              "tag": "FINAL-FR"}}
# tedx natural ending. fr: French typography puts a (no-break) space before the closing guillemet ("merci. »")
ENDING = {"it": r"([.?!]|\.\.\.|…)[\"'»”]?$", "fr": r"([.?!]|\.\.\.|…)(\s*[\"'»”])?$"}


def norm(t):   # = two_stage.norm (the scorer)
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


def normf(lang):
    """the text norm of a language: "it" = norm() above (== textnorm.norm(t, "it"), frozen), else common/textnorm."""
    if lang == "it":
        return norm
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    import textnorm
    return lambda t: textnorm.norm(t, lang)


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
        tf = "/root/.hf_token"   # optional: anonymous reads when absent
        self.tok = open(tf).read().strip() if os.path.exists(tf) else None
        self.fs = HfFileSystem(token=self.tok)
        self.cap = 0   # --hub-workers: cap on parallel column reads (0 = the frozen defaults 16 / 24)
        self.block = 0   # --hub-block-kb: read-ahead per range request (0 = the library default, several MB)

    def tree(self, repo, p):
        req = urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}/tree/main/{p}",
                                     headers={"Authorization": f"Bearer {self.tok}"} if self.tok else {})
        return json.load(urllib.request.urlopen(req, timeout=120))

    def column(self, repo, path, cols):   # column projection: only these column chunks are read
        import pyarrow.parquet as pq
        for k in range(6):   # HTTP 429 (rate limit): back off and retry; any other error is raised
            try:
                kw = {"block_size": self.block} if self.block else {}   # I/O only: same rows either way
                with self.fs.open(f"datasets/{repo}/{path}", **kw) as f:
                    return pq.ParquetFile(f).read(columns=cols).to_pylist()
            except Exception as e:  # noqa: BLE001
                if "429" not in str(e) or k == 5:
                    raise
                print(f"== HTTP 429 on {repo}/{path}: retry in {30 * 2 ** k} s", flush=True)
                time.sleep(30 * 2 ** k)

    def columns(self, repo, paths, cols, workers=16):
        workers = min(workers, self.cap) if self.cap else workers
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
        self.norm, self.plan, self.unchecked = normf(a.lang), getattr(a, "plan_only", False), 0

    def offer(self, uid, group, text, dur, load, min_words=2, ending=None):
        """dur None = unknown (--plan-only on a source without a duration column): the duration rule is skipped."""
        text = text.replace("’", "'").strip()
        nt = self.norm(text)
        if dur is not None and not (1.0 <= dur <= 20.0):
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
            self.unchecked += dur is None
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
                    if self.plan:   # metadata only: nothing is downloaded, decoded or written
                        self.seen["final"].add(nt); used[grp] += 1
                        rows.append({"utt_id": uid, "group": grp, "text": text, "bank": self.name})
                        break
                    x = load()
                    if x is None or not (1.0 <= len(x) / 16000 <= 20.0):
                        self.drop["audio"] += 1
                        continue
                    self.seen["final"].add(nt); used[grp] += 1
                    rows.append(self.write(uid, grp, text, x))
                    break
        print(f"== {self.name}{' (plan)' if self.plan else ''}: {len(rows)} clips from {len(self.cand)} candidates, "
              f"{len(used)} groups; dropped {dict(self.drop)}"
              + (f"; duration unchecked on {self.unchecked} candidates" if self.unchecked else ""), flush=True)
        return rows

    def write(self, uid, grp, text, x):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(uid))   # ITALIC ids are ints
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
    R, L = "facebook/multilingual_librispeech", SRC[a.lang]["mls"]
    spk = {str(r["speaker_id"]) for r in hub.columns(R, hub.files(R, L, "train") + hub.files(R, L, "test"),
                                                     ["speaker_id"])}
    dev = f"{L}/dev-00000-of-00001.parquet"
    rows = (hub.column(R, dev, ["id", "speaker_id", "transcript", "audio_duration"]) if a.plan_only else
            parquet_audio(fetch(f"{HF}/{R}/resolve/main/{dev}", f"{a.tmp}/mls_dev.parquet")))
    leak = {str(r["speaker_id"]) for r in rows} & spk
    assert not leak, f"MLS dev speakers in train/test: {leak}"
    b = Bank("mls", a, seen)
    for r in rows:
        b.offer(str(r["id"]), r["speaker_id"], r["transcript"], float(r["audio_duration"]),
                lambda r=r: decode(r["audio"]["bytes"]))
    return b.select(a.n)


def cv(a, hub, seen):
    R, L = "fixie-ai/common_voice_17_0", SRC[a.lang]["hf"]
    ex = {r["client_id"] for r in hub.columns(R, hub.files(R, L, "train") + hub.files(R, L, "test"), ["client_id"])}
    b = Bank("cv", a, seen)
    for i, p in enumerate(hub.files(R, L, "validation")):
        for r in (hub.column(R, p, ["client_id", "path", "sentence"]) if a.plan_only else
                  parquet_audio(fetch(f"{HF}/{R}/resolve/main/{p}", f"{a.tmp}/cv_val_{i}.parquet"))):
            if r["client_id"] in ex:
                b.drop["speaker in train/test"] += 1
                continue
            ab = None if a.plan_only else r["audio"]["bytes"]   # no duration column: read from the container header
            b.offer(Path(r["path"]).stem, r["client_id"], r["sentence"],         # (decoded only when kept)
                    None if ab is None else sf.info(io.BytesIO(ab)).duration, lambda ab=ab: decode(ab))
    return b.select(a.n)


def vp(a, hub, seen):
    R, L = "facebook/voxpopuli", SRC[a.lang]["hf"]
    if a.vp_exclude:   # amendment 2026-10-18: speakers of OUR VP pools / bank (VP splits are not speaker-disjoint)
        ex = {r["speaker"].split(":", 1)[-1] for p in a.vp_exclude.split(",") for r in map(json.loads, open(p)) if r.get("speaker")}
        print(f"== vp: {len(ex)} excluded speakers from our VP manifests", flush=True)
    else:
        ex = {str(r["speaker_id"]) for r in hub.columns(R, hub.files(R, L, "train") + hub.files(R, L, "test"), ["speaker_id"])}
    val = f"{L}/validation-00000-of-00001.parquet"
    rows = (hub.column(R, val, ["audio_id", "speaker_id", "normalized_text"]) if a.plan_only else
            parquet_audio(fetch(f"{HF}/{R}/resolve/main/{val}", f"{a.tmp}/vp_val.parquet")))
    b = Bank("vp", a, seen)
    for r in rows:
        if str(r["speaker_id"]) in ex:
            b.drop["speaker in train/test"] += 1
            continue
        ab = None if a.plan_only else r["audio"]["bytes"]
        b.offer(str(r["audio_id"]), r["speaker_id"], r["normalized_text"],
                None if ab is None else sf.info(io.BytesIO(ab)).duration, lambda ab=ab: decode(ab))
    return b.select(a.n)


def fleurs(a, hub, seen):
    base = f"{HF}/google/fleurs/resolve/main/data/{SRC[a.lang]['fleurs']}"

    def tsv(split):
        p = fetch(f"{base}/{split}.tsv", f"{a.tmp}/fleurs_{split}.tsv")
        return [r for r in csv.reader(open(p, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 7]
    other = {int(r[0]) for s in ("train", "test") for r in tsv(s)}
    dev = tsv("dev")
    leak = {int(r[0]) for r in dev} & other
    assert not leak, f"FLEURS dev sentence ids in train/test: {sorted(leak)[:5]}"
    audio = {}
    if not a.plan_only:   # plan: the audio tarball is not fetched (every dev row counted as present)
        with urllib.request.urlopen(f"{base}/audio/dev.tar.gz", timeout=300) as resp, tarfile.open(fileobj=resp, mode="r|gz") as t:
            for m in t:
                if m.isfile():
                    audio[m.name.rsplit("/", 1)[-1]] = t.extractfile(m).read()
    b = Bank("fleurs", a, seen)
    for r in dev:
        if a.plan_only or r[1] in audio:
            b.offer(r[1].rsplit(".", 1)[0], r[0], r[2], int(r[5]) / 16000, lambda f=r[1]: decode(audio[f]))
    return b.select(a.n, per_group=1)


def granary_ids(hub, lang="it", cache=None):
    """every Granary-<lang> original_audio_id (+ its 11-char YouTube-like tokens); cached as JSON when `cache` is set
    (the listing reads one column of every shard: ~2.4k shards for it, ~4.4k for fr)."""
    if cache and os.path.exists(cache):
        out = set(json.load(open(cache)))
        print(f"== granary-{lang}: {len(out)} ids from {cache}", flush=True)
        return out
    R = "espnet/yodas-granary"
    dirs = [d["path"] for d in hub.tree(R, "data") if d["type"] == "directory" and re.match(rf"data/{lang}\d", d["path"])]
    paths = []
    for d in dirs:
        for kind in ("asr_only", "ast"):
            try:
                paths += [e["path"] for e in hub.tree(R, f"{d}/{kind}") if e["path"].endswith(".parquet")]
            except Exception:  # noqa: BLE001  (a dir may have only one kind)
                pass
    ids = {str(r["original_audio_id"]) for r in hub.columns(R, paths, ["original_audio_id"], workers=24)}
    toks = {t for i in ids for t in re.findall(r"[A-Za-z0-9_-]{11}", i)}
    print(f"== granary-{lang}: {len(dirs)} dirs, {len(paths)} shards, {len(ids)} videos", flush=True)
    if cache:
        with open(cache, "w") as f:
            json.dump(sorted(ids | toks), f)
    return ids | toks


def tedx_shards(a, hub, R):
    """mTEDx TRAIN shards 0-1: it = the frozen 2026-10-18 paths (30 shards); other languages are listed (fr: 41)."""
    if a.lang == "it":
        return [f"it/train-{i:05d}-of-00030.parquet" for i in (0, 1)]
    tr = hub.files(R, SRC[a.lang]["hf"], "train")
    assert len(tr) >= 2 and "/train-00000-of-" in tr[0] and "/train-00001-of-" in tr[1], tr[:3]
    return tr[:2]


def tedx(a, hub, seen, name="tedx"):
    """name "tedx" = FINAL (train shards 0-1 minus the test / valid talks); "tedxdev" (fr, FR-OOD1) = the test + valid
    talks themselves (all of them, no train talk); both drop every Granary-<lang> original_audio_id (= YouTube id)."""
    R, L = "deepdml/mtedx", SRC[a.lang]["hf"]
    devp = [f"{L}/test-00000-of-00001.parquet", f"{L}/valid-00000-of-00001.parquet"]
    dev_talks = {r["talk_id"] for p in devp for r in hub.column(R, p, ["talk_id"])}
    gran = granary_ids(hub, a.lang, f"{a.tmp}/granary_{a.lang}_ids.json")
    b, end = Bank(name, a, seen), ENDING[a.lang]
    if name == "tedxdev":
        tr_talks = {r["talk_id"] for r in hub.columns(R, hub.files(R, L, "train"), ["talk_id"])}
        assert not (dev_talks & tr_talks), f"mTEDx dev talks in train: {sorted(dev_talks & tr_talks)[:5]}"
    for i, p in enumerate(tedx_shards(a, hub, R) if name == "tedx" else devp):
        for r in (hub.column(R, p, ["id", "talk_id", "transcript", "duration"]) if a.plan_only else
                  parquet_audio(fetch(f"{HF}/{R}/resolve/main/{p}", f"{a.tmp}/mtedx_{'train' if name == 'tedx' else 'dev'}_{i}.parquet"))):
            if name == "tedx" and r["talk_id"] in dev_talks:
                b.drop["dev talk"] += 1
                continue
            if r["talk_id"] in gran:
                b.drop[f"talk in Granary-{a.lang}"] += 1
                continue
            b.offer(r["id"], r["talk_id"], r["transcript"], float(r["duration"]),
                    lambda r=r: decode(r["audio"]["bytes"]), min_words=3,
                    ending=lambda t: bool(re.search(end, t)))
    return b.select(a.n, per_group=a.tedx_per_talk)


def tedxdev(a, hub, seen):
    return tedx(a, hub, seen, "tedxdev")


def italic(a, hub, seen):
    if a.plan_only:
        print("== italic (plan): skipped (needs the full Zenodo zip)", flush=True)
        return []
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


def smassive(a, hub, seen, name="smassive"):
    """name "smassive" = FINAL: Speech-MASSIVE fr-FR VALIDATION; "smtrain" (FR-OOD1) = the fr-FR train split (train-*
    only, not train_115), whose speakers FINAL excluded, so the two are speaker-disjoint; FINAL's validation speakers are
    dropped here too. Speech-MASSIVE fr-FR VALIDATION (FBK-MT/Speech-MASSIVE, CC BY-NC-SA 4.0: private evaluation only, never
    redistribute); group = speaker_id; speakers of train / train_115 (+ the gated test repo when readable) dropped."""
    R, L = "FBK-MT/Speech-MASSIVE", SRC[a.lang]["smassive"]
    tr = hub.files(R, L, "train")   # train-* and train_115-*
    assert any("/train-" in p for p in tr) and any("/train_115-" in p for p in tr), tr
    if name == "smtrain":
        val = hub.files(R, L, "validation")
        ex = {str(r["speaker_id"]) for r in hub.columns(R, val, ["speaker_id"])}
        src = [p for p in tr if "/train-" in p]
        print(f"== smtrain: {len(ex)} validation (FINAL) speakers excluded", flush=True)
    else:
        ex = {str(r["speaker_id"]) for r in hub.columns(R, tr, ["speaker_id"])}
        src = hub.files(R, L, "validation")
    T = "FBK-MT/Speech-MASSIVE-test"   # smtrain: also drop the gated test speakers when readable
    try:
        te = {str(r["speaker_id"]) for r in hub.columns(T, hub.files(T, L, "test"), ["speaker_id"])}
        print(f"== smassive: {len(ex)} train + {len(te)} test speakers excluded", flush=True)
        ex |= te
    except Exception as e:  # noqa: BLE001  (gated: needs an accepted licence on the token's account)
        print(f"== smassive: {len(ex)} train speakers excluded; {T} not readable ({type(e).__name__}): test speakers "
              f"NOT excluded", flush=True)
    b = Bank(name, a, seen)
    for i, p in enumerate(src):
        for r in (hub.column(R, p, ["path", "speaker_id", "utt", "is_validated"]) if a.plan_only else
                  parquet_audio(fetch(f"{HF}/{R}/resolve/main/{p}", f"{a.tmp}/{name}_{i}.parquet"))):
            if str(r["speaker_id"]) in ex:
                b.drop["speaker in train/test"] += 1
                continue
            if not r["is_validated"]:
                b.drop["not validated"] += 1
                continue
            ab = None if a.plan_only else r["audio"]["bytes"]
            b.offer(Path(r["path"]).stem, r["speaker_id"], r["utt"],
                    None if ab is None else sf.info(io.BytesIO(ab)).duration, lambda ab=ab: decode(ab))
    return b.select(a.n)


def smtrain(a, hub, seen):
    return smassive(a, hub, seen, "smtrain")


BANKS = {"mls": mls, "cv": cv, "vp": vp, "fleurs": fleurs, "tedx": tedx, "italic": italic, "smassive": smassive,
         "tedxdev": tedxdev, "smtrain": smtrain}   # tedxdev / smtrain: the FR-OOD1 dev bank (2026-10-20)
ONLY = {"italic": "it", "smassive": "fr", "tedxdev": "fr", "smtrain": "fr"}   # language-specific banks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="it", choices=sorted(SRC), help="it = the frozen FINAL-IT build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", help="wav root (required unless --plan-only)")
    ap.add_argument("--tmp", default=None, help="download dir (default /root/final-<lang>-dl)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tedx-per-talk", type=int, default=25)
    ap.add_argument("--exclude-texts", required=True, help="comma-separated globs of training / dev manifests")
    ap.add_argument("--banks", default=None, help="default per language: " +
                    "; ".join(f"{k}: {v['banks']}" for k, v in SRC.items()))
    ap.add_argument("--vp-exclude", default="", help="VP: exclude the speakers of these manifests (ours) instead of every "
                    "VP train / test speaker (amendment 2026-10-18: the latter left 3 speakers)")
    ap.add_argument("--seed-final", default="", help="comma-separated globs of ALREADY-BUILT FINAL banks (final_<bank>.json): "
                    "their texts seed the cross-bank dedup, as if built in the same run (completing a bank later)")
    ap.add_argument("--plan-only", action="store_true", help="metadata only: candidate counts, drops and the selection "
                    "order -> $OUT/plan_<bank>.json; no audio download / decode, no final_* / SHA256SUMS written")
    ap.add_argument("--hub-workers", type=int, default=0, help="cap on parallel Hub column reads (rate-limited IPs); "
                    "0 = the frozen defaults")
    ap.add_argument("--hub-block-kb", type=int, default=0, help="read-ahead (KiB) of Hub column reads: small = only the "
                    "projected column chunks cross the network (audio parquets have MB-sized row groups); 0 = default")
    a = ap.parse_args()
    a.tmp = a.tmp or f"/root/final-{a.lang}-dl"
    a.banks = a.banks or SRC[a.lang]["banks"]
    bad = [b for b in a.banks.split(",") if b not in BANKS or ONLY.get(b, a.lang) != a.lang]
    if bad:
        ap.error(f"banks {bad} not available for --lang {a.lang}")
    if a.plan_only:
        if os.path.exists(f"{a.out}/SHA256SUMS"):
            ap.error(f"--plan-only refuses a sealed --out ({a.out}/SHA256SUMS exists)")
        os.makedirs(a.out, exist_ok=True)
    elif not a.audio:
        ap.error("--audio is required unless --plan-only")
    for d in ((a.out, a.audio, a.tmp) if not a.plan_only else (a.tmp,)):   # plan: tmp holds the small FLEURS TSVs
        os.makedirs(d, exist_ok=True)
    nf = normf(a.lang)
    ex, nman, nskip = set(), 0, 0
    for g in a.exclude_texts.split(","):
        for p in sorted(glob.glob(g)):
            try:   # JSONL manifests only (hours.json, stats files ... are skipped)
                rows = [json.loads(l) for l in open(p) if l.strip()]
                txt = [r["text"] for r in rows]
            except (ValueError, KeyError, TypeError):
                nskip += 1
                continue
            nman += 1
            ex.update(nf(t) for t in txt)
    print(f"== exclusion texts: {len(ex)} from {nman} manifests ({nskip} non-manifest files skipped)", flush=True)
    seen, hub, allr = {"exclude": ex, "final": set()}, Hub(), []
    hub.cap, hub.block = a.hub_workers, a.hub_block_kb * 1024
    for g in filter(None, a.seed_final.split(",")):
        for p in sorted(glob.glob(g)):
            seen["final"].update(nf(json.loads(l)["text"]) for l in open(p) if l.strip())
    if a.seed_final:
        print(f"== seeded FINAL dedup with {len(seen['final'])} texts of already-built banks", flush=True)
    for name in a.banks.split(","):
        rows = BANKS[name](a, hub, seen)
        allr += rows
        if a.plan_only:
            with open(f"{a.out}/plan_{name}.json", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            continue
        for tag, key, dk in (("final", "audio_filepath", "duration"), ("final_trim", "trim_filepath", "trim_duration")):
            with open(f"{a.out}/{tag}_{name}.json", "w") as f:
                for r in rows:
                    f.write(json.dumps({**r, "audio_filepath": r[key], "duration": r[dk]}, ensure_ascii=False) + "\n")
    if a.plan_only:
        print(f"== PLAN-ONLY {SRC[a.lang]['tag']}: {len(allr)} clips would be selected; nothing downloaded / sealed")
        return
    # final_all / stats / SHA256SUMS always cover EVERY bank in --out (a bank completed later included)
    built = sorted(p for p in glob.glob(f"{a.out}/final_*.json") if not re.search(r"/final_(trim_|all)", p))
    allr = [json.loads(l) for p in built for l in open(p) if l.strip()]
    banks = [re.search(r"final_(.+)\.json$", p).group(1) for p in built]
    with open(f"{a.out}/final_all.json", "w") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{a.out}/stats.md", "w") as f:
        f.write("| bank | clips | h | groups | dur p10/p50/p90 s | trail_s p10/p50/p90 | trail < 0.2 s | words p50 |\n|---|---|---|---|---|---|---|---|\n")
        for name in banks:
            rs = [r for r in allr if r["bank"] == name]
            if not rs:
                continue
            d, t = np.array([r["duration"] for r in rs]), np.array([r["trail_s"] for r in rs])
            w = np.median([len(nf(r["text"]).split()) for r in rs])
            f.write(f"| {name} | {len(rs)} | {d.sum() / 3600:.2f} | {len({r['group'] for r in rs})} | "
                    f"{'/'.join(f'{v:.1f}' for v in np.percentile(d, [10, 50, 90]))} | "
                    f"{'/'.join(f'{v:.2f}' for v in np.percentile(t, [10, 50, 90]))} | {(t < 0.2).mean() * 100:.0f} % | {w:.0f} |\n")
    print(open(f"{a.out}/stats.md").read())
    wavs = sorted(str(p) for p in Path(a.audio).rglob("*.wav"))
    with open(f"{a.out}/SHA256SUMS", "w") as f:
        for p in sorted(glob.glob(f"{a.out}/final*.json")) + wavs:
            f.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {p}\n")
    print(f"== {SRC[a.lang]['tag']}-DONE {len(allr)} clips, {len(wavs)} wavs hashed")


if __name__ == "__main__":
    main()
