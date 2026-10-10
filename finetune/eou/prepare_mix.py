#!/usr/bin/env python3
"""Multi-domain data for the EOU 120M specialist (IT/FR): the sources that are NOT MLS.

MLS stays with canary/prepare_it.py (and the FR copy) (nested 5/20/40 h cuts, MLS test eval). This
script adds the domains MLS lacks, all public on the HF Hub and read from parquet
or streamed tarballs, one shard at a time (transient):

  cv       Common Voice 17 (CC0), mirror fixie-ai/common_voice_17_0 <lang>/train|test.
           Read speech, thousands of speakers/microphones. Cap: --cv-spk clips/speaker.
  vp       VoxPopuli (CC0), facebook/voxpopuli <lang>/train|test: parliament speech,
           spontaneous-ish, far-field. Cap: --vp-spk clips/speaker.
  fleurs   FLEURS train (CC-BY 4.0), google/fleurs <cfg>/audio/train.tar.gz.
           NB: once FLEURS train is in the mix, the FLEURS test bank is no longer a
           fully out-of-domain probe; cv/vp test are the extra held-out domains.

Filters (all sources): 1-20 s, no digits (the de-accented a-z targets cannot spell
them), latin script only. The budget per source is spread evenly over its train
shards so the cut touches every shard (speaker diversity), not the first one.

Output: 16 kHz mono PCM16 wav under --audio (default /dev/shm/mix/<lang>, RAM-backed:
the box disk is 32 GB) and NeMo manifests under --out (default /root/ft/mix/<lang>):
train_<src>.json, eval_<src>.json (--eval-n clips from the test split), hours.json.

  /root/nemo-venv/bin/python finetune/eou/prepare_mix.py --lang it --cv-hours 40 --vp-hours 40
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
import time
import unicodedata
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "canary"))
import prepare_it as D  # noqa: E402  (curl, decode_bytes, write_wav, fleurs tsv helpers, ftlib)

HF = "https://huggingface.co"
CV_REPO = "fixie-ai/common_voice_17_0"
VP_REPO = "facebook/voxpopuli"
FLEURS_CFG = {"it": "it_it", "fr": "fr_fr"}
TEXT_COLS = ("sentence", "raw_text", "normalized_text", "transcription", "text")
SPK_COLS = ("client_id", "speaker_id")
ID_COLS = ("path", "audio_id", "id")


def hf_token():
    try:
        return open("/root/.hf_token").read().strip()
    except OSError:
        return ""


def list_files(repo, path):
    req = urllib.request.Request(f"{HF}/api/datasets/{repo}/tree/main/{path}")
    if hf_token():
        req.add_header("Authorization", f"Bearer {hf_token()}")
    return sorted(x["path"] for x in json.load(urllib.request.urlopen(req, timeout=60)) if x["type"] == "file")


def text_ok(t):
    if not t or re.search(r"\d", t):
        return False
    letters = [c for c in unicodedata.normalize("NFKD", t) if c.isalpha()]
    return bool(letters) and all("a" <= c.lower() <= "z" for c in letters)


def pick(cols, names):
    for n in names:
        if n in cols:
            return n
    return None


def shard_job(src, repo, rel, budget_s, spk_cap, audio_dir, tmp_dir, seed, eval_n=0):
    """Download one parquet shard, keep a speaker-capped random subset up to budget_s
    (or eval_n clips), write wavs, delete the shard. Returns manifest rows."""
    import pyarrow.parquet as pq

    tmp = Path(tmp_dir) / f"{src}-{Path(rel).name}"
    D.curl(f"{HF}/datasets/{repo}/resolve/main/{rel}", tmp)
    rows = []
    try:
        pf = pq.ParquetFile(str(tmp))
        cols = pf.schema_arrow.names
        tc = pick(cols, ("normalized_text", "sentence") if src == "vp" else TEXT_COLS)
        sc, ic = pick(cols, SPK_COLS), pick(cols, ID_COLS)
        meta = pf.read(columns=[c for c in (tc, sc, ic) if c]).to_pylist()
        order = list(range(len(meta)))
        random.Random(f"{seed}-{rel}").shuffle(order)
        spk_n, want = {}, []
        for i in order:
            r = meta[i]
            if not text_ok(r[tc]):
                continue
            s = r.get(sc) or f"anon{i}"
            if spk_n.get(s, 0) >= spk_cap:
                continue
            spk_n[s] = spk_n.get(s, 0) + 1
            want.append(i)
        want_set, acc = set(want), 0.0
        # rows are taken in the shuffled order: read the audio of the candidates once,
        # in file order, then admit in shuffled order until the budget
        got = {}
        base = 0
        for batch in pf.iter_batches(columns=["audio"], batch_size=256):
            auds = batch.column(0).to_pylist()
            for j, au in enumerate(auds):
                if base + j in want_set:
                    got[base + j] = au["bytes"]
            base += len(auds)
            if len(got) >= (eval_n * 3 if eval_n else 10 ** 9) and eval_n:
                break
        for i in want:
            if i not in got:
                continue
            try:
                arr, sr = D.decode_bytes(got.pop(i))
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {src} {rel}#{i}: {e!r}", flush=True)
                continue
            dur = len(arr) / sr
            if not 1.0 <= dur <= 20.0:
                continue
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != D.SR:   # CV mp3 is 32/48 kHz; the venv has no torchaudio (prepare_it's resampler)
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, D.SR)
                arr, sr = resample_poly(arr, D.SR // g, sr // g).astype("float32"), D.SR
            r = meta[i]
            uid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(r.get(ic) or i)).rsplit(".", 1)[0]
            p = Path(audio_dir) / src / f"{uid}.wav"
            D.write_wav(p, arr, sr)
            rows.append({"audio_filepath": str(p), "duration": round(dur, 3), "text": r[tc].strip(),
                         "speaker": f"{src}:{r.get(sc) or uid}", "corpus": src})
            acc += dur
            if (eval_n and len(rows) >= eval_n) or (not eval_n and acc >= budget_s):
                break
    finally:
        tmp.unlink(missing_ok=True)
    return rel, rows


def fleurs_train(lang, audio_dir):
    cfg = FLEURS_CFG[lang]
    tsv = D.fleurs_tsv(cfg, "train")
    by = {r["file"]: r for r in tsv if text_ok(r["raw"]) and 1.0 <= r["duration"] <= 20.0}
    rows, url = [], f"{D.FLEURS}/{cfg}/audio/train.tar.gz"
    with urllib.request.urlopen(url, timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            base = m.name.rsplit("/", 1)[-1]
            if not m.isfile() or base not in by:
                continue
            arr, sr = D.decode_bytes(tar.extractfile(m).read())
            p = Path(audio_dir) / "fleurs" / base.replace(".wav", "") / ""
            p = p.with_suffix("").parent / (base.rsplit(".", 1)[0] + ".wav")
            dur = D.write_wav(p, arr, sr)
            r = by.pop(base)
            rows.append({"audio_filepath": str(p), "duration": round(dur, 3), "text": r["raw"].strip(),
                         "speaker": f"fleurs:{r['sid']}", "corpus": "fleurs"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=sorted(FLEURS_CFG))
    ap.add_argument("--cv-hours", type=float, default=40)
    ap.add_argument("--vp-hours", type=float, default=40)
    ap.add_argument("--cv-spk", type=int, default=20)
    ap.add_argument("--vp-spk", type=int, default=60)
    ap.add_argument("--no-fleurs", action="store_true")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--audio", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261010)
    a = ap.parse_args()
    audio = Path(a.audio or f"/dev/shm/mix/{a.lang}")
    out = Path(a.out or f"/root/ft/mix/{a.lang}")
    tmp = audio / "tmp"
    for d in (audio, out, tmp):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = []
    for src, repo, hours, cap in (("cv", CV_REPO, a.cv_hours, a.cv_spk), ("vp", VP_REPO, a.vp_hours, a.vp_spk)):
        if hours <= 0 or (out / f"train_{src}.json").exists():
            print(f"== skip {src}", flush=True)
            continue
        files = list_files(repo, a.lang)
        tr = [f for f in files if Path(f).name.startswith("train-") and f.endswith(".parquet")]
        te = [f for f in files if Path(f).name.startswith("test-") and f.endswith(".parquet")]
        print(f"== {src}: {len(tr)} train shards, {len(te)} test shards, budget {hours} h, cap {cap}/speaker", flush=True)
        jobs.append((src, repo, tr, te, hours * 3600 / len(tr), cap))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for src, repo, tr, te, per, cap in jobs:
            futs = [ex.submit(shard_job, src, repo, rel, per, cap, str(audio), str(tmp), a.seed) for rel in tr]
            ev = ex.submit(shard_job, src, repo, te[0], 0, 2, str(audio), str(tmp), a.seed, a.eval_n)
            rows = []
            for f in futs:
                rel, r = f.result()
                rows += r
                print(f"  {src} {rel}: {len(r)} utts {sum(x['duration'] for x in r) / 3600:.2f} h ({time.time() - t0:.0f}s)", flush=True)
            D.ftlib.write_manifest(out / f"train_{src}.json", rows)
            _, er = ev.result()
            D.ftlib.write_manifest(out / f"eval_{src}.json", er)
            print(f"== {src}: train {len(rows)} utts {sum(x['duration'] for x in rows) / 3600:.2f} h, "
                  f"{len({x['speaker'] for x in rows})} speakers; eval {len(er)}", flush=True)
    if not a.no_fleurs and not (out / "train_fleurs.json").exists():
        rows = fleurs_train(a.lang, audio)
        D.ftlib.write_manifest(out / "train_fleurs.json", rows)
        print(f"== fleurs train: {len(rows)} utts {sum(x['duration'] for x in rows) / 3600:.2f} h, "
              f"{len({x['speaker'] for x in rows})} sentence-ids ({time.time() - t0:.0f}s)", flush=True)
    hours = {}
    for p in sorted(out.glob("*.json")):
        if p.name == "hours.json":
            continue
        rs = D.ftlib.read_manifest(p)
        hours[p.stem] = {"n_utts": len(rs), "hours": round(sum(r["duration"] for r in rs) / 3600, 3),
                         "speakers": len({r.get("speaker") for r in rs})}
    (out / "hours.json").write_text(json.dumps(hours, indent=1))
    print(json.dumps(hours, indent=1))
    print(f"== DATA-MIX-DONE {a.lang} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
