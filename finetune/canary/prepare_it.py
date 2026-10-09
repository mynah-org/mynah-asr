#!/usr/bin/env python3
"""Italian data for the Canary 180M fine-tuning kit (finetune/canary/prepare_it.py).

Sources (all CC-BY-4.0, public on the HF Hub, no token needed):
  - MLS Italian, facebook/multilingual_librispeech (parquet, config `italian`):
    train = training pool (59,623 utts, ~247 h, 65 speakers); test = frozen eval.
  - FLEURS it_it, google/fleurs: test = frozen eval; train = optional pool
    extra (--with-fleurs-train). FLEURS en_us test: a few clips for the English
    forgetting check.
  - Common Voice Italian (CC0): OPTIONAL, only with CV_TARBALL=<path to the
    MDC per-language tarball> (Mozilla Data Collective needs an API key, see
    README / data_it.sh). Adds a `cv` source to the pool (share CV_SHARE,
    default 0.5, <= 20 clips per speaker) and a fixed 2,000-clip CV test eval.
    UNVERIFIED on a real tarball: member names are matched by suffix.

Stages (each leaves a marker in $FT_ROOT/done/, re-runs skip finished work):
  meta       MLS metadata by parquet column projection (no audio bytes),
             FLEURS tsv files.
  plan       frozen eval lists + NESTED 5 h c 20 h c 40 h cuts (fixed seed,
             speaker-capped, speaker-disjoint from eval) -> plan.json.
  audio      fetch ONLY the planned audio: MLS shards one at a time (transient,
             deleted after use), FLEURS tarballs streamed (never stored).
             Output: 16 kHz mono PCM16 wav.
  manifests  NeMo/lhotse manifests with canary2 fields, exact hours from the
             written wav files (hours.json), SentencePiece text pool.

  --dry-run      run meta + plan and print the plan; no audio.
  --synthetic    fabricate MLS-shaped metadata (laptop test of the plan logic,
                 no network at all). Implies --dry-run.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import ftlib  # noqa: E402
from subsets import balanced_order, merge_by_share, nested_cuts  # noqa: E402,F401

FT = ftlib.FT
SR = ftlib.SR
HF = "https://huggingface.co/datasets"
MLS_REPO = "facebook/multilingual_librispeech"
MLS_TRAIN = [f"italian/train-{i:05d}-of-00008.parquet" for i in range(8)]
MLS_TEST = ["italian/test-00000-of-00001.parquet"]
MLS_META_COLS = ["id", "speaker_id", "chapter_id", "transcript", "audio_duration"]
FLEURS = f"{HF}/google/fleurs/resolve/main/data"

# --------------------------------------------------------------------------
# small utils
# --------------------------------------------------------------------------


def done(name):
    return (FT / "done" / name).exists()


def mark(name):
    (FT / "done").mkdir(parents=True, exist_ok=True)
    (FT / "done" / name).write_text(time.strftime("%F %T") + "\n")


def curl(url, dst, timeout_s=3600):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    # HTTP/1.1 and an outer retry: the Hub's HTTP/2 streams were seen cancelled
    # mid-transfer (curl exit 92), which --retry does not cover; -C - resumes.
    cmd = ["curl", "-fsSL", "--http1.1", "--retry", "8", "--retry-all-errors", "--max-time", str(timeout_s),
           "-C", "-", "-o", str(tmp), url]
    for attempt in range(5):
        r = subprocess.run(cmd)
        if r.returncode == 0:
            break
        print(f"curl rc={r.returncode} on {url} (attempt {attempt + 1}/5), resuming", flush=True)
        time.sleep(10 * (attempt + 1))
    else:
        raise subprocess.CalledProcessError(r.returncode, cmd)
    os.replace(tmp, dst)


def write_wav(dst, audio, sr):
    import numpy as np
    import soundfile as sf

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio.functional as AF

        audio = AF.resample(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), sr, SR).numpy()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.wav")
    sf.write(str(tmp), audio, SR, subtype="PCM_16")
    os.replace(tmp, dst)
    return len(audio) / SR


def decode_bytes(b):
    import soundfile as sf

    try:
        a, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        return a, sr
    except Exception:
        # e.g. a codec this libsndfile lacks: fall back to ffmpeg if present
        if not shutil.which("ffmpeg"):
            raise
        p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR),
                            "-f", "wav", "pipe:1"], input=b, capture_output=True, check=True)
        a, sr = sf.read(io.BytesIO(p.stdout), dtype="float32", always_2d=True)
        return a, sr


# --------------------------------------------------------------------------
# stage: meta
# --------------------------------------------------------------------------


def mls_meta_file(relpath):
    """Read metadata columns only. First try HfFileSystem (range reads, no audio
    bytes); fall back to a transient full download of the shard."""
    import pyarrow.parquet as pq

    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        with fs.open(f"datasets/{MLS_REPO}/{relpath}", "rb", block_size=8 << 20) as f:
            t = pq.ParquetFile(f).read(columns=MLS_META_COLS)
        return t.to_pylist(), "range-read"
    except Exception as e:  # pragma: no cover - network dependent
        print(f"  column projection failed for {relpath} ({e!r}); transient download", flush=True)
        tmp = FT / "tmp" / Path(relpath).name
        curl(f"{HF}/{MLS_REPO}/resolve/main/{relpath}", tmp)
        try:
            t = pq.ParquetFile(str(tmp)).read(columns=MLS_META_COLS)
            return t.to_pylist(), "transient-download"
        finally:
            tmp.unlink(missing_ok=True)


def fleurs_tsv(cfg, split):
    dst = FT / "meta" / f"fleurs_{cfg}_{split}.tsv"
    if not dst.exists():
        curl(f"{FLEURS}/{cfg}/{split}.tsv", dst)
    rows, seen = [], set()
    with open(dst, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) < 7 or r[1] in seen:
                continue
            seen.add(r[1])
            rows.append({"sid": int(r[0]), "file": r[1], "raw": r[2], "norm": r[3],
                         "duration": int(r[5]) / SR, "gender": r[6].strip().lower()})
    rows.sort(key=lambda r: (r["sid"], r["file"]))  # deterministic, not tsv order
    return rows


def stage_meta(a):
    out = FT / "meta"
    out.mkdir(parents=True, exist_ok=True)
    for split, files in (("train", MLS_TRAIN), ("test", MLS_TEST)):
        dst = out / f"mls_{split}.jsonl"
        if dst.exists():
            continue
        rows = []
        for rel in files:
            t0 = time.time()
            rs, how = mls_meta_file(rel)
            for r in rs:
                r["shard"] = rel
            rows += rs
            print(f"  meta {rel}: {len(rs)} rows ({how}, {time.time() - t0:.0f}s)", flush=True)
        ftlib.write_manifest(dst, rows)
    fleurs_tsv("it_it", "test")
    fleurs_tsv("it_it", "train")
    fleurs_tsv("en_us", "test")
    if CV_TARBALL:
        cv_meta()


# --------------------------------------------------------------------------
# optional Common Voice (CC0) from a local MDC tarball
# --------------------------------------------------------------------------

CV_TARBALL = os.environ.get("CV_TARBALL", "")
CV_SHARE = float(os.environ.get("CV_SHARE", "0.5"))
CV_TSVS = ("/it/train.tsv", "/it/test.tsv", "/it/clip_durations.tsv")


def cv_meta():
    out = FT / "meta"
    if all((out / ("cv_" + t.rsplit("/", 1)[1])).exists() for t in CV_TSVS):
        return
    with tarfile.open(CV_TARBALL, mode="r|*") as tar:
        for m in tar:
            for t in CV_TSVS:
                if m.isfile() and m.name.endswith(t):
                    (out / ("cv_" + t.rsplit("/", 1)[1])).write_bytes(tar.extractfile(m).read())
    for t in CV_TSVS:
        if not (out / ("cv_" + t.rsplit("/", 1)[1])).exists():
            raise SystemExit(f"CV tarball has no member ending in {t}")


def cv_rows(split):
    dur = {}
    with open(FT / "meta" / "cv_clip_durations.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            dur[r["clip"]] = float(r["duration[ms]"]) / 1000.0
    rows = []
    with open(FT / "meta" / f"cv_{split}.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            p = r["path"]
            if p in dur:
                rows.append({"src": "cv", "id": p.rsplit(".", 1)[0], "file": p, "speaker": r["client_id"],
                             "duration": dur[p], "text": r["sentence"]})
    rows.sort(key=lambda u: u["id"])
    return rows


def cv_stream(files):
    want = {f for f in files if not wav_path("cv", f.rsplit(".", 1)[0]).exists()}
    n = 0
    if not want:
        return 0
    with tarfile.open(CV_TARBALL, mode="r|*") as tar:
        for m in tar:
            base = m.name.rsplit("/", 1)[-1]
            if m.isfile() and base in want:
                arr, sr = decode_bytes(tar.extractfile(m).read())
                write_wav(wav_path("cv", base.rsplit(".", 1)[0]), arr, sr)
                want.discard(base)
                n += 1
                if not want:
                    break
    if want:
        print(f"  WARNING cv: {len(want)} clips not found in the tarball", flush=True)
    return n


def synthetic_meta():
    """MLS-shaped fake metadata: 65 speakers, 59,623 utts, 10-20 s, ~247 h."""
    rng = random.Random(1)
    train, test = [], []
    spk_w = [rng.uniform(0.2, 3.0) for _ in range(65)]
    for i in range(59623):
        s = rng.choices(range(65), weights=spk_w)[0]
        train.append({"id": f"syn_{s}_{i}", "speaker_id": f"{s}", "chapter_id": "0",
                      "transcript": "testo di prova", "audio_duration": rng.uniform(10, 20),
                      "shard": MLS_TRAIN[i % 8]})
    for i in range(1262):
        s = 100 + i % 10
        test.append({"id": f"syn_{s}_{i}", "speaker_id": f"{s}", "chapter_id": "0",
                     "transcript": "testo di prova", "audio_duration": rng.uniform(10, 20), "shard": MLS_TEST[0]})
    fl_test = [{"sid": i // 3, "file": f"t{i}.wav", "raw": "Testo.", "norm": "testo", "duration": rng.uniform(5, 20),
                "gender": "female"} for i in range(865)]
    fl_train = [{"sid": 2000 + i // 2, "file": f"r{i}.wav", "raw": "Testo.", "norm": "testo",
                 "duration": rng.uniform(5, 20), "gender": "male"} for i in range(3030)]
    fl_en = [{"sid": i, "file": f"e{i}.wav", "raw": "Test.", "norm": "test", "duration": 8.0, "gender": "male"}
             for i in range(647)]
    return train, test, fl_test, fl_train, fl_en


# --------------------------------------------------------------------------
# stage: plan
# --------------------------------------------------------------------------


# balanced_order / merge_by_share / nested_cuts: finetune/common/subsets.py


def stage_plan(a, synthetic=False):
    if synthetic:
        mls_tr, mls_te, fl_te, fl_tr, fl_en = synthetic_meta()
    else:
        mls_tr = ftlib.read_manifest(FT / "meta" / "mls_train.jsonl")
        mls_te = ftlib.read_manifest(FT / "meta" / "mls_test.jsonl")
        fl_te, fl_tr, fl_en = fleurs_tsv("it_it", "test"), fleurs_tsv("it_it", "train"), fleurs_tsv("en_us", "test")

    # ---- frozen eval: ALL of FLEURS it test + ALL of MLS it test ----
    eval_mls = [{"src": "mls", "id": r["id"], "speaker": str(r["speaker_id"]), "duration": float(r["audio_duration"]),
                 "text": r["transcript"], "shard": r["shard"]} for r in mls_te]
    eval_fl = [{"src": "fleurs", "id": r["file"].rsplit(".", 1)[0], "file": r["file"], "sid": r["sid"],
                "duration": r["duration"], "text": r["raw"], "norm": r["norm"]} for r in fl_te]
    test_spk = {u["speaker"] for u in eval_mls}
    test_sid = {u["sid"] for u in eval_fl}

    # ---- training pool ----
    pool_mls, dropped = {}, {"duration": 0, "test_speaker": 0}
    for r in mls_tr:
        d = float(r["audio_duration"])
        if not (a.min_dur <= d <= a.max_dur):
            dropped["duration"] += 1
            continue
        if str(r["speaker_id"]) in test_spk:
            dropped["test_speaker"] += 1
            continue
        pool_mls.setdefault(str(r["speaker_id"]), []).append(
            {"src": "mls", "id": r["id"], "speaker": str(r["speaker_id"]), "duration": d, "text": r["transcript"],
             "shard": r["shard"]})
    for s in pool_mls:
        pool_mls[s].sort(key=lambda u: u["id"])  # input order must not depend on the shard read order
    seqs = {"mls": balanced_order(pool_mls, a.mls_cap_min * 60, a.seed)}
    shares = {"mls": 1.0}
    pool_fl_n = 0
    if a.with_fleurs_train:
        # FLEURS has no speaker id; cap = readings per sentence id, and the
        # sentence ids must be disjoint from FLEURS test (checked, then enforced).
        by_sid = {}
        for r in fl_tr:
            if r["sid"] in test_sid or not (a.min_dur <= r["duration"] <= a.max_dur):
                continue
            by_sid.setdefault(r["sid"], []).append(
                {"src": "fleurs", "id": r["file"].rsplit(".", 1)[0], "file": r["file"], "sid": r["sid"],
                 "duration": r["duration"], "text": r["raw"]})
        rng = random.Random(a.seed + 1)
        fl_seq = []
        sids = sorted(by_sid)
        rng.shuffle(sids)
        for s in sids:
            q = sorted(by_sid[s], key=lambda u: u["id"])
            rng.shuffle(q)
            fl_seq += q[: a.fleurs_per_sentence]
        pool_fl_n = len(fl_seq)
        seqs["fleurs"] = fl_seq
        shares = {"mls": 1.0 - a.fleurs_share, "fleurs": a.fleurs_share}
    eval_cv = []
    if CV_TARBALL:
        cv_te = cv_rows("test")
        rng = random.Random(a.seed + 2)
        eval_cv = sorted(rng.sample(cv_te, min(2000, len(cv_te))), key=lambda u: u["id"])
        cv_test_spk = {u["speaker"] for u in cv_te}
        by_spk = {}
        for u in cv_rows("train"):
            if u["speaker"] in cv_test_spk or not (a.min_dur <= u["duration"] <= a.max_dur):
                continue
            by_spk.setdefault(u["speaker"], []).append(u)
        rng = random.Random(a.seed + 3)
        capped = {}
        for s_ in sorted(by_spk):
            q = list(by_spk[s_])
            rng.shuffle(q)
            capped[s_] = q[:20]
        seqs["cv"] = balanced_order(capped, float("inf"), a.seed + 4)
        rest = 1.0 - CV_SHARE
        shares = {k: v * rest for k, v in shares.items()}
        shares["cv"] = CV_SHARE
    seq = merge_by_share(seqs, shares)

    cuts = nested_cuts(seq, a.hours)  # prefix lengths -> nested
    need = seq[: max(cuts.values())]

    # ---- English forgetting probe: first N FLEURS en_us test clips (sorted) ----
    en = [{"src": "fleurs_en", "id": r["file"].rsplit(".", 1)[0], "file": r["file"], "duration": r["duration"],
           "text": r["raw"]} for r in fl_en[: a.en_clips]]

    plan = {
        "seed": a.seed, "min_dur": a.min_dur, "max_dur": a.max_dur, "mls_cap_min": a.mls_cap_min,
        "with_fleurs_train": a.with_fleurs_train, "fleurs_share": a.fleurs_share if a.with_fleurs_train else 0.0,
        "synthetic": synthetic,
        "pool": {"mls_utts": sum(len(v) for v in pool_mls.values()), "mls_speakers": len(pool_mls),
                 "mls_hours": round(sum(u["duration"] for v in pool_mls.values() for u in v) / 3600, 2),
                 "fleurs_train_utts": pool_fl_n, "dropped": dropped},
        "checks": {"mls_train_test_speaker_overlap_in_metadata": len(test_spk & {str(r["speaker_id"]) for r in mls_tr}),
                   "fleurs_train_test_sentence_overlap": len({r["sid"] for r in fl_tr} & test_sid)},
        "cuts": {},
        "eval": {"fleurs_it_test": eval_fl, "mls_it_test": eval_mls, **({"cv_it_test2000": eval_cv} if eval_cv else {})},
        "en_sanity": en,
        "train_sequence": need,
    }
    for name, n in cuts.items():
        sub = seq[:n]
        by_src = {}
        for u in sub:
            by_src.setdefault(u["src"], 0.0)
            by_src[u["src"]] += u["duration"]
        spk = {}
        for u in sub:
            if u["src"] == "mls":
                spk[u["speaker"]] = spk.get(u["speaker"], 0.0) + u["duration"]
        plan["cuts"][name] = {
            "n_utts": n, "planned_hours": round(sum(u["duration"] for u in sub) / 3600, 4),
            "hours_by_source": {k2: round(v / 3600, 4) for k2, v in by_src.items()},
            "mls_speakers": len(spk),
            "mls_min_per_speaker_max": round(max(spk.values()) / 60, 2) if spk else 0,
            "mls_min_per_speaker_min": round(min(spk.values()) / 60, 2) if spk else 0,
        }
    ftlib.write_json(FT / ("plan.synthetic.json" if synthetic else "plan.json"), plan)
    print_plan(plan)
    if plan["checks"]["mls_train_test_speaker_overlap_in_metadata"]:
        print("NOTE: MLS train/test speaker overlap found in metadata; those train rows were dropped.")
    short = [n for n, c in plan["cuts"].items() if c["planned_hours"] < float(n[:-1]) * 0.98]
    if short:
        print(f"WARNING: cuts {short} are short of target: raise --mls-cap-min or add sources.")
    return plan


def print_plan(p):
    ev = p["eval"]
    print("== plan" + (" (SYNTHETIC metadata: logic test only)" if p["synthetic"] else ""))
    for name, rows in ev.items():
        print(f"  eval {name:15s} {len(rows):5d} utts {sum(r['duration'] for r in rows) / 3600:6.2f} h")
    print(f"  en_sanity       {len(p['en_sanity']):5d} utts {sum(r['duration'] for r in p['en_sanity']) / 60:6.1f} min")
    pl = p["pool"]
    print(f"  pool MLS {pl['mls_utts']} utts / {pl['mls_speakers']} speakers / {pl['mls_hours']} h; "
          f"FLEURS train {pl['fleurs_train_utts']} utts; dropped {pl['dropped']}")
    print(f"  checks {p['checks']}")
    for name, c in p["cuts"].items():
        print(f"  cut {name:>4s}: {c['n_utts']:6d} utts {c['planned_hours']:7.3f} h {c['hours_by_source']} "
              f"MLS speakers {c['mls_speakers']} ({c['mls_min_per_speaker_min']}-{c['mls_min_per_speaker_max']} min each)")
    seq_h = sum(u["duration"] for u in p["train_sequence"]) / 3600
    ev_h = sum(r["duration"] for rows in ev.values() for r in rows) / 3600
    print(f"  audio to fetch: train {seq_h:.2f} h + eval {ev_h:.2f} h -> ~{(seq_h + ev_h) * 0.115:.1f} GB of 16 kHz PCM16")


# --------------------------------------------------------------------------
# stage: audio
# --------------------------------------------------------------------------


def wav_path(src, uid):
    return FT / "audio" / src / f"{uid}.wav"


def mls_shard_job(rel, ids):
    """Download one shard (transient), write the wanted rows as wav, delete it."""
    import pyarrow.parquet as pq

    want = {i for i in ids if not wav_path("mls", i).exists()}
    if not want:
        return rel, 0, 0
    tmp = FT / "tmp" / Path(rel).name
    curl(f"{HF}/{MLS_REPO}/resolve/main/{rel}", tmp)
    n = fail = 0
    try:
        pf = pq.ParquetFile(str(tmp))
        for batch in pf.iter_batches(columns=["id", "audio"], batch_size=256):
            ids_b = batch.column(0).to_pylist()
            if not want.intersection(ids_b):
                continue
            auds = batch.column(1).to_pylist()
            for uid, au in zip(ids_b, auds):
                if uid not in want:
                    continue
                try:
                    arr, sr = decode_bytes(au["bytes"])
                    write_wav(wav_path("mls", uid), arr, sr)
                    n += 1
                except Exception as e:
                    fail += 1
                    print(f"  FAIL mls {uid}: {e!r}", flush=True)
    finally:
        tmp.unlink(missing_ok=True)
    return rel, n, fail


def fleurs_stream(cfg, split, files, dst_src):
    """Stream data/<cfg>/audio/<split>.tar.gz; keep only `files`; stop early when done."""
    want = {f for f in files if not wav_path(dst_src, f.rsplit(".", 1)[0]).exists()}
    if not want:
        return 0
    url = f"{FLEURS}/{cfg}/audio/{split}.tar.gz"
    n = 0
    with urllib.request.urlopen(url, timeout=120) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for m in tar:
                base = m.name.rsplit("/", 1)[-1]
                if not m.isfile() or base not in want:
                    continue
                b = tar.extractfile(m).read()
                arr, sr = decode_bytes(b)
                write_wav(wav_path(dst_src, base.rsplit(".", 1)[0]), arr, sr)
                want.discard(base)
                n += 1
                if not want:
                    break
    if want:
        print(f"  WARNING {cfg}/{split}: {len(want)} files not found in the tarball", flush=True)
    return n


def stage_audio(a):
    plan = json.loads((FT / "plan.json").read_text())
    (FT / "tmp").mkdir(parents=True, exist_ok=True)
    jobs = {}
    for u in plan["train_sequence"] + plan["eval"]["mls_it_test"]:
        if u["src"] == "mls":
            jobs.setdefault(u["shard"], []).append(u["id"])
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(mls_shard_job, rel, ids) for rel, ids in sorted(jobs.items())]
        for f in futs:
            rel, n, fail = f.result()
            print(f"  mls {rel}: wrote {n}, failed {fail} ({time.time() - t0:.0f}s)", flush=True)
    n = fleurs_stream("it_it", "test", [u["file"] for u in plan["eval"]["fleurs_it_test"]], "fleurs_it")
    print(f"  fleurs it test: wrote {n} ({time.time() - t0:.0f}s)", flush=True)
    fl_tr = [u["file"] for u in plan["train_sequence"] if u["src"] == "fleurs"]
    if fl_tr:
        n = fleurs_stream("it_it", "train", fl_tr, "fleurs_it")
        print(f"  fleurs it train: wrote {n} ({time.time() - t0:.0f}s)", flush=True)
    cv = [u["file"] for u in plan["train_sequence"] + plan["eval"].get("cv_it_test2000", []) if u["src"] == "cv"]
    if cv:
        n = cv_stream(cv)
        print(f"  cv: wrote {n} ({time.time() - t0:.0f}s)", flush=True)
    n = fleurs_stream("en_us", "test", [u["file"] for u in plan["en_sanity"]], "fleurs_en")
    print(f"  fleurs en test: wrote {n} ({time.time() - t0:.0f}s)", flush=True)
    shutil.rmtree(FT / "tmp", ignore_errors=True)


# --------------------------------------------------------------------------
# stage: manifests
# --------------------------------------------------------------------------


def stage_manifests(a):
    import soundfile as sf

    plan = json.loads((FT / "plan.json").read_text())
    man = FT / "manifests"
    hours = {"note": "seconds summed from the written 16 kHz wav files, not from metadata; "
                     "peak_dbfs = per-clip sample peak (level-robustness context)"}
    missing = {}

    def rows_for(units, lang, pnc_of):
        rows, miss = [], 0
        for u in units:
            src = {"mls": "mls", "fleurs": "fleurs_it", "fleurs_en": "fleurs_en", "cv": "cv"}[u["src"]]
            p = wav_path(src, u["id"])
            if not p.exists():
                miss += 1
                continue
            audio, sr = sf.read(str(p), dtype="float32")
            dur = len(audio) / sr
            extra = {"corpus": u["src"], "utt_id": u["id"], "peak_dbfs": ftlib.peak_dbfs(audio)}
            if "speaker" in u:
                extra["speaker"] = u["speaker"]
            rows.append(ftlib.canary_row(p, dur, u["text"], lang, pnc_of(u), **extra))
        return rows, miss

    # MLS text is lower-case, unpunctuated -> pnc=no; FLEURS raw text is cased
    # and punctuated -> pnc=yes. Scoring always normalises both sides.
    pnc_of = lambda u: "no" if u["src"] == "mls" else "yes"  # noqa: E731  (CV sentences are cased too)
    sets = {
        "eval_fleurs_it": (plan["eval"]["fleurs_it_test"], "it"),
        "eval_mls_it": (plan["eval"]["mls_it_test"], "it"),
        "en_sanity": (plan["en_sanity"], "en"),
    }
    if "cv_it_test2000" in plan["eval"]:
        sets["eval_cv_it"] = (plan["eval"]["cv_it_test2000"], "it")
    seq = plan["train_sequence"]
    for name, c in plan["cuts"].items():
        sets[f"train_{name}"] = (seq[: c["n_utts"]], "it")
    for name, (units, lang) in sets.items():
        rows, miss = rows_for(units, lang, pnc_of)
        ftlib.write_manifest(man / f"{name}.json", rows)
        secs = sum(r["duration"] for r in rows)
        pk = sorted(r["peak_dbfs"] for r in rows)
        pct = (lambda q: pk[min(len(pk) - 1, int(q * len(pk)))] if pk else None)  # noqa: E731
        hours[name] = {"n_utts": len(rows), "seconds": round(secs, 3), "hours": round(secs / 3600, 4),
                       "missing_audio": miss,
                       "peak_dbfs_p10_p50_p90": [pct(0.1), pct(0.5), pct(0.9)]}
        if miss:
            missing[name] = miss
        print(f"  {name:16s} {len(rows):6d} utts {secs / 3600:8.4f} h  missing {miss}  "
              f"peak dBFS p10/p50/p90 {hours[name]['peak_dbfs_p10_p50_p90']}", flush=True)
    ftlib.write_manifest(man / "eval_it_all.json",
                         ftlib.read_manifest(man / "eval_fleurs_it.json") + ftlib.read_manifest(man / "eval_mls_it.json"))
    # nesting check on what was actually written
    names = sorted([n for n in sets if n.startswith("train_")], key=lambda n: float(n[6:-1]))
    prev = set()
    for n in names:
        cur = {r["utt_id"] for r in ftlib.read_manifest(man / f"{n}.json")}
        assert prev <= cur, f"nesting broken at {n}"
        prev = cur
    hours["nested_check"] = "ok: " + " c ".join(names)
    # SentencePiece text: the whole TRAINING POOL (never eval text)
    pool_txt = FT / "text" / "it_pool.txt"
    pool_txt.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with open(pool_txt, "w", encoding="utf-8") as f:
        for r in ftlib.read_manifest(FT / "meta" / "mls_train.jsonl"):
            f.write(r["transcript"].strip() + "\n")
            n_lines += 1
        for u in plan["train_sequence"]:
            if u["src"] == "cv":  # only the selected CV clips, never CV test text
                f.write(u["text"].strip() + "\n")
                n_lines += 1
        if plan["with_fleurs_train"]:
            test_sid = {u["sid"] for u in plan["eval"]["fleurs_it_test"]}
            for r in fleurs_tsv("it_it", "train"):
                if r["sid"] not in test_sid:
                    f.write(r["raw"].strip() + "\n")
                    n_lines += 1
    hours["sentencepiece_text"] = {"path": str(pool_txt), "lines": n_lines,
                                   "sources": "MLS it train (all)" + (" + FLEURS it train" if plan["with_fleurs_train"] else "")}
    ftlib.write_json(FT / "hours.json", hours)
    if missing:
        print(f"WARNING missing audio: {missing} (re-run the audio stage; it only fetches what is missing)")
    du = subprocess.run(["du", "-sh", str(FT / "audio")], capture_output=True, text=True).stdout.strip()
    print(f"  audio on disk: {du}")


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "meta", "plan", "audio", "manifests"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--hours", default="5,20,40")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--mls-cap-min", type=float, default=60.0,
                    help="max minutes per MLS speaker in the largest cut (65 speakers -> 65 h ceiling)")
    ap.add_argument("--with-fleurs-train", action="store_true",
                    default=os.environ.get("WITH_FLEURS_TRAIN", "0") == "1")
    ap.add_argument("--fleurs-share", type=float, default=0.15)
    ap.add_argument("--fleurs-per-sentence", type=int, default=2)
    ap.add_argument("--en-clips", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4, help="MLS shards processed in parallel (each ~0.5 GB transient)")
    a = ap.parse_args()

    if a.synthetic:
        global FT
        FT = Path(os.environ.get("FT_ROOT", "/tmp/ft-synthetic"))
        ftlib.FT = FT
        stage_plan(a, synthetic=True)
        return 0
    stages = ["meta", "plan", "audio", "manifests"] if a.stage == "all" else [a.stage]
    if a.dry_run:
        stages = [s for s in stages if s in ("meta", "plan")]
    for s in stages:
        if done(f"data-{s}") and not a.dry_run:
            print(f"== skip data-{s}")
            continue
        print(f"== data-{s} {time.strftime('%T')}", flush=True)
        {"meta": stage_meta, "plan": stage_plan, "audio": stage_audio, "manifests": stage_manifests}[s](a)
        if not a.dry_run:
            mark(f"data-{s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
