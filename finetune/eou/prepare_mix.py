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

Granary (yg) at scale, incremental mode: any of --yg-dirs / --yg-split-long / --audio-ext /
--yg-hours / --yg-min-free-gb switches it on (without them the yg path is unchanged):

  prepare_mix.py --lang it --cv-hours 0 --vp-hours 0 --no-fleurs --yg-keep-all \
      --yg-dirs it000,it100,it101 --yg-spread 70,250,250 --yg-split-long 20 --audio-ext flac --yg-hours 450

  - --yg-spread takes one N per --yg-dirs entry (today's evenly-spread formula per dir, so
    `--yg-dirs it000 --yg-spread 70` is the same 70 shards as the legacy pool); one N alone spreads
    over the concatenated listing. Shards are pinned to the dataset revision (HF sha) of the run.
  - Each shard's rows go to <out>/yg_shards/<dir>-<shard>.json when the shard is done; the raw
    parquet is deleted right after extraction (as before). A rerun skips cached shards, so
    re-running with a larger --yg-hours (or more dirs) grows the pool as a superset: cached
    shards are admitted first, then new ones in spread-first order, round-robin over the dirs.
  - --yg-exclude-spread it000:70,it100:60,it101:55 drops the shards another pool's spread selected
    (it-yg + it-yg6), so a larger spread builds a pool of videos no earlier pool has seen.
  - --yg-hours H: no new shard is started once the admitted hours (+ the in-flight estimate)
    reach H; granularity = whole shards. --yg-min-free-gb G: same stop when the --audio
    filesystem has < G GB free.
  - --yg-split-long S: clips <= S s are kept whole (>= 1 s; --yg-max-s is ignored), longer ones
    are cut at speech pauses (prepare_yodas.speech_spans / pack_spans: Silero VAD, or the energy
    detector with --yg-vad energy) into pieces of --yg-piece-min-s..S s; a span longer than S
    with no pause is dropped (never cut mid-speech). A piece's `text` is EMPTY: the parent's
    Granary text does not align to it, so the teacher label is the only label (teacher_wer =
    None; pseudo_subsets.py --no-ref-keep decides). It keeps `ref_parent` (parent text),
    `parent_id`, `piece`, `n_pieces`.
  - Every yg row also carries utt_id, shard (repo path), row (index in the parquet), offset
    and duration within the parent (s), parent_dur, revision, and orig_offset / lang_orig when
    the parquet has them. --audio-ext flac writes FLAC PCM_16 16 kHz (soundfile) and deletes a
    stale <utt_id>.wav of the same clip.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
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
YG_REPO = "espnet/yodas-granary"   # Granary's YODAS: pre-segmented clips + Whisper-v3 labels (CC-BY-3.0)
FLEURS_CFG = {"it": "it_it", "fr": "fr_fr"}
TEXT_COLS = ("sentence", "raw_text", "normalized_text", "transcription", "text")
SPK_COLS = ("client_id", "speaker_id", "original_audio_id")
ID_COLS = ("utt_id", "path", "audio_id", "id")
YG_EXTRA_COLS = {"original_audio_offset": "orig_offset", "lang": "lang_orig"}   # kept when present


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


def hf_revision(repo):
    """Current commit sha of a dataset repo (None when the API is unreachable)."""
    req = urllib.request.Request(f"{HF}/api/datasets/{repo}/revision/main")
    if hf_token():
        req.add_header("Authorization", f"Bearer {hf_token()}")
    try:
        return json.load(urllib.request.urlopen(req, timeout=60)).get("sha")
    except Exception:  # noqa: BLE001
        return None


def write_audio(dst, audio, sr):
    """16 kHz mono PCM16, container from the suffix (.wav / .flac); atomic like D.write_wav."""
    import soundfile as sf
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp" + dst.suffix)
    sf.write(str(tmp), audio, sr, subtype="PCM_16")
    os.replace(tmp, dst)
    return len(audio) / sr


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


def shard_job(src, repo, rel, budget_s, spk_cap, audio_dir, tmp_dir, seed, eval_n=0, keep_all=False, max_s=20.0,
              yg=None):
    """Download one parquet shard, keep a speaker-capped random subset up to budget_s
    (or eval_n clips), write wavs, delete the shard. Returns manifest rows.
    yg: incremental-mode options (dict: split_s, piece_min_s, vad, ext, rev, threads) or None (legacy)."""
    import pyarrow.parquet as pq

    tmp = Path(tmp_dir) / f"{src}-{Path(rel).parent.parent.name}-{Path(rel).name}" if yg else Path(tmp_dir) / f"{src}-{Path(rel).name}"
    D.curl(f"{HF}/datasets/{repo}/resolve/{(yg or {}).get('rev') or 'main'}/{rel}", tmp)
    split_s = (yg or {}).get("split_s") or 0
    n_long = n_pieces = n_drop = 0
    lost_s = 0.0
    rows, fails = [], 0
    try:
        pf = pq.ParquetFile(str(tmp))
        cols = pf.schema_arrow.names
        tc = pick(cols, ("normalized_text", "sentence") if src == "vp" else TEXT_COLS)
        sc, ic = pick(cols, SPK_COLS), pick(cols, ID_COLS)
        xc = [c for c in YG_EXTRA_COLS if c in cols] if yg else []
        meta = pf.read(columns=[c for c in (tc, sc, ic) if c] + xc).to_pylist()
        order = list(range(len(meta)))
        random.Random(f"{seed}-{rel}").shuffle(order)
        spk_n, want = {}, []
        for i in order:
            r = meta[i]
            if not (r[tc] or "").strip() or (not keep_all and not text_ok(r[tc])):
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
                fails += 1
                continue
            dur = len(arr) / sr
            if not (1.0 <= dur if split_s else 1.0 <= dur <= max_s):
                continue
            src_sr = sr
            arr, sr = resample16(arr, sr)   # CV mp3 is 32/48 kHz
            r = meta[i]
            uid = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(str(r.get(ic) or i)).name).rsplit(".", 1)[0]
            if yg:
                segs = [(0, len(arr))]
                if split_s and dur > split_s:   # cut at speech pauses; the teacher labels the pieces
                    sys.path.insert(0, str(Path(__file__).resolve().parent))
                    import prepare_yodas as Y
                    segs, dl = Y.pack_spans(Y.speech_spans(arr, sr, yg["vad"], yg["threads"]), sr,
                                            yg["piece_min_s"], split_s)
                    n_long += 1; n_pieces += len(segs); n_drop += dl
                    lost_s += dur - sum(b - a for a, b in segs) / sr
                for k, (c0, c1) in enumerate(segs):
                    piece = bool(split_s) and dur > split_s
                    p = Path(audio_dir) / src / f"{uid}_p{k:02d}.{yg['ext']}" if piece else Path(audio_dir) / src / f"{uid}.{yg['ext']}"
                    d = write_audio(p, arr[c0:c1], sr)
                    if yg["ext"] != "wav":
                        p.with_suffix(".wav").unlink(missing_ok=True)   # stale legacy copy of the same clip
                    row = {"audio_filepath": str(p), "duration": round(d, 3), "text": "" if piece else r[tc].strip(),
                           "speaker": f"{src}:{r.get(sc) or uid}", "corpus": src, "src_sr": src_sr,
                           "utt_id": str(r.get(ic) or uid), "shard": rel, "row": i, "offset": round(c0 / sr, 3),
                           "parent_dur": round(dur, 3), "revision": yg["rev"]}
                    row.update({YG_EXTRA_COLS[c]: r[c] for c in xc})
                    if piece:
                        row.update(parent_id=row["utt_id"], ref_parent=r[tc].strip(), piece=k, n_pieces=len(segs))
                    rows.append(row)
                    acc += d
                continue
            p = Path(audio_dir) / src / f"{uid}.wav"
            D.write_wav(p, arr, sr)
            rows.append({"audio_filepath": str(p), "duration": round(dur, 3), "text": r[tc].strip(),
                         "speaker": f"{src}:{r.get(sc) or uid}", "corpus": src, "src_sr": src_sr})
            acc += dur
            if (eval_n and len(rows) >= eval_n) or (not eval_n and acc >= budget_s):
                break
    finally:
        tmp.unlink(missing_ok=True)   # raw shard deleted as soon as it is extracted
    if split_s:
        print(f"  {src} {rel}: {n_long} clips > {split_s:g} s -> {n_pieces} pieces, {n_drop} pause-less spans "
              f"dropped, {lost_s / 3600:.2f} h not kept (silence + dropped spans)", flush=True)
    if fails:
        print(f"  {src} {rel}: {fails} decode failures", flush=True)
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
            arr, src_sr = D.decode_bytes(tar.extractfile(m).read())
            arr, sr = resample16(arr, src_sr)
            p = Path(audio_dir) / "fleurs" / (base.rsplit(".", 1)[0] + ".wav")
            dur = D.write_wav(p, arr, sr)
            r = by.pop(base)
            rows.append({"audio_filepath": str(p), "duration": round(dur, 3), "text": r["raw"].strip(),
                         "speaker": f"fleurs:{r['sid']}", "corpus": "fleurs", "src_sr": src_sr})
    return rows


def resample16(arr, sr):
    """-> (float32 mono 16 kHz, 16000). scipy polyphase: no torchaudio dependency."""
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != D.SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, D.SR)
        arr = resample_poly(arr, D.SR // g, sr // g)
    return arr.astype("float32"), D.SR


def spread_first(n):
    """0..n-1 ordered so that every prefix is spread over the range (bit-reversal order)."""
    b = max(1, math.ceil(math.log2(max(2, n))))
    return sorted(range(n), key=lambda i: int(f"{i:0{b}b}"[::-1], 2))


def yg_incremental(a, spread, audio, tmp, out):
    """Granary pool built shard by shard into <out>/yg_shards (restartable, growable); see the docstring."""
    dirs = [d for d in (a.yg_dirs or f"{a.lang}000").split(",") if d]
    rev = hf_revision(YG_REPO)
    lists = {d: list_files(YG_REPO, f"data/{d}/ast") for d in dirs}
    per = {d: [] for d in dirs}
    if spread and len(spread) == len(dirs):
        for d, n in zip(dirs, spread):
            fs = lists[d]
            per[d] = [fs[i] for i in sorted({round(i * (len(fs) - 1) / max(1, n - 1)) for i in range(min(n, len(fs)))})]
    elif spread:   # one N over the concatenated listing
        fs = [f for d in dirs for f in lists[d]]
        for i in sorted({round(i * (len(fs) - 1) / max(1, spread[0] - 1)) for i in range(spread[0])}):
            per[fs[i].split("/")[1]].append(fs[i])
    else:          # --yg-shards: i (first dir) or dir:i
        for x in a.yg_shards.split(","):
            d, i = x.split(":") if ":" in x else (dirs[0], x)
            per[d].append(lists[d][int(i)])
    if a.yg_exclude_spread:   # drop the shards other pools already use (same per-dir spread formula)
        for x in a.yg_exclude_spread.split(","):
            d, n = x.split(":")
            fs = lists.get(d) or list_files(YG_REPO, f"data/{d}/ast")
            used = {fs[i] for i in {round(i * (len(fs) - 1) / max(1, int(n) - 1)) for i in range(min(int(n), len(fs)))}}
            if d in per:
                k = len(per[d]); per[d] = [f for f in per[d] if f not in used]
                print(f"== yg: exclude {d}:{n} -> {k - len(per[d])} of {k} selected shards dropped", flush=True)
    cache = out / "yg_shards"
    cache.mkdir(parents=True, exist_ok=True)
    cpath = lambda rel: cache / f"{rel.split('/')[1]}-{Path(rel).stem}.json"  # noqa: E731
    rr = [[fs[j] for j in spread_first(len(fs))] for fs in per.values()]
    order = [rr[k][j] for j in range(max(map(len, rr), default=0)) for k in range(len(rr)) if j < len(rr[k])]
    budget = a.yg_hours * 3600 if a.yg_hours else float("inf")
    got, acc = {}, 0.0
    for rel in [r for r in order if cpath(r).exists()]:   # cached shards first: the pool grows as a superset
        if acc >= budget:
            break
        got[rel] = D.ftlib.read_manifest(cpath(rel)); acc += sum(x["duration"] for x in got[rel])
    print(f"== yg: dirs {','.join(dirs)}, revision {rev}, {sum(map(len, rr))} shards selected, {len(got)} cached "
          f"({acc / 3600:.1f} h), budget {a.yg_hours or '-'} h", flush=True)
    todo = iter([r for r in order if not cpath(r).exists()])
    opts = {"split_s": a.yg_split_long, "piece_min_s": a.yg_piece_min_s, "ext": a.audio_ext, "rev": rev,
            "vad": a.yg_vad, "threads": a.yg_vad_threads}
    if opts["vad"] == "auto":
        try:
            import silero_vad  # noqa: F401
            opts["vad"] = "silero"
        except ImportError:
            opts["vad"] = "energy"
    from concurrent.futures import FIRST_COMPLETED, wait
    live, t0, stop = {}, time.time(), ""
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        while True:
            while len(live) < a.workers and not stop:
                est = acc / max(1, len(got)) * len(live) if got else 0.0   # hours already in flight
                if acc + est >= budget:
                    stop = f"budget {a.yg_hours} h"; break
                if a.yg_min_free_gb and shutil.disk_usage(audio).free < a.yg_min_free_gb * 1e9:
                    stop = f"< {a.yg_min_free_gb} GB free on {audio}"; break
                rel = next(todo, None)
                if rel is None:
                    break
                live[ex.submit(shard_job, "yg", YG_REPO, rel, 1e12, 10 ** 9, str(audio), str(tmp), a.seed, 0,
                               a.yg_keep_all, a.yg_max_s, opts)] = rel
            if not live:
                break
            done, _ = wait(live, return_when=FIRST_COMPLETED)
            for fu in done:
                rel, r = fu.result()
                del live[fu]
                D.ftlib.write_manifest(cpath(rel), r)
                got[rel] = r; acc += sum(x["duration"] for x in r)
                print(f"  yg {rel}: {len(r)} utts {sum(x['duration'] for x in r) / 3600:.2f} h, pool {acc / 3600:.1f} h "
                      f"({time.time() - t0:.0f}s)", flush=True)
    if stop:
        print(f"== yg: stopped admitting shards: {stop}", flush=True)
    rows = [x for rel in order if rel in got for x in got[rel]]
    D.ftlib.write_manifest(out / "train_yg.json", rows)
    np_ = sum(1 for x in rows if "parent_id" in x)
    print(f"== yg: {len(rows)} utts ({np_} pieces) {sum(x['duration'] for x in rows) / 3600:.2f} h from {len(got)} shards, "
          f"{len({x['speaker'] for x in rows})} videos", flush=True)


def selftest():
    """Provisioning gate: encode a 48 kHz tone as MP3 (what Common Voice ships), decode it
    with the kit's decoder, resample to 16 kHz, check length and pitch; write it as wav."""
    import io
    import tempfile
    import numpy as np
    import soundfile as sf
    t = np.arange(48000) / 48000.0
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    buf = io.BytesIO(); sf.write(buf, x, 48000, format="MP3"); b = buf.getvalue()
    arr, sr = D.decode_bytes(b)
    assert sr == 48000, sr
    y, sr2 = resample16(arr, sr)
    assert sr2 == 16000 and abs(len(y) - 16000) < 1200, len(y)
    f = np.abs(np.fft.rfft(y[2000:14000])).argmax() * 16000 / 12000
    assert abs(f - 440) < 5, f
    with tempfile.TemporaryDirectory() as d:
        dur = D.write_wav(Path(d) / "t.wav", y, sr2)
    print(f"prepare_mix selftest OK: mp3 48 kHz -> {len(y)} samples @16 kHz, tone {f:.1f} Hz, wav {dur:.2f} s")


def main():
    if "--selftest" in sys.argv:
        selftest(); return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=sorted(FLEURS_CFG))
    ap.add_argument("--cv-hours", type=float, default=40)
    ap.add_argument("--vp-hours", type=float, default=40)
    ap.add_argument("--cv-spk", type=int, default=20)
    ap.add_argument("--vp-spk", type=int, default=60)
    ap.add_argument("--no-fleurs", action="store_true")
    ap.add_argument("--yg-shards", default="", help="yodas-granary <lang>000/ast shard indices (e.g. 0,40,80): "
                    "every 1-20 s clip, speaker = video -> train_yg.json (Granary Whisper text kept as `text`)")
    ap.add_argument("--yg-keep-all", action="store_true", help="audit: keep rows with digits / non-latin text")
    ap.add_argument("--yg-max-s", type=float, default=20.0, help="Granary clips average ~19.5 s: 20 keeps < 1/3 of them")
    ap.add_argument("--yg-spread", default="", help="instead of --yg-shards: N shards evenly spread over the language "
                    "(incremental mode: one N per --yg-dirs entry, or one N over all of them)")
    ap.add_argument("--yg-dirs", default="", help="incremental mode: Granary dirs, e.g. it000,it100,it101 (default <lang>000)")
    ap.add_argument("--yg-exclude-spread", default="", help="incremental mode: dir:N,... shards to skip = the spread "
                    "selection of other pools (e.g. it000:70,it100:60,it101:55 = it-yg + it-yg6): a disjoint new pool")
    ap.add_argument("--yg-split-long", type=float, default=0, help="incremental mode: clips > S s cut at pauses into pieces <= S s")
    ap.add_argument("--yg-piece-min-s", type=float, default=2.0, help="shortest piece kept by --yg-split-long")
    ap.add_argument("--yg-vad", default="auto", choices=("auto", "silero", "energy"), help="pause detector of --yg-split-long "
                    "(auto: silero if installed)")
    ap.add_argument("--yg-vad-threads", type=int, default=1, help="torch threads per worker for silero")
    ap.add_argument("--yg-hours", type=float, default=0, help="incremental mode: stop starting shards once the pool has H h")
    ap.add_argument("--yg-min-free-gb", type=float, default=0, help="incremental mode: stop starting shards below G GB free on --audio")
    ap.add_argument("--audio-ext", default="wav", choices=("wav", "flac"), help="yg audio container (flac: incremental mode)")
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
    spread = [int(x) for x in a.yg_spread.split(",") if x.strip()]
    yg_new = bool(a.yg_dirs or a.yg_split_long or a.audio_ext != "wav" or a.yg_hours or a.yg_min_free_gb)
    if (a.yg_shards or any(spread)) and yg_new:
        yg_incremental(a, spread, audio, tmp, out)
    elif (a.yg_shards or any(spread)) and not (out / "train_yg.json").exists():
        files = list_files(YG_REPO, f"data/{a.lang}000/ast")
        yg_spread = spread[0] if spread else 0
        idx = ([round(i * (len(files) - 1) / max(1, yg_spread - 1)) for i in range(yg_spread)] if yg_spread
               else [int(i) for i in a.yg_shards.split(",")])
        pick = [files[i] for i in sorted(set(idx))]
        rows = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(shard_job, "yg", YG_REPO, f, 1e12, 10 ** 9, str(audio), str(tmp), a.seed, 0, a.yg_keep_all, a.yg_max_s) for f in pick]
            for fu in futs:
                rel, r = fu.result()
                rows += r
                print(f"  yg {rel}: {len(r)} utts {sum(x['duration'] for x in r) / 3600:.2f} h", flush=True)
        D.ftlib.write_manifest(out / "train_yg.json", rows)
        print(f"== yg: {len(rows)} utts {sum(x['duration'] for x in rows) / 3600:.2f} h, {len({x['speaker'] for x in rows})} videos", flush=True)
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
