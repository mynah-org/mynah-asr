#!/usr/bin/env python3
"""fetch_stress_bank.py - build samples/stress-en/, the English stress bank of the streaming load harness.

    cd tools && uv run python fetch_stress_bank.py --dry-run   # the plan only: TSVs, no audio
    cd tools && uv run python fetch_stress_bank.py             # the default bank (~640 MB)
    cd tools && uv run python fetch_stress_bank.py --per-class 64   # a laptop-sized bank

WHAT IT IS.  The SOAK path of `tools/bench/stream_load.py` stratifies its schedule over
duration classes: `classify()` sorts a clip into `short` / `medium` / `long` with the
boundaries of `--class-bounds` (default `8,20`: short < 8 s <= medium < 20 s <= long), and
`stratified_schedule()` then plays one clip of each class per round.  `samples/en/` holds
three clips; three clips cannot fill three classes for a 10-minute soak.  This tool builds
a bank that can, from the source and with the mechanism the repository already uses for
`samples/`: FLEURS (google/fleurs), CC-BY 4.0, TSV plus the audio tarball streamed out of
huggingface with `curl | tar`, rewritten as 16 kHz mono PCM16 - the only format the
harness accepts (`load_pcm()`).  Offline tooling: nothing at runtime reads this bank.

HOW "ENOUGH" WAS COMPUTED (the default `--per-class`).  The target is a 10-minute SOAK at
concurrency 32 in which no clip is played twice anywhere in the fleet.  The schedule is a
single list: stream `i` walks positions `i, i+32, i+64, ...` of it (`stream_main()`), class
`p mod 3` owns position `p`, and the j-th draw of a class takes pool entry `j mod P`.  So
the pool P must cover the DEEPEST draw index the fleet reaches, which is set by the fastest
stream, not by the average one.  A stream plays back to back at 1x pacing, spending
(audio + ~0.4 s of connect and finalize) per utterance:

    utt_s     = (mean(short) + mean(medium) + mean(long)) / 3 + 0.4 s
              = (6.6 + 11.4 + 22.0) / 3 + 0.4 = 13.73 s per utterance
    rounds    = streams * seconds / (3 * utt_s) = 32 * 600 / 41.2 = 466 draws per class
    per_class = ceil(466 * 1.08 * 1.05) = 529 clips per class

The 1.08 is the fastest stream against the mean one (a simulation of the real
`build_bank()` + `stratified_schedule()` over this bank gives k_max = 47 utterances against
a mean of 44.7); the 1.05 absorbs the error of the class means themselves.  That simulation
is the check behind the number: at 490 clips per class exactly one clip is played twice, at
510 and above none is.  The class means are measured over the 3,555 en_us recordings listed
in the FLEURS TSVs (short 6.58 s, medium 11.35 s); the long class is built to ~21.5 s
(below).  `--per-class` overrides the number, `--soak-streams` / `--soak-seconds` re-derive
it for a different target run.

COST.  At the default: 3 * 529 clips, ~21,000 s of audio, ~640 MB of PCM16 on disk, and
~1.8 GB of one-time transfer (the three en_us tarballs are streamed, never stored).  The
peak also holds one split's float32 sources in the cache; the projected bank, cache and
free space are printed and checked before anything is downloaded (`--max-bank-mb`).

WHY THE LONG CLASS IS SYNTHESISED.  FLEURS is a read-sentence corpus: of the 3,555 en_us
recordings only 68 reach 20 s, so the `long` class cannot be filled with natural clips.
Those 68 are used first; the rest are built exactly the way `fetch_fleurs_samples.py`
builds `samples/en/fleurs_long.wav` - unused recordings of the same split concatenated with
0.6 s of digital silence between them, until the clip passes `--long-min` (21 s, just over
the 20 s boundary, so the harness classifies it as `long` whatever rounding it applies).
Every clip of the bank is therefore distinct audio: a recording is used by exactly one clip.

DETERMINISM.  One seed (`--seed`, default 42, the harness's own default) drives a single
shuffle of the TSV records, which are read in a fixed order, so the plan - which recording
goes in which clip, under which name - is a pure function of (seed, per_class, splits,
bounds, long_min, pause).  Float32 -> PCM16 is done here (round-half-away-from-zero, clip
to int16) rather than by libsndfile, so the bytes do not depend on the writer's version.
The manifest records the seed and a `bank_hash` over the sorted per-clip sha256 values, so
a run records WHICH bank it used in one field.

IDEMPOTENT AND RESUMABLE.  A clip already on disk whose sha256 matches the manifest is not
rebuilt and its sources are not downloaded; a re-run with the same arguments touches no
audio at all.  A tarball is streamed only when some clip of that split is actually missing.

LEVELS ARE NOT TOUCHED.  FLEURS recordings differ widely in level - six dev clips fetched
while writing this tool peaked between -44.8 and -8.1 dBFS - and Nemotron's front end does
no per-utterance normalisation (`mel normalize "NA"`), so a quiet clip is a quiet clip for
the model too.  Nothing here normalises: the bank keeps the audio as FLEURS published it,
records `peak_dbfs` / `rms_dbfs` per clip, and reports how many clips peak under -30 dBFS.  A CER gate that wants a level floor filters on those fields; a
throughput SOAK does not care.

WHAT THIS BANK IS NOT.  Read speech, one sentence at a time, studio-clean, read by
volunteers from Wikipedia text.  It stresses the server's scheduling, not its robustness:
it contains no conversational or spontaneous speech, no disfluencies, no overlap, no
telephony or far-field channel, no background noise.  No claim about conversational ASR
can be made from it - that is a known gap, stated again in samples/stress-en/README.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

TOOL = "tools/fetch_stress_bank.py"
TOOL_VERSION = "1.0"

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
CACHE = Path(os.environ.get("MYNAH_FLEURS_CACHE", "/tmp/fleurs-stress"))

SOURCE = "FLEURS (google/fleurs) - Conneau et al. 2022, CC-BY 4.0"
URL = "https://huggingface.co/datasets/google/fleurs"
LICENCE = "CC-BY 4.0"

SR = 16000                      # the only rate the runtime and the harness accept
BYTES_PER_S = SR * 2            # PCM16 mono
SRC_BYTES_PER_S = SR * 4        # FLEURS ships float32 WAV
CLASSES = ("short", "medium", "long")
CLASS_BOUNDS = (8.0, 20.0)      # == stream_load.py --class-bounds default; do not invent
SOAK_STREAMS, SOAK_SECONDS = 32, 600.0      # the run the default bank must cover
UTT_OVERHEAD_S = 0.4            # connect + finalize, outside the audio, per utterance
CLASS_MEAN_S = {"short": 6.6, "medium": 11.4, "long": 22.0}
STREAM_SPREAD = 1.08            # the fastest stream reaches deeper into a pool than the mean one
SAFETY = 1.05                   # error of the class means themselves
LONG_MIN_S = 21.0               # > 20 s bound with margin
PAUSE_S = 0.6                   # as in fetch_fleurs_samples.py's long fixtures
DISK_MARGIN_BYTES = 512 << 20


def default_per_class(streams: float = SOAK_STREAMS, seconds: float = SOAK_SECONDS) -> int:
    """Clips per class so that a `streams` x `seconds` soak never replays a clip.  See header."""
    utt_s = sum(CLASS_MEAN_S.values()) / len(CLASSES) + UTT_OVERHEAD_S
    return math.ceil(streams * seconds / (len(CLASSES) * utt_s) * STREAM_SPREAD * SAFETY)


# --------------------------------------------------------------------------- FLEURS records

def load_records(cfg: str, splits: "list[str]") -> "list[dict]":
    """Every recording of the requested splits, in an order that does not depend on the TSV."""
    recs: list[dict] = []
    for split in splits:
        path = CACHE / f"{cfg}_{split}.tsv"
        if not path.exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "-sSL", f"{BASE}/{cfg}/{split}.tsv", "-o", str(path)],
                           check=True)
        seen: set[str] = set()
        with open(path, encoding="utf-8") as f:
            for r in csv.reader(f, delimiter="\t"):
                if len(r) < 7 or r[1] in seen:
                    continue
                seen.add(r[1])
                recs.append({"split": split, "fleurs_id": int(r[0]), "source_file": r[1],
                             "duration_sec": int(r[5]) / float(SR), "text": r[2],
                             "text_norm": r[3], "gender": r[6].strip().lower()})
    recs.sort(key=lambda r: (r["split"], r["fleurs_id"], r["source_file"]))
    return recs


# ------------------------------------------------------------------------------- the plan

def _natural(cls: str, seq: int, r: dict) -> dict:
    return {"file": f"{cls}/{cls}_{seq:04d}_f{r['fleurs_id']}.wav", "class": cls,
            "duration_sec": round(r["duration_sec"], 3), "sample_rate": SR,
            "text": r["text"], "text_norm": r["text_norm"], "fleurs_id": r["fleurs_id"],
            "fleurs_ids": [r["fleurs_id"]], "split": r["split"], "composed": False,
            "gender": r["gender"], "parts": [dict(r)]}


def _composed(seq: int, grp: "list[dict]", total: float, pause: float) -> dict:
    return {"file": f"long/long_{seq:04d}_cat{len(grp)}.wav", "class": "long",
            "duration_sec": round(total, 3), "sample_rate": SR,
            "text": " ".join(p["text"] for p in grp),
            "text_norm": " ".join(p["text_norm"] for p in grp),
            "fleurs_id": grp[0]["fleurs_id"], "fleurs_ids": [p["fleurs_id"] for p in grp],
            "split": grp[0]["split"], "composed": True, "pause_sec": pause,
            "gender": ",".join(sorted({p["gender"] for p in grp})),
            "parts": [dict(p) for p in grp]}


def take_group(queue: "list[dict]", long_min: float, pause: float):
    """Pop recordings off `queue` until the concatenation passes `long_min`.

    First-fit-just-above: the partner is the shortest recording that reaches the target, so
    a long clip lands a little over the bound instead of far past it (bank size is audio
    seconds).  Ties break on the file name, so the choice does not depend on list order."""
    if not queue:
        return None, 0.0
    grp = [queue.pop(0)]
    total = grp[0]["duration_sec"]
    while total < long_min:
        need = long_min - total - pause
        fits = [r for r in queue if r["duration_sec"] >= need]
        if fits:
            pick = min(fits, key=lambda r: (r["duration_sec"], r["source_file"]))
        elif queue:
            pick = max(queue, key=lambda r: (r["duration_sec"], r["source_file"]))
        else:                                   # not enough material left: undo and stop
            queue[0:0] = grp
            return None, 0.0
        queue.remove(pick)
        total += pause + pick["duration_sec"]
        grp.append(pick)
    return grp, total


def build_plan(recs: "list[dict]", per_class: int, seed: int, bounds, long_min: float,
               pause: float) -> "list[dict]":
    """The whole bank as clip descriptors.  Pure: no I/O, no audio, fully determined by the
    arguments.  A recording is used by exactly one clip."""
    pool = list(recs)
    random.Random(seed).shuffle(pool)
    used: set[str] = set()
    plan: list[dict] = []

    for cls, lo, hi in (("short", 0.0, bounds[0]), ("medium", bounds[0], bounds[1])):
        picked = [r for r in pool if lo <= r["duration_sec"] < hi][:per_class]
        if len(picked) < per_class:
            raise SystemExit(f"{cls}: only {len(picked)} FLEURS recordings in [{lo},{hi}) s, "
                             f"{per_class} needed - add a split (--splits) or lower --per-class")
        for seq, r in enumerate(picked, 1):
            used.add(r["source_file"])
            plan.append(_natural(cls, seq, r))

    natural_long = [r for r in pool if r["duration_sec"] >= bounds[1]][:per_class]
    for seq, r in enumerate(natural_long, 1):
        used.add(r["source_file"])
        plan.append(_natural("long", seq, r))

    seq = len(natural_long)
    queues: dict[str, list[dict]] = {}
    for r in pool:                              # leftovers, grouped by split: a composed clip
        if r["source_file"] not in used:        # never spans splits, so one tarball at a time
            queues.setdefault(r["split"], []).append(r)
    for split in sorted(queues, key=lambda s: (-len(queues[s]), s)):
        queue = queues[split]
        while seq < per_class:
            grp, total = take_group(queue, long_min, pause)
            if grp is None:
                break
            seq += 1
            plan.append(_composed(seq, grp, total, pause))
    if seq < per_class:
        raise SystemExit(f"long: only {seq} clips of >= {long_min} s could be built from the "
                         f"unused recordings, {per_class} needed - add a split (--splits), "
                         f"lower --per-class, or lower --long-min")
    plan.sort(key=lambda c: c["file"])
    return plan


# ------------------------------------------------------------------------------- the audio

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def levels(pcm) -> "tuple[float, float]":
    """(peak dBFS, rms dBFS) of int16 samples; silence reports -inf as -99.0."""
    import numpy as np
    x = np.asarray(pcm, dtype="float64") / 32768.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    to_db = lambda v: round(20.0 * math.log10(v), 1) if v > 0 else -99.0
    return to_db(peak), to_db(rms)


def levels_of_wav(path: Path) -> "tuple[float, float]":
    import numpy as np
    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
    return levels(np.frombuffer(raw, dtype="<i2"))


def extract(cfg: str, split: str, files: "list[str]", dest: Path) -> None:
    """Stream <split>.tar.gz and take only the wanted members (exact names, no wildcards:
    GNU tar and bsdtar treat them the same way).  The tarball is never stored."""
    want = [f for f in files if not (dest / split / f).exists()]
    if not want:
        return
    dest.mkdir(parents=True, exist_ok=True)
    fd, listfile = tempfile.mkstemp(suffix=".txt", prefix="fleurs-members-")
    with os.fdopen(fd, "w") as f:
        f.write("".join(f"{split}/{name}\n" for name in want))
    print(f"  streaming {cfg}/{split}.tar.gz for {len(want)} member(s)...", flush=True)
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
    missing = [f for f in want if not (dest / split / f).exists()]
    if missing:
        raise SystemExit(f"{cfg}/{split}: {len(missing)} member(s) not in the archive, "
                         f"e.g. {missing[0]}")


def read_f32(path: Path):
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz, expected {SR}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype="float32")


def to_pcm16(x):
    """float32 -> int16 here, not in libsndfile: the bytes (and so the sha256, and so the
    bank_hash) must not depend on the writer's version."""
    import numpy as np
    return np.clip(np.rint(np.asarray(x, dtype="float64") * 32767.0), -32768, 32767).astype("<i2")


def render(clip: dict, src_root: Path, out_root: Path) -> None:
    import numpy as np
    pieces = []
    for i, part in enumerate(clip["parts"]):
        if i:
            pieces.append(np.zeros(int(round(clip.get("pause_sec", PAUSE_S) * SR)), dtype="float32"))
        pieces.append(read_f32(src_root / part["split"] / part["source_file"]))
    pcm = to_pcm16(np.concatenate(pieces) if len(pieces) > 1 else pieces[0])
    out = out_root / clip["file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    with wave.open(str(tmp)) as w:               # the harness refuses anything else
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
            raise SystemExit(f"{out}: not 16 kHz mono s16")
        frames = w.getnframes()
    tmp.replace(out)
    clip["duration_sec"] = round(frames / float(SR), 3)
    clip["peak_dbfs"], clip["rms_dbfs"] = levels(pcm)


# -------------------------------------------------------------------------------- manifest

def bank_hash(clips: "list[dict]") -> str:
    h = hashlib.sha256()
    for digest in sorted(c["sha256"] for c in clips):
        h.update(digest.encode())
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


def manifest_entry(clip: dict) -> dict:
    entry = {"file": clip["file"], "class": clip["class"], "sha256": clip["sha256"],
             "duration_sec": clip["duration_sec"], "sample_rate": clip["sample_rate"],
             "text": clip["text"], "text_norm": clip["text_norm"],
             "fleurs_id": clip["fleurs_id"], "fleurs_ids": clip["fleurs_ids"],
             "split": clip["split"], "gender": clip["gender"], "composed": clip["composed"],
             "peak_dbfs": clip["peak_dbfs"], "rms_dbfs": clip["rms_dbfs"]}
    if clip["composed"]:
        entry["pause_sec"] = clip["pause_sec"]
        entry["parts"] = [{"fleurs_id": p["fleurs_id"], "split": p["split"],
                           "source_file": p["source_file"],
                           "duration_sec": round(p["duration_sec"], 3)} for p in clip["parts"]]
    else:
        entry["source_file"] = clip["parts"][0]["source_file"]
    return entry


def write_manifest(path: Path, clips: "list[dict]", a, splits, per_class) -> dict:
    by_class = {c: [k for k in clips if k["class"] == c] for c in CLASSES}
    audio_s = sum(k["duration_sec"] for k in clips)
    manifest = {
        "bank": f"stress-{a.lang}",
        "language": a.lang,
        "fleurs_config": a.config,
        "purpose": f"stratified {a.lang} bank for tools/bench/stream_load.py (SOAK/WAVE)",
        "source": SOURCE,
        "url": URL,
        "licence": LICENCE,
        "attribution": "FLEURS, Google Research (CC-BY 4.0) - redistributed unmodified in "
                       "content, re-encoded to 16 kHz mono PCM16; long clips are "
                       "concatenations of FLEURS recordings, see composed/parts",
        "read_speech_only": True,
        "not_covered": "read sentences only: no conversational or spontaneous speech, no "
                       "disfluencies, no overlap, no noisy or far-field channel",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "config": a.config,
        "lang": a.lang,
        "splits": splits,
        "seed": a.seed,
        "per_class": per_class,
        "class_bounds_s": list(a.bounds),
        "long_min_s": a.long_min,
        "pause_sec": a.pause,
        "soak_profile": {"streams": a.soak_streams, "seconds": a.soak_seconds,
                         "utt_overhead_s": UTT_OVERHEAD_S, "stream_spread": STREAM_SPREAD,
                         "safety": SAFETY},
        "bank_hash": bank_hash(clips),
        "totals": {"clips": len(clips), "audio_sec": round(audio_s, 1),
                   "bytes": sum(k["bytes"] for k in clips),
                   "per_class": {c: len(v) for c, v in by_class.items()},
                   "mean_sec_per_class": {c: round(sum(x["duration_sec"] for x in v) / len(v), 2)
                                          for c, v in by_class.items() if v},
                   "clips_below_-30_dbfs_peak": sum(1 for k in clips if k["peak_dbfs"] < -30.0)},
        "clips": [manifest_entry(k) for k in sorted(clips, key=lambda k: k["file"])],
    }
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


# ------------------------------------------------------------------------------------ main

def summarise(plan: "list[dict]") -> str:
    out = []
    for cls in CLASSES:
        v = [c["duration_sec"] for c in plan if c["class"] == cls]
        comp = sum(1 for c in plan if c["class"] == cls and c["composed"])
        out.append(f"  {cls:<6} {len(v):4d} clips  mean {sum(v)/len(v):5.1f} s  "
                   f"min {min(v):5.1f}  max {max(v):5.1f}  total {sum(v)/60:6.1f} min"
                   + (f"  ({comp} composed)" if comp else ""))
    total = sum(c["duration_sec"] for c in plan)
    out.append(f"  bank   {len(plan):4d} clips  {total/60:.1f} min of audio  "
               f"{total*BYTES_PER_S/(1<<20):.0f} MiB of PCM16")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "samples" / "stress-en"),
                    help="bank directory (default samples/stress-en)")
    ap.add_argument("--config", default="en_us", help="FLEURS config (default en_us)")
    ap.add_argument("--lang", default="en", help="language tag written into the manifest")
    ap.add_argument("--splits", default="train,dev,test",
                    help="FLEURS splits to draw from; dev,test alone cannot fill the short "
                         "class at the default --per-class")
    ap.add_argument("--per-class", type=int, default=None,
                    help="clips per duration class (default: computed for the soak profile)")
    ap.add_argument("--soak-streams", type=float, default=SOAK_STREAMS,
                    help="soak concurrency the bank must cover (default 32)")
    ap.add_argument("--soak-seconds", type=float, default=SOAK_SECONDS,
                    help="soak duration the bank must cover, seconds (default 600)")
    ap.add_argument("--seed", type=int, default=42, help="selection seed (harness default: 42)")
    ap.add_argument("--class-bounds", default="8,20",
                    help="short/medium and medium/long boundaries, seconds; must match the "
                         "stream_load.py --class-bounds of the run")
    ap.add_argument("--long-min", type=float, default=LONG_MIN_S,
                    help="target length of a composed long clip, seconds")
    ap.add_argument("--pause", type=float, default=PAUSE_S,
                    help="silence between the parts of a composed clip, seconds")
    ap.add_argument("--max-bank-mb", type=float, default=1024.0,
                    help="refuse a plan whose bank would exceed this (preflight, not a limit "
                         "on what the harness can use)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit: downloads the TSVs only, no audio")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the extracted float32 sources in the cache directory")
    ap.add_argument("--prune", action="store_true",
                    help="delete wavs of the bank directory that the plan does not contain")
    a = ap.parse_args()

    try:
        b0, b1 = (float(x) for x in a.class_bounds.split(","))
    except ValueError:
        raise SystemExit("--class-bounds wants two seconds, e.g. 8,20")
    a.bounds = (b0, b1)
    if (b0, b1) != CLASS_BOUNDS:
        print(f"note: --class-bounds {b0},{b1} differs from the stream_load.py default "
              f"{CLASS_BOUNDS[0]},{CLASS_BOUNDS[1]}: the run must pass the same bounds")
    if a.long_min <= b1:
        raise SystemExit(f"--long-min {a.long_min} must exceed the medium/long bound {b1}")
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    per_class = a.per_class or default_per_class(a.soak_streams, a.soak_seconds)
    out_root = Path(a.out)

    recs = load_records(a.config, splits)
    plan = build_plan(recs, per_class, a.seed, a.bounds, a.long_min, a.pause)
    bank_bytes = sum(c["duration_sec"] for c in plan) * BYTES_PER_S
    print(f"plan: {a.config} {'+'.join(splits)}, seed {a.seed}, {per_class} clips per class "
          f"(soak {a.soak_streams:g} x {a.soak_seconds:g} s)")
    print(summarise(plan))
    if bank_bytes > a.max_bank_mb * (1 << 20):
        raise SystemExit(f"refusing: the bank would be {bank_bytes/(1<<20):.0f} MiB > "
                         f"--max-bank-mb {a.max_bank_mb:g} (lower --per-class)")
    if a.dry_run:
        print("dry run: no audio fetched")
        return

    # what is already correct on disk?  sha256 against the previous manifest, nothing else.
    out_root.mkdir(parents=True, exist_ok=True)
    mpath = out_root / "manifest.json"
    known: dict[str, dict] = {}
    if mpath.exists():
        try:
            for e in json.loads(mpath.read_text()).get("clips", []):
                known[e["file"]] = e
        except (ValueError, KeyError):
            print("note: the existing manifest is unreadable, rebuilding the bank")
    todo = []
    for clip in plan:
        path = out_root / clip["file"]
        prev = known.get(clip["file"], {})
        digest = prev.get("sha256")
        if path.exists() and digest and sha256_file(path) == digest:
            clip["sha256"] = digest
            clip["bytes"] = path.stat().st_size
            if "peak_dbfs" in prev and "rms_dbfs" in prev:
                clip["peak_dbfs"], clip["rms_dbfs"] = prev["peak_dbfs"], prev["rms_dbfs"]
            else:                               # a manifest written before levels existed
                clip["peak_dbfs"], clip["rms_dbfs"] = levels_of_wav(path)
        else:
            todo.append(clip)
    print(f"{len(plan) - len(todo)} clip(s) already correct, {len(todo)} to build")

    if todo:
        need: dict[str, list[str]] = {}
        for clip in todo:
            for part in clip["parts"]:
                need.setdefault(part["split"], []).append(part["source_file"])
        cache_peak = max(sum(p["duration_sec"] for c in todo for p in c["parts"]
                             if p["split"] == s) for s in need) * SRC_BYTES_PER_S
        todo_bytes = sum(c["duration_sec"] for c in todo) * BYTES_PER_S
        free = shutil.disk_usage(out_root).free
        print(f"disk: {todo_bytes/(1<<20):.0f} MiB of bank + {cache_peak/(1<<20):.0f} MiB of "
              f"cache peak, {free/(1<<20):.0f} MiB free")
        if free < todo_bytes + cache_peak + DISK_MARGIN_BYTES:
            raise SystemExit("refusing: not enough free space for the bank, the extraction "
                             "cache and a 512 MiB margin (lower --per-class)")
        src_root = CACHE / a.config
        for split in sorted(need):
            extract(a.config, split, sorted(set(need[split])), src_root)
            built = 0
            for clip in todo:
                if clip["split"] != split:
                    continue
                render(clip, src_root, out_root)
                path = out_root / clip["file"]
                clip["sha256"] = sha256_file(path)
                clip["bytes"] = path.stat().st_size
                built += 1
            print(f"  {split}: {built} clip(s) written")
            if not a.keep_cache:
                for name in set(need[split]):
                    (src_root / split / name).unlink(missing_ok=True)

    planned = {c["file"] for c in plan}
    stale = sorted(str(p.relative_to(out_root)) for p in out_root.rglob("*.wav")
                   if str(p.relative_to(out_root)) not in planned)
    for name in stale:
        if a.prune:
            (out_root / name).unlink()
        else:
            print(f"note: {name} is not in the plan (--prune to delete it)")

    manifest = write_manifest(mpath, plan, a, splits, per_class)
    t = manifest["totals"]
    print(f"OK {out_root}: {t['clips']} clips, {t['audio_sec']/60:.1f} min, "
          f"{t['bytes']/(1<<20):.0f} MiB, bank_hash {manifest['bank_hash'][:23]}...")
    quiet = t["clips_below_-30_dbfs_peak"]
    if quiet:
        print(f"note: {quiet} clip(s) peak below -30 dBFS - FLEURS levels vary by ~35 dB and "
              f"nothing here normalises them; filter on peak_dbfs for a CER gate")
    print(f"use: python3 tools/bench/stream_load.py --mode soak --streams 32 --duration 600 "
          f"--bank short,medium,long --class-bounds {a.bounds[0]:g},{a.bounds[1]:g} "
          f"--seed {a.seed} --clips {out_root}/*/*.wav ...")


if __name__ == "__main__":
    sys.exit(main())
