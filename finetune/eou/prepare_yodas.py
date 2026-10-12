#!/usr/bin/env python3
"""YODAS2 long-form audio -> VAD-cut utterance clips for teacher pseudo-labelling.

    $PY prepare_yodas.py --lang it --shards 0,1 [--min-s 3 --max-s 20] [--audio /dev/shm/yodas/it] [--out /root/ft/mix/it/yodas]

espnet/yodas2 (CC-BY 3.0) data/<lang>000/audio/<NNNNNNNN>.tar.gz holds whole YouTube videos
(24 kHz mono wav, ~87 per shard, ~11.6 h); data/<lang>000/text/<NNNNNNNN>.json holds the
video's subtitles WITHOUT per-segment times, so they can only be checked per video. Per
shard (one at a time, transient): Silero VAD speech spans (min silence 300 ms), consecutive
spans packed into clips of --min-s..--max-s s cut only inside silences (a single span
longer than --max-s is dropped, never cut mid-speech), 16 kHz PCM16 wav. Manifest rows
carry an EMPTY text (the teacher fills it: teacher_ws.py --lang auto), speaker
yodas:<video>, video, start/end, plus a per-video file subs_<shard>.json with the subtitle
text for the video-level check. Nothing is filtered here.

speech_spans() and pack_spans() are shared with prepare_mix.py --yg-split-long (Granary clips
longer than S cut at silences).
"""
import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HF = "https://huggingface.co/datasets/espnet/yodas2/resolve/main/data"
_VAD = None


def energy_spans(x, sr, min_sil_ms=300, pad_ms=100, min_sp_ms=250):
    """Torch-free fallback of speech_spans: 20 ms frame energy above max(min(p10 + 10, p95 - 10), p95 - 35) dB,
    runs closer than min_sil_ms merged, padded by pad_ms, runs < min_sp_ms dropped."""
    hop = sr // 50
    n = len(x) // hop
    if n == 0:
        return []
    e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop).astype("float64") ** 2, axis=1) + 1e-10)
    p10, p95 = np.percentile(e, 10), np.percentile(e, 95)
    thr = max(min(p10 + 10, p95 - 10), p95 - 35)   # p95 - 10: a clip with (almost) no pause
    on = np.flatnonzero(e > thr)
    if not len(on):
        return []
    runs, gap = [], max(1, min_sil_ms // 20)
    s0 = p = on[0]
    for f in on[1:]:
        if f - p > gap:
            runs.append((s0, p + 1)); s0 = f
        p = f
    runs.append((s0, p + 1))
    pad = pad_ms * sr // 1000
    return [{"start": max(0, a * hop - pad), "end": min(len(x), b * hop + pad)} for a, b in runs
            if (b - a) * 20 >= min_sp_ms]


def speech_spans(x, sr, kind="silero", threads=0):
    """-> [{"start", "end"}] speech spans in samples. silero: Silero VAD, min silence 300 ms, pad 100 ms
    (this script's setting); energy: energy_spans (no torch, ~free)."""
    global _VAD
    if kind == "energy":
        return energy_spans(x, sr)
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    if threads:
        torch.set_num_threads(threads)
    if _VAD is None:
        _VAD = load_silero_vad()
    return get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), _VAD, sampling_rate=sr,
                                 min_silence_duration_ms=300, speech_pad_ms=100)


def pack_spans(spans, sr, min_s, max_s):
    """Consecutive spans packed into clips of min_s..max_s s, cut only inside silences; a single span
    longer than max_s is dropped (never cut mid-speech). -> ([(start, end)] samples, n dropped spans)."""
    clips, dropped = [], 0

    def flush(cur):
        if cur is not None and (cur[1] - cur[0]) / sr >= min_s:
            clips.append((cur[0], cur[1]))

    cur = None   # [start, end] samples of the clip being packed
    for sp in spans:
        if (sp["end"] - sp["start"]) / sr > max_s:
            flush(cur); cur = None; dropped += 1
        elif cur is None:
            cur = [sp["start"], sp["end"]]
        elif (sp["end"] - cur[0]) / sr <= max_s:
            cur[1] = sp["end"]
        else:
            flush(cur); cur = [sp["start"], sp["end"]]
    flush(cur)
    return clips, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--shards", default="0")
    ap.add_argument("--min-s", type=float, default=3.0)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--audio", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "canary"))
    import prepare_it as D  # curl (HTTP/1.1 + retries)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_mix import resample16

    torch.set_num_threads(8)
    audio = Path(a.audio or f"/dev/shm/yodas/{a.lang}"); out = Path(a.out or f"/root/ft/mix/{a.lang}/yodas")
    audio.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    for sh in (int(x) for x in a.shards.split(",")):
        name = f"{sh:08d}"
        man = out / f"clips_{name}.json"
        if man.exists():
            print(f"== skip shard {name}"); continue
        t0 = time.time(); tmp = audio / f"{name}.tar.gz"
        D.curl(f"{HF}/{a.lang}000/audio/{name}.tar.gz", tmp)
        D.curl(f"{HF}/{a.lang}000/text/{name}.json", out / f"subs_{name}.json")
        rows, vids, speech_s, total_s, dropped_long = [], 0, 0.0, 0.0, 0
        with tarfile.open(tmp, "r:gz") as tar:
            for m in tar:
                if not m.isfile() or not m.name.endswith(".wav"):
                    continue
                vid = Path(m.name).stem
                x, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
                x, sr = resample16(np.asarray(x), sr)
                vids += 1; total_s += len(x) / sr
                spans = speech_spans(x, sr)
                speech_s += sum(s["end"] - s["start"] for s in spans) / sr
                clips, dl = pack_spans(spans, sr, a.min_s, a.max_s)
                dropped_long += dl
                for c0, c1 in clips:
                    seg = x[c0:c1]
                    p = audio / "clips" / vid / f"{vid}_{c0 // 160:07d}.wav"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(p), seg, sr, subtype="PCM_16")
                    rows.append({"audio_filepath": str(p), "duration": round(len(seg) / sr, 3), "text": "",
                                 "speaker": f"yodas:{vid}", "corpus": "yodas", "video": vid, "shard": name,
                                 "start": round(c0 / sr, 2), "end": round(c1 / sr, 2)})
        tmp.unlink(missing_ok=True)
        with open(man, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = sum(r["duration"] for r in rows) / 3600
        print(f"== shard {name}: {vids} videos {total_s / 3600:.2f} h, VAD speech {speech_s / 3600:.2f} h, "
              f"clips {len(rows)} = {h:.2f} h (dropped spans > {a.max_s:g} s: {dropped_long}), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
