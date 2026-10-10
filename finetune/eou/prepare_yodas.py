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
    from silero_vad import get_speech_timestamps, load_silero_vad
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "canary"))
    import prepare_it as D  # curl (HTTP/1.1 + retries)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_mix import resample16

    torch.set_num_threads(8)
    vad = load_silero_vad()
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
                spans = get_speech_timestamps(torch.from_numpy(x), vad, sampling_rate=sr, min_silence_duration_ms=300,
                                              speech_pad_ms=100)
                speech_s += sum(s["end"] - s["start"] for s in spans) / sr
                def flush(cur):
                    if cur is None or (cur[1] - cur[0]) / sr < a.min_s:
                        return
                    seg = x[cur[0]:cur[1]]
                    p = audio / "clips" / vid / f"{vid}_{cur[0] // 160:07d}.wav"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(p), seg, sr, subtype="PCM_16")
                    rows.append({"audio_filepath": str(p), "duration": round(len(seg) / sr, 3), "text": "",
                                 "speaker": f"yodas:{vid}", "corpus": "yodas", "video": vid, "shard": name,
                                 "start": round(cur[0] / sr, 2), "end": round(cur[1] / sr, 2)})

                cur = None   # [start, end] samples of the clip being packed
                for sp in spans:
                    if (sp["end"] - sp["start"]) / sr > a.max_s:   # never cut inside speech
                        flush(cur); cur = None; dropped_long += 1
                    elif cur is None:
                        cur = [sp["start"], sp["end"]]
                    elif (sp["end"] - cur[0]) / sr <= a.max_s:
                        cur[1] = sp["end"]
                    else:
                        flush(cur); cur = [sp["start"], sp["end"]]
                flush(cur)
        tmp.unlink(missing_ok=True)
        with open(man, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = sum(r["duration"] for r in rows) / 3600
        print(f"== shard {name}: {vids} videos {total_s / 3600:.2f} h, VAD speech {speech_s / 3600:.2f} h, "
              f"clips {len(rows)} = {h:.2f} h (dropped spans > {a.max_s:g} s: {dropped_long}), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
