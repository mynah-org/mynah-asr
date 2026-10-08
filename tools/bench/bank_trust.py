#!/usr/bin/env python3
"""bank_trust.py - decide which clips of a stress bank are trustworthy enough to call validated.

    python3 tools/bench/bank_trust.py samples/stress-fr/manifest.json --lang fr \
        -m models/nemotron-3.5-asr-streaming-0.6b [--jobs 6 --cpus-per-job 5] [--limit N]

WHY.  A bank assembled automatically from a public corpus is not validated because it
exists: English showed FLEURS retakes whose reference does not match the audio (M-6).
Every clip is checked, every exclusion is counted with its reason, and the raw bank's
numbers stay reportable beside the validated subset's. Registered in
.work/french-validation-20260925.md before the first run.

CHECKS, per clip (a composed long clip is checked as the file it is):
  audio     16 kHz mono PCM16; duration within 0.05 s of the manifest; peak >= -30 dBFS
            (the qualification's --min-peak-dbfs); clipped samples < 0.1 %
  reference non-empty; for fr, not pure ASCII when longer than 8 words (French text
            without a single accent over that length is suspect)
  language  the model's own tag with --lang auto must start with the bank's language
  agreement WER of an OFFLINE (full-context) --lang <lang> transcript against the
            reference <= 0.5; above it the reference is suspect (a retake, a misread)
Writes <manifest dir>/trust.json (per clip) and manifest-validated.json (the same format
as the manifest, only the clips that pass). Standard library + streaming_metrics."""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, wave, array
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streaming_metrics import cer, wer, wer_format_free  # noqa: E402


def audio_check(path, want_s):
    with wave.open(path) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    bad = []
    if (sr, ch, sw) != (16000, 1, 2):
        bad.append(f"format {sr} Hz {ch} ch {8 * sw} bit")
        return bad, None, None
    a = array.array("h", raw)
    peak = max((abs(x) for x in a), default=0)
    peak_db = 20 * __import__("math").log10(max(peak, 1) / 32768.0)
    clip = sum(1 for x in a if abs(x) >= 32767) / max(len(a), 1)
    dur = n / sr
    if want_s is not None and abs(dur - want_s) > 0.05:
        bad.append(f"duration {dur:.2f} s vs manifest {want_s:.2f} s")
    if peak_db < -30.0:
        bad.append(f"peak {peak_db:.1f} dBFS < -30")
    if clip >= 0.001:
        bad.append(f"clipped {clip:.2%}")
    return bad, round(peak_db, 1), round(dur, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--lang", required=True)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--binary", default="./mynah-asr")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cpus-per-job", type=int, default=5)
    ap.add_argument("--first-cpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-wer", type=float, default=0.5)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    clips = man["clips"] if isinstance(man, dict) else man
    root = os.path.dirname(a.manifest)
    if a.limit:
        clips = clips[:a.limit]

    def one(k_c):
        k, c = k_c
        path = os.path.join(root, c["file"])
        lo = a.first_cpu + (k % a.jobs) * a.cpus_per_job
        cpus = f"{lo}-{lo + a.cpus_per_job - 1}"
        ref = c.get("text", "")
        why, peak, dur = audio_check(path, c.get("duration_sec"))
        if not ref.strip():
            why.append("empty reference")
        elif a.lang == "fr" and len(ref.split()) > 8 and all(ord(ch) < 128 for ch in ref):
            why.append("reference has no French orthography")
        # taskset keeps the jobs on disjoint cpu slices
        def pinned(lang):
            env = dict(os.environ, MYNAH_ASR_THREADS=str(a.cpus_per_job))
            r = subprocess.run(["taskset", "-c", cpus, a.binary, "transcribe", "-m", a.model,
                                "-i", path, "--quant", "int8", "--lang", lang],
                               capture_output=True, text=True, env=env)
            m = re.search(r"\blang=([A-Za-z-]+)\]", r.stderr)
            return r.stdout.strip(), (m.group(1) if m else None)
        hyp_auto, tag = pinned("auto")
        hyp, _ = pinned(a.lang)
        if not (tag or "").lower().startswith(a.lang):
            why.append(f"detected language {tag}")
        w = wer(hyp, ref) if ref.strip() else 1.0
        if w > a.max_wer:
            why.append(f"offline WER {w:.2f} > {a.max_wer}")
        return dict(file=c["file"], cls=c.get("class"), composed=bool(c.get("composed")),
                    duration_sec=dur, peak_dbfs=peak, detected=tag, wer=round(w, 4),
                    cer=round(cer(hyp, ref), 4) if ref.strip() else 1.0,
                    wer_ff=round(wer_format_free(hyp, ref, a.lang), 4) if ref.strip() else 1.0,
                    hyp=hyp, hyp_auto=hyp_auto, ref=ref, excluded=why)

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(one, enumerate(clips)))
    keep = [c for c, r in zip(clips, rows) if not r["excluded"]]
    reasons = {}
    for r in rows:
        for w in r["excluded"]:
            key = re.sub(r"[-\d.]+", "#", w)
            reasons[key] = reasons.get(key, 0) + 1
    json.dump(rows, open(os.path.join(root, "trust.json"), "w"), indent=1, ensure_ascii=False)
    out = dict(man) if isinstance(man, dict) else {"clips": []}
    out["clips"] = keep
    out["validated"] = {"tool": "tools/bench/bank_trust.py", "checked": len(rows), "kept": len(keep),
                        "excluded_by_reason": reasons, "max_wer": a.max_wer, "model": a.model}
    json.dump(out, open(os.path.join(root, "manifest-validated.json"), "w"), indent=1, ensure_ascii=False)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"checked {len(rows)}  kept {len(keep)}  excluded {len(rows) - len(keep)}")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    for name, rs in (("all", rows), ("kept", [r for r in rows if not r["excluded"]])):
        print(f"offline --lang {a.lang}, {name:4s}: WER mean {mean([r['wer'] for r in rs]):.4f}  "
              f"CER mean {mean([r['cer'] for r in rs]):.4f}  WER-ff mean {mean([r['wer_ff'] for r in rs]):.4f}")
    tags = {}
    for r in rows:
        tags[r["detected"]] = tags.get(r["detected"], 0) + 1
    print(f"detected languages under auto: {dict(sorted(tags.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
