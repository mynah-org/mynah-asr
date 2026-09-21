#!/usr/bin/env python3
"""rnnt_earliness.py — when did correct evidence first exist, and was it stable?

Q-2B/C/D. R-3 showed the wait before the first word is spent in RNNT blanks and
that the blank margin decays from about +9 to a crossing. That says WHEN blank
lost. It does not say whether the token that eventually won was already the best
non-blank several decisions earlier, which is the whole question: if it was,
there is decoder headroom; if it was not, the evidence simply is not there yet
and the lever is context or cadence instead.

Input: one MYNAH_ASR_TRACE_RNNT=1 stderr capture per clip, the streaming and
offline transcripts, the corpus manifest as the ONLY oracle, and a speech-onset
map. The model's own offline output is never used as truth -- it is reported
beside the reference precisely because the two disagree sometimes.

    MYNAH_ASR_TRACE_RNNT=1 ./mynah-asr stream -m MODEL -i clip.wav ... 2> t.err
    python3 tools/eval/rnnt_earliness.py DIR --onsets onsets.json [--n 3]

THE STABILITY RULE IS FIXED IN .work/first-partial-responsiveness.md AND WAS
WRITTEN BEFORE ANY AGGREGATE WAS COMPUTED. --n reports its sensitivity; a
conclusion that moves with n is reported as n-sensitive, not as a conclusion.
"""
import argparse
import glob
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import streaming_metrics as M   # noqa: E402  the single definition of CER and pct

FRAME_RE = re.compile(r"\[RNNT\] frame=(\d+) blank=(\S+) best_nonblank=(\d+):(\S+) "
                      r"margin=(\S+) chose=(\S+)")
EMIT_RE = re.compile(r"\[RNNT\] emit q=(\d+) audio_s=(\S+) tokens_added=(\d+) "
                     r"chars=(\d+) chars_emitted=(\d+)")


def norm_word(w):
    """NFKC, case-folded, SentencePiece word mark and punctuation stripped."""
    w = unicodedata.normalize("NFKC", w).replace("▁", "")
    w = "".join(c for c in w if not unicodedata.category(c).startswith("P"))
    return w.casefold().strip()


def compatible(tok_text, ref_first):
    """Is this token the start of the reference's first word (or vice versa)?"""
    a, b = norm_word(tok_text), norm_word(ref_first)
    if not a or not b:
        return False
    return a.startswith(b) or b.startswith(a)


def parse(path):
    """Frames in order, each tagged with the audio the chunk had consumed.

    Every frame of a chunk is decided after the SAME amount of audio -- the
    encoder produces the chunk's q frames in one step -- so the chunk's audio_s
    is the honest "audio consumed at this decision" for all of them."""
    pending, frames = [], []
    for line in open(path, errors="replace"):
        m = FRAME_RE.match(line)
        if m:
            pending.append({"frame": int(m.group(1)), "blank": float(m.group(2)),
                            "nb_id": int(m.group(3)), "nb": float(m.group(4)),
                            "margin": float(m.group(5)), "chose": m.group(6)})
            continue
        m = EMIT_RE.match(line)
        if m:
            audio_s = float(m.group(2))
            for f in pending:
                f["audio_s"] = audio_s
            frames.extend(pending)
            pending = []
    return frames


def classify(frames, ref_first, vocab, n):
    """The rule, applied. Returns (class, crossing, stable_start, token_text)."""
    cross = next((i for i, f in enumerate(frames) if f["chose"] == "TOKEN"), None)
    if cross is None:
        return "NO-CROSSING", None, None, None
    tok = vocab[frames[cross]["nb_id"]]
    # how far back the SAME best non-blank id runs, ending at the crossing
    run = 1
    while cross - run >= 0 and frames[cross - run]["nb_id"] == frames[cross]["nb_id"]:
        run += 1
    if cross + 1 < n:                      # not enough decisions to judge
        return "NO-EARLY-SIGNAL", cross, None, tok
    if run < n:
        return "UNSTABLE", cross, None, tok
    start = cross - run + 1
    cls = "STABLE-CORRECT" if compatible(tok, ref_first) else "STABLE-WRONG"
    return cls, cross, start, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--manifest", default="samples/manifest.json")
    ap.add_argument("--model", default="models_local/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--onsets", required=True)
    ap.add_argument("--n", type=int, default=3, help="frames of stability required")
    a = ap.parse_args()

    vocab = json.load(open(os.path.join(a.model, "tokens.json")))
    onsets = {k: float(v) for k, v in json.load(open(a.onsets)).items()
              if isinstance(v, (int, float))}
    ref = {x["file"]: x for x in json.load(open(a.manifest))["samples"]}

    rows = []
    for err in sorted(glob.glob(os.path.join(a.dir, "*.rnnt.err"))):
        tag = os.path.basename(err)[:-9]
        clip = tag.replace("_", "/", 1) + ".wav"
        if clip not in ref:
            continue
        frames = parse(err)
        if not frames:
            continue
        r = ref[clip]
        ref_first = r["text"].split()[0] if r["text"].split() else ""
        onset = onsets.get(clip) or M.reference_for(onsets, clip) or 0.0
        cls, cross, start, tok = classify(frames, ref_first, vocab, a.n)
        stream_txt = open(err[:-9] + ".stream.txt", errors="replace").read().strip()
        off_txt = open(err[:-9] + ".offline.txt", errors="replace").read().strip() \
            if os.path.exists(err[:-9] + ".offline.txt") else ""
        rows.append({
            "clip": clip, "lang": r["lang"], "ref_first": ref_first, "onset": onset,
            "class": cls, "token": tok,
            "emit_speech": (frames[cross]["audio_s"] - onset) if cross is not None else None,
            "stable_speech": (frames[start]["audio_s"] - onset) if start is not None else None,
            "margins": [f["margin"] for f in frames[:(cross + 1) if cross is not None else 0]],
            "cer_stream": M.cer(stream_txt, r["text"]), "cer_offline": M.cer(off_txt, r["text"]),
            "stream": stream_txt, "offline": off_txt,
        })
    return rows, a


if __name__ == "__main__":
    ROWS, A = main()
    print(f"Q-2B/C  {len(ROWS)} utterances, stability rule N={A.n} consecutive frames "
          f"ending at the crossing\n")
    print(f"{'clip':<22} {'class':<16} {'token':<10} {'ref word':<12} "
          f"{'emit@speech':>12} {'stable@speech':>14} {'headroom':>9} "
          f"{'CER str':>8} {'CER off':>8}")
    for r in sorted(ROWS, key=lambda x: x["clip"]):
        hr = (r["emit_speech"] - r["stable_speech"]) * 1000 \
            if r["stable_speech"] is not None else None
        print(f"{r['clip']:<22} {r['class']:<16} {(r['token'] or '-'):<10} "
              f"{r['ref_first'][:12]:<12} "
              f"{(r['emit_speech'] if r['emit_speech'] is not None else float('nan')):>11.3f}s "
              f"{(r['stable_speech'] if r['stable_speech'] is not None else float('nan')):>13.3f}s "
              f"{(hr if hr is not None else float('nan')):>8.0f}m "
              f"{(r['cer_stream'] if r['cer_stream'] is not None else float('nan')):>8.3f} "
              f"{(r['cer_offline'] if r['cer_offline'] is not None else float('nan')):>8.3f}")

    print(f"\n{'class':<16} {'n':>4} {'%':>6} {'median headroom':>16} {'p95':>8}")
    for cls in ("STABLE-CORRECT", "STABLE-WRONG", "UNSTABLE", "NO-EARLY-SIGNAL", "NO-CROSSING"):
        g = [r for r in ROWS if r["class"] == cls]
        if not g:
            continue
        hs = sorted((r["emit_speech"] - r["stable_speech"]) * 1000
                    for r in g if r["stable_speech"] is not None)
        med = f"{hs[len(hs) // 2]:.0f} ms" if hs else "n/a"
        p95 = f"{M.pct(hs, 95):.0f} ms" if hs else "n/a"
        print(f"{cls:<16} {len(g):>4} {100 * len(g) / len(ROWS):>5.1f}% {med:>16} {p95:>8}")

    # Q-2D. Identity alone cannot separate a correct early runner-up from a wrong
    # one -- both are stable. If the MARGIN behaves differently the two could be
    # told apart by something other than persistence, which is the only thing
    # that would make an emission rule safe.
    print("\nQ-2D  margin trajectory over the decisions leading to the crossing")
    print(f"  {'class':<16} {'n':>3} {'margin at cross':>16} {'1 before':>9} {'2 before':>9} "
          f"{'3 before':>9} {'monotone down':>14}")
    for cls in ("STABLE-CORRECT", "STABLE-WRONG", "UNSTABLE"):
        g = [r for r in ROWS if r["class"] == cls and len(r["margins"]) >= 4]
        if not g:
            continue
        def med(k):
            v = sorted(r["margins"][-1 - k] for r in g)
            return v[len(v) // 2]
        mono = sum(1 for r in g
                   if all(r["margins"][i] >= r["margins"][i + 1]
                          for i in range(len(r["margins"]) - 4, len(r["margins"]) - 1)))
        print(f"  {cls:<16} {len(g):>3} {med(0):>+16.2f} {med(1):>+9.2f} {med(2):>+9.2f} "
              f"{med(3):>+9.2f} {mono:>10}/{len(g)}")

    dis = [r for r in ROWS if r["stream"] != r["offline"]]
    print(f"\nstreaming vs offline final text: {len(dis)} of {len(ROWS)} differ "
          f"(the oracle is the reference, never the offline output)")
