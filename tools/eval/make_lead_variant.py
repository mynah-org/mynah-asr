#!/usr/bin/env python3
"""make_lead_variant.py — the same bank, with N seconds of silence in front.

    python3 tools/eval/make_lead_variant.py --lead 2.0 --out samples/eval-lead2

PHASE C candidate 1. Phase B found that utterances whose engine had been running
before speech began got a better first word (+11 to +12 points) and a lower WER,
in BOTH languages. That is a correlation: a recording with a clean lead-in may
simply be a better recording. This makes it causal, on the same 400 utterances,
by giving every one of them the lead-in.

WHY DIGITAL SILENCE and not room tone. It is what a real stream sees: a socket
opens, the encoder starts, and nobody has spoken yet. Room tone borrowed from
another recording would be an out-of-distribution intervention (see
.work/cache-priming-spec.md); this is the deployable shape of the same question.

WHAT THIS IS, STATED PRECISELY. A **controlled causal intervention that
preserves the target speech, not the original input stream.** The prepended
silence IS synthetic audio added to the stream. No future speech is leaked and
no frame is duplicated, and the bytes after the lead are identical -- but the
utterance the engine receives is not the utterance it received before, and a
result must not be reported as if it were.

That matters for what an improvement would MEAN, because at least three
different mechanisms would produce one and this experiment cannot separate them:

  * encoder warm-up / cache state -- the K/V and convolution caches hold real
    frames by the time speech starts;
  * pre-speech context -- the representation of the first speech frames has a
    non-empty past to attend to;
  * chunk phase / alignment -- the lead shifts where speech falls inside the
    q-frame chunk grid, which R-11 showed is not a neutral axis.

Discriminating them is a separate experiment. This one only asks whether the
effect exists at all.

The manifest is rewritten with the new durations and the SAME reference text, so
the gate scores exactly what it scored before.

Exit: 0 ok, 2 usage.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import wave


def prepend(src, dst, lead_s):
    with wave.open(src, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    pad = b"\x00" * (int(round(lead_s * sr)) * ch * sw)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with wave.open(dst, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(pad + raw)
    return (len(pad) + len(raw)) / float(sr * ch * sw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lead", type=float, required=True, help="seconds of silence to prepend")
    ap.add_argument("--src", default="samples/eval-bank")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.src, "manifest.json"), encoding="utf-8"))
    out = []
    for s in man["samples"]:
        src = os.path.join(a.src, s["file"])
        if not os.path.exists(src):
            continue
        dur = prepend(src, os.path.join(a.out, s["file"]), a.lead)
        out.append({**s, "duration_sec": round(dur, 3), "lead_s": a.lead})
    man = {**man, "note": f"{man.get('note','')} | {a.lead} s of digital silence prepended "
                          "(tools/eval/make_lead_variant.py): the audio after the lead is "
                          "byte-identical and the reference is unchanged",
           "samples": out}
    os.makedirs(a.out, exist_ok=True)
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"{a.out}: {len(out)} utterances, +{a.lead:.1f} s each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
