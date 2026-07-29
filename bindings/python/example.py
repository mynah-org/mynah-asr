#!/usr/bin/env python3
"""Python bindings example: ASR + timestamps (+ translation on AED models).

Usage: make shared && python3 bindings/python/example.py <model_dir> <file.wav> [lang]
"""

import sys

from mynah_asr import MynahASR, version

if len(sys.argv) < 3:
    sys.exit("usage: python3 example.py <model_dir> <file.wav> [lang|src>tgt]")
model_dir, wav = sys.argv[1], sys.argv[2]
lang = sys.argv[3] if len(sys.argv) > 3 else "auto"

print(f"libmynah_asr {version()}")
with MynahASR(model_dir) as m:
    text, words, detected = m.transcribe(wav, lang=lang, timestamps=True)
    print(f"[lang={detected or lang}]", file=sys.stderr)
    print(text)
    for w, t0, t1 in words[:8]:
        print(f"  {t0:6.2f} {t1:6.2f}  {w}")
