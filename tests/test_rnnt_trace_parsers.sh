#!/bin/sh
# The two trace analyses must READ the trace the runtime actually emits.
#
# They silently stopped doing so once: R-9 extended `dec_trace_line` and the
# regexes in tools/eval/rnnt_earliness.py and tools/eval/rnnt_silence.py went on
# matching the pre-R-9 shape, so both matched zero lines and would have reported
# an empty result as a finding. A parser that matches nothing is worse than a
# missing parser, because it is believed.
#
# The two lines below are REAL output of the current emitter, captured from
# `MYNAH_ASR_TRACE_RNNT=1 mynah-asr stream` on the streaming pack -- not written
# by hand against the printf, which is the same mistake one level down. If
# src/decoder.c changes the line, this fails and names the field.
set -e
cd "$(dirname "$0")/.."

python3 - <<'PY'
import sys
sys.path.insert(0, "tools/eval")
from rnnt_earliness import parse_frame as pe
from rnnt_silence import parse_frame as ps

BLANK_LINE = ("[RNNT] frame=0 audio_s=0.3000 enc=3.39 joint=0.73 pred=SOS post=0 "
              "blank=13087:-14.4590:r1 wmark=2:-22.4129:r2 lex=6:-29.3536:r3 "
              "nb=2:-22.4129:r2 chose=13087 margin_lex=14.8946 margin_nb=7.9539")
TOKEN_LINE = ("[RNNT] frame=29 audio_s=2.5000 enc=4.38 joint=2.13 pred=MOVED post=0 "
              "blank=13087:-56.2002:r4 wmark=2:-66.5932:r57 lex=130:-47.7977:r1 "
              "nb=130:-47.7977:r1 chose=130 margin_lex=-8.4025 margin_nb=-8.4025")

bad = 0
def eq(got, want, what):
    global bad
    if got != want:
        print(f"  FAIL {what}: got {got!r}, want {want!r}")
        bad += 1

for name, parse in (("rnnt_earliness", pe), ("rnnt_silence", ps)):
    b, t = parse(BLANK_LINE), parse(TOKEN_LINE)
    if b is None or t is None:
        print(f"  FAIL {name}: parsed NOTHING from a real trace line")
        bad += 1
        continue
    # A blank decision and a token decision must be told apart -- this is the
    # single fact every analysis downstream rests on.
    eq(b["chose"], "BLANK", f"{name} blank line chose")
    eq(t["chose"], "TOKEN", f"{name} token line chose")
    eq(b["frame"], 0, f"{name} frame")
    eq(t["frame"], 29, f"{name} frame")
    eq(b["audio_s"], 0.3, f"{name} audio_s")
    # The score inside blank=<id>:<score>:r<rank>, never the id.
    eq(b["blank"], -14.4590, f"{name} blank score")
    eq(t["nb_id"], 130, f"{name} best non-blank id")
    eq(t["nb"], -47.7977, f"{name} best non-blank score")
    # The old `margin` was blank minus best non-blank: that is margin_nb now.
    eq(b["margin"], 7.9539, f"{name} margin maps to margin_nb")
    eq(t["margin"], -8.4025, f"{name} margin maps to margin_nb")
    eq(b["pred"], "SOS", f"{name} predictor state")
    eq(t["pred"], "MOVED", f"{name} predictor state")

    # A line that is not a frame line must be rejected, not half-parsed.
    eq(parse("[RNNT] emit q=4 audio_s=0.9000 tokens_added=1 chars=3 chars_emitted=0"),
       None, f"{name} rejects the emit line")
    eq(parse("nonsense"), None, f"{name} rejects a non-trace line")

print("test_rnnt_trace_parsers: " + ("OK" if not bad else f"FAIL ({bad})"))
sys.exit(1 if bad else 0)
PY
