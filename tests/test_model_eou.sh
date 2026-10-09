#!/bin/sh
# The model's own end of utterance (<EOU>/<EOB>, src/mynah_asr.c
# stream_model_eou) on a pack trained for it, through the CLI stream path:
#
#   two.wav = A (samples/en/fleurs_1521.wav) + 1 s of silence + B
#             (samples/en/fleurs_1534.wav), FLEURS clips, CC-BY 4.0
#
# checks that
#   - exactly one model eou falls between A and B (after A's speech, before
#     B starts), and every eou line carries "source":"model";
#   - B is transcribed after it, byte-identical to B streamed alone (the reset
#     really returns the model to a fresh state), and A to A alone;
#   - no "<EOU>"/"<EOB>" ever reaches the text.
# With a second pack without those tokens (NEMOTRON_DIR, optional) it also
# checks that no model eou is ever reported there.
#
#   sh tests/test_model_eou.sh <eou_pack_dir> [nemotron_pack_dir]
# Exit: 0 ok, 1 failure, 77 skip (pack, binary or python3 missing).
EOU_DIR=${1:-models/parakeet-realtime-eou-120m}
NEMOTRON_DIR=${2:-}
A=samples/en/fleurs_1521.wav
B=samples/en/fleurs_1534.wav

[ -x ./mynah-asr ] || exit 77
[ -f "$EOU_DIR/mynah.json" ] || exit 77
command -v python3 >/dev/null 2>&1 || exit 77
[ -f "$A" ] && [ -f "$B" ] || exit 77

TMP=$(mktemp -d /tmp/mynah_asr_eou.XXXXXX) || exit 1
trap 'rm -rf "$TMP"' EXIT

python3 - "$A" "$B" "$TMP/two.wav" <<'PY' || exit 1
import sys, wave
def rd(p):
    w = wave.open(p)
    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, p
    d = w.readframes(w.getnframes()); w.close(); return d
a, b = rd(sys.argv[1]), rd(sys.argv[2])
o = wave.open(sys.argv[3], "wb")
o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000)
o.writeframes(a + b"\0\0" * 16000 + b); o.close()
PY

./mynah-asr stream -m "$EOU_DIR" -i "$A" --deltas > "$TMP/a.jsonl" 2>/dev/null || exit 1
./mynah-asr stream -m "$EOU_DIR" -i "$B" --deltas > "$TMP/b.jsonl" 2>/dev/null || exit 1
./mynah-asr stream -m "$EOU_DIR" -i "$TMP/two.wav" --deltas > "$TMP/two.jsonl" 2>/dev/null || exit 1
if [ -n "$NEMOTRON_DIR" ] && [ -f "$NEMOTRON_DIR/mynah.json" ]; then
    ./mynah-asr stream -m "$NEMOTRON_DIR" -i "$TMP/two.wav" --deltas > "$TMP/nem.jsonl" 2>/dev/null || exit 1
fi

python3 - "$TMP" "$A" <<'PY'
import json, os, sys, wave
tmp = sys.argv[1]
w = wave.open(sys.argv[2]); a_end = w.getnframes() / w.getframerate(); w.close()
b_start = a_end + 1.0
def load(n):
    return [json.loads(l) for l in open(os.path.join(tmp, n)) if l.startswith("{")]
def text(lines, lo=None, hi=None):
    # the deltas whose window ends inside [lo, hi), concatenated
    return "".join(x["text"] for x in lines if x["type"] == "delta"
                   and (lo is None or x["t1"] > lo) and (hi is None or x["t1"] <= hi))
bad = 0
def check(ok, what):
    global bad
    print(("OK   " if ok else "FAIL ") + what)
    bad |= not ok
a, b, two = load("a.jsonl"), load("b.jsonl"), load("two.jsonl")
eous = [x for x in two if x["type"] == "eou"]
check(all(x.get("source") == "model" for x in eous), "every eou on the EOU pack is a model eou")
between = [x for x in eous if a_end - 1.0 < x["t"] <= b_start]
check(len(between) == 1, f"one model eou between A (ends {a_end:.2f}s) and B (starts {b_start:.2f}s): {[x['t'] for x in eous]}")
cut = between[0]["t"] if between else b_start
idx = two.index(between[0]) if between else len(two)
ta = "".join(x["text"] for x in two[:idx] if x["type"] == "delta")
tb = "".join(x["text"] for x in two[idx:] if x["type"] == "delta")
check(ta.strip() == text(a).strip(), "A in the two-utterance stream == A alone")
check(tb.strip() == text(b).strip() and tb.strip() != "",
      "B after the eou == B alone: %r" % tb.strip()[:80])
final = two[-1]["text"] if two and two[-1]["type"] == "final" else ""
check("<EOU>" not in final and "<EOB>" not in final and "<" not in ta + tb, "no <EOU>/<EOB> in the text")
if os.path.exists(os.path.join(tmp, "nem.jsonl")):
    nem = load("nem.jsonl")
    check(not any(x["type"] == "eou" for x in nem), "a pack without <EOU>/<EOB> reports no model eou")
sys.exit(1 if bad else 0)
PY
