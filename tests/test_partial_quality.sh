#!/bin/sh
# End-to-end plumbing of tools/eval/partial_quality.py, with no model.
#
# The metric definitions have known-answer tests inside streaming_metrics.py.
# What THIS covers is everything between them and a result: the --deltas JSONL
# contract, manifest lookup, the onset map, arm parsing, the two-arm comparison
# and the regression exit code. A fake binary stands in for mynah-asr, so the
# gate runs anywhere -- including the CI runners, which have no converted pack,
# and this dev machine when the model volume is not mounted.
#
# Exit: 0 ok, 1 fail.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT
fail=0
say() { if [ "$1" = 0 ]; then echo "partial_quality ok:   $2"; else echo "partial_quality FAIL: $2"; fail=1; fi; }

mkdir -p "$TMP/pack" "$TMP/bank/it"
echo '{}' > "$TMP/pack/mynah.json"

# a 1 s 16 kHz WAV of silence: only clip_onset reads it, and only if asked
python3 - "$TMP/bank/it/a.wav" <<'PY'
import struct, sys
n = 16000
d = b"\x00\x00" * n
h = b"RIFF" + struct.pack("<I", 36 + len(d)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) + b"data" + struct.pack("<I", len(d))
open(sys.argv[1], "wb").write(h + d)
PY

cat > "$TMP/manifest.json" <<'JSON'
{"samples": [{"file": "it/a.wav", "lang": "it", "duration_sec": 1.0,
              "text": "Il satellite nello spazio."}]}
JSON
cat > "$TMP/onsets.json" <<'JSON'
{"it/a.wav": 0.30}
JSON

# The fake runtime. `base` publishes the right first word late; `fast` publishes
# a wrong one early -- the Q-2 shape the gate exists to catch.
cat > "$TMP/mynah-asr" <<'SH'
#!/bin/sh
mode=$1; shift
arm=base
for x in "$@"; do [ "$x" = "--fast" ] && arm=fast; done
if [ "$mode" = transcribe ]; then echo "Il satellite nello spazio."; exit 0; fi
if [ "$arm" = fast ]; then
  echo '{"type":"delta","i":0,"t0":0.0,"t1":0.60,"fed_s":0.60,"lang":"it","text":"La "}'
  echo '{"type":"delta","i":1,"t0":0.60,"t1":1.00,"fed_s":1.00,"lang":"it","text":"satellite nello spazio."}'
  echo '{"type":"final","deltas":2,"fed_s":1.0,"lang":"it","text":"La satellite nello spazio.","lib_text":"La satellite nello spazio."}'
else
  echo '{"type":"delta","i":0,"t0":0.0,"t1":0.90,"fed_s":0.90,"lang":"it","text":"Il "}'
  echo '{"type":"delta","i":1,"t0":0.90,"t1":1.00,"fed_s":1.00,"lang":"it","text":"satellite nello spazio."}'
  echo '{"type":"final","deltas":2,"fed_s":1.0,"lang":"it","text":"Il satellite nello spazio.","lib_text":"Il satellite nello spazio."}'
fi
SH
chmod +x "$TMP/mynah-asr"

run() {
  python3 "$ROOT/tools/eval/partial_quality.py" -m "$TMP/pack" --binary "$TMP/mynah-asr" \
    --manifest "$TMP/manifest.json" --root "$TMP/bank" --onsets "$TMP/onsets.json" "$@"
}

# --margin 1.0: this run is about the numbers, not the verdict; the exit code
# gets its own assertions below.
run --arm "base:" --arm "fast:--fast" --margin 1.0 --json "$TMP/out.json" > "$TMP/log" 2>&1
rc=$?
[ $rc -eq 0 ]; say $? "two arms run to completion (rc=$rc)"

python3 - "$TMP/out.json" <<'PY'
import json, sys
o = json.load(open(sys.argv[1]))
b, f = o["arms"]["base"], o["arms"]["fast"]
def chk(cond, what):
    print(("partial_quality ok:   " if cond else "partial_quality FAIL: ") + what)
    return 0 if cond else 1
bad = 0
bad += chk(b["summary"]["n"] == 1 and f["summary"]["n"] == 1, "one utterance per arm")
bad += chk(abs(b["utterances"][0]["speech_at_first_word"] - 0.60) < 1e-6,
           "base settles its first word at 0.60 s of speech (0.90 audio - 0.30 onset)")
bad += chk(abs(f["utterances"][0]["speech_at_first_word"] - 0.30) < 1e-6,
           "fast settles 300 ms earlier")
bad += chk(b["summary"]["wrong_first_word"]["rate"] == 0.0, "base gets the first word right")
bad += chk(f["summary"]["wrong_first_word"]["rate"] == 1.0,
           "and fast, the EARLIER arm, gets it wrong -- the two axes stay separate")
bad += chk(b["summary"]["cer"]["median"] == 0.0, "base CER 0 against the corpus")
bad += chk(f["summary"]["cer"]["median"] > 0.0, "fast pays for it in CER")
bad += chk(b["summary"]["prefix_divergence"]["rate"] == 0.0, "no published byte was rewritten")
bad += chk(b["summary"]["leading_silence"]["rate"] == 0.0, "nothing published before onset")
bad += chk(b["summary"]["disagrees_with_offline"]["rate"] == 0.0,
           "base agrees with its own offline transcript")
bad += chk(f["summary"]["disagrees_with_offline"]["rate"] == 1.0,
           "fast does not, and that is reported apart from the corpus CER")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || fail=1

# the regression exit code fires on the CER margin, not on latency
run --arm "base:" --arm "fast:--fast" --margin 0.001 > "$TMP/log2" 2>&1
[ $? -eq 1 ]; say $? "a CER regression past --margin exits 1"
grep -q REGRESSION "$TMP/log2"; say $? "and says which arm regressed"

run --arm "base:" --arm "fast:--fast" > "$TMP/log4" 2>&1
[ $? -eq 1 ]; say $? "the DEFAULT margin already refuses this trade"

# missing pieces skip, they do not fail
run --arm "base:" -m "$TMP/nope" > /dev/null 2>&1
[ $? -eq 77 ]; say $? "a missing pack SKIPs (77)"

echo "test_partial_quality: $([ $fail = 0 ] && echo OK || echo FAIL)"
exit $fail
