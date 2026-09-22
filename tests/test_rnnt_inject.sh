#!/bin/sh
# R-8's diagnostic injection must not leak into anything a user can see.
#
# The experiment is only worth running if the intervention does exactly one
# thing: move the predictor state. These are the invariants asserted rather than
# assumed. Needs a converted streaming pack; without one it SKIPs (77), which is
# how every model-dependent gate in this repo behaves.
#
# Exit: 0 ok, 1 fail, 77 skip.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$ROOT/mynah-asr
M=${MODEL_DIR:-$ROOT/models_local/nemotron-3.5-asr-streaming-0.6b}
WAV=${WAV:-$ROOT/samples/it/fleurs_1521.wav}
LANG=${LANG_TAG:-it}
[ -x "$BIN" ] || { echo "inject SKIP: no binary"; exit 77; }
[ -f "$M/mynah.json" ] || { echo "inject SKIP: no converted pack at $M"; exit 77; }
[ -f "$WAV" ] || { echo "inject SKIP: no clip"; exit 77; }
TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT
fail=0
say() { if [ "$1" = 0 ]; then echo "inject ok:   $2"; else echo "inject FAIL: $2"; fi; [ "$1" = 0 ] || fail=1; }

run() { # $1 = extra env spec ("" = none), writes stdout/stderr
  if [ -z "$1" ]; then
    "$BIN" stream -m "$M" -i "$WAV" --lang "$LANG" --quant int8 --deltas \
        > "$TMP/out" 2> "$TMP/err"
  else
    MYNAH_ASR_RNNT_INJECT="$1" MYNAH_ASR_TRACE_RNNT=1 "$BIN" stream -m "$M" -i "$WAV" \
        --lang "$LANG" --quant int8 --deltas > "$TMP/out" 2> "$TMP/err"
  fi
}
final() { grep '"type":"final"' "$TMP/out" | tail -1; }

# 1. default OFF is bit-identical to no flag at all
run ""; cp "$TMP/out" "$TMP/base.out"; BASE=$(final)
MYNAH_ASR_RNNT_INJECT="" "$BIN" stream -m "$M" -i "$WAV" --lang "$LANG" --quant int8 \
    --deltas > "$TMP/out" 2>/dev/null
[ "$(final)" = "$BASE" ]; say $? "an empty MYNAH_ASR_RNNT_INJECT changes nothing"

# the frame to inject at: the first frame the baseline decided, plus a few, so it
# lands inside the pre-first-token window on any clip
run "frame=6,mode=blank"
grep -q "^\[INJECT\] frame=6 token=" "$TMP/err"
say $? "the injection fires at the frame it was told, not one it chose"

# 2. it is not published and not counted as a natural emission
grep -q "not published, n_emitted still 0" "$TMP/err"
say $? "the injected token is not published and n_emitted is untouched"

python3 - "$TMP/err" <<'PY'
import re, sys
err = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
F = re.compile(r"\[RNNT\] frame=(\d+) audio_s=([\d.-]+) .* pred=(\w+) post=(\d)")
inj = next((i for i, l in enumerate(err) if l.startswith("[INJECT] frame=")), None)
bad = 0
def chk(c, w):
    global bad
    print(("inject ok:   " if c else "inject FAIL: ") + w)
    if not c: bad = 1
rows = [(i,) + F.search(l).groups() for i, l in enumerate(err) if F.search(l)]
after = [r for r in rows if r[0] > inj]
# the predictor state label must stay SOS: the injection is not an emission
sos_after = [r for r in after if r[3] == "SOS"]
chk(bool(sos_after), "decisions after the injection still report pred=SOS")
chk(all(r[4] == "1" for r in after), "and are all marked post=1, so no arm can be confused for the baseline")
chk(all(r[4] == "0" for r in rows if r[0] < inj), "while everything before it is post=0")
# audio must not move: the injection frame is re-decided at the SAME audio_s
at = [r for r in rows if r[1] == "6"]
chk(len(at) == 2, "the injection frame is decided twice, before and after (recorded, not hidden)")
chk(len(at) == 2 and at[0][2] == at[1][2], "at the same audio_s: the injection cannot advance the stream")
# frames must stay monotone apart from that one re-decision
fr = [int(r[1]) for r in rows]
chk(all(b >= a - 1 for a, b in zip(fr, fr[1:])), "frame indices never rewind by more than the re-decision")
sys.exit(bad)
PY
[ $? -eq 0 ] || fail=1

# 3. an arm that cannot be served aborts loudly instead of becoming another arm
run "frame=6,mode=id"
grep -q "\[INJECT\] ABORTED" "$TMP/err"
say $? "an arm with no token ABORTS instead of silently falling back to blank"

# 4. a frame the utterance never reaches leaves the run untouched
run "frame=999999,mode=blank"
[ "$(final)" = "$BASE" ]; say $? "an unreachable injection frame leaves the transcript identical"

echo "test_rnnt_inject: $([ $fail = 0 ] && echo OK || echo FAIL)"
exit $fail
