#!/bin/sh
# S1-3 gate: the per-chunk streaming step allocates NOTHING after warm-up.
#
# The counter (tests/malloc_count.c, preloaded/inserted) is a whole-process
# total, so the per-chunk number comes from DIFFERENCING two runs of the same
# command on the same model with inputs of different length: everything outside
# the chunk loop — model load, stream open, WAV load, the final flush — is
# identical in both, so the difference is exactly what the extra chunks cost.
#
#   first half  : allocations up to the half-length input (load + warm-up chunks)
#   second half : allocations(full) - allocations(half)   <- must be 0
#
# Usage: sh tests/test_stream_allocs.sh <model_dir> [wav]
# Exit: 0 ok, 1 the second half allocated, 77 skip (no model, no python3, or
#       library interposition does not work on this platform).
set -u

MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
WAV="${2:-tests/audio/test_it.wav}"

[ -f "$MODEL_DIR/mynah.json" ] || exit 77
[ -f "$WAV" ] || exit 77
[ -x ./mynah-asr ] || exit 77

case "$(uname -s)" in
    Darwin) LIB=tests/libmalloc_count.dylib ;;
    *)      LIB=tests/libmalloc_count.so ;;
esac
[ -f "$LIB" ] || { echo "SKIP stream-allocs: $LIB not built"; exit 77; }
command -v python3 >/dev/null 2>&1 || { echo "SKIP stream-allocs: python3 missing"; exit 77; }

# cache-aware streaming only exists for the streaming models
ENGINE=$(sed -n 's/.*"engine": "\([^"]*\)".*/\1/p' "$MODEL_DIR/mynah.json")
[ "$ENGINE" = "nemotron-streaming" ] || { echo "SKIP stream-allocs: $ENGINE has no streaming path"; exit 77; }

TMP=$(mktemp -d) || exit 77
trap 'rm -rf "$TMP"' EXIT

count_file="$TMP/count"
run_counted() { # -> prints the call count, or nothing when interposition failed
    rm -f "$count_file"
    if [ "$(uname -s)" = "Darwin" ]; then
        DYLD_INSERT_LIBRARIES="$LIB" DYLD_FORCE_FLAT_NAMESPACE=1 \
        MALLOC_COUNT_OUT="$count_file" "$@" >/dev/null 2>"$TMP/err"
    else
        LD_PRELOAD="$LIB" MALLOC_COUNT_OUT="$count_file" "$@" >/dev/null 2>"$TMP/err"
    fi
    [ -s "$count_file" ] && cat "$count_file"
}

# 0) does interposition work here at all?
probe=$(run_counted ./mynah-asr --version)
case "${probe:-0}" in
    ''|0) echo "SKIP stream-allocs: malloc interposition not effective on this platform"; exit 77 ;;
esac

# 1) the half-length input, written next to the full one
python3 - "$WAV" "$TMP/half.wav" <<'PY'
import sys, wave
src, dst = sys.argv[1], sys.argv[2]
r = wave.open(src, 'rb')
n = r.getnframes()
w = wave.open(dst, 'wb')
w.setnchannels(r.getnchannels()); w.setsampwidth(r.getsampwidth()); w.setframerate(r.getframerate())
w.writeframes(r.readframes(n // 2))
w.close(); r.close()
PY
[ -f "$TMP/half.wav" ] || { echo "SKIP stream-allocs: could not write the half-length input"; exit 77; }

sec_full=$(python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print('%.3f'%(w.getnframes()/w.getframerate()))" "$WAV")
sec_half=$(python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print('%.3f'%(w.getnframes()/w.getframerate()))" "$TMP/half.wav")

# 2) the same command on both inputs (the exact streaming command the gate names)
a_half=$(run_counted ./mynah-asr stream -m "$MODEL_DIR" -i "$TMP/half.wav" --quant int8)
a_full=$(run_counted ./mynah-asr stream -m "$MODEL_DIR" -i "$WAV"          --quant int8)
if [ -z "${a_half:-}" ] || [ -z "${a_full:-}" ]; then
    echo "stream-allocs FAIL: no count produced"; sed -n '1,20p' "$TMP/err"; exit 1
fi

# chunk accounting: one encoder chunk per (right+1) encoder frames
frame_ms=$(sed -n 's/.*"encoder_frame_ms": \([0-9.]*\).*/\1/p' "$MODEL_DIR/mynah.json")
[ -n "$frame_ms" ] || frame_ms=80
right=$(python3 - "$MODEL_DIR/mynah.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["streaming"]
print(c["att_context_presets"][c.get("default_preset_index", 0)][1])
PY
)
added=$(python3 -c "import sys;print(int((float(sys.argv[1])-float(sys.argv[2]))/(float(sys.argv[3])*(int(sys.argv[4])+1)/1000.0)))" \
        "$sec_full" "$sec_half" "$frame_ms" "$right")
[ "${added:-0}" -gt 0 ] 2>/dev/null || added=1

delta=$((a_full - a_half))
printf 'stream-allocs: %ss -> %lu calls, %ss -> %lu calls\n' "$sec_half" "$a_half" "$sec_full" "$a_full"
printf 'stream-allocs: second half = %d chunks, %d allocations (%s per chunk)\n' \
       "$added" "$delta" "$(python3 -c "import sys;print('%.2f'%(int(sys.argv[1])/int(sys.argv[2])))" "$delta" "$added")"

if [ "$delta" -eq 0 ]; then
    echo "stream-allocs OK: 0 allocations per chunk after warm-up"
    exit 0
fi
echo "stream-allocs FAIL: $delta allocations over $added steady-state chunks (expected 0)"
exit 1
