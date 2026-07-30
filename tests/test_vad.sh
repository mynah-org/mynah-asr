#!/bin/sh
# Silero VAD parity: converts the ONNX if needed, dumps the per-frame
# probabilities with onnxruntime (the reference implementation itself, not a
# rewrite of it), then compares src/vad.c against them frame by frame.
#
# Needs models/silero-vad/silero_vad.onnx (2.3 MB, MIT):
#   mkdir -p models/silero-vad && curl -sSLo models/silero-vad/silero_vad.onnx \
#     https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
#
# Exit: 0 ok, 1 mismatch, 77 skip (onnx absent, or uv/onnxruntime unavailable).
VAD_DIR=${1:-models/silero-vad}
WAV=${2:-samples/en/fleurs_1521.wav}

[ -x tests/test_vad ] || exit 77
[ -f "$WAV" ] || exit 77
[ -f "$VAD_DIR/silero_vad.onnx" ] || exit 77
command -v uv >/dev/null 2>&1 || exit 77

TMP=$(mktemp -d /tmp/mynah_asr_vad.XXXXXX) || exit 1
trap 'rm -rf "$TMP"' EXIT

ONNX=$(cd "$(dirname "$VAD_DIR")" && pwd)/$(basename "$VAD_DIR")/silero_vad.onnx
if [ ! -f "$VAD_DIR/mynah.json" ]; then
    if ! (cd tools && uv run --extra vad python convert_silero.py "$ONNX" "../$VAD_DIR"); then
        echo "vad parity SKIP (onnx package unavailable: offline?)"
        exit 77
    fi
fi

WAV_ABS=$(cd "$(dirname "$WAV")" && pwd)/$(basename "$WAV")
if ! (cd tools && uv run --extra vad python -m oracle.vad "$ONNX" "$WAV_ABS" \
        --out "$TMP/probs.npy" >"$TMP/oracle.log" 2>&1); then
    cat "$TMP/oracle.log"
    echo "vad parity SKIP (onnxruntime unavailable: offline?)"
    exit 77
fi
cat "$TMP/oracle.log"

# WRAP lets `make leaks` run the binary under a checker without rebuilding the dump
${WRAP} ./tests/test_vad "$VAD_DIR" "$WAV" "$TMP/probs.npy"
