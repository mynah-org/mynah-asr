#!/bin/sh
# Regenerate the long bench audio fixtures (tests/audio/long_{60s,300s}.wav)
# from LibriSpeech dev-clean (CC-BY 4.0): the first 40 flac files of speaker
# 1272, concatenated and reconverted to 16 kHz mono s16. Same files behind the
# numbers in docs/benchmarks.md (M1 and A100). Needs ffmpeg + curl (or aria2c).
set -eu
cd "$(dirname "$0")/.."
TMP=${TMPDIR:-/tmp}/mynah-librispeech
mkdir -p "$TMP" tests/audio
TAR="$TMP/dev-clean.tar.gz"
if [ ! -f "$TAR" ]; then
    URL=https://www.openslr.org/resources/12/dev-clean.tar.gz
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x8 -s8 --console-log-level=warn -d "$TMP" -o dev-clean.tar.gz "$URL"
    else
        curl -fL --retry 3 -o "$TAR" "$URL"
    fi
fi
tar xzf "$TAR" -C "$TMP" "LibriSpeech/dev-clean/1272"
ls "$TMP"/LibriSpeech/dev-clean/1272/*/*.flac | sort | head -40 \
    | sed "s/.*/file '&'/" > "$TMP/concat.txt"
ffmpeg -v error -f concat -safe 0 -i "$TMP/concat.txt" \
    -ar 16000 -ac 1 -sample_fmt s16 -t 60 -y tests/audio/long_60s.wav
ffmpeg -v error -f concat -safe 0 -i "$TMP/concat.txt" \
    -ar 16000 -ac 1 -sample_fmt s16 -t 300 -y tests/audio/long_300s.wav
ls -la tests/audio/long_*.wav
