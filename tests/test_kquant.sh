#!/bin/sh
# K-quant dequant parity against the REFERENCE implementation (llama.cpp's `gguf`
# package): tools/gen_kquant_ref.py writes a GGUF with its own writer and dumps
# the dequant of every tensor, then tests/test_gguf compares our loader against
# those .npy files. Needs no model — only uv and network access for the package.
# Exit: 0 ok, 1 mismatch, 77 skip (uv or the gguf package unavailable).
command -v uv >/dev/null 2>&1 || exit 77
[ -x tests/test_gguf ] || exit 77

TMP=$(mktemp -d /tmp/mynah_asr_kquant.XXXXXX) || exit 1
trap 'rm -rf "$TMP"' EXIT

if ! (cd tools && uv run --with gguf python gen_kquant_ref.py "$TMP" >/dev/null 2>&1); then
    echo "kquant parity SKIP (gguf package unavailable: offline?)"
    exit 77
fi

./tests/test_gguf "$TMP/kquant.gguf" "$TMP"
