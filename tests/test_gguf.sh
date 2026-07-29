#!/bin/sh
# Parità container GGUF vs safetensors sul modello CI (110m):
#   1. esporta model.gguf (f32) e q8_0 con tools/export_gguf.py
#   2. dir di soli symlink SENZA model.safetensors -> il runtime carica il GGUF
#   3. f32: trascrizione IDENTICA al load safetensors; q8_0: testo atteso
# Exit: 0 ok, 1 mismatch, 77 skip (modello convertito o uv assenti).
MODEL_DIR="${1:-models/parakeet-tdt_ctc-110m}"
WAV=tests/audio/test_en.wav
[ -f "$MODEL_DIR/model.safetensors" ] || exit 77
command -v uv >/dev/null 2>&1 || exit 77

ABS_MODEL=$(cd "$MODEL_DIR" && pwd)
TMP=$(mktemp -d /tmp/mynah_asr_gguf.XXXXXX) || exit 1
trap 'rm -rf "$TMP"' EXIT

(cd tools && uv run python export_gguf.py "$ABS_MODEL" --out "$TMP/f32.gguf" >/dev/null && \
             uv run python export_gguf.py "$ABS_MODEL" --dtype q8_0 --out "$TMP/q8.gguf" >/dev/null && \
             uv run python export_gguf.py "$ABS_MODEL" --dtype q4_k --out "$TMP/q4k.gguf" >/dev/null) || exit 1

mkdir "$TMP/model"
for f in "$ABS_MODEL"/*; do
    case "$(basename "$f")" in
        mel_filters.safetensors) ln -s "$f" "$TMP/model/" ;;
        *.safetensors|*.gguf) ;;                 # niente pesi primari: forza il GGUF
        *) ln -s "$f" "$TMP/model/" ;;
    esac
done

fail=0
ref=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$WAV" 2>/dev/null)

ln -s "$TMP/f32.gguf" "$TMP/model/model.gguf"
got=$(./mynah-asr transcribe -m "$TMP/model" -i "$WAV" 2>/dev/null)
if [ "$ref" = "$got" ] && [ -n "$got" ]; then
    echo "gguf e2e f32  OK: $got"
else
    echo "gguf e2e f32  FAIL:"; echo "  safetensors: $ref"; echo "  gguf:        $got"; fail=1
fi

for q in q8:q8_0 q4k:q4_k; do
    file="${q%%:*}"; name="${q##*:}"
    rm "$TMP/model/model.gguf"
    ln -s "$TMP/$file.gguf" "$TMP/model/model.gguf"
    got=$(./mynah-asr transcribe -m "$TMP/model" -i "$WAV" 2>/dev/null)
    case "$got" in
        *"speech recognition test"*) echo "gguf e2e $name OK: $got" ;;
        *) echo "gguf e2e $name FAIL: $got"; fail=1 ;;
    esac
done

# mynah-asr quantize a partire dai pesi GGUF (stessa API f32 -> deve funzionare):
# scrive model.int8.safetensors nella dir tmp e ritrascrive col checkpoint
rm "$TMP/model/model.gguf"
ln -s "$TMP/f32.gguf" "$TMP/model/model.gguf"
if ./mynah-asr quantize -m "$TMP/model" --quant int8 >/dev/null 2>&1; then
    got=$(./mynah-asr transcribe -m "$TMP/model" -i "$WAV" --quant int8 2>/dev/null)
    case "$got" in
        *"speech recognition test"*) echo "gguf quantize-int8 OK: $got" ;;
        *) echo "gguf quantize-int8 FAIL: $got"; fail=1 ;;
    esac
else
    echo "gguf quantize-int8 FAIL: mynah-asr quantize fallito"; fail=1
fi

exit $fail
