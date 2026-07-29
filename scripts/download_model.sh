#!/usr/bin/env bash
# Download a supported model from HuggingFace into models/<name>/.
#
# Interactive when run with no arguments (pick from the supported-model menu);
# scriptable with --model for CI / automation.
#
# Usage:
#   scripts/download_model.sh                      # interactive menu
#   scripts/download_model.sh --model nemotron     # by alias (or full name, or menu number)
#   scripts/download_model.sh --model canary-v2 --dir my-dir
#   scripts/download_model.sh --list
#
# After downloading, the script prints the convert (and optional quantize) commands.
# Note: mynah reads its own converted format (safetensors mmap), not GGUF —
# quantized int8/int4 checkpoints are produced locally with `mynah-asr quantize`.
set -euo pipefail

# key | HF repo id | mode (hf = native port, nemo = .nemo archive) | ~download | description
MODELS=(
  "nemotron|nvidia/nemotron-3.5-asr-streaming-0.6b|hf|2.6 GB|cache-aware STREAMING + offline, 40 languages, LID  (v1 target)"
  "tdt-v3|nvidia/parakeet-tdt-0.6b-v3|hf|2.4 GB|offline TDT, 25 EU languages, PnC+ITN, timestamps"
  "110m|nvidia/parakeet-tdt_ctc-110m|nemo|0.5 GB|hybrid TDT+CTC 114M, EN — the fastest (CI model)"
  "rnnt-0.6b|nvidia/parakeet-rnnt-0.6b|nemo|2.4 GB|offline RNNT, EN"
  "ctc-0.6b|nvidia/parakeet-ctc-0.6b|nemo|2.4 GB|offline CTC, EN"
  "rnnt-1.1b|nvidia/parakeet-rnnt-1.1b|nemo|4.3 GB|offline RNNT 42 layers, EN"
  "ctc-1.1b|nvidia/parakeet-ctc-1.1b|nemo|4.3 GB|offline CTC 42 layers, EN"
  "canary-180m|nvidia/canary-180m-flash|nemo|0.8 GB|ASR en/de/es/fr + TRANSLATION, word timestamps"
  "canary-1b-flash|nvidia/canary-1b-flash|nemo|3.5 GB|ASR en/de/es/fr + TRANSLATION, word+segment timestamps"
  "canary-v2|nvidia/canary-1b-v2|nemo|3.9 GB|ASR 25 EU languages + en<->24 TRANSLATION, ITN"
  "110m-gguf|handy-computer/parakeet-tdt_ctc-110m-gguf|gguf|90 MB|COMMUNITY Q4_K_M GGUF of the 110m — lightest download, no torch needed (verified; no CTC head)"
  "v3-gguf|handy-computer/parakeet-tdt-0.6b-v3-gguf|gguf|485 MB|COMMUNITY Q4_K_M GGUF of the tdt-v3 — 25 EU langs, no torch (verified; dequant to F32 at load, RAM ~= full model)"
)

# HF-native port files (mode=hf). Only the ones marked * are required; the others
# vary across repos and are skipped with a note if absent.
HF_REQUIRED=(config.json tokenizer.json model.safetensors)
HF_OPTIONAL=(generation_config.json processor_config.json preprocessor_config.json tokenizer_config.json)

CHOICE=""
DEST=""

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

list_models() {
  local i=1 key repo mode size desc
  printf "  %-3s %-16s %-38s %-7s %s\n" "#" "alias" "model" "size" "capabilities"
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r key repo mode size desc <<<"$entry"
    printf "  %-3s %-16s %-38s %-7s %s\n" "$i)" "$key" "${repo#nvidia/}" "$size" "$desc"
    i=$((i+1))
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) CHOICE="$2"; shift 2 ;;
    --dir)   DEST="$2";   shift 2 ;;
    --list)  list_models; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$CHOICE" ]]; then
  echo "Mynah — supported models (see docs/models.md):"
  echo ""
  list_models
  echo ""
  read -r -p "Model to download [1-${#MODELS[@]} or alias]: " CHOICE
fi

# resolve CHOICE -> menu entry (by number, alias, or repo/model name)
ENTRY=""
i=1
for entry in "${MODELS[@]}"; do
  IFS='|' read -r key repo _ _ _ <<<"$entry"
  if [[ "$CHOICE" == "$i" || "$CHOICE" == "$key" || "$CHOICE" == "$repo" || "$CHOICE" == "${repo#nvidia/}" ]]; then
    ENTRY="$entry"; break
  fi
  i=$((i+1))
done
if [[ -z "$ENTRY" ]]; then
  echo "Unknown model: '$CHOICE'"; echo ""; list_models; exit 1
fi

IFS='|' read -r KEY REPO_ID MODE SIZE DESC <<<"$ENTRY"
NAME="${REPO_ID##*/}"   # repo basename (holds for nvidia/* and for the community ggufs)
DEST="${DEST:-models/$NAME}"
BASE="https://huggingface.co/$REPO_ID/resolve/main"

fetch() { # fetch <remote-file> <required:yes|no>
  local f="$1" required="$2"
  if [[ -f "$DEST/$f" ]]; then
    echo "skip  $f (already present)"
    return 0
  fi
  echo "fetch $f"
  if curl -fL --retry 3 --progress-bar "$BASE/$f" -o "$DEST/$f.part"; then
    mv "$DEST/$f.part" "$DEST/$f"
  else
    rm -f "$DEST/$f.part"
    if [[ "$required" == yes ]]; then
      echo "ERROR: required file $f not found in $REPO_ID" >&2; exit 1
    fi
    echo "note: $f not in the repo, skipped (optional)"
  fi
}

echo ""
echo "Downloading $REPO_ID (~$SIZE) -> $DEST"
mkdir -p "$DEST"

if [[ "$MODE" == hf ]]; then
  for f in "${HF_REQUIRED[@]}"; do fetch "$f" yes; done
  for f in "${HF_OPTIONAL[@]}"; do fetch "$f" no;  done
elif [[ "$MODE" == gguf ]]; then
  # handy-computer naming: <basename-without-gguf-suffix>-Q4_K_M.gguf
  fetch "${NAME%-gguf}-Q4_K_M.gguf" yes
else
  fetch "$NAME.nemo" yes
fi

echo ""
echo "OK -> $DEST"
echo ""
echo "Next steps:"
if [[ "$MODE" == gguf ]]; then
  echo "  # 1. import (no torch! generates mynah.json/tokens/mel filters + renamed model.gguf)"
  echo "  cd tools && uv sync && uv run python import_gguf.py ../$DEST/*.gguf --out ../$DEST && cd .."
  echo "  # 2. transcribe"
  echo "  ./mynah-asr transcribe -m $DEST -i file.wav"
  exit 0
fi
echo "  # 1. convert in place (one-time; Python via uv, offline tooling only)"
echo "  cd tools && uv sync && uv run python convert_nemo.py ../$DEST && cd .."
if [[ "$MODE" == nemo ]]; then
  echo "  #    (the .nemo archive can be deleted after conversion)"
fi
echo "  # 2. optional: pre-quantized int8/int4 checkpoint (~1/3 RAM, instant load)"
echo "  ./mynah-asr quantize -m $DEST --quant int8"
echo "  #    or a GGUF container (docs/gguf.md): cd tools && uv run python export_gguf.py ../$DEST --dtype q8_0"
echo "  # 3. transcribe"
echo "  ./mynah-asr transcribe -m $DEST -i file.wav --lang auto"
