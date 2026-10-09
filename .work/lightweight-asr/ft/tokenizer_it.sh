#!/bin/bash
# Step 3: train the `it` SentencePiece (training-pool text only), append it
# after fr in the aggregate tokenizer, copy back the 5248 old embedding/head
# rows, and VERIFY (old rows bit-identical, EN transcripts identical, it ids in
# range) before writing $FT/models/canary-180m-flash-it.nemo.
#   IT_VOCAB=1024 bash tokenizer_it.sh
. "$(dirname "$0")/common.sh"
if step tokenizer; then
    [ -e "$FT/done/data" ] || { echo "run data_it.sh first"; exit 2; }
    timeout "${T_TOK:-2400}" "$PY" "$KIT/tokenizer_it.py" "$@" || fail tokenizer $?
    ok tokenizer
fi
