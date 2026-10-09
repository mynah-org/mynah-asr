#!/bin/bash
# Step 5: smoke fine-tunes, cheapest arm first.
#   A   = frozen encoder, train decoder + new it rows
#   B<N>= A + top N encoder layers unfrozen (encoder LR x ENC_LR_SCALE)
# Env: SUBSET (5h|20h|40h, default 5h), ARMS (default "A B"), TOP_N (4),
#      TAG_SUFFIX (appended to the run tag, e.g. -e50 for a longer schedule),
#      EPOCHS (10), BATCH_DURATION (300 s of audio per batch; fits a 24 GB L4),
#      LR (2e-4), RATE_USD_H (box price per GPU-hour, for the cost fields),
#      T_TRAIN (timeout per run, s), GAIN_AUG (0|1: random gain -30..+6 dB on
#      training batches; recorded A/B option, tag gets "-gain"), LEVELS
#      (peak dBFS list for the before/after level sweep, default -3,-20,-40).
#   RATE_USD_H=0.85 tmux new -d -s ft 'bash train_canary_it.sh 2>&1 | tee -a /root/ft/logs/train.log'
. "$(dirname "$0")/common.sh"
[ -e "$FT/done/tokenizer" ] || { echo "run tokenizer_it.sh first"; exit 2; }
[ -n "${RATE_USD_H:-}" ] || echo "WARNING: RATE_USD_H unset; cost fields will be 0"
SUBSET=${SUBSET:-5h}; TOP_N=${TOP_N:-4}
for arm in ${ARMS:-A B}; do
    tag="$arm$([ "$arm" = B ] && echo "$TOP_N")-$SUBSET$([ "${GAIN_AUG:-0}" = 1 ] && echo -gain)${TAG_SUFFIX:-}"
    if step "train-$tag"; then
        need_gb 3 "a run (.nemo ~0.75 GB + hypotheses)"
        timeout "${T_TRAIN:-14400}" "$PY" "$KIT/train_canary_it.py" --arm "$arm" --top-n "$TOP_N" --subset "$SUBSET" \
            --epochs "${EPOCHS:-10}" --batch-duration "${BATCH_DURATION:-300}" --lr "${LR:-2e-4}" \
            --enc-lr-scale "${ENC_LR_SCALE:-0.3}" --rate-usd-h "${RATE_USD_H:-0}" --gain-aug "${GAIN_AUG:-0}" --tag "$tag" 2>&1 \
            | tee "$FT/logs/train-$tag.log" | grep --line-buffered -E "^(==|  step|  VAL|  \[|Traceback|[A-Za-z]*Error)"
        rc=${PIPESTATUS[0]}
        [ "$rc" = 0 ] || fail "train-$tag" "$rc"
        ok "train-$tag"
    fi
done
"$PY" "$KIT/report.py"
