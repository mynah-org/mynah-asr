#!/bin/bash
# EOU stage 1, ablation-ladder step 1 (EXPERIMENTAL): micro-overfit the STOCK
# parakeet_realtime_eou_120m-v1 on 32 Italian utterances with the stock
# tokenizer, no padding, FastEmit 0, frozen encoder, lr 1e-4. A healthy RNNT
# must memorise them (train WER -> ~0, non-blank frames > 0, distinct
# hypotheses per clip); if it does not, the loop is broken before any EOU work.
# Env: FT_ROOT (required), MANIFEST (default $FT_ROOT/manifests/train_5h.json),
#      VAL (optional manifest), TAG (mo32), T_TRAIN (3600 s), EXTRA (more flags).
#   FT_ROOT=/root/ft tmux new -d -s mo 'bash finetune/jobs/micro_overfit.sh 2>&1 | tee -a /root/ft/logs/mo.log'
set -u
. "$(dirname "$0")/gpu_lock.sh"
STOCK=$FT_ROOT/models/parakeet_realtime_eou_120m-v1.nemo
MANIFEST=${MANIFEST:-$FT_ROOT/manifests/train_5h.json}
[ -s "$STOCK" ] || { echo "missing $STOCK (eou1.sh downloads it)"; exit 2; }
[ -s "$MANIFEST" ] || { echo "missing $MANIFEST (canary/prepare_it.sh builds it)"; exit 2; }
"$PY" "$FINETUNE/eou/plain_asr.py" --stock "$STOCK" --manifest "$MANIFEST" --out "$FT_ROOT/runs" \
    --tag "${TAG:-mo32}" --micro-overfit ${VAL:+--val "$VAL"} ${EXTRA:-} --dry-run || exit $?
gpu_lock
# shellcheck disable=SC2086  # EXTRA is a word list on purpose
timeout "${T_TRAIN:-3600}" "$PY" "$FINETUNE/eou/plain_asr.py" --stock "$STOCK" --manifest "$MANIFEST" \
    --out "$FT_ROOT/runs" --tag "${TAG:-mo32}" --micro-overfit ${VAL:+--val "$VAL"} ${EXTRA:-}
echo "== MICRO-OVERFIT-DONE rc=$? $(date +%T)"
