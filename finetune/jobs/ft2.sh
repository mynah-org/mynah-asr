#!/bin/bash
# ft2: the Italian FT with a real schedule. The 600-step smoke runs (8.7 epochs
# of 5 h, 55 s of training) were clearly undertrained (loss ~2.0). Same
# processed audio (~260 h) per subset so the curve compares data, not compute:
#   5 h x 52 epochs, 20 h x 13, 40 h x 6.5 (rounded to 7).
# 5 h: both arms (A frozen encoder, B4 top-4 unfrozen) with the long schedule;
# 20/40 h: ARM_LONG only (B: better Italian in the smoke runs, more EN forgetting).
# Produced the 2026-10-09 data curve (finetune/README.md), ARM_LONG=B.
#   FT_ROOT=/root/ft ARM_LONG=B tmux new -d -s ft2 'bash finetune/jobs/ft2.sh 2>&1 | tee /root/ft/logs/ft2.log'
set -u
. "$(dirname "$0")/gpu_lock.sh"
gpu_lock
echo "== ft2 lock held $(date +%T) arm=${ARM_LONG:?}"
K=$FINETUNE/canary
for spec in "5h 52 A B" "20h 13 $ARM_LONG" "40h 7 $ARM_LONG"; do
    set -- $spec; sub=$1; ep=$2; shift 2
    echo "== ft2 $sub epochs=$ep arms=$* $(date +%T)"
    SUBSET=$sub EPOCHS=$ep ARMS="$*" TAG_SUFFIX=-e$ep RATE_USD_H=${RATE_USD_H:-0} bash $K/train_it.sh
done
echo "== FT2-DONE $(date +%T)"
