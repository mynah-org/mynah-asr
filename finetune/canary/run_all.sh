#!/bin/bash
# Whole kit in order; every step is skipped if its marker exists.
#   FT_ROOT=/root/ft SUBSET=5h RATE_USD_H=0.85 \
#     tmux new -d -s ft 'bash <kit>/finetune/canary/run_all.sh 2>&1 | tee -a /root/ft/logs/run_all.log'
# FT_ROOT and SUBSET are required (no implicit multi-hour run).
. "$(dirname "$0")/env.sh"
echo "== run_all start $(date '+%F %T') FT=$FT KIT=$KIT"
bash "$KIT/setup.sh" || exit $?
# shellcheck disable=SC2086  # DATA_ARGS is a word list on purpose
bash "$KIT/prepare_it.sh" ${DATA_ARGS:-} || exit $?
# One GPU workload at a time on the box: every CUDA step holds /root/gpu.lock
# (the same lock the benchmark and gate jobs take); setup and data stay outside.
GPU_LOCK=${GPU_LOCK:-/root/gpu.lock}
exec 9>"$GPU_LOCK"; echo "== waiting for the GPU lock $(date '+%T')"; flock 9; echo "== GPU lock held $(date '+%T')"
bash "$KIT/tokenizer_it.sh" || exit $?
bash "$KIT/probe_it.sh" >"$FT/logs/probe.log" 2>&1 || { tail -30 "$FT/logs/probe.log"; exit 1; }
grep -E "^\s+\[|^== arm|wrote" "$FT/logs/probe.log"
bash "$KIT/train_it.sh" || exit $?
df -h "$FT" | tail -1
echo "== RUN-ALL-DONE $(date '+%F %T')"
