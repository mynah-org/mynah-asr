#!/bin/bash
# Multi-domain data for the EOU 120M specialist, one language: MLS (canary/prepare_it.sh,
# the nested 5/20/40 h cuts + frozen evals of 2026-10-09) and the non-MLS domains
# (eou/prepare_mix.py: Common Voice 17, VoxPopuli, FLEURS train) in parallel: both are
# network/CPU only, no GPU lock.
#   FT_ROOT=/root/ft tmux new -d -s data 'bash finetune/jobs/data_mix.sh it 2>&1 | tee -a /root/ft/logs/data_mix.log'
. "$(dirname "$0")/gpu_lock.sh"
LANG_=${1:-it}
echo "== data_mix $LANG_ $(date +%T)"
if [ "$LANG_" = it ]; then
    (timeout 7200 bash "$FINETUNE/canary/prepare_it.sh" > "$FT_ROOT/logs/data_mls.log" 2>&1; echo "== mls rc=$? $(date +%T)") &
fi
timeout 10800 "$PY" "$FINETUNE/eou/prepare_mix.py" --lang "$LANG_" ${MIX_ARGS:-} 2>&1 | grep --line-buffered -v -E "Warning|warn"
echo "== mix rc=${PIPESTATUS[0]} $(date +%T)"
wait
df -h / /dev/shm | tail -2
echo "== DATA-MIX-JOB-DONE $(date +%T)"
