#!/bin/bash
# Multi-domain data for the EOU 120M specialist, one language (it | fr): MLS (canary/prepare_it.sh [--lang fr],
# the nested 5/20/40 h cuts + frozen evals of 2026-10-09) and the non-MLS domains
# (eou/prepare_mix.py: Common Voice 17, VoxPopuli, FLEURS train) in parallel: both are
# network/CPU only, no GPU lock.
#   FT_ROOT=/root/ft tmux new -d -s data 'bash finetune/jobs/data_mix.sh it 2>&1 | tee -a /root/ft/logs/data_mix.log'
#   fr (A2-fr): ... data_mix.sh fr  -> manifests/*_fr.json (MLS) + mix/fr (CV 40 h <= 20/spk, VP 40 h <= 60/spk, FLEURS)
. "$(dirname "$0")/gpu_lock.sh"
LANG_=${1:-it}
echo "== data_mix $LANG_ $(date +%T)"
if [ "$LANG_" = it ]; then
    (timeout 7200 bash "$FINETUNE/canary/prepare_it.sh" > "$FT_ROOT/logs/data_mls.log" 2>&1; echo "== mls rc=$? $(date +%T)") &
else   # FR-P1: MLS-<lang> 40 h cut over all speakers (all train shards are touched: longer than IT)
    (T_DATA=${T_DATA_MLS:-14400} timeout ${T_DATA_MLS:-14400} bash "$FINETUNE/canary/prepare_it.sh" --lang "$LANG_" \
        > "$FT_ROOT/logs/data_mls_$LANG_.log" 2>&1; echo "== mls $LANG_ rc=$? $(date +%T)") &
fi
timeout 10800 "$PY" "$FINETUNE/eou/prepare_mix.py" --lang "$LANG_" ${MIX_ARGS:-} 2>&1 | grep --line-buffered -v -E "Warning|warn"
echo "== mix rc=${PIPESTATUS[0]} $(date +%T)"
wait
df -h / /dev/shm | tail -2
echo "== DATA-MIX-JOB-DONE $(date +%T)"
