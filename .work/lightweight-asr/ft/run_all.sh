#!/bin/bash
# Whole kit in order; every step is skipped if its marker exists.
#   RATE_USD_H=0.85 tmux new -d -s ft 'bash /root/ft-kit/run_all.sh 2>&1 | tee -a /root/ft/logs/run_all.log'
. "$(dirname "$0")/common.sh"
echo "== run_all start $(date '+%F %T') FT=$FT KIT=$KIT"
bash "$KIT/setup.sh" || exit $?
# shellcheck disable=SC2086  # DATA_ARGS is a word list on purpose
bash "$KIT/data_it.sh" ${DATA_ARGS:-} || exit $?
bash "$KIT/tokenizer_it.sh" || exit $?
bash "$KIT/probe.sh" >"$FT/logs/probe.log" 2>&1 || { tail -30 "$FT/logs/probe.log"; exit 1; }
grep -E "^\s+\[|^== arm|wrote" "$FT/logs/probe.log"
bash "$KIT/train_canary_it.sh" || exit $?
df -h "$FT" | tail -1
echo "== RUN-ALL-DONE $(date '+%F %T')"
