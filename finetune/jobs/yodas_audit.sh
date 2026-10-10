#!/bin/bash
# YODAS2 pseudo-label audit, one shard, NO filtering: VAD-cut clips (CPU), Nemotron 0.6B in the
# CUDA server with --lang auto (GPU lock), then the distributions (pseudo_stats.py) from which
# the gates are chosen.
#   FT_ROOT=/root/ft SHARD=0 tmux new -d -s yaudit 'FT_ROOT=/root/ft bash finetune/jobs/yodas_audit.sh 2>&1 | tee -a /root/ft/logs/yodas_audit.log'
. "$(dirname "$0")/gpu_lock.sh"
SHARD=${SHARD:-0}; N=$(printf %08d $SHARD); Y=$FT_ROOT/mix/it/yodas; L=$FT_ROOT/logs
cd "$FINETUNE/.." || exit 2; ulimit -n 65536
echo "== yodas shard $N: VAD cut $(date +%T)"
timeout 3600 "$PY" "$FINETUNE/eou/prepare_yodas.py" --lang it --shards $SHARD 2>&1 | grep -v -i warn
gpu_lock
./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 > $L/teacher_srv3.log 2>&1 & SP=$!
for i in $(seq 120); do curl -sf localhost:8295/v1/health >/dev/null && break; sleep 1; done
timeout 3600 python3 finetune/eou/diag/teacher_ws.py --manifest $Y/clips_$N.json --out $Y/clips_$N.teacher.json --lang auto --conc 128 | tail -2 | cut -c1-200
kill $SP; wait $SP 2>/dev/null; flock -u 9
python3 finetune/eou/diag/pseudo_stats.py --manifest $Y/clips_$N.teacher.json --subs $Y/subs_$N.json --json $Y/stats_$N.json
echo "== YODAS-AUDIT-DONE $N $(date +%T)"
