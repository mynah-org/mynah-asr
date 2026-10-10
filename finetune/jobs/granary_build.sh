#!/bin/bash
# Granary-it (espnet/yodas-granary) pool for the scaling curve: SPREAD shards over the whole
# language (video diversity), clips 1..YG_MAX_S s, every clip scored by the teacher (Nemotron
# 0.6B, CUDA server, forced --lang it) -> train_yg.teacher.json with the two-teacher agreement
# (teacher vs Granary-Whisper WER) per clip; NO filter here (thresholds are cheap later).
#   FT_ROOT=/root/ft YG_SPREAD=70 tmux new -d -s gbuild 'FT_ROOT=/root/ft bash finetune/jobs/granary_build.sh 2>&1 | tee -a /root/ft/logs/granary_build.log'
. "$(dirname "$0")/gpu_lock.sh"
SPREAD=${YG_SPREAD:-70}; MAXS=${YG_MAX_S:-30}; O=$FT_ROOT/mix/it-yg; L=$FT_ROOT/logs
cd "$FINETUNE/.." || exit 2; ulimit -n 65536
echo "== granary build: $SPREAD spread shards, clips <= $MAXS s $(date +%T)"
timeout 10800 "$PY" "$FINETUNE/eou/prepare_mix.py" --lang it --out $O --audio /dev/shm/mix/it-yg --cv-hours 0 --vp-hours 0 \
    --no-fleurs --yg-spread $SPREAD --yg-max-s $MAXS --yg-keep-all --workers 6 2>&1 | grep -E "^== yg|DATA-MIX|Traceback|Error"
gpu_lock
./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 > $L/teacher_srv4.log 2>&1 & SP=$!
for i in $(seq 120); do curl -sf localhost:8295/v1/health >/dev/null && break; sleep 1; done
timeout 7200 python3 finetune/eou/diag/teacher_ws.py --manifest $O/train_yg.json --out $O/train_yg.teacher.json --lang it --conc 128 | tail -1 | cut -c1-220
kill $SP; wait $SP 2>/dev/null; flock -u 9
python3 finetune/eou/diag/pseudo_stats.py --manifest $O/train_yg.teacher.json --json $O/stats_yg.json | head -60
df -h / /dev/shm | tail -2
echo "== GRANARY-BUILD-DONE $(date +%T)"
