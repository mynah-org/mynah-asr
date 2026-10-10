#!/bin/bash
# Pseudo-label audit of Granary's Italian YODAS (espnet/yodas-granary, pre-segmented clips with
# Whisper-v3 labels), NO filtering: 3 spread shards of <lang>000/ast (~15 h), every 1-20 s clip
# (digits kept), Nemotron 0.6B in the CUDA server on each clip (forced --lang it), then the
# distributions (pseudo_stats.py): two-teacher agreement (Nemotron vs Granary-Whisper WER) with the
# hours each threshold keeps, words/s, repeats, empties. The gates are chosen from this.
# (prepare_yodas.py cuts raw YODAS2 long-form audio; not needed for Italian since Granary exists.)
#   FT_ROOT=/root/ft tmux new -d -s yaudit 'FT_ROOT=/root/ft bash finetune/jobs/yodas_audit.sh 2>&1 | tee -a /root/ft/logs/yodas_audit.log'
. "$(dirname "$0")/gpu_lock.sh"
SHARDS=${SHARDS:-0,60,120}; O=$FT_ROOT/mix/it-ygaudit; L=$FT_ROOT/logs
cd "$FINETUNE/.." || exit 2; ulimit -n 65536
echo "== yodas-granary audit shards $SHARDS $(date +%T)"
timeout 3600 "$PY" "$FINETUNE/eou/prepare_mix.py" --lang it --out $O --audio /dev/shm/mix/it-ygaudit --cv-hours 0 --vp-hours 0 \
    --no-fleurs --yg-shards $SHARDS --yg-keep-all 2>&1 | grep -E "yg|==" | grep -v skip
gpu_lock
./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 > $L/teacher_srv3.log 2>&1 & SP=$!
for i in $(seq 120); do curl -sf localhost:8295/v1/health >/dev/null && break; sleep 1; done
timeout 3600 python3 finetune/eou/diag/teacher_ws.py --manifest $O/train_yg.json --out $O/train_yg.teacher.json --lang it --conc 128 | tail -1 | cut -c1-220
kill $SP; wait $SP 2>/dev/null; flock -u 9
python3 finetune/eou/diag/pseudo_stats.py --manifest $O/train_yg.teacher.json --json $O/stats_yg.json
echo "== YODAS-AUDIT-DONE $(date +%T)"
