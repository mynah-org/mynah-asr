#!/bin/bash
# Granary-it (espnet/yodas-granary) pool for the scaling curve: SPREAD shards over the whole
# language (video diversity), clips 1..YG_MAX_S s, every clip scored by the teacher (Nemotron
# 0.6B, CUDA server, forced --lang it) -> train_yg.teacher.json with the two-teacher agreement
# (teacher vs Granary-Whisper WER) per clip; NO filter here (thresholds are cheap later).
#   FT_ROOT=/root/ft YG_SPREAD=70 tmux new -d -s gbuild 'FT_ROOT=/root/ft bash finetune/jobs/granary_build.sh 2>&1 | tee -a /root/ft/logs/granary_build.log'
# Scale mode (prepare_mix.py incremental mode), on when any of YG_DIRS / YG_SPLIT / YG_EXT / YG_HOURS is set:
#   YG_DIRS=it000,it100,it101 YG_SPREAD=70,250,250 (one N per dir) YG_SPLIT=20 (clips > 20 s cut at
#   pauses) YG_EXT=flac YG_HOURS=450 (no new shard once the pool has 450 h)
#   [YG_MIN_FREE_GB=4 YG_WORKERS=6 YG_TIMEOUT=10800 TEACHER_TIMEOUT=7200 SHM_MIN_GB=2]
# Restartable and growable: done shards are cached in $O/yg_shards; the teacher reuses every score it
# has ($O/teacher_archive.json = the archived pool scores, copied once; the last output; an interrupted
# run's .tmp), so a rerun with a larger YG_HOURS only downloads and scores the new shards.
. "$(dirname "$0")/gpu_lock.sh"
# YG_OUT / YG_AUDIO: build a new pool next to the one a running arm reads (the FLAC mode deletes the
# .wav copy of every clip it rewrites)
SPREAD=${YG_SPREAD:-70}; MAXS=${YG_MAX_S:-30}; O=${YG_OUT:-$FT_ROOT/mix/it-yg}; AUD=${YG_AUDIO:-/dev/shm/mix/it-yg}; L=$FT_ROOT/logs
cd "$FINETUNE/.." || exit 2; ulimit -n 65536
NEW=; X=(); TPY=python3
[ -n "$YG_DIRS" ] && { NEW=1; X+=(--yg-dirs "$YG_DIRS"); }
[ -n "$YG_SPLIT" ] && { NEW=1; X+=(--yg-split-long "$YG_SPLIT"); }
[ -n "$YG_EXT" ] && { NEW=1; X+=(--audio-ext "$YG_EXT"); }
[ -n "$YG_HOURS" ] && { NEW=1; X+=(--yg-hours "$YG_HOURS"); }
[ -n "$NEW" ] && { X+=(--yg-min-free-gb "${YG_MIN_FREE_GB:-4}"); TPY=$PY; }   # $PY has soundfile (FLAC)
echo "== granary build: $SPREAD spread shards, clips <= $MAXS s ${X[*]:+${X[*]} }$(date +%T)"
timeout "${YG_TIMEOUT:-10800}" "$PY" "$FINETUNE/eou/prepare_mix.py" --lang it --out $O --audio $AUD --cv-hours 0 --vp-hours 0 \
    --no-fleurs --yg-spread $SPREAD --yg-max-s $MAXS --yg-keep-all --workers "${YG_WORKERS:-6}" "${X[@]}" 2>&1 | grep -E "^== yg|DATA-MIX|Traceback|Error"
RUN=; [ ! -s $O/train_yg.teacher.json ] && RUN=1   # resume_it.sh puts the archived pool scores there
if [ -n "$NEW" ]; then   # score only the clips no previous output covers
    [ -s $O/train_yg.json ] || { echo "== no $O/train_yg.json"; exit 3; }
    [ -s $O/train_yg.teacher.json ] && [ ! -s $O/teacher_archive.json ] && cp $O/train_yg.teacher.json $O/teacher_archive.json
    MISS=$("$PY" - $O/train_yg.json $O/train_yg.teacher.json <<'PY'
import json, os, sys
have = {json.loads(l)["audio_filepath"] for l in open(sys.argv[2])} if os.path.exists(sys.argv[2]) else set()
print(sum(1 for l in open(sys.argv[1]) if json.loads(l)["audio_filepath"] not in have))
PY
)
    echo "== teacher: $MISS clips of train_yg.json not scored yet"
    RUN=; [ "$MISS" != 0 ] && RUN=1
    REUSE=; for f in $O/teacher_archive.json $O/train_yg.teacher.json $O/train_yg.teacher.json.tmp; do [ -s $f ] && REUSE=$REUSE${REUSE:+,}$f; done
    df -h /dev/shm | tail -1
    FREE=$(df -Pk /dev/shm | awk 'NR==2 {print int($4 / 1048576)}')
    [ -n "$RUN" ] && [ "$FREE" -lt "${SHM_MIN_GB:-2}" ] && { echo "== /dev/shm: ${FREE} GB free < ${SHM_MIN_GB:-2} GB, stop before the teacher"; exit 4; }
fi
if [ -n "$RUN" ]; then
gpu_lock
./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 > $L/teacher_srv4.log 2>&1 & SP=$!
for i in $(seq 120); do curl -sf localhost:8295/v1/health >/dev/null && break; sleep 1; done
timeout "${TEACHER_TIMEOUT:-7200}" $TPY finetune/eou/diag/teacher_ws.py --manifest $O/train_yg.json --out $O/train_yg.teacher.json --lang it --conc 128 \
    ${REUSE:+--reuse $REUSE} | tail -1 | cut -c1-220
kill $SP; wait $SP 2>/dev/null; flock -u 9
fi
python3 finetune/eou/diag/pseudo_stats.py --manifest $O/train_yg.teacher.json --json $O/stats_yg.json | head -60
df -h / /dev/shm | tail -2
echo "== GRANARY-BUILD-DONE $(date +%T)"
