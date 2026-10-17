#!/bin/bash
# Phase D preparation: the it-yg7 candidate pool, scored by the teacher AND the frozen student, then the bucket table
# (d_select.py stats) -- the data distribution is read BEFORE the D0 / D1 amount is chosen and pre-registered.
# 1. granary_build.sh: it-yg7 = Granary-it it000/it100/it101 spread 300/300/300 minus the shards of it-yg / it-yg6
#    (0 shard / video overlap with the C250 pools), 500 h, clips <= 30 s, stored as FLAC on the root disk; teacher pass;
# 2. pseudo_subsets.py (C250 defaults: teacher text as label, digit clips dropped) -> subsets/yg_all.json;
# 3. student_score.py with the FROZEN scorer W4-C250-16k (STUDENT) on yg_all.json;
# 4. d_select.py stats -> $D/d_stats.md.
# The arms: d_select.py --hours H --write, then MIXFILE=$D/mix_D0.txt / mix_D1.txt bash c1_curriculum.sh.
#   tmux new -d -s dprep 'FT_ROOT=/root/ft bash /root/mynah-asr/finetune/jobs/d_yg7.sh 2>&1 | tee -a /root/ft/logs/d_yg7.log'
. "$(dirname "$0")/gpu_lock.sh"
set -u
POOL=${POOL:-$FT_ROOT/mix/it-yg7}; D=${D:-$FT_ROOT/d/yg7}
STUDENT=${STUDENT:-$FT_ROOT/runs/plain-it-w4-c250-16k/final.nemo}
cd "$FINETUNE/.." || exit 2
[ -s "$STUDENT" ] || { echo "== no frozen scorer $STUDENT"; exit 2; }
mkdir -p /root/disk/mix/it-yg7 $D
YG_OUT=$POOL YG_AUDIO=/root/disk/mix/it-yg7 YG_EXT=flac YG_DIRS=it000,it100,it101 YG_SPREAD=300,300,300 \
    YG_EXCLUDE=it000:70,it100:60,it101:55 YG_HOURS=500 YG_WORKERS=${YG_WORKERS:-8} YG_MIN_FREE_GB=${YG_MIN_FREE_GB:-10} \
    YG_TIMEOUT=14400 TEACHER_TIMEOUT=14400 bash "$FINETUNE/jobs/granary_build.sh" | tee -a $FT_ROOT/logs/granary_build_it-yg7.log
[ -s $POOL/train_yg.teacher.json ] || { echo "== no teacher scores"; exit 3; }
"$PY" "$FINETUNE/eou/pseudo_subsets.py" --pool $POOL/train_yg.teacher.json --out $POOL/subsets | tail -12
gpu_lock
timeout 3600 "$PY" "$FINETUNE/eou/diag/student_score.py" --nemo "$STUDENT" --manifests "yg7=$POOL/subsets/yg_all.json" \
    --out $POOL/student_w4_16k.jsonl --bs 64 | grep -E "^==|Traceback|Error"
flock -u 9
python3 "$FINETUNE/eou/d_select.py" --pool $POOL/subsets/yg_all.json --student $POOL/student_w4_16k.jsonl --out $D \
    $([ -s $D/dup_groups.json ] && echo --dups $D/dup_groups.json)
sha256sum $POOL/subsets/yg_all.json $POOL/student_w4_16k.jsonl "$STUDENT" | tee $D/hashes.txt
echo "== D-PREP-DONE $(date +%T)"
