#!/bin/bash
# S15-21 scaling of the additive regime (C): build a NEW-video Granary-it pool with the C recipe, then train
# p1_additive.sh on the old pool + the new one at the same passes (was /root/res/c250.sh, 2026-10-12).
# 1. granary_build.sh into $POOL (new dirs only: YG_DIRS / YG_SPREAD / YG_HOURS; clips <= 30 s, FLAC, teacher pass);
# 2. pseudo_subsets.py: agree<=15 subset (teacher text, digit clips dropped = the C recipe); gate MIN_H new hours;
# 3. p1_additive.sh EXT=<old agree<=15>,<new agree<=15> PASSES (A2 stays at E3-8k's exposure);
# 4. HF archive of the new run's final.nemo (close-out by hand after review).
# Needs bandwidth: ~0.6 GB per shard downloaded (2026-10-12 evening the box fell to ~1.5 MB/s: check first).
#   FT_ROOT=/root/ft TAG=it-c250 YG_DIRS=it100,it101 YG_SPREAD=60,55 YG_HOURS=235 \
#     tmux new -d -s cscale 'FT_ROOT=/root/ft bash finetune/jobs/c_scale.sh 2>&1 | tee -a /root/ft/logs/c_scale.log'
. "$(dirname "$0")/gpu_lock.sh"
set -u
POOL=${POOL:-$FT_ROOT/mix/it-yg6}; TAG=${TAG:-it-c250}; MIN_H=${MIN_H:-60}; PASSES=${PASSES:-1.5}
cd "$FINETUNE/.." || exit 2
YG_OUT=$POOL YG_AUDIO=/dev/shm/mix/$(basename $POOL) YG_EXT=flac YG_DIRS=${YG_DIRS:-it100,it101} YG_SPREAD=${YG_SPREAD:-60,55} \
    YG_HOURS=${YG_HOURS:-235} YG_WORKERS=${YG_WORKERS:-8} YG_MIN_FREE_GB=4 YG_TIMEOUT=${YG_TIMEOUT:-10800} TEACHER_TIMEOUT=7200 \
    bash "$FINETUNE/jobs/granary_build.sh" | tee -a $FT_ROOT/logs/granary_build_$(basename $POOL).log
"$PY" "$FINETUNE/eou/pseudo_subsets.py" --pool $POOL/train_yg.teacher.json --out $POOL/subsets | tail -12
N="$POOL/subsets/yg_agree<=15.json"
H=$(python3 -c "import json, sys; print(round(sum(json.loads(l)['duration'] for l in open(sys.argv[1])) / 3600, 1))" "$N")
echo "== new agree<=15 hours: $H"
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$H" "$MIN_H" || { echo "== NEW POOL TOO SMALL ($H h), stop"; exit 3; }
EXT="$FT_ROOT/mix/it-yg/subsets/yg_agree<=15.json,$N" PASSES=$PASSES TAG=$TAG bash "$FINETUNE/jobs/p1_additive.sh"
timeout 3600 "$PY" "$FINETUNE/common/hf_archive.py" --repo ${HUB:-gabrione/mynah-asr-canary-it-experiments} --ft-root $FT_ROOT \
    --token-file /root/.hf_token --nemo-tags "^plain-$TAG\$" --ckpt-tags "^\$" 2>&1 | grep -E "$TAG|VERIFY|pass done" | grep -v "%"
echo "== C-SCALE-DONE $TAG $(date +%T)"
