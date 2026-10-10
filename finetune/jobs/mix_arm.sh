#!/bin/bash
# (negative ranges need the = form: --gain-db=-10,10)
# One arm of the multi-domain EOU-IT ablation: experiment A0's data, sampler, seed, val sets,
# 4000 steps and top-4 policy, plus ONLY the extra flags given (the variable under test).
#   FT_ROOT=/root/ft tmux new -d -s arm 'FT_ROOT=/root/ft bash finetune/jobs/mix_arm.sh it-a1-gain10 --aug-gain 0.5 --gain-db=-10,10 2>&1 | tee -a /root/ft/logs/arm.log'
. "$(dirname "$0")/gpu_lock.sh"
TAG=$1; shift
L=$FT_ROOT/logs; MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; R=$FT_ROOT/runs
# VP=<manifest> swaps the VoxPopuli source (e.g. a teacher-filtered one) at the SAME weight
MIX="$MF/train_40h.json:0.35,$MX/train_cv.json:0.30,${VP:-$MX/train_vp.json}:0.20,$MX/train_fleurs.json:0.15"
VALS="mls=$MF/eval_mls_it.json,fleurs=$MF/eval_fleurs_it.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json"
POL=${POL:-"--freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5 --lr 3e-4 --bs 16"}
gpu_lock
echo "== $TAG $(date +%T): extra flags: $*"; ln -sf $L/$TAG.log $L/train_cur.log
timeout 7200 "$PY" "$FINETUNE/eou/two_stage.py" --tag "$TAG" --out $R --mix "$MIX" $POL --steps 4000 --eval-every 500 \
    --vals "$VALS" "$@" 2>&1 | grep --line-buffered -v -E "NeMo W|warn|Warning" > $L/$TAG.log
echo "   rc=${PIPESTATUS[0]}"; grep -E "done|Traceback|Error" $L/$TAG.log | tail -2
python3 "$FINETUNE/eou/val_table.py" --col P40=$L/base-p40.log --col A0@3000=$R/plain-it-a0-mix/metrics.json:3000 \
    --col "$TAG@best=$R/plain-$TAG/metrics.json:$(python3 -c "import json; print([r for r in json.load(open('$R/plain-$TAG/metrics.json'))['log'] if r.get('saved')][-1]['step'])")" \
    --delta "$TAG@best-A0@3000" --trajectory $R/plain-$TAG/metrics.json
echo "== ARM-DONE $TAG $(date +%T)"
