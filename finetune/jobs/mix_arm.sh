#!/bin/bash
# (negative ranges need the = form: --gain-db=-10,10; SMOKE=1 runs 20 steps of the policy first)
# One arm of the multi-domain EOU-IT ablation: experiment A0's data, sampler, seed, val sets,
# 4000 steps and top-4 policy, plus ONLY the extra flags given (the variable under test).
#   FT_ROOT=/root/ft tmux new -d -s arm 'FT_ROOT=/root/ft bash finetune/jobs/mix_arm.sh it-a1-gain10 --aug-gain 0.5 --gain-db=-10,10 2>&1 | tee -a /root/ft/logs/arm.log'
. "$(dirname "$0")/gpu_lock.sh"
TAG=$1; shift
L=$FT_ROOT/logs; MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; R=$FT_ROOT/runs
# VP=<manifest> swaps the VoxPopuli source (e.g. a teacher-filtered one) at the SAME weight
MIX=${MIX:-"$MF/train_40h.json:0.35,$MX/train_cv.json:0.30,${VP:-$MX/train_vp.json}:0.20,$MX/train_fleurs.json:0.15"}   # MIX= overrides the whole mix
# VALS= adds val sets (e.g. clean-it): two_stage.py val_wer is then the mean over ALL of them; compare per domain
VALS=${VALS:-"mls=$MF/eval_mls_it.json,fleurs=$MF/eval_fleurs_it.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json"}
POL=${POL:-"--freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5 --lr 3e-4 --bs 16"}
# disk gate: a run writes final.nemo (0.46 GB) + last.ckpt (1.38 GB); it-e2-enc2e-4-s4000 died of
# ENOSPC on the 32 GB root disk on 2026-10-11
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge 4 ] || { echo "== DISK GATE $TAG: ${FREE} GB free on $R, need 4"; exit 4; }
gpu_lock
if [ -n "${SMOKE:-}" ]; then   # 20 steps of the same policy first: a new code path fails in seconds, not at 20 %
    # the REAL mix (its longest clips set the memory peak), the same policy and extra flags
    timeout 900 "$PY" "$FINETUNE/eou/two_stage.py" --tag smoke-$TAG --out $R --mix "$MIX" $POL \
        --steps ${SMOKE_STEPS:-40} --eval-every ${SMOKE_STEPS:-40} --val $MF/eval_mls_it.json --val-n 16 "$@" 2>&1 \
        | grep -E "trainable|unfreeze|top-K|step [0-9]+/|VAL|done|Traceback|Error|nan|out of memory" > $L/smoke-$TAG.log
    cat $L/smoke-$TAG.log; rm -f $R/plain-smoke-$TAG/*.nemo $R/plain-smoke-$TAG/*.ckpt
    grep -q "plain-smoke-$TAG done" $L/smoke-$TAG.log || { echo "== SMOKE FAILED $TAG, stop"; exit 3; }
fi
echo "== $TAG $(date +%T): extra flags: $*"; ln -sf $L/$TAG.log $L/train_cur.log
STEPS=${STEPS:-4000}   # STEPS= lengthens the arm (cosine over the whole run, same eval cadence)
timeout $((STEPS * 2 + 1800)) "$PY" "$FINETUNE/eou/two_stage.py" --tag "$TAG" --out $R --mix "$MIX" $POL --steps $STEPS --eval-every 500 \
    --vals "$VALS" "$@" 2>&1 | grep -a --line-buffered -v -E "NeMo W|warn|Warning" > $L/$TAG.log
echo "   rc=${PIPESTATUS[0]}"; grep -E "done|Traceback|Error" $L/$TAG.log | tail -2
# summary only: the baseline columns exist on the box that ran them; a --final-eof run saves no "best" row (-> last step)
C=(); [ -s $L/base-p40.log ] && C+=(--col P40=$L/base-p40.log)
[ -s $R/plain-it-a0-mix/metrics.json ] && C+=(--col A0@3000=$R/plain-it-a0-mix/metrics.json:3000)
B=$(python3 -c "import json; l=json.load(open('$R/plain-$TAG/metrics.json'))['log']; s=[r for r in l if r.get('saved')] or [r for r in l if 'step' in r]; print(s[-1]['step'])" 2>/dev/null)
[ -n "$B" ] && python3 "$FINETUNE/eou/val_table.py" "${C[@]}" --col "$TAG@$B=$R/plain-$TAG/metrics.json:$B" \
    $([ -s $R/plain-it-a0-mix/metrics.json ] && echo --delta "$TAG@$B-A0@3000") --trajectory $R/plain-$TAG/metrics.json \
    || echo "== val_table summary skipped (diagnostic only)"
echo "== ARM-DONE $TAG $(date +%T)"
