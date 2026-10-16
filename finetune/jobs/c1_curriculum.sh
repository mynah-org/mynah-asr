#!/bin/bash
# Phase C1: difficulty curriculum INSIDE each source of the C250 mix, W4 at 8k, vs W4-C250-8k (same mix marginals,
# recipe, seed, steps). Needs DIR/mix.txt from curriculum_mix.py --mult (frozen multipliers, pre-registered).
# Screening harness policy: evals every 2000 steps in the flat phase and every 500 in the decay (diagnostic only),
# final.nemo = the last step (--final-eof 1). Probes: harness + clean-it, greedy / beam-8 / oracle-8, raw and 0.5 s flush.
#   FT_ROOT=/root/ft CUR=/root/ft/cur/c1 tmux new -d -s c1 'bash finetune/jobs/c1_curriculum.sh 2>&1 | tee -a /root/ft/logs/c1.log'
. "$(dirname "$0")/gpu_lock.sh"
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; R=$FT_ROOT/runs; TAG=${TAG:-it-c1-cur-8k}
MIX=$(cat ${CUR:?}/mix.txt) || exit 2
echo "== c1 $TAG $(date +%T)"; echo "   MIX=$MIX"
MIX="$MIX" POL="--freeze-enc 0 --enc-lr 4e-4 --lr 3e-4 --bs 16" STEPS=${STEPS:-8000} SMOKE=1 bash "$FINETUNE/jobs/mix_arm.sh" $TAG \
    --max-dur 30 --eval-steps 100,250,1000 --abort-wer 1000:85 --lr-shape wsd --decay-frac 0.2 \
    --eval-sparse 2000,500 --final-eof 1 --seed ${SEED:-0}
rm -f $R/plain-$TAG/last.ckpt
[ -s $R/plain-$TAG/final.nemo ] || exit 3
gpu_lock
for post in 0 0.5; do
sfx=$([ "$post" = 0 ] || echo "_post$post")
for bank in harness clean; do
    case $bank in
        harness) V="mls=$MF/eval_mls_it.json,fleurs=$MF/eval_fleurs_it.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json"; N=200 ;;
        clean) V="tedx=$MF/eval_tedx_it.json,italic=$MF/eval_italic_it.json"; N=100000 ;;
    esac
    timeout 3600 "$PY" "$FINETUNE/eou/diag/lm_fusion_probe.py" --nemo $R/plain-$TAG/final.nemo --vals "$V" --per-set $N \
        --alphas 0 --beam-alphas 0 --beams 8 --threads 32 --post-pad-s $post --out $R/plain-$TAG/probe_$bank$sfx.json 2>&1 \
        | grep -E "^\{|Traceback|Error" | sed "s/^/post=$post /" | cut -c1-400
done
done
echo "== C1-DONE $TAG $(date +%T)"
