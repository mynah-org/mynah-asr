#!/bin/bash
# S15-21 C: ADDITIVE external audio with E3-8k's absolute A2 exposure preserved.
# E3 recipe (stock -> whole encoder 2e-4, decoder+joint 3e-4, seed 0), one cosine over STEPS = 8000 / (1 - W):
# A2 keeps its four weights scaled by (1 - W), so the A2 draws = E3-8k's 8000 steps' worth; the external pool
# takes W of the draws ON TOP. External pool = E3G's pool unchanged (Granary-it agree<=15, teacher text, digit
# clips dropped, 98.1 h, max-dur 30), so vs E3G-8k the change is replacement -> addition (plus the extra steps),
# and vs E3-8k it is +external audio (plus the extra steps; unavoidable in an additive design).
# Default W=0.25 -> 10667 steps, ~1.5 passes over the pool. Read in processed hours (exposure_audit.py), not steps.
# Report: harness greedy/beam-8/oracle-8 AND clean-it (tedx, italic) greedy/beam-8/oracle-8; decisions at K=8.
#   FT_ROOT=/root/ft tmux new -d -s p1 'FT_ROOT=/root/ft bash finetune/jobs/p1_additive.sh 2>&1 | tee -a /root/ft/logs/p1_additive.log'
. "$(dirname "$0")/gpu_lock.sh"
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; YG=$FT_ROOT/mix/it-yg; R=$FT_ROOT/runs; K=$FT_ROOT/analysis/K
W=${W:-0.25}; STEPS=$(python3 -c "print(round(8000 / (1 - $W)))"); TAG=it-p1-add$(python3 -c "print(int($W * 100))")
MIX=$(python3 -c "
w = $W; a2 = [('$MF/train_40h.json', .35), ('$MX/train_cv.json', .30), ('$MX/train_vp_t50.json', .20), ('$MX/train_fleurs.json', .15)]
print(','.join(f'{p}:{x * (1 - w):.4f}' for p, x in a2) + f',$YG/subsets/yg_agree<=15.json:{w:.4f}')")
echo "== p1_additive W=$W STEPS=$STEPS TAG=$TAG $(date +%T)"; echo "   MIX=$MIX"
MIX="$MIX" POL="--freeze-enc 0 --enc-lr 2e-4 --lr 3e-4 --bs 16" STEPS=$STEPS SMOKE=1 bash "$FINETUNE/jobs/mix_arm.sh" $TAG \
    --max-dur 30 --eval-steps 100,250 --abort-wer 1000:85
rm -f $R/plain-$TAG/last.ckpt
[ -s $R/plain-$TAG/final.nemo ] || exit 3
gpu_lock
for bank in harness clean; do
    case $bank in
        harness) V="mls=$MF/eval_mls_it.json,fleurs=$MF/eval_fleurs_it.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json"; N=200 ;;
        clean) V="tedx=$MF/eval_tedx_it.json,italic=$MF/eval_italic_it.json"; N=100000 ;;
    esac
    timeout 3600 "$PY" "$FINETUNE/eou/diag/lm_fusion_probe.py" --nemo $R/plain-$TAG/final.nemo --vals "$V" --per-set $N \
        --alphas 0 --beam-alphas 0 --beams 8 --threads 32 --out $R/plain-$TAG/probe_$bank.json 2>&1 | grep -E "^\{|Traceback|Error" | cut -c1-400
done
echo "== P1-ADD-DONE $TAG $(date +%T)"
