#!/bin/bash
# S15-21 P0a + P1/P2: does teacher-labelled audio ADD information beyond more optimisation?
# Two fresh 8000-step arms (cosine over 8000, seed 0), each the 2x-steps twin of an existing 4k run:
#   it-e3-s8000   E3 recipe, A2 only                         (twin of E3  = plain-it-e2-enc2e-4-s4000, 36.52)
#   it-e3g-s8000  E3G mix: A2 x0.5 + Granary-it teacher labels (twin of E3G = plain-it-e3g-yg15, 36.55)
#                 agree<=15, max-dur 30
# -> a 2x2 {4k, 8k} x {A2, A2 + 50 % teacher-labelled YODAS draws}. Readings (each changes something else too):
#   E3-4k -> E3-8k    optimisation budget / A2 re-exposure scaling;
#   E3G-4k -> E3G-8k  scaling of the mixed regime;
#   E3-4k <-> E3G-8k  approximately E3's integrated A2 exposure PLUS Granary, but steps/schedule change too;
#   E3-8k <-> E3G-8k  the mix choice at equal optimizer budget (E3G has HALF the A2 exposure): not a pure
#                     "add data" test. Pre-registered for it: <= -1.0 macro (2x the noise floor) = the mixed
#                     regime wins; |diff| < 1.0 = inconclusive (one seed).
# After each arm: CPU greedy / beam-8 / 8-best oracle probe (alpha 0, no LM).
#   FT_ROOT=/root/ft tmux new -d -s s8k 'FT_ROOT=/root/ft bash finetune/jobs/scale_8k.sh 2>&1 | tee -a /root/ft/logs/scale_8k.log'
. "$(dirname "$0")/gpu_lock.sh"
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; YG=$FT_ROOT/mix/it-yg; R=$FT_ROOT/runs
A2="$MF/train_40h.json:0.35,$MX/train_cv.json:0.30,$MX/train_vp_t50.json:0.20,$MX/train_fleurs.json:0.15"
A2H="$MF/train_40h.json:0.175,$MX/train_cv.json:0.15,$MX/train_vp_t50.json:0.10,$MX/train_fleurs.json:0.075"
POL="--freeze-enc 0 --enc-lr 2e-4 --lr 3e-4 --bs 16"
for arm in ${ARMS:-it-e3-s8000 it-e3g-s8000}; do
    case $arm in
        it-e3-s8000) MIX="$A2" POL="$POL" STEPS=8000 bash "$FINETUNE/jobs/mix_arm.sh" $arm --eval-steps 100,250 --abort-wer 1000:85 ;;
        it-e3g-s8000) MIX="$A2H,$YG/subsets/yg_agree<=15.json:0.5" POL="$POL" STEPS=8000 SMOKE=1 bash "$FINETUNE/jobs/mix_arm.sh" $arm \
            --max-dur 30 --eval-steps 100,250 --abort-wer 1000:85 ;;
    esac
    [ -s $R/plain-$arm/final.nemo ] && CUDA_VISIBLE_DEVICES= timeout 3600 "$PY" "$FINETUNE/eou/diag/lm_fusion_probe.py" \
        --nemo $R/plain-$arm/final.nemo --per-set 200 --alphas 0 --beam 8 --beam-alphas 0 \
        --out $R/plain-$arm/beam_probe.json 2>&1 | grep -E "macro|Traceback|Error" | tail -4
    rm -f $R/plain-$arm/last.ckpt   # 1.38 GB on the 32 GB root; fresh arms are not resume points
done
cd $FT_ROOT && python3 "$FINETUNE/eou/val_table.py" --col E3=runs/plain-it-e2-enc2e-4-s4000/metrics.json:4000 \
    --col E3-8k=runs/plain-it-e3-s8000/metrics.json:8000 --col E3G-8k=runs/plain-it-e3g-s8000/metrics.json:8000 \
    --delta E3-8k-E3 --delta E3G-8k-E3-8k --trajectory runs/plain-it-e3g-s8000/metrics.json
echo "== SCALE-8K-DONE $(date +%T)"
