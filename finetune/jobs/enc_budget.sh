#!/bin/bash
# 2026-10-11 synthesis, the three discriminating arms (A2 mix, same seed/val sets as B2):
#   E1a it-e1a-decjoint  from B2, encoder FROZEN, decoder+joint 1e-4, 1000 steps
#   E1b it-e1b-all1e4    from B2, whole model 1e-4 (encoder incl. pre-encode), 1000 steps
#        E1a flat + E1b down => encoder-limited; E1a down ~ E1b => output side under-trained;
#        both flat => data/targets. (B2's lower encoder moved 0.01-0.04 % in 4000 steps.)
#   E2  it-e2-full1e4    from STOCK, whole encoder 1e-4 from step 0, decoder+joint 3e-4, 4000
#        steps: the recipe candidate, batch-paired with B2 (seeded sampler = same batches);
#        evals at 100/250 + every 500; collapse gate only (macro > 85 at 1000; stock 92.4, B2 56.9): a different but healthy curve must not be killed.
# After each arm: greedy vs beam-8 vs 8-best oracle on CPU (diag/lm_fusion_probe.py, alpha 0),
# so acoustic headroom (oracle) and the free beam gain are tracked per checkpoint.
#   FT_ROOT=/root/ft tmux new -d -s enc 'FT_ROOT=/root/ft bash finetune/jobs/enc_budget.sh 2>&1 | tee -a /root/ft/logs/enc_budget.log'
. "$(dirname "$0")/gpu_lock.sh"
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; R=$FT_ROOT/runs; B2=$R/plain-it-b2-gradual/final.nemo
A2="$MF/train_40h.json:0.35,$MX/train_cv.json:0.30,$MX/train_vp_t50.json:0.20,$MX/train_fleurs.json:0.15"
LM=${LM:-$FT_ROOT/analysis/D/lm4_trainmix.arpa}
probe() {   # probe <tag>: CPU, does not take the GPU lock
    [ -s $R/plain-$1/final.nemo ] || return 0
    CUDA_VISIBLE_DEVICES= timeout 3600 "$PY" "$FINETUNE/eou/diag/lm_fusion_probe.py" --nemo $R/plain-$1/final.nemo --lm $LM \
        --per-set 200 --alphas 0 --beam 8 --beam-alphas 0 --out $R/plain-$1/beam_probe.json 2>&1 | grep -E "macro|Traceback|Error" | tail -4
}
for arm in "it-e1a-decjoint|--freeze-enc 1 --lr 1e-4" "it-e1b-all1e4|--freeze-enc 0 --enc-lr 1e-4 --lr 1e-4"; do
    tag=${arm%%|*}; pol=${arm#*|}
    MIX="$A2" POL="$pol --bs 16" STEPS=1000 bash "$FINETUNE/jobs/mix_arm.sh" $tag --init $B2 --eval-every 250
    probe $tag
done
MIX="$A2" POL="--freeze-enc 0 --enc-lr 1e-4 --lr 3e-4 --bs 16" SMOKE=1 bash "$FINETUNE/jobs/mix_arm.sh" it-e2-full1e4 \
    --eval-steps 100,250 --abort-wer 1000:85
probe it-e2-full1e4
cd $FT_ROOT && python3 "$FINETUNE/eou/val_table.py" --col B2=runs/plain-it-b2-gradual/metrics.json:4000 \
    --col E1a=runs/plain-it-e1a-decjoint/metrics.json:1000 --col E1b=runs/plain-it-e1b-all1e4/metrics.json:1000 \
    --col E2=runs/plain-it-e2-full1e4/metrics.json:4000 --delta E1a-B2 --delta E1b-B2 --delta E2-B2 \
    --trajectory runs/plain-it-e2-full1e4/metrics.json
echo "== ENC-BUDGET-DONE $(date +%T)"
