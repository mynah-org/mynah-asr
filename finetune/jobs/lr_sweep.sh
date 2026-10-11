#!/bin/bash
# Encoder-LR sweep of the E2 recipe (stock -> whole encoder from step 0, decoder+joint 3e-4,
# A2 mix, seed 0): every arm keeps E2's schedule (cosine over 4000 steps) and therefore E2's
# exact batches, and stops at STOP (default 1000) with evals at 100/250/500/1000, so each arm's
# early curve is directly comparable with E2's own first 1000 steps (E2 itself is the 1e-4 arm).
# Only the encoder LR changes. A CPU greedy/beam-8/oracle probe follows each arm. EVALS= adds eval
# steps (e.g. denser around the LR peak to see instabilities; training is unchanged).
#   FT_ROOT=/root/ft LRS="3e-5 2e-4" tmux new -d -s lrs 'FT_ROOT=/root/ft bash finetune/jobs/lr_sweep.sh 2>&1 | tee -a /root/ft/logs/lr_sweep.log'
. "$(dirname "$0")/gpu_lock.sh"
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; R=$FT_ROOT/runs; STOP=${STOP:-1000}
A2="$MF/train_40h.json:0.35,$MX/train_cv.json:0.30,$MX/train_vp_t50.json:0.20,$MX/train_fleurs.json:0.15"
LM=${LM:-$FT_ROOT/analysis/D/lm4_trainmix.arpa}
for lr in ${LRS:-3e-5 2e-4}; do
    tag=it-e2-enc$lr-s$STOP
    MIX="$A2" POL="--freeze-enc 0 --enc-lr $lr --lr 3e-4 --bs 16" bash "$FINETUNE/jobs/mix_arm.sh" $tag \
        --stop-at $STOP --eval-steps ${EVALS:-100,250} --abort-wer 1000:85
    [ -s $R/plain-$tag/final.nemo ] && CUDA_VISIBLE_DEVICES= timeout 3600 "$PY" "$FINETUNE/eou/diag/lm_fusion_probe.py" \
        --nemo $R/plain-$tag/final.nemo --lm $LM --per-set 200 --alphas 0 --beam 8 --beam-alphas 0 \
        --out $R/plain-$tag/beam_probe.json 2>&1 | grep -E "Traceback|Error" | tail -2
    [ -n "${KEEP_CKPT:-}" ] || rm -f $R/plain-$tag/last.ckpt   # optimizer state: only for resume points
done
cd $FT_ROOT && python3 - $R $STOP ${LRS:-3e-5 2e-4} <<'PY'
import json, sys
R, stop, lrs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
arms = [("1e-4 (E2)", f"{R}/plain-it-e2-full1e4/metrics.json")] + [(lr, f"{R}/plain-it-e2-enc{lr}-s{stop}/metrics.json") for lr in lrs]
print("| enc lr | " + " | ".join(f"@{s}" for s in (100, 250, 500, 1000)) + " |")
print("|---" * 5 + "|")
for name, p in arms:
    try:
        log = {r["step"]: r for r in json.load(open(p))["log"]}
    except Exception as e:
        print(f"| {name} | {e} |"); continue
    print(f"| {name} | " + " | ".join(f"{log[s]['val_wer']:.2f} / {log[s]['val_cer']:.2f}" if s in log else "-" for s in (100, 250, 500, 1000)) + " |")
PY
echo "== LR-SWEEP-DONE $(date +%T)"
