#!/bin/bash
# Phase E1-short: encoder feature distillation probe vs its EXACT W4 control at the same short horizon.
# Both arms = phase B's W4-C250 run (p1_additive.sh, it-yg + it-yg6 agree<=15, PASSES 1.5 -> W 0.4388, W4 recipe,
# seed 0) at FIX_STEPS=$STEPS, sparse evals, final.nemo = EOF; the KD arm adds ONLY two_stage.py --kd-* (student
# layer S -> fresh Linear(512, 1024) -> teacher layer T of Nemotron 0.6B, 1 - cos, lambda annealed to 0).
# Probes (p1_additive.sh): harness + clean-it, greedy / beam-8 / oracle-8, raw and 0.5 s flush.
#   ARM=calib: --kd-calib on C250 batches (encoder grad norms RNNT vs KD at lambda 1), nothing trained.
#   ARM=ctl|kd tmux new -d -s e1 'FT_ROOT=/root/ft ARM=kd bash finetune/jobs/e1_kd.sh 2>&1 | tee -a /root/ft/logs/e1-kd.log'
set -u
: "${FT_ROOT:?}"; ARM=${ARM:?ctl or kd}; STEPS=${STEPS:-2000}
S=${KD_S:-7}; T=${KD_T:-17}; LAM=${KD_LAMBDA:-}; END=${KD_END:-1.0}
TEACHER=${TEACHER:-$FT_ROOT/models/nemotron-3.5-asr-streaming-0.6b.nemo}
W4="--lr-shape wsd --decay-frac 0.2 --enc-lr 4e-4 --eval-sparse 500,250 --final-eof 1"
[ "$ARM" = kd ] && [ -z "$LAM" ] && { echo "== KD_LAMBDA must be set (pre-registered from ARM=calib)"; exit 2; }
KD=""; [ "$ARM" = kd ] && KD="--kd-teacher $TEACHER --kd-s-layer $S --kd-t-layer $T --kd-weight $LAM --kd-end-frac $END"
if [ "$ARM" = calib ]; then   # unit calibration of the KD loss: no optimizer step, no eval, no WER (two_stage.py --kd-calib)
    . "$(dirname "$0")/gpu_lock.sh"; gpu_lock
    MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it
    MIX="$MF/train_40h.json:0.1964,$MX/train_cv.json:0.1684,$MX/train_vp_t50.json:0.1122,$MX/train_fleurs.json:0.0842"
    MIX="$MIX,$FT_ROOT/mix/it-yg/subsets/yg_agree<=15.json:0.1885,$FT_ROOT/mix/it-yg6/subsets/yg_agree<=15.json:0.2503"   # = C250
    timeout 1200 "$PY" "$FINETUNE/eou/two_stage.py" --tag e1-calib --out $FT_ROOT/runs --mix "$MIX" --freeze-enc 0 --lr 3e-4 --bs 16 \
        --steps $STEPS --max-dur 30 $W4 --seed 0 --val $MF/eval_mls_it.json --val-n 8 \
        --kd-teacher $TEACHER --kd-s-layer $S --kd-t-layer $T --kd-weight 1 --kd-calib ${CALIB_N:-16} 2>&1 \
        | grep -a -E "== KD|calib batch|trainable|Traceback|Error|out of memory"
    exit 0
fi
k=$((STEPS / 1000))k
TAG=it-e1-$ARM-$k EXT="$FT_ROOT/mix/it-yg/subsets/yg_agree<=15.json,$FT_ROOT/mix/it-yg6/subsets/yg_agree<=15.json" \
    PASSES=1.5 FIX_STEPS=$STEPS P1_EXTRA="$W4 $KD" PROBE_POST=0.5 bash "$(dirname "$0")/p1_additive.sh"
echo "== E1-DONE $ARM $(date +%T)"
