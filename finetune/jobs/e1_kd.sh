#!/bin/bash
# Phase E1-short: encoder feature distillation probe vs its EXACT W4 control at the same short horizon.
# Both arms = phase B's W4-C250 run (p1_additive.sh, it-yg + it-yg6 agree<=15, PASSES 1.5 -> W 0.4388, W4 recipe,
# seed 0) at FIX_STEPS=$STEPS, sparse evals, final.nemo = EOF; the KD arm adds ONLY two_stage.py --kd-* (student
# layer S -> fresh Linear(512, 1024) -> teacher layer T of Nemotron 0.6B, 1 - cos, lambda annealed to 0).
# Probes (p1_additive.sh): harness + clean-it, greedy / beam-8 / oracle-8, raw and 0.5 s flush.
#   ARM=ctl|kd tmux new -d -s e1 'FT_ROOT=/root/ft ARM=kd bash finetune/jobs/e1_kd.sh 2>&1 | tee -a /root/ft/logs/e1-kd.log'
set -u
: "${FT_ROOT:?}"; ARM=${ARM:?ctl or kd}; STEPS=${STEPS:-2000}
S=${KD_S:-7}; T=${KD_T:-17}; LAM=${KD_LAMBDA:-25}; END=${KD_END:-0.8}
TEACHER=${TEACHER:-$FT_ROOT/models/nemotron-3.5-asr-streaming-0.6b.nemo}
W4="--lr-shape wsd --decay-frac 0.2 --enc-lr 4e-4 --eval-sparse 500,250 --final-eof 1"
KD=""; [ "$ARM" = kd ] && KD="--kd-teacher $TEACHER --kd-s-layer $S --kd-t-layer $T --kd-weight $LAM --kd-end-frac $END"
k=$((STEPS / 1000))k
TAG=it-e1-$ARM-$k EXT="$FT_ROOT/mix/it-yg/subsets/yg_agree<=15.json,$FT_ROOT/mix/it-yg6/subsets/yg_agree<=15.json" \
    PASSES=1.5 FIX_STEPS=$STEPS P1_EXTRA="$W4 $KD" PROBE_POST=0.5 bash "$(dirname "$0")/p1_additive.sh"
echo "== E1-DONE $ARM $(date +%T)"
