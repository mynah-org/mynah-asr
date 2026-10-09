#!/bin/bash
# ft3: Italian FT with multilingual replay. ft2 showed long Italian-only
# training wiping out English (FLEURS-en WER 6.9 -> 40 after 3,120 steps, arm
# A; +20.9 for B4). Here ~REPLAY_RATIO of the training audio is FLEURS train in
# en/de/es/fr (the base model's languages), and the run scores en/de/es/fr
# forgetting before/after next to the Italian eval. last.ckpt is kept
# (SAVE_CKPT=1) so a later stage can resume the exact optimizer state.
# Data step (network + CPU, ~1 GB of wav) runs OUTSIDE the GPU lock.
# Env: SUBSET (20h), EPOCHS (13: IT epochs; steps grow by 1/(1-ratio)),
#      ARMS (B), REPLAY_RATIO (0.2), REPLAY_H (2 h per language),
#      TAG_SUFFIX (-e<EPOCHS>-rp<ratio%>), RATE_USD_H, T_TRAIN (21600 s).
#   tmux new -d -s ft3 'bash /root/ft3.sh 2>&1 | tee /root/ft/logs/ft3.log'
set -u
K=/root/ft-kit
FT=${FT_ROOT:-/root/ft}
PY=${VENV:-/root/nemo-venv}/bin/python
SUBSET=${SUBSET:-20h}; EPOCHS=${EPOCHS:-13}; ARMS=${ARMS:-B}
REPLAY_RATIO=${REPLAY_RATIO:-0.2}; REPLAY_H=${REPLAY_H:-2}
pct=$(awk "BEGIN{printf \"%d\", ${REPLAY_RATIO} * 100 + 0.5}")
TAG_SUFFIX=${TAG_SUFFIX:--e$EPOCHS-rp$pct}
export REPLAY_H FT_ROOT=$FT

marker=$FT/done/replay-${REPLAY_H}h
if [ ! -e "$marker" ]; then
    echo "== ft3 replay data ${REPLAY_H} h/lang $(date +%T); free $(df -h --output=avail "$FT" | tail -1)"
    timeout "${T_DATA:-7200}" "$PY" "$K/data_replay.py" --hours-per-lang "$REPLAY_H" \
        || { echo "== FT3-FAILED data rc=$? $(date +%T)"; exit 1; }
    date '+%F %T' >"$marker"
fi

exec 9>/root/gpu.lock; flock 9
echo "== ft3 lock held $(date +%T) subset=$SUBSET epochs=$EPOCHS arms=$ARMS replay=$REPLAY_RATIO tag+=$TAG_SUFFIX"
SUBSET=$SUBSET EPOCHS=$EPOCHS ARMS=$ARMS REPLAY_RATIO=$REPLAY_RATIO TAG_SUFFIX=$TAG_SUFFIX SAVE_CKPT=1 \
    T_TRAIN=${T_TRAIN:-21600} RATE_USD_H=${RATE_USD_H:-0} bash $K/train_canary_it.sh
echo "== FT3-DONE rc=$? $(date +%T)"
