#!/bin/bash
# Pack the prepared Italian C250 data (A2 + the two Granary agree<=15 pools + the eval banks) into a data snapshot
# on the private Hub repo, so that the next box restores it (resume_it.sh SNAPSHOT=it-c250) in minutes instead of
# rebuilding ~550 h of audio from the public sources. Run after resume_it.sh, while a GPU job trains (CPU + uplink).
# Only the audio the C250 recipe reads is packed: the full teacher pools are NOT (the subsets are what it trains on).
#   FT_ROOT=/root/ft tmux new -d -s snap 'FT_ROOT=/root/ft bash finetune/jobs/data_snapshot.sh 2>&1 | tee -a /root/ft/logs/snapshot.log'
. "$(dirname "$0")/gpu_lock.sh"
HUB=${HUB:-gabrione/mynah-asr-canary-it-experiments}; NAME=${NAME:-it-c250}
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; YG=$FT_ROOT/mix/it-yg; YG6=$FT_ROOT/mix/it-yg6
M="$MF/train_40h.json,$MX/train_cv.json,$MX/train_vp_t50.json,$MX/train_fleurs.json"
M="$M,$YG/subsets/yg_agree<=15.json,$YG6/subsets/yg_agree<=15.json"
M="$M,$MF/eval_mls_it.json,$MF/eval_fleurs_it.json,$MX/eval_cv.json,$MX/eval_vp.json,$MF/eval_tedx_it.json,$MF/eval_italic_it.json"
echo "== data_snapshot $NAME $(date +%T)"
nice -n 10 timeout 14400 "$PY" "$FINETUNE/common/data_snapshot.py" pack --repo $HUB --name $NAME --token-file /root/.hf_token \
    --manifests "$M" --meta "$MF/*.json,$MX/*.json,$YG/*.json,$YG/subsets/*.json,$YG6/*.json,$YG6/subsets/*.json" \
    --shard-gb ${SHARD_GB:-1.5} --workers ${WORKERS:-12} --tmp /root
echo "== DATA-SNAPSHOT-DONE rc=$? $(date +%T)"
