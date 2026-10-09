#!/bin/bash
# s2b: stage 2 again from the same plain-IT 40 h best, ONE variable changed:
# FastEmit 0.03 (the stock checkpoint's value) instead of 0, to pull the <EOU>
# decision earlier. Then a short Mynah eval: stream WER on 100 FLEURS-it clips and
# eou_metrics (40 clips, gap 1 s), same tools as m1.
#   tmux new -d -s s2b 'bash /root/s2b.sh 2>&1 | tee /root/ft/logs/s2b.log'
set -u
PY=/root/nemo-venv/bin/python; R=/root/ft/runs
exec 9>/root/gpu.lock; flock 9
cd /root/eou-kit
$PY plain_it2.py --init $R/plain-p40-b2pol/final.nemo --manifest /root/ft/manifests/train_40h.json --steps 1500 --bs 16 --lr 1e-4 \
    --freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5 --eou-append 1 --pad-prob 0.5 --pad-min 1 --pad-max 3 --fastemit 0.03 \
    --eval-every 250 --val /root/ft/manifests/eval_mls_it.json --val-n 200 --tag s2b-eou-fe03 2>&1 \
    | grep --line-buffered -v -E "NeMo W|warn" > /root/ft/logs/plain-s2b.log
grep VAL /root/ft/logs/plain-s2b.log | tail -2
flock -u 9
P2=/root/mynah-asr/models/parakeet-realtime-eou-120m-it-fe03
MYNAH=/root/mynah-asr PY=$PY timeout 1500 bash /root/eou-kit/export_to_mynah.sh $R/plain-s2b-eou-fe03/final.nemo $P2 2>&1 | tail -2
cd /root/mynah-eou
timeout 1500 python3 tools/eval/eou_metrics.py -m models/parakeet-realtime-eou-120m-it-fe03 --manifest samples/eval-bank-it/manifest.json \
    --root samples/eval-bank-it --langs it --mode both --gaps 1 --vad models/silero-vad --limit 40 --jobs 14 \
    --work-dir /root/ft/m2/eouw --json /root/ft/m2/eou-metrics.json > /root/ft/m2-eou.txt 2>&1; echo "rc=$?"; tail -10 /root/ft/m2-eou.txt
echo "== S2B-DONE $(date +%T)"
