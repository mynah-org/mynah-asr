#!/bin/bash
# After B1/B2 (plain Italian, 5 h): pick the winner by best val WER, run it on
# 40 h, then stage 2 (EOU) from the 40 h best. One GPU job at a time (lock).
#   tmux new -d -s chain 'bash /root/chain_it.sh 2>&1 | tee /root/ft/logs/chain.log'
set -u
K=/root/eou-kit; PY=/root/nemo-venv/bin/python; R=/root/ft/runs; L=/root/ft/logs
VAL="--val /root/ft/manifests/eval_mls_it.json --val-n 200"
until grep -q "plain-b2-5h-top4 done" $L/plain-b2.log 2>/dev/null; do sleep 20; done
best() { $PY -c "import json;d=json.load(open('$R/plain-$1/metrics.json'));print(min(x['val_wer'] for x in d['log'] if 'val_wer' in x))"; }
w1=$(best b1-5h-frz); w2=$(best b2-5h-top4); echo "== chain b1 best val $w1, b2 best val $w2 $(date +%T)"
if $PY -c "import sys; sys.exit(0 if float('$w2') < float('$w1') else 1)"; then
    POL="--freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5"; W=b2
else
    POL="--freeze-enc 1"; W=b1
fi
echo "== chain winner $W: $POL"
exec 9>/root/gpu.lock; flock 9
cd $K
$PY plain_it2.py --manifest /root/ft/manifests/train_40h.json --steps 4000 --bs 16 --lr 3e-4 $POL --eval-every 500 $VAL \
    --tag p40-$W 2>&1 | grep --line-buffered -v -E "NeMo W|warn" > $L/plain-p40.log
echo "== chain p40 done $(date +%T)"; grep VAL $L/plain-p40.log | tail -3
$PY plain_it2.py --init $R/plain-p40-$W/final.nemo --manifest /root/ft/manifests/train_40h.json --steps 1500 --bs 16 --lr 1e-4 $POL \
    --eou-append 1 --pad-prob 0.5 --pad-min 1 --pad-max 3 --fastemit 0 --eval-every 250 $VAL \
    --tag s2-eou-p40-$W 2>&1 | grep --line-buffered -v -E "NeMo W|warn" > $L/plain-s2.log
echo "== chain s2 done $(date +%T)"; grep VAL $L/plain-s2.log | tail -3
echo "== CHAIN-DONE $(date +%T)"
