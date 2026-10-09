#!/bin/bash
# After B1/B2 (plain Italian, 5 h): pick the winner by best val WER, run it on
# 40 h, then stage 2 (EOU) from the 40 h best. One GPU job at a time (lock).
#   tmux new -d -s chain 'bash /root/chain_it.sh 2>&1 | tee /root/ft/logs/chain.log'
set -u
K=/root/eou-kit; PY=/root/nemo-venv/bin/python; R=/root/ft/runs; L=/root/ft/logs
VAL="--val /root/ft/manifests/eval_mls_it.json --val-n 200"
# b1/b2 died in NeMo's batched greedy decoding during an eval (fixed in
# plain_it.py); b1's best (MLS-it val WER 59.12 / CER 26.57, frozen encoder) is
# archived. The 40 h run takes the top-4 policy (the better arm on Canary).
POL="--freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5"; W=b2pol
echo "== chain policy $W: $POL"
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
