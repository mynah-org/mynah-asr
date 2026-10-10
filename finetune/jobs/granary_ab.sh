#!/bin/bash
# C1 / C2: does two-teacher AGREEMENT filtering beat a random hours-matched draw of the same
# Granary-it pool? Both = B2 (gradual unfreeze to the whole encoder, 4000 steps, same seed and
# val sets) on a 50 % A2 mix (internal proportions unchanged) / 50 % Granary mix, labels = the
# teacher's text, --max-dur 30 (Granary clips are long; every A2 source is <= 20 s anyway).
#   C1 = hygiene + agreement <= 15 %  (yg_agree<=15, 98.1 h)
#   C2 = hygiene only, random, hours-matched (yg_rand_eq<=15, 98.1 h)
# A 40-step smoke on C1's real mix gates the memory peak at 30 s clips first.
#   FT_ROOT=/root/ft tmux new -d -s gab 'FT_ROOT=/root/ft bash finetune/jobs/granary_ab.sh 2>&1 | tee -a /root/ft/logs/granary_ab.log'
set -u
F=$(cd "$(dirname "$0")/.." && pwd); FT=${FT_ROOT:?}; MF=$FT/manifests; MX=$FT/mix/it; S=$FT/mix/it-yg/subsets
POL="--unfreeze-sched 0:4,0.2:8,0.5:all --enc-lr 3e-5 --enc-lr-mid 1.5e-5 --enc-lr-low 5e-6 --lr 3e-4 --bs 16"
A2="$MF/train_40h.json:0.175,$MX/train_cv.json:0.15,$MX/train_vp_t50.json:0.10,$MX/train_fleurs.json:0.075"
MIX="$A2,$S/yg_agree<=15.json:0.5" POL="$POL" SMOKE=1 bash $F/jobs/mix_arm.sh it-c1-yg15 --max-dur 30 || exit 3
MIX="$A2,$S/yg_rand_eq<=15.json:0.5" POL="$POL" bash $F/jobs/mix_arm.sh it-c2-ygrand --max-dur 30
cd $FT && python3 $F/eou/val_table.py --col B2=runs/plain-it-b2-gradual/metrics.json:4000 \
  --col C1=runs/plain-it-c1-yg15/metrics.json:$(python3 -c "import json; print([r for r in json.load(open('runs/plain-it-c1-yg15/metrics.json'))['log'] if r.get('saved')][-1]['step'])") \
  --col C2=runs/plain-it-c2-ygrand/metrics.json:$(python3 -c "import json; print([r for r in json.load(open('runs/plain-it-c2-ygrand/metrics.json'))['log'] if r.get('saved')][-1]['step'])") \
  --delta C1-B2 --delta C2-B2 --delta C1-C2
echo "== GRANARY-AB-DONE $(date +%T)"
