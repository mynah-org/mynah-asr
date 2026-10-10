#!/bin/bash
# The Italian ceiling on OUR harness: Nemotron 0.6B (CUDA server, the teacher) on exactly the
# val clips two_stage.py scores, next to the students P40 / A0 / A2, per utterance; report =
# WER / CER / empty per set + macro, the A2 - Nemotron gap, paired win/loss (paired_eval.py).
#   FT_ROOT=/root/ft tmux new -d -s ceil 'FT_ROOT=/root/ft bash finetune/jobs/it_ceiling.sh 2>&1 | tee -a /root/ft/logs/ceiling.log'
. "$(dirname "$0")/gpu_lock.sh"
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; R=$FT_ROOT/runs; O=$FT_ROOT/ceiling; L=$FT_ROOT/logs
PE="$PY $FINETUNE/eou/diag/paired_eval.py"
cd "$FINETUNE/.." || exit 2; ulimit -n 65536
$PE subsample --vals "mls=$MF/eval_mls_it.json,fleurs=$MF/eval_fleurs_it.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json" --out $O
gpu_lock
for m in "P40=$FT_ROOT/base/plain-p40-b2pol/final.nemo" "A0=$R/plain-it-a0-mix/final.nemo" "A2=$R/plain-it-a2-vpt50/final.nemo"; do
    timeout 1200 $PE decode --name ${m%%=*} --nemo ${m#*=} --out $O 2>&1 | grep -E "^   |Error|Traceback"
done
./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 > $L/teacher_srv2.log 2>&1 & SP=$!
for i in $(seq 120); do curl -sf localhost:8295/v1/health >/dev/null && break; sleep 1; done
for s in mls fleurs cv vp; do
    timeout 900 python3 finetune/eou/diag/teacher_ws.py --manifest $O/sub_$s.json --out $O/teacher_$s.json --conc 64 | tail -1 | cut -c1-120
done
kill $SP; wait $SP 2>/dev/null; flock -u 9
$PE report --out $O --models P40,A0,A2 --teacher Nemotron --pair A2
echo "== CEILING-DONE $(date +%T)"
