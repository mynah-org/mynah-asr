#!/bin/bash
# Is the stage-2 VoxPopuli regression (NeMo offline eval: no reset after <EOU>) also there in
# Mynah streaming (reset after every model <EOU>, as NVIDIA's voice agent does)? CPU only.
# Packs: v2 (stage 2 top-4 on B2) and, if present, the full-encoder stage 2; sets: VP and CV val.
#   FT_ROOT=/root/ft bash finetune/jobs/vp_mynah.sh 2>&1 | tee -a /root/ft/logs/vp_mynah.log
. "$(dirname "$0")/gpu_lock.sh"
R=$FT_ROOT/runs; MX=$FT_ROOT/mix/it
cd "$FINETUNE/.." || exit 2
for p in "v2=$R/plain-it-s2-b2/final.nemo" "v2full=$R/plain-it-s2-b2-full/final.nemo"; do
    n=${p%%=*}; nemo=${p#*=}; pack=models/parakeet-realtime-eou-120m-it-$n
    [ -s "$nemo" ] || { echo "   (no $nemo)"; continue; }
    [ -d $pack ] || MYNAH=$PWD PY=$PY CLIPS="$(ls samples/eval-bank-it/it/*.wav | head -2)" \
        timeout 1500 bash /root/eou-kit/export_to_mynah.sh "$nemo" $PWD/$pack > $FT_ROOT/logs/export-$n.log 2>&1
    for s in vp cv; do timeout 1800 python3 "$FINETUNE/eou/diag/mynah_wer.py" --pack $pack --manifest $MX/eval_$s.json --jobs 14; done
done
echo "== VP-MYNAH-DONE $(date +%T)"
