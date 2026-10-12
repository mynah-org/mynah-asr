#!/bin/bash
# S15-21 ranking/search audit of one plain checkpoint (2026-10-12 on E3-8k; was /root/res/{nbest,lmr,cleanprobe,decomp}.sh):
# 1. harness + MLS/FLEURS tune split: greedy, malsd beam 4/8/16/32, full N-best dump (lm_fusion_probe.py);
# 2. offline audit: oracle@k, winner-vs-oracle errors, length/insertion rescoring tuned on the tune split (nbest_audit.py);
# 3. neural-LM rescoring (generic + adapted on the A2 transcripts), weights chosen on the tune split (nbest_lm_rescore.py);
# 4. clean-it bank (eval_tedx_it / eval_italic_it): greedy, beam 8/32 + N-best, LM rescoring with the step-3 weights
#    passed as LMFIXED (never re-tuned there), and the teacher (Nemotron, CUDA server) on the same clips;
# 5. predictor/joint autopsy at K=8 on both dumps (nbest_rnnt_decomp.py).
#   FT_ROOT=/root/ft NEMO=/root/ft/runs/plain-it-e3-s8000/final.nemo NAME=e3s8k LMFIXED=generic:1.0:1,adapted:0.5:0 \
#     tmux new -d -s rank 'FT_ROOT=/root/ft bash finetune/jobs/ranking_audit.sh 2>&1 | tee -a /root/ft/logs/ranking_audit.log'
. "$(dirname "$0")/gpu_lock.sh"
: "${NEMO:?NEMO=<final.nemo>}"; NAME=${NAME:-$(basename "$(dirname "$NEMO")")}
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; K=$FT_ROOT/analysis/K; D="$FINETUNE/eou/diag"
mkdir -p $K; cd "$FINETUNE/.." || exit 2
ADAPT="--adapt-text $MF/train_40h.json,$MX/train_cv.json,$MX/train_vp_t50.json,$MX/train_fleurs.json"
EXCL=$MF/eval_mls_it.json,$MF/eval_fleurs_it.json,$MX/eval_cv.json,$MX/eval_vp.json,$MF/eval_tedx_it.json,$MF/eval_italic_it.json
gpu_lock
timeout 3600 "$PY" $D/lm_fusion_probe.py --nemo $NEMO --per-set 200 --alphas 0 --beam-alphas 0 --beams 4,8,16,32 --tune-extra 1 \
    --threads 32 --dump-nbest $K/$NAME --out $K/${NAME}_probe.json 2>&1 | grep -E "^\{|encoded|Traceback|Error" | cut -c1-300
python3 $D/nbest_audit.py --dir $K/$NAME --out $K/${NAME}_audit.json | tee $K/${NAME}_audit.txt
timeout 3600 "$PY" $D/nbest_lm_rescore.py --dir $K/$NAME --ks 8,32 $ADAPT --adapt-exclude $EXCL \
    --out $K/${NAME}_lm_rescore.json 2>&1 | grep -v -E "Warning|warn|Loading weights|dtype"
if [ -s $MF/eval_tedx_it.json ] && [ -s $MF/eval_italic_it.json ]; then
    timeout 3600 "$PY" $D/lm_fusion_probe.py --nemo $NEMO --vals tedx=$MF/eval_tedx_it.json,italic=$MF/eval_italic_it.json \
        --per-set 100000 --alphas 0 --beam-alphas 0 --beams 8,32 --threads 32 --dump-nbest $K/${NAME}_clean \
        --out $K/${NAME}_clean_probe.json 2>&1 | grep -E "^\{|encoded|Traceback|Error" | cut -c1-300
    [ -n "${LMFIXED:-}" ] && timeout 3600 "$PY" $D/nbest_lm_rescore.py --dir $K/${NAME}_clean --ks 8,32 --report tedx,italic --tune none \
        $ADAPT --adapt-exclude $EXCL --fixed "$LMFIXED" --out $K/${NAME}_clean_lm_rescore.json 2>&1 | grep -v -E "Warning|warn|Loading weights|dtype"
    if [ -n "${TEACHER:-}" ]; then   # teacher reference on clean-it (once per bank, not per checkpoint)
        ./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 > $FT_ROOT/logs/teacher_srv_clean.log 2>&1 & SP=$!
        for i in $(seq 120); do curl -sf localhost:8295/v1/health >/dev/null && break; sleep 1; done
        for s in tedx italic; do
            timeout 1800 python3 $D/teacher_ws.py --manifest $MF/eval_${s}_it.json --out $K/teacher_clean_$s.json --lang it --conc 64 | tail -1 | cut -c1-200
            python3 - $K/teacher_clean_$s.json $s <<'PY'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
n = sum(len(r["text"].split()) for r in rs)
e = sum(r["teacher_wer"] * len(r["text"].split()) / 100 for r in rs if r.get("teacher_wer") is not None)
print(f"== teacher {sys.argv[2]} pooled WER (approx, normalised refs) {100 * e / n:.2f} over {len(rs)} clips, "
      f"empty {sum(1 for r in rs if not r.get('teacher'))}")
PY
        done
        kill $SP; wait $SP 2>/dev/null
    fi
fi
DUMPS=$K/$NAME/nbest_b8.jsonl; [ -s $K/${NAME}_clean/nbest_b8.jsonl ] && DUMPS=$DUMPS,$K/${NAME}_clean/nbest_b8.jsonl
timeout 5400 "$PY" $D/nbest_rnnt_decomp.py --nemo $NEMO --dumps $DUMPS --out $K/${NAME}_decomp_b8.json \
    --dump-scores $K/${NAME}_decomp_scores_b8.jsonl 2>&1 | grep -E "^==|^   [a-z]|check|Traceback|Error"
echo "== RANKING-AUDIT-DONE $NAME $(date +%T)"
