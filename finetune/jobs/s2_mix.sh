#!/bin/bash
# Stage 2 (<EOU>) on a multi-domain plain checkpoint with the 2026-10-09 stage-2 recipe UNCHANGED
# (text + <EOU> targets, 50 % of samples with 1-3 s trailing quiet, lr 1e-4, top-4 encoder at
# 3e-5, 1500 steps, FastEmit 0, best = min WER among evals with EOU >= 80 % and <= 10 % empty)
# (S2POL overrides the encoder policy, e.g. the whole encoder: S2POL='--unfreeze-sched 0:all ...';
#  S2EXTRA adds flags, e.g. S2EXTRA='--pre-pad-prob 0.9 --pre-pad-max 4' for leading quiet)
# -- only the data is the plain stage's own mix (A2: MLS + CV + teacher-filtered VP + FLEURS),
# scored on the same four val sets; then the Mynah evaluation of 2026-10-09 (jobs/m1.sh).
# S2MIX overrides the mix (e.g. the parent's own C250 mix); PROBE=1 adds the harness + clean greedy / beam-8 /
# oracle-8 probes (raw and 0.5 s flush) of the stage-2 model AND of its parent on the same box (exact delta);
# M1=0 skips the Mynah evaluation (run later on the frozen candidate).
#   FT_ROOT=/root/ft INIT=/root/ft/runs/plain-it-b2-gradual/final.nemo TAG=it-s2-b2 \
#     tmux new -d -s s2 'FT_ROOT=/root/ft INIT=... TAG=... bash finetune/jobs/s2_mix.sh 2>&1 | tee -a /root/ft/logs/s2_mix.log'
. "$(dirname "$0")/gpu_lock.sh"
: "${INIT:?}" "${TAG:?}"
# LANG_=fr (FR stage 2, 2026-10-20): mix/fr + harness-fr, --lang fr to two_stage and the probes, no clean bank
# (the FR OOD bank is a one-shot retention check, never probed here); LANG_ unset = it, unchanged.
LG=${LANG_:-it}; LF=(); [ "$LG" = it ] || LF=(--lang "$LG")
L=$FT_ROOT/logs; MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/$LG; R=$FT_ROOT/runs
if [ "$LG" = it ]; then
    MIX=${S2MIX:-"$MF/train_40h.json:0.35,$MX/train_cv.json:0.30,${VP:-$MX/train_vp_t50.json}:0.20,$MX/train_fleurs.json:0.15"}
    VALS="mls=$MF/eval_mls_it.json,fleurs=$MF/eval_fleurs_it.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json"
else
    MIX=${S2MIX:?S2MIX= (the mix of the parent) is required for LANG_=$LG}
    VALS="mls=$MF/eval_mls_$LG.json,fleurs=$MF/eval_fleurs_$LG.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json"
fi
gpu_lock
echo "== $TAG stage 2 from $INIT $(date +%T)"; ln -sf $L/$TAG.log $L/train_cur.log
timeout 3600 "$PY" "$FINETUNE/eou/two_stage.py" --init "$INIT" --tag "$TAG" --out $R --mix "$MIX" \
    ${S2POL:---freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5} --lr 1e-4 --bs 16 --steps 1500 \
    --eou-append 1 --pad-prob 0.5 --pad-min 1 --pad-max 3 --fastemit 0 --eval-every 250 --vals "$VALS" "${LF[@]}" ${S2EXTRA:-} \
    2>&1 | grep --line-buffered -v -E "NeMo W|warn|Warning" > $L/$TAG.log
echo "   rc=${PIPESTATUS[0]}"; grep -E "VAL|done|Traceback" $L/$TAG.log | tail -2 | cut -c1-600
if [ "${PROBE:-0}" = 1 ]; then
    for nemo in $R/plain-$TAG/final.nemo "$INIT"; do
        d=$R/plain-$TAG/probe_$([ "$nemo" = "$INIT" ] && echo parent || echo s2); mkdir -p $d
        for post in 0 0.5; do
            sfx=$([ "$post" = 0 ] || echo "_post$post")
            for bank in harness $([ "$LG" = it ] && echo clean); do
                case $bank in
                    harness) V="$VALS"; N=200 ;;
                    clean) V="tedx=$MF/eval_tedx_it.json,italic=$MF/eval_italic_it.json"; N=100000 ;;
                esac
                timeout 3600 "$PY" "$FINETUNE/eou/diag/lm_fusion_probe.py" "${LF[@]}" --nemo $nemo --vals "$V" --per-set $N --alphas 0 \
                    --beam-alphas 0 --beams 8 --threads 32 --post-pad-s $post --out $d/probe_$bank$sfx.json 2>&1 \
                    | grep -E "^\{|Traceback|Error" | sed "s|^|$(basename $d) post=$post |" | cut -c1-400
            done
        done
    done
fi
flock -u 9
[ "$LG" = it ] && python3 "$FINETUNE/eou/val_table.py" --col "S2-old=$L/base-s2.log" \
    --col "$TAG@best=$R/plain-$TAG/metrics.json:$(python3 -c "import json; print([r for r in json.load(open('$R/plain-$TAG/metrics.json'))['log'] if r.get('saved')][-1]['step'])")" \
    --delta "$TAG@best-S2-old" --trajectory $R/plain-$TAG/metrics.json
[ "${M1:-1}" = 1 ] && echo "== Mynah evaluation (jobs/m1.sh) $(date +%T)" && NEMO=$R/plain-$TAG/final.nemo bash "$FINETUNE/jobs/m1.sh" 2>&1 | tee $L/m1-$TAG.log | grep -E "^== |MYNAH|rc=|speech end|missed|premature|gap|^ +1 |SERVER-CONFIG|PASS|FAIL" | cut -c1-260
echo "== S2-MIX-DONE $TAG $(date +%T)"
