#!/bin/bash
# FR-P1 (pre-registered 2026-10-19, release-it-fr-2026-10-19.md section I): ONE plain run = the IT W4 recipe, unchanged,
# on the A2-fr mix. Stock init, whole encoder 4e-4, decoder + joint 3e-4, WSD (5 % warm-up = two_stage.py's steps / 20,
# decay over the last 20 %), 8k steps, bs 16, max-dur 30, seed 0, --abort-wer 1000:85; no CTC / KD / tokenizer / stage 2.
# Mix A2-fr (IT A2+ weights): MLS-fr 40 h (all speakers) 0.35, CV17-fr 0.30, VP-fr teacher t50 0.20, FLEURS fr_fr 0.15.
# In-run evals = the 4 harness-fr sets (diagnostic only: final.nemo is the LAST step, --final-eof 1, so the harness
# never selects the checkpoint; eval cadence as c1_curriculum.sh). Then lm_fusion_probe.py --lang fr: harness-fr
# (200 / set) and clean-fr (all clips, only if its manifest exists), greedy / beam-8 / oracle-8, raw EOF and 0.5 s flush.
# Needs: data_mix.sh fr (MLS-fr + mix/fr) and LANG_=fr FILTER_ONLY=1 a2_vpfilter.sh (mix/fr/train_vp_t50.json).
#   FT_ROOT=/root/ft tmux new -d -s frp1 'FT_ROOT=/root/ft bash finetune/jobs/fr_p1.sh 2>&1 | tee -a /root/ft/logs/fr_p1.log'
# NEMO=<model.nemo> OUT=<dir>: probes only (baselines on the same harness / protocol), no training.
# HARNESS= / CLEAN= override the probe banks (name=path,...), e.g. once the frozen DEV harness-fr has its own manifests.
. "$(dirname "$0")/gpu_lock.sh"
export LANG_=fr
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/fr; R=$FT_ROOT/runs; TAG=${TAG:-fr-p1-w4-8k}
HARNESS=${HARNESS:-"mls=$MF/eval_mls_fr.json,fleurs=$MF/eval_fleurs_fr.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json"}
CLEAN=${CLEAN:-"tedx=$MF/eval_tedx_fr.json"}
if [ -z "${NEMO:-}" ]; then
    MIX="$MF/train_40h_fr.json:0.35,$MX/train_cv.json:0.30,$MX/train_vp_t50.json:0.20,$MX/train_fleurs.json:0.15"
    for p in ${MIX//,/ }; do [ -s "${p%:*}" ] || { echo "== missing ${p%:*}"; exit 2; }; done
    echo "== fr_p1 $TAG $(date +%T)"; echo "   MIX=$MIX"
    MIX="$MIX" VALS="$HARNESS" POL="--freeze-enc 0 --enc-lr 4e-4 --lr 3e-4 --bs 16" STEPS=8000 SMOKE=1 \
        bash "$FINETUNE/jobs/mix_arm.sh" $TAG --max-dur 30 --eval-steps 100,250,1000 --abort-wer 1000:85 \
        --lr-shape wsd --decay-frac 0.2 --eval-sparse 2000,500 --final-eof 1 --seed 0
    rm -f $R/plain-$TAG/last.ckpt
    NEMO=$R/plain-$TAG/final.nemo; OUT=$R/plain-$TAG
fi
[ -s "$NEMO" ] || { echo "== no $NEMO"; exit 3; }
OUT=${OUT:?OUT= with NEMO=}; mkdir -p "$OUT"
gpu_lock
for post in 0 0.5; do
sfx=$([ "$post" = 0 ] || echo "_post$post")
for bank in harness clean; do
    case $bank in
        harness) V=$HARNESS; N=200 ;;
        clean) V=$CLEAN; N=100000 ;;
    esac
    for p in ${V//,/ }; do [ -s "${p#*=}" ] || { echo "== $bank: no ${p#*=}, skipped"; continue 2; }; done
    timeout 3600 "$PY" "$FINETUNE/eou/diag/lm_fusion_probe.py" --lang fr --nemo "$NEMO" --vals "$V" --per-set $N \
        --alphas 0 --beam-alphas 0 --beams 8 --threads 32 --post-pad-s $post --out $OUT/probe_$bank$sfx.json 2>&1 \
        | grep -E "^\{|Traceback|Error" | sed "s/^/post=$post /" | cut -c1-400
done
done
echo "== FR-P1-DONE $TAG $(date +%T)"
