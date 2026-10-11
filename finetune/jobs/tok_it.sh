#!/bin/bash
# S15-15b: B2 (A2 mix, gradual unfreeze to the whole encoder, 4000 steps, same seed and val
# sets) with ONE change: an Italian SPE (same size 1024 + <EOU>/<EOB>, accented targets),
# warm-started from the stock decoder/joint rows (two_stage.py --tokenizer). Compare with
# B2 = 49.08 / 18.22 macro.
# 1. text pool = every A2 training transcript + the Granary-it teacher transcripts (raw,
#    accented; text only, no Granary audio in this arm) -> tokenizer_eou.py (CPU);
# 2. step-0 eval of the warm-started model (how much of the stock decoder survives the remap);
# 3. mix_arm.sh it-tok-b2 with SMOKE=1, the B2 policy and --tokenizer.
# DEACCENT=1: the vocabulary-only arm (SPE trained on de-accented text, de-accented targets as
# B2): separates the SPE vocabulary from accent preservation (tag it-tokna-b2).
#   FT_ROOT=/root/ft tmux new -d -s tok 'FT_ROOT=/root/ft bash finetune/jobs/tok_it.sh 2>&1 | tee -a /root/ft/logs/tok_it.log'
. "$(dirname "$0")/gpu_lock.sh"
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; YG=$FT_ROOT/mix/it-yg; NA=${DEACCENT:+na}; TOK=$FT_ROOT/models/tok${NA}-it-1024; TAG=it-tok${NA}-b2
TOKARGS="--tokenizer $TOK${DEACCENT:+ --accented 0}"
STOCK=$FT_ROOT/models/parakeet_realtime_eou_120m-v1.nemo
A2="$MF/train_40h.json:0.35,$MX/train_cv.json:0.30,$MX/train_vp_t50.json:0.20,$MX/train_fleurs.json:0.15"
POL="--unfreeze-sched 0:4,0.2:8,0.5:all --enc-lr 3e-5 --enc-lr-mid 1.5e-5 --enc-lr-low 5e-6 --lr 3e-4 --bs 16"
VALS="mls=$MF/eval_mls_it.json,fleurs=$MF/eval_fleurs_it.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json"
echo "== tok_it $(date +%T)"
if [ ! -s $TOK/tokenizer.model ]; then
    mkdir -p $TOK
    python3 - $MF/train_40h.json $MX/train_cv.json $MX/train_vp_t50.json $MX/train_fleurs.json $YG/train_yg.teacher.json > $TOK/pool.txt <<'PY'
import json, sys
for p in sys.argv[1:]:
    for l in open(p):
        r = json.loads(l)
        t = r.get("teacher_raw") if "teacher_raw" in r else r.get("text", "")
        if t:
            print(t.replace("\n", " "))
PY
    wc -l $TOK/pool.txt
    timeout 1800 "$PY" "$FINETUNE/eou/tokenizer_eou.py" --stock $STOCK --text $TOK/pool.txt --out $TOK ${DEACCENT:+--deaccent} 2>&1 | grep -E "text:|tokenizer OK|Error|assert" || exit 2
fi
"$PY" "$FINETUNE/eou/diag/tok_coverage.py" --nemo $STOCK --manifest $MF/eval_fleurs_it.json 2>&1 | grep "^ROW utts"
"$PY" -c "
import json, sys, sentencepiece as spm
sys.path.insert(0, '$FINETUNE/eou'); from tokenizer_eou import norm_text
sp = spm.SentencePieceProcessor(model_file='$TOK/tokenizer.model')
t = [norm_text(json.loads(l)['text']) for l in open('$MF/eval_fleurs_it.json')]
w = sum(len(x.split()) for x in t); ids = [sp.encode(x) for x in t]
print(f'ROW it-SPE${NA} utts {len(t)} words {w} tokens/word {sum(map(len, ids)) / w:.2f} unk {sum(i.count(sp.unk_id()) for i in ids)}')"
gpu_lock
for arm in stock "$TAG"; do   # step 0: the stock model, then the same weights after the vocabulary remap
    X=; [ "$arm" = stock ] || X=$TOKARGS
    [ "$arm" = stock ] && [ -s $FT_ROOT/runs/plain-stock-step0/metrics.json ] && continue
    echo "== step0 $arm"
    timeout 1200 "$PY" "$FINETUNE/eou/two_stage.py" --tag $arm-step0 --out $FT_ROOT/runs $X --eval-only 1 \
        --manifest $MF/train_40h.json --n 8 --vals "$VALS" 2>&1 | grep -E "^== tokenizer|warm-start|VAL|Traceback|Error"
done
flock -u 9
MIX="$A2" POL="$POL" SMOKE=1 bash "$FINETUNE/jobs/mix_arm.sh" $TAG $TOKARGS
cd $FT_ROOT && python3 "$FINETUNE/eou/val_table.py" --col B2=runs/plain-it-b2-gradual/metrics.json:4000 \
    --col TOK=runs/plain-$TAG/metrics.json:$(python3 -c "import json; print([r for r in json.load(open('runs/plain-$TAG/metrics.json'))['log'] if r.get('saved')][-1]['step'])") \
    --delta TOK-B2 --trajectory runs/plain-$TAG/metrics.json
echo "== TOK-IT-DONE $(date +%T)"
