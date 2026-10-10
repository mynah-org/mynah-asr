#!/bin/bash
# Experiment A of 2026-10-10: does DATA DIVERSITY close the MLS -> FLEURS gap of the
# EOU 120M specialist? Training algorithm = the validated 2026-10-09 stage 1, unchanged
# (top-4 encoder blocks at 3e-5, decoder/joint 3e-4, 4000 steps x bs 16, FastEmit 0,
# stock tokenizer, no padding); ONLY the training data changes:
#   A0 = MLS 40 h + Common Voice + VoxPopuli + FLEURS train, domain-balanced sampler
# (A1 = A0 + random gain is decided only after A0 is compared with yesterday's model.)
# Every candidate is scored on the SAME four val sets (200 clips each, accent-insensitive):
# MLS-it test (= yesterday's val), FLEURS-it test, CV-it test, VoxPopuli-it test; the
# checkpoint is selected on the MACRO mean, per-domain numbers are all logged.
# Gates before the real run: data stats (mix_stats.py); regression A/B of today's
# two_stage.py with every new option OFF against the 2026-10-09 file (b33f67e), same
# manifest/seed, 50 steps: the loss trace must match; yesterday's checkpoints on the four
# sets (--eval-only; P40 on MLS must reproduce 41.7); a 100-step mix smoke + one val pass.
#   FT_ROOT=/root/ft tmux new -d -s mixa 'FT_ROOT=/root/ft bash finetune/jobs/mix_a.sh 2>&1 | tee -a /root/ft/logs/mix_a.log'
. "$(dirname "$0")/gpu_lock.sh"
L=$FT_ROOT/logs; MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; R=$FT_ROOT/runs; B=$FT_ROOT/base
TS="$FINETUNE/eou/two_stage.py"
until [ -s $MF/train_40h.json ] && [ -s $MX/train_fleurs.json ] && [ -s $MX/hours.json ]; do sleep 30; done
echo "== data ready $(date +%T)"
MIX="$MF/train_40h.json:0.35,$MX/train_cv.json:0.30,$MX/train_vp.json:0.20,$MX/train_fleurs.json:0.15"
VALS="mls=$MF/eval_mls_it.json,fleurs=$MF/eval_fleurs_it.json,cv=$MX/eval_cv.json,vp=$MX/eval_vp.json"
POL="--freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5 --lr 3e-4 --bs 16"
run() {  # run <tag> <log> args...: one GPU step, time-boxed, warnings filtered
    local tag=$1 log=$2; shift 2
    echo "== $tag $(date +%T)"; ln -sf "$log" $L/train_cur.log
    timeout 7200 "$PY" "$TS" --tag "$tag" --out $R "$@" 2>&1 | grep --line-buffered -v -E "NeMo W|warn|Warning" > "$log"
    echo "   rc=${PIPESTATUS[0]}"; grep -E "VAL|done|Traceback|Error" "$log" | tail -3 | cut -c1-400
}
echo "== gate 1: data stats $(date +%T)"
timeout 1800 "$PY" "$FINETUNE/eou/mix_stats.py" --mix "$MIX" --evals "$VALS" --json $L/mix_stats_it.json 2>&1 | grep -v -i warn
gpu_lock
echo "== gate 2: regression A/B, new options OFF vs the 2026-10-09 file $(date +%T)"
git -C "$FINETUNE/.." show b33f67e:finetune/eou/two_stage.py > $FT_ROOT/two_stage_ref.py
REG="--manifest $MF/train_40h.json --n 400 $POL --steps 50 --eval-every 50 --val $MF/eval_mls_it.json --val-n 40 --seed 0"
for v in ref new; do
    f=$TS; [ $v = ref ] && f=$FT_ROOT/two_stage_ref.py
    timeout 1800 "$PY" "$f" --tag reg-$v --out $R $REG 2>&1 | grep -E "step (25|50)/|EVAL|VAL|trainable|Traceback" > $L/reg-$v.log
    rm -f $R/plain-reg-$v/*.nemo $R/plain-reg-$v/*.ckpt
done
paste -d'\n' $L/reg-ref.log $L/reg-new.log | cut -c1-220
# trainable params must be identical; losses within 1 % (cuDNN / atomics are not bit-exact)
"$PY" - $L/reg-ref.log $L/reg-new.log <<'PYEOF' || { echo "== REGRESSION: ref vs new differ, stop"; exit 4; }
import re, sys
def grab(f):
    t = open(f).read()
    return re.findall(r"trainable ([0-9.]+)", t), [float(x) for x in re.findall(r"step \d+/50 loss ([0-9.]+)", t)]
(tr, lr), (tn, ln) = grab(sys.argv[1]), grab(sys.argv[2])
ok = tr == tn and len(lr) == len(ln) == 2 and all(abs(a - b) <= 0.01 * max(a, b) for a, b in zip(lr, ln))
print(f"   regression A/B: trainable {tr} vs {tn}, losses {lr} vs {ln} -> {'OK' if ok else 'DIFFER'}")
sys.exit(0 if ok else 1)
PYEOF
echo "== gate 3: yesterday's checkpoints on the four val sets"
run base-p40 $L/base-p40.log --init $B/plain-p40-b2pol/final.nemo --manifest $MF/train_40h.json --n 8 --eval-only 1 --vals "$VALS"
run base-s2 $L/base-s2.log --init $B/plain-s2-eou-p40-b2pol/final.nemo --manifest $MF/train_40h.json --n 8 --eval-only 1 \
    --eou-append 1 --vals "$VALS"
echo "== gate 4: 100-step smoke on the mix $(date +%T)"
run smoke-mix $L/smoke-mix.log --mix "$MIX" $POL --steps 100 --eval-every 100 --vals "$VALS" --val-n 50
grep -q "plain-smoke-mix done" $L/smoke-mix.log || { echo "== SMOKE FAILED, stop"; exit 3; }
rm -f $R/plain-smoke-mix/*.nemo $R/plain-smoke-mix/*.ckpt
run it-a0-mix $L/it-a0-mix.log --mix "$MIX" $POL --steps 4000 --eval-every 500 --vals "$VALS"
echo "== MIX-A-DONE $(date +%T)"
