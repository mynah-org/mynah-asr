#!/bin/bash
# Resume the Italian EOU specialist on an EMPTY box (after prov.sh), no teacher re-run:
# 1. archived teacher scores + reference checkpoints from the private Hub repo (network);
# 2. data_mix.sh it (MLS + CV + VP + FLEURS audio in /dev/shm; deterministic, same paths);
# 3. the VP filter from the archived VP scores (a2_vpfilter.sh FILTER_ONLY=1);
# 4. the Granary-it pool audio (granary_build.sh; the archived pool scores skip the teacher)
#    and the nested subsets (pseudo_subsets.py);
# 5. the clean-it holdout bank (prepare_clean_it.py);
# 6. gate: every scored clip's audio exists (the scores are only valid on the same clips).
#   FT_ROOT=/root/ft tmux new -d -s resume 'FT_ROOT=/root/ft bash finetune/jobs/resume_it.sh 2>&1 | tee -a /root/ft/logs/resume.log'
. "$(dirname "$0")/gpu_lock.sh"
HUB=${HUB:-gabrione/mynah-asr-canary-it-experiments}
MX=$FT_ROOT/mix/it; YG=$FT_ROOT/mix/it-yg; R=$FT_ROOT/runs
mkdir -p $MX $YG $FT_ROOT/logs
hub() {  # hub <path in repo> <dest>: resumable, authenticated, skipped when present
    [ -s "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    curl -fsSL --retry 5 -C - -H "Authorization: Bearer $(cat /root/.hf_token)" \
        -o "$2" "https://huggingface.co/$HUB/resolve/main/$1" || { echo "== hub FAILED $1"; return 1; }
}
echo "== resume_it $(date +%T)"
hub data/it/train_vp.teacher.json $MX/train_vp.teacher.json || exit 2
hub data/it/granary_yg_pool.teacher.json $YG/train_yg.teacher.json || exit 2
for run in ${RUNS:-plain-it-e3-s8000 plain-it-p1-add25 plain-it-s2-e3-djlr}; do   # 2026-10-12 references: E3-8k, C (best plain), EOU-v3
    hub runs/$run/final.nemo $R/$run/final.nemo && hub runs/$run/metrics.json $R/$run/metrics.json || exit 2
done
echo "== hub ok $(date +%T)"
bash "$FINETUNE/jobs/data_mix.sh" it || exit 3
FILTER_ONLY=1 bash "$FINETUNE/jobs/a2_vpfilter.sh" || exit 4
bash "$FINETUNE/jobs/granary_build.sh" || exit 5
"$PY" "$FINETUNE/eou/pseudo_subsets.py" --pool $YG/train_yg.teacher.json --out $YG/subsets | tail -12
# clean-it holdout (mTEDx + ITALIC; deterministic selection, ~1.8 GB of downloads, scored post hoc only)
[ -s $FT_ROOT/manifests/eval_tedx_it.json ] && [ -s $FT_ROOT/manifests/eval_italic_it.json ] || \
    timeout 5400 "$PY" "$FINETUNE/eou/prepare_clean_it.py" --out $FT_ROOT/manifests --audio /dev/shm/clean-it | tail -3
"$PY" - $MX/train_vp.teacher.json $YG/train_yg.teacher.json $YG/subsets/yg_agree\<=15.json <<'PY' || exit 6
import json, os, sys
bad = 0
for p in sys.argv[1:]:
    rs = [json.loads(l) for l in open(p)]
    miss = sum(1 for r in rs if not os.path.exists(r["audio_filepath"]))
    print(f"== audio gate {os.path.basename(p)}: {len(rs) - miss}/{len(rs)} present")
    bad += miss
sys.exit(1 if bad else 0)
PY
df -h / /dev/shm | tail -2
echo "== RESUME-IT-DONE $(date +%T)"
