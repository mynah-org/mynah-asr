#!/bin/bash
# Resume the Italian EOU specialist on an EMPTY box (after prov.sh), no teacher re-run:
# 1. archived teacher scores + reference checkpoints from the private Hub repo (network);
# 2. data_mix.sh it (MLS + CV + VP + FLEURS audio in /dev/shm; deterministic, same paths);
# 3. the VP filter from the archived VP scores (a2_vpfilter.sh FILTER_ONLY=1);
# 4. the Granary-it pool audio (granary_build.sh; the archived pool scores skip the teacher)
#    and the nested subsets (pseudo_subsets.py);
# 5. the clean-it holdout bank (prepare_clean_it.py);
# 6. gate: every scored clip's audio exists (the scores are only valid on the same clips).
# SNAPSHOT=<name> (e.g. it-c250): steps 2-5 come from the data snapshot on the Hub (data_snapshot.sh) instead of
# the public sources; the gate then covers the snapshot's manifests (the full teacher pools are not in it).
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
# 2026-10-13 references: C250-20k (best plain), C250, C, E3-8k, EOU-v3
for run in ${RUNS:-plain-it-c250-20k plain-it-c250 plain-it-p1-add25 plain-it-e3-s8000 plain-it-s2-e3-djlr}; do
    hub runs/$run/final.nemo $R/$run/final.nemo && hub runs/$run/metrics.json $R/$run/metrics.json || exit 2
done
# the NEW-video pool (it100/it101, C250's second external source): its teacher scores live in the 2026-10-13
# bundle; with them in place granary_build.sh only re-downloads the audio (no teacher pass). POOL6=0 skips it.
YG6=$FT_ROOT/mix/it-yg6
if [ "${POOL6:-1}" = 1 ] && [ ! -s $YG6/train_yg.teacher.json ]; then
    hub results/survival-2026-10-13.tgz /root/survival-2026-10-13.tgz || exit 2
    mkdir -p $YG6 && tar xzf /root/survival-2026-10-13.tgz -O ft/mix/it-yg6/train_yg.teacher.json > $YG6/train_yg.teacher.json || exit 2
fi
echo "== hub ok $(date +%T)"
if [ -n "${SNAPSHOT:-}" ]; then
    timeout 5400 "$PY" "$FINETUNE/common/data_snapshot.py" restore --repo $HUB --name $SNAPSHOT --token-file /root/.hf_token \
        --workers ${SNAP_WORKERS:-16} --tmp /root || { echo "== snapshot restore FAILED"; exit 7; }
    df -h / /dev/shm | tail -2
    echo "== RESUME-IT-DONE (snapshot $SNAPSHOT) $(date +%T)"
    exit 0
fi
bash "$FINETUNE/jobs/data_mix.sh" it || exit 3
FILTER_ONLY=1 bash "$FINETUNE/jobs/a2_vpfilter.sh" || exit 4
bash "$FINETUNE/jobs/granary_build.sh" || exit 5
"$PY" "$FINETUNE/eou/pseudo_subsets.py" --pool $YG/train_yg.teacher.json --out $YG/subsets | tail -12
G6=()
if [ "${POOL6:-1}" = 1 ]; then   # same arguments as c_scale.sh's build (deterministic shard selection)
    YG_OUT=$YG6 YG_AUDIO=/dev/shm/mix/it-yg6 YG_EXT=flac YG_DIRS=it100,it101 YG_SPREAD=60,55 YG_HOURS=235 YG_WORKERS=8 \
        YG_MIN_FREE_GB=4 bash "$FINETUNE/jobs/granary_build.sh" || exit 5
    "$PY" "$FINETUNE/eou/pseudo_subsets.py" --pool $YG6/train_yg.teacher.json --out $YG6/subsets | tail -12
    G6=($YG6/train_yg.teacher.json "$YG6/subsets/yg_agree<=15.json")
fi
# clean-it holdout (mTEDx + ITALIC; deterministic selection, ~1.8 GB of downloads, scored post hoc only)
[ -s $FT_ROOT/manifests/eval_tedx_it.json ] && [ -s $FT_ROOT/manifests/eval_italic_it.json ] || \
    timeout 5400 "$PY" "$FINETUNE/eou/prepare_clean_it.py" --out $FT_ROOT/manifests --audio /dev/shm/clean-it | tail -3
"$PY" - $MX/train_vp.teacher.json $YG/train_yg.teacher.json $YG/subsets/yg_agree\<=15.json "${G6[@]}" <<'PY' || exit 6
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
