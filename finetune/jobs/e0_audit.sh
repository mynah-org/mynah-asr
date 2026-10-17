#!/bin/bash
# Phase E0: teacher (Nemotron 0.6B) vs student (EOU 120M stock) encoder audit, kd_audit.py on real clips + synthetic
# lengths. No training. NOLOCK=1 when the caller already owns the GPU (e.g. a lock holder that keeps a queued job waiting).
#   MF="mls=...:6,yg=...:6" tmux new -d -s e0 'FT_ROOT=/root/ft bash finetune/jobs/e0_audit.sh 2>&1 | tee -a /root/ft/logs/e0.log'
. "$(dirname "$0")/gpu_lock.sh"
OUT=${OUT:-$FT_ROOT/e0/kd_audit.json}; mkdir -p "$(dirname $OUT)"
T=${TEACHER:-$FT_ROOT/models/nemotron-3.5-asr-streaming-0.6b.nemo}; S=${STUDENT:-$FT_ROOT/models/parakeet_realtime_eou_120m-v1.nemo}
[ -n "${NOLOCK:-}" ] || gpu_lock
timeout 1800 "$PY" "$FINETUNE/eou/diag/kd_audit.py" --teacher $T --student $S --manifests "${MF:?}" --bench 1 --out $OUT 2>&1 \
    | grep -a -v -E "NeMo W|warn|Warning"
echo "== E0-DONE $(date +%T)"
