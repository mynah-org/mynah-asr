#!/bin/bash
# Autonomous end-of-day close-out on the box (the dev line may drop; the instance and its disk are
# lost when it is destroyed): wait for the last job's marker, stop the archiver loop, one final
# verified archival pass, a survival bundle (logs, manifests, metrics, audits, Mynah evals, teacher
# scores -- no audio, no weights) uploaded and size-verified, the repo listing, token removed.
#   WAIT_LOG=/root/ft/logs/granary_ab.log WAIT_FOR=GRANARY-AB-DONE BUNDLE=survival-2026-10-10 \
#     tmux new -d -s close 'bash finetune/jobs/close_day.sh'
set -u
F=$(cd "$(dirname "$0")/.." && pwd); FT=${FT_ROOT:-/root/ft}; REPO=${HF_REPO:-gabrione/mynah-asr-canary-it-experiments}
PY=${VENV:-/root/nemo-venv}/bin/python; B=${BUNDLE:?}; L=$FT/logs/close_day.log; exec >>$L 2>&1
echo "== close start $(date +%T), waiting for ${WAIT_FOR:-} in ${WAIT_LOG:-}"
[ -n "${WAIT_LOG:-}" ] && timeout ${WAIT_MAX:-14400} bash -c "until grep -q '${WAIT_FOR}' '${WAIT_LOG}' 2>/dev/null; do sleep 30; done"
touch /root/arch/STOP 2>/dev/null; tmux kill-session -t arch 2>/dev/null; sleep 5
echo "== final archival pass $(date +%T)"
timeout 7200 $PY $F/common/hf_archive.py --repo $REPO --ft-root $FT --token-file /root/.hf_token ${ARCH_EXTRA:-} \
    | grep -E "uploaded|VERIFY|pass done|Error" | tail -40
cd /root
tar czf /root/$B.tgz --exclude="*.nemo" --exclude="*.ckpt" --exclude="*.wav" --exclude="*.parquet" --exclude="*.safetensors" \
    --exclude="eouw*" --exclude="stream_wavs" \
    ft/runs ft/logs ft/manifests ft/mix ft/ceiling ft/audit-a0 ft/m1* res 2>/dev/null
ls -la /root/$B.tgz
$PY - "$REPO" "/root/$B.tgz" <<'PYEOF'
import sys
from pathlib import Path
from huggingface_hub import HfApi
repo, p = sys.argv[1], Path(sys.argv[2])
api = HfApi(token=Path("/root/.hf_token").read_text().strip())
api.upload_file(path_or_fileobj=str(p), path_in_repo=f"results/{p.name}", repo_id=repo, commit_message=f"Add the end-of-day survival bundle {p.name}")
i = api.get_paths_info(repo, [f"results/{p.name}"])[0]
print("VERIFY", p.name, (getattr(i, "size", None) or i.lfs.size) == p.stat().st_size, p.stat().st_size)
f = api.list_repo_files(repo)
print("REPO files", len(f), "nemo", sum(x.endswith(".nemo") for x in f), "ckpt", sum(x.endswith(".ckpt") for x in f))
print("REPO runs with nemo:", sorted({x.split("/")[1] for x in f if x.endswith("final.nemo")}))
print("REPO sha", api.repo_info(repo).sha)
PYEOF
rm -f /root/.hf_token; [ -e /root/.hf_token ] && echo "TOKEN STILL PRESENT" || echo "token removed"
echo "== CLOSE-DONE $(date +%T)"
