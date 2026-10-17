#!/bin/bash
# Day results to the Hub WITHOUT weights: stage the given files / dirs (small JSON / logs as they are, *.json and
# *.jsonl over 5 MB gzipped), upload the folder to results/$DEST, verify every file's size on the Hub, list it.
# Runs' metrics / probes / logs go through hf_archive.py --nemo-tags '^$' --ckpt-tags '^$' (no final.nemo, no
# last.ckpt): null arms keep their numbers, not their weights.
#   DEST=l40s-jp-2026-10-17 PATHS="/root/ft/d/yg7 /root/ft/e0 ..." bash finetune/jobs/archive_results.sh
set -u
: "${DEST:?}" "${PATHS:?}"
PY=${VENV:-/root/nemo-venv}/bin/python; REPO=${HF_REPO:-gabrione/mynah-asr-canary-it-experiments}
S=/root/stage-$DEST; rm -rf $S; mkdir -p $S
for p in $PATHS; do
    [ -e "$p" ] || { echo "== missing $p"; continue; }
    d=$S/$(echo "${p#/root/}" | tr / _)
    if [ -d "$p" ]; then cp -r "$p" "$d"; else cp "$p" "$d"; fi
done
find $S -type f \( -name "*.json" -o -name "*.jsonl" \) -size +5M -exec gzip -9 {} \;
du -sh $S; find $S -type f | wc -l
$PY - "$REPO" "$S" "$DEST" <<'PYEOF'
import sys
from pathlib import Path
from huggingface_hub import HfApi
repo, s, dest = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
api = HfApi(token=Path("/root/.hf_token").read_text().strip())
api.upload_folder(folder_path=str(s), path_in_repo=f"results/{dest}", repo_id=repo, commit_message=f"Add results/{dest}")
files = [p for p in s.rglob("*") if p.is_file()]
info = {i.path: (getattr(i, "size", None) or i.lfs.size) for i in
        api.get_paths_info(repo, [f"results/{dest}/{p.relative_to(s)}" for p in files])}
bad = [str(p) for p in files if info.get(f"results/{dest}/{p.relative_to(s)}") != p.stat().st_size]
print("VERIFY", dest, len(files), "files", "OK" if not bad else f"MISMATCH {bad[:5]}")
PYEOF
