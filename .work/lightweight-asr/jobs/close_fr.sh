#!/bin/bash
# Autonomous close-out on the box (the dev line may drop): wait for s2b, wait for
# its verified upload, upload + verify the final survival bundle, list the repo,
# stop the archiver, delete the HF token.
set -u
L=/root/ft/logs/close.log; exec >>$L 2>&1
echo "== close start $(date +%T)"
true
timeout 1800 bash -c 'until ls /root/arch/done | grep -q "nemo-plain-fr-s2-eou"; do sleep 20; done'
sleep 150   # one more archiver pass for the final metrics/logs
cd /root
tar czf /root/survival-final3.tgz --exclude="*.nemo" --exclude="*.ckpt" --exclude="*.wav" --exclude="*.parquet" --exclude="eouw*" \
    --exclude="bank-*" --exclude="*.csv" --exclude="bank-fr-peak" --exclude="stream_wavs" --exclude="diag" --exclude="*.safetensors" \
    ft/runs ft/logs ft/probe ft/m1 ft/m2 ft/m2-eou.txt ftfr/logs ftfr/manifests ftfr/eou-metrics.txt ftfr/eou-metrics.json ftfr/meta ft-kit eou-kit ft/models/tok-eou-it arch/README.md \
    res/*.log $(find res -name "*.json" -size -2M) $(find res -name "*.txt" -size -2M) 2>/dev/null
/root/nemo-venv/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi(token=Path("/root/.hf_token").read_text().strip()); repo = "gabrione/mynah-asr-canary-it-experiments"
p = Path("/root/survival-final3.tgz")
api.upload_file(path_or_fileobj=str(p), path_in_repo="results/survival-final3.tgz", repo_id=repo, commit_message="Add the end-of-session survival bundle")
api.upload_folder(folder_path="/root/ft/logs", path_in_repo="logs", repo_id=repo, commit_message="Final logs")
i = api.get_paths_info(repo, ["results/survival-final3.tgz"])[0]
print("VERIFY survival-final3", (getattr(i, "size", None) or i.lfs.size) == p.stat().st_size, p.stat().st_size)
f = api.list_repo_files(repo)
print("REPO files", len(f), "nemo", sum(x.endswith(".nemo") for x in f), "ckpt", sum(x.endswith(".ckpt") for x in f))
print("REPO runs with nemo:", sorted({x.split("/")[1] for x in f if x.endswith("final.nemo")}))
print("REPO sha", api.repo_info(repo).sha)
PY
tmux kill-session -t arch 2>/dev/null
rm -f /root/.hf_token; [ -e /root/.hf_token ] && echo "TOKEN STILL PRESENT" || echo "token removed"
echo "== CLOSE-DONE $(date +%T)"
