#!/bin/bash
# shellcheck disable=SC2034
# Sourced by the box jobs: one GPU workload at a time, every job time-boxed.
#   . "$(dirname "$0")/gpu_lock.sh"
#   gpu_lock                      # blocks until /root/gpu.lock (GPU_LOCK) is free, holds it on fd 9
#   timeout 10800 python ...      # every GPU step carries its own hard timeout
# Network/CPU steps (downloads, tokenizers, manifests) run BEFORE gpu_lock so
# they overlap another job's GPU time. The lock is released when the job exits.
# Benchmarks and gates on the same box must take the same lock file.
GPU_LOCK=${GPU_LOCK:-/root/gpu.lock}
FINETUNE=${FINETUNE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
: "${FT_ROOT:?set FT_ROOT explicitly (data, models and runs root, e.g. /root/ft)}"
export FT_ROOT PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
PY=${VENV:-/root/nemo-venv}/bin/python
gpu_lock() {
    exec 9>"$GPU_LOCK"
    echo "== waiting for $GPU_LOCK $(date +%T)"
    flock 9
    echo "== $GPU_LOCK held $(date +%T)"
}
