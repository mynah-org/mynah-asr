#!/bin/bash
# shellcheck disable=SC2034  # KIT, PY, V are used by the scripts that source this file
# Sourced by every step script of the Canary 180M Italian kit.
# Layout on the box:  $FT (default /root/ft) = data, models, runs, logs, markers.
#                     $KIT = this directory (scripts only, no state).
# Every step writes $FT/done/<step> on success and is skipped when re-run;
# delete the marker to redo a step.
set -u
FT=${FT_ROOT:-/root/ft}
export FT_ROOT=$FT
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
V=${VENV:-/root/nemo-venv}
PY=$V/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$FT/done" "$FT/logs"

step() { [ -e "$FT/done/$1" ] && { echo "== skip $1 (marker $FT/done/$1)"; return 1; }; echo "== $1 $(date +%T)"; return 0; }
ok() { date '+%F %T' >"$FT/done/$1"; echo "== ok $1 $(date +%T)"; }
fail() { echo "== FAILED $1 rc=$2 $(date +%T)"; exit "$2"; }
free_gb() { df -BG --output=avail "$FT" | tail -1 | tr -dc '0-9'; }
need_gb() {  # need_gb <GB> <what>
    local f; f=$(free_gb)
    if [ "${f:-0}" -lt "$1" ]; then echo "== not enough disk for $2: ${f} GB free, need $1 GB"; exit 4; fi
}
