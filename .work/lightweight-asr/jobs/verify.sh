#!/bin/bash
# PR #4 auto-default check on a real GPU + "does the CUDA server open this model".
# For each model: --dispatch-map, then a start-up with NO precision/gemm/loop
# flags and one with the explicit pre-PR-4 flags; record the [SERVER-CONFIG] lines.
#   tmux new -d -s verify 'bash /root/verify.sh 2>&1 | tee /root/res/verify.log'
set -u
cd /root/mynah-asr || exit 2
O=/root/res/verify; mkdir -p $O
up() {  # tag model extra-args...
    local tag=$1 m=$2; shift 2
    ./mynah-asr-server-cuda -m models/$m -p 8291 "$@" >$O/$tag.log 2>&1 & local pid=$!
    local t=0
    until curl -sf localhost:8291/v1/health >/dev/null 2>&1; do
        sleep 1; t=$((t + 1))
        kill -0 $pid 2>/dev/null || { echo "$tag: server exited"; tail -5 $O/$tag.log; return; }
        [ $t -ge 180 ] && { echo "$tag: not ready in 180 s"; break; }
    done
    echo "$tag: ready in ${t}s"; grep -E '^\[SERVER-CONFIG\]' $O/$tag.log | cut -c1-400
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/$tag VRAM /"
    kill $pid; wait $pid 2>/dev/null
}
for m in nemotron-3.5-asr-streaming-0.6b parakeet-realtime-eou-120m; do
    echo "== $m dispatch-map"
    timeout 120 ./mynah-asr-server-cuda -m models/$m --dispatch-map >$O/dm-$m.txt 2>&1; echo "rc=$?"
    grep -c . $O/dm-$m.txt; grep -i -E 'unknown|own-tc|bf16|graph' $O/dm-$m.txt | sort | uniq -c | head -12
    echo "== $m start-up, defaults"
    timeout 300 bash -c "$(declare -f up); O=$O; up def-$m $m"
    echo "== $m start-up, pre-PR-4 flags"
    timeout 300 bash -c "$(declare -f up); O=$O; up old-$m $m --precision f32 --gemm own --stage-ahead 0 --host-threads 1 --graphs off --warmup 0"
done
echo "== VERIFY-DONE $(date +%T)"
