#!/bin/bash
# S15-3 first concurrency screen on the CUDA server, PR #4 defaults (bf16 own-tc,
# stage-ahead, graphs, warm-up, host threads), one model at a time, nothing else
# on the box. Method of .work/l40s-2026-10-08-asr-cuda.md: gpu_qualify ladder,
# 150 s rungs, fresh server per rung, cohort 40 ms, server on the GPU's node
# (0-31,64-95), generator on the other (32-63,96-127), stress-en 498 (seed 42)
# judged against the unloaded reference q2 produced for the same model/arm.
#   tmux new -d -s l1 'bash /root/l1.sh 2>&1 | tee /root/res/l1.log'
set -u
# one GPU workload at a time (see ft/run_all.sh)
exec 9>/root/gpu.lock; flock 9
ulimit -n 65536
cd /root/mynah-eou || exit 2
O=/root/res/l1; mkdir -p $O
E=models/parakeet-realtime-eou-120m; M=models/nemotron-3.5-asr-streaming-0.6b
CORP="--corpus samples/stress-en/manifest.json --corpus-sample 498 --corpus-seed 42"
PIN="--server-cpus 0-31,64-95 --gen-cpus 32-63,96-127"
R() { ls /root/res/q2/$1/*/reference.json | head -1; }
lad() {  # tag model lang lookahead ladder cap ref
    echo "== l1 ladder $1 [$5] $(date +%T)"
    timeout 3600 gpu/tools/gpu_qualify.sh -m $2 $CORP --phase ladder --ladder "$5" --ladder-seconds 150 \
        --cap $6 --lang "$3" --lookahead $4 --reference-file $7 $PIN --out $O/$1 >$O/$1.log 2>&1
    echo "rc=$?"
    d=$(ls -d $O/$1/*/ | head -1)
    timeout 300 python3 tools/bench/v2_verdict.py $d >$O/$1-verdict.txt 2>&1; tail -30 $O/$1-verdict.txt
    grep -h '^\[GPU\]' $O/$1.log | cut -c1-220
}
lad eou  $E ""   1 "512 1024 1536 2048" 2048 $(R eou-auto)
lad nemo $M auto 3 "512 768 1024"       1024 $(R nemo-auto)
echo "== L1-DONE $(date +%T)"
