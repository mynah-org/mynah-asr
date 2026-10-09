#!/bin/bash
# l2: the multiplexed load generator (stream_load.py --mux) on the CUDA server.
#  1. EOU C1024: old generator vs --mux 16 (same server, bank, reference) ->
#     validates the generator and tells how much of l1's C1024 was harness.
#  2. EOU --mux 16 at C1536 and C2048 (the old generator died there).
#  3. Nemotron --mux 16 at C1024 (was l1's NOT QUALIFIED harness-limited?).
# Holds the GPU lock; starts only when the CPU is quiet (lvl and the FT data
# step finished), because gpu_qualify refuses a busy container.
#   tmux new -d -s l2 'bash /root/l2.sh 2>&1 | tee /root/res/l2.log'
set -u
exec 9>/root/gpu.lock; flock 9
echo "== l2 lock held, waiting for a quiet CPU $(date +%T)"
until grep -q LVL-DONE /root/res/lvl.log 2>/dev/null && ! pgrep -f "[d]ata_it" >/dev/null; do sleep 20; done
ulimit -n 65536
cd /root/mynah-eou || exit 2
git pull -q --ff-only && git log --oneline -1
O=/root/res/l2; mkdir -p $O
E=models/parakeet-realtime-eou-120m; M=models/nemotron-3.5-asr-streaming-0.6b
CORP="--corpus samples/stress-en/manifest.json --corpus-sample 498 --corpus-seed 42"
PIN="--server-cpus 0-31,64-95 --gen-cpus 32-63,96-127"
R() { ls /root/res/q2/$1/*/reference.json | head -1; }
lad() {  # tag mux model lang lookahead ladder cap ref
    echo "== l2 $1 mux=$2 [$6] $(date +%T)"
    LOAD_MUX=$2 timeout 3600 gpu/tools/gpu_qualify.sh -m $3 $CORP --phase ladder --ladder "$6" --ladder-seconds 150 \
        --cap $7 --lang "$4" --lookahead $5 --reference-file $8 $PIN --out $O/$1 >$O/$1.log 2>&1
    echo "rc=$?"; d=$(ls -d $O/$1/*/ | head -1)
    timeout 300 python3 tools/bench/v2_verdict.py $d >$O/$1-verdict.txt 2>&1
    grep -E "ladder-C|emission lag|finalization|per-window|RSS|=>|audio/wall" $O/$1-verdict.txt | cut -c1-170
    grep -h -E '^\[GPU\]|generator' $O/$1.log | cut -c1-200
}
lad eou-old   0  $E ""   1 "1024"      2048 $(R eou-auto)
lad eou-mux   16 $E ""   1 "1024 1536 2048" 2048 $(R eou-auto)
lad nemo-mux  16 $M auto 3 "1024"      1024 $(R nemo-auto)
echo "== L2-DONE $(date +%T)"
