#!/bin/bash
# a1: EOU 120M vs Nemotron 0.6B CUDA perf audit, step 1 = MEASURE (no code
# change). PR #4 defaults, same clips, server on the GPU's node.
#  host arm  : --profile-host (graphs on, the serving loop as shipped) at
#              C1/32/256/1024 -> [HOSTP-LEVEL] host vs device-wait vs cohort-wait
#              per phase, + [GPU] SM/W, + client lag;
#  stage arm : --profile-stages (per-stage CUDA events; graphs off by design)
#              at C32 and C1024 -> device ms per stage (GEMM, attention, conv,
#              RNNT predictor/joint, ...) and passes, normalised per audio s
#              in the report;
# Holds the GPU lock.
#   tmux new -d -s a1 'bash /root/a1.sh 2>&1 | tee /root/res/a1.log'
set -u
until grep -q FT2-DONE /root/ft/logs/ft2.log 2>/dev/null; do sleep 30; done   # FT first
exec 9>/root/gpu.lock; flock 9
ulimit -n 65536
cd /root/mynah-eou || exit 2
O=/root/res/a1; mkdir -p $O
CL="$(ls samples/stress-en/*/*.wav | tr '\n' ' ')"   # every class: the soak needs short/medium/long
knee() {  # tag model lang lookahead levels srv_args
    echo "== a1 $1 [$5] $(date +%T)"
    MODEL=$2 LANG_Q="$3" LOOKAHEAD=$4 LEVELS="$5" GEMM=auto SRV_ARGS="$6" DUR=60 WARM=15 \
    PIN=0-31,64-95 CLIPIN=32-63,96-127 CLIPS="$CL" TAG=$1 OUT=$O \
        timeout 2400 gpu/tools/gpu_knee.sh >$O/$1.txt 2>&1
    echo "rc=$?"; grep -E "^C|\[GPU\]|HOSTP-LEVEL|lag|p95" $O/$1.txt | cut -c1-230 | head -30
}
E=models/parakeet-realtime-eou-120m; M=models/nemotron-3.5-asr-streaming-0.6b
knee eou-host   $E ""   1 "1 32 256 1024" "--profile-host"
knee nemo-host  $M auto 3 "1 32 256 1024" "--profile-host"
knee eou-stage  $E ""   1 "32 1024"       "--profile-stages"
knee nemo-stage $M auto 3 "32 1024"       "--profile-stages"
# nsys is not installed on this box (only ncu): kernel/stage costs come from
# --profile-stages; launch counts per audio second are owed to a box with nsys.
echo "== A1-DONE $(date +%T)"
