#!/bin/bash
# Canary 180M in the CUDA server (branch lw-canary-cuda): first compile and the
# correctness gates (stage 2 host decoder, stage 3 GPU decoder f32, bf16 arm,
# server REST/WS/batch identity). Third checkout; models/banks symlinked.
#   tmux new -d -s c1 'bash /root/c1.sh 2>&1 | tee /root/res/c1.log'
set -u
# one GPU workload at a time (see ft/run_all.sh)
exec 9>/root/gpu.lock; flock 9
B=/root/mynah-asr; N=/root/mynah-canary; O=/root/res/c1; mkdir -p $O
M=models/canary-180m-flash
echo "== c1 build $(date +%T)"
[ -d $N ] || git clone -q https://github.com/mynah-org/mynah-asr $N
git -C $N fetch -q && git -C $N checkout -q lw-canary-cuda && git -C $N pull -q --ff-only
git -C $N log --oneline -1
rm -rf $N/models; ln -s $B/models $N/models
cd $N || exit 2
( timeout 1800 make -j32 lib && timeout 1800 make -j32 all && timeout 2400 make -C gpu -j32 CUDA_ARCH=sm_89 \
  && timeout 1200 make -C gpu test-aed CUDA_ARCH=sm_89 ) >$O/build.log 2>&1 || { echo "== c1 BUILD FAILED"; grep -E "error|Error" $O/build.log | head -30; exit 1; }
echo "== c1 dispatch-map $(date +%T)"
timeout 120 ./mynah-asr-server-cuda -m $M --dispatch-map 2>&1 | tail -15
echo "== c1 stage2 host decoder $(date +%T)"
timeout 900 tests/test_cuda_aed $M --aed-decoder host >$O/s2.txt 2>&1; echo "rc=$?"; grep -E "OK|FAIL|PASS|CER|differ" $O/s2.txt | head -20
echo "== c1 stage3 gpu decoder f32 $(date +%T)"
timeout 900 tests/test_cuda_aed $M >$O/s3.txt 2>&1; echo "rc=$?"; grep -E "OK|FAIL|PASS|CER|differ" $O/s3.txt | head -20
echo "== c1 stage3 bf16 $(date +%T)"
timeout 900 tests/test_cuda_aed $M --precision bf16 >$O/s3bf16.txt 2>&1; echo "rc=$?"; grep -E "OK|FAIL|PASS|CER|differ" $O/s3bf16.txt | head -20
echo "== c1 server $(date +%T)"
timeout 900 tests/test_cuda_aed_server.sh $M 8297 cuda >$O/srv.txt 2>&1; echo "rc=$?"; tail -15 $O/srv.txt
echo "== C1-DONE $(date +%T)"
