#!/bin/bash
# EOU 120M end-to-end gate, steps 1-7 of .work/parakeet-eou-l4.md 3a, on a GPU.
# New code in a SECOND checkout (/root/mynah-eou) so the q1 checkout keeps its
# binaries; models and banks are symlinked from /root/mynah-asr.
#   tmux new -d -s g1 'bash /root/g1.sh 2>&1 | tee /root/res/g1.log'
set -u
B=/root/mynah-asr; N=/root/mynah-eou; O=/root/res/g1; mkdir -p $O
E=models/parakeet-realtime-eou-120m; M=models/nemotron-3.5-asr-streaming-0.6b
echo "== g1 build $(date +%T)"
[ -d $N ] || git clone -q https://github.com/mynah-org/mynah-asr $N
git -C $N fetch -q && git -C $N checkout -q research/lightweight-asr && git -C $N pull -q --ff-only
git -C $N log --oneline -1
rm -rf $N/models; ln -s $B/models $N/models
for d in eval-bank stress-en; do [ -e $N/samples/$d ] || ln -s $B/samples/$d $N/samples/$d; done
cd $N || exit 2
( timeout 1800 make -j32 lib && timeout 1800 make -j32 all && timeout 2400 make -C gpu -j32 CUDA_ARCH=sm_89 \
  && timeout 1200 make -C gpu test-stream CUDA_ARCH=sm_89 ) >$O/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $O/build.log; exit 1; }
echo "== g1 base test-stream build $(date +%T)"
( cd $B && timeout 1200 make -C gpu test-stream CUDA_ARCH=sm_89 ) >$O/build-base.log 2>&1 || { echo "BASE BUILD FAILED"; tail -5 $O/build-base.log; }
python3 - <<'PY'
import wave
r = lambda p: (lambda w: w.readframes(w.getnframes()))(wave.open(p))
o = wave.open('/root/res/g1/two.wav', 'wb'); o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000)
o.writeframes(r('samples/en/fleurs_1521.wav') + b'\0\0' * 16000 + r('samples/en/fleurs_1534.wav'))
PY
CL="$O/two.wav samples/en/fleurs_1521.wav samples/en/fleurs_1534.wav samples/en/fleurs_long.wav"
echo "== g1 step1 dispatch-map + start-up $(date +%T)"
timeout 120 ./mynah-asr-server-cuda -m $E --dispatch-map >$O/dm-eou.txt 2>&1; echo "rc=$?"; grep -i -E "unknown|prompt" $O/dm-eou.txt
./mynah-asr-server-cuda -m $E -p 8291 >$O/srv-eou-def.log 2>&1 & P=$!
for i in $(seq 60); do curl -sf localhost:8291/v1/health >/dev/null && break; sleep 1; done
grep '^\[SERVER-CONFIG\]' $O/srv-eou-def.log | cut -c1-330; kill $P; wait $P 2>/dev/null
echo "== g1 step2-6 test_cuda_stream f32 own (CPU vs CUDA) $(date +%T)"
timeout 900 tests/test_cuda_stream $E --lookahead 1 $CL >$O/ts-eou-f32.txt 2>&1; echo "rc=$?"; grep -i -E "gate|eou|identical|differ|FAIL|PASS" $O/ts-eou-f32.txt | head -25
echo "== g1 step7 test_cuda_stream bf16 own-tc $(date +%T)"
timeout 900 tests/test_cuda_stream $E --lookahead 1 --gemm own-tc --precision bf16 $CL >$O/ts-eou-bf16.txt 2>&1; echo "rc=$?"; grep -i -E "gate|eou|identical|differ|FAIL|PASS" $O/ts-eou-bf16.txt | head -25
echo "== g1 server frames CPU server vs CUDA server f32 $(date +%T)"
./mynah-asr-server -m $E --quant f32 -p 8090 >$O/srv-cpu.log 2>&1 & P1=$!
./mynah-asr-server-cuda -m $E -p 8091 --precision f32 --gemm own >$O/srv-cuda.log 2>&1 & P2=$!
for i in $(seq 60); do curl -sf localhost:8090/v1/health >/dev/null && curl -sf localhost:8091/v1/health >/dev/null && break; sleep 1; done
for p in 8090 8091; do timeout 120 python3 tests/ws_probe.py --port $p --lang '' --lookahead '' frames --clip $O/two.wav --expect-model-eou 2 >$O/frames-$p.jsonl 2>&1; echo "probe $p rc=$?"; done
kill $P1 $P2; wait $P1 $P2 2>/dev/null
diff $O/frames-8090.jsonl $O/frames-8091.jsonl >$O/frames.diff && echo "frames IDENTICAL" || { echo "frames DIFFER"; head -20 $O/frames.diff; }
grep -E '"type": ?"eou"' $O/frames-8091.jsonl | head -4
echo "== g1 Nemotron identity base 4378448 vs HEAD $(date +%T)"
NC="samples/en/fleurs_1521.wav samples/en/fleurs_long.wav"
for g in "" "--gemm own-tc --precision bf16"; do
    (cd $B && timeout 900 tests/test_cuda_stream $M $g $NC) >$O/nemo-base.txt 2>&1
    timeout 900 tests/test_cuda_stream $M $g $NC >$O/nemo-head.txt 2>&1
    if diff <(grep -v -E "ms|time|took" $O/nemo-base.txt) <(grep -v -E "ms|time|took" $O/nemo-head.txt) >/dev/null; then echo "nemotron [$g] IDENTICAL"; else echo "nemotron [$g] DIFFER"; diff <(grep -v -E "ms|time|took" $O/nemo-base.txt) <(grep -v -E "ms|time|took" $O/nemo-head.txt) | head -10; fi
done
(cd $B && timeout 120 ./mynah-asr-server-cuda -m $M --dispatch-map) >$O/dm-nemo-base.txt 2>&1
timeout 120 ./mynah-asr-server-cuda -m $M --dispatch-map >$O/dm-nemo-head.txt 2>&1
diff $O/dm-nemo-base.txt $O/dm-nemo-head.txt >/dev/null && echo "nemotron dispatch-map IDENTICAL" || echo "nemotron dispatch-map DIFFER"
echo "== G1-DONE $(date +%T)"
