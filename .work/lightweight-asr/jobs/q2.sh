#!/bin/bash
# EOU 120M gate steps 7-9 with the model-EOU reset build (/root/mynah-eou):
#  C. GPU server unloaded transcripts on the 498-clip stress-en bank (seed 42):
#     EOU f32 own, EOU auto (bf16 own-tc), Nemotron auto -> transcript_ab
#     (bf16 parity on THIS model; EOU vs Nemotron on the same 498 clips).
#  A. FLEURS EN 200 (q1 bank) with the new CLI: WER after the reset.
#  B. eou_metrics: speech-end -> EOU latency, premature/missed, A+gap+B.
# C first and alone (gpu_qualify refuses a busy container), then A || B on CPU.
#   tmux new -d -s q2 'bash /root/q2.sh 2>&1 | tee /root/res/q2.log'
set -u
cd /root/mynah-eou || exit 2
O=/root/res/q2; mkdir -p $O
E=models/parakeet-realtime-eou-120m; M=models/nemotron-3.5-asr-streaming-0.6b
CORP="--corpus samples/stress-en/manifest.json --corpus-sample 498 --corpus-seed 42"
ref() {  # tag model lang lookahead extra-qualify-args
    echo "== q2 ref $1 $(date +%T)"
    timeout 2400 gpu/tools/gpu_qualify.sh -m $2 $CORP --phase reference --lang "$3" --lookahead $4 \
        --out $O/$1 ${@:5} >$O/$1.log 2>&1; echo "rc=$?"; ls $O/$1/*/reference.json 2>/dev/null | head -1
}
ref eou-f32  $E "" 1 --precision f32 --gemm own
ref eou-auto $E "" 1
ref nemo-auto $M auto 3
R() { ls $O/$1/*/reference.json | head -1; }
echo "== q2 AB eou f32 -> bf16 $(date +%T)"
python3 gpu/tools/transcript_ab.py $(R eou-f32) $(R eou-auto) --manifest samples/stress-en/manifest.json --lang en 2>&1 | tail -25
echo "== q2 AB nemotron(auto) -> eou(auto) $(date +%T)"
python3 gpu/tools/transcript_ab.py $(R nemo-auto) $(R eou-auto) --manifest samples/stress-en/manifest.json --lang en 2>&1 | grep -v -E "^\s*(A|B|REF)\s*:" | tail -14
echo "== q2 A+B start $(date +%T)"
make -s fetch-vad >/dev/null 2>&1
( timeout 3600 python3 tools/eval/lang_gate.py -m $E --quant f32 --mode stream --langs en --label eou-reset \
    --manifest samples/eval-bank/manifest.json --root samples/eval-bank --json $O/eou-reset-en.json >$O/eou-reset-en.txt 2>&1
  echo "== q2 lang_gate rc=$? $(date +%T)" ) &
( timeout 3600 python3 tools/eval/eou_metrics.py -m $E --manifest samples/eval-bank/manifest.json --root samples/eval-bank \
    --mode both --vad models/silero-vad --limit 200 --jobs 12 --work-dir $O/eouw --json $O/eou-metrics.json >$O/eou-metrics.txt 2>&1
  echo "== q2 eou_metrics rc=$? $(date +%T)" ) &
wait
grep -E "WER |WER\*|CER |pooled|empty" $O/eou-reset-en.txt
tail -30 $O/eou-metrics.txt
echo "== Q2-DONE $(date +%T)"
