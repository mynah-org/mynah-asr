#!/bin/bash
# S15-2 / S15-4 first quality pass: the SAME audio (FLEURS test, samples/eval-bank,
# tools/fetch_eval_bank.py --n 200), the SAME scorer and normaliser
# (tools/eval/lang_gate.py -> streaming_metrics.normalise), f32 weights for all.
#   EOU 120M   stream, lang en (the CLI ignores lang for a prompt-less pack)
#   Nemotron   stream, lang en (lang_gate passes the clip language) and lang auto
#              (product default; the later --lang in --extra wins in the CLI parser)
#   Canary 180M offline, lang = the clip's language
#   tmux new -d -s q1 'bash /root/q1.sh 2>&1 | tee /root/res/q1.log'
set -u
cd /root/mynah-asr || exit 2
O=/root/res/q1; mkdir -p $O
run() {  # tag model mode langs extra
    local tag=$1; echo "== $tag start $(date +%T)"
    timeout 3600 python3 tools/eval/lang_gate.py -m models/$2 --quant f32 --mode $3 --langs $4 \
        --extra "$5" --label $tag --json $O/$tag.json >$O/$tag.txt 2>&1
    echo "== $tag rc=$? $(date +%T)"
}
run eou-en       parakeet-realtime-eou-120m       stream  en "" &
run nemo-lang-en nemotron-3.5-asr-streaming-0.6b  stream  en "" &
run canary-en    canary-180m-flash                offline en "" &
wait
run nemo-auto-en nemotron-3.5-asr-streaming-0.6b  stream  en "--lang auto" &
run canary-fr    canary-180m-flash                offline fr "" &
run nemo-lang-fr nemotron-3.5-asr-streaming-0.6b  stream  fr "" &
wait
for f in $O/*.txt; do echo "--- $f"; grep -E -i "wer|cer|sub|del|ins|empty" $f | head -8; done
echo "== Q1-DONE $(date +%T)"
