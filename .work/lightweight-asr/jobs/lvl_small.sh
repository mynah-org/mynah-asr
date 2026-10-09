#!/bin/bash
# Replace lvl.sh's full EOU-metrics pass on the peak bank (100 clips, 3 gaps)
# with a short one (40 clips, gap 1 s): suspend the big one when it starts,
# run the short one, then kill the big one so lvl.sh reaches LVL-DONE after it.
set -u
cd /root/mynah-eou || exit 2
O=/root/res/lvl
until pgrep -f "[e]ouw-peak" >/dev/null; do sleep 5; done
pkill -STOP -f "[e]ouw-peak"; echo "== lvl-small big pass suspended $(date +%T)"
timeout 1800 python3 tools/eval/eou_metrics.py -m models/parakeet-realtime-eou-120m \
    --manifest $O/bank-peak/manifest.json --root $O/bank-peak --mode both --gaps 1 \
    --vad models/silero-vad --limit 40 --jobs 14 --work-dir $O/eouw-p40 \
    --json $O/eou-metrics-p40.json >$O/eou-metrics-p40.txt 2>&1
echo "== lvl-small rc=$? $(date +%T)"; tail -12 $O/eou-metrics-p40.txt
pkill -KILL -f "[e]ouw-peak"; echo "== lvl-small big pass killed $(date +%T)"
