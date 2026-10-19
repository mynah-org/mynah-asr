#!/bin/bash
# Smoke / reproduction test of the data_snapshot.py restore pool (box only; needs /root/.hf_token and a manifest).
# Packs a TINY snapshot (the clips of one manifest, ~1 MB shards -> ~20 shards, so as in it-c250 the extraction
# workers start while most downloads are still in flight), then restores it RUNS times with each start method into
# throw-away roots. A run that does not finish within 180 s is a hang: every process of it gets SIGUSR1 (forked
# workers inherit data_snapshot.py's faulthandler, so their Python stacks land in the log), then it is killed.
# Pass = every spawn run completes with the restore gate; the fork runs show whether the hang reproduces.
#   FT_ROOT=/root/ft MANIFEST=/root/ft/mix/it/eval_cv.json RUNS=6 bash finetune/tests/snapshot_pool_smoke.sh
set -u
. "$(dirname "$0")/../jobs/gpu_lock.sh"   # PY, FINETUNE (no GPU lock taken: CPU / network only)
HUB=${HUB:-gabrione/mynah-asr-canary-it-experiments}; NAME=${NAME:-smoke-pool}
MANIFEST=${MANIFEST:?}; RUNS=${RUNS:-6}; W=${WORKERS:-16}; OUT=${OUT:-/root/snaptest}
SNAP="$PY $FINETUNE/common/data_snapshot.py"
mkdir -p $OUT
if [ "${PACK:-1}" = 1 ]; then
    $SNAP pack --repo $HUB --name $NAME --token-file /root/.hf_token --manifests $MANIFEST --shard-gb 0.001 --workers 8 \
        2>&1 | grep -E "== pack|PACK-DONE|FAILED|Traceback" || exit 2
fi
for ctx in ${CTXS:-fork spawn}; do
    ok=0; hang=0; fail=0
    for i in $(seq 1 $RUNS); do
        root=$OUT/$ctx-$i; rm -rf $root; mkdir -p $root; log=$OUT/$ctx-$i.log
        setsid $SNAP restore --repo $HUB --name $NAME --token-file /root/.hf_token --workers $W --tmp $OUT \
            --root $root --mp-context $ctx > $log 2>&1 &
        pid=$!; t=0
        while kill -0 $pid 2>/dev/null && [ $t -lt 180 ]; do sleep 2; t=$((t + 2)); done
        if kill -0 $pid 2>/dev/null; then
            hang=$((hang + 1))
            pids="$pid $(pgrep -P $pid | tr '\n' ' ')"
            ps -o pid,stat,etime,cmd -p ${pids// /,} | cut -c1-80 >> $log
            echo "== SIGUSR1 to $pids" >> $log
            kill -USR1 $pids 2>/dev/null; sleep 3
            kill -9 -- -$pid 2>/dev/null; wait $pid 2>/dev/null
            echo "== $ctx run $i: HANG (stacks in $log)"
        elif grep -q "RESTORE-DONE" $log; then
            ok=$((ok + 1)); echo "== $ctx run $i: ok $(grep -o '[0-9]* s$' $log | tail -1)"
        else
            fail=$((fail + 1)); echo "== $ctx run $i: FAILED"; tail -3 $log
        fi
        rm -rf $root $OUT/snap-*
    done
    echo "== SMOKE $ctx: $ok ok, $hang hang, $fail failed of $RUNS"
done
