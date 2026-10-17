#!/bin/bash
# One anomaly/event check of the box, meant to be called every ~5 min by the dev machine's monitor:
#   ssh -p PORT root@HOST 'bash /root/mynah-asr/finetune/jobs/boxwatch.sh'
# Prints ONLY new events since the previous call (state in /root/.boxwatch): error lines (Traceback, Error, OOM,
# FAILED, Killed, GATE) and completion markers (*-DONE, == ok) in the logs, a log unchanged > STALE_MIN while
# its tmux session is alive, a tmux session that disappeared, a full disk / tmpfs, two GPU python processes.
S=/root/.boxwatch; mkdir -p $S; STALE_MIN=${STALE_MIN:-12}
LOGS=$(ls /root/res/*.log /root/ft/logs/*.log 2>/dev/null)
for f in $LOGS; do
    k=$S/$(echo "$f" | tr / _); n=$(wc -l <"$f"); o=$(cat "$k" 2>/dev/null || echo 0)
    [ "$n" -lt "$o" ] && o=0
    tail -n +$((o + 1)) "$f" | grep -aE "Traceback|Error|error:|OOM|out of memory|FAILED|Killed|GATE|-DONE|== ok |abort" \
        | grep -avE "^\s*$" | cut -c1-220 | sed "s|^|$(basename $f): |" | tail -8
    echo "$n" >"$k"
done
now=$(date +%s)
for t in $(tmux ls -F '#S' 2>/dev/null | grep -v ssh_tmux); do
    f=$(ls -t /root/ft/logs/$t*.log /root/res/$t*.log 2>/dev/null | head -1)
    [ -n "$f" ] && [ $(( (now - $(stat -c %Y "$f")) / 60 )) -gt "$STALE_MIN" ] && echo "STALE: tmux $t, $(basename $f) unchanged > $STALE_MIN min"
done
cur=$(tmux ls -F '#S' 2>/dev/null | grep -v ssh_tmux | sort | tr '\n' ' ')
prev=$(cat $S/tmux 2>/dev/null)
for t in $prev; do echo " $cur " | grep -q " $t " || echo "ENDED: tmux $t"; done
echo "$cur" >$S/tmux
df -P / /dev/shm | awk 'NR>1 && $5+0 >= 92 {print "DISK: " $6 " " $5 " used"}'
g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
[ "$g" -gt 2 ] && echo "GPU: $g compute processes"
exit 0
