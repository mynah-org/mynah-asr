#!/bin/bash
# One anomaly/event check of the box, meant to be called every ~5 min by the dev machine's monitor:
#   ssh -p PORT root@HOST 'bash /root/mynah-asr/finetune/jobs/boxwatch.sh'
# Prints ONLY new events since the previous call (state in /root/.boxwatch): error lines (Traceback, Error, OOM,
# FAILED, Killed, GATE) and completion markers (*-DONE, == ok) in the logs, a log unchanged > STALE_MIN while
# its tmux session is alive, a tmux session that disappeared, a full disk / tmpfs, two GPU python processes.
# Heartbeat mode (2026-10-22), run every 300 s by the on-box watchdog:
#   tmux new -d -s watchdog 'while :; do bash /root/mynah-asr/finetune/jobs/boxwatch.sh hb; sleep 300; done'
# appends one block to /root/ft/logs/heartbeat.log: per tmux job its progress token (log size + the optional
# command in /root/.boxwatch/progress/<session>, e.g. a step counter or an output-file count), GPU util / VRAM /
# power, CPU use relative to the cgroup quota, free disk, the last meaningful log line. A token unchanged for 2
# heartbeats -> CHECK, for 3 -> STALLED (unless listed in no_stale); a GPU job (in /root/.boxwatch/gpu_jobs) alive
# with GPU util < 10 % for 2 heartbeats -> GPU-IDLE. Alerts also go to /root/ft/logs/alerts.log (the dev machine
# reads only that file). Costs a few ms of CPU.
FTR=${FT_ROOT:-$HOME/ft}   # data / logs root (/root/ft on the root boxes)
S=$HOME/.boxwatch; mkdir -p $S $S/progress; STALE_MIN=${STALE_MIN:-12}
if [ "${1:-}" = hb ]; then
    H=$FTR/logs/heartbeat.log; A=$FTR/logs/alerts.log; mkdir -p $FTR/logs; ts=$(date +%FT%T)
    alert() { echo "$ts $*" | tee -a $A; }
    { read -r q p < /sys/fs/cgroup/cpu.max; } 2>/dev/null || { q=max; p=100000; }   # no cpu.max at the cgroup root (DGX)
    [ "$q" = max ] && q=$(( $(nproc) * p ))
    u1=$(awk '/usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat); t1=$(date +%s%N)
    u0=$(cat $S/cpu_u 2>/dev/null || echo "$u1"); t0=$(cat $S/cpu_t 2>/dev/null || echo "$t1"); echo "$u1" >$S/cpu_u; echo "$t1" >$S/cpu_t
    cpu=$(awk -v du=$((u1 - u0)) -v dt=$(( (t1 - t0) / 1000 )) -v q=$q -v p=$p 'BEGIN {c = dt > 0 ? du / dt : 0; printf "%.1f of %.1f CPUs (%.0f %%)", c, q / p, 100 * c * p / q}')
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv,noheader,nounits 2>/dev/null)
    gu=$(echo "$gpu" | cut -d, -f1 | tr -d ' ')
    {
    echo "== $ts | GPU util/VRAM MiB/total/W/pstate: $gpu | CPU $cpu | load $(cut -d' ' -f1-3 /proc/loadavg) | free / $(df -Ph / | awk 'NR==2 {print $4}') shm $(df -Ph /dev/shm | awk 'NR==2 {print $4}')"
    for t in $(tmux ls -F '#S' 2>/dev/null | grep -vxE 'ssh_tmux|watchdog'); do
        f=$(ls -t $FTR/logs/$t*.log $HOME/res/$t*.log 2>/dev/null | head -1)
        pid=$(tmux list-panes -t "$t" -F '#{pane_pid}' 2>/dev/null | head -1)
        tok="$( [ -n "$f" ] && stat -c %s "$f")|$( [ -s $S/progress/$t ] && bash $S/progress/$t 2>/dev/null | tr '\n' ' ')"
        old=$(cut -d' ' -f2- $S/hb_$t 2>/dev/null); n=$(cut -d' ' -f1 $S/hb_$t 2>/dev/null || echo 0)
        [ "$tok" = "$old" ] && n=$((n + 1)) || n=0; echo "$n $tok" >$S/hb_$t
        age=$( [ -n "$f" ] && echo $(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))m || echo "-")
        last=$( [ -n "$f" ] && grep -av -E "it/s\]|%\||^\s*$" "$f" | tail -1 | cut -c1-160)
        echo "  $t pid $pid | token [$tok] same x$n | log $(basename "${f:-none}") age $age | $last"
        if ! grep -qxF "$t" $S/no_stale 2>/dev/null; then
            [ $n -eq 2 ] && alert "CHECK: $t progress token unchanged for 2 heartbeats [$tok]"
            [ $n -eq 3 ] && alert "STALLED: $t progress token unchanged for 3 heartbeats [$tok]"
        fi
        if grep -qxF "$t" $S/gpu_jobs 2>/dev/null; then
            g=$(cat $S/gidle_$t 2>/dev/null || echo 0); [ "${gu:-0}" -lt 10 ] && g=$((g + 1)) || g=0; echo $g >$S/gidle_$t
            [ $g -eq 2 ] && alert "GPU-IDLE: $t alive but GPU util ${gu} % for 2 heartbeats"
        fi
        [ -n "$f" ] && tail -c 200000 "$f" | grep -aE "Traceback|CUDA error|out of memory|OOM|Killed" | tail -1 | grep -q . \
            && { e=$(tail -c 200000 "$f" | grep -aE "Traceback|CUDA error|out of memory|OOM|Killed" | tail -1 | cut -c1-160)
                 grep -qF "$t: $e" $S/errs 2>/dev/null || { echo "$t: $e" >>$S/errs; alert "ERROR: $t: $e"; }; }
    done
    prev=$(cat $S/hb_tmux 2>/dev/null); cur=$(tmux ls -F '#S' 2>/dev/null | grep -vxE 'ssh_tmux|watchdog' | sort | tr '\n' ' ')
    for t in $prev; do echo " $cur " | grep -q " $t " || alert "ENDED: tmux $t"; done; echo "$cur" >$S/hb_tmux
    [ -z "$cur" ] && [ "${gu:-0}" -lt 10 ] && echo "  no jobs, GPU idle"
    df -P / /dev/shm | awk 'NR>1 && $5+0 >= 92 {print "DISK: " $6 " " $5 " used"}' | while read -r l; do alert "$l"; done
    } >> $H 2>&1
    exit 0
fi
LOGS=$(ls $HOME/res/*.log $FTR/logs/*.log 2>/dev/null)
for f in $LOGS; do
    k=$S/$(echo "$f" | tr / _); n=$(wc -l <"$f"); o=$(cat "$k" 2>/dev/null || echo 0)
    [ "$n" -lt "$o" ] && o=0
    tail -n +$((o + 1)) "$f" | grep -aE "Traceback|Error|error:|OOM|out of memory|FAILED|Killed|GATE|-DONE|== ok |abort" \
        | grep -avE "^\s*$" | cut -c1-220 | sed "s|^|$(basename $f): |" | tail -8
    echo "$n" >"$k"
done
now=$(date +%s)
# sessions whose log is quiet by design (lock holders, long downloads) are listed in $HOME/.boxwatch/no_stale
for t in $(tmux ls -F '#S' 2>/dev/null | grep -v ssh_tmux | grep -vxF -f <(cat $S/no_stale 2>/dev/null; echo ssh_tmux)); do
    f=$(ls -t $FTR/logs/$t*.log $HOME/res/$t*.log 2>/dev/null | head -1)
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
