#!/bin/bash
# gpu/tools/topology.sh — one [TOPOLOGY] line for a GPU benchmark's evidence:
# which GPU and driver, which NUMA node the GPU hangs off, which CPU, how many
# CPUs this process may really use (affinity AND the cgroup quota), the
# governor, the load already on the host, and -- when the server and the load
# generator are pinned -- which NUMA nodes those CPU sets sit on.
#
#   gpu/tools/topology.sh [--device 0] [--server-cpus 0-31] [--gen-cpus 64-95]
#
# Prints the line on stdout and one WARNING line per contradiction it can see
# (server and generator sharing CPUs, the server away from the GPU's node, a
# quota below the affinity count). Exit 0 always: it reports, the caller decides.
set -u
DEV=0; SRV_CPUS=""; GEN_CPUS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --device) DEV="$2"; shift 2 ;;
        --server-cpus) SRV_CPUS="$2"; shift 2 ;;
        --gen-cpus) GEN_CPUS="$2"; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "topology: unknown option: $1" >&2; exit 2 ;;
    esac
done

q() { nvidia-smi -i "$DEV" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^ *//;s/ *$//'; }
GPU=$(q name); DRV=$(q driver_version); PCI=$(q pci.bus_id)
PL=$(q power.limit); SMMAX=$(q clocks.max.sm); VTOT=$(q memory.total)
# sysfs wants the 4-digit domain, lower case
SYS=$(echo "$PCI" | tr 'A-F' 'a-f' | sed 's/^0000\(....:\)/\1/')
GNODE=$(cat "/sys/bus/pci/devices/$SYS/numa_node" 2>/dev/null || echo "?")
[ "$GNODE" = "-1" ] && GNODE="none(-1)"

CPU=$(awk -F: '/^model name/{sub(/^[ \t]+/,"",$2); print $2; exit}' /proc/cpuinfo 2>/dev/null)
ONLINE=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "?")
USABLE=$(nproc 2>/dev/null || echo "?")
AFF=$(awk '/^Cpus_allowed_list/{print $2}' /proc/self/status 2>/dev/null)
QUOTA=none
if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r a b < /sys/fs/cgroup/cpu.max
    [ "$a" != "max" ] && QUOTA=$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.2f", a/b}')
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    a=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); b=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$a" -gt 0 ] 2>/dev/null && QUOTA=$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.2f", a/b}')
fi
NODES=$(cat /sys/devices/system/node/online 2>/dev/null || echo "?")
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a")
LOAD=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)

# the NUMA nodes a CPU list touches, e.g. "0-15,64-79" -> "0"
expand() {
    echo "$1" | tr ',' '\n' | while IFS=- read -r lo hi; do
        [ -z "$lo" ] && continue; [ -z "$hi" ] && hi=$lo; seq "$lo" "$hi"; done
}
nodes_of() {
    expand "$1" | while read -r c; do
        for n in /sys/devices/system/cpu/cpu$c/node*; do [ -e "$n" ] && basename "$n" | sed 's/node//'; done
    done | sort -un | paste -sd, -
}
SN="-"; GN="-"
[ -n "$SRV_CPUS" ] && SN=$(nodes_of "$SRV_CPUS")
[ -n "$GEN_CPUS" ] && GN=$(nodes_of "$GEN_CPUS")

echo "[TOPOLOGY] gpu=\"$GPU\" driver=$DRV pci=$PCI gpu_numa_node=$GNODE power_limit_w=$PL sm_clock_max_mhz=$SMMAX vram_total_mib=$VTOT cpu=\"$CPU\" online_cpus=$ONLINE usable_cpus=$USABLE affinity=$AFF cgroup_quota_cpus=$QUOTA numa_nodes=$NODES governor=$GOV loadavg=\"$LOAD\" server_cpus=${SRV_CPUS:-unpinned} server_nodes=${SN:-?} gen_cpus=${GEN_CPUS:-unpinned} gen_nodes=${GN:-?}"

if [ -n "$SRV_CPUS" ] && [ -n "$GEN_CPUS" ]; then
    both=$(comm -12 <(expand "$SRV_CPUS" | sort -n | uniq | sort) <(expand "$GEN_CPUS" | sort -n | uniq | sort) | wc -l)
    [ "$both" -gt 0 ] && echo "WARNING the server and the load generator share $both CPU(s): the generator's load lands on the server's cores"
fi
if [ -n "$SRV_CPUS" ] && [ -n "$SN" ] && [ "$GNODE" != "?" ] && [ "${GNODE#none}" = "$GNODE" ]; then
    case ",$SN," in *",$GNODE,"*) ;; *) echo "WARNING the server CPUs (node $SN) are not on the GPU's NUMA node $GNODE" ;; esac
fi
if [ "$QUOTA" != none ] && [ "$USABLE" != "?" ]; then
    awk -v q="$QUOTA" -v u="$USABLE" 'BEGIN{exit !(q + 0.5 < u)}' && \
        echo "WARNING the cgroup quota ($QUOTA CPUs) is below the affinity count ($USABLE): threads sized from nproc will be throttled"
fi
exit 0
