#!/bin/sh
# box_doctor.sh — what this machine IS, before anything is measured.
#
# The first command to run on a new box: before the build, before
# box_qualify.sh, before any number is quoted. It is READ-ONLY and it never
# measures throughput. It describes the machine, asks the BINARY what it will
# actually run on it, names the things that would invalidate a measurement, and
# PROPOSES candidate W x T topologies for the sweep docs/serving.md §3
# describes. It proposes; the sweep decides. Nothing here is a recommendation
# and nothing here is a capacity.
#
# Usage:
#   tools/bench/box_doctor.sh [--bin ./mynah-asr] [-o ~/asr-evidence]
#                             [--no-evidence] [--allow-load]
#
# Every line is a fact with its source named, or the word UNAVAILABLE with the
# reason it cannot be read here. Machine-readable twins are
# "[DOCTOR] v=1 key=value ...", in the style of [SERVER-CONFIG] and [TOPOLOGY]
# (server/obs.c, server/prefork.c): one field is one token, a space inside a
# value becomes '_', an empty value is '-'.
#
# Rehearsal: MYNAH_ASR_CPU_TOPOLOGY_ROOT, MYNAH_ASR_CGROUP_ROOT and
# MYNAH_ASR_DOCTOR_PROC_ROOT point the Linux reads at a fabricated tree, the
# way server/prefork.c's own macro does, so the sysfs parsing can be executed on
# a machine that has no /sys. A run with any of them set announces itself as a
# REHEARSAL and describes the tree, not the host.
#
# Exit: 0 described · 2 usage · 3 REFUSED — the box is not idle enough to
# measure on. The description printed above a refusal is still valid; what is
# refused is the measurement that would have followed.
set -u

BIN=./mynah-asr
OUT="$HOME/asr-evidence"
EVIDENCE=1
ALLOW_LOAD=0

# The one threshold this script invents. 2.0 is not chosen on its own merits:
# it is exactly tools/bench/box_qualify.sh's threshold, so that a box the
# doctor calls idle is a box the qualifier will accept. Like that one it is
# ABSOLUTE, not per-cpu — conservative on a 64-core box, which is the safe
# direction for a gate whose whole job is to refuse.
LOAD_MAX=2.0
# The three filesystem roots this script reads. They are variables and not
# literals for the reason server/prefork.c gives its own MYNAH_ASR_CPU_TOPOLOGY_
# ROOT macro: the Linux-only parsing here is the part complicated enough to be
# wrong, and on a machine with no /sys it would otherwise ship having never been
# executed. The first two names are the server's own; a run with any of them
# overridden is announced as a REHEARSAL and is not a report about this host.
SYSCPU=${MYNAH_ASR_CPU_TOPOLOGY_ROOT:-/sys/devices/system/cpu}
CGROOT=${MYNAH_ASR_CGROUP_ROOT:-/sys/fs/cgroup}
PROCFS=${MYNAH_ASR_DOCTOR_PROC_ROOT:-/proc}
# A round number well above what C100 needs (100 client sockets + listeners +
# metrics + model files, .work/fleet-observability.md §4 item 2), chosen so the
# warning fires long before the ceiling bites rather than at it.
NOFILE_MIN=4096
# The C100 precondition names listen(srv, 64) as too small for 100 clients
# arriving together; the kernel clamps the backlog to somaxconn, so somaxconn
# must have room above the fan-in, not at it.
SOMAXCONN_MIN=512
# .work/fleet-observability.md §4 item 4, quoted, not recomputed here.
PER_STREAM_MB=14

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN="$2"; shift 2 ;;
        -o) OUT="$2"; shift 2 ;;
        --no-evidence) EVIDENCE=0; shift ;;
        --allow-load) ALLOW_LOAD=1; shift ;;
        -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

OS=$(uname -s)
# Linux-SHAPED, which is not the same as "uname says Linux": the roots above can
# point at a fabricated tree, and then this path runs on a dev machine. That is
# the only way the sysfs parsing is ever executed before it reaches a real box.
if [ -r "$PROCFS/cpuinfo" ] && [ -d "$SYSCPU" ]; then LINUXFS=1; else LINUXFS=0; fi
REHEARSAL=no
[ "$SYSCPU" = /sys/devices/system/cpu ] || REHEARSAL=yes
[ "$CGROOT" = /sys/fs/cgroup ] || REHEARSAL=yes
[ "$PROCFS" = /proc ] || REHEARSAL=yes
if [ "$EVIDENCE" = 1 ]; then
    RUN="$OUT/doctor-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$RUN" || exit 2
    LOG="$RUN/doctor.log"
else
    RUN=""
    LOG=/dev/null
fi

WARN=0; NA=0; REFUSE=""
say()  { echo "$@" | tee -a "$LOG"; }
kv()   { echo "[DOCTOR] v=1 $*" | tee -a "$LOG"; }
warn() { WARN=$(( WARN + 1 )); say "  WARNING      $*"; }
na()   { NA=$(( NA + 1 )); say "  UNAVAILABLE  $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
rd()   { if [ -r "$1" ]; then head -n1 "$1" 2>/dev/null; fi; }
sc()   { sysctl -n "$1" 2>/dev/null; }
# One field, one token: whitespace becomes '_' and an empty value becomes '-',
# the same discipline topo_token() applies in server/prefork.c so that a
# [DOCTOR] line stays parseable by splitting on spaces.
tok() {
    v=$(printf '%s' "${1:-}" | tr '\n\t ' '___' | tr -d '\r')
    [ -n "$v" ] || v="-"
    printf '%s' "$v"
}
# "0-3,8,12-15" or "0-7:2" -> one cpu id per line, ascending as the kernel
# enumerates them (which is the order server/prefork.c plans on).
expand_list() {
    printf '%s' "${1:-}" | tr ',' '\n' | awk '
        $0 == "" { next }
        { n = split($0, p, ":"); step = (n > 1 ? p[2] : 1)
          m = split(p[1], q, "-")
          if (m == 1) { print q[1]; next }
          for (i = q[1]; i <= q[2]; i += step) print i }'
}
count_lines() { wc -l | tr -d ' '; }

say "box_doctor — read-only. It describes this machine and proposes; it measures nothing."
say "             docs/serving.md is the procedure; tools/bench/box_qualify.sh is the gate."
if [ "$REHEARSAL" = yes ]; then
    say ""
    say "  *** REHEARSAL: a filesystem root was overridden, so the facts below describe"
    say "      the fabricated tree and NOT this host. Roots in force:"
    say "        cpu topology  $SYSCPU"
    say "        cgroup        $CGROOT"
    say "        proc          $PROCFS"
    say "      Nothing printed under a rehearsal may be quoted about a machine. ***"
fi
[ -n "$RUN" ] && say "             evidence -> $RUN"
say ""

REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git status --porcelain 2>/dev/null | count_lines)
KERNEL=$(uname -r)
ARCH=$(uname -m)
say "0. HOST"
say "  os               $OS $KERNEL $ARCH            (uname -srm)"
say "  tree             commit $REV, $DIRTY file(s) dirty   (git)"
kv "section=host os=$(tok "$OS") kernel=$(tok "$KERNEL") arch=$(tok "$ARCH")" \
   "host=$(tok "$(hostname 2>/dev/null)") utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
   "commit=$(tok "$REV") dirty=$DIRTY"
if [ "$OS" != Linux ] && [ "$REHEARSAL" = no ]; then
    say "  note             production is Linux x86-64 / ARM64 (CLAUDE.md); this host's"
    say "                   numbers are development signals, and every /proc and /sys"
    say "                   fact below is marked UNAVAILABLE rather than guessed."
fi
say ""

# =========================================================== 1. the cpu
say "1. CPU — what the silicon is"
NCPU_ONLINE=""; NCPU_ALLOWED=""; CORES=""; SOCKETS=""; SMT=""
CPU_MODEL=""; ALLOWED_LIST=""; ONLINE_LIST=""; ORDER=""; CORE_MAJOR=no
if [ "$LINUXFS" = 1 ]; then
    CPU_MODEL=$(lscpu 2>/dev/null | sed -n 's/^ *Model name: *//p' | head -1)
    [ -n "$CPU_MODEL" ] || CPU_MODEL=$(sed -n 's/^model name[[:blank:]]*: *//p' "$PROCFS/cpuinfo" 2>/dev/null | head -1)
    [ -n "$CPU_MODEL" ] || CPU_MODEL=$(lscpu 2>/dev/null | sed -n 's/^ *BIOS Model name: *//p' | head -1)
    ONLINE_LIST=$(rd "$SYSCPU/online")
    NCPU_ONLINE=$(expand_list "$ONLINE_LIST" | count_lines)
    ALLOWED_LIST=$(sed -n 's/^Cpus_allowed_list:[[:blank:]]*//p' "$PROCFS/self/status" 2>/dev/null)
    [ -n "$ALLOWED_LIST" ] || ALLOWED_LIST="$ONLINE_LIST"
    NCPU_ALLOWED=$(expand_list "$ALLOWED_LIST" | count_lines)
    SOCKETS=$(lscpu 2>/dev/null | sed -n 's/^ *Socket(s): *//p' | head -1)
    # (package, core) pairs over the ALLOWED cpus: the same pair server/prefork.c
    # counts, so this number and --prefork-plan's cannot disagree.
    TOPO=""
    for c in $(expand_list "$ALLOWED_LIST"); do
        p=$(rd "$SYSCPU/cpu$c/topology/physical_package_id")
        k=$(rd "$SYSCPU/cpu$c/topology/core_id")
        [ -n "$p" ] && [ -n "$k" ] || { TOPO=""; break; }
        TOPO="$TOPO$c $p $k
"
    done
    if [ -n "$TOPO" ]; then
        CORES=$(printf '%s' "$TOPO" | awk '{print $2" "$3}' | sort -u | count_lines)
        SOCKETS=$(printf '%s' "$TOPO" | awk '{print $2}' | sort -u | count_lines)
        ORDER=$(printf '%s' "$TOPO" | awk '
            { c[NR]=$1; p[NR]=$2; k[NR]=$3 }
            END { for (i = 1; i <= NR; i++) {
                     key = p[i] SUBSEP k[i]
                     if (seen[key]++) continue
                     printf "%s%s", (first++ ? "," : ""), c[i]
                     for (j = i+1; j <= NR; j++)
                         if (p[j] == p[i] && k[j] == k[i]) printf ",%s", c[j] }
                  printf "\n" }')
        CORE_MAJOR=yes
        [ "$CORES" -gt 0 ] && [ $(( NCPU_ALLOWED % CORES )) -eq 0 ] && SMT=$(( NCPU_ALLOWED / CORES ))
    fi
    say "  model            ${CPU_MODEL:-unknown}   (lscpu 'Model name')"
    say "  logical cpus     $NCPU_ONLINE online, $NCPU_ALLOWED allowed to this process"
    say "                   ($SYSCPU/online, $PROCFS/self/status Cpus_allowed_list)"
    if [ -n "$CORES" ]; then
        say "  physical cores   $CORES over ${SOCKETS:-?} socket(s)   (topology/{physical_package_id,core_id})"
    else
        na "physical core count: sysfs topology unreadable. Without it a 'per-core'"
        say "                   number cannot be told from a 'per-thread' one."
    fi
    SMT_ACTIVE=$(rd "$SYSCPU/smt/active")
    SMT_CONTROL=$(rd "$SYSCPU/smt/control")
    if [ -n "$SMT_CONTROL" ]; then
        say "  SMT              control=$SMT_CONTROL active=${SMT_ACTIVE:-?} factor=${SMT:-unknown}"
        say "                   ($SYSCPU/smt/{control,active})"
    elif [ -n "$SMT" ]; then
        say "  SMT              factor $SMT (cpus/cores); $SYSCPU/smt absent"
    else
        na "SMT state: neither $SYSCPU/smt nor a whole-core mask."
    fi
    if [ -n "${SMT:-}" ] && [ "$SMT" -gt 1 ]; then
        warn "SMT is on ($SMT threads per core). Two threads of one core share its"
        say "               execution units: a throughput number that does not say whether"
        say "               SMT was on is not comparable with one that does, and a slice"
        say "               whose T is not a multiple of $SMT splits a core across workers."
    fi
    SIB=$(for c in $(expand_list "$ALLOWED_LIST"); do
              rd "$SYSCPU/cpu$c/topology/thread_siblings_list"
          done | sort -u -t, -k1,1n | tr '\n' ' ')
    if [ -n "$SIB" ]; then
        say "  sibling groups   $SIB"
        say "                   (cpu*/topology/thread_siblings_list — the pairs that must"
        say "                    stay inside ONE worker's mask)"
    else
        na "thread_siblings_list: cannot say which logical cpus share a core."
    fi
    ALLOWED_SET=" $(expand_list "$ALLOWED_LIST" | tr '\n' ' ')"
    SPLIT=0
    for c in $(expand_list "$ALLOWED_LIST"); do
        for sb in $(expand_list "$(rd "$SYSCPU/cpu$c/topology/thread_siblings_list")"); do
            case "$ALLOWED_SET" in *" $sb "*) : ;; *) SPLIT=$(( SPLIT + 1 )) ;; esac
        done
    done
    if [ "$SPLIT" -gt 0 ]; then
        warn "$SPLIT sibling(s) of the cpus in this mask are OUTSIDE it. SMT is on, so"
        say "               the other thread of each of those cores can be given to another"
        say "               process: a worker pinned inside this mask shares execution units"
        say "               with something it cannot see, and 'pinned' does not mean"
        say "               'isolated'. Any per-core number here is an upper bound at best."
    fi
    if [ "$CORE_MAJOR" = yes ]; then
        say "  server cpu order core-major: $ORDER"
        say "                   (the order server/prefork.c order_core_major() builds, so a"
        say "                    worker's slice keeps SMT siblings together)"
    else
        na "core-major order: sysfs topology unreadable, so the server would fall back"
        say "                   to logical-id order and a contiguous slice may hand two"
        say "                   workers the two threads of the same physical core."
    fi
else
    CPU_MODEL=$(sc machdep.cpu.brand_string)
    NCPU_ONLINE=$(sc hw.logicalcpu)
    NCPU_ALLOWED="$NCPU_ONLINE"
    CORES=$(sc hw.physicalcpu)
    SOCKETS=1
    [ -n "$CORES" ] && [ "$CORES" -gt 0 ] && [ $(( NCPU_ONLINE % CORES )) -eq 0 ] \
        && SMT=$(( NCPU_ONLINE / CORES ))
    say "  model            ${CPU_MODEL:-unknown}   (sysctl machdep.cpu.brand_string)"
    say "  logical cpus     $NCPU_ONLINE, physical cores $CORES, smt factor ${SMT:-?}"
    say "                   (sysctl hw.logicalcpu / hw.physicalcpu)"
    NPERF=$(sc hw.nperflevels)
    if [ -n "$NPERF" ] && [ "$NPERF" -gt 1 ]; then
        i=0
        while [ "$i" -lt "$NPERF" ]; do
            say "  perflevel $i      $(sc hw.perflevel$i.physicalcpu) physical / $(sc hw.perflevel$i.logicalcpu) logical, l2 $(sc hw.perflevel$i.l2cachesize) B"
            i=$(( i + 1 ))
        done
        warn "asymmetric cores (P and E): a per-core number depends on which cluster"
        say "               the thread landed on, and nothing here pins it — on this platform"
        say "               server/prefork.c prints pinning as UNAVAILABLE."
    fi
    na "thread_siblings_list, /sys topology, core-major order: no /sys on $OS."
    say "                   server/prefork.c falls back to logical-id order here, and"
    say "                   sched_setaffinity does not exist: workers float."
fi
kv "section=cpu model=$(tok "$CPU_MODEL") arch=$(tok "$ARCH")" \
   "logical_online=$(tok "$NCPU_ONLINE") logical_allowed=$(tok "$NCPU_ALLOWED")" \
   "physical_cores=$(tok "$CORES") sockets=$(tok "$SOCKETS") smt=$(tok "${SMT:-}")" \
   "core_major=$(tok "$CORE_MAJOR") order=$(tok "$ORDER")"

# ---- caches ----------------------------------------------------------------
if [ "$LINUXFS" = 1 ] && [ -d "$SYSCPU/cpu0/cache" ]; then
    for idx in "$SYSCPU"/cpu0/cache/index*; do
        [ -d "$idx" ] || continue
        lvl=$(rd "$idx/level"); typ=$(rd "$idx/type"); sz=$(rd "$idx/size")
        shl=$(rd "$idx/shared_cpu_list")
        nsh=$(expand_list "$shl" | count_lines)
        scope="shared by $nsh cpus ($shl)"
        [ "$nsh" = 1 ] && scope="private to this cpu"
        if [ -n "$NCPU_ONLINE" ] && [ "$nsh" != "$NCPU_ONLINE" ] && [ "${lvl:-0}" = 3 ]; then
            scope="$scope — PER-CLUSTER, not one box-wide L3"
        fi
        say "  L$lvl $typ        ${sz:-?}, $scope"
        kv "section=cache level=$(tok "$lvl") type=$(tok "$typ") size=$(tok "$sz")" \
           "shared_cpus=$nsh shared_cpu_list=$(tok "$shl")"
    done
    say "                   (cpu0/cache/index*/{level,type,size,shared_cpu_list})"
    # A hybrid box reports different L1d sizes on different cpus; a single
    # per-core number taken on one of them does not describe the other.
    L1SIZES=$(for c in $(expand_list "$ALLOWED_LIST"); do
                  rd "$SYSCPU/cpu$c/cache/index0/size"
              done | sort -u | count_lines)
    if [ "${L1SIZES:-1}" -gt 1 ]; then
        warn "cpus do not all have the same L1d size: this is a hybrid machine."
        say "               A stream's RTF then depends on which core it landed on, so an"
        say "               unpinned sweep on this box compares two different CPUs."
    fi
elif [ "$LINUXFS" = 1 ]; then
    na "cache geometry: $SYSCPU/cpu0/cache is absent."
else
    L1D=$(sc hw.l1dcachesize); L2=$(sc hw.l2cachesize); L3=$(sc hw.l3cachesize)
    say "  caches           l1d=${L1D:-?} B l2=${L2:-?} B l3=${L3:-not reported by sysctl} B"
    say "                   (sysctl hw.l1dcachesize/hw.l2cachesize/hw.l3cachesize)"
    kv "section=cache source=sysctl l1d_bytes=$(tok "$L1D") l2_bytes=$(tok "$L2")" \
       "l3_bytes=$(tok "$L3") shared=unavailable"
    na "cache sharing scope: $OS reports sizes, not shared_cpu_list. Whether the"
    say "                   last level is shared or per-cluster cannot be read here."
fi

# ---- the ISA features this runtime actually dispatches on ------------------
say ""
say "  ISA features that this runtime dispatches on (src/dispatch.c):"
FEATSRC="-"
case "$ARCH" in
    aarch64|arm64) FEATS="asimddp i8mm bf16 sve sve2" ;;
    x86_64|amd64)  FEATS="avx2 avx512f avx512vnni avx_vnni" ;;
    *)             FEATS="" ;;
esac
if [ "$LINUXFS" = 1 ]; then
    FEATSRC="$PROCFS/cpuinfo"
    CPUFLAGS=$(sed -n -e 's/^flags[[:blank:]]*: *//p' -e 's/^Features[[:blank:]]*: *//p' \
                      "$PROCFS/cpuinfo" | head -1)
    CPUINFO_KIND=unknown
    grep -q '^flags' "$PROCFS/cpuinfo" 2>/dev/null && CPUINFO_KIND=x86
    grep -q '^Features' "$PROCFS/cpuinfo" 2>/dev/null && CPUINFO_KIND=arm
    UNAME_KIND=other
    case "$ARCH" in aarch64|arm64) UNAME_KIND=arm ;; x86_64|amd64) UNAME_KIND=x86 ;; esac
    if [ "$CPUINFO_KIND" != unknown ] && [ "$CPUINFO_KIND" != "$UNAME_KIND" ]; then
        warn "$FEATSRC describes a $CPUINFO_KIND cpu while uname -m says $ARCH."
        say "               One of the two is not this machine (a fabricated root, a chroot"
        say "               or an image from another architecture). The feature list below"
        say "               was chosen from uname, so every ISA line here is unattributable."
    fi
    say "    (feature set chosen from uname -m = $ARCH; presence read from $FEATSRC)"
    for f in $FEATS; do
        case " $CPUFLAGS " in
            *" $f "*) st=yes ;;
            *) st=no ;;
        esac
        say "    $f: $st   ($FEATSRC $( [ "$ARCH" = x86_64 ] && echo flags || echo Features ))"
        kv "section=isa feature=$f present=$st source=$(tok "$FEATSRC")"
    done
    if [ "$UNAME_KIND" = x86 ]; then
        say "    Spelling differs by source and the two do not disagree: cpuinfo"
        say "    says avx_vnni where --dispatch-map's IDLE HARDWARE says avxvnni."
    fi
    [ -n "$CPUFLAGS" ] || na "$FEATSRC has no flags/Features line on this kernel."
elif [ "$ARCH" = arm64 ] && have sysctl; then
    FEATSRC=sysctl
    for f in $FEATS; do
        case "$f" in
            asimddp) key=hw.optional.arm.FEAT_DotProd ;;
            i8mm)    key=hw.optional.arm.FEAT_I8MM ;;
            bf16)    key=hw.optional.arm.FEAT_BF16 ;;
            sve)     key=hw.optional.arm.FEAT_SVE ;;
            sve2)    key=hw.optional.arm.FEAT_SVE2 ;;
            *)       key="" ;;
        esac
        v=$(sc "$key")
        case "${v:-}" in 1) st=yes ;; 0) st=no ;; *) st=unknown ;; esac
        say "    $f: $st   (sysctl $key — the same sysctl src/dispatch.c reads)"
        kv "section=isa feature=$f present=$st source=$(tok "sysctl $key")"
    done
else
    na "ISA features: no $PROCFS/cpuinfo and no known sysctl for $ARCH on $OS."
fi
say "    These are the CPU's capabilities, not the binary's. Section 2 is what"
say "    the binary does with them, and section 2 is the one that decides."
say ""

# ============================================ 2. what the binary will do here
say "2. BINARY — what it will actually run on this machine"
DMAP=""
if [ -x "$BIN" ]; then
    DMAP=$("$BIN" --dispatch-map 2>&1)
    [ -n "$RUN" ] && { printf '%s\n' "$DMAP" > "$RUN/dispatch.txt"; \
                       "$BIN" --flags > "$RUN/flags.txt" 2>&1; }
    printf '%s\n' "$DMAP" | sed -n -e '1p' -e '/^kernel\./p' -e '/^gemm\./p' \
                                  -e '/^backend\./p' -e '/^threads\./p' \
        | sed 's/^/  /' | cut -c1-120 | tee -a "$LOG"
    printf '%s\n' "$DMAP" | awk '
        /^(kernel|gemm|backend|threads)\./ {
            print "[DOCTOR] v=1 section=dispatch feature=" $1 " resolved=" $5 }' \
        | tee -a "$LOG"
    UNKNOWN=$(printf '%s\n' "$DMAP" | sed -n 's/^\([0-9][0-9]*\) row(s) UNKNOWN.*/\1/p' | tail -1)
    UNKNOWN=${UNKNOWN:-0}
    if [ "$UNKNOWN" != 0 ]; then
        warn "$UNKNOWN dispatch row(s) UNKNOWN: the binary cannot say which kernel it"
        say "               runs. Every number taken on this build is unattributable"
        say "               (ENGINEERING.md §5) — fix the build before measuring."
    fi
    # src/dispatch.c writes exactly "present, USED" / "present, IDLE" /
    # "absent" / "unknown". IDLE is silicon this binary never issues: not an
    # error, but it caps every number taken below, and a rebuild moves it.
    IDLE=$(printf '%s\n' "$DMAP" | awk '/IDLE HARDWARE/ {on=1; next}
                                         on && /present, IDLE/ {print $1}')
    IDLEU=$(printf '%s\n' "$DMAP" | awk '/IDLE HARDWARE/ {on=1; next}
                                          on && $2 == "unknown" {print $1}')
    IDLEN=$(printf '%s\n' "$IDLE" | sed '/^$/d' | count_lines)
    if [ -n "$IDLE" ]; then
        warn "IDLE HARDWARE: $(echo "$IDLE" | tr '\n' ' ')present on this CPU and never"
        say "               issued by this binary. A throughput number measured here is a"
        say "               number for THIS build, not for this machine, and a rebuild that"
        say "               lights those rows up invalidates the comparison."
    fi
    if [ -n "$IDLEU" ]; then
        warn "the binary cannot determine whether this CPU has: $(echo "$IDLEU" | tr '\n' ' ')"
        say "               (--dispatch-map prints 'unknown', not 'absent'). A report that"
        say "               says 'this box has no <feature>' is then a guess; say unknown."
    fi
    printf '%s\n' "$DMAP" | sed -n '/IDLE HARDWARE/,$p' | sed 's/^/  /' | tee -a "$LOG" >/dev/null
    kv "section=dispatch unknown_rows=$UNKNOWN idle_present_unused=${IDLEN:-0}" \
       "build=$(tok "$(printf '%s\n' "$DMAP" | sed -n 's/.*build=\([^ ]*\).*/\1/p' | head -1)")" \
       "blas=$(tok "$(printf '%s\n' "$DMAP" | sed -n 's/.*blas=\([^ ]*\).*/\1/p' | head -1)")" \
       "simd=$(tok "$(printf '%s\n' "$DMAP" | sed -n 's/.*simd=\([^ ]*\).*/\1/p' | head -1)")"
else
    na "$BIN is not executable: run 'make' first. A doctor that does not ask the"
    say "                   binary is reading a spec sheet — section 1 says what the CPU"
    say "                   has, only --dispatch-map says what this build issues."
    kv "section=dispatch unknown_rows=unavailable idle_present_unused=unavailable binary=$(tok "$BIN")"
fi
say ""

# ==================================================== 3. memory and its limit
say "3. MEMORY"
MEM_TOTAL_MB=""; MEM_AVAIL_MB=""; HP_TOTAL=""; HP_FREE=""; HP_SIZE=""
if [ "$LINUXFS" = 1 ] && [ -r "$PROCFS/meminfo" ]; then
    MEM_TOTAL_MB=$(awk '/^MemTotal:/{printf "%d", $2/1024}' "$PROCFS/meminfo")
    MEM_AVAIL_MB=$(awk '/^MemAvailable:/{printf "%d", $2/1024}' "$PROCFS/meminfo")
    HP_TOTAL=$(awk '/^HugePages_Total:/{print $2}' "$PROCFS/meminfo")
    HP_FREE=$(awk '/^HugePages_Free:/{print $2}' "$PROCFS/meminfo")
    HP_SIZE=$(awk '/^Hugepagesize:/{print $2}' "$PROCFS/meminfo")
    THP=$(rd /sys/kernel/mm/transparent_hugepage/enabled)
    say "  total            $MEM_TOTAL_MB MB, available $MEM_AVAIL_MB MB   ("$PROCFS/meminfo" MemTotal/MemAvailable)"
    say "  hugepages        ${HP_TOTAL:-0} reserved, ${HP_FREE:-0} free, ${HP_SIZE:-?} kB each; THP: ${THP:-unreadable}"
    say "                   ("$PROCFS/meminfo", /sys/kernel/mm/transparent_hugepage/enabled)"
else
    MEMB=$(sc hw.memsize)
    [ -n "$MEMB" ] && MEM_TOTAL_MB=$(( MEMB / 1048576 ))
    PGS=$(sc hw.pagesize)
    if have vm_stat && [ -n "$PGS" ]; then
        MEM_AVAIL_MB=$(vm_stat | awk -v p="$PGS" '
            /Pages free/          {f = $3 + 0}
            /Pages inactive/      {i = $3 + 0}
            /Pages speculative/   {s = $3 + 0}
            END { printf "%d", (f + i + s) * p / 1048576 }')
    fi
    say "  total            ${MEM_TOTAL_MB:-?} MB   (sysctl hw.memsize)"
    say "  available        ${MEM_AVAIL_MB:-?} MB   (vm_stat free+inactive+speculative — an"
    say "                   approximation, NOT the same quantity as Linux MemAvailable)"
    na "hugepages: no "$PROCFS/meminfo" on $OS; $OS has no equivalent knob to report."
fi
say "  per stream       ~$PER_STREAM_MB MB — KV cache 24x56x1024x4x2 ~11 MB, conv cache"
say "                   0.8 MB, decoder scratch 32x(640+13088)x4 ~1.7 MB, plus the PCM"
say "                   ring. Quoted from .work/fleet-observability.md §4 item 4, not"
say "                   recomputed here; the model weights are COW-shared across workers."
if [ -n "$MEM_AVAIL_MB" ]; then
    BYMEM=$(( MEM_AVAIL_MB / PER_STREAM_MB ))
    say "  arithmetic bound $MEM_AVAIL_MB MB / $PER_STREAM_MB MB = $BYMEM streams before per-stream state alone"
    say "                   exhausts available memory. This is a CEILING for a sanity check,"
    say "                   never a capacity: cpu, not memory, is what the sweep measures."
fi
kv "section=memory total_mb=$(tok "$MEM_TOTAL_MB") available_mb=$(tok "$MEM_AVAIL_MB")" \
   "hugepages_total=$(tok "$HP_TOTAL") hugepages_free=$(tok "$HP_FREE")" \
   "hugepage_kb=$(tok "$HP_SIZE") per_stream_mb=$PER_STREAM_MB" \
   "streams_by_memory=$(tok "${BYMEM:-}") per_stream_source=.work/fleet-observability.md#4.4"
say ""

# ============================================ 4. what would invalidate a run
say "4. NOISE — what would invalidate a measurement taken now"
if [ -r "$PROCFS/loadavg" ]; then
    LOAD=$(cut -d' ' -f1 "$PROCFS/loadavg"); LOAD5=$(cut -d' ' -f2 "$PROCFS/loadavg")
    LOADSRC="$PROCFS/loadavg"
else
    LOAD=$(sc vm.loadavg | tr -d '{}' | awk '{print $1}')
    LOAD5=$(sc vm.loadavg | tr -d '{}' | awk '{print $2}')
    LOADSRC="sysctl vm.loadavg"
fi
case "${LOAD:-}" in ''|*[!0-9.]*) LOAD="" ;; esac
if [ -z "$LOAD" ]; then
    LOAD=unknown
    na "loadavg: neither /proc/loadavg nor sysctl vm.loadavg answered."
    warn "the §10 refusal cannot be enforced on this host: nothing here can say"
    say "               whether another load owns the box, so 'the box was idle' is a"
    say "               claim no artefact backs. An unreadable loadavg is not a zero."
else
    say "  loadavg          $LOAD (1m), ${LOAD5:-?} (5m)   ($LOADSRC)"
fi
if [ "$LOAD" != unknown ] && awk -v l="$LOAD" -v m="$LOAD_MAX" 'BEGIN { exit !(l >= m) }'; then
    if [ "$ALLOW_LOAD" = 1 ]; then
        warn "loadavg $LOAD >= $LOAD_MAX and --allow-load was given. Anything measured"
        say "               while another load owns this box is DIAGNOSTIC and never"
        say "               qualifies (ENGINEERING.md §10)."
    else
        REFUSE="loadavg $LOAD >= $LOAD_MAX: another load owns this box. Every cadence"
        say "  REFUSED      loadavg $LOAD >= $LOAD_MAX — another load owns this box."
        say "               A WAVE or SOAK started now measures this machine plus whatever"
        say "               else is on it, and the percentiles cannot be separated after"
        say "               the fact (ENGINEERING.md §10). Find the load, or --allow-load"
        say "               to describe the box anyway and mark the run DIAGNOSTIC."
    fi
fi
GOV=""; DRV=""
if [ "$LINUXFS" = 1 ]; then
    GOV=$(rd "$SYSCPU/cpu0/cpufreq/scaling_governor")
    DRV=$(rd "$SYSCPU/cpu0/cpufreq/scaling_driver")
    if [ -n "$GOV" ]; then
        say "  cpufreq          governor=$GOV driver=$DRV   ($SYSCPU/cpu0/cpufreq/scaling_*)"
        NGOV=$(cat "$SYSCPU"/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | count_lines)
        [ "${NGOV:-1}" -gt 1 ] && warn "cpus do not share one governor: per-worker speed differs by slice."
        if [ "$GOV" != performance ]; then
            warn "governor is '$GOV', not 'performance'. A ramping clock makes the first"
            say "               window of a soak slower than the tenth, which the drift gate"
            say "               reads as degradation that is really the governor."
        fi
    else
        na "cpufreq governor: no cpu0/cpufreq (common on cloud ARM and in VMs, where"
        say "                   the hypervisor owns the clock). Frequency stability is then"
        say "                   unknowable from inside; say so in the report."
    fi
    THR=$(cat "$SYSCPU"/cpu*/thermal_throttle/core_throttle_count 2>/dev/null \
          | awk '{s += $1} END {print s+0}')
    if [ -n "$THR" ] && [ "$THR" != 0 ]; then
        warn "core_throttle_count totals $THR since boot: this box has thermally"
        say "               capped before. A long SOAK may lose frequency mid-run and the"
        say "               drift gate will blame the server."
    fi
    TZ=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
    if [ -n "$TZ" ]; then
        [ "$TZ" -ge 1000 ] && TZ=$(( TZ / 1000 ))   # millidegrees on most zones
        say "  temperature      $TZ C hottest zone   (/sys/class/thermal/thermal_zone*/temp)"
    fi
    VIRT=$(systemd-detect-virt 2>/dev/null)
    [ -n "$VIRT" ] || VIRT=$(rd /sys/class/dmi/id/sys_vendor)
    say "  virtualisation   ${VIRT:-unknown}   (systemd-detect-virt, else DMI sys_vendor)"
    case "${VIRT:-}" in
        ""|none|unknown) : ;;
        *) warn "this looks like a VM ($VIRT): steal time and a shared last-level cache"
           say "               are outside this process's view, so a p95 here includes a"
           say "               neighbour you cannot see or name in the report." ;;
    esac
    CGQ=$(rd "$CGROOT/cpu.max")
    [ -n "$CGQ" ] || CGQ=$(rd "$CGROOT/cpu/cpu.cfs_quota_us")
    say "  cgroup cpu quota ${CGQ:-none}   ($CGROOT/cpu.max)"
    case "${CGQ:-max}" in
        max|"max 100000"|-1|none) : ;;
        *) warn "a cgroup cpu quota is in force. The kernel will throttle the fleet at"
           say "               period boundaries; that arrives as a periodic latency spike and"
           say "               nproc will not mention it." ;;
    esac
    say "  affinity mask    $ALLOWED_LIST -> $NCPU_ALLOWED cpus; nproc says $(nproc 2>/dev/null || echo '?')"
    say "                   ($PROCFS/self/status Cpus_allowed_list — taskset -p \$\$ agrees)"
    if [ -n "$NCPU_ONLINE" ] && [ "$NCPU_ALLOWED" != "$NCPU_ONLINE" ]; then
        warn "this shell is confined to $NCPU_ALLOWED of $NCPU_ONLINE online cpus. Plan on the MASK:"
        say "               the server does (server/prefork.c cpus_allowed), and a W x T grid"
        say "               derived from $NCPU_ONLINE would oversubscribe by $(( NCPU_ONLINE / NCPU_ALLOWED ))x."
    fi
    FULL=$(rd "$SYSCPU/online")
    PINS=$(awk -v full="$FULL" '
        /^Name:/ { name = $2 }
        /^Cpus_allowed_list:/ { if ($2 != full) { n = split(FILENAME, f, "/"); print f[n-1] ":" name ":" $2 } }
    ' "$PROCFS"/[0-9]*/status 2>/dev/null)
    NPINS=$(printf '%s\n' "$PINS" | sed '/^$/d' | count_lines)
    if [ "${NPINS:-0}" -gt 0 ]; then
        say "  pinned processes ${NPINS} process(es) run on a subset of the cpus:"
        printf '%s\n' "$PINS" | head -8 | sed 's/^/                   /' | tee -a "$LOG"
        warn "something else already owns part of this machine. If one of those masks"
        say "               overlaps a worker slice, that worker is sharing cores and its"
        say "               slice is not the isolated slice the report will claim."
    fi
else
    na "cpu governor, scaling driver, thermal/frequency capping counters, cgroup"
    say "                   quota: no /sys on $OS. Frequency and thermal behaviour here are"
    say "                   not observable from a script and must not be asserted."
    VIRT=$(sc kern.hv_vmm_present)
    say "  virtualisation   kern.hv_vmm_present=${VIRT:-?}   (1 = running under a VMM)"
    na "inherited affinity mask: $OS has no cpu affinity API this runtime would use"
    say "                   (server/prefork.c says exactly this), so there is no mask to"
    say "                   read and workers cannot be pinned at all."
    warn "on $OS the fleet is UNPINNED: every W x T candidate in section 6 measures"
    say "               workers that float across all cores. Useful as a development"
    say "               signal, never as a qualification (CLAUDE.md: production is Linux)."
fi
kv "section=noise loadavg1=$(tok "$LOAD") loadavg5=$(tok "$LOAD5") load_max=$LOAD_MAX" \
   "governor=$(tok "$GOV") scaling_driver=$(tok "$DRV") virt=$(tok "${VIRT:-}")" \
   "cgroup_cpu=$(tok "${CGQ:-}") allowed_cpus=$(tok "$NCPU_ALLOWED")" \
   "online_cpus=$(tok "$NCPU_ONLINE") pinned_processes=$(tok "${NPINS:-unavailable}")"
say ""

# ==================================== 5. the limits that bite at concurrency
say "5. LIMITS — the ceilings that only appear at high concurrency"
NOFS=$(ulimit -n 2>/dev/null); NOFH=$(ulimit -Hn 2>/dev/null)
say "  ulimit -n        soft=$NOFS hard=$NOFH   (this shell; the server inherits it)"
say "                   Too small and the fleet stops accepting mid-run with EMFILE,"
say "                   which the harness records as errors, not as a refusal ladder:"
say "                   the run then measures the descriptor table (.work/fleet-"
say "                   observability.md §4 item 2)."
case "$NOFS" in
    unlimited) : ;;
    ''|*[!0-9]*) na "ulimit -n did not report a number." ;;
    *) [ "$NOFS" -lt "$NOFILE_MIN" ] && {
           warn "nofile soft=$NOFS < $NOFILE_MIN. At ~1 socket per stream plus listeners,"
           say "               metrics and model files, a C100 run can hit this ceiling and"
           say "               the errors will look like server failures." ; } ;;
esac
# lim <proc path> <label> <what its shortage causes>. The value comes back in
# LIMV, not on stdout: a $(lim ...) would capture this function's own human
# lines and store the failure-mode sentence in the variable.
lim() {
    LIMV=$(rd "$1")
    if [ -n "$LIMV" ]; then
        say "$(printf '  %-17s' "$2")$LIMV   ($1)"
        say "                   $3"
    else
        na "$2: $1 is absent on $OS. $3"
    fi
}
if [ "$LINUXFS" = 1 ]; then
    lim "$PROCFS"/sys/net/core/somaxconn "somaxconn" \
      "Below the fan-in the kernel silently drops SYNs at the listen queue: the clients retry with a 1s backoff and the run measures the accept queue, not the server."
    SOMAX="$LIMV"
    lim "$PROCFS"/sys/net/ipv4/tcp_max_syn_backlog "syn backlog" \
      "Same failure one stage earlier, for half-open connections: a burst of simultaneous WebSocket upgrades is exactly the shape that overflows it."
    SYNBL="$LIMV"
    lim "$PROCFS"/sys/net/core/rmem_max "rmem_max" \
      "The server's receive window ceiling. Too small and a client that sends audio in bursts blocks on the socket, which arrives as client-side pacing lateness and the harness marks the run INVALID."
    RMEM="$LIMV"
    lim "$PROCFS"/sys/net/core/wmem_max "wmem_max" \
      "The send-side twin: deltas queue in the worker instead of the socket, and the server's own lag_ms rises for a reason outside the model."
    WMEM="$LIMV"
    lim "$PROCFS"/sys/net/ipv4/ip_local_port_range "ephemeral ports" \
      "Bites the CLIENT side: a closed-loop soak that reconnects leaves sockets in TIME_WAIT, and when the range runs out the generator stalls and the stall is read as a server stall."
    PORTS="$LIMV"
    case "${SOMAX:-}" in
        ''|*[!0-9]*) : ;;
        *) [ "$SOMAX" -lt "$SOMAXCONN_MIN" ] && {
               warn "somaxconn=$SOMAX < $SOMAXCONN_MIN. server/main.c asks for a listen backlog"
               say "               that the kernel clamps to this value; at C100 the arrivals are"
               say "               simultaneous and the dropped SYNs are invisible in the server"
               say "               log (.work/fleet-observability.md §4 item 1)." ; } ;;
    esac
    NPORTS=$(printf '%s' "${PORTS:-}" | awk '{print $2 - $1}')
    kv "section=limits nofile_soft=$(tok "$NOFS") nofile_hard=$(tok "$NOFH")" \
       "somaxconn=$(tok "$SOMAX") tcp_max_syn_backlog=$(tok "$SYNBL")" \
       "rmem_max=$(tok "$RMEM") wmem_max=$(tok "$WMEM")" \
       "ephemeral_range=$(tok "$PORTS") ephemeral_count=$(tok "${NPORTS:-}")"
else
    SOMAX=$(sc kern.ipc.somaxconn)
    P1=$(sc net.inet.ip.portrange.first); P2=$(sc net.inet.ip.portrange.last)
    say "  somaxconn        ${SOMAX:-?}   (sysctl kern.ipc.somaxconn)"
    say "                   Below the fan-in the kernel drops SYNs and the run measures"
    say "                   the accept queue instead of the server."
    say "  ephemeral ports  ${P1:-?}-${P2:-?}   (sysctl net.inet.ip.portrange.first/last)"
    say "                   Bites the client side of a reconnecting soak, via TIME_WAIT."
    na "tcp_max_syn_backlog, rmem_max, wmem_max: these are Linux sysctls with no"
    say "                   one-to-one $OS equivalent. The production ceilings are not"
    say "                   knowable from this host."
    kv "section=limits nofile_soft=$(tok "$NOFS") nofile_hard=$(tok "$NOFH")" \
       "somaxconn=$(tok "$SOMAX") tcp_max_syn_backlog=unavailable rmem_max=unavailable" \
       "wmem_max=unavailable ephemeral_range=$(tok "${P1:-}-${P2:-}")"
fi
say ""

# ================================================= 6. the proposal, not a plan
say "6. PROPOSAL — candidate W x T topologies to SWEEP on this machine"
say "  These are CANDIDATES, not a recommendation. The sweep decides and nothing"
say "  else does: docs/serving.md §3 STEP 1 finds T* (the smallest pool width"
say "  within ~10% of the best single-stream RTF), STEP 2 sweeps W at constant"
say "  subscription W*T = cpus, STEP 3 raises concurrency until the verdict stops"
say "  being GOOD. ./mynah-asr-server --prefork-plan prints that procedure against"
say "  this host's own mask; this table only says which rows exist."
C=${NCPU_ALLOWED:-0}
case "$C" in ''|*[!0-9]*) C=0 ;; esac
if [ "$C" -lt 2 ]; then
    na "candidate grid: the allowed cpu count is unknown or 1; there is no grid."
else
    say "  Every row keeps W*T = $C, the whole allowed mask; only the SHAPE changes."
    say ""
    say "    W x T     what the row is testing"
    for T in 2 3 4 6 8 12 16; do
        [ $(( C % T )) -eq 0 ] || continue
        W=$(( C / T ))
        [ "$W" -ge 2 ] || continue
        shape=middle
        note1="the middle of the grid: neither extreme, and the row a sweep"
        note2="reaches last, once both ends have been seen"
        if [ "$T" -le 3 ]; then
            shape=narrow
            note1="many narrow workers: maximum isolation, the most processes and"
            note2="the most per-worker memory; tests whether a pool of $T still holds a stream real-time"
        elif [ "$T" -ge 8 ]; then
            shape=wide
            note1="few wide workers: tests whether the step still scales at width $T"
            note2="(the stream step is bandwidth bound, so it usually stops improving well below it)"
        fi
        say "$(printf '    %-9s %s' "${W}x${T}" "$note1")"
        say "              $note2"
        smt_clean=unknown
        if [ -n "${SMT:-}" ]; then
            if [ $(( T % SMT )) -eq 0 ]; then smt_clean=yes; else smt_clean=no; fi
        fi
        if [ "$smt_clean" = no ]; then
            say "              ^ T=$T is not a multiple of the SMT factor $SMT: the slice cuts a"
            say "                physical core in half, and two workers contend for it while"
            say "                taskset shows disjoint masks"
        fi
        if [ "$T" = 4 ]; then
            say "              ^ where tools/bench/box_qualify.sh starts when -W and -T are"
            say "                omitted (W = cpus/4, T = 4). That is where a sweep BEGINS,"
            say "                not where it ends: a starting point, not this box's answer."
        fi
        kv "section=candidate w=$W t=$T subscription=$(( W * T )) cpus=$C" \
           "shape=$shape smt_clean=$smt_clean" \
           "qualify_default=$( [ "$T" = 4 ] && echo yes || echo no )"
    done

    say ""
    say "  Fleet capacity for any row is W x C_cap (--cap is streams per WORKER), and"
    say "  section 3's memory bound is the only ceiling this script can compute. The"
    say "  admission ladder refuses past it with 503 server_at_capacity, which the"
    say "  harness counts as a REJECTION and not an error (docs/serving.md §3)."
fi
say ""

# ------------------------------------------------------------------ verdict
say "VERDICT"
say "  $WARN warning(s), $NA fact(s) unavailable on this platform."
kv "section=verdict warnings=$WARN unavailable=$NA rehearsal=$REHEARSAL" \
   "refused=$( [ -n "$REFUSE" ] && echo yes || echo no )" \
   "load_max=$LOAD_MAX nofile_min=$NOFILE_MIN somaxconn_min=$SOMAXCONN_MIN"
if [ -n "$RUN" ]; then
    say "  Evidence in $RUN (doctor.log, dispatch.txt, flags.txt)."
    if [ "$LINUXFS" = 1 ]; then
        (lscpu; echo; cat "$PROCFS/cpuinfo") > "$RUN/cpu.txt" 2>&1
        cp "$PROCFS/meminfo" "$RUN/meminfo.txt" 2>/dev/null
    else
        sysctl -a > "$RUN/sysctl.txt" 2>&1
    fi
fi
if [ -n "$REFUSE" ]; then
    say "  REFUSED: $REFUSE percentile taken now would include it."
    say "  The description above is still valid; the measurement that would have"
    say "  followed is not. Next step is to find the load, not to lower the bar."
    exit 3
fi
say "  Next: docs/serving.md §2 (build, then make the binary prove what it runs),"
say "  then §3 STEP 1 to find T*, then tools/bench/box_qualify.sh for the gate."
exit 0
