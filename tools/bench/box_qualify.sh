#!/bin/sh
# box_qualify.sh — the Linux serving qualification, as one command.
#
# ENGINEERING.md §5 (a benchmark is invalid until dispatch is proven), §10 (run
# lifecycle), §13 (what is committed is what was built) and docs/serving.md turned
# into a script, so that a qualifying run cannot skip a step by being typed by hand.
#
# Usage, from a clean checkout of the commit under test:
#   tools/bench/box_qualify.sh -m models/nemotron-3.5-asr-streaming-0.6b \
#       [-W workers] [-T threads] [-C cap] [-q int8] [-o ~/asr-evidence] \
#       [--wave "1 2 4 8 16"] [--soak 4] [--soak-seconds 600] [--allow-load]
#
# It refuses rather than produce a number it cannot support:
#   * a dirty tree            -> the run is labelled NON-QUALIFYING and continues
#   * loadavg >= 2.0          -> refuses (another load owns the box); --allow-load
#                                proceeds and marks the whole run NON-QUALIFYING
#   * a requested kernel that did not resolve  -> refuses (see --dispatch-map)
#   * an unpinned prefork on Linux             -> refuses
# Every phase writes into one directory with a manifest; nothing is printed that
# the artefacts do not contain.
set -u

MODEL=""; W=0; T=0; CAP=0; QUANT=int8; OUT="$HOME/asr-evidence"
WAVE="1 2 4 8"; SOAK_C=0; SOAK_S=600; PORT=8600; WINDOW=60; WARMUP=30; ALLOW_LOAD=0
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2 ;;
        -W) W="$2"; shift 2 ;;
        -T) T="$2"; shift 2 ;;
        -C) CAP="$2"; shift 2 ;;
        -q) QUANT="$2"; shift 2 ;;
        -o) OUT="$2"; shift 2 ;;
        -p) PORT="$2"; shift 2 ;;
        --wave) WAVE="$2"; shift 2 ;;
        --soak) SOAK_C="$2"; shift 2 ;;
        --soak-seconds) SOAK_S="$2"; shift 2 ;;
        --allow-load) ALLOW_LOAD=1; shift ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$MODEL" ] || { echo "box_qualify: -m <model_dir> is required" >&2; exit 2; }
[ -f "$MODEL/mynah.json" ] || { echo "box_qualify: $MODEL is not a converted model" >&2; exit 2; }
[ -x ./mynah-asr-server ] || { echo "box_qualify: build first (make)" >&2; exit 2; }

RUN="$OUT/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN" || exit 2
say() { echo "$@" | tee -a "$RUN/run.log"; }
die() { say "REFUSED: $*"; exit 3; }

# ---- 0. identity: what is being measured ---------------------------------
REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
BIN_SHA=$( (sha256sum ./mynah-asr-server 2>/dev/null || shasum -a 256 ./mynah-asr-server) | cut -d' ' -f1)
QUALIFY=yes
[ "$DIRTY" = "0" ] || { QUALIFY=no; say "WARNING dirty tree ($DIRTY files): this run is NON-QUALIFYING (ENGINEERING.md §13)"; }
say "box_qualify  commit=$REV dirty=$DIRTY binary=$BIN_SHA model=$MODEL quant=$QUANT"
say "             evidence -> $RUN"

# ---- 1. the machine ------------------------------------------------------
(uname -a; echo; lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null; echo;
 nproc 2>/dev/null; echo; free -g 2>/dev/null; cat /proc/loadavg 2>/dev/null) > "$RUN/host.txt" 2>&1
LOAD=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || uptime | sed 's/.*averages*: *//' | cut -d' ' -f1 | tr -d ,)
case "$LOAD" in ''|*[!0-9.]*) LOAD=0 ;; esac
if awk -v l="$LOAD" 'BEGIN { exit !(l >= 2.0) }'; then
    [ "$ALLOW_LOAD" = 1 ] || die "loadavg $LOAD >= 2.0, another load owns this box (ENGINEERING.md §10); --allow-load makes it a diagnostic"
    QUALIFY=no
    say "WARNING loadavg $LOAD >= 2.0 and --allow-load was given: this run is NON-QUALIFYING, DIAGNOSTIC only"
else
    say "host: $(nproc 2>/dev/null || echo ?) cpus, loadavg $LOAD -> idle enough"
fi

# ---- 2. what the binary will actually run --------------------------------
./mynah-asr --flags > "$RUN/flags.txt" 2>&1
./mynah-asr --dispatch-map > "$RUN/dispatch.txt" 2>&1
./mynah-asr --dispatch-map --json > "$RUN/dispatch.json" 2>&1
UNKNOWN=$(sed -n 's/^\([0-9][0-9]*\) row(s) UNKNOWN.*/\1/p' "$RUN/dispatch.txt" | tail -1)
[ "${UNKNOWN:-0}" = "0" ] || die "$UNKNOWN dispatch row(s) UNKNOWN: the binary cannot say what it runs"
say "dispatch: int8 kernel = $(sed -n 's/^kernel.int8_dot  *[^ ]*  *[^ ]*  *[^ ]*  *\([^ ]*\).*/\1/p' "$RUN/dispatch.txt" | head -1), blas = $(sed -n 's/.*blas=\([^ ]*\).*/\1/p' "$RUN/dispatch.txt" | head -1)"
grep -A8 "IDLE HARDWARE" "$RUN/dispatch.txt" | tee -a "$RUN/run.log" >/dev/null

# ---- 3. the topology plan ------------------------------------------------
./mynah-asr-server --prefork-plan > "$RUN/plan.txt" 2>&1
NCPU=$(nproc 2>/dev/null || echo 1)
[ "$W" -gt 0 ] || W=$(( NCPU / 4 )); [ "$W" -gt 0 ] || W=1
[ "$T" -gt 0 ] || T=$(( NCPU / W )); [ "$T" -gt 0 ] || T=1
[ "$CAP" -gt 0 ] || CAP=4
say "topology: $W workers x $T threads, cap $CAP (sweep it: docs/serving.md)"

# ---- 4. start the fleet, prove the masks ---------------------------------
./mynah-asr-server -m "$MODEL" --quant "$QUANT" -p "$PORT" \
    --prefork "$W" --prefork-threads "$T" --threads "$T" --cap "$CAP" \
    --metrics-port $(( PORT + 1000 )) > "$RUN/server.log" 2>&1 &
SRV=$!
i=0; while [ $i -lt 150 ]; do
    curl -sf -m 2 "http://localhost:$PORT/v1/health" >/dev/null 2>&1 && break
    sleep 0.2; i=$(( i + 1 ))
done
curl -sf -m 2 "http://localhost:$PORT/v1/health" >/dev/null 2>&1 || {
    cat "$RUN/server.log"; kill $SRV 2>/dev/null; die "the fleet never became ready"; }
grep -E "^\[(FLAGS|EFFECTIVE-CONFIG|SERVER-CONFIG|TOPOLOGY)\]" "$RUN/server.log" > "$RUN/banner.txt"
if [ "$(uname -s)" = "Linux" ]; then
    : > "$RUN/masks.txt"
    for pid in $(pgrep -P $SRV 2>/dev/null); do
        printf '%s %s\n' "$pid" "$(taskset -pc $pid 2>/dev/null | sed 's/.*: //')" >> "$RUN/masks.txt"
    done
    if [ ! -s "$RUN/masks.txt" ] || grep -q "0-$(( NCPU - 1 ))$" "$RUN/masks.txt"; then
        kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null
        die "workers are not pinned to disjoint slices (see $RUN/masks.txt): the throughput claim rests on pinning"
    fi
    say "masks: $(tr '\n' ' ' < "$RUN/masks.txt")"
fi

# ---- 5. WAVE screen: may disqualify, never promotes ----------------------
BANK="samples/en/fleurs_1521.wav samples/it/fleurs_1521.wav samples/fr/fleurs_1534.wav samples/de/fleurs_1534.wav samples/es/fleurs_1521.wav samples/en/fleurs_1534.wav samples/it/fleurs_1534.wav samples/fr/fleurs_1521.wav"
for C in $WAVE; do
    say "--- WAVE C=$C"
    python3 tools/bench/stream_load.py --mode wave --port "$PORT" --streams "$C" --repeat 2 \
        --clips $BANK --json "$RUN/wave-C$C.json" 2>&1 | tee -a "$RUN/run.log" | grep -E "TTFP|emission lag|finaliz|backlog|verdict|utterances|pacing"
done

# ---- 6. SOAK: the only thing that promotes ------------------------------
if [ "$SOAK_C" -gt 0 ]; then
    say "--- SOAK C=$SOAK_C for ${SOAK_S}s (warmup $WARMUP, window $WINDOW)"
    python3 tools/bench/stream_load.py --mode soak --port "$PORT" --streams "$SOAK_C" \
        --duration "$SOAK_S" --warmup "$WARMUP" --window "$WINDOW" --seed 42 \
        --clips $BANK --json "$RUN/soak-C$SOAK_C.json" 2>&1 | tee -a "$RUN/run.log" \
        | grep -E "TTFP|emission lag|finaliz|backlog|drift|verdict|utterances|pacing|window"
fi

# ---- 7. the server's own view, then stop it ----------------------------
curl -s -m 5 "http://localhost:$PORT/v1/health" > "$RUN/health-after.json" 2>&1
curl -s -m 5 "http://localhost:$(( PORT + 1000 ))/metrics" > "$RUN/metrics-after.txt" 2>&1
kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null
sleep 1
# Only OUR fleet: another server on another port is somebody else's business, and
# counting it would make a clean run look dirty.
SURV=$(pgrep -f "mynah-asr-server .* -p $PORT " | wc -l | tr -d ' ')
say "survivors=$SURV"

cat > "$RUN/manifest.json" <<JSON
{ "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "commit": "$REV", "dirty_files": $DIRTY,
  "qualifying": "$QUALIFY", "binary_sha256": "$BIN_SHA", "model": "$MODEL", "quant": "$QUANT",
  "workers": $W, "threads_per_worker": $T, "cap": $CAP, "cpus": ${NCPU},
  "loadavg_at_start": "$LOAD", "wave": "$WAVE", "soak_concurrency": $SOAK_C,
  "soak_seconds": $SOAK_S, "warmup_s": $WARMUP, "window_s": $WINDOW, "survivors": $SURV }
JSON
say "done. Evidence in $RUN (manifest.json, banner.txt, dispatch.txt, masks.txt, wave-*.json, soak-*.json)"
[ "$QUALIFY" = yes ] || say "REMINDER: dirty tree -> NON-QUALIFYING, screens and diagnostics only"
