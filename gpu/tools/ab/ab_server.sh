#!/bin/bash
# gpu/tools/ab/ab_server.sh -- an interleaved A/B (ABBA) of mynah-asr-server-cuda.
#
# usage: [VAR=value ...] gpu/tools/ab/ab_server.sh BASE_BIN PATCH_BIN
#
# Per level, the arms run in ORDER (CYCLES times), each run on a FRESH server:
# start, wait for /v1/health, one fixed-mode load (tools/bench/stream_load.py
# --mode wave --streams C --repeat REPEAT: every stream plays the same clips in
# every arm, so the transcripts and the latencies pair clip by clip), /v1/health,
# stop. Then gpu/tools/ab/ab_report.py OUT/TAG applies the gates and the verdict.
#
#   PACK        converted pack                (default models/nemotron-3.5-asr-streaming-0.6b)
#   LEVELS      concurrency levels            (default "16 64")
#   ORDER       arm letters per level         (default ABBA; e.g. ABCCBA for 3 arms)
#   CYCLES      ORDER repeats per level       (default 1)
#   REPEAT      utterances per stream          (default 2)
#   CLIPS       16 kHz WAVs, repo-relative    (default samples/*/fleurs_*.wav: 23 clips,
#               the longest fleurs_long 94.6 s; NOT samples/*/*.wav, whose 122/305 s
#               long clips outlive the client's done-timeout)
#   ONSETS      {clip: onset s} map, or none  (default gpu/tools/ab/onsets-fleurs.json,
#               tools/bench/clip_onset.py on the 23 clips with its defaults)
#   COMMON_ARGS server args of every arm      (default "--gemm own-tc": the bf16 serving
#               default, explicit; gpu_knee's GEMM=own would be f32)
#   A_ARGS, B_ARGS, C_ARGS ...  per-arm server args, AFTER COMMON_ARGS (last one wins)
#   A_BIN=BASE_BIN  B_BIN=PATCH_BIN  C_BIN.. (default PATCH_BIN)   A_LABEL, B_LABEL ...
#   CAP         --cap (default highest level + 16)   THREADS (default CAP + 32)
#   PORT 8391   DEVICE 0   LANG_Q auto   LOOKAHEAD 3
#   PIN, CLIPIN taskset CPU lists for the server / the load generator
#   LOCK        flock file held for the whole session (default ~/gpu.lock; none = no lock)
#   LOCK_WAIT   seconds to wait for it        (default 0: a busy GPU fails fast)
#   READY_TIMEOUT  s for /v1/health           (default 180)
#   DONE_TIMEOUT   stream_load --done-timeout (default 60)
#   LOAD_TIMEOUT   hard cap of one load, s    (default REPEAT*(longest clip+DONE_TIMEOUT)+120)
#   STOP_TIMEOUT   s between TERM and KILL    (default 30)
#   COOL_S      pause between runs, s         (default 5)
#   BUDGET_MIN  stop starting runs past this many minutes (default 0 = no cap); the runs
#               not started are reported as never run, so the session FAILs gate 2
#   REQUIRE_CUDA  1 = a banner without engine=cuda fails gate 2 (default 1)
#   OUT, TAG    results go to $OUT/$TAG       (default ./ab-res/ab-<utc>)
#   DRY_RUN=1   print the plan, the commands and the expected wall; run nothing
#
# In $OUT/$TAG: ab.json (arms, binaries' sha256, args, the run plan), per run
# server-<run>.log, load-<run>.json/.txt, health-<run>.json, meta-<run>.json.
set -u
[ $# -eq 2 ] || { sed -n '2,45p' "$0"; exit 2; }
CALLER=$PWD   # relative binaries and OUT resolve against the caller's directory
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$CALLER/$1" ;; esac; }
BASE_BIN=$(abspath "$1"); PATCH_BIN=$(abspath "$2")
OUT=$(abspath "${OUT:-$PWD/ab-res}")
cd "$(dirname "$0")/../../.." || exit 2
HERE=gpu/tools/ab
PY=${PY:-python3}

PACK=${PACK:-models/nemotron-3.5-asr-streaming-0.6b}
LEVELS=${LEVELS:-"16 64"}; ORDER=${ORDER:-ABBA}; CYCLES=${CYCLES:-1}; REPEAT=${REPEAT:-2}
CLIPS=${CLIPS:-$(ls samples/*/fleurs_*.wav)}
ONSETS=${ONSETS:-$HERE/onsets-fleurs.json}; [ "$ONSETS" = none ] && ONSETS=""
COMMON_ARGS=${COMMON_ARGS-"--gemm own-tc"}
TOP=$(printf '%s\n' $LEVELS | sort -n | tail -1)
CAP=${CAP:-$(( TOP + 16 ))}; THREADS=${THREADS:-$(( CAP + 32 ))}
PORT=${PORT:-8391}; DEVICE=${DEVICE:-0}; LANG_Q=${LANG_Q:-auto}; LOOKAHEAD=${LOOKAHEAD:-3}
LOCK=${LOCK:-$HOME/gpu.lock}; LOCK_WAIT=${LOCK_WAIT:-0}
READY_TIMEOUT=${READY_TIMEOUT:-180}; DONE_TIMEOUT=${DONE_TIMEOUT:-60}; STOP_TIMEOUT=${STOP_TIMEOUT:-30}
COOL_S=${COOL_S:-5}; BUDGET_MIN=${BUDGET_MIN:-0}; REQUIRE_CUDA=${REQUIRE_CUDA:-1}
TAG=${TAG:-ab-$(date -u +%Y%m%d-%H%M%S)}; R=$OUT/$TAG
SP=""; [ -n "${PIN:-}" ] && SP="taskset -c $PIN"
CP=""; [ -n "${CLIPIN:-}" ] && CP="taskset -c $CLIPIN"
LONGEST=$("$PY" -c 'import sys,wave
print(int(max(wave.open(c).getnframes()/16000.0 for c in sys.argv[1:]))+1)' $CLIPS) || exit 2
LOAD_TIMEOUT=${LOAD_TIMEOUT:-$(( REPEAT * (LONGEST + DONE_TIMEOUT) + 120 ))}

arm_var() { local v="${1}_$2"; echo "${!v-}"; }
ARMS=$(echo "$ORDER" | fold -w1 | sort -u | tr -d '\n')
ARM_SPECS=()
for X in $(echo "$ARMS" | fold -w1); do
    case $X in A) dflt=$BASE_BIN ;; *) dflt=$PATCH_BIN ;; esac
    b=$(arm_var "$X" BIN); b=$(abspath "${b:-$dflt}")
    ARM_SPECS+=(--arm "$X:$(arm_var "$X" LABEL):$b:$COMMON_ARGS $(arm_var "$X" ARGS)")
done
REV=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
PLAN=$("$PY" $HERE/ab_lib.py plan --kind server --out "$R" --tag "$TAG" --order "$ORDER" \
    --cycles "$CYCLES" --levels "$LEVELS" --repeat "$REPEAT" "${ARM_SPECS[@]}" \
    --clips $CLIPS ${ONSETS:+--onsets "$ONSETS"} --ready-est "${READY_EST:-30}" --stop-est $(( 10 + COOL_S )) --extra harness_rev="$REV" pack="$PACK" \
    cap="$CAP" require_cuda="$REQUIRE_CUDA" load_timeout_s="$LOAD_TIMEOUT") || exit 2

srv_cmd() {  # arm -> the server command line
    local X=$1 b
    b=$(arm_var "$X" BIN); [ -n "$b" ] || { case $X in A) b=$BASE_BIN ;; *) b=$PATCH_BIN ;; esac; }
    echo "$(abspath "$b") -m $PACK -p $PORT --device $DEVICE --cap $CAP --threads $THREADS $COMMON_ARGS $(arm_var "$X" ARGS)"
}
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "== ab_server DRY RUN tag=$TAG levels=[$LEVELS] order=$ORDER cycles=$CYCLES repeat=$REPEAT cap=$CAP load_timeout=${LOAD_TIMEOUT}s"
    for X in $(echo "$ARMS" | fold -w1); do echo "   arm $X: $(srv_cmd "$X")"; done
    echo "   load: stream_load.py --mode wave --streams C --repeat $REPEAT --done-timeout $DONE_TIMEOUT --clips <$(echo $CLIPS | wc -w) clips> ${ONSETS:+--onsets $ONSETS}"
    echo "$PLAN" | awk '{printf "   %s", $1} END {print ""}'
    rm -rf "$R"; exit 0
fi
[ -f "$PACK/mynah.json" ] || { echo "ab_server: no pack at $PACK" >&2; exit 2; }
for X in $(echo "$ARMS" | fold -w1); do
    b=$(srv_cmd "$X" | cut -d' ' -f1); [ -x "$b" ] || { echo "ab_server: arm $X binary $b is not executable" >&2; exit 2; }
done
if [ "$LOCK" != none ]; then
    command -v flock >/dev/null || { echo "ab_server: flock missing (LOCK=none to run unlocked)" >&2; exit 2; }
    exec 9>"$LOCK"
    flock -w "$LOCK_WAIT" 9 || { echo "ab_server: $LOCK is held (another GPU job); LOCK_WAIT=s to queue" >&2; exit 1; }
fi
curl -fsS -m 2 "localhost:$PORT/v1/health" >/dev/null 2>&1 && { echo "ab_server: port $PORT is busy" >&2; exit 1; }

SPID=""
stop_server() {
    [ -n "$SPID" ] || return 0
    kill -TERM "$SPID" 2>/dev/null
    for _ in $(seq 1 "$STOP_TIMEOUT"); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SPID" 2>/dev/null
    wait "$SPID" 2>/dev/null; SRV_RC=$?
    SPID=""
}
trap 'stop_server; exit 130' INT TERM
trap stop_server EXIT

echo "== ab_server $TAG harness=$REV levels=[$LEVELS] order=$ORDER cycles=$CYCLES repeat=$REPEAT cap=$CAP -> $R" | tee "$R/run.txt"
T0=$(date +%s)
while read -r RUN C X; do
    if [ "$BUDGET_MIN" -gt 0 ] && [ $(( $(date +%s) - T0 )) -ge $(( BUDGET_MIN * 60 )) ]; then
        echo "  budget of $BUDGET_MIN min spent: $RUN and later not started" | tee -a "$R/run.txt"; break
    fi
    read -r -a CMD <<< "$(srv_cmd "$X")"
    L="$R/server-$RUN.log"; t0=$(date +%s); SRV_RC=""
    # shellcheck disable=SC2086
    ( exec $SP "${CMD[@]}" ) > "$L" 2>&1 &
    SPID=$!
    ready=false
    for _ in $(seq 1 $(( READY_TIMEOUT * 2 ))); do
        kill -0 "$SPID" 2>/dev/null || break
        curl -fsS -m 2 "localhost:$PORT/v1/health" >/dev/null 2>&1 && { ready=true; break; }
        sleep 0.5
    done
    ready_s=$(( $(date +%s) - t0 ))
    cuda=false; grep -q "engine=cuda" "$L" && cuda=true
    rc=-1; timed_out=false; alive=false; hstat=none
    if $ready; then
        # shellcheck disable=SC2086
        timeout -k 10 "$LOAD_TIMEOUT" $CP "$PY" tools/bench/stream_load.py --port "$PORT" --mode wave \
            --streams "$C" --repeat "$REPEAT" --lang "$LANG_Q" --lookahead "$LOOKAHEAD" \
            --done-timeout "$DONE_TIMEOUT" --clips $CLIPS ${ONSETS:+--onsets "$ONSETS"} \
            --json "$R/load-$RUN.json" > "$R/load-$RUN.txt" 2>&1
        rc=$?
        [ $rc -eq 124 ] || [ $rc -eq 137 ] && timed_out=true
        kill -0 "$SPID" 2>/dev/null && alive=true
        curl -s -m 5 "localhost:$PORT/v1/health" > "$R/health-$RUN.json" 2>/dev/null
        hstat=$("$PY" -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("status","?"))
except Exception: print("unreachable")' "$R/health-$RUN.json")
    else
        echo "  $RUN: the server did not become ready in ${READY_TIMEOUT}s"; tail -5 "$L"
    fi
    stop_server
    cfg=$(grep -m1 '^\[SERVER-CONFIG\] mynah-asr-server' "$L")
    "$PY" $HERE/ab_lib.py meta "$R/meta-$RUN.json" run="$RUN" level="$C" arm="$X" ready=$ready \
        ready_s=$ready_s load_rc=$rc timed_out=$timed_out server_alive_after=$alive \
        health_status="$hstat" server_rc="${SRV_RC:-}" engine_cuda=$cuda wall_s=$(( $(date +%s) - t0 )) \
        config="$cfg"
    if $ready; then "$PY" $HERE/ab_lib.py line "$R/load-$RUN.json" "$RUN ready=${ready_s}s rc=$rc"; fi
    [ "$REQUIRE_CUDA" = 1 ] && ! $cuda && echo "  $RUN: WARNING the banner does not say engine=cuda"
    sleep "$COOL_S"
done <<< "$PLAN"
echo "== done in $(( ($(date +%s) - T0) / 60 )) min; report: $PY $HERE/ab_report.py $R" | tee -a "$R/run.txt"
"$PY" $HERE/ab_report.py "$R" --json "$R/verdict.json" | tee "$R/report.txt"
exit "${PIPESTATUS[0]}"
