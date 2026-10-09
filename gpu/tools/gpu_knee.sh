#!/bin/bash
# gpu/tools/gpu_knee.sh — a WAVE-class screen of mynah-asr-server-cuda over a
# list of concurrency levels: a fresh server per level, a closed-loop SOAK-shaped
# load for a short window, and per level one line of client latency, one line
# of what the GPU was doing ([GPU]: SM %, power, temperature, SM clock, throttle
# reasons, VRAM) and -- with --profile-host in SRV_ARGS -- the engine thread's
# host/device split for the measured window ([HOSTP-LEVEL]). A screen may
# disqualify, never promote (ENGINEERING.md §8); gpu_qualify.sh is the gate.
#
# usage: [VAR=value ...] gpu/tools/gpu_knee.sh
#
#   CLIPS     space-separated 16 kHz WAVs (required), e.g. "$(ls samples/fleurs-*/*.wav)"
#   LEVELS    concurrency levels, in order                    (default: "32 64")
#   DUR       measured seconds per level                      (default: 60)
#   WARM      warm-up seconds per level, discarded            (default: 20)
#   MODEL     converted pack       (default: models/nemotron-3.5-asr-streaming-0.6b)
#   CAP       --cap                                           (default: highest level + 16)
#   COHORT    --cohort-ms                                     (default: 40)
#   GEMM      --gemm                                          (default: own)
#   SRV_ARGS  extra server arguments, e.g. "--profile-host --pass-lanes cap"
#   PIN       taskset CPU list for the server (the GPU's NUMA node), e.g. 0-31
#   CLIPIN    taskset CPU list for the load generator (other CPUs / another node)
#   LANG_Q    lang query (default: auto)   LOOKAHEAD (default: 3)
#   DEVICE    GPU ordinal (default: 0)     PORT (default: 8391)   SAMPLE_MS (default: 500)
#   ONSETS    JSON {clip: speech onset s} from tools/bench/clip_onset.py; with it
#             TTFP is also measured from the speech onset (ttfp_on95), which a
#             clip that starts with silence does not inflate (default: none)
#   TAG, OUT  results go to $OUT/$TAG      (default: ./gpu-knee-res/run)
#   BIN       server binary                (default: ./mynah-asr-server-cuda)
#
# Per level, in $OUT/$TAG: server-C<n>.log (banner, [TOPOLOGY], [DUMP], [HOSTP]),
# load-C<n>.json/.txt (tools/bench/stream_load.py), gpu-C<n>.csv/.json,
# hostp-C<n>.json. The start-up [TOPOLOGY] line goes to stdout and topology.txt.
set -u
cd "$(dirname "$0")/../.." || exit 2
LEVELS=${LEVELS:-"32 64"}; DUR=${DUR:-60}; WARM=${WARM:-20}
MODEL=${MODEL:-models/nemotron-3.5-asr-streaming-0.6b}
TOP=$(printf '%s\n' $LEVELS | sort -n | tail -1)
CAP=${CAP:-$(( TOP + 16 ))}; COHORT=${COHORT:-40}; GEMM=${GEMM:-own}; SRV_ARGS=${SRV_ARGS:-}
LANG_Q=${LANG_Q:-auto}; LOOKAHEAD=${LOOKAHEAD:-3}; DEVICE=${DEVICE:-0}; PORT=${PORT:-8391}
SAMPLE_MS=${SAMPLE_MS:-500}; TAG=${TAG:-run}; OUT=${OUT:-$PWD/gpu-knee-res}; BIN=${BIN:-./mynah-asr-server-cuda}
PY=${PY:-python3}
[ -n "${CLIPS:-}" ] || { echo "gpu_knee: CLIPS is required (16 kHz WAVs)" >&2; exit 2; }
[ -x "$BIN" ] || { echo "gpu_knee: build first (make -C gpu)" >&2; exit 2; }
[ -f "$MODEL/mynah.json" ] || { echo "gpu_knee: no pack at $MODEL" >&2; exit 2; }
SP=""; [ -n "${PIN:-}" ] && SP="taskset -c $PIN"
CP=""; [ -n "${CLIPIN:-}" ] && CP="taskset -c $CLIPIN"
R=$OUT/$TAG; mkdir -p "$R" && R=$(cd "$R" && pwd)
curl -fsS -m 2 "localhost:$PORT/v1/health" >/dev/null 2>&1 && { echo "gpu_knee: port $PORT is busy" >&2; exit 1; }

REV=$(git rev-parse --short HEAD 2>/dev/null || echo "?"); DIRTY=$(git status --porcelain 2>/dev/null | grep -vc '^??')
echo "== gpu_knee $TAG commit=$REV dirty_files=$DIRTY binary=$(sha256sum "$BIN" | cut -c1-16) levels=[$LEVELS] dur=${DUR}s warm=${WARM}s cap=$CAP cohort_ms=$COHORT gemm=$GEMM srv_args=[$SRV_ARGS]" | tee "$R/run.txt"
gpu/tools/topology.sh --device "$DEVICE" ${PIN:+--server-cpus "$PIN"} ${CLIPIN:+--gen-cpus "$CLIPIN"} | tee "$R/topology.txt" | tee -a "$R/run.txt"
NCLIP=$(echo $CLIPS | wc -w)

SPID=""; SMP=""
cleanup() {
    [ -n "$SMP" ] && kill "$SMP" 2>/dev/null
    if [ -n "$SPID" ]; then
        kill -TERM "$SPID" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SPID" 2>/dev/null
    fi
    SPID=""; SMP=""
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

for C in $LEVELS; do
    [ "$C" -le "$CAP" ] || { echo "  C$C exceeds CAP $CAP; skipped"; continue; }
    L="$R/server-C$C.log"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    ( exec $SP "$BIN" -m "$MODEL" -p "$PORT" --device "$DEVICE" --gemm "$GEMM" --cap "$CAP" \
          --threads $(( CAP + 32 )) --cohort-ms "$COHORT" $SRV_ARGS ) > "$L" 2>&1 &
    SPID=$!
    ok=0
    for _ in $(seq 1 240); do
        kill -0 "$SPID" 2>/dev/null || break
        curl -fsS -m 2 "localhost:$PORT/v1/health" >/dev/null 2>&1 && { ok=1; break; }
        sleep 0.5
    done
    [ $ok = 1 ] || { echo "  C$C: the server did not become ready"; tail -5 "$L"; cleanup; continue; }
    ready_s=$(( $(date +%s) - t0 ))
    grep -q "engine=cuda" "$L" || echo "  C$C: WARNING the banner does not say engine=cuda"
    # the GPU sampler and the warm-up dump start when the warm-up ends
    ( sleep "$WARM"; kill -USR1 "$SPID" 2>/dev/null
      exec "$PY" gpu/tools/gpu_sample.py run --out "$R/gpu-C$C.csv" --device "$DEVICE" --interval-ms "$SAMPLE_MS" ) &
    SMP=$!
    # shellcheck disable=SC2086
    timeout $(( DUR + WARM + 120 )) $CP "$PY" tools/bench/stream_load.py --port "$PORT" --mode soak --streams "$C" \
        --duration $(( DUR + WARM )) --warmup "$WARM" --window 30 --seed 42 --lang "$LANG_Q" \
        --lookahead "$LOOKAHEAD" --bank short,medium,long --class-bounds 8,20 --clips $CLIPS ${ONSETS:+--onsets "$ONSETS"} \
        --json "$R/load-C$C.json" > "$R/load-C$C.txt" 2>&1
    kill -USR1 "$SPID" 2>/dev/null; sleep 1
    # the sampler is the python process exec()ed into nvidia-smi
    pkill -P "$SMP" 2>/dev/null; kill "$SMP" 2>/dev/null; wait "$SMP" 2>/dev/null; SMP=""
    curl -s -m 5 "localhost:$PORT/v1/health" > "$R/health-C$C.json" 2>/dev/null
    cleanup
    "$PY" - "$R/load-C$C.json" "$C" "$ready_s" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"  C{sys.argv[2]}: no load JSON ({e})"); sys.exit(0)
s, env = d["summary"], d.get("envelope", {})
m, c = s["metrics"], s["counts"]
g = lambda k, q="p95": m.get(k, {}).get(q)  # noqa: E731
f = lambda v, fmt="%.0f": "n/a" if v is None else fmt % v  # noqa: E731
print(f"  C{sys.argv[2]} ready={sys.argv[3]}s ok={c['ok']}/{c['utterances']} err={c['errors']} rej={c['rejected']} "
      f"lag95={f(g('emission_lag_ms'))}ms fin95={f(g('finalization_lag_ms'))}ms ttfp95={f(g('ttfp_ms'))}ms "
      f"ttfp_on95={f(g('ttfp_from_onset_ms'))}ms "
      f"backlog_max={f(g('backlog_s', 'max'), '%.2f')}s verdict={env.get('verdict', '?')}")
PY
    sed -n 's/^\(\[SERVER-CONFIG\].*\)vram_used_at_ready_mb=\([0-9]*\).*/    vram_used_at_ready_mb=\2/p' "$L" | head -1
    "$PY" gpu/tools/gpu_sample.py summary "$R/gpu-C$C.csv" --label "C$C" --json "$R/gpu-C$C.json" | sed 's/^/  /'
    grep "seq=.* vram used_mb" "$L" | tail -1 | sed 's/.*vram /    server vram /'
    if grep -q "^\[HOSTP\]" "$L"; then
        # seq 1 = end of warm-up, seq 2 = end of the window, seq 3 = shutdown
        "$PY" gpu/tools/hostp_level.py "$L" --from 1 --to 2 --label "C$C" --json "$R/hostp-C$C.json" | sed 's/^/  /'
    fi
done
echo "== results in $R"
