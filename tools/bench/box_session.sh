#!/bin/sh
# box_session.sh — one Linux box, thirty minutes, an answer to "what C does this
# server hold".
#
# box_qualify.sh is the QUALIFICATION: long soaks, a promotion. This is the
# session that comes before it — describe the machine, prove what the binary
# runs, predict from the box's own step table, then climb a ladder until the
# streaming envelope breaks and soak briefly at the last rung that held. It is
# meant to fit in one sitting, and every phase writes an artefact.
#
# Usage:
#   tools/bench/box_session.sh -m models/nemotron-3.5-asr-streaming-0.6b \
#       [--parakeet models/parakeet-tdt_ctc-110m-gguf] \
#       [--ladder "8 16 24 32 48 64 80 100"] [--soak-seconds 120] \
#       [--server-cpus 0-23] [--load-cpus 24-31] [-o ~/asr-evidence] [--allow-load]
#
# Why the two cpu lists are mandatory on Linux: the load generator is one
# PROCESS per stream, and at C=100 that is 100 processes. Sharing cores with the
# server does not slow the run down honestly, it moves the bottleneck into the
# harness — the sibling project measured an unpinned generator costing worker 0
# about 8 % of its first-audio latency on this exact machine shape. So the
# server gets a slice, the generator gets the rest, and the capacity reported is
# the capacity OF THAT SLICE, stated and never scaled up to the whole box.
#
# It refuses rather than produce a number it cannot support:
#   * loadavg >= 2.0 and no --allow-load  -> refuse (someone else owns the box)
#   * the server did not answer /v1/health -> refuse, with its log
#   * a model that cannot stream           -> its ladder is REST, not WAVE
# The health probe exists because its absence has already cost a session: a
# ladder was once run against a server that had exited on a bad flag, and five
# rungs of "connection refused" took 0.1 s each and looked like results.
#
# Exit: 0 ran · 2 usage · 3 refused
set -u

MODEL=""; PARAKEET=""; OUT="$HOME/asr-evidence"; PORT=8760; METRICS=9760
LADDER="8 16 24 32 48 64 80 100"; SOAK_S=120; SOAK_C=0
SRV_CPUS=""; LOAD_CPUS=""; ALLOW_LOAD=0; QUANT=int8; LOOKAHEAD=3; BUDGET_MIN=30
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2 ;;
        --parakeet) PARAKEET="$2"; shift 2 ;;
        -o) OUT="$2"; shift 2 ;;
        -p) PORT="$2"; shift 2 ;;
        --ladder) LADDER="$2"; shift 2 ;;
        --soak-seconds) SOAK_S="$2"; shift 2 ;;
        --soak-at) SOAK_C="$2"; shift 2 ;;
        --server-cpus) SRV_CPUS="$2"; shift 2 ;;
        --load-cpus) LOAD_CPUS="$2"; shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        --lookahead) LOOKAHEAD="$2"; shift 2 ;;
        --budget-min) BUDGET_MIN="$2"; shift 2 ;;
        --allow-load) ALLOW_LOAD=1; shift ;;
        -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "box_session: unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$MODEL" ] || { echo "box_session: -m <model_dir> is required" >&2; exit 2; }
[ -f "$MODEL/mynah.json" ] || { echo "box_session: $MODEL is not a converted model" >&2; exit 2; }
[ -x ./mynah-asr-server ] || { echo "box_session: build first (make)" >&2; exit 2; }

RUN="$OUT/session-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN" || exit 2
T0=$(date +%s)
say() { echo "$@" | tee -a "$RUN/session.log"; }
over_budget() { [ $(( ($(date +%s) - T0) / 60 )) -ge "$BUDGET_MIN" ]; }
srun() { if [ -n "$SRV_CPUS" ]; then taskset -c "$SRV_CPUS" "$@"; else "$@"; fi; }
lrun() { if [ -n "$LOAD_CPUS" ]; then taskset -c "$LOAD_CPUS" "$@"; else "$@"; fi; }

say "================ box_session $(date -u +%FT%TZ) ================"
say "run dir      $RUN"
say "budget       ${BUDGET_MIN} min"

# ---------------------------------------------------------------- P0 preflight
say ""; say "---- P0 the machine, before anything is measured"
{ uname -a; echo "nproc: $(nproc 2>/dev/null || echo '?')"; echo "loadavg: $(cat /proc/loadavg 2>/dev/null)";
  echo "mem: $(awk '/MemTotal|MemAvailable/{print $1,$2/1048576" GB"}' /proc/meminfo 2>/dev/null | tr '\n' ' ')";
  echo "users:"; who; echo "tree: $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"; } \
  > "$RUN/preflight.txt" 2>&1
cat "$RUN/preflight.txt" | tee -a "$RUN/session.log"
LOAD1=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
if [ "$ALLOW_LOAD" -eq 0 ] && [ "$(echo "$LOAD1 >= 2.0" | bc 2>/dev/null || echo 0)" = "1" ]; then
    say "REFUSED: loadavg $LOAD1 — another load owns this box; nothing measured here would be about the server."
    exit 3
fi

# ------------------------------------------------------------------- P1 doctor
say ""; say "---- P1 doctor (read-only: what this machine IS)"
if [ -x tools/bench/box_doctor.sh ]; then
    sh tools/bench/box_doctor.sh --bin ./mynah-asr -o "$RUN" --no-evidence \
        > "$RUN/doctor.txt" 2>&1 || true
    grep -E '^\[DOCTOR\]|topology|hugepage|governor|REFUS' "$RUN/doctor.txt" | head -30 \
        | tee -a "$RUN/session.log"
    say "  (full: $RUN/doctor.txt)"
fi

# ------------------------------------------------------------- P2 dispatch map
say ""; say "---- P2 what the binary resolves (ENGINEERING.md §5: a benchmark is invalid until dispatch is proven)"
./mynah-asr-server --dispatch-map > "$RUN/dispatch-map.txt" 2>&1 || true
cat "$RUN/dispatch-map.txt" | tee -a "$RUN/session.log"

# ---------------------------------------------------------------- P3 calibrate
say ""; say "---- P3 the box's own step table, and the prediction that follows"
# Read the weights once before timing anything. On a freshly booted box the
# model file is not in the page cache, and the first pass through it pays
# disk instead of DRAM -- which lands entirely in `a`, the fixed cost, and
# makes a cold box look like a slow one. This is also why the calibration
# runs on the SERVER's cpu slice: `a` and `b` are functions of how many
# cores walk the weights, so fitting them on the generator's slice
# describes the generator.
CACHED_BEFORE=$(awk "/^Cached:/{print \$2}" /proc/meminfo 2>/dev/null)
cat "$MODEL"/* > /dev/null 2>&1
say "  page cache warmed: Cached $CACHED_BEFORE kB -> $(awk '/^Cached:/{print $2}' /proc/meminfo 2>/dev/null) kB"
if [ -f tools/bench/box_advisor.py ]; then
    srun python3 tools/bench/box_advisor.py --model "$MODEL" \
        --json "$RUN/advisor.json" > "$RUN/advisor.txt" 2>&1 || true
    tail -40 "$RUN/advisor.txt" | tee -a "$RUN/session.log"
fi

# ------------------------------------------------------- the server, and proof
PHASE=0
start_server() {   # $1 = model dir, $2 = extra args
    PHASE=$((PHASE + 1))
    SRV_LOG="$RUN/server-$PHASE-$(basename "$1").log"
    # The port must be free BEFORE we start, or the new server exits on bind and
    # the probe below happily finds the OLD one still answering. That is not a
    # hypothetical: it happened twice in one morning and silently pointed a
    # Parakeet phase at a Nemotron process.
    i=0
    while curl -fsS --max-time 1 "http://localhost:$PORT/v1/health" >/dev/null 2>&1; do
        i=$((i + 1))
        if [ $i -gt 20 ]; then
            say "REFUSED: something is still serving on port $PORT after 20 s."
            return 1
        fi
        sleep 1
    done
    # Background the COMMAND, never a shell function: `f &` where f is a function
    # backgrounds a SUBSHELL, $! is that subshell, and taskset's child survives
    # the kill. The server then holds the port for every phase that follows.
    if [ -n "$SRV_CPUS" ]; then
        taskset -c "$SRV_CPUS" ./mynah-asr-server -m "$1" -p "$PORT" --quant "$QUANT" \
            --metrics-port "$METRICS" $2 > "$SRV_LOG" 2>&1 &
    else
        ./mynah-asr-server -m "$1" -p "$PORT" --quant "$QUANT" \
            --metrics-port "$METRICS" $2 > "$SRV_LOG" 2>&1 &
    fi
    SRV=$!
    # The probe that yesterday's run did not have. A server that died on a bad
    # flag answers nothing, and a ladder against a dead port produces five
    # perfectly formatted rungs of INVALID.
    i=0
    while [ $i -lt 120 ]; do
        if ! kill -0 "$SRV" 2>/dev/null; then
            say "REFUSED: the server exited before it listened. Its output:"
            sed 's/^/    | /' "$SRV_LOG" | head -40 | tee -a "$RUN/session.log"
            return 1
        fi
        if curl -fsS --max-time 2 "http://localhost:$PORT/v1/health" > "$RUN/health-up.json" 2>/dev/null; then
            # "Something answers" is not "my server answers". A fresh process has
            # served nothing; one with sessions behind it is somebody else's, and
            # a phase measured against it is measuring the wrong model or a
            # process that has already been beaten up by the previous rung.
            SESS=$(python3 -c "import json;print(json.load(open('$RUN/health-up.json')).get('sessions',-1))" 2>/dev/null || echo -1)
            if [ "$SESS" != "0" ]; then
                say "REFUSED: the server on port $PORT reports sessions=$SESS — it is not the one just started."
                sed 's/^/    | /' "$SRV_LOG" | head -20 | tee -a "$RUN/session.log"
                kill "$SRV" 2>/dev/null
                return 1
            fi
            say "  server up after ${i}s (fresh: sessions=0, pid $SRV)"
            return 0
        fi
        i=$((i + 1)); sleep 1
    done
    say "REFUSED: the server never answered /v1/health in 120 s."
    sed 's/^/    | /' "$SRV_LOG" | head -40 | tee -a "$RUN/session.log"
    kill "$SRV" 2>/dev/null
    return 1
}
stop_server() {
    [ -n "${SRV:-}" ] || return 0
    kill "$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
    # and do not return until the port is actually free, so the next phase does
    # not race a socket still in the kernel's hands
    i=0
    while curl -fsS --max-time 1 "http://localhost:$PORT/v1/health" >/dev/null 2>&1 && [ $i -lt 15 ]; do
        i=$((i + 1)); sleep 1
    done
    SRV=""
}
trap 'sample_stop; stop_server' EXIT INT TERM

# Two corpora, because the two phases ask different questions. A WAVE rung is
# paced at real time, so its wall clock is the LONGEST clip in it: one 94 s
# utterance turns an eight-rung ladder into half an hour of waiting for a
# screening result. The long clip belongs in the soak, where the duration is
# fixed and a long utterance is the point (it is the "long" bank class).
WAVE_CLIPS="samples/en/fleurs_1521.wav samples/en/fleurs_1534.wav tests/audio/test_en.wav"
SOAK_CLIPS="samples/en/fleurs_1521.wav samples/en/fleurs_1534.wav samples/en/fleurs_long.wav tests/audio/test_en.wav"
verdict_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['envelope']['verdict'])" "$1" 2>/dev/null || echo UNREADABLE; }

# A rung that breaks must say WHERE it broke, or the next session re-runs it to
# find out. On 2026-09-20 a c=8 rung emitted a first partial on all eight
# streams and then stopped: every reader timed out, pooled emission lag came
# back as 59 729 ms, and nothing in the artefacts could distinguish "the server
# wedged" from "the client could not keep up" -- the server-side counters were
# only read after the rung, when whatever it was had passed.
#
# So sample the metrics port THROUGH the rung. If mynah_asr_steps_total stops
# advancing while streams are open, the server stopped stepping; if it keeps
# advancing while the client sees nothing, the problem is downstream of the
# step. One curl every two seconds costs nothing and answers the question
# without a second run.
sample_start() {
    ( while :; do
        printf '%s ' "$(date +%s)"
        curl -fsS --max-time 1 "http://localhost:$METRICS/metrics" 2>/dev/null \
            | awk '/^mynah_asr_steps_total|^mynah_asr_deltas_total|^mynah_asr_emission_lag_ms_max/{printf "%s=%s ", $1, $2}'
        echo
        sleep 2
      done ) > "$1" 2>&1 &
    SAMPLER=$!
}
sample_stop() { [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null; SAMPLER=""; }
sample_verdict() {   # $1 = the sample file
    python3 - "$1" <<'PYEOF'
import re, sys
rows = []
for line in open(sys.argv[1]):
    t = re.match(r"(\d+) ", line)
    m = re.search(r"mynah_asr_steps_total\S*\s*(\d+(?:\.\d+)?)", line)
    if t and m:
        rows.append((int(t.group(1)), float(m.group(1))))
if len(rows) < 3:
    print("  steps: not sampled (metrics port unreadable)"); raise SystemExit
worst, at = 0, None
for (t0, s0), (t1, s1) in zip(rows, rows[1:]):
    if s1 == s0 and t1 - t0 > worst:
        worst, at = t1 - t0, t0
total = rows[-1][1] - rows[0][1]
if worst >= 6:
    print(f"  steps: the server STOPPED STEPPING for {worst}s "
          f"(longest flat stretch, at t={at}); {total:.0f} steps over the rung")
else:
    print(f"  steps: advanced throughout ({total:.0f} steps, longest pause {worst}s)")
PYEOF
}

# ------------------------------------------------------ P4 the streaming ladder
say ""; say "---- P4 WAVE ladder (screening: it may disqualify a rung, it never promotes one)"
STREAMS_OK=0
if [ -n "$(./mynah-asr stream -m "$MODEL" -i tests/audio/test_en.wav --lang en 2>&1 | grep 'offline-only')" ]; then
    say "  $MODEL is offline-only (no cache-aware streaming): no WAVE for it."
else
    # --threads is the server's HTTP thread pool and a WebSocket stream holds one
    # for its whole life (server/main.c: "one slot per HTTP thread"), so it is the
    # CONCURRENCY ceiling -- not the compute width. The compute pool sizes itself
    # from this process's affinity mask, which is what --server-cpus sets. Asking
    # for a rung above --threads measures the accept path, not the model.
    CAP=$(python3 -c "print(max([int(x) for x in '$LADDER'.split()]) + 8)")
    SRV_THREADS=$CAP
    say "  server: --threads $SRV_THREADS (HTTP/concurrency ceiling) --cap $CAP --quant $QUANT"
    say "          compute pool: from the affinity mask of cpus ${SRV_CPUS:-all}"
    say "  load:   cpus ${LOAD_CPUS:-all}, lookahead $LOOKAHEAD requested per stream"
    start_server "$MODEL" "--threads $SRV_THREADS --cap $CAP" || exit 3
    for C in $LADDER; do
        if over_budget; then say "  budget reached, ladder stops at C=$C (not attempted)"; break; fi
        say ""
        say "  ======== WAVE c=$C ========"
        sample_start "$RUN/steps-c$C.txt"
        lrun python3 tools/bench/stream_load.py --mode wave --streams "$C" --repeat 2 \
            --clips $WAVE_CLIPS --port "$PORT" --lang en --lookahead "$LOOKAHEAD" \
            --json "$RUN/wave-c$C.json" > "$RUN/wave-c$C.txt" 2>&1
        sample_stop
        grep -E '^\s+(wall|utterances|\[|verdict|rejections)' "$RUN/wave-c$C.txt" \
            | sed 's/^/  /' | tee -a "$RUN/session.log"
        sample_verdict "$RUN/steps-c$C.txt" | tee -a "$RUN/session.log"
        V=$(verdict_of "$RUN/wave-c$C.json")
        case "$V" in
            GOOD)     STREAMS_OK=$C ;;
            MARGINAL) STREAMS_OK=$C; say "  (MARGINAL: it held, with no headroom)" ;;
            *)        say "  ladder stops: c=$C is $V; the last rung that held is c=$STREAMS_OK"; break ;;
        esac
    done
    stop_server
fi
say ""
say "  ladder result: the highest concurrency whose envelope held is c=$STREAMS_OK"

# ------------------------------------------------------------------- P5 a soak
[ "$SOAK_C" -gt 0 ] || SOAK_C=$STREAMS_OK
if [ "$SOAK_C" -gt 0 ] && ! over_budget; then
    say ""; say "---- P5 SOAK ${SOAK_S}s at c=$SOAK_C (closed loop: this one can promote)"
    start_server "$MODEL" "--threads $((SOAK_C + 8)) --cap $((SOAK_C + 8))" || exit 3
    lrun python3 tools/bench/stream_load.py --mode soak --streams "$SOAK_C" \
        --duration "$SOAK_S" --warmup 20 --window 30 \
        --clips $SOAK_CLIPS --port "$PORT" --lang en --lookahead "$LOOKAHEAD" \
        --json "$RUN/soak-c$SOAK_C.json" > "$RUN/soak-c$SOAK_C.txt" 2>&1
    grep -E '^\s+(wall|utterances|audio|\[|verdict|per-window|\s+\[)' "$RUN/soak-c$SOAK_C.txt" \
        | sed 's/^/  /' | tee -a "$RUN/session.log"
    curl -fsS --max-time 3 "http://localhost:$METRICS/metrics" > "$RUN/metrics-after-soak.txt" 2>/dev/null \
        && say "  metrics scraped -> $RUN/metrics-after-soak.txt"
    stop_server
fi

# ----------------------------------------------------- P6 the offline ladder
if [ -n "$PARAKEET" ] && [ -f "$PARAKEET/mynah.json" ] && ! over_budget; then
    say ""; say "---- P6 $(basename "$PARAKEET"): REST ladder"
    WHY=$(./mynah-asr stream -m "$PARAKEET" -i tests/audio/test_en.wav --lang en 2>&1 | grep -i 'offline-only\|does not' | head -1)
    [ -n "$WHY" ] && say "  streaming refused by the runtime: $WHY"
    say "  so the question is throughput, not real-time streams."
    start_server "$PARAKEET" "--threads 72" || exit 3
    lrun python3 tools/bench/rest_load.py --port "$PORT" --clips $WAVE_CLIPS --lang en \
        --ladder "1,2,4,8,16,32,64" --requests-per-stream 2 \
        --json "$RUN/parakeet-rest.json" 2>&1 | tee -a "$RUN/session.log"
    stop_server
fi

say ""
say "================ done in $(( ($(date +%s) - T0) / 60 )) min — $RUN ================"
