#!/bin/sh
# tier.sh — the three questions, three budgets, one command each.
#
#   tools/bench/tier.sh 0 -m <model_dir> [--parakeet <dir>]    ~90 s   does it work
#   tools/bench/tier.sh 1 -m <model_dir> [--parakeet <dir>]    ~8 min  where is the knee
#   tools/bench/tier.sh 2 -m <model_dir> -C <streams>          ~30 min certify that point
#
# WHY THESE THREE AND NOT A LADDER.  The 2026-09-20 campaign spent two box days
# and produced one number.  Reading back what it cost: twelve fixed rungs of
# 90 s behind a 25 s warm-up on a server restarted for each, then a 600 s soak
# that scored ZERO transcripts because no reference corpus was ever fetched.
# The knee was found eleven rungs later than a bisection would have found it,
# and the thing a soak is for -- is the output still right -- was not measured
# at all.  These three tiers are that post-mortem turned into commands.
#
# Tier 0 answers "is this build serving correctly at all", in the time it takes
#   to read the output: the dispatch map resolves, the metric definitions pass
#   their own known-answer test, four real-time streams come back byte-identical
#   to the CLI, REST agrees with the CLI, the transcripts score against human
#   references, the process leaves nothing behind.  It is the gate to run after
#   every change, and the only one that belongs in a loop.
# Tier 1 answers "how many streams does THIS host carry", by bisection on one
#   warm server (tools/bench/knee.py).  It SCREENS.  It never promotes.
# Tier 2 answers "does the chosen point hold, and is the text still right",
#   with two identical soaks -- because one soak is an anecdote and two that
#   agree are a measurement -- against a bank with references.
#
# Nothing here defines a metric or a threshold.  Every number comes from
# tools/bench/streaming_metrics.py through stream_load.py, every verdict is the
# envelope that module computes, and this file only decides what to run next.
#
# Exit: 0 the tier passed · 1 it failed · 2 usage · 3 refused (environment)
set -u

TIER="${1:-}"; shift 2>/dev/null || true
MODEL=""; PARAKEET=""; QUANT=int8; PORT=8791; METRICS=9791; CONC=0
OUT="${TIER_OUT:-$PWD/tier-evidence}"; LOOKAHEAD=3; SOAK_S=600; CPUS=""
SERVER_ARGS=""

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}
[ "$TIER" = "0" ] || [ "$TIER" = "1" ] || [ "$TIER" = "2" ] || usage

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)     MODEL="$2"; shift 2 ;;
        --parakeet)     PARAKEET="$2"; shift 2 ;;
        --quant)        QUANT="$2"; shift 2 ;;
        -p|--port)      PORT="$2"; shift 2 ;;
        --metrics-port) METRICS="$2"; shift 2 ;;
        -C|--streams)   CONC="$2"; shift 2 ;;
        --lookahead)    LOOKAHEAD="$2"; shift 2 ;;
        --soak-seconds) SOAK_S="$2"; shift 2 ;;
        --cpus)         CPUS="$2"; shift 2 ;;
        --server-args)  SERVER_ARGS="$2"; shift 2 ;;
        --out)          OUT="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *) echo "tier.sh: unknown argument $1" >&2; usage ;;
    esac
done
[ -n "$MODEL" ] || { echo "tier.sh: -m <model_dir> is required" >&2; usage; }
[ -d "$MODEL" ] || { echo "tier.sh: $MODEL is not a directory" >&2; exit 2; }
[ -x ./mynah-asr-server ] || { echo "tier.sh: ./mynah-asr-server not built (run make)" >&2; exit 2; }
[ -x ./mynah-asr ] || { echo "tier.sh: ./mynah-asr not built (run make)" >&2; exit 2; }

RUN="$OUT/tier$TIER-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN" || exit 2
LOG="$RUN/tier.log"
say() { echo "$@" | tee -a "$LOG"; }
FAIL=0
note_fail() { say "  FAIL: $*"; FAIL=1; }

say "=== tier $TIER  $(date -u +%Y-%m-%dT%H:%M:%SZ)  $(./mynah-asr --version 2>/dev/null | head -1) ==="
say "  model $MODEL  quant $QUANT  lookahead $LOOKAHEAD  -> $RUN"
TREE="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY=""; git diff --quiet 2>/dev/null || DIRTY=" (dirty)"
say "  tree $TREE$DIRTY"
if [ "$TIER" = "2" ] && [ -n "$DIRTY" ]; then
    say "REFUSED: tier 2 promotes an operating point and the tree is dirty."
    say "         Commit first: a qualification that cannot be reproduced is not one."
    exit 3
fi

# --------------------------------------------------------------- the corpora
# Short clips for cadence, the committed sample bank for references. The stress
# bank (make fetch-stress-bank) replaces the latter when it exists: until then
# every CER below rests on a handful of clips and says so.
# Clips that carry a HUMAN REFERENCE, so tier 0's CER is a measurement and not
# an empty column. tests/audio/test_*.wav are fixtures with no entry in any
# manifest: four streams of them produce "0 utterances scored", which is exactly
# the hole this whole audit was about.
CLIPS_SHORT="samples/nl/fleurs_1521.wav samples/en/fleurs_1521.wav samples/fr/fleurs_1521.wav"
[ -f samples/en/fleurs_1521.wav ] || \
    CLIPS_SHORT="tests/audio/test_it.wav tests/audio/test_en.wav tests/audio/test_de.wav"
BANK_MANIFEST=""
if [ -f samples/stress-en/manifest.json ]; then
    BANK_MANIFEST=samples/stress-en/manifest.json
    BANK_CLIPS="$(ls samples/stress-en/*.wav 2>/dev/null | head -400)"
elif [ -f samples/manifest.json ]; then
    BANK_MANIFEST=samples/manifest.json
    BANK_CLIPS="$(ls samples/*/*.wav 2>/dev/null | grep -v fleurs_long | head -40)"
else
    BANK_CLIPS="$CLIPS_SHORT"
fi
[ -n "$BANK_MANIFEST" ] || say "  NOTE: no manifest -> no CER. Run 'make fetch-stress-bank'."

# ---------------------------------------------------------- server lifecycle
SRV=""
start_server() {   # $1 = model dir, $2 = extra args
    SRV_LOG="$RUN/server-$(basename "$1").log"
    i=0
    while curl -fsS --max-time 1 "http://localhost:$PORT/v1/health" >/dev/null 2>&1; do
        i=$((i + 1))
        [ $i -gt 20 ] && { say "REFUSED: port $PORT still serving after 20 s."; return 1; }
        sleep 1
    done
    # Background the COMMAND, never a shell function: `f &` backgrounds a
    # subshell, $! is that subshell, and the real child survives the kill.
    if [ -n "$CPUS" ] && command -v taskset >/dev/null 2>&1; then
        taskset -c "$CPUS" ./mynah-asr-server -m "$1" -p "$PORT" --quant "$QUANT" \
            --metrics-port "$METRICS" $2 > "$SRV_LOG" 2>&1 &
    else
        ./mynah-asr-server -m "$1" -p "$PORT" --quant "$QUANT" \
            --metrics-port "$METRICS" $2 > "$SRV_LOG" 2>&1 &
    fi
    SRV=$!
    i=0
    while [ $i -lt 120 ]; do
        kill -0 "$SRV" 2>/dev/null || {
            say "REFUSED: the server exited before it listened:"
            sed 's/^/    | /' "$SRV_LOG" | head -30 | tee -a "$LOG"; return 1; }
        if curl -fsS --max-time 2 "http://localhost:$PORT/v1/health" > "$RUN/health-up.json" 2>/dev/null; then
            SESS=$(python3 -c "import json;print(json.load(open('$RUN/health-up.json')).get('sessions',-1))" 2>/dev/null || echo -1)
            [ "$SESS" = "0" ] || { say "REFUSED: sessions=$SESS — that is not the server just started."
                                   kill "$SRV" 2>/dev/null; return 1; }
            say "  server up after ${i}s (fresh, pid $SRV)"
            return 0
        fi
        i=$((i + 1)); sleep 1
    done
    say "REFUSED: no answer from /v1/health in 120 s."
    sed 's/^/    | /' "$SRV_LOG" | head -30 | tee -a "$LOG"
    kill "$SRV" 2>/dev/null; return 1
}
stop_server() {
    [ -n "$SRV" ] || return 0
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    i=0
    while curl -fsS --max-time 1 "http://localhost:$PORT/v1/health" >/dev/null 2>&1 && [ $i -lt 15 ]; do
        i=$((i + 1)); sleep 1
    done
    SRV=""
}
survivors_check() {
    n=$(pgrep -f "mynah-asr-server -m $MODEL" 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" = "0" ] || note_fail "$n server process(es) survived the stop"
}
trap 'stop_server' EXIT INT TERM

# ------------------------------------------------------------------- tier 0
tier0() {
    say ""
    say "-- the build says what it resolved (no model needed) --"
    ./mynah-asr-server --dispatch-map > "$RUN/dispatch.txt" 2>&1 || note_fail "--dispatch-map"
    if grep -qi "UNKNOWN" "$RUN/dispatch.txt"; then
        say "  NOTE: a dispatch row is UNKNOWN; see $RUN/dispatch.txt"
    fi
    grep -E "int8_rows|int8_dot|simd" "$RUN/dispatch.txt" | head -4 | sed 's/^/    /' | tee -a "$LOG"
    python3 tools/bench/streaming_metrics.py --self-test > "$RUN/metrics-selftest.txt" 2>&1 \
        && say "  metric definitions: self-test PASS" \
        || note_fail "streaming_metrics self-test (see $RUN/metrics-selftest.txt)"

    say ""
    say "-- the reference transcripts, from the CLI, once --"
    REFJSON="$RUN/cli-reference.json"
    : > "$RUN/refs.txt"
    # The cache file is named after the WHOLE path with the slashes flattened,
    # never the bare file name: samples/{nl,en,fr}/fleurs_1521.wav are three
    # different clips with one basename, and naming them by it made all three
    # references the last language transcribed. That is the same collision the
    # Python side had, reproduced here in shell, and it is why this tier exists.
    refname() { echo "$1" | sed 's|[/. ]|_|g'; }
    for w in $CLIPS_SHORT; do
        ./mynah-asr transcribe -m "$MODEL" -i "$w" --quant "$QUANT" > "$RUN/ref-$(refname "$w").txt" 2>/dev/null \
            || note_fail "CLI transcribe of $w"
    done
    python3 - "$RUN" $CLIPS_SHORT > "$REFJSON" <<'PY'
import json, os, re, sys
run, clips = sys.argv[1], sys.argv[2:]
out = {}
for c in clips:
    p = os.path.join(run, "ref-" + re.sub(r"[/. ]", "_", c) + ".txt")
    if os.path.exists(p):
        out[c] = open(p).read().strip()
print(json.dumps(out, ensure_ascii=False))
PY
    say "  $(python3 -c "import json;print(len(json.load(open('$REFJSON'))))" ) reference transcript(s) cached"

    say ""
    say "-- four real-time streams, against those references --"
    start_server "$MODEL" "--threads 8 --cap 8 $SERVER_ARGS" || exit 3
    EXTRA=""
    [ -n "$BANK_MANIFEST" ] && EXTRA="--transcripts $BANK_MANIFEST"
    # Capture the status BEFORE piping: `cmd | tee | tail` reports tail's status,
    # which is always 0, and a gate that cannot fail is not a gate.
    python3 tools/bench/stream_load.py --mode wave --streams 4 --repeat 1 \
        --port "$PORT" --lookahead "$LOOKAHEAD" --clips $CLIPS_SHORT \
        --reference "$REFJSON" $EXTRA --json "$RUN/t0-wave.json" > "$RUN/t0-wave.txt" 2>&1
    T0RC=$?
    tail -14 "$RUN/t0-wave.txt" | tee -a "$LOG"
    [ "$T0RC" = "0" ] || note_fail "the 4-stream wave did not pass its envelope (rc=$T0RC)"

    say ""
    say "-- REST agrees with the CLI --"
    for w in $CLIPS_SHORT; do
        curl -fsS -X POST "http://localhost:$PORT/v1/audio/transcriptions" \
            -F "file=@$w" > "$RUN/rest-$(refname "$w").json" 2>/dev/null || {
            note_fail "REST transcribe of $w"; continue; }
        python3 - "$RUN/rest-$(refname "$w").json" "$RUN/ref-$(refname "$w").txt" <<'PY' || FAIL=1
import json, sys
got = (json.load(open(sys.argv[1])).get("text") or "").strip()
want = open(sys.argv[2]).read().strip()
if got != want:
    print(f"  FAIL: REST differs from the CLI\n    cli:  {want[:90]}\n    rest: {got[:90]}")
    raise SystemExit(1)
PY
    done
    [ "$FAIL" = "0" ] && say "  REST == CLI on every clip"

    say ""
    say "-- the server accounts for itself --"
    curl -fsS "http://localhost:$PORT/v1/health" > "$RUN/health-after.json" 2>/dev/null \
        || note_fail "/v1/health after load"
    curl -fsS "http://localhost:$METRICS/metrics" > "$RUN/metrics.txt" 2>/dev/null \
        || note_fail "/metrics"
    kill -USR1 "$SRV" 2>/dev/null; sleep 1
    grep -c "DUMP" "$SRV_LOG" >/dev/null 2>&1 && say "  SIGUSR1 dump present in the server log"
    python3 - "$RUN/health-after.json" <<'PY' || FAIL=1
import json, sys
# the counters live under "batch", not at the top level -- reading them in the
# wrong place is a gate that fails on a healthy server, which is its own defect
h = json.load(open(sys.argv[1]))
b = (h.get("batch") or {})
steps = b.get("batched_steps_total", 0) or 0
rows = b.get("rows_stacked_total", 0) or 0
if steps <= 0 or rows <= 0:
    print(f"  FAIL: the batched path never ran (batched_steps={steps}, rows_stacked={rows})")
    raise SystemExit(1)
print(f"  batched steps {steps}, rows stacked {rows}, ready mean "
      f"{b.get('ready_mean', 0):.2f} (the stacked path really executed)")
PY

    stop_server
    survivors_check

    if [ -n "$PARAKEET" ] && [ -d "$PARAKEET" ]; then
        say ""
        say "-- the light model, on REST, same question --"
        start_server "$PARAKEET" "--threads 8 --cap 8" || exit 3
        python3 tools/bench/rest_load.py --host 127.0.0.1 --port "$PORT" \
            --ladder 1,4 --requests-per-stream 2 --clips $CLIPS_SHORT \
            --json "$RUN/t0-parakeet-rest.json" > "$RUN/t0-parakeet.txt" 2>&1
        PRC=$?
        tail -8 "$RUN/t0-parakeet.txt" | tee -a "$LOG"
        [ "$PRC" = "0" ] || note_fail "the Parakeet REST probe (rc=$PRC)"
        stop_server
    fi
}

# ------------------------------------------------------------------- tier 1
tier1() {
    say ""
    say "-- quality first, on an idle box: a CER measured under load is two questions --"
    if [ -n "$BANK_MANIFEST" ]; then
        BASE="configs/quality/$(basename "$MODEL")-$QUANT-offline.json"
        [ -f "$BASE" ] && BASE_ARG="--baseline $BASE" || BASE_ARG=""
        python3 tools/eval/cer_offline.py --model "$MODEL" --quant "$QUANT" \
            --manifest "$BANK_MANIFEST" --json "$RUN/t1-cer.json" $BASE_ARG \
            > "$RUN/t1-cer.txt" 2>&1
        CRC=$?
        tail -8 "$RUN/t1-cer.txt" | tee -a "$LOG"
        [ "$CRC" = "0" ] || note_fail "the offline CER pass (rc=$CRC; a regression, or no reference)"
    else
        say "  skipped: no manifest"
    fi

    say ""
    say "-- the knee, by bisection, on ONE warm server --"
    HI=$(python3 -c "
import os
n = os.cpu_count() or 8
print(max(16, min(128, n * 3)))")
    start_server "$MODEL" "--threads $((HI + 8)) --cap $((HI + 8)) $SERVER_ARGS" || exit 3
    EXTRA=""
    [ -n "$BANK_MANIFEST" ] && EXTRA="--transcripts $BANK_MANIFEST"
    python3 tools/bench/knee.py --port "$PORT" --lookahead "$LOOKAHEAD" \
        --clips $BANK_CLIPS --lo 4 --hi "$HI" --resolution 4 --repeat 2 \
        --out "$RUN/probes" --json "$RUN/knee.json" $EXTRA > "$RUN/knee.txt" 2>&1
    KRC=$?
    cat "$RUN/knee.txt" | tee -a "$LOG"
    stop_server
    survivors_check
    [ "$KRC" = "0" ] || { note_fail "no knee was found (see $RUN/knee.json)"; return; }
    KNEE=$(python3 -c "import json;print(json.load(open('$RUN/knee.json'))['knee']['highest_holding'])")
    say ""
    say "  screened knee: C=$KNEE. It is a SCREEN. Promote it with:"
    say "    tools/bench/tier.sh 2 -m $MODEL -C $KNEE"
}

# ------------------------------------------------------------------- tier 2
tier2() {
    [ "$CONC" -gt 0 ] 2>/dev/null || { say "tier 2 needs -C <streams> (from tier 1)"; exit 2; }
    [ -n "$BANK_MANIFEST" ] || {
        say "REFUSED: tier 2 without a reference manifest would repeat the 2026-09-20"
        say "         soak, which held for ten minutes and scored zero transcripts."
        say "         Run 'make fetch-stress-bank' first."
        exit 3; }
    say ""
    say "-- two identical soaks at C=$CONC: one is an anecdote, two that agree are a measurement --"
    start_server "$MODEL" "--threads $((CONC + 8)) --cap $((CONC + 8)) $SERVER_ARGS" || exit 3
    for rep in 1 2; do
        say ""
        say "  soak $rep of 2, ${SOAK_S}s"
        python3 tools/bench/stream_load.py --mode soak --streams "$CONC" \
            --duration "$SOAK_S" --warmup 30 --window 60 --seed 42 \
            --port "$PORT" --lookahead "$LOOKAHEAD" --clips $BANK_CLIPS \
            --transcripts "$BANK_MANIFEST" --json "$RUN/soak-$rep.json" \
            > "$RUN/soak-$rep.txt" 2>&1
        SRC=$?
        tail -24 "$RUN/soak-$rep.txt" | tee -a "$LOG"
        [ "$SRC" = "0" ] || note_fail "soak $rep did not pass its envelope (rc=$SRC)"
    done
    stop_server
    survivors_check

    say ""
    say "-- do the two runs agree? --"
    python3 tools/bench/compare_runs.py "$RUN/soak-1.json" "$RUN/soak-2.json" \
        --json "$RUN/agreement.json" > "$RUN/compare.txt" 2>&1
    ARC=$?
    cat "$RUN/compare.txt" | tee -a "$LOG"
    [ "$ARC" = "0" ] || note_fail "the two soaks did not agree (rc=$ARC)"
}

case "$TIER" in
    0) tier0 ;;
    1) tier1 ;;
    2) tier2 ;;
esac

say ""
if [ "$FAIL" = "0" ]; then
    say "=== tier $TIER PASSED — $RUN ==="
else
    say "=== tier $TIER FAILED — $RUN ==="
fi
exit "$FAIL"
