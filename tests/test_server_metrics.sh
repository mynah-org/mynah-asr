#!/bin/sh
# S3-3/S3-4: the server's observability surface, asserted rather than described.
#
# Model-agnostic on purpose (it runs on the 110m in CI) and REST-only: what is
# under test is the accounting, not the transcript, and an offline job moves
# every counter this test cares about.
#
# What it asserts, and why each one is here:
#   - the banner: [FLAGS], [EFFECTIVE-CONFIG], [SERVER-CONFIG] and [TOPOLOGY] on
#     stderr at start, unconditionally. ENGINEERING.md §5 -- a run whose banner
#     is missing is a run that cannot be quoted.
#   - /metrics and /v1/health AGREE: the same offline-job count, the same slot
#     cap. Two renderings of one snapshot that disagree is the bug this gate
#     exists to catch.
#   - audio_seconds_total ~= 4 x the clip. It is THE throughput unit, so a
#     counter that is merely non-zero is not enough: it has to be the right
#     number of seconds.
#   - a second bind on the metrics port FAILS. SO_REUSEPORT is deliberately
#     absent; two processes sharing the port would answer half the scrapes with
#     the wrong counters under the same labels.
#   - a burst of 20 scrapes hits the 5/s token bucket and gets 429s.
#   - SIGUSR1 produces a bracketed [DUMP] and does NOT kill the process (the
#     default action for SIGUSR1 is to terminate, and it did, once).
#   - under --prefork 2 the PARENT answers /metrics and names both workers.
#
# Usage: test_server_metrics.sh [model_dir] [port]
# Exit: 0 ok, 1 fail, 77 skip (model, binary or python3 missing).
MODEL_DIR="${1:-models/parakeet-tdt_ctc-110m}"
PORT="${2:-8217}"
MPORT=$((PORT + 1000))
WAV=tests/audio/test_en.wav
[ -f "$MODEL_DIR/mynah.json" ] || exit 77
[ -f "$WAV" ] || exit 77
[ -x ./mynah-asr-server ] || exit 77
command -v curl >/dev/null 2>&1 || exit 77
command -v python3 >/dev/null 2>&1 || exit 77

TMP=$(mktemp -d /tmp/mynah_asr_metrics.XXXXXX) || exit 1
SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT
fail=0

num_field() { printf '%s' "$2" | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p"; }
# One Prometheus sample by exact metric name (labels and all), value only.
metric() { grep "^$1" "$2" | head -1 | awk '{print $NF}'; }

wait_ready() {
    ready=0
    i=0
    while [ $i -lt 100 ]; do
        if curl -sf "http://127.0.0.1:$1/v1/health" >/dev/null 2>&1; then ready=1; break; fi
        sleep 0.2
        i=$((i + 1))
    done
    [ $ready -eq 1 ]
}

# ---- single process -------------------------------------------------------
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT" --threads 2 --batch 1 \
    --metrics-port "$MPORT" >"$TMP/srv.out" 2>"$TMP/srv.log" &
SRV_PID=$!
wait_ready "$PORT" || { echo "server-metrics FAIL: server never became ready"; cat "$TMP/srv.log"; exit 1; }

# 1. the banner
for line in '\[FLAGS\] v=1' '\[EFFECTIVE-CONFIG\] v=1' '\[SERVER-CONFIG\] v=1' '\[TOPOLOGY\] v=1'; do
    grep -qE "$line" "$TMP/srv.log" || { echo "server-metrics banner FAIL: no $line line"; fail=1; }
done
grep -q 'metrics_port=' "$TMP/srv.log" || { echo "server-metrics banner FAIL: the banner does not name the metrics port"; fail=1; }
[ $fail -eq 0 ] && echo "server-metrics banner OK ([FLAGS]/[EFFECTIVE-CONFIG]/[SERVER-CONFIG]/[TOPOLOGY])"

# 2. four REST requests
i=1
while [ "$i" -le 4 ]; do
    curl -s -F file=@"$WAV" -F language=auto \
        "http://127.0.0.1:$PORT/v1/audio/transcriptions" -o "$TMP/r$i.json"
    grep -qE '"text"[[:space:]]*:' "$TMP/r$i.json" || { echo "server-metrics FAIL: request $i: $(cat "$TMP/r$i.json")"; fail=1; }
    i=$((i + 1))
done

# 3. the scrape, against /v1/health
curl -s "http://127.0.0.1:$MPORT/metrics" -o "$TMP/m1.txt"
health=$(curl -s "http://127.0.0.1:$PORT/v1/health")
grep -q '^mynah_asr_build_info{' "$TMP/m1.txt" || { echo "server-metrics FAIL: no build_info in the scrape"; fail=1; }

m_off=$(metric 'mynah_asr_offline_jobs_total' "$TMP/m1.txt")
h_off=$(printf '%s' "$health" | sed -n 's/.*"offline"[^}]*"done"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
if [ "$m_off" = "4" ] && [ "$h_off" = "4" ]; then
    echo "server-metrics offline-jobs OK (metrics $m_off == health $h_off == 4 requests)"
else
    echo "server-metrics offline-jobs FAIL: metrics='$m_off' health='$h_off' (expected 4)"; fail=1
fi

m_cap=$(metric 'mynah_asr_slots_cap' "$TMP/m1.txt")
h_cap=$(printf '%s' "$health" | sed -n 's/.*"slots"[^}]*"cap"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
if [ -n "$m_cap" ] && [ "$m_cap" = "$h_cap" ]; then
    echo "server-metrics slots-cap OK ($m_cap on both surfaces)"
else
    echo "server-metrics slots-cap FAIL: metrics='$m_cap' health='$h_cap'"; fail=1
fi

# audio_seconds_total ~= 4 x the clip. The clip's length comes from the file,
# not from a constant: a hardcoded duration is a test that passes after someone
# replaces the fixture.
bytes=$(wc -c < "$WAV" | tr -d ' ')
m_audio=$(metric 'mynah_asr_audio_seconds_total' "$TMP/m1.txt")
ok=$(awk -v b="$bytes" -v got="$m_audio" 'BEGIN{
    dur = (b - 44) / 2 / 16000; want = 4 * dur;
    print (got > want * 0.97 && got < want * 1.03) ? 1 : 0; }')
if [ "$ok" = "1" ]; then
    echo "server-metrics audio-seconds OK ($m_audio s = 4 x the clip)"
else
    echo "server-metrics audio-seconds FAIL: got '$m_audio', 4 clips are $(awk -v b="$bytes" 'BEGIN{printf "%.3f", 4*(b-44)/2/16000}') s"; fail=1
fi

# sessions/steps/deltas are stream counters and MUST be 0 on a REST-only run:
# a non-zero one here would mean something is counting the wrong thing.
for m in mynah_asr_sessions_total mynah_asr_steps_total mynah_asr_deltas_total; do
    v=$(metric "$m" "$TMP/m1.txt")
    [ "$v" = "0" ] || { echo "server-metrics FAIL: $m = $v on a REST-only run, expected 0"; fail=1; }
done
# and the lag pair must be present and consistent with zero deltas
lc=$(metric 'mynah_asr_emission_lag_ms_count' "$TMP/m1.txt")
[ "$lc" = "0" ] || { echo "server-metrics FAIL: emission_lag_ms_count = $lc, expected 0"; fail=1; }
grep -q 'mynah_asr_emission_lag_over_ms_total{.*le="' "$TMP/m1.txt" \
    || { echo "server-metrics FAIL: no exact threshold counters"; fail=1; }
# No client-side latency may ever appear here. Comment lines are excluded on
# purpose: the page SAYS "no TTFP, no stall rate" in its own header, and a
# check that cannot tell a series from a sentence would fail on the disclaimer.
if grep -v '^#' "$TMP/m1.txt" | grep -qiE 'ttfp|stall'; then
    echo "server-metrics FAIL: a client-side latency leaked into /metrics"; fail=1
else
    echo "server-metrics honest-boundary OK (no TTFP, no stall rate)"
fi

# 4. a second bind on the metrics port must fail
if python3 - "$MPORT" <<'PY'
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
    s.listen(1)
except OSError:
    sys.exit(1)      # refused, as it must be
sys.exit(0)          # bound: SO_REUSEPORT got in somehow
PY
then
    echo "server-metrics double-bind FAIL: a second process bound the live metrics port"; fail=1
else
    echo "server-metrics double-bind OK (the second bind was refused)"
fi

# 5. the token bucket: refill first, then 20 scrapes back to back
sleep 2
n200=0; n429=0
i=1
while [ "$i" -le 20 ]; do
    c=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$MPORT/metrics")
    [ "$c" = "200" ] && n200=$((n200 + 1))
    [ "$c" = "429" ] && n429=$((n429 + 1))
    i=$((i + 1))
done
if [ "$n429" -ge 1 ] && [ "$n200" -ge 5 ]; then
    echo "server-metrics rate-limit OK ($n200 served, $n429 refused with 429)"
else
    echo "server-metrics rate-limit FAIL: 200=$n200 429=$n429 (expected a burst then 429s)"; fail=1
fi

# 6. SIGUSR1: a bracketed dump, and the process survives it
kill -USR1 "$SRV_PID" 2>/dev/null
sleep 0.8
b=$(grep -cE '^\[DUMP\] v=1 worker=-?[0-9]+ seq=[0-9]+ begin' "$TMP/srv.log")
e=$(grep -cE '^\[DUMP\] v=1 worker=-?[0-9]+ seq=[0-9]+ end' "$TMP/srv.log")
if [ "$b" = "1" ] && [ "$e" = "1" ] && kill -0 "$SRV_PID" 2>/dev/null; then
    echo "server-metrics sigusr1 OK (one bracketed [DUMP], the process survived)"
else
    echo "server-metrics sigusr1 FAIL: begin=$b end=$e alive=$(kill -0 "$SRV_PID" 2>/dev/null && echo yes || echo no)"; fail=1
fi

kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=""

# ---- prefork: the ROUTER answers for the fleet ----------------------------
PORT2=$((PORT + 2))
MPORT2=$((MPORT + 2))
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT2" --threads 2 --batch 1 \
    --prefork 2 --cap 2 --metrics-port "$MPORT2" >"$TMP/pf.out" 2>"$TMP/pf.log" &
SRV_PID=$!
wait_ready "$PORT2" || { echo "server-metrics prefork FAIL: fleet never became ready"; cat "$TMP/pf.log"; exit 1; }

i=1
while [ "$i" -le 4 ]; do
    curl -s -F file=@"$WAV" -F language=auto \
        "http://127.0.0.1:$PORT2/v1/audio/transcriptions" -o /dev/null
    i=$((i + 1))
done
curl -s "http://127.0.0.1:$MPORT2/metrics" -o "$TMP/m2.txt"
labels=$(grep -o 'worker="[0-9]*"' "$TMP/m2.txt" | sort -u | wc -l | tr -d ' ')
if [ "$labels" = "2" ]; then
    echo "server-metrics prefork OK (the router answers for both workers)"
else
    echo "server-metrics prefork FAIL: $labels distinct worker labels, expected 2"
    head -30 "$TMP/m2.txt"; fail=1
fi
grep -q '^mynah_asr_refused_total{code=' "$TMP/m2.txt" \
    || { echo "server-metrics prefork FAIL: the router does not export its refusals by code"; fail=1; }
# Every worker in a prefork fleet printed its own TOPOLOGY line.
#
# Polled, not read once: readiness means the FIRST worker answered /v1/health,
# which says nothing about the second, and a worker prints this line after it
# pins itself. On a sanitized build everything is several times slower and the
# second line lands well after the first request is served. Waiting for a line
# the server does print is not the same as loosening the assertion: the count
# must still reach 2, and a fleet that only ever starts one worker still fails.
wait_lines() {  # pattern file count timeout_tenths
    j=0
    while [ $j -lt "$4" ]; do
        n=$(grep -cE "$1" "$2" 2>/dev/null || echo 0)
        [ "$n" -ge "$3" ] && { echo "$n"; return 0; }
        sleep 0.1
        j=$((j + 1))
    done
    grep -cE "$1" "$2" 2>/dev/null || echo 0
}
t=$(wait_lines '^\[TOPOLOGY\] v=1 worker=[01] ' "$TMP/pf.log" 2 300)
[ "$t" = "2" ] || { echo "server-metrics prefork FAIL: $t [TOPOLOGY] lines, expected 2"; fail=1; }

# SIGUSR1 on the parent reaches both workers, and nobody dies
kill -USR1 "$SRV_PID" 2>/dev/null
b=$(wait_lines '^\[DUMP\] v=1 worker=[01] seq=[0-9]+ begin' "$TMP/pf.log" 2 300)
alive=$(pgrep -f "mynah-asr-server -m $MODEL_DIR -p $PORT2" | wc -l | tr -d ' ')
if [ "$b" = "2" ] && [ "$alive" = "3" ]; then
    echo "server-metrics prefork sigusr1 OK (both workers dumped, all 3 processes alive)"
else
    echo "server-metrics prefork sigusr1 FAIL: dumps=$b processes=$alive (expected 2 and 3)"; fail=1
fi

kill -TERM "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=""
sleep 0.3
left=$(pgrep -f "mynah-asr-server -m $MODEL_DIR -p $PORT2" | wc -l | tr -d ' ')
[ "$left" = "0" ] || { echo "server-metrics prefork FAIL: $left survivors"; pkill -9 -f "mynah-asr-server -m $MODEL_DIR -p $PORT2"; fail=1; }

exit $fail
