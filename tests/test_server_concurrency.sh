#!/bin/sh
# Server under CONCURRENT load, model-agnostic on purpose.
#
# Why separate from test_server.sh: that one asserts specific Italian/English
# transcripts and the model name, so it only runs with Nemotron — which means CI
# (where the model is parakeet-tdt_ctc-110m) never exercised the server at all.
# This checks what holds for ANY model: concurrent requests all answer with a
# non-empty transcript, and the adaptive-BLAS accounting comes back to rest.
#
# That last part is the one nothing else can catch: an inference_begin() without
# its inference_end() leaves BLAS capped for the whole process life — same
# output, just slower.
#
# Then the same requests through `--prefork 2`: every response must be
# byte-identical to the single-process one (a request never depends on which
# worker served it), and with one slot and no queue the third of three
# concurrent requests must be refused with a 503 the client can actually read
# (error.code server_at_capacity, Retry-After), never a reset.
#
# Usage: test_server_concurrency.sh [model_dir] [port] [n_concurrent]
# Exit: 0 ok, 1 fail, 77 skip (model missing).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
PORT="${2:-8207}"
N="${3:-4}"
WAV=tests/audio/test_en.wav
[ -f "$MODEL_DIR/mynah.json" ] || exit 77
[ -f "$WAV" ] || exit 77
[ -x ./mynah-asr-server ] || exit 77

./mynah-asr-server -m "$MODEL_DIR" -p "$PORT" --threads 4 --batch 4 2>/dev/null &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null' EXIT

ready=0
for i in $(seq 1 100); do
    if curl -sf "http://localhost:$PORT/v1/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.2
done
[ $ready -eq 1 ] || { echo "server-concurrency FAIL: server never became ready"; exit 1; }

fail=0
TMP=$(mktemp -d /tmp/mynah_asr_conc.XXXXXX) || exit 1
trap 'kill $SRV_PID 2>/dev/null; rm -rf "$TMP"' EXIT

# Field readers tolerant of whitespace after the colon: cJSON prints none, but an
# assertion that hinges on that would fail confusingly the day it changes.
num_field() { printf '%s' "$2" | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p"; }

# health at rest: the budget must be the full thread count before any load
rest=$(curl -s "http://localhost:$PORT/v1/health")
nth=$(num_field threads "$rest")
bud=$(num_field blas_budget "$rest")
if [ -n "$nth" ] && [ "$nth" = "$bud" ]; then
    echo "server-concurrency health-at-rest OK (budget $bud == threads $nth)"
else
    echo "server-concurrency health-at-rest FAIL: $rest"; fail=1
fi

# N concurrent transcriptions (wait ONLY on the curls: a bare wait would also
# wait on the server process, which never exits)
PIDS=""
i=1
while [ "$i" -le "$N" ]; do
    curl -s -F file=@"$WAV" -F language=auto \
        "http://localhost:$PORT/v1/audio/transcriptions" -o "$TMP/r$i.json" &
    PIDS="$PIDS $!"
    i=$((i + 1))
done
wait $PIDS

i=1
all_ok=1
while [ "$i" -le "$N" ]; do
    grep -qE '"text"[[:space:]]*:[[:space:]]*"[^"]' "$TMP/r$i.json" 2>/dev/null \
        || { all_ok=0; echo "  request $i: $(cat "$TMP/r$i.json" 2>/dev/null)"; }
    i=$((i + 1))
done
[ $all_ok -eq 1 ] && echo "server-concurrency $N-parallel OK" \
                  || { echo "server-concurrency $N-parallel FAIL"; fail=1; }

# and back to rest: every begin paired with an end
after=$(curl -s "http://localhost:$PORT/v1/health")
infl=$(num_field inflight "$after")
if [ "$infl" = "0" ]; then
    echo "server-concurrency inflight-at-rest OK"
else
    echo "server-concurrency inflight-at-rest FAIL (inflight='$infl'): $after"; fail=1
fi
bud2=$(num_field blas_budget "$after")
if [ -n "$nth" ] && [ "$nth" = "$bud2" ]; then
    echo "server-concurrency budget-restored OK (budget $bud2 == threads $nth)"
else
    echo "server-concurrency budget-restored FAIL: budget='$bud2' threads='$nth'"; fail=1
fi

kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null
i=1; while [ "$i" -le "$N" ]; do mv "$TMP/r$i.json" "$TMP/single$i.json"; i=$((i + 1)); done

# ---- the same requests through the prefork router -------------------------
PORT2=$((PORT + 1))
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT2" --threads 2 --batch 1 --prefork 2 --cap 2 2>"$TMP/prefork.log" &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null; rm -rf "$TMP"' EXIT
ready=0
for i in $(seq 1 100); do
    if curl -sf "http://localhost:$PORT2/v1/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.2
done
[ $ready -eq 1 ] || { echo "server-concurrency prefork FAIL: fleet never became ready"; cat "$TMP/prefork.log"; exit 1; }
PIDS=""
i=1
while [ "$i" -le "$N" ]; do
    curl -s -F file=@"$WAV" -F language=auto \
        "http://localhost:$PORT2/v1/audio/transcriptions" -o "$TMP/pf$i.json" &
    PIDS="$PIDS $!"
    i=$((i + 1))
done
wait $PIDS
i=1
same=1
while [ "$i" -le "$N" ]; do
    cmp -s "$TMP/single$i.json" "$TMP/pf$i.json" || { same=0; echo "  request $i differs under prefork: $(cat "$TMP/pf$i.json")"; }
    i=$((i + 1))
done
[ $same -eq 1 ] && echo "server-concurrency prefork byte-identical OK ($N requests)" \
                || { echo "server-concurrency prefork byte-identical FAIL"; fail=1; }
health=$(curl -s "http://localhost:$PORT2/v1/health")
case "$health" in
    *'"worker":'[0-9]*) echo "server-concurrency prefork health names its worker OK" ;;
    *) echo "server-concurrency prefork health FAIL: $health"; fail=1 ;;
esac
kill -TERM $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null
sleep 0.3
left=$(pgrep -f "mynah-asr-server -m $MODEL_DIR -p $PORT2" | wc -l | tr -d ' ')
[ "$left" = "0" ] && echo "server-concurrency prefork shutdown leaves no worker OK" \
                  || { echo "server-concurrency prefork shutdown FAIL: $left survivors"; pkill -9 -f "mynah-asr-server -m $MODEL_DIR -p $PORT2"; fail=1; }

# ---- refusal is a status, not a reset: 1 slot, no queue, 3 at once --------
PORT3=$((PORT + 2))
MYNAH_ASR_PREFORK_QUEUE=0 ./mynah-asr-server -m "$MODEL_DIR" -p "$PORT3" --threads 1 --batch 1 --prefork 1 --cap 1 2>"$TMP/refuse.log" &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null; rm -rf "$TMP"' EXIT
ready=0
for i in $(seq 1 100); do
    if curl -sf "http://localhost:$PORT3/v1/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.2
done
[ $ready -eq 1 ] || { echo "server-concurrency refusal FAIL: server never became ready"; exit 1; }
PIDS=""
for i in 1 2 3; do
    curl -s -o "$TMP/rf$i.body" -w '%{http_code}' -D "$TMP/rf$i.hdr" -F file=@"$WAV" -F language=auto \
        "http://localhost:$PORT3/v1/audio/transcriptions" > "$TMP/rf$i.code" &
    PIDS="$PIDS $!"
done
wait $PIDS
n503=0; n200=0; readable=1
for i in 1 2 3; do
    code=$(cat "$TMP/rf$i.code")
    if [ "$code" = "503" ]; then
        n503=$((n503 + 1))
        grep -q '"code":"server_at_capacity"' "$TMP/rf$i.body" || readable=0
        grep -qi 'retry-after' "$TMP/rf$i.hdr" || readable=0
    elif [ "$code" = "200" ]; then n200=$((n200 + 1)); fi
done
if [ $n200 -ge 1 ] && [ $n503 -ge 1 ] && [ $readable -eq 1 ]; then
    echo "server-concurrency refusal OK ($n200 served, $n503 refused with a readable 503 + Retry-After)"
else
    echo "server-concurrency refusal FAIL: served=$n200 refused=$n503 readable=$readable"; fail=1
    for i in 1 2 3; do echo "  $i: $(cat "$TMP/rf$i.code") $(head -c 100 "$TMP/rf$i.body")"; done
fi
kill -TERM $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null

exit $fail
