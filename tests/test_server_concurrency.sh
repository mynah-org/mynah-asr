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

exit $fail
