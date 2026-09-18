#!/bin/sh
# WebSocket protocol v2 and the admission ladder inside a worker (S2-3 / S2-5).
#
# What it proves, in order:
#   1. THREE UTTERANCES ON ONE SOCKET, separated by finalize/reset, each
#      byte-identical to `mynah-asr transcribe --quant int8 --lang auto` of the
#      same clip, with `seq` continuing across all three. A control channel that
#      changed a transcript would be worse than no control channel.
#   2. An unknown control message is an `error` frame and the session SURVIVES.
#   3. A reset naming a language this model does not hold is an `error` frame
#      and the session survives that too.
#   4. An unknown query key, and an unserved `rate=`, are 400s BEFORE the
#      upgrade, naming the accepted set.
#   5. A stream that sends more than `--max-audio-seconds` is told `audio_limit`
#      and then FINALISED: the audio already accepted still gets its transcript.
#   6. A client that says nothing at all is cancelled at `--idle-ms` with an
#      `idle_timeout` frame it can read, while a client that answers the
#      server's pings is not (a pong is a frame like any other).
#   7. The server pings on its own at `--ping-ms`.
#   8. `--cap 1` plus a second WebSocket is a 503 before the upgrade, with a
#      readable body and Retry-After -- not an ECONNRESET.
#   9. SIGTERM with a live stream: an `error shutting_down` frame, exit 0, and
#      no survivors.
#
# Usage: test_server_protocol.sh [model_dir] [port]
# Exit: 0 ok, 1 fail, 77 skip (model, binaries or python missing).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
PORT="${2:-8253}"
CLIPS="tests/audio/test_it.wav tests/audio/test_en.wav tests/audio/test_de.wav"

[ -f "$MODEL_DIR/mynah.json" ] || exit 77
[ -x ./mynah-asr-server ] || exit 77
[ -x ./mynah-asr ] || exit 77
command -v python3 >/dev/null 2>&1 || exit 77
for c in $CLIPS; do [ -f "$c" ] || exit 77; done

TMP=$(mktemp -d /tmp/mynah_asr_proto.XXXXXX) || exit 1
SRV_PID=""
HOLD_PID=""
cleanup() {
    [ -n "$HOLD_PID" ] && kill "$HOLD_PID" 2>/dev/null
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT
fail=0

wait_ready() { # port
    i=0
    while [ "$i" -lt 150 ]; do
        curl -sf "http://localhost:$1/v1/health" >/dev/null 2>&1 && return 0
        sleep 0.2
        i=$((i + 1))
    done
    return 1
}

probe() { # label, ws_probe.py args...
    label=$1
    shift
    if python3 tests/ws_probe.py --host localhost "$@"; then
        echo "server-protocol $label OK"
    else
        echo "server-protocol $label FAIL"
        fail=1
    fi
}

stop_server() {
    [ -n "$SRV_PID" ] || return 0
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null      # ONLY this pid: never a bare wait (ENGINEERING 10)
    SRV_PID=""
}

# ---- the reference: what the offline CLI says about each clip ---------------
# Built here rather than committed: it is the SAME binary's answer the streams
# have to match, and a stale committed file would test the file.
: > "$TMP/ref.tsv"
for c in $CLIPS; do
    printf '%s\t' "$c" >> "$TMP/ref.tsv"
    ./mynah-asr transcribe -m "$MODEL_DIR" -i "$c" --quant int8 --lang auto 2>/dev/null \
        | tr -d '\n' >> "$TMP/ref.tsv"
    printf '\n' >> "$TMP/ref.tsv"
done
python3 - "$TMP/ref.tsv" "$TMP/ref.json" <<'PY'
import json, sys
ref = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.rstrip("\n")
    if not line:
        continue
    k, _, v = line.partition("\t")
    ref[k] = v
if not all(ref.values()):
    sys.exit("the CLI produced an empty reference for " +
             ", ".join(k for k, v in ref.items() if not v))
json.dump(ref, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
PY
if [ $? -ne 0 ]; then echo "server-protocol reference FAIL"; exit 1; fi

# ---- 1-4: the control channel and the pre-upgrade 400s ---------------------
# No ping and a generous idle here: a probe that blasts a clip and then waits
# for `done` says nothing for as long as the decode takes, and an idle cap
# shorter than that would be measuring the test's own pacing.
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT" --threads 4 --batch 1 --quant int8 \
    --cap 2 --idle-ms 30000 --ping-ms 0 > "$TMP/srv.log" 2>&1 &
SRV_PID=$!
wait_ready "$PORT" || { echo "server-protocol FAIL: server never became ready"; cat "$TMP/srv.log"; exit 1; }

probe three-utterances-one-socket --port "$PORT" utterances \
    --clips $CLIPS --reference "$TMP/ref.json"

probe unknown-control --port "$PORT" control \
    --clip tests/audio/test_it.wav --reference "$TMP/ref.json" \
    --message teleport --expect-code unknown_control

probe reset-bad-lang --port "$PORT" bad-lang \
    --clip tests/audio/test_it.wav --reference "$TMP/ref.json" --bad zz-ZZ

probe unknown-query-key --port "$PORT" http \
    --path '/v1/audio/stream?lang=auto&lookahed=3' --expect-status 400 \
    --expect-body unknown_query_parameter lookahead format rate

probe unsupported-rate --port "$PORT" http \
    --path '/v1/audio/stream?lang=auto&rate=44100' --expect-status 400 \
    --expect-body unsupported_rate 16000

stop_server

# ---- 5-7: the audio cap, the idle cap and the server's pings ---------------
# One server for the three, because together they state the contract: the audio
# probe is cut off by seconds of audio, the ping probe answers and stays (a pong
# is activity), the idle probe never answers and goes.
PORT2=$((PORT + 1))
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT2" --threads 4 --batch 1 --quant int8 \
    --cap 2 --idle-ms 2000 --ping-ms 1000 --max-audio-seconds 1 > "$TMP/srv2.log" 2>&1 &
SRV_PID=$!
wait_ready "$PORT2" || { echo "server-protocol idle FAIL: server never became ready"; cat "$TMP/srv2.log"; exit 1; }

probe audio-limit --port "$PORT2" hold \
    --clip tests/audio/test_de.wav --pace 1.0 --hold 15 --expect-code audio_limit
probe idle-timeout --port "$PORT2" idle --wait 8
probe server-pings --port "$PORT2" ping --wait 3 --min-pings 2

stop_server

# ---- 8: --cap 1, the second WebSocket is refused before the upgrade --------
PORT3=$((PORT + 2))
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT3" --threads 4 --batch 1 --quant int8 \
    --cap 1 --idle-ms 30000 --ping-ms 0 > "$TMP/srv3.log" 2>&1 &
SRV_PID=$!
wait_ready "$PORT3" || { echo "server-protocol cap FAIL: server never became ready"; cat "$TMP/srv3.log"; exit 1; }

python3 tests/ws_probe.py --host localhost --port "$PORT3" hold \
    --hold 12 --ready-file "$TMP/held" > "$TMP/hold.log" 2>&1 &
HOLD_PID=$!
i=0
while [ "$i" -lt 60 ] && [ ! -f "$TMP/held" ]; do sleep 0.1; i=$((i + 1)); done
if [ -f "$TMP/held" ]; then
    probe cap-1-second-stream-503 --port "$PORT3" http \
        --path '/v1/audio/stream?lang=auto&lookahead=3' --expect-status 503 \
        --expect-body server_at_capacity --expect-header Retry-After
else
    echo "server-protocol cap-1-second-stream-503 FAIL: the first stream never took its slot"
    fail=1
fi
kill $HOLD_PID 2>/dev/null; HOLD_PID=""
stop_server

# ---- 9: SIGTERM with a live stream ----------------------------------------
PORT4=$((PORT + 3))
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT4" --threads 4 --batch 1 --quant int8 \
    --cap 2 --idle-ms 30000 --ping-ms 0 > "$TMP/srv4.log" 2>&1 &
SRV_PID=$!
wait_ready "$PORT4" || { echo "server-protocol sigterm FAIL: server never became ready"; cat "$TMP/srv4.log"; exit 1; }

python3 tests/ws_probe.py --host localhost --port "$PORT4" hold \
    --clip tests/audio/test_de.wav --pace 1.0 --hold 20 \
    --ready-file "$TMP/live" --expect-code shutting_down > "$TMP/sigterm.log" 2>&1 &
HOLD_PID=$!
i=0
while [ "$i" -lt 60 ] && [ ! -f "$TMP/live" ]; do sleep 0.1; i=$((i + 1)); done
sleep 1.5                            # let the stream actually run before the signal
kill -TERM $SRV_PID 2>/dev/null
wait $HOLD_PID 2>/dev/null           # ONLY these two pids, by name
hold_rc=$?
wait $SRV_PID 2>/dev/null
srv_rc=$?
HOLD_PID=""; SRV_PID=""
if [ "$hold_rc" -eq 0 ]; then
    echo "server-protocol sigterm-shutting-down-frame OK"
else
    echo "server-protocol sigterm-shutting-down-frame FAIL"; cat "$TMP/sigterm.log"; fail=1
fi
if [ "$srv_rc" -eq 0 ]; then
    echo "server-protocol sigterm-exit-0 OK"
else
    echo "server-protocol sigterm-exit-0 FAIL: exit $srv_rc"; cat "$TMP/srv4.log"; fail=1
fi
sleep 0.5
left=$(pgrep -f "mynah-asr-server -m $MODEL_DIR -p $PORT4" | wc -l | tr -d ' ')
if [ "$left" = "0" ]; then
    echo "server-protocol sigterm-no-survivors OK"
else
    echo "server-protocol sigterm-no-survivors FAIL: $left survivors"
    pkill -9 -f "mynah-asr-server -m $MODEL_DIR -p $PORT4"
    fail=1
fi

exit $fail
