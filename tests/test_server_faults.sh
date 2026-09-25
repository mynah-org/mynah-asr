#!/bin/sh
# Provoked failures against a live server, and the invariants each must keep
# (S12-20): the slot comes back, exactly one outcome counter moves, the model
# stops working for a client that is gone, the books balance, a neighbour's
# transcript does not change, and RSS does not grow with the aborts.
#
# The assertions live in tests/fault_probe.py, next to the bytes that provoke
# them; this script only builds the reference, starts the server and reports.
#
# Usage: test_server_faults.sh [model_dir] [port]
# Exit: 0 ok, 1 fail, 77 skip (model missing).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
# Any cache-aware streaming model will do: nothing below asserts a model-specific
# transcript, only that a stream equals the same binary's offline answer.
PORT="${2:-8231}"
CLIP=tests/audio/test_en.wav
[ -f "$MODEL_DIR/mynah.json" ] || exit 77
[ -f "$CLIP" ] || exit 77
[ -x ./mynah-asr-server ] && [ -x ./mynah-asr ] || exit 77

TMP=$(mktemp -d /tmp/mynah_asr_faults.XXXXXX) || exit 1
SRV_PID=""
cleanup() {
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

CAP=4; IDLE=2000; MAXF=65536
# --threads is the HTTP thread count: kept above --cap, or the stream past the
# cap waits in accept() for a thread instead of reading its 503.
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT" --threads 8 --batch 4 --quant int8 \
    --cap $CAP --idle-ms $IDLE --ping-ms 0 --max-frame-bytes $MAXF > "$TMP/srv.log" 2>&1 &
SRV_PID=$!
ready=0
for i in $(seq 1 150); do
    if curl -sf "http://localhost:$PORT/v1/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.2
done
[ $ready -eq 1 ] || { echo "server-faults FAIL: server never became ready"; cat "$TMP/srv.log"; exit 1; }

# A model with no language prompt (the English-only EOU model) refuses any
# `lang` before the upgrade; ask the server rather than guessing from the config.
LANG_Q=auto
if python3 tests/ws_probe.py --host localhost --port "$PORT" http \
        --path '/v1/audio/stream?lang=auto' --expect-status 400 \
        --expect-body language_not_served >/dev/null 2>&1; then
    LANG_Q=
fi

# The reference is the offline CLI's answer, built here from the same binary.
ref=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$CLIP" --quant int8 --lang auto 2>/dev/null | tr -d '\n')
[ -n "$ref" ] || { echo "server-faults FAIL: empty CLI reference"; exit 1; }
python3 -c 'import json,sys; json.dump({sys.argv[1]: sys.argv[2]}, open(sys.argv[3], "w"), ensure_ascii=False)' \
    "$CLIP" "$ref" "$TMP/ref.json"

rc=0
if [ "$FAULT_CASES" != "worker-kill" ]; then
    python3 tests/fault_probe.py --port "$PORT" --clip "$CLIP" --reference "$TMP/ref.json" \
        --lang "$LANG_Q" --cap $CAP --idle-ms $IDLE --max-frame-bytes $MAXF \
        --server-pid "$SRV_PID" ${FAULT_CASES:-suite}
    rc=$?
fi

# macOS: every allocation still reachable after ~70 aborted sessions must be
# reachable ON PURPOSE. `leaks` on the live process, before the shutdown frees
# the table, so a slot or writer lost by an abort path shows up here.
if [ "${FAULT_LEAKS:-0}" = 1 ] && command -v leaks >/dev/null 2>&1; then
    leaks "$SRV_PID" > "$TMP/leaks.txt" 2>&1
    if grep -q " 0 leaks for 0 total leaked bytes" "$TMP/leaks.txt"; then
        echo "server-faults leaks OK (0 leaks in the live server)"
    else
        echo "server-faults leaks FAIL"; grep -E "leaks for|Leak:" "$TMP/leaks.txt" | head -20; rc=1
    fi
fi

kill -TERM "$SRV_PID" 2>/dev/null
wait "$SRV_PID" 2>/dev/null          # ONLY this pid: never a bare wait (ENGINEERING 10)
srv_rc=$?
SRV_PID=""
if [ $srv_rc -ne 0 ]; then
    echo "server-faults FAIL: server exit $srv_rc"; tail -20 "$TMP/srv.log"; rc=1
fi
# A sanitizer build (make ubsan's flags) reports here; a plain build never does.
if grep -q "runtime error:" "$TMP/srv.log"; then
    echo "server-faults FAIL: undefined behaviour"; grep "runtime error:" "$TMP/srv.log" | head; rc=1
fi
if grep -q "did not finish" "$TMP/srv.log"; then
    echo "server-faults FAIL: a slot was abandoned"; grep "did not finish" "$TMP/srv.log"; rc=1
fi

# ---- a worker process dies under a live stream (prefork) --------------------
if [ -z "$FAULT_CASES" ] || [ "$FAULT_CASES" = "worker-kill" ]; then
PORT2=$((PORT + 1)); MPORT=$((PORT + 2))
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT2" --prefork 2 --prefork-threads 2 --threads 4 \
    --cap 2 --idle-ms 5000 --ping-ms 0 --metrics-port "$MPORT" > "$TMP/pf.log" 2>&1 &
SRV_PID=$!
ready=0
for i in $(seq 1 150); do
    if curl -sf "http://localhost:$MPORT/metrics" 2>/dev/null | grep -q 'mynah_asr_worker_up{.*} 1'; then
        n=$(curl -sf "http://localhost:$MPORT/metrics" | grep -c 'mynah_asr_worker_up{.*} 1')
        [ "$n" -ge 2 ] && { ready=1; break; }
    fi
    sleep 0.2
done
if [ $ready -eq 1 ]; then
    python3 tests/fault_probe.py --port "$PORT2" --clip "$CLIP" --reference "$TMP/ref.json" \
        --lang "$LANG_Q" --server-pid "$SRV_PID" --metrics-port "$MPORT" worker-kill || rc=1
else
    echo "server-faults FAIL: prefork server never became ready"; tail -20 "$TMP/pf.log"; rc=1
fi
kill -TERM "$SRV_PID" 2>/dev/null
wait "$SRV_PID" 2>/dev/null
SRV_PID=""
grep -E "lost [0-9]+ connection|lost=" "$TMP/pf.log"
fi
[ $rc -eq 0 ] && echo "server-faults OK" || echo "server-faults FAIL"
exit $rc
