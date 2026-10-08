#!/bin/sh
# REST requests whose client goes away, against a live server: the offline
# half of the zombie question (S12-19 covered the WebSocket half).
#
# Model-agnostic and REST-only, so it runs on the 110m in CI (`make test`):
#   - rest-gone-queued: a request whose client closes while it waits behind
#     another is dropped before inference (offline.peer_gone +1) and the model
#     only ever runs the other one;
#   - rest-stalled-body: headers and part of a body, then silence: the server
#     answers 400 incomplete_body and frees the HTTP thread within --idle-ms
#     instead of holding it for ever.
#
# Usage: test_server_rest_faults.sh [model_dir] [port]
# Exit: 0 ok, 1 fail, 77 skip (model, binary, curl or python3 missing).
MODEL_DIR="${1:-models/parakeet-tdt_ctc-110m}"
PORT="${2:-8241}"
CLIP=tests/audio/test_en.wav
[ -f "$MODEL_DIR/mynah.json" ] || exit 77
[ -f "$CLIP" ] || exit 77
[ -x ./mynah-asr-server ] || exit 77
command -v curl >/dev/null 2>&1 || exit 77
command -v python3 >/dev/null 2>&1 || exit 77

TMP=$(mktemp -d /tmp/mynah_asr_rest_faults.XXXXXX) || exit 1
SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

# --batch 1: the queued request cannot ride in the running one's batch.
IDLE=2000
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT" --threads 8 --batch 1 \
    --idle-ms $IDLE --ping-ms 0 > "$TMP/srv.log" 2>&1 &
SRV_PID=$!
ready=0
for i in $(seq 1 150); do
    if curl -sf "http://localhost:$PORT/v1/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.2
done
[ $ready -eq 1 ] || { echo "server-rest-faults FAIL: server never became ready"; cat "$TMP/srv.log"; exit 1; }

python3 tests/fault_probe.py --port "$PORT" --clip "$CLIP" --idle-ms $IDLE \
    rest-gone-queued rest-stalled-body
rc=$?

kill -TERM "$SRV_PID" 2>/dev/null
wait "$SRV_PID" 2>/dev/null          # ONLY this pid: never a bare wait (ENGINEERING 10)
srv_rc=$?
SRV_PID=""
if [ $srv_rc -ne 0 ]; then
    echo "server-rest-faults FAIL: server exit $srv_rc"; tail -20 "$TMP/srv.log"; rc=1
fi
if grep -q "runtime error:" "$TMP/srv.log"; then
    echo "server-rest-faults FAIL: undefined behaviour"; grep "runtime error:" "$TMP/srv.log" | head; rc=1
fi
[ $rc -eq 0 ] && echo "server-rest-faults OK" || echo "server-rest-faults FAIL"
exit $rc
