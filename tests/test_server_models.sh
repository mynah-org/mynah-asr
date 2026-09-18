#!/bin/sh
# SEVERAL MODELS IN ONE FLEET (S2-6): one server, two worker groups, routing by
# model.
#
# What it proves, in order, on ONE server started with two `--model` groups --
# Nemotron (streaming, int8) and the 110m GGUF (offline-only) -- and `--prefork 2`:
#
#   1. /v1/models lists EXACTLY the two configured names, from the actual table
#      rather than a hard-coded one.
#   2. A REST transcription addressed to each group comes back with THAT model's
#      own transcript, byte-identical to what `mynah-asr transcribe` says with
#      the same model. That byte-identity is the whole proof that the request
#      reached the right weights: the two packs disagree about this clip, so a
#      mis-route cannot pass. Both addressing shapes are exercised -- the query
#      `?model=` and the multipart `model` field.
#   3. A WebSocket to the streaming group streams and its text is byte-identical
#      to the CLI's streaming answer.
#   4. A WebSocket to the OFFLINE group is refused 400 `model_not_streaming`,
#      before the upgrade, with a body the client can read.
#   5. An unknown model name is refused 404 `model_not_found` with the accepted
#      set named -- and with NO Retry-After, because retrying will fail
#      identically for as long as the server runs.
#   6. CAPACITY IS PER GROUP: the offline group has one slot and no admission
#      queue, so of three requests fired at it at once one is served and the
#      rest get a readable 503 `server_at_capacity` -- while a request to the
#      streaming group in that same moment is served. A free slot in another
#      group's worker is not capacity for this request.
#   7. SIGTERM leaves no worker behind.
#
# Usage: test_server_models.sh [streaming_model_dir] [offline_model_dir] [port]
# Exit: 0 ok, 1 fail, 77 skip (either model, the binaries or python3 missing).
STREAM_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
OFFLINE_DIR="${2:-models/parakeet-tdt_ctc-110m-gguf}"
PORT="${3:-8231}"
WAV=tests/audio/test_en.wav

[ -f "$STREAM_DIR/mynah.json" ] || exit 77
[ -f "$OFFLINE_DIR/mynah.json" ] || exit 77
[ -f "$WAV" ] || exit 77
[ -x ./mynah-asr-server ] || exit 77
[ -x ./mynah-asr ] || exit 77
command -v python3 >/dev/null 2>&1 || exit 77
command -v curl >/dev/null 2>&1 || exit 77

TMP=$(mktemp -d /tmp/mynah_asr_models.XXXXXX) || exit 1
SRV_PID=""
PIDS=""
cleanup() {
    [ -n "$PIDS" ] && kill $PIDS 2>/dev/null
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT
fail=0

# ---- the references: what each model says about this clip, from the CLI ------
# Built here and not committed: it is the SAME binary's answer that the server
# has to match, and a stale committed file would test the file.
./mynah-asr transcribe -m "$STREAM_DIR" -i "$WAV" --quant int8 --lang auto \
    2>/dev/null | tr -d '\n' > "$TMP/ref_stream.txt"
./mynah-asr transcribe -m "$OFFLINE_DIR" -i "$WAV" --lang auto \
    2>/dev/null | tr -d '\n' > "$TMP/ref_offline.txt"
./mynah-asr stream -m "$STREAM_DIR" -i "$WAV" --quant int8 --lang auto \
    2>/dev/null | tr -d '\n' > "$TMP/ref_ws.txt"
for f in ref_stream ref_offline ref_ws; do
    [ -s "$TMP/$f.txt" ] || { echo "server-models FAIL: the CLI produced an empty $f"; exit 1; }
done
# The gate is only meaningful if the two packs actually disagree about this
# clip: identical references would let a mis-route pass unnoticed. Refuse to
# report a pass we could not have failed (ENGINEERING.md §7).
if cmp -s "$TMP/ref_stream.txt" "$TMP/ref_offline.txt"; then
    echo "server-models SKIP-ish: the two models give the SAME transcript for $WAV,"
    echo "  so routing cannot be proven by identity on this clip. Use clips that differ."
    exit 77
fi

# ---- one server, two groups -------------------------------------------------
# `--cap 1` on the offline group plus MYNAH_ASR_PREFORK_QUEUE=0 is what makes
# phase 6 deterministic: one slot and no admission queue, so concurrent requests
# beyond the first are refused rather than merely delayed. `--cap 2` on the
# streaming group leaves it room to answer in the same moment.
MYNAH_ASR_PREFORK_QUEUE=0 \
./mynah-asr-server \
    --model streaming="$STREAM_DIR":quant=int8:workers=1:cap=2 \
    --model offline="$OFFLINE_DIR":workers=1:cap=1 \
    --default streaming \
    --prefork 2 --threads 2 --batch 1 -p "$PORT" > "$TMP/srv.log" 2>&1 &
SRV_PID=$!

i=0
ready=0
while [ "$i" -lt 250 ]; do
    if curl -sf "http://localhost:$PORT/v1/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.2
    i=$((i + 1))
done
[ "$ready" = "1" ] || { echo "server-models FAIL: the fleet never became ready"; cat "$TMP/srv.log"; exit 1; }

# ---- 1. /v1/models is the table ---------------------------------------------
curl -s "http://localhost:$PORT/v1/models" > "$TMP/models.json"
if python3 - "$TMP/models.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ids = [m["id"] for m in d.get("data", [])]
if sorted(ids) != ["offline", "streaming"]:
    sys.exit("expected exactly ['offline','streaming'], got %r" % ids)
by = {m["id"]: m for m in d["data"]}
if not by["streaming"].get("streaming"):
    sys.exit("the streaming group is not advertised as streaming")
if by["offline"].get("streaming"):
    sys.exit("the offline group is advertised as streaming")
if not by["streaming"].get("default"):
    sys.exit("--default streaming was not reflected")
print("    /v1/models: %s (default=streaming)" % ", ".join(sorted(ids)))
PY
then
    echo "server-models models-endpoint OK"
else
    echo "server-models models-endpoint FAIL"; fail=1
fi

# ---- 2. REST to each group, byte-identical to that model's CLI ---------------
# Two addressing shapes, because both are contracts: the query, which the router
# reads out of the request line, and the multipart field, which it reads out of
# the body. `model` is sent BEFORE `file` on purpose -- that is the documented
# way to keep it inside the prefix the router classifies on.
curl -s -F file=@"$WAV" -F language=auto \
    "http://localhost:$PORT/v1/audio/transcriptions?model=streaming" > "$TMP/r_stream.json"
curl -s -F model=offline -F file=@"$WAV" -F language=auto \
    "http://localhost:$PORT/v1/audio/transcriptions" > "$TMP/r_offline.json"
curl -s -F file=@"$WAV" -F language=auto \
    "http://localhost:$PORT/v1/audio/transcriptions" > "$TMP/r_default.json"

extract() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("text",""),end="")' "$1"; }
t_stream=$(extract "$TMP/r_stream.json")
t_offline=$(extract "$TMP/r_offline.json")
t_default=$(extract "$TMP/r_default.json")
want_stream=$(cat "$TMP/ref_stream.txt")
want_offline=$(cat "$TMP/ref_offline.txt")

if [ "$t_stream" = "$want_stream" ]; then
    echo "server-models rest-streaming-group OK (byte-identical to its own CLI)"
else
    echo "server-models rest-streaming-group FAIL"
    echo "  got:  $t_stream"
    echo "  want: $want_stream"
    fail=1
fi
if [ "$t_offline" = "$want_offline" ]; then
    echo "server-models rest-offline-group OK (multipart field, byte-identical)"
else
    echo "server-models rest-offline-group FAIL"
    echo "  got:  $t_offline"
    echo "  want: $want_offline"
    fail=1
fi
if [ "$t_default" = "$want_stream" ]; then
    echo "server-models rest-default-group OK (no model named -> --default)"
else
    echo "server-models rest-default-group FAIL: '$t_default'"; fail=1
fi

# ---- 3. a WebSocket to the streaming group ----------------------------------
python3 - "$TMP/ref_ws.txt" "$TMP/wsref.json" "$WAV" <<'PY'
import json, sys
json.dump({sys.argv[3]: open(sys.argv[1], encoding="utf-8").read()},
          open(sys.argv[2], "w"), ensure_ascii=False)
PY
python3 tools/bench/stream_load.py --host localhost --port "$PORT" \
    --streams 1 --repeat 1 --clips "$WAV" --model streaming \
    --reference "$TMP/wsref.json" --json "$TMP/ws.json" > "$TMP/ws.txt" 2>&1
if python3 - "$TMP/ws.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d["summary"]
bad = []
if s["counts"]["errors"]:
    bad.append("%d errors" % s["counts"]["errors"])
if s["reference_fail"]:
    bad.append("differs from the CLI: %s" % s["reference_fail"])
if s["counts"]["ok"] != 1:
    bad.append("%d/1 utterances" % s["counts"]["ok"])
if bad:
    sys.exit("; ".join(bad))
PY
then
    echo "server-models ws-streaming-group OK (byte-identical to the CLI's stream)"
else
    echo "server-models ws-streaming-group FAIL"; sed -n '1,20p' "$TMP/ws.txt"; fail=1
fi

# ---- 4/5. the two refusals, read off the wire -------------------------------
cat > "$TMP/wsprobe.py" <<'PY'
# One WebSocket upgrade, the raw response. No library: the point is exactly what
# the server put on the socket before (or instead of) the 101.
import base64, os, socket, sys
host, port, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
s = socket.create_connection((host, port), timeout=15)
key = base64.b64encode(os.urandom(16)).decode()
s.sendall(("GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\n"
           "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
           "Sec-WebSocket-Version: 13\r\n\r\n" % (path, host, port, key)).encode())
buf = b""
try:
    while b"\r\n\r\n" not in buf:
        c = s.recv(4096)
        if not c:
            break
        buf += c
    s.settimeout(2.0)
    while True:
        c = s.recv(4096)
        if not c:
            break
        buf += c
except OSError:
    pass
s.close()
sys.stdout.write(buf.decode("utf-8", "replace"))
PY

python3 "$TMP/wsprobe.py" localhost "$PORT" "/v1/audio/stream?model=offline" > "$TMP/ws_offline.txt"
if grep -q "^HTTP/1.1 400 " "$TMP/ws_offline.txt" && \
   grep -q '"code":"model_not_streaming"' "$TMP/ws_offline.txt"; then
    echo "server-models ws-offline-group-refused OK (400 model_not_streaming)"
else
    echo "server-models ws-offline-group-refused FAIL:"; sed -n '1,6p' "$TMP/ws_offline.txt"; fail=1
fi

python3 "$TMP/wsprobe.py" localhost "$PORT" "/v1/audio/stream?model=does-not-exist" > "$TMP/ws_unknown.txt"
if grep -q "^HTTP/1.1 404 " "$TMP/ws_unknown.txt" && \
   grep -q '"code":"model_not_found"' "$TMP/ws_unknown.txt" && \
   grep -q 'streaming' "$TMP/ws_unknown.txt" && \
   grep -q 'offline' "$TMP/ws_unknown.txt"; then
    if grep -qi "^Retry-After:" "$TMP/ws_unknown.txt"; then
        echo "server-models ws-unknown-model FAIL: model_not_found carried a Retry-After"
        fail=1
    else
        echo "server-models ws-unknown-model OK (404 model_not_found, accepted set named, no Retry-After)"
    fi
else
    echo "server-models ws-unknown-model FAIL:"; sed -n '1,6p' "$TMP/ws_unknown.txt"; fail=1
fi

# The same refusal on the REST path, because the router classifies the query for
# every method and not only for the upgrade.
code=$(curl -s -o "$TMP/rest_unknown.json" -w '%{http_code}' -F file=@"$WAV" \
        "http://localhost:$PORT/v1/audio/transcriptions?model=does-not-exist")
if [ "$code" = "404" ] && grep -q '"code":"model_not_found"' "$TMP/rest_unknown.json"; then
    echo "server-models rest-unknown-model OK (404 model_not_found)"
else
    echo "server-models rest-unknown-model FAIL ($code): $(cat "$TMP/rest_unknown.json")"; fail=1
fi

# ---- 6. capacity is per group -----------------------------------------------
# The offline group has ONE slot and no admission queue, so of THREE requests
# fired at it at once exactly one is admitted and the rest are refused
# immediately -- deterministic, unlike holding a slot and racing a probe against
# it. A request to the OTHER group is fired in the same moment and must be
# served. That pair is the assertion: a 503 alone would not distinguish "this
# group is full" from "the fleet is full".
PIDS=""
i=1
while [ "$i" -le 3 ]; do
    curl -s -o "$TMP/busy$i.json" -w '%{http_code}' -m 120 -F model=offline \
        -F file=@"$WAV" -F language=auto \
        "http://localhost:$PORT/v1/audio/transcriptions" > "$TMP/busy$i.code" &
    PIDS="$PIDS $!"
    i=$((i + 1))
done
curl -s -o "$TMP/other.json" -w '%{http_code}' -m 120 \
    "http://localhost:$PORT/v1/audio/transcriptions?model=streaming" \
    -F file=@"$WAV" -F language=auto > "$TMP/other.code" &
OTHER_PID=$!
wait $PIDS "$OTHER_PID"          # ONLY the clients; the server never exits
PIDS=""

refused=0
served=0
i=1
while [ "$i" -le 3 ]; do
    c=$(cat "$TMP/busy$i.code" 2>/dev/null)
    if [ "$c" = "503" ] && grep -q '"code":"server_at_capacity"' "$TMP/busy$i.json"; then
        refused=$((refused + 1))
    elif [ "$c" = "200" ]; then
        served=$((served + 1))
    fi
    i=$((i + 1))
done
other=$(cat "$TMP/other.code" 2>/dev/null)

if [ "$refused" -ge 1 ] && [ "$served" -ge 1 ]; then
    if [ "$other" = "200" ] && [ "$(extract "$TMP/other.json")" = "$want_stream" ]; then
        echo "server-models per-group-capacity OK (offline $served served / $refused refused" \
             "with a readable 503, streaming group served in the same moment)"
    else
        echo "server-models per-group-capacity FAIL: the other group answered $other"
        echo "  $(cat "$TMP/other.json")"
        fail=1
    fi
else
    echo "server-models per-group-capacity FAIL: offline served=$served refused=$refused"
    echo "  (expected at least one of each: cap=1 with the queue disabled)"
    i=1; while [ "$i" -le 3 ]; do
        echo "  [$i] $(cat "$TMP/busy$i.code" 2>/dev/null): $(head -c 160 "$TMP/busy$i.json" 2>/dev/null)"
        i=$((i + 1))
    done
    fail=1
fi

# ---- 7. SIGTERM leaves no worker --------------------------------------------
kill -TERM "$SRV_PID" 2>/dev/null
wait "$SRV_PID" 2>/dev/null          # ONLY the server we started; never a bare wait
SRV_PID=""
sleep 0.5
left=$(pgrep -f "mynah-asr-server --model streaming=$STREAM_DIR" | wc -l | tr -d ' ')
if [ "$left" = "0" ]; then
    echo "server-models shutdown OK (survivors=0)"
else
    echo "server-models shutdown FAIL: $left survivors"
    pkill -9 -f "mynah-asr-server --model streaming=$STREAM_DIR"
    fail=1
fi

exit $fail
