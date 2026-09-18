#!/bin/sh
# The streaming server under CONCURRENT WebSocket load (S2-2 / S2-7).
#
# What it proves, in order:
#   1. IDENTITY. Four streams paced at real time, two utterances each, produce
#      text byte-identical to `mynah-asr transcribe` of the same clip. A serving
#      change never changes a transcript: that is the gate, and the cadence
#      numbers printed beside it are a measurement of this machine, not a gate.
#   2. ISOLATION. A client that sends audio and never reads it back loses its
#      own slot and nobody else's: the two streams running next to it still
#      measure, and /v1/health shows the stalled slot cancelled.
#   3. The same identity through `--prefork 2 --cap 2`, so a transcript does not
#      depend on which worker served it.
#   4. BATCHED IDENTITY (S2-2b). Eight streams over five clips on ONE process
#      with `--cap 8`, so every step feeds a ready set of several streams through
#      one `mynah_asr_stream_step_batch` call: every text still byte-identical to
#      the CLI, and /v1/health proving the batched path is what ran
#      (`batched_steps_total` and `rows_stacked_total` both non-zero) rather than
#      a silent degradation to per-stream steps.
#
# Usage: test_server_stream.sh [model_dir] [port]
# Exit: 0 ok, 1 fail, 77 skip (model, binaries or python missing).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
PORT="${2:-8213}"
CLIPS="tests/audio/test_it.wav tests/audio/test_en.wav tests/audio/test_de.wav tests/audio/test_fr.wav"

[ -f "$MODEL_DIR/mynah.json" ] || exit 77
[ -x ./mynah-asr-server ] || exit 77
[ -x ./mynah-asr ] || exit 77
command -v python3 >/dev/null 2>&1 || exit 77
for c in $CLIPS; do [ -f "$c" ] || exit 77; done

TMP=$(mktemp -d /tmp/mynah_asr_stream.XXXXXX) || exit 1
SRV_PID=""
STALL_PID=""
cleanup() {
    [ -n "$STALL_PID" ] && kill "$STALL_PID" 2>/dev/null
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

# ---- the reference: what the offline CLI says about each clip ---------------
# Built here rather than committed, because it is the SAME binary's answer that
# the streams have to match; a stale committed file would test the file.
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
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    if not line:
        continue
    k, _, v = line.partition("\t")
    ref[k] = v
if not all(ref.values()):
    sys.exit("the CLI produced an empty reference for " +
             ", ".join(k for k, v in ref.items() if not v))
json.dump(ref, open(sys.argv[2], "w"), ensure_ascii=False)
PY
if [ $? -ne 0 ]; then echo "server-stream reference FAIL"; exit 1; fi

# Reads one stream_load JSON and says whether it may be believed.
cat > "$TMP/verdict.py" <<'PY'
# Reads one stream_load JSON and says whether it may be believed. The gate here is
# CORRECTNESS, not capacity: every utterance served, zero errors, every stream that
# played a clip produced the same bytes as every other and as the CLI. The envelope
# verdict (GOOD / NOT STREAMABLE) is printed for the record, because four streams on
# a laptop are over one scheduler's capacity and that is a serving question for the
# Linux box, not a defect of the server.
import json, sys
d = json.load(open(sys.argv[1]))
s = d["summary"]
c = s["counts"]
bad = []
if c["errors"]:
    bad.append("%d errors" % c["errors"])
if s["identity_fail"]:
    bad.append("streams disagreed: %s" % s["identity_fail"])
if s["reference_fail"]:
    bad.append("differs from the CLI: %s" % s["reference_fail"])
expected = int(sys.argv[2]) if len(sys.argv) > 2 else c["utterances"]
if c["ok"] != expected:
    bad.append("%d/%d utterances" % (c["ok"], expected))
if c["rejected"]:
    bad.append("%d rejected" % c["rejected"])
m = s["metrics"]
lag, ttfp = m.get("emission_lag_ms") or {}, m.get("ttfp_ms") or {}
print("    TTFP p50/p95 %s/%s ms   emission lag p50/p95 %s/%s ms   envelope %s   pacing %s" % (
    round(ttfp.get("p50") or 0), round(ttfp.get("p95") or 0),
    round(lag.get("p50") or 0), round(lag.get("p95") or 0),
    d.get("envelope", {}).get("verdict"), s["pacing"]["verdict"]))
if bad:
    print("    NOT BELIEVABLE: " + "; ".join(bad))
    sys.exit(1)
PY

# A client that speaks and never listens. Its receive buffer is deliberately
# tiny, so the server's writer meets a closed window early; whichever of the two
# per-stream caps fires first -- the send timeout or the idle cap -- the slot
# must go, and it must go alone.
cat > "$TMP/stalled.py" <<'PY'
import os, socket, struct, sys, time, wave, base64

host, port, wav, path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
with wave.open(wav) as w:
    pcm = w.readframes(w.getnframes())
s = socket.create_connection((host, port), timeout=10)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
key = base64.b64encode(os.urandom(16)).decode()
s.sendall(("GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\n"
           "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
           "Sec-WebSocket-Version: 13\r\n\r\n" % (path, host, port, key)).encode())
time.sleep(0.5)                       # the 101 is never read: that is the point
frame_bytes = 3200                    # 100 ms of s16le at 16 kHz
n = (len(pcm) + frame_bytes - 1) // frame_bytes
try:
    for i in range(n):
        payload = pcm[i * frame_bytes:(i + 1) * frame_bytes]
        mask = os.urandom(4)
        header = bytes([0x82, 0x80 | 126]) + struct.pack(">H", len(payload))
        s.sendall(header + mask +
                  bytes(b ^ mask[j % 4] for j, b in enumerate(payload)))
        time.sleep(0.1)
    time.sleep(30)                    # hold the socket open, still not reading
except OSError:
    pass
PY

# ---- 1. four streams, identity against the CLI -----------------------------
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT" --threads 4 --batch 4 \
    --quant int8 --cap 4 > "$TMP/srv.log" 2>&1 &
SRV_PID=$!
wait_ready "$PORT" || { echo "server-stream FAIL: server never became ready"; cat "$TMP/srv.log"; exit 1; }

python3 tools/bench/stream_load.py --host localhost --port "$PORT" \
    --streams 4 --repeat 2 --clips $CLIPS \
    --reference "$TMP/ref.json" --json "$TMP/load4.json" > "$TMP/load4.txt" 2>&1
if python3 "$TMP/verdict.py" "$TMP/load4.json" 8; then
    echo "server-stream 4-streams-identity OK"
else
    echo "server-stream 4-streams-identity FAIL"; sed -n '1,20p' "$TMP/load4.txt"; fail=1
fi
kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; SRV_PID=""

# ---- 2. a stalled reader takes only its own slot down ----------------------
PORT2=$((PORT + 1))
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT2" --threads 4 --batch 4 \
    --quant int8 --cap 4 --idle-ms 3000 > "$TMP/srv2.log" 2>&1 &
SRV_PID=$!
wait_ready "$PORT2" || { echo "server-stream stalled FAIL: server never became ready"; cat "$TMP/srv2.log"; exit 1; }

python3 "$TMP/stalled.py" localhost "$PORT2" tests/audio/test_de.wav \
    "/v1/audio/stream?lang=auto&lookahead=3" > "$TMP/stalled.log" 2>&1 &
STALL_PID=$!
sleep 1
python3 tools/bench/stream_load.py --host localhost --port "$PORT2" \
    --streams 2 --repeat 1 --clips tests/audio/test_it.wav tests/audio/test_en.wav \
    --reference "$TMP/ref.json" --json "$TMP/load2.json" > "$TMP/load2.txt" 2>&1 &
LOAD_PID=$!
wait $LOAD_PID                      # ONLY the client: the server never exits
if python3 "$TMP/verdict.py" "$TMP/load2.json" 2; then
    echo "server-stream stalled-reader-isolation OK (2 streams unaffected)"
else
    echo "server-stream stalled-reader-isolation FAIL"; sed -n '1,20p' "$TMP/load2.txt"; fail=1
fi

cancelled=0
i=0
while [ "$i" -lt 20 ]; do           # ~10 s, bounded
    h=$(curl -s "http://localhost:$PORT2/v1/health")
    cancelled=$(printf '%s' "$h" | sed -n 's/.*"cancelled"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
    [ -n "$cancelled" ] && [ "$cancelled" -ge 1 ] && break
    sleep 0.5
    i=$((i + 1))
done
if [ -n "$cancelled" ] && [ "$cancelled" -ge 1 ]; then
    echo "server-stream stalled-reader-cancelled OK ($cancelled slot(s) reclaimed)"
else
    echo "server-stream stalled-reader-cancelled FAIL: $(curl -s http://localhost:$PORT2/v1/health)"
    fail=1
fi
kill $STALL_PID 2>/dev/null; STALL_PID=""
kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; SRV_PID=""

# ---- 3. the same identity through the prefork router -----------------------
PORT3=$((PORT + 2))
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT3" --threads 2 --batch 1 \
    --quant int8 --prefork 2 --cap 2 > "$TMP/srv3.log" 2>&1 &
SRV_PID=$!
wait_ready "$PORT3" || { echo "server-stream prefork FAIL: fleet never became ready"; cat "$TMP/srv3.log"; exit 1; }
python3 tools/bench/stream_load.py --host localhost --port "$PORT3" \
    --streams 2 --repeat 2 --clips tests/audio/test_it.wav tests/audio/test_en.wav \
    --reference "$TMP/ref.json" --json "$TMP/loadpf.json" > "$TMP/loadpf.txt" 2>&1
if python3 "$TMP/verdict.py" "$TMP/loadpf.json" 4; then
    echo "server-stream prefork-identity OK"
else
    echo "server-stream prefork-identity FAIL"; sed -n '1,20p' "$TMP/loadpf.txt"; fail=1
fi
kill -TERM $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; SRV_PID=""
sleep 0.5
left=$(pgrep -f "mynah-asr-server -m $MODEL_DIR -p $PORT3" | wc -l | tr -d ' ')
if [ "$left" = "0" ]; then
    echo "server-stream prefork shutdown leaves no worker OK"
else
    echo "server-stream prefork shutdown FAIL: $left survivors"
    pkill -9 -f "mynah-asr-server -m $MODEL_DIR -p $PORT3"
    fail=1
fi

# ---- 4. eight streams through the batched step, on one process -------------
# The reference here is the CLI's own STREAMING answer, not `transcribe`: what
# this phase gates is that a stream batched with seven others says exactly what
# the same clip says alone, and the single-stream streaming path is that answer.
# It matters for test_es.wav, whose int8 streaming transcript already differs
# from its offline one ("a las muertes" vs "a las vuelve") -- a pre-existing
# streaming-vs-offline numerical difference recorded in .work/stream-api-v2.md,
# and nothing to do with batching. For the other four clips the two references
# are the same bytes.
CLIPS8="$CLIPS tests/audio/test_es.wav"
[ -f tests/audio/test_es.wav ] || CLIPS8="$CLIPS"

: > "$TMP/ref8.tsv"
for c in $CLIPS8; do
    printf '%s\t' "$c" >> "$TMP/ref8.tsv"
    ./mynah-asr stream -m "$MODEL_DIR" -i "$c" --quant int8 --lang auto 2>/dev/null \
        | tr -d '\n' >> "$TMP/ref8.tsv"
    printf '\n' >> "$TMP/ref8.tsv"
done
python3 - "$TMP/ref8.tsv" "$TMP/ref8.json" <<'PY'
import json, sys
ref = {}
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    if not line:
        continue
    k, _, v = line.partition("\t")
    ref[k] = v
if not all(ref.values()):
    sys.exit("the CLI produced an empty streaming reference for " +
             ", ".join(k for k, v in ref.items() if not v))
json.dump(ref, open(sys.argv[2], "w"), ensure_ascii=False)
PY
if [ $? -ne 0 ]; then echo "server-stream batched reference FAIL"; exit 1; fi

PORT4=$((PORT + 3))
./mynah-asr-server -m "$MODEL_DIR" -p "$PORT4" --threads 4 --batch 4 \
    --quant int8 --cap 8 > "$TMP/srv4.log" 2>&1 &
SRV_PID=$!
wait_ready "$PORT4" || { echo "server-stream batched FAIL: server never became ready"; cat "$TMP/srv4.log"; exit 1; }

n8=0
for c in $CLIPS8; do n8=$((n8 + 1)); done
python3 tools/bench/stream_load.py --host localhost --port "$PORT4" \
    --streams 8 --repeat 1 --clips $CLIPS8 \
    --reference "$TMP/ref8.json" --json "$TMP/load8.json" > "$TMP/load8.txt" 2>&1
if python3 "$TMP/verdict.py" "$TMP/load8.json" 8; then
    echo "server-stream batched-identity OK (8 streams, $n8 clips, --cap 8)"
else
    echo "server-stream batched-identity FAIL"; sed -n '1,20p' "$TMP/load8.txt"; fail=1
fi

# The counters are the point: a run that fell back to per-stream steps would
# still be correct and would still pass the identity check above, so the gate
# asks the server what path it ran (ENGINEERING.md §5, §6).
curl -s "http://localhost:$PORT4/v1/health" > "$TMP/health8.json"
if python3 - "$TMP/health8.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
b = h.get("batch") or {}
steps = b.get("batched_steps_total", 0)
rows = b.get("rows_stacked_total", 0)
print("    batched_steps_total %d   rows_stacked_total %d   ready_mean %.2f   "
      "step_wall_ms mean %.1f" % (steps, rows, b.get("ready_mean", 0.0),
                                  (b.get("step_wall_ms") or {}).get("mean", 0.0)))
by_b = b.get("by_b") or {}
for k in sorted(by_b, key=int):
    e = by_b[k]
    print("      B=%s  steps %d  wall mean %.1f ms" % (k, e["steps"], e["wall_ms_mean"]))
bad = []
if not steps:
    bad.append("batched_steps_total is 0: no step went through the batched call")
if not rows:
    bad.append("rows_stacked_total is 0: every step degraded to the single path")
if bad:
    print("    " + "; ".join(bad))
    sys.exit(1)
PY
then
    echo "server-stream batched-path-proven OK (/v1/health counted stacked rows)"
else
    echo "server-stream batched-path-proven FAIL"; fail=1
fi
kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; SRV_PID=""

exit $fail
