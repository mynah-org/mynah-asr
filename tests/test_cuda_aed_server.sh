#!/bin/sh
# mynah-asr-server-cuda on an AED pack (Canary): the OFFLINE mode, end to end.
# .work/canary-180m-l4.md section 5.
#
# What it proves, on one server started on the pack:
#   1. the mode comes from the pack: /v1/models says engine "aed", streaming false
#   2. the WebSocket is refused BEFORE the upgrade with 400 model_not_streaming
#      (the CPU server's answer for the same pack), with no Retry-After
#   3. a REST transcription of every committed FLEURS clip (samples/en, samples/fr,
#      each with its language) is byte-identical to `mynah-asr transcribe` (the
#      library's default decoding, f32) -- for the f32 arm; with --precision bf16
#      the texts are reported, not gated
#   4. the same clips fired CONCURRENTLY (so the batcher puts them in one GPU
#      batch) come back with the same bytes as one by one (contract 4)
#   5. a translation (en>de) is byte-identical to the CLI's --target-lang
#   6. a language the pack does not serve is a 400 language_not_supported
#   7. /v1/health books balance; SIGTERM ends the process
#
# Usage: test_cuda_aed_server.sh [model_dir] [port] [engine cuda|cpu] [extra server args...]
# Exit: 0 ok, 1 fail, 77 skip (no pack, no binaries, no curl/python3, no device).
DIR="${1:-models/canary-180m-flash}"
PORT="${2:-8297}"
ENGINE="${3:-cuda}"
if [ $# -ge 3 ]; then shift 3; else set --; fi
EXTRA="$*"
SRV=./mynah-asr-server-cuda
[ "$ENGINE" = cpu ] && [ ! -x "$SRV" ] && SRV=./mynah-asr-server-cuda-cpuonly

[ -f "$DIR/mynah.json" ] || { echo "SKIP: no pack at $DIR"; exit 77; }
[ -x "$SRV" ] || { echo "SKIP: $SRV not built"; exit 77; }
[ -x ./mynah-asr ] || { echo "SKIP: ./mynah-asr not built (make)"; exit 77; }
command -v curl >/dev/null 2>&1 || exit 77
command -v python3 >/dev/null 2>&1 || exit 77

TMP=$(mktemp -d /tmp/mynah_asr_aed_srv.XXXXXX) || exit 1
SRV_PID=""
cleanup() {
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT
fail=0
F32=1
case "$EXTRA" in *bf16*|*own-tc*) F32=0 ;; esac

CLIPS="samples/en/fleurs_1521.wav:en samples/en/fleurs_1534.wav:en samples/en/fleurs_long.wav:en samples/fr/fleurs_1521.wav:fr samples/fr/fleurs_1534.wav:fr"

# ---- the references, from the CLI (f32, the library's default decoding) -----
i=0
for c in $CLIPS; do
    wav=${c%%:*}; lang=${c##*:}
    ./mynah-asr transcribe -m "$DIR" -i "$wav" --lang "$lang" 2>/dev/null | tr -d '\n' > "$TMP/ref_$i.txt"
    i=$((i + 1))
done
./mynah-asr transcribe -m "$DIR" -i samples/en/fleurs_1534.wav --lang en --target-lang de 2>/dev/null \
    | tr -d '\n' > "$TMP/ref_tr.txt"

# ---- the server -------------------------------------------------------------
# shellcheck disable=SC2086
"$SRV" -m "$DIR" --engine "$ENGINE" -p "$PORT" --cap 16 --batch 8 --cohort-ms 100 $EXTRA \
    > "$TMP/srv.log" 2>&1 &
SRV_PID=$!
up=0
for _ in $(seq 1 600); do
    if curl -s "http://127.0.0.1:$PORT/v1/health" > /dev/null 2>&1; then up=1; break; fi
    kill -0 "$SRV_PID" 2>/dev/null || break
    sleep 0.2
done
if [ "$up" -ne 1 ]; then
    if grep -q -e "no CUDA device" -e "not compiled" "$TMP/srv.log"; then
        echo "SKIP: $(grep -e 'no CUDA device' -e 'not compiled' "$TMP/srv.log" | head -1)"
        exit 77
    fi
    echo "FAIL: the server did not come up"; cat "$TMP/srv.log"; exit 1
fi
grep "SERVER-CONFIG" "$TMP/srv.log"

# 1. the mode
curl -s "http://127.0.0.1:$PORT/v1/models" > "$TMP/models.json"
if python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["data"][0]; sys.exit(0 if m["engine"]=="aed" and m["streaming"] is False else 1)' "$TMP/models.json"; then
    echo "ok   1 /v1/models: engine aed, streaming false"
else
    echo "FAIL 1 /v1/models: $(cat "$TMP/models.json")"; fail=1
fi

# 2. the WebSocket
curl -s -i --max-time 5 "http://127.0.0.1:$PORT/v1/audio/stream?lang=en" \
    -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" \
    -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" > "$TMP/ws.txt"
if head -1 "$TMP/ws.txt" | grep -q " 400 " && grep -q model_not_streaming "$TMP/ws.txt" && ! grep -qi "Retry-After" "$TMP/ws.txt"; then
    echo "ok   2 WebSocket: 400 model_not_streaming before the upgrade"
else
    echo "FAIL 2 WebSocket: $(head -1 "$TMP/ws.txt") $(tail -1 "$TMP/ws.txt")"; fail=1
fi

post() {  # wav lang out [extra -F args]
    curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
        -F "file=@$1" -F "language=$2" -F "response_format=text" > "$3"
}

# 3. one by one
i=0
for c in $CLIPS; do
    wav=${c%%:*}; lang=${c##*:}
    post "$wav" "$lang" "$TMP/seq_$i.txt"
    if cmp -s "$TMP/ref_$i.txt" "$TMP/seq_$i.txt"; then
        echo "ok   3 $wav [$lang] identical to the CLI"
    else
        echo "DIFF 3 $wav [$lang]"; echo "     cli   : $(cat "$TMP/ref_$i.txt")"; echo "     server: $(cat "$TMP/seq_$i.txt")"
        [ "$F32" -eq 1 ] && fail=1
    fi
    i=$((i + 1))
done

# 4. concurrently (one batch), twice
for round in 1 2; do
    i=0; PIDS=""
    for c in $CLIPS; do
        wav=${c%%:*}; lang=${c##*:}
        post "$wav" "$lang" "$TMP/con_$i.txt" &
        PIDS="$PIDS $!"
        i=$((i + 1))
    done
    # shellcheck disable=SC2086
    wait $PIDS
    i=0; same=0; n=0
    for c in $CLIPS; do
        n=$((n + 1))
        if cmp -s "$TMP/seq_$i.txt" "$TMP/con_$i.txt"; then same=$((same + 1)); else echo "     DIFF ${c%%:*}: $(cat "$TMP/con_$i.txt")"; fi
        i=$((i + 1))
    done
    if [ "$same" -eq "$n" ]; then echo "ok   4 concurrent round $round: $same/$n identical to one by one"
    else echo "FAIL 4 concurrent round $round: $same/$n identical"; fail=1; fi
done

# 5. translation
curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/audio/translations" -F "file=@samples/en/fleurs_1534.wav" \
    -F "language=en" -F "target_language=de" -F "response_format=text" > "$TMP/tr.txt"
if cmp -s "$TMP/ref_tr.txt" "$TMP/tr.txt"; then echo "ok   5 translation en>de identical to the CLI"
else
    echo "DIFF 5 translation: cli '$(cat "$TMP/ref_tr.txt")' server '$(cat "$TMP/tr.txt")'"
    [ "$F32" -eq 1 ] && fail=1
fi

# 6. a language the pack does not serve
code=$(curl -s -o "$TMP/bad.txt" -w '%{http_code}' "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
    -F "file=@samples/en/fleurs_1521.wav" -F "language=xx")
if [ "$code" = 400 ] && grep -q language_not_supported "$TMP/bad.txt"; then echo "ok   6 language xx: 400 language_not_supported"
else echo "FAIL 6 language xx: $code $(cat "$TMP/bad.txt")"; fail=1; fi

# 7. books, then SIGTERM
curl -s "http://127.0.0.1:$PORT/v1/health" > "$TMP/health.json"
if python3 -c 'import json,sys; h=json.load(open(sys.argv[1])); sys.exit(0 if h["requests"]["balanced"] and h["mode"]=="offline" else 1)' "$TMP/health.json"; then
    echo "ok   7 health: books balanced ($(python3 -c 'import json,sys; h=json.load(open(sys.argv[1])); print("batches", h["batch"]["batches_total"], "items_mean", round(h["batch"]["items_mean"],2))' "$TMP/health.json"))"
else
    echo "FAIL 7 health: $(cat "$TMP/health.json")"; fail=1
fi
kill "$SRV_PID"
for _ in $(seq 1 50); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 0.2; done
if kill -0 "$SRV_PID" 2>/dev/null; then echo "FAIL 7 the server ignored SIGTERM"; fail=1; else SRV_PID=""; echo "ok   7 SIGTERM: exited"; fi

[ "$fail" -eq 0 ] && echo PASS || echo FAIL
exit "$fail"
