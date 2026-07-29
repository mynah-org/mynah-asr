#!/bin/sh
# Server reproducibility under concurrency: N IDENTICAL requests fired in
# parallel (same wav, same language) must produce responses that are
# BYTE-IDENTICAL to each other and to a sequential reference request.
# Covers: worker pool over a shared model + batch scheduler (--batch 4:
# requests get micro-batched together — batch vs B=1 parity is already
# covered by tests/test_batch; here we check that the non-determinism of
# concurrency/batching does not leak into the responses).
# Exit: 0 ok, 1 fail, 77 skip (model missing).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
PORT="${2:-8207}"
N=8
[ -f "$MODEL_DIR/mynah.json" ] || exit 77

./mynah-asr-server -m "$MODEL_DIR" -p "$PORT" --threads 4 --batch 4 2>/dev/null &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null' EXIT

up=0
for i in $(seq 1 150); do    # cold load of the large model: up to 30 s
    curl -sf "http://localhost:$PORT/v1/health" >/dev/null 2>&1 && { up=1; break; }
    sleep 0.2
done
[ $up -eq 1 ] || { echo "serve-repro FAIL: server never became ready"; exit 1; }

TMP=$(mktemp -d /tmp/mynah_asr_repro.XXXXXX) || exit 1
trap 'kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; rm -rf "$TMP"' EXIT

req() {
    curl -s -m 120 -F file=@tests/audio/test_it.wav -F language=it-IT \
         -F response_format=verbose_json \
         "http://localhost:$PORT/v1/audio/transcriptions"
}

req > "$TMP/ref.json"
grep -q '"text"' "$TMP/ref.json" || { echo "serve-repro FAIL: empty reference"; exit 1; }

# two concurrent waves: the first starts on an idle server, the second warm.
# NOTE: wait on the curl PIDs ONLY — a bare `wait` would also wait on the
# background server and never return (learned the hard way)
for wave in 1 2; do
    pids=""
    for i in $(seq 1 $N); do
        req > "$TMP/w${wave}_$i.json" &
        pids="$pids $!"
    done
    wait $pids
done

fail=0
for f in "$TMP"/w*_*.json; do
    if ! cmp -s "$TMP/ref.json" "$f"; then
        echo "serve-repro FAIL: $f differs from the reference"
        diff "$TMP/ref.json" "$f" | head -4
        fail=1
    fi
done
[ $fail -eq 0 ] && echo "serve-repro OK: $((2 * N)) concurrent responses byte-identical to the reference"
exit $fail
