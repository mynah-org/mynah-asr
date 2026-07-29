#!/bin/sh
# Riproducibilità del server sotto concorrenza: N richieste IDENTICHE sparate
# in parallelo (stesso wav, stessa lingua) devono produrre risposte
# BYTE-IDENTICHE tra loro e uguali a una richiesta sequenziale di riferimento.
# Copre: worker pool con modello condiviso + batch scheduler (--batch 4:
# le richieste vengono micro-batchate insieme — la parity batch vs B=1 è
# già coperta da tests/test_batch, qui si verifica che il non-determinismo
# da concorrenza/batching non trapeli nelle risposte).
# Exit: 0 ok, 1 fail, 77 skip (modello assente).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
PORT="${2:-8207}"
N=8
[ -f "$MODEL_DIR/mynah.json" ] || exit 77

./mynah-asr-server -m "$MODEL_DIR" -p "$PORT" --threads 4 --batch 4 2>/dev/null &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null' EXIT

up=0
for i in $(seq 1 150); do    # cold load del modello grande: fino a 30 s
    curl -sf "http://localhost:$PORT/v1/health" >/dev/null 2>&1 && { up=1; break; }
    sleep 0.2
done
[ $up -eq 1 ] || { echo "serve-repro FAIL: server mai pronto"; exit 1; }

TMP=$(mktemp -d /tmp/mynah_asr_repro.XXXXXX) || exit 1
trap 'kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; rm -rf "$TMP"' EXIT

req() {
    curl -s -m 120 -F file=@tests/audio/test_it.wav -F language=it-IT \
         -F response_format=verbose_json \
         "http://localhost:$PORT/v1/audio/transcriptions"
}

req > "$TMP/ref.json"
grep -q '"text"' "$TMP/ref.json" || { echo "serve-repro FAIL: riferimento vuoto"; exit 1; }

# due ondate concorrenti: la prima parte a server scarico, la seconda a caldo.
# NOTA: wait sui PID delle SOLE curl — `wait` nudo aspetterebbe anche il
# server in background e non ritornerebbe mai (imparato sul campo)
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
        echo "serve-repro FAIL: $f diverso dal riferimento"
        diff "$TMP/ref.json" "$f" | head -4
        fail=1
    fi
done
[ $fail -eq 0 ] && echo "serve-repro OK: $((2 * N)) risposte concorrenti byte-identiche al riferimento"
exit $fail
