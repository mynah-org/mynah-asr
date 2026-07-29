#!/bin/sh
# End-to-end: mynah-asr transcribe sui WAV fixture, confronto col testo atteso.
# Testi attesi per engine (mynah.json): nemotron-streaming o parakeet-tdt
# (Parakeet: niente --lang, ITN attiva -> numeri in cifre).
# Exit: 0 ok, 1 mismatch, 77 skip (modello assente).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
[ -f "$MODEL_DIR/mynah.json" ] || exit 77
ENGINE=$(sed -n 's/.*"engine": "\([^"]*\)".*/\1/p' "$MODEL_DIR/mynah.json")
NAME=$(sed -n 's/.*"name": "\([^"]*\)".*/\1/p' "$MODEL_DIR/mynah.json")

echo "--- e2e $NAME [$ENGINE]"
fail=0
check() { # wav, lang, expected substring
    out=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$1" --lang "$2" 2>/dev/null)
    case "$out" in
        *"$3"*) printf 'e2e %-16s OK: %s\n' "$(basename "$1")/$2" "$out" ;;
        *) printf 'e2e %-16s FAIL:\n  atteso  ~ %s\n  ottenuto: %s\n' "$(basename "$1")/$2" "$3" "$out"; fail=1 ;;
    esac
}

# wav e substring per i check comuni (quant, timestamps, metal, segmentazione)
Q_WAV=tests/audio/test_it.wav
Q_SUB="riconoscimento vocale in italiano"
SEG_TAIL="divano"

if [ "$ENGINE" = "parakeet-rnnt" ] || [ "$ENGINE" = "parakeet-ctc" ]; then
    # RNNT/CTC puri EN: lowercase, senza punteggiatura
    Q_WAV=tests/audio/test_en.wav
    Q_SUB="speech recognition test"
    SEG_TAIL="today"
    check tests/audio/test_en.wav auto "hello this is a speech recognition test the weather is nice today"
elif [ "$NAME" = "parakeet-tdt_ctc-110m" ]; then
    # 110M: solo inglese (candidato CI); hybrid -> anche la head CTC
    Q_WAV=tests/audio/test_en.wav
    Q_SUB="speech recognition test"
    SEG_TAIL="today"
    check tests/audio/test_en.wav auto "Hello, this is a speech recognition test. The weather is nice today."
    out=$(./mynah-asr transcribe -m "$MODEL_DIR" -i tests/audio/test_en.wav --decoder ctc 2>/dev/null)
    case "$out" in
        *"This is a speech recognition test. The weather is nice today."*)
            echo "e2e decoder-ctc OK: $out" ;;
        *) echo "e2e decoder-ctc FAIL: $out"; fail=1 ;;
    esac
elif [ "$ENGINE" = "parakeet-tdt" ]; then
    check tests/audio/test_it.wav auto "Ciao, questo è un test di riconoscimento vocale in italiano: il gatto dorme sul divano."
    check tests/audio/test_en.wav auto "Hello, this is a speech recognition test, the weather is nice today."
    check tests/audio/test_de.wav auto "die Besprechung beginnt um 9 Uhr im großen Saal"
    check tests/audio/test_fr.wav auto "la réunion commence à 9h dans la grande salle"
    check tests/audio/test_es.wav auto "la reunión empieza a las 9 en la sala grande"
elif [ "$ENGINE" = "canary-aed" ]; then
    # Canary: ASR + matrice di traduzione src>tgt. v2 (25 lingue EU) applica
    # l'ITN di default (numeri in CIFRE) -> attesi diversi dai flash
    Q_WAV=tests/audio/test_en.wav
    Q_SUB="speech recognition test"
    SEG_TAIL="today"
    if [ "$NAME" = "canary-1b-v2" ]; then
        check tests/audio/test_en.wav en "Hello, this is a speech recognition test, the weather is nice today."
        check tests/audio/test_de.wav de "die Besprechung beginnt um 9 Uhr"
        check tests/audio/test_fr.wav fr "la réunion commence à 9h"
        check tests/audio/test_es.wav es "la reunión empieza a las 9"
        check tests/audio/test_it.wav it "Il gatto dorme sul divano"
        TRX="en:de:Spracherkennungstest
en:fr:reconnaissance
en:es:reconocimiento
de:en:begins at 9
es:en:starts at 9
fr:en:at 9
it:en:cat"
    else
        check tests/audio/test_en.wav en "Hello, this is a speech recognition test. The weather is nice today."
        check tests/audio/test_de.wav de "Die Besprechung beginnt um neun Uhr"
        check tests/audio/test_fr.wav fr "la réunion commence à neuf heures"
        check tests/audio/test_es.wav es "la reunión empieza a las nueve"
        # coppie input > output atteso (substring; raccolte dagli output reali —
        # greedy deterministico). fr>en: il 180m emette EOS subito (limite del
        # modello, verificato anche con l'oracolo) -> solo sui modelli più grandi
        TRX="en:de:Spracherkennungstest
en:fr:reconnaissance
en:es:reconocimiento de voz
de:en:at nine o'clock
es:en:nine o'clock"
        [ "$NAME" = "canary-180m-flash" ] || TRX="$TRX
fr:en:nine o'clock"
    fi
    rm -f /tmp/mynah_asr_trx_fail
    printf '%s\n' "$TRX" | while IFS=: read -r src tgt want; do
        [ -n "$src" ] || continue
        out=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "tests/audio/test_$src.wav" \
              --lang "$src" --target-lang "$tgt" 2>/dev/null)
        case "$out" in
            *"$want"*) printf 'e2e translate %s>%s OK: %s\n' "$src" "$tgt" "$out" ;;
            *) printf 'e2e translate %s>%s FAIL:\n  atteso ~ %s\n  ottenuto: %s\n' \
                   "$src" "$tgt" "$want" "$out"; echo FAIL > /tmp/mynah_asr_trx_fail ;;
        esac
    done
    [ -f /tmp/mynah_asr_trx_fail ] && { rm -f /tmp/mynah_asr_trx_fail; fail=1; }
else
    check tests/audio/test_it.wav auto  "Ciao, questo è un test di riconoscimento vocale in italiano."
    check tests/audio/test_it.wav it-IT "Il gatto dorme sul divano."
    check tests/audio/test_en.wav auto  "Hello, this is a speech recognition test."
    check tests/audio/test_de.wav auto  "die Besprechung beginnt um neun Uhr"
    check tests/audio/test_fr.wav auto  "la réunion commence à neuf heures"
    check tests/audio/test_es.wav auto  "la reunión empieza"
fi

# checkpoint pre-quantizzati (se generati con: mynah-asr quantize)
checkq() { # quant, expected substring
    out=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" --quant "$1" 2>/dev/null)
    case "$out" in
        *"$2"*) printf 'e2e quant-%-6s OK: %s\n' "$1" "$out" ;;
        *) printf 'e2e quant-%-6s FAIL: %s\n' "$1" "$out"; fail=1 ;;
    esac
}
[ -f "$MODEL_DIR/model.int8.safetensors" ] && checkq int8 "$Q_SUB"
[ -f "$MODEL_DIR/model.int4.safetensors" ] && checkq int4 "$Q_SUB"

# timestamp per parola: righe "t0 t1 parola", t0 monotono non-decrescente,
# t1 entro la durata dell'audio (fixture <= 5.2s + margine di un frame).
# AED (Canary flash): prompt <|timestamp|> -> parole bracketate con <|N|>.
# Skip se il modello non li supporta (v2: allineatore esterno, non implementato)
if grep -q '"timestamp_tokens": false' "$MODEL_DIR/mynah.json"; then
    echo "e2e timestamps SKIP: il modello non ha i token <|timestamp|> generativi"
else
ts=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" --timestamps 2>/dev/null)
ts_ok=$(printf '%s\n' "$ts" | awk 'NF<3 {bad=1} $1+0>$2+0 {bad=1} $1+0<prev {bad=1}
    {prev=$1+0; n++} END {print (bad || n<5 || prev>5.3) ? "FAIL" : "OK"}')
if [ "$ts_ok" = "OK" ]; then
    echo "e2e timestamps OK: $(printf '%s\n' "$ts" | wc -l | tr -d ' ') parole"
else
    echo "e2e timestamps FAIL:"; printf '%s\n' "$ts"; fail=1
fi
fi

# segmentazione file lunghi: limite forzato a 4 s sul fixture -> 2 segmenti
# divisi sul silenzio, il testo concatenato deve restare completo
seg=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" --segment-sec 4 2>/dev/null)
case "$seg" in
    *"$Q_SUB"*"$SEG_TAIL"*) echo "e2e segment OK: $seg" ;;
    *) echo "e2e segment FAIL: $seg"; fail=1 ;;
esac

# backend Metal (solo macOS; per i modelli non-causali il kernel Metal non si
# applica e il gate ricade su CPU: il check verifica comunque il testo)
out=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" --backend metal 2>/dev/null)
case "$out" in
    *"$Q_SUB"*) echo "e2e backend-metal OK: $out" ;;
    *) echo "e2e backend-metal FAIL: $out"; fail=1 ;;
esac

exit $fail
