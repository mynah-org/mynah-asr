#!/bin/sh
# End-to-end: mynah-asr transcribe on the fixture WAVs, compared to the expected text.
# Expected texts per engine (mynah.json): nemotron-streaming or parakeet-tdt
# (Parakeet: no --lang, ITN on -> numbers in digits).
# Exit: 0 ok, 1 mismatch, 77 skip (model missing).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
[ -f "$MODEL_DIR/mynah.json" ] || exit 77
ENGINE=$(sed -n 's/.*"engine": "\([^"]*\)".*/\1/p' "$MODEL_DIR/mynah.json")
NAME=$(sed -n 's/.*"name": "\([^"]*\)".*/\1/p' "$MODEL_DIR/mynah.json")

echo "--- e2e $NAME [$ENGINE]"
fail=0
# cache-aware streaming exists only for the streaming models
STREAM_OK=""
STREAM_LANG="auto"
if [ "$ENGINE" = "nemotron-streaming" ]; then STREAM_OK=1; STREAM_LANG="it-IT"; fi
check() { # wav, lang, expected substring
    out=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$1" --lang "$2" 2>/dev/null)
    case "$out" in
        *"$3"*) printf 'e2e %-16s OK: %s\n' "$(basename "$1")/$2" "$out" ;;
        *) printf 'e2e %-16s FAIL:\n  expected ~ %s\n  got:       %s\n' "$(basename "$1")/$2" "$3" "$out"; fail=1 ;;
    esac
}

# wav and substring for the shared checks (quant, timestamps, metal, segmentation)
Q_WAV=tests/audio/test_it.wav
Q_SUB="riconoscimento vocale in italiano"
SEG_TAIL="divano"

if [ "$ENGINE" = "parakeet-rnnt" ] || [ "$ENGINE" = "parakeet-ctc" ]; then
    # pure EN RNNT/CTC: lowercase, no punctuation
    Q_WAV=tests/audio/test_en.wav
    Q_SUB="speech recognition test"
    SEG_TAIL="today"
    check tests/audio/test_en.wav auto "hello this is a speech recognition test the weather is nice today"
elif [ "$NAME" = "parakeet-tdt_ctc-110m" ]; then
    # 110M: English only (the CI model); hybrid -> the CTC head too
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
    # Canary: ASR + src>tgt translation matrix. v2 (25 EU languages) applies
    # ITN by default (numbers in DIGITS) -> expectations differ from the flash models
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
        # input > expected output pairs (substring; collected from real outputs —
        # greedy is deterministic). fr>en: the 180m emits EOS immediately (a model
        # limitation, confirmed against the oracle too) -> larger models only
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
            *) printf 'e2e translate %s>%s FAIL:\n  expected ~ %s\n  got:       %s\n' \
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

# The timestamp aligner canary-1b-v2 bundles, when the converter extracted it.
# It is itself a CTC ASR model, which is the strongest available check on the
# conversion: if the weights or the vocabulary were wrong it would not transcribe.
if [ -f "$MODEL_DIR/aligner/mynah.json" ]; then
    out=$(./mynah-asr transcribe -m "$MODEL_DIR/aligner" -i tests/audio/test_en.wav 2>/dev/null)
    case "$out" in
        *"speech recognition test"*) echo "e2e aligner OK: $out" ;;
        *) echo "e2e aligner FAIL: $out"; fail=1 ;;
    esac
fi

# pre-quantized checkpoints (when generated with: mynah-asr quantize)
checkq() { # quant, expected substring
    out=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" --quant "$1" 2>/dev/null)
    case "$out" in
        *"$2"*) printf 'e2e quant-%-6s OK: %s\n' "$1" "$out" ;;
        *) printf 'e2e quant-%-6s FAIL: %s\n' "$1" "$out"; fail=1 ;;
    esac
}
[ -f "$MODEL_DIR/model.int8.safetensors" ] && checkq int8 "$Q_SUB"
[ -f "$MODEL_DIR/model.int4.safetensors" ] && checkq int4 "$Q_SUB"

# per-word timestamps: "t0 t1 word" lines, t0 monotonically non-decreasing,
# t1 within the audio duration (fixture <= 5.2s + one frame of margin).
# AED (Canary flash): <|timestamp|> prompt -> words bracketed with <|N|>.
# Skipped when the model has no support (v2: external aligner, not implemented)
if grep -q '"timestamp_tokens": false' "$MODEL_DIR/mynah.json"; then
    echo "e2e timestamps SKIP: this model has no generative <|timestamp|> tokens"
else
ts=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" --timestamps 2>/dev/null)
ts_ok=$(printf '%s\n' "$ts" | awk 'NF<3 {bad=1} $1+0>$2+0 {bad=1} $1+0<prev {bad=1}
    {prev=$1+0; n++} END {print (bad || n<5 || prev>5.3) ? "FAIL" : "OK"}')
if [ "$ts_ok" = "OK" ]; then
    echo "e2e timestamps OK: $(printf '%s\n' "$ts" | wc -l | tr -d ' ') words"
else
    echo "e2e timestamps FAIL:"; printf '%s\n' "$ts"; fail=1
fi
fi

# long-file segmentation: limit forced to 4 s on the fixture -> 2 segments
# split on silence; the concatenated text must stay complete
seg=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" --segment-sec 4 2>/dev/null)
case "$seg" in
    *"$Q_SUB"*"$SEG_TAIL"*) echo "e2e segment OK: $seg" ;;
    *) echo "e2e segment FAIL: $seg"; fail=1 ;;
esac

# VAD segmentation (only when the checkpoint is there: make fetch-vad). On a
# single short utterance the VAD must find one span covering it, so the text has
# to come out the same as without it — if enabling the VAD changes a 4-second
# fixture, it is cutting speech.
if [ -f models/silero-vad/mynah.json ]; then
    plain=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" 2>/dev/null)
    vad=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" --vad models/silero-vad 2>/dev/null)
    if [ "$plain" = "$vad" ]; then
        echo "e2e vad OK: unchanged on a single utterance"
    else
        printf 'e2e vad FAIL:\n  without: %s\n  with:    %s\n' "$plain" "$vad"; fail=1
    fi
fi

# Streaming endpointing (only for streaming models, and only with the VAD there).
# Two things, and the first is the important one: the VAD does not touch decoding
# in streaming, it only REPORTS where speech ended, so the text must come out
# byte-identical. If it does not, endpointing has side effects it should not have.
if [ -n "$STREAM_OK" ] && [ -f models/silero-vad/mynah.json ]; then
    plain=$(./mynah-asr stream -m "$MODEL_DIR" -i "$Q_WAV" --lang "$STREAM_LANG" 2>/dev/null)
    vout=$(./mynah-asr stream -m "$MODEL_DIR" -i "$Q_WAV" --lang "$STREAM_LANG" \
           --vad models/silero-vad 2>&1 >/dev/null)
    vtext=$(./mynah-asr stream -m "$MODEL_DIR" -i "$Q_WAV" --lang "$STREAM_LANG" \
            --vad models/silero-vad 2>/dev/null)
    if [ "$plain" != "$vtext" ]; then
        printf 'e2e eou FAIL: the VAD changed the streamed text\n  without: %s\n  with:    %s\n' \
            "$plain" "$vtext"; fail=1
    elif ! printf '%s' "$vout" | grep -q "eou at"; then
        echo "e2e eou FAIL: no endpoint reported on a fixture that ends in silence"; fail=1
    else
        echo "e2e eou OK: $(printf '%s' "$vout" | grep -c 'eou at') endpoint(s), text unchanged"
    fi
fi

# Metal backend (macOS only; for non-causal models the Metal kernel does not
# apply and the gate falls back to CPU: the check validates the text anyway)
out=$(./mynah-asr transcribe -m "$MODEL_DIR" -i "$Q_WAV" --backend metal 2>/dev/null)
case "$out" in
    *"$Q_SUB"*) echo "e2e backend-metal OK: $out" ;;
    *) echo "e2e backend-metal FAIL: $out"; fail=1 ;;
esac

exit $fail
