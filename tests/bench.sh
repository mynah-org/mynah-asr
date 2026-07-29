#!/bin/sh
# Bench riproducibile: RTF warm (2a run) e picco RAM per ogni modello presente.
# Uso: sh tests/bench.sh [wav]   (default: fixture per-modello, come l'e2e)
# I numeri del README vengono da qui.
WAV_IT="${1:-tests/audio/test_it.wav}"
WAV_EN="${1:-tests/audio/test_en.wav}"

case "$(uname -s)" in
    Darwin) TIME="/usr/bin/time -l"; RSS_KEY="maximum resident set size"; RSS_DIV=1073741824 ;;  # byte
    *)      TIME="/usr/bin/time -v"; RSS_KEY="Maximum resident set size"; RSS_DIV=1048576 ;;     # KB
esac

printf '%-34s %8s %9s %10s\n' modello RTF warm-s "RAM-GB"
for m in models/*/; do
    [ -f "$m/mynah.json" ] || continue
    name=$(basename "$m")
    wav="$WAV_IT"
    case "$name" in *rnnt-0.6b|*ctc-0.6b|*110m) wav="$WAV_EN" ;; esac
    # 2 warm-up: con molti modelli grandi la page cache viene riciclata tra un
    # modello e l'altro e una sola run non basta a ripaginare tutto l'mmap
    ./mynah-asr transcribe -m "$m" -i "$wav" >/dev/null 2>&1
    ./mynah-asr transcribe -m "$m" -i "$wav" >/dev/null 2>&1
    out=$($TIME ./mynah-asr transcribe -m "$m" -i "$wav" 2>&1 >/dev/null)
    rtf=$(printf '%s' "$out" | sed -n 's/.*RTF \([0-9.]*\).*/\1/p')
    secs=$(printf '%s' "$out" | sed -n 's/.*inferenza \([0-9.]*\)s.*/\1/p')
    rss=$(printf '%s' "$out" | grep "$RSS_KEY" | sed 's/[^0-9]//g')
    ram=$(awk "BEGIN {printf \"%.2f\", $rss / $RSS_DIV}")
    printf '%-34s %8s %9s %10s\n' "$name" "$rtf" "$secs" "$ram"
done
