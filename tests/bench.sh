#!/bin/sh
# Reproducible bench: warm RTF (2nd run) and peak RAM for every model present.
# Usage: sh tests/bench.sh [wav]   (default: per-model fixture, like the e2e)
# The README numbers come from here.
WAV_IT="${1:-tests/audio/test_it.wav}"
WAV_EN="${1:-tests/audio/test_en.wav}"

case "$(uname -s)" in
    Darwin) TIME="/usr/bin/time -l"; RSS_KEY="maximum resident set size"; RSS_DIV=1073741824 ;;  # byte
    *)      TIME="/usr/bin/time -v"; RSS_KEY="Maximum resident set size"; RSS_DIV=1048576 ;;     # KB
esac

printf '%-34s %8s %9s %10s\n' model RTF warm-s "RAM-GB"
for m in models/*/; do
    [ -f "$m/mynah.json" ] || continue
    name=$(basename "$m")
    wav="$WAV_IT"
    case "$name" in *rnnt-0.6b|*ctc-0.6b|*110m) wav="$WAV_EN" ;; esac
    # 2 warm-ups: with many large models the page cache is recycled between one
    # model and the next, and a single run is not enough to page the mmap back in
    ./mynah-asr transcribe -m "$m" -i "$wav" >/dev/null 2>&1
    ./mynah-asr transcribe -m "$m" -i "$wav" >/dev/null 2>&1
    out=$($TIME ./mynah-asr transcribe -m "$m" -i "$wav" 2>&1 >/dev/null)
    rtf=$(printf '%s' "$out" | sed -n 's/.*RTF \([0-9.]*\).*/\1/p')
    secs=$(printf '%s' "$out" | sed -n 's/.*inference \([0-9.]*\)s.*/\1/p')
    rss=$(printf '%s' "$out" | grep "$RSS_KEY" | sed 's/[^0-9]//g')
    ram=$(awk "BEGIN {printf \"%.2f\", $rss / $RSS_DIV}")
    printf '%-34s %8s %9s %10s\n' "$name" "$rtf" "$secs" "$ram"
done
