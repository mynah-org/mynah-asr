#!/bin/bash
# Step 2: Italian data (MLS it + FLEURS it/en, all CC-BY-4.0; Common Voice
# optional). Restartable at stage level (markers data-meta/-plan/-audio/
# -manifests written by data_it.py).
#   bash data_it.sh --dry-run          # metadata + plan only, prints hours
#   tmux new -d -s ft 'bash data_it.sh 2>&1 | tee -a /root/ft/logs/data.log'
# Options pass through to data_it.py (e.g. --with-fleurs-train, --hours 5,20,40).
# Optional Common Voice (CC0, Mozilla Data Collective, API key needed):
#   CV_TARBALL=/path/to/it.tar.gz bash data_it.sh
#   or MDC_API_KEY=... CV_DATASET=<MDC dataset id> bash data_it.sh  (UNVERIFIED SDK call)
. "$(dirname "$0")/common.sh"

dry=0; for x in "$@"; do [ "$x" = "--dry-run" ] && dry=1; done

if [ -z "${CV_TARBALL:-}" ] && [ -n "${MDC_API_KEY:-}" ] && [ -n "${CV_DATASET:-}" ] && [ $dry = 0 ]; then
    need_gb 25 "the Common Voice tarball (~10.5 GB) + audio"
    mkdir -p "$FT/cv"
    timeout 600 uv pip install --python "$PY" datacollective || fail cv-sdk $?
    (cd "$FT/cv" && timeout 7200 "$PY" -c "from datacollective import download_dataset; print(download_dataset('$CV_DATASET'))") \
        || fail cv-download $?
    CV_TARBALL=$(find "$FT/cv" -name '*.tar*' -size +100M -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$CV_TARBALL" ] || { echo "no tarball found under $FT/cv"; exit 2; }
    export CV_TARBALL
fi

if [ $dry = 1 ]; then
    timeout 1800 "$PY" "$KIT/data_it.py" "$@"; exit $?
fi
if step data; then
    need_gb 12 "40 h + eval audio (~6-7 GB) + transient MLS shards"
    timeout "${T_DATA:-10800}" "$PY" "$KIT/data_it.py" "$@" || fail data $?
    [ -n "${CV_TARBALL:-}" ] && [ "${CV_DELETE_TARBALL:-0}" = 1 ] && rm -f "$CV_TARBALL"
    du -sh "$FT/audio" "$FT/manifests"
    ok data
fi
