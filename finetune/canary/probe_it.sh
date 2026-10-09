#!/bin/bash
# Step 4 (C0): zero-shot probe of the untouched Canary 180M on the frozen IT
# eval with <|it|> (and es/fr floors), plus EN sanity. -> $FT/probe/zeroshot.json
. "$(dirname "$0")/env.sh"
if step probe; then
    [ -e "$FT/done/data" ] || { echo "run prepare_it.sh first"; exit 2; }
    timeout "${T_PROBE:-5400}" "$PY" "$KIT/probe_it.py" "$@" || fail probe $?
    ok probe
fi
