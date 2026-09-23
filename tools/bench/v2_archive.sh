#!/bin/sh
# v2_archive.sh — bring a box's serving evidence home, so a campaign can be read
# after the box is gone.
#
# On 2026-09-22 the instance became unreachable minutes after a qualification
# passed, with 32 MB of per-utterance records, SIGUSR1 dumps and RSS samples
# still only on its disk. The run survived because the disk did; that was luck,
# and the profile could not be promoted from artefacts until the box came back.
#
#   tools/bench/v2_archive.sh user@host [remote_dir] [local_dir]
#
# Defaults: ~/asr-evidence/v2 -> .work/evidence/serving-v2 (untracked by design,
# .work/README.md rule 7). Incremental: a run already archived is not refetched.
set -u
[ $# -ge 1 ] || { echo "usage: v2_archive.sh user@host [remote_dir] [local_dir]" >&2; exit 2; }
HOST="$1"
REMOTE="${2:-asr-evidence/v2}"
LOCAL="${3:-.work/evidence/serving-v2}"
mkdir -p "$LOCAL" || exit 2

if command -v rsync >/dev/null 2>&1; then
    rsync -a --no-perms --no-owner --no-group "$HOST:$REMOTE/" "$LOCAL/" || exit 1
else
    # scp -r re-copies what is already here; rsync is preferred and usually present
    scp -q -r "$HOST:$REMOTE/*" "$LOCAL/" || exit 1
fi

printf '%s\n' "archived into $LOCAL:"
for d in "$LOCAL"/*/; do
    [ -d "$d" ] || continue
    n=$(ls "$d"/soak*.json "$d"/ladder-C*.json "$d"/genscale-*.json 2>/dev/null | wc -l | tr -d ' ')
    printf '  %-28s %6s  %s run(s)\n' "$(basename "$d")" "$(du -sh "$d" | cut -f1)" "$n"
done
