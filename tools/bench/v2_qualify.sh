#!/bin/sh
# v2_qualify.sh — qualify a server build as a PRODUCT, not screen it.
#
# box_qualify.sh screens a box: it starts one fleet, walks a wave ladder and
# optionally soaks once.  This does the thing `.work/server-v2-qualification.md`
# registered, and the differences are the whole point:
#
#   * the server is PINNED to a cpu slice and the load generator to a disjoint
#     one, because a generator that shares cores with the fleet is measuring
#     itself.  box_qualify leaves the parent on every cpu, so its workers carve
#     slices out of the whole machine and nothing is reserved.
#   * every rung and every soak gets a FRESH server.  A rung that inherits the
#     previous rung's heap is not an independent measurement.
#   * the unloaded C=1 pass over the corpus is captured as the reference, from
#     the SERVER, and replayed under load.  Rule 4 says a transcript never
#     depends on batching or threads; this is where that is checked rather than
#     assumed, and it is the one thing the 2026-09-20 soak could not do.
#   * SIGUSR1 dumps run throughout every soak, so "no stall" has evidence
#     instead of an absence of complaints.
#
# Usage:
#   tools/bench/v2_qualify.sh -m models/nemotron-3.5-asr-streaming-0.6b \
#       [--phase freeze|reference|ladder|soak|all] [-o ~/asr-evidence/v2]
#       [--ladder "8 16 24 32"] [--soak-c 16] [--soak-seconds 1800] [--soaks 2]
#       [--reference-file <reference.json from an earlier run of this tool>]
#       [--server-cpus 0-23] [--gen-cpus 24-31] [-W 3] [-T 8] [-C 96]
#
# It refuses rather than produce a number it cannot support: a dirty tree, a
# dispatch row the binary cannot name, a worker whose mask escapes the server
# slice, a fleet that never became ready, or a corpus class with no clip.
set -u

MODEL=""; OUT="$HOME/asr-evidence/v2"; PORT=8600; PHASE=all
W=3; T=8; CAP=96; QUANT=int8; LOOKAHEAD=3
LADDER="8 16 24 32"; SOAK_C=16; SOAK_S=1800; SOAKS=2
LADDER_S=90; WARMUP=30; WINDOW=60; DUMP_EVERY=30
SERVER_CPUS="0-23"; GEN_CPUS="24-31"; REF_IN=""
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2 ;;
        -o) OUT="$2"; shift 2 ;;
        -p) PORT="$2"; shift 2 ;;
        -W) W="$2"; shift 2 ;;
        -T) T="$2"; shift 2 ;;
        -C) CAP="$2"; shift 2 ;;
        -q) QUANT="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --ladder) LADDER="$2"; shift 2 ;;
        --ladder-seconds) LADDER_S="$2"; shift 2 ;;
        --soak-c) SOAK_C="$2"; shift 2 ;;
        --soak-seconds) SOAK_S="$2"; shift 2 ;;
        --soaks) SOAKS="$2"; shift 2 ;;
        --server-cpus) SERVER_CPUS="$2"; shift 2 ;;
        --gen-cpus) GEN_CPUS="$2"; shift 2 ;;
        --reference-file) REF_IN="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "v2_qualify: unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$MODEL" ] || { echo "v2_qualify: -m <model_dir> is required" >&2; exit 2; }
[ -f "$MODEL/mynah.json" ] || { echo "v2_qualify: $MODEL is not a converted model" >&2; exit 2; }
[ -x ./mynah-asr-server ] || { echo "v2_qualify: build first (make)" >&2; exit 2; }

RUN="$OUT/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN" || exit 2
say() { echo "$@" | tee -a "$RUN/run.log"; }
die() { say "REFUSED: $*"; exit 3; }

# The corpus: every committed clip under 20 s.  Mixed durations and mixed
# languages, so the fleet is never fed a synchronized equal-length herd.  The
# three clips over 90 s are excluded ON PURPOSE: one 305 s stream in a 1800 s
# soak is a different experiment, not a longer one.
BANK=$(python3 - <<'PY'
import glob, wave
out = []
for c in sorted(glob.glob("samples/*/*.wav") + glob.glob("tests/audio/*.wav")):
    try:
        w = wave.open(c); d = w.getnframes() / w.getframerate(); w.close()
    except Exception:
        continue
    if d < 20.0:
        out.append(c)
print(" ".join(out))
PY
)
[ -n "$BANK" ] || die "no clip under 20 s found under samples/ or tests/audio/"
NCLIP=$(echo "$BANK" | wc -w | tr -d ' ')

REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
BIN_SHA=$( (sha256sum ./mynah-asr-server 2>/dev/null || shasum -a 256 ./mynah-asr-server) | cut -d' ' -f1)
[ "$DIRTY" = "0" ] || die "dirty tree ($DIRTY files): a qualification measures a commit, not a desk (ENGINEERING.md §13)"

say "v2_qualify  commit=$REV binary=$BIN_SHA model=$MODEL quant=$QUANT"
say "            topology ${W}x${T} cap=$CAP  server_cpus=$SERVER_CPUS  gen_cpus=$GEN_CPUS"
say "            corpus $NCLIP clips < 20 s   evidence -> $RUN"

# ---------------------------------------------------------------- start a fleet
SRV=""
start_fleet() {   # start_fleet <tag>
    tag="$1"
    taskset -c "$SERVER_CPUS" ./mynah-asr-server -m "$MODEL" --quant "$QUANT" -p "$PORT" \
        --prefork "$W" --prefork-threads "$T" --threads "$T" --cap "$CAP" \
        --batch-window-ms 0 --metrics-port $(( PORT + 1000 )) \
        > "$RUN/server-$tag.log" 2>&1 &
    SRV=$!
    i=0; while [ $i -lt 300 ]; do
        curl -sf -m 2 "http://localhost:$PORT/v1/health" >/dev/null 2>&1 && break
        sleep 0.2; i=$(( i + 1 ))
    done
    curl -sf -m 2 "http://localhost:$PORT/v1/health" >/dev/null 2>&1 || {
        cat "$RUN/server-$tag.log"; kill $SRV 2>/dev/null; die "$tag: the fleet never became ready"; }
    # The pinning is the measurement's foundation, so it is proven, not trusted:
    # every worker mask must live INSIDE the server slice and no two may overlap.
    : > "$RUN/masks-$tag.txt"
    for pid in $(pgrep -P $SRV 2>/dev/null); do
        printf '%s %s\n' "$pid" "$(taskset -pc $pid 2>/dev/null | sed 's/.*: //')" >> "$RUN/masks-$tag.txt"
    done
    python3 - "$RUN/masks-$tag.txt" "$SERVER_CPUS" "$W" <<'PY' || { kill -TERM $SRV 2>/dev/null; die "$tag: worker pinning is wrong (see masks-$tag.txt)"; }
import sys
def expand(s):
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out |= set(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return out
path, slice_s, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
allowed = expand(slice_s)
rows = [l.split(None, 1) for l in open(path) if l.strip()]
if len(rows) != want:
    sys.exit(f"{len(rows)} worker(s) pinned, {want} expected")
seen = set()
for pid, mask in rows:
    m = expand(mask.strip())
    if not m <= allowed:
        sys.exit(f"worker {pid} mask {mask.strip()} escapes the server slice {slice_s}")
    if m & seen:
        sys.exit(f"worker {pid} mask {mask.strip()} overlaps another worker")
    seen |= m
print(f"masks ok: {len(rows)} disjoint workers inside {slice_s}")
PY
    say "$tag: fleet up, $(wc -l < "$RUN/masks-$tag.txt" | tr -d ' ') workers pinned inside $SERVER_CPUS"
}
stop_fleet() {
    [ -n "$SRV" ] || return 0
    curl -s -m 5 "http://localhost:$(( PORT + 1000 ))/metrics" > "$RUN/metrics-$1.txt" 2>&1
    kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""
    sleep 1
}
# SIGUSR1 every DUMP_EVERY seconds for as long as the fleet lives.  The parent
# forwards it to every worker (server/obs.c), so one signal dumps the fleet.
#
# The dump carries no RSS and no roster, so bounds 11 (memory growth) and 12
# (worker deaths) would have had no evidence at all. They are sampled from the
# OS on the same tick, which also dates every sample against the dump beside it.
dumper_start() {   # dumper_start <tag>
    ( while kill -0 $SRV 2>/dev/null; do
          sleep "$DUMP_EVERY"
          kill -USR1 $SRV 2>/dev/null
          printf 't=%s pids=%s\n' "$(date -u +%H:%M:%S)" "$(pgrep -P $SRV 2>/dev/null | tr '\n' ',')" \
              >> "$RUN/procsample-$1.txt"
          ps -o pid=,rss=,stat=,etimes= -p "$(pgrep -P $SRV 2>/dev/null | tr '\n' ',' | sed 's/,$//')" \
              2>/dev/null >> "$RUN/procsample-$1.txt"
      done ) & DUMPER=$!; }
dumper_stop()  { kill $DUMPER 2>/dev/null; DUMPER=""; }
trap 'dumper_stop; [ -n "$SRV" ] && kill -TERM $SRV 2>/dev/null; exit 130' INT TERM

# Where the speech in each clip actually starts. Without it TTFP charges the
# server for whatever room tone the clip leads with: the unloaded C=1 pass over
# this corpus reads p50 1516 ms and p95 4418 ms, and the gap is the bank, not
# the fleet. TTFP has no registered bound either way, but a number that is
# reported has to be a number that means something.
ONSETS="$RUN/onsets.json"
python3 tools/bench/clip_onset.py $BANK --json "$ONSETS" > "$RUN/onsets.txt" 2>&1 \
    || { say "WARNING onset detection failed; TTFP will be reported from stream open only"; ONSETS=""; }

load() { taskset -c "$GEN_CPUS" python3 tools/bench/stream_load.py --port "$PORT" \
             --lookahead "$LOOKAHEAD" --bank short,medium --class-bounds 8,20 \
             ${ONSETS:+--onsets "$ONSETS"} "$@"; }

# ------------------------------------------------------------------ 1. freeze
if [ "$PHASE" = all ] || [ "$PHASE" = freeze ]; then
    say "--- V2-1 freeze"
    (uname -a; echo; lscpu 2>/dev/null; echo; nproc; echo; free -g; cat /proc/loadavg) > "$RUN/host.txt" 2>&1
    ./mynah-asr --dispatch-map > "$RUN/dispatch.txt" 2>&1
    ./mynah-asr --dispatch-map --json > "$RUN/dispatch.json" 2>&1
    ./mynah-asr-server --prefork-plan > "$RUN/plan.txt" 2>&1
    UNKNOWN=$(sed -n 's/^\([0-9][0-9]*\) row(s) UNKNOWN.*/\1/p' "$RUN/dispatch.txt" | tail -1)
    [ "${UNKNOWN:-0}" = "0" ] || die "$UNKNOWN dispatch row(s) UNKNOWN: the binary cannot say what it runs"
    LOAD=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$LOAD" 'BEGIN { exit !(l >= 2.0) }' && die "loadavg $LOAD >= 2.0: another load owns this box"
    say "freeze ok: dispatch has no UNKNOWN row, loadavg $LOAD"
fi

# --------------------------------------------------- 2. the unloaded reference
# Phases can run in separate invocations, and a soak-only run must NOT quietly
# lose the parity check because the reference lived in another run directory.
# --reference-file carries it forward explicitly; nothing is inferred.
REFJSON="$RUN/reference.json"
if [ -n "$REF_IN" ]; then
    [ -f "$REF_IN" ] || die "--reference-file $REF_IN does not exist"
    cp "$REF_IN" "$REFJSON" || die "could not copy $REF_IN"
    say "reference: reusing $REF_IN ($(python3 -c "import json;print(len(json.load(open('$REFJSON'))))") clips)"
elif [ "$PHASE" = all ] || [ "$PHASE" = reference ]; then
    say "--- V2-6 reference: C=1, unloaded, $NCLIP clips, from the server"
    start_fleet ref
    load --mode wave --streams 1 --repeat "$NCLIP" --clips $BANK \
         --json "$RUN/ref-raw.json" 2>&1 | tee -a "$RUN/run.log" | grep -E "verdict|utterance|pacing" || true
    stop_fleet ref
    python3 - "$RUN/ref-raw.json" "$REFJSON" "$NCLIP" <<'PY' || die "the reference pass did not cover the corpus"
import json, sys
raw, out, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(raw))
ref, bad = {}, []
for u in d["utterances"]:
    if u.get("error") or u.get("rejected"):
        bad.append((u.get("clip"), u.get("error"))); continue
    c, t = u["clip"], u.get("text") or ""
    if c in ref and ref[c] != t:
        sys.exit(f"{c}: the UNLOADED server gave two different transcripts; "
                 f"there is nothing to compare under load until that is explained")
    ref[c] = t
if bad:
    sys.exit(f"{len(bad)} utterance(s) failed in the unloaded pass: {bad[:3]}")
if len(ref) != want:
    sys.exit(f"{len(ref)} clip(s) covered, {want} in the corpus: a reference with holes "
             f"silently exempts the clips it is missing")
empty = [c for c, t in ref.items() if not t.strip()]
if empty:
    sys.exit(f"{len(empty)} clip(s) transcribed to nothing unloaded: {empty[:3]}")
json.dump(ref, open(out, "w"), indent=1, ensure_ascii=False)
print(f"reference: {len(ref)} clips, all non-empty")
PY
    say "reference: $(python3 -c "import json;print(len(json.load(open('$REFJSON'))))") clips -> $REFJSON"
fi
[ -f "$REFJSON" ] || REFJSON=""

# -------------------------------------------------------------- 3. the ladder
if [ "$PHASE" = all ] || [ "$PHASE" = ladder ]; then
    say "--- V2-3 ladder ($LADDER), ${LADDER_S}s per rung, fresh server per rung"
    for C in $LADDER; do
        start_fleet "ladder-C$C"
        dumper_start "ladder-C$C"
        say "--- LADDER C=$C"
        load --mode soak --streams "$C" --duration "$LADDER_S" --warmup 10 --window 30 --seed 42 \
             --clips $BANK ${REFJSON:+--reference "$REFJSON"} \
             --json "$RUN/ladder-C$C.json" 2>&1 | tee -a "$RUN/run.log" \
             | grep -E "TTFP|emission lag|finaliz|backlog|drift|verdict|utterances|pacing|REFERENCE|identity" || true
        dumper_stop
        stop_fleet "ladder-C$C"
    done
fi

# ---------------------------------------------------------------- 4. the soaks
if [ "$PHASE" = all ] || [ "$PHASE" = soak ]; then
    n=1
    while [ "$n" -le "$SOAKS" ]; do
        say "--- V2-4 soak $n/$SOAKS: C=$SOAK_C for ${SOAK_S}s, fresh server"
        start_fleet "soak$n"
        dumper_start "soak$n"
        load --mode soak --streams "$SOAK_C" --duration "$SOAK_S" --warmup "$WARMUP" \
             --window "$WINDOW" --seed $(( 42 + n )) --clips $BANK \
             ${REFJSON:+--reference "$REFJSON"} \
             --json "$RUN/soak$n-C$SOAK_C.json" 2>&1 | tee -a "$RUN/run.log" \
             | grep -E "TTFP|emission lag|finaliz|backlog|drift|verdict|utterances|pacing|REFERENCE|identity|window" || true
        dumper_stop
        stop_fleet "soak$n"
        n=$(( n + 1 ))
    done
fi

cat > "$RUN/manifest.json" <<JSON
{ "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "commit": "$REV", "dirty_files": $DIRTY,
  "binary_sha256": "$BIN_SHA", "model": "$MODEL", "quant": "$QUANT", "lookahead": $LOOKAHEAD,
  "workers": $W, "threads_per_worker": $T, "cap": $CAP,
  "server_cpus": "$SERVER_CPUS", "gen_cpus": "$GEN_CPUS",
  "corpus_clips": $NCLIP, "ladder": "$LADDER", "ladder_seconds": $LADDER_S,
  "soak_concurrency": $SOAK_C, "soak_seconds": $SOAK_S, "soaks": $SOAKS,
  "warmup_s": $WARMUP, "window_s": $WINDOW, "dump_every_s": $DUMP_EVERY,
  "registered_bounds": ".work/server-v2-qualification.md V2-2" }
JSON
say "done. Evidence in $RUN"
