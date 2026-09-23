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
#       [--corpus samples/stress-en/manifest.json] [--min-peak-dbfs -30]
#       [--corpus-sample 500] [--corpus-seed 42]
#                             a bank manifest instead of the committed clips.
#                             --corpus-sample takes a deterministic, class-
#                             balanced subset: the unloaded reference pass runs
#                             at real time, so it costs exactly the corpus's own
#                             duration, and 1316 clips is 5.8 hours of it
#       [--http-threads 32]   per worker; W x this is the fleet's CONNECTION
#                             ceiling, NOT its compute width
#       --phase genscale [--gen-arms "24-31 24-27 24-25"] [--genscale-c 32]
#                             how few cpus the LOAD GENERATOR needs. The server
#                             is identical in every arm and the cpus the
#                             generator gives up are left EMPTY -- this is an
#                             attribution experiment, not a capacity one
#       [--server-cpus 0-23] [--gen-cpus 24-31] [-W 3] [-T 8] [-C 96]
#
# It refuses rather than produce a number it cannot support: a dirty tree, a
# dispatch row the binary cannot name, a worker whose mask escapes the server
# slice, a fleet that never became ready, or a corpus class with no clip.
set -u

MODEL=""; OUT="$HOME/asr-evidence/v2"; PORT=8600; PHASE=all
W=3; T=8; CAP=96; QUANT=int8; LOOKAHEAD=3; HTTP_T=32
LADDER="8 16 24 32"; SOAK_C=16; SOAK_S=1800; SOAKS=2
GEN_ARMS="24-31 24-27 24-25"; GENSCALE_C=32; GENSCALE_S=180
CORPUS=""; MIN_PEAK_DBFS=-30; CLASSES=""; CORPUS_SAMPLE=0; CORPUS_SEED=42; REF_C=1
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
        --http-threads) HTTP_T="$2"; shift 2 ;;
        -q) QUANT="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --ladder) LADDER="$2"; shift 2 ;;
        --ladder-seconds) LADDER_S="$2"; shift 2 ;;
        --soak-c) SOAK_C="$2"; shift 2 ;;
        --soak-seconds) SOAK_S="$2"; shift 2 ;;
        --soaks) SOAKS="$2"; shift 2 ;;
        --server-cpus) SERVER_CPUS="$2"; shift 2 ;;
        --gen-cpus) GEN_CPUS="$2"; shift 2 ;;
        --gen-arms) GEN_ARMS="$2"; shift 2 ;;
        --genscale-c) GENSCALE_C="$2"; shift 2 ;;
        --genscale-seconds) GENSCALE_S="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        --min-peak-dbfs) MIN_PEAK_DBFS="$2"; shift 2 ;;
        --corpus-sample) CORPUS_SAMPLE="$2"; shift 2 ;;
        --corpus-seed) CORPUS_SEED="$2"; shift 2 ;;
        --ref-c) REF_C="$2"; shift 2 ;;
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
# The corpus. Without --corpus: every committed clip under 20 s -- what a bare
# clone has, enough to run, and far too small to load-test with. 27 clips over a
# 1800 s soak at C=16 is each clip played about 145 times, which measures a warm
# page cache as much as a fleet.
#
# With --corpus <manifest.json>: a bank manifest (samples/stress-en). Clips whose
# peak is below --min-peak-dbfs are dropped, because FLEURS levels vary by about
# 35 dB and nothing normalises them: a clip nobody can hear still costs a full
# step, so it is fine as LOAD and useless for the transcript parity gate, which
# needs the unloaded pass to produce something to compare.
BANK=$(CORPUS="$CORPUS" MIN_PEAK="$MIN_PEAK_DBFS" SAMPLE="$CORPUS_SAMPLE" \
       SEED="$CORPUS_SEED" python3 - <<'BANKPY'
import glob, json, os, random, wave
man, minpeak = os.environ.get("CORPUS", ""), float(os.environ.get("MIN_PEAK", "-30"))
sample, seed = int(os.environ.get("SAMPLE", "0")), int(os.environ.get("SEED", "42"))
out = []
if man:
    d = json.load(open(man))
    root = os.path.dirname(man)
    clips = d["clips"] if isinstance(d, dict) and "clips" in d else d
    if isinstance(clips, dict):
        clips = [v for v in clips.values()]
    quiet = 0
    for c in clips:
        f = c.get("file") or c.get("clip")
        if not f:
            continue
        p = f if os.path.exists(f) else os.path.join(root, f)
        if not os.path.exists(p):
            continue
        pk = c.get("peak_dbfs")
        if pk is not None and pk < minpeak:
            quiet += 1
            continue
        out.append((c.get("class") or "?", p))
    if sample and sample < len(out):
        # Class-balanced and seeded, so the same --corpus-sample/--corpus-seed
        # always selects the same clips and the bank hash identifies WHICH.
        byc = {}
        for k, f in out:
            byc.setdefault(k, []).append(f)
        picked = []
        per = max(1, sample // max(1, len(byc)))
        for k in sorted(byc):
            pool = sorted(byc[k])
            random.Random(f"{seed}:{k}").shuffle(pool)
            picked += pool[:per]
        out = [("", f) for f in picked]
    print(" ".join(sorted(f for _, f in out)))
else:
    for c in sorted(glob.glob("samples/*/*.wav") + glob.glob("tests/audio/*.wav")):
        try:
            w = wave.open(c); dur = w.getnframes() / w.getframerate(); w.close()
        except Exception:
            continue
        if dur < 20.0:
            out.append(c)
    print(" ".join(out))
BANKPY
)
[ -n "$BANK" ] || die "the corpus is empty (--corpus '$CORPUS', --min-peak-dbfs $MIN_PEAK_DBFS)"
NCLIP=$(echo "$BANK" | wc -w | tr -d ' ')
# The corpus is part of the claim, so it is identified the way the qualified
# PocketTTS profile identifies its text bank: by a hash over the clips
# themselves, not by a count. A bank that changed under a promoted profile must
# be visible as a different bank.
BANK_SHA=$(for c in $BANK; do printf '%s %s\n' "$c" \
    "$( (sha256sum "$c" 2>/dev/null || shasum -a 256 "$c") | cut -d' ' -f1)"; done \
    | (sha256sum 2>/dev/null || shasum -a 256) | cut -c1-16)

REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
BIN_SHA=$( (sha256sum ./mynah-asr-server 2>/dev/null || shasum -a 256 ./mynah-asr-server) | cut -d' ' -f1)
[ "$DIRTY" = "0" ] || die "dirty tree ($DIRTY files): a qualification measures a commit, not a desk (ENGINEERING.md §13)"

for c in $BANK; do printf '%s %s\n' "$( (sha256sum "$c" 2>/dev/null || shasum -a 256 "$c") | cut -d' ' -f1)" "$c"; done > "$RUN/bank.txt"
say "v2_qualify  commit=$REV binary=$BIN_SHA model=$MODEL quant=$QUANT"
say "            topology ${W}x${T} cap=$CAP  server_cpus=$SERVER_CPUS  gen_cpus=$GEN_CPUS"
say "            corpus $NCLIP clips < 20 s, bank-sha256 $BANK_SHA   evidence -> $RUN"

# ---------------------------------------------------------------- start a fleet
SRV=""
start_fleet() {   # start_fleet <tag>
    tag="$1"
    # --threads is the HTTP pool and a WebSocket stream holds one of its threads
    # for its whole life (server/main.c: "one slot per HTTP thread"), so W x it
    # is the fleet's CONNECTION ceiling. It is NOT the compute width: each
    # worker's pool sizes itself from its affinity slice, which --prefork-threads
    # sets. Passing --threads $T (copied from box_qualify.sh) gave 3x8 = 24
    # connections and silently capped the C=24 and C=32 rungs of 2026-09-22 at
    # 8 active slots per worker -- C=32 ran 24 streams and reported fewer
    # utterances and LOWER throughput than C=24, which read as noise.
    taskset -c "$SERVER_CPUS" ./mynah-asr-server -m "$MODEL" --quant "$QUANT" -p "$PORT" \
        --prefork "$W" --prefork-threads "$T" --threads "$HTTP_T" --cap "$CAP" \
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
    # Proven from the worker's own banner, not from the flag we passed.
    RES_HTTP=$(sed -n 's/.*(\([0-9][0-9]*\) http threads.*/\1/p' "$RUN/server-$tag.log" | head -1)
    RES_SLOTS=$(sed -n 's/.*http threads, \([0-9][0-9]*\) stream slots.*/\1/p' "$RUN/server-$tag.log" | head -1)
    [ "${RES_HTTP:-0}" = "$HTTP_T" ] || { kill -TERM $SRV 2>/dev/null
        die "$tag: asked for $HTTP_T http threads, the worker reports ${RES_HTTP:-none}"; }
    [ "${RES_SLOTS:-0}" = "$CAP" ] || { kill -TERM $SRV 2>/dev/null
        die "$tag: asked for $CAP stream slots, the worker reports ${RES_SLOTS:-none}"; }
    say "$tag: fleet up, $(wc -l < "$RUN/masks-$tag.txt" | tr -d ' ') workers pinned inside $SERVER_CPUS, \
$RES_HTTP http threads x $W = $(( RES_HTTP * W )) connections, $RES_SLOTS slots each"
}

# A concurrency above the fleet's connection ceiling does not measure capacity,
# it measures the ceiling: the surplus clients sit waiting for an HTTP thread
# while the run reports no error and no 503.
check_ceiling() {   # check_ceiling <C>
    lim=$(( HTTP_T * W ))
    [ "$1" -le "$lim" ] || die "C=$1 exceeds this fleet's connection ceiling of $lim \
($W workers x $HTTP_T http threads): the rung would measure the ceiling, not the machine. \
Raise --http-threads."
}
stop_fleet() {
    [ -n "$SRV" ] || return 0
    # The router's /metrics carries no scheduler counter -- it says so itself.
    # A worker's /v1/health does: batched_steps_total, ready_mean and the
    # per-B step table, which is the only honest source for the batch.
    curl -s -m 5 "http://localhost:$PORT/v1/health" > "$RUN/health-$1.json" 2>&1
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

# Which duration classes this corpus actually populates. build_bank() exits if a
# named class has no clip, so naming them by hand ties the tool to one bank.
CLASSES=$(python3 - $BANK <<'CLSPY'
import sys, wave
have = set()
for c in sys.argv[1:]:
    try:
        w = wave.open(c); d = w.getnframes() / w.getframerate(); w.close()
    except Exception:
        continue
    have.add("short" if d < 8 else "medium" if d < 20 else "long")
print(",".join(k for k in ("short", "medium", "long") if k in have))
CLSPY
)
[ -n "$CLASSES" ] || die "no clip fell into a duration class"
say "            classes present: $CLASSES"

load() { taskset -c "$GEN_CPUS" python3 tools/bench/stream_load.py --port "$PORT" \
             --lookahead "$LOOKAHEAD" --bank "$CLASSES" --class-bounds 8,20 \
             ${ONSETS:+--onsets "$ONSETS"} "$@"; }

# ------------------------------------------------------------------ 1. freeze
# UNCONDITIONAL. A run that measures without recording what it measured is the
# failure mode this campaign exists to end, and --phase soak used to skip this
# block entirely: the soak that promotes a profile would have carried no
# dispatch map, no topology plan and no host record of its own.
if true; then
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
    # The TTFP baseline travels with the text reference or the paired gate has
    # nothing to pair against, and would quietly not be applied at all.
    REF_TTFP_IN=$(echo "$REF_IN" | sed 's/\.json$/-ttfp.json/')
    if [ -f "$REF_TTFP_IN" ]; then
        cp "$REF_TTFP_IN" "$RUN/reference-ttfp.json"
        say "reference: TTFP baseline carried ($(python3 -c "import json;print(len(json.load(open('$RUN/reference-ttfp.json'))))") clips)"
    else
        say "WARNING no TTFP baseline beside $REF_IN: the paired TTFP gate cannot be applied"
    fi
    # A reused reference that does not cover this corpus does not fail: the
    # clips it is missing are simply never compared. So coverage is checked
    # here too, not only where the reference is generated.
    python3 - "$REFJSON" $BANK <<'REFCOV' || die "the reused reference does not cover this corpus"
import json, sys
ref = json.load(open(sys.argv[1]))
clips = sys.argv[2:]
missing = [c for c in clips if c not in ref]
if missing:
    sys.exit(f"{len(missing)} clip(s) of this corpus have no reference: {missing[:3]}")
extra = [c for c in ref if c not in clips]
print(f"reference covers all {len(clips)} clips"
      + (f", {len(extra)} unused entries" if extra else ""))
REFCOV
    say "reference: reusing $REF_IN ($(python3 -c "import json;print(len(json.load(open('$REFJSON'))))") clips)"
elif [ "$PHASE" = all ] || [ "$PHASE" = reference ]; then
    # The pass runs at REAL TIME, so it costs exactly the corpus's own duration
    # divided by --ref-c. C=1 is the cleanest baseline and 498 clips of it is
    # 110 minutes; C=4 on a 24-core fleet is about 7% duty, which is unloaded by
    # any measure this campaign uses, and costs 27. Whatever queueing C=4 does
    # add inflates the BASELINE, which makes bound 13 more lenient, never
    # stricter -- so it cannot manufacture a pass that idle would have failed.
    say "--- V2-6 reference: C=$REF_C, $NCLIP clips, from the server"
    start_fleet ref
    load --mode wave --streams "$REF_C" --repeat $(( (NCLIP + REF_C - 1) / REF_C )) --clips $BANK \
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

# The unloaded TTFP of each clip, kept so a loaded run can be compared with it
# PAIRED. Subtracting two p95s conflates the corpus with the fleet: a clip that
# leads with 700 ms of silence carries those 700 ms loaded and unloaded alike,
# and in a paired difference it vanishes. Raw open->first-partial stays reported
# because it is what a listener waits, but it cannot be the server's gate.
base = {}
for u in d["utterances"]:
    if u.get("error") or u.get("rejected") or u.get("ttfp_ms") is None:
        continue
    base[u["clip"]] = {"ttfp_ms": u["ttfp_ms"],
                       "ttfp_from_onset_ms": u.get("ttfp_from_onset_ms"),
                       "first_delta_lag_ms": u.get("first_delta_lag_ms"),
                       "first_delta_audio_s": u.get("first_delta_audio_s")}
json.dump(base, open(out.replace(".json", "-ttfp.json"), "w"), indent=1)
print(f"reference: {len(ref)} clips, all non-empty; {len(base)} with an unloaded TTFP")
PY
    say "reference: $(python3 -c "import json;print(len(json.load(open('$REFJSON'))))") clips -> $REFJSON"
fi
[ -f "$REFJSON" ] || REFJSON=""

# -------------------------------------------------------------- 3. the ladder
if [ "$PHASE" = all ] || [ "$PHASE" = ladder ]; then
    say "--- V2-3 ladder ($LADDER), ${LADDER_S}s per rung, fresh server per rung"
    for C in $LADDER; do
        check_ceiling "$C"
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

# ------------------------------------------------- 3b. how much does the GENERATOR need
#
# Eight physical cores are withheld from the server so the Python load generator
# cannot contend with it. That reservation may be far larger than the generator
# needs, and in production this generator does not run on the inference host at
# all -- so withholding a quarter of the machine may be understating the server.
#
# This measures the generator ALONE. The server stays byte-identical at 3x8 on
# the same cpus in every arm, and the cpus the generator gives up are left
# EMPTY. Handing them to the server in the same experiment would change two
# things at once and attribute the result to whichever one we preferred.
if [ "$PHASE" = genscale ]; then
    say "--- G1 generator headroom: server FROZEN at ${W}x${T} on $SERVER_CPUS, \
recovered cpus deliberately left idle"
    for ARM in $GEN_ARMS; do
        check_ceiling "$GENSCALE_C"
        tag="gen$ARM"
        start_fleet "$tag"
        dumper_start "$tag"
        grep -E '^cpu[0-9]+ ' /proc/stat > "$RUN/cpu-before-$tag.txt"
        say "--- GENSCALE arm $ARM (C=$GENSCALE_C, ${GENSCALE_S}s)"
        GEN_CPUS="$ARM" load --mode soak --streams "$GENSCALE_C" --duration "$GENSCALE_S" \
             --warmup 20 --window 30 --seed 42 --clips $BANK \
             ${REFJSON:+--reference "$REFJSON"} \
             --json "$RUN/genscale-$tag.json" 2>&1 | tee -a "$RUN/run.log" \
             | grep -E "emission lag|backlog|verdict|utterances|pacing|late|REFERENCE|identity" || true
        grep -E '^cpu[0-9]+ ' /proc/stat > "$RUN/cpu-after-$tag.txt"
        dumper_stop
        stop_fleet "$tag"
        python3 - "$RUN/cpu-before-$tag.txt" "$RUN/cpu-after-$tag.txt" "$ARM" <<'CPUD' | tee -a "$RUN/run.log"
import sys
def load(p):
    out = {}
    for line in open(p):
        f = line.split()
        if f and f[0].startswith("cpu") and f[0][3:].isdigit():
            v = [int(x) for x in f[1:]]
            out[int(f[0][3:])] = (sum(v), v[3] + (v[4] if len(v) > 4 else 0))
    return out
def expand(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out
a, b, cpus = load(sys.argv[1]), load(sys.argv[2]), expand(sys.argv[3])
busy = []
for c in cpus:
    if c not in a or c not in b:
        continue
    dt, di = b[c][0] - a[c][0], b[c][1] - a[c][1]
    busy.append(0.0 if dt <= 0 else 1.0 - di / dt)
if busy:
    print("    generator cpu busy: " + "  ".join(f"cpu{c} {u*100:.1f}%"
          for c, u in zip(cpus, busy))
          + f"   mean {sum(busy)/len(busy)*100:.1f}%  total {sum(busy):.2f} cores")
CPUD
    done
fi

# ---------------------------------------------------------------- 4. the soaks
if [ "$PHASE" = all ] || [ "$PHASE" = soak ]; then
    n=1
    while [ "$n" -le "$SOAKS" ]; do
        check_ceiling "$SOAK_C"
        say "--- V2-4 soak $n/$SOAKS: C=$SOAK_C for ${SOAK_S}s, fresh server"
        start_fleet "soak$n"
        dumper_start "soak$n"
        # Qualification keeps the published partials for a deterministic slice,
        # so the PASS can be audited later without re-running the server. The
        # discovery ladder does not: it certifies nothing and the records would
        # be an order of magnitude larger for no reader.
        load --mode soak --streams "$SOAK_C" --duration "$SOAK_S" --warmup "$WARMUP" \
             --window "$WINDOW" --seed $(( 42 + n )) --clips $BANK --keep-events 25 \
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
  "workers": $W, "threads_per_worker": $T, "cap": $CAP, "http_threads_per_worker": $HTTP_T,
  "connection_ceiling": $(( HTTP_T * W )),
  "server_cpus": "$SERVER_CPUS", "gen_cpus": "$GEN_CPUS",
  "corpus_clips": $NCLIP, "bank_sha256": "$BANK_SHA",
  "corpus": "${CORPUS:-committed samples under 20 s}", "corpus_sample": $CORPUS_SAMPLE,
  "corpus_seed": $CORPUS_SEED, "min_peak_dbfs": $MIN_PEAK_DBFS, "reference_concurrency": $REF_C,
  "configuration": "shipped-default",
  "ladder": "$LADDER", "ladder_seconds": $LADDER_S,
  "soak_concurrency": $SOAK_C, "soak_seconds": $SOAK_S, "soaks": $SOAKS,
  "warmup_s": $WARMUP, "window_s": $WINDOW, "dump_every_s": $DUMP_EVERY,
  "registered_bounds": ".work/server-v2-qualification.md V2-2" }
JSON
say "done. Evidence in $RUN"
