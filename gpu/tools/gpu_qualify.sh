#!/bin/bash
# gpu/tools/gpu_qualify.sh — the V2 qualification methodology for
# mynah-asr-server-cuda (S14-7). It writes the SAME artefacts
# tools/bench/v2_qualify.sh writes, so tools/bench/v2_verdict.py and
# tools/bench/v2_promote.py judge a GPU run exactly as they judge a CPU one:
# manifest.json, reference.json (+ -ttfp.json), onsets.json, ladder-C*.json,
# soak*-C*.json, server-<tag>.log with the [DUMP] lines, procsample-<tag>.txt.
#
# Why a separate runner rather than a flag on v2_qualify.sh: that script is
# built around prefork (per-worker masks proven with taskset, a router, a
# connection ceiling of W x http threads) and a loadavg gate. None of those
# mean anything for one process on one GPU in a container, whose loadavg is
# the HOST's. The shared harness stays byte-identical (S14: the CPU path is not
# touched); what is shared is the corpus rule, the load generator, the metrics
# module and the verdict.
#
#   gpu/tools/gpu_qualify.sh -m models/nemotron-3.5-asr-streaming-0.6b \
#       --corpus samples/stress-en/manifest.json --corpus-sample 498 \
#       [--phase all|reference|ladder|soak] [--ladder "96 128 144 160"] [--ladder-seconds 90]
#       [--soak-c 128] [--soak-seconds 1800] [--soaks 2] [--seed 42]
#       [--cohort-ms 40] [--cap 192] [--lang auto] [--lookahead 3] [--ref-c 4] [--gemm own|splitk]
#       [--reference-file <run>/reference.json] [--out ~/asr-evidence/gpu]
#
# Refuses: a dirty tree, a GPU another process is using, a dispatch map with an
# UNKNOWN row, a container already busy on CPU (its own cgroup, not loadavg).
set -u
cd "$(dirname "$0")/../.." || exit 2

MODEL=""; PHASE=all; LADDER="96 128 144 160"; LADDER_S=90; SOAK_C=128; SOAK_S=1800; SOAKS=2
SEED=42; COHORT=40; CAP=192; LANG_Q=auto; LOOKAHEAD=3; REF_C=4; REF_IN=""
CORPUS=""; CORPUS_SAMPLE=0; CORPUS_SEED=42; MIN_PEAK_DBFS=-30
WARMUP=30; WINDOW=60; DUMP_EVERY=30; OUT="$HOME/asr-evidence/gpu"; PORT=8291; BIN=./mynah-asr-server-cuda; GEMM=own
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --ladder) LADDER="$2"; shift 2 ;;
        --ladder-seconds) LADDER_S="$2"; shift 2 ;;
        --soak-c) SOAK_C="$2"; shift 2 ;;
        --soak-seconds) SOAK_S="$2"; shift 2 ;;
        --soaks) SOAKS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --cohort-ms) COHORT="$2"; shift 2 ;;
        --cap) CAP="$2"; shift 2 ;;
        --lang) LANG_Q="$2"; shift 2 ;;
        --lookahead) LOOKAHEAD="$2"; shift 2 ;;
        --ref-c) REF_C="$2"; shift 2 ;;
        --reference-file) REF_IN="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        --corpus-sample) CORPUS_SAMPLE="$2"; shift 2 ;;
        --corpus-seed) CORPUS_SEED="$2"; shift 2 ;;
        --min-peak-dbfs) MIN_PEAK_DBFS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --window) WINDOW="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gemm) GEMM="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gpu_qualify: unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$MODEL" ] && [ -f "$MODEL/mynah.json" ] || { echo "gpu_qualify: -m <converted model dir> is required" >&2; exit 2; }
[ -x "$BIN" ] || { echo "gpu_qualify: build first (make -C gpu)" >&2; exit 2; }

RUN="$OUT/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN" || exit 2
say() { echo "$@" | tee -a "$RUN/run.log"; }
die() { say "REFUSED: $*"; exit 3; }

# ------------------------------------------------------------ the corpus
# The rule of tools/bench/v2_qualify.sh, verbatim in effect: with --corpus, clips
# below --min-peak-dbfs are dropped and --corpus-sample picks a class-balanced
# seeded sample, so the same flags select the same clips and the bank hash says
# which. The GPU run and the CPU runs it is compared with share that hash.
[ -n "$CORPUS" ] || die "--corpus is required (a qualification runs on a bank)"
BANK=$(CORPUS="$CORPUS" MIN_PEAK="$MIN_PEAK_DBFS" SAMPLE="$CORPUS_SAMPLE" SEED="$CORPUS_SEED" python3 - <<'BANKPY'
import json, os, random
man, minpeak = os.environ["CORPUS"], float(os.environ["MIN_PEAK"])
sample, seed = int(os.environ["SAMPLE"]), int(os.environ["SEED"])
d = json.load(open(man)); root = os.path.dirname(man)
clips = d["clips"] if isinstance(d, dict) and "clips" in d else d
if isinstance(clips, dict):
    clips = list(clips.values())
out = []
for c in clips:
    f = c.get("file") or c.get("clip")
    if not f:
        continue
    p = f if os.path.exists(f) else os.path.join(root, f)
    if not os.path.exists(p):
        continue
    pk = c.get("peak_dbfs")
    if pk is not None and pk < minpeak:
        continue
    out.append((c.get("class") or "?", p))
if sample and sample < len(out):
    byc = {}
    for k, f in out:
        byc.setdefault(k, []).append(f)
    picked, per = [], max(1, sample // max(1, len(byc)))
    for k in sorted(byc):
        pool = sorted(byc[k]); random.Random(f"{seed}:{k}").shuffle(pool); picked += pool[:per]
    out = [("", f) for f in picked]
print(" ".join(sorted(f for _, f in out)))
BANKPY
)
NCLIP=$(echo $BANK | wc -w | tr -d ' ')
[ "$NCLIP" -gt 0 ] || die "the corpus selected no clip"
BANK_SHA=$(for c in $BANK; do printf '%s %s\n' "$c" "$(sha256sum "$c" | cut -d' ' -f1)"; done | sha256sum | cut -c1-16)
for c in $BANK; do printf '%s %s\n' "$(sha256sum "$c" | cut -d' ' -f1)" "$c"; done > "$RUN/bank.txt"

REV=$(git rev-parse --short HEAD); DIRTY=$(git status --porcelain | grep -v '^??' | wc -l | tr -d ' ')
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
[ "$DIRTY" = "0" ] || die "dirty tree ($DIRTY tracked files): a qualification measures a commit (ENGINEERING.md §12)"
say "gpu_qualify commit=$REV binary=$BIN_SHA model=$MODEL gemm=$GEMM cohort_ms=$COHORT cap=$CAP lang=$LANG_Q lookahead=$LOOKAHEAD"
say "            corpus $NCLIP clips from $CORPUS, bank-sha256 $BANK_SHA   evidence -> $RUN"

# ------------------------------------------------------------ 1. freeze
{ uname -a; echo; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; echo; free -g; cat /proc/loadavg; } > "$RUN/host.txt" 2>&1
nvidia-smi -q > "$RUN/nvidia-smi.txt" 2>&1
"$BIN" -m "$MODEL" --gemm "$GEMM" --dispatch-map > "$RUN/dispatch.txt" 2>&1 || die "the server could not print its dispatch map (see dispatch.txt)"
UNKNOWN=$(sed -n 's/^\([0-9][0-9]*\) row(s) UNKNOWN.*/\1/p' "$RUN/dispatch.txt" | tail -1)
[ "${UNKNOWN:-1}" = "0" ] || die "dispatch map has UNKNOWN rows or none at all"
OTHERS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
[ "$OTHERS" = "0" ] || die "$OTHERS other process(es) on the GPU"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "${USED:-0}" -lt 1024 ] || die "${USED} MiB of VRAM already used by someone else"
# the container's own CPU use over 5 s, from its cgroup (loadavg is the host's)
cg=/sys/fs/cgroup/cpu.stat
if [ -r "$cg" ]; then
    u0=$(awk '/^usage_usec/{print $2}' $cg); sleep 5; u1=$(awk '/^usage_usec/{print $2}' $cg)
    BUSY=$(awk -v a="$u0" -v b="$u1" 'BEGIN{printf "%.2f", (b-a)/5e6}')
    awk -v b="$BUSY" 'BEGIN{exit !(b >= 1.0)}' && die "this container is already using $BUSY cores"
    say "freeze ok: dispatch has no UNKNOWN row, GPU idle (${USED} MiB), container cpu $BUSY cores"
else
    say "freeze ok: dispatch has no UNKNOWN row, GPU idle (${USED} MiB); no cgroup cpu.stat to read"
fi

# ------------------------------------------------------------ server lifecycle
SRV=""; DUMPER=""; SMI=""
start_server() {   # start_server <tag>
    "$BIN" -m "$MODEL" -p "$PORT" --gemm "$GEMM" --cap "$CAP" --threads $(( CAP + 32 )) --cohort-ms "$COHORT" \
        --metrics-port $(( PORT + 1000 )) > "$RUN/server-$1.log" 2>&1 &
    SRV=$!
    for i in $(seq 1 300); do curl -sf -m 2 "http://localhost:$PORT/v1/health" >/dev/null 2>&1 && break; sleep 0.5; done
    curl -sf -m 2 "http://localhost:$PORT/v1/health" >/dev/null 2>&1 || { cat "$RUN/server-$1.log"; die "$1: the server never became ready"; }
    grep -q "engine=cuda" "$RUN/server-$1.log" || die "$1: the banner does not say engine=cuda"
    nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active \
        --format=csv,noheader -l 2 > "$RUN/gpu-$1.csv" 2>&1 & SMI=$!
    # bounds 11/12 read RSS and the roster from here; the "worker" is the one process
    ( while kill -0 $SRV 2>/dev/null; do
          sleep "$DUMP_EVERY"; kill -USR1 $SRV 2>/dev/null
          printf 't=%s pids=%s\n' "$(date -u +%H:%M:%S)" "$SRV" >> "$RUN/procsample-$1.txt"
          ps -o pid=,rss=,stat=,etimes=,cputimes= -p "$SRV" 2>/dev/null >> "$RUN/procsample-$1.txt"
      done ) & DUMPER=$!
}
stop_server() {   # stop_server <tag>
    curl -s -m 5 "http://localhost:$PORT/v1/health" > "$RUN/health-$1.json" 2>&1
    curl -s -m 5 "http://localhost:$(( PORT + 1000 ))/metrics" > "$RUN/metrics-$1.txt" 2>&1
    kill -USR1 $SRV 2>/dev/null; sleep 1
    kill $DUMPER $SMI 2>/dev/null; wait $DUMPER $SMI 2>/dev/null
    kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""; sleep 1
}
trap '[ -n "$SRV" ] && kill -TERM $SRV 2>/dev/null; kill $DUMPER $SMI 2>/dev/null; exit 130' INT TERM

ONSETS="$RUN/onsets.json"
python3 tools/bench/clip_onset.py $BANK --json "$ONSETS" > "$RUN/onsets.txt" 2>&1 || { say "WARNING onset detection failed"; ONSETS=""; }
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
load() { python3 tools/bench/stream_load.py --port "$PORT" --lookahead "$LOOKAHEAD" --lang "$LANG_Q" \
             --bank "$CLASSES" --class-bounds 8,20 ${ONSETS:+--onsets "$ONSETS"} "$@"; }

# ------------------------------------------------------------ 2. reference
REFJSON="$RUN/reference.json"
if [ -n "$REF_IN" ]; then
    cp "$REF_IN" "$REFJSON" || die "cannot copy $REF_IN"
    t=$(echo "$REF_IN" | sed 's/\.json$/-ttfp.json/'); [ -f "$t" ] && cp "$t" "$RUN/reference-ttfp.json"
    say "reference: reusing $REF_IN"
elif [ "$PHASE" = all ] || [ "$PHASE" = reference ]; then
    say "--- reference: C=$REF_C over $NCLIP clips, from the server itself"
    start_server ref
    load --mode wave --streams "$REF_C" --repeat $(( (NCLIP + REF_C - 1) / REF_C )) --clips $BANK \
         --json "$RUN/ref-raw.json" >> "$RUN/run.log" 2>&1
    stop_server ref
    python3 - "$RUN/ref-raw.json" "$REFJSON" "$NCLIP" <<'PY' || die "the reference pass did not cover the corpus"
import json, sys
raw, out, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(raw)); ref, bad = {}, []
for u in d["utterances"]:
    if u.get("error") or u.get("rejected"):
        bad.append((u.get("clip"), u.get("error"))); continue
    c, t = u["clip"], u.get("text") or ""
    if c in ref and ref[c] != t:
        sys.exit(f"{c}: the UNLOADED server gave two different transcripts")
    ref[c] = t
if bad:
    sys.exit(f"{len(bad)} utterance(s) failed in the unloaded pass: {bad[:3]}")
if len(ref) != want:
    sys.exit(f"{len(ref)} clip(s) covered, {want} in the corpus")
empty = [c for c, t in ref.items() if not t.strip()]
if empty:
    sys.exit(f"{len(empty)} clip(s) transcribed to nothing unloaded: {empty[:3]}")
json.dump(ref, open(out, "w"), indent=1, ensure_ascii=False)
base = {u["clip"]: {"ttfp_ms": u["ttfp_ms"], "ttfp_from_onset_ms": u.get("ttfp_from_onset_ms"),
                    "first_delta_lag_ms": u.get("first_delta_lag_ms"),
                    "first_delta_audio_s": u.get("first_delta_audio_s")}
        for u in d["utterances"] if not (u.get("error") or u.get("rejected") or u.get("ttfp_ms") is None)}
json.dump(base, open(out.replace(".json", "-ttfp.json"), "w"), indent=1)
print(f"reference: {len(ref)} clips, all non-empty; {len(base)} with an unloaded TTFP")
PY
fi
[ -f "$REFJSON" ] || REFJSON=""

# ------------------------------------------------------------ 3. ladder
if [ "$PHASE" = all ] || [ "$PHASE" = ladder ]; then
    say "--- ladder ($LADDER), ${LADDER_S}s per rung, fresh server per rung"
    for C in $LADDER; do
        [ "$C" -le "$CAP" ] || die "C=$C exceeds --cap $CAP"
        start_server "ladder-C$C"
        load --mode soak --streams "$C" --duration "$LADDER_S" --warmup 10 --window 30 --seed "$SEED" \
             --clips $BANK ${REFJSON:+--reference "$REFJSON"} --json "$RUN/ladder-C$C.json" >> "$RUN/run.log" 2>&1
        stop_server "ladder-C$C"
        say "ladder C=$C done"
    done
fi

# ------------------------------------------------------------ 4. soaks
if [ "$PHASE" = all ] || [ "$PHASE" = soak ]; then
    n=1
    while [ "$n" -le "$SOAKS" ]; do
        [ "$SOAK_C" -le "$CAP" ] || die "C=$SOAK_C exceeds --cap $CAP"
        say "--- soak $n/$SOAKS: C=$SOAK_C for ${SOAK_S}s, seed $(( SEED + n - 1 )), fresh server"
        start_server "soak$n"
        load --mode soak --streams "$SOAK_C" --duration "$SOAK_S" --warmup "$WARMUP" --window "$WINDOW" \
             --seed $(( SEED + n - 1 )) --clips $BANK ${REFJSON:+--reference "$REFJSON"} \
             --json "$RUN/soak$n-C$SOAK_C.json" >> "$RUN/run.log" 2>&1
        stop_server "soak$n"
        n=$(( n + 1 ))
    done
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
cat > "$RUN/manifest.json" <<JSON
{ "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "commit": "$REV", "dirty_files": $DIRTY,
  "binary_sha256": "$BIN_SHA", "binary": "mynah-asr-server-cuda", "engine": "cuda", "gpu": "$GPU_NAME",
  "model": "$MODEL", "quant": "f32", "gemm": "$GEMM", "lookahead": $LOOKAHEAD, "cohort_ms": $COHORT,
  "workers": 1, "threads_per_worker": 1, "cap": $CAP, "http_threads_per_worker": $(( CAP + 32 )),
  "connection_ceiling": $(( CAP + 32 )), "server_cpus": "container", "gen_cpus": "container",
  "corpus_clips": $NCLIP, "bank_sha256": "$BANK_SHA", "corpus": "$CORPUS", "corpus_sample": $CORPUS_SAMPLE,
  "corpus_seed": $CORPUS_SEED, "min_peak_dbfs": $MIN_PEAK_DBFS, "reference_concurrency": $REF_C,
  "lang": "$LANG_Q", "configuration": "gpu-s14",
  "ladder": "$LADDER", "ladder_seconds": $LADDER_S,
  "soak_concurrency": $SOAK_C, "soak_seconds": $SOAK_S, "soaks": $SOAKS,
  "warmup_s": $WARMUP, "window_s": $WINDOW, "dump_every_s": $DUMP_EVERY,
  "registered_bounds": ".work/server-v2-qualification.md V2-2" }
JSON
say "done. Evidence in $RUN"
python3 tools/bench/v2_verdict.py "$RUN" --json "$RUN/verdict.json" | tee "$RUN/verdict.txt"
