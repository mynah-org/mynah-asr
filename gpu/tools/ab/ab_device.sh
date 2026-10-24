#!/bin/bash
# gpu/tools/ab/ab_device.sh -- an interleaved A/B (ABBA) of the device step alone.
#
# usage: [VAR=value ...] gpu/tools/ab/ab_device.sh BASE_TEST PATCH_TEST
#
# BASE_TEST / PATCH_TEST are tests/test_cuda_stream binaries (one per worktree,
# `make -C gpu test-stream`). Per level C, the arms run in ORDER (CYCLES times),
# each a fresh process:
#     <test> PACK [CLIPS] --precision P --gemm G --bench C --steps STEPS <X_ARGS>
# which steps C lanes together with the stage profile on and prints
#     BENCH C=.. full_steps=.. passes=.. device_ms_per_pass=.. step_wall_ms=..
#     BENCH   <stage> .. ms/pass          (14 stages)
# Then gpu/tools/ab/ab_report.py OUT/TAG (kind "device") gives the verdict on
# device_ms_per_pass (primary), step_wall_ms and every stage. There is no text
# in --bench: run the test's own gates (without --bench) for bit-exactness.
#
#   PACK        (default models/nemotron-3.5-asr-streaming-0.6b)
#   LEVELS      C values                      (default "64 128")
#   STEPS       --steps                       (default 40; the first 2 are warm-up)
#   PRECISION   --precision                   (default f32; bf16 goes with --gemm own-tc)
#   GEMM        --gemm                        (default own)
#   A_PRECISION, A_GEMM, B_PRECISION, B_GEMM ...  per-arm overrides of the two above, e.g.
#               A=f32 own against B_PRECISION=bf16 B_GEMM=own-tc on ONE binary: the
#               serving default's real stage shares (the report prints each stage's
#               share of the pass beside its ms)
#   CLIPS       clip paths, repo-relative     (default: none = the test's 5 built-in clips)
#   ORDER, CYCLES, A_ARGS, B_ARGS, A_BIN, B_BIN, A_LABEL, B_LABEL   as in ab_server.sh
#   RUN_TIMEOUT s per run (default 300)    COOL_S (default 3)
#   LOCK, LOCK_WAIT, BUDGET_MIN, OUT, TAG, DRY_RUN   as in ab_server.sh
set -u
[ $# -eq 2 ] || { sed -n '2,33p' "$0"; exit 2; }
CALLER=$PWD
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$CALLER/$1" ;; esac; }
BASE_BIN=$(abspath "$1"); PATCH_BIN=$(abspath "$2")
OUT=$(abspath "${OUT:-$PWD/ab-res}")
cd "$(dirname "$0")/../../.." || exit 2
HERE=gpu/tools/ab
PY=${PY:-python3}
PACK=${PACK:-models/nemotron-3.5-asr-streaming-0.6b}
LEVELS=${LEVELS:-"64 128"}; ORDER=${ORDER:-ABBA}; CYCLES=${CYCLES:-1}
STEPS=${STEPS:-40}; PRECISION=${PRECISION:-f32}; GEMM=${GEMM:-own}; CLIPS=${CLIPS:-}
RUN_TIMEOUT=${RUN_TIMEOUT:-300}; COOL_S=${COOL_S:-3}; BUDGET_MIN=${BUDGET_MIN:-0}
LOCK=${LOCK:-$HOME/gpu.lock}; LOCK_WAIT=${LOCK_WAIT:-0}
TAG=${TAG:-abdev-$(date -u +%Y%m%d-%H%M%S)}; R=$OUT/$TAG

arm_var() { local v="${1}_$2"; echo "${!v-}"; }
arm_prec() { local v; v=$(arm_var "$1" PRECISION); echo "${v:-$PRECISION}"; }
arm_gemm() { local v; v=$(arm_var "$1" GEMM); echo "${v:-$GEMM}"; }
arm_bin() { local b; b=$(arm_var "$1" BIN); [ -n "$b" ] || { [ "$1" = A ] && b=$BASE_BIN || b=$PATCH_BIN; }; abspath "$b"; }
ARMS=$(echo "$ORDER" | fold -w1 | sort -u | tr -d '\n')
ARM_SPECS=()
for X in $(echo "$ARMS" | fold -w1); do
    ARM_SPECS+=(--arm "$X:$(arm_var "$X" LABEL):$(arm_bin "$X"):--precision $(arm_prec "$X") --gemm $(arm_gemm "$X") --steps $STEPS $(arm_var "$X" ARGS)")
done
REV=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
PLAN=$("$PY" $HERE/ab_lib.py plan --kind device --out "$R" --tag "$TAG" --order "$ORDER" \
    --cycles "$CYCLES" --levels "$LEVELS" "${ARM_SPECS[@]}" --ready-est 25 --stop-est "$COOL_S" \
    --extra harness_rev="$REV" pack="$PACK" steps="$STEPS" precision="$PRECISION" gemm="$GEMM") || exit 2
run_cmd() { echo "$(arm_bin "$1") $PACK $CLIPS --precision $(arm_prec "$1") --gemm $(arm_gemm "$1") --bench $2 --steps $STEPS $(arm_var "$1" ARGS)"; }
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "== ab_device DRY RUN tag=$TAG levels=[$LEVELS] order=$ORDER cycles=$CYCLES"
    for X in $(echo "$ARMS" | fold -w1); do echo "   arm $X: $(run_cmd "$X" C)"; done
    echo "$PLAN" | awk '{printf "   %s", $1} END {print ""}'
    rm -rf "$R"; exit 0
fi
[ -f "$PACK/mynah.json" ] || { echo "ab_device: no pack at $PACK" >&2; exit 2; }
for X in $(echo "$ARMS" | fold -w1); do
    [ -x "$(arm_bin "$X")" ] || { echo "ab_device: arm $X binary $(arm_bin "$X") is not executable" >&2; exit 2; }
done
if [ "$LOCK" != none ]; then
    command -v flock >/dev/null || { echo "ab_device: flock missing (LOCK=none to run unlocked)" >&2; exit 2; }
    exec 9>"$LOCK"
    flock -w "$LOCK_WAIT" 9 || { echo "ab_device: $LOCK is held (another GPU job); LOCK_WAIT=s to queue" >&2; exit 1; }
fi
echo "== ab_device $TAG harness=$REV levels=[$LEVELS] order=$ORDER cycles=$CYCLES steps=$STEPS -> $R" | tee "$R/run.txt"
T0=$(date +%s)
while read -r RUN C X; do
    if [ "$BUDGET_MIN" -gt 0 ] && [ $(( $(date +%s) - T0 )) -ge $(( BUDGET_MIN * 60 )) ]; then
        echo "  budget of $BUDGET_MIN min spent: $RUN and later not started" | tee -a "$R/run.txt"; break
    fi
    t0=$(date +%s)
    # shellcheck disable=SC2046
    timeout -k 10 "$RUN_TIMEOUT" $(run_cmd "$X" "$C") > "$R/bench-$RUN.txt" 2>&1
    rc=$?; to=false; { [ $rc -eq 124 ] || [ $rc -eq 137 ]; } && to=true
    "$PY" $HERE/ab_lib.py meta "$R/meta-$RUN.json" run="$RUN" level="$C" arm="$X" rc=$rc \
        timed_out=$to wall_s=$(( $(date +%s) - t0 ))
    echo "  $RUN rc=$rc $(grep -m1 '^BENCH C=' "$R/bench-$RUN.txt" || echo 'no BENCH line')"
    sleep "$COOL_S"
done <<< "$PLAN"
echo "== done in $(( ($(date +%s) - T0) / 60 )) min" | tee -a "$R/run.txt"
"$PY" $HERE/ab_report.py "$R" --json "$R/verdict.json" | tee "$R/report.txt"
exit "${PIPESTATUS[0]}"
