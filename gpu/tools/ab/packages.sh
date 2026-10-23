#!/bin/bash
# gpu/tools/ab/packages.sh -- the registered A/B packages, ready to run on a GPU box.
#
# usage: [DRY_RUN=1] [PACK=/abs/pack] gpu/tools/ab/packages.sh <package> ARGS...
#
#   cohort    SERVER_BIN
#       --cohort-ms {40, 0, 20} on ONE binary. Arms A=40 (the shipped default,
#       the baseline), B=0, C=20; ORDER ABCCBA (= 40, 0, 20, 20, 0, 40) per level;
#       LEVELS 16 64 128 192; REPEAT 2 over the 23 FLEURS clips; cap 208.
#       24 runs x ~2.6 min (load 107-112 s + start/stop) = ~62 min.
#       The report's WINNER line names the cohort for package `release`.
#   release   SERVER_BIN_0711ddb [WINNER_MS]
#       --release timeout (A) vs work-conserving (B) on the 0711ddb binary, ABBA,
#       LEVELS 16 64 128 192, at --cohort-ms 40 and, when WINNER_MS is given and
#       is not 40, again at the sweep winner: 16 runs (~41 min) per cohort.
#   labelloop FEATURE_SERVER_BIN
#       --label-loop OFF (A, the binary's default) vs `--label-loop blocked` (B)
#       on the feature binary, ABBA, LEVELS 16 64 128 192, strict identity (a
#       label loop must not change a byte): 16 runs, ~41 min.
#   device-prec TEST_BIN
#       tests/test_cuda_stream --bench on ONE binary: A = f32 own, B = bf16
#       own-tc (the serving default), ABBA, C = 32 128, 40 steps: the stage
#       shares of the real serving path. 8 runs, ~4 min. (Needs the bench to take
#       --precision: tests/test_cuda_stream.c from this branch on.)
#   device-ab BASE_TEST_BIN PATCH_TEST_BIN
#       the same bench, f32 `--gemm own --steps 40`, base vs patch (e.g. PR #6
#       attention tiles), ABBA, C = 64 128: 8 runs, ~4 min.
#
# Every package holds ~/gpu.lock for its whole run (LOCK, LOCK_WAIT as in
# ab_server.sh), writes OUT/TAG (default ./ab-res/<package>-<utc>) and ends with
# gpu/tools/ab/ab_report.py's verdict. Any ab_server.sh / ab_device.sh variable
# set in the environment (LEVELS, CYCLES, BUDGET_MIN, PIN ...) overrides the
# package's value.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=${1:-}; shift || true
STAMP=$(date -u +%Y%m%d-%H%M%S)
srv() { echo "+ $*" >&2; env "$@"; }
SRV_LEVELS=${LEVELS:-"16 64 128 192"}
case $PKG in
cohort)
    [ $# -eq 1 ] || { sed -n '2,31p' "$0"; exit 2; }
    srv TAG="${TAG:-cohort-$STAMP}" LEVELS="$SRV_LEVELS" ORDER="${ORDER:-ABCCBA}" \
        A_LABEL=cohort40 A_ARGS="--cohort-ms 40" B_LABEL=cohort0 B_ARGS="--cohort-ms 0" \
        C_LABEL=cohort20 C_ARGS="--cohort-ms 20" "$HERE/ab_server.sh" "$1" "$1" ;;
release)
    [ $# -ge 1 ] || { sed -n '2,31p' "$0"; exit 2; }
    rc=0
    for MS in 40 ${2:-}; do
        [ "$MS" = 40 ] && [ -n "${done40:-}" ] && continue
        [ "$MS" = 40 ] && done40=1
        srv TAG="${TAG:-release}-c$MS-$STAMP" LEVELS="$SRV_LEVELS" ORDER="${ORDER:-ABBA}" \
            A_LABEL="timeout-c$MS" A_ARGS="--cohort-ms $MS --release timeout" \
            B_LABEL="wc-c$MS" B_ARGS="--cohort-ms $MS --release work-conserving" \
            "$HERE/ab_server.sh" "$1" "$1" || rc=$?
    done
    exit $rc ;;
labelloop)
    [ $# -eq 1 ] || { sed -n '2,31p' "$0"; exit 2; }
    srv TAG="${TAG:-labelloop-$STAMP}" LEVELS="$SRV_LEVELS" ORDER="${ORDER:-ABBA}" \
        A_LABEL=labelloop-off A_ARGS="" B_LABEL=labelloop-blocked B_ARGS="--label-loop blocked" \
        "$HERE/ab_server.sh" "$1" "$1" ;;
device-prec)
    [ $# -eq 1 ] || { sed -n '2,31p' "$0"; exit 2; }
    srv TAG="${TAG:-devprec-$STAMP}" LEVELS="${LEVELS:-32 128}" ORDER="${ORDER:-ABBA}" \
        PRECISION=f32 GEMM=own A_LABEL=f32-own B_LABEL=bf16-owntc B_PRECISION=bf16 B_GEMM=own-tc \
        "$HERE/ab_device.sh" "$1" "$1" ;;
device-ab)
    [ $# -eq 2 ] || { sed -n '2,31p' "$0"; exit 2; }
    srv TAG="${TAG:-devab-$STAMP}" LEVELS="${LEVELS:-64 128}" ORDER="${ORDER:-ABBA}" \
        PRECISION=f32 GEMM=own A_LABEL=base B_LABEL=patch "$HERE/ab_device.sh" "$1" "$2" ;;
*)
    sed -n '2,31p' "$0"; exit 2 ;;
esac
