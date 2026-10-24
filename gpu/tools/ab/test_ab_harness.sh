#!/bin/bash
# gpu/tools/ab/test_ab_harness.sh -- the A/B harness end to end on a CPU, with no GPU.
#
#   gpu/tools/ab/test_ab_harness.sh        (about 2 minutes; needs python3 and curl)
#
# 1. the report's unit tests on synthetic JSON (gpu/tools/ab/test_ab_report.py);
# 2. ab_server.sh against gpu/tools/ab/fake_server.py on short synthetic WAVs:
#    an arm that holds every delta back 400 ms must come out FAIL (WORSE on
#    emission lag), an arm that changes one transcript must fail gate 1 (exit 2),
#    a server that dies mid-run must fail gate 2 (exit 2);
# 3. ab_device.sh against a fake test_cuda_stream that prints BENCH lines.
# Exit 0 = every case behaved.
set -u
cd "$(dirname "$0")/../../.." || exit 2
HERE=gpu/tools/ab
PY=${PY:-python3}
command -v curl >/dev/null || { echo "SKIP test_ab_harness: curl missing"; exit 77; }
command -v timeout >/dev/null || { echo "SKIP test_ab_harness: timeout missing"; exit 77; }
T=$(mktemp -d "${TMPDIR:-/tmp}/abtest.XXXXXX")
trap 'rm -rf "$T"' EXIT
fail=0
check() { if [ "$1" = "$2" ]; then echo "OK   $3"; else echo "FAIL $3 (got $1, want $2)"; fail=1; fi; }

"$PY" $HERE/test_ab_report.py -q 2>&1 | tail -3
check "${PIPESTATUS[0]}" 0 "unit tests of ab_report.py"

"$PY" - "$T" <<'PY'
import math, random, struct, sys, wave
for i, secs in enumerate((1.2, 1.7, 2.1)):
    rnd = random.Random(i)
    with wave.open(f"{sys.argv[1]}/clip{i}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(t * 0.05 * (i + 1))
                                                     + rnd.randint(-300, 300)))
                               for t in range(int(16000 * secs))))
PY
mkdir -p "$T/pack"; echo '{}' > "$T/pack/mynah.json"
FAKE=$PWD/$HERE/fake_server.py
common() {
    env LOCK=none PACK="$T/pack" CLIPS="$T/clip0.wav $T/clip1.wav $T/clip2.wav" ONSETS=none \
        COOL_S=0 STOP_TIMEOUT=5 READY_TIMEOUT=20 DONE_TIMEOUT=10 PORT=${PORT:-18391} OUT="$T/out" "$@"
}

common TAG=slow LEVELS="2 3" B_ARGS="--fake-delay-ms 400" \
    $HERE/ab_server.sh "$FAKE" "$FAKE" > "$T/slow.txt" 2>&1
rc=$?
check $rc 0 "fake A/B: gates hold"
v=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdicts"]["B"])' "$T/out/slow/verdict.json" 2>/dev/null)
check "$v" FAIL "fake A/B: a 400 ms slower arm is FAIL"
grep -q "lag_p95\*.*WORSE" "$T/out/slow/report.txt"; check $? 0 "fake A/B: emission lag p95 WORSE"

common TAG=salt LEVELS="2" B_ARGS="--fake-salt x" \
    $HERE/ab_server.sh "$FAKE" "$FAKE" > "$T/salt.txt" 2>&1
check $? 2 "fake A/B: a changed transcript fails gate 1"
grep -q "gate 3 latency: NOT EVALUATED" "$T/out/salt/report.txt"; check $? 0 "fake A/B: no latency after gate 1"

common TAG=die LEVELS="2" B_ARGS="--fake-die-after 2" \
    $HERE/ab_server.sh "$FAKE" "$FAKE" > "$T/die.txt" 2>&1
check $? 2 "fake A/B: a dying server fails gate 2"

cat > "$T/fake_bench" <<'SH'
#!/bin/sh
# prints the BENCH lines of tests/test_cuda_stream --bench; FAKE_MS sets the pass time
C=0; for a in "$@"; do [ "$prev" = --bench ] && C=$a; prev=$a; done
MS=${FAKE_MS:-10.0}
echo "BENCH ready C=$C precision=f32 gemm=own vram_weights_mib=1.0 vram_arena_mib=1.0 vram_used_mib=2.0"
echo "BENCH C=$C full_steps=38 passes=38 device_ms_per_pass=$MS step_wall_ms=$MS"
for s in h2d subsample ffn1 att_proj att_core att_out conv ffn2 post dec_joint dec_pred dec_sync d2h other; do
  echo "BENCH   $s    0.500 ms/pass"
done
SH
chmod +x "$T/fake_bench"
printf '#!/bin/sh\nFAKE_MS=8.0 exec %s "$@"\n' "$T/fake_bench" > "$T/fake_bench_fast"; chmod +x "$T/fake_bench_fast"
env LOCK=none PACK="$T/pack" OUT="$T/out" TAG=dev LEVELS="64" COOL_S=0 \
    $HERE/ab_device.sh "$T/fake_bench" "$T/fake_bench_fast" > "$T/dev.txt" 2>&1
check $? 0 "fake device A/B: gates hold"
v=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdicts"]["B"])' "$T/out/dev/verdict.json" 2>/dev/null)
check "$v" PASS "fake device A/B: a 20% faster pass is PASS"

if [ $fail -ne 0 ]; then
    for f in "$T"/*.txt; do echo "---- $f"; tail -30 "$f"; done
fi
[ $fail -eq 0 ] && echo "test_ab_harness: PASS" || echo "test_ab_harness: FAIL"
exit $fail
