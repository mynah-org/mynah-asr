#!/bin/sh
# test_v2_promote.sh — the tool that writes the word "qualified" into a profile
# is tested on its REFUSALS first.
#
# configs/perf/*.json has carried `status: screened` since 2026-09-20 because a
# person read a table and declined to promote it. That judgement is now a
# script, so the script has to be at least as hard to fool as the person was.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fails=0
ok()  { echo "v2_promote ok:   $1"; }
bad() { echo "v2_promote FAIL: $1"; fails=$(( fails + 1 )); }

mkrun() {   # mkrun <dir> <python mutation over the list `soaks`>
    dir="$TMP/$1"; mkdir -p "$dir"
    python3 - "$dir" "$2" <<'PY'
import json, sys
dir, mutate = sys.argv[1], sys.argv[2]
def soak(n, seed, C=16, dur=1800.0, lag95=95.0, bank_sha="aa"):
    win = [{"window": i, "t0_s": i*60.0, "t1_s": (i+1)*60.0, "n": 1800,
            "p50": 40.0, "p95": lag95} for i in range(6)]
    return {"n": n, "seed": seed, "d": {
      "manifest": {"lookahead": 3, "concurrency": C, "duration_s": dur, "seed": seed,
                   "bank": {"short": [{"clip": "a.wav", "audio_s": 7.4, "sha256": bank_sha}]}},
      "summary": {
        "counts": {"utterances": 3300, "ok": 3300, "errors": 0, "rejected": 0,
                   "warmup_excluded": 20, "deltas": 50000, "eous": 3300,
                   "audio_s": 28710.0, "span_s": 1800.0},
        "pacing": {"paced": True, "verdict": "PACED"},
        "metrics": {
          "emission_lag_ms": {"p50": 40.0, "p95": lag95, "p99": 150.0, "max": 400.0, "n": 50000},
          "finalization_lag_ms": {"p50": 80.0, "p95": 300.0, "p99": 380.0, "max": 420.0, "n": 3300},
          "backlog_max_s": {"p95": 0.18, "max": 0.2, "n": 3300},
          "ttfp_ms": {"p95": 1600.0, "max": 2000.0, "n": 3300},
          "ttfp_from_onset_ms": {"p95": 900.0, "max": 1200.0, "n": 3300},
          "max_delta_gap_ms": {"p95": 1400.0, "max": 2000.0, "n": 3300},
          "audio_per_wall": {"max": 15.1, "p95": 15.1, "n": 1}},
        "drift": {"emission_lag_ms": {"windows": win, "trend_pct": 4.0,
                                      "max_drift_pct": 9.0, "pooled_p95": lag95}},
        "text_groups": {"a.wav": ["hello"]},
        "identity_fail": {}, "reference_fail": {},
        "quality": {"with_reference": 0, "without_reference": 3300, "worst": None}},
      "utterances": [{"lag_marks": [[0.0, 40.0], [1.0, lag95]]}]}}
soaks = [soak(1, 43), soak(2, 44)]
exec(mutate)
json.dump({"utc": "now", "commit": "abc1234"}, open(dir + "/manifest.json", "w"))
for s in soaks:
    n = s["n"]
    json.dump(s["d"], open(f"{dir}/soak{n}-C16.json", "w"))
    with open(f"{dir}/server-soak{n}.log", "w") as f:
        for seq in range(1, 7):
            for w in range(3):
                f.write(f"[DUMP] worker={w} seq={seq} slots active=6 cap=96 sessions=10 "
                        f"steps={seq*1000} deltas=50 eous=5 audio_s=100.0\n")
                f.write(f"[DUMP] worker={w} seq={seq} split model_busy_s={seq*20.0:.2f} (0.8) x\n")
    with open(f"{dir}/procsample-soak{n}.txt", "w") as f:
        for seq in range(6):
            f.write("t=10:00:0%d pids=101,102,103,\n" % seq)
            for pid in (101, 102, 103):
                f.write(f"  {pid} {700000 + seq*500} S {seq*30}\n")
PY
}
mkprofile() {
    mkdir -p "$TMP/perf"
    cat > "$TMP/perf/testbox.json" <<'JSON'
{ "schema_version": 1,
  "profile": {"id": "testbox", "description": "d", "status": "screened", "objective": {}},
  "measured": {"date": "2026-09-20",
               "soak": {"concurrency": 32, "duration_s": 600,
                        "why_not_qualifying": "transcripts were never checked"}} }
JSON
}
promote() { PYTHONPATH="$ROOT/tools/bench" python3 - "$@" <<'PY'
import sys, os, runpy
sys.argv = ["v2_promote.py"] + sys.argv[1:]
root = os.environ["TEST_ROOT"]
import v2_promote
v2_promote.PERF = os.environ["TEST_PERF"]
sys.exit(v2_promote.main())
PY
}
export TEST_ROOT="$ROOT" TEST_PERF="$TMP/perf"
run() { promote "$TMP/$1" --profile testbox ${2:-} 2>&1; }

mkprofile
mkrun good "pass"
run good | grep -q "would write profile.status = qualified" \
    && ok "two independent 1800 s soaks, every bound PASS, promote" \
    || bad "a clean pair of soaks was not promotable: $(run good | tail -2)"

mkrun one "soaks = soaks[:1]"
run one | grep -q "one long run is a run and not a qualification" \
    && ok "a single soak is refused, however good it is" || bad "one soak was promoted"

mkrun failing "soaks[1]['d']['summary']['counts']['errors'] = 2"
run failing | grep -q "REFUSED" && ok "a lost stream in the SECOND soak refuses the pair" \
    || bad "a losing soak was promoted"

mkrun ladder "soaks[1]['d']['manifest']['concurrency'] = 24"
run ladder | grep -q "that is a ladder" \
    && ok "two soaks at different concurrencies are a ladder, not a repeated experiment" \
    || bad "a ladder was promoted as a qualification"

mkrun sameseed "soaks[1]['seed'] = 43; soaks[1]['d']['manifest']['seed'] = 43"
run sameseed | grep -q "two replays of one schedule" \
    && ok "the same seed twice is refused: it does not show schedule independence" \
    || bad "two replays of one schedule were promoted"

mkrun otherbank "soaks[1]['d']['manifest']['bank']['short'][0]['sha256'] = 'zz'"
run otherbank | grep -q "not the same experiment twice" \
    && ok "two different banks are refused" || bad "two different banks were promoted"

mkrun short "soaks[1]['d']['manifest']['duration_s'] = 300.0"
run short | grep -q "under the 600s" && ok "a soak under ten minutes is refused" \
    || bad "a five-minute soak was promoted"

mkrun noev "pass"; rm "$TMP/noev/procsample-soak2.txt"
run noev | grep -q "REFUSED" && ok "a missing evidence file refuses promotion, it is not a pass" \
    || bad "a soak with no memory evidence was promoted"

# The claim has to survive its own bad run: the operating point takes the WORSE
# of the two, never the better one and never the mean.
mkrun worse "soaks[1]['d']['summary']['metrics']['emission_lag_ms']['p95'] = 210.0"
if run worse | python3 -c "
import json,sys
t=sys.stdin.read()
i=t.index('{'); j=t.rindex('}')+1
d=json.loads(t[i:j])
sys.exit(0 if d['emission_lag_p95_ms'] == 210.0 else 1)"; then
    ok "the operating point carries the WORSE run's p95, not the better one"
else
    bad "the operating point did not take the worse of the two runs"
fi

# --apply must actually flip the status and keep the old soak's reason.
mkrun apply2 "pass"
run apply2 --apply > /dev/null 2>&1
python3 - "$TMP/perf/testbox.json" <<'PY' && ok "--apply flips status and keeps the previous soak's reason in history" \
    || bad "--apply did not record the promotion correctly"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["profile"]["status"] == "qualified", d["profile"]["status"]
op = d["measured"]["long_soak_qualified"]
assert op["concurrency"] == 16 and op["runs"] == 2, op
assert op["established_streams_lost"] == 0
h = d["measured"]["history"]
assert len(h) == 1 and h[0]["concurrency"] == 32, h
assert "transcripts were never checked" in h[0]["why_not_promoted"], h[0]
PY

if [ "$fails" = 0 ]; then echo "test_v2_promote: OK"; else echo "test_v2_promote: $fails FAILED"; exit 1; fi
