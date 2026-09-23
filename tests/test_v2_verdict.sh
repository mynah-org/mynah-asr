#!/bin/sh
# test_v2_verdict.sh — the thing that says QUALIFIED has to be tested itself.
#
# v2_verdict.py turns artefacts into a product claim. If its parser is wrong,
# the claim is wrong in the direction nobody checks: a bound that silently
# reads as PASS. So each bound gets a synthetic run built to violate exactly
# it, and the assertion is that the verdict names THAT bound and no other.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fails=0
ok()   { echo "v2_verdict ok:   $1"; }
bad()  { echo "v2_verdict FAIL: $1"; fails=$(( fails + 1 )); }

# One healthy run, then copies of it each broken in one place.
mk() {  # mk <dir> <python mutation of the dict `d`>
    dir="$TMP/$1"; mkdir -p "$dir"
    python3 - "$dir" "$2" <<'PY'
import json, sys
dir, mutate = sys.argv[1], sys.argv[2]
win = [{"window": i, "t0_s": i*60.0, "t1_s": (i+1)*60.0, "n": 1800,
        "p50": 40.0, "p95": 90.0} for i in range(6)]
d = {
 # stream_load's REAL manifest field names. The first draft of this test
 # invented "streams" and "duration"; v2_verdict read those same invented names
 # and the test passed while both were None against a live run.
 "manifest": {"lookahead": 3, "concurrency": 16, "duration_s": 1800.0, "seed": 43,
              "chunk_period_ms": 320.0,
              "bank": {"short": [{"clip": "samples/en/a.wav", "audio_s": 7.4,
                                  "sha256": "aa"}],
                       "medium": [{"clip": "samples/fr/b.wav", "audio_s": 10.4,
                                   "sha256": "bb"}]}},
 "summary": {
   "counts": {"utterances": 600, "ok": 600, "errors": 0, "rejected": 0,
              "warmup_excluded": 20, "deltas": 9000, "eous": 600,
              "audio_s": 5220.0, "span_s": 1800.0},
   "pacing": {"paced": True, "verdict": "PACED"},
   "metrics": {
     "emission_lag_ms": {"p50": 40.0, "p95": 95.0, "max": 400.0, "n": 9000},
     "finalization_lag_ms": {"p95": 300.0, "max": 420.0, "n": 600},
     "backlog_max_s": {"max": 0.2, "p95": 0.18, "n": 600},
     "ttfp_ms": {"p95": 1600.0, "max": 2000.0, "n": 600},
     "audio_per_wall": {"max": 15.1, "p95": 15.1, "n": 1}},
   "drift": {"emission_lag_ms": {"windows": win, "trend_pct": 4.0,
                                 "max_drift_pct": 9.0, "pooled_p95": 95.0}},
   "text_groups": {"samples/en/a.wav": ["hello"], "samples/fr/b.wav": ["bonjour"]},
   "identity_fail": {}, "reference_fail": {},
   "quality": {"with_reference": 0, "without_reference": 600, "worst": None}},
 "utterances": []}
exec(mutate)
json.dump(d, open(dir + "/soak1-C16.json", "w"))
# three workers, steps always advancing, slots always busy
with open(dir + "/server-soak1.log", "w") as f:
    for w in range(3):
        f.write(f"mynah-asr-server 0.9.1: prefork worker {w} ready, group 'g' "
                f"(32 http threads, 96 stream slots, batch 8, streaming yes)\n")
    for seq in range(1, 7):
        for w in range(3):
            f.write(f"[DUMP] worker={w} seq={seq} slots active=6 cap=96 sessions=10 "
                    f"steps={seq*1000} deltas=50 eous=5 audio_s=100.0\n")
            f.write(f"[DUMP] worker={w} seq={seq} split model_busy_s={seq*20.0:.2f} (0.800) x\n")
with open(dir + "/procsample-soak1.txt", "w") as f:
    for seq in range(6):
        f.write("t=10:00:0%d pids=101,102,103,\n" % seq)
        for pid in (101, 102, 103):
            f.write(f"  {pid} {700000 + seq*1000} S {seq*30}\n")
PY
}
state() {  # state <dir> <bound number> -> PASS/FAIL/NO EVIDENCE
    python3 "$ROOT/tools/bench/v2_verdict.py" "$TMP/$1" 2>/dev/null \
        | sed -n "s/^  *$2  *[a-zA-Z0-9 ()-]*   *\([A-Z][A-Z ]*[A-Z]\)  *.*/\1/p" | head -1
}
verdict() { python3 "$ROOT/tools/bench/v2_verdict.py" "$TMP/$1" 2>/dev/null | sed -n 's/^  => //p' | head -1; }

mk healthy "pass"
[ "$(verdict healthy)" = "QUALIFIED" ] || bad "a healthy run should qualify (got '$(verdict healthy)')"
[ "$(verdict healthy)" = "QUALIFIED" ] && ok "a healthy synthetic run qualifies on all twelve bounds"

# A verdict that cannot say WHICH run it judged is not evidence. This caught a
# real defect: the tool read manifest fields named "streams" and "duration",
# which stream_load never writes, so every header printed C=None.
head_of() { python3 "$ROOT/tools/bench/v2_verdict.py" "$TMP/$1" 2>/dev/null | sed -n '2p'; }
case "$(head_of healthy)" in
    *"C=16"*"bank "*) ok "the verdict names the concurrency and the bank it judged" ;;
    *) bad "the header lost the run's identity: '$(head_of healthy)'" ;;
esac

# A run whose concurrency exceeds the fleet's connection ceiling measured the
# ceiling, not the machine: on 2026-09-22 the C=32 rung ran 24 streams, produced
# FEWER utterances than C=24 and lower throughput, and read as noise. It is
# INVALID, not merely worse, so it can never come out QUALIFIED.
mk ceiling "pass"
python3 -c "
import json,sys
p=sys.argv[1]+'/manifest.json'
json.dump({'connection_ceiling': 24}, open(p,'w'))" "$TMP/ceiling"
[ "$(verdict ceiling)" = "QUALIFIED" ] \
    && ok "C=16 inside a 24-connection ceiling stays qualified" \
    || bad "a run inside the ceiling was called invalid: $(verdict ceiling)"
mk ceiling2 "d['manifest']['concurrency'] = 32"
python3 -c "
import sys
p=sys.argv[1]+'/server-soak1.log'
s=open(p).read().replace('32 http threads','8 http threads')
open(p,'w').write(s)" "$TMP/ceiling2"
[ "$(verdict ceiling2)" = "INVALID" ] \
    && ok "C=32 against a 24-connection fleet is INVALID, not merely worse" \
    || bad "a rung that could not connect its own concurrency was scored: $(verdict ceiling2)"

mk lost "d['summary']['counts']['errors'] = 3"
[ "$(verdict lost)" = "NOT QUALIFIED" ] && ok "three lost streams disqualify" || bad "lost streams did not disqualify"

mk lag "d['summary']['metrics']['emission_lag_ms']['p95'] = 321.0"
[ "$(verdict lag)" = "NOT QUALIFIED" ] && ok "emission lag p95 one ms over the chunk period disqualifies" \
    || bad "lag p95 321 ms passed a 320 ms bound"

mk lag_ok "d['summary']['metrics']['emission_lag_ms']['p95'] = 319.0"
[ "$(verdict lag_ok)" = "QUALIFIED" ] && ok "and one ms under it does not" || bad "lag p95 319 ms failed a 320 ms bound"

# The bound is (lookahead+1)x80, so it MOVES with the preset instead of being
# three digits. At lookahead 6 the same 400 ms lag is inside the envelope.
mk la6 "d['manifest']['lookahead'] = 6; d['summary']['metrics']['emission_lag_ms']['p95'] = 400.0"
[ "$(verdict la6)" = "QUALIFIED" ] && ok "the lag bound follows the preset: 400 ms passes at lookahead 6" \
    || bad "the lag bound did not scale with lookahead"

mk win "d['summary']['drift']['emission_lag_ms']['windows'][3]['p95'] = 900.0"
[ "$(verdict win)" = "NOT QUALIFIED" ] && ok "one bad window disqualifies a run whose pooled p95 is fine" \
    || bad "a 900 ms window passed while the pooled p95 was 95 ms"

# A ramp window holds a fraction of a steady window's samples and is excluded
# rather than silently dropped -- otherwise every soak fails on its own warm-up.
mk ramp "d['summary']['drift']['emission_lag_ms']['windows'][0].update(n=40, p95=900.0)"
[ "$(verdict ramp)" = "QUALIFIED" ] && ok "a thin ramp window is excluded, not counted as a failure" \
    || bad "the warm-up window failed the run"

mk trend "d['summary']['drift']['emission_lag_ms']['trend_pct'] = 80.0"
[ "$(verdict trend)" = "NOT QUALIFIED" ] && ok "a p95 climbing 80% across the run disqualifies" \
    || bad "an 80% upward trend passed"

mk maxlag "d['summary']['metrics']['emission_lag_ms']['max'] = 9000.0"
[ "$(verdict maxlag)" = "NOT QUALIFIED" ] && ok "a nine-second stall disqualifies although every percentile is fine" \
    || bad "a 9 s max emission lag passed"

mk ident "d['summary']['identity_fail'] = {'samples/en/a.wav': ['hello', 'hallo']}"
[ "$(verdict ident)" = "NOT QUALIFIED" ] && ok "one clip transcribed two ways across streams disqualifies" \
    || bad "a transcript that depends on the stream passed"

mk reffail "d['summary']['reference_fail'] = {'samples/en/a.wav': {'expected': 'hello', 'got': ['hell']}}"
[ "$(verdict reffail)" = "NOT QUALIFIED" ] && ok "diverging from the unloaded reference disqualifies" \
    || bad "a load-dependent transcript passed"

# Missing evidence is not a pass. This is the whole reason the 2026-09-20 soak
# was only screened, so it gets its own assertion.
mk nodump "pass"; rm "$TMP/nodump/server-soak1.log"
[ "$(verdict nodump)" = "INCONCLUSIVE (missing evidence)" ] \
    && ok "no SIGUSR1 dump makes the run inconclusive, never qualified" \
    || bad "a run with no stall evidence claimed to be qualified (got '$(verdict nodump)')"

mk noproc "pass"; rm "$TMP/noproc/procsample-soak1.txt"
[ "$(verdict noproc)" = "INCONCLUSIVE (missing evidence)" ] \
    && ok "no rss/roster sample makes the run inconclusive" \
    || bad "a run with no memory evidence claimed to be qualified"

# A worker that dies and is replaced changes the roster.
mk died "pass"
python3 - "$TMP/died" <<'PY'
import sys
p = sys.argv[1] + "/procsample-soak1.txt"
lines = open(p).read().replace("pids=101,102,103,", "pids=101,102,104,", 1)
open(p, "w").write(lines)
PY
[ "$(verdict died)" = "NOT QUALIFIED" ] && ok "a worker replaced mid-run disqualifies" || bad "a changed roster passed"

# A stalled worker: slots active at both ends of the interval, steps frozen.
mk stalled "pass"
python3 - "$TMP/stalled" <<'PY'
import re, sys
p = sys.argv[1] + "/server-soak1.log"
out = []
for l in open(p):
    m = re.search(r"worker=0 seq=(\d+) slots", l)
    if m and int(m.group(1)) >= 4:
        l = re.sub(r"steps=\d+", "steps=3000", l)
    out.append(l)
open(p, "w").writelines(out)
PY
[ "$(verdict stalled)" = "NOT QUALIFIED" ] && ok "a worker whose step counter freezes while slots are active disqualifies" \
    || bad "a frozen worker passed"

# ... but a worker that goes idle because its streams ended is finished, not stalled.
mk drained "pass"
python3 - "$TMP/drained" <<'PY'
import re, sys
p = sys.argv[1] + "/server-soak1.log"
out = []
for l in open(p):
    m = re.search(r"worker=0 seq=(\d+) slots", l)
    if m and int(m.group(1)) >= 4:
        l = re.sub(r"steps=\d+", "steps=3000", re.sub(r"active=\d+", "active=0", l))
    out.append(l)
open(p, "w").writelines(out)
PY
[ "$(verdict drained)" = "QUALIFIED" ] && ok "a worker that went idle because its streams ended is not a stall" \
    || bad "an idle worker was called a stall"

mk rss "pass"
python3 - "$TMP/rss" <<'PY'
import sys
p = sys.argv[1] + "/procsample-soak1.txt"
out = []
seq = -1
for l in open(p):
    if l.startswith("t="):
        seq += 1; out.append(l); continue
    f = l.split()
    if len(f) >= 4:
        f[1] = str(700000 + seq * 60000)      # 1.0 -> 1.43x across six samples
        l = "  " + " ".join(f) + "\n"
    out.append(l)
open(p, "w").writelines(out)
PY
[ "$(verdict rss)" = "NOT QUALIFIED" ] && ok "rss growing 1.43x over the run disqualifies" || bad "a leaking worker passed"

# A 503 is the admission ladder working. It is counted, and it never fails a run.
mk rejected "d['summary']['counts']['rejected'] = 12"
[ "$(verdict rejected)" = "QUALIFIED" ] && ok "503 refusals are counted separately and do not fail a run" \
    || bad "admission refusals were treated as losses"

if [ "$fails" = 0 ]; then echo "test_v2_verdict: OK"; else echo "test_v2_verdict: $fails FAILED"; exit 1; fi
