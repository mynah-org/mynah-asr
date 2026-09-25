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
PEN = 0.0; RPEN = 0.0; WITH_BASELINE = False
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
              "audio_s": 5220.0, "span_s": 1800.0,
              # the whole-run accounting stream_load writes since 2026-09-25
              "errors_warmup": 0, "errors_total": 0, "error_kinds": {},
              "rejected_warmup": 0, "rejected_total": 0, "cut_at_deadline": 0,
              "started": 620, "stream_deaths": 0},
   "accounting": {"started": 620, "ok": 620, "rejected": 0, "cut_at_deadline": 0,
                  "errors": 0, "error_kinds": {}, "errors_warmup": 0,
                  "conservation": True, "conservation_detail": None},
   "streams": {"expected": 16, "dead": [], "exited_early": [],
               "terminated_at_deadline": []},
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
 "utterances": [{"clip": "a.wav", "ttfp_ms": 1000.0 + PEN,
                 "first_delta_lag_ms": 40.0 + RPEN,
                 "lag_marks": [[0.0, 40.0], [1.0, 60.0]]} for _ in range(120)]}
exec(mutate)
d["utterances"] = [{"clip": "a.wav", "ttfp_ms": 1000.0 + PEN,
                    "first_delta_lag_ms": 40.0 + RPEN,
                    "lag_marks": [[0.0, 40.0], [1.0, 60.0]]} for _ in range(120)]
json.dump(d, open(dir + "/soak1-C16.json", "w"))
if WITH_BASELINE:
    json.dump({"a.wav": {"ttfp_ms": 1000.0, "first_delta_lag_ms": 40.0}},
              open(dir + "/reference-ttfp.json", "w"))
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
            f.write(f"[DUMP] worker={w} seq={seq} books sessions={10 + seq} completed={seq} "
                    f"cancelled=2 aborted=0 active=8 balanced=1 abandoned=0 recovered=0\n")
            f.write(f"[DUMP] worker={w} seq={seq} cancelled=2 idle_timeout=1 peer_gone=1 "
                    f"frame_too_large=0 protocol_error=0 other=0\n")
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

# TTFP, registered 2026-09-23 for capacity rungs. A run that carries no unloaded
# baseline must SAY so and still be judged on the twelve bounds it registered --
# adding a thirteenth after the fact is what the frozen-qualification rule bans.
[ "$(verdict healthy)" = "QUALIFIED" ] \
    && ok "a run with no TTFP baseline keeps its verdict on the twelve registered bounds" \
    || bad "the new bound was applied retroactively"
python3 "$ROOT/tools/bench/v2_verdict.py" "$TMP/healthy" 2>/dev/null | grep -q "NOT REGISTERED" \
    && ok "and says NOT REGISTERED instead of passing quietly" || bad "the missing bound was silent"

mk ttfp_ok "WITH_BASELINE = True; PEN = 200.0"
[ "$(verdict ttfp_ok)" = "QUALIFIED" ] && ok "a paired TTFP penalty of 200 ms passes" \
    || bad "200 ms of paired TTFP penalty failed a 250 ms gate"
mk ttfp_degr "WITH_BASELINE = True; PEN = 300.0"
[ "$(verdict ttfp_degr)" = "NOT QUALIFIED" ] && ok "300 ms is DEGRADED and does not qualify" \
    || bad "a degraded TTFP penalty was qualified"
python3 "$ROOT/tools/bench/v2_verdict.py" "$TMP/ttfp_degr" 2>/dev/null | grep -q "DEGRADED" \
    && ok "and is named DEGRADED, not lumped in with a hard failure" || bad "DEGRADED was not reported"
mk ttfp_bad "WITH_BASELINE = True; PEN = 600.0"
[ "$(verdict ttfp_bad)" = "NOT QUALIFIED" ] && ok "600 ms of paired TTFP penalty fails" \
    || bad "600 ms of paired penalty passed"

# The paired construction is the point: a clip that leads with silence carries it
# on both sides, so a huge ABSOLUTE ttfp with a zero penalty must still pass.
mk ttfp_silence "WITH_BASELINE = True; PEN = 0.0
d2 = json.load(open(dir + '/soak1-C16.json')) if False else None"
python3 -c "
import json,sys
p=sys.argv[1]+'/reference-ttfp.json'
json.dump({'a.wav': {'ttfp_ms': 4400.0, 'first_delta_lag_ms': 40.0}}, open(p,'w'))
q=sys.argv[1]+'/soak1-C16.json'
d=json.load(open(q))
for u in d['utterances']: u['ttfp_ms']=4400.0
json.dump(d,open(q,'w'))" "$TMP/ttfp_silence"
[ "$(verdict ttfp_silence)" = "QUALIFIED" ] \
    && ok "4.4 s of leading silence cancels in the pairing and does not fail the server" \
    || bad "the paired gate charged the server for the corpus's silence"

mk ready_bad "WITH_BASELINE = True; RPEN = 400.0"
[ "$(verdict ready_bad)" = "NOT QUALIFIED" ] \
    && ok "losing more than one cadence before the first partial fails" \
    || bad "400 ms of extra first-frame lateness passed a 320 ms cadence bound"
mk ready_ok "WITH_BASELINE = True; RPEN = 300.0"
[ "$(verdict ready_ok)" = "QUALIFIED" ] && ok "and 300 ms, inside the cadence, does not" \
    || bad "300 ms failed a 320 ms bound"

mk lost "d['summary']['counts'].update(errors=3, errors_total=3, error_kinds={'timeout': 3})"
[ "$(verdict lost)" = "NOT QUALIFIED" ] && ok "three lost streams disqualify" || bad "lost streams did not disqualify"

# --- harness accounting (S12-17, AUDIT 2026-09-24). Each case is a run the OLD
# verdict read as QUALIFIED: it read `counts.errors`, which excludes the warm-up,
# and nothing else about the accounting.
mk warmerr "d['summary']['counts'].update(errors_warmup=2, errors_total=2, error_kinds={'server_disconnect': 2})"
[ "$(verdict warmerr)" = "NOT QUALIFIED" ] && [ "$(state warmerr 1)" = "FAIL" ] \
    && ok "two errors inside the warm-up fail bound 1 (the post-warm-up count is still 0)" \
    || bad "warm-up errors were dropped again: $(verdict warmerr) / bound 1 $(state warmerr 1)"

mk conserv "d['summary']['accounting'].update(conservation=False, conservation_detail='started 620 != 619')"
[ "$(verdict conserv)" = "NOT QUALIFIED" ] && [ "$(state conserv A)" = "FAIL" ] \
    && ok "an utterance missing from the accounting fails the run" \
    || bad "a broken conservation invariant passed: $(verdict conserv)"

mk deaths "d['summary']['counts']['stream_deaths'] = 1; d['summary']['streams']['dead'] = [7]"
[ "$(verdict deaths)" = "NOT QUALIFIED" ] && [ "$(state deaths A)" = "FAIL" ] \
    && ok "a client stream that died before the deadline fails the run" \
    || bad "a dead client stream passed: $(verdict deaths)"

mk nomarks "d['summary']['accounting'].update(conservation=None, conservation_detail='no start marks')"
[ "$(verdict nomarks)" = "INCONCLUSIVE (missing evidence)" ] \
    && ok "a new-format run whose conservation was not checked is inconclusive, not qualified" \
    || bad "an unchecked conservation passed: $(verdict nomarks)"

mk cut "d['summary']['counts']['cut_at_deadline'] = 3; d['summary']['accounting'].update(cut_at_deadline=3, started=623)"
[ "$(verdict cut)" = "QUALIFIED" ] \
    && ok "utterances cut at the soak deadline are counted and are not losses" \
    || bad "cut_at_deadline was treated as a loss: $(verdict cut)"

# An old run keeps its verdict and says what it cannot know.
mk legacy "[d['summary']['counts'].pop(k) for k in ('errors_warmup', 'errors_total', 'error_kinds', 'rejected_warmup', 'rejected_total', 'cut_at_deadline', 'started', 'stream_deaths')]; d['summary'].pop('accounting'); d['summary'].pop('streams')"
[ "$(verdict legacy)" = "QUALIFIED" ] && ok "a run recorded before the accounting fields keeps its verdict" \
    || bad "an old run changed verdict or crashed: '$(verdict legacy)'"
python3 "$ROOT/tools/bench/v2_verdict.py" "$TMP/legacy" 2>/dev/null | grep -q "warning: run predates errors_total" \
    && [ "$(state legacy A)" = "NOT REGISTERED" ] \
    && ok "and warns that warm-up errors and stream deaths are unknown" \
    || bad "an old run was judged without saying what it could not know"
mk legacylost "[d['summary']['counts'].pop(k) for k in ('errors_total', 'errors_warmup')]; d['summary']['counts']['errors'] = 3"
[ "$(verdict legacylost)" = "NOT QUALIFIED" ] && ok "an old run with post-warm-up losses still fails" \
    || bad "the fallback to counts.errors is broken"

# --- the server's session books (S12-18), from the worker dumps
[ "$(state healthy B)" = "PASS" ] && ok "balanced server books pass row B" \
    || bad "balanced books did not pass row B: $(state healthy B)"
mk unbal "pass"
sed -i.bak 's/worker=1 seq=4 books sessions=14 completed=4 cancelled=2 aborted=0 active=8 balanced=1/worker=1 seq=4 books sessions=14 completed=4 cancelled=1 aborted=0 active=8 balanced=0/' \
    "$TMP/unbal/server-soak1.log"
[ "$(verdict unbal)" = "NOT QUALIFIED" ] && [ "$(state unbal B)" = "FAIL" ] \
    && ok "one unbalanced books line in one worker's dumps fails the run" \
    || bad "a server that lost a session qualified: $(verdict unbal) / B $(state unbal B)"
mk abandon "pass"
sed -i.bak 's/worker=2 seq=6 books \(.*\) abandoned=0 recovered=0/worker=2 seq=6 books \1 abandoned=1 recovered=0/' \
    "$TMP/abandon/server-soak1.log"
[ "$(verdict abandon)" = "NOT QUALIFIED" ] && [ "$(state abandon B)" = "FAIL" ] \
    && ok "an abandoned slot never recovered by the end fails the run" \
    || bad "a leaked slot qualified: $(verdict abandon) / B $(state abandon B)"
mk oldserver "pass"
sed -i.bak '/ books sessions=/d' "$TMP/oldserver/server-soak1.log"
[ "$(verdict oldserver)" = "QUALIFIED" ] && [ "$(state oldserver B)" = "NOT REGISTERED" ] \
    && ok "a server that predates the books keeps its verdict, row B not registered" \
    || bad "an old server log changed verdict: $(verdict oldserver) / B $(state oldserver B)"

# --- fault injection (P0-d): 70 of 690 utterances aborted on purpose, 10 per point.
# They are not losses, and the healthy streams are judged alone; row F accounts them.
FAULTS="d['summary']['counts']['aborted_by_client'] = 70
d['summary']['accounting'].update(aborted_by_client=70, started=690)
pts = ['before_first_partial', 'before_first_partial_silence', 'mid_utterance',
       'mid_utterance_silence', 'partial_frame', 'during_finalization', 'idle_open']
d['summary']['faults'] = {'planned': {p: 10 for p in pts}, 'aborted': {p: 10 for p in pts},
    'preempted': {p: {} for p in pts}, 'methods': {p: {} for p in pts},
    'server_end': {p: ({'idle_timeout': 10} if p == 'idle_open' else {}) for p in pts},
    'planned_total': 70, 'aborted_total': 70, 'early_delta': 0, 'violations': [],
    'client': {'sessions': 690, 'sessions_unknown': 0, 'audio_sent_s': 5500.0,
               'audio_ok_s': 5220.0, 'audio_aborted_s': 280.0, 'audio_unknown_s': 0.0}}"
SRV_OK="d['summary']['faults']['server'] = {'checked': True, 'match': True, 'mismatches': 0,
    'rows': [{'bucket': 'peer_gone', 'server': 25, 'lo': 20, 'hi': 30, 'ok': True}],
    'residual': {'server_audio_s': 5499.0, 'client_sent_s': 5500.0, 'unknown_s': 0.0,
                 'residual_s': -1.0, 'bound_s': 22.4, 'aborts': 70, 'chunk_ms': 320.0,
                 'ok': True, 'aborted_consumed_s': 279.0, 'aborted_sent_s': 280.0},
    'detail': 'every bucket inside its interval'}"
[ -z "$(state healthy F)" ] && ok "a run without a fault plan has no row F" \
    || bad "row F appeared on a run with no plan: $(state healthy F)"
mk faults "$FAULTS
$SRV_OK"
[ "$(verdict faults)" = "QUALIFIED" ] && [ "$(state faults F)" = "PASS" ] \
    && [ "$(state faults 1)" = "PASS" ] && [ "$(state faults A)" = "PASS" ] \
    && ok "70 planned aborts are not losses; the healthy streams qualify and row F passes" \
    || bad "planned aborts were judged as losses: $(verdict faults) / 1 $(state faults 1) / F $(state faults F)"
FOUT=$(python3 "$ROOT/tools/bench/v2_verdict.py" "$TMP/faults" 2>/dev/null)
echo "$FOUT" | grep -q "mid_utterance_silence 10/10.*partial_frame 10/10.*server /v1/health delta: peer_gone 25 in \[20,30\]" \
    && ok "row F shows planned vs executed per point, silent and partial-frame included, and the server's count" \
    || bad "row F does not show the per-point accounting"
echo "$FOUT" | grep -q "residual model work -1.00 s.*bound 70 x 320 ms = 22.40 s" \
    && ok "row F shows the residual-model-work bound" \
    || bad "row F does not show the residual-model-work bound"
mk faults_srvbad "$FAULTS
$SRV_OK
d['summary']['faults']['server'].update(match=False, mismatches=1, detail='server peer_gone 12 outside [20, 30]')"
[ "$(verdict faults_srvbad)" = "NOT QUALIFIED" ] && [ "$(state faults_srvbad F)" = "FAIL" ] \
    && ok "a server that did not count the client's RSTs fails row F" \
    || bad "a server/client abort mismatch passed: $(verdict faults_srvbad)"
mk faults_viol "$FAULTS
$SRV_OK
d['summary']['faults']['violations'] = ['stream 3 rep 7: aborted_by_client without an executed plan']"
[ "$(state faults_viol F)" = "FAIL" ] && [ "$(verdict faults_viol)" = "NOT QUALIFIED" ] \
    && ok "an abort the harness cannot account for fails row F" \
    || bad "a fault accounting violation passed: $(state faults_viol F)"
mk faults_pre "$FAULTS
$SRV_OK
d['summary']['faults']['preempted']['mid_utterance'] = {'server_disconnect': 1}
d['summary']['counts'].update(errors_total=1, error_kinds={'server_disconnect': 1})"
[ "$(state faults_pre 1)" = "FAIL" ] && [ "$(verdict faults_pre)" = "NOT QUALIFIED" ] \
    && ok "an error BEFORE the abort point is a real loss for bound 1" \
    || bad "a failure before the abort point was excused by the plan: $(state faults_pre 1)"
# prefork: no /v1/health cross-check; the workers' dumps are the server's evidence
mk faults_dump "$FAULTS"
[ "$(verdict faults_dump)" = "QUALIFIED" ] && [ "$(state faults_dump F)" = "PASS" ] \
    && ok "prefork: the [DUMP] lines cross-check the aborts, sessions and audio (upper bounds)" \
    || bad "the dump cross-check failed a consistent run: $(verdict faults_dump) / F $(state faults_dump F)"
mk faults_dumpbad "$FAULTS"
sed -i.bak 's/worker=1 seq=6 cancelled=2 idle_timeout=1 peer_gone=1/worker=1 seq=6 cancelled=101 idle_timeout=1 peer_gone=100/' \
    "$TMP/faults_dumpbad/server-soak1.log"
[ "$(state faults_dumpbad F)" = "FAIL" ] && [ "$(verdict faults_dumpbad)" = "NOT QUALIFIED" ] \
    && ok "more peer_gone in the dumps than the client could have caused fails row F" \
    || bad "a server over-count passed: $(state faults_dumpbad F)"
mk faults_proto "$FAULTS"
sed -i.bak 's/worker=1 seq=6 cancelled=2 idle_timeout=1 peer_gone=1 frame_too_large=0 protocol_error=0/worker=1 seq=6 cancelled=3 idle_timeout=1 peer_gone=1 frame_too_large=0 protocol_error=1/' \
    "$TMP/faults_proto/server-soak1.log"
[ "$(state faults_proto F)" = "FAIL" ] \
    && ok "a protocol_error no planned abort can cause fails row F" \
    || bad "an unplanned protocol_error in the dumps passed: $(state faults_proto F)"
mk faults_sess "$FAULTS
d['summary']['faults']['client']['sessions'] = 40"
[ "$(state faults_sess F)" = "FAIL" ] \
    && ok "more server sessions than the client opened (a session missing client-side) fails row F" \
    || bad "a session count the client cannot explain passed: $(state faults_sess F)"
# the dumps say 100 + 100 + 5400 = 5600 s fed; the clients sent 5500 s, + 70 x 320 ms
mk faults_resid "$FAULTS"
sed -i.bak 's/\(worker=2 seq=6 slots .*\) audio_s=100.0/\1 audio_s=5400.0/' \
    "$TMP/faults_resid/server-soak1.log"
[ "$(state faults_resid F)" = "FAIL" ] \
    && python3 "$ROOT/tools/bench/v2_verdict.py" "$TMP/faults_resid" 2>/dev/null | grep -q "EXCEEDED" \
    && ok "model work beyond the audio the clients sent (+ one step per abort) fails row F" \
    || bad "residual model work over the bound passed: $(state faults_resid F)"
mk faults_nosrv "$FAULTS"
sed -i.bak '/ seq=[0-9]* cancelled=/d' "$TMP/faults_nosrv/server-soak1.log"
[ "$(state faults_nosrv F)" = "NO EVIDENCE" ] \
    && [ "$(verdict faults_nosrv)" = "INCONCLUSIVE (missing evidence)" ] \
    && ok "a fault run with no server evidence is inconclusive, never qualified" \
    || bad "an unchecked fault run qualified: $(verdict faults_nosrv) / F $(state faults_nosrv F)"

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
