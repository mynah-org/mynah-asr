#!/usr/bin/env python3
"""ttfp_waterfall.py — where the wait before the first word actually goes.

Reads a server log written with MYNAH_ASR_TRACE_TTFP=N beside the per-utterance
JSON of `stream_load.py`, and attributes every millisecond between the client's
first audio write and its first visible partial.

It exists because `lag_ms` on the wire stops at the model callback: framing, the
output ring and the socket write are behind it, and nothing could see them.

    MYNAH_ASR_TRACE_TTFP=14 ./mynah-asr-server -m MODEL -p 8097 > server.log 2>&1 &
    for c in samples/*/*.wav; do
        python3 tools/bench/stream_load.py --mode wave --streams 1 --repeat 1 \
            --clips "$c" --onsets onsets.json --port 8097 --frame-ms 20 \
            --json "out/$(echo "$c" | tr / _ | sed 's/.wav$//').json"
    done
    python3 tools/bench/ttfp_waterfall.py out/

Run it at C=1 first. A concurrency effect must not be allowed to hide a fixed
pipeline floor, and a span that is 0.1 ms on one stream is not the thing to
optimise whatever it becomes under load.

The residual of the waterfall is an algebraic identity -- the spans telescope --
so it proves nothing. The validation is the ordering check: every server mark
must fall inside the client's own window, which fails by whole seconds if a run
is mispaired or the two processes do not share a clock.
"""
"""R-2: the C=1 first-partial waterfall. Measurement only, one utterance at a time."""
import glob, json, os, re, sys

out = sys.argv[1]
# Optional: drop the first N seconds of utterances. A synchronised opening is a
# different question from the steady state and must not be averaged into it.
skip_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
# Three prefork workers write their trace to one inherited stderr, so under load
# a line can be cut by another worker's line. A truncated record is DROPPED and
# COUNTED: silently keeping one would put a half-read timestamp into a mean.
# `pid` is deliberately NOT required: it arrived later than the rest, and a
# report that cannot read the evidence it was written against is a report that
# will quietly be replaced by a fresh guess.
END_KEYS = ("slot", "open", "first_audio", "first_result",
            "first_queued", "first_send", "steps", "deltas", "audio_s")
STEP_KEYS = ("slot", "step", "consumed_s", "arrival", "ready", "sel",
             "mstart", "mend", "emitted")
steps, ends, torn = [], [], 0
for L in open(os.path.join(out, "server.log"), errors="replace").read().splitlines():
    if not L.startswith("[TTFP]"):
        continue
    kv = {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.]+)", L)}
    want, dst = (END_KEYS, ends) if " END " in L else (STEP_KEYS, steps)
    if any(k not in kv for k in want):
        torn += 1
        continue
    dst.append(kv)
if torn:
    print(f"NOTE: {torn} trace line(s) torn by interleaved worker output, dropped")

runs = []
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    d = json.load(open(f))
    if "utterances" not in d:
        continue
    for u in d["utterances"]:
        if u.get("error") or not u.get("lag_marks"):
            continue
        runs.append(u)
runs.sort(key=lambda u: u["t_start"])
if skip_s > 0.0 and runs:
    t0 = runs[0]["t_start"]
    kept = [u for u in runs if u["t_start"] >= t0 + skip_s]
    print(f"opening excluded: dropped {len(runs) - len(kept)} utterance(s) "
          f"started within {skip_s:.0f}s of the first")
    runs = kept

# PAIRING. Positional correspondence is only correct while one stream runs at a
# time; at concurrency the server lines interleave. There is no id shared by the
# two processes, so a pair has to be established by identity-like invariants and
# then PROVED by causality, never assumed:
#
#   identity   the server consumed the whole clip this utterance sent
#              (|END.audio_s - clip audio_s| <= 20 ms) and emitted exactly the
#              number of deltas the client received
#   causality  fw <= first_audio  and  first_result <= first_queued <= first_send
#              <= client_receive     -- a violation is physically impossible
#              within one stream, so it is a mispairing and is REJECTED
#
# Candidates are then assigned globally closest-first rather than in utterance
# order, so one greedy early choice cannot cascade. Everything that fails is
# COUNTED and reported: an aggregate computed over silently dropped samples is
# how a negative publication delay reached a summary table in the first place.
EPS = 1e-6


def causal(u, e, cli, fw):
    return (fw <= e["first_audio"] + EPS
            and e["first_audio"] <= e["first_result"] + EPS
            and e["first_result"] <= e["first_queued"] + EPS
            and e["first_queued"] <= e["first_send"] + EPS
            and e["first_send"] <= cli + EPS)


cands = []
for iu, u in enumerate(runs):
    cli = u["lag_marks"][0][0]
    fw = cli - u["ttfp_ms"] / 1000.0
    for ie, e in enumerate(ends):
        if e["first_result"] == 0.0 or e["first_send"] == 0.0:
            continue
        if abs(e["audio_s"] - u["audio_s"]) > 0.02:
            continue
        if int(e["deltas"]) != int(u["deltas"]):
            continue
        if not causal(u, e, cli, fw):
            continue
        cands.append((e["first_audio"] - fw, iu, ie))
cands.sort()
used_u, used_e, pairs = set(), set(), []
for _, iu, ie in cands:
    if iu in used_u or ie in used_e:
        continue
    used_u.add(iu)
    used_e.add(ie)
    pairs.append((runs[iu], ends[ie]))
pairs.sort(key=lambda p: p[0]["t_start"])

n_unpaired = len(runs) - len(pairs)
print(f"PAIRING  utterances {len(runs)}  server lines {len(ends)}  "
      f"PAIRED {len(pairs)}  unpaired utterances {n_unpaired}  "
      f"unclaimed server lines {len(ends) - len(used_e)}")
print("  a pair must match clip duration and delta count AND satisfy "
      "first_audio <= first_result <= first_queued <= first_send <= client_receive")
if n_unpaired:
    print(f"  {n_unpaired} utterance(s) had no causally consistent partner and are "
          f"EXCLUDED from every number below")

rows = []
for u, e in pairs:
    cli = u["lag_marks"][0][0]
    first_write = cli - u["ttfp_ms"] / 1000.0
    # The slot id is reused between runs, so the window is what selects a run's
    # steps: only those the model started while THIS stream was open.
    mine = sorted([s for s in steps
                   if s["slot"] == e["slot"] and s.get("pid", e.get("pid")) == e.get("pid")
                   and e["first_audio"] <= s["mstart"] <= e["first_result"] + 1e-9],
                  key=lambda s: s["mstart"])
    rows.append((u, e, cli, first_write, mine))

# The measurement host is whatever produced the artifacts, NOT wherever this
# script happens to run: a report that labels Axion data "Darwin" is worse than
# a report with no label at all.
_prov = os.path.join(out, "..", "..", "provenance.txt")
_host = "unknown (no provenance.txt beside the run)"
if os.path.exists(_prov):
    _t = open(_prov).read()
    _h = re.search(r"^host=(.*)$", _t, re.M)
    _c = re.search(r"Model name:\s*(.*)$", _t, re.M)
    _g = re.search(r"^commit=(\S+)", _t, re.M)
    _host = f"{_h.group(1) if _h else '?'} ({_c.group(1).strip() if _c else '?'})" \
            + (f" commit {_g.group(1)[:12]}" if _g else "")
print(f"TTFP waterfall: {len(rows)} utterance(s) paired")
print(f"measured on: {_host}\n")

# Causality already gated the pairing, so this re-states it as a measurement
# rather than as a filter: how far apart the two processes' marks actually are.
if not rows:
    print("nothing paired: no aggregate is computed, and none should be inferred")
    sys.exit(1)
ins = sorted((e["first_audio"] - fw) * 1000.0 for u, e, cli, fw, mine in rows)
outs = sorted((cli - e["first_send"]) * 1000.0 for u, e, cli, fw, mine in rows)
if rows:
    print(f"CLOCK  client write -> server first audio: "
          f"{ins[0]:.2f} .. {ins[-1]:.2f} ms (median {ins[len(ins)//2]:.2f})")
    print(f"       server first send -> client receive: "
          f"{outs[0]:.2f} .. {outs[-1]:.2f} ms (median {outs[len(outs)//2]:.2f})")
    print("       both non-negative: the two processes read one clock\n")

print("PER UTTERANCE, raw, no percentiles")
print(f"{'clip':<24} {'onset':>6} {'TTFP-open':>10} {'TTFP-speech':>12} {'audio@1st':>10} "
      f"{'speech@1st':>11} {'pub_delay':>10} {'blank steps':>12}")
for u, e, cli, fw, mine in rows:
    a1 = u["first_delta_audio_s"]
    print(f"{'/'.join(u['clip'].split('/')[-2:]):<24} {u['onset_s']:>6.2f} "
          f"{u['ttfp_ms']:>9.0f}m {u['ttfp_from_onset_ms']:>11.0f}m {a1:>9.3f}s "
          f"{a1 - u['onset_s']:>10.3f}s {(cli - e['first_result']) * 1000:>9.2f}m "
          f"{sum(1 for s in mine if s['emitted'] == 0):>12}")

print("\nWATERFALL. `model` is split: compute is the sum of the model's own step walls,")
print("audio-wait is the rest, which is the source audio arriving at 1x real time.")
print(f"{'clip':<24} {'net_in':>7} {'1st chunk':>10} {'audio-wait':>11} {'compute':>8} "
      f"{'publish':>8} {'net_out':>8} {'= TTFP-open':>12}")
agg = {"net_in": 0.0, "chunk": 0.0, "await": 0.0, "compute": 0.0, "publish": 0.0, "net_out": 0.0}
no_steps = [r for r in rows if not r[4]]
if no_steps:
    print(f"  {len(no_steps)} paired utterance(s) kept no step trace (torn lines, or the "
          f"first non-blank came after the traced steps) and are excluded from the split")
for u, e, cli, fw, mine in [r for r in rows if r[4]]:
    m0 = mine[0]["mstart"]
    compute = sum(s["mend"] - s["mstart"] for s in mine) * 1000.0
    span = (e["first_result"] - m0) * 1000.0
    v = {"net_in": (e["first_audio"] - fw) * 1000.0,
         "chunk": (m0 - e["first_audio"]) * 1000.0,
         "await": span - compute, "compute": compute,
         "publish": (e["first_send"] - e["first_result"]) * 1000.0,
         "net_out": (cli - e["first_send"]) * 1000.0}
    for k in agg:
        agg[k] += v[k]
    print(f"{'/'.join(u['clip'].split('/')[-2:]):<24} {v['net_in']:>7.2f} {v['chunk']:>10.1f} "
          f"{v['await']:>11.1f} {v['compute']:>8.1f} {v['publish']:>8.2f} {v['net_out']:>8.2f} "
          f"{sum(v.values()):>12.1f}")

n = len(rows) - len(no_steps)
grand = sum(agg.values()) if n else 1.0
owner = {"net_in": "CLIENT/loopback", "chunk": "MODEL: the first chunk needs 256 ms of audio",
         "await": "MODEL: further audio the RNNT wanted, arriving at 1x",
         "compute": "MODEL: encoder+RNNT wall (includes the cold first step)",
         "publish": "SERVING: framing + output ring + socket write",
         "net_out": "CLIENT/loopback"}
print(f"\nWHO OWNS THE WAIT, mean over {n} utterances")
print(f"  {'span':<9} {'mean ms':>9} {'share':>7}   owner")
for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<9} {v / n:>9.1f} {100 * v / grand:>6.1f}%   {owner[k]}")
print(f"  {'TOTAL':<9} {grand / n:>9.1f}")

print("\nEVERY MODEL STEP UNTIL THE FIRST NON-BLANK")
for u, e, cli, fw, mine in [r for r in rows if r[4]][:12]:
    print(f"\n  {'/'.join(u['clip'].split('/')[-2:])}  onset {u['onset_s']:.2f}s  "
          f"first non-blank after {u['first_delta_audio_s']:.3f}s of audio "
          f"({u['first_delta_audio_s'] - u['onset_s']:.3f}s of speech)")
    print(f"    {'step':>4} {'consumed_s':>11} {'idle_before_ms':>15} {'step_ms':>8} "
          f"{'B':>3} {'emitted':>8} {'minflt':>8} {'majflt':>7}")
    prev = None
    for s in mine:
        idle = (s["mstart"] - prev) * 1000.0 if prev else (s["mstart"] - e["first_audio"]) * 1000.0
        print(f"    {int(s['step']):>4} {s['consumed_s']:>11.3f} {idle:>15.1f} "
              f"{(s['mend'] - s['mstart']) * 1000:>8.1f} {int(s.get('B', 0)):>3} "
              f"{int(s['emitted']):>8} {int(s.get('minflt', -1)):>8} {int(s.get('majflt', -1)):>7}")
        prev = s["mend"]
