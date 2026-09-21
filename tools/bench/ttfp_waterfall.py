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
steps, ends = [], []
for L in open(os.path.join(out, "server.log")).read().splitlines():
    if not L.startswith("[TTFP]"):
        continue
    kv = {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.]+)", L)}
    (ends if " END " in L else steps).append(kv)

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

# PAIRING. Ordering by time works only while one stream runs at a time; at any
# real concurrency the END lines interleave. So each client utterance claims the
# unclaimed END line that is CONSISTENT with it -- its first audio arrived after
# that utterance's first write, its first send happened before that utterance saw
# text -- and, among those, the closest. On loopback the true match is separated
# from every impostor by whole milliseconds, and an utterance that finds no
# consistent line is reported rather than silently paired.
claimed, pairs, orphans = set(), [], []
for u in runs:
    cli = u["lag_marks"][0][0]
    fw = cli - u["ttfp_ms"] / 1000.0
    best, bestd = None, None
    for i, e in enumerate(ends):
        if i in claimed or e["first_result"] == 0.0:
            continue
        if not (fw <= e["first_audio"] and e["first_send"] <= cli):
            continue
        d = e["first_audio"] - fw
        if bestd is None or d < bestd:
            best, bestd = i, d
    if best is None:
        orphans.append(u)
        continue
    claimed.add(best)
    pairs.append((u, ends[best]))
if orphans:
    print(f"WARNING: {len(orphans)} utterance(s) matched no server END line; excluded")
if len(ends) - len(claimed):
    print(f"NOTE: {len(ends) - len(claimed)} server END line(s) unclaimed "
          f"(warm-up streams, or streams the client did not keep)")

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

import platform as _pf
print(f"TTFP waterfall: {len(rows)} utterance(s) paired, host {_pf.system()} {_pf.machine()}")
print("A run on the development host is DIAGNOSTIC: the mechanism transfers, the ms do not.\n")

print("PAIRING AND CLOCK CHECK -- every server mark must sit inside the client's window")
ok_all = True
for u, e, cli, fw, mine in rows:
    seq = [fw, e["first_audio"], e["first_result"], e["first_queued"], e["first_send"], cli]
    good = all(seq[i] <= seq[i + 1] + 1e-6 for i in range(len(seq) - 1))
    ok_all &= good
    print(f"  {'/'.join(u['clip'].split('/')[-2:]):<24} "
          f"in={(e['first_audio'] - fw) * 1000:>6.2f}ms  out={(cli - e['first_send']) * 1000:>6.2f}ms  "
          f"{'ordered OK' if good else 'OUT OF ORDER'}")
print(f"  => {'paired, and the two processes share one clock' if ok_all else 'BROKEN'}\n")

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
for u, e, cli, fw, mine in rows:
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

n = len(rows)
grand = sum(agg.values())
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
for u, e, cli, fw, mine in rows:
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
