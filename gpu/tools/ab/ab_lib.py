#!/usr/bin/env python3
"""ab_lib.py -- the bookkeeping of gpu/tools/ab/ab_server.sh and ab_device.sh.

    ab_lib.py plan  --kind server|device --out DIR --tag T --order ORDER --cycles N
                    --levels "16 64" --arm A:LABEL:BIN:ARGS ... [--repeat R] [--clips ...]
                    [--onsets F] [--load-timeout S] [--ready-est S] [--extra k=v ...]
        writes DIR/ab.json (the plan: one entry per run, in execution order) and
        prints one line per run "RUN LEVEL ARM" for the shell loop, then the
        expected wall time on stderr
    ab_lib.py meta  OUT_JSON key=value ...      one run's facts (ints/floats/bools parsed)
    ab_lib.py line  LOAD_JSON LABEL            the one-line summary of a finished run

The ORDER is a string of arm letters (ABBA, ABCCBA) played CYCLES times per
level; levels are the outer loop, so the runs a comparison pairs are minutes
apart, not an hour. Standard library only."""
import argparse
import hashlib
import json
import os
import sys
import time
import wave


def sha16(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()[:16]
    except OSError:
        return None


def clip_s(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def wave_wall_s(clips, streams, repeat, per_utt_s=1.5):
    """The wall of one stream_load --mode wave run: stream i plays the sorted clip list at
    positions i + k*streams (k < repeat), back to back, at real time; the run ends with
    its slowest stream. per_utt_s covers connect + finalisation."""
    cl = sorted(set(clips))
    ds = [clip_s(c) for c in cl]
    n = len(cl)
    worst = 0.0
    for i in range(streams):
        t = sum(ds[(i + k * streams) % n] + per_utt_s for k in range(repeat))
        worst = max(worst, t)
    return worst


def cmd_plan(a):
    arms = {}
    for spec in a.arm:
        letter, label, binp, args = (spec.split(":", 3) + ["", "", ""])[:4]
        arms[letter] = {"label": label or letter, "bin": binp, "bin_sha": sha16(binp),
                        "args": args}
    order = [c for c in a.order if not c.isspace()]
    bad = [c for c in order if c not in arms]
    if bad:
        raise SystemExit(f"ab_lib: ORDER uses arm(s) {bad} with no binary/args")
    first = []
    for c in order:
        if c not in first:
            first.append(c)
    for c in first:
        if order.count(c) * a.cycles < 2:
            print(f"ab_lib: WARNING arm {c} runs once per level: no noise floor, its verdict "
                  f"is NULL by construction", file=sys.stderr)
    levels = [int(x) for x in a.levels.split()]
    plan, seq = [], 0
    for lvl in levels:
        for cyc in range(a.cycles):
            for pos, arm in enumerate(order):
                seq += 1
                plan.append({"run": f"C{lvl:03d}-{seq:02d}-{arm}", "seq": seq, "level": lvl,
                             "arm": arm, "cycle": cyc, "pos": pos,
                             "streams": lvl if a.kind == "server" else None,
                             "repeat": a.repeat if a.kind == "server" else None})
    extra = dict(kv.split("=", 1) for kv in (a.extra or []))
    ab = {"kind": a.kind, "tag": a.tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "order": "".join(order), "cycles": a.cycles, "arms_order": first, "arms": arms,
          "levels": levels, "repeat": a.repeat, "clips": sorted(set(a.clips or [])),
          "onsets": a.onsets, "plan": plan, "extra": extra}
    est = []
    if a.kind == "server" and a.clips:
        per = {lvl: wave_wall_s(a.clips, lvl, a.repeat) for lvl in levels}
        for r in plan:
            est.append(a.ready_est + per[r["level"]] + a.stop_est)
        ab["expected_load_wall_s"] = per
    else:
        est = [a.ready_est + a.stop_est for _ in plan]
    ab["expected_wall_s"] = sum(est)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ab.json"), "w") as f:
        json.dump(ab, f, indent=1)
    for r in plan:
        print(r["run"], r["level"], r["arm"])
    per_lvl = ", ".join(f"C{k}: {v:.0f}s/run" for k, v in ab.get("expected_load_wall_s", {}).items())
    print(f"ab_lib: {len(plan)} runs, expected wall {sum(est) / 60:.1f} min"
          + (f" (load {per_lvl}; + {a.ready_est + a.stop_est:.0f}s start/stop each)" if per_lvl
             else ""), file=sys.stderr)
    return 0


def parse_val(v):
    if v in ("true", "false"):
        return v == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def cmd_meta(a):
    d = {}
    for kv in a.kv:
        k, v = kv.split("=", 1)
        d[k] = parse_val(v)
    with open(a.out_json, "w") as f:
        json.dump(d, f, indent=1)
    return 0


def cmd_line(a):
    try:
        d = json.load(open(a.load_json))
    except Exception as e:                       # noqa: BLE001
        print(f"  {a.label}: no load JSON ({e})")
        return 0
    s = d["summary"]
    c, m = s["counts"], s["metrics"]
    g = lambda k: (m.get(k) or {}).get("p95")  # noqa: E731
    f = lambda v: "n/a" if v is None else f"{v:.0f}"  # noqa: E731
    apw = (m.get("audio_per_wall") or {}).get("max")
    print(f"  {a.label} ok={c['ok']}/{c['utterances']} err={c.get('errors_total', c['errors'])} "
          f"lag95={f(g('emission_lag_ms'))}ms fin95={f(g('finalization_lag_ms'))}ms "
          f"ttfp95={f(g('ttfp_ms'))}ms ttfp_on95={f(g('ttfp_from_onset_ms'))}ms "
          f"thr={'n/a' if apw is None else f'{apw:.2f}'}x")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--kind", choices=("server", "device"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--order", required=True)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--levels", required=True)
    p.add_argument("--arm", action="append", required=True, help="LETTER:LABEL:BIN:ARGS")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--clips", nargs="*")
    p.add_argument("--onsets")
    p.add_argument("--ready-est", type=float, default=30.0)
    p.add_argument("--stop-est", type=float, default=10.0)
    p.add_argument("--extra", nargs="*")
    m = sub.add_parser("meta")
    m.add_argument("out_json")
    m.add_argument("kv", nargs="*")
    ln = sub.add_parser("line")
    ln.add_argument("load_json")
    ln.add_argument("label")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "meta": cmd_meta, "line": cmd_line}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
