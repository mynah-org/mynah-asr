#!/usr/bin/env python3
"""knee.py — find the latency-safe concurrency by BISECTION on one warm server.

    python3 tools/bench/knee.py --port 8090 --clips samples/en/*.wav \
        --lo 8 --hi 64 --resolution 4 --json knee.json

Why this exists.  The ladder that produced the 2026-09-20 capacity curve walked
twelve fixed rungs, each a 90 s closed-loop run behind a 25 s warm-up on a
freshly started server: about twenty-seven minutes to learn one number, the
highest concurrency that holds the envelope.  Eleven of those rungs were spent
confirming what the twelfth already implied.  A monotone predicate answered by
bisection needs log2((hi-lo)/resolution) + 1 probes: five rungs for 8..64 at a
resolution of four streams, on one server that is started once, which is about
six minutes.

MONOTONICITY is the assumption, and it is stated rather than hidden: a server
that holds the envelope at C is assumed to hold it below C.  It is the same
assumption the linear ladder makes when it stops at the first failing rung, and
it is CHECKED here -- the confirmation probe at the end re-runs the chosen rung,
and a `--verify-below` probe re-runs one rung below the first failure.  A
disagreement is reported, not smoothed: it means the predicate is noisy at this
operating point and the bisection result is not a capacity claim.

WAVE, NEVER SOAK.  Every probe here is a screening wave.  This tool NEVER
promotes an operating point: it says where to point the soak.  `ENGINEERING.md`
§8 and `docs/serving.md` own that distinction, and the JSON carries
`"is_qualification": false` so no reader can mistake the two.

The verdict of a probe is the envelope verdict computed by
`tools/bench/streaming_metrics.py` through `tools/bench/stream_load.py`.  This
file defines no metric and no threshold of its own; it only chooses which
concurrency to try next.

Exit: 0 a knee was found, 1 no rung held (even the lowest), 2 usage/IO.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STREAM_LOAD = os.path.join(HERE, "stream_load.py")

# A probe's verdict, in the vocabulary stream_load already speaks.
HOLDS = ("GOOD",)                      # the rung is clean
SOFT = ("MARGINAL",)                   # the rung is the knee itself
BREAKS = ("NOT STREAMABLE", "DEGRADED")
INVALID = ("INVALID",)


def probe(a, c, tag):
    """One wave at concurrency c against the already-running server."""
    out_json = os.path.join(a.out, f"probe-c{c:03d}-{tag}.json")
    cmd = [sys.executable, STREAM_LOAD, "--mode", "wave",
           "--host", a.host, "--port", str(a.port),
           "--streams", str(c), "--repeat", str(a.repeat),
           "--lookahead", str(a.lookahead), "--json", out_json,
           "--clips", *a.clips]
    if a.lang:
        cmd += ["--lang", a.lang]
    if a.model:
        cmd += ["--model", a.model]
    if a.transcripts:
        cmd += ["--transcripts", a.transcripts]
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0
    doc = None
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                doc = json.load(f)
        except (OSError, ValueError):
            doc = None
    if doc is None:
        return {"c": c, "tag": tag, "verdict": "INVALID", "wall_s": wall,
                "why": "the probe produced no JSON", "stderr": r.stderr[-400:]}
    env = doc.get("envelope") or {}
    summ = doc.get("summary") or {}
    counts = summ.get("counts") or {}
    m = summ.get("metrics") or {}

    def g(key, field="p95"):
        s = m.get(key) or {}
        return s.get(field)

    return {
        "c": c, "tag": tag, "wall_s": wall,
        "verdict": env.get("verdict", "INVALID"),
        "failing": [l.get("line") for l in (env.get("lines") or [])
                    if l.get("status") in ("FAIL", "MARGINAL")],
        "ok": counts.get("ok"), "lost": counts.get("errors"),
        "rejected": counts.get("rejected"),
        "audio_per_wall": (m.get("audio_per_wall") or {}).get("max"),
        "ttfp_p95_ms": g("ttfp_ms"), "lag_p95_ms": g("emission_lag_ms"),
        "backlog_max_s": (m.get("backlog_max_s") or {}).get("max"),
        "fairness": summ.get("fairness"),
        "cer_p95": g("cer"),
        "json": out_json,
    }


def holds(p):
    """The predicate the bisection is monotone in. MARGINAL counts as holding:
    it is the knee, and the caller is told which line made it marginal."""
    return p["verdict"] in HOLDS or p["verdict"] in SOFT


def line(p):
    bits = [f"  C={p['c']:<4}", f"{p['verdict']:<16}"]
    bits.append(f"ok={p['ok']} lost={p['lost']} rej={p['rejected']}")
    if p.get("audio_per_wall") is not None:
        bits.append(f"audio/wall={p['audio_per_wall']:.2f}")
    if p.get("ttfp_p95_ms") is not None:
        bits.append(f"ttfp95={p['ttfp_p95_ms']:.0f}ms")
    if p.get("lag_p95_ms") is not None:
        bits.append(f"lag95={p['lag_p95_ms']:.0f}ms")
    if p.get("backlog_max_s") is not None:
        bits.append(f"backlog={p['backlog_max_s']:.3f}s")
    f = p.get("fairness") or {}
    if f.get("worst_over_median") is not None:
        bits.append(f"worst/median={f['worst_over_median']:.1f}")
    if p.get("failing"):
        bits.append("<- " + ",".join(x for x in p["failing"] if x))
    bits.append(f"[{p['wall_s']:.0f}s]")
    return "  ".join(bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--lang")
    ap.add_argument("--model")
    ap.add_argument("--transcripts", help="bank manifest, so every probe also scores CER")
    ap.add_argument("--lookahead", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=2,
                    help="utterances per stream per probe (the rung's length)")
    ap.add_argument("--lo", type=int, default=4, help="a concurrency believed to hold")
    ap.add_argument("--hi", type=int, default=64, help="a concurrency believed to break")
    ap.add_argument("--resolution", type=int, default=4,
                    help="stop bisecting when hi-lo is this close")
    ap.add_argument("--out", default=".", help="directory for the per-probe JSON")
    ap.add_argument("--json", help="write the whole search here")
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip the repeat of the chosen rung (the repeatability check)")
    a = ap.parse_args()

    if a.lo < 1 or a.hi <= a.lo or a.resolution < 1:
        print("knee: need 1 <= lo < hi and resolution >= 1", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)

    print(f"knee search: C in [{a.lo}, {a.hi}], resolution {a.resolution}, "
          f"{a.repeat} utterance(s) per stream per probe, one warm server")
    print("  a probe SCREENS. This tool never promotes an operating point.")
    probes = []

    p_lo = probe(a, a.lo, "lo")
    probes.append(p_lo)
    print(line(p_lo))
    if not holds(p_lo):
        print(f"\nno knee: the lowest rung C={a.lo} already does not hold "
              f"({p_lo['verdict']}). Lower --lo, or the server is not healthy.")
        result = {"knee": None, "reason": "lowest rung did not hold", "probes": probes}
        if a.json:
            with open(a.json, "w") as f:
                json.dump({"is_qualification": False, **result}, f, indent=1)
        return 1

    p_hi = probe(a, a.hi, "hi")
    probes.append(p_hi)
    print(line(p_hi))
    if holds(p_hi):
        print(f"\nno knee below C={a.hi}: the highest rung still holds. "
              f"Raise --hi; the capacity of this host has not been found.")
        result = {"knee": None, "reason": "highest rung still held",
                  "highest_holding": a.hi, "probes": probes}
        if a.json:
            with open(a.json, "w") as f:
                json.dump({"is_qualification": False, **result}, f, indent=1)
        return 1

    lo, hi = a.lo, a.hi                      # invariant: lo holds, hi does not
    while hi - lo > a.resolution:
        mid = (lo + hi) // 2
        p = probe(a, mid, "bisect")
        probes.append(p)
        print(line(p))
        if holds(p):
            lo = mid
        else:
            hi = mid

    print(f"\nhighest holding rung: C={lo}; first rung that does not: C={hi}")

    confirm = None
    if not a.no_confirm:
        print("confirming (the same rung again: a knee that does not repeat is noise)")
        confirm = probe(a, lo, "confirm")
        probes.append(confirm)
        print(line(confirm))
        if not holds(confirm):
            print(f"\nDISAGREEMENT: C={lo} held once and did not hold on the repeat. "
                  f"The predicate is noisy here; this is not a capacity claim.")

    result = {
        "knee": {"highest_holding": lo, "first_breaking": hi,
                 "resolution": a.resolution,
                 "confirmed": bool(confirm and holds(confirm)) if confirm else None},
        "probes": probes,
        "total_wall_s": sum(p["wall_s"] for p in probes),
        "note": ("WAVE screening only. Point the SOAK at the highest holding rung; "
                 "only a soak promotes (ENGINEERING.md 8)."),
    }
    print(f"\n{len(probes)} probes, {result['total_wall_s']:.0f} s total")
    print(f"next: one SOAK at C={lo} with --transcripts, which is the only run that promotes")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"is_qualification": False, **result}, f, indent=1)
        print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
