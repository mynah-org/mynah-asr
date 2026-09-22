#!/usr/bin/env python3
"""v2_verdict.py — check a v2_qualify run against the bounds registered BEFORE it.

`.work/server-v2-qualification.md` §V2-2 registers twelve bounds. This reads a run
directory and checks every one of them, so that "qualified" is a thing a script
concluded from artefacts rather than a thing a person wrote after looking at the
numbers.

    tools/bench/v2_verdict.py ~/asr-evidence/v2/20260922T163000Z [--json out.json]

Three properties on purpose:

  * A bound with no evidence is NOT a pass. It reports NO EVIDENCE and the run
    does not qualify. The 2026-09-20 soak passed every serving gate it could
    measure and was still only screened, because the gate it could not measure
    was the one that mattered.
  * The worst WINDOW is reported beside the pooled percentile. A run that fails
    at minute 23 failed, however good its average.
  * Thresholds come from the registered note and from the manifest's own
    lookahead, never from the run being judged.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# Registered in .work/server-v2-qualification.md V2-2. Bounds 2 and 4 are
# expressed in chunk periods, as the profile expresses them, so they move with
# the preset instead of being three digits someone typed.
FIN_P95_MS = 500.0
LAG_MAX_MS = 3000.0          # bound 9: the max, not a percentile
TREND_MAX_PCT = 50.0         # bound 7: last third vs first third
RSS_GROWTH_MAX = 1.15        # bound 11
OK, BAD, NOEV = "PASS", "FAIL", "NO EVIDENCE"

# The qualified PocketTTS profile does not report a worst gap, it reports how
# MANY gaps crossed 250 ms and how many crossed 500 ms, because "max 700 ms"
# and "700 ms once in 64205 requests" are different products. The ASR analogue
# is the emission lag of every published delta, counted at the cadence the
# model sets and at multiples of it. Counts are reported for every run; only
# the 3000 ms column is a registered bound (bound 9).
STALL_MULTIPLES = (1.0, 2.0, 4.0)

# A 90 s rung cannot evidence every bound: a trend needs more windows than it
# has, and RSS over three samples is noise. So --pick judges a rung on the
# bounds a SCREEN can actually carry, and says so. This selects a candidate to
# soak; it promotes nothing. WAVE screens, SOAK promotes (ENGINEERING.md §8).
SCREEN_BOUNDS = (1, 2, 3, 4, 6, 9, 10)


def chunk_ms(lookahead):
    return (int(lookahead) + 1) * 80.0


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if d is not None else default


def slot_ceiling(dumps, run_manifest, streams):
    """Was this run able to CONNECT the concurrency it claims to measure?

    A WebSocket stream holds one HTTP thread for its whole life, so W x
    --threads is the fleet's connection ceiling. Above it the surplus clients
    wait for a thread and the run reports no error and no 503: on 2026-09-22 the
    C=32 rung ran 24 streams, produced fewer utterances than C=24 and lower
    throughput, and read as noise. A run that could not connect its own
    concurrency does not measure the machine, so it is INVALID rather than
    merely worse."""
    lim = (run_manifest or {}).get("connection_ceiling")
    peak = None
    for seqs in dumps.values():
        for rec in seqs.values():
            if rec.get("active") is not None:
                peak = rec["active"] if peak is None else max(peak, rec["active"])
    if lim is None:
        return None, peak
    if streams is not None and streams > lim:
        return (f"concurrency C={streams} is above this fleet's connection ceiling of "
                f"{lim}: the surplus streams waited for an HTTP thread and the rung "
                f"measured the ceiling, not the machine"), peak
    return None, peak


def parse_dumps(path):
    """Per worker, the ordered (seq, active, steps, model_busy_s) samples."""
    if not os.path.exists(path):
        return {}
    by = {}
    for line in open(path, errors="replace"):
        if not line.startswith("[DUMP]"):
            continue
        m = re.search(r"worker=(\d+) seq=(\d+)", line)
        if not m:
            continue
        w, seq = int(m.group(1)), int(m.group(2))
        rec = by.setdefault(w, {}).setdefault(seq, {})
        a = re.search(r"slots active=(\d+) cap=(\d+) sessions=(\d+) steps=(\d+)", line)
        if a:
            rec["active"], rec["steps"] = int(a.group(1)), int(a.group(4))
        b = re.search(r"split model_busy_s=([\d.]+)", line)
        if b:
            rec["busy"] = float(b.group(1))
    return by


def stall_check(dumps):
    """Bound 8: no interval in which slots were active and nothing progressed."""
    if not dumps:
        return NOEV, "no [DUMP] line in the server log: SIGUSR1 produced nothing", []
    viol, n_int = [], 0
    for w, seqs in sorted(dumps.items()):
        ordered = [seqs[s] for s in sorted(seqs)]
        for a, b in zip(ordered, ordered[1:]):
            if "steps" not in a or "steps" not in b:
                continue
            n_int += 1
            # Active at BOTH ends of the interval: a worker that went idle
            # because its streams ended is not stalled, it is finished.
            if a.get("active", 0) > 0 and b.get("active", 0) > 0 and b["steps"] == a["steps"]:
                viol.append(f"worker {w}: steps stuck at {a['steps']} with "
                            f"{a['active']}->{b.get('active')} slots active")
    if n_int == 0:
        return NOEV, "dumps carry no step counter to compare", []
    return (OK if not viol else BAD,
            f"{n_int} interval(s) across {len(dumps)} worker(s), {len(viol)} with no progress",
            viol)


def proc_check(path):
    """Bounds 11 and 12, from the OS samples taken on the dump tick.

    Returns ((rss_state, rss_msg), (deaths_state, deaths_msg))."""
    noev = lambda why: ((NOEV, why), (NOEV, why))
    if not os.path.exists(path):
        return noev("no procsample file: nothing sampled the workers")
    rosters, rss = [], {}
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("t="):
            m = re.search(r"pids=([\d,]*)", line)
            rosters.append(tuple(sorted(x for x in (m.group(1) or "").split(",") if x)))
            continue
        f = line.split()
        if len(f) >= 4 and f[0].isdigit() and f[1].isdigit():
            rss.setdefault(f[0], []).append(int(f[1]))
    if not rosters:
        return noev("no roster sample in the procsample file")

    stable = len(set(rosters)) == 1
    deaths = (OK if stable else BAD,
              f"{len(rosters)} roster sample(s), {len(set(rosters))} distinct worker set(s)"
              + ("" if stable else f"; sets seen: {sorted(set(rosters))[:3]}"))

    # Bound 11 compares the end against the FIRST sample, which is taken one
    # dump interval in -- by then the fleet has loaded its weights, so it is
    # already a post-warm-up figure rather than a cold one.
    detail, worst = [], 0.0
    for pid, series in sorted(rss.items()):
        if len(series) < 2 or not series[0]:
            continue
        ratio = series[-1] / series[0]
        worst = max(worst, ratio)
        detail.append(f"pid {pid} {series[0] // 1024} -> {series[-1] // 1024} MiB ({ratio:.3f}x)")
    if not detail:
        return (NOEV, "fewer than two rss samples per worker"), deaths
    return ((OK if worst <= RSS_GROWTH_MAX else BAD,
             f"worst {worst:.3f}x over the run, bound {RSS_GROWTH_MAX}x; " + "; ".join(detail)),
            deaths)


def bank_id(man):
    """A stable id for the audio a run actually played.

    The manifest already carries a sha256 per clip, so the bank identifies
    itself from inside the run record rather than from a sibling file that
    could belong to a different run."""
    bank = man.get("bank") or {}
    rows = sorted((e.get("clip", ""), e.get("sha256", ""))
                  for v in bank.values() for e in v)
    if not rows:
        return None
    import hashlib
    h = hashlib.sha256()
    for clip, sha in rows:
        h.update(f"{clip} {sha}\n".encode())
    return f"{h.hexdigest()[:16]} / {len(rows)} clips"


def stall_counts(d, cm):
    """How many published deltas crossed each threshold, and over how many."""
    utts = d.get("utterances") or []
    lags = [v for u in utts for _, v in (u.get("lag_marks") or [])]
    if not lags:
        return None
    rows = []
    for mult in STALL_MULTIPLES:
        thr = cm * mult
        rows.append((thr, sum(1 for v in lags if v > thr)))
    rows.append((LAG_MAX_MS, sum(1 for v in lags if v > LAG_MAX_MS)))
    return {"deltas": len(lags), "rows": rows,
            "worst_ms": max(lags), "over_1s": sum(1 for v in lags if v > 1000.0)}


def window_check(d, limit):
    """Bound 6: EVERY steady window, not the pooled percentile."""
    dr = g(d, "summary", "drift", "emission_lag_ms", default={})
    wins = dr.get("windows") or []
    usable = [w for w in wins if w.get("p95") is not None]
    if len(usable) < 2:
        return NOEV, "fewer than two windows with data", None
    ns = sorted(w["n"] for w in usable)
    med = ns[len(ns) // 2]
    steady = [w for w in usable if w["n"] >= 0.5 * med]
    excl = [w["window"] for w in usable if w["n"] < 0.5 * med]
    bad = [w for w in steady if w["p95"] > limit]
    worst = max(steady, key=lambda w: w["p95"]) if steady else None
    msg = (f"{len(steady)} steady window(s), worst p95 "
           f"{worst['p95']:.0f} ms in window {worst['window']} "
           f"[{worst['t0_s']:.0f}-{worst['t1_s']:.0f}s] vs bound {limit:.0f} ms")
    if excl:
        msg += f"; {len(excl)} ramp/drain window(s) excluded: {excl}"
    return (OK if not bad else BAD), msg, worst


def verdict_for(path, dump_path, proc_path, run_manifest=None):
    d = json.load(open(path))
    s = d["summary"]
    man = d.get("manifest") or {}
    la = man.get("lookahead", 3)
    cm = chunk_ms(la)
    m, c = s["metrics"], s["counts"]
    rows = []

    def row(n, name, state, detail):
        rows.append({"bound": n, "name": name, "state": state, "detail": detail})

    row(1, "established streams lost", OK if c["errors"] == 0 else BAD,
        f"{c['errors']} error(s) over {c['ok']} completed utterance(s)")
    p95 = g(m, "emission_lag_ms", "p95")
    row(2, "emission lag p95", NOEV if p95 is None else (OK if p95 <= cm else BAD),
        "no sample" if p95 is None else f"{p95:.0f} ms vs bound {cm:.0f} ms ((lookahead+1)x80)")
    f95 = g(m, "finalization_lag_ms", "p95")
    row(3, "finalization p95", NOEV if f95 is None else (OK if f95 <= FIN_P95_MS else BAD),
        "no sample" if f95 is None else f"{f95:.0f} ms vs bound {FIN_P95_MS:.0f} ms")
    bmax = g(m, "backlog_max_s", "max")
    blim = 2.0 * cm / 1000.0
    row(4, "backlog max", NOEV if bmax is None else (OK if bmax <= blim else BAD),
        "no sample" if bmax is None else f"{bmax:.3f} s vs bound {blim:.3f} s (2 chunks)")
    row(5, "503 admission refusals", OK,
        f"{c['rejected']} (counted separately; a 503 is the admission ladder, never a loss)")
    st, msg, _ = window_check(d, cm)
    row(6, "per-window drift", st, msg)
    trend = g(s, "drift", "emission_lag_ms", "trend_pct")
    row(7, "monotonic trend", NOEV if trend is None else (OK if trend <= TREND_MAX_PCT else BAD),
        "not computed" if trend is None else f"{trend:+.1f}% last third vs first third, bound +{TREND_MAX_PCT:.0f}%")
    dumps = parse_dumps(dump_path)
    st, msg, viol = stall_check(dumps)
    row(8, "server-side stall", st, msg + ("; " + "; ".join(viol[:3]) if viol else ""))
    lmax = g(m, "emission_lag_ms", "max")
    row(9, "client-observable stall", NOEV if lmax is None else (OK if lmax <= LAG_MAX_MS else BAD),
        "no sample" if lmax is None else f"max emission lag {lmax:.0f} ms vs bound {LAG_MAX_MS:.0f} ms")
    idf, rff = s.get("identity_fail") or {}, s.get("reference_fail") or {}
    ngroups = len(s.get("text_groups") or {})
    if not ngroups:
        st, msg = NOEV, "no transcript was grouped"
    elif not rff and not idf:
        st, msg = OK, f"{ngroups} clip(s), every transcript identical across streams and to the unloaded reference"
    else:
        st, msg = BAD, f"{len(idf)} clip(s) differ across streams, {len(rff)} differ from the reference"
    row(10, "quality parity", st, msg)
    (rss_state, rss_msg), (dead_state, dead_msg) = proc_check(proc_path)
    row(11, "worker RSS growth", rss_state, rss_msg)
    row(12, "worker deaths", dead_state, dead_msg)

    invalid, peak_slots = slot_ceiling(dumps, run_manifest, man.get("concurrency"))
    failed = [r for r in rows if r["state"] == BAD]
    missing = [r for r in rows if r["state"] == NOEV]
    return {
        "invalid": invalid, "peak_active_slots": peak_slots,
        "file": os.path.basename(path),
        # These are stream_load's OWN manifest field names. An earlier draft read
        # "streams" and "duration", which exist nowhere: both came back None and
        # the header printed C=None while every bound still read PASS.
        "streams": man.get("concurrency"), "duration_s": man.get("duration_s"),
        "seed": man.get("seed"), "chunk_period_ms": man.get("chunk_period_ms"),
        "utterances": c["ok"], "audio_s": round(c.get("audio_s", 0.0), 1),
        "audio_per_wall": g(m, "audio_per_wall", "max"),
        "ttfp_p95_ms": g(m, "ttfp_ms", "p95"),
        "pacing": g(s, "pacing", "verdict"),
        "stalls": stall_counts(d, cm),
        "max_delta_gap_p95_ms": g(m, "max_delta_gap_ms", "p95"),
        "fairness": s.get("fairness"),
        "bank_sha256": bank_id(man),
        "rows": rows,
        "verdict": "INVALID" if invalid else
                   ("QUALIFIED" if not failed and not missing else
                    ("NOT QUALIFIED" if failed else "INCONCLUSIVE (missing evidence)")),
    }


def pick(run):
    """The highest ladder rung that clears every bound a 90 s screen can carry.

    Prints one line per rung and the choice, so an unattended chain leaves the
    reason behind it rather than only the number."""
    rungs = []
    for path in sorted(glob.glob(os.path.join(run, "ladder-C*.json"))):
        tag = os.path.basename(path)[:-5]
        mp = os.path.join(run, "manifest.json")
        v = verdict_for(path,
                        os.path.join(run, f"server-{tag}.log"),
                        os.path.join(run, f"procsample-{tag}.txt"),
                        json.load(open(mp)) if os.path.exists(mp) else None)
        bad = [r for r in v["rows"] if r["bound"] in SCREEN_BOUNDS and r["state"] != OK]
        if v.get("invalid"):
            bad = [{"bound": 0, "name": "INVALID: " + v["invalid"]}] + bad
        rungs.append((v["streams"], not bad, v, bad))
    rungs.sort(key=lambda r: (r[0] is None, r[0]))
    for c, good, v, bad in rungs:
        why = "" if good else "  <- " + "; ".join(f"{r['bound']} {r['name']}" for r in bad)
        lag = g(v, "rows")
        print(f"  C={c:<4} {'CLEARS' if good else 'FAILS '} the screen bounds{why}",
              file=sys.stderr)
    passing = [c for c, good, _, _ in rungs if good and c]
    if not passing:
        print("  no rung cleared the screen bounds", file=sys.stderr)
        return None
    return max(passing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="a v2_qualify run directory")
    ap.add_argument("--json", help="write the verdicts here")
    ap.add_argument("--pick", action="store_true",
                    help="print the highest ladder rung that clears the screen "
                         "bounds, for an unattended chain to soak next")
    a = ap.parse_args()
    if a.pick:
        c = pick(a.run)
        if c is None:
            return 1
        print(c)
        return 0
    out = []
    mpath = os.path.join(a.run, "manifest.json")
    run_manifest = json.load(open(mpath)) if os.path.exists(mpath) else None
    for path in sorted(glob.glob(os.path.join(a.run, "soak*.json"))
                       + sorted(glob.glob(os.path.join(a.run, "ladder-C*.json")))):
        tag = os.path.basename(path)[:-5]
        for cand in (f"server-{tag}.log", f"server-{tag.split('-C')[0]}.log"):
            dump = os.path.join(a.run, cand)
            if os.path.exists(dump):
                break
        for cand in (f"procsample-{tag}.txt", f"procsample-{tag.split('-C')[0]}.txt"):
            proc = os.path.join(a.run, cand)
            if os.path.exists(proc):
                break
        out.append(verdict_for(path, dump, proc, run_manifest))
    if not out:
        sys.exit(f"no soak*.json or ladder-C*.json in {a.run}")
    for v in out:
        head = (f"{v['file']}  C={v['streams']}  {v['utterances']} utterances, "
                f"{v['audio_s']:.0f} audio-s, {v['pacing']}"
                + (f", bank {v['bank_sha256']}" if v.get("bank_sha256") else ""))
        print("\n" + head)
        print("-" * len(head))
        if v.get("invalid"):
            print(f"  !!  INVALID RUN: {v['invalid']}")
        for r in v["rows"]:
            print(f"  {r['bound']:2d}  {r['name']:26s} {r['state']:11s} {r['detail']}")
        if v.get("peak_active_slots") is not None:
            print(f"      peak active slots on one worker: {v['peak_active_slots']}")
        st = v.get("stalls")
        if st:
            cols = "   ".join(f">{t:.0f}ms: {n}" for t, n in st["rows"])
            print(f"      stalls over {st['deltas']} published deltas   {cols}"
                  f"   worst {st['worst_ms']:.0f} ms")
        fa = v.get("fairness")
        if fa and fa.get("worst_over_median"):
            print(f"      fairness  worst stream p95 {fa['worst_p95_ms']:.0f} ms vs median "
                  f"{fa['median_p95_ms']:.0f} ms ({fa['worst_over_median']:.2f}x over "
                  f"{fa['streams']} streams)")
        extra = []
        if v["max_delta_gap_p95_ms"]:
            extra.append(f"max gap between deltas p95 {v['max_delta_gap_p95_ms']:.0f} ms")
        if v["audio_per_wall"]:
            extra.append(f"audio/wall {v['audio_per_wall']:.2f}x")
        if v["ttfp_p95_ms"]:
            extra.append(f"TTFP p95 {v['ttfp_p95_ms']:.0f} ms (REPORTED, no gate)")
        if extra:
            print("      " + "   ".join(extra))
        print(f"  => {v['verdict']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")
    return 0 if all(v["verdict"] == "QUALIFIED" for v in out) else 1


if __name__ == "__main__":
    sys.exit(main())
