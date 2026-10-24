#!/usr/bin/env python3
"""ab_report.py -- the verdict of an interleaved A/B (ABBA) session.

    gpu/tools/ab/ab_report.py OUT/TAG [--json verdict.json] [--identity all|arm]
    gpu/tools/ab/ab_report.py OUT            # every session (dir with ab.json) below

Reads what gpu/tools/ab/ab_server.sh (kind "server": load-*.json from
tools/bench/stream_load.py) or gpu/tools/ab/ab_device.sh (kind "device":
bench-*.txt from tests/test_cuda_stream --bench) wrote, and applies the gates
IN THIS ORDER. A later gate is not evaluated when an earlier one fails, so a
latency number is never quoted for a run that lost or changed a transcript.

  1. IDENTITY (server only). Every OK utterance of a clip carries the same
     transcript hash: within a run (across its streams) and across all runs
     and arms. `--identity arm` relaxes the cross-ARM part to a report (for a
     patch that is meant to change numerics); within-run and within-arm
     identity are always required.
  2. INTEGRITY. Every planned run produced its JSON; errors, missing and
     rejected utterances, dead streams, a broken conservation, a server that
     did not come up, timed out, died or reported a device error: all 0.
     Device kind: exit 0, a BENCH line, full_steps > 0, no FAIL line.
  3. LATENCY (only then). Per run: emission lag p50/p95/p99 (pooled over
     deltas), finalisation p50/p95/p99, TTFP raw and TTFP from the speech
     onset p50/p95/p99, throughput (audio-s per wall-s), server steps,
     cohorts, mean lanes per cohort, mean step wall. Device kind:
     device_ms_per_pass, step_wall_ms and every stage's ms/pass.

Statistics, per concurrency level and per arm X against the baseline arm A
(the first arm of ORDER):

  median(X)   median over X's runs (ABBA repeats)
  spread(X)   max - min over X's runs; AA = spread(A) is the noise floor
  floor       max(spread(A), spread(X))       (the stricter of the two)
  threshold   max(K * floor, ABS_MIN, REL_MIN * |median(A)|)
              K = 2 (--k), ABS_MIN = 5 ms server / 0.02 ms device
              (--abs-min-ms), REL_MIN = 2 % server / 1 % device (--rel-min);
              throughput uses REL_MIN only
  effect      median(X) - median(A), and its % of median(A)
  pairs       the i-th run of A against the i-th run of X: ABBA gives the
              pairs (A1, B1) and (A2, B2), ABCCBA gives (A1, B1), (A2, B2)
              for B and the same for C -- mirror pairs that cancel drift

  A metric is BETTER (WORSE) when the effect AND every pair's delta are past
  the threshold in the improving (degrading) direction; anything else is
  NULL. It takes >= 2 runs per arm: a single pair has no noise floor, so it
  is NULL by construction.

  Level verdict for X: FAIL if any PRIMARY metric is WORSE; PASS if one is
  BETTER and none is WORSE; NULL otherwise. PRIMARY (server): emission lag
  p95, finalisation p95, TTFP-from-onset p95 (TTFP raw p95 when no onset
  map), throughput. PRIMARY (device): device_ms_per_pass. Everything else is
  reported with its effect but does not vote.
  Arm verdict: FAIL if any level FAILs, PASS if any level PASSes, else NULL.
  WINNER (for a sweep): the PASS arm with the largest mean relative
  improvement of emission lag p95 (device: device_ms_per_pass) over the
  levels; the baseline when no arm PASSes.

Exit 0 = the gates held (read the verdicts), 2 = a gate failed (the session
is not an A/B result), 1 = usage. Standard library only."""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                "tools", "bench"))
import streaming_metrics as M  # noqa: E402

# name, direction (-1 = lower is better, +1 = higher, 0 = informational), primary
SERVER_METRICS = [
    ("lag_p50", -1, False), ("lag_p95", -1, True), ("lag_p99", -1, False),
    ("fin_p50", -1, False), ("fin_p95", -1, True), ("fin_p99", -1, False),
    ("ttfp_p50", -1, False), ("ttfp_p95", -1, False), ("ttfp_p99", -1, False),
    ("ttfp_on_p50", -1, False), ("ttfp_on_p95", -1, True), ("ttfp_on_p99", -1, False),
    ("thr_x", +1, True),
    ("steps", 0, False), ("cohorts", 0, False), ("lanes_mean", 0, False),
    ("step_wall_ms", 0, False),
]
STAGES = ["h2d", "subsample", "ffn1", "att_proj", "att_core", "att_out", "conv", "ffn2",
          "post", "dec_joint", "dec_pred", "dec_sync", "d2h", "other"]
DEVICE_METRICS = [("device_ms_per_pass", -1, True), ("step_wall_ms", -1, False)] + \
    [(f"stage_{s}", -1, False) for s in STAGES] + [(f"share_{s}", 0, False) for s in STAGES]


def jload(path):
    with open(path) as f:
        return json.load(f)


def text_hash(t):
    return hashlib.sha256((t or "").encode("utf-8")).hexdigest()[:16]


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if not n:
        return None
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def p(xs, q):
    return M.pct(xs, q) if xs else None


# --------------------------------------------------------------------------- loading


def load_session(d):
    ab = jload(os.path.join(d, "ab.json"))
    req = str((ab.get("extra") or {}).get("require_cuda", "0")) == "1"
    runs = []
    for r in ab["plan"]:
        meta_p = os.path.join(d, f"meta-{r['run']}.json")
        meta = jload(meta_p) if os.path.exists(meta_p) else None
        runs.append(dict(r, meta=meta, dir=d, require_cuda=req))
    return ab, runs


def health_delta(before, after, *path):
    v0, v1 = before or {}, after or {}
    for k in path:
        v0 = v0.get(k) if isinstance(v0, dict) else None
        v1 = v1.get(k) if isinstance(v1, dict) else None
    if not isinstance(v0, (int, float)) or not isinstance(v1, (int, float)):
        return None
    return v1 - v0


def server_run(run):
    """(load JSON or None, integrity problems, metrics)."""
    problems = []
    meta = run["meta"]
    if meta is None:
        return None, ["never ran (no meta: budget, lock or an earlier abort)"], {}
    if not meta.get("ready"):
        problems.append("server did not become ready")
    if meta.get("timed_out"):
        problems.append(f"load timed out (rc {meta.get('load_rc')})")
    if meta.get("ready") and not meta.get("server_alive_after", True):
        problems.append("server died during the run")
    if meta.get("ready") and meta.get("health_status") not in (None, "ok"):
        problems.append(f"health status {meta.get('health_status')!r} after the run")
    if run.get("require_cuda") and meta.get("engine_cuda") is False:
        problems.append("the server banner does not say engine=cuda")
    path = os.path.join(run["dir"], f"load-{run['run']}.json")
    if not os.path.exists(path):
        problems.append("no load JSON")
        return None, problems, {}
    try:
        d = jload(path)
    except ValueError as e:
        problems.append(f"unreadable load JSON ({e})")
        return None, problems, {}
    s = d.get("summary", {})
    c = s.get("counts", {})
    utts = d.get("utterances", [])
    bad = [u for u in utts if M.outcome(u) != M.OK_OUTCOME]
    kinds = {}
    for u in bad:
        kinds[M.outcome(u)] = kinds.get(M.outcome(u), 0) + 1
    if bad:
        problems.append(f"{len(bad)} utterance(s) not ok: {kinds}")
    for k in ("errors_total", "missing", "stream_deaths", "rejected_total"):
        if c.get(k):
            problems.append(f"counts.{k} = {c[k]}")
    exp = (run.get("streams") or 0) * (run.get("repeat") or 0)
    if exp and len(utts) != exp:
        problems.append(f"{len(utts)} utterance records, expected {exp}")
    if s.get("accounting", {}).get("conservation") is False:
        problems.append("conservation broken: " + str(s["accounting"].get("conservation_detail")))
    env = d.get("envelope", {})
    if env.get("verdict") == "INVALID":
        problems.append("stream_load INVALID: " + "; ".join(env.get("invalid_reasons", [])))
    man = d.get("manifest", {})
    hb, ha = man.get("health_before") or {}, man.get("health_after") or {}
    if "unreachable" in ha:
        problems.append("health unreachable right after the load")
    if ha.get("balanced") is False:
        problems.append("server books not balanced after the load")
    dev_err = health_delta(hb, ha, "engine", "device_errors_total")
    if dev_err:
        problems.append(f"{dev_err:.0f} device error(s)")

    ok = [u for u in utts if M.outcome(u) == M.OK_OUTCOME]
    lag = [m[1] for u in ok for m in (u.get("lag_marks") or [])]
    fin = [u["fin_ms"] for u in ok if u.get("fin_ms") is not None]
    ttfp = [u["ttfp_ms"] for u in ok if u.get("ttfp_ms") is not None]
    ton = [u["ttfp_from_onset_ms"] for u in ok if u.get("ttfp_from_onset_ms") is not None]
    m = {}
    for name, xs in (("lag", lag), ("fin", fin), ("ttfp", ttfp), ("ttfp_on", ton)):
        for q in (50, 95, 99):
            m[f"{name}_p{q}"] = p(xs, q)
    apw = (s.get("metrics", {}).get("audio_per_wall") or {}).get("max")
    if apw is None:
        aud = sum(u.get("audio_s") or 0.0 for u in ok)
        t0 = min((u["t_start"] for u in utts if u.get("t_start") is not None), default=None)
        t1 = max((u["t_end"] for u in utts if u.get("t_end") is not None), default=None)
        apw = aud / (t1 - t0) if t0 is not None and t1 is not None and t1 > t0 else None
    m["thr_x"] = apw
    m["steps"] = health_delta(hb, ha, "steps")
    m["cohorts"] = health_delta(hb, ha, "batch", "batched_steps_total")
    lanes = health_delta(hb, ha, "batch", "ready_sum")
    m["lanes_mean"] = lanes / m["cohorts"] if lanes is not None and m["cohorts"] else None
    sw = health_delta(hb, ha, "batch", "step_wall_ms", "sum")
    swn = health_delta(hb, ha, "batch", "step_wall_ms", "count")
    m["step_wall_ms"] = sw / swn if sw is not None and swn else None
    return d, problems, m


BENCH_RE = re.compile(r"^BENCH C=(\d+) full_steps=(\d+) passes=(\d+) device_ms_per_pass=([\d.]+) "
                      r"step_wall_ms=([\d.]+)")
STAGE_RE = re.compile(r"^BENCH\s+(\w+)\s+([\d.]+) ms/pass")


def parse_bench(text):
    """{metric: value} from tests/test_cuda_stream --bench output, None without a BENCH line."""
    out = None
    for line in text.splitlines():
        mm = BENCH_RE.match(line)
        if mm:
            out = {"C": int(mm.group(1)), "full_steps": int(mm.group(2)),
                   "passes": int(mm.group(3)), "device_ms_per_pass": float(mm.group(4)),
                   "step_wall_ms": float(mm.group(5))}
            continue
        ms = STAGE_RE.match(line)
        if ms and out is not None and not line.startswith("BENCH ready"):
            out[f"stage_{ms.group(1)}"] = float(ms.group(2))
    return out


def device_run(run):
    problems = []
    meta = run["meta"]
    if meta is None:
        return None, ["never ran (no meta)"], {}
    if meta.get("timed_out"):
        problems.append(f"bench timed out (rc {meta.get('rc')})")
    elif meta.get("rc") not in (0, None):
        problems.append(f"bench exit {meta.get('rc')}")
    path = os.path.join(run["dir"], f"bench-{run['run']}.txt")
    if not os.path.exists(path):
        return None, problems + ["no bench output"], {}
    with open(path, errors="replace") as f:
        text = f.read()
    if any(line.startswith("FAIL") for line in text.splitlines()):
        problems.append("a FAIL line in the bench output")
    b = parse_bench(text)
    if b is None:
        return None, problems + ["no BENCH C= line"], {}
    if b["full_steps"] <= 0 or b["passes"] <= 0:
        problems.append(f"full_steps={b['full_steps']} passes={b['passes']}: nothing measured")
    tot = sum(b.get(f"stage_{s}", 0.0) for s in STAGES)
    for s in STAGES:
        if f"stage_{s}" in b and tot > 0:
            b[f"share_{s}"] = 100.0 * b[f"stage_{s}"] / tot     # % of the profiled pass
    if run.get("level") and b["C"] != run["level"]:
        problems.append(f"BENCH C={b['C']} but the run asked for C={run['level']}")
    return b, problems, b


# --------------------------------------------------------------------------- gates


def identity_gate(loaded, mode):
    """Gate 1. loaded = [(run, load JSON)]. Returns (failures, warnings, n_clips)."""
    fails, warns = [], []
    by_clip = {}   # clip -> {hash: set((arm, run))}
    for run, d in loaded:
        per_run = {}
        for u in d.get("utterances", []):
            if M.outcome(u) != M.OK_OUTCOME:
                continue
            h = text_hash(u.get("text"))
            per_run.setdefault(u.get("clip"), {}).setdefault(h, []).append(u.get("stream"))
            by_clip.setdefault(u.get("clip"), {}).setdefault(h, set()).add((run["arm"], run["run"]))
        for clip, hs in per_run.items():
            if len(hs) > 1:
                fails.append(f"within run {run['run']}: {clip} has {len(hs)} transcripts "
                             f"across streams " + ", ".join(f"{h}@streams{sorted(s, key=str)}"
                                                            for h, s in hs.items()))
    for clip, hs in sorted(by_clip.items(), key=lambda kv: str(kv[0])):
        if len(hs) <= 1:
            continue
        arms_of = {h: sorted({a for a, _ in where}) for h, where in hs.items()}
        # within one arm two hashes -> always a failure
        per_arm = {}
        for h, where in hs.items():
            for a, _ in where:
                per_arm.setdefault(a, set()).add(h)
        split = [a for a, s in per_arm.items() if len(s) > 1]
        desc = f"{clip}: " + "; ".join(f"{h} in arms {arms_of[h]}" for h in hs)
        if split or mode == "all":
            fails.append(("within arm(s) " + ",".join(sorted(split)) + ": " if split else
                          "across arms: ") + desc)
        else:
            warns.append("across arms (allowed by --identity arm): " + desc)
    return fails, warns, len(by_clip)


# --------------------------------------------------------------------------- statistics


def compare(a_vals, x_vals, direction, k, abs_min, rel_min):
    """The verdict of one metric for arm X against baseline A (see the module doc)."""
    a = [v for v in a_vals if v is not None]
    x = [v for v in x_vals if v is not None]
    res = {"a_median": median(a), "x_median": median(x), "aa_spread": None, "xx_spread": None,
           "threshold": None, "effect": None, "effect_pct": None, "pairs": [], "verdict": "n/a"}
    if not a or not x:
        return res
    res["aa_spread"] = max(a) - min(a) if len(a) >= 2 else None
    res["xx_spread"] = max(x) - min(x) if len(x) >= 2 else None
    res["effect"] = res["x_median"] - res["a_median"]
    if res["a_median"]:
        res["effect_pct"] = 100.0 * res["effect"] / abs(res["a_median"])
    pairs = [(av, xv) for av, xv in zip(a_vals, x_vals) if av is not None and xv is not None]
    res["pairs"] = [xv - av for av, xv in pairs]
    if direction == 0:
        res["verdict"] = "info"
        return res
    if len(a) < 2 or len(x) < 2 or len(pairs) < 2:
        res["verdict"] = "NULL"
        res["note"] = "fewer than 2 runs per arm: no noise floor"
        return res
    floor = max(res["aa_spread"], res["xx_spread"])
    thr = max(k * floor, abs_min, rel_min * abs(res["a_median"]))
    res["threshold"] = thr
    gain = direction * res["effect"]            # > 0 = X improves on A
    pair_gain = [direction * dd for dd in res["pairs"]]
    if gain > thr and all(g > thr for g in pair_gain):
        res["verdict"] = "BETTER"
    elif gain < -thr and all(g < -thr for g in pair_gain):
        res["verdict"] = "WORSE"
    else:
        res["verdict"] = "NULL"
    return res


def analyse(d, a):
    ab, runs = load_session(d)
    kind = ab.get("kind", "server")
    order = ab["arms_order"]                    # arms in first-appearance order
    base = order[0]
    levels = sorted({r["level"] for r in runs})
    rep = {"session": d, "kind": kind, "tag": ab.get("tag"), "arms": ab.get("arms", {}),
           "order": ab.get("order"), "levels": levels, "baseline": base,
           "gates": {}, "levels_report": {}, "verdicts": {}, "winner": None}

    # ---- gate 2 data first (it loads the files), but gate 1 is judged first
    loaded, integrity, metrics = [], [], {}
    for r in runs:
        if kind == "device":
            dd, probs, m = device_run(r)
        else:
            dd, probs, m = server_run(r)
        if dd is not None and kind == "server":
            loaded.append((r, dd))
        for pr in probs:
            integrity.append(f"{r['run']}: {pr}")
        metrics[r["run"]] = m

    if kind == "server":
        fails, warns, n_clips = identity_gate(loaded, a.identity)
        rep["gates"]["identity"] = {"pass": not fails, "failures": fails, "warnings": warns,
                                    "clips": n_clips, "mode": a.identity}
    else:
        rep["gates"]["identity"] = {"pass": True, "failures": [], "warnings": [],
                                    "note": "not applicable: the device bench emits no text "
                                            "(its gates A/C are run without --bench)"}
    rep["gates"]["integrity"] = {"pass": not integrity, "problems": integrity}
    gates_ok = rep["gates"]["identity"]["pass"] and rep["gates"]["integrity"]["pass"]
    rep["gates_pass"] = gates_ok
    if not gates_ok and not a.force_latency:
        return rep

    spec = DEVICE_METRICS if kind == "device" else SERVER_METRICS
    has_onset = any(metrics[r["run"]].get("ttfp_on_p95") is not None for r in runs)
    if kind == "device":
        abs_min = a.abs_min_ms if a.abs_min_ms is not None else 0.02
        rel_min = a.rel_min if a.rel_min is not None else 0.01
    else:
        abs_min = a.abs_min_ms if a.abs_min_ms is not None else 5.0
        rel_min = a.rel_min if a.rel_min is not None else 0.02
    primary = {n for n, _, pr in spec if pr}
    if kind == "server" and not has_onset:
        primary.discard("ttfp_on_p95")
        primary.add("ttfp_p95")
    rep["primary"] = sorted(primary)
    rep["thresholds"] = {"k": a.k, "abs_min_ms": abs_min, "rel_min": rel_min}

    level_v = {x: {} for x in order[1:]}
    for lvl in levels:
        runs_at = [r for r in runs if r["level"] == lvl]
        vals = {arm: {n: [metrics[r["run"]].get(n) for r in runs_at if r["arm"] == arm]
                      for n, _, _ in spec} for arm in order}
        meds = {arm: {n: (median([v for v in vals[arm][n] if v is not None])
                          if any(v is not None for v in vals[arm][n]) else None)
                      for n, _, _ in spec} for arm in order}
        cmp_ = {}
        for x in order[1:]:
            cmp_[x] = {}
            for n, direction, _ in spec:
                am = abs_min if not n.startswith("thr") else 0.0
                cmp_[x][n] = compare(vals[base][n], vals[x][n], direction, a.k, am, rel_min)
            vs = [cmp_[x][n]["verdict"] for n in primary if n in cmp_[x]]
            level_v[x][lvl] = ("FAIL" if "WORSE" in vs else "PASS" if "BETTER" in vs else "NULL")
        rep["levels_report"][lvl] = {"runs": [r["run"] for r in runs_at], "values": vals,
                                     "medians": meds, "compare": cmp_,
                                     "verdict": {x: level_v[x][lvl] for x in order[1:]}}
    key = "device_ms_per_pass" if kind == "device" else "lag_p95"
    best, best_gain = base, 0.0
    for x in order[1:]:
        vs = list(level_v[x].values())
        v = "FAIL" if "FAIL" in vs else "PASS" if "PASS" in vs else "NULL"
        rep["verdicts"][x] = v
        gains = [-(rep["levels_report"][lv]["compare"][x][key]["effect_pct"] or 0.0) for lv in levels]
        g = sum(gains) / len(gains) if gains else 0.0
        if v == "PASS" and g > best_gain:
            best, best_gain = x, g
    rep["winner"] = {"arm": best, "label": rep["arms"].get(best, {}).get("label", best),
                     "mean_gain_pct": best_gain, "on": key}
    return rep


# --------------------------------------------------------------------------- printing


def fmt(v, n=""):
    if v is None:
        return "-"
    if n in ("thr_x", "lanes_mean") or n.startswith("share_"):
        return f"{v:.2f}"
    if isinstance(v, float) and abs(v) < 10 and n.startswith(("stage_", "device", "step_wall")):
        return f"{v:.3f}"
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def print_report(rep, out=None):
    w = (out or sys.stdout).write
    arms = rep["arms"]
    lab = lambda x: f"{x}={arms.get(x, {}).get('label', x)}"  # noqa: E731
    w(f"== ab_report {rep['tag']} kind={rep['kind']} order={rep['order']} levels={rep['levels']}\n")
    for x, spec in arms.items():
        w(f"   arm {lab(x)} bin_sha={spec.get('bin_sha', '?')} args=[{spec.get('args', '')}]\n")
    g1, g2 = rep["gates"]["identity"], rep["gates"]["integrity"]
    w(f"-- gate 1 identity: {'PASS' if g1['pass'] else 'FAIL'}"
      + (f" ({g1.get('clips')} clips, mode {g1.get('mode')})" if "clips" in g1 else
         f" ({g1.get('note')})") + "\n")
    for f in g1["failures"][:20]:
        w(f"   FAIL {f}\n")
    for f in g1["warnings"][:20]:
        w(f"   WARN {f}\n")
    if not g1["pass"]:
        w("-- gate 2 integrity: not judged as a gate (gate 1 failed); problems seen:\n"
          if g2["problems"] else "-- gate 2 integrity: no problem seen (gate 1 failed)\n")
    else:
        w(f"-- gate 2 integrity: {'PASS' if g2['pass'] else 'FAIL'}\n")
    for pr in g2["problems"][:30]:
        w(f"   {pr}\n")
    if not rep["gates_pass"] and not rep["levels_report"]:
        w("-- gate 3 latency: NOT EVALUATED (an earlier gate failed). VERDICT: FAIL (gates)\n")
        return
    if not rep["gates_pass"]:
        w("-- gate 3 latency: DIAGNOSTIC ONLY (--force-latency past a failed gate)\n")
    t = rep["thresholds"]
    w(f"-- gate 3 latency: threshold = max({t['k']:g} x max(AA, XX spread), {t['abs_min_ms']:g} ms, "
      f"{100 * t['rel_min']:g}% of A); every pair must agree. primary: {', '.join(rep['primary'])}\n")
    for lvl, lr in rep["levels_report"].items():
        w(f"\n   C={lvl}  runs: {' '.join(lr['runs'])}\n")
        hdr = f"   {'metric':<20}" + "".join(f"{'med ' + x:>11}" for x in lr["medians"]) + \
            f"{'AA':>9}" + "".join(f"{'d ' + x:>10}{'%':>7}{'thr':>8}{'pairs':>16}  verdict"
                                   for x in lr["compare"])
        w(hdr + "\n")
        for n in next(iter(lr["medians"].values())):
            row = f"   {n + ('*' if n in rep['primary'] else ''):<20}" + "".join(
                f"{fmt(lr['medians'][x][n], n):>11}" for x in lr["medians"])
            first = next(iter(lr["compare"].values()), {}).get(n, {})
            row += f"{fmt(first.get('aa_spread'), n):>9}"
            for x, c in lr["compare"].items():
                cc = c[n]
                pr = ",".join(fmt(v, n) for v in cc["pairs"])
                ep = "-" if cc["effect_pct"] is None else f"{cc['effect_pct']:+.1f}"
                row += (f"{fmt(cc['effect'], n):>10}{ep:>7}{fmt(cc['threshold'], n):>8}"
                        f"{pr:>16}  {cc['verdict']}")
            w(row + "\n")
        w("   level verdict: " + "  ".join(f"{lab(x)} {v}" for x, v in lr["verdict"].items()) + "\n")
    if rep["gates_pass"]:
        w("\n== VERDICT vs " + lab(rep["baseline"]) + ": " +
          "  ".join(f"{lab(x)} {v}" for x, v in rep["verdicts"].items()) + "\n")
        wn = rep["winner"]
        w(f"== WINNER ({wn['on']}): {wn['arm']}={wn['label']} (mean gain {wn['mean_gain_pct']:+.1f}%)\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="a session dir (with ab.json) or a dir of sessions")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--identity", choices=("all", "arm"), default="all")
    ap.add_argument("--k", type=float, default=2.0, help="effect > K x noise floor (default 2)")
    ap.add_argument("--abs-min-ms", type=float, default=None)
    ap.add_argument("--rel-min", type=float, default=None)
    ap.add_argument("--force-latency", action="store_true",
                    help="print latency tables past a failed gate, labelled DIAGNOSTIC")
    a = ap.parse_args(argv)
    if os.path.exists(os.path.join(a.out, "ab.json")):
        sessions = [a.out]
    else:
        sessions = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.out, "*", "ab.json")))
    if not sessions:
        print(f"ab_report: no ab.json in or below {a.out}", file=sys.stderr)
        return 1
    reps = []
    for s in sessions:
        rep = analyse(s, a)
        print_report(rep)
        reps.append(rep)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(reps if len(reps) > 1 else reps[0], f, indent=1, default=str)
    return 0 if all(r["gates_pass"] for r in reps) else 2


if __name__ == "__main__":
    sys.exit(main())
