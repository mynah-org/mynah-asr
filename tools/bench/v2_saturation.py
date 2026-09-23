#!/usr/bin/env python3
"""v2_saturation.py — why the machine looks idle, per rung, with numbers.

`htop` showing 25% busy at C=32 has two possible explanations and they are
opposite. Either the fleet is far below its capacity and low utilisation is
correct for a real-time workload, or throughput has stopped growing while the
model stays idle, which is a scheduler or serialisation problem. A latency table
cannot tell them apart. This one can:

    C -> audio/wall -> model duty -> runnable idle -> no work -> chunks/step -> tails

If audio/wall and model duty climb together, low duty at a low rung was simply
a low rung. If audio/wall flattens while model duty stays put, the bottleneck is
not the model and the next question is which lock, thread or queue it is.

    tools/bench/v2_saturation.py <run_dir> [<run_dir> ...]

Everything comes from artefacts the campaign already writes: the run JSON for
throughput and tails, the fleet's SIGUSR1 dumps for duty.

There is NO batch column, and the reason is worth keeping. A first version
derived one as audio_s / steps / chunk_period and it agreed with a live
/v1/health at C=16 -- 0.974 against ready_mean 1.031 -- so it looked sound. Run
across the real ladder it printed 0.98 at every rung from C=32 to C=80, over a
2.1x range of throughput and a fleet going from 56% to 99.6% model duty. A
quantity that constant across that range is not measuring the batch: `steps` in
the dump counts per SLOT, not per batched step, so audio_s/steps is the chunk
period by construction and the division gives 1 whatever the batch does. The
real numbers live in a worker's /v1/health (batched_steps_total, ready_mean,
by_b), which stop_fleet now captures. A column that agrees with the truth at the
one point you checked it, and is constant by construction everywhere else, is
worse than no column.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streaming_metrics import chunk_period_ms                 # noqa: E402


def dump_tail(path):
    """The LAST sample of each worker: duty is cumulative over the run."""
    if not os.path.exists(path):
        return {}
    per = {}
    for line in open(path, errors="replace"):
        if not line.startswith("[DUMP]"):
            continue
        m = re.search(r"worker=(\d+) seq=(\d+)", line)
        if not m:
            continue
        w, seq = int(m.group(1)), int(m.group(2))
        rec = per.setdefault(w, {})
        if seq < rec.get("seq", -1):
            continue
        rec["seq"] = seq
        pk = re.search(r"slots active=(\d+)", line)
        if pk:
            rec["peak"] = max(rec.get("peak", 0), int(pk.group(1)))
        a = re.search(r"slots active=(\d+) cap=(\d+) sessions=(\d+) steps=(\d+) "
                      r"deltas=(\d+) eous=(\d+) audio_s=([\d.]+)", line)
        if a:
            rec.update(steps=int(a.group(4)), sessions=int(a.group(3)),
                       audio_s=float(a.group(7)))
        b = re.search(r"wall uptime_s=([\d.]+) accounted_s=([\d.]+) "
                      r"execution_duty=([\d.]+) model_duty=([\d.]+)", line)
        if b:
            rec.update(uptime=float(b.group(1)), exec_duty=float(b.group(3)),
                       model_duty=float(b.group(4)))
        c = re.search(r"split model_busy_s=([\d.]+) \(([\d.]+)\).*?"
                      r"runnable_idle_s=([\d.]+) \(([\d.]+)\) no_work_s=([\d.]+) \(([\d.]+)\)",
                      line)
        if c:
            rec.update(busy_frac=float(c.group(2)), idle_frac=float(c.group(4)),
                       nowork_frac=float(c.group(6)))
    return per


def cores_used(path):
    """Cores each worker actually used, from cumulative cpu-seconds.

    model_duty is the SCHEDULER thread's fraction of wall spent in the model.
    Multiplying it by the pool width assumes every thread was equally busy for
    that whole fraction, which nothing here measures. This does measure it:
    (cpu_seconds_end - cpu_seconds_start) / (elapsed_end - elapsed_start)."""
    if not os.path.exists(path):
        return None
    first, last = {}, {}
    for line in open(path, errors="replace"):
        f = line.split()
        # pid rss stat etimes cputimes -- cputimes only in samples taken after
        # 2026-09-23, so an older run simply reports nothing here.
        if len(f) < 5 or not f[0].isdigit() or not f[3].isdigit() or not f[4].isdigit():
            continue
        pid, el, cpu = f[0], int(f[3]), int(f[4])
        first.setdefault(pid, (el, cpu))
        last[pid] = (el, cpu)
    out = []
    for pid, (el0, c0) in first.items():
        el1, c1 = last[pid]
        if el1 > el0:
            out.append((c1 - c0) / (el1 - el0))
    return out or None


def rung(path, log):
    d = json.load(open(path))
    s, m, cnt = d["summary"], d["summary"]["metrics"], d["summary"]["counts"]
    man = d.get("manifest") or {}
    cm = man.get("chunk_period_ms") or chunk_period_ms(int(man.get("lookahead", 3)))
    per = dump_tail(log)
    cores = cores_used(log.replace("server-", "procsample-").replace(".log", ".txt"))
    def avg(k):
        v = [r[k] for r in per.values() if k in r]
        return sum(v) / len(v) if v else None
    g = lambda k, q="p95": (m.get(k) or {}).get(q)
    return {
        "C": man.get("concurrency"), "utt": cnt["ok"], "lost": cnt["errors"],
        "rejected": cnt["rejected"], "paced": s["pacing"]["paced"],
        "audio_per_wall": g("audio_per_wall", "max"),
        "model_duty": avg("model_duty"), "busy": avg("busy_frac"),
        "runnable_idle": avg("idle_frac"), "no_work": avg("nowork_frac"),
        "lag95": g("emission_lag_ms"), "lag99": g("emission_lag_ms", "p99"),
        "lagmax": g("emission_lag_ms", "max"),
        "backlog": g("backlog_max_s", "max"), "fin95": g("finalization_lag_ms"),
        "ttfp95": g("ttfp_ms"), "workers": len(per),
        "cores": sum(cores) if cores else None,
        # Streams the fleet actually held. A rung whose peak x workers falls
        # short of C did not run C: it ran what its connection ceiling allowed,
        # and its throughput and duty belong to that number, not to C.
        "peak": max((r.get("peak", 0) for r in per.values()), default=None),
        "held": (max((r.get("peak", 0) for r in per.values()), default=0) * len(per)) or None,
    }


def main(dirs):
    rows = []
    for run in dirs:
        for path in sorted(glob.glob(os.path.join(run, "ladder-C*.json"))
                           + glob.glob(os.path.join(run, "soak*.json"))):
            tag = os.path.basename(path)[:-5]
            base = tag.split("-C")[0]
            log = next((p for p in (os.path.join(run, f"server-{tag}.log"),
                                    os.path.join(run, f"server-{base}.log"))
                        if os.path.exists(p)), os.path.join(run, f"server-{tag}.log"))
            r = rung(path, log)
            r["run"] = os.path.basename(run.rstrip("/"))
            rows.append(r)
    rows.sort(key=lambda r: (r["C"] is None, r["C"]))
    f = lambda v, n=2: "  -  " if v is None else f"{v:.{n}f}"
    print(f"{'C':>4} {'utt':>6} {'lost':>5} {'a/wall':>7} {'duty':>6} {'idle':>6} "
          f"{'nowork':>7} {'lag95':>6} {'lag99':>6} {'lagmax':>7} "
          f"{'backlog':>8} {'fin95':>6} {'cores':>6} {'held':>6} {'paced':>6}")
    for r in rows:
        print(f"{r['C'] if r['C'] is not None else '?':>4} {r['utt']:>6} {r['lost']:>5} "
              f"{f(r['audio_per_wall']):>7} {f(r['model_duty'], 3):>6} "
              f"{f(r['runnable_idle'], 3):>6} {f(r['no_work'], 3):>7} "
              f"{f(r['lag95'], 0):>6} {f(r['lag99'], 0):>6} "
              f"{f(r['lagmax'], 0):>7} {f(r['backlog'], 3):>8} {f(r['fin95'], 0):>6} "
              f"{f(r['cores'], 1):>6} "
              f"{(str(r['held']) + ('*' if r['held'] and r['C'] and r['held'] < r['C'] else '')) if r['held'] else '-':>6} "
              f"{'yes' if r['paced'] else 'NO':>6}")
    short = [r for r in rows if r["held"] and r["C"] and r["held"] < r["C"]]
    if short:
        print("\n  * held < C: the fleet could not hold that concurrency. Its throughput "
              "and duty belong to the number it held, not to C.")
    # The reading the table exists to make, stated rather than left to the eye.
    have = [r for r in rows if r["audio_per_wall"] and r["model_duty"]]
    # Only when the rows span more than one concurrency. Two soaks at the same C
    # are a repeat, and comparing one with the other printed "neither throughput
    # nor duty moved: the bottleneck is upstream of the step" about a pair of
    # runs that agreed to 1%, which is what a repeat is supposed to do.
    if len(have) >= 2 and len({r["C"] for r in have}) >= 2:
        lo, hi = have[0], have[-1]
        dw = hi["audio_per_wall"] / lo["audio_per_wall"]
        dd = hi["model_duty"] / lo["model_duty"] if lo["model_duty"] else None
        print(f"\n  C={lo['C']} -> C={hi['C']}: throughput x{dw:.2f}, model duty x{dd:.2f}"
              if dd else "")
        if dd and dw > 1.2 and dd > 1.2:
            print("  throughput and duty are climbing together: the low rungs were low "
                  "rungs, not a stalled fleet.")
        elif dd and dw < 1.1 and dd < 1.1:
            print("  neither throughput nor duty moved: the fleet is NOT model-bound "
                  "here, and the bottleneck is upstream of the step.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: v2_saturation.py <run_dir> [...]")
    sys.exit(main(sys.argv[1:]))
