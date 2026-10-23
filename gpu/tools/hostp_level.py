#!/usr/bin/env python3
"""hostp_level.py — the engine thread's host profile for ONE level of a run.

    gpu/tools/hostp_level.py server.log [--from SEQ] [--to SEQ] [--label C64] [--json out]

`mynah-asr-server-cuda --profile-host` prints cumulative `[HOSTP] seq=N ...`
lines with every `[DUMP]` (SIGUSR1, and at shutdown). A level's profile is the
difference between two of them: the dump sent when the warm-up ended (--from)
and the one sent when the measured window ended (--to). Defaults: the first
and the last dump in the log; --from 0 means "since the server started".

Prints one `[HOSTP-LEVEL]` summary line (host / device-wait / cohort-wait /
idle shares of the engine thread's wall, per cycle and per lane) and one line
per phase, server phases first, then the engine's phases inside the step.
DIAGNOSTIC: a run with the profile on is never a latency headline.
"""
import argparse
import json
import re
import sys

PH = re.compile(r"\[HOSTP\] seq=(\d+) (phase|step)[=.](\S+)\s+kind=(\S+)\s+total_ms=\s*([\d.]+)")
SUM = re.compile(r"\[HOSTP\] seq=(\d+) summary wall_ms=([\d.]+) cycles=(\d+) lanes=(\d+) "
                 r"lanes_per_cycle=[\d.]+ chunks=(\d+) passes=(\d+) syncs=(\d+)")
TAIL = re.compile(r"\[HOSTP\] seq=(\d+) tail (.*)$")
RUS = re.compile(r"\[HOSTP\] seq=(\d+) rusage engine_thread utime_s=([\d.]+) stime_s=([\d.]+) .* "
                 r"nvcsw=(\d+) nivcsw=(\d+) minflt=(\d+)")


def parse(path):
    by = {}
    for line in open(path, errors="replace"):
        m = PH.search(line)
        if m:
            seq, kind = int(m.group(1)), m.group(2)
            d = by.setdefault(seq, {"phase": {}, "step": {}, "kind": {}})
            d[kind][m.group(3)] = float(m.group(5))
            d["kind"][(kind, m.group(3))] = m.group(4)
            continue
        m = SUM.search(line)
        if m:
            d = by.setdefault(int(m.group(1)), {"phase": {}, "step": {}, "kind": {}})
            d.update(wall=float(m.group(2)), cycles=int(m.group(3)), lanes=int(m.group(4)),
                     chunks=int(m.group(5)), passes=int(m.group(6)), syncs=int(m.group(7)))
            continue
        m = TAIL.search(line)
        if m:
            by.setdefault(int(m.group(1)), {"phase": {}, "step": {}, "kind": {}})["tail"] = m.group(2)
            continue
        m = RUS.search(line)
        if m:
            by.setdefault(int(m.group(1)), {"phase": {}, "step": {}, "kind": {}})["rusage"] = {
                "cpu_s": float(m.group(2)) + float(m.group(3)), "nvcsw": int(m.group(4)),
                "nivcsw": int(m.group(5)), "minflt": int(m.group(6))}
    return by


def zero_like(d):
    z = {"phase": {k: 0.0 for k in d["phase"]}, "step": {k: 0.0 for k in d["step"]}, "kind": d["kind"],
         "wall": 0.0, "cycles": 0, "lanes": 0, "chunks": 0, "passes": 0, "syncs": 0}
    if "rusage" in d:
        z["rusage"] = {k: 0 for k in d["rusage"]}
    return z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--from", dest="frm", type=int, default=None)
    ap.add_argument("--to", type=int, default=None)
    ap.add_argument("--label", default="level")
    ap.add_argument("--json")
    a = ap.parse_args()
    by = parse(a.log)
    seqs = sorted(s for s, d in by.items() if "wall" in d)
    if not seqs:
        print(f"[HOSTP-LEVEL] {a.label} no [HOSTP] lines (was the server started with --profile-host?)")
        return 1
    to = a.to if a.to is not None else seqs[-1]
    frm = a.frm if a.frm is not None else (seqs[0] if len(seqs) > 1 else 0)
    if to not in by:
        print(f"[HOSTP-LEVEL] {a.label} no dump seq={to} in the log")
        return 1
    B = by[to]
    A = by[frm] if frm in by else zero_like(B)
    dw = lambda k: B[k] - A[k]  # noqa: E731
    wall_ms = dw("wall")
    cyc = max(1, dw("cycles"))
    lanes = max(1, dw("lanes"))
    ph = {k: B["phase"][k] - A["phase"].get(k, 0.0) for k in B["phase"]}
    st = {k: B["step"][k] - A["step"].get(k, 0.0) for k in B["step"]}
    dev = sum(v for k, v in st.items() if k.endswith("_wait"))
    waits = ph.get("idle", 0.0) + ph.get("cohort_wait", 0.0)
    host = wall_ms - waits - dev
    pct = lambda v: 100.0 * v / wall_ms if wall_ms > 0 else 0.0  # noqa: E731
    out = {"label": a.label, "from_seq": frm, "to_seq": to, "wall_ms": round(wall_ms, 1),
           "cycles": dw("cycles"), "lanes": dw("lanes"), "lanes_per_cycle": round(dw("lanes") / cyc, 2),
           "passes": dw("passes"), "syncs": dw("syncs"),
           "host_pct": round(pct(host), 1), "device_wait_pct": round(pct(dev), 1),
           "cohort_wait_pct": round(pct(ph.get("cohort_wait", 0.0)), 1), "idle_pct": round(pct(ph.get("idle", 0.0)), 1),
           "host_us_per_cycle": round(host * 1e3 / cyc), "device_wait_us_per_cycle": round(dev * 1e3 / cyc),
           "host_us_per_lane": round(host * 1e3 / lanes, 1),
           "phases_ms": {k: round(v, 1) for k, v in ph.items()},
           "step_phases_ms": {k: round(v, 1) for k, v in st.items()},
           "tail_cumulative": B.get("tail")}
    if "rusage" in B and "rusage" in A:
        r = {k: B["rusage"][k] - A["rusage"][k] for k in B["rusage"]}
        out["rusage"] = r
        out["engine_thread_cpu_pct"] = round(100.0 * r["cpu_s"] * 1e3 / wall_ms, 1) if wall_ms > 0 else None
    print(f"[HOSTP-LEVEL] {a.label} seq {frm}->{to} wall_s={wall_ms / 1e3:.1f} cycles={out['cycles']} "
          f"lanes/cycle={out['lanes_per_cycle']} passes={out['passes']} syncs/pass="
          f"{(out['syncs'] / out['passes']) if out['passes'] else 0:.1f} | host={out['host_pct']}% "
          f"device_wait={out['device_wait_pct']}% cohort_wait={out['cohort_wait_pct']}% idle={out['idle_pct']}% | "
          f"host_us/cycle={out['host_us_per_cycle']} device_wait_us/cycle={out['device_wait_us_per_cycle']} "
          f"host_us/lane={out['host_us_per_lane']}"
          + (f" | engine_thread_cpu={out['engine_thread_cpu_pct']}%" if out.get("engine_thread_cpu_pct") is not None else ""))
    for k, v in ph.items():
        print(f"    {k:<16} {v:10.1f} ms  {v * 1e3 / cyc:8.0f} us/cycle  {v * 1e3 / lanes:7.1f} us/lane  {pct(v):5.1f} %")
    for k, v in st.items():
        print(f"    step.{k:<11} {v:10.1f} ms  {v * 1e3 / cyc:8.0f} us/cycle  {v * 1e3 / lanes:7.1f} us/lane  {pct(v):5.1f} %"
              + ("  (device wait)" if k.endswith("_wait") else ""))
    if B.get("tail"):
        print(f"    tail (since start) {B['tail']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
