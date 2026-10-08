#!/usr/bin/env python3
"""gpu_sample.py — sample the GPU during a benchmark level and summarise it.

    gpu/tools/gpu_sample.py run --out level.csv [--device 0] [--interval-ms 500]
        exec()s nvidia-smi (so killing this PID stops the sampler) writing one CSV
        row per interval: timestamp, SM utilisation %, power W, temperature C,
        SM clock MHz, the active clock-event (throttle) reasons bitmask, VRAM MiB.
    gpu/tools/gpu_sample.py summary level.csv [--label C64] [--json out.json]
        one `[GPU]` line: SM utilisation (mean over all samples and over busy
        ones), power mean/max, temperature max, SM clock mean/min while busy, VRAM
        max, and the share of busy samples each throttle reason was active.

Why both: a level's latency is only interpretable next to what the GPU was
doing (an L4 runs at its 72 W software power cap, so "the GPU got slower" is
often "the clock dropped"), and the TTS bench showed host-bound levels as SM
utilisation well below 100 % at full load. The sampler is nvidia-smi rather
than NVML bindings so nothing beyond the driver is required.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

# NVML clocks-event reasons (nvmlClocksEventReason*), bit -> name
REASONS = [
    (0x1, "gpu_idle"), (0x2, "app_clocks"), (0x4, "sw_power_cap"), (0x8, "hw_slowdown"),
    (0x10, "sync_boost"), (0x20, "sw_thermal"), (0x40, "hw_thermal"), (0x80, "hw_power_brake"),
    (0x100, "display_clocks"),
]
BUSY_PCT = 5.0


def reason_field():
    """`clocks_event_reasons.active` on current drivers, the older name otherwise."""
    for f in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        r = subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader"],
                           capture_output=True, text=True)
        if r.returncode == 0 and "not a valid field" not in (r.stdout + r.stderr).lower():
            return f
    return None


def cmd_run(a):
    if not shutil.which("nvidia-smi"):
        sys.exit("gpu_sample: nvidia-smi not found")
    rf = reason_field()
    fields = ["timestamp", "utilization.gpu", "power.draw", "temperature.gpu", "clocks.sm"]
    fields += [rf] if rf else []
    fields += ["memory.used"]
    with open(a.out, "w") as f:
        f.write("# " + ",".join(fields) + "\n")
    fd = os.open(a.out, os.O_WRONLY | os.O_APPEND)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.execvp("nvidia-smi", ["nvidia-smi", "-i", str(a.device), "--query-gpu=" + ",".join(fields),
                             "--format=csv,noheader,nounits", "-lms", str(a.interval_ms)])


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def summarise(path, label):
    rows, fields = [], None
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            fields = [f.strip() for f in line[1:].split(",")]
            continue
        rows.append([c.strip() for c in line.split(",")])
    if not fields or not rows:
        return {"label": label, "samples": 0}
    ix = {f: i for i, f in enumerate(fields)}
    rkey = next((f for f in fields if f.endswith("reasons.active")), None)

    def col(name, r):
        i = ix.get(name)
        return None if i is None or i >= len(r) else r[i]

    util = [num(col("utilization.gpu", r)) for r in rows]
    samples = [(u, r) for u, r in zip(util, rows) if u is not None]
    busy = [(u, r) for u, r in samples if u > BUSY_PCT]
    pw = [num(col("power.draw", r)) for _, r in samples]
    pw = [p for p in pw if p is not None]
    temp = [t for t in (num(col("temperature.gpu", r)) for _, r in samples) if t is not None]
    clk = [c for c in (num(col("clocks.sm", r)) for _, r in busy) if c is not None]
    vram = [v for v in (num(col("memory.used", r)) for _, r in samples) if v is not None]
    reasons = {}
    if rkey:
        for _, r in busy:
            try:
                m = int(col(rkey, r), 16)
            except (TypeError, ValueError):
                continue
            for bit, name in REASONS:
                if m & bit:
                    reasons[name] = reasons.get(name, 0) + 1
    us = sorted(u for u, _ in samples)
    out = {
        "label": label, "samples": len(samples), "busy_samples": len(busy),
        "sm_util_mean": round(sum(us) / len(us), 1) if us else None,
        "sm_util_busy_mean": round(sum(u for u, _ in busy) / len(busy), 1) if busy else None,
        "sm_util_p50": us[len(us) // 2] if us else None,
        "power_w_mean": round(sum(pw) / len(pw), 1) if pw else None,
        "power_w_max": max(pw) if pw else None,
        "temp_c_max": max(temp) if temp else None,
        "sm_mhz_busy_mean": round(sum(clk) / len(clk)) if clk else None,
        "sm_mhz_busy_min": min(clk) if clk else None,
        "vram_mib_max": max(vram) if vram else None,
        "vram_mib_min": min(vram) if vram else None,
        "throttle_busy_pct": {k: round(100.0 * v / len(busy), 1) for k, v in sorted(reasons.items())}
        if busy else {},
    }
    return out


def line(s):
    if not s.get("samples"):
        return f"[GPU] {s['label']} no samples"
    f = lambda v, fmt: "n/a" if v is None else fmt % v  # noqa: E731
    thr = ",".join(f"{k}:{v:.0f}%" for k, v in s["throttle_busy_pct"].items()) or "none"
    return (f"[GPU] {s['label']} samples={s['samples']} busy={s['busy_samples']} "
            f"sm_util_mean={f(s['sm_util_mean'], '%.1f')} sm_util_busy_mean={f(s['sm_util_busy_mean'], '%.1f')} "
            f"power_w_mean={f(s['power_w_mean'], '%.1f')} power_w_max={f(s['power_w_max'], '%.1f')} "
            f"temp_c_max={f(s['temp_c_max'], '%.0f')} sm_mhz_busy_mean={f(s['sm_mhz_busy_mean'], '%.0f')} "
            f"sm_mhz_busy_min={f(s['sm_mhz_busy_min'], '%.0f')} vram_mib_max={f(s['vram_mib_max'], '%.0f')} "
            f"throttle_busy={thr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--interval-ms", type=int, default=500)
    s = sub.add_parser("summary")
    s.add_argument("csv")
    s.add_argument("--label", default="level")
    s.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
        return 0
    out = summarise(a.csv, a.label)
    print(line(out))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
