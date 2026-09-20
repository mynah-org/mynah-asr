#!/usr/bin/env python3
"""perf_profile.py — turn a serving profile into a command, and refuse when the
world disagrees with it.

A profile (configs/perf/*.json) answers one question: given THIS box and THIS
objective, how should the server be run? It is a contract — topology, settings,
the dispatch rows the binary must resolve, the gates a run must pass, and what
was actually measured with the date beside it.

The reason this exists rather than a line in a runbook: every capacity number in
this repo between 2026-09-18 and 2026-09-20 was taken with the topology typed by
hand. One of those runs measured a server that had exited on a bad flag, another
measured the wrong model because a previous server still held the port, and a
third was scored by a gate that counted lost streams as a warning. None of those
were visible in the numbers they produced. A configuration that lives in shell
history cannot be compared with itself a week later.

    tools/perf_profile.py list
    tools/perf_profile.py validate                       # every profile
    tools/perf_profile.py show <id>
    tools/perf_profile.py command <id> -m MODEL [-p PORT]
    tools/perf_profile.py env <id>                       # what to export, what must be unset
    tools/perf_profile.py check <id> [--dispatch-map FILE | --bin ./mynah-asr-server]

`check` is the preflight: it compares the profile's `dispatch_required` against
what the binary says it resolves, and the profile's environment contract against
this shell. It exits non-zero on any mismatch, because a benchmark is invalid
until dispatch is proven (ENGINEERING.md §5) and a profile that is "mostly"
honoured is a profile nobody can reason about.
"""
import argparse, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF = os.path.join(HERE, "configs", "perf")

def load(pid):
    p = pid if pid.endswith(".json") else os.path.join(PERF, pid + ".json")
    if not os.path.exists(p):
        sys.exit(f"perf_profile: no such profile: {pid}\n"
                 f"  available: {', '.join(sorted(ids())) or '(none)'}")
    with open(p) as f:
        return json.load(f)

def ids():
    return [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(PERF, "*.json"))
            if os.path.basename(p) != "schema.json"]

REQUIRED = {
    "profile": ("id", "description", "status", "objective"),
    "hardware": ("architecture", "cpu_model", "logical_cpus", "physical_cores"),
    "build": ("blas", "simd_expected"),
    "server": ("prefork_workers", "threads_per_worker", "http_threads", "cap", "quant", "lookahead"),
    "gates": ("emission_lag_p95_ms", "finalization_p95_ms", "utterances_lost"),
}
STATUS = ("unmeasured", "screened", "qualified")

def validate_one(pid):
    d, bad = load(pid), []
    for sec, keys in REQUIRED.items():
        if sec not in d:
            bad.append(f"missing section `{sec}`"); continue
        for k in keys:
            if k not in d[sec]:
                bad.append(f"`{sec}.{k}` is missing")
    st = d.get("profile", {}).get("status")
    if st not in STATUS:
        bad.append(f"profile.status is {st!r}, not one of {STATUS}")
    if d.get("profile", {}).get("id") != pid.replace(".json", "").split("/")[-1]:
        bad.append("profile.id does not match the file name")
    # A claim needs evidence: `qualified` without a soak is the exact thing the
    # status ladder exists to prevent.
    m = d.get("measured") or {}
    if st == "qualified" and not m.get("soak"):
        bad.append("status is `qualified` but `measured.soak` is empty: a WAVE may "
                   "disqualify a rung, it never promotes one")
    if st in ("screened", "qualified") and not m.get("date"):
        bad.append(f"status is `{st}` but `measured.date` is missing")
    for var, spec in (d.get("runtime", {}).get("environment") or {}).items():
        if not isinstance(spec, dict) or "why" not in spec:
            bad.append(f"environment.{var} has no `why`: a flag with no reason "
                       "is a flag nobody can remove later")
    return bad

def cmd_validate(a):
    rc = 0
    for pid in sorted(ids()):
        bad = validate_one(pid)
        print(f"  {'FAIL' if bad else 'ok  '}  {pid}")
        for b in bad:
            print(f"          {b}"); rc = 1
    if not ids():
        print("  (no profiles)")
    return rc

def build_command(d, model, port):
    s = d["server"]
    argv = ["./mynah-asr-server", "-m", model, "-p", str(port),
            "--quant", s["quant"], "--threads", str(s["http_threads"]),
            "--cap", str(s["cap"])]
    if int(s.get("prefork_workers", 1)) > 1:
        argv += ["--prefork", str(s["prefork_workers"]),
                 "--prefork-threads", str(s["threads_per_worker"])]
    if int(s.get("batch_window_ms", 0)) > 0:
        argv += ["--batch-window-ms", str(s["batch_window_ms"])]
    if s.get("metrics_port"):
        argv += ["--metrics-port", str(s["metrics_port"])]
    pin = s.get("server_cpus")
    return (["taskset", "-c", str(pin)] if pin else []) + argv

def cmd_command(a):
    d = load(a.profile)
    s = d["server"]
    print(" ".join(build_command(d, a.model, a.port)))
    # The lookahead is NOT a server flag: it is a property of the stream and
    # travels in the request. Saying so here stops it being typed into the
    # server line again -- that mistake cost a whole ladder on 2026-09-19.
    print(f"# lookahead {s['lookahead']} is per-request (?lookahead={s['lookahead']}), not a server flag",
          file=sys.stderr)
    if s.get("reserved_cpus"):
        print(f"# cpus {s['reserved_cpus']} are reserved: pin the load generator there, "
              f"and the capacity this measures is the capacity of cpus {s.get('server_cpus')}",
              file=sys.stderr)
    return 0

def cmd_env(a):
    d = load(a.profile)
    for var, spec in (d.get("runtime", {}).get("environment") or {}).items():
        if spec.get("value") is None:
            print(f"unset {var}    # {spec['why']}")
        else:
            print(f"export {var}={spec['value']}    # {spec['why']}")
    return 0

def parse_dispatch(text):
    """--dispatch-map rows: `feature  compiled  supported  env  resolved  reason`.
    Only the feature and what it RESOLVED to are a contract; the rest is context."""
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 5 and ("." in f[0]) and not line.startswith(" "):
            out[f[0]] = f[4]
    return out

def cmd_check(a):
    d = load(a.profile)
    if a.dispatch_map:
        text = open(a.dispatch_map).read()
    else:
        try:
            text = subprocess.run([a.bin, "--dispatch-map"], capture_output=True,
                                  text=True, timeout=120).stdout
        except Exception as e:
            sys.exit(f"perf_profile: could not ask {a.bin} what it resolves: {e}")
    got, bad = parse_dispatch(text), []
    for feat, want in (d.get("dispatch_required") or {}).items():
        have = got.get(feat)
        if have is None:
            bad.append(f"{feat}: the binary did not report this row at all")
        elif str(have) != str(want):
            bad.append(f"{feat}: profile requires {want}, binary resolves {have}")
    for var, spec in (d.get("runtime", {}).get("environment") or {}).items():
        live = os.environ.get(var)
        if spec.get("value") is None and live is not None:
            bad.append(f"{var} is set to {live!r} and the profile says it must be ABSENT "
                       f"({spec['why']})")
        elif spec.get("value") is not None and live != str(spec["value"]):
            bad.append(f"{var} is {live!r}, the profile requires {spec['value']!r}")
    print(f"profile {d['profile']['id']}  status={d['profile']['status']}")
    for feat, want in (d.get("dispatch_required") or {}).items():
        mark = "ok  " if str(got.get(feat)) == str(want) else "FAIL"
        print(f"  {mark}  {feat:<22} required {want!s:<14} resolved {got.get(feat)}")
    if bad:
        print("\nREFUSED — the world does not match this profile:")
        for b in bad:
            print(f"  * {b}")
        print("\nA benchmark is invalid until dispatch is proven (ENGINEERING.md §5).")
        return 1
    print("\npreflight ok: what this binary resolves matches what the profile requires")
    return 0

def cmd_show(a):
    d = load(a.profile)
    p, s, m = d["profile"], d["server"], (d.get("measured") or {})
    print(f"{p['id']}  [{p['status']}]\n  {p['description']}\n")
    print(f"  topology   {s['prefork_workers']}x{s['threads_per_worker']}  "
          f"cap {s['cap']}  http {s['http_threads']}  {s['quant']}  "
          f"lookahead {s['lookahead']}  window {s.get('batch_window_ms', 0)} ms")
    if s.get("server_cpus"):
        print(f"  cpus       server {s['server_cpus']}, reserved {s.get('reserved_cpus', '-')}")
    if m.get("ceiling_concurrency"):
        print(f"  measured   ceiling c={m['ceiling_concurrency']} on {m.get('date', '?')}"
              f"{'' if m.get('soak') else '  (screened only: no soak has held it)'}")
    for row in m.get("ladder", []):
        print(f"     c={row['c']:<4} lag p95 {row['emission_lag_p95_ms']:>5} ms   "
              f"lost {row['lost']}   {row['verdict']}")
    return 0

def cmd_list(a):
    for pid in sorted(ids()):
        d = load(pid)
        m = d.get("measured") or {}
        c = m.get("ceiling_concurrency")
        print(f"  {pid:<46} {d['profile']['status']:<11} "
              f"{'c=' + str(c) if c else 'no measurement'}")
    return 0

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    s = sub.add_parser("show");  s.add_argument("profile");  s.set_defaults(fn=cmd_show)
    s = sub.add_parser("env");   s.add_argument("profile");  s.set_defaults(fn=cmd_env)
    s = sub.add_parser("command")
    s.add_argument("profile"); s.add_argument("-m", "--model", required=True)
    s.add_argument("-p", "--port", type=int, default=8090); s.set_defaults(fn=cmd_command)
    s = sub.add_parser("check")
    s.add_argument("profile"); s.add_argument("--dispatch-map")
    s.add_argument("--bin", default="./mynah-asr-server"); s.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    return a.fn(a)

if __name__ == "__main__":
    sys.exit(main())
