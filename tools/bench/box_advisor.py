#!/usr/bin/env python3
"""box_advisor — predict this box's streaming capacity and recommend how to serve it.

Three tools, three verbs, and they are deliberately separate:

    box_doctor.sh    DESCRIBES the machine and refuses when it is not fit to measure
    box_advisor.py   PREDICTS capacity and RECOMMENDS a configuration   <- this file
    box_qualify.sh   MEASURES, and only a SOAK promotes

------------------------------------------------------------------ what makes it work

Two ideas are lifted wholesale from the sibling TTS project's doctor, which earns its
reputation from them and not from its arithmetic:

1.  **The calibration is measured with the PRODUCTION path, on the machine that will
    serve.** The sibling has to calibrate a proxy kernel because its frame cost is a
    composition of three stages. This runtime does not: `tests/test_stream_batch` with
    MYNAH_ASR_STEP_TIME=1 steps real streams through the real encoder and decoder at
    B = 1..8 and prints what each step cost. That table IS the serving path, so the two
    coefficients below are a fit of the thing itself rather than a model of it.

2.  **Provenance is a first-class field, and the fallback ladder degrades the LABEL, not
    the number.** Every figure printed here carries one of:

        [MEASURED]      taken on this machine, in this run
        [CALIBRATED]    fitted from this machine's own step table
        [EXTRAPOLATED]  the fit evaluated outside the range that produced it
        [PREDICTED]     computed from config and a transferred constant
        [UNKNOWN]       nothing here supports a number, and it says so

    `predict()` returns None rather than inventing a coefficient. A prediction is where a
    measurement starts, never a claim — and the recommendation says that in its heading.

------------------------------------------------------------------------ the cost model

ASR differs from TTS in the one way that matters here: **the client paces the audio**, so
the deadline is not a throughput target, it is a wall clock. A stream delivers one chunk
every

    P = (lookahead + 1) x encoder_frame_ms          [both from the pack's mynah.json]

and the worker must finish one batched step for all of its streams inside P, with
headroom. The step cost is affine in the number of ready streams:

    T_step(B) = a + b*B

    a   the cost of walking the model once: every weight read, every layer visited,
        whether the worker serves one stream or fifty. Fixed.
    b   what one more stream costs while those weights are already being walked.

so capacity per worker is

    B_max = (rho*P - a) / b                          rho = headroom, default 0.8

and the fleet's is `W * B_max` -- with the sting that **every worker pays `a` again**.
Eight workers spend 8a of every period on fixed cost before serving anyone, which is why
this advisor recommends FEW WIDE workers and why that recommendation is falsifiable: if a
measured sweep finds capacity flat in W, the model is wrong and it should be told so.

Measured on a Neoverse-V2 (32 cores, int8, one process): a = 31.7 ms, b = 4.58 ms.
On an Apple M1 (8 threads, int8, Accelerate): a = 37.8 ms, b = 20.3 ms.

--------------------------------------------------------------------------- the honesty

The step table is measured at B <= 8. Every capacity figure past that is the fit evaluated
outside its own range and is labelled EXTRAPOLATED. On the Axion that is the difference
between a table that ends at 8 and an answer near 49, which is a long way to extrapolate:
the first thing the verification plan asks for is a step table at larger B.

CAL_POINTS below is the trust boundary -- predicted against measured, on every machine
where both exist, INCLUDING the misses. It starts almost empty, and that is the honest
state of this model today.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

# --------------------------------------------------------------------------- constants

RHO_PREFERRED = 0.80   # headroom the envelope wants at the operating point
RHO_HARD = 0.90        # past this a step has no room for jitter and the tail opens up
STEP_TABLE_MAX_B = 8   # what tests/test_stream_batch measures; beyond is extrapolation

# Transferred constants: measured elsewhere, applied here unchanged, and named so a reader
# can see which machine's physics is being borrowed.
TRANSFERRED = {
    "axion": {"a_ms": 31.2, "b_ms": 4.54, "cpus": 32,
              "where": "Neoverse-V2 32c (SMT absent, 32 physical), int8, one 32-thread "
                       "process, calibrated by this tool on an IDLE box (loadavg 0.01, "
                       "2026-09-19). A run earlier the same day on the same machine while "
                       "it was busy gave 31.7 / 4.58 -- under 2 % apart, so these "
                       "constants reproduce"},
    "m1":    {"a_ms": 42.4, "b_ms": 12.00, "cpus": 8,
              "where": "Apple M1 8 threads, int8, Accelerate f32 GEMM, calibrated by this "
                       "tool 2026-09-19 (fit over B=2..8, worst residual 3.6 ms = 4 %). "
                       "The S1-7 note recorded a = 37.8 / b = 20.3 on the same machine the "
                       "day before, under third-party load and best-of-4 interleaved; b "
                       "halving is unexplained and the two were not taken under comparable "
                       "conditions, so neither supersedes the other"},
}

# How far a transferred constant may travel before it stops meaning anything.
#
# `a` and `b` are both functions of how many cores walked the weights. Carrying the
# Axion's 32-core pair onto an 8-core box and printing a capacity from it is the same
# mistake the sibling project made once with bandwidth -- dividing a host roof by a
# worker's core count and calling the answer a worker roof. Their guard is mechanical
# because the mistake is mechanical, and so is this one: past this ratio the advisor
# still shows the arithmetic, and refuses to turn it into a recommendation.
TRANSFER_CPU_RATIO_MAX = 1.5

# The trust boundary: predicted against measured, including where the model lost.
# Rendered under the recommendation so a reader knows how far to trust it HERE before
# renting the next box. Empty entries are the honest state, not an oversight.
CAL_POINTS = [
    # (machine, topology, predicted B_max, measured B_max, note)
    ("Neoverse-V2 32c", "1x32 int8 la3", 49, None,
     "no WAVE run: unverified. a and b reproduced within 2 % across a busy and an idle "
     "run of the same box, so the CALIBRATION is stable; the capacity is not verified"),
    ("Apple M1 8c", "1x8 int8 la3", 18, None,
     "the only server evidence is 8 streams at emission lag p95 258 ms, inside the "
     "320 ms gate (2026-09-18) -- so >= 8, ceiling unknown. 18 is likely optimistic: "
     "the step table carries no ingest, no framing and no writer"),
]


def label(kind: str) -> str:
    return f"[{kind}]"


# ------------------------------------------------------------------------- model facts


def read_pack(model_dir: str) -> dict:
    """Every model number comes from the pack (repo rule 1). Nothing here is a #define."""
    path = os.path.join(model_dir, "mynah.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except OSError as e:
        raise SystemExit(f"box_advisor: cannot read {path}: {e}")

    enc = cfg.get("encoder") or {}
    st = cfg.get("streaming")
    dec = cfg.get("decoder") or {}
    feat = cfg.get("features") or {}

    presets = (st or {}).get("att_context_presets") or []
    lookaheads = [p[1] for p in presets if isinstance(p, list) and len(p) == 2]
    return {
        "name": cfg.get("name") or os.path.basename(model_dir.rstrip("/")),
        "engine": cfg.get("engine", "unknown"),
        "layers": enc.get("n_layers"),
        "d_model": enc.get("d_model"),
        "ffn": enc.get("ffn_dim"),
        "heads": enc.get("n_heads"),
        "vocab": dec.get("vocab_size"),
        "pred_hidden": dec.get("pred_hidden"),
        "decoder_type": dec.get("type"),
        "n_mels": feat.get("n_mels"),
        "streaming": st is not None,
        "lookaheads": lookaheads,
        "default_lookahead": (lookaheads[(st or {}).get("default_preset_index", 0)]
                              if lookaheads else None),
        "encoder_frame_ms": (st or {}).get("encoder_frame_ms"),
    }


def encoder_macs_per_frame(m: dict) -> int | None:
    """Weight MACs for one encoder frame: attention projections, both macaron FFNs, and
    the two pointwise convolutions. The depthwise conv and the norms are not weight
    matmuls and are not counted -- this is the term that scales with the weight stream."""
    L, d, ffn = m["layers"], m["d_model"], m["ffn"]
    if not (L and d and ffn):
        return None
    return L * (4 * d * d + 2 * (d * ffn + ffn * d) + 2 * d * d)


def chunk_period_ms(m: dict, lookahead: int) -> float | None:
    f = m["encoder_frame_ms"]
    return None if not f else (lookahead + 1) * f


# --------------------------------------------------------------------- machine + binary


def run(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def dispatch_facts(binary: str) -> dict:
    """What the BINARY says it will do. A doctor that does not ask the binary is reading a
    spec sheet, and an UNKNOWN row means the prediction has no foundation."""
    rc, out = run([binary, "--dispatch-map"])
    if rc != 0:
        return {"ok": False, "reason": f"{binary} --dispatch-map failed: {out.strip()[:200]}"}
    facts = {"ok": True, "rows": {}, "unknown": [], "blas": None, "simd": None}
    for line in out.splitlines():
        if line.startswith("[DISPATCH-MAP]"):
            for k in ("blas", "simd"):
                mt = re.search(rf"\b{k}=(\S+)", line)
                if mt:
                    facts[k] = mt.group(1)
            continue
        parts = line.split()
        if len(parts) >= 5 and "." in parts[0]:
            feature, resolved = parts[0], parts[4]
            facts["rows"][feature] = resolved
            if resolved.upper() == "UNKNOWN":
                facts["unknown"].append(feature)
    return facts


def online_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))          # the mask, not the host
    except AttributeError:
        return os.cpu_count() or 1


def loadavg() -> float | None:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return None


# --------------------------------------------------------------------- the calibration


STEP_RE = re.compile(r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")


def calibrate(binary_dir: str, model_dir: str, threads: int | None) -> dict:
    """Run the real step table and fit a and b.

    This is the whole idea: the numbers come from the serving path on this machine, not
    from a coefficient carried in from another one.

    B=1 used to be discarded here as "warm-up". That was wrong and it hid a defect for
    days: a ready set of one took the single path, which runs each encoder frame as its
    own [1, d] GEMV instead of the chunk as one [R, d] GEMM, and cost 86.7 ms against
    16.8 (24 Neoverse-V2 cores, 2026-09-20). The row was not warm-up, it was a slow path.
    So B=1 is FITTED now -- and when it still looks like the old shape, that is reported
    as the defect it is rather than smoothed away."""
    exe = os.path.join(binary_dir, "tests", "test_stream_batch")
    if not os.path.exists(exe):
        return {"ok": False, "reason": f"{exe} not built (make tests/test_stream_batch)"}
    env = dict(os.environ, MYNAH_ASR_STEP_TIME="1")
    if threads:
        env["MYNAH_ASR_THREADS"] = str(threads)
    try:
        p = subprocess.run([exe, model_dir], capture_output=True, text=True,
                           timeout=1800, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "reason": f"step table did not run: {e}"}
    rows = []
    for line in p.stdout.splitlines():
        mt = STEP_RE.match(line)
        if mt:
            rows.append((int(mt.group(1)), float(mt.group(3))))   # (B, batched ms)
    # Is B=1 on the slow path? A lone chunk through the stacked GEMM costs a little
    # LESS than a pair, so B=1 above the B=2 row means the solo-group path is off --
    # an old binary, or MYNAH_ASR_STACK_SOLO=0. Naming it is worth more than the fit.
    by_b = dict(rows)
    solo_defect = None
    if 1 in by_b and 2 in by_b and by_b[1] > by_b[2]:
        solo_defect = (by_b[1], by_b[2])
    fit_rows = [(b, ms) for b, ms in rows if b >= (2 if solo_defect else 1)]
    if len(fit_rows) < 3:
        return {"ok": False, "reason": f"step table gave {len(fit_rows)} usable rows "
                                       f"(need 3 at B>=2); output was {len(rows)} rows",
                "rows": rows}
    n = len(fit_rows)
    mx = sum(b for b, _ in fit_rows) / n
    my = sum(v for _, v in fit_rows) / n
    den = sum((b - mx) ** 2 for b, _ in fit_rows)
    b_coef = sum((b - mx) * (v - my) for b, v in fit_rows) / den
    a_coef = my - b_coef * mx
    resid = [v - (a_coef + b_coef * b) for b, v in fit_rows]
    worst = max(abs(r) for r in resid)
    return {"ok": True, "a_ms": a_coef, "b_ms": b_coef, "rows": rows,
            "solo_defect": solo_defect,
            "fit_rows": fit_rows, "worst_resid_ms": worst,
            "resid_pct": 100.0 * worst / my if my else None, "max_b": max(b for b, _ in fit_rows)}


# ------------------------------------------------------------------------- the prediction


def b_max(a_ms: float, b_ms: float, period_ms: float, rho: float) -> float | None:
    """Streams one worker can carry. None when the fixed cost alone blows the period:
    a worker that cannot walk the model inside a chunk cannot serve ONE stream."""
    if b_ms <= 0:
        return None
    room = rho * period_ms - a_ms
    return None if room <= 0 else room / b_ms


def predict_table(model: dict, cal: dict, rho: float) -> list:
    out = []
    for la in sorted(set(model["lookaheads"] or [])):
        P = chunk_period_ms(model, la)
        if P is None:
            continue
        bm = b_max(cal["a_ms"], cal["b_ms"], P, rho)
        kind = ("UNKNOWN" if bm is None else
                "EXTRAPOLATED" if bm > cal.get("max_b", STEP_TABLE_MAX_B) else "CALIBRATED")
        out.append({"lookahead": la, "period_ms": P, "b_max": bm, "label": kind,
                    "default": la == model["default_lookahead"]})
    return out


# How much worse the step is INSIDE THE SERVER than on the bench.
#
# The bench step runs alone. The serving step shares the machine with the ingest
# threads, the writers, the per-delta JSON and the sockets, and its per-stream term is
# the one that pays for that. Measured on the Axion on 2026-09-20 by fitting
# mynah_asr_step_b_wall_ms_sum / mynah_asr_step_b_count -- the step AS THE SCHEDULER
# CALLS IT -- against the bench table taken the same hour:
#
#     bench          b = 4.58 ms/stream
#     serving c=16   b = 8.57 ms/stream      (1.87x)
#     serving c=24   b = 10.13 ms/stream     (2.21x, and still climbing)
#
# So a prediction from the bench alone is optimistic by about two, which is exactly the
# size of the gap that went unexplained all morning (predicted 45, measured 16). One
# machine, one model: it is a correction with provenance, not a constant of nature, and
# it is applied to the SERVER estimate only -- the compute ceiling is still reported raw.
SERVING_B_FACTOR = 1.9
SERVING_B_FACTOR_SOURCE = ("Axion Neoverse-V2, 24 cpus, Nemotron int8, 2026-09-20: "
                           "serving b 8.57 at c=16 and 10.13 at c=24 against a bench b of 4.58")


def topologies(cpus: int) -> list:
    """Candidate W x T, WIDE FIRST -- and now with a measurement behind the ordering.

    The cost model says every worker re-pays `a`, so capacity is W*(rho*P - a)/b, which
    grows as W shrinks. The 2026-09-20 sweep on a 24-cpu Neoverse-V2 agrees, but only
    once a lost stream counts as a failure: as first scored, 8x3 looked twice as good as
    1x24 while quietly dropping one to four streams a rung. Re-scored, 1x24 held c=16
    with zero losses and the narrow shapes held c=8.

    So wide first, and the sweep still decides -- but it now starts from a result rather
    than from the arithmetic alone."""
    out = []
    for w in (1, 2, 4, 8, 16):
        if w > cpus or cpus % w:
            continue
        t = cpus // w
        if t < 2 and w > 1:
            continue
        out.append((w, t))
    return out


# ------------------------------------------------------------------------------ report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Predict streaming capacity and recommend a serving configuration.")
    ap.add_argument("-m", "--model", required=True, help="converted model directory")
    ap.add_argument("--bin-dir", default=".", help="tree holding mynah-asr and tests/")
    ap.add_argument("--rho", type=float, default=RHO_PREFERRED,
                    help=f"headroom at the operating point (default {RHO_PREFERRED})")
    ap.add_argument("--threads", type=int, default=None, help="calibrate at this pool width")
    ap.add_argument("--latency-budget-ms", type=float, default=None,
                    help="largest acceptable chunk period; picks the lookahead")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip the step table and use a TRANSFERRED constant (weaker)")
    ap.add_argument("--transfer", choices=sorted(TRANSFERRED), default="axion")
    ap.add_argument("--allow-load", action="store_true",
                    help="calibrate even on a busy box; the numbers are then not comparable")
    ap.add_argument("--json", help="write the whole prediction here")
    a = ap.parse_args()

    binary = os.path.join(a.bin_dir, "mynah-asr")
    if not os.path.exists(binary):
        print(f"box_advisor: {binary} not found (build it first)", file=sys.stderr)
        return 2

    print("box_advisor — a PREDICTION and a starting point, never a result.")
    print(f"  {label('MEASURED')} read here, now · {label('CALIBRATED')} fitted from this "
          f"machine's step table · {label('EXTRAPOLATED')} the fit outside its own range")
    print(f"  {label('PREDICTED')} config + a transferred constant · "
          f"{label('UNKNOWN')} nothing supports a number\n")

    model = read_pack(a.model)
    if not model["streaming"]:
        print(f"1. MODEL   {model['name']} ({model['engine']}) — NOT a streaming model:")
        print("   the pack carries no cache-aware presets, so there is no chunk period to")
        print("   serve against. It belongs in an offline group; see .work/multi-model-streaming.md")
        return 1
    macs = encoder_macs_per_frame(model)
    print(f"1. MODEL   {model['name']}  {label('MEASURED')} (from its mynah.json)")
    print(f"   {model['layers']} layers x d_model {model['d_model']} x ffn {model['ffn']}, "
          f"{model['n_mels']} mel, vocab {model['vocab']}, {model['decoder_type']}")
    print(f"   encoder frame {model['encoder_frame_ms']:.0f} ms · lookaheads "
          f"{model['lookaheads']} (default {model['default_lookahead']})")
    if macs:
        print(f"   encoder weight MACs per frame: {macs/1e6:.1f} M")

    disp = dispatch_facts(binary)
    print(f"\n2. BINARY  {label('MEASURED')} (--dispatch-map)")
    if not disp["ok"]:
        print(f"   UNKNOWN — {disp['reason']}")
        print("   A prediction built on a binary that will not say what it runs is a guess.")
        return 1
    print(f"   blas={disp['blas']} simd={disp['simd']}")
    for k in ("kernel.int8_rows", "kernel.int8_dot", "gemm.f32", "gemm.f32_kernel"):
        if k in disp["rows"]:
            print(f"   {k:<20} {disp['rows'][k]}")
    if disp["unknown"]:
        print(f"   WARNING {len(disp['unknown'])} row(s) UNKNOWN: {', '.join(disp['unknown'])}")
        print("           a capacity number attributed to a kernel nobody resolved is fiction.")

    cpus = online_cpus()
    la = loadavg()
    print(f"\n3. MACHINE {label('MEASURED')}")
    print(f"   cpus in this process's affinity mask: {cpus}"
          + (f" · loadavg(1m) {la:.2f}" if la is not None else " · loadavg unavailable"))
    busy = la is not None and la >= 2.0
    if busy and not a.allow_load and not a.no_calibrate:
        print("   REFUSED: another load owns this box, so a step table taken now describes")
        print("            this machine plus whatever else is on it. Find the load, or pass")
        print("            --allow-load and treat every number below as NON-QUALIFYING.")
        return 3

    print(f"\n4. CALIBRATION")
    transfer_too_far = False
    if a.no_calibrate:
        t = TRANSFERRED[a.transfer]
        cal = {"ok": True, "a_ms": t["a_ms"], "b_ms": t["b_ms"], "max_b": STEP_TABLE_MAX_B,
               "transferred": a.transfer, "transfer_cpus": t["cpus"]}
        print(f"   {label('PREDICTED')} transferred from {a.transfer}: a = {cal['a_ms']:.1f} ms, "
              f"b = {cal['b_ms']:.2f} ms/stream")
        print(f"   measured on {t['where']}")
        ratio = max(t["cpus"], cpus) / float(min(t["cpus"], cpus))
        if ratio > TRANSFER_CPU_RATIO_MAX:
            transfer_too_far = True
            print(f"   WARNING that constant was measured on {t['cpus']} cpus and this machine "
                  f"has {cpus}.")
            print(f"           Both a and b are functions of how many cores walk the weights, so "
                  f"carrying")
            print(f"           the pair across a {ratio:.1f}x difference does not describe either "
                  f"machine.")
            print(f"           The arithmetic below is shown; the RECOMMENDATION is refused.")
        else:
            print("   This is the weakest input here. Drop --no-calibrate to fit this machine.")
    else:
        print("   running the real step table (tests/test_stream_batch, "
              "MYNAH_ASR_STEP_TIME=1) — this is the SERVING path, not a proxy...")
        cal = calibrate(a.bin_dir, a.model, a.threads)
        if not cal["ok"]:
            print(f"   UNKNOWN — {cal['reason']}")
            print("   No calibration, no prediction. Re-run with --no-calibrate to see what a")
            print("   transferred constant would say, and label everything PREDICTED.")
            return 1
        print(f"   {label('CALIBRATED')} a = {cal['a_ms']:.1f} ms fixed, "
              f"b = {cal['b_ms']:.2f} ms per stream   (fit over B=2..{cal['max_b']}, "
              f"B=1 discarded: warm-up)")
        print(f"   worst residual {cal['worst_resid_ms']:.1f} ms"
              + (f" ({cal['resid_pct']:.0f} % of the mean step)" if cal.get("resid_pct") else ""))
        if cal.get("resid_pct") and cal["resid_pct"] > 10:
            print("   WARNING residuals over 10 %: the line is a poor fit here, which usually")
            print("           means the box was not quiet. Treat the capacity as a rough size.")

    print(f"\n5. CAPACITY per worker, and what it costs in latency")
    print(f"   B_max = (rho*P - a) / b   with rho = {a.rho}")
    rows = predict_table(model, cal, a.rho)
    print(f"   {'lookahead':>9} {'period':>8} {'B_max/worker':>13}  label")
    for r in rows:
        v = "cannot serve 1" if r["b_max"] is None else f"{r['b_max']:.0f}"
        star = " <- pack default" if r["default"] else ""
        print(f"   {r['lookahead']:>9} {r['period_ms']:>6.0f} ms {v:>13}  "
              f"{label(r['label'])}{star}")
    print(f"   Anything above B={cal.get('max_b', STEP_TABLE_MAX_B)} is the fit evaluated "
          f"outside the range that produced it.")
    print("   Lookahead is the largest lever here and it is a PRODUCT decision: it buys")
    print("   capacity with latency the speaker feels, not with hardware.")

    print(f"\n6. RECOMMENDATION  (a starting point, not a result)")
    if transfer_too_far:
        print(f"   REFUSED — the only cost model available came from a machine of a different")
        print(f"             size (see 4). A configuration derived from it would be a number")
        print(f"             about another box wearing this one's name.")
        print(f"   Do this instead:  python3 tools/bench/box_advisor.py -m {a.model}")
        print(f"                     (drop --no-calibrate; the step table takes ~1 minute)")
        return 1
    chosen = None
    if a.latency_budget_ms:
        fits = [r for r in rows if r["period_ms"] <= a.latency_budget_ms and r["b_max"]]
        chosen = max(fits, key=lambda r: r["b_max"]) if fits else None
        if chosen is None:
            print(f"   No lookahead meets a {a.latency_budget_ms:.0f} ms budget on this box.")
    if chosen is None:
        chosen = next((r for r in rows if r["default"] and r["b_max"]), None) \
                 or next((r for r in rows if r["b_max"]), None)
    if chosen is None:
        print("   UNKNOWN — no lookahead lets this machine carry a single stream.")
        return 1

    cands = topologies(cpus)
    w, t = cands[0] if cands else (1, cpus)
    fleet = chosen["b_max"] * w
    cap = max(1, int(chosen["b_max"] * 0.9))
    if w == 1:
        print(f"   NOTE      the cost model optimises CAPACITY, not availability. It picks")
        print(f"             W=1 because one worker pays `a` once -- but one worker is also")
        print(f"             one crash away from losing every stream, and it cannot be")
        print(f"             restarted under load. If the box serves anything that matters,")
        print(f"             run the W sweep in step (c) and take 2xN if it costs little.")
    print(f"   topology  {w}x{t}  (W workers x T threads)  — WIDE FIRST, and this is the")
    print(f"             model's own prediction: every worker re-pays a = {cal['a_ms']:.1f} ms,")
    print(f"             so {w} worker{'' if w == 1 else 's'} spend{'s' if w == 1 else ''} "
          f"{w*cal['a_ms']:.0f} ms of every {chosen['period_ms']:.0f} ms period on fixed cost.")
    print(f"   lookahead {chosen['lookahead']}  (chunk period {chosen['period_ms']:.0f} ms)")
    print(f"   quant     int8 — the serving default; on CPU it is 2.24x faster than f32")
    print(f"             for a documented mean CER 0.133 -> 0.145 (docs/quantization.md)")
    print(f"   --cap     {cap}  (90 % of the per-worker prediction, so the ladder refuses")
    print(f"             before the cadence does)")
    print(f"   fleet     ~{fleet:.0f} concurrent real-time streams  {label('EXTRAPOLATED')}")
    print()
    print("   $ " + " ".join([
        "./mynah-asr-server", "-m", a.model, "-p", "8090",
        "--quant", "int8", "--lookahead", str(chosen["lookahead"]),
        "--prefork", str(w), "--threads", str(t), "--cap", str(cap),
        "--metrics-port", "9109",
    ]))

    print(f"\n7. VERIFY NEXT  (each step replaces a label with a stronger one)")
    print(f"   a. extend the step table past B={cal.get('max_b', STEP_TABLE_MAX_B)}: every")
    print(f"      capacity figure above is an extrapolation until it reaches {cap}.")
    print(f"   b. WAVE at c = {max(1,int(fleet*0.5))}, {int(fleet*0.8)}, {int(fleet)} and find")
    print(f"      where emission lag p95 crosses one chunk period.")
    print(f"   c. sweep W: run {' '.join(f'{ww}x{tt}' for ww, tt in cands[:4])} at equal")
    print(f"      concurrency. If capacity is FLAT in W, the fixed-cost model is wrong and")
    print(f"      this recommendation changes -- that is the point of writing it down.")
    print(f"   d. SOAK 10-30 min at the concurrency (b) found, with --transcripts, so a run")
    print(f"      that held its cadence while the transcripts drifted comes out DEGRADED.")

    print(f"\n8. TRUST BOUNDARY  predicted vs measured, including the misses")
    for machine, topo, pred, meas, note in CAL_POINTS:
        got = "not measured" if meas is None else f"{meas}"
        print(f"   {machine:<18} {topo:<16} predicted {pred:<5} measured {got:<13} {note}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": model, "dispatch": disp, "cpus": cpus, "loadavg": la,
                       "calibration": cal, "rho": a.rho, "capacity": rows,
                       "recommendation": {"workers": w, "threads": t, "cap": cap,
                                          "lookahead": chosen["lookahead"],
                                          "fleet_b_max": fleet},
                       "cal_points": CAL_POINTS}, f, indent=1)
        print(f"\n   -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
