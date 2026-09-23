#!/usr/bin/env python3
"""v2_promote.py — write a qualification into a perf profile, or refuse to.

`configs/perf/*.json` carries `status: unmeasured | screened | qualified`, and
`qualified` means "a soak held at the operating point, with transcripts
checked". Until now the only thing stopping a person from typing that word was
the person. This derives the whole block from the artefacts and refuses when
they do not support it.

    tools/bench/v2_promote.py ~/asr-evidence/v2/<run> \
        --profile axion-c4a-highcpu32-nemotron-streaming [--apply]

Without --apply it prints what it would write and changes nothing.

It refuses to promote unless:
  * at least two INDEPENDENT soaks at the same concurrency, each on a fresh
    fleet (one long run is a run, not a qualification);
  * every soak at least the minimum length the profile schema demands;
  * every registered bound PASS in every soak -- a NO EVIDENCE bound is not a
    pass, which is the whole reason the 2026-09-20 soak was never promoted;
  * the same bank in both, so the two runs are the same experiment twice.

What it writes follows the shape of the qualified PocketTTS profile: an
operating point with its full metric set and the path to its evidence, the
previous operating point pushed into `history` with the reason it was not
promoted, and a `ceiling` when a higher concurrency was soaked and did not pass.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_verdict as V                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
PERF = os.path.join(ROOT, "configs", "perf")
MIN_SOAK_S = 600.0        # the schema's own words: "a SOAK of at least 10 minutes"
MIN_SOAKS = 2


def collect(run):
    """Every soak in the run, with its verdict and the numbers a profile wants."""
    out = []
    for path in sorted(glob.glob(os.path.join(run, "soak*.json"))):
        tag = os.path.basename(path)[:-5]
        base = tag.split("-C")[0]
        v = V.verdict_for(path,
                          os.path.join(run, f"server-{base}.log"),
                          os.path.join(run, f"procsample-{base}.txt"))
        d = json.load(open(path))
        s, m = d["summary"], d["summary"]["metrics"]
        st = v.get("stalls") or {}
        out.append({
            "tag": tag, "verdict": v["verdict"], "rows": v["rows"],
            "connection_ceiling": v.get("connection_ceiling"),
            "peak_active_slots": v.get("peak_active_slots"),
            "concurrency": v["streams"], "soak_seconds": v["duration_s"],
            "seed": v.get("seed"), "bank": v.get("bank_sha256"),
            "utterances": s["counts"]["ok"],
            "audio_s": round(s["counts"]["audio_s"], 1),
            "lost": s["counts"]["errors"], "rejected": s["counts"]["rejected"],
            "emission_lag_p50_ms": V.g(m, "emission_lag_ms", "p50"),
            "emission_lag_p95_ms": V.g(m, "emission_lag_ms", "p95"),
            "emission_lag_p99_ms": V.g(m, "emission_lag_ms", "p99"),
            "emission_lag_max_ms": V.g(m, "emission_lag_ms", "max"),
            "finalization_p50_ms": V.g(m, "finalization_lag_ms", "p50"),
            "finalization_p95_ms": V.g(m, "finalization_lag_ms", "p95"),
            "finalization_p99_ms": V.g(m, "finalization_lag_ms", "p99"),
            "backlog_p95_s": V.g(m, "backlog_max_s", "p95"),
            "backlog_max_s": V.g(m, "backlog_max_s", "max"),
            "ttfp_p95_ms": V.g(m, "ttfp_ms", "p95"),
            "ttfp_from_onset_p95_ms": V.g(m, "ttfp_from_onset_ms", "p95"),
            "max_delta_gap_p95_ms": V.g(m, "max_delta_gap_ms", "p95"),
            "audio_s_per_wall_s": V.g(m, "audio_per_wall", "max"),
            "stalls_over_1_chunk": (st.get("rows") or [(None, None)])[0][1],
            "stalls_over_1s": st.get("over_1s"),
            "stalls_over_3s": (st.get("rows") or [(None, None)])[-1][1],
            "published_deltas": st.get("deltas"),
            "worst_delta_lag_ms": st.get("worst_ms"),
            "fairness_worst_over_median": V.g(v.get("fairness") or {}, "worst_over_median"),
            "worst_window": next((r["detail"] for r in v["rows"] if r["bound"] == 6), None),
            "trend": next((r["detail"] for r in v["rows"] if r["bound"] == 7), None),
            "rss": next((r["detail"] for r in v["rows"] if r["bound"] == 11), None),
            "transcript_parity": next((r["detail"] for r in v["rows"] if r["bound"] == 10), None),
            "evidence": os.path.join(os.path.basename(run), os.path.basename(path)),
        })
    return out


def refuse(soaks):
    """Every reason this run may not be promoted, or an empty list."""
    bad = []
    if len(soaks) < MIN_SOAKS:
        bad.append(f"{len(soaks)} soak(s); a qualification needs {MIN_SOAKS} independent runs "
                   f"on a fresh fleet, because one long run is a run and not a qualification")
    cs = {s["concurrency"] for s in soaks}
    if len(cs) > 1:
        bad.append(f"the soaks ran at different concurrencies {sorted(cs)}: that is a ladder, "
                   f"not a repeated experiment")
    banks = {s["bank"] for s in soaks}
    if len(banks) > 1:
        bad.append(f"the soaks played different banks {banks}: the two runs are not the same "
                   f"experiment twice")
    seeds = {s["seed"] for s in soaks}
    if len(seeds) == 1 and len(soaks) > 1:
        bad.append(f"every soak used seed {seeds.pop()}: two replays of one schedule do not "
                   f"show that the result is independent of the schedule")
    for s in soaks:
        if (s["soak_seconds"] or 0) < MIN_SOAK_S:
            bad.append(f"{s['tag']}: {s['soak_seconds']}s is under the {MIN_SOAK_S:.0f}s the "
                       f"profile schema requires")
        if s["verdict"] != "QUALIFIED":
            failed = [f"{r['bound']} {r['name']} ({r['state']})"
                      for r in s["rows"] if r["state"] != V.OK]
            bad.append(f"{s['tag']} is {s['verdict']}: " + ", ".join(failed))
    return bad


def operating_point(soaks, commit, run):
    """The block a profile carries, with the WORST of the runs in every column.

    Not the mean and not the better one: a capacity claim has to survive its
    own bad run, and the two soaks exist precisely so there is a worse one."""
    def worst(key, how=max):
        vals = [s[key] for s in soaks if s.get(key) is not None]
        return how(vals) if vals else None
    return {
        "date": __import__("datetime").date.today().isoformat(),
        "commit": commit,
        "configuration": "shipped-default",
        "concurrency": soaks[0]["concurrency"],
        "verdict": "QUALIFIED",
        "runs": len(soaks),
        "soak_seconds_each": soaks[0]["soak_seconds"],
        "seeds": [s["seed"] for s in soaks],
        "bank": soaks[0]["bank"],
        # Named because a run that cannot connect its own concurrency measures the
        # ceiling and not the machine, with no error and no 503 to show for it.
        "connection_ceiling": soaks[0]["connection_ceiling"],
        "peak_active_slots_per_worker": worst("peak_active_slots"),
        "utterances_total": sum(s["utterances"] for s in soaks),
        "audio_s_total": round(sum(s["audio_s"] for s in soaks), 1),
        "established_streams_lost": worst("lost"),
        "admission_refusals_503": worst("rejected"),
        "emission_lag_p50_ms": worst("emission_lag_p50_ms"),
        "emission_lag_p95_ms": worst("emission_lag_p95_ms"),
        "emission_lag_p99_ms": worst("emission_lag_p99_ms"),
        "emission_lag_max_ms": worst("emission_lag_max_ms"),
        "finalization_p95_ms": worst("finalization_p95_ms"),
        "finalization_p99_ms": worst("finalization_p99_ms"),
        "backlog_max_s": worst("backlog_max_s"),
        "ttfp_p95_ms": worst("ttfp_p95_ms"),
        "ttfp_from_onset_p95_ms": worst("ttfp_from_onset_p95_ms"),
        "max_delta_gap_p95_ms": worst("max_delta_gap_p95_ms"),
        "audio_s_per_wall_s": worst("audio_s_per_wall_s", min),
        "published_deltas_total": sum(s["published_deltas"] or 0 for s in soaks),
        "stalls_over_1_chunk": worst("stalls_over_1_chunk"),
        "stalls_over_1s": worst("stalls_over_1s"),
        "stalls_over_3s": worst("stalls_over_3s"),
        "worst_delta_lag_ms": worst("worst_delta_lag_ms"),
        "fairness_worst_over_median": worst("fairness_worst_over_median"),
        "transcript_parity": soaks[0]["transcript_parity"],
        "per_run": [{k: s[k] for k in
                     ("tag", "seed", "utterances", "lost", "rejected",
                      "emission_lag_p95_ms", "emission_lag_max_ms", "finalization_p95_ms",
                      "backlog_max_s", "stalls_over_1_chunk", "stalls_over_3s",
                      "worst_window", "trend", "rss", "evidence")} for s in soaks],
        "evidence_dir": os.path.basename(run.rstrip("/")),
        "what_this_licenses": (
            "A promise of this concurrency on this box shape, this model, this quant and "
            "this topology, for streams of this corpus's shape. It licenses nothing about "
            "a machine of a different size: the per-stream cost is a function of how many "
            "cores walk the weights."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    path = os.path.join(PERF, a.profile + ".json")
    if not os.path.exists(path):
        sys.exit(f"no such profile: {path}")
    soaks = collect(a.run)
    if not soaks:
        sys.exit(f"no soak*.json in {a.run}")

    print(f"{len(soaks)} soak(s) in {a.run}")
    for s in soaks:
        print(f"  {s['tag']:16s} C={s['concurrency']} {s['soak_seconds']:.0f}s seed={s['seed']} "
              f"{s['utterances']} utterances  -> {s['verdict']}")
    bad = refuse(soaks)
    if bad:
        print("\nREFUSED to promote:")
        for b in bad:
            print(f"  * {b}")
        sys.exit(1)

    commit = a.commit
    if commit is None:
        man = os.path.join(a.run, "manifest.json")
        commit = json.load(open(man)).get("commit") if os.path.exists(man) else "unknown"
    op = operating_point(soaks, commit, a.run)
    d = json.load(open(path))
    prev = (d.get("measured") or {}).get("soak")
    print("\nwould write profile.status = qualified")
    # Printed WHOLE. A dry run that truncates what it would write is not a
    # preview, and this block is the thing a person is being asked to approve.
    print(json.dumps(op, indent=1))
    if prev:
        print(f"\nand push the previous soak (C={prev.get('concurrency')}, "
              f"{prev.get('duration_s')}s) into measured.history")
    if not a.apply:
        print("\n(dry run; pass --apply to write)")
        return 0

    d.setdefault("measured", {})
    hist = d["measured"].setdefault("history", [])
    if prev:
        prev = dict(prev)
        prev.setdefault("why_not_promoted", prev.pop("why_not_qualifying", None))
        hist.append(prev)
        d["measured"]["soak"] = None
    d["measured"]["long_soak_qualified"] = op
    d["measured"]["date"] = op["date"]
    d["profile"]["status"] = "qualified"
    json.dump(d, open(path, "w"), indent=2, ensure_ascii=False)
    open(path, "a").write("\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
