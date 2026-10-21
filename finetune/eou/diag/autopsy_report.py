#!/usr/bin/env python3
"""Distributions of eou_autopsy.py events: MLS false / MLS true / non-MLS false / non-MLS true (diagnostic only).

    autopsy_report.py events_dev.jsonl events_final.jsonl [--audit train_pause_audit.json] > report.md

False = premature under the frozen FE rule (>= 0.5 s of speech left). Sections: per-group distributions (p10 / p50 /
p90), mean dB trajectory around the event, the per-pause hazard P(<EOU> | internal pause of length L) for MLS vs
non-MLS (false <EOU> pauses vs pauses that did not fire), the EOU-margin separability (AUC), and two acceptance
rules evaluated as DIAGNOSTICS (nothing is selected or deployed; FINAL-FR is spent and is not a selection set):
  confirmation window W: accept an <EOU> only if no speech within W after it -> a false <EOU> is cancelled iff
    speech resumes within W; every true endpoint is reported W later (its latency + W);
  minimum pause P: accept an <EOU> only once the silence run has lasted P -> a false <EOU> survives iff its whole
    pause is >= P; a true endpoint is reported at max(t, silence onset + P).
Caveat: the runtime has already reset the model at the <EOU>; these rules change only when the event is reported.
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("events", nargs="+")
ap.add_argument("--audit", default="")
a = ap.parse_args()
ev = [json.loads(l) for p in a.events for l in open(p)]
E = [e for e in ev if e["kind"] == "eou"]
NP = [e for e in ev if e["kind"] == "pause_no_eou"]
G = {"MLS false": lambda e: e["set"] == "mls" and e["premature"], "MLS true": lambda e: e["set"] == "mls" and not e["premature"],
     "non-MLS false": lambda e: e["set"] != "mls" and e["premature"], "non-MLS true": lambda e: e["set"] != "mls" and not e["premature"]}


def q(v, qs=(10, 50, 90)):
    v = [x for x in v if x is not None]
    return "/".join(f"{np.percentile(v, x):.2f}" for x in qs) if v else "-"


def full_pause(e):   # the whole silence run the event sits in
    return e["pause_before_s"] + (e["resume_after_s"] - 0.08 if e["resume_after_s"] is not None else 0.0)


for tag in sorted({e["tag"] for e in E}) + ["dev+final"]:
    es = E if tag == "dev+final" else [e for e in E if e["tag"] == tag]
    print(f"\n## {tag}: {len(es)} <EOU> events on {len({e['audio'] for e in es})} clips\n")
    print("| group | events | pause_before s | resume_after s | full pause s | to_end s | speech_after s | rel_pos | clip dur s | "
          "words before | phase=1 | <EOU>/clip | margin at | pause dB re peak |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g, fn in G.items():
        gs = [e for e in es if fn(e)]
        if not gs:
            continue
        print(f"| {g} | {len(gs)} | {q([e['pause_before_s'] for e in gs])} | {q([e['resume_after_s'] for e in gs])} | "
              f"{q([full_pause(e) for e in gs])} | {q([e['to_end_s'] for e in gs])} | {q([e['speech_after_s'] for e in gs])} | "
              f"{q([e['rel_pos'] for e in gs])} | {q([e['dur'] for e in gs])} | {q([e['words_before'] for e in gs])} | "
              f"{100 * np.mean([e['phase'] for e in gs]):.0f} % | {q([e['n_eou_clip'] for e in gs])} | {q([e['margin_at'] for e in gs])} | "
              f"{q([e['pause_db_mean'] for e in gs])} |")
print("\n## mean frame dB re peak around the event (offset s: -0.96 .. +0.96, every 0.16 s), dev+final\n")
offs = list(range(0, 25, 2))
print("| group | " + " | ".join(f"{(o - 12) * 0.08:+.2f}" for o in offs) + " |")
print("|---|" + "---|" * len(offs))
for g, fn in G.items():
    gs = [e for e in E if fn(e)]
    if gs:
        m = [np.mean([e["traj_db"][o] for e in gs if e["traj_db"][o] is not None]) for o in offs]
        print(f"| {g} | " + " | ".join(f"{v:.0f}" for v in m) + " |")
print("\n## hazard: P(<EOU> fired | internal pause of length L), dev+final (false <EOU> pauses vs internal pauses with no <EOU>)\n")
B = [0.24, 0.4, 0.56, 0.8, 1.04, 1.6, 99]
print("| L s | MLS fired / pauses | MLS hazard | non-MLS fired / pauses | non-MLS hazard |")
print("|---|---|---|---|---|")
for lo, hi in zip(B[:-1], B[1:]):
    row = []
    for mls in (True, False):
        f = sum(1 for e in E if e["premature"] and (e["set"] == "mls") == mls and lo <= full_pause(e) < hi)
        n = sum(1 for p in NP if (p["set"] == "mls") == mls and lo <= p["pause_s"] < hi)
        row += [f"{f} / {f + n}", f"{100 * f / (f + n):.1f} %" if f + n else "-"]
    print(f"| [{lo}, {hi if hi < 99 else 'inf'}) | " + " | ".join(row) + " |")
print("\n## internal pauses that did NOT fire, dev+final\n")
for mls in (True, False):
    ps = [p for p in NP if (p["set"] == "mls") == mls]
    print(f"- {'MLS' if mls else 'non-MLS'}: {len(ps)} pauses, length p10/p50/p90 {q([p['pause_s'] for p in ps])} s, "
          f"max EOU margin p50/p90/p99 {q([p['margin_max'] for p in ps], (50, 90, 99))}")


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())) if len(pos) and len(neg) else float("nan")


tr = [e["margin_at"] for e in E if not e["premature"]]
fa = [e["margin_at"] for e in E if e["premature"]]
print(f"\n## EOU margin at the event: AUC(true > false) = {auc(tr, fa):.3f} (true {q(tr)}, false {q(fa)}); "
      f"MLS only {auc([e['margin_at'] for e in E if not e['premature'] and e['set'] == 'mls'], [e['margin_at'] for e in E if e['premature'] and e['set'] == 'mls']):.3f}")
print("\n## acceptance rules (DIAGNOSTIC, not selected): share of false <EOU> events removed vs true-endpoint latency cost, dev+final\n")
T = [e for e in E if not e["premature"]]
F = [e for e in E if e["premature"]]
FM = [e for e in F if e["set"] == "mls"]
lat0 = [e["t"] - e["end"] for e in T]
print(f"baseline: {len(F)} false events ({len(FM)} MLS), {len(T)} true endpoints, true latency p50/p90 {q(lat0, (50, 90))} s\n")
print("| rule | false removed (all) | false removed (MLS) | true latency p50 / p90 s |")
print("|---|---|---|---|")
for W in (0.16, 0.32, 0.48, 0.64, 0.8, 1.04, 1.28):
    rm = np.mean([e["resume_after_s"] < W for e in F]); rmm = np.mean([e["resume_after_s"] < W for e in FM]) if FM else 0
    print(f"| window W = {W:.2f} s | {100 * rm:.1f} % | {100 * rmm:.1f} % | {q([l + W for l in lat0], (50, 90))} |")
for P in (0.32, 0.48, 0.64, 0.8, 1.04):
    rm = np.mean([full_pause(e) < P for e in F]); rmm = np.mean([full_pause(e) < P for e in FM]) if FM else 0
    lat = [max(e["t"], e["t"] - e["pause_before_s"] + P) - e["end"] for e in T]
    print(f"| min pause P = {P:.2f} s | {100 * rm:.1f} % | {100 * rmm:.1f} % | {q(lat, (50, 90))} |")
if a.audit:
    au = json.load(open(a.audit))
    print("\n## stage-2 v4 training mix: internal pauses (implicit negatives), model-free\n")
    print("| source | mix w | clips | internal pauses / clip | pause s p50/p90/p99 | longest >= 0.56 / 0.8 / 1.04 / 1.6 s | pauses >= 0.56 s per h | trail s p10/p50/p90 |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in au.items():
        if k == "MIX_WEIGHTED":
            continue
        c = v["clips_with_longest_internal_pause_ge_s_%"]
        print(f"| {k} | {v['weight']} | {v['clips']} | {v['internal_pauses_per_clip']} | {v['internal_pause_s_p50_p90_p99']} | "
              f"{c['0.56']} / {c['0.8']} / {c['1.04']} / {c['1.6']} % | {v['internal_pauses_ge_0.56s_per_hour']} | {v['trail_s_p10_p50_p90']} |")
    c = au["MIX_WEIGHTED"]["clips_with_longest_internal_pause_ge_s_%"]
    print(f"| **mix-weighted** | | | | | {c['0.56']} / {c['0.8']} / {c['1.04']} / {c['1.6']} % | | |")
