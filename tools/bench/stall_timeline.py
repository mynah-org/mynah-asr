#!/usr/bin/env python3
"""stall_timeline.py — reconstruct what happened to ONE slot, and classify it.

An episode of the 2026-09-20 stall is not visible in an aggregate: during the
600 s soak the median emission lag stayed at 70-95 ms for the whole run while
p95 reached 55 s for three minutes at a time, new streams kept starting, and the
system recovered without a restart. Whatever is wrong is wrong for SOME streams.

So this reads the per-slot [DUMP] lines a server emits on SIGUSR1, finds the
slots whose service stops, and prints their timeline beside a healthy
neighbour's -- because "ring full and not stepping" only means something next to
a slot that is stepping at the same instant.

It then classifies each episode from the evidence, and refuses to classify when
the evidence does not fit one of the shapes:

  A  ring full, ready, scheduler-visible, no model progress
     -> the scheduler can see it, it has work, and it is not being served
  B  ring full, ready, NOT scheduler-visible (out=0)
     -> readiness/bookkeeping: the scheduler is skipping it by its own rule
  C  ring empty, RX stopped
     -> ingest/transport/backpressure. Do NOT blame the scheduler
  D  ring holds data but stays below the next-step requirement
     -> chunk accounting: it never becomes ready
  E  no slot on the worker makes progress
     -> worker/model execution stall, not per-slot fairness
  F  every slot degrades together
     -> capacity or a global execution problem, not selective starvation

Recovery matters as much as onset: the last section prints the sample at which a
stalled slot became schedulable again and what changed in it, because that is
the event that names the mechanism.

    tools/bench/stall_timeline.py server.log [--slot N] [--quiet-healthy]
"""
import argparse, re, sys

SLOT = re.compile(
    r"\[DUMP\] worker=(?P<worker>-?\d+) seq=(?P<seq>\d+) slot id=(?P<id>\d+) "
    r"state=(?P<state>\d+) out=(?P<out>\d+) stream=(?P<stream>\d+) la=(?P<la>-?\d+) "
    r"ready=(?P<ready>\d+) steps=(?P<steps>\d+) deltas=(?P<deltas>\d+) "
    r"ring_s=(?P<ring>[-\d.]+) need_s=(?P<need>[-\d.]+) age_s=(?P<age>[-\d.]+) "
    r"since_rx_s=(?P<rx>[-\d.]+) since_step_s=(?P<step>[-\d.]+) "
    r"lag_max_ms=(?P<lag>[-\d.]+)"
    r"(?: mu_owner=(?P<owner>[0-9a-f]+) mu_held_s=(?P<held>[-\d.]+) "
    r"mu_where=(?P<where>\S+))?")
PROC = re.compile(r"\[DUMP\] worker=(?P<worker>-?\d+) seq=(?P<seq>\d+) process pid=(?P<pid>\d+)")

# A slot that has not been served for this long, while a neighbour has, is the
# thing we are hunting. One chunk period is 320 ms at lookahead 3; two seconds is
# six periods and cannot be jitter.
STALL_S = 2.0


def parse(path):
    samples, pid_of = {}, {}
    for line in open(path, errors="replace"):
        mp = PROC.search(line)
        if mp:
            pid_of[int(mp.group("worker"))] = int(mp.group("pid"))
        m = SLOT.search(line)
        if not m:
            continue
        d = m.groupdict()
        seq = int(d["seq"])
        samples.setdefault((int(d["worker"]), seq), []).append({
            "worker": int(d["worker"]), "seq": seq, "id": int(d["id"]),
            "state": int(d["state"]), "out": int(d["out"]), "stream": int(d["stream"]),
            "la": int(d["la"]), "ready": int(d["ready"]), "steps": int(d["steps"]),
            "deltas": int(d["deltas"]), "ring": float(d["ring"]), "need": float(d["need"]),
            "age": float(d["age"]), "rx": float(d["rx"]), "step": float(d["step"]),
            "lag": float(d["lag"]),
            "owner": d.get("owner") or "0", "held": float(d.get("held") or -1.0),
            "where": d.get("where") or "-"})
    return samples, pid_of


def classify(rows, peers):
    """rows: this slot's samples during the episode. peers: every other slot's
    samples at the same instants. Returns (letter, one-line reason)."""
    worst = max(r["step"] for r in rows)
    ring = max(r["ring"] for r in rows)
    need = max(r["need"] for r in rows)
    ready = any(r["ready"] for r in rows)
    visible = all(r["out"] for r in rows)
    rx_stopped = min(r["rx"] for r in rows) > STALL_S
    peers_moving = any(p["step"] <= 1.0 for p in peers) if peers else None
    peers_stalled = peers and all(p["step"] > STALL_S for p in peers)

    if peers_stalled:
        # E and F are not told apart by how many slots are involved -- that was
        # the first version of this rule and it mislabelled the 2026-09-20
        # episode, in which sixteen slots froze together while one of them held
        # 4.6 s of audio it was ready to have stepped. What separates them is
        # whether RUNNABLE WORK EXISTED while nothing ran. Nothing running with
        # nothing to run is a quiet server; nothing running with a ready slot is
        # an execution stall.
        n_live = len({p["id"] for p in peers}) + 1
        runnable = [p for p in peers if p["ready"]] + [r for r in rows if r["ready"]]
        if runnable:
            most = max(runnable, key=lambda p: p["ring"])
            return "E", (f"all {n_live} live slots stopped together, and work was "
                         f"RUNNABLE while nothing ran: slot {most['id']} held "
                         f"{most['ring']:.1f}s of audio, ready=1, out=1, unserved for "
                         f"{most['step']:.1f}s. Execution stopped; it was not starved of work")
        return "F", (f"all {n_live} live slots stopped together and NONE was ready: "
                     f"no runnable work existed, so this window is not evidence of a "
                     f"scheduling defect")
    if rx_stopped and ring < need:
        return "C", (f"the ring is {ring:.1f}s against {need:.2f}s needed and no audio "
                     f"arrived for {min(r['rx'] for r in rows):.1f}s: the stream stopped "
                     f"being fed, so the scheduler is not the suspect")
    if ready and visible and worst > STALL_S:
        return "A", (f"ready, scheduler-visible (out=1), {ring:.1f}s of audio waiting, "
                     f"and not served for {worst:.1f}s while peers stepped")
    if ready and not visible:
        return "B", (f"ready with {ring:.1f}s waiting but out=0, so the scheduler skips "
                     f"it by its own rule")
    if ring >= 0 and not ready and ring > 0 and worst > STALL_S:
        return "D", (f"the ring holds {ring:.1f}s but never reaches the {need:.2f}s a step "
                     f"wants, so it never becomes ready")
    return "?", (f"does not match a known shape: ready={ready} out={visible} "
                 f"ring={ring:.1f}s need={need:.2f}s since_step={worst:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--slot", type=int, help="only this slot id")
    ap.add_argument("--stall-s", type=float, default=STALL_S)
    a = ap.parse_args()

    samples, pid_of = parse(a.log)
    if not samples:
        print("stall_timeline: no per-slot [DUMP] lines in that log.")
        print("  The server must be built with the per-slot dump and sent SIGUSR1")
        print("  DURING the run, not after it: an episode heals by itself.")
        return 2

    # every slot that was ever un-served for longer than the threshold
    stalled = {}
    for key in sorted(samples):
        for r in samples[key]:
            if r["step"] > a.stall_s and (a.slot is None or r["id"] == a.slot):
                stalled.setdefault((r["worker"], r["id"]), []).append(r)
    if not stalled:
        print(f"stall_timeline: no slot went more than {a.stall_s}s without a model step "
              f"in {len(samples)} sample(s). No episode in this log.")
        return 0

    print(f"stall_timeline: {len(stalled)} slot-episode(s) over {len(samples)} sample(s)\n")
    for (worker, sid), rows in sorted(stalled.items()):
        seqs = {r["seq"] for r in rows}
        peers = [r for key, rs in samples.items() if key[0] == worker
                 for r in rs if r["seq"] in seqs and r["id"] != sid]
        letter, why = classify(rows, peers)
        pid = pid_of.get(worker)
        print(f"  slot {sid} on worker {worker}"
              + (f" (pid {pid})" if pid else "") + f"  ->  CLASS {letter}")
        print(f"    {why}")
        print(f"    {'seq':>5} {'out':>3} {'rdy':>3} {'steps':>6} {'ring_s':>7} "
              f"{'since_rx':>9} {'since_step':>11} {'mu_owner':>10} {'held_s':>7}  who")
        # the whole life of the slot, so onset and recovery are both visible
        life = sorted((r for key, rs in samples.items() if key[0] == worker
                       for r in rs if r["id"] == sid), key=lambda r: r["seq"])
        prev = None
        for r in life:
            mark = ""
            if prev is not None:
                if prev["step"] > a.stall_s and r["step"] <= a.stall_s:
                    mark = "   <== RECOVERED here"
                elif prev["step"] <= a.stall_s and r["step"] > a.stall_s:
                    mark = "   <== onset"
                if r["steps"] > prev["steps"] and prev["step"] > a.stall_s:
                    mark += f" (+{r['steps'] - prev['steps']} steps)"
            own = r["owner"] if r["owner"] != "0" else "-"
            print(f"    {r['seq']:>5} {r['out']:>3} {r['ready']:>3} {r['steps']:>6} "
                  f"{r['ring']:>7.1f} {r['rx']:>9.1f} {r['step']:>11.1f} "
                  f"{own:>10} {r['held']:>7.1f}  {r['where']}{mark}")
            prev = r
        # a healthy neighbour at the same instants: "not stepping" means nothing alone
        if peers:
            worst_seq = max(rows, key=lambda r: r["step"])["seq"]
            same = sorted((p for p in peers if p["seq"] == worst_seq), key=lambda p: p["step"])
            if same:
                p = same[0]
                print(f"    healthiest peer at seq {worst_seq}: slot {p['id']} "
                      f"ring {p['ring']:.1f}s since_step {p['step']:.1f}s steps {p['steps']}")
        print()
    # The wait chain, if the dumps carry a lock owner. "held for N seconds" is
    # not the answer: the answer is why that owner cannot release, and the next
    # edge is whatever IT is waiting on.
    held = [r for key, rs in samples.items() for r in rs
            if r["owner"] != "0" and r["held"] > STALL_S]
    if held:
        print("  LOCK OWNERS HELD LONGER THAN THE THRESHOLD:")
        seen = set()
        for r in sorted(held, key=lambda r: -r["held"]):
            k = (r["id"], r["owner"], r["where"])
            if k in seen: continue
            seen.add(k)
            print(f"    slot {r['id']:<3} mutex held {r['held']:.1f}s by thread "
                  f"{r['owner']} taken in {r['where']}")
        print("    -> next edge: what is THAT thread waiting on? "
              "A held mutex is not a cause until its owner cannot release it.")
    else:
        print("  No slot mutex was held longer than the threshold in these dumps:")
        print("    whatever froze the scheduler, it was not a slot lock held by a peer.")
    print("  CLASS A scheduler/model-service starvation · B readiness bookkeeping ·")
    print("  C ingest/transport · D chunk accounting · E worker-wide · F global capacity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
