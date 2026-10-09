"""Deterministic, speaker-balanced, NESTED training subsets (pure Python).

Moved unchanged from the Canary kit's data step (canary/prepare_it.py) so the
EOU kit and the tests share one implementation. One ordered sequence is built
once; every subset is a PREFIX of it, so 5 h c 20 h c 40 h by construction.
"""
from __future__ import annotations

import random


def balanced_order(groups, cap_s, seed):
    """groups: {speaker: [utt...]} -> one deterministic sequence of utts.

    Each step takes the next (shuffled) utterance of the speaker with the least
    duration so far, skipping speakers at `cap_s`. Every prefix of the sequence
    is therefore speaker-balanced and capped, and cutting it at 5 / 20 / 40 h
    yields NESTED subsets by construction.
    """
    rng = random.Random(seed)
    spk = sorted(groups)
    rng.shuffle(spk)
    queues = {}
    for s in spk:
        q = list(groups[s])
        rng.shuffle(q)
        queues[s] = q
    acc = {s: 0.0 for s in spk}
    rank = {s: i for i, s in enumerate(spk)}
    seq = []
    live = set(spk)
    while live:
        s = min(live, key=lambda x: (acc[x], rank[x]))
        q = queues[s]
        if not q or acc[s] >= cap_s:
            live.discard(s)
            continue
        u = q.pop()
        acc[s] += u["duration"]
        seq.append(u)
    return seq


def merge_by_share(seqs, shares):
    """Interleave ordered source sequences so each prefix keeps the target
    duration shares (a source that runs out simply stops contributing)."""
    pos = {k: 0 for k in seqs}
    acc = {k: 0.0 for k in seqs}
    out = []
    while True:
        live = [k for k in seqs if pos[k] < len(seqs[k])]
        if not live:
            return out
        k = min(live, key=lambda x: (acc[x] / shares[x], x))
        u = seqs[k][pos[k]]
        pos[k] += 1
        acc[k] += u["duration"]
        out.append(u)


def nested_cuts(seq, hours):
    """{"5h": prefix_len, ...} for the comma list / iterable `hours`: the longest
    prefix of `seq` whose summed duration stays <= the target."""
    cuts, acc, k = {}, 0.0, 0
    targets = sorted(float(h) for h in (hours.split(",") if isinstance(hours, str) else hours))
    for h in targets:
        while k < len(seq) and acc + seq[k]["duration"] <= h * 3600 + 1e-6:
            acc += seq[k]["duration"]
            k += 1
        cuts[f"{h:g}h"] = k  # prefix length -> nested
    return cuts
