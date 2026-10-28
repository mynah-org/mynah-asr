#!/usr/bin/env python3
"""recovery_lab.py — items for, and the report of, the empty-turn recovery
mechanism experiment (S18-1c, .work/empty-turn-recovery.md).

    python3 tools/eval/recovery_lab.py items --manifest samples/eval-bank/manifest.json \
        --per-lang 30 --out items.tsv
    tests/recovery_lab MODEL_DIR items.tsv --quant int8 > lab.jsonl
    python3 tools/eval/recovery_lab.py report lab.jsonl --manifest samples/eval-bank/manifest.json

ITEMS.  A current turn is the first D seconds of speech of a FLEURS clip
(D = 0.4 / 0.7 / 1.0 s from an energy onset; short turns are where a streaming
transducer ends empty) or the whole clip (the no-harm control). Its previous
turn is another clip of the same language. A small "provoked" set tags French
audio as en-US: a wrong session language is the other known cause of empties.

SCORING.  The reference of a prefix is not known word for word, so a recovered
text is judged against the START of the clip's reference: a recovered word is
"supported" when it occurs among the first (n_recovered + 2) reference words.
correct = at least half supported · wrong = otherwise. Leakage = the first kept
word is one of the context transcript's last four words and not one of the
reference's first three. Whole clips are scored with CER.

Exit: 0 · 2 usage.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import wave
from array import array

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from streaming_metrics import cer, normalise  # noqa: E402

DURATIONS = (0.4, 0.7, 1.0)


def speech_onset(path: str) -> float:
    """First 20 ms frame whose RMS exceeds 10 % of the clip's loudest frame."""
    with wave.open(path, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        x = array("h", w.readframes(n))
    if w.getnchannels() != 1 or w.getsampwidth() != 2:
        return 0.0
    hop = sr // 50
    rms = []
    for i in range(0, len(x) - hop, hop):
        seg = x[i:i + hop]
        rms.append((sum(v * v for v in seg) / hop) ** 0.5)
    if not rms:
        return 0.0
    thr = 0.1 * max(rms)
    for i, r in enumerate(rms):
        if r > thr:
            return max(0.0, i * 0.02 - 0.1)
    return 0.0


def load_manifest(path: str):
    root = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        m = json.load(f)
    rows = m["samples"] if isinstance(m, dict) else m
    for r in rows:
        r["path"] = os.path.join(root, r["file"])
    return rows


def cmd_items(a):
    rows = load_manifest(a.manifest)
    rng = random.Random(a.seed)
    tag = {"en": "en-US", "fr": "fr-FR", "en-US": "en-US", "fr-FR": "fr-FR"}
    # --contexts: the previous turns are these clips (e.g. questions), in order
    fixed = {}
    if a.contexts:
        with open(a.contexts) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    fixed.setdefault(p[0], []).append(os.path.join(os.path.dirname(
                        os.path.abspath(a.manifest)), p[1]))
    out = []
    for lang in sorted({r["lang"] for r in rows}):
        pool = [r for r in rows if r["lang"] == lang]
        rng.shuffle(pool)
        if lang in fixed:
            ctx_paths = fixed[lang][:a.per_lang]
            pool = [r for r in pool if r["path"] not in set(fixed[lang])]
        else:
            ctx_paths = [r["path"] for r in pool[a.per_lang:2 * a.per_lang]]
        cur = pool[:len(ctx_paths)]
        short = lang.split("-")[0]
        for i, (c, x) in enumerate(zip(cur, ctx_paths)):
            on = speech_onset(c["path"])
            for d in DURATIONS:
                out.append((f"{short}-{i:02d}-d{int(d * 1000)}", tag[lang], x, c["path"],
                            on, on + d))
            out.append((f"{short}-{i:02d}-full", tag[lang], x, c["path"], 0.0, 0.0))
            if a.silence and on >= 0.3:
                # control: the current turn is only the lead-in before speech
                out.append((f"{short}-{i:02d}-sil", tag[lang], x, c["path"], 0.0, min(on, 0.6)))
            if short == "fr" and i < a.provoked:
                out.append((f"fr-{i:02d}-full-as-en", "en-US", x, c["path"], 0.0, 0.0))
    out.sort(key=lambda t: (t[2], t[1]))          # one context decode per (clip, language)
    with open(a.out, "w") as f:
        for t in out:
            f.write("\t".join([t[0], t[1], t[2], t[3], f"{t[4]:.3f}", f"{t[5]:.3f}"]) + "\n")
    print(f"{len(out)} items -> {a.out}")


def words(s: str):
    return normalise(s or "").split()


def judge(text: str, ref: str, ctx_text: str):
    """(verdict, leak) for a recovered text; verdict in empty/correct/wrong."""
    w = words(text)
    if not w:
        return "empty", False
    r = words(ref)
    window = set(r[: len(w) + 2])
    supported = sum(1 for x in w if x in window)
    leak = w[0] in set(words(ctx_text)[-4:]) and w[0] not in set(r[:3])
    return ("correct" if 2 * supported >= len(w) else "wrong"), leak


def recall(ref: str, hyp: str):
    """Fraction of the reference's words (as a multiset) found in hyp."""
    from collections import Counter
    r, h = Counter(words(ref)), Counter(words(hyp))
    n = sum(r.values())
    return sum(min(c, h[w]) for w, c in r.items()) / n if n else None


def arm_of(rp):
    """A replay's context arm: its length in ms, or "all" for the whole
    previous clip (requested as -1; its real length varies per clip)."""
    return rp["ctx_ms"] if rp["ctx_ms"] <= 3000 else "all"


def arm_key(x):
    return (1, 0) if x == "all" else (0, x)


def cmd_report(a):
    refs = {r["path"]: r["text"] for r in load_manifest(a.manifest)}
    cur_of = {}
    with open(a.items) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                cur_of[p[0]] = p[3]
    items = [json.loads(x) for x in open(a.lab) if x.strip()]
    for it in items:
        it["ref"] = refs.get(cur_of.get(it["id"], ""), "")
        i = it["id"]
        it["cond"] = ("provoked" if i.endswith("as-en") else "full" if i.endswith("full")
                      else "silence" if i.endswith("sil") else i.split("-")[-1])
        it["lng"] = i.split("-")[0]
        it["q"] = (it.get("ctx_text") or "").rstrip().endswith("?")
    rules = ("frame", "word", "text")
    arms = sorted({arm_of(rp) for it in items for rp in it.get("replay", [])}, key=arm_key)

    print(f"items {len(items)} | contexts ending with '?' (as decoded alone): "
          f"{sum(it['q'] for it in items)}")
    # ---- 1. recovery on primary-EMPTY turns, per condition and language
    print("\n== 1. primary EMPTY -> retry(auto) -> replay (verdict counts; denominators = EMPTY turns)")
    groups = sorted({(it["cond"], it["lng"]) for it in items})
    for g in groups:
        sel = [it for it in items if (it["cond"], it["lng"]) == g]
        emp = [it for it in sel if it.get("empty")]
        print(f"-- {g[1]} {g[0]}: {len(sel)} turns, primary EMPTY {len(emp)}")
        if not emp:
            continue
        rv = {}
        for it in emp:
            if "retry" in it:
                v = judge(it["retry"], it["ref"], it["ctx_text"])[0]
                v = v if it.get("retry_gate") == "ok" else "rej:" + it.get("retry_gate", "?")
                rv[v] = rv.get(v, 0) + 1
        print(f"   retry auto          : {dict(sorted(rv.items()))}")
        for arm in arms:
            for r in rules:
                c = {}
                for it in emp:
                    rp = next((x for x in it["replay"] if arm_of(x) == arm), None)
                    if rp is None:
                        continue
                    v, lk = judge(rp[r], it["ref"], it["ctx_text"])
                    if g[0] == "silence" and rp[r + "_gate"] == "ok":
                        v = "FALSE-RECOVERY"
                    elif rp[r + "_gate"] != "ok":
                        v = "rej:" + rp[r + "_gate"]
                    c[v] = c.get(v, 0) + 1
                    if lk and rp[r + "_gate"] == "ok":
                        c["leak"] = c.get("leak", 0) + 1
                print(f"   replay {str(arm):>4} {r:<5}  : {dict(sorted(c.items()))}")

    # ---- 2. ownership on every replay
    print("\n== 2. ownership on every replay (denominator = replays, all items)")
    for arm in arms:
        c = dict(n=0, q=0, punct=0, qmark=0, piece=0, text_words=0)
        for it in items:
            for rp in it.get("replay", []):
                if arm_of(rp) != arm:
                    continue
                c["n"] += 1
                c["q"] += int(it["q"])
                for lo, hi, name in (("frame", "word", "frame"), ("word", "text", "text")):
                    d = rp[lo][: len(rp[lo]) - len(rp[hi])].strip() if rp[lo].endswith(rp[hi]) else rp[lo]
                    if not d:
                        continue
                    if name == "text":
                        c["text_words"] += 1
                    elif words(d):
                        c["piece"] += 1
                    else:
                        c["punct"] += 1
                        c["qmark"] += int("?" in d)
        print(f"   ctx {str(arm):>4}: replays {c['n']} (question ctx {c['q']}) | FRAME handed the "
              f"current turn: context punctuation {c['punct']} (with '?' {c['qmark']}), a context "
              f"word/piece {c['piece']} | TEXT gave back whole words {c['text_words']}")

    # ---- 3. healthy turns: first REFERENCE word lost by a rule
    print("\n== 3. healthy turns (primary has text): reference first word kept by the primary "
          "but lost by the replay rule")
    for arm in arms:
        line = []
        for r in rules:
            n = lost = 0
            for it in items:
                if it.get("empty") or it["cond"] == "silence":
                    continue
                ref = words(it["ref"])
                prim = words(it["primary"])
                rp = next((x for x in it["replay"] if arm_of(x) == arm), None)
                if not ref or rp is None or ref[0] not in prim[:2]:
                    continue
                n += 1
                lost += int(ref[0] not in words(rp[r])[:2])
            line.append(f"{r} {lost}/{n}")
        print(f"   ctx {str(arm):>4}: " + " | ".join(line))

    # ---- 4. silence controls: anything a replay would publish
    sil = [it for it in items if it["cond"] == "silence"]
    if sil:
        print(f"\n== 4. silence/lead-in controls ({len(sil)}; primary non-empty "
              f"{sum(1 for it in sil if not it.get('empty'))}): replay text accepted by the gate")
        for arm in arms:
            line = []
            for r in rules:
                k = sum(1 for it in sil for rp in it["replay"]
                        if arm_of(rp) == arm and rp[r + "_gate"] == "ok")
                line.append(f"{r} {k}/{len(sil)}")
            print(f"   ctx {str(arm):>4}: " + " | ".join(line))

    # ---- 6. does the replay preserve the current turn at all?
    # Two failure modes: ATTRIBUTION (the replay decoded the current turn, the
    # rule cut it wrongly: FRAME recall high, WORD lower) and SUPPRESSION (the
    # replay did not decode the current turn: FRAME recall itself collapses).
    # For a prefix item only the reference's first words were spoken: the
    # reference is cut to the primary's word count + 1.
    print("\n== 6. current-turn preservation on healthy turns (silence/provoked excluded): recall of "
          "reference words; delta = FRAME - primary (p10/median/p90); SUPPRESSED = FRAME < 0.2 "
          "with primary >= 0.5; attribution = WORD < FRAME - 0.1")
    for cond in ("full", "prefix"):
        for lng in sorted({it["lng"] for it in items}):
            sel = [it for it in items if not it.get("empty") and it["lng"] == lng
                   and it["cond"] not in ("silence", "provoked")
                   and (it["cond"] == "full") == (cond == "full")]
            if not sel:
                continue
            print(f"-- {lng} {cond}: n={len(sel)}")
            for arm in arms:
                P = F = Wd = 0.0
                deltas, sup, att, m = [], 0, 0, 0
                for it in sel:
                    ref = it["ref"]
                    if cond == "prefix":
                        ref = " ".join(words(ref)[: len(words(it["primary"])) + 1])
                    rp = next((x for x in it["replay"] if arm_of(x) == arm), None)
                    p = recall(ref, it["primary"])
                    if rp is None or p is None:
                        continue
                    f, w = recall(ref, rp["frame"]), recall(ref, rp["word"])
                    m += 1
                    P, F, Wd = P + p, F + f, Wd + w
                    deltas.append(f - p)
                    sup += int(p >= 0.5 and f < 0.2)
                    att += int(w < f - 0.1)
                if not m:
                    continue
                deltas.sort()
                q = lambda x: deltas[min(len(deltas) - 1, int(x * len(deltas)))]
                print(f"   ctx {str(arm):>4}: recall primary {P / m:.2f} FRAME {F / m:.2f} WORD {Wd / m:.2f}"
                      f" | delta {q(0.1):+.2f}/{q(0.5):+.2f}/{q(0.9):+.2f} | SUPPRESSED {sup}/{m}"
                      f" | attribution {att}/{m}")

    # ---- 5. whole clips: replay CER against the primary's
    print("\n== 5. whole clips: mean CER vs reference, replay rule vs primary")
    for arm in arms:
        line = []
        for r in rules:
            cs, cp = [], []
            for it in items:
                if it["cond"] != "full" or not it["ref"]:
                    continue
                rp = next((x for x in it["replay"] if arm_of(x) == arm), None)
                if rp is None:
                    continue
                x, y = cer(normalise(rp[r]), normalise(it["ref"])), cer(normalise(it["primary"]), normalise(it["ref"]))
                if x is None or y is None:
                    continue
                cs.append(x)
                cp.append(y)
            if cs:
                line.append(f"{r} {sum(cs) / len(cs):.4f}")
        if line:
            print(f"   ctx {str(arm):>4}: " + " | ".join(line) + f" | primary {sum(cp) / len(cp):.4f} (n={len(cp)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("items")
    p.add_argument("--manifest", required=True)
    p.add_argument("--per-lang", type=int, default=30)
    p.add_argument("--provoked", type=int, default=10)
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--contexts", help="TSV lang<TAB>file (manifest-relative): fixed previous turns")
    p.add_argument("--silence", action="store_true", help="add a lead-in-only current turn per pair")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_items)
    p = sp.add_parser("report")
    p.add_argument("lab")
    p.add_argument("--items", required=True)
    p.add_argument("--manifest", required=True)
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
