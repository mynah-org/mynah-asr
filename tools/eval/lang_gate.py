#!/usr/bin/env python3
"""lang_gate.py — English and French quality, reported apart, never averaged.

    python3 tools/eval/lang_gate.py -m models_local/<pack> \
        --manifest samples/eval-bank/manifest.json --root samples/eval-bank \
        --json .work/evidence/gate.json
    python3 tools/eval/lang_gate.py ... --baseline .work/evidence/gate.json

WHY IT IS ITS OWN TOOL.  `partial_quality.py` compares two DECODER
configurations on a diagnostic corpus. This is the release gate: one
configuration, real ground truth, per language, with the spread of the estimate
attached to every number.

**EN and FR never collapse into one score.** An aggregate would let a French
regression hide behind English, which is exactly the failure a "top EN/FR"
claim has to be protected from.

**Ground truth is the oracle.** The dataset's own transcription, always. The
model's offline output is reported beside it as a diagnostic and is never
treated as truth -- on the 21-clip corpus the two disagreed on 7 of 21, in both
directions.

WHAT IT REPORTS, per language:

  WER and CER, each with a bootstrap 95 % interval, so "is this corpus big
    enough" is answered rather than assumed;
  substitutions / deletions / insertions -- one WER, three different failures,
    and an ASR that truncates is not an ASR that hallucinates;
  empty-transcript rate;
  WER by utterance duration, because short utterances are where first-emission
    behaviour bites and a long-utterance average hides it;
  first published lexical word correct, and the speech consumed before it;
  first CORRECT lexical partial: the same quantity, undefined when the first
    word is wrong, because this publication path appends and never retracts;
  published-prefix divergence (the only instability this delta gate permits).

Exit: 0 ok · 1 a regression past --margin against --baseline · 2 usage
      · 77 model, manifest or binary missing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import streaming_metrics as M          # noqa: E402
import clip_onset                      # noqa: E402


def boot_ci(vals, n=2000, seed=20260922, lo=2.5, hi=97.5):
    """Bootstrap percentile interval of the mean. Deterministic seed: a gate whose
    interval moves between runs cannot be compared with yesterday's."""
    if not vals:
        return None, None
    r = random.Random(seed)
    k = len(vals)
    means = []
    for _ in range(n):
        means.append(sum(vals[r.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return means[int(lo / 100 * n)], means[min(int(hi / 100 * n), n - 1)]


def run_clip(binary, model, quant, clip, lang, extra):
    p = subprocess.run([binary, "stream", "-m", model, "-i", clip, "--quant", quant,
                        "--lang", lang, "--deltas"] + extra, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    deltas, lib = [], None
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("type") == "delta":
            deltas.append(o)
        elif o.get("type") == "final":
            lib = o.get("lib_text")
    return deltas, lib


def summarise(rows):
    """Everything about one language. `rows` are per-utterance dicts."""
    wer = [r["wer"] for r in rows if r["wer"] is not None]
    cer = [r["cer"] for r in rows if r["cer"] is not None]
    al = [r["align"] for r in rows if r["align"]]
    s = {
        "n": len(rows),
        "audio_s": round(sum(r["duration_sec"] for r in rows), 1),
        "wer": {"mean": st.mean(wer) if wer else None,
                "median": st.median(wer) if wer else None,
                "ci95": boot_ci(wer), "n": len(wer)},
        "cer": {"mean": st.mean(cer) if cer else None,
                "median": st.median(cer) if cer else None,
                "ci95": boot_ci(cer), "n": len(cer)},
        "sub": sum(a["sub"] for a in al), "del": sum(a["del"] for a in al),
        "ins": sum(a["ins"] for a in al), "ref_words": sum(a["ref_words"] for a in al),
    }
    tot = s["ref_words"] or 1
    s["wer_pooled"] = (s["sub"] + s["del"] + s["ins"]) / tot
    for name, pred, guard in (
        ("empty", lambda r: not M.normalise(r["text"]), None),
        ("first_word_correct", lambda r: r["first_word_correct"] is True,
         lambda r: r["first_word_correct"] is not None),
        ("prefix_divergence", lambda r: r["prefix_divergence"] is True,
         lambda r: r["prefix_divergence"] is not None),
    ):
        el = [r for r in rows if guard is None or guard(r)]
        s[name] = {"rate": (sum(1 for r in el if pred(r)) / len(el)) if el else None,
                   "n": sum(1 for r in el if pred(r)), "of": len(el)}
    for k in ("speech_at_first_word", "speech_at_first_correct_word"):
        v = [r[k] for r in rows if r.get(k) is not None]
        s[k] = {"median": st.median(v) if v else None,
                "p95": M.pct(v, 95) if v else None, "n": len(v)}
    # duration buckets: short utterances are where first emission bites
    s["by_duration"] = {}
    for label, lo, hi in (("<6s", 0, 6), ("6-12s", 6, 12), (">=12s", 12, 1e9)):
        b = [r for r in rows if lo <= r["duration_sec"] < hi]
        bw = [r["wer"] for r in b if r["wer"] is not None]
        s["by_duration"][label] = {"n": len(b),
                                   "wer_mean": st.mean(bw) if bw else None,
                                   "wer_median": st.median(bw) if bw else None}
    return s


def fmt(v, nd=4):
    return "  -   " if v is None else f"{v:.{nd}f}"


def report(lang, s):
    print(f"\n=== {lang.upper()}   {s['n']} utterances, {s['audio_s']:.0f} s of audio")
    for k in ("wer", "cer"):
        d = s[k]
        ci = d["ci95"]
        band = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "—"
        width = (ci[1] - ci[0]) if ci[0] is not None else None
        print(f"  {k.upper():4} mean {fmt(d['mean'])}  median {fmt(d['median'])}  "
              f"95% CI {band}" + (f"  (width {width:.4f})" if width is not None else ""))
    print(f"  WER pooled over words: {s['wer_pooled']:.4f}   "
          f"S {s['sub']}  D {s['del']}  I {s['ins']}  over {s['ref_words']} reference words")
    for k, label in (("empty", "empty transcripts"),
                     ("first_word_correct", "first published word CORRECT"),
                     ("prefix_divergence", "published byte later rewritten")):
        d = s[k]
        r = "  -  " if d["rate"] is None else f"{100*d['rate']:5.1f}%"
        print(f"  {label:34} {r} ({d['n']}/{d['of']})")
    print(f"  speech before the first word          "
          f"median {fmt(s['speech_at_first_word']['median'],3)} s  "
          f"p95 {fmt(s['speech_at_first_word']['p95'],3)} s  (n={s['speech_at_first_word']['n']})")
    print(f"  speech before the first CORRECT word  "
          f"median {fmt(s['speech_at_first_correct_word']['median'],3)} s  "
          f"p95 {fmt(s['speech_at_first_correct_word']['p95'],3)} s  "
          f"(n={s['speech_at_first_correct_word']['n']}, undefined when the first word is wrong)")
    print("  WER by duration: " + "   ".join(
        f"{k} n={v['n']} mean {fmt(v['wer_mean'],3)}" for k, v in s["by_duration"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--manifest", default="samples/eval-bank/manifest.json")
    ap.add_argument("--root", default="samples/eval-bank")
    ap.add_argument("--binary", default="./mynah-asr")
    ap.add_argument("--extra", default="", help="extra CLI flags, e.g. '--lookahead 0'")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--langs", help="comma list; default every language in the manifest")
    ap.add_argument("--json")
    ap.add_argument("--baseline")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="WER a language may lose against the baseline before this exits 1")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    for p, what in ((a.binary, "binary"), (a.manifest, "manifest")):
        if not os.path.exists(p):
            print(f"lang_gate: no {what} at {p} — SKIP", file=sys.stderr)
            return 77
    if not os.path.exists(os.path.join(a.model, "mynah.json")):
        print(f"lang_gate: no converted pack at {a.model} — SKIP", file=sys.stderr)
        return 77

    man = json.load(open(a.manifest, encoding="utf-8"))
    samples = man["samples"][: a.limit] if a.limit else man["samples"]
    extra = a.extra.split() if a.extra else []
    by_lang, failed = {}, 0
    keep = set(a.langs.split(",")) if a.langs else None
    for s in samples:
        if keep and s["lang"] not in keep:
            continue
        clip = os.path.join(a.root, s["file"])
        if not os.path.exists(clip):
            continue
        got = run_clip(a.binary, a.model, a.quant, clip, s["lang"], extra)
        if got is None:
            failed += 1
            continue
        deltas, lib = got
        try:
            onset = clip_onset.onset_s(clip)
        except Exception:
            onset = None
        q = M.partial_quality(deltas, s["text"], lib_text=lib, onset_s=onset)
        row = {"file": s["file"], "duration_sec": s["duration_sec"], "text": q["text"],
               "wer": q["wer"], "cer": q["cer"],
               "align": M.align_counts(q["text"], s["text"]),
               "first_word": q["first_word"],
               "first_word_correct": q["first_word_correct"],
               "prefix_divergence": q["published_prefix_divergence"],
               "speech_at_first_word": q["speech_at_first_word"],
               "speech_at_first_correct_word": (q["speech_at_first_word"]
                                                if q["first_word_correct"] else None)}
        by_lang.setdefault(s["lang"], []).append(row)
    if failed:
        print(f"lang_gate: {failed} clip(s) the CLI refused", file=sys.stderr)

    out = {"model": a.model, "quant": a.quant, "manifest": a.manifest,
           "extra": extra, "label": a.label, "langs": {}}
    for lang in sorted(by_lang):
        s = summarise(by_lang[lang])
        out["langs"][lang] = {"summary": s, "utterances": by_lang[lang]}
        report(lang, s)

    rc = 0
    if a.baseline and os.path.exists(a.baseline):
        base = json.load(open(a.baseline, encoding="utf-8"))
        print(f"\n=== against {a.baseline}   (each language on its own)")
        for lang in sorted(out["langs"]):
            b = base.get("langs", {}).get(lang, {}).get("summary")
            if not b:
                print(f"  {lang.upper()}: not in the baseline")
                continue
            n = out["langs"][lang]["summary"]
            for k in ("wer", "cer"):
                d = n[k]["mean"] - b[k]["mean"]
                bad = k == "wer" and d > a.margin
                print(f"  {lang.upper()} {k.upper()} {b[k]['mean']:.4f} -> {n[k]['mean']:.4f} "
                      f"({d:+.4f}){'   REGRESSION' if bad else ''}")
                if bad:
                    rc = 1
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nwrote {a.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
