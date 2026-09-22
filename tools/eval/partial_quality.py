#!/usr/bin/env python3
"""partial_quality.py — is the first partial EARLY, or merely early and wrong?

    python3 tools/eval/partial_quality.py -m models/nemotron-... --quant int8 \
        --arm "base:--lookahead 3" --arm "fast:--lookahead 0" [--json pq.json]

WHY THIS EXISTS.  R-4 closed the serving half of first-partial latency:
publication costs about 0.1 ms end to end at every concurrency, so anything that
moves the first word moves the DECODER.  A decoder can be made to speak sooner by
speaking before it knows, and Q-2 measured exactly that: with the pre-registered
stability rule, one stable runner-up in four was the WRONG token, and the wrong
ones were no less early and no less confident than the right ones.  So "first
nonblank moved earlier" is not a result.  This is the gate that says what the
earliness cost.

SIX QUANTITIES, REPORTED SEPARATELY.  Merging them is how an arm that trades
correctness for latency passes:

  final CER/WER                 what the utterance was worth in the end
  first useful correct partial  speech consumed before the first SETTLED word,
                                and separately before the first COMPATIBLE
                                evidence -- a growing subword is evidence, not
                                yet a word
  wrong first lexical token     of the utterances where the question applies
  leading-silence hallucination text published having consumed only pre-onset
                                audio
  partial stability             the delta gate publishes a byte slice, so a
                                client's view only grows and there is nothing to
                                retract; what CAN happen is that the library
                                rewrites a byte already published, which is
                                `prefix_divergence`, and it is checked rather
                                than assumed
  streaming / offline / corpus  three numbers, never folded into one

THE ORACLE IS THE CORPUS.  samples/manifest.json, always.  The model's own
offline output is reported beside the reference precisely because the two
disagree on some clips (7 of 21 in the Q-2 corpus run, in both directions); using
it as truth would score the streaming path against a hypothesis.

THE AXIS IS AUDIO CONSUMED, NOT WALL CLOCK.  The same clips gave the same
speech-at-first-nonblank on the dev Mac and on the Axion, clip for clip (F30/E2),
so two decoder configurations can be compared here without renting a machine.
Wall-clock TTFP under load is a serving question and stays in the server harness.

Every metric comes from tools/bench/streaming_metrics.py, which has known-answer
tests for each of them on fixed synthetic transcripts.  This file runs the CLI
and arranges what comes back.

Exit: 0 ok · 1 an arm regressed against the first one past --margin · 2 usage
      · 77 the model, the manifest or the binary is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import streaming_metrics as M          # noqa: E402
import clip_onset                      # noqa: E402

BIN = "./mynah-asr"


def run_stream(binary, model, quant, clip, lang, extra):
    """One clip through the streaming path; returns (deltas, lib_text) or None."""
    cmd = [binary, "stream", "-m", model, "-i", clip, "--quant", quant,
           "--lang", lang, "--deltas"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
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
            continue                      # a torn line is dropped and counted by the caller
        if o.get("type") == "delta":
            deltas.append(o)
        elif o.get("type") == "final":
            lib = o.get("lib_text")
    return deltas, lib


def run_offline(binary, model, quant, clip, lang, extra):
    cmd = [binary, "transcribe", "-m", model, "-i", clip, "--quant", quant,
           "--lang", lang] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def parse_arm(spec):
    name, _, rest = spec.partition(":")
    if not name or not _:
        raise SystemExit(f"usage: --arm NAME:<cli flags>   (got {spec!r})")
    return name, rest.split()


def fmt(v, nd=3, unit=""):
    return "   -  " if v is None else f"{v:.{nd}f}{unit}"


def fmt_rate(r):
    if r["of"] == 0:
        return "   -   (0)"
    return f"{100.0 * r['rate']:5.1f}% ({r['n']}/{r['of']})"


def report(name, summ, rows, per_utterance=True):
    print(f"\n=== arm {name}: {summ['n']} utterance(s)")
    if per_utterance:
        # Raw values before percentiles, always: a median cannot show which clip
        # lost its first word, and the mechanism lives in the individual rows.
        print(f"  {'clip':34} {'sp@1st':>7} {'sp@word':>8} {'sp@evid':>8} "
              f"{'first word':16} {'ok':>3} {'CER':>6}")
        for r in rows:
            print(f"  {r['clip'][-34:]:34} {fmt(r['speech_at_first_partial'],3):>7} "
                  f"{fmt(r['speech_at_first_word'],3):>8} {fmt(r['speech_at_first_evidence'],3):>8} "
                  f"{(r['first_word'] or '-')[:16]:16} "
                  f"{'-' if r['first_word_correct'] is None else ('y' if r['first_word_correct'] else 'N'):>3} "
                  f"{fmt(r['cer'],4):>6}")
    for k, label in (("cer", "final CER"), ("wer", "final WER"),
                     ("speech_at_first_partial", "speech @ first partial"),
                     ("speech_at_first_word", "speech @ first SETTLED word"),
                     ("speech_at_first_evidence", "speech @ first compatible evidence"),
                     ("offline_cer", "offline CER (diagnostic, not the oracle)"),
                     ("streaming_vs_offline_cer", "streaming vs offline")):
        v = summ.get(k)
        if v:
            print(f"  {label:42} median {fmt(v['median'],4)}  mean {fmt(v['mean'],4)}"
                  f"  p95 {fmt(v['p95'],4)}  (n={v['n']})")
        else:
            print(f"  {label:42} no measurement")
    for k, label in (("wrong_first_word", "WRONG first lexical token"),
                     ("no_first_word", "published no settled word at all"),
                     ("leading_silence", "text during the leading silence"),
                     ("prefix_divergence", "published byte later rewritten"),
                     ("disagrees_with_offline", "streaming disagrees with offline")):
        print(f"  {label:42} {fmt_rate(summ[k])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--manifest", default="samples/manifest.json")
    ap.add_argument("--binary", default=BIN)
    ap.add_argument("--root", default="samples", help="where the manifest's files live")
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME:FLAGS", help="repeatable; the first arm is the baseline")
    ap.add_argument("--onsets", help="onset map (clip_onset.py); computed here when absent")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="CER a later arm may lose against the first before this exits 1")
    ap.add_argument("--limit", type=int, help="first N clips only (a smoke run)")
    ap.add_argument("--json", help="write the whole result here")
    ap.add_argument("--no-per-utterance", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.binary):
        print(f"partial_quality: {a.binary} not built — SKIP", file=sys.stderr)
        return 77
    if not os.path.isdir(a.model) or not os.path.exists(os.path.join(a.model, "mynah.json")):
        print(f"partial_quality: no converted pack at {a.model} — SKIP", file=sys.stderr)
        return 77
    if not os.path.exists(a.manifest):
        print(f"partial_quality: no manifest at {a.manifest} — SKIP", file=sys.stderr)
        return 77

    man = json.load(open(a.manifest, encoding="utf-8"))
    samples = man["samples"][: a.limit] if a.limit else man["samples"]
    arms = [parse_arm(x) for x in a.arm]

    onsets = json.load(open(a.onsets, encoding="utf-8")) if a.onsets else {}

    out = {"model": a.model, "quant": a.quant, "manifest": a.manifest, "arms": {}}
    base_cer = None
    rc = 0
    for name, extra in arms:
        rows = []
        failed = 0
        for s in samples:
            clip = os.path.join(a.root, s["file"])
            if not os.path.exists(clip):
                continue
            on = onsets.get(s["file"])
            if on is None:
                on = onsets.get(clip)
            if on is None:
                try:
                    on = clip_onset.onset_s(clip)
                except Exception:
                    on = None
            got = run_stream(a.binary, a.model, a.quant, clip, s["lang"], extra)
            if got is None:
                failed += 1
                continue
            deltas, lib = got
            offline = run_offline(a.binary, a.model, a.quant, clip, s["lang"], extra)
            q = M.partial_quality(deltas, s["text"], lib_text=lib,
                                  onset_s=on, offline=offline)
            q["clip"] = s["file"]
            q["onset_s"] = on
            rows.append(q)
        if failed:
            print(f"arm {name}: {failed} clip(s) the CLI refused", file=sys.stderr)
        summ = M.partial_quality_summary(rows)
        out["arms"][name] = {"flags": extra, "summary": summ, "utterances": rows,
                             "cli_failures": failed}
        report(name, summ, rows, per_utterance=not a.no_per_utterance)
        med = (summ.get("cer") or {}).get("median")
        if base_cer is None:
            base_cer = med
        elif med is not None and base_cer is not None and med > base_cer + a.margin:
            print(f"\n  REGRESSION: arm {name} median CER {med:.4f} is more than "
                  f"{a.margin} worse than the baseline {base_cer:.4f}")
            rc = 1

    if len(arms) == 2:
        (n0, _), (n1, _) = arms
        s0, s1 = out["arms"][n0]["summary"], out["arms"][n1]["summary"]
        print(f"\n=== {n1} against {n0}: every axis on its own")
        for k, label, better in (
                ("speech_at_first_word", "speech @ first settled word", "lower"),
                ("speech_at_first_evidence", "speech @ first evidence", "lower"),
                ("cer", "final CER", "lower"),
                ("wer", "final WER", "lower")):
            v0, v1 = s0.get(k), s1.get(k)
            if not v0 or not v1 or v0["median"] is None or v1["median"] is None:
                print(f"  {label:36} not comparable")
                continue
            d = v1["median"] - v0["median"]
            print(f"  {label:36} {v0['median']:.4f} -> {v1['median']:.4f}  "
                  f"({d:+.4f}, {better} is better)")
        for k, label in (("wrong_first_word", "wrong first lexical token"),
                         ("leading_silence", "leading-silence text"),
                         ("prefix_divergence", "published byte rewritten")):
            print(f"  {label:36} {fmt_rate(s0[k])}  ->  {fmt_rate(s1[k])}")
        print("\n  A later first word is not a loss and an earlier one is not a win:\n"
              "  read the correctness rows before the latency rows.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"\nwrote {a.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
