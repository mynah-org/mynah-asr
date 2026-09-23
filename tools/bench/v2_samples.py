#!/usr/bin/env python3
"""v2_samples.py — make a qualification audible.

A PASS that can only be re-checked by re-running the server is not auditable.
This turns a soak's per-utterance record into a folder a person can open months
later: the audio that was actually sent, what the reference says, what the
server said unloaded, what it said under load, the partials it published on the
way there with their timestamps, and the timings the gates were computed from.

    tools/bench/v2_samples.py <run_dir> [--out <run_dir>/samples] [--per-class 10] [--zip]

What it collects:

  * every NOTABLE utterance -- errored, rejected, transcript diverging from the
    unloaded reference, empty, or in the top 1% of this run's own TTFP or
    emission lag. These are what a reader wants and what a summary hides.

    The percentile is a SELECTION rule, not a verdict. Every run has a tail, so
    about 1% of a perfect run sits above its own p99; being kept here means
    "worth looking at", never "out of envelope". PASS/FAIL belongs to the
    registered gates in v2_verdict.py and nowhere else.
  * a deterministic representative sample of healthy ones, class-balanced.

It does NOT copy thousands of WAVs: one copy per distinct clip, referenced by
every row that played it, and the complete per-utterance JSON stays archived
beside it. `--keep-events N` on the load run is what puts the partial sequence
into that JSON; without it the rows carry every timing but no partials, and the
index says so rather than pretending.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streaming_metrics import normalise, pct                  # noqa: E402

# SELECTION, NOT JUDGEMENT. This percentile decides which utterances are worth
# putting in front of a reader. It is NOT a health criterion and must never
# become one: a perfect run has a tail by definition, so ~1% of any run sits
# above its own p99 and being there means "interesting", never "wrong".
#
# What decides PASS/FAIL is the registered absolute and paired gates in
# .work/server-v2-qualification.md, checked by v2_verdict.py. A row in this
# index marked ANOMALY may belong to a run that passed every one of them.
OUTLIER_P = 99


def load_side(run, name):
    p = os.path.join(run, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def collect(run, per_class):
    ref = load_side(run, "reference.json")
    ttfp_base = load_side(run, "reference-ttfp.json")
    onsets = load_side(run, "onsets.json")
    rows, clips_needed, sources = [], set(), []

    for path in sorted(glob.glob(os.path.join(run, "soak*.json"))
                       + glob.glob(os.path.join(run, "ladder-C*.json"))
                       + glob.glob(os.path.join(run, "genscale-*.json"))):
        d = json.load(open(path))
        tag = os.path.basename(path)[:-5]
        utts = d.get("utterances") or []
        if not utts:
            continue
        sources.append(tag)
        lags = [u["ttfp_ms"] for u in utts if u.get("ttfp_ms") is not None]
        ttfp_hi = pct(lags, OUTLIER_P) if len(lags) >= 100 else None
        # The p99 of per-utterance MAXIMA, not of every delta: an utterance's
        # worst delta is above the per-delta p99 most of the time, so comparing
        # the two marked 16% of a healthy run as anomalous.
        emis = [m for m in (max((v for _, v in (u.get("lag_marks") or [])), default=None)
                            for u in utts) if m is not None]
        lag_hi = pct(emis, OUTLIER_P) if len(emis) >= 100 else None

        # Prefer representatives whose partials were kept -- a row with the
        # published sequence is a far better exhibit. But a run taken before
        # --keep-events existed has none, and returning only anomalies from it
        # would be a worse artefact than returning rows that say so.
        any_events = any(u.get("events") for u in utts)
        healthy_by_class = {}
        for u in utts:
            clip = u.get("clip") or ""
            unloaded = ref.get(clip)
            why = []
            if u.get("error"):
                why.append("error")
            if u.get("rejected"):
                why.append("rejected")
            if unloaded is not None and not u.get("error") and \
                    normalise(u.get("text") or "") != normalise(unloaded):
                why.append("transcript diverges from the unloaded reference")
            if ttfp_hi is not None and (u.get("ttfp_ms") or 0) > ttfp_hi:
                why.append(f"top 1% of this run's ttfp (selection, not a gate)")
            umax = max((m for _, m in (u.get("lag_marks") or [])), default=None)
            if lag_hi is not None and umax is not None and umax > lag_hi:
                why.append("top 1% of this run's emission lag (selection, not a gate)")
            if not (u.get("text") or "").strip() and not u.get("rejected"):
                why.append("empty transcript")

            if why:
                rows.append(_row(u, tag, clip, ref, ttfp_base, onsets, "NOTABLE: " + "; ".join(why)))
                clips_needed.add(clip)
                continue
            # Representative healthy ones: deterministic, class-balanced, and
            # only utterances whose partials were kept -- a row with no partial
            # sequence is a weaker exhibit than one with it.
            k = u.get("class") or "?"
            bucket = healthy_by_class.setdefault(k, [])
            if len(bucket) < per_class and (u.get("events") or not any_events):
                bucket.append(u)
        for k, bucket in sorted(healthy_by_class.items()):
            for u in bucket:
                clip = u.get("clip") or ""
                rows.append(_row(u, tag, clip, ref, ttfp_base, onsets, "representative"))
                clips_needed.add(clip)
    return rows, clips_needed, sources


def _row(u, tag, clip, ref, ttfp_base, onsets, why):
    base = ttfp_base.get(clip) or {}
    t0 = u.get("t_start_abs")
    partials = []
    for e in (u.get("events") or []):
        if e.get("type") != "delta" or not (e.get("text") or ""):
            continue
        partials.append({"t_rel_s": None if (t0 is None or e.get("t") is None)
                         else round(e["t"] - t0, 3),
                         "audio_s": e.get("audio_s"), "lag_ms": e.get("lag_ms"),
                         "text": e.get("text")})
    pen = (None if (u.get("ttfp_ms") is None or base.get("ttfp_ms") is None)
           else round(u["ttfp_ms"] - base["ttfp_ms"], 1))
    return {
        "run": tag, "clip": clip, "wav": os.path.basename(clip),
        "class": u.get("class"), "stream": u.get("stream"), "rep": u.get("rep"),
        "why_kept": why,
        "reference_text": None,                       # filled by the caller if a bank has it
        "unloaded_transcript": ref.get(clip),
        "loaded_transcript": u.get("text"),
        "partials": partials or None,
        "partials_note": None if partials else
                         "not retained: run the load with --keep-events N",
        "onset_s": onsets.get(clip) if isinstance(onsets, dict) else None,
        "duration_s": u.get("audio_s"),
        "ttfp_ms": u.get("ttfp_ms"),
        "speech_to_first_partial_ms": u.get("ttfp_from_onset_ms"),
        "ttfp_load_penalty_ms": pen,
        "unloaded_ttfp_ms": base.get("ttfp_ms"),
        "first_delta_lag_ms": u.get("first_delta_lag_ms"),
        "first_delta_audio_s": u.get("first_delta_audio_s"),
        "finalization_ms": u.get("fin_ms"),
        "emission_lag_max_ms": max((m for _, m in (u.get("lag_marks") or [])), default=None),
        "backlog_max_s": u.get("backlog_max_s"),
        "deltas": u.get("deltas"),
        # Two different questions, kept apart. serving_regression compares the
        # loaded transcript with this clip's own UNLOADED one and is what the
        # server is judged on; wer/cer compare it with the human reference and
        # describe the checkpoint, which load did not cause and cannot fix.
        "serving_regression": (None if ref.get(clip) is None else
                               normalise(u.get("text") or "") != normalise(ref[clip])),
        "wer_vs_human": u.get("wer"), "cer_vs_human": u.get("cer"),
        "error": u.get("error"), "rejected": u.get("rejected"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-class", type=int, default=10,
                    help="healthy representatives per duration class per run file")
    ap.add_argument("--bank", default=None,
                    help="a bank manifest, to add the human reference text")
    ap.add_argument("--zip", action="store_true", help="also write samples.zip beside it")
    a = ap.parse_args()

    out = a.out or os.path.join(a.run, "samples")
    rows, clips, sources = collect(a.run, a.per_class)
    if not rows:
        sys.exit(f"no utterance record in {a.run}: nothing to sample")

    truth = {}
    if a.bank and os.path.exists(a.bank):
        d = json.load(open(a.bank))
        root = os.path.dirname(a.bank)
        for c in (d.get("clips") or []):
            f = c.get("file") or ""
            truth[os.path.join(root, f)] = c.get("text")
            truth[f] = c.get("text")
    for r in rows:
        r["reference_text"] = truth.get(r["clip"]) or truth.get(os.path.basename(r["clip"]))

    os.makedirs(os.path.join(out, "wav"), exist_ok=True)
    copied, missing = 0, 0
    for clip in sorted(clips):
        src = clip
        if not os.path.exists(src):
            missing += 1
            continue
        dst = os.path.join(out, "wav", os.path.basename(clip))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    anomalies = [r for r in rows if r["why_kept"].startswith("NOTABLE")]
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(f"""Qualification samples for {os.path.basename(a.run.rstrip('/'))}

{len(rows)} rows in index.jsonl over {len(sources)} run file(s): {', '.join(sources)}
{len(anomalies)} notable, {len(rows) - len(anomalies)} representative.

NOTABLE is a SELECTION rule, not a verdict. Errors, rejections, transcript
divergence and empty output are genuine faults; "top 1% of this run's ttfp or
emission lag" is not -- every run has a tail, and about 1% of a perfect one
sits above its own p99. Whether the run PASSED is decided by the registered
gates in .work/server-v2-qualification.md and reported by v2_verdict.py, never
by the presence of rows here.
{copied} wav file(s) under wav/ ({len(clips)} distinct clips, {missing} not found on this host).

One row per utterance INSTANCE; one wav per distinct clip, so a clip played
many times is stored once and referenced many times. Each row carries what the
server was sent, what it said unloaded, what it said under load, the partials
it published with timestamps where they were retained, and the timings the
gates were computed from.

The complete per-utterance record stays in the run's own soak*.json beside this
folder; this is the readable slice, not a replacement for it.

wer_vs_human and cer_vs_human are empty on utterances the run excluded from its
warm-up: those are played but not scored, which is what excluding them means.
Their serving fields -- timings, partials, and the comparison against the
unloaded transcript -- are all present.

serving_regression is the question the SERVER is judged on: did the loaded
transcript differ from this clip's own unloaded one. wer_vs_human and
cer_vs_human describe the CHECKPOINT and no amount of serving work changes them.
""")
    print(f"{len(rows)} row(s) ({len(anomalies)} notable -- a selection, not a verdict), "
          f"{copied} wav copied -> {out}")
    if missing:
        print(f"  {missing} clip(s) were not on this host and were not copied")
    if a.zip:
        z = out.rstrip("/") + ".zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out):
                for name in files:
                    p = os.path.join(root, name)
                    zf.write(p, os.path.relpath(p, os.path.dirname(out)))
        print(f"  -> {z} ({os.path.getsize(z) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
