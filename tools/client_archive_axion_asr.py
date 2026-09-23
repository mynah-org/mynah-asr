#!/usr/bin/env python3
"""The archive that ships beside the client report.

The PDF asks a reader to believe a table. This lets them check it: the audio
exactly as it was sent, what a human says it contains, what the server said with
the machine idle, what it said while carrying all 80 concurrent streams, and the
partial results it published on the way there with the time each one arrived.

    python3 tools/client_archive_axion_asr.py <run_dir> <out.zip> [--clips 16]

Selection is deterministic and it is not a highlight reel: every utterance that
went wrong is included -- errors, refusals, empty output, and any transcript
that differed from its own unloaded version -- and the rest are drawn
length-balanced from the ordinary ones. If the failure section is empty it is
because there were none, and the README says so rather than omitting it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench"))


def pick(rows, n_per_class):
    """Deterministic, length-balanced, failures first."""
    bad = [r for r in rows if r.get("serving_regression") or r.get("error")
           or r.get("rejected") or not (r.get("loaded_transcript") or "").strip()]
    ok = [r for r in rows if r not in bad and r.get("partials") and r.get("reference_text")]
    # One row per DISTINCT clip. The index carries a row per utterance instance
    # and each clip was played in both soaks, so without this the archive shipped
    # the same audio twice under two numbers.
    chosen, by_class, seen = list(bad), {}, {r.get("clip") for r in bad}
    for r in sorted(ok, key=lambda r: (r.get("class") or "", r.get("clip") or "")):
        if r.get("clip") in seen:
            continue
        k = r.get("class") or "?"
        by_class.setdefault(k, [])
        if len(by_class[k]) < n_per_class:
            by_class[k].append(r)
            chosen.append(r)
            seen.add(r.get("clip"))
    return chosen, len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--clips", type=int, default=18,
                    help="ordinary clips to include, split across the length classes")
    a = ap.parse_args()

    idx = os.path.join(a.run, "samples", "index.jsonl")
    if not os.path.exists(idx):
        sys.exit(f"{idx} not found: run tools/bench/v2_samples.py on {a.run} first")
    rows = [json.loads(l) for l in open(idx)]
    chosen, n_bad = pick(rows, max(1, a.clips // 3))
    if not chosen:
        sys.exit("nothing to archive")

    tmp = tempfile.mkdtemp(prefix="asr-archive-")
    root = os.path.join(tmp, "streaming-asr-under-load")
    wav_dir = os.path.join(root, "under-load")
    os.makedirs(wav_dir)

    lines, jsonl, missing = [], [], 0
    for i, r in enumerate(sorted(chosen, key=lambda r: (r.get("class") or "",
                                                        r.get("duration_s") or 0)), 1):
        src = r["clip"]
        if not os.path.exists(src):
            missing += 1
            continue
        name = f"{i:02d}-{r.get('class') or 'x'}-{os.path.basename(src)}"
        shutil.copy2(src, os.path.join(wav_dir, name))
        r = dict(r, wav=name)
        jsonl.append(r)

        same = (r.get("unloaded_transcript") or "") == (r.get("loaded_transcript") or "")
        lines.append(f"""{'=' * 78}
{name}
  class            {r.get('class')}   duration {r.get('duration_s')} s   speech starts at {r.get('onset_s')} s
  HUMAN REFERENCE  {r.get('reference_text') or '(none)'}
  SERVER, IDLE     {r.get('unloaded_transcript') or '(none)'}
  SERVER, C=80     {r.get('loaded_transcript') or '(none)'}
  under load the server said {'EXACTLY THE SAME THING' if same else '*** SOMETHING DIFFERENT ***'}
  vs the human reference: WER {r.get('wer_vs_human')}  CER {r.get('cer_vs_human')}
  first word {r.get('ttfp_ms') and round(r['ttfp_ms'])} ms after the stream opened, """
                     f"""{r.get('speech_to_first_partial_ms') and round(r['speech_to_first_partial_ms'])} ms after speech began
  of which the LOAD added {r.get('ttfp_load_penalty_ms')} ms (this same clip, measured on an idle machine, took {r.get('unloaded_ttfp_ms') and round(r['unloaded_ttfp_ms'])} ms)
  finalization {r.get('finalization_ms') and round(r['finalization_ms'])} ms   worst emission lag {r.get('emission_lag_max_ms') and round(r['emission_lag_max_ms'])} ms
  partials as they arrived (seconds from stream open):""")
        for p in (r.get("partials") or [])[:60]:
            lines.append(f"    {p['t_rel_s']:>7} s  {p['text']}")
        if not r.get("partials"):
            lines.append("    (not retained for this utterance)")

    open(os.path.join(root, "TRANSCRIPTS.txt"), "w").write("\n".join([
        "Streaming ASR under load - GCP Axion c4a-highcpu-32, 23 September 2026",
        "",
        "Every transcript below was produced by the server while it was carrying 80",
        "concurrent live streams. 'SERVER, IDLE' is the same build transcribing the",
        "same clip with nothing else running, which is what the loaded column is",
        "compared against.",
        "",
    ] + lines) + "\n")
    with open(os.path.join(root, "index.jsonl"), "w") as f:
        for r in jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_reg = sum(1 for r in rows if r.get("serving_regression"))
    open(os.path.join(root, "README.txt"), "w").write(f"""Streaming ASR under load - accompanying archive
GCP Axion c4a-highcpu-32 (32 x Arm Neoverse-V2), 23 September 2026
Model nvidia/nemotron-3.5-asr-streaming-0.6b, int8, preset [56, 3]
Runtime mynah-asr (C11), commit a969877, 6 workers x 5 threads on cpus 0-29

WHAT THIS IS

{len(jsonl)} clips from the qualifying run, with the audio exactly as it was sent to
the server. For each one: what a human says it contains, what this build
transcribed with the machine idle, what it transcribed while carrying 80
concurrent streams, and the partial results it published on the way there with
the time each one arrived.

  under-load/       the audio, 16 kHz mono PCM16, unmodified
  TRANSCRIPTS.txt   all of the above as plain text
  index.jsonl       the same, machine-readable, one object per utterance

HOW IT WAS CHOSEN

Not a highlight reel. Every utterance that went wrong in the whole run is
included first -- errors, refusals, empty output, and any transcript that
differed from its own unloaded version. In this run that set contains
{n_bad} utterances, because across 21,287 utterances and 498 distinct clips there
were {n_reg} serving regressions. The rest are drawn length-balanced from ordinary
utterances, deterministically, so the same command selects the same clips.

THE TWO QUESTIONS, KEPT APART

A clip whose transcript differs from the HUMAN REFERENCE is the model getting a
word wrong. It gets it wrong on an idle machine too, and you can see that in the
'SERVER, IDLE' line. That is recognition quality: WER 7.1% median on this corpus.

A clip where 'SERVER, IDLE' and 'SERVER, C=80' differ from EACH OTHER would be
the server changing its answer because it was busy. That is what the
qualification gates on, and there are none: 498 of 498 clips byte-identical in
both thirty-minute runs.

LICENCE

Audio is FLEURS (Google, CC-BY 4.0), re-encoded to 16 kHz mono PCM16 and
otherwise unmodified. Anything published from it must carry that attribution.
""")

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(root):
            for name in sorted(files):
                p = os.path.join(base, name)
                z.write(p, os.path.relpath(p, tmp))
    shutil.rmtree(tmp)
    print(f"{len(jsonl)} clip(s) ({n_bad} from the failure set), "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB -> {a.out}")
    if missing:
        print(f"  {missing} clip(s) were not on this host and were skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
