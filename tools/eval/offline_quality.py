#!/usr/bin/env python3
"""Offline quality screen for a pack the runtime refuses to stream.

Same corpus, same bank selection and the SAME scorers as the streaming
qualification -- wer/cer/normalise/wer_format_free are imported from
tools/bench/streaming_metrics.py, never reimplemented, so a number here and a
number from the C=80 evidence mean the same thing.

Differences from tools/eval/cer_offline.py, and why:
  * the whole hypothesis is kept, not the first 400 characters: this bank's
    long class runs to 42 s and a truncated hypothesis scores as a deletion.
  * clips run in parallel with MYNAH_ASR_THREADS=1 each, because a quality
    screen must not cost an hour. Rule 4 says a transcript does not depend on
    thread count; this run also PROVES it on a sample (--verify-serial N).
  * every row carries the manifest's own metadata, so 'composed' (a synthetic
    concatenation of two FLEURS utterances) is never silently mixed with
    original FLEURS material.

    python3 pk_quality.py --model models/<pack> --bank bank.txt \
        --manifest samples/stress-en/manifest.json --out q.json [--jobs 24]
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
from streaming_metrics import cer, cer_format_free, normalise, pct, wer, wer_format_free


def run_one(binary, model, clip, quant, threads, decoder):
    env = dict(os.environ, MYNAH_ASR_THREADS=str(threads))
    cmd = [binary, "transcribe", "-m", model, "-i", clip]
    if quant:
        cmd += ["--quant", quant]
    if decoder:
        cmd += ["--decoder", decoder]
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r, time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bank", required=True, help="the qualification's bank.txt: <sha256> <path>")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--binary", default="./mynah-asr")
    ap.add_argument("--quant", default="", help="empty = whatever the pack ships")
    ap.add_argument("--decoder", default="", help="parakeet 110m hybrid: '' = TDT, 'ctc' = aux head")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--verify-serial", type=int, default=8,
                    help="re-run N clips serially at full width and require byte-identical text")
    a = ap.parse_args()

    meta = {}
    for e in json.load(open(a.manifest))["clips"]:
        meta[os.path.basename(e["file"])] = e

    bank = []
    for line in open(a.bank):
        sha, path = line.split()
        bank.append((sha, path))
    print(f"{len(bank)} clip(s) from {a.bank}, model {a.model}, "
          f"quant {a.quant or '(pack default)'}, decoder {a.decoder or 'default/TDT'}, "
          f"jobs {a.jobs} x {a.threads} thread(s)", flush=True)

    rows, t0 = [], time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, a.binary, a.model, p, a.quant, a.threads, a.decoder): (s, p)
                for s, p in bank}
        for fu in cf.as_completed(futs):
            sha, clip = futs[fu]
            r, wall = fu.result()
            m = meta.get(os.path.basename(clip), {})
            ref = m.get("text") or ""
            row = {"clip": clip, "sha256": sha, "class": m.get("class"),
                   "duration_s": m.get("duration_sec"), "composed": bool(m.get("composed")),
                   "fleurs_id": m.get("fleurs_id"), "split": m.get("split"),
                   "peak_dbfs": m.get("peak_dbfs"), "wall_s": round(wall, 3)}
            if r.returncode != 0:
                row["error"] = (r.stderr or "")[-300:]
            else:
                hyp = (r.stdout or "").strip()
                row["hyp"] = hyp
                row["ref"] = ref
                row["empty"] = not normalise(hyp)
                if ref:
                    row["wer"] = wer(hyp, ref)
                    row["cer"] = cer(hyp, ref)
                    row["wer_ff"] = wer_format_free(hyp, ref, "en")
                    row["cer_ff"] = cer_format_free(hyp, ref, "en")
                    row["ref_chars"] = len(normalise(ref))
                    row["ref_words"] = len(normalise(ref).split())
            rows.append(row)
    wall = time.monotonic() - t0

    # Rule 4 gate: the parallel screen must agree with the serial full-width run.
    mismatch = []
    for sha, clip in bank[:a.verify_serial]:
        r, _ = run_one(a.binary, a.model, clip, a.quant, 0, a.decoder)
        want = next(x for x in rows if x["clip"] == clip).get("hyp")
        got = (r.stdout or "").strip()
        if r.returncode == 0 and got != want:
            mismatch.append({"clip": clip, "parallel": want, "serial": got})

    def agg(sel, key):
        v = [r[key] for r in sel if r.get(key) is not None]
        return {"n": len(v), "mean": sum(v) / len(v) if v else None,
                "p50": pct(v, 50), "p95": pct(v, 95), "max": max(v) if v else None}

    scored = [r for r in rows if r.get("wer") is not None]
    orig = [r for r in scored if not r["composed"]]
    comp = [r for r in scored if r["composed"]]
    tot_c = sum(r["ref_chars"] for r in scored)
    tot_w = sum(r["ref_words"] for r in scored)
    out = {
        "model": os.path.basename(os.path.abspath(a.model)), "quant": a.quant or "pack-default",
        "decoder": a.decoder or "default", "mode": "transcribe (whole file, offline)",
        "bank": a.bank, "manifest": a.manifest,
        "clips": len(rows), "scored": len(scored),
        "errors": sum(1 for r in rows if r.get("error")),
        "empty": sum(1 for r in rows if r.get("empty")),
        "jobs": a.jobs, "threads_each": a.threads, "wall_s": round(wall, 1),
        "audio_s": round(sum(r["duration_s"] or 0 for r in rows), 1),
        "thread_invariance": {"checked": min(a.verify_serial, len(bank)),
                              "mismatches": len(mismatch), "detail": mismatch[:3]},
        "wer_corpus": sum(r["wer"] * r["ref_words"] for r in scored) / tot_w if tot_w else None,
        "cer_corpus": sum(r["cer"] * r["ref_chars"] for r in scored) / tot_c if tot_c else None,
        "wer": agg(scored, "wer"), "cer": agg(scored, "cer"),
        "wer_ff": agg(scored, "wer_ff"), "cer_ff": agg(scored, "cer_ff"),
        "by_class": {k: agg([r for r in scored if r["class"] == k], "wer")
                     for k in sorted({r["class"] for r in scored if r["class"]})},
        "fleurs_original": dict(agg(orig, "wer"), cer=agg(orig, "cer")["mean"]),
        "synthetic_composed": dict(agg(comp, "wer"), cer=agg(comp, "cer")["mean"]),
        "rows": sorted(rows, key=lambda r: r["clip"]),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    s = {k: v for k, v in out.items() if k != "rows"}
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
