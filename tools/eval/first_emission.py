#!/usr/bin/env python3
"""B1/B2 — split the wait before the first word into AUDIO the model needed,
DECODER blanks, and the publication gate that may hold a decoded token.

Everything here is measured in AUDIO time, never wall time: `mynah-asr stream`
runs faster than real time, so its wall clock says nothing about a live stream,
while `audio_s` is exactly the quantity the qualification's
`first_delta_audio_s` reports and is therefore directly comparable to it.

Per clip it reads three things out of one run:
  * `[RNNT] frame= audio_s= ... chose=` -- every decoder decision, so the first
    NON-BLANK decision and the number of blank decisions before it are counted
    rather than inferred;
  * `[RNNT] emit q= audio_s= tokens_added= chars= chars_emitted=` -- the only
    line that links a DECODED token to whether anything was PUBLISHED, which is
    the publication gate at src/mynah_asr.c:1004;
  * the `--deltas` JSON on stdout -- what a client would actually have seen.

    python3 b1_emission.py --model <dir> --clips list.txt --onsets onsets.json \
        --out b1.json [--jobs 8] [--lookahead N]
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
from streaming_metrics import pct

FRAME = re.compile(r"\[RNNT\] frame=(-?\d+) audio_s=([\d.]+).*?chose=(-?\d+)"
                   r"(?:.*?margin_lex=(-?[\d.]+))?(?:.*?margin_nb=(-?[\d.]+))?")
BLANKF = re.compile(r"blank=(\d+):")
EMIT = re.compile(r"\[RNNT\] emit q=(\d+) audio_s=([\d.]+) tokens_added=(\d+) "
                  r"chars=(\d+) chars_emitted=(\d+)")


def one(binary, model, clip, quant, lookahead):
    env = dict(os.environ, MYNAH_ASR_TRACE_RNNT="1", MYNAH_ASR_THREADS="4")
    cmd = [binary, "stream", "-m", model, "-i", clip, "--deltas"]
    if quant:
        cmd += ["--quant", quant]
    if lookahead >= 0:
        cmd += ["--lookahead", str(lookahead)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return {"clip": clip, "error": (r.stderr or "")[-300:]}

    blank_id, first_nonblank, blanks_before, nonblank_ct = None, None, 0, 0
    first_frame_audio = None
    for line in r.stderr.splitlines():
        if first_frame_audio is None and " frame=" in line:
            m = FRAME.search(line)
            if m:
                first_frame_audio = float(m.group(2))
        if blank_id is None:
            b = BLANKF.search(line)
            if b:
                blank_id = int(b.group(1))
        m = FRAME.search(line)
        if m and blank_id is not None:
            chose = int(m.group(3))
            if chose == blank_id:
                if first_nonblank is None:
                    blanks_before += 1
            else:
                nonblank_ct += 1
                if first_nonblank is None:
                    first_nonblank = {"frame": int(m.group(1)), "audio_s": float(m.group(2))}

    # The audio time of the chunk that first DECODED a token. Deliberately not
    # compared against `chars_emitted` on the same line: that counter is the
    # state BEFORE this chunk publishes, so a "decoded but unpublished" count
    # built from it reads 1 on every clip and means nothing. The publication gap
    # below is measured instead against the delta the client actually received.
    first_decoded = None
    for line in r.stderr.splitlines():
        m = EMIT.search(line)
        if m and int(m.group(3)) > 0:
            first_decoded = float(m.group(2))
            break

    deltas = []
    for line in r.stdout.splitlines():
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") == "delta" and (o.get("text") or "").strip():
            deltas.append(o)
    first_delta = deltas[0] if deltas else None

    return {"clip": clip, "blank_id": blank_id,
            "first_frame_audio_s": first_frame_audio,
            "first_nonblank": first_nonblank, "blank_decisions_before": blanks_before,
            "nonblank_decisions": nonblank_ct,
            "first_decoded_audio_s": first_decoded,
            "first_delta_t1": first_delta and first_delta.get("t1"),
            "first_delta_text": first_delta and first_delta.get("text"),
            "n_deltas": len(deltas)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clips", required=True, help="file of clip paths, one per line")
    ap.add_argument("--onsets", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--binary", default="./mynah-asr")
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--lookahead", type=int, default=-1)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()

    onsets = json.load(open(a.onsets))
    man = {e["file"].split("/")[-1]: e for e in json.load(open(a.manifest))["clips"]}
    clips = [l.strip() for l in open(a.clips) if l.strip()]
    print(f"{len(clips)} clip(s), {a.model}, quant {a.quant}, "
          f"lookahead {a.lookahead if a.lookahead >= 0 else '(pack default)'}", flush=True)

    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(lambda c: one(a.binary, a.model, c, a.quant, a.lookahead), clips):
            c = r["clip"]
            o = onsets.get(c)
            m = man.get(os.path.basename(c), {})
            r["onset_s"] = o
            r["class"] = m.get("class")
            r["duration_s"] = m.get("duration_sec")
            if o is not None and not r.get("error"):
                fn = r.get("first_nonblank")
                if fn:
                    r["speech_to_first_nonblank_ms"] = (fn["audio_s"] - o) * 1000.0
                if r.get("first_delta_t1") is not None:
                    r["speech_to_first_published_ms"] = (r["first_delta_t1"] - o) * 1000.0
                if fn and r.get("first_delta_t1") is not None:
                    r["publication_gate_ms"] = (r["first_delta_t1"] - fn["audio_s"]) * 1000.0
            rows.append(r)

    def P(key):
        v = sorted(r[key] for r in rows if r.get(key) is not None)
        return {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99),
                "min": min(v) if v else None, "max": max(v) if v else None,
                "mean": sum(v) / len(v) if v else None}

    out = {"model": os.path.basename(os.path.abspath(a.model)), "quant": a.quant,
           "lookahead": a.lookahead, "clips": len(rows),
           "errors": sum(1 for r in rows if r.get("error")),
           "speech_to_first_nonblank_ms": P("speech_to_first_nonblank_ms"),
           "speech_to_first_published_ms": P("speech_to_first_published_ms"),
           "publication_gate_ms": P("publication_gate_ms"),
           "blank_decisions_before_first": P("blank_decisions_before"),
           "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
