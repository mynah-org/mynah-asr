#!/usr/bin/env python3
"""eou_metrics.py — when does the model say "the speaker is done", and is it right.

    python3 tools/eval/eou_metrics.py -m models/parakeet-realtime-eou-120m \
        --manifest samples/eval-bank/manifest.json --root samples/eval-bank \
        --mode both --limit 40 --jobs 4 --json .work/evidence/eou.json
    python3 tools/eval/eou_metrics.py ... --prepare-only      # write the audio, run nothing
    python3 tools/eval/eou_metrics.py ... --from-dir runs/ws/  # score saved WS client output
    python3 tools/eval/eou_metrics.py --self-test

WHY.  parakeet_realtime_eou_120m-v1 emits an end-of-utterance token (`<EOU>`)
from the decoder itself. A turn-taking client acts on it: it stops listening,
sends the text on, starts talking back. So the questions are not about WER but
about time, and each failure costs something different:

  late      speech ended, the EOU came N ms later: the reply waits that long;
  premature an EOU while the speaker is only pausing: the client cuts them off;
  missed    no EOU at all: the client waits for a timeout it should not need;
  multiple  several EOUs for one utterance: a client that acts on each one
            replies twice.

A single "EOU accuracy" number would let these trade against each other, so they
are reported apart, and premature EOUs are reported at SEVERAL pause lengths
instead of one threshold: an EOU inside a 2 s pause may be the right call, one
inside a 0.3 s pause never is, and where between them the line sits is the
caller's policy, not this tool's.

WHERE THE EOU COMES FROM (--source).  `events`: JSON lines {"type":"eou",...}
printed by `mynah-asr stream --deltas` (or a WS server's "eou" frame), time in
"t" (or "audio_s"). `trace`: the fallback that works on any build -- run with
MYNAH_ASR_TRACE_RNNT=1 and read src/decoder.c's per-frame line
`[RNNT] frame=F audio_s=A ... chose=K`. A decision on frame F is taken once the
stream has consumed A seconds of audio (samples_fed at that decode step), so A
is the EOU time a live client would see; F * frame_ms is where in the audio the
deciding frame sits and is used only to say WHICH pause an EOU fell into.
Trace limit: the line is printed for the first symbol of a frame only, so an
<EOU> emitted as the second symbol of a frame that already emitted a word is
invisible to it (the events source has no such hole). `auto` uses events when
the output has any and the trace otherwise. A trace run that parses zero
`[RNNT]` lines is a FAILED clip, not a clip without EOUs: a parser that matches
nothing would otherwise report "missed" for everything (tests/
test_rnnt_trace_parsers.sh tells that story).

An EOU decided on the final feed or at the close-time flush is set apart
(`at_close`) and never counts as detected: a live stream has no close, and this
checkpoint does fire at the flush (the encoder sees end-of-audio padding). To
leave room for a real decision, every clip is played with --tail-s seconds of
noise appended at the clip's own noise floor.

GROUND TRUTH FOR "SPEECH ENDED" (recorded as `gt_method` in the output).
  * Pauses and segment starts: an ENERGY segmentation, the method of
    tools/bench/clip_onset.py (10 ms RMS, floor = 10th percentile, speech =
    12 dB over it), closed only after 100 ms below it, spans under 60 ms
    dropped. It is an APPROXIMATION: it marks loud non-speech as speech, cuts a
    soft word tail a little early, and cannot see a pause shorter than 100 ms.
  * Speech end: with --vad <converted silero dir> each SOURCE clip is first
    streamed once with the repo's Silero VAD endpoint (docs/vad-silero.md);
    the last {"type":"eou","t1":...} it prints -- t1 = where the silence
    started, unpadded, 32 ms grain -- is the clip's speech end, and energy
    segments after it are dropped. This matters: on FLEURS the energy end
    lands on trailing breaths and clicks, up to ~2 s after the last word.
    Without --vad the energy end is used and `gt_method` says "energy".
    Silero's span STARTS are not exposed by the CLI, so onsets stay energy.
  * Composites (A + gap + B): A is cut 50 ms after its energy end and B starts
    50 ms before its energy onset, with noise in between, so the energy-measured
    pause is exactly the nominal gap.

TEXT.  In composite mode the transcript is split at A's end-EOU -- what a
turn-taking client would hand on as utterance A -- and A and B are scored apart
with the normaliser and WER of tools/bench/streaming_metrics.py (the same as
tools/eval/lang_gate.py). The best possible split (`oracle`) is reported beside
it, so the cost of WHERE the EOU fell is separated from recognition errors.

Exit: 0 ok · 1 no clip could be measured · 2 usage · 77 binary, pack, manifest or
      <EOU> token missing.
"""
from __future__ import annotations

import argparse
import array
import concurrent.futures as cf
import json
import math
import os
import random
import re
import statistics as st
import subprocess
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import streaming_metrics as M          # noqa: E402

SR = 16000
PAUSE_X = (0.3, 0.5, 0.8, 1.2, 2.0)       # premature EOUs counted in pauses shorter than X
MISS_N = (1.0, 2.0)                        # missed = no EOU within N s of the speech end
GAPS = (0.3, 0.6, 1.0, 2.0)
CUT_PAD = 0.05
TRACE_RE = re.compile(r"\[RNNT\] frame=(\d+) audio_s=(-?\d+(?:\.\d+)?)\s.*?\bchose=(\d+)\b")
WS_EOU_RE = re.compile(r"\[end of utterance at (\S+) s\]")
WS_DONE_RE = re.compile(r"\[done, language=[^\]]*\]")


# ----------------------------------------------------------------------------- audio
def read_pcm(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2 or w.getframerate() != SR:
            raise ValueError(f"{path}: 16 kHz PCM16 only")
        ch, raw = w.getnchannels(), w.readframes(w.getnframes())
    pcm = array.array("h")
    pcm.frombytes(raw)
    return array.array("h", pcm[::ch]) if ch > 1 else pcm


def write_pcm(path, pcm):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def frame_rms(pcm, frame_ms=10.0):
    step = int(SR * frame_ms / 1000)
    return [math.sqrt(sum(v * v for v in pcm[i:i + step]) / step)
            for i in range(0, len(pcm) - step + 1, step)]


def energy_segments(pcm, frame_ms=10.0, margin_db=12.0, min_speech_ms=60.0,
                    min_silence_ms=100.0):
    """Speech spans [(start_s, end_s)] and the noise-floor RMS. See the module
    docstring for what this is and is not."""
    rms = frame_rms(pcm, frame_ms)
    if not rms:
        return [], 0.0
    floor = max(sorted(rms)[max(0, int(0.10 * len(rms)) - 1)], 1.0)
    thr = floor * 10.0 ** (margin_db / 20.0)
    runs, start = [], None
    for i, v in enumerate(rms + [0.0]):
        if v > thr and start is None:
            start = i
        elif v <= thr and start is not None:
            runs.append([start, i])
            start = None
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) * frame_ms < min_silence_ms:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    f = frame_ms / 1000.0
    segs = [(round(a * f, 3), round(b * f, 3)) for a, b in merged
            if (b - a) * frame_ms >= min_speech_ms]
    return segs, floor


def noise(n, rms, seed):
    """Gaussian noise at the clip's own floor: a gap of digital zeros is a signal
    no microphone produces, and the encoder may treat it as one."""
    r = random.Random(seed)
    return array.array("h", (max(-32768, min(32767, int(r.gauss(0.0, rms)))) for _ in range(n)))


def seed_of(name):
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) & 0x7FFFFFFF


# ------------------------------------------------------------------------ parsing
def _event_time(o):
    for k in ("t", "audio_s"):
        if isinstance(o.get(k), (int, float)):
            return float(o[k])
    return None


def parse_json_lines(text):
    """CLI --deltas output (or WS frames saved one per line) -> ordered items,
    plus the Silero endpoint ends. A model EOU carries "t"/"audio_s" (or says
    source "model"); the VAD endpoint carries only "t1" (cli/main.c)."""
    items, ends, final, fed = [], [], None, None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        kind = o.get("type")
        if kind == "delta" or (kind is None and "text" in o):
            items.append({"kind": "delta", "t": o.get("fed_s", o.get("audio_s")),
                          "text": o.get("text", "")})
        elif kind == "eou":
            src = o.get("source")
            if src == "vad" or (src is None and _event_time(o) is None and "t1" in o):
                ends.append(float(o["t1"]))
            else:
                t = _event_time(o)
                items.append({"kind": "eou", "t": t, "t_loc": t, "src": "event"})
        elif kind in ("final", "done"):
            final = o.get("text", final)
            fed = o.get("fed_s", o.get("audio_s", fed))
    return items, ends, final, fed


def parse_trace(text, eou_id, eob_id=None, frame_ms=80.0):
    """MYNAH_ASR_TRACE_RNNT=1 stderr -> (EOU items, EOB count, lines parsed)."""
    eous, n_eob, n = [], 0, 0
    for line in text.splitlines():
        m = TRACE_RE.search(line)
        if not m:
            continue
        n += 1
        frame, audio_s, chose = int(m.group(1)), float(m.group(2)), int(m.group(3))
        if chose == eou_id:
            eous.append({"kind": "eou", "t": audio_s, "t_loc": frame * frame_ms / 1000.0,
                         "frame": frame, "src": "trace"})
        elif eob_id is not None and chose == eob_id:
            n_eob += 1
    return eous, n_eob, n


def parse_ws_text(text):
    """tools/eval/ws_client.py stdout: text printed as it arrives, with
    `[end of utterance at T s]` where an eou frame came. The marker's place in
    the text is the split, so no delta timing is needed."""
    items, pos = [], 0
    body = WS_DONE_RE.sub("", text)
    for m in WS_EOU_RE.finditer(body):
        if body[pos:m.start()].strip():
            items.append({"kind": "delta", "t": None, "text": body[pos:m.start()]})
        try:
            t = float(m.group(1))
        except ValueError:
            t = None
        items.append({"kind": "eou", "t": t, "t_loc": t, "src": "ws"})
        pos = m.end()
    if body[pos:].strip():
        items.append({"kind": "delta", "t": None, "text": body[pos:]})
    return items


def merge(items, eous):
    """Put trace EOUs into the delta sequence by audio time. A delta published at
    the same fed time as an EOU was decoded in the same step, before it."""
    out = [dict(x, _k=(x["t"] if x["t"] is not None else -1.0, 0, i))
           for i, x in enumerate(items) if x["kind"] == "delta"]
    out += [dict(e, _k=(e["t"], 1, i)) for i, e in enumerate(eous)]
    out.sort(key=lambda x: x["_k"])
    for x in out:
        del x["_k"]
    return out


def text_of(items):
    return "".join(x["text"] for x in items if x["kind"] == "delta")


# ------------------------------------------------------------------------ scoring
def pause_at(segs, t):
    """Length of the pause t falls in; 0.0 inside speech; None before the first
    segment or after the last."""
    for i, (s, e) in enumerate(segs):
        if s <= t <= e:
            return 0.0
        if i + 1 < len(segs) and e < t < segs[i + 1][0]:
            return round(segs[i + 1][0] - e, 3)
    return None


def score_utterances(utts, eous, close_t=None, tol=0.2, pause_x=PAUSE_X, miss_n=MISS_N):
    """`utts`: per utterance its speech segments [(s, e)] (one utterance per
    plain clip, two in a composite). `eous`: items with t (decision time) and
    t_loc (where in the audio). Returns one dict per utterance.

    end-EOU of k: the first EOU with t >= end_k - tol, and before both the next
    utterance's end - tol and its start + max(miss_n): a decision taken that
    long into the next utterance is about the next utterance. Premature for k: an EOU in k's window, after k's
    first segment started, with t < end_k - tol. Multiple: EOUs after k's
    end-EOU and before the next utterance starts (or ever, for the last one)."""
    live = sorted((e for e in eous if e.get("t") is not None
                   and not (close_t is not None and e["t"] >= close_t - 1e-4)),
                  key=lambda e: e["t"])
    at_close = sum(1 for e in eous if e.get("t") is not None and close_t is not None
                   and e["t"] >= close_t - 1e-4)
    out, prev_anchor = [], -1e9
    for k, segs in enumerate(utts):
        end = segs[-1][1]
        nxt_start = utts[k + 1][0][0] if k + 1 < len(utts) else 1e9
        nxt_end = (min(utts[k + 1][-1][1] - tol, nxt_start + max(miss_n))
                   if k + 1 < len(utts) else 1e9)
        after = [e for e in live if end - tol <= e["t"] < nxt_end]
        first = after[0] if after else None
        mult = [e for e in after if e["t"] < nxt_start] if first is not None else []
        if first is not None and first not in mult:
            mult = [first]
        window = [e for e in live if prev_anchor < e["t"] < end - tol]
        before_speech = [e for e in window if e.get("t_loc", e["t"]) < segs[0][0]]
        prem = [e for e in window if e not in before_speech]
        pauses = [pause_at(segs, e.get("t_loc", e["t"])) for e in prem]
        pauses = [0.0 if p is None else p for p in pauses]
        lat = None if first is None else round((first["t"] - end) * 1000.0, 1)
        r = {"start": segs[0][0], "end": end, "segments": segs,
             "eou_t": None if first is None else first["t"], "latency_ms": lat,
             "n_after_end": len(mult), "multiple": len(mult) > 1,
             "premature": len(prem), "premature_in_speech": sum(1 for p in pauses if p == 0.0),
             "premature_pauses_s": pauses,
             "premature_by_pause": {f"{x:g}": sum(1 for p in pauses if p < x) for x in pause_x},
             "before_speech": len(before_speech),
             "missed": {**{f"{n:g}": lat is None or lat > n * 1000.0 for n in miss_n},
                        "never": first is None}}
        if k + 1 < len(utts):
            r["eou_before_next"] = first is not None and first["t"] < nxt_start
        out.append(r)
        # the next window opens after this utterance's EOUs that land before the
        # next speech; with none, after this utterance's end
        prev_anchor = mult[-1]["t"] if mult else end - tol
    return out, at_close


def split_wer(hyp_a, hyp_b, ref_a, ref_b):
    return {"wer_a": M.wer(hyp_a, ref_a), "wer_b": M.wer(hyp_b, ref_b)}


def oracle_split(hyp, ref_a, ref_b):
    """The split point of the hypothesis words that minimises the total edits:
    what A/B would score with an EOU in the best possible place."""
    h = M.normalise(hyp).split()
    ra, rb = M.normalise(ref_a).split(), M.normalise(ref_b).split()
    best = None
    for i in range(len(h) + 1):
        ea, eb = M.edit_distance(h[:i], ra), M.edit_distance(h[i:], rb)
        if best is None or ea + eb < best[0]:
            best = (ea + eb, i, ea, eb)
    _, i, ea, eb = best
    return {"wer_a": ea / len(ra) if ra else None, "wer_b": eb / len(rb) if rb else None,
            "split_word": i}


def score_composite(meta, items, close_t, tol):
    utts = [meta["a_segments"], meta["b_segments"]]
    eous = [x for x in items if x["kind"] == "eou"]
    u, at_close = score_utterances(utts, eous, close_t, tol)
    a, b = u
    a_t = a["eou_t"]
    if a_t is not None:
        idx = next(i for i, x in enumerate(items) if x["kind"] == "eou" and x["t"] == a_t)
        ha, hb, how = text_of(items[:idx]), text_of(items[idx + 1:]), "eou"
    else:
        ha, hb, how = None, None, "none"
    full = text_of(items)
    r = {"gap_s": meta["gap_s"], "a": a, "b": b, "at_close": at_close,
         "eou_in_gap": a["eou_before_next"], "a_latency_ms": a["latency_ms"],
         "split": how, "hyp_a": ha, "hyp_b": hb,
         "oracle": oracle_split(full, meta["ref_a"], meta["ref_b"]),
         "wer_full": M.wer(full, meta["ref_a"] + " " + meta["ref_b"])}
    r.update(split_wer(ha, hb, meta["ref_a"], meta["ref_b"]) if ha is not None
             else {"wer_a": None, "wer_b": None})
    return r


def summarise(utts, extra=None):
    """Aggregate over utterance dicts from score_utterances."""
    n = len(utts)
    lat = [u["latency_ms"] for u in utts if u["latency_ms"] is not None]
    rate = lambda c: round(c / n, 4) if n else None  # noqa: E731
    s = {"n": n,
         "latency_ms": {"p50": M.pct(lat, 50), "p90": M.pct(lat, 90), "p95": M.pct(lat, 95),
                        "mean": round(st.mean(lat), 1) if lat else None, "n": len(lat)},
         "premature_rate": rate(sum(1 for u in utts if u["premature"])),
         "premature_eous": sum(u["premature"] for u in utts),
         "premature_by_pause": {x: rate(sum(1 for u in utts if u["premature_by_pause"][x]))
                                for x in (utts[0]["premature_by_pause"] if utts else {})},
         "premature_in_speech_rate": rate(sum(1 for u in utts if u["premature_in_speech"])),
         "before_speech_rate": rate(sum(1 for u in utts if u["before_speech"])),
         "missed_rate": {k: rate(sum(1 for u in utts if u["missed"][k]))
                         for k in (utts[0]["missed"] if utts else {})},
         "multiple_rate": rate(sum(1 for u in utts if u["multiple"]))}
    if extra:
        s.update(extra)
    return s


def summarise_gap(rows):
    a = [r["a"] for r in rows]
    s = summarise(a)
    m = lambda v: round(st.mean(v), 4) if v else None  # noqa: E731
    s.update({"eou_in_gap_rate": round(sum(1 for r in rows if r["eou_in_gap"]) / len(rows), 4),
              "wer_a": m([r["wer_a"] for r in rows if r["wer_a"] is not None]),
              "wer_b": m([r["wer_b"] for r in rows if r["wer_b"] is not None]),
              "wer_a_oracle": m([r["oracle"]["wer_a"] for r in rows
                                 if r["oracle"]["wer_a"] is not None]),
              "wer_b_oracle": m([r["oracle"]["wer_b"] for r in rows
                                 if r["oracle"]["wer_b"] is not None]),
              "b_summary": summarise([r["b"] for r in rows])})
    return s


# ----------------------------------------------------------------------- preparing
def prepare(samples, root, work, mode, gaps, tail_s, limit, a=None):
    """Write the played audio into `work` and return the jobs. Nothing here is a
    repo file: the bank is CC-BY 4.0 FLEURS and stays out of git (rule 6)."""
    jobs, cache = [], {}

    def load(s):
        if s["file"] not in cache:
            pcm = read_pcm(os.path.join(root, s["file"]))
            segs, floor = energy_segments(pcm)
            cache[s["file"]] = (pcm, segs, floor)
        return cache[s["file"]]

    usable = []
    for s in samples:
        try:
            if load(s)[1]:
                usable.append(s)
        except (OSError, ValueError, wave.Error, EOFError) as e:
            print(f"  skip {s['file']}: {e}", file=sys.stderr)
    if limit:                                  # a pair needs two clips
        usable = usable[: limit if mode == "single" else 2 * limit]
    method = {}
    if a is not None and a.vad:
        with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            ends = list(ex.map(lambda s: vad_ends(a, os.path.join(root, s["file"])), usable))
        for s, e in zip(usable, ends):
            pcm, segs, floor = cache[s["file"]]
            segs, method[s["file"]] = with_vad_end(segs, e)
            cache[s["file"]] = (pcm, segs, floor)
    if mode in ("single", "both"):
        for s in usable[:limit] if limit else usable:
            pcm, segs, floor = load(s)
            stem = os.path.splitext(os.path.basename(s["file"]))[0]
            out = os.path.join(work, "single", stem + ".wav")
            pcm2 = pcm + noise(int(tail_s * SR), floor, seed_of(stem))
            write_pcm(out, pcm2)
            jobs.append({"kind": "single", "name": stem, "wav": out, "ref": s.get("text", ""),
                         "segments": segs, "end_method": method.get(s["file"], "energy"),
                         "audio_s": round(len(pcm2) / SR, 3),
                         "speech_audio_s": round(len(pcm) / SR, 3)})
    if mode in ("composite", "both"):
        pairs = [(usable[i], usable[i + 1]) for i in range(0, len(usable) - 1, 2)]
        for sa, sb in pairs[:limit] if limit else pairs:
            pa, ga, fa = load(sa)
            pb, gb, fb = load(sb)
            cut_a = min(len(pa), int((ga[-1][1] + CUT_PAD) * SR))
            from_b = max(0, int((gb[0][0] - CUT_PAD) * SR))
            na = os.path.splitext(os.path.basename(sa["file"]))[0]
            nb = os.path.splitext(os.path.basename(sb["file"]))[0]
            for g in gaps:
                name = f"g{int(round(g * 1000))}_{na}__{nb}"
                fill = noise(max(0, int((g - 2 * CUT_PAD) * SR)), min(fa, fb), seed_of(name))
                pcm2 = pa[:cut_a] + fill + pb[from_b:]
                off = (cut_a + len(fill) - from_b) / SR
                b_segs = [(round(s + off, 3), round(e + off, 3)) for s, e in gb]
                pcm2 += noise(int(tail_s * SR), min(fa, fb), seed_of(name) + 1)
                out = os.path.join(work, "composite", name + ".wav")
                write_pcm(out, pcm2)
                jobs.append({"kind": "composite", "name": name, "wav": out, "gap_s": g,
                             "ref_a": sa.get("text", ""), "ref_b": sb.get("text", ""),
                             "a_segments": [tuple(x) for x in ga], "b_segments": b_segs,
                             "a_file": sa["file"], "b_file": sb["file"],
                             "end_method": [method.get(sa["file"], "energy"),
                                            method.get(sb["file"], "energy")],
                             "audio_s": round(len(pcm2) / SR, 3)})
    with open(os.path.join(work, "eou_jobs.json"), "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=1)
    return jobs


# -------------------------------------------------------------------------- running
def run_cli(job, a, eou_id, eob_id):
    cmd = [a.binary, "stream", "-m", a.model, "-i", job["wav"], "--quant", a.quant,
           "--deltas"] + a.extra_list
    env = dict(os.environ)
    if a.source in ("auto", "trace"):
        env["MYNAH_ASR_TRACE_RNNT"] = "1"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=a.timeout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if p.returncode != 0:
        return {"error": f"rc={p.returncode}: {p.stderr.strip().splitlines()[-1:]}"}
    return interpret(p.stdout, p.stderr, a.source, eou_id, eob_id, a.frame_ms)


def interpret(stdout, stderr, source, eou_id, eob_id, frame_ms):
    items, _, _, fed = parse_json_lines(stdout)
    ev = [x for x in items if x["kind"] == "eou"]
    if source == "events" or (source == "auto" and ev):
        return {"items": items, "fed_s": fed, "source": "events"}
    tr, n_eob, n = parse_trace(stderr or "", eou_id, eob_id, frame_ms)
    if n == 0:
        return {"error": "trace mode but no [RNNT] line parsed (trace off, or its format "
                         "changed: see tests/test_rnnt_trace_parsers.sh)"}
    deltas = [x for x in items if x["kind"] == "delta"]
    return {"items": merge(deltas, tr), "fed_s": fed,
            "source": "trace", "n_eob": n_eob, "trace_lines": n}


def vad_ends(a, wav):
    """Silero endpoint ends of one source clip: `stream --vad` prints
    {"type":"eou","t1":...} each time a speech span closes, t1 = where the
    silence started (unpadded, 32 ms grain; src/mynah_asr.c). The decode it rides
    on is incidental; the VAD does not touch it (docs/vad-silero.md)."""
    try:
        p = subprocess.run([a.binary, "stream", "-m", a.model, "-i", wav, "--quant", a.quant,
                            "--deltas", "--vad", a.vad], capture_output=True, text=True,
                           timeout=a.timeout)
    except subprocess.TimeoutExpired:
        return []
    return parse_json_lines(p.stdout)[1] if p.returncode == 0 else []


def with_vad_end(segs, ends):
    """Energy segments cut at the last Silero end: what the energy method calls
    speech after it (a breath, a click, a chair) is dropped. (segs, method)."""
    if not ends or not segs:
        return segs, "energy"
    end = ends[-1]
    out = [(s, min(e, end)) for s, e in segs if s < end]
    if not out:
        return segs, "energy"
    out[-1] = (out[-1][0], round(max(out[-1][1], end), 3))
    return out, "silero-vad"


def score_job(job, res, tol):
    if "error" in res:
        return {"name": job["name"], "kind": job["kind"], "error": res["error"]}
    close_t = res.get("fed_s") or job.get("audio_s")
    if job["kind"] == "single":
        segs = job["segments"]
        eous = [x for x in res["items"] if x["kind"] == "eou"]
        (u,), at_close = score_utterances([segs], eous, close_t, tol)
        text = text_of(res["items"])
        return {"name": job["name"], "kind": "single", "source": res["source"],
                "end_method": job["end_method"], "utt": u, "at_close": at_close,
                "eou_times": [x["t"] for x in eous], "wer": M.wer(text, job["ref"]),
                "text": text}
    r = score_composite(job, res["items"], close_t, tol)
    r.update({"name": job["name"], "kind": "composite", "source": res["source"],
              "end_method": job["end_method"],
              "eou_times": [x["t"] for x in res["items"] if x["kind"] == "eou"]})
    return r


# --------------------------------------------------------------------------- report
def fmt(v, nd=0, pct=False):
    if v is None:
        return "  -  "
    return f"{100 * v:5.1f}%" if pct else f"{v:.{nd}f}"


def report_block(title, s):
    L = s["latency_ms"]
    print(f"\n=== {title}   n={s['n']}")
    print(f"  speech end -> EOU  p50 {fmt(L['p50'])} ms  p90 {fmt(L['p90'])} ms  "
          f"p95 {fmt(L['p95'])} ms  (n={L['n']})")
    print(f"  missed  " + "  ".join(f"{k}{'' if k == 'never' else ' s'}: "
                                    f"{fmt(v, pct=True)}" for k, v in s["missed_rate"].items()))
    print(f"  premature (any) {fmt(s['premature_rate'], pct=True)}  in speech "
          f"{fmt(s['premature_in_speech_rate'], pct=True)}  in pause < X: " + "  ".join(
              f"{k}s {fmt(v, pct=True)}" for k, v in s["premature_by_pause"].items()))
    print(f"  multiple EOUs {fmt(s['multiple_rate'], pct=True)}   "
          f"EOU before any speech {fmt(s['before_speech_rate'], pct=True)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("-m", "--model")
    ap.add_argument("--quant", default="f32")
    ap.add_argument("--binary", default="./mynah-asr")
    ap.add_argument("--manifest", default="samples/eval-bank/manifest.json")
    ap.add_argument("--root", default="samples/eval-bank")
    ap.add_argument("--langs", default="en", help="comma list of manifest languages")
    ap.add_argument("--mode", choices=("single", "composite", "both"), default="both")
    ap.add_argument("--gaps", default=",".join(f"{g:g}" for g in GAPS))
    ap.add_argument("--work-dir", default=".work/evidence/eou-audio",
                    help="where the played audio is written (not in git)")
    ap.add_argument("--tail-s", type=float, default=3.0,
                    help="noise appended after each clip so a live EOU has room")
    ap.add_argument("--tol", type=float, default=0.2,
                    help="an EOU up to this many s before the measured end is not premature")
    ap.add_argument("--source", choices=("auto", "events", "trace"), default="auto")
    ap.add_argument("--vad", help="Silero VAD dir: utterance ends from its endpoint")
    ap.add_argument("--frame-ms", type=float, default=80.0, help="encoder frame period")
    ap.add_argument("--eou-id", type=int, help="default: '<EOU>' in the pack's tokens.json")
    ap.add_argument("--extra", default="", help="extra CLI flags, e.g. '--chunk-ms 160'")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, help="clips (single) and pairs (composite)")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--from-dir", help="score saved outputs <name>.txt|.jsonl|.log "
                                       "(CLI --deltas or tools/eval/ws_client.py) for the "
                                       "jobs in <work-dir>/eou_jobs.json")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    a.extra_list = a.extra.split() if a.extra else []
    try:
        gaps = [float(g) for g in a.gaps.split(",") if g]
    except ValueError:
        print("eou_metrics: --gaps is a comma list of seconds", file=sys.stderr)
        return 2

    eou_id, eob_id = a.eou_id, None
    if not a.from_dir:
        if not a.model:
            print("eou_metrics: -m <pack> is required", file=sys.stderr)
            return 2
        if not os.path.exists(a.binary) and not a.prepare_only:
            print(f"eou_metrics: no binary at {a.binary} — SKIP", file=sys.stderr)
            return 77
        tok = os.path.join(a.model, "tokens.json")
        if not os.path.exists(tok):
            print(f"eou_metrics: no pack at {a.model} — SKIP", file=sys.stderr)
            return 77
        vocab = json.load(open(tok, encoding="utf-8"))
        if eou_id is None:
            eou_id = vocab.index("<EOU>") if "<EOU>" in vocab else None
        eob_id = vocab.index("<EOB>") if "<EOB>" in vocab else None
        if eou_id is None and a.source != "events":
            print(f"eou_metrics: {a.model} has no <EOU> token — SKIP", file=sys.stderr)
            return 77

    if a.from_dir:
        jp = os.path.join(a.work_dir, "eou_jobs.json")
        if not os.path.exists(jp):
            print(f"eou_metrics: no {jp} (run --prepare-only first) — SKIP", file=sys.stderr)
            return 77
        jobs = json.load(open(jp, encoding="utf-8"))
    else:
        if not os.path.exists(a.manifest):
            print(f"eou_metrics: no manifest at {a.manifest} — SKIP", file=sys.stderr)
            return 77
        man = json.load(open(a.manifest, encoding="utf-8"))
        keep = set(a.langs.split(","))
        samples = [s for s in man["samples"] if s.get("lang") in keep]
        if a.vad and not os.path.exists(os.path.join(a.vad, "mynah.json")):
            print(f"eou_metrics: no converted Silero VAD at {a.vad} (make test-vad) — SKIP",
                  file=sys.stderr)
            return 77
        jobs = prepare(samples, a.root, a.work_dir, a.mode, gaps, a.tail_s, a.limit, a)
        print(f"prepared {len(jobs)} clip(s) in {a.work_dir}", file=sys.stderr)
        if a.prepare_only:
            return 0

    results = []
    if a.from_dir:
        for job in jobs:
            path = next((os.path.join(a.from_dir, job["name"] + x)
                         for x in (".jsonl", ".txt", ".log")
                         if os.path.exists(os.path.join(a.from_dir, job["name"] + x))), None)
            if not path:
                continue
            raw = open(path, encoding="utf-8").read()
            if any(line.lstrip().startswith("{") for line in raw.splitlines()):
                res = interpret(raw, "", "events", eou_id, eob_id, a.frame_ms)
            else:
                res = {"items": parse_ws_text(raw), "fed_s": None,
                       "source": "ws-text"}
            results.append(score_job(job, res, a.tol))
    else:
        with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futs = [ex.submit(run_cli, j, a, eou_id, eob_id) for j in jobs]
            for j, f in zip(jobs, futs):
                results.append(score_job(j, f.result(), a.tol))

    failed = [r for r in results if "error" in r]
    for r in failed[:5]:
        print(f"  FAILED {r['name']}: {r['error']}", file=sys.stderr)
    ok = [r for r in results if "error" not in r]
    singles = [r for r in ok if r["kind"] == "single"]
    comps = [r for r in ok if r["kind"] == "composite"]
    methods = sorted({r["end_method"] for r in singles} |
                     {m for r in comps for m in r["end_method"]})
    out = {"tool": "eou_metrics", "model": a.model, "quant": a.quant,
           "source": sorted({r["source"] for r in ok}), "eou_id": eou_id,
           "gt_method": {"utterance_end": methods,
                         "segments": "energy (clip_onset method: 10 ms RMS, 12 dB over the "
                                     "10th-percentile floor, 100 ms min silence, 60 ms min "
                                     "speech) -- an approximation",
                         "composite": f"A cut {CUT_PAD}s after its energy end, B from "
                                      f"{CUT_PAD}s before its onset, noise in between: "
                                      "energy pause == nominal gap"},
           "params": {"tol_s": a.tol, "tail_s": a.tail_s, "frame_ms": a.frame_ms,
                      "pause_x_s": list(PAUSE_X), "miss_n_s": list(MISS_N), "gaps_s": gaps,
                      "extra": a.extra_list, "vad": a.vad},
           "failed": len(failed), "single": None, "composite": None}
    print(f"\nEOU metrics  model={a.model or a.from_dir}  source={','.join(out['source'])}  "
          f"speech end={'/'.join(methods) or '-'}  tol={a.tol}s  failed={len(failed)}")
    if singles:
        s = summarise([r["utt"] for r in singles],
                      {"at_close_rate": round(sum(1 for r in singles if r["at_close"])
                                              / len(singles), 4)})
        out["single"] = {"summary": s, "clips": singles}
        report_block("single clips", s)
        print(f"  EOU only at stream close: {fmt(s['at_close_rate'], pct=True)} "
              "(not counted as detected)")
    if comps:
        by = {}
        for r in comps:
            by.setdefault(f"{r['gap_s']:g}", []).append(r)
        out["composite"] = {"by_gap": {g: summarise_gap(rs) for g, rs in by.items()},
                            "pairs": comps}
        print("\n=== composites A + gap + B   (A's end-EOU; WER split at it | oracle split)")
        print("  gap   n  EOU in gap  lat p50/p90/p95 ms      premature  multiple  "
              "WER A  WER B | oracle A  B   | B end->EOU p50  B missed@2s")
        for g, s in sorted(out["composite"]["by_gap"].items(), key=lambda kv: float(kv[0])):
            L = s["latency_ms"]
            print(f"  {g:>4} {s['n']:3}  {fmt(s['eou_in_gap_rate'], pct=True):>9}  "
                  f"{fmt(L['p50']):>5}/{fmt(L['p90']):>5}/{fmt(L['p95']):>5}  "
                  f"{fmt(s['premature_rate'], pct=True):>13}  {fmt(s['multiple_rate'], pct=True):>7}  "
                  f"{fmt(s['wer_a'], 3):>5}  {fmt(s['wer_b'], 3):>5} | "
                  f"{fmt(s['wer_a_oracle'], 3):>8}  {fmt(s['wer_b_oracle'], 3)} | "
                  f"{fmt(s['b_summary']['latency_ms']['p50']):>9} ms  "
                  f"{fmt(s['b_summary']['missed_rate'].get('2'), pct=True):>10}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nwrote {a.json}")
    return 0 if ok else 1


# ------------------------------------------------------------------------ self-test
def _eq(got, want, what, tol=1e-6):
    if isinstance(want, (str, list, tuple, dict, bool)) or want is None:
        ok = got == want
    else:
        ok = got is not None and abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: got {got!r}, want {want!r}")
    return 0 if ok else 1


def self_test():
    """Known answers, by hand, no model. The trace line is REAL output of the
    current emitter (parakeet-realtime-eou-120m, f32, fleurs_1521 at the close)."""
    bad = 0
    # 1. energy segmentation: 0.5 s floor, 1.0 s loud, 0.4 s floor, 0.8 s loud, 1 s floor
    r = random.Random(7)
    def blk(sec, amp):
        return array.array("h", (int(r.gauss(0, amp)) for _ in range(int(sec * SR))))
    pcm = blk(0.5, 30) + blk(1.0, 3000) + blk(0.4, 30) + blk(0.8, 3000) + blk(1.0, 30)
    segs, _ = energy_segments(pcm)
    bad += _eq(segs, [(0.5, 1.5), (1.9, 2.7)], "energy segments")
    segs2, _ = energy_segments(blk(0.5, 30) + blk(0.3, 3000) + blk(0.05, 30) + blk(0.3, 3000)
                               + blk(0.5, 30))
    bad += _eq(len(segs2), 1, "a 50 ms dip is not a pause")

    # 2. the trace parser, on a real line, a blank line and a torn line
    real = ("[RNNT] frame=92 audio_s=7.4200 enc=116.00 joint=118.13 pred=MOVED post=0 "
            "blank=1026:-878.8698:r2 wmark=996:-1058.7799:r151 lex=1024:-775.7393:r1 "
            "nb=1024:-775.7393:r1 chose=1024 margin_lex=-103.1304 margin_nb=-103.1304")
    blank = real.replace("frame=92", "frame=91").replace("chose=1024", "chose=1026")
    torn = "[RNNT] frame=93 audio_s=7.4200 enc=1.0 joint="
    eo, neob, n = parse_trace("\n".join([real, blank, torn, "[RNNT] emit q=2 x"]), 1024, 1025)
    bad += _eq(n, 2, "trace lines parsed (torn line skipped)")
    bad += _eq([(e["t"], e["frame"]) for e in eo], [(7.42, 92)], "trace EOU time and frame")
    bad += _eq(eo[0]["t_loc"], 92 * 0.08, "trace EOU location = frame * 80 ms")
    bad += _eq(interpret("", "", "trace", 1024, 1025, 80.0).get("error", "")[:10],
               "trace mode", "zero trace lines is a failure, not 'no EOU'")

    # 3. JSON lines: model EOU vs Silero endpoint, order kept
    out = "\n".join([
        '{"type":"delta","i":0,"t0":0,"t1":0.6,"fed_s":0.6,"lang":null,"text":"the"}',
        '{"type":"eou","t1":1.9000,"fed_s":2.1000}',
        '{"type":"eou","t":2.2,"frame":26}',
        '{"type":"delta","i":1,"t0":0.6,"t1":2.5,"fed_s":2.5,"lang":null,"text":" cat"}',
        '{"type":"final","deltas":2,"fed_s":3.0,"text":"the cat"}'])
    it, vad, final, fed = parse_json_lines(out)
    bad += _eq([x["kind"] for x in it], ["delta", "eou", "delta"], "event order")
    bad += _eq(vad, [1.9], "VAD endpoint (t1 only) is ground truth, not a model EOU")
    bad += _eq(fed, 3.0, "close time from the final line")

    # 4. ws_client text
    ws = "hello world\n[end of utterance at 2.5 s]\n how are you\n[done, language=en]\n"
    it = parse_ws_text(ws)
    bad += _eq([x["kind"] for x in it], ["delta", "eou", "delta"], "ws_client text items")
    bad += _eq(it[1]["t"], 2.5, "ws_client EOU time")

    # 5. one utterance, two segments 0.5-1.5 and 1.9-2.7 (a 0.4 s pause)
    segs = [(0.5, 1.5), (1.9, 2.7)]
    E = lambda t, loc=None: {"kind": "eou", "t": t, "t_loc": t if loc is None else loc}  # noqa
    (u,), cl = score_utterances([segs], [E(1.75, 1.7), E(3.0), E(3.2), E(5.7)], close_t=5.7)
    bad += _eq(u["latency_ms"], 300.0, "latency = first EOU after end - end")
    bad += _eq(u["premature"], 1, "the EOU in the 0.4 s pause is premature")
    bad += _eq(u["premature_by_pause"], {"0.3": 0, "0.5": 1, "0.8": 1, "1.2": 1, "2": 1},
               "premature counted by pause length")
    bad += _eq(u["multiple"], True, "two EOUs after the end")
    bad += _eq(cl, 1, "the EOU at close is set apart")
    bad += _eq(u["missed"], {"1": False, "2": False, "never": False}, "not missed")
    (u,), _ = score_utterances([segs], [E(0.2), E(2.3, 2.1), E(4.2), E(5.7)], close_t=5.7)
    bad += _eq((u["before_speech"], u["premature"], u["premature_in_speech"]), (1, 1, 1),
               "leading-silence EOU apart; EOU located inside speech")
    bad += _eq(u["missed"], {"1": True, "2": False, "never": False}, "late by 1.5 s")
    (u,), cl = score_utterances([segs], [E(5.7)], close_t=5.7)
    bad += _eq((u["missed"]["never"], u["latency_ms"], cl), (True, None, 1),
               "only at close = never")

    # 6. composite A 0.2-1.0, B 1.6-2.4 (0.6 s gap)
    utts = [[(0.2, 1.0)], [(1.6, 2.4)]]
    (ua, ub), _ = score_utterances(utts, [E(1.3), E(1.4), E(2.9)], close_t=6.0)
    bad += _eq((ua["latency_ms"], ua["eou_before_next"], ua["n_after_end"]), (300.0, True, 2),
               "EOU in the gap, twice")
    bad += _eq((ub["latency_ms"], ub["premature"]), (500.0, 0), "B's end-EOU; gap EOUs not B's")
    (ua, ub), _ = score_utterances(utts, [E(1.8), E(2.0), E(2.9)], close_t=6.0)
    bad += _eq((ua["latency_ms"], ua["eou_before_next"], ua["multiple"]), (800.0, False, False),
               "late A EOU lands in B")
    bad += _eq(ub["premature"], 1, "a second EOU inside B is B's premature")
    (ua, ub), _ = score_utterances([[(0.2, 1.0)], [(1.6, 6.0)]], [E(4.0)], close_t=9.0)
    bad += _eq((ua["eou_t"], ub["premature"]), (None, 1),
               "an EOU 2.4 s into B is B's (premature), not a very late A's")
    bad += _eq(with_vad_end([(0.5, 1.5), (1.9, 2.7), (3.5, 3.8)], [1.5, 2.6]),
               ([(0.5, 1.5), (1.9, 2.6)], "silero-vad"), "energy segments cut at the VAD end")

    # 7. text split at A's EOU, and the oracle split
    meta = {"gap_s": 0.6, "a_segments": utts[0], "b_segments": utts[1],
            "ref_a": "The cat.", "ref_b": "Sat down!"}
    items = [{"kind": "delta", "t": 0.9, "text": "the cat"}, E(1.3),
             {"kind": "delta", "t": 2.0, "text": " sat down"}, E(2.9)]
    c = score_composite(meta, items, 6.0, 0.2)
    bad += _eq((c["wer_a"], c["wer_b"], c["eou_in_gap"]), (0.0, 0.0, True), "split at the EOU")
    items = [{"kind": "delta", "t": 0.9, "text": "the"}, E(1.3),
             {"kind": "delta", "t": 2.0, "text": " cat sat down"}]
    c = score_composite(meta, items, 6.0, 0.2)
    bad += _eq((c["wer_a"], c["wer_b"], c["oracle"]["split_word"]), (0.5, 0.5, 2),
               "an early EOU costs words on both sides; the oracle does not")
    tr = merge([{"kind": "delta", "t": 1.3, "text": "x"}, {"kind": "delta", "t": 1.4,
                                                          "text": "y"}], [E(1.3)])
    bad += _eq([x.get("text", "EOU") for x in tr], ["x", "EOU", "y"],
               "a delta at the EOU's fed time precedes it")

    s = summarise([u])
    bad += _eq(s["n"], 1, "summary over one utterance")
    print(f"\neou_metrics self-test: {'FAIL' if bad else 'PASS'} ({bad} failures)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
