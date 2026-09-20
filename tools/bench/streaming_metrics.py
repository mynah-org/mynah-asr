#!/usr/bin/env python3
"""streaming_metrics.py — the single definition of every streaming ASR metric.

    python3 tools/bench/streaming_metrics.py --self-test

Every harness that measures `mynah-asr-server` streaming imports this module instead of
computing its own percentiles (`tools/bench/stream_load.py` today).  Two harnesses that
disagree about what "emission lag p95" means produce two numbers that cannot be compared
across hosts or commits, which is the whole point of `ENGINEERING.md` §8.

Standard library only: these harnesses run on a bare Linux box with no venv.

--------------------------------------------------------------------------------- input
One utterance is a plain dict (picklable, JSON-serialisable), recorded by the client:

    {
      "clip":     str,                 # path of the WAV played
      "audio_s":  float,               # its duration
      "sends":    [[t_send, cum_audio_s], ...],   # monotonic completion time of each frame
                                       #   and the audio seconds sent INCLUDING that frame
      "late_ms":  [float, ...],        # per frame: t_send - its scheduled due time
      "events":   [{"t": float,        # monotonic arrival of a server frame
                    "type": "delta"|"eou"|"done"|"error",
                    "audio_s": float|None,   # audio the server had consumed (v2 `audio_s`)
                    "lag_ms":  float|None,   # the server's own lag for that frame
                    "text":    str}, ...],
      "done_t":   float|None,          # arrival of the `done` frame
      "t_start":  float,               # monotonic start of the utterance (windowing)
      "error":    str|None,
      "rejected": bool,                # HTTP 503 before the upgrade: a legitimate outcome
    }

`sends[0][0]` is the first-audio-sent time.  Nothing here reads a socket or a clock: the
module is pure arithmetic over that record, which is why it can have a known-answer test.

------------------------------------------------------------------------------- metrics
All per utterance, milliseconds unless stated.

* TTFB                first SERVER FRAME OF ANY KIND arrival - first-audio-sent time.
                      "the server is answering". Distinct from TTFP on purpose: a stream
                      can be acknowledged long before it has produced a word, and when the
                      two separate it says WHERE the wait is -- transport and admission on
                      one side, the model's own emission delay on the other.
* TTFP                first non-empty delta arrival - first-audio-sent time.
                      "the user can see words". This is the ASR analogue of the sibling
                      TTS harness's time-to-first-audio, and it is the number a person
                      waiting at a microphone actually feels.
* CER                 character error rate of the final transcript against a HUMAN
                      reference from the bank manifest, over normalised text (§normalise
                      below). Optional: only utterances whose clip has a reference get one.
                      It is the only metric here that says whether the server was RIGHT
                      rather than fast, and it is why a run can be DEGRADED.
* emission lag        for each delta: arrival - the send time of the frame that carried
  (client-observed)   the LAST SAMPLE THE SERVER HAD CONSUMED when it produced the delta
                      (`audio_s`).  It is client-observed: socket, kernel and Python
                      buffering sit between the server's write and the mark, so it is an
                      UPPER bound on the server's own lateness.  The server's `lag_ms` is
                      carried beside it, never averaged into it.
* server lag_ms       the series the server reports, kept as its own metric.
* finalization lag    `done` arrival - last-audio-sent time.
* backlog proxy       at each delta: audio seconds sent by that instant minus the audio
                      seconds the server says it consumed; the max over the utterance.
                      A PROXY: it is what the client can see, not the server's ring.
* pacing lateness     max and p95 of `late_ms`.  A run whose max lateness exceeds half a
                      frame did not hold 1x and cannot support a cadence percentile.

------------------------------------------------------------------------------ verdicts
`paced` is the honesty valve, the analogue of the coalesced-read valve in the sibling
qwen-tts harness `playback_sim.py`: when the client could not keep the schedule, the
cadence values are still stored and still printed, labelled DIAGNOSTIC, and `reported` is
false so no caller quotes them as a measurement.

Percentiles are NEAREST RANK, 1-based: index = ceil(p/100 * n), clamped to [1, n].  Never
a ratio of percentiles, never a percentile of fewer than 2 samples (`stat()` then reports
the samples themselves).  Two aggregations exist and are labelled: POOLED OVER DELTAS
(every delta of every utterance in one bag) and PER UTTERANCE (one number per utterance,
then a percentile over utterances).  They answer different questions and are never mixed.

Drift: the same metric recomputed over wall-clock windows; the gate is the largest
relative distance between a window p95 and the pooled p95.

Normalise (CER only): lowercase, drop the punctuation in PUNCT, collapse runs of
whitespace, strip. Defined HERE and nowhere else, because a CER computed with a different
normaliser is a different number wearing the same name. It is deliberately minimal -- no
number expansion, no spelling map -- so it measures the engine and not a text pipeline.
"""
from __future__ import annotations

import argparse
import bisect
import math
import sys

EPS = 1e-9

# ---------------------------------------------------------------------------- percentiles


def pct(xs, p):
    """Percentile by nearest rank, 1-based: index = ceil(p/100 * n), clamped to [1, n]."""
    ys = sorted(x for x in xs if x == x)          # NaN is not a sample
    if not ys:
        return None
    k = int(math.ceil(p / 100.0 * len(ys)))
    return ys[min(max(k, 1), len(ys)) - 1]


def stat(xs, basis, unit="ms", reported=True, label="MEASURED"):
    """One metric's aggregate.  Percentiles only when the data can support them."""
    xs = [x for x in xs if x is not None and x == x]
    out = {"n": len(xs), "unit": unit, "basis": basis, "reported": bool(reported),
           "label": label, "p50": None, "p95": None, "max": None, "samples": None}
    if not xs:
        return out
    out["max"] = max(xs)
    if len(xs) < 2:
        out["samples"] = list(xs)                 # a percentile of one sample is the sample
        return out
    out["p50"] = pct(xs, 50)
    out["p95"] = pct(xs, 95)
    return out


def _num(v, unit):
    """Seconds need decimals; milliseconds and percents do not."""
    return f"{v:.3f}" if unit == "s" else f"{v:.0f}"


def fmt_stat(s):
    """One line for a stat, honest about what it cannot support."""
    u = s["unit"]
    if s["n"] == 0:
        return "n/a (no samples)"
    if s["p50"] is None:
        return " / ".join(_num(v, u) for v in s["samples"]) + f"  ({s['n']} sample)"
    tag = "" if s["reported"] else "   DIAGNOSTIC"
    return (f"{_num(s['p50'], u)} / {_num(s['p95'], u)}  "
            f"(max {_num(s['max'], u)}, n={s['n']}){tag}")


# --------------------------------------------------------------------- per-utterance work


def chunk_period_ms(lookahead, encoder_frame_ms=80.0):
    """The server's encoder chunk period: `encoder_frame_ms` x (lookahead + 1).

    Nemotron cache-aware streaming encodes 8(r+1) mel frames of 10 ms per step
    (`docs/nemotron-arch.md`), i.e. 320 ms at the default lookahead 3.  The envelope in
    `.work/serving-v2-design.md` §6 is written in chunk periods, not in client frames.
    """
    return float(encoder_frame_ms) * (int(lookahead) + 1)


def _cum(sends):
    return [s[1] for s in sends]


def send_time_for_audio(sends, audio_s):
    """Send time of the frame carrying the last sample of `audio_s` seconds of audio.

    The first frame whose cumulative audio reaches `audio_s`; before the first frame it is
    the first, past the last it is the last (the server cannot have consumed more than was
    sent, but a rounding in either clock must not raise an IndexError in a harness)."""
    if not sends:
        return None
    i = bisect.bisect_left(_cum(sends), audio_s - EPS)
    return sends[min(max(i, 0), len(sends) - 1)][0]


def audio_sent_at(sends, t):
    """Audio seconds handed to the socket by monotonic time `t` (0 before the first frame)."""
    if not sends:
        return 0.0
    i = bisect.bisect_right([s[0] for s in sends], t + EPS)
    return sends[i - 1][1] if i > 0 else 0.0


PUNCT = ".,;:!?\u00bf\u00a1\"'`()[]{}<>\u2013\u2014-\u2026\u201c\u201d\u2018\u2019"


def normalise(text):
    """The CER normaliser. See the note at the top: minimal on purpose."""
    t = (text or "").lower()
    t = "".join(" " if ch in PUNCT else ch for ch in t)
    return " ".join(t.split())


def edit_distance(a, b):
    """Levenshtein, two rows. Pure arithmetic so the self-test can pin it."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp, ref):
    """Character error rate over normalised text. None when the reference is empty:
    a rate with an empty denominator is not a large error, it is no measurement."""
    r = normalise(ref)
    h = normalise(hyp)
    if not r:
        return None
    return edit_distance(h, r) / len(r)


def analyze_utterance(rec, frame_ms=100.0, pace=1.0):
    """Every per-utterance metric.  Marks are [t, value] so they can be windowed later."""
    sends = [list(s) for s in rec.get("sends") or []]
    events = rec.get("events") or []
    late = [float(x) for x in (rec.get("late_ms") or [])]
    out = {
        "clip": rec.get("clip"), "audio_s": float(rec.get("audio_s") or 0.0),
        "stream": rec.get("stream"), "rep": rec.get("rep"),
        "t_start": rec.get("t_start"), "frames": len(sends),
        "error": rec.get("error"), "rejected": bool(rec.get("rejected")),
        "class": rec.get("class"),
        "ttfb_ms": None, "ttfp_ms": None, "fin_ms": None, "text": "", "deltas": 0, "eous": 0,
        "lag_marks": [], "server_lag_marks": [], "backlog_marks": [],
        "backlog_max_s": None, "max_late_ms": max(late) if late else 0.0,
        "p95_late_ms": pct(late, 95) if late else None,
        "paced": True, "t_end": None,
    }
    half_frame = frame_ms / 2.0
    out["paced"] = (out["max_late_ms"] <= half_frame + EPS) and abs(pace - 1.0) < EPS

    first_delta_t = None
    first_frame_t = None
    texts = []
    for ev in events:
        t = ev.get("t")
        kind = ev.get("type")
        if t is not None and (first_frame_t is None or t < first_frame_t):
            first_frame_t = t   # ANY frame: delta, eou, done or error
        if kind == "eou":
            out["eous"] += 1
        if ev.get("lag_ms") is not None and kind in ("delta", "eou"):
            out["server_lag_marks"].append([t, float(ev["lag_ms"])])
        text = ev.get("text") or ""
        if kind != "delta" or not text:
            continue
        texts.append(text)
        if first_delta_t is None:
            first_delta_t = t
        consumed = ev.get("audio_s")
        if consumed is not None and sends:
            t_send = send_time_for_audio(sends, float(consumed))
            out["lag_marks"].append([t, (t - t_send) * 1000.0])
            out["backlog_marks"].append([t, audio_sent_at(sends, t) - float(consumed)])

    out["deltas"] = len(texts)
    out["text"] = "".join(texts)
    if out["backlog_marks"]:
        out["backlog_max_s"] = max(m[1] for m in out["backlog_marks"])
    if first_frame_t is not None and sends:
        out["ttfb_ms"] = (first_frame_t - sends[0][0]) * 1000.0
    if first_delta_t is not None and sends:
        out["ttfp_ms"] = (first_delta_t - sends[0][0]) * 1000.0
    if rec.get("done_t") is not None and sends:
        out["fin_ms"] = (rec["done_t"] - sends[-1][0]) * 1000.0
    ends = [rec.get("done_t")] + [m[0] for m in out["lag_marks"]] + [s[0] for s in sends]
    ends = [e for e in ends if e is not None]
    out["t_end"] = max(ends) if ends else rec.get("t_start")
    return out


# --------------------------------------------------------------------------------- drift


def windows(marks, window_s, t0, min_n=2):
    """Per-window p50/p95 of [t, value] marks, windows of `window_s` from `t0`."""
    if window_s is None or window_s <= 0 or not marks:
        return []
    buckets = {}
    for t, v in marks:
        buckets.setdefault(int((t - t0) // window_s), []).append(v)
    out = []
    for i in sorted(buckets):
        vs = buckets[i]
        out.append({"window": i, "t0_s": i * window_s, "t1_s": (i + 1) * window_s,
                    "n": len(vs), "p50": pct(vs, 50) if len(vs) >= min_n else None,
                    "p95": pct(vs, 95) if len(vs) >= min_n else None})
    return out


def drift(win, pooled_p95):
    """Largest relative distance between a STEADY window p95 and the pooled p95.

    Windows are not all the same experiment. A soak ramps up while its streams start and
    drains while the last utterances finish, and those windows hold a fraction of the
    samples the steady ones do: a 600 s soak at c=16 on 2026-09-20 reported windows of
    n=471, then nine of n~1850, then n=1076 and n=61. The ramp window's p95 was 254 ms
    against 74-116 ms in the steady ones, and comparing it to the pooled figure produced
    a 192 % "drift" on a run whose absolute lag never left a quarter of its envelope and
    which lost nothing. That is the gate measuring its own warm-up.

    So a window counts as STEADY when it carries at least half the median window's
    samples; the rest are reported and excluded, never silently dropped. The spread is
    still measured strictly over what remains, and a separate TREND is measured on top
    of it -- the last third of the steady windows against the first third -- because
    degradation is a direction, not a spread. A p95 wandering inside its envelope is
    noise; a p95 climbing is the thing this gate exists to catch."""
    usable = [w for w in win if w["p95"] is not None]
    if len(usable) < 2 or not pooled_p95:
        return {"max_drift_pct": None, "worst_window": None, "pooled_p95": pooled_p95,
                "windows": win, "trend_pct": None, "excluded": [],
                "reason": "fewer than 2 usable windows" if len(usable) < 2
                else "pooled p95 is zero"}
    ns = sorted(w["n"] for w in usable)
    med = ns[len(ns) // 2]
    steady = [w for w in usable if w["n"] >= 0.5 * med]
    excluded = [w["window"] for w in usable if w["n"] < 0.5 * med]
    if len(steady) < 2:                       # too short to have a steady state
        steady, excluded = usable, []
    worst, best = None, -1.0
    for w in steady:
        d = abs(w["p95"] - pooled_p95) / pooled_p95 * 100.0
        if d > best:
            best, worst = d, w["window"]
    trend = None
    if len(steady) >= 4:
        k = max(1, len(steady) // 3)
        first = sum(w["p95"] for w in steady[:k]) / k
        late = sum(w["p95"] for w in steady[-k:]) / k
        if first > 0:
            trend = (late - first) / first * 100.0
    return {"max_drift_pct": best, "worst_window": worst, "pooled_p95": pooled_p95,
            "windows": win, "trend_pct": trend, "excluded": excluded, "reason": None}


# ----------------------------------------------------------------------------- aggregate


def group_texts(utts):
    """clip -> sorted distinct texts.  The same clip must always produce the same text."""
    by_clip = {}
    for u in utts:
        if u.get("error") or u.get("rejected"):
            continue
        by_clip.setdefault(u["clip"], set()).add(u["text"])
    return {c: sorted(t) for c, t in by_clip.items()}


def aggregate(utts, frame_ms=100.0, pace=1.0, window_s=None, warmup_s=0.0, t0=None,
              reference=None, transcripts=None):
    """Every run-level number.  `utts` are the dicts `analyze_utterance` returned."""
    if t0 is None:
        starts = [u["t_start"] for u in utts if u.get("t_start") is not None]
        t0 = min(starts) if starts else 0.0

    counted = [u for u in utts
               if u.get("t_start") is None or u["t_start"] - t0 >= warmup_s - EPS]
    warm = len(utts) - len(counted)
    rejected = [u for u in counted if u.get("rejected")]
    errored = [u for u in counted if u.get("error") and not u.get("rejected")]
    ok = [u for u in counted if not u.get("error") and not u.get("rejected")]

    max_late = max([u["max_late_ms"] for u in counted], default=0.0)
    paced = all(u["paced"] for u in counted) and abs(pace - 1.0) < EPS if counted else False
    rep = bool(paced)                      # cadence percentiles are refused when not paced
    lbl = "MEASURED" if paced else "DIAGNOSTIC"

    lag_marks = [m for u in ok for m in u["lag_marks"]]
    srv_marks = [m for u in ok for m in u["server_lag_marks"]]
    bkl_marks = [m for u in ok for m in u["backlog_marks"]]
    ttfb = [u["ttfb_ms"] for u in ok if u["ttfb_ms"] is not None]
    ttfp = [u["ttfp_ms"] for u in ok if u["ttfp_ms"] is not None]
    fin = [u["fin_ms"] for u in ok if u["fin_ms"] is not None]
    lag_utt = [pct(u["lag_marks"] and [m[1] for m in u["lag_marks"]], 95)
               for u in ok if len(u["lag_marks"]) >= 2]

    m = {
        "ttfb_ms": stat(ttfb, "per utterance", "ms", rep, lbl),
        "ttfp_ms": stat(ttfp, "per utterance", "ms", rep, lbl),
        "emission_lag_ms": stat([x[1] for x in lag_marks], "pooled over deltas", "ms", rep, lbl),
        "emission_lag_utt_p95_ms": stat(lag_utt, "per utterance (p95 of each)", "ms", rep, lbl),
        "server_lag_ms": stat([x[1] for x in srv_marks], "pooled over server frames", "ms", rep, lbl),
        "finalization_lag_ms": stat(fin, "per utterance", "ms", rep, lbl),
        "backlog_s": stat([x[1] for x in bkl_marks], "pooled over deltas", "s", rep, lbl),
        "backlog_max_s": stat([u["backlog_max_s"] for u in ok if u["backlog_max_s"] is not None],
                              "per utterance (max of each)", "s", rep, lbl),
        "pacing_late_ms": stat([u["max_late_ms"] for u in counted],
                               "per utterance (max of each)", "ms", True, "MEASURED"),
    }

    dr = {}
    if window_s:
        for name, marks in (("emission_lag_ms", lag_marks), ("server_lag_ms", srv_marks),
                            ("backlog_s", bkl_marks)):
            dr[name] = drift(windows(marks, window_s, t0), m[name]["p95"])
        ttfp_marks = [[u["t_start"], u["ttfp_ms"]] for u in ok
                      if u["ttfp_ms"] is not None and u.get("t_start") is not None]
        dr["ttfp_ms"] = drift(windows(ttfp_marks, window_s, t0), m["ttfp_ms"]["p95"])

    groups = group_texts(counted)
    identity_fail = {c: t for c, t in groups.items() if len(t) > 1}
    ref_fail = {}
    if reference:
        for c, texts in groups.items():
            exp = reference.get(c)
            if exp is None:
                exp = reference.get(c.rsplit("/", 1)[-1])
            if exp is not None and texts != [exp]:
                ref_fail[c] = {"expected": exp, "got": texts}

    # Quality, as opposed to speed. CER is per UTTERANCE against the bank's human
    # reference; it is NOT gated by pacing, because whether the transcript is right does
    # not depend on whether the client held its schedule -- only the cadence numbers do.
    cers, cer_worst = [], None
    if transcripts:
        for u in ok:
            clip = u.get("clip") or ""
            ref = transcripts.get(clip)
            if ref is None:
                ref = transcripts.get(clip.rsplit("/", 1)[-1])
            if ref is None:
                continue
            v = cer(u["text"], ref)
            if v is None:
                continue
            cers.append(v)
            if cer_worst is None or v > cer_worst[0]:
                cer_worst = (v, clip, u["text"], ref)
    m["cer"] = stat(cers, "per utterance", "", True, "MEASURED")
    quality = {
        "with_reference": len(cers),
        "without_reference": len(ok) - len(cers),
        "worst": ({"cer": cer_worst[0], "clip": cer_worst[1],
                   "got": cer_worst[2], "expected": cer_worst[3]}
                  if cer_worst else None),
    }

    audio_s = sum(u["audio_s"] for u in ok)
    span = max([u["t_end"] for u in counted if u.get("t_end") is not None], default=t0) - t0
    return {
        "counts": {"utterances": len(counted), "ok": len(ok), "errors": len(errored),
                   "rejected": len(rejected), "warmup_excluded": warm,
                   "deltas": sum(u["deltas"] for u in ok), "eous": sum(u["eous"] for u in ok),
                   "audio_s": audio_s, "span_s": span},
        "pacing": {"max_late_ms": max_late, "half_frame_ms": frame_ms / 2.0, "pace": pace,
                   "paced": paced,
                   "verdict": "PACED" if paced else "NOT PACED (cadence is DIAGNOSTIC)"},
        "metrics": m,
        "drift": dr,
        "text_groups": groups,
        "identity_fail": identity_fail,
        "reference_fail": ref_fail,
        "quality": quality,
    }


# ------------------------------------------------------------------------------ envelope
#
# The provisional envelope of .work/serving-v2-design.md §6, expressed in chunk periods so
# it moves with `--lookahead`.  Each line is checked separately and the failing line is
# named: a single word ("NOT STREAMABLE") without the line that produced it is not a
# report.

def default_thresholds(chunk_ms):
    return {
        # TTFP carries the MODEL's emission delay, not only the server's: measured on
        # the M1 with a clip that speaks from t=0, the first delta of Nemotron at
        # lookahead 3 arrives ~950 ms after the first byte whether the language is
        # explicit or auto (first chunk 250 ms + ~2 chunks of RNNT emission delay).
        # The server's own cost is the emission-lag line; TTFP is gated loosely.
        "ttfp_p95_ms": 3.0 * chunk_ms + 200.0,
        "emission_lag_p95_ms": chunk_ms,
        "finalization_p95_ms": 500.0,
        "backlog_max_s": 2.0 * chunk_ms / 1000.0,
        # Spread of a p95 ACROSS WINDOWS, and the direction it moves.
        #
        # 20 % was inherited, not derived, and it is too tight for a tail
        # statistic: a p95 over ~1800 samples has far more sampling variance
        # than a median, so a healthy run wanders. The sibling engine landed on
        # the same split from its own measurements -- 30 % for a p95, 20 % for a
        # p50 -- and the reason is statistical rather than a matter of taste.
        #
        # The pair is what makes this strict rather than lax. A soak on
        # 2026-09-20 (1x24, c=16, 600 s, zero losses, pooled lag p95 87 ms
        # against a 320 ms envelope) read 192 % under the old rule, 36 % spread
        # with a trend of -14 % under this one: wandering, and improving. A run
        # that wandered the same amount while CLIMBING fails on the trend line,
        # which no spread threshold would have caught.
        "max_drift_pct": 30.0,
        "max_trend_pct": 15.0,
        "marginal_factor": 1.5,
        # Quality, not cadence. A PLACEHOLDER until a measured baseline exists: the CER
        # of this model on this bank has never been recorded, so this number is a guard
        # against a collapse (a broken kernel, a wrong blank, a batched path that drifted),
        # NOT a quality target. Replace it with baseline + margin the first time the bank
        # is run on a quiet box, and say in the note which run set it.
        "cer_p95": 0.25,
    }


def envelope_verdict(summary, thr):
    """GOOD / MARGINAL / DEGRADED / NOT STREAMABLE / INVALID, and the line that decided it.

    INVALID first: a run whose text is not identical, that produced nothing, or that could
    not hold 1x pacing cannot be judged by an envelope at all — whatever the timing says.

    DEGRADED is the verdict this harness exists to be able to give: the server held its
    cadence and the transcripts got WORSE. Without it, "faster" and "better" are the same
    word, and a kernel that quietly lost accuracy would read as an improvement. It ranks
    below NOT STREAMABLE only because a server that cannot stream is not yet in a position
    to be inaccurate.
    """
    lines, invalid = [], []
    c = summary["counts"]
    if summary["identity_fail"]:
        invalid.append(f"text identity: {len(summary['identity_fail'])} clip(s) differ across streams")
    if summary["reference_fail"]:
        invalid.append(f"reference: {len(summary['reference_fail'])} clip(s) differ from the reference")
    if c["ok"] == 0:
        invalid.append("no utterance completed")
    if not summary["pacing"]["paced"]:
        invalid.append("client could not pace at 1x: cadence percentiles are refused")

    def line(name, value, limit, unit):
        if value is None:
            lines.append({"line": name, "value": None, "limit": limit, "unit": unit,
                          "status": "NO DATA"})
            return
        if value <= limit:
            st = "PASS"
        elif value <= limit * thr["marginal_factor"]:
            st = "MARGINAL"
        else:
            st = "FAIL"
        lines.append({"line": name, "value": value, "limit": limit, "unit": unit, "status": st})

    m = summary["metrics"]
    # Streams that never finished, as an envelope line with a limit of zero.
    #
    # This used to be worth a MARGINAL beside the percentiles, on the reasoning
    # that a handful of errors in a long run is noise. It is not: a stream whose
    # reader timed out is a stream a caller lost, and a run that loses four
    # utterances out of sixty-four has not "held with no headroom" -- it has
    # dropped four sessions. A W x T sweep on 2026-09-20 read as 8x3 holding
    # twice what 1x24 held, and every winning rung was losing three to four
    # streams to `reader: timed out` while the verdict said MARGINAL.
    #
    # A 503 is NOT counted here and stays a MARGINAL below: a refusal is the
    # admission ladder working, and the whole design says a full machine must
    # refuse. Being refused at the door and being dropped mid-sentence are
    # different outcomes and must not share a verdict.
    line("utterances lost", c["errors"], 0, "")
    line("TTFP p95", m["ttfp_ms"]["p95"], thr["ttfp_p95_ms"], "ms")
    # TTFB has no envelope limit and is printed as a FACT beside TTFP: the two together
    # say whether a wait is transport or model, and inventing a limit for it would be a
    # threshold nobody derived.
    line("emission lag p95 (pooled)", m["emission_lag_ms"]["p95"], thr["emission_lag_p95_ms"], "ms")
    line("finalization lag p95", m["finalization_lag_ms"]["p95"], thr["finalization_p95_ms"], "ms")
    line("backlog max", m["backlog_max_s"]["max"], thr["backlog_max_s"], "s")
    for name, d in sorted(summary.get("drift", {}).items()):
        if d.get("max_drift_pct") is not None:
            line(f"drift {name} p95", d["max_drift_pct"], thr["max_drift_pct"], "%")
        # Direction, not spread. A metric that climbs across the steady windows is
        # degrading even while every window is inside the envelope, and that is the
        # failure a soak exists to find; the threshold is deliberately tighter than
        # the spread's, because a trend has no innocent explanation.
        if d.get("trend_pct") is not None:
            line(f"trend {name} p95", abs(d["trend_pct"]), thr["max_trend_pct"], "%")

    q = summary.get("quality") or {}
    cer_p95 = summary["metrics"].get("cer", {}).get("p95")
    quality_lines = []
    if q.get("with_reference"):
        before = len(lines)
        line("CER p95", cer_p95, thr["cer_p95"], "")
        quality_lines = lines[before:]

    if invalid:
        verdict = "INVALID"
    elif any(l["status"] == "FAIL" for l in lines if l not in quality_lines):
        verdict = "NOT STREAMABLE"
    elif any(l["status"] == "FAIL" for l in quality_lines):
        verdict = "DEGRADED"
    elif any(l["status"] == "MARGINAL" for l in lines) or c["rejected"] > 0:
        verdict = "MARGINAL"
    elif all(l["status"] == "NO DATA" for l in lines):
        verdict = "INVALID"
        invalid.append("no envelope line had data")
    else:
        verdict = "GOOD"
    return {"verdict": verdict, "lines": lines, "invalid_reasons": invalid, "thresholds": thr}


# -------------------------------------------------------------------------------- report


def format_summary(summary, env=None, indent="  "):
    c, p, m = summary["counts"], summary["pacing"], summary["metrics"]
    out = []
    out.append(f"{indent}utterances {c['ok']}/{c['utterances']} ok, {c['errors']} errors, "
               f"{c['rejected']} rejected, {c['warmup_excluded']} excluded by warm-up")
    out.append(f"{indent}audio {c['audio_s']:.1f} s, {c['deltas']} deltas, span {c['span_s']:.1f} s")
    out.append(f"{indent}pacing: max lateness {p['max_late_ms']:.1f} ms vs half a frame "
               f"{p['half_frame_ms']:.0f} ms -> {p['verdict']}")
    for key, title in (("ttfb_ms", "TTFB (first frame)"),
                       ("ttfp_ms", "TTFP (first word)"),
                       ("emission_lag_ms", "emission lag (client)"),
                       ("emission_lag_utt_p95_ms", "emission lag per-utt p95"),
                       ("server_lag_ms", "server lag_ms"),
                       ("finalization_lag_ms", "finalization lag"),
                       ("backlog_max_s", "backlog max"),
                       ("cer", "CER vs reference"),
                       ("pacing_late_ms", "pacing lateness")):
        s = m[key]
        if s["n"] == 0:
            continue
        unit = s["unit"]
        out.append(f"{indent}{title:<26} p50/p95 {unit:<2}: {fmt_stat(s)}    [{s['basis']}]")
    for name, d in sorted(summary.get("drift", {}).items()):
        if d.get("max_drift_pct") is None:
            out.append(f"{indent}drift {name:<20}: not computed ({d.get('reason')})")
        else:
            out.append(f"{indent}drift {name:<20}: max {d['max_drift_pct']:.1f}% "
                       f"(window {d['worst_window']} vs pooled p95 {d['pooled_p95']:.1f}, "
                       f"{len(d['windows'])} windows)")
    if summary["identity_fail"]:
        out.append(f"{indent}TEXT IDENTITY FAIL on {len(summary['identity_fail'])} clip(s):")
        for clip, texts in summary["identity_fail"].items():
            out.append(f"{indent}  {clip}: {len(texts)} distinct texts")
    if summary["reference_fail"]:
        out.append(f"{indent}REFERENCE FAIL on {len(summary['reference_fail'])} clip(s)")
    q = summary.get("quality") or {}
    if q.get("with_reference") or q.get("without_reference"):
        out.append(f"{indent}quality: {q['with_reference']} utterance(s) scored against a "
                   f"reference, {q['without_reference']} without one")
        w = q.get("worst")
        if w and w["cer"] > 0.0:
            out.append(f"{indent}  worst CER {w['cer']:.3f} on {w['clip']}")
            out.append(f"{indent}    expected: {w['expected'][:96]}")
            out.append(f"{indent}    got:      {w['got'][:96]}")
    if env:
        for l in env["lines"]:
            v = "n/a" if l["value"] is None else _num(l["value"], l["unit"])
            out.append(f"{indent}[{l['status']:<8}] {l['line']:<28} {v:>9} {l['unit']:<2} "
                       f"(limit {_num(l['limit'], l['unit'])})")
        for r in env["invalid_reasons"]:
            out.append(f"{indent}[INVALID ] {r}")
        out.append(f"{indent}verdict: {env['verdict']}")
    return "\n".join(out)


# ------------------------------------------------------------------------------ self-test
#
# Known answers, computed by hand in the comments.  A metric module whose own arithmetic is
# not gated is an opinion; `make test` runs this without a model.

def _mk(clip, sends, events, done_t, late_ms=None, t_start=None, **kw):
    rec = {"clip": clip, "audio_s": sends[-1][1] if sends else 0.0, "sends": sends,
           "events": events, "done_t": done_t, "late_ms": late_ms or [0.0] * len(sends),
           "t_start": sends[0][0] if (t_start is None and sends) else t_start,
           "error": None, "rejected": False}
    rec.update(kw)
    return rec


def _ev(t, audio_s, text, lag_ms=None, kind="delta"):
    return {"t": t, "type": kind, "audio_s": audio_s, "lag_ms": lag_ms, "text": text}


def _eq(got, want, what, tol=1e-6):
    """Numbers compare within `tol`; anything else compares exactly. The normaliser and
    the text fields are strings, and a tolerance on a string is meaningless."""
    if isinstance(want, str) or isinstance(got, str):
        ok = got == want
    else:
        ok = (got is None and want is None) or (
            got is not None and want is not None and abs(got - want) <= tol)
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: got {got!r}, want {want!r}")
    return ok


def self_test():
    bad = 0

    print("nearest-rank percentiles")
    #  [1..6]: p50 -> ceil(3.0)=3 -> 3 ; p95 -> ceil(5.7)=6 -> 6 ; p0 -> rank 1 ; p100 -> 6
    for p, want in ((50, 3), (95, 6), (0, 1), (100, 6)):
        bad += not _eq(pct([4, 1, 6, 3, 5, 2], p), want, f"pct([1..6], {p})")
    #  [1..4]: p50 -> ceil(2.0)=2 -> 2 ; p25 -> ceil(1.0)=1 -> 1 ; p75 -> ceil(3.0)=3 -> 3
    for p, want in ((50, 2), (25, 1), (75, 3)):
        bad += not _eq(pct([1, 2, 3, 4], p), want, f"pct([1..4], {p})")
    bad += not _eq(pct([], 95), None, "pct([], 95)")
    s = stat([7.0], "per utterance")
    bad += not _eq(s["p50"], None, "stat of 1 sample has no p50")
    bad += not _eq(s["samples"][0], 7.0, "stat of 1 sample keeps the sample")

    print("chunk period")
    bad += not _eq(chunk_period_ms(3), 320.0, "lookahead 3 -> 320 ms")
    bad += not _eq(chunk_period_ms(0), 80.0, "lookahead 0 -> 80 ms")

    print("frame lookup")
    sends = [[10.0, 0.1], [10.1, 0.2], [10.2, 0.3], [10.3, 0.4], [10.4, 0.5]]
    bad += not _eq(send_time_for_audio(sends, 0.2), 10.1, "0.2 s consumed -> frame 2")
    bad += not _eq(send_time_for_audio(sends, 0.0), 10.0, "0 s consumed -> frame 1")
    bad += not _eq(send_time_for_audio(sends, 9.9), 10.4, "more than sent -> last frame")
    bad += not _eq(send_time_for_audio(sends, 0.25), 10.2, "mid frame -> the frame that holds it")
    bad += not _eq(audio_sent_at(sends, 9.0), 0.0, "before the first frame -> 0 s sent")
    bad += not _eq(audio_sent_at(sends, 10.25), 0.3, "at 10.25 -> 0.3 s sent")
    bad += not _eq(audio_sent_at(sends, 99.0), 0.5, "after the last frame -> 0.5 s sent")

    print("one canonical utterance (5 frames of 100 ms)")
    #  delta A at 10.25 s, server consumed 0.20 s -> carried by the frame sent at 10.1
    #     emission lag = 150 ms ; backlog = sent(10.25)=0.3 - 0.2 = 0.1 s
    #  delta B at 10.45 s, consumed 0.40 s -> frame sent at 10.3 -> 150 ms ; 0.5-0.4 = 0.1 s
    #  TTFP = 10.25 - 10.00 = 250 ms ; finalization = 10.60 - 10.40 = 200 ms
    u = analyze_utterance(_mk("a.wav", sends,
                              [_ev(10.25, 0.2, "one", 50.0), _ev(10.45, 0.4, " two", 30.0)],
                              10.6), frame_ms=100.0)
    bad += not _eq(u["ttfp_ms"], 250.0, "TTFP", 1e-6)
    bad += not _eq(u["lag_marks"][0][1], 150.0, "emission lag of delta A", 1e-6)
    bad += not _eq(u["lag_marks"][1][1], 150.0, "emission lag of delta B", 1e-6)
    bad += not _eq(u["backlog_max_s"], 0.1, "backlog max", 1e-9)
    bad += not _eq(u["fin_ms"], 200.0, "finalization lag", 1e-6)
    bad += not _eq(float(u["deltas"]), 2.0, "delta count")
    bad += not _eq(u["server_lag_marks"][1][1], 30.0, "server lag_ms series kept")
    print(f"  {'ok  ' if u['text'] == 'one two' else 'FAIL'} text: {u['text']!r}")
    bad += u["text"] != "one two"
    print(f"  {'ok  ' if u['paced'] else 'FAIL'} paced with zero lateness")
    bad += not u["paced"]

    print("edge: no deltas, done present")
    u0 = analyze_utterance(_mk("a.wav", sends, [], 10.6), frame_ms=100.0)
    bad += not _eq(u0["ttfp_ms"], None, "TTFP without a delta is None")
    bad += not _eq(u0["backlog_max_s"], None, "backlog without a delta is None")
    bad += not _eq(u0["fin_ms"], 200.0, "finalization still computable", 1e-6)

    print("edge: done missing")
    u1 = analyze_utterance(_mk("a.wav", sends, [_ev(10.25, 0.2, "x")], None), frame_ms=100.0)
    bad += not _eq(u1["fin_ms"], None, "finalization without done is None")
    bad += not _eq(u1["ttfp_ms"], 250.0, "TTFP still computable", 1e-6)

    print("edge: single frame, single delta")
    u2 = analyze_utterance(_mk("a.wav", [[5.0, 0.1]], [_ev(5.4, 0.1, "x")], 5.5), frame_ms=100.0)
    bad += not _eq(u2["ttfp_ms"], 400.0, "TTFP of a one-frame utterance", 1e-6)
    bad += not _eq(u2["lag_marks"][0][1], 400.0, "emission lag of a one-frame utterance", 1e-6)
    bad += not _eq(u2["fin_ms"], 500.0, "finalization of a one-frame utterance", 1e-6)

    print("edge: unpaced (max lateness 80 ms > half of a 100 ms frame)")
    uu = analyze_utterance(_mk("a.wav", sends, [_ev(10.25, 0.2, "x")], 10.6,
                               late_ms=[0.0, 80.0, 0.0, 0.0, 0.0]), frame_ms=100.0)
    bad += not _eq(uu["max_late_ms"], 80.0, "max lateness")
    print(f"  {'ok  ' if not uu['paced'] else 'FAIL'} not paced")
    bad += uu["paced"]
    agg_u = aggregate([uu], frame_ms=100.0)
    print(f"  {'ok  ' if agg_u['metrics']['ttfp_ms']['label'] == 'DIAGNOSTIC' else 'FAIL'} "
          f"cadence labelled {agg_u['metrics']['ttfp_ms']['label']}")
    bad += agg_u["metrics"]["ttfp_ms"]["label"] != "DIAGNOSTIC"
    print(f"  {'ok  ' if not agg_u['metrics']['ttfp_ms']['reported'] else 'FAIL'} cadence refused")
    bad += agg_u["metrics"]["ttfp_ms"]["reported"]
    print(f"  {'ok  ' if agg_u['metrics']['ttfp_ms']['n'] == 1 else 'FAIL'} the value is still stored")
    bad += agg_u["metrics"]["ttfp_ms"]["n"] != 1
    bad += not _eq(agg_u["metrics"]["pacing_late_ms"]["max"], 80.0, "lateness itself stays MEASURED")
    print(f"  {'ok  ' if envelope_verdict(agg_u, default_thresholds(320.0))['verdict'] == 'INVALID' else 'FAIL'} "
          f"unpaced run is INVALID for the envelope")
    bad += envelope_verdict(agg_u, default_thresholds(320.0))["verdict"] != "INVALID"

    print("quality: the normaliser, the distance and the rate")
    bad += not _eq(normalise("Hello, World!  "), "hello world", "normalise strips and folds")
    bad += not _eq(normalise("It's \u201cfine\u201d \u2014 really?"), "it s fine really",
                   "normalise drops the punctuation it declares")
    bad += not _eq(edit_distance("kitten", "sitting"), 3, "kitten -> sitting is 3")
    bad += not _eq(edit_distance("", "abc"), 3, "empty hypothesis costs the reference")
    bad += not _eq(edit_distance("abc", "abc"), 0, "identical is 0")
    bad += not _eq(cer("the cat sat", "The cat sat."), 0.0, "CER ignores case and stops", 1e-12)
    bad += not _eq(cer("the bat sat", "the cat sat"), 1.0 / 11.0, "one substitution in 11", 1e-12)
    bad += not _eq(cer("anything", ""), None, "an empty reference has no rate")

    print("quality: CER reaches the verdict, and only when a reference exists")
    snd_q = [[20.0, 0.1], [20.1, 0.2]]
    def utt_q(clip, text):
        return analyze_utterance(_mk(clip, snd_q, [_ev(20.15, 0.1, text)], 20.3, t_start=20.0),
                                 frame_ms=100.0)
    good = aggregate([utt_q("q.wav", "the cat sat")], frame_ms=100.0,
                     transcripts={"q.wav": "the cat sat"})
    bad += not _eq(good["metrics"]["cer"]["max"], 0.0, "a perfect transcript scores 0", 1e-12)
    bad += not _eq(good["quality"]["with_reference"], 1, "the utterance was scored")
    # TWO utterances: stat() refuses a percentile of one sample, so a single bad
    # transcript has no p95 and the line correctly reads NO DATA. That is the module
    # being honest, and the test has to respect it rather than work around it.
    worse = aggregate([utt_q("q1.wav", "zzzzzzzzzzz"), utt_q("q2.wav", "zzzzzzzzzzz")],
                      frame_ms=100.0,
                      transcripts={"q1.wav": "the cat sat", "q2.wav": "the cat sat"})
    thr_q = default_thresholds(320.0)
    vq = envelope_verdict(worse, thr_q)
    print(f"  {'ok  ' if vq['verdict'] == 'DEGRADED' else 'FAIL'} "
          f"cadence fine, transcript wrong -> {vq['verdict']}")
    bad += vq["verdict"] != "DEGRADED"
    bad += not _eq(worse["quality"]["worst"]["clip"], "q1.wav", "the worst utterance is named")
    none_ref = aggregate([utt_q("q1.wav", "zzzzzzzzzzz"), utt_q("q2.wav", "zzzzzzzzzzz")],
                         frame_ms=100.0)
    print(f"  {'ok  ' if envelope_verdict(none_ref, thr_q)['verdict'] != 'DEGRADED' else 'FAIL'} "
          f"no reference -> no quality verdict")
    bad += envelope_verdict(none_ref, thr_q)["verdict"] == "DEGRADED"

    print("TTFB is the first frame of any kind, TTFP the first word")
    evs_b = [{"t": 20.12, "type": "eou", "audio_s": None, "lag_ms": None, "text": ""},
             _ev(20.25, 0.2, "hello")]
    ub = analyze_utterance(_mk("b.wav", snd_q, evs_b, 20.4), frame_ms=100.0)
    bad += not _eq(ub["ttfb_ms"], 120.0, "TTFB counts the eou that came first", 1e-6)
    bad += not _eq(ub["ttfp_ms"], 250.0, "TTFP still waits for the word", 1e-6)

    print("aggregation: pooled over deltas vs per utterance")
    #  two utterances, 2 deltas each with lags 100, 200 and 300, 400
    #  pooled   [100,200,300,400] -> p50 = rank ceil(2)=2 -> 200 ; p95 = rank 4 -> 400
    #  per-utt p95 of each: ceil(1.9)=2 -> 200 and 400 ; over utterances p50 -> 200, p95 -> 400
    def lag_utt(clip, t0, lags):
        snd = [[t0 + i * 0.1, (i + 1) * 0.1] for i in range(5)]
        evs = [_ev(snd[1][0] + lags[0] / 1000.0, 0.2, "a"),
               _ev(snd[3][0] + lags[1] / 1000.0, 0.4, "b")]
        return analyze_utterance(_mk(clip, snd, evs, snd[-1][0] + 0.2, t_start=t0), frame_ms=100.0)

    agg = aggregate([lag_utt("a.wav", 0.0, [100.0, 200.0]),
                     lag_utt("b.wav", 20.0, [300.0, 400.0])], frame_ms=100.0)
    bad += not _eq(agg["metrics"]["emission_lag_ms"]["p50"], 200.0, "pooled p50", 1e-6)
    bad += not _eq(agg["metrics"]["emission_lag_ms"]["p95"], 400.0, "pooled p95", 1e-6)
    bad += not _eq(agg["metrics"]["emission_lag_utt_p95_ms"]["p50"], 200.0, "per-utt p95 -> p50", 1e-6)
    bad += not _eq(agg["metrics"]["emission_lag_utt_p95_ms"]["p95"], 400.0, "per-utt p95 -> p95", 1e-6)
    bad += not _eq(float(agg["counts"]["ok"]), 2.0, "two utterances counted")

    print("warm-up exclusion")
    aggw = aggregate([lag_utt("a.wav", 0.0, [100.0, 200.0]),
                      lag_utt("b.wav", 20.0, [300.0, 400.0])], frame_ms=100.0, warmup_s=10.0)
    bad += not _eq(float(aggw["counts"]["warmup_excluded"]), 1.0, "first utterance excluded")
    bad += not _eq(aggw["metrics"]["emission_lag_ms"]["p50"], 300.0, "pooled p50 after warm-up", 1e-6)

    print("text identity")
    a = analyze_utterance(_mk("a.wav", sends, [_ev(10.25, 0.2, "hello")], 10.6), frame_ms=100.0)
    b = analyze_utterance(_mk("a.wav", sends, [_ev(10.25, 0.2, "hallo")], 10.6), frame_ms=100.0)
    aggi = aggregate([a, b], frame_ms=100.0)
    print(f"  {'ok  ' if 'a.wav' in aggi['identity_fail'] else 'FAIL'} two texts for one clip fail")
    bad += "a.wav" not in aggi["identity_fail"]
    bad += envelope_verdict(aggi, default_thresholds(320.0))["verdict"] != "INVALID"
    aggj = aggregate([a, analyze_utterance(_mk("a.wav", sends, [_ev(10.25, 0.2, "hello")], 10.6),
                                           frame_ms=100.0)], frame_ms=100.0)
    print(f"  {'ok  ' if not aggj['identity_fail'] else 'FAIL'} identical texts pass")
    bad += bool(aggj["identity_fail"])

    print("drift across windows")
    #  window 0 (t in [0,10)): four 10s      -> p95 = rank ceil(3.8)=4 -> 10
    #  window 1 (t in [10,20)): 10,10,10,20  -> p95 = rank 4 -> 20
    #  window 2 (t in [20,30)): four 10s     -> p95 -> 10
    #  pooled (12 values, one 20) -> p95 = rank ceil(11.4)=12 -> 20
    #  drift = max(|10-20|, |20-20|, |10-20|)/20 = 50 %
    marks = ([[t, 10.0] for t in (1, 2, 3, 4)] + [[t, 10.0] for t in (11, 12, 13)]
             + [[14.0, 20.0]] + [[t, 10.0] for t in (21, 22, 23, 24)])
    win = windows(marks, 10.0, 0.0)
    bad += not _eq(float(len(win)), 3.0, "three windows")
    bad += not _eq(win[0]["p95"], 10.0, "window 0 p95")
    bad += not _eq(win[1]["p95"], 20.0, "window 1 p95")
    d = drift(win, pct([m[1] for m in marks], 95))
    bad += not _eq(d["pooled_p95"], 20.0, "pooled p95")
    bad += not _eq(d["max_drift_pct"], 50.0, "max relative drift")
    bad += not _eq(drift(win[:1], 20.0)["max_drift_pct"], None, "one window cannot drift")
    bad += not _eq(drift(win, 0.0)["max_drift_pct"], None, "a zero baseline cannot drift")

    print("envelope lines")
    #  chunk 320 ms -> TTFP limit 1160 (3 chunks + 200), lag limit 320, finalization 500, backlog 0.64 s
    thr = default_thresholds(320.0)
    bad += not _eq(thr["ttfp_p95_ms"], 1160.0, "TTFP limit")
    bad += not _eq(thr["backlog_max_s"], 0.64, "backlog limit", 1e-9)
    #  a percentile needs two samples, so every envelope fixture is a pair of identical
    #  utterances (identical text: the identity gate must stay green while the timing moves)
    def pair(delta_dt, consumed, done_dt, nframes=5):
        us = []
        for base in (10.0, 40.0):
            snd = [[base + i * 0.1, (i + 1) * 0.1] for i in range(nframes)]
            us.append(analyze_utterance(
                _mk("a.wav", snd, [_ev(base + delta_dt, consumed, "x", 40.0)], base + done_dt,
                    t_start=base), frame_ms=100.0))
        return aggregate(us, frame_ms=100.0)

    #  TTFP 150 ms, emission lag 50 ms, finalization 100 ms: every line inside
    ev = envelope_verdict(pair(0.15, 0.2, 0.5), thr)
    print(f"  {'ok  ' if ev['verdict'] == 'GOOD' else 'FAIL'} inside the envelope -> {ev['verdict']}")
    bad += ev["verdict"] != "GOOD"
    #  TTFP 2000 ms (> 1740 = 1.5 x 1160) and emission lag 1900 ms: two lines FAIL
    ev2 = envelope_verdict(pair(2.0, 0.2, 2.2), thr)
    failed = [l["line"] for l in ev2["lines"] if l["status"] == "FAIL"]
    print(f"  {'ok  ' if ev2['verdict'] == 'NOT STREAMABLE' else 'FAIL'} far outside -> "
          f"{ev2['verdict']} on {failed}")
    bad += ev2["verdict"] != "NOT STREAMABLE"
    bad += "TTFP p95" not in failed
    #  12 frames (1.2 s): TTFP 1400 ms (1160 < 1400 <= 1740), emission lag 400 ms
    #  (320 < 400 <= 480, the delta at 1.4 s covers audio sent at 1.0 s), finalization
    #  600 ms (500 < 600 <= 750): over the envelope, under the marginal factor ->
    #  MARGINAL, and nothing FAILs
    ev3 = envelope_verdict(pair(1.4, 1.1, 1.7, nframes=12), thr)
    print(f"  {'ok  ' if ev3['verdict'] == 'MARGINAL' else 'FAIL'} just outside -> {ev3['verdict']} "
          f"on {[l['line'] for l in ev3['lines'] if l['status'] == 'MARGINAL']}")
    bad += ev3["verdict"] != "MARGINAL"
    bad += any(l["status"] == "FAIL" for l in ev3["lines"])
    empty = aggregate([], frame_ms=100.0)
    bad += envelope_verdict(empty, thr)["verdict"] != "INVALID"
    print(f"  ok   an empty run is INVALID, never GOOD by absence of evidence")

    print("report formatting does not raise")
    print(format_summary(agg, envelope_verdict(agg, thr), indent="    "))

    print(f"\nstreaming_metrics self-test: {'FAIL' if bad else 'PASS'} ({int(bad)} failures)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="streaming metric definitions",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the known-answer test (no model, no server)")
    a = ap.parse_args()
    if not a.self_test:
        ap.print_help()
        return 0
    return self_test()


if __name__ == "__main__":
    sys.exit(main())
