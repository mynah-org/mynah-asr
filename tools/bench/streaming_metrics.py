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

Normalise (CER only): NFKC, strip a written-out language tag, lowercase, turn every
Unicode punctuation mark into a space, collapse runs of whitespace, strip. Defined HERE
and nowhere else, because a CER computed with a different normaliser is a different number
wearing the same name -- and this repo had two of them, one of which dropped the
guillemets the Spanish and German references use and one of which did not. It is
deliberately minimal beyond that -- no number expansion, no spelling map -- so it measures
the engine and not a text pipeline. `reference_for` is the matching rule that goes with
it: path suffixes, never an ambiguous bare file name.
"""
from __future__ import annotations

import argparse
import bisect
import math
import re
import sys
import unicodedata

EPS = 1e-9

# ---------------------------------------------------------------------------- percentiles


def pct(xs, p):
    """Percentile by nearest rank, 1-based: index = ceil(p/100 * n), clamped to [1, n].

    The rank is computed as `p * n / 100` and not as `(p / 100) * n`, with a
    guard below the ceiling. A fractional percentile makes the difference
    visible: 99.9 / 100 is not representable, so `(99.9 / 100) * 1000` is
    999.0000000000001, whose ceiling is 1000 — the p99.9 of a thousand samples
    silently became the maximum. Integer percentiles never showed it, which is
    why it survived until p99.9 had a caller."""
    ys = sorted(x for x in xs if x == x)          # NaN is not a sample
    if not ys:
        return None
    k = int(math.ceil(p * len(ys) / 100.0 - 1e-9))
    return ys[min(max(k, 1), len(ys)) - 1]


def stat(xs, basis, unit="ms", reported=True, label="MEASURED"):
    """One metric's aggregate.  Percentiles only when the data can support them.

    p99 and p99.9 appear only once the sample count can carry them. At nearest
    rank a p99 of 50 samples IS the maximum, and a p99.9 of 500 is too: printing
    them anyway invents a tail that was never observed. The thresholds are the
    smallest n for which the percentile names a rank below the top one."""
    xs = [x for x in xs if x is not None and x == x]
    out = {"n": len(xs), "unit": unit, "basis": basis, "reported": bool(reported),
           "label": label, "p50": None, "p95": None, "p99": None, "p999": None,
           "max": None, "samples": None}
    if not xs:
        return out
    out["max"] = max(xs)
    if len(xs) < 2:
        out["samples"] = list(xs)                 # a percentile of one sample is the sample
        return out
    out["p50"] = pct(xs, 50)
    out["p95"] = pct(xs, 95)
    if len(xs) >= 100:
        out["p99"] = pct(xs, 99)
    if len(xs) >= 1000:
        out["p999"] = pct(xs, 99.9)
    return out


def _num(v, unit):
    """Seconds and unitless ratios need decimals; milliseconds do not.

    A CER carries no unit, and with the old rule it printed through "%.0f":
    every rate below 0.5 rendered as "0" and every one above as "1". The
    envelope line read "CER p95 0 (limit 0)" while the measurement was 0.111
    against a limit of 0.25 -- a quality gate whose output could not show a
    quality number."""
    if unit == "s":
        return f"{v:.3f}"
    if unit in ("", "x"):
        return f"{v:.3f}"
    return f"{v:.0f}"


def fmt_stat(s):
    """One line for a stat, honest about what it cannot support."""
    u = s["unit"]
    if s["n"] == 0:
        return "n/a (no samples)"
    if s["p50"] is None:
        return " / ".join(_num(v, u) for v in s["samples"]) + f"  ({s['n']} sample)"
    tag = "" if s["reported"] else "   DIAGNOSTIC"
    tail = ""
    if s.get("p99") is not None:
        tail = f" / {_num(s['p99'], u)}"
        if s.get("p999") is not None:
            tail += f" / {_num(s['p999'], u)}"
    return (f"{_num(s['p50'], u)} / {_num(s['p95'], u)}{tail}  "
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

# Written-out language tag: some packs emit one at the head of the text.
_TAG_RE = re.compile(r"<[^<>]{1,12}>")


def normalise(text):
    """The CER normaliser, and the only one in this repo.

    NFKC first, so that a reference typed with composed characters and a
    hypothesis that spells them out are the same string; then lowercase; then
    every Unicode punctuation mark becomes a space, rather than the fixed list
    PUNCT once held -- that list was missing the guillemets and the low quote,
    which are exactly the marks the Spanish and German references use, so the
    same audio scored differently depending on which of this repo's two CER
    implementations ran. PUNCT is kept as the documented floor: a character in
    it is dropped even if a future Unicode table disagrees.

    Deliberately minimal beyond that -- no number expansion, no spelling map --
    so it measures the engine and not a text pipeline."""
    t = unicodedata.normalize("NFKC", text or "")
    t = _TAG_RE.sub(" ", t).lower()
    t = "".join(" " if (ch in PUNCT or unicodedata.category(ch).startswith("P")) else ch
                for ch in t)
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


def reference_for(transcripts, clip):
    """The human reference for a clip path, or None.

    A manifest keys a clip by its path INSIDE the bank ("it/fleurs_1521.wav")
    while the harness plays it by whatever path the caller typed on the command
    line ("samples/it/fleurs_1521.wav"), so the match walks path suffixes. It
    never falls back to the bare file name when that name is ambiguous: eleven
    languages of this repo's own sample bank share `fleurs_1521.wav`, and a
    basename match scored every one of them against whichever language the
    manifest happened to list last. None is the right answer there -- an
    utterance with no reference is reported as having none, which is a fact,
    while a CER against the wrong language is a number that looks like one."""
    if not transcripts or not clip:
        return None
    hit = transcripts.get(clip)
    if hit is not None:
        return hit
    parts = str(clip).replace("\\", "/").split("/")
    for i in range(1, len(parts) + 1):
        hit = transcripts.get("/".join(parts[-i:]))
        if hit is not None:
            return hit
    return None


def wer(hyp, ref):
    """Word error rate over the same normalised text as `cer`. Words are what a
    reader counts; characters are what a CJK or agglutinative reference needs.
    Reporting both costs one edit distance and stops the argument."""
    r = normalise(ref).split()
    h = normalise(hyp).split()
    if not r:
        return None
    return edit_distance(h, r) / len(r)


def cer(hyp, ref):
    """Character error rate over normalised text. None when the reference is empty:
    a rate with an empty denominator is not a large error, it is no measurement."""
    r = normalise(ref)
    h = normalise(hyp)
    if not r:
        return None
    return edit_distance(h, r) / len(r)


# ------------------------------------------------------------------- partial quality
# WHAT THE FIRST PARTIAL IS WORTH, not only when it arrives.
#
# R-4 closed the serving half: publication costs ~0.1 ms at every concurrency, so
# anything that moves the first word moves the DECODER, and a decoder change can
# buy earliness by being wrong. Q-2 already showed one stable runner-up in four is
# wrong, which is why "first nonblank moved earlier" is not a result on its own.
#
# The axis here is AUDIO CONSUMED, not wall clock: the same clip gave the same
# speech-at-first-nonblank on this Mac and on the Axion, clip for clip (F30/E2),
# so these numbers compare two decoder configurations without renting a machine.
# Wall-clock TTFP stays with the server harness, where load belongs.
#
# REVISION IS IMPOSSIBLE HERE, AND THAT IS THE POINT. stream_decode_emit
# publishes `text + chars_emitted`, a byte slice of the running transcript, so a
# client's view only ever grows: there is no retraction to count. The risk is the
# other one -- a wrong prefix that is published and never corrected -- so it is
# measured directly (`first_word_correct`) instead of being hidden behind a
# revision rate that is zero by construction. `published_prefix_divergence` keeps
# the assumption itself honest: if detokenisation ever rewrites a byte the client
# already holds, the concatenation and the library's own string stop matching.


def first_word_of(text):
    """The first normalised word of `text`, or None when there is none yet."""
    w = normalise(text).split()
    return w[0] if w else None


def word_settled(raw):
    """Is the first word of this partial COMPLETE, or still a growing subword?

    It is complete once a boundary has been published after it: either a second
    word has started, or the raw text ends in something `normalise` turns into a
    space (whitespace or punctuation). Until then "I" may yet become "Il", and
    scoring it against the reference would score the tokenizer's chunking."""
    if not raw:
        return False
    if len(normalise(raw).split()) >= 2:
        return True
    last = raw[-1]
    return last.isspace() or last in PUNCT or unicodedata.category(last).startswith("P")


def token_compatible(a, b):
    """Could these be the same word, one of them possibly truncated?

    Prefix-compatible in either direction, on normalised text. A partial that has
    published "sat" against a reference "satellite" is evidence that the right
    word is being produced; it is not yet proof, which is why earliness computed
    this way is reported separately from the settled first word."""
    a, b = normalise(a or ""), normalise(b or "")
    if not a or not b:
        return False
    return a.startswith(b) or b.startswith(a)


def partial_quality(deltas, ref, lib_text=None, onset_s=None, offline=None):
    """Six quantities about ONE streamed utterance, none of them merged.

    `deltas` is the published sequence, in order, each {"t1": audio consumed at
    publication, "text": the delta}. `ref` is the human reference -- the corpus,
    never the model's own offline output, which is reported beside it precisely
    because the two disagree on some clips. `lib_text` is what the library held
    at the end, when the caller can supply it. `onset_s` is speech onset.

    Every "speech_*" is audio minus onset and is None without an onset: an
    utterance with no onset has no speech axis, which is a fact, while silently
    using the audio axis instead would put two different measurements in one
    column."""
    out = {
        "n_deltas": len(deltas),
        "audio_at_first_partial": None, "speech_at_first_partial": None,
        "first_word": None, "audio_at_first_word": None, "speech_at_first_word": None,
        "first_word_correct": None,
        "audio_at_first_evidence": None, "speech_at_first_evidence": None,
        "leading_silence_deltas": 0,
        "published_prefix_divergence": None,
        "text": "", "cer": None, "wer": None,
        "offline_cer": None, "streaming_vs_offline_cer": None,
    }
    if not deltas:
        return out

    def speech(a):
        return None if (a is None or onset_s is None) else a - onset_s

    out["audio_at_first_partial"] = deltas[0].get("t1")
    out["speech_at_first_partial"] = speech(out["audio_at_first_partial"])

    ref_first = first_word_of(ref) if ref else None
    concat = ""
    for d in deltas:
        t1 = d.get("t1")
        if onset_s is not None and t1 is not None and t1 <= onset_s:
            # published having consumed only audio from BEFORE speech began
            out["leading_silence_deltas"] += 1
        concat += d.get("text") or ""
        if out["audio_at_first_evidence"] is None and ref_first:
            fw = first_word_of(concat)
            if fw and token_compatible(fw, ref_first):
                out["audio_at_first_evidence"] = t1
                out["speech_at_first_evidence"] = speech(t1)
        if out["audio_at_first_word"] is None and word_settled(concat):
            out["first_word"] = first_word_of(concat)
            out["audio_at_first_word"] = t1
            out["speech_at_first_word"] = speech(t1)
            if ref_first:
                out["first_word_correct"] = out["first_word"] == ref_first

    out["text"] = concat
    if lib_text is not None:
        out["published_prefix_divergence"] = normalise(concat) != normalise(lib_text)
    if ref:
        out["cer"], out["wer"] = cer(concat, ref), wer(concat, ref)
        if offline is not None:
            out["offline_cer"] = cer(offline, ref)
            out["streaming_vs_offline_cer"] = cer(concat, offline)
    return out


def _rate(rows, pred, guard=None):
    """(rate, numerator, denominator) over the rows where the question applies."""
    elig = [r for r in rows if guard is None or guard(r)]
    if not elig:
        return None, 0, 0
    n = sum(1 for r in elig if pred(r))
    return n / len(elig), n, len(elig)


def partial_quality_summary(rows):
    """Aggregate per-utterance partial_quality() dicts. Rates carry their
    denominator: "1 of 4" and "25 of 100" are not the same evidence."""
    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]

    s = {"n": len(rows)}
    for k in ("speech_at_first_partial", "speech_at_first_word",
              "speech_at_first_evidence", "cer", "wer",
              "streaming_vs_offline_cer", "offline_cer"):
        v = col(k)
        s[k] = {"n": len(v), "median": pct(v, 50), "p95": pct(v, 95),
                "mean": (sum(v) / len(v)) if v else None} if v else None
    for name, pred, guard in (
        ("wrong_first_word", lambda r: r.get("first_word_correct") is False,
         lambda r: r.get("first_word_correct") is not None),
        ("no_first_word", lambda r: r.get("audio_at_first_word") is None, None),
        ("leading_silence", lambda r: (r.get("leading_silence_deltas") or 0) > 0,
         lambda r: r.get("leading_silence_deltas") is not None),
        ("prefix_divergence", lambda r: r.get("published_prefix_divergence") is True,
         lambda r: r.get("published_prefix_divergence") is not None),
        ("disagrees_with_offline", lambda r: (r.get("streaming_vs_offline_cer") or 0) > 0,
         lambda r: r.get("streaming_vs_offline_cer") is not None),
    ):
        rate, num, den = _rate(rows, pred, guard)
        s[name] = {"rate": rate, "n": num, "of": den}
    return s


def analyze_utterance(rec, frame_ms=100.0, pace=1.0, onsets=None):
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
        # The two a latency table alone cannot show: the wait for the END of an
        # utterance, and the longest silence INSIDE one. An utterance can hold a
        # fine p95 while one gap of two seconds sat in the middle of it, and a
        # listener feels that gap and not the percentile.
        "eou_ms": None, "max_gap_ms": None,
        # TTFP decomposed. "4.4 seconds to the first word" is three different
        # claims wearing one number: how long the audio was silent, how much
        # SPEECH the model wanted before it would commit to anything, and how
        # late the server was on top of that. Only the third is a server
        # latency, and the emission lag beside it is usually two orders smaller.
        #   ttfp_from_onset_ms  first partial - the moment speech began
        #   first_delta_audio_s audio the server had CONSUMED when it emitted
        #   first_delta_lag_ms  the server's own lateness on that first frame
        "ttfp_from_onset_ms": None, "first_delta_audio_s": None,
        "first_delta_lag_ms": None, "onset_s": None,
    }
    half_frame = frame_ms / 2.0
    out["paced"] = (out["max_late_ms"] <= half_frame + EPS) and abs(pace - 1.0) < EPS

    first_delta_t = None
    first_frame_t = None
    texts = []
    eou_lat = []
    delta_ts = []
    for ev in events:
        t = ev.get("t")
        kind = ev.get("type")
        if t is not None and (first_frame_t is None or t < first_frame_t):
            first_frame_t = t   # ANY frame: delta, eou, done or error
        if kind == "eou":
            out["eous"] += 1
            if t is not None and sends and ev.get("audio_s") is not None:
                # end-of-utterance latency: the eou frame's arrival against the
                # moment the audio it closes was SENT, not against the run start
                eou_lat.append((t - send_time_for_audio(sends, float(ev["audio_s"]))) * 1000.0)
        if ev.get("lag_ms") is not None and kind in ("delta", "eou"):
            out["server_lag_marks"].append([t, float(ev["lag_ms"])])
        text = ev.get("text") or ""
        if kind != "delta" or not text:
            continue
        texts.append(text)
        if t is not None:
            delta_ts.append(t)
        if first_delta_t is None:
            first_delta_t = t
            out["first_delta_audio_s"] = ev.get("audio_s")
            out["first_delta_lag_ms"] = ev.get("lag_ms")
        consumed = ev.get("audio_s")
        if consumed is not None and sends:
            t_send = send_time_for_audio(sends, float(consumed))
            out["lag_marks"].append([t, (t - t_send) * 1000.0])
            out["backlog_marks"].append([t, audio_sent_at(sends, t) - float(consumed)])

    out["deltas"] = len(texts)
    out["text"] = "".join(texts)
    if eou_lat:
        out["eou_ms"] = max(eou_lat)
    if len(delta_ts) >= 2:
        ds = sorted(delta_ts)
        out["max_gap_ms"] = max((b - a) * 1000.0 for a, b in zip(ds, ds[1:]))
    if out["backlog_marks"]:
        out["backlog_max_s"] = max(m[1] for m in out["backlog_marks"])
    if first_frame_t is not None and sends:
        out["ttfb_ms"] = (first_frame_t - sends[0][0]) * 1000.0
    if first_delta_t is not None and sends:
        out["ttfp_ms"] = (first_delta_t - sends[0][0]) * 1000.0
        # From SPEECH ONSET, when the caller knows where the speech starts.
        # A clip with two seconds of room tone in front of it is not a server
        # that took two extra seconds, and until this line existed nothing in
        # the harness could tell the two apart.
        onset = onsets.get(rec.get("clip")) if onsets else None
        if onset is None and onsets:
            onset = reference_for(onsets, rec.get("clip") or "")
        if onset is not None:
            out["onset_s"] = float(onset)
            t_onset = send_time_for_audio(sends, float(onset))
            out["ttfp_from_onset_ms"] = (first_delta_t - t_onset) * 1000.0
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
            # same matching rule as the CER path: a bare file name is not a key
            exp = reference_for(reference, c)
            if exp is not None and texts != [exp]:
                ref_fail[c] = {"expected": exp, "got": texts}

    # Quality, as opposed to speed. CER is per UTTERANCE against the bank's human
    # reference; it is NOT gated by pacing, because whether the transcript is right does
    # not depend on whether the client held its schedule -- only the cadence numbers do.
    cers, cer_worst = [], None
    if transcripts:
        for u in ok:
            clip = u.get("clip") or ""
            ref = reference_for(transcripts, clip)
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

    # Throughput as a unit of WORK: audio seconds carried per wall second. A rate
    # of requests says nothing when the clips differ in length, and this is the
    # number that answers "how many real-time streams does this host carry".
    m["audio_per_wall"] = stat([audio_s / span] if span > EPS and audio_s > 0 else [],
                               "run", "x", True, "MEASURED")
    m["eou_ms"] = stat([u["eou_ms"] for u in ok if u.get("eou_ms") is not None],
                       "per utterance", "ms", rep, lbl)
    # The three parts of TTFP, reported apart so the number can be argued with.
    m["ttfp_from_onset_ms"] = stat(
        [u["ttfp_from_onset_ms"] for u in ok if u.get("ttfp_from_onset_ms") is not None],
        "per utterance (from speech onset)", "ms", rep, lbl)
    m["first_delta_audio_s"] = stat(
        [u["first_delta_audio_s"] for u in ok if u.get("first_delta_audio_s") is not None],
        "per utterance (audio the model wanted before it spoke)", "s", True, "MEASURED")
    m["first_delta_lag_ms"] = stat(
        [u["first_delta_lag_ms"] for u in ok if u.get("first_delta_lag_ms") is not None],
        "per utterance (server lateness on the first frame)", "ms", rep, lbl)
    m["max_delta_gap_ms"] = stat([u["max_gap_ms"] for u in ok if u.get("max_gap_ms") is not None],
                                 "per utterance (longest silence in each)", "ms", rep, lbl)

    # Fairness. A fleet p95 hides one starved stream among thirty healthy ones,
    # and a starved stream is a person hearing nothing. Per stream, the p95 of
    # its own deltas; then the WORST stream against the MEDIAN stream. A ratio
    # of 1 is a fleet that treats its streams alike; the run-level p95 cannot
    # tell 1 from 5.
    per_stream = {}
    for u in ok:
        sid = u.get("stream")
        if sid is None or len(u["lag_marks"]) < 2:
            continue
        per_stream.setdefault(sid, []).extend(v for _, v in u["lag_marks"])
    stream_p95 = sorted(p for p in (pct(v, 95) for v in per_stream.values()) if p is not None)
    fairness = None
    if len(stream_p95) >= 2:
        med = pct(stream_p95, 50)
        fairness = {"streams": len(stream_p95), "median_p95_ms": med,
                    "worst_p95_ms": stream_p95[-1], "best_p95_ms": stream_p95[0],
                    "worst_over_median": (stream_p95[-1] / med) if med and med > EPS else None}
    # (fairness is not a stat dict: it goes in its own key of the summary)
    return {
        "counts": {"utterances": len(counted), "ok": len(ok), "errors": len(errored),
                   "rejected": len(rejected), "warmup_excluded": warm,
                   "deltas": sum(u["deltas"] for u in ok), "eous": sum(u["eous"] for u in ok),
                   "audio_s": audio_s, "span_s": span},
        "pacing": {"max_late_ms": max_late, "half_frame_ms": frame_ms / 2.0, "pace": pace,
                   "paced": paced,
                   "verdict": "PACED" if paced else "NOT PACED (cadence is DIAGNOSTIC)"},
        "metrics": m,
        "fairness": fairness,
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
        # Spread of a p95 ACROSS STEADY WINDOWS, and the direction it moves.
        #
        # 20 % stays. It was inherited rather than derived, and a p95 over ~1800
        # samples genuinely has more sampling variance than a median -- but
        # "inherited" is not "wrong", and the moment to change a threshold is not
        # while looking at the first run that fails it. The number to replace it
        # with should come from the distribution of healthy runs, not from a
        # sibling project's choice for a different metric.
        #
        # Frozen here as the evidence that will decide it, from the 1x24 c=16
        # 600 s soak of 2026-09-20 (zero losses, pooled lag p95 87 ms against a
        # 320 ms envelope, backlog 0.484 of 0.640):
        #
        #     old metric     192 %   contaminated by ramp and drain windows
        #     steady spread   36 %   ramp/drain excluded
        #     steady trend   -14 %   improving, not degrading
        #
        # One healthy run is not a distribution. Collect more soaks first.
        "max_drift_pct": 20.0,
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
                       ("ttfp_from_onset_ms", "TTFP from speech onset"),
                       ("first_delta_audio_s", "audio before the first word"),
                       ("first_delta_lag_ms", "server lag on the first frame"),
                       ("eou_ms", "end-of-utterance lag"),
                       ("max_delta_gap_ms", "longest gap in a stream"),
                       ("audio_per_wall", "audio per wall second"),
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
    #  the rank is p*n/100, not (p/100)*n: 99.9 of 1..1000 is 999, not the maximum
    bad += not _eq(pct(list(range(1, 1001)), 99.9), 999, "fractional rank does not round up")
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
    # The marks the old fixed list missed, which is why there were two CER numbers.
    bad += not _eq(normalise("«casa» „Haus“ ¿qué?"),
                   "casa haus qué", "normalise drops guillemets and the low quote")
    bad += not _eq(normalise("<it> Ciao"), "ciao", "normalise strips a written-out tag")
    bad += not _eq(normalise("ﬁn"), "fin", "normalise applies NFKC")

    print("TTFP decomposed: silence, evidence, and the server's own lateness")
    #  two seconds of room tone, the model commits after 3.5 s of audio, and the
    #  server is 120 ms late on that frame: one number, three different causes
    snd_d = [[0.1 * i, 0.1 * (i + 1)] for i in range(60)]
    rec_d = _mk("it/x.wav", snd_d, [_ev(3.62, 3.5, "ciao ", lag_ms=120.0)], 6.2, t_start=0.0)
    ud = analyze_utterance(rec_d, frame_ms=100.0, onsets={"it/x.wav": 2.0})
    bad += not _eq(ud["ttfp_ms"], 3620.0, "TTFP from the stream open", 1e-6)
    bad += not _eq(ud["ttfp_from_onset_ms"], 1720.0, "TTFP from speech onset", 1e-6)
    bad += not _eq(ud["first_delta_audio_s"], 3.5, "audio the model wanted", 1e-9)
    bad += not _eq(ud["first_delta_lag_ms"], 120.0, "server lateness on that frame", 1e-9)
    #  no onset map: the decomposition is absent, never guessed
    un = analyze_utterance(rec_d, frame_ms=100.0)
    bad += not _eq(un["ttfp_from_onset_ms"], None, "no onset map, no onset-relative number")

    print("tails, throughput and fairness")
    #   p99 at nearest rank needs 100 samples to name a rank below the maximum
    small, big = list(range(1, 51)), list(range(1, 1001))
    bad += not _eq(stat(small, "t")["p99"], None, "p99 withheld below 100 samples")
    bad += not _eq(stat(big, "t")["p99"], 990, "p99 of 1..1000 is 990")
    bad += not _eq(stat(big, "t")["p999"], 999, "p999 of 1..1000 is 999")
    bad += not _eq(stat(small, "t")["p999"], None, "p999 withheld below 1000 samples")
    bad += not _eq(wer("the cat sat", "The cat sat."), 0.0, "WER folds case and stops", 1e-12)
    bad += not _eq(wer("the bat sat", "the cat sat"), 1.0 / 3.0, "one word of three", 1e-12)
    bad += not _eq(wer("x", ""), None, "an empty reference has no word rate")

    #   one starved stream among healthy ones: the fleet p95 cannot see it, the ratio can
    def _st(stream, lag_ms, t0):
        snd = [[t0 + 0.1 * i, 0.1 * (i + 1)] for i in range(12)]
        evs = [_ev(t0 + 0.1 * i + lag_ms / 1000.0, 0.1 * (i + 1), "w ") for i in range(12)]
        r = _mk("c.wav", snd, evs, t0 + 1.3, t_start=t0)
        r["stream"] = stream
        return analyze_utterance(r, frame_ms=100.0)

    ag = aggregate([_st(0, 10.0, 0.0), _st(1, 10.0, 2.0), _st(2, 300.0, 4.0)],
                   frame_ms=100.0, pace=1.0)
    fr = ag["fairness"]
    bad += not _eq(float(fr["streams"]), 3.0, "three streams got their own p95")
    bad += not _eq(fr["median_p95_ms"], 10.0, "the median stream is healthy", 1e-6)
    bad += not _eq(fr["worst_p95_ms"], 300.0, "the worst stream is not", 1e-6)
    bad += not _eq(fr["worst_over_median"], 30.0, "worst over median names the starvation", 1e-6)
    bad += not _eq(ag["metrics"]["max_delta_gap_ms"]["max"], 100.0,
                   "the longest gap inside an utterance", 1e-6)

    print("quality: a reference is matched by path suffix, never by an ambiguous name")
    tr = {"it/fleurs_1521.wav": "ciao", "de/fleurs_1521.wav": "hallo", "solo.wav": "alone"}
    bad += not _eq(reference_for(tr, "samples/it/fleurs_1521.wav"), "ciao",
                   "a longer played path matches the manifest key")
    bad += not _eq(reference_for(tr, "de/fleurs_1521.wav"), "hallo", "an exact key matches")
    bad += not _eq(reference_for(tr, "somewhere/fleurs_1521.wav"), None,
                   "an ambiguous bare name scores nothing rather than the wrong language")
    bad += not _eq(reference_for(tr, "bank/solo.wav"), "alone",
                   "an unambiguous bare name still matches")
    bad += not _eq(reference_for(None, "a.wav"), None, "no manifest, no reference")

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

    # ------------------------------------------------------- partial quality
    # Fixed synthetic transcripts, chosen so each assertion can only pass for
    # one reason. A pairing, percentile or oracle bug in this file has already
    # changed a conclusion once (the E3 publication delay came out NEGATIVE from
    # one mispaired row); these are the known answers that stop the next one.
    print("partial quality: first word, earliness, and what the gate refuses to guess")
    REF = "Il satellite nello spazio riceve il segnale."

    def d(t1, text):
        return {"t1": t1, "text": text}

    # the first word arrives as two subwords: settled only when a boundary follows
    q = partial_quality([d(0.80, "I"), d(1.10, "l"), d(1.40, " satellite")],
                        REF, onset_s=0.30)
    bad += not _eq(q["audio_at_first_partial"], 0.80, "audio at the first partial")
    bad += not _eq(q["speech_at_first_partial"], 0.50, "speech at the first partial")
    bad += not _eq(q["audio_at_first_evidence"], 0.80, "compatible evidence exists at the first subword")
    bad += not _eq(q["audio_at_first_word"], 1.40, "but the first word is not settled until a boundary follows")
    bad += not _eq(q["first_word"], "il", "and it is the whole word, not the subword")
    bad += not _eq(q["first_word_correct"], True, "first word correct")
    bad += not _eq(q["leading_silence_deltas"], 0, "nothing published before onset")

    # a wrong first word is published and, being a byte slice, never taken back
    q = partial_quality([d(0.80, "La "), d(1.40, "satellite")], REF, onset_s=0.30)
    bad += not _eq(q["first_word"], "la", "a wrong first word is reported as what it is")
    bad += not _eq(q["first_word_correct"], False, "and marked wrong")
    bad += not _eq(q["audio_at_first_evidence"], None, "with no compatible evidence at any point")

    # earliness alone must not read as a win: earlier AND wrong is the Q-2 case
    early = partial_quality([d(0.50, "La "), d(1.0, "satellite")], REF, onset_s=0.30)
    late = partial_quality([d(0.90, "Il "), d(1.4, "satellite")], REF, onset_s=0.30)
    bad += not _eq(early["speech_at_first_word"] < late["speech_at_first_word"], True,
               "the wrong arm IS earlier -- the summary must separate the two axes")
    summ = partial_quality_summary([early, late])
    bad += not _eq(summ["wrong_first_word"]["rate"], 0.5, "and the wrong-first-word rate says so")
    bad += not _eq(summ["wrong_first_word"]["of"], 2, "over the utterances where the question applies")

    # text on pre-onset audio only: a hallucination in the leading silence
    q = partial_quality([d(0.20, "Ehm "), d(1.40, "il satellite")], REF, onset_s=0.30)
    bad += not _eq(q["leading_silence_deltas"], 1, "a delta published before onset is counted")

    # the client's concatenation vs what the library holds
    q = partial_quality([d(0.8, "Il "), d(1.4, "satellite")], REF, lib_text="Il satellite")
    bad += not _eq(q["published_prefix_divergence"], False, "concatenation matches the library")
    q = partial_quality([d(0.8, "Il "), d(1.4, "satellite")], REF, lib_text="Lo satellite")
    bad += not _eq(q["published_prefix_divergence"], True, "and a rewritten byte is detected")

    # no onset: the speech axis must be absent, not silently the audio axis
    q = partial_quality([d(0.80, "Il "), d(1.4, "satellite")], REF)
    bad += not _eq(q["speech_at_first_partial"], None, "no onset -> no speech axis")
    bad += not _eq(q["audio_at_first_partial"], 0.80, "the audio axis still exists")

    # no reference: correctness is unknown, which is not the same as wrong
    q = partial_quality([d(0.80, "Il "), d(1.4, "satellite")], None, onset_s=0.3)
    bad += not _eq(q["first_word_correct"], None, "no reference -> correctness unknown, not False")
    bad += not _eq(partial_quality_summary([q])["wrong_first_word"]["of"], 0,
               "and it is excluded from the denominator")

    # an utterance that published nothing
    q = partial_quality([], REF, onset_s=0.3)
    bad += not _eq(q["n_deltas"], 0, "an utterance with no delta reports none")
    bad += not _eq(q["audio_at_first_partial"], None, "and no first-partial time")
    bad += not _eq(partial_quality_summary([q])["no_first_word"]["rate"], 1.0,
               "and counts as having produced no first word")

    # streaming vs offline vs reference stay three separate numbers
    q = partial_quality([d(0.8, "Il satellite nello spazio riceve il segnale.")],
                        REF, offline="Il satellite nello spazio riceve il segnale")
    bad += not _eq(q["cer"], 0.0, "streaming against the corpus")
    bad += not _eq(q["offline_cer"], 0.0, "offline against the corpus")
    bad += not _eq(q["streaming_vs_offline_cer"], 0.0, "and the two against each other")
    q = partial_quality([d(0.8, "Il satellite nello spasio")], REF,
                        offline="Il satellite nello spazio")
    bad += not _eq(q["streaming_vs_offline_cer"] > 0, True,
               "a streaming/offline disagreement is not folded into the corpus CER")
    bad += not _eq(q["offline_cer"] < q["cer"], True,
               "and the offline arm keeps its own, better, number")

    # the normaliser is shared: punctuation and case cannot decide a first word
    bad += not _eq(token_compatible("Il,", "il"), True, "token compatibility uses the one normaliser")
    bad += not _eq(word_settled("Il"), False, "a bare token is not settled")
    bad += not _eq(word_settled("Il "), True, "a trailing space settles it")
    bad += not _eq(word_settled("Il,"), True, "so does punctuation")

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
