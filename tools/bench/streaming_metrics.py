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
      "error_kind": str|None,          # one of OUTCOMES (below); absent in old records
      "rejected": bool,                # HTTP 503 before the upgrade: a legitimate outcome
    }

`sends[0][0]` is the first-audio-sent time.

------------------------------------------------------------------------- accounting
Every utterance a stream STARTED ends in exactly one outcome of `OUTCOMES`: `ok`,
`rejected` (a 503: counted, never a loss), `cut_at_deadline` (in flight when a soak's
collection closed: counted, never a loss, never dropped) or one ERROR kind. `ok` needs a
`done` frame: an utterance the server closed without one, or that received nothing at
all, is an error (`server_disconnect` / `no_done` / `no_events`), never a success whose
finalization quietly left the percentiles. Errors are counted over the WHOLE run, warm-up
included; only the latency percentiles skip the warm-up. `aggregate()` then checks the
conservation invariant

    started = ok + rejected + cut_at_deadline + sum(errors by kind)

and a run where it does not hold is INVALID: an utterance that vanished from the
statistics is a failure of the harness, and a harness that loses utterances can also
lose the errors in them.  Nothing here reads a socket or a clock: the
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


# ------------------------------------------------------------------ number format
# A QUARTER OF THE FLEURS BANK IS SCORED ON A CONVENTION, NOT ON RECOGNITION.
#
# FLEURS references keep digits: "Since 1966 ... 400 Royal Bengal tigers".
# This checkpoint verbalises: "since nineteen sixty-six ... four hundred royal
# bengal tigers". Measured on the frozen baseline, the utterances whose
# reference contains a digit are 23 % of the bank and carry WER 0.246 (EN) and
# 0.253 (FR) against 0.083 and 0.097 for the rest -- and they account for 76 of
# 105 English insertions and 100 of 137 French ones, because one reference token
# "2007" becomes three hypothesis tokens.
#
# THIS IS NOT LOOSENING THE SCORER. The words must still be right: only the
# written form of a number is forgiven, and only among forms that are legitimate
# spoken renderings of THAT number. "2007" may be read as "two thousand seven"
# or "twenty oh seven"; it may not become "two thousand eight". A strict score
# is reported beside it, always, and neither replaces the other.
#
# Standard library only, like the rest of this file: no num2words on a bare box.

_EN_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety")
_FR_ONES = ("zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize "
            "quatorze quinze seize dix-sept dix-huit dix-neuf").split()
_FR_TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}


def _en_below_1000(n):
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _EN_TENS[t] + (f" {_EN_ONES[r]}" if r else "")
    h, r = divmod(n, 100)
    return _EN_ONES[h] + " hundred" + (f" {_en_below_1000(r)}" if r else "")


def _fr_below_100(n):
    if n < 20:
        return _FR_ONES[n]
    if n < 70:
        t, r = divmod(n, 10)
        base = _FR_TENS[t * 10]
        if r == 1:
            return base + " et un"
        return base + (f" {_FR_ONES[r]}" if r else "")
    if n < 80:                                   # soixante-dix .. soixante-dix-neuf
        r = n - 60
        return "soixante" + (" et onze" if r == 11 else f" {_FR_ONES[r]}")
    r = n - 80
    if r == 0:
        return "quatre vingts"
    return "quatre vingt " + _FR_ONES[r]


def _fr_below_1000(n):
    if n < 100:
        return _fr_below_100(n)
    h, r = divmod(n, 100)
    head = "cent" if h == 1 else f"{_FR_ONES[h]} cent" + ("s" if r == 0 else "")
    return head + (f" {_fr_below_100(r)}" if r else "")


def _cardinal(n, lang):
    """0..999,999. Beyond that the reference is not a number a speaker reads out."""
    if n >= 1000000:
        return None
    th, r = divmod(n, 1000)
    if lang == "fr":
        if not th:
            return _fr_below_1000(r)
        head = "mille" if th == 1 else f"{_fr_below_1000(th)} mille"
        return head + (f" {_fr_below_1000(r)}" if r else "")
    if not th:
        return _en_below_1000(r)
    head = f"{_en_below_1000(th)} thousand"
    return head + (f" {_en_below_1000(r)}" if r else "")


def _year_forms(n, lang):
    """A four-digit number a speaker may read in pairs: 1966 -> nineteen sixty-six."""
    if not (1100 <= n <= 2099):
        return []
    hi, lo = divmod(n, 100)
    if lang == "fr":
        return []                                # French reads years as cardinals
    if lo == 0:
        return [f"{_en_below_1000(hi)} hundred"]
    if lo < 10:
        # "twenty oh seven", never "twenty seven": that is a different number.
        return [f"{_en_below_1000(hi)} oh {_EN_ONES[lo]}"]
    return [f"{_en_below_1000(hi)} {_en_below_1000(lo)}"]


_NUM_RE = re.compile(r"\d[\d  \u00a0\u202f.,]*\d|\d")


def number_variants(text, lang):
    """Every legitimate spoken rendering of `text`, digits expanded. The written
    form is kept as one of them, so a model that emits digits is not punished
    either. Returns a list, always non-empty, capped so a sentence full of
    numbers cannot explode."""
    spans = list(_NUM_RE.finditer(text))
    if not spans:
        return [text]
    outs = [text]
    for _ in range(1):                          # one pass: all numbers together
        for pick_year in (False, True):
            buf, last = [], 0
            for m in spans:
                raw = m.group(0)
                digits = re.sub(r"[^\d]", "", raw)
                if not digits or len(digits) > 6:
                    continue
                n = int(digits)
                forms = (_year_forms(n, lang) if pick_year else []) or [_cardinal(n, lang)]
                if not forms or forms[0] is None:
                    continue
                buf.append(text[last:m.start()])
                buf.append(forms[0])
                last = m.end()
            if buf:
                buf.append(text[last:])
                cand = "".join(buf)
                if cand not in outs:
                    outs.append(cand)
    return outs


def wer_format_free(hyp, ref, lang):
    """The best WER over the legitimate spoken renderings of the reference's
    numbers. Charitable about FORM, never about content: every word still has to
    be the right word. Report it beside the strict WER, never instead of it."""
    vals = [wer(hyp, v) for v in number_variants(ref, lang)]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def cer_format_free(hyp, ref, lang):
    vals = [cer(hyp, v) for v in number_variants(ref, lang)]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def align_counts(hyp, ref):
    """Substitutions, deletions and insertions over the normalised WORD sequence.

    A WER is one number and three different failures produce it. A model that
    drops the end of every utterance and one that hallucinates a clause both
    score 0.30, and only the split says which. Deletions are what an ASR under
    pressure does; insertions are what a hallucinating decoder does; they are not
    the same defect and must not be averaged.

    Backtracking Levenshtein over words: O(len(h) x len(r)) cells, which is fine
    for utterances and is never run per frame."""
    r = normalise(ref).split()
    h = normalise(hyp).split()
    if not r:
        return None
    n, m = len(h), len(r)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (h[i - 1] != r[j - 1]))
    sub = dele = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (h[i - 1] != r[j - 1]):
            sub += h[i - 1] != r[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ins += 1                       # a word in the hypothesis with no reference
            i -= 1
        else:
            dele += 1                      # a reference word the hypothesis never said
            j -= 1
    return {"sub": sub, "del": dele, "ins": ins, "ref_words": m, "hyp_words": n}


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
        # Q-5: today this reads 0 on every corpus we have, and that zero is the
        # point. A fine-tune that starts producing empties would otherwise show
        # up only as a WER that drifted.
        ("empty_transcript", lambda r: not normalise(r.get("text") or ""), None),
    ):
        rate, num, den = _rate(rows, pred, guard)
        s[name] = {"rate": rate, "n": num, "of": den}
    return s


# ------------------------------------------------------------------ outcome taxonomy
#
# A fixed set, so that "0 errors" can be read back as "0 of each of these" and a new way
# of failing has to be named here before it can be counted. `unclassified` exists so that
# an error string this module does not recognise is still an ERROR, never an OK.

OK_OUTCOME = "ok"
NOT_ERRORS = ("ok", "rejected", "cut_at_deadline")
ERROR_KINDS = (
    "connect_error",         # TCP connect / handshake failed before any HTTP status
    "http_error",            # refused before the upgrade with a status other than 503
    "timeout",               # no `done` within --done-timeout, or a socket read timed out
    "server_disconnect",     # the server closed (close frame, EOF, reset) before `done`
    "server_error",          # the server sent an `error` frame
    "no_done",               # events arrived, the utterance ended, `done` never came
    "no_events",             # the utterance ended and not one server frame arrived
    "protocol_error",        # a frame the client could not parse
    "client_exception",      # the harness itself raised while running the utterance
    "client_process_death",  # the stream process died with this utterance in flight
    "unclassified",          # an error string no rule below recognises
)
OUTCOMES = NOT_ERRORS + ERROR_KINDS


def _kind_from_message(err, rejected=False):
    """Classify an error string written by a harness that predates `error_kind`.

    Only used for old records: the client now names the kind where the failure happens."""
    e = (err or "").lower()
    if rejected:
        return "rejected"
    if e.startswith("refused"):
        return "http_error"
    if e.startswith("connect"):
        return "connect_error"
    if "timed out" in e or "timeout" in e:
        return "timeout"
    if e.startswith("server error"):
        return "server_error"
    if any(s in e for s in ("connection closed", "reset", "broken pipe", "close frame",
                            "aborted", "without done")):
        return "server_disconnect"
    if "expecting" in e or "json" in e or "decode" in e:
        return "protocol_error"
    return "unclassified"


def outcome(rec):
    """The one outcome of an utterance record (raw or analysed), from `OUTCOMES`.

    `ok` requires a `done`: an utterance with no error string and no `done` frame used to
    count as OK -- the reader broke on a close frame and recorded nothing -- and its
    finalization silently left the percentiles (AUDIT 2026-09-24, gap 1)."""
    if rec.get("outcome") in OUTCOMES:
        return rec["outcome"]                     # already decided by analyze_utterance
    if rec.get("rejected"):
        return "rejected"
    k = rec.get("error_kind")
    if k:
        return k if k in OUTCOMES else "unclassified"
    if rec.get("error"):
        return _kind_from_message(rec["error"])
    if rec.get("done_t") is None:
        return "no_events" if not rec.get("events") else "no_done"
    return OK_OUTCOME


def is_error(kind):
    return kind not in NOT_ERRORS


def analyze_utterance(rec, frame_ms=100.0, pace=1.0, onsets=None):
    """Every per-utterance metric.  Marks are [t, value] so they can be windowed later."""
    sends = [list(s) for s in rec.get("sends") or []]
    events = rec.get("events") or []
    late = [float(x) for x in (rec.get("late_ms") or [])]
    kind = outcome(rec)
    err = rec.get("error")
    if kind != OK_OUTCOME and not err:
        # an outcome that is not OK always carries an error string too, so that every
        # consumer that filters on `error` (text identity, paired TTFP, event keeping)
        # drops it without having to know the taxonomy
        err = {"no_done": "no done frame before the utterance ended",
               "no_events": "no server frame at all"}.get(kind, kind)
    out = {
        "clip": rec.get("clip"), "audio_s": float(rec.get("audio_s") or 0.0),
        "stream": rec.get("stream"), "rep": rec.get("rep"),
        "t_start": rec.get("t_start"), "frames": len(sends),
        "error": err, "error_kind": None if kind == OK_OUTCOME else kind, "outcome": kind,
        "rejected": bool(rec.get("rejected")),
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


# ---------------------------------------------------------------------------- reconcile
#
# The client side of the conservation invariant. A stream process announces every
# utterance BEFORE it plays it (a start mark: stream, rep, t, clip) and announces its own
# end when it leaves its loop (an end mark: stream, t, started, reason). The parent adds
# what only it can see: the exit code, and whether it had to terminate a process that
# was still alive when collection closed. From those three sources nothing can vanish:
#
#   * a start with no record, from a stream the parent terminated after a SOAK's
#     collection closed, is `cut_at_deadline` -- counted, not an error;
#   * any other start with no record is `client_process_death` -- an ERROR;
#   * a stream with no end mark that the parent did not terminate died; so did one that
#     exited non-zero; in SOAK so did one whose last sign of life came earlier than the
#     deadline minus one utterance (`tol_s`). WAVE's analogue is a stream that started
#     fewer than `repeat` utterances.


def reconcile(records, starts, status=None, n_streams=None, deadline=None, tol_s=0.0,
              repeat=None):
    """(synthetic records for every start that produced none, the per-stream report).

    `records` are raw (or analysed) utterance dicts carrying `stream` and `rep`;
    `starts` is [{"stream", "rep", "t", "clip"}]; `status` is
    {stream: {"end_t": float|None, "started": int|None, "exitcode": int|None,
              "terminated": bool}}. `deadline` is ABSOLUTE (same clock as the marks) and
    only given in SOAK; `repeat` only in WAVE. Pure arithmetic, so the self-test can
    plant every fault."""
    status = status or {}
    have = {(r.get("stream"), r.get("rep")) for r in records}
    synth = []
    for s in starts:
        key = (s.get("stream"), s.get("rep"))
        if key in have:
            continue
        st = status.get(s.get("stream")) or {}
        cut = bool(st.get("terminated")) and deadline is not None
        kind = "cut_at_deadline" if cut else "client_process_death"
        synth.append({"clip": s.get("clip"), "class": s.get("class"), "stream": key[0],
                      "rep": key[1], "t_start": s.get("t"), "audio_s": 0.0, "sends": [],
                      "late_ms": [], "events": [], "done_t": None, "rejected": False,
                      "error": ("in flight when the soak's collection closed" if cut else
                                "the stream process ended with this utterance in flight"),
                      "error_kind": kind, "synthetic": True})
        have.add(key)

    ids = sorted(set(range(n_streams)) if n_streams is not None else
                 {s.get("stream") for s in starts} | set(status))
    last_seen, n_started = {}, {}
    for s in starts:
        sid = s.get("stream")
        n_started[sid] = n_started.get(sid, 0) + 1
        if s.get("t") is not None:
            last_seen[sid] = max(last_seen.get(sid, s["t"]), s["t"])
    for r in records:
        sid = r.get("stream")
        for t in (r.get("t_end"), r.get("done_t"), r.get("t_start")):
            if t is not None:
                last_seen[sid] = max(last_seen.get(sid, t), t)
    dead, short, terminated, rows = [], [], [], {}
    for sid in ids:
        st = status.get(sid)
        end_t = (st or {}).get("end_t")
        if end_t is not None:
            last_seen[sid] = max(last_seen.get(sid, end_t), end_t)
        why = None
        if st is not None:
            term = bool(st.get("terminated"))
            if term:
                terminated.append(sid)
            code = st.get("exitcode")
            if end_t is None and not term:
                why = "exited without its end mark"
            elif code not in (0, None) and not term:
                why = f"exit code {code}"
        elif status:
            why = "no status: the process was never seen"
        if why is None and deadline is not None and not (st or {}).get("terminated"):
            seen = last_seen.get(sid)
            if seen is None or seen < deadline - tol_s:
                why = ("no sign of life at all" if seen is None else
                       f"last sign of life {deadline - seen:.1f} s before the deadline")
        if why is None and repeat is not None and n_started.get(sid, 0) < repeat:
            why = f"started {n_started.get(sid, 0)} of {repeat} utterance(s)"
        if why is not None:
            (short if st is not None and end_t is not None else dead).append(sid)
        rows[sid] = {"started": n_started.get(sid, 0), "last_seen": last_seen.get(sid),
                     "died": why}
    report = {"expected": len(ids), "dead": sorted(dead + short),
              "exited_early": sorted(short), "terminated_at_deadline": sorted(terminated),
              "started": sum(n_started.values()), "per_stream": rows,
              "tol_s": tol_s}
    return synth, report


# ----------------------------------------------------------------------------- aggregate


def group_texts(utts):
    """clip -> sorted distinct texts.  The same clip must always produce the same text."""
    by_clip = {}
    for u in utts:
        if u.get("error") or u.get("rejected"):
            continue
        by_clip.setdefault(u["clip"], set()).add(u["text"])
    return {c: sorted(t) for c, t in by_clip.items()}


def accounting(utts, started=None, streams=None, warmup_s=0.0, t0=0.0):
    """Outcome counts over the WHOLE run, the warm-up split, and the conservation check.

    The warm-up is a KPI window, not an amnesty: until 2026-09-25 `aggregate()` dropped the
    first `--warmup` seconds BEFORE counting errors, so a failure in the first 30 s of a
    soak never reached `counts.errors` (AUDIT 2026-09-24, gap 2). Here nothing is dropped;
    the warm-up share is reported beside the total."""
    by_kind, by_kind_warm = {}, {}
    n = {k: 0 for k in NOT_ERRORS}
    n_warm = {k: 0 for k in NOT_ERRORS}
    for u in utts:
        k = outcome(u)
        in_warm = u.get("t_start") is not None and u["t_start"] - t0 < warmup_s - EPS
        if is_error(k):
            by_kind[k] = by_kind.get(k, 0) + 1
            if in_warm:
                by_kind_warm[k] = by_kind_warm.get(k, 0) + 1
        else:
            n[k] += 1
            if in_warm:
                n_warm[k] += 1
    errors = sum(by_kind.values())
    total = n["ok"] + n["rejected"] + n["cut_at_deadline"] + errors
    if started is None:
        holds, reason = None, "no start marks (a caller or harness that predates them)"
    elif total == started and len(utts) == started:
        holds, reason = True, None
    else:
        holds = False
        reason = (f"started {started} != ok {n['ok']} + rejected {n['rejected']} + "
                  f"cut {n['cut_at_deadline']} + errors {errors} = {total}"
                  + (f" ({len(utts)} records)" if len(utts) != total else ""))
    deaths = None if streams is None else len(streams.get("dead") or [])
    return {
        "started": started, "ok": n["ok"], "rejected": n["rejected"],
        "cut_at_deadline": n["cut_at_deadline"], "errors": errors,
        "error_kinds": dict(sorted(by_kind.items())),
        "errors_warmup": sum(by_kind_warm.values()),
        "error_kinds_warmup": dict(sorted(by_kind_warm.items())),
        "rejected_warmup": n_warm["rejected"], "ok_warmup": n_warm["ok"],
        "stream_deaths": deaths, "conservation": holds, "conservation_detail": reason,
    }


def aggregate(utts, frame_ms=100.0, pace=1.0, window_s=None, warmup_s=0.0, t0=None,
              reference=None, transcripts=None, started=None, streams=None):
    """Every run-level number.  `utts` are the dicts `analyze_utterance` returned.

    `started` is the number of utterances the streams announced (their start marks) and
    `streams` the report of `reconcile()`; both None for a caller that has neither, in
    which case the conservation check reports itself as not checked instead of passing."""
    if t0 is None:
        starts = [u["t_start"] for u in utts if u.get("t_start") is not None]
        t0 = min(starts) if starts else 0.0

    acct = accounting(utts, started=started, streams=streams, warmup_s=warmup_s, t0=t0)
    counted = [u for u in utts
               if u.get("t_start") is None or u["t_start"] - t0 >= warmup_s - EPS]
    warm = len(utts) - len(counted)
    kinds = [outcome(u) for u in counted]
    rejected = [u for u, k in zip(counted, kinds) if k == "rejected"]
    errored = [u for u, k in zip(counted, kinds) if is_error(k)]
    ok = [u for u, k in zip(counted, kinds) if k == OK_OUTCOME]

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
    cers, wers, cer_worst = [], [], None
    if transcripts:
        for u in ok:
            clip = u.get("clip") or ""
            ref = reference_for(transcripts, clip)
            if ref is None:
                continue
            w = wer(u["text"], ref)
            if w is not None:
                wers.append(w)
                u["wer"] = w
            v = cer(u["text"], ref)
            if v is None:
                continue
            u["cer"] = v
            cers.append(v)
            if cer_worst is None or v > cer_worst[0]:
                cer_worst = (v, clip, u["text"], ref)
    m["cer"] = stat(cers, "per utterance", "", True, "MEASURED")
    # Beside CER because they fail differently: a dropped word is one deletion
    # in WER and a dozen characters in CER, and a serving campaign wants to see
    # both rather than argue about which one is the metric.
    m["wer"] = stat(wers, "per utterance", "", True, "MEASURED")
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
    # The CLIENT's own lateness, distributed rather than reduced to a flag. `paced`
    # says whether any utterance missed its schedule by more than half a frame;
    # this says by how much, across the population, which is what an experiment
    # on the generator's own resources has to compare between arms. Reported
    # ALWAYS, including on a run that did not pace -- there it is the evidence.
    m["send_late_ms"] = stat([u["max_late_ms"] for u in ok if u.get("max_late_ms") is not None],
                             "per utterance (worst frame in each)", "ms", True, "MEASURED")

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
        # `errors` / `rejected` keep their old meaning (the KPI window, after warm-up) so
        # that every reader of an old run still reads the same number. The loss gate reads
        # `errors_total`: the whole run, warm-up included.
        "counts": {"utterances": len(counted), "ok": len(ok), "errors": len(errored),
                   "rejected": len(rejected), "warmup_excluded": warm,
                   "errors_warmup": acct["errors_warmup"],
                   "errors_total": acct["errors"],
                   "rejected_warmup": acct["rejected_warmup"],
                   "rejected_total": acct["rejected"],
                   "cut_at_deadline": acct["cut_at_deadline"],
                   "error_kinds": acct["error_kinds"],
                   "error_kinds_warmup": acct["error_kinds_warmup"],
                   "started": acct["started"], "stream_deaths": acct["stream_deaths"],
                   "deltas": sum(u["deltas"] for u in ok), "eous": sum(u["eous"] for u in ok),
                   "audio_s": audio_s, "span_s": span},
        "accounting": acct,
        "streams": streams,
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
    acct = summary.get("accounting") or {}
    if acct.get("conservation") is False:
        invalid.append(f"accounting: {acct.get('conservation_detail')}")
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
    #
    # Over the WHOLE run: the warm-up excludes latencies, never losses.
    line("utterances lost", c.get("errors_total", c["errors"]), 0, "")
    # A stream process that died stopped producing records silently; its absence is a
    # loss the utterance count alone cannot show.
    if c.get("stream_deaths") is not None:
        line("client streams died", c["stream_deaths"], 0, "")
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
    a = summary.get("accounting")
    if a:
        kinds = ", ".join(f"{k} {v}" for k, v in a["error_kinds"].items()) or "none"
        out.append(f"{indent}whole run: {a['started'] if a['started'] is not None else '?'} "
                   f"started = {a['ok']} ok + {a['rejected']} rejected + "
                   f"{a['cut_at_deadline']} cut at deadline + {a['errors']} errors "
                   f"({a['errors_warmup']} in warm-up); errors by kind: {kinds}")
        cons = {True: "holds", False: "FAILS", None: "not checked"}[a["conservation"]]
        out.append(f"{indent}conservation {cons}"
                   + (f": {a['conservation_detail']}" if a.get("conservation_detail") else ""))
    st = summary.get("streams")
    if st:
        out.append(f"{indent}streams: {st['expected'] - len(st['dead'])}/{st['expected']} "
                   f"alive to the end, {len(st['terminated_at_deadline'])} terminated at "
                   f"collection close"
                   + (f"; DIED: {st['dead'][:8]}" if st["dead"] else ""))
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


def M_zero(v):
    return v is not None and abs(v) < 1e-9


def _eq(got, want, what, tol=1e-6):
    """Numbers compare within `tol`; anything else compares exactly. The normaliser and
    the text fields are strings, and a tolerance on a string is meaningless."""
    if isinstance(want, (str, list, dict)) or isinstance(got, (str, list, dict)):
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

    print("number format is forgiven; the words are not")
    bad += not _eq(_cardinal(1966, "en"), "one thousand nine hundred sixty six", "EN cardinal")
    bad += not _eq(_cardinal(1966, "fr"), "mille neuf cent soixante six", "FR cardinal")
    bad += not _eq(_cardinal(80, "fr"), "quatre vingts", "FR 80 takes the s")
    bad += not _eq(_cardinal(81, "fr"), "quatre vingt un", "FR 81 does not")
    bad += not _eq(_cardinal(71, "fr"), "soixante et onze", "FR 71 is soixante et onze")
    bad += not _eq(_cardinal(400, "fr"), "quatre cents", "FR round hundreds take the s")
    bad += not _eq(_year_forms(1966, "en")[0], "nineteen sixty six", "EN reads a year in pairs")
    bad += not _eq(len(_year_forms(1966, "fr")), 0, "French reads years as cardinals")
    REFN = "Since 1966 there were 400 tigers"
    bad += not _eq(wer("since nineteen sixty six there were four hundred tigers", REFN),
                   5.0 / 6.0, "strict WER charges 5 errors on 6 reference words for "
                   "two correctly recognised numbers")
    bad += not _eq(M_zero(wer_format_free("since nineteen sixty six there were four hundred tigers",
                                          REFN, "en")), True,
                   "format-free WER scores it correct")
    bad += not _eq(wer_format_free("since nineteen sixty seven there were four hundred tigers",
                                   REFN, "en") > 0, True,
                   "but a WRONG number is still an error -- only the form is forgiven")
    bad += not _eq(wer_format_free("since 1966 there were 400 tigers", REFN, "en"), 0.0,
                   "a model that emits digits is not punished either")

    print("word-level alignment: one WER, three different failures")
    a = align_counts("il satellite nello spazio", "il satellite nello spazio")
    bad += not _eq(a["sub"] + a["del"] + a["ins"], 0, "an exact transcript has no errors")
    a = align_counts("il satellite", "il satellite nello spazio")
    bad += not _eq(a["del"], 2, "a truncated transcript is 2 DELETIONS")
    bad += not _eq(a["ins"], 0, "and no insertions")
    a = align_counts("il satellite nello spazio ieri sera", "il satellite nello spazio")
    bad += not _eq(a["ins"], 2, "a hallucinated tail is 2 INSERTIONS")
    bad += not _eq(a["del"], 0, "and no deletions")
    a = align_counts("il satellite nella spazio", "il satellite nello spazio")
    bad += not _eq(a["sub"], 1, "one wrong word is 1 SUBSTITUTION")
    a1 = align_counts("il satellite", "il satellite nello spazio")
    a2 = align_counts("il satellite nello spazio ieri sera", "il satellite nello spazio")
    bad += not _eq(wer("il satellite", "il satellite nello spazio"),
                   wer("il satellite nello spazio ieri sera", "il satellite nello spazio"),
                   "deletion and insertion give the SAME wer -- which is why the split exists")
    bad += not _eq(a1["del"] != a2["del"], True, "and the split tells them apart")
    bad += not _eq(align_counts("anything", ""), None, "no reference -> no alignment, not zero errors")
    bad += not _eq(align_counts("", "il satellite")["del"], 2, "an EMPTY transcript is all deletions")

    print("empty-transcript rate is counted, not inferred from a large CER")
    e = partial_quality_summary([partial_quality([], "il satellite"),
                                 partial_quality([{"t1": 1.0, "text": "il satellite"}],
                                                 "il satellite")])
    bad += not _eq(e["empty_transcript"]["rate"], 0.5, "one of two utterances is empty")
    bad += not _eq(e["empty_transcript"]["of"], 2, "over both")

    # the normaliser is shared: punctuation and case cannot decide a first word
    bad += not _eq(token_compatible("Il,", "il"), True, "token compatibility uses the one normaliser")
    bad += not _eq(word_settled("Il"), False, "a bare token is not settled")
    bad += not _eq(word_settled("Il "), True, "a trailing space settles it")
    bad += not _eq(word_settled("Il,"), True, "so does punctuation")

    bad += _self_test_accounting(sends)

    print(f"\nstreaming_metrics self-test: {'FAIL' if bad else 'PASS'} ({int(bad)} failures)")
    return 1 if bad else 0


def _old_ok(rec):
    """The predicate `aggregate()` used for OK until 2026-09-25, kept ONLY so the planted
    faults below can show that the old code accepted them."""
    return not rec.get("error") and not rec.get("rejected")


def _check(cond, what):
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    return 0 if cond else 1


def _self_test_accounting(sends):
    """Planted faults for AUDIT 2026-09-24 gaps 1-3 and 5: each case is a record or a run
    the OLD accounting counted as fine, and the assertion is that it is now an error."""
    bad = 0
    thr = default_thresholds(320.0)

    print("planted fault: server closed without `done` (gap 1)")
    #  the reader broke on a close frame and recorded no error: events, no done_t
    rec_nd = _mk("a.wav", sends, [_ev(10.25, 0.2, "one")], None)
    bad += _check(_old_ok(rec_nd), "the OLD predicate counted it OK")
    u_nd = analyze_utterance(rec_nd, frame_ms=100.0)
    bad += not _eq(u_nd["error_kind"], "no_done", "now classified no_done")
    #  no event at all
    rec_ne = _mk("a.wav", sends, [], None)
    bad += _check(_old_ok(rec_ne), "a record with no event at all: the OLD predicate said OK")
    bad += not _eq(analyze_utterance(rec_ne, frame_ms=100.0)["error_kind"], "no_events",
                   "now classified no_events")
    #  the client now names it where it happens
    rec_sd = _mk("a.wav", sends, [_ev(10.25, 0.2, "one")], None,
                 error_kind="server_disconnect")
    bad += not _eq(analyze_utterance(rec_sd, frame_ms=100.0)["error_kind"],
                   "server_disconnect", "a client-named kind is kept")
    #  in a run: one good utterance and one closed without done
    good = _mk("a.wav", sends, [_ev(10.25, 0.2, "one")], 10.6)
    ag = aggregate([analyze_utterance(good, frame_ms=100.0), u_nd], frame_ms=100.0)
    bad += not _eq(ag["counts"]["ok"], 1, "the run counts one OK, not two")
    bad += not _eq(ag["counts"]["errors_total"], 1, "and one error")
    bad += not _eq(ag["counts"]["error_kinds"].get("no_done"), 1, "of kind no_done")
    ev = envelope_verdict(ag, thr)
    lost = [l for l in ev["lines"] if l["line"] == "utterances lost"][0]
    bad += not _eq(lost["status"], "FAIL", "and the loss line FAILs")

    print("legacy error strings keep a kind, and an unknown one is still an error")
    for msg, want in (("connect: [Errno 61] Connection refused", "connect_error"),
                      ("refused: HTTP/1.1 500 Internal Server Error", "http_error"),
                      ("reader: timed out", "timeout"),
                      ("timeout waiting for done", "timeout"),
                      ("reader: connection closed", "server_disconnect"),
                      ("send: [Errno 32] Broken pipe", "server_disconnect"),
                      ("server error: 500 boom", "server_error"),
                      ("reader: Expecting value: line 1 column 1", "protocol_error"),
                      ("something new", "unclassified")):
        bad += not _eq(outcome({"error": msg, "done_t": None}), want, f"'{msg}'")
    bad += not _eq(outcome({"error": "refused: HTTP/1.1 503", "rejected": True}), "rejected",
                   "a 503 stays a rejection")

    print("planted fault: an error inside the warm-up (gap 2)")
    def at(t0, done=True, **kw):
        snd = [[t0 + 0.1 * i, 0.1 * (i + 1)] for i in range(5)]
        return analyze_utterance(_mk("w.wav", snd, [_ev(t0 + 0.25, 0.2, "x")],
                                     t0 + 0.6 if done else None, t_start=t0, **kw),
                                 frame_ms=100.0)
    run_w = [at(0.0, done=False, error="reader: timed out", error_kind="timeout"),
             at(20.0), at(40.0)]
    agw = aggregate(run_w, frame_ms=100.0, warmup_s=10.0, t0=0.0)
    bad += not _eq(agw["counts"]["errors"], 0,
                   "the KPI-window count is 0 -- all the OLD gate ever read")
    bad += not _eq(agw["counts"]["errors_warmup"], 1, "the warm-up error is counted")
    bad += not _eq(agw["counts"]["errors_total"], 1, "and is in the total")
    bad += not _eq(agw["counts"]["warmup_excluded"], 1, "latencies still skip the warm-up")
    bad += not _eq(agw["metrics"]["ttfp_ms"]["n"], 2, "TTFP has the two steady utterances")
    evw = envelope_verdict(agw, thr)
    bad += not _eq([l["status"] for l in evw["lines"] if l["line"] == "utterances lost"][0],
                   "FAIL", "the loss line reads the total and FAILs")
    bad += not _eq(evw["verdict"], "NOT STREAMABLE", "the run is not streamable")
    rej_w = aggregate([at(0.0, done=False, error="refused: 503", rejected=True), at(20.0),
                       at(40.0)], frame_ms=100.0, warmup_s=10.0, t0=0.0)
    bad += not _eq(rej_w["counts"]["rejected_warmup"], 1, "a warm-up 503 is counted too")
    bad += not _eq(rej_w["counts"]["errors_total"], 0, "and is not a loss")

    print("planted fault: a stream process that died (gap 3), reconciled against its marks")
    #  soak: t0 = 0, deadline 100 s, three streams, tolerance one utterance (15 s)
    recs = []
    for sid in range(3):
        for k in range(4):
            recs.append({"stream": sid, "rep": k, "t_start": 30.0 * k, "done_t": 30.0 * k + 20,
                         "t_end": 30.0 * k + 20, "error": None, "rejected": False})
    #  stream 2 dies during its utterance 2 (started t=60): no record 2, 3, no end mark
    recs = [r for r in recs if not (r["stream"] == 2 and r["rep"] >= 2)]
    starts = [{"stream": r["stream"], "rep": r["rep"], "t": r["t_start"], "clip": "c.wav"}
              for r in recs] + [{"stream": 2, "rep": 2, "t": 60.0, "clip": "c.wav"}]
    status = {0: {"end_t": 110.0, "exitcode": 0, "terminated": False},
              1: {"end_t": 110.0, "exitcode": 0, "terminated": False},
              2: {"end_t": None, "exitcode": -9, "terminated": False}}
    #  the OLD harness saw 10 records, all OK, and nothing asked about stream 2
    bad += _check(all(_old_ok(r) for r in recs), "the OLD accounting saw only OK records")
    synth, rep = reconcile(recs, starts, status, n_streams=3, deadline=100.0, tol_s=15.0)
    bad += not _eq(len(synth), 1, "one start produced no record")
    bad += not _eq(synth[0]["error_kind"], "client_process_death", "it is client_process_death")
    bad += not _eq(rep["dead"], [2], "stream 2 is reported dead")
    allu = [analyze_utterance(dict(r, sends=[[r["t_start"], 1.0]]), frame_ms=100.0)
            for r in recs] + [analyze_utterance(s, frame_ms=100.0) for s in synth]
    ag3 = aggregate(allu, frame_ms=100.0, t0=0.0, started=len(starts), streams=rep)
    bad += not _eq(ag3["counts"]["stream_deaths"], 1, "the summary counts one dead stream")
    bad += not _eq(ag3["counts"]["error_kinds"].get("client_process_death"), 1,
                   "and its in-flight utterance as an error")
    bad += _check(ag3["accounting"]["conservation"] is True,
                  "conservation holds: the death is accounted, not lost")
    ev3 = envelope_verdict(ag3, thr)
    bad += not _eq([l["status"] for l in ev3["lines"] if l["line"] == "client streams died"][0],
                   "FAIL", "the stream-death line FAILs")
    #  a stream that ended cleanly but early: its last sign of life is before the deadline
    status_e = dict(status)
    status_e[2] = {"end_t": 50.0, "exitcode": 0, "terminated": False}
    starts_e = [s for s in starts if not (s["stream"] == 2 and s["rep"] == 2)]
    _, rep_e = reconcile(recs, starts_e, status_e, n_streams=3, deadline=100.0, tol_s=15.0)
    bad += not _eq(rep_e["exited_early"], [2], "a stream that left its loop early is caught")
    #  no status at all (records only): the last-activity rule still finds it
    _, rep_r = reconcile(recs, starts_e, None, n_streams=3, deadline=100.0, tol_s=15.0)
    bad += not _eq(rep_r["dead"], [2], "records alone: last activity 50 s < deadline - 15 s")
    #  a stream that never produced anything at all
    _, rep_n = reconcile(recs, starts_e, None, n_streams=4, deadline=100.0, tol_s=15.0)
    bad += not _eq(rep_n["dead"], [2, 3], "a stream with no record at all is dead")
    #  WAVE: a stream that started fewer than --repeat utterances
    _, rep_w = reconcile(recs, starts_e, status_e, n_streams=3, repeat=4)
    bad += not _eq(rep_w["dead"], [2], "WAVE: 2 of 4 started is a dead stream")

    print("in flight at the deadline is cut_at_deadline: counted, not an error, not dropped")
    status_c = dict(status)
    status_c[2] = {"end_t": None, "exitcode": -15, "terminated": True}
    synth_c, rep_c = reconcile(recs, starts, status_c, n_streams=3, deadline=100.0, tol_s=15.0)
    bad += not _eq(synth_c[0]["error_kind"], "cut_at_deadline", "terminated after collection")
    bad += not _eq(rep_c["dead"], [], "and the stream lived to the deadline")
    allc = [analyze_utterance(dict(r, sends=[[r["t_start"], 1.0]]), frame_ms=100.0)
            for r in recs] + [analyze_utterance(s, frame_ms=100.0) for s in synth_c]
    agc = aggregate(allc, frame_ms=100.0, t0=0.0, started=len(starts), streams=rep_c)
    bad += not _eq(agc["counts"]["cut_at_deadline"], 1, "one cut")
    bad += not _eq(agc["counts"]["errors_total"], 0, "no error")
    bad += _check(agc["accounting"]["conservation"] is True, "conservation holds")
    bad += _check("c.wav" not in group_texts(allc), "a cut utterance never enters text identity")

    print("planted fault: the conservation invariant itself (gap 5)")
    #  one start more than the records: something vanished
    agv = aggregate(allu, frame_ms=100.0, t0=0.0, started=len(starts) + 1, streams=rep)
    bad += _check(agv["accounting"]["conservation"] is False, "started > accounted FAILS")
    evv = envelope_verdict(agv, thr)
    bad += not _eq(evv["verdict"], "INVALID", "and the run is INVALID")
    #  a duplicated record (the same stream/rep twice) is not conservation either
    agd = aggregate(allu + allu[:1], frame_ms=100.0, t0=0.0, started=len(starts), streams=rep)
    bad += _check(agd["accounting"]["conservation"] is False, "a duplicated record FAILS it")
    #  a caller with no start marks is told so, never told it passed
    agn = aggregate(allu, frame_ms=100.0, t0=0.0)
    bad += _check(agn["accounting"]["conservation"] is None, "no start marks -> not checked")
    return bad


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
