"""One target / scorer text normalisation for the stock-SPE EOU path: lower case, de-accented, [a-z' ] only.

    from textnorm import norm, text_ok
    norm("L’œuvre d’art, peut-être.", "fr")  -> "l'oeuvre d'art peut etre"
    norm(text, "it")                          == two_stage.norm(text) on every input (frozen IT numbers)

lang="it" (and any other key of PRE without a map) is EXACTLY the rule copied in two_stage.py, plain_asr.py,
prepare_final_it.py, student_score.py, teacher_ws.py, s2_reset.py, verbalize_it.py ...: NFKD(lower), drop combining
marks, every char outside [a-z' ] -> space, collapse spaces. lang="fr" first maps, after lower():
  - the ligatures NFKD keeps whole: oe / ae (otherwise "cœur" -> "c ur", "Lætitia" -> "l titia");
  - the typographic apostrophes (U+2019 / U+2018 / U+02BC) -> ' (French elisions: "l’homme" -> "l'homme",
    not "l homme"; the IT rule turns them into a word break, which is kept frozen for IT);
then the IT rule. Hyphen policy (both languages, unchanged): '-' is a word break ("peut-être" -> "peut etre",
"c'est-à-dire" -> "c'est a dire"): refs and hyps are normalised alike, and the stock SPE has no hyphen piece.
No other FR-specific change: digits / symbols still become spaces here; the pools drop digit clips (text_ok).
"""
from __future__ import annotations

import re
import unicodedata

_FR = str.maketrans({"œ": "oe", "æ": "ae", "Œ": "oe", "Æ": "ae", "’": "'", "‘": "'", "ʼ": "'"})
PRE = {"it": None, "fr": _FR}   # per-language pre-map applied after lower(); None = the frozen IT rule only
_DROP = re.compile(r"[^a-z' ]")


def prep(t, lang="it"):
    """lower() + the language pre-map (no de-accenting): what the pool filters should look at."""
    m = PRE[lang]   # KeyError on an unknown language: no silent fallback to another language's rule
    t = t.lower()
    return t.translate(m) if m else t


def norm(t, lang="it"):
    t = unicodedata.normalize("NFKD", prep(t, lang))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(_DROP.sub(" ", t).split())


def text_ok(t, lang="it"):
    """prepare_mix.text_ok with the pre-map: no digit, >= 1 letter, every letter a-z after NFKD.
    lang="it" == prepare_mix.text_ok; lang="fr" keeps oe / ae words (the IT rule drops every "cœur" row)."""
    if not t or re.search(r"\d", t):
        return False
    if PRE[lang]:   # IT: the raw text, exactly as prepare_mix.text_ok
        t = prep(t, lang)
    letters = [c for c in unicodedata.normalize("NFKD", t) if c.isalpha()]
    return bool(letters) and all("a" <= c.lower() <= "z" for c in letters)
