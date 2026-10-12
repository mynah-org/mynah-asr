#!/usr/bin/env python3
"""Italian number verbaliser for teacher (pseudo-label) text: digits and number symbols -> words.

    python3 verbalize_it.py --test                                  # self-test (no data, no deps)
    python3 verbalize_it.py --text "Il 15 agosto 1940, alle 10:30, il 3,5 %."
    python3 verbalize_it.py --manifest pool.teacher.json --out pool.teacher.vb.json \
        [--drop-flags phone,longdigits,dotnum,slash,date2y] [--only-digits]

The labels the 120M streaming RNNT learns are lower case a-z + apostrophe + space (accents
stripped by the scorer's norm). The teacher's raw output (`teacher_raw`, teacher_ws.py) is
written text: digits, %, EUR, ordinals (1°, 2ª), times (10:30), decimals (3,5), thousands
(1.000), dates, roman numerals. norm() deletes all of that, so the target silently loses
the words the speaker said. verbalize() rewrites the RAW text into the spoken Italian form
BEFORE any normalisation; its output still has case, accents and punctuation, and the usual
norm() / norm_text() runs on it unchanged:

    teacher_raw --verbalize()--> spoken raw --norm()--> a-z target  (teacher_ws.norm, two_stage.norm)

verbalize_ex() also returns flags for the readings that are guesses (a phone number read
digit by digit, "802.11", "24/7", a two-digit year, an ordinal/degree call...), so a
caller can drop those clips instead of training on a wrong guess.

Manifest mode: every row keeps every field; rows whose `teacher_raw` contains a digit or a
handled symbol get `teacher_raw_vb` (verbalised raw), a new `teacher` = norm(teacher_raw_vb),
`teacher_vb_flags`, and `teacher_wer` recomputed against norm(verbalize(text)) (the
reference is verbalised too, so the agreement score compares like with like). The old
values are kept as `teacher_pre_vb` / `teacher_wer_pre_vb`. --drop-flags removes the rows
carrying any listed flag (counted). Standard library only (num2words is used, if
installed, only by --test as a cross-check).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata

# ---------------------------------------------------------------- cardinals / ordinals
_UNITS = ("zero uno due tre quattro cinque sei sette otto nove dieci undici dodici tredici quattordici "
          "quindici sedici diciassette diciotto diciannove").split()
_TENS = {2: "venti", 3: "trenta", 4: "quaranta", 5: "cinquanta", 6: "sessanta", 7: "settanta",
         8: "ottanta", 9: "novanta"}
_ORD = {1: "primo", 2: "secondo", 3: "terzo", 4: "quarto", 5: "quinto", 6: "sesto", 7: "settimo",
        8: "ottavo", 9: "nono", 10: "decimo"}
_MONTHS = ("gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre "
           "dicembre").split()
_DIGIT_WORD = dict(zip("0123456789", _UNITS[:10]))


def _lt100(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    t, u = divmod(n, 10)
    w = _TENS[t]
    if u == 0:
        return w
    if u in (1, 8):            # venti + uno -> ventuno, venti + otto -> ventotto
        return w[:-1] + _UNITS[u]
    return w + _UNITS[u]


def _lt1000(n: int) -> str:
    h, r = divmod(n, 100)
    if h == 0:
        return _lt100(r)
    hw = "cento" if h == 1 else _UNITS[h] + "cento"
    if r == 0:
        return hw
    rw = _lt100(r)
    if rw.startswith("o"):     # cento + otto/ottanta -> centotto, centottanta
        return hw[:-1] + rw
    return hw + rw


def _trunc_uno(w: str) -> str:
    """ventuno -> ventun (before mila / milioni / a noun); uno alone -> un."""
    return w[:-1] if w.endswith("uno") else w


def _accent(w: str) -> str:
    """Compound ending in -tre takes the accent (ventitré, centotré, duemilatré); 'tre' alone does not."""
    return w[:-3] + "tré" if w.endswith("tre") and w != "tre" else w


def cardinal(n: int) -> str:
    """Spoken Italian cardinal, standard orthography: 21000 -> ventunmila, 2000000 -> due milioni,
    1000001 -> un milione uno (no 'e' joining the groups). > 999 999 999 999: digit by digit."""
    if n < 0:
        return "meno " + cardinal(-n)
    if n >= 10 ** 12:
        return " ".join(_DIGIT_WORD[c] for c in str(n))
    parts = []
    for size, sg, pl in ((10 ** 9, "un miliardo", "miliardi"), (10 ** 6, "un milione", "milioni")):
        q, n = divmod(n, size)
        if q:
            parts.append(sg if q == 1 else _trunc_uno(cardinal(q)) + " " + pl)
    if n or not parts:
        k, r = divmod(n, 1000)
        w = ""
        if k:
            w = "mille" if k == 1 else _trunc_uno(_lt1000(k)) + "mila"
        if r or not k:
            w += _lt1000(r)
        parts.append(_accent(w))
    return " ".join(parts)


def ordinal(n: int, fem: bool = False, plural: bool = False) -> str:
    if n in _ORD:
        w = _ORD[n]
    else:
        c = cardinal(n).replace("tré", "tre")
        if c.endswith("mila"):
            w = c[:-4] + "millesimo"
        elif c.endswith(("tre", "sei")):        # ventitreesimo, ventiseiesimo
            w = c + "esimo"
        else:                                     # undicesimo, ventesimo, ventunesimo, millesimo
            w = c[:-1] + "esimo"
        w = w.replace(" ", "")
    if fem:
        w = w[:-1] + "a"
    if plural:
        w = w[:-1] + ("e" if fem else "i")
    return w


def digits(s: str) -> str:
    return " ".join(_DIGIT_WORD[c] for c in s if c.isdigit())


def decimal(int_part: str, frac: str) -> str:
    """3,5 -> tre virgola cinque; 3,14 -> tre virgola quattordici; 0,05 -> zero virgola zero cinque;
    more than two significant decimals: digit by digit."""
    w = cardinal(int(int_part)) + " virgola"
    z = len(frac) - len(frac.lstrip("0"))
    rest = frac[z:]
    w += " zero" * z
    if rest:
        w += " " + (cardinal(int(rest)) if len(rest) <= 2 else digits(rest))
    return w


# ---------------------------------------------------------------- context helpers
_MASC_A = {"problema", "sistema", "programma", "tema", "schema", "dramma", "clima", "pianeta", "poeta",
           "pilota", "papa", "cinema", "panorama", "diploma", "teorema", "enigma", "dilemma", "atleta",
           "collega", "giornalista", "artista", "ciclista", "regista", "dentista", "sofà", "gorilla"}
_FEM_E = {"notte", "parte", "legge", "chiave", "classe", "frase", "nave", "rete", "sede", "fonte",
          "volte", "mente", "morte", "arte", "gente", "fine", "voce", "luce", "pace", "croce", "neve",
          "fame", "sete", "tesi", "crisi", "analisi", "ipotesi", "foto", "radio", "auto", "moto", "mano",
          "città", "età", "università", "attività", "novità", "società", "unità", "virtù", "tribù"}
_MASC_ION = {"campione", "padiglione", "battaglione", "milione", "camion", "pallone", "bastone",
             "piccione", "leone", "cannone", "sapone", "cartone", "mattone", "maglione", "torrione"}


def _is_fem(word: str) -> bool:
    w = word.lower()
    if w in _MASC_A or w in _MASC_ION:
        return False
    if w in _FEM_E:
        return True
    if w.endswith(("zione", "sione", "gione", "tà", "tù")):
        return True
    return w.endswith("a")


def _one_before(word: str) -> str:
    """'1' before a noun: un / uno / una / un'."""
    w = word.lower()
    vowel = w[:1] in "aeiouàèéìòù"
    if _is_fem(w):
        return "un'" if vowel else "una"
    if re.match(r"(s[^aeiouàèéìòù]|z|gn|ps|pn|x|y)", w):
        return "uno"
    return "un"


_NEXT_WORD = re.compile(r"\s+([A-Za-zÀ-ÿ]+)")


def _next_word(s: str, pos: int):
    m = _NEXT_WORD.match(s, pos)
    return m.group(1) if m else None


def _with_noun(n_word: str, s: str, pos: int) -> str:
    """Agreement of a cardinal with the following noun: 1 -> un/una/un', 21 anni -> ventun anni."""
    nxt = _next_word(s, pos)
    if not nxt or not nxt[0].islower():
        return n_word
    if n_word == "uno":
        return _one_before(nxt)
    if n_word.endswith("uno") and nxt[0] in "aeiouàèéìòù" and not _is_fem(nxt):
        return n_word[:-1]                       # ventun anni, trentun euro
    return n_word


# ---------------------------------------------------------------- tables
_UNIT = {  # exact-case symbol -> (singular, plural); only expanded right after a number
    "%": ("per cento", "per cento"), "‰": ("per mille", "per mille"),
    "€": ("euro", "euro"), "$": ("dollaro", "dollari"), "£": ("sterlina", "sterline"),
    "EUR": ("euro", "euro"), "euro": ("euro", "euro"), "USD": ("dollaro", "dollari"),
    "km/h": ("chilometro orario", "chilometri orari"), "Km/h": ("chilometro orario", "chilometri orari"),
    "km²": ("chilometro quadrato", "chilometri quadrati"), "km2": ("chilometro quadrato", "chilometri quadrati"),
    "kmq": ("chilometro quadrato", "chilometri quadrati"),
    "m²": ("metro quadro", "metri quadri"), "mq": ("metro quadro", "metri quadri"),
    "m³": ("metro cubo", "metri cubi"), "mc": ("metro cubo", "metri cubi"),
    "km": ("chilometro", "chilometri"), "Km": ("chilometro", "chilometri"), "cm": ("centimetro", "centimetri"),
    "mm": ("millimetro", "millimetri"), "m": ("metro", "metri"), "kg": ("chilo", "chili"), "Kg": ("chilo", "chili"),
    "g": ("grammo", "grammi"), "mg": ("milligrammo", "milligrammi"), "t": ("tonnellata", "tonnellate"),
    "l": ("litro", "litri"), "ml": ("millilitro", "millilitri"),
    "h": ("ora", "ore"), "min": ("minuto", "minuti"), "sec": ("secondo", "secondi"),
    "kW": ("kilowatt", "kilowatt"), "kWh": ("kilowattora", "kilowattora"), "MW": ("megawatt", "megawatt"),
    "GW": ("gigawatt", "gigawatt"), "GB": ("giga", "giga"), "MB": ("mega", "mega"), "TB": ("tera", "tera"),
    "GHz": ("gigahertz", "gigahertz"), "MHz": ("megahertz", "megahertz"), "mi": ("miglio", "miglia"),
}
_SCALE = {"mila": "mila", "milione": "milione", "milioni": "milioni", "miliardo": "miliardo",
          "miliardi": "miliardi", "mln": "milioni", "mld": "miliardi", "mrd": "miliardi"}
_CURRENCY_PRE = {"€": "euro", "$": "dollari", "£": "sterline"}
_CURRENCY = {"€", "$", "£", "EUR", "euro", "USD"}
_BIG_SG = {"milioni": "milione", "miliardi": "miliardo"}
_BIG_PL = {"milione": "milioni", "miliardo": "miliardi"}
_ABBR = {"n.": "numero", "nr.": "numero", "art.": "articolo", "artt.": "articoli", "pag.": "pagina",
         "pagg.": "pagine", "cap.": "capitolo", "vol.": "volume", "tel.": "telefono", "ca.": "circa",
         "par.": "paragrafo", "co.": "comma"}
_FRACTION = {(1, 2): "un mezzo", (1, 3): "un terzo", (2, 3): "due terzi", (1, 4): "un quarto",
             (3, 4): "tre quarti", (1, 10): "un decimo"}
_TEMP_CTX = re.compile(r"(temperatur|grad[oi]|caldo|freddo|termometr|minim|massim|meno|sotto zero|febbre|afa)",
                       re.I)
_SCORE_CTX = re.compile(r"(finit|partita|punteggio|vint|pers[oa]|risultato|set\b|pareggi|battut|sconfitt|vittoria)", re.I)
_TIME_CTX = re.compile(r"(?:\b(?:alle|le|ore|dalle|verso le|entro le|fino alle|tra le|fra le|delle)\s+)$", re.I)
_ROMAN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_NOT = {"CD", "DC", "MD", "CV", "CC", "DI", "MI", "LI", "CIV", "DIV", "MIX", "MC", "CL", "DL", "CM",
              "MM", "XL", "LIV", "CI", "C", "D", "L", "M"}
_ROMAN_PRE_STOP = {"il", "la", "lo", "i", "gli", "le", "un", "una", "del", "della", "dello", "dei", "degli",
                   "delle", "nel", "nella", "al", "alla", "e", "ed", "o", "a", "di", "da", "in", "con", "su",
                   "per", "tra", "fra", "che", "è", "non", "ma", "se", "the"}


def _roman_val(s: str) -> int:
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    tot = 0
    for a, b in zip(s, s[1:] + " "):
        tot += -v[a] if b != " " and v[a] < v[b] else v[a]
    return tot


# ---------------------------------------------------------------- the rewrite
_L = "A-Za-zÀ-ÖØ-öø-ÿ"                                  # letters (ª º excluded on purpose)
_SCALE_ALT = "|".join(sorted(_SCALE, key=len, reverse=True))
_UNIT_ALT = "|".join(sorted((re.escape(k) for k in _UNIT), key=len, reverse=True))
_MAIN = re.compile(
    r"(?<![\w.,])(?P<sign>[-−+](?=\d))?"
    r"(?P<i>\d{1,3}(?:\.\d{3})+|\d+)(?:,(?P<f>\d+))?(?!\d)"          # 1.000.000 | 3,5 | 2023
    r"(?:\s?(?P<scale>(?i:" + _SCALE_ALT + r"))\b)?"
    r"(?:\s?(?P<deg>°\s?[CF]\b|°|º|ª)|\s?(?P<unit>" + _UNIT_ALT + r")(?![\w'²³]))?")
_ROMAN_TOK = re.compile(r"\b[IVXLCDM]+\b")
_ROMAN_LIST = re.compile(r"(?:\s*(?:,|e|ed|il|al|l'|all')\s*)+[IVXLCDM]+(?:(?:\s*(?:,|e|ed|il|al|l'|all')\s*)+[IVXLCDM]+)*\s+(?:secol|sec\.)")


def _abbr(m):
    k = m.group(1).lower()
    return ("numero" if k in ("n°", "nº") else _ABBR[k]) + " "


def verbalize_ex(text: str):
    """-> (verbalised text, sorted list of flags). Flags mark readings that are guesses."""
    flags = set()
    s = unicodedata.normalize("NFC", text).replace(" ", " ").replace(" ", " ").replace("’", "'")
    if not re.search(r"\d|[%€$£°ºª&@]|\b[IVXLCDM]+\b", s):
        return s, []

    # abbreviations that only expand before a number: "n. 5", "n° 5", "art. 3"
    s = re.sub(r"(?<![\w])((?:n|nr|art|artt|pag|pagg|cap|vol|tel|ca|par|co)\.|n°|nº)\s?(?=\d)", _abbr, s, flags=re.I)

    s = re.sub(r"\b([ad])\.\s?C\.", lambda m: "avanti Cristo" if m.group(1) == "a" else "dopo Cristo", s)

    # letter/digit glue: G7 -> G 7 (flag), 5km -> 5 km, 10mila -> 10 mila
    if re.search(rf"[{_L}]\d", s):
        flags.add("alnum")
        s = re.sub(rf"(?<=[{_L}])(?=\d)", " ", s)
    s = re.sub(rf"(?<=\d)(?=[{_L}])", " ", s)

    # dates d/m/y, d.m.y, d-m-y -> quindici agosto duemilaventitré
    def date(m):
        d, mo, y = int(m.group(1)), int(m.group(3)), m.group(4)
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        if len(y) == 2:
            flags.add("date2y")
            y = ("20" if int(y) <= 40 else "19") + y
        return f"{'primo' if d == 1 else cardinal(d)} {_MONTHS[mo - 1]} {cardinal(int(y))}"
    s = re.sub(r"(?<![\w.,/])(\d{1,2})([/.-])(\d{1,2})\2(\d{4}|\d{2})(?![\w/]|[.,]\d)", date, s)

    # times: 10:30 always, 10.30 only after alle / le / ore ...; :00 is the bare hour
    def time_(m):
        h, mi = int(m.group(1)), int(m.group(3))
        if h > 24 or mi > 59 or (m.group(2) == "." and not _TIME_CTX.search(s[:m.start()])):
            return m.group(0)
        out = cardinal(h) + ("" if mi == 0 else " e " + cardinal(mi))
        return out + (" e " + cardinal(int(m.group(4))) + " secondi" if m.group(4) else "")
    s = re.sub(r"(?<![\w.,:])(\d{1,2})([:.])(\d{2})(?::(\d{2}))?(?!\d|[.,]\d)", time_, s)

    # phone-like: +39 / 0x / 3xx followed by space-separated groups, >= 8 digits; leading-zero codes
    def phone(m):
        if sum(c.isdigit() for c in m.group(0)) < 8:
            return m.group(0)
        flags.add("phone")
        return digits(m.group(0))
    s = re.sub(r"(?<![\w.,])(?:\+\d{2}\s?)?[03]\d{1,3}(?:[ -]\d{2,4}){1,3}(?![\w,]|\.\d)", phone, s)

    def lead0(m):
        flags.add("phone")
        return digits(m.group(0))
    s = re.sub(r"(?<![\w.,])0\d+(?![\d,.]\d|\d)", lead0, s)

    # sports scores: "finita 2-1" -> due a uno (only with a score word nearby; else the hyphen stays)
    def score(m):
        if not _SCORE_CTX.search(s[max(0, m.start() - 40):m.start()]):
            return m.group(0)
        return f"{cardinal(int(m.group(1)))} a {cardinal(int(m.group(2)))}"
    s = re.sub(r"(?<![\w.,-])(\d{1,2})-(\d{1,2})(?![\w-])", score, s)

    # fractions / slashes: 1/2 -> un mezzo, 24/7 -> ventiquattro su sette (flag)
    def slash(m):
        a, b = int(m.group(1)), int(m.group(2))
        if (a, b) == (1, 2) and (_next_word(m.string, m.end()) or "A")[0].islower():
            return "mezzo"                            # 1/2 litro -> mezzo litro
        if (a, b) in _FRACTION:
            return _FRACTION[(a, b)]
        flags.add("slash")
        return f"{cardinal(a)} su {cardinal(b)}"
    s = re.sub(r"(?<![\w.,/])(\d{1,3})/(\d{1,3})(?![\w/])", slash, s)

    # decades: anni '80 -> anni ottanta
    s = re.sub(r"(?<![\w])'(\d{2})(?!\d)", lambda m: cardinal(int(m.group(1))), s)

    # currency before the number: € 5 -> 5 euro; € 2,5 miliardi -> 2,5 miliardi di euro
    def cur_pre(m):
        sym, num, sc = m.group(1), m.group(2), m.group(3)
        if sc:
            return f"{num} {sc} di {_CURRENCY_PRE[sym]}"
        return f"{num} {_CURRENCY_PRE[sym] if not re.fullmatch(r'0*1', num) else _UNIT[sym][0]}"
    s = re.sub(r"([€$£])\s?(\d[\d.]*(?:,\d+)?)(?:\s?((?i:" + _SCALE_ALT + r"))\b)?", cur_pre, s)

    # x.y that is neither thousands nor a time: 802.11 -> ottocentodue punto undici (flag)
    def dotnum(m):
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", m.group(0)):
            return m.group(0)
        flags.add("dotnum")
        return " punto ".join(digits(x) if x.startswith("0") and len(x) > 1 else cardinal(int(x))
                              for x in m.group(0).split("."))
    s = re.sub(r"(?<![\w.,])\d+(?:\.\d+)+(?!\d)", dotnum, s)

    # bare long digit strings (codes, ids): digit by digit (flag)
    def longd(m):
        flags.add("longdigits")
        return digits(m.group(0))
    s = re.sub(r"(?<![\w.,])\d{8,}(?![\w.,]\d|\d)", longd, s)

    def main(m):
        i, f = m.group("i").replace(".", ""), m.group("f")
        n = int(i)
        sign = {"-": "meno ", "−": "meno ", "+": "più "}.get(m.group("sign") or "", "")
        scale, deg, unit = m.group("scale"), m.group("deg"), m.group("unit")
        if deg:
            d = deg.replace(" ", "")
            if d in ("°C", "°F") or (d == "°" and _TEMP_CTX.search(s[max(0, m.start() - 40):m.start()])):
                if d == "°":
                    flags.add("degree")
                return sign + ("un grado" if n == 1 and not f else (decimal(i, f) if f else cardinal(n)) + " gradi")
            if f or sign:
                return m.group(0)
            flags.add("ordinal")
            return ordinal(n, fem=d == "ª")
        w = decimal(i, f) if f else cardinal(n)
        if scale:
            sc = _SCALE[scale.lower()]
            if sc == "mila":
                w = w + " mila" if f or n >= 1000 else ("mille" if n == 1 else _trunc_uno(cardinal(n)) + "mila")
            elif f:
                w += " " + _BIG_PL.get(sc, sc)
            elif n == 1:
                w = "un " + _BIG_SG.get(sc, sc)
            else:
                w = _trunc_uno(w) + " " + _BIG_PL.get(sc, sc)
        if unit:
            sg, pl = _UNIT[unit]
            if scale and _SCALE[scale.lower()] != "mila" and unit in _CURRENCY:
                return sign + w + " di " + pl
            one = n == 1 and not f and not scale
            if one and sg not in ("per cento", "per mille"):
                w = _one_before(sg.split()[0])
            return sign + w + " " + (sg if one else pl)
        if not f and not scale:
            w = _with_noun(w, m.string, m.end())
        return sign + w
    s = _MAIN.sub(main, s)

    # roman numerals: XXI secolo -> ventunesimo secolo; Giovanni Paolo II -> secondo,
    # Elisabetta II -> seconda; anything else is kept (flag roman_kept)
    def roman(m):
        tok = m.group(0)
        if not _ROMAN.match(tok):
            return tok
        v = _roman_val(tok)
        nxt = re.match(r"\s+(\S+)", s[m.end():])
        prv = re.search(r"(\S+)\s+$", s[:m.start()])
        if (nxt and nxt.group(1).lower().startswith(("secol", "sec."))) or _ROMAN_LIST.match(s, m.end()):
            flags.add("roman")
            return ordinal(v)
        p = prv.group(1) if prv else ""
        if tok not in _ROMAN_NOT and v <= 40 and p[:1].isupper() and p.isalpha() and not p.isupper() \
                and p.lower() not in _ROMAN_PRE_STOP:
            flags.add("roman")
            return ordinal(v, fem=p.lower().endswith("a"))
        if len(tok) >= 2 and tok not in _ROMAN_NOT:
            flags.add("roman_kept")
        return tok
    s = _ROMAN_TOK.sub(roman, s)

    # symbols left over
    s = s.replace("&", " e ").replace("%", " per cento ").replace("@", " chiocciola ")
    if re.search(r"\d", s):
        flags.add("residual")
        s = re.sub(r"\d+", lambda m: cardinal(int(m.group(0))) if len(m.group(0)) < 8 else digits(m.group(0)), s)
    s = re.sub(r"\b([Uu]n') +", r"\1", s)          # un' ora -> un'ora
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s, sorted(flags)


def verbalize(text: str) -> str:
    return verbalize_ex(text)[0]


def norm(t: str) -> str:
    """Scorer / stock-target normalisation (identical to teacher_ws.norm and two_stage.norm)."""
    t = "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def lev(x, y):
    p = list(range(len(y) + 1))
    for i, u in enumerate(x, 1):
        c = [i]
        for j, v in enumerate(y, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (u != v)))
        p = c
    return p[-1]


_HAS_SYM = re.compile(r"\d|[%€$£°ºª&@]|\b[IVXLCDM]{2,}\b")

# ---------------------------------------------------------------- self-test
_CARD = {0: "zero", 1: "uno", 11: "undici", 17: "diciassette", 18: "diciotto", 21: "ventuno", 23: "ventitré",
         28: "ventotto", 33: "trentatré", 81: "ottantuno", 100: "cento", 101: "centouno", 108: "centotto",
         118: "centodiciotto", 180: "centottanta", 200: "duecento", 1000: "mille", 1001: "milleuno",
         1003: "milletré", 1100: "millecento", 1981: "millenovecentottantuno", 2000: "duemila",
         2023: "duemilaventitré", 21000: "ventunmila", 23000: "ventitremila", 100000: "centomila",
         1000000: "un milione", 1000001: "un milione uno", 2000000: "due milioni",
         21000000: "ventun milioni", 2500000: "due milioni cinquecentomila", 10 ** 9: "un miliardo",
         3 * 10 ** 9: "tre miliardi"}
_ORDS = {(1, False): "primo", (3, False): "terzo", (11, False): "undicesimo", (21, False): "ventunesimo",
         (23, False): "ventitreesimo", (26, False): "ventiseiesimo", (100, False): "centesimo",
         (1000, False): "millesimo", (2000, False): "duemillesimo", (2, True): "seconda", (18, True): "diciottesima"}
# (raw, expected after norm()); news / parliament / podcast style
_SENT = [
    ("Il 15 agosto 1940 gli Alleati invasero la Francia del sud.",
     "il quindici agosto millenovecentoquaranta gli alleati invasero la francia del sud"),
    ("La Camera ha approvato il decreto con 312 voti favorevoli e 145 contrari.",
     "la camera ha approvato il decreto con trecentododici voti favorevoli e centoquarantacinque contrari"),
    ("L'inflazione è salita al 3,5% a settembre.", "l'inflazione e salita al tre virgola cinque per cento a settembre"),
    ("Il PIL è cresciuto dello 0,8 per cento nel secondo trimestre.",
     "il pil e cresciuto dello zero virgola otto per cento nel secondo trimestre"),
    ("La manovra vale 24 miliardi di euro.", "la manovra vale ventiquattro miliardi di euro"),
    ("Il governo stanzia € 2,5 miliardi per la sanità.",
     "il governo stanzia due virgola cinque miliardi di euro per la sanita"),
    ("Il biglietto costa 12 € e la tessera 1 €.", "il biglietto costa dodici euro e la tessera un euro"),
    ("La seduta riprende alle 15:30.", "la seduta riprende alle quindici e trenta"),
    ("Ci vediamo domani alle 9.45 in piazza.", "ci vediamo domani alle nove e quarantacinque in piazza"),
    ("Il 1° maggio si celebra la festa dei lavoratori.", "il primo maggio si celebra la festa dei lavoratori"),
    ("È la 2ª volta che succede quest'anno.", "e la seconda volta che succede quest'anno"),
    ("Oltre 800.000 persone hanno firmato la petizione.", "oltre ottocentomila persone hanno firmato la petizione"),
    ("Nel 2023 sono stati registrati 1.250.000 nuovi contratti.",
     "nel duemilaventitre sono stati registrati un milione duecentocinquantamila nuovi contratti"),
    ("Mio nonno ha compiuto 81 anni ieri.", "mio nonno ha compiuto ottantun anni ieri"),
    ("Ha vissuto 21 anni a Milano e 1 anno a Parigi.", "ha vissuto ventun anni a milano e un anno a parigi"),
    ("Ci vuole 1 ora di treno.", "ci vuole un'ora di treno"),
    ("Abbiamo 1 studente straniero in classe.", "abbiamo uno studente straniero in classe"),
    ("La temperatura minima sarà di -5 °C.", "la temperatura minima sara di meno cinque gradi"),
    ("Domani temperature fino a 38° al sud.", "domani temperature fino a trentotto gradi al sud"),
    ("L'articolo 21 della Costituzione tutela la libertà di stampa.",
     "l'articolo ventuno della costituzione tutela la liberta di stampa"),
    ("Il decreto n. 18 del 17/03/2020 è stato convertito.",
     "il decreto numero diciotto del diciassette marzo duemilaventi e stato convertito"),
    ("Papa Giovanni Paolo II visitò la città nel 1984.",
     "papa giovanni paolo secondo visito la citta nel millenovecentottantaquattro"),
    ("Nel XXI secolo la sfida è il clima.", "nel ventunesimo secolo la sfida e il clima"),
    ("Negli anni '80 la lira valeva poco.", "negli anni ottanta la lira valeva poco"),
    ("Il G7 si riunirà a giugno.", "il g sette si riunira a giugno"),
    ("Il treno viaggia a 300 km/h.", "il treno viaggia a trecento chilometri orari"),
    ("Ho comprato 2 kg di mele e 500 g di farina.", "ho comprato due chili di mele e cinquecento grammi di farina"),
    ("L'appartamento misura 85 mq.", "l'appartamento misura ottantacinque metri quadri"),
    ("La disoccupazione giovanile è al 22,3%.", "la disoccupazione giovanile e al ventidue virgola tre per cento"),
    ("Il deficit scende al 4,35 per cento.", "il deficit scende al quattro virgola trentacinque per cento"),
    ("Chiamate il numero 333 123 4567 per informazioni.",
     "chiamate il numero tre tre tre uno due tre quattro cinque sei sette per informazioni"),
    ("Il servizio è attivo 24/7.", "il servizio e attivo ventiquattro su sette"),
    ("La partita è finita 2-1 per la Roma.", "la partita e finita due a uno per la roma"),
    ("Il 35% degli italiani legge almeno 1 libro all'anno.",
     "il trentacinque per cento degli italiani legge almeno un libro all'anno"),
    ("La legge 104 del 1992 tutela i disabili.", "la legge centoquattro del millenovecentonovantadue tutela i disabili"),
    ("Il vertice inizia alle 12:00 GMT.", "il vertice inizia alle dodici gmt"),
    ("Gli autobus passano ogni 7 minuti dalle 6 alle 23.",
     "gli autobus passano ogni sette minuti dalle sei alle ventitre"),
    ("La regina Elisabetta II è morta nel 2022 a 96 anni.",
     "la regina elisabetta seconda e morta nel duemilaventidue a novantasei anni"),
    ("Il Milan ha speso 70 milioni per 3 giocatori.", "il milan ha speso settanta milioni per tre giocatori"),
    ("Ne abbiamo parlato nella puntata 118 del podcast, l'11 settembre 2001.",
     "ne abbiamo parlato nella puntata centodiciotto del podcast l'undici settembre duemilauno"),
    ("Sono arrivati 21.000 messaggi e 1,5 milioni di visualizzazioni.",
     "sono arrivati ventunmila messaggi e uno virgola cinque milioni di visualizzazioni"),
    ("Ha preso 1 milione di voti, l'8% in più.", "ha preso un milione di voti l'otto per cento in piu"),
    ("Le reti 4G e 5G coprono il 99% del territorio.",
     "le reti quattro g e cinque g coprono il novantanove per cento del territorio"),
    ("I casi di COVID-19 sono 1.024.", "i casi di covid diciannove sono milleventiquattro"),
    ("Carlo V e Luigi XIV regnarono a lungo.", "carlo quinto e luigi quattordicesimo regnarono a lungo"),
    ("I ragazzi della Serie C hanno comprato un CD.", "i ragazzi della serie c hanno comprato un cd"),
    ("Mezzo litro: 1/2 litro d'acqua.", "mezzo litro mezzo litro d'acqua"),
    ("Il servizio clienti risponde allo 06 1234 5678.",
     "il servizio clienti risponde allo zero sei uno due tre quattro cinque sei sette otto"),
    ("Lo standard 802.11 è vecchio.", "lo standard ottocentodue punto undici e vecchio"),
    ("Ha vinto 6-4 al terzo set.", "ha vinto sei a quattro al terzo set"),
    ("Il 31/12/24 chiude l'anno.", "il trentuno dicembre duemilaventiquattro chiude l'anno"),
]
_FLAGS = {"Il servizio è attivo 24/7.": {"slash"}, "Il G7 si riunirà a giugno.": {"alnum"},
          "Chiamate il numero 333 123 4567 per informazioni.": {"phone"},
          "Il 31/12/24 chiude l'anno.": {"date2y"}, "Lo standard 802.11 è vecchio.": {"dotnum"},
          "La seduta riprende alle 15:30.": set(), "Domani temperature fino a 38° al sud.": {"degree"}}


def selftest() -> int:
    fails, n = [], 0
    for k, want in _CARD.items():
        n += 1
        if cardinal(k) != want:
            fails.append(f"cardinal({k}) = {cardinal(k)!r} != {want!r}")
    for (k, fem), want in _ORDS.items():
        n += 1
        if ordinal(k, fem) != want:
            fails.append(f"ordinal({k}, fem={fem}) = {ordinal(k, fem)!r} != {want!r}")
    for raw, want in _SENT:
        n += 1
        out, fl = verbalize_ex(raw)
        got = norm(out)
        if got != want:
            fails.append(f"{raw!r}\n      got  {got!r}\n      want {want!r}  (verbalised {out!r})")
        if re.search(r"\d", out):
            fails.append(f"digit left in {out!r}")
        if raw in _FLAGS and set(fl) != _FLAGS[raw]:
            fails.append(f"flags of {raw!r}: {fl} != {sorted(_FLAGS[raw])}")
    n += 1
    if verbalize("niente numeri qui.") != "niente numeri qui.":
        fails.append("text without numbers must be unchanged")
    # informational cross-check against num2words (not installed: skipped)
    try:
        from num2words import num2words
        diff = [(k, cardinal(k), num2words(k, lang="it")) for k in list(range(0, 2101)) + [21000, 31000, 101000,
                118000, 2500000, 21000000, 1000001]
                if norm(cardinal(k)).replace(" ", "") != norm(num2words(k, lang="it")).replace(" ", "")]
        print(f"num2words cross-check: {len(diff)} differences out of 2108 (after norm, spaces ignored); e.g.")
        for d in diff[:8]:
            print(f"   {d[0]}: ours {d[1]!r}  num2words {d[2]!r}")
    except ImportError:
        print("num2words not installed: cross-check skipped")
    for f in fails:
        print("FAIL", f)
    print(f"verbalize_it self-test: {n - len(fails)}/{n} OK ({len(_SENT)} sentences)" if not fails
          else f"verbalize_it self-test: {len(fails)} FAILURES / {n} checks")
    return 1 if fails else 0


# ---------------------------------------------------------------- manifest CLI
def rewrite_manifest(src: str, dst: str, drop_flags: set, only_digits: bool) -> dict:
    st = {"rows": 0, "changed": 0, "dropped": 0, "flags": {}, "wer_pre": [], "wer_post": []}
    with open(src, encoding="utf-8") as fi, open(dst + ".tmp", "w", encoding="utf-8") as fo:
        for line in fi:
            r = json.loads(line)
            st["rows"] += 1
            raw = r.get("teacher_raw") or ""
            if raw and (re.search(r"\d", raw) if only_digits else _HAS_SYM.search(raw)):
                vb, fl = verbalize_ex(raw)
                for f in fl:
                    st["flags"][f] = st["flags"].get(f, 0) + 1
                if drop_flags & set(fl):
                    st["dropped"] += 1
                    continue
                new = norm(vb)
                if new != r.get("teacher"):
                    st["changed"] += 1
                r["teacher_pre_vb"], r["teacher_wer_pre_vb"] = r.get("teacher"), r.get("teacher_wer")
                r["teacher_raw_vb"], r["teacher_vb_flags"], r["teacher"] = vb, fl, new
                ref = norm(verbalize(r.get("text") or ""))
                if ref:
                    r["teacher_wer"] = round(100 * lev(ref.split(), new.split()) / max(1, len(ref.split())), 1)
                    if r["teacher_wer_pre_vb"] is not None:
                        st["wer_pre"].append(r["teacher_wer_pre_vb"]); st["wer_post"].append(r["teacher_wer"])
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    import os
    os.replace(dst + ".tmp", dst)
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="run the self-test")
    ap.add_argument("--text", default="", help="verbalise one string and print it with its flags")
    ap.add_argument("--manifest", default="", help="teacher_ws.py output (needs teacher_raw)")
    ap.add_argument("--out", default="")
    ap.add_argument("--drop-flags", default="phone,longdigits,dotnum,slash,date2y,residual",
                    help="rows whose verbalisation carries any of these flags are dropped ('' keeps all)")
    ap.add_argument("--only-digits", action="store_true",
                    help="rewrite only rows whose teacher_raw has a digit (default: also %%, EUR, roman numerals ...)")
    a = ap.parse_args()
    if a.test:
        return selftest()
    if a.text:
        out, fl = verbalize_ex(a.text)
        print(f"{out}\n  norm : {norm(out)}\n  flags: {fl}")
        return 0
    if not (a.manifest and a.out):
        ap.error("--test, --text or --manifest + --out")
    st = rewrite_manifest(a.manifest, a.out, {f for f in a.drop_flags.split(",") if f}, a.only_digits)
    wp, wq = sorted(st["wer_pre"]), sorted(st["wer_post"])
    med = lambda v: v[len(v) // 2] if v else None  # noqa: E731
    print(f"== {a.manifest}: {st['rows']} rows, {st['changed']} labels changed, {st['dropped']} dropped by flags "
          f"{a.drop_flags or '-'}; flags {json.dumps(st['flags'])}; rewritten rows with a reference: "
          f"teacher_wer median {med(wp)} -> {med(wq)}, mean {sum(wp) / max(1, len(wp)):.1f} -> "
          f"{sum(wq) / max(1, len(wq)):.1f} (n={len(wq)}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
