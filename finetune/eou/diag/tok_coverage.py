#!/usr/bin/env python3
"""Coverage of a (stock) EOU tokenizer on a language's transcripts.

Reports tokens/word, <unk> share, round-trip failures and the characters that
map to <unk>. On 2026-10-09 the stock English SPE of the EOU 120M covered
Italian MLS/FLEURS text except accented vowels (-> <unk>), which is why
plain_asr.py de-accents its targets instead of swapping the tokenizer.

    $PY tok_coverage.py --nemo models/parakeet_realtime_eou_120m-v1.nemo \\
        --manifest manifests/train_5h.json --manifest manifests/eval_fleurs_it.json

Audit mode (plain sentencepiece, no NeMo import, CPU only): compares several SPEs
on several named manifests with the target normalisation each one is trained
with (two_stage.py `norm` for the stock SPE: de-accented a-z + apostrophe;
tokenizer_eou.py `norm_text` for a language SPE: accents kept), and writes a JSON:

    $PY tok_coverage.py --spm stock=stock/tokenizer.model:deaccent \\
        --spm it=tok-it-1024/tokenizer.model:accented \\
        --manifest mls=manifests/eval_mls_it.json --manifest cv=mix/it/eval_cv.json ... \\
        --remap stock,it --frame-ms 80 --json tok_stats.json

  per set x SPE: tokens/word, tokens/char, target length U (p50/p90/p99/max),
  U per second of audio, U/T with T = encoder frames (frame-ms), clips with
  U/T above 0.5/0.8/1.0, <unk> rate, words split into 1/2/3/4+ pieces, words
  starting with a bare "▁" piece, apostrophe and accent behaviour, the
  segmentation of common Italian function words / suffixes, worst utterances.
  Also target-normalisation side effects (`norm_text` vs `norm` on the same raw
  text: word-count changes, glued apostrophes, dropped non-Italian letters) and
  accent-variant inconsistency in the training text (--variants).
  --remap OLD,NEW replays the two_stage.py --tokenizer warm-start source map
  (copied / mean-initialised / <unk> fallback, stock pieces per new piece,
  non-word-initial pieces initialised from a word-initial stock piece) and the
  "stock-path survival": the share of words whose stock segmentation uses only
  pieces that still exist (with their copied rows) in the new vocabulary.
"""
import argparse
import collections
import json
import math
import re
import sys
import unicodedata

WB = "▁"


def norm(t):  # what the stock model was trained on: lowercase, no punctuation except apostrophe
    t = t.lower()
    t = re.sub(r"[^\w' ]", " ", t)
    return " ".join(t.split())


def strip_acc(t):
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def norm_deaccent(t):  # == two_stage.py::norm (stock-SPE targets; refs/hyps are always scored with it)
    t = strip_acc(t.lower())
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


_KEEP = re.compile(r"[^a-zàáèéìíîòóùú' ]")


def norm_accented(s):  # == tokenizer_eou.py::norm_text (copied so this file needs no eou/ import)
    s = unicodedata.normalize("NFC", s.lower()).replace("’", "'").replace("`", "'")
    s = _KEEP.sub(" ", s)
    s = re.sub(r"\s*'\s*", "'", s)
    return re.sub(r"\s+", " ", s).strip()


NORMS = {"deaccent": norm_deaccent, "accented": norm_accented}
FUNC_WORDS = ["il", "lo", "la", "i", "gli", "le", "un", "una", "uno", "di", "del", "dello", "della", "dei", "degli",
              "delle", "nel", "nella", "nei", "al", "alla", "ai", "sul", "sulla", "che", "non", "per", "con", "è",
              "perché", "più", "anche", "questo", "essere"]
APOS_WORDS = ["l'acqua", "dell'unione", "all'interno", "un'altra", "nell'ambito", "c'è", "d'italia", "sull'argomento"]
SUFFIXES = ["zione", "zioni", "mente", "ando", "endo", "ità", "mento", "menti", "issimo", "abile"]


def pct(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(math.ceil(q / 100 * len(xs))) - 1))]


def load_spm(path):
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=path)


def pieces(sp, ids):
    return [sp.id_to_piece(i) for i in ids]


def word_pieces(sp, w):
    return pieces(sp, sp.encode(w))


def frames(dur, frame_ms):  # encoder frames: 10 ms mel hop, then x(frame_ms / 10) subsampling
    return int(dur * 100) // int(round(frame_ms / 10)) + 1


def set_stats(sp, normf, rows, frame_ms, n_worst=5):
    unk = sp.unk_id()
    toks = words = chars = n_unk = rt_fail = 0
    U, Ups, UT = [], [], []
    word_seg = collections.Counter()
    worst, high_ut = [], []
    for r in rows:
        t = normf(r["text"])
        if not t:
            continue
        ids = sp.encode(t)
        rt_fail += sp.decode(ids) != t
        ws = t.split()
        words += len(ws)
        chars += len(t.replace(" ", ""))
        toks += len(ids)
        n_unk += sum(1 for i in ids if i == unk)
        U.append(len(ids))
        d = float(r.get("duration") or 0)
        if d > 0:
            Ups.append(len(ids) / d)
            ut = len(ids) / frames(d, frame_ms)
            UT.append(ut)
            high_ut.append((ut, d, len(ids), t[:100]))
        word_seg.update(ws)
        worst.append((len(ids) / len(ws), t[:100], pieces(sp, ids)[:30]))
    split = collections.Counter()
    bare = apos_w = apos_t = acc_w = acc_t = 0
    for w, c in word_seg.items():
        p = word_pieces(sp, w)
        split[min(len(p), 4)] += c
        bare += c * (bool(p) and p[0] == WB)
        if "'" in w:
            apos_w += c; apos_t += len(p) * c
        if strip_acc(w) != w:
            acc_w += c; acc_t += len(p) * c
    worst.sort(key=lambda x: -x[0])
    high_ut.sort(key=lambda x: -x[0])
    nw = max(1, words)
    return {
        "utts": len(U), "words": words, "chars_nospace": chars, "tokens": toks,
        "tokens_per_word": round(toks / nw, 3), "tokens_per_char": round(toks / max(1, chars), 3),
        "U_p50": pct(U, 50), "U_p90": pct(U, 90), "U_p99": pct(U, 99), "U_max": max(U) if U else 0,
        "U_per_s_p50": round(pct(Ups, 50), 2), "U_per_s_p90": round(pct(Ups, 90), 2),
        "U_per_s_p99": round(pct(Ups, 99), 2),
        "UT_p50": round(pct(UT, 50), 3), "UT_p90": round(pct(UT, 90), 3), "UT_p99": round(pct(UT, 99), 3),
        "UT_max": round(max(UT), 3) if UT else 0,
        "UT_gt_0.5": sum(x > 0.5 for x in UT), "UT_gt_0.8": sum(x > 0.8 for x in UT),
        "UT_gt_1.0": sum(x > 1.0 for x in UT),
        "unk_tokens": n_unk, "unk_rate_pct": round(100 * n_unk / max(1, toks), 3), "roundtrip_fail": rt_fail,
        "words_1_2_3_4plus_pct": [round(100 * split[k] / nw, 1) for k in (1, 2, 3, 4)],
        "words_bare_wb_first_pct": round(100 * bare / nw, 2),
        "apostrophe_words_pct": round(100 * apos_w / nw, 2), "apostrophe_word_tokens": round(apos_t / max(1, apos_w), 2),
        "accented_words_pct": round(100 * acc_w / nw, 2), "accented_word_tokens": round(acc_t / max(1, acc_w), 2),
        "worst_tok_per_word": [{"tpw": round(a, 2), "text": b, "pieces": c} for a, b, c in worst[:n_worst]],
        "highest_UT": [{"UT": round(a, 3), "dur": b, "U": c, "text": d} for a, b, c, d in high_ut[:n_worst]],
    }


def norm_side_effects(rows):
    """norm_text (accented targets) vs norm (stock targets and scoring) on the same raw text."""
    n_wc = glued = typo = 0
    dropped = collections.Counter()
    ex = []
    for r in rows:
        raw = r["text"]
        a, b = norm_accented(raw), norm_deaccent(raw)
        b2 = norm_deaccent(raw.replace("’", "'").replace("`", "'"))  # same apostrophe mapping as norm_text
        typo += any(c in raw for c in "’‘`´")
        if strip_acc(a) != b2:
            n_wc += len(a.split()) != len(b2.split())
            glued += bool(re.search(r"[^\s]'[^\s]", a)) and len(a.split()) < len(b2.split())
            if len(ex) < 6:
                ex.append({"raw": raw[:110], "norm_text": a[:110], "norm": b[:110]})
        for ch in unicodedata.normalize("NFC", raw.lower()):
            if ch.isalpha() and _KEEP.match(ch) and strip_acc(ch) in "abcdefghijklmnopqrstuvwxyz":
                dropped[ch] += 1  # a Latin letter with a diacritic norm_text deletes (word gets split)
    return {"utts": len(rows), "utts_word_count_differs": n_wc, "utts_glued_apostrophe": glued,
            "utts_with_typographic_apostrophe": typo, "letters_dropped_by_norm_text": dict(dropped.most_common(12)),
            "examples": ex}


def accent_variants(rows):
    """Words whose de-accented form is written with more than one accent pattern."""
    forms = collections.defaultdict(collections.Counter)
    for r in rows:
        for w in norm_accented(r["text"]).split():
            forms[strip_acc(w)][w] += 1
    multi = {k: v for k, v in forms.items() if len(v) > 1}
    tot = sum(sum(v.values()) for v in forms.values())
    minority = sum(sum(v.values()) - max(v.values()) for v in multi.values())
    top = sorted(multi.items(), key=lambda kv: -sum(kv[1].values()))[:15]
    return {"word_tokens": tot, "types_with_variants": len(multi),
            "minority_variant_tokens_pct": round(100 * minority / max(1, tot), 3),
            "top": {k: dict(v) for k, v in top}}


def apostrophe_presence(rows):
    w = a = 0
    for r in rows:
        ws = norm_deaccent(r["text"].replace("’", "'")).split()
        w += len(ws)
        a += sum("'" in x for x in ws)
    return round(100 * a / max(1, w), 2)


def remap(old, new):
    """Replays two_stage.py `if a.tokenizer:` (stock source ids per new id), plus diagnostics."""
    old_ids = {old.id_to_piece(i): i for i in range(old.get_piece_size())}
    Vn = new.get_piece_size()
    src, kind = [], []
    for i in range(Vn):
        pc = new.id_to_piece(i)
        if pc in old_ids:
            src.append([old_ids[pc]]); kind.append("exact"); continue
        t = "".join(c for c in unicodedata.normalize("NFKD", pc.replace(WB, " ")) if not unicodedata.combining(c))
        ids = [j for j in old.encode(t.strip()) if j != old.unk_id()]
        if not pc.startswith(WB) and len(ids) > 1 and old.id_to_piece(ids[0]) == WB:
            ids = ids[1:]
        src.append(ids or [old.unk_id()]); kind.append("mean" if ids else "unk")
    nlen = collections.Counter(min(len(s), 5) for s, k in zip(src, kind) if k == "mean")
    # word-internal new piece initialised from a WORD-INITIAL stock piece: encode() adds the dummy
    # "▁" prefix and BPE merges it into the first piece unless it stays a lone "▁"
    wi = [(new.id_to_piece(i), pieces(old, s)) for i, (s, k) in enumerate(zip(src, kind))
          if k == "mean" and not new.id_to_piece(i).startswith(WB) and old.id_to_piece(s[0]).startswith(WB)]
    lone_stripped = sum(1 for i, k in enumerate(kind) if k == "mean" and not new.id_to_piece(i).startswith(WB)
                        and len(old.encode(strip_acc(new.id_to_piece(i)))) > 1
                        and old.id_to_piece(old.encode(strip_acc(new.id_to_piece(i)))[0]) == WB)
    fixed = {}
    old.override_normalizer_spec(add_dummy_prefix=False)
    try:
        for pc, _ in wi:
            fixed[pc] = [p for p in pieces(old, old.encode(strip_acc(pc))) if p != "<unk>"]
    finally:
        old.override_normalizer_spec(add_dummy_prefix=True)
    new_set = {new.id_to_piece(i) for i in range(Vn)}
    lost = [p for p in old_ids if p not in new_set]
    acc = [new.id_to_piece(i) for i in range(Vn) if strip_acc(new.id_to_piece(i)) != new.id_to_piece(i)]
    return {
        "V": Vn, "exact": kind.count("exact"), "mean": kind.count("mean"), "unk_fallback": kind.count("unk"),
        "unk_fallback_pieces": [new.id_to_piece(i) for i, k in enumerate(kind) if k == "unk"],
        "mean_n_sources_hist_1_2_3_4_5plus": [nlen[k] for k in (1, 2, 3, 4, 5)],
        "mean_single_source_pieces": [(new.id_to_piece(i), pieces(old, s)) for i, (s, k) in enumerate(zip(src, kind))
                                      if k == "mean" and len(s) == 1][:40],
        "special_ids_old_new": {p: (old.piece_to_id(p), new.piece_to_id(p)) for p in ("<unk>", "<EOU>", "<EOB>")},
        "word_internal_new_pieces_mean": sum(1 for i, k in enumerate(kind) if k == "mean"
                                             and not new.id_to_piece(i).startswith(WB)),
        "word_internal_lone_wb_stripped": lone_stripped,
        "word_internal_from_word_initial_first": len(wi),
        "word_internal_from_word_initial_examples": [{"new": a, "src": b, "src_no_prefix": fixed.get(a)}
                                                     for a, b in wi[:25]],
        "new_pieces_with_accent": len(acc), "accent_piece_examples": acc[:40],
        "stock_pieces_lost": len(lost), "stock_pieces_lost_examples": lost[:60],
        "_src": src, "_kind": kind,
    }


def survival(old, new_set, rows):
    """Share of words (de-accented, as the stock model emits them) whose stock segmentation uses only
    pieces still present in the new vocabulary (rows copied): what the step-0 decoder can still spell."""
    tot = ok = 0
    cache = {}
    for r in rows:
        for w in norm_deaccent(r["text"]).split():
            if w not in cache:
                cache[w] = all(p in new_set for p in word_pieces(old, w))
            tot += 1
            ok += cache[w]
    return round(100 * ok / max(1, tot), 2)


def audit(a):
    sps = {}
    for s in a.spm:
        name, rest = s.split("=", 1)
        path, nm = rest.rsplit(":", 1)
        sps[name] = (load_spm(path), NORMS[nm], nm)
    sets = {}
    for mp in a.manifest:
        name, path = mp.split("=", 1) if "=" in mp else (mp, mp)
        sets[name] = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        if "teacher_raw" in sets[name][0]:  # the Granary pool rows: tokenizer text = teacher_raw
            sets[name] = [dict(r, text=r.get("teacher_raw") or "") for r in sets[name]]
    out = {"frame_ms": a.frame_ms, "spm": {k: v[2] for k, v in sps.items()}, "sets": {}}
    for sn, rows in sets.items():
        o = out["sets"][sn] = {"apostrophe_words_pct_raw": apostrophe_presence(rows),
                               "norm_side_effects": norm_side_effects(rows)}
        for k, (sp, nf, _) in sps.items():
            st = o[k] = set_stats(sp, nf, rows, a.frame_ms)
            print(f"ROW {sn:8s} {k:6s} tok/w {st['tokens_per_word']:.2f} tok/ch {st['tokens_per_char']:.3f} "
                  f"U p50/90/99 {st['U_p50']}/{st['U_p90']}/{st['U_p99']} U/s p50/90 {st['U_per_s_p50']}/"
                  f"{st['U_per_s_p90']} U/T p50/99/max {st['UT_p50']}/{st['UT_p99']}/{st['UT_max']} "
                  f">0.5:{st['UT_gt_0.5']} unk {st['unk_rate_pct']}% 1/2/3/4+ {st['words_1_2_3_4plus_pct']}",
                  flush=True)
    if a.variants:
        out["accent_variants"] = accent_variants([r for n in a.variants.split(",") for r in sets[n]])
        print("ROW accent variants", {k: v for k, v in out["accent_variants"].items() if k != "top"})
    out["words"], out["suffixes"] = {}, {}
    for k, (sp, nf, nm) in sps.items():
        conv = strip_acc if nm == "deaccent" else (lambda w: w)
        out["words"][k] = {w: word_pieces(sp, conv(w)) for w in FUNC_WORDS + APOS_WORDS}
        suf = {}
        for sfx in SUFFIXES:  # words ending in the suffix: share whose LAST piece is exactly the suffix
            s2 = conv(sfx)
            ws = collections.Counter(w for rows in sets.values() for r in rows for w in nf(r["text"]).split()
                                     if w.endswith(s2) and len(w) > len(s2) + 1)
            n = sum(ws.values())
            if n:
                seg = {w: word_pieces(sp, w) for w in ws}
                suf[sfx] = {"words": n,
                            "suffix_is_last_piece_pct": round(100 * sum(c for w, c in ws.items()
                                                                        if seg[w][-1].lstrip(WB) == s2) / n, 1),
                            "tokens_per_word": round(sum(len(seg[w]) * c for w, c in ws.items()) / n, 2),
                            "examples": [(w, seg[w]) for w, _ in ws.most_common(3)]}
        out["suffixes"][k] = suf
    if a.remap:
        o, nn = a.remap.split(",")
        old, new = sps[o][0], sps[nn][0]
        rm = remap(old, new)
        new_set = {new.id_to_piece(i) for i in range(new.get_piece_size())}
        rm["stock_path_word_survival_pct"] = {sn: survival(old, new_set, rows) for sn, rows in sets.items()}
        src, kind = rm.pop("_src"), rm.pop("_kind")
        if a.remap_dump:
            with open(a.remap_dump, "w", encoding="utf-8") as f:
                for i, (s, k) in enumerate(zip(src, kind)):
                    f.write(f"{i}\t{new.id_to_piece(i)}\t{k}\t{' '.join(pieces(old, s))}\n")
        out["remap"] = rm
        print("ROW remap", {k: v for k, v in rm.items() if not k.endswith(("examples", "pieces"))})
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", help="NeMo mode: the .nemo whose tokenizer to check")
    ap.add_argument("--manifest", action="append", required=True, help="path (or name=path in --spm mode)")
    ap.add_argument("--spm", action="append", default=[], help="audit mode: name=tokenizer.model:deaccent|accented")
    ap.add_argument("--remap", default="", help="OLD,NEW (--spm names): replay the two_stage.py warm-start map")
    ap.add_argument("--remap-dump", default="", help="TSV: new id, piece, exact|mean|unk, stock source pieces")
    ap.add_argument("--variants", default="", help="comma list of set names for the accent-variant check")
    ap.add_argument("--frame-ms", type=float, default=80.0, help="encoder frame (10 ms hop x subsampling)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.spm:
        return audit(a)
    assert a.nemo, "--nemo or --spm"
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location="cpu")
    tk = m.tokenizer
    sp = tk.tokenizer
    print("ROW vocab", tk.vocab_size, "unk id", sp.unk_id(), "pieces sample", [sp.id_to_piece(i) for i in range(0, 40, 4)])
    rows = []
    for mp in a.manifest:
        rows += [json.loads(line) for line in open(mp, encoding="utf-8") if line.strip()]
    unk = toks = words = rt_fail = 0
    bad = collections.Counter()
    for r in rows:
        t = norm(r["text"])
        ids = tk.text_to_ids(t)
        unk += sum(1 for i in ids if i == sp.unk_id())
        toks += len(ids)
        words += len(t.split())
        if tk.ids_to_text(ids) != t:
            rt_fail += 1
            for ch in t:
                if not tk.text_to_ids(ch) or sp.unk_id() in tk.text_to_ids(ch):
                    bad[ch] += 1
    print("ROW utts", len(rows), "words", words, "tokens/word %.2f" % (toks / max(1, words)), "unk tokens", unk,
          "(%.3f%% of tokens)" % (100 * unk / max(1, toks)), "round-trip failures", rt_fail)
    print("ROW chars that map to UNK:", bad.most_common(15))
    for r in rows[:4] + rows[-3:]:
        t = norm(r["text"])
        ids = tk.text_to_ids(t)
        print("ROW ex", repr(t[:70]), "->", [sp.id_to_piece(i) for i in ids[:14]], "| back:", repr(tk.ids_to_text(ids)[:70]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
