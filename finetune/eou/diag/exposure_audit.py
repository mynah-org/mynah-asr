#!/usr/bin/env python3
"""Effective-exposure audit of --mix runs of two_stage.py (pure python, no torch, no GPU).

    python3 exposure_audit.py --metrics-dir DIR --runs a0=plain-it-a0-mix,c1=plain-it-c1-yg15 \
        --path-map /root/ft/=/root/ft/s14/ft/ --granary all=yg_all.json,agree15=...,rand15=... \
        --json out.json

For every run, the trainer's args (mix, steps, bs, max_dur, seed) are read from
<run>.metrics.json and the --mix sampler of two_stage.py is REPLAYED bit for bit
(random.seed(seed); per item random.choices over source weights, then the next row of
that source's own random.shuffle order, reshuffled when exhausted; rows filtered as
the trainer does: duration <= max_dur and non-empty normalised text). The replay is
exact only if nothing else consumed python `random` between the seed and the first
step; it is checked against the logged audio_h_seen (duration field vs decoded samples).

Per run and source: draws, drawn hours, unique clips / hours seen, passes (draws / rows),
mean revisits per seen clip, speakers (or Granary videos) seen and speakers per seen hour,
duration p10/p50/p90, cumulative drawn hours per source at each eval step (to compare
val curves at equal per-source exposure), audio seconds per optimizer step and padding
share (bs x longest clip vs sum).
Text: type/token counts per source, val OOV (tokens and types) per domain against each
run's training text (all rows and rows actually seen), exact duplicate transcripts within
and across sources, val-train exact transcript overlap and speaker overlap, apostrophe /
glued-elision conventions. Granary: clips per video, top-video share, Gini, and what the
two-teacher agreement gate selects compared with a random hours-matched subset.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import unicodedata
from pathlib import Path


def norm(t):   # = two_stage.norm
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def pct(v, qs=(0.1, 0.5, 0.9)):
    v = sorted(v)
    return [round(v[min(len(v) - 1, int(q * len(v)))], 2) for q in qs] if v else None


def gini(xs):
    xs = sorted(xs); n = len(xs); s = sum(xs)
    if not n or not s:
        return None
    return round(sum((2 * i - n + 1) * x for i, x in enumerate(xs)) / (n * s), 3)


def load(p):
    return [json.loads(l) for l in open(p)]


SPK_RE = re.compile(r"^(\d+)_\d+_\d+$")


def speaker(r):
    s = r.get("speaker")
    if s:
        return str(s)
    m = SPK_RE.match(str(r.get("utt_id", "")))   # MLS: <speaker>_<book>_<idx>
    return f"mls:{m.group(1)}" if m else None


FILLERS = {"eh", "ehm", "ehh", "mh", "mmh", "cioe", "diciamo", "insomma", "allora", "praticamente", "ok", "okay", "boh", "beh"}


class Run:
    def __init__(self, tag, metrics, pmap):
        self.tag, self.m = tag, metrics
        a = metrics["args"]
        self.steps, self.bs, self.max_dur, self.seed = a["steps"], a["bs"], a["max_dur"], a.get("seed", 0)
        self.eval_every = a.get("eval_every", 500)
        self.mix = []
        for x in a["mix"].split(","):
            p, w = x.rsplit(":", 1)
            for k, v in pmap:
                if p.startswith(k):
                    p = v + p[len(k):]
            self.mix.append((Path(p).stem, p, float(w)))


def replay(srcs, steps, bs, seed, eval_every):
    """srcs: [(name, rows, w)] -> per-source draw lists (row indices), per-step (sum, max) durations, cumulative h."""
    random.seed(seed)
    orders = {}
    draws = [[] for _ in srcs]
    per_step, cum = [], []
    run_h = [0.0] * len(srcs)
    ws = [w for _, _, w in srcs]
    for step in range(1, steps + 1):
        ds = []
        for _ in range(bs):
            si = random.choices(range(len(srcs)), weights=ws)[0]
            o = orders.get(si)
            if not o:
                o = orders[si] = list(range(len(srcs[si][1]))); random.shuffle(o)
            i = o.pop(); draws[si].append(i)
            d = srcs[si][1][i]["duration"]; ds.append(d); run_h[si] += d / 3600
        per_step.append((sum(ds), max(ds)))
        if step % eval_every == 0 or step == steps:
            cum.append({"step": step, **{srcs[k][0]: round(run_h[k], 2) for k in range(len(srcs))}})
    return draws, per_step, cum


def text_stats(texts):
    tok = collections.Counter(w for t in texts for w in t.split())
    return tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--runs", required=True, help="tag=metrics_stem,...")
    ap.add_argument("--path-map", default="", help="old_prefix=new_prefix,... applied to mix and val paths")
    ap.add_argument("--vals", default="", help="name=path,... (default: the run's --vals)")
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--granary", default="", help="name=path,... Granary subsets to profile")
    ap.add_argument("--mc-seeds", type=int, default=3, help="extra seeds to show the replay spread")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    pmap = [tuple(x.split("=", 1)) for x in a.path_map.split(",") if x]

    def mp(p):
        for k, v in pmap:
            if p.startswith(k):
                return v + p[len(k):]
        return p

    runs = []
    for x in a.runs.split(","):
        tag, stem = x.split("=", 1)
        runs.append(Run(tag, json.load(open(f"{a.metrics_dir}/{stem}.metrics.json")), pmap))
    out = {"runs": {}, "sources": {}, "vals": {}, "lexical": {}, "dups": {}, "granary": {}, "notes": []}

    # ---------- load sources (cache per path+max_dur, trainer filter)
    raw = {}
    for r in runs:
        for name, p, _ in r.mix:
            if p not in raw:
                raw[p] = load(p)
                for row in raw[p]:
                    row["_n"] = norm(row["text"])
    cache = {}

    def rows_for(p, max_dur):
        k = (p, max_dur)
        if k not in cache:
            cache[k] = [row for row in raw[p] if row.get("duration", 0) <= max_dur and row["_n"]]
        return cache[k]

    # ---------- per source description (full manifest, both filters)
    for p, rs in raw.items():
        name = Path(p).stem
        spk = collections.Counter(speaker(r) for r in rs if speaker(r))
        dur = [r["duration"] for r in rs]
        words = [len(r["_n"].split()) for r in rs]
        txt = collections.Counter(r["_n"] for r in rs)
        tok = text_stats(r["_n"] for r in rs)
        apos = sum(c for w, c in tok.items() if "'" in w) / max(1, sum(tok.values()))
        glued = sum(c for w, c in tok.items() if re.match(r"^(dell|dall|nell|sull|all)[aeiou]", w) and "'" not in w and w not in VOCAB_IT_SAFE)
        out["sources"][name] = {
            "path": p, "rows": len(rs), "hours": round(sum(dur) / 3600, 2),
            "rows_le20": sum(1 for d in dur if d <= 20), "rows_le30": sum(1 for d in dur if d <= 30),
            "rows_empty_norm": sum(1 for r in rs if not r["_n"]),
            "speakers": len(spk), "speakers_per_h": round(len(spk) / max(1e-9, sum(dur) / 3600), 2),
            "clips_per_speaker_p50_p90_max": [pct(list(spk.values()), (0.5,))[0], pct(list(spk.values()), (0.9,))[0], max(spk.values()) if spk else None],
            "dur_s_p10_p50_p90": pct(dur), "mean_dur_s": round(sum(dur) / len(dur), 2),
            "words_per_s_p10_p50_p90": pct([w / d for w, d in zip(words, dur) if d > 0]),
            "tokens": sum(tok.values()), "types": len(tok), "type_token_ratio": round(len(tok) / max(1, sum(tok.values())), 4),
            "dup_text_rows": sum(c for c in txt.values() if c > 1), "dup_text_rows_extra": sum(c - 1 for c in txt.values() if c > 1),
            "distinct_texts": len(txt), "max_text_repeat": max(txt.values()),
            "apostrophe_token_share": round(apos, 4),
            "glued_elision_token_share_heuristic": round(glued / max(1, sum(tok.values())), 4),
        }
        print(f"-- source {name}: {out['sources'][name]}", flush=True)

    # cross-source exact transcript overlap
    srcnames = list(raw)
    texts = {Path(p).stem: set(r["_n"] for r in raw[p]) for p in srcnames}
    out["dups"]["cross_source_shared_texts"] = {
        f"{x}|{y}": len(texts[x] & texts[y]) for i, x in enumerate(texts) for y in list(texts)[i + 1:] if texts[x] & texts[y]}

    # ---------- val sets (as the trainer subsamples them)
    vals = a.vals or runs[0].m["args"]["vals"]
    V = {}
    for x in vals.split(","):
        n, p = x.split("=", 1); p = mp(p)
        full = load(p)
        sub = full[:: max(1, len(full) // a.val_n)][: a.val_n]
        for r in full:
            r["_n"] = norm(r["text"])
        V[n] = (full, sub)
        vt = text_stats(r["_n"] for r in sub)
        vs = collections.Counter(speaker(r) for r in sub if speaker(r))
        out["vals"][n] = {"path": p, "full_rows": len(full), "scored_rows": len(sub), "hours_scored": round(sum(r["duration"] for r in sub) / 3600, 2),
                          "dur_s_p10_p50_p90": pct([r["duration"] for r in sub]), "tokens": sum(vt.values()), "types": len(vt),
                          "speakers": len(vs),
                          "apostrophe_token_share": round(sum(c for w, c in vt.items() if "'" in w) / max(1, sum(vt.values())), 4),
                          "words_per_s_p50": pct([len(r["_n"].split()) / r["duration"] for r in sub if r["duration"] > 0], (0.5,))[0]}
        # leakage: exact transcript / speaker overlap with every training source
        lk = {}
        for p2, rs2 in raw.items():
            s2 = Path(p2).stem
            t2 = texts[s2]; sp2 = set(speaker(r) for r in rs2 if speaker(r))
            lk[s2] = {"scored_text_in_train": sum(1 for r in sub if r["_n"] in t2),
                      "full_text_in_train": sum(1 for r in full if r["_n"] in t2),
                      "scored_speakers_in_train": sum(1 for s in vs if s in sp2)}
        out["vals"][n]["overlap_with_train"] = lk
        print(f"-- val {n}: {out['vals'][n]}", flush=True)

    # ---------- replay every run
    for r in runs:
        srcs = [(name, rows_for(p, r.max_dur), w) for name, p, w in r.mix]
        draws, per_step, cum = replay(srcs, r.steps, r.bs, r.seed, r.eval_every)
        tot_w = sum(w for _, _, w in srcs)
        sim_h = sum(s for s, _ in per_step) / 3600
        res = {"args": {k: r.m["args"][k] for k in ("mix", "steps", "bs", "max_dur", "seed", "unfreeze_sched", "freeze_enc", "lr", "enc_lr")},
               "audio_h_seen_logged": r.m.get("audio_h_seen"), "audio_h_replayed": round(sim_h, 3),
               "replay_matches_log": abs(sim_h - (r.m.get("audio_h_seen") or 0)) < 0.5,
               "wall_s": r.m.get("wall_s"),
               "audio_s_per_step_p10_p50_p90": pct([s for s, _ in per_step]),
               "audio_s_per_step_mean": round(sum(s for s, _ in per_step) / len(per_step), 1),
               "padding_share": round(1 - sum(s for s, _ in per_step) / sum(r.bs * m for _, m in per_step), 3),
               "cum_drawn_h": cum, "sources": {},
               "val": [{k: v for k, v in e.items() if k == "step" or (k.startswith("val_") and k.endswith("_wer"))} for e in r.m["log"]]}
        for k, (name, rows, w) in enumerate(srcs):
            dr = draws[k]; uniq = set(dr)
            sp = set(speaker(rows[i]) for i in uniq if speaker(rows[i]))
            uh = sum(rows[i]["duration"] for i in uniq) / 3600
            res["sources"][name] = {
                "weight_share": round(w / tot_w, 4), "rows": len(rows), "hours": round(sum(x["duration"] for x in rows) / 3600, 2),
                "draws": len(dr), "drawn_h": round(sum(rows[i]["duration"] for i in dr) / 3600, 2),
                "unique_clips_seen": len(uniq), "unique_h_seen": round(uh, 2),
                "unique_share": round(len(uniq) / max(1, len(rows)), 3),
                "passes": round(len(dr) / max(1, len(rows)), 3),
                "mean_visits_per_seen_clip": round(len(dr) / max(1, len(uniq)), 3),
                "speakers_seen": len(sp), "speakers_per_unique_h": round(len(sp) / max(1e-9, uh), 2),
                "drawn_dur_s_p10_p50_p90": pct([rows[i]["duration"] for i in dr]),
            }
        # spread over other seeds (the logged run used r.seed)
        spread = collections.defaultdict(list)
        for sd in range(1, a.mc_seeds + 1):
            d2, ps2, _ = replay(srcs, r.steps, r.bs, sd, r.eval_every)
            spread["audio_h"].append(round(sum(s for s, _ in ps2) / 3600, 2))
            for k, (name, rows, w) in enumerate(srcs):
                spread[name + ".passes"].append(round(len(d2[k]) / len(rows), 3))
        res["other_seed_spread"] = dict(spread)
        # lexical: run training text (all rows / seen rows) vs val vocab
        tr_all = collections.Counter(); tr_seen = collections.Counter()
        for k, (name, rows, w) in enumerate(srcs):
            for row in rows:
                for wd in row["_n"].split():
                    tr_all[wd] += 1
            for i in set(draws[k]):
                for wd in rows[i]["_n"].split():
                    tr_seen[wd] += 1
        lex = {"train_types_all": len(tr_all), "train_tokens_all": sum(tr_all.values()),
               "train_types_seen": len(tr_seen), "train_tokens_seen": sum(tr_seen.values()), "per_val": {}}
        for n, (full, sub) in V.items():
            vt = text_stats(r_["_n"] for r_ in sub)
            ntok = sum(vt.values())
            lex["per_val"][n] = {
                "oov_token_rate_all": round(sum(c for w_, c in vt.items() if w_ not in tr_all) / ntok * 100, 2),
                "oov_type_rate_all": round(sum(1 for w_ in vt if w_ not in tr_all) / len(vt) * 100, 2),
                "oov_token_rate_seen": round(sum(c for w_, c in vt.items() if w_ not in tr_seen) / ntok * 100, 2),
                "rare_le2_token_rate_seen": round(sum(c for w_, c in vt.items() if tr_seen.get(w_, 0) <= 2) / ntok * 100, 2)}
        res["lexical"] = lex
        out["runs"][r.tag] = res
        print(f"== run {r.tag}: h replay {sim_h:.2f} vs log {r.m.get('audio_h_seen')}", flush=True)
        for name, s in res["sources"].items():
            print(f"   {name:20s} {s}", flush=True)
        print(f"   lexical {lex}", flush=True)

    # ---------- Granary video concentration and agreement-selection effects
    for x in [x for x in a.granary.split(",") if x]:
        n, p = x.split("=", 1); rs = load(p)
        vids = collections.Counter(speaker(r) for r in rs)
        vh = collections.Counter()
        for r in rs:
            vh[speaker(r)] += r["duration"]
        tot_h = sum(vh.values()) / 3600
        top = sorted(vh.values(), reverse=True)
        nt = [norm(r["text"]) for r in rs]
        words = [len(t.split()) for t in nt]
        fill = sum(1 for t in nt for w in t.split() if w in FILLERS) / max(1, sum(words))
        refdig = sum(1 for r in rs if re.search(r"\d", r.get("ref_granary", "") or "")) / len(rs)
        refpunct = sum(len(re.findall(r"[,.;:?!]", r.get("ref_granary", "") or "")) for r in rs) / max(1, sum(words))
        out["granary"][n] = {
            "path": p, "clips": len(rs), "hours": round(tot_h, 2), "videos": len(vids),
            "clips_per_video_p50_p90_max": [pct(list(vids.values()), (0.5,))[0], pct(list(vids.values()), (0.9,))[0], max(vids.values())],
            "top1_video_hour_share": round(top[0] / 3600 / tot_h, 4), "top10_video_hour_share": round(sum(top[:10]) / 3600 / tot_h, 4),
            "top100_video_hour_share": round(sum(top[:100]) / 3600 / tot_h, 4),
            "gini_hours_per_video": gini(list(vh.values())), "videos_per_h": round(len(vids) / tot_h, 2),
            "dur_s_p10_p50_p90": pct([r["duration"] for r in rs]), "mean_dur_s": round(sum(r["duration"] for r in rs) / len(rs), 2),
            "share_dur_gt20": round(sum(1 for r in rs if r["duration"] > 20) / len(rs), 3),
            "words_per_s_p10_p50_p90": pct([w / r["duration"] for w, r in zip(words, rs) if r["duration"] > 0]),
            "chars_per_word_p50": pct([len(t.replace(" ", "")) / max(1, len(t.split())) for t in nt if t], (0.5,))[0],
            "teacher_wer_p10_p50_p90": pct([float(r["teacher_wer"]) for r in rs if r.get("teacher_wer") is not None]),
            "filler_token_rate": round(fill, 4), "ref_with_digits_share": round(refdig, 3), "ref_punct_per_word": round(refpunct, 3),
            "dup_text_rows_extra": sum(c - 1 for c in collections.Counter(nt).values() if c > 1),
            "lang": dict(collections.Counter(r.get("teacher_lang", "?") for r in rs).most_common(4)),
        }
        print(f"-- granary {n}: {out['granary'][n]}", flush=True)
    json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


# frequent real Italian words that start like an elision but are not glued forms
VOCAB_IT_SAFE = {"alla", "alle", "allo", "agli", "dalla", "dalle", "dallo", "della", "delle", "dello", "nella", "nelle", "nello",
                 "sulla", "sulle", "sullo", "una", "uno", "unione", "uniti", "unico", "unica", "universita", "uomo", "la", "le", "lo",
                 "li", "di", "da", "de", "ci", "ce", "co", "lei", "loro", "lui", "lavoro", "legge", "questa", "questo", "queste",
                 "questi", "quella", "quello", "quelle", "quelli", "allora", "altro", "altra", "altri", "altre", "anche", "dei",
                 "del", "dal", "nel", "sul", "al", "un", "unita", "ultimo", "ultima", "ue", "uso", "uomini", "ultimi", "ue"}

if __name__ == "__main__":
    main()
