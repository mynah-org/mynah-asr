#!/usr/bin/env python3
"""DEV-EOU-FR-v2 and FINAL-EOU-FR-v2: internal-continuation vs true-endpoint banks (model-free, natural audio).

    prepare_eou_v2.py --out /root/ft/eou-v2 --audio /root/disk/eou-v2 --exclude-texts "<globs>" \
        --exclude-speakers "<MLS manifests>" --granary-ids granary_fr_ids.json [--mls-shards 8] [--tedx-shards 8]

Sources: the only long-form French speech no run of ours has trained on, selected on or scored with:
  mls   MLS-fr TRAIN parquet shards (facebook/multilingual_librispeech french/), the LAST --mls-shards of the 34,
        minus every speaker of our MLS manifests (A2-fr train pools, harness DEV = MLS test, FINAL-FR = MLS dev);
        group = speaker; audiobook reading.
  tedx  mTEDx-fr TRAIN shards 2 .. 1 + --tedx-shards (FINAL-FR used shards 0-1, FR-OOD1 the test / valid talks), minus
        test / valid talks and every Granary-fr original_audio_id; group = talk; natural ending . ? ! ... required.
Disjointness guaranteed: speaker (MLS) / talk (mTEDx) disjoint from every training pool, DEV-fr, FR-OOD1 and FINAL-FR,
and between the two new banks (a group goes to DEV or FINAL by sha1(group) parity); normalised text absent from every
--exclude-texts manifest and unique across both banks. NOT guaranteed: that an MLS book (other reader) or a TED speaker
(other talk) never appears elsewhere; that Granary-fr has no re-upload of a TED talk under another video id.
Classes, from the frozen energy rule (80 ms frames, speech = within 35 dB of the clip peak), natural audio, nothing
synthesised:
  internal  the clip's longest internal pause (speech before AND >= 0.5 s of speech after it, starting >= 0.5 s after
            the first speech frame) is >= 0.8 s; stratum by that pause: 0.8-1.04 / 1.04-1.6 / 1.6-2.5 / > 2.5 s;
  endpoint  no internal pause >= 0.56 s (a clean single utterance), natural ending; scored with the frozen +2 s quiet.
Common rules: 1.0 <= duration <= 20 s, no digit / markup, >= 3 words after textnorm fr. Per bank, source and stratum at
most --n-internal clips, endpoint at most --n-endpoint, round-robin over groups in sha1 order (max diversity, at most
--per-group per group and class). Every internal pause >= 0.24 s is recorded per clip (start / end s) for the scorer.
Outputs: $OUT/{dev,final}/eou_<source>.json + eou_all.json, stats.md, SHA256SUMS (manifests + every wav). No model.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import prepare_final_it as P  # noqa: E402
import textnorm  # noqa: E402

FR = 0.08
STRATA = [(0.8, 1.04), (1.04, 1.6), (1.6, 2.5), (2.5, 99.0)]


def pauses(x):
    n = int(FR * 16000)
    f = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    sm = db > db.max() - 35
    on = np.nonzero(sm)[0]
    if not len(on):
        return None
    out, j, last = [], on[0], on[-1]
    while j <= last:
        if not sm[j]:
            k = j
            while not sm[k]:
                k += 1
            out.append((round(j * FR, 2), round(k * FR, 2), round(float(sm[k:].sum()) * FR, 2), round((j - on[0]) * FR, 2)))
            j = k
        else:
            j += 1
    return {"pauses": out, "first_s": round(on[0] * FR, 2), "end_s": round((last + 1) * FR, 2)}


def classify(info):
    """-> (class, stratum, longest qualifying internal pause s) or None"""
    if info is None:
        return None
    q = [p for p in info["pauses"] if p[2] >= 0.5 and p[3] >= 0.5]   # speech after >= 0.5 s, starts >= 0.5 s in
    longest = max((p[1] - p[0] for p in q), default=0.0)
    if longest >= 0.8:
        for i, (lo, hi) in enumerate(STRATA):
            if lo <= longest < hi:
                return "internal", i, round(longest, 2)
    if max((p[1] - p[0] for p in info["pauses"]), default=0.0) < 0.56:
        return "endpoint", None, 0.0
    return None


def analyse(item):
    uid, group, text, b = item
    try:
        x = P.decode(b)
    except Exception:  # noqa: BLE001
        return uid, None, None
    if not (1.0 <= len(x) / 16000 <= 20.0):
        return uid, None, None
    info = pauses(x)
    return uid, classify(info), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--tmp", default="/root/ft/tmp")
    ap.add_argument("--exclude-texts", required=True)
    ap.add_argument("--exclude-speakers", required=True, help="MLS manifests whose speakers are dropped (utt_id prefix)")
    ap.add_argument("--granary-ids", required=True)
    ap.add_argument("--mls-shards", type=int, default=8)
    ap.add_argument("--tedx-shards", type=int, default=8)
    ap.add_argument("--n-internal", type=int, default=50)
    ap.add_argument("--n-endpoint", type=int, default=100)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    a.lang = "fr"
    os.makedirs(a.tmp, exist_ok=True)
    hub = P.Hub()
    nf = lambda t: textnorm.norm(t.replace("’", "'"), "fr")  # noqa: E731
    excl = set()
    for g in a.exclude_texts.split(","):
        for p in glob.glob(g):
            try:
                rows = [json.loads(l) for l in open(p) if l.strip()]
            except json.JSONDecodeError:   # a JSON (not JSONL) file in the glob, e.g. mix/*/hours.json: no texts
                print(f"== skip non-manifest {p}", flush=True)
                continue
            excl |= {nf(r["text"]) for r in rows if isinstance(r, dict) and r.get("text")}
    spk = set()
    for g in a.exclude_speakers.split(","):
        for p in glob.glob(g):
            for l in open(p):
                r = json.loads(l)
                spk.add(str(r.get("speaker") or "").split(":")[-1] or r["utt_id"].split("_")[0])
                spk.add(Path(r["audio_filepath"]).stem.split("_")[0])
    spk.discard("")
    print(f"== exclusions: {len(excl)} texts, {len(spk)} MLS speakers", flush=True)
    gran = set(json.load(open(a.granary_ids)))
    cands = collections.defaultdict(list)   # source -> rows
    drop = collections.Counter()
    # --- MLS-fr train, last shards
    R = "facebook/multilingual_librispeech"
    shards = hub.files(R, "french", "train")
    assert len(shards) == 34, len(shards)
    for i, p in enumerate(shards[-a.mls_shards:]):
        rows = P.parquet_audio(P.fetch(f"{P.HF}/{R}/resolve/main/{p}", f"{a.tmp}/mls_train_{i}.parquet"))
        for r in rows:
            if str(r["speaker_id"]) in spk:
                drop["mls: our speaker"] += 1
                continue
            cands["mls"].append((str(r["id"]), f"mls:{r['speaker_id']}", r["transcript"], r["audio"]["bytes"]))
        print(f"== mls shard {p}: {len(rows)} rows", flush=True)
    # --- mTEDx-fr train shards 2 .. 1 + n
    R = "deepdml/mtedx"
    devp = ["fr/test-00000-of-00001.parquet", "fr/valid-00000-of-00001.parquet"]
    dev_talks = {r["talk_id"] for p in devp for r in hub.column(R, p, ["talk_id"])}
    tr = hub.files(R, "fr", "train")
    assert "/train-00000-of-" in tr[0] and "/train-00001-of-" in tr[1]
    final_talks = {r["talk_id"] for p in tr[:2] for r in hub.column(R, p, ["talk_id"])}
    end_re = re.compile(P.ENDING["fr"])
    for i, p in enumerate(tr[2: 2 + a.tedx_shards]):
        rows = P.parquet_audio(P.fetch(f"{P.HF}/{R}/resolve/main/{p}", f"{a.tmp}/mtedx_train_{i + 2}.parquet"))
        for r in rows:
            t = r["talk_id"]
            if t in dev_talks or t in final_talks:
                drop["tedx: dev / FINAL talk"] += 1
                continue
            if t in gran:
                drop["tedx: talk in Granary-fr"] += 1
                continue
            if not end_re.search(r["transcript"].strip()):
                drop["tedx: no natural ending"] += 1
                continue
            cands["tedx"].append((r["id"], f"tedx:{t}", r["transcript"], r["audio"]["bytes"]))
        print(f"== tedx shard {p}: {len(rows)} rows", flush=True)
    # --- text rules, then the energy classification (decoded PCM only)
    seen = set()
    for src in list(cands):
        keep = []
        for uid, g, text, b in cands[src]:
            n = nf(text)
            if P.BAD.search(text):
                drop[f"{src}: digit/markup"] += 1
            elif len(n.split()) < 3:
                drop[f"{src}: short"] += 1
            elif n in excl:
                drop[f"{src}: text in train/dev/final"] += 1
            elif n in seen:
                drop[f"{src}: dup text"] += 1
            else:
                seen.add(n); keep.append((uid, g, text, b))
        cands[src] = keep
    pool = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for src in cands:
            byid = {c[0]: c for c in cands[src]}
            for uid, cl, info in ex.map(analyse, cands[src], chunksize=64):
                if cl is None:
                    drop[f"{src}: no class / bad audio"] += 1
                    continue
                pool[(src, uid)] = (byid[uid], cl, info)
    print(f"== classified: {collections.Counter((s, c[1][0], c[1][1]) for (s, _), c in pool.items())}", flush=True)
    # --- selection: split by group parity, round-robin over groups in sha1 order
    out = {"dev": [], "final": []}
    for split in out:
        for src in sorted(cands):
            for cls, st in [("internal", i) for i in range(len(STRATA))] + [("endpoint", None)]:
                by = collections.defaultdict(list)
                for (s, uid), (c, (k, si, lp), info) in pool.items():
                    g = c[1]
                    if s != src or k != cls or si != st or (int(P.sha1(g), 16) % 2 == 0) != (split == "dev"):
                        continue
                    by[g].append((P.sha1(f"{src}:{uid}"), uid, c, lp, info))
                for g in by:
                    by[g].sort()
                order = sorted(by, key=lambda g: by[g][0][0])
                cap = a.n_internal if cls == "internal" else a.n_endpoint
                rows, used = [], collections.Counter()
                while len(rows) < cap and any(by[g] and used[g] < a.per_group for g in order):
                    for g in order:
                        if by[g] and used[g] < a.per_group and len(rows) < cap:
                            _, uid, c, lp, info = by[g].pop(0); used[g] += 1
                            x = P.decode(c[3])
                            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", uid)
                            w = Path(a.audio) / split / src / f"{safe}.wav"
                            w.parent.mkdir(parents=True, exist_ok=True)
                            sf.write(w, x, 16000, subtype="PCM_16")
                            rows.append({"audio_filepath": str(w), "duration": round(len(x) / 16000, 3), "text": c[2],
                                         "bank": f"eou-v2-{split}", "source": src, "group": g, "utt_id": uid,
                                         "class": cls, "stratum": None if st is None else
                                         f"{STRATA[st][0]}-{STRATA[st][1] if STRATA[st][1] < 99 else 'inf'}",
                                         "longest_internal_pause_s": lp, "speech_end_s": info["end_s"],
                                         "internal_pauses": [[p[0], p[1]] for p in info["pauses"] if p[1] - p[0] >= 0.24]})
                out[split] += rows
                print(f"== {split} {src} {cls} {'' if st is None else STRATA[st]}: {len(rows)} clips, {len(used)} groups", flush=True)
    for split, rows in out.items():
        d = Path(a.out) / split
        d.mkdir(parents=True, exist_ok=True)
        for src in sorted({r["source"] for r in rows}):
            with open(d / f"eou_{src}.json", "w") as f:
                for r in rows:
                    if r["source"] == src:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(d / "eou_all.json", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        g_dev = {r["group"] for r in out["dev"]}; g_fin = {r["group"] for r in out["final"]}
        assert not (g_dev & g_fin), "group overlap DEV / FINAL"
        st = collections.Counter((r["source"], r["class"], r["stratum"]) for r in rows)
        with open(d / "stats.md", "w") as f:
            f.write(f"# EOU-FR-v2 {split}: {len(rows)} clips, {len({r['group'] for r in rows})} groups, "
                    f"{sum(r['duration'] for r in rows) / 3600:.2f} h\n\n| source | class | stratum | clips |\n|---|---|---|---|\n")
            for k in sorted(st, key=str):
                f.write(f"| {k[0]} | {k[1]} | {k[2]} | {st[k]} |\n")
            f.write(f"\ndropped: {dict(drop)}\n")
        print(open(d / "stats.md").read(), flush=True)


if __name__ == "__main__":
    main()
