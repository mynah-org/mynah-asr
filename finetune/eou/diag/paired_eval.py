#!/usr/bin/env python3
"""Per-utterance, paired comparison of students and a teacher on the val clips of two_stage.py.

    paired_eval.py subsample --vals mls=a.json,cv=b.json [--val-n 200] --out DIR
        -> DIR/sub_<set>.json: EXACTLY the clips two_stage.py scores (rs[::len//n][:n])
    paired_eval.py decode --nemo run/final.nemo --name A2 --out DIR        (GPU, NeMo greedy)
        -> DIR/hyp_<name>_<set>.json  {audio_filepath: hypothesis}
    (teacher: teacher_ws.py --manifest DIR/sub_<set>.json --out DIR/teacher_<set>.json)
    paired_eval.py report --out DIR --models P40,A0,A2 [--teacher Nemotron] [--pair A2]
        -> per set and macro: WER / CER / empty of every model and the teacher, the gap
           <pair> - teacher, and the paired per-utterance win / tie / loss of <pair> vs the
           teacher overall and by duration and level (RMS dBFS) tercile.

Scores: accent-insensitive a-z normalisation and pooled WER/CER, as two_stage.py; the
student numbers must reproduce the run's metrics.json (a check of the harness).
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


def norm(t):
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


def subsample(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for spec in a.vals.split(","):
        n, p = spec.split("=", 1)
        rs = [json.loads(l) for l in open(p)]
        rs = rs[:: max(1, len(rs) // a.val_n)][: a.val_n]
        with open(out / f"sub_{n}.json", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"   sub_{n}: {len(rs)} clips")


def decode(a):
    import numpy as np
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import open_dict
    m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location="cuda").eval()
    dc = m.cfg.decoding
    with open_dict(dc):
        dc.strategy = "greedy"; dc.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(dc)
    for sub in sorted(Path(a.out).glob("sub_*.json")):
        rs = [json.loads(l) for l in open(sub)]
        hyps = {}
        for i in range(0, len(rs), 16):
            xs = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in rs[i:i + 16]]
            X = torch.zeros(len(xs), max(len(x) for x in xs))
            for j, x in enumerate(xs):
                X[j, :len(x)] = torch.from_numpy(np.asarray(x))
            with torch.no_grad():
                f, fl = m.preprocessor(input_signal=X.cuda(), length=torch.tensor([len(x) for x in xs]).cuda())
                e, el = m.encoder(audio_signal=f, length=fl)
                hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el, return_hypotheses=True)
            hy = hy[0] if isinstance(hy, tuple) else hy
            for r, h in zip(rs[i:i + 16], hy):
                h = h[0] if isinstance(h, list) else h
                ys = [t for t in (h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)) if t < 1024]
                hyps[r["audio_filepath"]] = m.tokenizer.ids_to_text(ys)
        name = sub.stem[4:]
        json.dump(hyps, open(Path(a.out) / f"hyp_{a.name}_{name}.json", "w"), ensure_ascii=False)
        print(f"   {a.name} {name}: {len(hyps)} hyps")


def scores(rows, hyp):
    we = wn = ce = cn = emp = 0
    for r in rows:
        ref, h = norm(r["text"]), norm(hyp.get(r["audio_filepath"], ""))
        we += lev(ref.split(), h.split()); wn += len(ref.split()); ce += lev(ref, h); cn += len(ref); emp += not h
    return 100 * we / max(1, wn), 100 * ce / max(1, cn), emp


def report(a):
    import numpy as np
    import soundfile as sf
    out = Path(a.out)
    names = a.models.split(",") + ([a.teacher] if a.teacher else [])
    sets = sorted(p.stem[4:] for p in out.glob("sub_*.json"))
    table, pairs = {}, []
    for s in sets:
        rows = [json.loads(l) for l in open(out / f"sub_{s}.json")]
        hyps = {n: json.load(open(out / f"hyp_{n}_{s}.json")) for n in a.models.split(",")}
        if a.teacher:
            hyps[a.teacher] = {r["audio_filepath"]: r["teacher"] for r in map(json.loads, open(out / f"teacher_{s}.json"))}
        table[s] = {n: scores(rows, hyps[n]) for n in names}
        if a.teacher and a.pair:
            for r in rows:
                ref = norm(r["text"]).split()
                ws = lev(ref, norm(hyps[a.pair].get(r["audio_filepath"], "")).split()) / max(1, len(ref))
                wt = lev(ref, norm(hyps[a.teacher].get(r["audio_filepath"], "")).split()) / max(1, len(ref))
                x, _ = sf.read(r["audio_filepath"], dtype="float32")
                rms = float(20 * np.log10(np.sqrt(np.mean(np.asarray(x) ** 2)) + 1e-9))
                pairs.append((s, r["duration"], rms, ws, wt))
    print("| set | " + " | ".join(names) + (f" | gap {a.pair}−{a.teacher}" if a.teacher and a.pair else "") + " |")
    print("|---" * (1 + len(names) + bool(a.teacher and a.pair)) + "|")
    for s in sets:
        row = [f"{w:.2f} / {c:.2f} / {e}" for w, c, e in (table[s][n] for n in names)]
        if a.teacher and a.pair:
            row.append(f"{table[s][a.pair][0] - table[s][a.teacher][0]:+.2f} / {table[s][a.pair][1] - table[s][a.teacher][1]:+.2f}")
        print(f"| {s} | " + " | ".join(row) + " |")
    mac = {n: (np.mean([table[s][n][0] for s in sets]), np.mean([table[s][n][1] for s in sets])) for n in names}
    row = [f"{mac[n][0]:.2f} / {mac[n][1]:.2f}" for n in names]
    if a.teacher and a.pair:
        row.append(f"{mac[a.pair][0] - mac[a.teacher][0]:+.2f} / {mac[a.pair][1] - mac[a.teacher][1]:+.2f}")
    print("| macro | " + " | ".join(row) + " |")
    if pairs:
        def wl(ps):
            w = sum(p[3] < p[4] - 1e-9 for p in ps); t = sum(abs(p[3] - p[4]) <= 1e-9 for p in ps)
            return f"{a.pair} better {w} / tie {t} / teacher better {len(ps) - w - t} (n {len(ps)}, median per-clip WER {a.pair} {100 * np.median([p[3] for p in ps]):.1f} vs {100 * np.median([p[4] for p in ps]):.1f})"
        print(f"\npaired per utterance, {a.pair} vs {a.teacher}:")
        print(f"   all: {wl(pairs)}")
        for s in sets:
            print(f"   {s}: {wl([p for p in pairs if p[0] == s])}")
        for k, lab in ((1, "duration s"), (2, "RMS dBFS")):
            v = sorted(p[k] for p in pairs); q1, q2 = v[len(v) // 3], v[2 * len(v) // 3]
            for lo, hi, tag in ((-1e9, q1, f"<{q1:.1f}"), (q1, q2, f"{q1:.1f}..{q2:.1f}"), (q2, 1e9, f">={q2:.1f}")):
                print(f"   {lab} {tag}: {wl([p for p in pairs if lo <= p[k] < hi])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["subsample", "decode", "report"])
    ap.add_argument("--vals", default=""); ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--out", required=True); ap.add_argument("--nemo", default=""); ap.add_argument("--name", default="")
    ap.add_argument("--models", default=""); ap.add_argument("--teacher", default=""); ap.add_argument("--pair", default="")
    a = ap.parse_args()
    {"subsample": subsample, "decode": decode, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
