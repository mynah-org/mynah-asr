#!/usr/bin/env python3
"""Which val clips decode EMPTY (or near-empty), and what they have in common.
--pad-lead S re-decodes with S seconds of quiet before the speech (onset hypothesis).

    $PY empty_clips.py --nemo run/final.nemo --val vp=eval_vp.json,cv=eval_cv.json [--val-n 200]

Same subsample and greedy decoder as two_stage.py. Per set: the empty / <=2-word
hypotheses with duration, sample peak and RMS dBFS, leading-silence length
(first sample above -40 dBFS), and the same stats over the non-empty clips.
"""
import argparse, json, re, unicodedata
from pathlib import Path
import numpy as np, soundfile as sf, torch
from omegaconf import open_dict
from nemo.collections.asr.models import EncDecRNNTBPEModel

ap = argparse.ArgumentParser()
ap.add_argument("--nemo", required=True); ap.add_argument("--val", required=True); ap.add_argument("--val-n", type=int, default=200)
ap.add_argument("--ref-nemo", default="", help="second model (e.g. the run before) decoded on the SAME clips: overlap table")
ap.add_argument("--dump", default="", help="dir: failures.json (+ wav copies) for a teacher check / listening")
ap.add_argument("--pad-lead", type=float, default=0.0, help="prepend this many s of -75 dBFS quiet (onset hypothesis)")
a = ap.parse_args()
def load_model(path):
    mm = EncDecRNNTBPEModel.restore_from(path, map_location="cuda").eval()
    dc = mm.cfg.decoding
    with open_dict(dc):
        dc.strategy = "greedy"; dc.greedy.use_cuda_graph_decoder = False
    mm.change_decoding_strategy(dc)
    return mm


models = [load_model(a.nemo)] + ([load_model(a.ref_nemo)] if a.ref_nemo else [])


def norm(t):
    t = "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def feats(x):
    db = 20 * np.log10(np.abs(x) + 1e-9)
    lead = int(np.argmax(db > -40)) / 16000 if (db > -40).any() else len(x) / 16000
    return len(x) / 16000, float(db.max()), float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)), lead


def lev(x, y):
    p = list(range(len(y) + 1))
    for i, u in enumerate(x, 1):
        c = [i]
        for j, v in enumerate(y, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (u != v)))
        p = c
    return p[-1]


def decode(mm, xs):
    X = torch.zeros(len(xs), max(len(x) for x in xs))
    for j, x in enumerate(xs):
        X[j, :len(x)] = torch.from_numpy(x)
    with torch.no_grad():
        f, fl = mm.preprocessor(input_signal=X.cuda(), length=torch.tensor([len(x) for x in xs]).cuda())
        e, el = mm.encoder(audio_signal=f, length=fl)
        hy = mm.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el, return_hypotheses=True)
    hy = hy[0] if isinstance(hy, tuple) else hy
    out = []
    for h in hy:
        h = h[0] if isinstance(h, list) else h
        ys = [t for t in (h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)) if t < 1024]
        out.append(mm.tokenizer.ids_to_text(ys))
    return out


dump = []
for spec in a.val.split(","):
    name, path = spec.split("=", 1)
    rs = [json.loads(l) for l in open(path)]
    rs = rs[:: max(1, len(rs) // a.val_n)][: a.val_n]
    rows = []   # (row, hyp, feats, ref_hyp)
    for i in range(0, len(rs), 16):
        xs = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in rs[i:i + 16]]
        if a.pad_lead:
            xs = [np.concatenate([(np.random.randn(int(16000 * a.pad_lead)) * 10 ** (-75 / 20)).astype(np.float32), x]) for x in xs]
        hs = [decode(mm, xs) for mm in models]
        for k, (r, x) in enumerate(zip(rs[i:i + 16], xs)):
            rows.append((r, hs[0][k], feats(x), hs[1][k] if len(hs) > 1 else None))
    fail = lambda h: len(h.split()) <= 2  # noqa: E731
    bad = [t for t in rows if fail(t[1])]
    good = [t for t in rows if not fail(t[1])]
    med = lambda ts, k: round(float(np.median([t[2][k] for t in ts])), 1) if ts else None  # noqa: E731
    print(f"== {name}: {len(bad)}/{len(rows)} empty or <=2 words (model)")
    for lab, ts in (("bad", bad), ("good", good)):
        print(f"   {lab:4s} median dur {med(ts, 0)} s, peak {med(ts, 1)} dBFS, RMS {med(ts, 2)} dBFS, lead-in {med(ts, 3)} s")
    if a.ref_nemo:
        both = sum(fail(t[1]) and fail(t[3]) for t in rows); only_new = sum(fail(t[1]) and not fail(t[3]) for t in rows)
        only_ref = sum(fail(t[3]) and not fail(t[1]) for t in rows)
        print(f"   overlap vs ref model: ref ok & model fails {only_new} | both fail {both} | ref fails & model ok {only_ref} | both ok {len(rows) - both - only_new - only_ref}")
    for r, h, (d, pk, rms, lead), h2 in sorted(bad, key=lambda t: -t[2][0]):
        ref = norm(r["text"])
        w2 = f" | ref-model WER {100 * lev(ref.split(), norm(h2).split()) / max(1, len(ref.split())):.0f}% '{h2[:50]}'" if h2 is not None else ""
        print(f"   {Path(r['audio_filepath']).stem[:44]:44s} spk {str(r.get('speaker', ''))[-10:]:10s} {d:5.1f}s pk {pk:6.1f} rms {rms:6.1f} | hyp '{h}'{w2}")
        print(f"      ref: {ref[:110]}")
        dump.append({"set": name, "audio_filepath": r["audio_filepath"], "speaker": r.get("speaker"), "duration": round(d, 2),
                     "peak_dbfs": round(pk, 1), "rms_dbfs": round(rms, 1), "ref": ref, "hyp": h, "ref_model_hyp": h2})
if a.dump:
    import shutil
    Path(a.dump).mkdir(parents=True, exist_ok=True)
    for x in dump:
        shutil.copy(x["audio_filepath"], Path(a.dump) / f"{x['set']}_{Path(x['audio_filepath']).name}")
    json.dump(dump, open(Path(a.dump) / "failures.json", "w"), indent=1, ensure_ascii=False)
    print(f"== dumped {len(dump)} failures + wavs to {a.dump}")
