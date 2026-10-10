#!/usr/bin/env python3
"""Which val clips decode EMPTY (or near-empty), and what they have in common.
--pad-lead S re-decodes with S seconds of quiet before the speech (onset hypothesis).

    $PY empty_clips.py --nemo run/final.nemo --val vp=eval_vp.json,cv=eval_cv.json [--val-n 200]

Same subsample and greedy decoder as two_stage.py. Per set: the empty / <=2-word
hypotheses with duration, sample peak and RMS dBFS, leading-silence length
(first sample above -40 dBFS), and the same stats over the non-empty clips.
"""
import argparse, json, re, unicodedata
import numpy as np, soundfile as sf, torch
from omegaconf import open_dict
from nemo.collections.asr.models import EncDecRNNTBPEModel

ap = argparse.ArgumentParser()
ap.add_argument("--nemo", required=True); ap.add_argument("--val", required=True); ap.add_argument("--val-n", type=int, default=200)
ap.add_argument("--pad-lead", type=float, default=0.0, help="prepend this many s of -75 dBFS quiet (onset hypothesis)")
a = ap.parse_args()
m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location="cuda").eval()
dc = m.cfg.decoding
with open_dict(dc):
    dc.strategy = "greedy"; dc.greedy.use_cuda_graph_decoder = False
m.change_decoding_strategy(dc)


def norm(t):
    t = "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def feats(x):
    db = 20 * np.log10(np.abs(x) + 1e-9)
    lead = int(np.argmax(db > -40)) / 16000 if (db > -40).any() else len(x) / 16000
    return len(x) / 16000, float(db.max()), float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)), lead


for spec in a.val.split(","):
    name, path = spec.split("=", 1)
    rs = [json.loads(l) for l in open(path)]
    rs = rs[:: max(1, len(rs) // a.val_n)][: a.val_n]
    rows = []
    for i in range(0, len(rs), 16):
        xs = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in rs[i:i + 16]]
        if a.pad_lead:
            xs = [np.concatenate([(np.random.randn(int(16000 * a.pad_lead)) * 10 ** (-75 / 20)).astype(np.float32), x]) for x in xs]
        X = torch.zeros(len(xs), max(len(x) for x in xs))
        for j, x in enumerate(xs):
            X[j, :len(x)] = torch.from_numpy(x)
        with torch.no_grad():
            f, fl = m.preprocessor(input_signal=X.cuda(), length=torch.tensor([len(x) for x in xs]).cuda())
            e, el = m.encoder(audio_signal=f, length=fl)
            hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el, return_hypotheses=True)
        hy = hy[0] if isinstance(hy, tuple) else hy
        for r, x, h in zip(rs[i:i + 16], xs, hy):
            h = h[0] if isinstance(h, list) else h
            ys = [t for t in (h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)) if t < 1024]
            rows.append((r, m.tokenizer.ids_to_text(ys), feats(x)))
    bad = [t for t in rows if len(t[1].split()) <= 2]
    good = [t for t in rows if len(t[1].split()) > 2]
    med = lambda ts, k: round(float(np.median([t[2][k] for t in ts])), 1) if ts else None  # noqa: E731
    print(f"== {name}: {len(bad)}/{len(rows)} empty or <=2 words")
    for lab, ts in (("bad", bad), ("good", good)):
        print(f"   {lab:4s} median dur {med(ts, 0)} s, peak {med(ts, 1)} dBFS, RMS {med(ts, 2)} dBFS, lead-in {med(ts, 3)} s")
    for r, h, (d, pk, rms, lead) in bad[:12]:
        print(f"   {d:5.1f}s pk {pk:6.1f} rms {rms:6.1f} lead {lead:4.1f}s | hyp '{h}' | ref {norm(r['text'])[:70]}")
