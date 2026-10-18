#!/usr/bin/env python3
"""Stage-2 (<EOU>) model decoded the way Mynah serves it: decoder reset after every <EOU> (CPU, offline, no training).

    s2_reset.py --parent plain/final.nemo --s2 s2/final.nemo --vals "mls=a.json,..." [--per-set 200] [--out r.json]

Mynah (docs/prior-art.md, issue #13) resets ONLY the decoder after a model <EOU> (prediction-network state to zero,
last token -> SOS), never the encoder cache. Here: one offline encoder pass per clip, then a frame-synchronous RNNT
greedy decode with three protocols on the same clips:
  parent        the plain parent (no <EOU> in its outputs);
  s2            stage 2, no reset (the NeMo offline protocol of the probes);
  s2_reset      stage 2, decoder reset after every <EOU> (the Mynah serving protocol); the transcript is the
                concatenation of the segments.
Per set: WER / CER, S / D / I, end-of-utterance deletions, <EOU> per clip, clips split into >= 2 segments, and
premature <EOU> = emitted while >= 0.5 s of speech remains (energy endpoint: last 80 ms frame within 35 dB of the
clip peak), i.e. a false endpoint at an internal pause. Words recovered = s2 errors - s2_reset errors.
Two separate quantities: ASR continuity after the reset, and EOU precision (false endpoint rate).
"""
import argparse
import json
import re
import unicodedata

import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--parent", required=True)
ap.add_argument("--s2", required=True)
ap.add_argument("--vals", required=True)
ap.add_argument("--per-set", type=int, default=200)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="")
a = ap.parse_args()
torch.set_num_threads(a.threads)
FR = 0.08


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def ops(r, h):
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            out.append("C" if r[i - 1] == h[j - 1] else "S"); i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            out.append("D"); i -= 1
        else:
            out.append("I"); j -= 1
    return out[::-1]


def cer(r, h):
    r, h = list(r), list(h)
    p = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        c = [i]
        for j, y in enumerate(h, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


from nemo.collections.asr.models import EncDecRNNTBPEModel

P = EncDecRNNTBPEModel.restore_from(a.parent, map_location="cpu").eval()
S = EncDecRNNTBPEModel.restore_from(a.s2, map_location="cpu").eval()
tk = S.tokenizer
EOU, BLANK = tk.token_to_id("<EOU>"), tk.vocab_size


def encode(m, x):
    f, fl = m.preprocessor(input_signal=torch.from_numpy(x)[None], length=torch.tensor([len(x)]))
    e, el = m.encoder(audio_signal=f, length=fl)
    return e[0, :, : int(el[0])].T


@torch.no_grad()
def greedy(m, enc, reset=False, max_symbols=10):
    ys, frames = [], []
    g, state = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
    for t in range(enc.shape[0]):
        f = enc[t][None, None, :]
        for _ in range(max_symbols):
            k = int(m.joint.joint(f, g)[0, 0, 0].argmax())
            if k == BLANK:
                break
            ys.append(k); frames.append(t)
            if reset and k == EOU:   # Mynah: decoder state to zero, last token -> SOS; encoder untouched
                g, state = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
                break
            g, state = m.decoder.predict(torch.tensor([[k]]), state=state, add_sos=False, batch_size=1)[:2]
    return ys, frames


def speech_end(x):
    n = int(FR * 16000)
    f = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    on = np.nonzero(db > db.max() - 35)[0]
    return (on[-1] + 1) * FR if len(on) else len(x) / 16000


res = {}
for spec in a.vals.split(","):
    sname, p = spec.split("=", 1)
    rows = [json.loads(l) for l in open(p)]
    rows = rows[:: max(1, len(rows) // a.per_set)][: a.per_set]
    acc = {k: dict(S=0, D=0, I=0, N=0, D_end=0, cerr=0, cN=0, eou=0, split=0, premature=0, prem_clips=0, clips=0)
           for k in ("parent", "s2", "s2_reset")}
    for r in rows:
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        end = speech_end(x)
        ref = norm(r["text"]).split()
        eP, eS = encode(P, x), encode(S, x)
        for k, (m, e, rs) in {"parent": (P, eP, False), "s2": (S, eS, False), "s2_reset": (S, eS, True)}.items():
            ys, fr = greedy(m, e, reset=rs)
            hyp = norm(tk.ids_to_text([t for t in ys if t < 1024])).split()
            o = ops(ref, hyp)
            c = acc[k]
            for op in o:
                if op != "C":
                    c[op] += 1
            j = 0
            for op in reversed(o):
                if op == "D":
                    j += 1
                elif op in ("C", "S"):
                    break
            c["D_end"] += j; c["N"] += len(ref); c["clips"] += 1
            c["cerr"] += cer(" ".join(ref), " ".join(hyp)); c["cN"] += len(" ".join(ref))
            ef = [f for y, f in zip(ys, fr) if y == EOU]
            c["eou"] += len(ef); c["split"] += len(ef) >= 2
            prem = [f for f in ef if f * FR < end - 0.5]
            c["premature"] += len(prem); c["prem_clips"] += bool(prem)
    out = {}
    for k, c in acc.items():
        err = c["S"] + c["D"] + c["I"]
        out[k] = {"wer": round(100 * err / c["N"], 2), "cer": round(100 * c["cerr"] / c["cN"], 2),
                  **{f"{o}%": round(100 * c[o] / c["N"], 2) for o in ("S", "D", "I", "D_end")},
                  "errors": err, "eou_per_clip": round(c["eou"] / c["clips"], 3), "clips_split": c["split"],
                  "premature_eou": c["premature"], "clips_with_premature": c["prem_clips"], "clips": c["clips"]}
    out["words_recovered_by_reset"] = out["s2"]["errors"] - out["s2_reset"]["errors"]
    res[sname] = out
    print("==", sname, json.dumps(out), flush=True)
macro = {k: round(float(np.mean([res[s][k]["wer"] for s in res])), 2) for k in ("parent", "s2", "s2_reset")}
print("== MACRO greedy WER", json.dumps(macro), flush=True)
if a.out:
    json.dump({"sets": res, "macro": macro}, open(a.out, "w"), indent=1)
