#!/usr/bin/env python3
"""Where a stage-2 (<EOU>) model loses words vs its plain parent (greedy, offline, same clips; CPU by default).

    s2_diag.py --parent plain/final.nemo --s2 s2/final.nemo --vals "mls=a.json,vp=b.json" [--per-set 200] [--out d.json]

Per model and set: WER split into S / D / I; deletions at the END of the reference (the last ref words missing
from the hypothesis, the signature of an early stop) vs elsewhere; per clip, <EOU> / <EOB> emissions and whether
an <EOU> comes BEFORE the last non-EOU token (premature, inside the utterance); words per hypothesis / reference.
The same clip subset as lm_fusion_probe.py (every len // per-set-th clip).
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
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--post-pad-s", type=float, default=0.0, help="digital silence appended (EOF flush), as lm_fusion_probe.py")
ap.add_argument("--out", default="")
a = ap.parse_args()
torch.set_num_threads(a.threads)


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def align(r, h):
    """word alignment backtrace -> list of ops ('C'|'S'|'D'|'I', ref index or None)."""
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("C" if r[i - 1] == h[j - 1] else "S", i - 1)); i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", i - 1)); i -= 1
        else:
            ops.append(("I", None)); j -= 1
    return ops[::-1]


from nemo.collections.asr.models import EncDecRNNTBPEModel
from omegaconf import open_dict

sets = []
for spec in a.vals.split(","):
    n, p = spec.split("=", 1)
    rs = [json.loads(l) for l in open(p)]
    sets.append((n, rs[:: max(1, len(rs) // a.per_set)][: a.per_set]))

res = {}
for tag, path in (("parent", a.parent), ("s2", a.s2)):
    m = EncDecRNNTBPEModel.restore_from(path, map_location=a.device).eval()
    tk = m.tokenizer
    eou, eob = tk.token_to_id("<EOU>"), tk.token_to_id("<EOB>")
    dc = m.cfg.decoding
    with open_dict(dc):
        dc.strategy = "greedy_batch"; dc.greedy.max_symbols = 10; dc.greedy.loop_labels = True
        dc.greedy.use_cuda_graph_decoder = False; dc.preserve_alignments = False; dc.compute_timestamps = False
    m.change_decoding_strategy(dc, verbose=False)
    for sname, rows in sets:
        c = {"S": 0, "D": 0, "I": 0, "N": 0, "D_end": 0, "clips_D_end": 0, "eou_clips": 0, "eou_premature": 0,
             "eob": 0, "hyp_words": 0, "clips": len(rows)}
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                ch = rows[i:i + 8]
                xs = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in ch]
                xs = [x.mean(1) if x.ndim > 1 else x for x in xs]
                xs = [np.concatenate([x, np.zeros(int(a.post_pad_s * 16000), np.float32)]) for x in xs]
                L = max(len(x) for x in xs)
                X = torch.zeros(len(xs), L)
                for b, x in enumerate(xs):
                    X[b, :len(x)] = torch.from_numpy(x)
                lens = torch.tensor([len(x) for x in xs])
                f, fl = m.preprocessor(input_signal=X.to(a.device), length=lens.to(a.device))
                e, el = m.encoder(audio_signal=f, length=fl)
                hyps = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el, return_hypotheses=True)
                hyps = hyps[0] if isinstance(hyps, tuple) else hyps
                for r, h in zip(ch, hyps):
                    ys = h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)
                    if eou in ys:
                        c["eou_clips"] += 1
                        last_text = max((k for k, t in enumerate(ys) if t < 1024), default=-1)
                        c["eou_premature"] += any(t == eou and k < last_text for k, t in enumerate(ys))
                    c["eob"] += ys.count(eob)
                    ref, hyp = norm(r["text"]).split(), norm(tk.ids_to_text([t for t in ys if t < 1024])).split()
                    ops = align(ref, hyp)
                    for o, _ in ops:
                        if o != "C":
                            c[o] += 1
                    k = 0   # trailing deletions: D ops on the last ref words, after the last matched/substituted word
                    for o, ri in reversed(ops):
                        if o == "D":
                            k += 1
                        elif o in ("C", "S"):
                            break
                    c["D_end"] += k; c["clips_D_end"] += k > 0
                    if k:
                        c.setdefault("D_end_clips", []).append({"dur": round(len(sf.info(r["audio_filepath"]).frames
                            and sf.info(r["audio_filepath"]).frames) / 16000, 2), "ref_words": len(ref), "missing_end": k})
                    c["N"] += len(ref); c["hyp_words"] += len(hyp)
        c.update({"wer": round(100 * (c["S"] + c["D"] + c["I"]) / c["N"], 2),
                  **{f"{o}%": round(100 * c[o] / c["N"], 2) for o in ("S", "D", "I", "D_end")}})
        res[f"{tag}/{sname}"] = c
        print(tag, f"post{a.post_pad_s:g}", sname, "D_end_clips", sorted((x["dur"], x["missing_end"]) for x in c.get("D_end_clips", [])))
        print(tag, f"post{a.post_pad_s:g}", sname, {k: c[k] for k in ("wer", "S%", "D%", "I%", "D_end%", "clips_D_end", "eou_clips",
                                            "eou_premature", "eob", "hyp_words", "N")}, flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
