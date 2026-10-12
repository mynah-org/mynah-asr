#!/usr/bin/env python3
"""Why does the RNNT rank a lexically worse hypothesis first when a better one is in its own N-best?

    python3 nbest_rnnt_decomp.py --nemo final.nemo --dumps dirA/nbest_b8.jsonl,dirB/nbest_b8.jsonl --out decomp.json

For every clip of the N-best dumps (lm_fusion_probe.py --dump-nbest; audio re-read from the dump's path), the encoder
output is computed once and EVERY hypothesis is teacher-forced through the prediction network + joint:
  full   log P(y | x), the exact transducer likelihood summed over all alignments (forward algorithm on the
         [T, U+1] lattice; the canonical SPE tokenisation of the hypothesis text);
  ilm    internal-LM log-prob of y: joint with the encoder term removed (enc bias kept), blank/<EOU>/<EOB>
         masked, softmax over the 1024 pieces (Meng et al. 2021); the predictor's own language prior;
  ac     full - ilm, the density-ratio "acoustic" share.
Then, on the clips where a lower-ranked hypothesis has fewer word errors than rank 1 (oracle != winner):
  search error  = the model's own full likelihood prefers the oracle hypothesis (beam picked against the model);
  model error   = full prefers the winner; split by whether the ILM and/or the acoustic share prefer the winner.
Rescoring (weights chosen on --tune sets only, applied unchanged to every other set):
  beam_score + a * full + b * ilm + c * words  (b < 0 = ILM subtraction), and `full` alone.
Sets are grouped read/scripted (cv, fleurs, italic, mls) vs spontaneous/presentation (vp, tedx).
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbest_audit import load, macro, wer  # noqa: E402

READ = {"cv", "fleurs", "italic", "mls", "mls_tune", "fleurs_tune"}



def rnnt_logp(lp, y, ylen):
    """batched exact log P(y|x): lp [K, T, Umax+1, V+1] log-softmax joint, y [K, Umax] (padded), ylen [K]."""
    K, T, U1, _ = lp.shape
    blank = lp[..., -1]                                                    # [K, T, U+1]
    if U1 > 1:
        lab = lp[:, :, :-1, :].gather(3, y[:, None, :, None].expand(K, T, U1 - 1, 1))[..., 0]   # [K, T, U]
    else:
        lab = lp.new_zeros(K, T, 0)
    C = torch.cat([lp.new_zeros(K, T, 1), torch.cumsum(lab, 2)], 2)        # [K, T, U+1]
    a0 = torch.full((K, U1), -1e30, device=lp.device); a0[:, 0] = 0
    alpha = C[:, 0] + torch.logcumsumexp(a0 - C[:, 0], 1)
    for t in range(1, T):
        a = alpha + blank[:, t - 1]
        alpha = C[:, t] + torch.logcumsumexp(a - C[:, t], 1)
    k = torch.arange(K, device=lp.device)
    return (alpha[k, ylen] + blank[k, T - 1, ylen]).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--dumps", required=True, help="comma-separated nbest_b<K>.jsonl files (same K)")
    ap.add_argument("--tune", default="mls_tune,fleurs_tune")
    ap.add_argument("--out", default="decomp.json")
    a = ap.parse_args()
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location=dev).eval()
    tk = m.tokenizer
    clips = [c for p in a.dumps.split(",") for c in load(p)]
    print(f"== {len(clips)} clips, {sum(len(c['hyps']) for c in clips)} hypotheses", flush=True)
    with torch.no_grad():
        for i, c in enumerate(clips):
            x, sr = sf.read(c["path"], dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            X = torch.from_numpy(x)[None].to(dev)
            f, fl = m.preprocessor(input_signal=X, length=torch.tensor([X.shape[1]], device=dev))
            e, el = m.encoder(audio_signal=f, length=fl)
            enc = e[:, :, : int(el[0])].transpose(1, 2)            # [1, T, D]
            toks = [[t for t in tk.text_to_ids(h["text"]) if t < 1024] if h["text"] else [] for h in c["hyps"]]
            K, U = len(toks), max(1, max(len(t) for t in toks))
            y = torch.zeros(K, U, dtype=torch.long, device=dev)
            for k, t in enumerate(toks):
                if t:
                    y[k, :len(t)] = torch.tensor(t, device=dev)
            ylen = torch.tensor([len(t) for t in toks], device=dev)
            g, _, _ = m.decoder(targets=y, target_length=ylen)                   # [K, H, U+1]
            g = g.transpose(1, 2)
            lp = torch.log_softmax(m.joint.joint(enc.expand(K, -1, -1), g).float(), -1)   # [K, T, U+1, V+1]
            full = rnnt_logp(lp, y, ylen)
            if i < 3:   # cross-check the forward algorithm against torchaudio's RNNT loss
                import torchaudio.functional as taf
                ref = -taf.rnnt_loss(lp, y.int(), torch.full((K,), lp.shape[1], dtype=torch.int32, device=dev), ylen.int(),
                                     blank=lp.shape[-1] - 1, reduction="none", fused_log_softmax=False)
                print(f"   check clip {i}: max |ours - torchaudio| = {max(abs(p - q) for p, q in zip(full, ref.tolist())):.4f}", flush=True)
            il = torch.log_softmax(m.joint.joint(torch.zeros_like(enc[:, :1]).expand(K, -1, -1), g)[:, 0, :, :1024].float(), -1)
            ilm = il[:, :-1, :].gather(2, y[:, :, None])[..., 0]                 # [K, U]
            mask = torch.arange(U, device=dev)[None] < ylen[:, None]
            ilm = (ilm * mask).sum(1).tolist()
            for h, fv, iv in zip(c["hyps"], full, ilm):
                h["full"], h["ilm"], h["ac"] = fv, iv, fv - iv
            if i % 500 == 0:
                print(f"   {i}/{len(clips)}", flush=True)
    sets = sorted({c["set"] for c in clips})
    tun = [s for s in a.tune.split(",") if s in sets]
    rep = [s for s in sets if s not in tun]
    res = {"sets": rep, "tune": tun}
    # 1. search vs model error on oracle != winner clips
    cls = {}
    for c in clips:
        o = min(c["hyps"], key=lambda h: (h["err"], c["hyps"].index(h)))
        w = c["hyps"][0]
        if o["err"] >= w["err"]:
            continue
        g = "read" if c["set"] in READ else "spont"
        k = cls.setdefault(c["set"], {"n": 0, "search": 0, "ilm_pref_w": 0, "ac_pref_w": 0, "both_pref_w": 0})
        k2 = cls.setdefault(g, {"n": 0, "search": 0, "ilm_pref_w": 0, "ac_pref_w": 0, "both_pref_w": 0})
        for d in (k, k2):
            d["n"] += 1
            if o["full"] > w["full"]:
                d["search"] += 1
            else:
                d["ilm_pref_w"] += w["ilm"] > o["ilm"]
                d["ac_pref_w"] += w["ac"] > o["ac"]
                d["both_pref_w"] += (w["ilm"] > o["ilm"]) and (w["ac"] > o["ac"])
    res["oracle_ne_winner"] = cls
    print("== oracle != winner: search error = full likelihood prefers the oracle; otherwise which share prefers the winner")
    for s, d in cls.items():
        n, se = d["n"], d["search"]
        me = n - se
        print(f"   {s:12s} n {n:4d}  search {se / n:5.1%}  model {me / n:5.1%}"
              + (f"  (of model errors: ilm prefers winner {d['ilm_pref_w'] / me:5.1%}, acoustic prefers winner {d['ac_pref_w'] / me:5.1%},"
                 f" both {d['both_pref_w'] / me:5.1%})" if me else ""))
    # 2. rescoring, chosen on tune only

    def pick(al, be, ga):
        return lambda c: max(c["hyps"], key=lambda h: (h["score"] if al is None else al * h["full"]) + (be or 0) * h["ilm"] + ga * h["words"])

    T = [c for c in clips if c["set"] in tun]
    grid = [(None, b, g) for b in (0, -0.1, -0.2, -0.3, -0.5) for g in (0, 0.5, 1, 2)] + \
           [(1.0, b, g) for b in (0, -0.1, -0.2, -0.3, -0.5) for g in (0, 0.5, 1, 2)]
    tw = {p: macro(wer(T, pick(*p)), tun) for p in grid}
    base = wer([c for c in clips if c["set"] in rep], lambda c: c["hyps"][0])
    res["beam"] = {s: round(base[s], 2) for s in rep}
    full_only = wer([c for c in clips if c["set"] in rep], pick(1.0, 0, 0))
    res["full_only"] = {s: round(full_only[s], 2) for s in rep}
    best = min(tw, key=tw.get)
    bw = wer([c for c in clips if c["set"] in rep], pick(*best))
    res["rescore"] = {"chosen_on_tune": list(best), "tune_macro": round(tw[best], 2),
                      "tune_beam": round(macro(wer(T, lambda c: c["hyps"][0]), tun), 2), **{s: round(bw[s], 2) for s in rep}}
    print("== beam rank 1      ", res["beam"])
    print("== argmax full logP ", res["full_only"])
    print("== tuned rescoring  ", res["rescore"])
    json.dump(res, open(a.out, "w"), indent=1)
    print("== DECOMP-DONE")


if __name__ == "__main__":
    main()
