#!/usr/bin/env python3
"""Neural-LM rescoring of an RNNT N-best dump (lm_fusion_probe.py --dump-nbest): is the beam -> oracle gap
recoverable with language knowledge?

    python3 nbest_lm_rescore.py --dir /root/ft/analysis/K/e3s8k --ks 8,32 --lm Qwen/Qwen2.5-0.5B \
        [--adapt-text a.json,b.json --adapt-exclude v1.json,v2.json --adapt-steps 600] --out lm_rescore.json

score' = rnnt_score + alpha * LM log-prob (sum over the hypothesis tokens + EOS, BOS-conditioned) + beta * words.
(alpha, beta) are chosen on the --tune sets ONLY (MLS/FLEURS val clips outside the harness subsample) and
reported on the --report sets (the harness clips). Two LMs per run:
  generic  the pretrained causal LM as is;
  adapted  the same LM after --adapt-steps of causal-LM fine-tuning on the training transcripts given in
           --adapt-text, normalised exactly like the hypotheses (lower case, de-accented, a-z ' space), minus
           every sentence that also occurs in --adapt-exclude (the val manifests). This adapts the FORMAT
           (no accents/punctuation) as well as the domain; the two cannot be told apart here.
Reported per K: beam (alpha 0), each LM tuned, oracle@K, per set and macro, and the oracle recovery fraction
(beam - LM) / (beam - oracle@K). Also the alpha curve (best beta per alpha) on tune and report.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import unicodedata

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbest_audit import load, macro, wer  # noqa: E402


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


@torch.no_grad()
def lm_scores(model, tok, texts, bs=128):
    """sum log p(text EOS | BOS) per text (natural log): the EOS term keeps an empty hypothesis from scoring 0."""
    model.eval()
    out = {}
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    texts = sorted(set(texts), key=len)
    for i in range(0, len(texts), bs):
        ch = texts[i:i + bs]
        ids = [[bos] + tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for t in ch]
        L = max(len(x) for x in ids)
        x = torch.full((len(ids), L), tok.pad_token_id if tok.pad_token_id is not None else 0)
        m = torch.zeros((len(ids), L), dtype=torch.long)
        for b, s in enumerate(ids):
            x[b, :len(s)] = torch.tensor(s); m[b, :len(s)] = 1
        x, m = x.cuda(), m.cuda()
        lp = torch.log_softmax(model(input_ids=x, attention_mask=m).logits.float(), -1)
        g = lp[:, :-1].gather(2, x[:, 1:, None])[..., 0] * m[:, 1:]
        for t, v in zip(ch, g.sum(1).tolist()):
            out[t] = v
    return out


def adapt(model, tok, texts, steps, lr=5e-5, bs=32, seed=0):
    random.seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    eos = tok.eos_token_id
    bos = tok.bos_token_id if tok.bos_token_id is not None else eos
    for step in range(steps):
        ch = random.sample(texts, bs)
        ids = [[bos] + tok(t, add_special_tokens=False)["input_ids"][:126] + [eos] for t in ch]
        L = max(len(x) for x in ids)
        x = torch.full((bs, L), tok.pad_token_id if tok.pad_token_id is not None else 0)
        lab = torch.full((bs, L), -100)
        for b, s in enumerate(ids):
            x[b, :len(s)] = torch.tensor(s); lab[b, :len(s)] = torch.tensor(s)
        loss = model(input_ids=x.cuda(), labels=lab.cuda()).loss
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 100 == 0 or step == steps - 1:
            print(f"   adapt step {step}/{steps} loss {loss.item():.3f}", flush=True)
    model.eval()


def evaluate(clips, rep, tun, lm, alphas, betas):
    tsets = [s for s in tun if any(c["set"] == s for c in clips)]
    R = [c for c in clips if c["set"] in rep]
    T = [c for c in clips if c["set"] in tsets]

    def pick(al, be):
        return lambda c: max(c["hyps"], key=lambda h: h["score"] + al * lm[h["text"]] + be * h["words"])

    curve, best = [], None
    for al in alphas:
        tb = min(((macro(wer(T, pick(al, be)), tsets), be) for be in betas))
        rr = wer(R, pick(al, tb[1]))
        curve.append({"alpha": al, "beta": tb[1], "tune_macro": round(tb[0], 2), "report_macro": round(macro(rr, rep), 2)})
        if best is None or tb[0] < best[0]:
            best = (tb[0], al, tb[1], rr)
    return best, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ks", default="8,32")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--adapt-text", default="", help="training manifests whose text adapts the LM ('' = generic only)")
    ap.add_argument("--adapt-exclude", default="", help="manifests whose sentences are removed from the adaptation text")
    ap.add_argument("--adapt-steps", type=int, default=600)
    ap.add_argument("--report", default="mls,fleurs,cv,vp")
    ap.add_argument("--tune", default="mls_tune,fleurs_tune")
    ap.add_argument("--fixed", default="", help="name:alpha:beta,... apply weights chosen elsewhere (e.g. on the tune "
                    "split of another dump) instead of tuning: for a holdout bank with no tune split")
    ap.add_argument("--out", default="lm_rescore.json")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rep, tun = a.report.split(","), a.tune.split(",")
    alphas = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    betas = [-1, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
    dumps = {int(k): load(os.path.join(a.dir, f"nbest_b{k}.jsonl")) for k in a.ks.split(",")}
    texts = {h["text"] for cl in dumps.values() for c in cl for h in c["hyps"]}
    print(f"== {len(texts)} distinct hypotheses over K={list(dumps)}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.lm)
    model = AutoModelForCausalLM.from_pretrained(a.lm, torch_dtype=torch.bfloat16).cuda()
    lms = {"generic": lm_scores(model, tok, list(texts))}
    if a.adapt_text:
        excl = {norm(json.loads(l)["text"]) for p in a.adapt_exclude.split(",") if p for l in open(p)}
        tr = [norm(json.loads(l)["text"]) for p in a.adapt_text.split(",") for l in open(p)]
        tr = [t for t in tr if t and t not in excl]
        print(f"== adapt on {len(tr)} normalised training sentences ({len(excl)} val sentences excluded)", flush=True)
        model = model.float()
        adapt(model, tok, tr, a.adapt_steps)
        lms["adapted"] = lm_scores(model, tok, list(texts))
    res = {}
    for K, clips in dumps.items():
        R = [c for c in clips if c["set"] in rep]
        beam = wer(R, lambda c: c["hyps"][0])
        orc = wer(R, lambda c: min(c["hyps"], key=lambda h: h["err"]))
        bm, om = macro(beam, rep), macro(orc, rep)
        res[K] = {"beam": {**{s: round(beam[s], 2) for s in rep}, "macro": round(bm, 2)},
                  "oracle": {**{s: round(orc[s], 2) for s in rep}, "macro": round(om, 2)}}
        print(f"== K={K}: beam {res[K]['beam']}\n   oracle {res[K]['oracle']}")
        fixed = {f.split(":")[0]: tuple(float(x) for x in f.split(":")[1:]) for f in a.fixed.split(",") if f}
        for name, lm in lms.items():
            if name in fixed:
                al, be = fixed[name]
                rr = wer(R, lambda c: max(c["hyps"], key=lambda h: h["score"] + al * lm[h["text"]] + be * h["words"]))
                lmm = macro(rr, rep)
                res[K][name] = {"alpha": al, "beta": be, "fixed": True, **{s: round(rr[s], 2) for s in rep},
                                "macro": round(lmm, 2), "recovery": round((bm - lmm) / (bm - om), 3) if bm > om else None}
                print(f"   {name} (fixed): " + json.dumps(res[K][name]))
                continue
            best, curve = evaluate(clips, rep, tun, lm, alphas, betas)
            _, al, be, rr = best
            lmm = macro(rr, rep)
            res[K][name] = {"alpha": al, "beta": be, **{s: round(rr[s], 2) for s in rep}, "macro": round(lmm, 2),
                            "recovery": round((bm - lmm) / (bm - om), 3) if bm > om else None, "curve": curve}
            print(f"   {name}: " + json.dumps({k: v for k, v in res[K][name].items() if k != "curve"}))
            print("      alpha curve (beta chosen on tune):", [(c["alpha"], c["beta"], c["tune_macro"], c["report_macro"]) for c in curve])
    json.dump(res, open(a.out, "w"), indent=1)
    print("== LM-RESCORE-DONE")


if __name__ == "__main__":
    main()
