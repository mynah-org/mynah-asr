#!/usr/bin/env python3
"""How far did a fine-tune move the encoder, and does its output still depend on the audio?

Compares a fine-tuned RNNT .nemo with its base (both EncDecRNNTBPEModel, same
encoder architecture) on CPU. Rewritten from the ad-hoc probe of 2026-10-09
whose readings were (collapsed cold EOU-IT run vs stock): encoder weights
moved little (median 0.9 %, max 9.3 % relative change), output cosine to stock
0.911, but the per-channel temporal std fell 0.169 -> 0.040; the frozen arm
collapsed as well, so encoder drift was NOT the primary cause.

Metrics:
  weights   per encoder tensor ||W_ft - W_base|| / ||W_base||: median, p90, max, top-k names
  per clip (encoder output [D, T] of base and ft on the same audio):
    output_cosine        cosine of the flattened outputs
    delta_norm           ||E_ft - E_base|| / ||E_base||
    temporal_std         mean over channels of the std over time (base and ft)
    near_constant_dims   channels whose temporal std < --const-thr x the base median channel std
  across clips:
    cross_utt_cosine     mean pairwise cosine of the time-pooled outputs of DIFFERENT clips
                         (close to 1 = the output barely depends on which clip it is)
    shuffle_cosine       cosine between the pooled output of a clip and of the same clip with
                         its 100 ms blocks shuffled (content destroyed, spectrum kept)
    audio_dependence     distinct greedy hypotheses / clips (base and ft)

    $PY encoder_diff.py --base models/parakeet_realtime_eou_120m-v1.nemo \\
        --ft runs/eou-it-5h-e52/final.nemo --manifest manifests/eval_fleurs_it.json --n 8 --json diff.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--const-thr", type=float, default=0.1)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    import rnnt_diag as D
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    base = EncDecRNNTBPEModel.restore_from(a.base, map_location=a.device).eval()
    ft = EncDecRNNTBPEModel.restore_from(a.ft, map_location=a.device).eval()

    # ---- weights
    sb, sf_ = base.encoder.state_dict(), ft.encoder.state_dict()
    rel = {}
    for k, w0 in sb.items():
        if k in sf_ and w0.dtype.is_floating_point and w0.numel() > 1 and w0.shape == sf_[k].shape:
            n0 = float(w0.float().norm())
            if n0 > 0:
                rel[k] = float((sf_[k].float() - w0.float()).norm()) / n0
    vals = np.array(sorted(rel.values()))
    weights = {"tensors": len(vals), "median": round(float(np.median(vals)), 5), "p90": round(float(np.quantile(vals, 0.9)), 5),
               "max": round(float(vals.max()), 5),
               "top": [[k, round(v, 5)] for k, v in sorted(rel.items(), key=lambda kv: -kv[1])[: a.top_k]]}

    # ---- outputs
    rows = [json.loads(line) for line in open(a.manifest, encoding="utf-8") if line.strip()][: a.n]

    def load(r):
        x, sr = sf.read(r["audio_filepath"], dtype="float32", always_2d=True)
        x = x.mean(1)
        if r.get("offset") is not None:
            s = int(float(r["offset"]) * sr)
            x = x[s:s + int(float(r["duration"]) * sr)]
        return x, sr

    def enc(model, x):
        t = torch.tensor(x)[None].to(a.device)
        with torch.no_grad():
            f, fl = model.preprocessor(input_signal=t, length=torch.tensor([t.shape[1]], device=a.device))
            e, el = model.encoder(audio_signal=f, length=fl)
        return e, el

    def greedy(model, e, el):
        with torch.no_grad():
            hy = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el,
                                                                return_hypotheses=True)
        hy = hy[0] if isinstance(hy, tuple) else hy
        V = model.tokenizer.vocab_size
        return model.tokenizer.ids_to_text([i for i in D.hyp_ids(hy[0] if isinstance(hy, list) else hy) if i < V - 2])

    per, pooled_b, pooled_f, shuf, hyp_b, hyp_f = [], [], [], [], [], []
    for r in rows:
        x, sr = load(r)
        eb, elb = enc(base, x)
        ef, elf = enc(ft, x)
        B = eb[0, :, : int(elb[0])].float().numpy()
        F = ef[0, :, : int(elf[0])].float().numpy()
        cos = float((B * F).sum() / (np.linalg.norm(B) * np.linalg.norm(F) + 1e-12))
        sd_b, sd_f = B.std(axis=1), F.std(axis=1)
        thr = a.const_thr * float(np.median(sd_b))
        per.append({"utt": Path(r["audio_filepath"]).name, "frames": int(B.shape[1]), "output_cosine": round(cos, 4),
                    "delta_norm": round(float(np.linalg.norm(F - B) / (np.linalg.norm(B) + 1e-12)), 4),
                    "temporal_std_base": round(float(sd_b.mean()), 4), "temporal_std_ft": round(float(sd_f.mean()), 4),
                    "near_constant_dims_base": int((sd_b < thr).sum()), "near_constant_dims_ft": int((sd_f < thr).sum())})
        pooled_b.append(B.mean(axis=1))
        pooled_f.append(F.mean(axis=1))
        blk = int(0.1 * sr)
        nb = len(x) // blk
        xs = x[: nb * blk].reshape(nb, blk)[rng.permutation(nb)].reshape(-1) if nb > 1 else x
        sh_b = enc(base, xs)[0][0].float().numpy().mean(axis=1)
        sh_f = enc(ft, xs)[0][0].float().numpy().mean(axis=1)
        c = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))  # noqa: E731
        shuf.append({"base": c(pooled_b[-1], sh_b), "ft": c(pooled_f[-1], sh_f)})
        hyp_b.append(greedy(base, eb, elb))
        hyp_f.append(greedy(ft, ef, elf))

    def cross(P):
        cs = [float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)) for u, v in itertools.combinations(P, 2)]
        return round(float(np.mean(cs)), 4) if cs else None

    summary = {
        "weights": weights,
        "output_cosine_mean": round(float(np.mean([p["output_cosine"] for p in per])), 4),
        "delta_norm_mean": round(float(np.mean([p["delta_norm"] for p in per])), 4),
        "temporal_std_base": round(float(np.mean([p["temporal_std_base"] for p in per])), 4),
        "temporal_std_ft": round(float(np.mean([p["temporal_std_ft"] for p in per])), 4),
        "near_constant_dims_base": round(float(np.mean([p["near_constant_dims_base"] for p in per])), 1),
        "near_constant_dims_ft": round(float(np.mean([p["near_constant_dims_ft"] for p in per])), 1),
        "cross_utt_cosine_base": cross(pooled_b), "cross_utt_cosine_ft": cross(pooled_f),
        "shuffle_cosine_base": round(float(np.mean([s["base"] for s in shuf])), 4),
        "shuffle_cosine_ft": round(float(np.mean([s["ft"] for s in shuf])), 4),
        "audio_dependence_base": D.audio_dependence(hyp_b), "audio_dependence_ft": D.audio_dependence(hyp_f),
        "clips": len(per),
    }
    for p, hb, hf in zip(per, hyp_b, hyp_f):
        print("ROW", json.dumps({**p, "hyp_base": hb[:60], "hyp_ft": hf[:60]}, ensure_ascii=False))
    print("SUMMARY", json.dumps(summary))
    if a.json:
        Path(a.json).write_text(json.dumps({"base": a.base, "ft": a.ft, "manifest": a.manifest, "summary": summary,
                                            "per_clip": per, "hyp_base": hyp_b, "hyp_ft": hyp_f},
                                           ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
