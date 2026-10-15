"""Weight average ("model soup") of fine-tuned .nemo checkpoints that share one initialisation.

    python soup.py --nemos a.nemo,b.nemo[,c.nemo] [--weights 0.5,0.5] --out soup.nemo

Every floating-point tensor of the state dict is the weighted mean of the inputs (encoder, predictor, joint,
BatchNorm running statistics); integer buffers (num_batches_tracked) are taken from the last input. The config,
tokenizer and preprocessor come from the last input. No training: evaluate the result like any checkpoint.
"""
import argparse

import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

ap = argparse.ArgumentParser()
ap.add_argument("--nemos", required=True)
ap.add_argument("--weights", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
paths = a.nemos.split(",")
w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0 / len(paths)] * len(paths)
assert len(w) == len(paths) and abs(sum(w) - 1) < 1e-6, "one weight per checkpoint, summing to 1"
sds = [EncDecRNNTBPEModel.restore_from(p, map_location="cpu").state_dict() for p in paths[:-1]]
m = EncDecRNNTBPEModel.restore_from(paths[-1], map_location="cpu")
sd = m.state_dict()
assert all(s.keys() == sd.keys() for s in sds), "checkpoints do not share one architecture"
avg = {}
for k, v in sd.items():
    if v.is_floating_point():
        avg[k] = sum(wi * s[k].float() for wi, s in zip(w[:-1], sds)) + w[-1] * v.float()
        avg[k] = avg[k].to(v.dtype)
    else:
        avg[k] = v
m.load_state_dict(avg)
m.save_to(a.out)
print(f"soup of {len(paths)} checkpoints, weights {w} -> {a.out}")
