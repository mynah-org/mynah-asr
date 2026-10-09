#!/usr/bin/env python3
"""Inspect what the EOU training dataloader actually feeds the model (CPU).

Builds the model exactly as train_eou.py does and the same lhotse EOU dataset
(padding, white noise), then prints per utterance: rms / peak / non-zero share
of the audio, the tail of the decoded target text and its last ids (<EOU> must
be last). On 2026-10-09 this showed a sane batch (rms 0.03-0.10, peak 0.3-0.7,
Italian text with <EOU>=1024 last), ruling the data pipeline out as the cause
of the blank collapse.

    FT_ROOT=/root/ft $PY batch_probe.py --stock models/parakeet_realtime_eou_120m-v1.nemo \\
        --tokenizer models/tok-eou-it --manifest runs/<tag>/train_5h_eou.json --eou-weight 0.9
"""
import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent.parent.parent / "common")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--manifest", required=True, help="a train_<subset>_eou.json written by train_eou.py")
    ap.add_argument("--eou-weight", type=float, required=True, help="padding probability to inspect")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--batch-duration", type=float, default=60.0)
    a = ap.parse_args()

    import torch
    from omegaconf import OmegaConf

    import train_eou as T

    torch.set_num_threads(8)
    model, _ = T.build_model(a.stock, a.tokenizer)
    common = {"sample_rate": 16000, "shuffle": True, "num_workers": 0, "pin_memory": False, "seed": 1,
              "batch_size": None, "batch_duration": a.batch_duration, "use_bucketing": False,
              "shuffle_buffer_size": 100, "min_duration": 0.3, "max_duration": 30.0,
              "drop_last": True, "ignore_eob_label": True, "use_lhotse": True,
              "window_stride": float(model.cfg.preprocessor.window_stride),
              "subsampling_factor": int(model.cfg.encoder.subsampling_factor),
              "augmentor": {"white_noise": {"prob": 0.9, "min_level": -90, "max_level": -46}}}
    cfg = OmegaConf.create({**common, "manifest_filepath": a.manifest,
                            "random_padding": {"prob": a.eou_weight, "min_post_pad_duration": 3.0,
                                               "min_pre_pad_duration": 0.0, "max_pad_duration": 6.0,
                                               "max_total_duration": 40.0, "pad_distribution": "uniform",
                                               "normal_mean": 0.5, "normal_std": 2.0,
                                               "pre_pad_duration": 0.2, "post_pad_duration": 3.0}})
    model.setup_training_data(cfg)
    it = iter(model._train_dl)
    for bi in range(a.batches):
        b = next(it)
        x, xl = b.audio_signal, b.audio_lengths
        print("ROW batch", bi, "type", type(b).__name__, "audio", tuple(x.shape), x.dtype,
              "lens_s", [round(float(v) / 16000, 2) for v in xl[:4]])
        for i in range(min(3, x.shape[0])):
            y = x[i, :xl[i]]
            nz = float((y.abs() > 1e-4).float().mean())
            n = int(b.text_token_lengths[i])
            print("ROW  utt", i, "rms %.4f peak %.4f nonzero %.2f" % (float(y.pow(2).mean().sqrt()), float(y.abs().max()), nz),
                  "| tokens:", repr(model.tokenizer.ids_to_text(b.text_tokens[i, :n].tolist())[-90:]),
                  "| last ids", b.text_tokens[i, max(0, n - 3):n].tolist())
        for k in ("eou_targets", "eou_target_lengths"):
            v = getattr(b, k, None)
            if v is not None:
                print("ROW ", k, tuple(v.shape), "sum per utt", v.float().sum(-1)[:3].tolist() if v.dim() > 1 else v[:3].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
