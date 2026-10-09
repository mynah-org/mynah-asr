"""Inspect what the EOU-IT training dataloader actually feeds the model (CPU)."""
import sys, torch
sys.path.insert(0, "/root/eou-kit")
from omegaconf import OmegaConf
import train_eou_it as T
torch.set_num_threads(8)
model, info = T.build_model("/root/ft/models/parakeet_realtime_eou_120m-v1.nemo", "/root/ft/models/tok-eou-it")
man = "/root/ft/runs/eou-it-5h-e26-frz/train_5h_eou.json"
common = {"sample_rate": 16000, "shuffle": True, "num_workers": 0, "pin_memory": False, "seed": 1,
          "batch_size": None, "batch_duration": 60, "use_bucketing": False,
          "shuffle_buffer_size": 100, "min_duration": 0.3, "max_duration": 30.0,
          "drop_last": True, "ignore_eob_label": True, "use_lhotse": True,
          "window_stride": float(model.cfg.preprocessor.window_stride),
          "subsampling_factor": int(model.cfg.encoder.subsampling_factor),
          "augmentor": {"white_noise": {"prob": 0.9, "min_level": -90, "max_level": -46}}}
cfg = OmegaConf.create({**common, "manifest_filepath": man,
      "random_padding": {"prob": 0.9, "min_post_pad_duration": 3.0, "min_pre_pad_duration": 0.0, "max_pad_duration": 6.0,
                         "max_total_duration": 40.0, "pad_distribution": "uniform", "normal_mean": 0.5, "normal_std": 2.0,
                         "pre_pad_duration": 0.2, "post_pad_duration": 3.0}})
model.setup_training_data(cfg)
it = iter(model._train_dl)
for bi in range(2):
    b = next(it)
    a, al = b.audio_signal, b.audio_lengths
    print("ROW batch", bi, "type", type(b).__name__, "audio", tuple(a.shape), a.dtype, "lens_s", [round(float(x)/16000, 2) for x in al[:4]])
    for i in range(min(3, a.shape[0])):
        x = a[i, :al[i]]
        nz = float((x.abs() > 1e-4).float().mean())
        print("ROW  utt", i, "rms %.4f peak %.4f nonzero %.2f" % (float(x.pow(2).mean().sqrt()), float(x.abs().max()), nz),
              "| tokens:", repr(model.tokenizer.ids_to_text(b.text_tokens[i, :b.text_token_lengths[i]].tolist())[-90:]),
              "| last ids", b.text_tokens[i, max(0, int(b.text_token_lengths[i]) - 3):b.text_token_lengths[i]].tolist())
    for k in ("eou_targets", "eou_target_lengths"):
        v = getattr(b, k, None)
        if v is not None:
            print("ROW ", k, tuple(v.shape), "sum per utt", v.float().sum(-1)[:3].tolist() if v.dim() > 1 else v[:3].tolist())
