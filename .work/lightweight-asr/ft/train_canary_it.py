#!/usr/bin/env python3
"""Smoke fine-tune of Canary 180M (+ it sub-tokenizer) on an Italian subset.

Arms
  A  encoder FROZEN (eval mode, so BatchNorm statistics do not drift);
     train the Transformer decoder, the encoder->decoder projection and the
     (weight-tied) embedding/head including the new `it` rows.
  B  as A, plus the top N encoder layers unfrozen (lower LR: --enc-lr-scale).
  C  full fine-tune (everything trainable; encoder at --enc-lr-scale).

Not the NeMo example script on purpose: we need freezing, our own validation
(leaderboard normaliser, same decode path as eval_it.py) and per-run cost
accounting, all of which are a few lines here and awkward through Hydra
overrides. The training step itself IS NeMo's (EncDecMultiTaskModel.
training_step, lhotse dataloader, canary2 prompt).

Level robustness (recorded A/B option): --gain-aug applies a random
per-utterance gain (default -30..+6 dB, p=0.8, clipped and re-quantised to
PCM16) to every training batch -- see ftlib.make_gain_aug. Before and after
training the IT validation subsets and the EN clips are also decoded with every
clip re-levelled to fixed peaks (--levels, default -3,-20,-40 dBFS), so level
robustness is measured with and without the augmentation. The preprocessor's
`normalize` setting is recorded (per_feature normalisation already cancels
most of a pure gain change; NA does not).

Accounting (written to <run>/metrics.json):
  audio-hours per GPU-hour, samples/s, peak allocated + reserved VRAM
  (torch.cuda.max_memory_*), equivalent-epoch wall time (the lhotse train
  loader is infinite, so an "epoch" = the subset's hours of audio processed),
  total GPU-hours (whole process) and cost at RATE_USD_H. Training wall time
  excludes the in-loop validation passes, which are reported separately.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftlib  # noqa: E402

FT = ftlib.FT
T_PROC = time.time()


def configure_arm(model, arm, top_n):
    for p in model.parameters():
        p.requires_grad_(True)
    enc = model.encoder
    frozen = []
    if arm == "A":
        for p in enc.parameters():
            p.requires_grad_(False)
        frozen = [enc]
    elif arm == "B":
        for p in enc.parameters():
            p.requires_grad_(False)
        layers = enc.layers
        assert 0 < top_n <= len(layers), f"top_n {top_n} vs {len(layers)} layers"
        for layer in layers[len(layers) - top_n:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        frozen = [m for n, m in enc.named_children() if n != "layers"] + list(layers[: len(layers) - top_n])
    elif arm != "C":
        raise SystemExit(f"unknown arm {arm}")
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, {"params_total": n_all, "params_trainable": n_tr}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default=str(FT / "models" / "canary-180m-flash-it.nemo"))
    ap.add_argument("--subset", default="5h")
    ap.add_argument("--arm", default="A", choices=["A", "B", "C"])
    ap.add_argument("--top-n", type=int, default=4, help="arm B: encoder layers unfrozen from the top")
    ap.add_argument("--epochs", type=float, default=10.0, help="sets max_steps if --max-steps is not given")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-duration", type=float, default=300.0, help="seconds of audio per batch (lhotse)")
    ap.add_argument("--num-buckets", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--enc-lr-scale", type=float, default=0.3)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-n", type=int, default=300, help="utterances per eval set for in-loop validation")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rate-usd-h", type=float, default=float(os.environ.get("RATE_USD_H", "0") or 0))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-final-eval", action="store_true")
    ap.add_argument("--gain-aug", type=int, default=int(os.environ.get("GAIN_AUG", "0")), choices=[0, 1])
    ap.add_argument("--gain-lo-db", type=float, default=-30.0)
    ap.add_argument("--gain-hi-db", type=float, default=6.0)
    ap.add_argument("--gain-prob", type=float, default=0.8)
    ap.add_argument("--levels", default=os.environ.get("LEVELS", "-3,-20,-40"),
                    help="peak dBFS levels for the before/after level sweep ('' = skip)")
    a = ap.parse_args()

    tag = a.tag or f"{a.arm}{a.top_n if a.arm == 'B' else ''}-{a.subset}{'-gain' if a.gain_aug else ''}"
    out = FT / "runs" / tag
    if (out / "metrics.json").exists():
        print(f"== skip {tag}: {out}/metrics.json exists")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    import lightning.pytorch as pl
    import torch
    from omegaconf import OmegaConf, open_dict

    from nemo.collections.asr.models import EncDecMultiTaskModel

    pl.seed_everything(a.seed)
    man = FT / "manifests"
    train_man = man / f"train_{a.subset}.json"
    hours = json.loads((FT / "hours.json").read_text())
    subset_s = hours[f"train_{a.subset}"]["seconds"]
    max_steps = a.max_steps or max(50, math.ceil(a.epochs * subset_s / a.batch_duration))
    warmup = max(10, int(a.warmup_frac * max_steps))

    model = EncDecMultiTaskModel.restore_from(a.init, map_location="cpu")
    pre = ftlib.preprocessor_info(model)
    print(f"  preprocessor: {pre}", flush=True)
    frozen, counts = configure_arm(model, a.arm, a.top_n)
    print(f"== {tag}: arm {a.arm} trainable {counts['params_trainable'] / 1e6:.1f}M / {counts['params_total'] / 1e6:.1f}M; "
          f"subset {subset_s / 3600:.3f} h; max_steps {max_steps} warmup {warmup}", flush=True)
    with open_dict(model.cfg):
        model.cfg.optim = OmegaConf.create({
            "name": "adamw", "lr": a.lr, "betas": [0.9, 0.98], "weight_decay": a.weight_decay,
            "sched": {"name": "CosineAnnealing", "warmup_steps": warmup, "min_lr": a.lr * 0.01},
        })
        if a.arm in ("B", "C"):
            model.cfg.optim_param_groups = OmegaConf.create({"encoder": {"lr": a.lr * a.enc_lr_scale}})
        elif "optim_param_groups" in model.cfg:
            del model.cfg.optim_param_groups
    train_cfg = OmegaConf.create({
        "use_lhotse": True, "manifest_filepath": str(train_man), "sample_rate": ftlib.SR,
        "shuffle": True, "num_workers": a.num_workers, "pin_memory": True, "seed": a.seed,
        "batch_size": None, "batch_duration": a.batch_duration, "use_bucketing": True, "num_buckets": a.num_buckets,
        "bucket_buffer_size": 5000, "shuffle_buffer_size": 2000, "min_duration": 0.3, "max_duration": 30.0,
        "text_field": "text", "lang_field": "target_lang",
    })

    # validation subsets: deterministic stride over each frozen eval set
    val_sets = {}
    for name in ("eval_fleurs_it", "eval_mls_it"):
        rows = ftlib.read_manifest(man / f"{name}.json")
        stride = max(1, len(rows) // a.val_n)
        sub = rows[::stride][: a.val_n]
        ftlib.write_manifest(out / f"val_{name}.json", sub)
        val_sets[name] = out / f"val_{name}.json"

    state = {"audio_s": 0.0, "samples": 0, "steps": 0, "eval_s": 0.0, "loss": [], "val": [],
             "t_start": None, "t_steady": None, "audio_steady": 0.0, "samples_steady": 0, "eval_steady": 0.0}
    steady_after = min(20, max_steps // 5)

    class Meter(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            state["t_start"] = time.time()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            for m in frozen:
                m.eval()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch is None:
                return
            state["audio_s"] += float(batch.audio_lens.sum().item()) / ftlib.SR
            state["samples"] += int(batch.audio.shape[0])
            state["steps"] += 1
            step = trainer.global_step
            if state["steps"] == steady_after:
                torch.cuda.synchronize()
                state.update(t_steady=time.time(), audio_steady=state["audio_s"], samples_steady=state["samples"],
                             eval_steady=state["eval_s"])
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if step % 10 == 0:
                state["loss"].append([step, round(float(loss), 4)])
            if step % 50 == 0:
                el = time.time() - state["t_start"] - state["eval_s"]
                print(f"  step {step}/{max_steps} loss {float(loss):.3f} audio {state['audio_s'] / 3600:.2f} h "
                      f"{state['audio_s'] / max(el, 1e-9):.0f}x RT, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                      flush=True)
            if a.val_every and step % a.val_every == 0 and step < max_steps:
                t = time.time()
                r = ftlib.eval_sets(pl_module, val_sets, "it", pnc="no", batch_size=32)
                for m in frozen:
                    m.eval()
                state["eval_s"] += time.time() - t
                state["val"].append({"step": step, "audio_h": round(state["audio_s"] / 3600, 3),
                                     **{k: v["wer"] for k, v in r.items() if isinstance(v, dict)}})
                print(f"  VAL step {step}: {state['val'][-1]}", flush=True)

    callbacks = [Meter()]
    gain_cb = None
    if a.gain_aug:
        gain_cb = ftlib.make_gain_aug(a.gain_lo_db, a.gain_hi_db, a.gain_prob, seed=a.seed)
        callbacks.insert(0, gain_cb)
        print(f"  gain augmentation ON: {a.gain_lo_db}..{a.gain_hi_db} dB, p={a.gain_prob}", flush=True)
    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision="bf16-mixed", max_steps=max_steps, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0, num_sanity_val_steps=0,
        log_every_n_steps=200, gradient_clip_val=1.0, use_distributed_sampler=False, accumulate_grad_batches=1,
        callbacks=callbacks,
    )
    model.set_trainer(trainer)
    model.setup_training_data(train_cfg)

    model = model.cuda()
    en_before = ftlib.eval_sets(model, {"fleurs_en": man / "en_sanity.json"}, "en", pnc="no", batch_size=32)
    val_before = ftlib.eval_sets(model, val_sets, "it", pnc="no", batch_size=32)
    state["val"].append({"step": 0, "audio_h": 0.0, **{k: v["wer"] for k, v in val_before.items() if isinstance(v, dict)}})
    levels = ftlib.parse_levels(a.levels)
    lv_before = {"it_val": ftlib.level_sweep(model, val_sets, "it", levels),
                 "en": ftlib.level_sweep(model, {"fleurs_en": man / "en_sanity.json"}, "en", levels)} if levels else None

    trainer.fit(model)
    torch.cuda.synchronize()
    t_end = time.time()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    train_wall = t_end - state["t_start"] - state["eval_s"]
    steady_wall = (t_end - state["t_steady"] - (state["eval_s"] - state["eval_steady"])) if state["t_steady"] else None

    model.save_to(str(out / "final.nemo"))
    model = model.cuda().eval()
    final = None
    if not a.no_final_eval:
        sets = {"fleurs_it": man / "eval_fleurs_it.json", "mls_it": man / "eval_mls_it.json"}
        if (man / "eval_cv_it.json").exists():
            sets["cv_it"] = man / "eval_cv_it.json"
        final = ftlib.eval_sets(model, sets, "it", pnc="no", batch_size=32, dump_dir=out / "final_eval")
    en_after = ftlib.eval_sets(model, {"fleurs_en": man / "en_sanity.json"}, "en", pnc="no", batch_size=32,
                               dump_dir=out / "final_eval")
    lv_after = {"it_val": ftlib.level_sweep(model, val_sets, "it", levels, dump_dir=out / "final_eval"),
                "en": ftlib.level_sweep(model, {"fleurs_en": man / "en_sanity.json"}, "en", levels)} if levels else None

    total_s = time.time() - T_PROC
    epochs_done = state["audio_s"] / subset_s
    rate = a.rate_usd_h
    m = {
        "tag": tag, "arm": a.arm, "top_n": a.top_n if a.arm == "B" else None, "subset": a.subset,
        "subset_hours_exact": round(subset_s / 3600, 4), "init": a.init, "env": ftlib.env_info(),
        "hparams": {k: v for k, v in vars(a).items() if k not in ("tag",)}, "max_steps": max_steps,
        "warmup_steps": warmup, **counts,
        "steps_done": state["steps"], "samples": state["samples"], "audio_hours_processed": round(state["audio_s"] / 3600, 4),
        "equiv_epochs": round(epochs_done, 3),
        "train_wall_s": round(train_wall, 1), "inloop_val_wall_s": round(state["eval_s"], 1),
        "audio_h_per_gpu_h": round(state["audio_s"] / train_wall, 1),
        "audio_h_per_gpu_h_steady": round((state["audio_s"] - state["audio_steady"]) / steady_wall, 1) if steady_wall else None,
        "samples_per_s": round(state["samples"] / train_wall, 2),
        "samples_per_s_steady": round((state["samples"] - state["samples_steady"]) / steady_wall, 2) if steady_wall else None,
        "epoch_wall_s": round(train_wall / epochs_done, 1) if epochs_done else None,
        "peak_alloc_gb": round(peak_alloc, 2), "peak_reserved_gb": round(peak_res, 2),
        "gpu_hours_train": round(train_wall / 3600, 4), "gpu_hours_total": round(total_s / 3600, 4),
        "rate_usd_h": rate,
        "cost_usd_train": round(rate * train_wall / 3600, 4), "cost_usd_total": round(rate * total_s / 3600, 4),
        "cost_usd_per_epoch": round(rate * train_wall / 3600 / epochs_done, 4) if epochs_done else None,
        "loss_curve": state["loss"], "val_curve": state["val"],
        "final_it": final, "en_before": en_before, "en_after": en_after,
        "preprocessor": pre, "gain_aug": gain_cb.summary() if gain_cb else None,
        "level_sweep": {"levels_peak_dbfs": levels, "before": lv_before, "after": lv_after,
                        "note": "val subsets (it) and EN clips re-levelled to a fixed peak and re-quantised to PCM16"},
        "en_wer_delta": round(en_after["fleurs_en"]["wer"] - en_before["fleurs_en"]["wer"], 3),
        "note": "validation subsets are drawn from the frozen eval (smoke test, no checkpoint selection: "
                "final = last step). GPU-hours = wall time on 1 GPU.",
    }
    ftlib.write_json(out / "metrics.json", m)
    with open(FT / "runs" / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({k: m[k] for k in ("tag", "subset_hours_exact", "steps_done", "equiv_epochs",
                                              "audio_h_per_gpu_h", "samples_per_s", "peak_alloc_gb", "peak_reserved_gb",
                                              "epoch_wall_s", "gpu_hours_total", "cost_usd_total", "en_wer_delta")}
                           | {"fleurs_it_wer": final["fleurs_it"]["wer"] if final else None,
                              "mls_it_wer": final["mls_it"]["wer"] if final else None}) + "\n")
    print(f"== {tag} done: {json.dumps({k: m[k] for k in ('audio_h_per_gpu_h', 'samples_per_s', 'peak_alloc_gb', 'peak_reserved_gb', 'epoch_wall_s', 'gpu_hours_total', 'cost_usd_total', 'en_wer_delta')})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
