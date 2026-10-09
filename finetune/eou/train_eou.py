#!/usr/bin/env python3
"""EOU 120M, combined language + <EOU> fine-tune (stage 2 machinery). EXPERIMENTAL.

KNOWN-BAD CONFIGURATION, kept to reproduce and ablate it: new Italian SPE
(tokenizer_eou.py) + fresh or stock decoder/joint + --lr 1e-3 + --eou-weight
0.9 (90 % padded with 3-6 s of zeros, 10 % plain) + FastEmit 0.03 COLLAPSES
TO BLANK (all four 2026-10-09 arms: WER ~100, empty greedy output, beam returns
one sentence for every input). There are therefore no defaults for --lr and
--eou-weight: every run states them. Do not use this as an Italian recipe; the
plan is plain_asr.py (stage 1, stock tokenizer) first, then a gentle EOU
curriculum (finetune/README.md).

    FT_ROOT=/root/ft $PY train_eou.py --stock /root/ft/models/parakeet_realtime_eou_120m-v1.nemo \
        --tokenizer /root/ft/models/tok-eou-it --subset 5h --epochs 52 --lr 1e-3 --eou-weight 0.9 --save-ckpt 1
    # ^ the exact collapsed "cold" arm of 2026-10-09 (eou-it-5h-e52)

Recipe (NeMo v3.0.0 facts with file:line: finetune/eou/README.md):
- model: stock config, `tokenizer.dir` -> the Italian SPE (+<EOU>,<EOB>), built as
  NeMo's EncDecRNNTBPEEOUModel (the EOU recipe's class: its lhotse dataset appends
  <EOU> to the text and pads the audio). Encoder + preprocessor weights copied from
  the stock .nemo; decoder + joint fresh (docs: init_from_nemo_model_exclude
  [decoder, joint] for a new vocabulary). Loss built from the stock cfg, so FastEmit
  0.03 is kept (change_vocabulary would drop it). Streaming keys asserted unchanged.
- data: train_<subset>.json (MLS it) -> rows trimmed to the voiced span by an energy
  rule (stand-in for align_eou.py), text normalised like the tokenizer pool. Blend
  0.9 EOU group (random zero padding: p 0.99, post-pad >= 3 s, <= 6 s, total <= 40 s,
  white noise -90..-46 dB) + 0.1 plain group (same rows, no padding).
- random gain -30..+6 dB p0.8 on the padded batch (ftlib.make_gain_aug), bf16-mixed,
  lhotse batch by duration, AdamW cosine, encoder at --lr x --enc-lr-scale.
- output: final.nemo saved as plain EncDecRNNTBPEModel (like the stock file), metrics.json
  with the Canary-kit economics fields, IT WER/CER/empty/EOU-rate at native and
  -3/-20/-40 dBFS peak, EN (FLEURS en 100) for information, and a cache-aware
  streaming check (conformer_stream_step) against offline on a few clips.
`--dry-run`: build the manifests + print the plan and the resolved config, no
NeMo/torch (needs FT_ROOT/manifests/train_<subset>.json and the audio for the trim).
Provenance: <run>/run.json (finetune/common/runmeta.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "common")]
import ftlib  # noqa: E402
import runmeta  # noqa: E402
from tokenizer_eou import check_special_ids, norm_text  # noqa: E402
from trim import voiced_span  # noqa: E402

FT = ftlib.FT
T_PROC = time.time()
SR = 16000
STREAM_KEYS = ("att_context_size", "att_context_style", "conv_context_size", "causal_downsampling",
               "subsampling_factor", "n_layers", "d_model")
EOU_RE = re.compile(r"<EO[UB]>")


# --------------------------------------------------------------------------- data
# voiced_span: finetune/common/trim.py


def build_train_manifest(subset, out_dir, trim=True):
    rows = ftlib.read_manifest(FT / "manifests" / f"train_{subset}.json")
    out, cut_s, empty = [], 0.0, 0
    for r in rows:
        t = norm_text(r["text"])
        if not t:
            empty += 1
            continue
        off, dur = voiced_span(r["audio_filepath"]) if trim else (0.0, float(r["duration"]))
        cut_s += float(r["duration"]) - dur
        out.append({"audio_filepath": r["audio_filepath"], "offset": off, "duration": dur,
                    "sampling_rate": SR, "text": t, "utt_id": r.get("utt_id")})
    p = out_dir / f"train_{subset}_eou.json"
    ftlib.write_manifest(p, out)
    info = {"rows": len(out), "dropped_empty_text": empty, "hours_after_trim": round(sum(r["duration"] for r in out) / 3600, 4),
            "trimmed_silence_s": round(cut_s, 1), "trim": "energy: frames within 35 dB of the loudest 25 ms frame, +-0.10 s" if trim else None}
    return p, info


# --------------------------------------------------------------------------- model
def build_model(stock_path, tok_dir, trainer=None, warm_dec_joint=False):
    from omegaconf import OmegaConf, open_dict

    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from nemo.collections.asr.models.asr_eou_models import EncDecRNNTBPEEOUModel

    stock = EncDecRNNTBPEModel.restore_from(str(stock_path), map_location="cpu")
    cfg = copy.deepcopy(stock.cfg)
    stream_before = {k: OmegaConf.to_container(cfg.encoder, resolve=True).get(k) for k in STREAM_KEYS}
    with open_dict(cfg):
        cfg.tokenizer.dir = str(tok_dir)
        cfg.tokenizer.type = "bpe"
        # The stock cfg points these at nemo: artifacts of the OLD tokenizer; the
        # monolingual setup registers model_path as an artifact and fails on a
        # missing key (TypeError on None), so point all three at the new files.
        cfg.tokenizer.model_path = str(Path(tok_dir) / "tokenizer.model")
        cfg.tokenizer.vocab_path = str(Path(tok_dir) / "vocab.txt")
        cfg.tokenizer.spe_tokenizer_vocab = str(Path(tok_dir) / "tokenizer.vocab")
        for k in ("train_ds", "validation_ds", "test_ds"):
            cfg.pop(k, None)
        cfg.target = "nemo.collections.asr.models.asr_eou_models.EncDecRNNTBPEEOUModel"
    model = EncDecRNNTBPEEOUModel(cfg=cfg, trainer=trainer)
    sd_stock = stock.state_dict()
    # NVIDIA guidance for a new vocabulary: keep encoder+preprocessor, reinitialise
    # decoder+joint. --warm-dec-joint is an EXPERIMENTAL arm: the new vocabulary has
    # the SAME size and the same <EOU>/<EOB>/blank ids, so the stock prediction net
    # and joint are kept too (text rows change meaning and are retrained; blank and
    # <EOU> keep their trained behaviour). Measured motivation: the cold arm learned
    # (loss 44.9 on text+<EOU> vs 192.9 text-only) yet greedy decoding emitted nothing.
    pre = ("encoder.", "preprocessor.") + (("decoder.", "joint.") if warm_dec_joint else ())
    keep = {k: v for k, v in sd_stock.items() if k.startswith(pre)}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    bad = [k for k in missing if k.startswith(pre)]
    assert not bad and not unexpected, f"encoder load: missing {bad[:5]} unexpected {unexpected[:5]}"
    stream_after = {k: OmegaConf.to_container(model.cfg.encoder, resolve=True).get(k) for k in STREAM_KEYS}
    assert stream_after == stream_before, (stream_before, stream_after)
    tk = model.tokenizer
    V = tk.vocab_size
    ids = check_special_ids(V, tk.token_to_id("<EOU>"), tk.token_to_id("<EOB>"),
                            blank_id=model.joint.num_classes_with_blank - 1)
    info = {"stock_class": type(stock).__name__, "streaming_cfg": stream_after, "ids": ids,
            "loss_cfg": OmegaConf.to_container(model.cfg.loss, resolve=True),
            "preprocessor": ftlib.preprocessor_info(model),
            "copied_tensors": len(keep), "fresh": "decoder.*, joint.*"}
    del stock
    return model, info


def save_plain(model, path):
    """Save as EncDecRNNTBPEModel (the stock .nemo's class) with the stock decoding cfg."""
    from omegaconf import open_dict

    from nemo.collections.asr.models import EncDecRNNTBPEModel

    cfg = copy.deepcopy(model.cfg)
    with open_dict(cfg):
        cfg.target = "nemo.collections.asr.models.EncDecRNNTBPEModel"
        cfg.decoding.preserve_alignments = False   # undo ASREOUModelMixin._patch_decoding_cfg
        cfg.decoding.compute_timestamps = False
        for k in ("train_ds", "validation_ds", "test_ds"):
            cfg.pop(k, None)
    plain = EncDecRNNTBPEModel(cfg=cfg)
    plain.load_state_dict(model.state_dict(), strict=True)
    plain.save_to(str(path))
    return plain


# --------------------------------------------------------------------------- eval
def decode_rnnt(model, rows, level=None, batch_size=32, min_dur=0.5):
    import numpy as np
    import torch

    dev = next(model.parameters()).device
    order = sorted(range(len(rows)), key=lambda i: -float(rows[i].get("duration", 0)))
    hyps = [None] * len(rows)
    was = model.training
    model.eval()
    for b in range(0, len(order), batch_size):
        idx = order[b: b + batch_size]
        wavs = [ftlib._load_audio(rows[i]["audio_filepath"], level) for i in idx]
        wavs = [np.pad(w, (0, max(0, int(min_dur * SR) - len(w)))) for w in wavs]
        lens = torch.tensor([len(w) for w in wavs], dtype=torch.long, device=dev)
        x = torch.zeros(len(wavs), int(lens.max()))
        for k, w in enumerate(wavs):
            x[k, : len(w)] = torch.from_numpy(w)
        with torch.inference_mode():
            enc, enc_len = model.forward(input_signal=x.to(dev), input_signal_length=lens)
            out = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                                 return_hypotheses=False)
        out = out[0] if isinstance(out, tuple) else out
        for i, h in zip(idx, out):
            hyps[i] = h.text if hasattr(h, "text") else str(h)
    if was:
        model.train()
    return hyps


def eval_sets(model, sets, lang, level=None, dump_dir=None, limit=None):
    res = {}
    for name, man in sets.items():
        rows = ftlib.read_manifest(man)[:limit] if limit else ftlib.read_manifest(man)
        t = time.time()
        raw = decode_rnnt(model, rows, level)
        dt = time.time() - t
        eou_n = sum(bool(EOU_RE.search(h)) for h in raw)
        hyps = [re.sub(r"\s+", " ", EOU_RE.sub(" ", h)).strip() for h in raw]
        sc, per = ftlib.score([(r.get("ref", r["text"]), h) for r, h in zip(rows, hyps)], lang)
        sc.update({"level_peak_dbfs": level, "eou_emitted_rate": round(eou_n / max(1, len(rows)), 4),
                   "decode_wall_s": round(dt, 1), "audio_h": round(sum(float(r["duration"]) for r in rows) / 3600, 4)})
        res[name] = sc
        if dump_dir:
            sfx = "" if level is None else f"_peak{int(level)}"
            ftlib.write_manifest(Path(dump_dir) / f"hyp_{name}{sfx}.jsonl",
                                 [{"audio_filepath": r["audio_filepath"], "ref": r.get("ref", r["text"]), "hyp_raw": h, **u}
                                  for r, h, u in zip(rows, raw, per)])
        print(f"  [{name}{'' if level is None else f' @{level:g} dBFS'}] n={sc['n_utts']} WER {sc['wer']:.2f} "
              f"CER {sc['cer']:.2f} empty={sc['empty_hyp']} eou={sc['eou_emitted_rate']:.2f} ({dt:.0f}s)", flush=True)
    return res


def streaming_check(model, rows, tmp_dir, tail_s=2.0):
    """Cache-aware streaming (NeMo's speech_to_text_cache_aware_streaming_infer.py
    loop) vs offline on the same audio + tail_s of zeros; <EOU> must appear."""
    import numpy as np
    import torch

    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    model.eval()
    out = []
    import soundfile as sf

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    for r in rows:
        a = ftlib._load_audio(r["audio_filepath"])
        a = np.concatenate([a, np.zeros(int(tail_s * SR), dtype=np.float32)])
        wav = Path(tmp_dir) / Path(r["audio_filepath"]).name
        sf.write(str(wav), a, SR, subtype="PCM_16")
        # offline reference on the SAME padded audio (full-context forward with the
        # chunked_limited mask = what the streaming encoder should reproduce)
        off = decode_rnnt(model, [{"audio_filepath": str(wav), "duration": len(a) / SR}])[0]
        try:
            buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
            buf.append_audio(a, stream_id=-1)
            c_ch, c_t, c_len = model.encoder.get_initial_cache_state(batch_size=1)
            prev_h = prev_out = None
            texts = None
            with torch.inference_mode():
                for step, (chunk, clen) in enumerate(iter(buf)):
                    drop = 0 if step == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded
                    (prev_out, texts, c_ch, c_t, c_len, prev_h) = model.conformer_stream_step(
                        processed_signal=chunk, processed_signal_length=clen, cache_last_channel=c_ch,
                        cache_last_time=c_t, cache_last_channel_len=c_len, keep_all_outputs=buf.is_buffer_empty(),
                        previous_hypotheses=prev_h, previous_pred_out=prev_out, drop_extra_pre_encoded=drop,
                        return_transcription=True)
            st = texts[0].text if hasattr(texts[0], "text") else str(texts[0])
            err = None
        except Exception as e:  # never lose the run over the check
            st, err = None, repr(e)[:300]
        out.append({"audio_filepath": r["audio_filepath"], "ref": r["text"], "offline": off, "stream": st,
                    "stream_eou": bool(st and EOU_RE.search(st)), "error": err,
                    "match_offline_text": (st is not None and EOU_RE.sub("", st).split() == EOU_RE.sub("", off).split())})
        print(f"  STREAM {Path(r['audio_filepath']).name}: offline={off!r} stream={st!r} err={err}", flush=True)
    ok = [o for o in out if o["error"] is None]
    return {"clips": out, "tail_s": tail_s, "n_ok": len(ok), "eou_rate": round(sum(o["stream_eou"] for o in ok) / max(1, len(ok)), 3),
            "text_match_rate": round(sum(o["match_offline_text"] for o in ok) / max(1, len(ok)), 3)}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", default=str(FT / "models" / "parakeet_realtime_eou_120m-v1.nemo"))
    ap.add_argument("--tokenizer", default=str(FT / "models" / "tok-eou-it"))
    ap.add_argument("--subset", required=True, choices=["5h", "20h", "40h"])
    ap.add_argument("--epochs", type=float, default=52.0, help="sets max_steps (on the subset's trimmed hours)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-duration", type=float, default=300.0, help="s of (unpadded) audio per batch")
    ap.add_argument("--num-buckets", type=int, default=8)
    ap.add_argument("--lr", type=float, required=True,
                    help="decoder + joint peak LR (no default: 1e-3 is part of the KNOWN-BAD 2026-10-09 config)")
    ap.add_argument("--enc-lr-scale", type=float, default=0.3)
    ap.add_argument("--freeze-encoder-layers", type=int, default=0, help="freeze subsampling + bottom N layers")
    ap.add_argument("--warmup-frac", type=float, default=0.08)
    ap.add_argument("--eou-weight", type=float, required=True,
                    help="share of padded EOU samples (no default: 0.9 is part of the KNOWN-BAD 2026-10-09 config)")
    ap.add_argument("--trim", type=int, default=1, choices=[0, 1])
    ap.add_argument("--gain-aug", type=int, default=1, choices=[0, 1])
    ap.add_argument("--levels", default="-3,-20,-40")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warm-dec-joint", type=int, default=int(os.environ.get("WARM_DEC_JOINT", "0") or 0), choices=[0, 1])
    ap.add_argument("--save-ckpt", type=int, default=int(os.environ.get("SAVE_CKPT", "0") or 0), choices=[0, 1])
    ap.add_argument("--rate-usd-h", type=float, default=float(os.environ.get("RATE_USD_H", "0") or 0))
    ap.add_argument("--stream-n", type=int, default=6)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--base-model-id", default="nvidia/parakeet_realtime_eou_120m-v1")
    ap.add_argument("--base-model-revision", default=None)
    ap.add_argument("--dry-run", action="store_true", help="manifests + plan only (no torch/NeMo)")
    a = ap.parse_args()
    runmeta.require_root()
    print("== EXPERIMENTAL: combined language+EOU fine-tune; the 2026-10-09 config (new SPE, lr 1e-3, "
          "eou-weight 0.9, FastEmit 0.03) collapses to blank. See finetune/README.md.", flush=True)

    tag = a.tag or f"eou-it-{a.subset}-e{a.epochs:g}"
    out = FT / "runs" / tag
    if (out / "metrics.json").exists():
        print(f"== skip {tag}: metrics.json exists")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    man = FT / "manifests"
    train_man, data_info = build_train_manifest(a.subset, out, trim=bool(a.trim))
    train_s = data_info["hours_after_trim"] * 3600
    max_steps = a.max_steps or max(50, math.ceil(a.epochs * train_s / a.batch_duration))
    warmup = max(10, int(a.warmup_frac * max_steps))
    print(f"== {tag}: data {data_info}; max_steps {max_steps} warmup {warmup} "
          f"(~{a.epochs * train_s / 3600:.0f} audio-h before padding)", flush=True)
    run = runmeta.build(
        "finetune/eou/train_eou.py", vars(a), probe_env=not a.dry_run,
        base_model={"id": a.base_model_id, "revision": a.base_model_revision, "path": a.stock,
                    "sha256": None if a.dry_run else runmeta.sha256_path(a.stock)},
        tokenizer={"path": a.tokenizer, "sha256": runmeta.sha256_path(Path(a.tokenizer) / "tokenizer.model"),
                   "note": "new SPE + <EOU>/<EOB> (tokenizer_eou.py)"},
        data={"dataset": "MLS-it train (canary/prepare_it.py), energy-trimmed", "subset": a.subset,
              "manifest": str(train_man), "hours": data_info["hours_after_trim"], "utterances": data_info["rows"],
              **data_info},
        seed=a.seed,
        optimizer={"name": "adamw", "betas": [0.9, 0.98], "weight_decay": 1e-3, "grad_clip": 1.0,
                   "precision": "bf16-mixed", "batch_duration_s": a.batch_duration, "num_buckets": a.num_buckets},
        lr_groups={"decoder+joint": a.lr, "encoder": a.lr * a.enc_lr_scale},
        scheduler={"name": "CosineAnnealing", "warmup_steps": warmup, "min_lr": a.lr * 0.01},
        fastemit_lambda="from the stock loss cfg (0.03 for parakeet_realtime_eou_120m-v1)",
        augmentation={"gain": {"lo_db": -30.0, "hi_db": 6.0, "prob": 0.8} if a.gain_aug else None,
                      "white_noise": {"prob": 0.9, "min_level": -90, "max_level": -46},
                      "padding": {"prob": a.eou_weight, "post_pad_s": [3.0, 6.0], "max_total_s": 40.0}},
        eou_plain_mix={"eou_share": a.eou_weight, "plain_share": round(1 - a.eou_weight, 4),
                       "note": "plain group still carries <EOU> (the dataset always appends it)"},
        freeze_policy={"freeze_encoder_layers": a.freeze_encoder_layers, "warm_dec_joint": bool(a.warm_dec_joint),
                       "enc_lr_scale": a.enc_lr_scale},
        max_steps=max_steps, epochs=a.epochs,
        notes="EXPERIMENTAL; the 2026-10-09 combination (lr 1e-3, eou_weight 0.9, new SPE) collapses to blank")
    runmeta.print_resolved(run)
    if a.dry_run:
        return 0
    runmeta.write(out / "run.json", run, status="started")

    import lightning.pytorch as pl
    import torch
    from omegaconf import OmegaConf, open_dict

    pl.seed_everything(a.seed)
    model, minfo = build_model(a.stock, a.tokenizer, warm_dec_joint=bool(a.warm_dec_joint))
    minfo["warm_dec_joint"] = bool(a.warm_dec_joint)
    print(f"  model: {json.dumps(minfo, default=str)}", flush=True)

    # trainable set
    for p in model.parameters():
        p.requires_grad_(True)
    frozen = []
    if a.freeze_encoder_layers > 0:
        enc = model.encoder
        assert a.freeze_encoder_layers <= len(enc.layers)
        frozen = [m for n, m in enc.named_children() if n != "layers"] + list(enc.layers[: a.freeze_encoder_layers])
        for m in frozen:
            for p in m.parameters():
                p.requires_grad_(False)
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open_dict(model.cfg):
        model.cfg.optim = OmegaConf.create({
            "name": "adamw", "lr": a.lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
            "sched": {"name": "CosineAnnealing", "warmup_steps": warmup, "min_lr": a.lr * 0.01}})
        model.cfg.optim_param_groups = OmegaConf.create({"encoder": {"lr": a.lr * a.enc_lr_scale}})
    common = {"sample_rate": SR, "shuffle": True, "num_workers": a.num_workers, "pin_memory": True, "seed": a.seed,
              "batch_size": None, "batch_duration": a.batch_duration, "use_bucketing": True, "num_buckets": a.num_buckets,
              "bucket_buffer_size": 5000, "shuffle_buffer_size": 2000, "min_duration": 0.3, "max_duration": 30.0,
              "drop_last": True, "ignore_eob_label": True, "use_lhotse": True,
              "window_stride": float(model.cfg.preprocessor.window_stride),
              "subsampling_factor": int(model.cfg.encoder.subsampling_factor),
              "augmentor": {"white_noise": {"prob": 0.9, "min_level": -90, "max_level": -46}}}
    # EOU group (padded) + plain group: one manifest, two lhotse groups. Padding is a
    # DATASET option, not per group, so the plain group gets its share through the
    # dataset's own per-sample draw: prob = eou_weight (README blend 0.9/0.1).
    train_cfg = OmegaConf.create({**common, "manifest_filepath": str(train_man),
                                  "random_padding": {"prob": a.eou_weight, "min_post_pad_duration": 3.0,
                                                     "min_pre_pad_duration": 0.0, "max_pad_duration": 6.0,
                                                     "max_total_duration": 40.0, "pad_distribution": "uniform",
                                                     "normal_mean": 0.5, "normal_std": 2.0,
                                                     "pre_pad_duration": 0.2, "post_pad_duration": 3.0}})

    val_sets = {}
    for name in ("eval_fleurs_it", "eval_mls_it"):
        rows = ftlib.read_manifest(man / f"{name}.json")
        sub = rows[:: max(1, len(rows) // a.val_n)][: a.val_n]
        ftlib.write_manifest(out / f"val_{name}.json", sub)
        val_sets[name] = out / f"val_{name}.json"

    state = {"audio_s": 0.0, "samples": 0, "steps": 0, "eval_s": 0.0, "loss": [], "val": [], "t_start": None}

    class Meter(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            state["t_start"] = time.time()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            for m in frozen:
                m.eval()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            state["audio_s"] += float(batch.audio_lengths.sum().item()) / SR  # padded audio, what the GPU saw
            state["samples"] += int(batch.audio_signal.shape[0])
            state["steps"] += 1
            step = trainer.global_step
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if step % 10 == 0:
                state["loss"].append([step, round(float(loss), 4)])
            if step % 50 == 0:
                el = time.time() - state["t_start"] - state["eval_s"]
                print(f"  step {step}/{max_steps} loss {float(loss):.3f} audio {state['audio_s'] / 3600:.2f} h "
                      f"{state['audio_s'] / max(el, 1e-9):.0f}x RT peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            if a.val_every and step % a.val_every == 0 and 0 < step < max_steps:
                t = time.time()
                r = eval_sets(pl_module, val_sets, "it")
                for m in frozen:
                    m.eval()
                state["eval_s"] += time.time() - t
                state["val"].append({"step": step, **{k: [v["wer"], v["eou_emitted_rate"]] for k, v in r.items()}})
                print(f"  VAL step {step}: {state['val'][-1]}", flush=True)

    callbacks = [Meter()]
    gain_cb = None
    if a.gain_aug:
        gain_cb = ftlib.make_gain_aug(-30.0, 6.0, 0.8, seed=a.seed, audio_attr="audio_signal")
        callbacks.insert(0, gain_cb)
    trainer = pl.Trainer(devices=1, accelerator="gpu", precision="bf16-mixed", max_steps=max_steps, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0,
                         num_sanity_val_steps=0, log_every_n_steps=100, gradient_clip_val=1.0,
                         use_distributed_sampler=False, callbacks=callbacks)
    model.set_trainer(trainer)
    model.setup_training_data(train_cfg)
    model = model.cuda()

    sets = {"fleurs_it": man / "eval_fleurs_it.json", "mls_it": man / "eval_mls_it.json"}
    en = {"fleurs_en": man / "en_sanity.json"}
    levels = ftlib.parse_levels(a.levels)
    trainer.fit(model)
    torch.cuda.synchronize()
    t_end = time.time()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    train_wall = t_end - state["t_start"] - state["eval_s"]

    ckpt = None
    if a.save_ckpt:
        try:
            trainer.save_checkpoint(str(out / "last.ckpt"))
            ckpt = {"path": str(out / "last.ckpt"), "size_gb": round((out / "last.ckpt").stat().st_size / 2**30, 3),
                    "global_step": trainer.global_step, "class": "EncDecRNNTBPEEOUModel"}
        except Exception as e:
            ckpt = {"error": repr(e)}
        print(f"  ckpt: {ckpt}", flush=True)
    plain = save_plain(model, out / "final.nemo").cuda().eval()
    del model
    torch.cuda.empty_cache()

    final = eval_sets(plain, sets, "it", dump_dir=out / "final_eval")
    en_after = eval_sets(plain, en, "en", dump_dir=out / "final_eval")
    lv = {f"{lv:g}": {"it": eval_sets(plain, sets, "it", level=lv), "en": eval_sets(plain, en, "en", level=lv)}
          for lv in levels}
    srows = ftlib.read_manifest(man / "eval_fleurs_it.json")[: a.stream_n // 2] + \
        ftlib.read_manifest(man / "eval_mls_it.json")[: a.stream_n - a.stream_n // 2]
    stream = streaming_check(plain, srows, out / "stream_wavs")

    total_s = time.time() - T_PROC
    ep = state["audio_s"] / train_s if train_s else 0
    rate = a.rate_usd_h
    m = {
        "tag": tag, "model": "parakeet_realtime_eou_120m-v1 -> it", "subset": a.subset, "env": ftlib.env_info(),
        "hparams": vars(a), "data": data_info, "model_info": minfo, "max_steps": max_steps, "warmup_steps": warmup,
        "params_total": n_all, "params_trainable": n_tr,
        "steps_done": state["steps"], "samples": state["samples"],
        "audio_hours_processed": round(state["audio_s"] / 3600, 4),
        "audio_hours_note": "padded audio (incl. synthetic silence) actually fed to the GPU",
        "equiv_epochs_padded": round(ep, 3),
        "train_wall_s": round(train_wall, 1), "inloop_val_wall_s": round(state["eval_s"], 1),
        "audio_h_per_gpu_h": round(state["audio_s"] / train_wall, 1) if train_wall else None,
        "samples_per_s": round(state["samples"] / train_wall, 2) if train_wall else None,
        "epoch_wall_s": round(train_wall / ep, 1) if ep else None,
        "peak_alloc_gb": round(peak_alloc, 2), "peak_reserved_gb": round(peak_res, 2),
        "gpu_hours_train": round(train_wall / 3600, 4), "gpu_hours_total": round(total_s / 3600, 4),
        "rate_usd_h": rate, "cost_usd_train": round(rate * train_wall / 3600, 4),
        "cost_usd_total": round(rate * total_s / 3600, 4),
        "loss_curve": state["loss"], "val_curve": state["val"],
        "final_it": final, "en_after": en_after, "level_sweep": lv,
        "gain_aug": gain_cb.summary() if gain_cb else None, "checkpoint": ckpt, "streaming_check": stream,
        "note": "no stock-model IT baseline here (stock = English vocab); EN for information only. "
                "final = last step, no checkpoint selection.",
    }
    ftlib.write_json(out / "metrics.json", m)
    runmeta.write(out / "run.json", run, status="done", outputs={
        "final_nemo": str(out / "final.nemo"), "metrics": str(out / "metrics.json"), "checkpoint": ckpt,
        "fleurs_it_wer": final["fleurs_it"]["wer"], "mls_it_wer": final["mls_it"]["wer"],
        "eou_rate_fleurs": final["fleurs_it"]["eou_emitted_rate"], "stream_eou_rate": stream["eou_rate"]})
    with open(FT / "runs" / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"tag": tag, "fleurs_it_wer": final["fleurs_it"]["wer"], "mls_it_wer": final["mls_it"]["wer"],
                            "eou_rate_fleurs": final["fleurs_it"]["eou_emitted_rate"],
                            "stream_eou_rate": stream["eou_rate"], "stream_text_match": stream["text_match_rate"],
                            **{k: m[k] for k in ("audio_h_per_gpu_h", "peak_alloc_gb", "gpu_hours_total", "cost_usd_total")}}) + "\n")
    print(f"== {tag} done: FLEURS-it WER {final['fleurs_it']['wer']} MLS-it WER {final['mls_it']['wer']} "
          f"EN WER {en_after['fleurs_en']['wer']} stream eou {stream['eou_rate']} match {stream['text_match_rate']} "
          f"{json.dumps({k: m[k] for k in ('audio_h_per_gpu_h', 'samples_per_s', 'peak_alloc_gb', 'gpu_hours_total', 'cost_usd_total')})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
