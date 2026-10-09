#!/usr/bin/env python3
"""EOU 120M, stage 1 of 2: plain ASR adaptation of the STOCK model, minimal loop.

EXPERIMENTAL (no working Italian EOU model exists yet; see finetune/README.md).
No Lightning, no lhotse, no EOU machinery: stock model, STOCK tokenizer (text
de-accented: e.g. è -> e, since the English SPE maps accented vowels to <unk>),
stock decoder/joint, FastEmit 0, no padding, no gain. Streaming config untouched.

    $PY plain_asr.py --manifest /root/ft/manifests/train_5h.json --out /root/ft/runs \\
        --tag a0 --subset-utts 32 --max-steps 400 --lr 3e-4 --freeze-enc 1 \\
        --eval-every 100 [--val /root/ft/manifests/eval_mls_it.json]

Micro-overfit / debug mode (step 1 of the ablation ladder: a healthy RNNT must
memorise a handful of utterances and greedy must return sensible text):

    $PY plain_asr.py --manifest .../train_5h.json --out /root/ft/runs --tag mo32 --micro-overfit
    # = --subset-utts 32 --max-steps 1000 --eval-steps 100,250,500,750,1000 --lr 1e-4
    #   --freeze-enc 1 --train-eval-n 32 (any of them can still be overridden)

At every eval step (--eval-steps list, else every --eval-every, plus the last
step) on the first --train-eval-n TRAIN clips: loss, greedy WER/CER, beam
(--beam-size, 0 = off) WER/CER, greedy and beam hypotheses, empty rate, <EOU>
rate, blank margin (max/mean), non-blank frame %, audio dependence (distinct
hypotheses / clips) and a silence probe (2 s of zeros should decode empty).
With --val: WER/CER, empty, <EOU> rate, non-blank % on --val-n clips; the best
val WER saves final.nemo + best.ckpt. Every eval step saves last.ckpt (torch
state {model, opt, sched, step, args}); --resume <ckpt> continues from it.
Per-step hypotheses: <run>/evals/step_<n>.jsonl. Provenance: <run>/run.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "common")]
import runmeta  # noqa: E402

MICRO = {"subset_utts": 32, "max_steps": 1000, "eval_steps": "100,250,500,750,1000", "lr": 1e-4, "freeze_enc": 1,
         "train_eval_n": 32}
PLAIN = {"subset_utts": 0, "max_steps": 400, "eval_steps": "", "lr": 3e-4, "freeze_enc": 1, "train_eval_n": 8}


def norm(t):
    """Stock-tokenizer target text: lower case, accents stripped, [a-z' ] only."""
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_cer(refs, hyps):
    """Corpus WER/CER in %, accent-insensitive (texts already normalised by norm())."""
    we = sum(lev(r.split(), h.split()) for r, h in zip(refs, hyps))
    wn = sum(len(r.split()) for r in refs)
    ce = sum(lev(r, h) for r, h in zip(refs, hyps))
    cn = sum(len(r) for r in refs)
    return 100 * we / max(1, wn), 100 * ce / max(1, cn)


def eval_steps_of(a):
    if a.eval_steps:
        return sorted({int(x) for x in str(a.eval_steps).split(",") if x.strip()} | {a.max_steps})
    if a.eval_every:
        return sorted(set(range(a.eval_every, a.max_steps + 1, a.eval_every)) | {a.max_steps})
    return [a.max_steps]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", default="/root/ft/models/parakeet_realtime_eou_120m-v1.nemo")
    ap.add_argument("--base-model-id", default="nvidia/parakeet_realtime_eou_120m-v1")
    ap.add_argument("--base-model-revision", default=None)
    ap.add_argument("--manifest", required=True, help="NeMo manifest (audio_filepath, duration, text)")
    ap.add_argument("--out", required=True, help="runs root; the run goes to <out>/plain-<tag>")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--micro-overfit", action="store_true", help="debug preset, see the module doc")
    ap.add_argument("--subset-utts", "--n", dest="subset_utts", type=int, default=None,
                    help="first N utterances (0 = all)")
    ap.add_argument("--max-steps", "--steps", dest="max_steps", type=int, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="decoder + joint peak LR (plain default 3e-4)")
    ap.add_argument("--enc-lr", type=float, default=0.0, help="lr for unfrozen encoder blocks (0 = lr x0.1)")
    ap.add_argument("--freeze-enc", type=int, default=None, help="1 = whole encoder frozen (default 1)")
    ap.add_argument("--unfreeze-top", type=int, default=0, help="with --freeze-enc 1: unfreeze the top K layers")
    ap.add_argument("--fastemit", type=float, default=0.0, help="FastEmit lambda of the RNNT loss (stage 1: 0)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-steps", default=None, help="comma list of steps; overrides --eval-every")
    ap.add_argument("--train-eval-n", type=int, default=None, help="TRAIN clips decoded at each eval")
    ap.add_argument("--beam-size", type=int, default=4, help="beam search on the train-eval clips too (0 = off)")
    ap.add_argument("--val", default="")
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="last.ckpt/best.ckpt of a previous run (torch state)")
    ap.add_argument("--dry-run", action="store_true", help="resolve + print the config; no torch, nothing written")
    a = ap.parse_args(argv)
    preset = MICRO if a.micro_overfit else PLAIN
    for k, v in preset.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def load_rows(path, max_dur, n):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("duration", 0) <= max_dur and norm(r["text"])]
    return rows[:n] if n else rows


def build_run(a, rows, val, tok_sha=None, groups_lr=None):
    return runmeta.build(
        "finetune/eou/plain_asr.py", vars(a), probe_env=not a.dry_run,
        base_model={"id": a.base_model_id, "revision": a.base_model_revision, "path": a.stock,
                    "sha256": None if a.dry_run else runmeta.sha256_path(a.stock)},
        tokenizer={"path": f"{a.stock}::tokenizer.model (STOCK)", "sha256": tok_sha,
                   "note": "stock SPE 1024 + <EOU>/<EOB>; targets de-accented [a-z' ]"},
        data={"dataset": a.manifest, "subset": f"first {a.subset_utts} utts" if a.subset_utts else "all",
              "manifest": a.manifest, "utterances": len(rows),
              "hours": round(sum(float(r["duration"]) for r in rows) / 3600, 4), "max_dur": a.max_dur,
              "val": a.val or None, "val_utterances": len(val)},
        seed=a.seed,
        optimizer={"name": "adamw", "betas": [0.9, 0.98], "weight_decay": 1e-3, "grad_clip": 1.0,
                   "precision": "bf16 autocast (loss fp32)", "batch_size": a.bs},
        lr_groups=groups_lr or {"decoder+joint": a.lr},
        scheduler={"name": "linear warmup + cosine to 1 %", "warmup_steps": max(10, a.max_steps // 20)},
        fastemit_lambda=a.fastemit,
        augmentation={"spec_augment": "model default (train mode)", "gain": None, "padding": None},
        eou_plain_mix={"eou_share": 0.0, "plain_share": 1.0, "note": "stage 1: plain ASR, no <EOU> targets"},
        freeze_policy={"freeze_enc": a.freeze_enc, "unfreeze_top": a.unfreeze_top, "preprocessor": "frozen"},
        max_steps=a.max_steps, epochs=None,
        notes="EXPERIMENTAL EOU stage 1" + (" (micro-overfit preset)" if a.micro_overfit else ""))


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out) / f"plain-{a.tag}"
    rows = load_rows(a.manifest, a.max_dur, a.subset_utts)
    val = []
    if a.val:
        vall = [json.loads(line) for line in open(a.val, encoding="utf-8") if line.strip()]
        val = vall[:: max(1, len(vall) // a.val_n)][: a.val_n]
    steps_eval = eval_steps_of(a)
    if a.dry_run:
        runmeta.print_resolved(build_run(a, rows, val))
        print(f"== dry run: plain-{a.tag} -> {out}; eval at steps {steps_eval} (nothing written)")
        return 0

    import hashlib

    import numpy as np
    import soundfile as sf
    import torch

    import rnnt_diag as D
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from tokenizer_eou import read_stock_tokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evals").mkdir(exist_ok=True)
    dev = "cuda"
    m = EncDecRNNTBPEModel.restore_from(a.stock, map_location=dev)
    m.joint._fuse_loss_wer = False
    tk = m.tokenizer
    V = tk.vocab_size  # 1026 incl. <EOU>/<EOB>; blank = V
    eou_id, eob_id = D.special_id(tk, "<EOU>"), D.special_id(tk, "<EOB>")
    loss_fn = RNNTLoss(num_classes=V, reduction="mean_batch", loss_name="warprnnt_numba",
                       loss_kwargs={"fastemit_lambda": a.fastemit})
    for r in rows:
        r["ids"] = tk.text_to_ids(norm(r["text"]))
    print(f"== plain-{a.tag}: train {len(rows)} utts {sum(r['duration'] for r in rows) / 3600:.3f} h, val {len(val)}, "
          f"<EOU> {eou_id} <EOB> {eob_id}, eval steps {steps_eval}", flush=True)

    # parameters (same groups as the 2026-10-09 loop)
    for p in m.parameters():
        p.requires_grad = True
    groups = [{"params": list(m.decoder.parameters()) + list(m.joint.parameters()), "lr": a.lr}]
    lr_groups = {"decoder+joint": a.lr}
    if a.freeze_enc:
        for p in m.encoder.parameters():
            p.requires_grad = False
        if a.unfreeze_top:
            top = list(m.encoder.layers)[-a.unfreeze_top:]
            ps = [p for layer in top for p in layer.parameters()]
            for p in ps:
                p.requires_grad = True
            groups.append({"params": ps, "lr": a.enc_lr or a.lr * 0.1})
            lr_groups[f"encoder top {a.unfreeze_top}"] = a.enc_lr or a.lr * 0.1
    else:
        groups.append({"params": list(m.encoder.parameters()), "lr": a.enc_lr or a.lr * 0.1})
        lr_groups["encoder"] = a.enc_lr or a.lr * 0.1
    for p in m.preprocessor.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=1e-3)
    warm = max(10, a.max_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.max_steps))) * 0.99 + 0.01)
    n_tr = sum(p.numel() for g in groups for p in g["params"])
    print("== trainable", n_tr / 1e6, "M", flush=True)

    try:
        tok_sha = hashlib.sha256(read_stock_tokenizer(Path(a.stock))).hexdigest()
    except Exception as e:  # never block training on provenance
        tok_sha = None
        print(f"  WARNING tokenizer sha256: {e!r}", flush=True)
    run = build_run(a, rows, val, tok_sha=tok_sha, groups_lr=lr_groups)
    run["config"]["trainable_params"] = n_tr
    runmeta.print_resolved(run)
    runmeta.write(out / "run.json", run, status="started")

    step = 0
    if a.resume:
        st = torch.load(a.resume, map_location=dev, weights_only=False)
        m.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = int(st["step"])
        print(f"== resumed {a.resume} at step {step}", flush=True)

    def audio(r):
        x, sr = sf.read(r["audio_filepath"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        return x

    def batch(rs, wavs=None):
        xs = wavs if wavs is not None else [audio(r) for r in rs]
        L = max(len(x) for x in xs)
        X = torch.zeros(len(xs), L)
        for i, x in enumerate(xs):
            X[i, : len(x)] = torch.from_numpy(x)
        lens = torch.tensor([len(x) for x in xs])
        ids = [r.get("ids") or [0] for r in rs]
        U = max(len(t) for t in ids)
        Y = torch.zeros(len(rs), U, dtype=torch.long)
        for i, t in enumerate(ids):
            Y[i, : len(t)] = torch.tensor(t)
        ylens = torch.tensor([len(t) for t in ids])
        return X.to(dev), lens.to(dev), Y.to(dev), ylens.to(dev)

    def encode(X, lens):
        f, fl = m.preprocessor(input_signal=X, length=lens)
        if m.training and m.spec_augmentation is not None:
            f = m.spec_augmentation(input_spec=f, length=fl)
        return m.encoder(audio_signal=f, length=fl)

    def to_text(ys):
        return tk.ids_to_text([t for t in ys if t < V and t not in (eou_id, eob_id)])

    @torch.no_grad()
    def decode(rs, bs=16, wavs=None, margins=True):
        m.eval()
        hyps, eous, diag = [], 0, []
        for i in range(0, len(rs), bs):
            X, lens, _, _ = batch(rs[i:i + bs], None if wavs is None else wavs[i:i + bs])
            e, el = encode(X, lens)
            hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el,
                                                            return_hypotheses=True)
            hy = hy[0] if isinstance(hy, tuple) else hy
            for h in hy:
                ys = D.hyp_ids(h)
                eous += int(eou_id is not None and eou_id in ys)
                hyps.append(to_text(ys))
            if margins:
                diag += D.sos_blank_margin(m, e, el)
        m.train()
        if a.freeze_enc:
            m.encoder.eval()
        return hyps, eous, diag

    def save_state(path):
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "args": vars(a)}, path)

    log, best, t0, seen_s = [], None, time.time(), 0.0
    m.train()
    if a.freeze_enc:
        m.encoder.eval()
    order = []
    tr = rows[: a.train_eval_n]
    refs_tr = [norm(r["text"]) for r in tr]
    silence = [np.zeros(2 * 16000, dtype=np.float32)]
    while step < a.max_steps:
        if not order:
            order = list(range(len(rows)))
            random.shuffle(order)
        rs = [rows[i] for i in order[: a.bs]]
        order = order[a.bs:]
        X, lens, Y, ylens = batch(rs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e, el = encode(X, lens)
            d, _, _ = m.decoder(targets=Y, target_length=ylens)
            j = m.joint(encoder_outputs=e, decoder_outputs=d)
        loss = loss_fn(log_probs=j.float(), targets=Y, input_lengths=el, target_lengths=ylens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        step += 1
        seen_s += float(lens.sum()) / 16000
        if step % 25 == 0:
            print(f"  step {step}/{a.max_steps} loss {loss.item():.3f} audio {seen_s / 3600:.2f} h "
                  f"{seen_s / (time.time() - t0):.0f}x RT peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
                  flush=True)
        if step not in steps_eval:
            continue
        hy, eo, dg = decode(tr)
        tw, tc = wer_cer(refs_tr, hy)
        rec = {"step": step, "loss": round(loss.item(), 3), "train_n": len(tr), "train_wer": round(tw, 2),
               "train_cer": round(tc, 2), "train_empty_rate": round(sum(1 for h in hy if not h.strip()) / len(tr), 4),
               "train_eou_rate": round(eo / len(tr), 4), "audio_dependence": D.audio_dependence(hy),
               "nonblank_frac": round(float(np.mean([x["nonblank_frac"] for x in dg])), 4),
               "margin_max_mean": round(float(np.mean([x["margin_max"] for x in dg])), 3),
               "margin_mean_mean": round(float(np.mean([x["margin_mean"] for x in dg])), 3)}
        sh, _, _ = decode([{"text": ""}], wavs=silence, margins=False)
        rec["silence_hyp"] = sh[0]
        bh = None
        if a.beam_size > 0:
            prev = D.set_strategy(m, "beam", a.beam_size)
            try:
                bh, _, _ = decode(tr, margins=False)
            finally:
                D.restore_strategy(m, prev)
            bw, bc = wer_cer(refs_tr, bh)
            rec.update({"train_beam_wer": round(bw, 2), "train_beam_cer": round(bc, 2),
                        "beam_audio_dependence": D.audio_dependence(bh)})
        with open(out / "evals" / f"step_{step}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(tr):
                f.write(json.dumps({"audio_filepath": r["audio_filepath"], "ref": refs_tr[i], "greedy": hy[i],
                                    "beam": bh[i] if bh else None, **dg[i]}, ensure_ascii=False) + "\n")
        print(f"  EVAL {json.dumps(rec, ensure_ascii=False)}", flush=True)
        for r, h in list(zip(refs_tr, hy))[:2]:
            print(f"    ref: {r[:80]}\n    hyp: {h[:80]}", flush=True)
        if val:
            vh, veo, vdg = decode(val)
            vw, vc = wer_cer([norm(r["text"]) for r in val], vh)
            rec.update({"val_wer": round(vw, 2), "val_cer": round(vc, 2),
                        "val_empty": sum(1 for h in vh if not h.strip()), "val_eou_rate": round(veo / len(val), 3),
                        "val_nonblank": round(float(np.mean([x["nonblank_frac"] for x in vdg])), 4)})
            print(f"  VAL {json.dumps({k: rec[k] for k in rec if k.startswith('val')})}", flush=True)
            if best is None or vw < best:
                best = vw
                m.save_to(str(out / "final.nemo"))
                save_state(out / "best.ckpt")
                rec["saved"] = True
        save_state(out / "last.ckpt")
        log.append(rec)
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"args": vars(a), "log": log, "wall_s": round(time.time() - t0, 1),
                       "audio_h_seen": round(seen_s / 3600, 3),
                       "audio_h_per_gpu_h": round(seen_s / (time.time() - t0), 1),
                       "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}, f, indent=1)
    if not val:
        m.save_to(str(out / "final.nemo"))
    runmeta.write(out / "run.json", run, status="done", outputs={
        "final_nemo": str(out / "final.nemo"), "metrics": str(out / "metrics.json"),
        "checkpoint": runmeta.ckpt_record(out / "last.ckpt", step, "torch", runmeta.TORCH_RESUME),
        "best_val_wer": best, "last_eval": log[-1] if log else None})
    print(f"== plain-{a.tag} done {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
