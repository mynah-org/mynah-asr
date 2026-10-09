"""Plain Italian ASR adaptation of the STOCK EOU 120M (stage 1 of 2), minimal loop.

No Lightning, no lhotse, no EOU machinery: stock model, STOCK tokenizer (Italian
de-accented: è->e etc., since the English SPE maps accented vowels to <unk>),
stock decoder/joint, FastEmit 0, no padding, no gain. Streaming config untouched.

  python plain_it.py --manifest /root/ft/manifests/train_5h.json --n 32 --steps 400 \
      --lr 3e-4 --freeze-enc 1 --eval-every 100 --tag a0 [--val /root/ft/manifests/eval_mls_it.json]

Every --eval-every steps: loss, greedy on the first 8 TRAIN clips (overfit check),
non-blank frame %, blank margin, audio-dependence (distinct hypotheses), and with
--val: WER/CER (accent-insensitive) on --val-n val clips. Saves best-by-val (or
last) final.nemo + a torch state (model+optimizer+step) for continuation.
"""
import argparse, json, math, os, random, re, time, unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stock", default="/root/ft/models/parakeet_realtime_eou_120m-v1.nemo")
ap.add_argument("--manifest", required=True)
ap.add_argument("--n", type=int, default=0, help="first N utterances (0 = all)")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--enc-lr", type=float, default=0.0, help="lr for unfrozen encoder blocks")
ap.add_argument("--freeze-enc", type=int, default=1, help="1 = whole encoder frozen")
ap.add_argument("--unfreeze-top", type=int, default=0, help="unfreeze the top K encoder layers")
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--val", default="")
ap.add_argument("--val-n", type=int, default=200)
ap.add_argument("--max-dur", type=float, default=20.0)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="/root/ft/runs")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--init", default="", help="start from this .nemo instead of --stock (stage 2 from the plain-IT best)")
ap.add_argument("--eou-append", type=int, default=0, help="stage 2: target = text + <EOU> (NVIDIA EOU dataset semantics)")
ap.add_argument("--pad-prob", type=float, default=0.0, help="stage 2: prob of trailing silence after the speech")
ap.add_argument("--pad-min", type=float, default=1.0)
ap.add_argument("--pad-max", type=float, default=3.0)
ap.add_argument("--fastemit", type=float, default=0.0)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
out = Path(a.out) / f"plain-{a.tag}"; out.mkdir(parents=True, exist_ok=True)
dev = "cuda"

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.losses.rnnt import RNNTLoss

m = EncDecRNNTBPEModel.restore_from(a.init or a.stock, map_location=dev)
m.joint._fuse_loss_wer = False
tk = m.tokenizer
V = tk.vocab_size                       # 1026 incl. <EOU>/<EOB>; blank = V
loss_fn = RNNTLoss(num_classes=V, reduction="mean_batch", loss_name="warprnnt_numba",
                   loss_kwargs={"fastemit_lambda": a.fastemit})


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


def load(path):
    return [json.loads(l) for l in open(path)]


rows = [r for r in load(a.manifest) if r.get("duration", 0) <= a.max_dur and norm(r["text"])]
if a.n:
    rows = rows[: a.n]
EOU_ID = tk.token_to_id("<EOU>")
for r in rows:
    r["ids"] = tk.text_to_ids(norm(r["text"])) + ([EOU_ID] if a.eou_append else [])
val = [r for r in load(a.val)][:: max(1, len(load(a.val)) // a.val_n)][: a.val_n] if a.val else []
print(f"== plain-{a.tag}: train {len(rows)} utts {sum(r['duration'] for r in rows) / 3600:.3f} h, val {len(val)}", flush=True)

# parameters
for p in m.parameters():
    p.requires_grad = True
groups = [{"params": list(m.decoder.parameters()) + list(m.joint.parameters()), "lr": a.lr}]
if a.freeze_enc:
    for p in m.encoder.parameters():
        p.requires_grad = False
    if a.unfreeze_top:
        top = list(m.encoder.layers)[-a.unfreeze_top:]
        ps = [p for l in top for p in l.parameters()]
        for p in ps:
            p.requires_grad = True
        groups.append({"params": ps, "lr": a.enc_lr or a.lr * 0.1})
else:
    groups.append({"params": list(m.encoder.parameters()), "lr": a.enc_lr or a.lr * 0.1})
for p in m.preprocessor.parameters():
    p.requires_grad = False
opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=1e-3)
warm = max(10, a.steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))) * 0.99 + 0.01)
print("== trainable", sum(p.numel() for g in groups for p in g["params"]) / 1e6, "M", flush=True)


def audio(r):
    x, sr = sf.read(r["audio_filepath"], dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return x


def batch(rs, train=False):
    xs = [audio(r) for r in rs]
    if train and a.pad_prob > 0:   # trailing silence with a faint noise floor (stage 2)
        xs = [np.concatenate([x, (np.random.randn(int(16000 * random.uniform(a.pad_min, a.pad_max))) * 10 ** (random.uniform(-90, -60) / 20)).astype(np.float32)])
              if random.random() < a.pad_prob else x for x in xs]
    if not train and a.eou_append:  # evaluation of stage 2: 2 s of trailing quiet so an EOU can fire
        xs = [np.concatenate([x, (np.random.randn(32000) * 10 ** (-75 / 20)).astype(np.float32)]) for x in xs]
    L = max(len(x) for x in xs)
    X = torch.zeros(len(xs), L)
    for i, x in enumerate(xs):
        X[i, : len(x)] = torch.from_numpy(x)
    lens = torch.tensor([len(x) for x in xs])
    ids = [r.get("ids") or [0] for r in rs]   # validation rows carry no ids (targets unused there)
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


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_cer(refs, hyps):
    we = sum(lev(r.split(), h.split()) for r, h in zip(refs, hyps)); wn = sum(len(r.split()) for r in refs)
    ce = sum(lev(r, h) for r, h in zip(refs, hyps)); cn = sum(len(r) for r in refs)
    return 100 * we / max(1, wn), 100 * ce / max(1, cn)


@torch.no_grad()
def decode(rs, bs=16):
    m.eval(); hyps, eous, margins, nb = [], 0, [], []
    for i in range(0, len(rs), bs):
        X, lens, _, _ = batch(rs[i:i + bs])
        e, el = encode(X, lens)
        hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el, return_hypotheses=True)
        hy = hy[0] if isinstance(hy, tuple) else hy
        for h in hy:
            h = h[0] if isinstance(h, list) else h
            ys = h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)
            eous += int(1024 in ys)
            hyps.append(tk.ids_to_text([t for t in ys if t < 1024]))
        d, _ = m.decoder.predict(None, state=None, add_sos=False, batch_size=e.shape[0])[:2]
        j = m.joint.joint(e.transpose(1, 2), d)[:, :, 0, :]
        mg = j[..., :V].max(-1).values - j[..., V]
        for b in range(e.shape[0]):
            t = int(el[b]); margins.append(float(mg[b, :t].max())); nb.append(float((mg[b, :t] > 0).float().mean()))
    m.train()
    if a.freeze_enc:
        m.encoder.eval()
    return hyps, eous, margins, nb


log = []; best = None; t0 = time.time(); step = 0; seen_s = 0.0
m.train()
if a.freeze_enc:
    m.encoder.eval()
order = []
while step < a.steps:
    if not order:
        order = list(range(len(rows))); random.shuffle(order)
    rs = [rows[i] for i in order[: a.bs]]; order = order[a.bs:]
    X, lens, Y, ylens = batch(rs, train=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e, el = encode(X, lens)
        d, _, _ = m.decoder(targets=Y, target_length=ylens)
        j = m.joint(encoder_outputs=e, decoder_outputs=d)
    loss = loss_fn(log_probs=j.float(), targets=Y, input_lengths=el, target_lengths=ylens)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
    opt.step(); sched.step(); step += 1; seen_s += float(lens.sum()) / 16000
    if step % 25 == 0:
        print(f"  step {step}/{a.steps} loss {loss.item():.3f} audio {seen_s / 3600:.2f} h "
              f"{seen_s / (time.time() - t0):.0f}x RT peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        tr = rows[:8]
        hy, eo, mg, nb = decode(tr)
        tw, tc = wer_cer([norm(r["text"]) for r in tr], hy)
        rec = {"step": step, "loss": round(loss.item(), 3), "train8_wer": round(tw, 2), "train8_cer": round(tc, 2),
               "train8_empty": sum(1 for h in hy if not h.strip()), "train8_distinct": len(set(hy)),
               "nonblank_frac": round(float(np.mean(nb)), 4), "margin_max_mean": round(float(np.mean(mg)), 3)}
        print(f"  EVAL {json.dumps(rec)}", flush=True)
        for r, h in list(zip(tr, hy))[:2]:
            print(f"    ref: {norm(r['text'])[:80]}\n    hyp: {h[:80]}", flush=True)
        if val:
            vh, veo, _, vnb = decode(val)
            vw, vc = wer_cer([norm(r["text"]) for r in val], vh)
            rec.update({"val_wer": round(vw, 2), "val_cer": round(vc, 2), "val_empty": sum(1 for h in vh if not h.strip()),
                        "val_eou_rate": round(veo / len(val), 3), "val_nonblank": round(float(np.mean(vnb)), 4)})
            print(f"  VAL {json.dumps({k: rec[k] for k in rec if k.startswith('val')})}", flush=True)
            # stage 2 keeps EOU: best = lowest WER among evals with EOU on >= 80 % of the
            # val clips AND <= 10 % empty; until one qualifies, the most EOU (fewest empties) wins
            ok = rec["val_eou_rate"] >= 0.8 and rec["val_empty"] <= 0.1 * len(val)
            score = vw if not a.eou_append else (vw if ok else 1000 - 100 * rec["val_eou_rate"] + rec["val_empty"])
            if best is None or score < best:
                best = score
                m.save_to(str(out / "final.nemo"))
                torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                            "args": vars(a)}, out / "last.ckpt")
                rec["saved"] = True
        log.append(rec)
        json.dump({"args": vars(a), "log": log, "wall_s": round(time.time() - t0, 1), "audio_h_seen": round(seen_s / 3600, 3),
                   "audio_h_per_gpu_h": round(seen_s / (time.time() - t0), 1),
                   "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
                  open(out / "metrics.json", "w"), indent=1)
if not val:
    m.save_to(str(out / "final.nemo"))
print(f"== plain-{a.tag} done {time.time() - t0:.0f}s", flush=True)
