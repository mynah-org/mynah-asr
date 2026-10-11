"""CTC probe on a FROZEN RNNT encoder: how much of the transcript is decodable from the encoder output alone.

Encodes N training clips + the val clips once with the checkpoint's encoder (same streaming config
as two_stage.py's eval), caches the frames, then trains only a CTC head (linear 512 -> 1024 pieces +
blank, or a 1-hidden-layer MLP with --hidden) on the cached frames and reports greedy-CTC WER/CER on
the val sets. No gradient ever reaches the encoder, so the result measures the encoder representation.
Compare with the RNNT greedy CER of the same checkpoint on the same clips (lm_fusion_probe.py prints it):
  probe CER <= RNNT CER  -> the encoder carries the information; the transducer output side loses it
  probe CER >> RNNT CER  -> weak evidence only (a small linear head is a lower bound on the encoder)

  CUDA_VISIBLE_DEVICES= python ctc_probe.py --nemo /root/ft/runs/plain-it-b2-gradual/final.nemo \
      --train /root/ft/manifests/train_40h.json,/root/ft/mix/it/train_cv.json,/root/ft/mix/it/train_vp_t50.json,/root/ft/mix/it/train_fleurs.json \
      --n-train 2000 --steps 3000 --out ctc_b2.json
"""
import argparse, json, random, re, time, unicodedata
import numpy as np, soundfile as sf, torch

ap = argparse.ArgumentParser()
ap.add_argument("--nemo", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--n-train", type=int, default=2000, help="training clips, spread evenly over the --train manifests")
ap.add_argument("--max-dur", type=float, default=15.0)
ap.add_argument("--vals", default="mls=/root/ft/manifests/eval_mls_it.json,fleurs=/root/ft/manifests/eval_fleurs_it.json,"
                "cv=/root/ft/mix/it/eval_cv.json,vp=/root/ft/mix/it/eval_vp.json")
ap.add_argument("--per-set", type=int, default=50)
ap.add_argument("--hidden", type=int, default=0, help="0 = linear head, else MLP hidden size")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--out", default="ctc_probe.json")
a = ap.parse_args()
torch.set_num_threads(a.threads); random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


def lev(x, y):
    prev = list(range(len(y) + 1))
    for i, p in enumerate(x, 1):
        cur = [i]
        for j, q in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != q)))
        prev = cur
    return prev[-1]


from nemo.collections.asr.models import EncDecRNNTBPEModel
m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location=dev).eval()
tk = m.tokenizer
V = 1024                                   # SPE pieces; CTC blank = 1024


@torch.no_grad()
def encode(rows):
    out = []
    for i in range(0, len(rows), 8):
        ch = rows[i:i + 8]
        xs = []
        for r in ch:
            x, _ = sf.read(r["audio_filepath"], dtype="float32")
            xs.append(x.mean(1) if x.ndim > 1 else x)
        X = torch.zeros(len(xs), max(len(x) for x in xs))
        for b, x in enumerate(xs):
            X[b, :len(x)] = torch.from_numpy(x)
        lens = torch.tensor([len(x) for x in xs])
        f, fl = m.preprocessor(input_signal=X.to(dev), length=lens.to(dev))
        e, el = m.encoder(audio_signal=f, length=fl)
        for b, r in enumerate(ch):
            out.append((e[b, :, : int(el[b])].T.half().cpu(), norm(r["text"])))
    return out


paths = a.train.split(",")
tr_rows = []
for p in paths:
    rs = [json.loads(l) for l in open(p)]
    rs = [r for r in rs if r.get("duration", 0) <= a.max_dur and norm(r["text"])]
    k = a.n_train // len(paths)
    tr_rows += rs[:: max(1, len(rs) // k)][:k]
vals = []
for spec in a.vals.split(","):
    n, p = spec.split("=", 1)
    rs = [json.loads(l) for l in open(p)]
    vals.append((n, rs[:: max(1, len(rs) // a.per_set)][: a.per_set]))
t0 = time.time()
tr = encode(tr_rows)
tr = [(e, tk.text_to_ids(t)) for e, t in tr if 0 < len(tk.text_to_ids(t)) <= e.shape[0]]
va = [(n, encode(rs)) for n, rs in vals]
print(f"encoded {len(tr)} train + {sum(len(v) for _, v in va)} val clips in {time.time() - t0:.0f}s on {dev}", flush=True)

D = tr[0][0].shape[1]
head = (torch.nn.Linear(D, V + 1) if not a.hidden else
        torch.nn.Sequential(torch.nn.Linear(D, a.hidden), torch.nn.ReLU(), torch.nn.Linear(a.hidden, V + 1))).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
ctc = torch.nn.CTCLoss(blank=V, zero_infinity=True)


def pad(batch):
    T = max(e.shape[0] for e, _ in batch)
    X = torch.zeros(len(batch), T, D)
    for b, (e, _) in enumerate(batch):
        X[b, : e.shape[0]] = e.float()
    return X.to(dev), torch.tensor([e.shape[0] for e, _ in batch])


@torch.no_grad()
def evaluate():
    head.eval(); rec = {}
    for n, rows in va:
        we = wn = ce = cn = 0
        for i in range(0, len(rows), 16):
            ch = rows[i:i + 16]
            X, L = pad(ch)
            pr = head(X).argmax(-1).cpu()
            for b, (_, ref) in enumerate(ch):
                seq, prev = [], -1
                for t in pr[b, : int(L[b])].tolist():
                    if t != prev and t != V:
                        seq.append(t)
                    prev = t
                hyp = norm(tk.ids_to_text(seq)) if seq else ""
                we += lev(ref.split(), hyp.split()); wn += len(ref.split()); ce += lev(ref, hyp); cn += len(ref)
        rec[f"{n}_wer"] = round(100 * we / wn, 2); rec[f"{n}_cer"] = round(100 * ce / cn, 2)
    rec["macro_wer"] = round(float(np.mean([rec[f"{n}_wer"] for n, _ in va])), 2)
    rec["macro_cer"] = round(float(np.mean([rec[f"{n}_cer"] for n, _ in va])), 2)
    head.train()
    return rec


log = []
for step in range(1, a.steps + 1):
    batch = random.sample(tr, a.bs)
    X, L = pad(batch)
    Y = torch.cat([torch.tensor(y) for _, y in batch]); YL = torch.tensor([len(y) for _, y in batch])
    lp = head(X).log_softmax(-1).transpose(0, 1)
    loss = ctc(lp, Y, L, YL)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % max(1, a.steps // 6) == 0 or step == a.steps:
        r = {"step": step, "loss": round(loss.item(), 3), **evaluate()}
        log.append(r); print(json.dumps(r), flush=True)
        json.dump({"args": vars(a), "log": log}, open(a.out, "w"), indent=1)
print("wrote", a.out)
