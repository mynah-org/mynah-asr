"""CPU-only probe of the RNNT prediction network + joint (no NeMo model instantiation).

For each .nemo (stock first) it reads model_weights.ckpt straight from the tar and reports:
  1. parameter counts per block (encoder / decoder / joint) and the decoder/joint shapes
  2. relative weight change ||W - W_stock|| / ||W_stock|| per decoder/joint tensor and per
     encoder block (the stock model is the reference)
  3. internal-LM (ILM) estimate of the transducer, Meng et al. 2021 style: the encoder
     term is removed from the joint, logits = joint_net(pred(g_u)) (variant "+encbias":
     the enc Linear bias is kept), blank and <EOU>/<EOB> are masked, softmax over the
     1024 SPE pieces. Token-level perplexity on the transcripts of the val manifests
     (Italian) and on en_sanity (English): how much Italian LM the predictor carries.
  4. context homophones (ha/a, hanno/anno, ho/o, hai/ai): for every occurrence in the val
     transcripts, ILM log-prob of the sentence with the true word vs the swapped word;
     reports the fraction where the ILM prefers the truth (0.5 = no LM knowledge).

  CUDA_VISIBLE_DEVICES= python pred_ilm_probe.py --out probe.json \
      stock=/root/ft/models/parakeet_realtime_eou_120m-v1.nemo b2=/root/ft/runs/plain-it-b2-gradual/final.nemo
"""
import argparse, io, json, math, re, sys, tarfile, tempfile, unicodedata
import torch

torch.set_num_threads(8)
ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", help="name=path.nemo (first one is the reference)")
ap.add_argument("--texts", default="mls=/root/ft/manifests/eval_mls_it.json,fleurs=/root/ft/manifests/eval_fleurs_it.json,"
                "cv=/root/ft/mix/it/eval_cv.json,vp=/root/ft/mix/it/eval_vp.json,en=/root/ft/manifests/en_sanity.json")
ap.add_argument("--max-sent", type=int, default=400, help="sentences per text set")
ap.add_argument("--out", default="probe.json")
a = ap.parse_args()


def norm(t):   # the de-accented target normalisation of two_stage.py (stock-tokenizer runs)
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


def load_nemo(path):
    t = tarfile.open(path)
    names = t.getnames()
    w = [n for n in names if n.endswith("model_weights.ckpt")][0]
    sd = torch.load(io.BytesIO(t.extractfile(w).read()), map_location="cpu")
    tok = [n for n in names if n.endswith("tokenizer.model")]
    sp = None
    if tok:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_proto=t.extractfile(tok[0]).read())
    return sd, sp


def block(k):
    if k.startswith("encoder.layers."):
        return "encoder.L%02d" % int(k.split(".")[2])
    return ".".join(k.split(".")[:2])


models = [x.split("=", 1) for x in a.models]
ref_sd, sp = load_nemo(models[0][1])
res = {"params": {}, "shapes": {}, "delta": {}, "ilm_ppl": {}, "homophones": {}}
for k, v in ref_sd.items():
    top = k.split(".")[0]
    res["params"][top] = res["params"].get(top, 0) + v.numel()
    if not k.startswith("encoder"):
        res["shapes"][k] = list(v.shape)

texts = {}
for spec in a.texts.split(","):
    n, p = spec.split("=", 1)
    try:
        rows = [json.loads(l) for l in open(p)]
    except FileNotFoundError:
        continue
    s = [norm(r["text"]) for r in rows if norm(r["text"])]
    texts[n] = s[:: max(1, len(s) // a.max_sent)][: a.max_sent]

V = 1024          # SPE pieces; 1024/1025 = <EOU>/<EOB>, 1026 = blank (= SOS / pad)
BLANK = 1026


class ILM:
    def __init__(self, sd):
        self.emb = sd["decoder.prediction.embed.weight"]
        pre = "decoder.prediction.dec_rnn.lstm."
        self.lstm = torch.nn.LSTM(self.emb.shape[1], sd[pre + "weight_hh_l0"].shape[1], num_layers=1, batch_first=True)
        self.lstm.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)})
        self.pw, self.pb = sd["joint.pred.weight"], sd["joint.pred.bias"]
        self.eb = sd["joint.enc.bias"]
        self.ow, self.ob = sd["joint.joint_net.2.weight"], sd["joint.joint_net.2.bias"]

    @torch.no_grad()
    def logprobs(self, ids_list, encbias=False):
        """per-sentence sum of log P_ilm(y_u | y_<u) over the SPE pieces, and token count."""
        out = []
        for i in range(0, len(ids_list), 64):
            chunk = ids_list[i:i + 64]
            U = max(len(x) for x in chunk)
            Y = torch.full((len(chunk), U + 1), BLANK, dtype=torch.long)
            for b, x in enumerate(chunk):
                Y[b, 1:len(x) + 1] = torch.tensor(x)
            e = self.emb[Y]
            e[Y == BLANK] = 0.0                      # blank_as_pad: SOS/pad embed is zero
            g, _ = self.lstm(e)                      # g[:, u] conditions on y_<=u
            h = g @ self.pw.T + self.pb
            if encbias:
                h = h + self.eb
            lo = torch.relu(h) @ self.ow.T + self.ob
            lp = torch.log_softmax(lo[..., :V], -1)  # mask blank + <EOU>/<EOB>
            tgt = Y[:, 1:]
            tok = lp[:, :-1].gather(-1, tgt.clamp(max=V - 1).unsqueeze(-1)).squeeze(-1)
            for b, x in enumerate(chunk):
                out.append((float(tok[b, :len(x)].sum()), len(x)))
        return out


def swaps(sents):
    pairs = {"ha": "a", "a": "ha", "hanno": "anno", "anno": "hanno", "ho": "o", "o": "ho", "hai": "ai", "ai": "hai"}
    out = []
    for s in sents:
        w = s.split()
        for i, x in enumerate(w):
            if x in pairs:
                out.append((x, " ".join(w), " ".join(w[:i] + [pairs[x]] + w[i + 1:])))
    return out


all_it = [s for n, ss in texts.items() if n != "en" for s in ss]
sw = swaps(all_it)
for name, path in models:
    sd = ref_sd if name == models[0][0] else load_nemo(path)[0]
    d = {}
    agg = {}
    for k, v in sd.items():
        if k not in ref_sd or not v.is_floating_point() or v.shape != ref_sd[k].shape:
            continue
        num = float((v.float() - ref_sd[k].float()).norm()); den = float(ref_sd[k].float().norm()) + 1e-12
        if not k.startswith("encoder"):
            d[k] = round(num / den, 4)
        b = block(k); s = agg.setdefault(b, [0.0, 0.0]); s[0] += num ** 2; s[1] += den ** 2
    d.update({"BLOCK " + b: round(math.sqrt(x / y), 4) for b, (x, y) in agg.items()})
    res["delta"][name] = d
    ilm = ILM(sd)
    pp = {}
    for variant in (False, True):
        for tn, ss in texts.items():
            r = ilm.logprobs([sp.encode(s) for s in ss], encbias=variant)
            pp[tn + ("+encbias" if variant else "")] = round(math.exp(-sum(x for x, _ in r) / sum(n for _, n in r)), 2)
    res["ilm_ppl"][name] = pp
    if sw:
        rt = ilm.logprobs([sp.encode(t) for _, t, _ in sw]); rf = ilm.logprobs([sp.encode(f) for _, _, f in sw])
        byw = {}
        for (w, _, _), (lt, _), (lf, _) in zip(sw, rt, rf):
            c = byw.setdefault(w, [0, 0]); c[0] += int(lt > lf); c[1] += 1
        res["homophones"][name] = {w: f"{c[0]}/{c[1]}" for w, c in sorted(byw.items())}
        res["homophones"][name]["ALL"] = round(sum(c[0] for c in byw.values()) / max(1, sum(c[1] for c in byw.values())), 3)
    print(name, json.dumps(pp), json.dumps(res["homophones"].get(name, {})), flush=True)
res["n_sent"] = {k: len(v) for k, v in texts.items()}
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
