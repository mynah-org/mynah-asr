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

Multi-domain extensions (2026-10-10; every default reproduces the 2026-10-09 runs):
  --mix a.json:0.35,b.json:0.30,...   domain-balanced sampling: each batch item picks a
                                      source by weight, then the next row of that source
                                      (instead of one uniform shuffle of --manifest)
  --vals mls=a.json,cv=b.json         several val sets; val_wer = MACRO mean over the sets
                                      (selection uses it), per-set keys val_<name>_wer ...
  --aug-gain P --gain-db LO,HI        random gain (then clip), --aug-speed P (0.9-1.1,
  --aug-noise P --snr LO,HI           resampled), additive white/brown noise at an SNR
  --unfreeze-sched 0:4,0.2:8,0.5:all  gradual unfreeze (fraction of steps : top-K blocks);
  --enc-lr / --enc-lr-mid / --enc-lr-low  discriminative lr: top 4 / blocks 5-8 / rest
  --concat-prob P --concat-gap LO,HI  stage 2 negatives: A + short pause + B as ONE turn,
                                      target text(A) text(B) <EOU> (no EOU inside)
  --pad-noise-db LO,HI                level of the trailing quiet (default -90,-60 dB)
  --tokenizer DIR                     Italian SPE (tokenizer_eou.py: same size, <EOU>/<EOB> last),
                                      WARM-STARTED: decoder/joint rows of pieces the stock SPE also
                                      has are copied, new pieces start from the mean of the rows
                                      the stock SPE splits their text into; blank/<EOU>/<EOB> keep
                                      their ids and rows. Targets keep accents (norm_text); WER/CER
                                      stay accent-insensitive (hypotheses normalised like refs)
  --lang it|fr                        stock-SPE targets and WER/CER refs/hyps through common/textnorm.norm(t, lang)
                                      (it, default = the frozen rule; fr maps oe / ae and typographic apostrophes)
  --accented 1|0                      accented targets (default: 1 with --tokenizer, else 0); 1 on a
                                      model that already carries such an SPE (stage 2 from a
                                      --tokenizer run), 0 with a --deaccent SPE (vocabulary-only arm)
"""
import argparse, collections, hashlib, json, math, os, random, re, sys, time, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import textnorm  # noqa: E402

import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stock", default="/root/ft/models/parakeet_realtime_eou_120m-v1.nemo")
ap.add_argument("--manifest", default="")
ap.add_argument("--mix", default="", help="path:weight,... domain-balanced sampling (replaces --manifest)")
ap.add_argument("--vals", default="", help="name=path,... extra val sets (macro-mean WER selects)")
ap.add_argument("--aug-gain", type=float, default=0.0)
ap.add_argument("--gain-db", default="-20,10")
ap.add_argument("--aug-speed", type=float, default=0.0)
ap.add_argument("--aug-noise", type=float, default=0.0)
ap.add_argument("--snr", default="10,40")
ap.add_argument("--unfreeze-sched", default="", help="frac:K,... (K = top blocks or 'all'); overrides --freeze-enc/--unfreeze-top")
ap.add_argument("--enc-lr-mid", type=float, default=0.0, help="lr of encoder blocks 5-8 from the top (default = --enc-lr)")
ap.add_argument("--enc-lr-low", type=float, default=0.0, help="lr of the lower blocks + pre-encode (default = --enc-lr)")
ap.add_argument("--concat-prob", type=float, default=0.0)
ap.add_argument("--concat-gap", default="0.15,0.6")
ap.add_argument("--pause-stretch-prob", type=float, default=0.0, help="stage 2 FR-v2: per training sample, prob of stretching "
                "ONE genuine internal pause (silence run >= 0.16 s, >= 0.5 s after the first speech frame, >= 0.5 s of speech "
                "after it; 80 ms frames, speech = within 35 dB of the clip peak) to U(--pause-stretch-s) s with the clip's own "
                "room tone; the target is unchanged (<EOU> only at the real end). Drawn from its OWN RNG: with 0 (default) "
                "the batches are those of v4 bit for bit, and with p > 0 every other sample is unchanged")
ap.add_argument("--pause-stretch-s", default="1.0,2.5")
ap.add_argument("--silence-audit", type=int, default=0, help="build this many TRAINING batches with the run's sampler and "
                "report the silence the model is shown (internal continuation pauses vs terminal silence before <EOU>, "
                "by duration bin), then exit (no step, nothing saved)")
ap.add_argument("--pad-noise-db", default="-90,-60")
ap.add_argument("--eval-only", type=int, default=0, help="1 = evaluate --init/--stock on the val sets once, no training")
ap.add_argument("--n", type=int, default=0, help="first N utterances (0 = all)")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--enc-lr", type=float, default=0.0, help="lr for unfrozen encoder blocks")
ap.add_argument("--freeze-enc", type=int, default=1, help="1 = whole encoder frozen")
ap.add_argument("--unfreeze-top", type=int, default=0, help="unfreeze the top K encoder layers")
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--eval-steps", default="", help="extra eval steps, e.g. 100,250 (curve shape; stage 1 training is unchanged:"
                " evals use no torch RNG and stage 1 no numpy RNG)")
ap.add_argument("--eval-sparse", default="", help="FLAT,DECAY: eval every FLAT steps before the WSD decay starts and every DECAY "
                "steps inside it (plus the last step), instead of --eval-every; training is unchanged (evals use no RNG)")
ap.add_argument("--final-eof", type=int, default=0, help="1 = final.nemo is the LAST step (pre-registered decision point), "
                "not the best in-run val; evals are diagnostic only and save nothing during the run")
ap.add_argument("--profile-steps", type=int, default=0,
                help="throughput profile only: run this many training steps (+20 skipped as warm-up), print where the step "
                     "time goes (host batch build vs GPU spans from CUDA events, no extra syncs) and exit; nothing saved")
ap.add_argument("--batch-audit", type=int, default=0, help="build this many TRAINING batches with the run's sampler, print "
                "per-batch sha256 of the audio (X, lens) and of the targets (Y, ylens) plus the share of targets ending in "
                "<EOU>, and exit (no step, nothing saved); np.random is seeded for the audit so the noise floor is hashable")
ap.add_argument("--stop-at", type=int, default=0, help="stop after this step, LR schedule still over --steps "
                "(early-curve sweeps that share a longer run's schedule and batches)")
ap.add_argument("--abort-wer", default="", help="STEP:WER -- stop (rc 5) if the macro val WER at STEP is above WER (collapse gate)")
ap.add_argument("--val", default="")
ap.add_argument("--val-n", type=int, default=200)
ap.add_argument("--max-dur", type=float, default=20.0)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="/root/ft/runs")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--init", default="", help="start from this .nemo instead of --stock (stage 2 from the plain-IT best)")
ap.add_argument("--eou-append", type=int, default=0, help="stage 2: target = text + <EOU> (NVIDIA EOU dataset semantics)")
ap.add_argument("--pad-prob", type=float, default=0.0, help="stage 2: prob of trailing silence after the speech")
ap.add_argument("--eou-min-tail", type=float, default=0.0, help="stage 2 v5-B: an UNPADDED sample gets <EOU> only if its "
                "manifest trail_s (natural trailing non-speech, diag/annotate_tail.py) is >= S s, otherwise text only; "
                "padded samples keep <EOU> (0 = every target ends in <EOU>, v4)")
ap.add_argument("--eou-amb-keep", type=float, default=0.0, help="stage 2 v5-CAL: with --eou-min-tail, an ambiguous "
                "unpadded end KEEPS its <EOU> with this probability per draw (0 = v5-B, 1 = v4); drawn from its own RNG, "
                "so the audio batches stay those of v4 / v5-B")
ap.add_argument("--pre-pad-prob", type=float, default=0.0, help="stage 2: prob of LEADING quiet U(0, --pre-pad-max) s "
                "(NVIDIA's EOU dataset pads before the speech too; VP clips start at frame 0)")
ap.add_argument("--pre-pad-max", type=float, default=4.0)
ap.add_argument("--pad-min", type=float, default=1.0)
ap.add_argument("--pad-max", type=float, default=3.0)
ap.add_argument("--fastemit", type=float, default=0.0)
ap.add_argument("--ctc-weight", type=float, default=0.0, help="auxiliary CTC loss on the encoder output (training-only "
                "linear head, not saved in final.nemo): loss = rnnt + w * ctc, both summed per utterance, batch mean")
ap.add_argument("--kd-teacher", default="", help="encoder feature distillation (training only, not in final.nemo): "
                "teacher .nemo (Nemotron 0.6B); its --kd-t-layer output (eval, no grad, clean features of the SAME "
                "augmented audio) is matched frame by frame (T_teacher == T_student, kd_audit.py) by the student's "
                "--kd-s-layer output through a NEW random-init Linear(d_s, d_t): loss = rnnt + lambda(t) * mean over valid "
                "frames of (1 - cos)")
ap.add_argument("--kd-s-layer", type=int, default=7)
ap.add_argument("--kd-t-layer", type=int, default=17)
ap.add_argument("--kd-weight", type=float, default=0.0, help="lambda at step 0")
ap.add_argument("--kd-end-frac", type=float, default=1.0, help="lambda(s) = kd_weight * max(0, 1 - s / (kd_end_frac * steps)): "
                "linear from step 0, 0 from kd_end_frac * steps on (1.0 = over the whole run)")
ap.add_argument("--kd-calib", type=int, default=0, help="N > 0: gradient calibration only (no optimizer step, no eval): "
                "on N training batches, encoder grad norms of the RNNT loss and of the KD loss at lambda 1, their ratio "
                "and cosine, printed as == KD-CALIB, then exit")
ap.add_argument("--kd-t-context", default="", help="teacher att_context_size 'L,R' (default: the checkpoint's)")
ap.add_argument("--lr-shape", default="cosine", choices=("cosine", "wsd"), help="wsd: warmup, flat peak LR, then a "
                "cosine decay over the last --decay-frac of --steps (warmup-stable-decay)")
ap.add_argument("--decay-frac", type=float, default=0.2)
ap.add_argument("--tokenizer", default="", help="warm-started language SPE dir (tokenizer_eou.py output)")
ap.add_argument("--accented", type=int, default=-1, help="accented targets (default: 1 with --tokenizer, else 0)")
ap.add_argument("--lang", default="it", choices=sorted(textnorm.PRE), help="stock-SPE target + scorer normaliser "
                "(common/textnorm.py; it = the frozen rule, fr = + oe / ae and typographic apostrophes)")
a = ap.parse_args()
a.accented = bool(a.tokenizer) if a.accented < 0 else bool(a.accented)
random.seed(a.seed); torch.manual_seed(a.seed)
CAL_RNG = random.Random(a.seed + 7919)   # v5-CAL label draws only: never touches the global stream of the batches
PS_RNG = random.Random(a.seed + 104729)   # FR-v2 pause stretching only: own stream, the v4 draws are untouched
PS_NP = np.random.RandomState(a.seed + 104729)
out = Path(a.out) / f"plain-{a.tag}"; out.mkdir(parents=True, exist_ok=True)
dev = "cuda"

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.losses.rnnt import RNNTLoss

m = EncDecRNNTBPEModel.restore_from(a.init or a.stock, map_location=dev)
if a.tokenizer:
    # change_vocabulary rebuilds decoder + joint from the config (random init): keep the trained
    # tensors, then remap only the vocabulary rows (dim 0 == V + 1, blank last)
    old_sp = m.tokenizer.tokenizer
    old_ids = {old_sp.id_to_piece(i): i for i in range(old_sp.get_piece_size())}
    keep_sd = {k: v.detach().clone() for k, v in m.state_dict().items() if k.startswith(("decoder.", "joint."))}
    m.change_vocabulary(new_tokenizer_dir=a.tokenizer, new_tokenizer_type="bpe")
    new_sp = m.tokenizer.tokenizer
    Vn = m.tokenizer.vocab_size
    assert Vn == old_sp.get_piece_size(), (Vn, old_sp.get_piece_size())
    from sentencepiece import sentencepiece_model_pb2 as _pb
    _mp = _pb.ModelProto(); _mp.ParseFromString(old_sp.serialized_model_proto())
    dummy0 = _mp.normalizer_spec.add_dummy_prefix   # restored after each word-internal split
    src, n_exact = [], 0
    for i in range(Vn):
        pc = new_sp.id_to_piece(i)
        if pc in old_ids:
            src.append([old_ids[pc]]); n_exact += 1; continue
        t = "".join(c for c in unicodedata.normalize("NFKD", pc.replace("\u2581", " ")) if not unicodedata.combining(c))
        # a word-internal piece ("zione") is split WITHOUT the dummy "\u2581" prefix, else BPE merges
        # it into the first stock piece and the row starts as a word-initial one ("\u2581z")
        internal = not pc.startswith("\u2581")
        if internal:
            old_sp.override_normalizer_spec(add_dummy_prefix=False)
        ids = [j for j in old_sp.encode(t.strip()) if j != old_sp.unk_id()]
        if internal:
            old_sp.override_normalizer_spec(add_dummy_prefix=dummy0)
        src.append(ids or [old_sp.unk_id()])
    src.append([Vn])                      # blank
    for k, v in keep_sd.items():
        if v.shape[0] == Vn + 1:
            keep_sd[k] = torch.stack([v[ids].mean(0) for ids in src])
            print(f"   warm-start rows {k} {tuple(v.shape)}", flush=True)
    missing, unexpected = m.load_state_dict(keep_sd, strict=False)
    assert not unexpected and not [k for k in missing if k.startswith(("decoder.", "joint."))], (missing, unexpected)
    m.to(dev)
    print(f"== tokenizer {a.tokenizer}: {Vn} pieces, {n_exact} copied from the stock SPE, "
          f"{Vn - n_exact} from the mean of their stock split", flush=True)
m.joint._fuse_loss_wer = False
# Evaluation decodes with the per-utterance greedy loop: NeMo's batched greedy
# (CUDA-graph label loop) hit "illegal memory access" in batched_hyps_to_hypotheses
# mid-run on this box (b1 after step 1500, b2 at step 500).
from omegaconf import open_dict
_dc = m.cfg.decoding
with open_dict(_dc):
    _dc.strategy = "greedy"
    if "greedy" in _dc:
        _dc.greedy.use_cuda_graph_decoder = False
m.change_decoding_strategy(_dc)
tk = m.tokenizer
V = tk.vocab_size                       # 1026 incl. <EOU>/<EOB>; blank = V
loss_fn = RNNTLoss(num_classes=V, reduction="mean_batch", loss_name="warprnnt_numba",
                   loss_kwargs={"fastemit_lambda": a.fastemit})


def norm(t):
    return textnorm.norm(t, a.lang)   # lang="it" == the historical rule of this file (byte-identical, tests)


if a.accented:
    from tokenizer_eou import norm_text    # lower case, accents and apostrophes kept
tnorm = norm_text if a.accented else norm   # training targets; refs/hyps are always scored with norm


def load(path):
    return [json.loads(l) for l in open(path)]


EOU_ID = tk.token_to_id("<EOU>")


def train_rows(path):
    rs = [r for r in load(path) if r.get("duration", 0) <= a.max_dur and tnorm(r["text"])]
    for r in rs:
        r["ids_t"] = tk.text_to_ids(tnorm(r["text"]))
        r["ids"] = r["ids_t"] + ([EOU_ID] if a.eou_append else [])
    return rs


if a.mix:   # [(name, rows, weight)]
    sources = [(Path(p).stem, train_rows(p), float(w)) for p, w in (x.rsplit(":", 1) for x in a.mix.split(","))]
else:
    sources = [("train", train_rows(a.manifest), 1.0)]
if a.n:
    sources = [(n, rs[: a.n], w) for n, rs, w in sources]
rows = sources[0][1]                     # first source: the train8 overfit probe
for n, rs, w in sources:
    print(f"   source {n}: {len(rs)} utts {sum(r['duration'] for r in rs) / 3600:.2f} h weight {w}", flush=True)


def subsample(path):
    rs = load(path)
    return rs[:: max(1, len(rs) // a.val_n)][: a.val_n]


vals = ([("val", subsample(a.val))] if a.val else []) + [(n, subsample(p)) for n, p in (x.split("=", 1) for x in a.vals.split(",") if x)]
val = [r for _, v in vals for r in v]
print(f"== plain-{a.tag}: train {sum(len(rs) for _, rs, _ in sources)} utts "
      f"{sum(r['duration'] for _, rs, _ in sources for r in rs) / 3600:.3f} h, val {[(n, len(v)) for n, v in vals]}", flush=True)

# parameters
for p in m.parameters():
    p.requires_grad = True
groups = [{"params": list(m.decoder.parameters()) + list(m.joint.parameters()), "lr": a.lr}]
enc_layers = list(m.encoder.layers)
unfreeze = []                            # [(step, top-K)] for --unfreeze-sched
if a.unfreeze_sched:
    # discriminative lr: blocks 1-4 from the top --enc-lr, 5-8 --enc-lr-mid, the rest
    # and pre-encode --enc-lr-low; frozen params keep grad None (AdamW skips them)
    base = a.enc_lr or a.lr * 0.1
    by_lr = {}
    for k, l in enumerate(reversed(enc_layers), 1):
        lr = base if k <= 4 else (a.enc_lr_mid or base) if k <= 8 else (a.enc_lr_low or base)
        by_lr.setdefault(lr, []).extend(l.parameters())
    rest = [p for n_, p in m.encoder.named_parameters() if not n_.startswith("layers.")]
    by_lr.setdefault(a.enc_lr_low or base, []).extend(rest)
    for lr, ps in by_lr.items():
        groups.append({"params": ps, "lr": lr})
    for x in a.unfreeze_sched.split(","):
        f, k = x.split(":")
        unfreeze.append((int(float(f) * a.steps), len(enc_layers) + 1 if k == "all" else int(k)))


def apply_unfreeze(step):
    """Top-K encoder blocks trainable (K > n_layers: also pre-encode); returns K."""
    k = max([kk for s, kk in unfreeze if s <= step], default=0)
    for i, l in enumerate(reversed(enc_layers), 1):
        for p in l.parameters():
            p.requires_grad = i <= k
    for n_, p in m.encoder.named_parameters():
        if not n_.startswith("layers."):
            p.requires_grad = k > len(enc_layers)
    return k


if a.unfreeze_sched:
    print(f"== unfreeze schedule {unfreeze}, top-K now {apply_unfreeze(0)}", flush=True)
elif a.freeze_enc:
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
ctc_head = None
if a.ctc_weight:
    ctc_head = torch.nn.Linear(int(m.cfg.encoder.d_model), V + 1).to(dev)   # V pieces incl. <EOU>/<EOB>, blank = V
    groups.append({"params": list(ctc_head.parameters()), "lr": a.lr})
kd_proj = kd_t = None
if a.kd_teacher:
    from nemo.collections.asr.models import ASRModel
    kd_t = ASRModel.restore_from(a.kd_teacher, map_location=dev).eval()
    for p_ in kd_t.parameters():
        p_.requires_grad = False
    if a.kd_t_context:
        kd_t.encoder.set_default_att_context_size([int(v) for v in a.kd_t_context.split(",")])
    with torch.no_grad():   # the teacher reads the student's features: the two preprocessors must be the same function
        m.preprocessor.eval(); x_ = torch.randn(2, 48000, device=dev) * 0.1; l_ = torch.tensor([48000, 30000], device=dev)
        fs_, _ = m.preprocessor(input_signal=x_, length=l_); ft_, _ = kd_t.preprocessor(input_signal=x_, length=l_)
        dpre = float((fs_ - ft_).abs().max()); m.preprocessor.train()
    print(f"== KD: preprocessor max |student - teacher| = {dpre:.2e} (eval, same input)", flush=True)
    if dpre > 1e-3:
        raise SystemExit("== KD: preprocessors differ, the shared-feature design does not hold")
    kd_t = kd_t.to(torch.bfloat16)   # inference only: halves the teacher's resident memory (1.3 GB)
    kd_store = {}
    kd_t.encoder.layers[a.kd_t_layer].register_forward_hook(
        lambda mod, inp, o: kd_store.__setitem__("t", (o[0] if isinstance(o, tuple) else o)))
    enc_layers[a.kd_s_layer].register_forward_hook(
        lambda mod, inp, o: kd_store.__setitem__("s", (o[0] if isinstance(o, tuple) else o)) if m.training else None)
    g_ = torch.Generator().manual_seed(a.seed + 1)   # fresh projection: nothing from the kd_audit.py probe
    kd_proj = torch.nn.Linear(int(m.cfg.encoder.d_model), int(kd_t.cfg.encoder.d_model))
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(kd_proj.weight, generator=g_); kd_proj.bias.zero_()
    kd_proj = kd_proj.to(dev)
    groups.append({"params": list(kd_proj.parameters()), "lr": a.lr})
    print(f"== KD: teacher {a.kd_teacher} L{a.kd_t_layer} (att_context {kd_t.encoder.att_context_size}) <- student "
          f"L{a.kd_s_layer} via Linear({kd_proj.in_features},{kd_proj.out_features}); lambda {a.kd_weight} -> 0 at "
          f"{a.kd_end_frac:g} of the steps; loss 1 - cos", flush=True)


def kd_lambda(s):
    return a.kd_weight * max(0.0, 1.0 - s / max(1.0, a.kd_end_frac * a.steps))


opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=1e-3)
warm = max(10, a.steps // 20)


def lr_factor(s):
    if a.lr_shape == "wsd":
        d0 = a.steps * (1 - a.decay_frac)
        f = 1.0 if s < d0 else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - d0) / max(1.0, a.steps - d0))))
    else:
        f = 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    return min(1.0, (s + 1) / warm) * f * 0.99 + 0.01


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
_ng = len(groups) - (kd_proj is not None)
group_names = (["decoder+joint"] + [f"encoder-{i}" for i in range(1, _ng - 1 - (ctc_head is not None))]
               + (["encoder"] if _ng - (ctc_head is not None) == 2 else []))[: _ng - (ctc_head is not None)] \
    + (["ctc_head"] if ctc_head is not None else [])
if kd_proj is not None:   # the projection group is the last one
    group_names = group_names[:len(groups) - 1] + ["kd_proj"]
lr_auc = [0.0] * len(groups)   # integrated LR per parameter group (sum over optimizer steps)
print("== trainable", sum(p.numel() for g in groups for p in g["params"]) / 1e6, "M", flush=True)


def audio(r):
    x, sr = sf.read(r["audio_filepath"], dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return x


def rng2(spec):
    lo, hi = (float(v) for v in spec.split(","))
    return random.uniform(lo, hi)


def quiet(sec):
    return (np.random.randn(int(16000 * sec)) * 10 ** (rng2(a.pad_noise_db) / 20)).astype(np.float32)


def speech_runs(x):
    """80 ms frames, speech = within 35 dB of the clip peak (the eval endpoint rule): (mask, first, last speech frame)"""
    n = 1280
    f = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    sm = db > db.max() - 35
    on = np.nonzero(sm)[0]
    return sm, (int(on[0]) if len(on) else -1), (int(on[-1]) if len(on) else -1)


def internal_pauses(sm, first, last, min_after=0.5, min_start=0.5, min_len=2):
    """silence runs (j, k) inside [first, last]: >= min_len frames, starting >= min_start s after the first speech frame,
    with >= min_after s of speech after them"""
    out, j = [], first
    while 0 <= j <= last:
        if not sm[j]:
            k = j
            while not sm[k]:
                k += 1
            if k - j >= min_len and (j - first) * 0.08 >= min_start and sm[k:].sum() * 0.08 >= min_after:
                out.append((j, k))
            j = k
        else:
            j += 1
    return out


def stretch_pause(x):
    """FR-v2: one internal pause stretched to U(--pause-stretch-s) s with the clip's own room tone (random 80 ms
    non-speech frames, 10 ms crossfades); the speech before / after is untouched. Returns (x, stretched length or 0)."""
    if PS_RNG.random() >= a.pause_stretch_prob:
        return x, 0.0
    sm, first, last = speech_runs(x)
    ps = internal_pauses(sm, first, last)
    quiet_frames = np.nonzero(~sm)[0]
    if not ps or len(quiet_frames) < 2:
        return x, 0.0
    j, k = ps[PS_RNG.randrange(len(ps))]
    lo, hi = (float(v) for v in a.pause_stretch_s.split(","))
    target = PS_RNG.uniform(lo, hi)
    need = target - (k - j) * 0.08
    if need <= 0:
        return x, 0.0
    n, xf = 1280, 160
    pieces, total = [], 0
    while total < int(need * 16000) + xf:
        q = int(quiet_frames[PS_NP.randint(len(quiet_frames))])
        pieces.append(x[q * n:(q + 1) * n].copy()); total += n - xf
    ramp = np.linspace(0, 1, xf, dtype=np.float32)
    fill = pieces[0]
    for p in pieces[1:]:
        fill = np.concatenate([fill[:-xf], fill[-xf:] * (1 - ramp) + p[:xf] * ramp, p[xf:]])
    fill = fill[: int(need * 16000) + xf]
    cut = (j + k) // 2 * n
    a_, b_ = x[:cut], x[cut:]
    mid = np.concatenate([a_[-xf:] * (1 - ramp) + fill[:xf] * ramp, fill[xf:-xf], fill[-xf:] * (1 - ramp) + b_[:xf] * ramp])
    return np.concatenate([a_[:-xf], mid, b_[xf:]]).astype(np.float32), round(target, 2)


def augment(x):
    """speed (resampled: pitch + tempo), additive white/brown noise at an SNR, gain + clip."""
    if a.aug_speed and random.random() < a.aug_speed:
        f = random.uniform(0.9, 1.1)
        x = np.interp(np.arange(0, len(x) - 1, f), np.arange(len(x)), x).astype(np.float32)
    if a.aug_noise and random.random() < a.aug_noise:
        n = np.random.randn(len(x)).astype(np.float32)
        if random.random() < 0.5:   # brown: integrated white, high-passed by mean removal
            n = np.cumsum(n); n -= np.convolve(n, np.ones(400) / 400, mode="same")
        rms = float(np.sqrt(np.mean(x ** 2))) + 1e-6
        x = x + n / (float(np.sqrt(np.mean(n ** 2))) + 1e-9) * rms * 10 ** (-rng2(a.snr) / 20)
    if a.aug_gain and random.random() < a.aug_gain:
        x = np.clip(x * 10 ** (rng2(a.gain_db) / 20), -1, 1)
    return x.astype(np.float32)


def batch(rs, train=False):
    xs = [audio(r) for r in rs]
    ids = [r.get("ids") or [0] for r in rs]   # validation rows carry no ids (targets unused there)
    if train and a.pause_stretch_prob > 0:   # FR-v2 hard negatives: a long internal pause, speech continues, no <EOU> there
        for i in range(len(xs)):
            xs[i], _ = stretch_pause(xs[i])
    if train and a.concat_prob > 0:   # stage 2 negatives: A + short pause + B, one turn, EOU only at the end
        for i, r in enumerate(rs):
            if random.random() < a.concat_prob:
                r2 = pick_row()
                if r["duration"] + r2["duration"] > a.max_dur:   # keep the joint (B x T x U x V) bounded
                    continue
                xs[i] = np.concatenate([xs[i], quiet(rng2(a.concat_gap)), audio(r2)])
                ids[i] = r["ids_t"] + r2["ids"]
    if train and a.pre_pad_prob > 0:   # leading silence, same noise floor (stage 2 v3)
        xs = [np.concatenate([quiet(random.uniform(0.0, a.pre_pad_max)), x])
              if random.random() < a.pre_pad_prob else x for x in xs]
    padded = [False] * len(xs)
    if train and a.pad_prob > 0:   # trailing silence with a faint noise floor (stage 2)
        for i, x in enumerate(xs):   # same random-call order as before (v4 draws are reproduced exactly)
            if random.random() < a.pad_prob:
                xs[i] = np.concatenate([x, quiet(random.uniform(a.pad_min, a.pad_max))]); padded[i] = True
    if train and a.eou_append and a.eou_min_tail > 0:   # v5-B: no <EOU> on an acoustically ambiguous (unpadded) end
        for i, r in enumerate(rs):
            if not padded[i] and r["trail_s"] < a.eou_min_tail and ids[i] and ids[i][-1] == EOU_ID:
                if a.eou_amb_keep > 0 and CAL_RNG.random() < a.eou_amb_keep:   # v5-CAL: keep a fraction q
                    continue
                ids[i] = ids[i][:-1]
    if train and (a.aug_speed or a.aug_noise or a.aug_gain):
        xs = [augment(x) for x in xs]
    if not train and a.eou_append:  # evaluation of stage 2: 2 s of trailing quiet so an EOU can fire
        xs = [np.concatenate([x, (np.random.randn(32000) * 10 ** (-75 / 20)).astype(np.float32)]) for x in xs]
    L = max(len(x) for x in xs)
    X = torch.zeros(len(xs), L)
    for i, x in enumerate(xs):
        X[i, : len(x)] = torch.from_numpy(x)
    lens = torch.tensor([len(x) for x in xs])
    U = max(len(t) for t in ids)
    Y = torch.zeros(len(rs), U, dtype=torch.long)
    for i, t in enumerate(ids):
        Y[i, : len(t)] = torch.tensor(t)
    ylens = torch.tensor([len(t) for t in ids])
    return X.to(dev), lens.to(dev), Y.to(dev), ylens.to(dev)


kd_feats = {}


def encode(X, lens):
    f, fl = m.preprocessor(input_signal=X, length=lens)
    if kd_t is not None and m.training:   # the teacher reads THESE features (same dither realisation), before SpecAugment;
        t_k0 = time.perf_counter()        # run BEFORE the student encoder / joint: its no-grad activations are freed first
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):   # preprocessors identical (checked at start)
            kd_t.encoder(audio_signal=f.detach(), length=fl)
        kd_feats["ms"] = (time.perf_counter() - t_k0) * 1000
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
            h_ = tk.ids_to_text([t for t in ys if t < 1024])
            hyps.append(norm(h_) if (a.accented or a.tokenizer) else h_)   # stock path unchanged (bit-identical)
        d, _ = m.decoder.predict(None, state=None, add_sos=False, batch_size=e.shape[0])[:2]
        j = m.joint.joint(e.transpose(1, 2), d)[:, :, 0, :]
        mg = j[..., :V].max(-1).values - j[..., V]
        for b in range(e.shape[0]):
            t = int(el[b]); margins.append(float(mg[b, :t].max())); nb.append(float((mg[b, :t] > 0).float().mean()))
    set_train_mode()
    return hyps, eous, margins, nb


cur_k = 0


def set_train_mode():
    # 2026-10-09 semantics: a (partly) frozen encoder runs in eval mode (no dropout);
    # with --unfreeze-sched only the fully unfrozen encoder trains with dropout
    m.train()
    if (a.unfreeze_sched and cur_k <= len(enc_layers)) or (not a.unfreeze_sched and a.freeze_enc):
        m.encoder.eval()


orders = {}


def next_row(si):
    o = orders.get(si)
    if not o:
        o = orders[si] = list(range(len(sources[si][1]))); random.shuffle(o)
    return sources[si][1][o.pop()]


def pick_row():
    return next_row(random.choices(range(len(sources)), weights=[w for _, _, w in sources])[0])


def profile_report(host, evs, wall, skip=20):
    """Where a training step's wall time goes. GPU spans come from CUDA events (read after one final sync), so no
    sync is added inside the loop; 'gpu idle' = GPU time between one step's last event and the next step's first,
    i.e. the GPU waiting for the host (batch build, launches): the upper bound of what a batch prefetch recovers."""
    torch.cuda.synchronize()
    h, E = host[skip:], evs[skip:]
    fwd = [e[0].elapsed_time(e[1]) for e in E]; bwd = [e[1].elapsed_time(e[2]) for e in E]
    optm = [e[2].elapsed_time(e[3]) for e in E]
    idle = [evs[i - 1][3].elapsed_time(evs[i][0]) for i in range(skip + 1, len(evs))]
    step_ms = (wall[-1] - wall[skip]) / (len(wall) - 1 - skip) * 1000
    r = {"steps": len(E), "step_ms": round(step_ms, 1), "steps_per_min": round(60000 / step_ms, 1),
         "host_batch_ms": round(1000 * float(np.mean(h)), 1), "gpu_fwd_loss_ms": round(float(np.mean(fwd)), 1),
         "gpu_bwd_ms": round(float(np.mean(bwd)), 1), "gpu_clip_opt_ms": round(float(np.mean(optm)), 1),
         "gpu_idle_between_steps_ms": round(float(np.mean(idle)), 1)}
    r["gpu_idle_frac"] = round(r["gpu_idle_between_steps_ms"] / step_ms, 3)
    r["host_batch_frac"] = round(r["host_batch_ms"] / step_ms, 3)
    print(f"== PROFILE {json.dumps(r)}", flush=True)
    raise SystemExit(0)


def is_eval(s):
    if not a.eval_sparse:
        return s % a.eval_every == 0
    flat, dec = (int(x) for x in a.eval_sparse.split(","))
    d0 = a.steps * (1 - a.decay_frac) if a.lr_shape == "wsd" else a.steps
    return s > 0 and (s % dec == 0 if s >= d0 else s % flat == 0)


if a.batch_audit:   # label-policy audit: same seed => X identical across --eou-min-tail / --eou-amb-keep, only Y differs
    np.random.seed(a.seed)
    hx, hy, pos, tot = hashlib.sha256(), hashlib.sha256(), 0, 0
    for k in range(a.batch_audit):
        X, lens, Y, ylens = batch([pick_row() for _ in range(a.bs)], train=True)
        bx = hashlib.sha256(X.float().cpu().numpy().tobytes() + lens.cpu().numpy().tobytes()).hexdigest()
        by = hashlib.sha256(Y.cpu().numpy().tobytes() + ylens.cpu().numpy().tobytes()).hexdigest()
        hx.update(bx.encode()); hy.update(by.encode())
        last = [int(Y[i, int(ylens[i]) - 1]) for i in range(len(ylens))]
        pos += sum(t == EOU_ID for t in last); tot += len(last)
        if k < 3 or k == a.batch_audit - 1:
            print(f"  audit batch {k}: X {bx[:16]} Y {by[:16]}", flush=True)
    print("== BATCH-AUDIT " + json.dumps({"batches": a.batch_audit, "X_sha256": hx.hexdigest(), "Y_sha256": hy.hexdigest(),
          "eou_target_share": round(pos / tot, 4), "samples": tot, "eou_min_tail": a.eou_min_tail,
          "eou_amb_keep": a.eou_amb_keep}), flush=True)
    raise SystemExit(0)
if a.silence_audit:   # FR-v2 exposure: the silences the model is SHOWN, from the run's own batches (no step, nothing saved)
    np.random.seed(a.seed)
    B = [0.0, 0.8, 1.04, 1.6, 2.5, 1e9]
    lab = ["<0.8", "0.8-1.04", "1.04-1.6", "1.6-2.5", ">2.5"]
    neg_draw = collections.Counter(); neg_n = collections.Counter(); pos = collections.Counter(); hours = 0.0; tot = 0
    for k in range(a.silence_audit):
        X, lens, Y, ylens = batch([pick_row() for _ in range(a.bs)], train=True)
        for i in range(len(lens)):
            x = X[i, : int(lens[i])].float().cpu().numpy(); hours += len(x) / 16000 / 3600; tot += 1
            sm, first, last = speech_runs(x)
            if last < 0:
                continue
            ps = [(kk - jj) * 0.08 for jj, kk in internal_pauses(sm, first, last, min_start=0.0)]
            for d in ps:
                neg_n[lab[np.searchsorted(B, d, side="right") - 1]] += 1
            if ps:
                neg_draw[lab[np.searchsorted(B, max(ps), side="right") - 1]] += 1
            if int(Y[i, int(ylens[i]) - 1]) == EOU_ID:
                pos[lab[np.searchsorted(B, (len(sm) - 1 - last) * 0.08, side="right") - 1]] += 1
    r = {"batches": a.silence_audit, "draws": tot, "audio_h": round(hours, 2), "pause_stretch_prob": a.pause_stretch_prob,
         "pause_stretch_s": a.pause_stretch_s,
         "draws_with_longest_internal_pause_in_bin_%": {b: round(100 * neg_draw[b] / tot, 2) for b in lab},
         "internal_pauses_per_h": {b: round(neg_n[b] / hours, 1) for b in lab},
         "eou_draws_terminal_silence_in_bin_%": {b: round(100 * pos[b] / tot, 2) for b in lab},
         "eou_terminal_silences_per_h": {b: round(pos[b] / hours, 1) for b in lab}}
    print("== SILENCE-AUDIT " + json.dumps(r), flush=True)
    raise SystemExit(0)
prof_host, prof_ev, prof_wall = [], [], []
calib = []; acc_r = acc_k = None
log = []; best = None; t0 = time.time(); step = 0; seen_s = 0.0; max_clip_s = 0.0
extra_evals = {int(x) for x in a.eval_steps.split(",") if x}
set_train_mode()
# --eval-only: a baseline on exactly the val sets / subsample / decoder of the runs it is
# compared with: one pass through the eval block at step 0, nothing trained or saved
order = []
while (not a.eval_only and step < (a.stop_at or a.steps)) or (a.eval_only and not log):
    if a.eval_only:
        loss = torch.zeros(())
    else:
        if a.unfreeze_sched and any(s_ == step for s_, _ in unfreeze):
            cur_k = apply_unfreeze(step); set_train_mode()
            print(f"  step {step}: encoder top-K = {cur_k}", flush=True)
        t_h0 = time.perf_counter()
        if a.mix:
            rs = [pick_row() for _ in range(a.bs)]
        else:   # the 2026-10-09 sampler, bit for bit
            if not order:
                order = list(range(len(rows))); random.shuffle(order)
            rs = [rows[i] for i in order[: a.bs]]; order = order[a.bs:]
        X, lens, Y, ylens = batch(rs, train=True)
        if a.profile_steps:
            prof_host.append(time.perf_counter() - t_h0); ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            ev[0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e, el = encode(X, lens)
            d, _, _ = m.decoder(targets=Y, target_length=ylens)
            j = m.joint(encoder_outputs=e, decoder_outputs=d)
        loss = loss_fn(log_probs=j.float(), targets=Y, input_lengths=el, target_lengths=ylens)
        if ctc_head is not None:
            lp = torch.log_softmax(ctc_head(e.transpose(1, 2).float()), -1).transpose(0, 1)   # [T, B, V + 1]
            ctc = torch.nn.functional.ctc_loss(lp, Y, el, ylens, blank=V, reduction="sum", zero_infinity=True) / len(rs)
            loss_rnnt = loss.detach()
            loss = loss + a.ctc_weight * ctc
        if kd_proj is not None:
            ht = kd_store.pop("t").float(); hs = kd_store.pop("s")
            T_ = min(ht.shape[1], hs.shape[1]); ht, hs = ht[:, :T_], hs[:, :T_]
            vm = (torch.arange(T_, device=dev)[None] < el[:, None]).float()   # valid frames (T equal by construction)
            cs = torch.nn.functional.cosine_similarity(kd_proj(hs.float()), ht, dim=-1)
            kd = ((1 - cs) * vm).sum() / vm.sum().clamp(min=1)
            kd_cos = float(((cs.detach() * vm).sum() / vm.sum().clamp(min=1)))
            kd_ms = kd_feats.pop("ms", 0.0)   # host-side launch time only (async): the real cost is the step time vs the control
            loss_rnnt = loss.detach()
            if a.kd_calib:
                ep = [p_ for p_ in m.encoder.parameters() if p_.requires_grad]
                gk = torch.autograd.grad(kd, ep, retain_graph=True, allow_unused=True)   # small graph first
                gr = torch.autograd.grad(loss, ep, allow_unused=True)
                vr = torch.cat([(g if g is not None else torch.zeros_like(p_)).flatten() for g, p_ in zip(gr, ep)])
                vk = torch.cat([(g if g is not None else torch.zeros_like(p_)).flatten() for g, p_ in zip(gk, ep)])
                acc_r = vr if acc_r is None else acc_r + vr; acc_k = vk if acc_k is None else acc_k + vk
                calib.append((float(vr.norm()), float(vk.norm()), float(torch.nn.functional.cosine_similarity(vr, vk, dim=0)),
                              float(loss_rnnt), float(kd)))
                print(f"  calib batch {len(calib)}: |g_rnnt| {calib[-1][0]:.3f} |g_kd@1| {calib[-1][1]:.4f} "
                      f"ratio {calib[-1][1] / calib[-1][0]:.4f} cos {calib[-1][2]:.4f} rnnt {calib[-1][3]:.2f} kd {calib[-1][4]:.4f}", flush=True)
                if len(calib) >= a.kd_calib:
                    c_ = np.array(calib)
                    print("== KD-CALIB " + json.dumps({"batches": len(c_), "g_rnnt_mean": round(float(c_[:, 0].mean()), 4),
                          "g_kd_at_lambda1_mean": round(float(c_[:, 1].mean()), 5),
                          "ratio_median": round(float(np.median(c_[:, 1] / c_[:, 0])), 5),
                          "ratio_mean_of_means": round(float(c_[:, 1].mean() / c_[:, 0].mean()), 5),
                          "ratio_of_summed_grads": round(float(acc_k.norm() / acc_r.norm()), 5),
                          "grad_cos_summed": round(float(torch.nn.functional.cosine_similarity(acc_r, acc_k, dim=0)), 4),
                          "grad_cos_mean": round(float(c_[:, 2].mean()), 4), "rnnt_mean": round(float(c_[:, 3].mean()), 3),
                          "kd_mean": round(float(c_[:, 4].mean()), 4), "s_layer": a.kd_s_layer, "t_layer": a.kd_t_layer}), flush=True)
                    raise SystemExit(0)
                continue
            loss = loss + kd_lambda(step) * kd
        if a.profile_steps:
            ev[1].record()
        opt.zero_grad(set_to_none=True); loss.backward()
        if a.profile_steps:
            ev[2].record()
        if (step + 1) % 25 == 0:   # encoder gradient norm before clipping (does the auxiliary loss reach the encoder?)
            eg = [p_.grad.norm() for p_ in m.encoder.parameters() if p_.grad is not None]
            enc_gn = float(torch.norm(torch.stack(eg))) if eg else 0.0
            if kd_proj is not None:
                kd_gn = float(torch.norm(torch.stack([p_.grad.norm() for p_ in kd_proj.parameters()])))
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        for gi, g in enumerate(opt.param_groups):
            lr_auc[gi] += g["lr"] if any(p_.grad is not None for p_ in g["params"]) else 0.0
        opt.step(); sched.step(); step += 1; seen_s += float(lens.sum()) / 16000
        if a.profile_steps:
            ev[3].record(); prof_ev.append(ev); prof_wall.append(time.perf_counter())
            if step == a.profile_steps + 20:
                profile_report(prof_host, prof_ev, prof_wall)
        max_clip_s = max(max_clip_s, float(lens.max()) / 16000)
        if step % 25 == 0:
            print(f"  step {step}/{a.steps} loss {loss.item():.3f}"
                  + (f" (rnnt {loss_rnnt.item():.3f} ctc {ctc.item():.3f})" if ctc_head is not None else "")
                  + (f" (rnnt {loss_rnnt.item():.3f} kd {kd.item():.4f} cos {kd_cos:.4f} lambda {kd_lambda(step):.3f} "
                     f"proj-gn {kd_gn:.3f} teacher {kd_ms:.0f} ms)" if kd_proj is not None else "") + f" enc-gn {enc_gn:.2f} audio {seen_s / 3600:.2f} h "
                  f"{seen_s / (time.time() - t0):.0f}x RT peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB "
                  f"reserved {torch.cuda.max_memory_reserved() / 1e9:.1f} GB max-clip {max_clip_s:.1f} s", flush=True)
    if a.eval_only or is_eval(step) or step == a.steps or step == a.stop_at or step in extra_evals:
        tr = rows[:8]
        hy, eo, mg, nb = decode(tr)
        tw, tc = wer_cer([norm(r["text"]) for r in tr], hy)
        rec = {"step": step, "loss": round(loss.item(), 3),
               **({"rnnt": round(loss_rnnt.item(), 3), "kd": round(kd.item(), 4), "kd_cos": round(kd_cos, 4),
                   "kd_lambda": round(kd_lambda(step), 4)} if kd_proj is not None and not a.eval_only else {}), "train8_wer": round(tw, 2), "train8_cer": round(tc, 2),
               "train8_empty": sum(1 for h in hy if not h.strip()), "train8_distinct": len(set(hy)),
               "nonblank_frac": round(float(np.mean(nb)), 4), "margin_max_mean": round(float(np.mean(mg)), 3)}
        print(f"  EVAL {json.dumps(rec)}", flush=True)
        for r, h in list(zip(tr, hy))[:2]:
            print(f"    ref: {norm(r['text'])[:80]}\n    hyp: {h[:80]}", flush=True)
        if val:
            per = []
            for vn, vr in vals:
                vh, veo, _, vnb = decode(vr)
                w_, c_ = wer_cer([norm(r["text"]) for r in vr], vh)
                per.append((vn, w_, c_, sum(1 for h in vh if not h.strip()), veo / len(vr), float(np.mean(vnb))))
            # macro mean over the val sets (= the single set's numbers when there is one)
            vw = float(np.mean([p_[1] for p_ in per])); vc = float(np.mean([p_[2] for p_ in per]))
            rec.update({"val_wer": round(vw, 2), "val_cer": round(vc, 2), "val_empty": sum(p_[3] for p_ in per),
                        "val_eou_rate": round(float(np.mean([p_[4] for p_ in per])), 3),
                        "val_nonblank": round(float(np.mean([p_[5] for p_ in per])), 4)})
            if len(per) > 1:
                for vn, w_, c_, e_, eo_, _ in per:
                    rec.update({f"val_{vn}_wer": round(w_, 2), f"val_{vn}_cer": round(c_, 2), f"val_{vn}_empty": e_,
                                f"val_{vn}_eou": round(eo_, 3)})
            print(f"  VAL {json.dumps({k: rec[k] for k in rec if k.startswith('val')})}", flush=True)
            # stage 2 keeps EOU: best = lowest WER among evals with EOU on >= 80 % of the
            # val clips AND <= 10 % empty; until one qualifies, the most EOU (fewest empties) wins
            ok = rec["val_eou_rate"] >= 0.8 and rec["val_empty"] <= 0.1 * len(val)
            score = vw if not a.eou_append else (vw if ok else 1000 - 100 * rec["val_eou_rate"] + rec["val_empty"])
            if (best is None or score < best) and not a.eval_only and not a.final_eof:
                best = score
                m.save_to(str(out / "final.nemo"))
                torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                            "args": vars(a)}, out / "last.ckpt")
                rec["saved"] = True
        log.append(rec)
        if a.abort_wer and val and step == int(a.abort_wer.split(":")[0]) and rec["val_wer"] > float(a.abort_wer.split(":")[1]):
            rec["aborted"] = True
        json.dump({"args": vars(a), "log": log, "wall_s": round(time.time() - t0, 1), "audio_h_seen": round(seen_s / 3600, 3),
                   "audio_h_per_gpu_h": round(seen_s / (time.time() - t0), 1),
                   "lr_auc": {n_: round(v_, 4) for n_, v_ in zip(group_names, lr_auc)},
                   "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2), "max_clip_s": round(max_clip_s, 2)},
                  open(out / "metrics.json", "w"), indent=1)
        if rec.get("aborted"):
            print(f"== plain-{a.tag} ABORTED at step {step}: macro val WER {rec['val_wer']} > gate {a.abort_wer}", flush=True)
            raise SystemExit(5)
if (not val or a.final_eof) and not a.eval_only:
    m.save_to(str(out / "final.nemo"))   # --final-eof: the pre-registered decision point is the last step
    if a.final_eof:
        print(f"== plain-{a.tag} final.nemo = step {step} (EOF)", flush=True)
print(f"== plain-{a.tag} done {time.time() - t0:.0f}s", flush=True)
