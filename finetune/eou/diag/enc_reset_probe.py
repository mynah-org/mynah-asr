#!/usr/bin/env python3
"""Which reset semantics explains the NeMo s2_reset.py vs Mynah divergence after a premature <EOU>? (CPU, no training)

    enc_reset_probe.py --s2 s2/final.nemo --parity parity.json --work W [--controls 60] --out probe.json

NeMo's own cache-aware streaming encoder (encoder.cache_aware_stream_step, the model's streaming_cfg chunks: first
chunk, then steady chunks with the pre-encode mel cache), frame-synchronous RNNT greedy (max 10 symbols), on the
PCM16 bytes Mynah was fed (W/clips.jsonl). Protocols:
  dec_reset    decoder reset right after the <EOU> symbol, encoder cache untouched (= s2_reset.py, streamed);
  full_reset   Mynah (src/mynah_asr.c stream_model_eou, NeMo voice-agent streaming_asr.py reset_state): the chunk
               holding the <EOU> is decoded to its end, THEN encoder caches back to get_initial_cache_state and the
               decoder to SOS; the chunk cadence is kept (the next chunk is a steady one);
  full_reset_z full_reset with the next chunk's pre-encode mel cache zeroed too (Mynah's subsampling state reset).
Clips: every clip parity.json lists as divergent for the run --run, plus --controls clips that matched exactly.
Reported per protocol and class: transcript / <EOU>-frame equality with Mynah and with the offline NeMo dump.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
import textnorm  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--s2", required=True)
ap.add_argument("--parity", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--nemo-dump", required=True)
ap.add_argument("--mynah", required=True, help="the Mynah run jsonl (cli.jsonl)")
ap.add_argument("--run", default="cli")
ap.add_argument("--controls", type=int, default=60)
ap.add_argument("--threads", type=int, default=20)
ap.add_argument("--lang", default="fr")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)

from nemo.collections.asr.models import ASRModel  # noqa: E402

m = ASRModel.restore_from(a.s2, map_location="cpu").eval()
m.encoder.setup_streaming_params()
cfg = m.encoder.streaming_cfg
print("== streaming_cfg", cfg, flush=True)
EOU, BLANK = 1024, m.joint.num_classes_with_blank - 1
tk = m.tokenizer


def first(v):
    return v[0] if isinstance(v, list) else v


def steady(v):
    return v[1] if isinstance(v, list) else v


@torch.no_grad()
def run(x, proto):
    sig = torch.from_numpy(x)[None]
    feats, fl = m.preprocessor(input_signal=sig, length=torch.tensor([len(x)]))
    T = int(fl[0]); feats = feats[:, :, :T]
    c_ch, c_t, c_len = m.encoder.get_initial_cache_state(batch_size=1)
    g, st = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
    ys, frames, idx, is_first, zero_pre, fbase = [], [], 0, True, False, 0
    while idx < T:
        cs = first(cfg.chunk_size) if is_first else steady(cfg.chunk_size)
        sh = first(cfg.shift_size) if is_first else steady(cfg.shift_size)
        pc = first(cfg.pre_encode_cache_size) if is_first else steady(cfg.pre_encode_cache_size)
        chunk = feats[:, :, idx: idx + cs]
        if is_first:
            pre = torch.zeros(1, feats.shape[1], pc)
        else:
            pre = feats[:, :, max(0, idx - pc): idx]
            if pre.shape[-1] < pc:
                pre = torch.cat([torch.zeros(1, feats.shape[1], pc - pre.shape[-1]), pre], -1)
            if zero_pre:
                pre = torch.zeros_like(pre)
        inp = torch.cat([pre, chunk], -1)
        last = idx + sh >= T
        enc, el, c_ch, c_t, c_len = m.encoder.cache_aware_stream_step(
            processed_signal=inp, processed_signal_length=torch.tensor([inp.shape[-1]]), cache_last_channel=c_ch,
            cache_last_time=c_t, cache_last_channel_len=c_len, keep_all_outputs=last,
            drop_extra_pre_encoded=0 if is_first else cfg.drop_extra_pre_encoded)
        enc = enc[0, :, : int(el[0])].T
        hit = False
        for t in range(enc.shape[0]):
            f = enc[t][None, None, :]
            for _ in range(10):
                k = int(m.joint.joint(f, g)[0, 0, 0].argmax())
                if k == BLANK:
                    break
                ys.append(k); frames.append(fbase + t)
                if k == EOU and proto == "dec_reset":
                    g, st = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
                    break
                hit |= k == EOU
                g, st = m.decoder.predict(torch.tensor([[k]]), state=st, add_sos=False, batch_size=1)[:2]
        fbase += enc.shape[0]
        zero_pre = False
        if hit and proto.startswith("full_reset"):
            c_ch, c_t, c_len = m.encoder.get_initial_cache_state(batch_size=1)
            g, st = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
            zero_pre = proto == "full_reset_z"
        idx += sh; is_first = False
    hyp = textnorm.norm(tk.ids_to_text([t for t in ys if t < 1024]), a.lang)
    return hyp, [f for y, f in zip(ys, frames) if y == EOU]


par = json.load(open(a.parity))["runs"][a.run]
div = {d["wav"]: d for d in par["divergent"]}
cl = [json.loads(l) for l in open(Path(a.work) / "clips.jsonl")]
nemo = {d["audio"]: d for d in map(json.loads, open(a.nemo_dump)) if d["protocol"] == "s2_reset"}
my = {d["wav"]: d for d in map(json.loads, open(a.mynah))}
ctrl = [c for c in cl if c["wav"] not in div]
random.Random(0).shuffle(ctrl)
pick = [c for c in cl if c["wav"] in div] + ctrl[: a.controls]
rows = []
for c in pick:
    x, _ = sf.read(c["wav"], dtype="float32")
    mh = textnorm.norm(my[c["wav"]]["text"], a.lang)
    mf = [int(round((t - 0.01) / 0.08)) for t in my[c["wav"]]["eou_t"]]
    r = {"wav": c["wav"], "set": c["set"], "class": div[c["wav"]]["class"] if c["wav"] in div else "MATCH"}
    for p in ("dec_reset", "full_reset", "full_reset_z"):
        h, f = run(x, p)
        r[p] = {"eq_mynah_text": h == mh, "eq_mynah_eou": f == mf, "eq_offline_text": h == nemo[c["wav"]]["hyp"],
                "eq_offline_eou": f == nemo[c["wav"]]["eou_frames"]}
    rows.append(r)
    print(json.dumps({k: v for k, v in r.items() if k != "wav"}), flush=True)
summ = {}
for cls in ("POST_RESET", "PRE_EOU", "MATCH"):
    rs = [r for r in rows if r["class"] == cls]
    if not rs:
        continue
    summ[cls] = {"clips": len(rs), **{p: {k: sum(r[p][k] for r in rs) for k in rs[0][p]} for p in ("dec_reset", "full_reset", "full_reset_z")}}
print("== SUMMARY", json.dumps(summ), flush=True)
json.dump({"streaming_cfg": str(cfg), "summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
