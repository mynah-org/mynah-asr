#!/usr/bin/env python3
"""Stage-2 (<EOU>) model decoded the way Mynah serves it: decoder reset after every <EOU> (CPU, offline, no training).

    s2_reset.py --parent plain/final.nemo --s2 s2/final.nemo --vals "mls=a.json,..." [--per-set 200] [--out r.json]

Mynah (docs/prior-art.md, issue #13) resets ONLY the decoder after a model <EOU> (prediction-network state to zero,
last token -> SOS), never the encoder cache. Here: one offline encoder pass per clip, then a frame-synchronous RNNT
greedy decode with three protocols on the same clips:
  parent        the plain parent (no <EOU> in its outputs);
  s2            stage 2, no reset (the NeMo offline protocol of the probes);
  s2_reset      stage 2, decoder reset after every <EOU> (the Mynah serving protocol); the transcript is the
                concatenation of the segments.
Per set: WER / CER, S / D / I, end-of-utterance deletions, <EOU> per clip, clips split into >= 2 segments, and
premature <EOU> = emitted while >= 0.5 s of speech remains (energy endpoint: last 80 ms frame within 35 dB of the
clip peak), i.e. a false endpoint at an internal pause. Words recovered = s2 errors - s2_reset errors.
Two separate quantities: ASR continuity after the reset, and EOU precision (false endpoint rate).
"""
import argparse
import json
import re
import unicodedata

import numpy as np
import soundfile as sf
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
import textnorm  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--parent", required=True)
ap.add_argument("--s2", required=True)
ap.add_argument("--vals", required=True)
ap.add_argument("--per-set", type=int, default=200)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--post-pad-s", type=float, default=0.0, help="quiet (-75 dB noise, as the in-run stage-2 eval) appended "
                "so a true end-of-utterance <EOU> can fire; endpoints are measured on the ORIGINAL clip")
ap.add_argument("--models", default="parent,s2,s2_reset", help="subset of protocols to run")
ap.add_argument("--eou-bias", type=float, default=0.0, help="decision-layer calibration of the stage-2 model (s2 / s2_reset "
                "only): z_EOU' = z_EOU + b on the joint output before the greedy argmax, at every symbol step; blank, text, "
                "decoder state, encoder and reset semantics unchanged (the parent never sees it)")
ap.add_argument("--margin-rule", type=float, default=0.0, help="scale rule only (no biased decode): on the UNBIASED s2_reset "
                "path, M_t = max over the frame's symbol steps of z_EOU - max(other outputs); per clip need = -max M_t over "
                "frames in [speech end, speech end + W s); prints need p10/p50/p90 and b = max(0, median need) rounded UP "
                "to 0.05, then exits (no FE / latency / WER is computed)")
ap.add_argument("--dump", default="", help="also write one JSON line per (clip, protocol): audio, ref, hyp, token ids, "
                "<EOU> frames, speech end and per-clip flags (artifact only; no computation changes)")
ap.add_argument("--out", default="")
ap.add_argument("--lang", default="it", help="it = the frozen s2_reset.norm (RC-IT protocol); other languages use "
                "common/textnorm.norm(t, lang) (fr: oe / ae and typographic apostrophes, then the same rule)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
FR = 0.08


def norm(t):
    if a.lang != "it":
        return textnorm.norm(t, a.lang)
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def ops(r, h):
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            out.append("C" if r[i - 1] == h[j - 1] else "S"); i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            out.append("D"); i -= 1
        else:
            out.append("I"); j -= 1
    return out[::-1]


def cer(r, h):
    r, h = list(r), list(h)
    p = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        c = [i]
        for j, y in enumerate(h, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


from nemo.collections.asr.models import EncDecRNNTBPEModel

P = EncDecRNNTBPEModel.restore_from(a.parent, map_location="cpu").eval()
S = EncDecRNNTBPEModel.restore_from(a.s2, map_location="cpu").eval()
tk = S.tokenizer
EOU, BLANK = tk.token_to_id("<EOU>"), tk.vocab_size


def encode(m, x):
    f, fl = m.preprocessor(input_signal=torch.from_numpy(x)[None], length=torch.tensor([len(x)]))
    e, el = m.encoder(audio_signal=f, length=fl)
    return e[0, :, : int(el[0])].T


@torch.no_grad()
def greedy(m, enc, reset=False, max_symbols=10, bias=0.0, margins=None):
    ys, frames = [], []
    g, state = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
    for t in range(enc.shape[0]):
        f = enc[t][None, None, :]
        for _ in range(max_symbols):
            z = m.joint.joint(f, g)[0, 0, 0]
            if margins is not None:   # z_EOU - max(other outputs), max over the frame's symbol steps
                oth = torch.cat([z[:EOU], z[EOU + 1:]]).max()
                margins[t] = max(margins.get(t, -1e9), float(z[EOU] - oth))
            if bias:
                z = z.clone(); z[EOU] += bias
            k = int(z.argmax())
            if k == BLANK:
                break
            ys.append(k); frames.append(t)
            if reset and k == EOU:   # Mynah: decoder state to zero, last token -> SOS; encoder untouched
                g, state = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
                break
            g, state = m.decoder.predict(torch.tensor([[k]]), state=state, add_sos=False, batch_size=1)[:2]
    return ys, frames


def speech_mask(x):
    n = int(FR * 16000)
    f = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    return db > db.max() - 35


def speech_end(x):
    on = np.nonzero(speech_mask(x))[0]
    return (on[-1] + 1) * FR if len(on) else len(x) / 16000


if a.margin_rule:   # scale rule for --eou-bias (pre-registered): no biased decode, no outcome metric
    need = []
    for spec in a.vals.split(","):
        sname, p = spec.split("=", 1)
        rows = [json.loads(l) for l in open(p)]
        rows = rows[:: max(1, len(rows) // a.per_set)][: a.per_set]
        for r in rows:
            x, _ = sf.read(r["audio_filepath"], dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            end = speech_end(x)
            if a.post_pad_s:
                x = np.concatenate([x, (np.random.RandomState(len(x)).randn(int(a.post_pad_s * 16000)) * 10 ** (-75 / 20)).astype(np.float32)])
            mg = {}
            greedy(S, encode(S, x), reset=True, margins=mg)
            w = [v for t, v in mg.items() if end <= t * FR < end + a.margin_rule]
            if w:
                need.append(-max(w))
    q = [round(float(np.percentile(need, v)), 3) for v in (10, 50, 90)]
    b = max(0.0, float(np.ceil(np.median(need) / 0.05) * 0.05))
    print("== MARGIN-RULE " + json.dumps({"W_s": a.margin_rule, "clips": len(need), "need_p10_p50_p90": q, "b": round(b, 2)}), flush=True)
    raise SystemExit(0)

res = {}
dump = open(a.dump, "w") if a.dump else None
for spec in a.vals.split(","):
    sname, p = spec.split("=", 1)
    rows = [json.loads(l) for l in open(p)]
    rows = rows[:: max(1, len(rows) // a.per_set)][: a.per_set]
    want = a.models.split(",")
    acc = {k: dict(S=0, D=0, I=0, N=0, D_end=0, cerr=0, cN=0, eou=0, split=0, premature=0, prem_clips=0, clips=0,
                   true_eou=0, lat=[], empty=0, fe_any=0) for k in want}
    for r in rows:
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        end = speech_end(x)
        sm = speech_mask(x)   # of the ORIGINAL clip (before any post-pad)
        if a.post_pad_s:
            x = np.concatenate([x, (np.random.RandomState(len(x)).randn(int(a.post_pad_s * 16000)) * 10 ** (-75 / 20)).astype(np.float32)])
        ref = norm(r["text"]).split()
        eP = encode(P, x) if "parent" in want else None
        eS = encode(S, x) if want != ["parent"] else None
        for k, (m, e, rs) in {"parent": (P, eP, False), "s2": (S, eS, False), "s2_reset": (S, eS, True)}.items():
            if k not in want:
                continue
            ys, fr = greedy(m, e, reset=rs, bias=a.eou_bias if k != "parent" else 0.0)
            hyp = norm(tk.ids_to_text([t for t in ys if t < 1024])).split()
            o = ops(ref, hyp)
            c = acc[k]
            for op in o:
                if op != "C":
                    c[op] += 1
            j = 0
            for op in reversed(o):
                if op == "D":
                    j += 1
                elif op in ("C", "S"):
                    break
            c["D_end"] += j; c["N"] += len(ref); c["clips"] += 1
            c["cerr"] += cer(" ".join(ref), " ".join(hyp)); c["cN"] += len(" ".join(ref))
            ef = [f for y, f in zip(ys, fr) if y == EOU]
            c["eou"] += len(ef); c["split"] += len(ef) >= 2
            prem = [f for f in ef if f * FR < end - 0.5]
            c["premature"] += len(prem); c["prem_clips"] += bool(prem)
            true = [f * FR for f in ef if f * FR >= end - 0.5]   # first <EOU> at / after the speech end (-0.5 s slack)
            if true:
                c["true_eou"] += 1; c["lat"].append(true[0] - end)
            c["empty"] += not hyp
            # FE_any (diagnostic, fixed 2026-10-18 before v5-B): an <EOU> followed by >= 0.16 s (2 frames) of speech
            c["fe_any"] += any(int(sm[f + 1:].sum()) >= 2 for f in ef)
            if dump:
                dump.write(json.dumps({"set": sname, "protocol": k, "audio": r["audio_filepath"], "ref": " ".join(ref),
                                       "hyp": " ".join(hyp), "ids": ys, "eou_frames": ef, "speech_end_s": round(end, 3),
                                       "premature": bool(prem), "fe_any": any(int(sm[f + 1:].sum()) >= 2 for f in ef),
                                       "true_eou_s": true[0] if true else None}, ensure_ascii=False) + "\n")
    out = {}
    for k, c in acc.items():
        err = c["S"] + c["D"] + c["I"]
        out[k] = {"wer": round(100 * err / c["N"], 2), "cer": round(100 * c["cerr"] / c["cN"], 2),
                  **{f"{o}%": round(100 * c[o] / c["N"], 2) for o in ("S", "D", "I", "D_end")},
                  "errors": err, "eou_per_clip": round(c["eou"] / c["clips"], 3), "clips_split": c["split"],
                  "premature_eou": c["premature"], "clips_with_premature": c["prem_clips"], "clips": c["clips"],
                  "false_endpoint_clip%": round(100 * c["prem_clips"] / c["clips"], 1),
                  "true_eou_recall%": round(100 * c["true_eou"] / c["clips"], 1), "empty": c["empty"],
                  "FE_any_clip%": round(100 * c["fe_any"] / c["clips"], 1), "_feany": c["fe_any"],
                  "eou_latency_s_p50_p90": [round(float(np.percentile(c["lat"], q)), 2) for q in (50, 90)] if c["lat"] else None,
                  "_lat": c["lat"], "_prem": c["prem_clips"], "_true": c["true_eou"]}
    if "s2" in out and "s2_reset" in out:
        out["words_recovered_by_reset"] = out["s2"]["errors"] - out["s2_reset"]["errors"]
    res[sname] = out
    print("==", sname, json.dumps({k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
                                    for k, v in out.items()}), flush=True)
macro = {}
for k in want:
    lat = [x for s_ in res for x in res[s_][k]["_lat"]]
    n = sum(res[s_][k]["clips"] for s_ in res)
    macro[k] = {"wer_macro": round(float(np.mean([res[s_][k]["wer"] for s_ in res])), 2),
                "cer_macro": round(float(np.mean([res[s_][k]["cer"] for s_ in res])), 2),
                "false_endpoint_clip%": round(100 * sum(res[s_][k]["_prem"] for s_ in res) / n, 2),
                "FE_any_clip%": round(100 * sum(res[s_][k]["_feany"] for s_ in res) / n, 2),
                "true_eou_recall%": round(100 * sum(res[s_][k]["_true"] for s_ in res) / n, 2),
                "eou_latency_s_p50_p90": [round(float(np.percentile(lat, q)), 2) for q in (50, 90)] if lat else None,
                "empty": sum(res[s_][k]["empty"] for s_ in res), "clips": n}
print("== MACRO", json.dumps(macro), flush=True)
for s_ in res:
    for k in want:
        for kk in ("_lat", "_prem", "_true", "_feany"):
            res[s_][k].pop(kk, None)
if a.out:
    json.dump({"sets": res, "macro": macro}, open(a.out, "w"), indent=1)
