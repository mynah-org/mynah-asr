#!/usr/bin/env python3
"""E0: teacher vs student encoder audit BEFORE any representation-distillation loss (no training here).

    python kd_audit.py --teacher nemotron-3.5-asr-streaming-0.6b.nemo --student parakeet_realtime_eou_120m-v1.nemo \
        --manifests "mls=test.json:6,cv=test_cv.json:6,vp=test_vp.json:6,yg=yg_all.json:6" --out kd_audit.json \
        [--synth 0.4,0.7,1,1.5,2,5,10,20] [--t-layers auto] [--bench 1]

Per model: class, parameter count, preprocessor (sample rate, window / hop, mels, normalisation, dither), encoder
(layers, d_model, subsampling type / factor, att_context_size, conv context, causal downsampling) and the
streaming parameters NeMo derives for it (chunk / shift / pre-encode cache sizes).
Per clip (identical waveform into both models, the training forward: preprocessor -> encoder(audio, length), eval
mode, no dither): mel frames, encoder output frames T for teacher and student, T_teacher - T_student and the ratio,
and the frame count the sample length predicts (mel / subsampling factor), so a mismatch is attributed.
Per hooked encoder layer (student: final; teacher: --t-layers, default 4 spread layers ~1/4, 1/2, 3/4, last):
mean / std / RMS / mean L2 norm per frame / temporal variance (variance over frames, averaged over channels).
Alignability (only if T matches on every clip): a ridge projection student-final (d_s) -> teacher layer (d_t) is
fitted on the frames of half the real clips and scored (frame cosine, R^2) on the other half, at relative frame
shifts -3..+3: the best shift is the empirical offset, the curve's sharpness says how frame-local the mapping is.
This is a DIAGNOSTIC: a high cosine does not prove the layer is a useful target.
--bench 1: teacher and student encoder forward ms for a batch of 16 x 10 s, and peak memory, on the device.
"""
import argparse, json, math, time

import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--teacher", required=True)
ap.add_argument("--student", required=True)
ap.add_argument("--manifests", default="", help="name=manifest:n,... (first n clips <= 20 s of each)")
ap.add_argument("--synth", default="0.4,0.7,1,1.5,2,5,10,20", help="durations (s) of synthetic clips (a real clip "
                "tiled / cut to that length, so the encoders see speech)")
ap.add_argument("--t-layers", default="auto")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--bench", type=int, default=0)
ap.add_argument("--ridge", type=float, default=1e-2, help="ridge on STANDARDISED features (x lambda * N)")
ap.add_argument("--t-contexts", default="", help="teacher att_context_size values to audit, e.g. '56,0;56,3;56,13' "
                "(the alignability and T checks are repeated per value; default = the checkpoint's default only)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf


def load(path):
    m = ASRModel.restore_from(path, map_location=a.device).eval()
    try:
        m.preprocessor.featurizer.dither = 0.0
        m.preprocessor.featurizer.pad_to = 0
    except AttributeError:
        pass
    return m


def describe(m):
    c = OmegaConf.to_container(m.cfg, resolve=True)
    pre, enc = c.get("preprocessor", {}), c.get("encoder", {})
    d = {"class": type(m).__name__, "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
         "preprocessor": {k: pre.get(k) for k in ("sample_rate", "window_size", "window_stride", "window", "features",
                                                    "n_fft", "normalize", "dither", "pad_to", "frame_splicing")},
         "encoder": {k: enc.get(k) for k in ("_target_", "n_layers", "d_model", "feat_in", "subsampling",
                                              "subsampling_factor", "subsampling_conv_channels", "causal_downsampling",
                                              "att_context_size", "att_context_style", "conv_kernel_size",
                                              "conv_context_size", "self_attention_model", "n_heads", "ff_expansion_factor")},
         "encoder_att_context_size_live": getattr(m.encoder, "att_context_size", None),
         "n_encoder_layers_live": len(m.encoder.layers)}
    try:
        m.encoder.setup_streaming_params()
        sc = m.encoder.streaming_cfg
        d["streaming_cfg"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in vars(sc).items()} \
            if hasattr(sc, "__dict__") else str(sc)
    except Exception as e:  # not every encoder is cache-aware
        d["streaming_cfg"] = f"n/a ({type(e).__name__}: {e})"
    return d


def hooks(m, idx):
    store = {}
    hs = [m.encoder.layers[i].register_forward_hook(
        lambda mod, inp, out, i=i: store.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach())) for i in idx]
    return store, hs


@torch.inference_mode()
def run(m, x, store):
    store.clear()
    X = torch.from_numpy(x)[None].to(a.device)
    L = torch.tensor([len(x)], device=a.device)
    f, fl = m.preprocessor(input_signal=X, length=L)
    e, el = m.encoder(audio_signal=f, length=fl)
    T = int(el[0])
    mel = f[0, :, :int(fl[0])].float().cpu().numpy().T   # frames x mels
    pad = np.zeros((T * 8, mel.shape[1]), np.float32); pad[:min(len(mel), T * 8)] = mel[:T * 8]
    store["_mel8"] = pad.reshape(T, -1)
    lay = {i: (v if isinstance(v, np.ndarray) else v[0, :T].float().cpu().numpy()) for i, v in store.items()}
    return int(fl[0]), T, int(e.shape[-1]), lay


def lstats(h):   # h: T x D
    return {"mean": float(h.mean()), "std": float(h.std()), "rms": float(np.sqrt((h ** 2).mean())),
            "frame_l2": float(np.linalg.norm(h, axis=1).mean()), "temporal_var": float(h.var(0).mean())}


def ridge_fit(Xs, Ys, lam):   # features standardised per channel on the TRAIN frames
    X = np.concatenate(Xs).astype(np.float64); Y = np.concatenate(Ys).astype(np.float64)
    mx, sx, my, sy = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6
    X = (X - mx) / sx; Y = (Y - my) / sy
    W = np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]), X.T @ Y)
    return W, (mx, sx, my, sy)


def score(W, st, Xs, Ys):   # held out: R^2 of the standardised target, cosine of the CENTRED frames
    mx, sx, my, sy = st
    X = (np.concatenate(Xs) - mx) / sx; Y = (np.concatenate(Ys) - my) / sy
    P = X @ W
    cos = (P * Y).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Y, axis=1) + 1e-9)
    r2 = 1 - ((Y - P) ** 2).sum() / (Y ** 2).sum()
    return float(cos.mean()), float(r2)


def shifted(s, t, k):   # pair student frame i with teacher frame i + k
    T = min(len(s), len(t))
    if k >= 0:
        return s[:T - k], t[k:T]
    return s[-k:T], t[:T + k]


T_m, S_m = load(a.teacher), load(a.student)
rep = {"teacher": describe(T_m), "student": describe(S_m)}
nt, ns = len(T_m.encoder.layers), len(S_m.encoder.layers)
t_layers = sorted({max(0, round(nt * q) - 1) for q in (0.25, 0.5, 0.75, 1.0)}) if a.t_layers == "auto" else \
    [int(x) for x in a.t_layers.split(",")]
s_layers = sorted({max(0, round(ns * q) - 1) for q in (0.5, 1.0)})
rep["t_layers"], rep["s_layers"] = t_layers, s_layers
Ts, Th = hooks(T_m, t_layers)
Ss, Sh = hooks(S_m, s_layers)
sr = rep["student"]["preprocessor"]["sample_rate"] or 16000
assert sr == (rep["teacher"]["preprocessor"]["sample_rate"] or 16000), "sample rates differ: resampling needed"
sub_t = rep["teacher"]["encoder"]["subsampling_factor"] or 8
sub_s = rep["student"]["encoder"]["subsampling_factor"] or 8

clips = []
for spec in (x for x in a.manifests.split(",") if x):
    name, rest = spec.split("=", 1); path, n = rest.rsplit(":", 1); k = 0
    for l in open(path):
        r = json.loads(l)
        if r.get("duration", 0) <= 20:
            x, s = sf.read(r["audio_filepath"], dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            assert s == sr, f"{r['audio_filepath']}: {s} Hz"
            clips.append((f"{name}:{k}", x)); k += 1
            if k >= int(n):
                break
if clips:
    base = np.concatenate([x for _, x in clips[:6]])
    for d in (float(v) for v in a.synth.split(",") if v):
        n = int(round(d * sr)); x = np.tile(base, math.ceil(n / len(base)))[:n]
        clips.append((f"synth:{d:g}s", x.astype(np.float32)))


def set_ctx(m, ctx):
    if ctx is not None:
        m.encoder.set_default_att_context_size(ctx)


def audit(ctx):
    set_ctx(T_m, ctx)
    per, pairs = [], []
    for name, x in clips:
        mt, tt, dt, lt = run(T_m, x, Ts)
        ms, ts, ds, ls = run(S_m, x, Ss)
        per.append({"clip": name, "sec": round(len(x) / sr, 3), "samples": len(x), "mel_t": mt, "mel_s": ms,
                    "T_teacher": tt, "T_student": ts, "dT": tt - ts, "ratio": round(tt / max(1, ts), 4),
                    "pred_T_teacher": math.ceil(mt / sub_t), "pred_T_student": math.ceil(ms / sub_s),
                    "teacher_layers": {str(i): lstats(v) for i, v in lt.items() if i != "_mel8"},
                    "student_layers": {str(i): lstats(v) for i, v in ls.items() if i != "_mel8"}})
        if not name.startswith("synth"):
            pairs.append((ls, lt))
    r = {"att_context_teacher": ctx or rep["teacher"]["encoder_att_context_size_live"], "clips": per,
         "T_equal_all": all(p["dT"] == 0 for p in per), "dT_set": sorted({p["dT"] for p in per})}
    print(f"== teacher att_context {r['att_context_teacher']}: T equal on all {len(per)} clips: {r['T_equal_all']} "
          f"(dT {r['dT_set']}); T at 1/5/10/20 s: " + ", ".join(f"{p['T_teacher']}/{p['T_student']}" for p in per
                                                           if p["clip"] in ("synth:1s", "synth:5s", "synth:10s", "synth:20s")), flush=True)
    if pairs and r["T_equal_all"]:
        tr, te = pairs[::2], pairs[1::2]; al = {}
        for src in ["_mel8"] + s_layers:   # mel8 = 8 stacked mel frames: the acoustic baseline a student must beat
            for tl in t_layers:
                row = {}
                for k in range(-4, 5):
                    Xa, Ya = zip(*[shifted(s_[src], t[tl], k) for s_, t in tr])
                    Xb, Yb = zip(*[shifted(s_[src], t[tl], k) for s_, t in te])
                    W, st = ridge_fit(Xa, Ya, a.ridge)
                    c, r2 = score(W, st, Xb, Yb)
                    row[str(k)] = {"cos": round(c, 4), "r2": round(r2, 4)}
                al[f"{'mel8' if src == '_mel8' else 'student' + str(src)}->teacher{tl}"] = row
                best = max(row, key=lambda k: row[k]["r2"])
                print(f"  {'mel8' if src == '_mel8' else 'student L' + str(src):>11} -> teacher L{tl:<2} R2 by shift "
                      + " ".join(f"{k}:{v['r2']:.3f}" for k, v in row.items()) + f" | best {best} cos {row[best]['cos']:.3f}", flush=True)
        r["alignability"] = al
        r["alignability_note"] = (f"standardised ridge {a.ridge}, fitted on the frames of {len(tr)} clips, scored on {len(te)}; "
                                  "key = teacher frame shift k (student frame i <-> teacher frame i + k); centred cosine / R^2 held out")
    return r


ctxs = [None] + [[int(v) for v in c.split(",")] for c in a.t_contexts.split(";") if c]
rep["audits"] = [audit(c) for c in ctxs]
set_ctx(T_m, ctxs[0] or rep["teacher"]["encoder_att_context_size_live"])
rep["T_equal_all"] = all(r["T_equal_all"] for r in rep["audits"])
rep["dT_set"] = sorted({d for r in rep["audits"] for d in r["dT_set"]})

if a.bench and a.device == "cuda":
    for tag, m in (("teacher", T_m), ("student", S_m)):
        X = torch.randn(16, 10 * sr, device="cuda") * 0.05; L = torch.full((16,), 10 * sr, device="cuda")
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for amp in (False, True):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    for _ in range(2):   # warm-up
                        f, fl = m.preprocessor(input_signal=X, length=L); m.encoder(audio_signal=f, length=fl)
                    torch.cuda.synchronize(); t0 = time.time()
                    for _ in range(5):
                        f, fl = m.preprocessor(input_signal=X, length=L); m.encoder(audio_signal=f, length=fl)
                    torch.cuda.synchronize()
                rep.setdefault("bench_16x10s", {})[f"{tag}_{'bf16' if amp else 'fp32'}_ms"] = round((time.time() - t0) / 5 * 1000, 1)
        rep["bench_16x10s"][f"{tag}_peak_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print("  bench", rep["bench_16x10s"], flush=True)
dt_ = rep["teacher"]["encoder"]["d_model"] or 1024
hop = rep["teacher"]["preprocessor"]["window_stride"] or 0.01
rep["precompute_GB_per_100h_one_layer_bf16"] = round(100 * 3600 / (hop * sub_t) * dt_ * 2 / 2**30, 1)
for h in Th + Sh:
    h.remove()
json.dump(rep, open(a.out, "w"), indent=1, default=str)
print(json.dumps({k: rep[k] for k in ("t_layers", "s_layers", "T_equal_all", "dT_set",
                                       "precompute_GB_per_100h_one_layer_bf16")}))
print(json.dumps({"teacher": rep["teacher"], "student": rep["student"]}, default=str)[:4000])
print("== KD-AUDIT-DONE", a.out, flush=True)
