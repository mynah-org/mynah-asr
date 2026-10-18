#!/usr/bin/env python3
"""Why a stage-2 (<EOU>) model stops emitting before the end of a clip (CPU, offline, no training).

    s2_frames.py --parent plain/final.nemo --s2 s2/final.nemo --vals "mls=a.json,fleurs=b.json,vp=c.json" \
        [--per-set 200] [--out frames.json]

1. Component swap (greedy WER, end-of-utterance deletions) on the same clips: parent, s2, parent encoder + s2
   decoder/joint, s2 encoder + parent decoder/joint -> is the damage in the encoder or in decoder / joint?
2. Pathological clips (s2 deletes >= 3 final reference words, the parent none): a hand-written frame-synchronous
   RNNT greedy decode records per 80 ms frame P(blank), P(<EOU>), the best text-token probability and the
   blank - best-text log margin at the first symbol step of the frame, plus the frame's energy. Reported around
   the frame where s2 emitted its last text token (stop): the pause at the stop (low-energy run), and the same
   statistics of s2 and of the parent in the window BEFORE the stop (2 s) and AFTER it (to the end of the clip),
   with the number of words the parent emitted after s2's stop.
Questions: does the stop follow an internal pause (blank persistence after silence)? does <EOU> rise without being
emitted? does the text probability collapse (the encoder still carries the words but decoder / joint sit in a
blank attractor), or does the encoder lose them (swap)?
"""
import argparse
import copy
import json
import re
import unicodedata

import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--parent", required=True)
ap.add_argument("--s2", required=True)
ap.add_argument("--vals", required=True)
ap.add_argument("--per-set", type=int, default=200)
ap.add_argument("--min-missing", type=int, default=3)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="")
a = ap.parse_args()
torch.set_num_threads(a.threads)
FR = 0.08   # encoder frame (x8 subsampling of 10 ms)


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def trailing_del(r, h):
    """number of final reference words missing (alignment backtrace: D ops after the last C / S)."""
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    i, j, k = n, m, 0
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            break
        if i and d[i, j] == d[i - 1, j] + 1:
            k += 1; i -= 1
        else:
            j -= 1
    return k, int(d[n, m])


from nemo.collections.asr.models import EncDecRNNTBPEModel

P = EncDecRNNTBPEModel.restore_from(a.parent, map_location="cpu").eval()
S = EncDecRNNTBPEModel.restore_from(a.s2, map_location="cpu").eval()
tk = S.tokenizer
EOU, EOB, BLANK = tk.token_to_id("<EOU>"), tk.token_to_id("<EOB>"), tk.vocab_size


def hybrid(enc_from, dj_from):
    h = copy.deepcopy(dj_from)
    h.encoder.load_state_dict(enc_from.encoder.state_dict())
    return h.eval()


MODELS = {"parent": P, "s2": S, "encP+djS2": hybrid(P, S), "encS2+djP": hybrid(S, P)}


def load(r):
    x, _ = sf.read(r["audio_filepath"], dtype="float32")
    return x.mean(1) if x.ndim > 1 else x


def encode(m, x):
    X = torch.from_numpy(x)[None]
    f, fl = m.preprocessor(input_signal=X, length=torch.tensor([len(x)]))
    e, el = m.encoder(audio_signal=f, length=fl)
    return e[0, :, : int(el[0])].T   # [T, D]


@torch.no_grad()
def greedy(m, enc, max_symbols=10, trace=False):
    """frame-synchronous RNNT greedy; returns token ids, emission frames and (trace) per-frame stats."""
    ys, frames, stats = [], [], []
    g, state = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
    for t in range(enc.shape[0]):
        f = enc[t][None, None, :]
        for s in range(max_symbols):
            lp = m.joint.joint(f, g)[0, 0, 0].log_softmax(-1)
            if trace and s == 0:
                text = lp[:1024]
                bt = float(text.max())
                stats.append({"blank": float(lp[BLANK].exp()), "eou": float(lp[EOU].exp()), "text": float(np.exp(bt)),
                              "margin": float(lp[BLANK]) - bt})
            k = int(lp.argmax())
            if k == BLANK:
                break
            ys.append(k); frames.append(t)
            g, state = m.decoder.predict(torch.tensor([[k]]), state=state, add_sos=False, batch_size=1)[:2]
    return ys, frames, stats


def energy(x):
    n = int(FR * 16000)
    f = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    return db - db.max()


sets = []
for spec in a.vals.split(","):
    n, p = spec.split("=", 1)
    rs = [json.loads(l) for l in open(p)]
    sets.append((n, rs[:: max(1, len(rs) // a.per_set)][: a.per_set]))

swap = {k: {} for k in MODELS}
patho = []
for sname, rows in sets:
    acc = {k: [0, 0, 0] for k in MODELS}   # errors, ref words, trailing deletions
    for r in rows:
        x = load(r)
        ref = norm(r["text"]).split()
        out = {}
        for k, m in MODELS.items():
            ys, fr, _ = greedy(m, encode(m, x))
            hyp = norm(tk.ids_to_text([t for t in ys if t < 1024])).split()
            td, err = trailing_del(ref, hyp)
            acc[k][0] += err; acc[k][1] += len(ref); acc[k][2] += td
            out[k] = (ys, fr, td)
        if out["s2"][2] >= a.min_missing and out["parent"][2] == 0:
            patho.append((sname, r, x, out))
    for k in MODELS:
        e, n, td = acc[k]
        swap[k][sname] = {"wer": round(100 * e / n, 2), "D_end%": round(100 * td / n, 2)}
    print("== swap", sname, {k: swap[k][sname] for k in MODELS}, flush=True)

report = []
for sname, r, x, out in patho:
    en = energy(x)
    ps = {}
    for k in ("parent", "s2"):
        m = MODELS[k]
        ys, fr, st = greedy(m, encode(m, x), trace=True)
        ps[k] = (ys, fr, st)
    ys, fr, st = ps["s2"]
    txt = [f for y, f in zip(ys, fr) if y < 1024]
    stop = txt[-1] if txt else 0
    T = len(st)
    q = en[: T] < -35 if len(en) >= T else np.pad(en, (0, T - len(en)), constant_values=0)[:T] < -35
    run = 0   # low-energy run starting within 0.5 s after the stop
    for t in range(stop + 1, min(T, stop + 1 + int(0.5 / FR) + 25)):
        if q[t]:
            run += 1
        elif run:
            break
    def win(stats, lo, hi):
        w = stats[max(0, lo): max(0, hi)]
        if not w:
            return None
        return {k: round(float(np.mean([s[k] for s in w])), 3) for k in ("blank", "eou", "text", "margin")} | \
               {"text>0.1": round(float(np.mean([s["text"] > 0.1 for s in w])), 3), "eou_max": round(max(s["eou"] for s in w), 3)}
    before, after = (stop - int(2 / FR), stop), (stop + 1, T)
    pys, pfr, pst = ps["parent"]
    parent_after = sum(1 for y, f in zip(pys, pfr) if y < 1024 and f > stop)
    s2_eou = [f for y, f in zip(ys, fr) if y == EOU]
    rec = {"set": sname, "dur": round(len(x) / 16000, 2), "missing_end_words": out["s2"][2], "stop_s": round(stop * FR, 2),
           "tail_s": round((T - 1 - stop) * FR, 2), "pause_after_stop_s": round(run * FR, 2),
           "speech_frames_after_stop": int((~q[stop + 1:]).sum()), "parent_tokens_after_stop": parent_after,
           "s2_eou_frames": [round(f * FR, 2) for f in s2_eou],
           "s2_before": win(st, *before), "s2_after": win(st, *after),
           "parent_before": win(pst, *before), "parent_after": win(pst, *after)}
    report.append(rec)
    print("== patho", json.dumps(rec), flush=True)

agg = {}
for k in ("s2_before", "s2_after", "parent_before", "parent_after"):
    ws = [r[k] for r in report if r[k]]
    if ws:
        agg[k] = {m: round(float(np.mean([w[m] for w in ws])), 3) for m in ws[0]}
summary = {"swap": swap, "n_patho": len(report),
           "pause_after_stop_s_median": float(np.median([r["pause_after_stop_s"] for r in report])) if report else None,
           "share_stop_followed_by_pause>=0.24s": round(float(np.mean([r["pause_after_stop_s"] >= 0.24 for r in report])), 3) if report else None,
           "windows": agg}
print("== SUMMARY", json.dumps(summary), flush=True)
if a.out:
    json.dump({"summary": summary, "clips": report}, open(a.out, "w"), indent=1)
