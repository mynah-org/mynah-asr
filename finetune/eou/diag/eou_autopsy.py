#!/usr/bin/env python3
"""Autopsy of every model <EOU> of a stage-2 checkpoint under the frozen s2_reset protocol (CPU, diagnostic only).

    eou_autopsy.py --s2 s2.nemo --vals "mls=a.json,..." [--per-set 200] --tag dev --out events.jsonl

Re-runs the s2_reset.py s2_reset path EXACTLY (same encode, greedy, max 10 symbols, decoder reset after <EOU>, 2 s
of -75 dB quiet, energy endpoint of the ORIGINAL clip) and logs, without changing any decision:
  - per <EOU> event: class (premature = >= 0.5 s of speech left, the frozen FE rule; else true), time to the speech
    end, speech after it, the silence run before it (pause_before) and the time until speech resumes (resume_after;
    None for a true endpoint), clip duration and relative position, hyp words before / after, frame phase (f % 2),
    the frame dB trajectory around it (-0.96 .. +0.96 s, relative to the clip peak), the EOU margin
    (z_EOU - max other output, max over the frame's symbol steps) at the event and its maximum over the preceding
    silence run;
  - per INTERNAL pause (silence run >= 0.24 s with speech after it) that did NOT end in an <EOU>: duration, its max
    EOU margin, position: the negatives the model got right.
`--check-dump D` asserts that the re-decoded ids equal an existing s2_reset.py --dump (same protocol, same clips).
Nothing here selects or deploys a threshold.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
import textnorm  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--s2", required=True)
ap.add_argument("--vals", required=True)
ap.add_argument("--per-set", type=int, default=200)
ap.add_argument("--tag", required=True)
ap.add_argument("--threads", type=int, default=20)
ap.add_argument("--check-dump", default="")
ap.add_argument("--lang", default="fr")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
from nemo.collections.asr.models import ASRModel  # noqa: E402

S = ASRModel.restore_from(a.s2, map_location="cpu").eval()
tk = S.tokenizer
EOU, BLANK, FR = 1024, S.joint.num_classes_with_blank - 1, 0.08


@torch.no_grad()
def encode(x):
    f, fl = S.preprocessor(input_signal=torch.from_numpy(x)[None], length=torch.tensor([len(x)]))
    e, el = S.encoder(audio_signal=f, length=fl)
    return e[0, :, : int(el[0])].T


@torch.no_grad()
def greedy(enc):
    """s2_reset.py greedy(reset=True) + per-frame EOU margin (max over the frame's symbol steps)."""
    ys, frames, marg = [], [], np.full(enc.shape[0], -1e9, dtype=np.float32)
    g, st = S.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
    for t in range(enc.shape[0]):
        f = enc[t][None, None, :]
        for _ in range(10):
            z = S.joint.joint(f, g)[0, 0, 0]
            oth = torch.cat([z[:EOU], z[EOU + 1:]]).max()
            marg[t] = max(marg[t], float(z[EOU] - oth))
            k = int(z.argmax())
            if k == BLANK:
                break
            ys.append(k); frames.append(t)
            if k == EOU:
                g, st = S.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
                break
            g, st = S.decoder.predict(torch.tensor([[k]]), state=st, add_sos=False, batch_size=1)[:2]
    return ys, frames, marg


def frames_db(x):
    """frame dB relative to the clip peak, and the peak (absolute dB)"""
    n = int(FR * 16000)
    f = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    return db - db.max(), float(db.max())


chk = {}
if a.check_dump:
    for d in map(json.loads, open(a.check_dump)):
        if d["protocol"] == "s2_reset":
            chk[d["audio"]] = d["ids"]
out = open(a.out, "w")
n_ok = n_chk = 0
for spec in a.vals.split(","):
    sname, p = spec.split("=", 1)
    rows = [json.loads(l) for l in open(p)]
    rows = rows[:: max(1, len(rows) // a.per_set)][: a.per_set]
    for r in rows:
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        rdb, peak = frames_db(x)           # ORIGINAL clip, as s2_reset.speech_mask
        sm = rdb > -35
        on = np.nonzero(sm)[0]
        end = (on[-1] + 1) * FR if len(on) else len(x) / 16000
        dur = len(x) / 16000
        xp = np.concatenate([x, (np.random.RandomState(len(x)).randn(32000) * 10 ** (-75 / 20)).astype(np.float32)])
        ys, fr, marg = greedy(encode(xp))
        if r["audio_filepath"] in chk:
            n_chk += 1; n_ok += chk[r["audio_filepath"]] == ys
        nf = len(sm)
        ef = [f for y, f in zip(ys, fr) if y == EOU]
        # hyp words per segment
        segs, cur = [], []
        for y in ys:
            if y == EOU:
                segs.append(cur); cur = []
            elif y < 1024:
                cur.append(y)
        segs.append(cur)
        seg_words = [len(textnorm.norm(tk.ids_to_text(s), a.lang).split()) if s else 0 for s in segs]
        ref_words = len(textnorm.norm(r["text"], a.lang).split())
        smp = np.concatenate([sm, np.zeros(26, dtype=bool)])   # the 2 s pad = non-speech frames
        dbp = np.concatenate([rdb, np.full(26, -75.0 - peak)])   # the pad: -75 dB absolute
        for i, f in enumerate(ef):
            t = f * FR
            pb = 0
            while f - pb >= 0 and not smp[min(f - pb, len(smp) - 1)]:
                pb += 1
            nxt = np.nonzero(sm[f + 1:])[0] if f + 1 < nf else []
            resume = (nxt[0] + 1) * FR if len(nxt) else None
            lo = max(0, f - pb)
            traj = [float(dbp[j]) if 0 <= j < len(dbp) else None for j in range(f - 12, f + 13)]
            ev = {"tag": a.tag, "set": sname, "audio": r["audio_filepath"], "kind": "eou",
                  "premature": bool(t < end - 0.5), "t": round(t, 2), "end": round(float(end), 2), "dur": round(dur, 2),
                  "to_end_s": round(float(end - t), 2), "speech_after_s": round(float(sm[f + 1:].sum() * FR), 2),
                  "pause_before_s": round(pb * FR, 2), "resume_after_s": round(float(resume), 2) if resume is not None else None,
                  "rel_pos": round(t / dur, 3), "words_before": int(sum(seg_words[: i + 1])),
                  "words_after_hyp": int(sum(seg_words[i + 1:])), "ref_words": ref_words, "phase": int(f % 2),
                  "n_eou_clip": len(ef), "margin_at": round(float(marg[f]), 2),
                  "margin_max_in_pause": round(float(marg[lo: f + 1].max()), 2),
                  "pause_db_mean": round(float(np.mean(dbp[lo: f + 1])), 1) if pb else None, "traj_db": traj}
            out.write(json.dumps(ev) + "\n")
        # internal pauses with speech after that did NOT fire an <EOU>: the correctly rejected negatives
        fired = set(ef)
        j = 0
        while j < nf:
            if not sm[j]:
                k = j
                while k < nf and not sm[k]:
                    k += 1
                if j > 0 and k < nf and (k - j) * FR >= 0.24 and not any(j <= f <= k for f in fired):
                    out.write(json.dumps({"tag": a.tag, "set": sname, "audio": r["audio_filepath"], "kind": "pause_no_eou",
                                          "pause_s": round((k - j) * FR, 2), "t": round(j * FR, 2), "rel_pos": round(j * FR / dur, 3),
                                          "margin_max": round(float(marg[j:k].max()), 2),
                                          "pause_db_mean": round(float(rdb[j:k].mean()), 1)}) + "\n")
                j = k
            else:
                j += 1
    print(f"== {a.tag} {sname}: {len(rows)} clips", flush=True)
if chk:
    print(f"== check vs dump: {n_ok}/{n_chk} clips with identical ids", flush=True)
