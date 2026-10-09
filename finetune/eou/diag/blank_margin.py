#!/usr/bin/env python3
"""Blank dominance vs beam search on an EOU/RNNT checkpoint (CPU by default).

Per clip: greedy text, beam-N text, number of <EOU> emitted, and the per-frame
margin max(non-blank) - blank of the joint at the SOS prediction state
(eou/rnnt_diag.py). On the collapsed 2026-10-09 cold checkpoint every frame of
10 clips had a negative margin (max -0.26 .. -0.65) and beam-4 returned the
same sentence for every input.

    $PY blank_margin.py --nemo runs/eou-it-5h-e52/final.nemo \\
        --manifest runs/eou-it-5h-e52/train_5h_eou.json --manifest manifests/eval_fleurs_it.json --n 5
Each --manifest contributes its first --n rows (offset/duration honoured);
--pad-s seconds of zeros are appended like the EOU training padding.
Writes --json if given; prints one ROW line per clip.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", required=True, help="local .nemo (EncDecRNNTBPEModel)")
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--pad-s", type=float, default=3.0)
    ap.add_argument("--beam-size", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    import rnnt_diag as D
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    torch.set_num_threads(a.threads)
    m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location=a.device).eval()
    m.joint._fuse_loss_wer = False
    eou = D.special_id(m.tokenizer, "<EOU>")
    V = m.tokenizer.vocab_size

    def enc(r):
        x, sr = sf.read(r["audio_filepath"], dtype="float32", always_2d=True)
        x = x.mean(1)
        if r.get("offset") is not None:
            s = int(float(r["offset"]) * sr)
            x = x[s:s + int(float(r["duration"]) * sr)]
        x = np.concatenate([x, np.zeros(int(a.pad_s * sr), dtype=np.float32)])
        t = torch.tensor(x)[None].to(a.device)
        with torch.no_grad():
            f, fl = m.preprocessor(input_signal=t, length=torch.tensor([t.shape[1]], device=a.device))
            return m.encoder(audio_signal=f, length=fl)

    def text_of(e, el):
        with torch.no_grad():
            hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el, return_hypotheses=True)
        hy = hy[0] if isinstance(hy, tuple) else hy
        ys = D.hyp_ids(hy[0] if isinstance(hy, list) else hy)
        return m.tokenizer.ids_to_text([i for i in ys if i < V and i != eou]), (ys.count(eou) if eou is not None else 0)

    rows = []
    for mp in a.manifest:
        rows += [(Path(mp).stem, json.loads(line)) for line in open(mp, encoding="utf-8") if line.strip()][: a.n]
    encs = [enc(r) for _, r in rows]
    res = []
    for (src, r), (e, el) in zip(rows, encs):
        g, g_eou = text_of(e, el)
        mg = D.sos_blank_margin(m, e, el)[0]
        res.append({"set": src, "ref": r["text"][:80], "greedy": g, "greedy_eou": g_eou, **mg})
    if a.beam_size > 0:
        prev = D.set_strategy(m, "beam", a.beam_size)
        for rec, (e, el) in zip(res, encs):
            rec["beam"], rec["beam_eou"] = text_of(e, el)
        D.restore_strategy(m, prev)
    for rec in res:
        print("ROW", json.dumps(rec, ensure_ascii=False))
    summ = {"clips": len(res), "all_margins_negative": all(r["margin_max"] < 0 for r in res),
            "greedy_empty_rate": round(sum(not r["greedy"].strip() for r in res) / max(1, len(res)), 3),
            "greedy_audio_dependence": D.audio_dependence([r["greedy"] for r in res]),
            "beam_audio_dependence": D.audio_dependence([r.get("beam", "") for r in res]) if a.beam_size > 0 else None}
    print("SUMMARY", json.dumps(summ))
    if a.json:
        Path(a.json).write_text(json.dumps({"nemo": a.nemo, "summary": summ, "rows": res}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
