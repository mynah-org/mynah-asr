"""NeMo reference transcripts of the EOU 120M on the q1 clips.

Usage: python nemo_ref.py <q1 eou-en.json> <bank root> <out dir>

Runs the checkpoint through NeMo with the special tokens KEPT (so <EOU>/<EOB>
are visible), on: every clip Mynah returned empty for, plus the same number of
clips Mynah transcribed. Writes nemo_ref.json (file, mynah text, nemo text,
nemo text with specials) and prints a summary. Offline (full-utterance) pass
with the model's own streaming attention context, i.e. what NeMo's
`transcribe()` does for this checkpoint; the cache-aware chunked pass is a
second arm only if the two disagree with each other.
"""
import json
import sys

import torch
import nemo.collections.asr as nemo_asr

q1, root, out = sys.argv[1:4]
rows = json.load(open(q1))["langs"]["en"]["utterances"]
empty = [r for r in rows if not r["text"].strip()]
full = [r for r in rows if r["text"].strip()][: len(empty)]
pick = empty + full
m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
    "models/parakeet-realtime-eou-120m/parakeet_realtime_eou_120m-v1.nemo", map_location="cpu")
m.eval()
# CPU on purpose: the GPU belongs to the CUDA gates, and the pip torch wheel
# targets a newer driver than this box has.
print("att_context_size", m.encoder.att_context_size, "preproc", m.cfg.preprocessor.get("normalize"),
      "features", m.cfg.preprocessor.get("features"))
paths = [f"{root}/{r['file']}" for r in pick]
hyps = m.transcribe(paths, batch_size=16, return_hypotheses=True)
if isinstance(hyps, tuple):
    hyps = hyps[0]
tok = m.tokenizer
res = []
for r, h in zip(pick, hyps):
    ids = [int(i) for i in (h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else h.y_sequence)]
    pieces = [tok.ids_to_tokens([i])[0] if i < tok.vocab_size else f"<{i}>" for i in ids]
    res.append({"file": r["file"], "mynah": r["text"], "nemo": h.text, "ids": ids,
                "pieces": "".join(pieces).replace("▁", " "), "mynah_empty": not r["text"].strip()})
json.dump(res, open(f"{out}/nemo_ref.json", "w"), indent=1)
ne = [x for x in res if x["mynah_empty"]]
print(f"mynah-empty clips: {len(ne)}; NeMo empty on them: {sum(1 for x in ne if not x['nemo'].strip())}")
print(f"clips with an <EOU> id (1024) in NeMo's sequence: {sum(1 for x in res if 1024 in x['ids'])}/{len(res)}")
for x in res[:6] + ne[:6]:
    print(f"{x['file'][:40]:40s} | mynah: {x['mynah'][:50]!r} | nemo: {x['pieces'][:70]!r}")
