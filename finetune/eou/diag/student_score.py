#!/usr/bin/env python3
"""Per-clip greedy score of a checkpoint on TRAINING manifests (curriculum / data selection input).

    python student_score.py --nemo runs/plain-it-w4-c250-16k/final.nemo \
        --manifests "mls=train_40h.json,cv=train_cv.json,..." --out student.jsonl [--bs 32]

Every clip of every manifest (no subsampling), decoded with the serving greedy decoder (NeMo greedy_batch, as
lm_fusion_probe.py's greedy), batched by duration. One JSON line per clip: source, audio_filepath, duration,
ref (a-z normalised label), student (normalised hypothesis), student_wer (%, vs that label), ref_words.
Streaming-safe: encoder outputs are decoded per batch and never accumulated; resumable (clips already in --out
are skipped).
"""
import argparse, json, os, re, time, unicodedata
import numpy as np, soundfile as sf, torch

ap = argparse.ArgumentParser()
ap.add_argument("--nemo", required=True)
ap.add_argument("--manifests", required=True, help="name=manifest,...")
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max-s", type=float, default=30.0, help="= two_stage.py --max-dur of the C250 runs (the sampled universe)")
a = ap.parse_args()


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def lev(x, y):
    prev = list(range(len(y) + 1))
    for i, p in enumerate(x, 1):
        cur = [i]
        for j, q in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != q)))
        prev = cur
    return prev[-1]


done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try:
            done.add(json.loads(l)["audio_filepath"])
        except Exception:
            pass
todo = []
for x in a.manifests.split(","):
    n, p = x.split("=", 1)
    for l in open(p):
        r = json.loads(l)
        if r["audio_filepath"] not in done and r.get("duration", 0) <= a.max_s:
            todo.append((n, r))
todo.sort(key=lambda t: t[1].get("duration", 0))
print(f"== student_score: {len(todo)} clips to score ({len(done)} already in {a.out})", flush=True)

from nemo.collections.asr.models import EncDecRNNTBPEModel
from omegaconf import open_dict
m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location="cuda").eval()
tk = m.tokenizer
dc = m.cfg.decoding
with open_dict(dc):   # = lm_fusion_probe.py's greedy (greedy_batch label-looping, max 10 symbols per frame)
    dc.strategy = "greedy_batch"; dc.preserve_alignments = False; dc.compute_timestamps = False
    dc.greedy.max_symbols = 10; dc.greedy.loop_labels = True; dc.greedy.use_cuda_graph_decoder = False
    dc.greedy.ngram_lm_model = None; dc.greedy.ngram_lm_alpha = 0.0
m.change_decoding_strategy(dc, verbose=False)
dec = m.decoding


def text_of(h):   # = lm_fusion_probe.py: drop ids >= 1024 (EOU / special)
    ys = h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)
    return norm(tk.ids_to_text([t for t in ys if t < 1024]))
t0 = time.time()
from concurrent.futures import ThreadPoolExecutor


def load(ch):
    xs = []
    for _, r in ch:
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        xs.append(x.mean(1) if x.ndim > 1 else x)
    return xs


chunks = [todo[i:i + a.bs] for i in range(0, len(todo), a.bs)]
pool = ThreadPoolExecutor(4)
futs = [pool.submit(load, c) for c in chunks[:4]]   # audio of the next batches is read while the GPU decodes
with open(a.out, "a") as fo, torch.inference_mode():
    for k, ch in enumerate(chunks):
        i = k * a.bs
        xs = futs[k].result(); futs[k] = None
        if k + 4 < len(chunks):
            futs.append(pool.submit(load, chunks[k + 4]))
        X = torch.zeros(len(xs), max(len(x) for x in xs))
        for b, x in enumerate(xs):
            X[b, :len(x)] = torch.from_numpy(x)
        lens = torch.tensor([len(x) for x in xs])
        f, fl = m.preprocessor(input_signal=X.cuda(), length=lens.cuda())
        e, el = m.encoder(audio_signal=f, length=fl)
        hyps = dec.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el, return_hypotheses=True)
        hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        for (n, r), h in zip(ch, hyps):
            hyp = text_of(h)
            ref = norm(r["text"]); rw = ref.split()
            w = 100.0 * lev(rw, hyp.split()) / max(1, len(rw))
            fo.write(json.dumps({"source": n, "audio_filepath": r["audio_filepath"], "duration": r.get("duration"),
                                 "ref": ref, "student": hyp, "student_wer": round(w, 2), "ref_words": len(rw)},
                                ensure_ascii=False) + "\n")
        if (i // a.bs) % 200 == 0:
            el_ = time.time() - t0
            print(f"  {i + len(ch)}/{len(todo)} clips {el_:.0f}s ({(i + len(ch)) / max(1, el_):.0f} clips/s)", flush=True)
print(f"== STUDENT-SCORE-DONE {len(todo)} clips {time.time() - t0:.0f}s", flush=True)
