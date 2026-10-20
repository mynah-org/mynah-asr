"""Output-side bottleneck probe: n-gram shallow fusion and N-best oracle on a fixed checkpoint.

The encoder output is computed once per clip (same streaming encoder config as the
training-time eval); then the SAME acoustic evidence is decoded several ways with NeMo 3.0's
own decoders (rnnt_decoding.py: greedy_batch label-looping and malsd_batch beam, both with
NGPU-LM fusion via decoding.{greedy,beam}.ngram_lm_model / ngram_lm_alpha):
  greedy a=0 | greedy + LM a | beam K a=0 | beam K + LM a  (+ N-best oracle WER for beams)
Large WER gains from fusion / a low oracle => the acoustic evidence is in the encoder output
and the prediction network + joint (internal LM / output side) is what limits; no gain and a
high oracle => the encoder representation itself is the bottleneck.

Runs on CPU or GPU (device follows CUDA_VISIBLE_DEVICES). WER/CER are scored like
two_stage.py (lower case, de-accented: textnorm.norm(t, --lang), default it). Clip choice = two_stage.py subsample() with --per-set.

  CUDA_VISIBLE_DEVICES= python lm_fusion_probe.py --nemo /root/ft/runs/plain-it-b2-gradual/final.nemo \
      --lm lm4_trainmix.arpa --per-set 60 --alphas 0.3,0.6 --beam 4 --out fusion_b2.json
"""
import argparse, json, re, sys, time, unicodedata
from pathlib import Path
import numpy as np, soundfile as sf, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import textnorm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nemo", required=True)
ap.add_argument("--lm", default="", help="token ARPA (build_token_arpa.py) or NGPU-LM .nemo; not needed when every alpha is 0")
ap.add_argument("--vals", default="mls=/root/ft/manifests/eval_mls_it.json,fleurs=/root/ft/manifests/eval_fleurs_it.json,"
                "cv=/root/ft/mix/it/eval_cv.json,vp=/root/ft/mix/it/eval_vp.json")
ap.add_argument("--per-set", type=int, default=60)
ap.add_argument("--alphas", default="0.3,0.6")
ap.add_argument("--beam", type=int, default=4)
ap.add_argument("--beams", default="", help="several beam sizes, e.g. 4,8,16 (overrides --beam): oracle@K vs K")
ap.add_argument("--dump-nbest", default="", help="DIR: per beam size, nbest_b<K>.jsonl with every hypothesis "
                "(text, score, tokens) per clip, for diag/nbest_audit.py (offline rescoring / ranking audit)")
ap.add_argument("--tune-extra", type=int, default=0, help="1: also decode the MLS/FLEURS val clips OUTSIDE the "
                "harness subsample (sets mls_tune / fleurs_tune; never in the macro): a disjoint split to tune "
                "rescoring on, so the harness clips stay report-only")
ap.add_argument("--beam-alphas", default="", help="LM weights for beam (default: --alphas)")
ap.add_argument("--blank-mode", default="lm_weighted_full", help="beam blank_lm_score_mode: lm_weighted_full | no_score")
ap.add_argument("--skip-greedy", type=int, default=0)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--post-pad-s", type=float, default=0.0, help="digital silence appended to every clip: lets the streaming "
                "encoder emit the last tokens (0 = raw end of file, the historical protocol)")
ap.add_argument("--lang", default="it", choices=sorted(textnorm.PRE), help="ref/hyp normaliser (common/textnorm.py)")
ap.add_argument("--out", default="fusion.json")
a = ap.parse_args()
torch.set_num_threads(a.threads)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def norm(t):
    return textnorm.norm(t, a.lang)   # lang="it" == the historical rule of this file


def lev(x, y):
    prev = list(range(len(y) + 1))
    for i, p in enumerate(x, 1):
        cur = [i]
        for j, q in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != q)))
        prev = cur
    return prev[-1]


from omegaconf import open_dict
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel

m = EncDecRNNTBPEModel.restore_from(a.nemo, map_location=dev).eval()
tk = m.tokenizer
blank = m.decoding.blank_id if hasattr(m.decoding, "blank_id") else tk.vocab_size
lm_path = a.lm
assert lm_path or not any(float(x) for x in (a.alphas + "," + a.beam_alphas).split(",") if x), "--lm needed for alpha > 0"
if lm_path.endswith(".arpa"):   # parse the ARPA once, reuse as .nemo for every decoding config
    t0 = time.time()
    lm = NGramGPULanguageModel.from_file(lm_path, vocab_size=tk.vocab_size)
    lm_path = lm_path[:-5] + ".ngpulm.nemo"
    lm.save_to(lm_path)
    print(f"LM parsed {time.time() - t0:.0f}s -> {lm_path}", flush=True)

sets, tune_sets = [], []
for spec in a.vals.split(","):
    n, p = spec.split("=", 1)
    rs = [json.loads(l) for l in open(p)]
    sub = rs[:: max(1, len(rs) // a.per_set)][: a.per_set]
    sets.append((n, sub))
    if a.tune_extra and n in ("mls", "fleurs"):
        used = {r["audio_filepath"] for r in sub}
        tune_sets.append((f"{n}_tune", [r for r in rs if r["audio_filepath"] not in used]))

enc = []    # (set, ref, enc[T, D], len)
t0 = time.time()
with torch.no_grad():
    for n, rs in sets + tune_sets:
        for i in range(0, len(rs), 8):
            ch = rs[i:i + 8]
            xs = []
            for r in ch:
                x, sr = sf.read(r["audio_filepath"], dtype="float32")
                x = x.mean(1) if x.ndim > 1 else x
                xs.append(np.concatenate([x, np.zeros(int(a.post_pad_s * 16000), np.float32)]) if a.post_pad_s else x)
            L = max(len(x) for x in xs)
            X = torch.zeros(len(xs), L)
            for b, x in enumerate(xs):
                X[b, :len(x)] = torch.from_numpy(x)
            lens = torch.tensor([len(x) for x in xs])
            f, fl = m.preprocessor(input_signal=X.to(dev), length=lens.to(dev))
            e, el = m.encoder(audio_signal=f, length=fl)
            for b, r in enumerate(ch):
                enc.append((n, norm(r["text"]), e[b, :, : int(el[b])].cpu(), int(el[b]), r["audio_filepath"]))
print(f"encoded {len(enc)} clips in {time.time() - t0:.0f}s on {dev}", flush=True)


def set_decoding(strategy, alpha, beam=1):
    dc = m.cfg.decoding
    with open_dict(dc):
        dc.strategy = strategy
        dc.preserve_alignments = False      # malsd_batch: "Preserve alignments is not supported"
        dc.compute_timestamps = False
        dc.greedy.max_symbols = 10
        dc.greedy.loop_labels = True
        dc.greedy.use_cuda_graph_decoder = False
        dc.greedy.ngram_lm_model = lm_path if alpha > 0 else None
        dc.greedy.ngram_lm_alpha = alpha
        dc.beam.beam_size = beam
        dc.beam.return_best_hypothesis = False
        dc.beam.allow_cuda_graphs = False
        dc.beam.pruning_mode = "late"                  # NeMo 3.0 writes the enum NAME back into cfg ("LATE"),
        dc.beam.blank_lm_score_mode = a.blank_mode  # which the next change_decoding_strategy rejects
        dc.beam.ngram_lm_model = lm_path if alpha > 0 else None
        dc.beam.ngram_lm_alpha = alpha
    m.change_decoding_strategy(dc, verbose=False)


def text_of(h):
    ys = h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)
    return norm(tk.ids_to_text([t for t in ys if t < 1024]))


def run(tag, dump=None):
    t0 = time.time()
    per = {}
    fd = open(dump, "w") if dump else None
    with torch.no_grad():
        for i in range(0, len(enc), 16):
            ch = enc[i:i + 16]
            T = max(x[3] for x in ch)
            E = torch.zeros(len(ch), ch[0][2].shape[0], T)
            for b, x in enumerate(ch):
                E[b, :, : x[3]] = x[2]
            el = torch.tensor([x[3] for x in ch])
            hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=E.to(dev), encoded_lengths=el.to(dev),
                                                            return_hypotheses=True)
            hy = hy[0] if isinstance(hy, tuple) else hy
            for (n, ref, _, _, path), h in zip(ch, hy):
                # beam with return_best_hypothesis=False: list[Hypothesis] per clip, best first
                nb = list(h) if isinstance(h, list) else list(h.n_best_hypotheses) if hasattr(h, "n_best_hypotheses") else [h]
                texts = [text_of(x) for x in nb]
                rw = ref.split()
                errs = [lev(rw, t.split()) for t in texts]
                s = per.setdefault(n, {"we": 0, "wn": 0, "ce": 0, "cn": 0, "oracle_we": 0, "hw": 0, "empty": 0, "n": 0})
                s["hw"] += len(texts[0].split()); s["empty"] += not texts[0].strip(); s["n"] += 1
                s["we"] += errs[0]; s["wn"] += len(rw); s["ce"] += lev(ref, texts[0]); s["cn"] += len(ref)
                s["oracle_we"] += min(errs)
                if fd:
                    fd.write(json.dumps({"set": n, "path": path, "ref": ref, "hyps": [
                        {"text": t, "score": float(x.score), "ntok": len([y for y in (x.y_sequence.tolist() if hasattr(x.y_sequence, "tolist") else x.y_sequence) if y < 1024])}
                        for t, x in zip(texts, nb)]}, ensure_ascii=False) + "\n")
    rec = {"tag": tag, "sec": round(time.time() - t0, 1)}
    for n, s in per.items():
        rec[f"{n}_wer"] = round(100 * s["we"] / s["wn"], 2); rec[f"{n}_cer"] = round(100 * s["ce"] / s["cn"], 2)
        rec[f"{n}_oracle_wer"] = round(100 * s["oracle_we"] / s["wn"], 2)
        rec[f"{n}_len_ratio"] = round(s["hw"] / s["wn"], 3)   # hypothesis words / reference words
        rec[f"{n}_empty"] = s["empty"]; rec[f"{n}_n"] = s["n"]   # empty 1-best hypotheses / clips (2026-10-20)
    if fd:
        fd.close()
    rec["macro_wer"] = round(float(np.mean([rec[f"{n}_wer"] for n, _ in sets])), 2)
    rec["macro_cer"] = round(float(np.mean([rec[f"{n}_cer"] for n, _ in sets])), 2)
    rec["macro_oracle_wer"] = round(float(np.mean([rec[f"{n}_oracle_wer"] for n, _ in sets])), 2)
    print(json.dumps(rec), flush=True)
    return rec


out = []
alphas = [float(x) for x in a.alphas.split(",") if x]
balphas = [float(x) for x in (a.beam_alphas or a.alphas).split(",") if x]
for al in ([] if a.skip_greedy else [0.0] + [x for x in alphas if x]):
    set_decoding("greedy_batch", al); out.append(run(f"greedy a={al}"))
    json.dump(out, open(a.out, "w"), indent=1)
if a.dump_nbest:
    import os
    os.makedirs(a.dump_nbest, exist_ok=True)
for K in ([int(x) for x in a.beams.split(",")] if a.beams else [a.beam] if a.beam > 1 else []):
    for al in [0.0] + [x for x in balphas if x]:
        set_decoding("malsd_batch", al, K)
        out.append(run(f"malsd beam={K} a={al}", f"{a.dump_nbest}/nbest_b{K}.jsonl" if a.dump_nbest and al == 0 else None))
        json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
