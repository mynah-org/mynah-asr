#!/usr/bin/env python3
"""CPU check of the two_stage.py --tokenizer warm-start surgery (no GPU, no training).

Replays the `if a.tokenizer:` block of two_stage.py on a CPU copy of the stock
model (the block is copied verbatim below; keep it in sync) and verifies:
  - which decoder/joint tensors have dim 0 == V + 1 (the remapped ones) and that
    no other tensor carries a vocabulary-sized dim;
  - the prediction-net embedding: padding_idx == blank == V, blank row zero
    before and after; exact-copied pieces are bit-identical to their stock rows;
    <unk>/<EOU>/<EOB>/blank keep ids and rows;
  - what change_vocabulary rebuilt: joint.vocabulary (the Mynah exporter reads it)
    in new-id order, decoder.vocab_size, num_classes_with_blank, decoding blank id,
    cfg.tokenizer / registered tokenizer artefacts, cfg.labels (stale or not);
  - save_to + restore_from round trip keeps the new tokenizer and the rows.
With --vals name=manifest,... --n N it also greedy-decodes the first N clips of
each set with the stock model and with the warm-started one (same encoder,
per-utterance greedy as two_stage.py) on CPU: a preview of the step-0 eval.

    CUDA_VISIBLE_DEVICES= $PY warmstart_check.py --tokenizer /root/ft/analysis/C/tok-it-1024 \
        --vals mls=...,fleurs=... --n 40 --threads 16 --out /root/ft/analysis/C/warmstart.json
"""
import argparse
import json
import os
import re
import sys
import unicodedata

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def norm(t):  # two_stage.py::norm (scoring)
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z' ]", " ", t)
    return " ".join(t.split())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def surgery(m, tokenizer_dir):
    """two_stage.py `if a.tokenizer:` block, verbatim except m.to(dev) and the return value."""
    import torch

    old_sp = m.tokenizer.tokenizer
    old_ids = {old_sp.id_to_piece(i): i for i in range(old_sp.get_piece_size())}
    keep_sd = {k: v.detach().clone() for k, v in m.state_dict().items() if k.startswith(("decoder.", "joint."))}
    m.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type="bpe")
    new_sp = m.tokenizer.tokenizer
    Vn = m.tokenizer.vocab_size
    assert Vn == old_sp.get_piece_size(), (Vn, old_sp.get_piece_size())
    from sentencepiece import sentencepiece_model_pb2 as _pb
    _mp = _pb.ModelProto(); _mp.ParseFromString(old_sp.serialized_model_proto())
    dummy0 = _mp.normalizer_spec.add_dummy_prefix
    src, n_exact = [], 0
    for i in range(Vn):
        pc = new_sp.id_to_piece(i)
        if pc in old_ids:
            src.append([old_ids[pc]]); n_exact += 1; continue
        t = "".join(c for c in unicodedata.normalize("NFKD", pc.replace("▁", " ")) if not unicodedata.combining(c))
        internal = not pc.startswith("▁")
        if internal:
            old_sp.override_normalizer_spec(add_dummy_prefix=False)
        ids = [j for j in old_sp.encode(t.strip()) if j != old_sp.unk_id()]
        if internal:
            old_sp.override_normalizer_spec(add_dummy_prefix=dummy0)
        src.append(ids or [old_sp.unk_id()])
    src.append([Vn])                      # blank
    remapped = []
    for k, v in keep_sd.items():
        if v.shape[0] == Vn + 1:
            keep_sd[k] = torch.stack([v[ids].mean(0) for ids in src])
            remapped.append(k)
    missing, unexpected = m.load_state_dict(keep_sd, strict=False)
    assert not unexpected and not [k for k in missing if k.startswith(("decoder.", "joint."))], (missing, unexpected)
    return old_sp, new_sp, src, n_exact, remapped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", default="/root/ft/models/parakeet_realtime_eou_120m-v1.nemo")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--vals", default="")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--save-check", default="", help="dir for the save_to/restore_from round trip")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import open_dict

    torch.set_num_threads(a.threads)
    assert not torch.cuda.is_available(), "CPU only"
    res = {}
    m = EncDecRNNTBPEModel.restore_from(a.stock, map_location="cpu").eval()
    V = m.tokenizer.vocab_size
    sd0 = {k: v.detach().clone() for k, v in m.state_dict().items()}
    res["vocab_dims"] = {k: tuple(v.shape) for k, v in sd0.items() if any(d in (V, V + 1, V - 1, V + 2) for d in v.shape)}
    emb = m.decoder.prediction["embed"]
    res["embed"] = {"num": emb.num_embeddings, "padding_idx": emb.padding_idx, "blank_idx": m.decoder.blank_idx,
                    "blank_as_pad": m.decoder.blank_as_pad,
                    "stock_blank_row_absmax": float(emb.weight[emb.padding_idx].abs().max())}
    res["stock_labels_len"] = len(m.cfg.labels) if "labels" in m.cfg else None

    def transcribe(rows):
        hyps, nb, eou = [], [], 0
        with torch.no_grad():
            for r in rows:
                x, sr = sf.read(r["audio_filepath"], dtype="float32")
                assert sr == 16000, sr
                if x.ndim > 1:
                    x = x.mean(1)
                X = torch.from_numpy(np.ascontiguousarray(x))[None]
                L = torch.tensor([X.shape[1]])
                f, fl = m.preprocessor(input_signal=X, length=L)
                e, el = m.encoder(audio_signal=f, length=fl)
                hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el,
                                                                return_hypotheses=True)
                hy = hy[0] if isinstance(hy, tuple) else hy
                h = hy[0][0] if isinstance(hy[0], list) else hy[0]
                ys = h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist") else list(h.y_sequence)
                eou += int(1024 in ys)
                hyps.append((ys, norm(m.tokenizer.ids_to_text([t for t in ys if t < 1024]))))
                d, _ = m.decoder.predict(None, state=None, add_sos=False, batch_size=1)[:2]
                j = m.joint.joint(e.transpose(1, 2), d)[:, :, 0, :]
                nb.append(float(((j[..., :V].max(-1).values - j[..., V])[0, : int(el[0])] > 0).float().mean()))
        return hyps, nb, eou

    def greedy_cfg():
        dc = m.cfg.decoding
        with open_dict(dc):
            dc.strategy = "greedy"
            if "greedy" in dc:
                dc.greedy.use_cuda_graph_decoder = False
        m.change_decoding_strategy(dc, verbose=False)

    sets = {}
    if a.vals:
        for kv in a.vals.split(","):
            n, p = kv.split("=", 1)
            sets[n] = [json.loads(l) for l in open(p)][: a.n]
    dec = {}
    greedy_cfg()
    for n, rows in sets.items():
        dec[("stock", n)] = transcribe(rows)

    old_sp, new_sp, src, n_exact, remapped = surgery(m, a.tokenizer)
    m.eval()
    Vn = m.tokenizer.vocab_size
    sd1 = m.state_dict()
    res["remapped"] = {k: tuple(sd1[k].shape) for k in remapped}
    res["dims_after"] = {k: tuple(v.shape) for k, v in sd1.items() if any(d in (Vn, Vn + 1) for d in v.shape)}
    res["not_remapped_but_vocab_sized"] = sorted(set(res["dims_after"]) - set(remapped))
    emb = m.decoder.prediction["embed"]
    res["embed_after"] = {"num": emb.num_embeddings, "padding_idx": emb.padding_idx,
                          "blank_row_absmax": float(emb.weight[emb.padding_idx].abs().max())}
    exact = [i for i, s in enumerate(src[:Vn]) if new_sp.id_to_piece(i) == old_sp.id_to_piece(s[0]) and len(s) == 1]
    bad = []
    for k in remapped:
        for i in exact + [Vn]:
            j = src[i][0]
            if not torch.equal(sd1[k][i].cpu(), sd0[k][j]):
                bad.append((k, i))
    res["exact_rows_bit_identical"] = not bad
    res["exact_rows_mismatch_examples"] = bad[:10]
    res["special_ids"] = {p: (old_sp.piece_to_id(p), new_sp.piece_to_id(p)) for p in ("<unk>", "<EOU>", "<EOB>")}
    jv = list(m.joint.vocabulary)
    res["joint_vocabulary_matches_new_spe"] = jv == [new_sp.id_to_piece(i) for i in range(Vn)]
    res["joint_vocab_len"] = len(jv)
    res["cfg_joint_vocab_matches"] = list(m.cfg.joint.vocabulary) == jv
    res["cfg_decoder_vocab_size"] = int(m.cfg.decoder.vocab_size)
    res["joint_num_classes_with_blank"] = int(m.joint.num_classes_with_blank)
    res["decoding_blank_id"] = int(getattr(m.decoding, "blank_id", -1))
    res["cfg_labels_stale"] = ("labels" in m.cfg) and list(m.cfg.labels)[:5] != jv[:5]
    res["cfg_tokenizer"] = {k: str(v) for k, v in m.cfg.tokenizer.items()}
    res["artifacts"] = {k: str(v.path) for k, v in getattr(m, "artifacts", {}).items()}
    res["loss_num_classes"] = getattr(m.loss, "_blank", None)
    res["decoding_strategy_after_change_vocab"] = str(m.cfg.decoding.strategy)
    greedy_cfg()
    for n, rows in sets.items():
        dec[("warm", n)] = transcribe(rows)

    if a.save_check:
        os.makedirs(a.save_check, exist_ok=True)
        p = os.path.join(a.save_check, "warm.nemo")
        m.save_to(p)
        m2 = EncDecRNNTBPEModel.restore_from(p, map_location="cpu")
        sp2 = m2.tokenizer.tokenizer
        res["roundtrip"] = {
            "tokenizer_same": all(sp2.id_to_piece(i) == new_sp.id_to_piece(i) for i in range(Vn)),
            "rows_same": all(torch.equal(m2.state_dict()[k], sd1[k]) for k in remapped),
            "probe": m2.tokenizer.text_to_ids("perché la città"),
        }
        os.remove(p)

    new_mean = {i for i, s in enumerate(src[:Vn]) if not (len(s) == 1 and new_sp.id_to_piece(i) == old_sp.id_to_piece(s[0]))}
    ev = {}
    for n, rows in sets.items():
        refs = [norm(r["text"]) for r in rows]
        for arm in ("stock", "warm"):
            hy, nb, eou = dec[(arm, n)]
            we = sum(lev(r.split(), h.split()) for r, (_, h) in zip(refs, hy))
            ce = sum(lev(r, h) for r, (_, h) in zip(refs, hy))
            toks = [t for ys, _ in hy for t in ys if t < 1024]
            ev[f"{n}/{arm}"] = {
                "wer": round(100 * we / max(1, sum(len(r.split()) for r in refs)), 2),
                "cer": round(100 * ce / max(1, sum(len(r) for r in refs)), 2),
                "emitted_tokens": len(toks), "ref_words": sum(len(r.split()) for r in refs),
                "hyp_words": sum(len(h.split()) for _, h in hy),
                "nonblank_frame_pct": round(100 * sum(nb) / max(1, len(nb)), 2),
                "eou_utts": eou, "unk_emitted": sum(t == 0 for t in toks),
                "mean_init_piece_share_pct": round(100 * sum(t in new_mean for t in toks) / max(1, len(toks)), 2),
                "examples": [(r[:90], h[:90]) for r, (_, h) in list(zip(refs, hy))[:3]],
            }
            print("EVAL", n, arm, {k: v for k, v in ev[f"{n}/{arm}"].items() if k != "examples"}, flush=True)
    res["preview_eval"] = ev
    print(json.dumps({k: v for k, v in res.items() if k != "preview_eval"}, ensure_ascii=False, indent=1, default=str))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
