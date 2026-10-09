#!/usr/bin/env python3
"""Add an Italian text sub-tokenizer to Canary 180M without losing anything.

Steps (all on the box, GPU used only for the two short EN decodes):
  1. Inspect the existing sub-tokenizers (order, sizes, offsets) and the `es`
     SentencePiece trainer/normaliser spec.
  2. Train an `it` SentencePiece model (default 1024 pieces) on the TRAINING
     POOL transcripts only ($FT/text/it_pool.txt), with the same trainer
     settings as the `es` sub-tokenizer (model type, coverage, special ids,
     normalisation rule, digit/whitespace splitting), via NeMo's own
     create_spt_model so the directory layout is NeMo's.
  3. Decode the EN sanity clips with the untouched model (reference).
  4. Build the aggregate tokenizer config = existing langs IN THEIR ORDER, then
     `it` appended LAST, so every existing id keeps its value (spl_tokens and
     en/de/es/fr = ids 0..5247; it = 5248..). Call
     `change_vocabulary(new_cfg, "agg")`: NeMo keeps the 4 decoder layers but
     re-initialises the token embedding (weight-tied output head) and the head
     bias (aed_multitask_models.py, v3.0.0).
  5. COPY BACK the old embedding rows and head-bias rows 0..V_old-1, and
     initialise the new `it` rows from the `es` rows of each piece's
     decomposition (exact piece match first, else the mean of the es pieces of
     the same surface string; bias the same way).
  6. save_to(canary-180m-flash-it.nemo), restore it FRESH (so the prompt
     formatter picks up the new tokenizer -- change_vocabulary does not rebuild
     model.prompt), and verify:
       - first V_old rows of embedding and bias bit-identical to the original;
       - EN transcripts of the sanity clips IDENTICAL to step 3;
       - `it` round trip: text -> ids (all >= V_old) -> text.
     Any failure exits non-zero and no `ok` marker is written.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftlib  # noqa: E402

FT = ftlib.FT


def sp_spec(model_file):
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(Path(model_file).read_bytes())
    ts, ns = m.trainer_spec, m.normalizer_spec
    return {
        "model_type": pb.TrainerSpec.ModelType.Name(ts.model_type).lower(),
        "vocab_size": ts.vocab_size,
        "character_coverage": ts.character_coverage,
        "byte_fallback": ts.byte_fallback,
        "split_digits": ts.split_digits,
        "split_by_whitespace": ts.split_by_whitespace,
        "split_by_unicode_script": ts.split_by_unicode_script,
        "max_sentencepiece_length": ts.max_sentencepiece_length,
        "unk_id": ts.unk_id, "bos_id": ts.bos_id, "eos_id": ts.eos_id, "pad_id": ts.pad_id,
        "user_defined_symbols": list(ts.user_defined_symbols),
        "control_symbols": list(ts.control_symbols),
        "normalization_rule_name": ns.name,
        "add_dummy_prefix": ns.add_dummy_prefix,
        "remove_extra_whitespaces": ns.remove_extra_whitespaces,
        "pieces": len(m.pieces),
    }


def train_it_spm(text, out_dir, ref, vocab):
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model

    out_dir = Path(out_dir)
    if (out_dir / "tokenizer.model").exists():
        print(f"  it tokenizer exists: {out_dir}")
        return
    create_spt_model(
        str(text),
        vocab_size=vocab,
        sample_size=-1,
        do_lower_case=ref["normalization_rule_name"] == "nmt_nfkc_cf",
        tokenizer_type=ref["model_type"],
        output_dir=str(out_dir),
        character_coverage=ref["character_coverage"],
        max_sentencepiece_length=ref["max_sentencepiece_length"],
        bos=ref["bos_id"] >= 0,
        eos=ref["eos_id"] >= 0,
        pad=ref["pad_id"] >= 0,
        control_symbols=ref["control_symbols"] or None,
        user_defined_symbols=ref["user_defined_symbols"] or None,
        byte_fallback=ref["byte_fallback"],
        split_digits=ref["split_digits"],
        split_by_whitespace=ref["split_by_whitespace"],
        split_by_unicode_script=ref["split_by_unicode_script"],
        remove_extra_whitespaces=ref["remove_extra_whitespaces"],
    )


def emb_and_bias(model):
    emb = model.transf_decoder.embedding.token_embedding.weight
    head = model.log_softmax.mlp.layer0
    assert head.weight.data_ptr() == emb.data_ptr(), "head is not weight-tied to the embedding"
    return emb, head.bias


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(FT / "models" / "canary-180m-flash.nemo"))
    ap.add_argument("--out", default=str(FT / "models" / "canary-180m-flash-it.nemo"))
    ap.add_argument("--text", default=str(FT / "text" / "it_pool.txt"))
    ap.add_argument("--vocab", type=int, default=int(os.environ.get("IT_VOCAB", "1024")))
    ap.add_argument("--init-from", default="es", help="sub-tokenizer whose rows seed the new it rows")
    ap.add_argument("--en", default=str(FT / "manifests" / "en_sanity.json"))
    ap.add_argument("--en-n", type=int, default=20, help="EN clips that must decode identically")
    a = ap.parse_args()

    import torch
    from omegaconf import OmegaConf, open_dict

    work = FT / "tokenizers"
    work.mkdir(parents=True, exist_ok=True)
    report = {"env": ftlib.env_info(), "base": a.base, "out": a.out}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. inspect ----
    model = ftlib.load_model(a.base, dev)
    tok = model.tokenizer
    langs = list(tok.langs)
    offsets = dict(tok.token_id_offset)
    v_old = tok.vocab_size
    vocab_old = list(tok.vocabulary)
    text_start = min(off for lg, off in offsets.items() if lg != "spl_tokens")
    rows_old = model.transf_decoder.embedding.token_embedding.weight.shape[0]
    print(f"  base langs {langs} offsets {offsets} vocab {v_old} (embedding rows {rows_old})")
    assert "it" not in langs, "base model already has an it sub-tokenizer"
    assert "<|it|>" in tok.special_tokens, "<|it|> missing from spl_tokens"
    old_dirs = work / "base"
    if not (old_dirs / langs[-1] / "tokenizer.model").exists():
        model.save_tokenizers(str(old_dirs))
    for lg in langs:
        assert (old_dirs / lg / "tokenizer.model").exists(), f"extracted tokenizer missing for {lg}"
    ref = sp_spec(old_dirs / a.init_from / "tokenizer.model")
    report["ref_spec_" + a.init_from] = ref
    print(f"  {a.init_from} spec: {ref}")
    report["base_tokenizer_cfg"] = OmegaConf.to_container(model.cfg.tokenizer, resolve=True)

    # ---- 2. train it SPE on the training pool text ----
    it_dir = work / f"it_{a.vocab}"
    train_it_spm(a.text, it_dir, ref, a.vocab)
    it_spec = sp_spec(it_dir / "tokenizer.model")
    report["it_spec"] = it_spec
    diffs = {k: (ref[k], it_spec[k]) for k in ref if k not in ("vocab_size", "pieces") and ref[k] != it_spec[k]}
    print(f"  it spec: {it_spec}\n  spec differences vs {a.init_from}: {diffs or 'none'}")
    report["spec_differences"] = diffs

    # ---- 3. EN reference with the untouched model ----
    en_rows = ftlib.read_manifest(a.en)[: a.en_n]
    en_before = ftlib.decode(model, en_rows, "en", "en", pnc="yes", batch_size=16, verbose=False)
    emb0, bias0 = emb_and_bias(model)
    emb0, bias0 = emb0.detach().clone().cpu(), bias0.detach().clone().cpu()

    # ---- 4. new aggregate config: old langs in order + it LAST ----
    new_langs = {lg: {"dir": str(old_dirs / lg), "type": "bpe"} for lg in langs}
    new_langs["it"] = {"dir": str(it_dir), "type": "bpe"}
    custom = model.cfg.tokenizer.get("custom_tokenizer", None)
    new_cfg = {"dir": None, "type": "agg", "langs": new_langs}
    new_cfg["custom_tokenizer"] = (OmegaConf.to_container(custom, resolve=True) if custom is not None else
                                   {"_target_": "nemo.collections.common.tokenizers.canary_tokenizer.CanaryTokenizer"})
    new_cfg = OmegaConf.create(new_cfg)
    with open_dict(model.cfg):
        model.cfg.tokenizer = new_cfg  # so the per-lang cfg write-back in _setup_aggregate_tokenizer finds `it`
    model.change_vocabulary(new_tokenizer_dir=new_cfg, new_tokenizer_type="agg")
    tok = model.tokenizer
    v_new = tok.vocab_size
    for lg in langs:
        assert tok.token_id_offset[lg] == offsets[lg], f"offset of {lg} moved: {offsets[lg]} -> {tok.token_id_offset[lg]}"
    assert list(tok.vocabulary[:v_old]) == vocab_old, "existing pieces changed position"
    print(f"  new vocab {v_new} (it offset {tok.token_id_offset['it']}, it size {v_new - v_old})")

    # ---- 5. copy back old rows, seed it rows ----
    emb, bias = emb_and_bias(model)
    n_rows = emb.shape[0]
    with torch.no_grad():
        emb[:v_old].copy_(emb0[:v_old].to(emb.device))
        bias[:v_old].copy_(bias0[:v_old].to(bias.device))
        src_tok = tok.tokenizers_dict[a.init_from]
        src_off = tok.token_id_offset[a.init_from]
        it_tok = tok.tokenizers_dict["it"]
        src_vocab = {p: i for i, p in enumerate(src_tok.vocab)}
        text_mean_e = emb0[text_start:v_old].mean(0)
        text_mean_b = bias0[text_start:v_old].mean(0)
        how = {"exact": 0, "decomposed": 0, "mean": 0}
        for j, piece in enumerate(it_tok.vocab):
            r = v_old + j
            if piece in src_vocab:
                ids = [src_off + src_vocab[piece]]
                how["exact"] += 1
            else:
                surface = piece.replace("▁", " ")
                ids = [src_off + i for i in src_tok.text_to_ids(surface)] if surface.strip() else []
                if ids and not piece.startswith("▁"):
                    toks = src_tok.ids_to_tokens([i - src_off for i in ids])
                    if toks and toks[0] == "▁" and len(ids) > 1:
                        ids = ids[1:]
                how["decomposed" if ids else "mean"] += 1
            if ids:
                emb[r].copy_(emb0[ids].mean(0).to(emb.device))
                bias[r].copy_(bias0[ids].mean(0).to(bias.device))
            else:
                emb[r].copy_(text_mean_e.to(emb.device))
                bias[r].copy_(text_mean_b.to(bias.device))
        # padding rows (vocab rounded up to a multiple of 8): never targets; keep them low
        if n_rows > v_new:
            emb[v_new:].zero_()
            bias[v_new:].fill_(-1e4)
    report["it_row_init"] = {"from": a.init_from, **how, "padding_rows": n_rows - v_new}
    print(f"  it rows init {report['it_row_init']}")

    # ---- 6. save, restore fresh, verify ----
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp.nemo"
    model.save_to(tmp)
    del model
    torch.cuda.empty_cache()
    m2 = ftlib.load_model(tmp, dev)
    emb2, bias2 = emb_and_bias(m2)
    rows_ok = bool(torch.equal(emb2[:v_old].cpu(), emb0[:v_old]) and torch.equal(bias2[:v_old].cpu(), bias0[:v_old]))
    en_after = ftlib.decode(m2, en_rows, "en", "en", pnc="yes", batch_size=16, verbose=False)
    en_same = [x == y for x, y in zip(en_before, en_after)]
    sample = [r["text"] for r in ftlib.read_manifest(FT / "manifests" / "train_5h.json")[:5]]
    try:
        it_unk = v_old + m2.tokenizer.tokenizers_dict["it"].tokenizer.unk_id()
    except Exception:
        it_unk = None
    rt = []
    for s in sample:
        ids = m2.tokenizer.text_to_ids(s, "it")
        back = m2.tokenizer.ids_to_text(ids).strip()
        rt.append({"text": s, "n_ids": len(ids), "min_id": min(ids), "unk": sum(1 for i in ids if i == it_unk),
                   "roundtrip_ok": back == s.strip()})
    it_ok = all(x["min_id"] >= v_old for x in rt)
    report.update({
        "v_old": v_old, "v_new": v_new, "embedding_rows": n_rows, "it_offset": m2.tokenizer.token_id_offset["it"],
        "old_rows_bit_identical": rows_ok,
        "en_identical": f"{sum(en_same)}/{len(en_same)}",
        "en_pairs": [{"before": x, "after": y} for x, y in zip(en_before, en_after)],
        "it_roundtrip": rt,
        "it_ids_all_in_it_range": it_ok,
        "prompt_it_ids": ftlib.prompt_ids(m2, "it", "it", "no").tolist(),
        "prompt_it_pieces": m2.tokenizer.ids_to_tokens(ftlib.prompt_ids(m2, "it", "it", "no").tolist()),
    })
    ftlib.write_json(FT / "tokenizers" / "surgery_report.json", report)
    print(f"  old rows bit-identical: {rows_ok}; EN identical {report['en_identical']}; "
          f"it ids in range: {it_ok}; roundtrip {sum(x['roundtrip_ok'] for x in rt)}/{len(rt)}")
    for x, y in zip(en_before[:3], en_after[:3]):
        print(f"    EN before: {x[:90]!r}\n    EN after : {y[:90]!r}")
    if not (rows_ok and all(en_same) and it_ok):
        print("SURGERY CHECK FAILED -- see tokenizers/surgery_report.json")
        return 1
    os.replace(tmp, a.out)
    print(f"  wrote {a.out} ({time.strftime('%T')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
