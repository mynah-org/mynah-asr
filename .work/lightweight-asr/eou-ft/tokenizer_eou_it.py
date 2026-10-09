#!/usr/bin/env python3
"""Italian SentencePiece for parakeet_realtime_eou_120m-v1, with <EOU>/<EOB> appended.

    $PY tokenizer_eou_it.py --stock /root/ft/models/parakeet_realtime_eou_120m-v1.nemo \
        --text /root/ft/text/it_pool.txt --out /root/ft/models/tok-eou-it

1. Read the stock tokenizer.model out of the .nemo: model type, vocab size,
   special ids, normaliser rule (the Italian model copies them; nothing assumed).
2. Normalise the training-pool text like the stock targets (lower case, no
   punctuation, apostrophes kept: `norm_text`) and train an Italian SPE with
   vocab = stock base vocab (1024 incl. <unk>).
3. Append <EOU>, <EOB> as USER_DEFINED pieces (type 4, score 0) at the END, the
   way NeMo's scripts/asr_eou/tokenizers/add_special_tokens_to_sentencepiece.py
   does (v3.0.0 l.124-138), write tokenizer.model / tokenizer.vocab / vocab.txt.
4. Verify by reload (sentencepiece AND NeMo's SentencePieceTokenizer):
   <EOU>=1024, <EOB>=1025, vocab 1026 (the RNNT blank becomes 1026), the last two
   ids decode to exactly those strings (what LhotseSpeechToTextBpeEOUDataset
   checks), "ciao come stai<EOU>" round-trips with <EOU> as ONE id.
`--dry-run` runs 2-4 on a tiny built-in text without a .nemo (laptop check).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import tempfile
import unicodedata
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
SPECIAL = ["<EOU>", "<EOB>"]
_KEEP = re.compile(r"[^a-zàáèéìíîòóùú' ]")


def norm_text(s: str) -> str:
    """Training-target normalisation (stock EOU targets are lower case, no PnC).
    Keeps Italian accented vowels and the apostrophe ("l'acqua"), maps typographic
    apostrophes to ', drops everything else (digits included: MLS has none)."""
    s = unicodedata.normalize("NFC", s.lower()).replace("’", "'").replace("`", "'")
    s = _KEEP.sub(" ", s)
    s = re.sub(r"\s*'\s*", "'", s)
    return re.sub(r"\s+", " ", s).strip()


def stock_spec(model_bytes: bytes) -> dict:
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(model_bytes)
    ts, ns = m.trainer_spec, m.normalizer_spec
    types = [p.type for p in m.pieces]
    return {
        "model_type": pb.TrainerSpec.ModelType.Name(ts.model_type).lower(),
        "pieces_total": len(m.pieces),
        "pieces_user_defined_tail": [p.piece for p in m.pieces[-2:]],
        "n_user_defined": sum(t == 4 for t in types),
        "vocab_size_trainer": ts.vocab_size,
        "character_coverage": ts.character_coverage,
        "byte_fallback": ts.byte_fallback,
        "split_digits": ts.split_digits,
        "unk_id": ts.unk_id, "bos_id": ts.bos_id, "eos_id": ts.eos_id, "pad_id": ts.pad_id,
        "normalization_rule_name": ns.name or "nmt_nfkc",
        "add_dummy_prefix": ns.add_dummy_prefix,
        "remove_extra_whitespaces": ns.remove_extra_whitespaces,
    }


def read_stock_tokenizer(nemo: Path) -> bytes:
    with tarfile.open(nemo) as tar:
        hits = [n for n in tar.getnames() if n.endswith("tokenizer.model")]
        assert len(hits) == 1, f"tokenizer.model in {nemo}: {hits}"
        return tar.extractfile(hits[0]).read()


def train_spm(text_file: Path, out_dir: Path, spec: dict, base_vocab: int) -> Path:
    import sentencepiece as spm

    prefix = out_dir / "base"
    spm.SentencePieceTrainer.train(
        input=str(text_file), model_prefix=str(prefix), vocab_size=base_vocab,
        model_type=spec["model_type"], character_coverage=1.0,
        byte_fallback=spec["byte_fallback"], split_digits=spec["split_digits"],
        unk_id=spec["unk_id"], bos_id=spec["bos_id"], eos_id=spec["eos_id"], pad_id=spec["pad_id"],
        normalization_rule_name=spec["normalization_rule_name"],
        input_sentence_size=2_000_000, shuffle_input_sentence=True, num_threads=os.cpu_count() or 4,
        hard_vocab_limit=True,
    )
    return prefix.with_suffix(".model")


def add_special(base_model: Path, out_dir: Path) -> None:
    """add_special_tokens_to_sentencepiece.py::edit_spt_model, is_userdefined=True."""
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(base_model.read_bytes())
    for tok in SPECIAL:
        assert all(p.piece != tok for p in m.pieces), f"{tok} already in the base model"
        m.pieces.append(pb.ModelProto.SentencePiece(piece=tok, score=0.0, type=4))
    (out_dir / "tokenizer.model").write_bytes(m.SerializeToString())
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(m.SerializeToString())
    with open(out_dir / "tokenizer.vocab", "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
    skip = {"<s>", "</s>", "<pad>", "<unk>", *SPECIAL}
    with open(out_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            p = sp.id_to_piece(i)
            if p in skip:
                continue
            t = p[1:] if p.startswith("▁") else f"##{p}"
            if t:
                f.write(t + "\n")


def verify(out_dir: Path, base_vocab: int, nemo_check: bool) -> dict:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(out_dir / "tokenizer.model"))
    n = sp.get_piece_size()
    eou, eob = sp.piece_to_id("<EOU>"), sp.piece_to_id("<EOB>")
    ids = sp.encode("ciao come stai<EOU>")
    res = {"vocab": n, "eou_id": eou, "eob_id": eob, "blank_id_in_model": n,
           "probe_ids": ids, "probe_pieces": [sp.id_to_piece(i) for i in ids]}
    assert n == base_vocab + 2, res
    assert (eou, eob) == (base_vocab, base_vocab + 1), res
    assert ids[-1] == eou and ids.count(eou) == 1, f"<EOU> not a single piece: {res}"
    if nemo_check:
        from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

        t = SentencePieceTokenizer(str(out_dir / "tokenizer.model"))
        last2 = {t.ids_to_text([t.vocab_size - 1]), t.ids_to_text([t.vocab_size - 2])}
        res["nemo_vocab_size"] = t.vocab_size
        res["nemo_last_two"] = sorted(last2)
        assert last2 == set(SPECIAL), res  # LhotseSpeechToTextBpeEOUDataset._check_special_tokens
        res["nemo_probe_text"] = t.ids_to_text(t.text_to_ids("ciao come stai<EOU>"))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", help="stock parakeet_realtime_eou_120m-v1.nemo")
    ap.add_argument("--text", help="training-pool transcripts, one per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="tiny synthetic text, no .nemo, no NeMo import")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.dry_run:
        spec = {"model_type": "bpe", "byte_fallback": False, "split_digits": False, "unk_id": 0, "bos_id": -1,
                "eos_id": -1, "pad_id": -1, "normalization_rule_name": "nmt_nfkc"}
        base_vocab = 120
        words = ("la casa è bella l'acqua scorre nel fiume perché così città più già "
                 "ciao come stai oggi domani sempre").split()
        lines = [" ".join(words[(i * 7 + j) % len(words)] for j in range(12)) for i in range(400)]
        raw = Path(tempfile.mkdtemp()) / "pool.txt"
        raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
        a.text = str(raw)
    else:
        assert a.stock and a.text, "--stock and --text are required"
        sb = read_stock_tokenizer(Path(a.stock))
        spec = stock_spec(sb)
        (out / "stock_tokenizer.model").write_bytes(sb)
        print(f"  stock tokenizer: {spec}", flush=True)
        assert spec["pieces_user_defined_tail"] == SPECIAL, spec
        base_vocab = spec["pieces_total"] - 2  # 1026 - <EOU> - <EOB>
    norm = out / "text_norm.txt"
    n_in = n_out = 0
    with open(a.text, encoding="utf-8") as fi, open(norm, "w", encoding="utf-8") as fo:
        for line in fi:
            n_in += 1
            t = norm_text(line)
            if t:
                fo.write(t + "\n")
                n_out += 1
    print(f"  text: {n_in} lines -> {n_out} normalised ({norm})", flush=True)
    base = train_spm(norm, out, spec, base_vocab)
    add_special(base, out)
    res = verify(out, base_vocab, nemo_check=not a.dry_run)
    res.update({"stock_spec": spec, "text_lines": n_out, "normaliser": "eou-ft/tokenizer_eou_it.py::norm_text"})
    (out / "tokenizer_info.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"== tokenizer OK: {json.dumps({k: res[k] for k in ('vocab', 'eou_id', 'eob_id', 'blank_id_in_model', 'probe_pieces')}, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
