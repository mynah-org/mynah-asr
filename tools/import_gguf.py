#!/usr/bin/env python3
"""Import a third-party GGUF (parakeet.cpp ecosystem) into a model directory
mynah completa — SENZA torch e senza scaricare il .nemo: dal solo file GGUF
genera mynah.json (dai metadata stt.*), tokens.json (tokenizer embedded),
mel_filters.safetensors (calcolate) e model.gguf (tensori RINOMINATI nel
naming HF-verbatim del runtime; payload quantizzati copiati verbatim).

Uso: uv run python import_gguf.py <file.gguf> --out <model_dir>

v1: architettura "parakeet" con head TDT (es. handy-computer/parakeet-*-gguf).
La config resta autorità di mynah.json (regola repo): i metadata GGUF vengono
tradotti una volta all'import, il runtime non li legge mai direttamente.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_nemo import melscale_fbanks  # noqa: E402  (clone slaney verificato)

ALIGN = 32
# geometry of the ggml types the runtime supports (block_elems, block_bytes)
GEOM = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 2: (32, 18), 12: (256, 144)}

# parakeet.cpp -> HF-verbatim naming (same semantics as the converter's
# _NEMO_RENAMES; verified tensor by tensor against the 110m converted from .nemo)
RENAMES = [
    ("enc.pre_encode.conv.", "encoder.subsampling.layers."),
    ("enc.pre_encode.out.", "encoder.subsampling.linear."),
    ("enc.blocks.", "encoder.layers."),
    (".norm_ff1.", ".norm_feed_forward1."),
    (".norm_ff2.", ".norm_feed_forward2."),
    (".ff1.", ".feed_forward1."),
    (".ff2.", ".feed_forward2."),
    (".norm_attn.", ".norm_self_att."),
    (".attn.linear_q.", ".self_attn.q_proj."),
    (".attn.linear_k.", ".self_attn.k_proj."),
    (".attn.linear_v.", ".self_attn.v_proj."),
    (".attn.linear_out.", ".self_attn.o_proj."),
    (".attn.linear_pos.", ".self_attn.relative_k_proj."),
    (".attn.pos_bias_u", ".self_attn.bias_u"),
    (".attn.pos_bias_v", ".self_attn.bias_v"),
    (".conv.bn.", ".conv.norm."),
    (".conv.depthwise.", ".conv.depthwise_conv."),
    (".conv.pointwise1.", ".conv.pointwise_conv1."),
    (".conv.pointwise2.", ".conv.pointwise_conv2."),
    ("pred.embed.", "decoder.embedding."),
    ("joint.enc.", "encoder_projector."),
    ("joint.pred.", "decoder.decoder_projector."),
    ("joint.out.", "joint.head."),
]


def gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


class Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.off = 0

    def u32(self):
        v = struct.unpack_from("<I", self.d, self.off)[0]
        self.off += 4
        return v

    def u64(self):
        v = struct.unpack_from("<Q", self.d, self.off)[0]
        self.off += 8
        return v

    def s(self):
        n = self.u64()
        v = self.d[self.off:self.off + n].decode()
        self.off += n
        return v

    SIMPLE = {0: ("B", 1), 1: ("b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
              5: ("<i", 4), 6: ("<f", 4), 7: ("B", 1), 10: ("<Q", 8), 11: ("<q", 8),
              12: ("<d", 8)}

    def value(self, typ):
        if typ == 8:
            return self.s()
        if typ == 9:
            et, cnt = self.u32(), self.u64()
            return [self.value(et) for _ in range(cnt)]
        fmt, sz = self.SIMPLE[typ]
        v = struct.unpack_from(fmt, self.d, self.off)[0]
        self.off += sz
        return v


import re  # noqa: E402

def rename(name: str) -> str:
    # LSTM: parakeet.cpp fuses the two torch biases into one (b_ih + b_hh, which
    # the LSTM sums anyway): bias -> bias_ih, and bias_hh is synthesized as 0
    m = re.fullmatch(r"pred\.lstm\.(\d+)\.(Wx|Wh|bias)", name)
    if m:
        part = {"Wx": "weight_ih", "Wh": "weight_hh", "bias": "bias_ih"}[m.group(2)]
        return f"decoder.lstm.{part}_l{m.group(1)}"
    for old, new in RENAMES:
        name = name.replace(old, new)
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = args.gguf.read_bytes()
    r = Reader(data)
    magic, ver = r.u32(), r.u32()
    if magic != 0x46554747 or ver < 2:
        print("error: not a GGUF v2/v3 file", file=sys.stderr)
        return 1
    n_t, n_kv = r.u64(), r.u64()
    meta = {}
    for _ in range(n_kv):
        key = r.s()
        meta[key] = r.value(r.u32())

    if meta.get("general.architecture") != "parakeet":
        print(f"error: unsupported architecture '{meta.get('general.architecture')}' "
              "(v1: parakeet)", file=sys.stderr)
        return 1
    if meta.get("stt.parakeet.head_kind", "tdt") != "tdt":
        print("error: TDT head only in v1", file=sys.stderr)
        return 1

    tensors = []
    for _ in range(n_t):
        name = r.s()
        rank = r.u32()
        ne = [r.u64() for _ in range(rank)]
        typ, toff = r.u32(), r.u64()
        if typ not in GEOM:
            print(f"error: tensor '{name}' has ggml type {typ}, unsupported by the runtime",
                  file=sys.stderr)
            return 1
        tensors.append((name, ne, typ, toff))
    align = int(meta.get("general.alignment", 32))
    base = (r.off + align - 1) // align * align

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- mynah.json from the metadata (one-off translation, config-driven) ----
    g = lambda k: meta[f"stt.parakeet.{k}"]
    f = lambda k: meta[f"stt.frontend.{k}"]
    # canonical name: basename of the upstream repo_url (e.g. parakeet-tdt_ctc-110m)
    # — it is the key tests and tooling use to recognize the model
    canon = meta.get("general.repo_url", "").rstrip("/").rsplit("/", 1)[-1] \
        or meta.get("general.name", args.out.name)
    mynah = {
        "mynah_asr_format": 1,
        "name": canon,
        "arch": "fastconformer_tdt",
        "engine": "parakeet-tdt",
        "weights": "model.gguf",
        "imported_from": str(args.gguf.name),
        "features": {
            "sample_rate": f("sample_rate"), "n_mels": f("num_mels"),
            "n_fft": f("n_fft"), "win_length": f("win_length"),
            "hop_length": f("hop_length"), "preemphasis": round(f("pre_emphasis"), 4),
            "log_zero_guard": 2.0 ** -24, "normalize": f("normalize"),
            "dither": 0.0,                      # inference: mai dither (trappole §6)
            "mel_filters": "mel_filters.safetensors",
        },
        "encoder": {
            "n_layers": g("encoder.n_layers"), "d_model": g("encoder.d_model"),
            "n_heads": g("encoder.n_heads"), "ffn_dim": g("encoder.d_ff"),
            "conv_kernel": g("encoder.conv_kernel"),
            "conv_norm": g("encoder.conv_norm_type"),
            "subsampling": "dw_striding",
            "subsampling_factor": g("encoder.subsampling_factor"),
            "subsampling_conv_channels": g("encoder.subsampling_channels"),
            "use_bias": bool(g("encoder.use_bias")),
            "activation": "silu",
            "att_context_style": g("encoder.att_context_style"),
            "pos_emb_max_len": g("encoder.pos_emb_max_len"),
            "xscaling": bool(g("encoder.xscaling")),
        },
        "decoder": {
            "type": "tdt_lstm",
            "pred_hidden": g("predictor.hidden"), "pred_layers": g("predictor.n_layers"),
            "joint_hidden": g("joint.hidden"), "joint_activation": g("joint.activation"),
            "vocab_size": g("predictor.vocab"),
            "blank_id": g("predictor.vocab") - 1,
            "max_symbols_per_step": g("tdt.max_symbols"),
            "durations": g("tdt.durations"),
        },
        "tokenizer": {"type": "spe_bpe", "pieces": "tokens.json"},
    }
    (args.out / "mynah.json").write_text(json.dumps(mynah, indent=1, ensure_ascii=False))

    # ---- tokens.json from the embedded tokenizer ----
    (args.out / "tokens.json").write_text(
        json.dumps(meta["tokenizer.ggml.tokens"], ensure_ascii=False))

    # ---- mel filterbank + window (same construction as the HF converter) ----
    fb = melscale_fbanks(f("n_fft") // 2 + 1, f("f_min"), f("f_max"),
                         f("num_mels"), f("sample_rate"))
    save_file({"mel_fb": np.ascontiguousarray(fb.astype(np.float32)),
               "window": np.hanning(f("win_length")).astype(np.float32)},
              args.out / "mel_filters.safetensors")

    # ---- model.gguf: tensori rinominati, payload verbatim ----
    infos, payload = bytearray(), bytearray()
    renamed = set()
    n_out = 0
    for name, ne, typ, toff in tensors:
        nn = rename(name)
        if nn in renamed:
            print(f"error: duplicate rename '{name}' -> '{nn}'", file=sys.stderr)
            return 1
        renamed.add(nn)
        be, bb = GEOM[typ]
        elems = int(np.prod(ne))
        nbytes = elems // be * bb
        while len(payload) % ALIGN != 0:
            payload += b"\0"
        infos += gguf_string(nn) + struct.pack("<I", len(ne))
        for d in ne:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", typ) + struct.pack("<Q", len(payload))
        payload += data[base + toff: base + toff + nbytes]
        n_out += 1
        if (m := re.fullmatch(r"decoder\.lstm\.bias_ih_l(\d+)", nn)):
            # synthetic zero bias_hh (parakeet.cpp's fused one is already in bias_ih)
            while len(payload) % ALIGN != 0:
                payload += b"\0"
            infos += gguf_string(f"decoder.lstm.bias_hh_l{m.group(1)}")
            infos += struct.pack("<I", 1) + struct.pack("<Q", ne[0])
            infos += struct.pack("<I", 0) + struct.pack("<Q", len(payload))
            payload += b"\0" * (int(ne[0]) * 4)
            n_out += 1
    header = struct.pack("<IIQQ", 0x46554747, 3, n_out, 2)
    header += gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("mynah")
    header += gguf_string("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", ALIGN)
    body = header + bytes(infos)
    pad = (ALIGN - len(body) % ALIGN) % ALIGN
    (args.out / "model.gguf").write_bytes(body + b"\0" * pad + bytes(payload))

    size_mb = (args.out / "model.gguf").stat().st_size / 1e6
    print(f"OK {args.out}: {len(tensors)} tensors renamed, model.gguf {size_mb:.1f} MB")
    print(f"   try it: ./mynah-asr transcribe -m {args.out} -i file.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
