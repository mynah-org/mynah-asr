#!/usr/bin/env python3
"""Esporta i pesi convertiti (model.safetensors, f32) in un container GGUF v3.

Il GGUF trasporta SOLO i pesi: mynah.json, mel_filters e tokens restano gli
stessi — il runtime li legge dalla dir del modello come sempre. Uso:

    uv run python export_gguf.py ../models/parakeet-tdt_ctc-110m [--dtype q8_0]

--dtype f32   (default) round-trip bit-esatto, stessa RAM
--dtype f16 | q8_0 | q4_0   file più piccolo; i tensori con l'ultima dim non
        multipla di 32 (bias, norm) restano f32, come fa llama.cpp.
Convenzioni ggml: dims invertite (ne[0] = la più veloce), alignment 32.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ALIGN = 32
GGML = {"f32": 0, "f16": 1, "q4_0": 2, "q8_0": 8, "q4_k": 12}


def gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def quantize_q8_0(x: np.ndarray) -> bytes:
    """Blocchi da 32: d f16 + 32 int8 (q = round(x/d), d = amax/127)."""
    blocks = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(blocks).max(axis=1)
    d = amax / 127.0
    q = np.where(d[:, None] > 0, np.round(blocks / np.where(d[:, None] > 0, d[:, None], 1)), 0)
    q = np.clip(q, -127, 127).astype(np.int8)
    out = bytearray()
    d16 = d.astype(np.float16)
    for i in range(blocks.shape[0]):
        out += d16[i].tobytes() + q[i].tobytes()
    return bytes(out)


def quantize_q4_0(x: np.ndarray) -> bytes:
    """Blocchi da 32: d f16 + 16 byte nibble (ggml: d = max_signed/-8)."""
    blocks = x.reshape(-1, 32).astype(np.float32)
    idx = np.abs(blocks).argmax(axis=1)
    maxs = blocks[np.arange(blocks.shape[0]), idx]          # col segno, come ggml
    d = maxs / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0)
    q = np.clip(np.round(blocks * inv[:, None]) + 8, 0, 15).astype(np.uint8)
    packed = (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)
    out = bytearray()
    d16 = d.astype(np.float16)
    for i in range(blocks.shape[0]):
        out += d16[i].tobytes() + packed[i].tobytes()
    return bytes(out)


def quantize_q4_k(x: np.ndarray) -> bytes:
    """Super-blocchi da 256 (8 sotto-blocchi da 32): x = d*sc*q - dmin*m.
    Quantizzatore semplice non-iterativo (ggml usa una ricerca più fine): basta
    per validare il layout del dequant C con la regola oracle."""
    blocks = x.reshape(-1, 8, 32).astype(np.float64)
    lo = np.minimum(blocks.min(axis=2), 0.0)              # m_j >= 0
    hi = np.maximum(blocks.max(axis=2), 0.0)
    scale = (hi - lo) / 15.0                              # per sotto-blocco
    mins = -lo
    d = scale.max(axis=1) / 63.0                          # globali del super-blocco
    dmin = mins.max(axis=1) / 63.0
    d16 = d.astype(np.float16).astype(np.float64)         # arrotonda come su file
    dmin16 = dmin.astype(np.float16).astype(np.float64)
    sc6 = np.where(d16[:, None] > 0, np.round(scale / np.where(d16[:, None] > 0, d16[:, None], 1)), 0)
    mn6 = np.where(dmin16[:, None] > 0, np.round(mins / np.where(dmin16[:, None] > 0, dmin16[:, None], 1)), 0)
    sc6 = np.clip(sc6, 0, 63).astype(np.uint8)
    mn6 = np.clip(mn6, 0, 63).astype(np.uint8)
    eff_d = d16[:, None] * sc6                            # scala/min effettivi
    eff_m = dmin16[:, None] * mn6
    q = np.where(eff_d[:, :, None] > 0,
                 np.round((blocks + eff_m[:, :, None]) / np.where(eff_d[:, :, None] > 0, eff_d[:, :, None], 1)), 0)
    q = np.clip(q, 0, 15).astype(np.uint8)
    out = bytearray()
    for b in range(blocks.shape[0]):
        scales = bytearray(12)
        for j in range(4):                                # layout ggml dei 6-bit
            scales[j] = sc6[b, j] | ((sc6[b, j + 4] >> 4) << 6)
            scales[j + 4] = mn6[b, j] | ((mn6[b, j + 4] >> 4) << 6)
            scales[j + 8] = (sc6[b, j + 4] & 0x0F) | ((mn6[b, j + 4] & 0x0F) << 4)
        qs = bytearray()
        for grp in range(4):                              # 4 gruppi da 64: i | i+32
            g = q[b, 2 * grp : 2 * grp + 2].reshape(64)
            qs += (g[:32] | (g[32:] << 4)).astype(np.uint8).tobytes()
        out += np.float16(d16[b]).tobytes() + np.float16(dmin16[b]).tobytes() + scales + qs
    return bytes(out)


def encode_tensor(x: np.ndarray, dtype: str) -> tuple[int, bytes]:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype != "f32" and (x.ndim < 2 or x.shape[-1] % 32 != 0):
        # solo le matrici si quantizzano: bias/norm/running-stats 1-D restano f32
        # (stile llama.cpp; quantizzare running_var rompe 1/sqrt(var) del BatchNorm)
        dtype = "f32"
    if dtype == "q4_k" and x.shape[-1] % 256 != 0:
        dtype = "q8_0"                      # fallback K-quant (come llama.cpp)
    if dtype == "f32":
        return GGML["f32"], x.tobytes()
    if dtype == "f16":
        return GGML["f16"], x.astype(np.float16).tobytes()
    if dtype == "q8_0":
        return GGML["q8_0"], quantize_q8_0(x)
    if dtype == "q4_k":
        return GGML["q4_k"], quantize_q4_k(x)
    return GGML["q4_0"], quantize_q4_0(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--dtype", choices=list(GGML), default="f32")
    ap.add_argument("--out", type=Path, default=None, help="default: <model_dir>/model.gguf")
    args = ap.parse_args()

    st_path = args.model_dir / "model.safetensors"
    if not st_path.exists():
        print(f"error: {st_path} is missing (a converted model is required)", file=sys.stderr)
        return 1
    name = args.model_dir.name
    cfg = args.model_dir / "mynah.json"
    if cfg.exists():
        name = json.loads(cfg.read_text()).get("name", name)

    tensors = load_file(st_path)
    out_path = args.out or (args.model_dir / "model.gguf")

    meta = (
        gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("mynah")
        + gguf_string("general.name") + struct.pack("<I", 8) + gguf_string(name)
        + gguf_string("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", ALIGN)
    )
    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 3) + meta

    infos, payload = bytearray(), bytearray()
    for tname, x in tensors.items():
        ggml_type, blob = encode_tensor(x, args.dtype)
        while len(payload) % ALIGN != 0:
            payload += b"\0"
        dims = list(x.shape)[::-1] or [1]   # convenzione ggml: ne[0] = la più veloce
        infos += gguf_string(tname) + struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", ggml_type) + struct.pack("<Q", len(payload))
        payload += blob

    body = header + bytes(infos)
    pad = (ALIGN - len(body) % ALIGN) % ALIGN
    out_path.write_bytes(body + b"\0" * pad + bytes(payload))
    size_mb = out_path.stat().st_size / 1e6
    print(f"OK {out_path} [{args.dtype}] {len(tensors)} tensors, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
