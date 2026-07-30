#!/usr/bin/env python3
"""Generate a GGUF K-quant fixture plus reference dequants from llama.cpp's own code.

Why: our C dequant for Q4_K/Q5_K/Q6_K (src/gguf.c) and the synthetic fixtures in
tests/test_gguf.c were written from the same reading of the spec, so a shared
misunderstanding of the layout would pass unnoticed. This script builds the
fixture with the `gguf` package (the reference implementation shipped with
llama.cpp): it WRITES the file with GGUFWriter and dequantizes with
gguf.quants.dequantize, so the resulting .npy files are an independent oracle for
both our parser (offsets, alignment, block geometry) and our dequant math.

Note: the package's pure-numpy `quantize` raises NotImplementedError for the
K-quants, so the blocks are filled with deterministic pseudo-random bytes instead
of quantizing real data. That is actually stronger coverage: every scale, min,
nibble and high-bit position gets exercised, whereas quantized smooth data leaves
most bit patterns untouched. The f16 scale fields are drawn from a finite,
moderate range so no NaN/Inf enters the comparison.

Usage: uv run --with gguf python gen_kquant_ref.py <out_dir>
Writes <out_dir>/kquant.gguf and <out_dir>/<tensor>.npy (float32, C order).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import dequantize

SEED = 20260730
N_BLOCKS = 6                    # super-blocks per tensor (256 elements each)
ROWS = 2                        # ne[0] = N_BLOCKS//ROWS * 256: blocks never straddle rows


def rand_bytes(rng: np.random.Generator, n: int) -> bytes:
    return bytes(rng.integers(0, 256, n, dtype=np.uint8))


def f16_bytes(rng: np.random.Generator, lo: float, hi: float) -> bytes:
    return np.asarray([rng.uniform(lo, hi)], dtype=np.float16).tobytes()


def build_q4k(rng: np.random.Generator) -> bytes:
    """144B: d f16 + dmin f16 + 12B packed 6-bit scales/mins + 128B nibbles."""
    out = bytearray()
    for _ in range(N_BLOCKS):
        out += f16_bytes(rng, 0.01, 0.5)      # d
        out += f16_bytes(rng, 0.01, 0.3)      # dmin
        out += rand_bytes(rng, 12)            # scales/mins
        out += rand_bytes(rng, 128)           # qs
    return bytes(out)


def build_q5k(rng: np.random.Generator) -> bytes:
    """176B: Q4_K layout plus qh[32] carrying the 5th bit of every quant."""
    out = bytearray()
    for _ in range(N_BLOCKS):
        out += f16_bytes(rng, 0.01, 0.5)      # d
        out += f16_bytes(rng, 0.01, 0.3)      # dmin
        out += rand_bytes(rng, 12)            # scales/mins
        out += rand_bytes(rng, 32)            # qh
        out += rand_bytes(rng, 128)           # qs
    return bytes(out)


def build_q6k(rng: np.random.Generator) -> bytes:
    """210B: ql[128] + qh[64] + 16 signed 8-bit scales + d f16 (no mins)."""
    out = bytearray()
    for _ in range(N_BLOCKS):
        out += rand_bytes(rng, 128)                                   # ql
        out += rand_bytes(rng, 64)                                    # qh
        out += bytes(rng.integers(-128, 128, 16, dtype=np.int8))      # signed scales
        out += f16_bytes(rng, 0.01, 0.5)                              # d
    return bytes(out)


CASES = (
    ("t.q4k", T.Q4_K, 144, build_q4k),
    ("t.q5k", T.Q5_K, 176, build_q5k),
    ("t.q6k", T.Q6_K, 210, build_q6k),
)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    blocks_per_row = N_BLOCKS // ROWS
    shape = (ROWS, blocks_per_row * 256)          # logical, row-major
    writer = GGUFWriter(str(out_dir / "kquant.gguf"), "mynah")
    refs = {}
    for name, qtype, block_bytes, build in CASES:
        raw = build(rng)
        # GGUFWriter wants raw_shape as the BYTE shape of the payload; it derives
        # the logical ne[] from it (and reverses it into ggml order)
        byte_shape = (ROWS, blocks_per_row * block_bytes)
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(byte_shape)
        writer.add_tensor(name, blocks, raw_shape=byte_shape, raw_dtype=qtype)
        refs[name] = dequantize(blocks, qtype).reshape(shape).astype(np.float32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    for name, ref in refs.items():
        np.save(out_dir / f"{name}.npy", ref)
    print(f"OK {out_dir}/kquant.gguf + {len(refs)} .npy references "
          f"({N_BLOCKS} super-blocks each, shape {shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
