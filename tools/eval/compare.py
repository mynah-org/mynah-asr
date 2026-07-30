#!/usr/bin/env python3
"""Compare two directories of .npy dumps with per-stage tolerances (M0.4).

Use cases: diffing the oracle goldens across converter/oracle versions
(regenerate into a new dir and compare with the old one), or any pair of
homogeneous dumps (same file names). The metric follows the C tests: maximum
absolute error SCALED by the stage's max|ref| (err_rel = max|a-b| / max|ref|).

Usage: uv run python -m eval.compare <dir_ref> <dir_test> [--tol nome=1e-4 ...]
Exit: 0 every stage within tolerance, 1 otherwise, 77 empty/missing dirs.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import numpy as np

# Default per-stage tolerances (fnmatch pattern on the file name without .npy),
# aligned with the C tests: bit-exact mel, deep stages via f32 accumulation.
DEFAULT_TOLS: list[tuple[str, float]] = [
    ("mel", 0.0),
    ("subsampling", 1e-4),
    ("layer_*", 3.5e-2),      # like tests/test_encoder.c (24-layer accumulation)
    ("encoder_out", 3.5e-2),
    ("enc_proj", 3.5e-2),
    ("*", 1e-3),              # default for unmapped stages
]


def tol_for(name: str, tols: list[tuple[str, float]]) -> float:
    for pat, t in tols:
        if fnmatch.fnmatch(name, pat):
            return t
    return 1e-3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir_ref", type=Path)
    ap.add_argument("dir_test", type=Path)
    ap.add_argument("--tol", action="append", default=[],
                    help="per-stage override: name=tolerance (repeatable)")
    args = ap.parse_args()

    tols = list(DEFAULT_TOLS)
    for spec in args.tol:
        name, _, val = spec.partition("=")
        tols.insert(0, (name, float(val)))

    refs = sorted(args.dir_ref.glob("*.npy")) if args.dir_ref.is_dir() else []
    if not refs:
        print(f"SKIP: no .npy in {args.dir_ref}")
        return 77

    fail = 0
    print(f"{'stage':16} {'shape':>16} {'max|d|':>10} {'mean|d|':>10} "
          f"{'err_rel':>10} {'tol':>9}  result")
    for ref_path in refs:
        name = ref_path.stem
        test_path = args.dir_test / ref_path.name
        if not test_path.exists():
            print(f"{name:16} {'-':>16} {'-':>10} {'-':>10} {'-':>10} {'-':>9}  MISSING")
            fail = 1
            continue
        a = np.load(ref_path).astype(np.float64)
        b = np.load(test_path).astype(np.float64)
        if a.shape != b.shape:
            print(f"{name:16} {str(a.shape):>16} vs {b.shape}  SHAPE MISMATCH")
            fail = 1
            continue
        d = np.abs(a - b)
        scale = max(float(np.abs(a).max()), 1e-12)
        rel = float(d.max()) / scale
        tol = tol_for(name, tols)
        ok = rel <= tol
        if not ok:
            fail = 1
        print(f"{name:16} {str(a.shape):>16} {d.max():>10.3e} {d.mean():>10.3e} "
              f"{rel:>10.3e} {tol:>9.0e}  {'OK' if ok else 'FAIL'}")

    extra = {p.name for p in args.dir_test.glob('*.npy')} - {p.name for p in refs}
    if extra:
        print(f"note: files only in {args.dir_test}: {sorted(extra)}")
    print("OK" if fail == 0 else "FAIL")
    return fail


if __name__ == "__main__":
    sys.exit(main())
