#!/usr/bin/env python3
"""Compare two directories of .npy dumps with per-stage tolerances (M0.4).

Casi d'uso: diffare i golden dell'oracolo tra versioni del converter/oracolo
(rigenera in una dir nuova e confronta con la vecchia), o qualunque coppia di
dump omogenei (stessi nomi file). La metrica segue i test C: errore assoluto
massimo SCALATO per max|ref| dello stadio (err_rel = max|a-b| / max|ref|).

Uso: uv run python -m eval.compare <dir_ref> <dir_test> [--tol nome=1e-4 ...]
Exit: 0 tutti gli stadi entro tolleranza, 1 altrimenti, 77 dir vuote/assenti.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import numpy as np

# Default per-stage tolerances (fnmatch pattern on the file name without .npy),
# allineate ai test C: mel bit-esatto, stadi profondi via accumulo f32.
DEFAULT_TOLS: list[tuple[str, float]] = [
    ("mel", 0.0),
    ("subsampling", 1e-4),
    ("layer_*", 3.5e-2),      # come tests/test_encoder.c (accumulo su 24 layer)
    ("encoder_out", 3.5e-2),
    ("enc_proj", 3.5e-2),
    ("*", 1e-3),              # default per stadi non mappati
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
                    help="override per stadio: nome=tolleranza (ripetibile)")
    args = ap.parse_args()

    tols = list(DEFAULT_TOLS)
    for spec in args.tol:
        name, _, val = spec.partition("=")
        tols.insert(0, (name, float(val)))

    refs = sorted(args.dir_ref.glob("*.npy")) if args.dir_ref.is_dir() else []
    if not refs:
        print(f"SKIP: nessun .npy in {args.dir_ref}")
        return 77

    fail = 0
    print(f"{'stadio':16} {'shape':>16} {'max|d|':>10} {'mean|d|':>10} "
          f"{'err_rel':>10} {'tol':>9}  esito")
    for ref_path in refs:
        name = ref_path.stem
        test_path = args.dir_test / ref_path.name
        if not test_path.exists():
            print(f"{name:16} {'-':>16} {'-':>10} {'-':>10} {'-':>10} {'-':>9}  MANCANTE")
            fail = 1
            continue
        a = np.load(ref_path).astype(np.float64)
        b = np.load(test_path).astype(np.float64)
        if a.shape != b.shape:
            print(f"{name:16} {str(a.shape):>16} vs {b.shape}  SHAPE DIVERSA")
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
        print(f"nota: file solo in {args.dir_test}: {sorted(extra)}")
    print("OK" if fail == 0 else "FAIL")
    return fail


if __name__ == "__main__":
    sys.exit(main())
