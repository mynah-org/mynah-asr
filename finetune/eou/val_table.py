#!/usr/bin/env python3
"""Markdown tables of per-domain val results from two_stage.py runs and --eval-only baselines.

    python3 val_table.py --col P40=logs/base-p40.log --col A0@3000=runs/plain-it-a0-mix/metrics.json:3000 \
        --col A0@4000=runs/plain-it-a0-mix/metrics.json:4000 --delta A0@3000-P40 \
        [--trajectory runs/plain-it-a0-mix/metrics.json]

A column is a two_stage log (its first VAL line) or metrics.json:<step>. Cells are
WER / CER / empty per domain, then the macro WER / CER. No numpy/NeMo needed.
"""
import argparse
import json


def load(spec):
    if spec.endswith(".log"):
        line = next(l for l in open(spec) if l.startswith("  VAL"))
        return json.loads(line.split("VAL ", 1)[1])
    path, step = spec.rsplit(":", 1)
    return next(r for r in json.load(open(path))["log"] if r["step"] == int(step))


def domains(r):
    return [k[4:-4] for k in r if k.startswith("val_") and k.endswith("_wer") and k != "val_wer"]


def cell(r, d):
    return f"{r[f'val_{d}_wer']:.2f} / {r[f'val_{d}_cer']:.2f} / {r[f'val_{d}_empty']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--col", action="append", default=[], help="NAME=source")
    ap.add_argument("--delta", action="append", default=[], help="A-B (column names)")
    ap.add_argument("--trajectory", default="")
    a = ap.parse_args()
    cols = {n: load(s) for n, s in (c.split("=", 1) for c in a.col)}
    ds = domains(next(iter(cols.values())))
    deltas = [d.split("-", 1) for d in a.delta]
    print("| domain | " + " | ".join(cols) + "".join(f" | Δ {x}−{y}" for x, y in deltas) + " |")
    print("|---" * (1 + len(cols) + len(deltas)) + "|")
    for d in ds:
        row = [cell(r, d) for r in cols.values()]
        for x, y in deltas:
            X, Y = cols[x], cols[y]
            row.append(f"{X[f'val_{d}_wer'] - Y[f'val_{d}_wer']:+.2f} / {X[f'val_{d}_cer'] - Y[f'val_{d}_cer']:+.2f} / "
                       f"{X[f'val_{d}_empty'] - Y[f'val_{d}_empty']:+d}")
        print(f"| {d} | " + " | ".join(row) + " |")
    row = [f"{r['val_wer']:.2f} / {r['val_cer']:.2f}" for r in cols.values()]
    for x, y in deltas:
        row.append(f"{cols[x]['val_wer'] - cols[y]['val_wer']:+.2f} / {cols[x]['val_cer'] - cols[y]['val_cer']:+.2f}")
    print("| macro | " + " | ".join(row) + " |")
    if a.trajectory:
        log = [r for r in json.load(open(a.trajectory))["log"] if "val_wer" in r]
        print("\n| step | " + " | ".join(ds) + " | macro WER / CER |")
        print("|---" * (2 + len(ds)) + "|")
        for r in log:
            print(f"| {r['step']}{' *' if r.get('saved') else ''} | " + " | ".join(cell(r, d) for d in ds)
                  + f" | {r['val_wer']:.2f} / {r['val_cer']:.2f} |")
        print("(* = checkpoint saved: best macro so far)")


if __name__ == "__main__":
    main()
