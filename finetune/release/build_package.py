#!/usr/bin/env python3
"""Assemble a publication-ready release directory for one EOU-120M language specialist (staged privately).

    build_package.py --lang it --name mynah-eou-it-120m --nemo final.nemo --pack <mynah pack dir> \
        --card finetune/release/it/README.md --provenance prov.json --results name=path,... --out <dir>

Layout of <out> (= the root of a future Hub model repo):
    README.md                 model card (copied from --card)
    LICENSE, NOTICE           NVIDIA Open Model License + attribution (--license / --notice files)
    model.nemo                the NeMo checkpoint (byte copy; sha256 must equal --nemo-sha if given)
    nemo/model_config.yaml    extracted from model.nemo (read-only reference)
    nemo/tokenizer.*          the SentencePiece files packed in model.nemo
    mynah/                    the Mynah pack: model.safetensors, mynah.json, tokens.json, mel_filters.safetensors
    inference.json            the frozen runtime / protocol (precision, chunking, reset semantics, normalisation)
    provenance.json           commits, parent / checkpoint hashes, recipe, data (from --provenance)
    results/                  the frozen result files (FINAL, DEV, Mynah parity, serving), copied as given
    examples/                 copied from --examples (NeMo + Mynah usage)
    SHA256SUMS                sha256 of every file above (relative paths)
Nothing is downloaded and no secret is read; the token never enters this tool.
"""
import argparse
import hashlib
import json
import os
import shutil
import tarfile

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--lang", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--nemo", required=True)
ap.add_argument("--nemo-sha", default="", help="expected sha256 prefix of --nemo")
ap.add_argument("--pack", required=True, help="Mynah pack dir (export_to_mynah.sh output)")
ap.add_argument("--pack-sha", default="", help="name=sha256prefix,... expected for the pack files")
ap.add_argument("--card", required=True)
ap.add_argument("--license", required=True)
ap.add_argument("--notice", required=True)
ap.add_argument("--inference", required=True, help="inference.json")
ap.add_argument("--provenance", required=True)
ap.add_argument("--results", default="", help="name=path,... copied to results/<name>")
ap.add_argument("--examples", default="", help="dir copied to examples/")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if os.path.exists(a.out) and os.listdir(a.out):
    raise SystemExit(f"== {a.out} exists and is not empty: stop (never overwrite a staged package)")
os.makedirs(a.out, exist_ok=True)
s = sha(a.nemo)
if a.nemo_sha and not s.startswith(a.nemo_sha):
    raise SystemExit(f"== {a.nemo}: sha256 {s[:16]} != expected {a.nemo_sha}: stop")
pack_files = ("model.safetensors", "mynah.json", "tokens.json", "mel_filters.safetensors")
exp = dict(x.split("=", 1) for x in a.pack_sha.split(",") if x)
for f in pack_files:
    p = os.path.join(a.pack, f)
    if not os.path.isfile(p):
        raise SystemExit(f"== pack file missing: {p}")
    if f in exp and not sha(p).startswith(exp[f]):
        raise SystemExit(f"== pack {f}: sha256 {sha(p)[:16]} != expected {exp[f]}: stop")
shutil.copyfile(a.nemo, f"{a.out}/model.nemo")
os.makedirs(f"{a.out}/nemo", exist_ok=True)
with tarfile.open(a.nemo) as t:
    for m in t.getmembers():
        b = os.path.basename(m.name)
        if m.isfile() and (b == "model_config.yaml" or "tokenizer" in b or b.endswith((".model", ".vocab")) or b == "vocab.txt"):
            with t.extractfile(m) as src, open(f"{a.out}/nemo/{b}", "wb") as dst:
                shutil.copyfileobj(src, dst)
os.makedirs(f"{a.out}/mynah", exist_ok=True)
for f in pack_files:
    shutil.copyfile(os.path.join(a.pack, f), f"{a.out}/mynah/{f}")
for src, dst in ((a.card, "README.md"), (a.license, "LICENSE"), (a.notice, "NOTICE"), (a.inference, "inference.json"),
                 (a.provenance, "provenance.json")):
    shutil.copyfile(src, f"{a.out}/{dst}")
for spec in filter(None, a.results.split(",")):
    n, p = spec.split("=", 1)
    os.makedirs(f"{a.out}/results", exist_ok=True)
    shutil.copyfile(p, f"{a.out}/results/{n}")
if a.examples:
    shutil.copytree(a.examples, f"{a.out}/examples")
for j in ("inference.json", "provenance.json"):
    json.load(open(f"{a.out}/{j}"))   # must parse
rows = []
for d, _, fs in os.walk(a.out):
    for f in sorted(fs):
        p = os.path.join(d, f)
        r = os.path.relpath(p, a.out)
        if r != "SHA256SUMS":
            rows.append(f"{sha(p)}  {r}")
open(f"{a.out}/SHA256SUMS", "w").write("\n".join(sorted(rows, key=lambda x: x.split("  ", 1)[1])) + "\n")
print(f"== package {a.name} ({a.lang}): {len(rows)} files, model.nemo {s[:16]}, "
      f"{sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(a.out) for f in fs) / 1e6:.1f} MB -> {a.out}")
