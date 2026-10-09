"""Run metadata (`run.json`) and checkpoint helpers shared by every training entry point.

Every trainer under finetune/ writes `<run>/run.json` twice: once before the
first step (status "started", the fully resolved configuration) and once at the
end (status "done", plus outputs). The schema is flat on purpose so a later
reader can diff two runs with `jq` or `diff`. Pure stdlib; torch/NeMo versions
are read lazily and recorded as `null` when the package is missing.

Fields (all present, `null` when unknown):
  schema, status, entry_point, argv, created_utc, updated_utc,
  base_model {id, revision, path, sha256}, git {sha, dirty},
  tokenizer {path, sha256, note}, data {dataset, subset, manifest, hours,
  utterances, ...}, seed, optimizer {name, betas, weight_decay, ...},
  lr_groups {group: lr}, scheduler {...}, fastemit_lambda, augmentation {...},
  eou_plain_mix {...}, freeze_policy, max_steps, epochs, hardware {gpu, ...},
  versions {python, torch, cuda, nemo, lightning, numpy, numba, numba_cuda},
  config (the resolved argparse namespace), outputs, notes.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "mynah-asr/finetune-run/1"
_FIELDS = ("base_model", "tokenizer", "data", "seed", "optimizer", "lr_groups", "scheduler", "fastemit_lambda",
           "augmentation", "eou_plain_mix", "freeze_policy", "max_steps", "epochs", "hardware", "versions",
           "config", "outputs", "notes")


def sha256_path(path, chunk=1 << 20):
    """sha256 of a file, or of a directory (sha256 over `relpath\\0filehash\\n`
    lines of every file, sorted). None if the path does not exist."""
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    if p.is_dir():
        h = hashlib.sha256()
        for q in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(f"{q.relative_to(p).as_posix()}\0{sha256_path(q)}\n".encode())
        return "dir:" + h.hexdigest()
    return None


def git_info(start=None):
    """SHA and dirty flag of the repository holding `start` (default: this file)."""
    d = str(Path(start or __file__).resolve().parent)
    try:
        sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "-C", d, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=10).stdout.strip()) if sha else None
    except (OSError, subprocess.SubprocessError):
        sha, dirty = None, None
    # a kit copied to a box without .git: the caller may pin it explicitly
    return {"sha": sha or os.environ.get("FINETUNE_GIT_SHA"), "dirty": dirty}


def versions():
    out = {"python": sys.version.split()[0]}
    for mod, key in (("torch", "torch"), ("nemo", "nemo"), ("lightning", "lightning"), ("numpy", "numpy"),
                     ("numba", "numba"), ("numba_cuda", "numba_cuda"), ("sentencepiece", "sentencepiece")):
        try:
            m = __import__(mod)
            out[key] = getattr(m, "__version__", "unknown")
        except Exception:
            out[key] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception:
        out["cuda"] = None
    return out


def hardware():
    hw = {"host_cpus": os.cpu_count(), "gpu": None, "gpu_count": 0, "gpu_mem_gb": None, "driver": None}
    try:
        import torch

        if torch.cuda.is_available():
            hw["gpu_count"] = torch.cuda.device_count()
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
            hw["bf16"] = torch.cuda.is_bf16_supported()
    except Exception:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        hw["driver"] = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return hw


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def build(entry_point, config, probe_env=True, **fields):
    """A run.json dict. `config` = vars(argparse namespace); `fields` = any of _FIELDS.
    probe_env=False skips torch/GPU probing (used by --dry-run and the tests)."""
    unknown = set(fields) - set(_FIELDS)
    assert not unknown, f"unknown run.json fields {sorted(unknown)}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    r = {"schema": SCHEMA, "status": "resolved", "entry_point": entry_point, "argv": list(sys.argv),
         "created_utc": now, "updated_utc": now, "git": git_info()}
    for k in _FIELDS:
        r[k] = None
    r["versions"] = versions() if probe_env else None
    r["hardware"] = hardware() if probe_env else None
    r["config"] = config
    r.update(fields)
    return _jsonable(r)


def write(path, run, status=None, **updates):
    """Atomic write of run.json; `status`/`updates` are merged in first."""
    if status:
        run["status"] = status
    for k, v in updates.items():
        run[k] = _jsonable(v)
    run["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(run, ensure_ascii=False, indent=1, sort_keys=False), encoding="utf-8")
    os.replace(tmp, p)
    return run


def print_resolved(run, keys=("entry_point", "base_model", "tokenizer", "data", "seed", "optimizer", "lr_groups",
                              "scheduler", "fastemit_lambda", "augmentation", "eou_plain_mix", "freeze_policy",
                              "max_steps", "epochs", "hardware", "versions", "git")):
    print("== resolved config", flush=True)
    for k in keys:
        print(f"   {k}: {json.dumps(run.get(k), ensure_ascii=False)}", flush=True)


def require_root(env="FT_ROOT"):
    """Box entry points refuse to run on an implicit default data/output root."""
    v = os.environ.get(env)
    if not v:
        raise SystemExit(f"{env} is not set: export {env}=<data+runs root> (e.g. /root/ft) explicitly")
    return Path(v)


# --------------------------------------------------------------------------- checkpoints
def ckpt_record(path, step, kind, resume):
    """What run.json/metrics.json say about a saved checkpoint."""
    p = Path(path)
    return {"path": str(p), "kind": kind, "global_step": step,
            "size_gb": round(p.stat().st_size / 2**30, 3) if p.exists() else None, "resume": resume}


LIGHTNING_RESUME = ("same model + data: trainer.fit(model, ckpt_path=<last.ckpt>) restores optimizer, scheduler, "
                    "AMP scaler and global step; the run continues only if --max-steps is larger than the saved step. "
                    "A NEW stage only needs --init <final.nemo>.")
TORCH_RESUME = ("plain torch state {model, opt, sched, step, args}: plain_asr.py --resume <file> --max-steps <larger>")
