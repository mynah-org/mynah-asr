"""Incremental archival of the FT experiments to a PRIVATE Hugging Face repo,
straight from the box (the dev machine's line is the bottleneck).

    /root/nemo-venv/bin/python arch.py <repo_id> [--loop]

Token: /root/.hf_token (mode 600), never printed, deleted by the caller at the
end. Uploads, once: the kit (recipe/config), the Italian tokenizer, the
survival bundle; per finished run (metrics.json present): metrics, evals,
logs, and final.nemo ONLY for long-schedule runs (tag contains "-e"); smoke
runs keep metrics only (a rerun costs ~1 GPU-minute). Every upload is
verified by size on the Hub before it is marked done in /root/arch/done/.
With --loop: repeats every 120 s until /root/arch/STOP exists.
"""
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

repo = sys.argv[1]
loop = "--loop" in sys.argv
api = HfApi(token=Path("/root/.hf_token").read_text().strip())
api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
D = Path("/root/arch/done"); D.mkdir(parents=True, exist_ok=True)


def verify(local, remote):
    info = api.get_paths_info(repo, [remote])
    if not info:
        return False
    size = getattr(info[0], "size", None) or getattr(getattr(info[0], "lfs", None), "size", None)
    return size == local.stat().st_size


def put(local, remote, key):
    local = Path(local)
    if (D / key).exists() or not local.exists():
        return
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo,
                    commit_message=f"Add {remote}")
    if verify(local, remote):
        (D / key).write_text(time.strftime("%F %T\n"))
        print(f"uploaded+verified {remote} ({local.stat().st_size / 1e6:.1f} MB)", flush=True)
    else:
        print(f"VERIFY FAILED {remote}", flush=True)


def put_dir(local, remote, key, skip=(".nemo", ".wav", ".parquet")):
    local = Path(local)
    if (D / key).exists() or not local.exists():
        return
    files = [p for p in local.rglob("*") if p.is_file() and not p.name.endswith(skip)]
    api.upload_folder(folder_path=str(local), path_in_repo=remote, repo_id=repo,
                      ignore_patterns=[f"*{s}" for s in skip], commit_message=f"Add {remote}/")
    bad = [p for p in files if not verify(p, f"{remote}/{p.relative_to(local)}")]
    if bad:
        print(f"VERIFY FAILED {remote}: {len(bad)} file(s)", flush=True)
    else:
        (D / key).write_text(time.strftime("%F %T\n"))
        print(f"uploaded+verified {remote}/ ({len(files)} files)", flush=True)


def once():
    put("/root/arch/README.md", "README.md", "readme")
    put_dir("/root/ft-kit", "recipe/ft-kit", "kit")
    put_dir("/root/ft/tokenizers", "tokenizers", "tokenizers")
    put_dir("/root/ft/probe", "results/probe", "probe")
    put("/root/survival.tgz", "results/survival-1427.tgz", "survival")
    for run in sorted(Path("/root/ft/runs").glob("*/")):
        if not (run / "metrics.json").exists():
            continue
        # keys carry the file mtime: a run that rewrites its best checkpoint or its
        # metrics while training (plain_it.py) gets the new version re-uploaded
        mt = lambda f: int((run / f).stat().st_mtime) if (run / f).exists() else 0
        put_dir(run, f"runs/{run.name}", f"run-{run.name}-{mt('metrics.json')}")
        if "-e" in run.name or run.name.startswith(("eou", "plain-")):
            put(run / "final.nemo", f"runs/{run.name}/final.nemo", f"nemo-{run.name}-{mt('final.nemo')}")
        # full training state (optimizer, scheduler, step) where a run saved one
        put(run / "last.ckpt", f"runs/{run.name}/last.ckpt", f"ckpt-{run.name}-{mt('last.ckpt')}")
    put_dir("/root/eou-kit", "recipe/eou-kit", "eou-kit")
    put_dir("/root/ft/models/tok-eou-it", "tokenizers/eou-it", "tok-eou-it")
    put_dir("/root/ft/logs", "logs", f"logs-{int(time.time() // 3600)}")


once()
while loop and not Path("/root/arch/STOP").exists():
    time.sleep(120)
    once()
print("archival pass done", time.strftime("%T"), flush=True)
