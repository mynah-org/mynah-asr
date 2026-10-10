#!/usr/bin/env python3
"""Incremental, size-verified archival of fine-tuning runs to a PRIVATE Hugging Face repo,
straight from the GPU box (the dev machine's line is usually the bottleneck).

    $PY hf_archive.py --repo <user>/<private-repo> --ft-root /root/ft --kit <path>/finetune [--loop]

Generalised from the 2026-10-09 box script (paths are arguments now). Token:
--token-file (default ~/.hf_token, mode 600), never printed; delete it when the
last pass is verified. Uploads, once: README (--readme), the kit (the finetune/
directory as the recipe), the tokenizers, the probe results, --extra files; per
finished run (metrics.json present): run.json, metrics, evals, logs, last.ckpt
where present, and final.nemo ONLY for runs whose tag matches --nemo-tags
(default: long-schedule "-e" runs, EOU and plain runs; smoke runs keep metrics
only, a rerun costs about one GPU-minute). Every upload is verified by size on
the Hub before a marker is written to --done-dir; re-runs skip marked items.
With --loop: repeats every --interval s until <done-dir>/../STOP exists.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="private model repo id (created if missing)")
    ap.add_argument("--ft-root", required=True, help="FT_ROOT: runs/, tokenizers/, probe/, logs/, models/tok-*")
    ap.add_argument("--kit", default=str(Path(__file__).resolve().parent.parent), help="the finetune/ directory")
    ap.add_argument("--token-file", default=str(Path.home() / ".hf_token"))
    ap.add_argument("--done-dir", default="/root/arch/done")
    ap.add_argument("--readme", default=None, help="README.md for the archive repo")
    ap.add_argument("--extra", action="append", default=[], help="local=remote file pairs (e.g. a survival tarball)")
    ap.add_argument("--nemo-tags", default=r"-e|^eou|^plain", help="regex of run tags whose final.nemo is uploaded")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    a = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=Path(a.token_file).read_text().strip())
    api.create_repo(a.repo, private=True, exist_ok=True, repo_type="model")
    D = Path(a.done_dir)
    D.mkdir(parents=True, exist_ok=True)
    ft = Path(a.ft_root)
    nemo_re = re.compile(a.nemo_tags)

    def verify(local, remote):
        info = api.get_paths_info(a.repo, [remote])
        if not info:
            return False
        size = getattr(info[0], "size", None) or getattr(getattr(info[0], "lfs", None), "size", None)
        return size == local.stat().st_size

    def put(local, remote, key):
        local = Path(local)
        if (D / key).exists() or not local.exists():
            return
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=a.repo, commit_message=f"Add {remote}")
        if verify(local, remote):
            (D / key).write_text(time.strftime("%F %T\n"))
            print(f"uploaded+verified {remote} ({local.stat().st_size / 1e6:.1f} MB)", flush=True)
        else:
            print(f"VERIFY FAILED {remote}", flush=True)

    def put_dir(local, remote, key, skip=(".nemo", ".wav", ".parquet", ".ckpt", ".pyc")):
        local = Path(local)
        if (D / key).exists() or not local.exists():
            return
        files = [p for p in local.rglob("*") if p.is_file() and not p.name.endswith(skip) and "__pycache__" not in p.parts]
        api.upload_folder(folder_path=str(local), path_in_repo=remote, repo_id=a.repo,
                          ignore_patterns=[f"*{s}" for s in skip] + ["**/__pycache__/**"], commit_message=f"Add {remote}/")
        bad = [p for p in files if not verify(p, f"{remote}/{p.relative_to(local)}")]
        if bad:
            print(f"VERIFY FAILED {remote}: {len(bad)} file(s)", flush=True)
        else:
            (D / key).write_text(time.strftime("%F %T\n"))
            print(f"uploaded+verified {remote}/ ({len(files)} files)", flush=True)

    def once():
        if a.readme:
            put(a.readme, "README.md", "readme")
        put_dir(a.kit, "recipe/finetune", "kit")
        put_dir(ft / "tokenizers", "tokenizers", "tokenizers")
        put_dir(ft / "probe", "results/probe", "probe")
        for pair in a.extra:
            local, remote = pair.split("=", 1)
            put(local, remote, "extra-" + remote.replace("/", "_"))
        for run in sorted((ft / "runs").glob("*/")):
            if not (run / "metrics.json").exists():
                continue
            # keys carry the file mtime: a run that rewrites its best checkpoint or its
            # metrics after an earlier pass is uploaded again (2026-10-09 box fix)
            mt = lambda f: int((run / f).stat().st_mtime) if (run / f).exists() else 0  # noqa: E731
            put_dir(run, f"runs/{run.name}", f"run-{run.name}-{mt('metrics.json')}")
            if nemo_re.search(run.name):
                put(run / "final.nemo", f"runs/{run.name}/final.nemo", f"nemo-{run.name}-{mt('final.nemo')}")
            # resume state where a run saved one (Lightning last.ckpt or the plain loop's torch state)
            put(run / "last.ckpt", f"runs/{run.name}/last.ckpt", f"ckpt-{run.name}-{mt('last.ckpt')}")
        for tok in sorted((ft / "models").glob("tok-*/")):
            put_dir(tok, f"tokenizers/{tok.name}", f"tok-{tok.name}")
        put_dir(ft / "logs", "logs", f"logs-{int(time.time() // 3600)}")

    once()
    while a.loop and not (D.parent / "STOP").exists():
        time.sleep(a.interval)
        once()
    print("archival pass done", time.strftime("%T"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
