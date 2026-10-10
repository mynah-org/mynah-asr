#!/usr/bin/env python3
"""Keep the private HF archive light: delete superseded weights, then squash the history.

    $PY hf_prune.py --repo <user>/<private-repo> --delete 'runs/eou-it-5h-*/*.nemo' --delete 'runs/plain-it-a0-mix/last.ckpt' \
        [--squash] [--dry-run] [--token-file /root/.hf_token]

On the Hub a deleted file still occupies storage through the commit history (every re-upload of
a best checkpoint is another LFS object). --squash runs super_squash_history after the delete:
the repo keeps ONLY its current files (irreversible for the history, the current files stay).
Only weights should be pruned; metrics, logs and evals are tiny and are the record. Prints the
size of the current files before and after, and every matched path (fnmatch globs).
"""
import argparse
import fnmatch
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--delete", action="append", default=[])
    ap.add_argument("--squash", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--token-file", default=str(Path.home() / ".hf_token"))
    a = ap.parse_args()
    from huggingface_hub import CommitOperationDelete, HfApi
    api = HfApi(token=Path(a.token_file).read_text().strip())

    def files():
        return {f.path: f.size for f in api.list_repo_tree(a.repo, recursive=True) if getattr(f, "size", None) is not None}

    cur = files()
    hit = sorted(p for p in cur if any(fnmatch.fnmatch(p, g) for g in a.delete))
    print(f"== {a.repo}: {len(cur)} files {sum(cur.values()) / 1e9:.2f} GB; to delete {len(hit)} files {sum(cur[p] for p in hit) / 1e9:.2f} GB")
    for p in hit:
        print(f"   - {p} {cur[p] / 1e9:.2f} GB")
    if a.dry_run:
        return
    if hit:
        api.create_commit(a.repo, operations=[CommitOperationDelete(path_in_repo=p) for p in hit],
                          commit_message=f"Prune {len(hit)} superseded weight files")
    if a.squash:
        api.super_squash_history(a.repo, commit_message="Squash history: keep only the current files")
        print("== history squashed")
    after = files()
    print(f"== after: {len(after)} files {sum(after.values()) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
