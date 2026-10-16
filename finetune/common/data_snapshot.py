#!/usr/bin/env python3
"""Snapshot of a prepared training/eval data set (audio + manifests) to a PRIVATE Hugging Face repo, and its
restore on an empty box: a new instance downloads a few large shards in parallel instead of rebuilding ~500 h of
audio from the public sources (download, parquet decode, resample, cut) one data set at a time.

    pack:    $PY data_snapshot.py pack --repo <user>/<repo> --name it-c250 --token-file /root/.hf_token \
                 --manifests a.json,b.json,... [--meta 'glob,glob'] [--shard-gb 1.5] [--workers 16]
    restore: $PY data_snapshot.py restore --repo <user>/<repo> --name it-c250 --token-file /root/.hf_token [--workers 16]

Only the audio referenced by --manifests is packed (e.g. the agree<=15 subsets, not the whole teacher pools).
Member names are the manifests' LOGICAL absolute paths (symlinks such as /dev/shm/mix -> /root/disk/mix are not
resolved), so a restore writes to the same paths and every manifest / archived teacher score stays valid; where
the audio lands (tmpfs or root disk) is decided on the restoring box by those symlinks. 16-bit PCM WAV is stored
as FLAC (lossless) and decoded back to 16-bit PCM WAV: the samples are bit-identical, the header bytes may differ.
Other files (FLAC, ...) are stored as they are. --meta files (manifests, teacher scores, subsets) go to meta.tgz.
Shards are built, uploaded and deleted one at a time per worker, so the local disk only holds a few in flight.
index.json (written last) lists every shard with its file count, size and sha256; restore verifies all three.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

FLAC_SUFFIX = ".pcm16.flac"   # marks a WAV stored as FLAC


def audio_paths(manifests):
    seen = {}
    for m in manifests:
        for line in open(m):
            if line.strip():
                p = json.loads(line)["audio_filepath"]
                seen.setdefault(p, None)
    return list(seen)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build_shard(args):
    """Write one tar shard (WAV PCM_16 -> FLAC members), return its record."""
    out, files = args
    import soundfile as sf
    n_flac = 0
    with tarfile.open(out, "w") as tar:
        for p in files:
            name = p.lstrip("/")
            data = None
            if p.endswith(".wav") and sf.info(p).subtype == "PCM_16":
                x, sr = sf.read(p, dtype="int16", always_2d=False)
                buf = io.BytesIO()
                sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
                data, name = buf.getvalue(), name + FLAC_SUFFIX
                n_flac += 1
            if data is None:
                tar.add(p, arcname=name, recursive=False)
            else:
                ti = tarfile.TarInfo(name)
                ti.size, ti.mtime = len(data), int(time.time())
                tar.addfile(ti, io.BytesIO(data))
    return {"name": os.path.basename(out), "files": len(files), "flac_from_wav": n_flac,
            "bytes": os.path.getsize(out), "sha256": sha256(out)}


def plan_shards(paths, shard_bytes):
    """Group the files (in manifest order, sorted by path for locality) into ~shard_bytes groups of source size."""
    groups, cur, size = [], [], 0
    for p in sorted(paths):
        s = os.path.getsize(p)
        if cur and size + s > shard_bytes:
            groups.append(cur); cur, size = [], 0
        cur.append(p); size += s
    if cur:
        groups.append(cur)
    return groups


def pack(a, api):
    manifests = [m for m in a.manifests.split(",") if m]
    paths = audio_paths(manifests)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"== {len(missing)} referenced audio files missing, e.g. {missing[:3]}")
    src_bytes = sum(os.path.getsize(p) for p in paths)
    groups = plan_shards(paths, int(a.shard_gb * 2**30))
    print(f"== pack {a.name}: {len(paths)} files, {src_bytes / 2**30:.1f} GiB source, {len(groups)} shards", flush=True)
    base = f"data/snapshots/{a.name}"
    tmp = Path(tempfile.mkdtemp(prefix="snap-", dir=a.tmp))
    meta_files = sorted({f for g in a.meta.split(",") if g for f in glob.glob(g)} | set(manifests))
    meta = tmp / "meta.tgz"
    with tarfile.open(meta, "w:gz") as tar:
        for f in meta_files:
            tar.add(f, arcname=f.lstrip("/"), recursive=False)
    api.upload_file(path_or_fileobj=str(meta), path_in_repo=f"{base}/meta.tgz", repo_id=a.repo,
                    commit_message=f"data snapshot {a.name}: meta")
    records, t0 = [], time.time()
    # one process per shard build; uploads happen in this process as builds finish (bounded in-flight shards)
    with ProcessPoolExecutor(a.workers) as ex:
        jobs = {}
        it = iter(enumerate(groups))
        def submit_next():
            try:
                i, g = next(it)
            except StopIteration:
                return
            jobs[ex.submit(build_shard, (str(tmp / f"shard-{i:04d}.tar"), g))] = i
        for _ in range(a.workers):
            submit_next()
        while jobs:
            done = next(as_completed(jobs))
            jobs.pop(done)
            r = done.result()
            p = tmp / r["name"]
            for k in range(5):
                try:
                    api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{base}/{r['name']}", repo_id=a.repo,
                                    commit_message=f"data snapshot {a.name}: {r['name']}")
                    break
                except Exception as e:  # transient Hub errors: retry, then give up loudly
                    print(f"== upload retry {r['name']} ({k + 1}): {e}", flush=True); time.sleep(30)
            else:
                sys.exit(f"== upload FAILED {r['name']}")
            p.unlink()
            records.append(r)
            print(f"== shard {r['name']} {r['files']} files {r['bytes'] / 2**30:.2f} GiB "
                  f"({len(records)}/{len(groups)}, {time.time() - t0:.0f} s)", flush=True)
            submit_next()
    records.sort(key=lambda r: r["name"])
    index = {"name": a.name, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "manifests": manifests,
             "meta_files": meta_files, "files": len(paths), "source_bytes": src_bytes,
             "stored_bytes": sum(r["bytes"] for r in records), "shards": records}
    (tmp / "index.json").write_text(json.dumps(index, indent=1))
    api.upload_file(path_or_fileobj=str(tmp / "index.json"), path_in_repo=f"{base}/index.json", repo_id=a.repo,
                    commit_message=f"data snapshot {a.name}: index")
    meta.unlink(); (tmp / "index.json").unlink(); tmp.rmdir()
    print(f"== PACK-DONE {a.name}: {len(records)} shards, {index['stored_bytes'] / 2**30:.1f} GiB stored "
          f"(source {src_bytes / 2**30:.1f} GiB), {time.time() - t0:.0f} s", flush=True)


def extract_shard(args):
    path, root = args
    import soundfile as sf
    n = 0
    with tarfile.open(path) as tar:
        for m in tar:
            if not m.isfile():
                continue
            name = m.name
            data = tar.extractfile(m).read()
            if name.endswith(FLAC_SUFFIX):
                x, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=False)
                dst = Path(root, name[: -len(FLAC_SUFFIX)])
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(dst), x, sr, format="WAV", subtype="PCM_16")
            else:
                dst = Path(root, name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
            n += 1
    os.unlink(path)
    return n


def restore(a, api):
    from huggingface_hub import hf_hub_download
    base = f"data/snapshots/{a.name}"
    tmp = Path(tempfile.mkdtemp(prefix="snap-", dir=a.tmp))
    kw = dict(repo_id=a.repo, token=api.token, local_dir=str(tmp))
    index = json.loads(Path(hf_hub_download(filename=f"{base}/index.json", **kw)).read_text())
    meta = hf_hub_download(filename=f"{base}/meta.tgz", **kw)
    with tarfile.open(meta) as tar:
        tar.extractall(a.root)
    print(f"== restore {a.name}: {index['files']} files in {len(index['shards'])} shards, "
          f"{index['stored_bytes'] / 2**30:.1f} GiB; meta {len(index['meta_files'])} files", flush=True)
    t0, n = time.time(), 0

    def fetch(r):
        p = hf_hub_download(filename=f"{base}/{r['name']}", **kw)
        if os.path.getsize(p) != r["bytes"] or sha256(p) != r["sha256"]:
            raise RuntimeError(f"shard {r['name']} size/sha256 mismatch")
        return p, r

    with ThreadPoolExecutor(a.workers) as dl, ProcessPoolExecutor(a.workers) as ex:
        futs = [dl.submit(fetch, r) for r in index["shards"]]
        ext = []
        for f in as_completed(futs):
            p, r = f.result()
            ext.append((ex.submit(extract_shard, (p, a.root)), r))
        for f, r in ext:
            got = f.result()
            if got != r["files"]:
                sys.exit(f"== shard {r['name']}: {got} files, expected {r['files']}")
            n += got
    missing = [p for p in audio_paths([str(Path(a.root, m.lstrip("/"))) for m in index["manifests"]])
               if not os.path.exists(Path(a.root, p.lstrip("/")))]
    if missing or n != index["files"]:
        sys.exit(f"== RESTORE GATE FAILED: {n}/{index['files']} files, {len(missing)} referenced paths missing")
    print(f"== RESTORE-DONE {a.name}: {n} files, gate ok, {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pack", "restore"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--name", required=True, help="snapshot name: data/snapshots/<name>/ in the repo")
    ap.add_argument("--token-file", default=os.path.expanduser("~/.hf_token"))
    ap.add_argument("--manifests", default="", help="pack: comma list of JSONL manifests whose audio is packed")
    ap.add_argument("--meta", default="", help="pack: comma list of globs of small files (teacher scores, subsets)")
    ap.add_argument("--shard-gb", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tmp", default="/root", help="where shards are staged (root disk, not tmpfs)")
    ap.add_argument("--root", default="/", help="restore: prefix for the stored absolute paths (tests)")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=Path(a.token_file).read_text().strip())
    pack(a, api) if a.cmd == "pack" else restore(a, api)


if __name__ == "__main__":
    main()
