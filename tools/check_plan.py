#!/usr/bin/env python3
"""check_plan.py — PLAN.md must never contain dangling references (ENGINEERING.md, Plan integrity).

    python3 tools/check_plan.py [--plan PLAN.md] [--work .work]

Exit 1 when any of these holds:
  * a `.work/...` path referenced by the plan does not exist;
  * a repository path referenced by the plan (docs/, tools/, tests/, scripts/, server/, src/, cli/, or a
    top-level *.md) does not exist;
  * two tasks carry the same id (P0.1, P3.3a, ...);
  * a task marked `[x]` references a detail/evidence file that is missing;
  * an addendum under .work/ names a task id (`Task: P1.4`) that PLAN.md does not contain.
A missing PLAN.md is not a failure (the plan is local and untracked): the check reports
it and exits 0.  Nothing here reads the content of the plan beyond paths and task ids.

WHAT "EXISTS" MEANS.  A path exists when GIT has it, not when the working tree does.
The two differ precisely where it hurts: `tests/test_stream_batch` is a gitignored
BUILT binary, present on the machine that wrote the plan line and absent from every
fresh clone, so a filesystem check passed locally and failed in CI on the same commit
(2026-09-19..2026-09-21, every `Build & Test` job).  A gate that is green only on the
author's disk is worse than no gate: it authorises "PASS" in a commit message.
`.work/evidence/` and `.work/private/` are untracked BY DESIGN (.work/README.md rule 7);
they are exempt and counted, never silently skipped.  Without git (a tarball) the check
degrades to the filesystem and says so.
"""
import argparse, os, re, subprocess, sys

UNTRACKED_BY_DESIGN = (".work/evidence/", ".work/private/")


def git_truth(root):
    """The set of paths a fresh clone would have: tracked files + their directories.

    Returns None when git cannot answer, so the caller can degrade loudly."""
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "-z"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    files = {f for f in out.stdout.split("\0") if f}
    if not files:
        return None
    known = set(files)
    for f in files:
        parts = f.split("/")
        for i in range(1, len(parts)):
            known.add("/".join(parts[:i]))
    return known

TASK_RE = re.compile(r"^\s*-\s*\[( |x|X|~|-)\]\s+(P\d+\.\d+[a-z]?|[A-Z]+\d*-\d+[a-z]?)\b")
PATH_RE = re.compile(r"(?<![\w/.-])(\.work/[\w./-]+|(?:docs|tools|tests|scripts|server|src|cli)/[\w./-]+|[\w-]+\.md)(?![\w/-])")
ADDENDUM_TASK_RE = re.compile(r"^\s*(?:[-*]\s*)?Task:\s*\**\s*(P\d+\.\d+[a-z]?|[A-Z]+\d*-\d+[a-z]?)", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="PLAN.md")
    ap.add_argument("--work", default=".work")
    a = ap.parse_args()
    root = os.getcwd()
    if not os.path.exists(a.plan):
        print(f"check_plan: {a.plan} absent (local, untracked) — nothing to check")
        return 0

    known = git_truth(root)

    def exists(p):
        """True when a FRESH CLONE would have this path."""
        p = p.rstrip("/")
        if p.startswith(UNTRACKED_BY_DESIGN):
            return None                       # exempt: counted, not judged
        if known is None:
            return os.path.exists(os.path.join(root, p))
        return p in known

    lines = open(a.plan, encoding="utf-8").read().splitlines()
    failures, ids, task_ids = [], {}, set()
    exempt = 0
    current = None          # (id, done)
    for n, line in enumerate(lines, 1):
        m = TASK_RE.match(line)
        if m:
            done, tid = m.group(1).lower() == "x", m.group(2)
            if tid in ids:
                failures.append(f"line {n}: duplicate task id {tid} (first at line {ids[tid]})")
            ids.setdefault(tid, n)
            task_ids.add(tid)
            current = (tid, done)
        elif line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            current = None  # a heading or paragraph ends the task block
        for p in PATH_RE.findall(line):
            p = p.rstrip(".,;:)")
            here = exists(p)
            if here is None:
                exempt += 1
                continue
            if p.endswith(".md") and "/" not in p and not here:
                # a bare *.md that is not a file here may be prose (e.g. "the runbook.md");
                # only flag it when the line says detail:/doc:/see:
                if not re.search(r"\b(detail|doc|see|evidence):", line):
                    continue
            if not here:
                who = f" (task {current[0]}{', marked [x]' if current[1] else ''})" if current else ""
                # The recurring version of this failure is a BUILT artefact:
                # the board names tests/foo, which exists on the machine that
                # wrote the line and is gitignored everywhere else, so it fails
                # only in CI on a fresh clone. Say so instead of leaving the
                # next reader to rediscover it.
                hint = ""
                for ext in (".c", ".sh", ".py"):
                    if exists(p + ext):
                        hint = (f" -- but {p}{ext} is tracked: the board names a built"
                                " binary, which is not in a fresh clone; name the source")
                        break
                if not hint and os.path.exists(os.path.join(root, p.rstrip("/"))):
                    hint = (" -- present on this disk but not in git: either track it"
                            " or stop the board depending on it")
                failures.append(f"line {n}: missing path {p}{who}{hint}")

    # addenda that name a task the plan does not have
    if os.path.isdir(a.work):
        for dp, _, fs in os.walk(a.work):
            for f in fs:
                if not f.endswith(".md"):
                    continue
                path = os.path.join(dp, f)
                try:
                    txt = open(path, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for tid in ADDENDUM_TASK_RE.findall(txt):
                    if tid not in task_ids:
                        failures.append(f"{path}: names task {tid} which is not in {a.plan}")

    n_tasks = len(task_ids)
    n_done = sum(1 for l in lines if TASK_RE.match(l) and TASK_RE.match(l).group(1).lower() == "x")
    if failures:
        how = "filesystem (no git here)" if known is None else "git (what a fresh clone has)"
        print(f"check_plan: FAIL ({len(failures)} problem(s), {n_tasks} tasks, "
              f"{n_done} done; paths resolved against {how})")
        for f in failures:
            print("  " + f)
        return 1
    how = "filesystem (no git here: WEAKER than CI)" if known is None else "git"
    extra = f", {exempt} exempt (untracked by design)" if exempt else ""
    print(f"check_plan: PASS ({n_tasks} tasks, {n_done} done, {len(lines)} lines, "
          f"every referenced path exists in {how}{extra})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
