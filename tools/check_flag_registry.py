#!/usr/bin/env python3
"""check_flag_registry.py — the flag registry must be complete, in both directions.

    python3 tools/check_flag_registry.py

src/flags.c holds ONE table of every environment variable this runtime reads
(ENGINEERING.md §5: a measurement is only as trustworthy as the description of
the environment that produced it). A table is only worth reading if it cannot
silently fall behind the code, so this script fails when

  * a name is READ in src/ cli/ server/ tests/ and is missing from the table
    -> a flag nobody can discover, and nothing tells an operator it was ignored
  * the table names a flag NOTHING reads
    -> a documented knob that does nothing, which is worse than no knob

"Read" means it appears as a string literal in getenv(), in one of the small env
wrappers (server/prefork.c env_int), or in the registry's own accessors
mynah_asr_flag_int()/mynah_asr_flag_str(). setenv() deliberately does NOT count:
a write is not a read, the read is what the table describes, and a test that
sets a deliberately unknown name to prove the accessors refuse it must not
thereby invent a flag.

One escape hatch, deliberately ugly so it stays rare: a read on a line that
also carries the marker `check_flag_registry: ignore` is skipped. It exists for
the test that proves the accessors REFUSE an unregistered name — that read must
not conjure the flag it is asserting does not exist.

Exit 0 and print the count, or exit 1 listing the offenders.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "src" / "flags.c"
SCAN_DIRS = ("src", "cli", "server", "tests")
SCAN_SUFFIX = (".c", ".h", ".m", ".cu", ".cpp")

# getenv("NAME") / env_int("NAME", ...) / the registry accessors
READ_RE = re.compile(
    r"\b(?:getenv|env_int|env_str|mynah_asr_flag_int|mynah_asr_flag_str)"
    r"\s*\(\s*\"([A-Z_][A-Z0-9_]*)\""
)
# a row of the table: {"NAME", MYNAH_ASR_FLAG_<SCOPE>, ...
ROW_RE = re.compile(r"\{\s*\"([A-Z_][A-Z0-9_]*)\"\s*,\s*MYNAH_ASR_FLAG_([A-Z]+)\s*,")
IGNORE_MARKER = "check_flag_registry: ignore"


def registry_names():
    txt = REGISTRY.read_text(encoding="utf-8")
    rows = ROW_RE.findall(txt)
    names = [n for n, _ in rows]
    dupes = sorted({n for n in names if names.count(n) > 1})
    return names, dupes


def read_names():
    found = {}
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SCAN_SUFFIX or not path.is_file():
                continue
            if path.resolve() == REGISTRY.resolve():
                continue  # the table itself is a description, not a read
            try:
                txt = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in txt.splitlines():
                if IGNORE_MARKER in line:
                    continue
                for name in READ_RE.findall(line):
                    found.setdefault(name, set()).add(str(path.relative_to(ROOT)))
    return found


def main():
    names, dupes = registry_names()
    if not names:
        print(f"FAIL: no rows parsed out of {REGISTRY.relative_to(ROOT)}")
        return 1
    registry = set(names)
    reads = read_names()

    fail = False
    for name in sorted(dupes):
        print(f"FAIL duplicate row in src/flags.c: {name}")
        fail = True
    for name in sorted(set(reads) - registry):
        where = ", ".join(sorted(reads[name]))
        print(f"FAIL read but not registered: {name}  (read in {where})")
        fail = True
    for name in sorted(registry - set(reads)):
        print(f"FAIL registered but nothing reads it: {name}")
        fail = True

    if fail:
        print("\nsrc/flags.c must describe exactly the environment this runtime reads.")
        return 1
    print(f"flag registry ok: {len(registry)} flags, all read, all described")
    return 0


if __name__ == "__main__":
    sys.exit(main())
