# Agents — repository entrypoint

Read `ENGINEERING.md` before changing the repository. It is the single
normative source for planning, evidence, backend claims, benchmark lifecycle,
privacy and commits; do not duplicate or override those rules here.

`PLAN.md` is the board. Load only the `.work/*.md` note linked by the item you
work on. Raw benchmark and profiler material belongs in the untracked evidence
areas described by `ENGINEERING.md` §2.

Build and gates: `make`, `make test`, `make test-server-concurrency`,
`make check`; on macOS `make leaks` and `make ubsan`, never ASan.
