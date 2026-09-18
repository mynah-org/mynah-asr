# Small integrity items the checkers found

Status: OPEN

Task: H-1, H-2, H-3
Question: keep `python3 tools/check_repo_integrity.py` and
`tools/check_plan.py` green from a fresh clone.

Found on 2026-09-18 (first run of the checkers)
- H-1 `tools/eval/test_langs.py` depends on `tests/audio/langs/manifest.json`,
  which exists locally but is not tracked: it is produced by
  `make fetch-lang-samples`. Either the script says so and exits 77 when it is
  missing, or the manifest is tracked. Decide and make the checker pass.
- H-2 `tools/gen_kquant_ref.py` names `tests/test_gguf.c`, which does not
  exist (the test moved). Fix the reference.
- H-3 the archived TODO carries entries already done (SiLU via vvexpf is
  implemented in `src/backend.c`); the archive is history and is not edited,
  but the board must not re-list them.

Gate: both checkers PASS in `make check`, and `make check` runs in CI.
