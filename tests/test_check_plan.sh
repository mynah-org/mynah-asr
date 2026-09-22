#!/bin/sh
# Does the plan checker exercise the path CI takes?
#
# It did not.  Until 2026-09-22 check_plan.py asked the FILESYSTEM whether a
# referenced path existed, so a board line naming a gitignored built binary was
# green on the author's disk and red in every fresh clone -- three days of red
# `Build & Test` jobs behind a locally green `make check`.  A gate that cannot
# fail on the machine that writes the code is not a gate.
#
# This builds a throwaway repository with exactly that shape (tracked source,
# untracked binary that IS on disk) and asserts the checker fails on it, then
# asserts it passes once the board names the source.  Exit: 0 ok, 1 fail.
set -u
CHECKER=$(cd "$(dirname "$0")/.." && pwd)/tools/check_plan.py
TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT
fail=0
say() { if [ "$1" = 0 ]; then echo "check_plan ok:   $2"; else echo "check_plan FAIL: $2"; fail=1; fi; }

D=tests                 # built by hand: a literal here would read as a dependency of THIS repo
mkdir -p "$TMP/repo/$D" "$TMP/repo/.work"
cd "$TMP/repo" || exit 1
git init -q . 2>/dev/null || { echo "check_plan SKIP: no git"; exit 0; }
git config user.email t@t; git config user.name t
echo "int main(void){return 0;}" > "$D/test_thing.c"
echo "$D/test_thing" > .gitignore
printf '# note\n' > .work/README.md
git add -A >/dev/null 2>&1
git commit -qm init >/dev/null 2>&1 || { echo "check_plan SKIP: no commit"; exit 0; }

# the built binary: on this disk, never in a clone
echo built > "$D/test_thing"
chmod +x "$D/test_thing"

printf -- "- [x] S1-1 gated by \`%s/test_thing\`\n" "$D" > PLAN.md
python3 "$CHECKER" >"$TMP/out1" 2>&1
[ $? -eq 1 ]; say $? "a board line naming a gitignored binary FAILS although the file is on disk"
grep -q 'built binary' "$TMP/out1"
say $? "and the message names the cause instead of leaving it to be rediscovered"

printf -- "- [x] S1-1 gated by \`%s/test_thing.c\`\n" "$D" > PLAN.md
python3 "$CHECKER" >"$TMP/out2" 2>&1
[ $? -eq 0 ]; say $? "naming the tracked source passes"

# untracked by design (.work/README.md rule 7) is exempt, not silently skipped
printf -- '- [ ] S1-2 raw at `.work/evidence/run-1.json`\n' > PLAN.md
python3 "$CHECKER" >"$TMP/out3" 2>&1
[ $? -eq 0 ]; say $? ".work/evidence is exempt even though no clone has it"
grep -q '1 exempt' "$TMP/out3"
say $? "and the exemption is counted in the PASS line"

# a file present and tracked is still fine
printf -- '- [ ] S1-3 see `.work/README.md`\n' > PLAN.md
python3 "$CHECKER" >"$TMP/out4" 2>&1
[ $? -eq 0 ]; say $? "a tracked .work note passes"

echo "test_check_plan: $([ $fail = 0 ] && echo OK || echo FAIL)"
exit $fail
