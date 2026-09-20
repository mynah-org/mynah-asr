# Serving profiles

**One question: given this box and this objective, how should the server be run?**

```
configs/perf/schema.json   the format and what each status means
configs/perf/<box>-<model>-<mode>.json   one box shape, one model, one objective
```

```bash
tools/perf_profile.py list
tools/perf_profile.py validate                                  # every profile
tools/perf_profile.py show  <id>
tools/perf_profile.py command <id> -m models/<dir> -p 8090      # the exact invocation
tools/perf_profile.py env   <id>                                # export / unset, with reasons
tools/perf_profile.py check <id>                                # PREFLIGHT: refuses on a mismatch
```

## Why a file and not a runbook line

Every capacity number in this repo between 2026-09-18 and 2026-09-20 was taken
with the topology typed by hand, and three of those runs were invalid in ways
the numbers did not show:

- a ladder ran five rungs against a server that had exited on a flag that does
  not exist (`--lookahead` is per-request, not per-process);
- a Parakeet phase was measured against a Nemotron server, because the previous
  server had never been killed and still held the port;
- a W x T sweep found narrow workers superior, because the gate scored a lost
  stream as a warning rather than a failure.

A configuration that lives in shell history cannot be compared with itself a
week later. `check` is the preflight: it asks the binary what it resolves, and
refuses when that is not what the profile requires. A benchmark is invalid until
dispatch is proven (`ENGINEERING.md` §5).

## The status ladder, and what may claim what

| status | what was run |
|---|---|
| `unmeasured` | written from arithmetic or from another box. **Nothing was run.** |
| `screened` | a WAVE ladder found a ceiling. Screening may disqualify a rung; it never promotes one. |
| `qualified` | a SOAK of at least 10 minutes held the operating point, transcripts checked. |

`validate` enforces it: a profile cannot say `qualified` with an empty
`measured.soak`.

## What a profile owns, and what it must not

It owns the topology, the settings, the dispatch rows and the gates. It does
**not** own the chunk size and it does not own a transcript: load never enters
the chunk (repo rule 5), and a transcript never depends on batching, threads or
ISA (rule 4).

Every environment variable carries a `value` and a `why`, and `value: null`
means the variable must be **absent** rather than set to something harmless. A
flag with no reason is a flag nobody can remove later.

## One box, one profile

`a` and `b` in `T_step(B) = a + b*B` are functions of how many cores walk the
weights, so a profile does not transfer to a machine of a different size —
`tools/bench/box_advisor.py` refuses to carry constants across a 1.5x cpu-count
difference for exactly this reason. Copy a profile to a new box, run
`tools/bench/box_session.sh`, and let the ladder write the numbers back in.

The shape of this directory, and the discipline in it — `value` + `why` per
variable, a preflight that refuses rather than warns, a status that cannot be
claimed without the run that earns it — is taken from the sibling engines'
`configs/perf`.
