# Campaign ledger

Task: C-1

One line per decision, newest phase last. FACT is measured. HYPOTHESIS is not.
A failed hypothesis stays here rather than disappearing, because the cost of
re-proposing it later is higher than the cost of a row.

Detail lives in the per-item notes; this is the index a new session reads first.

---

## Closed before this campaign (frozen, do not reopen without new evidence)

| id | claim | verdict |
|---|---|---|
| R-8 | initial predictor-state emission lock | **REJECTED** — injection delays emission (median +4 decisions), degrades CER, and disrupts equally after the first token |
| R-8 | the extreme blank logits are an int8 artefact | **REJECTED** — f32 reproduces them frame for frame |
| R-8 | forcing a first token as a fix | **REJECTED** — the only arm that advances emission (`▁`) drops first-word correctness 16/21 → 10/21, and at a deployable point doubles CER |
| R-5 | `[56,0]` as a quality-preserving latency win | **REJECTED** — mean CER 0.0405 → 0.0525 |
| R-10 | "`[56,3]`→`[56,0]` removes 240 ms" | **FALSE** — the attention is unmasked inside the chunk, so the expectation is `(q-1)/2 × 80` = **120 ms**; measured −0.100 s median |
| R-10 | cache fill depends on the preset | **FALSE** — `ceil(56/q)` steps of `q×80 ms` is **4480 ms in every preset** |
| R-11 | within-chunk position explains first-token timing | **UNTESTABLE** — 17/21 crossings at position 0 by construction of the greedy scan |
| R-11 | the within-chunk lookahead does real work | **FACT** — margin 2.26/4.58/4.88/6.11 across positions, 3101 balanced post-crossing decisions |
| R-13 | where the positions diverge | **FACT** — identical through the whole conformer stack; first divergence at the prompt projector's ReLU (input 1.13×, output 7.21×) |
| R-13 | the prompt one-hot dominates the ReLU gating | **HYPOTHESIS** — read from the code, not measured |
| R-13 | high encoder norm is harmful | **NOT ESTABLISHED** — co-occurrence with a large blank margin only |

---

## PHASE A — EN/FR quality gates

**EXPERIMENT A-1.** Is the committed corpus sufficient to claim English and
French quality? Run the new gate on it and look at the interval, not the point.

**RESULT A-1 — FACT. It is not, and now there is a number.**

| | n | WER mean | 95 % CI | width |
|---|---|---|---|---|
| EN | 4 | 0.0561 | [0.0000, 0.1122] | **0.1122** |
| FR | **2** | 0.1979 | [0.0625, 0.3333] | **0.2708** |

A French interval 27 points wide cannot distinguish a good model from a bad
one. Every quality statement made on this corpus so far was a DIAGNOSTIC
statement and must not be quoted as a quality claim.

**DECISION A-1.** Keep the 21-clip corpus for mechanism work, where it has been
excellent, and build a separate evaluation bank. FLEURS **test** split, English
and French, 200 utterances each (~34 min per language) — a split nothing in this
repo has ever been tuned on. Ground truth is the dataset transcription; the
model's own offline output is never an oracle.

**FACT A-2, found while building it.** FLEURS records the same sentence from
several speakers under one `fleurs_id`: 200 French records are only 100 distinct
ids. Naming the output by id silently overwrote half the bank and produced a
manifest with 400 rows over 209 files — some audio would have been scored
against another speaker's row. Caught by printing both counts; the fetcher now
names by source file and **asserts** the two counts agree.

**RESULT A-3 — FACT. The frozen `[56,3]` int8 baseline, 400 utterances.**

| | n | audio | WER mean | 95 % CI | CER mean | S / D / I | ref words |
|---|---|---|---|---|---|---|---|
| **EN** | 200 | 2054 s | **0.1211** | [0.1056, 0.1384] | 0.0713 | 388 / 63 / **105** | 4568 |
| **FR** | 200 | 2060 s | **0.1316** | [0.1148, 0.1496] | 0.0651 | 478 / 56 / **137** | 5283 |

| | empty | first published word CORRECT | prefix rewritten | speech → first word (med / p95) | speech → first CORRECT word |
|---|---|---|---|---|---|
| EN | 0/200 | **82.5 %** | 0/200 | 1.075 / 2.200 s | 1.020 s (n=165) |
| FR | 0/200 | **86.0 %** | 0/200 | 0.840 / 1.460 s | 0.810 s (n=172) |

WER by duration — **short utterances are the weak spot, and more so in French**:

| | < 6 s | 6–12 s | ≥ 12 s |
|---|---|---|---|
| EN | 0.147 (n=17) | 0.111 (n=132) | 0.138 (n=51) |
| FR | **0.212** (n=14) | 0.126 (n=138) | 0.125 (n=48) |

The interval went from 0.27 wide (FR, n=2) to **0.035** (n=200): the gate can now
see a 2-point WER move, which is what makes Phase C's candidates judgeable.

**Three things this baseline says that were not visible before.**

1. **Insertions outnumber deletions in both languages** — EN 105 against 63,
   FR 137 against 56. This engine leans toward saying too much, not toward
   truncating. Any latency change must be checked against the insertion count,
   not only against WER, because emitting earlier is exactly the pressure that
   would raise it.
2. **About one first published word in six is wrong** (17.5 % EN, 14.0 % FR) on
   real data, and this path never retracts. Q-2 measured this on 21 clips and it
   holds at n=400.
3. **Short utterances are the weak spot**, French especially. That is also where
   first-emission behaviour matters most, so the two problems overlap.

**NEXT.** Phase B: does the R-13 high-norm regime predict any of these?
