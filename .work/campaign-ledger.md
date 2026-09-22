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

**NEXT.** Freeze the `[56,3]` int8 baseline on the bank, then Phase B.
