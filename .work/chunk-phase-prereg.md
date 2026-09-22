# R-11 pre-registration — written before the grouped result was computed

Task: R-11

Data: `.work/evidence/r8-A-2026-09-22/`, the frozen `[56,3]` baseline, 21 clips,
q = 4. Nothing is run; nothing in the runtime changes.

## What the code says

The streaming attention is unmasked over `[valid cache + chunk]`
(`src/encoder.c`, softmax over `j in [0, K)`, `K = valid + Q`; no mask, no
`-INFINITY`). So all q frames of a chunk are decided at the same moment, having
consumed the SAME total audio -- the chunk's right edge -- and they differ only
in where their own centre sits inside it:

| position t | own audio ends at | future context |
|---|---|---|
| 0 | chunk end - 240 ms | 240 ms |
| 1 | chunk end - 160 ms | 160 ms |
| 2 | chunk end - 80 ms | 80 ms |
| 3 | chunk end | 0 ms |

Frame F covers audio ending at `(F+1) x 80 ms`; `t = F mod 4`.

## H-lookahead, and what would falsify it

**P1 (timing).** If the future context is doing work, a first token at position 0
can be decided on 240 ms LESS of its own audio than one at position 3. So

> `speech_at_frame = (F+1)*80ms - onset` at the crossing should INCREASE with t,
> monotonically, with a spread of about 240 ms between t=0 and t=3 at full scale.

Absent (flat), much smaller than 240 ms, or reversed falsifies P1.

**P2 (mechanism, paired within chunk).** All four frames of a chunk share the
same information pool if the attention really is bidirectional inside it.
Therefore, in the chunks BEFORE the crossing:

> if the lookahead is strong, `margin_lex(t)` should be roughly FLAT across t
> within a chunk; if each frame is instead dominated by its own past, the margin
> should fall steeply and monotonically with t.

This is the better test: it is paired inside a chunk, it uses every pre-crossing
decision rather than the single selected one, and it does not depend on which
frame happened to win.

## The confound, stated before looking

`F mod 4` is **not an exogenous treatment**. `mynah_asr_greedy_decode_scratch`
is called with `T = q = 4` and scans the chunk in order, breaking at the first
non-blank (`src/decoder.c`: `if (am != dec->blank) { first = b; best = am; break; }`).
So when more than one frame of a chunk would emit, **the lowest position always
wins by construction**. Any over-representation of low t is therefore expected
under BOTH hypotheses and carries no evidential weight. Only P1's magnitude and
P2's shape are informative, and neither is a causal claim.

Group sizes will be small and unbalanced. Counts are reported with every median
and no two groups are compared without them.
