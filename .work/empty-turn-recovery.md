# Recovery for streaming turns that end EMPTY

Task: S18-1
Task: S18-2
Task: S18-3
Task: S18-4

Status: V1 2026-10-28 (E14): library + CPU server + CUDA server, opt-in, awaiting review. Open: CPU default deadline, early reject. Written before the work, per `.work/README.md` rule 2.

## Task

Add an OPTIONAL, default-off recovery for a streaming Nemotron 3.5 turn whose
final transcript is empty. Two stages, tried in order:

1. **empty_retry** — re-decode the turn with an alternate language prompt
   (`auto` when the session pinned a tag). Accept only useful text whose
   language/script is sane for the session.
2. **context_recovery** — if still empty, re-decode
   `[trailing replay_context_ms of the PREVIOUS turn's audio][current turn]`
   from a fresh state, keep only the tokens that belong to the current turn
   (boundary filtering), then the same gate.

One sentence for the eventual PR: *add optional recovery for empty streaming
RNNT turns using a prompt retry followed by bounded previous-audio context
replay, with regression tests for replay boundaries.*

Out of scope: scheduler changes, CUDA kernels, quantisation, new models, a
serving redesign, a new quality suite, tuning. Defaults never change (§11).

## Question

- Does a fresh-state replay with a bounded audio prefix recover empty turns
  WITHOUT turning them into wrong text and without leaking context words?
- Is `token_frame >= keep_from` a sufficient rule for "this token belongs to
  the current turn" in an RNNT, or does the transducer emit tokens that
  semantically belong to the context (sentence-final `?` `.` `!`, a delayed
  word piece) AFTER the temporal boundary?

## Known facts (this repo)

- The language prompt is injected AFTER the conformer stack
  (`mynah_asr_encoder_post_scratch`, src/encoder.c): conformer caches are
  prompt-independent. `auto` is prompt 101, the pack default. With `auto` the
  model emits a `<xx-XX>` tag the detokeniser lifts out (src/tokenizer.c).
- Greedy RNNT already records the absolute encoder frame of every emission when
  given a `frames` buffer (src/decoder.c); the streaming path passes NULL.
  One encoder frame = hop 160 x subsampling 8 = 1280 samples = 80 ms.
- A turn on the CPU server: reset -> feeds -> `sched_finalize` ->
  `mynah_asr_stream_finish` -> `done`. Deltas are final and never retracted; an
  empty turn published nothing, so a recovered text can go out before `done`
  without contradicting the client.
- No past PCM is retained anywhere (slot ring is FIFO, mel window is short).
- A second stream is ~13 MB and shares only the read-only model.
- Finalize is already the largest serial phase of the scheduler (F20): a
  replay must be bounded and skippable under load (rule 5: when full, refuse).
- Earlier own research on empty turns (branch `inv/nemotron-empty-transcript`,
  not on main): a wrong language tag empties ~50 % of long turns and a turn
  starting in another language than the tag 15-21 %, 0 % with `auto` — the
  retry is well supported. `auto` does NOT rescue babble/crosstalk. **Prepending
  closed-sentence speech to a short turn RAISED empties there (FLEURS FR
  24.6 -> 37 %)** — but as stream context, without fresh state or boundary
  filtering. The replay must be shown to differ before it is built.

## Unknowns

- U1 Replay recovery rate on primary-EMPTY turns, and its wrong-text rate.
- U2 Leakage: how often the first kept token belongs to the context, as a
  function of the context's last punctuation (`.` `?` `!` none) and of
  `replay_context_ms` (500 / 1000 / 2000 / 3000).
- U3 Whether a word-start rule (drop leading kept pieces that do not open a
  word: continuation pieces, bare punctuation) fixes leakage without dropping
  real current-turn words.
- U4 Whether the replay should use the primary tag or the retry's language.

## Plan

S18-1 **Mechanism first, CPU Mac, no server.**
- a. A pure, model-free module: boundary selection over `(tokens, frames,
  keep_from)` with rule `frame` (v0) and `frame + word-start` (v1), and the
  recovery gate (has letters; script matches the session language; not a
  repetition of the context transcript tail). Model-free unit test on
  synthetic sequences: context ends `.` `?` `!` / no punctuation, delayed
  lexical piece, very short current turn, silence at the boundary.
- b. A lab binary (built on demand, not in `make test`): per item prints the
  primary text, the retry text, the full replay token/frame sequence,
  `keep_from`, kept/dropped pieces and both rules' texts, as JSONL.
- c. Items from the committed/fetched FLEURS eval bank (CC-BY 4.0), en + fr:
  context = a full clip; current = a word-aligned prefix of another clip
  (0.4 / 0.7 / 1.0 / 1.5 s, reference = its words), plus full clips as the
  no-harm control.

S18-2 Library API (only if S18-1 passes): pooled recovery stream, frames
recorded in the stream decode, `mynah_asr_stream_recover(...)`; byte-identical
output with recovery off; recovery output independent of B / PAR / FIN_STACK.

S18-3 CPU server opt-in: `--empty-recovery` + per-session query key,
preallocated per-slot PCM history (prev tail + current turn, capped),
admission budget (skip and count when busy), counters, `ws_probe` subcommand.

S18-4 CUDA server: only after S18-3 and only if asked; rented box.

## Acceptance gate (pre-registered for S18-1c)

On primary-EMPTY items, per language:
- **retry**: ships if it recovers >= 30 % of empties into correct-or-near text
  (CER <= 0.3 vs the reference) with wrong (CER > 0.5) <= 1/3 of recovered.
- **replay**: ships only if it beats retry-only on the residual empties by the
  same rule AND leakage (first kept word from the context) <= 2 % of recovered
  under the chosen boundary rule. Otherwise REJECTED with the numbers here.
- **no harm**: on non-empty primaries nothing changes by construction (the
  recovery runs only on empty turns); asserted in the lab output.

## Files/functions inspected

src/mynah_asr.c (stream struct, `stream_decode`, `mynah_asr_stream_finish`,
`mynah_asr_stream_reset`, `resolve_prompt`), src/decoder.c
(`mynah_asr_greedy_decode_scratch` frames), src/encoder.c
(`mynah_asr_encoder_post_scratch`), src/tokenizer.c, server/sched.c
(`sched_finalize`, `sched_emit_done`), server/slot.c, gpu/server/main.c.

## Evidence

S18-1, 2026-10-28, macOS (Apple Silicon, int8 pre-quantized pack) unless noted;
streaming path, default preset [56,3] (4-frame = 320 ms encoder chunk), B=1.
Tool: `tests/recovery_lab` + `tools/eval/recovery_lab.py` (items, report).
Data: FLEURS eval bank (CC-BY 4.0), en/fr; MINDS-14 (CC-BY 4.0) en-US/fr-FR,
fetched locally, not committed.

Item sets (denominators):
- FLEURS: 15 current clips per language x {0.4, 0.7, 1.0 s from speech onset,
  whole clip} = 120, + 8 fr clips tagged en-US ("provoked") = 128. Previous turn
  = another clip of the language. No FLEURS context decodes ending in `?`.
- MINDS-q: 10 previous turns per language chosen as questions AND decoded alone
  with a final `?` (36/40 en, 31/40 fr candidates did), x {0.4, 0.7, 1.0, whole}
  = 80, + 18 lead-in-only "silence" controls = 98.
- Replay arms per item: 480 / 960 / 2000 / 2960 ms and the whole previous clip.
  Dose sweep on short turns (168 = FLEURS 90 + MINDS 78): 0 / 240 / 320 / 480 /
  560 / 720 / 960 / 1440 ms (requested lengths floored to whole 80 ms frames),
  repeated with a -60 dBFS noise prefix instead of previous-turn audio.

E1 Ownership (A, B). FRAME handed the current turn the context's sentence
punctuation in 109/490 MINDS-q replays starting with `?` (whole-clip arm: 67/98
replays), and a context word piece in 5-18 replays per arm. Example: context
"... scorch." -> `.` emitted at frame 132, boundary 127. WORD/TEXT: 0/1130
replays start with punctuation. **Attribution loss** (WORD recall < FRAME
recall - 0.1, healthy turns): **0 in every cell** (both sets, both languages,
every arm and dose). Model-free test: 83 checks, incl. the `?` regression.

E2 Suppression (healthy short turns, recall of reference words). Long
previous-turn context makes the replay drop the current turn: SUPPRESSED (FRAME
recall < 0.2 with primary >= 0.5) FLEURS en 2/24 (480 ms) -> 11/24 (whole);
MINDS-q fr 0/12 (480) -> 8/12 (2960). Whole (complete) turns: <= 1 per cell.
Example: after "... au guichet automatique?" the current "Comment faire pour
deposer de l'argent" is absent from the replay. The severe suppression is
associated with linguistic context and is not reproduced by the matched
-60 dBFS noise prefix (1440 ms: FLEURS en recall 0.19 context vs 0.28 noise;
MINDS-q fr 0.56 vs 0.70): duration/phase alone are insufficient to explain it
under the tested conditions. Length, phase and content were not isolated as
independent factors.

E3 Chunk phase. The dose curve is periodic in (prefix frames mod 4): phase 3
(240 / 560 ms) is the worst dose in all four groups, with context AND with the
noise prefix alike (FLEURS en 0.27/0.23 at 240, 0.25/0.23 at 560). The gains of
short context on French (FLEURS 0.61 vs primary 0.55; MINDS-q 0.70 vs 0.56 at
480 ms) are matched by the noise prefix (0.60, 0.70): **on healthy turns a phase
effect, not a context effect.** On primary-EMPTY turns, trying three short
prefixes (240/320/480 ms, first gate-ok wins) recovers, correct / wrong /
accepted-on-silence: FLEURS 36 empties: noise 8 / 2 / -, context 11 / 3 / -;
MINDS-q 28 empties: noise 4 / 6 / 0 of 14, context 6 / 5 / 2 of 14. Phase
explains most of the recovery; short context adds a few recoveries AND false
recoveries on silence (small n: a few items either way).

E4 Retry with `auto`: on natural short-turn empties it recovered nothing (all
still empty). The 8 provoked en-US-tagged French clips did not end empty
(0/8), so the wrong-tag class was not exercised here.

E5 Recovery on natural empties (WORD, gate ok, judged vs the reference start):
FLEURS 36 empties: 480 ms 4 correct / 1 wrong; 2000 ms 8 / 5; 2960 ms 8 / 0;
whole 1 / 0. MINDS-q 28 empties: 480 ms 2 / 0; 2000 ms 8 / 2; whole 4 / 0.
Pre-registered gate (>= 30 % recovered, wrong <= 1/3 of recovered): **not met**
by any arm (best 29 %, MINDS-q 2000 ms).

E6 Silence/lead-in controls (18): replay text accepted by the gate at 480 /
960 ms context: 3 / 5 (all rules); >= 2000 ms with WORD/TEXT: 0.

E7 Cross-machine replica (x86-64, AVX-512, runtime int8 from f32 on a rented
box): not byte-identical to the Mac run (primary identical 101/128 FLEURS,
88/98 MINDS) because the int8 weights differ (pre-quantized pack vs runtime
quantization) and so do the kernels; read as a statistical replica only. It
replicates E1-E2: attribution loss 0 in every cell; SUPPRESSED FLEURS en whole
arm 12/24 (Mac 11/24), MINDS-q fr 2960 ms 8/12 (Mac 8/12); FRAME `?` leaks on
the whole arm 65 (Mac 67; the box decoded 5 of the 98 contexts without `?`).

Caveat on the test bed: a "short turn" is the first D seconds after an energy
onset, cut mid-speech with no trailing silence. Real short turns end with a
pause; the judge for prefixes is approximate (words among the reference's
first n+2). The mechanism results (E1-E3) do not depend on the judge.

E8 Healthy-path cost of the library change (emission frames stored for every
token), x86-64 box, 1 thread, int8, three long files (95 s FLEURS stream,
en_long, de_long), 5 interleaved OFF (d947275) / ON reps: transcripts
byte-identical 15/15; paired wall-time difference median 0.0 %, mean +0.1 %,
sd 9.6 % (n = 15). The rented box's run-to-run noise (up to 13-21 % on one
file) bounds the resolution to about +-5 % on the mean: no effect detectable,
and the change is one int store per emitted token.

CUDA replica (S18-4 lab only, 2026-10-28, rented L40S, sm_89): the same
frozen items, arms, doses, noise control, rules and gate, through the GPU
server engine (`tests/cuda_recovery_lab`, `make -C gpu recovery-lab`), three
numeric arms: f32 (the engine default), bf16 own-tc, int8 weights.
Instrumentation added: `asr_engine_cfg.record_frames` + `asr_engine_slot_tokens`
(off by default: no extra copy). Gates: `test_cuda_stream` A/B/C PASS; a
400-clip FLEURS bank is byte-identical OFF (d947275) vs ON.

E9 A latent bookkeeping bug, found by this replica: the CUDA label loop wrote
`tok_frame = t_abs + dec_t` after `kv_advance` had already moved `t_abs` past
the chunk, so every emission frame read one chunk late (+4 frames; +q on the
tail). Text was unaffected (gate B compares text), and `tok_frame` had no
reader before S18. With frames one chunk late, a boundary computed from audio
hands the context's last word to the current turn: the first CUDA run showed
WORD keeping "ATM?" from the context. Fixed in `gpu/cuda/kernels.cu`
(`t_abs - r.q + dec_t`); after the fix the CPU reference engine and the CUDA
engine report identical frames on 36/39 same-text replays (the 3 others +-1-2
frames on a final `?`, int8 CPU vs f32 GPU). All CUDA numbers below are from
the rerun.

E10 CPU vs CUDA (f32 / bf16 / int8 weights), raw counts:
- Primary EMPTY: FLEURS prefixes CPU 36/90, CUDA 37/37/36; MINDS-q prefixes
  28/60 vs 29/29/29; silence controls 14/18 vs 16/16/16.
- Ownership: FRAME `?`-led replays MINDS-q 109/490 vs 110/110/106; WORD/TEXT
  punctuation-led 0 everywhere; attribution loss 0/260 vs 0/255 (all arms).
- Suppression, short turns, whole previous clip: FLEURS en 11/24 vs 8/7/8;
  FLEURS fr 6/30 vs 5/5/6; MINDS-q fr 2960 ms 8/12 vs 8/8/6. Same direction
  and dose-dependence; CUDA somewhat lower on FLEURS en.
- Phase: the phase-3 dip (240/560 ms) is present on every arm, with context
  and with noise (FLEURS en noise 0.23/0.20 vs primary 0.38-0.41).
- Recovery (try 240/320/480 ms, first gate-ok): FLEURS context 11 ok / 3 wrong
  vs 10/2 on all CUDA arms; noise 8/2 vs 10/2, 10/2, 8/2. MINDS-q context 6/5
  vs 8/5, 8/5, 8/7; noise 4/6 vs 7/5 on all. **Silence false recovery with the
  noise prefix: CPU 0/14, CUDA 2/16 on all arms** (context: 2/14 vs 3/3/4 of 16).
- Pre-registered gate: failed on every arm and platform except one cell (int8
  weights, MINDS-q 2000 ms: 9/29 = 31 %, wrong 2), which clears 30 % by one
  item and is not read as a pass.
- Retry with `auto`: 0/36 and 0/28 on CPU; 0/37 and 1/29 on CUDA.

E11 Acceptance signal (from the sweep/noise data of E3 and E10, no new runs):
agreement between attempts. For each primary-EMPTY turn, the most frequent
gate-ok WORD text over the 7 doses and how many doses produced it: support >= 3
keeps silence false recoveries at 0-1 on every platform while keeping most
correct ones (CUDA f32: correct 8, wrong 2, silence 0 of 16). Cheap form:
3 phase attempts (240/320/480 ms noise prefix) + a 480 ms replay as a 4th vote,
accept when >= 2 agree on the words: CPU 6 correct / 4 wrong / 0 silence;
CUDA f32 and bf16 9 / 2 / 0 of 16; int8 weights 8 / 2 / 0 (of 64-66 empties).
The replay vote adds +2 correct on every platform with no added wrong or
silence acceptance. Most "wrong" accepted texts are artefacts of the prefix
test bed: "uh" (a spoken filler the MINDS reference omits, support 5-6),
"pour" (the 0.7 s cut inside "pourriez"); one real error ("however" for "I
would like"). Phase-only re-decode without the vote accepted 2/16 silences on
CUDA (E10): the vote is what makes it silence-safe on these data.

E12 Ownership study (corrected CUDA dumps, 226 items x 3 arms = 678 replays,
every token with its frame, the previous turn and the current turn decoded
alone as acoustic references):
- A, delayed punctuation: 270 crossings. The context's final `?`/`.` is
  emitted only when the next speech arrives, typically on the same frame as
  the current turn's first word (`?` and "I" both at frame 40); up to 27 frames
  after the boundary on long turns. It belongs to the word it closes: WORD and
  TEXT give it back.
- B, a word opened before the boundary finishing after it: 42 ("automatique":
  `ma` at 37, `t` `ique` at 38, boundary 37). Belongs to its opener: WORD/TEXT.
- C, a whole context word opened at or after the boundary: 5, all "you" of
  "... thank you." opened 0-1 frame after keep_from; the previous turn decoded
  alone emits it at frame 72.4 of the replay, its audio ends at 73. WORD keeps
  it (wrong); TEXT gives it back (right).
- D, genuine current first words: opened >= 3 frames after the boundary.
- TEXT differs from WORD in exactly those 5 replays and in no other of the 678.
Rule adopted: TEXT with a one-chunk window (a word is given back only when its
first piece is emitted less than lookahead + 1 frames after keep_from), the
invariant stated in `src/recovery.h`. Its cost (a current turn whose first word
repeats the previous turn's last word within the first chunk loses it) is
pinned by the model-free tests.

E13 The library path end to end (`mynah_asr_recover_turn`, default config:
phases 3/4/6 frames + 6-frame replay vote, >= 2 votes, TEXT with a one-chunk
window), x86-64 box, int8, the E1 items with the whole previous clip as the
previous turn (`tests/recovery_lab --recover`): FLEURS short turns 37 EMPTY ->
5 accepted, 5 correct, 0 wrong; MINDS-q short turns 30 EMPTY -> 5 accepted, 4
correct, 1 "Uh." (a filler actually spoken, absent from the reference);
silence/lead-in controls 15 EMPTY -> 0 accepted. Whole clips: no EMPTY, so
recovery never ran. Recovered texts: "What's my current?", "Comment faire",
"Une bombe.", "Est-ce que je", "Les ports sont". Precision high, recall modest
(about 13 % of the EMPTY turns of this cut-mid-speech test bed).

E14 V1 on the CUDA server (L40S, `mynah-asr-server-cuda --cap 64`, the 98
MINDS-q items as two-turn sessions, the exact f32 samples of
`tests/test_cuda_empty_recovery --dump-dir`, `gpu/tools/recovery_ws_check.py`):
- OFF vs ON with `recover=0`: 98/98 sessions identical (all frames but timing).
- OFF vs ON: 92 identical, 6 differ only by a recovered turn, 0 otherwise.
  45 EMPTY turns: 41 attempted, 6 accepted (all won by a phase attempt), 35
  rejected, 4 skipped (all spare slots taken). Recovered: "What's", "What's my
  current?", "Uh.", "Est-ce que je", "J'ai", "Comment faire?".
- `--recovery-deadline-ms 1`: 44 attempted, 44 ended at the deadline, 0
  published; output identical to OFF.
- `--recovery-slots 1` (fewer than a job's 4 attempts): 45 skipped, identical
  to OFF.
- `test_cuda_stream` A/B/C PASS (f32, bf16 own-tc); `test_cuda_empty_recovery`
  MINDS-q f32 and bf16: decisions 98/98 equal to the CPU reference, silence and
  noise (digital silence, -60/-45/-30 dBFS, 0.4-3.2 s) 0/32 accepted, live
  lanes beside recovery lanes byte-identical 8/8.
CPU server: `tests/test_server_protocol.sh` 13/13 on macOS and Linux x86
(recovery case: noise/silence never recovered, a healthy turn identical ON vs
OFF, an EMPTY cut recovered as "Hello." with the duty admission forced off,
bounded fallback at 400 ms). On a loaded box one stream stepped at 444 ms per
320 ms chunk and the default admission (duty 0.8) skipped every recovery, which
is the intended behaviour on a saturated worker. One recovery costs the lane
~1 s on the Mac and 5 s on that loaded box: at the 400 ms default deadline CPU
recovery is effectively off; the default is left open until a quiet x86
measurement.

## Conclusion

(provisional, pending the healthy-path A/B and review)
- Boundary ownership by WORD is the right rule: it removes the context's
  delayed punctuation and word pieces with no measured attribution loss.
- Context replay is bounded by suppression: on short turns, long previous-turn
  linguistic context is associated with the decoder dropping the current turn
  (not reproduced by a matched noise prefix).
- What recovers most empties at short doses is the chunk-phase shift, which a
  content-free prefix gives as well; short real context adds a few recoveries
  and a few false ones on silence. A phase-shifted re-decode of the current
  turn alone is the hypothesis to test next, with no previous audio, no echo
  and no ownership problem; short context stays a second arm to compare.
- Neither stage meets the pre-registered 30 % recall gate on this test bed.
  After review the work moved from "ship if recall >= 30 %" to a conservative
  V1: precision first (E13, E14), recall about 13 % of the EMPTY turns, no
  silence or noise ever accepted, healthy turns untouched.

E15 WORD refinement (2026-10-29). Three boundary shapes the WORD rule handled
wrongly by construction, none present in the E12 bank (587 CUDA f32 replays,
en/fr only):
- the current turn's first word started by a bare ▁ one frame before keep_from
  (chunk lookahead) with its letters after it: "▁" | "ye" "s" lost "yes";
- an unmarked piece after a language tag ("." "<en-US>" | "J" "o");
- scripts written without spaces (zh/ja/th...): no ▁ after the boundary, so
  WORD returned n and the turn was EMPTY.
Fix: unmarked pieces from keep_from on continue the context's last word only
when that word already holds content and is not ended by a special piece; in
an unspaced script they are the current turn's. A context tail alone still
gives n. `tests/test_recovery` cases 18-21 (130/130). The old and new rules
give identical WORD and TEXT output on all 587 E12 replays (a C driver over the
dumped tokens and frames, linked once with each `recovery.c`).

## Next action

S18-1a: the pure boundary/gate module and its model-free test.
