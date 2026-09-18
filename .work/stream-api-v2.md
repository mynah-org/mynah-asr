# S1 — library seams the v2 server needs

Status: OPEN

Task: S1-1, S1-2, S1-3, S1-4
Question: which changes to `libmynah_asr` are required so a single scheduler
thread can drive B streams as slots, allocation-free, with cross-stream
batching of the encoder step, without changing any transcript?

Known facts (HEAD 5f0f802)
- `mynah_asr_stream_open/feed/finish/close` only; no reset. A session restart
  re-allocates ~13 MB and re-opens a VAD instance.
- Deltas are always final (`is_final=true`); `t0` is always 0; `is_eou` is
  emitted with empty text and the server drops the distinction.
- Feed re-detokenises the whole token history each chunk and mallocs the text.
- `mynah_asr_enc_stream_step` is allocation-free, but subsampling, encoder_post,
  greedy decode and (macOS) SiLU allocate per chunk.
- `mynah_asr_encoder_forward_batch` exists for offline only; there is no batched
  stream step. Weight-stationary batching of B slots' chunks would turn 24×B
  skinny GEMMs of Q rows into 24 GEMMs of B·Q rows.
- A model is shareable read-only across threads; the mutators
  (`set_decoder`, `set_target_lang`, `set_segment_limit`, `enable_vad`) are not
  thread-safe and must be called before serving.

Unknowns
- Whether a batched stream step can be made **byte-identical** to B single
  steps (it must: the transcript of a stream must not depend on who it was
  batched with). BLAS sgemm on a [B·Q, d] matrix may differ in accumulation
  order from Q rows alone; the own-GEMM path (S1-6) has the deterministic
  row-block contract, so S1-4 may depend on it.
- Whether the decode loop should move to a lane; only after S0 blames it.

Plan
- S1-1 `mynah_asr_stream_reset(s, lang, lookahead)`: clears caches, decoder
  state, VAD state, counters; keeps the allocations. Slots are pooled by the
  server. Gate: reset+feed ≡ close+open+feed byte-identical on the fixtures.
- S1-2 result struct gains `t0` per delta (frame of the first emitted token),
  `chunk_arrival` passthrough (caller-provided timestamp of the last sample of
  the chunk, returned on the delta so the server can compute `lag_ms`), and
  `is_eou` stays; server surfaces `final`, `eou`, `t0/t1`. No text change.
- S1-3 allocation-free chunk: per-stream scratch for subsampling/post/decode
  carved at open; incremental detokenisation (emit only new pieces; keep the
  ▁/UTF-8 boundary logic); SiLU scratch. Gate: LD_PRELOAD count per chunk = 0
  after warm-up, transcript byte-identical (`make test`).
- S1-4 `mynah_asr_enc_stream_step_batch(streams[], B)`: one pass over the
  weights per layer for B slots, then per-slot decode. Gate: for B ∈ {2,4,8},
  ragged presets allowed or refused explicitly, output byte-identical to B
  single steps on the same slots; the refused path is the old path.

Acceptance gate: `make test` and `make test-server-concurrency` green; the
new tests above green; `docs/api.md` updated; no transcript in `tests/` changed.

Evidence / Conclusion / Next action: pending S0.

## Evidence — S1-3 (allocation-free chunk), 2026-09-18, macOS arm64 (Accelerate)

Measurement (`tests/malloc_count.c` inserted with `DYLD_INSERT_LIBRARIES` +
`DYLD_FORCE_FLAT_NAMESPACE`, `tests/test_stream_allocs.sh`): the counter is a
whole-process total, so the per-chunk number is the DIFFERENCE between two runs
of the same command on the same model with inputs of different length — model
load, stream open, WAV load and the final flush are identical in both, so the
difference is exactly what the extra chunks cost. Command in both runs:
`./mynah-asr stream -m <nemotron-3.5-asr-streaming-0.6b> -i <wav> --quant int8`,
preset [56,3] (q = 4 encoder frames = 320 ms per chunk), `tests/audio/test_it.wav`
(5.229 s) against its first half (2.615 s), i.e. 8 steady-state chunks.

| tree | 2.615 s | 5.229 s | second half | per chunk |
|---|---|---|---|---|
| `354743c` (before) | 45975 | 46671 | 696 | **87** |
| this change (after) | 45171 | 45171 | 0 | **0** |

The 87 are exactly the audited sites: SiLU on Accelerate 24 × (2 FFN + 1 conv
module) = 72 · subsampling 9 (`a`, `bbuf`, `flat`, `xp` × 3 stages, the stage-0
im2col `P`, the depthwise pad × 2 stages — the audit listed 6, the im2col and
the two pads were inside `conv2d_s2`) · `mynah_asr_encoder_post` 3 (`cat`,
`mid`, `fused`) · `mynah_asr_greedy_decode` 2 (`jin`, `logits`) ·
`mynah_asr_detokenize` 1 (the whole transcript, re-decoded every chunk).

What was done: every one of those buffers is now carved at stream open —
`mynah_asr_ss_stream` grew its own `scr` (sized from `max_n_mel = 8*(right+1)+1`,
`sflat` aliasing `sb`), `mynah_asr_enc_stream` grew `ssilu` and `spost`,
`mynah_asr_stream` grew `dec_scr` and a `mynah_asr_detok`. The offline entry
points (`mynah_asr_silu`, `mynah_asr_encoder_post`, `mynah_asr_greedy_decode`,
`mynah_asr_subsampling_forward`) keep allocating: they call the same code with a
NULL scratch, so their arithmetic and their outputs are untouched.

Detokenisation is now incremental (`mynah_asr_detok_append`): the raw transcript
is kept UNSTRIPPED in a per-stream buffer and the ▁-expansion, the inline
`<xx-XX>` strip and the leading/trailing space strip are reproduced as a view
over it, so at every chunk the bytes equal `mynah_asr_detokenize` over the whole
history. Two cases needed care and both are covered by a model-free test
(`tests/test_tokenize.c` §6, every prefix × chunkings of 1..4 tokens):
a language tag spelled out across several tokens (the pass restarts at the first
`'<'` it could not resolve) and the double-space collapse after a tag stripped at
the very end of the buffer (undecidable until the next chunk arrives, so it is
deferred rather than skipped — this was a real divergence the test caught).
There is NO fallback to the whole-history path: no case needed one.

What remains: the transcript buffer (reserved 8192 B) and the token array
(4096 ids) still realloc geometrically if a stream outgrows them, as does the mel
stream's own sample buffer — amortised, not per chunk, and the measurement above
shows none of them fires on a 5 s clip. `mynah_asr_greedy_decode` still
dequantizes the head into a temporary when T > 16; no Nemotron preset reaches
that (the largest, [56,13], gives q = 14), but a model with a larger lookahead
would allocate 1 per chunk there. `MYNAH_ASR_QGEMM=1` (off by default) allocates
inside `mynah_asr_qmat_mul`. Linux is NOT measured here: the counter has an
`LD_PRELOAD`/`dlsym(RTLD_NEXT)` arm and the test skips (77) where interposition
does not work, but the numbers above are macOS/Accelerate only — S0-4 is the
Linux count.
