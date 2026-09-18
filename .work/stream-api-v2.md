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
