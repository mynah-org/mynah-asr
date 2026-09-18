# S0 — baseline: the current server under N real-time streams

Status: OPEN

Task: S0-3
Question: what does the v1 server (blocking pool, one worker per stream, BLAS
threads = ncpu/inflight) actually do under 1, 2, 4, 8, 16 concurrent
real-time Nemotron streams on the 32-core Neoverse-V2 box, and what are `a`
and `b` in `T_step(B) = a + b·B` for the stream step today?

Known facts
- Single stream, Apple Silicon: ~26 ms compute per 80 ms chunk f32, ~9 ms int4
  (docs/benchmarks.md). No Linux ARM streaming number exists. No N-stream
  number exists anywhere.
- Server ceiling by construction: `--threads` streams (default 4); the 5th
  waits in a 128-fd ring with no deadline.
- `mynah_asr_parallel_for` runs inline and serial when two requests overlap
  (src/threads.c); BLAS budget is process-global and coarse (per WS frame).
- ~55–60 malloc/free per chunk around the encoder step (subsampling ×6,
  encoder_post ×3, decode ×2, SiLU ×48 on Accelerate only, detokenize ×2).

Unknowns
- Chunk service time per preset (0/1/3/6/13) f32 vs int8, 1 thread vs T threads.
- How service time and emission lag degrade with N when N ≤ threads and when
  N > threads (queueing in the ring).
- Arrival-phase distribution of chunks across streams (is within-worker
  batching at the chunk grid real?).
- Allocation count per chunk on Linux/glibc (Accelerate SiLU path is macOS only).

Files/functions to inspect: server/main.c handle_ws, src/mynah_asr.c
stream_feed/stream_flush_chunk, src/encoder.c mynah_asr_enc_stream_step,
src/threads.c.

Plan
1. Build at the pinned HEAD on the box, `make test` (77 where no model), record
   commit, flags, `lscpu`, openblas version.
2. `mynah-asr stream` on a 60 s fixture per preset, f32 and int8, T=1/2/4/8:
   chunk service time histogram (add a `--chunk-stats` stderr line if missing:
   that is a diagnostic print, not a behaviour change).
3. S0-2 tool: N stdlib-Python WebSocket clients, each pacing a FLEURS clip at
   1x in 100 ms frames, recording per-delta client lag and TTFP; run N ∈
   {1,2,4,8,16} against `--threads 4` and `--threads 16`.
4. LD_PRELOAD malloc counter over 100 chunks: count per chunk, mmap count.
5. Write the numbers here with the manifest (commit, binary sha256, masks,
   env); derive `a`, `b` by OLS over B, label everything MEASURED.

Acceptance gate: this note holds a table of TTFP / emission lag p50,p95 /
backlog max / STREAM_RTF for each N and thread setting, the `a + b·B` fit with
R², the allocation count, and a one-paragraph reading of where the time goes.
Only then does S1 start.

Evidence: (none yet)
Conclusion: (pending)
Next action: S0-1 (box build) then S0-2 (tool).
