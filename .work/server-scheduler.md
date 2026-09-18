# S2 — one scheduler per worker, slots, the sink

Status: OPEN

Task: S2-2
Question: replace "one blocking thread per connection" with one scheduler
thread that owns the model and drives B slots, with ingest threads that never
touch the model, and a sink vtable as the only boundary.

Known facts
- Sibling contract: `sink { next_job, on_done, cancelled, running }` plus a
  per-chunk callback; admission happens at the top of every step; a new job
  joins the running batch at the next step boundary; `cancelled()` is polled
  once per slot per step and wins over every other policy; the "only this
  thread enters the model" invariant is asserted in every callback.
- ASR slot state = one `mynah_asr_stream` (S1) + a bounded PCM ring filled by
  the ingest thread + the connection's `stream_out`. Offline REST files become
  slots fed from the body; offline-only models feed segments per step through
  `transcribe_batch`. One code path for WS and REST (repo rule 2).
- Step trigger: run a step when any slot has a full chunk ready; slots without
  a ready chunk skip the step (they are behind real time or idle). The first
  chunk of a new slot never waits for a tick.
- Each slot consumes ≤ 1 chunk per step: fairness by construction; a faster-
  than-real-time uploader fills its ring and is throttled by TCP.
- Lookahead is per slot; S1-4 decides whether ragged presets batch or run as
  separate groups per step (start with one group per preset).

Unknowns
- Whether the RNNT greedy loops (per slot, serial) dominate the step at B ≥ 8;
  if so the decoder lane (thread-local redirect, pinned tail cpus) is the
  measured next step, never a second submitter on the same pool.
- The right size of the per-slot PCM ring (start: 30 s of audio, refuse beyond).

Plan
1. `server/sched.c`: slot table (cap from the CLI), `step()` as described,
   sink implementation over `stream_out`; `synth_assert_scheduler` analogue.
2. `server/ingest.c`: WS frame parser and REST body reader, pushing into the
   slot ring; control messages (`finalize`, `reset`) as slot flags read at the
   step boundary.
3. Delete the REST micro-batcher and the `g_inflight` BLAS knob.
4. Acceptance: `grep -n pthread_mutex_lock server/*.c` shows only the
   connection queue, the slot rings and the writer; `make test-server` and
   `test-server-concurrency` green; the byte-identity gate of S2-7.

Evidence / Conclusion / Next action: after S1-1..S1-4 and S2-1.
