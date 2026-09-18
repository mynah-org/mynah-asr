# Campaign — first build and baseline on the 32-core Neoverse-V2 box

Status: IN PROGRESS (started 2026-09-18)

Task: S0-1
Question: does the tree build and pass its gates on the production-class ARM
host, and what does one Nemotron stream cost there per chunk, f32 and int8?

Host (public-safe): GCP Axion C4A class, 32× Neoverse-V2, SMT off, 62 GB, ~7.6
GB free disk at start, gcc 15.2, OpenBLAS 0.3.32 pthread, Python 3.14 stdlib
(no venv). Flags of interest present and idle: i8mm, bf16, sve, sve2, svei8mm,
svebf16. Shared with a mynah-tts soak until it ends; nothing is measured while
another load runs (`loadavg` gate).

Plan
1. Clone at the pinned commit; `make` (Linux → OpenBLAS); `make test` (77 where
   the model is missing); record `make`'s effective CFLAGS.
2. Download `nvidia/nemotron-3.5-asr-streaming-0.6b` with the repo script;
   convert; `mynah-asr quantize --quant int8`.
3. Single stream, 60 s fixture, presets 3 and 0, f32 and int8,
   `MYNAH_ASR_THREADS` = 1, 2, 4, 8: RTF and chunk service time.
4. Hand the numbers to `baseline-streaming-concurrency.md`.

Evidence: (pending)
Conclusion: (pending)
Next action: wait for the box to be idle, then step 1.
