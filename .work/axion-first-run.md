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

Evidence (2026-09-18, tree 5f0f802 via `git archive`, gcc 15.2, `-O3 -march=native
-ffast-math`, OpenBLAS 0.3.32 pthread, box idle, `loadavg` checked)
- `make`: 1.5 s wall (-j16). `make test`: every model-gated test skipped with 77 and
  the reason printed; model-free tests pass. Nothing failed.
- Nemotron download 2.55 GB in 10m42s; convert 0.5 s (numpy, uv-managed Python
  3.12 because 3.14 lacks wheels); `quantize --quant int8` 0.6 s → 0.79 GB.
- Single stream, `tests/audio/test_en.wav` (4.3 s), transcript identical to the
  laptop in all four runs:

  | config | inference | RTF |
  |---|---|---|
  | f32, unpinned, OpenBLAS default = 32 threads (v1 out of the box) | 6.30 s cold / 9.02 s warm | **1.45 / 2.08** |
  | f32, `taskset 0-3`, OPENBLAS_NUM_THREADS=MYNAH_ASR_THREADS=4 | 2.23 s | 0.514 |
  | int8, same 4-thread pin | 0.87 s | **0.201** |

  Reading: on this box the v1 defaults are **slower than real time** for a
  single stream, and pinning four threads alone is 4x faster. That is
  OpenBLAS's pool thrashing on Q=4-row GEMMs and it is the strongest single
  argument for S1-6 (BLAS leaves the worker) measured so far. int8 at four
  threads is 2.6x faster than f32 at the same pin; the per-core speed is well
  below the M1 (f32 RTF 0.055 there), as expected for a server core.
Conclusion: build and gates hold on Linux ARM; the model is in place; the
thread matrix and the server baseline are running as phase 2 (see
`baseline-streaming-concurrency.md` for the S0-3 table when it lands).
Next action: S0-3.
