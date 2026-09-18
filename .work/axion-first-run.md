# Campaign — first build and baseline on the 32-core Neoverse-V2 box

Status: IN PROGRESS (started 2026-09-18; paused — box shared with mynah-tts soaks, resume only on the owner's go)

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
- **DIAGNOSTIC, NOT QUALIFYING** — offline `bench` on `samples/en/fleurs_long.wav`
  (94.6 s), taken while a mynah-tts soak was running on the same host (loadavg
  20–28 at start; the script printed it and did not refuse, which is the
  contradiction rule 7 forbids). Kept only as an order of magnitude:

  | T (pinned) | f32 RTF | int8 RTF |
  |---|---|---|
  | 1 | 0.238 | 0.226 |
  | 2 | — | 0.119 |
  | 4 | 0.127 | 0.066 |
  | 8 | — | 0.038 |
  | 16 | 0.033 | 0.025 |
  | 32 | 0.018 | 0.018 |
  | unpinned default | 0.018 | 0.020 |

  Two readings that survive the contamination: on long offline audio (big
  GEMMs) OpenBLAS at 32 threads is fine and int8 barely helps at high T, so
  the 4x pathology of the 4.3 s clip is specific to short/streaming shapes,
  exactly the shapes a streaming server runs; and int8 scales near-linearly to
  8 threads then flattens, consistent with narrow workers. Peak RAM 2.8 GB f32
  / 1.2 GB int8.
- Process lesson recorded: both phases were started with another load on the
  box. Every box script now begins with a `loadavg` gate that refuses above 2.0
  (as the sibling `bench-suite` does), and nothing runs on a shared box without
  the owner's explicit go.
Conclusion: build and gates hold on Linux ARM; the model is in place. The one
measurement still owed is the pinned single-stream step cost at Q=4 int8,
T=1..4, on an idle box.
Next action: paused for the implementation; when the box is free the run is ONE
command, `tools/bench/box_qualify.sh` (S4-3):

```
tools/bench/box_qualify.sh -m models/nemotron-3.5-asr-streaming-0.6b \
    -W 8 -T 4 -C 4 --wave "1 4 8 16 32" --soak 8 --soak-seconds 600
```

It refuses rather than produce a number it cannot support: a `loadavg` at or above
2.0 (another load owns the box), a dispatch row that resolved UNKNOWN, prefork
workers that are not pinned to disjoint slices on Linux. A dirty tree does not
stop it but labels the whole run NON-QUALIFYING, and `--allow-load` does the same
for a deliberately diagnostic run on a busy host. It writes one directory per run
holding the manifest (commit, binary sha256, masks, topology, loadavg), the
banner, the dispatch map, the per-concurrency WAVE JSONs, the SOAK JSON with its
drift windows, and the server's own `/v1/health` and `/metrics` at the end.
Smoke-tested on the dev host end to end (the load gate, the dirty-tree label, the
WAVE phase and the teardown all behaved); the mask assertion is Linux-only and has
therefore never fired here.
