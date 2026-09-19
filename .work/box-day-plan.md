# The box day — what to run on Linux, in order

Status: OPEN (written 2026-09-18 on the dev host; every step below is a command,
not a decision)

Task: S0-5, S1-6a, S1-8, S4-3, S5-1 (the Linux half of each)
Question: what does this server actually do on the production target, and which
of the claims made on macOS survive?

## Why everything waited

Every number in this repository so far is an Apple M1 development signal, and
several gates could not run here at all: the host has `dotprod` but **no i8mm**,
so the SMMLA kernel is compiled and never issued; it links Accelerate, so the
OpenBLAS-vs-own GEMM question is not even asked; it has no `sched_setaffinity`,
so prefork runs unpinned and "configured mask" versus "actual mask" is
vacuous; and the box was loaded for most of the day, which disqualifies timing
under ENGINEERING.md §10.

## Order of the day

**1. Bring the tree and the model.** On the box, from a clean clone of `main`:

```
git clone <repo> mynah-asr && cd mynah-asr && git log -1        # record the hash
make                                                            # BLAS default per platform
make check && make test                                         # model-free gates first
scripts/download_model.sh --model nemotron && \
  (cd tools && uv run python convert_nemo.py ../models/nemotron-3.5-asr-streaming-0.6b) && \
  ./mynah-asr quantize -m models/nemotron-3.5-asr-streaming-0.6b --quant int8
```

**2. Prove what the binary runs before measuring anything** (§5):

```
./mynah-asr --dispatch-map          # expect int8_rows = neon-smmla on Neoverse
./mynah-asr --flags
./tests/test_qmat                   # the line to look for is `neon-smmla exact OK`
```

A `SKIPPED` on `neon-smmla` here means the i8mm claim is still open and nothing
below may be attributed to it. Then the caps ladder, which is the cheap proof
that the fallbacks are real: `MYNAH_ASR_CAPS=sdot ./tests/test_qmat` and
`=scalar`, both green.

**3. The identity gates with the model** (correctness before capacity):

```
make test MODEL_DIR=models/nemotron-3.5-asr-streaming-0.6b
./tests/test_stream_batch models/nemotron-3.5-asr-streaming-0.6b     # both levels, B=1..8
make test-stream-allocs MODEL_DIR=models/nemotron-3.5-asr-streaming-0.6b   # S0-4, the Linux count
```

**4. The two questions macOS could not ask.**

- **S1-8, f32 row stability under OpenBLAS.** `MYNAH_ASR_BATCH_F32=1
  ./tests/test_stream_batch <model>`: if the stacked f32 step is byte-identical
  there too, the batched f32 default turns on for Linux; if it is not, it stays
  off and the note records the exact shapes that moved. int8 is exact either way
  and is the production path regardless.
- **S1-6a, the A/B that USED to own the Linux default — now a verification.**
  `BLAS=none` is the Linux default since 2026-09-19: the reason `own` had lost
  on the M1 was a defect in the DOT kernel (one FMA chain per output element,
  8.7 GF/s on a 100 GF/s core), it is register-tiled and byte-identical now, and
  at the 4-16 row shapes a streaming step issues it beats Accelerate
  (`.work/threadpool-and-lane.md`, 2026-09-19). What is owed here is the same
  comparison against OpenBLAS on Neoverse and on Zen, and it is two commands:

  ```
  make BLAS=openblas && ./mynah-asr --dispatch-map          # prove the arm
  MYNAH_ASR_GEMM_PROFILE=1 ./mynah-asr transcribe \
      -m models/nemotron-3.5-asr-streaming-0.6b tests/audio/test_it.wav 2> shapes.txt
  tests/bench_gemm_shapes shapes.txt
  ```

  `bench_gemm_shapes` times BOTH arms interleaved in one process and weights
  every shape by the call count the model really issued, so the footer line
  `[GEMM-BENCH-TOTAL]` is the whole answer: modelled GEMM milliseconds per run,
  ours against OpenBLAS's. If OpenBLAS wins by enough to cost capacity, flip the
  Makefile line back and write the number into the note. Then confirm with the
  end-to-end arm that matters — the same `tests/test_server_stream.sh` on each
  build — rather than trusting the model.
  Worth doing per family while the models are there: the profile of a Parakeet
  offline pass and of a Canary decode are different tables, and knowing them is
  how the next kernel is chosen rather than guessed.

- **S1-6a, the 2T-threads hazard.** Inside a prefork worker pinned to T cpus the
  pool builds T threads and OpenBLAS builds T more. Measure, do not guess:
  `OPENBLAS_NUM_THREADS=1` against the default, same topology, same bank, and
  read the context-switch rate beside the cadence numbers.

**5. The qualification itself — one command** (S4-3):

```
tools/bench/box_qualify.sh -m models/nemotron-3.5-asr-streaming-0.6b \
    -W 8 -T 4 -C 4 --wave "1 4 8 16 32" --soak 8 --soak-seconds 600
```

It refuses on a loaded box, on an UNKNOWN dispatch row and on workers that are
not pinned to disjoint slices, and it writes one evidence directory per run.
Sweep W×T around the answer the WAVE gives (`docs/serving.md` has the procedure:
find T\* where one worker stops improving, then sweep W at constant
subscription). **A wave may disqualify and never promotes; only the soak, on a
mixed-length bank, with its drift gate, promotes.**

**6. Multi-model, which is the reason the fleet exists** (S2-6's open half):

```
./mynah-asr-server --model streaming=models/nemotron-3.5-asr-streaming-0.6b:cpus=24:cap=4 \
                   --model offline=models/parakeet-tdt_ctc-110m:cpus=8:cap=2 --default streaming
```

Then the measurement the design actually promises and nobody has taken: saturate
one group and show the other group's cadence percentiles do not move.

## What a result must contain before it is quoted

WHAT CHANGED · WHAT PATH ACTUALLY RAN · WHAT WAS MEASURED · WHAT REMAINS
UNKNOWN · VERDICT (§14), with the evidence directory path. The provisional
envelope (`.work/serving-v2-design.md` §6) is a starting point, not a law: if the
box says the TTFP line or the lag line is wrong for this model, recalibrate it
there and say why.

Evidence: (pending — the box)
Conclusion: (pending)
Next action: step 1.
