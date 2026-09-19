# The Axion session — first real Linux run of serving v2

Status: OPEN (the identity and dispatch half is done and quoted below; the
capacity half — step table, ladder, soak — is still running)

Task: S0-5, S1-8, S5-1, and the groundwork for S4-3
Question: which of the claims made on the development Mac survive on the
production target, and what does this machine actually run?

## The machine

`gabrielemastrapasqua@35.239.91.20` — **Neoverse-V2, 32 cores, 1 socket, 62 GB
RAM, 80 MiB L3**, `aarch64`, tmux 3.6, gcc (no clang), no OpenBLAS installed.
Flags include **`i8mm`**, `bf16`, `sve`, `sve2`, `asimddp`. Disk is the one tight
resource: 4.2 GB free with the converted Nemotron pack (3.2 GB) already present.

Everything below is from a clean build of this tree on that box. Nothing here is
a timing claim: the box was not verified idle for these runs, and none of these
gates measure speed.

## 1. What the binary actually runs (ENGINEERING.md §5)

```
[DISPATCH-MAP] v=1 build=dev blas=own simd=neon+dotprod+i8mm
kernel.int8_dot      neon-sdot        smmla    -> neon-sdot
kernel.int8_rows     neon-smmla+sdot  smmla    -> neon-smmla
gemm.f32             own              -        -> own
gemm.f32_kernel      yes              yes      -> neon
threads.pool         yes              yes      -> 32
```

Two firsts in that block:

- **`kernel.int8_rows` resolves to `neon-smmla`.** The i8mm kernel has been
  compiled and unexecuted since it was written, because the development Mac has
  `dotprod` and no `i8mm`. This is the first machine that can issue it.
- **`gemm.f32 = own`.** The box has no OpenBLAS installed, so this is what a
  clean clone gets after yesterday's default flip — the first run of `BLAS=none`
  as the default on a production-shaped machine.

## 2. The kernels are exact (S5-1 closed)

`tests/test_qmat`:

```
  dot-per-row  exact OK  82 shape/T cases, 763 kernel blocks
  neon-smmla   exact OK  82 shape/T cases, 466 kernel blocks
  neon-sdot    exact OK  82 shape/T cases, 466 kernel blocks
  avx512vnni / avxvnni / avx2  SKIPPED (not compiled): not an x86 build
```

`exact` is `==` on the int32 accumulator AND on the f32 epilogue, both arms
forced in one process, 466 kernel blocks actually executed — not skipped, not
degraded to SDOT.

`tests/test_sgemm` and `tests/test_threads` pass here too, which puts the
thread-count determinism sweep, the DOT tiling identity, the row-stability gates
and the pool's first-dispatch deadlock gate on a **second ISA and a second
compiler** (gcc, not clang).

## 3. The batched step is byte-identical on Linux (S1-8, the int8 half)

`tests/test_stream_batch` on the converted Nemotron pack, B = 1, 2, 3, 4, 8:

```
  [int8] B=8: 17 steps, 416 stacked rows | dot 2509 dot_rows 3840 ... | IDENTICAL OK
  [int8] B=8: int8 kernels  neon-sdot 2509  neon-smmla 208896  | resolved neon-smmla
  [f32]  encoder bit-exact B=8: 16 steps, 266240 floats compared, 0 differ,
         0/24 caches differ | rel-pos shared 2496 private 0 group 384 | EXACT OK
  stream batch identity: IDENTICAL OK
```

Read those counters, not just the verdict: **208,896 SMMLA kernel blocks ran
during the identity check**, so this is not "identical because the fast path was
never taken". The rel-pos sharing counters show the shared projection happened
(2496 shared, 0 private at B=8).

The f32 arm is `own` here, so this is also the first Linux evidence for the f32
row-stability question S1-8 asks — for OUR sgemm. It is byte-exact, which is
what `sgemm.h` claims by construction (the stacked linear is the DOT family,
one dot per output element over the whole of k). **The OpenBLAS arm of that
question is still open** and needs `make BLAS=openblas` with the package
installed.

## 4. A finding the gate reported rather than hid

```
  [int8] es-ES: single stream != offline (pre-existing, not a batching effect;
                the batched gate below is unaffected)
    off: Blavos días, la reunión empieza a las vuelve en la sala grande.
    one: Blavos días, la reunión empieza a las muertes en la sala grande.
```

This is the es-ES clip already known to be unstable from the S1-6 work. It
matters twice: the transcript differs between the offline and the single-stream
path on this clip at int8, AND the test attributes it correctly instead of
letting it discredit the batching gate. It is worth its own item — an int8
instability on one clip is either a quantisation edge or a real defect, and
nobody has decided which.

## 5. The step table, and what it implies for C100

`MYNAH_ASR_STEP_TIME=1 ./tests/test_stream_batch`, int8, ONE process with 32
threads, unpinned, on a box that was NOT verified idle (my own builds were on
it). **DIAGNOSTIC**: it says the SHAPE of the answer, not the answer.

```
    B |  single ms | batched ms |  ratio | ms/stream
    1 |      87.32 |      88.11 |   1.01 | 88.11      <- carries the warm-up
    2 |     155.37 |      37.75 |   0.24 | 18.88
    4 |     281.12 |      52.70 |   0.19 | 13.17
    6 |     427.37 |      54.63 |   0.13 | 9.11
    8 |     557.88 |      69.27 |   0.12 | 8.66
```

Least squares over B = 2..8 (B=1 excluded: it is the first step and pays the
first touch of the weights):

```
    a = 31.7 ms      b = 4.58 ms per stream
    residuals: -3.1 +1.4 +2.7 +3.8 -4.6 -1.0 +0.9 ms
```

Those residuals are ±4.6 ms on a 55 ms step — about 8 %, which is what an
unquiet box looks like. Good enough to size an experiment, not to promote.

**b is 4.4x smaller than on the M1** (20.3 ms/stream there, 4.58 here). That is
the whole point of the week's kernel work arriving at once: SMMLA on the stacked
rows, 32 threads instead of 8, and our own sgemm on the f32 remainder.

Feeding the cadence law `T_step(B) = a + b·B <= rho·P` with `P = 320 ms`:

| rho | B_max in ONE 32-thread worker |
|---|---|
| 0.7 | 42 |
| 0.8 | **49** |
| 0.9 | 56 |

against `B_max ~ 10.7` from the M1 numbers. So **C100 is in reach but not free**:
one fat worker is worth about half of it, and whether the other half comes from
the prefork fleet (W narrow workers contending less) or does not come at all is
exactly what the W x T sweep is for. Nothing here is a fleet number: this is one
process with every core, which is not the serving topology.

One defect found by running it: the table printed "macOS dev signal" on a
Neoverse. It now reads the platform with `uname` and prints the thread count,
says DIAGNOSTIC in the line itself, and states that B=1 carries the warm-up — a
label that names the wrong machine is worse than none, because it travels into
a note as if it were provenance.

## What is still owed here

- The OpenBLAS comparison arm (S1-6a) — needs the package.
- Pinned prefork: the mask read back from the kernel, which macOS cannot test.
- The ladder and the soak: what concurrency this box actually holds.
- `MYNAH_ASR_BATCH_F32=1` against OpenBLAS, which decides that default there.

Evidence: `~/mynah-asr-v2` on the box, `~/build.log` and `~/model.log`, tmux
sessions `asr` and `model`.
Conclusion: dispatch proven, kernels exact, batching identical, and a diagnostic
step table that puts one 32-thread worker near 49 concurrent real-time streams.
Next action: the ladder on the real server, then a soak at the concurrency it
finds.
