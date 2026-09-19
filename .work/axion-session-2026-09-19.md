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

## What is still owed here

- The step table (`a` and `b` of `T_step(B) = a + b·B`) on this machine, which
  is what turns capacity into a number.
- The OpenBLAS comparison arm (S1-6a) — needs the package.
- Pinned prefork: the mask read back from the kernel, which macOS cannot test.
- The ladder and the soak: what concurrency this box actually holds.
- `MYNAH_ASR_BATCH_F32=1` against OpenBLAS, which decides that default there.

Evidence: `~/mynah-asr-v2` on the box, `~/build.log` and `~/model.log`, tmux
sessions `asr` and `model`.
Conclusion: dispatch proven, kernels exact, batching identical. No speed claim.
Next action: the step table, then the ladder.
