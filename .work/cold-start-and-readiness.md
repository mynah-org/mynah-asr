# Cold start and readiness

Task: S11-1

F30 attributed the Linux cold start causally: 12 fresh processes, page cache
dropped, first step **3260-3270 ms with 6555-6624 major faults** (6 of 6); warm,
**36 ms with 0-2 major faults** (5 of 5); the second step 12.6-12.8 ms with 0
major faults (12 of 12). A warm-cache layer of 17-21k MINOR faults sits on top.
That is a fact about the Axion, on Linux. It was deliberately left unfixed.

This note is the audit of what the FIRST REQUEST still pays that the server
could have paid at startup, and of the lifecycle hole that makes the question
matter. Source only, 2026-09-22, nothing implemented.

## What initialisation is, by owner

| action | where | when it happens today | could move to startup |
|---|---|---|---|
| model file I/O | `third_party/ingot/src/safetensors.c:446` | `mmap(MAP_PRIVATE)` at load, no `MAP_POPULATE` | n/a, already at load |
| **page population** | the same mapping | **on first touch, i.e. inside the first request** | yes — this is F30's 3.26 s |
| weight packing | `src/qmat.c:121` | none: a pre-quantized pack is **zero-copy from the mmap** | n/a — nothing to pack, but also nothing touched |
| our initialisation | `src/mynah_asr.c:374-378` | the rel-pos table is built at LOAD, explicitly so the prefork parent shares the pages | already moved, deliberately |
| **thread-pool creation** | `src/threads.c` `pool_init` under `pthread_once` | **on the first `parallel_for`, i.e. inside the first request** | yes |
| per-stream allocation | `src/mynah_asr.c:854` | at stream open, carved once (S1-3) | no — it belongs to the stream |
| BLAS runtime | `blas=own` on Linux | no vendor library to initialise | n/a |

So exactly two predictable costs are still charged to whoever arrives first:
**the mapped weight pages** and **the worker threads**. Everything else is
already paid at load or genuinely belongs to the request.

## The lifecycle hole

**FACT.** There is no readiness endpoint. `/v1/health` (`server/obs.c:201`)
reports model, process, kernels and refusals; nothing in it says "this process
has done its first forward". The server accepts work as soon as it is listening,
so the first real stream pays the 3.26 s and the client sees it as latency.

The deployment invariant that would fix it is not "warm up faster", it is:
**readiness must be false until the predictable first-request work is done.**
With three prefork workers behind one listener, per-worker readiness matters:
a worker that has not faulted its pages in is a 3-second stall waiting for
whichever connection lands on it.

## What is NOT established

- Nothing here is measured on this host: `models/*` are symlinks into an
  external volume that is not mounted, so no cold start was reproduced today.
- macOS will not reproduce the Linux number anyway. F30's 3.26 s and its major
  fault counts are Axion facts and must not be restated as portable ones.
- Whether touching the mapping at startup actually removes the first-request
  cost, rather than moving it, is a MEASUREMENT: on a box with the page cache
  dropped, first-step wall and `ru_majflt` before and after. `MADV_WILLNEED` is
  a hint, not a guarantee, and the prefork parent touching pages before fork is
  a different experiment from each worker touching its own.

## Order, when it is picked up

1. Add readiness as lifecycle correctness — it is testable without any of the
   above being true, and it is what makes a warm-up observable rather than
   assumed.
2. Only then measure whether a startup touch pays, on Linux, against dropped
   caches, reporting wall AND major faults per F30's method.

Not started. Do not implement the warm-up before the measurement in 2.
