# S2 — several models in one fleet

Status: LANDED on the M1 dev host (2026-09-18); no Linux/Axion run yet.

Task: S2-6
Question: serve Nemotron (streaming, 40 languages), Parakeet TDT (offline, 25
languages) and Canary (offline, translation) from one `mynah-asr-server`
invocation, with routing by model and language per request.

Known facts
- One process holds one model: a batch is one model by the shape of the
  address space (sibling decision B; the in-process registry was costed and
  rejected there because switching weights per request costs a whole pass).
- Unlike the TTS siblings, **language is per request here**: Nemotron's prompt
  is a per-row post-encoder one-hot, Canary's is a per-item prompt. A batch
  may mix languages. Routing is by model only; `lang` travels with the request.
- `--lid-model` already exists: a detector model in front of models that
  cannot detect. It stays a resident model in the parent, not a group; nothing
  routes to it.
- Router classification is a bounded non-consuming `MSG_PEEK` ≤ 8 KiB for the
  request line (query `model=`) and, for multipart REST, the `model` field if
  it is inside the peeked prefix; otherwise the default group. Ambiguity is
  refused, never resolved.

The key insight, verified before building on it
- `server/prefork.{c,h}`, lifted from mynah-tts, ALREADY carried the whole
  group machinery, because there a group was a LANGUAGE (one pack per worker,
  a batch can never mix languages): `languages[]`/`language_count` in the
  config, contiguous worker groups, `mynah_asr_prefork_worker_language()`,
  per-group rungs 1–3, the non-consuming `MSG_PEEK` classifier, the
  `language_not_served` rung-0 refusal.
- So S2-6 is mostly RENAMING the concept and feeding it from a new CLI, plus
  two real changes: the classifier looks in a different place (the request
  line's `model=` query and a multipart `model` part, not a JSON `"language"`
  key), and the per-group knobs (`workers`, `cpus`, `cap`) had to become real
  arrays instead of an even split.

Files/functions inspected
- `server/prefork.h` (contract), `server/prefork.c`
  (`mynah_asr_prefork_run` group planning, `classify_language`, `group_pick` /
  `group_live` / `group_queued` / `router_admit` / `router_drain`, the
  `REFUSAL[]` table, `pf_render_metrics`), `server/main.c`
  (`read_model_meta`, `ws_parse_query`, `handle_transcribe`, `parse_multipart`,
  `handle_conn`, `main`), `server/obs.{c,h}`,
  `tests/test_server_{stream,concurrency,protocol,metrics}.sh`,
  `.work/serving-v2-design.md` §3 §4.

CLI shape (implemented)
```
mynah-asr-server \
  --model nemotron=models/nemotron-3.5-asr-streaming-0.6b:workers=6:cpus=24:cap=8:quant=int8 \
  --model parakeet=models/parakeet-tdt-0.6b-v3:workers=2:cpus=8:cap=4:quant=int8 \
  --default nemotron -p 8090
```
`-m dir` is one group named by its own `mynah.json`. Two or more `--model`
imply prefork: with no `--prefork` the group count becomes it and the banner
says so; with a `--prefork` that does not fit the groups the server refuses to
start rather than dropping a model. Each group's workers open ONLY their own
model — the parent opens them all before the fork (COW-shared weights, and a
bad path is a start-up failure), the child closes every group but its own.

Acceptance gate (stated before the work)
One server, two groups, `--prefork 2`: `/v1/models` lists exactly the two
names; a REST transcription to each group returns that model's own transcript,
byte-identical to the CLI for that model; a WebSocket to the streaming group
streams and is byte-identical; a WebSocket to the offline group is 400
`model_not_streaming`; an unknown name is 404 `model_not_found` with a readable
body; filling one group's slots refuses `server_at_capacity` while the OTHER
group still serves; SIGTERM leaves no worker.

## Evidence

Host: M1 (Apple Silicon, 8 cpus, UNPINNED — macOS has no affinity API this
server uses, and the banner says so). Build `v0.9.1-53-gc414388` + this change,
`blas=accelerate simd=neon+dotprod int8_kernel=neon-sdot`. Models:
`nemotron-3.5-asr-streaming-0.6b` (int8) and `parakeet-tdt_ctc-110m-gguf`.
Clip: `tests/audio/test_en.wav`.

**The identity gate is only meaningful because the two packs disagree**, and
the test refuses to run when they do not (ENGINEERING.md §7):

```
nemotron: "Hello, this is a speech recognition test.  The weather is nice today."
tiny:     "Hello, this is a speech recognition test. The weather is nice today."
                                                   ^ one space, not two
```

`sh tests/test_server_models.sh` — one server, `--model streaming=<nemotron>`
+ `--model offline=<110m gguf>`, `--prefork 2`, exit 0:

```
    /v1/models: offline, streaming (default=streaming)
server-models models-endpoint OK
server-models rest-streaming-group OK (byte-identical to its own CLI)
server-models rest-offline-group OK (multipart field, byte-identical)
server-models rest-default-group OK (no model named -> --default)
server-models ws-streaming-group OK (byte-identical to the CLI's stream)
server-models ws-offline-group-refused OK (400 model_not_streaming)
server-models ws-unknown-model OK (404 model_not_found, accepted set named, no Retry-After)
server-models rest-unknown-model OK (404 model_not_found)
server-models per-group-capacity OK (offline 1 served / 2 refused with a readable 503, streaming group served in the same moment)
server-models shutdown OK (survivors=0)
```

Three consecutive runs of that gate, all exit 0 with the same capacity line.
The first shape of the capacity phase held a slot and raced a probe against it,
and on this host the 110m finished first often enough to report INCONCLUSIVE:
an assertion that only sometimes fires is an assertion that will one day pass
while broken (ENGINEERING.md §7). It now fires THREE concurrent requests at a
group with one slot and the queue disabled, so "one served, the rest refused"
is arithmetic rather than timing.

The fleet's plan as the server itself prints it, which is the evidence that the
topology is the one that was asked for rather than one inferred from a flag:

```
prefork: 2 workers x 4 threads over 8 allowed cpus (4 per worker, logical slices), 2 slots each
prefork: models      2 resident, one model per worker (a batch is one process's slots,
                     so a batch is one model; `lang` stays a per-request parameter)
prefork:   nemotron                 workers 0-0 (1) · 4 cpus · 2 slots each (2) · rung2 queue per worker
prefork:   tiny                     workers 1-1 (1) · 4 cpus · 2 slots each (2) · rung2 queue per worker
prefork:   default    nemotron (a request naming no model)
[SERVER-CONFIG] v=1 ... model=parakeet-tdt_ctc-110m group=tiny engine=parakeet-tdt quant=f32 streaming=no ... worker=1
[SERVER-CONFIG] v=1 ... model=nemotron-3.5-asr-streaming-0.6b group=nemotron engine=nemotron-streaming quant=int8 streaming=yes ... worker=0
[SERVER-CONFIG] v=1 prefork_workers=2 prefork_threads=4 group_plan=nemotron=1,tiny=1
```

`/v1/health` on the fleet (the worker the router happened to pick):
`{"group":"tiny","groups":"nemotron=1 tiny=1", "model":{"name":"parakeet-tdt_ctc-110m",…}}`.

No regression in the suites that already existed, each run to completion with
its real exit code: `tests/test_server_stream.sh` (4-stream identity, stalled
reader, `--prefork 2` identity, 8-stream batched identity, batched path proven),
`tests/test_server_protocol.sh`, `tests/test_server_concurrency.sh`,
`tests/test_server_metrics.sh`. `make check` passes.

**`-fsanitize=undefined` pass of the server suites** (Makefile ubsan flags,
`-O2 -g -fsanitize=undefined`): **zero runtime errors reported, anywhere.**
`server-models`, `server-concurrency`, `server-metrics` and `server-protocol`
all exit 0. `server-stream` exits 1, and the reason is capacity, not
correctness: the sanitized build costs `step_wall_ms mean 935.9` against
`124.8` on the -O3 build (7.5x), so on this 8-cpu laptop the 8-stream phase
cannot hold a 320 ms chunk period — TTFP p95 4936 ms, emission lag p95
11916 ms, two clients timing out at connect. Everything that did complete was
still byte-identical (the harness reports no reference or cross-stream
mismatch). That phase also runs `-m MODEL --cap 8` with NO `--prefork`, i.e.
one process and one group, where `groups <= 1` and the classifier and the model
router never execute at all — so it exercises none of this item's code.

## Conclusion

PROMOTE the mechanism, on this host. Routing by model works on both transports
and both addressing shapes, a request provably reaches the group's own weights
(byte-identity against that model's CLI), capacity is per group, and the three
refusals carry the right status, the right code and — for `model_not_found` —
no `Retry-After`.

What the gate does NOT establish, stated rather than omitted:
- **No Linux run and no pinned run.** macOS has no affinity API this server
  uses, so `:cpus=` was exercised as a *plan* (the slices are computed and
  printed) and not as isolation. The three-groups-on-the-Axion part of the
  original gate is still open.
- **Canary / AED was not exercised**: no Canary pack on this host, so
  `/v1/audio/translations` from a group of its own and `--lid-model` in a
  multi-group fleet are untested paths, not broken ones.
- **The classifier's residual is real**: a multipart `model` field that lands
  after the audio is past the 8 KiB peek and goes to the default group. The
  worker then refuses it (404 with a message saying where to put the name)
  rather than answering from the wrong weights, so the failure mode is a wrong
  refusal. Not measured under load.
- Rung 4 (the per-request service cap) is still observed and not enforced at a
  frame boundary — unchanged by this item, see S2-3.

## Next action

- Run the three-group gate on the Axion with pinning on, `:cpus=` carving real
  slices, and a Canary group for `/v1/audio/translations` (folds into S4).
- `tools/bench/stream_load.py` now takes `--model`; a multi-group SOAK that
  shows one group's saturation not moving the other group's cadence
  percentiles is the measurement this design actually promises and does not
  yet have.
