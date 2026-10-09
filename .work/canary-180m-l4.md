# Canary 180M Flash on L4

Track: S15 (`.work/lightweight-asr-plan.md`). Model: `nvidia/canary-180m-flash` (`.nemo`
only, CC-BY-4.0). Baseline for English: `nemotron-3.5-asr-streaming-0.6b`.
Branch audited: `research/lightweight-asr` at `4378448` (= `asr-cuda-perf`, PR #4).
Every claim cites file:line in this tree; UNKNOWN means nothing in the repo establishes it.

## 1. Audit of current support (2026-10-09)

### 1.1 Variants implemented

All three go through the same AED engine (`engine: canary-aed`, `arch: fastconformer_aed`,
`tools/convert_nemo.py:456-497`):

| pack dir | HF repo / file | encoder / decoder | ASR langs | translation | timestamps | source |
|---|---|---|---|---|---|---|
| `canary-180m-flash` | `nvidia/canary-180m-flash` / `canary-180m-flash.nemo` | FastConformer 17L x 512 x 8h, non-causal, batch_norm, biases / Transformer 4L x 1024, vocab 5248 (aggregate) | en, de, es, fr | en <-> de/es/fr; **fr->en emits EOS immediately** | generative `<\|N\|>` tokens (80 ms), "experimental" | `docs/canary-arch.md:13-23`, `docs/canary-usage.md:7-11`, `:30-31` |
| `canary-1b-flash` | `nvidia/canary-1b-flash` | 32L x 1024 / 4L x 1024 | en, de, es, fr | en <-> de/es/fr | generative tokens | `docs/canary-arch.md:25-26`, `docs/models.md:138` |
| `canary-1b-v2` | `nvidia/canary-1b-v2` | 32L x 1024 / 8L | 25 EU (`CANARY_V2_LANGS`, `tools/convert_nemo.py:451-453`) | en <-> 24 | NOT generative (`timestamp_tokens: false`, `tools/convert_nemo.py:596-599`); bundled CTC aligner -> `<dir>/aligner/` (`:633-640`, `docs/canary-arch.md:114-167`) | `docs/models.md:139` |

The model files on the dev machine are symlinks to an unmounted share
(`models/canary-* -> /Volumes/shared/...`), so the converted 180m `mynah.json` could not be
re-read today; the numbers above are from `docs/canary-arch.md` (read from the `.nemo`
2026-07-18).

Capabilities in Mynah:
- **PnC / ITN / task tokens**: the `canary2` prompt template and its defaults come from the
  yaml (`tools/convert_nemo.py:486-495`): pnc `<|pnc|>`, itn `<|noitn|>`, timestamp
  `<|notimestamp|>`, diarize `<|nodiarize|>` (`docs/canary-arch.md:88-103`). So the output is
  punctuated and capitalised; the flash models spell numbers out, v2 applies ITN
  (`docs/canary-usage.md:68-70`). No CLI/server switch for pnc/itn was found (grep `pnc` in
  `cli/`, `server/` is empty): changing them means editing `prompt.defaults` in `mynah.json`.
- **Translation**: target != source (`aed_build_prompt`, `src/mynah_asr.c:1431-1500`);
  CLI `--target-lang`, API `mynah_asr_set_target_lang` or `lang "src>tgt"`, server
  `POST /v1/audio/translations` (`docs/canary-usage.md:13-22`, `server/main.c:4`, `:1241-1243`).
- **Language ID**: none. `--lang auto` falls back to `en` with a stderr note
  (`docs/canary-usage.md:24-28`). Detection only by pairing a Nemotron `--lid-model`
  (CLI and CPU server, `docs/canary-usage.md:30-60`).
- **Timestamps**: flash = generative `<|N|>` tokens (`src/mynah_asr.c:1540`, `:1677-1680`);
  asking for them changes the prompt and therefore the text, so a batch never returns AED words
  and a words request takes the single path (`src/mynah_asr.c:530-536`, `:562-573`).
  180m: "experimental"; no accuracy measurement in the repo (UNKNOWN).

### 1.2 Conversion

- `.nemo` route only (the HF-native port of 1b-flash is encoder-only, `docs/canary-arch.md:9-11`).
  Dispatch on `target` ending in `EncDecMultiTaskModel` (`tools/convert_nemo.py:593-595`);
  asserts `prompt_format == canary2`, pre-LN, relu (`:465-466`); renames `_AED_RENAMES`
  (`:244-265`); tokenizer from the SPE models inside the tar, aggregate for flash (`:500-525`);
  checks every prompt token exists (`:617-622`).
- Needs `sentencepiece` (base deps) + torch (`--extra oracle`) (`tools/pyproject.toml`).
- Layout: `models/canary-180m-flash/canary-180m-flash.nemo` -> converted in place.
  `scripts/download_model.sh --model canary-180m` fetches exactly that (`scripts/download_model.sh:27`,
  `:89-91`, `:122`).

### 1.3 "Streaming" for Canary: none, by design

- PLAN M-5 (`PLAN.md:126`): "out of v2 by design — its AED decoder keeps no state between calls,
  so it re-decodes the window on top of the 7x encoder".
- The pack has no `streaming` section (`build_canary_from_yaml` writes none,
  `tools/convert_nemo.py:467-497`), so `n_lookaheads == 0`: `mynah_asr_stream_open` refuses
  ("offline-only", `src/mynah_asr.c:833-836`). The encoder would also fail every streaming gate
  (biases, folded batch_norm, symmetric padding, per-feature mel: `src/mynah_asr.c:805-830`).
- CPU server: WebSocket refused before the upgrade with `400 model_not_streaming`
  (`server/main.c:859-868`); REST only (`/v1/audio/transcriptions`, `/v1/audio/translations`).
  Multi-model fleets: "no Canary/AED group" was ever tested (`PLAN.md:66`).
- `docs/models.md:132`: upstream offers only chunked offline inference
  (`speech_to_text_aed_chunked_infer.py`), not implemented here. Any upstream streaming mode for
  Canary 180M (e.g. wait-k / chunked AED) is UNKNOWN in this repo -> `.work/lightweight-asr-upstream.md`.

### 1.4 CUDA

- **The CUDA server cannot load Canary.** `asr_pack_open` refuses any decoder other than
  `rnnt_lstm` ("the GPU engine serves the RNNT greedy decoder only", `gpu/pack.c:65-68`), would
  then refuse batch_norm, biases, symmetric padding, per-feature mel and the missing streaming
  presets (`gpu/pack.c:91`, `:96-111`). It also has no REST: `POST /v1/audio/transcriptions`
  returns 501 (`gpu/server/main.c:1301-1303`). None of PR #4 (bf16 own-tc, stage-ahead, graphs,
  warm-up, host threads, quantised arms) applies to Canary.
- The only GPU route today is the LIBRARY backend: `make cuda` builds the CLI and CPU server with
  `-DMYNAH_ASR_CUDA` and offloads big GEMMs (T >= 24) to cuBLAS; the AED decode stays on CPU
  (`Makefile:326-337`, `docs/backends.md:64-74`, `docs/benchmarks.md:61-66`). Selected with
  `--backend cuda` (`docs/backends.md:3`). Note `make cuda` starts with `make clean`
  (`Makefile:331`), so build it in a separate checkout from the `make lib` + `make -C gpu` tree.

### 1.5 CPU server batching for Canary

- Offline requests go through the scheduler, which drains up to `--batch` (default 8, max 64)
  queued jobs into one `mynah_asr_transcribe_batch_ts` call (`server/sched.c:910-995`,
  `server/main.c:66`, `:493-498`). Inside: per-item features and decode in a `parallel_for`, a
  batched encoder; the AED greedy decode is per item, not batched across items
  (`src/mynah_asr.c:497-540`). Word timestamps force the single path (`:562-573`).
- Long audio is segmented on silence (planner, `src/mynah_asr.c:476-490`).
- Default server quant int8 (`server/main.c:85`); int8 halves the AED decode
  (`docs/benchmarks.md:53-56`).
- Concurrency / throughput of Canary on the CPU server: UNKNOWN (no REST ladder recorded).

### 1.6 Tests

- `tests/test_e2e.sh:61-200` (Canary ASR + translation matrix, `--lid-model`, v2 aligner,
  flash timestamps) and `tests/test_server.sh` run only when a Canary pack is on disk
  (`MODEL_DIR=...`); CI does not download Canary (CI models: the 110m and the EOU 120M,
  `.github/workflows/ci.yml:246-283`).

### 1.7 Evaluation and load tooling (Canary-relevant)

- Scorer: `tools/bench/streaming_metrics.py` `normalise` (NFKC, lowercase, strip `<..>` and all
  Unicode punctuation, `:249-267`), `wer`/`cer`/`align_counts` S/D/I (`:312-328`, `:468-516`),
  `wer_format_free` (number verbalisation, **EN and FR only**, `:331-466`). Canary flash spells
  numbers out, which this format-free variant forgives for EN/FR; for DE/ES only the strict WER is
  meaningful. Canary's PnC is stripped by the normaliser.
- Offline WER: `tools/eval/lang_gate.py --mode offline` (CLI `transcribe --lang <l>`,
  `tools/eval/lang_gate.py:70-79`, `:200-213`; per-language WER/CER + bootstrap CI, S/D/I),
  `tools/eval/offline_quality.py` (498 bank from a `bank.txt`, parallel CLI), `tools/eval/cer_offline.py`.
- Banks: `samples/eval-bank` = FLEURS **test**, 200 EN + 200 FR (`tools/fetch_eval_bank.py:43`:
  `LANGS = {"en": "en_us", "fr": "fr_fr"}`; `--langs de,es` is rejected by that table, so DE/ES
  need a two-entry extension `de_de`, `es_419` — a code change, not done). Committed DE/ES audio
  today: 2 FLEURS dev sentences each (`tools/fetch_fleurs_samples.py:28-33`) — diagnostic only.
  `samples/stress-en` (498-clip bank, 349 originals) as in the EOU note; note its long class
  (up to 42 s) exceeds Canary's ~30 s comfort zone and goes through the segmenter.
- Load: `tools/bench/rest_load.py --port P --clips ... --ladder 1,2,4,8,16,32 --lang en --json`
  (`tools/bench/rest_load.py:1-27`): xRT and per-request latency per rung, 503 counted as
  refusal. Streaming tools (`stream_load.py`, `gpu_qualify.sh`, `gpu_knee.sh`, `v2_verdict.py`)
  do not apply.

## 2. Prior evidence in this repo

| measurement | value | source |
|---|---|---|
| RTF f32, 4-5 s fixtures, M-series CPU | 0.127, RAM 0.71 GB | `docs/benchmarks.md:17` |
| RTF ~65 s, M-series: f32 CPU / Metal / int8 CPU | 0.060 / 0.054 / **0.030** | `docs/benchmarks.md:31` |
| RTF A100 + EPYC 22 vCPU, library `make cuda`: 5 s cpu/cuda, 60 s, 300 s | 0.060/0.051, 0.046/0.044, 0.045/0.043 (GPU offload barely helps: decode-bound) | `docs/benchmarks.md:61-86` |
| int8 checkpoint size | 0.22 GB | `docs/benchmarks.md:57` |
| CPU vs CUDA (library) transcripts | byte-identical on every model (2026-07-20) | `docs/benchmarks.md:63-66` |
| fr->en on the 180m | EOS immediately (also in the oracle) | `docs/canary-usage.md:30-31` |

- WER/CER on any bank for Canary 180M: **none** in the repo. EN/FR quality notes
  (`.work/french-validation-20260925.md`) are Nemotron-only. Server load numbers: none.
  GPU-server numbers: impossible today (1.4).

## 3. Open questions to measure

1. WER/CER/S/D/I EN (same audio as the EOU/Nemotron EN rows: `samples/eval-bank/en`) and
   DE/ES/FR on FLEURS test (needs the `fetch_eval_bank.py` DE/ES extension), same normaliser,
   int8 and f32, greedy.
2. Streaming verdict from primary sources (upstream chunked/wait-k AED for 180M?) vs the code
   facts above; no fake streaming mode.
3. Offline speed in the semantics it has: RTFx per clip length on L4 (CPU library build vs
   `make cuda` library build), batch scaling via the CPU server `--batch`, REST ladder
   (`rest_load.py`), C1 latency, VRAM (library cuBLAS weight cache), CPU share of the AED decode.
4. Whether a GPU AED decode (batched, resident) is worth building, given the 180M decode-bound
   profile (A100 60 s: 0.046 cpu vs 0.044 cuda).
5. Timestamps quality (generative `<|N|>`) on the 180M: UNKNOWN.

## 3a. Serving semantics in the CUDA server (decided 2026-10-09)

Canary goes into `mynah-asr-server-cuda` with the same infrastructure
(admission, concurrency, observability) and DIFFERENT serving semantics:
AED on the GPU, batched requests, final transcript. It is not cache-aware and
is never presented as true streaming. A pseudo-streaming / chunked mode for UX
(progressive windows) may be studied later, labelled as such, with the
recompute it costs measured. Not started before the EOU 120M passes its
end-to-end gate.

### Gate before Canary counts as supported in the CUDA server (decided 2026-10-09)

In order: CPU library == CUDA f32 (transcripts) -> batch=1 == batch=N ->
bf16 quality parity on a bank (not 4 clips) -> EN and FR -> REST under
concurrency -> no cross-request state contamination (a request's transcript
independent of what shares its wave). Only then a separate throughput ladder.
First compile + gates: `.work/lightweight-asr/jobs/c1.sh`.

## 4. Measurements

### 2026-10-09, L40S box, q1: FLEURS test 200 clips per language, f32, CPU CLI offline (build 4378448)

Same bank, scorer and normaliser as the EOU/Nemotron rows in
`.work/parakeet-eou-l4.md` (Nemotron streams; Canary is offline — the rows
compare quality, not serving semantics).

| model / lang | WER mean | WER* | CER mean | pooled WER | S / D / I | empty |
|---|---|---|---|---|---|---|
| Canary 180M, en | **0.0970** | 0.0669 | 0.0658 | 0.0987 | 261 / 88 / 102 | 1.0 % |
| Nemotron, en (lang en) | 0.1155 | 0.0863 | 0.0685 | 0.1160 | 373 / 60 / 97 | 0 % |
| Canary 180M, fr | **0.0886** | 0.0601 | 0.0521 | 0.0867 | 310 / 46 / 102 | 0 % |
| Nemotron, fr (lang fr) | 0.1310 | 0.1018 | 0.0652 | 0.1266 | 489 / 47 / 133 | 0 % |

DE/ES: `tools/fetch_eval_bank.py` now knows de/es/it (lw-eou-metrics); not run yet.


## 5. CUDA server design (2026-10-09)

Goal: `mynah-asr-server-cuda` serves Canary 180M Flash with honest offline
semantics -- the final transcript of a file, batched across requests on the
GPU -- and never presents it as streaming (3a).

### 5.1 Contract (mirrors the CPU server for the same pack)

- `POST /v1/audio/transcriptions` and `/v1/audio/translations` (multipart
  `file`, `language`, `response_format` json|text|verbose_json, `model`,
  `target_language`; or a raw WAV body): the final transcript, nothing before it.
  Same fields and formats as `server/main.c` `handle_transcribe`.
- `GET /v1/audio/stream`: `400 model_not_streaming` BEFORE the upgrade, no
  Retry-After -- what the CPU server answers (`server/main.c` `handle_ws_stream`).
  The WebSocket is not "accepted and answered at finalisation": that would be a
  second contract for the same pack on two servers, and a client could not tell
  it from a stalled stream. A client that has a file POSTs it.
- `/v1/models` says `"engine":"aed","streaming":false`; `/v1/health` says
  `"mode":"offline","streaming":false` with request books
  (`total = completed + failed + refused_* + queued + inflight`, `balanced`).
- Admission: `--cap` requests queued + in flight; past it `503
  server_at_capacity` with Retry-After (rule 5). A language the pack refuses is a
  `400 language_not_supported` before admission (`mynah_asr_lang_supported`), so
  it cannot fail a batch.

### 5.2 Capability, not model name

`asr_pack_mode` (`gpu/pack.c`) reads `decoder.type` from `mynah.json`:
`aed_transformer` -> `ASR_PACK_OFFLINE_AED` -> the offline mode
(`gpu/server/rest.c`), everything else -> the streaming engine unchanged. No
`if canary` anywhere: canary-1b-flash / 1b-v2 take the same route (their encoder
fits the same kernels; only VRAM and T_max change).

### 5.3 A sibling interface, not asr_engine

`gpu/asr_engine.h` is a per-slot, per-chunk step whose every call returns the
text one chunk appended. An AED decoder keeps no state between calls and only
has text once it has the whole segment, so behind that interface it would
either fake partials or run a step that does nothing until finalisation. The
honest unit is a request, hence `gpu/asr_offline.h`: `transcribe(n files) ->
n final texts`, one thread, the server's batcher.

### 5.4 One host half: the library, through an offload seam

The transcript must equal the CPU library's default decoding. Everything that
is not arithmetic -- WAV, segmentation on silence (30 s limit), the
per-feature mel, the canary2 prompt and its defaults, the generation budget
`T + max_generation_delta`, the greedy stopping rule, detokenisation, segment
stitching -- already exists once in `src/mynah_asr.c`. Re-implementing it in
`gpu/` would be a second path (rule 2). So the library gains one narrow seam,
`mynah_asr_offload` (`src/mynah_asr.h`): two optional hooks, `encode` (feats ->
post-projector encoder output, the batched encoder's contract) and `aed_decode`
(greedy decode of n segments with the library's stopping rule). NULL hooks =
the library's own code, byte for byte as before; no CPU path changes. The batch
path calls `aed_decode` once per wave (all its segments), the single path with
n = 1.

`gpu/offline_engine.c` = the library model (f32) + optionally the GPU offload:
`--engine cpu` is the reference (the CPU server's `mynah_asr_transcribe_batch`
call), `--engine cuda` the same call with `gpu/cuda/aed.cu` installed.

### 5.5 The GPU engine (`gpu/cuda/aed.cu`)

Weights: the library's own loaders (`mynah_asr_encoder_init`,
`mynah_asr_aed_init`) over the f32 pack, uploaded once. The batch_norm of the
conv module is folded at load into a per-channel affine (the library's
`bn_fold`, the same scale/shift the CPU applies); folding it further into the
depthwise weights was rejected because it changes the association of the sums
against the CPU.

Encoder, per wave of n <= `--batch` segments:
- dw_striding subsampling on the HOST with the library's code (per segment,
  ~10 % of the encoder FLOPs, non-causal pad 1/1); a later step can move it.
- the conformer layers on the device over PACKED rows `[sum T, d]` (no padding,
  no masks: a segment's rows are contiguous and every per-segment kernel reads
  its own offset/length):
  ½FFN (+bias, SiLU) -> LN -> q/k/v (+bias) -> full-context rel-pos attention
  -> o (+bias, accumulate) -> LN -> pw1 (+bias) -> GLU -> 'same' depthwise k9
  (+bias) -> BN affine -> SiLU -> pw2 (+bias, accumulate) -> ½FFN -> LN.
  GEMMs: the own row-stable f32 kernel (`k_gemm_wt`), or own-tc bf16 as the A/B
  arm (`--precision bf16 --gemm own-tc`; split count from the weight's shape
  only). Attention: one block per (row, head), scores over the segment's T keys
  in shared memory, `rk` read from the rel-pos TABLE the library builds and
  checks (`mynah_asr_enc_relpos_table_init`, kmax = T_max): rk(T)[p] =
  RK[p + kmax - T], so the index of key j for query t is `kmax - 1 + j - t`,
  identical values to the CPU's per-T projection.
- T_max = frames of 1.1 x the segment limit (the planner's largest segment);
  a longer segment fails that request, never the process.
- the post-projector output stays RESIDENT for the decoder (`from_encode`).

Decoder (AED, greedy), all segments of the wave stepped together:
- `enc_dec_proj` and the cross K/V of every layer once per wave over the packed
  rows; a per-segment self K/V cache `[sum max_len, d]` per layer.
- step p: one row per live segment: embedding + pos[p] + LN -> per layer
  (LN, q/k/v, append k/v at p, attention over [0..p], o accumulate; LN, cq,
  cross-attention over the segment's T rows, co accumulate; LN, FFN ReLU,
  accumulate) -> final LN -> head -> argmax (lowest index among equal maxima,
  the CPU's `>` rule). The host applies the library's rule per segment: inside
  the prompt the next token is the prompt's; else EOS ends it, the cap ends it,
  max_len = min(n_prompt + cap, max_seq) ends it. A finished segment leaves the
  step (rows compacted), so per-request stop costs nothing to the others.
- LN of the decoder reproduces `src/decoder_aed.c` `ln` (mean/var in double,
  the normalised value in double before the cast), attention scores
  `scale * q.k`, softmax `exp(s - max)` / sum.
- `--aed-decoder host` installs only the encoder hook: the library's CPU decoder
  decodes (stage 2, and the A/B arm).

### 5.6 Batch invariance and parity

- Contract 4 by construction: a GEMM row is one fixed-order fma chain over k
  (`gpu/cuda/gemm.cu`), every attention/conv/norm reads only its segment's rows,
  the decoder's rows are per segment. Gated, not trusted:
  `tests/test_cuda_aed` transcribes each clip alone, all together and in a
  rotated order, and requires identical bytes.
- Parity with the CPU library (f32): byte identity of FLOATS is impossible --
  the GEMM reduction order differs from `src/sgemm.c` / BLAS, and the
  transcendental functions differ (`expf`, sigmoid) -- so the gate is on
  TRANSCRIPTS: byte-identical to the library on the committed FLEURS clips
  (`samples/en`, `samples/fr`); a difference is reported per clip with its CER,
  and the bf16 arm is gated on WER against the f32 transcripts, never on bytes.

### 5.7 Stages

1. pack capability, the seam, the offline engine with the cpu reference, the
   server's offline mode (REST + 400 on the WebSocket), the GPU engine refusing
   clearly.
2. GPU encoder, library CPU decoder (`--aed-decoder host`).
3. GPU decoder, batched across the wave (`--aed-decoder gpu`, the default).

All three are written; none has run on a GPU yet (written on a machine without
nvcc: the .cu was syntax-checked with clang's CUDA front end only). Box order:
`make lib && make && make -C gpu && make -C gpu test-aed`, then
`tests/test_cuda_aed models/canary-180m-flash` (f32: gates A, B, C),
`... --aed-decoder host` (stage 2 alone), `... --precision bf16` (CER report),
`tests/test_cuda_aed_server.sh models/canary-180m-flash 8297 cuda`.

### 5.8 Open after the first box run

- Subsampling on the device (it is per segment on the host today; measure its
  share with `host_subsampling_ms_total` in /v1/health first).
- The decoder steps one row per segment through M <= --batch GEMMs and syncs
  once per step for the argmax: launch-bound at small batch; CUDA graphs per
  (rows) or a device-side stopping rule are the candidates, after a profile.
- bf16 own-tc: a WER gate on `samples/eval-bank` (en, fr) against f32 before it
  could become a default.
- canary-1b-flash / 1b-v2: same code path; VRAM and T_max to check.

### c1: Canary 180M in the CUDA server, first GPU run (2026-10-09, L40S, lw-canary-cuda 96ae1ec)

`.work/lightweight-asr/jobs/c1.sh`. First nvcc build failed at link
(`block_sum_d` defined in both aed.cu and kernels.cu; fixed in 96ae1ec, made
file-local). Then, on 7 clips (samples/en, samples/fr, FLEURS):
- dispatch map: 0 UNKNOWN; encoder (full-context rel-pos attention, 'same'
  depthwise conv with folded batch_norm) and AED decoder on the GPU,
  subsampling + mel + prompt + detok in the library on the host;
- A parity with the CPU library: 7/7 byte-identical transcripts, with the
  host decoder (stage 2), the GPU decoder f32 (stage 3) and the bf16 arm;
- B batch identity: 7/7 identical batched vs one by one; C word path identical;
- server: /v1/models streaming false; WS 400 model_not_streaming before the
  upgrade; REST identical to the CLI on 3 EN + 2 FR clips; two concurrent
  rounds 5/5 identical to one by one (no cross-request contamination on this
  set); translation en>de identical; unknown language 400; books balanced;
  SIGTERM clean.
- first cost picture (7 clips, batched, f32): enc 542 ms, dec 9858 ms, host
  subsampling 1787 ms; batched wall 2.7 s vs 27.9 s for the CPU library
  reference. The decoder dominates (one GEMM wave per token step), as the
  design note predicted.
Still owed before "supported": bf16 parity on a BANK (not 7 clips), EN+FR
bank WER on the CUDA path, REST concurrency ladder.
