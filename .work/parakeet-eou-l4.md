# Parakeet Realtime EOU 120M on L4

Track: S15 (`.work/lightweight-asr-plan.md`). Model: `nvidia/parakeet_realtime_eou_120m-v1`
(`.nemo` only, NVIDIA Open Model License). Baseline: `nemotron-3.5-asr-streaming-0.6b`.
Branch audited: `research/lightweight-asr` at `4378448` (= `asr-cuda-perf`, PR #4).
Every claim below cites file:line in this tree; UNKNOWN means nothing in the repo
establishes it.

## 1. Audit of current support (2026-10-09)

### 1.1 The checkpoint, as converted

Converted pack on the dev machine: `models/parakeet-realtime-eou-120m/` holds
`mynah.json`, `model.safetensors` (459 MB f32), `model.int8.safetensors` (151 MB),
`mel_filters.safetensors`, `tokens.json` and the original `.nemo`.

| field | `.nemo` `model_config.yaml` (read 2026-10-09) | `mynah.json` |
|---|---|---|
| mel | `features: 128`, `normalize: NA`, n_fft 512, win 0.025, hop 0.01, `dither: 1e-5` | n_mels 128, normalize NA, win 400, hop 160, preemph 0.97, dither 0.0 (inference) |
| encoder | 17 L, d_model 512, 8 heads, ff_expansion 4, conv k 9, `conv_norm_type: layer_norm`, `conv_context_size: causal`, `dw_striding` x8 `causal_downsampling: true`, 256 ch, `use_bias: false`, rel_pos, xscaling false | n_layers 17, d_model 512, n_heads 8, ffn_dim 2048, conv_kernel 9, layer_norm, `dw_striding_causal`, use_bias false |
| attention | `att_context_size: [70, 1]`, `chunked_limited` | `att_context_presets: [[70,1]]`, default index 0, `encoder_frame_ms: 80` |
| decoder | RNNTDecoder, `pred_hidden 640`, `pred_rnn_layers 1`, joint 640 relu, `num_classes 1026` | `rnnt_lstm`, pred_hidden 640, pred_layers 1, vocab 1027, blank 1026, max_symbols 10 |
| vocab | `<EOU>` and `<EOB>` are the last two vocabulary entries | `tokens.json`: `<unk>`=0, `<EOU>`=1024, `<EOB>`=1025, `<blank>`=1026 |
| decoding | `greedy_batch`, `max_symbols 10`, `preserve_alignments`/`compute_timestamps: true` | greedy only |
| loss | FastEmit lambda 0.03 | (training-time only) |

n_mels is 128 in both; the converter reads it from the yaml (`tools/convert_nemo.py:293`).
No `prompt` section: the model has no language conditioning (`tools/convert_nemo.py:349-352`).

### 1.2 Conversion (`tools/convert_nemo.py`)

- Input = a directory holding only the `.nemo` (no `config.json`): `convert()` takes the
  `.nemo` route (`tools/convert_nemo.py:643-648`), needs torch (`uv sync --extra oracle`,
  `tools/convert_nemo.py:577`, `tools/pyproject.toml` extra `oracle`).
- Dispatch: `chunked_limited` encoder -> `build_streaming_rnnt_from_yaml`
  (`tools/convert_nemo.py:602-603`, builder at `:340-399`). It asserts
  `chunked_limited` <=> `causal_downsampling` (`:313-318`), `left % (right+1) == 0`
  (`:365-370`; 70 % 2 = 0) and refuses a streaming TDT (`:372-373`).
- `vocab_size = num_classes + 1`, `blank_id = num_classes` (`:389-390`); pieces = yaml
  `joint.vocabulary` + `<blank>` (`:625-626`).
- Tensor renames: `_NEMO_RENAMES` (`:227-242`); the S13-3b run recorded "465 tensors f32,
  1027 pieces", no unmatched key (`.work/nvidia-asr-streaming-landscape-20260923.md:553-563`).
- The mel filterbank is taken FROM the checkpoint (`:541-547`).
- Layout: `models/<any-dir-name>/<file>.nemo`; the dir name becomes `mynah.json` `name`
  (`:377`), which is what `?model=` must match on the servers. CI uses
  `models/parakeet_realtime_eou_120m-v1` (`.github/workflows/ci.yml:253-277`); the dev
  machine uses `models/parakeet-realtime-eou-120m`. The model is NOT in
  `scripts/download_model.sh`'s menu (`scripts/download_model.sh:19-32`): download with curl.
- Int8 pack: `./mynah-asr quantize -m <dir> --quant int8` writes `model.int8.safetensors`
  (CI: `.github/workflows/ci.yml:281`); the library loads it when `quant != f32`
  (`src/mynah_asr.c:183-188`). The CUDA engine never reads it: it opens the f32
  `weights` file only (`gpu/pack.c:49-53`).
- Numeric parity against the NeMo/Python oracle: **NOT run** (S13-3b "What this is NOT",
  `.work/nvidia-asr-streaming-landscape-20260923.md:582-587`; Q1 at `:607-616`). Only
  self-consistency (offline == streaming text, int8 == f32 on 3 clips) is recorded (`:562-567`).

### 1.3 Encoder, decoder, streaming state (CPU library)

- The pack runs on the generic cache-aware path (`engine: nemotron-streaming`). The
  streaming gates (`mynah_asr_stream_unsupported`, `src/mynah_asr.c:805-830`: no linear
  biases, no folded batch_norm, no xscaling, causal, normalize != per_feature, subsampling 8)
  all pass for this pack.
- No prompt projector: `src/encoder.c:594-600` takes the "model without prompt" branch
  (encoder_projector only); `resolve_prompt` returns -1 when the pack has no default prompt
  (`src/mynah_asr.c:458-463`).
- RNNT decoder: LSTM layers discovered by tensor name, any count up to
  `MYNAH_ASR_MAX_PRED_LAYERS` (`src/decoder.c:33-57`); hidden <= 1024 (`:57`; 640 OK).
  SOS = `pred_step(blank)` on a zero state (`:477`); state advances only on emission (`:585`).
- Streaming state per stream: mel stream, encoder K/V + conv caches (`mynah_asr_enc_stream_init`,
  `src/mynah_asr.c:853-854`), decoder state reset at open (`:858`). The whole state is reset only by
  `mynah_asr_stream_reset` (`src/mynah_asr.c:912-924`: mel, encoder cache, decoder, detok), i.e.
  on an explicit client `reset`, never by the model.

### 1.4 `<EOU>` / `<EOB>` handling: what is surfaced TODAY

**FACT: nothing surfaces the model's `<EOU>` token.**

- The greedy decoder treats id 1024/1025 like any non-blank token: it is emitted and fed back
  through `pred_step` (`src/decoder.c:580-585`); there is no id-specific branch anywhere
  (`grep -i eou src/decoder.c` is empty).
- The detokenisers drop every `<...>` piece as a "special" (`src/tokenizer.c:168-175` batch,
  `src/tokenizer.c:280-285` streaming), so `<EOU>`/`<EOB>` vanish from the text silently.
- **Decoder state after `<EOU>`: NOT reset.** The predictor keeps the post-`<EOU>` state;
  encoder caches continue. Upstream NeMo behaviour (whether its EOU pipeline resets the
  decoder after `<EOU>`) is UNKNOWN in this repo (to be settled in
  `.work/lightweight-asr-upstream.md`).
- Observed consequence (S13-3c, `.work/nvidia-asr-streaming-landscape-20260923.md:806-816`):
  on the 53 synthetic two-utterance (`cat2`) clips that were truncated, the 120M
  "transcribes the first utterance correctly and stops at the seam" (WER 0.6314 vs the full
  reference, 0.1516 mean vs the reference prefix). That is consistent with `<EOU>` firing and the
  un-reset decoder emitting nothing afterwards, but the `<EOU>` emission itself was never
  logged: the mechanism is a HYPOTHESIS. The S13-3b note says `<EOU>` "was not observed" on the
  clips tried, because it is stripped before any output (`:589-591`).
- What the `eou` event in Mynah IS: a **Silero-VAD endpoint**, not the model token.
  `mynah_asr_result.is_eou` is documented as "the VAD saw the utterance END"
  (`src/mynah_asr.h:28`, `:197-210`); it is produced only by `stream_vad_scan` /
  `stream_emit_eou` (`src/mynah_asr.c:1111-1150`, finish at `:1419-1427`) when
  `mynah_asr_enable_vad` was called.
  - CLI: `mynah-asr stream --vad <dir>` prints `{"type":"eou","t1":..}` (`cli/main.c:244-250`).
  - CPU server (`server/`): the scheduler would forward an `is_eou` result as
    `{"type":"eou"}` and count it (`server/sched.c:433-438`, `server/obs.c:293-296`), but the
    server never calls `mynah_asr_enable_vad` (no `vad` anywhere in `server/*.c`), so in practice
    it sends **no** `eou` frames.
  - GPU server (`gpu/server`): VAD/eou deliberately absent (`gpu/server/main.c:16`);
    `eous` is hard-coded 0 (`gpu/server/main.c:1063`, `:1195`); the cpu reference engine
    discards `is_eou` results (`gpu/engine_cpu.c:48`); `asr_step_out` has no EOU field
    (`gpu/asr_engine.h:78-86`).
- Diagnostic route that exists without code change: `MYNAH_ASR_TRACE_RNNT=1` prints each
  greedy decision with `chose=<id>` and `audio_s` (`src/decoder.c:123-130`, `:334-343`), so a
  `chose=1024` line timestamps an `<EOU>` decision on the CLI. Whether every decision of a
  multi-symbol frame is traced is UNKNOWN (the trace is per frame of the blocked loop).
- PLAN S13-7 (`PLAN.md:208`) records the gap: "what is new is the CONTRACT of surfacing it as an
  event distinct from text"; nothing implemented.

### 1.5 CPU path and CPU server

- CLI offline + streaming: work (S13-3b, `.work/nvidia-asr-streaming-landscape-20260923.md:553-575`;
  first delta `the` at t1 = 0.600 s on `fleurs_1521.wav`). Output lowercase, unpunctuated.
- CPU server (`mynah-asr-server`): streams it (it is the CI fault-suite model:
  `.github/workflows/ci.yml:246-283`). Batching: the scheduler stacks streams of the same preset
  into one batched step (`--batch`, default 8, max 64: `server/main.c:66`, `:1665`, `:1691-1692`).
  Default server quant is int8 (`server/main.c:85`).
- Gotcha, `lang`: the pack has no prompt dictionary, so ANY explicit `lang=` (including
  `lang=auto`) is refused 400 `language_not_served` on the CPU server (`server/main.c:646-653`
  via `mynah_asr_lang_id`, `src/mynah_asr.c:412-418`). Omit the key: `stream_load.py --lang ''`
  (`tools/bench/stream_load.py:1131-1132`, `:411`).
- Lookahead: the only preset is `1` (chunk = 2 frames = 160 ms). `stream_load.py`,
  `gpu_qualify.sh` and `gpu_knee.sh` default to `--lookahead 3`
  (`gpu/tools/gpu_qualify.sh:37`, `gpu/tools/gpu_knee.sh:42`): pass `--lookahead 1` or the
  server refuses (`lookahead_not_available`, `gpu/server/main.c:765-776`).
- CPU-server capacity / concurrency for this model: UNKNOWN (no load run recorded; only the CI
  fault suite with `MYNAH_ASR_THREADS=1`).

### 1.6 CUDA path (`gpu/`): the CUDA engine REFUSES this pack today

`asr_engine_open_cuda` checks the geometry and refuses when the pack has no prompt projector:

```
gpu/cuda/engine.cu:734-738
if (dm.H * dm.dk != dm.d || dm.Hdec != dm.dout || dm.kmax < dm.left + e->pack.qmax ||
    e->pack.qmax > GPU_QMAX_HARD || dm.left + e->pack.qmax > GPU_KMAX_HARD || !enc.prompt_l1_w) {
    snprintf(err, ..., "model geometry outside what the kernels serve (... prompt=%s)", ...);
```

Expected message for this pack: `... prompt=no`. Not yet observed on a GPU (the CI GPU-server
job uses the cpu-only build: `.github/workflows/ci.yml:284-285`, `Makefile:492-494`). The
`verify.sh` job (`.work/lightweight-asr/jobs/verify.sh`) will record it.

Prompt assumptions that must change for this pack (all Nemotron-shaped):

| where | what |
|---|---|
| `gpu/cuda/engine.cu:735` | refuses `!enc.prompt_l1_w` |
| `gpu/cuda/engine.cu:542-545` | uploads `prompt_l1/l2` unconditionally |
| `gpu/cuda/engine.cu:606-607` | bf16 copies of `pl1`/`pl2` (own-tc) unconditionally |
| `gpu/cuda/engine.cu:1013-1017` | post-encoder ALWAYS runs `k_prompt_cat` + two prompt GEMMs before the encoder projector (CPU equivalent: `src/encoder.c:594-600` skips them) |
| `gpu/cuda/engine.cu:874-875` + `gpu/pack.c:148-153` | `cuda_slot_reset` refuses `prompt_id < 0`; `asr_pack_lang_id` returns `default_prompt`, which is -1 for a prompt-less pack (`gpu/pack.c:129-131`), so even `lang=auto` would fail every slot reset |
| `gpu/server/main.c:758-764` | `lang=auto` validated through the same lookup |

Geometry that IS accepted (no other hard-coded Nemotron constant found in the open path):

| check | EOU 120M | limit / source |
|---|---|---|
| H x dk == d | 8 x 64 = 512 | `engine.cu:734`; attention threads = round-up(dk,32) >= 128 (`kernels.cu:241-245`) |
| Hdec == dout | 640 == 640 (encoder_projector 512 -> 640) | `engine.cu:734` |
| qmax = max(right)+1 | 2 | `GPU_QMAX_HARD 32` (`kernels.cuh:17`) |
| left + qmax | 72 | `GPU_KMAX_HARD 160` (`kernels.cuh:18`); rel-pos table built to left+qmax+2 (`pack.c:133`) |
| left % chunk | 70 % 2 = 0 | `pack.c:125-127` |
| conv kernel | 9 | `<= 16` (`kernels.cu:361`) |
| conv norm | layer_norm | batch_norm refused (`pack.c:102`) |
| biases / xscaling / causal | none / false / causal | `pack.c:99-104` |
| subsampling | 3 stride-2 stages, 128 mel -> 16, x256 ch = 4096 inputs to the linear | `engine.cu:726-729`, `:740-744` |
| decoder | `rnnt_lstm`, no durations, 1 LSTM layer (generic `pred_layers`) | `pack.c:66-72`, `engine.cu:490-491`, `:550-573` |
| vocab | 1027 (head argmax is a strided block reduce over V) | `kernels.cu:564-575` |
| per-lane cohort cap | `pass_lanes <= 1024` | `engine.cu:666-668` |

"24 layers" appears only in comments/profile labels (`engine.cu:6`, `asr_engine.h:146`); layer
count, d_model, ffn and heads are read from the pack (`engine.cu:723-732`).

**Own-tc bf16 GEMM shape constraints** (`gpu/cuda/gemm.cu:460-695`): none that refuse a shape.
Boundary tiles are masked (`m >= M || n >= N`, `gemm.cu:616`); the 16-byte vector load path needs
`K % 8 == 0` and `lda % 4 == 0`, otherwise a scalar path runs with the same arithmetic
(`gemm.cu:683`). EOU shapes: K in {512, 2048, 640, 4096, 256} are all multiples of 8; N = 1027
(joint head) and 2560 (LSTM 4H) are handled by masking. Split count S = f(N, K) only
(`gemm.cu:653-660`), so results stay batch-invariant. own-tc covers every big linear incl. the
subsampling pointwise/linear, encoder projector, joint head and LSTM (`engine.cu:593-612`).
Whether bf16 changes this smaller model's WER more than Nemotron's (0.14469 vs 0.14483 f32 on the
498 bank, `gpu/README.md` "Defaults") is UNKNOWN.

### 1.7 Yesterday's CUDA work (PR #4): does it apply?

All of it is model-agnostic in code, and none of it can run on this pack until 1.6 is fixed:

| lever | default (`gpu/server/main.c:1496-1517`) | applies to EOU? | must A/B on EOU |
|---|---|---|---|
| bf16 own-tc (`--precision auto --gemm auto` -> bf16 own-tc on sm_80+) | ON | yes once it opens | quality gate (WER/CER + EOU timing), device time per pass |
| `--stage-ahead 1` | ON | yes; byte-identical by construction | throughput only (host mel is a larger share for a 4x cheaper encoder) |
| `--host-threads auto` | ON | yes | as above; 128-mel host front end is identical to Nemotron's |
| `--graphs buckets` (8..128 lanes, row bucket 64) | ON | yes; row buckets sized from qmax (2 vs 4) | capture time/VRAM; replay share |
| `--warmup 1` | ON | yes (uses the default preset) | start-up time only |
| `--kv-dtype bf16/int8`, `--weights int8` | OFF | yes; per-slot K/V = 17 L x 2 x 70 x 512 x 4 B = 4.9 MB f32 (vs 10.5 MiB Nemotron, `gpu/README.md`) | only if VRAM-bound; L4 24 GB likely compute-bound first (UNKNOWN) |
| `--cohort-ms 40` | default | the model's chunk is 160 ms (vs 320 ms): a 40 ms cohort window is a larger fraction of the emission-lag bound (160 ms via `v2_verdict.py:101-102`) | A/B 20/40 ms |

`gpu_knee.sh` passes `--gemm "$GEMM"` with default `own` (`gpu/tools/gpu_knee.sh:39`, `:74`),
which forces f32: set `GEMM=auto` to measure the PR #4 default. `gpu_qualify.sh` defaults to
`auto` (`gpu/tools/gpu_qualify.sh:39`). `gpu_knee.sh` cannot omit `lang` (`LANG_Q=${LANG_Q:-auto}`,
`:41`), which a CUDA engine fixed per 1.6 must accept as "no prompt".

### 1.8 Tests that exercise it

- CI `server-faults-streaming` (`.github/workflows/ci.yml:239-285`): download + convert + int8,
  WebSocket fault suite on the CPU server and on the GPU server's **cpu-only** build. Correctness
  of text is not asserted there; the CUDA engine is never exercised with this pack.
- No test asserts `<EOU>` behaviour, EOU timing or WER for this model. `tests/test_cuda_stream`
  (gates A/B/C) takes any pack but cannot open this one (1.6).

### 1.9 Evaluation tooling relevant to EOU (what exists / what is missing)

- Scorer: `tools/bench/streaming_metrics.py` — `normalise()` (NFKC, lowercase, strips `<..>` tags
  and all Unicode punctuation, no number expansion; `:249-267`), `wer`/`cer` (Levenshtein,
  `:270-328`, `:508-516`), `wer_format_free`/`cer_format_free` (EN/FR number verbalisation,
  `:453-466`), `align_counts` (S/D/I, `:468-505`). Self-test via `make test`. The normaliser
  hides PnC by design, so Nemotron's PnC advantage is invisible in WER (landscape note `:304-307`).
- Per-language release gate: `tools/eval/lang_gate.py -m <pack> --manifest
  samples/eval-bank/manifest.json --root samples/eval-bank --mode stream|offline --langs en`
  (CLI `stream --deltas` or `transcribe`, `tools/eval/lang_gate.py:70-85`, `:200-213`): WER/CER with
  bootstrap 95 % CI, S/D/I, empty rate, WER by duration, first-word correctness. CLI `--lang en`
  is harmless on this pack (`resolve_prompt` returns -1 without a prompt, `src/mynah_asr.c:459`).
  `tools/eval/offline_quality.py` runs the 498 bank in parallel from a `bank.txt`.
- Earliness: `tools/eval/first_emission.py` (CLI `stream --deltas`, speech-onset to first
  non-blank / first published, from `MYNAH_ASR_TRACE_RNNT` traces), `tools/eval/earliness_cost.py`
  (paired dt vs dWER, Spearman, quartiles; `--drop-composed` removes the `cat2` clips,
  `tools/eval/earliness_cost.py:78`), `tools/bench/clip_onset.py` (energy ONSET only).
- Serving EOU metric: `streaming_metrics.analyze_utterance` counts `eou` frames and an `eou_ms`
  = eou arrival minus send time of the audio it closes (`streaming_metrics.py:818-852`) — a
  serving lag, not a speech-end-to-EOU latency.
- **Missing:** (a) any producer of an `eou` event from the `<EOU>` token (1.4); (b) a speech-END
  reference per clip (`clip_onset.py` gives onsets only; FLEURS has no word alignments); (c) a
  scorer for premature EOU (EOU before the reference's last word), missed EOU (no EOU within N ms
  after speech end), and speech-end -> EOU latency distribution; (d) a bank with real turn
  boundaries (the stress-en `cat2` long clips are the only multi-utterance audio; 0.6 s pauses,
  `tools/fetch_stress_bank.py:449`).

### 1.10 Banks

- `samples/stress-en/` (`tools/fetch_stress_bank.py`): FLEURS en_us train+dev+test, 3 x 529 clips,
  short/medium/long, long class = synthetic concatenations (`composed`), seed 42, gitignored.
  The "498-clip bank" = `--corpus-sample 498` seed 42 of it (bank `04a7753aa1e80f9a`,
  `docs/benchmarks.md:172-174`); its `bank.txt` lives under `.work/evidence/` (gitignored, so not
  on a fresh clone; regenerated deterministically by `gpu_qualify.sh --corpus-sample 498
  --corpus-seed 42`). "349 originals" = its non-composed clips; 149 are `cat2` concatenations.
  Caveat: FLEURS train/dev are plausibly in NVIDIA training data (UNKNOWN).
- `samples/eval-bank/` (`tools/fetch_eval_bank.py`): FLEURS **test** split, 200 EN + 200 FR
  (`LANGS = {"en","fr"}`, `tools/fetch_eval_bank.py:43`). The clean held-out EN bank for S15-2.

## 2. Prior evidence in this repo

All CPU (Axion c4a, 2026-09-23), CLI `mynah-asr stream --deltas`, int8, default preset, same
committed tool, same session for both models. Raw JSON under
`.work/evidence/first-word-2026-09-23/box/` (gitignored, dev machine only).

| measurement | Nemotron 3.5 0.6B `[56,3]` | EOU 120M `[70,1]` | source |
|---|---|---|---|
| 498 bank, WER mean / p50 | 0.1063 / 0.0714 | 0.2290 / 0.150 (incl. seam truncation) | `w498-*.json` |
| 498 bank, corpus WER / CER | 0.1462 / 0.0887 | 0.3258 / 0.2759 | `w498-*.json` |
| 349 originals, WER mean / p50 | 0.1040 / 0.0667 | 0.1164 / 0.0769 | `PLAN.md:206`, landscape `:795-799` |
| paired on 349: dWER mean / dCER mean | — | +0.0124 / +0.0079 (z = -3.26) | `earliness-cost-original.json`, `PLAN.md:207` |
| speech onset -> first non-blank p50 / p95 (498) | 860 / 1990 ms | 570 / 1590 ms | `w498-*.json` |
| earlier on (349 paired) | — | 311 earlier, 4 later, 34 tied, median -300 ms | `PLAN.md:207` |
| blank decisions before first token p50 | 17 | 16 | `w498-*.json` |
| `cat2` seam truncation | 0 / 149 | 53 / 149 stop at the seam | landscape `:806-816` |

- Normaliser: `streaming_metrics.normalise` for every row; references = FLEURS transcriptions.
- Upstream card numbers (not ours): Open ASR avg WER 9.30 at 160 ms, EOU latency p50 160 ms,
  p90 280, p95 320 ms; Nemotron EN 7.67 at 160 ms (landscape `:299-307`).
- S13-5b replay: EOU emits after ~2.2 fewer decoder decisions and on a finer publication grid
  (landscape `:704-760`); FastEmit 3e-2 vs 5e-3 noted as a candidate cause.
- GPU numbers for this model: **none**. EOU detection (token) numbers: **none**. Oracle parity: **not run**.

## 3. Open questions to measure

1. Oracle parity (NeMo vs Mynah, per stage and transcript) for this pack — Q1, never run.
2. Does the model emit `<EOU>` (id 1024) / `<EOB>` (1025) on our banks, where, and how often
   (trace route, 1.4)? What does NeMo do with the decoder state after `<EOU>` (reset or not)?
3. WER/CER/S/D/I on `samples/eval-bank/en` (FLEURS test) vs Nemotron, same normaliser, int8 and
   f32 (and bf16 once on CUDA).
4. EOU quality: premature EOU rate, missed EOU rate, speech-end -> EOU latency (needs a speech-end
   reference and a producer of the event).
5. CUDA: can the engine be made to open a prompt-less pack (1.6) byte-identical to the CPU
   reference (gate B), and then C1 step / first partial / finalisation / RTFx / VRAM per stream /
   SM % / W on L4 (`sm_89`), ladder to the knee vs Nemotron's.
6. PR #4 levers on THIS model: bf16 own-tc quality gate, stage-ahead/host-threads/graphs gain
   when the encoder is ~4x cheaper per frame but runs twice as often (chunk 160 ms vs 320 ms).
7. CPU-server capacity (C knee) on the same box, as a cross-check.

## 3a. End-to-end gate sequence before any large benchmark (decided 2026-10-09)

The EOU is the reason this model is interesting; a good WER and a large knee
with a broken endpoint would answer the wrong question. In order, each step
must pass before the next one is run:

1. the CUDA engine loads the pack (the prompt projector is optional);
2. one utterance, transcript sane;
3. `<EOU>` really emitted, surfaced as an event with its audio time, never as text;
4. decoder state reset after `<EOU>` (as the NeMo reference does — to be cited);
5. utterance B on the SAME stream after A's `<EOU>`: transcript sane
   (A + gap + B composites, gaps 0.3/0.6/1.0/2.0 s);
6. CPU engine vs CUDA engine parity (f32 own);
7. bf16 own-tc vs f32 quality parity on this model;
8. WER/CER bank against Nemotron, same audio, same normaliser;
9. speech-end -> `<EOU>` latency, premature and missed EOU, measured
   separately from transcript quality (`tools/eval/eou_metrics.py`);
10. only then the concurrency ladder on the CUDA server, Nemotron and EOU on
    the same card, same methodology.

## 3b. Latency decomposition and the CUDA perf audit (decided 2026-10-09)

Visible EOU latency ~= L_model (audio time) + L_server (wall clock):
- L_model: `tools/eval/eou_metrics.py`, run faster than real time, the eou's
  `t` = end of the encoder frame that emitted it -> acoustic evidence the
  checkpoint needs after the speech end. Architectural floor: 160 ms chunk +
  1 frame (80 ms) lookahead. Kernels cannot reduce this term; only the
  checkpoint (FT / endpointing data) or an external VAD/silence policy can.
- L_server: `lag_ms` of the eou frames in the l1 ladder, per C.

The l1 ladder is an ARCHITECTURAL BASELINE: the CUDA path was tuned on
Nemotron and this model just entered it. After l1, a perf audit that does
not touch quality: profile unloaded, C1, C32, C256 and near the knee
(`--profile-stages`, `--profile-host`); per point, normalised per AUDIO
SECOND and side by side with Nemotron: GPU compute ms, host ms, kernel
launches, GEMM ms, RNNT predictor+joint ms; plus GEMM shapes, graph coverage,
syncs/H2D per chunk, serialised per-stream work. Decision rule: the parameter ratio (~5x) is a REFERENCE, not a target —
inference cost does not scale linearly with parameters (fixed per-chunk host
and launch costs, mel, RNNT predictor/joint, GEMM shapes and chunking scale
differently). If the EOU is much smaller but its GPU ms per audio second
drops surprisingly little against Nemotron's, profile-guided optimisation is
warranted (targeted A/Bs, each with its byte-identity / quality gate); a 5x
is not chased. No optimisation before a proven bottleneck.

## 4. Measurements

### 2026-10-09, L40S box, q1: FLEURS EN test 200 clips, f32, CPU CLI stream (build 4378448, BEFORE the model-EOU reset)

Command: `.work/lightweight-asr/jobs/q1.sh` (tools/eval/lang_gate.py, the
streaming_metrics normaliser; bank: tools/fetch_eval_bank.py --n 200).

| model | WER mean | WER* (format-free) | CER mean | pooled WER | S / D / I | empty |
|---|---|---|---|---|---|---|
| EOU 120M (no reset) | 0.1607 | 0.3106 | 0.1194 | 0.3014 | 291 / **1001** / 85 | **20.0 %** (40/200) |
| Nemotron, lang en | 0.1155 | 0.0863 | 0.0685 | 0.1160 | 373 / 60 / 97 | 0 % |
| Nemotron, lang auto | 0.1227 | 0.0985 | 0.0738 | 0.1237 | 377 / 93 / 95 | 0.5 % |

CORRECTION (same day): lang_gate's "WER mean" is over NON-EMPTY transcripts
only; the empty ones are counted in "empty" and in the pooled WER. So 0.1607
is already the EOU's WER on the 160 clips it accepted; with the blanks the
pooled WER is 0.3014.

Diagnostic, NOT a headline (conditioning on the model's own acceptance is
cherry-picking): the SAME 160 clips the EOU accepted, every model scored on
them:

| same 160 clips | WER | WER* | CER |
|---|---|---|---|
| EOU 120M | 0.1607 | 0.1383 | 0.1194 |
| Nemotron, lang en | 0.1074 | 0.0792 | 0.0640 |
| Nemotron, lang auto | 0.1103 | 0.0821 | 0.0648 |
| Canary 180M (offline) | 0.0964 | 0.0673 | 0.0674 |

Paired against Nemotron (lang en) on those 160: EOU better on 29, worse on
71, tied on 60. The 40 blank clips are not hard audio (Nemotron 0.148,
Canary 0.100 on them) and are shorter (7.4 s mean vs 11.0 s). Reading: BOTH
a high catastrophic-blank rate AND lower quality when it does transcribe
(CER nearly 2x Nemotron's) on this bank. Both matter for what an FT can fix.

### NeMo reference parity (gate 6 against the reference implementation)

`.work/lightweight-asr/jobs/nemo_ref.{sh,py}`: NeMo 3.0.0, the same `.nemo`,
`EncDecRNNTBPEModel.transcribe` on CPU, special tokens kept. On the 40 clips
Mynah returned empty plus 40 it transcribed:

- the 40 transcribed clips: Mynah text == NeMo text on **40/40** once `<EOU>`
  is removed from NeMo's;
- the 40 empty clips: NeMo is empty on **40/40** as well — 33 with no token at
  all, 7 with a lone `<EOU>` and nothing after it (no reset in an offline
  `transcribe`).

So the 20 % empty rate is the CHECKPOINT on this audio, not a Mynah defect;
the 7 "EOU first, then silent" clips are what the model-EOU reset addresses.
Why the checkpoint emits nothing on 33 FLEURS clips is open (read speech,
levels, leading silence: to be probed, not assumed).

### g1: end-to-end gate on the GPU (build aabc16f, `.work/lightweight-asr/jobs/g1.sh`)

- CUDA engine loads the pack: dispatch map 0 UNKNOWN, `post-encoder
  encproj-only`; start-up with defaults bf16 own-tc, 9 graphs, 1195 ms,
  **1940 MiB VRAM** (Nemotron with the same defaults: 6862 MiB).
- `tests/test_cuda_stream --lookahead 1` on A+1 s+B, fleurs_1521, fleurs_1534,
  fleurs_long: gate A (batch identity, repeat, slot independence), gate B
  (CPU f32 == GPU, text AND model eous: 7.69/20.28 s on A+B, 7 eous on
  fleurs_long) and gate C (stage-ahead, host threads, graphs, warm-up, all
  together) **PASS in f32 own and in bf16 own-tc** (bf16: same text and eous
  as CPU f32 on these 4 clips; bank-level parity is q2).
- CPU server vs CUDA server WebSocket frames on A+1 s+B: **identical**, eou
  frames `{"source":"model","t":7.69}` and `{"t":20.28}`.


### l1: first concurrency screen, CUDA server, L40S (2026-10-09, build aabc16f)

`.work/lightweight-asr/jobs/l1.sh`: gpu_qualify ladder, 150 s rungs, fresh
server per rung, PR #4 defaults (bf16 own-tc, stage-ahead, graphs, warm-up,
host threads), cohort 40 ms, server on CPUs 0-31,64-95, generator on
32-63,96-127, stress-en 498 (seed 42), judged against the q2 unloaded
references; EOU cap 2048 lookahead 1, Nemotron cap 1024 lookahead 3, lang auto.
Same C side by side (absolute numbers; the bounds differ: EOU 160 ms,
Nemotron 320 ms, i.e. (lookahead+1) x 80):

| C | model | verdict (own bound) | lag p95 | final p95 | worst window p95 | SM % busy | W mean | RSS growth | audio/wall |
|---|---|---|---|---|---|---|---|---|---|
| 512 | Nemotron 0.6B | QUALIFIED | 56 ms | 64 ms | 56 ms | 34.6 | 151 | 1.033x | 389x |
| 512 | EOU 120M | QUALIFIED | 48 ms | 54 ms | 49 ms | 19.5 | 107 | 1.058x | 390x |
| 768 | Nemotron 0.6B | QUALIFIED | 106 ms | 125 ms | 116 ms | 44.3 | 170 | 1.050x | 573x |
| 1024 | Nemotron 0.6B | NOT (lag, final, drift) | 542 ms | 796 ms | 562 ms | 55.7 | 184 (sw_power_cap) | 1.059x | 716x |
| 1024 | EOU 120M | NOT (drift 176 > 160; RSS 1.177x > 1.15x) | 148 ms | 218 ms | 176 ms | 31.2 | 148 | 1.177x | 764x |

Every rung: 0 errors, 0 rejected, 498/498 transcripts identical to the
unloaded reference. EOU C1536/C2048 NOT MEASURED: the load generator died
(`RuntimeError: can't start new thread`; stream_load.py is one process + one
reader thread per stream). The container CPU quota (~30.7) is SHARED by the
server and ~C generator processes, so the C1024 rungs of both models may be
partly generator-limited: a multiplexed generator (lw-async-load) and an A/B
against the old one come before any C >= 1024 claim.
Reading: at C1024 the EOU's lag is 3.7x lower than Nemotron's at ~0.56x the
busy SM and -36 W. SM utilisation is NOT GPU compute cost (kernel mix,
occupancy and shapes differ between the models), so the 31/56 ratio is a
HINT that motivates the 3b audit, not evidence; no optimisation is decided
on it. Idle VRAM (1.9 vs 6.9 GiB) is the comparable memory number; the
loaded VRAM is dominated by the cap-sized arena. Screens, not soaks.

### lvl: level robustness, DIAGNOSTIC (2026-10-09, `.work/lightweight-asr/jobs/lvl.sh`, `lvl_small.sh`)

Same 200 FLEURS EN test clips, same f32 weights, CLI stream, same scorer;
three deterministic offline transforms (not a runtime change):

| arm | pooled WER | WER mean | CER mean | S / D / I | empty |
|---|---|---|---|---|---|
| EOU, original audio (q1) | 0.3014 | 0.1607 (non-empty only) | 0.1194 (non-empty) | 291 / 1001 / 85 | 20.0 % |
| EOU, peak -3 dBFS | **0.1224** | 0.1223 | 0.0718 | 394 / 56 / 109 | 0 % |
| EOU, RMS -23 dBFS | 0.1252 | 0.1245 | 0.0730 | 402 / 55 / 115 | 0 % |
| EOU, causal AGC (300 ms window -> -23 dBFS, <= +40 dB) | 0.1257 | 0.1255 | 0.0735 | 403 / 59 / 112 | 0 % |
| Nemotron, original (q1, lang en) | 0.1160 | 0.1155 | 0.0685 | 373 / 60 / 97 | 0 % |
| Nemotron, peak -3 dBFS (control) | 0.1119 | 0.1100 | 0.0640 | 359 / 52 / 100 | 0 % |

The FLEURS collapse is a LEVEL sensitivity of the checkpoint (normalize: NA),
removed by any of the three, including a causal streamable AGC; Nemotron
barely moves. At sane levels the EOU is ~1 point of WER behind Nemotron on
this bank, as on the 498-clip bank (0.1545 vs 0.1447).

EOU behaviour on the peak bank (40 clips, gap 1 s; the full 100-clip pass was
replaced by this short one to save time):

| | original (q2, 200 clips) | peak -3 dBFS (40 clips) |
|---|---|---|
| speech end -> EOU p50 / p90 / p95 (single) | 1738 / 3466 / 3946 ms | 1706 / 2986 / 3274 ms |
| missed 1 s / 2 s / never | 85.5 / 55.0 / 27.5 % | 82.5 / 37.5 / 2.5 % |
| premature | 0 % | 0 % |
| A + 1 s + B: EOU in gap, latency p50 | 35 %, 394 ms | 67.5 %, 586 ms |
| WER A / B (split at the EOU) | 0.160 / 0.198 | 0.113 / 0.130 |

Level explains the "never" EOUs and part of the misses, NOT the ~1.7 s
median decision latency on single read-speech clips: that is the checkpoint
on this audio (the card's 160 ms p50 is on TTS dialogue audio). Levers for it:
endpointing data in FT, or an external VAD/silence policy; not kernels.
Implication for FT: random gain augmentation (the MLS-it pool sits at peak
p50 -7 dBFS, clean levels).

### l2: multiplexed load generator + EOU beyond C1024 (2026-10-09, `.work/lightweight-asr/jobs/l2.sh`)

Same method as l1; `stream_load.py --mux 16` (lw-async-load) where marked.

| run | C | lag p95 | final p95 | worst window p95 | RSS growth | audio/wall | SM % busy | W mean | verdict (own bound) |
|---|---|---|---|---|---|---|---|---|---|
| EOU, old generator | 1024 | 158 ms | 218 ms | 172 ms | 1.172x | 764x | 30.5 | 152 | NOT (drift, RSS) |
| EOU, mux 16 | 1024 | 161 ms | 218 ms | 173 ms | 1.157x | 765x | 30.8 | 111 | NOT (lag by 1 ms, drift, RSS) |
| EOU, mux 16 | 1536 | 301 ms | 409 ms | 305 ms | 1.103x | 1135x | 42.7 | 141 | NOT (lag, drift vs 160) |
| EOU, mux 16 | 2048 | 19380 ms | 24152 ms | 21144 ms | 1.095x | 727x | 40.2 | 127 | NOT (collapse) |
| Nemotron, mux 16 | 1024 | 505 ms | 759 ms | 547 ms | 1.049x | 721x | 56.7 | 186 | NOT (lag, final, drift) |

- The generator was NOT the limit at C1024: old and mux agree within 3 ms on
  every latency; l1's C1024 readings stand, and the EOU's RSS growth at C1024
  is the server's (to be looked at in the audit).
- EOU knee between C1536 and C2048, reached with the GPU at ~40 % busy SM:
  host/scheduler-bound, the first target of the 3b audit.
- Against a COMMON 320 ms lag bound: Nemotron's last passing rung is C768
  (C1024 = 505 ms); the EOU stays under it up to C1536 (301 ms; one window at
  305 ms, finalisation 409 ms < 500) -> about 2x the sessions on this L40S,
  with idle VRAM 1.9 vs 6.9 GiB. Screens, not soaks; L4 rerun owed.
