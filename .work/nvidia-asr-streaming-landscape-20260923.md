# NVIDIA ASR streaming landscape — research ledger, opened 2026-09-23

Status: **OPEN — research phase.** This is the ledger for S13. It is a ledger,
not a report: entries are labelled and are appended, never quietly rewritten.

Task: S13-1
Question: what has NVIDIA actually released by September 2026, how are those
checkpoints really related internally, which of their primitives does mynah-asr
already execute, and what are the cheapest high-information experiments that
would make the existing engine materially better?

Labels used throughout: **FACT** (directly observed, with its source) ·
**SOURCE** · **OBSERVATION** (our reading of a fact) · **HYPOTHESIS** ·
**QUESTION** · **EXPERIMENT** · **RESULT** · **DECISION** · **NEXT**.
NVIDIA-documented statements are kept apart from our inferences. Anything not
confirmed is **UNKNOWN**, never a guess.

## What this phase inherits, and may not contradict

These are frozen results from the qualification phase on `main`. They are the
baseline every experiment here is measured against.

- **FACT.** Nemotron 3.5 streaming 0.6B int8 at `[56,3]` is QUALIFIED at C=80
  concurrent real-time streams on a 32-vCPU Axion (2 x 1800 s, 21287
  utterances, 0 lost, transcript parity 498/498 in both runs). Frozen. Not to
  be recomputed on this branch.
- **FACT (F32 / R-14).** The wait before the first word IS the speech the model
  consumes: paired per clip, `ttfp_from_speech - speech_consumed` is within
  [-78, +7] ms. Load does not move it (p95 1996 ms unloaded, 1986 ms at C=80).
  Publication is quantised to `0.256 + 0.32k` with no exception over 498 clips.
- **FACT (R-15).** The publication gate contributes **0 ms**: on 120 of 120
  clips the first non-blank decoder decision and the first published delta are
  the same distribution. There is no hidden early hypothesis.
- **FACT (M-6).** Parakeet TDT 110M Q4_K_M offline beats Nemotron 0.6B int8 on
  the same 498-clip English bank, same scorers, same session — WER mean 0.0916
  against 0.1045 offline and 0.1063 streaming, paired 167 better / 86 worse /
  245 tied, sign test z = +5.09 — while being 5.2x faster. It is **offline and
  English-only**; it is not a replacement for Nemotron.
- **OBSERVATION (M-6 / S12-7c).** Process CPU occupancy plateaus materially
  below the 30 allowed server cpus near throughput saturation on BOTH models —
  18.7-19.8 of 30 for offline Parakeet, 21.6-23.5 of 30 for streaming Nemotron.
  Different model, different decoder, different serving path. **Mechanism
  UNKNOWN**; a Nemotron-specific explanation is now less likely.
- **DECISION carried in.** Do not spend effort on publication policy or the
  scheduler for the first-word problem. The remaining delay is upstream of
  publication.
- **DECISION carried in.** The ~2 s p95 is tail-dominated (96.8 % of clips sit
  at 500-960 ms of speech consumed; the 1.6 % at `k >= 10` produce the p95).
  Investigate the tail, do not treat 2 s as typical.

---

# S13-2 — Checkpoint architecture forensics

## Method, and why it cost almost nothing

**FACT.** A `.nemo` file is a tar archive and `model_config.yaml` is its FIRST
member. An HTTP range request for the first 3 MB is therefore enough to read the
complete architecture of a checkpoint without downloading it.

**RESULT.** Primary configs for all five models obtained for **6 MB of
transfer** instead of ~3 GB: the 120M EOU `.nemo` is 460 MB and the multitalker
is 2.49 GB, and neither was fetched. Three further configs were already tracked
under `reference/`. **No Axion disk was touched** (the box is at 93 %).

SOURCE, fetched 2026-09-23:
- `reference/nemotron-3.5-asr-streaming-0.6b/model_config.yaml` (tracked)
- `reference/parakeet-tdt-0.6b-v3/model_config.yaml` (tracked)
- `reference/parakeet-tdt_ctc-110m/model_config.yaml` (tracked)
- `huggingface.co/nvidia/parakeet_realtime_eou_120m-v1` — `.nemo` head, config
  dated 2025-10-02, repo last modified 2025-12-03
- `huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1` — `.nemo` head,
  config dated 2025-10-15, repo last modified 2026-01-28
- `huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b` — `config.json`,
  repo last modified 2026-08-05; it also publishes a **q8_0 GGUF**

## FACT — every one of them is the same NeMo encoder class

All five instantiate `nemo.collections.asr.modules.ConformerEncoder` with
`nemo.collections.asr.modules.RNNTDecoder` + `RNNTJoint`. **There is no separate
"streaming encoder" class.** Streaming is switched on by configuration.

| | Nemotron 3.5 0.6B | **Parakeet Realtime EOU 120M** | Multitalker 0.6B | Parakeet TDT 0.6B v3 | Parakeet TDT/CTC 110M |
|---|---|---|---|---|---|
| encoder class | ConformerEncoder | ConformerEncoder | ConformerEncoder | ConformerEncoder | ConformerEncoder |
| n_layers x d_model | 24 x 1024 | **17 x 512** | 24 x 1024 | 24 x 1024 | 17 x 512 |
| `use_bias` | false | **false** | false | true | true |
| `causal_downsampling` | true | **true** | true | false | false |
| `att_context_style` | chunked_limited | **chunked_limited** | chunked_limited | regular | regular |
| `att_context_size` | `[[56,3],[56,0],[56,6],[56,13]]` | **`[70,1]`** | `[[70,13],[70,6],…]` | `[-1,-1]` | `[-1,-1]` |
| `conv_norm_type` | layer_norm | **layer_norm** | layer_norm | batch_norm | batch_norm |
| `xscaling` | false | **false** | false | false | false |
| `subsampling` | dw_striding x8, 256ch | **same** | same | same | same |
| mel `features` / `normalize` | 128 / NA | **128 / NA** | 128 / NA | 128 / per_feature | 80 / per_feature |
| `pred_rnn_layers` | 2 | **1** | 2 | 2 | 1 |
| decoder objective | RNNT | **RNNT** | RNNT | TDT | TDT (+CTC head) |
| `num_classes` | 13087 | **1026** | 1024 | 8192 | 1024 |
| language conditioning | `num_prompts: 128`, `prompt_field: target_lang` | **NONE** | none found | none | none |

**FACT — the offline/streaming split is four config values, not an
architecture.** Between offline Parakeet and streaming Nemotron the encoder
differs only in `att_context_size` (`[-1,-1]` versus a finite pair),
`att_context_style` (`regular` versus `chunked_limited`), `causal_downsampling`
and `conv_norm_type`, plus the mel `normalize` mode. The class, the subsampling
frontend, the attention type (`rel_pos`), the kernel size, the head count and
the decoder family are shared.

**FACT — Nemotron 3.5 is multi-context; the 120M is single-context.** Nemotron's
`att_context_size` is a LIST of four, so one checkpoint is trained to run at
`[56,3]`, `[56,0]`, `[56,6]` and `[56,13]`. The 120M declares exactly one,
`[70,1]`. The multitalker is multi-context again (`[[70,13],[70,6],…]`).

**OBSERVATION.** `[70,1]` gives a chunk of `(1+1) x 80 = 160 ms` against
Nemotron's `(3+1) x 80 = 320 ms` at `[56,3]` — **half the cadence** — with a
left context of 70 frames rather than 56. `q = 2` divides 70 exactly, so the
invariant `src/encoder.h` rests its unmasked attention on (R-10) holds.

## FACT — mynah's own streaming refusal is exactly this config difference

`mynah_asr_stream_unsupported()` (`src/mynah_asr.c`) refuses a pack on six
grounds. Mapped against the configs above:

| mynah gate | NeMo config key | Nemotron 3.5 | **EOU 120M** | Parakeet 110M / v3 |
|---|---|---|---|---|
| linear biases | `use_bias` | pass | **pass** | FAIL |
| folded batch_norm | `conv_norm_type` | pass | **pass** | FAIL |
| xscaling | `xscaling` | pass | **pass** | pass |
| symmetric (non-causal) conv padding | `causal_downsampling` | pass | **pass** | FAIL |
| per-feature mel normalisation | preprocessor `normalize` | pass | **pass** | FAIL |
| subsampling factor != 8 | `subsampling_factor` | pass | **pass** | pass |

**RESULT — Parakeet Realtime EOU 120M passes all six gates.** The runtime's
existing streaming path has no architectural objection to it.

## FACT — the prompt projector is already optional in mynah

`src/encoder.c:594` carries the branch `if (!enc->prompt_l1_w)` with the comment
*"model without prompt (Parakeet): encoder_projector only"*, and
`src/encoder.h:41` declares the prompt and post-encoder projector *"both
optional (NULL when absent)"*. It exists because the offline Parakeet needed it.

**OBSERVATION — the 120M is, structurally, Nemotron minus the prompt projector
at 110M-Parakeet dimensions.** Every dimension it needs is already exercised:
17 x 512 with a 1-layer predictor is the shape of the 110M pack mynah already
runs, and 128 mels with `normalize: NA`, causal downsampling, layer-norm conv
and chunked-limited attention are what mynah already runs for Nemotron.

**This makes it an unusually good scientific control, for a reason beyond its
size.** R-13 localised the high-norm early-frame encoder regime to the ReLU of
the **prompt projector** — `mid = ReLU([x, one-hot(prompt)] @ W^T + b)`, where
the one-hot is 1.0 against an `x` of RMS ~0.045. The 120M **has no prompt
projector at all**. It therefore removes precisely the module R-13 implicated,
while keeping the encoder family, the decoder family and the frontend.

## FACT — `<EOU>` is a vocabulary token

The 120M's tokenizer vocabulary contains `<EOU>` and `<EOB>` as ordinary
entries, at the end of the list after the alphabet. `num_classes: 1026` against
the 110M's 1024. So end-of-utterance is produced by the **RNNT joint like any
other token**, not by a side head and not by a VAD.

**OBSERVATION.** That means EOU costs no new inference machinery: it is a token
id to recognise. What is new is the **contract** — surfacing it as an event
distinct from transcript text.

**UNKNOWN.** The exact blank index and whether `<EOB>` is a second boundary
class; NVIDIA's own definition of EOU latency; false-positive behaviour;
whether `preserve_alignments`/`compute_timestamps: true` in its decoding block
changes anything we must honour.

## FACT — the multitalker's conditioning contract, from its config

`freeze_diar: true`, `spk_supervision_strategy: rttm`, `num_speakers: 4`,
`shuffle_spk_mapping`, `binarize_diar_preds_threshold`. Its encoder is
Nemotron-shaped (24 x 1024) and **also passes all six mynah gates**.

**OBSERVATION.** Speaker supervision arrives as RTTM-style speaker activity,
with a fixed maximum of 4 speakers, and the diarizer is frozen — i.e. a
separate model whose output conditions the ASR.

---

# What is NOT established

- **That "passes the six gates" means "runs correctly".** The gates are
  necessary, not sufficient. Tensor NAMES and layout inside the `.nemo`, the
  converter's ability to read this checkpoint, and numeric parity against the
  oracle are all unproven. No transcript has been produced.
- **That the 120M will be faster per stream in mynah.** 17 x 512 against
  24 x 1024 is ~4x fewer encoder MACs per frame, but `[70,1]` runs the encoder
  **twice as often** as `[56,3]`. The net is arithmetic we have not done and
  must measure, not assume.
- **That the 120M emits earlier.** That is the whole point of using it as a
  control and it is an EXPERIMENT, not a prediction.
- Anything about NeMo's runtime code, NIM, or the papers: those are S13-1 items
  still in flight at the time of writing.

# NEXT

1. Prove the tensor-level boundary: list the `.nemo` tensor names and shapes for
   the 120M and diff them against what `tools/convert_nemo.py` expects. Until
   that is done, "importer work only" is a HYPOTHESIS.
2. Finish the ecosystem audit (S13-1) and fold in the NeMo implementation facts.
3. Only then price the experiments and rank them.

---

# Addendum — the tensor-name boundary, argued from the converter

**FACT.** `tools/convert_nemo.py` carries ONE rename table, `_NEMO_RENAMES`, for
the whole ConformerEncoder/RNNT family: `encoder.pre_encode.conv.`,
`encoder.pre_encode.out.`, `.self_attn.linear_q/k/v/out.`,
`.self_attn.linear_pos.`, `.self_attn.pos_bias_u/v`, `.conv.batch_norm.`,
`decoder.prediction.embed.`, `decoder.prediction.dec_rnn.lstm.`, `joint.enc.`,
`joint.pred.`. A separate `_AED_RENAMES` exists only for Canary's transformer
decoder.

**FACT.** The family branch in the converter is on `durations` (TDT versus
RNNT) and it reads `pred_rnn_layers` from the config rather than assuming it.

**OBSERVATION.** Those keys are NeMo *module attribute paths*. They are
determined by the classes — `ConformerEncoder`, `RNNTDecoder`, `RNNTJoint` —
and not by the individual checkpoint. Since S13-2 established that all five
models instantiate exactly those classes, the existing table is **expected** to
apply to the EOU 120M unchanged, and its RNNT (not TDT) objective puts it on the
same branch as Nemotron.

**This remains an ARGUMENT, not a RESULT.** It predicts that the converter runs;
it does not show that it does, and it says nothing about numeric parity. The
experiment that settles it is cheap and is specified below.

**EXPERIMENT (S13-3a).** Download the 460 MB `.nemo` **on the development host,
never on the Axion** (which is at 93 %), run `tools/convert_nemo.py`, and record
one of two outcomes: a pack that loads, or the exact list of keys the table does
not cover. Rejection condition: any tensor the converter cannot name, any shape
the runtime refuses, or a `mynah_asr_stream_unsupported()` refusal at load —
each of which converts the hypothesis into a named piece of work rather than a
vague one. **Producing a pack is not producing a correct transcript**: numeric
parity against the Python oracle is a separate gate and rule 3 still applies.

---

# S13-1 — Ecosystem audit, September 2026

Sources fetched 2026-09-23 from `huggingface.co` model cards, their `config.json`
and `processor_config.json`, and the HF API. Where a card and a config disagree
that is recorded as a discrepancy, not resolved by preference.

## The findings that change what we do

**FACT — NVIDIA ships GGUF and its own C++ runtime.** `nemotron-3.5-asr-streaming-0.6b`,
`nemotron-speech-streaming-en-0.6b`, `parakeet-tdt-0.6b-v3` and `parakeet-ctc-1.1b`
each publish a `*.q8_0.gguf`, and those cards open their usage section with
"Run locally with NeMo-Speech.cpp" (`github.com/NVIDIA/NeMo-Speech.cpp`),
invoked as `nemo-speech transcribe audio.wav --model models/….q8_0.gguf`.
**OBSERVATION.** That is a first-party runtime in this repo's exact niche,
documented on the very model card mynah targets. It is not a reason to stop; it
is a reason to know what it does and does not do, and it belongs in the backlog
as an intelligence item before it is treated as either threat or irrelevance.

**FACT — the two Nemotron streaming checkpoints do NOT share a left context.**
The multilingual 3.5 declares `sliding_window=57` / `att_context_size=[56,R]`;
the English `nemotron-speech-streaming-en-0.6b` declares `sliding_window=71` /
`[70,R]`. Our own config forensics independently read `[[56,3],[56,0],[56,6],[56,13]]`
for 3.5. **Anything that hard-codes 70 as "the Nemotron left context" is wrong
for the multilingual checkpoint, and anything that hard-codes 56 is wrong for
the English one.** The EOU 120M's `[70,1]` shares its left context with the
English Nemotron, not with the multilingual one we serve.

**FACT — NVIDIA publishes its own evidence that a TDT Parakeet collapses under
chunked streaming.** From the `parakeet-unified-en-0.6b` card, `parakeet-tdt-0.6b-v2`
under chunking: **1.12 s → WER 22.83, 0.56 s → 69.55, 0.40 s → 95.12.**
**DECISION for S13-4a.** The prefix ladder is still worth running on v3 in EN and
FR, but it is now an experiment to locate *where* usable output ends, with a
strong prior that short prefixes are unusable. It is not a search for a hidden
streaming mode. This also retires any hope of cheaply pseudo-streaming v3 at low
latency.

**FACT — a model we had not listed exists and is directly relevant:
`parakeet-unified-en-0.6b`** (created 2026-04-07). "Unified-FastConformer-RNNT",
24 layers, 600M, English, PnC, ONE checkpoint for offline and streaming with
latency selectable 2080 → 160 ms in 80 ms steps. Its card states plainly: "The
current inference pipeline supports only **buffered streaming** (left context is
recomputed for each chunk)." Open ASR avg WER: offline 5.91, 1.12 s 6.29,
0.56 s 6.52, 0.32 s 6.92, 0.16 s 8.44, 0.08 s **15.63** — and NVIDIA itself
recommends switching to the cache-aware streaming model at 80 ms.
**OBSERVATION.** This is NVIDIA's own answer to "can one checkpoint do both",
and the answer is yes above ~240 ms and no below it. `.nemo` only; no
safetensors, no GGUF.

**FACT — NVIDIA's description of the language prompt matches our code.** The 3.5
card: "Language Encoding expands a 128-dim one-hot language vector across the
time axis → (K=128, T)… Concatenation along the feature axis → fused tensor
(D + K, T). Projection layer maps the fused features to the RNNT decoder." That
is exactly `mid = ReLU([x, one-hot(prompt)] @ W^T + b)` as read in R-13, and
`num_prompts: 128` as read from the config. The `<xx-XX>` tokens in the
tokenizer are OUTPUT tags emitted in `target_lang=auto` mode, not inputs.

**FACT — the EOU 120M is a research drop, not a product.** 621 downloads against
817k for Nemotron 3.5 and 568k for TDT v3; `.nemo` only, no HF-transformers
path, card unchanged since 2025-12-03. Its documented numbers: Open ASR average
WER **9.30** measured at 160 ms streaming, EOU detection latency p50 **160 ms**,
p90 280 ms, p95 320 ms, and it outputs **no punctuation and no capitalisation**.
**OBSERVATION.** WER 9.30 at 160 ms against Nemotron English 7.67 at 160 ms is a
real quality gap, and the missing PnC is a product-visible difference our corpus
scoring would hide, because `normalise()` strips both.

**FACT — multitalker costs one full model instance per speaker** ("the number of
model instances matches the number of speakers") and requires an external
streaming Sortformer diarizer (`nvidia/diar_streaming_sortformer_4spk-v2.1`),
with no enrollment audio and no speaker embeddings. Reported cpWER with that
diarizer: AMI-IHM 21.26, AMI-SDM 37.44, CH109 15.81, Mixer6 23.81.
**OBSERVATION.** For a serving fleet that is N x the compute per session rather
than a batched head — a capacity question before it is an architecture question.

**FACT — licences differ across one family.** Nemotron 3.5 is **OpenMDW-1.1**,
`nemotron-speech-streaming-en` and the EOU 120M are NVIDIA Open Model License,
Parakeet TDT v3 is **CC-BY-4.0**.

## Discrepancies recorded rather than resolved

- **FACT.** The 3.5 card advertises right contexts {0,1,3,6,13} while its own
  `processor_config.json` declares `supported_num_lookahead_tokens: [3,0,6,13]`
  — **no 1**. Our config read `[[56,3],[56,0],[56,6],[56,13]]`, which agrees with
  the processor and not with the prose. So the advertised 160 ms point has no
  declared preset in the checkpoint we serve.
- **FACT.** `processor_config.prompt_dictionary` carries 121 keys → 84 distinct
  prompt ids out of `num_prompts: 128`, including languages the card never
  claims. `auto` is id 101 and `default_prompt_id: 101`, while the card's
  pipeline note says the default is "index 0, en-US".
- **FACT.** `mt-MT` has a tier and a prompt id but **no `<mt-MT>` output tag**
  among the 39 locale tokens, so `auto` mode cannot emit a Maltese tag.

## What the cards do NOT say, and our forensics does

The EOU 120M repo publishes **no `config.json`**, so the card leaves hidden
size, vocabulary, subsampling and predictor depth UNKNOWN. S13-2's range-fetch
of the `.nemo`'s own `model_config.yaml` supplies exactly those: 17 x 512,
128 mels with `normalize: NA`, `subsampling_factor: 8`, `num_classes: 1026`,
`pred_rnn_layers: 1`, `use_bias: false`, `causal_downsampling: true`,
`conv_norm_type: layer_norm`. The two sources are complementary and agree
wherever they overlap (17 layers, `[70,1]`, RNNT, English, 120M).

---

# S13-1b — NeMo implementation intelligence

Read from the NeMo source (3.1.0 local checkout, fidelity-checked against
`NVIDIA-NeMo/NeMo@main`: 13 differing lines in `conformer_encoder.py`, **none**
touching `cache`, `att_context` or `streaming`) plus the NeMo user guide.

## The finding that closes a Track-B question

**FACT — FastEmit is TRAINING-time only. There is no inference-time emission
knob anywhere in NeMo.** It lives in the transducer loss gradient
(`nemo/collections/asr/losses/rnnt.py`, `fastemit_lambda` → `RNNTLossNumba`,
`grads[:, u, l] = (1 + fastemit_lambda) * grads[:, u, l]`). Shipped values:
offline `0.0`, **cache-aware streaming `5e-3`**, and the **EOU model's xlarge
config `3e-2`**. No decoding config carries an emission threshold, a delay
penalty or a FastEmit setting.

**DECISION.** This retires a whole family of ideas for our ~860 ms median: the
first-word latency of a transducer is **baked into the weights**, and a runtime
cannot trade latency for accuracy at decode time the way NVIDIA trades it at
training time. It also explains R-8 in hindsight — every intervention there
perturbed decoder state at inference, which is precisely the lever that does not
exist.

**OBSERVATION, and it sharpens the control.** The EOU 120M is trained with a
FastEmit lambda **6x larger** than the streaming default. If it emits earlier on
our clips, "trained to emit earlier" is a documented, quantified candidate cause
rather than a vague architectural one. If it does NOT emit earlier despite that,
the finding moves upstream of training to the frontend and the runtime, which is
exactly the fork S13-5 was set up to take.

## Corroborations of our own results, from NVIDIA's code

- **FACT.** NeMo **enforces** `att_context_size[0] % (att_context_size[1]+1) == 0`
  for `chunked_limited` and raises if violated. That is R-10's invariant,
  arrived at independently from our own encoder, confirmed in the reference
  implementation. It also confirms why `[56,2]` cannot exist for Nemotron.
- **FACT.** In `chunked_limited` the right context does **not** compound across
  layers (every frame in a chunk sees the whole chunk), so `cache_drop_size = 0`
  and nothing is recomputed. In `regular` style it **does** compound —
  effective look-ahead `= R x n_layers`. **OBSERVATION.** This is the precise
  mechanism behind R-10's finding that `[56,3] -> [56,0]` is worth ~120 ms and
  not 240: within a chunk the last frame has no look-ahead in any preset.
- **FACT.** `chunk_size = att_context_size[1] + 1` frames and
  `left_chunks_num = att_context_size[0] // chunk_size` — the same arithmetic
  mynah derives, from the same quantities.

## Where mynah is already ahead of the reference implementation

**FACT (NeMo).** `cache_last_channel` does **not** hold K/V. It holds the
post-`norm_self_att` layer input `x`, and `update_cache` concatenates it before
the projections, so **NeMo recomputes the Q/K/V linear projections over the
whole cached history on every step** (`multi_head_attention.py:207`,
`conformer_modules.py:197`).

**FACT (mynah).** `src/encoder.h:107` declares `float *k_cache, *v_cache;
/* [n_layers, left, d_model] */` — mynah caches the **projected** K and V.

**OBSERVATION.** Since the projections are linear and their inputs are frozen
once cached, the two are numerically equivalent and mynah's is strictly less
work per step. This is recorded not as a boast but because it removes a
candidate explanation: whatever costs us at saturation, it is **not** this
redundancy, and a future "optimise the cache like NeMo" suggestion should be
rejected on sight.

## Mechanisms worth taking, with their cost

- **A bit-exactness oracle for the cache, free.** NeMo ships
  `streaming.use_cache: false`: keep `left_context_size` frames of audio,
  re-encode them every chunk, then drop the first `left_context` output frames.
  It needs no retrain and no new model. **OBSERVATION.** That is exactly the
  kind of gate rule 4 asks for, for the one piece of streaming state mynah has
  no independent check on.
- **The endpointer NVIDIA actually ships is not the EOU token.** There are TWO
  EOU mechanisms and the production pipeline uses the second:
  `GreedyEndpointing` walks backwards over decoded token ids counting silent
  tokens, fires when `n_silent_tokens > stop_history_eou` and snaps to a word
  boundary. Shipped config: `stop_history_eou: 800` (ms), `residue_tokens_at_end: 2`,
  `word_boundary_tolerance: 4`. **It is model-agnostic integer bookkeeping over
  token ids — roughly fifty lines of C — and it would work on Nemotron today,
  with no new checkpoint.**
- **NVIDIA's own streaming latency metric is defined per EOU segment.** LAAL
  ("how far behind the audio the transcription is committed") is **skipped
  entirely when endpointing is disabled**. OBSERVATION: that is a more honest
  framing than a bare TTFP and is worth adopting alongside ours.
- **The slot allocator is the same design as our fleet.**
  `CacheAwareContextManager` with `num_slots` (default 256) >= `batch_size`, a
  free-list queue, stream->slot maps, one big cache tensor indexed by slot, and
  `index_fill_(0.0)` on release. Nothing to copy; it is corroboration that the
  serving shape mynah qualified at C=80 is the shape NVIDIA also arrived at.

## Multitalker, priced

**FACT.** Speaker conditioning is injected by `register_forward_pre_hook` on
`encoder.layers[0]` — **after subsampling, before Conformer layer 0** — as
`x = x + kernel(x * spk_mask)`, where the kernel is
`Linear(d->d_model) -> ReLU -> Dropout -> Linear(d_model->d)`. The encoder class
is never modified. The alternative (`masked_asr`) masks the mel or the
subsampled embedding instead.

**FACT.** The ASR takes **`(B,T)` per speaker**, not `(B,T,N)`: a target-speaker
mask plus the OR of all other speakers. The `(B,T,N) -> (B,T)` slice happens
outside the model. Masks are built at 100 fps from RTTM and averaged over 8 mel
frames to the 12.5 fps encoder rate. Values are **per-speaker independent
sigmoids, not a softmax over speakers**.

**FACT.** *"If there are at most N speakers and the batch size is B, then the
real batch size for inference is at most B * N"* — the entire cache-aware
encoder state and forward pass are replicated **per active speaker per
session**. `num_spks: 4` is a fixed architectural maximum in the diarizer's
output head; more speakers needs retraining. A **serial** single-cache path also
exists (`perform_serial_streaming_stt_spk`), which is the shape a CPU runtime
would take.

**OBSERVATION.** For our fleet this prices multitalker before any design work:
it is an N-fold capacity question first and an architecture question second. At
the qualified C=80, four speakers would mean C=20 sessions at the same compute.

## GPU-specific, and therefore not ours to copy

`use_triton` fused subsampling (upstream only, GPU-only, PyTorch fallback for
CPU/export) and `use_cuda_graph_decoder` (NVRTC + CUDA-graph conditional nodes)
— **which NVIDIA itself ships disabled** in the cache-aware config, commented
"Disabled due to issues with decoding". `loop_labels=True` is GPU-shaped
throughput scheduling and is documented as result-equivalent to the scalar
reference, so a C port is bit-compatible with the batched path by construction.
Everything else in the streaming, decoding, EOU and multitalker paths is plain
tensor and integer work that transfers to CPU.
