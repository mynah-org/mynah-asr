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

---

# S13-3a — RESULT: the converter refuses, and names exactly what is missing

**EXPERIMENT, run 2026-09-23 on the development host** (never on the Axion). The
460 MB `.nemo` was downloaded to the Mac — 460,062,720 bytes, the size the HF API
declares — the oracle environment installed (`torch 2.13.0`), and
`tools/convert_nemo.py` run against it.

**RESULT — a named refusal, not a crash:**

```
tools/convert_nemo.py:306, in yaml_encoder_section
AssertionError: only non-causal offline encoders from .nemo (for now)
```

**FACT — the gap is a matrix hole, and it is one cell.** The converter has two
independent builders:

| | HF-native (`config.json` + safetensors) | `.nemo` archive |
|---|---|---|
| **offline, non-causal** | `build_parakeet_tdt` | `build_parakeet_tdt_from_yaml` |
| **streaming, causal** | `build_nemotron` | **nothing** |

Nemotron reached mynah through the HF-native path, because NVIDIA publishes
`config.json` and `model.safetensors` for it. The EOU 120M publishes **only a
`.nemo`** (confirmed against the repo file list) and is **causal +
chunked_limited**. It is the one combination with no route.

**FACT — the precise field delta**, read by diffing `build_nemotron` (the
streaming builder that works) against `yaml_encoder_section` (the `.nemo` reader
that refuses), with the 120M's own yaml values in the last column:

| mynah.json field | present in the yaml reader? | source in the 120M yaml | value |
|---|---|---|---|
| `arch` / `engine` | writes the offline pair | — | `fastconformer_rnnt_streaming` |
| `encoder.subsampling` | passes `dw_striding` through | derive from `causal_downsampling: true` | **`dw_striding_causal`** |
| `encoder.att_context_style` | asserted `regular` | `att_context_style` | `chunked_limited` |
| `encoder.conv_norm` | passed through | `conv_norm_type` | `layer_norm` (already right) |
| `encoder.use_bias` | passed through | `use_bias` | `false` (already right) |
| `features.normalize` | passed through | `preprocessor.normalize` | `NA` (already right) |
| `streaming.att_context_presets` | **absent** | `att_context_size` | `[[70, 1]]` |
| `streaming.default_preset_index` | **absent** | — | `0` (one preset) |
| `streaming.encoder_frame_ms` | **absent** | `hop/sr x subsampling x 1000` | `80.0` |
| `prompt` | **absent** | — | correctly omitted: this model has none |

**OBSERVATION.** Nine of those ten rows are either already produced correctly by
the yaml reader or are a direct copy of a field the yaml already carries. The
only derivation is `causal_downsampling: true -> dw_striding_causal`, and the
only genuinely new block is `streaming`, which `build_nemotron` already knows
how to shape.

**DECISION.** S13-3's question C is answered with evidence: support for this
checkpoint is **importer work**, not new inference machinery, and the importer
work is one builder in `tools/convert_nemo.py`. The earlier hypothesis is
upgraded to a result **for the config layer only**.

**Still NOT established, and the order matters:**
1. That the `.nemo` tensor names match what `_NEMO_RENAMES` expects — argued in
   the addendum above from the shared NeMo classes, not yet observed.
2. That the tokenizer extracts (the 120M's SPE files sit **after** the 460 MB
   weights in the tar, so they cost a full read, unlike the config).
3. That the runtime loads the pack and `mynah_asr_stream_unsupported()` returns
   NULL in practice rather than on paper.
4. **That any transcript it produces is correct.** Rule 3 is not suspended on a
   research branch: numeric parity against the Python oracle is a separate gate
   and a pack that loads is not a pack that works.

**NEXT.** Write the builder, then take those four in order. A failure at any of
them is a finding and is recorded, not worked around.

---

# S13-3b — RESULT: it converts, it loads, it streams. Engine unchanged.

**EXPERIMENT.** Extend `tools/convert_nemo.py` with the missing cell — a
cache-aware builder on the `.nemo` route — and run it. **No change to `src/`.**

**RESULT.**

```
OK parakeet-realtime-eou-120m [nemotron-streaming] from .nemo:
   465 tensors f32, 1027 pieces, mel fb (1, 128, 257) from the checkpoint
```

- **FACT — `_NEMO_RENAMES` covered every tensor.** No key was unmatched. The
  argument from shared NeMo classes is now a result, for the naming layer.
- **FACT — it transcribes offline**: 7.4 s clip, RTF 0.139 f32 on the
  development host, *"the satellite in space gets the call and then reflects it
  back down almost instantly"*.
- **FACT — it streams cache-aware**: 18 incremental deltas, and the streaming
  final text is identical to the offline text.
- **FACT — int8 matches f32** on all three committed English clips, and the
  int8 pack is 150 MB against 460 MB.
- **FACT — it runs on the Axion** on a stress-bank clip from the qualification
  corpus.
- **FACT — the output is lowercase and unpunctuated**, which is exactly what
  NVIDIA's card says this model does. That is a cheap identity check: we are
  running the model we think we are, not a mis-shaped copy of something else.

**FACT — the first delta on `fleurs_1521.wav` lands at `t1 = 0.600 s`** with
the text `"the"`. For Nemotron on the qualification corpus the EARLIEST
publication ever observed over 498 clips was 0.896 s of audio, and the median
was 1.536 s. **OBSERVATION.** One clip is an anecdote, and it is recorded as
one; the distribution is what decides, and it is being measured on the same 120
clips that produced the Nemotron baseline.

## What this is NOT

- **Numeric parity against the Python oracle has NOT been run.** Rule 3 is not
  suspended on a research branch. Plausible text from a brand-new importer is
  the weakest evidence in this repo, not the strongest: the offline and
  streaming paths agreeing shows they agree with *each other*, not with NeMo.
- No WER has been measured on any corpus for this model here.
- No concurrency, no capacity, no soak. Nothing about this may be promoted.
- `<EOU>` was not observed in the output on the clips tried so far. That is
  expected-ish — the card's EOU latency is measured with trailing silence and
  these clips end near the last word — but it is **UNKNOWN**, not explained.

## What it cost

One builder, about eighty lines, in an offline tool. The inference engine, the
server and the scheduler were not touched. That is the answer to S13-3's
question C, and it is worth stating plainly because the S13-2 analysis
predicted it before the code was written.

---

# S13-9 — Ranked backlog, by evidence and cost

Ranked by **information per unit of cost**, not by how interesting the model is.
EN+FR is the product minimum throughout, so nothing English-only is ever
proposed as a Nemotron replacement.

## QUICK EXPERIMENT

**Q1. Oracle parity for the 120M pack.** *Gain: decisive — it is the only thing
standing between "it produces plausible text" and "it is correct", and every
claim in S13-3b depends on it. Effort: small, the oracle and the per-stage
harness exist. Axion: none, development host. Disk: already paid. EN only.
Reuse: total. Unknown: whether `tools/oracle/` covers this arch without work.*
**This is the top item and nothing downstream of it should be believed until it
passes.** Rule 3 is not suspended on a research branch.

**Q2. The 120M as the first-word control** (running). *Gain: high — it is the
experiment S13-5 was set up for, and the FastEmit finding makes either outcome
informative. Effort: none, the tool and the corpus exist. Axion: minutes.*

**Q3. Read NeMo-Speech.cpp.** *Gain: high and cheap — NVIDIA ships a first-party
C++ runtime for the exact GGUF checkpoints this repo targets, documented on the
model cards we use. Effort: a few hours of reading. Axion: none. Unknown:
everything; we have not looked. Not a threat assessment, an intelligence gap.*

**Q4. TDT v3 prefix ladder, EN and FR.** *Gain: moderate and bounded — NVIDIA's
own figures already say a TDT Parakeet under chunking reads WER 22.83 at 1.12 s
and 95.12 at 0.40 s, so this locates where usable output ends rather than
discovering a mode. Effort: small. Axion: under an hour. EN+FR: directly
relevant, this is the only multilingual offline pack we hold.*

## SMALL IMPLEMENTATION

**S1. The silence-run endpointer, in C.** *Gain: high and PRODUCT-facing.
Effort: ~50 lines plus a test. Axion: a short soak to show it costs nothing.
EN+FR: full — it is model-agnostic integer bookkeeping over decoded token ids
and works on the Nemotron we already qualified, with no new checkpoint and no
retrain.* NVIDIA's shipped parameters are `stop_history_eou: 800` ms,
`residue_tokens_at_end: 2`, `word_boundary_tolerance: 4`, snapping to a word
boundary. **This is the best value item on the list**: it buys voice-agent
turn-taking on the multilingual model, which the English-only EOU 120M cannot.

**S2. A bit-exactness oracle for the streaming cache.** *Gain: high for
confidence, zero for performance. Effort: small. Axion: none.* NeMo ships
`streaming.use_cache: false` — keep the left context as audio, re-encode every
chunk, drop the first `left_context` output frames. It needs no retrain and no
new model, and it is an independent check on the one piece of streaming state
mynah currently has no gate for. Rule 4 asks for exactly this.

## MEDIUM IMPLEMENTATION

**M1. Surface EOU as an event distinct from text.** *Gain: moderate, and it is
the contract half of S1. Effort: protocol plus server plus client. Unknown:
whether the event belongs on the same channel as deltas.* **Not before S1**:
the endpointer is what produces the event on the model we actually serve.

**M2. Capability descriptors instead of model-name special cases.** *Gain:
removes future special cases; buys nothing today. Effort: medium and it touches
the server. Rejection condition: if it would weaken an optimised path, it does
not ship.* S13-6 is a design study and stays one until S1 and Q1 are done.

## RESEARCH

**R1. `parakeet-unified-en-0.6b`.** NVIDIA's own answer to "one checkpoint,
both modes": offline 5.91 WER, 6.29 at 1.12 s, 8.44 at 0.16 s, 15.63 at 0.08 s,
buffered-only because the left context is recomputed per chunk. English-only,
`.nemo` only. *Worth understanding before anyone proposes a unified mode here.*

**R2. Streaming fine-tune feasibility for a multilingual Parakeet.** S13-2
showed the encoder difference is configuration, which makes this architecturally
plausible and says nothing about cost. *No training is started.*

## NOT CURRENTLY JUSTIFIED

**N1. Multitalker as an implementation.** NVIDIA's own design replicates the
whole cache-aware encoder state and forward pass **per active speaker**: "the
real batch size for inference is at most B * N". At our qualified C=80 that is
C=20 sessions at four speakers, plus a second model (a streaming Sortformer
diarizer) in the fleet. *It is a capacity decision before it is an architecture
decision, and the capacity is not there.* Keep it as the read-only study S13-8.

**N2. Replacing Nemotron with the EOU 120M.** English-only. Its documented WER
is 9.30 at 160 ms against Nemotron English 7.67 at 160 ms, and it emits no
punctuation or capitalisation — a product-visible difference our own corpus
scoring would hide, because `normalise()` strips both. **It is a control and
possibly a lightweight English lane. It is not a replacement.**

**N3. Optimising the attention cache "like NeMo".** NeMo caches pre-projection
activations and recomputes Q/K/V over the history every step; mynah already
caches projected K/V. Moving toward the reference here would be strictly more
work for identical numbers.

---

# S13-5 / Q2 — RESULT: the control emits earlier, and the gain splits in two

**EXPERIMENT.** Both models, **the same 120 clips** (40 per length class from the
qualification's own bank), the same onsets, the same committed tool
(`tools/eval/first_emission.py`), the same box, the same session, both int8,
each at its own pack-default preset. Measured in AUDIO time, so faster-than-real-time
CLI execution cannot contaminate it. Commit `b1f74db`, clean tree.

| | Nemotron 3.5 0.6B `[56,3]` | Parakeet Realtime EOU 120M `[70,1]` |
|---|---|---|
| speech → first non-blank, p50 | 860 ms | **570 ms** |
| p95 | 1990 ms | **1590 ms** |
| p99 | 2960 ms | **2460 ms** |
| blank decisions before the first | 17 | 16 |
| publication gate | 0 ms | **0 ms** |
| errors | 0 | 0 |

**Paired per clip**, which is what decides it:

- **The 120M is earlier on 112 of 120 clips, later on 1, tied on 7.**
  Sign test **z = +10.44**.
- Median **−300 ms**, mean −326 ms, p05 −700 ms, p95 +0 — it is essentially
  never materially later.

**FACT — R-15 reproduces on a second, independent checkpoint.** The 120M's
`publication_gate_ms` is **0.0 at every percentile** too. Two different models,
two different vocabularies, two different cadences, same answer: the
append-only contract withholds nothing. That closes component C harder than one
model could.

## Where the 300 ms comes from — both halves measured

**FACT — the model commits after fewer decisions.** Paired, the 120M needs
**2.2 fewer decoder decisions** on average before its first non-blank. At
80 ms per encoder frame in both models that is **≈ −176 ms**.

**FACT — and it publishes on a grid twice as fine.** `[70,1]` gives a 2-frame
chunk (160 ms); `[56,3]` gives 4 frames (320 ms). Over the 120 clips the first
non-blank lands on **9 distinct audio values for Nemotron and 17 for the 120M**,
with minima of **0.900 s and 0.600 s**. The average rounding-up cost of a
320 ms grid over a 160 ms one is 80 ms, and the residual of the paired mean
(−326 + 176 = **−150 ms**) is consistent with that plus interaction.

**OBSERVATION — so it is roughly half model and half cadence**, and neither half
was assumed: the decision count is a measured paired count, and the grid is a
measured set of distinct values.

## DECISION

The user's pre-registered fork — *"if Realtime EOU emits substantially earlier
on the same clips, architecture / training / decoder / context become the
primary suspects"* — **is taken, on the earlier branch.** A shared frontend or
runtime assumption is ruled out as the dominant cause, because the same
frontend and the same runtime produced both numbers.

The candidate causes are now specific and two of them are documented rather
than speculative:
1. **Training.** NeMo's shipped FastEmit lambda is `5e-3` for cache-aware
   streaming and `3e-2` for the EOU model's config — **6x** — and FastEmit is
   training-time only, which is exactly a "commit earlier" regulariser.
2. **Cadence.** `[70,1]` against `[56,3]`, measured above as ~150 ms of it.
3. Depth and width (17 x 512 against 24 x 1024) and the absence of the prompt
   projector remain untested as causes.

**NOT established.** Which of those produces the −176 ms decision-count half.
FastEmit is a documented, quantified candidate; it is not yet a measured cause,
and the honest next step is a Nemotron preset sweep (`[56,0]` is already known
to buy ~100 ms at a CER cost) rather than a claim.

## And the cost side, which must not be forgotten

The 120M's documented Open-ASR WER is **9.30 at 160 ms** against Nemotron
English's **7.67 at 160 ms**, it is **English-only**, and it emits **no
punctuation or capitalisation**. Our own corpus scoring would hide that last
one, because `normalise()` strips both. **This is a control and possibly a
lightweight English lane. It is not a replacement for a multilingual model.**

## Oracle parity (Q1), partial

`oracle == offline == streaming`, byte-identical, on **4 of 4** committed
English clips including the long one. **OBSERVATION and limitation:** the oracle reads
the SAME converted pack, so this validates the C implementation against the
reference math — it does not independently validate the conversion against
NeMo. The strong evidence that the conversion is right is different and
circumstantial: the transcripts are correct English matching the audio, and a
mis-assigned tensor does not produce that.

---

# S13-3c — Quality on the 498-clip bank, and whether the head start was paid for

**Validity check first.** The Nemotron arm of this run reads WER mean
**0.10634** and corpus **0.14618** on the 498 clips. Scoring the FROZEN C=80
reference transcripts gave 0.1063 and 0.1462. An independent streaming pass
reproduces the frozen evidence to four decimals, so the harness is sound before
any comparison is read out of it.

## FACT — the corpus-wide number is an artefact of the corpus

| subset | n | Nemotron WER mean / p50 | EOU 120M WER mean / p50 |
|---|---|---|---|
| all | 498 | 0.1063 / 0.0714 | 0.2290 / 0.1500 |
| **synthetic concatenations** | 149 | 0.1119 / 0.0789 | **0.4928 / 0.5106** |
| **original FLEURS** | **349** | **0.1040 / 0.0667** | **0.1164 / 0.0769** |

53 transcripts came back severely short (under half the reference's words).
**All 53 are `cat2` composed clips. Zero are original recordings.** Truncation
rate: composed **35.6 %**, original **0.0 %**. Duration is not the cause — the
truncated clips run 16.9-21.2 s while intact long clips run to 42.4 s.

## FACT — the truncation is EOU working, measured rather than assumed

On those 53 clips, the 120M's transcript scores **WER 0.6314** against the full
two-utterance reference and **0.1516 mean / 0.0769 p50** against the reference
PREFIX of the same length. Worked example, `long_0070_cat2.wav`,
`fleurs_ids [675, 1009]`: the hypothesis is the **first utterance, complete and
correct**, ending exactly where the second begins.

**DECISION.** The stress bank concatenates two turns into one file, and a model
trained to detect end-of-utterance correctly stops at the seam. **This bank is
structurally unfair to an EOU model**, and every quality comparison with it must
drop the composed clips or it measures the corpus. `earliness_cost.py` grew a
`--drop-composed` flag whose help text says why, so it cannot later be mistaken
for dropping inconvenient clips.

## RESULT — the paired question, on the 349 original recordings

| paired, 120M minus Nemotron | p50 | mean | better / worse / tied | z |
|---|---|---|---|---|
| **first non-blank, ms** | **−300** | −301 | **311 / 4 / 34** | **+17.30** |
| WER | +0.0000 | +0.0124 | 62 / 104 / 183 | −3.26 |
| CER | +0.0000 | +0.0079 | 68 / 117 / 164 | −3.60 |
| WER format-free | +0.0000 | +0.0148 | 57 / 110 / 182 | −4.10 |

So there **is** a quality cost and it is statistically real — about **1.2 WER
points on the mean** — but the **median delta is exactly zero**: on 183 of 349
clips the two models score identically.

## RESULT — and the head start is NOT what pays for it

Spearman **rho(Δ first non-blank, Δ WER) = +0.1198, z = +2.24**. Positive means
the clips where the 120M saves the most time are the clips where its WER delta
is LOWEST. The quartile table makes it concrete:

| bucket | n | Δt p50 | **Δ WER mean** |
|---|---|---|---|
| Q1, earliest | 97 | −500 ms | **−0.0001** |
| Q2 | 79 | −300 ms | +0.0131 |
| Q3 | 91 | −200 ms | +0.0110 |
| Q4, least early | 82 | −100 ms | **+0.0283** |

**On the quarter of clips where it gains half a second, it costs nothing at
all. On the quarter where it gains least, it is at its worst.** The relationship
runs opposite to a latency-for-accuracy trade.

**OBSERVATION, not a mechanism.** The reading consistent with this is that a
clip which is hard for a 120M model is hard in both ways at once — it commits
late *and* it is wrong — rather than earliness causing error. Two checkpoints
differ in depth, width, data, objective, cadence and vocabulary simultaneously,
so this is a property of the pair. It says the trade is worth investigating on
ONE model; it does not establish a cause.

By length class on the original subset: short 0.0757 → 0.0873, medium
0.1111 → 0.1240, long (17 clips) 0.3105 → 0.3263. Both models score badly on the
long originals, which is the FLEURS retake artefact recorded in M-6.

## Where this leaves the 120M

**A defensible lightweight English lane, and a better control than expected.**
It emits 300 ms earlier on 311 of 349 clips and costs ~1.2 WER points, with no
truncation on genuine single-utterance audio. It remains English-only, without
punctuation or capitalisation, and unqualified — no concurrency, no soak,
nothing promoted.

**And its EOU behaviour is now demonstrated on our own corpus**, by accident: it
ends the utterance at a turn boundary, correctly, 53 times out of 149
opportunities. That is the capability S13-7 was going to have to demonstrate
some other way.

---

# S13-1c — NVIDIA/NeMo-Speech.cpp, audited against this engine

**FACT.** The repository is real and first-party: `github.com/NVIDIA/NeMo-Speech.cpp`,
Apache-2.0, created 2026-07-15, **last push 2026-09-23** (today), 120 stars,
23 open issues. ~55k lines of C++17 on **ggml + llama.cpp submodules** (not
standalone). One release, `v0.1.0`, 2026-08-19; 15 commits on `main`.

**FACT.** Its scope is far broader than ours — ASR (the same four q8_0
checkpoints), plus Sortformer diarization, NMT, TTS and a full-duplex
speech-to-speech pipeline. `docs/server.md` states: *"NVIDIA NIM is the
supported production deployment path; this server is intended for local use and
direct integration."*

## FACT — they independently made our central architectural call

their `fastconformer.h` lines 114-121, verbatim:

> *"Follows NeMo's per-layer cur/next cache convention … **except we cache
> projected K/V rather than the pre-projection layer input.**"*

**OBSERVATION.** Two independent C/C++ ports of the same model family both
departed from the Python reference in the same place and for the same reason.
That retires N3 from the backlog with more confidence than our own reasoning
gave it, and it is worth recording in `docs/prior-art.md`: it also distinguishes
them from `mudler/parakeet.cpp`, which caches the pre-projection input.

## Where they are ahead, and what is worth taking

**The one cheap idea, and it is genuinely cheap.** They advance a `ring_head_`
modulo `cache_left_ctx` and read the arena at that offset
(their `cache_aware_encoder.cpp:487-491`). We do a `memmove` instead — verified in
`src/encoder.c`, `update_kv_cache()`: once the cache saturates, every chunk
memmoves `from_old` rows per layer, for K and for V.

Arithmetic for Nemotron at `[56,3]`: 56 rows x 1024 f32 = 229 KB per layer per
tensor, x 24 layers x 2 tensors = **~11 MB moved per chunk per stream**. At a
320 ms cadence and the qualified C=80 that is **~2.75 GB/s of memory traffic
that computes nothing.**

**HYPOTHESIS, and it connects to an open item.** S12-7c records that measured
CPU occupancy plateaus at 21.6-23.5 cores of 30 at saturation and *falls* under
overload, mechanism UNKNOWN — and M-6 found the same shape on a different model
and a different code path. Cache-shift bandwidth is a candidate that fits both:
it scales with streams, not with cores, and a bandwidth-bound step leaves cores
stalled rather than busy. **NOT established** — it is arithmetic plus a shape
match, and the rejection condition is explicit: if a ring head removes the
memmove and neither the core plateau nor the throughput ceiling moves, the
hypothesis is dead and is recorded as such.

Our `left % (r+1) == 0` invariant makes the ring *cleaner* than theirs: the wrap
always lands on a chunk boundary. And the bit-exactness gates are unchanged by
construction — same values, different addresses.

**Ahead on batching breadth, and the reason matters.** They micro-batch the
frontend, encoder, **predictor**, **joint**, fused TDT, VAD and PnC, with a
work-conserving scheduler and an **ingress cohort coordinator** that releases a
batch as soon as the declared wave has arrived instead of paying a timer
(their `batching.h:141-210`). We stack only the encoder.

**OBSERVATION, against our own measurement.** `server/prefork.h` already records
cross-worker batching as built, measured and REJECTED: requests that could batch
at B>=3 coincided **1.6 % of the time** within +/-0.25 ms against a 25 % bar.
Their design manufactures that coincidence with the cohort coordinator, on a GPU
where one wide graph is the whole point. So the part of their design that
survives our measurement is the **early release** idea — our scheduler already
knows how many slots are ACTIVE and due this period, and that number *is* the
cohort target — not wholesale cross-worker batching.

## Where this engine is ahead, factually

No thread pool of their own (`ggml_backend_set_n_threads` only); **one
`std::mutex` serializing every ggml graph**, which their own doc states
(their `runtime.h:196-198` and `asr-batching.md:131-134`); HTTP worker pool default **4**;
**no admission control** beyond `throw std::runtime_error("state arena is full")`;
**zero hand-written CPU SIMD** — 19 of 21 ggml patches are CUDA, one Metal, and
their AVX2-host issue #23 is an open **SIGILL** on the published v0.1.0 tarball,
which is exactly the failure our `src/dispatch.h` guard converts into exit 78.

**And the gates differ in kind, not degree.** Their batching parity is a
tolerance — `edit_limit = max(2, 5% of tokens)` (their `test_transducer_offline.cpp:175-177`)
— and their own fast GEMM header says *"results are not bit-identical to mmq"*.
Ours is identity: `==`/memcmp on every chunk of every stream plus the final K/V
and conv caches. They have a Python-oracle parity suite **only for the
diarizer**; there is no per-stage ASR oracle. There is **no statement anywhere**
that a transcript is invariant to thread count or ISA.

**FACT — they publish no performance numbers at all.** Not one RTF, latency,
memory or concurrency figure in the README, the 28 docs files or the release
notes. They ship the harness (`nemo-speech bench asr --concurrency 1,8,16,32`)
and no measured output.

**UNKNOWN.** Their actual CPU RTF, latency, memory and concurrency ceiling;
whether their cross-stream batching pays on CPU at all (the ring-cache fast path
is gated `use_gpu`); their WER on any corpus. Nothing was built or run.

## Backlog additions

**S1b (QUICK-ish, LOW cost) — ring-buffer the K/V cache.** Removes ~11 MB of
memmove per chunk per stream at the qualified operating point. Tests unchanged.
Rejection condition stated above. **This is now the top mechanical item.**

**S1c (LOW-MEDIUM) — release a batch when the due-slot count is reached**,
instead of always waiting `--batch-window-ms`. One function, and the A/B harness
exists.

**Q3 is answered**; what remains is a `docs/prior-art.md` entry, because a
first-party runtime in the same niche that independently reached the same cache
decision belongs in the prior-art record and not only in a research ledger.

---

# S13-1e — early batch release: REJECTED 2026-09-24, premise absent

**Question.** How much latency does the scheduler add by waiting after the
complete due cohort is already ready?

**FACT — zero in the qualified path, by construction.** The frozen C=80 fleet
is started with `--batch-window-ms 0` (`tools/bench/v2_qualify.sh`), which the
profile `configs/perf/axion-c4a-highcpu32-nemotron-streaming.json` justifies
by measurement (40 ms took the mean ready set 3.69 -> 5.85 at c=16 without
moving the ceiling; 80 ms cost a rung). `sched_collect()` in `server/sched.c`
returns immediately when the window is 0: a step runs the moment any slot is
ready. There is no window to close early.

**FACT — early release already exists when a window IS configured.** The
collect loop breaks on `ready >= live`, is woken by every arrival, and exempts
a stream's first chunk and any finalizing stream. The only residual gap from
NVIDIA's ingress cohort coordinator is the target: all LIVE slots rather than
the slots DUE this period, so a live but not-yet-due stream would hold the
window to its deadline. That matters only with a non-zero window.

**DECISION.** Rejected without code and without Axion time. Reopen only if a
non-zero batch window is ever re-adopted, and then as "target the due count",
not as a new mechanism.

**OBSERVATION carried forward (one worker, C=80, CACHE-RING-1 shift rung):**
ready_mean 1.52, 64 % of steps at B=1, step wall 20.9 / 36.9 / 52.6 / 69.1 ms
at B = 1..4 — a fixed cost of roughly 5 ms and ~16 ms per row, so wider
batches could save at most the fixed part.

---

# S13-5b — Is Nemotron's first token late because blank wins, or because the evidence is late?

**Question.** On the 120 h2h clips Nemotron's first non-blank comes ~300 ms of
audio after the EOU 120M's (S13-5). Does the token Nemotron eventually emits
already compete earlier (A: commitment), or does it not (B: evidence)?

**Correction carried in (read-only audit, 2026-09-24).** S13-5 split the gap
into "2.2 fewer decisions, ~176 ms" plus "a finer grid, ~150 ms". Those are not
independent: both models commit mostly on the FIRST frame of a newly arrived
chunk (frame mod chunk = 0 on 90/120 Nemotron and 117/120 EOU clips), so
decision count and grid overlap and must not be added.

**Method.** Raw `MYNAH_ASR_TRACE_RNNT` traces of the same 120 clips, both models,
exactly `first_emission.py`'s command (int8, default lookahead, 4 threads), on
the box, 2026-09-24, commit `7accee0`; pack hashes 2f5e1434 (Nemotron) and
5c533297 (EOU). The trace already carries, per decision, blank score, best
non-blank id/score/rank and `margin_nb` (logit difference = log-probability
difference). No instrumentation was added. Latency is measured in AUDIO
CONSUMED at the decision (`audio_s`), never as frames x 80 ms: a decision is
published when its chunk completes. Rule registered in the analyzer before it
ran: `.work/evidence/ttfp-trace-20260924/ttfp_ab.py` (untracked, with traces).

**FACT — the baseline reproduces.** Speech -> first non-blank p50: Nemotron
860 ms, EOU 580 ms, paired difference p50 300 ms. 90 of 120 clips are
contested (EOU first), 5 go the other way, 25 tie.

**FACT — the eventual token is usually already there, one to three decisions
early, a few nats under blank — in BOTH models.**

| decisions before the first emission | Nemotron: eventual token is best non-blank | margin under blank (p50) | EOU: eventual token is best non-blank | margin (p50) |
|---|---|---|---|---|
| 1 | 71.7 % | 2.78 | 62.5 % | 3.30 |
| 2 | 55.8 % | 2.21 | 53.3 % | 2.48 |
| 3 | **41.7 %** | 2.59 | **16.7 %** | 3.82 |
| 4 | **26.7 %** | 4.09 | **15.8 %** | 3.82 |

In the contested window (from EOU's first frame to Nemotron's), Nemotron's
eventual token is best non-blank within tau of blank on 11 / 24 / 47 / 74 % of
clips for tau = 0.5 / 1 / 2 / 4. No window lies in the extreme |blank| > 100
regime.

**RESULT — mostly A, and not Nemotron-specific in kind.** "Right token, blank
ahead by ~2-3 nats" is the normal state before a crossing in BOTH models; what
differs is duration: Nemotron's eventual token is competitive three and four
decisions ahead far more often (42 / 27 % against 17 / 16 %). The evidence is
there; the commitment comes later.

**Counterfactual (exact before the first emission).** Until the first token
the predictor sits at SOS and the encoder does not depend on the decoder, so
a blank bias `delta` applied only until the first token can be replayed on the
stored logits exactly: first commit = first decision with margin_nb < delta.

| delta | clips moved | first token CHANGED | audio saved p50 / mean | new speech -> first p50 | at or before EOU |
|---|---|---|---|---|---|
| 0 | — | — | — | 860 ms | — |
| 1 | 27 | 2 | 0 / 65 ms | 820 ms | 14 / 120 |
| 2 | 55 | 4 | 0 / 128 ms | 745 ms | 28 / 120 |
| 3 | 73 | 7 | 150 / 180 ms | 710 ms | 41 / 120 |
| 4 | 83 | 10 | 300 / 213 ms | 680 ms | 49 / 120 |

About two thirds of the 300 ms gap is recoverable with delta = 3-4 at a cost
of 6-8 % of clips starting with a DIFFERENT first token. What the replay cannot
say: anything after the first token (predictor state changes), hence WER/CER.

**NOT established.** Why Nemotron holds blank longer. FastEmit weight, context
geometry and the prompt projector remain candidates; none is shown by this.

**NEXT (plan only, nothing built).** A default-off debug knob
`MYNAH_ASR_BLANK_BIAS=delta=<x>,scope=first` at the two argmax sites, gated by
(i) delta 0 byte-identical and (ii) the live first frame equal to this replay
on all 120 clips; then delta in {2, 3, 4} scored for WER/CER on the 349
original recordings with the existing scorers (`streaming_metrics`,
`earliness_cost.py` paired against delta 0) and first-word correctness with
`partial_quality.py`. The first-token identity changes above are the risk to
watch: a biased commit that emits a DIFFERENT token is the hallucination case.
