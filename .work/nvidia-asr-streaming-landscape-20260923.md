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
