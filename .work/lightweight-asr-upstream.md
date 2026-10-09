# Lightweight ASR: upstream facts (verified 2026-10-09)

Scope: the "small/extensible" tier candidates next to the cache-aware FastConformer-RNNT
0.6B baseline (`nvidia/nemotron-speech-streaming-en-0.6b`, plus the multilingual
`nvidia/nemotron-3.5-asr-streaming-0.6b`):

- `nvidia/parakeet_realtime_eou_120m-v1` (cache-aware FastConformer-RNNT with an `<EOU>` token)
- `nvidia/canary-180m-flash` (FastConformer encoder + Transformer decoder, AED)

Method. Every row below was checked against a primary source on 2026-10-09: the HF model
card (raw `README.md`), the `.nemo` archive itself (both checkpoints were downloaded and
their `model_config.yaml`, tokenizers and `model_weights.ckpt` inspected), the NeMo source
tree (`NVIDIA-NeMo/NeMo` `main` at `09e7f98`, 2026-10-09, `nemo/package_info.py` = 3.1.0),
NeMo release tags (via the GitHub API), papers and dataset cards. Community material is
labelled **secondary**. **UNKNOWN** means no primary source states it.

Companion file: `.work/lightweight-asr-ft.md` (Italian adaptation, datasets, experiment plan).

---

## 1. `nvidia/parakeet_realtime_eou_120m-v1`

Sources: card https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 (repo created
2025-10-10, last modified 2025-12-03); `.nemo` 460,062,720 bytes; NeMo
`examples/asr/asr_eou/README.md`, `examples/asr/asr_eou/speech_to_text_rnnt_eou_train.py`,
`nemo/collections/asr/data/audio_to_eou_label_lhotse.py`,
`nemo/collections/asr/models/asr_eou_models.py`, `tools/nemo_forced_aligner/align_eou.py`,
`scripts/asr_eou/*`, `examples/asr/conf/asr_eou/*.yaml`.

### 1.1 What the checkpoint is

| item | value | source |
|---|---|---|
| params | card: 120M. Counted from `model_weights.ckpt`: encoder 109.52M + decoder (pred net) 3.94M + joint 1.40M = **114.9M** float32 | card; checkpoint count |
| class saved in `.nemo` | `target: nemo.collections.asr.models.EncDecRNNTBPEModel` (plain RNNT, not the EOU subclass); `nemo_version: 2.6.0rc0` | `model_config.yaml` |
| encoder | ConformerEncoder, 17 layers, d_model 512, 8 heads, ff x4, conv k=9, `conv_norm_type: layer_norm`, `conv_context_size: causal`, `dw_striding` x8 with `causal_downsampling: true`, 128 mel, rel_pos, `use_bias: false` | `model_config.yaml` |
| attention | `att_context_size: [70, 1]`, `att_context_style: chunked_limited` (single context, no multi-lookahead list) | `model_config.yaml` |
| streaming | cache-aware ("cache-aware streaming FastConformer [2] with 17 encoder layers (attention context = [70,1])"). [70,1] = 1 frame lookahead (80 ms), 2-frame (160 ms) chunks per the NeMo EOU README | card; `examples/asr/asr_eou/README.md` |
| decoder | RNNTDecoder, pred_hidden 640, 1 LSTM layer; joint 640; `num_classes: 1026` (+ blank) | `model_config.yaml` |
| tokenizer | SentencePiece, **1026 pieces** = 1024 English BPE pieces + `<EOU>` (id 1024) + `<EOB>` (id 1025); lower-case, no punctuation | tokenizer `.model` inspected |
| training loss | RNNT with FastEmit `fastemit_lambda: 0.03` | `model_config.yaml` |
| saved optimizer | AdamW, Noam, lr 0.113, warmup 2500 (training-time metadata only) | `model_config.yaml` |
| language | English only, no PnC | card |
| output | text with optional `<EOU>` ("what is your name<EOU>"); may be empty without speech; input >= 160 ms, 16 kHz mono | card |
| `<EOB>` | present in vocabulary ("end-of-backchannel"); the card does not mention it. NeMo README: `<EOB>` is optional; configs default `ignore_eob_label: true` (EOB treated as EOU) | tokenizer; NeMo README |
| runtime | card: "NeMo 2.5.3+"; supported HW listed Ampere, Blackwell, Hopper, Volta (Lovelace/L4 not listed, but nothing arch-specific in the model) | card |

### 1.2 Reported numbers (card)

WER, HF Open ASR Leaderboard sets, **160 ms streaming**, open_asr_leaderboard normalizer:

| Avg | AMI | Earnings22 | Gigaspeech | LS clean | LS other | SPGI | Tedlium | Voxpopuli |
|---|---|---|---|---|---|---|---|---|
| 9.30 | 15.62 | 15.76 | 13.31 | 3.61 | 7.79 | 3.79 | 5.48 | 9.07 |

EOU latency on TTS-generated DialogStudio audio with 3 s appended silence: p50 160 ms,
p90 280 ms, p95 320 ms. The card warns real-world numbers will vary. No EOU
precision/recall/false-cutoff numbers are published (UNKNOWN).

### 1.3 Training data (card)

AMI; DialogStudio (task-oriented subset "with commercial license"); Granary; Google Speech
Commands; LibriTTS; 10,000 h of "human-transcribed NeMo ASR Set 3.0" (LibriSpeech 960 h,
Fisher, NSC Part 1, VCTK, Europarl-ASR, MLS, MCV 7.0). Collection "Hybrid: Human,
Synthetic" (some audio is TTS); labels "Hybrid: Human, Synthetic" (some ASR-generated).
Exact hours per source, and **the exact EOU-labelled subset and its label pipeline used
for this checkpoint: UNKNOWN** (not published). The DialogStudio TTS eval set is not
released.

### 1.4 How EOU labels are produced (public recipe, NeMo main / v3.0.0)

The recipe IS public, as of NeMo **v3.0.0** (tag 2026-08-07) and `main`; it is **absent
from v2.5.3, v2.6.0, v2.7.0 and v2.7.3** (`examples/asr/asr_eou` and
`nemo/collections/asr/data/audio_to_eou_label_lhotse.py` return 404 at those tags). Facts
from `examples/asr/asr_eou/README.md` and the code:

1. **Tokenizer**: `scripts/asr_eou/tokenizers/add_special_tokens_to_sentencepiece.py` appends
   `<EOU>` and `<EOB>` to an existing SentencePiece model (1024 -> 1026). The dataset class
   checks that the last two ids are exactly those strings (`_check_special_tokens`).
2. **Timestamps**: `tools/nemo_forced_aligner/align_eou.py pretrained_name=nvidia/parakeet-ctc-0.6b`
   (any **CTC** NeMo model; English in the example) forced-aligns each utterance and writes
   `sou_time` / `eou_time` into the manifest. Utterances must contain a single utterance
   with no leading/trailing silence ("otherwise the model's EOU prediction accuracy will be
   degraded").
3. **Optional backchannel**: `scripts/asr_eou/add_eob_labels.py` marks `is_backchannel`
   from a hard-coded phrase list ("uh-huh", "yeah", "okay", ... — English).
4. **Frame labels** (`LhotseSpeechToTextBpeEOUDataset`): pad 0 = non-speech, 1 = speech,
   2 = EOU (last speech frame), 3 = EOB; the text target gets `<EOU>` appended
   (`add_eou_to_text: true`). Random silence padding (`random_padding`, e.g. prob 0.9,
   min post-pad 3 s, max 6 s) and noise/gain augmentation; `speed`, `time_stretch`,
   `random_segment` augmentations are refused because they would break the timestamps.
5. **Blend**: lhotse `input_cfg` with e.g. weight 0.1 plain ASR + 0.9 EOU data "such that ASR
   WER is not significantly degraded".
6. **Init**: `speech_to_text_rnnt_eou_train.py ++init_from_nemo_model=...` copies encoder,
   pred-net LSTM, joint enc/pred projections, and the embedding/joint output rows of the
   original vocabulary; the two new rows are initialised by `token_init_method`
   (`constant` with bias **-1000** by default, i.e. EOU starts "never emitted").
   `init_from_pretrained_nemo` requires the new vocab to be exactly old vocab **+ 2**.
7. Recommended `fastemit_lambda: 3e-2`, `att_context_size: [70,1]`; the README example
   starts from `nemotron-speech-streaming-en-0.6b` (xlarge config); a `..._large.yaml`
   config ("~120M", d_model 512, 17 layers) and an adapter config also exist.
8. Evaluation: `speech_to_text_eou_eval.py` reports WER plus EOU latency, early-cutoff rate,
   miss rate; `scripts/asr_eou/generate_noisy_eval_data.py` pads/noises clean audio.

Consequence for a **tokenizer swap** (e.g. Italian SPE): the `.nemo` is a plain RNNT model, so
`change_vocabulary()` works, but it **re-creates the whole RNNT decoder (prediction net) and
joint from scratch** (`rnnt_bpe_models.py:271-345`): EOU behaviour is lost entirely and must be
re-learned with EOU-labelled target-language data through the pipeline above. The
"+2 rows" init path cannot be used (the base vocab changes), and `add_eob_labels.py` and the
default aligner are English. An Italian CTC aligner exists:
`nvidia/stt_it_fastconformer_hybrid_large_pc` (CC-BY-4.0; hybrid RNNT/CTC — usable by NFA
through its CTC head; whether `align_eou.py` accepts a hybrid model directly is **UNKNOWN**,
not tested).

### 1.5 License — NVIDIA Open Model License

Card metadata: `license: other`, `license_name: nvidia-open-model-license`, link
https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
(page version read: "October 24, 2025"). Card: "ready for commercial/non-commercial use".
Agreement terms (primary, summarised):

- perpetual, worldwide, non-exclusive, no-charge, royalty-free licence to use, reproduce,
  distribute, create **Derivative Models** (fine-tunes included) and distribute them;
  commercial use allowed; NVIDIA claims no ownership of outputs or of your derivatives;
- redistribution must carry the notice "Licensed by NVIDIA Corporation under the NVIDIA
  Open Model License";
- rights terminate if you bypass/disable/reduce the efficacy of the model's guardrails
  without a substantially similar replacement, or if you start IP litigation alleging the
  model infringes;
- use must also follow NVIDIA's Trustworthy AI terms; no trademark rights.

Not a standard OSI/CC licence: the guardrail and Trustworthy-AI clauses are extra
obligations compared with CC-BY-4.0. Whether the Trustworthy AI terms impose anything
concrete for an ASR model: UNKNOWN (referenced, not reproduced).

### 1.6 Intended use

Voice-agent pipelines (NeMo Voice Agent, `stt.model: nvidia/parakeet_realtime_eou_120m-v1`),
streaming ASR with integrated endpointing. Note that the NeMo voice agent also has a
separate, silence-based endpointer (`nemo/agents/voice_agent/pipecat/services/nemo/`), so the
`<EOU>` token is one of two mechanisms.

---

## 2. `nvidia/canary-180m-flash`

Sources: card https://huggingface.co/nvidia/canary-180m-flash (created 2025-03-11, modified
2025-03-18); `.nemo` 736,665,600 bytes (`nemo_version: 2.3.0rc0`); NeMo
`nemo/collections/asr/models/aed_multitask_models.py`,
`examples/asr/asr_chunked_inference/aed/*`,
`nemo/collections/asr/parts/submodules/aed_decoding/aed_batched_streaming.py`; docs
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/streaming_decoding/canary_chunked_and_streaming_decoding.html ;
issue https://github.com/NVIDIA-NeMo/NeMo/issues/14886, PR #14765.

### 2.1 What the checkpoint is

| item | value | source |
|---|---|---|
| params | card: 182M. Checkpoint tensors: encoder 109.57M, `encoder_decoder_proj` 0.53M, `transf_decoder` 73.61M, `log_softmax` 5.38M (output layer, weight-tied to the decoder token embedding by NeMo) -> ~183.7M unique | card; checkpoint count; `aed_multitask_models.py` weight tying |
| encoder | FastConformer, 17 layers, d_model 512, 8 heads, conv k=9, `conv_norm_type: batch_norm`, `dw_striding` x8, `causal_downsampling: false`, **`att_context_size: [-1, -1]`** (full, bidirectional), 128 mel | `model_config.yaml` |
| decoder | Transformer, **4 layers**, hidden 1024, inner 4096, 8 heads, max_seq 1024; `transf_encoder.num_layers: 0` | `model_config.yaml` |
| prompt | `prompt_format: canary2`; slots `source_lang`, `target_lang`, `pnc`, `itn`, `timestamp`, `diarize`, `emotion`, `decodercontext` | `model_config.yaml` |
| tokenizer | `type: agg` (aggregate/concatenated): `spl_tokens` 1152 pieces + `en`, `de`, `es`, `fr` 1024 each = **5248 classes** (`head.num_classes: 5248`); `CanaryTokenizer` | config; tokenizers inspected |
| special tokens | `spl_tokens` contains task/control tokens (`<|startoftranscript|>`, `<|pnc|>`/`<|nopnc|>`, `<|itn|>`, `<|timestamp|>`/`<|notimestamp|>`, `<|diarize|>`, emotion tokens), **~180 ISO language tokens including `<|it|>`**, 900 numeric timestamp tokens `<|0|>`...`<|899|>`, and `<|spltoken0..29|>` spares | `spl_tokens` model inspected |
| languages | ASR: en, de, es, fr. AST: en<->de, en<->es, en<->fr | card |
| PnC | yes, toggled by prompt (`pnc: yes/no`) | card |
| input | 16 kHz mono; designed for < 40 s; utterances < 1 s are zero-padded to 1 s in their eval | card |
| training | 85K h (31K public, 20K Suno, 34K in-house); EN 25.5K, DE 2.5K, ES 1.4K, FR 1.8K public; 219K steps on 32x A100 80GB, 2D bucketing + OOMptimizer; script `examples/asr/speech_multitask/speech_to_text_aed.py`, config `examples/asr/conf/speech_multitask/fast-conformer_aed.yaml` | card |
| license | CC-BY-4.0, commercial use allowed (attribution required) | card metadata + text |

### 2.2 Timestamps: native tokens, not a runtime forced aligner

`EncDecMultiTaskModel` has two timestamp paths (`aed_multitask_models.py:1095-1108`): if the
`.nemo` bundles an auxiliary CTC model (`timestamps_asr_model`), timestamps come from
`get_forced_aligned_timestamps_with_external_model` (Viterbi forced alignment — the
Canary-1B-v2 route); otherwise `process_aed_timestamp_outputs` parses timestamp tokens the
decoder emits. The 180M `.nemo` contains **no auxiliary model** (only weights, config and
tokenizers) and its special-token vocabulary has 900 frame-index tokens, so its timestamps
are **decoder-predicted tokens** (card: "experimental"; F1 at 200 ms collar 93.48 / 91.38 on
LibriSpeech test-clean / test-other). Card recommends `chunk_len_in_secs=10` for timestamps
on audio > 10 s.

### 2.3 Reported WER / BLEU (card)

English, Open ASR Leaderboard sets, no PnC, whisper-normalizer, batch 128, greedy:

| RTFx (A100) | AMI | GigaSpeech | LS clean | LS other | Earnings22 | SPGI | Tedlium | Voxpopuli |
|---|---|---|---|---|---|---|---|---|
| 1233 | 14.86 | 10.51 | 1.87 | 3.83 | 13.33 | 2.26 | 3.98 | 6.35 |

RTFx H100: 2041. MLS test: DE 4.81, ES 3.17, FR 4.75. MCV-16.1 test: EN 9.53, DE 5.94,
ES 4.90, FR 8.19. FLEURS BLEU: En->De 28.18, En->Es 20.47, En->Fr 36.66, De->En 32.08,
Es->En 20.09, Fr->En 29.75. Noise robustness (LS clean + white noise) SNR 10/5/0/-5:
3.23/5.34/12.21/34.03.

**Inconsistency in the card (verified):** the YAML `model-index` metadata block reports
different numbers (LS other 2.87, SPGI 1.95, MCV-16.1 EN 6.99 / DE 4.03 / ES 3.31 / FR 5.88,
FLEURS En->De BLEU 32.27, En->Fr 41.22, ...) from the prose tables (3.83, 2.26, 9.53/5.94/4.90/
8.19, 28.18, 36.66). The metadata block looks copied from a larger Canary card. Use the
prose tables; do not quote the model-index values.

### 2.4 Streaming verdict

(a) **Is canary-180m-flash itself cache-aware / incremental? No.** Its encoder has
`att_context_size: [-1, -1]` (full bidirectional attention), non-causal subsampling,
`batch_norm` convolution (not causal), and no cache-aware config; the AED decoder
cross-attends over the whole encoder output. The card only offers offline transcription and
"chunked inference" for long audio.

(b) **Official NVIDIA streaming Canary variant? None found.** HF `nvidia` org search for
"canary" returns canary-1b, canary-1b-flash, canary-180m-flash, canary-1b-v2, canary-qwen-2.5b;
none is cache-aware. NVIDIA's streaming multilingual answer is a different family:
`nvidia/nemotron-3.5-asr-streaming-0.6b` (cache-aware FastConformer-RNNT with prompt, HF
created 2026-05-15, card release 2026-06-04, 40 language-locales incl. **it-IT**, contexts
[56,0]/[56,1]/[56,3]/[56,6]/[56,13] = 80 ms ... 1.12 s, licence **OpenMDW-1.1**, card FLEURS
it WER 4.25 at 1.12 s chunk; requires "NeMo 26.06"). That is 0.6B, not the small tier.

(c) **Official streaming recipe for AED: decoding policies on a re-encoded sliding window.**
`examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py` (PR #14765,
merged 2025-10-13; first release containing it: **v2.6.0**, absent in v2.5.3) implements
**Wait-k** (arXiv 1810.08398) and **AlignAtt** (arXiv 2305.11408; emit while the most-attended
encoder frame is >= `alignatt_thr` frames, default 8, from the buffer end) via
`AEDStreamingDecodingConfig` (`waitk_lagging` default 2, `max_tokens_per_alignatt_step` 30,
hallucination detector on). Latency is reported as LAAL. Docs list canary-1b-flash and
canary-1b-v2 as supported; 180m-flash is not listed but uses the same class (expected to run,
**not verified**). No official WER/latency table is published for these policies; one
community report in issue #14886 (secondary) saw ~12% WER on LibriSpeech test-other and asked
whether streaming-specific training is needed — unanswered in the issue. There is **no
NVIDIA paper** reporting Canary streaming accuracy that I could find; third-party
simultaneous-ST work (e.g. IWSLT 2026 submissions, arXiv 2606.03948 / 2606.03967) uses
AlignAtt-style policies on offline models (secondary for our purpose).

(d) **Does it re-run the encoder on overlapping windows? Yes — quantified from the code.**
Per step the script feeds the whole buffer `[left | chunk | right]` to
`asr_model(input_signal=buffer.samples, ...)` (`speech_to_text_aed_streaming_infer.py:370-378`): the encoder re-encodes left context + new chunk + right
lookahead every chunk; only the transformer decoder's self-attention memories
(`decoder_mems_list`) are carried between steps. Encoder cost per second of new audio at
steady state = (L + C + R) / C:

| setting (L-C-R, s) | theoretical latency C+R | encoder recompute factor |
|---|---|---|
| 10-1-0.5 (script's "1.5 s latency" recommendation) | 1.5 s | **11.5x** |
| 10-2-2 (script defaults) | 4 s | **7x** |
| cache-aware FastConformer [70,1] (for comparison) | 0.16 s | 1x (each frame encoded once; attention reads cached K/V) |

On top, self-attention inside each window is quadratic in window length (FastConformer full
attention over 11.5 s = ~144 encoder frames per pass). Chunked (long-form, non-streaming)
inference `speech_to_text_aed_chunked_infer.py` splits audio into **non-overlapping**
segments and concatenates transcripts (docs) — no recompute but no streaming either.
Verdict: Canary streaming in NeMo is **buffered/simultaneous decoding over an offline
encoder**, not comparable to cache-aware streaming in cost or in latency floor; making the
180M truly incremental would need retraining the encoder with limited-context attention
(cache-aware training) — **no NVIDIA recipe exists for a cache-aware AED**.

---

## 3. Evaluation sets for EN/DE/ES/FR (identical audio + normalization)

| set | languages | split / size | licence | notes | source |
|---|---|---|---|---|---|
| FLEURS | en_us, de_de, es_419, fr_fr, it_it | test: en 647, de 862, es 908, fr 676, it 865 utts; it test = 3.52 h; test tarballs en 290 MB, de 569 MB, es 582 MB, fr 349 MB, it 658 MB (16 kHz wav) | CC-BY-4.0 | read Wikipedia (FLoRes) sentences, ~3 recordings per sentence; `raw_transcription` + `transcription` (lower-cased, unpunctuated) | https://huggingface.co/datasets/google/fleurs (card + `data/<lang>/*.tsv`, hours computed from `num_samples`) |
| MLS | de, es, fr, it (+ en) | it test 1262 utts / 5.27 h; dev 5.18 h | CC-BY-4.0 | LibriVox audiobooks; transcripts are auto-aligned book text (paper: human check 5.37% WER on it test) | https://www.openslr.org/94/ ; arXiv 2012.03411 Table 2 |
| LibriSpeech test-clean / test-other | en | 5.4 h / 5.1 h | CC-BY-4.0 | audiobooks | https://www.openslr.org/12 |
| Common Voice test | all | CV 27.0 test buckets: en 16,403, de 16,209, es 15,907, fr 16,207, it 15,183 clips | CC0-1.0 | read prompts, mp3 48 kHz -> resample; speaker-disjoint splits; download via Mozilla Data Collective only (see ft file) | https://github.com/common-voice/cv-dataset `datasets/scripted-speech/cv-corpus-27.0-2026-09-11.json` |
| Open ASR Leaderboard multilingual pack | fleurs_{en,de,es,fr,it}, mcv_{en,de,es,fr,it}, mls_{es,fr,it} (no mls_de in the pack) | test only; fleurs_it 865, mcv_it 8951, mls_it 657 utts (a subset of MLS test), mcv_en 15531, mcv_de 13511, mcv_es 13221, mcv_fr 14760 | dataset card says **CC-BY-NC-SA-4.0** for the packaged repo (stricter than the sources) | ready-made identical audio for all models; CV version used: UNKNOWN | https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-multilingual-datasets |
| Open ASR Leaderboard English pack | ami, earnings22, gigaspeech, librispeech, spgispeech, tedlium, voxpopuli, common_voice | test | per-source (several non-commercial) | what the EOU 120M and Canary cards report on | https://huggingface.co/datasets/hf-audio/open-asr-leaderboard |

Normalisation used by the leaderboards (primary: https://github.com/huggingface/open_asr_leaderboard
`normalizer/data_utils.py`): English = Whisper `EnglishTextNormalizer`; multilingual =
`MultilingualNormalizer`, a subclass of Whisper's `BasicMultilingualTextNormalizer` built with
`remove_diacritics=False`, which with `lang=` also maps digits to words via `num2words` and
removes per-language fillers (filler list currently empty). The EOU 120M card links this
normalizer; the Canary 180M card says "whisper-normalizer" (PyPI) for English. Recommendation:
score all three models with this exact module (EN normalizer for en, multilingual one with
`lang=` for de/es/fr/it) on the same audio files; report both raw-PnC-stripped and normalised
WER only if needed. Canary should be decoded with `pnc: no` (or PnC then normalised) and the
correct `source_lang`/`target_lang`; the EOU model must have `<EOU>`/`<EOB>` stripped before
scoring.

---

## 4. Economics

- **Vast.ai L4 price (secondary):** getdeploying.com tracks Vast.ai L4 on-demand from
  ~$0.27-0.32 per GPU-hour and interruptible/spot from ~$0.13; cross-provider on-demand
  median ~$0.84/h as of 2026-09-28 (https://getdeploying.com/gpus/nvidia-l4). Marketplace
  prices move daily; budget **$0.30-0.50/h on-demand** for a planning range.
- **Published fine-tuning throughput for ~100-200M FastConformer models: none from NVIDIA.**
  Cards give only pre-training scale (Canary 180M: 219K steps on 32x A100). The closest
  published datapoint is third-party: Typhoon ASR Real-time (arXiv 2601.13044, a 115M
  FastConformer-Transducer initialised from the English Large checkpoint) reports ~11,000 h
  of Thai, 1 epoch, on 2x H100 in ~17 h, i.e. **~320 audio-hours per H100-hour** (derived).
  Translating to an L4 is an estimate only (L4 bf16 dense peak ~1/8 of H100, memory
  bandwidth ~1/11): plan on **~25-60 audio-hours per L4-hour** for a 115-185M model, to be
  replaced by measurement. At that rate a 40 h subset x 20 epochs = 800 audio-hours =
  ~13-32 L4-hours = **~$4-16** at $0.30-0.50/h; 5 h / 20 h subsets scale linearly.
- bf16 on L4 (sm_89, Ada) is supported in hardware; NeMo configs use `trainer.precision`
  (`bf16-mixed` in `fast-conformer_aed.yaml`; the EOU configs default to `32`). NeMo's RNNT
  loss defaults to `force_float32: True` (loss computed in fp32 even under AMP;
  `nemo/collections/asr/losses/rnnt.py:85`).

---

## 5. Fact table

| claim (as given in the brief or commonly repeated) | verified? | source |
|---|---|---|
| EOU 120M is FastConformer-RNNT, 17 layers, att_context [70,1], cache-aware | YES | card; `.nemo` config |
| EOU 120M has 120M params | ~YES (card 120M; checkpoint counts 114.9M) | checkpoint |
| EOU 120M latency 80-160 ms; EOU p50/p90/p95 160/280/320 ms | YES (card; TTS DialogStudio + 3 s silence) | card |
| EOU 120M has `<EOU>` and `<EOB>` tokens | YES (ids 1024/1025 in a 1026-piece SPE; card mentions only `<EOU>`) | tokenizer |
| EOU training recipe/data public | Recipe YES (NeMo v3.0.0+/main); the actual EOU training data mix for this checkpoint NO/UNKNOWN | NeMo `examples/asr/asr_eou`; card |
| EOU labels come from forced alignment | YES for the public recipe (NFA `align_eou.py`, CTC model); for the released checkpoint UNKNOWN | NeMo README |
| EOU 120M English-only, no PnC | YES | card |
| NVIDIA Open Model License allows commercial use, fine-tunes, redistribution | YES, with notice requirement, guardrail and litigation termination clauses, Trustworthy AI terms | licence page (v. 2025-10-24) |
| EOU 120M WER avg 9.30 (160 ms streaming) | YES | card |
| EOU 120M fine-tunable with NeMo | YES (plain `EncDecRNNTBPEModel`; EOU recipe script `speech_to_text_rnnt_eou_train.py`) | `.nemo` target; NeMo |
| tokenizer swap keeps EOU | **NO** — `change_vocabulary` rebuilds pred-net + joint; EOU must be retrained | `rnnt_bpe_models.py` |
| card says NeMo 2.5.3+ | YES for inference; EOU **training** code only from v3.0.0 | card; tags |
| Canary 180M: 182M params, 17 enc / 4 dec layers | YES | card; config |
| Canary 180M languages en/de/es/fr, AST en<->x | YES | card |
| Canary 180M lacks Italian | Text tokenizer: YES (no `it` sub-tokenizer). Prompt: `<|it|>` language token already exists in `spl_tokens` | tokenizers |
| Canary 180M timestamps via forced aligner | **NO** — native decoder timestamp tokens (900 `<|n|>` tokens, no bundled CTC model); NFA route is the Canary-1B-v2 path | `aed_multitask_models.py`; `.nemo` contents |
| Canary 180M licence CC-BY-4.0 | YES | card |
| Canary 180M is streaming / cache-aware | **NO** (`att_context_size [-1,-1]`) | config |
| an official streaming Canary exists | **NO** (NVIDIA streaming multilingual = Nemotron 3.5 ASR Streaming 0.6B, RNNT) | HF org listing; nemotron-3.5 card |
| NeMo has `speech_to_text_aed_streaming_infer.py` with Wait-k/AlignAtt | YES (v2.6.0+) | NeMo |
| AED streaming re-encodes overlapping windows | YES — whole [L|C|R] buffer per chunk; 11.5x at 10-1-0.5 | script source |
| Granary IT guide fine-tunes Canary 1B-Flash | YES (Canary-1b-flash; v2 mentioned as alternative) | Discussion #14758 |
| Open ASR Leaderboard uses Whisper EnglishTextNormalizer | YES (EN); multilingual uses BasicMultilingualTextNormalizer(remove_diacritics=False)+num2words | open_asr_leaderboard repo |
| Vast.ai L4 ~$0.3/h | secondary only: $0.27-0.32 on-demand, $0.13 spot | getdeploying.com |
