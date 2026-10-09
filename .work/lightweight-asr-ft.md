# Lightweight ASR: low-data language adaptation (Italian first)

Question: can the small tier (`nvidia/canary-180m-flash`, `nvidia/parakeet_realtime_eou_120m-v1`)
be adapted cheaply to Italian with 5 / 20 / 40 h of open data on one NVIDIA L4 24 GB, and
what does the official NVIDIA recipe actually do? Facts verified 2026-10-09 against primary
sources (NeMo source at `main` `09e7f98`, release tags, HF cards and the `.nemo` archives,
dataset cards). Model facts and the streaming analysis live in
`.work/lightweight-asr-upstream.md`. **UNKNOWN** = no primary source says it.

---

## 1. The official Granary + Italian guide (NeMo Discussion #14758)

Source: https://github.com/NVIDIA-NeMo/NeMo/discussions/14758 ("Guide to Fine-tune Nvidia
NeMo models with Granary Data", authors @Ssofja, @nithinraok; also published as
https://nvidia-nemo.github.io/blog/2025/08/13/granary-data-for-fine-tune/). Read in full via
the GitHub API; the discussion has **no comments**.

What it does:

| aspect | what the guide says |
|---|---|
| base checkpoint | `nvidia/canary-1b-flash` (883M; 32 enc / 4 dec layers, hidden 1024). `canary-1b-v2` mentioned as an alternative. The guide's own table lists `canary-180m-flash` params (17 enc, 4 dec, `asr_enc_hidden` 512, `lm_dec_hidden` 1024, max_seq 1024) as the values to set if you use it — i.e. the same recipe is presented as applicable to the 180M |
| goal | one checkpoint doing IT ASR, IT->EN AST, EN->IT AST (bilingual EN+IT) |
| what is reused | `init_from_pretrained_model: {model0: {name: nvidia/canary-1b-flash, include: ["encoder"], exclude: ["encoder.pre_encode.out"]}}` — **only the encoder is loaded**, minus the subsampling output projection |
| what is re-initialised | the whole Transformer decoder, the output head ("joiner" in the guide's wording) and `encoder.pre_encode.out` |
| tokenizer, option A | **aggregate** (`type: agg`): `spl_tokens` (special-token model from `canary/special_tokens_multitask/`) + `en` SPE unigram 1024 (trained on all English Granary text) + `it` SPE unigram 1024 (trained on IT ASR + EN-IT text); IDs offset (EN 0-1023, IT 1024-2047 in their description), `custom_tokenizer: CanaryTokenizer` |
| tokenizer, option B | **unified** SPE BPE over EN+IT text (2048 tokens in the results table; the YAML snippet path says `tokenizer_spe_bpe_v1024`), `custom_tokenizer: CanaryBPETokenizer` |
| prompt format | not discussed; unchanged Canary prompt (language tokens via `lang_field: target_lang`) |
| data | Granary IT subsets processed with NeMo-speech-data-processor (`dataset_configs/multilingual/granary/yodas.yaml`, `params.source_lang="it" params.en_translation=True`; same for MOSEL); optionally Mozilla Common Voice IT and CoVoST via the SDP Italian MCV config; FLEURS "can be applied" the same way. Character whitelist per language (`partials/subregex_params/it.yaml`) |
| data loading | lhotse, `input_cfg` with per-corpus weights = hours; tarred (`nemo_tarred`) |
| hyper-parameters given | `max_duration 40`, `min_duration 0.01`, `text_field: answer`, `lang_field: target_lang`, `use_bucketing`, 8 2D buckets with `bucket_batch_size [35,32,30,30,30,29,21,21]`, `bucket_buffer_size 20000`, `shuffle_buffer_size 10000`, `max_tps` per bucket ("generated and optimized by the NeMo framework") |
| NOT given | learning rate, schedule, steps/epochs, precision, GPU type/count, wall time, total hours actually used — all **UNKNOWN** |
| script | `examples/asr/speech_multitask/speech_to_text_aed.py` (+ `fast-conformer_aed.yaml`) |
| results | MLS-IT WER 7.16 (unified 2048) / 7.33 (agg 1024); FLEURS IT WER 7.89 / 3.46 (*column labelling in the HTML table is inconsistent — header order and value order do not match; treat individual FLEURS/CoVoST numbers as unreliable*); CoVoST WER 4.17 / 3.94; plus BLEU/COMET for IT<->EN |

Takeaways for a **low-data** setting: the guide is a *high-resource* recipe (Granary IT is
thousands of hours; see §4) that throws away the decoder. It is not evidence that 5-40 h are
enough. Its scale is a ceiling reference: a 883M encoder + fresh decoder on Granary-scale IT
reaches ~7% MLS WER.

## 2. NeMo mechanics that matter (verified in source)

### 2.1 Canary 180M (`EncDecMultiTaskModel`)

- The 180M tokenizer is `agg`: `spl_tokens` (1152) + en/de/es/fr (1024 each) = 5248 classes.
  **`<|it|>` already exists** in `spl_tokens`, so the canary2 prompt can address Italian with
  no special-token work; what is missing is an Italian *text* sub-tokenizer.
- `change_vocabulary(new_tokenizer_dir=<agg DictConfig>, new_tokenizer_type="agg")`
  (`aed_multitask_models.py:298-420`): rebuilds `transf_decoder` with the new vocab size
  (rounded up to a multiple of 8) and **copies every decoder tensor whose shape is unchanged**
  — i.e. all 4 Transformer layers survive — but the **token embedding (shape changes) and the
  output head are re-initialised** (head is weight-tied to the embedding and re-initialised
  with `transformer_weights_init`). Side effect: **all old EN/DE/ES/FR/special-token
  embeddings are lost too**, including prompt tokens, unless restored by hand.
- Cheap surgery (not an official recipe; to be validated): append the `it` sub-tokenizer
  **after** `fr` in the `langs` mapping so the existing 5248 IDs keep their positions
  (IT = 5248..6271, total 6272, already a multiple of 8), call `change_vocabulary`, then copy
  the old 5248 embedding rows back and init the IT rows (e.g. mean of the old text rows, or of
  the EN/ES/FR rows of the pieces' character decomposition). This preserves the prompt, the
  special tokens, and EN/DE/ES/FR. Whether `CanaryTokenizer` assumes a fixed language order:
  UNKNOWN (inspect before relying on it).
- Alternative with no surgery: replace one sub-tokenizer slot (e.g. train an `it` model and
  drop `es`) — same reinit problem, worse forgetting. Not recommended.
- Training precision: `fast-conformer_aed.yaml` sets `trainer.precision: bf16-mixed`; L4
  (sm_89) supports bf16. Freezing: the AED script has no built-in "freeze encoder" switch;
  use `model.encoder.freeze()` in a small wrapper, or NeMo adapters (UNKNOWN whether
  adapters are wired for `EncDecMultiTaskModel`; adapters are documented for CTC/RNNT).
- NeMo version: the 180M was saved with `nemo_version: 2.3.0rc0`; card says "NeMo main"
  at publication; AED streaming decoding exists from v2.6.0. Use **NeMo >= 2.6** (or 3.x).

### 2.2 EOU 120M (`EncDecRNNTBPEModel` in the `.nemo`)

- English SPE 1024 + `<EOU>`, `<EOB>` (1026) — see upstream file §1.
- Italian needs a new tokenizer: `change_vocabulary()` on an RNNT BPE model **re-creates the
  prediction network and the joint from scratch** (`rnnt_bpe_models.py:271-345`) — only the
  encoder (109.5M of 114.9M params) is reused. EOU and EOB are therefore gone and must be
  re-learned.
- The official EOU fine-tuning path (`examples/asr/asr_eou`, **NeMo v3.0.0+ only**, absent
  in v2.7.3) expects: tokenizer = base tokenizer + 2 tokens appended
  (`scripts/asr_eou/tokenizers/add_special_tokens_to_sentencepiece.py`); manifests with
  `sou_time`/`eou_time` from `tools/nemo_forced_aligner/align_eou.py` (a CTC model);
  `LhotseSpeechToTextBpeEOUDataset` adds random leading/trailing silence and puts `<EOU>` at
  the end of the text; blend EOU data (0.9) with plain ASR data (0.1).
- So the Italian EOU route is a two-step job: (1) Italian ASR on the 120M encoder with an
  Italian SPE that **already contains** `<EOU>`/`<EOB>` as its last two pieces (train the
  SPE, then run `add_special_tokens_to_sentencepiece.py` on it); (2) EOU training with
  Italian manifests aligned by an Italian CTC model — candidate
  `nvidia/stt_it_fastconformer_hybrid_large_pc` (CC-BY-4.0, ~115M, MCV12 220 h + MLS 214 h +
  VoxPopuli 53 h, card WER 5.64 MCV12 / 11.39 MLS / 16.22 VoxPopuli without PnC). Whether
  `align_eou.py` takes a hybrid RNNT-CTC checkpoint directly: UNKNOWN (NFA supports
  hybrid models' CTC head in general). Fallback for labels: VAD-free cheap proxy =
  utterance end from MLS/CV segment boundaries after trimming silence with an energy VAD
  (not NVIDIA's method; label noise higher).
- `add_eob_labels.py` is English-only (hard-coded phrase list) — skip EOB
  (`ignore_eob_label: true`, the default).
- Training precision: the EOU YAMLs default `trainer.precision: 32`; RNNT loss is forced to
  fp32 anyway (`losses/rnnt.py:85`), encoder can run `bf16-mixed`.

### 2.3 Reference point already in the open: an Italian 115M FastConformer exists

`nvidia/stt_it_fastconformer_hybrid_large_pc` (2023, CC-BY-4.0) is an offline Italian
FastConformer hybrid trained on 487 h. It is the natural **upper reference** for what a
~115M FastConformer reaches on open Italian data with full training; any 5-40 h adaptation
should be compared to it on the same test sets. It is not cache-aware (card describes no
streaming attention context; config not inspected — UNKNOWN), so it does not replace the
EOU 120M as a streaming model, but its encoder could seed a cache-aware fine-tune (not an
official path).

## 3. Italian open datasets

Hours verified from primary stats files where possible (method in each row).

| dataset | URL | licence | hours (IT) | splits | audio | domain | transcripts | training use | download size |
|---|---|---|---|---|---|---|---|---|---|
| Common Voice Scripted Speech **27.0** (2026-09-11) | https://mozilladatacollective.com (stats: https://github.com/common-voice/cv-dataset `datasets/scripted-speech/cv-corpus-27.0-2026-09-11.json`) | **CC0-1.0**; MDC terms: do not try to identify speakers; **"forbidden to re-host or re-share this dataset"** | total 430.5 h, **validated 363.9 h**; 7,350 speakers | clips: train 173,631, dev 15,185, test 15,183, validated 240,628, other 23,094, invalidated 20,945 (avg 5.44 s -> train ~262 h, test ~23 h) | MP3 48 kHz mono (resample to 16 kHz) | read sentences (Wikipedia/public-domain prompts), many mics | human-validated reading prompts (≥2 up-votes) | allowed (CC0; MDC says intended for ASR training) | **10.54 GB** per-language tarball |
| FLEURS it_it | https://huggingface.co/datasets/google/fleurs | CC-BY-4.0 | train 9.00 h (3,030), dev 1.55 h (391), test **3.52 h (865)** — computed from `num_samples` in the tsv files | speaker-disjoint train vs dev/test | 16 kHz wav | read FLoRes Wikipedia sentences; ~3 readings per sentence | `raw_transcription` (cased, punctuated) + normalised `transcription` | allowed (attribution) | train 1.64 GB, dev 0.29 GB, test 0.66 GB |
| Multilingual LibriSpeech Italian | https://www.openslr.org/94/ ; HF `facebook/multilingual_librispeech` config `italian` | CC-BY-4.0 | train **247.38 h** (65 speakers), dev 5.18 h, test **5.27 h** (arXiv 2012.03411 Table 2) | HF: train 59,623 / dev 1,248 / test 1,262 utts; also `9_hours` and `1_hours` limited-supervision subsets | 16 kHz (FLAC 15 GB or Opus 3.8 GB on OpenSLR; HF parquet 4.2 GB total) | LibriVox audiobooks (old literary prose) | book text auto-aligned; paper's human check: 5.37% WER on IT test | allowed | HF test parquet 83 MB, train 3.9 GB; OpenSLR opus 3.8 GB |
| VoxPopuli it (transcribed) | https://github.com/facebookresearch/voxpopuli ; HF `facebook/voxpopuli` config `it` | data **CC0** (code CC-BY-NC 4.0); EP acknowledgement | **91 h**, 306 speakers, 757K tokens (README table) | HF parquet train / validation / test | 16 kHz | European Parliament speeches 2009-2020 | official EP transcripts aligned/filtered (not verbatim; numbers/hesitations differ) | allowed | HF: train 5 shards × ~3.06 GB = 15.3 GB, validation 0.90 GB, test 0.87 GB |
| Granary IT (manifests) | https://huggingface.co/datasets/nvidia/Granary | repo CC-BY-4.0; card's own licence section: **YODAS-Granary CC-BY-3.0**, **MOSEL CC-BY-4.0** (transcripts), "original audio corpora: see respective source licenses" | computed 2026-10-09 by summing `duration`: `it/yodas/it_asr.jsonl` 86,587 utts **120.2 h** (ASR-only rows) + `it/yodas/it_ast-en.jsonl` 1,226,663 utts **6,024.8 h** (these rows also carry the Italian transcript in `text`, so YODAS IT transcribed audio ≈ **6,145 h**); `it/ytc/it_asr.jsonl` 19,957 utts **101.2 h** (+ `it_ast-en` 18,540 utts 96.6 h, same audio); `it/voxpopuli/it_asr.jsonl` 2,815,857 utts, hours see note ¹ | ASR + AST(it->en) manifests | YODAS: 16 kHz wav embedded in `espnet/yodas-granary`; VoxPopuli FLAC and YouTube-Commons WAV must be fetched from the sources | YouTube (CC-licensed uploads) + EP | **pseudo-labels**: Whisper-large-v3 two-pass + LID + hallucination filters, PnC restored with Qwen-2.5-7B; AST by EuroLLM-9B | transcripts CC-BY; **audio licensing is per source**: YODAS/YTC audio = YouTube videos whose uploader chose a CC licence (per-video, not re-verified by us), fetching YTC audio means downloading from YouTube (platform ToS apply) | manifests small (ytc 15 MB, yodas 36 MB); YODAS IT audio is part of a ~700 GB multi-subset HF repo (per-language size UNKNOWN) |
| MOSEL (source of Granary VoxPopuli/YTC labels) | https://huggingface.co/datasets/FBK-MT/mosel | CC-BY-4.0 (transcripts only; no audio) | card: 16,713 h of IT transcripts across VoxPopuli (unlabelled 400K-h release) + YouTube-Commons | — | — | EP + YouTube | Whisper-large-v3 pseudo-labels with hallucination flags | transcripts yes; audio per source | — |
| Europarl-ST (it source) | https://www.mllp.upv.es/europarl-st/ | not stated on the landing page (UNKNOWN; commonly cited as CC-BY-NC 4.0 — secondary) | ~37 h IT-source train (+15 h train-noisy); the IT audio overlaps EP sessions also in VoxPopuli | per direction | — | EP | EP transcripts + translations | **UNKNOWN licence -> exclude** until verified | — |

¹ The VoxPopuli IT ASR manifest is 2.45 GB of JSONL (2,815,857 rows) indexing the large
*unlabelled* VoxPopuli release (`voxpopuli/it/<session>_it_<n>.ogg`) with Whisper
pseudo-labels; its `duration` field is null on every row, so hours cannot be summed from the
manifest (UNKNOWN; MOSEL's card gives 16,713 h of IT transcripts across VoxPopuli + YTC).
Note these are pseudo-labels on the unlabelled EP audio, not the 91 h human-transcribed set.

Access notes (verified):
- **Common Voice is only on Mozilla Data Collective since Oct 2025** (CV 23.0 was the first MDC
  release, 2025-09-30). Programmatic download works on a rented box: `pip install
  datacollective`, create an API key in the MDC profile ("Credentials"), `export
  MDC_API_KEY=...`, `from datacollective import download_dataset; download_dataset("<dataset id
  or slug>")` (resumable). The SDK README says to read and agree to the dataset's terms
  first; whether a click-through on the website is required before the API serves a given
  dataset: UNKNOWN (test with a small language first). Re-hosting/re-sharing is forbidden,
  so pull directly on the box rather than mirroring a subset somewhere shared.
  Old HF `mozilla-foundation/common_voice_*` repos now point to MDC.
- The Open ASR Leaderboard multilingual pack
  (https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-multilingual-datasets) ships
  ready test sets `fleurs_it` (865), `mcv_it` (8,951), `mls_it` (657, a subset of MLS test);
  its card declares **CC-BY-NC-SA-4.0**. Fine for evaluation; the CV version inside is UNKNOWN.

### 3.1 Recommended Italian eval set (held out, frozen)

- **FLEURS it_it test** (865 utts, 3.52 h, 0.66 GB): read, clean, cased text available;
  directly comparable to the Nemotron 3.5 streaming card (it-IT FLEURS WER 4.25 at 1.12 s).
- **MLS it test** (1,262 utts, 5.27 h, 83 MB parquet): long-form-ish audiobook domain;
  comparable to the Granary guide (MLS 7.16-7.33 for 1B) and the stt_it 115M (11.39).
- **CV 27.0 it test, fixed random 2,000-clip subset** (~3 h; seed recorded): crowd mics,
  hardest acoustics; full test is 15,183 clips (~23 h) — too slow for every rung.
- Optional domain probe: VoxPopuli it test (0.87 GB).
Normalisation: open_asr_leaderboard `MultilingualNormalizer(remove_diacritics=False)` with
`lang="it"` (num2words Italian). Keep the exact file lists in the repo.

### 3.2 Training pool and 5 / 20 / 40 h cuts

Pool (all commercial-use friendly, speaker-disjoint from the eval sets by construction):
CV 27.0 it **train** bucket (~262 h; CV splits are speaker-disjoint) + MLS it **train**
(247 h; MLS test speakers are disjoint) + FLEURS it **train** (9 h; speaker-disjoint from FLEURS test, and sentence-disjoint —
checked 2026-10-09 on the `it_it` tsv files: 1,485 train / 147 dev / 346 test sentence ids,
zero overlap train∩test and dev∩test; still kept out of the first runs so FLEURS stays a
pure out-of-domain-text test). VoxPopuli it train (91 h) is a fourth domain held in
reserve.

Cut recipe: nested subsets (5 h ⊂ 20 h ⊂ 40 h), each 50% CV / 50% MLS by duration, capped
utterances per speaker (CV: ≤ 20 clips/speaker; MLS: ≤ 30 min/speaker), durations 1-20 s,
seed recorded, manifest committed. Nesting makes the data-scaling curve monotone by design.

Disk budget on a ~32 GB box (16 kHz mono PCM16 = 115 MB per hour):

| item | download | kept on disk |
|---|---|---|
| CV 27.0 it tarball | 10.54 GB (stream-extract only the selected `clips/*.mp3` + `train.tsv`/`test.tsv`; delete tarball) | 40 h CV half = 20 h wav = 2.3 GB; 2,000 test clips = 0.35 GB |
| MLS it (HF parquet) | train 3.9 GB + test 0.08 GB | 20 h wav = 2.3 GB; test 0.6 GB |
| FLEURS it test | 0.66 GB | 0.66 GB |
| models (.nemo) | 0.46 + 0.74 GB | 1.2 GB |
| NeMo + PyTorch env, checkpoints | — | ~10-12 GB (UNKNOWN exact; container dependent) |
| **peak** | ~11 GB transient (CV tarball) | **~18-20 GB** |

Avoid holding the CV tarball and the extracted copy at once (`tar -xzf ... --files-from
list.txt` streaming, then delete). Granary/YODAS is out of scope for a 32 GB box.

## 4. Can the Granary-guide approach apply to the two small models?

- **canary-180m-flash: yes, mechanically.** Same model class, same agg tokenizer machinery,
  the guide's own table lists 180M dimensions, `<|it|>` already in the prompt vocabulary.
  Differences we should make for 5-40 h: keep the pretrained decoder (do not use
  `include: [encoder]` only), add `it` as a fifth sub-tokenizer with row-copy surgery (§2.1),
  mix some EN/DE/ES/FR replay data to limit forgetting. Smaller IT sub-tokenizer (256-512)
  is likely better than 1024 for 5-40 h of text (sparser embeddings to learn); UNKNOWN —
  measure.
- **EOU 120M: only partly.** The *data* side (manifests, lhotse) applies; the *tokenizer* side
  does not (RNNT single SPE, no agg; `change_vocabulary` reinitialises pred-net + joint);
  EOU must be re-taught with the separate EOU recipe (NeMo ≥ v3.0.0). An agg tokenizer for
  RNNT exists in NeMo (`type: agg` is accepted by `rnnt_bpe_models.change_vocabulary`) but the
  EOU dataset checks that the last two vocab ids are `<EOU>`/`<EOB>`, so the IT SPE must carry
  them at its end.

## 5. Proposed staged experiment plan (cheapest first)

Common protocol: one L4 24 GB, `bf16-mixed` where the loss allows, NeMo pinned (≥ 3.0.0 for
any EOU step; same version for all rungs), eval on §3.1 sets with the leaderboard
normaliser, every rung logs wall time, audio-hours processed, peak VRAM, WER per set, and
EN WER drift (LS test-clean + FLEURS en) for forgetting. Each rung is time-boxed; stop a
branch when the next rung does not beat the previous by > 1 abs WER on FLEURS it.

### Track C — Canary 180M

- **C0 zero-shot probe (minutes).** Prompt `source_lang=target_lang=it` on the untouched
  model. Expected garbage (no IT text pieces), but it measures how the decoder behaves with
  an unseen language token and gives the floor. Also run `source_lang=es`/`fr` on Italian
  audio as a "nearest language" floor.
- **C1 tokenizer surgery smoke test (< 1 h).** Train `it` SPE (unigram 512 and 1024) on the 40 h
  pool text only; append after `fr`; `change_vocabulary` + restore 5248 rows; verify EN/DE/ES/FR
  WER unchanged (bit-identical decode on 50 clips) before any training.
- **C2 frozen encoder, train decoder + new rows (5 h, then 20 h).** Encoder frozen; LR ~1e-4
  AdamW, warmup 500, ~2-5k steps; 25% replay EN/ES/FR batches. Hypothesis: this alone gets
  FLEURS-it into a usable range because Italian phonetics overlap ES/FR strongly; UNKNOWN.
- **C3 unfreeze top N encoder layers (N = 4, 8) at 20 h and 40 h.**
- **C4 full fine-tune at 40 h** (encoder LR 0.3x decoder LR), the comparison point against
  `stt_it_fastconformer_hybrid_large_pc` (487 h, offline) and against Nemotron 3.5 streaming
  (it-IT FLEURS 4.25).
- **C5 streaming check** only if C3/C4 are good: run `speech_to_text_aed_streaming_infer.py`
  (AlignAtt, 10-1-0.5) on FLEURS it test; record WER, LAAL and encoder recompute cost
  (expected 11.5x encoder work vs offline, see upstream §2.4). This decides whether Canary
  stays an "offline/batch" tier in mynah.

### Track E — EOU 120M

- **E0 English sanity (minutes).** Reproduce card behaviour on LS test-clean with NeMo
  (WER, `<EOU>` presence) to validate the toolchain and the mynah conversion side by side.
- **E1 tokenizer swap smoke test (< 1 h).** IT SPE 512/1024 (+`<EOU>`,`<EOB>` appended via
  `add_special_tokens_to_sentencepiece.py`), `change_vocabulary`, 5 h ASR-only training with
  frozen encoder for 1-2k steps — checks the loop runs and loss falls; expect high WER.
- **E2 ASR adaptation 20 h / 40 h**, encoder unfrozen at low LR, `att_context_size [70,1]`
  kept, FastEmit 0.03; plain ASR manifests (no EOU yet). Measures what the cache-aware 115M
  encoder can do in Italian at 160 ms.
- **E3 EOU label synthesis.** Run `align_eou.py` with an Italian CTC model
  (`stt_it_fastconformer_hybrid_large_pc` if accepted; else `stt_it_conformer_ctc_large`) on
  the 20/40 h manifests -> `sou_time`/`eou_time`; spot-check 50 alignments by ear. Requires
  single-utterance clips: CV and FLEURS fit; MLS segments are mid-sentence cuts — exclude MLS
  from EOU data or keep it as the 0.1-weight ASR-only group.
- **E4 EOU training** with `speech_to_text_rnnt_eou_train.py` from the E2 checkpoint, EOU
  group 0.9 / ASR group 0.1, random padding as in the README, noise manifest (MUSAN or
  similar, licence check needed). Eval with `speech_to_text_eou_eval.py` on FLEURS it test
  padded by `generate_noisy_eval_data.py`: WER, EOU latency p50/p90, early-cutoff and miss
  rates; compare with the English model's card numbers (p50 160 ms on TTS data — different
  data, so only indicative).

### Explicit uncertainties

- No public number shows how far 5-40 h moves either model; the only Italian reference
  points are 487 h (stt_it 115M offline) and Granary-scale (1B guide). The curve is the
  deliverable.
- Canary 180M's decoder is 74M params trained on 4 languages; with 5 h the risk is
  overfitting new embeddings, with full FT the risk is forgetting — replay ratio is a guess.
- The EOU recipe's exact data mix for the released 120M is unpublished; Italian EOU quality
  depends on alignment quality from a 2023 Italian model.
- `align_eou.py` + hybrid model compatibility, CanaryTokenizer language-order assumptions,
  adapters for AED: all UNKNOWN until run.
- L4 throughput is an estimate (upstream §4: ~25-60 audio-h per L4-hour); the first rung
  measures it.
- Licence mix: Canary derivative = CC-BY-4.0 attribution; EOU derivative = NVIDIA Open Model
  License notice + guardrail clause; data CC0/CC-BY only (exclude Europarl-ST,
  leaderboard pack for training).

## Measurements (none yet)

| rung | model | data (h) | trainable | steps | wall (L4 h) | audio-h/L4-h | peak VRAM | FLEURS-it WER | MLS-it WER | CV-it WER | EN drift | EOU p50 / cutoff / miss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | canary-180m | 0 | — | — | — | — | — | — | — | — | — | n/a |
| E0 | eou-120m | 0 | — | — | — | — | — | n/a | n/a | n/a | — | — |

## Economics are a measured result, not an estimate (decided 2026-10-09)

The $4-16 figure above is an ESTIMATE from a third-party throughput number and
is not quoted anywhere as a result. Every FT run records, from its own logs:
audio-hours per GPU-hour, samples/s, peak allocated and reserved VRAM, epoch
wall time, total GPU-hours, and cost per epoch and per run at the box's actual
hourly rate (the rate is written down when the box is rented).

## Kit (Track C, Canary 180M): `.work/lightweight-asr/ft/`

Restartable scripts for one rented GPU box (tmux, `timeout` on every long step,
marker files in `/root/ft/done/`): `setup.sh` (torch -> cu128 wheel in the NeMo
3.0.0 venv, normaliser, checkpoint, CUDA/bf16 check) -> `data_it.sh` (MLS it +
FLEURS it/en, frozen eval = FLEURS it test + MLS it test, nested 5/20/40 h MLS
cuts, exact hours; Common Voice optional) -> `tokenizer_it.sh` (C1: `it` SPE
appended after `fr`, 5,248 old rows copied back, EN identity check) ->
`probe.sh` (C0) -> `train_canary_it.sh` (C2 arm A frozen encoder, C3 arm B
top-N layers; per-run audio-h/GPU-h, samples/s, peak VRAM, epoch wall,
GPU-hours, cost at `RATE_USD_H`, EN forgetting) -> `eval_it.py` (WER/CER with
S/D/I, level sweep at -3/-20/-40 dBFS peak). Level robustness is measured
before/after every run and random-gain augmentation is a recorded A/B option
(motivated by the EOU 120M empty-transcript finding on quiet clips). Command
order, wall-time and disk estimates: `.work/lightweight-asr/ft/README.md`.
Track E (EOU 120M) is not in the kit yet; its trainer should reuse
`ftlib.make_gain_aug` with gain ON by default.

## Measurements

### 2026-10-09, L40S (sm_89), NeMo 3.0.0, Canary 180M Flash -> Italian, smoke runs

Kit: `.work/lightweight-asr/ft/` (run_all.sh). Data: MLS-it train, nested
subsets 4.9988 / 19.996 / 39.998 h, 65 speakers, no speaker overlap with the
eval; pool peak p50 -7 dBFS. Frozen eval: FLEURS-it test (865) + MLS-it test
(1262), leaderboard multilingual normaliser (lang it). EN forgetting: 100
FLEURS-en clips. Rate not set yet (cost fields 0); GPU-hours are measured.

Zero-shot probe (pretrained, after the tokenizer surgery):

| prompt (src=tgt) | FLEURS-it WER / CER | MLS-it WER / CER |
|---|---|---|
| it (no Italian text tokens yet) | 100.86 / 100.71 (635 empty) | 99.89 / 99.31 |
| es | 87.67 / 30.00 | 82.01 / 28.13 |
| fr | 97.44 / 38.77 | 89.76 / 36.35 |
| EN sanity, FLEURS-en 100 clips | 6.86 / 3.46 (6.99 at -40 dBFS) | |

The encoder already "hears" Italian phonetically (CER ~28-30 % through the
Spanish prompt); the missing part is the text side.

Smoke runs: 5 h subset, 600 steps (8.7 equivalent epochs), lr 2e-4, bf16,
300 s of audio per batch:

| run | trainable | FLEURS-it WER / CER | MLS-it WER / CER | EN WER delta | train wall | audio-h per GPU-h | peak VRAM alloc / reserved | GPU-h incl. evals |
|---|---|---|---|---|---|---|---|---|
| A (frozen encoder) | 74.1 M / 183.7 M | 71.32 / 20.72 | 56.70 / 14.05 | +0.70 | 55 s | 2843 | 8.05 / 8.6 GB | 0.131 |
| B4 (top 4 encoder layers, enc lr x0.3) | 99.4 M | 68.40 / 19.30 | 54.24 / 12.93 | +2.51 | ~65 s | 2404 | 8.33 / 9.72 GB | 0.134 |

Level robustness after A (FLEURS-it): 72.06 / 71.89 / 69.61 WER at peak
-3 / -20 / -40 dBFS: FT did not narrow it.
Reading: a clear slope from empty output in about a minute of training, but
undertrained (loss ~2.0 at the end): a smoke schedule. VRAM fits an L4.
Next (ft2): ~260 h of processed audio per subset (5 h x 52, 20 h x 13,
40 h x 7 epochs), A and B4 at 5 h, B4 at 20/40 h.
