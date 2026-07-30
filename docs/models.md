# Mynah — Target model catalog (NVIDIA NeMo speech)

> References verified on HuggingFace on **2026-07-16** (fetched the model cards, the
> `config.json` files, and the actual file trees). Keep up to date as NVIDIA releases variants.

## ✅ Supported by the runtime (2026-07-18)

| Model | Engine | Notes |
|---|---|---|
| nemotron-3.5-asr-streaming-0.6b | nemotron-streaming | cache-aware streaming + offline, 40 languages, LID |
| parakeet-tdt-0.6b-v3 | parakeet-tdt | 25 EU languages, ITN, accurate timestamps |
| parakeet-tdt_ctc-110m | parakeet-tdt (+ctc) | EN; CI model; `--decoder ctc` |
| parakeet-rnnt-0.6b / ctc-0.6b | parakeet-rnnt/ctc | EN, lowercase — import from `.nemo`¹ |
| parakeet-rnnt-1.1b / ctc-1.1b | parakeet-rnnt/ctc | EN, 42L — import from `.nemo`¹ |
| canary-180m-flash / 1b-flash | canary-aed | ASR en/de/es/fr + **translation** + word-ts |
| canary-1b-v2 | canary-aed | ASR **25 EU languages** (it!) + translation en↔24, ITN; word-ts via the CTC aligner bundled in its `.nemo` |

All with: int8/int4 (`mynah-asr quantize`), Metal, batch, long-file segmentation,
REST/WS server. Measured RTFs in [benchmarks.md](benchmarks.md).

¹ The classic rnnt/ctc Parakeets ship an HF-native `model.safetensors` too, but
`convert_nemo.py` has HF builders only for `nemotron3_5_asr` and `parakeet_tdt`:
these four convert via the `.nemo` archive (torch, `uv sync --extra oracle`).
`download_model.sh` fetches the right format per model (found the hard way on the
first clean-server run, 2026-07-20 — the menu wrongly listed them as `hf`).

Traits shared by almost the whole family (from the verified `config.json` files):
- **FastConformer encoder**: depthwise-separable conv subsampling **8×**, d_model **1024**,
  **8 attention heads**, conv module kernel **9**. 24 layers = "XL" 0.6B variants,
  42 layers = "XXL" 1.1B, 17 layers = small.
- **Features**: 16 kHz mono, log-mel, hop **160** (10 ms), window **400** (25 ms), n_fft **512**,
  preemphasis **0.97**. **80 mel** for the classic Parakeets, **128 mel** for tdt-0.6b-v3 and Nemotron.
- **Tokenizer**: SentencePiece (Unigram 1024 for the classic EN models; unified 8192 for tdt-v3;
  unified 13088 for Nemotron 3.5; 16384 for Canary v2).
- Distribution: `.nemo` (tar: weights + `model_config.yaml` + tokenizer); the 2025+ models
  often also have an **HF-native port** (`model.safetensors` + `config.json` + `tokenizer.json`).

---

## 🎯 v1 target — Nemotron Speech Streaming

### nvidia/nemotron-3.5-asr-streaming-0.6b ← **THE v1 model**
**https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b** (released 2026-06-04)

- **Architecture** (from the verified `config.json`, HF class `Nemotron3_5AsrForRNNT`):
  - Cache-aware streaming FastConformer encoder: **24 layers, d_model 1024, 8 heads
    (8 KV), FFN 4096, conv kernel 9, subsampling 8×** → 1 encoder frame = **80 ms**.
    Windowed causal attention: `sliding_window: 57` (left 56 + current). Mel **128**.
  - **RNNT** decoder: prediction network hidden **640, 2 layers** (presumably LSTM — to be
    confirmed from the YAML in the `.nemo`), `max_symbols_per_step: 10`. Joint dim not in the config.
  - **Language conditioning**: **128-dim** one-hot (`num_prompts: 128`, default id 101)
    concatenated to the features + projection (intermediate 2048); dictionary of 105 locales
    in the processor. `target_lang=auto` → language detection with the language tag emitted in the output.
- **Streaming / latency** configurable at runtime via `att_context_size = [left, right]`:
  `[56,0]`=80 ms, `[56,1]`=160 ms, `[56,3]`=320 ms (default), `[56,6]`=560 ms, `[56,13]`=1.12 s.
- **Feature extractor** (from `processor_config.json`): 16 kHz, 128 mel, n_fft 512, hop 160,
  win 400, preemph 0.97. Normalization NOT declared → read `model_config.yaml` in the `.nemo`.
- **Tokenizer**: class `ParakeetTokenizer`, `tokenizer.json` 752 KB, vocab **13088**,
  blank id **13087**, `<pad>`=0, 39 special language tokens.
- **Languages**: 40 locales in 3 tiers — transcription-ready (19, **it-IT included**),
  broad-coverage (13), adaptation-ready (8). Native punctuation & capitalization.
- **License**: **OpenMDW-1.1** (very permissive).
- **Files in the repo (~4.9 GB)**: `model.safetensors` (2.55 GB) ✨, `.nemo` (2.37 GB),
  `config.json`, `processor_config.json`, `tokenizer.json`, `generation_config.json`.
  → the HF-native port lets us build a converter **without depending on nemo_toolkit**.
- **Benchmarks**: FLEURS @1.12s: **it-IT 4.25%**, es 4.11%, pt 5.48%, en 7.91%;
  H100: ~240 streams @80ms → ~2400 @1.12s.
- **To close out in M0**: prediction network type and feature normalization (YAML in the `.nemo`);
  timestamps on the 3.5 (documented only for the EN model); tokenizer algorithm (BPE vs unigram).

### nvidia/nemotron-speech-streaming-en-0.6b (EN-only variant, Jan 2026)
**https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b**
- Same architecture, English only, tokenizer 400 KB smaller. **Caution**: left
  context **70** (`[70,0..13]`) versus the 3.5's **56** → the runtime must read it from the config.
- Average WER 6.93% @1.12s (LS clean 2.32%); median time-to-final 24 ms; ~560 streams @320ms on H100.
- License: NVIDIA Open Model License (more restrictive than the 3.5 — to be re-checked).
- Demo Space: https://huggingface.co/spaces/nvidia/nemotron-speech-streaming-en-0.6b
- Collection: https://huggingface.co/collections/nvidia/nemotron-speech

Nemotron resources:
- HF blog (Jan 2026): https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents
- Fine-tuning blog (Jun 2026): https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr
- Fine-tuning notebook: https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb
- Deploy examples: https://github.com/modal-projects/modal-nvidia-asr · https://github.com/pipecat-ai/nemotron-january-2026
- Third-party benchmark (EN streaming on constrained hardware): https://arxiv.org/abs/2604.14493

---

## Parakeet family (pure ASR)

### nvidia/parakeet-tdt-0.6b-v3 ← reference multilingual model (v0.4)
**https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3** (2025-08-14)
- FastConformer 24L·1024·8h·**128 mel** / **TDT** decoder: pred 2L·640, durations **[0,1,2,3,4]**,
  max_symbols_per_step 10, **blank_id 8192**. Unified SPE tokenizer **8192**.
- **25 EU languages** (it included) with auto language detection, automatic PnC,
  **word/segment/char** timestamps. Long audio: 24 min full attention, up to **3 h** with
  local attention `change_attention_model("rel_pos_local_attn", [256,256])`.
- Streaming: not cache-aware; chunked via `speech_to_text_streaming_infer_rnnt.py`
  (chunk 2s, right 2s, left 10s).
- CC-BY-4.0 license. Files: `.nemo` **+ complete HF-native port** (safetensors, tokenizer.json).
- Open ASR Leaderboard avg WER **6.34%**; FLEURS it **3.00%**, MLS it 10.08%, CoVoST it 3.69%.
- Tech report (joint with Canary v2): https://arxiv.org/abs/2509.14128

### The other Parakeets (all verified, table)

| Model | URL | Arch | Languages | Notes | License | Files |
|---|---|---|---|---|---|---|
| parakeet-tdt-0.6b-v2 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 | 24L / TDT | EN | PnC, ts, RTFx 3380, avg WER 6.05% | CC-BY-4.0 | .nemo only |
| parakeet-tdt-1.1b | https://huggingface.co/nvidia/parakeet-tdt-1.1b | 42L / TDT | EN | lowercase, no PnC | CC-BY-4.0 | .nemo only |
| parakeet-rnnt-1.1b | https://huggingface.co/nvidia/parakeet-rnnt-1.1b | 42L·80mel / RNNT 2L LSTM·640 | **EN only** (⚠️ the PDF listed it as multilingual: wrong) | lowercase | CC-BY-4.0 | .nemo + safetensors |
| parakeet-rnnt-0.6b | https://huggingface.co/nvidia/parakeet-rnnt-0.6b | 24L / RNNT | EN | lowercase | CC-BY-4.0 | .nemo + safetensors |
| parakeet-ctc-1.1b | https://huggingface.co/nvidia/parakeet-ctc-1.1b | 42L / CTC | EN | most downloaded (~974k) | CC-BY-4.0 | .nemo + safetensors |
| parakeet-ctc-0.6b | https://huggingface.co/nvidia/parakeet-ctc-0.6b | 24L / CTC | EN | | CC-BY-4.0 | .nemo + safetensors |
| parakeet-tdt_ctc-110m | https://huggingface.co/nvidia/parakeet-tdt_ctc-110m | Hybrid TDT+CTC, 114M | EN | PnC; **CI model candidate** | CC-BY-4.0 | .nemo only |
| parakeet-tdt_ctc-1.1b | https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b | Hybrid, local attn+global token | EN | up to **11 h** in one pass | CC-BY-4.0 | .nemo only |

### New streaming releases 2025/2026 (post-v1 candidates)

| Model | URL | What it does | License |
|---|---|---|---|
| **parakeet-unified-en-0.6b** | https://huggingface.co/nvidia/parakeet-unified-en-0.6b | Unified-FastConformer-RNNT: **one model for offline AND streaming** (latency 160 ms–2.08 s, chunked self-attention + Dynamic Chunked Convolutions), PnC | NVIDIA OML |
| **parakeet_realtime_eou_120m-v1** | https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 | Cache-aware 17L / RNNT, 120M, 80–160 ms latency, emits an `<EOU>` (end-of-utterance) token — voice agent | NVIDIA OML |
| **multitalker-parakeet-streaming-0.6b-v1** | https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1 | Streaming multitalker based on Nemotron-Speech-Streaming, speaker kernels + diarization, one instance per speaker | NVIDIA OML |

Per-language variants: parakeet-tdt_ctc-0.6b-ja (JA), parakeet-ctc-0.6b-Vietnamese (VI),
parakeet-rnnt-110m-da-dk (DA).

---

## Canary family (ASR + Speech Translation, AED) — flash ✅ supported

All offline (no native streaming; long audio via `speech_to_text_aed_chunked_infer.py`).
Task conditioning via decoder prompt tokens.

| Model | URL | Params | Encoder/Decoder | Languages | Translation | Timestamps | License |
|---|---|---|---|---|---|---|---|
| canary-180m-flash | https://huggingface.co/nvidia/canary-180m-flash | 182M | FC 17L / Transformer 4L | 4 (EN/DE/FR/ES) | EN↔DE/FR/ES | yes (experim.) | CC-BY-4.0 |
| canary-1b-flash | https://huggingface.co/nvidia/canary-1b-flash | 883M | FC 32L / Transformer 4L | 4 | EN↔DE/FR/ES | word+seg | CC-BY-4.0 |
| **canary-1b-v2** | https://huggingface.co/nvidia/canary-1b-v2 | 978M | FC 32L / Transformer 8L | **25 EU** (it included) | EN↔24 | yes, also on AST | CC-BY-4.0 |
| canary-qwen-2.5b | https://huggingface.co/nvidia/canary-qwen-2.5b | 2.5B | SALM: FC + Qwen3-1.7B frozen + LoRA | EN only | no | no | CC-BY-4.0 |
| canary-1b (2024) | https://huggingface.co/nvidia/canary-1b | 1B | FC 24L / Transformer 24L | 4 | EN↔DE/FR/ES | no | **CC-BY-NC-4.0** ⚠️ |

- Flash/1b: prompt tokens `<target language>`, `<task>`, `<toggle timestamps>`, `<toggle PnC>`;
  v2: `source_lang`/`target_lang` (equal = ASR, different = AST); v2 tokenizer: unified SPE **16384**.
- canary-1b-flash and canary-qwen also have safetensors; 180m-flash and v2 `.nemo` only.
- Papers: Canary "Less is More" https://arxiv.org/abs/2406.19674 · Flash efficiency
  https://arxiv.org/abs/2503.05931 · Canary v2 + Parakeet v3 tech report
  https://arxiv.org/abs/2509.14128 · Granary dataset https://arxiv.org/abs/2505.13404 ·
  SALM https://arxiv.org/abs/2310.09424

---

## Implications for the Mynah runtime

1. **The config must drive everything**: mel 80 vs 128, left context 56 vs 70, vocab 1024/8192/13088/16384,
   layers 17/24/32/42 — no hardcoded constants (confirms the anti-#define choice).
2. **HF-native safetensors loader first, `.nemo` later**: Nemotron 3.5, parakeet-tdt-v3,
   rnnt/ctc 0.6b/1.1b already have safetensors + tokenizer.json → the converter can start
   from those without nemo_toolkit. For the `.nemo`-only models (110m, unified…) the tar
   extractor will be needed.
3. **Nemotron's RNNT decoder (pred 640, 2L) is the same design as the Parakeet RNNT/TDT**
   (same hidden/layers) → the decode loop is reusable; TDT adds the durations [0..4].
4. **parakeet-tdt_ctc-110m** is the natural candidate for CI tests (114M, CC-BY-4.0);
   note: `.nemo` only, so the tar extraction tooling is needed soon anyway.
5. **Licenses**: classic family CC-BY-4.0, Nemotron 3.5 OpenMDW-1.1 (great),
   the new EN streaming models (unified, eou, multitalker) NVIDIA Open Model License (verify
   the terms before redistributing converted weights); original canary-1b is NC → avoid.
