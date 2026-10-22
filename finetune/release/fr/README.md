---
# DRAFT model card -- release candidate RC-FR v1 (2026-10-22). Not published. Name TBD (no NVIDIA / Parakeet branding).
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/parakeet_realtime_eou_120m-v1
language: [fr]
library_name: nemo
pipeline_tag: automatic-speech-recognition
tags: [streaming, rnnt, end-of-utterance, fastconformer, french]
---

# mynah-eou-fr-120m (draft)

French streaming speech recognition with an integrated end-of-utterance token (`<EOU>`), fine-tuned from
NVIDIA Parakeet Realtime EOU 120M (`nvidia/parakeet_realtime_eou_120m-v1`). 120M parameters, cache-aware
FastConformer encoder ([70, 1] attention context, 80 ms frames), RNNT decoder, the stock 1024-piece SPE vocabulary
plus `<EOU>` / `<EOB>` (blank 1026). All weights fine-tuned; vocabulary unchanged.
Runs in NeMo and in Mynah (CPU f32, CUDA bf16; ~1.9 GB VRAM for the CUDA server incl. a 128-stream arena).

Licensed by NVIDIA Corporation under the NVIDIA Open Model License. See LICENSE and NOTICE.

## Results

### FINAL-FR (sealed held-out set, evaluated once; characterisation, not selection)
Protocol (frozen before the decode, git 6f00ac3):
- greedy RNNT, max 10 symbols / frame, no beam, no LM;
- decoder reset after `<EOU>`, the transcript = concatenation of segments;
- 2.0 s of -75 dB quiet appended as finalisation;
- text lowercased, accents stripped, oe / ae and typographic apostrophes mapped, [a-z'] only;
- endpoint = energy endpoint of the original clip.
"Mynah" = the same clips through the Mynah CUDA server (bf16, the serving default), same scorer.

WER / CER are the unweighted macro over the six banks; EOU metrics are pooled over clips.

| bank | clips | WER | CER | false endpoint | FE_any | EOU recall | endpoint latency p50 / p90 | empty | Mynah WER |
|---|---|---|---|---|---|---|---|---|---|
| Common Voice 17 fr | 400 | 27.42 | 10.45 | 4.0 % | 5.5 % | 95.2 % | 0.96 / 1.92 s | 1 | 27.42 |
| FLEURS fr | 113 | 28.57 | 11.26 | 6.2 % | 5.3 % | 92.9 % | 0.80 / 1.12 s | 0 | 28.76 |
| MLS fr | 400 | 26.90 | 11.08 | **25.2 %** | 25.2 % | 96.8 % | 0.72 / 1.36 s | 3 | 27.12 |
| Speech-MASSIVE fr-FR | 400 | 31.01 | 12.14 | 0.5 % | 2.8 % | 97.5 % | 0.96 / 2.00 s | 5 | 31.05 |
| mTEDx fr (held-out talks) | 400 | 31.52 | 15.45 | 4.8 % | 4.8 % | 98.0 % | 0.88 / 1.60 s | 1 | 31.78 |
| VoxPopuli fr | 400 | 21.13 | 9.36 | 3.8 % | 3.8 % | 97.2 % | 0.48 / 0.88 s | 0 | 21.25 |
| **all** | 2,113 | **27.76** | **11.62** | **7.57 %** | **8.23 %** | **96.73 %** | **0.80 / 1.60 s** | 10 | **27.90** |

- No success band was pre-registered for French; these numbers are a characterisation.
- false endpoint = an `<EOU>` emitted while >= 0.5 s of speech remains; FE_any = followed by >= 0.16 s of speech;
  recall = an `<EOU>` at >= speech end - 0.5 s; latency from the energy speech end.

### Development sets, same protocol (DEV-fr: MLS / FLEURS / CV / VP, 200 clips each)
- reset-WER 25.64, CER 10.31, false endpoint 7.25 %, recall 97.5 %, latency p50 / p90 0.72 / 1.20 s, 0 empty.
- Mynah CUDA server, realtime WebSocket feed, 16 streams: 25.68, FE 7.50 %, recall 97.38 %.
- FR-OOD1 (held-out mTEDx dev talks + Speech-MASSIVE train speakers, retention set): 29.53.

### Mynah runtime parity (DEV-fr 800 clips, the same PCM16 bytes)
- Mynah CPU f32 == Mynah CUDA f32 on 800 / 800 clips (text and `<EOU>` times).
- NeMo f32 vs Mynah f32: 752 / 800 identical; no text difference before the first `<EOU>`; the 43 clips that differ
  all follow a premature `<EOU>` and come from the runtime's reset semantics (below), +0.06 WER.
- CUDA bf16 (serving default) vs f32: metric-neutral (-0.01 WER).
- VRAM 1,941 MiB with the model loaded, flat up to 128 concurrent streams; realtime vs fast feed identical.

## Known limitations
- **Long-form / audiobook internal pauses can trigger a premature `<EOU>`**, especially on MLS-like read speech
  (false endpoints 25.2 % on FINAL MLS, 18 % on DEV MLS; every other bank <= 6.2 %). The endpoint is acoustic
  (pause-driven), not semantic; use a runtime commit / debounce policy for turn-taking on long reads. A training-side
  fix (pause-stretch augmentation) was tested and did not reduce it.
- Reset semantics: the Mynah runtime resets the encoder caches and the decoder after the chunk holding an `<EOU>`;
  the characterisation protocol above resets the decoder only. Transcripts differ only after a premature `<EOU>`
  (+0.06 DEV WER, +0.14 FINAL WER).
- Output is lowercase without accents, punctuation or casing (the stock vocabulary has no accented letters; scoring is
  accent-insensitive); numbers are not verbalised.
- Some MLS-fr training transcripts have deleted accented letters (data); Speech-MASSIVE and mTEDx are the hardest
  banks.

## Training
- Stage 1 (plain ASR, FR-G1E) on 216.3 h (sampling weights):
  - MLS-fr 40.0 h, all 142 speakers (0.2356);
  - Common Voice 17 fr 40.0 h (0.2020);
  - VoxPopuli fr, teacher-filtered (WER <= 50), 39.7 h (0.1347);
  - FLEURS fr train 7.7 h (0.1010);
  - YODAS-Granary fr 89.0 h, labelled with nvidia/nemotron-3.5-asr-streaming-0.6b transcripts where they agree
    with the Granary reference (<= 20 % WER) (0.3267).
- Stage 1 recipe: stock init, whole encoder lr 4e-4, decoder / joint 3e-4, AdamW, WSD (5 % warm-up, decay over the
  last 20 %), 11,881 steps, bs 16, max 30 s, seed 0; final checkpoint = last step; 65 min on one RTX 6000 Ada.
- Stage 2 (`<EOU>`) recipe (= the Italian stage 2 v4):
  - same mix; targets text + `<EOU>`; 50 % of samples padded with 1-3 s of quiet;
  - top-4 encoder blocks 3e-5, decoder / joint 2e-5, cosine, 1500 steps, max 20 s;
  - checkpoint = min WER with >= 80 % EOU and <= 10 % empty (step 250); 14 min.
- Checkpoint sha256 7814141d515ca347004a862435fc5029d553d885252c8ac5e5b23ce793d48105 (parent 1271deff).
- One seed per stage; the training runs were not re-executed (no bit-exact training reproducibility claim).

## Data and attribution
- MLS (CC BY 4.0, Pratap et al. 2020);
- Common Voice 17 (CC0; do not attempt to identify speakers);
- VoxPopuli (CC0, European Parliament source);
- FLEURS (CC BY 4.0, Conneau et al. 2022);
- YODAS-Granary (CC BY 3.0, YouTube CC-BY uploads; Li et al. 2023, NVIDIA Granary);
- pseudo-label agreement filter by nvidia/nemotron-3.5-asr-streaming-0.6b (OpenMDW-1.1);
- evaluation only: Multilingual TEDx (CC BY-NC-ND 4.0) and Speech-MASSIVE (CC BY-NC-SA 4.0): numbers only, data not
  redistributed.

## Usage
See `examples/` and `inference.json`.
