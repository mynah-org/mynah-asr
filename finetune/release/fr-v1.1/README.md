---
# DRAFT model card -- RC-FR v1.1 CANDIDATE (2026-10-22). Not published. Name TBD (no NVIDIA / Parakeet branding).
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/parakeet_realtime_eou_120m-v1
language: [fr]
library_name: nemo
pipeline_tag: automatic-speech-recognition
tags: [streaming, rnnt, end-of-utterance, fastconformer, french]
---

# mynah-eou-fr-120m v1.1 (candidate, draft)

French streaming speech recognition with an integrated end-of-utterance token (`<EOU>`), fine-tuned from
NVIDIA Parakeet Realtime EOU 120M (`nvidia/parakeet_realtime_eou_120m-v1`). Same architecture, vocabulary, data and
stage-2 recipe as v1; the plain stage was trained twice as long (23,762 instead of 11,881 steps).

Licensed by NVIDIA Corporation under the NVIDIA Open Model License. See LICENSE and NOTICE.

**Status: candidate without a held-out FINAL number.** The sealed FINAL-FR set was spent on v1 (27.76 WER);
v1.1 is characterised on the development set, on FR-OOD1 (a retention set held out from training) and on an mTEDx
dev bank. Its numbers are not directly comparable with v1's FINAL number.

## Results (same protocol as v1: greedy, decoder reset after `<EOU>`, 2 s quiet, accent-insensitive)

| set | v1 | **v1.1** | delta |
|---|---|---|---|
| DEV-fr macro WER (MLS / FLEURS / CV / VP, 800 clips) | 25.64 | **22.08** | -3.56 |
| FR-OOD1 macro WER (mTEDx dev talks + Speech-MASSIVE train speakers, 800 clips) | 29.53 | **24.97** | -4.56 |
| DEV false endpoint / EOU recall / latency p50 | 7.25 % / 97.5 % / 0.72 s | 7.75 % / 97.12 % / 0.72 s | |
| OOD1 false endpoint / EOU recall | | 8.5 % / 96.75 % | |

Per DEV source (v1.1): MLS 20.11 (false endpoint 18.0 %), FLEURS 26.22, CV 23.94, VP 18.06. Plain stage (no
`<EOU>`, 0.5 s flush): DEV 22.06 vs 25.54, OOD1 25.66 vs 29.66, mTEDx dev 23.24 vs 27.15.

## Mynah runtime qualification (2026-10-22, DEV-fr 800 clips, the same PCM16 bytes)
- Mynah CPU f32 == Mynah CUDA f32 on 800 / 800 clips (text and `<EOU>` times).
- NeMo f32 vs Mynah f32: 768 / 800 identical texts; no text difference before the first `<EOU>` (the 9 PRE_EOU
  clips differ only in the placement of the end-of-stream `<EOU>`); the 105 POST_RESET clips follow an earlier
  `<EOU>` (runtime encoder + decoder reset). Runtime macro WER 22.20 vs 22.08 (v1: +0.06), FE 7.75 %, recall 97.0 %,
  p50 / p90 0.72 / 1.20 s.
- CUDA bf16 (serving default): 22.24; realtime WebSocket at 16 / 64 / 128 streams identical to the fast feed, 0
  errors, delta lag p99 47-54 ms, VRAM 1,966 MiB with the model loaded (flat at 64 / 128 streams).
- CUDA engine gate (batch identity, repeat, slot independence): all OK.

## Known limitations
- As v1: **long-form / audiobook internal pauses can trigger a premature `<EOU>`**, especially on MLS-like read
  speech (18 % false endpoints on DEV MLS); the endpoint is acoustic, not semantic.
- No FINAL (held-out) number; one training seed per stage.
- As v1: lowercase output without accents / punctuation; numbers not verbalised; reset semantics of the runtime
  (encoder + decoder) vs the characterisation protocol (decoder only).

## Training
- Stage 1 (FR-H2): the v1 stage-1 recipe and mix (216.3 h; MLS-fr 0.2356, CV-fr 0.2020, VP-fr t50 0.1347,
  FLEURS-fr 0.1010, Granary-fr agree <= 20 0.3267) at 23,762 steps; WSD is defined fractionally (5 % warm-up, decay
  over the last 20 %), so it stretched with the horizon. 7,774 s on one RTX 6000 Ada. Pre-registered (180a05d) as the
  single remaining WER experiment: class LEVER.
- Stage 2: the v1 stage-2 recipe unchanged on FR-H2; checkpoint = min in-run WER with >= 80 % EOU and <= 10 % empty
  (step 1250); 816 s. All seven pre-registered stage-2 gates passed.
- Checkpoint sha256 3130534d577e340df129fea9074a63a070aaa05292bcf3aadd96b4aa39fd969c (parent d2062aca).

## Data and attribution
As v1 (see NOTICE): MLS (CC BY 4.0), Common Voice 17 (CC0), VoxPopuli (CC0), FLEURS (CC BY 4.0), YODAS-Granary
(CC BY 3.0), Nemotron agreement filter (OpenMDW-1.1); evaluation-only data (mTEDx, Speech-MASSIVE) not redistributed.

## Usage
See `examples/` and `inference.json`.
