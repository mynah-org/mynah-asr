---
# DRAFT model card -- release candidate RC-IT v4 (2026-10-19, qualified 2026-10-22). Not published. Name TBD (no NVIDIA / Parakeet branding).
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/parakeet_realtime_eou_120m-v1
language: [it]
library_name: nemo
pipeline_tag: automatic-speech-recognition
tags: [streaming, rnnt, end-of-utterance, fastconformer, italian]
---

# mynah-eou-it-120m (draft)

Italian streaming speech recognition with an integrated end-of-utterance token (`<EOU>`), fine-tuned from
NVIDIA Parakeet Realtime EOU 120M (`nvidia/parakeet_realtime_eou_120m-v1`). 120M parameters, cache-aware
FastConformer encoder ([70, 1] attention context, 80 ms frames), RNNT decoder, the stock 1024-piece SPE vocabulary
plus `<EOU>` / `<EOB>` (blank 1026). All weights fine-tuned; vocabulary unchanged.
Runs in NeMo and in Mynah (CPU f32, CUDA bf16; ~1.9 GB VRAM for the CUDA server incl. a 128-stream arena).

Licensed by NVIDIA Corporation under the NVIDIA Open Model License. See LICENSE and NOTICE.

## Results

### FINAL-IT (sealed held-out set, evaluated once; characterisation, not selection)
Protocol (frozen before the decode, git 9f0bc8d):
- greedy RNNT, max 10 symbols / frame, no beam, no LM;
- decoder reset after `<EOU>` (Mynah semantics), the transcript = concatenation of segments;
- 2.0 s of -75 dB quiet appended as finalisation;
- text lowercased, accents stripped, [a-z'] only;
- endpoint = energy endpoint of the original clip.

WER / CER are the unweighted macro over the six banks; EOU metrics are pooled over clips.

| bank | clips | WER | CER | false endpoint | FE_any | EOU recall | endpoint latency p50 / p90 | empty |
|---|---|---|---|---|---|---|---|---|
| MLS-it dev | 400 | 23.44 | 5.90 | 2.5 % | 2.8 % | 97.5 % | 0.56 / 2.24 s | 0 |
| Common Voice 17 it validation | 400 | 19.44 | 5.51 | 2.5 % | 3.5 % | 97.2 % | 0.80 / 2.32 s | 0 |
| VoxPopuli it validation | 110 | 20.35 | 10.55 | 7.3 % | 7.3 % | 98.2 % | 0.32 / 1.04 s | 1 |
| FLEURS it dev | 116 | 12.56 | 3.02 | **16.4 %** | 16.4 % | **84.5 %** | 0.96 / 2.63 s | 0 |
| mTEDx it (held-out talks) | 400 | 21.45 | 9.79 | 4.2 % | 4.2 % | 95.5 % | 0.96 / 2.08 s | 8 |
| ITALIC (held-out speakers) | 400 | 25.68 | 6.91 | 0.0 % | 0.0 % | 97.2 % | 1.12 / 2.40 s | 0 |
| **all** | 1,826 | **20.49** | **6.95** | **3.50 %** | **3.78 %** | **96.17 %** | **0.88 / 2.24 s** | 9 |

- The pre-registered success band was macro WER < 20; **it was not reached (20.49)**.
- Clips cut at the energy endpoint + 50 ms (no trailing silence) give 20.52 / FE 3.40 % / p50 0.80 s **under this
  protocol** (2 s quiet finalisation). This is not a general claim of independence from trailing silence.
- false endpoint = an `<EOU>` emitted while >= 0.5 s of speech remains; FE_any = followed by >= 0.16 s of speech;
  recall = an `<EOU>` at >= speech end - 0.5 s; latency from the energy speech end.

### Development set (DEV730: MLS / FLEURS / CV test 200 each, VP test 130), same protocol

| model | WER | false endpoint | recall | latency p50 / p90 |
|---|---|---|---|---|
| this model | 22.54 | 8.36 % | 94.38 % | 0.72 / 1.92 s |
| previous Italian EOU model (EOU-v3) | 36.07 | 5.89 % | 94.66 % | 0.64 / 1.92 s |

Offline without the reset (NeMo default decode, flush 0.5 s): see the private archive. Beam search is not part
of the release configuration. The plain (no `<EOU>`) parent reaches 22.27 greedy / 20.53 beam-8 / 16.25 oracle-8
on DEV; those are separate decodes, not this model's release numbers.

### Mynah streaming (FLEURS-it test 200, CLI stream f32, accent-insensitive, pooled)
- WER 17.17, CER 7.19, 0 empty; model `<EOU>` on 196 / 200 clips.
- Speech end -> `<EOU>` p50 778 ms / p90 2794 ms; premature 3.3 %.
- A + 1 s + B composites: `<EOU>` in the gap 83.3 %, p50 426 ms.
- CUDA: batch identity, slot independence and CPU f32 == GPU gates PASS.

## Known limitations
- False endpoints at internal pauses, worst on FLEURS-style read Wikipedia sentences (16.4 % on FINAL-IT FLEURS,
  17 % on DEV FLEURS): the endpoint is acoustic (pause-driven), not semantic. A runtime debounce / commit policy is
  advisable for turn-taking.
- Output is lowercase without accents, punctuation or casing; numbers are not verbalised (digit clips were excluded
  from training).
- Spontaneous / parliamentary speech (VoxPopuli) and short read commands (ITALIC) remain the hardest banks.
- A precision-first research variant exists (DEV false endpoints 0.82 % but latency p50 1.84 s); it is not this
  release.

## Training
- Stage 1 (plain ASR) on the C250 mix, ~357 h (sampling weights):
  - MLS-it 40 h (0.1964);
  - Common Voice 17 it 40 h (0.1684);
  - VoxPopuli it, teacher-filtered, 32.6 h (0.1122);
  - FLEURS it train 6.8 h (0.0842);
  - YODAS-Granary it, two pools of 98.1 h and 130.2 h (0.1885 / 0.2503), labelled with
    nvidia/nemotron-3.5-asr-streaming-0.6b transcripts where they agree with the Granary reference (<= 15 % WER).
- Stage 1 recipe: whole encoder lr 2e-4, decoder / joint 3e-4, AdamW, one cosine over 36k steps, bs 16, max 30 s;
  ~5 h on one L40S.
- Stage 2 (`<EOU>`) recipe:
  - same mix; targets text + `<EOU>`; 50 % of samples padded with 1-3 s of quiet;
  - top-4 encoder blocks 3e-5, decoder / joint 2e-5, 1500 steps;
  - checkpoint = min WER with >= 80 % EOU and <= 10 % empty (step 250); ~0.5 h.
- Checkpoint sha256 0a4d1b103297e169af74b6673ff97864747bb2c54b70a7ec91253d9583c003ff.
- Mynah pack: model.safetensors 2aa4783d..., mynah.json ff6d7c5a..., tokens.json 70a5a46e...,
  mel_filters.safetensors 2dde1d9b...

## Data and attribution
- MLS (CC BY 4.0, Pratap et al. 2020);
- Common Voice 17 (CC0; do not attempt to identify speakers);
- VoxPopuli (CC0, European Parliament source);
- FLEURS (CC BY 4.0, Conneau et al. 2022);
- YODAS-Granary (CC BY 3.0, YouTube CC-BY uploads; Li et al. 2023, NVIDIA Granary);
- pseudo-labels by nvidia/nemotron-3.5-asr-streaming-0.6b (OpenMDW-1.1);
- evaluation only: Multilingual TEDx (CC BY-NC-ND 4.0, numbers only, data not redistributed) and ITALIC (CC BY 4.0).

## Release qualification (2026-10-22, DEV730, the same PCM16 bytes)
- Mynah CPU f32 == Mynah CUDA f32 on 730 / 730 clips (text and `<EOU>` times).
- NeMo f32 vs Mynah f32: no text difference before the first `<EOU>`; the clips that differ after it follow a
  premature `<EOU>` (runtime reset semantics, see `inference.json`); macro WER 22.56 vs 22.54.
- CUDA bf16 (serving default) vs f32: +0.04 macro WER.
- CUDA server: VRAM 1,966 MiB with the model loaded, flat at 16 / 64 / 128 realtime streams, 0 errors, realtime ==
  fast feed on every clip.

## Usage
See `examples/` (NeMo `transcribe_nemo.py`, validated against the frozen decode on 40 / 40 clips; Mynah CLI and
server in `mynah.md`) and `inference.json`. Mynah runtime tested: v0.9.1 (sources identical from 52e5271 to 6d485b5).
