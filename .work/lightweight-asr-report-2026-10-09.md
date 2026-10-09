# Lightweight ASR: end-of-session report (2026-10-09)

One NVIDIA L40S 48 GB (Vast.ai, Japan), one day. Details, commands and raw
numbers in `.work/parakeet-eou-l4.md`, `.work/canary-180m-l4.md`,
`.work/lightweight-asr-ft.md`, `.work/eou-it-ft.md`; jobs in
`.work/lightweight-asr/jobs/`. Artifacts: private HF repo
`gabrione/mynah-asr-canary-it-experiments` (size-verified uploads).
All performance numbers are L40S; the L4 rerun is owed.

## Comparison

| | Canary 180M IT specialist (B4 40 h e7) | Canary 180M multilingual replay (B4 20 h e13 rp20) | EOU 120M stock | EOU 120M IT FT |
|---|---|---|---|---|
| params | 183.7 M (99.4 M trained) | 183.7 M (99.4 M trained) | 114.9 M counted (card: 120 M) | 114.9 M |
| architecture | FastConformer 17L + Transformer AED 4L | same | cache-aware FastConformer 17L [70,1] + RNNT (1L LSTM) | same, Italian SPE 1024 + <EOU>/<EOB> |
| serving | offline AED (Mynah CUDA server: batched REST, WS 400 model_not_streaming) | offline AED | TRUE streaming, CPU + CUDA servers, model EOU events | streaming by construction (config asserted unchanged) |
| training data | MLS-it 40 h | MLS-it 20 h + 8 h FLEURS en/de/es/fr replay | n/a | MLS-it 5 h |
| audio-hours seen | ~281 h | ~325 h | n/a | see below |
| IT WER / CER FLEURS | 42.06 / 12.75 | 44.17 / 13.44 | n/a (English only) | see below |
| IT WER / CER MLS | 29.42 / 7.04 | 30.27 / 7.17 | n/a | see below |
| original-language regression | EN 6.86 -> 54.53 | EN +1.1, DE +0.9, ES +2.7, FR +0.2 | n/a | EN not a gate (specialist) |
| EN quality (Mynah, same audio as Nemotron) | Canary base: FLEURS-en 0.097 mean WER (Nemotron 0.116) | | 498-clip bank 0.1545 vs Nemotron 0.1447; FLEURS-en 0.122 after level normalisation (0.301 raw: level sensitivity) | |
| EOU latency (audio timeline) | n/a | n/a | p50 ~1.7 s on read speech (FLEURS, normalised); 0 % premature | not reached |
| empty rate | 0 % | 0 % | 20 % on raw FLEURS (quiet clips), 0 % normalised | see below |
| peak training VRAM | 8.33 GB alloc (fits an L4) | 8.33 GB | n/a | 16.9-17.3 GB alloc (full encoder), 8.2 GB (frozen) |
| training wall | ~10 min train, 0.34 GPU-h incl. evals | 0.39 GPU-h incl. evals | n/a | ~25 min per 5 h e52 run |
| throughput | 2,865 audio-h per GPU-h | 2,706 | n/a | ~1,190 (padding + RNNT loss); ~2,040 frozen |
| cost | rate not set; at USD 0.3/h L4-class it is cents per run | same | n/a | same |
| serving (CUDA server, L40S) | 7 clips batched 2.7 s vs 27.9 s CPU (decoder-bound; no ladder yet) | | ~2x Nemotron's sessions under a common 320 ms bound (C1536 vs C768); idle VRAM 1.9 vs 6.9 GiB | |
| HF artifact | runs/B4-40h-e7 (final.nemo) | runs/B4-20h-e13-rp20 (final.nemo + last.ckpt) | upstream | runs/eou-it-* (metrics, logs, final.nemo, last.ckpt for the cold run) |
| resumable checkpoint | no (rerun = 10 min) | yes, last.ckpt step 3900 | n/a | yes for the cold run |

## Answers so far

1. EOU 120M is substantially cheaper to serve than Nemotron 0.6B in Mynah
   (about 2x the sessions at a common latency bound on the L40S, a quarter
   of the idle VRAM, CPU/CUDA parity incl. EOU events) at ~1 WER point behind
   on sane audio levels. Its weaknesses: level sensitivity (fixed by a causal
   AGC in the diagnostic) and a ~1.7 s endpoint decision on read speech,
   which is model behaviour, not serving.
2. Canary 180M is a strong small EN/DE/ES/FR tier in quality (better WER than
   Nemotron on FLEURS EN/FR in this bank) but offline: it is served honestly
   as batched AED in the CUDA server, never as streaming.
3. Canary 180M can be adapted to Italian on one GPU in minutes per run, and
   EXTENDED (20 % replay keeps EN/DE/ES/FR within ~1-3 points). Italian WER
   is still high (42 FLEURS / 29 MLS at 40 h): the acoustic side transfers
   (in-domain CER 7 %), the decoder still learns the language.
4. EOU 120M -> Italian did NOT work tonight: see the EOU section; the failure
   is isolated to the training recipe, reproducible, and archived.

## EOU 120M -> Italian: what happened

Four arms on MLS-it 5 h (cold/warm decoder+joint x trainable/frozen encoder),
lr 1e-3 on decoder/joint, the NVIDIA EOU recipe's padding (3-6 s, p 0.9) and
FastEmit 0.03: all end at WER ~100 with empty greedy output. Diagnostics
(`.work/eou-it-ft.md`): the objective converges while the RNNT becomes an
audio-independent text prior (beam-4 returns the same sentence for every
input, blank wins every frame); a real batch is sane; freezing the encoder
does not help; starting from the working stock model with a frozen encoder,
the loop removes text tokens first and <EOU> next. The recipe/optimisation
is the cause; the next session starts with a micro-overfit and a
one-variable-at-a-time ablation ladder (documented), not with a bigger run.
Training economics measured: ~1,190 audio-h per GPU-h with the full encoder
(16.9-17.3 GB peak), ~2,050 frozen (8.1 GB, fits an L4).

## Is EOU-120M a credible base for cheap per-language streaming specialists?

Not yet answered. As a SERVING base it is excellent (CUDA parity incl. EOU,
~2x Nemotron's sessions, 1.9 GiB idle). As a FINE-TUNING base, tonight's
recipe failed for a reason isolated to the training loop, not to the model;
the acoustic side is not the limit (Canary's encoder of the same family
transferred to Italian at 7 % in-domain CER). The answer needs the ablation
ladder (one evening of GPU at most, given ~0.1 GPU-h per 5 h run).

## Artifacts at shutdown

Private HF repo `gabrione/mynah-asr-canary-it-experiments`, revision
f6d4a3076a7884fef62465ecc10c9ba22aea0f95: 201 files, 8 `.nemo`, 4 `last.ckpt`
(Canary replay, three EOU-IT arms), recipes (`recipe/ft-kit`, `recipe/eou-kit`),
tokenizers, metrics/evals/logs per run, `results/survival-final.tgz`
(13.7 MB, also on the dev machine). Every upload verified by size. Code and
notes: branch `research/lightweight-asr` of mynah-asr. The HF token was
removed from the box after the last verified upload.

## Update, same evening: EOU 120M -> Italian WORKS with two stages

Stage 1 plain Italian ASR from the stock model and STOCK tokenizer (de-accented),
stage 2 text + <EOU>: MLS-it 40.4 WER / 12.5 CER, 0 empty, <EOU> on 98 % of the
clips (stage 1 alone 41.7 / 12.3). Inside Mynah: FLEURS-it stream 55.1 / 19.0
(out of domain), CPU/CUDA parity incl. EOU events, CUDA server 1.9 GiB, 0 %
premature EOU; single-clip endpoint p50 2.35 s (slower than the English stock),
A + 1 s + B EOU latency p50 0.30-0.33 s. FastEmit 0.03 ablation: better
turn-taking, +1.2 WER. Training cost: ~13.5 min (stage 1, 40 h) + ~7 min
(stage 2) on one L40S. So: YES, EOU-120M is a credible base for cheap per-language
streaming specialists; endpoint latency and accents are the open work. Details:
`.work/eou-it-ft.md`. Code: `finetune/eou/two_stage.py` (branch lw-finetune-tooling).
Final HF revision: d836773ad2863d352095898608c45e4bd1e5c950 (230 files, 14 .nemo,
9 checkpoints, survival-final2.tgz verified); HF token removed from the box.

## French replica (same night, no tuning)

MLS-fr 38.6 h (37 speakers): plain 51.0 / 26.9 -> stage 2 EOU 49.2 / 25.9,
EOU 97 %, ~21 GPU minutes. In Mynah on FLEURS-fr: 78.7 WER (73.4 after level
normalisation), EOU p50 2.15 s, 0 % premature, A + 1 s + B EOU in gap 70 %.
The method replicates; out-of-domain generalisation needs more speaker
diversity than 6 MLS shards give.
