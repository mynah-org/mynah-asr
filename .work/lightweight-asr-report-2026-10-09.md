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
