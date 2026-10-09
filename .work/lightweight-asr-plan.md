# Lightweight ASR: Parakeet EOU / Canary / low-data language adaptation

Opened 2026-10-09 on `research/lightweight-asr` (branched from `asr-cuda-perf`,
PR #4, so the bf16 own-tc GEMM and the serving-loop defaults are available to
A/B). PLAN.md carries the TODOs (section S15); this file carries the method,
the order, the decisions and the links to the detailed notes.

## The question

Can the small models Mynah-ASR already supports give a much cheaper/faster tier
than Nemotron Speech Streaming 0.6B while keeping competitive transcription
quality, and are they cheap enough to adapt to new European languages with
small open datasets?

Four questions decide it, and every measurement below exists to answer one:

1. Is Parakeet Realtime EOU 120M substantially cheaper/faster than Nemotron 0.6B
   in Mynah, with acceptable English WER/CER and EOU behaviour?
2. Is Canary 180M Flash a compelling lightweight EN/DE/ES/FR model in Mynah, and
   what are its actual streaming capabilities and limits?
3. Can either model be adapted to Italian with ~5-40 h of open speech on one
   cheap GPU (L4 class)?
4. Should Mynah-ASR treat these models as a first-class "small/extensible" tier?

None of the answers is presupposed. "UNKNOWN" is a valid cell until measured.

## Notes

| file | content |
|---|---|
| [`.work/lightweight-asr-upstream.md`](lightweight-asr-upstream.md) | upstream facts verified against NVIDIA/NeMo primary sources |
| [`.work/parakeet-eou-l4.md`](parakeet-eou-l4.md) | EOU 120M: audit of current support, quality, EOU behaviour, CUDA/server performance |
| [`.work/canary-180m-l4.md`](canary-180m-l4.md) | Canary 180M Flash: audit, streaming verdict, EN/DE/ES/FR quality, performance |
| [`.work/lightweight-asr-ft.md`](lightweight-asr-ft.md) | NeMo FT recipes, Italian datasets and licences, the FT experiments and their economics |
| `.work/lightweight-asr/jobs/` | the box scripts, restartable, every number's command |

## Order (scope discipline)

1. PR #4 (yesterday's CUDA work) is open; this track does not extend it.
2. Planning and audit only (this commit).
3. EOU 120M correctness and quality.
4. EOU 120M CUDA/server performance.
5. Canary 180M correctness and quality.
6. Canary's actual streaming capabilities.
7. Canary server/performance characterisation.
8. Official NVIDIA FT recipes.
9. Cheap Italian FT smoke test.
10. 5 h -> 20 h -> 40 h only if the smoke test justifies it.

## Method rules for this track

- Same audio, same normaliser, same scorer for every model compared in one
  table. A number from another bank or normaliser is not put in the same row.
- Decoding settings fixed per model and recorded (greedy, chunk/preset, beam,
  endpointing). Nothing changes silently between models.
- A numerical change (precision, GEMM, quantisation) is A/B'd on THIS model with
  a quality gate; an optimisation is not enabled because it helped Nemotron.
- Open data only, licence recorded before download. No private audio, no
  credentials, on a rented box. The HF read token lives in `/root/.hf_token`
  (mode 600) only during model downloads and is deleted afterwards.
- Boxes can disappear: everything runs in tmux, every job is restartable,
  compact results come back to the dev machine after each phase.
- Short screens (2-3 min rungs), only the arms that discriminate.

## Hardware

Primary target: L4 24 GB (cheap, an interesting deployment class). The first
session of this track (2026-10-09) runs on an L40S 48 GB (Xeon Gold 6430,
2 NUMA nodes, ~30.7 CPU quota, 32 GB disk) because that is the box available;
quality, EOU behaviour and correctness do not depend on the card, while every
performance number is labelled with its GPU and an L4 rerun of the decisive
rungs is owed before a number is quoted as "L4".

## Decision matrix (filled from measurements only)

|  | Nemotron 0.6B | Parakeet EOU 120M | Canary 180M Flash |
|---|---|---|---|
| parameters | | | |
| licence | | | |
| languages | | | |
| true streaming? | | | |
| EOU? | | | |
| timestamps? | | | |
| translation? | | | |
| WER/CER EN | | | |
| WER/CER EU languages | | | |
| C1 latency | | | |
| streaming lag | | | |
| finalisation | | | |
| RTFx / throughput | | | |
| L4 C knee | | | |
| VRAM | | | |
| FT path | | | |
| Italian 5 h | | | |
| Italian 20 h | | | |
| Italian 40 h | | | |
| training GPU-hours | | | |
| estimated rental cost | | | |
| known limitations | | | |

## Log

- 2026-10-09: track opened; L40S provisioned from scratch on the box (repo
  clone, CUDA build, models downloaded from HF and converted there, FLEURS
  banks fetched there): `.work/lightweight-asr/jobs/prov.sh`.
